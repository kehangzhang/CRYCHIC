"""Run the CellChat 2.1.2 condition-aware comparison protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.cellchat.run_by_sample import (
    SINGLE_THREAD_ENVIRONMENT,
    _ProcessRegistry,
    _run_registered_process,
)
from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)

METHOD_VERSION = "2.1.2"
METHOD_COMMIT = "a0d3b2d231d46c8787177fffeac908270c253747"
RESOURCE_ID = "ConnectomeDB2020_Hou_2020_human"
RESOURCE_SHA256 = (
    "e781363288a26c15e03246500111bfecb818eef997f5ebe1b936aaa465151c3a"
)
DATABASE_SHA256 = (
    "f62f1fe041462c73d0449ca51b8a4a1d8038ec77905734974b4eaf169974a650"
)
RESOURCE_ROWS = 2293
SCHEMA_VERSION = "crychic-cellchat-condition-aware-paper-v1"
RANKING_FILENAME = "condition_cell_pair_rankings.tsv"
SELECTED_FILENAME = "selected_differential_interactions.tsv.gz"
RECEPTOR_SENSITIVITY_RANKING_FILENAME = (
    "receptor_sensitivity_condition_cell_pair_rankings.tsv"
)
RECEPTOR_SENSITIVITY_SELECTED_FILENAME = (
    "receptor_sensitivity_selected_interactions.tsv.gz"
)
RANKING_COLUMNS = (
    "dataset",
    "method",
    "method_version",
    "resource",
    "ranking_semantics",
    "condition",
    "sender",
    "receiver",
    "ranked_strength",
    "status",
    "reason_code",
)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return cast(dict[str, Any], value)


def _input_manifest_sha256(payload: Mapping[str, Any], filename: str) -> str:
    output = payload.get("output")
    if isinstance(output, Mapping) and output.get("filename") == filename:
        digest = output.get("sha256")
        if isinstance(digest, str):
            return digest
    if output == filename:
        digest = payload.get("output_sha256")
        if isinstance(digest, str):
            return digest
    raise ValueError("input manifest does not bind the requested h5ad checksum")


def _validate_resource(resource_path: Path, manifest_path: Path) -> dict[str, Any]:
    if sha256_file(resource_path) != RESOURCE_SHA256:
        raise ValueError("ConnectomeDB2020 payload checksum does not match the pin")
    manifest = _read_json_object(manifest_path, label="resource manifest")
    if manifest.get("resource_id") != RESOURCE_ID:
        raise ValueError("ConnectomeDB2020 resource_id does not match the pin")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping) or payload.get("sha256") != RESOURCE_SHA256:
        raise ValueError("ConnectomeDB2020 manifest payload checksum is inconsistent")
    table = pd.read_csv(resource_path, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "harmonized_interaction_id",
        "ligand",
        "receptor",
        "cellchat_source_interaction_id",
        "cellchat_covered",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"ConnectomeDB2020 columns are missing: {sorted(missing)}")
    covered = table["cellchat_covered"].str.lower().eq("true")
    if len(table) != RESOURCE_ROWS or not covered.all():
        raise ValueError("CellChat must cover all 2293 ConnectomeDB2020 pairs")
    if table["cellchat_source_interaction_id"].eq("").any():
        raise ValueError("ConnectomeDB2020 contains an empty CellChat identifier")
    if table.duplicated(["ligand", "receptor"]).any():
        raise ValueError("ConnectomeDB2020 contains duplicate directed pairs")
    if table["cellchat_source_interaction_id"].duplicated().any():
        raise ValueError("ConnectomeDB2020 contains duplicate CellChat identifiers")
    return manifest


def _validate_environment_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json_object(path, label="CellChat environment manifest")
    cellchat = manifest.get("cellchat")
    if not isinstance(cellchat, Mapping):
        raise ValueError("CellChat environment manifest lacks cellchat provenance")
    if cellchat.get("version") != METHOD_VERSION:
        raise ValueError("CellChat environment manifest does not pin version 2.1.2")
    if cellchat.get("git_commit") != METHOD_COMMIT:
        raise ValueError("CellChat environment manifest does not pin the paper commit")
    protocol = manifest.get("paper_protocol")
    if not isinstance(protocol, Mapping) or protocol.get(
        "supplementary_section"
    ) != "S4":
        raise ValueError("CellChat environment manifest does not bind paper section S4")
    return manifest


def _r_environment(rscript: Path) -> dict[str, Any]:
    expression = (
        "packages <- c('CellChat','presto','reticulate','Matrix','future',"
        "'future.apply','parallelly'); versions <- vapply(packages, "
        "function(package) as.character(packageVersion(package)), character(1)); "
        "cat(paste(c(as.character(getRversion()), versions, "
        "paste(RNGkind(), collapse='|')), collapse='\\t'))"
    )
    environment = os.environ.copy()
    environment.update(SINGLE_THREAD_ENVIRONMENT)
    environment["R_FUTURE_PLAN"] = "sequential"
    completed = subprocess.run(
        [str(rscript), "--vanilla", "-e", expression],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    fields = completed.stdout.strip().split("\t")
    if len(fields) != 9:
        raise RuntimeError("unexpected CellChat environment query output")
    r_version, *package_versions, rng_kind = fields
    packages = dict(
        zip(
            (
                "CellChat",
                "presto",
                "reticulate",
                "Matrix",
                "future",
                "future.apply",
                "parallelly",
            ),
            package_versions,
            strict=True,
        )
    )
    if packages["CellChat"] != METHOD_VERSION:
        raise RuntimeError("the R environment does not contain CellChat 2.1.2")
    return {
        "r": r_version,
        "packages": packages,
        "rng_kind": rng_kind.split("|"),
    }


def _prepare_metadata(
    input_h5ad: Path,
    *,
    cell_type_key: str,
    condition_key: str,
    target: str,
    reference: str,
    min_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = ad.read_h5ad(input_h5ad, backed="r")
    try:
        required = {cell_type_key, condition_key}
        missing = required.difference(source.obs.columns)
        if missing:
            raise ValueError(f"prepared h5ad metadata is missing: {sorted(missing)}")
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("prepared h5ad cell and gene identifiers must be unique")
        values = source.obs.loc[:, [cell_type_key, condition_key]].copy()
        if values.isna().any().any():
            raise ValueError("CellChat benchmark metadata contains null values")
        values = values.astype(str)
        observed = set(values[condition_key])
        if observed != {target, reference}:
            raise ValueError(
                f"observed conditions {sorted(observed)} do not match the contrast"
            )
        metadata = pd.DataFrame(
            {
                "cell": source.obs_names.astype(str),
                "cell_type": values[cell_type_key].to_numpy(),
                "condition": values[condition_key].to_numpy(),
            }
        )
        support = (
            metadata.groupby(["condition", "cell_type"], observed=True, sort=True)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        cell_types = sorted(metadata["cell_type"].unique())
        axis = pd.MultiIndex.from_product(
            ([target, reference], cell_types), names=["condition", "cell_type"]
        ).to_frame(index=False)
        support = axis.merge(
            support, on=["condition", "cell_type"], how="left", validate="one_to_one"
        )
        support["n_cells"] = support["n_cells"].fillna(0).astype(int)
        support["condition_present"] = support["n_cells"].gt(0)
        # CellChat 2.1.2 filterCommunication excludes groups with <= min.cells.
        support["native_network_supported"] = support["n_cells"].gt(min_cells)
        comparable = (
            support.groupby("cell_type", observed=True)["condition_present"]
            .all()
            .rename("de_comparable")
        )
        support = support.merge(comparable, on="cell_type", validate="many_to_one")
        support["reason_code"] = ""
        support.loc[
            ~support["condition_present"], "reason_code"
        ] = "cell_type_absent_in_condition"
        structural_zero = (
            support["condition_present"] & ~support["native_network_supported"]
        )
        support.loc[
            structural_zero, "reason_code"
        ] = "native_structural_zero_low_cell_support"
        de_comparable = sorted(
            comparable.index[comparable].astype(str).tolist()
        )
        if not de_comparable:
            raise ValueError("no cell type is present in both conditions")
        audit = {
            "shape": [int(source.n_obs), int(source.n_vars)],
            "x_backend": type(source.X).__name__,
            "source_cell_types": cell_types,
            "condition_cell_types": {
                condition: sorted(
                    support.loc[
                        support["condition"].eq(condition)
                        & support["condition_present"],
                        "cell_type",
                    ].astype(str)
                )
                for condition in (target, reference)
            },
            "de_comparable_cell_types": de_comparable,
            "absent_cell_types_by_condition": {
                condition: sorted(
                    support.loc[
                        support["condition"].eq(condition)
                        & ~support["condition_present"],
                        "cell_type",
                    ].astype(str)
                )
                for condition in (target, reference)
            },
            "native_structural_zero_cell_types_by_condition": {
                condition: sorted(
                    support.loc[
                        support["condition"].eq(condition)
                        & support["condition_present"]
                        & ~support["native_network_supported"],
                        "cell_type",
                    ].astype(str)
                )
                for condition in (target, reference)
            },
            "source_cells": len(metadata),
            "analysis_cells": len(metadata),
            "cells_by_condition": {
                str(key): int(value)
                for key, value in (
                    metadata.groupby("condition", observed=True).size().items()
                )
            },
        }
        return metadata, support, audit
    finally:
        source.file.close()


def preflight(
    input_h5ad: Path,
    input_manifest: Path,
    resource_path: Path,
    resource_manifest: Path,
    database_rds: Path,
    environment_manifest: Path,
    rscript: Path,
    *,
    cell_type_key: str,
    condition_key: str,
    target: str,
    reference: str,
    min_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in (
        input_h5ad,
        input_manifest,
        resource_path,
        resource_manifest,
        database_rds,
        environment_manifest,
        rscript,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    input_payload = _read_json_object(input_manifest, label="input manifest")
    expected_input_sha = _input_manifest_sha256(input_payload, input_h5ad.name)
    actual_input_sha = sha256_file(input_h5ad)
    if actual_input_sha != expected_input_sha:
        raise ValueError("prepared h5ad checksum does not match its manifest")
    actual_database_sha = sha256_file(database_rds)
    if actual_database_sha != DATABASE_SHA256:
        raise ValueError("CellChat ConnectomeDB2020 database checksum mismatch")
    resource_payload = _validate_resource(resource_path, resource_manifest)
    environment_payload = _validate_environment_manifest(environment_manifest)
    metadata, support, input_audit = _prepare_metadata(
        input_h5ad,
        cell_type_key=cell_type_key,
        condition_key=condition_key,
        target=target,
        reference=reference,
        min_cells=min_cells,
    )
    return metadata, support, {
        "input": {
            "filename": input_h5ad.name,
            "sha256": actual_input_sha,
            "manifest_filename": input_manifest.name,
            "manifest_sha256": sha256_file(input_manifest),
            **input_audit,
        },
        "resource": {
            "resource_id": resource_payload["resource_id"],
            "filename": resource_path.name,
            "sha256": RESOURCE_SHA256,
            "rows": RESOURCE_ROWS,
            "manifest_filename": resource_manifest.name,
            "manifest_sha256": sha256_file(resource_manifest),
            "cellchat_database_filename": database_rds.name,
            "cellchat_database_sha256": actual_database_sha,
        },
        "environment": {
            **_r_environment(rscript),
            "manifest_filename": environment_manifest.name,
            "manifest_sha256": sha256_file(environment_manifest),
            "source": environment_payload,
        },
    }


def _validate_rankings(
    path: Path,
    *,
    dataset_id: str,
    target: str,
    reference: str,
    cell_types: Sequence[str],
) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype={"reason_code": str})
    if tuple(table.columns) != RANKING_COLUMNS:
        raise ValueError("CellChat condition-aware ranking schema mismatch")
    expected_pairs = len(cell_types) * (len(cell_types) + 1) // 2
    if len(table) != 2 * expected_pairs:
        raise ValueError("CellChat ranking does not materialize the full pair axis")
    if set(table["dataset"]) != {dataset_id}:
        raise ValueError("CellChat ranking dataset identifier mismatch")
    if set(table["condition"]) != {target, reference}:
        raise ValueError("CellChat ranking condition axis mismatch")
    if set(table["method"]) != {"cellchat_condition_aware"}:
        raise ValueError("CellChat ranking method identifier mismatch")
    if set(table["method_version"]) != {METHOD_VERSION}:
        raise ValueError("CellChat ranking method version mismatch")
    if set(table["resource"]) != {RESOURCE_ID}:
        raise ValueError("CellChat ranking resource identifier mismatch")
    if table["ranking_semantics"].isna().any() or table[
        "ranking_semantics"
    ].astype(str).eq("").any():
        raise ValueError("CellChat ranking semantics must be explicit")
    keys = ["condition", "sender", "receiver"]
    if table.duplicated(keys).any():
        raise ValueError("CellChat ranking contains duplicate condition/pair rows")
    if (table["sender"].astype(str) > table["receiver"].astype(str)).any():
        raise ValueError(
            "CellChat ranking cell pairs are not canonical unordered pairs"
        )
    ordered_types = sorted(cell_types)
    expected_axis = {
        (condition, ordered_types[left], ordered_types[right])
        for condition in (target, reference)
        for left in range(len(ordered_types))
        for right in range(left, len(ordered_types))
    }
    observed_axis = set(table.loc[:, keys].itertuples(index=False, name=None))
    if observed_axis != expected_axis:
        raise ValueError("CellChat ranking pair labels do not match the input axis")
    if not set(table["status"]).issubset({"observed", "not_estimable"}):
        raise ValueError("CellChat ranking contains an invalid status")
    values = pd.to_numeric(table["ranked_strength"], errors="coerce")
    observed = table["status"].eq("observed")
    if values.loc[observed].isna().any() or (values.loc[observed] < 0).any():
        raise ValueError("observed CellChat rankings need non-negative cardinality")
    if not values.loc[observed].mod(1).eq(0).all():
        raise ValueError("CellChat cardinality ranking must be integer-valued")
    if values.loc[~observed].notna().any():
        raise ValueError("not-estimable CellChat rankings must not contain a score")
    if table.loc[~observed, "reason_code"].fillna("").astype(str).eq("").any():
        raise ValueError("not-estimable CellChat rankings need a reason code")
    return table


def _validate_selected_and_rankings(
    selected_path: Path,
    rankings: pd.DataFrame,
    *,
    resource_path: Path,
    target: str,
    reference: str,
    support: pd.DataFrame,
    sensitivity: bool,
) -> pd.DataFrame:
    selected = pd.read_csv(
        selected_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    required = {
        "condition",
        "source",
        "target",
        "harmonized_interaction_id",
        "interaction_name",
        "ligand",
        "receptor",
        "datasets",
    }
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(
            f"CellChat selected interaction columns are missing: {sorted(missing)}"
        )
    if not set(selected["condition"]).issubset({target, reference}):
        raise ValueError("CellChat selected interaction condition mismatch")
    if not selected["condition"].eq(selected["datasets"]).all():
        raise ValueError("CellChat selected interaction dataset labels mismatch")
    source_cell_types = set(support["cell_type"].astype(str))
    if not set(selected["source"]).issubset(source_cell_types) or not set(
        selected["target"]
    ).issubset(source_cell_types):
        raise ValueError("CellChat selected a cell type outside the input axis")
    support_index = support.set_index(["condition", "cell_type"])
    if len(selected) > 0:
        sender_keys = pd.MultiIndex.from_frame(selected[["condition", "source"]])
        receiver_keys = pd.MultiIndex.from_frame(
            selected[["condition", "target"]]
        )
        sender_present = support_index.loc[
            sender_keys, "condition_present"
        ].to_numpy(dtype=bool)
        receiver_present = support_index.loc[
            receiver_keys, "condition_present"
        ].to_numpy(dtype=bool)
        if not sender_present.all() or not receiver_present.all():
            raise ValueError("CellChat selected an absent condition/cell-type row")
        sender_network = support_index.loc[
            sender_keys, "native_network_supported"
        ].to_numpy(dtype=bool)
        receiver_network = support_index.loc[
            receiver_keys, "native_network_supported"
        ].to_numpy(dtype=bool)
        if not sender_network.all() or not receiver_network.all():
            raise ValueError("CellChat selected a native structural-zero row")
        sender_de = support_index.loc[
            sender_keys, "de_comparable"
        ].to_numpy(dtype=bool)
        receiver_de = support_index.loc[
            receiver_keys, "de_comparable"
        ].to_numpy(dtype=bool)
        if not sender_de.all() or (sensitivity and not receiver_de.all()):
            raise ValueError("CellChat selected a row without required DE support")

    ranking_sender_keys = pd.MultiIndex.from_frame(
        rankings[["condition", "sender"]]
    )
    ranking_receiver_keys = pd.MultiIndex.from_frame(
        rankings[["condition", "receiver"]]
    )
    ranking_sender = support_index.loc[ranking_sender_keys]
    ranking_receiver = support_index.loc[ranking_receiver_keys]
    sender_present = ranking_sender["condition_present"].to_numpy(dtype=bool)
    receiver_present = ranking_receiver["condition_present"].to_numpy(dtype=bool)
    sender_de = ranking_sender["de_comparable"].to_numpy(dtype=bool)
    receiver_de = ranking_receiver["de_comparable"].to_numpy(dtype=bool)
    if sensitivity:
        expected_observed = (
            sender_present & receiver_present & sender_de & receiver_de
        )
    else:
        expected_observed = (
            sender_present & receiver_present & (sender_de | receiver_de)
        )
    expected_status = pd.Series(
        np.where(expected_observed, "observed", "not_estimable"),
        index=rankings.index,
    )
    if not rankings["status"].eq(expected_status).all():
        raise ValueError("CellChat ranking condition-specific status mismatch")
    sender_network = ranking_sender["native_network_supported"].to_numpy(
        dtype=bool
    )
    receiver_network = ranking_receiver["native_network_supported"].to_numpy(
        dtype=bool
    )
    structural_zero = expected_observed & (~sender_network | ~receiver_network)
    strengths = pd.to_numeric(rankings["ranked_strength"], errors="coerce")
    if not strengths.loc[structural_zero].eq(0).all():
        raise ValueError("CellChat native structural-zero rankings must be zero")
    reasons = rankings["reason_code"].fillna("").astype(str)
    absent = ~expected_observed & (~sender_present | ~receiver_present)
    if sensitivity:
        expected_reasons = np.where(
            structural_zero,
            "native_structural_zero_low_cell_support",
            np.where(
                absent,
                "cell_type_absent_in_condition",
                np.where(
                    ~expected_observed,
                    "ligand_or_receptor_de_not_comparable",
                    "",
                ),
            ),
        )
    else:
        partial = expected_observed & (sender_de ^ receiver_de) & ~structural_zero
        expected_reasons = np.where(
            structural_zero,
            "native_structural_zero_low_cell_support",
            np.where(
                partial,
                "partial_directional_de_support",
                np.where(
                    absent,
                    "cell_type_absent_in_condition",
                    np.where(
                        ~expected_observed,
                        "no_de_comparable_ligand_direction",
                        "",
                    ),
                ),
            ),
        )
    if not reasons.eq(pd.Series(expected_reasons, index=rankings.index)).all():
        raise ValueError("CellChat ranking reason-code semantics mismatch")
    keys = ["condition", "source", "target", "harmonized_interaction_id"]
    if selected.duplicated(keys).any():
        raise ValueError("CellChat selected duplicate directed LR rows")

    resource = pd.read_csv(
        resource_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if resource["harmonized_interaction_id"].duplicated().any():
        raise ValueError("ConnectomeDB2020 harmonized identifiers are duplicated")
    resource = resource.set_index("harmonized_interaction_id")
    if not set(selected["harmonized_interaction_id"]).issubset(resource.index):
        raise ValueError("CellChat selected an interaction outside the resource")
    if len(selected) > 0:
        expected = resource.loc[
            selected["harmonized_interaction_id"],
            ["cellchat_source_interaction_id", "ligand", "receptor"],
        ].reset_index(drop=True)
        observed = selected.loc[
            :, ["interaction_name", "ligand", "receptor"]
        ].reset_index(drop=True)
        expected.columns = observed.columns
        if not observed.equals(expected):
            raise ValueError("CellChat selected interaction resource mapping mismatch")

    collapsed = selected.assign(
        sender=selected[["source", "target"]].min(axis=1),
        receiver=selected[["source", "target"]].max(axis=1),
    )
    counts = (
        collapsed.groupby(
            ["condition", "sender", "receiver"], observed=True, sort=False
        )
        .size()
        .rename("expected_strength")
    )
    observed_rankings = rankings.loc[rankings["status"].eq("observed")].copy()
    expected_strength = (
        observed_rankings.set_index(["condition", "sender", "receiver"])
        .join(counts, how="left")["expected_strength"]
        .fillna(0)
        .to_numpy()
    )
    actual_strength = pd.to_numeric(
        observed_rankings["ranked_strength"], errors="raise"
    ).to_numpy()
    if not (actual_strength == expected_strength).all():
        raise ValueError("CellChat ranking cardinality disagrees with selected rows")
    return selected


def _external_failure(
    completed: subprocess.CompletedProcess[str] | None,
    error: BaseException,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    if completed is not None:
        payload.update(
            {
                "returncode": completed.returncode,
                "stderr_log": "stderr.log",
                "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-40:]),
            }
        )
    return payload


def run(
    input_h5ad: Path,
    output_dir: Path,
    *,
    input_manifest: Path,
    resource_path: Path,
    resource_manifest: Path,
    database_rds: Path,
    environment_manifest: Path,
    rscript: Path,
    python_executable: Path,
    dataset_id: str,
    cell_type_key: str,
    condition_key: str,
    target: str,
    reference: str,
    threads: int,
    min_cells: int,
    emit_receptor_sensitivity: bool,
    timeout_seconds: float | None,
    preflight_only: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if target == reference:
        raise ValueError("target and reference must differ")
    for name, value in (("threads", threads), ("min_cells", min_cells)):
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when provided")
    metadata, support, audit = preflight(
        input_h5ad,
        input_manifest,
        resource_path,
        resource_manifest,
        database_rds,
        environment_manifest,
        rscript,
        cell_type_key=cell_type_key,
        condition_key=condition_key,
        target=target,
        reference=reference,
        min_cells=min_cells,
    )
    if not python_executable.is_file():
        raise FileNotFoundError(python_executable)
    output = prepare_output(output_dir, overwrite=overwrite)
    started = time.perf_counter()
    metadata_path = output / "cellchat_condition_metadata.tsv"
    support_path = output / "cellchat_condition_support.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=False)
    support.to_csv(support_path, sep="\t", index=False)
    protocol = {
        "paper_section": "S4",
        "contrast_order": [target, reference],
        "condition_inference": (
            "independent CellChat objects; computeCommunProb(type='triMean') "
            "with CellChat 2.1.2 defaults"
        ),
        "different_composition_alignment": (
            "per-condition inference on every cell type present; native "
            "filterCommunication(min.cells); liftCellChat(group.new=union of "
            "condition cell types) before mergeCellChat"
        ),
        "condition_overexpression": (
            "identifyOverExpressedGenes(min.cells=min_cells) with all other "
            "CellChat 2.1.2 defaults; identifyOverExpressedInteractions()"
        ),
        "condition_overexpression_defaults": {
            "thresh_pc": 0,
            "thresh_fc": 0,
            "thresh_p": 0.05,
            "only_positive": True,
        },
        "merge": "mergeCellChat(reference,target)",
        "de_unit": "cell_type",
        "de_group_dataset": "datasets",
        "de_positive_dataset": target,
        "de_only_positive": False,
        "de_group_combined": False,
        "de_p_threshold": 0.05,
        "de_logfc_threshold": 0.05,
        "de_expression_fraction_threshold": 0.1,
        "mapping": "netMappingDEG(variable.all=TRUE)",
        "primary_selection_label": "official_vignette_ligand_only",
        "primary_selection": {
            target: "datasets=target; ligand.logFC>=0.05; receptor.logFC=NULL",
            reference: (
                "datasets=reference; ligand.logFC<=-0.05; receptor.logFC=NULL"
            ),
        },
        "receptor_sensitivity_emitted": emit_receptor_sensitivity,
        "receptor_sensitivity_label": (
            "S4_literal_ligand_receptor_concordant_sensitivity"
        ),
        "ranking": (
            "cardinality of selected directed LR rows after unordered cell-pair "
            "collapse; full condition-by-cell-pair axis"
        ),
        "min_cells": min_cells,
        "future_workers": threads,
        "configured_blas_openmp_threads_per_r_session": 1,
        "future_globals_guard_bytes": 8 * 1024**3,
        "future_globals_guard_semantics": (
            "per-future serialized-globals guard, not an RSS limit"
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "preflight_complete" if preflight_only else "running",
        "dataset_id": dataset_id,
        "method": {
            "id": "cellchat_condition_aware",
            "name": "CellChat",
            "version": METHOD_VERSION,
            "git_commit": METHOD_COMMIT,
        },
        "protocol": protocol,
        "preflight": audit,
        "metadata": {
            "filename": metadata_path.name,
            "sha256": sha256_file(metadata_path),
            "rows": len(metadata),
            "support_filename": support_path.name,
            "support_sha256": sha256_file(support_path),
            "support_rows": len(support),
        },
        "statistical_scope": {
            "unit": "cell within cell type",
            "sample_aware_between_condition_inference": False,
            "formal_p_or_q_emitted": False,
            "reason_code": "cellchat_vignette_cell_level_de_not_sample_aware",
        },
        "code": {
            **git_metadata(Path(__file__).resolve().parents[3]),
            "python_runner_sha256": sha256_file(Path(__file__)),
            "r_runner_sha256": sha256_file(
                Path(__file__).with_name("run_condition_aware.R")
            ),
        },
        "failure": None,
    }
    manifest_path = output / "run_manifest.json"
    if preflight_only:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(manifest_path, manifest)
        return manifest

    command = [
        str(rscript),
        "--vanilla",
        str(Path(__file__).with_name("run_condition_aware.R")),
        str(input_h5ad),
        str(metadata_path),
        str(support_path),
        str(resource_path),
        str(database_rds),
        str(output),
        dataset_id,
        target,
        reference,
        str(threads),
        str(min_cells),
        str(emit_receptor_sensitivity).upper(),
    ]
    manifest["command"] = command
    write_json(manifest_path, manifest)
    process_registry = _ProcessRegistry()
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = _run_registered_process(
            command,
            process_registry=process_registry,
            timeout_seconds=timeout_seconds,
            environment_overrides={
                "RETICULATE_PYTHON": str(python_executable),
                "R_FUTURE_PLAN": "sequential",
            },
        )
        (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"CellChat condition-aware R runner exited {completed.returncode}"
            )
        input_audit = cast(dict[str, Any], audit["input"])
        cell_types = cast(list[str], input_audit["source_cell_types"])
        primary_rankings = _validate_rankings(
            output / RANKING_FILENAME,
            dataset_id=dataset_id,
            target=target,
            reference=reference,
            cell_types=cell_types,
        )
        primary_selected = _validate_selected_and_rankings(
            output / SELECTED_FILENAME,
            primary_rankings,
            resource_path=resource_path,
            target=target,
            reference=reference,
            support=support,
            sensitivity=False,
        )
        result_counts: dict[str, int] = {
            "primary_selected_interactions": len(primary_selected),
            "primary_ranking_rows": len(primary_rankings),
        }
        if emit_receptor_sensitivity:
            receptor_rankings = _validate_rankings(
                output / RECEPTOR_SENSITIVITY_RANKING_FILENAME,
                dataset_id=dataset_id,
                target=target,
                reference=reference,
                cell_types=cell_types,
            )
            receptor_selected = _validate_selected_and_rankings(
                output / RECEPTOR_SENSITIVITY_SELECTED_FILENAME,
                receptor_rankings,
                resource_path=resource_path,
                target=target,
                reference=reference,
                support=support,
                sensitivity=True,
            )
            result_counts.update(
                {
                    "receptor_sensitivity_selected_interactions": len(
                        receptor_selected
                    ),
                    "receptor_sensitivity_ranking_rows": len(
                        receptor_rankings
                    ),
                }
            )
        expected_outputs = [
            "condition_communication_networks.rds",
            "mapped_differential_network.rds",
            SELECTED_FILENAME,
            RANKING_FILENAME,
            "session_info.txt",
        ]
        if emit_receptor_sensitivity:
            expected_outputs.extend(
                [
                    RECEPTOR_SENSITIVITY_SELECTED_FILENAME,
                    RECEPTOR_SENSITIVITY_RANKING_FILENAME,
                ]
            )
        outputs: dict[str, dict[str, object]] = {}
        for filename in expected_outputs:
            path = output / filename
            if not path.is_file():
                raise RuntimeError(f"CellChat output is missing: {filename}")
            outputs[filename] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest.update(
            {
                "status": "complete",
                "returncode": 0,
                "elapsed_seconds": time.perf_counter() - started,
                "outputs": outputs,
                "result_counts": result_counts,
            }
        )
        write_json(manifest_path, manifest)
        return manifest
    except BaseException as error:
        cleanup_errors = process_registry.terminate_all()
        for cleanup_error in cleanup_errors:
            error.add_note(f"CellChat cleanup: {cleanup_error}")
        if completed is not None:
            (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        manifest.update(
            {
                "status": "failed",
                "returncode": None if completed is None else completed.returncode,
                "elapsed_seconds": time.perf_counter() - started,
                "failure": _external_failure(completed, error),
            }
        )
        write_json(manifest_path, manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--resource-manifest", required=True, type=Path)
    parser.add_argument("--cellchat-database", required=True, type=Path)
    parser.add_argument("--environment-manifest", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--condition-key", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--emit-receptor-sensitivity", action="store_true")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = run(
        args.input_h5ad,
        args.output_dir,
        input_manifest=args.input_manifest,
        resource_path=args.resource,
        resource_manifest=args.resource_manifest,
        database_rds=args.cellchat_database,
        environment_manifest=args.environment_manifest,
        rscript=args.rscript,
        python_executable=args.python_executable,
        dataset_id=args.dataset_id,
        cell_type_key=args.cell_type_key,
        condition_key=args.condition_key,
        target=args.target,
        reference=args.reference,
        threads=args.threads,
        min_cells=args.min_cells,
        emit_receptor_sensitivity=args.emit_receptor_sensitivity,
        timeout_seconds=args.timeout_seconds,
        preflight_only=args.preflight_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DATABASE_SHA256",
    "METHOD_COMMIT",
    "METHOD_VERSION",
    "RANKING_COLUMNS",
    "RESOURCE_ID",
    "RESOURCE_ROWS",
    "RESOURCE_SHA256",
    "_input_manifest_sha256",
    "_prepare_metadata",
    "_validate_rankings",
    "_validate_resource",
    "_validate_selected_and_rankings",
    "preflight",
    "run",
]
