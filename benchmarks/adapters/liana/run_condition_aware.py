"""Run the paper-matched LIANA+ 1.5.0 multi-sample benchmark arm."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from anndata.io import read_elem

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXACT_SCRIPT = Path(__file__).with_name("run_condition_aware_exact.py")
METHOD_ID = "liana_plus_de"
METHOD_VERSION = "1.5.0"
METHOD_TAG = "v1.5.0"
METHOD_COMMIT = "8f8f3d6617b190aaaf0d50fdff68aa16426abaf8"
METHOD_WHEEL_SHA256 = "180e14705f1ce0ad645c78ceb7f7d7f5ea261bacef4eb91abf59bd8e202ab1d6"
DECOUPLER_VERSION = "1.8.0"
DECOUPLER_WHEEL_SHA256 = (
    "726244bd809e70412ac82b51defc92b848b5a8f347084d1b4479d9b16ecd6228"
)
PYDESEQ2_VERSION = "0.5.0"
PYDESEQ2_WHEEL_SHA256 = (
    "6cd89bd3bdf48ec62cc0ffca8ca92a4ad59b2ea751be22b1f594ce2a53a58372"
)
ANNDATA_VERSION = "0.10.8"
NUMBA_VERSION = "0.60.0"
NUMPY_VERSION = "1.26.4"
PANDAS_VERSION = "2.2.3"
SCANPY_VERSION = "1.10.4"
VIGNETTE_GIT_BLOB = "105c49abcab2a278f0f5e88ade5a57d0e85a7d48"
VIGNETTE_SHA256 = "56712d7c35ce9fb558f3adfa066dad904b0bbdbce728fe006696454f7eac621e"
VIGNETTE_URL = (
    "https://raw.githubusercontent.com/saezlab/liana-py/v1.5.0/"
    "docs/source/notebooks/targeted.ipynb"
)
RESOURCE_ID = "ConnectomeDB2020_Hou_2020_human"
RESOURCE_SHA256 = "e781363288a26c15e03246500111bfecb818eef997f5ebe1b936aaa465151c3a"
RESOURCE_ROWS = 2293
SCHEMA_VERSION = "crychic-liana-condition-aware-s4-benchmark-v1"
SEED = 20260717
DEFAULT_CORES = 8
MAX_CORES = 16
EXPR_PROP = 0.1
ALPHA = 0.05
MIN_CELLS_PER_PSEUDOBULK = 10
MIN_COUNTS_PER_PSEUDOBULK = 10_000
MIN_GENE_COUNT = 5
MIN_GENE_TOTAL_COUNT = 10
MIN_REPLICATES_PER_CONDITION = 2
START_MEMORY_LIMIT_PERCENT = 60.0
STOP_MEMORY_LIMIT_PERCENT = 68.0
PROCESS_POLL_SECONDS = 5.0
EXACT_OUTPUTS = (
    "pseudobulk_counts.h5ad",
    "pseudobulk_support.tsv",
    "dea_results.tsv.gz",
    "liana_lr_results.tsv.gz",
    "condition_specific_calls.tsv.gz",
    "analysis_cell_types.tsv",
    "environment_freeze.txt",
    "exact_analysis_manifest.json",
)
SINGLE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}
EXPECTED_ENVIRONMENT = {
    "liana": METHOD_VERSION,
    "decoupler": DECOUPLER_VERSION,
    "pydeseq2": PYDESEQ2_VERSION,
    "anndata": ANNDATA_VERSION,
    "numba": NUMBA_VERSION,
    "numpy": NUMPY_VERSION,
    "pandas": PANDAS_VERSION,
    "scanpy": SCANPY_VERSION,
}


class ExactEnvironmentUnavailable(RuntimeError):
    """Raised when the paper-pinned Python environment is not usable."""


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    metadata_path: Path
    genes_path: Path
    resource_path: Path
    source_cell_types: tuple[str, ...]
    matched_interactions: int
    audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ProcessOutcome:
    returncode: int
    peak_memory_percent: float
    memory_guard_triggered: bool


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], payload)


def _input_manifest_sha256(payload: dict[str, Any], filename: str) -> str:
    output = payload.get("output")
    if isinstance(output, dict) and output.get("filename") == filename:
        digest = output.get("sha256")
        if isinstance(digest, str):
            return digest
    if output == filename:
        digest = payload.get("output_sha256")
        if isinstance(digest, str):
            return digest
    raise ValueError("input manifest does not bind the requested h5ad checksum")


def _validate_resource(
    resource_path: Path,
    resource_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    digest = sha256_file(resource_path)
    if digest != RESOURCE_SHA256:
        raise ValueError("ConnectomeDB2020 payload checksum does not match the pin")
    manifest = _read_json_object(resource_manifest_path, label="resource manifest")
    payload = manifest.get("payload")
    if (
        manifest.get("resource_id") != RESOURCE_ID
        or manifest.get("rows") != RESOURCE_ROWS
        or not isinstance(payload, dict)
        or payload.get("sha256") != RESOURCE_SHA256
    ):
        raise ValueError("ConnectomeDB2020 manifest does not match the frozen resource")
    table = pd.read_csv(resource_path, sep="\t", dtype=str)
    required = {
        "harmonized_interaction_id",
        "ligand",
        "receptor",
        "liana_covered",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"ConnectomeDB2020 columns are missing: {sorted(missing)}")
    if (
        len(table) != RESOURCE_ROWS
        or not table["liana_covered"].str.lower().eq("true").all()
    ):
        raise ValueError("LIANA+ must cover all 2293 ConnectomeDB2020 pairs")
    if table[["harmonized_interaction_id", "ligand", "receptor"]].isna().any().any():
        raise ValueError("ConnectomeDB2020 simple pairs must be complete")
    if table.duplicated(["ligand", "receptor"]).any():
        raise ValueError(
            "ConnectomeDB2020 directed ligand-receptor pairs must be unique"
        )
    selected = table.loc[
        :,
        ["harmonized_interaction_id", "ligand", "receptor"],
    ].rename(columns={"harmonized_interaction_id": "interaction_id"})
    return selected, manifest


def _probe_environment(exact_python: Path) -> dict[str, Any]:
    if not exact_python.is_file():
        raise ExactEnvironmentUnavailable(
            f"exact Python executable is missing: {exact_python}"
        )
    completed = subprocess.run(
        [str(exact_python), str(EXACT_SCRIPT), "--probe"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ExactEnvironmentUnavailable(
            "LIANA+ exact-environment probe failed: " + completed.stderr[-2000:]
        )
    try:
        payload: object = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ExactEnvironmentUnavailable(
            "LIANA+ exact-environment probe returned invalid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ExactEnvironmentUnavailable("LIANA+ environment probe is not an object")
    result = cast(dict[str, Any], payload)
    versions = result.get("packages")
    if not isinstance(versions, dict):
        raise ExactEnvironmentUnavailable("LIANA+ environment probe lacks packages")
    mismatches = {
        name: {"expected": expected, "observed": versions.get(name)}
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if versions.get(name) != expected
    }
    if mismatches:
        raise ExactEnvironmentUnavailable(
            f"paper-pinned package versions do not match: {mismatches}"
        )
    return result


def _memory_used_percent() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(value.strip().split()[0])
    if set(values) != {"MemTotal", "MemAvailable"}:
        raise RuntimeError("cannot determine system memory availability")
    return 100.0 * (values["MemTotal"] - values["MemAvailable"]) / values["MemTotal"]


def _prepare_inputs(
    input_h5ad: Path,
    output_dir: Path,
    resource: pd.DataFrame,
    *,
    replicate_key: str,
    subject_key: str,
    condition_key: str,
    cell_type_key: str,
    counts_layer: str,
    target: str,
    reference: str,
) -> _PreparedInputs:
    with h5py.File(input_h5ad, "r") as handle:
        if "obs" not in handle or "var" not in handle:
            raise ValueError("input h5ad lacks obs or var axes")
        obs = read_elem(handle["obs"])
        var = read_elem(handle["var"])
        counts_path = f"layers/{counts_layer}"
        if counts_path not in handle:
            raise ValueError(f"input h5ad lacks raw counts layer {counts_layer!r}")
        counts = handle[counts_path]
        if not isinstance(counts, h5py.Group):
            raise ValueError("raw counts layer must use sparse CSR encoding")
        encoding = counts.attrs.get("encoding-type")
        if encoding != "csr_matrix" or set(counts.keys()) != {
            "data",
            "indices",
            "indptr",
        }:
            raise ValueError("raw counts layer is not a canonical CSR matrix")
        shape = tuple(int(value) for value in counts.attrs["shape"])
        counts_dtype = str(counts["data"].dtype)
        counts_nnz = int(counts["data"].shape[0])
    if shape != (len(obs), len(var)):
        raise ValueError("raw counts shape disagrees with the h5ad axes")
    required = {replicate_key, subject_key, condition_key, cell_type_key}
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(f"prepared h5ad metadata is missing: {sorted(missing)}")
    if not obs.index.is_unique or not var.index.is_unique:
        raise ValueError("prepared h5ad cell and gene identifiers must be unique")
    metadata_columns = list(
        dict.fromkeys([replicate_key, subject_key, condition_key, cell_type_key])
    )
    selected = obs.loc[:, metadata_columns].copy()
    if selected.isna().any().any():
        raise ValueError("prepared h5ad benchmark metadata contains null values")
    selected = selected.astype(str)
    conditions = set(selected[condition_key])
    if conditions != {target, reference}:
        raise ValueError(
            f"observed conditions {sorted(conditions)} do not match the contrast"
        )
    metadata = pd.DataFrame(
        {
            "cell_id": obs.index.astype(str),
            "replicate_id": selected[replicate_key].to_numpy(),
            "subject_id": selected[subject_key].to_numpy(),
            "condition": selected[condition_key].to_numpy(),
            "cell_type": selected[cell_type_key].to_numpy(),
        }
    )
    if (
        metadata.astype(str)
        .apply(lambda column: column.str.contains("\t|\n"))
        .any()
        .any()
    ):
        raise ValueError("exported metadata cannot contain tabs or newlines")
    design = metadata.loc[
        :, ["replicate_id", "subject_id", "condition"]
    ].drop_duplicates(ignore_index=True)
    if design.duplicated("replicate_id").any():
        raise ValueError("replicate IDs map to multiple subjects or conditions")
    genes = var.index.astype(str)
    if genes.str.contains("\t|\n").any():
        raise ValueError("gene identifiers cannot contain tabs or newlines")
    metadata_path = output_dir / "liana_metadata.tsv"
    genes_path = output_dir / "liana_genes.tsv"
    resource_path = output_dir / "liana_connectomedb2020.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=False)
    pd.Series(genes, name="gene").to_csv(genes_path, sep="\t", index=False)
    resource.to_csv(resource_path, sep="\t", index=False)
    present = set(genes)
    matched = resource["ligand"].isin(present) & resource["receptor"].isin(present)
    matched_interactions = int(matched.sum())
    if matched_interactions == 0:
        raise ValueError("no complete ConnectomeDB2020 interaction is present")
    source_cell_types = tuple(sorted(metadata["cell_type"].unique()))
    support = (
        metadata.groupby(["condition", "cell_type"], observed=True, sort=True)
        .agg(n_cells=("cell_id", "size"), n_replicates=("replicate_id", "nunique"))
        .reset_index()
    )
    audit: dict[str, Any] = {
        "shape": [shape[0], shape[1]],
        "counts_layer": counts_layer,
        "counts_hdf5_path": f"layers/{counts_layer}",
        "counts_encoding": "csr_matrix",
        "counts_dtype": counts_dtype,
        "counts_nnz": counts_nnz,
        "replicate_key": replicate_key,
        "analysis_unit": "subject" if replicate_key == subject_key else "sample",
        "n_replicates": int(metadata["replicate_id"].nunique()),
        "n_subjects": int(metadata["subject_id"].nunique()),
        "replicates_by_condition": {
            str(key): int(value)
            for key, value in design.groupby("condition", observed=True)["replicate_id"]
            .nunique()
            .items()
        },
        "subjects_by_condition": {
            str(key): int(value)
            for key, value in design.groupby("condition", observed=True)["subject_id"]
            .nunique()
            .items()
        },
        "cells_by_condition": {
            str(key): int(value)
            for key, value in metadata.groupby("condition", observed=True)
            .size()
            .items()
        },
        "source_cell_types": list(source_cell_types),
        "cell_type_condition_support": support.to_dict(orient="records"),
        "resource_interactions": len(resource),
        "matched_interactions": matched_interactions,
    }
    return _PreparedInputs(
        metadata_path=metadata_path,
        genes_path=genes_path,
        resource_path=resource_path,
        source_cell_types=source_cell_types,
        matched_interactions=matched_interactions,
        audit=audit,
    )


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _run_exact_process(
    command: list[str],
    *,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> _ProcessOutcome:
    peak_memory = _memory_used_percent()
    guard = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                memory = _memory_used_percent()
                peak_memory = max(peak_memory, memory)
                if memory >= STOP_MEMORY_LIMIT_PERCENT:
                    guard = True
                    _terminate_process_group(process)
                    break
                time.sleep(PROCESS_POLL_SECONDS)
        except BaseException:
            _terminate_process_group(process)
            raise
        returncode = process.wait()
    return _ProcessOutcome(
        returncode=returncode,
        peak_memory_percent=peak_memory,
        memory_guard_triggered=guard,
    )


def _build_rankings(
    calls: pd.DataFrame,
    lr_results: pd.DataFrame,
    *,
    source_cell_types: tuple[str, ...],
    eligible_cell_types: tuple[str, ...],
    target: str,
    reference: str,
    dataset_id: str,
    replicate_key: str,
    subject_key: str,
) -> pd.DataFrame:
    call_required = {"condition", "source", "target", "interaction_id"}
    missing_calls = call_required.difference(calls.columns)
    if missing_calls:
        raise ValueError(
            f"condition-specific call columns are missing: {sorted(missing_calls)}"
        )
    lr_required = {"source", "target", "interaction_id"}
    missing_lr = lr_required.difference(lr_results.columns)
    if missing_lr:
        raise ValueError(f"LIANA+ result columns are missing: {sorted(missing_lr)}")
    if not calls.empty and not set(calls["condition"]).issubset({target, reference}):
        raise ValueError("condition-specific calls contain an unknown condition")
    if calls.duplicated(["condition", "source", "target", "interaction_id"]).any():
        raise ValueError("condition-specific calls contain duplicate directed LR keys")
    if lr_results.duplicated(["source", "target", "interaction_id"]).any():
        raise ValueError("LIANA+ results contain duplicate directed LR keys")

    normalized = calls.loc[
        :, ["condition", "source", "target", "interaction_id"]
    ].copy()
    normalized["sender"] = normalized[["source", "target"]].min(axis=1)
    normalized["receiver"] = normalized[["source", "target"]].max(axis=1)
    call_counts = (
        normalized.groupby(
            ["condition", "sender", "receiver"], observed=True, sort=True
        )
        .size()
        .rename("ranked_strength")
        .reset_index()
    )
    estimable = lr_results.loc[:, ["source", "target", "interaction_id"]].copy()
    estimable["sender"] = estimable[["source", "target"]].min(axis=1)
    estimable["receiver"] = estimable[["source", "target"]].max(axis=1)
    estimable_counts = (
        estimable.groupby(["sender", "receiver"], observed=True, sort=True)
        .size()
        .rename("estimable_directed_lr")
        .reset_index()
    )
    pairs = pd.DataFrame(
        combinations_with_replacement(source_cell_types, 2),
        columns=["sender", "receiver"],
    )
    universe = pd.concat(
        [pairs.assign(condition=condition) for condition in (target, reference)],
        ignore_index=True,
    )
    result = universe.merge(
        call_counts,
        on=["condition", "sender", "receiver"],
        how="left",
        validate="one_to_one",
    ).merge(
        estimable_counts,
        on=["sender", "receiver"],
        how="left",
        validate="many_to_one",
    )
    eligible = set(eligible_cell_types)
    result["pair_eligible"] = result["sender"].isin(eligible) & result["receiver"].isin(
        eligible
    )
    result["ranked_strength"] = result["ranked_strength"].astype("Float64")
    observed_missing = result["pair_eligible"] & result["ranked_strength"].isna()
    result.loc[observed_missing, "ranked_strength"] = 0.0
    result.loc[~result["pair_eligible"], "ranked_strength"] = pd.NA
    result["estimable_directed_lr"] = result["estimable_directed_lr"].astype("Float64")
    result.loc[
        result["pair_eligible"] & result["estimable_directed_lr"].isna(),
        "estimable_directed_lr",
    ] = 0.0
    result.loc[~result["pair_eligible"], "estimable_directed_lr"] = pd.NA
    result["dataset"] = dataset_id
    result["method"] = METHOD_ID
    result["method_version"] = METHOD_VERSION
    result["resource"] = RESOURCE_ID
    analysis_unit = "subject" if replicate_key == subject_key else "sample"
    result["ranking_semantics"] = (
        "cardinality_of_unadjusted_interaction_pvalue_lt_0.05_condition_specific_"
        "directed_lr_by_interaction_stat_sign_after_unordered_cell_pair_collapse;"
        f"{analysis_unit}_pseudobulk;LIANA_df_to_lr_expr_prop_0.1;PyDESeq2_Wald_stat"
    )
    result["condition_specific_directed_lr"] = result["ranked_strength"]
    result["status"] = np.where(result["pair_eligible"], "observed", "not_estimable")
    result["reason_code"] = np.where(
        result["pair_eligible"],
        "",
        f"cell_type_not_estimable_for_{analysis_unit}_pseudobulk_de",
    )
    return result.loc[
        :,
        [
            "dataset",
            "method",
            "method_version",
            "resource",
            "ranking_semantics",
            "condition",
            "sender",
            "receiver",
            "ranked_strength",
            "condition_specific_directed_lr",
            "estimable_directed_lr",
            "status",
            "reason_code",
        ],
    ].sort_values(
        ["dataset", "method", "condition", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )


def _output_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _failure_payload(
    outcome: _ProcessOutcome,
    stderr_path: Path,
) -> dict[str, Any]:
    stderr_tail = "\n".join(
        stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
    )
    if outcome.memory_guard_triggered:
        return {
            "type": "MemoryGuardExceeded",
            "message": (
                f"system memory usage reached {STOP_MEMORY_LIMIT_PERCENT:.0f}% and "
                "the LIANA+ process group was terminated"
            ),
            "returncode": outcome.returncode,
            "stderr_tail": stderr_tail,
        }
    return {
        "type": "ExternalProcessError",
        "message": f"LIANA+ exact runner exited with code {outcome.returncode}",
        "returncode": outcome.returncode,
        "stderr_tail": stderr_tail,
    }


def _record_failed_outcome(
    manifest: dict[str, Any],
    outcome: _ProcessOutcome,
    stderr_path: Path,
) -> bool:
    if outcome.returncode == 0:
        return False
    manifest["status"] = "failed"
    manifest["failure"] = _failure_payload(outcome, stderr_path)
    return True


def run(
    input_h5ad: Path,
    output_dir: Path,
    *,
    input_manifest: Path,
    resource_path: Path,
    resource_manifest: Path,
    exact_python: Path,
    dataset_id: str,
    replicate_key: str,
    subject_key: str,
    condition_key: str,
    target: str,
    reference: str,
    cell_type_key: str = "cell_type",
    counts_layer: str = "counts",
    cores: int = DEFAULT_CORES,
    prepare_only: bool = False,
    overwrite: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if target == reference:
        raise ValueError("target and reference must differ")
    if (
        isinstance(cores, bool)
        or not isinstance(cores, int)
        or not 1 <= cores <= MAX_CORES
    ):
        raise ValueError(f"cores must be an integer between 1 and {MAX_CORES}")
    for path in (
        input_h5ad,
        input_manifest,
        resource_path,
        resource_manifest,
        EXACT_SCRIPT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = prepare_output(output_dir, overwrite=overwrite)
    run_manifest_path = output / "run_manifest.json"
    started = time.time()
    starting_memory = _memory_used_percent()
    base_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "dataset_id": dataset_id,
        "method": {
            "id": METHOD_ID,
            "version": METHOD_VERSION,
            "git_tag": METHOD_TAG,
            "git_commit": METHOD_COMMIT,
            "wheel_sha256": METHOD_WHEEL_SHA256,
        },
        "analysis_unit": {
            "replicate_key": replicate_key,
            "subject_key": subject_key,
            "primary_panel": replicate_key != subject_key,
            "paper_figure3_primary_panel": replicate_key != subject_key,
            "technical_sections_collapsed_before_de": replicate_key == subject_key,
        },
        "contrast": {"target": target, "reference": reference},
        "resource": {
            "id": RESOURCE_ID,
            "rows": RESOURCE_ROWS,
            "sha256": RESOURCE_SHA256,
        },
        "protocol": {
            "vignette": {
                "path": "docs/source/notebooks/targeted.ipynb",
                "git_tag": METHOD_TAG,
                "git_commit": METHOD_COMMIT,
                "git_blob": VIGNETTE_GIT_BLOB,
                "sha256": VIGNETTE_SHA256,
                "url": VIGNETTE_URL,
            },
            "liana_entrypoint": "liana.multi.df_to_lr",
            "expr_prop": EXPR_PROP,
            "complex_col": "stat",
            "stat_keys": ["stat", "pvalue", "padj"],
            "pseudobulk": {
                "implementation": "decoupler.get_pseudobulk",
                "mode": "sum",
                "min_cells": MIN_CELLS_PER_PSEUDOBULK,
                "min_counts": MIN_COUNTS_PER_PSEUDOBULK,
            },
            "gene_filter": {
                "implementation": "decoupler.filter_by_expr",
                "min_count": MIN_GENE_COUNT,
                "min_total_count": MIN_GENE_TOTAL_COUNT,
            },
            "de": {
                "implementation": "pydeseq2",
                "design": "~condition",
                "test": "Wald",
                "refit_cooks": True,
                "lfc_shrink": True,
                "deseqstats_inference": "reuse_dds.inference",
                "min_replicates_per_condition": MIN_REPLICATES_PER_CONDITION,
                "batch_covariates": [],
            },
            "condition_specific_selection": {
                "p_value_column": "interaction_pvalue",
                "p_value_threshold": ALPHA,
                "additional_multiple_testing_correction": False,
                "direction_column": "interaction_stat",
                "target_rule": "interaction_stat > 0",
                "reference_rule": "interaction_stat < 0",
                "gene_level_padj_retained": True,
            },
            "expression_filter_condition": target,
            "failure_policy": {
                "pydeseq2_or_liana_computation_error": "fail_entire_run",
                "predefined_support_ineligibility": "not_estimable",
            },
            "compatibility_bridges": [
                {
                    "id": "decoupler_1.8.0_sparse_pandas_boolean_mask_to_numpy",
                    "scope": "process_local_during_get_pseudobulk",
                    "semantic_change": False,
                    "installed_files_modified": False,
                }
            ],
            "seed": SEED,
        },
        "version_pins": {
            "decoupler": {
                "version": DECOUPLER_VERSION,
                "wheel_sha256": DECOUPLER_WHEEL_SHA256,
            },
            "pydeseq2": {
                "version": PYDESEQ2_VERSION,
                "wheel_sha256": PYDESEQ2_WHEEL_SHA256,
            },
            "anndata": ANNDATA_VERSION,
            "numba": NUMBA_VERSION,
            "numpy": NUMPY_VERSION,
            "pandas": PANDAS_VERSION,
            "scanpy": SCANPY_VERSION,
        },
        "memory_guard": {
            "start_below_percent": START_MEMORY_LIMIT_PERCENT,
            "terminate_at_percent": STOP_MEMORY_LIMIT_PERCENT,
            "starting_used_percent": starting_memory,
        },
        "git": git_metadata(repo_root),
    }
    try:
        driver_script = Path(__file__).resolve()
        exact_script = EXACT_SCRIPT.resolve()
        driver_script_sha256 = sha256_file(driver_script)
        exact_script_sha256 = sha256_file(exact_script)
        base_manifest["code"] = {
            "driver": {
                "path": str(driver_script),
                "sha256": driver_script_sha256,
            },
            "exact_worker": {
                "path": str(exact_script),
                "sha256": exact_script_sha256,
            },
        }
        input_payload = _read_json_object(input_manifest, label="input manifest")
        expected_input_sha = _input_manifest_sha256(input_payload, input_h5ad.name)
        actual_input_sha = sha256_file(input_h5ad)
        if expected_input_sha != actual_input_sha:
            raise ValueError("prepared h5ad checksum does not match its manifest")
        resource, resource_payload = _validate_resource(
            resource_path, resource_manifest
        )
        probe = _probe_environment(exact_python)
        prepared = _prepare_inputs(
            input_h5ad,
            output,
            resource,
            replicate_key=replicate_key,
            subject_key=subject_key,
            condition_key=condition_key,
            cell_type_key=cell_type_key,
            counts_layer=counts_layer,
            target=target,
            reference=reference,
        )
        base_manifest["inputs"] = {
            "h5ad": _output_record(input_h5ad),
            "manifest": _output_record(input_manifest),
            "resource_manifest": _output_record(resource_manifest),
        }
        base_manifest["resource_manifest"] = resource_payload
        base_manifest["environment"] = probe
        base_manifest["prepared_input"] = prepared.audit
        base_manifest["exports"] = {
            "metadata": _output_record(
                prepared.metadata_path,
                rows=len(pd.read_csv(prepared.metadata_path, sep="\t")),
            ),
            "genes": _output_record(
                prepared.genes_path, rows=prepared.audit["shape"][1]
            ),
            "resource": _output_record(prepared.resource_path, rows=RESOURCE_ROWS),
        }
        if prepare_only:
            base_manifest["status"] = "prepared"
            base_manifest["runtime"] = {"elapsed_seconds": time.time() - started}
            write_json(run_manifest_path, base_manifest)
            return base_manifest
        memory_before_analysis = _memory_used_percent()
        if memory_before_analysis >= START_MEMORY_LIMIT_PERCENT:
            raise RuntimeError(
                "system memory usage is "
                f"{memory_before_analysis:.1f}%; must be below 60%"
            )
        stdout_path = output / "stdout.log"
        stderr_path = output / "stderr.log"
        scratch = output / "scratch"
        scratch.mkdir()
        command = [
            str(exact_python),
            str(EXACT_SCRIPT),
            "--input-h5ad",
            str(input_h5ad.resolve()),
            "--metadata",
            str(prepared.metadata_path.resolve()),
            "--genes",
            str(prepared.genes_path.resolve()),
            "--resource",
            str(prepared.resource_path.resolve()),
            "--output-dir",
            str(output.resolve()),
            "--dataset-id",
            dataset_id,
            "--counts-layer",
            counts_layer,
            "--target",
            target,
            "--reference",
            reference,
            "--cores",
            str(cores),
            "--seed",
            str(SEED),
            "--driver-script",
            str(driver_script),
            "--driver-sha256",
            driver_script_sha256,
            "--exact-script-sha256",
            exact_script_sha256,
        ]
        environment = os.environ.copy()
        environment.update(SINGLE_THREAD_ENVIRONMENT)
        environment.update(
            {
                "PYTHONHASHSEED": "0",
                "PYTHONPATH": "",
                "TMPDIR": str(scratch.resolve()),
                "JOBLIB_TEMP_FOLDER": str(scratch.resolve()),
                "MPLCONFIGDIR": str((scratch / "matplotlib").resolve()),
                "NUMBA_CACHE_DIR": str((scratch / "numba").resolve()),
            }
        )
        outcome = _run_exact_process(
            command,
            environment=environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        base_manifest["execution"] = {
            "command": command,
            "cores": cores,
            "peak_system_memory_used_percent": outcome.peak_memory_percent,
            "memory_guard_triggered": outcome.memory_guard_triggered,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
        }
        if _record_failed_outcome(base_manifest, outcome, stderr_path):
            base_manifest["runtime"] = {"elapsed_seconds": time.time() - started}
            write_json(run_manifest_path, base_manifest)
            return base_manifest
        missing_outputs = [
            name for name in EXACT_OUTPUTS if not (output / name).is_file()
        ]
        if missing_outputs:
            raise RuntimeError(f"LIANA+ runner omitted outputs: {missing_outputs}")
        exact_manifest = _read_json_object(
            output / "exact_analysis_manifest.json", label="exact analysis manifest"
        )
        if exact_manifest.get("status") != "complete":
            raise RuntimeError("LIANA+ exact analysis did not report completion")
        calls = pd.read_csv(output / "condition_specific_calls.tsv.gz", sep="\t")
        lr_results = pd.read_csv(output / "liana_lr_results.tsv.gz", sep="\t")
        cell_types = pd.read_csv(output / "analysis_cell_types.tsv", sep="\t")
        eligible = tuple(
            sorted(
                cell_types.loc[cell_types["status"].eq("analyzed"), "cell_type"].astype(
                    str
                )
            )
        )
        rankings = _build_rankings(
            calls,
            lr_results,
            source_cell_types=prepared.source_cell_types,
            eligible_cell_types=eligible,
            target=target,
            reference=reference,
            dataset_id=dataset_id,
            replicate_key=replicate_key,
            subject_key=subject_key,
        )
        ranking_path = output / "condition_cell_pair_rankings.tsv"
        rankings.to_csv(ranking_path, sep="\t", index=False)
        output_rows = {
            "pseudobulk_support.tsv": len(
                pd.read_csv(output / "pseudobulk_support.tsv", sep="\t")
            ),
            "dea_results.tsv.gz": len(
                pd.read_csv(output / "dea_results.tsv.gz", sep="\t")
            ),
            "liana_lr_results.tsv.gz": len(lr_results),
            "condition_specific_calls.tsv.gz": len(calls),
            "analysis_cell_types.tsv": len(cell_types),
            "condition_cell_pair_rankings.tsv": len(rankings),
        }
        base_manifest["status"] = "complete"
        base_manifest["exact_analysis"] = exact_manifest
        base_manifest["results"] = {
            "analyzed_cell_types": list(eligible),
            "n_analyzed_cell_types": len(eligible),
            "n_source_cell_types": len(prepared.source_cell_types),
            "n_lr_results": len(lr_results),
            "n_condition_specific_calls": len(calls),
            "calls_by_condition": {
                str(key): int(value)
                for key, value in calls.groupby("condition", observed=True)
                .size()
                .items()
            },
        }
        all_outputs = [output / name for name in EXACT_OUTPUTS] + [ranking_path]
        base_manifest["outputs"] = {
            path.name: _output_record(path, rows=output_rows.get(path.name))
            for path in all_outputs
        }
        base_manifest["runtime"] = {
            "elapsed_seconds": time.time() - started,
            "memory_before_analysis_percent": memory_before_analysis,
        }
        write_json(run_manifest_path, base_manifest)
        return base_manifest
    except ExactEnvironmentUnavailable as error:
        base_manifest["status"] = "skipped"
        base_manifest["skip"] = {
            "reason_code": "paper_exact_environment_unavailable",
            "message": str(error),
        }
        base_manifest["runtime"] = {"elapsed_seconds": time.time() - started}
        write_json(run_manifest_path, base_manifest)
        return base_manifest
    except BaseException as error:
        base_manifest["status"] = "failed"
        base_manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        base_manifest["runtime"] = {"elapsed_seconds": time.time() - started}
        write_json(run_manifest_path, base_manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--resource-manifest", required=True, type=Path)
    parser.add_argument("--exact-python", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--replicate-key", required=True)
    parser.add_argument("--subject-key", default="subject_id")
    parser.add_argument("--condition-key", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--cores", type=int, default=DEFAULT_CORES)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.input_h5ad,
        args.output_dir,
        input_manifest=args.input_manifest,
        resource_path=args.resource,
        resource_manifest=args.resource_manifest,
        exact_python=args.exact_python,
        dataset_id=args.dataset_id,
        replicate_key=args.replicate_key,
        subject_key=args.subject_key,
        condition_key=args.condition_key,
        target=args.target,
        reference=args.reference,
        cell_type_key=args.cell_type_key,
        counts_layer=args.counts_layer,
        cores=args.cores,
        prepare_only=args.prepare_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
