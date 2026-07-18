"""Run the paper-matched scDiffCom 1.1.1 condition-aware benchmark."""

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

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_ID = "scdiffcom"
METHOD_VERSION = "1.1.1"
METHOD_COMMIT = "7877de254380cfb3a0457d5548969129aeb47137"
RESOURCE_ID = "ConnectomeDB2020_Hou_2020_human"
RESOURCE_SHA256 = "e781363288a26c15e03246500111bfecb818eef997f5ebe1b936aaa465151c3a"
RESOURCE_ROWS = 2293
SCHEMA_VERSION = "crychic-scdiffcom-condition-aware-benchmark-v1"
ENVIRONMENT_SCHEMA = "crychic-cellchat-paper-environment-v1"
SEED = 20260717
DEFAULT_ITERATIONS = 1000
MAX_CORES = 8
START_MEMORY_LIMIT_PERCENT = 60.0
STOP_MEMORY_LIMIT_PERCENT = 68.0
PROCESS_POLL_SECONDS = 5.0
OUTPUT_FILENAMES = (
    "scdiffcom_result.rds",
    "detected_calls.tsv.gz",
    "condition_specific_calls.tsv.gz",
    "analysis_cell_types.tsv",
    "session_info.txt",
    "condition_cell_pair_rankings.tsv",
)
LRI_COLUMNS = (
    "LRI",
    "LIGAND_1",
    "LIGAND_2",
    "RECEPTOR_1",
    "RECEPTOR_2",
    "RECEPTOR_3",
)
SINGLE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


@dataclass(frozen=True, slots=True)
class _PreparedExport:
    source_cell_types: tuple[str, ...]
    matched_interactions: int
    audit: dict[str, Any]
    paths: dict[str, Path]


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


def _validate_environment_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json_object(path, label="environment manifest")
    if manifest.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise ValueError("environment manifest schema is unsupported")
    method = manifest.get("scdiffcom")
    if not isinstance(method, dict):
        raise ValueError("environment manifest is missing scdiffcom")
    if method.get("version") != METHOD_VERSION:
        raise ValueError("environment scDiffCom version does not match 1.1.1")
    if method.get("git_commit") != METHOD_COMMIT:
        raise ValueError("environment scDiffCom commit does not match the pin")
    return manifest


def _validate_resource(
    resource_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if sha256_file(resource_path) != RESOURCE_SHA256:
        raise ValueError("ConnectomeDB2020 payload checksum does not match the pin")
    manifest = _read_json_object(manifest_path, label="resource manifest")
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
        "scdiffcom_covered",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"ConnectomeDB2020 columns are missing: {sorted(missing)}")
    covered = table["scdiffcom_covered"].str.lower().eq("true")
    if len(table) != RESOURCE_ROWS or not covered.all():
        raise ValueError("scDiffCom must cover all 2293 ConnectomeDB2020 pairs")
    if table.duplicated(["ligand", "receptor"]).any():
        raise ValueError(
            "ConnectomeDB2020 directed ligand-receptor pairs must be unique"
        )
    if table[["ligand", "receptor"]].isna().any().any():
        raise ValueError("ConnectomeDB2020 simple pairs cannot contain missing genes")
    lri = pd.DataFrame(
        {
            "LRI": table["harmonized_interaction_id"].astype(str),
            "LIGAND_1": table["ligand"].astype(str),
            "LIGAND_2": pd.Series(pd.NA, index=table.index, dtype="string"),
            "RECEPTOR_1": table["receptor"].astype(str),
            "RECEPTOR_2": pd.Series(pd.NA, index=table.index, dtype="string"),
            "RECEPTOR_3": pd.Series(pd.NA, index=table.index, dtype="string"),
        }
    )
    return table, lri.loc[:, list(LRI_COLUMNS)], manifest


def _r_environment(rscript: Path) -> dict[str, str]:
    packages = ("scDiffCom", "Seurat", "future", "future.apply", "data.table")
    expression = "cat('R\\t', as.character(getRversion()), '\\n', sep=''); " + " ".join(
        f"cat('{package}\\t', as.character(packageVersion('{package}')), "
        "'\\n', sep='');"
        for package in packages
    )
    completed = subprocess.run(
        [str(rscript), "-e", expression],
        check=True,
        capture_output=True,
        text=True,
    )
    versions: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, value = line.split("\t", 1)
        versions[key] = value
    if versions.get("scDiffCom") != METHOD_VERSION:
        raise RuntimeError("R environment does not contain scDiffCom 1.1.1")
    return versions


def _memory_used_percent() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(value.strip().split()[0])
    if set(values) != {"MemTotal", "MemAvailable"}:
        raise RuntimeError("cannot determine system memory availability")
    return 100.0 * (values["MemTotal"] - values["MemAvailable"]) / values["MemTotal"]


def _write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _prepare_exports(
    input_h5ad: Path,
    resource: pd.DataFrame,
    lri: pd.DataFrame,
    output_dir: Path,
    *,
    cell_type_key: str,
    condition_key: str,
    target: str,
    reference: str,
    counts_layer: str,
) -> _PreparedExport:
    source = ad.read_h5ad(input_h5ad, backed="r")
    try:
        required = {cell_type_key, condition_key}
        missing = required.difference(source.obs.columns)
        if missing:
            raise ValueError(f"prepared h5ad metadata is missing: {sorted(missing)}")
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("prepared h5ad cell and gene identifiers must be unique")
        metadata = source.obs.loc[:, [cell_type_key, condition_key]].copy()
        if metadata.isna().any().any():
            raise ValueError("prepared h5ad benchmark metadata contains null values")
        conditions = set(metadata[condition_key].astype(str))
        if conditions != {target, reference}:
            raise ValueError(
                f"observed conditions {sorted(conditions)} do not match the contrast"
            )
        cell_ids = source.obs_names.astype(str).tolist()
        if any("\t" in value or "\n" in value for value in cell_ids):
            raise ValueError("cell identifiers cannot contain tabs or newlines")
        if counts_layer not in source.layers:
            raise ValueError(
                f"prepared h5ad is missing raw counts layer {counts_layer!r}"
            )
        counts = source.layers[counts_layer]
        if not sparse.issparse(counts):
            raise ValueError("raw counts layer must be sparse")
        resource_genes = set(resource["ligand"]).union(resource["receptor"])
        source_genes = source.var_names.astype(str).tolist()
        selected_indices = [
            index for index, gene in enumerate(source_genes) if gene in resource_genes
        ]
        selected_genes = [source_genes[index] for index in selected_indices]
        if not selected_genes:
            raise ValueError("no ConnectomeDB2020 gene is present in the input")
        selected_counts = counts[:, selected_indices].tocsr()
        if selected_counts.data.size and (
            np.any(selected_counts.data < 0)
            or not np.all(
                np.equal(selected_counts.data, np.floor(selected_counts.data))
            )
        ):
            raise ValueError("counts layer must contain non-negative integers")
        present = set(selected_genes)
        matched = resource["ligand"].isin(present) & resource["receptor"].isin(present)
        matched_interactions = int(matched.sum())
        if matched_interactions == 0:
            raise ValueError(
                "no complete ConnectomeDB2020 pair is present in the input"
            )

        counts_path = output_dir / "scdiffcom_counts.mtx"
        genes_path = output_dir / "scdiffcom_genes.tsv"
        cells_path = output_dir / "scdiffcom_cells.tsv"
        metadata_path = output_dir / "scdiffcom_metadata.tsv"
        lri_path = output_dir / "scdiffcom_connectomedb2020_lri.tsv"
        mmwrite(counts_path, selected_counts.transpose().tocoo(), field="integer")
        _write_lines(genes_path, selected_genes)
        _write_lines(cells_path, cell_ids)
        export_metadata = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "cell_type": metadata[cell_type_key].astype(str).to_numpy(),
                "condition": metadata[condition_key].astype(str).to_numpy(),
            }
        )
        export_metadata.to_csv(metadata_path, sep="\t", index=False)
        lri.to_csv(lri_path, sep="\t", index=False, na_rep="NA")
        cell_type_counts = (
            export_metadata.groupby(["condition", "cell_type"], sort=True)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        source_cell_types = tuple(sorted(export_metadata["cell_type"].unique()))
        audit = {
            "shape": [int(source.n_obs), int(source.n_vars)],
            "export_shape_genes_by_cells": [len(selected_genes), len(cell_ids)],
            "counts_layer": counts_layer,
            "counts_nnz_exported": int(selected_counts.nnz),
            "resource_genes": len(resource_genes),
            "resource_genes_present": len(selected_genes),
            "resource_genes_missing": len(resource_genes.difference(present)),
            "resource_interactions": len(resource),
            "matched_interactions": matched_interactions,
            "source_cell_types": list(source_cell_types),
            "cells_by_condition": {
                str(key): int(value)
                for key, value in export_metadata.groupby("condition").size().items()
            },
            "cell_type_condition_support": cell_type_counts.to_dict(orient="records"),
        }
        paths = {
            "counts": counts_path,
            "genes": genes_path,
            "cells": cells_path,
            "metadata": metadata_path,
            "lri": lri_path,
        }
        return _PreparedExport(
            source_cell_types=source_cell_types,
            matched_interactions=matched_interactions,
            audit=audit,
            paths=paths,
        )
    finally:
        source.file.close()


def _build_rankings(
    calls: pd.DataFrame,
    *,
    source_cell_types: tuple[str, ...],
    eligible_cell_types: tuple[str, ...],
    target: str,
    reference: str,
    dataset_id: str,
    matched_interactions: int,
) -> pd.DataFrame:
    required = {"condition", "sender", "receiver", "LRI"}
    missing = required.difference(calls.columns)
    if missing:
        raise ValueError(
            f"condition-specific call columns are missing: {sorted(missing)}"
        )
    if not calls.empty and not set(calls["condition"]).issubset({target, reference}):
        raise ValueError("condition-specific calls contain an unknown condition")
    normalized = calls.loc[:, ["condition", "sender", "receiver", "LRI"]].copy()
    normalized["sender_unordered"] = normalized[["sender", "receiver"]].min(axis=1)
    normalized["receiver_unordered"] = normalized[["sender", "receiver"]].max(axis=1)
    counts = (
        normalized.groupby(
            ["condition", "sender_unordered", "receiver_unordered"],
            observed=True,
            sort=True,
        )
        .size()
        .rename("ranked_strength")
        .reset_index()
        .rename(
            columns={
                "sender_unordered": "sender",
                "receiver_unordered": "receiver",
            }
        )
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
        counts,
        on=["condition", "sender", "receiver"],
        how="left",
        validate="one_to_one",
    )
    eligible = set(eligible_cell_types)
    result["pair_eligible"] = result["sender"].isin(eligible) & result["receiver"].isin(
        eligible
    )
    result["ranked_strength"] = result["ranked_strength"].astype("Float64")
    fill = result["pair_eligible"] & result["ranked_strength"].isna()
    result.loc[fill, "ranked_strength"] = 0.0
    result.loc[~result["pair_eligible"], "ranked_strength"] = pd.NA
    result["dataset"] = dataset_id
    result["method"] = METHOD_ID
    result["method_version"] = METHOD_VERSION
    result["resource"] = RESOURCE_ID
    result["ranking_semantics"] = (
        "cardinality_of_scdiffcom_detected_BH_significant_UP_DOWN_directed_lr_"
        "after_unordered_cell_pair_collapse;GetTableCCI_detected_simplified_FALSE;"
        "default_specificity_and_score_filters;Seurat_LogNormalize_from_raw_counts"
    )
    result["condition_specific_directed_lr"] = result["ranked_strength"]
    result["estimable_directed_lr"] = np.where(
        result["pair_eligible"],
        np.where(
            result["sender"].eq(result["receiver"]),
            matched_interactions,
            2 * matched_interactions,
        ),
        pd.NA,
    )
    result["status"] = np.where(result["pair_eligible"], "observed", "not_estimable")
    result["reason_code"] = np.where(
        result["pair_eligible"],
        "",
        "cell_type_removed_by_scdiffcom_default_min_cells",
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
    ]


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


def _run_r_process(
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


def _failure_from_logs(
    *,
    returncode: int,
    stderr_path: Path,
    memory_guard_triggered: bool,
) -> dict[str, Any]:
    lines = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if memory_guard_triggered:
        return {
            "type": "MemoryGuardExceeded",
            "message": (
                f"system memory usage reached {STOP_MEMORY_LIMIT_PERCENT:.0f}% and "
                "the scDiffCom process group was terminated"
            ),
            "returncode": returncode,
            "stderr_tail": "\n".join(lines[-40:]),
        }
    return {
        "type": "ExternalProcessError",
        "message": f"scDiffCom R runner exited with code {returncode}",
        "returncode": returncode,
        "stderr_tail": "\n".join(lines[-40:]),
    }


def run(
    input_h5ad: Path,
    output_dir: Path,
    *,
    input_manifest: Path,
    resource_path: Path,
    resource_manifest: Path,
    environment_manifest: Path,
    rscript: Path,
    dataset_id: str,
    condition_key: str,
    target: str,
    reference: str,
    cell_type_key: str = "cell_type",
    counts_layer: str = "counts",
    cores: int = MAX_CORES,
    iterations: int = DEFAULT_ITERATIONS,
    prepare_only: bool = False,
    overwrite: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if target == reference:
        raise ValueError("target and reference must differ")
    if isinstance(cores, bool) or not isinstance(cores, int) or not 1 <= cores <= 8:
        raise ValueError("cores must be an integer between 1 and 8")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("iterations must be a positive integer")
    for path in (
        input_h5ad,
        input_manifest,
        resource_path,
        resource_manifest,
        environment_manifest,
        rscript,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    starting_memory = _memory_used_percent()
    if starting_memory >= START_MEMORY_LIMIT_PERCENT:
        raise RuntimeError(
            f"system memory usage is {starting_memory:.1f}%; must be below 60%"
        )
    output = prepare_output(output_dir, overwrite=overwrite)
    started = time.time()
    input_payload = _read_json_object(input_manifest, label="input manifest")
    expected_input_sha = _input_manifest_sha256(input_payload, input_h5ad.name)
    actual_input_sha = sha256_file(input_h5ad)
    if actual_input_sha != expected_input_sha:
        raise ValueError("prepared h5ad checksum does not match its manifest")
    resource, lri, resource_payload = _validate_resource(
        resource_path, resource_manifest
    )
    environment_payload = _validate_environment_manifest(environment_manifest)
    r_versions = _r_environment(rscript)
    prepared = _prepare_exports(
        input_h5ad,
        resource,
        lri,
        output,
        cell_type_key=cell_type_key,
        condition_key=condition_key,
        target=target,
        reference=reference,
        counts_layer=counts_layer,
    )
    export_records = {
        name: {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in prepared.paths.items()
    }
    protocol = {
        "article_doi": "10.1093/nargab/lqaf084",
        "supplementary_section": "S4",
        "scenario": "condition_aware",
        "contrast_order": {"cond1_reference": reference, "cond2_target": target},
        "iterations": iterations,
        "threshold_p_value_de": 0.05,
        "multiple_testing": "Benjamini-Hochberg",
        "threshold_logfc": float(np.log(1.5)),
        "condition_specific_calls": "REGULATION_UP_target_or_DOWN_reference",
        "other_run_interaction_analysis_parameters": "package_defaults",
        "normalization": "Seurat_LogNormalize_default_scale_factor_10000",
        "normalization_deviation": (
            "raw counts were re-normalized because the frozen counts-ready H5AD "
            "does not preserve the exact paper-side Seurat normalized matrix"
        ),
        "seurat5_compatibility": (
            "isolated_runtime_namespace_bridge_maps_legacy_GetAssayData_slot_to_"
            "identical_layer_semantics;installed_packages_unmodified"
        ),
        "future_backend": "multicore_on_unix_else_multisession",
        "future_workers": cores,
        "blas_threads_per_worker": 1,
        "seed": SEED,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared" if prepare_only else "running",
        "dataset_id": dataset_id,
        "method": {
            "id": METHOD_ID,
            "version": METHOD_VERSION,
            "git_commit": METHOD_COMMIT,
        },
        "protocol": protocol,
        "input": {
            "filename": input_h5ad.name,
            "sha256": actual_input_sha,
            "manifest_filename": input_manifest.name,
            "manifest_sha256": sha256_file(input_manifest),
            **prepared.audit,
        },
        "resource": {
            "resource_id": RESOURCE_ID,
            "rows": RESOURCE_ROWS,
            "sha256": RESOURCE_SHA256,
            "manifest_sha256": sha256_file(resource_manifest),
            "manifest_schema": resource_payload.get("schema_version"),
        },
        "environment": {
            "manifest_filename": environment_manifest.name,
            "manifest_sha256": sha256_file(environment_manifest),
            "environment_name": environment_payload.get("environment_name"),
            "r_packages": r_versions,
        },
        "exports": export_records,
        "memory_guard": {
            "start_limit_percent": START_MEMORY_LIMIT_PERCENT,
            "stop_limit_percent": STOP_MEMORY_LIMIT_PERCENT,
            "starting_used_percent": starting_memory,
        },
        "code": git_metadata(repo_root),
        "formal_p_or_q_emitted": True,
        "failure": None,
    }
    manifest_path = output / "run_manifest.json"
    write_json(manifest_path, manifest)
    if prepare_only:
        manifest["elapsed_seconds"] = time.time() - started
        write_json(manifest_path, manifest)
        return manifest
    memory_before_r = _memory_used_percent()
    if memory_before_r >= START_MEMORY_LIMIT_PERCENT:
        manifest.update(
            {
                "status": "failed",
                "elapsed_seconds": time.time() - started,
                "failure": {
                    "type": "MemoryStartGuardExceeded",
                    "message": (
                        f"memory usage rose to {memory_before_r:.1f}% before R start"
                    ),
                },
            }
        )
        write_json(manifest_path, manifest)
        raise RuntimeError("memory usage must be below 60% before scDiffCom starts")
    r_runner = Path(__file__).with_name("run.R")
    command = [
        str(rscript),
        str(r_runner),
        str(prepared.paths["counts"]),
        str(prepared.paths["genes"]),
        str(prepared.paths["cells"]),
        str(prepared.paths["metadata"]),
        str(prepared.paths["lri"]),
        str(output),
        dataset_id,
        target,
        reference,
        str(cores),
        str(iterations),
        str(SEED),
    ]
    manifest["command"] = command
    write_json(manifest_path, manifest)
    process_environment = os.environ.copy()
    process_environment.update(SINGLE_THREAD_ENVIRONMENT)
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    outcome = _run_r_process(
        command,
        environment=process_environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    manifest["memory_guard"]["peak_used_percent"] = outcome.peak_memory_percent
    if outcome.returncode != 0 or outcome.memory_guard_triggered:
        manifest.update(
            {
                "status": "failed",
                "returncode": outcome.returncode,
                "elapsed_seconds": time.time() - started,
                "failure": _failure_from_logs(
                    returncode=outcome.returncode,
                    stderr_path=stderr_path,
                    memory_guard_triggered=outcome.memory_guard_triggered,
                ),
            }
        )
        write_json(manifest_path, manifest)
        raise RuntimeError("scDiffCom R runner failed; inspect run_manifest.json")
    calls_path = output / "condition_specific_calls.tsv.gz"
    eligible_path = output / "analysis_cell_types.tsv"
    calls = pd.read_csv(calls_path, sep="\t", compression="gzip")
    eligible_table = pd.read_csv(eligible_path, sep="\t", dtype=str)
    if list(eligible_table.columns) != ["cell_type"]:
        raise RuntimeError("analysis_cell_types.tsv schema is invalid")
    eligible_cell_types = tuple(sorted(eligible_table["cell_type"].dropna().unique()))
    if not set(eligible_cell_types).issubset(prepared.source_cell_types):
        raise RuntimeError("scDiffCom returned a cell type outside the source universe")
    rankings = _build_rankings(
        calls,
        source_cell_types=prepared.source_cell_types,
        eligible_cell_types=eligible_cell_types,
        target=target,
        reference=reference,
        dataset_id=dataset_id,
        matched_interactions=prepared.matched_interactions,
    )
    ranking_path = output / "condition_cell_pair_rankings.tsv"
    rankings.to_csv(ranking_path, sep="\t", index=False, lineterminator="\n")
    outputs: dict[str, Any] = {}
    for filename in (*OUTPUT_FILENAMES, "stdout.log", "stderr.log"):
        path = output / filename
        if not path.is_file():
            raise RuntimeError(f"scDiffCom output is missing: {filename}")
        outputs[filename] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest.update(
        {
            "status": "complete",
            "returncode": 0,
            "elapsed_seconds": time.time() - started,
            "analysis_cell_types": list(eligible_cell_types),
            "outputs": outputs,
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--resource-manifest", required=True, type=Path)
    parser.add_argument("--environment-manifest", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--condition-key", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--cores", type=int, default=MAX_CORES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.input_h5ad,
        args.output_dir,
        input_manifest=args.input_manifest,
        resource_path=args.resource,
        resource_manifest=args.resource_manifest,
        environment_manifest=args.environment_manifest,
        rscript=args.rscript,
        dataset_id=args.dataset_id,
        condition_key=args.condition_key,
        target=args.target,
        reference=args.reference,
        cell_type_key=args.cell_type_key,
        counts_layer=args.counts_layer,
        cores=args.cores,
        iterations=args.iterations,
        prepare_only=args.prepare_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
