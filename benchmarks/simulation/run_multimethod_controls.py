"""Run every external method on the frozen synthetic control collection."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json

SUITE_SCHEMA = "crychic-synthetic-multimethod-suite-v1"
EXECUTION_SCHEMA = "crychic-synthetic-multimethod-execution-v1"
METHODS = ("crychic", "nichenet", "liana", "cellphonedb", "cellchat")
METHOD_TRACKS = {
    "cellchat": ("lr_stlr", "H-common"),
    "cellphonedb": ("lr_stlr", "H-common"),
    "liana": ("lr_stlr", "H-common"),
    "nichenet": ("ligand_target_program", "native"),
    "crychic": ("lr_stlr", "H-common"),
}


@dataclass(frozen=True)
class ProcessResult:
    """Measured outcome of one isolated adapter process."""

    exit_code: int
    wall_time_seconds: float
    peak_rss_mb: float


@dataclass(frozen=True)
class RunnerSettings:
    """Frozen execution parameters shared by all synthetic scenarios."""

    database_root: Path
    harmonized_resource: Path
    harmonized_manifest: Path
    cellchat_environment: str
    cellphonedb_python: Path
    liana_python: Path
    crychic_python: Path
    min_cells: int = 10
    cellchat_nboot: int = 100
    cellchat_threads: int = 1
    cellphonedb_iterations: int = 0
    cellphonedb_threads: int = 2
    liana_n_perms: int = 100
    liana_jobs: int = 2
    nichenet_layer: str = "counts"
    nichenet_min_targets: int = 3
    nichenet_threads: int = 1
    crychic_benchmark_config: Path | None = None
    crychic_threads: int = 4


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return cast(dict[str, Any], value)


def _load_source_records(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = _json_object(manifest_path)
    if manifest.get("schema_version") != "crychic-multimethod-synthetic-v1":
        raise ValueError(
            f"unsupported synthetic manifest schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}"
        )
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("synthetic manifest records must be a non-empty list")
    records: list[dict[str, Any]] = []
    seen_scenarios: set[str] = set()
    seen_datasets: set[str] = set()
    source_root = manifest_path.parent.resolve()
    for ordinal, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError(f"synthetic record {ordinal} must be an object")
        record = cast(dict[str, Any], raw)
        scenario = str(record.get("scenario", ""))
        dataset_id = str(record.get("dataset_id", ""))
        relative = Path(str(record.get("path", "")))
        expected_sha256 = str(record.get("sha256", ""))
        if not scenario or not dataset_id or not relative.name or not expected_sha256:
            raise ValueError(
                f"synthetic record {ordinal} lacks scenario, dataset_id, path, "
                "or sha256"
            )
        input_path = (source_root / relative).resolve()
        if source_root not in input_path.parents:
            raise ValueError(
                f"synthetic record escapes its manifest directory: {relative}"
            )
        if scenario in seen_scenarios or dataset_id in seen_datasets:
            raise ValueError(
                f"synthetic scenario/dataset identifiers must be unique: {scenario}"
            )
        seen_scenarios.add(scenario)
        seen_datasets.add(dataset_id)
        if not input_path.is_file():
            raise FileNotFoundError(f"synthetic input is missing: {input_path}")
        observed_sha256 = sha256_file(input_path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"synthetic input checksum mismatch for {scenario}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        records.append(
            {
                **record,
                "scenario": scenario,
                "dataset_id": dataset_id,
                "input_path": input_path,
                "input_sha256": observed_sha256,
            }
        )
    return records


def _validate_resource(settings: RunnerSettings) -> dict[str, Any]:
    if not settings.database_root.is_dir():
        raise FileNotFoundError(
            f"benchmark database root is missing: {settings.database_root}"
        )
    if not settings.harmonized_resource.is_file():
        raise FileNotFoundError(
            f"harmonized resource is missing: {settings.harmonized_resource}"
        )
    if not settings.harmonized_manifest.is_file():
        raise FileNotFoundError(
            f"harmonized manifest is missing: {settings.harmonized_manifest}"
        )
    manifest = _json_object(settings.harmonized_manifest)
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("harmonized manifest payload must be an object")
    expected_filename = str(payload.get("filename", ""))
    expected_sha256 = str(payload.get("sha256", ""))
    if expected_filename != settings.harmonized_resource.name:
        raise ValueError(
            "harmonized manifest payload filename does not match the selected table"
        )
    observed_sha256 = sha256_file(settings.harmonized_resource)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "harmonized resource checksum mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    if settings.min_cells < 1:
        raise ValueError("min_cells must be positive")
    if not settings.nichenet_layer.strip():
        raise ValueError("nichenet_layer must be non-empty")
    for name, value in (
        ("cellchat_nboot", settings.cellchat_nboot),
        ("cellchat_threads", settings.cellchat_threads),
        ("cellphonedb_threads", settings.cellphonedb_threads),
        ("liana_n_perms", settings.liana_n_perms),
        ("liana_jobs", settings.liana_jobs),
        ("nichenet_min_targets", settings.nichenet_min_targets),
        ("nichenet_threads", settings.nichenet_threads),
        ("crychic_threads", settings.crychic_threads),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if settings.cellphonedb_iterations < 0 or (
        0 < settings.cellphonedb_iterations < 100
    ):
        raise ValueError("cellphonedb_iterations must be zero or at least 100")
    for name, executable in (
        ("cellphonedb_python", settings.cellphonedb_python),
        ("liana_python", settings.liana_python),
        ("crychic_python", settings.crychic_python),
    ):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"{name} is not executable: {executable}")
    if "/" in settings.cellchat_environment:
        cellchat_root = Path(settings.cellchat_environment)
        rscript = cellchat_root / "bin/Rscript"
        if not rscript.is_file() or not os.access(rscript, os.X_OK):
            raise FileNotFoundError(
                f"CellChat environment lacks an executable bin/Rscript: {cellchat_root}"
            )
    elif shutil.which("conda") is None:
        raise FileNotFoundError("conda is required to resolve the CellChat environment")
    return {
        "resource_id": manifest.get("resource_id"),
        "version": manifest.get("version"),
        "payload_sha256": observed_sha256,
        "manifest_sha256": sha256_file(settings.harmonized_manifest),
        "rows": payload.get("rows"),
    }


def _cellchat_environment_name(value: str) -> str:
    return Path(value).name if "/" in value else value


def _build_command(
    method: str,
    record: dict[str, Any],
    output_dir: Path,
    settings: RunnerSettings,
) -> tuple[list[str], int]:
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    input_path = cast(Path, record["input_path"])
    dataset_id = str(record["dataset_id"])
    seed = int(record["scenario_seed"])
    if method == "crychic":
        if settings.crychic_benchmark_config is None:
            raise ValueError("crychic_benchmark_config is required for CRYCHIC")
        threads = settings.crychic_threads
        return (
            [
                str(settings.crychic_python),
                "-m",
                "benchmarks.adapters.crychic.run_hcommon",
                str(settings.crychic_benchmark_config),
                dataset_id,
                str(output_dir),
                "--harmonized-resource",
                str(settings.harmonized_resource),
                "--harmonized-manifest",
                str(settings.harmonized_manifest),
                "--database-root",
                str(settings.database_root),
                "--communication-mode",
                "state",
                "--blas-threads",
                str(threads),
            ],
            threads,
        )
    common = [
        str(input_path),
        str(output_dir),
        "--dataset-id",
        dataset_id,
        "--context-key",
        "condition",
        "--min-cells",
        str(settings.min_cells),
    ]
    harmonized = [
        "--resource-mode",
        "H-common",
        "--harmonized-resource",
        str(settings.harmonized_resource),
        "--harmonized-manifest",
        str(settings.harmonized_manifest),
    ]
    if method == "cellchat":
        threads = settings.cellchat_threads
        command = [
            str(settings.crychic_python),
            "-m",
            "benchmarks.adapters.cellchat.run_by_sample",
            *common,
            "--database-root",
            str(settings.database_root),
            *harmonized,
            "--nboot",
            str(settings.cellchat_nboot),
            "--threads",
            str(threads),
            "--seed",
            str(seed),
            "--environment",
            _cellchat_environment_name(settings.cellchat_environment),
        ]
    elif method == "cellphonedb":
        threads = settings.cellphonedb_threads
        command = [
            str(settings.cellphonedb_python),
            "-m",
            "benchmarks.adapters.cellphonedb.run_by_sample",
            *common,
            "--database-root",
            str(settings.database_root),
            *harmonized,
            "--iterations",
            str(settings.cellphonedb_iterations),
            "--threads",
            str(threads),
            "--seed",
            str(seed),
        ]
    elif method == "liana":
        threads = settings.liana_jobs
        command = [
            str(settings.liana_python),
            "-m",
            "benchmarks.adapters.liana.run_by_sample",
            *common,
            *harmonized,
            "--n-perms",
            str(settings.liana_n_perms),
            "--n-jobs",
            str(threads),
            "--seed",
            str(seed),
        ]
    else:
        threads = settings.nichenet_threads
        command = [
            str(settings.crychic_python),
            "-m",
            "benchmarks.adapters.nichenet.run_by_sample",
            *common,
            "--database-root",
            str(settings.database_root),
            "--layer",
            settings.nichenet_layer,
            "--min-targets",
            str(settings.nichenet_min_targets),
            "--threads",
            str(threads),
        ]
    return command, threads


def _process_tree_rss(process_id: int) -> int:
    import psutil  # type: ignore[import-untyped]

    try:
        root = psutil.Process(process_id)
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        return 0
    total = 0
    for process in processes:
        try:
            total += int(process.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return total


def _execute_process(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    threads: int,
) -> ProcessResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
            "VECLIB_MAXIMUM_THREADS": str(threads),
        }
    )
    started = time.perf_counter()
    peak_rss = 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                peak_rss = max(peak_rss, _process_tree_rss(process.pid))
                time.sleep(0.2)
            peak_rss = max(peak_rss, _process_tree_rss(process.pid))
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
            raise
    return ProcessResult(
        exit_code=int(process.returncode),
        wall_time_seconds=time.perf_counter() - started,
        peak_rss_mb=peak_rss / (1024**2),
    )


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _assess_adapter_output(
    output_dir: Path, process_result: ProcessResult
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "failed",
        "reason_code": "adapter_process_failed",
        "exit_code": process_result.exit_code,
        "adapter_manifest_status": None,
        "adapter_manifest_sha256": None,
        "table_sha256": None,
        "rows": None,
        "ok_rows": None,
        "failed_samples": None,
        "status_counts": None,
    }
    if process_result.exit_code != 0:
        return result
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        result["reason_code"] = "adapter_manifest_missing"
        return result
    try:
        manifest = _json_object(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["reason_code"] = "adapter_manifest_invalid"
        result["validation_error"] = str(exc)
        return result
    result["adapter_manifest_status"] = manifest.get("status")
    result["adapter_manifest_sha256"] = sha256_file(manifest_path)
    parameters = manifest.get("parameters")
    if isinstance(parameters, dict):
        result["resolved_expression_transform"] = parameters.get(
            "resolved_expression_transform"
        )
    if manifest.get("status") != "complete":
        result["reason_code"] = "adapter_manifest_failed"
        return result
    output = manifest.get("output")
    if not isinstance(output, dict):
        result["reason_code"] = "adapter_output_metadata_missing"
        return result
    table_path = output_dir / str(output.get("table", ""))
    if not table_path.is_file():
        result["reason_code"] = "adapter_table_missing"
        return result
    observed_table_sha256 = sha256_file(table_path)
    result["table_sha256"] = observed_table_sha256
    if observed_table_sha256 != str(output.get("sha256", "")):
        result["reason_code"] = "adapter_table_checksum_mismatch"
        return result
    try:
        table = pd.read_parquet(table_path, columns=["status"])
    except (OSError, ValueError) as exc:
        result["reason_code"] = "adapter_table_invalid"
        result["validation_error"] = str(exc)
        return result
    counts = table["status"].astype(str).value_counts().sort_index()
    result["rows"] = len(table)
    result["ok_rows"] = int(counts.get("ok", 0))
    result["status_counts"] = {str(key): int(value) for key, value in counts.items()}
    sample_failures = manifest.get("sample_failures")
    if isinstance(sample_failures, dict):
        result["failed_samples"] = len(sample_failures)
        if sample_failures:
            result["reason_code"] = "adapter_sample_failures"
            return result
    else:
        result["failed_samples"] = 0
    if len(table) != int(output.get("rows", -1)):
        result["reason_code"] = "adapter_row_count_mismatch"
        return result
    result["status"] = "complete"
    result["reason_code"] = None
    return result


def _execution_path(output_dir: Path) -> Path:
    return output_dir / "runner_execution.json"


def _resumable_record(output_dir: Path) -> dict[str, Any] | None:
    execution_path = _execution_path(output_dir)
    if not execution_path.is_file():
        return None
    try:
        record = _json_object(execution_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if record.get("schema_version") != EXECUTION_SCHEMA:
        return None
    if record.get("status") != "complete" or record.get("exit_code") != 0:
        return None
    table_path = output_dir / "interactions_long.parquet"
    manifest_path = output_dir / "manifest.json"
    if not table_path.is_file() or not manifest_path.is_file():
        return None
    if record.get("table_sha256") != sha256_file(table_path):
        return None
    if record.get("adapter_manifest_sha256") != sha256_file(manifest_path):
        return None
    record["resumed"] = True
    record["execution_action"] = "resumed"
    return record


def _refresh_cellchat_seed_provenance(
    output_dir: Path,
    execution: dict[str, Any],
) -> dict[str, Any]:
    """Backfill deterministic seed provenance without changing score artifacts."""
    if (
        execution.get("method_id") != "cellchat"
        or execution.get("status") != "complete"
    ):
        return execution
    manifest_path = output_dir / "manifest.json"
    table_path = output_dir / "interactions_long.parquet"
    runner_path = _execution_path(output_dir)
    if (
        not manifest_path.is_file()
        or not table_path.is_file()
        or not runner_path.is_file()
    ):
        return execution
    manifest = _json_object(manifest_path)
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"CellChat manifest lacks parameters: {manifest_path}")
    requested = parameters.get("requested_seed", parameters.get("seed"))
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise ValueError(f"CellChat manifest lacks an integer seed: {manifest_path}")
    sample_ids = sorted(
        pd.read_parquet(table_path, columns=["sample_id"])["sample_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    from benchmarks.adapters.cellchat.run_by_sample import (
        R_SEED_MODULUS,
        _seed_provenance,
    )

    expected = _seed_provenance(sample_ids, requested)
    existing = parameters.get("effective_sample_seeds")
    if existing is not None:
        if existing != expected["effective_sample_seeds"]:
            raise ValueError(
                "CellChat effective seed provenance disagrees with output: "
                f"{output_dir}"
            )
        return execution
    if any(
        not 1 <= requested + ordinal <= R_SEED_MODULUS
        for ordinal in range(len(sample_ids))
    ):
        raise ValueError(
            "overflowed CellChat runs must be rerun; provenance cannot be "
            f"backfilled for {output_dir}"
        )
    parameters.update(expected)
    amendments = manifest.setdefault("provenance_amendments", [])
    if not isinstance(amendments, list):
        raise ValueError(f"invalid provenance_amendments in {manifest_path}")
    amendments.append(
        {
            "type": "deterministic_seed_provenance_backfill",
            "reason": "adapter contract upgrade for an already-valid R seed run",
            "score_table_sha256_unchanged": sha256_file(table_path),
        }
    )
    write_json(manifest_path, manifest)
    updated = dict(execution)
    updated["adapter_manifest_sha256"] = sha256_file(manifest_path)
    updated["seed_provenance_backfilled"] = True
    runner_execution = _json_object(runner_path)
    runner_execution.update(
        {
            "adapter_manifest_sha256": updated["adapter_manifest_sha256"],
            "seed_provenance_backfilled": True,
        }
    )
    write_json(runner_path, runner_execution)
    return updated


def _settings_payload(settings: RunnerSettings) -> dict[str, Any]:
    return {
        "min_cells": settings.min_cells,
        "cellchat": {
            "environment": settings.cellchat_environment,
            "nboot": settings.cellchat_nboot,
            "threads": settings.cellchat_threads,
        },
        "cellphonedb": {
            "python": str(settings.cellphonedb_python),
            "iterations": settings.cellphonedb_iterations,
            "threads": settings.cellphonedb_threads,
        },
        "liana": {
            "python": str(settings.liana_python),
            "n_perms": settings.liana_n_perms,
            "n_jobs": settings.liana_jobs,
        },
        "nichenet": {
            "python": str(settings.crychic_python),
            "layer": settings.nichenet_layer,
            "min_targets": settings.nichenet_min_targets,
            "threads": settings.nichenet_threads,
            "sender_scope": "source_agnostic_track_b",
        },
        "crychic": {
            "python": str(settings.crychic_python),
            "benchmark_config": (
                None
                if settings.crychic_benchmark_config is None
                else str(settings.crychic_benchmark_config)
            ),
            "benchmark_config_sha256": (
                None
                if settings.crychic_benchmark_config is None
                else sha256_file(settings.crychic_benchmark_config)
            ),
            "communication_mode": "state",
            "threads": settings.crychic_threads,
        },
        "rss_measurement": "sum of root process and descendants, sampled every 0.2 s",
    }


def _write_truth_tables(
    output_dir: Path,
    *,
    source_records: list[dict[str, Any]],
    executions: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    track_b_rows = []
    for record in source_records:
        track_b_rows.append(
            {
                "dataset": str(record["dataset_id"]),
                "scenario": str(record["scenario"]),
                "contrast": "stim_vs_ctrl",
                "expected_receiver_response": bool(
                    record.get("expected_receiver_response", False)
                ),
                "truth_scope": "simulation_scenario_level",
                "metric_scope": "ligand_target_program",
                "lr_edge_truth_available": False,
                "reason_code": (
                    "track_b_is_ligand_target_program_not_lr_edge_truth"
                ),
            }
        )
    track_b_path = output_dir / "track_b_scenario_truth.tsv"
    pd.DataFrame(track_b_rows).to_csv(track_b_path, sep="\t", index=False)

    track_a_methods = {"crychic", "cellchat", "cellphonedb", "liana"}
    edge_frames: list[pd.DataFrame] = []
    for record in source_records:
        scenario = str(record["scenario"])
        dataset_id = str(record["dataset_id"])
        candidates = [
            method
            for method in track_a_methods
            if executions.get((scenario, method), {}).get("status") == "complete"
        ]
        if not candidates:
            continue
        universes: set[str] = set()
        edge_tables: list[pd.DataFrame] = []
        columns = [
            "universe_id",
            "sender",
            "receiver",
            "interaction_id",
            "ligand",
            "receptor",
        ]
        for method in sorted(candidates):
            table_path = output_dir / scenario / method / "interactions_long.parquet"
            table = pd.read_parquet(table_path, columns=columns).drop_duplicates()
            method_universes = set(table["universe_id"].astype(str))
            if len(method_universes) != 1:
                raise ValueError(
                    f"Track-A {scenario}/{method} has multiple frozen universes"
                )
            universes.update(method_universes)
            edge_tables.append(table)
        if len(universes) != 1:
            raise ValueError(
                f"Track-A methods disagree on the frozen universe for {scenario}"
            )
        reference = edge_tables[0].sort_values(columns, ignore_index=True)
        reference_keys = set(
            reference[columns].itertuples(index=False, name=None)
        )
        for table in edge_tables[1:]:
            observed_keys = set(table[columns].itertuples(index=False, name=None))
            if observed_keys != reference_keys:
                raise ValueError(
                    f"Track-A methods disagree on frozen edge keys for {scenario}"
                )
        resource_edges = reference[
            ["interaction_id", "ligand", "receptor"]
        ].drop_duplicates()
        if len(resource_edges) != 5 or len(reference) != 45:
            raise ValueError(
                f"synthetic Track-A {scenario} must contain 5 LR x 9 cell pairs"
            )
        positive = (
            reference["sender"].astype(str).eq("Sender")
            & reference["receiver"].astype(str).eq("Receiver")
            & reference["ligand"].astype(str).eq("CXCL10")
            & reference["receptor"].astype(str).eq("CXCR3")
            & (scenario == "active")
        )
        truth = reference.copy()
        truth.insert(0, "dataset", dataset_id)
        truth.insert(1, "contrast", "stim_vs_ctrl")
        truth["is_positive"] = positive.astype(int)
        truth["truth_scope"] = "simulation"
        truth["truth_status"] = (
            "estimable" if scenario == "active" else "not_estimable"
        )
        truth["reason_code"] = (
            None
            if scenario == "active"
            else "truth_has_single_class_no_positive_integrated_edge"
        )
        edge_frames.append(truth)

    outputs: dict[str, Any] = {
        "track_b_scenario_truth": {
            "filename": track_b_path.name,
            "sha256": sha256_file(track_b_path),
            "rows": len(track_b_rows),
            "lr_edge_truth": False,
        }
    }
    if edge_frames:
        edge_truth = pd.concat(edge_frames, ignore_index=True)
        edge_path = output_dir / "track_a_edge_truth.tsv"
        edge_truth.to_csv(edge_path, sep="\t", index=False)
        outputs["track_a_edge_truth"] = {
            "filename": edge_path.name,
            "sha256": sha256_file(edge_path),
            "rows": len(edge_truth),
            "truth_scope": "simulation",
            "single_class_reason": (
                "truth_has_single_class_no_positive_integrated_edge"
            ),
        }
    return outputs


def _write_suite(
    output_dir: Path,
    *,
    manifest_path: Path,
    resource_metadata: dict[str, Any],
    settings: RunnerSettings,
    source_records: list[dict[str, Any]],
    executions: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    all_keys = {
        (str(record["scenario"]), method)
        for method in METHODS
        for record in source_records
    }
    complete_keys = {
        key
        for key, execution in executions.items()
        if execution.get("status") == "complete"
    }
    failed = sum(
        execution.get("status") == "failed" for execution in executions.values()
    )
    if all_keys.issubset(complete_keys):
        status = "complete"
    elif failed:
        status = "partial_failed"
    else:
        status = "partial"
    ordered = sorted(
        executions.values(),
        key=lambda row: (
            METHODS.index(str(row["method_id"])),
            str(row["scenario"]),
        ),
    )
    truth_outputs = _write_truth_tables(
        output_dir,
        source_records=source_records,
        executions=executions,
    )
    suite: dict[str, Any] = {
        "schema_version": SUITE_SCHEMA,
        "status": status,
        "updated_at": _utc_now(),
        "source_manifest": {
            "filename": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "records": len(source_records),
        },
        "harmonized_track_a_resource": resource_metadata,
        "methods": list(METHODS),
        "method_tracks": {
            method: {"analysis_track": track, "resource_mode": mode}
            for method, (track, mode) in METHOD_TRACKS.items()
        },
        "parameters": _settings_payload(settings),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "expected_runs": len(all_keys),
        "recorded_runs": len(executions),
        "complete_runs": len(complete_keys),
        "failed_runs": failed,
        "truth_outputs": truth_outputs,
        "executions": ordered,
    }
    write_json(output_dir / "suite_manifest.json", suite)
    if ordered:
        pd.DataFrame(ordered).to_csv(output_dir / "runs.tsv", sep="\t", index=False)
    return suite


def run_multimethod_controls(
    manifest_path: Path,
    output_dir: Path,
    *,
    settings: RunnerSettings,
    methods: tuple[str, ...] = METHODS,
    scenarios: tuple[str, ...] | None = None,
    resume: bool = True,
    overwrite: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Execute selected method/scenario pairs and persist resumable provenance."""
    unknown_methods = set(methods).difference(METHODS)
    if unknown_methods:
        raise ValueError(f"unsupported methods: {sorted(unknown_methods)}")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be non-empty and unique")
    if "crychic" in methods:
        benchmark_config = settings.crychic_benchmark_config
        if benchmark_config is None or not benchmark_config.is_file():
            raise FileNotFoundError(
                "a frozen crychic_benchmark_config is required for CRYCHIC runs"
            )
    source_records = _load_source_records(manifest_path)
    resource_metadata = _validate_resource(settings)
    available_scenarios = {str(record["scenario"]) for record in source_records}
    selected_scenarios = available_scenarios if scenarios is None else set(scenarios)
    unknown_scenarios = selected_scenarios.difference(available_scenarios)
    if unknown_scenarios:
        raise ValueError(f"unknown synthetic scenarios: {sorted(unknown_scenarios)}")
    if not selected_scenarios:
        raise ValueError("scenarios must select at least one synthetic dataset")

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not resume
        and not overwrite
    ):
        raise FileExistsError(
            "output directory is not empty and neither resume nor selected-run "
            f"overwrite is enabled: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    previous_path = output_dir / "suite_manifest.json"
    executions: dict[tuple[str, str], dict[str, Any]] = {}
    if previous_path.is_file():
        previous = _json_object(previous_path)
        raw_executions = previous.get("executions", [])
        if isinstance(raw_executions, list):
            for raw in raw_executions:
                if isinstance(raw, dict) and "scenario" in raw and "method_id" in raw:
                    executions[(str(raw["scenario"]), str(raw["method_id"]))] = cast(
                        dict[str, Any], raw
                    )
    for key, loaded_execution in list(executions.items()):
        scenario, method = key
        if method == "cellchat":
            executions[key] = _refresh_cellchat_seed_provenance(
                output_dir / scenario / method,
                loaded_execution,
            )

    selected_records = [
        record
        for record in source_records
        if str(record["scenario"]) in selected_scenarios
    ]
    selected_records.sort(key=lambda record: str(record["scenario"]))
    if overwrite:
        for method in methods:
            analysis_track, resource_mode = METHOD_TRACKS[method]
            for record in selected_records:
                scenario = str(record["scenario"])
                executions[(scenario, method)] = {
                    "schema_version": EXECUTION_SCHEMA,
                    "scenario": scenario,
                    "dataset_id": str(record["dataset_id"]),
                    "method_id": method,
                    "analysis_track": analysis_track,
                    "resource_mode": resource_mode,
                    "status": "pending",
                    "reason_code": "selected_overwrite_pending",
                    "execution_action": "pending_overwrite",
                }
        _write_suite(
            output_dir,
            manifest_path=manifest_path,
            resource_metadata=resource_metadata,
            settings=settings,
            source_records=source_records,
            executions=executions,
        )
    for method in methods:
        for record in selected_records:
            scenario = str(record["scenario"])
            run_output = output_dir / scenario / method
            if resume and not overwrite:
                resumed = _resumable_record(run_output)
                if resumed is not None:
                    executions[(scenario, method)] = resumed
                    _write_suite(
                        output_dir,
                        manifest_path=manifest_path,
                        resource_metadata=resource_metadata,
                        settings=settings,
                        source_records=source_records,
                        executions=executions,
                    )
                    continue
            command, threads = _build_command(method, record, run_output, settings)
            log_root = output_dir / "logs" / scenario
            stdout_path = log_root / f"{method}.stdout.log"
            stderr_path = log_root / f"{method}.stderr.log"
            started_at = _utc_now()
            analysis_track, resource_mode = METHOD_TRACKS[method]
            executions[(scenario, method)] = {
                "schema_version": EXECUTION_SCHEMA,
                "scenario": scenario,
                "dataset_id": str(record["dataset_id"]),
                "method_id": method,
                "analysis_track": analysis_track,
                "resource_mode": resource_mode,
                "started_at": started_at,
                "status": "running",
                "reason_code": "execution_in_progress",
                "execution_action": "executing",
            }
            _write_suite(
                output_dir,
                manifest_path=manifest_path,
                resource_metadata=resource_metadata,
                settings=settings,
                source_records=source_records,
                executions=executions,
            )
            if run_output.exists():
                shutil.rmtree(run_output)
            process_result = _execute_process(
                command,
                cwd=repo_root,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                threads=threads,
            )
            assessment = _assess_adapter_output(run_output, process_result)
            execution: dict[str, Any] = {
                "schema_version": EXECUTION_SCHEMA,
                "scenario": scenario,
                "dataset_id": str(record["dataset_id"]),
                "input_sha256": str(record["input_sha256"]),
                "method_id": method,
                "analysis_track": analysis_track,
                "resource_mode": resource_mode,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "command": command,
                "threads": threads,
                "wall_time_seconds": process_result.wall_time_seconds,
                "peak_rss_mb": process_result.peak_rss_mb,
                "peak_rss_scope": "root_process_plus_descendants_sampled_0.2s",
                "output_bytes": _directory_bytes(run_output),
                "stdout_log": str(stdout_path.relative_to(output_dir)),
                "stderr_log": str(stderr_path.relative_to(output_dir)),
                "log_bytes": _directory_bytes(log_root),
                "resumed": False,
                "execution_action": "executed",
                **assessment,
            }
            run_output.mkdir(parents=True, exist_ok=True)
            write_json(_execution_path(run_output), execution)
            executions[(scenario, method)] = execution
            _write_suite(
                output_dir,
                manifest_path=manifest_path,
                resource_metadata=resource_metadata,
                settings=settings,
                source_records=source_records,
                executions=executions,
            )
            if execution["status"] == "failed" and fail_fast:
                return _write_suite(
                    output_dir,
                    manifest_path=manifest_path,
                    resource_metadata=resource_metadata,
                    settings=settings,
                    source_records=source_records,
                    executions=executions,
                )
    return _write_suite(
        output_dir,
        manifest_path=manifest_path,
        resource_metadata=resource_metadata,
        settings=settings,
        source_records=source_records,
        executions=executions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--database-root", required=True, type=Path)
    parser.add_argument("--harmonized-resource", required=True, type=Path)
    parser.add_argument("--harmonized-manifest", required=True, type=Path)
    parser.add_argument("--cellchat-environment", default="r_cellchat")
    parser.add_argument("--cellphonedb-python", required=True, type=Path)
    parser.add_argument("--liana-python", required=True, type=Path)
    parser.add_argument("--crychic-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--crychic-benchmark-config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "configs/synthetic_multimethod_nocap_v02.json"
        ),
    )
    parser.add_argument("--crychic-threads", type=int, default=4)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--cellchat-nboot", type=int, default=100)
    parser.add_argument("--cellchat-threads", type=int, default=1)
    parser.add_argument("--cellphonedb-iterations", type=int, default=0)
    parser.add_argument("--cellphonedb-threads", type=int, default=2)
    parser.add_argument("--liana-n-perms", type=int, default=100)
    parser.add_argument("--liana-jobs", type=int, default=2)
    parser.add_argument("--nichenet-min-targets", type=int, default=3)
    parser.add_argument("--nichenet-layer", default="counts")
    parser.add_argument("--nichenet-threads", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    settings = RunnerSettings(
        database_root=args.database_root,
        harmonized_resource=args.harmonized_resource,
        harmonized_manifest=args.harmonized_manifest,
        cellchat_environment=args.cellchat_environment,
        cellphonedb_python=args.cellphonedb_python,
        liana_python=args.liana_python,
        crychic_python=args.crychic_python,
        crychic_benchmark_config=args.crychic_benchmark_config,
        crychic_threads=args.crychic_threads,
        min_cells=args.min_cells,
        cellchat_nboot=args.cellchat_nboot,
        cellchat_threads=args.cellchat_threads,
        cellphonedb_iterations=args.cellphonedb_iterations,
        cellphonedb_threads=args.cellphonedb_threads,
        liana_n_perms=args.liana_n_perms,
        liana_jobs=args.liana_jobs,
        nichenet_layer=args.nichenet_layer,
        nichenet_min_targets=args.nichenet_min_targets,
        nichenet_threads=args.nichenet_threads,
    )
    suite = run_multimethod_controls(
        args.manifest,
        args.output_dir,
        settings=settings,
        methods=tuple(args.method) if args.method else METHODS,
        scenarios=tuple(args.scenario) if args.scenario else None,
        resume=not args.no_resume,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(suite, indent=2, sort_keys=True))
    selected_methods = set(args.method or METHODS)
    selected_scenarios = set(args.scenario or ())
    selected_failures = [
        execution
        for execution in suite["executions"]
        if execution["method_id"] in selected_methods
        and (
            not selected_scenarios
            or execution["scenario"] in selected_scenarios
        )
        and execution["status"] == "failed"
    ]
    if selected_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
