"""Derive E5 topology swaps from a completed v7 effect campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import platform
import shutil
import sys
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from benchmarks.adapters.common import (
    git_metadata,
    json_safe,
    package_versions,
    sha256_file,
    write_json,
)
from benchmarks.simulation.run_v7_campaign import (
    _auto_worker_count,
    _DatasetLogger,
    _eta_seconds,
    _PeakRSSMonitor,
    _system_memory_fraction,
)
from benchmarks.simulation.v7_dgp import build_v7_dgp_resource
from benchmarks.simulation.v7_hypergraph_swaps import run_v7_e5_hypergraph_swap

SCHEMA_VERSION = "crychic-suggest-next2-v7-e5-derived-campaign-v1"
DATASET_SCHEMA_VERSION = "crychic-suggest-next2-v7-e5-derived-dataset-v1"
SUPPORTED_SOURCE_SCHEMAS = {
    "crychic-suggest-next2-v7-campaign-v2",
    "crychic-suggest-next2-v7-campaign-v3",
    "crychic-suggest-next2-v7-campaign-v4",
    "crychic-suggest-next2-v7-campaign-v5",
}
RUN_COLUMNS = (
    "dataset_id",
    "dgp_family",
    "design_kind",
    "seed",
    "status",
    "reason_code",
    "elapsed_seconds",
    "peak_process_tree_rss_bytes",
    "result_directory",
    "manifest_sha256",
)
OUTPUT_TABLES = (
    "e5_edge_estimates",
    "e5_metrics",
    "e5_fit_diagnostics",
    "e5_topology_diagnostics",
)
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_parquet(table: pd.DataFrame, path: Path) -> dict[str, object]:
    table.to_parquet(path, index=False, compression="zstd")
    return {
        "filename": path.name,
        "rows": len(table),
        "columns": list(map(str, table.columns)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _source_campaign(source: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = source / "campaign_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") not in SUPPORTED_SOURCE_SCHEMAS:
        raise ValueError("source campaign schema is not E5-derivable")
    if manifest.get("status") != "completed" or manifest.get("failed_datasets") != 0:
        raise ValueError("source campaign must be complete and failure-free")
    run_record = manifest.get("runs")
    if not isinstance(run_record, Mapping):
        raise ValueError("source campaign lacks a runs record")
    runs_path = source / str(run_record.get("filename"))
    if sha256_file(runs_path) != run_record.get("sha256"):
        raise ValueError("source campaign runs checksum differs")
    runs = pd.read_csv(runs_path, sep="\t")
    if (
        len(runs) != int(manifest["planned_datasets"])
        or not runs["status"].eq("completed").all()
        or runs["dataset_id"].duplicated().any()
    ):
        raise ValueError("source runs do not cover the planned dataset universe")
    return manifest, runs


def _source_dataset(record: Mapping[str, object]) -> dict[str, object]:
    directory = Path(str(record["result_directory"])).resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source dataset manifest is missing: {directory}")
    expected_manifest = record.get("manifest_sha256")
    if not isinstance(expected_manifest, str) or (
        sha256_file(manifest_path) != expected_manifest
    ):
        raise ValueError("source dataset manifest checksum differs")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("source dataset is not complete")
    plan = manifest.get("dataset_plan")
    outputs = manifest.get("outputs")
    if not isinstance(plan, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("source dataset lacks plan or output records")
    if str(plan.get("dataset_id")) != str(record["dataset_id"]):
        raise ValueError("source dataset ID differs between runs and manifest")
    paths: dict[str, Path] = {}
    for name in ("effects", "truth"):
        output = outputs.get(name)
        if not isinstance(output, Mapping):
            raise ValueError(f"source dataset lacks {name} output")
        path = directory / str(output.get("filename"))
        if not path.is_file() or sha256_file(path) != output.get("sha256"):
            raise ValueError(f"source dataset {name} checksum differs")
        paths[name] = path
    return {
        "dataset_id": str(plan["dataset_id"]),
        "dgp_family": str(plan["dgp_family"]),
        "design_kind": str(plan["design_kind"]),
        "seed": int(plan["seed"]),
        "source_directory": str(directory),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": expected_manifest,
        "effects": str(paths["effects"]),
        "effects_sha256": str(outputs["effects"]["sha256"]),
        "truth": str(paths["truth"]),
        "truth_sha256": str(outputs["truth"]["sha256"]),
    }


def _validate_completed(path: Path) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("status") != "completed"
    ):
        return None
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(OUTPUT_TABLES):
        return None
    for output in outputs.values():
        if not isinstance(output, Mapping):
            return None
        candidate = path / str(output.get("filename"))
        if not candidate.is_file() or sha256_file(candidate) != output.get("sha256"):
            return None
    return manifest


def _record_from_manifest(
    path: Path, manifest: Mapping[str, object]
) -> dict[str, object]:
    source = manifest["source"]
    if not isinstance(source, Mapping):
        raise ValueError("derived dataset manifest lacks source")
    return {
        "dataset_id": source["dataset_id"],
        "dgp_family": source["dgp_family"],
        "design_kind": source["design_kind"],
        "seed": source["seed"],
        "status": manifest["status"],
        "reason_code": None,
        "elapsed_seconds": manifest["elapsed_seconds"],
        "peak_process_tree_rss_bytes": manifest["peak_process_tree_rss_bytes"],
        "result_directory": str(path),
        "manifest_sha256": sha256_file(path / "manifest.json"),
    }


def run_e5_source_dataset(
    source: Mapping[str, object],
    *,
    output_root: Path,
    maximum_memory_fraction: float,
    overwrite: bool = False,
) -> dict[str, object]:
    """Derive and atomically persist E5 for one authenticated source dataset."""

    dataset_id = str(source["dataset_id"])
    final = output_root / "datasets" / dataset_id
    existing = _validate_completed(final)
    if existing is not None and not overwrite:
        if existing.get("source", {}).get("source_manifest_sha256") != source.get(
            "source_manifest_sha256"
        ):
            raise ValueError("resumed E5 result references a different source dataset")
        return _record_from_manifest(final, existing)
    if final.exists() and not overwrite:
        raise FileExistsError(f"incomplete derived E5 output exists: {final}")
    temporary = (
        output_root
        / "datasets"
        / (f".{dataset_id}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    )
    temporary.mkdir(parents=True, exist_ok=False)
    logger = _DatasetLogger(temporary / "run.jsonl", dataset_id=dataset_id)
    monitor = _PeakRSSMonitor()
    started = time.monotonic()
    try:
        if _system_memory_fraction() >= maximum_memory_fraction:
            raise MemoryError("system memory is at the derived campaign limit")
        with monitor, threadpool_limits(limits=1):
            with logger.stage("read_authenticated_source"):
                if (
                    sha256_file(Path(str(source["effects"])))
                    != source["effects_sha256"]
                ):
                    raise ValueError("effects changed after source validation")
                if sha256_file(Path(str(source["truth"]))) != source["truth_sha256"]:
                    raise ValueError("truth changed after source validation")
                effects = pd.read_parquet(Path(str(source["effects"])))
                truth = pd.read_parquet(Path(str(source["truth"])))
                resource = build_v7_dgp_resource(str(source["dgp_family"]))
            with logger.stage("e5_hypergraph_topology_swaps"):
                result = run_v7_e5_hypergraph_swap(
                    effects,
                    resource=resource,
                    truth=truth,
                    dataset_id=dataset_id,
                    dgp_family=str(source["dgp_family"]),
                    design_kind=str(source["design_kind"]),
                    root_seed=int(source["seed"]),
                )
            with logger.stage("persist_outputs"):
                tables = {
                    "e5_edge_estimates": result.edge_estimates,
                    "e5_metrics": result.metrics,
                    "e5_fit_diagnostics": result.fit_diagnostics,
                    "e5_topology_diagnostics": result.topology_diagnostics,
                }
                outputs = {
                    name: _write_parquet(table, temporary / f"{name}.parquet")
                    for name, table in tables.items()
                }
        elapsed = time.monotonic() - started
        if monitor.maximum_system_memory_fraction >= maximum_memory_fraction:
            raise MemoryError("derived E5 execution reached the memory limit")
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "status": "completed",
            "source": dict(source),
            "elapsed_seconds": elapsed,
            "peak_process_tree_rss_bytes": monitor.peak_rss_bytes,
            "maximum_system_memory_fraction": monitor.maximum_system_memory_fraction,
            "stage_timings": logger.stage_timings,
            "outputs": outputs,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": package_versions(
                    (
                        "CRYCHIC",
                        "numpy",
                        "pandas",
                        "pyarrow",
                        "scikit-learn",
                        "scipy",
                    )
                ),
                "git": git_metadata(Path(__file__).resolve().parents[2]),
            },
            "formal_inference_allowed": False,
            "formal_inference_reason": (
                "E5_smoke_uses_conditional_SE_until_full_pipeline_resampling"
            ),
        }
        write_json(temporary / "manifest.json", manifest)
        if final.exists():
            shutil.rmtree(final)
        temporary.replace(final)
        return _record_from_manifest(final, manifest)
    except BaseException as error:
        logger.event(
            "failed",
            stage="dataset",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _worker(payload: Mapping[str, object]) -> dict[str, object]:
    for name, value in _THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    return run_e5_source_dataset(
        payload["source"],  # type: ignore[arg-type]
        output_root=Path(str(payload["output_root"])),
        maximum_memory_fraction=float(payload["maximum_memory_fraction"]),
        overwrite=bool(payload["overwrite"]),
    )


def _write_state(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
    *,
    source_root: Path,
    source_manifest: Mapping[str, object],
    source_manifest_sha256: str,
    planned_datasets: int,
    effective_jobs: int,
    maximum_memory_fraction: float,
    started_utc: str,
    status: str,
) -> None:
    runs = pd.DataFrame.from_records(records, columns=RUN_COLUMNS)
    runs_path = output_root / "runs.tsv"
    runs.to_csv(runs_path, sep="\t", index=False)
    write_json(
        output_root / "campaign_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "started_utc": started_utc,
            "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "planned_datasets": planned_datasets,
            "completed_datasets": int(runs["status"].eq("completed").sum()),
            "failed_datasets": int(runs["status"].eq("failed").sum()),
            "effective_jobs": effective_jobs,
            "maximum_memory_fraction": maximum_memory_fraction,
            "source_campaign": {
                "directory": str(source_root),
                "manifest_sha256": source_manifest_sha256,
                "schema_version": source_manifest["schema_version"],
                "protocol_digest": source_manifest["protocol"]["protocol_digest"],
            },
            "runs": {
                "filename": runs_path.name,
                "sha256": sha256_file(runs_path),
                "rows": len(runs),
            },
            "formal_inference_allowed": False,
        },
    )


def _aggregate(output_root: Path, records: Sequence[Mapping[str, object]]) -> None:
    completed = [
        Path(str(record["result_directory"]))
        for record in records
        if record["status"] == "completed"
    ]
    for name in ("e5_metrics", "e5_fit_diagnostics", "e5_topology_diagnostics"):
        parts = [pd.read_parquet(path / f"{name}.parquet") for path in completed]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(
                output_root / f"all_{name}.parquet",
                index=False,
                compression="zstd",
            )
    metrics_path = output_root / "all_e5_metrics.parquet"
    if metrics_path.is_file():
        metrics = pd.read_parquet(metrics_path)
        summary = (
            metrics.loc[metrics["status"].eq("observed")]
            .groupby(
                [
                    "dgp_family",
                    "design_kind",
                    "score_view",
                    "metric",
                ],
                observed=True,
                sort=True,
            )["value"]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
        )
        summary.to_csv(output_root / "e5_metric_summary.tsv", sep="\t", index=False)


def run_derived_campaign(
    source_campaign: Path,
    output_root: Path,
    *,
    jobs: int = 0,
    maximum_memory_fraction: float = 0.8,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run resumable E5 derivation without repeating upstream cross-fitting."""

    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 0:
        raise ValueError("jobs must be zero (auto) or a positive integer")
    if not 0.0 < maximum_memory_fraction <= 0.8:
        raise ValueError("maximum_memory_fraction must lie in (0, 0.8]")
    source_root = source_campaign.resolve()
    output_root = output_root.resolve()
    source_manifest_path = source_root / "campaign_manifest.json"
    source_manifest, source_runs = _source_campaign(source_root)
    sources = [_source_dataset(row._asdict()) for row in source_runs.itertuples()]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "datasets").mkdir(exist_ok=True)
    started_utc = pd.Timestamp.now(tz="UTC").isoformat()
    records: list[dict[str, object]] = []
    durations: list[float] = []

    first = run_e5_source_dataset(
        sources[0],
        output_root=output_root,
        maximum_memory_fraction=maximum_memory_fraction,
        overwrite=overwrite,
    )
    records.append(first)
    durations.append(float(first["elapsed_seconds"]))
    effective_jobs = _auto_worker_count(
        pilot_peak_rss_bytes=max(1, int(first["peak_process_tree_rss_bytes"])),
        maximum_memory_fraction=maximum_memory_fraction,
        requested_jobs=jobs,
    )
    print(
        json.dumps(
            {
                "event": "pilot_completed",
                "dataset_id": first["dataset_id"],
                "elapsed_seconds": first["elapsed_seconds"],
                "peak_rss_bytes": first["peak_process_tree_rss_bytes"],
                "effective_jobs": effective_jobs,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    state_arguments = {
        "source_root": source_root,
        "source_manifest": source_manifest,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "planned_datasets": len(sources),
        "effective_jobs": effective_jobs,
        "maximum_memory_fraction": maximum_memory_fraction,
        "started_utc": started_utc,
    }
    _write_state(output_root, records, status="running", **state_arguments)
    remaining = sources[1:]
    if remaining:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_jobs,
            mp_context=context,
        ) as executor:
            pending = {
                executor.submit(
                    _worker,
                    {
                        "source": source,
                        "output_root": str(output_root),
                        "maximum_memory_fraction": maximum_memory_fraction,
                        "overwrite": overwrite,
                    },
                ): source
                for source in remaining
            }
            for future in concurrent.futures.as_completed(pending):
                source = pending[future]
                try:
                    record = future.result()
                except BaseException as error:
                    record = {
                        "dataset_id": source["dataset_id"],
                        "dgp_family": source["dgp_family"],
                        "design_kind": source["design_kind"],
                        "seed": source["seed"],
                        "status": "failed",
                        "reason_code": f"{type(error).__name__}:{error}",
                        "elapsed_seconds": 0.0,
                        "peak_process_tree_rss_bytes": 0,
                        "result_directory": "",
                        "manifest_sha256": None,
                    }
                records.append(record)
                durations.append(float(record["elapsed_seconds"]))
                eta = _eta_seconds(
                    durations, len(sources) - len(records), effective_jobs
                )
                print(
                    json.dumps(
                        {
                            "event": "dataset_completed",
                            "dataset_id": record["dataset_id"],
                            "status": record["status"],
                            "completed": len(records),
                            "total": len(sources),
                            "eta_seconds": eta,
                            "system_memory_fraction": _system_memory_fraction(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                _write_state(output_root, records, status="running", **state_arguments)
    _aggregate(output_root, records)
    status = (
        "completed"
        if all(record["status"] == "completed" for record in records)
        else "completed_with_failures"
    )
    _write_state(output_root, records, status=status, **state_arguments)
    return {
        "status": status,
        "datasets": len(sources),
        "completed": sum(record["status"] == "completed" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "effective_jobs": effective_jobs,
        "output_root": str(output_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--maximum-memory-fraction", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_derived_campaign(
        arguments.source_campaign,
        arguments.output_dir,
        jobs=arguments.jobs,
        maximum_memory_fraction=arguments.maximum_memory_fraction,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "OUTPUT_TABLES",
    "SCHEMA_VERSION",
    "main",
    "run_derived_campaign",
    "run_e5_source_dataset",
]
