"""Execute the checksum-bound suggest-next2 v7 integrated campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import platform
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psutil  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits

from benchmarks.adapters.common import (
    git_metadata,
    json_safe,
    package_versions,
    sha256_file,
    write_json,
)
from benchmarks.simulation.v7_dgp import generate_v7_dgp
from benchmarks.simulation.v7_integrated import run_v7_integrated_matrix
from benchmarks.simulation.v7_metrics import evaluate_v7_integrated_matrix
from benchmarks.simulation.v7_protocol import (
    DEFAULT_CONFIG,
    PHASES,
    PROFILES,
    V7BenchmarkProtocol,
    expand_v7_benchmark_plan,
    load_v7_benchmark_protocol,
)
from crychic.workflow import build_v7_diagnostics, run_subject_crossfit

CAMPAIGN_SCHEMA_VERSION = "crychic-suggest-next2-v7-campaign-v1"
DATASET_SCHEMA_VERSION = "crychic-suggest-next2-v7-dataset-result-v1"
RUN_COLUMNS = (
    "dataset_id",
    "dgp_family",
    "design_kind",
    "replicate_index",
    "seed",
    "candidate_sender_count",
    "cells_per_type",
    "subjects_per_level",
    "status",
    "reason_code",
    "elapsed_seconds",
    "peak_process_tree_rss_bytes",
    "result_directory",
    "manifest_sha256",
)
_DATASET_PLAN_COLUMNS = (
    "phase",
    "family_role",
    "dgp_family",
    "design_kind",
    "replicate_index",
    "seed",
    "dataset_id",
    "candidate_sender_count",
    "cells_per_type",
    "subjects_per_level",
)
DATASET_TABLES = (
    "truth",
    "score_views",
    "effects",
    "aligned_effects",
    "metrics",
    "gate_attrition",
    "score_geometry",
    "candidate_sender_bias",
    "resolution_performance",
)
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CRYCHIC_RECEIVER_JOBS": "1",
    "CRYCHIC_RECEIVER_PROCESS_JOBS": "1",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _system_memory_fraction() -> float:
    memory = psutil.virtual_memory()
    return float(1.0 - memory.available / memory.total)


def _process_tree_rss(process: psutil.Process) -> int:
    total = 0
    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.Error, OSError):
        processes = [process]
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return total


class _PeakRSSMonitor:
    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        self._process = psutil.Process()
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_rss_bytes = 0
        self.maximum_system_memory_fraction = 0.0

    def _sample_once(self) -> None:
        self.peak_rss_bytes = max(
            self.peak_rss_bytes,
            _process_tree_rss(self._process),
        )
        self.maximum_system_memory_fraction = max(
            self.maximum_system_memory_fraction,
            _system_memory_fraction(),
        )

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample_once()

    def __enter__(self) -> _PeakRSSMonitor:
        self._sample_once()
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample_once()


class _DatasetLogger:
    def __init__(self, path: Path, *, dataset_id: str) -> None:
        self.path = path
        self.dataset_id = dataset_id
        self.started = time.monotonic()
        self.stage_timings: list[dict[str, object]] = []

    def event(self, event: str, *, stage: str, **fields: object) -> None:
        record = {
            "timestamp_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - self.started,
            "dataset_id": self.dataset_id,
            "event": event,
            "stage": stage,
            "pid": os.getpid(),
            "process_tree_rss_bytes": _process_tree_rss(psutil.Process()),
            "system_memory_fraction": _system_memory_fraction(),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    json_safe(record),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )

    @contextmanager
    def stage(self, name: str) -> Iterable[None]:
        started = time.monotonic()
        self.event("started", stage=name)
        try:
            yield
        except BaseException as error:
            duration = time.monotonic() - started
            self.stage_timings.append(
                {"stage": name, "status": "failed", "seconds": duration}
            )
            self.event(
                "failed",
                stage=name,
                duration_seconds=duration,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        duration = time.monotonic() - started
        self.stage_timings.append(
            {"stage": name, "status": "completed", "seconds": duration}
        )
        self.event("completed", stage=name, duration_seconds=duration)


@dataclass(frozen=True, slots=True)
class V7DatasetRunSummary:
    dataset_id: str
    dgp_family: str
    design_kind: str
    replicate_index: int
    seed: int
    candidate_sender_count: int
    cells_per_type: int
    subjects_per_level: int
    status: str
    reason_code: str | None
    elapsed_seconds: float
    peak_process_tree_rss_bytes: int
    result_directory: str
    manifest_sha256: str | None

    def to_record(self) -> dict[str, object]:
        return {column: getattr(self, column) for column in RUN_COLUMNS}


def build_v7_campaign_plan(
    protocol: V7BenchmarkProtocol,
    *,
    phase: str,
    profiles: Sequence[str],
    maximum_replicates: int | None = None,
    maximum_datasets: int | None = None,
) -> pd.DataFrame:
    """Union profiles while retaining every profile-specific run identity."""

    normalized_profiles = tuple(dict.fromkeys(profiles))
    if not normalized_profiles or any(
        item not in PROFILES for item in normalized_profiles
    ):
        raise ValueError(f"profiles must be a non-empty subset of {list(PROFILES)}")
    parts = [
        expand_v7_benchmark_plan(
            protocol,
            phase=phase,
            profile=profile,
            maximum_replicates=maximum_replicates,
        )
        for profile in normalized_profiles
    ]
    plan = pd.concat(parts, ignore_index=True)
    if plan["run_id"].duplicated().any():
        raise RuntimeError("campaign profiles generated duplicate run IDs")
    dataset_order = plan.loc[:, list(_DATASET_PLAN_COLUMNS)].drop_duplicates()
    if dataset_order["dataset_id"].duplicated().any():
        raise RuntimeError("campaign profiles disagree on dataset generation fields")
    if maximum_datasets is not None:
        if (
            isinstance(maximum_datasets, bool)
            or not isinstance(maximum_datasets, int)
            or maximum_datasets < 1
        ):
            raise ValueError("maximum_datasets must be a positive integer")
        selected = set(dataset_order.head(maximum_datasets)["dataset_id"])
        plan = plan.loc[plan["dataset_id"].isin(selected)].copy()
    return plan.sort_values(
        [
            "family_role",
            "dgp_family",
            "design_kind",
            "replicate_index",
            "profile",
            "generator_id",
            "inference_id",
        ],
        kind="stable",
        ignore_index=True,
    )


def _dataset_groups(plan: pd.DataFrame) -> list[tuple[dict[str, object], pd.DataFrame]]:
    groups: list[tuple[dict[str, object], pd.DataFrame]] = []
    for dataset_id, rows in plan.groupby("dataset_id", observed=True, sort=False):
        dataset = rows.loc[:, list(_DATASET_PLAN_COLUMNS)].drop_duplicates()
        if len(dataset) != 1:
            raise RuntimeError(
                f"dataset {dataset_id} has inconsistent generation fields"
            )
        groups.append((dataset.iloc[0].to_dict(), rows.copy(deep=True)))
    return groups


def _safe_dataset_id(value: object) -> str:
    dataset_id = str(value)
    if (
        not dataset_id
        or dataset_id != dataset_id.strip()
        or dataset_id in {".", ".."}
        or Path(dataset_id).name != dataset_id
    ):
        raise ValueError("dataset_id must be a safe path component")
    return dataset_id


def _write_parquet(table: pd.DataFrame, path: Path) -> dict[str, object]:
    table.to_parquet(path, index=False, compression="zstd")
    return {
        "filename": path.name,
        "rows": len(table),
        "columns": list(map(str, table.columns)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_completed_dataset(path: Path) -> dict[str, object] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("status") != "completed":
        return None
    outputs = value.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for record in outputs.values():
        if not isinstance(record, dict):
            return None
        filename = record.get("filename")
        expected = record.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected, str):
            return None
        candidate = path / filename
        if not candidate.is_file() or sha256_file(candidate) != expected:
            return None
    return value


def _summary_from_manifest(
    path: Path,
    manifest: Mapping[str, object],
) -> V7DatasetRunSummary:
    plan = manifest["dataset_plan"]
    if not isinstance(plan, Mapping):
        raise ValueError("dataset manifest lacks dataset_plan")
    return V7DatasetRunSummary(
        dataset_id=str(plan["dataset_id"]),
        dgp_family=str(plan["dgp_family"]),
        design_kind=str(plan["design_kind"]),
        replicate_index=int(plan["replicate_index"]),
        seed=int(plan["seed"]),
        candidate_sender_count=int(plan["candidate_sender_count"]),
        cells_per_type=int(plan["cells_per_type"]),
        subjects_per_level=int(plan["subjects_per_level"]),
        status=str(manifest["status"]),
        reason_code=None,
        elapsed_seconds=float(manifest["elapsed_seconds"]),
        peak_process_tree_rss_bytes=int(manifest["peak_process_tree_rss_bytes"]),
        result_directory=str(path),
        manifest_sha256=sha256_file(path / "manifest.json"),
    )


def _diagnostic_truth(truth: pd.DataFrame) -> pd.DataFrame:
    return truth.loc[
        :,
        [
            "contrast_name",
            "sender",
            "receiver",
            "interaction_id",
            "truth_causal_sender",
            "truth_score_effect",
            "mechanism_class",
            "pathway",
        ],
    ].rename(
        columns={
            "truth_causal_sender": "truth",
            "truth_score_effect": "truth_effect",
        }
    )


def _dataset_manifest_base(
    protocol: V7BenchmarkProtocol,
    dataset_plan: Mapping[str, object],
    arm_plan: pd.DataFrame,
) -> dict[str, object]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "protocol": protocol.to_manifest(),
        "dataset_plan": {
            column: json_safe(dataset_plan[column]) for column in _DATASET_PLAN_COLUMNS
        },
        "method_arms": arm_plan.loc[
            :, ["profile", "generator_id", "inference_id", "run_id"]
        ].to_dict(orient="records"),
        "created_utc": _utc_now(),
    }


def run_v7_dataset(
    *,
    protocol_path: Path,
    dataset_plan: Mapping[str, object],
    arm_records: Sequence[Mapping[str, object]],
    output_root: Path,
    maximum_memory_fraction: float = 0.8,
    overwrite: bool = False,
) -> V7DatasetRunSummary:
    """Execute one dataset once, persist all arms atomically, and return status."""

    started = time.monotonic()
    protocol = load_v7_benchmark_protocol(protocol_path)
    dataset_id = _safe_dataset_id(dataset_plan["dataset_id"])
    final = output_root / "datasets" / dataset_id
    existing = _validate_completed_dataset(final)
    if existing is not None and not overwrite:
        return _summary_from_manifest(final, existing)
    if final.exists() and not overwrite:
        raise FileExistsError(
            f"incomplete or corrupt dataset output exists: {final}; pass overwrite"
        )
    temporary = output_root / "datasets" / (
        f".{dataset_id}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    temporary.mkdir(parents=True, exist_ok=False)
    logger = _DatasetLogger(temporary / "run.jsonl", dataset_id=dataset_id)
    arm_plan = pd.DataFrame.from_records(arm_records)
    arms = tuple(
        sorted(
            {
                (str(row.generator_id), str(row.inference_id))
                for row in arm_plan.itertuples(index=False)
            }
        )
    )
    manifest = _dataset_manifest_base(protocol, dataset_plan, arm_plan)
    monitor = _PeakRSSMonitor()
    try:
        if _system_memory_fraction() >= maximum_memory_fraction:
            raise MemoryError(
                "system memory is already at or above the configured campaign limit"
            )
        with monitor, threadpool_limits(limits=1):
            with logger.stage("generate_raw_dgp"):
                fixture = generate_v7_dgp(
                    dataset_id=dataset_id,
                    dgp_family=str(dataset_plan["dgp_family"]),
                    design_kind=str(dataset_plan["design_kind"]),
                    seed=int(dataset_plan["seed"]),
                    candidate_sender_count=int(
                        dataset_plan["candidate_sender_count"]
                    ),
                    cells_per_type=int(dataset_plan["cells_per_type"]),
                    subjects_per_level=int(dataset_plan["subjects_per_level"]),
                )
            with logger.stage("subject_crossfit"):
                crossfit = run_subject_crossfit(
                    fixture.adata,
                    fixture.config,
                    fixture.resource,
                    fixture.target_prior,
                    spec=fixture.crossfit_spec,
                    n_jobs=1,
                )
            with logger.stage("integrated_score_inference_matrix"):
                integrated = run_v7_integrated_matrix(
                    crossfit,
                    dataset_id=dataset_id,
                    design=fixture.differential_design,
                    sample_metadata=fixture.sample_metadata,
                    arms=arms,
                )
            with logger.stage("truth_metrics"):
                aligned, metrics = evaluate_v7_integrated_matrix(
                    integrated.score_views,
                    integrated.effects,
                    fixture.truth,
                    dgp_family=fixture.dgp_family,
                    design_kind=fixture.design_kind,
                )
            with logger.stage("v7_diagnostics"):
                diagnostics = build_v7_diagnostics(
                    crossfit,
                    dataset_id=dataset_id,
                    truth=_diagnostic_truth(fixture.truth),
                    cell_counts=fixture.cell_counts,
                )
            with logger.stage("persist_outputs"):
                tables = {
                    "truth": fixture.truth,
                    "score_views": integrated.score_views,
                    "effects": integrated.effects,
                    "aligned_effects": aligned,
                    "metrics": metrics,
                    "gate_attrition": diagnostics.gate_attrition,
                    "score_geometry": diagnostics.score_geometry,
                    "candidate_sender_bias": diagnostics.candidate_sender_bias,
                    "resolution_performance": diagnostics.resolution_performance,
                }
                outputs = {
                    name: _write_parquet(table, temporary / f"{name}.parquet")
                    for name, table in tables.items()
                }
        elapsed = time.monotonic() - started
        if monitor.maximum_system_memory_fraction >= maximum_memory_fraction:
            raise MemoryError(
                "dataset execution reached the configured system-memory limit"
            )
        manifest.update(
            {
                "status": "completed",
                "completed_utc": _utc_now(),
                "elapsed_seconds": elapsed,
                "peak_process_tree_rss_bytes": monitor.peak_rss_bytes,
                "maximum_system_memory_fraction": (
                    monitor.maximum_system_memory_fraction
                ),
                "stage_timings": logger.stage_timings,
                "fixture": fixture.to_manifest(),
                "crossfit_id": crossfit.crossfit_id,
                "diagnostics": diagnostics.to_manifest(),
                "g0_g2_equivalence": dict(integrated.equivalence_diagnostic),
                "outputs": outputs,
                "runtime": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                    "thread_environment": {
                        name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
                    },
                    "packages": package_versions(
                        (
                            "CRYCHIC",
                            "anndata",
                            "numpy",
                            "pandas",
                            "psutil",
                            "pyarrow",
                            "scikit-learn",
                            "scipy",
                        )
                    ),
                    "git": git_metadata(Path(__file__).resolve().parents[2]),
                },
                "formal_inference_allowed": False,
                "formal_inference_reason": (
                    "analytic_I0_I1_I2_require_full_pipeline_subject_resampling"
                ),
            }
        )
        write_json(temporary / "manifest.json", manifest)
        if final.exists():
            shutil.rmtree(final)
        temporary.replace(final)
        return _summary_from_manifest(final, manifest)
    except BaseException as error:
        elapsed = time.monotonic() - started
        logger.event(
            "failed",
            stage="dataset",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        failure = {
            **manifest,
            "status": "failed",
            "completed_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "peak_process_tree_rss_bytes": monitor.peak_rss_bytes,
            "maximum_system_memory_fraction": monitor.maximum_system_memory_fraction,
            "stage_timings": logger.stage_timings,
            "reason_code": type(error).__name__,
            "error": str(error),
        }
        write_json(temporary / "manifest.json", failure)
        failed = output_root / "failed" / dataset_id
        failed.parent.mkdir(parents=True, exist_ok=True)
        if failed.exists():
            shutil.rmtree(failed)
        temporary.replace(failed)
        return V7DatasetRunSummary(
            dataset_id=dataset_id,
            dgp_family=str(dataset_plan["dgp_family"]),
            design_kind=str(dataset_plan["design_kind"]),
            replicate_index=int(dataset_plan["replicate_index"]),
            seed=int(dataset_plan["seed"]),
            candidate_sender_count=int(dataset_plan["candidate_sender_count"]),
            cells_per_type=int(dataset_plan["cells_per_type"]),
            subjects_per_level=int(dataset_plan["subjects_per_level"]),
            status="failed",
            reason_code=type(error).__name__,
            elapsed_seconds=elapsed,
            peak_process_tree_rss_bytes=monitor.peak_rss_bytes,
            result_directory=str(failed),
            manifest_sha256=sha256_file(failed / "manifest.json"),
        )


def _worker(payload: Mapping[str, object]) -> dict[str, object]:
    for name, value in _THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    summary = run_v7_dataset(
        protocol_path=Path(str(payload["protocol_path"])),
        dataset_plan=payload["dataset_plan"],  # type: ignore[arg-type]
        arm_records=payload["arm_records"],  # type: ignore[arg-type]
        output_root=Path(str(payload["output_root"])),
        maximum_memory_fraction=float(payload["maximum_memory_fraction"]),
        overwrite=bool(payload["overwrite"]),
    )
    return summary.to_record()


def _payload(
    protocol: V7BenchmarkProtocol,
    dataset_plan: Mapping[str, object],
    arms: pd.DataFrame,
    *,
    output_root: Path,
    maximum_memory_fraction: float,
    overwrite: bool,
) -> dict[str, object]:
    return {
        "protocol_path": str(protocol.path),
        "dataset_plan": dict(dataset_plan),
        "arm_records": arms.to_dict(orient="records"),
        "output_root": str(output_root),
        "maximum_memory_fraction": maximum_memory_fraction,
        "overwrite": overwrite,
    }


def _auto_worker_count(
    *,
    pilot_peak_rss_bytes: int,
    maximum_memory_fraction: float,
    requested_jobs: int,
) -> int:
    memory = psutil.virtual_memory()
    current_used = memory.total - memory.available
    headroom = max(0, int(maximum_memory_fraction * memory.total) - current_used)
    conservative_peak = max(1, math.ceil(1.35 * pilot_peak_rss_bytes))
    memory_workers = max(1, headroom // conservative_peak)
    cpu_workers = max(1, os.cpu_count() or 1)
    requested = cpu_workers if requested_jobs == 0 else requested_jobs
    return max(1, min(requested, cpu_workers, memory_workers))


def _eta_seconds(durations: Sequence[float], remaining: int, workers: int) -> float:
    if not durations or remaining <= 0:
        return 0.0
    window = durations[-min(20, len(durations)) :]
    return float(sum(window) / len(window) * remaining / max(1, workers))


def _write_campaign_state(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
    *,
    plan: pd.DataFrame,
    protocol: V7BenchmarkProtocol,
    phase: str,
    profiles: Sequence[str],
    effective_jobs: int,
    started_utc: str,
    status: str,
) -> None:
    runs = pd.DataFrame.from_records(records, columns=RUN_COLUMNS)
    runs_path = output_root / "runs.tsv"
    runs.to_csv(runs_path, sep="\t", index=False)
    manifest = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "status": status,
        "phase": phase,
        "profiles": list(profiles),
        "started_utc": started_utc,
        "updated_utc": _utc_now(),
        "protocol": protocol.to_manifest(),
        "planned_datasets": int(plan["dataset_id"].nunique()),
        "planned_method_rows": len(plan),
        "completed_datasets": int(runs["status"].eq("completed").sum()),
        "failed_datasets": int(runs["status"].eq("failed").sum()),
        "effective_jobs": effective_jobs,
        "maximum_memory_fraction": float(
            protocol.config["execution"]["maximum_memory_fraction"]
        ),
        "runs": {
            "filename": runs_path.name,
            "sha256": sha256_file(runs_path),
            "rows": len(runs),
        },
    }
    write_json(output_root / "campaign_manifest.json", manifest)


def _aggregate_campaign_metrics(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    parts: list[pd.DataFrame] = []
    for record in records:
        if record["status"] != "completed":
            continue
        path = Path(str(record["result_directory"])) / "metrics.parquet"
        parts.append(pd.read_parquet(path))
    if not parts:
        return
    metrics = pd.concat(parts, ignore_index=True)
    metrics.to_parquet(
        output_root / "all_dataset_metrics.parquet",
        index=False,
        compression="zstd",
    )
    numeric = metrics.loc[metrics["status"].eq("observed")].copy()
    summary = (
        numeric.groupby(
            [
                "dgp_family",
                "design_kind",
                "generator_id",
                "score_view",
                "inference_id",
                "metric",
            ],
            observed=True,
            sort=True,
        )["value"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    summary.to_csv(output_root / "metric_summary.tsv", sep="\t", index=False)


def run_v7_campaign(
    *,
    protocol_path: Path = DEFAULT_CONFIG,
    phase: str,
    profiles: Sequence[str],
    output_root: Path,
    maximum_replicates: int | None = None,
    maximum_datasets: int | None = None,
    jobs: int = 0,
    overwrite: bool = False,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Execute a resumable campaign with pilot-derived process parallelism."""

    if phase not in PHASES:
        raise ValueError(f"phase must be one of {list(PHASES)}")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 0:
        raise ValueError("jobs must be zero (auto) or a positive integer")
    protocol = load_v7_benchmark_protocol(protocol_path)
    maximum_memory_fraction = float(
        protocol.config["execution"]["maximum_memory_fraction"]
    )
    repository = Path(__file__).resolve().parents[2]
    git = git_metadata(repository)
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("formal v7 campaign refuses a dirty worktree")
    plan = build_v7_campaign_plan(
        protocol,
        phase=phase,
        profiles=profiles,
        maximum_replicates=maximum_replicates,
        maximum_datasets=maximum_datasets,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "datasets").mkdir(exist_ok=True)
    plan.to_csv(output_root / "run_plan.tsv", sep="\t", index=False)
    groups = _dataset_groups(plan)
    started_utc = _utc_now()
    records: list[dict[str, object]] = []
    durations: list[float] = []

    first_plan, first_arms = groups[0]
    first = _worker(
        _payload(
            protocol,
            first_plan,
            first_arms,
            output_root=output_root,
            maximum_memory_fraction=maximum_memory_fraction,
            overwrite=overwrite,
        )
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
                "status": first["status"],
                "elapsed_seconds": first["elapsed_seconds"],
                "peak_rss_bytes": first["peak_process_tree_rss_bytes"],
                "effective_jobs": effective_jobs,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _write_campaign_state(
        output_root,
        records,
        plan=plan,
        protocol=protocol,
        phase=phase,
        profiles=profiles,
        effective_jobs=effective_jobs,
        started_utc=started_utc,
        status="running",
    )
    remaining_groups = groups[1:]
    if remaining_groups:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_jobs,
            mp_context=context,
        ) as executor:
            pending: dict[concurrent.futures.Future[dict[str, object]], str] = {}
            iterator = iter(remaining_groups)
            exhausted = False
            while pending or not exhausted:
                while len(pending) < effective_jobs and not exhausted:
                    if _system_memory_fraction() >= maximum_memory_fraction:
                        break
                    try:
                        dataset_plan, arm_plan = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    payload = _payload(
                        protocol,
                        dataset_plan,
                        arm_plan,
                        output_root=output_root,
                        maximum_memory_fraction=maximum_memory_fraction,
                        overwrite=overwrite,
                    )
                    future = executor.submit(_worker, payload)
                    pending[future] = str(dataset_plan["dataset_id"])
                if not pending:
                    if exhausted:
                        break
                    time.sleep(1.0)
                    continue
                done, _ = concurrent.futures.wait(
                    pending,
                    timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    dataset_id = pending.pop(future)
                    try:
                        record = future.result()
                    except BaseException as error:
                        record = {
                            "dataset_id": dataset_id,
                            "dgp_family": "unknown",
                            "design_kind": "unknown",
                            "replicate_index": -1,
                            "seed": -1,
                            "candidate_sender_count": -1,
                            "cells_per_type": -1,
                            "subjects_per_level": -1,
                            "status": "failed",
                            "reason_code": type(error).__name__,
                            "elapsed_seconds": 0.0,
                            "peak_process_tree_rss_bytes": 0,
                            "result_directory": "",
                            "manifest_sha256": None,
                        }
                    records.append(record)
                    durations.append(float(record["elapsed_seconds"]))
                    remaining = len(groups) - len(records)
                    eta = _eta_seconds(durations, remaining, effective_jobs)
                    print(
                        json.dumps(
                            {
                                "event": "dataset_completed",
                                "dataset_id": record["dataset_id"],
                                "status": record["status"],
                                "completed": len(records),
                                "total": len(groups),
                                "eta_seconds": eta,
                                "system_memory_fraction": _system_memory_fraction(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    _write_campaign_state(
                        output_root,
                        records,
                        plan=plan,
                        protocol=protocol,
                        phase=phase,
                        profiles=profiles,
                        effective_jobs=effective_jobs,
                        started_utc=started_utc,
                        status="running",
                    )
    _aggregate_campaign_metrics(output_root, records)
    final_status = (
        "completed"
        if all(record["status"] == "completed" for record in records)
        else "completed_with_failures"
    )
    _write_campaign_state(
        output_root,
        records,
        plan=plan,
        protocol=protocol,
        phase=phase,
        profiles=profiles,
        effective_jobs=effective_jobs,
        started_utc=started_utc,
        status=final_status,
    )
    return {
        "status": final_status,
        "datasets": len(groups),
        "completed": sum(record["status"] == "completed" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "effective_jobs": effective_jobs,
        "output_root": str(output_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        default=list(PROFILES),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-replicates", type=int)
    parser.add_argument("--maximum-datasets", type=int)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="0 derives worker count from pilot RSS, host memory, and CPU count",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    for name, value in _THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    summary = run_v7_campaign(
        protocol_path=arguments.protocol,
        phase=arguments.phase,
        profiles=tuple(arguments.profiles),
        output_root=arguments.output_dir,
        maximum_replicates=arguments.maximum_replicates,
        maximum_datasets=arguments.maximum_datasets,
        jobs=arguments.jobs,
        overwrite=arguments.overwrite,
        allow_dirty=arguments.allow_dirty,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
