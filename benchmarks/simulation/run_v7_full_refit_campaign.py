"""Run resumable PR10 full-refit distributions on the frozen v7 DGP plan."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
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
from anndata import AnnData
from threadpoolctl import threadpool_limits

from benchmarks.adapters.common import (
    git_metadata,
    json_safe,
    package_versions,
    sha256_file,
    write_json,
)
from benchmarks.simulation.run_v7_campaign import build_v7_campaign_plan
from benchmarks.simulation.v7_dgp import V7DGPFixture, generate_v7_dgp
from benchmarks.simulation.v7_protocol import AMENDMENT_CONFIG as DEFAULT_PROTOCOL
from benchmarks.simulation.v7_protocol import (
    DESIGN_KINDS,
    PHASES,
    V7BenchmarkProtocol,
    load_v7_benchmark_protocol,
)
from crychic.core import SeedLineage, stable_id
from crychic.inference import DifferentialDesignKind, TwoPartOccurrenceV2Spec
from crychic.resampling import ContextPermutationOperation
from crychic.scoring import (
    FrozenHypergraphPrior,
    UncertaintyAwareHypergraphShrinkageV2Spec,
    freeze_hypergraph_prior,
)
from crychic.workflow import (
    V7EstimatorSpec,
    V7FullPipelineResampleRecord,
    V7FullPipelineResamplingResult,
    finalize_v7_full_pipeline_inference,
    run_v7_full_pipeline_resampling,
)

CAMPAIGN_SCHEMA_VERSION = "crychic-suggest-next2-v7-pr10-campaign-v1"
DATASET_SCHEMA_VERSION = "crychic-suggest-next2-v7-pr10-dataset-v1"
_TOPOLOGY_VIEWS = ("sender", "ligand", "receptor", "receiver", "pathway")
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
RUN_COLUMNS = (
    "dataset_id",
    "dgp_family",
    "design_kind",
    "replicate_index",
    "seed",
    "status",
    "reason_code",
    "n_bootstraps",
    "n_permutations",
    "run_loso",
    "planned_resamples",
    "successful_resamples",
    "elapsed_seconds",
    "peak_process_tree_rss_bytes",
    "result_directory",
    "manifest_sha256",
)
OUTPUT_TABLES = (
    "truth",
    "point_continuous_effects",
    "point_occurrence_subject_events",
    "point_occurrence_effects",
    "point_hypergraph_effects",
    "continuous_resample_ledger",
    "occurrence_resample_ledger",
    "hypergraph_resample_ledger",
    "full_refit_continuous_effects",
    "full_refit_occurrence_effects",
    "full_refit_hypergraph_effects",
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
    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.Error, OSError):
        processes = [process]
    total = 0
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
        self._lock = threading.Lock()

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
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + "\n"
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
class V7FullRefitRunSummary:
    dataset_id: str
    dgp_family: str
    design_kind: str
    replicate_index: int
    seed: int
    status: str
    reason_code: str | None
    n_bootstraps: int
    n_permutations: int
    run_loso: bool
    planned_resamples: int
    successful_resamples: int
    elapsed_seconds: float
    peak_process_tree_rss_bytes: int
    result_directory: str
    manifest_sha256: str | None

    def to_record(self) -> dict[str, object]:
        return {column: getattr(self, column) for column in RUN_COLUMNS}


def _count(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


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


def build_v7_full_refit_plan(
    protocol: V7BenchmarkProtocol,
    *,
    phase: str,
    maximum_replicates: int | None = None,
    maximum_datasets: int | None = None,
    dgp_families: Sequence[str] | None = None,
    design_kinds: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select unique raw datasets from the frozen paired method plan."""

    arms = build_v7_campaign_plan(
        protocol,
        phase=phase,
        profiles=("score_primary",),
        maximum_replicates=maximum_replicates,
        dgp_families=dgp_families,
    )
    plan = arms.loc[:, list(_DATASET_PLAN_COLUMNS)].drop_duplicates()
    if plan["dataset_id"].duplicated().any():
        raise RuntimeError("full-refit plan contains duplicate dataset IDs")
    if design_kinds is not None:
        selected = tuple(dict.fromkeys(map(str, design_kinds)))
        if not selected or not set(selected).issubset(DESIGN_KINDS):
            raise ValueError("design_kinds must be a non-empty supported subset")
        plan = plan.loc[plan["design_kind"].isin(selected)].copy()
    plan = plan.sort_values(
        ["family_role", "dgp_family", "design_kind", "replicate_index"],
        kind="stable",
        ignore_index=True,
    )
    if maximum_datasets is not None:
        if (
            isinstance(maximum_datasets, bool)
            or not isinstance(maximum_datasets, int)
            or maximum_datasets < 1
        ):
            raise ValueError("maximum_datasets must be a positive integer")
        plan = plan.head(maximum_datasets).copy()
    if plan.empty:
        raise ValueError("full-refit filters selected no frozen datasets")
    return plan.reset_index(drop=True)


def _write_parquet(table: pd.DataFrame, path: Path) -> dict[str, object]:
    table.to_parquet(path, index=False, compression="zstd")
    return {
        "filename": path.name,
        "rows": len(table),
        "columns": list(map(str, table.columns)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_jsonl_gzip(
    records: Sequence[Mapping[str, object]],
    path: Path,
) -> dict[str, object]:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(
                json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + "\n"
            )
    return {
        "filename": path.name,
        "rows": len(records),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "compression": "gzip",
    }


def _validate_completed_dataset(path: Path) -> dict[str, object] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != DATASET_SCHEMA_VERSION
        or value.get("status") != "completed"
    ):
        return None
    for collection in ("outputs", "resampling_artifacts"):
        records = value.get(collection)
        if not isinstance(records, dict):
            return None
        for record in records.values():
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
) -> V7FullRefitRunSummary:
    plan = manifest["dataset_plan"]
    request = manifest["resampling_request"]
    result = manifest["resampling"]
    if not all(isinstance(value, Mapping) for value in (plan, request, result)):
        raise ValueError("completed PR10 manifest is malformed")
    plan = plan  # type: ignore[assignment]
    request = request  # type: ignore[assignment]
    result = result  # type: ignore[assignment]
    successful = result["successful_counts"]
    if not isinstance(successful, Mapping):
        raise ValueError("completed PR10 manifest lacks success counts")
    return V7FullRefitRunSummary(
        dataset_id=str(plan["dataset_id"]),
        dgp_family=str(plan["dgp_family"]),
        design_kind=str(plan["design_kind"]),
        replicate_index=int(plan["replicate_index"]),
        seed=int(plan["seed"]),
        status="completed",
        reason_code=None,
        n_bootstraps=int(request["n_bootstraps"]),
        n_permutations=int(request["n_permutations"]),
        run_loso=bool(request["run_loso"]),
        planned_resamples=sum(int(value) for value in result["plan_counts"].values()),
        successful_resamples=sum(int(value) for value in successful.values()),
        elapsed_seconds=float(manifest["elapsed_seconds"]),
        peak_process_tree_rss_bytes=int(manifest["peak_process_tree_rss_bytes"]),
        result_directory=str(path),
        manifest_sha256=sha256_file(path / "manifest.json"),
    )


def _enriched_adata(fixture: V7DGPFixture) -> tuple[AnnData, tuple[str, ...]]:
    design = fixture.differential_design
    required = {
        design.condition_column,
        *design.batch_columns,
        *design.continuous_covariates,
        *design.categorical_covariates,
    }
    if design.cohort_column is not None:
        required.add(design.cohort_column)
    missing = tuple(sorted(required.difference(fixture.adata.obs.columns)))
    if not missing:
        return fixture.adata, ()
    absent = set(missing).difference(fixture.sample_metadata.columns)
    if absent:
        raise ValueError(
            f"fixture sample metadata lacks design fields: {sorted(absent)}"
        )
    adata = fixture.adata.copy()
    metadata = fixture.sample_metadata.set_index("sample_id")
    samples = adata.obs[fixture.config.sample_key].astype(str)
    for column in missing:
        adata.obs[column] = samples.map(metadata[column])
        if adata.obs[column].isna().any():
            raise ValueError(f"failed to bind design metadata column {column!r}")
    return adata, missing


def _freeze_full_hypergraph_prior(
    fixture: V7DGPFixture,
    adata: AnnData,
) -> FrozenHypergraphPrior:
    cell_types = tuple(
        sorted(adata.obs[fixture.config.cell_type_key].astype(str).unique())
    )
    rows: list[dict[str, str]] = []
    for sender in cell_types:
        for receiver in cell_types:
            for interaction in fixture.resource.interactions:
                rows.append(
                    {
                        "edge_id": stable_id(
                            "sample_edge_differential_event",
                            {
                                "score_head": "sender_detection_raw",
                                "sender": sender,
                                "receiver": receiver,
                                "interaction_id": interaction.interaction_id,
                            },
                            schema_version="1",
                        ),
                        "sender": sender,
                        "ligand": interaction.ligand_name,
                        "receptor": interaction.receptor_name,
                        "receiver": receiver,
                        "pathway": (interaction.pathway or "__unannotated_pathway__"),
                    }
                )
    return freeze_hypergraph_prior(
        pd.DataFrame.from_records(rows),
        view_columns=_TOPOLOGY_VIEWS,
    )


def _estimator_spec(
    fixture: V7DGPFixture,
    protocol: V7BenchmarkProtocol,
) -> V7EstimatorSpec:
    experiments = protocol.config["experiments"]
    if not isinstance(experiments, Mapping):
        raise TypeError("protocol experiments must be a mapping")
    m4 = experiments.get("E6_m4_occurrence")
    if not isinstance(m4, Mapping):
        raise ValueError("PR10 requires the frozen E6 M4 amendment")
    design = fixture.differential_design
    occurrence = TwoPartOccurrenceV2Spec(
        design=design,
        activity_head=str(m4["activity_head"]),
        activity_threshold_raw=float(m4["activity_threshold_raw"]),
        probability_transition_scale=float(m4["probability_transition_scale"]),
        occurrence_state_source=str(m4["occurrence_state_source"]),
        active_probability_threshold=float(m4["active_probability_threshold"]),
        active_probability_mapping=str(m4["active_probability_mapping"]),
    )
    if DifferentialDesignKind(design.design_kind) is DifferentialDesignKind.CONTINUOUS:
        contrast_name = f"slope:{design.condition_column}"
    else:
        names = {contrast.name for contrast in design.contrasts}
        contrast_name = "B_vs_A" if "B_vs_A" in names else sorted(names)[0]
    return V7EstimatorSpec(
        design=design,
        score_head="sender_detection_raw",
        occurrence_spec=occurrence,
        hypergraph_spec=UncertaintyAwareHypergraphShrinkageV2Spec(
            minimum_observed_edges=4
        ),
        hypergraph_contrast_name=contrast_name,
    )


def _point_tables(
    resampling: V7FullPipelineResamplingResult,
) -> dict[str, pd.DataFrame]:
    point = resampling.point
    occurrence = point.occurrence
    hypergraph = point.hypergraph
    if occurrence is None or hypergraph is None:
        raise RuntimeError("PR10 campaign requires configured M4 and M5 point stages")
    return {
        "point_continuous_effects": point.differential.effects,
        "point_occurrence_subject_events": occurrence.occurrence.subject_events,
        "point_occurrence_effects": occurrence.occurrence.effects,
        "point_hypergraph_effects": hypergraph.shrinkage,
    }


def run_v7_full_refit_dataset(
    *,
    protocol_path: Path,
    dataset_plan: Mapping[str, object],
    output_root: Path,
    n_bootstraps: int,
    n_permutations: int,
    run_loso: bool,
    resample_jobs: int = 1,
    maximum_memory_fraction: float = 0.8,
    overwrite: bool = False,
) -> V7FullRefitRunSummary:
    """Execute and atomically persist one complete PR10 distribution."""

    bootstraps = _count(n_bootstraps, field_name="n_bootstraps")
    permutations = _count(n_permutations, field_name="n_permutations")
    if not isinstance(run_loso, bool):
        raise TypeError("run_loso must be boolean")
    if not any((bootstraps, permutations, run_loso)):
        raise ValueError("request at least one PR10 resampling operation")
    if (
        isinstance(resample_jobs, bool)
        or not isinstance(resample_jobs, int)
        or resample_jobs < 1
    ):
        raise ValueError("resample_jobs must be a positive integer")
    protocol = load_v7_benchmark_protocol(protocol_path)
    dataset_id = _safe_dataset_id(dataset_plan["dataset_id"])
    final = output_root / "datasets" / dataset_id
    existing = _validate_completed_dataset(final)
    if existing is not None and not overwrite:
        existing_request = existing.get("resampling_request")
        existing_protocol = existing.get("protocol")
        if (
            not isinstance(existing_request, Mapping)
            or not isinstance(existing_protocol, Mapping)
            or int(existing_request.get("n_bootstraps", -1)) != bootstraps
            or int(existing_request.get("n_permutations", -1)) != permutations
            or bool(existing_request.get("run_loso")) != run_loso
            or existing_protocol.get("protocol_digest") != protocol.protocol_digest
        ):
            raise FileExistsError(
                "completed PR10 output uses a different protocol or resampling "
                "request; choose another output root or pass overwrite"
            )
        return _summary_from_manifest(final, existing)
    if final.exists() and not overwrite:
        raise FileExistsError(
            f"incomplete or corrupt PR10 output exists: {final}; pass overwrite"
        )
    temporary = (
        output_root
        / "datasets"
        / (f".{dataset_id}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    )
    temporary.mkdir(parents=True, exist_ok=False)
    logger = _DatasetLogger(temporary / "run.jsonl", dataset_id=dataset_id)
    started = time.monotonic()
    monitor = _PeakRSSMonitor()
    manifest: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "protocol": protocol.to_manifest(),
        "dataset_plan": {
            column: json_safe(dataset_plan[column]) for column in _DATASET_PLAN_COLUMNS
        },
        "resampling_request": {
            "n_bootstraps": bootstraps,
            "n_permutations": permutations,
            "run_loso": run_loso,
            "resample_jobs": resample_jobs,
            "calibration_gate_supplied": False,
            "formal_minimum_bootstraps": 1_000,
            "formal_minimum_permutations": 1_000,
        },
    }
    try:
        if _system_memory_fraction() >= maximum_memory_fraction:
            raise MemoryError("system memory is already at the PR10 campaign limit")
        with monitor, threadpool_limits(limits=1):
            with logger.stage("generate_raw_dgp"):
                fixture = generate_v7_dgp(
                    dataset_id=dataset_id,
                    dgp_family=str(dataset_plan["dgp_family"]),
                    design_kind=str(dataset_plan["design_kind"]),
                    seed=int(dataset_plan["seed"]),
                    candidate_sender_count=int(dataset_plan["candidate_sender_count"]),
                    cells_per_type=int(dataset_plan["cells_per_type"]),
                    subjects_per_level=int(dataset_plan["subjects_per_level"]),
                )
                adata, added_design_columns = _enriched_adata(fixture)
            with logger.stage("freeze_outcome_blind_hypergraph_prior"):
                hypergraph_prior = _freeze_full_hypergraph_prior(fixture, adata)
                estimator_spec = _estimator_spec(fixture, protocol)
            progress_started = time.monotonic()
            completed_durations: list[float] = []
            last_completed = progress_started

            def progress(
                record: V7FullPipelineResampleRecord,
                completed: int,
                total: int,
            ) -> None:
                nonlocal last_completed
                now = time.monotonic()
                completed_durations.append(now - last_completed)
                last_completed = now
                window = completed_durations[-min(20, len(completed_durations)) :]
                eta = sum(window) / len(window) * (total - completed)
                logger.event(
                    "completed",
                    stage="full_refit_resample",
                    operation=record.operation.value,
                    resample_index=record.resample_index,
                    status=record.status.value,
                    failure_code=record.failure_code,
                    completed=completed,
                    total=total,
                    eta_seconds=eta,
                )

            with logger.stage("full_pipeline_resampling"):
                multi_context_operation = (
                    ContextPermutationOperation.WITHIN_SUBJECT_COMPLETE_PERMUTATION
                    if fixture.design_kind == "repeated"
                    else None
                )
                resampling = run_v7_full_pipeline_resampling(
                    adata,
                    fixture.config,
                    fixture.resource,
                    fixture.target_prior,
                    crossfit_spec=fixture.crossfit_spec,
                    estimator_spec=estimator_spec,
                    hypergraph_prior=hypergraph_prior,
                    n_bootstraps=bootstraps,
                    n_permutations=permutations,
                    run_loso=run_loso,
                    multi_context_permutation_operation=multi_context_operation,
                    seed_lineage=SeedLineage(fixture.seed).derive(
                        "pr10_full_refit_campaign",
                        dataset_id,
                    ),
                    n_jobs=resample_jobs,
                    progress_callback=progress,
                )
            with logger.stage("full_refit_inference_finalizer"):
                inference = finalize_v7_full_pipeline_inference(resampling)
            with logger.stage("persist_outputs"):
                tables = {
                    "truth": fixture.truth,
                    **_point_tables(resampling),
                    "continuous_resample_ledger": (
                        resampling.continuous_effect_ledger()
                    ),
                    "occurrence_resample_ledger": (
                        resampling.occurrence_effect_ledger()
                    ),
                    "hypergraph_resample_ledger": (
                        resampling.hypergraph_effect_ledger()
                    ),
                    "full_refit_continuous_effects": inference.continuous_effects,
                    "full_refit_occurrence_effects": inference.occurrence_effects,
                    "full_refit_hypergraph_effects": inference.hypergraph_effects,
                }
                if set(tables) != set(OUTPUT_TABLES):
                    raise RuntimeError("PR10 persisted table axis changed")
                outputs = {
                    name: _write_parquet(table, temporary / f"{name}.parquet")
                    for name, table in tables.items()
                }
                resampling_manifest = resampling.to_manifest()
                plans = resampling_manifest.pop("plans")
                records = resampling_manifest.pop("records")
                artifacts = {
                    "plans": _write_jsonl_gzip(
                        plans,
                        temporary / "resampling_plans.jsonl.gz",
                    ),
                    "records": _write_jsonl_gzip(
                        records,
                        temporary / "resampling_records.jsonl.gz",
                    ),
                }
        elapsed = time.monotonic() - started
        if monitor.maximum_system_memory_fraction >= maximum_memory_fraction:
            raise MemoryError("PR10 dataset reached the configured memory limit")
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
                "added_design_metadata_columns": list(added_design_columns),
                "estimator_spec": estimator_spec.to_dict(),
                "hypergraph_prior": hypergraph_prior.to_dict(),
                "resampling": resampling_manifest,
                "inference": inference.to_manifest(),
                "outputs": outputs,
                "resampling_artifacts": artifacts,
                "formal_inference_allowed": False,
                "formal_inference_reason": "calibration_gate_missing",
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
                            "scipy",
                            "statsmodels",
                        )
                    ),
                    "git": git_metadata(Path(__file__).resolve().parents[2]),
                },
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
        return V7FullRefitRunSummary(
            dataset_id=dataset_id,
            dgp_family=str(dataset_plan["dgp_family"]),
            design_kind=str(dataset_plan["design_kind"]),
            replicate_index=int(dataset_plan["replicate_index"]),
            seed=int(dataset_plan["seed"]),
            status="failed",
            reason_code=type(error).__name__,
            n_bootstraps=bootstraps,
            n_permutations=permutations,
            run_loso=run_loso,
            planned_resamples=0,
            successful_resamples=0,
            elapsed_seconds=elapsed,
            peak_process_tree_rss_bytes=monitor.peak_rss_bytes,
            result_directory=str(failed),
            manifest_sha256=sha256_file(failed / "manifest.json"),
        )


def _worker(payload: Mapping[str, object]) -> dict[str, object]:
    for name, value in _THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    summary = run_v7_full_refit_dataset(
        protocol_path=Path(str(payload["protocol_path"])),
        dataset_plan=payload["dataset_plan"],  # type: ignore[arg-type]
        output_root=Path(str(payload["output_root"])),
        n_bootstraps=int(payload["n_bootstraps"]),
        n_permutations=int(payload["n_permutations"]),
        run_loso=bool(payload["run_loso"]),
        resample_jobs=int(payload["resample_jobs"]),
        maximum_memory_fraction=float(payload["maximum_memory_fraction"]),
        overwrite=bool(payload["overwrite"]),
    )
    return summary.to_record()


def _payload(
    protocol: V7BenchmarkProtocol,
    dataset_plan: Mapping[str, object],
    *,
    output_root: Path,
    n_bootstraps: int,
    n_permutations: int,
    run_loso: bool,
    resample_jobs: int,
    maximum_memory_fraction: float,
    overwrite: bool,
) -> dict[str, object]:
    return {
        "protocol_path": str(protocol.path),
        "dataset_plan": dict(dataset_plan),
        "output_root": str(output_root),
        "n_bootstraps": n_bootstraps,
        "n_permutations": n_permutations,
        "run_loso": run_loso,
        "resample_jobs": resample_jobs,
        "maximum_memory_fraction": maximum_memory_fraction,
        "overwrite": overwrite,
    }


def _auto_worker_count(
    *,
    pilot_peak_rss_bytes: int,
    maximum_memory_fraction: float,
    requested_jobs: int,
    resample_jobs: int,
) -> int:
    memory = psutil.virtual_memory()
    current_used = memory.total - memory.available
    headroom = max(0, int(maximum_memory_fraction * memory.total) - current_used)
    conservative_peak = max(1, math.ceil(1.35 * pilot_peak_rss_bytes))
    memory_workers = max(1, headroom // conservative_peak)
    cpu_workers = max(1, (os.cpu_count() or 1) // resample_jobs)
    requested = cpu_workers if requested_jobs == 0 else requested_jobs
    return max(1, min(requested, cpu_workers, memory_workers))


def _write_campaign_state(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
    *,
    plan: pd.DataFrame,
    protocol: V7BenchmarkProtocol,
    n_bootstraps: int,
    n_permutations: int,
    run_loso: bool,
    resample_jobs: int,
    effective_jobs: int,
    started_utc: str,
    status: str,
) -> None:
    runs = pd.DataFrame.from_records(records, columns=RUN_COLUMNS)
    runs_path = output_root / "runs.tsv"
    runs.to_csv(runs_path, sep="\t", index=False)
    write_json(
        output_root / "campaign_manifest.json",
        {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "status": status,
            "started_utc": started_utc,
            "updated_utc": _utc_now(),
            "protocol": protocol.to_manifest(),
            "planned_datasets": len(plan),
            "completed_datasets": int(runs["status"].eq("completed").sum()),
            "failed_datasets": int(runs["status"].eq("failed").sum()),
            "resampling_request": {
                "n_bootstraps": n_bootstraps,
                "n_permutations": n_permutations,
                "run_loso": run_loso,
                "resample_jobs": resample_jobs,
            },
            "effective_dataset_jobs": effective_jobs,
            "maximum_memory_fraction": 0.8,
            "runs": {
                "filename": runs_path.name,
                "sha256": sha256_file(runs_path),
                "rows": len(runs),
            },
        },
    )


def _aggregate_outputs(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    completed = [
        record
        for record in records
        if record["status"] == "completed" and record["result_directory"]
    ]
    for table_name in (
        "full_refit_continuous_effects",
        "full_refit_occurrence_effects",
        "full_refit_hypergraph_effects",
    ):
        parts: list[pd.DataFrame] = []
        for record in completed:
            table = pd.read_parquet(
                Path(str(record["result_directory"])) / f"{table_name}.parquet"
            )
            table.insert(0, "dataset_id", str(record["dataset_id"]))
            table.insert(1, "dgp_family", str(record["dgp_family"]))
            table.insert(2, "design_kind", str(record["design_kind"]))
            parts.append(table)
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(
                output_root / f"all_{table_name}.parquet",
                index=False,
                compression="zstd",
            )


def run_v7_full_refit_campaign(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    phase: str,
    output_root: Path,
    n_bootstraps: int,
    n_permutations: int,
    run_loso: bool = True,
    maximum_replicates: int | None = None,
    maximum_datasets: int | None = None,
    dgp_families: Sequence[str] | None = None,
    design_kinds: Sequence[str] | None = None,
    jobs: int = 0,
    resample_jobs: int = 1,
    overwrite: bool = False,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Execute a memory-bounded dataset-parallel PR10 campaign."""

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
        raise RuntimeError("formal PR10 campaign refuses a dirty worktree")
    plan = build_v7_full_refit_plan(
        protocol,
        phase=phase,
        maximum_replicates=maximum_replicates,
        maximum_datasets=maximum_datasets,
        dgp_families=dgp_families,
        design_kinds=design_kinds,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "datasets").mkdir(exist_ok=True)
    plan.to_csv(output_root / "run_plan.tsv", sep="\t", index=False)
    plans = [row._asdict() for row in plan.itertuples(index=False)]
    started_utc = _utc_now()
    records: list[dict[str, object]] = []
    durations: list[float] = []

    first = _worker(
        _payload(
            protocol,
            plans[0],
            output_root=output_root,
            n_bootstraps=n_bootstraps,
            n_permutations=n_permutations,
            run_loso=run_loso,
            resample_jobs=resample_jobs,
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
        resample_jobs=resample_jobs,
    )
    print(
        json.dumps(
            {
                "event": "pilot_completed",
                "dataset_id": first["dataset_id"],
                "status": first["status"],
                "elapsed_seconds": first["elapsed_seconds"],
                "peak_rss_bytes": first["peak_process_tree_rss_bytes"],
                "effective_dataset_jobs": effective_jobs,
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
        n_bootstraps=n_bootstraps,
        n_permutations=n_permutations,
        run_loso=run_loso,
        resample_jobs=resample_jobs,
        effective_jobs=effective_jobs,
        started_utc=started_utc,
        status="running",
    )
    remaining = plans[1:]
    if remaining:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_jobs,
            mp_context=context,
        ) as executor:
            pending: dict[concurrent.futures.Future[dict[str, object]], str] = {}
            iterator = iter(remaining)
            exhausted = False
            while pending or not exhausted:
                while len(pending) < effective_jobs and not exhausted:
                    if _system_memory_fraction() >= maximum_memory_fraction:
                        break
                    try:
                        dataset_plan = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    future = executor.submit(
                        _worker,
                        _payload(
                            protocol,
                            dataset_plan,
                            output_root=output_root,
                            n_bootstraps=n_bootstraps,
                            n_permutations=n_permutations,
                            run_loso=run_loso,
                            resample_jobs=resample_jobs,
                            maximum_memory_fraction=maximum_memory_fraction,
                            overwrite=overwrite,
                        ),
                    )
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
                            "status": "failed",
                            "reason_code": type(error).__name__,
                            "n_bootstraps": n_bootstraps,
                            "n_permutations": n_permutations,
                            "run_loso": run_loso,
                            "planned_resamples": 0,
                            "successful_resamples": 0,
                            "elapsed_seconds": 0.0,
                            "peak_process_tree_rss_bytes": 0,
                            "result_directory": "",
                            "manifest_sha256": None,
                        }
                    records.append(record)
                    durations.append(float(record["elapsed_seconds"]))
                    outstanding = len(plans) - len(records)
                    window = durations[-min(20, len(durations)) :]
                    eta = sum(window) / len(window) * outstanding / effective_jobs
                    print(
                        json.dumps(
                            {
                                "event": "dataset_completed",
                                "dataset_id": record["dataset_id"],
                                "status": record["status"],
                                "completed": len(records),
                                "total": len(plans),
                                "eta_seconds": eta,
                                "system_memory_fraction": (_system_memory_fraction()),
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
                        n_bootstraps=n_bootstraps,
                        n_permutations=n_permutations,
                        run_loso=run_loso,
                        resample_jobs=resample_jobs,
                        effective_jobs=effective_jobs,
                        started_utc=started_utc,
                        status="running",
                    )
    _aggregate_outputs(output_root, records)
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
        n_bootstraps=n_bootstraps,
        n_permutations=n_permutations,
        run_loso=run_loso,
        resample_jobs=resample_jobs,
        effective_jobs=effective_jobs,
        started_utc=started_utc,
        status=final_status,
    )
    return {
        "status": final_status,
        "datasets": len(plans),
        "completed": sum(record["status"] == "completed" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "effective_dataset_jobs": effective_jobs,
        "output_root": str(output_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, required=True)
    parser.add_argument("--permutations", type=int, required=True)
    parser.add_argument("--no-loso", action="store_true")
    parser.add_argument("--maximum-replicates", type=int)
    parser.add_argument("--maximum-datasets", type=int)
    parser.add_argument("--dgp-family", action="append", dest="dgp_families")
    parser.add_argument(
        "--design-kind",
        action="append",
        choices=DESIGN_KINDS,
        dest="design_kinds",
    )
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--resample-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    for name, value in _THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    summary = run_v7_full_refit_campaign(
        protocol_path=arguments.protocol,
        phase=arguments.phase,
        output_root=arguments.output_dir,
        n_bootstraps=arguments.bootstraps,
        n_permutations=arguments.permutations,
        run_loso=not arguments.no_loso,
        maximum_replicates=arguments.maximum_replicates,
        maximum_datasets=arguments.maximum_datasets,
        dgp_families=arguments.dgp_families,
        design_kinds=arguments.design_kinds,
        jobs=arguments.jobs,
        resample_jobs=arguments.resample_jobs,
        overwrite=arguments.overwrite,
        allow_dirty=arguments.allow_dirty,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
