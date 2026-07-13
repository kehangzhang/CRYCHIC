"""Freeze multi-condition benchmark metrics into the report input contract.

The finalizer is intentionally a metrics and provenance stage. It consumes
already materialized adapter outputs, never reruns a communication method, and
never interprets real-data supportive observations as edge-level truth.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from benchmarks.metrics.multicondition import (
    EDGE_KEYS,
    METHOD_IDENTITY_KEYS,
    SYNTHETIC_TRUTH_SCOPES,
    aggregate_loso_primary_endpoint,
    cross_method_concordance,
    external_long_to_score_table,
    paired_differential_loso_reproducibility,
    paired_edge_effects,
    score_coverage_summary,
    summarize_loso_primary_endpoint,
    summarize_run_performance,
    synthetic_edge_truth_metrics,
    unpaired_differential_split_half_reproducibility,
    unpaired_edge_effects,
    unpaired_leave_one_subject_influence,
    within_context_reproducibility,
)
from benchmarks.metrics.multicondition_rank_stability import (
    RankStabilityParameters,
    evaluate_multicondition_rank_stability,
    not_estimable_rank_stability,
)

SPEC_SCHEMA_VERSION = "crychic-multicondition-finalize-v1"
REPORT_INPUT_SCHEMA_VERSION = "multicondition-report-inputs.v1"
SCORE_INDEX_SCHEMA_VERSION = "crychic-unified-score-index-v1"
FINALIZATION_SCHEMA_VERSION = "crychic-multicondition-finalization-v1"

METRIC_FILES: dict[str, str] = {
    "dataset_design": "dataset_design.tsv",
    "coverage": "coverage_summary.tsv",
    "loso_primary": "loso_primary_endpoint.tsv",
    "stability": "stability_summary.tsv",
    "concordance": "concordance_summary.tsv",
    "performance": "performance_summary.tsv",
    "biology_support": "biology_support.tsv",
    "simulation_truth": "simulation_truth_metrics.tsv",
    "iteration_comparison": "iteration_comparison.tsv",
    "ranking_agreement": "ranking_agreement_summary.tsv",
    "ranking_top_k_curve": "ranking_top_k_stability_curve.tsv",
    "ranking_intervals": "bootstrap_rank_intervals.tsv",
    "ranking_tiers": "stable_ranking_tiers.tsv",
}

EXTERNAL_LR_COLUMNS = (
    "run_id",
    "dataset_id",
    "method_id",
    "method_version",
    "analysis_track",
    "resource_id",
    "resource_version",
    "resource_mode",
    "universe_id",
    "universe_member",
    "universe_size",
    "sample_id",
    "subject_id",
    "context_json",
    *EDGE_KEYS,
    "score",
    "score_name",
    "score_direction",
    "status",
)

TRACK_METADATA_COLUMNS = (
    "run_id",
    "dataset_id",
    "method_id",
    "method_version",
    "analysis_track",
    "resource_id",
    "resource_version",
    "resource_mode",
    "universe_id",
    "score_name",
)

PARQUET_METADATA_BATCH_SIZE = 65_536

REAL_TRUTH_METRIC_NAMES = frozenset(
    {
        "auroc",
        "auc",
        "auprc",
        "average_precision",
        "precision_recall_auc",
    }
)

BIOLOGY_SUPPORT_STATUSES = frozenset(
    {
        "supported",
        "partial",
        "discordant",
        "not_covered",
        "not_estimable",
        "not_evaluated",
    }
)
PERFORMANCE_ROLES = frozenset(
    {
        "method_total",
        "core_fit",
        "source_pipeline_total",
        "adapter_readback",
        "excluded",
    }
)
METHOD_RUNTIME_ROLES = frozenset(
    {"method_total", "core_fit", "source_pipeline_total"}
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    design: Mapping[str, Any]
    comparison: Mapping[str, Any]
    context_key: str
    truth_scope: str
    simulation_truth: Path | None
    dataset_manifest: Path | None
    adapter_runs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RunIdentity:
    dataset: str
    method: str
    method_version: str
    analysis_track: str
    resource: str
    resource_version: str
    resource_mode: str
    score_semantics: str
    universe_id: str
    contrast: str

    def metric_values(self) -> dict[str, str]:
        """Return the identity columns shared by score-derived metrics."""
        return {
            "dataset": self.dataset,
            "method": self.method,
            "method_version": self.method_version,
            "analysis_track": self.analysis_track,
            "resource": self.resource,
            "resource_version": self.resource_version,
            "resource_mode": self.resource_mode,
            "score_semantics": self.score_semantics,
            "universe_id": self.universe_id,
            "contrast": self.contrast,
        }


@dataclass(frozen=True)
class ScoreView:
    run_id: str
    label: str
    role: str
    include: bool
    contrast: str
    contrast_candidate: str | None
    explicit: bool

    @property
    def primary(self) -> bool:
        return self.include and self.role == "primary"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return cast(dict[str, Any], payload)


def _resolve(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _optional_path(root: Path, value: object, *, field: str) -> Path | None:
    if value is None:
        return None
    return _resolve(root, value, field=field)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"unsupported table format: {path}")


def _canonical_run_id(value: object, *, field_name: str = "run_id") -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must contain canonical non-empty strings")
    return value


def _validate_parquet_adapter_schema(
    parquet: Any,
    *,
    require_universe_member: bool = False,
) -> None:
    """Reject adapter identifiers and membership flags with lossy Arrow types."""

    schema = parquet.schema_arrow
    if "run_id" not in schema.names:
        raise ValueError("adapter long table is missing required column: 'run_id'")
    run_id_type = schema.field("run_id").type
    if not (pa.types.is_string(run_id_type) or pa.types.is_large_string(run_id_type)):
        raise ValueError(
            "adapter long table run_id must use an Arrow string type; "
            f"got {run_id_type}"
        )
    if not require_universe_member:
        return
    if "universe_member" not in schema.names:
        raise ValueError(
            "adapter long table is missing required column: 'universe_member'"
        )
    universe_type = schema.field("universe_member").type
    if not pa.types.is_boolean(universe_type):
        raise ValueError(
            "adapter long table universe_member must use an Arrow boolean type; "
            f"got {universe_type}"
        )


def _validate_universe_member_values(values: pd.Series) -> None:
    """Require a null-free logical boolean before any dtype conversion."""

    if values.isna().any() or not pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(
            "adapter long table universe_member must be null-free boolean values"
        )


def _read_unique_parquet_metadata(
    path: Path,
    columns: Sequence[str],
    *,
    batch_size: int = PARQUET_METADATA_BATCH_SIZE,
) -> pd.DataFrame:
    """Scan identity metadata with memory bounded by one Arrow batch.

    Adapter long tables repeat identity metadata for every edge. Reading those
    columns through pandas still materializes one value per edge, which can be
    a large allocation before any score view is selected. Arrow computes unique
    scalars per batch; non-view identity columns are checked immediately and a
    compact one-row-per-run frame is returned to the existing validators.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(path)
    missing = set(columns).difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(
            f"adapter long table is missing metadata columns: {sorted(missing)}"
        )
    if "run_id" not in columns:
        raise ValueError("metadata projection must include run_id")
    _validate_parquet_adapter_schema(parquet)
    unique: dict[str, dict[tuple[object, ...], object]] = {
        column: {} for column in columns
    }

    def value_key(value: object) -> tuple[object, ...]:
        if isinstance(value, float) and math.isnan(value):
            return ("missing", "nan")
        try:
            hash(value)
        except TypeError:
            return (type(value).__qualname__, repr(value))
        return (type(value).__qualname__, value)

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=list(columns),
        use_threads=True,
    ):
        for column in columns:
            values = pc.unique(batch.column(batch.schema.get_field_index(column)))
            for value in values.to_pylist():
                if column == "run_id":
                    value = _canonical_run_id(value)
                unique[column].setdefault(value_key(value), value)
            if column != "run_id" and len(unique[column]) > 1:
                raise ValueError(
                    f"adapter column {column!r} must contain one value"
                )
    run_ids = list(unique["run_id"].values())
    if not run_ids:
        return pd.DataFrame(columns=list(columns))
    data = {
        column: (
            run_ids
            if column == "run_id"
            else [next(iter(unique[column].values()))] * len(run_ids)
        )
        for column in columns
    }
    return pd.DataFrame(data, columns=list(columns))


def _read_parquet_score_view(
    path: Path,
    run_id: str,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read exactly one score view using a Parquet predicate and projection."""
    run_id = _canonical_run_id(run_id, field_name="requested run_id")
    if columns is not None and "run_id" not in columns:
        raise ValueError("score-view projection must include run_id")
    parquet = pq.ParquetFile(path)
    _validate_parquet_adapter_schema(
        parquet,
        require_universe_member=(
            columns is not None and "universe_member" in columns
        ),
    )
    frame = pd.read_parquet(
        path,
        columns=None if columns is None else list(columns),
        filters=[("run_id", "==", run_id)],
        dtype_backend="pyarrow",
    )
    if frame.empty:
        raise ValueError(f"adapter score view {run_id!r} contains no rows")
    observed = tuple(_canonical_run_id(value) for value in frame["run_id"])
    if any(value != run_id for value in observed):
        raise RuntimeError(f"Parquet score-view boundary failed for {run_id!r}")
    if "universe_member" in frame:
        _validate_universe_member_values(frame["universe_member"])
    return frame


def _write_tsv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    result = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(part for part in result.split("_") if part) or "unknown"


def _one_text(table: pd.DataFrame, column: str) -> str:
    values = table[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"adapter column {column!r} must contain one value")
    return str(values.iloc[0])


def _manifest_value(payload: Mapping[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _manifest_identity(
    dataset: DatasetSpec,
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> RunIdentity:
    identity = run.get("identity", {})
    if not isinstance(identity, Mapping):
        raise ValueError("adapter_runs[].identity must be an object")

    def choose(
        name: str,
        manifest_paths: Sequence[tuple[str, ...]],
        default: str,
    ) -> str:
        value = identity.get(name)
        if value is None:
            for manifest_path in manifest_paths:
                value = _manifest_value(manifest, *manifest_path)
                if value is not None:
                    break
        return str(default if value is None else value)

    method = choose("method", (("method", "id"),), "unknown_method")
    analysis_default = (
        "ligand_target_program" if "nichenet" in method.lower() else "lr_stlr"
    )
    return RunIdentity(
        dataset=dataset.dataset,
        method=method,
        method_version=choose("method_version", (("method", "version"),), "unknown"),
        analysis_track=choose(
            "analysis_track", (("analysis_track",),), analysis_default
        ),
        resource=choose(
            "resource",
            (("resource", "resource_id"), ("resource", "id")),
            "unknown",
        ),
        resource_version=choose(
            "resource_version", (("resource", "version"),), "unknown"
        ),
        resource_mode=choose("resource_mode", (("resource", "mode"),), "native"),
        score_semantics=choose(
            "score_semantics",
            (
                ("score_semantics", "primary_score"),
                ("score_semantics", "name"),
            ),
            "unknown",
        ),
        universe_id=choose(
            "universe_id", (("output", "universe_id"),), "not_available"
        ),
        contrast=str(dataset.comparison.get("contrast", "not_available")),
    )


def _load_spec(path: Path) -> tuple[dict[str, Any], tuple[DatasetSpec, ...]]:
    payload = _read_json(path)
    schema = payload.get("schema_version")
    if schema != SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"run specification schema must be {SPEC_SCHEMA_VERSION!r}; got {schema!r}"
        )
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("run specification datasets must be a non-empty list")
    root = path.parent
    datasets: list[DatasetSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_datasets):
        if not isinstance(raw, Mapping):
            raise ValueError(f"datasets[{index}] must be an object")
        dataset = raw.get("dataset_id")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"datasets[{index}].dataset_id must be a non-empty string")
        if dataset in seen:
            raise ValueError(f"duplicate dataset_id: {dataset!r}")
        seen.add(dataset)
        design = raw.get("design")
        comparison = raw.get("comparison")
        adapter_runs = raw.get("adapter_runs", [])
        if not isinstance(design, Mapping):
            raise ValueError(f"datasets[{index}].design must be an object")
        if not isinstance(comparison, Mapping):
            raise ValueError(f"datasets[{index}].comparison must be an object")
        if not isinstance(adapter_runs, list) or not all(
            isinstance(item, Mapping) for item in adapter_runs
        ):
            raise ValueError(
                f"datasets[{index}].adapter_runs must be a list of objects"
            )
        design_type = comparison.get("design")
        if design_type not in {"paired", "unpaired", "unsupported"}:
            raise ValueError(
                f"datasets[{index}].comparison.design must be paired, unpaired, "
                "or unsupported"
            )
        context_key = raw.get("context_key")
        if not isinstance(context_key, str) or not context_key:
            raise ValueError(f"datasets[{index}].context_key must be a string")
        if design_type != "unsupported":
            for field in ("reference", "target", "contrast"):
                if not isinstance(comparison.get(field), str) or not comparison[field]:
                    raise ValueError(
                        f"datasets[{index}].comparison.{field} must be a string"
                    )
        truth_scope = str(raw.get("truth_scope", "real_data"))
        simulation_truth = _optional_path(
            root,
            raw.get("simulation_truth"),
            field=f"datasets[{index}].simulation_truth",
        )
        if simulation_truth is not None:
            if truth_scope not in SYNTHETIC_TRUTH_SCOPES:
                raise ValueError(
                    "AUROC/AUPRC are forbidden for real datasets; simulation_truth "
                    f"requires an explicit synthetic truth_scope for {dataset!r}"
                )
            if not simulation_truth.is_file():
                raise FileNotFoundError(simulation_truth)
        dataset_manifest = _optional_path(
            root,
            raw.get("dataset_manifest"),
            field=f"datasets[{index}].dataset_manifest",
        )
        if dataset_manifest is not None and not dataset_manifest.is_file():
            raise FileNotFoundError(dataset_manifest)
        datasets.append(
            DatasetSpec(
                dataset=dataset,
                design=cast(Mapping[str, Any], design),
                comparison=cast(Mapping[str, Any], comparison),
                context_key=context_key,
                truth_scope=truth_scope,
                simulation_truth=simulation_truth,
                dataset_manifest=dataset_manifest,
                adapter_runs=tuple(cast(Sequence[Mapping[str, Any]], adapter_runs)),
            )
        )
    return payload, tuple(datasets)


def _parameters(payload: Mapping[str, Any]) -> dict[str, int]:
    raw = payload.get("parameters", {})
    if not isinstance(raw, Mapping):
        raise ValueError("parameters must be an object")
    defaults = {
        "top_k": 100,
        "minimum_shared_edges": 20,
        "n_bootstrap": 2000,
        "n_split_repeats": 200,
        "min_subjects_per_half": 2,
        "random_seed": 20260712,
    }
    result: dict[str, int] = {}
    for name, default in defaults.items():
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"parameters.{name} must be an integer")
        minimum = 0 if name == "random_seed" else 1
        if name in {"minimum_shared_edges", "n_split_repeats", "min_subjects_per_half"}:
            minimum = 2
        if name == "minimum_shared_edges":
            minimum = 3
        if value < minimum:
            raise ValueError(f"parameters.{name} must be >= {minimum}")
        result[name] = value
    return result


def _dataset_design_table(datasets: Sequence[DatasetSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scale_fields = ("n_cells", "n_samples", "n_subjects", "n_contexts", "n_cell_types")
    for dataset in datasets:
        design_type = str(dataset.comparison["design"])
        row: dict[str, Any] = {"dataset": dataset.dataset}
        row.update(dataset.design)
        for field in scale_fields:
            row.setdefault(field, np.nan)
        row.setdefault("design_type", design_type)
        row.setdefault(
            "status", "observed" if design_type != "unsupported" else "not_estimable"
        )
        row.setdefault(
            "reason_code",
            None if design_type != "unsupported" else "unsupported_comparison_design",
        )
        row["truth_scope"] = dataset.truth_scope
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_source_hash(path: Path, manifest: Mapping[str, Any]) -> str:
    digest = _sha256(path)
    declared = _manifest_value(manifest, "output", "sha256")
    if declared is not None and str(declared) != digest:
        raise ValueError(
            f"adapter output SHA256 disagrees with manifest for {path}: "
            f"declared={declared}, observed={digest}"
        )
    return digest


def _identity_from_track_metadata(
    metadata: pd.DataFrame, dataset: DatasetSpec
) -> RunIdentity:
    if _one_text(metadata, "dataset_id") != dataset.dataset:
        raise ValueError("adapter dataset_id disagrees with run specification")
    return RunIdentity(
        dataset=dataset.dataset,
        method=_one_text(metadata, "method_id"),
        method_version=_one_text(metadata, "method_version"),
        analysis_track=_one_text(metadata, "analysis_track"),
        resource=_one_text(metadata, "resource_id"),
        resource_version=_one_text(metadata, "resource_version"),
        resource_mode=_one_text(metadata, "resource_mode"),
        score_semantics=_one_text(metadata, "score_name"),
        universe_id=_one_text(metadata, "universe_id"),
        contrast=str(dataset.comparison.get("contrast", "not_available")),
    )


def _manifest_score_views(
    manifest: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    raw = _manifest_value(manifest, "source_result", "score_views")
    if raw is None:
        raw = manifest.get("score_views")
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise ValueError("adapter manifest score_views must be a list")
    result: dict[str, tuple[str, ...]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or "run_id" not in item:
            raise ValueError("adapter manifest score_views require run_id")
        run_id = str(item["run_id"])
        candidates = item.get("contrast_candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("score view contrast_candidates must be a list")
        result[run_id] = tuple(map(str, candidates))
    return result


def _manifest_score_view_record(
    manifest: Mapping[str, Any], run_id: str
) -> Mapping[str, Any] | None:
    raw = _manifest_value(manifest, "source_result", "score_views")
    if raw is None:
        raw = manifest.get("score_views")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("adapter manifest score_views must be a list")
    matches = [
        item
        for item in raw
        if isinstance(item, Mapping) and str(item.get("run_id")) == run_id
    ]
    if len(matches) > 1:
        raise ValueError(f"adapter manifest contains duplicate score view {run_id!r}")
    return cast(Mapping[str, Any] | None, matches[0] if matches else None)


def _rank_scope_for_view(
    run: Mapping[str, Any], manifest: Mapping[str, Any], view: ScoreView
) -> tuple[str, str | None]:
    requested = run.get("rank_scope")
    if requested is not None and str(requested) not in {
        "global_common_functional",
        "within_receiver_macro",
        "not_estimable",
    }:
        raise ValueError(
            "adapter_runs[].rank_scope must be global_common_functional, "
            "within_receiver_macro, or not_estimable"
        )
    record = _manifest_score_view_record(manifest, view.run_id)
    common_claim = record.get("common_functional_claim") if record else None
    if common_claim is False:
        if requested == "global_common_functional":
            raise ValueError(
                "common_functional_claim=false forbids global rank_scope"
            )
        # The current finalizer cannot apply one receiver-stratified estimand to
        # every primary, concordance, and supportive-biology endpoint yet.
        return (
            "not_estimable_receiver_child_functionals",
            "receiver_child_functionals_not_globally_comparable",
        )
    if requested == "within_receiver_macro":
        return (
            "not_estimable_within_receiver_pipeline_incomplete",
            "within_receiver_rank_scope_not_applied_to_all_endpoints",
        )
    if requested == "not_estimable":
        return "not_estimable_by_specification", "rank_scope_disabled_by_specification"
    return "global_common_functional", None


def _annotate_rank_scope(table: pd.DataFrame, rank_scope: str) -> pd.DataFrame:
    result = table.copy(deep=True)
    result["rank_scope"] = rank_scope
    return result


def _score_views(
    metadata: pd.DataFrame,
    dataset: DatasetSpec,
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[ScoreView, ...]:
    observed = tuple(sorted(metadata["run_id"].astype(str).unique()))
    if not observed:
        raise ValueError("adapter long table has no run_id values")
    manifest_views = _manifest_score_views(manifest)
    raw = run.get("score_views")
    if raw is None:
        if len(observed) > 1:
            raise ValueError(
                "adapter long table contains multiple run_id score views; "
                "adapter_runs[].score_views must map every view explicitly"
            )
        candidates = manifest_views.get(observed[0], ())
        candidate = candidates[0] if len(candidates) == 1 else None
        return (
            ScoreView(
                run_id=observed[0],
                label="default",
                role="primary",
                include=True,
                contrast=str(dataset.comparison["contrast"]),
                contrast_candidate=candidate,
                explicit=False,
            ),
        )
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise ValueError("adapter_runs[].score_views must be a list of objects")
    entries = cast(Sequence[Mapping[str, Any]], raw)
    views: list[ScoreView] = []
    for index, item in enumerate(entries):
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"score_views[{index}].run_id must be a string")
        role = str(item.get("role", "sensitivity"))
        if role not in {"primary", "sensitivity", "excluded"}:
            raise ValueError(
                f"score_views[{index}].role must be primary, sensitivity, or excluded"
            )
        include = item.get("include", role != "excluded")
        if not isinstance(include, bool):
            raise ValueError(f"score_views[{index}].include must be boolean")
        if role == "excluded" and include:
            raise ValueError("an excluded score view cannot have include=true")
        contrast = item.get("contrast", dataset.comparison.get("contrast"))
        if not isinstance(contrast, str) or not contrast:
            raise ValueError(f"score_views[{index}].contrast must be a string")
        candidate_value = item.get("contrast_candidate")
        candidate = None if candidate_value is None else str(candidate_value)
        declared = manifest_views.get(run_id, ())
        if declared and candidate not in declared:
            raise ValueError(
                f"score view {run_id!r} contrast_candidate must be one of "
                f"manifest candidates {list(declared)!r}"
            )
        label = str(item.get("label", candidate or run_id))
        if not label:
            raise ValueError(f"score_views[{index}].label must be non-empty")
        views.append(
            ScoreView(
                run_id=run_id,
                label=label,
                role=role,
                include=include,
                contrast=contrast,
                contrast_candidate=candidate,
                explicit=True,
            )
        )
    mapped_ids = tuple(sorted(view.run_id for view in views))
    if len(mapped_ids) != len(set(mapped_ids)):
        raise ValueError("adapter_runs[].score_views contains duplicate run_id values")
    if mapped_ids != observed:
        raise ValueError(
            "adapter_runs[].score_views must exactly map observed run_id values; "
            f"observed={list(observed)!r}, mapped={list(mapped_ids)!r}"
        )
    primary = [view for view in views if view.primary]
    if len(primary) != 1:
        raise ValueError("exactly one included score view must have role='primary'")
    labels = [view.label for view in views if view.include]
    if len(labels) != len(set(labels)):
        raise ValueError("included score view labels must be unique")
    if len(observed) > 1:
        preregistered = dataset.comparison.get("primary_score_view")
        if not isinstance(preregistered, str) or not preregistered:
            raise ValueError(
                "multi-view adapters require comparison.primary_score_view"
            )
        if primary[0].contrast_candidate != preregistered:
            raise ValueError(
                "the primary score view does not match the preregistered "
                f"comparison.primary_score_view={preregistered!r}"
            )
    return tuple(views)


def _view_identity(
    base: RunIdentity,
    view: ScoreView,
    *,
    multiple_views: bool,
) -> RunIdentity:
    semantics = base.score_semantics
    if multiple_views or view.explicit:
        semantics = f"{semantics}|score_view={view.label}"
    return RunIdentity(
        dataset=base.dataset,
        method=base.method,
        method_version=base.method_version,
        analysis_track=base.analysis_track,
        resource=base.resource,
        resource_version=base.resource_version,
        resource_mode=base.resource_mode,
        score_semantics=semantics,
        universe_id=base.universe_id,
        contrast=view.contrast,
    )


def _annotate_view(table: pd.DataFrame, view: ScoreView) -> pd.DataFrame:
    result = table.copy(deep=True)
    result["score_view_run_id"] = view.run_id
    result["score_view_label"] = view.label
    result["score_view_role"] = view.role
    result["score_view_contrast_candidate"] = view.contrast_candidate
    result["score_view_primary"] = view.primary
    return result


def _annotate_truth_scope(
    table: pd.DataFrame,
    dataset_truth_scopes: Mapping[str, str],
) -> pd.DataFrame:
    """Attach the frozen dataset truth scope without overwriting explicit scope."""
    result = table.copy(deep=True)
    if result.empty:
        if "truth_scope" not in result:
            result["truth_scope"] = pd.Series(dtype=str)
        return result
    if "dataset" not in result:
        raise ValueError("truth-scoped metric tables require a dataset column")
    mapped = result["dataset"].astype(str).map(dataset_truth_scopes)
    if "truth_scope" in result:
        supplied = result["truth_scope"].notna()
        conflict = (
            supplied
            & mapped.notna()
            & result["truth_scope"].astype(str).ne(mapped)
        )
        if conflict.any():
            datasets = sorted(result.loc[conflict, "dataset"].astype(str).unique())
            raise ValueError(
                "metric truth_scope disagrees with the frozen dataset scope for "
                f"datasets: {datasets}"
            )
        resolved = result["truth_scope"].where(supplied, mapped)
    else:
        resolved = mapped
    if resolved.isna().any():
        datasets = sorted(result.loc[resolved.isna(), "dataset"].astype(str).unique())
        raise ValueError(f"metric rows reference unknown dataset scopes: {datasets}")
    result["truth_scope"] = resolved.astype(str)
    return result


def _not_estimable_coverage(identity: RunIdentity, reason: str) -> dict[str, Any]:
    return identity.metric_values() | {
        "n_samples": 0,
        "n_subjects": 0,
        "frozen_universe_edges": np.nan,
        "resource_covered_edges": np.nan,
        "resource_coverage_fraction": np.nan,
        "eligible_universe_rows": 0,
        "comparison_eligible_rows": 0,
        "comparison_coverage_fraction": np.nan,
        "observed_rows": 0,
        "not_predicted_rows": 0,
        "missing_rows": 0,
        "cell_type_missing_rows": 0,
        "not_estimable_rows": 0,
        "failed_rows": 0,
        "not_supported_rows": 0,
        "median_sample_comparison_coverage": np.nan,
        "minimum_sample_comparison_coverage": np.nan,
        "status": "not_estimable",
        "reason_code": reason,
    }


def _not_estimable_primary(identity: RunIdentity, reason: str) -> dict[str, Any]:
    return identity.metric_values() | {
        "endpoint": "not_estimable",
        "endpoint_scope": "dataset",
        "estimate": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "confidence_level": 0.95,
        "n_subjects_total": 0,
        "n_subjects_estimable": 0,
        "status": "not_estimable",
        "reason_code": reason,
    }


def _not_estimable_stability(identity: RunIdentity, reason: str) -> dict[str, Any]:
    return identity.metric_values() | {
        "context": "all",
        "metric": "lr_reproducibility",
        "estimate": np.nan,
        "status": "not_estimable",
        "reason_code": reason,
    }


def _stability_long(table: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "median_spearman",
        "median_top_k_jaccard",
        "effect_spearman",
        "mean_direction_agreement",
        "valid_repeat_fraction",
    )
    rows: list[dict[str, Any]] = []
    for raw_record in table.to_dict(orient="records"):
        record = cast(dict[str, Any], raw_record)
        for metric in metric_columns:
            if metric not in record:
                continue
            value = pd.to_numeric(pd.Series([record[metric]]), errors="coerce").iloc[0]
            row = dict(record)
            row["metric"] = metric
            row["estimate"] = value
            if not np.isfinite(value):
                row["status"] = "not_estimable"
                row["reason_code"] = row.get("reason_code") or f"{metric}_not_estimable"
            rows.append(row)
    return pd.DataFrame(rows)


def _influence_summary(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    grouping = [*METHOD_IDENTITY_KEYS, "contrast"]
    metrics = (
        "family_macro_rank_similarity_to_full",
        "direction_agreement_to_full",
        "mean_absolute_effect_change",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in table.groupby(grouping, sort=False, observed=True):
        observed = group.loc[group["status"].eq("descriptive")]
        identity = dict(zip(grouping, keys, strict=True))
        for metric in metrics:
            values = pd.to_numeric(observed[metric], errors="coerce")
            values = values.loc[np.isfinite(values)]
            rows.append(
                identity
                | {
                    "context": "leave_one_subject",
                    "metric": f"influence_{metric}",
                    "estimate": float(values.median()) if len(values) else np.nan,
                    "n_exclusions_estimable": len(values),
                    "interpretation": (
                        "influence_diagnostic_not_independent_replication"
                    ),
                    "status": "observed" if len(values) else "not_estimable",
                    "reason_code": None if len(values) else "no_estimable_exclusions",
                }
            )
    return pd.DataFrame(rows)


def _unpaired_primary(split: pd.DataFrame) -> pd.DataFrame:
    result = split.copy(deep=True)
    result["endpoint_scope"] = "dataset"
    result["n_subjects_total"] = pd.to_numeric(
        result["n_reference_subjects"], errors="coerce"
    ) + pd.to_numeric(result["n_target_subjects"], errors="coerce")
    result["n_subjects_estimable"] = np.where(
        result["status"].eq("observed"), result["n_subjects_total"], 0
    )
    return result


def _paired_primary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy(deep=True)
    result["endpoint_scope"] = "dataset"
    return result


def _cross_dataset_primary(
    loso: pd.DataFrame, parameters: Mapping[str, int]
) -> pd.DataFrame:
    if "truth_scope" not in loso:
        raise ValueError("cross-dataset LOSO aggregation requires truth_scope")
    real_loso = loso.loc[
        ~loso["truth_scope"].astype(str).isin(SYNTHETIC_TRUTH_SCOPES)
    ].copy()
    if real_loso.empty:
        return pd.DataFrame()
    result = aggregate_loso_primary_endpoint(
        real_loso,
        n_bootstrap=parameters["n_bootstrap"],
        random_seed=parameters["random_seed"],
    )
    result["dataset"] = "__cross_dataset__"
    result["contrast"] = "prespecified_per_dataset"
    result["universe_id"] = "__multiple_frozen_universes__"
    result["endpoint_scope"] = "cross_dataset"
    result["n_subjects_total"] = result["n_subjects_estimable"]
    result["truth_scope"] = "real_data"
    result["rank_scope"] = "global_common_functional"
    return result


def _track_b_reason(identity: RunIdentity) -> str:
    return (
        "track_b_ligand_target_program_not_lr_stlr_comparable"
        if identity.analysis_track == "ligand_target_program"
        else "analysis_track_not_lr_stlr_comparable"
    )


def _manifest_output_bytes(
    manifest: Mapping[str, Any], manifest_path: Path | None
) -> int | None:
    declared = _manifest_value(manifest, "output", "bytes")
    if declared is not None:
        value = int(declared)
        if value < 0:
            raise ValueError("manifest output.bytes must be non-negative")
        return value
    table = _manifest_value(manifest, "output", "table")
    if isinstance(table, str) and table and manifest_path is not None:
        output_path = (manifest_path.parent / table).resolve()
        if output_path.is_file():
            return output_path.stat().st_size
    return None


def _derived_performance_record(
    identity: RunIdentity,
    manifest: Mapping[str, Any],
    long_path: Path | None,
    *,
    performance_role: str = "method_total",
    source_manifest_path: Path | None = None,
    source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if performance_role not in PERFORMANCE_ROLES:
        raise ValueError(
            f"performance_role must be one of {sorted(PERFORMANCE_ROLES)}"
        )
    status = str(manifest.get("status", "failed"))
    if status not in {"complete", "failed", "not_supported", "skipped"}:
        status = "failed"
    threads = _manifest_value(manifest, "environment", "threads")
    return {
        "dataset": identity.dataset,
        "method": identity.method,
        "method_version": identity.method_version,
        "analysis_track": identity.analysis_track,
        "resource": identity.resource,
        "resource_version": identity.resource_version,
        "resource_mode": identity.resource_mode,
        "run_id": str(manifest.get("run_id", f"manifest_{_slug(identity.method)}")),
        "status": status,
        "wall_time_seconds": manifest.get("elapsed_seconds"),
        "peak_rss_mb": _manifest_value(manifest, "performance", "peak_rss_mb"),
        "output_bytes": (
            long_path.stat().st_size
            if long_path and long_path.is_file()
            else _manifest_output_bytes(manifest, source_manifest_path)
        ),
        "threads": 1 if threads is None else threads,
        "determinism_key": _manifest_value(manifest, "parameters", "determinism_key"),
        "output_sha256": (
            _manifest_value(manifest, "output", "sha256")
            if status == "complete"
            else None
        ),
        "performance_role": performance_role,
        "include_in_method_runtime": performance_role in METHOD_RUNTIME_ROLES,
        "performance_source_manifest": (
            source_manifest_path.as_posix() if source_manifest_path else None
        ),
        "performance_source_manifest_sha256": source_manifest_sha256,
    }


def _performance_records_for_run(
    *,
    spec_root: Path,
    identity: RunIdentity,
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    long_path: Path | None,
) -> tuple[list[dict[str, Any]], tuple[Path, ...]]:
    """Resolve method-runtime versus adapter-only performance provenance."""
    role = str(run.get("performance_role", "method_total"))
    current_digest = _sha256(manifest_path)
    current = _derived_performance_record(
        identity,
        manifest,
        long_path,
        performance_role=role,
        source_manifest_path=manifest_path,
        source_manifest_sha256=current_digest,
    )
    override = run.get("performance_override")
    if override is None:
        return [current], ()
    if not isinstance(override, Mapping):
        raise ValueError("adapter_runs[].performance_override must be an object")
    unknown = set(override).difference(
        {"source_manifest", "source_role", "expected_sha256"}
    )
    if unknown:
        raise ValueError(
            "performance_override contains unsupported fields: "
            f"{sorted(unknown)}"
        )
    if role not in {"adapter_readback", "excluded"}:
        raise ValueError(
            "performance_override requires performance_role='adapter_readback' "
            "or 'excluded' so adapter elapsed time cannot also enter method runtime"
        )
    source_path = _resolve(
        spec_root,
        override.get("source_manifest"),
        field="adapter_runs[].performance_override.source_manifest",
    )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_digest = _sha256(source_path)
    expected_digest = override.get("expected_sha256")
    if expected_digest is not None and str(expected_digest) != source_digest:
        raise ValueError(
            "performance override source manifest SHA256 mismatch: "
            f"expected={expected_digest}, observed={source_digest}"
        )
    source_role = str(override.get("source_role", "source_pipeline_total"))
    if source_role not in METHOD_RUNTIME_ROLES:
        raise ValueError(
            "performance_override.source_role must identify a method runtime: "
            f"{sorted(METHOD_RUNTIME_ROLES)}"
        )
    source_manifest = _read_json(source_path)
    source_bindings = {
        "dataset": source_manifest.get("dataset_id"),
        "method": _manifest_value(source_manifest, "method", "id")
        or source_manifest.get("method_id"),
        "resource": _manifest_value(source_manifest, "resource", "id")
        or _manifest_value(source_manifest, "resource", "resource_id")
        or source_manifest.get("resource_id"),
        "resource_mode": _manifest_value(source_manifest, "resource", "mode")
        or source_manifest.get("resource_mode"),
    }
    expected_bindings = {
        "dataset": identity.dataset,
        "method": identity.method,
        "resource": identity.resource,
        "resource_mode": identity.resource_mode,
    }
    missing_bindings = [
        name for name, value in source_bindings.items() if value is None
    ]
    if missing_bindings:
        raise ValueError(
            "performance override source manifest lacks identity bindings: "
            f"{missing_bindings}"
        )
    mismatched_bindings = {
        name: {"expected": expected_bindings[name], "observed": str(value)}
        for name, value in source_bindings.items()
        if str(value) != expected_bindings[name]
    }
    if mismatched_bindings:
        raise ValueError(
            "performance override source manifest identity mismatch: "
            f"{mismatched_bindings}"
        )
    source = _derived_performance_record(
        identity,
        source_manifest,
        None,
        performance_role=source_role,
        source_manifest_path=source_path,
        source_manifest_sha256=source_digest,
    )
    source["run_id"] = f"{source['run_id']}:{source_role}"
    current["run_id"] = f"{current['run_id']}:{role}"
    return [source, current], (source_path,)


def _prepare_performance_runtime_input(records: pd.DataFrame) -> pd.DataFrame:
    """Exclude adapter-only timing while retaining an explicit NE identity."""
    prepared = records.copy(deep=True)
    if "performance_role" not in prepared:
        prepared["performance_role"] = "method_total"
    prepared["performance_role"] = (
        prepared["performance_role"].fillna("method_total").replace("", "method_total")
    )
    prepared["include_in_method_runtime"] = prepared["performance_role"].isin(
        METHOD_RUNTIME_ROLES
    )
    invalid_roles = set(prepared["performance_role"].astype(str)).difference(
        PERFORMANCE_ROLES
    )
    if invalid_roles:
        raise ValueError(f"unsupported performance roles: {sorted(invalid_roles)}")
    identity = [
        "dataset",
        "method",
        "method_version",
        "analysis_track",
        "resource",
        "resource_version",
        "resource_mode",
    ]
    runtime_rows: list[pd.DataFrame] = []
    for _, group in prepared.groupby(identity, sort=False, observed=True):
        included = group.loc[group["include_in_method_runtime"].astype(bool)].copy()
        included_roles = set(included["performance_role"].astype(str))
        if len(included_roles) > 1:
            raise ValueError(
                "one method identity cannot mix multiple included runtime roles: "
                f"{sorted(included_roles)}"
            )
        if included.empty:
            placeholder = group.iloc[[0]].copy()
            placeholder["status"] = "skipped"
            placeholder["wall_time_seconds"] = np.nan
            placeholder["peak_rss_mb"] = np.nan
            placeholder["output_bytes"] = np.nan
            placeholder["performance_role"] = "excluded"
            placeholder["run_id"] = (
                placeholder["run_id"].astype(str) + ":method_runtime_excluded"
            )
            runtime_rows.append(placeholder)
        else:
            runtime_rows.append(included)
    return pd.concat(runtime_rows, ignore_index=True, sort=False)


def _performance_component_summary(records: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "dataset",
        "method",
        "method_version",
        "analysis_track",
        "resource",
        "resource_version",
        "resource_mode",
    ]
    prepared = records.copy(deep=True)
    if "performance_role" not in prepared:
        prepared["performance_role"] = "method_total"
    prepared["performance_role"] = (
        prepared["performance_role"].fillna("method_total").replace("", "method_total")
    )
    prepared["include_in_method_runtime"] = prepared["performance_role"].isin(
        METHOD_RUNTIME_ROLES
    )
    rows: list[dict[str, Any]] = []
    for keys, group in prepared.groupby(identity, sort=False, observed=True):
        row = dict(zip(identity, keys, strict=True))
        complete = group.loc[group["status"].eq("complete")]
        roles = sorted(
            set(
                complete.loc[
                    complete["include_in_method_runtime"].astype(bool),
                    "performance_role",
                ].astype(str)
            )
        )
        row["method_runtime_role"] = roles[0] if len(roles) == 1 else "not_available"
        row["performance_component_count"] = len(group)
        for role in sorted(PERFORMANCE_ROLES):
            values = pd.to_numeric(
                complete.loc[
                    complete["performance_role"].eq(role), "wall_time_seconds"
                ],
                errors="coerce",
            ).dropna()
            row[f"median_{role}_wall_time_seconds"] = (
                float(values.median()) if len(values) else np.nan
            )
        source_digests = sorted(
            set(
                complete.get(
                    "performance_source_manifest_sha256", pd.Series(dtype=str)
                )
                .dropna()
                .astype(str)
            )
        )
        row["performance_source_manifest_sha256"] = ";".join(source_digests)
        rows.append(row)
    return pd.DataFrame(rows)


def _truth_observations(path: Path) -> pd.DataFrame:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("supportive biology truth must contain a YAML object")
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("supportive biology truth datasets must be an object")
    rows: list[dict[str, str]] = []
    for dataset, raw in datasets.items():
        if not isinstance(raw, Mapping):
            continue
        observations = raw.get("expected_observations", [])
        if not isinstance(observations, list):
            raise ValueError("expected_observations must be a list")
        for observation in observations:
            if not isinstance(observation, Mapping) or "id" not in observation:
                raise ValueError("every supportive observation requires an id")
            rows.append(
                {
                    "dataset": str(dataset),
                    "observation_id": str(observation["id"]),
                    "expected_direction": str(
                        observation.get("direction", "not_directional")
                    ),
                    "expected_metric": str(
                        observation.get("metric", "supportive_only")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _biology_dataset_aliases(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = payload.get("supportive_biology_dataset_aliases", {})
    if not isinstance(raw, Mapping):
        raise ValueError("supportive_biology_dataset_aliases must be an object")
    aliases = {str(source): str(target) for source, target in raw.items()}
    if any(not source or not target for source, target in aliases.items()):
        raise ValueError("supportive biology dataset aliases must be non-empty")
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("supportive biology dataset aliases must be one-to-one")
    return aliases


def _biology_support_table(
    truth_path: Path,
    evidence_path: Path | None,
    variants: pd.DataFrame,
    *,
    dataset_aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    truth = _truth_observations(truth_path)
    if truth.empty:
        return pd.DataFrame(
            [
                {
                    "dataset": "__none__",
                    "observation_id": "no_locked_observations",
                    "method": "No method result",
                    "resource_mode": "not_available",
                    "support_status": "not_evaluated",
                    "observed_direction": "",
                    "evidence_note": "supportive_biology_truth_has_no_observations",
                    "status": "not_estimable",
                    "reason_code": "supportive_biology_truth_has_no_observations",
                }
            ]
        )
    aliases = dict(dataset_aliases or {})
    unknown_aliases = set(aliases).difference(truth["dataset"].astype(str))
    if unknown_aliases:
        raise ValueError(
            "supportive biology dataset aliases reference unknown locked datasets: "
            f"{sorted(unknown_aliases)}"
        )
    truth["truth_dataset"] = truth["dataset"].astype(str)
    truth["dataset"] = truth["dataset"].astype(str).replace(aliases)
    if truth.duplicated(["dataset", "observation_id"]).any():
        raise ValueError("supportive biology aliases create duplicate observations")

    variant_columns = ["dataset", "method", "resource_mode"]
    if variants.empty or "rank_scope" not in variants:
        rank_scope_policy = pd.DataFrame(
            columns=[*variant_columns, "rank_scope"]
        )
    else:
        rank_scope_policy = variants.loc[
            :, [*variant_columns, "rank_scope"]
        ].dropna(subset=["rank_scope"]).drop_duplicates()
        duplicated_scope = rank_scope_policy.duplicated(
            variant_columns, keep=False
        )
        if duplicated_scope.any():
            raise ValueError(
                "one biology method/resource variant cannot mix rank_scope values"
            )
    if variants.empty:
        variants = pd.DataFrame(columns=variant_columns)
    else:
        variants = variants.loc[:, variant_columns].drop_duplicates()
    key = ["dataset", "observation_id", "method", "resource_mode"]
    if evidence_path is None:
        evidence = pd.DataFrame(
            columns=[
                *key,
                "support_status",
                "observed_direction",
                "evidence_note",
                "status",
                "reason_code",
            ]
        )
    else:
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        evidence = _read_table(evidence_path)
        missing = set(key).difference(evidence.columns)
        if missing:
            raise ValueError(
                f"biology support evidence is missing columns: {sorted(missing)}"
            )
        if evidence.duplicated(key).any():
            raise ValueError("biology support evidence contains duplicate rows")
        evidence = evidence.copy(deep=True)
        evidence["dataset"] = evidence["dataset"].astype(str).replace(aliases)
        if "rank_scope" in evidence:
            evidence = evidence.rename(
                columns={"rank_scope": "evidence_rank_scope"}
            )
        else:
            evidence["evidence_rank_scope"] = "global_common_functional"
        if evidence.duplicated(key).any():
            raise ValueError(
                "supportive biology aliases create duplicate evidence rows"
            )
        unknown = set(
            evidence[["dataset", "observation_id"]].itertuples(index=False, name=None)
        ).difference(
            set(truth[["dataset", "observation_id"]].itertuples(index=False, name=None))
        )
        if unknown:
            raise ValueError(
                "biology support evidence contains unlocked observations: "
                f"{sorted(unknown)}"
            )
        invalid = set(
            evidence.get("support_status", pd.Series(dtype=str)).astype(str)
        ).difference(BIOLOGY_SUPPORT_STATUSES)
        if invalid:
            raise ValueError(
                "biology support evidence has invalid support_status: "
                f"{sorted(invalid)}"
            )
        evidence_variants = evidence.loc[:, variant_columns].drop_duplicates()
        variants = pd.concat(
            [variants, evidence_variants], ignore_index=True
        ).drop_duplicates(ignore_index=True)

    grids: list[pd.DataFrame] = []
    for dataset, observations in truth.groupby("dataset", sort=False, observed=True):
        methods = variants.loc[variants["dataset"].eq(dataset)]
        if methods.empty:
            methods = pd.DataFrame(
                [
                    {
                        "dataset": dataset,
                        "method": "No method result",
                        "resource_mode": "not_available",
                    }
                ]
            )
        grid = observations.merge(
            methods, on="dataset", how="inner", validate="many_to_many"
        )
        grids.append(grid)
    grid = pd.concat(grids, ignore_index=True)
    result = grid.merge(evidence, on=key, how="left", validate="one_to_one")
    result = result.merge(
        rank_scope_policy.rename(columns={"rank_scope": "benchmark_rank_scope"}),
        on=variant_columns,
        how="left",
        validate="many_to_one",
    )
    if "evidence_rank_scope" not in result:
        result["evidence_rank_scope"] = "not_recorded"
    result["evidence_rank_scope"] = result["evidence_rank_scope"].fillna(
        "not_recorded"
    )
    result["rank_scope"] = result["benchmark_rank_scope"].fillna(
        result["evidence_rank_scope"].replace("not_recorded", pd.NA)
    ).fillna(
        "annotation_only_unbound_rank_scope"
    )
    defaults: dict[str, str] = {
        "support_status": "not_evaluated",
        "observed_direction": "",
        "evidence_note": "supportive_evidence_not_supplied",
        "status": "not_estimable",
        "reason_code": "supportive_evidence_not_supplied",
        "source_biology_file": "",
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
        else:
            result[column] = result[column].fillna(default)
    invalid_scope = result["rank_scope"].astype(str).str.startswith("not_estimable")
    result.loc[invalid_scope, "support_status"] = "not_estimable"
    result.loc[invalid_scope, "status"] = "not_estimable"
    result.loc[invalid_scope, "reason_code"] = (
        "receiver_child_functionals_not_globally_comparable"
    )
    result.loc[invalid_scope, "evidence_note"] = (
        "supportive_biology_evidence_rejected_by_rank_scope_guard"
    )
    scope_mismatch = (
        result["benchmark_rank_scope"].notna()
        & result["evidence_rank_scope"].ne("not_recorded")
        & result["benchmark_rank_scope"].ne(result["evidence_rank_scope"])
        & ~invalid_scope
    )
    result.loc[scope_mismatch, "support_status"] = "not_estimable"
    result.loc[scope_mismatch, "status"] = "not_estimable"
    result.loc[scope_mismatch, "reason_code"] = (
        "supportive_biology_rank_scope_mismatch"
    )
    result.loc[scope_mismatch, "evidence_note"] = (
        "supportive_biology_evidence_scope_does_not_match_benchmark_estimand"
    )
    return result


def _empty_simulation_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "__none__",
                "method": "No simulation input",
                "resource_mode": "not_available",
                "truth_scope": "synthetic",
                "metric": "not_estimable",
                "estimate": np.nan,
                "status": "not_estimable",
                "reason_code": "simulation_truth_not_supplied",
            }
        ]
    )


def _iteration_table(path: Path | None, real_datasets: frozenset[str]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(
            [
                {
                    "dataset": "__none__",
                    "method": "CRYCHIC",
                    "metric": "iteration_not_supplied",
                    "iteration_from": "not_available",
                    "iteration_to": "not_available",
                    "before": np.nan,
                    "after": np.nan,
                    "metric_direction": "higher",
                    "status": "not_estimable",
                    "reason_code": "iteration_comparison_not_supplied",
                }
            ]
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    result = _read_table(path)
    required = {
        "dataset",
        "method",
        "metric",
        "iteration_from",
        "iteration_to",
        "before",
        "after",
        "metric_direction",
        "status",
        "reason_code",
    }
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"iteration comparison is missing columns: {sorted(missing)}")
    normalized_metric = (
        result["metric"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    truth_metric = normalized_metric.map(
        lambda value: (
            value in REAL_TRUTH_METRIC_NAMES
            or "auroc" in value
            or "auprc" in value
            or "average_precision" in value
            or "precision_recall_auc" in value
        )
    )
    forbidden = result["dataset"].astype(str).isin(real_datasets) & truth_metric
    if forbidden.any():
        raise ValueError("real-data AUROC/AUPRC iteration metrics are forbidden")
    invalid_direction = set(result["metric_direction"].astype(str)).difference(
        {"higher", "lower"}
    )
    if invalid_direction:
        raise ValueError(
            f"invalid iteration metric_direction: {sorted(invalid_direction)}"
        )
    return result


def _copy_input(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _source_record(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "scope": "input",
            "role": role,
            "path": path.as_posix(),
            "bytes": np.nan,
            "sha256": None,
            "status": "missing",
        }
    return {
        "scope": "input",
        "role": role,
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "status": "verified_present",
    }


def _output_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS.tsv":
            continue
        rows.append(
            {
                "scope": "output",
                "role": "finalized benchmark artifact",
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "status": "generated",
            }
        )
    return rows


def _prepare_output(output_dir: Path, overwrite: bool) -> Path:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    return Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )


def _install_output(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    os.replace(staging, output)


def finalize(
    specification: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Finalize one run specification and return its machine-readable manifest."""
    spec_path = Path(specification).resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    payload, datasets = _load_spec(spec_path)
    parameters = _parameters(payload)
    dataset_truth_scopes = {
        dataset.dataset: dataset.truth_scope for dataset in datasets
    }
    spec_root = spec_path.parent
    output = Path(output_dir).resolve()
    staging = _prepare_output(output, overwrite)

    coverage_frames: list[pd.DataFrame] = []
    primary_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    effects_frames: list[pd.DataFrame] = []
    paired_loso_frames: list[pd.DataFrame] = []
    influence_frames: list[pd.DataFrame] = []
    simulation_frames: list[pd.DataFrame] = []
    sensitivity_coverage_frames: list[pd.DataFrame] = []
    sensitivity_primary_frames: list[pd.DataFrame] = []
    sensitivity_stability_frames: list[pd.DataFrame] = []
    sensitivity_effects_frames: list[pd.DataFrame] = []
    sensitivity_loso_frames: list[pd.DataFrame] = []
    sensitivity_influence_frames: list[pd.DataFrame] = []
    sensitivity_simulation_frames: list[pd.DataFrame] = []
    ranking_agreement_frames: list[pd.DataFrame] = []
    ranking_curve_frames: list[pd.DataFrame] = []
    ranking_interval_frames: list[pd.DataFrame] = []
    ranking_tier_frames: list[pd.DataFrame] = []
    score_index_rows: list[dict[str, Any]] = []
    performance_records: list[dict[str, Any]] = []
    variant_rows: list[dict[str, str]] = []
    adapter_manifest_outputs: list[Path] = []
    score_outputs: list[Path] = []
    dataset_manifest_outputs: list[Path] = []
    source_records: list[dict[str, Any]] = [
        _source_record(spec_path, "run specification")
    ]
    # Ranking stability has a separately preregistered, fixed resampling policy.
    rank_parameters = RankStabilityParameters()

    def collect_rank_tables(
        tables: Any,
        *,
        truth_scope: str,
        view: ScoreView | None = None,
    ) -> None:
        frames_and_tables = (
            (ranking_agreement_frames, tables.agreement),
            (ranking_curve_frames, tables.top_k_curve),
            (ranking_interval_frames, tables.rank_intervals),
            (ranking_tier_frames, tables.stable_tiers),
        )
        for frames, table in frames_and_tables:
            annotated = table.copy(deep=True)
            if view is not None:
                annotated = _annotate_view(annotated, view)
            annotated["truth_scope"] = truth_scope
            frames.append(annotated)

    try:
        copied_spec = _copy_input(
            spec_path, staging / "provenance" / "run_specification.json"
        )
        del copied_spec
        design = _dataset_design_table(datasets)
        _write_tsv(staging / "metrics" / METRIC_FILES["dataset_design"], design)

        for dataset_index, dataset in enumerate(datasets):
            dataset_rank_parameters = replace(
                rank_parameters,
                min_subjects=max(
                    3, int(dataset.comparison.get("min_subjects", 3))
                ),
            )
            generated_dataset_manifest = (
                staging / "datasets" / _slug(dataset.dataset) / "dataset_manifest.json"
            )
            _write_json(
                generated_dataset_manifest,
                {
                    "schema_version": "crychic-finalized-dataset-manifest-v1",
                    "dataset_id": dataset.dataset,
                    "input": dict(dataset.design),
                    "design_audit": {
                        "design_type": dataset.comparison["design"],
                        "status": design.loc[
                            design["dataset"].eq(dataset.dataset), "status"
                        ].iloc[0],
                        "reason_code": design.loc[
                            design["dataset"].eq(dataset.dataset), "reason_code"
                        ].iloc[0],
                    },
                    "comparison": dict(dataset.comparison),
                    "truth_scope": dataset.truth_scope,
                },
            )
            dataset_manifest_outputs.append(generated_dataset_manifest)
            if dataset.dataset_manifest is not None:
                source_records.append(
                    _source_record(dataset.dataset_manifest, "source dataset manifest")
                )
                _copy_input(
                    dataset.dataset_manifest,
                    staging
                    / "provenance"
                    / "dataset_manifests"
                    / f"{dataset_index:03d}_{dataset.dataset_manifest.name}",
                )

            truth = (
                _read_table(dataset.simulation_truth)
                if dataset.simulation_truth is not None
                else None
            )
            if dataset.simulation_truth is not None:
                source_records.append(
                    _source_record(dataset.simulation_truth, "synthetic edge truth")
                )

            if not dataset.adapter_runs:
                if str(dataset.comparison.get("design")) == "unsupported":
                    reason = str(
                        dataset.design.get("reason_code")
                        or dataset.comparison.get("reason_code")
                        or "unsupported_comparison_design"
                    )
                else:
                    reason = "dataset_has_no_adapter_runs"
                identity = RunIdentity(
                    dataset=dataset.dataset,
                    method="No adapter run",
                    method_version="not_available",
                    analysis_track="lr_stlr",
                    resource="not_available",
                    resource_version="not_available",
                    resource_mode="native",
                    score_semantics="not_available",
                    universe_id="not_available",
                    contrast=str(dataset.comparison.get("contrast", "not_available")),
                )
                coverage_frames.append(
                    pd.DataFrame([_not_estimable_coverage(identity, reason)])
                )
                primary_frames.append(
                    pd.DataFrame([_not_estimable_primary(identity, reason)])
                )
                stability_frames.append(
                    pd.DataFrame([_not_estimable_stability(identity, reason)])
                )
                collect_rank_tables(
                    not_estimable_rank_stability(
                        identity.metric_values(),
                        reason_code=reason,
                        parameters=dataset_rank_parameters,
                    ),
                    truth_scope=dataset.truth_scope,
                )
                score_index_rows.append(
                    {
                        "schema_version": SCORE_INDEX_SCHEMA_VERSION,
                        **identity.metric_values(),
                        "source_path": None,
                        "output_path": None,
                        "rows": 0,
                        "status": "not_estimable",
                        "reason_code": reason,
                    }
                )
                performance_records.append(
                    {
                        "dataset": identity.dataset,
                        "method": identity.method,
                        "method_version": identity.method_version,
                        "analysis_track": identity.analysis_track,
                        "resource": identity.resource,
                        "resource_version": identity.resource_version,
                        "resource_mode": identity.resource_mode,
                        "run_id": f"no_adapter_{_slug(dataset.dataset)}",
                        "status": "skipped",
                        "wall_time_seconds": None,
                        "peak_rss_mb": None,
                        "output_bytes": None,
                        "threads": 1,
                    }
                )
                if truth is not None:
                    simulation_frames.append(
                        pd.DataFrame(
                            [
                                identity.metric_values()
                                | {
                                    "truth_scope": dataset.truth_scope,
                                    "metric": "edge_truth",
                                    "estimate": np.nan,
                                    "status": "not_estimable",
                                    "reason_code": reason,
                                }
                            ]
                        )
                    )
                continue

            for run_index, run in enumerate(dataset.adapter_runs):
                manifest_path = _resolve(
                    spec_root,
                    run.get("manifest"),
                    field=f"{dataset.dataset}.adapter_runs[{run_index}].manifest",
                )
                if not manifest_path.is_file():
                    raise FileNotFoundError(manifest_path)
                manifest = _read_json(manifest_path)
                source_records.append(_source_record(manifest_path, "adapter manifest"))
                copied_manifest = _copy_input(
                    manifest_path,
                    staging
                    / "provenance"
                    / "adapter_manifests"
                    / f"{dataset_index:03d}_{run_index:03d}_{manifest_path.name}",
                )
                adapter_manifest_outputs.append(copied_manifest)
                fallback_identity = _manifest_identity(dataset, run, manifest)
                long_value = run.get("long_table")
                long_path = (
                    _resolve(
                        spec_root,
                        long_value,
                        field=f"{dataset.dataset}.adapter_runs[{run_index}].long_table",
                    )
                    if long_value is not None
                    else None
                )
                run_performance, performance_sources = _performance_records_for_run(
                    spec_root=spec_root,
                    identity=fallback_identity,
                    run=run,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    long_path=long_path,
                )
                performance_records.extend(run_performance)
                for performance_source in performance_sources:
                    source_records.append(
                        _source_record(
                            performance_source,
                            "method runtime source manifest",
                        )
                    )
                    _copy_input(
                        performance_source,
                        staging
                        / "provenance"
                        / "performance_manifests"
                        / (
                            f"{dataset_index:03d}_{run_index:03d}_"
                            f"{performance_source.name}"
                        ),
                    )
                if long_path is None or not long_path.is_file():
                    reason = (
                        "adapter_long_table_missing_after_failed_run"
                        if str(manifest.get("status")) != "complete"
                        else "adapter_long_table_missing"
                    )
                    source_records.append(
                        _source_record(
                            long_path or Path(str(long_value or "not_declared")),
                            "adapter long table",
                        )
                    )
                    coverage_frames.append(
                        pd.DataFrame(
                            [_not_estimable_coverage(fallback_identity, reason)]
                        )
                    )
                    primary_frames.append(
                        pd.DataFrame(
                            [_not_estimable_primary(fallback_identity, reason)]
                        )
                    )
                    stability_frames.append(
                        pd.DataFrame(
                            [_not_estimable_stability(fallback_identity, reason)]
                        )
                    )
                    collect_rank_tables(
                        not_estimable_rank_stability(
                            fallback_identity.metric_values(),
                            reason_code=reason,
                            parameters=dataset_rank_parameters,
                        ),
                        truth_scope=dataset.truth_scope,
                    )
                    score_index_rows.append(
                        {
                            "schema_version": SCORE_INDEX_SCHEMA_VERSION,
                            **fallback_identity.metric_values(),
                            "source_path": None
                            if long_path is None
                            else long_path.as_posix(),
                            "output_path": None,
                            "rows": 0,
                            "status": "not_estimable",
                            "reason_code": reason,
                        }
                    )
                    variant_rows.append(
                        {
                            "dataset": fallback_identity.dataset,
                            "method": fallback_identity.method,
                            "resource_mode": fallback_identity.resource_mode,
                        }
                    )
                    continue

                source_records.append(_source_record(long_path, "adapter long table"))
                source_digest = _validate_source_hash(long_path, manifest)
                track_metadata = _read_unique_parquet_metadata(
                    long_path, TRACK_METADATA_COLUMNS
                )
                base_identity = _identity_from_track_metadata(track_metadata, dataset)
                views = _score_views(track_metadata, dataset, run, manifest)
                multiple_views = len(views) > 1
                variant_rows.append(
                    {
                        "dataset": base_identity.dataset,
                        "method": base_identity.method,
                        "resource_mode": base_identity.resource_mode,
                    }
                )
                if base_identity.analysis_track != "lr_stlr":
                    for view_index, view in enumerate(views):
                        identity = _view_identity(
                            base_identity, view, multiple_views=multiple_views
                        )
                        index_base = {
                            "schema_version": SCORE_INDEX_SCHEMA_VERSION,
                            **identity.metric_values(),
                            "score_view_run_id": view.run_id,
                            "score_view_label": view.label,
                            "score_view_role": view.role,
                            "score_view_contrast_candidate": view.contrast_candidate,
                            "score_view_primary": view.primary,
                            "source_path": long_path.as_posix(),
                            "source_sha256": source_digest,
                        }
                        if not view.include:
                            score_index_rows.append(
                                index_base
                                | {
                                    "output_path": None,
                                    "rows": 0,
                                    "status": "not_estimable",
                                    "reason_code": (
                                        "score_view_excluded_by_preregistered_selection"
                                    ),
                                }
                            )
                            continue
                        selected_track = _read_parquet_score_view(
                            long_path, view.run_id
                        )
                        selected_track["score_view_label"] = view.label
                        selected_track["score_view_role"] = view.role
                        selected_track["score_view_contrast_candidate"] = (
                            view.contrast_candidate
                        )
                        output_name = (
                            f"{dataset_index:03d}_{run_index:03d}_{view_index:03d}_"
                            f"{_slug(dataset.dataset)}_{_slug(identity.method)}_"
                            f"{_slug(view.label)}.parquet"
                        )
                        destination = staging / "track_b_tables" / output_name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        selected_track.to_parquet(destination, index=False)
                        score_outputs.append(destination)
                        reason = _track_b_reason(identity)
                        score_index_rows.append(
                            index_base
                            | {
                                "output_path": destination.relative_to(
                                    staging
                                ).as_posix(),
                                "output_sha256": _sha256(destination),
                                "rows": len(selected_track),
                                "status": "not_estimable",
                                "reason_code": reason,
                            }
                        )
                        coverage_ne = pd.DataFrame(
                            [_not_estimable_coverage(identity, reason)]
                        )
                        primary_ne = pd.DataFrame(
                            [_not_estimable_primary(identity, reason)]
                        )
                        stability_ne = pd.DataFrame(
                            [_not_estimable_stability(identity, reason)]
                        )
                        if view.primary:
                            coverage_frames.append(_annotate_view(coverage_ne, view))
                            primary_frames.append(_annotate_view(primary_ne, view))
                            stability_frames.append(_annotate_view(stability_ne, view))
                            collect_rank_tables(
                                not_estimable_rank_stability(
                                    identity.metric_values(),
                                    reason_code=reason,
                                    parameters=dataset_rank_parameters,
                                ),
                                truth_scope=dataset.truth_scope,
                                view=view,
                            )
                        else:
                            sensitivity_coverage_frames.append(
                                _annotate_view(coverage_ne, view)
                            )
                            sensitivity_primary_frames.append(
                                _annotate_view(primary_ne, view)
                            )
                            sensitivity_stability_frames.append(
                                _annotate_view(stability_ne, view)
                            )
                        if truth is not None:
                            truth_ne = pd.DataFrame(
                                [
                                    identity.metric_values()
                                    | {
                                        "truth_scope": dataset.truth_scope,
                                        "metric": "track_b_truth_metric",
                                        "estimate": np.nan,
                                        "status": "not_estimable",
                                        "reason_code": (
                                            "track_b_requires_track_specific_"
                                            "truth_metric"
                                        ),
                                    }
                                ]
                            )
                            target_frames = (
                                simulation_frames
                                if view.primary
                                else sensitivity_simulation_frames
                            )
                            target_frames.append(_annotate_view(truth_ne, view))
                        del selected_track
                        gc.collect()
                    del track_metadata
                    gc.collect()
                    continue

                for view_index, view in enumerate(views):
                    base_view_identity = _identity_from_track_metadata(
                        track_metadata.loc[
                            track_metadata["run_id"].astype(str).eq(view.run_id)
                        ],
                        dataset,
                    )
                    identity = _view_identity(
                        base_view_identity, view, multiple_views=multiple_views
                    )
                    rank_scope, rank_scope_reason = _rank_scope_for_view(
                        run, manifest, view
                    )
                    index_base = {
                        "schema_version": SCORE_INDEX_SCHEMA_VERSION,
                        **identity.metric_values(),
                        "score_view_run_id": view.run_id,
                        "score_view_label": view.label,
                        "score_view_role": view.role,
                        "score_view_contrast_candidate": view.contrast_candidate,
                        "score_view_primary": view.primary,
                        "source_path": long_path.as_posix(),
                        "source_sha256": source_digest,
                        "rank_scope": rank_scope,
                    }
                    if not view.include:
                        score_index_rows.append(
                            index_base
                            | {
                                "output_path": None,
                                "rows": 0,
                                "status": "not_estimable",
                                "reason_code": (
                                    "score_view_excluded_by_preregistered_selection"
                                ),
                            }
                        )
                        continue
                    selected_external = _read_parquet_score_view(
                        long_path,
                        view.run_id,
                        columns=EXTERNAL_LR_COLUMNS,
                    )
                    # Keep high-cardinality strings Arrow-backed, while using
                    # NumPy dtypes for arithmetic required by score validation.
                    _validate_universe_member_values(
                        selected_external["universe_member"]
                    )
                    selected_external["universe_member"] = selected_external[
                        "universe_member"
                    ].astype(bool)
                    selected_external["universe_size"] = selected_external[
                        "universe_size"
                    ].astype("int64")
                    selected_external["score"] = selected_external["score"].astype(
                        "float64"
                    )
                    selected_external["score_name"] = identity.score_semantics
                    mapped = external_long_to_score_table(
                        selected_external,
                        context_key=dataset.context_key,
                        contrast=view.contrast,
                        dataset=dataset.dataset,
                    )
                    del selected_external
                    # Validation performs sorting/ranking using ordinary NumPy
                    # dtypes where required. Return repeated string identifiers
                    # to Arrow storage before downstream metric fan-out.
                    mapped = mapped.convert_dtypes(dtype_backend="pyarrow")
                    mapped = _annotate_view(mapped, view)
                    mapped = _annotate_rank_scope(mapped, rank_scope)
                    output_name = (
                        f"{dataset_index:03d}_{run_index:03d}_{view_index:03d}_"
                        f"{_slug(dataset.dataset)}_{_slug(identity.method)}_"
                        f"{_slug(view.label)}.parquet"
                    )
                    destination = staging / "score_tables" / output_name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    mapped.to_parquet(destination, index=False)
                    score_outputs.append(destination)
                    score_index_rows.append(
                        index_base
                        | {
                            "output_path": destination.relative_to(staging).as_posix(),
                            "output_sha256": _sha256(destination),
                            "rows": len(mapped),
                            "status": "observed",
                            "reason_code": None,
                        }
                    )

                    coverage_result = _annotate_view(
                        score_coverage_summary(mapped, validated=True), view
                    )
                    coverage_result = _annotate_rank_scope(
                        coverage_result, rank_scope
                    )
                    coverage_target = (
                        coverage_frames if view.primary else sensitivity_coverage_frames
                    )
                    stability_target = (
                        stability_frames
                        if view.primary
                        else sensitivity_stability_frames
                    )
                    coverage_target.append(coverage_result)
                    design_type = str(dataset.comparison["design"])
                    reference = str(dataset.comparison.get("reference", ""))
                    target = str(dataset.comparison.get("target", ""))
                    minimum = int(dataset.comparison.get("min_subjects", 3))
                    support = mapped.loc[
                        mapped["context"].astype(str).isin((reference, target)),
                        ["subject_id", "context"],
                    ].drop_duplicates()
                    reference_subjects = set(
                        support.loc[
                            support["context"].astype(str).eq(reference), "subject_id"
                        ].astype(str)
                    )
                    target_subjects = set(
                        support.loc[
                            support["context"].astype(str).eq(target), "subject_id"
                        ].astype(str)
                    )
                    paired_subjects = reference_subjects & target_subjects
                    variant_rows.append(
                        {
                            "dataset": identity.dataset,
                            "method": identity.method,
                            "resource_mode": identity.resource_mode,
                            "rank_scope": rank_scope,
                        }
                    )
                    if rank_scope_reason is not None:
                        primary_ne = _annotate_rank_scope(
                            _annotate_view(
                                pd.DataFrame(
                                    [
                                        _not_estimable_primary(
                                            identity, rank_scope_reason
                                        )
                                    ]
                                ),
                                view,
                            ),
                            rank_scope,
                        )
                        stability_ne = _annotate_rank_scope(
                            _annotate_view(
                                pd.DataFrame(
                                    [
                                        _not_estimable_stability(
                                            identity, rank_scope_reason
                                        )
                                    ]
                                ),
                                view,
                            ),
                            rank_scope,
                        )
                        if view.primary:
                            primary_frames.append(primary_ne)
                            stability_frames.append(stability_ne)
                            if dataset.truth_scope == "real_data":
                                collect_rank_tables(
                                    not_estimable_rank_stability(
                                        identity.metric_values(),
                                        reason_code=rank_scope_reason,
                                        parameters=dataset_rank_parameters,
                                        design=design_type,
                                        reference=reference,
                                        target=target,
                                        rank_scope=rank_scope,
                                        n_reference_subjects=len(reference_subjects),
                                        n_target_subjects=len(target_subjects),
                                        n_paired_subjects=len(paired_subjects),
                                    ),
                                    truth_scope=dataset.truth_scope,
                                    view=view,
                                )
                        else:
                            sensitivity_primary_frames.append(primary_ne)
                            sensitivity_stability_frames.append(stability_ne)
                        if truth is not None:
                            truth_ne = _annotate_rank_scope(
                                _annotate_view(
                                    pd.DataFrame(
                                        [
                                            identity.metric_values()
                                            | {
                                                "truth_scope": dataset.truth_scope,
                                                "metric": "edge_truth",
                                                "estimate": np.nan,
                                                "status": "not_estimable",
                                                "reason_code": rank_scope_reason,
                                            }
                                        ]
                                    ),
                                    view,
                                ),
                                rank_scope,
                            )
                            truth_target = (
                                simulation_frames
                                if view.primary
                                else sensitivity_simulation_frames
                            )
                            truth_target.append(truth_ne)
                        del mapped
                        gc.collect()
                        continue

                    within = _annotate_rank_scope(
                        _annotate_view(
                            _stability_long(
                                within_context_reproducibility(
                                    mapped,
                                    top_k=parameters["top_k"],
                                    validated=True,
                                )
                            ),
                            view,
                        ),
                        rank_scope,
                    )
                    stability_target.append(within)
                    if view.primary and dataset.truth_scope == "real_data":
                        if design_type in {"paired", "unpaired"}:
                            collect_rank_tables(
                                evaluate_multicondition_rank_stability(
                                    mapped,
                                    reference=reference,
                                    target=target,
                                    design=cast(Any, design_type),
                                    rank_scope=cast(Any, rank_scope),
                                    parameters=dataset_rank_parameters,
                                    validated=True,
                                ),
                                truth_scope=dataset.truth_scope,
                                view=view,
                            )
                        else:
                            collect_rank_tables(
                                not_estimable_rank_stability(
                                    identity.metric_values(),
                                    reason_code="unsupported_comparison_design",
                                    parameters=dataset_rank_parameters,
                                ),
                                truth_scope=dataset.truth_scope,
                                view=view,
                            )
                    if design_type == "paired":
                        effects = _annotate_view(
                            paired_edge_effects(
                                mapped,
                                reference=reference,
                                target=target,
                                min_pairs=minimum,
                                validated=True,
                            ),
                            view,
                        )
                        effects = _annotate_rank_scope(effects, rank_scope)
                        loso = _annotate_view(
                            paired_differential_loso_reproducibility(
                                mapped,
                                reference=reference,
                                target=target,
                                min_subjects=minimum,
                                top_k=parameters["top_k"],
                                validated=True,
                            ),
                            view,
                        )
                        loso = _annotate_rank_scope(loso, rank_scope)
                        summary = _annotate_view(
                            summarize_loso_primary_endpoint(
                                loso,
                                n_bootstrap=parameters["n_bootstrap"],
                                random_seed=parameters["random_seed"],
                            ),
                            view,
                        )
                        summary = _annotate_rank_scope(summary, rank_scope)
                        if view.primary:
                            effects_frames.append(effects)
                            paired_loso_frames.append(loso)
                            primary_frames.append(_paired_primary(summary))
                            stability_frames.append(_stability_long(loso))
                        else:
                            sensitivity_effects_frames.append(effects)
                            sensitivity_loso_frames.append(loso)
                            sensitivity_primary_frames.append(_paired_primary(summary))
                            sensitivity_stability_frames.append(_stability_long(loso))
                    elif design_type == "unpaired":
                        effects = _annotate_view(
                            unpaired_edge_effects(
                                mapped,
                                reference=reference,
                                target=target,
                                min_subjects=minimum,
                                validated=True,
                            ),
                            view,
                        )
                        effects = _annotate_rank_scope(effects, rank_scope)
                        split = _annotate_view(
                            unpaired_differential_split_half_reproducibility(
                                mapped,
                                reference=reference,
                                target=target,
                                n_repeats=parameters["n_split_repeats"],
                                min_subjects_per_half=parameters[
                                    "min_subjects_per_half"
                                ],
                                top_k=parameters["top_k"],
                                random_seed=parameters["random_seed"],
                                validated=True,
                            ),
                            view,
                        )
                        split = _annotate_rank_scope(split, rank_scope)
                        influence = _annotate_view(
                            unpaired_leave_one_subject_influence(
                                mapped,
                                reference=reference,
                                target=target,
                                min_remaining_subjects=max(2, minimum - 1),
                                top_k=parameters["top_k"],
                                validated=True,
                            ),
                            view,
                        )
                        influence = _annotate_rank_scope(influence, rank_scope)
                        if view.primary:
                            effects_frames.append(effects)
                            influence_frames.append(influence)
                            primary_frames.append(_unpaired_primary(split))
                            stability_frames.append(_stability_long(split))
                            stability_frames.append(
                                _annotate_view(_influence_summary(influence), view)
                            )
                        else:
                            sensitivity_effects_frames.append(effects)
                            sensitivity_influence_frames.append(influence)
                            sensitivity_primary_frames.append(_unpaired_primary(split))
                            sensitivity_stability_frames.append(_stability_long(split))
                            sensitivity_stability_frames.append(
                                _annotate_view(_influence_summary(influence), view)
                            )
                    else:
                        reason = "unsupported_comparison_design"
                        primary_ne = _annotate_view(
                            pd.DataFrame([_not_estimable_primary(identity, reason)]),
                            view,
                        )
                        stability_ne = _annotate_view(
                            pd.DataFrame([_not_estimable_stability(identity, reason)]),
                            view,
                        )
                        if view.primary:
                            primary_frames.append(primary_ne)
                            stability_frames.append(stability_ne)
                        else:
                            sensitivity_primary_frames.append(primary_ne)
                            sensitivity_stability_frames.append(stability_ne)

                    if truth is not None:
                        truth_result = _annotate_view(
                            synthetic_edge_truth_metrics(
                                mapped,
                                truth,
                                top_k=parameters["top_k"],
                                validated=True,
                            ),
                            view,
                        )
                        truth_result = _annotate_rank_scope(truth_result, rank_scope)
                        truth_target = (
                            simulation_frames
                            if view.primary
                            else sensitivity_simulation_frames
                        )
                        truth_target.append(truth_result)
                    del mapped
                    gc.collect()
                del track_metadata
                gc.collect()

        if paired_loso_frames:
            all_loso = _annotate_truth_scope(
                pd.concat(paired_loso_frames, ignore_index=True, sort=False),
                dataset_truth_scopes,
            )
            cross_dataset = _cross_dataset_primary(all_loso, parameters)
            if not cross_dataset.empty:
                primary_frames.append(cross_dataset)
            derived_dir = staging / "derived"
            derived_dir.mkdir(parents=True, exist_ok=True)
            all_loso.to_parquet(derived_dir / "paired_loso_folds.parquet", index=False)
        else:
            all_loso = pd.DataFrame()

        if effects_frames:
            all_effects = _annotate_truth_scope(
                pd.concat(effects_frames, ignore_index=True, sort=False),
                dataset_truth_scopes,
            )
            concordance = cross_method_concordance(
                all_effects,
                minimum_shared_edges=parameters["minimum_shared_edges"],
            )
            derived_dir = staging / "derived"
            derived_dir.mkdir(parents=True, exist_ok=True)
            all_effects.to_parquet(derived_dir / "edge_effects.parquet", index=False)
        else:
            all_effects = pd.DataFrame()
            concordance = pd.DataFrame()

        concordance = _annotate_truth_scope(concordance, dataset_truth_scopes)
        concordance["rank_scope"] = "global_common_functional"
        if concordance.empty or not concordance["truth_scope"].eq("real_data").any():
            real_data_ne = pd.DataFrame(
                [
                    {
                        "dataset": "__none_real_data__",
                        "method_left": "NE",
                        "method_right": "NE",
                        "analysis_track": "lr_stlr",
                        "resource_mode": "H-common",
                        "truth_scope": "real_data",
                        "contrast": "prespecified_per_dataset",
                        "effect_spearman": np.nan,
                        "direction_agreement": np.nan,
                        "shared_edges": 0,
                        "status": "not_estimable",
                        "reason_code": "no_comparable_real_data_lr_method_pair",
                    }
                ]
            )
            concordance = pd.concat(
                [concordance, real_data_ne], ignore_index=True, sort=False
            )

        if influence_frames:
            all_influence = pd.concat(influence_frames, ignore_index=True, sort=False)
            derived_dir = staging / "derived"
            derived_dir.mkdir(parents=True, exist_ok=True)
            all_influence.to_parquet(
                derived_dir / "unpaired_leave_one_subject_influence.parquet",
                index=False,
            )

        sensitivity_tables = {
            "sensitivity_coverage.tsv": sensitivity_coverage_frames,
            "sensitivity_primary_endpoint.tsv": sensitivity_primary_frames,
            "sensitivity_stability.tsv": sensitivity_stability_frames,
            "sensitivity_simulation_truth.tsv": sensitivity_simulation_frames,
        }
        for filename, frames in sensitivity_tables.items():
            if frames:
                table = _annotate_truth_scope(
                    pd.concat(frames, ignore_index=True, sort=False),
                    dataset_truth_scopes,
                )
                _write_tsv(
                    staging / "derived" / filename,
                    table,
                )
        sensitivity_parquets = {
            "sensitivity_edge_effects.parquet": sensitivity_effects_frames,
            "sensitivity_paired_loso_folds.parquet": sensitivity_loso_frames,
            "sensitivity_unpaired_influence.parquet": (sensitivity_influence_frames),
        }
        for filename, frames in sensitivity_parquets.items():
            if frames:
                path = staging / "derived" / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                _annotate_truth_scope(
                    pd.concat(frames, ignore_index=True, sort=False),
                    dataset_truth_scopes,
                ).to_parquet(path, index=False)

        coverage = (
            pd.concat(coverage_frames, ignore_index=True, sort=False)
            if coverage_frames
            else pd.DataFrame()
        )
        explicit_coverage = _optional_path(
            spec_root,
            payload.get("coverage_records"),
            field="coverage_records",
        )
        if explicit_coverage is not None:
            if not explicit_coverage.is_file():
                raise FileNotFoundError(explicit_coverage)
            source_records.append(
                _source_record(explicit_coverage, "precomputed coverage records")
            )
            coverage_input = _read_table(explicit_coverage)
            required_coverage = {
                "dataset",
                "method",
                "resource_mode",
                "status",
                "reason_code",
            }
            missing_coverage = required_coverage.difference(coverage_input.columns)
            if missing_coverage:
                raise ValueError(
                    "coverage_records is missing columns: "
                    f"{sorted(missing_coverage)}"
                )
            coverage = pd.concat(
                [coverage, coverage_input], ignore_index=True, sort=False
            )
        coverage = _annotate_truth_scope(coverage, dataset_truth_scopes)
        primary = (
            pd.concat(primary_frames, ignore_index=True, sort=False)
            if primary_frames
            else pd.DataFrame()
        )
        primary = _annotate_truth_scope(primary, dataset_truth_scopes)
        stability = (
            pd.concat(stability_frames, ignore_index=True, sort=False)
            if stability_frames
            else pd.DataFrame()
        )
        stability = _annotate_truth_scope(stability, dataset_truth_scopes)
        simulation = (
            pd.concat(simulation_frames, ignore_index=True, sort=False)
            if simulation_frames
            else _empty_simulation_table()
        )
        explicit_simulation = _optional_path(
            spec_root,
            payload.get("simulation_records"),
            field="simulation_records",
        )
        if explicit_simulation is not None:
            if not explicit_simulation.is_file():
                raise FileNotFoundError(explicit_simulation)
            source_records.append(
                _source_record(explicit_simulation, "external simulation metrics")
            )
            simulation_input = _read_table(explicit_simulation)
            required_simulation = {
                "truth_scope",
                "method",
                "metric",
                "estimate",
                "status",
                "reason_code",
            }
            missing_simulation = required_simulation.difference(
                simulation_input.columns
            )
            if missing_simulation:
                raise ValueError(
                    "simulation_records is missing columns: "
                    f"{sorted(missing_simulation)}"
                )
            invalid_scopes = set(
                simulation_input["truth_scope"].astype(str)
            ).difference(SYNTHETIC_TRUTH_SCOPES)
            if invalid_scopes:
                raise ValueError(
                    "simulation_records contains non-synthetic truth scopes: "
                    f"{sorted(invalid_scopes)}"
                )
            simulation = (
                pd.concat(
                    [simulation, simulation_input], ignore_index=True, sort=False
                )
                if simulation_frames
                else simulation_input
            )

        explicit_performance = _optional_path(
            spec_root,
            payload.get("performance_records"),
            field="performance_records",
        )
        if explicit_performance is not None:
            if not explicit_performance.is_file():
                raise FileNotFoundError(explicit_performance)
            source_records.append(
                _source_record(explicit_performance, "performance run records")
            )
            performance_input = _read_table(explicit_performance)
        else:
            performance_input = pd.DataFrame(performance_records)
        runtime_input = _prepare_performance_runtime_input(performance_input)
        performance = summarize_run_performance(runtime_input)
        performance_components = _performance_component_summary(performance_input)
        performance = performance.merge(
            performance_components,
            on=[
                "dataset",
                "method",
                "method_version",
                "analysis_track",
                "resource",
                "resource_version",
                "resource_mode",
            ],
            how="left",
            validate="one_to_one",
        )
        performance = _annotate_truth_scope(performance, dataset_truth_scopes)

        truth_value = payload.get("supportive_biology_truth")
        if truth_value is None:
            raise ValueError("supportive_biology_truth is required and must be frozen")
        truth_path = _resolve(spec_root, truth_value, field="supportive_biology_truth")
        if not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        source_records.append(
            _source_record(truth_path, "locked supportive biology truth")
        )
        copied_truth = _copy_input(
            truth_path,
            staging / "truth" / "multicondition_supportive_biology.yaml",
        )
        evidence_path = _optional_path(
            spec_root,
            payload.get("biology_support"),
            field="biology_support",
        )
        if evidence_path is not None:
            source_records.append(
                _source_record(evidence_path, "supportive biology evidence")
            )
        variants = pd.DataFrame(variant_rows)
        biology_dataset_aliases = _biology_dataset_aliases(payload)
        biology = _biology_support_table(
            copied_truth,
            evidence_path,
            variants,
            dataset_aliases=biology_dataset_aliases,
        )

        iteration_path = _optional_path(
            spec_root,
            payload.get("iteration_comparison"),
            field="iteration_comparison",
        )
        if iteration_path is not None:
            source_records.append(
                _source_record(iteration_path, "iteration comparison")
            )
        real_datasets = frozenset(
            dataset.dataset
            for dataset in datasets
            if dataset.truth_scope not in SYNTHETIC_TRUTH_SCOPES
        )
        iteration = _iteration_table(iteration_path, real_datasets)

        if ranking_agreement_frames:
            ranking_agreement = pd.concat(
                ranking_agreement_frames, ignore_index=True, sort=False
            )
            ranking_curve = pd.concat(
                ranking_curve_frames, ignore_index=True, sort=False
            )
            ranking_intervals = pd.concat(
                ranking_interval_frames, ignore_index=True, sort=False
            )
            ranking_tiers = pd.concat(
                ranking_tier_frames, ignore_index=True, sort=False
            )
        else:
            fallback_rank = not_estimable_rank_stability(
                {
                    "dataset": "__none_real_data__",
                    "method": "NE",
                    "method_version": "not_available",
                    "analysis_track": "lr_stlr",
                    "resource": "not_available",
                    "resource_version": "not_available",
                    "resource_mode": "native",
                    "score_semantics": "not_available",
                    "universe_id": "not_available",
                    "contrast": "not_available",
                },
                reason_code="no_real_data_ranking_inputs",
                parameters=rank_parameters,
            )
            ranking_agreement = fallback_rank.agreement
            ranking_curve = fallback_rank.top_k_curve
            ranking_intervals = fallback_rank.rank_intervals
            ranking_tiers = fallback_rank.stable_tiers
            for table in (
                ranking_agreement,
                ranking_curve,
                ranking_intervals,
                ranking_tiers,
            ):
                table["truth_scope"] = "real_data"

        metric_tables = {
            "coverage": coverage,
            "loso_primary": primary,
            "stability": stability,
            "concordance": concordance,
            "performance": performance,
            "biology_support": biology,
            "simulation_truth": simulation,
            "iteration_comparison": iteration,
            "ranking_agreement": ranking_agreement,
            "ranking_top_k_curve": ranking_curve,
            "ranking_intervals": ranking_intervals,
            "ranking_tiers": ranking_tiers,
        }
        for key, table in metric_tables.items():
            _write_tsv(staging / "metrics" / METRIC_FILES[key], table)

        score_index = pd.DataFrame(score_index_rows)
        _write_tsv(staging / "score_tables" / "score_table_index.tsv", score_index)

        report_inputs = {
            "schema_version": REPORT_INPUT_SCHEMA_VERSION,
            "supportive_biology_dataset_aliases": biology_dataset_aliases,
            "dataset_truth_scopes": dataset_truth_scopes,
            "adapter_manifests": [
                path.relative_to(staging).as_posix()
                for path in adapter_manifest_outputs
            ],
            "score_tables": [
                path.relative_to(staging).as_posix() for path in score_outputs
            ],
            "dataset_manifests": [
                path.relative_to(staging).as_posix()
                for path in dataset_manifest_outputs
            ],
            "truth_yaml": copied_truth.relative_to(staging).as_posix(),
            "metrics": {
                key: f"metrics/{filename}" for key, filename in METRIC_FILES.items()
            },
        }
        _write_json(staging / "report_inputs.json", report_inputs)

        generated_at = str(payload.get("generated_at", datetime.now(UTC).isoformat()))
        frozen_main_inputs = payload.get("frozen_main_inputs", {})
        if not isinstance(frozen_main_inputs, Mapping):
            raise ValueError("frozen_main_inputs must be an annotation object")
        final_manifest: dict[str, Any] = {
            "schema_version": FINALIZATION_SCHEMA_VERSION,
            "specification_schema_version": SPEC_SCHEMA_VERSION,
            "generated_at": generated_at,
            "parameters": parameters,
            "ranking_parameters": {
                **asdict(rank_parameters),
                "family_top_k_curve": [1, 25],
                "lr_top_k_curve": [1, 100],
                "sender_top_k_curve": [1, 25],
                "sender_receiver_pair_top_k_curve": [1, 25],
                "lr_family_mapping_status": "not_available_in_score_contract",
                "resampling_unit": "subject",
                "rank_interval_conditioning": "conditional_on_rank_availability",
                "rank_availability_frequency_denominator": (
                    "all_requested_replicates"
                ),
                "top_k_frequency_denominator": "all_requested_replicates",
                "tie_policy": "average_rank_and_tie_inclusive_top_k",
            },
            "frozen_main_inputs": dict(frozen_main_inputs),
            "frozen_main_inputs_validation": {
                "status": "annotation_only_unless_bound_by_path_contract",
                "reason_code": (
                    "legacy frozen_main_inputs keys do not declare source paths; "
                    "adapter output hashes, performance_override.expected_sha256, "
                    "and copied input checksums are validated separately"
                ),
            },
            "datasets": [dataset.dataset for dataset in datasets],
            "counts": {
                "adapter_runs": sum(len(dataset.adapter_runs) for dataset in datasets),
                "lr_score_tables": int(
                    score_index["analysis_track"].eq("lr_stlr").sum()
                )
                if not score_index.empty
                else 0,
                "track_b_tables": int(score_index["analysis_track"].ne("lr_stlr").sum())
                if not score_index.empty
                else 0,
                "primary_score_views": int(
                    score_index["score_view_primary"].fillna(False).astype(bool).sum()
                )
                if "score_view_primary" in score_index
                else 0,
                "sensitivity_score_views": int(
                    score_index["score_view_role"].eq("sensitivity").sum()
                )
                if "score_view_role" in score_index
                else 0,
                "excluded_score_views": int(
                    score_index["score_view_role"].eq("excluded").sum()
                )
                if "score_view_role" in score_index
                else 0,
                "coverage_rows": len(coverage),
                "primary_endpoint_rows": len(primary),
                "stability_rows": len(stability),
                "concordance_rows": len(concordance),
                "simulation_truth_rows": len(simulation),
                "ranking_agreement_rows": len(ranking_agreement),
                "ranking_top_k_curve_rows": len(ranking_curve),
                "ranking_interval_rows": len(ranking_intervals),
                "ranking_tier_rows": len(ranking_tiers),
            },
            "guardrails": {
                "real_data_edge_auroc_reported": False,
                "supportive_biology_is_edge_ground_truth": False,
                "track_b_mixed_with_lr_concordance": False,
                "preregistered_track_b_primary_reported": False,
                "track_b_proxy_mislabelled_as_native_nichenet": False,
                "unpaired_leave_one_out_labelled_independent_replication": False,
                "nichenet_lr_or_sender_ranking_reported": False,
                "ranking_missing_items_imputed_as_zero": False,
                "receiver_child_functionals_ranked_globally": False,
            },
            "supportive_biology_dataset_aliases": biology_dataset_aliases,
            "report_inputs": "report_inputs.json",
            "checksum_manifest": "SHA256SUMS.tsv",
        }
        _write_json(staging / "finalization_manifest.json", final_manifest)
        checksums = pd.DataFrame([*source_records, *_output_records(staging)])
        _write_tsv(staging / "SHA256SUMS.tsv", checksums)
        _install_output(staging, output, overwrite)
        return final_manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize paired/unpaired multi-condition adapter outputs into the "
            "publication report input contract."
        )
    )
    parser.add_argument(
        "--spec", required=True, type=Path, help="JSON run specification"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = finalize(args.spec, args.output_dir, overwrite=args.overwrite)
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
