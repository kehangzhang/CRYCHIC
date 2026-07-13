"""Auditable mixed paired/unpaired repeated-measures benchmark effects.

The backend intentionally stops at exploratory point estimates and diagnostic
cluster-robust standard errors.  It does not emit calibrated p-values or
q-values.  A design is frozen from sample metadata once, independently of any
method score, then reused for every method and edge.

Scores from replicate samples in the same subject/context/adjustment cell are
averaged before fitting.  The fixed-effect model therefore gives every design
cell equal weight while the CR1 sandwich groups all cells from one subject in
the same cluster.  Singleton and repeated subjects may coexist.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from crychic.core import stable_id

_MODEL_VERSION = "categorical_ols_subject_cluster_cr1_v1"
_SCHEMA_VERSION = "1.0.0"
_PRODUCER_MARKER = "crychic.repeated_measures.frozen_design.v1"


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...

    def hexdigest(self) -> str: ...


def _field_names(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(f"{name} must contain non-empty field names")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _label(value: object, *, field_name: str) -> str:
    try:
        missing = pd.isna(cast(Any, value))
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ValueError(f"{field_name} must not contain missing values")
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain scalar labels")
    if isinstance(value, Number):
        try:
            finite = bool(np.isfinite(cast(Any, value)))
        except TypeError:
            try:
                finite = math.isfinite(float(cast(Any, value)))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{field_name} numeric labels must be finite scalars"
                ) from error
        if not finite:
            raise ValueError(f"{field_name} numeric labels must be finite")
    result = str(value)
    if not result or result != result.strip():
        raise ValueError(f"{field_name} must contain canonical non-empty labels")
    return result


def _require_complete_group_keys(
    table: pd.DataFrame,
    keys: Sequence[str],
    *,
    table_name: str,
) -> None:
    names = tuple(keys)
    missing_columns = set(names).difference(table.columns)
    if missing_columns:
        raise ValueError(
            f"{table_name} is missing group keys: {sorted(missing_columns)}"
        )
    missing_values = [name for name in names if table[name].isna().any()]
    if missing_values:
        raise ValueError(
            f"{table_name} group keys must not contain missing values: "
            f"{missing_values}"
        )


def _digest_part(digest: _Digest, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _table_fingerprint(table: pd.DataFrame) -> str:
    """Fingerprint every fit-relevant DataFrame value and structural attribute."""

    digest = hashlib.sha256()
    metadata = {
        "columns": [str(value) for value in table.columns],
        "dtypes": [str(value) for value in table.dtypes],
        "index_class": type(table.index).__qualname__,
        "index_names": [
            None if value is None else str(value) for value in table.index.names
        ],
        "shape": list(table.shape),
    }
    _digest_part(
        digest,
        "metadata",
        json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii"),
    )
    row_hashes = pd.util.hash_pandas_object(
        table,
        index=True,
        categorize=False,
    ).to_numpy(dtype=np.uint64, copy=True)
    _digest_part(digest, "row_hashes", row_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_float_array(values: np.ndarray) -> np.ndarray:
    """Return a producer-owned float64 ndarray backed by immutable bytes."""

    owned = np.asarray(values, dtype=np.float64, order="C").copy(order="C")
    immutable = cast(
        np.ndarray,
        np.frombuffer(owned.tobytes(order="C"), dtype=np.float64).reshape(
            owned.shape
        ),
    )
    immutable.setflags(write=False)
    return immutable


def _array_fingerprint(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    metadata = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "writeable": bool(array.flags.writeable),
    }
    _digest_part(
        digest,
        "metadata",
        json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii"),
    )
    _digest_part(
        digest,
        "values",
        np.ascontiguousarray(array).tobytes(order="C"),
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedMeasuresSpec:
    """Frozen design and support policy for one context contrast.

    ``region_key`` may name the context itself (as in the Kuppe region
    contrasts) or a separate categorical adjustment factor.  Batch fields are
    categorical.  Continuous covariates are deliberately unsupported here.
    """

    context_key: str
    reference: str
    target: str
    subject_key: str = "subject_id"
    sample_key: str = "sample_id"
    region_key: str | None = None
    batch_keys: tuple[str, ...] = ()
    min_subjects_per_context: int = 3
    min_subject_clusters: int = 6
    max_condition_number: float = 1.0e10
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        names = _field_names(
            (self.context_key, self.subject_key, self.sample_key),
            name="design keys",
        )
        context_key, subject_key, sample_key = names
        for field_name in ("reference", "target"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must be a canonical non-empty label")
        if self.reference == self.target:
            raise ValueError("reference and target must differ")
        region_key = self.region_key
        if region_key is not None:
            region_key = _field_names((region_key,), name="region_key")[0]
            if region_key in {subject_key, sample_key}:
                raise ValueError("region_key cannot be the subject or sample key")
        batch_keys = _field_names(self.batch_keys, name="batch_keys")
        reserved = {context_key, subject_key, sample_key}
        if reserved.intersection(batch_keys):
            raise ValueError("batch_keys cannot repeat context/subject/sample keys")
        if region_key is not None and region_key != context_key:
            if region_key in batch_keys:
                raise ValueError("region_key cannot also be declared as a batch key")
        for field_name in ("min_subjects_per_context", "min_subject_clusters"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{field_name} must be an integer >= 2")
        if (
            isinstance(self.max_condition_number, bool)
            or not math.isfinite(self.max_condition_number)
            or self.max_condition_number <= 1.0
        ):
            raise ValueError("max_condition_number must be finite and > 1")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        payload = {
            "batch_keys": list(batch_keys),
            "context_key": context_key,
            "max_condition_number": self.max_condition_number,
            "min_subject_clusters": self.min_subject_clusters,
            "min_subjects_per_context": self.min_subjects_per_context,
            "model_version": _MODEL_VERSION,
            "reference": self.reference,
            "region_key": region_key,
            "sample_key": sample_key,
            "schema_version": self.schema_version,
            "subject_key": subject_key,
            "target": self.target,
        }
        object.__setattr__(self, "context_key", context_key)
        object.__setattr__(self, "subject_key", subject_key)
        object.__setattr__(self, "sample_key", sample_key)
        object.__setattr__(self, "region_key", region_key)
        object.__setattr__(self, "batch_keys", batch_keys)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "repeated_measures_spec",
                payload,
                schema_version=self.schema_version,
            ),
        )

    @property
    def adjustment_keys(self) -> tuple[str, ...]:
        """Return unique categorical nuisance factors in model order."""

        region = (
            ()
            if self.region_key is None or self.region_key == self.context_key
            else (self.region_key,)
        )
        return (*region, *self.batch_keys)


@dataclass(frozen=True, slots=True, init=False)
class FrozenRepeatedMeasuresDesign:
    """Producer-owned score-independent design with mutation detection."""

    spec: RepeatedMeasuresSpec
    sample_table: pd.DataFrame
    cell_table: pd.DataFrame
    design_matrix: np.ndarray
    model_columns: tuple[str, ...]
    contrast_vector: np.ndarray
    factor_levels: tuple[tuple[str, tuple[str, ...]], ...]
    design_kind: str
    design_rank: int
    condition_number: float
    status: str
    reason_code: str | None
    design_id: str
    _integrity_digest: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenRepeatedMeasuresDesign is producer-owned; "
            "use freeze_repeated_measures_design()"
        )

    @classmethod
    def _from_frozen(
        cls,
        *,
        spec: RepeatedMeasuresSpec,
        sample_table: pd.DataFrame,
        cell_table: pd.DataFrame,
        design_matrix: np.ndarray,
        model_columns: tuple[str, ...],
        contrast_vector: np.ndarray,
        factor_levels: tuple[tuple[str, tuple[str, ...]], ...],
        design_kind: str,
        design_rank: int,
        condition_number: float,
        status: str,
        reason_code: str | None,
        design_id: str,
    ) -> FrozenRepeatedMeasuresDesign:
        owned_sample_table = sample_table.copy(deep=True)
        owned_cell_table = cell_table.copy(deep=True)
        owned_matrix = _immutable_float_array(design_matrix)
        owned_contrast = _immutable_float_array(contrast_vector)
        self = object.__new__(cls)
        values: dict[str, object] = {
            "spec": spec,
            "sample_table": owned_sample_table,
            "cell_table": owned_cell_table,
            "design_matrix": owned_matrix,
            "model_columns": tuple(model_columns),
            "contrast_vector": owned_contrast,
            "factor_levels": tuple(
                (str(field), tuple(str(level) for level in levels))
                for field, levels in factor_levels
            ),
            "design_kind": design_kind,
            "design_rank": design_rank,
            "condition_number": condition_number,
            "status": status,
            "reason_code": reason_code,
            "design_id": design_id,
            "_integrity_digest": "",
            "_producer_marker": _PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_integrity_digest", self._current_integrity_digest())
        return self

    def _current_integrity_digest(self) -> str:
        payload = {
            "cell_table_fingerprint": _table_fingerprint(self.cell_table),
            "condition_number": self.condition_number,
            "contrast_vector_fingerprint": _array_fingerprint(
                self.contrast_vector
            ),
            "design_id": self.design_id,
            "design_kind": self.design_kind,
            "design_matrix_fingerprint": _array_fingerprint(self.design_matrix),
            "design_rank": self.design_rank,
            "factor_levels": self.factor_levels,
            "model_columns": self.model_columns,
            "reason_code": self.reason_code,
            "sample_table_fingerprint": _table_fingerprint(self.sample_table),
            "spec": {
                "batch_keys": self.spec.batch_keys,
                "context_key": self.spec.context_key,
                "max_condition_number": self.spec.max_condition_number,
                "min_subject_clusters": self.spec.min_subject_clusters,
                "min_subjects_per_context": self.spec.min_subjects_per_context,
                "reference": self.spec.reference,
                "region_key": self.spec.region_key,
                "sample_key": self.spec.sample_key,
                "schema_version": self.spec.schema_version,
                "spec_id": self.spec.spec_id,
                "subject_key": self.spec.subject_key,
                "target": self.spec.target,
            },
            "status": self.status,
        }
        return stable_id(
            "repeated_measures_integrity",
            payload,
            schema_version=self.spec.schema_version,
            digest_length=64,
        )

    def _integrity_valid(self) -> bool:
        if self._producer_marker != _PRODUCER_MARKER:
            return False
        try:
            observed = self._current_integrity_digest()
        except Exception:
            return False
        return hmac.compare_digest(self._integrity_digest, observed)

    @property
    def estimable(self) -> bool:
        return self.status == "ready"


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedMeasuresFit:
    """One context effect with explicit exploratory uncertainty semantics."""

    effect: float
    diagnostic_standard_error: float
    n_observations: int
    n_subject_clusters: int
    n_contrast_subject_clusters: int
    n_reference_subjects: int
    n_target_subjects: int
    n_paired_subjects: int
    n_repeated_subject_clusters: int
    design_kind: str
    design_rank: int
    condition_number: float
    status: str
    reason_code: str | None
    design_id: str
    spec_id: str

    def as_record(self) -> dict[str, object]:
        """Return a flat benchmark record without inferential p/q fields."""

        return {
            "effect": self.effect,
            "effect_semantics": "target_minus_reference_adjusted_mean",
            "diagnostic_standard_error": self.diagnostic_standard_error,
            "uncertainty_semantics": (
                "subject_cluster_cr1_diagnostic_only_no_calibrated_p_or_q"
            ),
            "n_observations": self.n_observations,
            "n_subject_clusters": self.n_subject_clusters,
            "n_contrast_subject_clusters": self.n_contrast_subject_clusters,
            "n_reference_subjects": self.n_reference_subjects,
            "n_target_subjects": self.n_target_subjects,
            "n_paired_subjects": self.n_paired_subjects,
            "n_repeated_subject_clusters": self.n_repeated_subject_clusters,
            "design_kind": self.design_kind,
            "design_rank": self.design_rank,
            "condition_number": self.condition_number,
            "status": self.status,
            "reason_code": self.reason_code,
            "repeated_measures_design_id": self.design_id,
            "repeated_measures_spec_id": self.spec_id,
            "model_version": _MODEL_VERSION,
            "formal_inference_allowed": False,
        }


def _classify_design(cell_table: pd.DataFrame, spec: RepeatedMeasuresSpec) -> str:
    selected = cell_table.loc[
        cell_table[spec.context_key].isin((spec.reference, spec.target))
    ]
    _require_complete_group_keys(
        selected,
        (spec.subject_key, spec.context_key),
        table_name="contrast cell table",
    )
    allocations = [
        set(group[spec.context_key].astype(str))
        for _, group in selected.groupby(spec.subject_key, observed=True, sort=False)
    ]
    paired = [
        spec.reference in contexts and spec.target in contexts
        for contexts in allocations
    ]
    singleton = [len(contexts) == 1 for contexts in allocations]
    if any(paired) and any(singleton):
        return "mixed_paired_unpaired_subject_cluster_ols"
    if any(paired):
        return "paired_subject_cluster_ols"
    return "unpaired_subject_cluster_ols"


def _factor_encoding(
    table: pd.DataFrame,
    spec: RepeatedMeasuresSpec,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    tuple[tuple[str, tuple[str, ...]], ...],
]:
    columns: list[np.ndarray] = [np.ones(len(table), dtype=float)]
    names = ["intercept"]
    contrast = [0.0]
    factor_levels: list[tuple[str, tuple[str, ...]]] = []

    observed_contexts = set(table[spec.context_key].astype(str))
    if not {spec.reference, spec.target}.issubset(observed_contexts):
        raise ValueError("sample design does not contain both contrast contexts")
    context_levels = (
        spec.reference,
        *sorted(observed_contexts.difference({spec.reference})),
    )
    factor_levels.append((spec.context_key, context_levels))
    for level in context_levels[1:]:
        columns.append(table[spec.context_key].eq(level).to_numpy(dtype=float))
        names.append(f"context:{spec.context_key}={level}")
        contrast.append(1.0 if level == spec.target else 0.0)

    for key in spec.adjustment_keys:
        levels = tuple(sorted(table[key].astype(str).unique()))
        factor_levels.append((key, levels))
        for level in levels[1:]:
            columns.append(table[key].eq(level).to_numpy(dtype=float))
            names.append(f"adjustment:{key}={level}")
            contrast.append(0.0)
    return (
        np.column_stack(columns),
        tuple(names),
        np.asarray(contrast, dtype=float),
        tuple(factor_levels),
    )


def freeze_repeated_measures_design(
    sample_design: pd.DataFrame,
    *,
    spec: RepeatedMeasuresSpec,
) -> FrozenRepeatedMeasuresDesign:
    """Freeze a categorical design independently of method scores."""

    if not isinstance(spec, RepeatedMeasuresSpec):
        raise TypeError("spec must be a RepeatedMeasuresSpec")
    required = {
        spec.sample_key,
        spec.subject_key,
        spec.context_key,
        *spec.adjustment_keys,
    }
    missing = required.difference(sample_design.columns)
    if missing:
        raise ValueError(f"sample_design is missing columns: {sorted(missing)}")
    if sample_design.empty:
        raise ValueError("sample_design must not be empty")
    table = sample_design.loc[:, sorted(required)].copy(deep=True)
    _require_complete_group_keys(
        table,
        (spec.sample_key, spec.subject_key, spec.context_key, *spec.adjustment_keys),
        table_name="sample_design",
    )
    for key in required:
        table[key] = table[key].map(lambda value, k=key: _label(value, field_name=k))
    if table[spec.sample_key].duplicated().any():
        raise ValueError("sample_design must contain exactly one row per sample")

    cell_keys = (spec.subject_key, spec.context_key, *spec.adjustment_keys)
    cell_table = (
        table.loc[:, list(cell_keys)]
        .drop_duplicates()
        .sort_values(list(cell_keys), kind="stable", ignore_index=True)
    )
    cell_table["__design_row"] = np.arange(len(cell_table), dtype=int)
    sample_table = table.merge(
        cell_table,
        on=list(cell_keys),
        how="left",
        validate="many_to_one",
    ).sort_values(spec.sample_key, kind="stable", ignore_index=True)
    matrix, columns, contrast, levels = _factor_encoding(cell_table, spec)
    rank = int(np.linalg.matrix_rank(matrix))
    condition_number = float(np.linalg.cond(matrix))
    reason: str | None = None
    if rank < matrix.shape[1]:
        reason = "rank_deficient_declared_design"
    elif not math.isfinite(condition_number) or (
        condition_number > spec.max_condition_number
    ):
        reason = "ill_conditioned_declared_design"
    design_kind = _classify_design(cell_table, spec)
    manifest = sample_table.loc[:, [*sorted(required), "__design_row"]].to_dict(
        orient="records"
    )
    design_id = stable_id(
        "repeated_measures_design",
        {
            "contrast_vector": contrast.tolist(),
            "factor_levels": levels,
            "model_columns": columns,
            "model_version": _MODEL_VERSION,
            "sample_manifest": manifest,
            "spec_id": spec.spec_id,
        },
        schema_version=spec.schema_version,
    )
    return FrozenRepeatedMeasuresDesign._from_frozen(
        spec=spec,
        sample_table=sample_table,
        cell_table=cell_table,
        design_matrix=matrix,
        model_columns=columns,
        contrast_vector=contrast,
        factor_levels=levels,
        design_kind=design_kind,
        design_rank=rank,
        condition_number=condition_number,
        status="ready" if reason is None else "not_estimable",
        reason_code=reason,
        design_id=design_id,
    )


def _not_estimable(
    design: FrozenRepeatedMeasuresDesign,
    *,
    reason_code: str,
    n_observations: int = 0,
    n_subject_clusters: int = 0,
    n_contrast_subject_clusters: int = 0,
    n_reference_subjects: int = 0,
    n_target_subjects: int = 0,
    n_paired_subjects: int = 0,
    n_repeated_subject_clusters: int = 0,
    design_rank: int | None = None,
    condition_number: float | None = None,
) -> RepeatedMeasuresFit:
    return RepeatedMeasuresFit(
        effect=math.nan,
        diagnostic_standard_error=math.nan,
        n_observations=n_observations,
        n_subject_clusters=n_subject_clusters,
        n_contrast_subject_clusters=n_contrast_subject_clusters,
        n_reference_subjects=n_reference_subjects,
        n_target_subjects=n_target_subjects,
        n_paired_subjects=n_paired_subjects,
        n_repeated_subject_clusters=n_repeated_subject_clusters,
        design_kind=design.design_kind,
        design_rank=design.design_rank if design_rank is None else design_rank,
        condition_number=(
            design.condition_number
            if condition_number is None
            else condition_number
        ),
        status="not_estimable",
        reason_code=reason_code,
        design_id=design.design_id,
        spec_id=design.spec.spec_id,
    )


def _support(
    cells: pd.DataFrame,
    spec: RepeatedMeasuresSpec,
) -> tuple[int, int, int, int, int, int]:
    _require_complete_group_keys(
        cells,
        (spec.subject_key, spec.context_key, *spec.adjustment_keys),
        table_name="edge cell table",
    )
    reference = set(
        cells.loc[
            cells[spec.context_key].eq(spec.reference), spec.subject_key
        ].astype(str)
    )
    target = set(
        cells.loc[
            cells[spec.context_key].eq(spec.target), spec.subject_key
        ].astype(str)
    )
    counts = cells.groupby(spec.subject_key, observed=True).size()
    return (
        len(reference),
        len(target),
        len(reference.intersection(target)),
        len(reference.union(target)),
        int(counts.size),
        int(counts.gt(1).sum()),
    )


def fit_repeated_measures_contrast(
    design: FrozenRepeatedMeasuresDesign,
    scores: pd.DataFrame,
    *,
    value_key: str = "value",
) -> RepeatedMeasuresFit:
    """Fit a mixed paired/unpaired contrast with subject-cluster CR1 SE.

    ``scores`` must contain at most one finite value per sample.  Missing sample
    rows are allowed and trigger edge-specific support and rank checks.
    """

    if not isinstance(design, FrozenRepeatedMeasuresDesign):
        raise TypeError("design must be a FrozenRepeatedMeasuresDesign")
    if not design._integrity_valid():
        return _not_estimable(
            design,
            reason_code="frozen_design_integrity_violation",
        )
    spec = design.spec
    required = {spec.sample_key, value_key}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"scores are missing columns: {sorted(missing)}")
    values = scores.loc[:, [spec.sample_key, value_key]].copy()
    if values[spec.sample_key].isna().any():
        raise ValueError("score sample identifiers must not be missing")
    values[spec.sample_key] = values[spec.sample_key].map(
        lambda value: _label(value, field_name=spec.sample_key)
    )
    if values[spec.sample_key].duplicated().any():
        raise ValueError("scores must contain at most one row per sample")
    numeric = pd.to_numeric(values[value_key], errors="coerce")
    invalid = values[value_key].notna() & numeric.isna()
    if invalid.any() or np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
        raise ValueError("observed scores must be finite numeric values")
    values[value_key] = numeric
    unknown = set(values[spec.sample_key]).difference(
        set(design.sample_table[spec.sample_key])
    )
    if unknown:
        raise ValueError(
            "scores contain samples outside the frozen design: "
            f"{sorted(unknown)}"
        )
    if not design.estimable:
        return _not_estimable(
            design,
            reason_code=design.reason_code or "declared_design_not_estimable",
        )

    mapped = design.sample_table.merge(
        values,
        on=spec.sample_key,
        how="left",
        validate="one_to_one",
    )
    _require_complete_group_keys(
        mapped,
        (
            "__design_row",
            spec.subject_key,
            spec.context_key,
            *spec.adjustment_keys,
        ),
        table_name="mapped score table",
    )
    cell_values = mapped.groupby("__design_row", observed=True, sort=True)[
        value_key
    ].mean()
    usable_rows = cell_values.dropna().index.to_numpy(dtype=int)
    cells = design.cell_table.set_index("__design_row").loc[usable_rows].copy()
    response = cell_values.loc[usable_rows].to_numpy(dtype=float)
    (
        n_reference,
        n_target,
        n_paired,
        n_contrast_clusters,
        n_clusters,
        n_repeated,
    ) = _support(cells, spec)
    support = {
        "n_observations": len(response),
        "n_subject_clusters": n_clusters,
        "n_contrast_subject_clusters": n_contrast_clusters,
        "n_reference_subjects": n_reference,
        "n_target_subjects": n_target,
        "n_paired_subjects": n_paired,
        "n_repeated_subject_clusters": n_repeated,
    }
    if min(n_reference, n_target) < spec.min_subjects_per_context:
        return _not_estimable(
            design,
            reason_code="insufficient_subjects_per_context",
            **support,
        )
    if n_contrast_clusters < spec.min_subject_clusters:
        return _not_estimable(
            design,
            reason_code="insufficient_subject_clusters",
            **support,
        )

    matrix = design.design_matrix[usable_rows]
    rank = int(np.linalg.matrix_rank(matrix))
    condition_number = float(np.linalg.cond(matrix))
    if rank < matrix.shape[1]:
        return _not_estimable(
            design,
            reason_code="rank_deficient_edge_design",
            design_rank=rank,
            condition_number=condition_number,
            **support,
        )
    if not math.isfinite(condition_number) or (
        condition_number > spec.max_condition_number
    ):
        return _not_estimable(
            design,
            reason_code="ill_conditioned_edge_design",
            design_rank=rank,
            condition_number=condition_number,
            **support,
        )
    if len(response) <= matrix.shape[1]:
        return _not_estimable(
            design,
            reason_code="insufficient_residual_degrees_of_freedom",
            design_rank=rank,
            condition_number=condition_number,
            **support,
        )
    if n_clusters <= matrix.shape[1]:
        return _not_estimable(
            design,
            reason_code="insufficient_clusters_for_sandwich",
            design_rank=rank,
            condition_number=condition_number,
            **support,
        )

    gram_inverse = np.linalg.inv(matrix.T @ matrix)
    coefficients = gram_inverse @ matrix.T @ response
    residual = response - matrix @ coefficients
    subjects = cells[spec.subject_key].astype(str).to_numpy()
    meat = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    for subject in sorted(set(subjects)):
        selected = subjects == subject
        cluster_score = matrix[selected].T @ residual[selected]
        meat += np.outer(cluster_score, cluster_score)
    correction = (n_clusters / (n_clusters - 1)) * (
        (len(response) - 1) / (len(response) - rank)
    )
    covariance = correction * gram_inverse @ meat @ gram_inverse
    variance = float(design.contrast_vector @ covariance @ design.contrast_vector)
    if not math.isfinite(variance) or variance < -1.0e-12:
        return _not_estimable(
            design,
            reason_code="invalid_subject_cluster_covariance",
            design_rank=rank,
            condition_number=condition_number,
            **support,
        )
    effect = float(design.contrast_vector @ coefficients)
    standard_error = math.sqrt(max(0.0, variance))
    return RepeatedMeasuresFit(
        effect=effect,
        diagnostic_standard_error=standard_error,
        design_kind=design.design_kind,
        design_rank=rank,
        condition_number=condition_number,
        status="exploratory",
        reason_code=None,
        design_id=design.design_id,
        spec_id=spec.spec_id,
        **support,
    )


def repeated_measures_effects(
    table: pd.DataFrame,
    sample_design: pd.DataFrame,
    *,
    spec: RepeatedMeasuresSpec,
    group_keys: Sequence[str],
    value_key: str = "value",
    status_key: str | None = None,
    observed_statuses: Sequence[str] = ("observed", "ok", "exploratory"),
) -> pd.DataFrame:
    """Apply one frozen repeated-measures design to grouped benchmark scores."""

    groups = _field_names(group_keys, name="group_keys")
    if not groups:
        raise ValueError("group_keys must contain at least one identity field")
    required = {spec.sample_key, value_key, *groups}
    if status_key is not None:
        required.add(status_key)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"table is missing columns: {sorted(missing)}")
    _require_complete_group_keys(
        table,
        (*groups, spec.sample_key),
        table_name="grouped score table",
    )
    if table.duplicated([*groups, spec.sample_key]).any():
        raise ValueError("table contains repeated group/sample score rows")
    design = freeze_repeated_measures_design(sample_design, spec=spec)
    allowed = frozenset(str(value) for value in observed_statuses)
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = groups[0] if len(groups) == 1 else list(groups)
    for keys, group in table.groupby(grouper, observed=True, sort=False):
        key_tuple = (
            (keys,)
            if len(groups) == 1
            else cast(tuple[object, ...], keys)
        )
        identity = dict(zip(groups, key_tuple, strict=True))
        if status_key is not None:
            usable = group.loc[group[status_key].astype(str).isin(allowed)]
        else:
            usable = group
        fit = fit_repeated_measures_contrast(
            design,
            usable,
            value_key=value_key,
        )
        rows.append(identity | fit.as_record())
    return pd.DataFrame.from_records(rows)


__all__ = [
    "FrozenRepeatedMeasuresDesign",
    "RepeatedMeasuresFit",
    "RepeatedMeasuresSpec",
    "fit_repeated_measures_contrast",
    "freeze_repeated_measures_design",
    "repeated_measures_effects",
]
