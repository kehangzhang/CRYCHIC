"""Frozen subject-cluster designs for repeated-measures receiver responses."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id

from .audit import audit_sample_design, default_design_formula
from .context_encoding import canonical_context, node_context_fields
from .contrasts import ContrastSpec

_MODEL_VERSION = "formula_ols_subject_cluster_cr1_v1"
_SCHEMA_VERSION = "1.0.0"
_PRODUCER_MARKER = "crychic.design.repeated_measures.v1"


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(f"{field_name} must contain canonical non-empty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique names")
    return result


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must contain canonical non-empty strings")
    return value


def _plain_scalar(value: object, *, field_name: str) -> Hashable:
    plain = value.item() if isinstance(value, np.generic) else value
    try:
        missing = pd.isna(cast(Any, plain))
    except (TypeError, ValueError):
        missing = False
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain scalar values")
    if bool(missing):
        raise ValueError(f"{field_name} must not contain missing values")
    if isinstance(plain, float) and not math.isfinite(plain):
        raise ValueError(f"{field_name} numeric values must be finite")
    if not isinstance(plain, (bool, int, float, str)):
        raise TypeError(
            f"{field_name} values must be JSON-compatible scalar labels or numbers"
        )
    return plain


def _typed_value(value: object, *, field_name: str) -> dict[str, object]:
    plain = _plain_scalar(value, field_name=field_name)
    return {
        "type": f"{type(plain).__module__}.{type(plain).__qualname__}",
        "value": plain,
    }


def _array_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    if np.any(~np.isfinite(array)):
        raise ValueError("repeated-measures design arrays must be finite")
    digest = hashlib.sha256()
    digest.update(canonical_json({"shape": list(array.shape)}).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_array(values: np.ndarray) -> np.ndarray:
    owned = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if np.any(~np.isfinite(owned)):
        raise ValueError("repeated-measures design arrays must be finite")
    result = cast(
        np.ndarray,
        np.frombuffer(owned.tobytes(order="C"), dtype="<f8").reshape(owned.shape),
    )
    result.setflags(write=False)
    return result


def _immutable_indices(values: Sequence[int]) -> np.ndarray:
    owned = np.asarray(values, dtype="<i8", order="C").copy(order="C")
    result = cast(
        np.ndarray,
        np.frombuffer(owned.tobytes(order="C"), dtype="<i8").reshape(owned.shape),
    )
    result.setflags(write=False)
    return result


def _vector_estimable(
    matrix: np.ndarray, vector: np.ndarray, *, tolerance: float = 1.0e-8
) -> bool:
    if matrix.size == 0 or np.linalg.norm(vector) <= tolerance:
        return False
    projection = np.linalg.pinv(matrix) @ matrix
    residual = vector - vector @ projection
    return bool(
        np.linalg.norm(residual) <= tolerance * max(1.0, float(np.linalg.norm(vector)))
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedMeasuresDesignSpec:
    """Score-independent formula and support policy for one receiver contrast."""

    context_keys: tuple[str, ...]
    covariates: tuple[str, ...] = ()
    categorical_covariates: tuple[str, ...] = ()
    formula: str | None = None
    sample_key: str = "sample_id"
    subject_key: str = "subject_id"
    subject_fixed_effects: bool = False
    min_subjects_per_context: int = 3
    min_subject_clusters: int = 6
    max_condition_number: float = 1.0e10
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        contexts = _names(self.context_keys, field_name="context_keys")
        covariates = _names(self.covariates, field_name="covariates")
        categorical = _names(
            self.categorical_covariates,
            field_name="categorical_covariates",
        )
        sample_key = _identifier(self.sample_key, field_name="sample_key")
        subject_key = _identifier(self.subject_key, field_name="subject_key")
        if sample_key == subject_key:
            raise ValueError("sample_key and subject_key must differ")
        roles = (*contexts, *covariates, sample_key, subject_key)
        if len(set(roles)) != len(roles):
            raise ValueError("design field roles must be distinct")
        unknown_categorical = set(categorical).difference(covariates)
        if unknown_categorical:
            raise ValueError(
                "categorical_covariates must be declared covariates: "
                + ",".join(sorted(unknown_categorical))
            )
        if not isinstance(self.subject_fixed_effects, bool):
            raise TypeError("subject_fixed_effects must be boolean")
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
        formula = self.formula or default_design_formula(contexts, covariates)
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError("formula must be a non-empty one-sided formula")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        payload = {
            "categorical_covariates": list(categorical),
            "context_keys": list(contexts),
            "covariates": list(covariates),
            "formula": formula,
            "max_condition_number": self.max_condition_number,
            "min_subject_clusters": self.min_subject_clusters,
            "min_subjects_per_context": self.min_subjects_per_context,
            "model_version": _MODEL_VERSION,
            "sample_key": sample_key,
            "schema_version": self.schema_version,
            "subject_fixed_effects": self.subject_fixed_effects,
            "subject_key": subject_key,
        }
        object.__setattr__(self, "context_keys", contexts)
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "categorical_covariates", categorical)
        object.__setattr__(self, "formula", formula)
        object.__setattr__(self, "sample_key", sample_key)
        object.__setattr__(self, "subject_key", subject_key)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "repeated_measures_design_spec",
                payload,
                schema_version=self.schema_version,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the frozen design policy without data-dependent fields."""

        return {
            "context_keys": list(self.context_keys),
            "covariates": list(self.covariates),
            "categorical_covariates": list(self.categorical_covariates),
            "formula": self.formula,
            "sample_key": self.sample_key,
            "subject_key": self.subject_key,
            "subject_fixed_effects": self.subject_fixed_effects,
            "min_subjects_per_context": self.min_subjects_per_context,
            "min_subject_clusters": self.min_subject_clusters,
            "max_condition_number": self.max_condition_number,
            "schema_version": self.schema_version,
            "spec_id": self.spec_id,
        }


@dataclass(frozen=True, slots=True, init=False)
class FrozenRepeatedMeasuresDesign:
    """Producer-owned subject-cluster design with sample-to-cell lineage."""

    spec: RepeatedMeasuresDesignSpec
    contrast: ContrastSpec
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_cell_indices: np.ndarray
    cell_ids: tuple[str, ...]
    cell_subject_ids: tuple[str, ...]
    cell_context_ids: tuple[str, ...]
    cell_sample_counts: tuple[int, ...]
    contrast_context_ids: tuple[str, ...]
    contrast_context_weights: tuple[float, ...]
    design_matrix: np.ndarray
    model_columns: tuple[str, ...]
    contrast_vector: np.ndarray
    design_rank: int
    condition_number: float
    residual_df: int
    n_subject_clusters: int
    n_contrast_subject_clusters: int
    n_repeated_subject_clusters: int
    n_complete_contrast_subjects: int
    context_subject_counts: tuple[tuple[str, int], ...]
    status: str
    reason_code: str | None
    design_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenRepeatedMeasuresDesign is producer-owned; "
            "use freeze_repeated_measures_design()"
        )

    @classmethod
    def _from_frozen(cls, **values: object) -> FrozenRepeatedMeasuresDesign:
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_producer_marker", _PRODUCER_MARKER)
        return self

    @property
    def estimable(self) -> bool:
        return self.status == "ready"

    @property
    def model_version(self) -> str:
        return _MODEL_VERSION

    def _identity_payload(self) -> dict[str, object]:
        return {
            "cell_rows": [
                {
                    "cell_id": cell_id,
                    "context_id": context_id,
                    "n_samples": n_samples,
                    "subject_id": subject_id,
                }
                for cell_id, subject_id, context_id, n_samples in zip(
                    self.cell_ids,
                    self.cell_subject_ids,
                    self.cell_context_ids,
                    self.cell_sample_counts,
                    strict=True,
                )
            ],
            "condition_number": (
                self.condition_number if math.isfinite(self.condition_number) else None
            ),
            "contrast_context_ids": list(self.contrast_context_ids),
            "contrast_context_weights": list(self.contrast_context_weights),
            "contrast": self.contrast.to_dict(),
            "contrast_vector_digest": _array_digest(self.contrast_vector),
            "design_matrix_digest": _array_digest(self.design_matrix),
            "design_rank": self.design_rank,
            "context_subject_counts": [
                {"context_id": context_id, "n_subjects": count}
                for context_id, count in self.context_subject_counts
            ],
            "model_columns": list(self.model_columns),
            "model_version": _MODEL_VERSION,
            "n_complete_contrast_subjects": self.n_complete_contrast_subjects,
            "n_contrast_subject_clusters": self.n_contrast_subject_clusters,
            "n_repeated_subject_clusters": self.n_repeated_subject_clusters,
            "n_subject_clusters": self.n_subject_clusters,
            "reason_code": self.reason_code,
            "residual_df": self.residual_df,
            "sample_rows": [
                {
                    "cell_index": int(cell_index),
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                }
                for sample_id, subject_id, cell_index in zip(
                    self.sample_ids,
                    self.sample_subject_ids,
                    self.sample_cell_indices,
                    strict=True,
                )
            ],
            "spec": self.spec.to_dict(),
            "status": self.status,
        }

    def _require_intact(self) -> None:
        try:
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and not self.design_matrix.flags.writeable
                and not self.contrast_vector.flags.writeable
                and not self.sample_cell_indices.flags.writeable
                and self.design_matrix.shape
                == (len(self.cell_ids), len(self.model_columns))
                and self.contrast_vector.shape == (len(self.model_columns),)
                and self.sample_cell_indices.shape == (len(self.sample_ids),)
                and self.design_id
                == stable_id(
                    "repeated_measures_design",
                    self._identity_payload(),
                    schema_version=self.spec.schema_version,
                )
            )
        except Exception as error:
            raise ContractError(
                "Repeated-measures design failed integrity validation",
                code="repeated_measures_design_integrity_violation",
                field="design_id",
                remediation="Refreeze the design from intact sample metadata",
            ) from error
        if not valid:
            raise ContractError(
                "Repeated-measures design failed integrity validation",
                code="repeated_measures_design_integrity_violation",
                field="design_id",
                remediation="Refreeze the design from intact sample metadata",
            )

    def to_dict(self) -> dict[str, object]:
        """Return score-independent design diagnostics and lineage."""

        self._require_intact()
        return {
            "design_id": self.design_id,
            "spec": self.spec.to_dict(),
            "contrast": self.contrast.to_dict(),
            "model_version": _MODEL_VERSION,
            "model_columns": list(self.model_columns),
            "design_rank": self.design_rank,
            "condition_number": (
                self.condition_number if math.isfinite(self.condition_number) else None
            ),
            "residual_df": self.residual_df,
            "n_samples": len(self.sample_ids),
            "n_design_cells": len(self.cell_ids),
            "n_subject_clusters": self.n_subject_clusters,
            "n_contrast_subject_clusters": self.n_contrast_subject_clusters,
            "n_repeated_subject_clusters": self.n_repeated_subject_clusters,
            "n_complete_contrast_subjects": self.n_complete_contrast_subjects,
            "context_subject_counts": [
                {"context_id": context_id, "n_subjects": count}
                for context_id, count in self.context_subject_counts
            ],
            "status": self.status,
            "reason_code": self.reason_code,
        }


def _context_node(row: pd.Series, context_keys: tuple[str, ...]) -> Hashable:
    canonical = canonical_context(row, context_keys)
    result: Hashable = canonical[0][1] if len(canonical) == 1 else canonical
    return result


def _support_diagnostics(
    cell_subject_ids: tuple[str, ...],
    cell_context_ids: tuple[str, ...],
    contrast_context_ids: tuple[str, ...],
) -> tuple[int, int, int, int, tuple[tuple[str, int], ...]]:
    subject_sets = {
        context_id: {
            subject_id
            for subject_id, observed_context in zip(
                cell_subject_ids, cell_context_ids, strict=True
            )
            if observed_context == context_id
        }
        for context_id in contrast_context_ids
    }
    contrast_subjects = set().union(*subject_sets.values())
    complete = set.intersection(*subject_sets.values()) if subject_sets else set()
    counts_by_subject: dict[str, int] = {}
    for subject_id in contrast_subjects:
        counts_by_subject[subject_id] = sum(
            subject_id in subject_sets[context_id]
            for context_id in contrast_context_ids
        )
    repeated = sum(count > 1 for count in counts_by_subject.values())
    counts = tuple(
        (context_id, len(subject_sets[context_id]))
        for context_id in contrast_context_ids
    )
    return (
        len(set(cell_subject_ids)),
        len(contrast_subjects),
        repeated,
        len(complete),
        counts,
    )


def freeze_repeated_measures_design(
    sample_metadata: pd.DataFrame,
    *,
    contrast: ContrastSpec,
    spec: RepeatedMeasuresDesignSpec,
) -> FrozenRepeatedMeasuresDesign:
    """Freeze one formula design while preserving subject-cluster membership."""

    if not isinstance(spec, RepeatedMeasuresDesignSpec):
        raise TypeError("spec must be a RepeatedMeasuresDesignSpec")
    if not isinstance(contrast, ContrastSpec):
        raise TypeError("contrast must be a ContrastSpec")
    required = {
        spec.sample_key,
        spec.subject_key,
        *spec.context_keys,
        *spec.covariates,
    }
    missing = required.difference(sample_metadata.columns)
    if missing:
        raise ValueError(f"sample_metadata is missing columns: {sorted(missing)}")
    if sample_metadata.empty:
        raise ValueError("sample_metadata must not be empty")
    table = sample_metadata.loc[:, sorted(required)].copy(deep=True)
    for key in required:
        table[key] = table[key].map(
            lambda value, name=key: _plain_scalar(value, field_name=name)
        )
    table[spec.sample_key] = table[spec.sample_key].map(
        lambda value: _identifier(value, field_name=spec.sample_key)
    )
    table[spec.subject_key] = table[spec.subject_key].map(
        lambda value: _identifier(value, field_name=spec.subject_key)
    )
    if table[spec.sample_key].duplicated().any():
        raise ValueError("sample_metadata must contain exactly one row per sample")
    table = table.sort_values(spec.sample_key, kind="stable", ignore_index=True)

    cell_fields = (spec.subject_key, *spec.context_keys, *spec.covariates)
    table["__cell_key"] = table.apply(
        lambda row: canonical_json(
            {
                "fields": [
                    {
                        "name": key,
                        **_typed_value(row[key], field_name=key),
                    }
                    for key in cell_fields
                ]
            }
        ),
        axis=1,
    )
    unique_cells = (
        table.drop_duplicates("__cell_key", keep="first")
        .sort_values("__cell_key", kind="stable", ignore_index=True)
        .copy()
    )
    unique_cells["__cell_id"] = unique_cells["__cell_key"].map(
        lambda key: stable_id(
            "repeated_measure_cell",
            {"cell_key": key, "spec_id": spec.spec_id},
            schema_version=spec.schema_version,
        )
    )
    cell_index = {key: index for index, key in enumerate(unique_cells["__cell_key"])}
    sample_cell_indices = _immutable_indices(
        tuple(cell_index[key] for key in table["__cell_key"])
    )
    sample_counts = table.groupby("__cell_key", sort=False).size().to_dict()
    cell_sample_counts = tuple(
        int(sample_counts[key]) for key in unique_cells["__cell_key"]
    )

    audit = audit_sample_design(
        unique_cells,
        context_keys=spec.context_keys,
        covariates=spec.covariates,
        categorical_covariates=spec.categorical_covariates,
        formula=spec.formula,
        sample_key="__cell_id",
        condition_number_warning=spec.max_condition_number,
    )
    matrix = audit.design_matrix.to_numpy(dtype=np.float64)
    model_columns = audit.column_names
    reason: str | None = None
    coefficient_contrast: np.ndarray = np.zeros(matrix.shape[1], dtype=np.float64)
    if not audit.ready:
        reason = audit.reason_codes[0] if audit.reason_codes else "design_blocked"
    elif not contrast.estimable:
        reason = contrast.reason_code or "contrast_declared_not_estimable"
    else:
        try:
            coefficient_contrast = audit.coefficient_contrast(contrast)
        except ValueError:
            reason = "contrast_context_absent_from_design"
        else:
            if not audit.contrast_estimable(contrast):
                reason = "contrast_not_estimable"

    context_nodes = tuple(
        _context_node(row, spec.context_keys) for _, row in unique_cells.iterrows()
    )
    cell_context_ids = tuple(
        node_context_fields(node, spec.context_keys)[0] for node in context_nodes
    )
    contrast_fields = tuple(
        (
            node_context_fields(node, spec.context_keys)[0],
            float(weight),
        )
        for node, weight in contrast.weights.items()
    )
    contrast_context_ids = tuple(item[0] for item in contrast_fields)
    contrast_context_weights = tuple(item[1] for item in contrast_fields)
    cell_subject_ids = tuple(unique_cells[spec.subject_key].astype(str))

    if spec.subject_fixed_effects:
        subjects = tuple(sorted(set(cell_subject_ids)))
        codes = {subject: index for index, subject in enumerate(subjects)}
        subject_columns = np.zeros((len(cell_subject_ids), max(0, len(subjects) - 1)))
        for row_index, subject in enumerate(cell_subject_ids):
            subject_index = codes[subject]
            if subject_index > 0:
                subject_columns[row_index, subject_index - 1] = 1.0
        matrix = np.column_stack((matrix, subject_columns))
        coefficient_contrast = np.concatenate(
            (coefficient_contrast, np.zeros(subject_columns.shape[1]))
        )
        model_columns = (
            *model_columns,
            *(f"subject_fixed_effect:{subject}" for subject in subjects[1:]),
        )
        if reason is None and not _vector_estimable(matrix, coefficient_contrast):
            reason = "subject_fixed_effects_absorb_contrast"

    rank = int(np.linalg.matrix_rank(matrix))
    condition_number = float(np.linalg.cond(matrix)) if matrix.size else math.inf
    residual_df = len(matrix) - rank
    (
        n_subject_clusters,
        n_contrast_subject_clusters,
        n_repeated_subject_clusters,
        n_complete_contrast_subjects,
        context_subject_counts,
    ) = _support_diagnostics(
        cell_subject_ids,
        cell_context_ids,
        contrast_context_ids,
    )
    if reason is None and any(
        count < spec.min_subjects_per_context for _, count in context_subject_counts
    ):
        reason = "insufficient_subjects_per_context"
    if reason is None and n_contrast_subject_clusters < spec.min_subject_clusters:
        reason = "insufficient_subject_clusters"
    if reason is None and rank < matrix.shape[1]:
        reason = (
            "rank_deficient_subject_fixed_effect_design"
            if spec.subject_fixed_effects
            else "rank_deficient_repeated_measures_design"
        )
    if reason is None and (
        not math.isfinite(condition_number)
        or condition_number > spec.max_condition_number
    ):
        reason = "ill_conditioned_repeated_measures_design"
    if reason is None and residual_df < 1:
        reason = "insufficient_residual_degrees_of_freedom"

    frozen_matrix = _immutable_array(matrix)
    frozen_contrast = _immutable_array(coefficient_contrast)
    values: dict[str, object] = {
        "spec": spec,
        "contrast": contrast,
        "sample_ids": tuple(table[spec.sample_key].astype(str)),
        "sample_subject_ids": tuple(table[spec.subject_key].astype(str)),
        "sample_cell_indices": sample_cell_indices,
        "cell_ids": tuple(unique_cells["__cell_id"].astype(str)),
        "cell_subject_ids": cell_subject_ids,
        "cell_context_ids": cell_context_ids,
        "cell_sample_counts": cell_sample_counts,
        "contrast_context_ids": contrast_context_ids,
        "contrast_context_weights": contrast_context_weights,
        "design_matrix": frozen_matrix,
        "model_columns": model_columns,
        "contrast_vector": frozen_contrast,
        "design_rank": rank,
        "condition_number": condition_number,
        "residual_df": residual_df,
        "n_subject_clusters": n_subject_clusters,
        "n_contrast_subject_clusters": n_contrast_subject_clusters,
        "n_repeated_subject_clusters": n_repeated_subject_clusters,
        "n_complete_contrast_subjects": n_complete_contrast_subjects,
        "context_subject_counts": context_subject_counts,
        "status": "ready" if reason is None else "not_estimable",
        "reason_code": reason,
    }
    temporary = FrozenRepeatedMeasuresDesign._from_frozen(
        **values,
        design_id="pending",
    )
    design_id = stable_id(
        "repeated_measures_design",
        temporary._identity_payload(),
        schema_version=spec.schema_version,
    )
    return FrozenRepeatedMeasuresDesign._from_frozen(
        **values,
        design_id=design_id,
    )


__all__ = [
    "FrozenRepeatedMeasuresDesign",
    "RepeatedMeasuresDesignSpec",
    "freeze_repeated_measures_design",
]
