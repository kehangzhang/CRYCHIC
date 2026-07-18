"""Fold-scoped receiver response artifacts with frozen design lineage."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id
from crychic.core._validation import (
    record_validation,
    validation_is_cached,
    validation_scope,
)
from crychic.design import (
    FrozenDesignApplication,
    FrozenDesignEncoder,
    node_context_fields,
)
from crychic.pseudobulk import PseudobulkDataset

_CPM_SCALE = 1_000_000.0
_TRAINING_PRODUCER_MARKER = "crychic.response.fold_gene_response.v2"
_TRAINING_SCHEMA_VERSION = "2"
_APPLICATION_PRODUCER_MARKER = "crychic.response.fold_gene_application.v1"
_INDEPENDENT_METHOD = "frozen_formula_independent_subjects_v1"
_PAIRED_METHOD = "frozen_formula_paired_subject_effects_v1"
_NOT_ESTIMABLE_METHOD = "not_estimable"


def _scope_name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _names(
    values: Sequence[str], *, field_name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if (not result and not allow_empty) or any(
        not isinstance(value, str) or not value for value in result
    ):
        qualifier = "" if allow_empty else " non-empty"
        raise ValueError(f"{field_name} must contain unique{qualifier} strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique strings")
    return result


def _aligned_names(
    values: Sequence[str], *, length: int, field_name: str
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if len(result) != length or any(
        not isinstance(value, str) or not value for value in result
    ):
        raise ValueError(f"{field_name} must contain {length} aligned strings")
    return result


def _canonical_name_key(value: str) -> str:
    """Match the typed canonical sample ordering frozen by the design encoder."""

    result: str = canonical_json({"type": "builtins.str", "value": value})
    return result


def _immutable_array(values: np.ndarray) -> np.ndarray:
    owned = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if np.any(np.isinf(owned)):
        raise ValueError("response artifact arrays cannot contain infinity")
    # Canonicalize NaN payload bits before hashing and freezing.
    owned[np.isnan(owned)] = np.nan
    result = cast(
        np.ndarray,
        np.frombuffer(owned.tobytes(order="C"), dtype="<f8").reshape(owned.shape),
    )
    result.setflags(write=False)
    return result


def _is_immutable_byte_backed(values: np.ndarray) -> bool:
    if values.flags.writeable or not values.flags.c_contiguous:
        return False
    base: object = values
    while isinstance(base, np.ndarray):
        base = base.base
    return isinstance(base, bytes)


def _numeric_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if np.any(np.isinf(array)):
        raise ValueError("response digest arrays cannot contain infinity")
    digest = hashlib.sha256()
    digest.update(b'{"components":{"float64_tokens":[')
    flattened = array.ravel(order="C")
    chunk_size = 16_384
    for start in range(0, flattened.size, chunk_size):
        chunk = flattened[start : start + chunk_size]
        tokens = [
            "null"
            if math.isnan(numeric := float(value))
            else f'"{numeric.hex()}"'
            for value in chunk
        ]
        if start:
            digest.update(b",")
        digest.update(",".join(tokens).encode("ascii"))
    digest.update(b'],"shape":[')
    digest.update(",".join(str(size) for size in array.shape).encode("ascii"))
    digest.update(b']},"kind":"fold_response_array","schema_version":"1"}')
    return f"fold_response_array_{digest.hexdigest()}"


def _row_payload(
    sample_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> list[dict[str, str]]:
    return [
        {
            "context_id": context_id,
            "sample_id": sample_id,
            "subject_id": subject_id,
        }
        for sample_id, subject_id, context_id in zip(
            sample_ids, subject_ids, context_ids, strict=True
        )
    ]


def _context_identifier(context: object, context_keys: tuple[str, ...]) -> str:
    if not isinstance(context, tuple):
        raise ValueError("aggregate context must be a canonical tuple")
    mapping: dict[str, Hashable] = {}
    for item in context:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("aggregate context entries must be (field, value) pairs")
        key, value = item
        if not isinstance(key, str) or key in mapping:
            raise ValueError("aggregate context fields must be unique strings")
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, Hashable):
            raise ValueError("aggregate context values must be hashable")
        mapping[key] = value
    if set(mapping) != set(context_keys):
        raise ValueError(
            "aggregate context fields do not match the frozen design context keys"
        )
    identifier: str = node_context_fields(
        tuple((key, mapping[key]) for key in sorted(mapping)), context_keys
    )[0]
    return identifier


def _contrast_context_weights(
    encoder: FrozenDesignEncoder,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for node, weight in encoder.contrast.weights.items():
        identifier = node_context_fields(node, encoder.context_keys)[0]
        if identifier in result:
            raise ValueError("contrast nodes collapse to duplicate canonical contexts")
        result[identifier] = float(weight)
    return result


@dataclass(frozen=True, slots=True)
class _ExtractedResponse:
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_context_ids: tuple[str, ...]
    values: np.ndarray
    missing_sample_ids: tuple[str, ...]


def _validate_matrix_row(aggregate: PseudobulkDataset, row: pd.Series) -> int:
    raw_index = row["matrix_row"]
    if pd.isna(raw_index):
        raise ValueError(
            f"state-eligible unit {row['unit_id']!r} has no aggregate matrix row"
        )
    if isinstance(raw_index, (bool, np.bool_)):
        raise ValueError("aggregate matrix_row must contain integer positions")
    index = int(raw_index)
    if float(raw_index) != index or not 0 <= index < aggregate.counts.shape[0]:
        raise ValueError("aggregate matrix_row contains an invalid position")
    if aggregate.matrix_unit_ids[index] != row["unit_id"]:
        raise ValueError(
            "aggregate matrix_row does not map to the declared matrix_unit_id"
        )
    return index


def _extract_receiver_response(
    aggregate: PseudobulkDataset,
    *,
    receiver: str,
    sample_ids: tuple[str, ...],
    sample_subject_ids: tuple[str, ...],
    sample_context_ids: tuple[str, ...],
    context_keys: tuple[str, ...],
) -> _ExtractedResponse:
    metadata = aggregate.unit_metadata.copy(deep=True)
    required = {
        "unit_id",
        "sample_id",
        "subject_id",
        "cell_type",
        "context",
        "matrix_row",
        "state_eligible",
    }
    missing_columns = required.difference(metadata.columns)
    if missing_columns:
        raise ValueError(
            "aggregate unit_metadata is missing response fields: "
            f"{sorted(missing_columns)}"
        )
    expected = {
        sample_id: (subject_id, context_id)
        for sample_id, subject_id, context_id in zip(
            sample_ids, sample_subject_ids, sample_context_ids, strict=True
        )
    }
    relevant = metadata.loc[metadata["sample_id"].isin(expected)].copy()
    for _, row in relevant.iterrows():
        sample_id = row["sample_id"]
        subject_id = row["subject_id"]
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(subject_id, str)
            or not subject_id
        ):
            raise ValueError(
                "aggregate sample_id and subject_id must be non-empty strings"
            )
        context_id = _context_identifier(row["context"], context_keys)
        if (subject_id, context_id) != expected[sample_id]:
            raise ValueError(
                f"aggregate sample {sample_id!r} conflicts with its frozen "
                "subject/context mapping"
            )

    selected = relevant.loc[
        relevant["cell_type"].map(lambda value: value == receiver)
    ].copy()
    if selected["unit_id"].duplicated().any():
        raise ValueError("receiver aggregate unit_id values must be unique")
    selected["_unit_order"] = selected["unit_id"].map(str)
    selected = selected.sort_values("_unit_order", kind="stable")

    rows_by_sample: dict[str, list[pd.Series]] = {
        sample_id: [] for sample_id in sample_ids
    }
    for _, row in selected.iterrows():
        sample_id = row["sample_id"]
        subject_id = row["subject_id"]
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(subject_id, str)
            or not subject_id
        ):
            raise ValueError(
                "receiver aggregate sample_id and subject_id must be non-empty strings"
            )
        rows_by_sample[sample_id].append(row)

    observed_samples: list[str] = []
    observed_subjects: list[str] = []
    observed_contexts: list[str] = []
    response_rows: list[np.ndarray] = []
    missing_samples: list[str] = []
    for sample_id in sample_ids:
        technical_values: list[np.ndarray] = []
        for row in rows_by_sample[sample_id]:
            if not bool(row["state_eligible"]):
                continue
            matrix_row = _validate_matrix_row(aggregate, row)
            counts = (
                aggregate.counts.getrow(matrix_row).toarray().ravel().astype(np.float64)
            )
            if np.any(~np.isfinite(counts)) or np.any(counts < 0):
                raise ValueError(
                    "receiver aggregate counts must be finite and non-negative"
                )
            library_size = float(np.sum(counts))
            if library_size <= 0:
                continue
            technical_values.append(np.log1p(counts / library_size * _CPM_SCALE))
        if not technical_values:
            missing_samples.append(sample_id)
            continue
        subject_id, context_id = expected[sample_id]
        observed_samples.append(sample_id)
        observed_subjects.append(subject_id)
        observed_contexts.append(context_id)
        # Technical receiver units are averaged before any subject-level model fit.
        response_rows.append(np.vstack(technical_values).mean(axis=0))
    values = (
        np.vstack(response_rows)
        if response_rows
        else np.empty((0, len(aggregate.feature_ids)), dtype=np.float64)
    )
    return _ExtractedResponse(
        sample_ids=tuple(observed_samples),
        sample_subject_ids=tuple(observed_subjects),
        sample_context_ids=tuple(observed_contexts),
        values=values,
        missing_sample_ids=tuple(missing_samples),
    )


def _vector_estimable(
    matrix: np.ndarray, vector: np.ndarray, *, tolerance: float = 1e-8
) -> bool:
    if matrix.size == 0 or np.linalg.norm(vector) <= tolerance:
        return False
    projection = np.linalg.pinv(matrix) @ matrix
    residual = vector - vector @ projection
    return bool(
        np.linalg.norm(residual) <= tolerance * max(1.0, float(np.linalg.norm(vector)))
    )


@dataclass(frozen=True, slots=True)
class _FitResult:
    effect: np.ndarray
    standard_error: np.ndarray
    raw_precision: np.ndarray
    status: str
    reason_code: str | None
    method: str
    n_model_samples: int
    n_model_subjects: int
    residual_df: int | None


def _not_estimable_fit(
    n_features: int,
    reason_code: str,
    *,
    n_model_samples: int,
    n_model_subjects: int,
) -> _FitResult:
    missing: np.ndarray = np.full(n_features, np.nan, dtype=np.float64)
    return _FitResult(
        effect=missing,
        standard_error=missing.copy(),
        raw_precision=missing.copy(),
        status="not_estimable",
        reason_code=reason_code,
        method=_NOT_ESTIMABLE_METHOD,
        n_model_samples=n_model_samples,
        n_model_subjects=n_model_subjects,
        residual_df=None,
    )


def _fit_formula_response(
    extracted: _ExtractedResponse,
    encoder: FrozenDesignEncoder,
    *,
    min_subjects_per_context: int,
    n_features: int,
) -> _FitResult:
    weights = _contrast_context_weights(encoder)
    training_rows = pd.DataFrame(
        {
            "sample_id": encoder.training_sample_ids,
            "subject_id": encoder.training_sample_subject_ids,
            "context_id": encoder.training_sample_context_ids,
        }
    )
    referenced_training = training_rows.loc[training_rows["context_id"].isin(weights)]
    subject_sets = [
        set(
            referenced_training.loc[
                referenced_training["context_id"].eq(context_id), "subject_id"
            ]
        )
        for context_id in weights
    ]
    if any(not subjects for subjects in subject_sets):
        return _not_estimable_fit(
            n_features,
            "contrast_context_absent_from_training_design",
            n_model_samples=0,
            n_model_subjects=0,
        )
    pairwise_overlap = any(
        left.intersection(right)
        for index, left in enumerate(subject_sets)
        for right in subject_sets[index + 1 :]
    )
    paired = len(weights) == 2 and subject_sets[0] == subject_sets[1]
    if pairwise_overlap and not paired:
        reason = (
            "mixed_paired_unpaired_design_not_supported"
            if len(weights) == 2
            else "repeated_multi_context_design_not_supported"
        )
        return _not_estimable_fit(
            n_features,
            reason,
            n_model_samples=0,
            n_model_subjects=0,
        )

    rows = pd.DataFrame(
        {
            "sample_id": extracted.sample_ids,
            "subject_id": extracted.sample_subject_ids,
            "context_id": extracted.sample_context_ids,
            "response_row": np.arange(len(extracted.sample_ids), dtype=np.int64),
        }
    )
    rows = rows.loc[rows["context_id"].isin(weights)].copy()
    receiver_subject_sets = [
        set(rows.loc[rows["context_id"].eq(context_id), "subject_id"])
        for context_id in weights
    ]
    if any(not subjects for subjects in receiver_subject_sets):
        return _not_estimable_fit(
            n_features,
            "missing_context_support",
            n_model_samples=len(rows),
            n_model_subjects=int(rows["subject_id"].nunique()),
        )
    if paired:
        complete_subjects = set.intersection(*receiver_subject_sets)
        if len(complete_subjects) < min_subjects_per_context:
            return _not_estimable_fit(
                n_features,
                "insufficient_complete_pair_support",
                n_model_samples=len(rows),
                n_model_subjects=len(complete_subjects),
            )
        rows = rows.loc[rows["subject_id"].isin(complete_subjects)].copy()
    elif any(
        len(subjects) < min_subjects_per_context for subjects in receiver_subject_sets
    ):
        return _not_estimable_fit(
            n_features,
            "insufficient_subject_support",
            n_model_samples=len(rows),
            n_model_subjects=int(rows["subject_id"].nunique()),
        )

    design_by_sample = {
        sample_id: index for index, sample_id in enumerate(encoder.training_sample_ids)
    }
    collapsed_values: list[np.ndarray] = []
    collapsed_design: list[np.ndarray] = []
    collapsed_subjects: list[str] = []
    grouped = rows.groupby(
        ["subject_id", "context_id"], observed=True, sort=True, dropna=False
    )
    for (subject_id, _), group in grouped:
        response_indices = group["response_row"].to_numpy(dtype=np.int64)
        design_indices = np.asarray(
            [design_by_sample[value] for value in group["sample_id"]],
            dtype=np.int64,
        )
        collapsed_values.append(extracted.values[response_indices].mean(axis=0))
        collapsed_design.append(
            np.column_stack(
                (
                    encoder.training_nuisance_matrix[design_indices],
                    encoder.training_context_regressor[design_indices],
                )
            ).mean(axis=0)
        )
        collapsed_subjects.append(str(subject_id))
    model_values = np.vstack(collapsed_values)
    model_design = np.vstack(collapsed_design)
    coefficient_contrast = np.zeros(model_design.shape[1], dtype=np.float64)
    coefficient_contrast[-1] = 1.0
    if paired:
        subjects = tuple(sorted(set(collapsed_subjects)))
        codes = {subject: index for index, subject in enumerate(subjects)}
        subject_columns = np.zeros((len(collapsed_subjects), max(0, len(subjects) - 1)))
        for row_index, subject in enumerate(collapsed_subjects):
            subject_index = codes[subject]
            if subject_index > 0:
                subject_columns[row_index, subject_index - 1] = 1.0
        model_design = np.column_stack((model_design, subject_columns))
        coefficient_contrast = np.concatenate(
            (coefficient_contrast, np.zeros(subject_columns.shape[1]))
        )
    n_model_samples = len(rows)
    n_model_subjects = len(set(collapsed_subjects))
    if not _vector_estimable(model_design, coefficient_contrast):
        return _not_estimable_fit(
            n_features,
            "receiver_formula_contrast_not_estimable",
            n_model_samples=n_model_samples,
            n_model_subjects=n_model_subjects,
        )
    rank = int(np.linalg.matrix_rank(model_design))
    residual_df = len(model_design) - rank
    if residual_df < 1:
        return _not_estimable_fit(
            n_features,
            "insufficient_residual_degrees_of_freedom",
            n_model_samples=n_model_samples,
            n_model_subjects=n_model_subjects,
        )
    coefficients = np.linalg.pinv(model_design) @ model_values
    effect = coefficient_contrast @ coefficients
    residual = model_values - model_design @ coefficients
    residual_variance = np.sum(residual * residual, axis=0) / residual_df
    contrast_variance = float(
        coefficient_contrast
        @ np.linalg.pinv(model_design.T @ model_design)
        @ coefficient_contrast
    )
    standard_error = np.sqrt(np.maximum(0.0, residual_variance * contrast_variance))
    raw_precision: np.ndarray = np.full(n_features, np.nan, dtype=np.float64)
    valid_precision = np.isfinite(standard_error) & (standard_error > 0)
    raw_precision[valid_precision] = 1.0 / standard_error[valid_precision] ** 2
    if np.any(~np.isfinite(effect)) or np.any(~np.isfinite(standard_error)):
        return _not_estimable_fit(
            n_features,
            "non_finite_response_estimate",
            n_model_samples=n_model_samples,
            n_model_subjects=n_model_subjects,
        )
    return _FitResult(
        effect=effect,
        standard_error=standard_error,
        raw_precision=raw_precision,
        status="ok",
        reason_code=None,
        method=_PAIRED_METHOD if paired else _INDEPENDENT_METHOD,
        n_model_samples=n_model_samples,
        n_model_subjects=n_model_subjects,
        residual_df=residual_df,
    )


@dataclass(frozen=True, slots=True, init=False)
class FoldGeneResponseArtifact:
    """Producer-owned receiver response fitted inside one physical fold."""

    receiver: str
    contrast_name: str
    fold_id: str
    training_input_digest: str
    encoder_id: str
    context_regressor_id: str
    nuisance_design_id: str
    training_sample_manifest_digest: str
    training_design_digest: str
    coefficient_contrast_digest: str
    reparameterization_digest: str
    context_keys: tuple[str, ...]
    feature_ids: tuple[str, ...]
    training_sample_ids: tuple[str, ...]
    training_sample_subject_ids: tuple[str, ...]
    training_sample_context_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_context_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    missing_sample_ids: tuple[str, ...]
    sample_values: np.ndarray
    effect: np.ndarray
    standard_error: np.ndarray
    raw_precision: np.ndarray
    response_input_digest: str
    sample_values_digest: str
    effect_digest: str
    standard_error_digest: str
    raw_precision_digest: str
    method: str
    value_scale: str
    min_subjects_per_context: int
    n_model_samples: int
    n_model_subjects: int
    residual_df: int | None
    status: str
    reason_code: str | None
    artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FoldGeneResponseArtifact is producer-owned; use fit_fold_gene_response()"
        )

    @classmethod
    def _from_fit(
        cls,
        *,
        encoder: FrozenDesignEncoder,
        receiver: str,
        fold_id: str,
        training_input_digest: str,
        extracted: _ExtractedResponse,
        fit: _FitResult,
        feature_ids: tuple[str, ...],
        min_subjects_per_context: int,
    ) -> FoldGeneResponseArtifact:
        features = _names(feature_ids, field_name="feature_ids")
        samples = _names(
            extracted.sample_ids, field_name="sample_ids", allow_empty=True
        )
        sample_subjects = _aligned_names(
            extracted.sample_subject_ids,
            length=len(samples),
            field_name="sample_subject_ids",
        )
        sample_contexts = _aligned_names(
            extracted.sample_context_ids,
            length=len(samples),
            field_name="sample_context_ids",
        )
        values = _immutable_array(extracted.values)
        effect = _immutable_array(fit.effect)
        standard_error = _immutable_array(fit.standard_error)
        raw_precision = _immutable_array(fit.raw_precision)
        if values.shape != (len(samples), len(features)):
            raise ValueError(
                "sample response matrix does not align with rows and features"
            )
        for name, array in (
            ("effect", effect),
            ("standard_error", standard_error),
            ("raw_precision", raw_precision),
        ):
            if array.shape != (len(features),):
                raise ValueError(f"{name} must align with feature_ids")
        subjects = tuple(sorted(set(sample_subjects)))
        training_samples = _names(
            encoder.training_sample_ids, field_name="training_sample_ids"
        )
        training_sample_subjects = _aligned_names(
            encoder.training_sample_subject_ids,
            length=len(training_samples),
            field_name="training_sample_subject_ids",
        )
        training_sample_contexts = _aligned_names(
            encoder.training_sample_context_ids,
            length=len(training_samples),
            field_name="training_sample_context_ids",
        )
        training_subjects = tuple(encoder.training_subject_ids)
        missing_samples = tuple(sorted(extracted.missing_sample_ids))
        if set(samples).union(missing_samples) != set(training_samples):
            raise ValueError(
                "observed and missing receiver rows must cover the training samples"
            )
        training_mapping = dict(
            zip(
                training_samples,
                zip(training_sample_subjects, training_sample_contexts, strict=True),
                strict=True,
            )
        )
        if any(
            training_mapping[sample] != (subject, context)
            for sample, subject, context in zip(
                samples, sample_subjects, sample_contexts, strict=True
            )
        ):
            raise ValueError(
                "receiver response rows conflict with training design rows"
            )
        sample_values_digest = _numeric_digest(values)
        effect_digest = _numeric_digest(effect)
        standard_error_digest = _numeric_digest(standard_error)
        raw_precision_digest = _numeric_digest(raw_precision)
        response_input_digest = stable_id(
            "fold_gene_response_input",
            {
                "encoder_id": encoder.encoder_id,
                "context_keys": list(encoder.context_keys),
                "feature_ids": list(features),
                "min_subjects_per_context": min_subjects_per_context,
                "missing_sample_ids": list(missing_samples),
                "receiver": receiver,
                "rows": _row_payload(samples, sample_subjects, sample_contexts),
                "sample_values_digest": sample_values_digest,
                "training_rows": _row_payload(
                    training_samples,
                    training_sample_subjects,
                    training_sample_contexts,
                ),
                "training_input_digest": training_input_digest,
            },
            schema_version="1",
            digest_length=64,
        )
        self = object.__new__(cls)
        attributes: dict[str, Any] = {
            "receiver": receiver,
            "contrast_name": encoder.contrast.name,
            "fold_id": fold_id,
            "training_input_digest": training_input_digest,
            "encoder_id": encoder.encoder_id,
            "context_regressor_id": encoder.context_regressor_id,
            "nuisance_design_id": encoder.nuisance_design_id,
            "training_sample_manifest_digest": encoder.training_sample_manifest_digest,
            "training_design_digest": encoder.training_design_digest,
            "coefficient_contrast_digest": encoder.coefficient_contrast_digest,
            "reparameterization_digest": encoder.reparameterization_digest,
            "context_keys": encoder.context_keys,
            "feature_ids": features,
            "training_sample_ids": training_samples,
            "training_sample_subject_ids": training_sample_subjects,
            "training_sample_context_ids": training_sample_contexts,
            "training_subject_ids": training_subjects,
            "sample_ids": samples,
            "sample_subject_ids": sample_subjects,
            "sample_context_ids": sample_contexts,
            "subject_ids": subjects,
            "missing_sample_ids": missing_samples,
            "sample_values": values,
            "effect": effect,
            "standard_error": standard_error,
            "raw_precision": raw_precision,
            "response_input_digest": response_input_digest,
            "sample_values_digest": sample_values_digest,
            "effect_digest": effect_digest,
            "standard_error_digest": standard_error_digest,
            "raw_precision_digest": raw_precision_digest,
            "method": fit.method,
            "value_scale": "log1p_cpm",
            "min_subjects_per_context": min_subjects_per_context,
            "n_model_samples": fit.n_model_samples,
            "n_model_subjects": fit.n_model_subjects,
            "residual_df": fit.residual_df,
            "status": fit.status,
            "reason_code": fit.reason_code,
            "_producer_marker": _TRAINING_PRODUCER_MARKER,
        }
        for name, value in attributes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "artifact_id",
            stable_id(
                "fold_gene_response",
                self._identity_payload(),
                schema_version=_TRAINING_SCHEMA_VERSION,
            ),
        )
        record_validation(self)
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "coefficient_contrast_digest": self.coefficient_contrast_digest,
            "context_regressor_id": self.context_regressor_id,
            "context_keys": list(self.context_keys),
            "contrast_name": self.contrast_name,
            "effect_digest": self.effect_digest,
            "encoder_id": self.encoder_id,
            "fold_id": self.fold_id,
            "method": self.method,
            "min_subjects_per_context": self.min_subjects_per_context,
            "missing_sample_ids": list(self.missing_sample_ids),
            "n_model_samples": self.n_model_samples,
            "n_model_subjects": self.n_model_subjects,
            "residual_df": self.residual_df,
            "nuisance_design_id": self.nuisance_design_id,
            "raw_precision_digest": self.raw_precision_digest,
            "reason_code": self.reason_code,
            "reparameterization_digest": self.reparameterization_digest,
            "response_input_digest": self.response_input_digest,
            "standard_error_digest": self.standard_error_digest,
            "status": self.status,
            "training_design_digest": self.training_design_digest,
            "training_sample_manifest_digest": self.training_sample_manifest_digest,
            "training_rows": _row_payload(
                self.training_sample_ids,
                self.training_sample_subject_ids,
                self.training_sample_context_ids,
            ),
            "training_subject_ids": list(self.training_subject_ids),
            "value_scale": self.value_scale,
        }

    @validation_scope()
    def _require_intact(self) -> None:
        if validation_is_cached(self):
            return
        try:
            samples = _names(self.sample_ids, field_name="sample_ids", allow_empty=True)
            sample_subjects = _aligned_names(
                self.sample_subject_ids,
                length=len(samples),
                field_name="sample_subject_ids",
            )
            sample_contexts = _aligned_names(
                self.sample_context_ids,
                length=len(samples),
                field_name="sample_context_ids",
            )
            features = _names(self.feature_ids, field_name="feature_ids")
            context_keys = _names(self.context_keys, field_name="context_keys")
            training_samples = _names(
                self.training_sample_ids, field_name="training_sample_ids"
            )
            training_sample_subjects = _aligned_names(
                self.training_sample_subject_ids,
                length=len(training_samples),
                field_name="training_sample_subject_ids",
            )
            training_sample_contexts = _aligned_names(
                self.training_sample_context_ids,
                length=len(training_samples),
                field_name="training_sample_context_ids",
            )
            training_subjects = tuple(
                sorted(
                    _names(
                        self.training_subject_ids,
                        field_name="training_subject_ids",
                    )
                )
            )
            missing_samples = tuple(
                sorted(
                    _names(
                        self.missing_sample_ids,
                        field_name="missing_sample_ids",
                        allow_empty=True,
                    )
                )
            )
            lineage = (
                self.training_input_digest,
                self.encoder_id,
                self.context_regressor_id,
                self.nuisance_design_id,
                self.training_sample_manifest_digest,
                self.training_design_digest,
                self.coefficient_contrast_digest,
                self.reparameterization_digest,
            )
            if self._producer_marker != _TRAINING_PRODUCER_MARKER or any(
                not isinstance(value, str) or not value for value in lineage
            ):
                raise ValueError("response artifact producer lineage is invalid")
            arrays = (
                self.sample_values,
                self.effect,
                self.standard_error,
                self.raw_precision,
            )
            if any(not _is_immutable_byte_backed(array) for array in arrays):
                raise ValueError("response artifact arrays are not immutable")
            if self.sample_values.shape != (len(samples), len(features)) or any(
                array.shape != (len(features),) for array in arrays[1:]
            ):
                raise ValueError("response artifact arrays are misaligned")
            if np.any(~np.isfinite(self.sample_values)):
                raise ValueError("training sample responses must be finite")
            if self.subject_ids != tuple(sorted(set(sample_subjects))):
                raise ValueError("response subject identity is inconsistent")
            if training_subjects != tuple(sorted(set(training_sample_subjects))):
                raise ValueError("training response subject manifest is inconsistent")
            if set(samples).intersection(missing_samples):
                raise ValueError("observed and missing response samples overlap")
            if set(samples).union(missing_samples) != set(training_samples):
                raise ValueError("response rows do not cover the training samples")
            training_mapping = dict(
                zip(
                    training_samples,
                    zip(
                        training_sample_subjects,
                        training_sample_contexts,
                        strict=True,
                    ),
                    strict=True,
                )
            )
            if any(
                training_mapping[sample] != (subject, context)
                for sample, subject, context in zip(
                    samples, sample_subjects, sample_contexts, strict=True
                )
            ):
                raise ValueError("response rows conflict with the training manifest")
            if samples != tuple(sorted(samples, key=_canonical_name_key)):
                raise ValueError("response sample rows are not canonical")
            if self.status not in {"ok", "not_estimable"} or (
                (self.status == "ok") == (self.reason_code is not None)
            ):
                raise ValueError("response estimability fields are inconsistent")
            if (
                isinstance(self.min_subjects_per_context, bool)
                or not isinstance(self.min_subjects_per_context, int)
                or self.min_subjects_per_context < 2
            ):
                raise ValueError("response minimum subject support is invalid")
            if self.status == "ok":
                if self.method not in {_INDEPENDENT_METHOD, _PAIRED_METHOD}:
                    raise ValueError("observed response method is invalid")
                if np.any(~np.isfinite(self.effect)) or np.any(
                    ~np.isfinite(self.standard_error)
                ):
                    raise ValueError("observed response estimates must be finite")
                expected_precision = np.full(len(features), np.nan)
                valid = self.standard_error > 0
                expected_precision[valid] = 1.0 / self.standard_error[valid] ** 2
                if not np.array_equal(
                    self.raw_precision, expected_precision, equal_nan=True
                ):
                    raise ValueError("raw response precision is inconsistent")
                if (
                    isinstance(self.residual_df, bool)
                    or not isinstance(self.residual_df, int)
                    or self.residual_df < 1
                    or self.residual_df >= self.n_model_samples
                ):
                    raise ValueError("observed response residual df is invalid")
            elif (
                self.method != _NOT_ESTIMABLE_METHOD
                or self.residual_df is not None
                or not all(np.isnan(array).all() for array in arrays[1:])
            ):
                raise ValueError("not-estimable response estimates must be all NaN")
            sample_values_digest = _numeric_digest(self.sample_values)
            effect_digest = _numeric_digest(self.effect)
            standard_error_digest = _numeric_digest(self.standard_error)
            raw_precision_digest = _numeric_digest(self.raw_precision)
            response_input_digest = stable_id(
                "fold_gene_response_input",
                {
                    "encoder_id": self.encoder_id,
                    "context_keys": list(context_keys),
                    "feature_ids": list(features),
                    "min_subjects_per_context": self.min_subjects_per_context,
                    "missing_sample_ids": list(missing_samples),
                    "receiver": self.receiver,
                    "rows": _row_payload(samples, sample_subjects, sample_contexts),
                    "sample_values_digest": sample_values_digest,
                    "training_rows": _row_payload(
                        training_samples,
                        training_sample_subjects,
                        training_sample_contexts,
                    ),
                    "training_input_digest": self.training_input_digest,
                },
                schema_version="1",
                digest_length=64,
            )
            expected_id = stable_id(
                "fold_gene_response",
                self._identity_payload(),
                schema_version=_TRAINING_SCHEMA_VERSION,
            )
            valid_identity = (
                sample_values_digest == self.sample_values_digest
                and effect_digest == self.effect_digest
                and standard_error_digest == self.standard_error_digest
                and raw_precision_digest == self.raw_precision_digest
                and response_input_digest == self.response_input_digest
                and expected_id == self.artifact_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Fold gene response failed integrity validation",
                code="fold_gene_response_integrity_violation",
                field="artifact_id",
                remediation="Refit the receiver response inside its training fold",
            ) from error
        if not valid_identity:
            raise ContractError(
                "Fold gene response failed integrity validation",
                code="fold_gene_response_integrity_violation",
                field="artifact_id",
                remediation="Refit the receiver response inside its training fold",
            )
        record_validation(self)

    def require_compatible(self, encoder: FrozenDesignEncoder) -> None:
        """Validate exact frozen-design parentage before downstream use."""

        if not isinstance(encoder, FrozenDesignEncoder):
            raise TypeError("encoder must be a FrozenDesignEncoder")
        encoder.to_dict()
        self._require_intact()
        expected = (
            encoder.encoder_id,
            encoder.context_regressor_id,
            encoder.nuisance_design_id,
            encoder.training_sample_manifest_digest,
            encoder.training_design_digest,
            encoder.coefficient_contrast_digest,
            encoder.reparameterization_digest,
            encoder.contrast.name,
            encoder.context_keys,
            encoder.training_sample_ids,
            encoder.training_sample_subject_ids,
            encoder.training_sample_context_ids,
            encoder.training_subject_ids,
        )
        observed = (
            self.encoder_id,
            self.context_regressor_id,
            self.nuisance_design_id,
            self.training_sample_manifest_digest,
            self.training_design_digest,
            self.coefficient_contrast_digest,
            self.reparameterization_digest,
            self.contrast_name,
            self.context_keys,
            self.training_sample_ids,
            self.training_sample_subject_ids,
            self.training_sample_context_ids,
            self.training_subject_ids,
        )
        if observed != expected:
            raise ContractError(
                "Fold gene response does not match its frozen design encoder",
                code="fold_gene_response_scope_mismatch",
                field="artifact_id",
                remediation="Use the response fitted from this encoder",
            )

    def to_dict(self) -> dict[str, object]:
        """Return response provenance without expanding numeric arrays."""

        self._require_intact()
        return {
            "artifact_id": self.artifact_id,
            "receiver": self.receiver,
            "contrast_name": self.contrast_name,
            "fold_id": self.fold_id,
            "training_input_digest": self.training_input_digest,
            "encoder_id": self.encoder_id,
            "context_regressor_id": self.context_regressor_id,
            "nuisance_design_id": self.nuisance_design_id,
            "training_sample_manifest_digest": self.training_sample_manifest_digest,
            "training_design_digest": self.training_design_digest,
            "coefficient_contrast_digest": self.coefficient_contrast_digest,
            "reparameterization_digest": self.reparameterization_digest,
            "context_keys": list(self.context_keys),
            "feature_ids": list(self.feature_ids),
            "training_sample_ids": list(self.training_sample_ids),
            "training_sample_subject_ids": list(self.training_sample_subject_ids),
            "training_sample_context_ids": list(self.training_sample_context_ids),
            "training_subject_ids": list(self.training_subject_ids),
            "sample_ids": list(self.sample_ids),
            "sample_subject_ids": list(self.sample_subject_ids),
            "sample_context_ids": list(self.sample_context_ids),
            "subject_ids": list(self.subject_ids),
            "missing_sample_ids": list(self.missing_sample_ids),
            "response_input_digest": self.response_input_digest,
            "sample_values_digest": self.sample_values_digest,
            "effect_digest": self.effect_digest,
            "standard_error_digest": self.standard_error_digest,
            "raw_precision_digest": self.raw_precision_digest,
            "method": self.method,
            "value_scale": self.value_scale,
            "min_subjects_per_context": self.min_subjects_per_context,
            "n_model_samples": self.n_model_samples,
            "n_model_subjects": self.n_model_subjects,
            "residual_df": self.residual_df,
            "status": self.status,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True, init=False)
class FoldGeneResponseApplication:
    """Held-out receiver rows applied without response-model refitting."""

    training_response_id: str
    design_application_id: str
    encoder_id: str
    context_regressor_id: str
    nuisance_design_id: str
    receiver: str
    contrast_name: str
    fold_id: str
    feature_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_context_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    missing_sample_ids: tuple[str, ...]
    sample_values: np.ndarray
    sample_values_digest: str
    response_input_digest: str
    status: str
    reason_code: str | None
    value_scale: str
    application_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FoldGeneResponseApplication is producer-owned; "
            "use apply_fold_gene_response()"
        )

    @classmethod
    def _from_application(
        cls,
        *,
        training_response: FoldGeneResponseArtifact,
        design_application: FrozenDesignApplication,
        sample_values: np.ndarray,
        missing_sample_ids: tuple[str, ...],
        status: str,
        reason_code: str | None,
    ) -> FoldGeneResponseApplication:
        values = _immutable_array(sample_values)
        samples = tuple(design_application.sample_ids)
        sample_subjects = tuple(design_application.sample_subject_ids)
        sample_contexts = tuple(design_application.sample_context_ids)
        if values.shape != (len(samples), len(training_response.feature_ids)):
            raise ValueError("held-out response values do not align with frozen design")
        missing = tuple(
            sorted(
                _names(
                    missing_sample_ids,
                    field_name="missing_sample_ids",
                    allow_empty=True,
                )
            )
        )
        if not set(missing).issubset(samples):
            raise ValueError("missing_sample_ids must belong to the held-out design")
        values_digest = _numeric_digest(values)
        response_input_digest = stable_id(
            "fold_gene_response_application_input",
            {
                "design_application_id": design_application.application_id,
                "feature_ids": list(training_response.feature_ids),
                "missing_sample_ids": list(missing),
                "rows": _row_payload(samples, sample_subjects, sample_contexts),
                "sample_values_digest": values_digest,
                "training_response_id": training_response.artifact_id,
            },
            schema_version="1",
            digest_length=64,
        )
        self = object.__new__(cls)
        attributes: dict[str, Any] = {
            "training_response_id": training_response.artifact_id,
            "design_application_id": design_application.application_id,
            "encoder_id": training_response.encoder_id,
            "context_regressor_id": training_response.context_regressor_id,
            "nuisance_design_id": training_response.nuisance_design_id,
            "receiver": training_response.receiver,
            "contrast_name": training_response.contrast_name,
            "fold_id": training_response.fold_id,
            "feature_ids": training_response.feature_ids,
            "sample_ids": samples,
            "sample_subject_ids": sample_subjects,
            "sample_context_ids": sample_contexts,
            "subject_ids": tuple(sorted(set(sample_subjects))),
            "missing_sample_ids": missing,
            "sample_values": values,
            "sample_values_digest": values_digest,
            "response_input_digest": response_input_digest,
            "status": status,
            "reason_code": reason_code,
            "value_scale": training_response.value_scale,
            "_producer_marker": _APPLICATION_PRODUCER_MARKER,
        }
        for name, value in attributes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "application_id",
            stable_id(
                "fold_gene_response_application",
                self._identity_payload(),
                schema_version="1",
            ),
        )
        record_validation(self)
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "context_regressor_id": self.context_regressor_id,
            "contrast_name": self.contrast_name,
            "design_application_id": self.design_application_id,
            "encoder_id": self.encoder_id,
            "fold_id": self.fold_id,
            "nuisance_design_id": self.nuisance_design_id,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "response_input_digest": self.response_input_digest,
            "status": self.status,
            "training_response_id": self.training_response_id,
            "value_scale": self.value_scale,
        }

    @validation_scope()
    def _require_intact(self) -> None:
        if validation_is_cached(self):
            return
        try:
            samples = _names(self.sample_ids, field_name="sample_ids")
            sample_subjects = _aligned_names(
                self.sample_subject_ids,
                length=len(samples),
                field_name="sample_subject_ids",
            )
            sample_contexts = _aligned_names(
                self.sample_context_ids,
                length=len(samples),
                field_name="sample_context_ids",
            )
            features = _names(self.feature_ids, field_name="feature_ids")
            missing = _names(
                self.missing_sample_ids,
                field_name="missing_sample_ids",
                allow_empty=True,
            )
            if self._producer_marker != _APPLICATION_PRODUCER_MARKER:
                raise ValueError("response application is not producer-owned")
            if self.sample_values.shape != (len(samples), len(features)) or not (
                _is_immutable_byte_backed(self.sample_values)
            ):
                raise ValueError("held-out response matrix is invalid")
            if samples != tuple(
                sorted(samples, key=_canonical_name_key)
            ) or self.subject_ids != tuple(sorted(set(sample_subjects))):
                raise ValueError("held-out response row identity is inconsistent")
            if not set(missing).issubset(samples):
                raise ValueError("held-out missing samples are invalid")
            missing_positions = [samples.index(sample_id) for sample_id in missing]
            observed_positions = [
                index for index in range(len(samples)) if index not in missing_positions
            ]
            if (
                missing_positions
                and not np.isnan(self.sample_values[missing_positions]).all()
            ):
                raise ValueError("missing receiver samples must remain NaN")
            if observed_positions and np.any(
                ~np.isfinite(self.sample_values[observed_positions])
            ):
                raise ValueError("observed receiver samples must be finite")
            if self.status not in {"ok", "not_estimable"} or (
                (self.status == "ok") == (self.reason_code is not None)
            ):
                raise ValueError("response application status is inconsistent")
            if self.status == "ok" and missing:
                raise ValueError("observed response application cannot omit samples")
            values_digest = _numeric_digest(self.sample_values)
            response_input_digest = stable_id(
                "fold_gene_response_application_input",
                {
                    "design_application_id": self.design_application_id,
                    "feature_ids": list(features),
                    "missing_sample_ids": list(missing),
                    "rows": _row_payload(samples, sample_subjects, sample_contexts),
                    "sample_values_digest": values_digest,
                    "training_response_id": self.training_response_id,
                },
                schema_version="1",
                digest_length=64,
            )
            expected_id = stable_id(
                "fold_gene_response_application",
                self._identity_payload(),
                schema_version="1",
            )
            valid = (
                values_digest == self.sample_values_digest
                and response_input_digest == self.response_input_digest
                and expected_id == self.application_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Fold gene response application failed integrity validation",
                code="fold_gene_response_application_integrity_violation",
                field="application_id",
                remediation="Reapply the training response to held-out data",
            ) from error
        if not valid:
            raise ContractError(
                "Fold gene response application failed integrity validation",
                code="fold_gene_response_application_integrity_violation",
                field="application_id",
                remediation="Reapply the training response to held-out data",
            )
        record_validation(self)

    def require_compatible(
        self,
        training_response: FoldGeneResponseArtifact,
        design_application: FrozenDesignApplication,
    ) -> None:
        """Validate integrity and both exact producer parents."""

        if not isinstance(training_response, FoldGeneResponseArtifact):
            raise TypeError("training_response must be a FoldGeneResponseArtifact")
        if not isinstance(design_application, FrozenDesignApplication):
            raise TypeError("design_application must be a FrozenDesignApplication")
        training_response._require_intact()
        design_application.to_dict()
        self._require_intact()
        expected = (
            training_response.artifact_id,
            design_application.application_id,
            training_response.encoder_id,
            training_response.context_regressor_id,
            training_response.nuisance_design_id,
            training_response.feature_ids,
        )
        observed = (
            self.training_response_id,
            self.design_application_id,
            self.encoder_id,
            self.context_regressor_id,
            self.nuisance_design_id,
            self.feature_ids,
        )
        if observed != expected:
            raise ContractError(
                "Fold gene response application does not match its parents",
                code="fold_gene_response_application_scope_mismatch",
                field="application_id",
                remediation="Use the application produced from these exact parents",
            )

    def to_dict(self) -> dict[str, object]:
        """Return held-out response provenance without expanding its matrix."""

        self._require_intact()
        return {
            "application_id": self.application_id,
            "training_response_id": self.training_response_id,
            "design_application_id": self.design_application_id,
            "encoder_id": self.encoder_id,
            "context_regressor_id": self.context_regressor_id,
            "nuisance_design_id": self.nuisance_design_id,
            "receiver": self.receiver,
            "contrast_name": self.contrast_name,
            "fold_id": self.fold_id,
            "feature_ids": list(self.feature_ids),
            "sample_ids": list(self.sample_ids),
            "sample_subject_ids": list(self.sample_subject_ids),
            "sample_context_ids": list(self.sample_context_ids),
            "subject_ids": list(self.subject_ids),
            "missing_sample_ids": list(self.missing_sample_ids),
            "sample_values_digest": self.sample_values_digest,
            "response_input_digest": self.response_input_digest,
            "status": self.status,
            "reason_code": self.reason_code,
            "value_scale": self.value_scale,
        }


def fit_fold_gene_response(
    aggregate: PseudobulkDataset,
    encoder: FrozenDesignEncoder,
    *,
    receiver: str,
    fold_id: str,
    training_input_digest: str,
    min_subjects_per_context: int = 2,
) -> FoldGeneResponseArtifact:
    """Fit a sample-level receiver response using only encoder training rows."""

    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("aggregate must be a PseudobulkDataset")
    if not isinstance(encoder, FrozenDesignEncoder):
        raise TypeError("encoder must be a FrozenDesignEncoder")
    encoder.to_dict()
    receiver_name = _scope_name(receiver, field_name="receiver")
    fold = _scope_name(fold_id, field_name="fold_id")
    input_digest = _scope_name(
        training_input_digest, field_name="training_input_digest"
    )
    if (
        not isinstance(min_subjects_per_context, int)
        or isinstance(min_subjects_per_context, bool)
        or min_subjects_per_context < 2
    ):
        raise ValueError("min_subjects_per_context must be an integer >= 2")
    feature_ids = _names(aggregate.feature_ids, field_name="feature_ids")
    samples = _names(encoder.training_sample_ids, field_name="training_sample_ids")
    subjects = _aligned_names(
        encoder.training_sample_subject_ids,
        length=len(samples),
        field_name="training_sample_subject_ids",
    )
    contexts = _aligned_names(
        encoder.training_sample_context_ids,
        length=len(samples),
        field_name="training_sample_context_ids",
    )
    extracted = _extract_receiver_response(
        aggregate,
        receiver=receiver_name,
        sample_ids=samples,
        sample_subject_ids=subjects,
        sample_context_ids=contexts,
        context_keys=encoder.context_keys,
    )
    fit = _fit_formula_response(
        extracted,
        encoder,
        min_subjects_per_context=min_subjects_per_context,
        n_features=len(feature_ids),
    )
    return FoldGeneResponseArtifact._from_fit(
        encoder=encoder,
        receiver=receiver_name,
        fold_id=fold,
        training_input_digest=input_digest,
        extracted=extracted,
        fit=fit,
        feature_ids=feature_ids,
        min_subjects_per_context=min_subjects_per_context,
    )


def apply_fold_gene_response(
    aggregate: PseudobulkDataset,
    training_response: FoldGeneResponseArtifact,
    design_application: FrozenDesignApplication,
) -> FoldGeneResponseApplication:
    """Extract held-out receiver rows without refitting response statistics."""

    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("aggregate must be a PseudobulkDataset")
    if not isinstance(training_response, FoldGeneResponseArtifact):
        raise TypeError("training_response must be a FoldGeneResponseArtifact")
    if not isinstance(design_application, FrozenDesignApplication):
        raise TypeError("design_application must be a FrozenDesignApplication")
    training_response._require_intact()
    design_application.to_dict()
    design_lineage = (
        design_application.encoder_id,
        design_application.context_regressor_id,
        design_application.nuisance_design_id,
    )
    response_lineage = (
        training_response.encoder_id,
        training_response.context_regressor_id,
        training_response.nuisance_design_id,
    )
    if design_lineage != response_lineage:
        raise ContractError(
            "Held-out design does not match the training response encoder",
            code="fold_gene_response_application_scope_mismatch",
            field="design_application_id",
            remediation="Apply the design encoder that produced the training response",
        )
    if tuple(aggregate.feature_ids) != training_response.feature_ids:
        raise ContractError(
            "Held-out response feature order differs from training",
            code="fold_gene_response_application_feature_mismatch",
            field="feature_ids",
            remediation="Use the exact ordered training feature universe",
        )
    overlap = set(design_application.sample_subject_ids).intersection(
        training_response.training_subject_ids
    )
    if overlap:
        raise ValueError(
            "held-out response overlaps training subjects: "
            + ", ".join(sorted(overlap))
        )
    extracted = _extract_receiver_response(
        aggregate,
        receiver=training_response.receiver,
        sample_ids=tuple(design_application.sample_ids),
        sample_subject_ids=tuple(design_application.sample_subject_ids),
        sample_context_ids=tuple(design_application.sample_context_ids),
        context_keys=training_response.context_keys,
    )
    observed_by_sample = {
        sample_id: extracted.values[index]
        for index, sample_id in enumerate(extracted.sample_ids)
    }
    aligned: np.ndarray = np.full(
        (len(design_application.sample_ids), len(training_response.feature_ids)),
        np.nan,
        dtype=np.float64,
    )
    for index, sample_id in enumerate(design_application.sample_ids):
        if sample_id in observed_by_sample:
            aligned[index] = observed_by_sample[sample_id]
    if design_application.status != "observed":
        status = "not_estimable"
        reason = "design_application_not_estimable:" + str(
            design_application.reason_code
        )
    elif training_response.status != "ok":
        status = "not_estimable"
        reason = "training_response_not_estimable:" + str(training_response.reason_code)
    elif extracted.missing_sample_ids:
        status = "not_estimable"
        reason = "incomplete_heldout_receiver_response"
    else:
        status = "ok"
        reason = None
    return FoldGeneResponseApplication._from_application(
        training_response=training_response,
        design_application=design_application,
        sample_values=aligned,
        missing_sample_ids=extracted.missing_sample_ids,
        status=status,
        reason_code=reason,
    )


__all__ = [
    "FoldGeneResponseApplication",
    "FoldGeneResponseArtifact",
    "apply_fold_gene_response",
    "fit_fold_gene_response",
]
