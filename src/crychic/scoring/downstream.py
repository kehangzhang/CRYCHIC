"""Fold-frozen receiver-program scoring for held-out samples."""

from __future__ import annotations

import math
from dataclasses import InitVar, dataclass, field
from typing import Any, cast

import numpy as np
from scipy import sparse

from crychic.attribution import SolverStatus, solve_nonnegative_elastic_net
from crychic.core import ContractError, stable_id

from .contracts import float64_array_digest

_MAD_GAUSSIAN_CONSISTENCY = 1.4826
_INCREMENTAL_METHOD = "sample_keyed_incremental_downstream_v2"
_INCREMENTAL_PRODUCER_MARKER = "crychic.scoring.incremental_downstream.v2"
_FAMILY_GAIN_ESTIMAND = "joint_fit_family_only_prediction_v1"


def _names(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = tuple(value.strip() for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    return normalized


def _readonly_vector(
    values: np.ndarray, *, length: int, field_name: str, positive: bool = False
) -> np.ndarray:
    result: np.ndarray = np.asarray(values, dtype=np.float64).copy()
    if result.shape != (length,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{field_name} must be a finite vector of length {length}")
    if positive and np.any(result <= 0):
        raise ValueError(f"{field_name} must be strictly positive")
    result.setflags(write=False)
    return result


def _matrix_digest(matrix: sparse.csc_matrix) -> str:
    result: str = stable_id(
        "sparse_matrix",
        {
            "data_digest": float64_array_digest(matrix.data),
            "indices": matrix.indices.astype(int).tolist(),
            "indptr": matrix.indptr.astype(int).tolist(),
            "shape": list(matrix.shape),
        },
    )
    return result


def _canonical_reference_input(
    expression: np.ndarray,
    *,
    sample_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    feature_ids: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], str]:
    """Canonicalize reference rows and bind their full producer input."""

    samples = _names(tuple(sample_ids), field_name="reference_sample_ids")
    if len(subject_ids) != len(samples):
        raise ValueError("reference subject IDs must align with sample IDs")
    aligned_subjects = tuple(
        value.strip()
        for value in subject_ids
        if isinstance(value, str) and value.strip()
    )
    if len(aligned_subjects) != len(samples):
        raise ValueError("reference subject IDs must contain non-empty strings")
    values = np.asarray(expression, dtype=np.float64)
    expected_shape = (len(samples), len(feature_ids))
    if values.shape != expected_shape or np.any(~np.isfinite(values)):
        raise ValueError(
            "reference expression must align with sample and feature identifiers"
        )
    order = np.asarray(sorted(range(len(samples)), key=samples.__getitem__), dtype=int)
    canonical_samples = tuple(samples[index] for index in order)
    canonical_subjects = tuple(aligned_subjects[index] for index in order)
    canonical_expression = np.asarray(values[order], dtype=np.float64, order="C")
    input_digest = stable_id(
        "downstream_reference_input",
        {
            "expression_digest": float64_array_digest(canonical_expression),
            "feature_ids": list(feature_ids),
            "rows": [
                {"sample_id": sample_id, "subject_id": subject_id}
                for sample_id, subject_id in zip(
                    canonical_samples, canonical_subjects, strict=True
                )
            ],
        },
    )
    return (
        canonical_expression,
        canonical_samples,
        canonical_subjects,
        input_digest,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DownstreamFunctional:
    """Training-fold transform applied unchanged to every compared context."""

    receiver: str
    contrast_name: str
    fold_id: str
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    reference_sample_ids: tuple[str, ...]
    feature_center: np.ndarray
    feature_scale: np.ndarray
    target_weight_matrix: sparse.csc_matrix
    family_support: np.ndarray
    minimum_scale: float
    reference_expression: InitVar[np.ndarray]
    reference_subject_ids: InitVar[tuple[str, ...]]
    center_method: str = "reference_context_feature_median_v1"
    scale_method: str = "reference_context_scaled_mad_floor_v1"
    program_transform: str = "positive_z_saturating_v1"
    reference_input_digest: str = field(init=False)
    downstream_functional_id: str = field(init=False)

    def __post_init__(
        self,
        reference_expression: np.ndarray,
        reference_subject_ids: tuple[str, ...],
    ) -> None:
        for field_name in (
            "receiver",
            "contrast_name",
            "fold_id",
            "center_method",
            "scale_method",
            "program_transform",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        features = _names(tuple(self.feature_ids), field_name="feature_ids")
        families = _names(tuple(self.family_ids), field_name="family_ids")
        subjects = tuple(
            sorted(
                _names(
                    tuple(self.training_subject_ids), field_name="training_subject_ids"
                )
            )
        )
        reference, samples, reference_subjects, reference_input_digest = (
            _canonical_reference_input(
                reference_expression,
                sample_ids=tuple(self.reference_sample_ids),
                subject_ids=reference_subject_ids,
                feature_ids=features,
            )
        )
        if set(reference_subjects).difference(subjects):
            raise ValueError("reference subjects must belong to training_subject_ids")
        if not math.isfinite(self.minimum_scale) or self.minimum_scale <= 0:
            raise ValueError("minimum_scale must be finite and positive")
        center = _readonly_vector(
            self.feature_center,
            length=len(features),
            field_name="feature_center",
        )
        scale = _readonly_vector(
            self.feature_scale,
            length=len(features),
            field_name="feature_scale",
            positive=True,
        )
        if np.any(scale < self.minimum_scale - 1e-12):
            raise ValueError("feature_scale must respect minimum_scale")
        expected_center = np.median(reference, axis=0)
        expected_mad = np.median(np.abs(reference - expected_center), axis=0)
        expected_scale = np.maximum(
            _MAD_GAUSSIAN_CONSISTENCY * expected_mad, self.minimum_scale
        )
        if not np.array_equal(center, expected_center) or not np.array_equal(
            scale, expected_scale
        ):
            raise ValueError(
                "feature center and scale must derive from reference expression"
            )
        matrix = sparse.csc_matrix(self.target_weight_matrix, dtype=np.float64).copy()
        if matrix.shape != (len(features), len(families)):
            raise ValueError(
                "target_weight_matrix shape must equal features x families"
            )
        if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0):
            raise ValueError("target weights must be finite and non-negative")
        column_sums = np.asarray(matrix.sum(axis=0)).ravel()
        if np.any(~np.isclose(column_sums, 1.0, atol=1e-12, rtol=1e-12)):
            raise ValueError("every target-weight column must sum to one")
        matrix.sort_indices()
        support = _readonly_vector(
            self.family_support,
            length=len(families),
            field_name="family_support",
        )
        if np.any((support < 0) | (support > 1)):
            raise ValueError("family_support must lie in [0, 1]")
        matrix.data.setflags(write=False)
        matrix.indices.setflags(write=False)
        matrix.indptr.setflags(write=False)
        payload = {
            "center_digest": float64_array_digest(center),
            "center_method": self.center_method,
            "contrast_name": self.contrast_name,
            "family_ids": list(families),
            "family_support_digest": float64_array_digest(support),
            "feature_ids": list(features),
            "fold_id": self.fold_id,
            "minimum_scale": self.minimum_scale,
            "program_transform": self.program_transform,
            "receiver": self.receiver,
            "reference_input_digest": reference_input_digest,
            "reference_sample_ids": list(samples),
            "scale_digest": float64_array_digest(scale),
            "scale_method": self.scale_method,
            "target_weight_matrix_id": _matrix_digest(matrix),
            "training_subject_ids": list(subjects),
        }
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "family_ids", families)
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "reference_sample_ids", samples)
        object.__setattr__(self, "reference_input_digest", reference_input_digest)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "target_weight_matrix", matrix)
        object.__setattr__(self, "family_support", support)
        object.__setattr__(
            self,
            "downstream_functional_id",
            stable_id("downstream_functional", payload),
        )

    def to_dict(self) -> dict[str, object]:
        """Return provenance without expanding learned numeric arrays."""

        return {
            "downstream_functional_id": self.downstream_functional_id,
            "receiver": self.receiver,
            "contrast_name": self.contrast_name,
            "fold_id": self.fold_id,
            "feature_ids": list(self.feature_ids),
            "family_ids": list(self.family_ids),
            "training_subject_ids": list(self.training_subject_ids),
            "reference_sample_ids": list(self.reference_sample_ids),
            "reference_input_digest": self.reference_input_digest,
            "feature_center_digest": float64_array_digest(self.feature_center),
            "feature_scale_digest": float64_array_digest(self.feature_scale),
            "target_weight_matrix_id": _matrix_digest(self.target_weight_matrix),
            "family_support_digest": float64_array_digest(self.family_support),
            "minimum_scale": self.minimum_scale,
            "center_method": self.center_method,
            "scale_method": self.scale_method,
            "program_transform": self.program_transform,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class DownstreamApplication:
    """Held-out receiver-program values produced by one frozen functional."""

    downstream_functional_id: str
    raw_program: np.ndarray
    receiver_program_score: np.ndarray
    supported_program_score: np.ndarray

    def __post_init__(self) -> None:
        if not self.downstream_functional_id:
            raise ValueError("downstream_functional_id must not be empty")
        arrays = (
            np.asarray(self.raw_program, dtype=np.float64).copy(),
            np.asarray(self.receiver_program_score, dtype=np.float64).copy(),
            np.asarray(self.supported_program_score, dtype=np.float64).copy(),
        )
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("downstream application arrays must be two-dimensional")
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("downstream application arrays must share one shape")
        finite_or_nan = all(
            not np.any(np.isinf(array)) and not np.any(array < 0) for array in arrays
        )
        if not finite_or_nan:
            raise ValueError("downstream application values must be non-negative")
        for array in arrays:
            array.setflags(write=False)
        object.__setattr__(self, "raw_program", arrays[0])
        object.__setattr__(self, "receiver_program_score", arrays[1])
        object.__setattr__(self, "supported_program_score", arrays[2])


def _readonly_array(
    values: np.ndarray, *, shape: tuple[int, ...], field_name: str
) -> np.ndarray:
    owned = np.asarray(values, dtype=np.float64, order="C").copy(order="C")
    if owned.shape != shape or np.any(~np.isfinite(owned)):
        raise ValueError(f"{field_name} must be finite with shape {shape}")
    result = cast(
        np.ndarray,
        np.frombuffer(owned.tobytes(order="C"), dtype=np.float64).reshape(shape),
    )
    result.setflags(write=False)
    return result


def _immutable_native_vector(values: np.ndarray) -> np.ndarray:
    owned = np.asarray(values).copy(order="C")
    result = cast(
        np.ndarray,
        np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape),
    )
    result.setflags(write=False)
    return result


def _immutable_csc(matrix: sparse.csc_matrix) -> sparse.csc_matrix:
    data = _immutable_native_vector(np.asarray(matrix.data, dtype=np.float64))
    indices = _immutable_native_vector(np.asarray(matrix.indices))
    indptr = _immutable_native_vector(np.asarray(matrix.indptr))
    result = sparse.csc_matrix((data, indices, indptr), shape=matrix.shape, copy=False)
    result.data.setflags(write=False)
    result.indices.setflags(write=False)
    result.indptr.setflags(write=False)
    return result


def _aligned_names(
    values: tuple[str, ...], *, length: int, field_name: str
) -> tuple[str, ...]:
    if len(values) != length or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in values
    ):
        raise ValueError(
            f"{field_name} must contain {length} canonical non-empty strings"
        )
    return tuple(values)


def _dense_input_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64, order="C")
    result: str = stable_id(
        "dense_float64_input",
        {
            "shape": list(array.shape),
            "values_digest": float64_array_digest(array),
        },
        schema_version="2",
        digest_length=64,
    )
    return result


def _numeric_array_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64, order="C")
    if np.any(np.isinf(array)):
        raise ValueError("numeric artifact arrays must not contain infinity")
    tokens = [
        None if math.isnan(float(value)) else float(value).hex() for value in array.flat
    ]
    result: str = stable_id(
        "numeric_array",
        {"shape": list(array.shape), "float64_tokens": tokens},
        schema_version="2",
        digest_length=64,
    )
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class DownstreamRowManifest:
    """Explicit row identity for response matrices used by incremental scoring."""

    sample_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        samples = _names(tuple(self.sample_ids), field_name="sample_ids")
        subjects = _aligned_names(
            tuple(self.subject_ids),
            length=len(samples),
            field_name="subject_ids",
        )
        contexts = _aligned_names(
            tuple(self.context_ids),
            length=len(samples),
            field_name="context_ids",
        )
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "subject_ids", subjects)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(
            self,
            "manifest_id",
            stable_id(
                "downstream_row_manifest",
                self._identity_payload(),
                schema_version="2",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        rows = sorted(
            (
                {
                    "context_id": context_id,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                }
                for sample_id, subject_id, context_id in zip(
                    self.sample_ids,
                    self.subject_ids,
                    self.context_ids,
                    strict=True,
                )
            ),
            key=lambda row: row["sample_id"],
        )
        return {"rows": rows}

    def _require_intact(self) -> None:
        expected = stable_id(
            "downstream_row_manifest",
            self._identity_payload(),
            schema_version="2",
        )
        if expected != self.manifest_id:
            raise ContractError(
                "Downstream row manifest failed integrity validation",
                code="downstream_row_manifest_integrity_violation",
                field="manifest_id",
                remediation="Rebuild the row manifest from source metadata",
            )


@dataclass(frozen=True, slots=True)
class _CanonicalIncrementalRows:
    response: np.ndarray
    nuisance: np.ndarray
    regressor: np.ndarray
    sample_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    reference: np.ndarray | None


def _canonical_incremental_rows(
    response_matrix: np.ndarray,
    *,
    row_manifest: DownstreamRowManifest,
    nuisance_matrix: np.ndarray,
    context_regressor: np.ndarray,
    design_sample_ids: tuple[str, ...],
    n_features: int,
    n_nuisance: int,
    reference_mask: np.ndarray | None = None,
) -> _CanonicalIncrementalRows:
    if not isinstance(row_manifest, DownstreamRowManifest):
        raise TypeError("row_manifest must be a DownstreamRowManifest")
    row_manifest._require_intact()
    response = np.asarray(response_matrix, dtype=np.float64)
    n_samples = len(row_manifest.sample_ids)
    if response.shape != (n_samples, n_features) or np.any(~np.isfinite(response)):
        raise ValueError(
            "response_matrix must be finite and align with row_manifest x features"
        )
    design_ids = _names(tuple(design_sample_ids), field_name="design_sample_ids")
    if len(design_ids) != n_samples or set(design_ids) != set(row_manifest.sample_ids):
        raise ValueError(
            "design_sample_ids must exactly match the response row sample universe"
        )
    nuisance = np.asarray(nuisance_matrix, dtype=np.float64)
    regressor = np.asarray(context_regressor, dtype=np.float64)
    if nuisance.shape != (n_samples, n_nuisance) or np.any(~np.isfinite(nuisance)):
        raise ValueError(
            "nuisance_matrix must be finite and align with design_sample_ids"
        )
    if regressor.shape != (n_samples,) or np.any(~np.isfinite(regressor)):
        raise ValueError(
            "context_regressor must be finite and align with design_sample_ids"
        )
    reference: np.ndarray | None = None
    if reference_mask is not None:
        reference = np.asarray(reference_mask)
        if reference.shape != (n_samples,) or reference.dtype.kind != "b":
            raise ValueError(
                "reference_mask must be a boolean vector aligned with design_sample_ids"
            )
    response_by_sample = {
        sample_id: index for index, sample_id in enumerate(row_manifest.sample_ids)
    }
    design_by_sample = {sample_id: index for index, sample_id in enumerate(design_ids)}
    canonical_samples = tuple(sorted(row_manifest.sample_ids))
    response_order = np.asarray(
        [response_by_sample[sample_id] for sample_id in canonical_samples], dtype=int
    )
    design_order = np.asarray(
        [design_by_sample[sample_id] for sample_id in canonical_samples], dtype=int
    )
    canonical_subjects = tuple(
        row_manifest.subject_ids[index] for index in response_order
    )
    canonical_contexts = tuple(
        row_manifest.context_ids[index] for index in response_order
    )
    return _CanonicalIncrementalRows(
        response=np.asarray(response[response_order], dtype=np.float64, order="C"),
        nuisance=np.asarray(nuisance[design_order], dtype=np.float64, order="C"),
        regressor=np.asarray(regressor[design_order], dtype=np.float64, order="C"),
        sample_ids=canonical_samples,
        subject_ids=canonical_subjects,
        context_ids=canonical_contexts,
        reference=(
            None
            if reference is None
            else np.asarray(reference[design_order], dtype=bool)
        ),
    )


def _subject_context_weights(
    subject_ids: tuple[str, ...], context_ids: tuple[str, ...]
) -> np.ndarray:
    """Give subjects, then contexts, then technical rows equal total weight."""

    pairs = tuple(zip(subject_ids, context_ids, strict=True))
    contexts_by_subject = {
        subject: {context for row_subject, context in pairs if row_subject == subject}
        for subject in set(subject_ids)
    }
    pair_counts = {pair: sum(value == pair for value in pairs) for pair in set(pairs)}
    weights: np.ndarray = np.asarray(
        [
            1.0 / (len(contexts_by_subject[subject]) * pair_counts[(subject, context)])
            for subject, context in pairs
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("subject-context weights must be finite and positive")
    return weights


def _weighted_lstsq(
    design: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    root = np.sqrt(np.asarray(weights, dtype=np.float64))
    weighted_design = design * root[:, None]
    weighted_values = values * (root if values.ndim == 1 else root[:, None])
    result: np.ndarray = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[
        0
    ]
    return result


def _subject_reference_summary(
    response: np.ndarray,
    *,
    reference: np.ndarray,
    subject_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> np.ndarray:
    """Average technical rows and reference contexts before robust subjects."""

    reference_subjects = tuple(
        sorted(
            {
                subject
                for subject, selected in zip(subject_ids, reference, strict=True)
                if bool(selected)
            }
        )
    )
    summaries: list[np.ndarray] = []
    for subject in reference_subjects:
        subject_contexts = tuple(
            sorted(
                {
                    context
                    for row_subject, context, selected in zip(
                        subject_ids, context_ids, reference, strict=True
                    )
                    if row_subject == subject and bool(selected)
                }
            )
        )
        context_means = [
            np.mean(
                response[
                    np.asarray(
                        [
                            row_subject == subject
                            and row_context == context
                            and bool(selected)
                            for row_subject, row_context, selected in zip(
                                subject_ids,
                                context_ids,
                                reference,
                                strict=True,
                            )
                        ],
                        dtype=bool,
                    )
                ],
                axis=0,
            )
            for context in subject_contexts
        ]
        summaries.append(np.mean(context_means, axis=0))
    result: np.ndarray = np.asarray(summaries, dtype=np.float64)
    return result


@dataclass(frozen=True, slots=True, init=False)
class IncrementalDownstreamFunctional:
    """Producer-owned sample-keyed M0/M1 models learned on training subjects."""

    receiver: str
    contrast_name: str
    fold_id: str
    context_regressor_id: str
    nuisance_design_id: str
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    nuisance_column_ids: tuple[str, ...]
    training_sample_ids: tuple[str, ...]
    training_sample_subject_ids: tuple[str, ...]
    training_sample_context_ids: tuple[str, ...]
    training_context_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    reference_sample_ids: tuple[str, ...]
    training_row_manifest_id: str
    training_input_digest: str
    feature_center: np.ndarray
    feature_scale: np.ndarray
    family_basis: sparse.csc_matrix
    family_coefficients: np.ndarray
    null_nuisance_coefficients: np.ndarray
    full_nuisance_coefficients: np.ndarray
    family_nuisance_coefficients: np.ndarray
    precision_weights: np.ndarray
    minimum_scale: float
    null_loss_floor: float
    lambda1: float
    lambda2: float
    identity_scope: str
    family_gain_estimand: str
    certification_status: str
    incremental_functional_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "IncrementalDownstreamFunctional is producer-owned; "
            "use fit_incremental_downstream_functional()"
        )

    @classmethod
    def _from_training(
        cls,
        *,
        receiver: str,
        contrast_name: str,
        fold_id: str,
        context_regressor_id: str,
        nuisance_design_id: str,
        feature_ids: tuple[str, ...],
        family_ids: tuple[str, ...],
        nuisance_column_ids: tuple[str, ...],
        training_sample_ids: tuple[str, ...],
        training_sample_subject_ids: tuple[str, ...],
        training_sample_context_ids: tuple[str, ...],
        training_subject_ids: tuple[str, ...],
        reference_sample_ids: tuple[str, ...],
        training_row_manifest_id: str,
        training_input_digest: str,
        feature_center: np.ndarray,
        feature_scale: np.ndarray,
        family_basis: sparse.csc_matrix,
        family_coefficients: np.ndarray,
        null_nuisance_coefficients: np.ndarray,
        full_nuisance_coefficients: np.ndarray,
        family_nuisance_coefficients: np.ndarray,
        precision_weights: np.ndarray,
        minimum_scale: float,
        null_loss_floor: float,
        lambda1: float,
        lambda2: float,
    ) -> IncrementalDownstreamFunctional:
        names: dict[str, str] = {}
        for field_name, scope_value in (
            ("receiver", receiver),
            ("contrast_name", contrast_name),
            ("fold_id", fold_id),
            ("context_regressor_id", context_regressor_id),
            ("nuisance_design_id", nuisance_design_id),
            ("training_row_manifest_id", training_row_manifest_id),
            ("training_input_digest", training_input_digest),
        ):
            if (
                not isinstance(scope_value, str)
                or not scope_value
                or scope_value != scope_value.strip()
            ):
                raise ValueError(f"{field_name} must be a canonical non-empty string")
            names[field_name] = scope_value
        features = _names(feature_ids, field_name="feature_ids")
        families = _names(family_ids, field_name="family_ids")
        nuisance_ids = _names(nuisance_column_ids, field_name="nuisance_column_ids")
        samples = _names(training_sample_ids, field_name="training_sample_ids")
        sample_subjects = _aligned_names(
            training_sample_subject_ids,
            length=len(samples),
            field_name="training_sample_subject_ids",
        )
        sample_contexts = _aligned_names(
            training_sample_context_ids,
            length=len(samples),
            field_name="training_sample_context_ids",
        )
        contexts = tuple(sorted(set(sample_contexts)))
        if len(contexts) < 2:
            raise ValueError("incremental training requires at least two context IDs")
        subjects = tuple(
            sorted(
                _names(
                    training_subject_ids,
                    field_name="training_subject_ids",
                )
            )
        )
        references = _names(reference_sample_ids, field_name="reference_sample_ids")
        if not set(references).issubset(samples):
            raise ValueError("reference_sample_ids must belong to training samples")
        for field_name, numeric_value, allow_zero in (
            ("minimum_scale", minimum_scale, False),
            ("null_loss_floor", null_loss_floor, False),
            ("lambda1", lambda1, True),
            ("lambda2", lambda2, True),
        ):
            valid = math.isfinite(numeric_value) and (
                numeric_value >= 0 if allow_zero else numeric_value > 0
            )
            if not valid:
                relation = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{field_name} must be finite and {relation}")
        n_features = len(features)
        n_families = len(families)
        n_nuisance = len(nuisance_ids)
        center = _readonly_array(
            feature_center, shape=(n_features,), field_name="feature_center"
        )
        scale = _readonly_array(
            feature_scale, shape=(n_features,), field_name="feature_scale"
        )
        if np.any(scale <= 0) or np.any(scale < minimum_scale - 1e-12):
            raise ValueError("feature_scale must be positive and respect minimum_scale")
        basis = sparse.csc_matrix(family_basis, dtype=np.float64).copy()
        if basis.shape != (n_features, n_families):
            raise ValueError("family_basis shape must equal features x families")
        if np.any(~np.isfinite(basis.data)) or np.any(basis.data < 0):
            raise ValueError("family_basis must be finite and non-negative")
        basis.sum_duplicates()
        basis.sort_indices()
        column_norms = np.sqrt(np.asarray(basis.power(2).sum(axis=0)).ravel())
        if np.any(~np.isclose(column_norms, 1.0, atol=1e-12, rtol=1e-12)):
            raise ValueError("every family basis column must have unit L2 norm")
        coefficients = _readonly_array(
            family_coefficients,
            shape=(n_families,),
            field_name="family_coefficients",
        )
        if np.any(coefficients < 0):
            raise ValueError("family_coefficients must be non-negative")
        null_coefficients = _readonly_array(
            null_nuisance_coefficients,
            shape=(n_nuisance, n_features),
            field_name="null_nuisance_coefficients",
        )
        full_coefficients = _readonly_array(
            full_nuisance_coefficients,
            shape=(n_nuisance, n_features),
            field_name="full_nuisance_coefficients",
        )
        family_nuisance = _readonly_array(
            family_nuisance_coefficients,
            shape=(n_families, n_nuisance, n_features),
            field_name="family_nuisance_coefficients",
        )
        precision = _readonly_array(
            precision_weights,
            shape=(n_features,),
            field_name="precision_weights",
        )
        if np.any(precision < 0) or not np.any(precision > 0):
            raise ValueError("precision_weights must be non-negative and not all zero")
        basis = _immutable_csc(basis)
        self = object.__new__(cls)
        attributes: dict[str, Any] = {
            **names,
            "feature_ids": features,
            "family_ids": families,
            "nuisance_column_ids": nuisance_ids,
            "training_sample_ids": samples,
            "training_sample_subject_ids": sample_subjects,
            "training_sample_context_ids": sample_contexts,
            "training_context_ids": contexts,
            "training_subject_ids": subjects,
            "reference_sample_ids": references,
            "feature_center": center,
            "feature_scale": scale,
            "family_basis": basis,
            "family_coefficients": coefficients,
            "null_nuisance_coefficients": null_coefficients,
            "full_nuisance_coefficients": full_coefficients,
            "family_nuisance_coefficients": family_nuisance,
            "precision_weights": precision,
            "minimum_scale": minimum_scale,
            "null_loss_floor": null_loss_floor,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "identity_scope": "sample_keyed_v2",
            "family_gain_estimand": _FAMILY_GAIN_ESTIMAND,
            "certification_status": "partial_not_oof_certified",
            "_producer_marker": _INCREMENTAL_PRODUCER_MARKER,
        }
        for name, value in attributes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "incremental_functional_id",
            stable_id(
                "incremental_downstream_functional",
                self._identity_payload(),
                schema_version="2",
            ),
        )
        return self

    @property
    def is_oof_certified(self) -> bool:
        """Remain false until precision, tuning, and public crossfit are frozen."""

        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm": _INCREMENTAL_METHOD,
            "center_digest": float64_array_digest(self.feature_center),
            "certification_status": self.certification_status,
            "context_regressor_id": self.context_regressor_id,
            "contrast_name": self.contrast_name,
            "family_basis_id": _matrix_digest(self.family_basis),
            "family_coefficient_digest": float64_array_digest(self.family_coefficients),
            "family_gain_estimand": self.family_gain_estimand,
            "family_ids": list(self.family_ids),
            "family_nuisance_digest": float64_array_digest(
                self.family_nuisance_coefficients
            ),
            "feature_ids": list(self.feature_ids),
            "fold_id": self.fold_id,
            "full_nuisance_digest": float64_array_digest(
                self.full_nuisance_coefficients
            ),
            "identity_scope": self.identity_scope,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "minimum_scale": self.minimum_scale,
            "nuisance_column_ids": list(self.nuisance_column_ids),
            "nuisance_design_id": self.nuisance_design_id,
            "null_loss_floor": self.null_loss_floor,
            "null_nuisance_digest": float64_array_digest(
                self.null_nuisance_coefficients
            ),
            "precision_digest": float64_array_digest(self.precision_weights),
            "receiver": self.receiver,
            "reference_sample_ids": list(self.reference_sample_ids),
            "scale_digest": float64_array_digest(self.feature_scale),
            "training_input_digest": self.training_input_digest,
            "training_row_manifest_id": self.training_row_manifest_id,
            "training_rows": [
                {
                    "context_id": context_id,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                }
                for sample_id, subject_id, context_id in zip(
                    self.training_sample_ids,
                    self.training_sample_subject_ids,
                    self.training_sample_context_ids,
                    strict=True,
                )
            ],
            "training_context_ids": list(self.training_context_ids),
            "training_subject_ids": list(self.training_subject_ids),
        }

    def _require_intact(self) -> None:
        if self._producer_marker != _INCREMENTAL_PRODUCER_MARKER:
            raise TypeError("incremental downstream functional is not producer-owned")
        try:
            expected = stable_id(
                "incremental_downstream_functional",
                self._identity_payload(),
                schema_version="2",
            )
        except (ValueError, TypeError) as error:
            raise ContractError(
                "Incremental downstream artifact failed integrity validation",
                code="incremental_functional_integrity_violation",
                field="incremental_functional_id",
                remediation="Refit the functional from frozen training inputs",
            ) from error
        if expected != self.incremental_functional_id:
            raise ContractError(
                "Incremental downstream artifact failed integrity validation",
                code="incremental_functional_integrity_violation",
                field="incremental_functional_id",
                remediation="Refit the functional from frozen training inputs",
            )

    def to_dict(self) -> dict[str, object]:
        """Return learned-artifact provenance without materializing coefficients."""

        self._require_intact()
        return {
            "incremental_functional_id": self.incremental_functional_id,
            "receiver": self.receiver,
            "contrast_name": self.contrast_name,
            "fold_id": self.fold_id,
            "context_regressor_id": self.context_regressor_id,
            "nuisance_design_id": self.nuisance_design_id,
            "feature_ids": list(self.feature_ids),
            "family_ids": list(self.family_ids),
            "nuisance_column_ids": list(self.nuisance_column_ids),
            "training_sample_ids": list(self.training_sample_ids),
            "training_context_ids": list(self.training_context_ids),
            "training_subject_ids": list(self.training_subject_ids),
            "reference_sample_ids": list(self.reference_sample_ids),
            "training_row_manifest_id": self.training_row_manifest_id,
            "training_input_digest": self.training_input_digest,
            "feature_center_digest": float64_array_digest(self.feature_center),
            "feature_scale_digest": float64_array_digest(self.feature_scale),
            "family_basis_id": _matrix_digest(self.family_basis),
            "family_coefficient_digest": float64_array_digest(self.family_coefficients),
            "null_nuisance_digest": float64_array_digest(
                self.null_nuisance_coefficients
            ),
            "full_nuisance_digest": float64_array_digest(
                self.full_nuisance_coefficients
            ),
            "family_nuisance_digest": float64_array_digest(
                self.family_nuisance_coefficients
            ),
            "precision_digest": float64_array_digest(self.precision_weights),
            "minimum_scale": self.minimum_scale,
            "null_loss_floor": self.null_loss_floor,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "identity_scope": self.identity_scope,
            "family_gain_estimand": self.family_gain_estimand,
            "certification_status": self.certification_status,
            "is_oof_certified": self.is_oof_certified,
        }


@dataclass(frozen=True, slots=True, init=False)
class IncrementalDownstreamApplication:
    """Producer-owned held-out losses with subject-equal aggregation."""

    incremental_functional_id: str
    application_id: str
    status: str
    reason_code: str | None
    family_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_context_ids: tuple[str, ...]
    heldout_subject_ids: tuple[str, ...]
    heldout_row_manifest_id: str
    heldout_input_digest: str
    null_loss: float | None
    full_loss: float | None
    model_gain: float | None
    raw_model_gain: float | None
    family_losses: np.ndarray
    family_gains: np.ndarray
    raw_family_gains: np.ndarray
    sample_null_losses: np.ndarray
    sample_full_losses: np.ndarray
    sample_family_losses: np.ndarray
    subject_ids: tuple[str, ...]
    subject_null_losses: np.ndarray
    subject_full_losses: np.ndarray
    subject_family_losses: np.ndarray
    loss_aggregation: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "IncrementalDownstreamApplication is producer-owned; "
            "use apply_incremental_downstream_functional()"
        )

    @classmethod
    def _from_application(
        cls,
        *,
        functional: IncrementalDownstreamFunctional,
        status: str,
        reason_code: str | None,
        sample_ids: tuple[str, ...],
        sample_subject_ids: tuple[str, ...],
        sample_context_ids: tuple[str, ...],
        heldout_row_manifest_id: str,
        heldout_input_digest: str,
        null_loss: float | None,
        full_loss: float | None,
        model_gain: float | None,
        raw_model_gain: float | None,
        family_losses: np.ndarray,
        family_gains: np.ndarray,
        raw_family_gains: np.ndarray,
        sample_null_losses: np.ndarray,
        sample_full_losses: np.ndarray,
        sample_family_losses: np.ndarray,
        subject_ids: tuple[str, ...],
        subject_null_losses: np.ndarray,
        subject_full_losses: np.ndarray,
        subject_family_losses: np.ndarray,
    ) -> IncrementalDownstreamApplication:
        functional._require_intact()
        if status not in {"observed", "not_estimable"}:
            raise ValueError("status must be observed or not_estimable")
        observed = status == "observed"
        if observed == (reason_code is not None):
            raise ValueError("reason_code must be present exactly when not estimable")
        samples = _names(sample_ids, field_name="sample_ids")
        sample_subjects = _aligned_names(
            sample_subject_ids,
            length=len(samples),
            field_name="sample_subject_ids",
        )
        sample_contexts = _aligned_names(
            sample_context_ids,
            length=len(samples),
            field_name="sample_context_ids",
        )
        for field_name, value in (
            ("heldout_row_manifest_id", heldout_row_manifest_id),
            ("heldout_input_digest", heldout_input_digest),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        n_samples = len(samples)
        n_subjects = len(subject_ids)
        n_families = len(functional.family_ids)
        arrays = {
            "family_losses": (family_losses, (n_families,)),
            "family_gains": (family_gains, (n_families,)),
            "raw_family_gains": (raw_family_gains, (n_families,)),
            "sample_null_losses": (sample_null_losses, (n_samples,)),
            "sample_full_losses": (sample_full_losses, (n_samples,)),
            "sample_family_losses": (sample_family_losses, (n_samples, n_families)),
            "subject_null_losses": (subject_null_losses, (n_subjects,)),
            "subject_full_losses": (subject_full_losses, (n_subjects,)),
            "subject_family_losses": (subject_family_losses, (n_subjects, n_families)),
        }
        frozen: dict[str, np.ndarray] = {}
        for field_name, (values, shape) in arrays.items():
            array = np.asarray(values, dtype=np.float64)
            if array.shape != shape or np.any(np.isinf(array)):
                raise ValueError(f"{field_name} has invalid shape or infinite values")
            if observed and np.any(~np.isfinite(array)):
                raise ValueError(f"observed {field_name} must be finite")
            if not observed and not np.isnan(array).all():
                raise ValueError(f"not-estimable {field_name} must be NaN")
            owned = np.asarray(array, dtype=np.float64, order="C").copy(order="C")
            immutable = np.frombuffer(
                owned.tobytes(order="C"), dtype=np.float64
            ).reshape(shape)
            immutable.setflags(write=False)
            frozen[field_name] = immutable
        scalars = (null_loss, full_loss, model_gain, raw_model_gain)
        if observed:
            if any(value is None or not math.isfinite(value) for value in scalars):
                raise ValueError("observed incremental evidence requires finite losses")
            if model_gain is None or not 0 <= model_gain <= 1:
                raise ValueError("observed model_gain must lie in [0, 1]")
            if np.any((frozen["family_gains"] < 0) | (frozen["family_gains"] > 1)):
                raise ValueError("observed family_gains must lie in [0, 1]")
            loss_fields = (
                "family_losses",
                "sample_null_losses",
                "sample_full_losses",
                "sample_family_losses",
                "subject_null_losses",
                "subject_full_losses",
                "subject_family_losses",
            )
            if any(np.any(frozen[field_name] < 0) for field_name in loss_fields):
                raise ValueError("observed prediction losses must be non-negative")
            assert null_loss is not None
            assert full_loss is not None
            assert raw_model_gain is not None
            assert model_gain is not None
            if null_loss <= functional.null_loss_floor or full_loss < 0:
                raise ValueError("observed aggregate losses are invalid")
            expected_raw_model = (null_loss - full_loss) / null_loss
            if not math.isclose(
                raw_model_gain,
                expected_raw_model,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ) or not math.isclose(
                model_gain,
                float(np.clip(raw_model_gain, 0.0, 1.0)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("model gains do not match aggregate losses")
            expected_raw_family = (null_loss - frozen["family_losses"]) / null_loss
            if not np.allclose(
                frozen["raw_family_gains"],
                expected_raw_family,
                rtol=1e-12,
                atol=1e-12,
            ) or not np.allclose(
                frozen["family_gains"],
                np.clip(frozen["raw_family_gains"], 0.0, 1.0),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("family gains do not match aggregate losses")
            aggregate_checks = (
                (null_loss, frozen["subject_null_losses"]),
                (full_loss, frozen["subject_full_losses"]),
            )
            if any(
                not math.isclose(
                    aggregate,
                    float(np.mean(values)),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for aggregate, values in aggregate_checks
            ) or not np.allclose(
                frozen["family_losses"],
                np.mean(frozen["subject_family_losses"], axis=0),
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError("aggregate losses do not match subject losses")
        elif any(value is not None for value in scalars):
            raise ValueError("not-estimable incremental evidence must omit losses")
        subjects = tuple(sorted(_names(subject_ids, field_name="subject_ids")))
        heldout_subjects = tuple(sorted(set(sample_subjects)))
        if subjects != heldout_subjects:
            raise ValueError("subject_ids must equal the held-out row subject universe")
        self = object.__new__(cls)
        attributes: dict[str, Any] = {
            "incremental_functional_id": functional.incremental_functional_id,
            "status": status,
            "reason_code": reason_code,
            "family_ids": functional.family_ids,
            "sample_ids": samples,
            "sample_subject_ids": sample_subjects,
            "sample_context_ids": sample_contexts,
            "heldout_subject_ids": heldout_subjects,
            "heldout_row_manifest_id": heldout_row_manifest_id,
            "heldout_input_digest": heldout_input_digest,
            "null_loss": null_loss,
            "full_loss": full_loss,
            "model_gain": model_gain,
            "raw_model_gain": raw_model_gain,
            "subject_ids": subjects,
            "loss_aggregation": (
                "mean_technical_within_subject_context_then_equal_context_"
                "within_subject_then_equal_subject_mean_v2"
            ),
            "_producer_marker": _INCREMENTAL_PRODUCER_MARKER,
            **frozen,
        }
        for name, value in attributes.items():
            object.__setattr__(self, name, value)
        payload = self._identity_payload()
        object.__setattr__(
            self,
            "application_id",
            stable_id(
                "incremental_downstream_application",
                payload,
                schema_version="2",
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm": _INCREMENTAL_METHOD,
            "family_ids": list(self.family_ids),
            "family_losses_digest": _numeric_array_digest(self.family_losses),
            "family_gains_digest": _numeric_array_digest(self.family_gains),
            "full_loss": self.full_loss,
            "heldout_input_digest": self.heldout_input_digest,
            "heldout_row_manifest_id": self.heldout_row_manifest_id,
            "heldout_rows": [
                {
                    "context_id": context_id,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                }
                for sample_id, subject_id, context_id in zip(
                    self.sample_ids,
                    self.sample_subject_ids,
                    self.sample_context_ids,
                    strict=True,
                )
            ],
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "incremental_functional_id": self.incremental_functional_id,
            "loss_aggregation": self.loss_aggregation,
            "model_gain": self.model_gain,
            "null_loss": self.null_loss,
            "raw_family_gains_digest": _numeric_array_digest(self.raw_family_gains),
            "raw_model_gain": self.raw_model_gain,
            "reason_code": self.reason_code,
            "sample_family_losses_digest": _numeric_array_digest(
                self.sample_family_losses
            ),
            "sample_full_losses_digest": _numeric_array_digest(self.sample_full_losses),
            "sample_null_losses_digest": _numeric_array_digest(self.sample_null_losses),
            "status": self.status,
            "subject_family_losses_digest": _numeric_array_digest(
                self.subject_family_losses
            ),
            "subject_ids": list(self.subject_ids),
            "subject_full_losses_digest": _numeric_array_digest(
                self.subject_full_losses
            ),
            "subject_null_losses_digest": _numeric_array_digest(
                self.subject_null_losses
            ),
        }

    def _require_intact(self) -> None:
        if self._producer_marker != _INCREMENTAL_PRODUCER_MARKER:
            raise TypeError("incremental downstream application is not producer-owned")
        try:
            expected = stable_id(
                "incremental_downstream_application",
                self._identity_payload(),
                schema_version="2",
            )
        except (ValueError, TypeError) as error:
            raise ContractError(
                "Incremental application failed integrity validation",
                code="incremental_application_integrity_violation",
                field="application_id",
                remediation="Reapply the frozen functional to held-out samples",
            ) from error
        if expected != self.application_id:
            raise ContractError(
                "Incremental application failed integrity validation",
                code="incremental_application_integrity_violation",
                field="application_id",
                remediation="Reapply the frozen functional to held-out samples",
            )

    def to_dict(self) -> dict[str, object]:
        """Return held-out provenance without expanding loss arrays."""

        self._require_intact()
        return {
            "application_id": self.application_id,
            "incremental_functional_id": self.incremental_functional_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "family_ids": list(self.family_ids),
            "sample_ids": list(self.sample_ids),
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "heldout_row_manifest_id": self.heldout_row_manifest_id,
            "heldout_input_digest": self.heldout_input_digest,
            "null_loss": self.null_loss,
            "full_loss": self.full_loss,
            "model_gain": self.model_gain,
            "raw_model_gain": self.raw_model_gain,
            "family_losses_digest": _numeric_array_digest(self.family_losses),
            "family_gains_digest": _numeric_array_digest(self.family_gains),
            "raw_family_gains_digest": _numeric_array_digest(self.raw_family_gains),
            "loss_aggregation": self.loss_aggregation,
        }


def _normalize_family_basis(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    n_features: int,
    n_families: int,
) -> sparse.csc_matrix:
    basis = sparse.csc_matrix(matrix, dtype=np.float64).copy()
    if basis.shape != (n_features, n_families):
        raise ValueError("family_basis shape must equal features x families")
    if np.any(~np.isfinite(basis.data)) or np.any(basis.data < 0):
        raise ValueError("family_basis must be finite and non-negative")
    norms = np.sqrt(np.asarray(basis.power(2).sum(axis=0)).ravel())
    if np.any(norms <= 0):
        raise ValueError("every family basis column must have positive L2 norm")
    normalized = basis @ sparse.diags(1.0 / norms, format="csc")
    normalized.sum_duplicates()
    normalized.sort_indices()
    return sparse.csc_matrix(normalized)


def fit_incremental_downstream_functional(
    response_matrix: np.ndarray,
    *,
    row_manifest: DownstreamRowManifest,
    design_sample_ids: tuple[str, ...],
    reference_mask: np.ndarray,
    nuisance_matrix: np.ndarray,
    context_regressor: np.ndarray,
    receiver: str,
    contrast_name: str,
    fold_id: str,
    context_regressor_id: str,
    nuisance_design_id: str,
    feature_ids: tuple[str, ...],
    family_ids: tuple[str, ...],
    nuisance_column_ids: tuple[str, ...],
    training_subject_ids: tuple[str, ...],
    family_basis: sparse.spmatrix | np.ndarray,
    precision_weights: np.ndarray | None = None,
    minimum_scale: float = 0.25,
    null_loss_floor: float = 1e-8,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
) -> IncrementalDownstreamFunctional:
    """Fit sample-keyed nuisance and non-negative LR-family response models."""

    features = _names(feature_ids, field_name="feature_ids")
    families = _names(family_ids, field_name="family_ids")
    nuisance_ids = _names(nuisance_column_ids, field_name="nuisance_column_ids")
    canonical = _canonical_incremental_rows(
        response_matrix,
        row_manifest=row_manifest,
        nuisance_matrix=nuisance_matrix,
        context_regressor=context_regressor,
        design_sample_ids=design_sample_ids,
        n_features=len(features),
        n_nuisance=len(nuisance_ids),
        reference_mask=reference_mask,
    )
    response = canonical.response
    nuisance = canonical.nuisance
    regressor = canonical.regressor
    assert canonical.reference is not None
    reference = canonical.reference
    n_samples, n_features = response.shape
    if n_samples < 3:
        raise ValueError("response_matrix requires at least three complete samples")
    if int(np.count_nonzero(reference)) < 2:
        raise ValueError("reference_mask must select at least two training samples")
    reference_subjects = {
        canonical.subject_ids[index] for index in np.flatnonzero(reference)
    }
    if len(reference_subjects) < 2:
        raise ValueError("reference_mask must select at least two training subjects")
    if np.linalg.matrix_rank(nuisance) != nuisance.shape[1]:
        raise ValueError("training nuisance_matrix must have full column rank")
    if not math.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("minimum_scale must be finite and positive")
    if not math.isfinite(null_loss_floor) or null_loss_floor <= 0:
        raise ValueError("null_loss_floor must be finite and positive")
    reference_by_subject = _subject_reference_summary(
        response,
        reference=reference,
        subject_ids=canonical.subject_ids,
        context_ids=canonical.context_ids,
    )
    center = np.median(reference_by_subject, axis=0)
    mad = np.median(np.abs(reference_by_subject - center), axis=0)
    scale = np.maximum(_MAD_GAUSSIAN_CONSISTENCY * mad, minimum_scale)
    standardized = (response - center) / scale
    training_weights = _subject_context_weights(
        canonical.subject_ids, canonical.context_ids
    )

    null_coefficients = _weighted_lstsq(nuisance, standardized, training_weights)
    null_residual = standardized - nuisance @ null_coefficients
    regressor_nuisance = _weighted_lstsq(nuisance, regressor, training_weights)
    residualized_regressor = regressor - nuisance @ regressor_nuisance
    regressor_norm = float(
        np.dot(
            residualized_regressor * training_weights,
            residualized_regressor,
        )
    )
    if regressor_norm <= null_loss_floor:
        raise ValueError("context regressor has no variation after nuisance projection")
    effect = (
        (residualized_regressor * training_weights) @ null_residual / regressor_norm
    )
    directional_effect = np.maximum(np.asarray(effect, dtype=np.float64), 0.0)
    basis = _normalize_family_basis(
        family_basis,
        n_features=n_features,
        n_families=len(families),
    )
    precision = (
        np.ones(n_features, dtype=np.float64)
        if precision_weights is None
        else np.asarray(precision_weights, dtype=np.float64)
    )
    solution = solve_nonnegative_elastic_net(
        basis,
        directional_effect,
        precision_weights=precision,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    if solution.diagnostics.status is not SolverStatus.CONVERGED:
        raise ValueError(
            "incremental downstream family solver did not converge: "
            f"{solution.diagnostics.failure_reason}"
        )
    predicted_effect = np.asarray(basis @ solution.coefficients).ravel()
    full_adjusted = standardized - np.outer(regressor, predicted_effect)
    full_coefficients = _weighted_lstsq(nuisance, full_adjusted, training_weights)
    family_nuisance = np.empty(
        (len(families), nuisance.shape[1], n_features), dtype=np.float64
    )
    for family_index, coefficient in enumerate(solution.coefficients):
        contribution = (
            np.asarray(basis.getcol(family_index).toarray()).ravel() * coefficient
        )
        adjusted = standardized - np.outer(regressor, contribution)
        family_nuisance[family_index] = _weighted_lstsq(
            nuisance, adjusted, training_weights
        )
    declared_subjects = tuple(
        sorted(_names(training_subject_ids, field_name="training_subject_ids"))
    )
    derived_subjects = tuple(sorted(set(canonical.subject_ids)))
    if declared_subjects != derived_subjects:
        raise ValueError(
            "training_subject_ids must exactly match the row manifest subject universe"
        )
    reference_sample_ids = tuple(
        sample_id
        for sample_id, selected in zip(canonical.sample_ids, reference, strict=True)
        if bool(selected)
    )
    training_input_digest = stable_id(
        "incremental_training_input",
        {
            "algorithm": _INCREMENTAL_METHOD,
            "family_basis_id": _matrix_digest(basis),
            "feature_ids": list(features),
            "nuisance_column_ids": list(nuisance_ids),
            "nuisance_digest": _dense_input_digest(nuisance),
            "precision_digest": _dense_input_digest(precision),
            "reference_sample_ids": list(reference_sample_ids),
            "regressor_digest": _dense_input_digest(regressor),
            "response_digest": _dense_input_digest(response),
            "row_manifest_id": row_manifest.manifest_id,
        },
        schema_version="2",
    )
    return IncrementalDownstreamFunctional._from_training(
        receiver=receiver,
        contrast_name=contrast_name,
        fold_id=fold_id,
        context_regressor_id=context_regressor_id,
        nuisance_design_id=nuisance_design_id,
        feature_ids=features,
        family_ids=families,
        nuisance_column_ids=nuisance_ids,
        training_sample_ids=canonical.sample_ids,
        training_sample_subject_ids=canonical.subject_ids,
        training_sample_context_ids=canonical.context_ids,
        training_subject_ids=derived_subjects,
        reference_sample_ids=reference_sample_ids,
        training_row_manifest_id=row_manifest.manifest_id,
        training_input_digest=training_input_digest,
        feature_center=center,
        feature_scale=scale,
        family_basis=basis,
        family_coefficients=solution.coefficients,
        null_nuisance_coefficients=null_coefficients,
        full_nuisance_coefficients=full_coefficients,
        family_nuisance_coefficients=family_nuisance,
        precision_weights=precision,
        minimum_scale=minimum_scale,
        null_loss_floor=null_loss_floor,
        lambda1=lambda1,
        lambda2=lambda2,
    )


def _prediction_losses(
    observed: np.ndarray, predicted: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    residual = observed - predicted
    result: np.ndarray = np.asarray(
        np.sum(residual * residual * precision, axis=1), dtype=np.float64
    )
    return result


def _subject_context_prediction_losses(
    observed: np.ndarray,
    predicted: np.ndarray,
    precision: np.ndarray,
    sample_subject_ids: tuple[str, ...],
    sample_context_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray, float]:
    observed_values = np.asarray(observed, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    if (
        observed_values.ndim != 2
        or predicted_values.shape != observed_values.shape
        or observed_values.shape[0] != len(sample_subject_ids)
        or len(sample_context_ids) != len(sample_subject_ids)
        or precision.shape != (observed_values.shape[1],)
    ):
        raise ValueError("predictions must align with subject/context feature rows")
    residual = observed_values - predicted_values
    subjects = tuple(sorted(set(sample_subject_ids)))
    subject_losses: np.ndarray = np.asarray(
        [
            np.mean(
                [
                    np.sum(
                        np.square(
                            np.mean(
                                residual[
                                    np.asarray(
                                        [
                                            row_subject == subject
                                            and row_context == context
                                            for row_subject, row_context in zip(
                                                sample_subject_ids,
                                                sample_context_ids,
                                                strict=True,
                                            )
                                        ],
                                        dtype=bool,
                                    )
                                ],
                                axis=0,
                            )
                        )
                        * precision
                    )
                    for context in sorted(
                        {
                            row_context
                            for row_subject, row_context in zip(
                                sample_subject_ids,
                                sample_context_ids,
                                strict=True,
                            )
                            if row_subject == subject
                        }
                    )
                ]
            )
            for subject in subjects
        ],
        dtype=np.float64,
    )
    return subjects, subject_losses, float(np.mean(subject_losses))


def _not_estimable_incremental_application(
    functional: IncrementalDownstreamFunctional,
    *,
    reason_code: str,
    canonical: _CanonicalIncrementalRows,
    heldout_row_manifest_id: str,
    heldout_input_digest: str,
) -> IncrementalDownstreamApplication:
    n_samples = len(canonical.sample_ids)
    n_families = len(functional.family_ids)
    subject_ids = tuple(sorted(set(canonical.subject_ids)))
    missing_family: np.ndarray = np.full(n_families, np.nan, dtype=np.float64)
    missing_sample: np.ndarray = np.full(n_samples, np.nan, dtype=np.float64)
    missing_subject: np.ndarray = np.full(len(subject_ids), np.nan, dtype=np.float64)
    return IncrementalDownstreamApplication._from_application(
        functional=functional,
        status="not_estimable",
        reason_code=reason_code,
        sample_ids=canonical.sample_ids,
        sample_subject_ids=canonical.subject_ids,
        sample_context_ids=canonical.context_ids,
        heldout_row_manifest_id=heldout_row_manifest_id,
        heldout_input_digest=heldout_input_digest,
        null_loss=None,
        full_loss=None,
        model_gain=None,
        raw_model_gain=None,
        family_losses=missing_family,
        family_gains=missing_family.copy(),
        raw_family_gains=missing_family.copy(),
        sample_null_losses=missing_sample,
        sample_full_losses=missing_sample.copy(),
        sample_family_losses=np.full((n_samples, n_families), np.nan),
        subject_ids=subject_ids,
        subject_null_losses=missing_subject,
        subject_full_losses=missing_subject.copy(),
        subject_family_losses=np.full((len(subject_ids), n_families), np.nan),
    )


def apply_incremental_downstream_functional(
    functional: IncrementalDownstreamFunctional,
    response_matrix: np.ndarray,
    *,
    row_manifest: DownstreamRowManifest,
    design_sample_ids: tuple[str, ...],
    nuisance_matrix: np.ndarray,
    context_regressor: np.ndarray,
    context_regressor_id: str,
    nuisance_design_id: str,
    feature_ids: tuple[str, ...],
    nuisance_column_ids: tuple[str, ...],
) -> IncrementalDownstreamApplication:
    """Compute held-out family gains relative to the frozen receiver-null model."""

    functional._require_intact()
    if context_regressor_id != functional.context_regressor_id:
        raise ValueError("test context_regressor_id must match the frozen functional")
    if nuisance_design_id != functional.nuisance_design_id:
        raise ValueError("test nuisance_design_id must match the frozen functional")
    if tuple(feature_ids) != functional.feature_ids:
        raise ValueError("test feature_ids must exactly match the frozen functional")
    if tuple(nuisance_column_ids) != functional.nuisance_column_ids:
        raise ValueError(
            "test nuisance columns must exactly match the frozen functional"
        )
    canonical = _canonical_incremental_rows(
        response_matrix,
        row_manifest=row_manifest,
        nuisance_matrix=nuisance_matrix,
        context_regressor=context_regressor,
        design_sample_ids=design_sample_ids,
        n_features=len(functional.feature_ids),
        n_nuisance=len(functional.nuisance_column_ids),
    )
    response = canonical.response
    nuisance = canonical.nuisance
    regressor = canonical.regressor
    n_samples = response.shape[0]
    if n_samples < 1:
        raise ValueError("test response_matrix must contain complete finite samples")
    heldout_subjects = set(canonical.subject_ids)
    sample_overlap = set(canonical.sample_ids).intersection(
        functional.training_sample_ids
    )
    if sample_overlap:
        raise ValueError(
            "heldout sample IDs overlap training samples: "
            + ", ".join(sorted(sample_overlap))
        )
    overlap = heldout_subjects.intersection(functional.training_subject_ids)
    if overlap:
        raise ValueError(
            "heldout subjects overlap training subjects: " + ", ".join(sorted(overlap))
        )
    heldout_input_digest = stable_id(
        "incremental_heldout_input",
        {
            "nuisance_digest": _dense_input_digest(nuisance),
            "nuisance_design_id": nuisance_design_id,
            "context_regressor_id": context_regressor_id,
            "regressor_digest": _dense_input_digest(regressor),
            "response_digest": _dense_input_digest(response),
            "row_manifest_id": row_manifest.manifest_id,
        },
        schema_version="2",
    )
    heldout_contexts = tuple(sorted(set(canonical.context_ids)))
    if heldout_contexts != functional.training_context_ids:
        return _not_estimable_incremental_application(
            functional,
            reason_code="heldout_contrast_context_universe_mismatch",
            canonical=canonical,
            heldout_row_manifest_id=row_manifest.manifest_id,
            heldout_input_digest=heldout_input_digest,
        )
    if float(np.ptp(regressor)) <= functional.null_loss_floor:
        return _not_estimable_incremental_application(
            functional,
            reason_code="heldout_context_regressor_lacks_variation",
            canonical=canonical,
            heldout_row_manifest_id=row_manifest.manifest_id,
            heldout_input_digest=heldout_input_digest,
        )
    standardized = (response - functional.feature_center) / functional.feature_scale
    null_prediction = nuisance @ functional.null_nuisance_coefficients
    sample_null_losses = _prediction_losses(
        standardized, null_prediction, functional.precision_weights
    )
    subject_ids, subject_null_losses, null_loss = _subject_context_prediction_losses(
        standardized,
        null_prediction,
        functional.precision_weights,
        canonical.subject_ids,
        canonical.context_ids,
    )
    n_families = len(functional.family_ids)
    if null_loss <= functional.null_loss_floor:
        return _not_estimable_incremental_application(
            functional,
            reason_code="heldout_receiver_null_loss_below_floor",
            canonical=canonical,
            heldout_row_manifest_id=row_manifest.manifest_id,
            heldout_input_digest=heldout_input_digest,
        )
    predicted_effect = np.asarray(
        functional.family_basis @ functional.family_coefficients
    ).ravel()
    full_prediction = nuisance @ functional.full_nuisance_coefficients + np.outer(
        regressor, predicted_effect
    )
    sample_full_losses = _prediction_losses(
        standardized, full_prediction, functional.precision_weights
    )
    full_subject_ids, subject_full_losses, full_loss = (
        _subject_context_prediction_losses(
            standardized,
            full_prediction,
            functional.precision_weights,
            canonical.subject_ids,
            canonical.context_ids,
        )
    )
    if full_subject_ids != subject_ids:
        raise RuntimeError("subject loss aggregation changed subject order")
    raw_model_gain = float((null_loss - full_loss) / null_loss)
    model_gain = float(np.clip(raw_model_gain, 0.0, 1.0))
    family_losses: np.ndarray = np.empty(n_families, dtype=np.float64)
    family_gains: np.ndarray = np.empty(n_families, dtype=np.float64)
    raw_family_gains: np.ndarray = np.empty(n_families, dtype=np.float64)
    sample_family_losses: np.ndarray = np.empty(
        (n_samples, n_families), dtype=np.float64
    )
    subject_family_losses: np.ndarray = np.empty(
        (len(subject_ids), n_families), dtype=np.float64
    )
    for family_index, coefficient in enumerate(functional.family_coefficients):
        contribution = (
            np.asarray(functional.family_basis.getcol(family_index).toarray()).ravel()
            * coefficient
        )
        prediction = nuisance @ functional.family_nuisance_coefficients[
            family_index
        ] + np.outer(regressor, contribution)
        sample_loss = _prediction_losses(
            standardized, prediction, functional.precision_weights
        )
        family_subject_ids, subject_loss, loss = _subject_context_prediction_losses(
            standardized,
            prediction,
            functional.precision_weights,
            canonical.subject_ids,
            canonical.context_ids,
        )
        if family_subject_ids != subject_ids:
            raise RuntimeError("family loss aggregation changed subject order")
        sample_family_losses[:, family_index] = sample_loss
        subject_family_losses[:, family_index] = subject_loss
        family_losses[family_index] = loss
        raw_family_gains[family_index] = (null_loss - loss) / null_loss
        family_gains[family_index] = np.clip(raw_family_gains[family_index], 0.0, 1.0)
    return IncrementalDownstreamApplication._from_application(
        functional=functional,
        status="observed",
        reason_code=None,
        sample_ids=canonical.sample_ids,
        sample_subject_ids=canonical.subject_ids,
        sample_context_ids=canonical.context_ids,
        heldout_row_manifest_id=row_manifest.manifest_id,
        heldout_input_digest=heldout_input_digest,
        null_loss=null_loss,
        full_loss=full_loss,
        model_gain=model_gain,
        raw_model_gain=raw_model_gain,
        family_losses=family_losses,
        family_gains=family_gains,
        raw_family_gains=raw_family_gains,
        sample_null_losses=sample_null_losses,
        sample_full_losses=sample_full_losses,
        sample_family_losses=sample_family_losses,
        subject_ids=subject_ids,
        subject_null_losses=subject_null_losses,
        subject_full_losses=subject_full_losses,
        subject_family_losses=subject_family_losses,
    )


def fit_downstream_functional(
    reference_expression: np.ndarray,
    *,
    receiver: str,
    contrast_name: str,
    fold_id: str,
    feature_ids: tuple[str, ...],
    family_ids: tuple[str, ...],
    reference_sample_ids: tuple[str, ...],
    reference_subject_ids: tuple[str, ...],
    training_subject_ids: tuple[str, ...],
    target_weight_matrix: sparse.spmatrix | np.ndarray,
    family_support: np.ndarray,
    minimum_scale: float = 0.25,
) -> DownstreamFunctional:
    """Fit median/MAD transforms using training reference samples only."""

    expression, samples, subjects, _ = _canonical_reference_input(
        reference_expression,
        sample_ids=reference_sample_ids,
        subject_ids=reference_subject_ids,
        feature_ids=feature_ids,
    )
    if expression.shape[0] < 2:
        raise ValueError(
            "reference_expression requires at least two complete finite samples"
        )
    declared_training_subjects = tuple(sorted(set(training_subject_ids)))
    if set(subjects).difference(declared_training_subjects):
        raise ValueError("reference subjects must belong to training_subject_ids")
    if not math.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("minimum_scale must be finite and positive")
    center = np.median(expression, axis=0)
    mad = np.median(np.abs(expression - center), axis=0)
    scale = np.maximum(_MAD_GAUSSIAN_CONSISTENCY * mad, minimum_scale)
    matrix = sparse.csc_matrix(target_weight_matrix, dtype=np.float64).copy()
    if matrix.shape != (len(feature_ids), len(family_ids)):
        raise ValueError("target_weight_matrix shape must equal features x families")
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError("target weights must be finite and non-negative")
    column_sums = np.asarray(matrix.sum(axis=0)).ravel()
    if np.any(column_sums <= 0):
        raise ValueError("every family must have positive target-weight mass")
    matrix = matrix @ sparse.diags(1.0 / column_sums, format="csc")
    return DownstreamFunctional(
        receiver=receiver,
        contrast_name=contrast_name,
        fold_id=fold_id,
        feature_ids=feature_ids,
        family_ids=family_ids,
        training_subject_ids=declared_training_subjects,
        reference_sample_ids=samples,
        feature_center=center,
        feature_scale=scale,
        target_weight_matrix=matrix,
        family_support=family_support,
        minimum_scale=minimum_scale,
        reference_expression=expression,
        reference_subject_ids=subjects,
    )


def apply_downstream_functional(
    functional: DownstreamFunctional,
    sample_expression: np.ndarray,
    *,
    feature_ids: tuple[str, ...],
) -> DownstreamApplication:
    """Apply a training-fold receiver-program transform without refitting."""

    if tuple(feature_ids) != functional.feature_ids:
        raise ValueError("test feature_ids must exactly match the frozen functional")
    expression = np.asarray(sample_expression, dtype=np.float64)
    if expression.ndim == 1:
        expression = expression.reshape(1, -1)
    if expression.ndim != 2 or expression.shape[1] != len(functional.feature_ids):
        raise ValueError("sample_expression must be samples x frozen features")
    if np.any(np.isinf(expression)):
        raise ValueError("sample_expression must not contain infinite values")
    standardized = (expression - functional.feature_center) / functional.feature_scale
    directional = np.maximum(standardized, 0.0)
    raw_program = np.asarray(
        directional @ functional.target_weight_matrix,
        dtype=np.float64,
    )
    program_score = raw_program / (1.0 + raw_program)
    supported = program_score * functional.family_support
    return DownstreamApplication(
        downstream_functional_id=functional.downstream_functional_id,
        raw_program=raw_program,
        receiver_program_score=program_score,
        supported_program_score=supported,
    )


__all__ = [
    "DownstreamApplication",
    "DownstreamFunctional",
    "DownstreamRowManifest",
    "IncrementalDownstreamApplication",
    "IncrementalDownstreamFunctional",
    "apply_downstream_functional",
    "apply_incremental_downstream_functional",
    "fit_downstream_functional",
    "fit_incremental_downstream_functional",
]
