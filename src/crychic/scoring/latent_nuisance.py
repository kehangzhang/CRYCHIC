"""Training-fold latent nuisance programs learned from control features only.

This module deliberately stops at a fold-frozen nuisance artifact.  It does not
claim that a learned component is a reviewed receiver-autonomous biological
program, and it does not certify calibration or held-out application.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from scipy import sparse

from crychic.core import ContractError, stable_id

_ALGORITHM_CONTRACT = "training_fold_control_only_weighted_pca_v1"
_PRODUCER_MARKER = "crychic.scoring.fold_latent_nuisance.v1"
_ARTIFACT_SCHEMA_VERSION = "1"

LatentNuisanceStatus = Literal["observed", "not_estimable"]


def _positive_integer(value: object, *, field_name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{field_name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return result


def _unit_interval(
    value: object, *, field_name: str, include_one: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    upper_valid = result <= 1.0 if include_one else result < 1.0
    if not math.isfinite(result) or result <= 0.0 or not upper_valid:
        qualifier = "(0, 1]" if include_one else "(0, 1)"
        raise ValueError(f"{field_name} must be in {qualifier}")
    return result


def _identifier(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _unique_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(_identifier(value, field_name=field_name) for value in values)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique values")
    return result


def _aligned_names(
    values: Sequence[str], *, length: int, field_name: str
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(_identifier(value, field_name=field_name) for value in values)
    if len(result) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    return result


def _immutable_float64(values: np.ndarray) -> np.ndarray:
    canonical = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if np.any(~np.isfinite(canonical)):
        raise ValueError("latent nuisance arrays must contain only finite values")
    canonical[canonical == 0.0] = 0.0
    result = cast(
        np.ndarray,
        np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(
            canonical.shape
        ),
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


def _array_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if np.any(~np.isfinite(canonical)):
        raise ValueError("array digest requires finite values")
    canonical[canonical == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_family_basis(
    values: sparse.spmatrix | np.ndarray,
    *,
    n_features: int,
) -> sparse.csc_matrix:
    result = sparse.csc_matrix(values, dtype="<f8", copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if result.ndim != 2 or result.shape[0] != n_features or result.shape[1] < 1:
        raise ValueError("family_basis must have shape features x non-empty families")
    if np.any(~np.isfinite(result.data)):
        raise ValueError("family_basis must contain only finite values")
    result.data[result.data == 0.0] = 0.0
    return result


def _sparse_digest(values: sparse.csc_matrix) -> str:
    matrix = sparse.csc_matrix(values, dtype="<f8", copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    return digest.hexdigest()


def _weighted_rank(values: np.ndarray, *, rcond: float) -> tuple[int, np.ndarray]:
    singular_values = np.linalg.svd(values, compute_uv=False, full_matrices=False)
    if not singular_values.size or singular_values[0] <= 0.0:
        return 0, singular_values
    threshold = rcond * float(singular_values[0])
    return int(np.count_nonzero(singular_values > threshold)), singular_values


def _weighted_span_residual(
    values: np.ndarray,
    *,
    span_basis: np.ndarray,
    precision: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, int]:
    """Remove a feature-space span in the precision-weighted geometry."""

    observed = np.asarray(values, dtype=np.float64)
    span = np.asarray(span_basis, dtype=np.float64)
    weights = np.asarray(precision, dtype=np.float64)
    if observed.ndim != 2 or span.ndim != 2:
        raise ValueError("weighted span inputs must be two-dimensional")
    if observed.shape[0] != span.shape[0] or weights.shape != (observed.shape[0],):
        raise ValueError("weighted span inputs must share one feature axis")
    sqrt_precision = np.sqrt(weights)
    weighted_span = sqrt_precision[:, np.newaxis] * span
    left, singular_values, _ = np.linalg.svd(weighted_span, full_matrices=False)
    rank = 0
    if singular_values.size and singular_values[0] > 0.0:
        rank = int(
            np.count_nonzero(singular_values > rcond * float(singular_values[0]))
        )
    weighted_values = sqrt_precision[:, np.newaxis] * observed
    if rank:
        orthonormal_span = left[:, :rank]
        # A second projection suppresses residual round-off for ill-scaled spans.
        for _ in range(2):
            weighted_values -= orthonormal_span @ (
                orthonormal_span.T @ weighted_values
            )
    result = np.zeros_like(observed)
    supported = sqrt_precision > 0.0
    result[supported] = (
        weighted_values[supported] / sqrt_precision[supported, np.newaxis]
    )
    return result, rank


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenLatentNuisanceSpec:
    """Immutable control-only factor-learning procedure for one training fold."""

    max_components: int = 5
    min_control_features: int = 20
    min_training_subjects: int = 4
    svd_rcond: float = 1e-10
    minimum_explained_fraction: float = 0.05
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        max_components = _positive_integer(
            self.max_components, field_name="max_components"
        )
        min_controls = _positive_integer(
            self.min_control_features,
            field_name="min_control_features",
            minimum=2,
        )
        min_subjects = _positive_integer(
            self.min_training_subjects,
            field_name="min_training_subjects",
            minimum=2,
        )
        rcond = _unit_interval(self.svd_rcond, field_name="svd_rcond")
        explained = _unit_interval(
            self.minimum_explained_fraction,
            field_name="minimum_explained_fraction",
            include_one=True,
        )
        values = {
            "max_components": max_components,
            "min_control_features": min_controls,
            "min_training_subjects": min_subjects,
            "svd_rcond": rcond,
            "minimum_explained_fraction": explained,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "latent_nuisance_spec",
                {**values, "algorithm_contract": _ALGORITHM_CONTRACT},
                schema_version="1",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_contract": _ALGORITHM_CONTRACT,
            "max_components": self.max_components,
            "min_control_features": self.min_control_features,
            "min_training_subjects": self.min_training_subjects,
            "svd_rcond": self.svd_rcond,
            "minimum_explained_fraction": self.minimum_explained_fraction,
        }

    def _require_intact(self) -> None:
        try:
            repeated = FrozenLatentNuisanceSpec(
                max_components=self.max_components,
                min_control_features=self.min_control_features,
                min_training_subjects=self.min_training_subjects,
                svd_rcond=self.svd_rcond,
                minimum_explained_fraction=self.minimum_explained_fraction,
            )
            valid = repeated.spec_id == self.spec_id
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Latent nuisance specification failed integrity validation",
                code="latent_nuisance_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the frozen latent nuisance specification",
            ) from error
        if not valid:
            raise ContractError(
                "Latent nuisance specification failed integrity validation",
                code="latent_nuisance_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the frozen latent nuisance specification",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, init=False)
class FoldLatentNuisanceArtifact:
    """Producer-owned latent nuisance basis learned from one training fold."""

    receiver: str
    contrast_name: str
    fold_id: str
    status: LatentNuisanceStatus
    reason_code: str | None
    spec: FrozenLatentNuisanceSpec
    spec_id: str
    feature_ids: tuple[str, ...]
    control_feature_ids: tuple[str, ...]
    training_sample_ids: tuple[str, ...]
    training_sample_subject_ids: tuple[str, ...]
    training_sample_context_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    training_row_manifest_id: str
    program_ids: tuple[str, ...]
    coordinate_basis: np.ndarray
    factor_scores: np.ndarray
    control_score_basis: np.ndarray
    static_coordinate_basis: np.ndarray
    precision_weights: np.ndarray
    static_coordinate_basis_id: str | None
    static_program_ids: tuple[str, ...]
    response_digest: str
    family_basis_digest: str
    nuisance_digest: str
    training_weights_digest: str
    feature_center_digest: str
    feature_scale_digest: str
    precision_weights_digest: str
    static_coordinate_basis_digest: str
    training_input_digest: str
    control_score_input_digest: str | None
    control_residual_digest: str | None
    coordinate_basis_digest: str
    factor_scores_digest: str
    control_score_basis_digest: str
    singular_values: np.ndarray
    explained_variance_fractions: np.ndarray
    retained_explained_fraction: float
    control_numerical_rank: int
    static_numerical_rank: int
    static_control_numerical_rank: int
    static_orthogonality_max_abs: float
    family_count: int
    nuisance_column_count: int
    artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FoldLatentNuisanceArtifact is producer-owned; use "
            "fit_fold_latent_nuisance()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_contract": _ALGORITHM_CONTRACT,
            "receiver": self.receiver,
            "contrast_name": self.contrast_name,
            "fold_id": self.fold_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "spec_id": self.spec_id,
            "feature_ids": list(self.feature_ids),
            "control_feature_ids": list(self.control_feature_ids),
            "training_sample_ids": list(self.training_sample_ids),
            "training_sample_subject_ids": list(
                self.training_sample_subject_ids
            ),
            "training_sample_context_ids": list(
                self.training_sample_context_ids
            ),
            "training_subject_ids": list(self.training_subject_ids),
            "training_row_manifest_id": self.training_row_manifest_id,
            "program_ids": list(self.program_ids),
            "static_coordinate_basis_id": self.static_coordinate_basis_id,
            "static_program_ids": list(self.static_program_ids),
            "response_digest": self.response_digest,
            "family_basis_digest": self.family_basis_digest,
            "nuisance_digest": self.nuisance_digest,
            "training_weights_digest": self.training_weights_digest,
            "feature_center_digest": self.feature_center_digest,
            "feature_scale_digest": self.feature_scale_digest,
            "precision_weights_digest": self.precision_weights_digest,
            "static_coordinate_basis_digest": (
                self.static_coordinate_basis_digest
            ),
            "training_input_digest": self.training_input_digest,
            "control_score_input_digest": self.control_score_input_digest,
            "control_residual_digest": self.control_residual_digest,
            "coordinate_basis_digest": self.coordinate_basis_digest,
            "factor_scores_digest": self.factor_scores_digest,
            "control_score_basis_digest": self.control_score_basis_digest,
            "singular_values_digest": _array_digest(self.singular_values),
            "explained_variance_fractions_digest": _array_digest(
                self.explained_variance_fractions
            ),
            "retained_explained_fraction": self.retained_explained_fraction,
            "control_numerical_rank": self.control_numerical_rank,
            "static_numerical_rank": self.static_numerical_rank,
            "static_control_numerical_rank": self.static_control_numerical_rank,
            "static_orthogonality_max_abs": self.static_orthogonality_max_abs,
            "family_count": self.family_count,
            "nuisance_column_count": self.nuisance_column_count,
        }

    def _require_producer_owned(self) -> None:
        if self._producer_marker != _PRODUCER_MARKER:
            raise TypeError("latent nuisance artifact is not producer-owned")
        self._require_intact()

    def _require_intact(self) -> None:
        try:
            self.spec._require_intact()
            n_rows = len(self.training_sample_ids)
            n_features = len(self.feature_ids)
            n_controls = len(self.control_feature_ids)
            n_programs = len(self.program_ids)
            arrays = (
                self.coordinate_basis,
                self.factor_scores,
                self.control_score_basis,
                self.static_coordinate_basis,
                self.precision_weights,
                self.singular_values,
                self.explained_variance_fractions,
            )
            valid = all(_is_immutable_byte_backed(value) for value in arrays)
            valid = valid and self.spec_id == self.spec.spec_id
            valid = valid and len(set(self.feature_ids)) == n_features
            valid = valid and len(set(self.control_feature_ids)) == n_controls
            valid = valid and set(self.control_feature_ids).issubset(
                self.feature_ids
            )
            valid = valid and len(set(self.program_ids)) == n_programs
            valid = valid and len(self.training_sample_subject_ids) == n_rows
            valid = valid and len(self.training_sample_context_ids) == n_rows
            valid = valid and len(set(self.training_sample_ids)) == n_rows
            valid = valid and self.training_subject_ids == tuple(
                sorted(set(self.training_sample_subject_ids))
            )
            valid = valid and self.coordinate_basis.shape == (
                n_features,
                n_programs,
            )
            valid = valid and self.factor_scores.shape == (n_rows, n_programs)
            valid = valid and self.control_score_basis.shape == (
                n_controls,
                n_programs,
            )
            valid = valid and self.static_coordinate_basis.shape == (
                n_features,
                len(self.static_program_ids),
            )
            valid = valid and self.precision_weights.shape == (n_features,)
            valid = valid and bool(np.all(self.precision_weights >= 0.0))
            valid = valid and bool(np.any(self.precision_weights > 0.0))
            valid = valid and self.singular_values.ndim == 1
            valid = valid and self.explained_variance_fractions.shape == (
                self.singular_values.size,
            )
            valid = valid and bool(np.all(self.singular_values >= 0.0))
            valid = valid and bool(
                np.all(self.explained_variance_fractions >= 0.0)
            )
            valid = valid and 0 <= self.control_numerical_rank <= (
                self.singular_values.size
            )
            valid = valid and self.coordinate_basis_digest == _array_digest(
                self.coordinate_basis
            )
            valid = valid and self.factor_scores_digest == _array_digest(
                self.factor_scores
            )
            valid = valid and self.control_score_basis_digest == _array_digest(
                self.control_score_basis
            )
            valid = valid and self.static_coordinate_basis_digest == _array_digest(
                self.static_coordinate_basis
            )
            valid = valid and self.precision_weights_digest == _array_digest(
                self.precision_weights
            )
            valid = valid and (
                (
                    self.status == "observed"
                    and self.reason_code is None
                    and n_programs > 0
                )
                or (
                    self.status == "not_estimable"
                    and isinstance(self.reason_code, str)
                    and bool(self.reason_code)
                    and n_programs == 0
                )
            )
            valid = valid and (self.static_coordinate_basis_id is None) == (
                not self.static_program_ids
            )
            valid = valid and self.training_row_manifest_id == _row_manifest_id(
                self.training_sample_ids,
                self.training_sample_subject_ids,
                self.training_sample_context_ids,
            )
            valid = valid and self.training_input_digest == _training_input_digest(
                row_manifest_id=self.training_row_manifest_id,
                feature_ids=self.feature_ids,
                response_digest=self.response_digest,
                family_basis_digest=self.family_basis_digest,
                nuisance_digest=self.nuisance_digest,
                training_weights_digest=self.training_weights_digest,
                feature_center_digest=self.feature_center_digest,
                feature_scale_digest=self.feature_scale_digest,
                precision_weights_digest=self.precision_weights_digest,
                static_coordinate_basis_digest=(
                    self.static_coordinate_basis_digest
                ),
                static_coordinate_basis_id=self.static_coordinate_basis_id,
                static_program_ids=self.static_program_ids,
                family_count=self.family_count,
                nuisance_column_count=self.nuisance_column_count,
            )
            valid = valid and self.artifact_id == stable_id(
                "fold_latent_nuisance_artifact",
                self._identity_payload(),
                schema_version=_ARTIFACT_SCHEMA_VERSION,
            )
            if valid and self.status == "observed":
                expected_program_ids = tuple(
                    stable_id(
                        "latent_nuisance_program",
                        {
                            "component_index": component,
                            "control_score_input_digest": (
                                self.control_score_input_digest
                            ),
                            "contrast_name": self.contrast_name,
                            "fold_id": self.fold_id,
                            "receiver": self.receiver,
                            "spec_id": self.spec_id,
                        },
                        schema_version="1",
                    )
                    for component in range(n_programs)
                )
                valid = valid and self.program_ids == expected_program_ids
                valid = valid and math.isclose(
                    self.retained_explained_fraction,
                    float(
                        np.sum(
                            self.explained_variance_fractions[:n_programs]
                        )
                    ),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                weighted_basis = (
                    np.sqrt(self.precision_weights)[:, np.newaxis]
                    * self.coordinate_basis
                )
                precision_norms = np.linalg.norm(weighted_basis, axis=0)
                valid = bool(np.all(precision_norms > 0.0))
                rank, _ = _weighted_rank(
                    weighted_basis,
                    rcond=self.spec.svd_rcond,
                )
                valid = valid and rank == n_programs
                if self.static_program_ids:
                    static_weighted = (
                        np.sqrt(self.precision_weights)[:, np.newaxis]
                        * self.static_coordinate_basis
                    )
                    static_rank, _ = _weighted_rank(
                        static_weighted,
                        rcond=self.spec.svd_rcond,
                    )
                    cross_product = self.static_coordinate_basis.T @ (
                        self.precision_weights[:, np.newaxis]
                        * self.coordinate_basis
                    )
                    observed_orthogonality = float(np.max(np.abs(cross_product)))
                    valid = valid and static_rank == self.static_numerical_rank
                    valid = valid and math.isclose(
                        observed_orthogonality,
                        self.static_orthogonality_max_abs,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                else:
                    valid = valid and self.static_orthogonality_max_abs == 0.0
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Fold latent nuisance artifact failed integrity validation",
                code="fold_latent_nuisance_integrity_violation",
                field="artifact_id",
                remediation="Refit the artifact from intact training-fold inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Fold latent nuisance artifact failed integrity validation",
                code="fold_latent_nuisance_integrity_violation",
                field="artifact_id",
                remediation="Refit the artifact from intact training-fold inputs",
            )

    def to_dict(self) -> dict[str, object]:
        """Return frozen lineage and numerical diagnostics, without release claims."""

        self._require_producer_owned()
        return {"artifact_id": self.artifact_id, **self._identity_payload()}


def _row_manifest_id(
    sample_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> str:
    result: str = stable_id(
        "latent_nuisance_training_rows",
        {
            "rows": [
                {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "context_id": context_id,
                }
                for sample_id, subject_id, context_id in zip(
                    sample_ids, subject_ids, context_ids, strict=True
                )
            ]
        },
        schema_version="1",
    )
    return result


def _training_input_digest(
    *,
    row_manifest_id: str,
    feature_ids: tuple[str, ...],
    response_digest: str,
    family_basis_digest: str,
    nuisance_digest: str,
    training_weights_digest: str,
    feature_center_digest: str,
    feature_scale_digest: str,
    precision_weights_digest: str,
    static_coordinate_basis_digest: str,
    static_coordinate_basis_id: str | None,
    static_program_ids: tuple[str, ...],
    family_count: int,
    nuisance_column_count: int,
) -> str:
    result: str = stable_id(
        "latent_nuisance_training_input",
        {
            "row_manifest_id": row_manifest_id,
            "feature_ids": list(feature_ids),
            "response_digest": response_digest,
            "family_basis_digest": family_basis_digest,
            "nuisance_digest": nuisance_digest,
            "training_weights_digest": training_weights_digest,
            "feature_center_digest": feature_center_digest,
            "feature_scale_digest": feature_scale_digest,
            "precision_weights_digest": precision_weights_digest,
            "static_coordinate_basis_digest": static_coordinate_basis_digest,
            "static_coordinate_basis_id": static_coordinate_basis_id,
            "static_program_ids": list(static_program_ids),
            "family_count": family_count,
            "nuisance_column_count": nuisance_column_count,
        },
        schema_version="1",
        digest_length=64,
    )
    return result


def _build_artifact(
    *,
    receiver: str,
    contrast_name: str,
    fold_id: str,
    status: LatentNuisanceStatus,
    reason_code: str | None,
    spec: FrozenLatentNuisanceSpec,
    feature_ids: tuple[str, ...],
    control_feature_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
    row_manifest_id: str,
    program_ids: tuple[str, ...],
    coordinate_basis: np.ndarray,
    factor_scores: np.ndarray,
    control_score_basis: np.ndarray,
    static_coordinate_basis: np.ndarray,
    precision_weights: np.ndarray,
    static_coordinate_basis_id: str | None,
    static_program_ids: tuple[str, ...],
    response_digest: str,
    family_basis_digest: str,
    nuisance_digest: str,
    training_weights_digest: str,
    feature_center_digest: str,
    feature_scale_digest: str,
    precision_weights_digest: str,
    training_input_digest: str,
    control_score_input_digest: str | None,
    control_residual_digest: str | None,
    singular_values: np.ndarray,
    explained_variance_fractions: np.ndarray,
    retained_explained_fraction: float,
    control_numerical_rank: int,
    static_numerical_rank: int,
    static_control_numerical_rank: int,
    static_orthogonality_max_abs: float,
    family_count: int,
    nuisance_column_count: int,
) -> FoldLatentNuisanceArtifact:
    self = object.__new__(FoldLatentNuisanceArtifact)
    frozen_coordinate = _immutable_float64(coordinate_basis)
    frozen_scores = _immutable_float64(factor_scores)
    frozen_control_basis = _immutable_float64(control_score_basis)
    frozen_static = _immutable_float64(static_coordinate_basis)
    frozen_precision = _immutable_float64(precision_weights)
    frozen_singular = _immutable_float64(singular_values)
    frozen_explained = _immutable_float64(explained_variance_fractions)
    values: dict[str, Any] = {
        "receiver": receiver,
        "contrast_name": contrast_name,
        "fold_id": fold_id,
        "status": status,
        "reason_code": reason_code,
        "spec": spec,
        "spec_id": spec.spec_id,
        "feature_ids": feature_ids,
        "control_feature_ids": control_feature_ids,
        "training_sample_ids": sample_ids,
        "training_sample_subject_ids": subject_ids,
        "training_sample_context_ids": context_ids,
        "training_subject_ids": tuple(sorted(set(subject_ids))),
        "training_row_manifest_id": row_manifest_id,
        "program_ids": program_ids,
        "coordinate_basis": frozen_coordinate,
        "factor_scores": frozen_scores,
        "control_score_basis": frozen_control_basis,
        "static_coordinate_basis": frozen_static,
        "precision_weights": frozen_precision,
        "static_coordinate_basis_id": static_coordinate_basis_id,
        "static_program_ids": static_program_ids,
        "response_digest": response_digest,
        "family_basis_digest": family_basis_digest,
        "nuisance_digest": nuisance_digest,
        "training_weights_digest": training_weights_digest,
        "feature_center_digest": feature_center_digest,
        "feature_scale_digest": feature_scale_digest,
        "precision_weights_digest": precision_weights_digest,
        "static_coordinate_basis_digest": _array_digest(frozen_static),
        "training_input_digest": training_input_digest,
        "control_score_input_digest": control_score_input_digest,
        "control_residual_digest": control_residual_digest,
        "coordinate_basis_digest": _array_digest(frozen_coordinate),
        "factor_scores_digest": _array_digest(frozen_scores),
        "control_score_basis_digest": _array_digest(frozen_control_basis),
        "singular_values": frozen_singular,
        "explained_variance_fractions": frozen_explained,
        "retained_explained_fraction": float(retained_explained_fraction),
        "control_numerical_rank": int(control_numerical_rank),
        "static_numerical_rank": int(static_numerical_rank),
        "static_control_numerical_rank": int(static_control_numerical_rank),
        "static_orthogonality_max_abs": float(static_orthogonality_max_abs),
        "family_count": family_count,
        "nuisance_column_count": nuisance_column_count,
        "_producer_marker": _PRODUCER_MARKER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "artifact_id",
        stable_id(
            "fold_latent_nuisance_artifact",
            self._identity_payload(),
            schema_version=_ARTIFACT_SCHEMA_VERSION,
        ),
    )
    self._require_producer_owned()
    return self


def fit_fold_latent_nuisance(
    response_matrix: np.ndarray,
    *,
    feature_ids: Sequence[str],
    family_basis: sparse.spmatrix | np.ndarray,
    nuisance_matrix: np.ndarray,
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    context_ids: Sequence[str],
    training_weights: np.ndarray,
    feature_center: np.ndarray,
    feature_scale: np.ndarray,
    precision_weights: np.ndarray,
    fold_id: str,
    receiver: str,
    contrast_name: str,
    spec: FrozenLatentNuisanceSpec,
    static_coordinate_basis: np.ndarray | None = None,
    static_coordinate_basis_id: str | None = None,
    static_program_ids: Sequence[str] = (),
) -> FoldLatentNuisanceArtifact:
    """Learn latent nuisance coordinates without consulting held-out rows.

    The PCA score map is learned exclusively from precision-supported features
    with exactly zero support in ``family_basis``.  Responses from family-targeted
    features are consulted only after those scores are frozen, when the producer
    estimates the full feature-by-component coordinate basis.
    """

    if not isinstance(spec, FrozenLatentNuisanceSpec):
        raise TypeError("spec must be a FrozenLatentNuisanceSpec")
    spec._require_intact()
    normalized_receiver = _identifier(receiver, field_name="receiver")
    normalized_contrast = _identifier(contrast_name, field_name="contrast_name")
    normalized_fold = _identifier(fold_id, field_name="fold_id")
    features = _unique_names(feature_ids, field_name="feature_ids")
    n_features = len(features)
    response = np.asarray(response_matrix, dtype="<f8")
    if response.ndim != 2 or response.shape[1] != n_features:
        raise ValueError("response_matrix must have shape samples x features")
    if response.shape[0] < 1 or np.any(~np.isfinite(response)):
        raise ValueError("response_matrix must be non-empty and finite")
    n_rows = response.shape[0]
    samples = _aligned_names(sample_ids, length=n_rows, field_name="sample_ids")
    if len(set(samples)) != n_rows:
        raise ValueError("sample_ids must be unique")
    subjects = _aligned_names(subject_ids, length=n_rows, field_name="subject_ids")
    contexts = _aligned_names(context_ids, length=n_rows, field_name="context_ids")
    nuisance = np.asarray(nuisance_matrix, dtype="<f8")
    if (
        nuisance.ndim != 2
        or nuisance.shape[0] != n_rows
        or np.any(~np.isfinite(nuisance))
    ):
        raise ValueError("nuisance_matrix must be a finite samples x columns matrix")
    weights = np.asarray(training_weights, dtype="<f8")
    center = np.asarray(feature_center, dtype="<f8")
    scale = np.asarray(feature_scale, dtype="<f8")
    precision = np.asarray(precision_weights, dtype="<f8")
    if (
        weights.shape != (n_rows,)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("training_weights must be finite and strictly positive")
    for values, name in (
        (center, "feature_center"),
        (scale, "feature_scale"),
        (precision, "precision_weights"),
    ):
        if values.shape != (n_features,) or np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must be a finite feature-aligned vector")
    if np.any(scale <= 0.0):
        raise ValueError("feature_scale must be strictly positive")
    if np.any(precision < 0.0) or not np.any(precision > 0.0):
        raise ValueError(
            "precision_weights must be non-negative with positive feature support"
        )
    families = _canonical_family_basis(family_basis, n_features=n_features)

    supplied_static_ids = tuple(static_program_ids)
    static_basis: np.ndarray
    static_id: str | None
    if static_coordinate_basis is None:
        if static_coordinate_basis_id is not None or supplied_static_ids:
            raise ValueError(
                "static basis ID/program IDs require static_coordinate_basis"
            )
        static_id = None
        static_ids: tuple[str, ...] = ()
        static_basis = np.empty((n_features, 0), dtype="<f8")
    else:
        static_basis = np.asarray(static_coordinate_basis, dtype="<f8")
        if (
            static_basis.ndim != 2
            or static_basis.shape[0] != n_features
            or static_basis.shape[1] < 1
            or np.any(~np.isfinite(static_basis))
        ):
            raise ValueError(
                "static_coordinate_basis must be finite features x programs"
            )
        static_id = _identifier(
            static_coordinate_basis_id,
            field_name="static_coordinate_basis_id",
        )
        static_ids = _unique_names(
            supplied_static_ids, field_name="static_program_ids"
        )
        if len(static_ids) != static_basis.shape[1]:
            raise ValueError(
                "static_program_ids must align with static basis columns"
            )

    order = np.asarray(sorted(range(n_rows), key=samples.__getitem__), dtype=int)
    response = np.asarray(response[order], dtype="<f8", order="C")
    nuisance = np.asarray(nuisance[order], dtype="<f8", order="C")
    weights = np.asarray(weights[order], dtype="<f8", order="C")
    samples = tuple(samples[index] for index in order)
    subjects = tuple(subjects[index] for index in order)
    contexts = tuple(contexts[index] for index in order)
    row_manifest = _row_manifest_id(samples, subjects, contexts)

    response_digest = _array_digest(response)
    family_digest = _sparse_digest(families)
    nuisance_digest = _array_digest(nuisance)
    weight_digest = _array_digest(weights)
    center_digest = _array_digest(center)
    scale_digest = _array_digest(scale)
    precision_digest = _array_digest(precision)
    static_digest = _array_digest(static_basis)
    input_digest = _training_input_digest(
        row_manifest_id=row_manifest,
        feature_ids=features,
        response_digest=response_digest,
        family_basis_digest=family_digest,
        nuisance_digest=nuisance_digest,
        training_weights_digest=weight_digest,
        feature_center_digest=center_digest,
        feature_scale_digest=scale_digest,
        precision_weights_digest=precision_digest,
        static_coordinate_basis_digest=static_digest,
        static_coordinate_basis_id=static_id,
        static_program_ids=static_ids,
        family_count=families.shape[1],
        nuisance_column_count=nuisance.shape[1],
    )
    zero_support = np.asarray(families.getnnz(axis=1)).ravel() == 0
    control_mask = zero_support & (precision > 0.0)
    control_indices = np.flatnonzero(control_mask)
    control_ids = tuple(features[index] for index in control_indices)

    empty_coordinate: np.ndarray = np.empty((n_features, 0), dtype="<f8")
    empty_scores: np.ndarray = np.empty((n_rows, 0), dtype="<f8")
    empty_control_basis: np.ndarray = np.empty(
        (len(control_ids), 0), dtype="<f8"
    )
    empty_diagnostic: np.ndarray = np.empty(0, dtype="<f8")

    def not_estimable(
        reason_code: str,
        *,
        singular_values: np.ndarray = empty_diagnostic,
        explained_fractions: np.ndarray = empty_diagnostic,
        control_rank: int = 0,
        static_rank: int = 0,
        static_control_rank: int = 0,
        control_input_digest: str | None = None,
        control_residual_digest: str | None = None,
    ) -> FoldLatentNuisanceArtifact:
        return _build_artifact(
            receiver=normalized_receiver,
            contrast_name=normalized_contrast,
            fold_id=normalized_fold,
            status="not_estimable",
            reason_code=reason_code,
            spec=spec,
            feature_ids=features,
            control_feature_ids=control_ids,
            sample_ids=samples,
            subject_ids=subjects,
            context_ids=contexts,
            row_manifest_id=row_manifest,
            program_ids=(),
            coordinate_basis=empty_coordinate,
            factor_scores=empty_scores,
            control_score_basis=empty_control_basis,
            static_coordinate_basis=static_basis,
            precision_weights=precision,
            static_coordinate_basis_id=static_id,
            static_program_ids=static_ids,
            response_digest=response_digest,
            family_basis_digest=family_digest,
            nuisance_digest=nuisance_digest,
            training_weights_digest=weight_digest,
            feature_center_digest=center_digest,
            feature_scale_digest=scale_digest,
            precision_weights_digest=precision_digest,
            training_input_digest=input_digest,
            control_score_input_digest=control_input_digest,
            control_residual_digest=control_residual_digest,
            singular_values=singular_values,
            explained_variance_fractions=explained_fractions,
            retained_explained_fraction=0.0,
            control_numerical_rank=control_rank,
            static_numerical_rank=static_rank,
            static_control_numerical_rank=static_control_rank,
            static_orthogonality_max_abs=0.0,
            family_count=families.shape[1],
            nuisance_column_count=nuisance.shape[1],
        )

    if len(set(subjects)) < spec.min_training_subjects:
        return not_estimable("latent_nuisance_insufficient_training_subjects")
    if len(control_ids) < spec.min_control_features:
        return not_estimable("latent_nuisance_insufficient_control_features")

    sqrt_weights = np.sqrt(weights)
    weighted_nuisance = sqrt_weights[:, np.newaxis] * nuisance
    nuisance_rank, _ = _weighted_rank(
        weighted_nuisance, rcond=spec.svd_rcond
    )
    if nuisance_rank != nuisance.shape[1]:
        return not_estimable("latent_nuisance_formula_rank_not_estimable")
    standardized = (response - center[np.newaxis, :]) / scale[np.newaxis, :]
    if nuisance.shape[1]:
        coefficients = np.linalg.lstsq(
            weighted_nuisance,
            sqrt_weights[:, np.newaxis] * standardized,
            rcond=spec.svd_rcond,
        )[0]
        residual = standardized - nuisance @ coefficients
    else:
        residual = standardized.copy()

    if static_basis.shape[1]:
        full_projected, static_rank = _weighted_span_residual(
            residual.T,
            span_basis=static_basis,
            precision=precision,
            rcond=spec.svd_rcond,
        )
        if static_rank != static_basis.shape[1]:
            return not_estimable(
                "latent_nuisance_static_basis_rank_not_estimable",
                static_rank=static_rank,
            )
        full_residual = full_projected.T
        control_projected, static_control_rank = _weighted_span_residual(
            residual[:, control_indices].T,
            span_basis=static_basis[control_indices],
            precision=precision[control_indices],
            rcond=spec.svd_rcond,
        )
        control_residual = control_projected.T
    else:
        static_rank = 0
        static_control_rank = 0
        full_residual = residual
        control_residual = residual[:, control_indices]

    weighted_mean = np.sum(
        weights[:, np.newaxis] * control_residual, axis=0
    ) / float(np.sum(weights))
    centered_controls = control_residual - weighted_mean[np.newaxis, :]
    control_residual_digest = _array_digest(centered_controls)
    control_input_digest = stable_id(
        "latent_nuisance_control_score_input",
        {
            "algorithm_contract": _ALGORITHM_CONTRACT,
            "training_row_manifest_id": row_manifest,
            "control_feature_ids": list(control_ids),
            "control_residual_digest": control_residual_digest,
            "control_precision_digest": _array_digest(precision[control_indices]),
            "training_weights_digest": weight_digest,
            "static_coordinate_basis_id": static_id,
            "static_control_basis_digest": _array_digest(
                static_basis[control_indices]
            ),
            "spec_id": spec.spec_id,
        },
        schema_version="1",
        digest_length=64,
    )
    weighted_controls = (
        sqrt_weights[:, np.newaxis]
        * centered_controls
        * np.sqrt(precision[control_indices])[np.newaxis, :]
    )
    left, singular_values, right = np.linalg.svd(
        weighted_controls, full_matrices=False
    )
    del left
    control_rank, _ = _weighted_rank(
        weighted_controls, rcond=spec.svd_rcond
    )
    variance = singular_values * singular_values
    total_variance = float(np.sum(variance))
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        fractions = np.zeros_like(singular_values)
        return not_estimable(
            "latent_nuisance_control_variance_not_estimable",
            singular_values=singular_values,
            explained_fractions=fractions,
            control_rank=control_rank,
            static_rank=static_rank,
            static_control_rank=static_control_rank,
            control_input_digest=control_input_digest,
            control_residual_digest=control_residual_digest,
        )
    fractions = variance / total_variance
    eligible_count = int(
        np.count_nonzero(
            fractions[:control_rank] >= spec.minimum_explained_fraction
        )
    )
    n_components = min(spec.max_components, eligible_count)
    if n_components < 1:
        return not_estimable(
            "latent_nuisance_explained_fraction_not_estimable",
            singular_values=singular_values,
            explained_fractions=fractions,
            control_rank=control_rank,
            static_rank=static_rank,
            static_control_rank=static_control_rank,
            control_input_digest=control_input_digest,
            control_residual_digest=control_residual_digest,
        )

    loadings = right[:n_components].T.copy()
    for component in range(n_components):
        absolute = np.abs(loadings[:, component])
        maximum = float(np.max(absolute))
        tied = np.flatnonzero(absolute == maximum)
        anchor = min(tied, key=lambda index: control_ids[int(index)])
        if loadings[anchor, component] < 0.0:
            loadings[:, component] *= -1.0
    control_score_basis = (
        np.sqrt(precision[control_indices])[:, np.newaxis] * loadings
    )
    factor_scores = centered_controls @ control_score_basis
    weighted_scores = sqrt_weights[:, np.newaxis] * factor_scores
    score_rank, _ = _weighted_rank(weighted_scores, rcond=spec.svd_rcond)
    if score_rank != n_components:
        return not_estimable(
            "latent_nuisance_factor_score_rank_not_estimable",
            singular_values=singular_values,
            explained_fractions=fractions,
            control_rank=control_rank,
            static_rank=static_rank,
            static_control_rank=static_control_rank,
            control_input_digest=control_input_digest,
            control_residual_digest=control_residual_digest,
        )

    feature_coefficients = np.linalg.lstsq(
        weighted_scores,
        sqrt_weights[:, np.newaxis] * full_residual,
        rcond=spec.svd_rcond,
    )[0].T
    if static_basis.shape[1]:
        feature_coefficients, repeated_static_rank = _weighted_span_residual(
            feature_coefficients,
            span_basis=static_basis,
            precision=precision,
            rcond=spec.svd_rcond,
        )
        if repeated_static_rank != static_rank:
            return not_estimable(
                "latent_nuisance_static_basis_rank_not_estimable",
                singular_values=singular_values,
                explained_fractions=fractions,
                control_rank=control_rank,
                static_rank=repeated_static_rank,
                static_control_rank=static_control_rank,
                control_input_digest=control_input_digest,
                control_residual_digest=control_residual_digest,
            )
    feature_coefficients[precision <= 0.0] = 0.0
    weighted_basis = np.sqrt(precision)[:, np.newaxis] * feature_coefficients
    basis_rank, _ = _weighted_rank(weighted_basis, rcond=spec.svd_rcond)
    if basis_rank != n_components:
        return not_estimable(
            "latent_nuisance_coordinate_basis_rank_not_estimable",
            singular_values=singular_values,
            explained_fractions=fractions,
            control_rank=control_rank,
            static_rank=static_rank,
            static_control_rank=static_control_rank,
            control_input_digest=control_input_digest,
            control_residual_digest=control_residual_digest,
        )
    norms = np.linalg.norm(weighted_basis, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        return not_estimable(
            "latent_nuisance_coordinate_basis_support_not_estimable",
            singular_values=singular_values,
            explained_fractions=fractions,
            control_rank=control_rank,
            static_rank=static_rank,
            static_control_rank=static_control_rank,
            control_input_digest=control_input_digest,
            control_residual_digest=control_residual_digest,
        )
    coordinate_basis = feature_coefficients / norms[np.newaxis, :]
    orthogonality = (
        0.0
        if not static_basis.shape[1]
        else float(
            np.max(
                np.abs(
                    static_basis.T
                    @ (precision[:, np.newaxis] * coordinate_basis)
                )
            )
        )
    )
    program_ids = tuple(
        stable_id(
            "latent_nuisance_program",
            {
                "component_index": component,
                "control_score_input_digest": control_input_digest,
                "contrast_name": normalized_contrast,
                "fold_id": normalized_fold,
                "receiver": normalized_receiver,
                "spec_id": spec.spec_id,
            },
            schema_version="1",
        )
        for component in range(n_components)
    )
    return _build_artifact(
        receiver=normalized_receiver,
        contrast_name=normalized_contrast,
        fold_id=normalized_fold,
        status="observed",
        reason_code=None,
        spec=spec,
        feature_ids=features,
        control_feature_ids=control_ids,
        sample_ids=samples,
        subject_ids=subjects,
        context_ids=contexts,
        row_manifest_id=row_manifest,
        program_ids=program_ids,
        coordinate_basis=coordinate_basis,
        factor_scores=factor_scores,
        control_score_basis=control_score_basis,
        static_coordinate_basis=static_basis,
        precision_weights=precision,
        static_coordinate_basis_id=static_id,
        static_program_ids=static_ids,
        response_digest=response_digest,
        family_basis_digest=family_digest,
        nuisance_digest=nuisance_digest,
        training_weights_digest=weight_digest,
        feature_center_digest=center_digest,
        feature_scale_digest=scale_digest,
        precision_weights_digest=precision_digest,
        training_input_digest=input_digest,
        control_score_input_digest=control_input_digest,
        control_residual_digest=control_residual_digest,
        singular_values=singular_values,
        explained_variance_fractions=fractions,
        retained_explained_fraction=float(np.sum(fractions[:n_components])),
        control_numerical_rank=control_rank,
        static_numerical_rank=static_rank,
        static_control_numerical_rank=static_control_rank,
        static_orthogonality_max_abs=orthogonality,
        family_count=families.shape[1],
        nuisance_column_count=nuisance.shape[1],
    )


__all__ = [
    "FoldLatentNuisanceArtifact",
    "FrozenLatentNuisanceSpec",
    "LatentNuisanceStatus",
    "fit_fold_latent_nuisance",
]
