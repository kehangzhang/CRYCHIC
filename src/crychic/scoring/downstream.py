"""Fold-frozen receiver-program scoring for held-out samples."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from crychic.attribution import SolverStatus, solve_nonnegative_elastic_net
from crychic.core import stable_id

from .contracts import float64_array_digest

_MAD_GAUSSIAN_CONSISTENCY = 1.4826


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
    return stable_id(
        "sparse_matrix",
        {
            "data_digest": float64_array_digest(matrix.data),
            "indices": matrix.indices.astype(int).tolist(),
            "indptr": matrix.indptr.astype(int).tolist(),
            "shape": list(matrix.shape),
        },
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
    feature_center: np.ndarray
    feature_scale: np.ndarray
    target_weight_matrix: sparse.csc_matrix
    family_support: np.ndarray
    minimum_scale: float
    center_method: str = "reference_context_feature_median_v1"
    scale_method: str = "reference_context_scaled_mad_floor_v1"
    program_transform: str = "positive_z_saturating_v1"
    downstream_functional_id: str = field(init=False)

    def __post_init__(self) -> None:
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
            "scale_digest": float64_array_digest(scale),
            "scale_method": self.scale_method,
            "target_weight_matrix_id": _matrix_digest(matrix),
            "training_subject_ids": list(subjects),
        }
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "family_ids", families)
        object.__setattr__(self, "training_subject_ids", subjects)
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
    result: np.ndarray = np.asarray(values, dtype=np.float64).copy()
    if result.shape != shape or np.any(~np.isfinite(result)):
        raise ValueError(f"{field_name} must be finite with shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class IncrementalDownstreamFunctional:
    """Frozen receiver-null and LR-family models learned on training subjects."""

    receiver: str
    contrast_name: str
    fold_id: str
    context_regressor_id: str
    nuisance_design_id: str
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    nuisance_column_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
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
    incremental_functional_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "receiver",
            "contrast_name",
            "fold_id",
            "context_regressor_id",
            "nuisance_design_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        features = _names(tuple(self.feature_ids), field_name="feature_ids")
        families = _names(tuple(self.family_ids), field_name="family_ids")
        nuisance_ids = _names(
            tuple(self.nuisance_column_ids), field_name="nuisance_column_ids"
        )
        subjects = tuple(
            sorted(
                _names(
                    tuple(self.training_subject_ids),
                    field_name="training_subject_ids",
                )
            )
        )
        for field_name, value, allow_zero in (
            ("minimum_scale", self.minimum_scale, False),
            ("null_loss_floor", self.null_loss_floor, False),
            ("lambda1", self.lambda1, True),
            ("lambda2", self.lambda2, True),
        ):
            valid = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
            if not valid:
                relation = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{field_name} must be finite and {relation}")
        n_features = len(features)
        n_families = len(families)
        n_nuisance = len(nuisance_ids)
        center = _readonly_vector(
            self.feature_center,
            length=n_features,
            field_name="feature_center",
        )
        scale = _readonly_vector(
            self.feature_scale,
            length=n_features,
            field_name="feature_scale",
            positive=True,
        )
        if np.any(scale < self.minimum_scale - 1e-12):
            raise ValueError("feature_scale must respect minimum_scale")
        basis = sparse.csc_matrix(self.family_basis, dtype=np.float64).copy()
        if basis.shape != (n_features, n_families):
            raise ValueError("family_basis shape must equal features x families")
        if np.any(~np.isfinite(basis.data)) or np.any(basis.data < 0):
            raise ValueError("family_basis must be finite and non-negative")
        column_norms = np.sqrt(np.asarray(basis.power(2).sum(axis=0)).ravel())
        if np.any(~np.isclose(column_norms, 1.0, atol=1e-12, rtol=1e-12)):
            raise ValueError("every family basis column must have unit L2 norm")
        basis.sort_indices()
        coefficients = _readonly_vector(
            self.family_coefficients,
            length=n_families,
            field_name="family_coefficients",
        )
        if np.any(coefficients < 0):
            raise ValueError("family_coefficients must be non-negative")
        null_coefficients = _readonly_array(
            self.null_nuisance_coefficients,
            shape=(n_nuisance, n_features),
            field_name="null_nuisance_coefficients",
        )
        full_coefficients = _readonly_array(
            self.full_nuisance_coefficients,
            shape=(n_nuisance, n_features),
            field_name="full_nuisance_coefficients",
        )
        family_nuisance = _readonly_array(
            self.family_nuisance_coefficients,
            shape=(n_families, n_nuisance, n_features),
            field_name="family_nuisance_coefficients",
        )
        precision = _readonly_vector(
            self.precision_weights,
            length=n_features,
            field_name="precision_weights",
        )
        if np.any(precision < 0) or not np.any(precision > 0):
            raise ValueError("precision_weights must be non-negative and not all zero")
        basis.data.setflags(write=False)
        basis.indices.setflags(write=False)
        basis.indptr.setflags(write=False)
        payload = {
            "center_digest": float64_array_digest(center),
            "context_regressor_id": self.context_regressor_id,
            "contrast_name": self.contrast_name,
            "family_basis_id": _matrix_digest(basis),
            "family_coefficient_digest": float64_array_digest(coefficients),
            "family_ids": list(families),
            "family_nuisance_digest": float64_array_digest(family_nuisance),
            "feature_ids": list(features),
            "fold_id": self.fold_id,
            "full_nuisance_digest": float64_array_digest(full_coefficients),
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "minimum_scale": self.minimum_scale,
            "nuisance_column_ids": list(nuisance_ids),
            "nuisance_design_id": self.nuisance_design_id,
            "null_loss_floor": self.null_loss_floor,
            "null_nuisance_digest": float64_array_digest(null_coefficients),
            "precision_digest": float64_array_digest(precision),
            "receiver": self.receiver,
            "scale_digest": float64_array_digest(scale),
            "training_subject_ids": list(subjects),
        }
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "family_ids", families)
        object.__setattr__(self, "nuisance_column_ids", nuisance_ids)
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "family_basis", basis)
        object.__setattr__(self, "family_coefficients", coefficients)
        object.__setattr__(self, "null_nuisance_coefficients", null_coefficients)
        object.__setattr__(self, "full_nuisance_coefficients", full_coefficients)
        object.__setattr__(self, "family_nuisance_coefficients", family_nuisance)
        object.__setattr__(self, "precision_weights", precision)
        object.__setattr__(
            self,
            "incremental_functional_id",
            stable_id("incremental_downstream_functional", payload),
        )

    def to_dict(self) -> dict[str, object]:
        """Return learned-artifact provenance without materializing coefficients."""

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
            "training_subject_ids": list(self.training_subject_ids),
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
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class IncrementalDownstreamApplication:
    """Held-out null/full losses and bounded family-specific gains."""

    incremental_functional_id: str
    status: str
    reason_code: str | None
    null_loss: float | None
    full_loss: float | None
    model_gain: float | None
    family_losses: np.ndarray
    family_gains: np.ndarray

    def __post_init__(self) -> None:
        if not self.incremental_functional_id:
            raise ValueError("incremental_functional_id must not be empty")
        if self.status not in {"observed", "not_estimable"}:
            raise ValueError("status must be observed or not_estimable")
        observed = self.status == "observed"
        if observed == (self.reason_code is not None):
            raise ValueError("reason_code must be present exactly when not estimable")
        losses: np.ndarray = np.asarray(self.family_losses, dtype=np.float64).copy()
        gains: np.ndarray = np.asarray(self.family_gains, dtype=np.float64).copy()
        if losses.ndim != 1 or gains.shape != losses.shape:
            raise ValueError("family loss and gain vectors must align")
        scalars = (self.null_loss, self.full_loss, self.model_gain)
        if observed:
            if any(value is None or not math.isfinite(value) for value in scalars):
                raise ValueError("observed incremental evidence requires finite losses")
            if np.any(~np.isfinite(losses)) or np.any(losses < 0):
                raise ValueError(
                    "observed family losses must be finite and non-negative"
                )
            if np.any(~np.isfinite(gains)) or np.any((gains < 0) | (gains > 1)):
                raise ValueError("observed family gains must lie in [0, 1]")
            if self.model_gain is None or not 0 <= self.model_gain <= 1:
                raise ValueError("observed model_gain must lie in [0, 1]")
        else:
            if any(value is not None for value in scalars):
                raise ValueError("not-estimable incremental evidence must omit losses")
            if not np.isnan(losses).all() or not np.isnan(gains).all():
                raise ValueError("not-estimable family evidence must be NaN")
        losses.setflags(write=False)
        gains.setflags(write=False)
        object.__setattr__(self, "family_losses", losses)
        object.__setattr__(self, "family_gains", gains)


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
    """Fit training-only nuisance and non-negative LR-family response models."""

    response = np.asarray(response_matrix, dtype=np.float64)
    nuisance = np.asarray(nuisance_matrix, dtype=np.float64)
    regressor = np.asarray(context_regressor, dtype=np.float64)
    reference = np.asarray(reference_mask, dtype=bool)
    if response.ndim != 2 or response.shape[1] != len(feature_ids):
        raise ValueError("response_matrix must be samples x declared features")
    n_samples, n_features = response.shape
    if n_samples < 3 or np.any(~np.isfinite(response)):
        raise ValueError("response_matrix requires at least three complete samples")
    if nuisance.shape != (n_samples, len(nuisance_column_ids)):
        raise ValueError("nuisance_matrix shape must match samples x nuisance columns")
    if np.any(~np.isfinite(nuisance)):
        raise ValueError("nuisance_matrix must be finite")
    if regressor.shape != (n_samples,) or np.any(~np.isfinite(regressor)):
        raise ValueError("context_regressor must be a finite sample vector")
    if reference.shape != (n_samples,) or int(np.count_nonzero(reference)) < 2:
        raise ValueError("reference_mask must select at least two training samples")
    if np.linalg.matrix_rank(nuisance) != nuisance.shape[1]:
        raise ValueError("training nuisance_matrix must have full column rank")
    if not math.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("minimum_scale must be finite and positive")
    if not math.isfinite(null_loss_floor) or null_loss_floor <= 0:
        raise ValueError("null_loss_floor must be finite and positive")
    center = np.median(response[reference], axis=0)
    mad = np.median(np.abs(response[reference] - center), axis=0)
    scale = np.maximum(_MAD_GAUSSIAN_CONSISTENCY * mad, minimum_scale)
    standardized = (response - center) / scale

    null_coefficients = np.linalg.lstsq(nuisance, standardized, rcond=None)[0]
    null_residual = standardized - nuisance @ null_coefficients
    regressor_nuisance = np.linalg.lstsq(nuisance, regressor, rcond=None)[0]
    residualized_regressor = regressor - nuisance @ regressor_nuisance
    regressor_norm = float(np.dot(residualized_regressor, residualized_regressor))
    if regressor_norm <= null_loss_floor:
        raise ValueError("context regressor has no variation after nuisance projection")
    effect = residualized_regressor @ null_residual / regressor_norm
    directional_effect = np.maximum(np.asarray(effect, dtype=np.float64), 0.0)
    basis = _normalize_family_basis(
        family_basis,
        n_features=n_features,
        n_families=len(family_ids),
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
    full_coefficients = np.linalg.lstsq(nuisance, full_adjusted, rcond=None)[0]
    family_nuisance = np.empty(
        (len(family_ids), nuisance.shape[1], n_features), dtype=np.float64
    )
    for family_index, coefficient in enumerate(solution.coefficients):
        contribution = (
            np.asarray(basis.getcol(family_index).toarray()).ravel() * coefficient
        )
        adjusted = standardized - np.outer(regressor, contribution)
        family_nuisance[family_index] = np.linalg.lstsq(nuisance, adjusted, rcond=None)[
            0
        ]
    return IncrementalDownstreamFunctional(
        receiver=receiver,
        contrast_name=contrast_name,
        fold_id=fold_id,
        context_regressor_id=context_regressor_id,
        nuisance_design_id=nuisance_design_id,
        feature_ids=feature_ids,
        family_ids=family_ids,
        nuisance_column_ids=nuisance_column_ids,
        training_subject_ids=training_subject_ids,
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


def _prediction_loss(
    observed: np.ndarray, predicted: np.ndarray, precision: np.ndarray
) -> float:
    residual = observed - predicted
    return float(np.mean(np.sum(residual * residual * precision, axis=1)))


def apply_incremental_downstream_functional(
    functional: IncrementalDownstreamFunctional,
    response_matrix: np.ndarray,
    *,
    nuisance_matrix: np.ndarray,
    context_regressor: np.ndarray,
    feature_ids: tuple[str, ...],
    nuisance_column_ids: tuple[str, ...],
) -> IncrementalDownstreamApplication:
    """Compute held-out family gains relative to the frozen receiver-null model."""

    if tuple(feature_ids) != functional.feature_ids:
        raise ValueError("test feature_ids must exactly match the frozen functional")
    if tuple(nuisance_column_ids) != functional.nuisance_column_ids:
        raise ValueError(
            "test nuisance columns must exactly match the frozen functional"
        )
    response = np.asarray(response_matrix, dtype=np.float64)
    nuisance = np.asarray(nuisance_matrix, dtype=np.float64)
    regressor = np.asarray(context_regressor, dtype=np.float64)
    if response.ndim != 2 or response.shape[1] != len(functional.feature_ids):
        raise ValueError("response_matrix must be test samples x frozen features")
    n_samples = response.shape[0]
    if n_samples < 1 or np.any(~np.isfinite(response)):
        raise ValueError("test response_matrix must contain complete finite samples")
    if nuisance.shape != (n_samples, len(functional.nuisance_column_ids)):
        raise ValueError("test nuisance_matrix has the wrong shape")
    if np.any(~np.isfinite(nuisance)):
        raise ValueError("test nuisance_matrix must be finite")
    if regressor.shape != (n_samples,) or np.any(~np.isfinite(regressor)):
        raise ValueError("test context_regressor must be a finite sample vector")
    standardized = (response - functional.feature_center) / functional.feature_scale
    null_prediction = nuisance @ functional.null_nuisance_coefficients
    null_loss = _prediction_loss(
        standardized, null_prediction, functional.precision_weights
    )
    n_families = len(functional.family_ids)
    if null_loss <= functional.null_loss_floor:
        missing: np.ndarray = np.full(n_families, np.nan, dtype=np.float64)
        return IncrementalDownstreamApplication(
            incremental_functional_id=functional.incremental_functional_id,
            status="not_estimable",
            reason_code="heldout_receiver_null_loss_below_floor",
            null_loss=None,
            full_loss=None,
            model_gain=None,
            family_losses=missing,
            family_gains=missing.copy(),
        )
    predicted_effect = np.asarray(
        functional.family_basis @ functional.family_coefficients
    ).ravel()
    full_prediction = nuisance @ functional.full_nuisance_coefficients + np.outer(
        regressor, predicted_effect
    )
    full_loss = _prediction_loss(
        standardized, full_prediction, functional.precision_weights
    )
    model_gain = float(np.clip((null_loss - full_loss) / null_loss, 0.0, 1.0))
    family_losses: np.ndarray = np.empty(n_families, dtype=np.float64)
    family_gains: np.ndarray = np.empty(n_families, dtype=np.float64)
    for family_index, coefficient in enumerate(functional.family_coefficients):
        contribution = (
            np.asarray(functional.family_basis.getcol(family_index).toarray()).ravel()
            * coefficient
        )
        prediction = nuisance @ functional.family_nuisance_coefficients[
            family_index
        ] + np.outer(regressor, contribution)
        loss = _prediction_loss(standardized, prediction, functional.precision_weights)
        family_losses[family_index] = loss
        family_gains[family_index] = np.clip((null_loss - loss) / null_loss, 0.0, 1.0)
    return IncrementalDownstreamApplication(
        incremental_functional_id=functional.incremental_functional_id,
        status="observed",
        reason_code=None,
        null_loss=null_loss,
        full_loss=full_loss,
        model_gain=model_gain,
        family_losses=family_losses,
        family_gains=family_gains,
    )


def fit_downstream_functional(
    reference_expression: np.ndarray,
    *,
    receiver: str,
    contrast_name: str,
    fold_id: str,
    feature_ids: tuple[str, ...],
    family_ids: tuple[str, ...],
    training_subject_ids: tuple[str, ...],
    target_weight_matrix: sparse.spmatrix | np.ndarray,
    family_support: np.ndarray,
    minimum_scale: float = 0.25,
) -> DownstreamFunctional:
    """Fit median/MAD transforms using training reference samples only."""

    expression = np.asarray(reference_expression, dtype=np.float64)
    if expression.ndim != 2 or expression.shape[1] != len(feature_ids):
        raise ValueError("reference_expression must be samples x declared feature_ids")
    if expression.shape[0] < 2 or np.any(~np.isfinite(expression)):
        raise ValueError(
            "reference_expression requires at least two complete finite samples"
        )
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
        training_subject_ids=training_subject_ids,
        feature_center=center,
        feature_scale=scale,
        target_weight_matrix=matrix,
        family_support=family_support,
        minimum_scale=minimum_scale,
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
    "IncrementalDownstreamApplication",
    "IncrementalDownstreamFunctional",
    "apply_downstream_functional",
    "apply_incremental_downstream_functional",
    "fit_downstream_functional",
    "fit_incremental_downstream_functional",
]
