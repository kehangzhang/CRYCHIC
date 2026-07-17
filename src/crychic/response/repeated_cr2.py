"""Formal-ready CR2 receiver effects for repeated-measures designs.

This module deliberately does not replace :mod:`repeated_measures`, whose CR1
output remains exploratory.  It provides a stricter backend that can feed the
full-pipeline resampling and G3-F release chain.  Analytic p-values are never
produced here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from crychic.core import ContractError, stable_id
from crychic.design import FrozenRepeatedMeasuresDesign, default_design_formula

from .repeated_measures import _names, _numeric_digest, _support, _vector_estimable

_BACKEND = "subject_equal_wls_cluster_cr2_v1"
_PAIRED_BACKEND = "subject_equal_paired_contrast_cr2_v1"
_DF_METHOD = "contrast_support_clusters_minus_one_guard_only_v1"
_RELEASE_REQUIREMENT = "full_pipeline_resampling_and_g3f_gate"
_SCHEMA_VERSION = "1.0.0"
_MIN_FORMAL_CLUSTERS = 6
_LEVERAGE_TOLERANCE = 1.0e-10
_COVARIANCE_TOLERANCE = 1.0e-10
_FEATURE_TOKEN = object()
_RESULT_TOKEN = object()


class RepeatedMeasuresCR2Status(StrEnum):
    """Eligibility of one feature for the strict CR2 backend."""

    ELIGIBLE = "eligible"
    NOT_ESTIMABLE = "not_estimable"


def _identity_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return 0.0 if value == 0.0 else value


@dataclass(frozen=True, slots=True, init=False)
class RepeatedMeasuresCR2FeatureEffect:
    """One producer-owned effect eligible for full-pipeline formal inference."""

    feature_id: str
    effect: float | None
    standard_error: float | None
    raw_precision: float | None
    n_input_samples: int
    n_model_cells: int
    n_subject_clusters: int
    n_contrast_subject_clusters: int
    n_effective_clusters: int
    n_context_subjects_min: int
    n_repeated_subject_clusters: int
    n_complete_contrast_subjects: int
    design_rank: int
    residual_df: int | None
    cluster_df: int | None
    condition_number: float | None
    maximum_cluster_leverage: float | None
    minimum_cr2_adjustment_eigenvalue: float | None
    status: RepeatedMeasuresCR2Status
    reason_code: str | None
    diagnostic_codes: tuple[str, ...]
    backend: str
    degrees_of_freedom_method: str
    effect_id: str
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "RepeatedMeasuresCR2FeatureEffect is producer-owned; "
            "use fit_repeated_measures_cr2_receiver_effect()"
        )

    @classmethod
    def _from_fit(cls, **values: object) -> RepeatedMeasuresCR2FeatureEffect:
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_producer_token", _FEATURE_TOKEN)
        object.__setattr__(self, "effect_id", "pending")
        object.__setattr__(
            self,
            "effect_id",
            stable_id(
                "repeated_measures_cr2_feature_effect",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    @property
    def formal_backend_eligible(self) -> bool:
        """Whether the feature can enter resampled formal inference."""

        self._require_intact()
        return self.status is RepeatedMeasuresCR2Status.ELIGIBLE

    @property
    def formal_inference_allowed(self) -> bool:
        """Analytic release is never authorized by this point-fit artifact."""

        self._require_intact()
        return False

    @property
    def exploratory_only(self) -> bool:
        """The CR2 backend is not the legacy exploratory CR1 backend."""

        self._require_intact()
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "cluster_df": self.cluster_df,
            "condition_number": _identity_float(self.condition_number),
            "degrees_of_freedom_method": self.degrees_of_freedom_method,
            "design_rank": self.design_rank,
            "diagnostic_codes": list(self.diagnostic_codes),
            "effect": _identity_float(self.effect),
            "feature_id": self.feature_id,
            "maximum_cluster_leverage": _identity_float(
                self.maximum_cluster_leverage
            ),
            "minimum_cr2_adjustment_eigenvalue": _identity_float(
                self.minimum_cr2_adjustment_eigenvalue
            ),
            "n_complete_contrast_subjects": self.n_complete_contrast_subjects,
            "n_context_subjects_min": self.n_context_subjects_min,
            "n_contrast_subject_clusters": self.n_contrast_subject_clusters,
            "n_effective_clusters": self.n_effective_clusters,
            "n_input_samples": self.n_input_samples,
            "n_model_cells": self.n_model_cells,
            "n_repeated_subject_clusters": self.n_repeated_subject_clusters,
            "n_subject_clusters": self.n_subject_clusters,
            "raw_precision": _identity_float(self.raw_precision),
            "reason_code": self.reason_code,
            "release_requirement": _RELEASE_REQUIREMENT,
            "minimum_formal_clusters": _MIN_FORMAL_CLUSTERS,
            "residual_df": self.residual_df,
            "standard_error": _identity_float(self.standard_error),
            "status": self.status.value,
        }

    def _require_intact(self) -> None:
        try:
            counts = (
                self.n_input_samples,
                self.n_model_cells,
                self.n_subject_clusters,
                self.n_contrast_subject_clusters,
                self.n_effective_clusters,
                self.n_context_subjects_min,
                self.n_repeated_subject_clusters,
                self.n_complete_contrast_subjects,
                self.design_rank,
            )
            counts_valid = bool(
                all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in counts
                )
                and self.n_contrast_subject_clusters <= self.n_subject_clusters
                and self.n_effective_clusters <= self.n_subject_clusters
                and self.n_context_subjects_min <= self.n_subject_clusters
                and self.n_repeated_subject_clusters <= self.n_subject_clusters
                and self.n_complete_contrast_subjects <= self.n_subject_clusters
            )
            valid = bool(
                self._producer_token is _FEATURE_TOKEN
                and isinstance(self.feature_id, str)
                and bool(self.feature_id)
                and self.feature_id == self.feature_id.strip()
                and self.backend in {_BACKEND, _PAIRED_BACKEND}
                and self.degrees_of_freedom_method == _DF_METHOD
                and counts_valid
                and isinstance(self.diagnostic_codes, tuple)
                and self.effect_id
                == stable_id(
                    "repeated_measures_cr2_feature_effect",
                    self._identity_payload(),
                    schema_version=_SCHEMA_VERSION,
                )
            )
            if self.status is RepeatedMeasuresCR2Status.ELIGIBLE:
                valid = bool(
                    valid
                    and self.reason_code is None
                    and self.effect is not None
                    and math.isfinite(self.effect)
                    and self.standard_error is not None
                    and math.isfinite(self.standard_error)
                    and self.standard_error > 0.0
                    and self.raw_precision is not None
                    and math.isfinite(self.raw_precision)
                    and self.raw_precision > 0.0
                    and math.isclose(
                        self.raw_precision,
                        1.0 / self.standard_error**2,
                        rel_tol=1.0e-10,
                        abs_tol=0.0,
                    )
                    and self.design_rank > 0
                    and self.residual_df is not None
                    and self.residual_df >= 1
                    and self.cluster_df is not None
                    and self.n_effective_clusters >= _MIN_FORMAL_CLUSTERS
                    and self.cluster_df == self.n_effective_clusters - 1
                    and self.condition_number is not None
                    and math.isfinite(self.condition_number)
                    and self.condition_number >= 1.0
                    and self.maximum_cluster_leverage is not None
                    and math.isfinite(self.maximum_cluster_leverage)
                    and 0.0
                    <= self.maximum_cluster_leverage
                    < 1.0 - _LEVERAGE_TOLERANCE
                    and self.minimum_cr2_adjustment_eigenvalue is not None
                    and math.isfinite(self.minimum_cr2_adjustment_eigenvalue)
                    and self.minimum_cr2_adjustment_eigenvalue
                    > _LEVERAGE_TOLERANCE
                    and (
                        (
                            self.backend == _PAIRED_BACKEND
                            and self.diagnostic_codes
                            == ("paired_subject_contrast",)
                        )
                        or (
                            self.backend == _BACKEND
                            and not self.diagnostic_codes
                        )
                    )
                )
            else:
                condition_valid = bool(
                    self.condition_number is None
                    or (
                        math.isfinite(self.condition_number)
                        and self.condition_number >= 1.0
                    )
                )
                valid = bool(
                    valid
                    and self.status is RepeatedMeasuresCR2Status.NOT_ESTIMABLE
                    and self.reason_code
                    and self.effect is None
                    and self.standard_error is None
                    and self.raw_precision is None
                    and self.maximum_cluster_leverage is None
                    and self.minimum_cr2_adjustment_eigenvalue is None
                    and not self.diagnostic_codes
                    and condition_valid
                    and self.cluster_df
                    == (
                        self.n_effective_clusters - 1
                        if self.n_effective_clusters > 0
                        else None
                    )
                )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Repeated-measures CR2 feature failed integrity validation",
                code="repeated_measures_cr2_feature_integrity_violation",
                field="effect_id",
                remediation="Refit the CR2 receiver effect from intact inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Repeated-measures CR2 feature failed integrity validation",
                code="repeated_measures_cr2_feature_integrity_violation",
                field="effect_id",
                remediation="Refit the CR2 receiver effect from intact inputs",
            )

    def as_record(self) -> dict[str, object]:
        """Return a flat record without analytic p/q fields."""

        self._require_intact()
        return {
            "effect_id": self.effect_id,
            "feature_id": self.feature_id,
            "effect": self.effect,
            "effect_semantics": "declared_context_contrast_adjusted_mean",
            "standard_error": self.standard_error,
            "raw_precision": self.raw_precision,
            "n_input_samples": self.n_input_samples,
            "n_model_cells": self.n_model_cells,
            "n_subject_clusters": self.n_subject_clusters,
            "n_contrast_subject_clusters": self.n_contrast_subject_clusters,
            "n_effective_clusters": self.n_effective_clusters,
            "n_context_subjects_min": self.n_context_subjects_min,
            "n_repeated_subject_clusters": self.n_repeated_subject_clusters,
            "n_complete_contrast_subjects": self.n_complete_contrast_subjects,
            "design_rank": self.design_rank,
            "residual_df": self.residual_df,
            "cluster_df": self.cluster_df,
            "condition_number": self.condition_number,
            "maximum_cluster_leverage": self.maximum_cluster_leverage,
            "minimum_cr2_adjustment_eigenvalue": (
                self.minimum_cr2_adjustment_eigenvalue
            ),
            "status": self.status.value,
            "reason_code": self.reason_code,
            "diagnostic_codes": self.diagnostic_codes,
            "backend": self.backend,
            "degrees_of_freedom_method": self.degrees_of_freedom_method,
            "formal_backend_eligible": self.formal_backend_eligible,
            "formal_inference_allowed": False,
            "exploratory_only": False,
            "release_requirement": _RELEASE_REQUIREMENT,
            "minimum_formal_clusters": _MIN_FORMAL_CLUSTERS,
        }


@dataclass(frozen=True, slots=True, init=False)
class RepeatedMeasuresCR2ReceiverEffect:
    """Producer-owned multi-feature formal-ready receiver result."""

    design_id: str
    contrast_name: str
    receiver: str
    sample_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    sample_values_digest: str
    response_input_id: str
    feature_effects: tuple[RepeatedMeasuresCR2FeatureEffect, ...]
    artifact_id: str
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "RepeatedMeasuresCR2ReceiverEffect is producer-owned; "
            "use fit_repeated_measures_cr2_receiver_effect()"
        )

    @classmethod
    def _from_fit(
        cls,
        *,
        design: FrozenRepeatedMeasuresDesign,
        receiver: str,
        sample_ids: tuple[str, ...],
        feature_ids: tuple[str, ...],
        sample_values_digest: str,
        response_input_id: str,
        feature_effects: tuple[RepeatedMeasuresCR2FeatureEffect, ...],
    ) -> RepeatedMeasuresCR2ReceiverEffect:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "design_id": design.design_id,
            "contrast_name": design.contrast.name,
            "receiver": receiver,
            "sample_ids": sample_ids,
            "feature_ids": feature_ids,
            "sample_values_digest": sample_values_digest,
            "response_input_id": response_input_id,
            "feature_effects": feature_effects,
            "_producer_token": _RESULT_TOKEN,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "artifact_id", "pending")
        object.__setattr__(
            self,
            "artifact_id",
            stable_id(
                "repeated_measures_cr2_receiver_result",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "contrast_name": self.contrast_name,
            "design_id": self.design_id,
            "effect_ids": [effect.effect_id for effect in self.feature_effects],
            "feature_ids": list(self.feature_ids),
            "receiver": self.receiver,
            "response_input_id": self.response_input_id,
            "sample_ids": list(self.sample_ids),
            "sample_values_digest": self.sample_values_digest,
        }

    @property
    def all_features_formal_backend_eligible(self) -> bool:
        """Whether every requested feature passed the strict CR2 checks."""

        self._require_intact()
        return all(effect.formal_backend_eligible for effect in self.feature_effects)

    @property
    def formal_inference_allowed(self) -> bool:
        self._require_intact()
        return False

    @property
    def exploratory_only(self) -> bool:
        self._require_intact()
        return False

    def _require_intact(self) -> None:
        try:
            for effect in self.feature_effects:
                effect._require_intact()
            valid = (
                self._producer_token is _RESULT_TOKEN
                and tuple(effect.feature_id for effect in self.feature_effects)
                == self.feature_ids
                and len(self.feature_effects) == len(self.feature_ids)
                and self.artifact_id
                == stable_id(
                    "repeated_measures_cr2_receiver_result",
                    self._identity_payload(),
                    schema_version=_SCHEMA_VERSION,
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Repeated-measures CR2 result failed integrity validation",
                code="repeated_measures_cr2_result_integrity_violation",
                field="artifact_id",
                remediation="Refit the CR2 receiver effect from intact inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Repeated-measures CR2 result failed integrity validation",
                code="repeated_measures_cr2_result_integrity_violation",
                field="artifact_id",
                remediation="Refit the CR2 receiver effect from intact inputs",
            )

    def effect_for(self, feature_id: str) -> RepeatedMeasuresCR2FeatureEffect:
        self._require_intact()
        try:
            return self.feature_effects[self.feature_ids.index(feature_id)]
        except ValueError as error:
            raise KeyError(feature_id) from error

    def to_frame(self) -> pd.DataFrame:
        self._require_intact()
        return pd.DataFrame.from_records(
            [
                {
                    "artifact_id": self.artifact_id,
                    "response_input_id": self.response_input_id,
                    "design_id": self.design_id,
                    "contrast": self.contrast_name,
                    "receiver": self.receiver,
                    **effect.as_record(),
                }
                for effect in self.feature_effects
            ]
        )


@dataclass(frozen=True, slots=True)
class _CR2Fit:
    effect: float
    standard_error: float
    precision: float
    rank: int
    residual_df: int
    cluster_df: int
    condition_number: float
    maximum_cluster_leverage: float
    minimum_adjustment_eigenvalue: float


class _CR2Failure(RuntimeError):
    def __init__(
        self,
        reason_code: str,
        *,
        n_effective_clusters: int | None = None,
        n_model_cells: int | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.n_effective_clusters = n_effective_clusters
        self.n_model_cells = n_model_cells


def _subject_equal_cr2(
    matrix: np.ndarray,
    response: np.ndarray,
    contrast: np.ndarray,
    subjects: np.ndarray,
    *,
    n_effective_clusters: int,
    max_condition_number: float,
) -> _CR2Fit:
    if n_effective_clusters < _MIN_FORMAL_CLUSTERS:
        raise _CR2Failure(
            "insufficient_subject_clusters",
            n_effective_clusters=n_effective_clusters,
        )
    counts = {subject: int(np.sum(subjects == subject)) for subject in set(subjects)}
    weights = np.asarray([1.0 / counts[subject] for subject in subjects])
    sqrt_weights = np.sqrt(weights)
    weighted_matrix = sqrt_weights[:, np.newaxis] * matrix
    weighted_response = sqrt_weights * response
    rank = int(np.linalg.matrix_rank(weighted_matrix))
    if rank < matrix.shape[1]:
        raise _CR2Failure("rank_deficient_feature_design")
    if not _vector_estimable(weighted_matrix, contrast):
        raise _CR2Failure("feature_contrast_not_estimable")
    condition_number = float(np.linalg.cond(weighted_matrix))
    if (
        not math.isfinite(condition_number)
        or condition_number > max_condition_number
    ):
        raise _CR2Failure("ill_conditioned_feature_design")
    residual_df = len(response) - rank
    if residual_df < 1:
        raise _CR2Failure("insufficient_residual_degrees_of_freedom")
    gram = weighted_matrix.T @ weighted_matrix
    try:
        bread = np.linalg.solve(gram, np.eye(gram.shape[0]))
    except np.linalg.LinAlgError as error:
        raise _CR2Failure("rank_deficient_feature_design") from error
    coefficients = bread @ weighted_matrix.T @ weighted_response
    residual = response - matrix @ coefficients
    meat = np.zeros_like(bread)
    maximum_leverage = 0.0
    minimum_adjustment = math.inf
    for subject in sorted(set(subjects.tolist())):
        selected = subjects == subject
        cluster_matrix = weighted_matrix[selected]
        cluster_residual = sqrt_weights[selected] * residual[selected]
        leverage = cluster_matrix @ bread @ cluster_matrix.T
        leverage = 0.5 * (leverage + leverage.T)
        leverage_max = float(np.max(np.linalg.eigvalsh(leverage)))
        maximum_leverage = max(maximum_leverage, leverage_max)
        adjustment_matrix = np.eye(int(selected.sum())) - leverage
        eigenvalues, eigenvectors = np.linalg.eigh(adjustment_matrix)
        smallest = float(np.min(eigenvalues))
        minimum_adjustment = min(minimum_adjustment, smallest)
        if not np.all(np.isfinite(eigenvalues)) or smallest <= _LEVERAGE_TOLERANCE:
            raise _CR2Failure("cr2_cluster_leverage_not_estimable")
        adjustment = (
            eigenvectors
            @ np.diag(1.0 / np.sqrt(eigenvalues))
            @ eigenvectors.T
        )
        score = cluster_matrix.T @ (adjustment @ cluster_residual)
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise _CR2Failure("non_finite_cr2_covariance")
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    covariance_scale = float(np.max(np.abs(covariance_eigenvalues)))
    if covariance_scale > 0.0 and float(np.min(covariance_eigenvalues)) < -(
        _COVARIANCE_TOLERANCE * covariance_scale
    ):
        raise _CR2Failure("indefinite_cr2_covariance")
    effect = float(contrast @ coefficients)
    variance = float(contrast @ covariance @ contrast)
    if not math.isfinite(effect) or not math.isfinite(variance):
        raise _CR2Failure("non_finite_cr2_estimate")
    if variance <= 0.0:
        raise _CR2Failure("degenerate_cr2_contrast_variance")
    standard_error = math.sqrt(variance)
    precision = 1.0 / variance
    if not math.isfinite(precision):
        raise _CR2Failure("non_finite_cr2_precision")
    return _CR2Fit(
        effect=effect,
        standard_error=standard_error,
        precision=precision,
        rank=rank,
        residual_df=residual_df,
        cluster_df=n_effective_clusters - 1,
        condition_number=condition_number,
        maximum_cluster_leverage=maximum_leverage,
        minimum_adjustment_eigenvalue=minimum_adjustment,
    )


def _paired_subject_contrast_cr2(
    design: FrozenRepeatedMeasuresDesign,
    cell_values: np.ndarray,
    usable_rows: np.ndarray,
) -> tuple[_CR2Fit, int, int]:
    if design.spec.covariates:
        raise _CR2Failure("subject_fixed_effect_cr2_covariates_unsupported")
    if design.spec.formula != default_design_formula(design.spec.context_keys):
        raise _CR2Failure("subject_fixed_effect_cr2_formula_unsupported")
    subjects = np.asarray(design.cell_subject_ids, dtype=object)
    contexts = np.asarray(design.cell_context_ids, dtype=object)
    usable: np.ndarray = np.zeros(len(cell_values), dtype=bool)
    usable[usable_rows] = True
    context_weights = dict(
        zip(
            design.contrast_context_ids,
            design.contrast_context_weights,
            strict=True,
        )
    )
    contrast_values: list[float] = []
    for subject in sorted(set(subjects.tolist())):
        values: dict[str, float] = {}
        valid = True
        for context in design.contrast_context_ids:
            selected = (subjects == subject) & (contexts == context) & usable
            if int(selected.sum()) != 1:
                valid = False
                break
            values[context] = float(cell_values[selected][0])
        if valid:
            contrast_values.append(
                sum(context_weights[context] * values[context] for context in values)
            )
    n_clusters = len(contrast_values)
    if n_clusters < max(design.spec.min_subject_clusters, _MIN_FORMAL_CLUSTERS):
        raise _CR2Failure(
            "insufficient_complete_subject_clusters",
            n_effective_clusters=n_clusters,
            n_model_cells=n_clusters * len(design.contrast_context_ids),
        )
    response = np.asarray(contrast_values, dtype=float)
    matrix: np.ndarray = np.ones((n_clusters, 1), dtype=float)
    contrast: np.ndarray = np.ones(1, dtype=float)
    fit = _subject_equal_cr2(
        matrix,
        response,
        contrast,
        np.asarray([f"paired-{index}" for index in range(n_clusters)]),
        n_effective_clusters=n_clusters,
        max_condition_number=design.spec.max_condition_number,
    )
    return fit, n_clusters, n_clusters * len(design.contrast_context_ids)


def _not_estimable_feature(
    *,
    feature_id: str,
    reason_code: str,
    n_input_samples: int,
    n_model_cells: int,
    n_subject_clusters: int,
    n_contrast_subject_clusters: int,
    n_effective_clusters: int,
    n_context_subjects_min: int,
    n_repeated_subject_clusters: int,
    n_complete_contrast_subjects: int,
    rank: int,
    residual_df: int | None,
    condition_number: float | None,
    backend: str,
) -> RepeatedMeasuresCR2FeatureEffect:
    if condition_number is not None and not math.isfinite(condition_number):
        condition_number = None
    return RepeatedMeasuresCR2FeatureEffect._from_fit(
        feature_id=feature_id,
        effect=None,
        standard_error=None,
        raw_precision=None,
        n_input_samples=n_input_samples,
        n_model_cells=n_model_cells,
        n_subject_clusters=n_subject_clusters,
        n_contrast_subject_clusters=n_contrast_subject_clusters,
        n_effective_clusters=n_effective_clusters,
        n_context_subjects_min=n_context_subjects_min,
        n_repeated_subject_clusters=n_repeated_subject_clusters,
        n_complete_contrast_subjects=n_complete_contrast_subjects,
        design_rank=rank,
        residual_df=residual_df,
        cluster_df=(n_effective_clusters - 1 if n_effective_clusters > 0 else None),
        condition_number=condition_number,
        maximum_cluster_leverage=None,
        minimum_cr2_adjustment_eigenvalue=None,
        status=RepeatedMeasuresCR2Status.NOT_ESTIMABLE,
        reason_code=reason_code,
        diagnostic_codes=(),
        backend=backend,
        degrees_of_freedom_method=_DF_METHOD,
    )


def _fit_feature(
    *,
    design: FrozenRepeatedMeasuresDesign,
    feature_id: str,
    cell_values: np.ndarray,
    n_input_samples: int,
) -> RepeatedMeasuresCR2FeatureEffect:
    usable_rows = np.flatnonzero(np.isfinite(cell_values))
    support = _support(design, usable_rows)
    matrix = design.design_matrix[usable_rows]
    response = cell_values[usable_rows]
    subjects = np.asarray(design.cell_subject_ids, dtype=object)[usable_rows]
    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    condition_number = float(np.linalg.cond(matrix)) if matrix.size else None
    residual_df = len(response) - rank if matrix.size else None
    n_effective_clusters = support.n_contrast_subject_clusters
    backend = _PAIRED_BACKEND if design.spec.subject_fixed_effects else _BACKEND
    reason = design.reason_code if not design.estimable else None
    if (
        reason is None
        and support.n_context_subjects_min < design.spec.min_subjects_per_context
    ):
        reason = "insufficient_subjects_per_context"
    if (
        reason is None
        and not design.spec.subject_fixed_effects
        and support.n_contrast_subject_clusters
        < max(design.spec.min_subject_clusters, _MIN_FORMAL_CLUSTERS)
    ):
        reason = "insufficient_subject_clusters"
    if reason is not None:
        return _not_estimable_feature(
            feature_id=feature_id,
            reason_code=reason,
            n_input_samples=n_input_samples,
            n_model_cells=len(response),
            n_subject_clusters=support.n_subject_clusters,
            n_contrast_subject_clusters=support.n_contrast_subject_clusters,
            n_effective_clusters=n_effective_clusters,
            n_context_subjects_min=support.n_context_subjects_min,
            n_repeated_subject_clusters=support.n_repeated_subject_clusters,
            n_complete_contrast_subjects=support.n_complete_contrast_subjects,
            rank=rank,
            residual_df=residual_df,
            condition_number=condition_number,
            backend=backend,
        )
    try:
        if design.spec.subject_fixed_effects:
            fit, n_effective_clusters, n_model_cells = _paired_subject_contrast_cr2(
                design, cell_values, usable_rows
            )
        else:
            fit = _subject_equal_cr2(
                matrix,
                response,
                design.contrast_vector,
                subjects,
                n_effective_clusters=n_effective_clusters,
                max_condition_number=design.spec.max_condition_number,
            )
            n_model_cells = len(response)
    except _CR2Failure as error:
        if error.n_effective_clusters is not None:
            n_effective_clusters = error.n_effective_clusters
        failed_model_cells = (
            error.n_model_cells
            if error.n_model_cells is not None
            else len(response)
        )
        return _not_estimable_feature(
            feature_id=feature_id,
            reason_code=error.reason_code,
            n_input_samples=n_input_samples,
            n_model_cells=failed_model_cells,
            n_subject_clusters=support.n_subject_clusters,
            n_contrast_subject_clusters=support.n_contrast_subject_clusters,
            n_effective_clusters=n_effective_clusters,
            n_context_subjects_min=support.n_context_subjects_min,
            n_repeated_subject_clusters=support.n_repeated_subject_clusters,
            n_complete_contrast_subjects=support.n_complete_contrast_subjects,
            rank=rank,
            residual_df=residual_df,
            condition_number=condition_number,
            backend=backend,
        )
    return RepeatedMeasuresCR2FeatureEffect._from_fit(
        feature_id=feature_id,
        effect=fit.effect,
        standard_error=fit.standard_error,
        raw_precision=fit.precision,
        n_input_samples=n_input_samples,
        n_model_cells=n_model_cells,
        n_subject_clusters=support.n_subject_clusters,
        n_contrast_subject_clusters=support.n_contrast_subject_clusters,
        n_effective_clusters=n_effective_clusters,
        n_context_subjects_min=support.n_context_subjects_min,
        n_repeated_subject_clusters=support.n_repeated_subject_clusters,
        n_complete_contrast_subjects=support.n_complete_contrast_subjects,
        design_rank=fit.rank,
        residual_df=fit.residual_df,
        cluster_df=fit.cluster_df,
        condition_number=fit.condition_number,
        maximum_cluster_leverage=fit.maximum_cluster_leverage,
        minimum_cr2_adjustment_eigenvalue=fit.minimum_adjustment_eigenvalue,
        status=RepeatedMeasuresCR2Status.ELIGIBLE,
        reason_code=None,
        diagnostic_codes=(
            ("paired_subject_contrast",) if design.spec.subject_fixed_effects else ()
        ),
        backend=backend,
        degrees_of_freedom_method=_DF_METHOD,
    )


def fit_repeated_measures_cr2_receiver_effect(
    design: FrozenRepeatedMeasuresDesign,
    sample_values: np.ndarray,
    *,
    sample_ids: Sequence[str],
    feature_ids: Sequence[str],
    receiver: str,
) -> RepeatedMeasuresCR2ReceiverEffect:
    """Fit strict subject-equal CR2 effects without releasing analytic tests."""

    if not isinstance(design, FrozenRepeatedMeasuresDesign):
        raise TypeError("design must be a FrozenRepeatedMeasuresDesign")
    design._require_intact()
    samples = _names(sample_ids, field_name="sample_ids", allow_empty=True)
    features = _names(feature_ids, field_name="feature_ids")
    if not isinstance(receiver, str) or not receiver or receiver != receiver.strip():
        raise ValueError("receiver must be a canonical non-empty string")
    values = np.asarray(sample_values, dtype=np.float64)
    if values.shape != (len(samples), len(features)):
        raise ValueError("sample_values must align with sample_ids x feature_ids")
    if np.any(np.isinf(values)):
        raise ValueError("sample_values cannot contain infinity")
    unknown = set(samples).difference(design.sample_ids)
    if unknown:
        raise ValueError(
            "sample_values contain samples outside the frozen design: "
            + ",".join(sorted(unknown))
        )
    input_by_sample = {sample: index for index, sample in enumerate(samples)}
    aligned: np.ndarray = np.full(
        (len(design.sample_ids), len(features)), np.nan, dtype=np.float64
    )
    for output_index, sample_id in enumerate(design.sample_ids):
        input_index = input_by_sample.get(sample_id)
        if input_index is not None:
            aligned[output_index] = values[input_index]
    values_digest = _numeric_digest(aligned)
    response_input_id = stable_id(
        "repeated_measures_cr2_receiver_input",
        {
            "backend": _BACKEND,
            "design_id": design.design_id,
            "feature_ids": list(features),
            "receiver": receiver,
            "sample_ids": list(design.sample_ids),
            "sample_values_digest": values_digest,
        },
        schema_version=_SCHEMA_VERSION,
        digest_length=64,
    )
    cell_values: np.ndarray = np.full(
        (len(design.cell_ids), len(features)), np.nan, dtype=np.float64
    )
    for cell_index in range(len(design.cell_ids)):
        selected = design.sample_cell_indices == cell_index
        cell_samples = aligned[selected]
        finite = np.isfinite(cell_samples)
        counts = finite.sum(axis=0)
        sums = np.where(finite, cell_samples, 0.0).sum(axis=0)
        observed = counts > 0
        cell_values[cell_index, observed] = sums[observed] / counts[observed]
    effects = tuple(
        _fit_feature(
            design=design,
            feature_id=feature_id,
            cell_values=cell_values[:, feature_index],
            n_input_samples=int(np.isfinite(aligned[:, feature_index]).sum()),
        )
        for feature_index, feature_id in enumerate(features)
    )
    return RepeatedMeasuresCR2ReceiverEffect._from_fit(
        design=design,
        receiver=receiver,
        sample_ids=tuple(design.sample_ids),
        feature_ids=features,
        sample_values_digest=values_digest,
        response_input_id=response_input_id,
        feature_effects=effects,
    )


__all__ = [
    "RepeatedMeasuresCR2FeatureEffect",
    "RepeatedMeasuresCR2ReceiverEffect",
    "RepeatedMeasuresCR2Status",
    "fit_repeated_measures_cr2_receiver_effect",
]
