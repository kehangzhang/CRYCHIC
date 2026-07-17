"""Deterministic relative-penalty scaling and subject-equal selection.

Verified workflow records may condition on a representation frozen across the
complete outer-training fold.  Their status names make that boundary explicit;
they do not claim fully nested refitting of precision or family construction.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import numpy as np
from scipy import sparse

from crychic.core import ContractError, stable_id

_SELECTION_RULE = (
    "subject_equal_paired_delta_one_se_l1_then_l2_sparsity_priority_v2"
)
_CANDIDATE_PRIORITY = "l1_fraction_desc_then_l2_fraction_desc_v1"
_THRESHOLD_SEMANTICS = (
    "candidate_minus_empirical_best_subject_loss_standard_error_v1"
)
_SELECTION_INTERPRETATION = (
    "descriptive_paired_delta_heuristic_not_ci_or_noninferiority_test_v1"
)
_PAIRED_SE_METHOD = "sample_sd_ddof1_over_sqrt_n_subjects_v1"
_EVALUATION_MARKER = "crychic.attribution.penalty_fold_evaluation.v2"
_SCALE_MARKER = "crychic.attribution.penalty_scale_resolution.v1"
_TUNING_MARKER = "crychic.attribution.penalty_tuning.v2"
_COMPARISON_MARKER = "crychic.attribution.penalty_candidate_comparison.v1"
_OBSERVED = "observed"
_NOT_ESTIMABLE = "not_estimable"
_FAILED = "failed"
_STATUSES = frozenset({_OBSERVED, _NOT_ESTIMABLE, _FAILED})
_DIRECTIONAL_SCALE_SPACE = "direction_compatible_nonnegative_v1"
_RESIDUAL_SCALE_SPACE = "signed_residual_nonnegative_coefficients_v1"
_SCALE_SPACES = frozenset({_DIRECTIONAL_SCALE_SPACE, _RESIDUAL_SCALE_SPACE})
_CALLER_RECORDED_EVALUATION = "caller_recorded_fold_losses_unverified"
_NONCERTIFYING_TUNING = (
    "caller_recorded_paired_delta_inner_losses_selection_only_v2"
)
_VERIFIED_EVALUATION = (
    "verified_outer_frozen_representation_subject_blocked_inner_fit_apply_v1"
)
_VERIFIED_TUNING = (
    "verified_outer_frozen_representation_subject_blocked_inner_selection_v2"
)
_NOT_ESTIMABLE_TUNING = "subject_blocked_inner_tuning_not_estimable_v2"
_EVALUATION_VERIFICATION_STATUSES = frozenset(
    {_CALLER_RECORDED_EVALUATION, _VERIFIED_EVALUATION}
)
_WORKFLOW_SUBJECT_BLOCKED_PRODUCER_TOKEN = object()


class PenaltyValidationLossEstimand(StrEnum):
    """Validation loss unit carried by each candidate/fold evaluation."""

    CALLER_RECORDED_SUBJECT = "caller_recorded_subject_validation_loss_v1"
    PAIRED_SUBJECT_CONTRAST = "paired_subject_contrast_prediction_loss_v1"
    INDEPENDENT_SUBJECT_PREDICTION = (
        "independent_subject_full_prediction_loss_v1"
    )
    MIXED_SUBJECT_PREDICTION = "mixed_subject_equal_full_prediction_loss_v1"


def _validation_loss_estimand(
    value: PenaltyValidationLossEstimand | str,
) -> PenaltyValidationLossEstimand:
    try:
        return PenaltyValidationLossEstimand(value)
    except (TypeError, ValueError) as error:
        raise ValueError("validation_loss_estimand is not supported") from error


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(
    values: Sequence[str], *, field_name: str, minimum: int = 1
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if len(result) < minimum:
        raise ValueError(f"{field_name} must contain at least {minimum} values")
    for value in result:
        _name(value, field_name=field_name)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique values")
    return result


def _fraction(value: object, *, field_name: str, upper_bound: float | None) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be a finite numeric scalar") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    if upper_bound is not None and result > upper_bound:
        raise ValueError(f"{field_name} must not exceed {upper_bound:g}")
    return 0.0 if result == 0.0 else result


def _immutable_vector(values: np.ndarray) -> np.ndarray:
    canonical = np.asarray(values, dtype="<f8", order="C").copy(order="C")
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
        raise ValueError("tuning arrays must contain only finite values")
    canonical[canonical == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _matrix_digest(matrix: sparse.csc_matrix) -> str:
    canonical = sparse.csc_matrix(matrix, dtype="<f8", copy=True)
    canonical.sum_duplicates()
    canonical.eliminate_zeros()
    canonical.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(canonical.data, dtype="<f8").tobytes())
    digest.update(np.asarray(canonical.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(canonical.indptr, dtype="<i8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class RelativePenaltyCandidate:
    """One scale-free elastic-net candidate."""

    lambda1_fraction: float
    lambda2_fraction: float
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        lambda1 = _fraction(
            self.lambda1_fraction,
            field_name="lambda1_fraction",
            upper_bound=1.0,
        )
        lambda2 = _fraction(
            self.lambda2_fraction,
            field_name="lambda2_fraction",
            upper_bound=None,
        )
        payload = {
            "lambda1_fraction": lambda1,
            "lambda2_fraction": lambda2,
        }
        object.__setattr__(self, "lambda1_fraction", lambda1)
        object.__setattr__(self, "lambda2_fraction", lambda2)
        object.__setattr__(
            self,
            "candidate_id",
            stable_id("relative_penalty_candidate", payload, schema_version="1"),
        )

    def _require_intact(self) -> None:
        expected = RelativePenaltyCandidate(
            lambda1_fraction=self.lambda1_fraction,
            lambda2_fraction=self.lambda2_fraction,
        )
        if expected.candidate_id != self.candidate_id:
            raise ContractError(
                "Relative penalty candidate failed integrity validation",
                code="penalty_candidate_integrity_violation",
                field="candidate_id",
                remediation="Recreate the candidate from canonical fractions",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the scale-free candidate manifest."""

        self._require_intact()
        return {
            "candidate_id": self.candidate_id,
            "lambda1_fraction": self.lambda1_fraction,
            "lambda2_fraction": self.lambda2_fraction,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PenaltyTuningSpec:
    """Pre-registered Cartesian grid and deterministic selection policy."""

    lambda1_fractions: tuple[float, ...] = (1.0, 0.3, 0.1, 0.03, 0.01)
    lambda2_fractions: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
    inner_allowed_n_splits: tuple[int, ...] = (5, 4, 3, 2)
    min_inner_train_subjects_per_context: int = 2
    min_inner_validation_subjects_per_context: int = 1
    root_seed: int = 0
    selection_rule: str = _SELECTION_RULE
    candidates: tuple[RelativePenaltyCandidate, ...] = field(init=False)
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        lambda1 = tuple(
            sorted(
                {
                    _fraction(
                        value,
                        field_name="lambda1_fractions",
                        upper_bound=1.0,
                    )
                    for value in self.lambda1_fractions
                },
                reverse=True,
            )
        )
        lambda2 = tuple(
            sorted(
                {
                    _fraction(
                        value,
                        field_name="lambda2_fractions",
                        upper_bound=None,
                    )
                    for value in self.lambda2_fractions
                },
                reverse=True,
            )
        )
        if not lambda1 or not lambda2:
            raise ValueError("penalty fraction grids must not be empty")
        allowed = tuple(self.inner_allowed_n_splits)
        if (
            not allowed
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in allowed
            )
            or tuple(sorted(set(allowed), reverse=True)) != allowed
        ):
            raise ValueError(
                "inner_allowed_n_splits must be unique integers >= 2 in "
                "descending order"
            )
        for field_name in (
            "min_inner_train_subjects_per_context",
            "min_inner_validation_subjects_per_context",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be an integer >= 1")
        if (
            isinstance(self.root_seed, bool)
            or not isinstance(self.root_seed, int)
            or not 0 <= self.root_seed <= 2**63 - 1
        ):
            raise ValueError("root_seed must be a non-negative 63-bit integer")
        if self.selection_rule != _SELECTION_RULE:
            raise ValueError(f"selection_rule must be {_SELECTION_RULE!r}")
        candidates = tuple(
            RelativePenaltyCandidate(
                lambda1_fraction=lambda1_fraction,
                lambda2_fraction=lambda2_fraction,
            )
            for lambda1_fraction in lambda1
            for lambda2_fraction in lambda2
        )
        payload = {
            "candidate_priority": _CANDIDATE_PRIORITY,
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "inner_allowed_n_splits": list(allowed),
            "lambda1_fractions": list(lambda1),
            "lambda2_fractions": list(lambda2),
            "min_inner_train_subjects_per_context": (
                self.min_inner_train_subjects_per_context
            ),
            "min_inner_validation_subjects_per_context": (
                self.min_inner_validation_subjects_per_context
            ),
            "root_seed": self.root_seed,
            "selection_rule": self.selection_rule,
        }
        object.__setattr__(self, "lambda1_fractions", lambda1)
        object.__setattr__(self, "lambda2_fractions", lambda2)
        object.__setattr__(self, "inner_allowed_n_splits", allowed)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "spec_id",
            stable_id("penalty_tuning_spec", payload, schema_version="2"),
        )

    def _require_intact(self) -> None:
        try:
            expected = PenaltyTuningSpec(
                lambda1_fractions=self.lambda1_fractions,
                lambda2_fractions=self.lambda2_fractions,
                inner_allowed_n_splits=self.inner_allowed_n_splits,
                min_inner_train_subjects_per_context=(
                    self.min_inner_train_subjects_per_context
                ),
                min_inner_validation_subjects_per_context=(
                    self.min_inner_validation_subjects_per_context
                ),
                root_seed=self.root_seed,
                selection_rule=self.selection_rule,
            )
            valid = self.candidates == expected.candidates and (
                self.spec_id == expected.spec_id
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                "Penalty tuning specification failed integrity validation",
                code="penalty_tuning_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the pre-registered candidate grid",
            ) from error
        if not valid:
            raise ContractError(
                "Penalty tuning specification failed integrity validation",
                code="penalty_tuning_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the pre-registered candidate grid",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical candidate-grid manifest."""

        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "candidate_priority": _CANDIDATE_PRIORITY,
            "lambda1_fractions": list(self.lambda1_fractions),
            "lambda2_fractions": list(self.lambda2_fractions),
            "inner_allowed_n_splits": list(self.inner_allowed_n_splits),
            "min_inner_train_subjects_per_context": (
                self.min_inner_train_subjects_per_context
            ),
            "min_inner_validation_subjects_per_context": (
                self.min_inner_validation_subjects_per_context
            ),
            "root_seed": self.root_seed,
            "selection_rule": self.selection_rule,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True, init=False)
class ResolvedPenalty:
    """Absolute penalty values resolved for one training scope."""

    candidate_id: str
    scale_resolution_id: str
    lambda1_fraction: float
    lambda2_fraction: float
    lambda1: float
    lambda2: float
    resolved_penalty_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ResolvedPenalty is producer-owned; use PenaltyScaleResolution.resolve()"
        )

    @classmethod
    def _from_resolution(
        cls,
        candidate: RelativePenaltyCandidate,
        resolution: PenaltyScaleResolution,
    ) -> ResolvedPenalty:
        candidate._require_intact()
        resolution._require_intact()
        if not resolution.estimable:
            raise ContractError(
                "Cannot resolve a candidate from a non-estimable penalty scale",
                code="penalty_scale_not_estimable",
                field="scale_resolution_id",
                remediation="Increase positive weighted family-basis support",
            )
        assert resolution.lambda1_max is not None
        assert resolution.lambda2_scale is not None
        self = object.__new__(cls)
        values = {
            "candidate_id": candidate.candidate_id,
            "scale_resolution_id": resolution.scale_resolution_id,
            "lambda1_fraction": candidate.lambda1_fraction,
            "lambda2_fraction": candidate.lambda2_fraction,
            "lambda1": candidate.lambda1_fraction * resolution.lambda1_max,
            "lambda2": candidate.lambda2_fraction * resolution.lambda2_scale,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "resolved_penalty_id",
            stable_id("resolved_penalty", values, schema_version="1"),
        )
        object.__setattr__(self, "_producer_marker", _SCALE_MARKER)
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "lambda1": self.lambda1,
            "lambda1_fraction": self.lambda1_fraction,
            "lambda2": self.lambda2,
            "lambda2_fraction": self.lambda2_fraction,
            "scale_resolution_id": self.scale_resolution_id,
        }

    def _require_intact(self) -> None:
        try:
            lambda1_fraction = _fraction(
                self.lambda1_fraction,
                field_name="lambda1_fraction",
                upper_bound=1.0,
            )
            lambda2_fraction = _fraction(
                self.lambda2_fraction,
                field_name="lambda2_fraction",
                upper_bound=None,
            )
            valid = (
                self._producer_marker == _SCALE_MARKER
                and lambda1_fraction == self.lambda1_fraction
                and lambda2_fraction == self.lambda2_fraction
                and math.isfinite(self.lambda1)
                and self.lambda1 >= 0
                and math.isfinite(self.lambda2)
                and self.lambda2 >= 0
                and stable_id(
                    "resolved_penalty",
                    self._identity_payload(),
                    schema_version="1",
                )
                == self.resolved_penalty_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Resolved penalty failed integrity validation",
                code="resolved_penalty_integrity_violation",
                field="resolved_penalty_id",
                remediation="Resolve the candidate from an intact training scale",
            ) from error
        if not valid:
            raise ContractError(
                "Resolved penalty failed integrity validation",
                code="resolved_penalty_integrity_violation",
                field="resolved_penalty_id",
                remediation="Resolve the candidate from an intact training scale",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the absolute penalty manifest."""

        self._require_intact()
        return {
            "resolved_penalty_id": self.resolved_penalty_id,
            **self._identity_payload(),
        }


@dataclass(frozen=True, slots=True, init=False)
class PenaltyScaleResolution:
    """Producer-owned relative-to-absolute penalty scale."""

    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    problem_space: str
    basis_digest: str
    response_digest: str
    precision_digest: str
    weighted_correlations: np.ndarray
    weighted_gram_diagonal: np.ndarray
    lambda1_max: float | None
    lambda2_scale: float | None
    estimable: bool
    reason_code: str | None
    scale_resolution_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "PenaltyScaleResolution is producer-owned; use resolve_penalty_scale()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "basis_digest": self.basis_digest,
            "estimable": self.estimable,
            "family_ids": list(self.family_ids),
            "feature_ids": list(self.feature_ids),
            "lambda1_max": self.lambda1_max,
            "lambda2_scale": self.lambda2_scale,
            "precision_digest": self.precision_digest,
            "problem_space": self.problem_space,
            "reason_code": self.reason_code,
            "response_digest": self.response_digest,
            "weighted_correlations_digest": _array_digest(
                self.weighted_correlations
            ),
            "weighted_gram_diagonal_digest": _array_digest(
                self.weighted_gram_diagonal
            ),
        }

    def _require_intact(self) -> None:
        try:
            correlations = np.asarray(self.weighted_correlations, dtype="<f8")
            gram_diagonal = np.asarray(self.weighted_gram_diagonal, dtype="<f8")
            if (
                correlations.shape != (len(self.family_ids),)
                or gram_diagonal.shape != correlations.shape
                or np.any(gram_diagonal < 0)
                or not _is_immutable_byte_backed(self.weighted_correlations)
                or not _is_immutable_byte_backed(self.weighted_gram_diagonal)
            ):
                raise ValueError("invalid weighted scale arrays")
            if self.problem_space not in _SCALE_SPACES or (
                self.problem_space == _DIRECTIONAL_SCALE_SPACE
                and np.any(correlations < 0)
            ):
                raise ValueError("invalid penalty problem space")
            positive_gram = gram_diagonal[gram_diagonal > 0]
            expected_estimable = bool(positive_gram.size)
            expected_lambda1 = (
                float(2.0 * max(0.0, float(np.max(correlations))))
                if expected_estimable
                else None
            )
            expected_lambda2 = (
                float(np.median(positive_gram)) if expected_estimable else None
            )
            valid_status = self.estimable == (self.reason_code is None)
            valid_values = (
                self.lambda1_max is not None
                and self.lambda2_scale is not None
                and math.isfinite(self.lambda1_max)
                and self.lambda1_max >= 0
                and math.isfinite(self.lambda2_scale)
                and self.lambda2_scale > 0
                if self.estimable
                else self.lambda1_max is None and self.lambda2_scale is None
            )
            expected = stable_id(
                "penalty_scale_resolution",
                self._identity_payload(),
                schema_version="1",
            )
            valid = (
                self._producer_marker == _SCALE_MARKER
                and valid_status
                and valid_values
                and self.estimable == expected_estimable
                and self.lambda1_max == expected_lambda1
                and self.lambda2_scale == expected_lambda2
                and self.reason_code
                == (None if expected_estimable else "no_positive_weighted_gram_scale")
                and expected == self.scale_resolution_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Penalty scale resolution failed integrity validation",
                code="penalty_scale_integrity_violation",
                field="scale_resolution_id",
                remediation="Recompute the scale from intact ordered inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Penalty scale resolution failed integrity validation",
                code="penalty_scale_integrity_violation",
                field="scale_resolution_id",
                remediation="Recompute the scale from intact ordered inputs",
            )

    def resolve(self, candidate: RelativePenaltyCandidate) -> ResolvedPenalty:
        """Resolve one relative candidate against this training scale."""

        if not isinstance(candidate, RelativePenaltyCandidate):
            raise TypeError("candidate must be a RelativePenaltyCandidate")
        return ResolvedPenalty._from_resolution(candidate, self)

    def to_dict(self) -> dict[str, object]:
        """Return scale lineage without materializing numerical inputs."""

        self._require_intact()
        return {
            "scale_resolution_id": self.scale_resolution_id,
            **self._identity_payload(),
        }


def _resolve_penalty_scale(
    family_basis: sparse.spmatrix | np.ndarray,
    response: np.ndarray,
    precision_weights: np.ndarray,
    *,
    feature_ids: Sequence[str],
    family_ids: Sequence[str],
    problem_space: str,
) -> PenaltyScaleResolution:
    """Derive lambda scales from one typed weighted training problem."""

    features = _names(feature_ids, field_name="feature_ids")
    families = _names(family_ids, field_name="family_ids")
    basis = sparse.csc_matrix(family_basis, dtype="<f8", copy=True)
    basis.sum_duplicates()
    basis.eliminate_zeros()
    basis.sort_indices()
    y = np.asarray(response, dtype="<f8").copy()
    precision = np.asarray(precision_weights, dtype="<f8").copy()
    y[y == 0.0] = 0.0
    precision[precision == 0.0] = 0.0
    if basis.shape != (len(features), len(families)):
        raise ValueError("family_basis must align with feature_ids and family_ids")
    if y.shape != (len(features),) or precision.shape != y.shape:
        raise ValueError("response and precision_weights must align with feature_ids")
    if problem_space not in _SCALE_SPACES:
        raise ValueError("problem_space is not supported")
    directional = problem_space == _DIRECTIONAL_SCALE_SPACE
    if np.any(~np.isfinite(basis.data)) or (
        directional and np.any(basis.data < 0)
    ):
        qualifier = "finite and non-negative" if directional else "finite"
        raise ValueError(f"family_basis must be {qualifier}")
    if np.any(~np.isfinite(y)) or (directional and np.any(y < 0)):
        qualifier = "finite and non-negative" if directional else "finite"
        raise ValueError(f"response must be {qualifier}")
    if (
        np.any(~np.isfinite(precision))
        or np.any(precision < 0)
        or not np.any(precision > 0)
    ):
        raise ValueError(
            "precision_weights must be finite, non-negative, and not all zero"
        )
    weighted_response = precision * y
    correlations = np.asarray(basis.T @ weighted_response).ravel()
    gram_diagonal = np.asarray(basis.power(2).T @ precision).ravel()
    positive_gram = gram_diagonal[gram_diagonal > 0]
    estimable = bool(positive_gram.size)
    lambda1_max = (
        float(2.0 * max(0.0, float(np.max(correlations)))) if estimable else None
    )
    lambda2_scale = float(np.median(positive_gram)) if estimable else None
    reason = None if estimable else "no_positive_weighted_gram_scale"
    self = object.__new__(PenaltyScaleResolution)
    values: dict[str, Any] = {
        "feature_ids": features,
        "family_ids": families,
        "problem_space": problem_space,
        "basis_digest": _matrix_digest(basis),
        "response_digest": _array_digest(y),
        "precision_digest": _array_digest(precision),
        "weighted_correlations": _immutable_vector(correlations),
        "weighted_gram_diagonal": _immutable_vector(gram_diagonal),
        "lambda1_max": lambda1_max,
        "lambda2_scale": lambda2_scale,
        "estimable": estimable,
        "reason_code": reason,
        "_producer_marker": _SCALE_MARKER,
    }
    for attribute_name, attribute_value in values.items():
        object.__setattr__(self, attribute_name, attribute_value)
    object.__setattr__(
        self,
        "scale_resolution_id",
        stable_id(
            "penalty_scale_resolution", self._identity_payload(), schema_version="1"
        ),
    )
    return self


def resolve_penalty_scale(
    family_basis: sparse.spmatrix | np.ndarray,
    response: np.ndarray,
    precision_weights: np.ndarray,
    *,
    feature_ids: Sequence[str],
    family_ids: Sequence[str],
) -> PenaltyScaleResolution:
    """Resolve scales for a non-negative direction-compatible training problem."""

    return _resolve_penalty_scale(
        family_basis,
        response,
        precision_weights,
        feature_ids=feature_ids,
        family_ids=family_ids,
        problem_space=_DIRECTIONAL_SCALE_SPACE,
    )


def resolve_residualized_penalty_scale(
    family_basis: sparse.spmatrix | np.ndarray,
    response: np.ndarray,
    precision_weights: np.ndarray,
    *,
    feature_ids: Sequence[str],
    family_ids: Sequence[str],
) -> PenaltyScaleResolution:
    """Resolve scales for signed residuals with non-negative coefficients."""

    return _resolve_penalty_scale(
        family_basis,
        response,
        precision_weights,
        feature_ids=feature_ids,
        family_ids=family_ids,
        problem_space=_RESIDUAL_SCALE_SPACE,
    )


@dataclass(frozen=True, slots=True, init=False)
class PenaltyFoldEvaluation:
    """One candidate's subject-level validation losses in one inner fold."""

    candidate_id: str
    inner_fold_id: str
    inner_fold_manifest_id: str | None
    inner_training_subject_ids: tuple[str, ...]
    validation_subject_ids: tuple[str, ...]
    validation_loss_estimand: PenaltyValidationLossEstimand
    subject_losses: np.ndarray
    subject_losses_digest: str
    scale_resolution_id: str | None
    resolved_penalty_id: str | None
    resolved_lambda1: float | None
    resolved_lambda2: float | None
    training_functional_id: str | None
    heldout_application_id: str | None
    status: str
    reason_code: str | None
    verification_status: str
    evaluation_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "PenaltyFoldEvaluation is producer-owned; "
            "use record_penalty_fold_evaluation()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "heldout_application_id": self.heldout_application_id,
            "inner_fold_id": self.inner_fold_id,
            "inner_fold_manifest_id": self.inner_fold_manifest_id,
            "inner_training_subject_ids": list(self.inner_training_subject_ids),
            "reason_code": self.reason_code,
            "resolved_lambda1": self.resolved_lambda1,
            "resolved_lambda2": self.resolved_lambda2,
            "resolved_penalty_id": self.resolved_penalty_id,
            "scale_resolution_id": self.scale_resolution_id,
            "status": self.status,
            "subject_losses_digest": self.subject_losses_digest,
            "training_functional_id": self.training_functional_id,
            "validation_loss_estimand": self.validation_loss_estimand.value,
            "validation_subject_ids": list(self.validation_subject_ids),
            "verification_status": self.verification_status,
        }

    def _require_intact(self) -> None:
        try:
            losses = np.asarray(self.subject_losses, dtype="<f8")
            observed = self.status == _OBSERVED
            validation_subjects = _names(
                self.validation_subject_ids,
                field_name="validation_subject_ids",
            )
            validation_loss_estimand = _validation_loss_estimand(
                self.validation_loss_estimand
            )
            verified = self.verification_status == _VERIFIED_EVALUATION
            if verified:
                training_subjects = _names(
                    self.inner_training_subject_ids,
                    field_name="inner_training_subject_ids",
                )
                if set(training_subjects).intersection(validation_subjects):
                    raise ValueError("inner training and validation subjects overlap")
                if self.inner_fold_manifest_id != self.inner_fold_id:
                    raise ValueError("inner fold identity does not match its manifest")
                for field_name in (
                    "inner_fold_manifest_id",
                    "scale_resolution_id",
                    "resolved_penalty_id",
                    "training_functional_id",
                    "heldout_application_id",
                ):
                    value = getattr(self, field_name)
                    if observed:
                        _name(cast(str, value), field_name=field_name)
                    elif value is not None:
                        _name(value, field_name=field_name)
                if (self.resolved_penalty_id is None) != (
                    self.scale_resolution_id is None
                ):
                    raise ValueError(
                        "resolved penalty and scale resolution must co-occur"
                    )
                resolved_values = (
                    self.resolved_lambda1,
                    self.resolved_lambda2,
                )
                all_resolved_values_missing = all(
                    value is None for value in resolved_values
                )
                all_resolved_values_present = all(
                    value is not None for value in resolved_values
                )
                if not (
                    (
                        self.resolved_penalty_id is None
                        and all_resolved_values_missing
                    )
                    or (
                        self.resolved_penalty_id is not None
                        and all_resolved_values_present
                    )
                ):
                    raise ValueError(
                        "resolved penalty and numerical lambdas must co-occur"
                    )
                if self.resolved_penalty_id is not None and any(
                    not math.isfinite(cast(float, value))
                    or cast(float, value) < 0
                    for value in resolved_values
                ):
                    raise ValueError(
                        "resolved lambdas must be finite and non-negative"
                    )
                if self.heldout_application_id is not None and (
                    self.training_functional_id is None
                ):
                    raise ValueError(
                        "heldout application requires a training functional parent"
                    )
            else:
                training_subjects = ()
                if (
                    self.inner_training_subject_ids
                    or self.inner_fold_manifest_id is not None
                    or self.scale_resolution_id is not None
                    or self.resolved_penalty_id is not None
                    or self.resolved_lambda1 is not None
                    or self.resolved_lambda2 is not None
                    or self.training_functional_id is not None
                    or self.heldout_application_id is not None
                ):
                    raise ValueError(
                        "caller-recorded evaluations cannot claim verified parents"
                    )
            valid = (
                self._producer_marker == _EVALUATION_MARKER
                and self.verification_status in _EVALUATION_VERIFICATION_STATUSES
                and self.status in _STATUSES
                and observed == (self.reason_code is None)
                and tuple(validation_subjects) == self.validation_subject_ids
                and validation_loss_estimand is self.validation_loss_estimand
                and tuple(training_subjects) == self.inner_training_subject_ids
                and losses.ndim == 1
                and _is_immutable_byte_backed(self.subject_losses)
                and np.all(np.isfinite(losses))
                and np.all(losses >= 0)
                and (
                    len(losses) == len(self.validation_subject_ids)
                    if observed
                    else len(losses) == 0
                )
                and _array_digest(losses) == self.subject_losses_digest
                and stable_id(
                    "penalty_fold_evaluation",
                    self._identity_payload(),
                    schema_version="2",
                )
                == self.evaluation_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Penalty fold evaluation failed integrity validation",
                code="penalty_fold_evaluation_integrity_violation",
                field="evaluation_id",
                remediation="Re-record the fold result from held-out subject losses",
            ) from error
        if not valid:
            raise ContractError(
                "Penalty fold evaluation failed integrity validation",
                code="penalty_fold_evaluation_integrity_violation",
                field="evaluation_id",
                remediation="Re-record the fold result from held-out subject losses",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the validation status without expanding subject losses."""

        self._require_intact()
        return {"evaluation_id": self.evaluation_id, **self._identity_payload()}


def record_penalty_fold_evaluation(
    candidate: RelativePenaltyCandidate,
    *,
    inner_fold_id: str,
    validation_subject_ids: Sequence[str],
    subject_losses: np.ndarray | None = None,
    validation_loss_estimand: PenaltyValidationLossEstimand | str = (
        PenaltyValidationLossEstimand.CALLER_RECORDED_SUBJECT
    ),
    status: str = _OBSERVED,
    reason_code: str | None = None,
) -> PenaltyFoldEvaluation:
    """Record one complete candidate/fold result with explicit failure semantics."""

    if not isinstance(candidate, RelativePenaltyCandidate):
        raise TypeError("candidate must be a RelativePenaltyCandidate")
    candidate._require_intact()
    fold_id = _name(inner_fold_id, field_name="inner_fold_id")
    supplied_subjects = _names(
        validation_subject_ids, field_name="validation_subject_ids"
    )
    subjects = tuple(sorted(supplied_subjects))
    loss_estimand = _validation_loss_estimand(validation_loss_estimand)
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {sorted(_STATUSES)}")
    if status == _OBSERVED:
        if reason_code is not None:
            raise ValueError("observed evaluation cannot have a reason_code")
        if subject_losses is None:
            raise ValueError("observed evaluation requires subject_losses")
        supplied_losses = np.asarray(subject_losses, dtype="<f8")
        if supplied_losses.shape != (len(subjects),):
            raise ValueError("subject_losses must align with validation_subject_ids")
        if np.any(~np.isfinite(supplied_losses)) or np.any(supplied_losses < 0):
            raise ValueError("subject_losses must be finite and non-negative")
        loss_by_subject = dict(zip(supplied_subjects, supplied_losses, strict=True))
        losses = np.asarray([loss_by_subject[subject] for subject in subjects])
        frozen_losses = _immutable_vector(losses)
    else:
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ValueError("unavailable evaluation requires a non-empty reason_code")
        if subject_losses is not None and np.asarray(subject_losses).size:
            raise ValueError("unavailable evaluation cannot retain numerical losses")
        reason_code = reason_code.strip()
        frozen_losses = _immutable_vector(np.empty(0, dtype="<f8"))
    self = object.__new__(PenaltyFoldEvaluation)
    values: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "inner_fold_id": fold_id,
        "inner_fold_manifest_id": None,
        "inner_training_subject_ids": (),
        "validation_subject_ids": subjects,
        "validation_loss_estimand": loss_estimand,
        "subject_losses": frozen_losses,
        "subject_losses_digest": _array_digest(frozen_losses),
        "scale_resolution_id": None,
        "resolved_penalty_id": None,
        "resolved_lambda1": None,
        "resolved_lambda2": None,
        "training_functional_id": None,
        "heldout_application_id": None,
        "status": status,
        "reason_code": reason_code,
        "verification_status": _CALLER_RECORDED_EVALUATION,
        "_producer_marker": _EVALUATION_MARKER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "evaluation_id",
        stable_id(
            "penalty_fold_evaluation", self._identity_payload(), schema_version="2"
        ),
    )
    return self


def _record_subject_blocked_penalty_fold_evaluation(
    candidate: RelativePenaltyCandidate,
    *,
    _producer_token: object,
    inner_fold_id: str,
    inner_fold_manifest_id: str,
    training_subject_ids: Sequence[str],
    validation_subject_ids: Sequence[str],
    validation_loss_estimand: PenaltyValidationLossEstimand | str,
    subject_losses: np.ndarray | None = None,
    scale_resolution_id: str | None = None,
    resolved_penalty_id: str | None = None,
    resolved_lambda1: float | None = None,
    resolved_lambda2: float | None = None,
    training_functional_id: str | None = None,
    heldout_application_id: str | None = None,
    status: str = _OBSERVED,
    reason_code: str | None = None,
) -> PenaltyFoldEvaluation:
    """Record one workflow-verified subject-blocked inner fit/apply result."""

    if _producer_token is not _WORKFLOW_SUBJECT_BLOCKED_PRODUCER_TOKEN:
        raise TypeError(
            "verified subject-blocked evaluations are workflow-producer-owned"
        )
    if not isinstance(candidate, RelativePenaltyCandidate):
        raise TypeError("candidate must be a RelativePenaltyCandidate")
    candidate._require_intact()
    fold_id = _name(inner_fold_id, field_name="inner_fold_id")
    manifest_id = _name(
        inner_fold_manifest_id, field_name="inner_fold_manifest_id"
    )
    if manifest_id != fold_id:
        raise ValueError("inner_fold_manifest_id must equal inner_fold_id")
    training_subjects = tuple(
        sorted(
            _names(training_subject_ids, field_name="training_subject_ids")
        )
    )
    supplied_validation = _names(
        validation_subject_ids, field_name="validation_subject_ids"
    )
    validation_subjects = tuple(sorted(supplied_validation))
    loss_estimand = _validation_loss_estimand(validation_loss_estimand)
    if loss_estimand is PenaltyValidationLossEstimand.CALLER_RECORDED_SUBJECT:
        raise ValueError(
            "verified evaluations require a workflow-defined validation loss estimand"
        )
    if set(training_subjects).intersection(validation_subjects):
        raise ValueError("inner training and validation subjects must be disjoint")
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {sorted(_STATUSES)}")
    parent_values = {
        "scale_resolution_id": scale_resolution_id,
        "resolved_penalty_id": resolved_penalty_id,
        "training_functional_id": training_functional_id,
        "heldout_application_id": heldout_application_id,
    }
    if status == _OBSERVED:
        if reason_code is not None:
            raise ValueError("observed evaluation cannot have a reason_code")
        if subject_losses is None:
            raise ValueError("observed evaluation requires subject_losses")
        for field_name, value in parent_values.items():
            _name(cast(str, value), field_name=field_name)
        for lambda_field_name, resolved_value in (
            ("resolved_lambda1", resolved_lambda1),
            ("resolved_lambda2", resolved_lambda2),
        ):
            if (
                resolved_value is None
                or not math.isfinite(resolved_value)
                or resolved_value < 0
            ):
                raise ValueError(
                    f"{lambda_field_name} must be finite and non-negative"
                )
        supplied_losses = np.asarray(subject_losses, dtype="<f8")
        if supplied_losses.shape != (len(validation_subjects),):
            raise ValueError("subject_losses must align with validation_subject_ids")
        if np.any(~np.isfinite(supplied_losses)) or np.any(supplied_losses < 0):
            raise ValueError("subject_losses must be finite and non-negative")
        loss_by_subject = dict(
            zip(supplied_validation, supplied_losses, strict=True)
        )
        losses = _immutable_vector(
            np.asarray(
                [loss_by_subject[subject] for subject in validation_subjects],
                dtype="<f8",
            )
        )
    else:
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ValueError("unavailable evaluation requires a non-empty reason_code")
        if subject_losses is not None and np.asarray(subject_losses).size:
            raise ValueError("unavailable evaluation cannot retain numerical losses")
        for field_name, value in parent_values.items():
            if value is not None:
                _name(value, field_name=field_name)
        if (resolved_penalty_id is None) != (scale_resolution_id is None):
            raise ValueError("resolved penalty and scale resolution must co-occur")
        resolved_values = (resolved_lambda1, resolved_lambda2)
        if not (
            (
                resolved_penalty_id is None
                and all(value is None for value in resolved_values)
            )
            or (
                resolved_penalty_id is not None
                and all(value is not None for value in resolved_values)
            )
        ):
            raise ValueError(
                "resolved penalty and numerical lambdas must co-occur"
            )
        if any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in resolved_values
        ):
            raise ValueError("resolved lambdas must be finite and non-negative")
        if heldout_application_id is not None and training_functional_id is None:
            raise ValueError(
                "heldout application requires a training functional parent"
            )
        reason_code = reason_code.strip()
        losses = _immutable_vector(np.empty(0, dtype="<f8"))
    self = object.__new__(PenaltyFoldEvaluation)
    values: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "inner_fold_id": fold_id,
        "inner_fold_manifest_id": manifest_id,
        "inner_training_subject_ids": training_subjects,
        "validation_subject_ids": validation_subjects,
        "validation_loss_estimand": loss_estimand,
        "subject_losses": losses,
        "subject_losses_digest": _array_digest(losses),
        **parent_values,
        "resolved_lambda1": resolved_lambda1,
        "resolved_lambda2": resolved_lambda2,
        "status": status,
        "reason_code": reason_code,
        "verification_status": _VERIFIED_EVALUATION,
        "_producer_marker": _EVALUATION_MARKER,
    }
    for attribute_name, attribute_value in values.items():
        object.__setattr__(self, attribute_name, attribute_value)
    object.__setattr__(
        self,
        "evaluation_id",
        stable_id(
            "penalty_fold_evaluation", self._identity_payload(), schema_version="2"
        ),
    )
    return self


@dataclass(frozen=True, slots=True, init=False)
class PenaltyCandidateSummary:
    """Subject-equal aggregate for one relative candidate."""

    candidate_id: str
    status: str
    reason_code: str | None
    subject_ids: tuple[str, ...]
    subject_losses: np.ndarray
    mean_loss: float | None
    standard_error: float | None
    summary_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "PenaltyCandidateSummary is producer-owned; use select_penalty_candidate()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "mean_loss": self.mean_loss,
            "reason_code": self.reason_code,
            "standard_error": self.standard_error,
            "status": self.status,
            "subject_ids": list(self.subject_ids),
            "subject_losses_digest": _array_digest(self.subject_losses),
        }

    def _require_intact(self) -> None:
        try:
            _name(self.candidate_id, field_name="candidate_id")
            losses = np.asarray(self.subject_losses, dtype="<f8")
            observed = self.status == _OBSERVED
            valid_status = (
                observed
                and self.reason_code is None
                and len(self.subject_ids) >= 2
                and len(losses) == len(self.subject_ids)
                and self.mean_loss is not None
                and self.standard_error is not None
            ) or (
                self.status in {_NOT_ESTIMABLE, _FAILED}
                and isinstance(self.reason_code, str)
                and bool(self.reason_code)
                and not self.subject_ids
                and len(losses) == 0
                and self.mean_loss is None
                and self.standard_error is None
            )
            valid_values = (
                losses.ndim == 1
                and np.all(np.isfinite(losses))
                and np.all(losses >= 0)
                and _is_immutable_byte_backed(self.subject_losses)
            )
            if observed:
                expected_mean = float(np.mean(losses))
                expected_se = float(
                    np.std(losses, ddof=1) / math.sqrt(len(losses))
                )
                valid_values = valid_values and (
                    self.subject_ids == tuple(sorted(set(self.subject_ids)))
                    and self.mean_loss == expected_mean
                    and self.standard_error == expected_se
                )
            expected_id = stable_id(
                "penalty_candidate_summary", self._identity_payload()
            )
            valid = (
                self._producer_marker == _TUNING_MARKER
                and valid_status
                and valid_values
                and expected_id == self.summary_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Penalty candidate summary failed integrity validation",
                code="penalty_candidate_summary_integrity_violation",
                field="summary_id",
                remediation="Recompute the summary from intact fold evaluations",
            ) from error
        if not valid:
            raise ContractError(
                "Penalty candidate summary failed integrity validation",
                code="penalty_candidate_summary_integrity_violation",
                field="summary_id",
                remediation="Recompute the summary from intact fold evaluations",
            )

    def to_dict(self) -> dict[str, object]:
        """Return candidate-level subject-equal validation metrics."""

        self._require_intact()
        return {"summary_id": self.summary_id, **self._identity_payload()}


def _summarize_candidate(
    candidate: RelativePenaltyCandidate,
    evaluations: tuple[PenaltyFoldEvaluation, ...],
    *,
    training_subject_ids: tuple[str, ...],
) -> PenaltyCandidateSummary:
    statuses = {evaluation.status for evaluation in evaluations}
    if statuses == {_OBSERVED}:
        subject_losses: dict[str, float] = {}
        for evaluation in evaluations:
            for subject, loss in zip(
                evaluation.validation_subject_ids,
                evaluation.subject_losses,
                strict=True,
            ):
                if subject in subject_losses:
                    raise ValueError(
                        "each training subject must occur in exactly one "
                        "validation fold"
                    )
                subject_losses[subject] = float(loss)
        if set(subject_losses) != set(training_subject_ids):
            raise ValueError(
                "candidate validation subjects must exactly cover training_subject_ids"
            )
        losses = _immutable_vector(
            np.asarray([subject_losses[subject] for subject in training_subject_ids])
        )
        mean_loss: float | None = float(np.mean(losses))
        standard_error: float | None = float(
            np.std(losses, ddof=1) / math.sqrt(len(losses))
        )
        status = _OBSERVED
        reason_code = None
        subjects = training_subject_ids
    else:
        status = _FAILED if _FAILED in statuses else _NOT_ESTIMABLE
        reason_code = "candidate_incomplete_inner_coverage"
        subjects = ()
        losses = _immutable_vector(np.empty(0, dtype="<f8"))
        mean_loss = None
        standard_error = None
    self = object.__new__(PenaltyCandidateSummary)
    values = {
        "candidate_id": candidate.candidate_id,
        "status": status,
        "reason_code": reason_code,
        "subject_ids": subjects,
        "subject_losses": losses,
        "mean_loss": mean_loss,
        "standard_error": standard_error,
        "_producer_marker": _TUNING_MARKER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "summary_id",
        stable_id("penalty_candidate_summary", self._identity_payload()),
    )
    return self


@dataclass(frozen=True, slots=True, init=False)
class PenaltyCandidateComparison:
    """Descriptive paired loss comparison, not a CI or noninferiority test."""

    candidate_id: str
    best_candidate_id: str
    subject_ids: tuple[str, ...]
    loss_differences: np.ndarray
    mean_loss_difference: float
    standard_error: float
    one_se_threshold: float
    within_one_se: bool
    comparison_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "PenaltyCandidateComparison is producer-owned; "
            "use select_penalty_candidate()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "best_candidate_id": self.best_candidate_id,
            "candidate_id": self.candidate_id,
            "loss_differences_digest": _array_digest(self.loss_differences),
            "mean_loss_difference": self.mean_loss_difference,
            "n_subjects": len(self.subject_ids),
            "one_se_threshold": self.one_se_threshold,
            "standard_error_method": _PAIRED_SE_METHOD,
            "standard_error": self.standard_error,
            "subject_ids": list(self.subject_ids),
            "threshold_semantics": _THRESHOLD_SEMANTICS,
            "selection_interpretation": _SELECTION_INTERPRETATION,
            "within_one_se": self.within_one_se,
        }

    def _require_intact(self) -> None:
        try:
            subjects = _names(
                self.subject_ids,
                field_name="subject_ids",
                minimum=2,
            )
            differences = np.asarray(self.loss_differences, dtype="<f8")
            expected_mean = float(np.mean(differences))
            expected_se = float(
                np.std(differences, ddof=1) / math.sqrt(len(differences))
            )
            expected_within = expected_mean <= expected_se
            valid = (
                self._producer_marker == _COMPARISON_MARKER
                and self.subject_ids == tuple(sorted(subjects))
                and differences.shape == (len(subjects),)
                and np.all(np.isfinite(differences))
                and _is_immutable_byte_backed(self.loss_differences)
                and self.mean_loss_difference == expected_mean
                and self.standard_error == expected_se
                and self.one_se_threshold == expected_se
                and self.within_one_se == expected_within
                and stable_id(
                    "penalty_candidate_comparison",
                    self._identity_payload(),
                    schema_version="1",
                )
                == self.comparison_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Penalty candidate comparison failed integrity validation",
                code="penalty_candidate_comparison_integrity_violation",
                field="comparison_id",
                remediation="Recompute paired comparisons from intact summaries",
            ) from error
        if not valid:
            raise ContractError(
                "Penalty candidate comparison failed integrity validation",
                code="penalty_candidate_comparison_integrity_violation",
                field="comparison_id",
                remediation="Recompute paired comparisons from intact summaries",
            )

    def to_dict(self) -> dict[str, object]:
        """Return paired-delta decision provenance without expanding differences."""

        self._require_intact()
        return {"comparison_id": self.comparison_id, **self._identity_payload()}


def _compare_candidate_to_best(
    summary: PenaltyCandidateSummary,
    best: PenaltyCandidateSummary,
) -> PenaltyCandidateComparison:
    summary._require_intact()
    best._require_intact()
    if summary.status != _OBSERVED or best.status != _OBSERVED:
        raise ValueError("paired one-SE comparison requires observed summaries")
    if summary.subject_ids != best.subject_ids or len(summary.subject_ids) < 2:
        raise ValueError(
            "paired one-SE comparison requires identical subject units"
        )
    differences = _immutable_vector(
        np.asarray(summary.subject_losses, dtype="<f8")
        - np.asarray(best.subject_losses, dtype="<f8")
    )
    if np.any(~np.isfinite(differences)):
        raise ValueError("paired one-SE loss differences must be finite")
    mean_difference = float(np.mean(differences))
    standard_error = float(
        np.std(differences, ddof=1) / math.sqrt(len(differences))
    )
    if not math.isfinite(mean_difference) or not math.isfinite(standard_error):
        raise ValueError("paired one-SE statistics must be finite")
    self = object.__new__(PenaltyCandidateComparison)
    values: dict[str, object] = {
        "candidate_id": summary.candidate_id,
        "best_candidate_id": best.candidate_id,
        "subject_ids": summary.subject_ids,
        "loss_differences": differences,
        "mean_loss_difference": mean_difference,
        "standard_error": standard_error,
        "one_se_threshold": standard_error,
        "within_one_se": mean_difference <= standard_error,
        "_producer_marker": _COMPARISON_MARKER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "comparison_id",
        stable_id(
            "penalty_candidate_comparison",
            self._identity_payload(),
            schema_version="1",
        ),
    )
    return self


@dataclass(frozen=True, slots=True, init=False)
class PenaltyTuningArtifact:
    """Producer-owned descriptive paired-delta one-SE heuristic."""

    spec: PenaltyTuningSpec
    tuning_scope_id: str
    inner_fold_plan_id: str | None
    validation_loss_estimand: PenaltyValidationLossEstimand | None
    training_subject_ids: tuple[str, ...]
    inner_fold_ids: tuple[str, ...]
    evaluations: tuple[PenaltyFoldEvaluation, ...]
    summaries: tuple[PenaltyCandidateSummary, ...]
    candidate_comparisons: tuple[PenaltyCandidateComparison, ...]
    best_candidate_id: str | None
    selected_candidate_id: str | None
    best_mean_loss: float | None
    one_se_threshold: float | None
    status: str
    reason_code: str | None
    certification_status: str
    tuning_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "PenaltyTuningArtifact is producer-owned; use select_penalty_candidate()"
        )

    @property
    def selected_candidate(self) -> RelativePenaltyCandidate | None:
        """Return the selected relative candidate, if tuning was estimable."""

        self._require_intact()
        if self.selected_candidate_id is None:
            return None
        return next(
            candidate
            for candidate in self.spec.candidates
            if candidate.candidate_id == self.selected_candidate_id
        )

    @property
    def is_oof_certified(self) -> bool:
        """Return whether typed inner folds verify conditional candidate selection."""

        return (
            self.certification_status == _VERIFIED_TUNING
            and self.status == "selected"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "best_candidate_id": self.best_candidate_id,
            "best_mean_loss": self.best_mean_loss,
            "certification_status": self.certification_status,
            "candidate_comparison_ids": [
                item.comparison_id for item in self.candidate_comparisons
            ],
            "evaluation_ids": [item.evaluation_id for item in self.evaluations],
            "inner_fold_ids": list(self.inner_fold_ids),
            "inner_fold_plan_id": self.inner_fold_plan_id,
            "one_se_threshold": self.one_se_threshold,
            "reason_code": self.reason_code,
            "selected_candidate_id": self.selected_candidate_id,
            "selection_rule": self.spec.selection_rule,
            "selection_interpretation": _SELECTION_INTERPRETATION,
            "threshold_semantics": _THRESHOLD_SEMANTICS,
            "spec_id": self.spec.spec_id,
            "status": self.status,
            "summary_ids": [item.summary_id for item in self.summaries],
            "training_subject_ids": list(self.training_subject_ids),
            "tuning_scope_id": self.tuning_scope_id,
            "validation_loss_estimand": (
                None
                if self.validation_loss_estimand is None
                else self.validation_loss_estimand.value
            ),
        }

    def _require_intact(self) -> None:
        try:
            for summary in self.summaries:
                summary._require_intact()
            for comparison in self.candidate_comparisons:
                comparison._require_intact()
            if self.certification_status == _NOT_ESTIMABLE_TUNING:
                repeated = not_estimable_penalty_tuning(
                    self.spec,
                    tuning_scope_id=self.tuning_scope_id,
                    training_subject_ids=self.training_subject_ids,
                    reason_code=cast(str, self.reason_code),
                    inner_fold_plan_id=self.inner_fold_plan_id,
                    validation_loss_estimand=self.validation_loss_estimand,
                )
            else:
                repeated = select_penalty_candidate(
                    self.spec,
                    self.evaluations,
                    tuning_scope_id=self.tuning_scope_id,
                    training_subject_ids=self.training_subject_ids,
                    inner_fold_ids=self.inner_fold_ids,
                    inner_fold_plan_id=self.inner_fold_plan_id,
                )
            valid = (
                self._producer_marker == _TUNING_MARKER
                and self.certification_status
                in {
                    _NONCERTIFYING_TUNING,
                    _VERIFIED_TUNING,
                    _NOT_ESTIMABLE_TUNING,
                }
                and repeated.tuning_id == self.tuning_id
                and repeated._identity_payload() == self._identity_payload()
            )
        except (ContractError, StopIteration, TypeError, ValueError) as error:
            raise ContractError(
                "Penalty tuning artifact failed integrity validation",
                code="penalty_tuning_integrity_violation",
                field="tuning_id",
                remediation="Rerun deterministic selection from intact fold results",
            ) from error
        if not valid:
            raise ContractError(
                "Penalty tuning artifact failed integrity validation",
                code="penalty_tuning_integrity_violation",
                field="tuning_id",
                remediation="Rerun deterministic selection from intact fold results",
            )

    def to_dict(self) -> dict[str, object]:
        """Return complete selection provenance without expanding subject losses."""

        self._require_intact()
        return {
            "tuning_id": self.tuning_id,
            **self._identity_payload(),
            "spec": self.spec.to_dict(),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "candidate_comparisons": [
                comparison.to_dict()
                for comparison in self.candidate_comparisons
            ],
        }


def select_penalty_candidate(
    spec: PenaltyTuningSpec,
    evaluations: Sequence[PenaltyFoldEvaluation],
    *,
    tuning_scope_id: str,
    training_subject_ids: Sequence[str],
    inner_fold_ids: Sequence[str],
    inner_fold_plan_id: str | None = None,
) -> PenaltyTuningArtifact:
    """Select eligible candidates with explicit L1-first sparsity priority."""

    if not isinstance(spec, PenaltyTuningSpec):
        raise TypeError("spec must be a PenaltyTuningSpec")
    spec._require_intact()
    scope = _name(tuning_scope_id, field_name="tuning_scope_id")
    subjects = tuple(
        sorted(
            _names(
                training_subject_ids,
                field_name="training_subject_ids",
                minimum=2,
            )
        )
    )
    folds = tuple(
        sorted(_names(inner_fold_ids, field_name="inner_fold_ids", minimum=2))
    )
    results = tuple(evaluations)
    if any(not isinstance(item, PenaltyFoldEvaluation) for item in results):
        raise TypeError("evaluations must contain PenaltyFoldEvaluation values")
    for result in results:
        result._require_intact()
    validation_loss_estimands = {
        result.validation_loss_estimand for result in results
    }
    if len(validation_loss_estimands) != 1:
        raise ValueError(
            "all candidate/fold evaluations must use one validation loss estimand"
        )
    validation_loss_estimand = next(iter(validation_loss_estimands))
    verification_statuses = {result.verification_status for result in results}
    if len(verification_statuses) != 1:
        raise ValueError(
            "all candidate/fold evaluations must use one verification contract"
        )
    verification_status = next(iter(verification_statuses))
    if verification_status == _VERIFIED_EVALUATION:
        plan_id = _name(cast(str, inner_fold_plan_id), field_name="inner_fold_plan_id")
    else:
        if inner_fold_plan_id is not None:
            raise ValueError(
                "caller-recorded evaluations cannot claim an inner fold plan"
            )
        plan_id = None
    results = tuple(
        sorted(results, key=lambda item: (item.candidate_id, item.inner_fold_id))
    )
    keys = [(item.candidate_id, item.inner_fold_id) for item in results]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate/fold evaluations must be unique")
    expected_keys = {
        (candidate.candidate_id, fold_id)
        for candidate in spec.candidates
        for fold_id in folds
    }
    if set(keys) != expected_keys:
        raise ValueError("evaluations must form the complete candidate x fold grid")
    validation_by_fold: dict[str, tuple[str, ...]] = {}
    for fold_id in folds:
        partitions = {
            item.validation_subject_ids
            for item in results
            if item.inner_fold_id == fold_id
        }
        if len(partitions) != 1:
            raise ValueError("all candidates must use identical validation partitions")
        validation_by_fold[fold_id] = next(iter(partitions))
    validation_counts = dict.fromkeys(subjects, 0)
    for validation_subjects in validation_by_fold.values():
        unknown = set(validation_subjects).difference(subjects)
        if unknown:
            raise ValueError("inner validation subjects are outside the tuning scope")
        for subject in validation_subjects:
            validation_counts[subject] += 1
    if any(count != 1 for count in validation_counts.values()):
        raise ValueError(
            "inner folds must validate every training subject exactly once"
        )
    if verification_status == _VERIFIED_EVALUATION:
        subject_set = set(subjects)
        for result in results:
            if result.inner_fold_manifest_id != result.inner_fold_id:
                raise ValueError("verified evaluation has incompatible fold lineage")
            expected_training = tuple(
                sorted(subject_set.difference(result.validation_subject_ids))
            )
            if result.inner_training_subject_ids != expected_training:
                raise ValueError(
                    "verified inner fold must partition the complete tuning scope"
                )

    summaries = tuple(
        _summarize_candidate(
            candidate,
            tuple(
                item
                for item in results
                if item.candidate_id == candidate.candidate_id
            ),
            training_subject_ids=subjects,
        )
        for candidate in spec.candidates
    )
    observed = tuple(summary for summary in summaries if summary.status == _OBSERVED)
    best_candidate_id: str | None = None
    selected_candidate_id: str | None = None
    best_mean_loss: float | None = None
    threshold: float | None = None
    candidate_comparisons: tuple[PenaltyCandidateComparison, ...] = ()
    reason_code: str | None = None
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in spec.candidates
    }
    if observed:
        best = min(
            observed,
            key=lambda summary: (
                cast(float, summary.mean_loss),
                -candidate_by_id[summary.candidate_id].lambda1_fraction,
                -candidate_by_id[summary.candidate_id].lambda2_fraction,
                summary.candidate_id,
            ),
        )
        best_candidate_id = best.candidate_id
        best_mean_loss = cast(float, best.mean_loss)
        candidate_comparisons = tuple(
            _compare_candidate_to_best(summary, best) for summary in observed
        )
        comparison_by_candidate = {
            comparison.candidate_id: comparison
            for comparison in candidate_comparisons
        }
        eligible = tuple(
            summary
            for summary in observed
            if comparison_by_candidate[summary.candidate_id].within_one_se
        )
        selected = min(
            eligible,
            key=lambda summary: (
                -candidate_by_id[summary.candidate_id].lambda1_fraction,
                -candidate_by_id[summary.candidate_id].lambda2_fraction,
                summary.candidate_id,
            ),
        )
        selected_candidate_id = selected.candidate_id
        status = "selected"
    else:
        status = (
            _FAILED
            if any(item.status == _FAILED for item in results)
            else _NOT_ESTIMABLE
        )
        reason_code = "no_candidate_complete_inner_coverage"
    self = object.__new__(PenaltyTuningArtifact)
    values: dict[str, Any] = {
        "spec": spec,
        "tuning_scope_id": scope,
        "inner_fold_plan_id": plan_id,
        "validation_loss_estimand": validation_loss_estimand,
        "training_subject_ids": subjects,
        "inner_fold_ids": folds,
        "evaluations": results,
        "summaries": summaries,
        "candidate_comparisons": candidate_comparisons,
        "best_candidate_id": best_candidate_id,
        "selected_candidate_id": selected_candidate_id,
        "best_mean_loss": best_mean_loss,
        "one_se_threshold": threshold,
        "status": status,
        "reason_code": reason_code,
        "certification_status": (
            _VERIFIED_TUNING
            if verification_status == _VERIFIED_EVALUATION
            else _NONCERTIFYING_TUNING
        ),
        "_producer_marker": _TUNING_MARKER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "tuning_id",
        stable_id(
            "penalty_tuning_artifact",
            self._identity_payload(),
            schema_version="2",
        ),
    )
    return self


def not_estimable_penalty_tuning(
    spec: PenaltyTuningSpec,
    *,
    tuning_scope_id: str,
    training_subject_ids: Sequence[str],
    reason_code: str,
    inner_fold_plan_id: str | None = None,
    validation_loss_estimand: PenaltyValidationLossEstimand | str | None = None,
) -> PenaltyTuningArtifact:
    """Create a typed tuning artifact when inner-fold production cannot start."""

    if not isinstance(spec, PenaltyTuningSpec):
        raise TypeError("spec must be a PenaltyTuningSpec")
    spec._require_intact()
    scope = _name(tuning_scope_id, field_name="tuning_scope_id")
    subjects = tuple(
        sorted(
            _names(
                training_subject_ids,
                field_name="training_subject_ids",
                minimum=2,
            )
        )
    )
    reason = _name(reason_code, field_name="reason_code")
    plan_id = (
        None
        if inner_fold_plan_id is None
        else _name(inner_fold_plan_id, field_name="inner_fold_plan_id")
    )
    loss_estimand = (
        None
        if validation_loss_estimand is None
        else _validation_loss_estimand(validation_loss_estimand)
    )
    self = object.__new__(PenaltyTuningArtifact)
    values: dict[str, Any] = {
        "spec": spec,
        "tuning_scope_id": scope,
        "inner_fold_plan_id": plan_id,
        "validation_loss_estimand": loss_estimand,
        "training_subject_ids": subjects,
        "inner_fold_ids": (),
        "evaluations": (),
        "summaries": (),
        "candidate_comparisons": (),
        "best_candidate_id": None,
        "selected_candidate_id": None,
        "best_mean_loss": None,
        "one_se_threshold": None,
        "status": _NOT_ESTIMABLE,
        "reason_code": reason,
        "certification_status": _NOT_ESTIMABLE_TUNING,
        "_producer_marker": _TUNING_MARKER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "tuning_id",
        stable_id(
            "penalty_tuning_artifact",
            self._identity_payload(),
            schema_version="2",
        ),
    )
    return self


__all__ = [
    "PenaltyCandidateComparison",
    "PenaltyCandidateSummary",
    "PenaltyFoldEvaluation",
    "PenaltyScaleResolution",
    "PenaltyTuningArtifact",
    "PenaltyTuningSpec",
    "RelativePenaltyCandidate",
    "ResolvedPenalty",
    "not_estimable_penalty_tuning",
    "record_penalty_fold_evaluation",
    "resolve_penalty_scale",
    "resolve_residualized_penalty_scale",
    "select_penalty_candidate",
]
