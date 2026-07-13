"""Deterministic relative-penalty scaling and subject-equal selection."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy import sparse

from crychic.core import ContractError, stable_id

_SELECTION_RULE = "subject_equal_one_se_strongest_regularization_v1"
_EVALUATION_MARKER = "crychic.attribution.penalty_fold_evaluation.v1"
_SCALE_MARKER = "crychic.attribution.penalty_scale_resolution.v1"
_TUNING_MARKER = "crychic.attribution.penalty_tuning.v1"
_OBSERVED = "observed"
_NOT_ESTIMABLE = "not_estimable"
_FAILED = "failed"
_STATUSES = frozenset({_OBSERVED, _NOT_ESTIMABLE, _FAILED})
_DIRECTIONAL_SCALE_SPACE = "direction_compatible_nonnegative_v1"
_RESIDUAL_SCALE_SPACE = "signed_residual_nonnegative_coefficients_v1"
_SCALE_SPACES = frozenset({_DIRECTIONAL_SCALE_SPACE, _RESIDUAL_SCALE_SPACE})
_CALLER_RECORDED_EVALUATION = "caller_recorded_fold_losses_unverified"
_NONCERTIFYING_TUNING = "caller_recorded_inner_losses_selection_only"


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
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "lambda1_fractions": list(lambda1),
            "lambda2_fractions": list(lambda2),
            "selection_rule": self.selection_rule,
        }
        object.__setattr__(self, "lambda1_fractions", lambda1)
        object.__setattr__(self, "lambda2_fractions", lambda2)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "spec_id",
            stable_id("penalty_tuning_spec", payload, schema_version="1"),
        )

    def _require_intact(self) -> None:
        try:
            expected = PenaltyTuningSpec(
                lambda1_fractions=self.lambda1_fractions,
                lambda2_fractions=self.lambda2_fractions,
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
            "lambda1_fractions": list(self.lambda1_fractions),
            "lambda2_fractions": list(self.lambda2_fractions),
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
    for name, value in values.items():
        object.__setattr__(self, name, value)
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
    validation_subject_ids: tuple[str, ...]
    subject_losses: np.ndarray
    subject_losses_digest: str
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
            "inner_fold_id": self.inner_fold_id,
            "reason_code": self.reason_code,
            "status": self.status,
            "subject_losses_digest": self.subject_losses_digest,
            "validation_subject_ids": list(self.validation_subject_ids),
            "verification_status": self.verification_status,
        }

    def _require_intact(self) -> None:
        try:
            losses = np.asarray(self.subject_losses, dtype="<f8")
            observed = self.status == _OBSERVED
            valid = (
                self._producer_marker == _EVALUATION_MARKER
                and self.verification_status == _CALLER_RECORDED_EVALUATION
                and self.status in _STATUSES
                and observed == (self.reason_code is None)
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
                    schema_version="1",
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
    values = {
        "candidate_id": candidate.candidate_id,
        "inner_fold_id": fold_id,
        "validation_subject_ids": subjects,
        "subject_losses": frozen_losses,
        "subject_losses_digest": _array_digest(frozen_losses),
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
            "penalty_fold_evaluation", self._identity_payload(), schema_version="1"
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
class PenaltyTuningArtifact:
    """Producer-owned one-SE selection with complete candidate/fold coverage."""

    spec: PenaltyTuningSpec
    tuning_scope_id: str
    training_subject_ids: tuple[str, ...]
    inner_fold_ids: tuple[str, ...]
    evaluations: tuple[PenaltyFoldEvaluation, ...]
    summaries: tuple[PenaltyCandidateSummary, ...]
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
        """Return false until typed inner-fold fit/apply parents produce losses."""

        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "best_candidate_id": self.best_candidate_id,
            "best_mean_loss": self.best_mean_loss,
            "certification_status": self.certification_status,
            "evaluation_ids": [item.evaluation_id for item in self.evaluations],
            "inner_fold_ids": list(self.inner_fold_ids),
            "one_se_threshold": self.one_se_threshold,
            "reason_code": self.reason_code,
            "selected_candidate_id": self.selected_candidate_id,
            "selection_rule": self.spec.selection_rule,
            "spec_id": self.spec.spec_id,
            "status": self.status,
            "summary_ids": [item.summary_id for item in self.summaries],
            "training_subject_ids": list(self.training_subject_ids),
            "tuning_scope_id": self.tuning_scope_id,
        }

    def _require_intact(self) -> None:
        try:
            for summary in self.summaries:
                summary._require_intact()
            repeated = select_penalty_candidate(
                self.spec,
                self.evaluations,
                tuning_scope_id=self.tuning_scope_id,
                training_subject_ids=self.training_subject_ids,
                inner_fold_ids=self.inner_fold_ids,
            )
            valid = (
                self._producer_marker == _TUNING_MARKER
                and self.certification_status == _NONCERTIFYING_TUNING
                and not self.is_oof_certified
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
        }


def select_penalty_candidate(
    spec: PenaltyTuningSpec,
    evaluations: Sequence[PenaltyFoldEvaluation],
    *,
    tuning_scope_id: str,
    training_subject_ids: Sequence[str],
    inner_fold_ids: Sequence[str],
) -> PenaltyTuningArtifact:
    """Select the strongest candidate within one SE of minimum subject loss."""

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
        threshold = best_mean_loss + cast(float, best.standard_error)
        eligible = tuple(
            summary
            for summary in observed
            if cast(float, summary.mean_loss) <= threshold
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
        "training_subject_ids": subjects,
        "inner_fold_ids": folds,
        "evaluations": results,
        "summaries": summaries,
        "best_candidate_id": best_candidate_id,
        "selected_candidate_id": selected_candidate_id,
        "best_mean_loss": best_mean_loss,
        "one_se_threshold": threshold,
        "status": status,
        "reason_code": reason_code,
        "certification_status": _NONCERTIFYING_TUNING,
        "_producer_marker": _TUNING_MARKER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "tuning_id",
        stable_id("penalty_tuning_artifact", self._identity_payload()),
    )
    return self


__all__ = [
    "PenaltyCandidateSummary",
    "PenaltyFoldEvaluation",
    "PenaltyScaleResolution",
    "PenaltyTuningArtifact",
    "PenaltyTuningSpec",
    "RelativePenaltyCandidate",
    "ResolvedPenalty",
    "record_penalty_fold_evaluation",
    "resolve_penalty_scale",
    "resolve_residualized_penalty_scale",
    "select_penalty_candidate",
]
