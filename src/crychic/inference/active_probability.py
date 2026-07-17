"""ADR-013 active-null empirical p-values and G3-P-gated probabilities.

This module is deliberately workflow-neutral.  Workflow code owns subject-equal
edge aggregation and full-pipeline null reruns; inference consumes only the
typed, complete candidate-by-plan artifact defined here.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
from scipy.optimize import minimize

from crychic.core import ContractError, stable_id

from .g3p_calibration import (
    G3PCalibrationEvidence,
    G3PCalibrationGate,
    G3PCalibrationScenarioResult,
    G3PGateStatus,
    build_g3p_calibration_gate,
)

_SCHEMA_VERSION = "1.0.0"
_MIN_CANDIDATE_EDGES = 200
_MIN_NULL_PLANS = 200

_PI0_BOUNDS = (0.5, 1.0)
_A_BOUNDS = (0.05, 0.95)
_LOG_LIKELIHOOD_AGREEMENT = 1e-8
_PROJECTED_GRADIENT_LIMIT = 1e-6
_BOUNDARY_TOLERANCE = 1e-7
_CURVATURE_TOLERANCE = 1e-10
_BUM_STARTS = (
    (0.55, 0.10),
    (0.55, 0.50),
    (0.55, 0.90),
    (0.70, 0.10),
    (0.70, 0.50),
    (0.70, 0.90),
    (0.85, 0.10),
    (0.85, 0.50),
    (0.85, 0.90),
    (0.98, 0.20),
    (0.98, 0.80),
)

_BUM_ESTIMATOR = "beta_uniform_mixture_pi0_0.5_1_a_0.05_0.95_v1"
_STRATUM_POLICY = (
    "score_version_view_mode_contrast_context_receiver_no_cross_pooling_v1"
)
_EMPIRICAL_P_SEMANTICS = "matched_edge_right_tail_plus_one_v1"
_PROBABILITY_SEMANTICS = "one_minus_beta_uniform_mixture_local_fdr_v1"

_FIT_PRODUCER = "crychic.inference.active_probability.bum_fit.v1"
_RESULT_PRODUCER = "crychic.inference.active_probability.collection.v1"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_name(value: object, *, field_name: str) -> str | None:
    return None if value is None else _name(value, field_name=field_name)


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if result == 0.0 else result


def _unit(value: object, *, field_name: str) -> float:
    result = _finite(value, field_name=field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1]")
    return result


def _positive_count(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer >= 1")
    return value


def _nonnegative_count(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _contract_error(
    message: str,
    *,
    code: str,
    field: str,
    remediation: str,
) -> ContractError:
    return ContractError(
        message,
        code=code,
        field=field,
        remediation=remediation,
    )


class ActiveEdgeScoreStatus(StrEnum):
    """Availability of one subject-equal point or null edge statistic."""

    OBSERVED = "observed"
    STRUCTURAL_ZERO = "structural_zero"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


def _score_state(
    score: float | None,
    status: ActiveEdgeScoreStatus,
    reason_code: str | None,
) -> tuple[float | None, str | None]:
    if status is ActiveEdgeScoreStatus.OBSERVED:
        value = _unit(score, field_name="score")
        if reason_code is not None:
            raise ValueError("observed edge scores cannot carry a reason_code")
        return value, None
    if status is ActiveEdgeScoreStatus.STRUCTURAL_ZERO:
        value = _unit(score, field_name="score")
        if value != 0.0:
            raise ValueError("structural-zero edge scores must equal zero")
        return 0.0, _name(reason_code, field_name="reason_code")
    if score is not None:
        raise ValueError("not-estimable and failed edge scores must have score=None")
    return None, _name(reason_code, field_name="reason_code")


@dataclass(frozen=True, slots=True, kw_only=True)
class PointActiveEdgeScoreRecord:
    """One subject-equal observed edge statistic from the point workflow."""

    candidate_edge_id: str
    stratum_id: str
    score_version: str
    source_score_collection_id: str
    n_subjects: int
    score: float | None
    status: ActiveEdgeScoreStatus
    reason_code: str | None
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = {
            name: _name(getattr(self, name), field_name=name)
            for name in (
                "candidate_edge_id",
                "stratum_id",
                "score_version",
                "source_score_collection_id",
            )
        }
        status = ActiveEdgeScoreStatus(self.status)
        score, reason = _score_state(self.score, status, self.reason_code)
        n_subjects = _nonnegative_count(self.n_subjects, field_name="n_subjects")
        if (
            status
            in {
                ActiveEdgeScoreStatus.OBSERVED,
                ActiveEdgeScoreStatus.STRUCTURAL_ZERO,
            }
            and n_subjects < 1
        ):
            raise ValueError(
                "observed and structural-zero point scores require n_subjects >= 1"
            )
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "n_subjects", n_subjects)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "point_active_edge_score",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "candidate_edge_id": self.candidate_edge_id,
            "stratum_id": self.stratum_id,
            "score_version": self.score_version,
            "source_score_collection_id": self.source_score_collection_id,
            "n_subjects": self.n_subjects,
            "score": self.score,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "aggregation": "technical_sample_then_repeat_then_equal_subject_mean_v1",
        }

    def _require_intact(self) -> None:
        try:
            repeated = PointActiveEdgeScoreRecord(
                candidate_edge_id=self.candidate_edge_id,
                stratum_id=self.stratum_id,
                score_version=self.score_version,
                source_score_collection_id=self.source_score_collection_id,
                n_subjects=self.n_subjects,
                score=self.score,
                status=self.status,
                reason_code=self.reason_code,
            )
            valid = repeated.record_id == self.record_id
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "Point active-edge score failed integrity validation",
                code="point_active_edge_score_integrity_violation",
                field="record_id",
                remediation="Rebuild the record from the intact point score adapter",
            ) from error
        if not valid:
            raise _contract_error(
                "Point active-edge score failed integrity validation",
                code="point_active_edge_score_integrity_violation",
                field="record_id",
                remediation="Rebuild the record from the intact point score adapter",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"record_id": self.record_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class NullActiveEdgeScoreRecord:
    """One candidate-edge cell from one full-pipeline active-null rerun."""

    candidate_edge_id: str
    plan_id: str
    stratum_id: str
    score_version: str
    null_rerun_record_id: str
    score: float | None
    status: ActiveEdgeScoreStatus
    reason_code: str | None
    source_score_collection_id: str | None = None
    n_subjects: int | None = None
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = {
            name: _name(getattr(self, name), field_name=name)
            for name in (
                "candidate_edge_id",
                "plan_id",
                "stratum_id",
                "score_version",
                "null_rerun_record_id",
            )
        }
        status = ActiveEdgeScoreStatus(self.status)
        score, reason = _score_state(self.score, status, self.reason_code)
        source_collection = _optional_name(
            self.source_score_collection_id,
            field_name="source_score_collection_id",
        )
        if self.n_subjects is None:
            n_subjects: int | None = None
        else:
            n_subjects = _nonnegative_count(self.n_subjects, field_name="n_subjects")
        if status in {
            ActiveEdgeScoreStatus.OBSERVED,
            ActiveEdgeScoreStatus.STRUCTURAL_ZERO,
        } and (source_collection is None or n_subjects is None or n_subjects < 1):
            raise ValueError(
                "observed and structural-zero null scores require collection "
                "and subjects"
            )
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_score_collection_id", source_collection)
        object.__setattr__(self, "n_subjects", n_subjects)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "null_active_edge_score",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "candidate_edge_id": self.candidate_edge_id,
            "plan_id": self.plan_id,
            "stratum_id": self.stratum_id,
            "score_version": self.score_version,
            "null_rerun_record_id": self.null_rerun_record_id,
            "source_score_collection_id": self.source_score_collection_id,
            "n_subjects": self.n_subjects,
            "score": self.score,
            "status": self.status.value,
            "reason_code": self.reason_code,
        }

    def _require_intact(self) -> None:
        try:
            repeated = NullActiveEdgeScoreRecord(
                candidate_edge_id=self.candidate_edge_id,
                plan_id=self.plan_id,
                stratum_id=self.stratum_id,
                score_version=self.score_version,
                null_rerun_record_id=self.null_rerun_record_id,
                source_score_collection_id=self.source_score_collection_id,
                n_subjects=self.n_subjects,
                score=self.score,
                status=self.status,
                reason_code=self.reason_code,
            )
            valid = repeated.record_id == self.record_id
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "Null active-edge score failed integrity validation",
                code="null_active_edge_score_integrity_violation",
                field="record_id",
                remediation="Rebuild the cell from the exact null rerun",
            ) from error
        if not valid:
            raise _contract_error(
                "Null active-edge score failed integrity validation",
                code="null_active_edge_score_integrity_violation",
                field="record_id",
                remediation="Rebuild the cell from the exact null rerun",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"record_id": self.record_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class NullScoreDistribution:
    """Exact candidate-edge by active-null-plan rectangle."""

    active_null_id: str
    active_null_spec_id: str
    candidate_universe_id: str
    candidate_universe_policy_id: str
    score_spec_id: str
    point_records: tuple[PointActiveEdgeScoreRecord, ...]
    plan_ids: tuple[str, ...]
    null_records: tuple[NullActiveEdgeScoreRecord, ...]
    distribution_id: str = field(init=False)

    def __post_init__(self) -> None:
        active_null_id = _name(self.active_null_id, field_name="active_null_id")
        active_null_spec_id = _name(
            self.active_null_spec_id, field_name="active_null_spec_id"
        )
        universe_id = _name(
            self.candidate_universe_id, field_name="candidate_universe_id"
        )
        universe_policy_id = _name(
            self.candidate_universe_policy_id,
            field_name="candidate_universe_policy_id",
        )
        score_spec_id = _name(self.score_spec_id, field_name="score_spec_id")
        points = tuple(self.point_records)
        if not points or any(
            not isinstance(item, PointActiveEdgeScoreRecord) for item in points
        ):
            raise ValueError("point_records must contain typed records")
        for record in points:
            record._require_intact()
        points = tuple(sorted(points, key=lambda item: item.candidate_edge_id))
        edge_ids = tuple(item.candidate_edge_id for item in points)
        if len(edge_ids) != len(set(edge_ids)):
            raise _contract_error(
                "Point active-edge records must be candidate-unique",
                code="active_null_universe_mismatch",
                field="candidate_edge_id",
                remediation="Retain one point record per frozen candidate edge",
            )
        source_collections = {item.source_score_collection_id for item in points}
        if len(source_collections) != 1:
            raise _contract_error(
                "Point records bind multiple score collections",
                code="active_null_source_mismatch",
                field="source_score_collection_id",
                remediation="Bind one unique contrast-common score collection",
            )
        score_versions = {item.score_version for item in points}
        if len(score_versions) != 1:
            raise _contract_error(
                "Point records bind multiple score versions",
                code="active_null_source_mismatch",
                field="score_version",
                remediation="Freeze a separate active universe per score version",
            )
        plans = tuple(_name(item, field_name="plan_id") for item in self.plan_ids)
        if not plans or len(plans) != len(set(plans)):
            raise ValueError("plan_ids must contain unique active-null plan IDs")
        plans = tuple(sorted(plans))
        nulls = tuple(self.null_records)
        if any(not isinstance(item, NullActiveEdgeScoreRecord) for item in nulls):
            raise TypeError("null_records must contain typed null score records")
        for null_record in nulls:
            null_record._require_intact()
        nulls = tuple(
            sorted(nulls, key=lambda item: (item.candidate_edge_id, item.plan_id))
        )
        observed_pairs = tuple((item.candidate_edge_id, item.plan_id) for item in nulls)
        expected_pairs = tuple(
            (edge_id, plan_id) for edge_id in edge_ids for plan_id in plans
        )
        if observed_pairs != expected_pairs:
            raise _contract_error(
                "Null scores do not form the exact candidate-by-plan rectangle",
                code="active_null_incomplete_score_matrix",
                field="candidate_edge_id,plan_id",
                remediation="Emit one typed cell for every candidate and null plan",
            )
        point_by_edge = {item.candidate_edge_id: item for item in points}
        for null_record in nulls:
            point = point_by_edge[null_record.candidate_edge_id]
            if (
                null_record.stratum_id != point.stratum_id
                or null_record.score_version != point.score_version
            ):
                raise _contract_error(
                    "Null cell stratum or score version differs from its point edge",
                    code="active_null_source_mismatch",
                    field="stratum_id,score_version",
                    remediation="Adapt point and null scores through one frozen spec",
                )
        for plan_id in plans:
            plan_rows = tuple(item for item in nulls if item.plan_id == plan_id)
            rerun_ids = {item.null_rerun_record_id for item in plan_rows}
            if len(rerun_ids) != 1:
                raise _contract_error(
                    "One null plan binds multiple rerun records",
                    code="active_null_source_mismatch",
                    field="null_rerun_record_id",
                    remediation="Materialize every plan from one full-pipeline rerun",
                )
            observed_collections = {
                item.source_score_collection_id
                for item in plan_rows
                if item.source_score_collection_id is not None
            }
            if len(observed_collections) > 1:
                raise _contract_error(
                    "One null plan binds multiple score collections",
                    code="active_null_source_mismatch",
                    field="source_score_collection_id",
                    remediation=(
                        "Use one contrast-common child collection per null plan"
                    ),
                )
        object.__setattr__(self, "active_null_id", active_null_id)
        object.__setattr__(self, "active_null_spec_id", active_null_spec_id)
        object.__setattr__(self, "candidate_universe_id", universe_id)
        object.__setattr__(self, "candidate_universe_policy_id", universe_policy_id)
        object.__setattr__(self, "score_spec_id", score_spec_id)
        object.__setattr__(self, "point_records", points)
        object.__setattr__(self, "plan_ids", plans)
        object.__setattr__(self, "null_records", nulls)
        object.__setattr__(
            self,
            "distribution_id",
            stable_id(
                "active_null_score_distribution",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    @property
    def score_version(self) -> str:
        return self.point_records[0].score_version

    @property
    def source_score_collection_id(self) -> str:
        return self.point_records[0].source_score_collection_id

    @property
    def candidate_edge_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_edge_id for item in self.point_records)

    @property
    def complete_rectangle(self) -> bool:
        return True

    def _identity_payload(self) -> dict[str, object]:
        return {
            "active_null_id": self.active_null_id,
            "active_null_spec_id": self.active_null_spec_id,
            "candidate_universe_id": self.candidate_universe_id,
            "candidate_universe_policy_id": self.candidate_universe_policy_id,
            "score_spec_id": self.score_spec_id,
            "point_record_ids": [item.record_id for item in self.point_records],
            "plan_ids": list(self.plan_ids),
            "null_record_ids": [item.record_id for item in self.null_records],
            "complete_rectangle": True,
            "null_semantics": "degree_evidence_matched_lr_target_reassignment_v1",
        }

    def _require_intact(self) -> None:
        try:
            for point_record in self.point_records:
                point_record._require_intact()
            for null_record in self.null_records:
                null_record._require_intact()
            repeated = NullScoreDistribution(
                active_null_id=self.active_null_id,
                active_null_spec_id=self.active_null_spec_id,
                candidate_universe_id=self.candidate_universe_id,
                candidate_universe_policy_id=self.candidate_universe_policy_id,
                score_spec_id=self.score_spec_id,
                point_records=self.point_records,
                plan_ids=self.plan_ids,
                null_records=self.null_records,
            )
            valid = repeated.distribution_id == self.distribution_id
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "Active-null distribution failed integrity validation",
                code="active_null_distribution_integrity_violation",
                field="distribution_id",
                remediation="Rebuild it from the exact point and null adapters",
            ) from error
        if not valid:
            raise _contract_error(
                "Active-null distribution failed integrity validation",
                code="active_null_distribution_integrity_violation",
                field="distribution_id",
                remediation="Rebuild it from the exact point and null adapters",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "distribution_id": self.distribution_id,
            **self._identity_payload(),
            "point_records": [item.to_dict() for item in self.point_records],
            "null_records": [item.to_dict() for item in self.null_records],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveProbabilitySpec:
    """The only ADR-013 v1 runtime BUM specification."""

    minimum_candidate_edges: int = _MIN_CANDIDATE_EDGES
    minimum_null_plans: int = _MIN_NULL_PLANS
    estimator: str = _BUM_ESTIMATOR
    stratum_policy: str = _STRATUM_POLICY
    spec_id: str = field(init=False)
    estimator_id: str = field(init=False)
    stratum_policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.minimum_candidate_edges != _MIN_CANDIDATE_EDGES:
            raise ValueError("ADR-013 fixes minimum_candidate_edges=200")
        if self.minimum_null_plans != _MIN_NULL_PLANS:
            raise ValueError("ADR-013 fixes minimum_null_plans=200")
        if self.estimator != _BUM_ESTIMATOR:
            raise ValueError("ADR-013 permits only the frozen BUM estimator")
        if self.stratum_policy != _STRATUM_POLICY:
            raise ValueError("ADR-013 permits only the frozen stratum policy")
        estimator_id = stable_id(
            "active_probability_estimator",
            {
                "estimator": self.estimator,
                "pi0_bounds": list(_PI0_BOUNDS),
                "a_bounds": list(_A_BOUNDS),
                "starts": [list(item) for item in _BUM_STARTS],
                "log_likelihood_agreement": _LOG_LIKELIHOOD_AGREEMENT,
                "projected_gradient_limit": _PROJECTED_GRADIENT_LIMIT,
            },
            schema_version=_SCHEMA_VERSION,
        )
        stratum_policy_id = stable_id(
            "active_probability_stratum_policy",
            {"stratum_policy": self.stratum_policy},
            schema_version=_SCHEMA_VERSION,
        )
        object.__setattr__(self, "estimator_id", estimator_id)
        object.__setattr__(self, "stratum_policy_id", stratum_policy_id)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "active_probability_spec",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "minimum_candidate_edges": self.minimum_candidate_edges,
            "minimum_null_plans": self.minimum_null_plans,
            "estimator": self.estimator,
            "estimator_id": self.estimator_id,
            "stratum_policy": self.stratum_policy,
            "stratum_policy_id": self.stratum_policy_id,
            "empirical_p_semantics": _EMPIRICAL_P_SEMANTICS,
            "probability_semantics": _PROBABILITY_SEMANTICS,
        }

    def _require_intact(self) -> None:
        try:
            repeated = ActiveProbabilitySpec(
                minimum_candidate_edges=self.minimum_candidate_edges,
                minimum_null_plans=self.minimum_null_plans,
                estimator=self.estimator,
                stratum_policy=self.stratum_policy,
            )
            valid = (
                repeated.spec_id == self.spec_id
                and repeated.estimator_id == self.estimator_id
                and repeated.stratum_policy_id == self.stratum_policy_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "Active-probability spec failed integrity validation",
                code="active_probability_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the frozen ADR-013 specification",
            ) from error
        if not valid:
            raise _contract_error(
                "Active-probability spec failed integrity validation",
                code="active_probability_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the frozen ADR-013 specification",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"spec_id": self.spec_id, **self._identity_payload()}


class LocalFDRFitStatus(StrEnum):
    """Runtime availability of one stratum's BUM fit."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, init=False)
class LocalFDRStratumDiagnostics:
    """Producer-owned deterministic BUM fit and runtime support diagnostics."""

    stratum_id: str
    status: LocalFDRFitStatus
    reason_code: str | None
    n_candidate_edges: int
    n_null_plans: int
    complete_score_matrix: bool
    pi0: float | None
    a: float | None
    log_likelihood: float | None
    projected_gradient_norm: float | None
    curvature_minimum_eigenvalue: float | None
    n_starts: int
    n_successful_starts: int
    log_likelihood_spread: float | None
    diagnostic_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError("Use active-probability inference to create diagnostics")

    @classmethod
    def _from_values(cls, **values: object) -> LocalFDRStratumDiagnostics:
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_producer_marker", _FIT_PRODUCER)
        object.__setattr__(
            self,
            "diagnostic_id",
            stable_id(
                "local_fdr_stratum_diagnostic",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "stratum_id": self.stratum_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_candidate_edges": self.n_candidate_edges,
            "n_null_plans": self.n_null_plans,
            "complete_score_matrix": self.complete_score_matrix,
            "pi0": self.pi0,
            "a": self.a,
            "log_likelihood": self.log_likelihood,
            "projected_gradient_norm": self.projected_gradient_norm,
            "curvature_minimum_eigenvalue": self.curvature_minimum_eigenvalue,
            "n_starts": self.n_starts,
            "n_successful_starts": self.n_successful_starts,
            "log_likelihood_spread": self.log_likelihood_spread,
            "estimator": _BUM_ESTIMATOR,
            "producer_marker": _FIT_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            expected_id = stable_id(
                "local_fdr_stratum_diagnostic",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _FIT_PRODUCER
                and self.diagnostic_id == expected_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "Local-FDR diagnostic failed integrity validation",
                code="local_fdr_diagnostic_integrity_violation",
                field="diagnostic_id",
                remediation="Recompute from the intact active-null distribution",
            ) from error
        if not valid:
            raise _contract_error(
                "Local-FDR diagnostic failed integrity validation",
                code="local_fdr_diagnostic_integrity_violation",
                field="diagnostic_id",
                remediation="Recompute from the intact active-null distribution",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"diagnostic_id": self.diagnostic_id, **payload}


def _bum_terms(
    parameters: np.ndarray,
    log_p: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    pi0 = float(parameters[0])
    a = float(parameters[1])
    beta_density = np.exp(math.log(a) + (a - 1.0) * log_p)
    density = pi0 + (1.0 - pi0) * beta_density
    if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        return -math.inf, np.full(2, math.nan), np.full((2, 2), math.nan)
    log_likelihood = float(np.log(density).sum())
    d_pi = 1.0 - beta_density
    h = (1.0 / a) + log_p
    d_a = (1.0 - pi0) * beta_density * h
    gradient = np.asarray(
        [float((d_pi / density).sum()), float((d_a / density).sum())],
        dtype=float,
    )
    d_pi_a = -beta_density * h
    d_a_a = (1.0 - pi0) * beta_density * (h * h - (1.0 / (a * a)))
    hessian = np.asarray(
        [
            [
                float((-(d_pi * d_pi) / (density * density)).sum()),
                float(
                    ((d_pi_a / density) - ((d_pi * d_a) / (density * density))).sum()
                ),
            ],
            [
                0.0,
                float(((d_a_a / density) - ((d_a * d_a) / (density * density))).sum()),
            ],
        ],
        dtype=float,
    )
    hessian[1, 0] = hessian[0, 1]
    return log_likelihood, gradient, hessian


def _projected_gradient_norm(parameters: np.ndarray, gradient: np.ndarray) -> float:
    projected = gradient.copy()
    bounds = (_PI0_BOUNDS, _A_BOUNDS)
    for index, (lower, upper) in enumerate(bounds):
        value = float(parameters[index])
        component = float(gradient[index])
        if value <= lower + _BOUNDARY_TOLERANCE:
            projected[index] = max(component, 0.0)
        elif value >= upper - _BOUNDARY_TOLERANCE:
            projected[index] = min(component, 0.0)
    return float(np.linalg.norm(projected, ord=np.inf))


def _empty_fit_diagnostic(
    *,
    stratum_id: str,
    reason_code: str,
    n_candidate_edges: int,
    n_null_plans: int,
    complete_score_matrix: bool,
) -> LocalFDRStratumDiagnostics:
    return LocalFDRStratumDiagnostics._from_values(
        stratum_id=stratum_id,
        status=LocalFDRFitStatus.NOT_ESTIMABLE,
        reason_code=reason_code,
        n_candidate_edges=n_candidate_edges,
        n_null_plans=n_null_plans,
        complete_score_matrix=complete_score_matrix,
        pi0=None,
        a=None,
        log_likelihood=None,
        projected_gradient_norm=None,
        curvature_minimum_eigenvalue=None,
        n_starts=len(_BUM_STARTS),
        n_successful_starts=0,
        log_likelihood_spread=None,
    )


def fit_beta_uniform_mixture(
    p_values: Sequence[float],
    *,
    stratum_id: str,
    n_null_plans: int,
) -> LocalFDRStratumDiagnostics:
    """Fit the frozen ADR-013 BUM by deterministic constrained multistart MLE."""

    identifier = _name(stratum_id, field_name="stratum_id")
    plans = _positive_count(n_null_plans, field_name="n_null_plans")
    values = np.asarray(tuple(p_values), dtype=float)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("BUM p_values must be finite and lie in (0, 1]")
    # Canonical order makes the floating-point reduction and stable diagnostic ID
    # independent of adapter row order.
    log_p = np.log(np.sort(values, kind="stable"))

    def objective(parameters: np.ndarray) -> float:
        likelihood, _, _ = _bum_terms(parameters, log_p)
        return -likelihood if math.isfinite(likelihood) else math.inf

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        _, gradient, _ = _bum_terms(parameters, log_p)
        return -gradient

    outcomes: list[tuple[np.ndarray, float]] = []
    for start in _BUM_STARTS:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            jac=jacobian,
            method="SLSQP",
            bounds=(_PI0_BOUNDS, _A_BOUNDS),
            options={"ftol": 1e-14, "maxiter": 10_000},
        )
        parameters = np.asarray(result.x, dtype=float)
        likelihood, _, _ = _bum_terms(parameters, log_p)
        if bool(result.success) and math.isfinite(likelihood):
            outcomes.append((parameters, likelihood))
    if len(outcomes) != len(_BUM_STARTS):
        return LocalFDRStratumDiagnostics._from_values(
            stratum_id=identifier,
            status=LocalFDRFitStatus.FAILED,
            reason_code="local_fdr_nonconvergent",
            n_candidate_edges=len(values),
            n_null_plans=plans,
            complete_score_matrix=True,
            pi0=None,
            a=None,
            log_likelihood=None,
            projected_gradient_norm=None,
            curvature_minimum_eigenvalue=None,
            n_starts=len(_BUM_STARTS),
            n_successful_starts=len(outcomes),
            log_likelihood_spread=None,
        )
    likelihoods = np.asarray([item[1] for item in outcomes], dtype=float)
    spread = float(likelihoods.max() - likelihoods.min())
    maximum_likelihood = float(likelihoods.max())
    tied = tuple(
        item
        for item in outcomes
        if maximum_likelihood - item[1] <= _LOG_LIKELIHOOD_AGREEMENT
    )

    def outcome_projected_norm(item: tuple[np.ndarray, float]) -> float:
        _, gradient, _ = _bum_terms(item[0], log_p)
        return _projected_gradient_norm(item[0], gradient)

    selected_parameters, selected_likelihood = min(tied, key=outcome_projected_norm)
    if spread > _LOG_LIKELIHOOD_AGREEMENT:
        return LocalFDRStratumDiagnostics._from_values(
            stratum_id=identifier,
            status=LocalFDRFitStatus.FAILED,
            reason_code="local_fdr_nonidentifiable",
            n_candidate_edges=len(values),
            n_null_plans=plans,
            complete_score_matrix=True,
            pi0=None,
            a=None,
            log_likelihood=selected_likelihood,
            projected_gradient_norm=None,
            curvature_minimum_eigenvalue=None,
            n_starts=len(_BUM_STARTS),
            n_successful_starts=len(outcomes),
            log_likelihood_spread=spread,
        )
    pi0 = float(selected_parameters[0])
    a = float(selected_parameters[1])
    null_only = pi0 >= _PI0_BOUNDS[1] - _BOUNDARY_TOLERANCE
    if null_only:
        selected_parameters = np.asarray([1.0, 0.5], dtype=float)
        pi0, a = 1.0, 0.5
        selected_likelihood, gradient, _ = _bum_terms(selected_parameters, log_p)
        projected_norm = _projected_gradient_norm(selected_parameters, gradient)
        curvature_minimum: float | None = None
    else:
        selected_likelihood, gradient, hessian = _bum_terms(selected_parameters, log_p)
        projected_norm = _projected_gradient_norm(selected_parameters, gradient)
        information = -hessian
        if np.any(~np.isfinite(information)):
            curvature_minimum = math.nan
        else:
            curvature_minimum = float(np.linalg.eigvalsh(information).min())
    reason: str | None = None
    status = LocalFDRFitStatus.OBSERVED
    if not null_only and (
        pi0 <= _PI0_BOUNDS[0] + _BOUNDARY_TOLERANCE
        or a <= _A_BOUNDS[0] + _BOUNDARY_TOLERANCE
        or a >= _A_BOUNDS[1] - _BOUNDARY_TOLERANCE
    ):
        status = LocalFDRFitStatus.NOT_ESTIMABLE
        reason = "local_fdr_parameter_boundary"
    elif (
        not math.isfinite(projected_norm) or projected_norm > _PROJECTED_GRADIENT_LIMIT
    ):
        status = LocalFDRFitStatus.FAILED
        reason = "local_fdr_nonconvergent"
    elif not null_only and (
        curvature_minimum is None
        or not math.isfinite(curvature_minimum)
        or curvature_minimum <= _CURVATURE_TOLERANCE
    ):
        status = LocalFDRFitStatus.NOT_ESTIMABLE
        reason = "local_fdr_nonidentifiable"
    return LocalFDRStratumDiagnostics._from_values(
        stratum_id=identifier,
        status=status,
        reason_code=reason,
        n_candidate_edges=len(values),
        n_null_plans=plans,
        complete_score_matrix=True,
        pi0=pi0,
        a=a,
        log_likelihood=selected_likelihood,
        projected_gradient_norm=projected_norm,
        curvature_minimum_eigenvalue=curvature_minimum,
        n_starts=len(_BUM_STARTS),
        n_successful_starts=len(outcomes),
        log_likelihood_spread=spread,
    )


@dataclass(frozen=True, slots=True, init=False)
class ActiveProbabilityRecord:
    """One candidate value and separately gated released probability."""

    candidate_edge_id: str
    point_record_id: str
    stratum_id: str
    score_version: str
    point_score: float | None
    point_status: ActiveEdgeScoreStatus
    point_reason_code: str | None
    active_null_empirical_p_value: float | None
    candidate_local_fdr: float | None
    candidate_comm_probability: float | None
    comm_probability: float | None
    probability_reason_code: str | None
    local_fdr_diagnostic_id: str
    calibration_gate_id: str | None
    record_id: str

    def __init__(self) -> None:
        raise TypeError("Use estimate_active_probabilities()")

    @classmethod
    def _from_values(cls, **values: object) -> ActiveProbabilityRecord:
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "active_probability_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    @property
    def released(self) -> bool:
        return self.comm_probability is not None

    def _identity_payload(self) -> dict[str, object]:
        return {
            "candidate_edge_id": self.candidate_edge_id,
            "point_record_id": self.point_record_id,
            "stratum_id": self.stratum_id,
            "score_version": self.score_version,
            "point_score": self.point_score,
            "point_status": self.point_status.value,
            "point_reason_code": self.point_reason_code,
            "active_null_empirical_p_value": self.active_null_empirical_p_value,
            "candidate_local_fdr": self.candidate_local_fdr,
            "candidate_comm_probability": self.candidate_comm_probability,
            "comm_probability": self.comm_probability,
            "probability_reason_code": self.probability_reason_code,
            "local_fdr_diagnostic_id": self.local_fdr_diagnostic_id,
            "calibration_gate_id": self.calibration_gate_id,
            "empirical_p_semantics": _EMPIRICAL_P_SEMANTICS,
            "probability_semantics": _PROBABILITY_SEMANTICS,
        }

    def _require_intact(self) -> None:
        expected_id = stable_id(
            "active_probability_record",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        )
        if self.record_id != expected_id:
            raise _contract_error(
                "Active-probability record failed integrity validation",
                code="active_probability_record_integrity_violation",
                field="record_id",
                remediation="Recompute from the intact active-null distribution",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"record_id": self.record_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True)
class _Evaluation:
    records: tuple[ActiveProbabilityRecord, ...]
    diagnostics: tuple[LocalFDRStratumDiagnostics, ...]
    empirical_p_by_edge: Mapping[str, float | None]


def _empirical_p(
    point: PointActiveEdgeScoreRecord,
    nulls: Sequence[NullActiveEdgeScoreRecord],
) -> float | None:
    if point.score is None:
        return None
    values = tuple(item.score for item in nulls)
    if any(value is None for value in values):
        return None
    observed = point.score
    numeric = cast(tuple[float, ...], values)
    return (1.0 + sum(value >= observed for value in numeric)) / (len(numeric) + 1.0)


def _candidate_probability(
    p_value: float,
    diagnostic: LocalFDRStratumDiagnostics,
) -> tuple[float, float]:
    if diagnostic.status is not LocalFDRFitStatus.OBSERVED:
        raise ValueError("candidate probability requires an observed BUM fit")
    if diagnostic.pi0 is None or diagnostic.a is None:
        raise ValueError("observed BUM fit requires finite pi0 and a parameters")
    if diagnostic.pi0 == 1.0:
        return 1.0, 0.0
    pi0 = diagnostic.pi0
    a = diagnostic.a
    alternative = (1.0 - pi0) * a * math.exp((a - 1.0) * math.log(p_value))
    local_fdr = pi0 / (pi0 + alternative)
    local_fdr = min(max(local_fdr, 0.0), 1.0)
    return local_fdr, min(max(1.0 - local_fdr, 0.0), 1.0)


def _runtime_diagnostic(
    *,
    stratum_id: str,
    point_records: tuple[PointActiveEdgeScoreRecord, ...],
    null_records: tuple[NullActiveEdgeScoreRecord, ...],
    plan_ids: tuple[str, ...],
    empirical: Mapping[str, float | None],
    spec: ActiveProbabilitySpec,
) -> LocalFDRStratumDiagnostics:
    complete = all(
        item.status
        in {ActiveEdgeScoreStatus.OBSERVED, ActiveEdgeScoreStatus.STRUCTURAL_ZERO}
        for item in null_records
    )
    candidates = tuple(
        item
        for item in point_records
        if item.status is ActiveEdgeScoreStatus.OBSERVED
        and empirical[item.candidate_edge_id] is not None
    )
    if not complete:
        return _empty_fit_diagnostic(
            stratum_id=stratum_id,
            reason_code="active_null_incomplete_score_matrix",
            n_candidate_edges=len(candidates),
            n_null_plans=len(plan_ids),
            complete_score_matrix=False,
        )
    if len(plan_ids) < spec.minimum_null_plans:
        return _empty_fit_diagnostic(
            stratum_id=stratum_id,
            reason_code="active_null_insufficient_plans",
            n_candidate_edges=len(candidates),
            n_null_plans=len(plan_ids),
            complete_score_matrix=True,
        )
    if len(candidates) < spec.minimum_candidate_edges:
        return _empty_fit_diagnostic(
            stratum_id=stratum_id,
            reason_code="active_null_stratum_too_small",
            n_candidate_edges=len(candidates),
            n_null_plans=len(plan_ids),
            complete_score_matrix=True,
        )
    p_values = cast(
        tuple[float, ...],
        tuple(empirical[item.candidate_edge_id] for item in candidates),
    )
    return fit_beta_uniform_mixture(
        p_values,
        stratum_id=stratum_id,
        n_null_plans=len(plan_ids),
    )


def _validate_gate_binding(
    distribution: NullScoreDistribution,
    spec: ActiveProbabilitySpec,
    gate: G3PCalibrationGate | None,
) -> None:
    if gate is None:
        return
    if not isinstance(gate, G3PCalibrationGate):
        raise TypeError("calibration_gate must be producer-owned G3PCalibrationGate")
    gate._require_intact()
    if (
        gate.active_null_spec_id != distribution.active_null_spec_id
        or gate.active_probability_spec_id != spec.spec_id
        or gate.score_spec_id != distribution.score_spec_id
        or gate.candidate_universe_policy_id
        != distribution.candidate_universe_policy_id
        or gate.estimator_id != spec.estimator_id
        or gate.stratum_policy_id != spec.stratum_policy_id
        or gate.score_version != distribution.score_version
    ):
        raise _contract_error(
            "G3-P gate does not bind the runtime active-null contracts",
            code="g3p_gate_runtime_binding_mismatch",
            field=(
                "active_null_spec_id,active_probability_spec_id,score_spec_id,"
                "candidate_universe_policy_id,estimator_id,stratum_policy_id,"
                "score_version"
            ),
            remediation="Use calibration evidence for the exact runtime contracts",
        )


def _evaluate(
    distribution: NullScoreDistribution,
    spec: ActiveProbabilitySpec,
    gate: G3PCalibrationGate | None,
) -> _Evaluation:
    points_by_stratum: dict[str, list[PointActiveEdgeScoreRecord]] = {}
    nulls_by_stratum: dict[str, list[NullActiveEdgeScoreRecord]] = {}
    nulls_by_edge: dict[str, list[NullActiveEdgeScoreRecord]] = {}
    for point in distribution.point_records:
        points_by_stratum.setdefault(point.stratum_id, []).append(point)
    for null in distribution.null_records:
        nulls_by_stratum.setdefault(null.stratum_id, []).append(null)
        nulls_by_edge.setdefault(null.candidate_edge_id, []).append(null)
    empirical: dict[str, float | None] = {}
    for stratum_id, points in points_by_stratum.items():
        matrix_complete = all(
            item.status
            in {ActiveEdgeScoreStatus.OBSERVED, ActiveEdgeScoreStatus.STRUCTURAL_ZERO}
            for item in nulls_by_stratum[stratum_id]
        )
        for point in points:
            empirical[point.candidate_edge_id] = (
                _empirical_p(point, nulls_by_edge[point.candidate_edge_id])
                if matrix_complete
                else None
            )
    diagnostics = tuple(
        _runtime_diagnostic(
            stratum_id=stratum_id,
            point_records=tuple(points_by_stratum[stratum_id]),
            null_records=tuple(nulls_by_stratum[stratum_id]),
            plan_ids=distribution.plan_ids,
            empirical=empirical,
            spec=spec,
        )
        for stratum_id in sorted(points_by_stratum)
    )
    diagnostic_by_stratum = {item.stratum_id: item for item in diagnostics}
    records: list[ActiveProbabilityRecord] = []
    for point in distribution.point_records:
        diagnostic = diagnostic_by_stratum[point.stratum_id]
        empirical_p = empirical[point.candidate_edge_id]
        candidate_lfdr: float | None = None
        candidate_probability: float | None = None
        if (
            diagnostic.status is LocalFDRFitStatus.OBSERVED
            and empirical_p is not None
            and point.status
            in {ActiveEdgeScoreStatus.OBSERVED, ActiveEdgeScoreStatus.STRUCTURAL_ZERO}
        ):
            if point.status is ActiveEdgeScoreStatus.STRUCTURAL_ZERO:
                candidate_lfdr, candidate_probability = 1.0, 0.0
            else:
                candidate_lfdr, candidate_probability = _candidate_probability(
                    empirical_p, diagnostic
                )
        gate_passed = bool(gate is not None and gate.comm_probability_release_allowed)
        comm_probability = candidate_probability if gate_passed else None
        if point.status in {
            ActiveEdgeScoreStatus.NOT_ESTIMABLE,
            ActiveEdgeScoreStatus.FAILED,
        }:
            reason = point.reason_code
        elif diagnostic.status is not LocalFDRFitStatus.OBSERVED:
            reason = diagnostic.reason_code
        elif not gate_passed:
            reason = (
                "g3p_gate_not_passed"
                if gate is None or gate.reason_code is None
                else gate.reason_code
            )
        else:
            reason = None
        records.append(
            ActiveProbabilityRecord._from_values(
                candidate_edge_id=point.candidate_edge_id,
                point_record_id=point.record_id,
                stratum_id=point.stratum_id,
                score_version=point.score_version,
                point_score=point.score,
                point_status=point.status,
                point_reason_code=point.reason_code,
                active_null_empirical_p_value=empirical_p,
                candidate_local_fdr=candidate_lfdr,
                candidate_comm_probability=candidate_probability,
                comm_probability=comm_probability,
                probability_reason_code=reason,
                local_fdr_diagnostic_id=diagnostic.diagnostic_id,
                calibration_gate_id=None if gate is None else gate.gate_id,
            )
        )
    return _Evaluation(
        records=tuple(records),
        diagnostics=diagnostics,
        empirical_p_by_edge=empirical,
    )


@dataclass(frozen=True, slots=True, init=False)
class ActiveProbabilityCollection:
    """Producer-owned exact-universe candidate and released probability result."""

    distribution_id: str
    active_null_id: str
    candidate_universe_id: str
    score_spec_id: str
    active_probability_spec_id: str
    calibration_gate_id: str | None
    calibration_gate_status: G3PGateStatus | None
    records: tuple[ActiveProbabilityRecord, ...]
    diagnostics: tuple[LocalFDRStratumDiagnostics, ...]
    status_counts: tuple[tuple[str, int], ...]
    collection_id: str
    _distribution: NullScoreDistribution
    _spec: ActiveProbabilitySpec
    _gate: G3PCalibrationGate | None
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError("Use estimate_active_probabilities()")

    @property
    def comm_probability_release_allowed(self) -> bool:
        return bool(
            self._gate is not None and self._gate.comm_probability_release_allowed
        )

    @property
    def complete_candidate_coverage(self) -> bool:
        return True

    def _identity_payload(self) -> dict[str, object]:
        return {
            "distribution_id": self.distribution_id,
            "active_null_id": self.active_null_id,
            "candidate_universe_id": self.candidate_universe_id,
            "score_spec_id": self.score_spec_id,
            "active_probability_spec_id": self.active_probability_spec_id,
            "calibration_gate_id": self.calibration_gate_id,
            "calibration_gate_status": (
                None
                if self.calibration_gate_status is None
                else self.calibration_gate_status.value
            ),
            "record_ids": [item.record_id for item in self.records],
            "diagnostic_ids": [item.diagnostic_id for item in self.diagnostics],
            "status_counts": [list(item) for item in self.status_counts],
            "complete_candidate_coverage": True,
            "comm_probability_release_allowed": (self.comm_probability_release_allowed),
            "producer_marker": _RESULT_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            self._distribution._require_intact()
            self._spec._require_intact()
            if self._gate is not None:
                self._gate._require_intact()
            expected = _evaluate(self._distribution, self._spec, self._gate)
            for record in self.records:
                record._require_intact()
            for diagnostic in self.diagnostics:
                diagnostic._require_intact()
            expected_id = stable_id(
                "active_probability_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _RESULT_PRODUCER
                and tuple(item.record_id for item in self.records)
                == tuple(item.record_id for item in expected.records)
                and tuple(item.diagnostic_id for item in self.diagnostics)
                == tuple(item.diagnostic_id for item in expected.diagnostics)
                and self.collection_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "Active-probability collection failed integrity validation",
                code="active_probability_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute from the exact active-null distribution",
            ) from error
        if not valid:
            raise _contract_error(
                "Active-probability collection failed integrity validation",
                code="active_probability_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute from the exact active-null distribution",
            )

    def result_for(self, candidate_edge_id: str) -> ActiveProbabilityRecord:
        self._require_intact()
        identifier = _name(candidate_edge_id, field_name="candidate_edge_id")
        matched = tuple(
            item for item in self.records if item.candidate_edge_id == identifier
        )
        if len(matched) != 1:
            raise KeyError(identifier)
        return matched[0]

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "collection_id": self.collection_id,
            **payload,
            "records": [item.to_dict() for item in self.records],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def estimate_active_probabilities(
    distribution: NullScoreDistribution,
    *,
    spec: ActiveProbabilitySpec,
    calibration_gate: G3PCalibrationGate | None = None,
) -> ActiveProbabilityCollection:
    """Compute p fallback, candidate BUM values, and separately gated release."""

    if not isinstance(distribution, NullScoreDistribution):
        raise TypeError("distribution must be NullScoreDistribution")
    if not isinstance(spec, ActiveProbabilitySpec):
        raise TypeError("spec must be ActiveProbabilitySpec")
    distribution._require_intact()
    spec._require_intact()
    _validate_gate_binding(distribution, spec, calibration_gate)
    evaluation = _evaluate(distribution, spec, calibration_gate)
    counts = Counter(item.point_status.value for item in evaluation.records)
    self = object.__new__(ActiveProbabilityCollection)
    values: dict[str, object] = {
        "distribution_id": distribution.distribution_id,
        "active_null_id": distribution.active_null_id,
        "candidate_universe_id": distribution.candidate_universe_id,
        "score_spec_id": distribution.score_spec_id,
        "active_probability_spec_id": spec.spec_id,
        "calibration_gate_id": (
            None if calibration_gate is None else calibration_gate.gate_id
        ),
        "calibration_gate_status": (
            None if calibration_gate is None else calibration_gate.status
        ),
        "records": evaluation.records,
        "diagnostics": evaluation.diagnostics,
        "status_counts": tuple(sorted(counts.items())),
        "_distribution": distribution,
        "_spec": spec,
        "_gate": calibration_gate,
        "_producer_marker": _RESULT_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "collection_id",
        stable_id(
            "active_probability_collection",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    return self


__all__ = [
    "ActiveEdgeScoreStatus",
    "ActiveProbabilityCollection",
    "ActiveProbabilityRecord",
    "ActiveProbabilitySpec",
    "G3PCalibrationEvidence",
    "G3PCalibrationGate",
    "G3PCalibrationScenarioResult",
    "G3PGateStatus",
    "LocalFDRFitStatus",
    "LocalFDRStratumDiagnostics",
    "NullActiveEdgeScoreRecord",
    "NullScoreDistribution",
    "PointActiveEdgeScoreRecord",
    "build_g3p_calibration_gate",
    "estimate_active_probabilities",
    "fit_beta_uniform_mixture",
]
