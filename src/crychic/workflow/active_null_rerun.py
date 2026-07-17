"""Bounded full-pipeline ADR-013 active-null reruns.

The point workflow freezes the candidate universe.  Every null opportunity
then replaces only the target-prior content and reruns the same raw-input
cross-fit.  Planning, materialization, child-workflow, or score-adaptation
failures are represented by a complete typed candidate column.

Parallel execution uses threads so every worker shares the one immutable raw
snapshot and read-only resource objects.  Each active worker still allocates
its own fold-local cross-fit matrices and model state, so peak memory can
approach ``effective_n_jobs`` times the working memory of one null cross-fit.
Users must budget that memory and avoid multiplying thread-pool concurrency by
large BLAS/OpenMP thread counts.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial

from anndata import AnnData

from crychic.core import (
    ContractError,
    CrychicConfig,
    CrychicError,
    SeedLineage,
    stable_id,
)
from crychic.inference import (
    ActiveEdgeScoreStatus,
    NullActiveEdgeScoreRecord,
    NullScoreDistribution,
    PointActiveEdgeScoreRecord,
)
from crychic.resampling import (
    ActiveNullPlan,
    ActiveNullPlanStatus,
    ActiveNullSpec,
    materialize_active_null_target_prior,
    plan_active_null_target_prior,
)
from crychic.resources import ResourceBundle, TargetPrior
from crychic.scoring import (
    ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
    FrozenActiveEdgeUniverse,
)

from .active_edge_scores import (
    CrossFitActiveEdgePointRecords,
    adapt_crossfit_active_edge_point_records,
    adapt_crossfit_active_edge_records_against_universe,
)
from .crossfit import CrossFitArtifacts, _run_subject_crossfit
from .training import (
    SanitizedRawInputSnapshot,
    _resource_bundle_content_id,
    _sanitized_raw_input_snapshot,
    _target_prior_content_id,
)

_SCHEMA_VERSION = "1.0.0"
_REQUEST_PRODUCER = "crychic.workflow.active_null_plan_request.v1"
_RECORD_PRODUCER = "crychic.workflow.active_null_rerun_record.v1"
_RESULT_PRODUCER = "crychic.workflow.active_null_rerun_result.v1"
_EXECUTION_METADATA_PRODUCER = "crychic.workflow.active_null_execution_metadata.v1"
_RERUN_POLICY_ID = "same_raw_config_resource_spec_seed_replace_target_prior_only_v1"
_SERIAL_BACKEND = "serial_v1"
_THREAD_BACKEND = "bounded_shared_snapshot_thread_pool_v1"
_PARALLEL_ORDERING_POLICY = "preplanned_request_order_v1"
_SHARED_SNAPSHOT_POLICY = "one_immutable_snapshot_shared_by_reference_v1"
_RERUN_STAGES = (
    "raw_fold_preparation",
    "feature_and_interaction_filtering",
    "availability_and_receptor_gating",
    "driver_family_clustering",
    "hyperparameter_tuning",
    "receiver_attribution",
    "sender_assignment",
    "cross_receiver_common_scoring",
)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_name(value: object, *, field_name: str) -> str | None:
    return None if value is None else _name(value, field_name=field_name)


def _nonnegative(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer >= 1")
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


def _failure_metadata(error: Exception) -> tuple[str, str]:
    failure_type = f"{type(error).__module__}.{type(error).__qualname__}"
    if isinstance(error, CrychicError):
        return failure_type, error.details.code
    return failure_type, type(error).__qualname__


class ActiveNullRerunStatus(StrEnum):
    """Execution status of one preplanned null opportunity."""

    SUCCEEDED = "succeeded"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


class ActiveNullRerunResultStatus(StrEnum):
    """Aggregate status across all requested null opportunities."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, init=False)
class ActiveNullPlanRequest:
    """Producer-owned plan identity that exists before planning can fail."""

    plan_index: int
    active_null_spec_id: str
    source_prior_content_id: str
    seed_lineage: SeedLineage
    plan_id: str
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ActiveNullPlanRequest is producer-owned; use run_active_null_reruns()"
        )

    @classmethod
    def _from_workflow(
        cls,
        *,
        plan_index: int,
        active_null_spec_id: str,
        source_prior_content_id: str,
        seed_lineage: SeedLineage,
    ) -> ActiveNullPlanRequest:
        index = _nonnegative(plan_index, field_name="plan_index")
        spec_id = _name(active_null_spec_id, field_name="active_null_spec_id")
        prior_id = _name(
            source_prior_content_id,
            field_name="source_prior_content_id",
        )
        if not isinstance(seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        self = object.__new__(cls)
        object.__setattr__(self, "plan_index", index)
        object.__setattr__(self, "active_null_spec_id", spec_id)
        object.__setattr__(self, "source_prior_content_id", prior_id)
        object.__setattr__(self, "seed_lineage", seed_lineage)
        object.__setattr__(self, "_producer_marker", _REQUEST_PRODUCER)
        identifier: str = stable_id(
            "active_null_plan_request",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        )
        object.__setattr__(
            self,
            "plan_id",
            identifier,
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "plan_index": self.plan_index,
            "active_null_spec_id": self.active_null_spec_id,
            "source_prior_content_id": self.source_prior_content_id,
            "seed_lineage": self.seed_lineage.to_dict(),
            "producer_marker": _REQUEST_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            expected = stable_id(
                "active_null_plan_request",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _REQUEST_PRODUCER
                and self.plan_index >= 0
                and self.plan_id == expected
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "Active-null plan request failed integrity validation",
                code="active_null_plan_request_integrity_violation",
                field="plan_id",
                remediation="Regenerate the request from the frozen rerun inputs",
            ) from error
        if not valid:
            raise _contract_error(
                "Active-null plan request failed integrity validation",
                code="active_null_plan_request_integrity_violation",
                field="plan_id",
                remediation="Regenerate the request from the frozen rerun inputs",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the preplanned identity and deterministic seed lineage."""

        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"plan_id": self.plan_id, **payload}


def _plan_diagnostics_id(plan: ActiveNullPlan) -> str:
    plan.require_intact()
    identifier: str = stable_id(
        "active_null_plan_diagnostics",
        {
            "active_null_plan_id": plan.active_null_plan_id,
            "source_edge_count": plan.source_edge_count,
            "accepted_switch_count": plan.accepted_switch_count,
            "attempted_switch_count": plan.attempted_switch_count,
            "source_overlap_count": plan.source_overlap_count,
            "source_overlap_fraction": plan.source_overlap_fraction,
            "tier_diagnostics": [item.to_dict() for item in plan.tier_diagnostics],
        },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


@dataclass(frozen=True, slots=True, init=False)
class ActiveNullRerunRecord:
    """Producer-owned lineage for one complete null score column."""

    active_null_id: str
    plan_index: int
    plan_id: str
    plan_seed_lineage: SeedLineage
    status: ActiveNullRerunStatus
    reason_code: str | None
    failure_type: str | None
    planner_status: ActiveNullPlanStatus | None
    active_null_plan_id: str | None
    planner_diagnostics_id: str | None
    source_edge_count: int | None
    accepted_switch_count: int | None
    attempted_switch_count: int | None
    source_overlap_fraction: float | None
    null_prior_content_id: str | None
    child_crossfit_id: str | None
    child_fold_plan_id: str | None
    child_oof_audit_id: str | None
    child_score_collection_id: str | None
    source_score_collection_id: str | None
    child_score_record_ids: tuple[str, ...]
    _child: CrossFitArtifacts | None = field(repr=False)
    null_rerun_record_id: str
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ActiveNullRerunRecord is producer-owned; use run_active_null_reruns()"
        )

    @classmethod
    def _from_workflow(
        cls,
        *,
        active_null_id: str,
        request: ActiveNullPlanRequest,
        status: ActiveNullRerunStatus,
        reason_code: str | None,
        failure_type: str | None,
        planner_status: ActiveNullPlanStatus | None,
        active_null_plan_id: str | None,
        planner_diagnostics_id: str | None,
        source_edge_count: int | None,
        accepted_switch_count: int | None,
        attempted_switch_count: int | None,
        source_overlap_fraction: float | None,
        null_prior_content_id: str | None,
        child_crossfit_id: str | None,
        child_fold_plan_id: str | None,
        child_oof_audit_id: str | None,
        child_score_collection_id: str | None,
        source_score_collection_id: str | None,
        child_score_record_ids: tuple[str, ...],
        child: CrossFitArtifacts | None,
    ) -> ActiveNullRerunRecord:
        request._require_intact()
        self = object.__new__(cls)
        object.__setattr__(
            self,
            "active_null_id",
            _name(active_null_id, field_name="active_null_id"),
        )
        object.__setattr__(self, "plan_index", request.plan_index)
        object.__setattr__(self, "plan_id", request.plan_id)
        object.__setattr__(self, "plan_seed_lineage", request.seed_lineage)
        object.__setattr__(self, "status", ActiveNullRerunStatus(status))
        object.__setattr__(
            self,
            "reason_code",
            _optional_name(reason_code, field_name="reason_code"),
        )
        object.__setattr__(
            self,
            "failure_type",
            _optional_name(failure_type, field_name="failure_type"),
        )
        object.__setattr__(
            self,
            "planner_status",
            None if planner_status is None else ActiveNullPlanStatus(planner_status),
        )
        for identifier_name, identifier_value in (
            ("active_null_plan_id", active_null_plan_id),
            ("planner_diagnostics_id", planner_diagnostics_id),
            ("null_prior_content_id", null_prior_content_id),
            ("child_crossfit_id", child_crossfit_id),
            ("child_fold_plan_id", child_fold_plan_id),
            ("child_oof_audit_id", child_oof_audit_id),
            ("child_score_collection_id", child_score_collection_id),
            ("source_score_collection_id", source_score_collection_id),
        ):
            object.__setattr__(
                self,
                identifier_name,
                _optional_name(identifier_value, field_name=identifier_name),
            )
        for count_name, count_value in (
            ("source_edge_count", source_edge_count),
            ("accepted_switch_count", accepted_switch_count),
            ("attempted_switch_count", attempted_switch_count),
        ):
            object.__setattr__(
                self,
                count_name,
                (
                    None
                    if count_value is None
                    else _nonnegative(count_value, field_name=count_name)
                ),
            )
        overlap = (
            None if source_overlap_fraction is None else float(source_overlap_fraction)
        )
        if overlap is not None and (
            not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0
        ):
            raise ValueError("source_overlap_fraction must lie in [0, 1]")
        object.__setattr__(self, "source_overlap_fraction", overlap)
        score_ids = tuple(
            _name(value, field_name="child_score_record_id")
            for value in child_score_record_ids
        )
        if len(score_ids) != len(set(score_ids)):
            raise ValueError("child_score_record_ids must be unique")
        object.__setattr__(self, "child_score_record_ids", score_ids)
        if child is not None and not isinstance(child, CrossFitArtifacts):
            raise TypeError("child must be CrossFitArtifacts or None")
        object.__setattr__(self, "_child", child)
        object.__setattr__(self, "_producer_marker", _RECORD_PRODUCER)
        self._validate_state()
        object.__setattr__(
            self,
            "null_rerun_record_id",
            stable_id(
                "active_null_rerun_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    def _validate_state(self) -> None:
        planner_fields = (
            self.active_null_plan_id,
            self.planner_diagnostics_id,
            self.source_edge_count,
            self.accepted_switch_count,
            self.attempted_switch_count,
        )
        if self.planner_status is None:
            if any(value is not None for value in planner_fields):
                raise ValueError("missing planner status cannot expose planner output")
        elif any(value is None for value in planner_fields):
            raise ValueError("planner output requires complete diagnostic lineage")

        if self.status is ActiveNullRerunStatus.SUCCEEDED:
            required = (
                self.active_null_plan_id,
                self.planner_diagnostics_id,
                self.null_prior_content_id,
                self.child_crossfit_id,
                self.child_fold_plan_id,
                self.child_oof_audit_id,
                self.child_score_collection_id,
                self.source_score_collection_id,
            )
            if (
                self.planner_status is not ActiveNullPlanStatus.OBSERVED
                or any(value is None for value in required)
                or not self.child_score_record_ids
                or self.reason_code is not None
                or self.failure_type is not None
            ):
                raise ValueError("successful active-null rerun lineage is incomplete")
        elif self.status is ActiveNullRerunStatus.NOT_ESTIMABLE:
            if (
                self.planner_status is not ActiveNullPlanStatus.NOT_ESTIMABLE
                or self.reason_code is None
                or self.failure_type is not None
                or self.null_prior_content_id is not None
                or any(
                    value is not None
                    for value in (
                        self.child_crossfit_id,
                        self.child_fold_plan_id,
                        self.child_oof_audit_id,
                        self.child_score_collection_id,
                        self.source_score_collection_id,
                    )
                )
                or self.child_score_record_ids
                or self._child is not None
            ):
                raise ValueError("not-estimable active-null rerun lineage is invalid")
        elif (
            self.reason_code is None
            or self.failure_type is None
            or self.child_score_collection_id is not None
            or self.source_score_collection_id is not None
            or self.child_score_record_ids
            or self._child is not None
        ):
            raise ValueError("failed active-null rerun lineage is invalid")
        if (
            self._child is not None
            and self._child.crossfit_id != self.child_crossfit_id
        ):
            raise ValueError("retained child does not match child_crossfit_id")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "active_null_id": self.active_null_id,
            "plan_index": self.plan_index,
            "plan_id": self.plan_id,
            "plan_seed_lineage": self.plan_seed_lineage.to_dict(),
            "status": self.status.value,
            "reason_code": self.reason_code,
            "failure_type": self.failure_type,
            "planner_status": (
                None if self.planner_status is None else self.planner_status.value
            ),
            "active_null_plan_id": self.active_null_plan_id,
            "planner_diagnostics_id": self.planner_diagnostics_id,
            "source_edge_count": self.source_edge_count,
            "accepted_switch_count": self.accepted_switch_count,
            "attempted_switch_count": self.attempted_switch_count,
            "source_overlap_fraction": self.source_overlap_fraction,
            "null_prior_content_id": self.null_prior_content_id,
            "child_crossfit_id": self.child_crossfit_id,
            "child_fold_plan_id": self.child_fold_plan_id,
            "child_oof_audit_id": self.child_oof_audit_id,
            "child_score_collection_id": self.child_score_collection_id,
            "source_score_collection_id": self.source_score_collection_id,
            "child_score_record_ids": list(self.child_score_record_ids),
            "rerun_policy_id": _RERUN_POLICY_ID,
            "producer_marker": _RECORD_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            self._validate_state()
            if self._child is not None:
                self._child._require_intact()
            expected = stable_id(
                "active_null_rerun_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _RECORD_PRODUCER
                and self.null_rerun_record_id == expected
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "Active-null rerun record failed integrity validation",
                code="active_null_rerun_record_integrity_violation",
                field="null_rerun_record_id",
                remediation="Rerun the exact preplanned null opportunity",
            ) from error
        if not valid:
            raise _contract_error(
                "Active-null rerun record failed integrity validation",
                code="active_null_rerun_record_integrity_violation",
                field="null_rerun_record_id",
                remediation="Rerun the exact preplanned null opportunity",
            )

    @property
    def child(self) -> CrossFitArtifacts | None:
        """Return the retained child only when explicitly requested."""

        return self._child

    def to_dict(self) -> dict[str, object]:
        """Return log-safe plan, child, and failure lineage."""

        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "null_rerun_record_id": self.null_rerun_record_id,
            **payload,
            "child_retained": self._child is not None,
        }


def _require_point_source_alignment(
    snapshot: SanitizedRawInputSnapshot,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    point_artifacts: CrossFitArtifacts,
    universe: FrozenActiveEdgeUniverse,
    point_scores: CrossFitActiveEdgePointRecords,
) -> tuple[str, str, SeedLineage]:
    resource_id = _resource_bundle_content_id(resource_bundle)
    prior_id = _target_prior_content_id(target_prior)
    root = point_artifacts.root_input_identity
    if (
        root.identity_id != snapshot.identity.identity_id
        or root.input_digest != snapshot.identity.input_digest
        or root.config_digest != config.digest
    ):
        raise _contract_error(
            "Active-null raw input or configuration differs from the point workflow",
            code="active_null_source_mismatch",
            field="root_input_identity,config_digest",
            remediation="Use the exact raw counts and configuration from the point run",
        )
    point_resource_ids = {
        fold.training.resource_bundle_content_id for fold in point_artifacts.folds
    }
    point_prior_ids = {
        fold.training.target_prior_content_id for fold in point_artifacts.folds
    }
    point_config_ids = {fold.training.config.digest for fold in point_artifacts.folds}
    if (
        point_resource_ids != {resource_id}
        or point_prior_ids != {prior_id}
        or point_config_ids != {config.digest}
    ):
        raise _contract_error(
            "Active-null resources or source prior differ from the point workflow",
            code="active_null_source_mismatch",
            field=("resource_bundle_content_id,target_prior_content_id,config_digest"),
            remediation="Use the exact point-run resource bundle and target prior",
        )
    expected_crossfit_seed = SeedLineage(config.random_seed).derive(
        "subject_crossfit",
        point_artifacts.spec.spec_id,
    )
    if (
        point_artifacts.fold_plan.seed_lineage.to_dict()
        != expected_crossfit_seed.to_dict()
        or point_artifacts.fold_plan.repeat_id != point_artifacts.spec.repeat_id
    ):
        raise _contract_error(
            "Point fold plan does not use the frozen configuration seed tree",
            code="active_null_source_mismatch",
            field="fold_plan.seed_lineage,repeat_id",
            remediation="Regenerate the point cross-fit from its frozen specification",
        )
    point_scores._require_intact()
    if (
        point_scores.source_crossfit_id != point_artifacts.crossfit_id
        or point_scores.active_edge_universe_id != universe.universe_id
        or point_scores.repeat_id != point_artifacts.spec.repeat_id
        or point_scores.score_version != universe.score_version
        or tuple(record.candidate_edge_id for record in point_scores.records)
        != universe.candidate_edge_ids
    ):
        raise _contract_error(
            "Point score collection does not bind the frozen active-edge source",
            code="active_null_universe_mismatch",
            field=("point_collection,candidate_universe_id,repeat_id,score_version"),
            remediation="Re-adapt point scores from the exact frozen universe",
        )
    return resource_id, prior_id, expected_crossfit_seed


def _require_child_alignment(
    child: CrossFitArtifacts,
    *,
    snapshot: SanitizedRawInputSnapshot,
    config: CrychicConfig,
    resource_bundle_content_id: str,
    null_prior_content_id: str,
    point_artifacts: CrossFitArtifacts,
) -> None:
    child._require_intact()
    if (
        child.root_input_identity.identity_id != snapshot.identity.identity_id
        or child.root_input_identity.input_digest != snapshot.identity.input_digest
        or child.root_input_identity.config_digest != config.digest
        or child.spec.spec_id != point_artifacts.spec.spec_id
        or child.spec.repeat_id != point_artifacts.spec.repeat_id
    ):
        raise _contract_error(
            "Active-null child changed raw input, configuration, or cross-fit policy",
            code="active_null_source_mismatch",
            field="root_input_identity,config_digest,crossfit_spec_id,repeat_id",
            remediation="Rerun with the exact point raw input, config, and spec",
        )
    if child.fold_plan.to_dict() != point_artifacts.fold_plan.to_dict():
        raise _contract_error(
            "Active-null child changed the point subject-fold or seed plan",
            code="active_null_train_test_leakage",
            field="fold_plan",
            remediation="Keep the exact point fold policy and seed tree",
        )
    if (
        child.receiver_universe.universe_id
        != point_artifacts.receiver_universe.universe_id
        or child.receiver_universe.receiver_axis_id
        != point_artifacts.receiver_universe.receiver_axis_id
        or child.receiver_universe.receiver_ids
        != point_artifacts.receiver_universe.receiver_ids
    ):
        raise _contract_error(
            "Active-null child changed the frozen receiver universe",
            code="active_null_receiver_universe_mismatch",
            field="receiver_universe_id,receiver_axis_id,receiver_ids",
            remediation=(
                "Reuse the exact point receiver universe for every active-null rerun"
            ),
        )
    resource_ids = {fold.training.resource_bundle_content_id for fold in child.folds}
    prior_ids = {fold.training.target_prior_content_id for fold in child.folds}
    config_ids = {fold.training.config.digest for fold in child.folds}
    if (
        resource_ids != {resource_bundle_content_id}
        or prior_ids != {null_prior_content_id}
        or config_ids != {config.digest}
    ):
        raise _contract_error(
            "Active-null child changed more than target-prior content",
            code="active_null_source_mismatch",
            field=("resource_bundle_content_id,target_prior_content_id,config_digest"),
            remediation="Replace only TargetPrior in the full cross-fit rerun",
        )


def _require_adapted_alignment(
    adapted: CrossFitActiveEdgePointRecords,
    *,
    child: CrossFitArtifacts,
    point_scores: CrossFitActiveEdgePointRecords,
    universe: FrozenActiveEdgeUniverse,
) -> None:
    adapted._require_intact()
    if (
        adapted.source_crossfit_id != child.crossfit_id
        or adapted.active_edge_universe_id != universe.universe_id
        or adapted.contrast_id != point_scores.contrast_id
        or adapted.repeat_id != point_scores.repeat_id
        or adapted.score_version != point_scores.score_version
        or adapted.score_spec_id != point_scores.score_spec_id
        or tuple(record.candidate_edge_id for record in adapted.records)
        != universe.candidate_edge_ids
    ):
        raise _contract_error(
            "Active-null child scores do not match the frozen point score contract",
            code="active_null_universe_mismatch",
            field=(
                "candidate_universe_id,contrast_id,repeat_id,score_version,score_spec_id"
            ),
            remediation="Adapt the null child against the authoritative point universe",
        )


def _uniform_null_cells(
    point_records: tuple[PointActiveEdgeScoreRecord, ...],
    *,
    request: ActiveNullPlanRequest,
    rerun_record: ActiveNullRerunRecord,
    status: ActiveEdgeScoreStatus,
    reason_code: str,
) -> tuple[NullActiveEdgeScoreRecord, ...]:
    return tuple(
        NullActiveEdgeScoreRecord(
            candidate_edge_id=point.candidate_edge_id,
            plan_id=request.plan_id,
            stratum_id=point.stratum_id,
            score_version=point.score_version,
            null_rerun_record_id=rerun_record.null_rerun_record_id,
            source_score_collection_id=None,
            n_subjects=None,
            score=None,
            status=status,
            reason_code=reason_code,
        )
        for point in point_records
    )


def _adapted_null_cells(
    adapted: CrossFitActiveEdgePointRecords,
    *,
    request: ActiveNullPlanRequest,
    rerun_record: ActiveNullRerunRecord,
) -> tuple[NullActiveEdgeScoreRecord, ...]:
    return tuple(
        NullActiveEdgeScoreRecord(
            candidate_edge_id=record.candidate_edge_id,
            plan_id=request.plan_id,
            stratum_id=record.stratum_id,
            score_version=record.score_version,
            null_rerun_record_id=rerun_record.null_rerun_record_id,
            source_score_collection_id=record.source_score_collection_id,
            n_subjects=record.n_subjects,
            score=record.score,
            status=record.status,
            reason_code=record.reason_code,
        )
        for record in adapted.records
    )


@dataclass(frozen=True, slots=True)
class _PlanLineage:
    planner_status: ActiveNullPlanStatus | None
    active_null_plan_id: str | None
    planner_diagnostics_id: str | None
    source_edge_count: int | None
    accepted_switch_count: int | None
    attempted_switch_count: int | None
    source_overlap_fraction: float | None


def _plan_fields(plan: ActiveNullPlan | None) -> _PlanLineage:
    if plan is None:
        return _PlanLineage(
            planner_status=None,
            active_null_plan_id=None,
            planner_diagnostics_id=None,
            source_edge_count=None,
            accepted_switch_count=None,
            attempted_switch_count=None,
            source_overlap_fraction=None,
        )
    return _PlanLineage(
        planner_status=plan.status,
        active_null_plan_id=plan.active_null_plan_id,
        planner_diagnostics_id=_plan_diagnostics_id(plan),
        source_edge_count=plan.source_edge_count,
        accepted_switch_count=plan.accepted_switch_count,
        attempted_switch_count=plan.attempted_switch_count,
        source_overlap_fraction=plan.source_overlap_fraction,
    )


def _execute_request(
    request: ActiveNullPlanRequest,
    *,
    active_null_id: str,
    snapshot: SanitizedRawInputSnapshot,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    source_prior: TargetPrior,
    active_null_spec: ActiveNullSpec,
    point_artifacts: CrossFitArtifacts,
    point_scores: CrossFitActiveEdgePointRecords,
    universe: FrozenActiveEdgeUniverse,
    resource_bundle_content_id: str,
    source_prior_content_id: str,
    retain_child: bool,
) -> tuple[ActiveNullRerunRecord, tuple[NullActiveEdgeScoreRecord, ...]]:
    plan: ActiveNullPlan | None = None
    null_prior_content_id: str | None = None
    child: CrossFitArtifacts | None = None
    try:
        planned = plan_active_null_target_prior(
            source_prior,
            seed_lineage=request.seed_lineage,
            spec=active_null_spec,
        )
        if not isinstance(planned, ActiveNullPlan):
            raise TypeError("active-null planner must return ActiveNullPlan")
        planned.require_intact()
        if (
            planned.source_prior_content_id != source_prior_content_id
            or planned.active_null_spec_id != active_null_spec.active_null_spec_id
            or planned.seed_lineage.to_dict() != request.seed_lineage.to_dict()
        ):
            raise _contract_error(
                "Active-null planner returned incompatible source lineage",
                code="active_null_source_mismatch",
                field="source_prior_content_id,active_null_spec_id,seed_lineage",
                remediation="Regenerate the plan from the exact preplanned request",
            )
        plan = planned
        if plan.status is ActiveNullPlanStatus.NOT_ESTIMABLE:
            if plan.reason_code is None:  # pragma: no cover - planner contract
                raise RuntimeError("not-estimable plan lacks a reason code")
            plan_lineage = _plan_fields(plan)
            record = ActiveNullRerunRecord._from_workflow(
                active_null_id=active_null_id,
                request=request,
                status=ActiveNullRerunStatus.NOT_ESTIMABLE,
                reason_code=plan.reason_code.value,
                failure_type=None,
                null_prior_content_id=None,
                child_crossfit_id=None,
                child_fold_plan_id=None,
                child_oof_audit_id=None,
                child_score_collection_id=None,
                source_score_collection_id=None,
                child_score_record_ids=(),
                child=None,
                planner_status=plan_lineage.planner_status,
                active_null_plan_id=plan_lineage.active_null_plan_id,
                planner_diagnostics_id=plan_lineage.planner_diagnostics_id,
                source_edge_count=plan_lineage.source_edge_count,
                accepted_switch_count=plan_lineage.accepted_switch_count,
                attempted_switch_count=plan_lineage.attempted_switch_count,
                source_overlap_fraction=(plan_lineage.source_overlap_fraction),
            )
            return record, _uniform_null_cells(
                point_scores.records,
                request=request,
                rerun_record=record,
                status=ActiveEdgeScoreStatus.NOT_ESTIMABLE,
                reason_code=plan.reason_code.value,
            )

        null_prior = materialize_active_null_target_prior(source_prior, plan)
        if not isinstance(null_prior, TargetPrior):
            raise TypeError("active-null materializer must return TargetPrior")
        null_prior_content_id = _target_prior_content_id(null_prior)
        if null_prior_content_id == source_prior_content_id:
            raise _contract_error(
                "Observed active-null plan did not change target-prior content",
                code="active_null_insufficient_rewiring",
                field="target_prior_content_id",
                remediation="Reject the unchanged pseudo-null plan",
            )
        executed_child = _run_subject_crossfit(
            snapshot,
            config,
            resource_bundle,
            null_prior,
            spec=point_artifacts.spec,
        )
        if not isinstance(executed_child, CrossFitArtifacts):
            raise TypeError("active-null runner must return CrossFitArtifacts")
        child = executed_child
        _require_child_alignment(
            child,
            snapshot=snapshot,
            config=config,
            resource_bundle_content_id=resource_bundle_content_id,
            null_prior_content_id=null_prior_content_id,
            point_artifacts=point_artifacts,
        )
        adapted = adapt_crossfit_active_edge_records_against_universe(child, universe)
        if not isinstance(adapted, CrossFitActiveEdgePointRecords):
            raise TypeError("active-null score adapter returned an unsupported value")
        _require_adapted_alignment(
            adapted,
            child=child,
            point_scores=point_scores,
            universe=universe,
        )
        plan_lineage = _plan_fields(plan)
        record = ActiveNullRerunRecord._from_workflow(
            active_null_id=active_null_id,
            request=request,
            status=ActiveNullRerunStatus.SUCCEEDED,
            reason_code=None,
            failure_type=None,
            null_prior_content_id=null_prior_content_id,
            child_crossfit_id=child.crossfit_id,
            child_fold_plan_id=child.fold_plan.plan_id,
            child_oof_audit_id=adapted.source_oof_audit_id,
            child_score_collection_id=adapted.collection_id,
            source_score_collection_id=adapted.source_score_collection_id,
            child_score_record_ids=tuple(item.record_id for item in adapted.records),
            child=child if retain_child else None,
            planner_status=plan_lineage.planner_status,
            active_null_plan_id=plan_lineage.active_null_plan_id,
            planner_diagnostics_id=plan_lineage.planner_diagnostics_id,
            source_edge_count=plan_lineage.source_edge_count,
            accepted_switch_count=plan_lineage.accepted_switch_count,
            attempted_switch_count=plan_lineage.attempted_switch_count,
            source_overlap_fraction=plan_lineage.source_overlap_fraction,
        )
        return record, _adapted_null_cells(
            adapted,
            request=request,
            rerun_record=record,
        )
    except Exception as error:
        failure_type, failure_code = _failure_metadata(error)
        child_crossfit_id = None if child is None else child.crossfit_id
        child_fold_plan_id = None if child is None else child.fold_plan.plan_id
        plan_lineage = _plan_fields(plan)
        record = ActiveNullRerunRecord._from_workflow(
            active_null_id=active_null_id,
            request=request,
            status=ActiveNullRerunStatus.FAILED,
            reason_code=failure_code,
            failure_type=failure_type,
            null_prior_content_id=null_prior_content_id,
            child_crossfit_id=child_crossfit_id,
            child_fold_plan_id=child_fold_plan_id,
            child_oof_audit_id=None,
            child_score_collection_id=None,
            source_score_collection_id=None,
            child_score_record_ids=(),
            child=None,
            planner_status=plan_lineage.planner_status,
            active_null_plan_id=plan_lineage.active_null_plan_id,
            planner_diagnostics_id=plan_lineage.planner_diagnostics_id,
            source_edge_count=plan_lineage.source_edge_count,
            accepted_switch_count=plan_lineage.accepted_switch_count,
            attempted_switch_count=plan_lineage.attempted_switch_count,
            source_overlap_fraction=plan_lineage.source_overlap_fraction,
        )
        return record, _uniform_null_cells(
            point_scores.records,
            request=request,
            rerun_record=record,
            status=ActiveEdgeScoreStatus.FAILED,
            reason_code=failure_code,
        )


@dataclass(frozen=True, slots=True, init=False)
class ActiveNullRerunResult:
    """Producer-owned exact null distribution and complete source registry."""

    source_point_crossfit_id: str
    source_point_collection_id: str
    source_point_oof_audit_id: str
    source_point_registry_id: str
    source_point_score_collection_id: str
    source_input_identity_id: str
    source_input_digest: str
    source_snapshot_id: str
    config_digest: str
    crossfit_spec_id: str
    repeat_id: str
    fold_plan_id: str
    crossfit_seed_lineage: SeedLineage
    partition_seed_lineage: SeedLineage | None
    resource_bundle_content_id: str
    source_target_prior_content_id: str
    candidate_universe_id: str
    score_version: str
    score_spec_id: str
    active_null_spec_id: str
    root_seed_lineage: SeedLineage
    active_null_id: str
    requests: tuple[ActiveNullPlanRequest, ...]
    records: tuple[ActiveNullRerunRecord, ...]
    distribution: NullScoreDistribution
    retain_children: bool
    requested_n_jobs: int
    effective_n_jobs: int
    execution_metadata_id: str
    result_id: str
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ActiveNullRerunResult is producer-owned; use run_active_null_reruns()"
        )

    @classmethod
    def _from_workflow(
        cls,
        *,
        source_point_crossfit_id: str,
        source_point_collection_id: str,
        source_point_oof_audit_id: str,
        source_point_registry_id: str,
        source_point_score_collection_id: str,
        source_input_identity_id: str,
        source_input_digest: str,
        source_snapshot_id: str,
        config_digest: str,
        crossfit_spec_id: str,
        repeat_id: str,
        fold_plan_id: str,
        crossfit_seed_lineage: SeedLineage,
        partition_seed_lineage: SeedLineage | None,
        resource_bundle_content_id: str,
        source_target_prior_content_id: str,
        candidate_universe_id: str,
        score_version: str,
        score_spec_id: str,
        active_null_spec_id: str,
        root_seed_lineage: SeedLineage,
        active_null_id: str,
        requests: tuple[ActiveNullPlanRequest, ...],
        records: tuple[ActiveNullRerunRecord, ...],
        distribution: NullScoreDistribution,
        retain_children: bool,
        requested_n_jobs: int,
        effective_n_jobs: int,
    ) -> ActiveNullRerunResult:
        self = object.__new__(cls)
        for name, value in (
            ("source_point_crossfit_id", source_point_crossfit_id),
            ("source_point_collection_id", source_point_collection_id),
            ("source_point_oof_audit_id", source_point_oof_audit_id),
            ("source_point_registry_id", source_point_registry_id),
            ("source_point_score_collection_id", source_point_score_collection_id),
            ("source_input_identity_id", source_input_identity_id),
            ("source_input_digest", source_input_digest),
            ("source_snapshot_id", source_snapshot_id),
            ("config_digest", config_digest),
            ("crossfit_spec_id", crossfit_spec_id),
            ("repeat_id", repeat_id),
            ("fold_plan_id", fold_plan_id),
            ("resource_bundle_content_id", resource_bundle_content_id),
            ("source_target_prior_content_id", source_target_prior_content_id),
            ("candidate_universe_id", candidate_universe_id),
            ("score_version", score_version),
            ("score_spec_id", score_spec_id),
            ("active_null_spec_id", active_null_spec_id),
            ("active_null_id", active_null_id),
        ):
            object.__setattr__(self, name, _name(value, field_name=name))
        if not isinstance(crossfit_seed_lineage, SeedLineage):
            raise TypeError("crossfit_seed_lineage must be a SeedLineage")
        if partition_seed_lineage is not None and not isinstance(
            partition_seed_lineage, SeedLineage
        ):
            raise TypeError("partition_seed_lineage must be SeedLineage or None")
        if not isinstance(root_seed_lineage, SeedLineage):
            raise TypeError("root_seed_lineage must be a SeedLineage")
        if not isinstance(retain_children, bool):
            raise TypeError("retain_children must be boolean")
        object.__setattr__(self, "crossfit_seed_lineage", crossfit_seed_lineage)
        object.__setattr__(self, "partition_seed_lineage", partition_seed_lineage)
        object.__setattr__(self, "root_seed_lineage", root_seed_lineage)
        object.__setattr__(self, "requests", tuple(requests))
        object.__setattr__(self, "records", tuple(records))
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "retain_children", retain_children)
        object.__setattr__(
            self,
            "requested_n_jobs",
            _positive(requested_n_jobs, field_name="requested_n_jobs"),
        )
        object.__setattr__(
            self,
            "effective_n_jobs",
            _positive(effective_n_jobs, field_name="effective_n_jobs"),
        )
        object.__setattr__(self, "_producer_marker", _RESULT_PRODUCER)
        object.__setattr__(
            self,
            "execution_metadata_id",
            stable_id(
                "active_null_execution_metadata",
                self._execution_metadata_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._validate_state()
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "active_null_rerun_result",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    def _validate_state(self) -> None:
        if not isinstance(self.distribution, NullScoreDistribution):
            raise TypeError("distribution must be a NullScoreDistribution")
        self.distribution._require_intact()
        requests = self.requests
        records = self.records
        if not requests or len(requests) != len(records):
            raise ValueError("requests and records must align one-to-one")
        if self.effective_n_jobs != min(self.requested_n_jobs, len(requests)):
            raise ValueError(
                "effective_n_jobs must equal min(requested_n_jobs, n_plans)"
            )
        expected_execution_id = stable_id(
            "active_null_execution_metadata",
            self._execution_metadata_payload(),
            schema_version=_SCHEMA_VERSION,
        )
        if self.execution_metadata_id != expected_execution_id:
            raise ValueError("active-null execution metadata identity is invalid")
        for request in requests:
            request._require_intact()
        for record in records:
            record._require_intact()
        if tuple(request.plan_index for request in requests) != tuple(
            range(len(requests))
        ):
            raise ValueError("plan requests must exactly cover sequential indices")
        if any(
            request.plan_id != record.plan_id
            or request.plan_index != record.plan_index
            or request.seed_lineage.to_dict() != record.plan_seed_lineage.to_dict()
            or record.active_null_id != self.active_null_id
            for request, record in zip(requests, records, strict=True)
        ):
            raise ValueError("plan requests and rerun records are misaligned")
        request_ids = tuple(request.plan_id for request in requests)
        if (
            len(request_ids) != len(set(request_ids))
            or set(request_ids) != set(self.distribution.plan_ids)
            or self.distribution.active_null_id != self.active_null_id
            or self.distribution.candidate_universe_id != self.candidate_universe_id
            or self.distribution.score_spec_id != self.score_spec_id
            or self.distribution.score_version != self.score_version
            or self.distribution.source_score_collection_id
            != self.source_point_score_collection_id
        ):
            raise ValueError("active-null distribution lineage is incompatible")
        rerun_by_plan = {record.plan_id: record for record in records}
        if any(
            cell.null_rerun_record_id
            != rerun_by_plan[cell.plan_id].null_rerun_record_id
            for cell in self.distribution.null_records
        ):
            raise ValueError("null cells do not bind their plan rerun record")
        cells_by_plan = {
            plan_id: tuple(
                cell
                for cell in self.distribution.null_records
                if cell.plan_id == plan_id
            )
            for plan_id in request_ids
        }
        for record in records:
            cells = cells_by_plan[record.plan_id]
            statuses = {cell.status for cell in cells}
            reasons = {cell.reason_code for cell in cells}
            if record.status is ActiveNullRerunStatus.SUCCEEDED:
                if ActiveEdgeScoreStatus.FAILED in statuses or {
                    cell.source_score_collection_id for cell in cells
                } != {record.source_score_collection_id}:
                    raise ValueError(
                        "successful rerun cells do not match their score collection"
                    )
            elif record.status is ActiveNullRerunStatus.NOT_ESTIMABLE:
                if statuses != {ActiveEdgeScoreStatus.NOT_ESTIMABLE} or reasons != {
                    record.reason_code
                }:
                    raise ValueError(
                        "not-estimable rerun must emit one uniform typed column"
                    )
            elif statuses != {ActiveEdgeScoreStatus.FAILED} or reasons != {
                record.reason_code
            }:
                raise ValueError("failed rerun must emit one uniform failed column")
        if self.retain_children:
            if any(
                record.status is ActiveNullRerunStatus.SUCCEEDED
                and record.child is None
                for record in records
            ):
                raise ValueError("retained mode requires every successful child")
        elif any(record.child is not None for record in records):
            raise ValueError("non-retained mode cannot contain cross-fit children")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "source_point_crossfit_id": self.source_point_crossfit_id,
            "source_point_collection_id": self.source_point_collection_id,
            "source_point_oof_audit_id": self.source_point_oof_audit_id,
            "source_point_registry_id": self.source_point_registry_id,
            "source_point_score_collection_id": (self.source_point_score_collection_id),
            "source_input_identity_id": self.source_input_identity_id,
            "source_input_digest": self.source_input_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "config_digest": self.config_digest,
            "crossfit_spec_id": self.crossfit_spec_id,
            "repeat_id": self.repeat_id,
            "fold_plan_id": self.fold_plan_id,
            "crossfit_seed_lineage": self.crossfit_seed_lineage.to_dict(),
            "partition_seed_lineage": (
                None
                if self.partition_seed_lineage is None
                else self.partition_seed_lineage.to_dict()
            ),
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "source_target_prior_content_id": self.source_target_prior_content_id,
            "candidate_universe_id": self.candidate_universe_id,
            "score_version": self.score_version,
            "score_spec_id": self.score_spec_id,
            "active_null_spec_id": self.active_null_spec_id,
            "root_seed_lineage": self.root_seed_lineage.to_dict(),
            "active_null_id": self.active_null_id,
            "plan_ids": [request.plan_id for request in self.requests],
            "rerun_record_ids": [
                record.null_rerun_record_id for record in self.records
            ],
            "distribution_id": self.distribution.distribution_id,
            "retain_children": self.retain_children,
            "rerun_policy_id": _RERUN_POLICY_ID,
            "producer_marker": _RESULT_PRODUCER,
        }

    def _execution_metadata_payload(self) -> dict[str, object]:
        return {
            "active_null_id": self.active_null_id,
            "plan_ids": [request.plan_id for request in self.requests],
            "requested_n_jobs": self.requested_n_jobs,
            "effective_n_jobs": self.effective_n_jobs,
            "execution_backend": (
                _SERIAL_BACKEND if self.effective_n_jobs == 1 else _THREAD_BACKEND
            ),
            "parallel_ordering_policy": _PARALLEL_ORDERING_POLICY,
            "shared_snapshot_policy": _SHARED_SNAPSHOT_POLICY,
            "producer_marker": _EXECUTION_METADATA_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            self._validate_state()
            expected = stable_id(
                "active_null_rerun_result",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _RESULT_PRODUCER and self.result_id == expected
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise _contract_error(
                "Active-null rerun result failed integrity validation",
                code="active_null_rerun_result_integrity_violation",
                field="result_id",
                remediation="Rerun the exact frozen active-null request set",
            ) from error
        if not valid:
            raise _contract_error(
                "Active-null rerun result failed integrity validation",
                code="active_null_rerun_result_integrity_violation",
                field="result_id",
                remediation="Rerun the exact frozen active-null request set",
            )

    @property
    def status(self) -> ActiveNullRerunResultStatus:
        statuses = tuple(record.status for record in self.records)
        if all(value is ActiveNullRerunStatus.SUCCEEDED for value in statuses):
            return ActiveNullRerunResultStatus.SUCCEEDED
        if all(value is ActiveNullRerunStatus.NOT_ESTIMABLE for value in statuses):
            return ActiveNullRerunResultStatus.NOT_ESTIMABLE
        if all(value is ActiveNullRerunStatus.FAILED for value in statuses):
            return ActiveNullRerunResultStatus.FAILED
        return ActiveNullRerunResultStatus.PARTIAL

    @property
    def children(self) -> tuple[CrossFitArtifacts, ...]:
        """Return explicitly retained successful full-pipeline children."""

        return tuple(
            record.child for record in self.records if record.child is not None
        )

    def to_manifest(self) -> dict[str, object]:
        """Return complete lineage without embedding child cross-fit artifacts."""

        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "schema_version": _SCHEMA_VERSION,
            "result_id": self.result_id,
            "status": self.status.value,
            **payload,
            "execution_backend": (
                _SERIAL_BACKEND if self.effective_n_jobs == 1 else _THREAD_BACKEND
            ),
            "requested_n_jobs": self.requested_n_jobs,
            "effective_n_jobs": self.effective_n_jobs,
            "execution_metadata_id": self.execution_metadata_id,
            "parallel_ordering_policy": _PARALLEL_ORDERING_POLICY,
            "shared_read_only_snapshot": True,
            "shared_snapshot_policy": _SHARED_SNAPSHOT_POLICY,
            "parallel_memory_budget": (
                "peak working memory may approach effective_n_jobs times one "
                "null cross-fit; also cap BLAS/OpenMP threads"
            ),
            "full_pipeline_refit_per_plan": True,
            "only_target_prior_content_changes": True,
            "score_table_relabeling_only": False,
            "rerun_stages": list(_RERUN_STAGES),
            "n_candidates": len(self.distribution.candidate_edge_ids),
            "n_plans": len(self.requests),
            "n_null_cells": len(self.distribution.null_records),
            "requests": [request.to_dict() for request in self.requests],
            "records": [record.to_dict() for record in self.records],
        }


def _active_null_id(
    *,
    point_scores: CrossFitActiveEdgePointRecords,
    snapshot: SanitizedRawInputSnapshot,
    config_digest: str,
    crossfit_spec_id: str,
    repeat_id: str,
    fold_plan_id: str,
    resource_bundle_content_id: str,
    source_prior_content_id: str,
    universe_id: str,
    active_null_spec_id: str,
    root_seed_lineage: SeedLineage,
    requests: tuple[ActiveNullPlanRequest, ...],
) -> str:
    identifier: str = stable_id(
        "active_null_full_pipeline_run",
        {
            "source_point_crossfit_id": point_scores.source_crossfit_id,
            "source_point_collection_id": point_scores.collection_id,
            "source_input_identity_id": snapshot.identity.identity_id,
            "source_input_digest": snapshot.identity.input_digest,
            "source_snapshot_id": snapshot.snapshot_id,
            "config_digest": config_digest,
            "crossfit_spec_id": crossfit_spec_id,
            "repeat_id": repeat_id,
            "fold_plan_id": fold_plan_id,
            "resource_bundle_content_id": resource_bundle_content_id,
            "source_prior_content_id": source_prior_content_id,
            "candidate_universe_id": universe_id,
            "score_version": point_scores.score_version,
            "score_spec_id": point_scores.score_spec_id,
            "active_null_spec_id": active_null_spec_id,
            "root_seed_lineage": root_seed_lineage.to_dict(),
            "plan_ids": [request.plan_id for request in requests],
            "rerun_policy_id": _RERUN_POLICY_ID,
        },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


def run_active_null_reruns(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    point_artifacts: CrossFitArtifacts,
    universe: FrozenActiveEdgeUniverse,
    *,
    n_plans: int,
    n_jobs: int = 1,
    active_null_spec: ActiveNullSpec | None = None,
    retain_children: bool = False,
) -> ActiveNullRerunResult:
    """Rerun exact raw cross-fit plans after replacing only ``TargetPrior``.

    ``n_jobs`` is a positive integer and concurrency is bounded by
    ``min(n_jobs, n_plans)``.  Threads share the immutable sanitized snapshot,
    resources, point artifacts, and source prior.  Results are always collected
    in preplanned request order, independent of worker completion order.

    The shared snapshot avoids one full AnnData copy per worker, but each worker
    still creates fold-local matrices and fitted models.  Choose ``n_jobs`` from
    an external memory budget and coordinate it with BLAS/OpenMP thread limits.
    """

    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig")
    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be a TargetPrior")
    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    if not isinstance(universe, FrozenActiveEdgeUniverse):
        raise TypeError("universe must be a FrozenActiveEdgeUniverse")
    plans = _nonnegative(n_plans, field_name="n_plans")
    if plans < 1:
        raise ValueError("n_plans must be an integer >= 1")
    requested_jobs = _positive(n_jobs, field_name="n_jobs")
    effective_jobs = min(requested_jobs, plans)
    if not isinstance(retain_children, bool):
        raise TypeError("retain_children must be boolean")
    resolved_spec = ActiveNullSpec() if active_null_spec is None else active_null_spec
    if not isinstance(resolved_spec, ActiveNullSpec):
        raise TypeError("active_null_spec must be ActiveNullSpec or None")
    if resolved_spec.to_dict() != ActiveNullSpec().to_dict():
        raise _contract_error(
            "Active-null specification identity is not intact",
            code="active_null_spec_integrity_violation",
            field="active_null_spec_id",
            remediation="Rebuild the accepted ADR-013 active-null specification",
        )

    point_artifacts._require_intact()
    universe._require_intact()
    snapshot = _sanitized_raw_input_snapshot(adata, config)
    snapshot._require_intact()
    point_scores = adapt_crossfit_active_edge_point_records(point_artifacts, universe)
    if not isinstance(point_scores, CrossFitActiveEdgePointRecords):
        raise TypeError("point score adapter returned an unsupported value")
    resource_id, prior_id, crossfit_seed = _require_point_source_alignment(
        snapshot,
        config,
        resource_bundle,
        target_prior,
        point_artifacts,
        universe,
        point_scores,
    )

    root_seed = SeedLineage(config.random_seed).derive(
        "active_null_rerun_v1",
        f"point_crossfit_id={point_artifacts.crossfit_id}",
        f"candidate_universe_id={universe.universe_id}",
        f"active_null_spec_id={resolved_spec.active_null_spec_id}",
    )
    requests = tuple(
        ActiveNullPlanRequest._from_workflow(
            plan_index=index,
            active_null_spec_id=resolved_spec.active_null_spec_id,
            source_prior_content_id=prior_id,
            seed_lineage=root_seed.derive("plan", f"index={index:06d}"),
        )
        for index in range(plans)
    )
    active_null_id = _active_null_id(
        point_scores=point_scores,
        snapshot=snapshot,
        config_digest=config.digest,
        crossfit_spec_id=point_artifacts.spec.spec_id,
        repeat_id=point_artifacts.spec.repeat_id,
        fold_plan_id=point_artifacts.fold_plan.plan_id,
        resource_bundle_content_id=resource_id,
        source_prior_content_id=prior_id,
        universe_id=universe.universe_id,
        active_null_spec_id=resolved_spec.active_null_spec_id,
        root_seed_lineage=root_seed,
        requests=requests,
    )

    execute = partial(
        _execute_request,
        active_null_id=active_null_id,
        snapshot=snapshot,
        config=config,
        resource_bundle=resource_bundle,
        source_prior=target_prior,
        active_null_spec=resolved_spec,
        point_artifacts=point_artifacts,
        point_scores=point_scores,
        universe=universe,
        resource_bundle_content_id=resource_id,
        source_prior_content_id=prior_id,
        retain_child=retain_children,
    )
    if effective_jobs == 1:
        outcomes = tuple(execute(request) for request in requests)
    else:
        with ThreadPoolExecutor(
            max_workers=effective_jobs,
            thread_name_prefix="crychic-active-null",
        ) as executor:
            outcomes = tuple(executor.map(execute, requests))

    records: list[ActiveNullRerunRecord] = []
    null_records: list[NullActiveEdgeScoreRecord] = []
    for record, cells in outcomes:
        records.append(record)
        null_records.extend(cells)

    distribution = NullScoreDistribution(
        active_null_id=active_null_id,
        active_null_spec_id=resolved_spec.active_null_spec_id,
        candidate_universe_id=universe.universe_id,
        candidate_universe_policy_id=ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
        score_spec_id=point_scores.score_spec_id,
        point_records=point_scores.records,
        plan_ids=tuple(request.plan_id for request in requests),
        null_records=tuple(null_records),
    )
    partition_seed = point_artifacts.fold_plan.partition_seed_lineage
    return ActiveNullRerunResult._from_workflow(
        source_point_crossfit_id=point_artifacts.crossfit_id,
        source_point_collection_id=point_scores.collection_id,
        source_point_oof_audit_id=point_scores.source_oof_audit_id,
        source_point_registry_id=point_scores.source_registry_id,
        source_point_score_collection_id=point_scores.source_score_collection_id,
        source_input_identity_id=snapshot.identity.identity_id,
        source_input_digest=snapshot.identity.input_digest,
        source_snapshot_id=snapshot.snapshot_id,
        config_digest=config.digest,
        crossfit_spec_id=point_artifacts.spec.spec_id,
        repeat_id=point_artifacts.spec.repeat_id,
        fold_plan_id=point_artifacts.fold_plan.plan_id,
        crossfit_seed_lineage=crossfit_seed,
        partition_seed_lineage=partition_seed,
        resource_bundle_content_id=resource_id,
        source_target_prior_content_id=prior_id,
        candidate_universe_id=universe.universe_id,
        score_version=point_scores.score_version,
        score_spec_id=point_scores.score_spec_id,
        active_null_spec_id=resolved_spec.active_null_spec_id,
        root_seed_lineage=root_seed,
        active_null_id=active_null_id,
        requests=requests,
        records=tuple(records),
        distribution=distribution,
        retain_children=retain_children,
        requested_n_jobs=requested_jobs,
        effective_n_jobs=effective_jobs,
    )


__all__ = [
    "ActiveNullPlanRequest",
    "ActiveNullRerunRecord",
    "ActiveNullRerunResult",
    "ActiveNullRerunResultStatus",
    "ActiveNullRerunStatus",
    "run_active_null_reruns",
]
