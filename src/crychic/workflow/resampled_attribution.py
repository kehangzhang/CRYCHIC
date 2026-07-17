"""Single-target diagnostic outer-training selection over subject bootstraps.

Complete-universe batch publication is intentionally unavailable. This module
keeps one target in memory; performance batching is deferred to a later phase.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from crychic.core import ContractError, CrychicError, stable_id
from crychic.inference import HypothesisRole
from crychic.resampling import SubjectBootstrapPlan
from crychic.scoring import (
    FamilyCommonScoringApplication,
    FamilyCommonScoringFunctional,
)

from .crossfit import CrossFitArtifacts, CrossFitFoldArtifacts
from .full_pipeline_effects import FrozenFamilyEffectTarget
from .full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResampleStatus,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
)

_SCHEMA_VERSION = "1.0.0"
_PRIMARY_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_EVENT_MARKER = "crychic.workflow.resampled_family_selection_event.v1"
_COLLECTION_MARKER = "crychic.workflow.resampled_family_attribution.v1"
_FORMAL_STATUS = "diagnostic_only_full_pipeline_bootstrap_selection_v1"
_SELECTION_RULE_SEMANTICS = (
    "outer_training_incremental_family_coefficient_strictly_greater_than_zero_v1"
)
_SELECTION_THRESHOLD = 0.0
_SELECTION_RULE_ID: str = stable_id(
    "resampled_family_selection_rule",
    {
        "semantics": _SELECTION_RULE_SEMANTICS,
        "coefficient_source": "outer_training_incremental_functional_v1",
        "operator": "strictly_greater_than",
        "threshold": _SELECTION_THRESHOLD,
        "heldout_application_used_for_selection": False,
    },
    schema_version=_SCHEMA_VERSION,
)
_PUBLIC_RELEASE_STATUS = (
    "complete_universe_batch_public_release_unavailable_single_target_diagnostic_v1"
)
_PERFORMANCE_STATUS = "serial_in_memory_per_target_batching_deferred_v1"


class ResampledAttributionEventStatus(StrEnum):
    """Availability of one bootstrap-fold family selection event."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _failure_reason(error: Exception) -> str:
    if isinstance(error, CrychicError):
        reason: str = error.details.code
        return reason
    return f"resampled_attribution_adapter_failed_{type(error).__qualname__}"


def _require_target(target: FrozenFamilyEffectTarget) -> None:
    if not isinstance(target, FrozenFamilyEffectTarget):
        raise TypeError("target must be a FrozenFamilyEffectTarget")
    target._require_intact()
    if (
        target.role is not HypothesisRole.PRIMARY
        or target.endpoint != _PRIMARY_ENDPOINT
    ):
        raise ContractError(
            "Resampled attribution requires an included primary family omnibus target",
            code="resampled_attribution_primary_target_mismatch",
            field="role,endpoint",
            remediation="Build the target from an included frozen primary omnibus",
        )


def _bootstrap_pairs(
    resampling: FullPipelineResamplingResult,
) -> tuple[tuple[SubjectBootstrapPlan, FullPipelineResampleRecord], ...]:
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    resampling._require_intact()
    if not resampling.retain_children:
        raise ContractError(
            "Resampled attribution requires retained full-pipeline children",
            code="resampled_attribution_children_not_retained",
            field="retain_children",
            remediation="Rerun subject bootstraps with retain_children=True",
        )
    if not resampling.plans or len(resampling.plans) != len(resampling.records):
        raise ContractError(
            "Resampling plans and workflow records are not exactly aligned",
            code="resampled_attribution_plan_alignment_mismatch",
            field="plans,records",
            remediation="Use one intact full-pipeline subject-bootstrap result",
        )
    if any(not isinstance(plan, SubjectBootstrapPlan) for plan in resampling.plans):
        raise ContractError(
            "Resampled attribution accepts subject-bootstrap plans only",
            code="resampled_attribution_nonbootstrap_plan_forbidden",
            field="plans",
            remediation="Use a resampling result containing no context permutations",
        )
    values: list[tuple[SubjectBootstrapPlan, FullPipelineResampleRecord]] = []
    for plan, record in zip(resampling.plans, resampling.records, strict=True):
        if not isinstance(plan, SubjectBootstrapPlan):
            raise TypeError("plans must contain SubjectBootstrapPlan values")
        if not isinstance(record, FullPipelineResampleRecord):
            raise TypeError("records must contain FullPipelineResampleRecord values")
        if (
            record.operation is not FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP
            or record.plan_id != plan.bootstrap_id
            or record.resample_index != plan.resample_index
            or record.exchangeability_id != plan.exchangeability_id
            or record.plan_seed_lineage != plan.seed_lineage
        ):
            raise ContractError(
                "Bootstrap plan and workflow record lineage are misaligned",
                code="resampled_attribution_plan_alignment_mismatch",
                field="operation,plan_id,resample_index,exchangeability_id,seed_lineage",
                remediation="Use intact aligned bootstrap plan/record pairs",
            )
        values.append((plan, record))
    return tuple(
        sorted(values, key=lambda item: (item[0].resample_index, item[0].bootstrap_id))
    )


def _heldout_subject_ids(fold: CrossFitFoldArtifacts) -> tuple[str, ...]:
    values = tuple(fold.application.heldout_subject_ids)
    if not values or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in values
    ):
        raise ValueError("heldout subject IDs must be canonical and non-empty")
    if len(values) != len(set(values)):
        raise ValueError("heldout subject IDs must be unique within a fold")
    return values


def _safe_heldout_subject_ids(fold: CrossFitFoldArtifacts) -> tuple[str, ...]:
    try:
        return _heldout_subject_ids(fold)
    except (AttributeError, TypeError, ValueError):
        return ()


def _optional_source_id(source: object | None, field_name: str) -> str | None:
    if source is None:
        return None
    try:
        return _name(getattr(source, field_name), field_name=field_name)
    except (AttributeError, TypeError, ValueError):
        return None


def _integrity_failure(call: Callable[[], None]) -> str | None:
    try:
        call()
    except Exception as error:
        return _failure_reason(error)
    return None


def _child_crossfit_spec_id(child: CrossFitArtifacts | None) -> str | None:
    if child is None:
        return None
    try:
        return _name(child.spec.spec_id, field_name="child_crossfit_spec_id")
    except (AttributeError, TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True, init=False)
class ResampledFamilySelectionEvent:
    """Producer-owned selection event for one bootstrap plan and outer fold."""

    resampling_result_id: str
    crossfit_spec_id: str
    plan_id: str
    full_pipeline_record_id: str
    resample_index: int
    exchangeability_id: str
    crossfit_id: str | None
    fold_id: str | None
    universe_id: str
    declaration_id: str
    hypothesis_id: str
    target_id: str
    contrast_name: str
    receiver: str
    family_id: str
    mode: str
    scoring_functional_id: str | None
    scoring_application_id: str | None
    incremental_functional_id: str | None
    selection_rule_id: str
    selection_rule_semantics: str
    selection_threshold: float
    score_version: str | None
    tuning_manifest_id: str | None
    selected_penalty_id: str | None
    heldout_subject_ids: tuple[str, ...]
    n_heldout_subjects: int
    n_subject_exposures: int
    family_estimable: bool | None
    family_selected: bool | None
    family_coefficient: float | None
    training_reason_code: str | None
    status: ResampledAttributionEventStatus
    reason_code: str | None
    event_id: str
    _resampling: FullPipelineResamplingResult
    _plan: SubjectBootstrapPlan
    _workflow_record: FullPipelineResampleRecord
    _target: FrozenFamilyEffectTarget
    _child: CrossFitArtifacts | None
    _fold: CrossFitFoldArtifacts | None
    _functional: FamilyCommonScoringFunctional | None
    _application: FamilyCommonScoringApplication | None
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ResampledFamilySelectionEvent is producer-owned; use "
            "summarize_resampled_family_selection()"
        )

    @classmethod
    def _from_sources(
        cls,
        resampling: FullPipelineResamplingResult,
        plan: SubjectBootstrapPlan,
        workflow_record: FullPipelineResampleRecord,
        target: FrozenFamilyEffectTarget,
        *,
        child: CrossFitArtifacts | None,
        fold: CrossFitFoldArtifacts | None,
        functional: FamilyCommonScoringFunctional | None,
        application: FamilyCommonScoringApplication | None,
        heldout_subject_ids: Sequence[str],
        n_subject_exposures: int | None,
        family_estimable: bool | None,
        family_selected: bool | None,
        family_coefficient: float | None,
        training_reason_code: str | None,
        status: ResampledAttributionEventStatus,
        reason_code: str | None,
    ) -> ResampledFamilySelectionEvent:
        heldout = tuple(heldout_subject_ids)
        exposures = len(heldout) if n_subject_exposures is None else n_subject_exposures
        coefficient = None if family_coefficient is None else float(family_coefficient)
        incremental = None if functional is None else functional.incremental_functional
        self = object.__new__(cls)
        values: dict[str, object] = {
            "resampling_result_id": resampling.result_id,
            "crossfit_spec_id": resampling.crossfit_spec_id,
            "plan_id": plan.bootstrap_id,
            "full_pipeline_record_id": workflow_record.record_id,
            "resample_index": plan.resample_index,
            "exchangeability_id": plan.exchangeability_id,
            "crossfit_id": None if child is None else child.crossfit_id,
            "fold_id": None if fold is None else fold.fold_id,
            "universe_id": target.universe_id,
            "declaration_id": target.declaration_id,
            "hypothesis_id": target.hypothesis_id,
            "target_id": target.target_id,
            "contrast_name": target.contrast_name,
            "receiver": target.receiver,
            "family_id": target.family_id,
            "mode": target.mode.value,
            "scoring_functional_id": _optional_source_id(
                functional,
                "family_common_functional_id",
            ),
            "scoring_application_id": _optional_source_id(
                application,
                "application_id",
            ),
            "incremental_functional_id": _optional_source_id(
                incremental,
                "incremental_functional_id",
            ),
            "selection_rule_id": _SELECTION_RULE_ID,
            "selection_rule_semantics": _SELECTION_RULE_SEMANTICS,
            "selection_threshold": _SELECTION_THRESHOLD,
            "score_version": _optional_source_id(functional, "score_version"),
            "tuning_manifest_id": _optional_source_id(
                functional,
                "tuning_manifest_id",
            ),
            "selected_penalty_id": _optional_source_id(
                functional,
                "selected_penalty_id",
            ),
            "heldout_subject_ids": heldout,
            "n_heldout_subjects": len(heldout),
            "n_subject_exposures": exposures,
            "family_estimable": family_estimable,
            "family_selected": family_selected,
            "family_coefficient": coefficient,
            "training_reason_code": training_reason_code,
            "status": ResampledAttributionEventStatus(status),
            "reason_code": reason_code,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_resampling", resampling)
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_workflow_record", workflow_record)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_child", child)
        object.__setattr__(self, "_fold", fold)
        object.__setattr__(self, "_functional", functional)
        object.__setattr__(self, "_application", application)
        object.__setattr__(self, "_producer_marker", _EVENT_MARKER)
        object.__setattr__(
            self,
            "event_id",
            stable_id(
                "resampled_family_selection_event",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact(validate_resampling=False)
        return self

    @property
    def formal_inference_allowed(self) -> bool:
        return False

    @property
    def p_value(self) -> None:
        return None

    @property
    def q_value(self) -> None:
        return None

    @property
    def communication_probability(self) -> None:
        return None

    def _identity_payload(self) -> dict[str, object]:
        return {
            "resampling_result_id": self.resampling_result_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "plan_id": self.plan_id,
            "full_pipeline_record_id": self.full_pipeline_record_id,
            "resample_index": self.resample_index,
            "exchangeability_id": self.exchangeability_id,
            "crossfit_id": self.crossfit_id,
            "fold_id": self.fold_id,
            "universe_id": self.universe_id,
            "declaration_id": self.declaration_id,
            "hypothesis_id": self.hypothesis_id,
            "target_id": self.target_id,
            "contrast_name": self.contrast_name,
            "receiver": self.receiver,
            "family_id": self.family_id,
            "mode": self.mode,
            "scoring_functional_id": self.scoring_functional_id,
            "scoring_application_id": self.scoring_application_id,
            "incremental_functional_id": self.incremental_functional_id,
            "selection_rule_id": self.selection_rule_id,
            "selection_rule_semantics": self.selection_rule_semantics,
            "selection_threshold": self.selection_threshold,
            "score_version": self.score_version,
            "tuning_manifest_id": self.tuning_manifest_id,
            "selected_penalty_id": self.selected_penalty_id,
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "n_heldout_subjects": self.n_heldout_subjects,
            "n_subject_exposures": self.n_subject_exposures,
            "family_estimable": self.family_estimable,
            "family_selected": self.family_selected,
            "family_coefficient": self.family_coefficient,
            "training_reason_code": self.training_reason_code,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "formal_inference_status": _FORMAL_STATUS,
            "p_value": None,
            "q_value": None,
            "communication_probability": None,
            "public_release_status": _PUBLIC_RELEASE_STATUS,
            "performance_status": _PERFORMANCE_STATUS,
        }

    def _require_intact(self, *, validate_resampling: bool = True) -> None:
        try:
            self._target._require_intact()
            if validate_resampling:
                self._resampling._require_intact()
            self._workflow_record._require_intact()
            incremental = (
                None
                if self._functional is None
                else self._functional.incremental_functional
            )
            exact_pair = any(
                plan is self._plan and record is self._workflow_record
                for plan, record in zip(
                    self._resampling.plans,
                    self._resampling.records,
                    strict=True,
                )
            )
            child_valid = self._workflow_record.child is self._child
            source_integrity_failures = tuple(
                reason
                for reason in (
                    (
                        None
                        if self._child is None
                        else _integrity_failure(self._child._require_intact)
                    ),
                    (
                        None
                        if self._functional is None
                        else _integrity_failure(self._functional._require_intact)
                    ),
                    (
                        None
                        if self._application is None
                        else _integrity_failure(self._application._require_intact)
                    ),
                    (
                        None
                        if incremental is None
                        else _integrity_failure(incremental._require_intact)
                    ),
                )
                if reason is not None
            )
            source_integrity_valid = not source_integrity_failures or (
                self.status is ResampledAttributionEventStatus.FAILED
                and self.reason_code in source_integrity_failures
            )
            fold_valid = self._fold is None or (
                self._child is not None
                and any(fold is self._fold for fold in self._child.folds)
            )
            chain_valid = (self._functional is None and self._application is None) or (
                self._fold is not None
                and self._functional is not None
                and self._application is not None
                and any(
                    functional is self._functional and application is self._application
                    for functional, application in zip(
                        self._fold.family_common_functionals,
                        self._fold.family_common_applications,
                        strict=True,
                    )
                )
            )
            heldout = tuple(self.heldout_subject_ids)
            heldout_failure: str | None = None
            exact_heldout: tuple[str, ...] = ()
            if self._fold is not None:
                try:
                    exact_heldout = _heldout_subject_ids(self._fold)
                except Exception as error:
                    heldout_failure = _failure_reason(error)
            heldout_valid = (not heldout_failure and heldout == exact_heldout) or (
                heldout_failure is not None
                and self.status is ResampledAttributionEventStatus.FAILED
                and self.reason_code == heldout_failure
            )
            if self._fold is None:
                heldout_valid = not heldout
            child_spec_id = _child_crossfit_spec_id(self._child)
            child_spec_valid = (
                self._child is None
                or child_spec_id == self._resampling.crossfit_spec_id
                or (
                    self.status is ResampledAttributionEventStatus.FAILED
                    and self.reason_code
                    == "resampled_attribution_child_crossfit_spec_mismatch"
                )
            )
            if (
                isinstance(self.n_subject_exposures, bool)
                or not isinstance(self.n_subject_exposures, int)
                or self.n_subject_exposures < 1
            ):
                raise ValueError("n_subject_exposures must be an integer >= 1")
            expected_exposures = (
                len(self._plan.draws) if self._fold is None else len(exact_heldout)
            )
            exposure_valid = self.n_subject_exposures == expected_exposures
            coefficient = self.family_coefficient
            coefficient_valid = coefficient is None or (
                isinstance(coefficient, float)
                and math.isfinite(coefficient)
                and coefficient >= 0.0
            )
            status_valid = False
            if self.status is ResampledAttributionEventStatus.OBSERVED:
                status_valid = (
                    self._child is not None
                    and self._fold is not None
                    and self._functional is not None
                    and self._application is not None
                    and incremental is not None
                    and self.family_estimable is True
                    and isinstance(self.family_selected, bool)
                    and coefficient is not None
                    and self.family_selected == (coefficient > _SELECTION_THRESHOLD)
                    and self.training_reason_code is None
                    and self.reason_code is None
                    and self.score_version is not None
                    and self.tuning_manifest_id is not None
                    and self.selected_penalty_id is not None
                    and bool(heldout)
                )
            elif self.status is ResampledAttributionEventStatus.NOT_ESTIMABLE:
                status_valid = (
                    self._child is not None
                    and self._fold is not None
                    and self.family_selected is None
                    and bool(self.reason_code)
                )
            else:
                status_valid = self.family_selected is None and bool(self.reason_code)
            workflow_failure_valid = True
            if self._workflow_record.status is FullPipelineResampleStatus.FAILED:
                workflow_failure_valid = (
                    self.status is ResampledAttributionEventStatus.FAILED
                    and self._child is None
                    and self._fold is None
                    and self.reason_code == self._workflow_record.failure_code
                )
            expected_id = stable_id(
                "resampled_family_selection_event",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _EVENT_MARKER
                and exact_pair
                and child_valid
                and source_integrity_valid
                and fold_valid
                and chain_valid
                and heldout_valid
                and child_spec_valid
                and exposure_valid
                and coefficient_valid
                and workflow_failure_valid
                and self.resampling_result_id == self._resampling.result_id
                and self.crossfit_spec_id == self._resampling.crossfit_spec_id
                and self.plan_id == self._plan.bootstrap_id
                and self.full_pipeline_record_id == self._workflow_record.record_id
                and self.resample_index == self._plan.resample_index
                and self.exchangeability_id == self._plan.exchangeability_id
                and self.crossfit_id
                == (None if self._child is None else self._child.crossfit_id)
                and self.fold_id == (None if self._fold is None else self._fold.fold_id)
                and self.target_id == self._target.target_id
                and self.universe_id == self._target.universe_id
                and self.declaration_id == self._target.declaration_id
                and self.hypothesis_id == self._target.hypothesis_id
                and self.contrast_name == self._target.contrast_name
                and self.receiver == self._target.receiver
                and self.family_id == self._target.family_id
                and self.mode == self._target.mode.value
                and self.scoring_functional_id
                == _optional_source_id(
                    self._functional,
                    "family_common_functional_id",
                )
                and self.scoring_application_id
                == _optional_source_id(self._application, "application_id")
                and self.incremental_functional_id
                == _optional_source_id(incremental, "incremental_functional_id")
                and self.selection_rule_id == _SELECTION_RULE_ID
                and self.selection_rule_semantics == _SELECTION_RULE_SEMANTICS
                and self.selection_threshold == _SELECTION_THRESHOLD
                and self.score_version
                == _optional_source_id(self._functional, "score_version")
                and self.tuning_manifest_id
                == _optional_source_id(self._functional, "tuning_manifest_id")
                and self.selected_penalty_id
                == _optional_source_id(self._functional, "selected_penalty_id")
                and self.n_heldout_subjects == len(heldout)
                and len(heldout) == len(set(heldout))
                and status_valid
                and self.event_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Resampled family selection event failed integrity validation",
                code="resampled_attribution_event_integrity_violation",
                field="event_id",
                remediation="Rebuild events from intact bootstrap children",
            ) from error
        if not valid:
            raise ContractError(
                "Resampled family selection event failed integrity validation",
                code="resampled_attribution_event_integrity_violation",
                field="event_id",
                remediation="Rebuild events from intact bootstrap children",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "event_id": self.event_id,
            **self._identity_payload(),
            "formal_inference_allowed": False,
        }


def _event(
    resampling: FullPipelineResamplingResult,
    plan: SubjectBootstrapPlan,
    workflow_record: FullPipelineResampleRecord,
    target: FrozenFamilyEffectTarget,
    *,
    child: CrossFitArtifacts | None,
    fold: CrossFitFoldArtifacts | None,
    functional: FamilyCommonScoringFunctional | None = None,
    application: FamilyCommonScoringApplication | None = None,
    heldout_subject_ids: Sequence[str] = (),
    n_subject_exposures: int | None = None,
    family_estimable: bool | None = None,
    family_selected: bool | None = None,
    family_coefficient: float | None = None,
    training_reason_code: str | None = None,
    status: ResampledAttributionEventStatus,
    reason_code: str | None,
) -> ResampledFamilySelectionEvent:
    return ResampledFamilySelectionEvent._from_sources(
        resampling,
        plan,
        workflow_record,
        target,
        child=child,
        fold=fold,
        functional=functional,
        application=application,
        heldout_subject_ids=heldout_subject_ids,
        n_subject_exposures=n_subject_exposures,
        family_estimable=family_estimable,
        family_selected=family_selected,
        family_coefficient=family_coefficient,
        training_reason_code=training_reason_code,
        status=status,
        reason_code=reason_code,
    )


def _fold_event(
    resampling: FullPipelineResamplingResult,
    plan: SubjectBootstrapPlan,
    workflow_record: FullPipelineResampleRecord,
    target: FrozenFamilyEffectTarget,
    child: CrossFitArtifacts,
    fold: CrossFitFoldArtifacts,
) -> ResampledFamilySelectionEvent:
    functional: FamilyCommonScoringFunctional | None = None
    application: FamilyCommonScoringApplication | None = None
    heldout = _safe_heldout_subject_ids(fold)
    try:
        heldout = _heldout_subject_ids(fold)
        functionals = tuple(fold.family_common_functionals)
        applications = tuple(fold.family_common_applications)
        if len(functionals) != len(applications):
            raise ContractError(
                "Fold family-common functionals and applications are not aligned",
                code="resampled_attribution_chain_alignment_mismatch",
                field="family_common_functionals,family_common_applications",
                remediation="Use an intact cross-fit child fold",
            )
        matches = tuple(
            (candidate_functional, candidate_application)
            for candidate_functional, candidate_application in zip(
                functionals,
                applications,
                strict=True,
            )
            if candidate_functional.contrast_name == target.contrast_name
            and candidate_functional.receiver == target.receiver
        )
        if not matches:
            return _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=child,
                fold=fold,
                heldout_subject_ids=heldout,
                status=ResampledAttributionEventStatus.NOT_ESTIMABLE,
                reason_code="family_common_target_chain_missing",
            )
        if len(matches) != 1:
            raise ContractError(
                "Fold contains duplicate target family-common chains",
                code="duplicate_resampled_attribution_target_chain",
                field="contrast_name,receiver",
                remediation="Use a fold with one chain per contrast and receiver",
            )
        functional, application = matches[0]
        functional._require_intact()
        application._require_intact()
        if (
            application.functional is not functional
            or application.functional.family_common_functional_id
            != functional.family_common_functional_id
        ):
            raise ContractError(
                "Family-common application does not belong to its matched functional",
                code="resampled_attribution_application_functional_mismatch",
                field="family_common_functional_id",
                remediation="Use the exact fold functional/application pair",
            )
        if target.family_id not in functional.family_ids:
            return _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=child,
                fold=fold,
                functional=functional,
                application=application,
                heldout_subject_ids=heldout,
                status=ResampledAttributionEventStatus.NOT_ESTIMABLE,
                reason_code="family_not_in_training_fold_universe",
            )
        if target.family_id not in functional.active_family_ids:
            return _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=child,
                fold=fold,
                functional=functional,
                application=application,
                heldout_subject_ids=heldout,
                status=ResampledAttributionEventStatus.NOT_ESTIMABLE,
                reason_code="family_not_active_in_outer_training_fold",
            )
        incremental = functional.incremental_functional
        if incremental is None:
            reason = (
                functional.incremental_reason_code
                or "outer_training_incremental_functional_missing"
            )
            return _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=child,
                fold=fold,
                functional=functional,
                application=application,
                heldout_subject_ids=heldout,
                training_reason_code=reason,
                status=ResampledAttributionEventStatus.NOT_ESTIMABLE,
                reason_code=reason,
            )
        incremental._require_intact()
        if target.family_id not in incremental.family_ids:
            raise ContractError(
                "Active family is absent from the incremental training functional",
                code="resampled_attribution_incremental_family_mismatch",
                field="family_id",
                remediation="Use an intact family-common training functional",
            )
        family_index = incremental.family_ids.index(target.family_id)
        estimable = bool(incremental.family_estimable[family_index])
        coefficient = float(incremental.family_coefficients[family_index])
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("outer-training family coefficient must be non-negative")
        training_reason = incremental.family_reason_codes[family_index]
        if not estimable:
            reason = training_reason or "outer_training_family_not_estimable"
            return _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=child,
                fold=fold,
                functional=functional,
                application=application,
                heldout_subject_ids=heldout,
                family_estimable=False,
                family_coefficient=coefficient,
                training_reason_code=reason,
                status=ResampledAttributionEventStatus.NOT_ESTIMABLE,
                reason_code=reason,
            )
        if training_reason is not None:
            raise ContractError(
                "Estimable outer-training family retains an unavailable reason",
                code="resampled_attribution_training_reason_mismatch",
                field="family_estimable,family_reason_codes",
                remediation="Use an intact incremental training functional",
            )
        selected = coefficient > _SELECTION_THRESHOLD
        return _event(
            resampling,
            plan,
            workflow_record,
            target,
            child=child,
            fold=fold,
            functional=functional,
            application=application,
            heldout_subject_ids=heldout,
            family_estimable=True,
            family_selected=selected,
            family_coefficient=coefficient,
            training_reason_code=None,
            status=ResampledAttributionEventStatus.OBSERVED,
            reason_code=None,
        )
    except Exception as error:
        return _event(
            resampling,
            plan,
            workflow_record,
            target,
            child=child,
            fold=fold,
            functional=functional,
            application=application,
            heldout_subject_ids=heldout,
            status=ResampledAttributionEventStatus.FAILED,
            reason_code=_failure_reason(error),
        )


def _events_for_pair(
    resampling: FullPipelineResamplingResult,
    plan: SubjectBootstrapPlan,
    workflow_record: FullPipelineResampleRecord,
    target: FrozenFamilyEffectTarget,
) -> tuple[ResampledFamilySelectionEvent, ...]:
    if workflow_record.status is FullPipelineResampleStatus.FAILED:
        return (
            _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=None,
                fold=None,
                n_subject_exposures=len(plan.draws),
                status=ResampledAttributionEventStatus.FAILED,
                reason_code=(
                    workflow_record.failure_code
                    or "full_pipeline_subject_bootstrap_failed"
                ),
            ),
        )
    child = workflow_record.child
    if child is None:
        return (
            _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=None,
                fold=None,
                n_subject_exposures=len(plan.draws),
                status=ResampledAttributionEventStatus.FAILED,
                reason_code="resampled_attribution_child_missing",
            ),
        )
    try:
        folds = tuple(child.folds)
    except (AttributeError, TypeError, ValueError) as error:
        return (
            _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=child,
                fold=None,
                n_subject_exposures=len(plan.draws),
                status=ResampledAttributionEventStatus.FAILED,
                reason_code=_failure_reason(error),
            ),
        )
    if not folds:
        return (
            _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=child,
                fold=None,
                n_subject_exposures=len(plan.draws),
                status=ResampledAttributionEventStatus.FAILED,
                reason_code="resampled_attribution_child_folds_missing",
            ),
        )
    child_spec_id = _child_crossfit_spec_id(child)
    if child_spec_id != resampling.crossfit_spec_id:
        return tuple(
            _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=child,
                fold=fold,
                heldout_subject_ids=_safe_heldout_subject_ids(fold),
                status=ResampledAttributionEventStatus.FAILED,
                reason_code="resampled_attribution_child_crossfit_spec_mismatch",
            )
            for fold in folds
        )
    try:
        child._require_intact()
    except Exception as error:
        reason = _failure_reason(error)
        return tuple(
            _event(
                resampling,
                plan,
                workflow_record,
                target,
                child=child,
                fold=fold,
                heldout_subject_ids=_safe_heldout_subject_ids(fold),
                status=ResampledAttributionEventStatus.FAILED,
                reason_code=reason,
            )
            for fold in folds
        )
    return tuple(
        _fold_event(
            resampling,
            plan,
            workflow_record,
            target,
            child,
            fold,
        )
        for fold in sorted(folds, key=lambda item: item.fold_id)
    )


def _build_events(
    resampling: FullPipelineResamplingResult,
    target: FrozenFamilyEffectTarget,
) -> tuple[ResampledFamilySelectionEvent, ...]:
    return tuple(
        event
        for plan, record in _bootstrap_pairs(resampling)
        for event in _events_for_pair(resampling, plan, record, target)
    )


@dataclass(frozen=True, slots=True)
class _Summary:
    n_event_rows_total: int
    n_observed_outer_fold_events: int
    n_not_estimable_outer_fold_events: int
    n_failed_event_rows: int
    n_foldless_failed_plan_events: int
    n_selected_observed_outer_fold_events: int
    outer_fold_denominator_complete: bool
    diagnostic_conditional_observed_outer_fold_selection_frequency: float | None
    n_subject_exposures_total: int
    n_subject_exposures_estimable: int
    n_subject_exposures_not_estimable: int
    n_subject_exposures_failed: int
    n_subject_exposures_selected: int
    subject_exposure_accounting_complete: bool
    diagnostic_subject_exposure_weighted_selection_frequency: float | None


def _summarize_events(events: Sequence[ResampledFamilySelectionEvent]) -> _Summary:
    supplied = tuple(events)
    counts = Counter(event.status for event in supplied)
    observed = tuple(
        event
        for event in supplied
        if event.status is ResampledAttributionEventStatus.OBSERVED
    )
    n_selected = sum(event.family_selected is True for event in observed)
    n_estimable = len(observed)
    fit_frequency = None if not n_estimable else n_selected / n_estimable
    exposure_total = sum(event.n_subject_exposures for event in supplied)
    exposure_estimable = sum(event.n_subject_exposures for event in observed)
    exposure_not_estimable = sum(
        event.n_subject_exposures
        for event in supplied
        if event.status is ResampledAttributionEventStatus.NOT_ESTIMABLE
    )
    exposure_failed = sum(
        event.n_subject_exposures
        for event in supplied
        if event.status is ResampledAttributionEventStatus.FAILED
    )
    exposure_selected = sum(
        event.n_subject_exposures for event in observed if event.family_selected is True
    )
    exposure_frequency = (
        None if not exposure_estimable else exposure_selected / exposure_estimable
    )
    return _Summary(
        n_event_rows_total=len(supplied),
        n_observed_outer_fold_events=n_estimable,
        n_not_estimable_outer_fold_events=counts[
            ResampledAttributionEventStatus.NOT_ESTIMABLE
        ],
        n_failed_event_rows=counts[ResampledAttributionEventStatus.FAILED],
        n_foldless_failed_plan_events=sum(
            event.status is ResampledAttributionEventStatus.FAILED
            and event.fold_id is None
            for event in supplied
        ),
        n_selected_observed_outer_fold_events=n_selected,
        outer_fold_denominator_complete=all(
            event.fold_id is not None for event in supplied
        ),
        diagnostic_conditional_observed_outer_fold_selection_frequency=(fit_frequency),
        n_subject_exposures_total=exposure_total,
        n_subject_exposures_estimable=exposure_estimable,
        n_subject_exposures_not_estimable=exposure_not_estimable,
        n_subject_exposures_failed=exposure_failed,
        n_subject_exposures_selected=exposure_selected,
        subject_exposure_accounting_complete=True,
        diagnostic_subject_exposure_weighted_selection_frequency=(exposure_frequency),
    )


@dataclass(frozen=True, slots=True, init=False)
class FrozenResampledAttributionCollection:
    """Producer-owned diagnostic selection summary over exact bootstrap plans."""

    resampling_result_id: str
    crossfit_spec_id: str
    target_id: str
    universe_id: str
    hypothesis_id: str
    bootstrap_plan_ids: tuple[str, ...]
    full_pipeline_record_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    events: tuple[ResampledFamilySelectionEvent, ...]
    n_bootstrap_plans_total: int
    n_bootstrap_plans_succeeded: int
    n_bootstrap_plans_failed: int
    n_event_rows_total: int
    n_observed_outer_fold_events: int
    n_not_estimable_outer_fold_events: int
    n_failed_event_rows: int
    n_foldless_failed_plan_events: int
    n_selected_observed_outer_fold_events: int
    outer_fold_denominator_complete: bool
    diagnostic_conditional_observed_outer_fold_selection_frequency: float | None
    n_subject_exposures_total: int
    n_subject_exposures_estimable: int
    n_subject_exposures_not_estimable: int
    n_subject_exposures_failed: int
    n_subject_exposures_selected: int
    subject_exposure_accounting_complete: bool
    diagnostic_subject_exposure_weighted_selection_frequency: float | None
    collection_id: str
    _resampling: FullPipelineResamplingResult
    _target: FrozenFamilyEffectTarget
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenResampledAttributionCollection is producer-owned; use "
            "summarize_resampled_family_selection()"
        )

    @classmethod
    def _from_events(
        cls,
        resampling: FullPipelineResamplingResult,
        target: FrozenFamilyEffectTarget,
        events: Sequence[ResampledFamilySelectionEvent],
    ) -> FrozenResampledAttributionCollection:
        pairs = _bootstrap_pairs(resampling)
        ordered = tuple(
            sorted(
                events,
                key=lambda item: (
                    item.resample_index,
                    item.plan_id,
                    item.fold_id is None,
                    item.fold_id or "",
                ),
            )
        )
        observed_rules = {
            (
                event.selection_rule_id,
                event.selection_rule_semantics,
                event.selection_threshold,
                event.score_version,
            )
            for event in ordered
            if event.status is ResampledAttributionEventStatus.OBSERVED
        }
        if len(observed_rules) > 1:
            raise ContractError(
                "Observed outer folds use inconsistent selection rules or "
                "score versions",
                code="resampled_attribution_observed_rule_mismatch",
                field="selection_rule_id,selection_threshold,score_version",
                remediation="Use one frozen training selection rule across all folds",
            )
        summary = _summarize_events(ordered)
        workflow_counts = Counter(record.status for _, record in pairs)
        self = object.__new__(cls)
        values: dict[str, object] = {
            "resampling_result_id": resampling.result_id,
            "crossfit_spec_id": resampling.crossfit_spec_id,
            "target_id": target.target_id,
            "universe_id": target.universe_id,
            "hypothesis_id": target.hypothesis_id,
            "bootstrap_plan_ids": tuple(plan.bootstrap_id for plan, _ in pairs),
            "full_pipeline_record_ids": tuple(record.record_id for _, record in pairs),
            "event_ids": tuple(event.event_id for event in ordered),
            "events": ordered,
            "n_bootstrap_plans_total": len(pairs),
            "n_bootstrap_plans_succeeded": workflow_counts[
                FullPipelineResampleStatus.SUCCEEDED
            ],
            "n_bootstrap_plans_failed": workflow_counts[
                FullPipelineResampleStatus.FAILED
            ],
            **{
                field_name: getattr(summary, field_name)
                for field_name in summary.__dataclass_fields__
            },
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_resampling", resampling)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_producer_marker", _COLLECTION_MARKER)
        object.__setattr__(
            self,
            "collection_id",
            stable_id(
                "frozen_resampled_attribution_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    @property
    def formal_inference_allowed(self) -> bool:
        return False

    @property
    def p_value(self) -> None:
        return None

    @property
    def q_value(self) -> None:
        return None

    @property
    def communication_probability(self) -> None:
        return None

    @property
    def diagnostic_status(self) -> str:
        if self.n_observed_outer_fold_events == 0:
            return "diagnostic_not_estimable"
        if self.n_not_estimable_outer_fold_events or self.n_failed_event_rows:
            return "diagnostic_partial"
        return "diagnostic_complete"

    def _identity_payload(self) -> dict[str, object]:
        return {
            "resampling_result_id": self.resampling_result_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "target_id": self.target_id,
            "universe_id": self.universe_id,
            "hypothesis_id": self.hypothesis_id,
            "bootstrap_plan_ids": list(self.bootstrap_plan_ids),
            "full_pipeline_record_ids": list(self.full_pipeline_record_ids),
            "event_ids": list(self.event_ids),
            "n_bootstrap_plans_total": self.n_bootstrap_plans_total,
            "n_bootstrap_plans_succeeded": self.n_bootstrap_plans_succeeded,
            "n_bootstrap_plans_failed": self.n_bootstrap_plans_failed,
            "n_event_rows_total": self.n_event_rows_total,
            "n_observed_outer_fold_events": self.n_observed_outer_fold_events,
            "n_not_estimable_outer_fold_events": (
                self.n_not_estimable_outer_fold_events
            ),
            "n_failed_event_rows": self.n_failed_event_rows,
            "n_foldless_failed_plan_events": self.n_foldless_failed_plan_events,
            "n_selected_observed_outer_fold_events": (
                self.n_selected_observed_outer_fold_events
            ),
            "outer_fold_denominator_complete": (self.outer_fold_denominator_complete),
            "diagnostic_conditional_observed_outer_fold_selection_frequency": (
                self.diagnostic_conditional_observed_outer_fold_selection_frequency
            ),
            "n_subject_exposures_total": self.n_subject_exposures_total,
            "n_subject_exposures_estimable": self.n_subject_exposures_estimable,
            "n_subject_exposures_not_estimable": (
                self.n_subject_exposures_not_estimable
            ),
            "n_subject_exposures_failed": self.n_subject_exposures_failed,
            "n_subject_exposures_selected": self.n_subject_exposures_selected,
            "subject_exposure_accounting_complete": (
                self.subject_exposure_accounting_complete
            ),
            "diagnostic_subject_exposure_weighted_selection_frequency": (
                self.diagnostic_subject_exposure_weighted_selection_frequency
            ),
            "formal_inference_status": _FORMAL_STATUS,
            "p_value": None,
            "q_value": None,
            "communication_probability": None,
            "public_release_status": _PUBLIC_RELEASE_STATUS,
            "performance_status": _PERFORMANCE_STATUS,
        }

    def _require_exact_plan_coverage(
        self,
        pairs: Sequence[tuple[SubjectBootstrapPlan, FullPipelineResampleRecord]],
    ) -> None:
        by_plan: defaultdict[str, list[ResampledFamilySelectionEvent]] = defaultdict(
            list
        )
        for event in self.events:
            by_plan[event.plan_id].append(event)
        expected_plan_ids = {plan.bootstrap_id for plan, _ in pairs}
        if set(by_plan) != expected_plan_ids:
            raise ContractError(
                "Resampled attribution does not exactly cover bootstrap plans",
                code="resampled_attribution_bootstrap_coverage_mismatch",
                field="plan_id",
                remediation="Retain at least one typed event for every bootstrap plan",
            )
        for plan, record in pairs:
            plan_events = by_plan[plan.bootstrap_id]
            if any(
                event.full_pipeline_record_id != record.record_id
                for event in plan_events
            ):
                raise ContractError(
                    "Bootstrap events bind a different workflow record",
                    code="resampled_attribution_record_coverage_mismatch",
                    field="full_pipeline_record_id",
                    remediation="Rebuild events from exact plan/record pairs",
                )
            if sum(event.n_subject_exposures for event in plan_events) != len(
                plan.draws
            ):
                raise ContractError(
                    "Bootstrap event subject exposure does not equal its draw count",
                    code="resampled_attribution_subject_exposure_mismatch",
                    field="n_subject_exposures",
                    remediation="Account for every bootstrap subject draw exactly once",
                )
            if record.status is FullPipelineResampleStatus.FAILED:
                valid = (
                    len(plan_events) == 1
                    and plan_events[0].fold_id is None
                    and plan_events[0].status is ResampledAttributionEventStatus.FAILED
                )
            else:
                child = record.child
                if child is None or not child.folds:
                    valid = (
                        len(plan_events) == 1
                        and plan_events[0].fold_id is None
                        and plan_events[0].status
                        is ResampledAttributionEventStatus.FAILED
                    )
                else:
                    valid = {event.fold_id for event in plan_events} == {
                        fold.fold_id for fold in child.folds
                    } and all(event.fold_id is not None for event in plan_events)
            if not valid:
                raise ContractError(
                    "Bootstrap plan does not have exact typed fold-event coverage",
                    code="resampled_attribution_fold_coverage_mismatch",
                    field="plan_id,fold_id",
                    remediation=(
                        "Emit every successful child fold or one failed plan row"
                    ),
                )

    def _require_intact(self) -> None:
        try:
            _require_target(self._target)
            pairs = _bootstrap_pairs(self._resampling)
            for event in self.events:
                event._require_intact(validate_resampling=False)
            self._require_exact_plan_coverage(pairs)
            expected_events = _build_events(self._resampling, self._target)
            expected_summary = _summarize_events(expected_events)
            workflow_counts = Counter(record.status for _, record in pairs)
            expected_id = stable_id(
                "frozen_resampled_attribution_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _COLLECTION_MARKER
                and self.resampling_result_id == self._resampling.result_id
                and self.crossfit_spec_id == self._resampling.crossfit_spec_id
                and self.target_id == self._target.target_id
                and self.universe_id == self._target.universe_id
                and self.hypothesis_id == self._target.hypothesis_id
                and self.bootstrap_plan_ids
                == tuple(plan.bootstrap_id for plan, _ in pairs)
                and self.full_pipeline_record_ids
                == tuple(record.record_id for _, record in pairs)
                and self.event_ids == tuple(event.event_id for event in self.events)
                and tuple(event.event_id for event in self.events)
                == tuple(event.event_id for event in expected_events)
                and self.n_bootstrap_plans_total == len(pairs)
                and self.n_bootstrap_plans_succeeded
                == workflow_counts[FullPipelineResampleStatus.SUCCEEDED]
                and self.n_bootstrap_plans_failed
                == workflow_counts[FullPipelineResampleStatus.FAILED]
                and all(
                    getattr(self, field_name) == getattr(expected_summary, field_name)
                    for field_name in expected_summary.__dataclass_fields__
                )
                and self.collection_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Resampled attribution collection failed integrity validation",
                code="resampled_attribution_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute from the intact bootstrap result and target",
            ) from error
        if not valid:
            raise ContractError(
                "Resampled attribution collection failed integrity validation",
                code="resampled_attribution_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute from the intact bootstrap result and target",
            )

    def events_for_plan(
        self,
        plan_id: str,
    ) -> tuple[ResampledFamilySelectionEvent, ...]:
        self._require_intact()
        identifier = _name(plan_id, field_name="plan_id")
        matched = tuple(event for event in self.events if event.plan_id == identifier)
        if not matched:
            raise ContractError(
                "Bootstrap plan is absent from the resampled attribution collection",
                code="resampled_attribution_plan_not_found",
                field="plan_id",
                remediation="Use an exact bootstrap plan ID from this collection",
            )
        return matched

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "collection_id": self.collection_id,
            **self._identity_payload(),
            "diagnostic_status": self.diagnostic_status,
            "formal_inference_allowed": False,
            "events": [event.to_dict() for event in self.events],
        }


def summarize_resampled_family_selection(
    resampling: FullPipelineResamplingResult,
    target: FrozenFamilyEffectTarget,
) -> FrozenResampledAttributionCollection:
    """Collect conditional family selection over full-pipeline bootstraps."""

    _require_target(target)
    events = _build_events(resampling, target)
    return FrozenResampledAttributionCollection._from_events(
        resampling,
        target,
        events,
    )


__all__ = [
    "FrozenResampledAttributionCollection",
    "ResampledAttributionEventStatus",
    "ResampledFamilySelectionEvent",
    "summarize_resampled_family_selection",
]
