"""Complete-universe workflow authentication for selection frequency."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from crychic.core import CommunicationMode, ContractError, CrychicError, stable_id
from crychic.inference.bootstrap_selection import (
    BootstrapSelectionOpportunity,
    BootstrapSelectionRecord,
    SelectionFrequencyResult,
    SelectionFrequencySpec,
    SelectionFrequencyStatus,
    summarize_selection_frequency,
)
from crychic.inference.hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisDeclaration,
    HypothesisPrefilterStatus,
    HypothesisRole,
)
from crychic.resampling import SubjectBootstrapPlan
from crychic.scoring import FAMILY_COMMON_SCORE_VERSION

from .full_pipeline_effects import (
    FrozenFamilyEffectTarget,
    build_frozen_family_effect_target,
)
from .full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
    aligned_full_pipeline_resamples,
)
from .resampled_attribution import (
    FrozenResampledAttributionCollection,
    ResampledAttributionEventStatus,
    ResampledFamilySelectionEvent,
    summarize_resampled_family_selection,
)

_SCHEMA_VERSION = "1.0.0"
_PRIMARY_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_SOURCE_BINDING_STATUS = "frozen_v03_11_selection_events_authenticated_v1"
_BATCH_SEMANTICS = "exact_applicable_primary_hypothesis_universe_coverage_v1"
_RELEASE_SCOPE = "observed_included_rows_after_exact_universe_batch_only_v1"
_PERFORMANCE_STATUS = "serial_universe_batch_parallel_execution_deferred_v1"
_CHILD_MARKER = "crychic.workflow.frozen_family_selection_frequency.v1"
_RECORD_MARKER = "crychic.workflow.selection_frequency_universe_record.v1"
_COLLECTION_MARKER = "crychic.workflow.selection_frequency_universe.v1"
_EXCLUDED_OUTPUT_KINDS = (
    "p_value",
    "q_value",
    "posterior_probability",
    "communication_probability",
)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _failure_reason(error: Exception) -> str:
    if isinstance(error, CrychicError):
        reason: str = error.details.code
        return reason
    return f"selection_frequency_adapter_failed_{type(error).__qualname__}"


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


def _applicable_declarations(
    universe: FrozenHypothesisUniverse,
) -> tuple[HypothesisDeclaration, ...]:
    if not isinstance(universe, FrozenHypothesisUniverse):
        raise TypeError("universe must be a FrozenHypothesisUniverse")
    universe._require_intact()
    applicable = tuple(
        declaration
        for declaration in universe.declarations
        if declaration.role is HypothesisRole.PRIMARY
        and declaration.endpoint == _PRIMARY_ENDPOINT
        and declaration.mode is CommunicationMode.STATE
    )
    if not applicable:
        raise _contract_error(
            "Frozen universe has no applicable primary family omnibus",
            code="selection_frequency_universe_primary_missing",
            field="role,endpoint",
            remediation="Freeze at least one driver-family primary omnibus",
        )
    return applicable


@dataclass(frozen=True, slots=True)
class _BootstrapLineage:
    plan_ids: tuple[str, ...]
    workflow_record_ids: tuple[str, ...]


def _bootstrap_lineage(
    resampling: FullPipelineResamplingResult,
) -> _BootstrapLineage:
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    resampling._require_intact()
    if not resampling.retain_children:
        raise _contract_error(
            "Selection frequency requires retained full-pipeline children",
            code="selection_frequency_universe_children_not_retained",
            field="retain_children",
            remediation="Rerun subject bootstraps with retain_children=True",
        )
    all_pairs = aligned_full_pipeline_resamples(resampling)
    bootstrap_pairs = aligned_full_pipeline_resamples(
        resampling,
        operation=FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP,
    )
    if not bootstrap_pairs or len(bootstrap_pairs) != len(all_pairs):
        raise _contract_error(
            "Selection frequency universe accepts subject bootstraps only",
            code="selection_frequency_universe_nonbootstrap_forbidden",
            field="operation",
            remediation="Use a bootstrap-only full-pipeline result",
        )
    values: list[tuple[int, str, str]] = []
    for plan, record in bootstrap_pairs:
        if not isinstance(plan, SubjectBootstrapPlan) or not isinstance(
            record, FullPipelineResampleRecord
        ):
            raise TypeError("bootstrap lineage must contain typed plan/record pairs")
        if (
            record.operation is not FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP
            or record.plan_id != plan.bootstrap_id
            or record.resample_index != plan.resample_index
        ):
            raise _contract_error(
                "Bootstrap plan and workflow record lineage are misaligned",
                code="selection_frequency_universe_plan_alignment_mismatch",
                field="plan_id,resample_index,operation",
                remediation="Use one intact full-pipeline bootstrap result",
            )
        values.append((plan.resample_index, plan.bootstrap_id, record.record_id))
    if len({item[1] for item in values}) != len(values) or len(
        {item[0] for item in values}
    ) != len(values):
        raise _contract_error(
            "Bootstrap plan lineage contains duplicate provenance",
            code="selection_frequency_universe_duplicate_plan",
            field="plan_id,resample_index",
            remediation="Use unique subject-bootstrap plans",
        )
    ordered = tuple(sorted(values))
    return _BootstrapLineage(
        plan_ids=tuple(item[1] for item in ordered),
        workflow_record_ids=tuple(item[2] for item in ordered),
    )


def _resolved_event_lineage(
    source: FrozenResampledAttributionCollection,
) -> tuple[str, str]:
    rules = {event.selection_rule_id for event in source.events}
    if len(rules) != 1:
        raise _contract_error(
            "V03-11 events do not share one frozen selection rule",
            code="frozen_selection_frequency_rule_mismatch",
            field="selection_rule_id",
            remediation="Rebuild the intact single-rule V03-11 collection",
        )
    versions = {
        event.score_version
        for event in source.events
        if event.score_version is not None
    }
    if versions and versions != {FAMILY_COMMON_SCORE_VERSION}:
        raise _contract_error(
            "V03-11 events do not use the released family-common score version",
            code="frozen_selection_frequency_score_version_mismatch",
            field="score_version",
            remediation="Use one frozen score version across bootstrap folds",
        )
    return next(iter(rules)), FAMILY_COMMON_SCORE_VERSION


def _numeric_sources(
    source: FrozenResampledAttributionCollection,
    target: FrozenFamilyEffectTarget,
) -> tuple[
    SelectionFrequencySpec,
    tuple[BootstrapSelectionRecord, ...],
    SelectionFrequencyResult,
]:
    source._require_intact()
    target._require_intact()
    if not source.events or any(
        not isinstance(event, ResampledFamilySelectionEvent) for event in source.events
    ):
        raise TypeError("V03-11 source must contain typed selection events")
    if any(
        not isinstance(event.status, ResampledAttributionEventStatus)
        for event in source.events
    ):
        raise TypeError("V03-11 source events have invalid status values")
    if (
        source._target is not target
        or source.target_id != target.target_id
        or source.universe_id != target.universe_id
        or source.hypothesis_id != target.hypothesis_id
    ):
        raise _contract_error(
            "V03-11 collection does not belong to the frozen target",
            code="frozen_selection_frequency_target_mismatch",
            field="target_id,universe_id,hypothesis_id",
            remediation="Summarize selection from the exact frozen target",
        )
    rule_id, score_version = _resolved_event_lineage(source)
    opportunities = tuple(
        BootstrapSelectionOpportunity(
            plan_id=event.plan_id,
            resample_index=event.resample_index,
            fold_id=event.fold_id,
        )
        for event in source.events
    )
    records = tuple(
        BootstrapSelectionRecord(
            plan_id=event.plan_id,
            resample_index=event.resample_index,
            fold_id=event.fold_id,
            target_id=event.target_id,
            hypothesis_universe_id=event.universe_id,
            hypothesis_id=event.hypothesis_id,
            crossfit_spec_id=event.crossfit_spec_id,
            selection_rule_id=rule_id,
            score_version=score_version,
            selected=event.family_selected,
            status=SelectionFrequencyStatus(event.status.value),
            reason_code=event.reason_code,
        )
        for event in source.events
    )
    spec = SelectionFrequencySpec(
        target_id=target.target_id,
        hypothesis_universe_id=target.universe_id,
        hypothesis_id=target.hypothesis_id,
        crossfit_spec_id=source.crossfit_spec_id,
        selection_rule_id=rule_id,
        score_version=score_version,
        subject_bootstrap_plan_ids=source.bootstrap_plan_ids,
        expected_opportunities=opportunities,
    )
    result = summarize_selection_frequency(spec, records)
    return spec, records, result


@dataclass(frozen=True, slots=True, init=False)
class FrozenFamilySelectionFrequency:
    """Authenticated single-target child; universe batching is still required."""

    target_id: str
    declaration_id: str
    universe_id: str
    hypothesis_id: str
    resampling_result_id: str
    crossfit_spec_id: str
    bootstrap_plan_ids: tuple[str, ...]
    workflow_record_ids: tuple[str, ...]
    source_collection_id: str
    source_event_ids: tuple[str, ...]
    selection_rule_id: str
    score_version: str
    opportunity_manifest_id: str
    numeric_spec_id: str
    numeric_result_id: str
    selection_frequency: float | None
    status: SelectionFrequencyStatus
    reason_code: str | None
    n_bootstrap_plans_total: int
    n_event_rows_total: int
    n_selected_event_rows: int
    result_id: str
    _resampling: FullPipelineResamplingResult
    _target: FrozenFamilyEffectTarget
    _source: FrozenResampledAttributionCollection
    _numeric_spec: SelectionFrequencySpec
    _numeric_records: tuple[BootstrapSelectionRecord, ...]
    _numeric_result: SelectionFrequencyResult
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenFamilySelectionFrequency is producer-owned; use "
            "summarize_frozen_family_selection_frequency()"
        )

    @classmethod
    def _from_sources(
        cls,
        resampling: FullPipelineResamplingResult,
        target: FrozenFamilyEffectTarget,
        source: FrozenResampledAttributionCollection,
    ) -> FrozenFamilySelectionFrequency:
        lineage = _bootstrap_lineage(resampling)
        if (
            source._resampling is not resampling
            or source.resampling_result_id != resampling.result_id
            or source.crossfit_spec_id != resampling.crossfit_spec_id
            or source.bootstrap_plan_ids != lineage.plan_ids
            or source.full_pipeline_record_ids != lineage.workflow_record_ids
        ):
            raise _contract_error(
                "V03-11 collection does not match bootstrap workflow lineage",
                code="frozen_selection_frequency_resampling_mismatch",
                field=(
                    "resampling_result_id,crossfit_spec_id,bootstrap_plan_ids,"
                    "workflow_record_ids"
                ),
                remediation="Use the exact V03-11 collection from this resampling",
            )
        spec, records, numeric = _numeric_sources(source, target)
        self = object.__new__(cls)
        values: dict[str, object] = {
            "target_id": target.target_id,
            "declaration_id": target.declaration_id,
            "universe_id": target.universe_id,
            "hypothesis_id": target.hypothesis_id,
            "resampling_result_id": resampling.result_id,
            "crossfit_spec_id": source.crossfit_spec_id,
            "bootstrap_plan_ids": source.bootstrap_plan_ids,
            "workflow_record_ids": source.full_pipeline_record_ids,
            "source_collection_id": source.collection_id,
            "source_event_ids": source.event_ids,
            "selection_rule_id": spec.selection_rule_id,
            "score_version": spec.score_version,
            "opportunity_manifest_id": spec.opportunity_manifest_id,
            "numeric_spec_id": spec.spec_id,
            "numeric_result_id": numeric.result_id,
            "selection_frequency": numeric.selection_frequency,
            "status": numeric.status,
            "reason_code": numeric.reason_code,
            "n_bootstrap_plans_total": numeric.n_bootstrap_plans_total,
            "n_event_rows_total": numeric.n_event_rows_total,
            "n_selected_event_rows": numeric.n_selected_event_rows,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_resampling", resampling)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_numeric_spec", spec)
        object.__setattr__(self, "_numeric_records", records)
        object.__setattr__(self, "_numeric_result", numeric)
        object.__setattr__(self, "_producer_marker", _CHILD_MARKER)
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "frozen_family_selection_frequency",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    @property
    def selection_frequency_release_allowed(self) -> bool:
        return False

    @property
    def source_binding_status(self) -> str:
        return _SOURCE_BINDING_STATUS

    def _identity_payload(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "declaration_id": self.declaration_id,
            "universe_id": self.universe_id,
            "hypothesis_id": self.hypothesis_id,
            "resampling_result_id": self.resampling_result_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "bootstrap_plan_ids": list(self.bootstrap_plan_ids),
            "workflow_record_ids": list(self.workflow_record_ids),
            "source_collection_id": self.source_collection_id,
            "source_event_ids": list(self.source_event_ids),
            "selection_rule_id": self.selection_rule_id,
            "score_version": self.score_version,
            "opportunity_manifest_id": self.opportunity_manifest_id,
            "numeric_spec_id": self.numeric_spec_id,
            "numeric_result_id": self.numeric_result_id,
            "selection_frequency": self.selection_frequency,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_bootstrap_plans_total": self.n_bootstrap_plans_total,
            "n_event_rows_total": self.n_event_rows_total,
            "n_selected_event_rows": self.n_selected_event_rows,
            "source_binding_status": _SOURCE_BINDING_STATUS,
            "selection_frequency_release_allowed": False,
            "producer_marker": _CHILD_MARKER,
        }

    def _require_intact(self) -> None:
        try:
            self._resampling._require_intact()
            self._target._require_intact()
            self._source._require_intact()
            self._numeric_spec._require_intact()
            self._numeric_result._require_intact()
            lineage = _bootstrap_lineage(self._resampling)
            spec, records, numeric = _numeric_sources(self._source, self._target)
            expected = FrozenFamilySelectionFrequency._from_values(
                self._resampling,
                self._target,
                self._source,
                spec,
                records,
                numeric,
            )
            expected_id = stable_id(
                "frozen_family_selection_frequency",
                expected,
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _CHILD_MARKER
                and self._source._resampling is self._resampling
                and self._source.resampling_result_id == self._resampling.result_id
                and self._source.crossfit_spec_id == self._resampling.crossfit_spec_id
                and self._source.bootstrap_plan_ids == lineage.plan_ids
                and self._source.full_pipeline_record_ids == lineage.workflow_record_ids
                and self._identity_payload() == expected
                and self.result_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen family selection frequency failed integrity validation",
                code="frozen_family_selection_frequency_integrity_violation",
                field="result_id",
                remediation="Rebuild from intact V03-11 selection events",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen family selection frequency failed integrity validation",
                code="frozen_family_selection_frequency_integrity_violation",
                field="result_id",
                remediation="Rebuild from intact V03-11 selection events",
            )

    @staticmethod
    def _from_values(
        resampling: FullPipelineResamplingResult,
        target: FrozenFamilyEffectTarget,
        source: FrozenResampledAttributionCollection,
        spec: SelectionFrequencySpec,
        records: tuple[BootstrapSelectionRecord, ...],
        numeric: SelectionFrequencyResult,
    ) -> dict[str, object]:
        return {
            "target_id": target.target_id,
            "declaration_id": target.declaration_id,
            "universe_id": target.universe_id,
            "hypothesis_id": target.hypothesis_id,
            "resampling_result_id": resampling.result_id,
            "crossfit_spec_id": source.crossfit_spec_id,
            "bootstrap_plan_ids": list(source.bootstrap_plan_ids),
            "workflow_record_ids": list(source.full_pipeline_record_ids),
            "source_collection_id": source.collection_id,
            "source_event_ids": list(source.event_ids),
            "selection_rule_id": spec.selection_rule_id,
            "score_version": spec.score_version,
            "opportunity_manifest_id": spec.opportunity_manifest_id,
            "numeric_spec_id": spec.spec_id,
            "numeric_result_id": numeric.result_id,
            "selection_frequency": numeric.selection_frequency,
            "status": numeric.status.value,
            "reason_code": numeric.reason_code,
            "n_bootstrap_plans_total": numeric.n_bootstrap_plans_total,
            "n_event_rows_total": len(records),
            "n_selected_event_rows": numeric.n_selected_event_rows,
            "source_binding_status": _SOURCE_BINDING_STATUS,
            "selection_frequency_release_allowed": False,
            "producer_marker": _CHILD_MARKER,
        }

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"result_id": self.result_id, **payload}


def summarize_frozen_family_selection_frequency(
    resampling: FullPipelineResamplingResult,
    target: FrozenFamilyEffectTarget,
) -> FrozenFamilySelectionFrequency:
    """Authenticate one included target without granting batch release."""

    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    if not isinstance(target, FrozenFamilyEffectTarget):
        raise TypeError("target must be FrozenFamilyEffectTarget")
    resampling._require_intact()
    target._require_intact()
    if (
        target.role is not HypothesisRole.PRIMARY
        or target.endpoint != _PRIMARY_ENDPOINT
    ):
        raise _contract_error(
            "Selection frequency requires a primary driver-family omnibus target",
            code="frozen_selection_frequency_target_mismatch",
            field="role,endpoint",
            remediation="Use an included applicable primary declaration",
        )
    source = summarize_resampled_family_selection(resampling, target)
    if not isinstance(source, FrozenResampledAttributionCollection):
        raise TypeError("V03-11 adapter returned an invalid collection")
    if source._resampling is not resampling:
        raise _contract_error(
            "V03-11 collection does not bind the supplied resampling result",
            code="frozen_selection_frequency_resampling_mismatch",
            field="resampling_result_id",
            remediation="Use the exact source resampling result",
        )
    return FrozenFamilySelectionFrequency._from_sources(resampling, target, source)


def _batch_coverage_id(
    universe: FrozenHypothesisUniverse,
    resampling: FullPipelineResamplingResult,
    applicable: Sequence[HypothesisDeclaration],
    lineage: _BootstrapLineage,
) -> str:
    identifier: str = stable_id(
        "frozen_selection_frequency_batch_coverage",
        {
                "universe_id": universe.universe_id,
                "universe_declaration_ids": [
                    item.declaration_id for item in universe.declarations
                ],
                "applicable_declaration_ids": [
                    item.declaration_id for item in applicable
                ],
                "applicable_hypothesis_ids": [
                    item.hypothesis_id for item in applicable
                ],
                "resampling_result_id": resampling.result_id,
                "crossfit_spec_id": resampling.crossfit_spec_id,
                "bootstrap_plan_ids": list(lineage.plan_ids),
                "workflow_record_ids": list(lineage.workflow_record_ids),
                "batch_semantics": _BATCH_SEMANTICS,
            },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


@dataclass(frozen=True, slots=True, init=False)
class FrozenSelectionFrequencyUniverseRecord:
    """One exact applicable-primary row in the complete universe batch."""

    universe_id: str
    declaration_id: str
    hypothesis_id: str
    endpoint: str
    role: HypothesisRole
    prefilter_status: HypothesisPrefilterStatus
    filter_reason_code: str | None
    resampling_result_id: str
    crossfit_spec_id: str
    batch_coverage_id: str
    target_id: str | None
    child_result_id: str | None
    selection_frequency: float | None
    status: SelectionFrequencyStatus
    reason_code: str | None
    n_bootstrap_plans_total: int
    n_event_rows_total: int
    n_selected_event_rows: int
    record_id: str
    _universe: FrozenHypothesisUniverse
    _declaration: HypothesisDeclaration
    _resampling: FullPipelineResamplingResult
    _child: FrozenFamilySelectionFrequency | None
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError("FrozenSelectionFrequencyUniverseRecord is producer-owned")

    @classmethod
    def _from_parts(
        cls,
        universe: FrozenHypothesisUniverse,
        declaration: HypothesisDeclaration,
        resampling: FullPipelineResamplingResult,
        batch_coverage_id: str,
        *,
        child: FrozenFamilySelectionFrequency | None,
        failure_reason: str | None = None,
    ) -> FrozenSelectionFrequencyUniverseRecord:
        if declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED:
            status = SelectionFrequencyStatus.NOT_ESTIMABLE
            reason = declaration.filter_reason_code
        elif child is not None:
            status = child.status
            reason = child.reason_code
        else:
            status = SelectionFrequencyStatus.FAILED
            reason = _name(failure_reason, field_name="failure_reason")
        self = object.__new__(cls)
        values: dict[str, object] = {
            "universe_id": universe.universe_id,
            "declaration_id": declaration.declaration_id,
            "hypothesis_id": declaration.hypothesis_id,
            "endpoint": declaration.endpoint,
            "role": declaration.role,
            "prefilter_status": declaration.prefilter_status,
            "filter_reason_code": declaration.filter_reason_code,
            "resampling_result_id": resampling.result_id,
            "crossfit_spec_id": resampling.crossfit_spec_id,
            "batch_coverage_id": batch_coverage_id,
            "target_id": None if child is None else child.target_id,
            "child_result_id": None if child is None else child.result_id,
            "selection_frequency": (
                None if child is None else child.selection_frequency
            ),
            "status": status,
            "reason_code": reason,
            "n_bootstrap_plans_total": (
                0 if child is None else child.n_bootstrap_plans_total
            ),
            "n_event_rows_total": 0 if child is None else child.n_event_rows_total,
            "n_selected_event_rows": (
                0 if child is None else child.n_selected_event_rows
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_universe", universe)
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_resampling", resampling)
        object.__setattr__(self, "_child", child)
        object.__setattr__(self, "_producer_marker", _RECORD_MARKER)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "frozen_selection_frequency_universe_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    @property
    def selection_frequency_release_allowed(self) -> bool:
        return (
            self.status is SelectionFrequencyStatus.OBSERVED
            and self._child is not None
            and self.prefilter_status is HypothesisPrefilterStatus.INCLUDED
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "declaration_id": self.declaration_id,
            "hypothesis_id": self.hypothesis_id,
            "endpoint": self.endpoint,
            "role": self.role.value,
            "prefilter_status": self.prefilter_status.value,
            "filter_reason_code": self.filter_reason_code,
            "resampling_result_id": self.resampling_result_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "batch_coverage_id": self.batch_coverage_id,
            "target_id": self.target_id,
            "child_result_id": self.child_result_id,
            "selection_frequency": self.selection_frequency,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_bootstrap_plans_total": self.n_bootstrap_plans_total,
            "n_event_rows_total": self.n_event_rows_total,
            "n_selected_event_rows": self.n_selected_event_rows,
            "selection_frequency_release_allowed": (
                self.selection_frequency_release_allowed
            ),
            "batch_semantics": _BATCH_SEMANTICS,
            "producer_marker": _RECORD_MARKER,
        }

    def _require_intact(self) -> None:
        try:
            self._universe._require_intact()
            self._declaration._require_intact()
            self._resampling._require_intact()
            if self._child is not None:
                self._child._require_intact()
            resolved = self._universe.declaration_for(self.hypothesis_id)
            applicable = _applicable_declarations(self._universe)
            lineage = _bootstrap_lineage(self._resampling)
            expected_batch_id = _batch_coverage_id(
                self._universe,
                self._resampling,
                applicable,
                lineage,
            )
            expected_id = stable_id(
                "frozen_selection_frequency_universe_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            if self.prefilter_status is HypothesisPrefilterStatus.FILTERED:
                status_valid = (
                    self._child is None
                    and self.status is SelectionFrequencyStatus.NOT_ESTIMABLE
                    and self.reason_code == self.filter_reason_code
                    and self.selection_frequency is None
                )
            elif self._child is None:
                status_valid = (
                    self.status is SelectionFrequencyStatus.FAILED
                    and bool(self.reason_code)
                    and self.selection_frequency is None
                )
            else:
                status_valid = (
                    self.status is self._child.status
                    and self.reason_code == self._child.reason_code
                    and self.selection_frequency == self._child.selection_frequency
                    and self.child_result_id == self._child.result_id
                    and self.target_id == self._child.target_id
                )
            valid = (
                self._producer_marker == _RECORD_MARKER
                and resolved is self._declaration
                and any(item is resolved for item in applicable)
                and self.universe_id == self._universe.universe_id
                and self.declaration_id == resolved.declaration_id
                and self.endpoint == resolved.endpoint
                and self.role is HypothesisRole.PRIMARY
                and self.prefilter_status is resolved.prefilter_status
                and self.filter_reason_code == resolved.filter_reason_code
                and self.resampling_result_id == self._resampling.result_id
                and self.crossfit_spec_id == self._resampling.crossfit_spec_id
                and self.batch_coverage_id == expected_batch_id
                and status_valid
                and self.record_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Selection-frequency universe record failed integrity validation",
                code="selection_frequency_universe_record_integrity_violation",
                field="record_id",
                remediation="Rebuild the complete universe batch",
            ) from error
        if not valid:
            raise ContractError(
                "Selection-frequency universe record failed integrity validation",
                code="selection_frequency_universe_record_integrity_violation",
                field="record_id",
                remediation="Rebuild the complete universe batch",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"record_id": self.record_id, **payload}


@dataclass(frozen=True, slots=True, init=False)
class FrozenSelectionFrequencyUniverseCollection:
    """Producer-owned exact batch over every applicable primary declaration."""

    universe_id: str
    universe_declaration_ids: tuple[str, ...]
    applicable_declaration_ids: tuple[str, ...]
    applicable_hypothesis_ids: tuple[str, ...]
    resampling_result_id: str
    crossfit_spec_id: str
    bootstrap_plan_ids: tuple[str, ...]
    workflow_record_ids: tuple[str, ...]
    batch_coverage_id: str
    records: tuple[FrozenSelectionFrequencyUniverseRecord, ...]
    record_ids: tuple[str, ...]
    child_result_ids: tuple[str | None, ...]
    n_records_total: int
    n_observed: int
    n_not_estimable: int
    n_failed: int
    collection_id: str
    _universe: FrozenHypothesisUniverse
    _resampling: FullPipelineResamplingResult
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenSelectionFrequencyUniverseCollection is producer-owned; use "
            "summarize_frozen_selection_frequency_universe()"
        )

    @classmethod
    def _from_records(
        cls,
        universe: FrozenHypothesisUniverse,
        resampling: FullPipelineResamplingResult,
        lineage: _BootstrapLineage,
        records: Sequence[FrozenSelectionFrequencyUniverseRecord],
    ) -> FrozenSelectionFrequencyUniverseCollection:
        applicable = _applicable_declarations(universe)
        ordered = tuple(sorted(records, key=lambda item: item.hypothesis_id))
        expected_ids = tuple(item.hypothesis_id for item in applicable)
        if tuple(item.hypothesis_id for item in ordered) != expected_ids:
            raise _contract_error(
                "Selection-frequency records do not exactly cover applicable primaries",
                code="selection_frequency_universe_coverage_mismatch",
                field="hypothesis_id",
                remediation="Retain one record for every applicable primary",
            )
        batch_id = _batch_coverage_id(universe, resampling, applicable, lineage)
        if any(item.batch_coverage_id != batch_id for item in ordered):
            raise _contract_error(
                "Selection-frequency records bind a different batch coverage ID",
                code="selection_frequency_universe_batch_mismatch",
                field="batch_coverage_id",
                remediation="Rebuild every record in one exact batch",
            )
        counts = Counter(item.status for item in ordered)
        self = object.__new__(cls)
        values: dict[str, object] = {
            "universe_id": universe.universe_id,
            "universe_declaration_ids": tuple(
                item.declaration_id for item in universe.declarations
            ),
            "applicable_declaration_ids": tuple(
                item.declaration_id for item in applicable
            ),
            "applicable_hypothesis_ids": expected_ids,
            "resampling_result_id": resampling.result_id,
            "crossfit_spec_id": resampling.crossfit_spec_id,
            "bootstrap_plan_ids": lineage.plan_ids,
            "workflow_record_ids": lineage.workflow_record_ids,
            "batch_coverage_id": batch_id,
            "records": ordered,
            "record_ids": tuple(item.record_id for item in ordered),
            "child_result_ids": tuple(item.child_result_id for item in ordered),
            "n_records_total": len(ordered),
            "n_observed": counts[SelectionFrequencyStatus.OBSERVED],
            "n_not_estimable": counts[SelectionFrequencyStatus.NOT_ESTIMABLE],
            "n_failed": counts[SelectionFrequencyStatus.FAILED],
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_universe", universe)
        object.__setattr__(self, "_resampling", resampling)
        object.__setattr__(self, "_producer_marker", _COLLECTION_MARKER)
        object.__setattr__(
            self,
            "collection_id",
            stable_id(
                "frozen_selection_frequency_universe_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    @property
    def exact_universe_coverage(self) -> bool:
        return True

    @property
    def performance_status(self) -> str:
        return _PERFORMANCE_STATUS

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "universe_declaration_ids": list(self.universe_declaration_ids),
            "applicable_declaration_ids": list(self.applicable_declaration_ids),
            "applicable_hypothesis_ids": list(self.applicable_hypothesis_ids),
            "resampling_result_id": self.resampling_result_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "bootstrap_plan_ids": list(self.bootstrap_plan_ids),
            "workflow_record_ids": list(self.workflow_record_ids),
            "batch_coverage_id": self.batch_coverage_id,
            "record_ids": list(self.record_ids),
            "child_result_ids": list(self.child_result_ids),
            "n_records_total": self.n_records_total,
            "n_observed": self.n_observed,
            "n_not_estimable": self.n_not_estimable,
            "n_failed": self.n_failed,
            "exact_universe_coverage": True,
            "batch_semantics": _BATCH_SEMANTICS,
            "selection_frequency_release_scope": _RELEASE_SCOPE,
            "excluded_output_kinds": list(_EXCLUDED_OUTPUT_KINDS),
            "performance_status": _PERFORMANCE_STATUS,
            "producer_marker": _COLLECTION_MARKER,
        }

    def _require_intact(self) -> None:
        try:
            self._universe._require_intact()
            self._resampling._require_intact()
            lineage = _bootstrap_lineage(self._resampling)
            applicable = _applicable_declarations(self._universe)
            for record in self.records:
                record._require_intact()
            counts = Counter(item.status for item in self.records)
            expected_batch_id = _batch_coverage_id(
                self._universe,
                self._resampling,
                applicable,
                lineage,
            )
            expected_id = stable_id(
                "frozen_selection_frequency_universe_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _COLLECTION_MARKER
                and self.universe_id == self._universe.universe_id
                and self.universe_declaration_ids
                == tuple(item.declaration_id for item in self._universe.declarations)
                and self.applicable_declaration_ids
                == tuple(item.declaration_id for item in applicable)
                and self.applicable_hypothesis_ids
                == tuple(item.hypothesis_id for item in applicable)
                and tuple(item.hypothesis_id for item in self.records)
                == self.applicable_hypothesis_ids
                and self.resampling_result_id == self._resampling.result_id
                and self.crossfit_spec_id == self._resampling.crossfit_spec_id
                and self.bootstrap_plan_ids == lineage.plan_ids
                and self.workflow_record_ids == lineage.workflow_record_ids
                and self.batch_coverage_id == expected_batch_id
                and self.record_ids == tuple(item.record_id for item in self.records)
                and self.child_result_ids
                == tuple(item.child_result_id for item in self.records)
                and self.n_records_total == len(self.records)
                and self.n_observed == counts[SelectionFrequencyStatus.OBSERVED]
                and self.n_not_estimable
                == counts[SelectionFrequencyStatus.NOT_ESTIMABLE]
                and self.n_failed == counts[SelectionFrequencyStatus.FAILED]
                and self.collection_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Selection-frequency universe collection failed integrity validation",
                code="selection_frequency_universe_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute the exact complete-universe batch",
            ) from error
        if not valid:
            raise ContractError(
                "Selection-frequency universe collection failed integrity validation",
                code="selection_frequency_universe_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute the exact complete-universe batch",
            )

    def result_for(
        self,
        hypothesis_id: str,
    ) -> FrozenSelectionFrequencyUniverseRecord:
        self._require_intact()
        identifier = _name(hypothesis_id, field_name="hypothesis_id")
        matched = tuple(
            item for item in self.records if item.hypothesis_id == identifier
        )
        if len(matched) != 1:
            raise _contract_error(
                "Hypothesis is absent from the selection-frequency universe batch",
                code="selection_frequency_universe_result_not_found",
                field="hypothesis_id",
                remediation="Use an applicable primary hypothesis ID",
            )
        return matched[0]

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "collection_id": self.collection_id,
            **payload,
            "records": [item.to_dict() for item in self.records],
        }


def summarize_frozen_selection_frequency_universe(
    universe: FrozenHypothesisUniverse,
    resampling: FullPipelineResamplingResult,
) -> FrozenSelectionFrequencyUniverseCollection:
    """Summarize every applicable primary while retaining typed prefilters."""

    applicable = _applicable_declarations(universe)
    lineage = _bootstrap_lineage(resampling)
    batch_id = _batch_coverage_id(universe, resampling, applicable, lineage)
    records: list[FrozenSelectionFrequencyUniverseRecord] = []
    for declaration in applicable:
        child: FrozenFamilySelectionFrequency | None = None
        failure_reason: str | None = None
        if declaration.prefilter_status is HypothesisPrefilterStatus.INCLUDED:
            try:
                target = build_frozen_family_effect_target(
                    universe,
                    declaration.hypothesis_id,
                )
                child = summarize_frozen_family_selection_frequency(
                    resampling,
                    target,
                )
            except Exception as error:
                failure_reason = _failure_reason(error)
        records.append(
            FrozenSelectionFrequencyUniverseRecord._from_parts(
                universe,
                declaration,
                resampling,
                batch_id,
                child=child,
                failure_reason=failure_reason,
            )
        )
    return FrozenSelectionFrequencyUniverseCollection._from_records(
        universe,
        resampling,
        lineage,
        records,
    )


__all__ = [
    "FrozenFamilySelectionFrequency",
    "FrozenSelectionFrequencyUniverseCollection",
    "FrozenSelectionFrequencyUniverseRecord",
    "summarize_frozen_family_selection_frequency",
    "summarize_frozen_selection_frequency_universe",
]
