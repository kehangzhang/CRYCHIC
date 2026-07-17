"""Complete frozen-universe batches of authenticated specificity support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from crychic.core import CommunicationMode, ContractError, CrychicError, stable_id
from crychic.inference.bootstrap_support import SpecificitySupportStatus
from crychic.inference.full_pipeline import FullPipelineEffectDistributionSpec
from crychic.inference.hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisDeclaration,
    HypothesisPrefilterStatus,
    HypothesisRole,
)
from crychic.resampling import SubjectBootstrapPlan

from .crossfit import CrossFitArtifacts
from .frozen_specificity_support import (
    FrozenFamilySpecificitySupport,
    _require_point_resampling_source_alignment,
    summarize_frozen_family_specificity_support,
)
from .full_pipeline_effects import (
    FrozenFamilyEffectTarget,
    build_frozen_family_effect_target,
)
from .full_pipeline_resampling import (
    FullPipelineResampleStatus,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
    aligned_full_pipeline_resamples,
)

_SCHEMA_VERSION = "1.0.0"
_PRIMARY_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_SECONDARY_ENDPOINT = "family_common_integrated_lr_context_effect_v1"
_RECORD_MARKER = "crychic.workflow.frozen_specificity_universe_record.v1"
_COLLECTION_MARKER = "crychic.workflow.frozen_specificity_universe.v1"
_COVERAGE_SEMANTICS = "all_frozen_secondary_specificity_declarations_v1"


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


def _failure_reason(error: Exception) -> str:
    if isinstance(error, CrychicError):
        reason: str = error.details.code
        return reason
    return f"specificity_universe_adapter_failed_{type(error).__qualname__}"


@dataclass(frozen=True, slots=True)
class _BootstrapWorkflowLineage:
    manifest_id: str
    plan_ids: tuple[str, ...]
    workflow_record_ids: tuple[str, ...]


def _bootstrap_workflow_lineage(
    resampling: FullPipelineResamplingResult,
) -> _BootstrapWorkflowLineage:
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    pairs = aligned_full_pipeline_resamples(
        resampling,
        operation=FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP,
    )
    if not pairs:
        raise _contract_error(
            "Specificity universe batch requires subject-bootstrap plans",
            code="specificity_universe_bootstrap_plans_missing",
            field="plans",
            remediation="Run full-pipeline subject bootstraps with retained children",
        )
    if not resampling.retain_children:
        raise _contract_error(
            "Specificity universe batch requires retained bootstrap children",
            code="specificity_universe_children_not_retained",
            field="retain_children",
            remediation="Rerun full-pipeline resampling with retain_children=True",
        )
    selected = []
    for plan, record in pairs:
        if not isinstance(plan, SubjectBootstrapPlan):
            raise _contract_error(
                "Bootstrap operation resolved to an unsupported plan type",
                code="specificity_universe_bootstrap_plan_type_mismatch",
                field="plan",
                remediation="Use the intact full-pipeline resampling result",
            )
        if record.status is FullPipelineResampleStatus.SUCCEEDED:
            child = record.child
            if child is None:
                raise _contract_error(
                    "Successful bootstrap is missing its retained child",
                    code="specificity_universe_bootstrap_child_missing",
                    field="record_id",
                    remediation="Rerun resampling with retained children",
                )
            child._require_intact()
            if (
                child.crossfit_id != record.crossfit_id
                or child.spec.spec_id != resampling.crossfit_spec_id
            ):
                raise _contract_error(
                    "Bootstrap child does not match its workflow record",
                    code="specificity_universe_bootstrap_child_mismatch",
                    field="crossfit_id,crossfit_spec_id",
                    remediation="Reject the corrupted resampling result",
                )
        selected.append((plan.resample_index, plan.bootstrap_id, record.record_id))
    ordered = tuple(sorted(selected))
    plan_ids = tuple(item[1] for item in ordered)
    record_ids = tuple(item[2] for item in ordered)
    manifest_id = stable_id(
        "specificity_universe_bootstrap_manifest",
        {
                "exchangeability_id": resampling.exchangeability.exchangeability_id,
                "source_input_identity_id": resampling.source_input_identity_id,
                "source_input_digest": resampling.source_input_digest,
                "source_snapshot_id": resampling.source_snapshot_id,
                "config_digest": resampling.config_digest,
                "crossfit_spec_id": resampling.crossfit_spec_id,
                "resource_bundle_content_id": resampling.resource_bundle_content_id,
                "target_prior_content_id": resampling.target_prior_content_id,
                "root_seed_lineage": resampling.root_seed_lineage.to_dict(),
                "bootstrap_plan_ids": list(plan_ids),
                "workflow_record_ids": list(record_ids),
                "coverage_semantics": _COVERAGE_SEMANTICS,
            },
        schema_version=_SCHEMA_VERSION,
    )
    return _BootstrapWorkflowLineage(
        manifest_id=manifest_id,
        plan_ids=plan_ids,
        workflow_record_ids=record_ids,
    )


def _secondary_declarations(
    universe: FrozenHypothesisUniverse,
) -> tuple[HypothesisDeclaration, ...]:
    universe._require_intact()
    return tuple(
        declaration
        for declaration in universe.declarations
        if declaration.role is HypothesisRole.SECONDARY
        and declaration.endpoint == _SECONDARY_ENDPOINT
        and declaration.mode is CommunicationMode.STATE
    )


def _parent_for(
    universe: FrozenHypothesisUniverse,
    declaration: HypothesisDeclaration,
) -> HypothesisDeclaration:
    if declaration.parent_key is None:
        raise _contract_error(
            "Specificity secondary is missing its primary parent key",
            code="specificity_universe_parent_missing",
            field="parent_key",
            remediation="Refreeze the secondary under a primary omnibus",
        )
    matched = tuple(
        item
        for item in universe.declarations
        if item.hypothesis_key == declaration.parent_key
    )
    if len(matched) != 1:
        raise _contract_error(
            "Specificity secondary parent is absent or ambiguous",
            code="specificity_universe_parent_missing",
            field="parent_key",
            remediation="Refreeze one exact primary omnibus parent",
        )
    parent = matched[0]
    if (
        parent.role is not HypothesisRole.PRIMARY
        or parent.endpoint != _PRIMARY_ENDPOINT
        or (parent.receiver, parent.family_id, parent.mode)
        != (declaration.receiver, declaration.family_id, declaration.mode)
    ):
        raise _contract_error(
            "Specificity secondary has the wrong frozen primary parent",
            code="specificity_universe_parent_mismatch",
            field="endpoint,role,receiver,family_id,mode",
            remediation="Use the matching primary family omnibus declaration",
        )
    return parent


def _validated_specs(
    declarations: tuple[HypothesisDeclaration, ...],
    supplied: (
        Sequence[FullPipelineEffectDistributionSpec]
        | Mapping[str, FullPipelineEffectDistributionSpec]
    ),
) -> tuple[FullPipelineEffectDistributionSpec, ...]:
    expected = tuple(
        item
        for item in declarations
        if item.prefilter_status is HypothesisPrefilterStatus.INCLUDED
    )
    expected_ids = {item.hypothesis_id for item in expected}
    entries: list[tuple[str, FullPipelineEffectDistributionSpec]] = []
    if isinstance(supplied, Mapping):
        for hypothesis_id, spec in supplied.items():
            if not isinstance(hypothesis_id, str) or not hypothesis_id:
                raise ValueError(
                    "distribution spec mapping keys must be hypothesis IDs"
                )
            entries.append((hypothesis_id, spec))
    elif isinstance(supplied, Sequence):
        entries.extend(
            (spec.effect_spec.hypothesis_id, spec)
            for spec in supplied
            if isinstance(spec, FullPipelineEffectDistributionSpec)
        )
        if len(entries) != len(supplied):
            raise TypeError("distribution specs must be typed distribution specs")
    else:
        raise TypeError("distribution_specs must be a sequence or mapping")
    if any(
        not isinstance(spec, FullPipelineEffectDistributionSpec) for _, spec in entries
    ):
        raise TypeError("distribution specs must be typed distribution specs")
    for hypothesis_id, spec in entries:
        spec._require_intact()
        if hypothesis_id != spec.effect_spec.hypothesis_id:
            raise _contract_error(
                "Distribution spec mapping key differs from its hypothesis",
                code="specificity_universe_spec_key_mismatch",
                field="hypothesis_id",
                remediation="Key every spec by its exact effect hypothesis ID",
            )
    observed_ids = tuple(hypothesis_id for hypothesis_id, _ in entries)
    if len(set(observed_ids)) != len(observed_ids):
        raise _contract_error(
            "Specificity universe contains duplicate distribution specs",
            code="specificity_universe_duplicate_spec",
            field="hypothesis_id",
            remediation="Supply every included secondary spec exactly once",
        )
    missing = expected_ids.difference(observed_ids)
    extra = set(observed_ids).difference(expected_ids)
    if missing:
        raise _contract_error(
            "Included specificity secondaries are missing distribution specs",
            code="specificity_universe_missing_spec",
            field="hypothesis_id",
            remediation="Supply one preregistered spec for every included secondary",
        )
    if extra:
        raise _contract_error(
            "Distribution specs include non-applicable frozen hypotheses",
            code="specificity_universe_extra_spec",
            field="hypothesis_id",
            remediation="Remove specs for filtered or non-secondary declarations",
        )
    by_id = {hypothesis_id: spec for hypothesis_id, spec in entries}
    values = tuple(by_id[item.hypothesis_id] for item in expected)
    for declaration, spec in zip(expected, values, strict=True):
        if (
            spec.effect_spec.hypothesis_id != declaration.hypothesis_id
            or spec.effect_spec.contrast_name != declaration.contrast_name
        ):
            raise _contract_error(
                "Distribution spec does not match its exact frozen declaration",
                code="specificity_universe_spec_declaration_mismatch",
                field="hypothesis_id,contrast_name",
                remediation="Build the spec for the exact frozen secondary",
            )
    return values


@dataclass(frozen=True, slots=True, init=False)
class FrozenSpecificitySupportUniverseRecord:
    """One exact secondary declaration in a complete specificity batch."""

    universe_id: str
    declaration_id: str
    hypothesis_id: str
    parent_declaration_id: str
    parent_hypothesis_id: str
    target_id: str | None
    score_target_id: str | None
    distribution_spec_id: str | None
    child_result_id: str | None
    point_crossfit_id: str
    crossfit_spec_id: str
    bootstrap_workflow_manifest_id: str
    bootstrap_plan_ids: tuple[str, ...]
    bootstrap_workflow_record_ids: tuple[str, ...]
    status: SpecificitySupportStatus
    reason_code: str | None
    specificity_support: float | None
    n_bootstrap_total: int
    n_bootstrap_observed: int
    n_bootstrap_not_estimable: int
    n_bootstrap_failed: int
    record_id: str
    _point_artifacts: CrossFitArtifacts
    _resampling: FullPipelineResamplingResult
    _universe: FrozenHypothesisUniverse
    _declaration: HypothesisDeclaration
    _parent_declaration: HypothesisDeclaration
    _target: FrozenFamilyEffectTarget | None
    _score_target: FrozenFamilyEffectTarget | None
    _distribution_spec: FullPipelineEffectDistributionSpec | None
    _child_result: FrozenFamilySpecificitySupport | None
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenSpecificitySupportUniverseRecord is producer-owned; use "
            "summarize_frozen_specificity_support_universe()"
        )

    @property
    def source_authenticated(self) -> bool:
        return True

    @property
    def specificity_support_release_allowed(self) -> bool:
        return bool(
            self._child_result is not None
            and self._child_result.specificity_support_release_allowed
        )

    @property
    def formal_pq_inference_allowed(self) -> bool:
        return False

    @property
    def is_posterior_probability(self) -> bool:
        return False

    @property
    def is_comm_probability(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "declaration_id": self.declaration_id,
            "hypothesis_id": self.hypothesis_id,
            "parent_declaration_id": self.parent_declaration_id,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "target_id": self.target_id,
            "score_target_id": self.score_target_id,
            "distribution_spec_id": self.distribution_spec_id,
            "child_result_id": self.child_result_id,
            "point_crossfit_id": self.point_crossfit_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "bootstrap_workflow_manifest_id": self.bootstrap_workflow_manifest_id,
            "bootstrap_plan_ids": list(self.bootstrap_plan_ids),
            "bootstrap_workflow_record_ids": list(self.bootstrap_workflow_record_ids),
            "status": self.status.value,
            "reason_code": self.reason_code,
            "specificity_support": self.specificity_support,
            "n_bootstrap_total": self.n_bootstrap_total,
            "n_bootstrap_observed": self.n_bootstrap_observed,
            "n_bootstrap_not_estimable": self.n_bootstrap_not_estimable,
            "n_bootstrap_failed": self.n_bootstrap_failed,
            "source_authenticated": True,
            "specificity_support_release_allowed": (
                self.specificity_support_release_allowed
            ),
            "formal_pq_inference_allowed": False,
            "is_posterior_probability": False,
            "is_comm_probability": False,
            "coverage_semantics": _COVERAGE_SEMANTICS,
            "producer_marker": _RECORD_MARKER,
        }

    def _require_intact(self) -> None:
        try:
            self._point_artifacts._require_intact()
            self._point_artifacts.spec._require_intact()
            self._resampling._require_intact()
            _require_point_resampling_source_alignment(
                self._point_artifacts,
                self._resampling,
            )
            self._universe._require_intact()
            self._declaration._require_intact()
            self._parent_declaration._require_intact()
            resolved = self._universe.declaration_for(self.hypothesis_id)
            parent = _parent_for(self._universe, resolved)
            lineage = _bootstrap_workflow_lineage(self._resampling)
            filtered = resolved.prefilter_status is HypothesisPrefilterStatus.FILTERED
            if filtered:
                sources_valid = (
                    self._target is None
                    and self._score_target is None
                    and self._distribution_spec is None
                    and self._child_result is None
                    and self.target_id is None
                    and self.score_target_id is None
                    and self.distribution_spec_id is None
                    and self.child_result_id is None
                    and self.status is SpecificitySupportStatus.NOT_ESTIMABLE
                    and self.reason_code == resolved.filter_reason_code
                    and self.specificity_support is None
                    and (
                        self.n_bootstrap_total,
                        self.n_bootstrap_observed,
                        self.n_bootstrap_not_estimable,
                        self.n_bootstrap_failed,
                    )
                    == (0, 0, 0, 0)
                )
            else:
                sources_valid = False
                if all(
                    value is not None
                    for value in (
                        self._target,
                        self._score_target,
                        self._distribution_spec,
                    )
                ):
                    target = cast(FrozenFamilyEffectTarget, self._target)
                    score_target = cast(FrozenFamilyEffectTarget, self._score_target)
                    spec = cast(
                        FullPipelineEffectDistributionSpec,
                        self._distribution_spec,
                    )
                    target._require_intact()
                    score_target._require_intact()
                    spec._require_intact()
                    base_sources_valid = (
                        target._universe is self._universe
                        and score_target._universe is self._universe
                        and target.hypothesis_id == resolved.hypothesis_id
                        and score_target.hypothesis_id == parent.hypothesis_id
                        and spec.effect_spec.hypothesis_id == resolved.hypothesis_id
                        and spec.effect_spec.contrast_name == resolved.contrast_name
                        and self.target_id == target.target_id
                        and self.score_target_id == score_target.target_id
                        and self.distribution_spec_id == spec.spec_id
                    )
                    if self._child_result is None:
                        sources_valid = (
                            base_sources_valid
                            and self.child_result_id is None
                            and self.status is SpecificitySupportStatus.FAILED
                            and bool(self.reason_code)
                            and self.specificity_support is None
                            and (
                                self.n_bootstrap_total,
                                self.n_bootstrap_observed,
                                self.n_bootstrap_not_estimable,
                                self.n_bootstrap_failed,
                            )
                            == (0, 0, 0, 0)
                        )
                    else:
                        child = self._child_result
                        child._require_intact()
                        sources_valid = (
                            base_sources_valid
                            and child._point_artifacts is self._point_artifacts
                            and child._resampling is self._resampling
                            and child._target is target
                            and child._score_target is score_target
                            and child._distribution_spec is spec
                            and self.target_id == child.target_id
                            and self.score_target_id == child.score_target_id
                            and self.distribution_spec_id == child.distribution_spec_id
                            and self.child_result_id == child.result_id
                            and self.status is child.status
                            and self.reason_code == child.reason_code
                            and self.specificity_support == child.specificity_support
                            and self.n_bootstrap_total == child.n_bootstrap_total
                            and self.n_bootstrap_observed == child.n_bootstrap_observed
                            and self.n_bootstrap_not_estimable
                            == child.n_bootstrap_not_estimable
                            and self.n_bootstrap_failed == child.n_bootstrap_failed
                            and self.bootstrap_plan_ids == child.bootstrap_plan_ids
                            and self.bootstrap_workflow_record_ids
                            == child.workflow_record_ids
                        )
            expected_id = stable_id(
                "frozen_specificity_universe_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _RECORD_MARKER
                and resolved is self._declaration
                and parent is self._parent_declaration
                and self.universe_id == self._universe.universe_id
                and self.declaration_id == resolved.declaration_id
                and self.parent_declaration_id == parent.declaration_id
                and self.parent_hypothesis_id == parent.hypothesis_id
                and self.point_crossfit_id == self._point_artifacts.crossfit_id
                and self.crossfit_spec_id == self._point_artifacts.spec.spec_id
                and self.crossfit_spec_id == self._resampling.crossfit_spec_id
                and self.bootstrap_workflow_manifest_id == lineage.manifest_id
                and self.bootstrap_plan_ids == lineage.plan_ids
                and self.bootstrap_workflow_record_ids == lineage.workflow_record_ids
                and sources_valid
                and self.record_id == expected_id
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Frozen specificity universe record failed integrity validation",
                code="frozen_specificity_universe_record_integrity_violation",
                field="record_id",
                remediation="Rebuild the complete specificity universe batch",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen specificity universe record failed integrity validation",
                code="frozen_specificity_universe_record_integrity_violation",
                field="record_id",
                remediation="Rebuild the complete specificity universe batch",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"record_id": self.record_id, **payload}


def _new_record(
    *,
    point_artifacts: CrossFitArtifacts,
    resampling: FullPipelineResamplingResult,
    universe: FrozenHypothesisUniverse,
    declaration: HypothesisDeclaration,
    parent: HypothesisDeclaration,
    lineage: _BootstrapWorkflowLineage,
    target: FrozenFamilyEffectTarget | None,
    score_target: FrozenFamilyEffectTarget | None,
    distribution_spec: FullPipelineEffectDistributionSpec | None,
    child: FrozenFamilySpecificitySupport | None,
    failure_reason: str | None = None,
) -> FrozenSpecificitySupportUniverseRecord:
    filtered = declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED
    sources = (target, score_target, distribution_spec)
    if filtered and (
        child is not None
        or failure_reason is not None
        or any(value is not None for value in sources)
    ):
        raise _contract_error(
            "Filtered specificity rows cannot retain fitted sources",
            code="specificity_universe_prefilter_child_mismatch",
            field="prefilter_status",
            remediation="Keep filtered rows as source-free typed not-estimable",
        )
    if not filtered:
        if any(value is None for value in sources):
            raise _contract_error(
                "Included specificity row is missing frozen source identity",
                code="specificity_universe_included_source_missing",
                field="target,score_target,distribution_spec",
                remediation="Retain declaration, parent, target, and spec identity",
            )
        if child is None:
            if (
                not isinstance(failure_reason, str)
                or not failure_reason
                or failure_reason != failure_reason.strip()
            ):
                raise ValueError(
                    "included child failure requires a canonical reason code"
                )
        elif failure_reason is not None:
            raise ValueError("successful specificity child cannot have failure_reason")
    self = object.__new__(FrozenSpecificitySupportUniverseRecord)
    values: dict[str, object] = {
        "universe_id": universe.universe_id,
        "declaration_id": declaration.declaration_id,
        "hypothesis_id": declaration.hypothesis_id,
        "parent_declaration_id": parent.declaration_id,
        "parent_hypothesis_id": parent.hypothesis_id,
        "target_id": None if target is None else target.target_id,
        "score_target_id": None if score_target is None else score_target.target_id,
        "distribution_spec_id": (
            None if distribution_spec is None else distribution_spec.spec_id
        ),
        "child_result_id": None if child is None else child.result_id,
        "point_crossfit_id": point_artifacts.crossfit_id,
        "crossfit_spec_id": point_artifacts.spec.spec_id,
        "bootstrap_workflow_manifest_id": lineage.manifest_id,
        "bootstrap_plan_ids": lineage.plan_ids,
        "bootstrap_workflow_record_ids": lineage.workflow_record_ids,
        "status": (
            SpecificitySupportStatus.NOT_ESTIMABLE
            if filtered
            else (SpecificitySupportStatus.FAILED if child is None else child.status)
        ),
        "reason_code": (
            declaration.filter_reason_code
            if filtered
            else (failure_reason if child is None else child.reason_code)
        ),
        "specificity_support": (None if child is None else child.specificity_support),
        "n_bootstrap_total": 0 if child is None else child.n_bootstrap_total,
        "n_bootstrap_observed": 0 if child is None else child.n_bootstrap_observed,
        "n_bootstrap_not_estimable": (
            0 if child is None else child.n_bootstrap_not_estimable
        ),
        "n_bootstrap_failed": 0 if child is None else child.n_bootstrap_failed,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(self, "_point_artifacts", point_artifacts)
    object.__setattr__(self, "_resampling", resampling)
    object.__setattr__(self, "_universe", universe)
    object.__setattr__(self, "_declaration", declaration)
    object.__setattr__(self, "_parent_declaration", parent)
    object.__setattr__(self, "_target", target)
    object.__setattr__(self, "_score_target", score_target)
    object.__setattr__(self, "_distribution_spec", distribution_spec)
    object.__setattr__(self, "_child_result", child)
    object.__setattr__(self, "_producer_marker", _RECORD_MARKER)
    object.__setattr__(
        self,
        "record_id",
        stable_id(
            "frozen_specificity_universe_record",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    self._require_intact()
    return self


@dataclass(frozen=True, slots=True, init=False)
class FrozenSpecificitySupportUniverseCollection:
    """Producer-owned complete coverage of applicable frozen secondaries."""

    universe_id: str
    universe_declaration_ids: tuple[str, ...]
    applicable_secondary_hypothesis_ids: tuple[str, ...]
    included_secondary_hypothesis_ids: tuple[str, ...]
    filtered_secondary_hypothesis_ids: tuple[str, ...]
    point_crossfit_id: str
    crossfit_spec_id: str
    bootstrap_workflow_manifest_id: str
    bootstrap_plan_ids: tuple[str, ...]
    bootstrap_workflow_record_ids: tuple[str, ...]
    distribution_spec_ids: tuple[str, ...]
    child_result_ids: tuple[str | None, ...]
    record_ids: tuple[str, ...]
    records: tuple[FrozenSpecificitySupportUniverseRecord, ...]
    n_applicable: int
    n_included: int
    n_filtered: int
    n_observed: int
    n_not_estimable: int
    n_failed: int
    collection_id: str
    _point_artifacts: CrossFitArtifacts
    _resampling: FullPipelineResamplingResult
    _universe: FrozenHypothesisUniverse
    _distribution_specs: tuple[FullPipelineEffectDistributionSpec, ...]
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenSpecificitySupportUniverseCollection is producer-owned; use "
            "summarize_frozen_specificity_support_universe()"
        )

    @property
    def complete_secondary_coverage(self) -> bool:
        return True

    @property
    def source_authenticated(self) -> bool:
        return True

    @property
    def formal_pq_inference_allowed(self) -> bool:
        return False

    @property
    def is_posterior_probability(self) -> bool:
        return False

    @property
    def is_comm_probability(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "universe_declaration_ids": list(self.universe_declaration_ids),
            "applicable_secondary_hypothesis_ids": list(
                self.applicable_secondary_hypothesis_ids
            ),
            "included_secondary_hypothesis_ids": list(
                self.included_secondary_hypothesis_ids
            ),
            "filtered_secondary_hypothesis_ids": list(
                self.filtered_secondary_hypothesis_ids
            ),
            "point_crossfit_id": self.point_crossfit_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "bootstrap_workflow_manifest_id": self.bootstrap_workflow_manifest_id,
            "bootstrap_plan_ids": list(self.bootstrap_plan_ids),
            "bootstrap_workflow_record_ids": list(self.bootstrap_workflow_record_ids),
            "distribution_spec_ids": list(self.distribution_spec_ids),
            "child_result_ids": list(self.child_result_ids),
            "record_ids": list(self.record_ids),
            "n_applicable": self.n_applicable,
            "n_included": self.n_included,
            "n_filtered": self.n_filtered,
            "n_observed": self.n_observed,
            "n_not_estimable": self.n_not_estimable,
            "n_failed": self.n_failed,
            "complete_secondary_coverage": True,
            "source_authenticated": True,
            "formal_pq_inference_allowed": False,
            "is_posterior_probability": False,
            "is_comm_probability": False,
            "coverage_semantics": _COVERAGE_SEMANTICS,
            "producer_marker": _COLLECTION_MARKER,
        }

    def _require_intact(self) -> None:
        try:
            self._point_artifacts._require_intact()
            self._point_artifacts.spec._require_intact()
            self._resampling._require_intact()
            self._universe._require_intact()
            for spec in self._distribution_specs:
                spec._require_intact()
            for record in self.records:
                record._require_intact()
            declarations = _secondary_declarations(self._universe)
            included = tuple(
                item
                for item in declarations
                if item.prefilter_status is HypothesisPrefilterStatus.INCLUDED
            )
            filtered = tuple(
                item
                for item in declarations
                if item.prefilter_status is HypothesisPrefilterStatus.FILTERED
            )
            specs = _validated_specs(declarations, self._distribution_specs)
            lineage = _bootstrap_workflow_lineage(self._resampling)
            expected_ids = tuple(item.hypothesis_id for item in declarations)
            observed_ids = tuple(item.hypothesis_id for item in self.records)
            by_id = {item.hypothesis_id: item for item in self.records}
            expected_record_ids = tuple(by_id[item].record_id for item in expected_ids)
            child_ids = tuple(
                by_id[item.hypothesis_id].child_result_id for item in included
            )
            counts = {
                status: sum(record.status is status for record in self.records)
                for status in SpecificitySupportStatus
            }
            expected_id = stable_id(
                "frozen_specificity_universe_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _COLLECTION_MARKER
                and self.universe_id == self._universe.universe_id
                and self.universe_declaration_ids
                == tuple(item.declaration_id for item in self._universe.declarations)
                and self.applicable_secondary_hypothesis_ids == expected_ids
                and self.included_secondary_hypothesis_ids
                == tuple(item.hypothesis_id for item in included)
                and self.filtered_secondary_hypothesis_ids
                == tuple(item.hypothesis_id for item in filtered)
                and self.point_crossfit_id == self._point_artifacts.crossfit_id
                and self.crossfit_spec_id == self._point_artifacts.spec.spec_id
                and self.crossfit_spec_id == self._resampling.crossfit_spec_id
                and self.bootstrap_workflow_manifest_id == lineage.manifest_id
                and self.bootstrap_plan_ids == lineage.plan_ids
                and self.bootstrap_workflow_record_ids == lineage.workflow_record_ids
                and self.distribution_spec_ids == tuple(item.spec_id for item in specs)
                and self.child_result_ids == child_ids
                and observed_ids == expected_ids
                and self.record_ids == expected_record_ids
                and self.n_applicable == len(declarations)
                and self.n_included == len(included)
                and self.n_filtered == len(filtered)
                and self.n_observed == counts[SpecificitySupportStatus.OBSERVED]
                and self.n_not_estimable
                == counts[SpecificitySupportStatus.NOT_ESTIMABLE]
                and self.n_failed == counts[SpecificitySupportStatus.FAILED]
                and all(
                    record.specificity_support_release_allowed
                    == (record.status is SpecificitySupportStatus.OBSERVED)
                    for record in self.records
                )
                and self.collection_id == expected_id
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Frozen specificity universe collection failed integrity validation",
                code="frozen_specificity_universe_collection_integrity_violation",
                field="collection_id",
                remediation="Rebuild from the exact universe and workflow parents",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen specificity universe collection failed integrity validation",
                code="frozen_specificity_universe_collection_integrity_violation",
                field="collection_id",
                remediation="Rebuild from the exact universe and workflow parents",
            )

    def result_for(
        self,
        hypothesis_id: str,
    ) -> FrozenSpecificitySupportUniverseRecord:
        self._require_intact()
        matched = tuple(
            item for item in self.records if item.hypothesis_id == hypothesis_id
        )
        if len(matched) != 1:
            raise _contract_error(
                "Hypothesis is absent from the specificity universe batch",
                code="specificity_universe_result_not_found",
                field="hypothesis_id",
                remediation="Use an applicable frozen secondary hypothesis ID",
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


def summarize_frozen_specificity_support_universe(
    point_artifacts: CrossFitArtifacts,
    resampling: FullPipelineResamplingResult,
    universe: FrozenHypothesisUniverse,
    distribution_specs: (
        Sequence[FullPipelineEffectDistributionSpec]
        | Mapping[str, FullPipelineEffectDistributionSpec]
    ),
) -> FrozenSpecificitySupportUniverseCollection:
    """Summarize every applicable secondary in one exact frozen universe."""

    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    if not isinstance(universe, FrozenHypothesisUniverse):
        raise TypeError("universe must be FrozenHypothesisUniverse")
    point_artifacts._require_intact()
    point_artifacts.spec._require_intact()
    resampling._require_intact()
    universe._require_intact()
    if point_artifacts.spec.spec_id != resampling.crossfit_spec_id:
        raise _contract_error(
            "Point and bootstrap workflows use different cross-fit specs",
            code="specificity_universe_crossfit_spec_mismatch",
            field="crossfit_spec_id",
            remediation="Use one exact CrossFitSpec for point and bootstrap runs",
        )
    _require_point_resampling_source_alignment(point_artifacts, resampling)
    declarations = _secondary_declarations(universe)
    if not declarations:
        raise _contract_error(
            "Frozen universe contains no applicable specificity secondaries",
            code="specificity_universe_no_applicable_secondaries",
            field="declarations",
            remediation="Declare secondary family context-effect hypotheses",
        )
    parents = {item.hypothesis_id: _parent_for(universe, item) for item in declarations}
    specs = _validated_specs(declarations, distribution_specs)
    spec_by_id = {item.effect_spec.hypothesis_id: item for item in specs}
    lineage = _bootstrap_workflow_lineage(resampling)
    records: list[FrozenSpecificitySupportUniverseRecord] = []
    for declaration in declarations:
        parent = parents[declaration.hypothesis_id]
        if declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED:
            record = _new_record(
                point_artifacts=point_artifacts,
                resampling=resampling,
                universe=universe,
                declaration=declaration,
                parent=parent,
                lineage=lineage,
                target=None,
                score_target=None,
                distribution_spec=None,
                child=None,
            )
        else:
            target = build_frozen_family_effect_target(
                universe,
                declaration.hypothesis_id,
            )
            score_target = build_frozen_family_effect_target(
                universe,
                parent.hypothesis_id,
            )
            spec = spec_by_id[declaration.hypothesis_id]
            failure_reason: str | None = None
            try:
                child = summarize_frozen_family_specificity_support(
                    point_artifacts,
                    resampling,
                    target,
                    spec,
                    score_target=score_target,
                )
            except Exception as error:
                child = None
                failure_reason = _failure_reason(error)
            record = _new_record(
                point_artifacts=point_artifacts,
                resampling=resampling,
                universe=universe,
                declaration=declaration,
                parent=parent,
                lineage=lineage,
                target=target,
                score_target=score_target,
                distribution_spec=spec,
                child=child,
                failure_reason=failure_reason,
            )
        records.append(record)
    ordered_records = tuple(sorted(records, key=lambda item: item.hypothesis_id))
    included = tuple(
        item
        for item in declarations
        if item.prefilter_status is HypothesisPrefilterStatus.INCLUDED
    )
    filtered = tuple(
        item
        for item in declarations
        if item.prefilter_status is HypothesisPrefilterStatus.FILTERED
    )
    by_id = {item.hypothesis_id: item for item in ordered_records}
    child_ids = tuple(by_id[item.hypothesis_id].child_result_id for item in included)
    counts = {
        status: sum(item.status is status for item in ordered_records)
        for status in SpecificitySupportStatus
    }
    self = object.__new__(FrozenSpecificitySupportUniverseCollection)
    values: dict[str, object] = {
        "universe_id": universe.universe_id,
        "universe_declaration_ids": tuple(
            item.declaration_id for item in universe.declarations
        ),
        "applicable_secondary_hypothesis_ids": tuple(
            item.hypothesis_id for item in declarations
        ),
        "included_secondary_hypothesis_ids": tuple(
            item.hypothesis_id for item in included
        ),
        "filtered_secondary_hypothesis_ids": tuple(
            item.hypothesis_id for item in filtered
        ),
        "point_crossfit_id": point_artifacts.crossfit_id,
        "crossfit_spec_id": point_artifacts.spec.spec_id,
        "bootstrap_workflow_manifest_id": lineage.manifest_id,
        "bootstrap_plan_ids": lineage.plan_ids,
        "bootstrap_workflow_record_ids": lineage.workflow_record_ids,
        "distribution_spec_ids": tuple(item.spec_id for item in specs),
        "child_result_ids": child_ids,
        "record_ids": tuple(item.record_id for item in ordered_records),
        "records": ordered_records,
        "n_applicable": len(declarations),
        "n_included": len(included),
        "n_filtered": len(filtered),
        "n_observed": counts[SpecificitySupportStatus.OBSERVED],
        "n_not_estimable": counts[SpecificitySupportStatus.NOT_ESTIMABLE],
        "n_failed": counts[SpecificitySupportStatus.FAILED],
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(self, "_point_artifacts", point_artifacts)
    object.__setattr__(self, "_resampling", resampling)
    object.__setattr__(self, "_universe", universe)
    object.__setattr__(self, "_distribution_specs", specs)
    object.__setattr__(self, "_producer_marker", _COLLECTION_MARKER)
    object.__setattr__(
        self,
        "collection_id",
        stable_id(
            "frozen_specificity_universe_collection",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    self._require_intact()
    return self


__all__ = [
    "FrozenSpecificitySupportUniverseCollection",
    "FrozenSpecificitySupportUniverseRecord",
    "summarize_frozen_specificity_support_universe",
]
