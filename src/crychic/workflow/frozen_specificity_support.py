"""Frozen-workflow authentication for family specificity support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from crychic.core import ContractError, stable_id
from crychic.inference.bootstrap_support import (
    SpecificitySupportResult,
    SpecificitySupportSpec,
    SpecificitySupportStatus,
    summarize_specificity_support,
)
from crychic.inference.effects import OOFContextEffectResult
from crychic.inference.full_pipeline import (
    FullPipelineEffectDistribution,
    FullPipelineEffectDistributionSpec,
    FullPipelineEffectResamplingKind,
    FullPipelineResampleEffectRecord,
    FullPipelineResampleEffectStatus,
    SpecificityDirection,
    summarize_full_pipeline_effect_distribution,
)
from crychic.inference.hypotheses import HypothesisRole
from crychic.resampling import SubjectBootstrapPlan

from . import full_pipeline_effects as _effect_adapter
from .crossfit import CrossFitArtifacts
from .full_pipeline_effects import FrozenFamilyEffectTarget
from .full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResampleStatus,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
    aligned_full_pipeline_resamples,
)

_SCHEMA_VERSION = "1.0.0"
_SCORE_COLUMN = "integrated_lr_score"
_SCORE_SEMANTICS = "family_common_integrated_lr_score_v1"
_EFFECT_ENDPOINT = "family_common_integrated_lr_context_effect_v1"
_OMNIBUS_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_SOURCE_BINDING_STATUS = "frozen_workflow_authenticated_v1"
_AUTHENTICATION_SEMANTICS = (
    "exact_frozen_target_point_and_subject_bootstrap_workflow_binding_v1"
)
_PRODUCER_MARKER = "crychic.workflow.frozen_family_specificity_support.v1"


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


def _require_point_resampling_source_alignment(
    point_artifacts: CrossFitArtifacts,
    resampling: FullPipelineResamplingResult,
) -> None:
    """Require point and bootstrap workflows to derive from one exact source."""

    identity = point_artifacts.root_input_identity
    if (
        identity.identity_id != resampling.source_input_identity_id
        or identity.input_digest != resampling.source_input_digest
    ):
        raise _contract_error(
            "Point and bootstrap workflows use different sanitized raw inputs",
            code="frozen_specificity_input_identity_mismatch",
            field="source_input_identity_id,source_input_digest",
            remediation=(
                "Fit the point estimate and bootstrap reruns from the exact same "
                "sanitized AnnData input"
            ),
        )
    if identity.config_digest != resampling.config_digest:
        raise _contract_error(
            "Point and bootstrap workflows use different analysis configurations",
            code="frozen_specificity_config_mismatch",
            field="config_digest",
            remediation=(
                "Fit the point estimate and bootstrap reruns with one exact "
                "CrychicConfig"
            ),
        )
    resource_ids = {
        fold.training.resource_bundle_content_id for fold in point_artifacts.folds
    }
    if resource_ids != {resampling.resource_bundle_content_id}:
        raise _contract_error(
            "Point and bootstrap workflows use different ligand-receptor resources",
            code="frozen_specificity_resource_bundle_mismatch",
            field="resource_bundle_content_id",
            remediation=(
                "Fit every point and bootstrap fold with one exact versioned "
                "ligand-receptor resource"
            ),
        )
    target_prior_ids = {
        fold.training.target_prior_content_id for fold in point_artifacts.folds
    }
    if target_prior_ids != {resampling.target_prior_content_id}:
        raise _contract_error(
            "Point and bootstrap workflows use different ligand-target priors",
            code="frozen_specificity_target_prior_mismatch",
            field="target_prior_content_id",
            remediation=(
                "Fit every point and bootstrap fold with one exact versioned "
                "ligand-target prior"
            ),
        )


def _require_target_binding(
    target: FrozenFamilyEffectTarget,
    score_target: FrozenFamilyEffectTarget | None,
) -> FrozenFamilyEffectTarget:
    if not isinstance(target, FrozenFamilyEffectTarget):
        raise TypeError("target must be a FrozenFamilyEffectTarget")
    target._require_intact()
    if target.endpoint != _EFFECT_ENDPOINT:
        raise _contract_error(
            "Frozen specificity requires a family context-effect target",
            code="frozen_specificity_target_endpoint_mismatch",
            field="endpoint",
            remediation="Use the frozen family context-effect hypothesis endpoint",
        )
    if target.role is not HypothesisRole.SECONDARY:
        raise _contract_error(
            "Frozen specificity requires a secondary family-effect target",
            code="frozen_specificity_target_role_mismatch",
            field="role",
            remediation="Use a secondary contrast under its primary omnibus parent",
        )
    if score_target is None:
        raise _contract_error(
            "Secondary specificity requires its frozen primary score target",
            code="frozen_specificity_score_target_missing",
            field="score_target",
            remediation="Resolve the primary parent from the same frozen universe",
        )
    if not isinstance(score_target, FrozenFamilyEffectTarget):
        raise TypeError("score_target must be FrozenFamilyEffectTarget or None")
    score_target._require_intact()
    if (
        target._universe is not score_target._universe
        or target.universe_id != score_target.universe_id
        or target.role is not HypothesisRole.SECONDARY
        or target.endpoint != _EFFECT_ENDPOINT
        or score_target.role is not HypothesisRole.PRIMARY
        or score_target.endpoint != _OMNIBUS_ENDPOINT
        or target.parent_key != score_target.hypothesis_key
        or (target.receiver, target.family_id, target.mode)
        != (score_target.receiver, score_target.family_id, score_target.mode)
    ):
        raise _contract_error(
            "Specificity score target is not the exact frozen primary parent",
            code="frozen_specificity_score_target_mismatch",
            field="universe_id,parent_key,endpoint,role,receiver,family_id,mode",
            remediation="Resolve both targets from one intact frozen universe",
        )
    return score_target


def _effect_scale_id(spec: FullPipelineEffectDistributionSpec) -> str:
    spec._require_intact()
    identifier: str = stable_id(
        "family_common_integrated_effect_scale",
        {
                "score_column": _SCORE_COLUMN,
                "score_semantics": _SCORE_SEMANTICS,
                "effect_backend": spec.effect_spec.backend,
            },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


@dataclass(frozen=True, slots=True)
class _BootstrapBinding:
    plan: SubjectBootstrapPlan
    workflow_record: FullPipelineResampleRecord
    effect_record: FullPipelineResampleEffectRecord


def _bootstrap_bindings(
    resampling: FullPipelineResamplingResult,
    effect_records: tuple[FullPipelineResampleEffectRecord, ...],
    distribution_spec: FullPipelineEffectDistributionSpec,
) -> tuple[_BootstrapBinding, ...]:
    pairs = aligned_full_pipeline_resamples(
        resampling,
        operation=FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP,
    )
    if not pairs:
        raise _contract_error(
            "Frozen specificity requires subject-bootstrap workflow plans",
            code="frozen_specificity_bootstrap_plans_missing",
            field="plans",
            remediation="Run full-pipeline subject bootstraps with retained children",
        )
    if any(
        not isinstance(item, FullPipelineResampleEffectRecord)
        for item in effect_records
    ):
        raise TypeError("effect_records must contain typed effect records")
    for item in effect_records:
        item._require_intact()
    if any(
        item.resampling_kind
        is not FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP
        for item in effect_records
    ):
        raise _contract_error(
            "Frozen specificity stores subject-bootstrap effect records only",
            code="frozen_specificity_nonbootstrap_effect_record",
            field="resampling_kind",
            remediation="Filter context-permutation effects before authentication",
        )
    by_plan = {item.plan_id: item for item in effect_records}
    if len(by_plan) != len(effect_records):
        raise _contract_error(
            "Frozen specificity effect records contain duplicate plan provenance",
            code="frozen_specificity_duplicate_effect_plan",
            field="plan_id",
            remediation="Retain every subject-bootstrap effect exactly once",
        )
    values: list[_BootstrapBinding] = []
    for plan, workflow_record in pairs:
        if not isinstance(plan, SubjectBootstrapPlan):
            raise _contract_error(
                "Subject-bootstrap operation resolved to a different plan type",
                code="frozen_specificity_bootstrap_plan_type_mismatch",
                field="plan",
                remediation="Use the intact full-pipeline resampling result",
            )
        effect_record = by_plan.get(plan.bootstrap_id)
        if effect_record is None:
            raise _contract_error(
                "Subject-bootstrap workflow plan lacks its effect record",
                code="frozen_specificity_effect_record_missing",
                field="plan_id",
                remediation="Adapt every retained subject-bootstrap child",
            )
        if (
            workflow_record.operation
            is not FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP
            or workflow_record.plan_id != plan.bootstrap_id
            or workflow_record.resample_index != plan.resample_index
            or effect_record.plan_id != plan.bootstrap_id
            or effect_record.full_pipeline_record_id != workflow_record.record_id
            or effect_record.resample_index != plan.resample_index
            or effect_record.effect_spec_id != distribution_spec.effect_spec.spec_id
            or effect_record.hypothesis_id
            != distribution_spec.effect_spec.hypothesis_id
        ):
            raise _contract_error(
                "Bootstrap plan, workflow record, and effect record are misaligned",
                code="frozen_specificity_bootstrap_lineage_mismatch",
                field="plan_id,record_id,resample_index,effect_spec_id,hypothesis_id",
                remediation="Rebuild effects from the exact retained workflow children",
            )
        if workflow_record.status is FullPipelineResampleStatus.FAILED:
            if (
                workflow_record.child is not None
                or workflow_record.crossfit_id is not None
                or effect_record.crossfit_id is not None
                or effect_record.status is not FullPipelineResampleEffectStatus.FAILED
                or effect_record.reason_code != workflow_record.failure_code
            ):
                raise _contract_error(
                    "Failed bootstrap workflow lineage is inconsistent",
                    code="frozen_specificity_failed_bootstrap_mismatch",
                    field="status,crossfit_id,reason_code",
                    remediation="Preserve the exact failed workflow record",
                )
        else:
            child = workflow_record.child
            if child is None:
                raise _contract_error(
                    "Successful bootstrap is missing its retained cross-fit child",
                    code="frozen_specificity_bootstrap_child_missing",
                    field="record_id",
                    remediation="Rerun resampling with retain_children=True",
                )
            child._require_intact()
            if (
                workflow_record.crossfit_id != child.crossfit_id
                or effect_record.crossfit_id != child.crossfit_id
                or child.spec.spec_id != resampling.crossfit_spec_id
            ):
                raise _contract_error(
                    "Retained bootstrap child differs from bound workflow provenance",
                    code="frozen_specificity_bootstrap_child_mismatch",
                    field="crossfit_id,crossfit_spec_id",
                    remediation="Reject the resampling result and rerun the workflow",
                )
        values.append(
            _BootstrapBinding(
                plan=plan,
                workflow_record=workflow_record,
                effect_record=effect_record,
            )
        )
    if len(values) != len(effect_records):
        raise _contract_error(
            "Bootstrap effect record collection has missing or extra provenance",
            code="frozen_specificity_effect_record_set_mismatch",
            field="plan_id",
            remediation="Use only effects generated from the exact bootstrap plans",
        )
    return tuple(
        sorted(
            values,
            key=lambda item: (item.plan.resample_index, item.plan.bootstrap_id),
        )
    )


def _bootstrap_source_binding_id(
    resampling: FullPipelineResamplingResult,
    bindings: tuple[_BootstrapBinding, ...],
) -> str:
    identifier: str = stable_id(
        "frozen_bootstrap_workflow_binding",
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
                "bootstrap_plan_ids": [item.plan.bootstrap_id for item in bindings],
                "workflow_record_ids": [
                    item.workflow_record.record_id for item in bindings
                ],
                "retained_crossfit_ids": [
                    item.workflow_record.crossfit_id for item in bindings
                ],
                "binding_semantics": _AUTHENTICATION_SEMANTICS,
            },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


def _validate_source_chain(
    *,
    point_artifacts: CrossFitArtifacts,
    resampling: FullPipelineResamplingResult,
    target: FrozenFamilyEffectTarget,
    score_target: FrozenFamilyEffectTarget,
    distribution_spec: FullPipelineEffectDistributionSpec,
    point_effect: OOFContextEffectResult,
    effect_records: tuple[FullPipelineResampleEffectRecord, ...],
    distribution: FullPipelineEffectDistribution,
    specificity_spec: SpecificitySupportSpec,
    numeric_result: SpecificitySupportResult,
) -> tuple[tuple[_BootstrapBinding, ...], str, str]:
    _require_target_binding(target, score_target)
    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    point_artifacts._require_intact()
    point_artifacts.spec._require_intact()
    resampling._require_intact()
    distribution_spec._require_intact()
    point_effect._require_intact()
    distribution._require_intact()
    specificity_spec._require_intact()
    numeric_result._require_intact()
    if not resampling.retain_children:
        raise _contract_error(
            "Frozen specificity requires retained full-pipeline children",
            code="frozen_specificity_children_not_retained",
            field="retain_children",
            remediation="Rerun resampling with retain_children=True",
        )
    if point_artifacts.spec.spec_id != resampling.crossfit_spec_id:
        raise _contract_error(
            "Point and bootstrap workflows use different cross-fit specs",
            code="frozen_specificity_crossfit_spec_mismatch",
            field="crossfit_spec_id",
            remediation="Use one exact CrossFitSpec for point and bootstrap runs",
        )
    _require_point_resampling_source_alignment(point_artifacts, resampling)
    if (
        distribution_spec.effect_spec.hypothesis_id != target.hypothesis_id
        or distribution_spec.effect_spec.contrast_name != target.contrast_name
    ):
        raise _contract_error(
            "Specificity distribution spec does not belong to the frozen target",
            code="frozen_specificity_distribution_target_mismatch",
            field="hypothesis_id,contrast_name",
            remediation="Build the effect spec from the exact frozen target",
        )
    if (
        point_effect.spec_id != distribution_spec.effect_spec.spec_id
        or point_effect.hypothesis_id != target.hypothesis_id
    ):
        raise _contract_error(
            "Point effect does not match the frozen specificity estimand",
            code="frozen_specificity_point_effect_mismatch",
            field="spec_id,hypothesis_id",
            remediation="Fit the point effect from this exact target and spec",
        )
    bindings = _bootstrap_bindings(resampling, effect_records, distribution_spec)
    canonical_records = tuple(item.effect_record for item in bindings)
    plan_ids = tuple(item.plan.bootstrap_id for item in bindings)
    scale_id = _effect_scale_id(distribution_spec)
    if (
        distribution.distribution_spec_id != distribution_spec.spec_id
        or distribution.point_effect_result_id != point_effect.result_id
        or distribution.resample_record_ids
        != tuple(item.record_id for item in canonical_records)
        or distribution.n_permutation_total != 0
        or distribution.n_permutation_observed != 0
        or len(distribution.permutation_effects) != 0
    ):
        raise _contract_error(
            "Bootstrap-only source distribution does not match authenticated inputs",
            code="frozen_specificity_distribution_source_mismatch",
            field="distribution_spec_id,point_effect_result_id,resample_record_ids",
            remediation="Re-summarize the authenticated bootstrap effects",
        )
    if (
        specificity_spec.effect_distribution_spec.spec_id != distribution_spec.spec_id
        or specificity_spec.hypothesis_universe_id != target.universe_id
        or specificity_spec.effect_scale_id != scale_id
        or specificity_spec.subject_bootstrap_plan_ids != tuple(sorted(plan_ids))
    ):
        raise _contract_error(
            "Numeric specificity spec is not bound to frozen workflow sources",
            code="frozen_specificity_numeric_spec_mismatch",
            field="hypothesis_universe_id,effect_scale_id,subject_bootstrap_plan_ids",
            remediation="Derive the numeric spec only through this workflow adapter",
        )
    if (
        numeric_result.source_distribution_id != distribution.distribution_id
        or numeric_result.specificity_support_spec_id != specificity_spec.spec_id
        or numeric_result.hypothesis_universe_id != target.universe_id
        or numeric_result.effect_scale_id != scale_id
        or numeric_result.source_resample_record_ids
        != tuple(item.record_id for item in canonical_records)
    ):
        raise _contract_error(
            "Numeric specificity result differs from authenticated workflow sources",
            code="frozen_specificity_numeric_result_mismatch",
            field="numeric_result_id",
            remediation="Recompute support through the frozen workflow adapter",
        )
    return (
        bindings,
        _bootstrap_source_binding_id(resampling, bindings),
        scale_id,
    )


def _result_values(
    *,
    point_artifacts: CrossFitArtifacts,
    target: FrozenFamilyEffectTarget,
    score_target: FrozenFamilyEffectTarget,
    distribution_spec: FullPipelineEffectDistributionSpec,
    point_effect: OOFContextEffectResult,
    distribution: FullPipelineEffectDistribution,
    specificity_spec: SpecificitySupportSpec,
    numeric_result: SpecificitySupportResult,
    bindings: tuple[_BootstrapBinding, ...],
    bootstrap_source_binding_id: str,
    effect_scale_id: str,
) -> dict[str, object]:
    return {
        "target_id": target.target_id,
        "score_target_id": score_target.target_id,
        "declaration_id": target.declaration_id,
        "hypothesis_universe_id": target.universe_id,
        "hypothesis_id": target.hypothesis_id,
        "contrast_name": target.contrast_name,
        "receiver": target.receiver,
        "family_id": target.family_id,
        "mode": target.mode.value,
        "effect_scale_id": effect_scale_id,
        "point_crossfit_id": point_artifacts.crossfit_id,
        "crossfit_spec_id": point_artifacts.spec.spec_id,
        "point_effect_result_id": point_effect.result_id,
        "bootstrap_source_binding_id": bootstrap_source_binding_id,
        "bootstrap_plan_ids": tuple(item.plan.bootstrap_id for item in bindings),
        "workflow_record_ids": tuple(
            item.workflow_record.record_id for item in bindings
        ),
        "effect_record_ids": tuple(item.effect_record.record_id for item in bindings),
        "distribution_spec_id": distribution_spec.spec_id,
        "source_distribution_id": distribution.distribution_id,
        "effect_spec_id": distribution_spec.effect_spec.spec_id,
        "specificity_support_spec_id": specificity_spec.spec_id,
        "bootstrap_plan_set_id": specificity_spec.bootstrap_plan_set_id,
        "numeric_result_id": numeric_result.result_id,
        "specificity_semantics": numeric_result.specificity_semantics,
        "minimum_effect": distribution_spec.minimum_effect,
        "specificity_direction": distribution_spec.specificity_direction,
        "minimum_bootstraps": specificity_spec.minimum_bootstraps,
        "specificity_support": numeric_result.specificity_support,
        "status": numeric_result.status,
        "reason_code": numeric_result.reason_code,
        "n_bootstrap_total": numeric_result.n_bootstrap_total,
        "n_bootstrap_observed": numeric_result.n_bootstrap_observed,
        "n_bootstrap_not_estimable": numeric_result.n_bootstrap_not_estimable,
        "n_bootstrap_failed": numeric_result.n_bootstrap_failed,
    }


def _identity_payload_from_values(values: dict[str, object]) -> dict[str, object]:
    status = cast(SpecificitySupportStatus, values["status"])
    direction = cast(SpecificityDirection, values["specificity_direction"])
    return {
        **values,
        "bootstrap_plan_ids": list(
            cast(tuple[str, ...], values["bootstrap_plan_ids"])
        ),
        "workflow_record_ids": list(
            cast(tuple[str, ...], values["workflow_record_ids"])
        ),
        "effect_record_ids": list(
            cast(tuple[str, ...], values["effect_record_ids"])
        ),
        "specificity_direction": direction.value,
        "status": status.value,
        "source_authenticated": True,
        "source_binding_status": _SOURCE_BINDING_STATUS,
        "authentication_semantics": _AUTHENTICATION_SEMANTICS,
        "specificity_support_release_allowed": (
            status is SpecificitySupportStatus.OBSERVED
        ),
        "formal_pq_inference_allowed": False,
        "is_posterior_probability": False,
        "is_comm_probability": False,
        "formal_inference_allowed": False,
        "producer_marker": _PRODUCER_MARKER,
    }


@dataclass(frozen=True, slots=True, init=False)
class FrozenFamilySpecificitySupport:
    """Authenticated family support over exact frozen subject bootstraps."""

    target_id: str
    score_target_id: str
    declaration_id: str
    hypothesis_universe_id: str
    hypothesis_id: str
    contrast_name: str
    receiver: str
    family_id: str
    mode: str
    effect_scale_id: str
    point_crossfit_id: str
    crossfit_spec_id: str
    point_effect_result_id: str
    bootstrap_source_binding_id: str
    bootstrap_plan_ids: tuple[str, ...]
    workflow_record_ids: tuple[str, ...]
    effect_record_ids: tuple[str, ...]
    distribution_spec_id: str
    source_distribution_id: str
    effect_spec_id: str
    specificity_support_spec_id: str
    bootstrap_plan_set_id: str
    numeric_result_id: str
    specificity_semantics: str
    minimum_effect: float
    specificity_direction: SpecificityDirection
    minimum_bootstraps: int
    specificity_support: float | None
    status: SpecificitySupportStatus
    reason_code: str | None
    n_bootstrap_total: int
    n_bootstrap_observed: int
    n_bootstrap_not_estimable: int
    n_bootstrap_failed: int
    result_id: str
    _point_artifacts: CrossFitArtifacts
    _resampling: FullPipelineResamplingResult
    _target: FrozenFamilyEffectTarget
    _score_target: FrozenFamilyEffectTarget
    _distribution_spec: FullPipelineEffectDistributionSpec
    _point_effect: OOFContextEffectResult
    _effect_records: tuple[FullPipelineResampleEffectRecord, ...]
    _distribution: FullPipelineEffectDistribution
    _specificity_spec: SpecificitySupportSpec
    _numeric_result: SpecificitySupportResult
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenFamilySpecificitySupport is producer-owned; use "
            "summarize_frozen_family_specificity_support()"
        )

    @property
    def source_authenticated(self) -> bool:
        return True

    @property
    def source_binding_status(self) -> str:
        return _SOURCE_BINDING_STATUS

    @property
    def is_posterior_probability(self) -> bool:
        return False

    @property
    def is_comm_probability(self) -> bool:
        return False

    @property
    def specificity_support_release_allowed(self) -> bool:
        return self.status is SpecificitySupportStatus.OBSERVED

    @property
    def formal_pq_inference_allowed(self) -> bool:
        return False

    @property
    def formal_inference_allowed(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return _identity_payload_from_values(
            {
                "target_id": self.target_id,
                "score_target_id": self.score_target_id,
                "declaration_id": self.declaration_id,
                "hypothesis_universe_id": self.hypothesis_universe_id,
                "hypothesis_id": self.hypothesis_id,
                "contrast_name": self.contrast_name,
                "receiver": self.receiver,
                "family_id": self.family_id,
                "mode": self.mode,
                "effect_scale_id": self.effect_scale_id,
                "point_crossfit_id": self.point_crossfit_id,
                "crossfit_spec_id": self.crossfit_spec_id,
                "point_effect_result_id": self.point_effect_result_id,
                "bootstrap_source_binding_id": self.bootstrap_source_binding_id,
                "bootstrap_plan_ids": self.bootstrap_plan_ids,
                "workflow_record_ids": self.workflow_record_ids,
                "effect_record_ids": self.effect_record_ids,
                "distribution_spec_id": self.distribution_spec_id,
                "source_distribution_id": self.source_distribution_id,
                "effect_spec_id": self.effect_spec_id,
                "specificity_support_spec_id": self.specificity_support_spec_id,
                "bootstrap_plan_set_id": self.bootstrap_plan_set_id,
                "numeric_result_id": self.numeric_result_id,
                "specificity_semantics": self.specificity_semantics,
                "minimum_effect": self.minimum_effect,
                "specificity_direction": self.specificity_direction,
                "minimum_bootstraps": self.minimum_bootstraps,
                "specificity_support": self.specificity_support,
                "status": self.status,
                "reason_code": self.reason_code,
                "n_bootstrap_total": self.n_bootstrap_total,
                "n_bootstrap_observed": self.n_bootstrap_observed,
                "n_bootstrap_not_estimable": self.n_bootstrap_not_estimable,
                "n_bootstrap_failed": self.n_bootstrap_failed,
            }
        )

    def _require_intact(self) -> None:
        try:
            bindings, source_binding_id, scale_id = _validate_source_chain(
                point_artifacts=self._point_artifacts,
                resampling=self._resampling,
                target=self._target,
                score_target=self._score_target,
                distribution_spec=self._distribution_spec,
                point_effect=self._point_effect,
                effect_records=self._effect_records,
                distribution=self._distribution,
                specificity_spec=self._specificity_spec,
                numeric_result=self._numeric_result,
            )
            expected = _result_values(
                point_artifacts=self._point_artifacts,
                target=self._target,
                score_target=self._score_target,
                distribution_spec=self._distribution_spec,
                point_effect=self._point_effect,
                distribution=self._distribution,
                specificity_spec=self._specificity_spec,
                numeric_result=self._numeric_result,
                bindings=bindings,
                bootstrap_source_binding_id=source_binding_id,
                effect_scale_id=scale_id,
            )
            expected_payload = _identity_payload_from_values(expected)
            expected_id = stable_id(
                "frozen_family_specificity_support",
                expected_payload,
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and self._identity_payload() == expected_payload
                and self.result_id == expected_id
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
                "Frozen family specificity support failed integrity validation",
                code="frozen_family_specificity_support_integrity_violation",
                field="result_id",
                remediation="Rebuild support from intact frozen workflow sources",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen family specificity support failed integrity validation",
                code="frozen_family_specificity_support_integrity_violation",
                field="result_id",
                remediation="Rebuild support from intact frozen workflow sources",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"result_id": self.result_id, **payload}


def summarize_frozen_family_specificity_support(
    point_artifacts: CrossFitArtifacts,
    resampling: FullPipelineResamplingResult,
    target: FrozenFamilyEffectTarget,
    distribution_spec: FullPipelineEffectDistributionSpec,
    *,
    score_target: FrozenFamilyEffectTarget | None = None,
) -> FrozenFamilySpecificitySupport:
    """Authenticate specificity against frozen point and bootstrap workflows."""

    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    if not isinstance(distribution_spec, FullPipelineEffectDistributionSpec):
        raise TypeError("distribution_spec must be FullPipelineEffectDistributionSpec")
    authenticated_score_target = _require_target_binding(target, score_target)
    point_artifacts._require_intact()
    resampling._require_intact()
    distribution_spec._require_intact()
    if point_artifacts.spec.spec_id != resampling.crossfit_spec_id:
        raise _contract_error(
            "Point and bootstrap workflows use different cross-fit specs",
            code="frozen_specificity_crossfit_spec_mismatch",
            field="crossfit_spec_id",
            remediation="Use one exact CrossFitSpec for point and bootstrap runs",
        )
    _require_point_resampling_source_alignment(point_artifacts, resampling)
    if (
        distribution_spec.effect_spec.hypothesis_id != target.hypothesis_id
        or distribution_spec.effect_spec.contrast_name != target.contrast_name
    ):
        raise _contract_error(
            "Specificity distribution spec does not belong to the frozen target",
            code="frozen_specificity_distribution_target_mismatch",
            field="hypothesis_id,contrast_name",
            remediation="Build the effect spec from the exact frozen target",
        )
    point_effect = _effect_adapter.fit_crossfit_family_effect(
        point_artifacts,
        target,
        distribution_spec.effect_spec,
        score_target=authenticated_score_target,
    )
    generated = _effect_adapter.full_pipeline_family_effect_records(
        resampling,
        target,
        distribution_spec.effect_spec,
        score_target=authenticated_score_target,
    )
    effect_records = tuple(
        item
        for item in generated
        if item.resampling_kind
        is FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP
    )
    bindings = _bootstrap_bindings(resampling, effect_records, distribution_spec)
    canonical_records = tuple(item.effect_record for item in bindings)
    distribution = summarize_full_pipeline_effect_distribution(
        point_effect,
        distribution_spec,
        canonical_records,
    )
    scale_id = _effect_scale_id(distribution_spec)
    specificity_spec = SpecificitySupportSpec(
        effect_distribution_spec=distribution_spec,
        hypothesis_universe_id=target.universe_id,
        effect_scale_id=scale_id,
        subject_bootstrap_plan_ids=tuple(
            item.plan.bootstrap_id for item in bindings
        ),
    )
    numeric_result = summarize_specificity_support(
        distribution,
        specificity_spec,
        canonical_records,
    )
    source_binding_id = _bootstrap_source_binding_id(resampling, bindings)
    values = _result_values(
        point_artifacts=point_artifacts,
        target=target,
        score_target=authenticated_score_target,
        distribution_spec=distribution_spec,
        point_effect=point_effect,
        distribution=distribution,
        specificity_spec=specificity_spec,
        numeric_result=numeric_result,
        bindings=bindings,
        bootstrap_source_binding_id=source_binding_id,
        effect_scale_id=scale_id,
    )
    self = object.__new__(FrozenFamilySpecificitySupport)
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(self, "_point_artifacts", point_artifacts)
    object.__setattr__(self, "_resampling", resampling)
    object.__setattr__(self, "_target", target)
    object.__setattr__(self, "_score_target", authenticated_score_target)
    object.__setattr__(self, "_distribution_spec", distribution_spec)
    object.__setattr__(self, "_point_effect", point_effect)
    object.__setattr__(self, "_effect_records", canonical_records)
    object.__setattr__(self, "_distribution", distribution)
    object.__setattr__(self, "_specificity_spec", specificity_spec)
    object.__setattr__(self, "_numeric_result", numeric_result)
    object.__setattr__(self, "_producer_marker", _PRODUCER_MARKER)
    object.__setattr__(
        self,
        "result_id",
        stable_id(
            "frozen_family_specificity_support",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    self._require_intact()
    return self


__all__ = [
    "FrozenFamilySpecificitySupport",
    "summarize_frozen_family_specificity_support",
]
