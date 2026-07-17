"""Strict workflow adapter for primary driver-family context omnibus tests."""

from __future__ import annotations

from crychic.core import ContractError, CrychicError
from crychic.inference.effects import OOFEffectSpec
from crychic.inference.hypotheses import HypothesisRole
from crychic.inference.omnibus import (
    FullPipelineOmnibusDistribution,
    FullPipelineOmnibusRecord,
    FullPipelineOmnibusRecordStatus,
    OOFContextOmnibusResult,
    OOFContextOmnibusSpec,
    build_full_pipeline_omnibus_record,
    fit_oof_context_omnibus,
    summarize_full_pipeline_omnibus,
)
from crychic.resampling import ContextPermutationPlan, SubjectBootstrapPlan

from .crossfit import CrossFitArtifacts
from .full_pipeline_effects import (
    FrozenFamilyEffectTarget,
    _matched_chains,
    family_effect_oof_score_table,
)
from .full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResampleStatus,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
)

_OMNIBUS_ENDPOINT = "driver_family_receiver_context_omnibus_v1"


def _require_primary_omnibus_target(target: FrozenFamilyEffectTarget) -> None:
    if not isinstance(target, FrozenFamilyEffectTarget):
        raise TypeError("target must be a frozen family-effect target")
    target._require_intact()
    if target.endpoint != _OMNIBUS_ENDPOINT:
        raise ContractError(
            "Family omnibus target uses the wrong frozen endpoint",
            code="family_omnibus_target_endpoint_mismatch",
            field="endpoint",
            remediation=f"Use endpoint {_OMNIBUS_ENDPOINT!r}",
        )
    if target.role is not HypothesisRole.PRIMARY:
        raise ContractError(
            "Family omnibus target must be a primary hypothesis",
            code="family_omnibus_target_role_mismatch",
            field="role",
            remediation="Build the target from a frozen primary declaration",
        )


def _omnibus_name(target: FrozenFamilyEffectTarget) -> str:
    # Binding target_id here prevents reuse after the frozen universe changes.
    return f"{target.contrast_name}:{target.target_id}"


def _context_ids(
    artifacts: CrossFitArtifacts,
    target: FrozenFamilyEffectTarget,
) -> tuple[str, ...]:
    observed: set[tuple[str, ...]] = set()
    for _, functional, _ in _matched_chains(artifacts, target):
        supplied = tuple(functional.context_ids)
        if len(set(supplied)) != len(supplied):
            raise ContractError(
                "Family omnibus functional contains duplicate context IDs",
                code="family_omnibus_duplicate_context",
                field="context_ids",
                remediation="Use the intact family-common scoring functional",
            )
        observed.add(tuple(sorted(supplied)))
    if len(observed) != 1:
        raise ContractError(
            "Family omnibus context universe differs across folds",
            code="family_omnibus_fold_context_mismatch",
            field="context_ids",
            remediation="Use one frozen context universe across all folds",
        )
    return next(iter(observed))


def _require_target_spec(
    target: FrozenFamilyEffectTarget,
    omnibus_spec: OOFContextOmnibusSpec,
) -> None:
    _require_primary_omnibus_target(target)
    if not isinstance(omnibus_spec, OOFContextOmnibusSpec):
        raise TypeError("omnibus_spec must be an OOFContextOmnibusSpec")
    omnibus_spec._require_intact()
    if (
        omnibus_spec.hypothesis_id != target.hypothesis_id
        or omnibus_spec.omnibus_name != _omnibus_name(target)
    ):
        raise ContractError(
            "Omnibus specification does not belong to the frozen family target",
            code="family_omnibus_spec_target_mismatch",
            field="hypothesis_id,omnibus_name",
            remediation="Build the omnibus spec from the exact frozen target",
        )


def build_family_omnibus_spec(
    artifacts: CrossFitArtifacts,
    target: FrozenFamilyEffectTarget,
    *,
    minimum_clusters_for_diagnostic_se: int = 8,
) -> OOFContextOmnibusSpec:
    """Freeze all exact family-common context IDs for one primary omnibus."""

    if not isinstance(artifacts, CrossFitArtifacts):
        raise TypeError("artifacts must be CrossFitArtifacts")
    _require_primary_omnibus_target(target)
    return OOFContextOmnibusSpec(
        hypothesis_id=target.hypothesis_id,
        omnibus_name=_omnibus_name(target),
        context_ids=_context_ids(artifacts, target),
        minimum_clusters_for_diagnostic_se=minimum_clusters_for_diagnostic_se,
    )


def _score_extraction_spec(
    target: FrozenFamilyEffectTarget,
    omnibus_spec: OOFContextOmnibusSpec,
) -> OOFEffectSpec:
    support = omnibus_spec.support_effect_spec()
    return OOFEffectSpec(
        hypothesis_id=target.hypothesis_id,
        contrast_name=target.contrast_name,
        contrast_weights=support.contrast_weights,
        minimum_clusters_for_diagnostic_se=(
            omnibus_spec.minimum_clusters_for_diagnostic_se
        ),
    )


def fit_crossfit_family_omnibus(
    artifacts: CrossFitArtifacts,
    target: FrozenFamilyEffectTarget,
    omnibus_spec: OOFContextOmnibusSpec,
) -> OOFContextOmnibusResult:
    """Fit the primary Wald omnibus from exact family-common OOF scores."""

    if not isinstance(artifacts, CrossFitArtifacts):
        raise TypeError("artifacts must be CrossFitArtifacts")
    _require_target_spec(target, omnibus_spec)
    if _context_ids(artifacts, target) != omnibus_spec.context_ids:
        raise ContractError(
            "Cross-fit contexts do not match the frozen omnibus specification",
            code="family_omnibus_context_universe_mismatch",
            field="context_ids",
            remediation="Build the omnibus spec from these exact point artifacts",
        )
    scores = family_effect_oof_score_table(
        artifacts,
        target,
        _score_extraction_spec(target, omnibus_spec),
    )
    return fit_oof_context_omnibus(scores, omnibus_spec)


def _aligned_context_permutations(
    resampling: FullPipelineResamplingResult,
) -> tuple[tuple[ContextPermutationPlan, FullPipelineResampleRecord], ...]:
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be a FullPipelineResamplingResult")
    if not resampling.retain_children:
        raise ContractError(
            "Omnibus refitting requires retained full-pipeline children",
            code="full_pipeline_omnibus_children_not_retained",
            field="retain_children",
            remediation="Rerun context permutations with retain_children=True",
        )
    if len(resampling.plans) != len(resampling.records):
        raise ContractError(
            "Full-pipeline resampling plans and records are not aligned",
            code="full_pipeline_omnibus_plan_alignment_mismatch",
            field="plans,records",
            remediation="Use an intact full-pipeline resampling result",
        )
    selected: list[tuple[ContextPermutationPlan, FullPipelineResampleRecord]] = []
    exchangeability_id = resampling.exchangeability.exchangeability_id
    for plan, record in zip(resampling.plans, resampling.records, strict=True):
        if isinstance(plan, ContextPermutationPlan):
            operation = FullPipelineResamplingOperation.CONTEXT_PERMUTATION
            plan_id = plan.permutation_id
        elif isinstance(plan, SubjectBootstrapPlan):
            operation = FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP
            plan_id = plan.bootstrap_id
        else:  # pragma: no cover - guarded by the result contract
            raise TypeError("resampling result contains an unsupported plan")
        if (
            record.operation is not operation
            or record.plan_id != plan_id
            or record.resample_index != plan.resample_index
            or plan.exchangeability_id != exchangeability_id
            or record.exchangeability_id != exchangeability_id
            or record.plan_seed_lineage != plan.seed_lineage
        ):
            raise ContractError(
                "Full-pipeline resampling plan lineage is misaligned",
                code="full_pipeline_omnibus_plan_alignment_mismatch",
                field="plan_id,resample_index,exchangeability_id,seed_lineage",
                remediation="Use the intact plan and execution record pairs",
            )
        if isinstance(plan, ContextPermutationPlan):
            selected.append((plan, record))
    if not selected:
        raise ContractError(
            "Family omnibus requires full-pipeline context permutations",
            code="full_pipeline_omnibus_permutations_missing",
            field="operation",
            remediation="Run at least one legal context permutation",
        )
    return tuple(selected)


def _failure_reason(error: Exception) -> str:
    if isinstance(error, CrychicError):
        reason_code: str = error.details.code
        return reason_code
    return f"family_omnibus_adapter_failed_{type(error).__qualname__}"


def full_pipeline_family_omnibus_records(
    resampling: FullPipelineResamplingResult,
    target: FrozenFamilyEffectTarget,
    omnibus_spec: OOFContextOmnibusSpec,
) -> tuple[FullPipelineOmnibusRecord, ...]:
    """Refit one typed omnibus for every context-permutation child."""

    _require_target_spec(target, omnibus_spec)
    pairs = _aligned_context_permutations(resampling)
    values: list[FullPipelineOmnibusRecord] = []
    for plan, workflow_record in pairs:
        if workflow_record.status is FullPipelineResampleStatus.FAILED:
            values.append(
                build_full_pipeline_omnibus_record(
                    full_pipeline_record_id=workflow_record.record_id,
                    plan_id=plan.permutation_id,
                    crossfit_id=None,
                    resample_index=plan.resample_index,
                    omnibus_spec_id=omnibus_spec.spec_id,
                    hypothesis_id=omnibus_spec.hypothesis_id,
                    omnibus_result_id=None,
                    wald_statistic=None,
                    status=FullPipelineOmnibusRecordStatus.FAILED,
                    reason_code=workflow_record.failure_code,
                )
            )
            continue
        child = workflow_record.child
        if child is None:
            raise ContractError(
                "Successful context permutation is missing its retained child",
                code="full_pipeline_omnibus_child_missing",
                field="record_id",
                remediation="Reject the result and rerun with child retention",
            )
        if child.crossfit_id != workflow_record.crossfit_id:
            raise ContractError(
                "Retained context-permutation child differs from its record",
                code="full_pipeline_omnibus_child_provenance_mismatch",
                field="crossfit_id",
                remediation="Reject the corrupted resampling result",
            )
        if child.spec.spec_id != resampling.crossfit_spec_id:
            raise ContractError(
                "Retained child uses a different cross-fit specification",
                code="full_pipeline_omnibus_child_spec_mismatch",
                field="crossfit_spec_id",
                remediation="Use children from the frozen resampling spec",
            )
        try:
            result = fit_crossfit_family_omnibus(child, target, omnibus_spec)
            observed = result.observed
            values.append(
                build_full_pipeline_omnibus_record(
                    full_pipeline_record_id=workflow_record.record_id,
                    plan_id=plan.permutation_id,
                    crossfit_id=child.crossfit_id,
                    resample_index=plan.resample_index,
                    omnibus_spec_id=omnibus_spec.spec_id,
                    hypothesis_id=omnibus_spec.hypothesis_id,
                    omnibus_result_id=result.result_id,
                    wald_statistic=(result.wald_statistic if observed else None),
                    status=(
                        FullPipelineOmnibusRecordStatus.OBSERVED
                        if observed
                        else FullPipelineOmnibusRecordStatus.NOT_ESTIMABLE
                    ),
                    reason_code=None if observed else result.reason_code,
                )
            )
        except Exception as error:
            values.append(
                build_full_pipeline_omnibus_record(
                    full_pipeline_record_id=workflow_record.record_id,
                    plan_id=plan.permutation_id,
                    crossfit_id=child.crossfit_id,
                    resample_index=plan.resample_index,
                    omnibus_spec_id=omnibus_spec.spec_id,
                    hypothesis_id=omnibus_spec.hypothesis_id,
                    omnibus_result_id=None,
                    wald_statistic=None,
                    status=FullPipelineOmnibusRecordStatus.FAILED,
                    reason_code=_failure_reason(error),
                )
            )
    return tuple(values)


def summarize_family_omnibus_full_pipeline(
    point_artifacts: CrossFitArtifacts,
    resampling: FullPipelineResamplingResult,
    target: FrozenFamilyEffectTarget,
    omnibus_spec: OOFContextOmnibusSpec,
) -> FullPipelineOmnibusDistribution:
    """Fit and summarize a primary omnibus over full-pipeline permutations."""

    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    _require_target_spec(target, omnibus_spec)
    if point_artifacts.spec.spec_id != resampling.crossfit_spec_id:
        raise ContractError(
            "Point and permutation runs use different cross-fit specifications",
            code="full_pipeline_omnibus_crossfit_spec_mismatch",
            field="crossfit_spec_id",
            remediation="Use one frozen CrossFitSpec for point and permutations",
        )
    point = fit_crossfit_family_omnibus(point_artifacts, target, omnibus_spec)
    records = full_pipeline_family_omnibus_records(
        resampling,
        target,
        omnibus_spec,
    )
    return summarize_full_pipeline_omnibus(point, omnibus_spec, records)


__all__ = [
    "build_family_omnibus_spec",
    "fit_crossfit_family_omnibus",
    "full_pipeline_family_omnibus_records",
    "summarize_family_omnibus_full_pipeline",
]
