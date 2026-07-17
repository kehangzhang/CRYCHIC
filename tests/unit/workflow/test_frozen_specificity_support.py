from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest

import crychic.workflow.full_pipeline_effects as effects_module
from crychic.core import ContractError, SeedLineage
from crychic.design import balanced_contrast
from crychic.inference.effects import (
    OOFContextEffectResult,
    OOFEffectSpec,
    fit_oof_context_effect,
)
from crychic.inference.full_pipeline import (
    FullPipelineEffectDistributionSpec,
    SpecificityDirection,
)
from crychic.inference.hypotheses import (
    HypothesisDeclaration,
    HypothesisRole,
    freeze_hypothesis_universe,
)
from crychic.resampling import (
    ContextPermutationPlan,
    ExchangeabilityMap,
    SubjectBootstrapPlan,
    build_exchangeability_map,
    plan_context_permutations,
    plan_subject_bootstraps,
)
from crychic.workflow.crossfit import CrossFitArtifacts, CrossFitSpec
from crychic.workflow.frozen_specificity_support import (
    FrozenFamilySpecificitySupport,
    summarize_frozen_family_specificity_support,
)
from crychic.workflow.full_pipeline_effects import (
    FamilyEffectTarget,
    FrozenFamilyEffectTarget,
    build_frozen_family_effect_target,
)
from crychic.workflow.full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResampleStatus,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
)
from crychic.workflow.receiver_universe import ReceiverTrainingSupportStatus

_ENDPOINT = "family_common_integrated_lr_context_effect_v1"


@dataclass(frozen=True, slots=True)
class _WorkflowChain:
    point: CrossFitArtifacts
    resampling: FullPipelineResamplingResult
    target: FrozenFamilyEffectTarget
    score_target: FrozenFamilyEffectTarget
    distribution_spec: FullPipelineEffectDistributionSpec
    point_effect: OOFContextEffectResult
    bootstrap_effect: OOFContextEffectResult
    not_estimable_effect: OOFContextEffectResult


def _targets() -> tuple[FrozenFamilyEffectTarget, FrozenFamilyEffectTarget]:
    primary = HypothesisDeclaration(
        endpoint="driver_family_receiver_context_omnibus_v1",
        contrast_name="driver-context-omnibus",
        receiver="Receiver",
        family_id="family-1",
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="family-specificity-primary",
    )
    secondary = HypothesisDeclaration(
        endpoint=_ENDPOINT,
        contrast_name="B-vs-A",
        receiver="Receiver",
        family_id="family-1",
        mode="state",
        role=HypothesisRole.SECONDARY,
        multiplicity_family="family-specificity-secondary",
        parent_key=primary.hypothesis_key,
    )
    universe = freeze_hypothesis_universe(
        (primary, secondary),
        universe_name="frozen-family-specificity-v1",
    )
    return (
        build_frozen_family_effect_target(universe, secondary.hypothesis_id),
        build_frozen_family_effect_target(universe, primary.hypothesis_id),
    )


def _crossfit_spec() -> CrossFitSpec:
    return CrossFitSpec(
        contrasts=(balanced_contrast(("B",), ("A",), name="B-vs-A"),),
        allowed_n_splits=(2,),
    )


def _artifact(
    crossfit_id: str,
    spec: CrossFitSpec,
    *,
    config_digest: str = "crossfit-config-digest",
) -> CrossFitArtifacts:
    artifact = object.__new__(CrossFitArtifacts)
    object.__setattr__(artifact, "crossfit_id", crossfit_id)
    object.__setattr__(artifact, "spec", spec)
    object.__setattr__(
        artifact,
        "root_input_identity",
        SimpleNamespace(
            identity_id="source-input-identity",
            input_digest="source-input-digest",
            config_digest=config_digest,
        ),
    )
    object.__setattr__(
        artifact,
        "folds",
        (
            SimpleNamespace(
                training=SimpleNamespace(
                    resource_bundle_content_id="resource-content",
                    target_prior_content_id="target-prior-content",
                ),
                receiver_training_support=(
                    SimpleNamespace(
                        receiver_id="Receiver",
                        status=ReceiverTrainingSupportStatus.OBSERVED,
                        reason_code=None,
                    ),
                ),
            ),
        ),
    )
    return artifact


def _effect(
    effect_spec: OOFEffectSpec,
    *,
    shift: float,
    observed: bool = True,
) -> OOFContextEffectResult:
    rows = [
        {
            "subject_id": f"s{subject:02d}",
            "sample_id": f"s{subject:02d}-{context}",
            "fold_id": f"fold-{subject % 2}",
            "context_id": context,
            "score": float(subject) / 10 + (shift if context == "B" else 0.0),
            "score_status": "observed",
            "scoring_function_id": f"functional-{subject % 2}",
        }
        for subject in range(12)
        for context in ("A", "B")
    ]
    if not observed:
        rows[0]["score_status"] = "not_estimable"
    return fit_oof_context_effect(pd.DataFrame(rows), effect_spec)


def _exchangeability() -> ExchangeabilityMap:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{subject}-{context}",
                "subject_id": subject,
                "condition": context,
            }
            for subject in ("s1", "s2", "s3", "s4")
            for context in ("A", "B")
        ]
    )
    return build_exchangeability_map(
        metadata,
        sample_key="sample_id",
        subject_key="subject_id",
        context_keys=("condition",),
    )


def _workflow_record(
    plan: SubjectBootstrapPlan | ContextPermutationPlan,
    spec: CrossFitSpec,
    *,
    failed: bool,
) -> FullPipelineResampleRecord:
    is_bootstrap = isinstance(plan, SubjectBootstrapPlan)
    plan_id = plan.bootstrap_id if is_bootstrap else plan.permutation_id
    child = None if failed else _artifact(f"child-{plan_id}", spec)
    return FullPipelineResampleRecord(
        operation=(
            FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP
            if is_bootstrap
            else FullPipelineResamplingOperation.CONTEXT_PERMUTATION
        ),
        resample_index=plan.resample_index,
        plan_id=plan_id,
        exchangeability_id=plan.exchangeability_id,
        materialized_input_id=f"materialized-{plan_id}",
        plan_seed_lineage=plan.seed_lineage,
        crossfit_seed_lineage=plan.seed_lineage.derive("crossfit"),
        crossfit_config_digest="crossfit-config-digest",
        status=(
            FullPipelineResampleStatus.FAILED
            if failed
            else FullPipelineResampleStatus.SUCCEEDED
        ),
        crossfit_id=None if child is None else child.crossfit_id,
        failure_type="ContractError" if failed else None,
        failure_code="synthetic_pipeline_failure" if failed else None,
        n_obs=None if failed else 8,
        n_samples=None if failed else 8,
        n_subjects=None if failed else 4,
        _child=child,
    )


def _chain(
    *,
    n_bootstraps: int,
    failed_bootstraps: frozenset[int] = frozenset(),
    include_permutation: bool = False,
) -> _WorkflowChain:
    target, score_target = _targets()
    crossfit_spec = _crossfit_spec()
    point = _artifact(
        "point-crossfit",
        crossfit_spec,
        config_digest="config-digest",
    )
    exchangeability = _exchangeability()
    root_seed = SeedLineage(91)
    bootstraps = plan_subject_bootstraps(
        exchangeability,
        n_bootstraps=n_bootstraps,
        seed_lineage=root_seed.derive("bootstrap-plans"),
    )
    plans: tuple[SubjectBootstrapPlan | ContextPermutationPlan, ...] = bootstraps
    if include_permutation:
        permutation = plan_context_permutations(
            exchangeability,
            n_permutations=1,
            seed_lineage=root_seed.derive("permutation-plans"),
        )
        plans = (*bootstraps, *permutation)
    records = tuple(
        _workflow_record(
            plan,
            crossfit_spec,
            failed=(
                isinstance(plan, SubjectBootstrapPlan)
                and plan.resample_index in failed_bootstraps
            ),
        )
        for plan in plans
    )
    resampling = FullPipelineResamplingResult(
        exchangeability=exchangeability,
        plans=plans,
        records=records,
        source_input_identity_id="source-input-identity",
        source_input_digest="source-input-digest",
        source_snapshot_id="source-snapshot",
        config_digest="config-digest",
        crossfit_spec_id=crossfit_spec.spec_id,
        resource_bundle_content_id="resource-content",
        target_prior_content_id="target-prior-content",
        root_seed_lineage=root_seed,
        retain_children=True,
    )
    effect_spec = OOFEffectSpec(
        hypothesis_id=target.hypothesis_id,
        contrast_name=target.contrast_name,
        contrast_weights=(("A", -1.0), ("B", 1.0)),
    )
    distribution_spec = FullPipelineEffectDistributionSpec(
        effect_spec=effect_spec,
        minimum_effect=2.0,
        specificity_direction=SpecificityDirection.GREATER,
    )
    return _WorkflowChain(
        point=point,
        resampling=resampling,
        target=target,
        score_target=score_target,
        distribution_spec=distribution_spec,
        point_effect=_effect(effect_spec, shift=2.5),
        bootstrap_effect=_effect(effect_spec, shift=3.0),
        not_estimable_effect=_effect(effect_spec, shift=3.0, observed=False),
    )


def _install_artifact_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    def require_intact(_: CrossFitArtifacts) -> None:
        return None

    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", require_intact)


def _install_effect_fit(
    monkeypatch: pytest.MonkeyPatch,
    chain: _WorkflowChain,
    *,
    child_not_estimable: bool = False,
) -> None:
    def fit(
        artifacts: CrossFitArtifacts,
        target: FamilyEffectTarget | FrozenFamilyEffectTarget,
        effect_spec: OOFEffectSpec,
        *,
        score_target: FrozenFamilyEffectTarget | None = None,
    ) -> OOFContextEffectResult:
        del target, effect_spec, score_target
        if artifacts.crossfit_id == chain.point.crossfit_id:
            return chain.point_effect
        if child_not_estimable:
            return chain.not_estimable_effect
        return chain.bootstrap_effect

    monkeypatch.setattr(effects_module, "fit_crossfit_family_effect", fit)


def _summarize(
    monkeypatch: pytest.MonkeyPatch,
    chain: _WorkflowChain,
    *,
    child_not_estimable: bool = False,
) -> FrozenFamilySpecificitySupport:
    _install_artifact_integrity(monkeypatch)
    _install_effect_fit(
        monkeypatch,
        chain,
        child_not_estimable=child_not_estimable,
    )
    return summarize_frozen_family_specificity_support(
        chain.point,
        chain.resampling,
        chain.target,
        chain.distribution_spec,
        score_target=chain.score_target,
    )


def test_near_real_1000_bootstrap_chain_is_source_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(n_bootstraps=1_000)

    result = _summarize(monkeypatch, chain)

    assert result.source_authenticated is True
    assert result.source_binding_status == "frozen_workflow_authenticated_v1"
    assert result.status.value == "observed"
    assert result.specificity_support == 1.0
    assert result.n_bootstrap_total == 1_000
    assert result.n_bootstrap_observed == 1_000
    assert result.n_bootstrap_not_estimable == 0
    assert result.n_bootstrap_failed == 0
    assert result.hypothesis_universe_id == chain.target.universe_id
    assert result.score_target_id == chain.score_target.target_id
    assert result.effect_scale_id.startswith("family_common_integrated_effect_scale_")
    assert result.is_posterior_probability is False
    assert result.is_comm_probability is False
    assert result.specificity_support_release_allowed is True
    assert result._numeric_result.public_release_allowed is False
    assert result.formal_pq_inference_allowed is False
    assert result.formal_inference_allowed is False
    assert result._target is chain.target
    assert result._resampling is chain.resampling
    assert len(result._effect_records) == 1_000
    assert result.workflow_record_ids == tuple(
        record.record_id for record in chain.resampling.records
    )
    payload = result.to_dict()
    assert payload["source_authenticated"] is True
    assert payload["is_posterior_probability"] is False
    assert payload["is_comm_probability"] is False
    assert payload["specificity_support_release_allowed"] is True
    assert payload["formal_pq_inference_allowed"] is False
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenFamilySpecificitySupport()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field", "forged_value", "expected_code"),
    (
        (
            "source_input_identity_id",
            "other-input-identity",
            "frozen_specificity_input_identity_mismatch",
        ),
        (
            "config_digest",
            "other-config",
            "frozen_specificity_config_mismatch",
        ),
        (
            "resource_bundle_content_id",
            "other-resource",
            "frozen_specificity_resource_bundle_mismatch",
        ),
        (
            "target_prior_content_id",
            "other-target-prior",
            "frozen_specificity_target_prior_mismatch",
        ),
    ),
)
def test_point_and_bootstrap_sources_must_match_exactly(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    forged_value: str,
    expected_code: str,
) -> None:
    chain = _chain(n_bootstraps=2)
    _install_artifact_integrity(monkeypatch)
    _install_effect_fit(monkeypatch, chain)
    mismatched = chain.resampling
    if field == "resource_bundle_content_id":
        chain.point.folds[0].training.resource_bundle_content_id = forged_value
    elif field == "target_prior_content_id":
        chain.point.folds[0].training.target_prior_content_id = forged_value
    else:
        mismatched = replace(chain.resampling, **{field: forged_value})

    with pytest.raises(ContractError) as error:
        summarize_frozen_family_specificity_support(
            chain.point,
            mismatched,
            chain.target,
            chain.distribution_spec,
            score_target=chain.score_target,
        )

    assert error.value.details.code == expected_code


def test_failed_and_not_estimable_bootstraps_remain_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_chain = _chain(n_bootstraps=2, failed_bootstraps=frozenset({0}))
    failed = _summarize(monkeypatch, failed_chain)

    assert failed.status.value == "failed"
    assert failed.specificity_support is None
    assert failed.n_bootstrap_failed == 1
    assert failed.reason_code == "specificity_bootstrap_record_failed"
    assert failed.specificity_support_release_allowed is False
    assert failed.formal_pq_inference_allowed is False

    ne_chain = _chain(n_bootstraps=2)
    not_estimable = _summarize(
        monkeypatch,
        ne_chain,
        child_not_estimable=True,
    )

    assert not_estimable.status.value == "not_estimable"
    assert not_estimable.specificity_support is None
    assert not_estimable.n_bootstrap_not_estimable == 2
    assert not_estimable.reason_code == (
        "specificity_bootstrap_record_not_estimable"
    )
    assert not_estimable.specificity_support_release_allowed is False
    assert not_estimable.formal_pq_inference_allowed is False


def test_mixed_permutations_do_not_enter_identity_or_numeric_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_only = _chain(n_bootstraps=1_000)
    mixed = _chain(n_bootstraps=1_000, include_permutation=True)
    first = _summarize(monkeypatch, bootstrap_only)
    second = _summarize(monkeypatch, mixed)

    assert bootstrap_only.resampling.result_id != mixed.resampling.result_id
    assert first.bootstrap_plan_ids == second.bootstrap_plan_ids
    assert first.workflow_record_ids == second.workflow_record_ids
    assert first.bootstrap_source_binding_id == second.bootstrap_source_binding_id
    assert first.source_distribution_id == second.source_distribution_id
    assert first.numeric_result_id == second.numeric_result_id
    assert first.result_id == second.result_id
    assert second.n_bootstrap_total == 1_000
    assert len(second._effect_records) == 1_000


def test_target_and_distribution_source_forgery_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(n_bootstraps=2)
    _install_artifact_integrity(monkeypatch)
    _install_effect_fit(monkeypatch, chain)
    legacy = FamilyEffectTarget(
        contrast_name=chain.target.contrast_name,
        receiver=chain.target.receiver,
        family_id=chain.target.family_id,
        mode=chain.target.mode,
    )
    with pytest.raises(TypeError, match="FrozenFamilyEffectTarget"):
        summarize_frozen_family_specificity_support(
            chain.point,
            chain.resampling,
            cast(FrozenFamilyEffectTarget, legacy),
            chain.distribution_spec,
            score_target=chain.score_target,
        )

    forged = object.__new__(FrozenFamilyEffectTarget)
    with pytest.raises(ContractError, match="integrity validation"):
        summarize_frozen_family_specificity_support(
            chain.point,
            chain.resampling,
            forged,
            chain.distribution_spec,
            score_target=chain.score_target,
        )

    wrong_target, _ = _targets()
    object.__setattr__(wrong_target, "hypothesis_id", "forged-hypothesis")
    with pytest.raises(ContractError, match="integrity validation"):
        summarize_frozen_family_specificity_support(
            chain.point,
            chain.resampling,
            wrong_target,
            chain.distribution_spec,
            score_target=chain.score_target,
        )


def test_secondary_target_requires_exact_primary_omnibus_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(n_bootstraps=2)
    _install_artifact_integrity(monkeypatch)
    _install_effect_fit(monkeypatch, chain)

    with pytest.raises(ContractError) as missing:
        summarize_frozen_family_specificity_support(
            chain.point,
            chain.resampling,
            chain.target,
            chain.distribution_spec,
        )
    assert missing.value.details.code == "frozen_specificity_score_target_missing"

    primary_effect = HypothesisDeclaration(
        endpoint=_ENDPOINT,
        contrast_name=chain.target.contrast_name,
        receiver=chain.target.receiver,
        family_id=chain.target.family_id,
        mode=chain.target.mode,
        role=HypothesisRole.PRIMARY,
        multiplicity_family="invalid-primary-specificity",
    )
    primary_universe = freeze_hypothesis_universe(
        (primary_effect,),
        universe_name="invalid-primary-specificity",
    )
    primary_target = build_frozen_family_effect_target(
        primary_universe,
        primary_effect.hypothesis_id,
    )
    with pytest.raises(ContractError) as role_error:
        summarize_frozen_family_specificity_support(
            chain.point,
            chain.resampling,
            primary_target,
            chain.distribution_spec,
            score_target=chain.score_target,
        )
    assert role_error.value.details.code == (
        "frozen_specificity_target_role_mismatch"
    )

    _, independently_frozen_parent = _targets()
    with pytest.raises(ContractError) as parent_error:
        summarize_frozen_family_specificity_support(
            chain.point,
            chain.resampling,
            chain.target,
            chain.distribution_spec,
            score_target=independently_frozen_parent,
        )
    assert parent_error.value.details.code == (
        "frozen_specificity_score_target_mismatch"
    )


def test_universe_and_effect_scale_are_not_caller_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = inspect.signature(
        summarize_frozen_family_specificity_support
    ).parameters
    assert "hypothesis_universe_id" not in parameters
    assert "effect_scale_id" not in parameters

    chain = _chain(n_bootstraps=2)
    result = _summarize(monkeypatch, chain)

    assert result.hypothesis_universe_id == chain.target.universe_id
    assert result.effect_scale_id == result._specificity_spec.effect_scale_id
    assert result.effect_scale_id != "caller-forged-effect-scale"


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "source",
    (
        "target",
        "score_target",
        "universe",
        "scale",
        "crossfit",
        "plan",
        "workflow_record",
        "effect_record",
    ),
)
def test_authenticated_wrapper_rejects_source_tampering(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    chain = _chain(n_bootstraps=2)
    result = _summarize(monkeypatch, chain)

    if source == "target":
        object.__setattr__(chain.target, "receiver", "forged-receiver")
    elif source == "score_target":
        object.__setattr__(chain.score_target, "receiver", "forged-parent")
    elif source == "universe":
        object.__setattr__(chain.target._universe, "universe_id", "forged-universe")
    elif source == "scale":
        object.__setattr__(result, "effect_scale_id", "forged-scale")
    elif source == "crossfit":
        object.__setattr__(chain.point.spec, "spec_id", "forged-crossfit-spec")
    elif source == "plan":
        object.__setattr__(chain.resampling.plans[0], "bootstrap_id", "forged-plan")
    elif source == "workflow_record":
        object.__setattr__(chain.resampling.records[0], "record_id", "forged-record")
    else:
        object.__setattr__(
            result._effect_records[0],
            "full_pipeline_record_id",
            "forged-effect-source-record",
        )

    with pytest.raises(ContractError, match="support failed integrity"):
        result.to_dict()
