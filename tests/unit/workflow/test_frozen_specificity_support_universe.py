from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

import crychic.workflow.frozen_specificity_support_universe as universe_module
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
    FrozenHypothesisUniverse,
    HypothesisDeclaration,
    HypothesisPrefilterPolicy,
    HypothesisPrefilterStatus,
    HypothesisRole,
    freeze_hypothesis_universe,
)
from crychic.resampling import (
    ExchangeabilityMap,
    build_exchangeability_map,
    plan_subject_bootstraps,
)
from crychic.workflow.crossfit import CrossFitArtifacts, CrossFitSpec
from crychic.workflow.frozen_specificity_support_universe import (
    FrozenSpecificitySupportUniverseCollection,
    FrozenSpecificitySupportUniverseRecord,
    summarize_frozen_specificity_support_universe,
)
from crychic.workflow.full_pipeline_effects import FrozenFamilyEffectTarget
from crychic.workflow.full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResampleStatus,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
)
from crychic.workflow.receiver_universe import ReceiverTrainingSupportStatus

_PRIMARY_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_SECONDARY_ENDPOINT = "family_common_integrated_lr_context_effect_v1"


@dataclass(frozen=True, slots=True)
class _BatchChain:
    point: CrossFitArtifacts
    resampling: FullPipelineResamplingResult
    universe: FrozenHypothesisUniverse
    primary: HypothesisDeclaration
    included: HypothesisDeclaration
    filtered: HypothesisDeclaration
    spec: FullPipelineEffectDistributionSpec
    point_effect: OOFContextEffectResult
    bootstrap_effect: OOFContextEffectResult
    not_estimable_effect: OOFContextEffectResult


def _universe(
    *,
    primary_endpoint: str = _PRIMARY_ENDPOINT,
) -> tuple[
    FrozenHypothesisUniverse,
    HypothesisDeclaration,
    HypothesisDeclaration,
    HypothesisDeclaration,
]:
    primary = HypothesisDeclaration(
        endpoint=primary_endpoint,
        contrast_name="driver-context-omnibus",
        receiver="Receiver",
        family_id="family-1",
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="specificity-primary",
    )
    included = HypothesisDeclaration(
        endpoint=_SECONDARY_ENDPOINT,
        contrast_name="B-vs-A",
        receiver="Receiver",
        family_id="family-1",
        mode="state",
        role=HypothesisRole.SECONDARY,
        multiplicity_family="specificity-secondary",
        parent_key=primary.hypothesis_key,
    )
    filtered = HypothesisDeclaration(
        endpoint=_SECONDARY_ENDPOINT,
        contrast_name="C-vs-A",
        receiver="Receiver",
        family_id="family-1",
        mode="state",
        role=HypothesisRole.SECONDARY,
        multiplicity_family="specificity-secondary",
        parent_key=primary.hypothesis_key,
        prefilter_policy=HypothesisPrefilterPolicy.EXTERNAL_RESOURCE,
        prefilter_status=HypothesisPrefilterStatus.FILTERED,
        filter_reason_code="external_resource_support_absent",
    )
    unrelated = HypothesisDeclaration(
        endpoint="unrelated-primary-endpoint-v1",
        contrast_name="unrelated-primary",
        receiver="OtherReceiver",
        family_id="family-other",
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="unrelated-primary",
    )
    frozen = freeze_hypothesis_universe(
        (primary, included, filtered, unrelated),
        universe_name="specificity-support-complete-universe-v1",
    )
    return frozen, primary, included, filtered


def _crossfit_spec() -> CrossFitSpec:
    return CrossFitSpec(
        contrasts=(
            balanced_contrast(("B",), ("A",), name="B-vs-A"),
            balanced_contrast(("C",), ("A",), name="C-vs-A"),
        ),
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


def _effect(
    spec: OOFEffectSpec,
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
    return fit_oof_context_effect(pd.DataFrame(rows), spec)


def _resampling(
    spec: CrossFitSpec,
    *,
    n_bootstraps: int,
    fail_first: bool = False,
) -> FullPipelineResamplingResult:
    exchangeability = _exchangeability()
    root = SeedLineage(707)
    plans = plan_subject_bootstraps(
        exchangeability,
        n_bootstraps=n_bootstraps,
        seed_lineage=root.derive("bootstrap-plans"),
    )
    records = []
    for plan in plans:
        failed = fail_first and plan.resample_index == 0
        child = None if failed else _artifact(f"child-{plan.bootstrap_id}", spec)
        records.append(
            FullPipelineResampleRecord(
                operation=FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP,
                resample_index=plan.resample_index,
                plan_id=plan.bootstrap_id,
                exchangeability_id=plan.exchangeability_id,
                materialized_input_id=f"materialized-{plan.bootstrap_id}",
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
        )
    return FullPipelineResamplingResult(
        exchangeability=exchangeability,
        plans=plans,
        records=tuple(records),
        source_input_identity_id="source-input-identity",
        source_input_digest="source-input-digest",
        source_snapshot_id="source-snapshot",
        config_digest="config-digest",
        crossfit_spec_id=spec.spec_id,
        resource_bundle_content_id="resource-content",
        target_prior_content_id="target-prior-content",
        root_seed_lineage=root,
        retain_children=True,
    )


def _chain(
    *,
    n_bootstraps: int,
    fail_first: bool = False,
    primary_endpoint: str = _PRIMARY_ENDPOINT,
) -> _BatchChain:
    universe, primary, included, filtered = _universe(primary_endpoint=primary_endpoint)
    crossfit_spec = _crossfit_spec()
    point = _artifact(
        "point-crossfit",
        crossfit_spec,
        config_digest="config-digest",
    )
    effect_spec = OOFEffectSpec(
        hypothesis_id=included.hypothesis_id,
        contrast_name=included.contrast_name,
        contrast_weights=(("A", -1.0), ("B", 1.0)),
    )
    distribution_spec = FullPipelineEffectDistributionSpec(
        effect_spec=effect_spec,
        minimum_effect=2.0,
        specificity_direction=SpecificityDirection.GREATER,
    )
    return _BatchChain(
        point=point,
        resampling=_resampling(
            crossfit_spec,
            n_bootstraps=n_bootstraps,
            fail_first=fail_first,
        ),
        universe=universe,
        primary=primary,
        included=included,
        filtered=filtered,
        spec=distribution_spec,
        point_effect=_effect(effect_spec, shift=2.5),
        bootstrap_effect=_effect(effect_spec, shift=3.0),
        not_estimable_effect=_effect(effect_spec, shift=3.0, observed=False),
    )


def _install_sources(
    monkeypatch: pytest.MonkeyPatch,
    chain: _BatchChain,
    *,
    child_not_estimable: bool = False,
) -> None:
    def require_intact(_: CrossFitArtifacts) -> None:
        return None

    def fit(
        artifacts: CrossFitArtifacts,
        target: FrozenFamilyEffectTarget,
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

    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", require_intact)
    monkeypatch.setattr(effects_module, "fit_crossfit_family_effect", fit)


def _summarize(
    monkeypatch: pytest.MonkeyPatch,
    chain: _BatchChain,
    *,
    child_not_estimable: bool = False,
) -> FrozenSpecificitySupportUniverseCollection:
    _install_sources(
        monkeypatch,
        chain,
        child_not_estimable=child_not_estimable,
    )
    return summarize_frozen_specificity_support_universe(
        chain.point,
        chain.resampling,
        chain.universe,
        (chain.spec,),
    )


def test_complete_universe_batch_retains_filtered_secondary_and_release_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(n_bootstraps=1_000)

    result = _summarize(monkeypatch, chain)

    assert result.complete_secondary_coverage is True
    assert result.source_authenticated is True
    assert result.universe_declaration_ids == tuple(
        item.declaration_id for item in chain.universe.declarations
    )
    assert result.applicable_secondary_hypothesis_ids == tuple(
        item.hypothesis_id
        for item in chain.universe.declarations
        if item.role is HypothesisRole.SECONDARY
        and item.endpoint == _SECONDARY_ENDPOINT
    )
    assert result.n_applicable == 2
    assert result.n_included == 1
    assert result.n_filtered == 1
    assert result.n_observed == 1
    assert result.n_not_estimable == 1
    assert result.n_failed == 0
    observed = result.result_for(chain.included.hypothesis_id)
    filtered = result.result_for(chain.filtered.hypothesis_id)
    assert observed.specificity_support == 1.0
    assert observed.specificity_support_release_allowed is True
    assert observed.score_target_id is not None
    assert filtered.status.value == "not_estimable"
    assert filtered.reason_code == chain.filtered.filter_reason_code
    assert filtered.target_id is None
    assert filtered.child_result_id is None
    assert filtered.specificity_support_release_allowed is False
    assert result.formal_pq_inference_allowed is False
    payload = result.to_dict()
    assert "p_value" not in payload
    assert "q_value" not in payload
    assert payload["is_posterior_probability"] is False
    assert payload["is_comm_probability"] is False
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenSpecificitySupportUniverseRecord()
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenSpecificitySupportUniverseCollection()


def test_global_source_mismatch_is_not_downgraded_to_failed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(n_bootstraps=2)
    _install_sources(monkeypatch, chain)
    chain.point.folds[0].training.resource_bundle_content_id = "other-resource"

    with pytest.raises(ContractError) as error:
        summarize_frozen_specificity_support_universe(
            chain.point,
            chain.resampling,
            chain.universe,
            (chain.spec,),
        )

    assert error.value.details.code == "frozen_specificity_resource_bundle_mismatch"


def test_runtime_ne_and_failed_children_never_release_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ne_chain = _chain(n_bootstraps=2)
    not_estimable = _summarize(
        monkeypatch,
        ne_chain,
        child_not_estimable=True,
    ).result_for(ne_chain.included.hypothesis_id)
    assert not_estimable.status.value == "not_estimable"
    assert not_estimable.specificity_support_release_allowed is False

    failed_chain = _chain(n_bootstraps=2, fail_first=True)
    failed = _summarize(monkeypatch, failed_chain).result_for(
        failed_chain.included.hypothesis_id
    )
    assert failed.status.value == "failed"
    assert failed.specificity_support_release_allowed is False


def test_included_adapter_exception_retains_failed_row_and_frozen_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(n_bootstraps=2)
    _install_sources(monkeypatch, chain)

    def fail_adapter(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ContractError(
            "synthetic adapter failure",
            code="synthetic_specificity_adapter_failure",
        )

    monkeypatch.setattr(
        universe_module,
        "summarize_frozen_family_specificity_support",
        fail_adapter,
    )
    first = summarize_frozen_specificity_support_universe(
        chain.point,
        chain.resampling,
        chain.universe,
        (chain.spec,),
    )
    repeated = summarize_frozen_specificity_support_universe(
        chain.point,
        chain.resampling,
        chain.universe,
        (chain.spec,),
    )
    failed = first.result_for(chain.included.hypothesis_id)
    filtered = first.result_for(chain.filtered.hypothesis_id)

    assert failed.status.value == "failed"
    assert failed.reason_code == "synthetic_specificity_adapter_failure"
    assert failed.target_id is not None
    assert failed.score_target_id is not None
    assert failed.distribution_spec_id == chain.spec.spec_id
    assert failed.child_result_id is None
    assert failed._target is not None
    assert failed._score_target is not None
    assert failed._distribution_spec is chain.spec
    assert failed.specificity_support is None
    assert failed.specificity_support_release_allowed is False
    assert (
        failed.n_bootstrap_total,
        failed.n_bootstrap_observed,
        failed.n_bootstrap_not_estimable,
        failed.n_bootstrap_failed,
    ) == (0, 0, 0, 0)
    assert first.child_result_ids == (None,)
    assert first.n_observed == 0
    assert first.n_not_estimable == 1
    assert first.n_failed == 1
    assert filtered.status.value == "not_estimable"
    assert filtered.reason_code == chain.filtered.filter_reason_code
    assert filtered.target_id is None
    assert filtered.distribution_spec_id is None
    assert first.collection_id == repeated.collection_id
    assert (
        failed.record_id == repeated.result_for(chain.included.hypothesis_id).record_id
    )

    object.__setattr__(failed, "reason_code", "forged_failure_reason")
    with pytest.raises(ContractError, match="record failed integrity"):
        failed.to_dict()


def test_distribution_spec_coverage_is_exact_and_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(n_bootstraps=2)
    _install_sources(monkeypatch, chain)

    with pytest.raises(ContractError) as missing:
        summarize_frozen_specificity_support_universe(
            chain.point,
            chain.resampling,
            chain.universe,
            (),
        )
    assert missing.value.details.code == "specificity_universe_missing_spec"

    with pytest.raises(ContractError) as duplicate:
        summarize_frozen_specificity_support_universe(
            chain.point,
            chain.resampling,
            chain.universe,
            (chain.spec, chain.spec),
        )
    assert duplicate.value.details.code == "specificity_universe_duplicate_spec"

    filtered_effect = OOFEffectSpec(
        hypothesis_id=chain.filtered.hypothesis_id,
        contrast_name=chain.filtered.contrast_name,
        contrast_weights=(("A", -1.0), ("C", 1.0)),
    )
    extra_spec = FullPipelineEffectDistributionSpec(
        effect_spec=filtered_effect,
        minimum_effect=2.0,
        specificity_direction=SpecificityDirection.GREATER,
    )
    with pytest.raises(ContractError) as extra:
        summarize_frozen_specificity_support_universe(
            chain.point,
            chain.resampling,
            chain.universe,
            (chain.spec, extra_spec),
        )
    assert extra.value.details.code == "specificity_universe_extra_spec"

    sequence = summarize_frozen_specificity_support_universe(
        chain.point,
        chain.resampling,
        chain.universe,
        (chain.spec,),
    )
    mapping = summarize_frozen_specificity_support_universe(
        chain.point,
        chain.resampling,
        chain.universe,
        {chain.included.hypothesis_id: chain.spec},
    )
    assert sequence.collection_id == mapping.collection_id


def test_wrong_primary_parent_endpoint_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(n_bootstraps=2, primary_endpoint="wrong-primary-endpoint-v1")
    _install_sources(monkeypatch, chain)

    with pytest.raises(ContractError) as error:
        summarize_frozen_specificity_support_universe(
            chain.point,
            chain.resampling,
            chain.universe,
            (chain.spec,),
        )
    assert error.value.details.code == "specificity_universe_parent_mismatch"


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "source",
    ("collection", "record", "universe", "spec", "child", "target"),
)
def test_collection_rejects_nested_source_tampering(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    chain = _chain(n_bootstraps=2)
    result = _summarize(monkeypatch, chain)
    observed = result.result_for(chain.included.hypothesis_id)

    if source == "collection":
        object.__setattr__(result, "n_observed", 99)
    elif source == "record":
        object.__setattr__(observed, "reason_code", "forged-reason")
    elif source == "universe":
        object.__setattr__(chain.universe, "universe_id", "forged-universe")
    elif source == "spec":
        object.__setattr__(chain.spec, "spec_id", "forged-spec")
    elif source == "child":
        assert observed._child_result is not None
        object.__setattr__(observed._child_result, "result_id", "forged-child")
    else:
        assert observed._target is not None
        object.__setattr__(observed._target, "receiver", "forged-target")

    with pytest.raises(ContractError, match="collection failed integrity"):
        result.to_dict()
