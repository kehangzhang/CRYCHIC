from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from tests.support.g3f import passed_g3f_gate

import crychic.workflow.frozen_hypothesis_inference as batch_module
import crychic.workflow.full_pipeline_effects as effects_module
from crychic.core import ContractError
from crychic.inference.effects import OOFEffectSpec
from crychic.inference.full_pipeline import (
    FullPipelineEffectDistribution,
    FullPipelineEffectDistributionSpec,
    SpecificityDirection,
)
from crychic.inference.hierarchical import (
    HierarchicalFDRReleaseStatus,
    freeze_hierarchical_fdr_spec,
)
from crychic.inference.hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisCoverageStatus,
    HypothesisDeclaration,
    HypothesisPrefilterPolicy,
    HypothesisPrefilterStatus,
    HypothesisRole,
    freeze_hypothesis_universe,
)
from crychic.inference.omnibus import FullPipelineOmnibusDistribution
from crychic.resampling import ContextPermutationPlan
from crychic.workflow.crossfit import CrossFitArtifacts
from crychic.workflow.frozen_hypothesis_inference import (
    FrozenHypothesisInferenceCollection,
    FrozenHypothesisInferenceInput,
    FrozenHypothesisInferenceRecord,
    run_frozen_hypothesis_inference,
)
from crychic.workflow.full_pipeline_effects import (
    build_frozen_family_effect_target,
    family_effect_oof_score_table,
)
from crychic.workflow.full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
)

_PRIMARY_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_SECONDARY_ENDPOINT = "family_common_integrated_lr_context_effect_v1"


@pytest.fixture(autouse=True)
def _stub_large_parent_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", lambda self: None)
    monkeypatch.setattr(
        FullPipelineResamplingResult,
        "_require_intact",
        lambda self: None,
    )


def _primary(
    *,
    contrast_name: str = "all_contexts_v1",
    mode: str = "state",
) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint=_PRIMARY_ENDPOINT,
        contrast_name=contrast_name,
        receiver="Receiver",
        family_id="family-1",
        mode=mode,
        role=HypothesisRole.PRIMARY,
        multiplicity_family="gene-state-primary",
    )


def _secondary(
    parent: HypothesisDeclaration,
    *,
    contrast_name: str = "stim_vs_control",
    endpoint: str = _SECONDARY_ENDPOINT,
    filtered: bool = False,
) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint=endpoint,
        contrast_name=contrast_name,
        receiver=parent.receiver,
        family_id=parent.family_id,
        mode=parent.mode,
        role=HypothesisRole.SECONDARY,
        multiplicity_family="gene-state-secondary",
        parent_key=parent.hypothesis_key,
        prefilter_policy=(
            HypothesisPrefilterPolicy.POOLED_CONTEXT_BLIND_SUPPORT
            if filtered
            else HypothesisPrefilterPolicy.NONE
        ),
        prefilter_status=(
            HypothesisPrefilterStatus.FILTERED
            if filtered
            else HypothesisPrefilterStatus.INCLUDED
        ),
        filter_reason_code="pooled_support_absent" if filtered else None,
    )


def _universe() -> tuple[
    FrozenHypothesisUniverse,
    HypothesisDeclaration,
    HypothesisDeclaration,
    HypothesisDeclaration,
]:
    primary = _primary()
    secondary = _secondary(primary)
    filtered = _secondary(
        primary,
        contrast_name="late_vs_early",
        filtered=True,
    )
    universe = freeze_hypothesis_universe(
        (primary, secondary, filtered),
        universe_name="frozen-batch-fixture-v1",
    )
    return universe, primary, secondary, filtered


def _artifacts(identifier: str) -> CrossFitArtifacts:
    artifacts = object.__new__(CrossFitArtifacts)
    object.__setattr__(artifacts, "crossfit_id", identifier)
    return artifacts


def _resampling(
    identifier: str,
    *,
    plan_prefix: str = "permutation",
) -> FullPipelineResamplingResult:
    plans: list[ContextPermutationPlan] = []
    records: list[FullPipelineResampleRecord] = []
    for index in range(1_000):
        plan = object.__new__(ContextPermutationPlan)
        object.__setattr__(plan, "permutation_id", f"{plan_prefix}-{index:04d}")
        object.__setattr__(plan, "exchangeability_id", "exchangeability-v1")
        object.__setattr__(plan, "resample_index", index)
        object.__setattr__(plan, "seed_lineage", f"seed-{plan_prefix}-{index:04d}")
        record = object.__new__(FullPipelineResampleRecord)
        object.__setattr__(
            record,
            "operation",
            FullPipelineResamplingOperation.CONTEXT_PERMUTATION,
        )
        object.__setattr__(record, "plan_id", plan.permutation_id)
        object.__setattr__(record, "resample_index", index)
        object.__setattr__(record, "exchangeability_id", "exchangeability-v1")
        object.__setattr__(record, "plan_seed_lineage", plan.seed_lineage)
        object.__setattr__(
            record,
            "record_id",
            f"record-{identifier}-{index:04d}",
        )
        plans.append(plan)
        records.append(record)
    result = object.__new__(FullPipelineResamplingResult)
    object.__setattr__(result, "plans", tuple(plans))
    object.__setattr__(result, "records", tuple(records))
    object.__setattr__(
        result,
        "exchangeability",
        SimpleNamespace(exchangeability_id="exchangeability-v1"),
    )
    object.__setattr__(result, "retain_children", True)
    object.__setattr__(result, "result_id", identifier)
    return result


def _secondary_spec(
    declaration: HypothesisDeclaration,
) -> FullPipelineEffectDistributionSpec:
    effect_spec = OOFEffectSpec(
        hypothesis_id=declaration.hypothesis_id,
        contrast_name=declaration.contrast_name,
        contrast_weights=(("context-a", -1.0), ("context-b", 1.0)),
    )
    return FullPipelineEffectDistributionSpec(
        effect_spec=effect_spec,
        minimum_effect=0.1,
        specificity_direction=SpecificityDirection.TWO_SIDED,
    )


def _inputs(
    primary: HypothesisDeclaration,
    secondary: HypothesisDeclaration,
    *,
    secondary_plan_prefix: str = "permutation",
) -> tuple[FrozenHypothesisInferenceInput, ...]:
    return (
        FrozenHypothesisInferenceInput(
            hypothesis_id=primary.hypothesis_id,
            point_artifacts=_artifacts("point-shared"),
            resampling=_resampling("resampling-shared"),
        ),
        FrozenHypothesisInferenceInput(
            hypothesis_id=secondary.hypothesis_id,
            point_artifacts=_artifacts("point-shared"),
            resampling=_resampling(
                "resampling-shared",
                plan_prefix=secondary_plan_prefix,
            ),
            secondary_distribution_spec=_secondary_spec(secondary),
        ),
    )


def _install_success_adapters(
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary_p: float = 0.01,
    secondary_p: float = 0.02,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        FullPipelineOmnibusDistribution,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        FullPipelineEffectDistribution,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        batch_module,
        "build_family_omnibus_spec",
        lambda *args: SimpleNamespace(spec_id="omnibus-spec"),
    )

    def primary_summary(
        point: CrossFitArtifacts,
        resampling: FullPipelineResamplingResult,
        target: object,
        spec: object,
    ) -> FullPipelineOmnibusDistribution:
        distribution = object.__new__(FullPipelineOmnibusDistribution)
        object.__setattr__(distribution, "hypothesis_id", target.hypothesis_id)
        object.__setattr__(distribution, "omnibus_spec_id", spec.spec_id)
        object.__setattr__(
            distribution,
            "permutation_plan_ids",
            tuple(plan.permutation_id for plan in resampling.plans),
        )
        object.__setattr__(
            distribution,
            "full_pipeline_record_ids",
            tuple(record.record_id for record in resampling.records),
        )
        object.__setattr__(distribution, "diagnostic_empirical_p", primary_p)
        object.__setattr__(
            distribution,
            "diagnostic_status",
            "diagnostic_complete_minimum_1000",
        )
        object.__setattr__(distribution, "n_permutation_total", 1_000)
        object.__setattr__(distribution, "n_permutation_observed", 1_000)
        object.__setattr__(
            distribution,
            "formal_reason_code",
            "g3_frequency_calibration_gate_not_applied",
        )
        object.__setattr__(distribution, "distribution_id", "primary-distribution")
        return distribution

    monkeypatch.setattr(
        batch_module,
        "summarize_family_omnibus_full_pipeline",
        primary_summary,
    )
    monkeypatch.setattr(
        batch_module,
        "build_family_effect_oof_spec",
        lambda artifacts, target: _secondary_spec(target._declaration).effect_spec,
    )

    def secondary_summary(
        point: CrossFitArtifacts,
        resampling: FullPipelineResamplingResult,
        target: object,
        distribution_spec: FullPipelineEffectDistributionSpec,
        *,
        calibration_gate: object,
        score_target: object,
    ) -> FullPipelineEffectDistribution:
        captured["child_target"] = target
        captured["score_target"] = score_target
        captured["calibration_gate"] = calibration_gate
        distribution = object.__new__(FullPipelineEffectDistribution)
        object.__setattr__(distribution, "hypothesis_id", target.hypothesis_id)
        object.__setattr__(
            distribution,
            "effect_spec_id",
            distribution_spec.effect_spec.spec_id,
        )
        object.__setattr__(
            distribution,
            "distribution_spec_id",
            distribution_spec.spec_id,
        )
        object.__setattr__(distribution, "p_value", None)
        object.__setattr__(distribution, "q_value", None)
        object.__setattr__(distribution, "calibration_gate_id", None)
        object.__setattr__(distribution, "point_effect", 0.5)
        object.__setattr__(distribution, "point_n_clusters", 8)
        object.__setattr__(distribution, "minimum_point_clusters", 8)
        object.__setattr__(
            distribution,
            "point_uncertainty_status",
            "observed_cr2_diagnostic",
        )
        object.__setattr__(distribution, "n_permutation_total", 1_000)
        object.__setattr__(distribution, "n_permutation_observed", 1_000)
        object.__setattr__(
            distribution,
            "diagnostic_empirical_permutation_p",
            secondary_p,
        )
        object.__setattr__(
            distribution,
            "distribution_id",
            "secondary-distribution",
        )
        return distribution

    monkeypatch.setattr(
        batch_module,
        "summarize_family_effect_full_pipeline",
        secondary_summary,
    )
    return captured


def _passed_gate(
    universe: FrozenHypothesisUniverse,
) -> object:
    return passed_g3f_gate(universe, freeze_hierarchical_fdr_spec())


def test_parent_bound_effect_uses_primary_scores_and_child_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _primary()
    secondary = _secondary(primary)
    universe = freeze_hypothesis_universe(
        (primary, secondary),
        universe_name="parent-bound-effect",
    )
    parent_target = build_frozen_family_effect_target(
        universe,
        primary.hypothesis_id,
    )
    child_target = build_frozen_family_effect_target(
        universe,
        secondary.hypothesis_id,
    )
    child_functional = SimpleNamespace(
        context_ids=("context-a", "context-b"),
        family_common_functional_id="child-functional",
    )
    parent_functional = SimpleNamespace(
        context_ids=("context-b", "context-a"),
        family_common_functional_id="parent-functional",
    )
    child_application = SimpleNamespace(
        family_scores=pd.DataFrame(
            {
                "family_id": ["family-1", "family-1"],
                "mode": ["state", "state"],
                "sample_id": ["sample-a", "sample-b"],
                "subject_id": ["subject-a", "subject-b"],
                "context_id": ["context-a", "context-b"],
                "status": ["ok", "ok"],
                "integrated_lr_score": [99.0, 99.0],
            }
        )
    )
    parent_application = SimpleNamespace(
        family_scores=pd.DataFrame(
            {
                "family_id": ["family-1", "family-1"],
                "mode": ["state", "state"],
                "sample_id": ["sample-a", "sample-b"],
                "subject_id": ["subject-a", "subject-b"],
                "context_id": ["context-a", "context-b"],
                "status": ["ok", "ok"],
                "integrated_lr_score": [1.0, 2.0],
            }
        )
    )

    def chains(
        artifacts: CrossFitArtifacts,
        target: object,
    ) -> tuple[tuple[str, object, object], ...]:
        if target.hypothesis_id == child_target.hypothesis_id:
            return (("fold-1", child_functional, child_application),)
        return (("fold-1", parent_functional, parent_application),)

    monkeypatch.setattr(effects_module, "_matched_chains", chains)
    with pytest.raises(ContractError) as missing_parent:
        family_effect_oof_score_table(
            _artifacts("point-missing-parent"),
            child_target,
            _secondary_spec(secondary).effect_spec,
        )
    assert missing_parent.value.details.code == (
        "family_effect_secondary_parent_score_target_missing"
    )

    table = family_effect_oof_score_table(
        _artifacts("point-parent-bound"),
        child_target,
        _secondary_spec(secondary).effect_spec,
        score_target=parent_target,
    )

    assert table["score"].tolist() == [1.0, 2.0]
    assert set(table["scoring_function_id"]) == {"parent-functional"}

    parent_functional.context_ids = ("context-a", "context-c")
    with pytest.raises(ContractError) as error:
        family_effect_oof_score_table(
            _artifacts("point-context-mismatch"),
            child_target,
            _secondary_spec(secondary).effect_spec,
            score_target=parent_target,
        )
    assert error.value.details.code == (
        "family_effect_parent_child_context_universe_mismatch"
    )


def test_complete_batch_retains_filtered_row_and_parent_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, primary, secondary, filtered = _universe()
    captured = _install_success_adapters(monkeypatch)

    result = run_frozen_hypothesis_inference(
        universe,
        _inputs(primary, secondary),
    )

    assert result.coverage.complete is True
    assert result.coverage.n_observed == 2
    assert result.coverage.n_not_estimable == 1
    assert result.coverage.n_failed == 0
    assert result.context_permutation_plan_set_synchronized is True
    assert len(result.common_context_permutation_plan_ids) == 1_000
    assert result.q_value_release_allowed is False
    assert (
        result.hierarchical_fdr.release_status
        is HierarchicalFDRReleaseStatus.CANDIDATE_ONLY
    )
    primary_row = result.result_for(primary.hypothesis_id)
    secondary_row = result.result_for(secondary.hypothesis_id)
    filtered_row = result.result_for(filtered.hypothesis_id)
    assert primary_row.candidate_p_value == pytest.approx(0.01)
    assert secondary_row.candidate_p_value == pytest.approx(0.02)
    assert secondary_row.score_target_id != secondary_row.target_id
    assert captured["score_target"].hypothesis_id == primary.hypothesis_id
    assert captured["child_target"].hypothesis_id == secondary.hypothesis_id
    assert captured["calibration_gate"] is None
    assert filtered_row.status is HypothesisCoverageStatus.NOT_ESTIMABLE
    assert filtered_row.reason_code == filtered.filter_reason_code
    assert filtered_row.candidate_p_value is None
    assert filtered_row.effective_p_value == 1.0
    assert filtered_row.context_permutation_plan_ids == ()
    assert all(row.q_value is None for row in result.hierarchical_fdr.records)
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenHypothesisInferenceRecord()
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenHypothesisInferenceCollection()


def test_collection_rejects_nested_coverage_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, primary, secondary, _ = _universe()
    _install_success_adapters(monkeypatch)
    result = run_frozen_hypothesis_inference(
        universe,
        _inputs(primary, secondary),
    )

    object.__setattr__(result.coverage, "n_observed", 999)
    with pytest.raises(ContractError) as error:
        result.to_dict()
    assert error.value.details.code == (
        "frozen_hypothesis_inference_collection_integrity_violation"
    )


def test_matching_complete_gate_is_the_only_formal_q_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, primary, secondary, _ = _universe()
    _install_success_adapters(monkeypatch)

    result = run_frozen_hypothesis_inference(
        universe,
        _inputs(primary, secondary),
        calibration_gate=_passed_gate(universe),
    )

    assert result.q_value_release_allowed is True
    assert result.hierarchical_fdr.release_status is (
        HierarchicalFDRReleaseStatus.RELEASED
    )
    for declaration in (primary, secondary):
        hierarchical = result.hierarchical_fdr.result_for(declaration.hypothesis_id)
        assert hierarchical.q_value == hierarchical.candidate_q_value


def test_adapter_exception_becomes_failed_without_dropping_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, primary, secondary, filtered = _universe()
    _install_success_adapters(monkeypatch)

    def fail_secondary(*args: object, **kwargs: object) -> object:
        raise ContractError(
            "synthetic secondary failure",
            code="synthetic_secondary_failure",
            field="secondary",
            remediation="fixture",
        )

    monkeypatch.setattr(
        batch_module,
        "summarize_family_effect_full_pipeline",
        fail_secondary,
    )
    result = run_frozen_hypothesis_inference(
        universe,
        _inputs(primary, secondary),
    )

    assert len(result.records) == universe.multiplicity_denominator
    assert result.coverage.n_observed == 1
    assert result.coverage.n_failed == 1
    assert result.coverage.n_not_estimable == 1
    failed = result.result_for(secondary.hypothesis_id)
    assert failed.status is HypothesisCoverageStatus.FAILED
    assert failed.reason_code == "synthetic_secondary_failure"
    assert failed.effective_p_value == 1.0
    assert result.result_for(filtered.hypothesis_id).status is (
        HypothesisCoverageStatus.NOT_ESTIMABLE
    )
    assert result.hierarchical_fdr.release_status is (
        HierarchicalFDRReleaseStatus.BLOCKED
    )


def test_mismatched_observed_plan_sets_fail_closed_and_keep_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, primary, secondary, _ = _universe()
    _install_success_adapters(monkeypatch)

    result = run_frozen_hypothesis_inference(
        universe,
        _inputs(
            primary,
            secondary,
            secondary_plan_prefix="different-permutation",
        ),
    )

    assert result.coverage.n_observed == 0
    assert result.coverage.n_failed == 2
    assert result.common_context_permutation_plan_ids == ()
    for declaration, expected_p in ((primary, 0.01), (secondary, 0.02)):
        row = result.result_for(declaration.hypothesis_id)
        assert row.status is HypothesisCoverageStatus.FAILED
        assert row.reason_code == (
            "frozen_hypothesis_context_permutation_plan_set_mismatch"
        )
        assert row.candidate_p_value == pytest.approx(expected_p)
        assert row.effective_p_value == 1.0
    assert result.q_value_release_allowed is False
    assert all(row.q_value is None for row in result.hierarchical_fdr.records)


def test_incomplete_permutation_diagnostic_p_is_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, primary, secondary, _ = _universe()
    _install_success_adapters(monkeypatch)
    complete_summary = batch_module.summarize_family_effect_full_pipeline

    def incomplete_summary(*args: Any, **kwargs: Any) -> FullPipelineEffectDistribution:
        distribution = complete_summary(*args, **kwargs)
        object.__setattr__(distribution, "n_permutation_observed", 999)
        return distribution

    monkeypatch.setattr(
        batch_module,
        "summarize_family_effect_full_pipeline",
        incomplete_summary,
    )

    result = run_frozen_hypothesis_inference(
        universe,
        _inputs(primary, secondary),
        calibration_gate=_passed_gate(universe),
    )

    secondary_row = result.result_for(secondary.hypothesis_id)
    assert secondary_row.status is HypothesisCoverageStatus.NOT_ESTIMABLE
    assert secondary_row.candidate_p_value == pytest.approx(0.02)
    assert secondary_row.effective_p_value == 1.0
    assert result.q_value_release_allowed is False
    assert all(row.q_value is None for row in result.hierarchical_fdr.records)


def test_small_cluster_secondary_diagnostic_p_is_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, primary, secondary, _ = _universe()
    _install_success_adapters(monkeypatch)
    complete_summary = batch_module.summarize_family_effect_full_pipeline

    def small_cluster_summary(
        *args: Any, **kwargs: Any
    ) -> FullPipelineEffectDistribution:
        distribution = complete_summary(*args, **kwargs)
        object.__setattr__(distribution, "point_n_clusters", 7)
        return distribution

    monkeypatch.setattr(
        batch_module,
        "summarize_family_effect_full_pipeline",
        small_cluster_summary,
    )

    result = run_frozen_hypothesis_inference(
        universe,
        _inputs(primary, secondary),
        calibration_gate=_passed_gate(universe),
    )

    secondary_row = result.result_for(secondary.hypothesis_id)
    assert secondary_row.status is HypothesisCoverageStatus.NOT_ESTIMABLE
    assert secondary_row.candidate_p_value == pytest.approx(0.02)
    assert secondary_row.reason_code == "full_pipeline_point_clusters_below_minimum"
    assert result.q_value_release_allowed is False


def test_inputs_and_v1_scope_are_exact_contracts() -> None:
    universe, primary, secondary, _ = _universe()
    with pytest.raises(ContractError) as missing:
        run_frozen_hypothesis_inference(
            universe,
            _inputs(primary, secondary)[:1],
        )
    assert missing.value.details.code == ("missing_frozen_hypothesis_inference_input")

    mismatched_parents = (
        FrozenHypothesisInferenceInput(
            hypothesis_id=primary.hypothesis_id,
            point_artifacts=_artifacts("point-primary"),
            resampling=_resampling("resampling-primary"),
        ),
        FrozenHypothesisInferenceInput(
            hypothesis_id=secondary.hypothesis_id,
            point_artifacts=_artifacts("point-secondary"),
            resampling=_resampling("resampling-secondary"),
            secondary_distribution_spec=_secondary_spec(secondary),
        ),
    )
    with pytest.raises(ContractError) as parent_error:
        run_frozen_hypothesis_inference(universe, mismatched_parents)
    assert parent_error.value.details.code == (
        "frozen_hypothesis_inference_parent_run_mismatch"
    )

    ecosystem_primary = _primary(mode="ecosystem")
    ecosystem_secondary = _secondary(ecosystem_primary)
    ecosystem_universe = freeze_hypothesis_universe(
        (ecosystem_primary, ecosystem_secondary),
        universe_name="unsupported-ecosystem-hierarchy",
    )
    with pytest.raises(ContractError) as mode_error:
        run_frozen_hypothesis_inference(ecosystem_universe, ())
    assert mode_error.value.details.code == "frozen_hypothesis_v1_mode_mismatch"

    legacy_primary = _primary()
    legacy_secondary = _secondary(
        legacy_primary,
        endpoint="driver_family_receiver_context_posthoc_v1",
    )
    legacy_universe = freeze_hypothesis_universe(
        (legacy_primary, legacy_secondary),
        universe_name="unsupported-secondary-endpoint",
    )
    with pytest.raises(ContractError) as endpoint_error:
        run_frozen_hypothesis_inference(legacy_universe, ())
    assert endpoint_error.value.details.code == (
        "frozen_hypothesis_v1_endpoint_mismatch"
    )
