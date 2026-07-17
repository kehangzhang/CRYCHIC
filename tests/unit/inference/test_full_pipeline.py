from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from tests.support.g3f import insufficient_g3f_gate, passed_g3f_gate

from crychic.core import ContractError
from crychic.inference import (
    FullPipelineEffectDistributionSpec,
    FullPipelineEffectResamplingKind,
    FullPipelineResampleEffectRecord,
    FullPipelineResampleEffectStatus,
    G3FrequencyCalibrationGate,
    G3FrequencyGateStatus,
    OOFEffectSpec,
    SpecificityDirection,
    fit_oof_context_effect,
    summarize_full_pipeline_effect_distribution,
)
from crychic.inference.hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisDeclaration,
    HypothesisRole,
    freeze_hypothesis_universe,
)


def _effect_spec() -> OOFEffectSpec:
    return OOFEffectSpec(
        hypothesis_id="receiver-R-family-F",
        contrast_name="B-vs-A",
        contrast_weights=(("B", 1.0), ("A", -1.0)),
    )


def _point_effect():
    rows = [
        {
            "subject_id": f"s{subject:02d}",
            "sample_id": f"s{subject:02d}-{context}",
            "fold_id": f"fold-{subject % 2}",
            "context_id": context,
            "score": float(subject) / 10 + (2.0 if context == "B" else 0.0),
            "score_status": "observed",
            "scoring_function_id": f"functional-{subject % 2}",
        }
        for subject in range(12)
        for context in ("A", "B")
    ]
    return fit_oof_context_effect(pd.DataFrame(rows), _effect_spec())


@lru_cache(maxsize=1)
def _gate_universe() -> FrozenHypothesisUniverse:
    primary = HypothesisDeclaration(
        endpoint="driver_family_receiver_context_omnibus_v1",
        contrast_name="all-contexts",
        receiver="receiver-R",
        family_id="family-F",
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="g3f-primary",
    )
    child = HypothesisDeclaration(
        endpoint="family_common_integrated_lr_context_effect_v1",
        contrast_name="B-vs-A",
        receiver=primary.receiver,
        family_id=primary.family_id,
        mode="state",
        role=HypothesisRole.SECONDARY,
        multiplicity_family="g3f-secondary",
        parent_key=primary.hypothesis_key,
    )
    return freeze_hypothesis_universe(
        (primary, child), universe_name="full-pipeline-unit-g3f"
    )


def _passed_gate() -> G3FrequencyCalibrationGate:
    return passed_g3f_gate(_gate_universe())


def _record(
    kind: FullPipelineEffectResamplingKind,
    index: int,
    effect: float | None,
    *,
    status: FullPipelineResampleEffectStatus = (
        FullPipelineResampleEffectStatus.OBSERVED
    ),
) -> FullPipelineResampleEffectRecord:
    observed = status is FullPipelineResampleEffectStatus.OBSERVED
    return FullPipelineResampleEffectRecord(
        full_pipeline_record_id=f"pipeline-record-{kind.value}-{index}",
        plan_id=f"plan-{kind.value}-{index}",
        crossfit_id=f"crossfit-{kind.value}-{index}",
        resampling_kind=kind,
        resample_index=index,
        effect_spec_id=_effect_spec().spec_id,
        hypothesis_id=_effect_spec().hypothesis_id,
        effect_result_id=f"effect-result-{kind.value}-{index}",
        effect=effect if observed else None,
        status=status,
        reason_code=None if observed else "resample_effect_not_estimable",
    )


def test_prefit_not_estimable_effect_record_does_not_fabricate_result_id() -> None:
    record = FullPipelineResampleEffectRecord(
        full_pipeline_record_id="workflow-record-prefit-ne",
        plan_id="bootstrap-plan-prefit-ne",
        crossfit_id="crossfit-prefit-ne",
        resampling_kind=FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP,
        resample_index=7,
        effect_spec_id="effect-spec-prefit-ne",
        hypothesis_id="hypothesis-prefit-ne",
        effect_result_id=None,
        effect=None,
        status=FullPipelineResampleEffectStatus.NOT_ESTIMABLE,
        reason_code="receiver_absent_in_outer_training",
    )

    assert record.effect_result_id is None
    assert record.to_dict()["effect_result_id"] is None


def _records() -> tuple[FullPipelineResampleEffectRecord, ...]:
    bootstrap = tuple(
        _record(FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP, index, effect)
        for index, effect in enumerate((1.0, 2.0, 3.0, 4.0))
    )
    permutation = tuple(
        _record(FullPipelineEffectResamplingKind.CONTEXT_PERMUTATION, index, effect)
        for index, effect in enumerate((-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0))
    )
    missing = _record(
        FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP,
        99,
        None,
        status=FullPipelineResampleEffectStatus.NOT_ESTIMABLE,
    )
    return (*bootstrap, *permutation, missing)


def _release_records() -> tuple[FullPipelineResampleEffectRecord, ...]:
    bootstrap = tuple(
        _record(
            FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP,
            index,
            2.0 + 0.25 * math.sin(index),
        )
        for index in range(1_000)
    )
    permutation = tuple(
        _record(
            FullPipelineEffectResamplingKind.CONTEXT_PERMUTATION,
            index,
            -3.0 + 6.0 * index / 999,
        )
        for index in range(1_000)
    )
    return (*bootstrap, *permutation)


def _distribution_spec() -> FullPipelineEffectDistributionSpec:
    return FullPipelineEffectDistributionSpec(
        effect_spec=_effect_spec(),
        minimum_effect=2.0,
        specificity_direction=SpecificityDirection.GREATER,
    )


def test_g3_frequency_gate_is_producer_owned_and_enforces_thresholds() -> None:
    passed = _passed_gate()
    insufficient = insufficient_g3f_gate(_gate_universe())

    assert passed.status is G3FrequencyGateStatus.PASSED
    assert passed.formal_release_allowed is True
    assert passed.hierarchical_q_release_allowed is True
    assert insufficient.status is G3FrequencyGateStatus.NOT_ESTIMABLE
    assert insufficient.reason_code == "g3_frequency_scenario_replicates_below_1000"
    assert passed.to_dict()["thresholds"]["type_i_upper_limit"] == 0.06
    assert passed.to_dict()["thresholds"]["mixed_fdr_upper_limit"] == 0.07
    with pytest.raises(TypeError, match="producer-owned"):
        G3FrequencyCalibrationGate()


def test_hierarchical_gate_requires_bound_primary_and_selective_fdr_evidence() -> None:
    passed = _passed_gate()

    assert passed.formal_release_allowed is True
    assert passed.hierarchical_q_release_allowed is True
    assert passed.hypothesis_universe_id == _gate_universe().universe_id
    assert passed.primary_fdr_upper_maximum == 0.0
    assert passed.selective_child_fdr_upper_maximum == 0.0
    assert passed.to_dict()["hierarchical_q_release_allowed"] is True


def test_g3_gate_rejects_tampered_binding() -> None:
    gate = _passed_gate()
    original = gate.hypothesis_universe_id
    object.__setattr__(gate, "hypothesis_universe_id", "forged-universe")

    try:
        with pytest.raises(ContractError) as error:
            gate.to_dict()
    finally:
        object.__setattr__(gate, "hypothesis_universe_id", original)

    assert error.value.details.code == "g3_frequency_gate_integrity_violation"


def test_g3_gate_rejects_tampered_evidence_lineage() -> None:
    gate = _passed_gate()
    original = gate.evidence_id
    object.__setattr__(gate, "evidence_id", "forged-evidence")

    try:
        with pytest.raises(ContractError) as error:
            gate.to_dict()
    finally:
        object.__setattr__(gate, "evidence_id", original)

    assert error.value.details.code == "g3_frequency_gate_integrity_violation"


def test_effect_summary_rejects_tampered_spec_record_and_distribution() -> None:
    point = _point_effect()
    spec = _distribution_spec()
    records = _records()
    object.__setattr__(records[0], "effect", 999.0)

    with pytest.raises(ContractError) as record_error:
        summarize_full_pipeline_effect_distribution(point, spec, records)

    assert (
        record_error.value.details.code
        == "full_pipeline_effect_record_integrity_violation"
    )

    spec = _distribution_spec()
    object.__setattr__(spec.effect_spec, "contrast_name", "forged-contrast")
    with pytest.raises(ContractError) as spec_error:
        summarize_full_pipeline_effect_distribution(point, spec, _records())

    assert (
        spec_error.value.details.code
        == "full_pipeline_effect_distribution_spec_integrity_violation"
    )

    distribution = summarize_full_pipeline_effect_distribution(
        _point_effect(),
        _distribution_spec(),
        _records(),
    )
    object.__setattr__(distribution, "point_effect", 999.0)
    with pytest.raises(ContractError) as distribution_error:
        distribution.to_dict()

    assert (
        distribution_error.value.details.code
        == "full_pipeline_effect_distribution_integrity_violation"
    )


def test_without_passed_gate_only_diagnostic_distribution_is_available() -> None:
    result = summarize_full_pipeline_effect_distribution(
        _point_effect(),
        _distribution_spec(),
        _records(),
    )

    assert result.point_effect == pytest.approx(2.0)
    assert result.n_bootstrap_total == 5
    assert result.n_bootstrap_observed == 4
    assert result.diagnostic_bootstrap_standard_error == pytest.approx(
        np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    )
    assert result.diagnostic_specificity_frequency == pytest.approx(0.5)
    assert result.diagnostic_empirical_permutation_p == pytest.approx(3 / 8)
    assert result.standard_error is None
    assert result.ci_lower is None and result.ci_upper is None
    assert result.p_value is None and result.q_value is None
    assert result.formal_inference_allowed is False
    assert result.q_value_status == (
        "available_only_from_complete_hierarchical_collection_g3_gated"
    )
    assert "probability" not in result.to_dict()
    assert not result.bootstrap_effects.flags.writeable


def test_passed_gate_releases_se_ci_p_but_never_ordinary_bh_q() -> None:
    gate = _passed_gate()
    result = summarize_full_pipeline_effect_distribution(
        _point_effect(),
        _distribution_spec(),
        _release_records(),
        calibration_gate=gate,
    )

    assert result.formal_inference_allowed is True
    assert result.calibration_gate_id == gate.gate_id
    assert result.standard_error == result.diagnostic_bootstrap_standard_error
    assert result.ci_lower == result.diagnostic_ci_lower
    assert result.ci_upper == result.diagnostic_ci_upper
    assert result.p_value == result.diagnostic_empirical_permutation_p
    assert result.q_value is None
    assert result.q_value_status == (
        "available_only_from_complete_hierarchical_collection_g3_gated"
    )


def test_passed_gate_cannot_override_small_point_cluster_support() -> None:
    rows = [
        {
            "subject_id": f"s{subject:02d}",
            "sample_id": f"s{subject:02d}-{context}",
            "fold_id": f"fold-{subject % 2}",
            "context_id": context,
            "score": float(subject) / 10 + (2.0 if context == "B" else 0.0),
            "score_status": "observed",
            "scoring_function_id": f"functional-{subject % 2}",
        }
        for subject in range(6)
        for context in ("A", "B")
    ]
    point = fit_oof_context_effect(pd.DataFrame(rows), _effect_spec())

    result = summarize_full_pipeline_effect_distribution(
        point,
        _distribution_spec(),
        _release_records(),
        calibration_gate=_passed_gate(),
    )

    assert point.uncertainty_status == "diagnostic_small_cluster_support"
    assert result.point_n_clusters == 6
    assert result.minimum_point_clusters == 8
    assert result.formal_inference_allowed is False
    assert result.p_value is None
    assert result.formal_reason_code == "full_pipeline_point_clusters_below_minimum"


def test_passed_gate_cannot_override_incomplete_runtime_distribution() -> None:
    gate = _passed_gate()
    result = summarize_full_pipeline_effect_distribution(
        _point_effect(),
        _distribution_spec(),
        _records(),
        calibration_gate=gate,
    )

    assert result.formal_inference_allowed is False
    assert result.standard_error is None and result.p_value is None
    assert result.formal_reason_code == (
        "full_pipeline_bootstrap_distribution_incomplete"
    )


def test_nonrelease_gate_does_not_release_formal_fields() -> None:
    gate = insufficient_g3f_gate(_gate_universe())
    result = summarize_full_pipeline_effect_distribution(
        _point_effect(),
        _distribution_spec(),
        _records(),
        calibration_gate=gate,
    )

    assert result.standard_error is None
    assert result.p_value is None
    assert result.formal_reason_code == gate.reason_code


def test_distribution_constructor_rejects_count_and_release_contradictions() -> None:
    result = summarize_full_pipeline_effect_distribution(
        _point_effect(),
        _distribution_spec(),
        _records(),
    )

    with pytest.raises(ValueError, match="counts do not match"):
        replace(result, n_bootstrap_observed=result.n_bootstrap_observed + 1)
    with pytest.raises(ValueError, match="released formal fields"):
        replace(
            result,
            formal_inference_status=(
                "g3_frequency_calibrated_se_ci_p_released_q_collection_only"
            ),
            formal_reason_code=None,
            calibration_gate_id="forged-gate",
        )


def test_distribution_rejects_duplicate_or_wrong_hypothesis_provenance() -> None:
    records = _records()
    with pytest.raises(ContractError) as duplicate_error:
        summarize_full_pipeline_effect_distribution(
            _point_effect(),
            _distribution_spec(),
            (records[0], records[0]),
        )
    assert duplicate_error.value.details.code == (
        "duplicate_full_pipeline_effect_record"
    )

    wrong = replace(records[0], hypothesis_id="different-hypothesis")
    with pytest.raises(ContractError) as mismatch_error:
        summarize_full_pipeline_effect_distribution(
            _point_effect(),
            _distribution_spec(),
            (wrong,),
        )
    assert mismatch_error.value.details.code == (
        "full_pipeline_effect_record_spec_mismatch"
    )
