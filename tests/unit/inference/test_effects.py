from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crychic.core import ContractError, stable_id
from crychic.inference import (
    OOFContextEffectResult,
    OOFEffectFormalEligibility,
    OOFEffectFormalEligibilityStatus,
    OOFEffectSpec,
    assess_oof_effect_formal_eligibility,
    fit_oof_context_effect,
)


def _spec(
    *,
    minimum_clusters: int = 8,
    maximum_condition_number: float = 1.0e10,
) -> OOFEffectSpec:
    return OOFEffectSpec(
        hypothesis_id="receiver-R-family-F",
        contrast_name="B-vs-A",
        contrast_weights=(("B", 1.0), ("A", -1.0)),
        minimum_clusters_for_diagnostic_se=minimum_clusters,
        maximum_condition_number=maximum_condition_number,
    )


def _paired_scores(
    *, n_subjects: int = 12, repeats_per_context: int = 2
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_index in range(n_subjects):
        subject = f"s{subject_index:02d}"
        fold = f"fold-{subject_index % 2}"
        fold_shift = 4.0 if fold == "fold-1" else 0.0
        subject_shift = (subject_index - n_subjects / 2) * 0.1
        for context, context_effect in (("A", 0.0), ("B", 2.0)):
            for repeat in range(repeats_per_context):
                rows.append(
                    {
                        "subject_id": subject,
                        "sample_id": f"{subject}-{context}-{repeat}",
                        "fold_id": fold,
                        "context_id": context,
                        "score": fold_shift + subject_shift + context_effect,
                        "score_status": "observed",
                        "scoring_function_id": f"functional-{fold}",
                    }
                )
    return pd.DataFrame(rows)


def test_paired_repeated_samples_are_averaged_before_clustered_effect() -> None:
    result = fit_oof_context_effect(_paired_scores(), _spec())

    assert result.effect_status == "observed"
    assert result.effect == pytest.approx(2.0)
    assert result.n_samples == 48
    assert result.n_subject_context_rows == 24
    assert result.n_clusters == 12
    assert result.n_folds == 2
    assert result.uncertainty_status == "observed_cr2_diagnostic"
    assert result.standard_error is not None
    assert result.design_condition_number is not None
    assert result.design_condition_number >= 1.0
    assert result.maximum_cluster_leverage is not None
    assert 0.0 <= result.maximum_cluster_leverage < 1.0
    assert result.minimum_cr2_adjustment_eigenvalue is not None
    assert result.minimum_cr2_adjustment_eigenvalue > 0.0
    assert result.formal_inference_allowed is False
    assert not result.coefficients.flags.writeable
    assert not result.covariance.flags.writeable


def test_region_multiplicity_does_not_reweight_a_subject() -> None:
    table = _paired_scores(repeats_per_context=1)
    duplicated = pd.concat(
        [
            table,
            table.loc[table["subject_id"].eq("s00")].assign(
                sample_id=lambda frame: frame["sample_id"] + "-extra"
            ),
        ],
        ignore_index=True,
    )

    ordinary = fit_oof_context_effect(table, _spec())
    repeated = fit_oof_context_effect(duplicated, _spec())

    assert ordinary.effect == pytest.approx(2.0)
    assert repeated.effect == pytest.approx(ordinary.effect)
    assert repeated.n_subject_context_rows == ordinary.n_subject_context_rows


def test_fold_specific_constant_shift_is_absorbed_by_fold_nuisance() -> None:
    table = _paired_scores(repeats_per_context=1)
    shifted = table.copy()
    shifted.loc[shifted["fold_id"].eq("fold-1"), "score"] += 100.0

    result = fit_oof_context_effect(shifted, _spec())

    assert result.effect == pytest.approx(2.0)
    assert "fold:fold-1" in result.design_columns


def test_small_cluster_support_is_explicitly_diagnostic() -> None:
    result = fit_oof_context_effect(
        _paired_scores(n_subjects=4, repeats_per_context=1), _spec()
    )

    assert result.effect == pytest.approx(2.0)
    assert result.uncertainty_status == "diagnostic_small_cluster_support"
    assert result.formal_inference_status == (
        "requires_full_pipeline_resampling_not_released"
    )

    eligibility = assess_oof_effect_formal_eligibility(result, _spec())
    assert eligibility.status is OOFEffectFormalEligibilityStatus.NOT_ESTIMABLE
    assert eligibility.reason_code == (
        "oof_effect_insufficient_subject_clusters_for_formal_cr2"
    )
    assert not eligibility.formal_backend_eligible
    assert not eligibility.formal_inference_allowed


def test_oof_cr2_fit_has_typed_formal_backend_eligibility() -> None:
    spec = _spec()
    table = _paired_scores()
    deviations = {
        f"s{index:02d}": (index - 5.5) * 0.02 for index in range(12)
    }
    selected = table["context_id"].eq("B")
    table.loc[selected, "score"] += table.loc[selected, "subject_id"].map(
        deviations
    )
    result = fit_oof_context_effect(table, spec)

    eligibility = assess_oof_effect_formal_eligibility(result, spec)

    assert eligibility.status is OOFEffectFormalEligibilityStatus.ELIGIBLE
    assert eligibility.reason_code is None
    assert eligibility.n_clusters == 12
    assert eligibility.minimum_clusters == 8
    assert eligibility.cluster_df == 11
    assert eligibility.degrees_of_freedom_method == (
        "subject_clusters_minus_one_guard_only_v1"
    )
    assert eligibility.formal_backend_eligible
    assert not eligibility.formal_inference_allowed
    assert eligibility.to_dict()["release_requirement"] == (
        "full_pipeline_resampling_and_g3f_gate"
    )


def test_oof_cr2_eligibility_is_response_scale_equivariant() -> None:
    spec = _spec()
    table = _paired_scores()
    selected = table["context_id"].eq("B")
    deviations = {
        f"s{index:02d}": (index - 5.5) * 0.02 for index in range(12)
    }
    table.loc[selected, "score"] += table.loc[selected, "subject_id"].map(
        deviations
    )
    scaled = table.copy()
    scaled["score"] *= 1.0e-8

    ordinary = fit_oof_context_effect(table, spec)
    tiny = fit_oof_context_effect(scaled, spec)

    assert ordinary.uncertainty_status == "observed_cr2_diagnostic"
    assert tiny.uncertainty_status == "observed_cr2_diagnostic"
    assert tiny.effect == pytest.approx(ordinary.effect * 1.0e-8)
    assert tiny.standard_error == pytest.approx(ordinary.standard_error * 1.0e-8)
    assert assess_oof_effect_formal_eligibility(
        ordinary, spec
    ).formal_backend_eligible
    assert assess_oof_effect_formal_eligibility(tiny, spec).formal_backend_eligible


def test_formal_backend_enforces_six_cluster_floor() -> None:
    spec = _spec(minimum_clusters=2)
    result = fit_oof_context_effect(
        _paired_scores(n_subjects=5, repeats_per_context=1),
        spec,
    )

    eligibility = assess_oof_effect_formal_eligibility(result, spec)

    assert result.uncertainty_status == "diagnostic_small_cluster_support"
    assert eligibility.minimum_clusters == 6
    assert eligibility.status is OOFEffectFormalEligibilityStatus.NOT_ESTIMABLE
    assert eligibility.reason_code == (
        "oof_effect_insufficient_subject_clusters_for_formal_cr2"
    )


def test_condition_number_threshold_fails_closed_before_cr2() -> None:
    result = fit_oof_context_effect(
        _paired_scores(),
        _spec(maximum_condition_number=1.01),
    )

    assert result.effect_status == "not_estimable"
    assert result.reason_code == "oof_effect_ill_conditioned_design"
    assert result.design_condition_number is not None
    assert result.design_condition_number > 1.01


def test_not_estimable_oof_fit_propagates_typed_reason() -> None:
    spec = _spec()
    table = _paired_scores(repeats_per_context=1)
    table = table.loc[table["context_id"].eq("A")]
    result = fit_oof_context_effect(table, spec)

    eligibility = assess_oof_effect_formal_eligibility(result, spec)

    assert eligibility.status is OOFEffectFormalEligibilityStatus.NOT_ESTIMABLE
    assert eligibility.reason_code == "oof_effect_context_universe_mismatch"


def test_formal_eligibility_rejects_spec_mismatch_and_tampering() -> None:
    spec = _spec()
    result = fit_oof_context_effect(_paired_scores(), spec)
    other = OOFEffectSpec(
        hypothesis_id="other",
        contrast_name="B-vs-A",
        contrast_weights=(("B", 1.0), ("A", -1.0)),
    )
    with pytest.raises(ContractError) as mismatch:
        assess_oof_effect_formal_eligibility(result, other)
    assert mismatch.value.details.code == "oof_effect_formal_spec_mismatch"

    eligibility = assess_oof_effect_formal_eligibility(result, spec)
    object.__setattr__(eligibility, "n_clusters", 1)
    with pytest.raises(ContractError) as tampered:
        eligibility.to_dict()
    assert tampered.value.details.code == (
        "oof_effect_formal_eligibility_integrity_violation"
    )

    with pytest.raises(TypeError, match="producer-owned"):
        OOFEffectFormalEligibility()

    with pytest.raises(TypeError, match="producer-owned"):
        OOFContextEffectResult()


def test_formal_eligibility_recomputes_hypothesis_effect_and_variance() -> None:
    spec = _spec()
    table = _paired_scores()
    selected = table["context_id"].eq("B")
    table.loc[selected, "score"] += table.loc[selected, "subject_id"].map(
        {f"s{index:02d}": index * 0.01 for index in range(12)}
    )

    wrong_hypothesis = fit_oof_context_effect(table, spec)
    object.__setattr__(wrong_hypothesis, "hypothesis_id", "wrong-hypothesis")
    object.__setattr__(
        wrong_hypothesis,
        "result_id",
        stable_id(
            "oof_context_effect",
            wrong_hypothesis._result_identity_payload(),
            schema_version="2",
        ),
    )
    with pytest.raises(ContractError) as mismatch:
        assess_oof_effect_formal_eligibility(wrong_hypothesis, spec)
    assert mismatch.value.details.code == "oof_effect_formal_spec_mismatch"

    wrong_effect = fit_oof_context_effect(table, spec)
    assert wrong_effect.effect is not None
    object.__setattr__(wrong_effect, "effect", wrong_effect.effect + 0.5)
    object.__setattr__(
        wrong_effect,
        "result_id",
        stable_id(
            "oof_context_effect",
            wrong_effect._result_identity_payload(),
            schema_version="2",
        ),
    )
    eligibility = assess_oof_effect_formal_eligibility(wrong_effect, spec)
    assert eligibility.status is OOFEffectFormalEligibilityStatus.NOT_ESTIMABLE
    assert eligibility.reason_code == "oof_effect_contrast_coefficient_mismatch"

    wrong_variance = fit_oof_context_effect(table, spec)
    assert wrong_variance.standard_error is not None
    object.__setattr__(
        wrong_variance,
        "standard_error",
        wrong_variance.standard_error * 2.0,
    )
    object.__setattr__(
        wrong_variance,
        "result_id",
        stable_id(
            "oof_context_effect",
            wrong_variance._result_identity_payload(),
            schema_version="2",
        ),
    )
    eligibility = assess_oof_effect_formal_eligibility(wrong_variance, spec)
    assert eligibility.status is OOFEffectFormalEligibilityStatus.NOT_ESTIMABLE
    assert eligibility.reason_code == "oof_effect_degenerate_cr2_contrast_variance"


def test_one_common_scoring_function_is_required_within_each_fold() -> None:
    table = _paired_scores(repeats_per_context=1)
    table.loc[
        table["sample_id"].eq("s00-B-0"), "scoring_function_id"
    ] = "different"

    with pytest.raises(ContractError) as error:
        fit_oof_context_effect(table, _spec())

    assert error.value.details.code == "oof_effect_noncommon_functional"


def test_subject_cannot_appear_in_multiple_oof_folds() -> None:
    table = _paired_scores(repeats_per_context=1)
    table.loc[table["sample_id"].eq("s00-B-0"), "fold_id"] = "fold-1"
    table.loc[table["sample_id"].eq("s00-B-0"), "scoring_function_id"] = (
        "functional-fold-1"
    )

    with pytest.raises(ContractError) as error:
        fit_oof_context_effect(table, _spec())

    assert error.value.details.code == "oof_effect_subject_fold_leakage"


def test_not_estimable_input_score_fails_closed_without_dropping_sample() -> None:
    table = _paired_scores(repeats_per_context=1)
    table.loc[table["sample_id"].eq("s00-B-0"), ["score", "score_status"]] = [
        np.nan,
        "not_estimable",
    ]

    result = fit_oof_context_effect(table, _spec())

    assert result.effect_status == "not_estimable"
    assert result.effect is None
    assert result.standard_error is None
    assert result.reason_code == "incomplete_oof_score_coverage"


def test_structural_zero_must_be_exactly_zero() -> None:
    table = _paired_scores(repeats_per_context=1)
    table.loc[table["sample_id"].eq("s00-B-0"), "score_status"] = (
        "structural_zero"
    )

    with pytest.raises(ContractError) as error:
        fit_oof_context_effect(table, _spec())

    assert error.value.details.code == "oof_effect_invalid_structural_zero"


def test_context_universe_mismatch_is_typed_not_estimable() -> None:
    table = _paired_scores(repeats_per_context=1)
    table = table.loc[table["context_id"].eq("A")]

    result = fit_oof_context_effect(table, _spec())

    assert result.effect_status == "not_estimable"
    assert result.reason_code == "oof_effect_context_universe_mismatch"


def test_spec_identity_is_order_invariant_and_requires_a_contrast() -> None:
    first = _spec()
    second = OOFEffectSpec(
        hypothesis_id="receiver-R-family-F",
        contrast_name="B-vs-A",
        contrast_weights=(("A", -1), ("B", 1)),
    )

    assert first.spec_id == second.spec_id
    with pytest.raises(ValueError, match="positive and negative"):
        OOFEffectSpec(
            hypothesis_id="bad",
            contrast_name="bad",
            contrast_weights=(("A", 0.0), ("B", 1.0)),
        )
