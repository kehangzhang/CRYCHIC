from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest
from benchmarks.simulation.v7_dgp import V7DGPFixture, generate_v7_dgp
from benchmarks.simulation.v7_integrated import (
    EFFECT_COLUMNS,
    SCORE_VIEW_COLUMNS,
    V7IntegratedMatrixResult,
    build_v7_score_views,
    g0_g2_equivalence_diagnostic,
    run_v7_integrated_matrix,
)
from benchmarks.simulation.v7_metrics import (
    ALIGNED_EFFECT_COLUMNS,
    METRIC_COLUMNS,
    align_v7_effect_truth,
    evaluate_v7_integrated_matrix,
)

from crychic.workflow import CrossFitArtifacts, run_subject_crossfit


@dataclass(frozen=True)
class _Prepared:
    fixture: V7DGPFixture
    crossfit: CrossFitArtifacts


@pytest.fixture(scope="module")
def prepared() -> _Prepared:
    fixture = generate_v7_dgp(
        dataset_id="v7-integrated-paired",
        dgp_family="expression_joint",
        design_kind="paired",
        seed=1701,
        candidate_sender_count=2,
        cells_per_type=2,
        subjects_per_level=8,
    )
    crossfit = run_subject_crossfit(
        fixture.adata,
        fixture.config,
        fixture.resource,
        fixture.target_prior,
        spec=fixture.crossfit_spec,
    )
    return _Prepared(fixture=fixture, crossfit=crossfit)


@pytest.fixture(scope="module")
def integrated(prepared: _Prepared) -> V7IntegratedMatrixResult:
    return run_v7_integrated_matrix(
        prepared.crossfit,
        dataset_id=prepared.fixture.dataset_id,
        design=prepared.fixture.differential_design,
        sample_metadata=prepared.fixture.sample_metadata,
        arms=(
            ("G0", "I1"),
            ("G1", "I1"),
            ("G2", "I1"),
            ("G3", "I0"),
            ("G3", "I1"),
            ("G3", "I2"),
            ("G4", "I1"),
            ("G5", "I1"),
        ),
    )


def test_score_matrix_covers_frozen_generators_and_keeps_estimands_separate(
    prepared: _Prepared,
) -> None:
    scores = build_v7_score_views(
        prepared.crossfit,
        dataset_id=prepared.fixture.dataset_id,
    )

    assert tuple(scores.columns) == SCORE_VIEW_COLUMNS
    assert set(scores["generator_id"]) == {f"G{index}" for index in range(6)}
    assert scores["out_of_fold"].all()
    assert not scores.duplicated(
        [
            "generator_id",
            "score_view",
            "contrast_scope",
            "event_id",
            "sample_id",
        ]
    ).any()
    legacy = scores.loc[scores["generator_id"].eq("G1")]
    assert not legacy.empty
    assert not legacy["outcome_agnostic"].any()
    assert legacy["condition_gate_used"].all()
    assert set(legacy["contrast_scope"]) == {"B_vs_A"}

    g5 = scores.loc[scores["generator_id"].eq("G5")]
    assert {
        "primary_sender_detection",
        "program_signed_annotation",
        "sender_attribution_with_m2",
        "coupling_prior_annotation",
    }.issubset(set(g5["score_view"]))
    assert set(
        g5.loc[g5["primary_view"], "score_view"]
    ) == {"primary_sender_detection"}


def test_g0_g2_frozen_formulas_report_constant_scale_equivalence(
    integrated: V7IntegratedMatrixResult,
) -> None:
    diagnostic = integrated.equivalence_diagnostic
    assert diagnostic == g0_g2_equivalence_diagnostic(integrated.score_views)
    assert diagnostic["rows_g0"] == diagnostic["rows_g2"]
    assert diagnostic["paired_finite_rows"] > 0
    assert diagnostic["equivalent_within_1e_12"] is True
    assert float(diagnostic["maximum_absolute_scale_residual"]) <= 1.0e-12


def test_inference_matrix_runs_i0_i1_i2_and_withholds_formal_fields(
    integrated: V7IntegratedMatrixResult,
) -> None:
    effects = integrated.effects
    assert tuple(effects.columns) == EFFECT_COLUMNS
    observed_arms = set(
        effects.loc[:, ["generator_id", "inference_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert {
        ("G0", "I1"),
        ("G1", "I1"),
        ("G2", "I1"),
        ("G3", "I0"),
        ("G3", "I1"),
        ("G3", "I2"),
        ("G4", "I1"),
        ("G5", "I1"),
    }.issubset(observed_arms)
    assert not effects["formal_inference_allowed"].any()
    assert effects["p_value"].isna().all()
    assert effects["q_value"].isna().all()
    assert effects.loc[
        effects["generator_id"].eq("G3")
        & effects["inference_id"].eq("I0"),
        "covariance_method",
    ].str.contains("raw_paired_difference").any()
    moderated = effects.loc[
        effects["generator_id"].eq("G3")
        & effects["inference_id"].eq("I2")
        & effects["status"].eq("observed")
    ]
    assert not moderated.empty
    assert moderated["prior_df"].notna().all()
    assert moderated["prior_variance"].gt(0.0).all()


def test_g4_is_post_effect_fixed_blend_and_g5_does_not_change_raw_intensity(
    integrated: V7IntegratedMatrixResult,
) -> None:
    effects = integrated.effects
    combined = effects.loc[
        effects["generator_id"].eq("G4")
        & effects["score_view"].eq("primary_fixed_effect_z_blend")
    ]
    assert not combined.empty
    assert combined["standard_error"].isna().all()
    assert combined["diagnostic_p_value"].isna().all()
    assert set(combined["covariance_method"]) == {"post_effect_fixed_z_blend"}

    columns = ["event_id", "contrast_name", "effect", "standard_error", "status"]
    g3 = effects.loc[
        effects["generator_id"].eq("G3")
        & effects["inference_id"].eq("I1")
        & effects["score_view"].eq("primary_sender_detection"),
        columns,
    ].sort_values(["event_id", "contrast_name"], ignore_index=True)
    g5 = effects.loc[
        effects["generator_id"].eq("G5")
        & effects["inference_id"].eq("I1")
        & effects["score_view"].eq("primary_sender_detection"),
        columns,
    ].sort_values(["event_id", "contrast_name"], ignore_index=True)
    pd.testing.assert_frame_equal(g3, g5)


def test_metrics_align_each_estimand_and_never_convert_unknown_truth_to_negative(
    prepared: _Prepared,
    integrated: V7IntegratedMatrixResult,
) -> None:
    aligned, metrics = evaluate_v7_integrated_matrix(
        integrated.score_views,
        integrated.effects,
        prepared.fixture.truth,
        dgp_family=prepared.fixture.dgp_family,
        design_kind=prepared.fixture.design_kind,
    )
    assert tuple(aligned.columns) == ALIGNED_EFFECT_COLUMNS
    assert tuple(metrics.columns) == METRIC_COLUMNS
    assert {
        "event_auprc",
        "event_auroc",
        "event_partial_auroc_fpr_0_10",
        "effect_spearman",
        "effect_rmse_raw_scale",
        "direction_accuracy",
        "diagnostic_false_positive_rate_alpha_0_05",
        "score_zero_fraction",
        "score_tie_fraction",
        "score_na_fraction",
    }.issubset(set(metrics["metric"]))
    combined = aligned.loc[
        aligned["score_view"].eq("primary_fixed_effect_z_blend")
    ]
    assert not combined.empty
    assert set(combined["truth_effect_kind"]) == {
        "fixed_parent_program_z_blend"
    }
    program = aligned.loc[aligned["estimand"].eq("signed_receiver_program")]
    assert not program.empty
    assert set(program["truth_effect_kind"]) == {"program_effect"}

    g0 = metrics.loc[
        metrics["generator_id"].eq("G0")
        & metrics["inference_id"].eq("I1")
        & metrics["metric"].isin(["event_auprc", "event_auroc"]),
        ["contrast_name", "metric", "value", "status"],
    ].sort_values(["contrast_name", "metric"], ignore_index=True)
    g2 = metrics.loc[
        metrics["generator_id"].eq("G2")
        & metrics["inference_id"].eq("I1")
        & metrics["metric"].isin(["event_auprc", "event_auroc"]),
        ["contrast_name", "metric", "value", "status"],
    ].sort_values(["contrast_name", "metric"], ignore_index=True)
    pd.testing.assert_frame_equal(g0, g2)

    unknown_truth = prepared.fixture.truth.copy(deep=True)
    target = unknown_truth.index[0]
    unknown_truth["truth_causal_sender"] = unknown_truth[
        "truth_causal_sender"
    ].astype("boolean")
    unknown_truth.loc[target, "truth_causal_sender"] = pd.NA
    unknown_aligned = align_v7_effect_truth(integrated.effects, unknown_truth)
    truth_row = unknown_truth.loc[target]
    unknown_event = unknown_aligned.loc[
        unknown_aligned["sender"].eq(str(truth_row["sender"]))
        & unknown_aligned["receiver"].eq(str(truth_row["receiver"]))
        & unknown_aligned["interaction_id"].eq(str(truth_row["interaction_id"]))
        & unknown_aligned["contrast_name"].eq(str(truth_row["contrast_name"]))
        & unknown_aligned["resolution"].eq("sender_lr_receiver_child")
    ]
    assert not unknown_event.empty
    assert not unknown_event["truth_known"].any()
    assert unknown_event["truth_label"].isna().all()
