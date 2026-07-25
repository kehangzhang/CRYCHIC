from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest
from benchmarks.simulation.v7_component_swaps import (
    E2_ANNOTATION_ARMS,
    E2_ARMS,
    E2_BASE_ARM,
    E2_HARD_GATE_ARMS,
    build_e2_component_swap_score_views,
    run_v7_e2_component_swap,
)
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
from benchmarks.simulation.v7_sender_swaps import (
    E3_ARMS,
    E3_DETECTION_ARM,
    E3_LEGACY_ARM,
    E3_M2_ARM,
    E3_METRICS,
    E3_NULL_SENDER_ARM,
    build_e3_sender_swap_score_views,
    run_v7_e3_sender_swap,
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


def test_e2_annotation_arms_are_exact_g3_copies_and_hard_gates_are_typed(
    prepared: _Prepared,
) -> None:
    scores, ledger = build_e2_component_swap_score_views(
        prepared.crossfit,
        dataset_id=prepared.fixture.dataset_id,
    )
    assert set(scores["score_view"]) == set(E2_ARMS)
    comparison_key = ["contrast_scope", "event_id", "fold_id", "sample_id"]
    base = scores.loc[
        scores["score_view"].eq(E2_BASE_ARM),
        [*comparison_key, "score", "score_status"],
    ].sort_values(comparison_key, ignore_index=True)
    for arm in E2_ANNOTATION_ARMS:
        annotation = scores.loc[
            scores["score_view"].eq(arm),
            [*comparison_key, "score", "score_status"],
        ].sort_values(comparison_key, ignore_index=True)
        pd.testing.assert_frame_equal(base, annotation)

    join_key = [
        "fold_id",
        "sample_id",
        "subject_id",
        "condition",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
    ]
    usable = {"observed", "low_evidence", "structural_impossible"}
    for arm, stage in E2_HARD_GATE_ARMS.items():
        hard = scores.loc[scores["score_view"].eq(arm)].merge(
            ledger.loc[
                :,
                ["contrast_name", *join_key, f"{stage}_status"],
            ],
            left_on=["contrast_scope", *join_key],
            right_on=["contrast_name", *join_key],
            how="left",
            validate="one_to_one",
        )
        reference = scores.loc[
            scores["score_view"].eq(E2_BASE_ARM),
            ["contrast_scope", *join_key, "score", "score_status"],
        ].rename(columns={"score": "base_score", "score_status": "base_status"})
        hard = hard.merge(
            reference,
            on=["contrast_scope", *join_key],
            how="left",
            validate="one_to_one",
        )
        retained = hard[f"{stage}_status"].eq("passed")
        failed = (
            hard[f"{stage}_status"].eq("failed")
            & hard["base_status"].isin(usable)
            & hard["base_score"].notna()
        )
        pd.testing.assert_series_equal(
            hard.loc[retained, "score"].reset_index(drop=True),
            hard.loc[retained, "base_score"].reset_index(drop=True),
            check_names=False,
        )
        assert hard.loc[failed, "score"].eq(0.0).all()
        assert hard.loc[failed, "score_status"].eq("low_evidence").all()
        unavailable = ~(retained | failed)
        assert hard.loc[unavailable, "score"].isna().all()
        assert hard.loc[unavailable, "score_status"].eq("not_estimable").all()


def test_e2_runs_every_arm_through_i1_without_formal_fields(
    prepared: _Prepared,
) -> None:
    result = run_v7_e2_component_swap(
        prepared.crossfit,
        dataset_id=prepared.fixture.dataset_id,
        design=prepared.fixture.differential_design,
        sample_metadata=prepared.fixture.sample_metadata,
        truth=prepared.fixture.truth,
        dgp_family=prepared.fixture.dgp_family,
        design_kind=prepared.fixture.design_kind,
    )

    assert set(result.effects["score_view"]) == set(E2_ARMS)
    assert set(result.effects["inference_id"]) == {"I1"}
    assert not result.effects["formal_inference_allowed"].any()
    assert result.effects[["p_value", "q_value"]].isna().all(axis=None)
    assert {"event_auprc", "effect_spearman"}.issubset(
        set(result.metrics["metric"])
    )


def test_e3_builds_detection_legacy_null_sender_and_m2_on_one_oof_axis(
    prepared: _Prepared,
) -> None:
    scores, auxiliary = build_e3_sender_swap_score_views(
        prepared.crossfit,
        dataset_id=prepared.fixture.dataset_id,
    )

    assert set(scores["score_view"]) == set(E3_ARMS)
    main = build_v7_score_views(
        prepared.crossfit,
        dataset_id=prepared.fixture.dataset_id,
    )
    key = ["event_id", "fold_id", "sample_id"]
    expected_detection = main.loc[
        main["generator_id"].eq("G3")
        & main["score_view"].eq("primary_sender_detection"),
        [*key, "score", "score_status"],
    ].sort_values(key, ignore_index=True)
    observed_detection = scores.loc[
        scores["score_view"].eq(E3_DETECTION_ARM),
        [*key, "score", "score_status"],
    ].sort_values(key, ignore_index=True)
    pd.testing.assert_frame_equal(expected_detection, observed_detection)

    legacy = scores.loc[scores["score_view"].eq(E3_LEGACY_ARM), "score"].dropna()
    assert legacy.between(0.0, 1.0).all()
    attribution = scores.loc[
        scores["score_view"].isin([E3_NULL_SENDER_ARM, E3_M2_ARM]), "score"
    ].dropna()
    assert attribution.between(0.0, 1.0).all()
    observed_entropy = pd.to_numeric(
        auxiliary["attribution_entropy"], errors="coerce"
    ).dropna()
    assert observed_entropy.between(0.0, 1.0).all()

    parent_means = (
        auxiliary.loc[
            auxiliary["score_view"].isin([E3_NULL_SENDER_ARM, E3_M2_ARM])
        ]
        .groupby("score_view", observed=True)["parent_score"]
        .mean()
    )
    assert parent_means[E3_NULL_SENDER_ARM] == pytest.approx(
        parent_means[E3_M2_ARM]
    )


def test_e3_runs_matched_i1_and_emits_all_sender_metrics(
    prepared: _Prepared,
) -> None:
    result = run_v7_e3_sender_swap(
        prepared.crossfit,
        dataset_id=prepared.fixture.dataset_id,
        design=prepared.fixture.differential_design,
        sample_metadata=prepared.fixture.sample_metadata,
        truth=prepared.fixture.truth,
        dgp_family=prepared.fixture.dgp_family,
        design_kind=prepared.fixture.design_kind,
        candidate_sender_count=2,
    )

    assert set(result.effects["score_view"]) == set(E3_ARMS)
    assert set(result.effects["inference_id"]) == {"I1"}
    assert not result.effects["formal_inference_allowed"].any()
    assert result.effects[["p_value", "q_value"]].isna().all(axis=None)
    assert set(result.sender_metrics["metric"]) == set(E3_METRICS)
    assert set(result.sender_metrics["candidate_sender_count"]) == {2}
    assert set(result.sender_metrics["score_view"]) == set(E3_ARMS)
    detection_attribution = result.sender_metrics.loc[
        result.sender_metrics["score_view"].eq(E3_DETECTION_ARM)
        & result.sender_metrics["metric"].eq("mean_max_attribution")
    ]
    assert detection_attribution["status"].eq("not_estimable").all()
