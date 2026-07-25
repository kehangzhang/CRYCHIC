from __future__ import annotations

import pandas as pd

from benchmarks.simulation.v7_dgp import generate_v7_dgp
from benchmarks.simulation.v7_occurrence_benchmark import (
    BETA_BINOMIAL_METHOD,
    DCST_METHOD,
    FISHER_METHOD,
    M4_ALIGNED_EFFECT_COLUMNS,
    M4_BASELINE_COLUMNS,
    M4_EFFECT_COLUMNS,
    M4_METHOD,
    M4_METHODS,
    M4_METRIC_COLUMNS,
    M4_SUBJECT_EVENT_COLUMNS,
    RAW_METHOD,
    run_v7_m4_occurrence_benchmark,
)
from crychic.workflow import run_subject_crossfit


def test_v7_m4_benchmark_preserves_unknown_truth_and_typed_comparators() -> None:
    fixture = generate_v7_dgp(
        dataset_id="v7-m4-contract",
        dgp_family="occurrence_heterogeneity",
        design_kind="independent_two_group",
        seed=12345,
        candidate_sender_count=2,
        cells_per_type=1,
        subjects_per_level=4,
    )
    crossfit = run_subject_crossfit(
        fixture.adata,
        fixture.config,
        fixture.resource,
        fixture.target_prior,
        spec=fixture.crossfit_spec,
        n_jobs=1,
    )

    result = run_v7_m4_occurrence_benchmark(
        crossfit,
        dataset_id=fixture.dataset_id,
        design=fixture.differential_design,
        sample_metadata=fixture.sample_metadata,
        truth=fixture.truth,
        dgp_family=fixture.dgp_family,
        design_kind=fixture.design_kind,
    )

    assert tuple(result.subject_events.columns) == M4_SUBJECT_EVENT_COLUMNS
    assert tuple(result.effects.columns) == M4_EFFECT_COLUMNS
    assert tuple(result.aligned_effects.columns) == M4_ALIGNED_EFFECT_COLUMNS
    assert tuple(result.baselines.columns) == M4_BASELINE_COLUMNS
    assert tuple(result.metrics.columns) == M4_METRIC_COLUMNS
    assert result.effects["event_id"].nunique() == 14
    assert len(result.effects) == 14
    assert result.aligned_effects["truth_known"].sum() == 7
    unknown = result.aligned_effects.loc[~result.aligned_effects["truth_known"]]
    assert unknown["truth_label"].isna().all()
    assert unknown["truth_occurrence_effect"].isna().all()
    assert result.subject_events["out_of_fold"].all()
    assert result.subject_events["truth_state_known"].sum() == 56

    assert set(result.baselines["method"]) == set(M4_METHODS)
    raw = result.baselines.loc[result.baselines["method"].eq(RAW_METHOD)]
    fisher = result.baselines.loc[result.baselines["method"].eq(FISHER_METHOD)]
    dcst = result.baselines.loc[result.baselines["method"].eq(DCST_METHOD)]
    beta = result.baselines.loc[result.baselines["method"].eq(BETA_BINOMIAL_METHOD)]
    assert raw["status"].eq("observed").all()
    assert fisher["status"].eq("observed").all()
    assert dcst["status"].eq("observed").all()
    assert beta["status"].eq("not_estimable").all()
    assert (
        beta["reason_code"]
        .eq("one_bernoulli_trial_per_subject_has_no_identifiable_overdispersion")
        .all()
    )
    assert not dcst["formal_inference_allowed"].any()
    assert fisher["p_value"].tolist() == dcst["p_value"].tolist()
    assert fisher["ranking_score"].tolist() == dcst["ranking_score"].tolist()

    metric_keys = set(
        zip(result.metrics["method"], result.metrics["metric"], strict=True)
    )
    assert (RAW_METHOD, "occurrence_auprc") in metric_keys
    assert (DCST_METHOD, "type1_error_alpha_0_05") in metric_keys
    assert (
        "crychic_m4_working_probability",
        "subject_brier_score",
    ) in metric_keys
    assert (
        "raw_leave_one_out_prevalence",
        "calibration_slope",
    ) in metric_keys
    manifest = result.to_manifest()
    assert manifest["dcst_scope"].startswith("protocol-compatible")
    assert manifest["occurrence"]["crossfit_id"] == crossfit.crossfit_id

    fold_fitted = run_v7_m4_occurrence_benchmark(
        crossfit,
        dataset_id=fixture.dataset_id,
        design=fixture.differential_design,
        sample_metadata=fixture.sample_metadata,
        truth=fixture.truth,
        dgp_family=fixture.dgp_family,
        design_kind=fixture.design_kind,
        occurrence_state_source="fold_fitted_parent_ecdf",
        active_probability_threshold=0.8,
    )
    comparator_methods = set(M4_METHODS).difference({M4_METHOD})
    comparator_columns = [
        "method",
        "event_id",
        "contrast_name",
        "prevalence_effect",
        "log_odds_ratio",
        "ranking_score",
        "p_value",
        "q_value",
        "status",
        "reason_code",
        "formal_inference_allowed",
    ]
    pd.testing.assert_frame_equal(
        result.baselines.loc[
            result.baselines["method"].isin(comparator_methods), comparator_columns
        ].reset_index(drop=True),
        fold_fitted.baselines.loc[
            fold_fitted.baselines["method"].isin(comparator_methods),
            comparator_columns,
        ].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        result.subject_events["raw_leave_one_out_prevalence"],
        fold_fitted.subject_events["raw_leave_one_out_prevalence"],
    )
    raw_brier = result.metrics.loc[
        result.metrics["method"].eq("crychic_m4_working_probability")
        & result.metrics["metric"].eq("subject_brier_score"),
        "value",
    ].iloc[0]
    fold_brier = fold_fitted.metrics.loc[
        fold_fitted.metrics["method"].eq("crychic_m4_working_probability")
        & fold_fitted.metrics["metric"].eq("subject_brier_score"),
        "value",
    ].iloc[0]
    assert fold_brier < raw_brier
    known_ids = set(
        fold_fitted.aligned_effects.loc[
            fold_fitted.aligned_effects["truth_known"], "event_id"
        ]
    )
    optimized = fold_fitted.baselines.loc[
        fold_fitted.baselines["method"].eq(M4_METHOD)
        & fold_fitted.baselines["event_id"].isin(known_ids)
    ]
    assert optimized["status"].isin({"observed", "descriptive"}).all()
    optimized_ap = fold_fitted.metrics.loc[
        fold_fitted.metrics["method"].eq(M4_METHOD)
        & fold_fitted.metrics["metric"].eq("occurrence_auprc")
    ].iloc[0]
    assert optimized_ap["status"] == "observed"


def test_v7_m4_benchmark_accepts_continuous_design_metadata() -> None:
    fixture = generate_v7_dgp(
        dataset_id="v7-m4-continuous",
        dgp_family="global_null",
        design_kind="continuous",
        seed=54321,
        candidate_sender_count=2,
        cells_per_type=1,
        subjects_per_level=4,
    )
    crossfit = run_subject_crossfit(
        fixture.adata,
        fixture.config,
        fixture.resource,
        fixture.target_prior,
        spec=fixture.crossfit_spec,
        n_jobs=1,
    )

    result = run_v7_m4_occurrence_benchmark(
        crossfit,
        dataset_id=fixture.dataset_id,
        design=fixture.differential_design,
        sample_metadata=fixture.sample_metadata,
        truth=fixture.truth,
        dgp_family=fixture.dgp_family,
        design_kind=fixture.design_kind,
        occurrence_state_source="fold_fitted_parent_ecdf",
    )

    assert set(result.effects["contrast_name"]) == {"slope:dose"}
    assert result.subject_events["condition"].isin({"A", "B"}).all()
    assert result.subject_events["raw_leave_one_out_prevalence"].isna().all()
    assert result.subject_events["truth_state_known"].any()
