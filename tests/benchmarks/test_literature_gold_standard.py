from __future__ import annotations

import math

import pandas as pd
import pytest
from benchmarks.metrics.literature_gold_standard import (
    LiteratureGoldStandardSpec,
    evaluate_literature_gold_standard,
)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["gold"] * 5,
            "edge": ["p1", "p2", "n1", "n2", "n3"],
            "is_positive": [1, 1, 0, 0, 0],
        }
    )


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["gold"] * 6,
            "method": ["perfect"] * 6,
            "edge": ["p1", "p2", "n1", "n2", "n3", "outside"],
            "score": [0.9, 0.8, 0.2, 0.1, None, 1.0],
            "status": [
                "observed",
                "observed",
                "observed",
                "observed",
                "filtered",
                "observed",
            ],
            "score_direction": ["higher"] * 6,
        }
    )


def _spec(**overrides: object) -> LiteratureGoldStandardSpec:
    values: dict[str, object] = {
        "key_columns": ("edge",),
        "method_columns": ("method",),
        "stratum_columns": ("dataset",),
        "top_k": 2,
        "n_bootstrap": 40,
        "n_negative_samples": 30,
        "negative_to_positive_ratio": 1.0,
        "random_seed": 11,
    }
    values.update(overrides)
    return LiteratureGoldStandardSpec(**values)  # type: ignore[arg-type]


def test_perfect_ranking_reports_classification_coverage_and_resampling() -> None:
    result = evaluate_literature_gold_standard(_scores(), _truth(), _spec())
    point = result.point_estimates.iloc[0]

    for metric in (
        "auroc",
        "auprc",
        "average_precision",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "mcc",
    ):
        assert point[metric] == pytest.approx(1.0)
    assert point["truth_universe_edges"] == 5
    assert point["matched_score_edges"] == 5
    assert point["evaluable_edges"] == 4
    assert point["truth_coverage_fraction"] == pytest.approx(0.8)
    assert point["positive_coverage_fraction"] == pytest.approx(1.0)
    assert point["negative_coverage_fraction"] == pytest.approx(2 / 3)
    assert point["n_unlabeled_score_rows"] == 1
    assert point["decision_rule"] == "top_k_tie_expanded"
    assert point["top_k_requested"] == 2
    assert point["status"] == "observed"

    bootstrap = result.bootstrap_summary
    negative = result.negative_sampling_summary
    assert set(bootstrap["metric"]) == {
        "auroc",
        "average_precision",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "mcc",
    }
    assert set(bootstrap["n_repeats_requested"]) == {40}
    assert set(negative["n_repeats_requested"]) == {30}
    assert set(negative["n_positive_per_repeat"]) == {2}
    assert set(negative["n_negative_per_repeat"]) == {2}
    assert set(negative["negative_to_positive_ratio_requested"]) == {1.0}
    assert set(negative["replacement_within_class"]) == {False}
    assert set(bootstrap["interval_semantics"]) == {
        "descriptive_benchmark_uncertainty_not_biological_inference"
    }
    assert set(bootstrap["status"]) == {"observed"}
    assert set(negative["status"]) == {"observed"}


def test_lower_is_better_orientation_and_tie_expanded_top_k() -> None:
    truth = pd.DataFrame(
        {"edge": ["p1", "p2", "n1", "n2"], "is_positive": [1, 1, 0, 0]}
    )
    scores = pd.DataFrame(
        {
            "method": ["lower"] * 4,
            "edge": ["p1", "p2", "n1", "n2"],
            "score": [0.1, 0.4, 0.2, 0.8],
            "status": ["observed"] * 4,
            "score_direction": ["lower"] * 4,
        }
    )
    result = evaluate_literature_gold_standard(
        scores,
        truth,
        _spec(stratum_columns=(), n_bootstrap=20, n_negative_samples=20),
    )
    point = result.point_estimates.iloc[0]

    assert point["auroc"] == pytest.approx(0.75)
    assert point["precision"] == pytest.approx(0.5)
    assert point["sensitivity"] == pytest.approx(0.5)
    assert point["specificity"] == pytest.approx(0.5)
    assert point["f1"] == pytest.approx(0.5)
    assert point["mcc"] == pytest.approx(0.0)
    assert point["true_positive"] == 1
    assert point["false_positive"] == 1
    assert point["true_negative"] == 1
    assert point["false_negative"] == 1


def test_multiple_methods_are_evaluated_independently() -> None:
    perfect = _scores()
    reverse = perfect.assign(method="reverse").copy()
    reverse["score"] = reverse["edge"].map(
        {"p1": 0.1, "p2": 0.2, "n1": 0.8, "n2": 0.9, "outside": 1.0}
    )
    result = evaluate_literature_gold_standard(
        pd.concat([perfect, reverse], ignore_index=True),
        _truth(),
        _spec(n_bootstrap=20, n_negative_samples=20),
    )
    point = result.point_estimates.set_index("method")

    assert point.loc["perfect", "auroc"] == pytest.approx(1.0)
    assert point.loc["reverse", "auroc"] == pytest.approx(0.0)
    assert point.loc["reverse", "sensitivity"] == pytest.approx(0.0)
    assert point.loc["reverse", "specificity"] == pytest.approx(0.0)
    seeds = result.bootstrap_summary.groupby("method")["derived_random_seed"].first()
    assert seeds.nunique() == 2


def test_single_class_after_missing_scores_is_explicitly_not_estimable() -> None:
    scores = _scores()
    scores.loc[scores["edge"].isin(["n1", "n2"]), "status"] = "filtered"
    scores.loc[scores["edge"].isin(["n1", "n2"]), "score"] = None
    result = evaluate_literature_gold_standard(scores, _truth(), _spec())
    point = result.point_estimates.iloc[0]

    assert point["status"] == "not_estimable"
    assert point["reason_code"] == "truth_has_single_class_after_coverage"
    assert math.isnan(float(point["auroc"]))
    assert math.isnan(float(point["specificity"]))
    assert set(result.bootstrap_summary["status"]) == {"not_estimable"}
    assert set(result.negative_sampling_summary["n_repeats_valid"]) == {0}


def test_explicit_oriented_threshold_and_resampling_are_order_invariant() -> None:
    spec = _spec(
        top_k=None,
        oriented_score_threshold=0.5,
        n_bootstrap=25,
        n_negative_samples=25,
    )
    first = evaluate_literature_gold_standard(_scores(), _truth(), spec)
    second = evaluate_literature_gold_standard(
        _scores().sample(frac=1, random_state=3).reset_index(drop=True),
        _truth().sample(frac=1, random_state=5).reset_index(drop=True),
        spec,
    )

    pd.testing.assert_frame_equal(first.point_estimates, second.point_estimates)
    pd.testing.assert_frame_equal(first.bootstrap_summary, second.bootstrap_summary)
    pd.testing.assert_frame_equal(
        first.negative_sampling_summary,
        second.negative_sampling_summary,
    )
    assert first.point_estimates.iloc[0]["decision_rule"] == (
        "oriented_score_threshold"
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda truth, scores: (
                pd.concat([truth, truth.iloc[[0]]], ignore_index=True),
                scores,
            ),
            "duplicate labeled keys",
        ),
        (
            lambda truth, scores: (
                truth.assign(is_positive=[1, 1, 0, 0, 0.5]),
                scores,
            ),
            "binary 0/1",
        ),
        (
            lambda truth, scores: (
                truth,
                scores.assign(
                    score=lambda table: table["score"].where(~table["edge"].eq("p1"))
                ),
            ),
            "eligible score rows require finite scores",
        ),
    ],
)
def test_invalid_truth_and_score_contracts_fail_closed(mutator, message: str) -> None:
    truth, scores = mutator(_truth(), _scores())
    with pytest.raises(ValueError, match=message):
        evaluate_literature_gold_standard(scores, truth, _spec())
