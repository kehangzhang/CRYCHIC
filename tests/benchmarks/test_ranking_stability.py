from __future__ import annotations

import pytest
from benchmarks.metrics import (
    Ranking,
    RankInterval,
    assign_stable_tiers,
    bootstrap_rank_intervals,
    rank_biased_overlap,
    top_k_stability_curve,
    weighted_kendall_tau,
)

UNIVERSE = ("a", "b", "c", "d")


def test_ranking_validates_frozen_universe_and_explicit_ne_state() -> None:
    with pytest.raises(ValueError, match="unique item IDs"):
        Ranking.observed(("a",), universe=("a", "a"))
    with pytest.raises(ValueError, match="ranked items must be unique"):
        Ranking.observed(("a", "a"), universe=UNIVERSE)
    with pytest.raises(ValueError, match="outside the frozen universe"):
        Ranking.observed(("unknown",), universe=UNIVERSE)
    with pytest.raises(ValueError, match="require a reason_code"):
        Ranking(items=(), universe=UNIVERSE, status="not_estimable")
    with pytest.raises(ValueError, match="must not contain ranked items"):
        Ranking(
            items=("a",),
            universe=UNIVERSE,
            status="not_estimable",
            reason_code="fold_failed",
        )


def test_ranking_reports_missing_items_in_frozen_order() -> None:
    ranking = Ranking.observed(("c", "a"), universe=UNIVERSE)

    assert ranking.items == ("c", "a")
    assert ranking.missing_items == ("b", "d")
    assert (
        Ranking.not_estimable(
            universe=UNIVERSE, reason_code="rank_deficient"
        ).missing_items
        == ()
    )


def test_rank_biased_overlap_is_top_weighted_and_hand_computable() -> None:
    left = Ranking.observed(("a", "b", "c"), universe=UNIVERSE)
    right = Ranking.observed(("a", "c", "b"), universe=UNIVERSE)

    result = rank_biased_overlap(left, right, persistence=0.5)

    # Prefix agreements are 1, 1/2, 1 with normalized weights 4/7, 2/7, 1/7.
    assert result.estimate == pytest.approx(6 / 7)
    assert result.status == "observed"
    assert result.n_left_missing == 1
    assert result.n_right_missing == 1
    assert result.n_shared_ranked == 3
    assert result.missing_policy == "unranked_not_imputed"


def test_rank_biased_overlap_bounds_and_partial_list_penalty() -> None:
    full = Ranking.observed(UNIVERSE, universe=UNIVERSE)
    identical = rank_biased_overlap(full, full)
    disjoint_left = Ranking.observed(("a", "b"), universe=UNIVERSE)
    disjoint_right = Ranking.observed(("c", "d"), universe=UNIVERSE)
    partial = Ranking.observed(("a",), universe=UNIVERSE)
    longer = Ranking.observed(("a", "b"), universe=UNIVERSE)

    assert identical.estimate == pytest.approx(1.0)
    assert rank_biased_overlap(disjoint_left, disjoint_right).estimate == 0.0
    partial_result = rank_biased_overlap(partial, longer, persistence=0.5)
    assert partial_result.estimate == pytest.approx(5 / 6)


def test_rank_biased_overlap_propagates_ne_without_numeric_fill() -> None:
    observed = Ranking.observed(("a",), universe=UNIVERSE)
    unavailable = Ranking.not_estimable(universe=UNIVERSE, reason_code="rank_deficient")

    result = rank_biased_overlap(observed, unavailable)

    assert result.estimate is None
    assert result.status == "not_estimable"
    assert result.reason_code == "right:rank_deficient"
    assert result.n_right_missing == 0


def test_rank_biased_overlap_rejects_invalid_persistence() -> None:
    ranking = Ranking.observed(("a",), universe=UNIVERSE)
    for persistence in (0.0, 1.0, -0.1, float("inf")):
        with pytest.raises(ValueError, match="persistence"):
            rank_biased_overlap(ranking, ranking, persistence=persistence)


def test_weighted_kendall_tau_has_expected_extremes() -> None:
    forward = Ranking.observed(UNIVERSE, universe=UNIVERSE)
    reverse = Ranking.observed(tuple(reversed(UNIVERSE)), universe=UNIVERSE)

    assert weighted_kendall_tau(forward, forward).estimate == pytest.approx(1.0)
    assert weighted_kendall_tau(forward, reverse).estimate == pytest.approx(-1.0)


def test_weighted_kendall_tau_is_symmetric() -> None:
    left = Ranking.observed(("a", "c", "b"), universe=UNIVERSE)
    right = Ranking.observed(("b", "a"), universe=UNIVERSE)

    left_right = weighted_kendall_tau(left, right)
    right_left = weighted_kendall_tau(right, left)

    assert left_right.estimate == pytest.approx(right_left.estimate)
    assert left_right.n_left_missing == right_left.n_right_missing
    assert left_right.n_right_missing == right_left.n_left_missing


def test_weighted_kendall_ties_missing_items_at_bottom() -> None:
    left = Ranking.observed(("a", "b"), universe=UNIVERSE)
    right = Ranking.observed(("a", "b", "c"), universe=UNIVERSE)

    result = weighted_kendall_tau(left, right)

    assert result.status == "observed"
    assert result.estimate is not None
    assert 0.0 < result.estimate < 1.0
    assert result.missing_policy == "tied_at_bottom"
    assert result.n_left_missing == 2
    assert result.n_right_missing == 1


def test_weighted_kendall_returns_ne_when_every_pair_is_tied() -> None:
    empty = Ranking.observed((), universe=UNIVERSE)

    result = weighted_kendall_tau(empty, empty)

    assert result.estimate is None
    assert result.status == "not_estimable"
    assert result.reason_code == "all_pairs_tied"


def test_rank_metrics_require_the_same_frozen_universe() -> None:
    left = Ranking.observed(("a",), universe=("a", "b"))
    right = Ranking.observed(("a",), universe=("a", "c"))

    with pytest.raises(ValueError, match="same ordered frozen-universe"):
        rank_biased_overlap(left, right)
    with pytest.raises(ValueError, match="same ordered frozen-universe"):
        weighted_kendall_tau(left, right)
    with pytest.raises(ValueError, match="same ordered frozen-universe"):
        top_k_stability_curve(left, right)


def test_rank_metrics_reject_reordered_frozen_universe_contract() -> None:
    left = Ranking.observed(("a", "b"), universe=UNIVERSE)
    right = Ranking.observed(("a", "b"), universe=tuple(reversed(UNIVERSE)))

    with pytest.raises(ValueError, match="ordered frozen-universe contract"):
        rank_biased_overlap(left, right)
    with pytest.raises(ValueError, match="ordered frozen-universe contract"):
        weighted_kendall_tau(left, right)
    with pytest.raises(ValueError, match="ordered frozen-universe contract"):
        top_k_stability_curve(left, right)


def test_weighted_kendall_rejects_invalid_weight_power() -> None:
    ranking = Ranking.observed(("a", "b"), universe=UNIVERSE)
    for weight_power in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="weight_power"):
            weighted_kendall_tau(ranking, ranking, weight_power=weight_power)


def test_top_k_curve_reports_each_cutoff_and_unavailable_tail() -> None:
    left = Ranking.observed(("a", "b", "c"), universe=UNIVERSE)
    right = Ranking.observed(("a", "c"), universe=UNIVERSE)

    curve = top_k_stability_curve(left, right)

    assert [point.k for point in curve] == [1, 2, 3, 4]
    assert curve[0].estimate == pytest.approx(1.0)
    assert curve[1].estimate == pytest.approx(1 / 3)
    assert curve[1].overlap_size == 1
    assert curve[1].union_size == 3
    assert curve[2].estimate is None
    assert curve[2].status == "not_estimable"
    assert curve[2].reason_code == "insufficient_ranked_items"
    assert curve[3].reason_code == "insufficient_ranked_items"


def test_top_k_curve_propagates_ne_to_every_requested_cutoff() -> None:
    left = Ranking.not_estimable(universe=UNIVERSE, reason_code="fit_failed")
    right = Ranking.observed(("a", "b"), universe=UNIVERSE)

    curve = top_k_stability_curve(left, right, max_k=2)

    assert len(curve) == 2
    assert all(point.estimate is None for point in curve)
    assert {point.reason_code for point in curve} == {"left:fit_failed"}


def test_top_k_curve_rejects_cutoffs_outside_the_universe() -> None:
    ranking = Ranking.observed(("a",), universe=UNIVERSE)
    for max_k in (0, 5):
        with pytest.raises(ValueError, match="max_k"):
            top_k_stability_curve(ranking, ranking, max_k=max_k)


def test_bootstrap_intervals_separate_rank_from_selection_frequency() -> None:
    rankings = (
        Ranking.observed(("a", "b", "c"), universe=UNIVERSE),
        Ranking.observed(("b", "a"), universe=UNIVERSE),
        Ranking.observed(("a", "c"), universe=UNIVERSE),
        Ranking.not_estimable(universe=UNIVERSE, reason_code="fit_failed"),
    )

    intervals = {
        interval.item_id: interval
        for interval in bootstrap_rank_intervals(
            rankings, confidence=0.5, min_observed_ranks=2
        )
    }

    assert intervals["a"].lower_rank == pytest.approx(1.0)
    assert intervals["a"].median_rank == pytest.approx(1.0)
    assert intervals["a"].upper_rank == pytest.approx(1.5)
    assert intervals["a"].selection_frequency == pytest.approx(1.0)
    assert intervals["a"].n_not_estimable == 1
    assert intervals["b"].selection_frequency == pytest.approx(2 / 3)
    assert intervals["b"].n_missing == 1
    assert intervals["c"].status == "observed"
    assert intervals["d"].status == "not_estimable"
    assert intervals["d"].reason_code == "never_ranked"
    assert intervals["d"].selection_frequency == 0.0


def test_bootstrap_intervals_report_all_ne_replicates() -> None:
    rankings = (
        Ranking.not_estimable(universe=UNIVERSE, reason_code="rank_deficient"),
        Ranking.not_estimable(universe=UNIVERSE, reason_code="fit_failed"),
    )

    intervals = bootstrap_rank_intervals(rankings)

    assert len(intervals) == len(UNIVERSE)
    assert all(interval.status == "not_estimable" for interval in intervals)
    assert {interval.reason_code for interval in intervals} == {
        "no_estimable_replicates"
    }
    assert all(interval.selection_frequency is None for interval in intervals)
    assert all(interval.n_not_estimable == 2 for interval in intervals)


def test_bootstrap_intervals_require_enough_observed_ranks() -> None:
    interval = bootstrap_rank_intervals(
        (Ranking.observed(("a",), universe=UNIVERSE),),
        min_observed_ranks=2,
    )[0]

    assert interval.status == "not_estimable"
    assert interval.reason_code == "too_few_observed_ranks"
    assert interval.selection_frequency == 1.0
    assert interval.lower_rank is None


def test_bootstrap_intervals_validate_arguments_and_universe() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_rank_intervals(())
    ranking = Ranking.observed(("a",), universe=UNIVERSE)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_rank_intervals((ranking,), confidence=1.0)
    with pytest.raises(ValueError, match="min_observed_ranks"):
        bootstrap_rank_intervals((ranking,), min_observed_ranks=0)
    other = Ranking.observed(("a",), universe=("a", "b"))
    with pytest.raises(ValueError, match="ordered frozen-universe"):
        bootstrap_rank_intervals((ranking, other))
    reordered = Ranking.observed(("a",), universe=tuple(reversed(UNIVERSE)))
    with pytest.raises(ValueError, match="ordered frozen-universe"):
        bootstrap_rank_intervals((ranking, reordered))


def _interval(
    item: str,
    *,
    lower: float | None,
    upper: float | None,
    frequency: float | None,
    status: str = "observed",
    reason_code: str | None = None,
) -> RankInterval:
    observed = status == "observed"
    n_observed = int(10 * frequency) if frequency is not None else 0
    return RankInterval(
        item_id=item,
        lower_rank=lower,
        median_rank=(lower + upper) / 2 if observed and lower and upper else None,
        upper_rank=upper,
        selection_frequency=frequency,
        n_replicates=10,
        n_estimable_rankings=10,
        n_observed_ranks=n_observed,
        n_missing=10 - n_observed,
        n_not_estimable=0,
        status="observed" if observed else "not_estimable",
        reason_code=reason_code,
    )


def test_stable_tiers_distinguish_supported_uncertain_and_ne_items() -> None:
    intervals = (
        _interval("top", lower=1.0, upper=2.0, frequency=1.0),
        _interval("crossing", lower=2.0, upper=4.0, frequency=1.0),
        _interval("below", lower=4.0, upper=6.0, frequency=1.0),
        _interval("sparse", lower=1.0, upper=2.0, frequency=0.5),
        _interval(
            "ne",
            lower=None,
            upper=None,
            frequency=0.0,
            status="not_estimable",
            reason_code="never_ranked",
        ),
    )

    tiers = {
        tier.item_id: tier
        for tier in assign_stable_tiers(
            intervals, top_k=3, minimum_selection_frequency=0.8
        )
    }

    assert tiers["top"].tier == "stable_top_k"
    assert tiers["crossing"].tier == "possible_top_k"
    assert tiers["crossing"].reason_code == "rank_interval_crosses_top_k"
    assert tiers["below"].tier == "stable_below_top_k"
    assert tiers["sparse"].tier == "unstable"
    assert tiers["sparse"].reason_code == "selection_frequency_below_threshold"
    assert tiers["ne"].tier == "not_estimable"
    assert tiers["ne"].status == "not_estimable"
    assert tiers["ne"].reason_code == "never_ranked"


def test_stable_tiers_validate_thresholds_and_observed_interval_contract() -> None:
    interval = _interval("a", lower=1.0, upper=2.0, frequency=1.0)
    with pytest.raises(ValueError, match="top_k"):
        assign_stable_tiers((interval,), top_k=0)
    with pytest.raises(ValueError, match="minimum_selection_frequency"):
        assign_stable_tiers((interval,), top_k=1, minimum_selection_frequency=1.1)
    with pytest.raises(ValueError, match="require all rank estimates"):
        _interval("a", lower=None, upper=None, frequency=1.0)


def test_rank_interval_rejects_inconsistent_public_contract() -> None:
    valid = {
        "item_id": "a",
        "lower_rank": 1.0,
        "median_rank": 2.0,
        "upper_rank": 3.0,
        "selection_frequency": 0.5,
        "n_replicates": 4,
        "n_estimable_rankings": 4,
        "n_observed_ranks": 2,
        "n_missing": 2,
        "n_not_estimable": 0,
        "status": "observed",
        "reason_code": None,
    }
    RankInterval(**valid)

    invalid_cases = (
        {**valid, "selection_frequency": 1.1},
        {**valid, "lower_rank": 4.0},
        {**valid, "n_missing": -1},
        {**valid, "reason_code": "unexpected"},
        {**valid, "upper_rank": None},
    )
    for values in invalid_cases:
        with pytest.raises(ValueError):
            RankInterval(**values)
