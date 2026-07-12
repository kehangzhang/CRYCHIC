"""Auditable, top-sensitive diagnostics for resampled rankings.

The metrics in this module require an explicit frozen universe.  An item that is
not present in an observed ranking is therefore known to be missing from that
ranking; it is never silently dropped from the comparison.  A whole ranking can
also be marked ``not_estimable`` and is then propagated without inventing a
numeric value.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

MetricStatus = Literal["observed", "not_estimable"]
AgreementMetric = Literal["rank_biased_overlap", "weighted_kendall_tau"]
StableTierLabel = Literal[
    "stable_top_k",
    "possible_top_k",
    "stable_below_top_k",
    "unstable",
    "not_estimable",
]


@dataclass(frozen=True, slots=True)
class Ranking:
    """One ordered, possibly partial ranking over a frozen universe.

    Use :meth:`observed` for a ranking that was produced, even when some
    universe members were not returned.  Use :meth:`not_estimable` when the
    resample or fold did not produce a valid ranking at all.
    """

    items: tuple[str, ...]
    universe: tuple[str, ...]
    status: MetricStatus = "observed"
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if not self.universe:
            raise ValueError("ranking universe must not be empty")
        if len(set(self.universe)) != len(self.universe):
            raise ValueError("ranking universe must contain unique item IDs")
        if any(not item for item in self.universe):
            raise ValueError("ranking universe item IDs must not be empty")
        if len(set(self.items)) != len(self.items):
            raise ValueError("ranked items must be unique")
        unknown = set(self.items).difference(self.universe)
        if unknown:
            raise ValueError(
                f"ranked items are outside the frozen universe: {sorted(unknown)}"
            )
        if self.status == "observed":
            if self.reason_code is not None:
                raise ValueError("observed rankings must not have a reason_code")
        elif self.status == "not_estimable":
            if self.items:
                raise ValueError("not-estimable rankings must not contain ranked items")
            if not self.reason_code:
                raise ValueError("not-estimable rankings require a reason_code")
        else:
            raise ValueError(f"unsupported ranking status: {self.status}")

    @classmethod
    def observed(cls, items: Sequence[str], *, universe: Sequence[str]) -> Ranking:
        """Construct an observed ranking without mutating caller sequences."""
        return cls(items=tuple(items), universe=tuple(universe))

    @classmethod
    def not_estimable(cls, *, universe: Sequence[str], reason_code: str) -> Ranking:
        """Construct an explicitly non-estimable ranking."""
        return cls(
            items=(),
            universe=tuple(universe),
            status="not_estimable",
            reason_code=reason_code,
        )

    @property
    def missing_items(self) -> tuple[str, ...]:
        """Return frozen-universe members absent from this observed ranking."""
        if self.status == "not_estimable":
            return ()
        ranked = set(self.items)
        return tuple(item for item in self.universe if item not in ranked)


@dataclass(frozen=True, slots=True)
class RankAgreement:
    """A rank-agreement estimate with coverage and missingness diagnostics."""

    metric: AgreementMetric
    estimate: float | None
    status: MetricStatus
    reason_code: str | None
    n_universe: int
    n_left_ranked: int
    n_right_ranked: int
    n_shared_ranked: int
    n_left_missing: int
    n_right_missing: int
    missing_policy: str


@dataclass(frozen=True, slots=True)
class TopKStabilityPoint:
    """Jaccard stability at one preregistered top-k cutoff."""

    k: int
    estimate: float | None
    overlap_size: int | None
    union_size: int | None
    status: MetricStatus
    reason_code: str | None
    n_left_ranked: int
    n_right_ranked: int


@dataclass(frozen=True, slots=True)
class RankInterval:
    """Conditional rank interval plus unconditional selection diagnostics."""

    item_id: str
    lower_rank: float | None
    median_rank: float | None
    upper_rank: float | None
    selection_frequency: float | None
    n_replicates: int
    n_estimable_rankings: int
    n_observed_ranks: int
    n_missing: int
    n_not_estimable: int
    status: MetricStatus
    reason_code: str | None

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("rank interval item_id must not be empty")
        counts = (
            self.n_replicates,
            self.n_estimable_rankings,
            self.n_observed_ranks,
            self.n_missing,
            self.n_not_estimable,
        )
        if self.n_replicates < 1 or any(count < 0 for count in counts[1:]):
            raise ValueError("rank interval counts must be non-negative")
        if self.n_replicates != self.n_estimable_rankings + self.n_not_estimable:
            raise ValueError(
                "n_replicates must equal estimable plus not-estimable rankings"
            )
        if self.n_estimable_rankings != self.n_observed_ranks + self.n_missing:
            raise ValueError(
                "estimable rankings must equal observed ranks plus missing ranks"
            )
        expected_frequency = (
            self.n_observed_ranks / self.n_estimable_rankings
            if self.n_estimable_rankings > 0
            else None
        )
        if expected_frequency is None:
            if self.selection_frequency is not None:
                raise ValueError(
                    "selection_frequency must be None without estimable rankings"
                )
        elif (
            self.selection_frequency is None
            or not math.isfinite(self.selection_frequency)
            or not math.isclose(
                self.selection_frequency, expected_frequency, abs_tol=1e-12
            )
        ):
            raise ValueError(
                "selection_frequency must equal observed/estimable rankings"
            )

        ranks = (self.lower_rank, self.median_rank, self.upper_rank)
        if self.status == "observed":
            if self.reason_code is not None:
                raise ValueError("observed rank intervals must not have a reason_code")
            if any(rank is None for rank in ranks):
                raise ValueError("observed rank intervals require all rank estimates")
            lower, median, upper = ranks
            assert lower is not None and median is not None and upper is not None
            if (
                not all(
                    math.isfinite(rank) and rank >= 1.0
                    for rank in (lower, median, upper)
                )
                or not lower <= median <= upper
            ):
                raise ValueError("rank estimates must be finite, positive, and ordered")
            if self.n_observed_ranks < 1:
                raise ValueError("observed rank intervals require observed ranks")
        elif self.status == "not_estimable":
            if not self.reason_code:
                raise ValueError("not-estimable rank intervals require a reason_code")
            if any(rank is not None for rank in ranks):
                raise ValueError(
                    "not-estimable rank intervals must not contain rank estimates"
                )
        else:
            raise ValueError(f"unsupported rank interval status: {self.status}")


@dataclass(frozen=True, slots=True)
class StableTier:
    """Stable-tier assignment derived from a rank interval."""

    item_id: str
    tier: StableTierLabel
    status: MetricStatus
    reason_code: str | None
    selection_frequency: float | None
    lower_rank: float | None
    upper_rank: float | None

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("stable tier item_id must not be empty")
        if self.selection_frequency is not None and (
            not math.isfinite(self.selection_frequency)
            or not 0.0 <= self.selection_frequency <= 1.0
        ):
            raise ValueError("stable tier selection_frequency must be between 0 and 1")
        ranks = (self.lower_rank, self.upper_rank)
        if self.status == "observed":
            if self.tier == "not_estimable":
                raise ValueError("observed stable tiers cannot be not_estimable")
            if self.selection_frequency is None or any(rank is None for rank in ranks):
                raise ValueError(
                    "observed stable tiers require frequency and rank estimates"
                )
            lower, upper = ranks
            assert lower is not None and upper is not None
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower < 1.0
                or lower > upper
            ):
                raise ValueError(
                    "stable tier ranks must be finite, positive, and ordered"
                )
            expected_reason = {
                "stable_top_k": None,
                "stable_below_top_k": None,
                "possible_top_k": "rank_interval_crosses_top_k",
                "unstable": "selection_frequency_below_threshold",
            }[self.tier]
            if self.reason_code != expected_reason:
                raise ValueError("stable tier reason_code is inconsistent with tier")
        elif self.status == "not_estimable":
            if self.tier != "not_estimable" or not self.reason_code:
                raise ValueError(
                    "not-estimable stable tiers require matching tier and reason_code"
                )
            if any(rank is not None for rank in ranks):
                raise ValueError(
                    "not-estimable stable tiers must not contain rank estimates"
                )
        else:
            raise ValueError(f"unsupported stable tier status: {self.status}")


def _validate_comparable(left: Ranking, right: Ranking) -> None:
    if left.universe != right.universe:
        raise ValueError("rankings must use the same ordered frozen-universe contract")


def _not_estimable_reason(left: Ranking, right: Ranking) -> str | None:
    reasons: list[str] = []
    if left.status == "not_estimable":
        reasons.append(f"left:{left.reason_code}")
    if right.status == "not_estimable":
        reasons.append(f"right:{right.reason_code}")
    return ";".join(reasons) or None


def _agreement(
    left: Ranking,
    right: Ranking,
    *,
    metric: AgreementMetric,
    estimate: float | None,
    reason_code: str | None,
    missing_policy: str,
) -> RankAgreement:
    status: MetricStatus = "observed" if estimate is not None else "not_estimable"
    return RankAgreement(
        metric=metric,
        estimate=estimate,
        status=status,
        reason_code=reason_code,
        n_universe=len(left.universe),
        n_left_ranked=len(left.items),
        n_right_ranked=len(right.items),
        n_shared_ranked=len(set(left.items).intersection(right.items)),
        n_left_missing=(
            len(left.universe) - len(left.items) if left.status == "observed" else 0
        ),
        n_right_missing=(
            len(right.universe) - len(right.items) if right.status == "observed" else 0
        ),
        missing_policy=missing_policy,
    )


def rank_biased_overlap(
    left: Ranking, right: Ranking, *, persistence: float = 0.9
) -> RankAgreement:
    """Return normalized finite rank-biased overlap.

    Prefix agreement at depth ``d`` is weighted by
    ``(1 - persistence) * persistence ** (d - 1)`` and normalized over the
    observed comparison depth.  Unequal list lengths are allowed and absent
    suffixes reduce agreement.  This is a descriptive finite-list estimand, not
    an extrapolation of an unobserved infinite ranking.
    """
    _validate_comparable(left, right)
    if not math.isfinite(persistence) or not 0.0 < persistence < 1.0:
        raise ValueError("persistence must be finite and strictly between 0 and 1")
    unavailable = _not_estimable_reason(left, right)
    if unavailable is not None:
        return _agreement(
            left,
            right,
            metric="rank_biased_overlap",
            estimate=None,
            reason_code=unavailable,
            missing_policy="unranked_not_imputed",
        )
    depth = max(len(left.items), len(right.items))
    if depth == 0:
        return _agreement(
            left,
            right,
            metric="rank_biased_overlap",
            estimate=None,
            reason_code="no_ranked_items",
            missing_policy="unranked_not_imputed",
        )

    weighted_agreement = 0.0
    left_prefix: set[str] = set()
    right_prefix: set[str] = set()
    for index in range(depth):
        if index < len(left.items):
            left_prefix.add(left.items[index])
        if index < len(right.items):
            right_prefix.add(right.items[index])
        prefix_agreement = len(left_prefix.intersection(right_prefix)) / (index + 1)
        weighted_agreement += (
            (1.0 - persistence) * persistence**index * prefix_agreement
        )
    normalizer = 1.0 - persistence**depth
    estimate = min(1.0, max(0.0, weighted_agreement / normalizer))
    return _agreement(
        left,
        right,
        metric="rank_biased_overlap",
        estimate=estimate,
        reason_code=None,
        missing_policy="unranked_not_imputed",
    )


def _sign(left: int, right: int) -> int:
    return (left > right) - (left < right)


def weighted_kendall_tau(
    left: Ranking, right: Ranking, *, weight_power: float = 1.0
) -> RankAgreement:
    """Return symmetric top-weighted Kendall tau-b over the frozen universe.

    Missing items are tied at rank ``n_universe + 1`` rather than dropped.
    Each item receives hyperbolic importance based on its best rank across the
    two lists, and a pair receives the sum of its item weights.  The tau-b
    denominator retains ties caused by missing items.
    """
    _validate_comparable(left, right)
    if not math.isfinite(weight_power) or weight_power <= 0.0:
        raise ValueError("weight_power must be finite and positive")
    unavailable = _not_estimable_reason(left, right)
    if unavailable is not None:
        return _agreement(
            left,
            right,
            metric="weighted_kendall_tau",
            estimate=None,
            reason_code=unavailable,
            missing_policy="tied_at_bottom",
        )
    if len(left.universe) < 2:
        return _agreement(
            left,
            right,
            metric="weighted_kendall_tau",
            estimate=None,
            reason_code="fewer_than_two_universe_items",
            missing_policy="tied_at_bottom",
        )

    bottom_rank = len(left.universe) + 1
    left_positions = {item: rank for rank, item in enumerate(left.items, start=1)}
    right_positions = {item: rank for rank, item in enumerate(right.items, start=1)}
    numerator = 0.0
    left_mass = 0.0
    right_mass = 0.0
    for first, second in itertools.combinations(left.universe, 2):
        left_first = left_positions.get(first, bottom_rank)
        left_second = left_positions.get(second, bottom_rank)
        right_first = right_positions.get(first, bottom_rank)
        right_second = right_positions.get(second, bottom_rank)
        left_sign = _sign(left_second, left_first)
        right_sign = _sign(right_second, right_first)
        first_importance = min(left_first, right_first) ** (-weight_power)
        second_importance = min(left_second, right_second) ** (-weight_power)
        pair_weight = first_importance + second_importance
        numerator += pair_weight * left_sign * right_sign
        left_mass += pair_weight * left_sign * left_sign
        right_mass += pair_weight * right_sign * right_sign

    denominator = math.sqrt(left_mass * right_mass)
    if denominator == 0.0:
        return _agreement(
            left,
            right,
            metric="weighted_kendall_tau",
            estimate=None,
            reason_code="all_pairs_tied",
            missing_policy="tied_at_bottom",
        )
    estimate = min(1.0, max(-1.0, numerator / denominator))
    return _agreement(
        left,
        right,
        metric="weighted_kendall_tau",
        estimate=estimate,
        reason_code=None,
        missing_policy="tied_at_bottom",
    )


def top_k_stability_curve(
    left: Ranking, right: Ranking, *, max_k: int | None = None
) -> tuple[TopKStabilityPoint, ...]:
    """Return top-k Jaccard stability for every cutoff through ``max_k``.

    A cutoff larger than either observed list is explicitly not estimable; the
    curve never fills missing entries with arbitrary bottom ranks.
    """
    _validate_comparable(left, right)
    limit = len(left.universe) if max_k is None else max_k
    if limit < 1 or limit > len(left.universe):
        raise ValueError("max_k must be between 1 and the frozen universe size")
    unavailable = _not_estimable_reason(left, right)
    points: list[TopKStabilityPoint] = []
    for k in range(1, limit + 1):
        reason_code = unavailable
        if reason_code is None and (len(left.items) < k or len(right.items) < k):
            reason_code = "insufficient_ranked_items"
        if reason_code is not None:
            points.append(
                TopKStabilityPoint(
                    k=k,
                    estimate=None,
                    overlap_size=None,
                    union_size=None,
                    status="not_estimable",
                    reason_code=reason_code,
                    n_left_ranked=len(left.items),
                    n_right_ranked=len(right.items),
                )
            )
            continue
        left_top = set(left.items[:k])
        right_top = set(right.items[:k])
        intersection = left_top.intersection(right_top)
        union = left_top.union(right_top)
        points.append(
            TopKStabilityPoint(
                k=k,
                estimate=len(intersection) / len(union),
                overlap_size=len(intersection),
                union_size=len(union),
                status="observed",
                reason_code=None,
                n_left_ranked=len(left.items),
                n_right_ranked=len(right.items),
            )
        )
    return tuple(points)


def _validate_resampled_rankings(rankings: Sequence[Ranking]) -> tuple[str, ...]:
    if not rankings:
        raise ValueError("at least one resampled ranking is required")
    universe = rankings[0].universe
    if any(ranking.universe != universe for ranking in rankings[1:]):
        raise ValueError(
            "all resampled rankings must use the same ordered frozen-universe contract"
        )
    return universe


def bootstrap_rank_intervals(
    rankings: Sequence[Ranking],
    *,
    confidence: float = 0.95,
    min_observed_ranks: int = 2,
) -> tuple[RankInterval, ...]:
    """Return percentile rank intervals across bootstrap/resampling replicates.

    Rank quantiles are conditional on an item being ranked.  Selection
    frequency and missing counts are reported separately so absence is not
    converted into an artificial last rank.  Entire non-estimable replicates
    are excluded from the selection-frequency denominator and counted.
    """
    universe = _validate_resampled_rankings(rankings)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    if min_observed_ranks < 1:
        raise ValueError("min_observed_ranks must be positive")

    estimable = [ranking for ranking in rankings if ranking.status == "observed"]
    n_estimable = len(estimable)
    n_not_estimable = len(rankings) - n_estimable
    rank_maps = [
        {item: rank for rank, item in enumerate(ranking.items, start=1)}
        for ranking in estimable
    ]
    alpha = (1.0 - confidence) / 2.0
    intervals: list[RankInterval] = []
    for item in universe:
        observed_ranks = [rank_map[item] for rank_map in rank_maps if item in rank_map]
        n_observed = len(observed_ranks)
        n_missing = n_estimable - n_observed
        selection_frequency = n_observed / n_estimable if n_estimable > 0 else None
        reason_code: str | None = None
        if n_estimable == 0:
            reason_code = "no_estimable_replicates"
        elif n_observed == 0:
            reason_code = "never_ranked"
        elif n_observed < min_observed_ranks:
            reason_code = "too_few_observed_ranks"

        lower_rank: float | None
        median_rank: float | None
        upper_rank: float | None
        if reason_code is None:
            values = np.asarray(observed_ranks, dtype=float)
            lower_rank = float(np.quantile(values, alpha))
            median_rank = float(np.quantile(values, 0.5))
            upper_rank = float(np.quantile(values, 1.0 - alpha))
            status: MetricStatus = "observed"
        else:
            lower_rank = median_rank = upper_rank = None
            status = "not_estimable"
        intervals.append(
            RankInterval(
                item_id=item,
                lower_rank=lower_rank,
                median_rank=median_rank,
                upper_rank=upper_rank,
                selection_frequency=selection_frequency,
                n_replicates=len(rankings),
                n_estimable_rankings=n_estimable,
                n_observed_ranks=n_observed,
                n_missing=n_missing,
                n_not_estimable=n_not_estimable,
                status=status,
                reason_code=reason_code,
            )
        )
    return tuple(intervals)


def assign_stable_tiers(
    intervals: Sequence[RankInterval],
    *,
    top_k: int,
    minimum_selection_frequency: float = 0.8,
) -> tuple[StableTier, ...]:
    """Classify interval-supported top-k membership without filling NE values."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if (
        not math.isfinite(minimum_selection_frequency)
        or not 0.0 <= minimum_selection_frequency <= 1.0
    ):
        raise ValueError("minimum_selection_frequency must be between 0 and 1")

    tiers: list[StableTier] = []
    for interval in intervals:
        if interval.status == "not_estimable":
            tier: StableTierLabel = "not_estimable"
            status: MetricStatus = "not_estimable"
            reason_code = interval.reason_code
        else:
            if (
                interval.selection_frequency is None
                or interval.lower_rank is None
                or interval.upper_rank is None
            ):
                raise ValueError("observed rank intervals require complete estimates")
            status = "observed"
            if interval.selection_frequency < minimum_selection_frequency:
                tier = "unstable"
                reason_code = "selection_frequency_below_threshold"
            elif interval.upper_rank <= top_k:
                tier = "stable_top_k"
                reason_code = None
            elif interval.lower_rank > top_k:
                tier = "stable_below_top_k"
                reason_code = None
            else:
                tier = "possible_top_k"
                reason_code = "rank_interval_crosses_top_k"
        tiers.append(
            StableTier(
                item_id=interval.item_id,
                tier=tier,
                status=status,
                reason_code=reason_code,
                selection_frequency=interval.selection_frequency,
                lower_rank=interval.lower_rank,
                upper_rank=interval.upper_rank,
            )
        )
    return tuple(tiers)


__all__ = [
    "MetricStatus",
    "RankAgreement",
    "RankInterval",
    "Ranking",
    "StableTier",
    "StableTierLabel",
    "TopKStabilityPoint",
    "assign_stable_tiers",
    "bootstrap_rank_intervals",
    "rank_biased_overlap",
    "top_k_stability_curve",
    "weighted_kendall_tau",
]
