"""Tie-aware, subject-resampled ranking stability for multi-condition CCC.

The implementation keeps aggregation members frozen, blocks technical
replicates within biological subjects, preserves effect ties, and separates
rank availability from actual tie-inclusive top-k membership.  It never uses
an item identifier to break a scientific tie.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata, weightedtau

from benchmarks.metrics.multicondition import (
    COMPARISON_ELIGIBLE_STATUSES,
    EDGE_KEYS,
    METHOD_IDENTITY_KEYS,
    validate_score_table,
)

RANK_STABILITY_SCHEMA_VERSION = "crychic-multicondition-rank-stability-v2"
RANKING_LEVELS = ("lr_family", "lr", "sender", "sender_receiver_pair")
RankLevel = Literal["lr_family", "lr", "sender", "sender_receiver_pair"]
Design = Literal["paired", "unpaired"]
RankScope = Literal["global_common_functional", "within_receiver_macro"]

RBO_PERSISTENCE = 0.9
WEIGHTED_KENDALL_POWER = 1.0
FAMILY_MAX_K = 25
LR_MAX_K = 100
SENDER_MAX_K = 25
SENDER_RECEIVER_MAX_K = 25
BOOTSTRAP_REPLICATES = 2000
SPLIT_REPEATS = 200
CONFIDENCE_LEVEL = 0.95
MINIMUM_TOP_K_FREQUENCY = 0.8
MINIMUM_ESTIMABLE_REPLICATE_FRACTION = 0.8
MINIMUM_SUBJECTS = 3
MINIMUM_OBSERVED_RANKS = 2
RANDOM_SEED = 20260712

_LEVEL_MAX_K: Mapping[RankLevel, int] = {
    "lr_family": FAMILY_MAX_K,
    "lr": LR_MAX_K,
    "sender": SENDER_MAX_K,
    "sender_receiver_pair": SENDER_RECEIVER_MAX_K,
}


@dataclass(frozen=True, slots=True)
class RankStabilityParameters:
    """Preregistered design, resampling, and stable-tier parameters."""

    n_bootstrap: int = BOOTSTRAP_REPLICATES
    n_split_repeats: int = SPLIT_REPEATS
    min_subjects: int = MINIMUM_SUBJECTS
    min_subjects_per_half: int = 2
    min_observed_ranks: int = MINIMUM_OBSERVED_RANKS
    minimum_estimable_replicate_fraction: float = MINIMUM_ESTIMABLE_REPLICATE_FRACTION
    confidence: float = CONFIDENCE_LEVEL
    minimum_top_k_frequency: float = MINIMUM_TOP_K_FREQUENCY
    rbo_persistence: float = RBO_PERSISTENCE
    weighted_kendall_power: float = WEIGHTED_KENDALL_POWER
    random_seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        integer_values = {
            "n_bootstrap": self.n_bootstrap,
            "n_split_repeats": self.n_split_repeats,
            "min_subjects": self.min_subjects,
            "min_subjects_per_half": self.min_subjects_per_half,
            "min_observed_ranks": self.min_observed_ranks,
            "random_seed": self.random_seed,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be positive")
        if self.n_split_repeats < 2:
            raise ValueError("n_split_repeats must be >= 2")
        if self.min_subjects < 3:
            raise ValueError("min_subjects must be >= 3")
        if self.min_subjects_per_half < 2:
            raise ValueError("min_subjects_per_half must be >= 2")
        if self.min_observed_ranks < 2:
            raise ValueError("min_observed_ranks must be >= 2")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        bounded = {
            "confidence": self.confidence,
            "minimum_top_k_frequency": self.minimum_top_k_frequency,
            "minimum_estimable_replicate_fraction": (
                self.minimum_estimable_replicate_fraction
            ),
        }
        for bounded_name, bounded_value in bounded.items():
            if not math.isfinite(bounded_value) or not 0.0 < bounded_value <= 1.0:
                raise ValueError(f"{bounded_name} must be finite in (0, 1]")
        if not 0.0 < self.rbo_persistence < 1.0:
            raise ValueError("rbo_persistence must be strictly between 0 and 1")
        if self.weighted_kendall_power <= 0.0:
            raise ValueError("weighted_kendall_power must be positive")


@dataclass(frozen=True, slots=True)
class RankStabilityTables:
    agreement: pd.DataFrame
    top_k_curve: pd.DataFrame
    rank_intervals: pd.DataFrame
    stable_tiers: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _SubjectEdgeData:
    values: np.ndarray
    row_index: tuple[tuple[str, str], ...]
    edge_metadata: pd.DataFrame
    reference_subjects: tuple[str, ...]
    target_subjects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LevelData:
    level: RankLevel
    universe: tuple[str, ...]
    item_metadata: pd.DataFrame
    receiver_groups: Mapping[str, np.ndarray]
    paired_effects: np.ndarray | None
    paired_subjects: tuple[str, ...]
    reference_values: np.ndarray | None
    reference_subjects: tuple[str, ...]
    target_values: np.ndarray | None
    target_subjects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SplitResults:
    left: np.ndarray
    right: np.ndarray
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class _BootstrapResults:
    effects: np.ndarray
    reason_code: str | None


def _level_keys(level: RankLevel, rank_scope: RankScope) -> tuple[str, ...]:
    if level == "lr_family":
        return ()
    if level == "lr":
        base = ("interaction_id", "ligand", "receptor")
        return ("receiver", *base) if rank_scope == "within_receiver_macro" else base
    if level == "sender":
        return (
            ("receiver", "sender")
            if rank_scope == "within_receiver_macro"
            else ("sender",)
        )
    return ("sender", "receiver")


def _json_id(kind: str, values: Mapping[str, object]) -> str:
    return json.dumps(
        {"kind": kind, **{key: str(value) for key, value in values.items()}},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _item_label(level: RankLevel, values: Mapping[str, object]) -> str:
    receiver = str(values.get("receiver", ""))
    receiver_prefix = f"{receiver} | " if receiver else ""
    if level == "lr":
        return f"{receiver_prefix}{values['ligand']} - {values['receptor']}"
    if level == "sender":
        return f"{receiver_prefix}{values['sender']}"
    return f"{values['sender']} -> {values['receiver']}"


def _identity_values(group: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in [*METHOD_IDENTITY_KEYS, "contrast"]:
        unique = group[column].drop_duplicates()
        if len(unique) != 1:
            raise ValueError(f"{column} must be constant within a ranking identity")
        result[column] = str(unique.iloc[0])
    return result


def _seed_for(identity: Mapping[str, str], level: RankLevel, seed: int) -> int:
    payload = json.dumps(
        {"identity": dict(identity), "level": level, "seed": seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _parameter_values(parameters: RankStabilityParameters) -> dict[str, object]:
    return {
        "rank_stability_schema_version": RANK_STABILITY_SCHEMA_VERSION,
        "n_bootstrap": parameters.n_bootstrap,
        "n_split_repeats": parameters.n_split_repeats,
        "minimum_subjects": parameters.min_subjects,
        "minimum_subjects_per_half": parameters.min_subjects_per_half,
        "minimum_observed_ranks": parameters.min_observed_ranks,
        "minimum_estimable_replicate_fraction": (
            parameters.minimum_estimable_replicate_fraction
        ),
        "rank_interval_quantile_level": parameters.confidence,
        "minimum_top_k_frequency": parameters.minimum_top_k_frequency,
        "rbo_persistence": parameters.rbo_persistence,
        "weighted_kendall_power": parameters.weighted_kendall_power,
        "random_seed": parameters.random_seed,
        "resampling_unit": "subject",
        "technical_replicate_policy": "mean_only_when_all_replicates_eligible",
        "aggregate_member_policy": "all_frozen_members_required",
        "missing_policy": "missing_propagated_not_imputed",
        "tie_policy": "average_rank_and_tie_inclusive_top_k",
    }


def _design_values(
    *,
    design: str,
    reference: str,
    target: str,
    rank_scope: str,
    reference_subjects: int,
    target_subjects: int,
    paired_subjects: int,
) -> dict[str, object]:
    return {
        "design": design,
        "reference": reference,
        "target": target,
        "rank_scope": rank_scope,
        "receiver_aggregation": (
            "macro_equal_receiver"
            if rank_scope == "within_receiver_macro"
            else "global"
        ),
        "n_reference_subjects": reference_subjects,
        "n_target_subjects": target_subjects,
        "n_paired_subjects": paired_subjects,
    }


def _not_estimable_for_design(
    identity: Mapping[str, object],
    *,
    reason_code: str,
    parameters: RankStabilityParameters,
    design_values: Mapping[str, object],
) -> RankStabilityTables:
    return not_estimable_rank_stability(
        identity,
        reason_code=reason_code,
        parameters=parameters,
        design=str(design_values["design"]),
        reference=str(design_values["reference"]),
        target=str(design_values["target"]),
        rank_scope=str(design_values["rank_scope"]),
        n_reference_subjects=cast(int, design_values["n_reference_subjects"]),
        n_target_subjects=cast(int, design_values["n_target_subjects"]),
        n_paired_subjects=cast(int, design_values["n_paired_subjects"]),
    )


def _within_receiver_strength(table: pd.DataFrame) -> pd.Series:
    eligible = table["status"].isin(COMPARISON_ELIGIBLE_STATUSES)
    values = table["comparison_strength"].where(eligible)
    frame = pd.DataFrame(
        {
            "sample_id": table["sample_id"].astype(str),
            "receiver": table["receiver"].astype(str),
            "value": values,
            "eligible": eligible,
        },
        index=table.index,
    )
    groups = frame.groupby(["sample_id", "receiver"], sort=False, observed=True)
    eligible_size = groups["eligible"].transform("sum")
    ranks = groups["value"].rank(ascending=False, method="average")
    strength = 1.0 - (ranks - 1.0) / eligible_size
    return strength.where(eligible)


def _subject_edge_data(
    group: pd.DataFrame,
    *,
    reference: str,
    target: str,
    rank_scope: RankScope,
) -> _SubjectEdgeData:
    working = group.loc[group["context"].astype(str).isin((reference, target))].copy()
    if rank_scope == "within_receiver_macro":
        working["_rank_strength"] = _within_receiver_strength(working)
    else:
        working["_rank_strength"] = working["comparison_strength"]
    edge_state = working.groupby(list(EDGE_KEYS), sort=False, observed=True)["status"]
    resource_edges = [
        edge
        for edge, statuses in edge_state
        if not statuses.eq("resource_unavailable").all()
    ]
    edge_metadata = pd.DataFrame(resource_edges, columns=EDGE_KEYS)
    if edge_metadata.empty:
        return _SubjectEdgeData(
            values=np.empty((0, 0), dtype=float),
            row_index=(),
            edge_metadata=edge_metadata,
            reference_subjects=(),
            target_subjects=(),
        )
    edge_metadata["edge_id"] = [
        _json_id("edge", dict(zip(EDGE_KEYS, edge, strict=True)))
        for edge in edge_metadata.loc[:, EDGE_KEYS].itertuples(index=False, name=None)
    ]
    edge_metadata = edge_metadata.sort_values(
        "edge_id", kind="stable", ignore_index=True
    )
    working = working.merge(
        edge_metadata[[*EDGE_KEYS, "edge_id"]],
        on=list(EDGE_KEYS),
        how="inner",
        validate="many_to_one",
    )
    working["_eligible"] = working["status"].isin(COMPARISON_ELIGIBLE_STATUSES)
    expected_samples = (
        working.groupby(["subject_id", "context"], sort=False, observed=True)[
            "sample_id"
        ]
        .nunique()
        .rename("expected_samples")
    )
    aggregate = (
        working.groupby(["subject_id", "context", "edge_id"], sort=False, observed=True)
        .agg(
            rows=("sample_id", "nunique"),
            eligible_rows=("_eligible", "sum"),
            value=("_rank_strength", "mean"),
        )
        .join(expected_samples, on=["subject_id", "context"])
    )
    complete = aggregate["rows"].eq(aggregate["expected_samples"]) & aggregate[
        "eligible_rows"
    ].eq(aggregate["expected_samples"])
    aggregate["value"] = aggregate["value"].where(complete)
    matrix = (
        aggregate["value"].unstack("edge_id").reindex(columns=edge_metadata["edge_id"])
    )
    row_index = tuple((str(subject), str(context)) for subject, context in matrix.index)
    values = matrix.to_numpy(dtype=float, copy=True)
    reference_subjects = tuple(
        sorted({subject for subject, context in row_index if context == reference})
    )
    target_subjects = tuple(
        sorted({subject for subject, context in row_index if context == target})
    )
    return _SubjectEdgeData(
        values=values,
        row_index=row_index,
        edge_metadata=edge_metadata,
        reference_subjects=reference_subjects,
        target_subjects=target_subjects,
    )


def _rows_for(
    edge_data: _SubjectEdgeData, subjects: Sequence[str], context: str
) -> np.ndarray:
    positions = {key: index for index, key in enumerate(edge_data.row_index)}
    rows = [positions.get((subject, context)) for subject in subjects]
    result = np.full((len(subjects), edge_data.values.shape[1]), np.nan, dtype=float)
    for output_index, source_index in enumerate(rows):
        if source_index is not None:
            result[output_index] = edge_data.values[source_index]
    return cast(np.ndarray, result)


def _level_data(
    edge_data: _SubjectEdgeData,
    *,
    level: RankLevel,
    rank_scope: RankScope,
    design: Design,
    reference: str,
    target: str,
) -> _LevelData:
    keys = _level_keys(level, rank_scope)
    edge_metadata = edge_data.edge_metadata
    records: list[dict[str, object]] = []
    member_indexes: list[np.ndarray] = []
    for raw_keys, members in edge_metadata.groupby(
        list(keys), sort=False, observed=True
    ):
        key_values = raw_keys if isinstance(raw_keys, tuple) else (raw_keys,)
        values = {key: str(value) for key, value in zip(keys, key_values, strict=True)}
        member_index = members.index.to_numpy(dtype=int)
        item_id = _json_id(level, values)
        records.append(
            {
                "item_id": item_id,
                "item_label": _item_label(level, values),
                "receiver_scope": (
                    str(values.get("receiver", "__global__"))
                    if rank_scope == "within_receiver_macro"
                    else "__global__"
                ),
                "frozen_member_count": len(member_index),
                **{f"item_{key}": value for key, value in values.items()},
            }
        )
        member_indexes.append(member_index)
    order = np.argsort([str(record["item_id"]) for record in records], kind="stable")
    records = [records[index] for index in order]
    member_indexes = [member_indexes[index] for index in order]
    item_values = np.full(
        (edge_data.values.shape[0], len(records)), np.nan, dtype=float
    )
    for item_index, edge_indexes in enumerate(member_indexes):
        block = edge_data.values[:, edge_indexes]
        observed_count = np.isfinite(block).sum(axis=1)
        complete = observed_count == len(edge_indexes)
        if complete.any():
            item_values[complete, item_index] = block[complete].mean(axis=1)
        records[item_index]["minimum_observed_member_count"] = int(
            observed_count.min(initial=len(edge_indexes))
        )
        records[item_index]["minimum_member_coverage_fraction"] = float(
            observed_count.min(initial=len(edge_indexes)) / len(edge_indexes)
        )
        records[item_index]["complete_subject_context_fraction"] = float(
            complete.mean() if len(complete) else 0.0
        )
    metadata = pd.DataFrame(records)
    universe = tuple(metadata["item_id"].astype(str))
    receiver_groups = {
        str(receiver): indexes.to_numpy(dtype=int)
        for receiver, indexes in metadata.groupby(
            "receiver_scope", sort=False, observed=True
        ).groups.items()
    }
    item_edge_data = _SubjectEdgeData(
        values=item_values,
        row_index=edge_data.row_index,
        edge_metadata=metadata,
        reference_subjects=edge_data.reference_subjects,
        target_subjects=edge_data.target_subjects,
    )
    if design == "paired":
        paired_subjects = tuple(
            sorted(
                set(edge_data.reference_subjects).intersection(
                    edge_data.target_subjects
                )
            )
        )
        reference_values = _rows_for(item_edge_data, paired_subjects, reference)
        target_values = _rows_for(item_edge_data, paired_subjects, target)
        return _LevelData(
            level=level,
            universe=universe,
            item_metadata=metadata,
            receiver_groups=receiver_groups,
            paired_effects=target_values - reference_values,
            paired_subjects=paired_subjects,
            reference_values=None,
            reference_subjects=(),
            target_values=None,
            target_subjects=(),
        )
    overlap = set(edge_data.reference_subjects).intersection(edge_data.target_subjects)
    if overlap:
        raise ValueError(
            "unpaired ranking stability requires disjoint subject groups; "
            f"overlap={sorted(overlap)}"
        )
    return _LevelData(
        level=level,
        universe=universe,
        item_metadata=metadata,
        receiver_groups=receiver_groups,
        paired_effects=None,
        paired_subjects=(),
        reference_values=_rows_for(
            item_edge_data, edge_data.reference_subjects, reference
        ),
        reference_subjects=edge_data.reference_subjects,
        target_values=_rows_for(item_edge_data, edge_data.target_subjects, target),
        target_subjects=edge_data.target_subjects,
    )


def _strict_mean(values: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    selected = values[indexes]
    complete = np.isfinite(selected).all(axis=0)
    result = np.full(values.shape[1], np.nan, dtype=float)
    if complete.any():
        result[complete] = selected[:, complete].mean(axis=0)
    return cast(np.ndarray, result)


def _split_effects(
    data: _LevelData,
    *,
    design: Design,
    parameters: RankStabilityParameters,
    rng: np.random.Generator,
) -> _SplitResults:
    repeats = parameters.n_split_repeats
    left: np.ndarray = np.full((repeats, len(data.universe)), np.nan, dtype=float)
    right = np.full_like(left, np.nan)
    minimum_total = 2 * parameters.min_subjects_per_half
    if design == "paired":
        values = cast(np.ndarray, data.paired_effects)
        n_subjects = len(data.paired_subjects)
        if n_subjects < max(parameters.min_subjects, minimum_total):
            return _SplitResults(
                left,
                right,
                "insufficient_paired_subjects_for_split_half",
            )
        left_size = n_subjects // 2
        for repeat in range(repeats):
            order = rng.permutation(n_subjects)
            left[repeat] = _strict_mean(values, order[:left_size])
            right[repeat] = _strict_mean(values, order[left_size:])
        return _SplitResults(left, right, None)
    reference = cast(np.ndarray, data.reference_values)
    target = cast(np.ndarray, data.target_values)
    n_reference = len(data.reference_subjects)
    n_target = len(data.target_subjects)
    if n_reference < max(parameters.min_subjects, minimum_total) or n_target < max(
        parameters.min_subjects, minimum_total
    ):
        return _SplitResults(
            left,
            right,
            "insufficient_independent_subjects_for_split_half",
        )
    reference_left = n_reference // 2
    target_left = n_target // 2
    for repeat in range(repeats):
        reference_order = rng.permutation(n_reference)
        target_order = rng.permutation(n_target)
        left_reference = _strict_mean(reference, reference_order[:reference_left])
        left_target = _strict_mean(target, target_order[:target_left])
        right_reference = _strict_mean(reference, reference_order[reference_left:])
        right_target = _strict_mean(target, target_order[target_left:])
        complete_left = np.isfinite(left_reference) & np.isfinite(left_target)
        complete_right = np.isfinite(right_reference) & np.isfinite(right_target)
        left[repeat, complete_left] = (
            left_target[complete_left] - left_reference[complete_left]
        )
        right[repeat, complete_right] = (
            right_target[complete_right] - right_reference[complete_right]
        )
    return _SplitResults(left, right, None)


def _bootstrap_weighted_mean(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    selected_missing = (counts > 0).astype(np.int16) @ (~finite).astype(np.int16)
    means = counts @ np.nan_to_num(values, nan=0.0) / counts.sum(axis=1)[:, None]
    means[selected_missing > 0] = np.nan
    return cast(np.ndarray, means)


def _bootstrap_effects(
    data: _LevelData,
    *,
    design: Design,
    parameters: RankStabilityParameters,
    rng: np.random.Generator,
) -> _BootstrapResults:
    replicates = parameters.n_bootstrap
    empty: np.ndarray = np.full((replicates, len(data.universe)), np.nan, dtype=float)
    if design == "paired":
        values = cast(np.ndarray, data.paired_effects)
        n_subjects = len(data.paired_subjects)
        if n_subjects < parameters.min_subjects:
            return _BootstrapResults(
                empty, "insufficient_paired_subjects_for_bootstrap"
            )
        counts = rng.multinomial(
            n_subjects,
            np.repeat(1.0 / n_subjects, n_subjects),
            size=replicates,
        )
        return _BootstrapResults(_bootstrap_weighted_mean(values, counts), None)
    reference = cast(np.ndarray, data.reference_values)
    target = cast(np.ndarray, data.target_values)
    n_reference = len(data.reference_subjects)
    n_target = len(data.target_subjects)
    if n_reference < parameters.min_subjects or n_target < parameters.min_subjects:
        return _BootstrapResults(
            empty, "insufficient_independent_subjects_for_bootstrap"
        )
    reference_counts = rng.multinomial(
        n_reference,
        np.repeat(1.0 / n_reference, n_reference),
        size=replicates,
    )
    target_counts = rng.multinomial(
        n_target,
        np.repeat(1.0 / n_target, n_target),
        size=replicates,
    )
    reference_mean = _bootstrap_weighted_mean(reference, reference_counts)
    target_mean = _bootstrap_weighted_mean(target, target_counts)
    complete = np.isfinite(reference_mean) & np.isfinite(target_mean)
    effects = np.full_like(reference_mean, np.nan)
    effects[complete] = target_mean[complete] - reference_mean[complete]
    return _BootstrapResults(effects, None)


def _all_tied(values: np.ndarray) -> bool:
    return len(values) < 2 or bool(
        np.allclose(values, values[0], atol=1e-12, rtol=0.0)
    )


def _tie_inclusive_top(
    values: np.ndarray, k: int
) -> tuple[np.ndarray, int, int] | None:
    finite = np.isfinite(values)
    finite_values = values[finite]
    if len(finite_values) < k or _all_tied(finite_values):
        return None
    threshold = float(np.partition(finite_values, len(finite_values) - k)[-k])
    selected = finite & (values >= threshold)
    boundary_size = int(
        np.isclose(values, threshold, atol=1e-12, rtol=0.0).sum()
    )
    return selected, int(selected.sum()), boundary_size


def _tie_aware_rbo(left: np.ndarray, right: np.ndarray, persistence: float) -> float:
    left_min = rankdata(-left, method="min").astype(int)
    right_min = rankdata(-right, method="min").astype(int)
    depth = len(left)
    left_counts = np.cumsum(np.bincount(left_min, minlength=depth + 1))[1:]
    right_counts = np.cumsum(np.bincount(right_min, minlength=depth + 1))[1:]
    overlap_counts = np.cumsum(
        np.bincount(np.maximum(left_min, right_min), minlength=depth + 1)
    )[1:]
    denominator = np.maximum(left_counts, right_counts)
    agreement = np.divide(
        overlap_counts,
        denominator,
        out=np.zeros(depth, dtype=float),
        where=denominator > 0,
    )
    weights = (1.0 - persistence) * persistence ** np.arange(depth)
    return float(np.dot(weights, agreement) / weights.sum())


def _weighted_tau(left: np.ndarray, right: np.ndarray, power: float) -> float:
    result = weightedtau(
        left,
        right,
        rank=True,
        weigher=lambda rank: float((rank + 1) ** (-power)),
        additive=True,
    ).statistic
    return float(result)


def _split_metric_values(
    split: _SplitResults,
    data: _LevelData,
    *,
    metric: str,
    parameters: RankStabilityParameters,
) -> tuple[list[float], list[int], Counter[str]]:
    values: list[float] = []
    receiver_counts: list[int] = []
    reasons: Counter[str] = Counter()
    if split.reason_code:
        reasons[split.reason_code] += parameters.n_split_repeats
        return values, receiver_counts, reasons
    for left_row, right_row in zip(split.left, split.right, strict=True):
        receiver_values: list[float] = []
        repeat_reasons: Counter[str] = Counter()
        for indexes in data.receiver_groups.values():
            left = left_row[indexes]
            right = right_row[indexes]
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                repeat_reasons["frozen_item_missing_in_split_repeat"] += 1
                continue
            if len(left) < 2:
                repeat_reasons["fewer_than_two_comparable_items"] += 1
                continue
            if _all_tied(left) or _all_tied(right):
                repeat_reasons["all_effects_tied"] += 1
                continue
            estimate = (
                _tie_aware_rbo(left, right, parameters.rbo_persistence)
                if metric == "rank_biased_overlap"
                else _weighted_tau(left, right, parameters.weighted_kendall_power)
            )
            if math.isfinite(estimate):
                receiver_values.append(estimate)
        if len(receiver_values) == len(data.receiver_groups):
            values.append(float(np.mean(receiver_values)))
            receiver_counts.append(len(receiver_values))
        else:
            reasons.update(
                repeat_reasons or {"incomplete_fixed_receiver_strata": 1}
            )
    return values, receiver_counts, reasons


def _envelope(values: Sequence[float], confidence: float) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.median(array)),
        float(np.quantile(array, alpha)),
        float(np.quantile(array, 1.0 - alpha)),
    )


def _base_values(
    *,
    identity: Mapping[str, str],
    data: _LevelData,
    parameters: RankStabilityParameters,
    design_values: Mapping[str, object],
) -> dict[str, object]:
    return {
        **identity,
        **_parameter_values(parameters),
        **design_values,
        "ranking_level": data.level,
        "ranking_universe_size": len(data.universe),
        "ranking_estimand": "target_minus_reference_subject_rank_strength",
        "aggregation": "all_frozen_members_then_equal_receiver_macro",
        "n_receiver_strata": len(data.receiver_groups),
    }


def _agreement_tables(
    split: _SplitResults,
    data: _LevelData,
    *,
    identity: Mapping[str, str],
    parameters: RankStabilityParameters,
    design_values: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _base_values(
        identity=identity,
        data=data,
        parameters=parameters,
        design_values=design_values,
    )
    agreement_rows: list[dict[str, object]] = []
    for metric in ("rank_biased_overlap", "weighted_kendall_tau"):
        values, receiver_counts, reasons = _split_metric_values(
            split, data, metric=metric, parameters=parameters
        )
        fraction = len(values) / parameters.n_split_repeats
        if values and fraction >= parameters.minimum_estimable_replicate_fraction:
            estimate, lower, upper = _envelope(values, parameters.confidence)
            status = "observed"
            reason_code = None
        else:
            estimate = lower = upper = math.nan
            status = "not_estimable"
            if reasons and set(reasons) == {"all_effects_tied"}:
                reason_code = "all_effects_tied"
            elif split.reason_code:
                reason_code = split.reason_code
            else:
                reason_code = "insufficient_estimable_split_repeat_fraction"
        agreement_rows.append(
            base
            | {
                "metric": metric,
                "estimate": estimate,
                "envelope_lower": lower,
                "envelope_upper": upper,
                "interval_type": "split_repeat_quantile_envelope",
                "n_repeats_requested": parameters.n_split_repeats,
                "n_repeats_estimable": len(values),
                "estimable_repeat_fraction": fraction,
                "median_receivers_estimable": (
                    float(np.median(receiver_counts)) if receiver_counts else math.nan
                ),
                "status": status,
                "reason_code": reason_code,
            }
        )

    curve_rows: list[dict[str, object]] = []
    max_k = _LEVEL_MAX_K[data.level]
    for k in range(1, max_k + 1):
        estimates: list[float] = []
        realized_left: list[float] = []
        realized_right: list[float] = []
        boundary_left: list[float] = []
        boundary_right: list[float] = []
        boundary_tied: list[float] = []
        if split.reason_code is None:
            for left_row, right_row in zip(split.left, split.right, strict=True):
                receiver_estimates: list[float] = []
                receiver_realized_left: list[int] = []
                receiver_realized_right: list[int] = []
                receiver_boundary_left: list[int] = []
                receiver_boundary_right: list[int] = []
                for indexes in data.receiver_groups.values():
                    left = left_row[indexes]
                    right = right_row[indexes]
                    if not np.isfinite(left).all() or not np.isfinite(right).all():
                        continue
                    if len(left) < k or _all_tied(left) or _all_tied(right):
                        continue
                    left_top = _tie_inclusive_top(left, k)
                    right_top = _tie_inclusive_top(right, k)
                    if left_top is None or right_top is None:
                        continue
                    left_mask, left_realized, left_boundary = left_top
                    right_mask, right_realized, right_boundary = right_top
                    union = left_mask | right_mask
                    if not union.any():
                        continue
                    receiver_estimates.append(
                        float((left_mask & right_mask).sum() / union.sum())
                    )
                    receiver_realized_left.append(left_realized)
                    receiver_realized_right.append(right_realized)
                    receiver_boundary_left.append(left_boundary)
                    receiver_boundary_right.append(right_boundary)
                if len(receiver_estimates) == len(data.receiver_groups):
                    estimates.append(float(np.mean(receiver_estimates)))
                    realized_left.append(float(np.mean(receiver_realized_left)))
                    realized_right.append(float(np.mean(receiver_realized_right)))
                    boundary_left.append(float(np.mean(receiver_boundary_left)))
                    boundary_right.append(float(np.mean(receiver_boundary_right)))
                    boundary_tied.append(
                        float(
                            any(value > 1 for value in receiver_boundary_left)
                            or any(value > 1 for value in receiver_boundary_right)
                        )
                    )
        fraction = len(estimates) / parameters.n_split_repeats
        if estimates and fraction >= parameters.minimum_estimable_replicate_fraction:
            estimate, lower, upper = _envelope(estimates, parameters.confidence)
            status = "observed"
            reason_code = None
        else:
            estimate = lower = upper = math.nan
            status = "not_estimable"
            reason_code = split.reason_code or (
                "all_effects_tied"
                if k == 1 and not estimates
                else "insufficient_estimable_split_repeat_fraction"
            )
        curve_rows.append(
            base
            | {
                "metric": "tie_inclusive_top_k_jaccard",
                "k": k,
                "estimate": estimate,
                "envelope_lower": lower,
                "envelope_upper": upper,
                "interval_type": "split_repeat_quantile_envelope",
                "n_repeats_requested": parameters.n_split_repeats,
                "n_repeats_estimable": len(estimates),
                "estimable_repeat_fraction": fraction,
                "median_realized_k_left": (
                    float(np.median(realized_left)) if realized_left else math.nan
                ),
                "median_realized_k_right": (
                    float(np.median(realized_right)) if realized_right else math.nan
                ),
                "median_boundary_tie_size_left": (
                    float(np.median(boundary_left)) if boundary_left else math.nan
                ),
                "median_boundary_tie_size_right": (
                    float(np.median(boundary_right)) if boundary_right else math.nan
                ),
                "boundary_tie_repeat_fraction": (
                    float(np.mean(boundary_tied)) if boundary_tied else math.nan
                ),
                "status": status,
                "reason_code": reason_code,
            }
        )
    return pd.DataFrame(agreement_rows), pd.DataFrame(curve_rows)


def _bootstrap_item_tables(
    bootstrap: _BootstrapResults,
    data: _LevelData,
    *,
    identity: Mapping[str, str],
    parameters: RankStabilityParameters,
    design_values: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _base_values(
        identity=identity,
        data=data,
        parameters=parameters,
        design_values=design_values,
    )
    ranks = np.full_like(bootstrap.effects, np.nan)
    top_membership = np.zeros_like(bootstrap.effects, dtype=bool)
    replicate_estimable: np.ndarray = np.zeros(parameters.n_bootstrap, dtype=bool)
    reasons: Counter[str] = Counter()
    for replicate, effects in enumerate(bootstrap.effects):
        replicate_ranks = np.full(effects.shape, np.nan, dtype=float)
        replicate_top_membership = np.zeros(effects.shape, dtype=bool)
        receiver_estimable = True
        for indexes in data.receiver_groups.values():
            values = effects[indexes]
            if not np.isfinite(values).all():
                reasons["frozen_item_missing_in_bootstrap_replicate"] += 1
                receiver_estimable = False
                break
            if len(values) < 2:
                reasons["fewer_than_two_comparable_items"] += 1
                receiver_estimable = False
                break
            if _all_tied(values):
                reasons["all_effects_tied"] += 1
                receiver_estimable = False
                break
            replicate_ranks[indexes] = rankdata(-values, method="average")
            receiver_top_k = min(_LEVEL_MAX_K[data.level], len(indexes))
            top = _tie_inclusive_top(values, receiver_top_k)
            if top is not None:
                replicate_top_membership[indexes[top[0]]] = True
        replicate_estimable[replicate] = receiver_estimable
        if receiver_estimable:
            ranks[replicate] = replicate_ranks
            top_membership[replicate] = replicate_top_membership
    estimable_count = int(replicate_estimable.sum())
    estimable_fraction = estimable_count / parameters.n_bootstrap
    endpoint_estimable = (
        bootstrap.reason_code is None
        and estimable_fraction >= parameters.minimum_estimable_replicate_fraction
    )
    if bootstrap.reason_code:
        endpoint_reason = bootstrap.reason_code
    elif reasons and set(reasons) == {"all_effects_tied"}:
        endpoint_reason = "all_effects_tied"
    elif not endpoint_estimable:
        endpoint_reason = "insufficient_estimable_bootstrap_fraction"
    else:
        endpoint_reason = None
    alpha = (1.0 - parameters.confidence) / 2.0
    interval_rows: list[dict[str, object]] = []
    tier_rows: list[dict[str, object]] = []
    for item_index, item in data.item_metadata.iterrows():
        observed_ranks = ranks[:, item_index]
        available = np.isfinite(observed_ranks)
        n_available = int(available.sum())
        rank_availability_frequency = n_available / parameters.n_bootstrap
        top_k_count = int(top_membership[:, item_index].sum())
        top_k_frequency = top_k_count / parameters.n_bootstrap
        status = "observed"
        reason_code: str | None = None
        if not endpoint_estimable:
            status = "not_estimable"
            reason_code = endpoint_reason
        elif n_available < parameters.min_observed_ranks:
            status = "not_estimable"
            reason_code = "too_few_observed_ranks"
        if status == "observed":
            values = observed_ranks[available]
            lower = float(np.quantile(values, alpha))
            median = float(np.median(values))
            upper = float(np.quantile(values, 1.0 - alpha))
            nominal_top_k = min(
                _LEVEL_MAX_K[data.level],
                len(data.receiver_groups[str(item["receiver_scope"])]),
            )
            if (
                top_k_frequency >= parameters.minimum_top_k_frequency
                and upper <= nominal_top_k
            ):
                tier = "stable_top_k"
                tier_reason = None
            elif top_k_frequency == 0.0 and lower > nominal_top_k:
                tier = "stable_below_top_k"
                tier_reason = None
            elif top_k_frequency > 0.0:
                tier = "possible_top_k"
                tier_reason = "top_k_frequency_or_rank_interval_uncertain"
            else:
                tier = "unstable"
                tier_reason = "rank_availability_or_top_k_frequency_low"
        else:
            lower = median = upper = math.nan
            tier = "not_estimable"
            tier_reason = reason_code
        item_values = {
            key: value for key, value in item.to_dict().items() if key != "item_id"
        }
        common: dict[str, object] = (
            base
            | cast(dict[str, object], item_values)
            | {
                "item_id": str(item["item_id"]),
                "lower_rank": lower,
                "median_rank": median,
                "upper_rank": upper,
                "rank_availability_frequency": rank_availability_frequency,
                "top_k_frequency": top_k_frequency,
                "n_replicates_requested": parameters.n_bootstrap,
                "n_replicates_estimable": estimable_count,
                "estimable_replicate_fraction": estimable_fraction,
                "n_rank_available": n_available,
                "n_top_k": top_k_count,
                "n_not_estimable_replicates": parameters.n_bootstrap - estimable_count,
                "status": status,
                "reason_code": reason_code,
            }
        )
        interval_rows.append(common)
        tier_rows.append(
            common
            | {
                "tier": tier,
                "tier_reason_code": tier_reason,
            }
        )
    return pd.DataFrame(interval_rows), pd.DataFrame(tier_rows)


def not_estimable_rank_stability(
    identity: Mapping[str, object],
    *,
    reason_code: str,
    parameters: RankStabilityParameters | None = None,
    design: str = "unsupported",
    reference: str = "not_available",
    target: str = "not_available",
    rank_scope: str = "not_available",
    n_reference_subjects: int = 0,
    n_target_subjects: int = 0,
    n_paired_subjects: int = 0,
) -> RankStabilityTables:
    params = parameters or RankStabilityParameters()
    base_identity = {key: str(value) for key, value in identity.items()}
    base = {
        **base_identity,
        **_parameter_values(params),
        **_design_values(
            design=design,
            reference=reference,
            target=target,
            rank_scope=rank_scope,
            reference_subjects=n_reference_subjects,
            target_subjects=n_target_subjects,
            paired_subjects=n_paired_subjects,
        ),
        "ranking_universe_size": 0,
        "ranking_estimand": "target_minus_reference_subject_rank_strength",
        "aggregation": "all_frozen_members_then_equal_receiver_macro",
        "n_receiver_strata": 0,
    }
    agreement_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    tier_rows: list[dict[str, object]] = []
    for level in cast(tuple[RankLevel, ...], RANKING_LEVELS):
        level_base = base | {"ranking_level": level}
        for metric in ("rank_biased_overlap", "weighted_kendall_tau"):
            agreement_rows.append(
                level_base
                | {
                    "metric": metric,
                    "estimate": math.nan,
                    "envelope_lower": math.nan,
                    "envelope_upper": math.nan,
                    "interval_type": "split_repeat_quantile_envelope",
                    "n_repeats_requested": params.n_split_repeats,
                    "n_repeats_estimable": 0,
                    "estimable_repeat_fraction": 0.0,
                    "median_receivers_estimable": math.nan,
                    "status": "not_estimable",
                    "reason_code": reason_code,
                }
            )
        for k in range(1, _LEVEL_MAX_K[level] + 1):
            curve_rows.append(
                level_base
                | {
                    "metric": "tie_inclusive_top_k_jaccard",
                    "k": k,
                    "estimate": math.nan,
                    "envelope_lower": math.nan,
                    "envelope_upper": math.nan,
                    "interval_type": "split_repeat_quantile_envelope",
                    "n_repeats_requested": params.n_split_repeats,
                    "n_repeats_estimable": 0,
                    "estimable_repeat_fraction": 0.0,
                    "median_realized_k_left": math.nan,
                    "median_realized_k_right": math.nan,
                    "median_boundary_tie_size_left": math.nan,
                    "median_boundary_tie_size_right": math.nan,
                    "boundary_tie_repeat_fraction": math.nan,
                    "status": "not_estimable",
                    "reason_code": reason_code,
                }
            )
        item = level_base | {
            "item_id": "__not_estimable__",
            "item_label": "NE",
            "receiver_scope": "not_available",
            "frozen_member_count": 0,
            "minimum_observed_member_count": 0,
            "minimum_member_coverage_fraction": math.nan,
            "complete_subject_context_fraction": 0.0,
            "lower_rank": math.nan,
            "median_rank": math.nan,
            "upper_rank": math.nan,
            "rank_availability_frequency": 0.0,
            "top_k_frequency": 0.0,
            "n_replicates_requested": params.n_bootstrap,
            "n_replicates_estimable": 0,
            "estimable_replicate_fraction": 0.0,
            "n_rank_available": 0,
            "n_top_k": 0,
            "n_not_estimable_replicates": params.n_bootstrap,
            "status": "not_estimable",
            "reason_code": reason_code,
        }
        interval_rows.append(item)
        tier_rows.append(
            item | {"tier": "not_estimable", "tier_reason_code": reason_code}
        )
    return RankStabilityTables(
        agreement=pd.DataFrame(agreement_rows),
        top_k_curve=pd.DataFrame(curve_rows),
        rank_intervals=pd.DataFrame(interval_rows),
        stable_tiers=pd.DataFrame(tier_rows),
    )


def evaluate_multicondition_rank_stability(
    table: pd.DataFrame,
    *,
    reference: str,
    target: str,
    design: Design,
    rank_scope: RankScope = "global_common_functional",
    parameters: RankStabilityParameters | None = None,
    validated: bool = False,
) -> RankStabilityTables:
    """Evaluate tie-aware ranking stability for one or more score identities."""
    if reference == target:
        raise ValueError("reference and target contexts must differ")
    if design not in {"paired", "unpaired"}:
        raise ValueError("design must be paired or unpaired")
    if rank_scope not in {"global_common_functional", "within_receiver_macro"}:
        raise ValueError("unsupported rank_scope")
    params = parameters or RankStabilityParameters()
    scores = table.copy(deep=False) if validated else validate_score_table(table)
    required = {"comparison_strength", "comparison_eligible"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(
            f"validated score table is missing derived columns: {sorted(missing)}"
        )
    contexts = set(scores["context"].astype(str))
    absent = {reference, target}.difference(contexts)
    if absent:
        raise ValueError(f"score table is missing contexts: {sorted(absent)}")

    agreement_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    tier_frames: list[pd.DataFrame] = []
    grouping = [*METHOD_IDENTITY_KEYS, "contrast"]
    for _, group in scores.groupby(grouping, sort=False, observed=True):
        identity = _identity_values(group)
        edge_data = _subject_edge_data(
            group,
            reference=reference,
            target=target,
            rank_scope=rank_scope,
        )
        paired_count = len(
            set(edge_data.reference_subjects).intersection(edge_data.target_subjects)
        )
        design_values = _design_values(
            design=design,
            reference=reference,
            target=target,
            rank_scope=rank_scope,
            reference_subjects=len(edge_data.reference_subjects),
            target_subjects=len(edge_data.target_subjects),
            paired_subjects=paired_count,
        )
        for level in cast(tuple[RankLevel, ...], RANKING_LEVELS):
            if level == "lr_family":
                ne = _not_estimable_for_design(
                    identity,
                    reason_code="lr_family_mapping_not_available_in_score_contract",
                    parameters=params,
                    design_values=design_values,
                )
                agreement_frames.append(
                    ne.agreement.loc[ne.agreement["ranking_level"].eq(level)]
                )
                curve_frames.append(
                    ne.top_k_curve.loc[ne.top_k_curve["ranking_level"].eq(level)]
                )
                interval_frames.append(
                    ne.rank_intervals.loc[ne.rank_intervals["ranking_level"].eq(level)]
                )
                tier_frames.append(
                    ne.stable_tiers.loc[ne.stable_tiers["ranking_level"].eq(level)]
                )
                continue
            data = _level_data(
                edge_data,
                level=level,
                rank_scope=rank_scope,
                design=design,
                reference=reference,
                target=target,
            )
            if not data.universe:
                ne = _not_estimable_for_design(
                    identity,
                    reason_code="no_resource_eligible_items",
                    parameters=params,
                    design_values=design_values,
                )
                agreement_frames.append(
                    ne.agreement.loc[ne.agreement["ranking_level"].eq(level)]
                )
                curve_frames.append(
                    ne.top_k_curve.loc[ne.top_k_curve["ranking_level"].eq(level)]
                )
                interval_frames.append(
                    ne.rank_intervals.loc[ne.rank_intervals["ranking_level"].eq(level)]
                )
                tier_frames.append(
                    ne.stable_tiers.loc[ne.stable_tiers["ranking_level"].eq(level)]
                )
                continue
            rng = np.random.default_rng(_seed_for(identity, level, params.random_seed))
            split = _split_effects(data, design=design, parameters=params, rng=rng)
            bootstrap = _bootstrap_effects(
                data, design=design, parameters=params, rng=rng
            )
            agreement, curve = _agreement_tables(
                split,
                data,
                identity=identity,
                parameters=params,
                design_values=design_values,
            )
            intervals, tiers = _bootstrap_item_tables(
                bootstrap,
                data,
                identity=identity,
                parameters=params,
                design_values=design_values,
            )
            agreement_frames.append(agreement)
            curve_frames.append(curve)
            interval_frames.append(intervals)
            tier_frames.append(tiers)
    if not agreement_frames:
        raise ValueError("score table contains no ranking identities")
    return RankStabilityTables(
        agreement=pd.concat(agreement_frames, ignore_index=True, sort=False),
        top_k_curve=pd.concat(curve_frames, ignore_index=True, sort=False),
        rank_intervals=pd.concat(interval_frames, ignore_index=True, sort=False),
        stable_tiers=pd.concat(tier_frames, ignore_index=True, sort=False),
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CONFIDENCE_LEVEL",
    "FAMILY_MAX_K",
    "LR_MAX_K",
    "MINIMUM_ESTIMABLE_REPLICATE_FRACTION",
    "MINIMUM_OBSERVED_RANKS",
    "MINIMUM_SUBJECTS",
    "MINIMUM_TOP_K_FREQUENCY",
    "RANDOM_SEED",
    "RANKING_LEVELS",
    "RANK_STABILITY_SCHEMA_VERSION",
    "RBO_PERSISTENCE",
    "SENDER_MAX_K",
    "SENDER_RECEIVER_MAX_K",
    "SPLIT_REPEATS",
    "WEIGHTED_KENDALL_POWER",
    "RankStabilityParameters",
    "RankStabilityTables",
    "evaluate_multicondition_rank_stability",
    "not_estimable_rank_stability",
]
