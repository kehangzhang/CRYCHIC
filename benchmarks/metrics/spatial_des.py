"""Tie-safe spatial distance enrichment scores for differential CCC benchmarks.

The metric compares a condition-specific ranked list of *ordered* sender to
receiver cell-type pairs with a condition-specific expected spatial set.  It
uses the unweighted GSEA running sum: hits add ``1 / n_hits`` and misses
subtract ``1 / n_misses``.  ``score_type="pos"`` returns the largest positive
excursion and is the pure-Python analogue of fgsea with ``gseaParam=0`` and
``scoreType="pos"``.  ``score_type="abs"`` returns the largest absolute
excursion; this is an explicitly labelled diagnostic and is not a literal
fgsea ``scoreType`` (fgsea calls its signed two-sided mode ``"std"``).

Scientific ties are processed as simultaneous score blocks.  The running sum
is inspected only after the whole block, so row order and cell-type names
cannot break a tie.  Missing or non-estimable ranked rows are excluded from
the walk and reported as coverage loss; they are never imputed as zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

SUPPORTED_TOP_FRACTIONS = (0.1, 0.2, 0.3, 0.4)
RANK_STATUSES = frozenset(
    {
        "observed",
        "missing",
        "not_estimable",
        "not_predicted",
        "resource_unavailable",
        "cell_type_missing",
        "filtered",
        "failed",
        "not_supported",
    }
)
ScoreType = Literal["pos", "abs"]
CellPairMode = Literal["ordered", "unordered"]


def _column_names(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if any(not value or value != value.strip() for value in values):
        raise ValueError(f"{field} must contain canonical non-empty names")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique names")
    return values


def _canonical_fraction(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{field} must contain only supported fractions {SUPPORTED_TOP_FRACTIONS}"
        )
    if not isinstance(value, (str, Real)):
        raise ValueError(
            f"{field} must contain only supported fractions {SUPPORTED_TOP_FRACTIONS}"
        )
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(
            f"{field} must contain only supported fractions {SUPPORTED_TOP_FRACTIONS}"
        ) from error
    if not math.isfinite(numeric):
        raise ValueError(
            f"{field} must contain only supported fractions {SUPPORTED_TOP_FRACTIONS}"
        )
    for supported in SUPPORTED_TOP_FRACTIONS:
        if math.isclose(numeric, supported, rel_tol=0.0, abs_tol=1e-12):
            return supported
    raise ValueError(
        f"{field} must contain only supported fractions {SUPPORTED_TOP_FRACTIONS}"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialDESSpec:
    """Frozen input schema and scoring policy for spatial DES evaluation.

    Expected-set rows represent members when ``expected_member_column`` is
    ``None``.  Set it to a boolean column (for example ``"is_expected"``) when
    the input materializes the full spatial pair universe.  The latter form
    can represent an observed but empty expected set.
    """

    method_columns: tuple[str, ...] = ("method",)
    stratum_columns: tuple[str, ...] = ("dataset",)
    condition_column: str = "condition"
    sender_column: str = "sender"
    receiver_column: str = "receiver"
    strength_column: str = "ranked_strength"
    status_column: str | None = "status"
    eligible_statuses: tuple[str, ...] = ("observed",)
    direction_column: str | None = None
    higher_is_better: bool = True
    fraction_column: str = "top_fraction"
    expected_member_column: str | None = None
    top_fractions: tuple[float, ...] = SUPPORTED_TOP_FRACTIONS
    score_type: ScoreType = "pos"
    cell_pair_mode: CellPairMode = "ordered"

    def __post_init__(self) -> None:
        methods = _column_names(self.method_columns, field="method_columns")
        strata = _column_names(self.stratum_columns, field="stratum_columns")
        if not methods:
            raise ValueError("method_columns must not be empty")

        scalar_columns = {
            "condition_column": self.condition_column,
            "sender_column": self.sender_column,
            "receiver_column": self.receiver_column,
            "strength_column": self.strength_column,
            "fraction_column": self.fraction_column,
        }
        scalar_columns.update(
            {
                name: value
                for name, value in {
                    "status_column": self.status_column,
                    "direction_column": self.direction_column,
                    "expected_member_column": self.expected_member_column,
                }.items()
                if value is not None
            }
        )
        if any(
            not value or value != value.strip() for value in scalar_columns.values()
        ):
            raise ValueError("spatial DES column names must be canonical and non-empty")

        identity_columns = (
            *strata,
            *methods,
            self.condition_column,
            self.sender_column,
            self.receiver_column,
        )
        auxiliary_columns = tuple(scalar_columns.values())
        if len(set(identity_columns)) != len(identity_columns):
            raise ValueError(
                "stratum, method, condition, sender, and receiver columns must not "
                "overlap"
            )
        if len(set(auxiliary_columns)) != len(auxiliary_columns):
            raise ValueError("spatial DES value columns must be distinct")
        if set((*strata, *methods)).intersection(auxiliary_columns):
            raise ValueError(
                "identity columns cannot also be spatial DES value columns"
            )

        statuses = _column_names(self.eligible_statuses, field="eligible_statuses")
        if not statuses:
            raise ValueError("eligible_statuses must not be empty")
        if self.status_column is not None:
            invalid_statuses = set(statuses).difference(RANK_STATUSES)
            if invalid_statuses:
                raise ValueError(
                    "eligible_statuses contains unsupported values: "
                    f"{sorted(invalid_statuses)}"
                )
        if not isinstance(self.higher_is_better, bool):
            raise ValueError("higher_is_better must be boolean")
        if self.score_type not in {"pos", "abs"}:
            raise ValueError("score_type must be 'pos' or 'abs'")
        if self.cell_pair_mode not in {"ordered", "unordered"}:
            raise ValueError("cell_pair_mode must be 'ordered' or 'unordered'")
        if not self.top_fractions:
            raise ValueError("top_fractions must not be empty")
        fractions = tuple(
            _canonical_fraction(value, field="top_fractions")
            for value in self.top_fractions
        )
        if len(set(fractions)) != len(fractions):
            raise ValueError("top_fractions must not contain duplicates")

        object.__setattr__(self, "method_columns", methods)
        object.__setattr__(self, "stratum_columns", strata)
        object.__setattr__(self, "eligible_statuses", statuses)
        object.__setattr__(self, "top_fractions", tuple(sorted(fractions)))


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialDESTables:
    """Tidy DES estimates and fixed-denominator coverage diagnostics."""

    scores: pd.DataFrame
    coverage: pd.DataFrame

    def __post_init__(self) -> None:
        for field in ("scores", "coverage"):
            table = getattr(self, field)
            if not isinstance(table, pd.DataFrame) or table.empty:
                raise ValueError(f"{field} must be a non-empty DataFrame")
            object.__setattr__(self, field, table.copy(deep=True))


def _validate_identifiers(
    table: pd.DataFrame, columns: list[str], *, label: str
) -> None:
    if table.loc[:, columns].isna().any().any():
        raise ValueError(f"{label} identifiers must not contain missing values")
    for column in columns:
        canonical = table[column].map(
            lambda value: (
                isinstance(value, str) and bool(value) and value == value.strip()
            )
        )
        if not canonical.all():
            raise ValueError(
                f"{label} identifier {column!r} must contain canonical non-empty "
                "strings"
            )


def _validate_pair_mode(
    table: pd.DataFrame, spec: SpatialDESSpec, *, label: str
) -> None:
    if spec.cell_pair_mode == "ordered":
        return
    sender = table[spec.sender_column].astype(str)
    receiver = table[spec.receiver_column].astype(str)
    if sender.gt(receiver).any():
        raise ValueError(
            f"{label} unordered cell pairs must be pre-collapsed and canonicalized "
            "with sender <= receiver"
        )


def _validate_ranked_strengths(
    table: pd.DataFrame, spec: SpatialDESSpec
) -> pd.DataFrame:
    required = {
        *spec.stratum_columns,
        *spec.method_columns,
        spec.condition_column,
        spec.sender_column,
        spec.receiver_column,
        spec.strength_column,
    }
    if spec.status_column is not None:
        required.add(spec.status_column)
    if spec.direction_column is not None:
        required.add(spec.direction_column)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"ranked strength table is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("ranked strength table must not be empty")

    identifiers = [
        *spec.stratum_columns,
        *spec.method_columns,
        spec.condition_column,
        spec.sender_column,
        spec.receiver_column,
    ]
    _validate_identifiers(table, identifiers, label="ranked strength")
    _validate_pair_mode(table, spec, label="ranked strength")
    if table.duplicated(identifiers).any():
        raise ValueError(
            "ranked strength table contains duplicate method-condition ordered "
            "cell-pair rows"
        )

    result = table.loc[:, list(required)].copy(deep=True)
    if spec.status_column is None:
        result["_status"] = "observed"
    else:
        status = result[spec.status_column]
        canonical_status = status.map(
            lambda value: (
                isinstance(value, str) and bool(value) and value == value.strip()
            )
        )
        if not canonical_status.all():
            raise ValueError("ranked strength status values must be canonical strings")
        invalid = set(status.astype(str)).difference(RANK_STATUSES)
        if invalid:
            raise ValueError(
                f"ranked strength table contains invalid statuses: {sorted(invalid)}"
            )
        result["_status"] = status.astype(str)

    eligible = result["_status"].isin(spec.eligible_statuses)
    numeric = pd.to_numeric(result[spec.strength_column], errors="coerce")
    supplied = result[spec.strength_column].notna()
    invalid_numeric = (supplied & numeric.isna()) | np.isinf(numeric.fillna(0.0))
    if invalid_numeric.any():
        raise ValueError("ranked strength values must be finite numeric values")
    if numeric.loc[eligible].isna().any():
        raise ValueError("eligible ranked strength rows require finite scores")
    if numeric.loc[~eligible].notna().any():
        raise ValueError("non-eligible ranked strength rows require missing scores")
    result[spec.strength_column] = numeric

    if spec.direction_column is None:
        multiplier = 1.0 if spec.higher_is_better else -1.0
        result["_oriented_strength"] = numeric * multiplier
    else:
        directions = result[spec.direction_column]
        if not directions.isin({"higher", "lower"}).all():
            raise ValueError("score direction must be 'higher' or 'lower'")
        group_columns = [
            *spec.stratum_columns,
            *spec.method_columns,
            spec.condition_column,
        ]
        if (
            result.groupby(group_columns, sort=False, observed=True)[
                spec.direction_column
            ].nunique()
            > 1
        ).any():
            raise ValueError(
                "score direction must be constant per method-condition ranking"
            )
        direction = directions.map({"higher": 1.0, "lower": -1.0})
        result["_oriented_strength"] = numeric * direction

    result["_eligible"] = eligible
    return result


def _validate_expected_sets(table: pd.DataFrame, spec: SpatialDESSpec) -> pd.DataFrame:
    required = {
        *spec.stratum_columns,
        spec.condition_column,
        spec.sender_column,
        spec.receiver_column,
        spec.fraction_column,
    }
    if spec.expected_member_column is not None:
        required.add(spec.expected_member_column)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"expected spatial table is missing columns: {sorted(missing)}"
        )

    result = table.loc[:, list(required)].copy(deep=True)
    if result.empty:
        result["_is_expected"] = pd.Series(dtype=bool)
        return result

    identifiers = [
        *spec.stratum_columns,
        spec.condition_column,
        spec.sender_column,
        spec.receiver_column,
    ]
    _validate_identifiers(result, identifiers, label="expected spatial")
    _validate_pair_mode(result, spec, label="expected spatial")
    result[spec.fraction_column] = result[spec.fraction_column].map(
        lambda value: _canonical_fraction(value, field=spec.fraction_column)
    )
    unexpected = set(result[spec.fraction_column]).difference(spec.top_fractions)
    if unexpected:
        raise ValueError(
            "expected spatial table contains fractions not requested by top_fractions: "
            f"{sorted(unexpected)}"
        )
    duplicate_key = [
        *spec.stratum_columns,
        spec.condition_column,
        spec.fraction_column,
        spec.sender_column,
        spec.receiver_column,
    ]
    if result.duplicated(duplicate_key).any():
        raise ValueError(
            "expected spatial table contains duplicate condition-fraction ordered "
            "cell-pair rows"
        )

    if spec.expected_member_column is None:
        result["_is_expected"] = True
    else:
        member_flags = result[spec.expected_member_column]
        if member_flags.isna().any() or not pd.api.types.is_bool_dtype(member_flags):
            raise ValueError("expected_member_column must contain non-missing booleans")
        result["_is_expected"] = member_flags.astype(bool)

    nesting_columns = [*spec.stratum_columns, spec.condition_column]
    pair_columns = [spec.sender_column, spec.receiver_column]
    for _, group in result.groupby(
        nesting_columns, sort=False, observed=True, dropna=False
    ):
        previous: set[tuple[str, str]] | None = None
        previous_fraction: float | None = None
        spatial_universe: set[tuple[str, str]] | None = None
        for fraction in sorted(group[spec.fraction_column].unique()):
            fraction_group = group.loc[group[spec.fraction_column].eq(fraction)]
            if spec.expected_member_column is not None:
                current_universe = set(
                    fraction_group.loc[:, pair_columns].itertuples(
                        index=False, name=None
                    )
                )
                if (
                    spatial_universe is not None
                    and current_universe != spatial_universe
                ):
                    raise ValueError(
                        "full expected membership tables must use the same spatial "
                        "cell-pair universe at every available top fraction"
                    )
                spatial_universe = current_universe
            members = group.loc[
                group[spec.fraction_column].eq(fraction) & group["_is_expected"],
                pair_columns,
            ]
            current = set(members.itertuples(index=False, name=None))
            if previous is not None and not previous.issubset(current):
                raise ValueError(
                    "expected spatial sets must be nested across increasing top "
                    f"fractions; {previous_fraction} is not a subset of {fraction}"
                )
            previous = current
            previous_fraction = float(fraction)
    return result


def _identity_record(columns: list[str], key: object) -> dict[str, object]:
    values = key if isinstance(key, tuple) else (key,)
    return dict(zip(columns, values, strict=True))


def _pair_set(table: pd.DataFrame, spec: SpatialDESSpec) -> set[tuple[str, str]]:
    return set(
        table.loc[:, [spec.sender_column, spec.receiver_column]].itertuples(
            index=False, name=None
        )
    )


def _tie_diagnostics(table: pd.DataFrame) -> tuple[int, int]:
    counts = table["_oriented_strength"].value_counts(sort=False)
    ties = counts.loc[counts > 1]
    return len(ties), int(ties.sum())


def _unweighted_es(
    ranked: pd.DataFrame,
    expected: set[tuple[str, str]],
    spec: SpatialDESSpec,
) -> tuple[float, float, int]:
    pair_columns = [spec.sender_column, spec.receiver_column]
    ranked = ranked.sort_values(
        "_oriented_strength", ascending=False, kind="stable", ignore_index=True
    )
    pairs = list(ranked.loc[:, pair_columns].itertuples(index=False, name=None))
    hit_count = sum(pair in expected for pair in pairs)
    miss_count = len(pairs) - hit_count
    if hit_count < 1 or miss_count < 1:
        raise ValueError("unweighted ES requires at least one hit and one miss")

    running = 0.0
    peak_signed = 0.0
    peak_rank = 0
    traversed = 0
    for _, block in ranked.groupby("_oriented_strength", sort=False, observed=True):
        block_pairs = block.loc[:, pair_columns].itertuples(index=False, name=None)
        block_hits = sum(pair in expected for pair in block_pairs)
        block_misses = len(block) - block_hits
        running += block_hits / hit_count - block_misses / miss_count
        traversed += len(block)
        running = min(1.0, max(-1.0, running))
        current = running if spec.score_type == "pos" else abs(running)
        best = peak_signed if spec.score_type == "pos" else abs(peak_signed)
        if current > best + 1e-15:
            peak_signed = running
            peak_rank = traversed

    if abs(peak_signed) < 1e-15:
        peak_signed = 0.0
    score = max(0.0, peak_signed) if spec.score_type == "pos" else abs(peak_signed)
    return float(score), float(peak_signed), peak_rank


def _score_semantics(score_type: ScoreType) -> tuple[str, str]:
    if score_type == "pos":
        return (
            "maximum_positive_unweighted_running_sum",
            "fgsea_scoreType=pos;gseaParam=0",
        )
    return (
        "maximum_absolute_unweighted_running_sum_nonnegative",
        "custom_abs_excursion;not_a_literal_fgsea_scoreType;gseaParam=0",
    )


def evaluate_spatial_des(
    ranked_strengths: pd.DataFrame,
    expected_spatial_sets: pd.DataFrame,
    specification: SpatialDESSpec | None = None,
) -> SpatialDESTables:
    """Evaluate condition-specific spatial DES for one or more methods.

    ``ranked_strengths`` must materialize one row per method, condition, and
    ordered sender-receiver pair.  ``expected_spatial_sets`` has no method
    dimension: the same spatial evidence is applied to every method in a
    stratum.  Every requested top fraction produces one score row, including
    explicit ``not_estimable`` rows when spatial evidence or ranked coverage is
    insufficient.
    """

    spec = specification or SpatialDESSpec()
    ranked = _validate_ranked_strengths(ranked_strengths, spec)
    expected = _validate_expected_sets(expected_spatial_sets, spec)

    score_group_columns = [
        *spec.stratum_columns,
        *spec.method_columns,
        spec.condition_column,
    ]
    expected_group_columns = [
        *spec.stratum_columns,
        spec.condition_column,
        spec.fraction_column,
    ]
    expected_groups = {
        (key if isinstance(key, tuple) else (key,)): group
        for key, group in expected.groupby(
            expected_group_columns,
            sort=False,
            observed=True,
            dropna=False,
        )
    }
    semantics, fgsea_analogue = _score_semantics(spec.score_type)
    score_records: list[dict[str, object]] = []
    coverage_records: list[dict[str, object]] = []

    for key, group in ranked.groupby(
        score_group_columns, sort=True, observed=True, dropna=False
    ):
        identity = _identity_record(score_group_columns, key)
        stratum_condition_key = tuple(
            identity[column]
            for column in [*spec.stratum_columns, spec.condition_column]
        )
        eligible = group.loc[group["_eligible"]].copy()
        eligible_pairs = _pair_set(eligible, spec)
        tie_blocks, pairs_in_ties = _tie_diagnostics(eligible)
        status_counts = {
            str(status): int(count)
            for status, count in group["_status"].value_counts(sort=False).items()
        }

        for fraction in spec.top_fractions:
            expected_key = (*stratum_condition_key, fraction)
            expected_group = expected_groups.get(expected_key)
            expected_available = expected_group is not None
            if expected_group is None:
                expected_rows = 0
                expected_members: set[tuple[str, str]] = set()
            else:
                expected_rows = len(expected_group)
                expected_members = _pair_set(
                    expected_group.loc[expected_group["_is_expected"]], spec
                )

            covered_members = expected_members.intersection(eligible_pairs)
            background_pairs = eligible_pairs.difference(expected_members)
            expected_count = len(expected_members)
            covered_count = len(covered_members)
            expected_coverage = (
                covered_count / expected_count if expected_count else math.nan
            )

            reason_code: str | None = None
            des = math.nan
            signed_peak = math.nan
            peak_rank: int | None = None
            if not expected_available:
                reason_code = "expected_spatial_set_missing"
            elif expected_count == 0:
                reason_code = "expected_spatial_set_empty"
            elif len(eligible) == 0:
                reason_code = "no_eligible_ranked_pairs"
            elif covered_count == 0:
                reason_code = "no_expected_pairs_covered"
            elif len(background_pairs) == 0:
                reason_code = "no_ranked_background_pairs"
            else:
                des, signed_peak, peak_rank = _unweighted_es(
                    eligible, expected_members, spec
                )
            status = "observed" if reason_code is None else "not_estimable"

            common = {
                **identity,
                spec.fraction_column: fraction,
                "status": status,
                "reason_code": reason_code,
            }
            score_records.append(
                {
                    **common,
                    "metric": "spatial_des",
                    "des": des,
                    "signed_peak_es": signed_peak,
                    "peak_rank": peak_rank,
                    "score_type": spec.score_type,
                    "score_semantics": semantics,
                    "fgsea_analogue": fgsea_analogue,
                    "weight_exponent": 0.0,
                    "tie_policy": "simultaneous_equal_strength_blocks",
                    "cell_pair_direction": (
                        "ordered_sender_to_receiver"
                        if spec.cell_pair_mode == "ordered"
                        else "unordered_directions_collapsed"
                    ),
                    "missing_policy": "exclude_and_report_coverage_never_impute_zero",
                }
            )
            coverage_records.append(
                {
                    **common,
                    "expected_set_available": expected_available,
                    "expected_input_rows": expected_rows,
                    "expected_input_mode": (
                        "full_membership_table"
                        if spec.expected_member_column is not None
                        else "set_members_only"
                    ),
                    "expected_pairs": expected_count,
                    "expected_pairs_covered": covered_count,
                    "expected_pairs_missing": expected_count - covered_count,
                    "expected_pair_coverage_fraction": expected_coverage,
                    "rank_rows_total": len(group),
                    "rank_pairs_eligible": len(eligible),
                    "rank_pairs_noneligible": len(group) - len(eligible),
                    "rank_eligible_fraction": len(eligible) / len(group),
                    "ranked_background_pairs": len(background_pairs),
                    "tied_strength_blocks": tie_blocks,
                    "ranked_pairs_in_ties": pairs_in_ties,
                    "rank_status_counts_json": json.dumps(
                        status_counts,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                }
            )

    scores = pd.DataFrame.from_records(score_records)
    coverage = pd.DataFrame.from_records(coverage_records)
    ordering = [*score_group_columns, spec.fraction_column]
    scores = scores.sort_values(ordering, kind="stable", ignore_index=True)
    coverage = coverage.sort_values(ordering, kind="stable", ignore_index=True)
    return SpatialDESTables(scores=scores, coverage=coverage)


__all__ = [
    "RANK_STATUSES",
    "SUPPORTED_TOP_FRACTIONS",
    "CellPairMode",
    "ScoreType",
    "SpatialDESSpec",
    "SpatialDESTables",
    "evaluate_spatial_des",
]
