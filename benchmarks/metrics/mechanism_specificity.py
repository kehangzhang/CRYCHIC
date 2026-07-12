"""Pure paired-seed evaluation for the G1.5 mechanism-specificity gate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SCHEMA_VERSION = "crychic-mechanism-specificity-v2"
ACTIVE_SCENARIO = "active"
LIGAND_ONLY_SCENARIO = "ligand_only"
RECEPTOR_KNOCKOUT_SCENARIO = "receptor_knockout"
REQUIRED_SCENARIOS = (
    ACTIVE_SCENARIO,
    "global_null",
    "abundance_only",
    LIGAND_ONLY_SCENARIO,
    "target_only",
    "receiver_autonomous",
    RECEPTOR_KNOCKOUT_SCENARIO,
)
COMPONENT_COLUMNS = (
    "availability_effect",
    "receptor_gate",
    "receiver_program_effect",
    "incremental_downstream_effect",
    "sender_effect",
    "integrated_lr_effect",
)
REQUIRED_COLUMNS = {
    "seed",
    "scenario",
    *COMPONENT_COLUMNS,
    "comparison_coverage",
    "reference_comparison_coverage",
    "known_edge_rank",
    "known_edge_positive_direction",
    "status",
    "reason_code",
}
ALLOWED_STATUSES = frozenset({"observed", "not_estimable", "failed", "missing"})
OUTPUT_COLUMNS = (
    "evaluation_phase",
    "gate_role",
    "metric",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence_level",
    "interval_type",
    "threshold",
    "operator",
    "gate_passed",
    "status",
    "reason_code",
    "n_expected_seeds",
    "n_eligible_seeds",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MechanismSpecificitySpecification:
    """Frozen thresholds for one development or independent-holdout phase."""

    evaluation_phase: str
    expected_seed_count: int
    minimum_margin: float
    confidence_level: float
    require_ci_lower_above: float
    active_max_rank: int
    active_min_positive_direction_fraction: float
    max_ligand_only_to_active_ratio: float
    max_comparison_coverage_loss: float
    integrated_positive_threshold: float
    max_negative_false_activation_rate: float
    negative_scenarios: tuple[str, ...]
    receptor_knockout_must_be_zero_or_negative: bool

    def __post_init__(self) -> None:
        if not self.evaluation_phase:
            raise ValueError("evaluation_phase must be non-empty")
        if self.expected_seed_count < 2:
            raise ValueError("expected_seed_count must be at least 2")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if self.active_max_rank < 1:
            raise ValueError("active_max_rank must be positive")
        for field_name in (
            "active_min_positive_direction_fraction",
            "max_ligand_only_to_active_ratio",
            "max_comparison_coverage_loss",
            "max_negative_false_activation_rate",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be finite and lie in [0, 1]")
        for field_name in (
            "minimum_margin",
            "require_ci_lower_above",
            "integrated_positive_threshold",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        scenarios = tuple(self.negative_scenarios)
        if not scenarios or len(set(scenarios)) != len(scenarios):
            raise ValueError("negative_scenarios must be non-empty and unique")
        unknown = set(scenarios).difference(REQUIRED_SCENARIOS)
        if unknown or ACTIVE_SCENARIO in scenarios:
            raise ValueError("negative_scenarios must be known non-active scenarios")
        if RECEPTOR_KNOCKOUT_SCENARIO not in scenarios:
            raise ValueError("negative_scenarios must include receptor_knockout")
        object.__setattr__(self, "negative_scenarios", scenarios)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def specification_from_config(
    config: Mapping[str, object], *, evaluation_phase: str
) -> MechanismSpecificitySpecification:
    """Resolve one phase without reading files or mutating the configuration."""

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")
    phases = _mapping(config.get("evaluation_phases"), field="evaluation_phases")
    if evaluation_phase not in phases:
        raise ValueError(f"unknown evaluation_phase: {evaluation_phase!r}")
    phase = _mapping(phases[evaluation_phase], field=evaluation_phase)
    primary = _mapping(config.get("primary_endpoint"), field="primary_endpoint")
    secondary = _mapping(config.get("secondary_gates"), field="secondary_gates")
    negative = secondary.get("negative_scenarios")
    if not isinstance(negative, Sequence) or isinstance(negative, (str, bytes)):
        raise ValueError("secondary_gates.negative_scenarios must be an array")
    return MechanismSpecificitySpecification(
        evaluation_phase=evaluation_phase,
        expected_seed_count=int(cast(Any, phase["required_paired_seeds"])),
        minimum_margin=float(cast(Any, primary["minimum_margin"])),
        confidence_level=float(cast(Any, primary["confidence_level"])),
        require_ci_lower_above=float(cast(Any, primary["require_ci_lower_above"])),
        active_max_rank=int(cast(Any, secondary["active_max_rank"])),
        active_min_positive_direction_fraction=float(
            cast(Any, secondary["active_min_positive_direction_fraction"])
        ),
        max_ligand_only_to_active_ratio=float(
            cast(Any, secondary["max_ligand_only_to_active_ratio"])
        ),
        max_comparison_coverage_loss=float(
            cast(Any, secondary["max_comparison_coverage_loss"])
        ),
        integrated_positive_threshold=float(
            cast(Any, secondary["integrated_positive_threshold"])
        ),
        max_negative_false_activation_rate=float(
            cast(Any, secondary["max_negative_false_activation_rate"])
        ),
        negative_scenarios=tuple(str(value) for value in negative),
        receptor_knockout_must_be_zero_or_negative=bool(
            secondary["receptor_knockout_must_be_zero_or_negative"]
        ),
    )


def _prepare_records(records: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    missing = REQUIRED_COLUMNS.difference(records.columns)
    if missing:
        raise ValueError(f"component evidence records are missing: {sorted(missing)}")
    table = records.copy(deep=True)
    if table.empty:
        raise ValueError("component evidence records must not be empty")
    table["scenario"] = table["scenario"].astype("string")
    if table["scenario"].isna().any() or table["scenario"].eq("").any():
        raise ValueError("scenario must contain non-empty identifiers")
    unknown = set(table["scenario"].astype(str)).difference(REQUIRED_SCENARIOS)
    if unknown:
        raise ValueError(
            f"component evidence contains unknown scenarios: {sorted(unknown)}"
        )
    seeds = pd.to_numeric(table["seed"], errors="coerce")
    if (
        seeds.isna().any()
        or np.isinf(seeds).any()
        or not np.equal(seeds, np.floor(seeds)).all()
    ):
        raise ValueError("seed must contain finite integer identifiers")
    table["seed"] = seeds.astype(np.int64)
    if table.duplicated(["seed", "scenario"]).any():
        raise ValueError("component evidence must have one row per seed and scenario")
    table["status"] = table["status"].astype("string")
    invalid_status = set(table["status"].dropna().astype(str)).difference(
        ALLOWED_STATUSES
    )
    if table["status"].isna().any() or invalid_status:
        raise ValueError(
            f"component evidence has invalid status: {sorted(invalid_status)}"
        )
    observed = table["status"].eq("observed")
    reasons = table["reason_code"].astype("string")
    if reasons.loc[observed].notna().any():
        raise ValueError("observed component evidence must not carry a reason_code")
    if reasons.loc[~observed].isna().any() or reasons.loc[~observed].eq("").any():
        raise ValueError("non-observed component evidence requires a reason_code")

    numeric_columns = (
        *COMPONENT_COLUMNS,
        "comparison_coverage",
        "reference_comparison_coverage",
        "known_edge_rank",
    )
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    required_observed = [*COMPONENT_COLUMNS, "comparison_coverage"]
    if table.loc[observed, required_observed].isna().any().any():
        raise ValueError(
            "observed evidence requires finite component and coverage values"
        )
    observed_values = table.loc[observed, required_observed].to_numpy(dtype=float)
    if np.isinf(observed_values).any():
        raise ValueError("observed component evidence must be finite")
    for column in ("receptor_gate", "comparison_coverage"):
        values = table.loc[observed, column]
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must lie in [0, 1]")
    active = observed & table["scenario"].eq(ACTIVE_SCENARIO)
    if table.loc[active, "reference_comparison_coverage"].isna().any():
        raise ValueError("observed active rows require reference_comparison_coverage")
    reference_coverage = table.loc[active, "reference_comparison_coverage"]
    if ((reference_coverage < 0) | (reference_coverage > 1)).any():
        raise ValueError("reference_comparison_coverage must lie in [0, 1]")
    ranks = table.loc[active, "known_edge_rank"]
    if ranks.isna().any() or (ranks < 1).any() or np.isinf(ranks).any():
        raise ValueError("observed active rows require a finite rank >= 1")
    directions = table.loc[active, "known_edge_positive_direction"]
    if not directions.map(lambda value: value in (True, False, 0, 1)).all():
        raise ValueError("observed active direction must be boolean")
    table.loc[active, "known_edge_positive_direction"] = directions.astype(bool)
    return table


def _not_estimable_row(
    specification: MechanismSpecificitySpecification,
    *,
    role: str,
    metric: str,
    reason_code: str,
    eligible_seeds: int,
) -> dict[str, object]:
    return {
        "evaluation_phase": specification.evaluation_phase,
        "gate_role": role,
        "metric": metric,
        "estimate": math.nan,
        "ci_lower": math.nan,
        "ci_upper": math.nan,
        "confidence_level": math.nan,
        "interval_type": None,
        "threshold": math.nan,
        "operator": None,
        "gate_passed": None,
        "status": "not_estimable",
        "reason_code": reason_code,
        "n_expected_seeds": specification.expected_seed_count,
        "n_eligible_seeds": eligible_seeds,
    }


def _observed_row(
    specification: MechanismSpecificitySpecification,
    *,
    role: str,
    metric: str,
    estimate: float,
    threshold: float,
    operator: str,
    gate_passed: bool,
    eligible_seeds: int,
    ci_lower: float = math.nan,
    ci_upper: float = math.nan,
    confidence_level: float = math.nan,
    interval_type: str | None = None,
) -> dict[str, object]:
    return {
        "evaluation_phase": specification.evaluation_phase,
        "gate_role": role,
        "metric": metric,
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "interval_type": interval_type,
        "threshold": threshold,
        "operator": operator,
        "gate_passed": gate_passed,
        "status": "observed",
        "reason_code": None if gate_passed else "gate_threshold_not_met",
        "n_expected_seeds": specification.expected_seed_count,
        "n_eligible_seeds": eligible_seeds,
    }


def _observed_scenario(table: pd.DataFrame, scenario: str) -> pd.DataFrame:
    return table.loc[
        table["scenario"].eq(scenario) & table["status"].eq("observed")
    ].sort_values("seed", kind="stable")


def _support_reason(
    *, metric: str, expected: int, observed: int, paired: bool = False
) -> str:
    support = "paired_seed" if paired else "seed"
    return (
        f"insufficient_{support}_support:{metric}:"
        f"expected={expected}:observed={observed}"
    )


def _paired_scenarios(
    table: pd.DataFrame,
    left: str,
    right: str,
    *,
    expected: int,
) -> pd.DataFrame:
    left_rows = _observed_scenario(table, left)
    right_rows = _observed_scenario(table, right)
    paired = left_rows.merge(
        right_rows,
        on="seed",
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    left_seeds = set(left_rows["seed"])
    right_seeds = set(right_rows["seed"])
    if (
        len(left_rows) != expected
        or len(right_rows) != expected
        or left_seeds != right_seeds
        or len(paired) != expected
    ):
        return paired.iloc[0:0]
    return paired


def _mean_t_interval(
    values: np.ndarray, *, confidence_level: float
) -> tuple[float, float, float]:
    estimate = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    critical = float(
        student_t.ppf(1.0 - (1.0 - confidence_level) / 2.0, len(values) - 1)
    )
    half_width = critical * standard_error
    return estimate, estimate - half_width, estimate + half_width


def evaluate_mechanism_specificity(
    records: pd.DataFrame,
    specification: MechanismSpecificitySpecification,
) -> pd.DataFrame:
    """Evaluate all G1.5 gates without imputation or unpaired seed fallback."""

    if not isinstance(specification, MechanismSpecificitySpecification):
        raise TypeError("specification must be MechanismSpecificitySpecification")
    table = _prepare_records(records)
    expected = specification.expected_seed_count
    rows: list[dict[str, object]] = []

    paired = _paired_scenarios(
        table, ACTIVE_SCENARIO, LIGAND_ONLY_SCENARIO, expected=expected
    )
    primary_metric = "paired_known_edge_active_minus_ligand_only"
    if len(paired) != expected:
        eligible = len(
            _observed_scenario(table, ACTIVE_SCENARIO).merge(
                _observed_scenario(table, LIGAND_ONLY_SCENARIO), on="seed"
            )
        )
        rows.append(
            _not_estimable_row(
                specification,
                role="primary",
                metric=primary_metric,
                reason_code=_support_reason(
                    metric=primary_metric,
                    expected=expected,
                    observed=eligible,
                    paired=True,
                ),
                eligible_seeds=eligible,
            )
        )
        rows.append(
            _not_estimable_row(
                specification,
                role="secondary",
                metric="ligand_only_to_active_ratio",
                reason_code=_support_reason(
                    metric="ligand_only_to_active_ratio",
                    expected=expected,
                    observed=eligible,
                    paired=True,
                ),
                eligible_seeds=eligible,
            )
        )
    else:
        margins = paired["integrated_lr_effect_left"].to_numpy(dtype=float) - paired[
            "integrated_lr_effect_right"
        ].to_numpy(dtype=float)
        estimate, lower, upper = _mean_t_interval(
            margins, confidence_level=specification.confidence_level
        )
        primary_passed = (
            estimate >= specification.minimum_margin
            and lower > specification.require_ci_lower_above
        )
        rows.append(
            _observed_row(
                specification,
                role="primary",
                metric=primary_metric,
                estimate=estimate,
                threshold=specification.minimum_margin,
                operator=">=_and_ci_lower>",
                gate_passed=primary_passed,
                eligible_seeds=len(paired),
                ci_lower=lower,
                ci_upper=upper,
                confidence_level=specification.confidence_level,
                interval_type="paired_seed_t_interval",
            )
        )
        active_mean = float(paired["integrated_lr_effect_left"].mean())
        if active_mean <= 0:
            rows.append(
                _not_estimable_row(
                    specification,
                    role="secondary",
                    metric="ligand_only_to_active_ratio",
                    reason_code="nonpositive_active_mean_for_ratio",
                    eligible_seeds=len(paired),
                )
            )
        else:
            ratio = float(paired["integrated_lr_effect_right"].mean()) / active_mean
            rows.append(
                _observed_row(
                    specification,
                    role="secondary",
                    metric="ligand_only_to_active_ratio",
                    estimate=ratio,
                    threshold=specification.max_ligand_only_to_active_ratio,
                    operator="<=",
                    gate_passed=(
                        ratio <= specification.max_ligand_only_to_active_ratio
                    ),
                    eligible_seeds=len(paired),
                )
            )

    active = _observed_scenario(table, ACTIVE_SCENARIO)
    active_seed_set = set(active["seed"])
    active_metrics = (
        "active_max_rank",
        "active_positive_direction_fraction",
        "active_comparison_coverage_loss",
    )
    if len(active) != expected:
        for metric in active_metrics:
            rows.append(
                _not_estimable_row(
                    specification,
                    role="secondary",
                    metric=metric,
                    reason_code=_support_reason(
                        metric=metric, expected=expected, observed=len(active)
                    ),
                    eligible_seeds=len(active),
                )
            )
    else:
        maximum_rank = float(active["known_edge_rank"].max())
        rows.append(
            _observed_row(
                specification,
                role="secondary",
                metric="active_max_rank",
                estimate=maximum_rank,
                threshold=float(specification.active_max_rank),
                operator="<=",
                gate_passed=maximum_rank <= specification.active_max_rank,
                eligible_seeds=len(active),
            )
        )
        positive_fraction = float(
            active["known_edge_positive_direction"].astype(bool).mean()
        )
        rows.append(
            _observed_row(
                specification,
                role="secondary",
                metric="active_positive_direction_fraction",
                estimate=positive_fraction,
                threshold=specification.active_min_positive_direction_fraction,
                operator=">=",
                gate_passed=(
                    positive_fraction
                    >= specification.active_min_positive_direction_fraction
                ),
                eligible_seeds=len(active),
            )
        )
        coverage_loss = float(
            (
                active["reference_comparison_coverage"] - active["comparison_coverage"]
            ).mean()
        )
        rows.append(
            _observed_row(
                specification,
                role="secondary",
                metric="active_comparison_coverage_loss",
                estimate=coverage_loss,
                threshold=specification.max_comparison_coverage_loss,
                operator="<=",
                gate_passed=(
                    coverage_loss <= specification.max_comparison_coverage_loss
                ),
                eligible_seeds=len(active),
            )
        )

    for scenario in specification.negative_scenarios:
        negative = _observed_scenario(table, scenario)
        negative_seed_set = set(negative["seed"])
        paired_negative_seeds = len(active_seed_set.intersection(negative_seed_set))
        metric = f"{scenario}_false_activation_rate"
        if (
            len(active) != expected
            or len(negative) != expected
            or active_seed_set != negative_seed_set
        ):
            rows.append(
                _not_estimable_row(
                    specification,
                    role="secondary",
                    metric=metric,
                    reason_code=_support_reason(
                        metric=metric,
                        expected=expected,
                        observed=paired_negative_seeds,
                        paired=True,
                    ),
                    eligible_seeds=paired_negative_seeds,
                )
            )
            continue
        rate = float(
            (
                negative["integrated_lr_effect"]
                > specification.integrated_positive_threshold
            ).mean()
        )
        rows.append(
            _observed_row(
                specification,
                role="secondary",
                metric=metric,
                estimate=rate,
                threshold=specification.max_negative_false_activation_rate,
                operator="<=",
                gate_passed=(rate <= specification.max_negative_false_activation_rate),
                eligible_seeds=len(negative),
            )
        )

    if specification.receptor_knockout_must_be_zero_or_negative:
        knockout = _observed_scenario(table, RECEPTOR_KNOCKOUT_SCENARIO)
        knockout_seed_set = set(knockout["seed"])
        paired_knockout_seeds = len(active_seed_set.intersection(knockout_seed_set))
        metric = "receptor_knockout_mean_integrated_effect"
        if (
            len(active) != expected
            or len(knockout) != expected
            or active_seed_set != knockout_seed_set
        ):
            rows.append(
                _not_estimable_row(
                    specification,
                    role="secondary",
                    metric=metric,
                    reason_code=_support_reason(
                        metric=metric,
                        expected=expected,
                        observed=paired_knockout_seeds,
                        paired=True,
                    ),
                    eligible_seeds=paired_knockout_seeds,
                )
            )
        else:
            effect = float(knockout["integrated_lr_effect"].mean())
            rows.append(
                _observed_row(
                    specification,
                    role="secondary",
                    metric=metric,
                    estimate=effect,
                    threshold=0.0,
                    operator="<=",
                    gate_passed=effect <= 0.0,
                    eligible_seeds=len(knockout),
                )
            )

    gate_rows = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    unavailable = gate_rows.loc[gate_rows["status"].ne("observed"), "metric"].tolist()
    if unavailable:
        overall = _not_estimable_row(
            specification,
            role="overall",
            metric="g1_5_mechanism_specificity_gate",
            reason_code="required_gate_not_estimable:" + ",".join(unavailable),
            eligible_seeds=0,
        )
    else:
        failed = gate_rows.loc[gate_rows["gate_passed"].eq(False), "metric"].tolist()
        overall = _observed_row(
            specification,
            role="overall",
            metric="g1_5_mechanism_specificity_gate",
            estimate=0.0 if failed else 1.0,
            threshold=1.0,
            operator="all_required_gates",
            gate_passed=not failed,
            eligible_seeds=expected,
        )
        if failed:
            overall["reason_code"] = "failed_gate:" + ",".join(failed)
    return pd.concat(
        [gate_rows, pd.DataFrame([overall], columns=OUTPUT_COLUMNS)],
        ignore_index=True,
    )
