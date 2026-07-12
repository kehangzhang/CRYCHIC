"""Pure multi-edge, paired-seed evaluation for the G1.5 benchmark gate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SCHEMA_VERSION = "crychic-mechanism-specificity-v2"
TRUTH_SCHEMA_VERSION = "crychic-component-truth-matrix-v1"
ACTIVE_SCENARIO = "active"
LIGAND_ONLY_SCENARIO = "ligand_only"
RECEPTOR_KNOCKOUT_SCENARIO = "receptor_knockout"
MACRO_EDGE_ID = "__macro_equal_edge__"
OVERALL_EDGE_ID = "__all_known_edges__"
REQUIRED_SCENARIOS = (
    ACTIVE_SCENARIO,
    "global_null",
    "abundance_only",
    LIGAND_ONLY_SCENARIO,
    "target_only",
    "receiver_autonomous",
    RECEPTOR_KNOCKOUT_SCENARIO,
)
COMPONENT_TO_COLUMN = {
    "availability": "availability_effect",
    "receptor": "receptor_gate",
    "receiver_program": "receiver_program_effect",
    "incremental_downstream": "incremental_downstream_effect",
    "sender": "sender_effect",
    "integrated": "integrated_lr_effect",
}
COMPONENTS = tuple(COMPONENT_TO_COLUMN)
COMPONENT_COLUMNS = tuple(COMPONENT_TO_COLUMN.values())
TRUTH_CODES = frozenset({"positive", "zero", "allowed_positive", "ecosystem_only"})
REQUIRED_COLUMNS = {
    "seed",
    "scenario",
    "known_edge_id",
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
    "aggregation",
    "known_edge_id",
    "scenario",
    "component",
    "truth_code",
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
    "n_expected_edges",
    "n_eligible_edges",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentTruthMatrix:
    """Frozen known-edge catalog and scenario-by-component truth codes."""

    truth_set_id: str
    known_edge_ids: tuple[str, ...]
    truth_records: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        if not self.truth_set_id:
            raise ValueError("truth_set_id must be non-empty")
        edges = tuple(self.known_edge_ids)
        if (
            not edges
            or len(set(edges)) != len(edges)
            or any(not edge for edge in edges)
        ):
            raise ValueError("known_edge_ids must be non-empty and unique")
        expected = {
            (scenario, component)
            for scenario in REQUIRED_SCENARIOS
            for component in COMPONENTS
        }
        observed = {
            (scenario, component) for scenario, component, _ in self.truth_records
        }
        if observed != expected or len(self.truth_records) != len(expected):
            raise ValueError(
                "truth matrix must define every scenario-component pair once"
            )
        invalid = {code for _, _, code in self.truth_records if code not in TRUTH_CODES}
        if invalid:
            raise ValueError(f"truth matrix has invalid codes: {sorted(invalid)}")
        object.__setattr__(self, "known_edge_ids", edges)

    def code(self, scenario: str, component: str) -> str:
        """Return the frozen code for one scenario-component pair."""

        for truth_scenario, truth_component, code in self.truth_records:
            if truth_scenario == scenario and truth_component == component:
                return code
        raise KeyError((scenario, component))


@dataclass(frozen=True, slots=True, kw_only=True)
class MechanismSpecificitySpecification:
    """Frozen thresholds for one development or independent-holdout phase."""

    evaluation_phase: str
    expected_seed_count: int
    minimum_known_edges: int
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
    component_positive_threshold: float
    component_zero_maximum: float
    minimum_component_conformance_rate: float

    def __post_init__(self) -> None:
        if not self.evaluation_phase:
            raise ValueError("evaluation_phase must be non-empty")
        if self.expected_seed_count < 2:
            raise ValueError("expected_seed_count must be at least 2")
        if self.minimum_known_edges < 2:
            raise ValueError("minimum_known_edges must be at least 2")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if self.active_max_rank < 1:
            raise ValueError("active_max_rank must be positive")
        for field_name in (
            "active_min_positive_direction_fraction",
            "max_ligand_only_to_active_ratio",
            "max_comparison_coverage_loss",
            "max_negative_false_activation_rate",
            "minimum_component_conformance_rate",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be finite and lie in [0, 1]")
        for field_name in (
            "minimum_margin",
            "require_ci_lower_above",
            "integrated_positive_threshold",
            "component_positive_threshold",
            "component_zero_maximum",
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


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return cast(Sequence[object], value)


def component_truth_from_mapping(truth: Mapping[str, object]) -> ComponentTruthMatrix:
    """Validate and freeze the YAML-decoded component truth matrix."""

    if truth.get("schema_version") != TRUTH_SCHEMA_VERSION:
        raise ValueError(f"truth schema_version must equal {TRUTH_SCHEMA_VERSION!r}")
    catalog = _sequence(truth.get("known_edges"), field="known_edges")
    edge_ids: list[str] = []
    for index, raw_edge in enumerate(catalog):
        edge = _mapping(raw_edge, field=f"known_edges[{index}]")
        edge_id = str(edge.get("known_edge_id", "")).strip()
        if not edge_id:
            raise ValueError("each known edge requires known_edge_id")
        edge_ids.append(edge_id)
    scenarios = _mapping(truth.get("scenarios"), field="scenarios")
    records: list[tuple[str, str, str]] = []
    if set(scenarios) != set(REQUIRED_SCENARIOS):
        raise ValueError("truth scenarios must match the seven G1.5 scenarios")
    for scenario in REQUIRED_SCENARIOS:
        components = _mapping(scenarios[scenario], field=f"scenarios.{scenario}")
        if set(components) != set(COMPONENTS):
            raise ValueError(f"scenario {scenario!r} must define all components")
        for component in COMPONENTS:
            records.append((scenario, component, str(components[component])))
    truth_set_id = str(truth.get("truth_set_id", "")).strip()
    return ComponentTruthMatrix(
        truth_set_id=truth_set_id,
        known_edge_ids=tuple(edge_ids),
        truth_records=tuple(records),
    )


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
    component = _mapping(
        config.get("component_truth_gates"), field="component_truth_gates"
    )
    edge_contract = _mapping(
        config.get("known_edge_contract"), field="known_edge_contract"
    )
    decision = _mapping(config.get("decision_policy"), field="decision_policy")
    if bool(decision.get("gate_alone_switches_default")) or bool(
        decision.get("candidate_default_change_allowed")
    ):
        raise ValueError("G1.5 configuration must not switch the default method")
    negative = _sequence(
        secondary.get("negative_scenarios"),
        field="secondary_gates.negative_scenarios",
    )
    return MechanismSpecificitySpecification(
        evaluation_phase=evaluation_phase,
        expected_seed_count=int(cast(Any, phase["required_paired_seeds"])),
        minimum_known_edges=int(cast(Any, edge_contract["minimum_known_edges"])),
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
        component_positive_threshold=float(cast(Any, component["positive_threshold"])),
        component_zero_maximum=float(cast(Any, component["zero_maximum"])),
        minimum_component_conformance_rate=float(
            cast(Any, component["minimum_conformance_rate"])
        ),
    )


def _prepare_records(
    records: pd.DataFrame, truth: ComponentTruthMatrix
) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    missing = REQUIRED_COLUMNS.difference(records.columns)
    if missing:
        raise ValueError(f"component evidence records are missing: {sorted(missing)}")
    table = records.copy(deep=True)
    if table.empty:
        raise ValueError("component evidence records must not be empty")
    for column in ("scenario", "known_edge_id"):
        table[column] = table[column].astype("string")
        if table[column].isna().any() or table[column].eq("").any():
            raise ValueError(f"{column} must contain non-empty identifiers")
    unknown_scenarios = set(table["scenario"].astype(str)).difference(
        REQUIRED_SCENARIOS
    )
    if unknown_scenarios:
        raise ValueError(
            "component evidence contains unknown scenarios: "
            f"{sorted(unknown_scenarios)}"
        )
    unknown_edges = set(table["known_edge_id"].astype(str)).difference(
        truth.known_edge_ids
    )
    if unknown_edges:
        raise ValueError(
            f"component evidence contains unregistered edges: {sorted(unknown_edges)}"
        )
    seeds = pd.to_numeric(table["seed"], errors="coerce")
    if (
        seeds.isna().any()
        or np.isinf(seeds).any()
        or not np.equal(seeds, np.floor(seeds)).all()
    ):
        raise ValueError("seed must contain finite integer identifiers")
    table["seed"] = seeds.astype(np.int64)
    key = ["seed", "scenario", "known_edge_id"]
    if table.duplicated(key).any():
        raise ValueError("component evidence must have one row per seed/scenario/edge")
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


def _row_context(
    specification: MechanismSpecificitySpecification,
    truth: ComponentTruthMatrix,
    *,
    aggregation: str,
    known_edge_id: str,
    scenario: str | None,
    component: str | None,
    truth_code: str | None,
    role: str,
    metric: str,
    eligible_seeds: int,
    eligible_edges: int,
) -> dict[str, object]:
    return {
        "evaluation_phase": specification.evaluation_phase,
        "aggregation": aggregation,
        "known_edge_id": known_edge_id,
        "scenario": scenario,
        "component": component,
        "truth_code": truth_code,
        "gate_role": role,
        "metric": metric,
        "n_expected_seeds": specification.expected_seed_count,
        "n_eligible_seeds": eligible_seeds,
        "n_expected_edges": len(truth.known_edge_ids),
        "n_eligible_edges": eligible_edges,
    }


def _not_estimable_row(
    specification: MechanismSpecificitySpecification,
    truth: ComponentTruthMatrix,
    *,
    aggregation: str,
    known_edge_id: str,
    role: str,
    metric: str,
    reason_code: str,
    eligible_seeds: int,
    eligible_edges: int,
    scenario: str | None = None,
    component: str | None = None,
    truth_code: str | None = None,
) -> dict[str, object]:
    return _row_context(
        specification,
        truth,
        aggregation=aggregation,
        known_edge_id=known_edge_id,
        scenario=scenario,
        component=component,
        truth_code=truth_code,
        role=role,
        metric=metric,
        eligible_seeds=eligible_seeds,
        eligible_edges=eligible_edges,
    ) | {
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
    }


def _observed_row(
    specification: MechanismSpecificitySpecification,
    truth: ComponentTruthMatrix,
    *,
    aggregation: str,
    known_edge_id: str,
    role: str,
    metric: str,
    estimate: float,
    threshold: float,
    operator: str,
    gate_passed: bool | None,
    eligible_seeds: int,
    eligible_edges: int,
    scenario: str | None = None,
    component: str | None = None,
    truth_code: str | None = None,
    ci_lower: float = math.nan,
    ci_upper: float = math.nan,
    confidence_level: float = math.nan,
    interval_type: str | None = None,
    reason_code: str | None = None,
) -> dict[str, object]:
    resolved_reason = reason_code
    if gate_passed is False and resolved_reason is None:
        resolved_reason = "gate_threshold_not_met"
    return _row_context(
        specification,
        truth,
        aggregation=aggregation,
        known_edge_id=known_edge_id,
        scenario=scenario,
        component=component,
        truth_code=truth_code,
        role=role,
        metric=metric,
        eligible_seeds=eligible_seeds,
        eligible_edges=eligible_edges,
    ) | {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "interval_type": interval_type,
        "threshold": threshold,
        "operator": operator,
        "gate_passed": gate_passed,
        "status": "observed",
        "reason_code": resolved_reason,
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
) -> tuple[pd.DataFrame, int]:
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
    complete = (
        len(left_rows) == expected
        and len(right_rows) == expected
        and left_seeds == right_seeds
        and len(paired) == expected
    )
    return (paired if complete else paired.iloc[0:0], len(paired))


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


def _evaluate_standard_scope(
    table: pd.DataFrame,
    specification: MechanismSpecificitySpecification,
    truth: ComponentTruthMatrix,
    *,
    aggregation: str,
    known_edge_id: str,
    eligible_edges: int,
) -> list[dict[str, object]]:
    expected = specification.expected_seed_count
    rows: list[dict[str, object]] = []
    paired, paired_support = _paired_scenarios(
        table, ACTIVE_SCENARIO, LIGAND_ONLY_SCENARIO, expected=expected
    )
    primary_metric = "paired_known_edge_active_minus_ligand_only"
    if len(paired) != expected:
        for role, metric in (
            ("primary", primary_metric),
            ("secondary", "ligand_only_to_active_ratio"),
        ):
            rows.append(
                _not_estimable_row(
                    specification,
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role=role,
                    metric=metric,
                    reason_code=_support_reason(
                        metric=metric,
                        expected=expected,
                        observed=paired_support,
                        paired=True,
                    ),
                    eligible_seeds=paired_support,
                    eligible_edges=eligible_edges,
                    scenario="active_vs_ligand_only",
                    component="integrated",
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
                truth,
                aggregation=aggregation,
                known_edge_id=known_edge_id,
                role="primary",
                metric=primary_metric,
                estimate=estimate,
                threshold=specification.minimum_margin,
                operator=">=_and_ci_lower>",
                gate_passed=primary_passed,
                eligible_seeds=len(paired),
                eligible_edges=eligible_edges,
                scenario="active_vs_ligand_only",
                component="integrated",
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
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role="secondary",
                    metric="ligand_only_to_active_ratio",
                    reason_code="nonpositive_active_mean_for_ratio",
                    eligible_seeds=len(paired),
                    eligible_edges=eligible_edges,
                    scenario="active_vs_ligand_only",
                    component="integrated",
                )
            )
        else:
            ratio = float(paired["integrated_lr_effect_right"].mean()) / active_mean
            rows.append(
                _observed_row(
                    specification,
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role="secondary",
                    metric="ligand_only_to_active_ratio",
                    estimate=ratio,
                    threshold=specification.max_ligand_only_to_active_ratio,
                    operator="<=",
                    gate_passed=(
                        ratio <= specification.max_ligand_only_to_active_ratio
                    ),
                    eligible_seeds=len(paired),
                    eligible_edges=eligible_edges,
                    scenario="active_vs_ligand_only",
                    component="integrated",
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
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role="secondary",
                    metric=metric,
                    reason_code=_support_reason(
                        metric=metric, expected=expected, observed=len(active)
                    ),
                    eligible_seeds=len(active),
                    eligible_edges=eligible_edges,
                    scenario=ACTIVE_SCENARIO,
                    component="integrated",
                )
            )
    else:
        maximum_rank = float(active["known_edge_rank"].max())
        positive_fraction = float(
            pd.to_numeric(
                active["known_edge_positive_direction"], errors="coerce"
            ).mean()
        )
        coverage_loss = float(
            (
                active["reference_comparison_coverage"] - active["comparison_coverage"]
            ).mean()
        )
        rows.extend(
            [
                _observed_row(
                    specification,
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role="secondary",
                    metric="active_max_rank",
                    estimate=maximum_rank,
                    threshold=float(specification.active_max_rank),
                    operator="<=",
                    gate_passed=maximum_rank <= specification.active_max_rank,
                    eligible_seeds=len(active),
                    eligible_edges=eligible_edges,
                    scenario=ACTIVE_SCENARIO,
                    component="integrated",
                ),
                _observed_row(
                    specification,
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role="secondary",
                    metric="active_positive_direction_fraction",
                    estimate=positive_fraction,
                    threshold=(specification.active_min_positive_direction_fraction),
                    operator=">=",
                    gate_passed=(
                        positive_fraction
                        >= specification.active_min_positive_direction_fraction
                    ),
                    eligible_seeds=len(active),
                    eligible_edges=eligible_edges,
                    scenario=ACTIVE_SCENARIO,
                    component="integrated",
                ),
                _observed_row(
                    specification,
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role="secondary",
                    metric="active_comparison_coverage_loss",
                    estimate=coverage_loss,
                    threshold=specification.max_comparison_coverage_loss,
                    operator="<=",
                    gate_passed=(
                        coverage_loss <= specification.max_comparison_coverage_loss
                    ),
                    eligible_seeds=len(active),
                    eligible_edges=eligible_edges,
                    scenario=ACTIVE_SCENARIO,
                    component="integrated",
                ),
            ]
        )

    for scenario in specification.negative_scenarios:
        negative = _observed_scenario(table, scenario)
        negative_seed_set = set(negative["seed"])
        paired_negative = len(active_seed_set.intersection(negative_seed_set))
        metric = f"{scenario}_false_activation_rate"
        if (
            len(active) != expected
            or len(negative) != expected
            or active_seed_set != negative_seed_set
        ):
            rows.append(
                _not_estimable_row(
                    specification,
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role="secondary",
                    metric=metric,
                    reason_code=_support_reason(
                        metric=metric,
                        expected=expected,
                        observed=paired_negative,
                        paired=True,
                    ),
                    eligible_seeds=paired_negative,
                    eligible_edges=eligible_edges,
                    scenario=scenario,
                    component="integrated",
                    truth_code=truth.code(scenario, "integrated"),
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
                truth,
                aggregation=aggregation,
                known_edge_id=known_edge_id,
                role="secondary",
                metric=metric,
                estimate=rate,
                threshold=specification.max_negative_false_activation_rate,
                operator="<=",
                gate_passed=(rate <= specification.max_negative_false_activation_rate),
                eligible_seeds=len(negative),
                eligible_edges=eligible_edges,
                scenario=scenario,
                component="integrated",
                truth_code=truth.code(scenario, "integrated"),
            )
        )

    if specification.receptor_knockout_must_be_zero_or_negative:
        knockout = _observed_scenario(table, RECEPTOR_KNOCKOUT_SCENARIO)
        knockout_seed_set = set(knockout["seed"])
        paired_knockout = len(active_seed_set.intersection(knockout_seed_set))
        metric = "receptor_knockout_mean_integrated_effect"
        if (
            len(active) != expected
            or len(knockout) != expected
            or active_seed_set != knockout_seed_set
        ):
            rows.append(
                _not_estimable_row(
                    specification,
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role="secondary",
                    metric=metric,
                    reason_code=_support_reason(
                        metric=metric,
                        expected=expected,
                        observed=paired_knockout,
                        paired=True,
                    ),
                    eligible_seeds=paired_knockout,
                    eligible_edges=eligible_edges,
                    scenario=RECEPTOR_KNOCKOUT_SCENARIO,
                    component="integrated",
                    truth_code=truth.code(RECEPTOR_KNOCKOUT_SCENARIO, "integrated"),
                )
            )
        else:
            effect = float(knockout["integrated_lr_effect"].mean())
            rows.append(
                _observed_row(
                    specification,
                    truth,
                    aggregation=aggregation,
                    known_edge_id=known_edge_id,
                    role="secondary",
                    metric=metric,
                    estimate=effect,
                    threshold=0.0,
                    operator="<=",
                    gate_passed=effect <= 0.0,
                    eligible_seeds=len(knockout),
                    eligible_edges=eligible_edges,
                    scenario=RECEPTOR_KNOCKOUT_SCENARIO,
                    component="integrated",
                    truth_code=truth.code(RECEPTOR_KNOCKOUT_SCENARIO, "integrated"),
                )
            )
    return rows


def _evaluate_component_truth(
    table: pd.DataFrame,
    specification: MechanismSpecificitySpecification,
    truth: ComponentTruthMatrix,
    *,
    known_edge_id: str,
) -> list[dict[str, object]]:
    expected = specification.expected_seed_count
    rows: list[dict[str, object]] = []
    active = _observed_scenario(table, ACTIVE_SCENARIO)
    active_seeds = set(active["seed"])
    for scenario in REQUIRED_SCENARIOS:
        scenario_rows = _observed_scenario(table, scenario)
        scenario_seeds = set(scenario_rows["seed"])
        paired_support = len(active_seeds.intersection(scenario_seeds))
        complete = (
            len(active) == expected
            and len(scenario_rows) == expected
            and active_seeds == scenario_seeds
        )
        for component in COMPONENTS:
            code = truth.code(scenario, component)
            column = COMPONENT_TO_COLUMN[component]
            if code == "allowed_positive":
                metric = "component_allowed_positive_rate"
                if not complete:
                    rows.append(
                        _not_estimable_row(
                            specification,
                            truth,
                            aggregation="edge",
                            known_edge_id=known_edge_id,
                            role="diagnostic",
                            metric=metric,
                            reason_code=_support_reason(
                                metric=metric,
                                expected=expected,
                                observed=paired_support,
                                paired=True,
                            ),
                            eligible_seeds=paired_support,
                            eligible_edges=1 if paired_support else 0,
                            scenario=scenario,
                            component=component,
                            truth_code=code,
                        )
                    )
                    continue
                positive = scenario_rows[column] > specification.component_zero_maximum
                rate = float(positive.mean())
                rows.append(
                    _observed_row(
                        specification,
                        truth,
                        aggregation="edge",
                        known_edge_id=known_edge_id,
                        role="diagnostic",
                        metric=metric,
                        estimate=rate,
                        threshold=math.nan,
                        operator="diagnostic_only",
                        gate_passed=None,
                        eligible_seeds=len(scenario_rows),
                        eligible_edges=1,
                        scenario=scenario,
                        component=component,
                        truth_code=code,
                        reason_code=(
                            "allowed_positive_is_compatible_but_not_integrated_evidence"
                        ),
                    )
                )
                continue

            metric = "component_truth_conformance_rate"
            if not complete:
                rows.append(
                    _not_estimable_row(
                        specification,
                        truth,
                        aggregation="edge",
                        known_edge_id=known_edge_id,
                        role="component",
                        metric=metric,
                        reason_code=_support_reason(
                            metric=f"{scenario}:{component}",
                            expected=expected,
                            observed=paired_support,
                            paired=True,
                        ),
                        eligible_seeds=paired_support,
                        eligible_edges=1 if paired_support else 0,
                        scenario=scenario,
                        component=component,
                        truth_code=code,
                    )
                )
                if code in {"zero", "ecosystem_only"}:
                    rows.append(
                        _not_estimable_row(
                            specification,
                            truth,
                            aggregation="edge",
                            known_edge_id=known_edge_id,
                            role="diagnostic",
                            metric="component_false_activation_rate",
                            reason_code=_support_reason(
                                metric=f"{scenario}:{component}:false_activation",
                                expected=expected,
                                observed=paired_support,
                                paired=True,
                            ),
                            eligible_seeds=paired_support,
                            eligible_edges=1 if paired_support else 0,
                            scenario=scenario,
                            component=component,
                            truth_code=code,
                        )
                    )
                continue

            values = scenario_rows[column].to_numpy(dtype=float)
            if code == "positive":
                conformance = float(
                    np.mean(values > specification.component_positive_threshold)
                )
            else:
                conformance = float(
                    np.mean(values <= specification.component_zero_maximum)
                )
            rows.append(
                _observed_row(
                    specification,
                    truth,
                    aggregation="edge",
                    known_edge_id=known_edge_id,
                    role="component",
                    metric=metric,
                    estimate=conformance,
                    threshold=specification.minimum_component_conformance_rate,
                    operator=">=",
                    gate_passed=(
                        conformance >= specification.minimum_component_conformance_rate
                    ),
                    eligible_seeds=len(scenario_rows),
                    eligible_edges=1,
                    scenario=scenario,
                    component=component,
                    truth_code=code,
                )
            )
            if code in {"zero", "ecosystem_only"}:
                false_activation = float(
                    np.mean(values > specification.component_zero_maximum)
                )
                reason = (
                    "ecosystem_change_allowed_but_state_availability_must_not_activate"
                    if code == "ecosystem_only"
                    else None
                )
                rows.append(
                    _observed_row(
                        specification,
                        truth,
                        aggregation="edge",
                        known_edge_id=known_edge_id,
                        role="diagnostic",
                        metric="component_false_activation_rate",
                        estimate=false_activation,
                        threshold=0.0,
                        operator="<=",
                        gate_passed=None,
                        eligible_seeds=len(scenario_rows),
                        eligible_edges=1,
                        scenario=scenario,
                        component=component,
                        truth_code=code,
                        reason_code=reason,
                    )
                )
    return rows


def _macro_scenario(
    table: pd.DataFrame,
    truth: ComponentTruthMatrix,
    scenario: str,
    *,
    expected: int,
    required_seed_set: set[int] | None,
) -> pd.DataFrame:
    selected = table.loc[
        table["scenario"].eq(scenario) & table["status"].eq("observed")
    ]
    seed_sets: list[set[int]] = []
    for edge_id in truth.known_edge_ids:
        edge_rows = selected.loc[selected["known_edge_id"].eq(edge_id)]
        if len(edge_rows) != expected:
            return selected.iloc[0:0]
        seed_sets.append(set(edge_rows["seed"]))
    if not seed_sets or any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
        return selected.iloc[0:0]
    if required_seed_set is not None and seed_sets[0] != required_seed_set:
        return selected.iloc[0:0]
    grouped = selected.groupby("seed", sort=True, observed=True)
    macro = grouped[
        [
            *COMPONENT_COLUMNS,
            "comparison_coverage",
            "reference_comparison_coverage",
        ]
    ].mean()
    macro["known_edge_rank"] = grouped["known_edge_rank"].max()
    macro["known_edge_positive_direction"] = grouped[
        "known_edge_positive_direction"
    ].mean()
    macro = macro.reset_index()
    macro["scenario"] = scenario
    macro["known_edge_id"] = MACRO_EDGE_ID
    macro["status"] = "observed"
    macro["reason_code"] = None
    return macro


def _macro_table(
    table: pd.DataFrame,
    truth: ComponentTruthMatrix,
    *,
    expected: int,
) -> pd.DataFrame:
    active = _macro_scenario(
        table,
        truth,
        ACTIVE_SCENARIO,
        expected=expected,
        required_seed_set=None,
    )
    active_seeds = set(active["seed"]) if len(active) == expected else set()
    parts = [active]
    for scenario in REQUIRED_SCENARIOS[1:]:
        parts.append(
            _macro_scenario(
                table,
                truth,
                scenario,
                expected=expected,
                required_seed_set=active_seeds,
            )
        )
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame(columns=table.columns)
    return pd.concat(nonempty, ignore_index=True)


def _macro_component_rows(
    edge_rows: pd.DataFrame,
    specification: MechanismSpecificitySpecification,
    truth: ComponentTruthMatrix,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected = edge_rows.loc[
        edge_rows["gate_role"].isin(["component", "diagnostic"])
        & edge_rows["aggregation"].eq("edge")
    ]
    keys = ["scenario", "component", "truth_code", "gate_role", "metric"]
    for key, group in selected.groupby(keys, sort=True, dropna=False, observed=True):
        scenario, component, truth_code, role, metric = map(str, key)
        complete = (
            len(group) == len(truth.known_edge_ids)
            and group["known_edge_id"].nunique() == len(truth.known_edge_ids)
            and group["status"].eq("observed").all()
        )
        if not complete:
            eligible = int(
                group.loc[group["status"].eq("observed"), "known_edge_id"].nunique()
            )
            rows.append(
                _not_estimable_row(
                    specification,
                    truth,
                    aggregation="macro_equal_edge",
                    known_edge_id=MACRO_EDGE_ID,
                    role=role,
                    metric=metric,
                    reason_code=(
                        "insufficient_known_edge_support:"
                        f"expected={len(truth.known_edge_ids)}:observed={eligible}"
                    ),
                    eligible_seeds=0,
                    eligible_edges=eligible,
                    scenario=scenario,
                    component=component,
                    truth_code=truth_code,
                )
            )
            continue
        estimate = float(group["estimate"].mean())
        if role == "component":
            passed = bool(group["gate_passed"].eq(True).all())
            threshold = specification.minimum_component_conformance_rate
            operator = "all_edges_and_macro_mean>="
            reason = None
        else:
            passed = None
            threshold = math.nan
            operator = "diagnostic_only"
            reasons = group["reason_code"].dropna().astype(str).unique()
            reason = str(reasons[0]) if len(reasons) == 1 else None
        rows.append(
            _observed_row(
                specification,
                truth,
                aggregation="macro_equal_edge",
                known_edge_id=MACRO_EDGE_ID,
                role=role,
                metric=metric,
                estimate=estimate,
                threshold=threshold,
                operator=operator,
                gate_passed=passed,
                eligible_seeds=specification.expected_seed_count,
                eligible_edges=len(truth.known_edge_ids),
                scenario=scenario,
                component=component,
                truth_code=truth_code,
                reason_code=reason,
            )
        )
    return rows


def evaluate_mechanism_specificity(
    records: pd.DataFrame,
    specification: MechanismSpecificitySpecification,
    truth: ComponentTruthMatrix,
) -> pd.DataFrame:
    """Evaluate per-edge and equal-edge macro G1.5 gates with no imputation."""

    if not isinstance(specification, MechanismSpecificitySpecification):
        raise TypeError("specification must be MechanismSpecificitySpecification")
    if not isinstance(truth, ComponentTruthMatrix):
        raise TypeError("truth must be ComponentTruthMatrix")
    if len(truth.known_edge_ids) < specification.minimum_known_edges:
        raise ValueError(
            "component truth has fewer known edges than the frozen minimum"
        )
    table = _prepare_records(records, truth)
    rows: list[dict[str, object]] = []
    for edge_id in truth.known_edge_ids:
        edge = table.loc[table["known_edge_id"].eq(edge_id)].copy()
        rows.extend(
            _evaluate_standard_scope(
                edge,
                specification,
                truth,
                aggregation="edge",
                known_edge_id=edge_id,
                eligible_edges=1 if not edge.empty else 0,
            )
        )
        rows.extend(
            _evaluate_component_truth(
                edge,
                specification,
                truth,
                known_edge_id=edge_id,
            )
        )

    edge_results = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    macro = _macro_table(table, truth, expected=specification.expected_seed_count)
    rows.extend(
        _evaluate_standard_scope(
            macro,
            specification,
            truth,
            aggregation="macro_equal_edge",
            known_edge_id=MACRO_EDGE_ID,
            eligible_edges=(len(truth.known_edge_ids) if not macro.empty else 0),
        )
    )
    rows.extend(_macro_component_rows(edge_results, specification, truth))

    gate_rows = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    required = gate_rows.loc[
        gate_rows["gate_role"].isin(["primary", "secondary", "component"])
    ]
    unavailable = required.loc[required["status"].ne("observed")]
    if not unavailable.empty:
        unavailable_ids = (
            unavailable["known_edge_id"].astype(str)
            + ":"
            + unavailable["metric"].astype(str)
        ).tolist()
        overall = _not_estimable_row(
            specification,
            truth,
            aggregation="overall",
            known_edge_id=OVERALL_EDGE_ID,
            role="overall",
            metric="g1_5_mechanism_specificity_gate",
            reason_code="required_gate_not_estimable:" + ",".join(unavailable_ids),
            eligible_seeds=0,
            eligible_edges=int(
                required.loc[
                    required["status"].eq("observed"), "known_edge_id"
                ].nunique()
            ),
        )
    else:
        failed = required.loc[required["gate_passed"].eq(False)]
        failed_ids = (
            failed["known_edge_id"].astype(str) + ":" + failed["metric"].astype(str)
        ).tolist()
        overall = _observed_row(
            specification,
            truth,
            aggregation="overall",
            known_edge_id=OVERALL_EDGE_ID,
            role="overall",
            metric="g1_5_mechanism_specificity_gate",
            estimate=0.0 if failed_ids else 1.0,
            threshold=1.0,
            operator="all_required_edge_macro_and_component_gates",
            gate_passed=not failed_ids,
            eligible_seeds=specification.expected_seed_count,
            eligible_edges=len(truth.known_edge_ids),
            reason_code=("failed_gate:" + ",".join(failed_ids) if failed_ids else None),
        )
    return pd.concat(
        [gate_rows, pd.DataFrame([overall], columns=OUTPUT_COLUMNS)],
        ignore_index=True,
    )
