"""Preregistered paired-seed metrics for the graph-fusion G2 gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    roc_auc_score,
)

G2_METRIC_SCHEMA_VERSION = "crychic-g2-graph-fusion-metrics-v1"
PRIMARY_METRIC = "driver_family_x_context_macro_auprc"
UNFUSED_METHOD = "unfused"
FUSED_METHOD = "fused_correct"
NO_TOPOLOGY_METHOD = "fused_no_topology"
WRONG_TOPOLOGY_METHOD = "fused_wrong_topology"
METHODS = (
    UNFUSED_METHOD,
    FUSED_METHOD,
    NO_TOPOLOGY_METHOD,
    WRONG_TOPOLOGY_METHOD,
)
GateStatus = Literal["PASS", "FAIL", "NE"]

_REQUIRED_RESULT_COLUMNS = {
    "scenario_id",
    "replicate",
    "scenario_seed",
    "method",
    "macro_auprc",
    "status",
    "reason_code",
}


def _binary_matrices(
    truth: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(truth)
    values = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 2 or values.shape != labels.shape:
        raise ValueError("truth and scores must share a context x family shape")
    if np.any(~np.isfinite(values)):
        raise ValueError("family recovery scores must be finite")
    if not np.isin(labels, (False, True, 0, 1)).all():
        raise ValueError("family recovery truth must be binary")
    return labels.astype(bool), values


def driver_family_context_macro_auprc(
    truth: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Return equal-context average precision over driver-family labels."""

    labels, values = _binary_matrices(truth, scores)
    context_scores: list[float] = []
    for context_index in range(labels.shape[0]):
        context_truth = labels[context_index]
        if context_truth.all() or not context_truth.any():
            raise ValueError(
                "each context requires positive and negative driver-family truth"
            )
        context_scores.append(
            float(
                average_precision_score(
                    context_truth.astype(np.int8), values[context_index]
                )
            )
        )
    return float(np.mean(context_scores))


def driver_family_context_macro_auroc(
    truth: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Return the secondary equal-context family recovery AUROC."""

    labels, values = _binary_matrices(truth, scores)
    context_scores: list[float] = []
    for context_index in range(labels.shape[0]):
        context_truth = labels[context_index]
        if context_truth.all() or not context_truth.any():
            raise ValueError(
                "each context requires positive and negative driver-family truth"
            )
        context_scores.append(
            float(roc_auc_score(context_truth, values[context_index]))
        )
    return float(np.mean(context_scores))


def graph_jump_localization_auprc(
    truth_coefficients: np.ndarray,
    estimated_coefficients: np.ndarray,
    edge_indices: tuple[tuple[int, int], ...],
    *,
    truth_tolerance: float = 1e-12,
) -> float | None:
    """Score biological-edge jump localization as a secondary endpoint."""

    truth = np.asarray(truth_coefficients, dtype=np.float64)
    estimated = np.asarray(estimated_coefficients, dtype=np.float64)
    if truth.ndim != 2 or estimated.shape != truth.shape:
        raise ValueError("jump inputs must share a context x family shape")
    if np.any(~np.isfinite(truth)) or np.any(~np.isfinite(estimated)):
        raise ValueError("jump inputs must be finite")
    if not edge_indices:
        return None
    labels: list[bool] = []
    scores: list[float] = []
    for left, right in edge_indices:
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or left < 0
            or right < 0
            or left >= truth.shape[0]
            or right >= truth.shape[0]
            or left == right
        ):
            raise ValueError("jump edges contain invalid context indices")
        true_difference = np.abs(truth[left] - truth[right])
        estimated_difference = np.abs(estimated[left] - estimated[right])
        labels.append(bool(np.any(true_difference > truth_tolerance)))
        scores.append(float(np.sum(estimated_difference)))
    binary = np.asarray(labels, dtype=bool)
    if binary.all() or not binary.any():
        return None
    return float(average_precision_score(binary.astype(np.int8), scores))


@dataclass(frozen=True, slots=True, kw_only=True)
class G2GateSpecification:
    """Frozen primary aggregation, margins, and paired bootstrap policy."""

    scenario_ids: tuple[str, ...]
    minimum_paired_replicates_per_scenario: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float = 0.95
    improvement_margin: float = 0.02
    noninferiority_margin: float = -0.02

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenario_ids)
        if (
            not scenarios
            or len(set(scenarios)) != len(scenarios)
            or any(not value for value in scenarios)
        ):
            raise ValueError("scenario_ids must be non-empty and unique")
        if self.minimum_paired_replicates_per_scenario < 2:
            raise ValueError("G2 gate requires at least two paired replicates per cell")
        if self.bootstrap_replicates < 100:
            raise ValueError("paired bootstrap requires at least 100 replicates")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(
            self.bootstrap_seed, int
        ):
            raise ValueError("bootstrap_seed must be an integer")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")
        for field_name in ("improvement_margin", "noninferiority_margin"):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        object.__setattr__(self, "scenario_ids", scenarios)


@dataclass(frozen=True, slots=True, kw_only=True)
class PairedBootstrapInterval:
    """Scenario-equal point estimate and paired-seed percentile interval."""

    comparison: str
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    bootstrap_replicates: int
    scenario_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison": self.comparison,
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "bootstrap_replicates": self.bootstrap_replicates,
            "scenario_count": self.scenario_count,
            "aggregation": "paired_seed_within_scenario_then_equal_scenario_cell",
            "interval_type": "paired_seed_stratified_percentile_bootstrap",
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class G2GateReport:
    """Machine-readable G2 decision that cannot treat insufficient smoke as pass."""

    status: GateStatus
    reason_code: str
    primary_interval: PairedBootstrapInterval | None
    control_intervals: tuple[PairedBootstrapInterval, ...]
    complete_pairs_by_comparison: dict[str, dict[str, int]]
    specification: G2GateSpecification

    @property
    def gate_passed(self) -> bool | None:
        if self.status == "NE":
            return None
        return self.status == "PASS"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": G2_METRIC_SCHEMA_VERSION,
            "status": self.status,
            "gate_passed": self.gate_passed,
            "reason_code": self.reason_code,
            "primary_metric": PRIMARY_METRIC,
            "primary_interval": (
                None
                if self.primary_interval is None
                else self.primary_interval.to_dict()
            ),
            "control_intervals": [
                interval.to_dict() for interval in self.control_intervals
            ],
            "complete_pairs_by_comparison": self.complete_pairs_by_comparison,
            "minimum_paired_replicates_per_scenario": (
                self.specification.minimum_paired_replicates_per_scenario
            ),
            "improvement_margin": self.specification.improvement_margin,
            "noninferiority_margin": self.specification.noninferiority_margin,
            "confidence_level": self.specification.confidence_level,
        }


def _validate_results(results: pd.DataFrame, spec: G2GateSpecification) -> pd.DataFrame:
    if not isinstance(results, pd.DataFrame):
        raise TypeError("G2 results must be a pandas DataFrame")
    missing = _REQUIRED_RESULT_COLUMNS.difference(results.columns)
    if missing:
        raise ValueError(f"G2 results are missing columns: {sorted(missing)}")
    frame = results.copy(deep=True)
    if frame.empty:
        return frame
    unknown_scenarios = set(frame["scenario_id"].astype(str)).difference(
        spec.scenario_ids
    )
    if unknown_scenarios:
        raise ValueError(f"G2 results contain unknown scenarios: {unknown_scenarios}")
    unknown_methods = set(frame["method"].astype(str)).difference(METHODS)
    if unknown_methods:
        raise ValueError(f"G2 results contain unknown methods: {unknown_methods}")
    if frame.duplicated(["scenario_id", "replicate", "method"]).any():
        raise ValueError("G2 result grain must be scenario x replicate x method")
    seed_grain = frame.loc[
        :, ["scenario_id", "replicate", "scenario_seed"]
    ].drop_duplicates()
    if seed_grain.duplicated(["scenario_id", "replicate"]).any():
        raise ValueError("all paired methods must share one scenario_seed")
    if seed_grain.duplicated(["scenario_id", "scenario_seed"]).any():
        raise ValueError("scenario_seed must be unique across replicates in one cell")
    observed = frame["status"].astype(str).eq("observed")
    observed_values = pd.to_numeric(
        frame.loc[observed, "macro_auprc"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any(~np.isfinite(observed_values)) or np.any(
        (observed_values < 0) | (observed_values > 1)
    ):
        raise ValueError("observed macro_auprc values must be finite in [0, 1]")
    return frame


def _paired_differences(
    frame: pd.DataFrame,
    spec: G2GateSpecification,
    method: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    differences: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for scenario_id in spec.scenario_ids:
        selected = frame.loc[
            frame["scenario_id"].astype(str).eq(scenario_id)
            & frame["method"].astype(str).isin((UNFUSED_METHOD, method))
        ]
        paired: list[float] = []
        for _, group in selected.groupby("replicate", sort=True):
            by_method = {
                str(row.method): row for row in group.itertuples(index=False)
            }
            if set(by_method) != {UNFUSED_METHOD, method}:
                continue
            reference = by_method[UNFUSED_METHOD]
            candidate = by_method[method]
            if int(cast(Any, reference.scenario_seed)) != int(
                cast(Any, candidate.scenario_seed)
            ):
                raise ValueError("G2 paired methods must share one scenario_seed")
            if (
                str(reference.status) != "observed"
                or str(candidate.status) != "observed"
            ):
                continue
            reference_score = float(cast(Any, reference.macro_auprc))
            candidate_score = float(cast(Any, candidate.macro_auprc))
            if not math.isfinite(reference_score) or not math.isfinite(candidate_score):
                continue
            paired.append(candidate_score - reference_score)
        differences[scenario_id] = np.asarray(paired, dtype=np.float64)
        counts[scenario_id] = len(paired)
    return differences, counts


def _bootstrap_interval(
    differences: dict[str, np.ndarray],
    spec: G2GateSpecification,
    *,
    comparison: str,
    seed_offset: int,
) -> PairedBootstrapInterval:
    if any(len(differences[scenario]) == 0 for scenario in spec.scenario_ids):
        raise ValueError("bootstrap requires at least one pair in every scenario")
    estimate = float(
        np.mean(
            [float(np.mean(differences[scenario])) for scenario in spec.scenario_ids]
        )
    )
    rng = np.random.default_rng(spec.bootstrap_seed + seed_offset)
    samples: NDArray[np.float64] = np.empty(
        spec.bootstrap_replicates, dtype=np.float64
    )
    for bootstrap_index in range(spec.bootstrap_replicates):
        scenario_means: list[float] = []
        for scenario in spec.scenario_ids:
            values = differences[scenario]
            indices = rng.integers(0, len(values), size=len(values))
            scenario_means.append(float(np.mean(values[indices])))
        samples[bootstrap_index] = float(np.mean(scenario_means))
    alpha = (1.0 - spec.confidence_level) / 2.0
    quantiles: NDArray[np.float64] = np.asarray(
        np.quantile(samples, (alpha, 1.0 - alpha)), dtype=np.float64
    )
    return PairedBootstrapInterval(
        comparison=comparison,
        estimate=estimate,
        ci_lower=float(quantiles[0]),
        ci_upper=float(quantiles[1]),
        confidence_level=spec.confidence_level,
        bootstrap_replicates=spec.bootstrap_replicates,
        scenario_count=len(spec.scenario_ids),
    )


def evaluate_graph_fusion_g2_gate(
    results: pd.DataFrame,
    specification: G2GateSpecification,
) -> G2GateReport:
    """Evaluate the preregistered G2 superiority and topology controls."""

    if not isinstance(specification, G2GateSpecification):
        raise TypeError("specification must be a G2GateSpecification")
    frame = _validate_results(results, specification)
    comparisons = (
        (FUSED_METHOD, "fused_correct_minus_unfused"),
        (NO_TOPOLOGY_METHOD, "no_topology_minus_unfused"),
        (WRONG_TOPOLOGY_METHOD, "wrong_topology_minus_unfused"),
    )
    paired: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, dict[str, int]] = {}
    for method, comparison in comparisons:
        differences, observed_counts = _paired_differences(
            frame, specification, method
        )
        paired[comparison] = differences
        counts[comparison] = observed_counts
    required = specification.minimum_paired_replicates_per_scenario
    if any(
        count < required
        for comparison_counts in counts.values()
        for count in comparison_counts.values()
    ):
        return G2GateReport(
            status="NE",
            reason_code="insufficient_complete_paired_replicates_per_scenario",
            primary_interval=None,
            control_intervals=(),
            complete_pairs_by_comparison=counts,
            specification=specification,
        )

    primary = _bootstrap_interval(
        paired["fused_correct_minus_unfused"],
        specification,
        comparison="fused_correct_minus_unfused",
        seed_offset=0,
    )
    no_topology = _bootstrap_interval(
        paired["no_topology_minus_unfused"],
        specification,
        comparison="no_topology_minus_unfused",
        seed_offset=1,
    )
    wrong_topology = _bootstrap_interval(
        paired["wrong_topology_minus_unfused"],
        specification,
        comparison="wrong_topology_minus_unfused",
        seed_offset=2,
    )
    passed = (
        primary.ci_lower >= specification.improvement_margin
        and no_topology.ci_lower >= specification.noninferiority_margin
        and wrong_topology.ci_lower >= specification.noninferiority_margin
    )
    return G2GateReport(
        status="PASS" if passed else "FAIL",
        reason_code=(
            "all_preregistered_g2_margins_met"
            if passed
            else "one_or_more_preregistered_g2_margins_not_met"
        ),
        primary_interval=primary,
        control_intervals=(no_topology, wrong_topology),
        complete_pairs_by_comparison=counts,
        specification=specification,
    )


__all__ = [
    "FUSED_METHOD",
    "G2_METRIC_SCHEMA_VERSION",
    "METHODS",
    "NO_TOPOLOGY_METHOD",
    "PRIMARY_METRIC",
    "UNFUSED_METHOD",
    "WRONG_TOPOLOGY_METHOD",
    "G2GateReport",
    "G2GateSpecification",
    "PairedBootstrapInterval",
    "driver_family_context_macro_auprc",
    "driver_family_context_macro_auroc",
    "evaluate_graph_fusion_g2_gate",
    "graph_jump_localization_auprc",
]
