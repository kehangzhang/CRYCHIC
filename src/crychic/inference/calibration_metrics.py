"""Strict binary-calibration primitives for G3-P release campaigns.

The functions in this module compute numeric evidence only.  They do not own
the ADR-013 release decision or persistence contract.  Calibration rows are
canonicalized by replicate and observation identifier so every deterministic
result, including seeded cluster bootstrap output, is invariant to input row
order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import beta as beta_distribution

_DEFAULT_PROBABILITY_CLIP = 1e-6
_DEFAULT_NEWTON_TOLERANCE = 1e-8
_DEFAULT_CURVATURE_TOLERANCE = 1e-10
_DEFAULT_MAX_ITERATIONS = 100
_MAX_ABSOLUTE_COEFFICIENT = 40.0

FloatVector: TypeAlias = NDArray[np.float64]


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _probability(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be a finite probability") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and lie in [0, 1]")
    return 0.0 if result == 0.0 else result


def _binary(value: object, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return int(value)
    raise ValueError(f"{field_name} must be exactly binary (0 or 1)")


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer >= 1")
    result = int(value)
    if result < 1:
        raise ValueError(f"{field_name} must be an integer >= 1")
    return result


def _confidence_level(value: object) -> float:
    result = _probability(value, field_name="confidence_level")
    if not 0.5 < result < 1.0:
        raise ValueError("confidence_level must lie strictly between 0.5 and 1")
    return result


def _clip_epsilon(value: object) -> float:
    result = _probability(value, field_name="probability_clip")
    if not 0.0 < result < 0.5:
        raise ValueError("probability_clip must lie strictly between 0 and 0.5")
    return result


class CalibrationNotEstimableError(ValueError):
    """A calibration estimand is undefined; no numeric placeholder is returned."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = _name(reason_code, field_name="reason_code")


@dataclass(frozen=True, slots=True, kw_only=True)
class BinaryCalibrationRecord:
    """One known binary truth and predicted probability in one replicate."""

    replicate_id: str
    observation_id: str
    outcome: int
    probability: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "replicate_id", _name(self.replicate_id, field_name="replicate_id")
        )
        object.__setattr__(
            self,
            "observation_id",
            _name(self.observation_id, field_name="observation_id"),
        )
        object.__setattr__(self, "outcome", _binary(self.outcome, field_name="outcome"))
        object.__setattr__(
            self,
            "probability",
            _probability(self.probability, field_name="probability"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationBin:
    """One fixed-width probability bin; empty bins carry no pseudo-statistic."""

    index: int
    lower: float
    upper: float
    includes_upper: bool
    n_observations: int
    mean_probability: float | None
    observed_frequency: float | None
    absolute_gap: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplicateFixedBinECE:
    """Fixed-width ECE computed within one simulated replicate."""

    replicate_id: str
    n_bins: int
    n_observations: int
    bins: tuple[CalibrationBin, ...]
    ece: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedBinECE:
    """Equal-replicate mean of deterministic within-replicate ECE values."""

    n_bins: int
    n_observations: int
    n_replicates: int
    replicates: tuple[ReplicateFixedBinECE, ...]
    ece: float
    binning_semantics: str = "equal_width_left_closed_last_bin_right_closed_v1"
    aggregation_semantics: str = "within_replicate_then_equal_replicate_mean_v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class LogisticRecalibration:
    """Conventional logistic calibration intercept and slope diagnostics.

    ``calibration_in_the_large`` is the intercept in a logistic model with the
    clipped prediction logit as an offset (slope fixed to one).  ``slope`` is
    estimated jointly with ``intercept`` in a second unpenalized logistic
    recalibration model.
    """

    calibration_in_the_large: float
    intercept: float
    slope: float
    probability_clip: float
    n_clipped_low: int
    n_clipped_high: int
    n_iterations: int
    gradient_infinity_norm: float
    curvature_minimum_eigenvalue: float
    semantics: str = (
        "equal_replicate_weighted_logistic_offset_citl_and_intercept_slope_v1"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class BinaryCalibrationMetrics:
    """Point calibration metrics for one complete campaign cell."""

    n_observations: int
    n_replicates: int
    observed_prevalence: float
    baseline_probability: float
    brier_score: float
    prevalence_only_brier_score: float
    brier_relative_improvement: float
    fixed_bin_ece: FixedBinECE
    logistic_recalibration: LogisticRecalibration
    aggregation_semantics: str = "within_replicate_then_equal_replicate_mean_v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class ClusterBootstrapECE:
    """One-sided percentile upper bound from whole-replicate resampling."""

    ece: float
    upper_bound: float
    confidence_level: float
    n_resamples: int
    n_replicates: int
    seed: int
    n_bins: int
    resampling_semantics: str = (
        "whole_replicate_cluster_percentile_upper_order_statistic_v1"
    )


class BinomialBoundSide(StrEnum):
    """Direction of an exact one-sided binomial confidence bound."""

    LOWER = "lower"
    UPPER = "upper"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactBinomialBound:
    """Exact one-sided Clopper-Pearson bound for a binomial proportion."""

    successes: int
    trials: int
    estimate: float
    side: BinomialBoundSide
    confidence_level: float
    bound: float
    semantics: str = "exact_clopper_pearson_one_sided_v1"


def _canonical_records(
    records: Sequence[BinaryCalibrationRecord],
) -> tuple[BinaryCalibrationRecord, ...]:
    values = tuple(records)
    if not values:
        raise ValueError("records must contain at least one calibration observation")
    if any(not isinstance(item, BinaryCalibrationRecord) for item in values):
        raise TypeError("records must contain BinaryCalibrationRecord values")
    keys = [(item.replicate_id, item.observation_id) for item in values]
    if len(keys) != len(set(keys)):
        raise ValueError("records must have unique (replicate_id, observation_id) keys")
    return tuple(
        sorted(values, key=lambda item: (item.replicate_id, item.observation_id))
    )


def _vectors(
    records: Sequence[BinaryCalibrationRecord],
) -> tuple[tuple[BinaryCalibrationRecord, ...], FloatVector, FloatVector]:
    canonical = _canonical_records(records)
    outcomes = np.fromiter(
        (item.outcome for item in canonical), dtype=np.float64, count=len(canonical)
    )
    probabilities = np.fromiter(
        (item.probability for item in canonical),
        dtype=np.float64,
        count=len(canonical),
    )
    return canonical, outcomes, probabilities


def _replicate_groups(
    records: tuple[BinaryCalibrationRecord, ...],
) -> tuple[tuple[str, tuple[BinaryCalibrationRecord, ...]], ...]:
    return tuple(
        (replicate_id, tuple(group))
        for replicate_id, group in groupby(records, key=lambda item: item.replicate_id)
    )


def _equal_replicate_weights(
    records: tuple[BinaryCalibrationRecord, ...],
) -> FloatVector:
    groups = _replicate_groups(records)
    counts = {replicate_id: len(group) for replicate_id, group in groups}
    n_replicates = len(groups)
    weights: FloatVector = np.fromiter(
        (1.0 / (n_replicates * counts[item.replicate_id]) for item in records),
        dtype=np.float64,
        count=len(records),
    )
    return weights


def _one_replicate_fixed_bin_ece(
    replicate_id: str,
    records: tuple[BinaryCalibrationRecord, ...],
    *,
    n_bins: int,
) -> ReplicateFixedBinECE:
    outcomes = np.fromiter(
        (item.outcome for item in records), dtype=np.float64, count=len(records)
    )
    probabilities = np.fromiter(
        (item.probability for item in records),
        dtype=np.float64,
        count=len(records),
    )
    indices = np.minimum(np.floor(probabilities * n_bins).astype(np.int64), n_bins - 1)
    output: list[CalibrationBin] = []
    weighted_gap = 0.0
    for index in range(n_bins):
        mask = indices == index
        count = int(np.count_nonzero(mask))
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if count == 0:
            mean_probability = None
            observed_frequency = None
            gap = None
        else:
            mean_probability = float(np.mean(probabilities[mask]))
            observed_frequency = float(np.mean(outcomes[mask]))
            gap = abs(mean_probability - observed_frequency)
            weighted_gap += (count / len(probabilities)) * gap
        output.append(
            CalibrationBin(
                index=index,
                lower=lower,
                upper=upper,
                includes_upper=index == n_bins - 1,
                n_observations=count,
                mean_probability=mean_probability,
                observed_frequency=observed_frequency,
                absolute_gap=gap,
            )
        )
    return ReplicateFixedBinECE(
        replicate_id=replicate_id,
        n_bins=n_bins,
        n_observations=len(records),
        bins=tuple(output),
        ece=float(weighted_gap),
    )


def compute_fixed_bin_ece(
    records: Sequence[BinaryCalibrationRecord],
    *,
    n_bins: int = 10,
) -> FixedBinECE:
    """Compute ECE within each replicate, then average replicates equally."""

    bins_count = _positive_integer(n_bins, field_name="n_bins")
    canonical = _canonical_records(records)
    replicates = tuple(
        _one_replicate_fixed_bin_ece(
            replicate_id,
            group,
            n_bins=bins_count,
        )
        for replicate_id, group in _replicate_groups(canonical)
    )
    return FixedBinECE(
        n_bins=bins_count,
        n_observations=len(canonical),
        n_replicates=len(replicates),
        replicates=replicates,
        ece=float(np.mean([item.ece for item in replicates])),
    )


def _require_two_outcome_classes(outcomes: FloatVector) -> None:
    if not np.any(outcomes == 0.0) or not np.any(outcomes == 1.0):
        raise CalibrationNotEstimableError(
            "logistic calibration requires both outcome classes",
            reason_code="calibration_single_outcome_class",
        )


def _calibration_in_the_large(
    logits: FloatVector,
    outcomes: FloatVector,
    observation_weights: FloatVector,
) -> float:
    prevalence = float(observation_weights @ outcomes)

    def score(intercept: float) -> float:
        return float(observation_weights @ expit(intercept + logits) - prevalence)

    lower = -_MAX_ABSOLUTE_COEFFICIENT
    upper = _MAX_ABSOLUTE_COEFFICIENT
    if score(lower) >= 0.0 or score(upper) <= 0.0:
        raise CalibrationNotEstimableError(
            "calibration-in-the-large root is not finite within safety bounds",
            reason_code="calibration_citl_unbounded",
        )
    result = float(brentq(score, lower, upper, xtol=1e-12, rtol=1e-14))
    if not math.isfinite(result) or abs(result) >= _MAX_ABSOLUTE_COEFFICIENT:
        raise CalibrationNotEstimableError(
            "calibration-in-the-large is non-finite or unbounded",
            reason_code="calibration_citl_unbounded",
        )
    return 0.0 if result == 0.0 else result


def _logistic_recalibration_fit(
    logits: FloatVector,
    outcomes: FloatVector,
    observation_weights: FloatVector,
    *,
    tolerance: float,
    curvature_tolerance: float,
    max_iterations: int,
) -> tuple[float, float, int, float, float]:
    center = float(observation_weights @ logits)
    scale = float(np.sqrt(observation_weights @ np.square(logits - center)))
    if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise CalibrationNotEstimableError(
            "calibration slope requires non-constant prediction logits",
            reason_code="calibration_constant_prediction",
        )
    zero_logits = logits[outcomes == 0.0]
    one_logits = logits[outcomes == 1.0]
    if float(np.max(zero_logits)) <= float(np.min(one_logits)) or float(
        np.min(zero_logits)
    ) >= float(np.max(one_logits)):
        raise CalibrationNotEstimableError(
            "calibration slope has complete or quasi-complete separation",
            reason_code="calibration_slope_separation",
        )
    standardized = (logits - center) / scale
    design = np.column_stack((np.ones(len(logits), dtype=np.float64), standardized))
    prevalence = float(observation_weights @ outcomes)
    theta = np.array([math.log(prevalence / (1.0 - prevalence)), 0.0], dtype=np.float64)

    def objective(coefficients: FloatVector) -> float:
        eta = design @ coefficients
        return float(observation_weights @ (np.logaddexp(0.0, eta) - outcomes * eta))

    gradient_norm = math.inf
    curvature_minimum = math.nan
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        eta = design @ theta
        fitted = expit(eta)
        gradient = design.T @ (observation_weights * (fitted - outcomes))
        gradient_norm = float(np.max(np.abs(gradient)))
        hessian_weights = observation_weights * fitted * (1.0 - fitted)
        hessian = design.T @ (design * hessian_weights[:, None])
        eigenvalues = np.linalg.eigvalsh(hessian)
        curvature_minimum = float(eigenvalues[0])
        if (
            not np.all(np.isfinite(theta))
            or not np.all(np.isfinite(gradient))
            or not np.all(np.isfinite(hessian))
            or curvature_minimum <= curvature_tolerance
        ):
            raise CalibrationNotEstimableError(
                "logistic recalibration has singular curvature",
                reason_code="calibration_slope_nonidentifiable",
            )
        if gradient_norm <= tolerance:
            break
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise CalibrationNotEstimableError(
                "logistic recalibration Hessian is singular",
                reason_code="calibration_slope_nonidentifiable",
            ) from error
        descent = float(gradient @ step)
        current = objective(theta)
        step_fraction = 1.0
        accepted = False
        while step_fraction >= 2.0**-24:
            candidate = theta - step_fraction * step
            if objective(candidate) <= current - 1e-4 * step_fraction * descent:
                theta = candidate
                accepted = True
                break
            step_fraction *= 0.5
        if not accepted:
            raise CalibrationNotEstimableError(
                "logistic recalibration line search did not converge",
                reason_code="calibration_slope_nonconvergent",
            )
    else:
        raise CalibrationNotEstimableError(
            "logistic recalibration exceeded max_iterations",
            reason_code="calibration_slope_nonconvergent",
        )

    slope = float(theta[1] / scale)
    intercept = float(theta[0] - slope * center)
    if (
        not math.isfinite(intercept)
        or not math.isfinite(slope)
        or abs(intercept) >= _MAX_ABSOLUTE_COEFFICIENT
        or abs(slope) >= _MAX_ABSOLUTE_COEFFICIENT
    ):
        raise CalibrationNotEstimableError(
            "logistic recalibration coefficients are non-finite or unbounded",
            reason_code="calibration_slope_nonidentifiable",
        )
    return intercept, slope, iterations, gradient_norm, curvature_minimum


def compute_logistic_recalibration(
    records: Sequence[BinaryCalibrationRecord],
    *,
    probability_clip: float = _DEFAULT_PROBABILITY_CLIP,
    tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
    curvature_tolerance: float = _DEFAULT_CURVATURE_TOLERANCE,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> LogisticRecalibration:
    """Compute CITL and calibration slope, refusing undefined logistic fits.

    Valid probabilities equal to zero or one are clipped only for the two
    logit-based diagnostics.  Brier score and ECE always consume the original
    probabilities.  Invalid or non-finite probabilities are never clipped.
    """

    clip = _clip_epsilon(probability_clip)
    tolerance_value = float(tolerance)
    curvature_value = float(curvature_tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(curvature_value) or curvature_value <= 0.0:
        raise ValueError("curvature_tolerance must be finite and positive")
    maximum = _positive_integer(max_iterations, field_name="max_iterations")
    canonical, outcomes, probabilities = _vectors(records)
    observation_weights = _equal_replicate_weights(canonical)
    _require_two_outcome_classes(outcomes)
    n_clipped_low = int(np.count_nonzero(probabilities < clip))
    n_clipped_high = int(np.count_nonzero(probabilities > 1.0 - clip))
    clipped = np.clip(probabilities, clip, 1.0 - clip)
    logits = np.log(clipped) - np.log1p(-clipped)
    citl = _calibration_in_the_large(logits, outcomes, observation_weights)
    intercept, slope, iterations, gradient_norm, curvature_minimum = (
        _logistic_recalibration_fit(
            logits,
            outcomes,
            observation_weights,
            tolerance=tolerance_value,
            curvature_tolerance=curvature_value,
            max_iterations=maximum,
        )
    )
    return LogisticRecalibration(
        calibration_in_the_large=citl,
        intercept=intercept,
        slope=slope,
        probability_clip=clip,
        n_clipped_low=n_clipped_low,
        n_clipped_high=n_clipped_high,
        n_iterations=iterations,
        gradient_infinity_norm=gradient_norm,
        curvature_minimum_eigenvalue=curvature_minimum,
    )


def compute_binary_calibration_metrics(
    records: Sequence[BinaryCalibrationRecord],
    *,
    baseline_probability: float,
    n_bins: int = 10,
    probability_clip: float = _DEFAULT_PROBABILITY_CLIP,
) -> BinaryCalibrationMetrics:
    """Compute the point metrics required by one ADR-013 campaign cell."""

    baseline = _probability(baseline_probability, field_name="baseline_probability")
    canonical, outcomes, probabilities = _vectors(records)
    observation_weights = _equal_replicate_weights(canonical)
    brier = float(observation_weights @ np.square(probabilities - outcomes))
    baseline_brier = float(observation_weights @ np.square(baseline - outcomes))
    if baseline_brier <= 0.0:
        raise CalibrationNotEstimableError(
            "prevalence-only Brier score is zero, so relative improvement is undefined",
            reason_code="calibration_baseline_brier_zero",
        )
    ece = compute_fixed_bin_ece(canonical, n_bins=n_bins)
    logistic = compute_logistic_recalibration(
        canonical, probability_clip=probability_clip
    )
    return BinaryCalibrationMetrics(
        n_observations=len(canonical),
        n_replicates=len({item.replicate_id for item in canonical}),
        observed_prevalence=float(observation_weights @ outcomes),
        baseline_probability=baseline,
        brier_score=brier,
        prevalence_only_brier_score=baseline_brier,
        brier_relative_improvement=(baseline_brier - brier) / baseline_brier,
        fixed_bin_ece=ece,
        logistic_recalibration=logistic,
    )


def replicate_cluster_bootstrap_ece_upper_bound(
    records: Sequence[BinaryCalibrationRecord],
    *,
    n_bins: int = 10,
    confidence_level: float = 0.95,
    n_resamples: int,
    seed: int,
) -> ClusterBootstrapECE:
    """Bootstrap complete replicate clusters and return a one-sided ECE bound.

    Replicate identifiers are sorted before RNG indices are interpreted.  ECE
    is first computed within every replicate; each bootstrap draw resamples
    those complete replicate-level ECE values and averages them equally.
    """

    bins_count = _positive_integer(n_bins, field_name="n_bins")
    confidence = _confidence_level(confidence_level)
    resamples = _positive_integer(n_resamples, field_name="n_resamples")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an explicit non-negative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be an explicit non-negative integer")
    canonical = _canonical_records(records)
    replicate_ids = tuple(sorted({item.replicate_id for item in canonical}))
    if len(replicate_ids) < 2:
        raise CalibrationNotEstimableError(
            "cluster bootstrap requires at least two distinct replicates",
            reason_code="calibration_cluster_bootstrap_too_few_replicates",
        )
    point_result = compute_fixed_bin_ece(canonical, n_bins=bins_count)
    replicate_ece = np.fromiter(
        (item.ece for item in point_result.replicates),
        dtype=np.float64,
        count=len(point_result.replicates),
    )
    point = point_result.ece
    generator = np.random.default_rng(seed_value)
    bootstrap_values: np.ndarray = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = generator.integers(0, len(replicate_ids), size=len(replicate_ids))
        bootstrap_values[index] = float(np.mean(replicate_ece[sampled]))
    ordered = np.sort(bootstrap_values)
    order_rank = min(resamples, max(1, math.ceil(confidence * (resamples + 1))))
    upper = max(point, float(ordered[order_rank - 1]))
    return ClusterBootstrapECE(
        ece=point,
        upper_bound=upper,
        confidence_level=confidence,
        n_resamples=resamples,
        n_replicates=len(replicate_ids),
        seed=seed_value,
        n_bins=bins_count,
    )


def exact_binomial_one_sided_bound(
    successes: int,
    trials: int,
    *,
    side: BinomialBoundSide,
    confidence_level: float = 0.95,
) -> ExactBinomialBound:
    """Return an exact one-sided Clopper-Pearson binomial bound."""

    total = _positive_integer(trials, field_name="trials")
    if isinstance(successes, (bool, np.bool_)) or not isinstance(
        successes, (int, np.integer)
    ):
        raise ValueError("successes must be an integer in [0, trials]")
    observed = int(successes)
    if not 0 <= observed <= total:
        raise ValueError("successes must be an integer in [0, trials]")
    direction = BinomialBoundSide(side)
    confidence = _confidence_level(confidence_level)
    alpha = 1.0 - confidence
    if direction is BinomialBoundSide.UPPER:
        bound = (
            1.0
            if observed == total
            else float(
                beta_distribution.ppf(confidence, observed + 1, total - observed)
            )
        )
    else:
        bound = (
            0.0
            if observed == 0
            else float(beta_distribution.ppf(alpha, observed, total - observed + 1))
        )
    if not math.isfinite(bound) or not 0.0 <= bound <= 1.0:
        raise RuntimeError("exact binomial bound computation returned an invalid value")
    return ExactBinomialBound(
        successes=observed,
        trials=total,
        estimate=observed / total,
        side=direction,
        confidence_level=confidence,
        bound=bound,
    )


__all__ = [
    "BinaryCalibrationMetrics",
    "BinaryCalibrationRecord",
    "BinomialBoundSide",
    "CalibrationBin",
    "CalibrationNotEstimableError",
    "ClusterBootstrapECE",
    "ExactBinomialBound",
    "FixedBinECE",
    "LogisticRecalibration",
    "ReplicateFixedBinECE",
    "compute_binary_calibration_metrics",
    "compute_fixed_bin_ece",
    "compute_logistic_recalibration",
    "exact_binomial_one_sided_bound",
    "replicate_cluster_bootstrap_ece_upper_bound",
]
