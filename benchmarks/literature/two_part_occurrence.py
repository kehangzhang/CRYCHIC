"""Subject-level occurrence and positive-magnitude benchmark evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class HurdleChannelReliability:
    """Unsupervised reliability shrinkage for one hurdle channel."""

    channel: str
    sign_concordance: float
    reliability: float
    maximum_alpha: float
    alpha_cap: float
    effective_alpha: float
    n_concordance_edges: int
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty edge-by-subject matrix")
    return matrix


def _beta_summary(
    values: np.ndarray,
    group: np.ndarray,
    *,
    prior: float,
    minimum_subjects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = values[:, group]
    finite = np.isfinite(selected)
    count = finite.sum(axis=1)
    successes = np.where(finite, selected, 0.0).sum(axis=1)
    alpha = successes + prior
    beta = count - successes + prior
    total = alpha + beta
    mean = np.divide(
        alpha,
        total,
        out=np.full(len(values), np.nan, dtype=float),
        where=count >= minimum_subjects,
    )
    variance = np.divide(
        alpha * beta,
        total**2 * (total + 1.0),
        out=np.full(len(values), np.nan, dtype=float),
        where=count >= minimum_subjects,
    )
    return mean, variance, count


def _standardized_effect(effect: np.ndarray, standard_error: np.ndarray) -> np.ndarray:
    statistic = np.full(len(effect), np.nan, dtype=float)
    positive = np.isfinite(effect) & np.isfinite(standard_error) & (standard_error > 0)
    statistic[positive] = effect[positive] / standard_error[positive]
    separated = np.isfinite(effect) & (standard_error == 0) & (effect != 0)
    statistic[separated] = np.sign(effect[separated]) * np.inf
    return statistic


def fit_subject_hurdle_effects(
    presence: Any,
    positive_magnitude: Any,
    conditions: Any,
    *,
    reference: str,
    target: str,
    beta_prior: float = 0.5,
    minimum_subjects: int = 3,
    minimum_active_subjects: int = 3,
    conditional_presence_threshold: float = 0.5,
    magnitude_log_sd_floor: float = 0.05,
) -> pd.DataFrame:
    """Estimate separate subject-level occurrence and positive-magnitude effects."""

    occurrence = _validated_matrix(presence, name="presence")
    magnitude = _validated_matrix(positive_magnitude, name="positive_magnitude")
    if occurrence.shape != magnitude.shape:
        raise ValueError("presence and positive_magnitude must have identical shapes")
    condition = np.asarray(conditions, dtype=object)
    if condition.ndim != 1 or len(condition) != occurrence.shape[1]:
        raise ValueError("conditions must contain one label per subject")
    if reference == target or set(condition) != {reference, target}:
        raise ValueError("conditions must contain exactly reference and target labels")
    finite_presence = occurrence[np.isfinite(occurrence)]
    if ((finite_presence < 0.0) | (finite_presence > 1.0)).any():
        raise ValueError("finite presence values must lie in [0, 1]")
    finite_magnitude = magnitude[np.isfinite(magnitude)]
    if (finite_magnitude <= 0.0).any():
        raise ValueError("finite positive_magnitude values must be positive")
    if not np.isfinite(beta_prior) or beta_prior <= 0.0:
        raise ValueError("beta_prior must be finite and positive")
    if minimum_subjects < 2 or minimum_active_subjects < 2:
        raise ValueError("minimum subject counts must be at least two")
    if not 0.0 <= conditional_presence_threshold <= 1.0:
        raise ValueError("conditional_presence_threshold must lie in [0, 1]")
    if not np.isfinite(magnitude_log_sd_floor) or magnitude_log_sd_floor <= 0.0:
        raise ValueError("magnitude_log_sd_floor must be finite and positive")

    reference_mask = condition == reference
    target_mask = condition == target
    reference_mean, reference_variance, reference_count = _beta_summary(
        occurrence,
        reference_mask,
        prior=beta_prior,
        minimum_subjects=minimum_subjects,
    )
    target_mean, target_variance, target_count = _beta_summary(
        occurrence,
        target_mask,
        prior=beta_prior,
        minimum_subjects=minimum_subjects,
    )
    occurrence_effect = target_mean - reference_mean
    occurrence_se = np.sqrt(reference_variance + target_variance)
    occurrence_z = _standardized_effect(occurrence_effect, occurrence_se)

    edge_count = occurrence.shape[0]
    magnitude_effect = np.full(edge_count, np.nan, dtype=float)
    magnitude_se = np.full(edge_count, np.nan, dtype=float)
    magnitude_reference_count = np.zeros(edge_count, dtype=int)
    magnitude_target_count = np.zeros(edge_count, dtype=int)
    variance_floor = magnitude_log_sd_floor**2
    for edge in range(edge_count):
        active = (
            np.isfinite(occurrence[edge])
            & (occurrence[edge] >= conditional_presence_threshold)
            & np.isfinite(magnitude[edge])
        )
        reference_values = np.log(magnitude[edge, active & reference_mask])
        target_values = np.log(magnitude[edge, active & target_mask])
        magnitude_reference_count[edge] = len(reference_values)
        magnitude_target_count[edge] = len(target_values)
        if (
            len(reference_values) < minimum_active_subjects
            or len(target_values) < minimum_active_subjects
        ):
            continue
        magnitude_effect[edge] = target_values.mean() - reference_values.mean()
        reference_var = max(float(reference_values.var(ddof=1)), variance_floor)
        target_var = max(float(target_values.var(ddof=1)), variance_floor)
        magnitude_se[edge] = np.sqrt(
            reference_var / len(reference_values) + target_var / len(target_values)
        )
    magnitude_z = _standardized_effect(magnitude_effect, magnitude_se)
    occurrence_observed = np.isfinite(occurrence_effect)
    magnitude_observed = np.isfinite(magnitude_effect)
    return pd.DataFrame(
        {
            "occurrence_effect": occurrence_effect,
            "occurrence_standard_error": occurrence_se,
            "occurrence_z": occurrence_z,
            "occurrence_reference_posterior_mean": reference_mean,
            "occurrence_target_posterior_mean": target_mean,
            "occurrence_reference_subjects": reference_count,
            "occurrence_target_subjects": target_count,
            "occurrence_status": np.where(
                occurrence_observed, "observed", "not_estimable"
            ),
            "magnitude_effect": magnitude_effect,
            "magnitude_standard_error": magnitude_se,
            "magnitude_z": magnitude_z,
            "magnitude_reference_active_subjects": magnitude_reference_count,
            "magnitude_target_active_subjects": magnitude_target_count,
            "magnitude_status": np.where(
                magnitude_observed, "observed", "not_estimable"
            ),
        }
    )


def fit_hurdle_channel_reliability(
    baseline_statistic: Any,
    channel_effect: Any,
    *,
    channel: str,
    maximum_alpha: float,
    alpha_cap: float,
    minimum_edges: int,
) -> HurdleChannelReliability:
    """Shrink one hurdle channel by agreement with an independent baseline."""

    baseline = np.asarray(baseline_statistic, dtype=float)
    effect = np.asarray(channel_effect, dtype=float)
    if baseline.ndim != 1 or effect.ndim != 1 or baseline.shape != effect.shape:
        raise ValueError("baseline_statistic and channel_effect must align")
    if channel not in {"occurrence", "magnitude"}:
        raise ValueError("channel must be occurrence or magnitude")
    if not np.isfinite(maximum_alpha) or maximum_alpha < 0.0:
        raise ValueError("maximum_alpha must be finite and nonnegative")
    if not np.isfinite(alpha_cap) or alpha_cap <= 0.0:
        raise ValueError("alpha_cap must be finite and positive")
    if minimum_edges < 1:
        raise ValueError("minimum_edges must be positive")
    eligible = (
        np.isfinite(baseline)
        & np.isfinite(effect)
        & (baseline != 0.0)
        & (effect != 0.0)
    )
    n_edges = int(eligible.sum())
    if n_edges < minimum_edges:
        concordance = float("nan")
        reliability = 0.0
    else:
        concordance = float(
            np.mean(np.sign(baseline[eligible]) == np.sign(effect[eligible]))
        )
        reliability = float(np.clip(2.0 * (concordance - 0.5), 0.0, 1.0))
    return HurdleChannelReliability(
        channel=channel,
        sign_concordance=concordance,
        reliability=reliability,
        maximum_alpha=maximum_alpha,
        alpha_cap=alpha_cap,
        effective_alpha=min(alpha_cap, maximum_alpha * reliability),
        n_concordance_edges=n_edges,
    )


def hurdle_directional_weights(
    direction: Any,
    occurrence_z: Any,
    magnitude_z: Any,
    occurrence_fit: HurdleChannelReliability,
    magnitude_fit: HurdleChannelReliability,
    *,
    log_weight_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a bounded weight while preserving separate hurdle evidence."""

    sign = np.asarray(direction, dtype=float)
    occurrence = np.asarray(occurrence_z, dtype=float)
    magnitude = np.asarray(magnitude_z, dtype=float)
    if (
        sign.ndim != 1
        or occurrence.shape != sign.shape
        or magnitude.shape != sign.shape
    ):
        raise ValueError("direction and hurdle statistics must align")
    if not np.isin(sign[np.isfinite(sign)], (-1.0, 1.0)).all():
        raise ValueError("finite direction values must equal -1 or 1")
    if occurrence_fit.channel != "occurrence" or magnitude_fit.channel != "magnitude":
        raise ValueError("hurdle reliability channels are inconsistent")
    if occurrence_fit.formal_release_allowed or magnitude_fit.formal_release_allowed:
        raise ValueError("benchmark-only hurdle release contract was altered")
    if not np.isfinite(log_weight_cap) or log_weight_cap <= 0.0:
        raise ValueError("log_weight_cap must be finite and positive")

    occurrence_evidence = np.zeros(len(sign), dtype=float)
    magnitude_evidence = np.zeros(len(sign), dtype=float)
    occurrence_available = np.isfinite(sign) & np.isfinite(occurrence)
    magnitude_available = np.isfinite(sign) & np.isfinite(magnitude)
    occurrence_evidence[occurrence_available] = (
        2.0 * norm.cdf(sign[occurrence_available] * occurrence[occurrence_available])
        - 1.0
    )
    magnitude_evidence[magnitude_available] = (
        2.0 * norm.cdf(sign[magnitude_available] * magnitude[magnitude_available]) - 1.0
    )
    log_weight = (
        occurrence_fit.effective_alpha * occurrence_evidence
        + magnitude_fit.effective_alpha * magnitude_evidence
    )
    weights = np.exp(np.clip(log_weight, -log_weight_cap, log_weight_cap))
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise AssertionError("hurdle weights must be finite and positive")
    return weights, occurrence_evidence, magnitude_evidence


__all__ = [
    "HurdleChannelReliability",
    "fit_hurdle_channel_reliability",
    "fit_subject_hurdle_effects",
    "hurdle_directional_weights",
]
