"""Exploratory signed expected-cardinality head for multi-group benchmarks.

This module deliberately lives under ``benchmarks``.  Its probabilities are
from a working empirical-Bayes model fitted to analytic effect/SE summaries;
they are not CRYCHIC's released full-pipeline posterior probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, logit, logsumexp
from scipy.stats import norm

MIN_FIT_EDGES = 200


@dataclass(frozen=True)
class SpikeNormalFit:
    """Parameters and diagnostics for the benchmark-only working prior."""

    null_weight: float
    slab_sd: float
    scale: float
    min_slab_scale_fraction: float
    n_fit_edges: int
    converged: bool
    negative_log_likelihood: float
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DirectionalSpikeNormalFit:
    """Three-component working prior with separate positive/negative slabs."""

    null_weight: float
    target_weight: float
    reference_weight: float
    target_slab_sd: float
    reference_slab_sd: float
    scale: float
    min_slab_scale_fraction: float
    n_fit_edges: int
    converged: bool
    negative_log_likelihood: float
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_effect_se(
    effect: np.ndarray, standard_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(effect, dtype=float)
    se = np.asarray(standard_error, dtype=float)
    if y.ndim != 1 or se.ndim != 1 or y.shape != se.shape:
        raise ValueError("effect and standard_error must be equal-length vectors")
    invalid_zero = np.isfinite(y) & (se == 0.0) & (y != 0.0)
    if invalid_zero.any():
        raise ValueError(
            "nonzero effects with zero standard error are not eligible for the "
            "signed-cardinality working model"
        )
    return y, se


def fit_spike_normal_working_prior(
    effect: np.ndarray,
    standard_error: np.ndarray,
    *,
    min_fit_edges: int = MIN_FIT_EDGES,
    min_slab_scale_fraction: float = 0.01,
) -> SpikeNormalFit:
    """Fit a zero-spike/symmetric-normal-slab marginal model by MLE."""

    y, se = _validated_effect_se(effect, standard_error)
    if (
        not np.isfinite(min_slab_scale_fraction)
        or min_slab_scale_fraction <= 0.0
        or min_slab_scale_fraction > 10.0
    ):
        raise ValueError(
            "min_slab_scale_fraction must be finite and in the interval (0, 10]"
        )
    eligible = np.isfinite(y) & np.isfinite(se) & (se > 0.0)
    y_fit = y[eligible]
    se_fit = se[eligible]
    if len(y_fit) < min_fit_edges:
        raise ValueError(
            "signed-cardinality prior fit needs at least "
            f"{min_fit_edges} finite positive-SE edges; found {len(y_fit)}"
        )

    scale = float(
        max(
            np.quantile(np.abs(y_fit), 0.75),
            np.median(se_fit),
            np.finfo(float).tiny,
        )
    )
    log_two_pi = float(np.log(2.0 * np.pi))

    def objective(parameters: np.ndarray) -> float:
        slab_weight = float(expit(parameters[0]))
        slab_sd = scale * float(np.exp(parameters[1]))
        null_variance = np.square(se_fit)
        slab_variance = null_variance + slab_sd**2
        log_null = -0.5 * (
            log_two_pi + np.log(null_variance) + np.square(y_fit) / null_variance
        )
        log_slab = -0.5 * (
            log_two_pi + np.log(slab_variance) + np.square(y_fit) / slab_variance
        )
        log_marginal = np.logaddexp(
            np.log1p(-slab_weight) + log_null,
            np.log(slab_weight) + log_slab,
        )
        return float(-np.sum(log_marginal))

    bounds = (
        (float(logit(1e-4)), float(logit(1.0 - 1e-4))),
        (float(np.log(min_slab_scale_fraction)), float(np.log(100.0))),
    )
    candidates = []
    for slab_weight in (0.05, 0.2, 0.5):
        for relative_sd in (0.5, 1.0, 2.0):
            result = minimize(
                objective,
                np.array([logit(slab_weight), np.log(relative_sd)], dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
            )
            candidates.append(result)
    best = min(candidates, key=lambda item: float(item.fun))
    slab_weight = float(expit(best.x[0]))
    return SpikeNormalFit(
        null_weight=1.0 - slab_weight,
        slab_sd=scale * float(np.exp(best.x[1])),
        scale=scale,
        min_slab_scale_fraction=min_slab_scale_fraction,
        n_fit_edges=len(y_fit),
        converged=bool(best.success),
        negative_log_likelihood=float(best.fun),
    )


def _directional_log_density(
    effect: np.ndarray,
    standard_error: np.ndarray,
    slab_sd: float,
    *,
    direction: int,
) -> np.ndarray:
    variance = np.square(standard_error) + slab_sd**2
    posterior_z = (
        effect
        * slab_sd
        / (standard_error * np.sqrt(variance))
    )
    return (
        np.log(2.0)
        + norm.logpdf(effect, loc=0.0, scale=np.sqrt(variance))
        + log_ndtr(direction * posterior_z)
    )


def fit_directional_spike_normal_working_prior(
    effect: np.ndarray,
    standard_error: np.ndarray,
    *,
    min_fit_edges: int = MIN_FIT_EDGES,
    min_slab_scale_fraction: float = 0.01,
) -> DirectionalSpikeNormalFit:
    """Fit zero, positive-half-normal, and negative-half-normal components."""

    y, se = _validated_effect_se(effect, standard_error)
    if (
        not np.isfinite(min_slab_scale_fraction)
        or min_slab_scale_fraction <= 0.0
        or min_slab_scale_fraction > 10.0
    ):
        raise ValueError(
            "min_slab_scale_fraction must be finite and in the interval (0, 10]"
        )
    eligible = np.isfinite(y) & np.isfinite(se) & (se > 0.0)
    y_fit = y[eligible]
    se_fit = se[eligible]
    if len(y_fit) < min_fit_edges:
        raise ValueError(
            "directional signed-cardinality prior fit needs at least "
            f"{min_fit_edges} finite positive-SE edges; found {len(y_fit)}"
        )
    scale = float(
        max(
            np.quantile(np.abs(y_fit), 0.75),
            np.median(se_fit),
            np.finfo(float).tiny,
        )
    )
    log_null_density = norm.logpdf(y_fit, loc=0.0, scale=se_fit)

    def objective(parameters: np.ndarray) -> float:
        logits = np.array([0.0, parameters[0], parameters[1]], dtype=float)
        log_weights = logits - logsumexp(logits)
        target_sd = scale * float(np.exp(parameters[2]))
        reference_sd = scale * float(np.exp(parameters[3]))
        log_components = np.column_stack(
            [
                log_weights[0] + log_null_density,
                log_weights[1]
                + _directional_log_density(
                    y_fit, se_fit, target_sd, direction=1
                ),
                log_weights[2]
                + _directional_log_density(
                    y_fit, se_fit, reference_sd, direction=-1
                ),
            ]
        )
        return float(-np.sum(logsumexp(log_components, axis=1)))

    weight_bound = float(abs(logit(1e-4)))
    scale_bounds = (
        float(np.log(min_slab_scale_fraction)),
        float(np.log(100.0)),
    )
    bounds = (
        (-weight_bound, weight_bound),
        (-weight_bound, weight_bound),
        scale_bounds,
        scale_bounds,
    )
    starts = (
        (0.90, 0.05, 0.05, 1.5, 1.5),
        (0.80, 0.10, 0.10, 3.0, 3.0),
        (0.80, 0.15, 0.05, 1.5, 3.0),
        (0.80, 0.05, 0.15, 3.0, 1.5),
    )
    candidates = []
    for null_weight, target_weight, reference_weight, target_sd, reference_sd in starts:
        initial = np.array(
            [
                np.log(target_weight / null_weight),
                np.log(reference_weight / null_weight),
                np.log(max(target_sd, min_slab_scale_fraction)),
                np.log(max(reference_sd, min_slab_scale_fraction)),
            ],
            dtype=float,
        )
        candidates.append(
            minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
        )
    best = min(candidates, key=lambda item: float(item.fun))
    logits = np.array([0.0, best.x[0], best.x[1]], dtype=float)
    weights = np.exp(logits - logsumexp(logits))
    return DirectionalSpikeNormalFit(
        null_weight=float(weights[0]),
        target_weight=float(weights[1]),
        reference_weight=float(weights[2]),
        target_slab_sd=scale * float(np.exp(best.x[2])),
        reference_slab_sd=scale * float(np.exp(best.x[3])),
        scale=scale,
        min_slab_scale_fraction=min_slab_scale_fraction,
        n_fit_edges=len(y_fit),
        converged=bool(best.success),
        negative_log_likelihood=float(best.fun),
    )


def signed_working_probabilities(
    effect: np.ndarray,
    standard_error: np.ndarray,
    fit: SpikeNormalFit,
    *,
    delta_fraction: float,
) -> dict[str, np.ndarray]:
    """Return signed minimum-effect probabilities under the working model."""

    if not np.isfinite(delta_fraction) or delta_fraction < 0.0:
        raise ValueError("delta_fraction must be finite and nonnegative")
    if fit.formal_release_allowed or fit.fit_status != "candidate_unreleased":
        raise ValueError("benchmark working-prior release contract was altered")
    y, se = _validated_effect_se(effect, standard_error)
    n_rows = len(y)
    outputs = {
        "working_p_target_active": np.full(n_rows, np.nan, dtype=float),
        "working_p_reference_active": np.full(n_rows, np.nan, dtype=float),
        "working_local_false_sign_rate": np.full(n_rows, np.nan, dtype=float),
        "working_posterior_effect_mean": np.full(n_rows, np.nan, dtype=float),
        "working_posterior_effect_sd": np.full(n_rows, np.nan, dtype=float),
        "working_prob_effect_gt_delta": np.full(n_rows, np.nan, dtype=float),
        "working_prob_effect_lt_minus_delta": np.full(n_rows, np.nan, dtype=float),
    }
    eligible = np.isfinite(y) & np.isfinite(se) & (se > 0.0)
    structural_zero = np.isfinite(y) & (y == 0.0) & (se == 0.0)
    if eligible.any():
        y_ok = y[eligible]
        se_ok = se[eligible]
        tau2 = fit.slab_sd**2
        se2 = np.square(se_ok)
        null_variance = se2
        slab_variance = se2 + tau2
        log_null = np.log(fit.null_weight) + norm.logpdf(
            y_ok, loc=0.0, scale=np.sqrt(null_variance)
        )
        log_slab = np.log1p(-fit.null_weight) + norm.logpdf(
            y_ok, loc=0.0, scale=np.sqrt(slab_variance)
        )
        slab_probability = expit(log_slab - log_null)
        shrinkage = tau2 / slab_variance
        conditional_mean = shrinkage * y_ok
        conditional_variance = shrinkage * se2
        conditional_sd = np.sqrt(conditional_variance)
        delta = delta_fraction * fit.slab_sd

        conditional_target = norm.sf(
            (delta - conditional_mean) / conditional_sd
        )
        conditional_reference = norm.cdf(
            (-delta - conditional_mean) / conditional_sd
        )
        p_target = slab_probability * conditional_target
        p_reference = slab_probability * conditional_reference
        p_positive = slab_probability * norm.cdf(conditional_mean / conditional_sd)
        p_negative = slab_probability * norm.cdf(-conditional_mean / conditional_sd)
        posterior_mean = slab_probability * conditional_mean
        posterior_second = slab_probability * (
            conditional_variance + np.square(conditional_mean)
        )
        posterior_sd = np.sqrt(
            np.maximum(posterior_second - np.square(posterior_mean), 0.0)
        )

        outputs["working_p_target_active"][eligible] = p_target
        outputs["working_p_reference_active"][eligible] = p_reference
        outputs["working_local_false_sign_rate"][eligible] = 1.0 - np.maximum(
            p_positive, p_negative
        )
        outputs["working_posterior_effect_mean"][eligible] = posterior_mean
        outputs["working_posterior_effect_sd"][eligible] = posterior_sd
        outputs["working_prob_effect_gt_delta"][eligible] = p_target
        outputs["working_prob_effect_lt_minus_delta"][eligible] = p_reference

    for values in outputs.values():
        values[structural_zero] = 0.0
    outputs["working_local_false_sign_rate"][structural_zero] = 1.0
    return outputs


def signed_directional_working_probabilities(
    effect: np.ndarray,
    standard_error: np.ndarray,
    fit: DirectionalSpikeNormalFit,
    *,
    delta_fraction: float,
) -> dict[str, np.ndarray]:
    """Return signed probabilities from the directional three-group model."""

    if not np.isfinite(delta_fraction) or delta_fraction < 0.0:
        raise ValueError("delta_fraction must be finite and nonnegative")
    if fit.formal_release_allowed or fit.fit_status != "candidate_unreleased":
        raise ValueError("benchmark working-prior release contract was altered")
    y, se = _validated_effect_se(effect, standard_error)
    n_rows = len(y)
    outputs = {
        "working_p_target_active": np.full(n_rows, np.nan, dtype=float),
        "working_p_reference_active": np.full(n_rows, np.nan, dtype=float),
        "working_local_false_sign_rate": np.full(n_rows, np.nan, dtype=float),
        "working_posterior_effect_mean": np.full(n_rows, np.nan, dtype=float),
        "working_posterior_effect_sd": np.full(n_rows, np.nan, dtype=float),
        "working_prob_effect_gt_delta": np.full(n_rows, np.nan, dtype=float),
        "working_prob_effect_lt_minus_delta": np.full(n_rows, np.nan, dtype=float),
    }
    eligible = np.isfinite(y) & np.isfinite(se) & (se > 0.0)
    structural_zero = np.isfinite(y) & (y == 0.0) & (se == 0.0)
    if eligible.any():
        y_ok = y[eligible]
        se_ok = se[eligible]
        target_tau = fit.target_slab_sd
        reference_tau = fit.reference_slab_sd
        log_components = np.column_stack(
            [
                np.log(fit.null_weight)
                + norm.logpdf(y_ok, loc=0.0, scale=se_ok),
                np.log(fit.target_weight)
                + _directional_log_density(
                    y_ok, se_ok, target_tau, direction=1
                ),
                np.log(fit.reference_weight)
                + _directional_log_density(
                    y_ok, se_ok, reference_tau, direction=-1
                ),
            ]
        )
        responsibilities = np.exp(
            log_components - logsumexp(log_components, axis=1, keepdims=True)
        )

        def conditional(
            slab_sd: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            variance = np.square(se_ok) + slab_sd**2
            shrinkage = slab_sd**2 / variance
            mean = shrinkage * y_ok
            sd = np.sqrt(shrinkage * np.square(se_ok))
            return mean, sd

        target_mean, target_sd = conditional(target_tau)
        reference_mean, reference_sd = conditional(reference_tau)
        target_delta = delta_fraction * target_tau
        reference_delta = delta_fraction * reference_tau
        target_tail = np.exp(
            log_ndtr((target_mean - target_delta) / target_sd)
            - log_ndtr(target_mean / target_sd)
        )
        reference_tail = np.exp(
            log_ndtr((-reference_delta - reference_mean) / reference_sd)
            - log_ndtr(-reference_mean / reference_sd)
        )
        p_target = responsibilities[:, 1] * np.clip(target_tail, 0.0, 1.0)
        p_reference = responsibilities[:, 2] * np.clip(
            reference_tail, 0.0, 1.0
        )

        target_alpha = -target_mean / target_sd
        target_mills = np.exp(
            np.clip(
                norm.logpdf(target_alpha) - log_ndtr(-target_alpha),
                -745.0,
                50.0,
            )
        )
        target_truncated_mean = target_mean + target_sd * target_mills
        target_truncated_variance = np.square(target_sd) * np.maximum(
            1.0
            + target_alpha * target_mills
            - np.square(target_mills),
            0.0,
        )
        reference_beta = -reference_mean / reference_sd
        reference_mills = np.exp(
            np.clip(
                norm.logpdf(reference_beta) - log_ndtr(reference_beta),
                -745.0,
                50.0,
            )
        )
        reference_truncated_mean = (
            reference_mean - reference_sd * reference_mills
        )
        reference_truncated_variance = np.square(reference_sd) * np.maximum(
            1.0
            - reference_beta * reference_mills
            - np.square(reference_mills),
            0.0,
        )
        posterior_mean = (
            responsibilities[:, 1] * target_truncated_mean
            + responsibilities[:, 2] * reference_truncated_mean
        )
        posterior_second = responsibilities[:, 1] * (
            target_truncated_variance + np.square(target_truncated_mean)
        ) + responsibilities[:, 2] * (
            reference_truncated_variance + np.square(reference_truncated_mean)
        )
        posterior_sd = np.sqrt(
            np.maximum(posterior_second - np.square(posterior_mean), 0.0)
        )

        outputs["working_p_target_active"][eligible] = p_target
        outputs["working_p_reference_active"][eligible] = p_reference
        outputs["working_local_false_sign_rate"][eligible] = 1.0 - np.maximum(
            responsibilities[:, 1], responsibilities[:, 2]
        )
        outputs["working_posterior_effect_mean"][eligible] = posterior_mean
        outputs["working_posterior_effect_sd"][eligible] = posterior_sd
        outputs["working_prob_effect_gt_delta"][eligible] = p_target
        outputs["working_prob_effect_lt_minus_delta"][eligible] = p_reference

    for values in outputs.values():
        values[structural_zero] = 0.0
    outputs["working_local_false_sign_rate"][structural_zero] = 1.0
    return outputs


def build_signed_expected_cardinality_head(
    effects: pd.DataFrame,
    *,
    dataset_id: str,
    arm: str,
    delta_fraction: float,
    min_fit_edges: int = MIN_FIT_EDGES,
    min_slab_scale_fraction: float = 0.01,
) -> tuple[pd.DataFrame, SpikeNormalFit]:
    """Build edge weights compatible with component-swap pair aggregation."""

    required = {
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "reference_condition",
        "target_condition",
        "effect_target_minus_reference",
        "effect_standard_error_hc2",
        "n_samples_reference",
        "n_samples_target",
        "status",
        "reason_code",
    }
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"effect table is missing: {sorted(missing)}")
    observed = effects["status"].eq("observed").to_numpy()
    effect = pd.to_numeric(
        effects["effect_target_minus_reference"], errors="coerce"
    ).to_numpy(dtype=float)
    standard_error = pd.to_numeric(
        effects["effect_standard_error_hc2"], errors="coerce"
    ).to_numpy(dtype=float)
    fit = fit_spike_normal_working_prior(
        effect[observed],
        standard_error[observed],
        min_fit_edges=min_fit_edges,
        min_slab_scale_fraction=min_slab_scale_fraction,
    )
    probabilities = signed_working_probabilities(
        effect, standard_error, fit, delta_fraction=delta_fraction
    )
    identity = ["sender", "receiver", "interaction_id", "ligand", "receptor"]
    result = effects.loc[:, identity].copy()
    result.insert(0, "dataset", dataset_id)
    result.insert(1, "arm", arm)
    result["reference_condition"] = str(effects["reference_condition"].iloc[0])
    result["target_condition"] = str(effects["target_condition"].iloc[0])
    result["effect"] = effect
    result["standard_error"] = standard_error
    result["test_statistic"] = np.divide(
        effect,
        standard_error,
        out=np.zeros(len(effect), dtype=float),
        where=standard_error > 0.0,
    )
    result["p_value"] = np.nan
    result["n_reference"] = effects["n_samples_reference"].to_numpy()
    result["n_target"] = effects["n_samples_target"].to_numpy()
    result["status"] = effects["status"].to_numpy()
    result["reason_code"] = effects["reason_code"].to_numpy()
    for name, values in probabilities.items():
        result[name] = values
    result["target_weight"] = np.where(
        observed,
        np.nan_to_num(result["working_p_target_active"].to_numpy(dtype=float)),
        0.0,
    )
    result["reference_weight"] = np.where(
        observed,
        np.nan_to_num(result["working_p_reference_active"].to_numpy(dtype=float)),
        0.0,
    )
    result["selected_target"] = result["target_weight"].ge(0.5)
    result["selected_reference"] = result["reference_weight"].ge(0.5)
    result["selection_rule"] = (
        "candidate_unreleased_spike_normal_expected_cardinality_"
        f"delta={delta_fraction:g}*slab_sd"
    )
    result["probability_status"] = "candidate_unreleased"
    result["formal_release_allowed"] = False
    return result, fit


def build_directional_signed_expected_cardinality_head(
    effects: pd.DataFrame,
    *,
    dataset_id: str,
    arm: str,
    delta_fraction: float,
    min_fit_edges: int = MIN_FIT_EDGES,
    min_slab_scale_fraction: float = 0.01,
) -> tuple[pd.DataFrame, DirectionalSpikeNormalFit]:
    """Build an edge head using separately fitted positive/negative slabs."""

    required = {
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "reference_condition",
        "target_condition",
        "effect_target_minus_reference",
        "effect_standard_error_hc2",
        "n_samples_reference",
        "n_samples_target",
        "status",
        "reason_code",
    }
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"effect table is missing: {sorted(missing)}")
    observed = effects["status"].eq("observed").to_numpy()
    effect = pd.to_numeric(
        effects["effect_target_minus_reference"], errors="coerce"
    ).to_numpy(dtype=float)
    standard_error = pd.to_numeric(
        effects["effect_standard_error_hc2"], errors="coerce"
    ).to_numpy(dtype=float)
    fit = fit_directional_spike_normal_working_prior(
        effect[observed],
        standard_error[observed],
        min_fit_edges=min_fit_edges,
        min_slab_scale_fraction=min_slab_scale_fraction,
    )
    probabilities = signed_directional_working_probabilities(
        effect, standard_error, fit, delta_fraction=delta_fraction
    )
    identity = ["sender", "receiver", "interaction_id", "ligand", "receptor"]
    result = effects.loc[:, identity].copy()
    result.insert(0, "dataset", dataset_id)
    result.insert(1, "arm", arm)
    result["reference_condition"] = str(effects["reference_condition"].iloc[0])
    result["target_condition"] = str(effects["target_condition"].iloc[0])
    result["effect"] = effect
    result["standard_error"] = standard_error
    result["test_statistic"] = np.divide(
        effect,
        standard_error,
        out=np.zeros(len(effect), dtype=float),
        where=standard_error > 0.0,
    )
    result["p_value"] = np.nan
    result["n_reference"] = effects["n_samples_reference"].to_numpy()
    result["n_target"] = effects["n_samples_target"].to_numpy()
    result["status"] = effects["status"].to_numpy()
    result["reason_code"] = effects["reason_code"].to_numpy()
    for name, values in probabilities.items():
        result[name] = values
    result["target_weight"] = np.where(
        observed,
        np.nan_to_num(result["working_p_target_active"].to_numpy(dtype=float)),
        0.0,
    )
    result["reference_weight"] = np.where(
        observed,
        np.nan_to_num(result["working_p_reference_active"].to_numpy(dtype=float)),
        0.0,
    )
    result["selected_target"] = result["target_weight"].ge(0.5)
    result["selected_reference"] = result["reference_weight"].ge(0.5)
    result["selection_rule"] = (
        "candidate_unreleased_directional_spike_normal_expected_cardinality_"
        f"delta={delta_fraction:g}*directional_slab_sd"
    )
    result["probability_status"] = "candidate_unreleased"
    result["formal_release_allowed"] = False
    return result, fit


__all__ = [
    "MIN_FIT_EDGES",
    "DirectionalSpikeNormalFit",
    "SpikeNormalFit",
    "build_directional_signed_expected_cardinality_head",
    "build_signed_expected_cardinality_head",
    "fit_directional_spike_normal_working_prior",
    "fit_spike_normal_working_prior",
    "signed_directional_working_probabilities",
    "signed_working_probabilities",
]
