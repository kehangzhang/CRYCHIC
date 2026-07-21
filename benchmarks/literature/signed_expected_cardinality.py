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
from scipy.special import expit, logit
from scipy.stats import norm

MIN_FIT_EDGES = 200


@dataclass(frozen=True)
class SpikeNormalFit:
    """Parameters and diagnostics for the benchmark-only working prior."""

    null_weight: float
    slab_sd: float
    scale: float
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
) -> SpikeNormalFit:
    """Fit a zero-spike/symmetric-normal-slab marginal model by MLE."""

    y, se = _validated_effect_se(effect, standard_error)
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

    bounds = ((float(logit(1e-4)), float(logit(1.0 - 1e-4))), (-4.605, 4.605))
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


def build_signed_expected_cardinality_head(
    effects: pd.DataFrame,
    *,
    dataset_id: str,
    arm: str,
    delta_fraction: float,
    min_fit_edges: int = MIN_FIT_EDGES,
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
        effect[observed], standard_error[observed], min_fit_edges=min_fit_edges
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


__all__ = [
    "MIN_FIT_EDGES",
    "SpikeNormalFit",
    "build_signed_expected_cardinality_head",
    "fit_spike_normal_working_prior",
    "signed_working_probabilities",
]
