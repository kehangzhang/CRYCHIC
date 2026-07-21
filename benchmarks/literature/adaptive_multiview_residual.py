"""Adaptive nonnegative view profiles for the signed LIANA residual."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.literature.liana_hypergraph_residual import (
    multiview_codes,
    robustly_scale_anchor,
    stable_edge_folds,
)


@dataclass(frozen=True)
class AdaptiveMultiviewResidualFit:
    """Cross-validated profile, penalties, and exact fallback decision."""

    profile_name: str
    view_weights: tuple[float, ...]
    lambda_hypergraph: float
    anchor_weight: float
    cv_mse: float
    cv_baseline_mse: float
    cv_relative_improvement: float
    profile_relative_improvement_over_equal: float
    fallback_gate: float
    folds: int
    views: tuple[str, ...]
    converged: bool
    iterations: int
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _vector(values: Any, *, field: str, allow_missing: bool) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    if not allow_missing and not np.isfinite(result).all():
        raise ValueError(f"{field} must contain finite values")
    return result


def normalize_view_weights(
    weights: Sequence[float], *, view_count: int
) -> tuple[float, ...]:
    """Normalize nonnegative profile weights to fixed total penalty mass."""

    result = np.asarray(tuple(weights), dtype=float)
    if result.shape != (view_count,):
        raise ValueError("view profile length must match the number of views")
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise ValueError("view profile weights must be finite and nonnegative")
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("a view profile must retain at least one view")
    normalized = result * (view_count / total)
    return tuple(float(value) for value in normalized)


def _group_projection(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    counts = np.bincount(codes)
    sums = np.bincount(codes, weights=values)
    return (sums / counts)[codes]


def estimate_oof_view_weights(
    edges: pd.DataFrame,
    baseline: Any,
    *,
    views: tuple[str, ...],
    key_columns: tuple[str, ...],
    folds: int,
    seed: int,
    full_reliability_improvement: float = 0.25,
) -> tuple[tuple[float, ...], pd.DataFrame]:
    """Estimate nonnegative view reliability from held-out group means."""

    b = _vector(baseline, field="baseline", allow_missing=False)
    if len(edges) != len(b):
        raise ValueError("edge table and baseline length mismatch")
    if (
        not np.isfinite(full_reliability_improvement)
        or full_reliability_improvement <= 0.0
    ):
        raise ValueError("full reliability improvement must be finite and positive")
    codes = multiview_codes(edges, views)
    fold_id = stable_edge_folds(edges, key_columns=key_columns, folds=folds, seed=seed)
    squared_error = np.zeros(len(codes), dtype=float)
    baseline_squared_error = 0.0
    for fold in range(folds):
        holdout = fold_id == fold
        training = ~holdout
        training_mean = float(b[training].mean())
        baseline_squared_error += float(np.square(b[holdout] - training_mean).sum())
        for index, code in enumerate(codes):
            group_count = np.bincount(code[training], minlength=code.max() + 1)
            group_sum = np.bincount(
                code[training], weights=b[training], minlength=code.max() + 1
            )
            group_mean = np.divide(
                group_sum,
                group_count,
                out=np.full(len(group_sum), training_mean, dtype=float),
                where=group_count > 0,
            )
            prediction = group_mean[code[holdout]]
            squared_error[index] += float(np.square(b[holdout] - prediction).sum())
    baseline_mse = baseline_squared_error / len(b)
    view_mse = squared_error / len(b)
    if baseline_mse > 0.0:
        raw_improvement = 1.0 - view_mse / baseline_mse
    else:
        raw_improvement = np.zeros(len(codes), dtype=float)
    reliability = np.maximum(raw_improvement, 0.0)
    if float(reliability.sum()) > 0.0:
        reliability_profile = reliability * (len(codes) / reliability.sum())
        shrinkage_strength = min(
            1.0,
            float(reliability.max()) / full_reliability_improvement,
        )
        weights = (1.0 - shrinkage_strength) * np.ones(
            len(codes), dtype=float
        ) + shrinkage_strength * reliability_profile
    else:
        shrinkage_strength = 0.0
        weights = np.ones(len(codes), dtype=float)
    normalized = normalize_view_weights(weights, view_count=len(codes))
    diagnostics = pd.DataFrame(
        {
            "view": views,
            "oof_mse": view_mse,
            "oof_global_mean_mse": baseline_mse,
            "oof_relative_improvement": raw_improvement,
            "nonnegative_reliability": reliability,
            "shrinkage_strength": shrinkage_strength,
            "normalized_view_weight": normalized,
        }
    )
    return normalized, diagnostics


def solve_adaptive_multiview_residual(
    baseline: Any,
    anchor: Any,
    codes: tuple[np.ndarray, ...],
    *,
    view_weights: Sequence[float],
    lambda_hypergraph: float,
    anchor_weight: float,
    observation_weight: Any | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> tuple[np.ndarray, bool, int]:
    """Solve a signed quadratic residual with nonnegative view weights."""

    b = _vector(baseline, field="baseline", allow_missing=False)
    r = _vector(anchor, field="anchor", allow_missing=True)
    if b.shape != r.shape:
        raise ValueError("baseline and anchor must have equal length")
    if not codes or any(code.shape != b.shape for code in codes):
        raise ValueError("every view code must match the baseline length")
    weights_by_view = np.asarray(
        normalize_view_weights(view_weights, view_count=len(codes)), dtype=float
    )
    if lambda_hypergraph < 0.0 or not np.isfinite(lambda_hypergraph):
        raise ValueError("lambda_hypergraph must be finite and nonnegative")
    if anchor_weight < 0.0 or not np.isfinite(anchor_weight):
        raise ValueError("anchor_weight must be finite and nonnegative")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("solver tolerance and max_iterations must be positive")
    if observation_weight is None:
        observed_weight = np.ones(len(b), dtype=float)
    else:
        observed_weight = _vector(
            observation_weight,
            field="observation_weight",
            allow_missing=False,
        )
        if observed_weight.shape != b.shape or (observed_weight < 0.0).any():
            raise ValueError(
                "observation_weight must be nonnegative and match baseline length"
            )
    anchor_available = np.isfinite(r).astype(float)
    anchor_value = np.nan_to_num(r, nan=0.0)
    training = observed_weight > 0.0
    fallback = float(np.average(b, weights=observed_weight)) if training.any() else 0.0
    theta = np.where(
        training,
        b,
        np.where(anchor_available > 0.0, anchor_value, fallback),
    )
    denominator = (
        observed_weight
        + anchor_weight * anchor_available
        + lambda_hypergraph * float(weights_by_view.sum())
    )
    for iteration in range(1, max_iterations + 1):
        projection = np.zeros(len(b), dtype=float)
        for weight, code in zip(weights_by_view, codes, strict=True):
            if weight > 0.0:
                projection += weight * _group_projection(theta, code)
        numerator = (
            observed_weight * b
            + anchor_weight * anchor_available * anchor_value
            + lambda_hypergraph * projection
        )
        updated = np.divide(
            numerator,
            denominator,
            out=np.full(len(b), fallback, dtype=float),
            where=denominator > 0.0,
        )
        difference = float(np.max(np.abs(updated - theta)))
        theta = updated
        if difference <= tolerance:
            return theta, True, iteration
    return theta, False, max_iterations


def _validated_profiles(
    profiles: Mapping[str, Sequence[float]], *, view_count: int
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if "all_equal" not in profiles:
        raise ValueError("adaptive profiles must include all_equal")
    result = tuple(
        (
            str(name),
            normalize_view_weights(weights, view_count=view_count),
        )
        for name, weights in profiles.items()
    )
    if len({name for name, _ in result}) != len(result):
        raise ValueError("adaptive profile names must be unique")
    equal = dict(result)["all_equal"]
    if not np.allclose(equal, np.ones(view_count), rtol=0.0, atol=1e-12):
        raise ValueError("all_equal must assign equal mass to every view")
    return result


def fit_adaptive_cross_validated_residual(
    edges: pd.DataFrame,
    baseline: Any,
    anchor: Any,
    *,
    profiles: Mapping[str, Sequence[float]],
    views: tuple[str, ...],
    key_columns: tuple[str, ...],
    lambda_grid: tuple[float, ...] = (0.05, 0.25, 1.0),
    anchor_grid: tuple[float, ...] = (0.0, 0.1, 0.25),
    folds: int = 3,
    seed: int = 1729,
    minimum_profile_relative_improvement: float = 0.005,
    minimum_cv_improvement: float = 0.01,
    full_gate_improvement: float = 0.05,
) -> tuple[np.ndarray, AdaptiveMultiviewResidualFit, pd.DataFrame]:
    """Select a conservative view profile and gated signed residual."""

    b = _vector(baseline, field="baseline", allow_missing=False)
    r = robustly_scale_anchor(b, anchor)
    if len(edges) != len(b):
        raise ValueError("edge table and baseline length mismatch")
    if minimum_profile_relative_improvement < 0.0:
        raise ValueError("profile improvement threshold must be nonnegative")
    if minimum_cv_improvement < 0.0 or full_gate_improvement <= 0.0:
        raise ValueError("CV gate thresholds must be nonnegative and positive")
    codes = multiview_codes(edges, views)
    candidates = _validated_profiles(profiles, view_count=len(codes))
    fold_id = stable_edge_folds(edges, key_columns=key_columns, folds=folds, seed=seed)
    records: list[dict[str, object]] = []
    for profile_priority, (profile_name, view_weights) in enumerate(candidates):
        for lambda_hypergraph in lambda_grid:
            for anchor_weight in anchor_grid:
                if lambda_hypergraph == 0.0 and anchor_weight == 0.0:
                    continue
                squared_errors: list[np.ndarray] = []
                baseline_errors: list[np.ndarray] = []
                converged = True
                iterations = 0
                for fold in range(folds):
                    holdout = fold_id == fold
                    prediction, fold_converged, fold_iterations = (
                        solve_adaptive_multiview_residual(
                            b,
                            r,
                            codes,
                            view_weights=view_weights,
                            lambda_hypergraph=float(lambda_hypergraph),
                            anchor_weight=float(anchor_weight),
                            observation_weight=(~holdout).astype(float),
                        )
                    )
                    training_mean = float(b[~holdout].mean())
                    squared_errors.append(np.square(prediction[holdout] - b[holdout]))
                    baseline_errors.append(np.square(training_mean - b[holdout]))
                    converged = converged and fold_converged
                    iterations = max(iterations, fold_iterations)
                mse = float(np.concatenate(squared_errors).mean())
                baseline_mse = float(np.concatenate(baseline_errors).mean())
                improvement = 1.0 - mse / baseline_mse if baseline_mse > 0.0 else 0.0
                records.append(
                    {
                        "profile_name": profile_name,
                        "profile_priority": profile_priority,
                        "view_weights": json_view_weights(view_weights),
                        "lambda_hypergraph": float(lambda_hypergraph),
                        "anchor_weight": float(anchor_weight),
                        "cv_mse": mse,
                        "cv_baseline_mse": baseline_mse,
                        "cv_relative_improvement": improvement,
                        "converged": converged,
                        "iterations": iterations,
                    }
                )
    diagnostics = pd.DataFrame.from_records(records).sort_values(
        [
            "cv_mse",
            "profile_priority",
            "lambda_hypergraph",
            "anchor_weight",
        ],
        kind="stable",
        ignore_index=True,
    )
    if diagnostics.empty:
        raise ValueError("adaptive residual candidate grid is empty")
    best = diagnostics.iloc[0]
    equal_best = (
        diagnostics.loc[diagnostics["profile_name"].eq("all_equal")]
        .sort_values(["cv_mse", "lambda_hypergraph", "anchor_weight"], kind="stable")
        .iloc[0]
    )
    equal_mse = float(equal_best["cv_mse"])
    profile_gain = 1.0 - float(best["cv_mse"]) / equal_mse if equal_mse > 0.0 else 0.0
    use_adaptive = (
        str(best["profile_name"]) != "all_equal"
        and profile_gain >= minimum_profile_relative_improvement
    )
    selected = best if use_adaptive else equal_best
    diagnostics["profile_relative_improvement_over_equal"] = profile_gain
    diagnostics["selected"] = False
    diagnostics.loc[selected.name, "selected"] = True
    diagnostics["selection_reason"] = np.where(
        diagnostics["selected"],
        "adaptive_profile_cv_gain_passed"
        if use_adaptive
        else "all_equal_conservative_fallback",
        None,
    )

    improvement = float(selected["cv_relative_improvement"])
    gate = (
        0.0
        if improvement < minimum_cv_improvement
        else min(1.0, improvement / full_gate_improvement)
    )
    selected_weights = dict(candidates)[str(selected["profile_name"])]
    raw_theta, converged, iterations = solve_adaptive_multiview_residual(
        b,
        r,
        codes,
        view_weights=selected_weights,
        lambda_hypergraph=float(selected["lambda_hypergraph"]),
        anchor_weight=float(selected["anchor_weight"]),
    )
    theta = b + gate * (raw_theta - b)
    if gate == 0.0 and not np.array_equal(theta, b):
        raise AssertionError("zero adaptive gate must preserve LIANA exactly")
    fit = AdaptiveMultiviewResidualFit(
        profile_name=str(selected["profile_name"]),
        view_weights=selected_weights,
        lambda_hypergraph=float(selected["lambda_hypergraph"]),
        anchor_weight=float(selected["anchor_weight"]),
        cv_mse=float(selected["cv_mse"]),
        cv_baseline_mse=float(selected["cv_baseline_mse"]),
        cv_relative_improvement=improvement,
        profile_relative_improvement_over_equal=profile_gain,
        fallback_gate=gate,
        folds=folds,
        views=views,
        converged=converged,
        iterations=iterations,
    )
    return theta, fit, diagnostics


def json_view_weights(weights: Sequence[float]) -> str:
    """Serialize normalized weights without locale-dependent formatting."""

    return "[" + ",".join(f"{float(value):.12g}" for value in weights) + "]"


__all__ = [
    "AdaptiveMultiviewResidualFit",
    "estimate_oof_view_weights",
    "fit_adaptive_cross_validated_residual",
    "json_view_weights",
    "normalize_view_weights",
    "solve_adaptive_multiview_residual",
]
