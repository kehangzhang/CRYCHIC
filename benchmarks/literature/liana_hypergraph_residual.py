"""Signed multi-view residual smoother with an exact LIANA fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_VIEWS = (
    "sender",
    "receiver",
    "ligand",
    "receptor",
    "interaction_id",
)


@dataclass(frozen=True)
class HypergraphResidualFit:
    """Cross-validated residual parameters and fallback decision."""

    lambda_hypergraph: float
    anchor_weight: float
    cv_mse: float
    cv_baseline_mse: float
    cv_relative_improvement: float
    fallback_gate: float
    folds: int
    views: tuple[str, ...]
    converged: bool
    iterations: int
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_vector(values: Any, *, field: str, allow_missing: bool) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{field} must be a one-dimensional vector")
    if not allow_missing and not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result


def multiview_codes(
    edges: pd.DataFrame, views: tuple[str, ...] = DEFAULT_VIEWS
) -> tuple[np.ndarray, ...]:
    """Encode each hypergraph view without materializing clique adjacency."""

    if not views or len(set(views)) != len(views):
        raise ValueError("views must contain unique column names")
    missing = set(views).difference(edges.columns)
    if missing:
        raise ValueError(f"hypergraph views are missing: {sorted(missing)}")
    codes = []
    for view in views:
        values = edges[view]
        if values.isna().any():
            raise ValueError(f"hypergraph view {view} contains missing labels")
        code, _ = pd.factorize(values.astype(str), sort=True)
        codes.append(code.astype(np.int64, copy=False))
    return tuple(codes)


def _group_projection(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    counts = np.bincount(codes)
    sums = np.bincount(codes, weights=values)
    return (sums / counts)[codes]


def solve_multiview_residual(
    baseline: Any,
    anchor: Any,
    codes: tuple[np.ndarray, ...],
    *,
    lambda_hypergraph: float,
    anchor_weight: float,
    observation_weight: Any | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> tuple[np.ndarray, bool, int]:
    """Solve the quadratic signed multi-view objective by fixed-point updates."""

    b = _validated_vector(baseline, field="baseline", allow_missing=False)
    r = _validated_vector(anchor, field="anchor", allow_missing=True)
    if b.shape != r.shape:
        raise ValueError("baseline and anchor must have equal length")
    if not codes or any(code.shape != b.shape for code in codes):
        raise ValueError("every hypergraph code vector must match baseline length")
    if lambda_hypergraph < 0.0 or not np.isfinite(lambda_hypergraph):
        raise ValueError("lambda_hypergraph must be finite and nonnegative")
    if anchor_weight < 0.0 or not np.isfinite(anchor_weight):
        raise ValueError("anchor_weight must be finite and nonnegative")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("solver tolerance and max_iterations must be positive")
    if observation_weight is None:
        observed_weight = np.ones(len(b), dtype=float)
    else:
        observed_weight = _validated_vector(
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
    fallback = (
        float(np.average(b, weights=observed_weight)) if training.any() else 0.0
    )
    theta = np.where(
        training,
        b,
        np.where(anchor_available > 0.0, anchor_value, fallback),
    )
    denominator = (
        observed_weight
        + anchor_weight * anchor_available
        + lambda_hypergraph * len(codes)
    )
    for iteration in range(1, max_iterations + 1):
        projection = sum(_group_projection(theta, code) for code in codes)
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


def robustly_scale_anchor(baseline: Any, anchor: Any) -> np.ndarray:
    """Put a signed anchor on the baseline MAD scale without shifting zero."""

    b = _validated_vector(baseline, field="baseline", allow_missing=False)
    r = _validated_vector(anchor, field="anchor", allow_missing=True)
    if b.shape != r.shape:
        raise ValueError("baseline and anchor must have equal length")
    finite = np.isfinite(r)
    result = np.full(len(r), np.nan, dtype=float)
    if finite.sum() < 20:
        return result
    baseline_mad = float(np.median(np.abs(b - np.median(b))))
    anchor_values = r[finite]
    anchor_mad = float(
        np.median(np.abs(anchor_values - np.median(anchor_values)))
    )
    if baseline_mad <= 0.0 or anchor_mad <= 0.0:
        return result
    result[finite] = anchor_values * (baseline_mad / anchor_mad)
    return result


def stable_edge_folds(
    edges: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    folds: int,
    seed: int,
) -> np.ndarray:
    """Assign deterministic edge folds from biological identity columns."""

    if folds < 2:
        raise ValueError("folds must be at least two")
    missing = set(key_columns).difference(edges.columns)
    if missing:
        raise ValueError(f"edge fold keys are missing: {sorted(missing)}")
    if edges.duplicated(list(key_columns)).any():
        raise ValueError("edge fold keys must uniquely identify rows")
    hashed = pd.util.hash_pandas_object(
        edges.loc[:, list(key_columns)].astype(str),
        index=False,
        hash_key="0123456789123456",
    ).to_numpy(dtype=np.uint64)
    mixed = hashed ^ np.uint64(seed * 0x9E3779B1)
    return (mixed % np.uint64(folds)).astype(np.int64)


def fit_cross_validated_residual(
    edges: pd.DataFrame,
    baseline: Any,
    anchor: Any,
    *,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    key_columns: tuple[str, ...] = (
        "sender",
        "receiver",
        "interaction_id",
    ),
    lambda_grid: tuple[float, ...] = (0.05, 0.25, 1.0),
    anchor_grid: tuple[float, ...] = (0.0, 0.1, 0.25),
    folds: int = 3,
    seed: int = 1729,
    minimum_cv_improvement: float = 0.01,
    full_gate_improvement: float = 0.05,
) -> tuple[np.ndarray, HypergraphResidualFit, pd.DataFrame]:
    """Select residual penalties internally and return a gated full-data fit."""

    b = _validated_vector(baseline, field="baseline", allow_missing=False)
    r = robustly_scale_anchor(b, anchor)
    if len(edges) != len(b):
        raise ValueError("edge table and baseline length mismatch")
    if minimum_cv_improvement < 0.0 or full_gate_improvement <= 0.0:
        raise ValueError("CV gate thresholds must be nonnegative and positive")
    codes = multiview_codes(edges, views)
    fold_id = stable_edge_folds(
        edges, key_columns=key_columns, folds=folds, seed=seed
    )
    records: list[dict[str, object]] = []
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
                weights = (~holdout).astype(float)
                prediction, fold_converged, fold_iterations = (
                    solve_multiview_residual(
                        b,
                        r,
                        codes,
                        lambda_hypergraph=float(lambda_hypergraph),
                        anchor_weight=float(anchor_weight),
                        observation_weight=weights,
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
        ["cv_mse", "lambda_hypergraph", "anchor_weight"],
        kind="stable",
        ignore_index=True,
    )
    if diagnostics.empty:
        raise ValueError("residual candidate grid contains no estimable candidate")
    selected = diagnostics.iloc[0]
    improvement = float(selected["cv_relative_improvement"])
    gate = (
        0.0
        if improvement < minimum_cv_improvement
        else min(1.0, improvement / full_gate_improvement)
    )
    raw_theta, converged, iterations = solve_multiview_residual(
        b,
        r,
        codes,
        lambda_hypergraph=float(selected["lambda_hypergraph"]),
        anchor_weight=float(selected["anchor_weight"]),
    )
    theta = b + gate * (raw_theta - b)
    if gate == 0.0 and not np.array_equal(theta, b):
        raise AssertionError("zero residual gate must preserve LIANA baseline exactly")
    fit = HypergraphResidualFit(
        lambda_hypergraph=float(selected["lambda_hypergraph"]),
        anchor_weight=float(selected["anchor_weight"]),
        cv_mse=float(selected["cv_mse"]),
        cv_baseline_mse=float(selected["cv_baseline_mse"]),
        cv_relative_improvement=improvement,
        fallback_gate=gate,
        folds=folds,
        views=views,
        converged=converged,
        iterations=iterations,
    )
    return theta, fit, diagnostics


def conservative_sign_statistic(
    baseline: Any,
    residual: Any,
    fit: HypergraphResidualFit,
    *,
    maximum_abs_baseline: float,
) -> np.ndarray:
    """Use residual signs only where the baseline direction is uncertain."""

    b = _validated_vector(baseline, field="baseline", allow_missing=False)
    theta = _validated_vector(residual, field="residual", allow_missing=False)
    if b.shape != theta.shape:
        raise ValueError("baseline and residual must have equal length")
    if maximum_abs_baseline <= 0.0 or not np.isfinite(maximum_abs_baseline):
        raise ValueError("maximum_abs_baseline must be finite and positive")
    if fit.fallback_gate == 0.0:
        return b.copy()
    uncertain = np.abs(b) <= maximum_abs_baseline
    return np.where(uncertain, theta, b)


__all__ = [
    "DEFAULT_VIEWS",
    "HypergraphResidualFit",
    "conservative_sign_statistic",
    "fit_cross_validated_residual",
    "multiview_codes",
    "robustly_scale_anchor",
    "solve_multiview_residual",
    "stable_edge_folds",
]
