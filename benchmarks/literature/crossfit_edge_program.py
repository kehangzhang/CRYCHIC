"""Label-free cross-fitted low-rank reconstruction of subject-edge scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossfitProgramFit:
    """Aggregate held-out reconstruction diagnostics for one candidate."""

    candidate: str
    rank: int
    projection_shrinkage: float
    baseline_mse: float
    reconstruction_mse: float
    relative_improvement: float
    reliability: float
    heldout_values: int
    subjects: int
    edges: int
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_candidates(
    candidates: Any,
) -> tuple[tuple[str, int, float], ...]:
    collected = []
    for value in candidates:
        if not isinstance(value, dict):
            raise ValueError("program candidates must contain mappings")
        name = value.get("name")
        rank = value.get("rank")
        shrinkage = value.get("projection_shrinkage")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("program candidate names must be canonical strings")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("program candidate rank must be a nonnegative integer")
        if (
            isinstance(shrinkage, bool)
            or not isinstance(shrinkage, int | float)
            or not np.isfinite(shrinkage)
            or not 0.0 <= float(shrinkage) <= 1.0
        ):
            raise ValueError("projection_shrinkage must lie in [0, 1]")
        if rank == 0 and float(shrinkage) != 0.0:
            raise ValueError("rank-zero reference must have zero shrinkage")
        collected.append((name, rank, float(shrinkage)))
    if not collected or len({value[0] for value in collected}) != len(collected):
        raise ValueError("program candidates must be non-empty and uniquely named")
    return tuple(collected)


def _edge_means(training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(training)
    count = finite.sum(axis=1)
    total = np.where(finite, training, 0.0).sum(axis=1)
    mean = np.divide(
        total,
        count,
        out=np.full(training.shape[0], np.nan, dtype=float),
        where=count > 0,
    )
    return mean, count


def crossfit_program_candidates(
    matrix: Any,
    candidates: Any,
    *,
    minimum_training_subjects: int = 3,
    minimum_edges: int = 200,
    minimum_relative_improvement: float = 0.01,
    full_reliability_improvement: float = 0.05,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Reconstruct each subject from loadings fitted without that subject."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 4:
        raise ValueError("matrix must be a non-empty edge-by-subject array")
    finite = values[np.isfinite(values)]
    if finite.size and ((finite < 0.0) | (finite > 1.0)).any():
        raise ValueError("finite program inputs must lie in [0, 1]")
    if minimum_training_subjects < 2:
        raise ValueError("minimum_training_subjects must be at least two")
    if minimum_edges < 1:
        raise ValueError("minimum_edges must be positive")
    if (
        not np.isfinite(minimum_relative_improvement)
        or minimum_relative_improvement < 0.0
        or minimum_relative_improvement >= 1.0
    ):
        raise ValueError("minimum_relative_improvement must lie in [0, 1)")
    if (
        not np.isfinite(full_reliability_improvement)
        or full_reliability_improvement <= minimum_relative_improvement
        or full_reliability_improvement > 1.0
    ):
        raise ValueError(
            "full_reliability_improvement must exceed the minimum and be at most one"
        )
    specs = _validated_candidates(candidates)
    edge_count, subject_count = values.shape
    max_rank = max(rank for _, rank, _ in specs)
    reconstructed = {
        name: np.full_like(values, np.nan, dtype=float) for name, _, _ in specs
    }
    baseline_sse = {name: 0.0 for name, _, _ in specs}
    candidate_sse = {name: 0.0 for name, _, _ in specs}
    candidate_count = {name: 0 for name, _, _ in specs}
    fold_records: list[dict[str, object]] = []

    for heldout in range(subject_count):
        training = np.delete(values, heldout, axis=1)
        edge_mean, training_count = _edge_means(training)
        supported = training_count >= minimum_training_subjects
        heldout_values = values[:, heldout]
        evaluable = supported & np.isfinite(heldout_values)
        supported_edges = int(supported.sum())
        if supported_edges < minimum_edges or not evaluable.any():
            for name, rank, shrinkage in specs:
                reconstructed[name][:, heldout] = heldout_values
                fold_records.append(
                    {
                        "heldout_subject_index": heldout,
                        "candidate": name,
                        "rank": rank,
                        "projection_shrinkage": shrinkage,
                        "supported_edges": supported_edges,
                        "heldout_values": int(evaluable.sum()),
                        "baseline_mse": np.nan,
                        "reconstruction_mse": np.nan,
                        "reason_code": "insufficient_supported_edges",
                    }
                )
            continue

        train_supported = training[supported]
        means = edge_mean[supported]
        filled_training = np.where(
            np.isfinite(train_supported), train_supported, means[:, None]
        )
        centered_training = (filled_training - means[:, None]).T
        _, _, right = np.linalg.svd(centered_training, full_matrices=False)
        available_rank = min(max_rank, len(right), len(training) - 1)
        heldout_supported = heldout_values[supported]
        filled_heldout = np.where(
            np.isfinite(heldout_supported), heldout_supported, means
        )
        centered_heldout = filled_heldout - means
        baseline_prediction = means
        supported_evaluable = np.isfinite(heldout_supported)
        baseline_error = (
            heldout_supported[supported_evaluable]
            - baseline_prediction[supported_evaluable]
        )

        for name, rank, shrinkage in specs:
            if rank == 0 or available_rank == 0:
                prediction = heldout_supported.copy()
            else:
                used_rank = min(rank, available_rank)
                basis = right[:used_rank]
                projection = centered_heldout @ basis.T @ basis
                prediction = means + shrinkage * projection
            output = heldout_values.copy()
            output[supported] = prediction
            output[~np.isfinite(heldout_values)] = np.nan
            reconstructed[name][:, heldout] = output
            if rank == 0:
                reconstruction_error = baseline_error
            else:
                reconstruction_error = (
                    heldout_supported[supported_evaluable]
                    - prediction[supported_evaluable]
                )
            baseline_sum = float(np.dot(baseline_error, baseline_error))
            reconstruction_sum = float(
                np.dot(reconstruction_error, reconstruction_error)
            )
            count = int(supported_evaluable.sum())
            baseline_sse[name] += baseline_sum
            candidate_sse[name] += reconstruction_sum
            candidate_count[name] += count
            fold_records.append(
                {
                    "heldout_subject_index": heldout,
                    "candidate": name,
                    "rank": rank,
                    "projection_shrinkage": shrinkage,
                    "supported_edges": supported_edges,
                    "heldout_values": count,
                    "baseline_mse": baseline_sum / count,
                    "reconstruction_mse": reconstruction_sum / count,
                    "reason_code": None,
                }
            )

    fits = []
    outputs: dict[str, np.ndarray] = {}
    for name, rank, shrinkage in specs:
        count = candidate_count[name]
        baseline_mse = baseline_sse[name] / count if count else float("nan")
        reconstruction_mse = candidate_sse[name] / count if count else float("nan")
        if rank == 0:
            relative_improvement = 0.0
            reliability = 0.0
            outputs[name] = values.copy()
        elif (
            np.isfinite(baseline_mse)
            and baseline_mse > 0.0
            and np.isfinite(reconstruction_mse)
        ):
            relative_improvement = float(1.0 - reconstruction_mse / baseline_mse)
            reliability = float(
                np.clip(
                    (relative_improvement - minimum_relative_improvement)
                    / (full_reliability_improvement - minimum_relative_improvement),
                    0.0,
                    1.0,
                )
            )
            outputs[name] = values + reliability * (reconstructed[name] - values)
        else:
            relative_improvement = float("nan")
            reliability = 0.0
            outputs[name] = values.copy()
        np.clip(outputs[name], 0.0, 1.0, out=outputs[name])
        outputs[name][~np.isfinite(values)] = np.nan
        fit = CrossfitProgramFit(
            candidate=name,
            rank=rank,
            projection_shrinkage=shrinkage,
            baseline_mse=baseline_mse,
            reconstruction_mse=reconstruction_mse,
            relative_improvement=relative_improvement,
            reliability=reliability,
            heldout_values=count,
            subjects=subject_count,
            edges=edge_count,
        )
        fits.append(fit.to_dict())
    return (
        outputs,
        pd.DataFrame.from_records(fits),
        pd.DataFrame.from_records(fold_records),
    )


__all__ = ["CrossfitProgramFit", "crossfit_program_candidates"]
