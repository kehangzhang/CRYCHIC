"""Scale-invariant paired-condition contrast for benchmark pair rankings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class PairedRankContrastFit:
    """Diagnostics for a benchmark-only paired rank contrast."""

    subtraction_fraction: float
    target_observed_pairs: int
    reference_observed_pairs: int
    common_observed_pairs: int
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    eligible = observed & np.isfinite(values)
    count = int(eligible.sum())
    if count:
        result[eligible] = rankdata(values[eligible], method="average") / count
    return result


def paired_rank_contrast(
    target_score: Any,
    reference_score: Any,
    *,
    subtraction_fraction: float,
    target_observed: Any | None = None,
    reference_observed: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, PairedRankContrastFit]:
    """Subtract the opposite-condition percentile while preserving missingness."""

    target = np.asarray(target_score, dtype=float)
    reference = np.asarray(reference_score, dtype=float)
    if target.ndim != 1 or reference.shape != target.shape:
        raise ValueError("target and reference scores must be equal-length vectors")
    if not np.isfinite(subtraction_fraction) or not (
        0.0 <= subtraction_fraction <= 1.0
    ):
        raise ValueError("subtraction_fraction must lie in [0, 1]")
    target_mask = (
        np.isfinite(target)
        if target_observed is None
        else np.asarray(target_observed).astype(bool, copy=False)
    )
    reference_mask = (
        np.isfinite(reference)
        if reference_observed is None
        else np.asarray(reference_observed).astype(bool, copy=False)
    )
    if target_mask.shape != target.shape or reference_mask.shape != target.shape:
        raise ValueError("observed masks must match score vectors")
    target_rank = _percentile(target, target_mask)
    reference_rank = _percentile(reference, reference_mask)
    target_result = target_rank.copy()
    reference_result = reference_rank.copy()
    common = np.isfinite(target_rank) & np.isfinite(reference_rank)
    target_result[common] = np.maximum(
        target_rank[common] - subtraction_fraction * reference_rank[common], 0.0
    )
    reference_result[common] = np.maximum(
        reference_rank[common] - subtraction_fraction * target_rank[common], 0.0
    )
    fit = PairedRankContrastFit(
        subtraction_fraction=subtraction_fraction,
        target_observed_pairs=int(np.isfinite(target_rank).sum()),
        reference_observed_pairs=int(np.isfinite(reference_rank).sum()),
        common_observed_pairs=int(common.sum()),
    )
    return target_result, reference_result, fit
