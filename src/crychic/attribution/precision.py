"""Fold-frozen precision transforms for receiver attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from crychic.core import ContractError, stable_id


@dataclass(frozen=True, slots=True)
class PrecisionTransformResult:
    """Winsorized precision weights and their fitted transform provenance."""

    values: np.ndarray
    lower_quantile: float
    upper_quantile: float
    lower_bound: float | None
    upper_bound: float | None
    normalization_median: float | None
    n_positive_features: int
    min_positive_features: int
    precision_transform_id: str
    estimable: bool
    reason_code: str | None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).copy()
        if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ContractError(
                "Precision transform values must be a finite non-negative vector",
                code="invalid_precision_transform",
                field="values",
                remediation="Build precision weights with the released transform",
            )
        if int(np.count_nonzero(values)) != self.n_positive_features:
            raise ContractError(
                "Precision support count must match positive transformed weights",
                code="invalid_precision_transform",
                field="n_positive_features",
                remediation="Recompute precision transform support diagnostics",
            )
        if self.estimable != (self.n_positive_features >= self.min_positive_features):
            raise ContractError(
                "Precision estimability must follow the minimum feature gate",
                code="invalid_precision_transform",
                field="estimable",
                remediation="Apply the declared minimum positive feature count",
            )
        if self.estimable == (self.reason_code is not None):
            raise ContractError(
                "Precision reason_code must be present exactly when not estimable",
                code="invalid_precision_transform",
                field="reason_code",
                remediation="Record insufficient precision support explicitly",
            )
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


def winsorized_normalized_precision(
    raw_precision: np.ndarray,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    min_positive_features: int = 2,
) -> PrecisionTransformResult:
    """Winsorize positive precision and normalize its positive median to one."""

    raw = np.asarray(raw_precision, dtype=float)
    if raw.ndim != 1:
        raise ValueError("raw_precision must be one-dimensional")
    if (
        not math.isfinite(lower_quantile)
        or not math.isfinite(upper_quantile)
        or not 0 <= lower_quantile <= upper_quantile <= 1
    ):
        raise ValueError("precision quantiles must satisfy 0 <= lower <= upper <= 1")
    if min_positive_features < 1:
        raise ValueError("min_positive_features must be positive")

    valid = np.isfinite(raw) & (raw > 0)
    result = np.zeros(raw.shape, dtype=float)
    n_valid = int(np.count_nonzero(valid))
    lower_bound: float | None = None
    upper_bound: float | None = None
    normalization_median: float | None = None
    if n_valid:
        bounds = np.asarray(
            np.quantile(raw[valid], [lower_quantile, upper_quantile]),
            dtype=float,
        )
        lower_bound = float(bounds[0])
        upper_bound = float(bounds[1])
        clipped = np.clip(raw[valid], lower_bound, upper_bound)
        normalization_median = float(np.median(clipped))
        if normalization_median > 0:
            result[valid] = clipped / normalization_median

    n_positive = int(np.count_nonzero(result))
    estimable = n_positive >= min_positive_features
    reason_code = None if estimable else "insufficient_response_precision_support"
    transform_id = stable_id(
        "precision_transform",
        {
            "lower_bound": lower_bound,
            "lower_quantile": lower_quantile,
            "method": "winsorized_median_normalized_v1",
            "min_positive_features": min_positive_features,
            "normalization_median": normalization_median,
            "positive_feature_indices": np.flatnonzero(result > 0).tolist(),
            "upper_bound": upper_bound,
            "upper_quantile": upper_quantile,
        },
    )
    return PrecisionTransformResult(
        values=result,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        normalization_median=normalization_median,
        n_positive_features=n_positive,
        min_positive_features=min_positive_features,
        precision_transform_id=transform_id,
        estimable=estimable,
        reason_code=reason_code,
    )
