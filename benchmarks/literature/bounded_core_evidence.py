"""Bounded rank evidence blending for benchmark-only multi-context heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BoundedCoreEvidencePolicy:
    """Frozen weights for a base ranking and one secondary evidence ranking."""

    expert_fraction: float
    missing_base_penalty: float
    fit_status: str = "benchmark_candidate_unreleased"
    formal_inference_allowed: bool = False

    def __post_init__(self) -> None:
        for field, value in (
            ("expert_fraction", self.expert_fraction),
            ("missing_base_penalty", self.missing_base_penalty),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must lie in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedCoreEvidenceDiagnostics:
    """Coverage and scale diagnostics for one condition-specific blend."""

    condition: str
    rows: int
    base_observed: int
    expert_observed: int
    common_observed: int
    base_only: int
    expert_only: int
    neither_observed: int
    base_observed_mean: float
    mapped_expert_mean_on_common: float
    blended_mean_on_common: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_KEY_COLUMNS = ("condition", "sender", "receiver")
_VALUE_COLUMNS = ("ranked_strength", "estimable_directed_lr", "status")


def _validate_rankings(table: pd.DataFrame, *, name: str) -> None:
    missing = set((*_KEY_COLUMNS, *_VALUE_COLUMNS)).difference(table.columns)
    if missing:
        raise ValueError(f"{name} is missing ranking columns: {sorted(missing)}")
    if table.empty:
        raise ValueError(f"{name} must not be empty")
    if table.duplicated(list(_KEY_COLUMNS)).any():
        raise ValueError(f"{name} ranking keys must be unique")
    score = pd.to_numeric(table["ranked_strength"], errors="coerce")
    supplied = table["ranked_strength"].notna()
    if (supplied & score.isna()).any() or np.isinf(score.dropna()).any():
        raise ValueError(f"{name}.ranked_strength must be finite or missing")
    if (score.dropna() < 0.0).any():
        raise ValueError(f"{name}.ranked_strength must be non-negative")
    observed = table["status"].astype(str).eq("observed")
    if score.loc[observed].isna().any():
        raise ValueError(f"{name} observed rows require ranked_strength")


def _quantile_match(
    expert: pd.Series,
    *,
    expert_observed: pd.Series,
    reference: pd.Series,
    reference_observed: pd.Series,
) -> pd.Series:
    """Map expert ranks onto the empirical scale of an observed reference."""

    result = pd.Series(np.nan, index=expert.index, dtype=float)
    source = pd.to_numeric(expert.loc[expert_observed], errors="raise").astype(float)
    target = np.sort(
        pd.to_numeric(reference.loc[reference_observed], errors="raise").to_numpy(
            dtype=float
        )
    )
    if source.empty or target.size == 0:
        return result
    ranks = source.rank(method="average").to_numpy(dtype=float)
    percentiles = (
        (ranks - 1.0) / (len(source) - 1.0)
        if len(source) > 1
        else np.full(1, 0.5, dtype=float)
    )
    result.loc[source.index] = np.quantile(target, percentiles, method="linear")
    return result


def bounded_core_evidence_rankings(
    base: pd.DataFrame,
    expert: pd.DataFrame,
    *,
    policy: BoundedCoreEvidencePolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend condition-specific rankings without converting missingness to zero.

    The expert is quantile-matched to the base within each condition before a
    convex blend. An expert-only row receives only the expert's allocated mass,
    multiplied by the frozen missing-base penalty. This head is a ranking
    diagnostic and never changes the inferential status of either source.
    """

    _validate_rankings(base, name="base")
    _validate_rankings(expert, name="expert")
    base_values = base.loc[:, [*_KEY_COLUMNS, *_VALUE_COLUMNS]].rename(
        columns={
            "ranked_strength": "_base_score",
            "estimable_directed_lr": "_base_estimable",
            "status": "_base_status",
        }
    )
    expert_values = expert.loc[:, [*_KEY_COLUMNS, *_VALUE_COLUMNS]].rename(
        columns={
            "ranked_strength": "_expert_score",
            "estimable_directed_lr": "_expert_estimable",
            "status": "_expert_status",
        }
    )
    merged = base_values.merge(
        expert_values,
        on=list(_KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        sort=True,
    )
    merged["_mapped_expert"] = np.nan
    diagnostics: list[dict[str, Any]] = []
    fraction = float(policy.expert_fraction)
    for condition, index in merged.groupby("condition", observed=True).groups.items():
        selected = merged.loc[index]
        base_observed = (
            selected["_base_status"].astype(str).eq("observed")
            & pd.to_numeric(selected["_base_score"], errors="coerce").notna()
        )
        expert_observed = (
            selected["_expert_status"].astype(str).eq("observed")
            & pd.to_numeric(selected["_expert_score"], errors="coerce").notna()
        )
        mapped = _quantile_match(
            selected["_expert_score"],
            expert_observed=expert_observed,
            reference=selected["_base_score"],
            reference_observed=base_observed,
        )
        common = base_observed & expert_observed & mapped.notna()
        if common.any() and float(mapped.loc[common].mean()) > 0.0:
            mapped *= float(
                pd.to_numeric(
                    selected.loc[common, "_base_score"], errors="raise"
                ).mean()
                / mapped.loc[common].mean()
            )
        merged.loc[index, "_mapped_expert"] = mapped.to_numpy(copy=False)
        base_score = pd.to_numeric(
            selected.loc[common, "_base_score"], errors="raise"
        ).astype(float)
        mapped_common = mapped.loc[common]
        blended = (1.0 - fraction) * base_score + fraction * mapped_common
        diagnostics.append(
            BoundedCoreEvidenceDiagnostics(
                condition=str(condition),
                rows=len(selected),
                base_observed=int(base_observed.sum()),
                expert_observed=int(expert_observed.sum()),
                common_observed=int(common.sum()),
                base_only=int((base_observed & ~expert_observed).sum()),
                expert_only=int((~base_observed & expert_observed).sum()),
                neither_observed=int((~base_observed & ~expert_observed).sum()),
                base_observed_mean=(
                    float(base_score.mean()) if len(base_score) else float("nan")
                ),
                mapped_expert_mean_on_common=(
                    float(mapped_common.mean()) if len(mapped_common) else float("nan")
                ),
                blended_mean_on_common=(
                    float(blended.mean()) if len(blended) else float("nan")
                ),
            ).to_dict()
        )

    base_score = pd.to_numeric(merged["_base_score"], errors="coerce")
    mapped_expert = pd.to_numeric(merged["_mapped_expert"], errors="coerce")
    base_observed = (
        merged["_base_status"].astype(str).eq("observed") & base_score.notna()
    )
    expert_observed = (
        merged["_expert_status"].astype(str).eq("observed") & mapped_expert.notna()
    )
    common = base_observed & expert_observed
    base_only = base_observed & ~expert_observed
    expert_only = ~base_observed & expert_observed
    if fraction == 0.0:
        expert_only = pd.Series(False, index=merged.index, dtype=bool)
    result = merged.loc[:, list(_KEY_COLUMNS)].copy()
    result["ranked_strength"] = np.nan
    result.loc[common, "ranked_strength"] = (1.0 - fraction) * base_score.loc[
        common
    ] + fraction * mapped_expert.loc[common]
    result.loc[base_only, "ranked_strength"] = base_score.loc[base_only]
    result.loc[expert_only, "ranked_strength"] = (
        fraction * float(policy.missing_base_penalty) * mapped_expert.loc[expert_only]
    )
    result["estimable_directed_lr"] = pd.to_numeric(
        merged["_base_estimable"], errors="coerce"
    ).where(
        pd.to_numeric(merged["_base_estimable"], errors="coerce").notna(),
        pd.to_numeric(merged["_expert_estimable"], errors="coerce"),
    )
    result["status"] = np.where(
        result["ranked_strength"].notna(), "observed", "not_estimable"
    )
    result["reason_code"] = np.select(
        (common, base_only, expert_only),
        (
            "base_and_quantile_matched_expert_observed",
            "expert_not_estimable_used_base_only",
            "base_not_estimable_used_penalized_expert_only",
        ),
        default="base_and_expert_not_estimable",
    )
    return (
        result.sort_values(list(_KEY_COLUMNS), kind="stable", ignore_index=True),
        pd.DataFrame.from_records(diagnostics).sort_values(
            "condition", kind="stable", ignore_index=True
        ),
    )


__all__ = [
    "BoundedCoreEvidenceDiagnostics",
    "BoundedCoreEvidencePolicy",
    "bounded_core_evidence_rankings",
]
