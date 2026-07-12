"""Metrics that do not pretend real canonical datasets have edge-level truth."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def paired_descriptive_effects(
    table: pd.DataFrame,
    *,
    value: str,
    edge_keys: Sequence[str],
    subject_key: str = "subject_id",
    context_key: str = "context",
    reference: str,
    target: str,
) -> pd.DataFrame:
    """Summarize target-minus-reference paired effects without p-values."""
    needed = {*edge_keys, subject_key, context_key, value}
    missing = needed.difference(table.columns)
    if missing:
        raise ValueError(f"paired effect table is missing columns: {sorted(missing)}")
    subset = table.loc[table[context_key].isin([reference, target])].copy()
    duplicate_keys = [*edge_keys, subject_key, context_key]
    if subset.duplicated(duplicate_keys).any():
        raise ValueError("paired effect input has duplicate edge-subject-context rows")
    wide = subset.pivot(
        index=[*edge_keys, subject_key], columns=context_key, values=value
    )
    if reference not in wide.columns or target not in wide.columns:
        columns = [
            *edge_keys,
            "effect",
            "diagnostic_standard_error",
            "standardized_effect",
            "median_effect",
            "direction_consistency",
            "n_pairs",
            "status",
            "reason_code",
        ]
        return pd.DataFrame(columns=columns)
    paired = wide[[reference, target]].dropna()
    paired["difference"] = paired[target] - paired[reference]

    records: list[dict[str, object]] = []
    levels = list(range(len(edge_keys)))
    for key, group in paired.groupby(level=levels, sort=False, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        difference = group["difference"].to_numpy(dtype=float)
        n_pairs = len(difference)
        effect = float(np.mean(difference))
        standard_error = (
            float(np.std(difference, ddof=1) / np.sqrt(n_pairs))
            if n_pairs >= 2
            else np.nan
        )
        standard_deviation = (
            float(np.std(difference, ddof=1)) if n_pairs >= 2 else np.nan
        )
        standardized = (
            effect / standard_deviation
            if np.isfinite(standard_deviation) and standard_deviation > 0
            else np.nan
        )
        if effect > 0:
            consistency = float(np.mean(difference > 0))
        elif effect < 0:
            consistency = float(np.mean(difference < 0))
        else:
            consistency = float(np.mean(difference == 0))
        row: dict[str, object] = dict(zip(edge_keys, key_tuple, strict=True))
        row.update(
            {
                "effect": effect,
                "diagnostic_standard_error": standard_error,
                "standardized_effect": standardized,
                "median_effect": float(np.median(difference)),
                "direction_consistency": consistency,
                "n_pairs": n_pairs,
                "status": "exploratory" if n_pairs >= 2 else "not_estimable",
                "reason_code": None if n_pairs >= 2 else "fewer_than_two_pairs",
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def top_k_jaccard(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: Sequence[str],
    score: str,
    k: int,
) -> float:
    """Return the Jaccard overlap of two deterministic top-k edge sets."""
    if k < 1:
        raise ValueError("k must be positive")
    for name, table in (("left", left), ("right", right)):
        missing = {*key, score}.difference(table.columns)
        if missing:
            raise ValueError(f"{name} table is missing columns: {sorted(missing)}")
    sort_columns = [score, *key]
    ascending = [False, *([True] * len(key))]
    left_top = left.sort_values(sort_columns, ascending=ascending).head(k)
    right_top = right.sort_values(sort_columns, ascending=ascending).head(k)
    left_set = set(map(tuple, left_top.loc[:, key].itertuples(index=False, name=None)))
    right_set = set(
        map(tuple, right_top.loc[:, key].itertuples(index=False, name=None))
    )
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else np.nan


def rank_concordance(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: Sequence[str],
    left_score: str,
    right_score: str,
) -> dict[str, float | int]:
    """Compute Spearman concordance only on shared, uniquely keyed rows."""
    if left.duplicated(list(key)).any() or right.duplicated(list(key)).any():
        raise ValueError("rank concordance requires unique keys in each table")
    merged = left[[*key, left_score]].merge(
        right[[*key, right_score]], on=list(key), how="inner", validate="one_to_one"
    )
    merged = merged.dropna(subset=[left_score, right_score])
    if len(merged) < 3:
        return {"spearman_rho": np.nan, "shared_rows": len(merged)}
    result = spearmanr(merged[left_score], merged[right_score])
    return {"spearman_rho": float(result.statistic), "shared_rows": len(merged)}


def paired_proportion_stability(
    proportions: pd.DataFrame,
    *,
    subject_key: str = "subject_id",
    context_key: str = "context",
    cell_type_key: str = "cell_type",
    value: str = "cell_proportion",
    reference: str,
    target: str,
) -> dict[str, float | int]:
    """Summarize paired composition stability across cell types."""
    needed = {subject_key, context_key, cell_type_key, value}
    missing = needed.difference(proportions.columns)
    if missing:
        raise ValueError(f"proportion table is missing columns: {sorted(missing)}")
    if proportions.duplicated([subject_key, context_key, cell_type_key]).any():
        raise ValueError(
            "proportion table has duplicate subject-context-cell_type rows"
        )
    wide = proportions.pivot(
        index=subject_key, columns=[context_key, cell_type_key], values=value
    ).fillna(0.0)
    common = sorted(
        set(wide.get(reference, pd.DataFrame()).columns)
        | set(wide.get(target, pd.DataFrame()).columns)
    )
    correlations: list[float] = []
    absolute_changes: list[float] = []
    paired_subjects = 0
    for subject in wide.index:
        ref = np.array(
            [wide.loc[subject].get((reference, cell_type), 0.0) for cell_type in common]
        )
        tgt = np.array(
            [wide.loc[subject].get((target, cell_type), 0.0) for cell_type in common]
        )
        if ref.sum() == 0 or tgt.sum() == 0:
            continue
        paired_subjects += 1
        absolute_changes.extend(np.abs(tgt - ref).tolist())
        correlation = spearmanr(ref, tgt).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
    return {
        "paired_subjects": paired_subjects,
        "median_subject_spearman": (
            float(np.median(correlations)) if correlations else np.nan
        ),
        "median_absolute_fraction_change": (
            float(np.median(absolute_changes)) if absolute_changes else np.nan
        ),
        "max_absolute_fraction_change": (
            float(np.max(absolute_changes)) if absolute_changes else np.nan
        ),
    }
