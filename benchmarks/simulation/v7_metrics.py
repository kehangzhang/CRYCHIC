"""Truth alignment and estimand-specific metrics for the v7 matrix."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.simulation.v7_integrated import (
    EFFECT_COLUMNS,
    PARENT_SENDER,
    SCORE_VIEW_COLUMNS,
)

SCHEMA_VERSION = "crychic-suggest-next2-v7-metrics-v1"

METRIC_COLUMNS = (
    "schema_version",
    "dataset_id",
    "dgp_family",
    "design_kind",
    "generator_id",
    "score_view",
    "estimand",
    "resolution",
    "inference_id",
    "contrast_name",
    "metric",
    "value",
    "status",
    "reason_code",
    "n_observed",
    "n_positive",
    "n_negative",
)

ALIGNED_EFFECT_COLUMNS = (
    *EFFECT_COLUMNS,
    "truth_known",
    "truth_label",
    "truth_effect",
    "truth_effect_kind",
)

_TRUTH_REQUIRED = {
    "contrast_name",
    "sender",
    "receiver",
    "interaction_id",
    "truth_score_effect",
    "truth_parent_effect",
    "truth_program_effect",
    "truth_causal_sender",
    "truth_causal_parent",
}


def _truth_boolean(value: object) -> bool | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    if type(value) in {bool, np.bool_}:
        return bool(value)
    raise ValueError("truth labels must contain booleans or missing values")


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric)]
    if len(finite) < 2:
        return pd.Series(np.nan, index=values.index, dtype=float)
    scale = float(np.std(finite, ddof=0))
    if scale <= 0.0 or not math.isfinite(scale):
        return pd.Series(0.0, index=values.index, dtype=float).where(
            numeric.notna(), np.nan
        )
    return (numeric - float(np.mean(finite))) / scale


def _parent_truth(truth: pd.DataFrame) -> pd.DataFrame:
    keys = ["contrast_name", "receiver", "interaction_id"]
    columns = [
        "truth_parent_effect",
        "truth_program_effect",
        "truth_causal_parent",
    ]
    consistency = truth.groupby(keys, observed=True)[columns].nunique(dropna=False)
    if consistency.gt(1).any(axis=None):
        raise ValueError("parent/program truth must be sender-invariant")
    result = truth.drop_duplicates(keys).loc[:, [*keys, *columns]].copy()
    result["sender"] = PARENT_SENDER
    return result


def _combined_parent_truth(parent: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, local in parent.groupby("contrast_name", observed=True, sort=False):
        local = local.copy()
        local["truth_combined_effect"] = _zscore(
            local["truth_parent_effect"]
        ) + 0.25 * _zscore(local["truth_program_effect"])
        program_nonzero = pd.to_numeric(
            local["truth_program_effect"], errors="coerce"
        ).abs().gt(0.0)
        local["truth_combined_label"] = [
            None
            if _truth_boolean(parent_label) is None
            else bool(_truth_boolean(parent_label) or program_active)
            for parent_label, program_active in zip(
                local["truth_causal_parent"], program_nonzero, strict=True
            )
        ]
        parts.append(local)
    return pd.concat(parts, ignore_index=True)


def align_v7_effect_truth(
    effects: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """Bind effects to the matching child, parent, program, or blend truth."""

    if tuple(effects.columns) != EFFECT_COLUMNS:
        raise ValueError("effects do not match the integrated v7 contract")
    if not isinstance(truth, pd.DataFrame):
        raise TypeError("truth must be a pandas DataFrame")
    missing = _TRUTH_REQUIRED.difference(truth.columns)
    if missing:
        raise ValueError(f"truth lacks v7 fields: {sorted(missing)}")
    child = truth.loc[
        :,
        [
            "contrast_name",
            "sender",
            "receiver",
            "interaction_id",
            "truth_score_effect",
            "truth_causal_sender",
        ],
    ].copy()
    if child.duplicated(
        ["contrast_name", "sender", "receiver", "interaction_id"]
    ).any():
        raise ValueError("child truth keys must be unique")
    parent = _parent_truth(truth)
    combined = _combined_parent_truth(parent)
    keys = ["contrast_name", "sender", "receiver", "interaction_id"]
    frames: list[pd.DataFrame] = []
    grouping = [
        "generator_id",
        "score_view",
        "estimand",
        "resolution",
        "inference_id",
    ]
    for group_key, effect_group in effects.groupby(
        grouping, observed=True, sort=False
    ):
        _, score_view, estimand, resolution, _ = group_key
        if score_view == "primary_fixed_effect_z_blend":
            source = combined.rename(
                columns={
                    "truth_combined_effect": "truth_effect",
                    "truth_combined_label": "truth_label",
                }
            )
            kind = "fixed_parent_program_z_blend"
        elif estimand == "signed_receiver_program":
            source = parent.rename(
                columns={
                    "truth_program_effect": "truth_effect",
                    "truth_causal_parent": "truth_label",
                }
            )
            kind = "program_effect"
        elif resolution == "lr_receiver_parent":
            source = parent.rename(
                columns={
                    "truth_parent_effect": "truth_effect",
                    "truth_causal_parent": "truth_label",
                }
            )
            kind = "parent_effect"
        else:
            source = child.rename(
                columns={
                    "truth_score_effect": "truth_effect",
                    "truth_causal_sender": "truth_label",
                }
            )
            kind = (
                "sender_identity_label"
                if estimand
                in {
                    "conditional_sender_attribution",
                    "training_fold_sender_identity_prior",
                }
                else "sender_score_effect"
            )
        selected = source.loc[:, [*keys, "truth_effect", "truth_label"]]
        merged = effect_group.merge(
            selected,
            on=keys,
            how="left",
            validate="many_to_one",
            sort=False,
        )
        merged["truth_known"] = [
            _truth_boolean(value) is not None for value in merged["truth_label"]
        ]
        merged["truth_label"] = pd.array(
            [_truth_boolean(value) for value in merged["truth_label"]],
            dtype="boolean",
        )
        merged["truth_effect"] = pd.to_numeric(
            merged["truth_effect"], errors="coerce"
        ).astype(float)
        merged["truth_effect_kind"] = kind
        frames.append(merged.loc[:, list(ALIGNED_EFFECT_COLUMNS)])
    if not frames:
        return pd.DataFrame(columns=ALIGNED_EFFECT_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(
        [
            "generator_id",
            "inference_id",
            "score_view",
            "contrast_name",
            "event_id",
        ],
        kind="stable",
        ignore_index=True,
    )


def _metric_row(
    group: dict[str, object],
    *,
    metric: str,
    value: float | None,
    n_observed: int,
    n_positive: int,
    n_negative: int,
    reason_code: str | None = None,
) -> dict[str, object]:
    observed = value is not None and math.isfinite(value)
    return {
        "schema_version": SCHEMA_VERSION,
        **group,
        "metric": metric,
        "value": value if observed else None,
        "status": "observed" if observed else "not_estimable",
        "reason_code": None if observed else reason_code or "metric_not_estimable",
        "n_observed": n_observed,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def _binary_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    function: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float | None, str | None]:
    if len(labels) < 2 or len(np.unique(labels)) != 2:
        return None, "truth_is_single_class"
    try:
        value = float(function(labels, scores))
    except ValueError:
        return None, "binary_metric_failed"
    return (value, None) if math.isfinite(value) else (None, "binary_metric_nonfinite")


def _effect_metric_rows(
    table: pd.DataFrame,
    *,
    dgp_family: str,
    design_kind: str,
) -> list[dict[str, object]]:
    first = table.iloc[0]
    group = {
        "dataset_id": str(first["dataset_id"]),
        "dgp_family": dgp_family,
        "design_kind": design_kind,
        "generator_id": str(first["generator_id"]),
        "score_view": str(first["score_view"]),
        "estimand": str(first["estimand"]),
        "resolution": str(first["resolution"]),
        "inference_id": str(first["inference_id"]),
        "contrast_name": str(first["contrast_name"]),
    }
    ranking = pd.to_numeric(table["ranking_score"], errors="coerce")
    predicted = pd.to_numeric(table["effect"], errors="coerce")
    truth_effect = pd.to_numeric(table["truth_effect"], errors="coerce")
    known = table["truth_known"].astype(bool)
    usable_rank = known & ranking.notna() & np.isfinite(ranking)
    labels = table.loc[usable_rank, "truth_label"].astype(bool).to_numpy()
    scores = ranking.loc[usable_rank].to_numpy(dtype=float)
    n_observed = int(usable_rank.sum())
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - labels.sum())
    rows: list[dict[str, object]] = []
    for metric, function in (
        ("event_auprc", average_precision_score),
        ("event_auroc", roc_auc_score),
        (
            "event_partial_auroc_fpr_0_10",
            lambda y, x: roc_auc_score(y, x, max_fpr=0.10),
        ),
    ):
        value, reason = _binary_metric(labels, scores, function)
        rows.append(
            _metric_row(
                group,
                metric=metric,
                value=value,
                n_observed=n_observed,
                n_positive=n_positive,
                n_negative=n_negative,
                reason_code=reason,
            )
        )

    usable_effect = (
        known
        & predicted.notna()
        & np.isfinite(predicted)
        & truth_effect.notna()
        & np.isfinite(truth_effect)
    )
    predicted_values = predicted.loc[usable_effect].to_numpy(dtype=float)
    truth_values = truth_effect.loc[usable_effect].to_numpy(dtype=float)
    effect_count = len(predicted_values)
    if (
        effect_count >= 3
        and len(np.unique(predicted_values)) > 1
        and len(np.unique(truth_values)) > 1
    ):
        spearman = float(spearmanr(predicted_values, truth_values).statistic)
        spearman_reason = None
    else:
        spearman = None
        spearman_reason = "fewer_than_three_or_constant_effects"
    rows.append(
        _metric_row(
            group,
            metric="effect_spearman",
            value=spearman,
            n_observed=effect_count,
            n_positive=n_positive,
            n_negative=n_negative,
            reason_code=spearman_reason,
        )
    )
    rmse = (
        float(np.sqrt(np.mean(np.square(predicted_values - truth_values))))
        if effect_count
        else None
    )
    rows.append(
        _metric_row(
            group,
            metric="effect_rmse_raw_scale",
            value=rmse,
            n_observed=effect_count,
            n_positive=n_positive,
            n_negative=n_negative,
            reason_code="no_effect_pairs" if effect_count == 0 else None,
        )
    )
    nonzero = usable_effect & truth_effect.ne(0.0)
    direction_count = int(nonzero.sum())
    direction = (
        float(
            np.mean(
                np.sign(predicted.loc[nonzero].to_numpy(dtype=float))
                == np.sign(truth_effect.loc[nonzero].to_numpy(dtype=float))
            )
        )
        if direction_count
        else None
    )
    rows.append(
        _metric_row(
            group,
            metric="direction_accuracy",
            value=direction,
            n_observed=direction_count,
            n_positive=n_positive,
            n_negative=n_negative,
            reason_code="no_nonzero_truth_effects" if direction_count == 0 else None,
        )
    )
    diagnostic_p = pd.to_numeric(table["diagnostic_p_value"], errors="coerce")
    null = usable_effect & truth_effect.eq(0.0) & diagnostic_p.notna()
    null_count = int(null.sum())
    false_positive = (
        float(diagnostic_p.loc[null].lt(0.05).mean()) if null_count else None
    )
    rows.append(
        _metric_row(
            group,
            metric="diagnostic_false_positive_rate_alpha_0_05",
            value=false_positive,
            n_observed=null_count,
            n_positive=n_positive,
            n_negative=n_negative,
            reason_code=(
                "no_null_effects_with_diagnostic_p" if null_count == 0 else None
            ),
        )
    )
    return rows


def _score_geometry_rows(
    score_views: pd.DataFrame,
    *,
    dgp_family: str,
    design_kind: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouping = [
        "dataset_id",
        "generator_id",
        "score_view",
        "estimand",
        "resolution",
        "contrast_scope",
    ]
    for _, table in score_views.groupby(grouping, observed=True, sort=False):
        first = table.iloc[0]
        group = {
            "dataset_id": str(first["dataset_id"]),
            "dgp_family": dgp_family,
            "design_kind": design_kind,
            "generator_id": str(first["generator_id"]),
            "score_view": str(first["score_view"]),
            "estimand": str(first["estimand"]),
            "resolution": str(first["resolution"]),
            "inference_id": "__score__",
            "contrast_name": str(first["contrast_scope"]),
        }
        numeric = pd.to_numeric(table["score"], errors="coerce")
        finite = numeric[numeric.notna() & np.isfinite(numeric)]
        total = len(table)
        unique = int(finite.nunique())
        metrics = {
            "score_na_fraction": float(1.0 - len(finite) / total),
            "score_zero_fraction": (
                float(finite.eq(0.0).mean()) if not finite.empty else None
            ),
            "score_tie_fraction": (
                float(1.0 - unique / len(finite)) if not finite.empty else None
            ),
            "score_unique_value_count": float(unique),
        }
        for metric, value in metrics.items():
            rows.append(
                _metric_row(
                    group,
                    metric=metric,
                    value=value,
                    n_observed=len(finite),
                    n_positive=0,
                    n_negative=0,
                    reason_code="no_finite_scores" if finite.empty else None,
                )
            )
    return rows


def evaluate_v7_integrated_matrix(
    score_views: pd.DataFrame,
    effects: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    dgp_family: str,
    design_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return event-level truth alignment and compact long-form metrics."""

    if tuple(score_views.columns) != SCORE_VIEW_COLUMNS:
        raise ValueError("score_views do not match the integrated v7 contract")
    aligned = align_v7_effect_truth(effects, truth)
    metric_rows: list[dict[str, object]] = _score_geometry_rows(
        score_views,
        dgp_family=dgp_family,
        design_kind=design_kind,
    )
    grouping = [
        "dataset_id",
        "generator_id",
        "score_view",
        "estimand",
        "resolution",
        "inference_id",
        "contrast_name",
    ]
    for _, table in aligned.groupby(grouping, observed=True, sort=False):
        metric_rows.extend(
            _effect_metric_rows(
                table,
                dgp_family=dgp_family,
                design_kind=design_kind,
            )
        )
    metrics = pd.DataFrame.from_records(metric_rows, columns=METRIC_COLUMNS)
    return aligned, metrics.sort_values(
        [
            "generator_id",
            "inference_id",
            "score_view",
            "contrast_name",
            "metric",
        ],
        kind="stable",
        ignore_index=True,
    )


__all__ = [
    "ALIGNED_EFFECT_COLUMNS",
    "METRIC_COLUMNS",
    "SCHEMA_VERSION",
    "align_v7_effect_truth",
    "evaluate_v7_integrated_matrix",
]
