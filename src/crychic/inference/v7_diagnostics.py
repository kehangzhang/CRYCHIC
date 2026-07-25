"""Pure descriptive metrics for v7 out-of-fold diagnostic ledgers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

V7_DIAGNOSTIC_METRICS_VERSION = "v7_oof_descriptive_metrics_v1"

V7_DIAGNOSTIC_SCORE_HEADS = (
    "sender_detection_raw",
    "parent_peak_raw",
    "parent_total_raw",
    "parent_mean_raw",
    "program_signed",
    "coupling_prior",
    "active_probability",
    "sender_attribution",
    "null_sender_attribution",
    "attribution_entropy",
    "mechanism_support",
)

_PARENT_SCORE_HEADS = {
    "parent_peak_raw",
    "parent_total_raw",
    "parent_mean_raw",
    "program_signed",
    "active_probability",
    "null_sender_attribution",
    "attribution_entropy",
}
_PARENT_KEY = (
    "fold_id",
    "sample_id",
    "context_id",
    "receiver",
    "interaction_id",
)
_RESOLUTION_LEVELS = {
    "lr_only",
    "lr_receiver_parent",
    "sender_lr_receiver_child",
    "pathway",
    "exact_hyperedge",
}

SCORE_GEOMETRY_COLUMNS = (
    "dataset_id",
    "contrast_name",
    "fold_id",
    "score_head",
    "score_grain",
    "n_rows",
    "n_finite",
    "n_missing",
    "zero_fraction",
    "na_fraction",
    "unique_value_count",
    "tie_fraction",
    "iqr",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "truth_positive_q05",
    "truth_positive_q25",
    "truth_positive_q50",
    "truth_positive_q75",
    "truth_positive_q95",
    "truth_negative_q05",
    "truth_negative_q25",
    "truth_negative_q50",
    "truth_negative_q75",
    "truth_negative_q95",
    "within_condition_variance",
    "between_condition_variance",
    "null_effect_sd",
    "status",
    "reason_code",
)

CANDIDATE_SENDER_BIAS_COLUMNS = (
    "dataset_id",
    "contrast_name",
    "fold_id",
    "candidate_sender_bin",
    "n_parents",
    "n_sender_rows",
    "n_truth_positive_rows",
    "n_truth_negative_rows",
    "sender_detection_mean",
    "truth_positive_sender_detection_mean",
    "truth_negative_sender_detection_mean",
    "mean_max_attribution",
    "mean_attribution_entropy",
    "false_positive_rate",
    "sender_auprc",
    "sender_auroc",
    "true_sender_top1_accuracy",
    "parent_score_mean",
    "sender_detection_threshold",
    "status",
    "reason_code",
)

RESOLUTION_PERFORMANCE_COLUMNS = (
    "dataset_id",
    "contrast_name",
    "score_head",
    "resolution_level",
    "n_units",
    "n_scored_units",
    "n_truth_positive",
    "n_truth_negative",
    "auprc",
    "auroc",
    "effect_spearman",
    "effect_rmse",
    "direction_accuracy",
    "status",
    "reason_code",
)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _require_columns(
    table: pd.DataFrame,
    columns: set[str],
    *,
    table_name: str,
) -> None:
    missing = columns.difference(table.columns)
    if missing:
        raise ValueError(f"{table_name} is missing columns: {sorted(missing)}")


def _validate_boolean_series(values: pd.Series, *, field_name: str) -> None:
    if values.isna().any() or any(
        not isinstance(value, bool | np.bool_) for value in values.tolist()
    ):
        raise ValueError(f"{field_name} must contain non-missing booleans")


def _quantiles(values: pd.Series) -> tuple[float, float, float, float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(finite):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    quantiles = np.asarray(
        np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95]),
        dtype=float,
    ).reshape(-1)
    return (
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
        float(quantiles[3]),
        float(quantiles[4]),
    )


def _score_grain(head: str) -> str:
    return "parent" if head in _PARENT_SCORE_HEADS else "sender_child"


def _head_table(source: pd.DataFrame, head: str) -> pd.DataFrame:
    if head not in _PARENT_SCORE_HEADS:
        return source.copy()
    parent_key = list(_PARENT_KEY)
    truth = (
        source.groupby(parent_key, observed=True)["truth_class"]
        .agg(
            any_positive=lambda values: values.eq("truth_positive").any(),
            any_unknown=lambda values: values.eq("unknown").any(),
        )
        .reset_index()
    )
    truth["truth_class"] = np.select(
        [truth["any_positive"], truth["any_unknown"]],
        ["truth_positive", "unknown"],
        default="truth_negative",
    )
    parent = source.drop_duplicates(parent_key).drop(columns="truth_class")
    return parent.merge(
        truth.loc[:, [*parent_key, "truth_class"]],
        on=parent_key,
        how="left",
        validate="one_to_one",
        sort=False,
    )


def _variance_geometry(
    table: pd.DataFrame,
    *,
    head: str,
) -> tuple[float, float, float]:
    identity = (
        ["sender", "receiver", "interaction_id"]
        if _score_grain(head) == "sender_child"
        else ["receiver", "interaction_id"]
    )
    finite = table.loc[table[head].notna(), [*identity, "condition", head]].copy()
    if finite.empty:
        return np.nan, np.nan, np.nan
    finite[head] = pd.to_numeric(finite[head], errors="coerce")
    finite = finite.dropna(subset=[head])
    condition_key = [*identity, "condition"]
    group_mean = finite.groupby(condition_key, observed=True)[head].transform("mean")
    within_df = len(finite) - finite.groupby(condition_key, observed=True).ngroups
    within = (
        float(np.square(finite[head].to_numpy() - group_mean.to_numpy()).sum())
        / within_df
        if within_df > 0
        else np.nan
    )
    condition_means = (
        finite.groupby(condition_key, observed=True)[head]
        .agg(["mean", "size"])
        .reset_index()
    )
    event_mean = condition_means.groupby(identity, observed=True)["mean"].transform(
        "mean"
    )
    between_df = (
        len(condition_means) - condition_means.groupby(identity, observed=True).ngroups
    )
    between = (
        float(
            (
                condition_means["size"].to_numpy(dtype=float)
                * np.square(condition_means["mean"].to_numpy() - event_mean.to_numpy())
            ).sum()
        )
        / between_df
        if between_df > 0
        else np.nan
    )
    null_table = table.loc[table["truth_class"].eq("truth_negative")]
    null_table = null_table.loc[
        null_table[head].notna(), [*identity, "condition", head]
    ]
    amplitudes: list[float] = []
    for _, event in null_table.groupby(identity, observed=True, sort=False):
        means = (
            event.groupby("condition", observed=True)[head]
            .mean()
            .sort_index()
            .to_numpy(dtype=float)
        )
        if len(means) == 2:
            amplitudes.append(float(means[1] - means[0]))
        elif len(means) > 2:
            amplitudes.append(float(np.max(means) - np.min(means)))
    null_sd = float(np.std(amplitudes, ddof=1)) if len(amplitudes) > 1 else np.nan
    return within, between, null_sd


def summarize_v7_score_geometry(
    scores: pd.DataFrame,
    *,
    dataset_id: str,
    score_heads: tuple[str, ...] = V7_DIAGNOSTIC_SCORE_HEADS,
) -> pd.DataFrame:
    """Summarize OOF score geometry without changing any score value."""

    dataset = _name(dataset_id, field_name="dataset_id")
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        raise ValueError("scores must be a non-empty pandas DataFrame")
    heads = tuple(score_heads)
    if not heads or len(heads) != len(set(heads)):
        raise ValueError("score_heads must be non-empty and unique")
    unknown = set(heads).difference(V7_DIAGNOSTIC_SCORE_HEADS)
    if unknown:
        raise ValueError(f"unsupported score_heads: {sorted(unknown)}")
    _require_columns(
        scores,
        {
            "contrast_name",
            "fold_id",
            "sample_id",
            "context_id",
            "condition",
            "sender",
            "receiver",
            "interaction_id",
            "truth_class",
            *heads,
        },
        table_name="scores",
    )
    records: list[dict[str, object]] = []
    fold_scopes = [*sorted(scores["fold_id"].astype(str).unique()), "__all__"]
    for contrast in sorted(scores["contrast_name"].astype(str).unique()):
        contrast_rows = scores.loc[scores["contrast_name"].astype(str).eq(contrast)]
        for fold_id in fold_scopes:
            scope = (
                contrast_rows
                if fold_id == "__all__"
                else contrast_rows.loc[contrast_rows["fold_id"].astype(str).eq(fold_id)]
            )
            for head in heads:
                table = _head_table(scope, head)
                numeric = pd.to_numeric(table[head], errors="coerce")
                finite = numeric.dropna()
                n_rows = len(table)
                n_finite = len(finite)
                unique = int(finite.nunique())
                tie_fraction = (
                    (n_finite - unique) / (n_finite - 1)
                    if n_finite > 1
                    else 0.0
                    if n_finite == 1
                    else np.nan
                )
                all_q = _quantiles(numeric)
                positive_q = _quantiles(
                    numeric.loc[table["truth_class"].eq("truth_positive")]
                )
                negative_q = _quantiles(
                    numeric.loc[table["truth_class"].eq("truth_negative")]
                )
                within, between, null_sd = _variance_geometry(table, head=head)
                records.append(
                    {
                        "dataset_id": dataset,
                        "contrast_name": contrast,
                        "fold_id": fold_id,
                        "score_head": head,
                        "score_grain": _score_grain(head),
                        "n_rows": n_rows,
                        "n_finite": n_finite,
                        "n_missing": n_rows - n_finite,
                        "zero_fraction": (
                            float(finite.eq(0.0).mean()) if n_finite else np.nan
                        ),
                        "na_fraction": (
                            (n_rows - n_finite) / n_rows if n_rows else np.nan
                        ),
                        "unique_value_count": unique,
                        "tie_fraction": tie_fraction,
                        "iqr": (
                            float(finite.quantile(0.75) - finite.quantile(0.25))
                            if n_finite
                            else np.nan
                        ),
                        "q05": all_q[0],
                        "q25": all_q[1],
                        "q50": all_q[2],
                        "q75": all_q[3],
                        "q95": all_q[4],
                        "truth_positive_q05": positive_q[0],
                        "truth_positive_q25": positive_q[1],
                        "truth_positive_q50": positive_q[2],
                        "truth_positive_q75": positive_q[3],
                        "truth_positive_q95": positive_q[4],
                        "truth_negative_q05": negative_q[0],
                        "truth_negative_q25": negative_q[1],
                        "truth_negative_q50": negative_q[2],
                        "truth_negative_q75": negative_q[3],
                        "truth_negative_q95": negative_q[4],
                        "within_condition_variance": within,
                        "between_condition_variance": between,
                        "null_effect_sd": null_sd,
                        "status": "observed" if n_finite else "not_estimable",
                        "reason_code": None if n_finite else "score_head_has_no_values",
                    }
                )
    return pd.DataFrame.from_records(
        records, columns=SCORE_GEOMETRY_COLUMNS
    ).sort_values(
        ["dataset_id", "contrast_name", "fold_id", "score_head"],
        kind="stable",
        ignore_index=True,
    )


def _binary_metrics(
    truth: np.ndarray,
    score: np.ndarray,
) -> tuple[float, float]:
    valid = np.isfinite(score)
    labels = truth[valid].astype(bool)
    values = score[valid].astype(float)
    n_positive = int(np.count_nonzero(labels))
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        return np.nan, np.nan
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    sorted_labels = labels[order]
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_values) != 0), len(values) - 1]
    true_positive = np.cumsum(sorted_labels)[boundaries]
    false_positive = np.cumsum(~sorted_labels)[boundaries]
    recall = true_positive / n_positive
    precision = true_positive / (true_positive + false_positive)
    previous_recall = np.r_[0.0, recall[:-1]]
    auprc = float(np.sum((recall - previous_recall) * precision))

    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    rank_sum = float(ranks[labels].sum())
    auroc = (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return auprc, float(auroc)


def summarize_v7_candidate_sender_bias(
    scores: pd.DataFrame,
    *,
    dataset_id: str,
    detection_threshold: float,
) -> pd.DataFrame:
    """Summarize sender metrics by fixed candidate-cardinality bins."""

    dataset = _name(dataset_id, field_name="dataset_id")
    threshold = float(detection_threshold)
    if not math.isfinite(threshold):
        raise ValueError("detection_threshold must be finite")
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        raise ValueError("scores must be a non-empty pandas DataFrame")
    _require_columns(
        scores,
        {
            "contrast_name",
            "fold_id",
            "sample_id",
            "context_id",
            "receiver",
            "interaction_id",
            "candidate_sender_bin",
            "sender_detection_raw",
            "truth_class",
            "sender_attribution",
            "attribution_entropy",
            "parent_mean_raw",
        },
        table_name="scores",
    )
    records: list[dict[str, object]] = []
    fold_scopes = [*sorted(scores["fold_id"].astype(str).unique()), "__all__"]
    for contrast in sorted(scores["contrast_name"].astype(str).unique()):
        contrast_rows = scores.loc[scores["contrast_name"].astype(str).eq(contrast)]
        for fold_id in fold_scopes:
            fold_rows = (
                contrast_rows
                if fold_id == "__all__"
                else contrast_rows.loc[contrast_rows["fold_id"].astype(str).eq(fold_id)]
            )
            for sender_bin in ("1", "2", "3-5", "6-10", ">10", "unknown"):
                table = fold_rows.loc[
                    fold_rows["candidate_sender_bin"].eq(sender_bin)
                ].copy()
                if table.empty:
                    continue
                parent_key = list(_PARENT_KEY)
                parent = table.drop_duplicates(parent_key)
                detection = pd.to_numeric(
                    table["sender_detection_raw"], errors="coerce"
                )
                truth_known = table["truth_class"].isin(
                    ["truth_positive", "truth_negative"]
                )
                labels = (
                    table.loc[truth_known, "truth_class"]
                    .eq("truth_positive")
                    .to_numpy(dtype=bool)
                )
                truth_scores = detection.loc[truth_known].to_numpy(dtype=float)
                auprc, auroc = _binary_metrics(labels, truth_scores)
                negative = table["truth_class"].eq("truth_negative") & detection.notna()
                fpr = (
                    float(detection.loc[negative].gt(threshold).mean())
                    if negative.any()
                    else np.nan
                )
                max_attribution = table.groupby(parent_key, observed=True)[
                    "sender_attribution"
                ].max()
                top1: list[float] = []
                for _, group in table.groupby(parent_key, observed=True, sort=False):
                    if group["truth_class"].eq("unknown").any():
                        continue
                    group_scores = pd.to_numeric(
                        group["sender_detection_raw"], errors="coerce"
                    )
                    if (
                        group_scores.notna().sum() == 0
                        or not group["truth_class"].eq("truth_positive").any()
                    ):
                        continue
                    maximum = float(group_scores.max())
                    winners = group.loc[group_scores.eq(maximum), "truth_class"]
                    top1.append(float(winners.eq("truth_positive").any()))
                positive = table["truth_class"].eq("truth_positive")
                records.append(
                    {
                        "dataset_id": dataset,
                        "contrast_name": contrast,
                        "fold_id": fold_id,
                        "candidate_sender_bin": sender_bin,
                        "n_parents": len(parent),
                        "n_sender_rows": len(table),
                        "n_truth_positive_rows": int(positive.sum()),
                        "n_truth_negative_rows": int(
                            table["truth_class"].eq("truth_negative").sum()
                        ),
                        "sender_detection_mean": (
                            float(detection.mean())
                            if detection.notna().any()
                            else np.nan
                        ),
                        "truth_positive_sender_detection_mean": (
                            float(detection.loc[positive].mean())
                            if detection.loc[positive].notna().any()
                            else np.nan
                        ),
                        "truth_negative_sender_detection_mean": (
                            float(detection.loc[negative].mean())
                            if negative.any()
                            else np.nan
                        ),
                        "mean_max_attribution": (
                            float(max_attribution.mean())
                            if max_attribution.notna().any()
                            else np.nan
                        ),
                        "mean_attribution_entropy": (
                            float(
                                pd.to_numeric(
                                    parent["attribution_entropy"], errors="coerce"
                                ).mean()
                            )
                            if parent["attribution_entropy"].notna().any()
                            else np.nan
                        ),
                        "false_positive_rate": fpr,
                        "sender_auprc": auprc,
                        "sender_auroc": auroc,
                        "true_sender_top1_accuracy": (
                            float(np.mean(top1)) if top1 else np.nan
                        ),
                        "parent_score_mean": (
                            float(
                                pd.to_numeric(
                                    parent["parent_mean_raw"], errors="coerce"
                                ).mean()
                            )
                            if parent["parent_mean_raw"].notna().any()
                            else np.nan
                        ),
                        "sender_detection_threshold": threshold,
                        "status": (
                            "observed" if detection.notna().any() else "not_estimable"
                        ),
                        "reason_code": (
                            None
                            if detection.notna().any()
                            else "sender_detection_has_no_values"
                        ),
                    }
                )
    return pd.DataFrame.from_records(
        records, columns=CANDIDATE_SENDER_BIAS_COLUMNS
    ).sort_values(
        ["dataset_id", "contrast_name", "fold_id", "candidate_sender_bin"],
        kind="stable",
        ignore_index=True,
    )


def summarize_v7_resolution_performance(
    evaluation: pd.DataFrame | None,
    *,
    dataset_id: str,
) -> pd.DataFrame:
    """Summarize an explicit, benchmark-owned multi-resolution score ledger."""

    dataset = _name(dataset_id, field_name="dataset_id")
    if evaluation is None:
        return pd.DataFrame(columns=RESOLUTION_PERFORMANCE_COLUMNS)
    if not isinstance(evaluation, pd.DataFrame):
        raise TypeError("resolution_evaluation must be a pandas DataFrame or None")
    required = {
        "contrast_name",
        "score_head",
        "resolution_level",
        "unit_id",
        "score",
        "truth",
    }
    _require_columns(evaluation, required, table_name="resolution_evaluation")
    source = evaluation.copy(deep=True)
    for column in (
        "contrast_name",
        "score_head",
        "resolution_level",
        "unit_id",
    ):
        if source[column].isna().any():
            raise ValueError(f"resolution_evaluation.{column} cannot be missing")
        source[column] = source[column].astype(str)
    if not set(source["resolution_level"]).issubset(_RESOLUTION_LEVELS):
        raise ValueError("resolution_evaluation contains an unsupported level")
    _validate_boolean_series(source["truth"], field_name="resolution_evaluation.truth")
    numeric_score = pd.to_numeric(source["score"], errors="coerce")
    if (source["score"].notna() & numeric_score.isna()).any():
        raise ValueError("resolution_evaluation.score must be numeric or missing")
    source["score"] = numeric_score
    if np.isinf(source["score"].dropna()).any():
        raise ValueError("resolution_evaluation.score must be finite or missing")
    if "truth_effect" not in source:
        source["truth_effect"] = np.nan
    else:
        numeric_effect = pd.to_numeric(source["truth_effect"], errors="coerce")
        if (source["truth_effect"].notna() & numeric_effect.isna()).any():
            raise ValueError("resolution truth_effect must be numeric or missing")
        source["truth_effect"] = numeric_effect
        if np.isinf(source["truth_effect"].dropna()).any():
            raise ValueError("resolution truth_effect must be finite or missing")
    key = ["contrast_name", "score_head", "resolution_level", "unit_id"]
    if source.duplicated(key).any():
        raise ValueError("resolution_evaluation unit keys must be unique")

    records: list[dict[str, object]] = []
    group_key = ["contrast_name", "score_head", "resolution_level"]
    for values, group in source.groupby(group_key, observed=True, sort=True):
        contrast, score_head, level = (str(value) for value in values)
        scored = group["score"].notna()
        labels = group.loc[scored, "truth"].to_numpy(dtype=bool)
        score = group.loc[scored, "score"].to_numpy(dtype=float)
        auprc, auroc = _binary_metrics(labels, score)
        effect_rows = scored & group["truth_effect"].notna()
        if int(effect_rows.sum()) >= 2:
            predicted = group.loc[effect_rows, "score"].to_numpy(dtype=float)
            expected = group.loc[effect_rows, "truth_effect"].to_numpy(dtype=float)
            spearman = float(
                pd.Series(predicted)
                .rank(method="average")
                .corr(pd.Series(expected).rank(method="average"))
            )
            rmse = float(np.sqrt(np.mean(np.square(predicted - expected))))
            nonzero = ~np.isclose(expected, 0.0)
            direction = (
                float(
                    np.mean(np.sign(predicted[nonzero]) == np.sign(expected[nonzero]))
                )
                if np.any(nonzero)
                else np.nan
            )
        else:
            spearman = np.nan
            rmse = np.nan
            direction = np.nan
        n_scored = int(scored.sum())
        records.append(
            {
                "dataset_id": dataset,
                "contrast_name": contrast,
                "score_head": score_head,
                "resolution_level": level,
                "n_units": len(group),
                "n_scored_units": n_scored,
                "n_truth_positive": int(group["truth"].sum()),
                "n_truth_negative": int((~group["truth"]).sum()),
                "auprc": auprc,
                "auroc": auroc,
                "effect_spearman": spearman,
                "effect_rmse": rmse,
                "direction_accuracy": direction,
                "status": "observed" if n_scored else "not_estimable",
                "reason_code": None if n_scored else "resolution_scores_missing",
            }
        )
    return pd.DataFrame.from_records(
        records, columns=RESOLUTION_PERFORMANCE_COLUMNS
    ).sort_values(group_key, kind="stable", ignore_index=True)


__all__ = [
    "CANDIDATE_SENDER_BIAS_COLUMNS",
    "RESOLUTION_PERFORMANCE_COLUMNS",
    "SCORE_GEOMETRY_COLUMNS",
    "V7_DIAGNOSTIC_METRICS_VERSION",
    "V7_DIAGNOSTIC_SCORE_HEADS",
    "summarize_v7_candidate_sender_bias",
    "summarize_v7_resolution_performance",
    "summarize_v7_score_geometry",
]
