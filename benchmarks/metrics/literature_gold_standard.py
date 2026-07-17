"""Generic metrics for literature-derived labeled gold standards.

The truth table defines the evaluation universe. Missing method rows and
non-eligible statuses are never converted to score zero; they reduce reported
coverage. AUROC and AUPRC use only eligible, finite scores. Threshold-dependent
metrics require a frozen top-k or oriented-score threshold supplied before
evaluation.

Intervals are descriptive benchmark uncertainty summaries. The stratified
edge bootstrap resamples labeled positives and negatives independently. The
negative-sampling analysis keeps every positive and repeatedly samples
negatives without replacement. Neither procedure is biological-replicate
inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.metrics.multicondition import _binary_rank_metrics

EVALUATION_METRICS = (
    "auroc",
    "average_precision",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
)


def _column_names(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if any(not value or value != value.strip() for value in values):
        raise ValueError(f"{field} must contain canonical non-empty names")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique names")
    return values


@dataclass(frozen=True, slots=True, kw_only=True)
class LiteratureGoldStandardSpec:
    """Frozen column, decision, and resampling policy for one evaluation."""

    key_columns: tuple[str, ...]
    method_columns: tuple[str, ...] = ("method",)
    stratum_columns: tuple[str, ...] = ()
    score_column: str = "score"
    label_column: str = "is_positive"
    status_column: str | None = "status"
    eligible_statuses: tuple[str, ...] = ("observed",)
    direction_column: str | None = "score_direction"
    higher_is_better: bool = True
    top_k: int | None = None
    oriented_score_threshold: float | None = None
    n_bootstrap: int = 1000
    n_negative_samples: int = 1000
    negative_to_positive_ratio: float = 1.0
    confidence_level: float = 0.95
    random_seed: int = 0

    def __post_init__(self) -> None:
        key_columns = _column_names(self.key_columns, field="key_columns")
        method_columns = _column_names(self.method_columns, field="method_columns")
        stratum_columns = _column_names(self.stratum_columns, field="stratum_columns")
        if not key_columns or not method_columns:
            raise ValueError("key_columns and method_columns must not be empty")
        identity_columns = (*stratum_columns, *method_columns, *key_columns)
        if len(set(identity_columns)) != len(identity_columns):
            raise ValueError("stratum, method, and key columns must not overlap")
        auxiliary = [self.score_column, self.label_column]
        auxiliary.extend(
            value
            for value in (self.status_column, self.direction_column)
            if value is not None
        )
        if any(not value or value != value.strip() for value in auxiliary):
            raise ValueError("score/label/status/direction columns must be canonical")
        if len(set(auxiliary)) != len(auxiliary):
            raise ValueError("score/label/status/direction columns must be distinct")
        if set(identity_columns).intersection(auxiliary):
            raise ValueError("identity columns cannot also be value columns")
        if self.status_column is not None:
            statuses = _column_names(self.eligible_statuses, field="eligible_statuses")
            if not statuses:
                raise ValueError("eligible_statuses must not be empty")
        if not isinstance(self.higher_is_better, bool):
            raise ValueError("higher_is_better must be boolean")
        has_top_k = self.top_k is not None
        has_threshold = self.oriented_score_threshold is not None
        if has_top_k == has_threshold:
            raise ValueError("set exactly one of top_k or oriented_score_threshold")
        if has_top_k and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 1
        ):
            raise ValueError("top_k must be an integer >= 1")
        threshold = self.oriented_score_threshold
        if has_threshold:
            if threshold is None or isinstance(threshold, bool):
                raise ValueError("oriented_score_threshold must be finite")
            if not math.isfinite(threshold):
                raise ValueError("oriented_score_threshold must be finite")
        for field in ("n_bootstrap", "n_negative_samples"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be an integer >= 1")
        if (
            isinstance(self.negative_to_positive_ratio, bool)
            or not math.isfinite(self.negative_to_positive_ratio)
            or self.negative_to_positive_ratio <= 0
        ):
            raise ValueError("negative_to_positive_ratio must be finite and > 0")
        if (
            isinstance(self.confidence_level, bool)
            or not math.isfinite(self.confidence_level)
            or not 0 < self.confidence_level < 1
        ):
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        object.__setattr__(self, "key_columns", key_columns)
        object.__setattr__(self, "method_columns", method_columns)
        object.__setattr__(self, "stratum_columns", stratum_columns)


@dataclass(frozen=True, slots=True, kw_only=True)
class LiteratureGoldStandardResult:
    """Point estimates and two auditable resampling summaries."""

    point_estimates: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    negative_sampling_summary: pd.DataFrame

    def __post_init__(self) -> None:
        for field in (
            "point_estimates",
            "bootstrap_summary",
            "negative_sampling_summary",
        ):
            table = getattr(self, field)
            if not isinstance(table, pd.DataFrame) or table.empty:
                raise ValueError(f"{field} must be a non-empty DataFrame")
            object.__setattr__(self, field, table.copy(deep=True))


def _validate_truth(
    truth: pd.DataFrame, spec: LiteratureGoldStandardSpec
) -> pd.DataFrame:
    required = {
        *spec.stratum_columns,
        *spec.key_columns,
        spec.label_column,
    }
    missing = required.difference(truth.columns)
    if missing:
        raise ValueError(f"truth table is missing columns: {sorted(missing)}")
    if truth.empty:
        raise ValueError("truth table must not be empty")
    identifiers = [*spec.stratum_columns, *spec.key_columns]
    if truth.loc[:, identifiers].isna().any().any():
        raise ValueError("truth identifiers must not contain missing values")
    if truth.duplicated(identifiers).any():
        raise ValueError("truth table contains duplicate labeled keys")
    labels = pd.to_numeric(truth[spec.label_column], errors="coerce")
    if (
        labels.isna().any()
        or (labels % 1 != 0).any()
        or not set(labels.astype(int)).issubset({0, 1})
    ):
        raise ValueError(f"{spec.label_column} must contain binary 0/1 labels")
    result = truth.loc[:, [*identifiers, spec.label_column]].copy(deep=True)
    result[spec.label_column] = labels.astype(int)
    return result


def _validate_scores(
    scores: pd.DataFrame, spec: LiteratureGoldStandardSpec
) -> pd.DataFrame:
    required = {
        *spec.stratum_columns,
        *spec.method_columns,
        *spec.key_columns,
        spec.score_column,
    }
    if spec.status_column is not None:
        required.add(spec.status_column)
    if spec.direction_column is not None:
        required.add(spec.direction_column)
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"score table is missing columns: {sorted(missing)}")
    if scores.empty:
        raise ValueError("score table must not be empty")
    identifiers = [
        *spec.stratum_columns,
        *spec.method_columns,
        *spec.key_columns,
    ]
    if scores.loc[:, identifiers].isna().any().any():
        raise ValueError("score identifiers must not contain missing values")
    if scores.duplicated(identifiers).any():
        raise ValueError("score table contains duplicate method-key rows")
    numeric = pd.to_numeric(scores[spec.score_column], errors="coerce")
    supplied = scores[spec.score_column].notna()
    if ((supplied & numeric.isna()) | np.isinf(numeric.fillna(0.0))).any():
        raise ValueError("supplied scores must be finite numeric values")
    result = scores.loc[:, sorted(required)].copy(deep=True)
    result[spec.score_column] = numeric
    if spec.status_column is not None:
        if result[spec.status_column].isna().any():
            raise ValueError("score statuses must not contain missing values")
        eligible = result[spec.status_column].astype(str).isin(spec.eligible_statuses)
        if result.loc[eligible, spec.score_column].isna().any():
            raise ValueError("eligible score rows require finite scores")
    if spec.direction_column is not None:
        direction = result[spec.direction_column]
        if direction.isna().any() or not set(direction.astype(str)).issubset(
            {"higher", "lower"}
        ):
            raise ValueError("score direction must contain only higher/lower")
        grouping = [*spec.stratum_columns, *spec.method_columns]
        if (
            result.groupby(grouping, sort=False, observed=True)[
                spec.direction_column
            ].nunique()
            > 1
        ).any():
            raise ValueError("score direction must be constant per method group")
    return result


def _selected_truth(
    truth: pd.DataFrame,
    spec: LiteratureGoldStandardSpec,
    identity: dict[str, Any],
) -> pd.DataFrame:
    selected = truth
    for column in spec.stratum_columns:
        selected = selected.loc[selected[column].eq(identity[column])]
    if selected.empty:
        values = {column: identity[column] for column in spec.stratum_columns}
        raise ValueError(f"score group has no labeled truth stratum: {values}")
    return selected


def _decision_predictions(
    scores: np.ndarray, spec: LiteratureGoldStandardSpec
) -> tuple[np.ndarray, float, str]:
    if spec.top_k is not None:
        realized_k = min(spec.top_k, len(scores))
        cutoff = float(np.partition(scores, len(scores) - realized_k)[-realized_k])
        return scores >= cutoff, cutoff, "top_k_tie_expanded"
    threshold = spec.oriented_score_threshold
    if threshold is None:
        raise RuntimeError("validated score threshold is unexpectedly missing")
    return scores >= threshold, threshold, "oriented_score_threshold"


def _metric_values(
    labels: np.ndarray,
    scores: np.ndarray,
    spec: LiteratureGoldStandardSpec,
) -> dict[str, float | int | str]:
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    if n_positive == 0 or n_negative == 0:
        return {
            **{metric: math.nan for metric in EVALUATION_METRICS},
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
            "n_predicted_positive": 0,
            "decision_cutoff_oriented": math.nan,
            "decision_rule": (
                "top_k_tie_expanded"
                if spec.top_k is not None
                else "oriented_score_threshold"
            ),
        }
    auroc, average_precision = _binary_rank_metrics(labels, scores)
    predicted, cutoff, decision_rule = _decision_predictions(scores, spec)
    positive = labels == 1
    negative = ~positive
    true_positive = int((predicted & positive).sum())
    false_positive = int((predicted & negative).sum())
    true_negative = int(((~predicted) & negative).sum())
    false_negative = int(((~predicted) & positive).sum())
    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator if precision_denominator > 0 else math.nan
    )
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else math.nan
    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (true_positive * true_negative - false_positive * false_negative)
        / mcc_denominator
        if mcc_denominator
        else math.nan
    )
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "n_predicted_positive": int(predicted.sum()),
        "decision_cutoff_oriented": cutoff,
        "decision_rule": decision_rule,
    }


def _derived_seed(
    *,
    base_seed: int,
    identity: dict[str, Any],
    scheme: str,
) -> int:
    payload = {
        "base_seed": base_seed,
        "identity": [
            {"column": key, "type": type(value).__name__, "value": str(value)}
            for key, value in identity.items()
        ],
        "scheme": scheme,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _replicate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    spec: LiteratureGoldStandardSpec,
    *,
    identity: dict[str, Any],
    scheme: str,
    n_repeats: int,
) -> tuple[dict[str, np.ndarray], int, int, int]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    derived_seed = _derived_seed(
        base_seed=spec.random_seed,
        identity=identity,
        scheme=scheme,
    )
    rng = np.random.default_rng(derived_seed)
    output: dict[str, np.ndarray] = {
        metric: np.full(n_repeats, np.nan, dtype=float) for metric in EVALUATION_METRICS
    }
    if scheme == "stratified_edge_bootstrap":
        n_negative_per_repeat = len(negative)
    elif scheme == "repeated_negative_subsampling":
        n_negative_per_repeat = min(
            len(negative),
            max(1, math.ceil(len(positive) * spec.negative_to_positive_ratio)),
        )
    else:
        raise ValueError(f"unknown resampling scheme: {scheme}")
    for repeat in range(n_repeats):
        if scheme == "stratified_edge_bootstrap":
            selected_positive = rng.choice(positive, size=len(positive), replace=True)
            selected_negative = rng.choice(negative, size=len(negative), replace=True)
        else:
            selected_positive = positive
            selected_negative = rng.choice(
                negative, size=n_negative_per_repeat, replace=False
            )
        selected = np.concatenate((selected_positive, selected_negative))
        values = _metric_values(labels[selected], scores[selected], spec)
        for metric in EVALUATION_METRICS:
            output[metric][repeat] = float(values[metric])
    return output, derived_seed, len(positive), n_negative_per_repeat


def _resampling_summary(
    *,
    identity: dict[str, Any],
    point: dict[str, float | int | str],
    replicates: dict[str, np.ndarray] | None,
    spec: LiteratureGoldStandardSpec,
    scheme: str,
    n_repeats: int,
    derived_seed: int,
    n_positive_per_repeat: int,
    n_negative_per_repeat: int,
    reason_code: str | None,
) -> list[dict[str, object]]:
    alpha = 1.0 - spec.confidence_level
    rows: list[dict[str, object]] = []
    for metric in EVALUATION_METRICS:
        values = (
            np.asarray([], dtype=float) if replicates is None else replicates[metric]
        )
        finite = values[np.isfinite(values)]
        if len(finite):
            quantiles = np.asarray(
                np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0]),
                dtype=float,
            )
            lower = float(quantiles[0])
            upper = float(quantiles[1])
            replicate_mean = float(finite.mean())
            replicate_median = float(np.median(finite))
            replicate_standard_deviation = (
                float(finite.std(ddof=1)) if len(finite) >= 2 else math.nan
            )
            status = "observed"
            metric_reason = None
        else:
            lower = math.nan
            upper = math.nan
            replicate_mean = math.nan
            replicate_median = math.nan
            replicate_standard_deviation = math.nan
            status = "not_estimable"
            metric_reason = reason_code or "metric_undefined_in_all_resamples"
        rows.append(
            {
                **identity,
                "resampling_scheme": scheme,
                "metric": metric,
                "full_evaluation_estimate": float(point[metric]),
                "replicate_mean": replicate_mean,
                "replicate_median": replicate_median,
                "replicate_standard_deviation": replicate_standard_deviation,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": spec.confidence_level,
                "n_repeats_requested": n_repeats,
                "n_repeats_valid": len(finite),
                "valid_repeat_fraction": len(finite) / n_repeats,
                "n_positive_per_repeat": n_positive_per_repeat,
                "n_negative_per_repeat": n_negative_per_repeat,
                "negative_to_positive_ratio_requested": (
                    spec.negative_to_positive_ratio
                    if scheme == "repeated_negative_subsampling"
                    else math.nan
                ),
                "top_k_requested": spec.top_k,
                "oriented_score_threshold_requested": (spec.oriented_score_threshold),
                "base_random_seed": spec.random_seed,
                "derived_random_seed": derived_seed,
                "replacement_within_class": (scheme == "stratified_edge_bootstrap"),
                "resampling_unit": "labeled_edge",
                "interval_semantics": (
                    "descriptive_benchmark_uncertainty_not_biological_inference"
                ),
                "status": status,
                "reason_code": metric_reason,
            }
        )
    return rows


def evaluate_literature_gold_standard(
    scores: pd.DataFrame,
    truth: pd.DataFrame,
    spec: LiteratureGoldStandardSpec,
) -> LiteratureGoldStandardResult:
    """Evaluate method scores against an explicitly labeled truth universe.

    Score rows outside the labeled truth universe are ignored but counted as
    ``n_unlabeled_score_rows``. Truth rows without an eligible finite score are
    retained in coverage denominators and excluded from metric numerators.
    ``average_precision`` is the tie-aware step-integral definition of AUPRC;
    ``auprc`` is emitted as an exact alias in the point-estimate table.
    """

    truth_table = _validate_truth(truth, spec)
    score_table = _validate_scores(scores, spec)
    grouping = [*spec.stratum_columns, *spec.method_columns]
    point_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    for raw_keys, method_scores in score_table.groupby(
        grouping, sort=True, observed=True
    ):
        keys = raw_keys if isinstance(raw_keys, tuple) else (raw_keys,)
        identity = dict(zip(grouping, keys, strict=True))
        selected_truth = _selected_truth(truth_table, spec, identity)
        selected_truth = selected_truth.sort_values(
            list(spec.key_columns), kind="stable", ignore_index=True
        )
        value_columns = [*spec.key_columns, spec.score_column]
        if spec.status_column is not None:
            value_columns.append(spec.status_column)
        merged = selected_truth.merge(
            method_scores.loc[:, value_columns],
            on=list(spec.key_columns),
            how="left",
            validate="one_to_one",
            indicator="_score_merge",
        )
        matched = merged["_score_merge"].eq("both")
        if spec.status_column is None:
            eligible = matched
        else:
            eligible = matched & merged[spec.status_column].astype(str).isin(
                spec.eligible_statuses
            )
        comparable = eligible & merged[spec.score_column].notna()
        direction = (
            str(method_scores[spec.direction_column].iloc[0])
            if spec.direction_column is not None
            else ("higher" if spec.higher_is_better else "lower")
        )
        direction_multiplier = 1.0 if direction == "higher" else -1.0
        labels = merged.loc[comparable, spec.label_column].to_numpy(dtype=int)
        oriented_scores = (
            merged.loc[comparable, spec.score_column].to_numpy(dtype=float)
            * direction_multiplier
        )
        point = _metric_values(labels, oriented_scores, spec)
        n_truth = len(merged)
        n_truth_positive = int(merged[spec.label_column].sum())
        n_truth_negative = n_truth - n_truth_positive
        n_evaluable_positive = int((labels == 1).sum())
        n_evaluable_negative = int((labels == 0).sum())
        estimable = n_evaluable_positive > 0 and n_evaluable_negative > 0
        if estimable:
            reason_code = None
        elif not len(labels):
            reason_code = "no_eligible_scores_in_truth_universe"
        else:
            reason_code = "truth_has_single_class_after_coverage"
        score_key_index = pd.MultiIndex.from_frame(
            method_scores.loc[:, list(spec.key_columns)]
        )
        truth_key_index = pd.MultiIndex.from_frame(
            selected_truth.loc[:, list(spec.key_columns)]
        )
        n_unlabeled_score_rows = int((~score_key_index.isin(truth_key_index)).sum())
        point_rows.append(
            {
                **identity,
                **point,
                "auprc": float(point["average_precision"]),
                "auprc_definition": "average_precision_step_integral_tie_aware",
                "score_direction": direction,
                "truth_universe_edges": n_truth,
                "matched_score_edges": int(matched.sum()),
                "evaluable_edges": len(labels),
                "truth_coverage_fraction": len(labels) / n_truth,
                "score_match_fraction": float(matched.mean()),
                "truth_positive_edges": n_truth_positive,
                "truth_negative_edges": n_truth_negative,
                "evaluable_positive_edges": n_evaluable_positive,
                "evaluable_negative_edges": n_evaluable_negative,
                "positive_coverage_fraction": (
                    n_evaluable_positive / n_truth_positive
                    if n_truth_positive
                    else math.nan
                ),
                "negative_coverage_fraction": (
                    n_evaluable_negative / n_truth_negative
                    if n_truth_negative
                    else math.nan
                ),
                "n_unlabeled_score_rows": n_unlabeled_score_rows,
                "top_k_requested": spec.top_k,
                "oriented_score_threshold_requested": (spec.oriented_score_threshold),
                "missing_score_policy": "exclude_and_report_coverage_never_impute_zero",
                "status": "observed" if estimable else "not_estimable",
                "reason_code": reason_code,
            }
        )
        if estimable:
            bootstrap, bootstrap_seed, n_positive, n_negative = _replicate_metrics(
                labels,
                oriented_scores,
                spec,
                identity=identity,
                scheme="stratified_edge_bootstrap",
                n_repeats=spec.n_bootstrap,
            )
            negative, negative_seed, _, sampled_negative = _replicate_metrics(
                labels,
                oriented_scores,
                spec,
                identity=identity,
                scheme="repeated_negative_subsampling",
                n_repeats=spec.n_negative_samples,
            )
        else:
            bootstrap = None
            negative = None
            bootstrap_seed = _derived_seed(
                base_seed=spec.random_seed,
                identity=identity,
                scheme="stratified_edge_bootstrap",
            )
            negative_seed = _derived_seed(
                base_seed=spec.random_seed,
                identity=identity,
                scheme="repeated_negative_subsampling",
            )
            n_positive = n_evaluable_positive
            n_negative = n_evaluable_negative
            sampled_negative = min(
                n_evaluable_negative,
                max(
                    1,
                    math.ceil(n_evaluable_positive * spec.negative_to_positive_ratio),
                ),
            )
        bootstrap_rows.extend(
            _resampling_summary(
                identity=identity,
                point=point,
                replicates=bootstrap,
                spec=spec,
                scheme="stratified_edge_bootstrap",
                n_repeats=spec.n_bootstrap,
                derived_seed=bootstrap_seed,
                n_positive_per_repeat=n_positive,
                n_negative_per_repeat=n_negative,
                reason_code=reason_code,
            )
        )
        negative_rows.extend(
            _resampling_summary(
                identity=identity,
                point=point,
                replicates=negative,
                spec=spec,
                scheme="repeated_negative_subsampling",
                n_repeats=spec.n_negative_samples,
                derived_seed=negative_seed,
                n_positive_per_repeat=n_evaluable_positive,
                n_negative_per_repeat=sampled_negative,
                reason_code=reason_code,
            )
        )
    return LiteratureGoldStandardResult(
        point_estimates=pd.DataFrame.from_records(point_rows),
        bootstrap_summary=pd.DataFrame.from_records(bootstrap_rows),
        negative_sampling_summary=pd.DataFrame.from_records(negative_rows),
    )


__all__ = [
    "EVALUATION_METRICS",
    "LiteratureGoldStandardResult",
    "LiteratureGoldStandardSpec",
    "evaluate_literature_gold_standard",
]
