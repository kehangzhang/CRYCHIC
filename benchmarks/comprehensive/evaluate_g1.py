"""Evaluate standardized G1 event predictions without imputing missing results.

The input is one row per frozen event and method/scenario arm. Only rows whose
``status`` is exactly ``ok`` are eligible for metrics. Non-finite predictions
and every non-``ok`` status remain missing and are reflected in coverage; they
are never converted to zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from benchmarks.adapters.common import git_metadata, sha256_file

SCHEMA_VERSION = "crychic-comprehensive-g1-metrics-v1"
METRIC_REGISTRY_PATH = Path(__file__).with_name("metrics.tsv")
GROUP_COLUMNS = ("dataset_id", "method_id", "resource_mode", "scenario_id")
REQUIRED_COLUMNS = (
    *GROUP_COLUMNS,
    "event_id",
    "status",
    "truth_label",
    "truth_effect",
    "predicted_score",
    "predicted_effect",
)
NUMERIC_COLUMNS = (
    "truth_label",
    "truth_effect",
    "predicted_score",
    "predicted_effect",
)
METRIC_IDS = (
    "prevalence_adjusted_ap",
    "differential_auroc",
    "differential_mcc",
    "effect_rmse",
    "effect_spearman",
    "sign_accuracy",
)


def _as_one_dimensional(
    labels: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    numeric_labels = np.asarray(labels, dtype=float)
    y_score = np.asarray(scores, dtype=float)
    if (
        numeric_labels.ndim != 1
        or y_score.ndim != 1
        or numeric_labels.size != y_score.size
    ):
        raise ValueError(
            "labels and scores must be equal-length one-dimensional arrays"
        )
    if numeric_labels.size == 0:
        raise ValueError("labels and scores must not be empty")
    if not np.isfinite(numeric_labels).all():
        raise ValueError("labels must be finite")
    if not np.isfinite(y_score).all():
        raise ValueError("scores must be finite")
    if not set(np.unique(numeric_labels)).issubset({0.0, 1.0}):
        raise ValueError("labels must be binary")
    return numeric_labels.astype(np.int8), y_score


def _binary_class_counts(labels: np.ndarray) -> tuple[int, int]:
    n_positive = int(labels.sum())
    return n_positive, int(labels.size - n_positive)


def _threshold_groups(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        sorted_scores.size - 1,
    ]
    cumulative_positive = np.cumsum(sorted_labels, dtype=np.int64)[group_ends]
    cumulative_negative = group_ends + 1 - cumulative_positive
    return cumulative_positive.astype(float), cumulative_negative.astype(float)


def average_precision(
    labels: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
) -> float:
    """Return tie-aware step-integral average precision."""
    y_true, y_score = _as_one_dimensional(labels, scores)
    n_positive, n_negative = _binary_class_counts(y_true)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("average precision requires positive and negative labels")
    true_positive, false_positive = _threshold_groups(y_true, y_score)
    recall = true_positive / n_positive
    precision = true_positive / (true_positive + false_positive)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def prevalence_adjusted_average_precision(
    labels: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    *,
    target_prevalence: float,
) -> float:
    """Return AP after mathematically reweighting both truth classes.

    Every positive receives weight ``target_prevalence / n_positive`` and every
    negative receives weight ``(1 - target_prevalence) / n_negative``. Recall
    increments are consequently unchanged, while precision is evaluated at the
    preregistered target prevalence. Tied scores enter the curve as one group.
    """
    if not 0.0 < target_prevalence < 1.0:
        raise ValueError("target_prevalence must lie strictly between zero and one")
    y_true, y_score = _as_one_dimensional(labels, scores)
    n_positive, n_negative = _binary_class_counts(y_true)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("adjusted average precision requires both truth classes")
    true_positive, false_positive = _threshold_groups(y_true, y_score)
    recall = true_positive / n_positive
    weighted_positive = true_positive * target_prevalence / n_positive
    weighted_negative = false_positive * (1.0 - target_prevalence) / n_negative
    precision = weighted_positive / (weighted_positive + weighted_negative)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def tie_aware_auroc(
    labels: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
) -> float:
    """Return Mann-Whitney AUROC with average ranks for tied scores."""
    y_true, y_score = _as_one_dimensional(labels, scores)
    n_positive, n_negative = _binary_class_counts(y_true)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC requires positive and negative labels")
    ranks = rankdata(y_score, method="average")
    rank_sum = float(ranks[y_true == 1].sum())
    statistic = rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(statistic / (n_positive * n_negative))


def matthews_correlation(
    labels: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    *,
    threshold: float,
) -> float:
    """Return MCC after applying one fixed raw-score threshold."""
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    y_true, y_score = _as_one_dimensional(labels, scores)
    n_positive, n_negative = _binary_class_counts(y_true)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("MCC requires positive and negative labels")
    predicted = y_score >= threshold
    positive = y_true == 1
    true_positive = int((predicted & positive).sum())
    false_positive = int((predicted & ~positive).sum())
    true_negative = int((~predicted & ~positive).sum())
    false_negative = int((~predicted & positive).sum())
    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    if denominator == 0.0:
        # This is the conventional finite value used when predictions have one
        # class but the truth still contains both classes.
        return 0.0
    numerator = true_positive * true_negative - false_positive * false_negative
    return float(numerator / denominator)


def _read_event_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"event table does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.name.lower().endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    raise ValueError("event table must be TSV/TSV.GZ or Parquet")


def _coerce_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    source = frame[column]
    missing = source.isna() | source.astype(str).str.strip().eq("")
    values = pd.to_numeric(source.where(~missing), errors="coerce")
    invalid = ~missing & values.isna()
    if invalid.any():
        examples = source.loc[invalid].astype(str).drop_duplicates().head(3).tolist()
        raise ValueError(f"column {column!r} contains non-numeric values: {examples}")
    return values.astype(float)


def validate_event_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the standardized event table."""
    missing = set(REQUIRED_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"event table is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("event table is empty")
    result = table.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in (*GROUP_COLUMNS, "event_id", "status"):
        if result[column].isna().any():
            raise ValueError(f"identifier column {column!r} contains null values")
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"identifier column {column!r} contains empty values")
    duplicate_key = [*GROUP_COLUMNS, "event_id"]
    if result.duplicated(duplicate_key, keep=False).any():
        raise ValueError("event table contains duplicate group/event keys")
    for column in NUMERIC_COLUMNS:
        result[column] = _coerce_numeric(result, column)
    finite_labels = result.loc[np.isfinite(result["truth_label"]), "truth_label"]
    invalid_labels = ~finite_labels.isin([0.0, 1.0])
    if invalid_labels.any():
        values = sorted(finite_labels.loc[invalid_labels].unique().tolist())
        raise ValueError(f"truth_label must be binary where present: {values}")
    return result.sort_values(
        [*GROUP_COLUMNS, "event_id"], kind="stable", ignore_index=True
    )


def _metric_row(
    identity: dict[str, str],
    *,
    metric_id: str,
    value: float,
    status: str,
    reason_code: str | None,
    n_events_total: int,
    n_status_ok: int,
    n_truth_eligible: int,
    n_used: int,
    n_positive: int,
    n_negative: int,
    native_prevalence: float,
    native_ap_diagnostic: float,
    coverage_diagnostic: float,
    target_prevalence: float,
    decision_threshold: float,
    effect_coverage_diagnostic: float,
) -> dict[str, Any]:
    return {
        **identity,
        "metric_id": metric_id,
        "value": value,
        "status": status,
        "reason_code": reason_code,
        "n_events_total": n_events_total,
        "n_status_ok": n_status_ok,
        "n_truth_eligible": n_truth_eligible,
        "n_used": n_used,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "native_prevalence": native_prevalence,
        "native_ap_diagnostic": native_ap_diagnostic,
        "coverage_diagnostic": coverage_diagnostic,
        "target_prevalence": target_prevalence,
        "decision_threshold": decision_threshold,
        "effect_coverage_diagnostic": effect_coverage_diagnostic,
    }


def _effect_spearman(
    truth: np.ndarray, predicted: np.ndarray
) -> tuple[float, str, str | None]:
    if truth.size == 0:
        return math.nan, "NE", "no_valid_effect_pairs"
    if truth.size < 2:
        return math.nan, "NE", "insufficient_effect_pairs"
    if np.unique(truth).size < 2:
        return math.nan, "NE", "constant_truth_effect"
    if np.unique(predicted).size < 2:
        return math.nan, "NE", "constant_predicted_effect"
    value = float(spearmanr(truth, predicted).statistic)
    if not math.isfinite(value):
        return math.nan, "NE", "undefined_spearman"
    return value, "ok", None


def evaluate_g1_events(
    table: pd.DataFrame,
    *,
    target_prevalence: float,
    threshold: float,
) -> pd.DataFrame:
    """Compute one long-form metric table per frozen G1 evaluation group."""
    if not 0.0 < target_prevalence < 1.0:
        raise ValueError("target_prevalence must lie strictly between zero and one")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    events = validate_event_table(table)
    rows: list[dict[str, Any]] = []
    grouped = events.groupby(list(GROUP_COLUMNS), sort=True, observed=True)
    for raw_identity, group in grouped:
        keys = raw_identity if isinstance(raw_identity, tuple) else (raw_identity,)
        identity = dict(zip(GROUP_COLUMNS, map(str, keys), strict=True))
        n_events_total = len(group)
        ok = group["status"].eq("ok").to_numpy()
        labels_all = group["truth_label"].to_numpy(dtype=float)
        scores_all = group["predicted_score"].to_numpy(dtype=float)
        rankable = ok & np.isfinite(labels_all) & np.isfinite(scores_all)
        labels = labels_all[rankable].astype(np.int8)
        scores = scores_all[rankable]
        n_status_ok = int(ok.sum())
        n_rankable = int(rankable.sum())
        n_binary_truth = int(np.isfinite(labels_all).sum())
        n_positive, n_negative = _binary_class_counts(labels)
        native_prevalence = float(n_positive / n_rankable) if n_rankable else math.nan

        truth_effect_all = group["truth_effect"].to_numpy(dtype=float)
        predicted_effect_all = group["predicted_effect"].to_numpy(dtype=float)
        finite_truth_effect = np.isfinite(truth_effect_all)
        effect_pairs = ok & finite_truth_effect & np.isfinite(predicted_effect_all)
        truth_effect = truth_effect_all[effect_pairs]
        predicted_effect = predicted_effect_all[effect_pairs]
        n_effect_truth = int(finite_truth_effect.sum())
        n_effect_used = int(effect_pairs.sum())
        effect_coverage_diagnostic = (
            float(n_effect_used / n_effect_truth) if n_effect_truth else math.nan
        )

        binary_values: dict[str, float] = {}
        binary_reason: str | None = None
        if n_rankable == 0:
            binary_reason = "no_rankable_events"
        elif n_positive == 0 or n_negative == 0:
            binary_reason = "single_class_truth"
        else:
            binary_values = {
                "native_ap_diagnostic": average_precision(labels, scores),
                "prevalence_adjusted_ap": prevalence_adjusted_average_precision(
                    labels, scores, target_prevalence=target_prevalence
                ),
                "differential_auroc": tie_aware_auroc(labels, scores),
                "differential_mcc": matthews_correlation(
                    labels, scores, threshold=threshold
                ),
            }
        shared = {
            "n_events_total": n_events_total,
            "n_status_ok": n_status_ok,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "native_prevalence": native_prevalence,
            "native_ap_diagnostic": binary_values.get("native_ap_diagnostic", math.nan),
            "coverage_diagnostic": float(n_rankable / n_events_total),
            "target_prevalence": target_prevalence,
            "decision_threshold": threshold,
            "effect_coverage_diagnostic": effect_coverage_diagnostic,
        }
        for metric_id in METRIC_IDS[:3]:
            rows.append(
                _metric_row(
                    identity,
                    metric_id=metric_id,
                    value=binary_values.get(metric_id, math.nan),
                    status="ok" if binary_reason is None else "NE",
                    reason_code=binary_reason,
                    n_truth_eligible=n_binary_truth,
                    n_used=n_rankable,
                    **shared,
                )
            )

        if n_effect_used:
            rmse = float(np.sqrt(np.mean(np.square(predicted_effect - truth_effect))))
            rmse_status, rmse_reason = "ok", None
        else:
            rmse, rmse_status, rmse_reason = (
                math.nan,
                "NE",
                "no_valid_effect_pairs",
            )
        rows.append(
            _metric_row(
                identity,
                metric_id="effect_rmse",
                value=rmse,
                status=rmse_status,
                reason_code=rmse_reason,
                n_truth_eligible=n_effect_truth,
                n_used=n_effect_used,
                **shared,
            )
        )

        spearman, spearman_status, spearman_reason = _effect_spearman(
            truth_effect, predicted_effect
        )
        rows.append(
            _metric_row(
                identity,
                metric_id="effect_spearman",
                value=spearman,
                status=spearman_status,
                reason_code=spearman_reason,
                n_truth_eligible=n_effect_truth,
                n_used=n_effect_used,
                **shared,
            )
        )

        nonzero_truth = truth_effect != 0.0
        n_nonzero_truth = int(nonzero_truth.sum())
        if n_nonzero_truth:
            sign_accuracy = float(
                np.mean(
                    np.sign(predicted_effect[nonzero_truth])
                    == np.sign(truth_effect[nonzero_truth])
                )
            )
            sign_status, sign_reason = "ok", None
        else:
            sign_accuracy, sign_status, sign_reason = (
                math.nan,
                "NE",
                "no_nonzero_truth_effects",
            )
        rows.append(
            _metric_row(
                identity,
                metric_id="sign_accuracy",
                value=sign_accuracy,
                status=sign_status,
                reason_code=sign_reason,
                n_truth_eligible=int(
                    (finite_truth_effect & (truth_effect_all != 0)).sum()
                ),
                n_used=n_nonzero_truth,
                **shared,
            )
        )
    columns = [
        *GROUP_COLUMNS,
        "metric_id",
        "value",
        "status",
        "reason_code",
        "n_events_total",
        "n_status_ok",
        "n_truth_eligible",
        "n_used",
        "n_positive",
        "n_negative",
        "native_prevalence",
        "native_ap_diagnostic",
        "coverage_diagnostic",
        "target_prevalence",
        "decision_threshold",
        "effect_coverage_diagnostic",
    ]
    return pd.DataFrame.from_records(rows, columns=columns)


def _publish_directory(staged: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise NotADirectoryError(f"output path is not a directory: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output directory exists: {output}; pass --overwrite to replace it"
        )
    if not output.exists():
        os.replace(staged, output)
        return
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def _metric_registry_record(repo_root: Path) -> dict[str, Any]:
    registry = pd.read_csv(METRIC_REGISTRY_PATH, sep="\t", dtype=str)
    if "metric_id" not in registry:
        raise ValueError("metric registry is missing the metric_id column")
    if registry["metric_id"].duplicated().any():
        raise ValueError("metric registry contains duplicate metric IDs")
    registered = set(registry["metric_id"])
    unknown = set(METRIC_IDS).difference(registered)
    if unknown:
        raise ValueError(
            f"G1 evaluator contains unregistered metrics: {sorted(unknown)}"
        )
    return {
        "path": str(METRIC_REGISTRY_PATH.relative_to(repo_root)),
        "sha256": sha256_file(METRIC_REGISTRY_PATH),
        "metric_ids": list(METRIC_IDS),
    }


def run_evaluation(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    target_prevalence: float,
    threshold: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate an event table and atomically publish metrics plus provenance."""
    source = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise NotADirectoryError(f"output path is not a directory: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output directory exists: {output}; pass --overwrite to replace it"
        )
    source_sha256 = sha256_file(source)
    events = validate_event_table(_read_event_table(source))
    metrics = evaluate_g1_events(
        events,
        target_prevalence=target_prevalence,
        threshold=threshold,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    published = False
    try:
        metrics_path = staged / "metrics.tsv"
        metrics.to_csv(metrics_path, sep="\t", index=False, lineterminator="\n")
        repo_root = Path(__file__).resolve().parents[2]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "complete",
            "input": {
                "filename": source.name,
                "bytes": source.stat().st_size,
                "rows": len(events),
                "sha256": source_sha256,
                "status_counts": dict(
                    sorted(Counter(events["status"].astype(str)).items())
                ),
            },
            "parameters": {
                "group_columns": list(GROUP_COLUMNS),
                "ok_status": "ok",
                "score_direction": "higher_is_more_positive",
                "target_prevalence": target_prevalence,
                "decision_threshold": threshold,
            },
            "metric_registry": _metric_registry_record(repo_root),
            "metrics": {
                "ids": list(METRIC_IDS),
                "diagnostic_fields": [
                    "native_ap_diagnostic",
                    "coverage_diagnostic",
                    "effect_coverage_diagnostic",
                ],
                "groups": int(
                    metrics.loc[:, list(GROUP_COLUMNS)].drop_duplicates().shape[0]
                ),
                "rows": len(metrics),
                "status_counts": dict(sorted(Counter(metrics["status"]).items())),
            },
            "output": {
                "filename": metrics_path.name,
                "bytes": metrics_path.stat().st_size,
                "rows": len(metrics),
                "sha256": sha256_file(metrics_path),
            },
            "interpretation": {
                "non_ok_imputed_as_zero": False,
                "nonfinite_predictions_imputed_as_zero": False,
                "coverage_definition": (
                    "status_ok_and_finite_truth_label_and_predicted_score / "
                    "all_frozen_events"
                ),
                "adjusted_ap_weighting": (
                    "positive_weight=target_prevalence/n_positive;"
                    "negative_weight=(1-target_prevalence)/n_negative"
                ),
                "calibration_metrics_computed": False,
            },
            "code": {
                **git_metadata(repo_root),
                "entrypoint": str(Path(__file__).relative_to(repo_root)),
                "entrypoint_sha256": sha256_file(__file__),
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _publish_directory(staged, output, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-prevalence", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_evaluation(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_prevalence=args.target_prevalence,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "metrics": manifest["metrics"],
                "output_dir": args.output_dir.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
