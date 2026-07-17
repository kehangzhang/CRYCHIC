"""Shared contracts for the Open Problems source-target benchmark."""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

TASK_ID = "cell_cell_communication_source_target"
TASK_VERSION = "v1.0.0"
OFFICIAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/openproblems-bio/website/main/results/"
    "cell_cell_communication_source_target/data/results.json"
)
OFFICIAL_V1_RANDOM_AUPRC = 0.04334026605274756
OFFICIAL_V1_RANDOM_ODDS_RATIO = 0.2753623188405796
MERGE_KEYS = ("source", "target")
PREDICTION_COLUMNS = (
    "task_id",
    "task_version",
    "method_id",
    "method_name",
    "method_scope",
    "resource_id",
    "aggregation",
    "source",
    "target",
    "score",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_truth(input_h5ad: str | Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    data = ad.read_h5ad(input_h5ad, backed="r")
    try:
        if "label" not in data.obs or "ccc_target" not in data.uns:
            raise ValueError("Open Problems input lacks label or ccc_target")
        truth = data.uns["ccc_target"].copy()
        label_values = data.obs["label"]
        if isinstance(label_values.dtype, pd.CategoricalDtype):
            labels = tuple(map(str, label_values.cat.categories))
        else:
            labels = tuple(map(str, pd.unique(label_values.astype(str))))
    finally:
        data.file.close()
    required = {"source", "target", "response"}
    if required.difference(truth.columns):
        raise ValueError("source-target truth has an invalid schema")
    truth = truth.loc[:, ["source", "target", "response"]].copy()
    truth["source"] = truth["source"].astype(str)
    truth["target"] = truth["target"].astype(str)
    truth["response"] = pd.to_numeric(truth["response"], errors="raise").astype(int)
    if (
        truth.duplicated(list(MERGE_KEYS)).any()
        or not set(truth["response"]).issubset({0, 1})
        or set(truth["source"]).difference(labels)
        or set(truth["target"]).difference(labels)
    ):
        raise ValueError("source-target truth violates the Open Problems contract")
    # The released odds-ratio metric resolves tied scores in the original truth
    # row order. Preserve that order so zero-heavy methods match the platform.
    return truth.reset_index(drop=True), labels


def aggregate_lr_scores(
    table: pd.DataFrame,
    *,
    value_column: str,
    method_id: str,
    method_name: str,
    method_scope: str,
    resource_id: str,
    aggregation: str,
    labels: tuple[str, ...],
    higher_is_better: bool = True,
) -> pd.DataFrame:
    required = {"source", "target", value_column}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"LR score table is missing columns: {sorted(missing)}")
    if aggregation not in {"max", "sum"}:
        raise ValueError("aggregation must be max or sum")
    source = table.loc[:, ["source", "target", value_column]].copy()
    source["source"] = source["source"].astype(str)
    source["target"] = source["target"].astype(str)
    source[value_column] = pd.to_numeric(source[value_column], errors="coerce")
    source = source.loc[
        source["source"].isin(labels)
        & source["target"].isin(labels)
        & source[value_column].notna()
    ].copy()
    if not higher_is_better:
        source[value_column] = -source[value_column]
    if source.empty:
        raise ValueError(f"{method_id} emitted no finite in-universe LR scores")
    grouped = (
        source.groupby(list(MERGE_KEYS), sort=True, observed=True)[value_column]
        .agg(aggregation)
        .rename("score")
        .reset_index()
    )
    return materialize_complete_pairs(
        grouped,
        method_id=method_id,
        method_name=method_name,
        method_scope=method_scope,
        resource_id=resource_id,
        aggregation=aggregation,
        labels=labels,
    )


def materialize_complete_pairs(
    scores: pd.DataFrame,
    *,
    method_id: str,
    method_name: str,
    method_scope: str,
    resource_id: str,
    aggregation: str,
    labels: tuple[str, ...],
) -> pd.DataFrame:
    if set(MERGE_KEYS).difference(scores.columns) or "score" not in scores:
        raise ValueError("source-target scores require source, target, and score")
    if scores.duplicated(list(MERGE_KEYS)).any():
        raise ValueError("source-target scores contain duplicate pairs")
    numeric = pd.to_numeric(scores["score"], errors="coerce")
    if numeric.notna().sum() == 0 or np.isinf(numeric.dropna()).any():
        raise ValueError("source-target scores require at least one finite value")
    finite_min = float(numeric.dropna().min())
    fill_value = float(np.nextafter(finite_min, -np.inf))
    universe = pd.DataFrame(product(labels, labels), columns=list(MERGE_KEYS))
    result = universe.merge(
        scores.loc[:, [*MERGE_KEYS, "score"]],
        on=list(MERGE_KEYS),
        how="left",
        validate="one_to_one",
    )
    result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(
        fill_value
    )
    result.insert(0, "aggregation", aggregation)
    result.insert(0, "resource_id", resource_id)
    result.insert(0, "method_scope", method_scope)
    result.insert(0, "method_name", method_name)
    result.insert(0, "method_id", method_id)
    result.insert(0, "task_version", TASK_VERSION)
    result.insert(0, "task_id", TASK_ID)
    result = result.loc[:, list(PREDICTION_COLUMNS)].sort_values(
        list(MERGE_KEYS), kind="stable", ignore_index=True
    )
    validate_predictions(result, labels=labels)
    return result


def validate_predictions(
    predictions: pd.DataFrame,
    *,
    labels: tuple[str, ...],
) -> None:
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("prediction columns differ from the released contract")
    expected = len(labels) ** 2
    if (
        len(predictions) != expected
        or predictions.duplicated(list(MERGE_KEYS)).any()
        or set(predictions["source"]) != set(labels)
        or set(predictions["target"]) != set(labels)
    ):
        raise ValueError("predictions do not cover the complete source-target grid")
    scores = pd.to_numeric(predictions["score"], errors="coerce")
    if scores.isna().any() or np.isinf(scores).any():
        raise ValueError("prediction scores must be finite")


def write_predictions(output_dir: str | Path, predictions: pd.DataFrame) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    method_ids = predictions["method_id"].astype(str).unique()
    if len(method_ids) != 1:
        raise ValueError("one prediction file must contain one method_id")
    path = output / f"{method_ids[0]}.parquet"
    predictions.to_parquet(path, index=False)
    return path


def raw_official_metrics(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    joined = truth.merge(
        predictions.loc[:, [*MERGE_KEYS, "score"]],
        on=list(MERGE_KEYS),
        how="left",
        validate="one_to_one",
    )
    if joined["score"].notna().sum() == 0:
        raise ValueError("no predictions intersect the benchmark truth")
    minimum = float(joined["score"].dropna().min())
    joined["score"] = joined["score"].fillna(
        minimum - np.finfo(float).eps
    )
    response = joined["response"].to_numpy(dtype=int)
    score = joined["score"].to_numpy(dtype=float)
    precision, recall, _ = precision_recall_curve(response, score, pos_label=1)
    auprc = float(auc(recall, precision))

    ranked = joined.sort_values("score", ascending=False).reset_index(drop=True)
    top_n = int(len(ranked) * 0.05)
    top = ranked.iloc[:top_n]
    remainder = ranked.iloc[top_n:]
    tp = int(top["response"].eq(1).sum())
    fp = int(top["response"].eq(0).sum())
    fn = int(remainder["response"].eq(1).sum())
    tn = int(remainder["response"].eq(0).sum())
    numerator = tp * tn
    denominator = fp * fn
    if denominator == 0:
        odds_ratio = float("nan") if numerator == 0 else float("inf")
    else:
        odds_ratio = numerator / denominator
    transformed_or = (
        float("nan")
        if np.isnan(odds_ratio)
        else 1.0
        if np.isinf(odds_ratio)
        else float(1.0 - 1.0 / (1.0 + odds_ratio / 2.0))
    )
    return {
        "odds_ratio_raw": transformed_or,
        "precision_recall_auc_raw": auprc,
        "auroc_supplementary": float(roc_auc_score(response, score)),
        "average_precision_supplementary": float(
            average_precision_score(response, score)
        ),
        "truth_rows": len(joined),
        "truth_positive": int(response.sum()),
        "top_n": top_n,
        "top_true_positive": tp,
        "top_false_positive": fp,
        "remainder_false_negative": fn,
        "remainder_true_negative": tn,
    }


def random_events_predictions(
    *,
    labels: tuple[str, ...],
    resource_rows: int,
    resource_id: str,
    n_events: int = 1000,
    seed: int = 1,
) -> pd.DataFrame:
    """Reproduce the released source-target ``random_events`` baseline.

    The upstream method draws ligands before source and target labels. Ligands
    are irrelevant to this task's merge keys, but their RNG consumption is not.
    Advancing the generator here is therefore required for platform-equivalent
    baseline scaling.
    """

    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be a non-empty unique ordered axis")
    if isinstance(resource_rows, bool) or resource_rows < 1:
        raise ValueError("resource_rows must be an integer >= 1")
    if isinstance(n_events, bool) or n_events < 1:
        raise ValueError("n_events must be an integer >= 1")
    rng = np.random.default_rng(seed=seed)
    rng.choice(resource_rows, n_events)
    sparse_scores = pd.DataFrame(
        {
            "source": rng.choice(labels, n_events),
            "target": rng.choice(labels, n_events),
            "score": rng.uniform(0.0, 1.0, n_events),
        }
    ).drop_duplicates(list(MERGE_KEYS), keep="first")
    return materialize_complete_pairs(
        sparse_scores,
        method_id="random_events",
        method_name="Random Events",
        method_scope="openproblems_v1_reproduced_baseline",
        resource_id=resource_id,
        aggregation="first",
        labels=labels,
    )


def official_metrics(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    random_predictions: pd.DataFrame | None = None,
) -> dict[str, float | int]:
    """Return raw task metrics and website-equivalent baseline-scaled scores.

    By default this uses the frozen Random Events metric values published for
    the v1.0.0 Mouse brain atlas leaderboard. A caller may instead provide a
    locally reproduced random prediction table for a resource-sensitivity run.
    """

    raw = raw_official_metrics(truth, predictions)
    baseline: dict[str, float | int]
    if random_predictions is None:
        baseline = {
            "precision_recall_auc_raw": OFFICIAL_V1_RANDOM_AUPRC,
            "odds_ratio_raw": OFFICIAL_V1_RANDOM_ODDS_RATIO,
        }
    else:
        baseline = raw_official_metrics(truth, random_predictions)

    def scale(metric: str) -> float:
        value = float(raw[metric])
        reference = float(baseline[metric])
        if not np.isfinite(value) or not np.isfinite(reference) or reference >= 1.0:
            return float("nan")
        return float((value - reference) / (1.0 - reference))

    scaled_auprc = scale("precision_recall_auc_raw")
    scaled_odds = scale("odds_ratio_raw")
    result = dict(raw)
    result.update(
        {
            "openproblems_score": float(
                np.nanmean([scaled_auprc, scaled_odds])
            ),
            "odds_ratio": scaled_odds,
            "precision_recall_auc": scaled_auprc,
            "random_odds_ratio_raw": baseline["odds_ratio_raw"],
            "random_precision_recall_auc_raw": baseline[
                "precision_recall_auc_raw"
            ],
        }
    )
    return result


def precision_recall_points(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    joined = truth.merge(
        predictions.loc[:, [*MERGE_KEYS, "score"]],
        on=list(MERGE_KEYS),
        how="left",
        validate="one_to_one",
    )
    minimum = float(joined["score"].dropna().min())
    scores = joined["score"].fillna(minimum - np.finfo(float).eps)
    precision, recall, thresholds = precision_recall_curve(
        joined["response"], scores, pos_label=1
    )
    return pd.DataFrame(
        {
            "recall": recall,
            "precision": precision,
            "threshold": np.append(thresholds, np.nan),
        }
    )


__all__ = [
    "MERGE_KEYS",
    "OFFICIAL_RESULTS_URL",
    "OFFICIAL_V1_RANDOM_AUPRC",
    "OFFICIAL_V1_RANDOM_ODDS_RATIO",
    "PREDICTION_COLUMNS",
    "TASK_ID",
    "TASK_VERSION",
    "aggregate_lr_scores",
    "load_truth",
    "materialize_complete_pairs",
    "official_metrics",
    "precision_recall_points",
    "random_events_predictions",
    "raw_official_metrics",
    "sha256_file",
    "validate_predictions",
    "write_json",
    "write_predictions",
]
