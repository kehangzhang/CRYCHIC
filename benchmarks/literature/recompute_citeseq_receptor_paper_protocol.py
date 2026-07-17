"""Recompute the real CITE-seq receptor benchmark with the paper protocol.

This audit intentionally keeps the statistically comparable H-common arm
separate from the literature-style method-independent returned universes.
The latter mirrors the released benchmark design but cannot support a direct
method ranking because each method may contribute a different label universe.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.adapters.common import sha256_file
from benchmarks.literature.evaluate_citeseq import (
    component_scores_from_cellchat,
    component_scores_from_liana,
    evaluate_citeseq_components,
    scores_from_crychic_availability,
)

PAPER_SEED = 1234
PAPER_REPEATS = 100


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    short_name: str
    dataset_id: str


DATASETS = (
    DatasetSpec("5k_pbmc", "5k_pbmc_protein_v3"),
    DatasetSpec("5k_pbmc_nextgem", "5k_pbmc_protein_v3_nextgem"),
    DatasetSpec("pbmc_10k", "pbmc_10k_protein_v3"),
    DatasetSpec("malt_10k", "malt_10k_protein_v3"),
)

METHOD_IMPLEMENTATIONS = {
    "CRYCHIC availability-state": (
        "CRYCHIC static sample-level availability diagnostic"
    ),
    "LIANA magnitude consensus": "LIANA 1.7.3 rank_aggregate component",
    "LIANA specificity consensus": "LIANA 1.7.3 rank_aggregate component",
    "CellPhoneDB composite": ("CellPhoneDB component implemented through LIANA 1.7.3"),
    "CellPhoneDB p-value": ("CellPhoneDB component implemented through LIANA 1.7.3"),
    "Connectome specificity": "Connectome component through LIANA 1.7.3",
    "logFC specificity": "logFC component through LIANA 1.7.3",
    "NATMI specificity": "NATMI component through LIANA 1.7.3",
    "SingleCellSignalR LRscore": ("SingleCellSignalR component through LIANA 1.7.3"),
    "CellChat composite": "CellChat component implemented through LIANA 1.7.3",
    "CellChat p-value": "CellChat component implemented through LIANA 1.7.3",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _record_equality(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    artifact: str,
    field: str,
    expected: Any,
    observed: Any,
    path: Path,
) -> None:
    passed = observed == expected
    records.append(
        {
            "dataset": dataset,
            "artifact": artifact,
            "field": field,
            "expected": str(expected),
            "observed": str(observed),
            "passed": passed,
            "path": str(path),
        }
    )
    if not passed:
        raise ValueError(
            f"{artifact} {field} mismatch for {dataset}: "
            f"expected {expected}, observed {observed}"
        )


def _record_sha256(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    artifact: str,
    path: Path,
    expected: str,
) -> str:
    observed = sha256_file(path)
    _record_equality(
        records,
        dataset=dataset,
        artifact=artifact,
        field="sha256",
        expected=expected,
        observed=observed,
        path=path,
    )
    return observed


def _pr_curve_yardstick_008(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the old yardstick-style threshold PR points, including (0, 1)."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.size != scores.size:
        raise ValueError(
            "labels and scores must be equal-length one-dimensional arrays"
        )
    if labels.size == 0 or not np.isfinite(scores).all():
        raise ValueError("labels/scores must be non-empty and scores finite")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    n_positive = int(labels.sum())
    if n_positive == 0:
        return np.array([0.0]), np.array([1.0])

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        sorted_scores.size - 1,
    ]
    cumulative_positive = np.cumsum(sorted_labels)[group_ends]
    predicted_positive = group_ends + 1
    recall = np.r_[0.0, cumulative_positive / n_positive]
    precision = np.r_[1.0, cumulative_positive / predicted_positive]
    return recall.astype(float), precision.astype(float)


def yardstick_008_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware trapezoidal PR AUC used by yardstick 0.0.8."""

    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0:
        return math.nan
    recall, precision = _pr_curve_yardstick_008(labels, scores)
    return float(np.trapezoid(precision, recall))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-integral average precision, retained only as a diagnostic."""

    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0:
        return math.nan
    recall, precision = _pr_curve_yardstick_008(labels, scores)
    return float(np.sum(np.diff(recall) * precision[1:]))


def tie_aware_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = pd.Series(scores).rank(method="average", ascending=True).to_numpy()
    rank_sum = float(ranks[positive].sum())
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def paper_balanced_auprc(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    dataset: str,
    method: str,
    universe_mode: str,
    repeats: int = PAPER_REPEATS,
    seed: int = PAPER_SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Apply the released paper's 1:1, with-replacement negative sampling."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive_index = np.flatnonzero(labels == 1)
    negative_index = np.flatnonzero(labels == 0)
    if positive_index.size == 0 or negative_index.size == 0:
        summary = {
            "paper_balanced_auprc_mean": math.nan,
            "paper_balanced_auprc_sd": math.nan,
            "paper_balanced_auprc_q025": math.nan,
            "paper_balanced_auprc_q975": math.nan,
            "released_row_weighted_auprc_diagnostic": math.nan,
            "paper_repeats_valid": 0,
            "paper_positive_per_repeat": int(positive_index.size),
            "paper_negative_per_repeat": 0,
        }
        return summary, pd.DataFrame()

    rng = np.random.RandomState(seed)
    records: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        sampled_negative = rng.choice(
            negative_index, size=positive_index.size, replace=True
        )
        selected = np.r_[positive_index, sampled_negative]
        selected_labels = labels[selected]
        selected_scores = scores[selected]
        recall, _ = _pr_curve_yardstick_008(selected_labels, selected_scores)
        records.append(
            {
                "dataset": dataset,
                "method": method,
                "universe_mode": universe_mode,
                "repeat": repeat,
                "auprc": yardstick_008_pr_auc(selected_labels, selected_scores),
                "average_precision_diagnostic": average_precision(
                    selected_labels, selected_scores
                ),
                "pr_curve_rows": int(recall.size),
                "n_positive": int(positive_index.size),
                "n_negative_draws": int(sampled_negative.size),
                "n_unique_negative_draws": int(np.unique(sampled_negative).size),
                "replacement": True,
                "seed": seed,
            }
        )
    replicates = pd.DataFrame.from_records(records)
    values = replicates["auprc"].to_numpy(dtype=float)
    weights = replicates["pr_curve_rows"].to_numpy(dtype=float)
    summary = {
        "paper_balanced_auprc_mean": float(values.mean()),
        "paper_balanced_auprc_sd": float(values.std(ddof=1)),
        "paper_balanced_auprc_q025": float(np.quantile(values, 0.025)),
        "paper_balanced_auprc_q975": float(np.quantile(values, 0.975)),
        "released_row_weighted_auprc_diagnostic": float(
            np.average(values, weights=weights)
        ),
        "paper_repeats_valid": int(values.size),
        "paper_positive_per_repeat": int(positive_index.size),
        "paper_negative_per_repeat": int(positive_index.size),
    }
    return summary, replicates


def _orient_scores(method: pd.DataFrame) -> np.ndarray:
    directions = method["score_direction"].astype(str).unique()
    if directions.size != 1 or directions[0] not in {"higher", "lower"}:
        raise ValueError("one valid score direction is required per method")
    scores = pd.to_numeric(method["score"], errors="raise").to_numpy(dtype=float)
    return -scores if directions[0] == "lower" else scores


def _truth_key_coverage(returned: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    truth_keys = truth.loc[:, ["receiver", "receptor", "is_positive"]].copy()
    truth_keys["receptor"] = truth_keys["receptor"].astype(str).str.upper()
    truth_keys = truth_keys.drop_duplicates()
    returned_keys = returned.loc[:, ["receiver", "receptor"]].copy()
    returned_keys["receptor"] = returned_keys["receptor"].astype(str).str.upper()
    returned_keys = returned_keys.drop_duplicates()
    covered = truth_keys.merge(returned_keys, on=["receiver", "receptor"], how="inner")
    n_truth = len(truth_keys)
    n_positive = int(truth_keys["is_positive"].sum())
    n_covered = len(covered)
    n_positive_covered = int(covered["is_positive"].sum())
    return {
        "truth_receiver_receptor_keys": n_truth,
        "truth_positive_receiver_receptor_keys": n_positive,
        "returned_truth_receiver_receptor_keys": n_covered,
        "returned_positive_receiver_receptor_keys": n_positive_covered,
        "truth_key_coverage_fraction": n_covered / n_truth,
        "positive_truth_key_coverage_fraction": (
            n_positive_covered / n_positive if n_positive else math.nan
        ),
    }


def _summarize_retained(
    retained: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    comparison_arm: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_records: list[dict[str, Any]] = []
    replicate_frames: list[pd.DataFrame] = []
    grouping = ["dataset", "resource_mode", "universe_mode", "method"]
    for identity, method in retained.groupby(grouping, sort=True, observed=True):
        dataset, resource_mode, universe_mode, method_name = map(str, identity)
        scores = _orient_scores(method)
        labels = method["is_positive"].astype(int).to_numpy()
        returned_mask = (
            method["raw_returned"].astype(bool).to_numpy()
            if "raw_returned" in method.columns
            else np.ones(len(method), dtype=bool)
        )
        returned = method.loc[returned_mask]
        returned_scores = scores[returned_mask]
        n_unique_scores = int(np.unique(scores).size)
        n_unique_returned_scores = int(np.unique(returned_scores).size)
        paper_summary, replicates = paper_balanced_auprc(
            labels,
            scores,
            dataset=dataset,
            method=method_name,
            universe_mode=comparison_arm,
        )
        if not replicates.empty:
            replicate_frames.append(replicates)
        truth_dataset = truth.loc[truth["dataset"].astype(str).eq(dataset)]
        metric_records.append(
            {
                "dataset": dataset,
                "resource_mode": resource_mode,
                "universe_mode": universe_mode,
                "comparison_arm": comparison_arm,
                "method": method_name,
                "implementation": METHOD_IMPLEMENTATIONS.get(
                    method_name, "unclassified"
                ),
                "n_evaluated_edges": len(method),
                "n_evaluated_lr_pairs": int(
                    method.loc[:, ["ligand", "receptor"]].drop_duplicates().shape[0]
                ),
                "n_sender_types": int(method["sender"].nunique()),
                "n_receiver_types": int(method["receiver"].nunique()),
                "n_positive_edges": int(labels.sum()),
                "n_negative_edges": int((labels == 0).sum()),
                "raw_returned_edges": int(returned_mask.sum()),
                "raw_absent_edges_filled_worst": int((~returned_mask).sum()),
                "raw_returned_positive_edges": int(
                    method.loc[returned_mask, "is_positive"].astype(int).sum()
                ),
                "n_unique_oriented_scores": n_unique_scores,
                "n_unique_raw_returned_scores": n_unique_returned_scores,
                "all_evaluated_scores_tied": n_unique_scores == 1,
                "all_raw_returned_scores_tied": n_unique_returned_scores == 1,
                "auroc": tie_aware_auroc(labels, scores),
                "full_pr_auc_trapezoid": yardstick_008_pr_auc(labels, scores),
                "full_average_precision_diagnostic": average_precision(labels, scores),
                **paper_summary,
                **_truth_key_coverage(returned, truth_dataset),
                "negative_sampling_replacement": True,
                "negative_sampling_seed": PAPER_SEED,
                "negative_sampling_rng": (
                    "NumPy RandomState MT19937; paper-design aligned, "
                    "not byte-identical to R sample()"
                ),
                "direct_method_comparison_valid": (
                    comparison_arm == "H-common/resource_fixed"
                ),
            }
        )
    metrics = pd.DataFrame.from_records(metric_records)
    replicates = (
        pd.concat(replicate_frames, ignore_index=True)
        if replicate_frames
        else pd.DataFrame()
    )
    return metrics, replicates


def _dataset_paths(
    workspace: Path, crychic_hcommon_root: Path, spec: DatasetSpec
) -> dict[str, Path]:
    reproduction = workspace / "benchmark_work" / "literature_reproduction"
    result_root = reproduction / "results" / "citeseq" / spec.short_name
    prepared = reproduction / "citeseq" / "prepared" / spec.dataset_id
    return {
        "h5ad": prepared / f"{spec.dataset_id}.h5ad",
        "truth": result_root / "truth" / "receptor_protein_truth.tsv",
        "truth_manifest": result_root / "truth" / "manifest.json",
        "hcommon_liana": (result_root / "H-common" / "liana_cellchat_literature"),
        "hcommon_crychic": crychic_hcommon_root / spec.short_name,
        "native_liana": result_root / "native" / "liana_cellchat_literature",
        "native_crychic": (result_root / "native-cellphonedb" / "crychic_availability"),
    }


def _audit_liana_bundle(
    *,
    records: list[dict[str, Any]],
    dataset: str,
    directory: Path,
    h5ad_sha256: str,
    expected_resource_sha256: str | None,
    artifact_prefix: str,
) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    _record_equality(
        records,
        dataset=dataset,
        artifact=artifact_prefix,
        field="status",
        expected="complete",
        observed=manifest.get("status"),
        path=manifest_path,
    )
    _record_equality(
        records,
        dataset=dataset,
        artifact=artifact_prefix,
        field="input_sha256",
        expected=h5ad_sha256,
        observed=manifest["input"]["sha256"],
        path=manifest_path,
    )
    if expected_resource_sha256 is not None:
        _record_equality(
            records,
            dataset=dataset,
            artifact=artifact_prefix,
            field="resource_payload_sha256",
            expected=expected_resource_sha256,
            observed=manifest["resource"]["payload_sha256"],
            path=manifest_path,
        )
    for output_name in ("rank_aggregate", "cellchat"):
        output = manifest["output"][output_name]
        _record_sha256(
            records,
            dataset=dataset,
            artifact=f"{artifact_prefix}:{output_name}",
            path=directory / output["filename"],
            expected=output["sha256"],
        )
    return manifest


def _audit_crychic_bundle(
    *,
    records: list[dict[str, Any]],
    dataset: str,
    directory: Path,
    h5ad_sha256: str,
    expected_resource_sha256: str | None,
    artifact_prefix: str,
) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    _record_equality(
        records,
        dataset=dataset,
        artifact=artifact_prefix,
        field="status",
        expected="complete",
        observed=manifest.get("status"),
        path=manifest_path,
    )
    _record_equality(
        records,
        dataset=dataset,
        artifact=artifact_prefix,
        field="input_sha256",
        expected=h5ad_sha256,
        observed=manifest["input"]["sha256"],
        path=manifest_path,
    )
    if expected_resource_sha256 is not None:
        _record_equality(
            records,
            dataset=dataset,
            artifact=artifact_prefix,
            field="resource_table_sha256",
            expected=expected_resource_sha256,
            observed=manifest["resource"]["table_sha256"],
            path=manifest_path,
        )
    output = manifest["output"]
    _record_sha256(
        records,
        dataset=dataset,
        artifact=f"{artifact_prefix}:availability",
        path=directory / output["table"],
        expected=output["table_sha256"],
    )
    return manifest


def _prediction_inventory(
    *,
    spec: DatasetSpec,
    hcommon_liana: dict[str, Any],
    hcommon_crychic: dict[str, Any],
    native_liana: dict[str, Any],
    native_crychic: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method, implementation in METHOD_IMPLEMENTATIONS.items():
        bundle = "hcommon_crychic" if method.startswith("CRYCHIC") else "hcommon_liana"
        records.append(
            {
                "dataset": spec.dataset_id,
                "method": method,
                "implementation": implementation,
                "resource": "H-common 638 simple LR pairs",
                "prediction_exists": True,
                "prediction_is_real": True,
                "formally_evaluated": True,
                "bundle": bundle,
                "exclusion_reason": "",
            }
        )
    records.extend(
        [
            {
                "dataset": spec.dataset_id,
                "method": "LIANA native component bundle",
                "implementation": native_liana["method"]["entrypoints"],
                "resource": native_liana["resource"]["id"],
                "prediction_exists": True,
                "prediction_is_real": True,
                "formally_evaluated": False,
                "bundle": "native_liana",
                "exclusion_reason": (
                    "method-native resource differs from CRYCHIC native resource"
                ),
            },
            {
                "dataset": spec.dataset_id,
                "method": "CRYCHIC native availability-state",
                "implementation": native_crychic["method"]["id"],
                "resource": native_crychic["resource"]["id"],
                "prediction_exists": True,
                "prediction_is_real": True,
                "formally_evaluated": False,
                "bundle": "native_crychic",
                "exclusion_reason": (
                    "method-native resource differs from LIANA native resource"
                ),
            },
        ]
    )
    for method in ("standalone CellPhoneDB", "standalone CellChat R", "NicheNet"):
        records.append(
            {
                "dataset": spec.dataset_id,
                "method": method,
                "implementation": "not available in the audited CITE-seq outputs",
                "resource": "not available",
                "prediction_exists": False,
                "prediction_is_real": False,
                "formally_evaluated": False,
                "bundle": "none",
                "exclusion_reason": "no real prediction artifact",
            }
        )
    return records


def _method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    ranked["auroc_rank"] = ranked.groupby(["comparison_arm", "dataset"], observed=True)[
        "auroc"
    ].rank(ascending=False, method="min", na_option="bottom")
    ranked["auprc_rank"] = ranked.groupby(["comparison_arm", "dataset"], observed=True)[
        "paper_balanced_auprc_mean"
    ].rank(ascending=False, method="min", na_option="bottom")
    return (
        ranked.groupby(["comparison_arm", "method"], observed=True, as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_auroc=("auroc", "mean"),
            median_auroc=("auroc", "median"),
            mean_paper_balanced_auprc=("paper_balanced_auprc_mean", "mean"),
            median_paper_balanced_auprc=("paper_balanced_auprc_mean", "median"),
            mean_truth_key_coverage=("truth_key_coverage_fraction", "mean"),
            mean_positive_truth_key_coverage=(
                "positive_truth_key_coverage_fraction",
                "mean",
            ),
            n_all_scores_tied=("all_evaluated_scores_tied", "sum"),
            mean_auroc_rank=("auroc_rank", "mean"),
            mean_auprc_rank=("auprc_rank", "mean"),
        )
        .sort_values(
            ["comparison_arm", "mean_paper_balanced_auprc"],
            ascending=[True, False],
            kind="stable",
            ignore_index=True,
        )
    )


def _crychic_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    crychic = metrics.loc[
        metrics["method"].eq("CRYCHIC availability-state"),
        [
            "comparison_arm",
            "dataset",
            "auroc",
            "paper_balanced_auprc_mean",
            "truth_key_coverage_fraction",
        ],
    ].rename(
        columns={
            "auroc": "crychic_auroc",
            "paper_balanced_auprc_mean": "crychic_paper_balanced_auprc",
            "truth_key_coverage_fraction": "crychic_truth_key_coverage",
        }
    )
    baselines = metrics.loc[~metrics["method"].eq("CRYCHIC availability-state")]
    compared = baselines.merge(
        crychic, on=["comparison_arm", "dataset"], how="left", validate="many_to_one"
    )
    compared = compared.rename(
        columns={
            "method": "baseline_method",
            "auroc": "baseline_auroc",
            "paper_balanced_auprc_mean": "baseline_paper_balanced_auprc",
            "truth_key_coverage_fraction": "baseline_truth_key_coverage",
        }
    )
    compared["crychic_minus_baseline_auroc"] = (
        compared["crychic_auroc"] - compared["baseline_auroc"]
    )
    compared["crychic_minus_baseline_paper_balanced_auprc"] = (
        compared["crychic_paper_balanced_auprc"]
        - compared["baseline_paper_balanced_auprc"]
    )
    columns = [
        "comparison_arm",
        "dataset",
        "baseline_method",
        "crychic_auroc",
        "baseline_auroc",
        "crychic_minus_baseline_auroc",
        "crychic_paper_balanced_auprc",
        "baseline_paper_balanced_auprc",
        "crychic_minus_baseline_paper_balanced_auprc",
        "crychic_truth_key_coverage",
        "baseline_truth_key_coverage",
    ]
    return compared.loc[:, columns].sort_values(
        ["comparison_arm", "dataset", "baseline_method"], ignore_index=True
    )


def _format_summary_table(summary: pd.DataFrame, arm: str) -> str:
    selected = summary.loc[summary["comparison_arm"].eq(arm)].copy()
    selected = selected.sort_values(
        "mean_paper_balanced_auprc", ascending=False, kind="stable"
    )
    lines = [
        (
            "| Method | Mean AUROC | Mean balanced AUPRC | "
            "Truth-key coverage | All-tied strata |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.mean_auroc:.3f} | "
            f"{row.mean_paper_balanced_auprc:.3f} | "
            f"{row.mean_truth_key_coverage:.3f} | "
            f"{int(row.n_all_scores_tied)}/{int(row.n_datasets)} |"
        )
    return "\n".join(lines)


def _build_report(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    inventory: pd.DataFrame,
) -> str:
    fixed = metrics.loc[metrics["comparison_arm"].eq("H-common/resource_fixed")]
    dataset_count = fixed["dataset"].nunique()
    method_count = fixed["method"].nunique()
    missing = inventory.loc[
        inventory["method"].isin(
            ["standalone CellPhoneDB", "standalone CellChat R", "NicheNet"]
        )
        & ~inventory["prediction_exists"]
    ]["method"].unique()
    return f"""# Real CITE-seq receptor benchmark audit

## Scope

This is a formal real-data receptor-protein validation over {dataset_count}
CITE-seq datasets and {method_count} evaluated score columns. Protein values
were held out from all communication methods and used only to define receptor
positives (`z >= 1.645`) per receiver cell cluster.

The primary `H-common/resource_fixed` panel uses the same 638-pair simple LR
resource and the same sender x receiver x LR evaluation universe for every
method. The ADT truth covers only a subset of that master resource: 9, 9, 3,
and 3 LR pairs in the four datasets, respectively. Missing returned
predictions are ranked after every returned edge.

## Primary comparable result

{_format_summary_table(summary, "H-common/resource_fixed")}

These values are descriptive across four single-sample strata. They do not by
themselves establish general superiority, and the truth labels validate
receiver-receptor specificity rather than a complete ligand-receptor event.

## Paper protocol

- AUROC is tie-aware on each complete evaluated stratum.
- Balanced AUPRC keeps every positive and samples the same number of negatives
  with replacement, 100 times, resetting seed 1234 per method/stratum.
- AUPRC is trapezoidal PR AUC with the yardstick 0.0.8 `(recall=0,
  precision=1)` boundary, not average precision.
- Under this old trapezoidal convention, an all-tied balanced predictor yields
  AUPRC 0.75 despite AUROC 0.5. `n_unique_oriented_scores` and
  `all_evaluated_scores_tied` identify this interpolation artifact; such a
  value is not evidence of discrimination.
- The released R code duplicates each repeat AUC across threshold rows before
  averaging. `released_row_weighted_auprc_diagnostic` preserves that behavior
  as a diagnostic; `paper_balanced_auprc_mean` is the interpretable mean of the
  100 repeat-level AUCs.
- NumPy MT19937 follows the paper's sampling design and seed but is not
  byte-identical to R `sample()`, so individual negative draws may differ.

## Non-comparable literature-style panel

`H-common/independent_returned` follows the released paper's method-specific
returned universes. Its AUROC/AUPRC values are not direct rankings because
coverage and positive/negative label sets differ by method. Coverage is
therefore reported beside every metric.

## Prediction provenance

All evaluated baseline columns are real LIANA 1.7.3 outputs. CellPhoneDB,
CellChat, Connectome, NATMI, logFC, and SingleCellSignalR names refer to LIANA
component implementations, not separate standalone package runs. Existing
native LIANA and native CRYCHIC predictions passed input/output checksum audit,
but were excluded from the primary comparison because their native resources
differ (LIANA consensus versus CellPhoneDB v5).

No real artifacts were found for: {", ".join(sorted(missing))}.

## Biological and statistical limits

- Each prepared H5AD has one sample, one subject, and one CITE-seq context;
  condition-specific or differential communication is not estimable here.
- CRYCHIC's evaluated column is a static availability-state diagnostic, not a
  communication probability, differential effect, p-value, or q-value.
- ADT truth says whether a receptor is unusually high in a receiver cluster.
  It does not prove ligand binding, sender causality, or downstream signaling.
- The 9/9/3/3 truth-covered LR subsets are narrow relative to the 638-pair
  H-common resource, so this benchmark cannot characterize full-resource
  performance.
- Cell clusters are reused across many LR edges, so edge rows are not
  independent biological replicates. The four datasets are reported
  separately and aggregated descriptively.
- Only the resource-fixed panel supports direct same-universe comparison.
"""


def run_audit(
    *,
    workspace: Path,
    crychic_hcommon_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    resource_dir = (
        workspace
        / "benchmark_work"
        / "multicondition_v01"
        / "resources"
        / "harmonized_simple_lr"
    )
    resource_path = resource_dir / "harmonized_lr.tsv"
    resource_manifest_path = resource_dir / "manifest.json"
    resource_manifest = _read_json(resource_manifest_path)
    expected_resource_sha = resource_manifest["payload"]["sha256"]
    audit_records: list[dict[str, Any]] = []
    _record_sha256(
        audit_records,
        dataset="all",
        artifact="H-common resource",
        path=resource_path,
        expected=expected_resource_sha,
    )
    resource = pd.read_csv(resource_path, sep="\t")

    metrics_frames: list[pd.DataFrame] = []
    replicate_frames: list[pd.DataFrame] = []
    retained_frames: list[pd.DataFrame] = []
    inventory_records: list[dict[str, Any]] = []
    dataset_records: list[dict[str, Any]] = []

    for spec in DATASETS:
        paths = _dataset_paths(workspace, crychic_hcommon_root, spec)
        h5ad_sha = sha256_file(paths["h5ad"])
        truth_manifest = _read_json(paths["truth_manifest"])
        _record_sha256(
            audit_records,
            dataset=spec.dataset_id,
            artifact="receptor truth",
            path=paths["truth"],
            expected=truth_manifest["output_sha256"],
        )
        hcommon_liana = _audit_liana_bundle(
            records=audit_records,
            dataset=spec.dataset_id,
            directory=paths["hcommon_liana"],
            h5ad_sha256=h5ad_sha,
            expected_resource_sha256=expected_resource_sha,
            artifact_prefix="H-common LIANA",
        )
        hcommon_crychic = _audit_crychic_bundle(
            records=audit_records,
            dataset=spec.dataset_id,
            directory=paths["hcommon_crychic"],
            h5ad_sha256=h5ad_sha,
            expected_resource_sha256=expected_resource_sha,
            artifact_prefix="H-common CRYCHIC",
        )
        native_liana = _audit_liana_bundle(
            records=audit_records,
            dataset=spec.dataset_id,
            directory=paths["native_liana"],
            h5ad_sha256=h5ad_sha,
            expected_resource_sha256=None,
            artifact_prefix="native LIANA inventory",
        )
        native_crychic = _audit_crychic_bundle(
            records=audit_records,
            dataset=spec.dataset_id,
            directory=paths["native_crychic"],
            h5ad_sha256=h5ad_sha,
            expected_resource_sha256=None,
            artifact_prefix="native CRYCHIC inventory",
        )
        inventory_records.extend(
            _prediction_inventory(
                spec=spec,
                hcommon_liana=hcommon_liana,
                hcommon_crychic=hcommon_crychic,
                native_liana=native_liana,
                native_crychic=native_crychic,
            )
        )
        dataset_records.append(
            {
                "dataset": spec.dataset_id,
                "h5ad_sha256": h5ad_sha,
                "cells": hcommon_liana["input"]["shape"][0],
                "genes": hcommon_liana["input"]["shape"][1],
                "cell_clusters": hcommon_liana["input"]["groups"],
                "truth_rows": truth_manifest["rows"],
                "truth_positive_rows": truth_manifest["positive_rows"],
                "samples": 1,
                "subjects": 1,
                "contexts": 1,
                "differential_inference_estimable": False,
            }
        )

        truth = pd.read_csv(paths["truth"], sep="\t")
        rank_raw = pd.read_parquet(
            paths["hcommon_liana"] / "rank_aggregate_raw.parquet"
        )
        cellchat_raw = pd.read_parquet(paths["hcommon_liana"] / "cellchat_raw.parquet")
        crychic_raw = pd.read_parquet(
            paths["hcommon_crychic"] / "availability_scores.parquet"
        )
        scores = pd.concat(
            [
                component_scores_from_liana(
                    rank_raw, dataset=spec.dataset_id, resource_mode="H-common"
                ),
                component_scores_from_cellchat(
                    cellchat_raw,
                    dataset=spec.dataset_id,
                    resource_mode="H-common",
                ),
                scores_from_crychic_availability(
                    crychic_raw,
                    dataset=spec.dataset_id,
                    resource_mode="H-common",
                ),
            ],
            ignore_index=True,
        )

        for universe_mode, comparison_arm in (
            ("resource_fixed", "H-common/resource_fixed"),
            ("independent", "H-common/independent_returned"),
        ):
            retained, _ = evaluate_citeseq_components(
                scores,
                truth,
                universe_mode=universe_mode,
                resource=resource if universe_mode == "resource_fixed" else None,
                n_bootstrap=1,
                n_negative_samples=1,
                random_seed=PAPER_SEED,
            )
            retained = retained.copy()
            retained["comparison_arm"] = comparison_arm
            metrics, replicates = _summarize_retained(
                retained,
                truth,
                comparison_arm=comparison_arm,
            )
            metrics_frames.append(metrics)
            replicate_frames.append(replicates)
            retained_frames.append(retained)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    replicates = pd.concat(replicate_frames, ignore_index=True)
    retained = pd.concat(retained_frames, ignore_index=True, sort=False)
    inventory = pd.DataFrame.from_records(inventory_records)
    datasets = pd.DataFrame.from_records(dataset_records)
    audit = pd.DataFrame.from_records(audit_records)
    summary = _method_summary(metrics)
    comparison = _crychic_comparison(metrics)

    outputs = {
        "metrics_by_dataset.tsv": metrics,
        "paper_balanced_auprc_replicates.tsv": replicates,
        "method_summary.tsv": summary,
        "crychic_vs_baselines.tsv": comparison,
        "prediction_inventory.tsv": inventory,
        "dataset_inventory.tsv": datasets,
        "artifact_audit.tsv": audit,
    }
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, sep="\t", index=False)
    retained.to_parquet(output_dir / "evaluated_scores.parquet", index=False)
    report = _build_report(metrics, summary, inventory)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")

    artifact_paths = [output_dir / name for name in outputs]
    artifact_paths.extend(
        [output_dir / "evaluated_scores.parquet", output_dir / "REPORT.md"]
    )
    manifest = {
        "schema_version": "crychic-citeseq-receptor-paper-audit-v1",
        "status": "complete",
        "datasets": [spec.dataset_id for spec in DATASETS],
        "formal_data_type": "real CITE-seq; no synthetic or smoke results",
        "primary_comparison": "H-common/resource_fixed",
        "paper_protocol": {
            "auroc": "tie-aware complete-stratum AUROC",
            "auprc": "yardstick 0.0.8 style trapezoidal PR AUC",
            "negative_sampling": "all positives plus 1:1 negatives with replacement",
            "repeats": PAPER_REPEATS,
            "seed": PAPER_SEED,
            "rng": "NumPy RandomState MT19937; not byte-identical to R sample()",
        },
        "resource": {
            "path": str(resource_path),
            "sha256": expected_resource_sha,
            "manifest_path": str(resource_manifest_path),
            "manifest_sha256": sha256_file(resource_manifest_path),
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
        },
        "artifacts": [
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(artifact_paths)
        ],
        "warnings": [
            "independent returned universes are not directly comparable",
            "LIANA component implementations are not standalone package runs",
            "CRYCHIC score is static availability, not differential communication",
            "ADT truth validates receptor specificity, not complete LR causality",
            "Python and R RNG draws are not byte-identical",
            "all-tied scores yield trapezoidal balanced PR AUC 0.75",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(".."),
        help="crychic_dev workspace root",
    )
    parser.add_argument(
        "--crychic-hcommon-root",
        type=Path,
        default=Path(
            "../benchmark_work/citeseq_receptor_audit_20260717/"
            "method_runs/crychic_hcommon"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../benchmark_work/citeseq_receptor_audit_20260717/evaluation_v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_audit(
        workspace=args.workspace,
        crychic_hcommon_root=args.crychic_hcommon_root.resolve(),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
