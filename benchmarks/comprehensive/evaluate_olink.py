"""Evaluate frozen ligand predictions against held-out MIS-C Olink truth."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmarks.adapters.common import (
    canonical_digest,
    git_metadata,
    json_safe,
    sha256_file,
)

TRUTH_SCHEMA = "crychic-misc-olink-truth-v1"
OUTPUT_SCHEMA = "crychic-misc-olink-evaluation-v1"
DATASET_ID = "misc_olink"
CONTRAST_SIGNS = {"misc_m_vs_s": 1.0, "misc_s_vs_m": -1.0}
COVERAGE_THRESHOLD = 0.80
PREDICTION_COLUMNS = (
    "dataset_id",
    "method_id",
    "resource_mode",
    "universe_id",
    "contrast",
    "ligand",
    "score",
    "status",
)
TRUTH_COLUMNS = (
    "schema_version",
    "dataset_id",
    "source_row",
    "contrast_id",
    "contrast_label",
    "contrast_numerator",
    "contrast_denominator",
    "effect_semantics",
    "q_value_semantics",
    "analyte_original",
    "analyte_make_names",
    "variables",
    "mean_d0",
    "mean_hc",
    "mean_diff_d0",
    "unadj_p_values",
    "adj_p_values",
)
OBSERVED_STATUSES = frozenset({"observed", "ok"})
ALLOWED_STATUSES = frozenset(
    {
        *OBSERVED_STATUSES,
        "not_estimable",
        "not_returned",
        "resource_unavailable",
        "unsupported_resource",
        "insufficient_cells",
        "method_failed",
        "missing",
    }
)
RESOURCE_MODES = frozenset({"H-common", "H-covered", "native"})
METRICS_COLUMNS = (
    "schema_version",
    "dataset_id",
    "method_id",
    "resource_mode",
    "universe_id",
    "contrast_id",
    "status",
    "rank_eligible",
    "ranking_exclusion_reason",
    "coverage_threshold",
    "n_truth_olink_analytes",
    "n_truth_directional_q_positive",
    "n_representable_measured_ligands",
    "n_representable_directional_q_positive",
    "n_prediction_ligands",
    "n_prediction_ligands_unmeasured",
    "n_measured_ligands_scored",
    "truth_representable_fraction",
    "representable_scored_fraction",
    "prediction_measured_fraction",
    "n_scored_directional_q_positive",
    "n_scored_ap_negative",
    "olink_ligand_ap",
    "olink_ligand_ap_status",
    "olink_ligand_ap_reason",
    "olink_logfc_spearman",
    "olink_logfc_spearman_status",
    "olink_logfc_spearman_reason",
    "protein_ndcg",
    "protein_ndcg_status",
    "protein_ndcg_reason",
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return cast(dict[str, Any], value)


def _require_sha256(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{label} must be a lowercase or uppercase SHA256 digest")
    return normalized


def ligand_universe_id(ligands: Sequence[str]) -> str:
    """Return the canonical identity of one Olink-blind ligand universe."""
    normalized = sorted({str(ligand) for ligand in ligands})
    if not normalized or any(not ligand for ligand in normalized):
        raise ValueError("ligand universe must contain non-empty identifiers")
    return canonical_digest(
        {"ligands": normalized},
        prefix="olink_ligand_universe",
    )


def _strict_strings(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    if frame.loc[:, columns].isna().any().any():
        raise ValueError(f"{label} identifiers contain null values")
    for column in columns:
        frame[column] = frame[column].astype(str)
        if frame[column].str.len().eq(0).any():
            raise ValueError(f"{label} identifier {column!r} contains empty values")
        if frame[column].str.contains("[\t\r\n]", regex=True).any():
            raise ValueError(f"{label} identifier {column!r} is not one-line text")


def _read_predictions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, sep="\t")
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("predictions must be TSV, CSV, or Parquet")
    missing = set(PREDICTION_COLUMNS).difference(frame.columns)
    extra = set(frame.columns).difference(PREDICTION_COLUMNS)
    if missing or extra:
        raise ValueError(
            "prediction columns differ from the frozen Olink-free schema: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if frame.empty:
        raise ValueError("prediction table must not be empty")
    frame = cast(pd.DataFrame, frame.loc[:, PREDICTION_COLUMNS].copy())
    identifier_columns = [
        "dataset_id",
        "method_id",
        "resource_mode",
        "universe_id",
        "contrast",
        "ligand",
        "status",
    ]
    _strict_strings(frame, identifier_columns, label="prediction")
    if set(frame["dataset_id"]) != {DATASET_ID}:
        raise ValueError(f"prediction dataset_id must be {DATASET_ID!r}")
    invalid_modes = set(frame["resource_mode"]).difference(RESOURCE_MODES)
    if invalid_modes:
        raise ValueError(
            f"prediction resource modes are invalid: {sorted(invalid_modes)}"
        )
    invalid_contrasts = set(frame["contrast"]).difference(CONTRAST_SIGNS)
    if invalid_contrasts:
        raise ValueError(
            "only canonical misc_m_vs_s and derived misc_s_vs_m are supported: "
            f"{sorted(invalid_contrasts)}"
        )
    invalid_statuses = set(frame["status"]).difference(ALLOWED_STATUSES)
    if invalid_statuses:
        raise ValueError(f"prediction statuses are invalid: {sorted(invalid_statuses)}")

    numeric_score = pd.to_numeric(frame["score"], errors="coerce")
    supplied = frame["score"].notna()
    if (supplied & numeric_score.isna()).any() or np.isinf(
        numeric_score.fillna(0).to_numpy(dtype=float)
    ).any():
        raise ValueError("supplied prediction scores must be finite numeric values")
    observed = frame["status"].isin(OBSERVED_STATUSES)
    if numeric_score.loc[observed].isna().any():
        raise ValueError("observed prediction status requires a score")
    if numeric_score.loc[~observed].notna().any():
        raise ValueError("non-observed prediction statuses require missing scores")
    frame["score"] = numeric_score.astype(float)
    arm_key = ["dataset_id", "method_id", "resource_mode", "contrast"]
    if frame.groupby(arm_key, observed=True)["universe_id"].nunique().gt(1).any():
        raise ValueError("each prediction arm must declare exactly one universe_id")
    for key_values, arm in frame.groupby(arm_key, sort=True, observed=True):
        declared = str(arm["universe_id"].iloc[0])
        expected = ligand_universe_id(arm["ligand"].astype(str).tolist())
        if declared != expected:
            raise ValueError(
                "prediction universe_id does not match its complete ligand set; "
                f"affected={key_values!r}, expected={expected!r}"
            )
    key = [*arm_key, "ligand"]
    if frame.duplicated(key, keep=False).any():
        raise ValueError("prediction table contains duplicate ligand-arm keys")
    _validate_prediction_universes(frame)
    return frame


def _validate_prediction_universes(frame: pd.DataFrame) -> None:
    """Require like-for-like H-common arms to materialize one fixed universe."""
    common = frame.loc[frame["resource_mode"].eq("H-common")]
    if common.empty:
        return
    for key, group in common.groupby(
        ["dataset_id", "contrast"], sort=True, observed=True
    ):
        universe_ids = set(group["universe_id"].astype(str))
        if len(universe_ids) != 1:
            raise ValueError(
                "H-common prediction arms for a dataset/contrast must share one "
                f"universe_id; affected={key!r}"
            )
        ligand_sets = {
            method_id: frozenset(method["ligand"].astype(str))
            for method_id, method in group.groupby(
                "method_id", sort=True, observed=True
            )
        }
        reference = next(iter(ligand_sets.values()))
        mismatched = sorted(
            method_id
            for method_id, ligands in ligand_sets.items()
            if ligands != reference
        )
        if mismatched:
            raise ValueError(
                "methods sharing an H-common universe_id must materialize the "
                f"same ligand universe; affected={mismatched}"
            )

    forward = common.loc[common["contrast"].eq("misc_m_vs_s")]
    reverse = common.loc[common["contrast"].eq("misc_s_vs_m")]
    if forward.empty or reverse.empty:
        return
    forward_sets = {
        (str(method_id), str(resource_mode)): frozenset(group["ligand"].astype(str))
        for (method_id, resource_mode), group in forward.groupby(
            ["method_id", "resource_mode"], sort=True, observed=True
        )
    }
    for key, group in reverse.groupby(
        ["method_id", "resource_mode"], sort=True, observed=True
    ):
        identity = (str(key[0]), str(key[1]))
        expected = forward_sets.get(identity)
        if expected is not None and frozenset(group["ligand"].astype(str)) != expected:
            raise ValueError(
                "forward and derived reverse H-common arms must materialize the "
                f"same ligand universe; affected={identity!r}"
            )


def _validate_truth_manifest(
    manifest: Mapping[str, Any], truth_path: Path
) -> dict[str, Any]:
    if manifest.get("schema_version") != TRUTH_SCHEMA:
        raise ValueError("truth manifest has an unsupported schema_version")
    if manifest.get("status") != "complete":
        raise ValueError("truth manifest is not complete")
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("truth manifest has an unexpected dataset_id")
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("truth manifest output must be an object")
    truth_record = output.get("truth_tsv")
    if not isinstance(truth_record, Mapping):
        raise ValueError("truth manifest does not pin truth_tsv")
    expected = _require_sha256(
        str(truth_record.get("sha256", "")), label="truth manifest checksum"
    )
    if sha256_file(truth_path) != expected:
        raise ValueError("truth TSV checksum differs from its frozen manifest")

    contract = manifest.get("truth_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("truth manifest truth_contract must be an object")
    expected_contract = {
        "canonical_contrast_id": "misc_m_vs_s",
        "contrast_label": "M-vs-S",
        "effect_column": "mean_diff_d0",
        "reverse_contrast_id": "misc_s_vs_m",
        "reverse_contrast_label": "S-vs-M",
        "reverse_is_evaluator_derived": True,
        "reverse_effect": "-mean_diff_d0",
        "q_value_column": "adj_p_values",
        "olink_is_held_out_evaluation_only": True,
        "olink_must_not_enter_method_scores": True,
        "missing_analytes_are_not_zero": True,
    }
    for field, expected_value in expected_contract.items():
        if contract.get(field) != expected_value:
            raise ValueError(
                f"truth manifest contract {field!r} is not {expected_value!r}"
            )
    return dict(contract)


def _read_truth(path: Path, manifest: Mapping[str, Any]) -> pd.DataFrame:
    _validate_truth_manifest(manifest, path)
    frame = pd.read_csv(path, sep="\t")
    missing = set(TRUTH_COLUMNS).difference(frame.columns)
    extra = set(frame.columns).difference(TRUTH_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"truth TSV columns differ: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    if frame.empty:
        raise ValueError("truth TSV must not be empty")
    frame = cast(pd.DataFrame, frame.loc[:, TRUTH_COLUMNS].copy())
    identifiers = [
        "schema_version",
        "dataset_id",
        "contrast_id",
        "contrast_label",
        "contrast_numerator",
        "contrast_denominator",
        "effect_semantics",
        "q_value_semantics",
        "analyte_original",
        "analyte_make_names",
        "variables",
    ]
    _strict_strings(frame, identifiers, label="truth")
    if set(frame["schema_version"]) != {TRUTH_SCHEMA}:
        raise ValueError("truth TSV has an unsupported schema_version")
    if set(frame["dataset_id"]) != {DATASET_ID}:
        raise ValueError("truth TSV has an unexpected dataset_id")
    if set(frame["contrast_id"]) != {"misc_m_vs_s"}:
        raise ValueError("truth TSV must store canonical contrast misc_m_vs_s")
    if set(frame["contrast_label"]) != {"M-vs-S"}:
        raise ValueError("truth TSV contrast_label must be M-vs-S")
    if not frame["variables"].eq(frame["analyte_original"]).all():
        raise ValueError("truth TSV does not preserve the source variables mapping")
    if frame["analyte_original"].duplicated().any():
        raise ValueError("truth source analytes must be unique")
    if frame["analyte_make_names"].duplicated().any():
        raise ValueError("truth make.names analytes must be unique")

    source_rows = pd.to_numeric(frame["source_row"], errors="coerce")
    expected_rows = np.arange(1, len(frame) + 1, dtype=float)
    if source_rows.isna().any() or not np.array_equal(
        source_rows.to_numpy(dtype=float), expected_rows
    ):
        raise ValueError("truth source_row must preserve one-based worksheet order")
    numeric_columns = [
        "mean_d0",
        "mean_hc",
        "mean_diff_d0",
        "unadj_p_values",
        "adj_p_values",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"truth column {column!r} must be finite numeric")
        frame[column] = values.astype(float)
    for column in ("unadj_p_values", "adj_p_values"):
        if frame[column].lt(0).any() or frame[column].gt(1).any():
            raise ValueError(f"truth column {column!r} must lie in [0, 1]")
    residual = frame["mean_diff_d0"] - (frame["mean_d0"] - frame["mean_hc"])
    if residual.abs().max() > 1e-10:
        raise ValueError("truth mean_diff_d0 is not mean_d0 minus mean_hc")
    return frame


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold-group average precision with deterministic tie handling."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("average precision requires both truth classes")
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    boundaries = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), len(scores) - 1]
    cumulative_true = np.cumsum(ordered_labels, dtype=float)
    cumulative_total = boundaries + 1
    true_at_threshold = cumulative_true[boundaries]
    precision = true_at_threshold / cumulative_total
    recall = true_at_threshold / positives
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def _tie_aware_dcg(relevance: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_relevance = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, len(scores) + 2, dtype=float))
    result = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and ordered_scores[end] == ordered_scores[start]:
            end += 1
        result += float(
            ordered_relevance[start:end].sum() * discounts[start:end].mean()
        )
        start = end
    return result


def _directional_ndcg(relevance: np.ndarray, scores: np.ndarray) -> float:
    relevance = np.asarray(relevance, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(relevance) == 0 or relevance.max(initial=0.0) <= 0:
        raise ValueError("NDCG requires positive directional relevance")
    observed = _tie_aware_dcg(relevance, scores)
    ideal = _tie_aware_dcg(relevance, relevance)
    if ideal <= 0:
        raise ValueError("NDCG ideal DCG is zero")
    return float(observed / ideal)


def _metric_result(
    values: pd.DataFrame,
    *,
    signed_effect: pd.Series,
) -> dict[str, Any]:
    if values.empty:
        return {
            "ap": math.nan,
            "ap_status": "not_estimable",
            "ap_reason": "no_measured_ligands_with_observed_scores",
            "spearman": math.nan,
            "spearman_status": "not_estimable",
            "spearman_reason": "no_measured_ligands_with_observed_scores",
            "ndcg": math.nan,
            "ndcg_status": "not_estimable",
            "ndcg_reason": "no_measured_ligands_with_observed_scores",
            "n_positive": 0,
            "n_negative": 0,
        }

    scores = values["score"].to_numpy(dtype=float)
    effects = signed_effect.loc[values.index].to_numpy(dtype=float)
    labels = (values["adj_p_values"].to_numpy(dtype=float) <= 0.05) & (effects > 0)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        ap = math.nan
        ap_status = "not_estimable"
        ap_reason = "single_class_direction_matched_q_0_05_truth"
    else:
        ap = _average_precision(labels, scores)
        ap_status = "observed"
        ap_reason = None

    if len(scores) < 2:
        correlation = math.nan
        correlation_status = "not_estimable"
        correlation_reason = "fewer_than_two_scored_measured_ligands"
    elif np.unique(scores).size < 2 or np.unique(effects).size < 2:
        correlation = math.nan
        correlation_status = "not_estimable"
        correlation_reason = "constant_score_or_truth_effect"
    else:
        correlation = float(spearmanr(scores, effects).statistic)
        if np.isfinite(correlation):
            correlation_status = "observed"
            correlation_reason = None
        else:
            correlation = math.nan
            correlation_status = "not_estimable"
            correlation_reason = "non_finite_spearman_result"

    relevance = np.maximum(effects, 0.0)
    if relevance.max(initial=0.0) <= 0:
        ndcg = math.nan
        ndcg_status = "not_estimable"
        ndcg_reason = "no_positive_directional_effect_magnitude"
    else:
        ndcg = _directional_ndcg(relevance, scores)
        ndcg_status = "observed"
        ndcg_reason = None
    return {
        "ap": ap,
        "ap_status": ap_status,
        "ap_reason": ap_reason,
        "spearman": correlation,
        "spearman_status": correlation_status,
        "spearman_reason": correlation_reason,
        "ndcg": ndcg,
        "ndcg_status": ndcg_status,
        "ndcg_reason": ndcg_reason,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate every method/resource/contrast arm without imputing missing scores."""
    records: list[dict[str, Any]] = []
    n_truth = len(truth)
    truth_ligands = frozenset(truth["analyte_make_names"].astype(str))
    group_columns = [
        "dataset_id",
        "method_id",
        "resource_mode",
        "universe_id",
        "contrast",
    ]
    for key, arm in predictions.groupby(group_columns, sort=True, observed=True):
        dataset_id, method_id, resource_mode, universe_id, contrast = map(str, key)
        sign = CONTRAST_SIGNS[contrast]
        arm = arm.copy()
        arm["__prediction_present"] = True
        representable = truth.merge(
            arm.loc[:, ["ligand", "score", "status", "__prediction_present"]],
            left_on="analyte_make_names",
            right_on="ligand",
            how="inner",
            sort=False,
            validate="one_to_one",
        )
        scored = representable["status"].isin(OBSERVED_STATUSES)
        scored_values = representable.loc[scored].copy()
        signed_effect = representable["mean_diff_d0"] * sign
        metric = _metric_result(scored_values, signed_effect=signed_effect)

        prediction_ligands = frozenset(arm["ligand"].astype(str))
        n_overlap = len(prediction_ligands.intersection(truth_ligands))
        n_prediction = len(prediction_ligands)
        n_scored = int(scored.sum())
        truth_representable = n_overlap / n_truth
        representable_scored = n_scored / n_overlap if n_overlap else 0.0
        prediction_measured = n_overlap / n_prediction if n_prediction else math.nan
        truth_labels = (truth["adj_p_values"] <= 0.05) & (
            truth["mean_diff_d0"] * sign > 0
        )
        representable_labels = (representable["adj_p_values"] <= 0.05) & (
            representable["mean_diff_d0"] * sign > 0
        )
        coverage_eligible = representable_scored >= COVERAGE_THRESHOLD
        primary_estimable = metric["ap_status"] == "observed"
        is_preregistered_primary = (
            contrast == "misc_m_vs_s" and resource_mode == "H-common"
        )
        rank_eligible = bool(
            is_preregistered_primary and coverage_eligible and primary_estimable
        )
        if contrast != "misc_m_vs_s":
            exclusion = "derived_reverse_contrast_diagnostic_only"
        elif resource_mode != "H-common":
            exclusion = "non_h_common_resource_diagnostic_only"
        elif n_overlap == 0:
            exclusion = "no_olink_measured_ligands_in_frozen_universe"
        elif not coverage_eligible:
            exclusion = "representable_scored_coverage_below_0_80_diagnostic_only"
        elif not primary_estimable:
            exclusion = str(metric["ap_reason"])
        else:
            exclusion = None
        metric_statuses = {
            str(metric["ap_status"]),
            str(metric["spearman_status"]),
            str(metric["ndcg_status"]),
        }
        overall_status = (
            "observed" if "observed" in metric_statuses else "not_estimable"
        )
        records.append(
            {
                "schema_version": OUTPUT_SCHEMA,
                "dataset_id": dataset_id,
                "method_id": method_id,
                "resource_mode": resource_mode,
                "universe_id": universe_id,
                "contrast_id": contrast,
                "status": overall_status,
                "rank_eligible": rank_eligible,
                "ranking_exclusion_reason": exclusion,
                "coverage_threshold": COVERAGE_THRESHOLD,
                "n_truth_olink_analytes": n_truth,
                "n_truth_directional_q_positive": int(truth_labels.sum()),
                "n_representable_measured_ligands": n_overlap,
                "n_representable_directional_q_positive": int(
                    representable_labels.sum()
                ),
                "n_prediction_ligands": n_prediction,
                "n_prediction_ligands_unmeasured": n_prediction - n_overlap,
                "n_measured_ligands_scored": n_scored,
                "truth_representable_fraction": truth_representable,
                "representable_scored_fraction": representable_scored,
                "prediction_measured_fraction": prediction_measured,
                "n_scored_directional_q_positive": metric["n_positive"],
                "n_scored_ap_negative": metric["n_negative"],
                "olink_ligand_ap": metric["ap"],
                "olink_ligand_ap_status": metric["ap_status"],
                "olink_ligand_ap_reason": metric["ap_reason"],
                "olink_logfc_spearman": metric["spearman"],
                "olink_logfc_spearman_status": metric["spearman_status"],
                "olink_logfc_spearman_reason": metric["spearman_reason"],
                "protein_ndcg": metric["ndcg"],
                "protein_ndcg_status": metric["ndcg_status"],
                "protein_ndcg_reason": metric["ndcg_reason"],
            }
        )
    result = pd.DataFrame.from_records(records, columns=METRICS_COLUMNS)
    if result.empty:
        raise ValueError("prediction table contains no evaluable arms")
    return result


def _strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _publish_directory(staged: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    if not output.exists():
        os.replace(staged, output)
        return
    backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
    os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def run(
    predictions_path: Path,
    truth_path: Path,
    truth_manifest_path: Path,
    output_dir: Path,
    *,
    predictions_sha256: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify frozen inputs, evaluate them, and atomically publish metrics."""
    predictions_path = predictions_path.resolve()
    truth_path = truth_path.resolve()
    truth_manifest_path = truth_manifest_path.resolve()
    output_dir = output_dir.resolve()
    for path, label in (
        (predictions_path, "predictions"),
        (truth_path, "truth TSV"),
        (truth_manifest_path, "truth manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_dir}; pass --overwrite to replace it"
        )

    expected_prediction_sha = _require_sha256(
        predictions_sha256, label="predictions_sha256"
    )
    actual_prediction_sha = sha256_file(predictions_path)
    if actual_prediction_sha != expected_prediction_sha:
        raise ValueError("prediction file checksum differs from the frozen checksum")

    # The frozen prediction table is validated before Olink outcomes are opened.
    predictions = _read_predictions(predictions_path)
    truth_manifest = _json_object(truth_manifest_path)
    truth = _read_truth(truth_path, truth_manifest)
    metrics = evaluate_predictions(predictions, truth)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        metrics_path = staged / "metrics.tsv"
        metrics.to_csv(
            metrics_path,
            sep="\t",
            index=False,
            lineterminator="\n",
            na_rep="",
        )
        repo_root = Path(__file__).resolve().parents[2]
        manifest: dict[str, Any] = {
            "schema_version": OUTPUT_SCHEMA,
            "status": "complete",
            "dataset_id": DATASET_ID,
            "inputs": {
                "predictions": {
                    "filename": predictions_path.name,
                    "bytes": predictions_path.stat().st_size,
                    "rows": len(predictions),
                    "expected_sha256": expected_prediction_sha,
                    "sha256": actual_prediction_sha,
                },
                "truth_tsv": {
                    "filename": truth_path.name,
                    "bytes": truth_path.stat().st_size,
                    "rows": len(truth),
                    "sha256": sha256_file(truth_path),
                },
                "truth_manifest": {
                    "filename": truth_manifest_path.name,
                    "bytes": truth_manifest_path.stat().st_size,
                    "sha256": sha256_file(truth_manifest_path),
                },
            },
            "evaluation_contract": {
                "supported_contrasts": list(CONTRAST_SIGNS),
                "reverse_contrast_negates_mean_diff_d0_only": True,
                "primary_ranking_resource_mode": "H-common",
                "h_common_methods_share_exact_ligand_universe": True,
                "universe_id_is_canonical_ligand_set_digest": True,
                "prediction_rows_materialize_complete_olink_blind_universe": True,
                "universe_frozen_before_olink_is_opened": True,
                "q_value_column": "adj_p_values",
                "q_threshold": 0.05,
                "ap_truth": "q<=0.05 and signed effect>0",
                "spearman_truth": "signed mean_diff_d0",
                "ndcg_relevance": "max(signed mean_diff_d0, 0); linear gain",
                "ndcg_ties": "average discount within tied prediction scores",
                "metric_universe": (
                    "intersection of measured Olink analytes and the frozen "
                    "Olink-blind ligand universe"
                ),
                "representability_denominator": "all measured Olink analytes",
                "scored_coverage_denominator": (
                    "Olink-measured ligands in the frozen representable universe"
                ),
                "minimum_representable_scored_coverage_for_ranking": (
                    COVERAGE_THRESHOLD
                ),
                "below_threshold_values_are_diagnostic_only": True,
                "native_and_h_covered_values_are_diagnostic_only": True,
                "single_class_metrics_are_not_estimable": True,
                "unmeasured_or_unscored_ligands_are_not_imputed": True,
                "olink_values_used_only_as_held_out_outcomes": True,
                "prediction_schema_rejects_olink_columns": True,
                "prediction_score_is_never_recalculated_from_olink": True,
                "truth_contract": truth_manifest["truth_contract"],
            },
            "summary": {
                "arms": len(metrics),
                "rank_eligible_arms": int(metrics["rank_eligible"].sum()),
                "diagnostic_only_arms": int((~metrics["rank_eligible"]).sum()),
            },
            "records": metrics.to_dict(orient="records"),
            "output": {
                "metrics_tsv": {
                    "filename": metrics_path.name,
                    "rows": len(metrics),
                    "bytes": metrics_path.stat().st_size,
                    "sha256": sha256_file(metrics_path),
                }
            },
            "code": {
                **git_metadata(repo_root),
                "entrypoint": str(Path(__file__).relative_to(repo_root)),
                "entrypoint_sha256": sha256_file(__file__),
            },
        }
        manifest = cast(dict[str, Any], json_safe(manifest))
        _strict_json(staged / "manifest.json", manifest)
        _publish_directory(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--truth-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(
        args.predictions,
        args.truth,
        args.truth_manifest,
        args.output_dir,
        predictions_sha256=args.predictions_sha256,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dataset_id": manifest["dataset_id"],
                **cast(dict[str, Any], manifest["summary"]),
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
