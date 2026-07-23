"""Evaluate multi-condition CCC methods on the frozen three-group event axis."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.evaluate_g1 import (
    average_precision,
    prevalence_adjusted_average_precision,
    tie_aware_auroc,
)
from benchmarks.metrics.multicondition import (
    external_long_to_score_table,
    unpaired_edge_effects,
)

EDGE_KEYS = ("sender", "receiver", "interaction_id", "ligand", "receptor")
EVENT_KEYS = ("sender", "receiver", "ligand", "receptor")
METHODS = ("crychic", "scseqcommdiff", "cellchat", "liana_rank_aggregate")
METHOD_LABELS = {
    "crychic": "CRYCHIC generic baseline",
    "scseqcommdiff": "scSeqCommDiff",
    "cellchat": "CellChat",
    "liana_rank_aggregate": "LIANA",
}
EXPECTED_METHOD_VERSIONS = {
    "scseqcommdiff": "2.0.0",
    "cellchat": "2.1.2",
    "liana_rank_aggregate": "1.7.3",
}
EXTERNAL_SCHEMA = "crychic-external-adapter-manifest-v1"
SCSEQ_SCHEMA = "crychic-scseqcommdiff-paper-benchmark-v2"
MINIMUM_COVERAGE = 0.80
TARGET_PREVALENCE = 0.10
METRIC_EFFECT_DECIMALS = 12
CRYCHIC_SCORE_HEAD_VERSION = "0.1.0-exploratory"
MULTIGROUP_METRIC_SCHEMA = "crychic-three-group-decomposed-metrics-v1"


@dataclass(frozen=True)
class RunRecord:
    method: str
    dataset_id: str
    contrast: str | None
    directory: Path
    manifest: Mapping[str, Any]
    kind: str


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _discover_runs(root: Path) -> tuple[list[RunRecord], list[dict[str, str]]]:
    records: list[RunRecord] = []
    excluded: list[dict[str, str]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        external_path = directory / "manifest.json"
        scseq_path = directory / "run_manifest.json"
        if external_path.is_file():
            manifest = _read_json(external_path)
            if manifest.get("schema_version") != EXTERNAL_SCHEMA:
                continue
            method_object = manifest.get("method")
            method = (
                str(method_object.get("id"))
                if isinstance(method_object, Mapping)
                else "unknown"
            )
            if method not in METHODS:
                continue
            if manifest.get("status") != "complete":
                excluded.append(
                    {
                        "directory": directory.name,
                        "method": method,
                        "reason": f"run_status_{manifest.get('status')}",
                    }
                )
                continue
            dataset_id = str(manifest["dataset_id"])
            contrast = dataset_id.rsplit("__", 1)[1] if "__" in dataset_id else None
            records.append(
                RunRecord(
                    method=method,
                    dataset_id=dataset_id,
                    contrast=contrast,
                    directory=directory,
                    manifest=manifest,
                    kind="external",
                )
            )
        if scseq_path.is_file():
            manifest = _read_json(scseq_path)
            if manifest.get("schema_version") != SCSEQ_SCHEMA:
                continue
            if manifest.get("status") != "complete":
                excluded.append(
                    {
                        "directory": directory.name,
                        "method": "scseqcommdiff",
                        "reason": f"run_status_{manifest.get('status')}",
                    }
                )
                continue
            protocol = manifest.get("protocol")
            order = (
                protocol.get("contrast_order")
                if isinstance(protocol, Mapping)
                else None
            )
            if not isinstance(order, list) or len(order) != 2:
                raise ValueError(
                    f"scSeqCommDiff contrast order is invalid: {directory}"
                )
            records.append(
                RunRecord(
                    method="scseqcommdiff",
                    dataset_id=str(manifest["dataset_id"]),
                    contrast=f"{order[0]}_vs_{order[1]}",
                    directory=directory,
                    manifest=manifest,
                    kind="scseq",
                )
            )
    return records, excluded


def _external_table(record: RunRecord) -> pd.DataFrame:
    output = record.manifest.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"external output manifest is invalid: {record.directory}")
    path = record.directory / str(output.get("table"))
    if not path.is_file() or sha256_file(path) != output.get("sha256"):
        raise ValueError(f"external output checksum mismatch: {path}")
    table = pd.read_parquet(path)
    if len(table) != int(output.get("rows", -1)):
        raise ValueError(f"external output row count mismatch: {path}")
    method = _require_mapping(
        record.manifest.get("method"), label="external method provenance"
    )
    resource = _require_mapping(
        record.manifest.get("resource"), label="external resource provenance"
    )
    resource_id = resource.get("id", resource.get("resource_id"))
    table_method_version = (
        str(method.get("score_head_version", CRYCHIC_SCORE_HEAD_VERSION))
        if record.method == "crychic"
        else str(method.get("version"))
    )
    identities = {
        "dataset_id": record.dataset_id,
        "method_id": record.method,
        "method_version": table_method_version,
        "resource_id": str(resource_id),
        "resource_mode": "H-common",
    }
    for column, expected in identities.items():
        if column not in table or set(table[column].astype(str)) != {expected}:
            raise ValueError(
                f"external output {column} identity disagrees with its manifest"
            )
    return table


def _target_crychic_view(
    table: pd.DataFrame, manifest: Mapping[str, Any], *, target: str
) -> pd.DataFrame:
    source = manifest.get("source_result")
    views = source.get("score_views") if isinstance(source, Mapping) else None
    if not isinstance(views, list):
        raise ValueError("CRYCHIC score view provenance is absent")
    candidate = f"global:'{target}'"
    selected = [
        str(view["run_id"])
        for view in views
        if isinstance(view, Mapping)
        and candidate in view.get("contrast_candidates", [])
    ]
    if not selected:
        selected = [
            str(view["run_id"])
            for view in views
            if isinstance(view, Mapping)
            and view.get("view_scope")
            == "context_invariant_mechanistic_sample_score"
            and view.get("primary_score") is True
            and view.get("contrast_candidates") == []
        ]
    if len(selected) != 1:
        raise ValueError(
            "expected one CRYCHIC target or context-invariant primary view for "
            f"{candidate}: {selected}"
        )
    result = table.loc[table["run_id"].astype(str).eq(selected[0])].copy()
    if result.empty:
        raise ValueError("selected CRYCHIC target view is empty")
    return result


def _external_effects(
    record: RunRecord, *, contrast: str, target: str, reference: str
) -> pd.DataFrame:
    table = _external_table(record)
    if record.method == "crychic":
        table = _target_crychic_view(table, record.manifest, target=target)
    mapped = external_long_to_score_table(
        table,
        context_key="condition",
        contrast=contrast,
        dataset=record.dataset_id,
    )
    effects = unpaired_edge_effects(
        mapped,
        reference=reference,
        target=target,
        min_subjects=4,
        contrast=contrast,
        validated=True,
    )
    result = effects.loc[:, [*EDGE_KEYS, "effect", "status", "reason_code"]].copy()
    result["p_value"] = np.nan
    result["effect_semantics"] = "independent_subject_comparison_rank_strength"
    return result


def _scseq_effects(record: RunRecord, truth: pd.DataFrame) -> pd.DataFrame:
    outputs = record.manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"scSeqCommDiff outputs are invalid: {record.directory}")
    output = outputs.get("differential_event_scores.tsv.gz")
    if not isinstance(output, Mapping):
        raise ValueError("scSeqCommDiff event output is absent")
    path = record.directory / "differential_event_scores.tsv.gz"
    if not path.is_file() or sha256_file(path) != output.get("sha256"):
        raise ValueError(f"scSeqCommDiff event checksum mismatch: {path}")
    table = pd.read_csv(path, sep="\t")
    result = table.rename(columns={"cluster_L": "sender", "cluster_R": "receiver"})
    interaction_map = truth.loc[
        :, ["ligand", "receptor", "interaction_id"]
    ].drop_duplicates()
    if interaction_map.duplicated(["ligand", "receptor"]).any():
        raise ValueError("truth maps one molecular LR pair to multiple interactions")
    result = result.merge(
        interaction_map,
        on=["ligand", "receptor"],
        how="inner",
        validate="many_to_one",
    )
    result["effect"] = pd.to_numeric(result["effect"], errors="coerce")
    result["p_value"] = pd.to_numeric(result["p_value"], errors="coerce")
    result["effect_semantics"] = "native_S_inter_target_minus_reference"
    return result.loc[
        :,
        [*EDGE_KEYS, "effect", "p_value", "status", "reason_code", "effect_semantics"],
    ]


def _find_record(
    records: Sequence[RunRecord], *, method: str, dataset_id: str, contrast: str
) -> RunRecord | None:
    expected_dataset = dataset_id
    matches = [
        record
        for record in records
        if record.method == method
        and record.dataset_id == expected_dataset
        and (record.contrast in {None, contrast})
    ]
    if len(matches) > 1:
        raise ValueError(
            f"duplicate compatible runs for {method}/{dataset_id}/{contrast}: "
            f"{[record.directory.name for record in matches]}"
        )
    return matches[0] if matches else None


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is absent or invalid")
    return value


def _validate_run_binding(
    record: RunRecord,
    *,
    expected_input_sha256: str,
    expected_resource_sha256: str,
    expected_code_commit: str,
) -> None:
    method = _require_mapping(
        record.manifest.get("method"), label=f"{record.method} method provenance"
    )
    if method.get("id") != record.method:
        raise ValueError(f"method identity mismatch for {record.directory}")
    expected_version = EXPECTED_METHOD_VERSIONS.get(record.method)
    if expected_version is not None and method.get("version") != expected_version:
        raise ValueError(f"method version mismatch for {record.directory}")
    if record.kind == "scseq":
        preflight = _require_mapping(
            record.manifest.get("preflight"), label="scSeqCommDiff preflight"
        )
        input_record = _require_mapping(
            preflight.get("input"), label="scSeqCommDiff input provenance"
        )
        resource = _require_mapping(
            preflight.get("resource"), label="scSeqCommDiff resource provenance"
        )
        resource_mode = record.manifest.get("protocol", {}).get("resource_mode")
    else:
        input_record = _require_mapping(
            record.manifest.get("input"), label=f"{record.method} input provenance"
        )
        resource = _require_mapping(
            record.manifest.get("resource"),
            label=f"{record.method} resource provenance",
        )
        resource_mode = resource.get("mode")
    if input_record.get("sha256") != expected_input_sha256:
        raise ValueError(
            f"input checksum mismatch for {record.method}/{record.dataset_id}"
        )
    resource_sha = resource.get("sha256")
    if resource_sha is None:
        resource_sha = resource.get("table_sha256", resource.get("payload_sha256"))
    if resource_mode != "H-common" or resource_sha != expected_resource_sha256:
        raise ValueError(
            "H-common resource binding mismatch for "
            f"{record.method}/{record.dataset_id}"
        )
    code = _require_mapping(
        record.manifest.get("code"), label=f"{record.method} code provenance"
    )
    if code.get("dirty") is not False or code.get("commit") != expected_code_commit:
        raise ValueError(
            f"code provenance is not the clean frozen commit for "
            f"{record.method}/{record.dataset_id}"
        )
    if record.method == "crychic":
        algorithm_code = _require_mapping(
            method.get("algorithm_code"), label="CRYCHIC algorithm provenance"
        )
        if (
            method.get("benchmark_identity") != "generic_multigroup_baseline"
            or method.get("entrypoint")
            != "benchmarks.adapters.crychic.run_hcommon"
            or algorithm_code.get("dirty") is not False
            or algorithm_code.get("commit") != expected_code_commit
        ):
            raise ValueError("CRYCHIC generic baseline provenance mismatch")
        parameters = _require_mapping(
            record.manifest.get("parameters"), label="CRYCHIC parameters"
        )
        workflow = _require_mapping(
            parameters.get("workflow"), label="CRYCHIC workflow"
        )
        if (
            parameters.get("benchmark_scope")
            != "independent_subject_three_group_hcommon"
            or parameters.get("input_mode") != "counts"
            or parameters.get("counts_layer") != "counts"
            or workflow.get("subject_fixed_effects") is not False
            or workflow.get("min_cells") != 10
        ):
            raise ValueError("CRYCHIC simulation protocol mismatch")
    elif record.method == "cellchat":
        parameters = _require_mapping(
            record.manifest.get("parameters"), label="CellChat parameters"
        )
        if (
            parameters.get("layer") is not None
            or parameters.get("nboot") != 100
            or parameters.get("min_cells") != 10
            or parameters.get("sample_level_execution") is not True
        ):
            raise ValueError("CellChat simulation protocol mismatch")
    elif record.method == "liana_rank_aggregate":
        parameters = _require_mapping(
            record.manifest.get("parameters"), label="LIANA parameters"
        )
        if (
            parameters.get("layer") is not None
            or parameters.get("n_perms_per_sample") != 100
            or parameters.get("min_cells") != 10
            or parameters.get("sample_level_execution") is not True
        ):
            raise ValueError("LIANA simulation protocol mismatch")
    elif record.method == "scseqcommdiff":
        protocol = _require_mapping(
            record.manifest.get("protocol"), label="scSeqCommDiff protocol"
        )
        input_audit = _require_mapping(
            _require_mapping(
                record.manifest.get("preflight"), label="scSeqCommDiff preflight"
            ).get("input"),
            label="scSeqCommDiff input audit",
        )
        if (
            protocol.get("scenario") != "multi-sample"
            or protocol.get("min_cells") != 10
            or protocol.get("resource_mode") != "H-common"
            or input_audit.get("sample_unit_key") != "sample_id"
        ):
            raise ValueError("scSeqCommDiff simulation protocol mismatch")


def _tie_inclusive_top_k(score: pd.Series, k: int) -> pd.Series:
    if k < 1 or score.empty:
        return pd.Series(False, index=score.index)
    ordered = score.sort_values(ascending=False, kind="stable")
    cutoff = float(ordered.iloc[min(k, len(ordered)) - 1])
    return score.ge(cutoff)


def _mcc_from_selection(labels: pd.Series, selected: pd.Series) -> float:
    truth = labels.astype(bool)
    predicted = selected.astype(bool)
    if not truth.index.equals(predicted.index):
        predicted = predicted.reindex(truth.index)
    if predicted.isna().any():
        raise ValueError("MCC selection does not cover the declared truth axis")
    true_positive = int((truth & predicted).sum())
    false_positive = int((~truth & predicted).sum())
    true_negative = int((~truth & ~predicted).sum())
    false_negative = int((truth & ~predicted).sum())
    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    if denominator == 0.0:
        return 0.0
    return float(
        (true_positive * true_negative - false_positive * false_negative)
        / denominator
    )


def _top_effect_recovery(
    truth_effect: pd.Series,
    predicted_effect: pd.Series,
) -> tuple[int, int, int, float]:
    if not truth_effect.index.equals(predicted_effect.index):
        predicted_effect = predicted_effect.reindex(truth_effect.index)
    active = truth_effect.ne(0.0)
    if not active.any():
        return 0, 0, 0, math.nan
    requested = max(1, int(math.ceil(0.10 * int(active.sum()))))
    truth_top = _tie_inclusive_top_k(truth_effect.abs(), requested)
    predicted_top = _tie_inclusive_top_k(predicted_effect.abs(), requested)
    recovered = int((truth_top & predicted_top).sum())
    realized = int(truth_top.sum())
    return requested, realized, int(predicted_top.sum()), float(recovered / realized)


def _active_metrics(table: pd.DataFrame) -> dict[str, Any]:
    observed = (
        table["status"].isin({"exploratory", "observed"}) & table["effect"].notna()
    )
    coverage = float(observed.mean())
    positives = table["truth_label"].eq(1)
    eligible = (
        coverage >= MINIMUM_COVERAGE
        and observed.loc[positives].all()
        and table.loc[observed, "truth_label"].nunique() == 2
    )
    result: dict[str, Any] = {
        "coverage": coverage,
        "n_observed": int(observed.sum()),
        "n_events": len(table),
        "n_positives": int(positives.sum()),
        "metric_status": "observed" if eligible else "NE",
        "reason_code": None
        if eligible
        else "insufficient_event_coverage_or_truth_support",
        "auroc": math.nan,
        "auprc": math.nan,
        "prevalence_adjusted_ap": math.nan,
        "top_k_requested": int(positives.sum()),
        "top_k_realized": 0,
        "top_k_precision": math.nan,
        "top_k_recall": math.nan,
        "direction_accuracy": math.nan,
        "conditional_direction_accuracy": math.nan,
        "positive_direction_ap": math.nan,
        "negative_direction_ap": math.nan,
        "effect_rmse_native_scale": math.nan,
        "effect_mae_native_scale": math.nan,
        "effect_pearson": math.nan,
        "effect_spearman": math.nan,
        "top_effect_k_requested": 0,
        "top_effect_k_truth_realized": 0,
        "top_effect_k_prediction_realized": 0,
        "top_effect_recovery": math.nan,
        "ci_coverage": math.nan,
        "ci_coverage_status": "NE_no_comparable_confidence_intervals",
    }
    if not eligible:
        return result
    available = table.loc[observed].copy()
    available["effect"] = pd.to_numeric(
        available["effect"], errors="raise"
    ).round(METRIC_EFFECT_DECIMALS)
    labels = available["truth_label"].to_numpy(dtype=int)
    magnitude = available["effect"].abs()
    result["auroc"] = tie_aware_auroc(labels, magnitude)
    result["auprc"] = average_precision(labels, magnitude)
    result["prevalence_adjusted_ap"] = prevalence_adjusted_average_precision(
        labels,
        magnitude,
        target_prevalence=TARGET_PREVALENCE,
    )
    selected = _tie_inclusive_top_k(magnitude, int(positives.sum()))
    true_positive = int(available.loc[selected, "truth_label"].sum())
    result["top_k_realized"] = int(selected.sum())
    result["top_k_precision"] = float(true_positive / selected.sum())
    result["top_k_recall"] = float(true_positive / positives.sum())
    truth_positive = table.loc[positives].copy()
    truth_positive["effect"] = pd.to_numeric(
        truth_positive["effect"], errors="raise"
    ).round(METRIC_EFFECT_DECIMALS)
    result["direction_accuracy"] = float(
        np.mean(
            np.sign(truth_positive["effect"].to_numpy(dtype=float))
            == truth_positive["truth_direction"].to_numpy(dtype=int)
        )
    )
    detected_active = available.loc[selected & available["truth_label"].eq(1)]
    if not detected_active.empty:
        result["conditional_direction_accuracy"] = float(
            np.mean(
                np.sign(detected_active["effect"].to_numpy(dtype=float))
                == detected_active["truth_direction"].to_numpy(dtype=int)
            )
        )
    for direction, metric_name, oriented_score in (
        (1, "positive_direction_ap", available["effect"]),
        (-1, "negative_direction_ap", -available["effect"]),
    ):
        directional_labels = available["truth_direction"].eq(direction).astype(int)
        if directional_labels.nunique() == 2:
            result[metric_name] = average_precision(
                directional_labels.to_numpy(dtype=int),
                oriented_score.to_numpy(dtype=float),
            )
    residual = (
        available["effect"].to_numpy(dtype=float)
        - available["truth_effect"].to_numpy(dtype=float)
    )
    result["effect_rmse_native_scale"] = float(np.sqrt(np.mean(residual**2)))
    result["effect_mae_native_scale"] = float(np.mean(np.abs(residual)))
    if available["truth_effect"].nunique() > 1 and available["effect"].nunique() > 1:
        correlation = pearsonr(
            available["truth_effect"], available["effect"]
        ).statistic
        result["effect_pearson"] = (
            float(correlation) if np.isfinite(correlation) else math.nan
        )
    if available["truth_effect"].nunique() > 1 and available["effect"].nunique() > 1:
        rho = spearmanr(available["truth_effect"], available["effect"]).statistic
        result["effect_spearman"] = float(rho) if np.isfinite(rho) else math.nan
    truth_effect = available["truth_effect"].astype(float)
    predicted_effect = available["effect"].astype(float)
    requested, truth_realized, prediction_realized, recovery = (
        _top_effect_recovery(truth_effect, predicted_effect)
    )
    result["top_effect_k_requested"] = requested
    result["top_effect_k_truth_realized"] = truth_realized
    result["top_effect_k_prediction_realized"] = prediction_realized
    result["top_effect_recovery"] = recovery
    return result


def _null_metrics(table: pd.DataFrame) -> dict[str, Any]:
    observed = (
        table["status"].isin({"exploratory", "observed"}) & table["effect"].notna()
    )
    magnitude = (
        pd.to_numeric(table.loc[observed, "effect"], errors="raise")
        .round(METRIC_EFFECT_DECIMALS)
        .abs()
    )
    finite_p = pd.to_numeric(table.loc[observed, "p_value"], errors="coerce").dropna()
    return {
        "coverage": float(observed.mean()),
        "n_observed": int(observed.sum()),
        "n_events": len(table),
        "null_mean_abs_effect": float(magnitude.mean()) if len(magnitude) else math.nan,
        "null_p95_abs_effect": (
            float(magnitude.quantile(0.95)) if len(magnitude) else math.nan
        ),
        "null_max_abs_effect": float(magnitude.max()) if len(magnitude) else math.nan,
        "null_native_p_lt_0_05_fraction": (
            float(finite_p.lt(0.05).mean()) if len(finite_p) else math.nan
        ),
        "null_native_p_rows": len(finite_p),
        "metric_status": "observed" if len(magnitude) else "NE",
        "reason_code": None if len(magnitude) else "no_estimable_null_effects",
    }


def _multigroup_defaults() -> dict[str, Any]:
    return {
        "metric_schema": MULTIGROUP_METRIC_SCHEMA,
        "n_contrasts": 0,
        "n_events": 0,
        "n_event_contrast_cells": 0,
        "n_complete_events": 0,
        "n_observed_event_contrast_cells": 0,
        "event_coverage": math.nan,
        "contrast_cell_coverage": math.nan,
        "omnibus_score_semantics": "max_absolute_pairwise_effect",
        "omnibus_status": "NE",
        "omnibus_reason_code": "not_evaluated",
        "omnibus_auprc": math.nan,
        "omnibus_prevalence_adjusted_ap": math.nan,
        "omnibus_auroc": math.nan,
        "omnibus_mcc_at_truth_k": math.nan,
        "omnibus_k_requested": 0,
        "omnibus_k_realized": 0,
        "omnibus_precision_at_k": math.nan,
        "omnibus_recall_at_k": math.nan,
        "localization_status": "NE",
        "localization_reason_code": "not_evaluated",
        "localization_macro_auprc": math.nan,
        "localization_micro_auprc": math.nan,
        "localization_hamming_loss": math.nan,
        "localization_exact_set_accuracy": math.nan,
        "direction_status": "NE",
        "direction_reason_code": "not_evaluated",
        "positive_direction_ap": math.nan,
        "negative_direction_ap": math.nan,
        "direction_accuracy_all_active": math.nan,
        "direction_accuracy_detected_active": math.nan,
        "effect_status": "NE",
        "effect_reason_code": "not_evaluated",
        "effect_rmse_native_scale": math.nan,
        "effect_mae_native_scale": math.nan,
        "effect_pearson": math.nan,
        "effect_spearman": math.nan,
        "effect_sign_concordance": math.nan,
        "top_effect_k_requested": 0,
        "top_effect_k_truth_realized": 0,
        "top_effect_k_prediction_realized": 0,
        "top_effect_recovery": math.nan,
        "ci_coverage": math.nan,
        "ci_coverage_status": "NE_no_comparable_confidence_intervals",
        "formal_calibration_status": "NE_no_common_full_pipeline_resampling",
        "effect_all_zero_fraction": math.nan,
        "effect_unique_values": 0,
        "effect_standard_deviation": math.nan,
        "effect_iqr": math.nan,
        "effect_dynamic_range": math.nan,
    }


def _event_matrices(
    table: pd.DataFrame,
) -> tuple[
    tuple[str, ...],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required = {
        *EVENT_KEYS,
        "contrast",
        "truth_label",
        "truth_effect",
        "truth_direction",
        "effect",
        "status",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"multigroup event table is missing: {sorted(missing)}")
    if table.duplicated([*EVENT_KEYS, "contrast"]).any():
        raise ValueError("multigroup event table contains duplicate event contrasts")
    contexts = tuple(sorted(table["contrast"].astype(str).unique()))
    if len(contexts) < 2:
        raise ValueError("multigroup metrics require at least two contrasts")
    prepared = table.copy()
    prepared["contrast"] = prepared["contrast"].astype(str)
    prepared["effect"] = pd.to_numeric(
        prepared["effect"], errors="coerce"
    ).round(METRIC_EFFECT_DECIMALS)
    for column in ("truth_label", "truth_effect", "truth_direction"):
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")

    def pivot(column: str) -> pd.DataFrame:
        return (
            prepared.pivot(
                index=list(EVENT_KEYS), columns="contrast", values=column
            )
            .reindex(columns=contexts)
            .sort_index()
        )

    truth_label = pivot("truth_label")
    truth_effect = pivot("truth_effect")
    truth_direction = pivot("truth_direction")
    effect = pivot("effect")
    status = pivot("status")
    if (
        truth_label.isna().any().any()
        or truth_effect.isna().any().any()
        or truth_direction.isna().any().any()
    ):
        raise ValueError("truth must cover every event by contrast cell")
    observed = status.isin({"exploratory", "observed"}) & effect.notna()
    return contexts, truth_label, truth_effect, truth_direction, effect, observed


def _dynamic_range_metrics(
    effect: pd.DataFrame, observed: pd.DataFrame
) -> dict[str, Any]:
    values = effect.where(observed).stack().astype(float)
    if values.empty:
        return {
            "effect_all_zero_fraction": math.nan,
            "effect_unique_values": 0,
            "effect_standard_deviation": math.nan,
            "effect_iqr": math.nan,
            "effect_dynamic_range": math.nan,
        }
    return {
        "effect_all_zero_fraction": float(values.eq(0.0).mean()),
        "effect_unique_values": int(values.nunique()),
        "effect_standard_deviation": float(values.std(ddof=0)),
        "effect_iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "effect_dynamic_range": float(values.max() - values.min()),
    }


def _active_multigroup_metrics(
    table: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    (
        contrasts,
        truth_label,
        truth_effect,
        truth_direction,
        effect,
        observed,
    ) = _event_matrices(table)
    complete = observed.all(axis=1)
    event_truth = truth_label.max(axis=1).astype(int)
    event_coverage = float(complete.mean())
    contrast_coverage = float(observed.to_numpy().mean())
    eligible = bool(
        event_coverage >= MINIMUM_COVERAGE
        and complete.loc[event_truth.eq(1)].all()
        and event_truth.loc[complete].nunique() == 2
    )
    values = _multigroup_defaults()
    values.update(
        {
            "n_contrasts": len(contrasts),
            "n_events": len(event_truth),
            "n_event_contrast_cells": int(truth_label.size),
            "n_complete_events": int(complete.sum()),
            "n_observed_event_contrast_cells": int(observed.to_numpy().sum()),
            "event_coverage": event_coverage,
            "contrast_cell_coverage": contrast_coverage,
            **_dynamic_range_metrics(effect, observed),
        }
    )
    if not eligible:
        reason = "insufficient_complete_event_coverage_or_truth_support"
        for prefix in ("omnibus", "localization", "direction", "effect"):
            values[f"{prefix}_status"] = "NE"
            values[f"{prefix}_reason_code"] = reason
        return values, []

    available_truth = truth_label.loc[complete].astype(int)
    available_effect = effect.loc[complete].astype(float)
    available_direction = truth_direction.loc[complete].astype(int)
    available_truth_effect = truth_effect.loc[complete].astype(float)
    event_scores = available_effect.abs().max(axis=1)
    available_event_truth = event_truth.loc[complete]
    k = int(event_truth.sum())
    selected_events = _tie_inclusive_top_k(event_scores, k)
    true_positive = int(available_event_truth.loc[selected_events].sum())
    values.update(
        {
            "omnibus_status": "observed",
            "omnibus_reason_code": None,
            "omnibus_auprc": average_precision(
                available_event_truth.to_numpy(dtype=int),
                event_scores.to_numpy(dtype=float),
            ),
            "omnibus_prevalence_adjusted_ap": (
                prevalence_adjusted_average_precision(
                    available_event_truth.to_numpy(dtype=int),
                    event_scores.to_numpy(dtype=float),
                    target_prevalence=TARGET_PREVALENCE,
                )
            ),
            "omnibus_auroc": tie_aware_auroc(
                available_event_truth.to_numpy(dtype=int),
                event_scores.to_numpy(dtype=float),
            ),
            "omnibus_mcc_at_truth_k": _mcc_from_selection(
                available_event_truth, selected_events
            ),
            "omnibus_k_requested": k,
            "omnibus_k_realized": int(selected_events.sum()),
            "omnibus_precision_at_k": float(
                true_positive / int(selected_events.sum())
            ),
            "omnibus_recall_at_k": float(true_positive / k),
        }
    )

    selected_contrasts = pd.DataFrame(
        False, index=available_truth.index, columns=available_truth.columns
    )
    macro_ap: list[float] = []
    confusion: list[dict[str, Any]] = []
    for contrast in contrasts:
        labels = available_truth[contrast]
        scores = available_effect[contrast].abs()
        requested = int(truth_label[contrast].sum())
        if labels.nunique() == 2:
            macro_ap.append(
                average_precision(
                    labels.to_numpy(dtype=int), scores.to_numpy(dtype=float)
                )
            )
        selected = (
            _tie_inclusive_top_k(scores, requested)
            if requested > 0
            else pd.Series(False, index=scores.index)
        )
        selected_contrasts.loc[:, contrast] = selected
        truth_boolean = labels.astype(bool)
        predicted_boolean = selected.astype(bool)
        confusion.append(
            {
                "metric_schema": MULTIGROUP_METRIC_SCHEMA,
                "contrast": contrast,
                "k_requested": requested,
                "k_realized": int(predicted_boolean.sum()),
                "true_positive": int((truth_boolean & predicted_boolean).sum()),
                "false_positive": int((~truth_boolean & predicted_boolean).sum()),
                "true_negative": int((~truth_boolean & ~predicted_boolean).sum()),
                "false_negative": int((truth_boolean & ~predicted_boolean).sum()),
                "n_complete_events": len(labels),
                "event_coverage": event_coverage,
                "status": "observed",
                "reason_code": None,
            }
        )
    flat_labels = available_truth.to_numpy(dtype=int).ravel()
    flat_magnitude = available_effect.abs().to_numpy(dtype=float).ravel()
    values.update(
        {
            "localization_status": "observed",
            "localization_reason_code": None,
            "localization_macro_auprc": (
                float(np.mean(macro_ap))
                if len(macro_ap) == len(contrasts)
                else math.nan
            ),
            "localization_micro_auprc": average_precision(
                flat_labels, flat_magnitude
            ),
            "localization_hamming_loss": float(
                np.mean(
                    selected_contrasts.to_numpy(dtype=bool)
                    != available_truth.to_numpy(dtype=bool)
                )
            ),
            "localization_exact_set_accuracy": float(
                np.mean(
                    np.all(
                        selected_contrasts.to_numpy(dtype=bool)
                        == available_truth.to_numpy(dtype=bool),
                        axis=1,
                    )
                )
            ),
        }
    )

    flat_direction = available_direction.to_numpy(dtype=int).ravel()
    flat_effect = available_effect.to_numpy(dtype=float).ravel()
    active = flat_direction != 0
    detected_active = active & selected_contrasts.to_numpy(dtype=bool).ravel()
    positive_labels = (flat_direction == 1).astype(int)
    negative_labels = (flat_direction == -1).astype(int)
    positive_ap = (
        average_precision(positive_labels, flat_effect)
        if np.unique(positive_labels).size == 2
        else math.nan
    )
    negative_ap = (
        average_precision(negative_labels, -flat_effect)
        if np.unique(negative_labels).size == 2
        else math.nan
    )
    values.update(
        {
            "direction_status": "observed",
            "direction_reason_code": None,
            "positive_direction_ap": positive_ap,
            "negative_direction_ap": negative_ap,
            "direction_accuracy_all_active": float(
                np.mean(np.sign(flat_effect[active]) == flat_direction[active])
            ),
            "direction_accuracy_detected_active": (
                float(
                    np.mean(
                        np.sign(flat_effect[detected_active])
                        == flat_direction[detected_active]
                    )
                )
                if detected_active.any()
                else math.nan
            ),
        }
    )

    truth_series = available_truth_effect.stack().astype(float)
    predicted_series = available_effect.stack().astype(float)
    residual = predicted_series.to_numpy() - truth_series.to_numpy()
    pearson = (
        pearsonr(truth_series, predicted_series).statistic
        if truth_series.nunique() > 1 and predicted_series.nunique() > 1
        else math.nan
    )
    spearman = (
        spearmanr(truth_series, predicted_series).statistic
        if truth_series.nunique() > 1 and predicted_series.nunique() > 1
        else math.nan
    )
    requested, truth_realized, prediction_realized, recovery = (
        _top_effect_recovery(truth_series, predicted_series)
    )
    values.update(
        {
            "effect_status": "observed_native_scale_diagnostic",
            "effect_reason_code": None,
            "effect_rmse_native_scale": float(np.sqrt(np.mean(residual**2))),
            "effect_mae_native_scale": float(np.mean(np.abs(residual))),
            "effect_pearson": float(pearson) if np.isfinite(pearson) else math.nan,
            "effect_spearman": (
                float(spearman) if np.isfinite(spearman) else math.nan
            ),
            "effect_sign_concordance": values["direction_accuracy_all_active"],
            "top_effect_k_requested": requested,
            "top_effect_k_truth_realized": truth_realized,
            "top_effect_k_prediction_realized": prediction_realized,
            "top_effect_recovery": recovery,
        }
    )
    return values, confusion


def _null_multigroup_metrics(table: pd.DataFrame) -> dict[str, Any]:
    contrasts, truth_label, _, _, effect, observed = _event_matrices(table)
    if truth_label.to_numpy(dtype=int).any():
        raise ValueError("global-null multigroup table contains active truth")
    complete = observed.all(axis=1)
    values = _multigroup_defaults()
    values.update(
        {
            "n_contrasts": len(contrasts),
            "n_events": len(truth_label),
            "n_event_contrast_cells": int(truth_label.size),
            "n_complete_events": int(complete.sum()),
            "n_observed_event_contrast_cells": int(observed.to_numpy().sum()),
            "event_coverage": float(complete.mean()),
            "contrast_cell_coverage": float(observed.to_numpy().mean()),
            "omnibus_reason_code": "global_null_has_no_detection_truth_class",
            "localization_reason_code": "global_null_has_no_active_contrasts",
            "direction_reason_code": "global_null_has_no_direction_truth",
            "effect_status": "observed_null_diagnostics"
            if observed.to_numpy().any()
            else "NE",
            "effect_reason_code": None
            if observed.to_numpy().any()
            else "no_estimable_null_effects",
            **_dynamic_range_metrics(effect, observed),
        }
    )
    return values


def _multigroup_metrics(
    effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"dataset_id", "scenario", "seed", "method"}
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"event effects are missing identities: {sorted(missing)}")
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    groups = effects.groupby(
        ["dataset_id", "scenario", "seed", "method"],
        observed=True,
        sort=True,
    )
    for (dataset_id, scenario, seed, method), selected in groups:
        if str(scenario) == "active":
            values, confusion = _active_multigroup_metrics(selected)
        elif str(scenario) == "global_null":
            values = _null_multigroup_metrics(selected)
            confusion = []
        else:
            raise ValueError(f"unknown three-group scenario: {scenario}")
        identity = {
            "dataset_id": str(dataset_id),
            "scenario": str(scenario),
            "seed": int(seed),
            "method": str(method),
            "method_label": METHOD_LABELS.get(str(method), str(method)),
            "run_directories": ";".join(
                sorted(selected.get("run_directory", pd.Series(dtype=str)).unique())
            ),
        }
        metric_rows.append({**identity, **values})
        confusion_rows.extend({**identity, **record} for record in confusion)
    return (
        pd.DataFrame.from_records(metric_rows),
        pd.DataFrame.from_records(confusion_rows),
    )


def _method_summary(
    metrics: pd.DataFrame,
    *,
    expected_active_contrasts: int | None = None,
) -> pd.DataFrame:
    active = metrics.loc[
        metrics["scenario"].eq("active") & metrics["metric_status"].eq("observed")
    ]
    expected = (
        len(
            metrics.loc[
                metrics["scenario"].eq("active"), ["dataset_id", "contrast"]
            ].drop_duplicates()
        )
        if expected_active_contrasts is None
        else int(expected_active_contrasts)
    )
    observed = active.groupby("method", observed=True).size()
    summary = (
        active.groupby("method", observed=True)
        .agg(
            mean_prevalence_adjusted_ap=("prevalence_adjusted_ap", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_auroc=("auroc", "mean"),
            mean_top_k_recall=("top_k_recall", "mean"),
            mean_direction_accuracy=("direction_accuracy", "mean"),
            mean_effect_spearman=("effect_spearman", "mean"),
            mean_coverage=("coverage", "mean"),
            active_contrasts_scored=("contrast", "size"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    summary["active_contrasts_expected"] = expected
    summary["active_contrasts_scored"] = (
        summary["method"].map(observed).fillna(0).astype(int)
    )
    summary["rank_eligible"] = (
        summary["active_contrasts_expected"].gt(0)
        & summary["active_contrasts_scored"].eq(summary["active_contrasts_expected"])
        & summary["mean_coverage"].ge(MINIMUM_COVERAGE)
    )
    for metric in (
        "mean_prevalence_adjusted_ap",
        "mean_auprc",
        "mean_auroc",
        "mean_top_k_recall",
        "mean_direction_accuracy",
        "mean_effect_spearman",
    ):
        rank = (
            summary[metric]
            .where(summary["rank_eligible"])
            .rank(ascending=False, method="average")
        )
        summary[f"{metric}_rank"] = rank.astype("Float64")
    summary["primary_rank"] = summary["mean_prevalence_adjusted_ap_rank"]
    summary.insert(1, "method_label", summary["method"].map(METHOD_LABELS))
    return summary.sort_values(
        ["primary_rank", "method_label"], na_position="last", ignore_index=True
    )


def _multigroup_method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    active_all = metrics.loc[metrics["scenario"].eq("active")]
    expected = active_all["dataset_id"].nunique()
    active = active_all.loc[active_all["omnibus_status"].eq("observed")]
    observed = active.groupby("method", observed=True).size()
    summary = (
        active.groupby("method", observed=True)
        .agg(
            mean_omnibus_prevalence_adjusted_ap=(
                "omnibus_prevalence_adjusted_ap",
                "mean",
            ),
            mean_omnibus_auprc=("omnibus_auprc", "mean"),
            mean_omnibus_auroc=("omnibus_auroc", "mean"),
            mean_localization_macro_auprc=("localization_macro_auprc", "mean"),
            mean_localization_micro_auprc=("localization_micro_auprc", "mean"),
            mean_positive_direction_ap=("positive_direction_ap", "mean"),
            mean_negative_direction_ap=("negative_direction_ap", "mean"),
            mean_direction_accuracy_all_active=(
                "direction_accuracy_all_active",
                "mean",
            ),
            mean_effect_rmse_native_scale=("effect_rmse_native_scale", "mean"),
            mean_effect_spearman=("effect_spearman", "mean"),
            mean_event_coverage=("event_coverage", "mean"),
            active_datasets_scored=("dataset_id", "size"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    summary["active_datasets_expected"] = expected
    summary["active_datasets_scored"] = (
        summary["method"].map(observed).fillna(0).astype(int)
    )
    summary["rank_eligible"] = (
        summary["active_datasets_expected"].gt(0)
        & summary["active_datasets_scored"].eq(summary["active_datasets_expected"])
        & summary["mean_event_coverage"].ge(MINIMUM_COVERAGE)
    )
    rank_metrics = (
        "mean_omnibus_prevalence_adjusted_ap",
        "mean_omnibus_auprc",
        "mean_omnibus_auroc",
        "mean_localization_macro_auprc",
        "mean_localization_micro_auprc",
        "mean_positive_direction_ap",
        "mean_negative_direction_ap",
        "mean_direction_accuracy_all_active",
        "mean_effect_spearman",
    )
    for metric in rank_metrics:
        summary[f"{metric}_rank"] = (
            summary[metric]
            .where(summary["rank_eligible"])
            .rank(ascending=False, method="average")
            .astype("Float64")
        )
    summary["primary_rank"] = summary[
        "mean_omnibus_prevalence_adjusted_ap_rank"
    ]
    summary.insert(1, "method_label", summary["method"].map(METHOD_LABELS))
    return summary.sort_values(
        ["primary_rank", "method_label"], na_position="last", ignore_index=True
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _report(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    multigroup_summary: pd.DataFrame | None = None,
) -> str:
    active = metrics.loc[metrics["scenario"].eq("active")]
    n_seeds = int(active["seed"].nunique())
    lines = [
        "# Three-group cell-communication benchmark",
        "",
        (
            f"This result contains {n_seeds} simulation seed(s), three pairwise "
            "contrasts per seed, and a matched global-null control."
        ),
        "",
        (
            "Primary ranking is mean prevalence-adjusted AP at 10% target "
            "prevalence. Native AUPRC is retained as a secondary metric. Missing "
            "or structural absence is never converted to a native zero; methods "
            "below 80% event coverage are not ranked."
        ),
        "",
        (
            "| Rank | Method | Adj. AP | AUPRC | AUROC | Top-k recall | "
            "Direction | Spearman | Coverage |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        rank = "NE" if pd.isna(row.primary_rank) else f"{float(row.primary_rank):g}"
        values = [
            rank,
            str(row.method_label),
            (
                f"{row.mean_prevalence_adjusted_ap:.4f}"
                if pd.notna(row.mean_prevalence_adjusted_ap)
                else "NE"
            ),
            f"{row.mean_auprc:.4f}" if pd.notna(row.mean_auprc) else "NE",
            f"{row.mean_auroc:.4f}" if pd.notna(row.mean_auroc) else "NE",
            f"{row.mean_top_k_recall:.4f}" if pd.notna(row.mean_top_k_recall) else "NE",
            (
                f"{row.mean_direction_accuracy:.4f}"
                if pd.notna(row.mean_direction_accuracy)
                else "NE"
            ),
            (
                f"{row.mean_effect_spearman:.4f}"
                if pd.notna(row.mean_effect_spearman)
                else "NE"
            ),
            f"{row.mean_coverage:.3f}" if pd.notna(row.mean_coverage) else "NE",
        ]
        lines.append("| " + " | ".join(values) + " |")
    if multigroup_summary is not None:
        lines.extend(
            [
                "",
                "## Decomposed multi-group endpoints",
                "",
                (
                    "The event-level omnibus score is the maximum absolute "
                    "pairwise effect. Localization, signed direction, and effect "
                    "estimation are evaluated separately. Native-scale RMSE is "
                    "diagnostic only because method score scales are not common."
                ),
                "",
                (
                    "| Rank | Method | Omnibus adj. AP | Omnibus AUPRC | "
                    "Localization macro AP | AP+ | AP- | Direction | Coverage |"
                ),
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in multigroup_summary.itertuples(index=False):
            rank = "NE" if pd.isna(row.primary_rank) else f"{float(row.primary_rank):g}"

            def display(value: object) -> str:
                return f"{float(value):.4f}" if pd.notna(value) else "NE"

            lines.append(
                "| "
                + " | ".join(
                    (
                        rank,
                        str(row.method_label),
                        display(row.mean_omnibus_prevalence_adjusted_ap),
                        display(row.mean_omnibus_auprc),
                        display(row.mean_localization_macro_auprc),
                        display(row.mean_positive_direction_ap),
                        display(row.mean_negative_direction_ap),
                        display(row.mean_direction_accuracy_all_active),
                        display(row.mean_event_coverage),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            (
                "CRYCHIC, CellChat, and LIANA are fit once to each complete A/B/C "
                "data set. CellChat and LIANA remain within-sample methods; all "
                "three are mapped to the same independent-subject mean-contrast "
                "operator. scSeqCommDiff uses its native two-condition multi-sample "
                "arm for each frozen pairwise contrast."
            ),
            "",
            (
                "AUROC/AUPRC are not defined for the global-null rows. Null "
                "outputs report effect magnitudes, and only scSeqCommDiff has a "
                "native between-condition p-value."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(
    fixture_dir: Path,
    runs_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    fixture_dir = fixture_dir.resolve()
    runs_dir = runs_dir.resolve()
    output_dir = output_dir.resolve()
    fixture_manifest = _read_json(fixture_dir / "manifest.json")
    if (
        fixture_manifest.get("schema_version") != "crychic-three-group-fixture-v2"
        or fixture_manifest.get("design") != "independent_subject_groups"
    ):
        raise ValueError("evaluation requires the independent-group v2 fixture")
    code_record = _require_mapping(
        fixture_manifest.get("code"), label="fixture code provenance"
    )
    expected_code_commit = str(code_record.get("commit", ""))
    if code_record.get("dirty") is not False or not expected_code_commit:
        raise ValueError("fixture must be generated from a clean frozen commit")
    evaluator_code = git_metadata(Path(__file__).resolve().parents[2])
    if evaluator_code.get("dirty") is not False or not evaluator_code.get("commit"):
        raise ValueError("evaluator must run from a clean committed revision")
    resource_record = _require_mapping(
        fixture_manifest.get("resource"), label="fixture resource provenance"
    )
    expected_resource_sha256 = str(resource_record.get("sha256", ""))
    fixture_records_object = fixture_manifest.get("records")
    if not isinstance(fixture_records_object, list):
        raise ValueError("fixture records are absent")
    fixture_records = {
        str(record["dataset_id"]): record
        for record in fixture_records_object
        if isinstance(record, Mapping)
    }
    truth_path = fixture_dir / str(fixture_manifest["outputs"]["truth"]["filename"])
    if sha256_file(truth_path) != fixture_manifest["outputs"]["truth"]["sha256"]:
        raise ValueError("fixture truth checksum mismatch")
    truth = pd.read_csv(truth_path, sep="\t")
    records, excluded = _discover_runs(runs_dir)
    effect_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    missing_runs: list[dict[str, str]] = []
    truth_groups = truth.groupby(["dataset_id", "contrast"], observed=True, sort=True)
    for (dataset_id, contrast), selected_truth in truth_groups:
        target = str(selected_truth["target"].iloc[0])
        reference = str(selected_truth["reference"].iloc[0])
        scenario = str(selected_truth["scenario"].iloc[0])
        seed = int(selected_truth["seed"].iloc[0])
        for method in METHODS:
            record = _find_record(
                records,
                method=method,
                dataset_id=str(dataset_id),
                contrast=str(contrast),
            )
            if record is None:
                missing_runs.append(
                    {
                        "dataset_id": str(dataset_id),
                        "contrast": str(contrast),
                        "method": method,
                    }
                )
                continue
            fixture_record = _require_mapping(
                fixture_records.get(str(dataset_id)), label="fixture dataset record"
            )
            if method == "scseqcommdiff":
                pair_records = fixture_record.get("pairs")
                if not isinstance(pair_records, list):
                    raise ValueError("fixture pair records are absent")
                pair_matches = [
                    pair
                    for pair in pair_records
                    if isinstance(pair, Mapping) and pair.get("contrast") == contrast
                ]
                if len(pair_matches) != 1:
                    raise ValueError("fixture pair record is missing or duplicated")
                expected_manifest_path = fixture_dir / str(pair_matches[0]["manifest"])
            else:
                expected_manifest_path = fixture_dir / str(fixture_record["manifest"])
            expected_manifest = _read_json(expected_manifest_path)
            expected_input = _require_mapping(
                expected_manifest.get("output"), label="fixture input checksum"
            )
            _validate_run_binding(
                record,
                expected_input_sha256=str(expected_input["sha256"]),
                expected_resource_sha256=expected_resource_sha256,
                expected_code_commit=expected_code_commit,
            )
            effects = (
                _scseq_effects(record, selected_truth)
                if method == "scseqcommdiff"
                else _external_effects(
                    record,
                    contrast=str(contrast),
                    target=target,
                    reference=reference,
                )
            )
            if effects.duplicated(list(EVENT_KEYS)).any():
                raise ValueError(
                    f"duplicate event effects for {method}/{dataset_id}/{contrast}"
                )
            effects["effect"] = pd.to_numeric(
                effects["effect"], errors="coerce"
            ).round(METRIC_EFFECT_DECIMALS)
            merged = selected_truth.merge(
                effects,
                on=list(EDGE_KEYS),
                how="left",
                validate="one_to_one",
            )
            merged["method"] = method
            merged["method_label"] = METHOD_LABELS[method]
            merged["run_directory"] = record.directory.name
            missing_event = merged["status"].isna()
            merged["status"] = merged["status"].fillna("not_estimable")
            merged["reason_code"] = merged["reason_code"].astype("string")
            merged.loc[missing_event, "reason_code"] = "event_not_returned"
            merged.loc[~missing_event & merged["reason_code"].isna(), "reason_code"] = (
                ""
            )
            effect_frames.append(merged)
            values = (
                _active_metrics(merged)
                if scenario == "active"
                else _null_metrics(merged)
            )
            values.update(
                {
                    "dataset_id": str(dataset_id),
                    "scenario": scenario,
                    "seed": seed,
                    "contrast": str(contrast),
                    "target": target,
                    "reference": reference,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "run_directory": record.directory.name,
                }
            )
            metric_rows.append(values)

    effects = (
        pd.concat(effect_frames, ignore_index=True) if effect_frames else pd.DataFrame()
    )
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("no compatible benchmark runs were discovered")
    expected_active_contrasts = len(
        truth.loc[
            truth["scenario"].eq("active"), ["dataset_id", "contrast"]
        ].drop_duplicates()
    )
    summary = _method_summary(
        metrics,
        expected_active_contrasts=expected_active_contrasts,
    )
    multigroup_metrics, contrast_confusion = _multigroup_metrics(effects)
    multigroup_summary = _multigroup_method_summary(multigroup_metrics)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        effects_path = staged / "event_effects.tsv.gz"
        metrics_path = staged / "per_contrast_metrics.tsv"
        summary_path = staged / "method_summary.tsv"
        multigroup_metrics_path = staged / "multigroup_metrics.tsv"
        multigroup_summary_path = staged / "multigroup_method_summary.tsv"
        confusion_path = staged / "contrast_confusion.tsv"
        effects.to_csv(effects_path, sep="\t", index=False)
        metrics.to_csv(metrics_path, sep="\t", index=False)
        summary.to_csv(summary_path, sep="\t", index=False)
        multigroup_metrics.to_csv(multigroup_metrics_path, sep="\t", index=False)
        multigroup_summary.to_csv(multigroup_summary_path, sep="\t", index=False)
        contrast_confusion.to_csv(confusion_path, sep="\t", index=False)
        (staged / "REPORT.md").write_text(
            _report(
                summary,
                metrics,
                multigroup_summary=multigroup_summary,
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "crychic-three-group-evaluation-v2",
            "status": "complete",
            "fixture_manifest_sha256": sha256_file(fixture_dir / "manifest.json"),
            "truth_sha256": sha256_file(truth_path),
            "minimum_rank_coverage": MINIMUM_COVERAGE,
            "prevalence_adjusted_ap_target_prevalence": TARGET_PREVALENCE,
            "metric_effect_decimal_places": METRIC_EFFECT_DECIMALS,
            "multigroup_metric_schema": MULTIGROUP_METRIC_SCHEMA,
            "omnibus_score_semantics": "max_absolute_pairwise_effect",
            "localization_threshold": "truth_cardinality_tie_inclusive_top_k",
            "effect_rmse_ranking_policy": (
                "native_scale_diagnostic_only_not_cross_method_ranked"
            ),
            "confidence_interval_policy": (
                "NE_without_common_full_pipeline_resampling_intervals"
            ),
            "design": "independent_subject_groups",
            "resource_mode": "H-common",
            "frozen_code_commit": expected_code_commit,
            "method_output_code_commit": expected_code_commit,
            "evaluator_code": evaluator_code,
            "evaluator_script_sha256": sha256_file(Path(__file__)),
            "methods_requested": list(METHODS),
            "runs_discovered": len(records),
            "excluded_runs": excluded,
            "missing_runs": missing_runs,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in (
                    effects_path,
                    metrics_path,
                    summary_path,
                    multigroup_metrics_path,
                    multigroup_summary_path,
                    confusion_path,
                    staged / "REPORT.md",
                )
            },
        }
        _write_json(staged / "manifest.json", manifest)
        if output_dir.exists() and not overwrite:
            raise FileExistsError(f"output exists: {output_dir}; pass --overwrite")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staged, output_dir)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = evaluate(
        args.fixture_dir,
        args.runs_dir,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
