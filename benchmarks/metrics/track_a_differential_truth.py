"""Track-A paired differential truth metrics for frozen synthetic LR edges.

The primary estimand is the paired target-minus-reference difference in each
method's direction-oriented native score. Native scales are not compared across
methods. A direction-normalized rank-strength difference is emitted as a
sensitivity estimand. Adapter omissions are never imputed as native-score zero;
``not_returned`` enters only the rank-strength sensitivity as the contractually
defined bottom rank.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json
from benchmarks.metrics.multicondition import (
    EDGE_KEYS,
    SYNTHETIC_TRUTH_SCOPES,
    _binary_rank_metrics,
    external_long_to_score_table,
)

TRACK_A_SUITE_SCHEMA = "crychic-track-a-differential-suite-v1"
TRACK_A_OUTPUT_SCHEMA = "crychic-track-a-differential-metrics-v1"
ANALYSIS_TRACK = "lr_stlr"
PRIMARY_ESTIMAND = "paired_oriented_native_score_difference"
SENSITIVITY_ESTIMAND = "paired_rank_strength_difference"
NEGATIVE_CONTROL_DIAGNOSTIC = (
    "preregistered_known_edge_diagnostic_not_significance"
)
SINGLE_CLASS_REASON = "truth_has_single_class_no_positive_integrated_edge"
EXPECTED_METHOD_IDS = {
    "crychic": "crychic",
    "cellchat": "cellchat",
    "cellphonedb": "cellphonedb",
    "liana": "liana_rank_aggregate",
}
FINALIZER_REQUIRED_COLUMNS = {
    "truth_scope",
    "method",
    "metric",
    "estimate",
    "status",
    "reason_code",
}


@dataclass(frozen=True)
class ScenarioSpec:
    """One frozen synthetic scenario and its expected dataset identifier."""

    scenario: str
    dataset: str


@dataclass(frozen=True)
class SuiteSpec:
    """Resolved Track-A differential truth suite configuration."""

    suite_root: Path
    truth_path: Path
    track_b_records_path: Path
    context_key: str
    reference: str
    target: str
    contrast: str
    min_pairs: int
    top_k: int
    truth_universe_size: int
    positive_scenario: str
    crychic_primary_contrast_candidate: str
    known_edge: Mapping[str, str]
    methods: tuple[str, ...]
    scenarios: tuple[ScenarioSpec, ...]


@dataclass(frozen=True)
class PreparedEstimand:
    """Subject-context edge values for one differential estimand."""

    estimand: str
    subject_context: pd.DataFrame
    missing_policy: str
    scale_comparability: str


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _resolve(path: str, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return cast(dict[str, Any], value)


def _constant(table: pd.DataFrame, column: str) -> str:
    values = table[column].astype("string").dropna().astype(str).drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"{column} must be constant per selected Track-A run")
    return str(values.iloc[0])


def _load_specification(path: Path, repo_root: Path) -> SuiteSpec:
    raw = _json_object(path)
    if raw.get("schema_version") != TRACK_A_SUITE_SCHEMA:
        raise ValueError("unsupported Track-A differential suite specification")
    methods_raw = raw.get("methods")
    if not isinstance(methods_raw, list) or not methods_raw:
        raise ValueError("Track-A specification requires a non-empty methods list")
    methods = tuple(str(value) for value in methods_raw)
    if len(set(methods)) != len(methods):
        raise ValueError("Track-A methods must be unique")
    invalid_methods = set(methods).difference(EXPECTED_METHOD_IDS)
    if invalid_methods:
        raise ValueError(f"unsupported Track-A methods: {sorted(invalid_methods)}")

    scenarios_raw = raw.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("Track-A specification requires synthetic scenarios")
    scenarios: list[ScenarioSpec] = []
    for value in scenarios_raw:
        if not isinstance(value, dict):
            raise ValueError("Track-A scenario specifications must be mappings")
        scenarios.append(
            ScenarioSpec(scenario=str(value["scenario"]), dataset=str(value["dataset"]))
        )
    if len({item.scenario for item in scenarios}) != len(scenarios):
        raise ValueError("Track-A scenario names must be unique")
    if len({item.dataset for item in scenarios}) != len(scenarios):
        raise ValueError("Track-A scenario datasets must be unique")

    known_raw = raw.get("known_edge")
    if not isinstance(known_raw, dict):
        raise ValueError("Track-A specification requires known_edge")
    known_edge = {key: str(known_raw[key]) for key in EDGE_KEYS if key in known_raw}
    missing_known = set(EDGE_KEYS).difference(known_edge)
    if missing_known:
        raise ValueError(f"known_edge is missing keys: {sorted(missing_known)}")

    min_pairs = int(raw.get("min_pairs", 4))
    top_k = int(raw.get("top_k", 5))
    truth_universe_size = int(raw.get("truth_universe_size", 45))
    if min_pairs < 1 or top_k < 1 or truth_universe_size < 2:
        raise ValueError("min_pairs/top_k must be positive and truth universe >= 2")
    positive_scenario = str(raw.get("positive_scenario", "active"))
    if positive_scenario not in {item.scenario for item in scenarios}:
        raise ValueError("positive_scenario is absent from scenarios")
    reference = str(raw.get("reference", ""))
    target = str(raw.get("target", ""))
    if not reference or not target or reference == target:
        raise ValueError("Track-A reference and target must be distinct and non-empty")
    return SuiteSpec(
        suite_root=_resolve(str(raw["suite_root"]), repo_root),
        truth_path=_resolve(str(raw["track_a_truth"]), repo_root),
        track_b_records_path=_resolve(
            str(raw["track_b_simulation_records"]), repo_root
        ),
        context_key=str(raw.get("context_key", "condition")),
        reference=reference,
        target=target,
        contrast=str(raw.get("contrast", f"{target}_vs_{reference}")),
        min_pairs=min_pairs,
        top_k=top_k,
        truth_universe_size=truth_universe_size,
        positive_scenario=positive_scenario,
        crychic_primary_contrast_candidate=str(
            raw.get("crychic_primary_contrast_candidate", "global:'stim'")
        ),
        known_edge=known_edge,
        methods=methods,
        scenarios=tuple(scenarios),
    )


def select_crychic_run_id(
    manifest: Mapping[str, object], *, contrast_candidate: str
) -> str:
    """Select exactly one pre-registered CRYCHIC score view from provenance."""

    source_result = manifest.get("source_result")
    if not isinstance(source_result, dict):
        raise ValueError("CRYCHIC manifest lacks source_result provenance")
    score_views = source_result.get("score_views")
    if not isinstance(score_views, list):
        raise ValueError("CRYCHIC manifest lacks score_views provenance")
    selected: list[str] = []
    for raw_view in score_views:
        if not isinstance(raw_view, dict):
            continue
        candidates = raw_view.get("contrast_candidates")
        run_id = raw_view.get("run_id")
        if (
            isinstance(candidates, list)
            and contrast_candidate in {str(value) for value in candidates}
            and isinstance(run_id, str)
            and run_id
        ):
            selected.append(run_id)
    if len(selected) != 1:
        raise ValueError(
            "CRYCHIC primary contrast candidate must identify exactly one score view; "
            f"candidate={contrast_candidate!r}, matches={selected}"
        )
    return selected[0]


def _validate_input_artifact(
    path: Path,
    *,
    method: str,
    crychic_primary_contrast_candidate: str,
) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Track-A input is missing: {path}")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Track-A input manifest is missing: {manifest_path}")
    manifest = _json_object(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"Track-A input manifest is not complete: {manifest_path}")
    output = manifest.get("output")
    if not isinstance(output, dict) or output.get("table") != path.name:
        raise ValueError("Track-A manifest output does not name the selected table")
    observed_sha = sha256_file(path)
    if output.get("sha256") != observed_sha:
        raise ValueError("Track-A table checksum disagrees with its manifest")
    method_metadata = manifest.get("method")
    if (
        not isinstance(method_metadata, dict)
        or str(method_metadata.get("id")) != EXPECTED_METHOD_IDS[method]
    ):
        raise ValueError(f"Track-A manifest method disagrees with {method}")
    selected_run_id = (
        select_crychic_run_id(
            manifest, contrast_candidate=crychic_primary_contrast_candidate
        )
        if method == "crychic"
        else None
    )
    return (
        {
            "table": str(path),
            "table_sha256": observed_sha,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        selected_run_id,
    )


def _load_truth(spec: SuiteSpec) -> pd.DataFrame:
    if not spec.truth_path.is_file():
        raise FileNotFoundError(f"Track-A truth table is missing: {spec.truth_path}")
    truth = pd.read_csv(spec.truth_path, sep="\t")
    required = {
        "dataset",
        "contrast",
        "universe_id",
        *EDGE_KEYS,
        "is_positive",
        "truth_scope",
        "truth_status",
        "reason_code",
    }
    missing = required.difference(truth.columns)
    if missing:
        raise ValueError(f"Track-A truth table is missing columns: {sorted(missing)}")
    key = ["dataset", "contrast", "universe_id", *EDGE_KEYS]
    if truth.duplicated(key).any():
        raise ValueError("Track-A truth table contains duplicate frozen edges")
    labels = pd.to_numeric(truth["is_positive"], errors="coerce")
    if (
        labels.isna().any()
        or (labels % 1 != 0).any()
        or not set(labels.astype(int)).issubset({0, 1})
    ):
        raise ValueError("Track-A is_positive must contain only binary 0/1 labels")
    truth = truth.copy()
    truth["is_positive"] = labels.astype(int)
    invalid_scopes = set(truth["truth_scope"].astype(str)).difference(
        SYNTHETIC_TRUTH_SCOPES
    )
    if invalid_scopes:
        raise ValueError(f"invalid Track-A truth scopes: {sorted(invalid_scopes)}")

    configured_datasets = {item.dataset for item in spec.scenarios}
    if set(truth["dataset"].astype(str)) != configured_datasets:
        raise ValueError("Track-A truth datasets disagree with configured scenarios")
    known_key = tuple(spec.known_edge[key] for key in EDGE_KEYS)
    reference_edges: set[tuple[object, ...]] | None = None
    for scenario in spec.scenarios:
        selected = truth.loc[truth["dataset"].astype(str).eq(scenario.dataset)]
        if set(selected["contrast"].astype(str)) != {spec.contrast}:
            raise ValueError(f"truth contrast mismatch for {scenario.scenario}")
        if len(selected) != spec.truth_universe_size:
            raise ValueError(
                f"truth universe for {scenario.scenario} must contain exactly "
                f"{spec.truth_universe_size} edges"
            )
        edge_set = set(selected.loc[:, EDGE_KEYS].itertuples(index=False, name=None))
        if reference_edges is None:
            reference_edges = edge_set
        elif edge_set != reference_edges:
            raise ValueError("every Track-A scenario must use the same frozen edges")
        if known_key not in edge_set:
            raise ValueError(f"known edge is absent from {scenario.scenario} truth")
        positive_edges = set(
            selected.loc[selected["is_positive"].eq(1), EDGE_KEYS].itertuples(
                index=False, name=None
            )
        )
        if scenario.scenario == spec.positive_scenario:
            if positive_edges != {known_key}:
                raise ValueError("positive scenario must label only the known edge")
            if set(selected["truth_status"].astype(str)) != {"estimable"}:
                raise ValueError("positive scenario truth_status must be estimable")
        elif positive_edges:
            raise ValueError("negative-control truth must remain all-zero/single-class")
        else:
            if set(selected["truth_status"].astype(str)) != {"not_estimable"}:
                raise ValueError(
                    "negative-control truth_status must be not_estimable"
                )
            reasons = set(selected["reason_code"].astype(str))
            if reasons != {SINGLE_CLASS_REASON}:
                raise ValueError(
                    "negative-control truth must carry the single-class reason"
                )
    return truth


def prepare_estimands(
    external_table: pd.DataFrame,
    *,
    context_key: str,
    contrast: str,
) -> tuple[pd.DataFrame, tuple[PreparedEstimand, PreparedEstimand]]:
    """Validate an adapter table and prepare primary plus sensitivity values."""

    score_table = external_long_to_score_table(
        external_table, context_key=context_key, contrast=contrast
    )
    direction = _constant(score_table, "score_direction")
    if direction not in {"higher", "lower"}:
        raise ValueError("Track-A score direction must be higher or lower")

    observed = score_table.loc[score_table["status"].eq("observed")].copy()
    observed["value"] = pd.to_numeric(observed["score"], errors="coerce")
    if direction == "lower":
        observed["value"] = -observed["value"]
    native_subject_context = (
        observed.groupby(
            ["subject_id", "context", *EDGE_KEYS],
            observed=True,
            sort=False,
            dropna=False,
        )["value"]
        .mean()
        .reset_index()
    )

    comparable = score_table.loc[score_table["comparison_eligible"]].copy()
    comparable["value"] = pd.to_numeric(
        comparable["comparison_strength"], errors="coerce"
    )
    rank_subject_context = (
        comparable.groupby(
            ["subject_id", "context", *EDGE_KEYS],
            observed=True,
            sort=False,
            dropna=False,
        )["value"]
        .mean()
        .reset_index()
    )
    return (
        score_table,
        (
            PreparedEstimand(
                estimand=PRIMARY_ESTIMAND,
                subject_context=native_subject_context,
                missing_policy="paired_observed_native_scores_only_no_zero_imputation",
                scale_comparability="native_score_scale_not_cross_method_comparable",
            ),
            PreparedEstimand(
                estimand=SENSITIVITY_ESTIMAND,
                subject_context=rank_subject_context,
                missing_policy="not_predicted_bottom_rank_failures_remain_missing",
                scale_comparability="within_sample_direction_normalized_rank_strength",
            ),
        ),
    )


def paired_edge_differences(
    subject_context: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    reference: str,
    target: str,
    min_pairs: int,
    estimand: str,
) -> pd.DataFrame:
    """Compute subject-paired target-minus-reference effects over frozen edges."""

    required = {"subject_id", "context", *EDGE_KEYS, "value"}
    missing = required.difference(subject_context.columns)
    if missing:
        raise ValueError(f"subject-context values are missing: {sorted(missing)}")
    truth_required = {*EDGE_KEYS, "is_positive", "truth_scope"}
    missing_truth = truth_required.difference(truth.columns)
    if missing_truth:
        raise ValueError(f"edge truth is missing columns: {sorted(missing_truth)}")

    reference_rows = subject_context.loc[
        subject_context["context"].astype(str).eq(reference),
        ["subject_id", *EDGE_KEYS, "value"],
    ].rename(columns={"value": "reference_value"})
    target_rows = subject_context.loc[
        subject_context["context"].astype(str).eq(target),
        ["subject_id", *EDGE_KEYS, "value"],
    ].rename(columns={"value": "target_value"})
    paired = reference_rows.merge(
        target_rows,
        on=["subject_id", *EDGE_KEYS],
        validate="one_to_one",
    )
    paired["difference"] = paired["target_value"] - paired["reference_value"]
    grouped = paired.groupby(list(EDGE_KEYS), observed=True, sort=False)
    effects = grouped.agg(
        reference_mean=("reference_value", "mean"),
        target_mean=("target_value", "mean"),
        effect=("difference", "mean"),
        median_effect=("difference", "median"),
        n_pairs=("difference", "size"),
        n_positive_pairs=("difference", lambda values: int((values > 0).sum())),
        n_negative_pairs=("difference", lambda values: int((values < 0).sum())),
        n_zero_pairs=("difference", lambda values: int((values == 0).sum())),
    ).reset_index()
    effects["positive_direction_fraction"] = (
        effects["n_positive_pairs"] / effects["n_pairs"]
    )
    result = truth.loc[:, [*EDGE_KEYS, "is_positive", "truth_scope"]].merge(
        effects, on=list(EDGE_KEYS), how="left", validate="one_to_one"
    )
    count_columns = [
        "n_pairs",
        "n_positive_pairs",
        "n_negative_pairs",
        "n_zero_pairs",
    ]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    estimable = result["n_pairs"].ge(min_pairs)
    result["status"] = np.where(estimable, "observed", "not_estimable")
    insufficient_reason = (
        "insufficient_paired_observed_native_scores"
        if estimand == PRIMARY_ESTIMAND
        else "insufficient_paired_comparison_scores"
    )
    result["reason_code"] = np.where(estimable, None, insufficient_reason)
    effect_columns = [
        "reference_mean",
        "target_mean",
        "effect",
        "median_effect",
        "positive_direction_fraction",
    ]
    result.loc[~estimable, effect_columns] = np.nan
    result["effect_rank"] = np.nan
    result["effect_percentile"] = np.nan
    if estimable.any():
        ranks = result.loc[estimable, "effect"].rank(
            ascending=False, method="average"
        )
        result.loc[estimable, "effect_rank"] = ranks
        n_estimable = int(estimable.sum())
        result.loc[estimable, "effect_percentile"] = (
            1.0
            if n_estimable == 1
            else 1.0 - (ranks - 1.0) / (n_estimable - 1.0)
        )
    result["estimand"] = estimand
    return result


def summarize_differential_truth(
    edges: pd.DataFrame,
    *,
    top_k: int,
    known_edge: Mapping[str, str],
) -> dict[str, object]:
    """Summarize classification truth and the pre-registered known edge."""

    if len(edges) == 0:
        raise ValueError("Track-A differential edge table is empty")
    labels = pd.to_numeric(edges["is_positive"], errors="raise").astype(int)
    n_truth_positive = int(labels.sum())
    n_truth_negative = int(len(labels) - n_truth_positive)
    estimable = edges.loc[edges["status"].eq("observed")].copy()
    n_estimable_positive = int(estimable["is_positive"].eq(1).sum())
    n_estimable_negative = int(estimable["is_positive"].eq(0).sum())

    known_mask = pd.Series(True, index=edges.index)
    for key in EDGE_KEYS:
        known_mask &= edges[key].astype(str).eq(str(known_edge[key]))
    known = edges.loc[known_mask]
    if len(known) != 1:
        raise ValueError("known edge must match exactly one frozen truth row")
    known_row = known.iloc[0]

    single_class = n_truth_positive == 0 or n_truth_negative == 0
    if single_class:
        status = "not_estimable"
        reason_code = SINGLE_CLASS_REASON
    elif n_estimable_positive == 0:
        status = "not_estimable"
        reason_code = "known_positive_edge_not_estimable"
    elif n_estimable_negative == 0:
        status = "not_estimable"
        reason_code = "insufficient_estimable_truth_classes"
    else:
        status = "observed"
        reason_code = None

    auroc = math.nan
    average_precision = math.nan
    precision = math.nan
    recall = math.nan
    f1 = math.nan
    selected_count = 0
    if status == "observed":
        y_true = estimable["is_positive"].to_numpy(dtype=int)
        y_score = estimable["effect"].to_numpy(dtype=float)
        auroc, average_precision = _binary_rank_metrics(y_true, y_score)
        ordered_ranks = estimable["effect_rank"].sort_values(kind="stable")
        cutoff = float(ordered_ranks.iloc[min(top_k, len(ordered_ranks)) - 1])
        selected = estimable.loc[estimable["effect_rank"].le(cutoff)]
        selected_count = len(selected)
        true_positive = int(selected["is_positive"].eq(1).sum())
        precision = true_positive / selected_count if selected_count else math.nan
        recall = true_positive / n_estimable_positive
        if np.isfinite(precision):
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

    known_estimable = str(known_row["status"]) == "observed"
    return {
        "truth_universe_edges": len(edges),
        "truth_positive_edges": n_truth_positive,
        "truth_negative_edges": n_truth_negative,
        "estimable_edges": len(estimable),
        "estimable_positive_edges": n_estimable_positive,
        "estimable_negative_edges": n_estimable_negative,
        "estimable_edge_coverage_fraction": len(estimable) / len(edges),
        "auroc": auroc,
        "average_precision": average_precision,
        "top_k_requested": top_k,
        "top_k_realized_tie_inclusive": selected_count,
        "top_k_precision": precision,
        "top_k_recall": recall,
        "top_k_f1": f1,
        "status": status,
        "reason_code": reason_code,
        "known_edge_effect": (
            float(known_row["effect"]) if known_estimable else math.nan
        ),
        "known_edge_reference_mean": (
            float(known_row["reference_mean"]) if known_estimable else math.nan
        ),
        "known_edge_target_mean": (
            float(known_row["target_mean"]) if known_estimable else math.nan
        ),
        "known_edge_effect_rank": (
            float(known_row["effect_rank"]) if known_estimable else math.nan
        ),
        "known_edge_effect_percentile": (
            float(known_row["effect_percentile"]) if known_estimable else math.nan
        ),
        "known_edge_positive_direction_fraction": (
            float(known_row["positive_direction_fraction"])
            if known_estimable
            else math.nan
        ),
        "known_edge_n_pairs": int(known_row["n_pairs"]),
        "known_edge_status": str(known_row["status"]),
        "known_edge_reason_code": (
            None
            if known_estimable
            else str(known_row["reason_code"])
            if pd.notna(known_row["reason_code"])
            else "known_edge_not_estimable"
        ),
    }


def simulation_truth_records(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert Track-A summaries to finalizer-compatible simulation records."""

    classification_metrics = {
        "differential_auroc": "auroc",
        "differential_average_precision": "average_precision",
        "differential_top_k_precision": "top_k_precision",
        "differential_top_k_recall": "top_k_recall",
        "differential_top_k_f1": "top_k_f1",
    }
    known_metrics = {
        "known_cxcl10_effect": "known_edge_effect",
        "known_cxcl10_reference_mean": "known_edge_reference_mean",
        "known_cxcl10_target_mean": "known_edge_target_mean",
        "known_cxcl10_effect_rank": "known_edge_effect_rank",
        "known_cxcl10_effect_percentile": "known_edge_effect_percentile",
        "known_cxcl10_positive_direction_fraction": (
            "known_edge_positive_direction_fraction"
        ),
        "known_cxcl10_n_pairs": "known_edge_n_pairs",
    }
    rows: list[dict[str, object]] = []
    identity_columns = [
        "dataset",
        "scenario",
        "method",
        "method_version",
        "analysis_track",
        "resource",
        "resource_version",
        "resource_mode",
        "score_semantics",
        "score_direction",
        "universe_id",
        "estimand",
        "missing_policy",
        "scale_comparability",
    ]
    for _, row in summary.iterrows():
        identity = {column: row[column] for column in identity_columns}
        identity.update(
            {
                "truth_scope": "simulation",
                "benchmark_track": "track_a_differential_lr_truth",
            }
        )
        for metric, column in classification_metrics.items():
            value = row[column]
            rows.append(
                identity
                | {
                    "metric": metric,
                    "estimate": float(value) if pd.notna(value) else math.nan,
                    "status": str(row["status"]),
                    "reason_code": (
                        None
                        if str(row["status"]) == "observed"
                        else str(row["reason_code"])
                    ),
                }
            )
        for metric, column in known_metrics.items():
            value = row[column]
            known_observed = str(row["known_edge_status"]) == "observed"
            reason = (
                NEGATIVE_CONTROL_DIAGNOSTIC
                if known_observed and int(row["truth_positive_edges"]) == 0
                else None
                if known_observed
                else str(row["known_edge_reason_code"])
            )
            rows.append(
                identity
                | {
                    "metric": metric,
                    "estimate": float(value) if pd.notna(value) else math.nan,
                    "status": "observed" if known_observed else "not_estimable",
                    "reason_code": reason,
                }
            )
        rows.append(
            identity
            | {
                "metric": "estimable_edge_coverage_fraction",
                "estimate": float(row["estimable_edge_coverage_fraction"]),
                "status": "observed",
                "reason_code": None,
            }
        )
    return pd.DataFrame(rows)


def merge_simulation_records(
    track_a_records: pd.DataFrame, track_b_records: pd.DataFrame
) -> pd.DataFrame:
    """Validate and combine Track-A and Track-B finalizer simulation records."""

    for label, table in (("Track A", track_a_records), ("Track B", track_b_records)):
        missing = FINALIZER_REQUIRED_COLUMNS.difference(table.columns)
        if missing:
            raise ValueError(f"{label} records are missing columns: {sorted(missing)}")
        invalid_scopes = set(table["truth_scope"].astype(str)).difference(
            SYNTHETIC_TRUTH_SCOPES
        )
        if invalid_scopes:
            raise ValueError(f"{label} records have invalid truth scopes")
    left = track_a_records.copy()
    right = track_b_records.copy()
    left["record_source"] = "track_a_differential_truth"
    right["record_source"] = "track_b_receiver_program_truth"
    return pd.concat([left, right], ignore_index=True, sort=False)


def _markdown_table(table: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if value is None or value is pd.NA:
            return ""
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            return "" if math.isnan(numeric) else f"{numeric:.6g}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _report_markdown(summary: pd.DataFrame, spec: SuiteSpec) -> str:
    primary = summary.loc[summary["estimand"].eq(PRIMARY_ESTIMAND)].copy()
    active = primary.loc[primary["scenario"].eq(spec.positive_scenario)]
    columns = [
        "method",
        "estimable_edges",
        "auroc",
        "average_precision",
        "known_edge_effect",
        "known_edge_effect_rank",
        "known_edge_positive_direction_fraction",
    ]
    lines = [
        "# Track A paired differential synthetic truth",
        "",
        "The primary estimand is the paired target-minus-reference change in each ",
        "method's direction-oriented native score. Native effect magnitudes are not ",
        "cross-method comparable. The rank-strength estimand is a sensitivity ",
        "analysis.",
        "Adapter omissions are not imputed as native zero, and no p-value, q-value, ",
        "FDR, or type-I-error estimate is produced.",
        "",
        "## Active positive control (primary estimand)",
        "",
        _markdown_table(active.loc[:, columns]),
        "",
        "## Known CXCL10-CXCR3 edge across scenarios",
        "",
    ]
    known_columns = [
        "scenario",
        "method",
        "known_edge_effect",
        "known_edge_effect_rank",
        "known_edge_positive_direction_fraction",
        "known_edge_status",
    ]
    lines.append(_markdown_table(primary.loc[:, known_columns]))
    lines.extend(
        [
            "",
            "All-negative scenarios have single-class truth. Their AUROC, average ",
            "precision, and top-k truth metrics are explicitly not estimable; ",
            "known-edge effects there are pre-registered directional diagnostics, ",
            "not significance.",
            "",
        ]
    )
    return "\n".join(lines)


def _selected_truth(truth: pd.DataFrame, dataset: str) -> pd.DataFrame:
    selected = truth.loc[truth["dataset"].astype(str).eq(dataset)].copy()
    if selected.empty:
        raise ValueError(f"Track-A truth is absent for dataset {dataset}")
    return selected


def _identity(score_table: pd.DataFrame) -> dict[str, str]:
    return {
        "dataset": _constant(score_table, "dataset"),
        "method": _constant(score_table, "method"),
        "method_version": _constant(score_table, "method_version"),
        "analysis_track": _constant(score_table, "analysis_track"),
        "resource": _constant(score_table, "resource"),
        "resource_version": _constant(score_table, "resource_version"),
        "resource_mode": _constant(score_table, "resource_mode"),
        "score_semantics": _constant(score_table, "score_semantics"),
        "score_direction": _constant(score_table, "score_direction"),
        "universe_id": _constant(score_table, "universe_id"),
    }


def run_track_a_differential_suite(
    specification_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate every frozen Track-A scenario/method and persist strict outputs."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    spec_path = Path(specification_path).resolve()
    spec = _load_specification(spec_path, root)
    truth = _load_truth(spec)
    if not spec.track_b_records_path.is_file():
        raise FileNotFoundError(
            f"Track-B simulation records are missing: {spec.track_b_records_path}"
        )

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Track-A output directory is not empty: {output}")

    edge_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    input_records: list[dict[str, Any]] = []
    for scenario in spec.scenarios:
        scenario_truth = _selected_truth(truth, scenario.dataset)
        for method in spec.methods:
            path = (
                spec.suite_root
                / scenario.scenario
                / method
                / "interactions_long.parquet"
            )
            artifact, selected_run_id = _validate_input_artifact(
                path,
                method=method,
                crychic_primary_contrast_candidate=(
                    spec.crychic_primary_contrast_candidate
                ),
            )
            external_table = pd.read_parquet(path)
            if selected_run_id is not None:
                external_table = external_table.loc[
                    external_table["run_id"].astype(str).eq(selected_run_id)
                ].copy()
                if external_table.empty:
                    raise ValueError("selected CRYCHIC score view has no adapter rows")
            elif external_table["run_id"].astype(str).nunique() != 1:
                raise ValueError("non-CRYCHIC Track-A input must contain one run_id")

            score_table, estimands = prepare_estimands(
                external_table, context_key=spec.context_key, contrast=spec.contrast
            )
            identity = _identity(score_table)
            if identity["dataset"] != scenario.dataset:
                raise ValueError("configured dataset disagrees with Track-A adapter")
            if identity["method"] != EXPECTED_METHOD_IDS[method]:
                raise ValueError("configured method disagrees with Track-A adapter")
            if identity["analysis_track"] != ANALYSIS_TRACK:
                raise ValueError("Track-A differential suite requires lr_stlr")
            if identity["resource_mode"] != "H-common":
                raise ValueError("Track-A differential suite requires H-common")
            truth_universe = _constant(scenario_truth, "universe_id")
            if identity["universe_id"] != truth_universe:
                raise ValueError("Track-A adapter and truth universe_id disagree")

            score_edges = set(
                score_table.loc[:, EDGE_KEYS].itertuples(index=False, name=None)
            )
            truth_edges = set(
                scenario_truth.loc[:, EDGE_KEYS].itertuples(index=False, name=None)
            )
            if score_edges != truth_edges:
                raise ValueError("adapter edges must exactly match Track-A truth")

            for prepared in estimands:
                edges = paired_edge_differences(
                    prepared.subject_context,
                    scenario_truth,
                    reference=spec.reference,
                    target=spec.target,
                    min_pairs=spec.min_pairs,
                    estimand=prepared.estimand,
                )
                summary = summarize_differential_truth(
                    edges, top_k=spec.top_k, known_edge=spec.known_edge
                )
                shared: dict[str, object] = {
                    **identity,
                    "scenario": scenario.scenario,
                    "reference": spec.reference,
                    "target": spec.target,
                    "contrast": spec.contrast,
                    "estimand": prepared.estimand,
                    "missing_policy": prepared.missing_policy,
                    "scale_comparability": prepared.scale_comparability,
                    "minimum_pairs": spec.min_pairs,
                }
                edges = edges.assign(**shared)
                edge_tables.append(edges)
                summaries.append(shared | summary)
            input_records.append(
                artifact
                | {
                    "dataset": scenario.dataset,
                    "scenario": scenario.scenario,
                    "method": method,
                    "selected_run_id": selected_run_id,
                }
            )

    edges_table = pd.concat(edge_tables, ignore_index=True, sort=False)
    summary_table = pd.DataFrame(summaries)
    expected_summary_rows = len(spec.scenarios) * len(spec.methods) * 2
    if len(summary_table) != expected_summary_rows:
        raise RuntimeError("Track-A summary row count disagrees with suite design")
    track_a_records = simulation_truth_records(summary_table)
    track_b_records = pd.read_csv(spec.track_b_records_path, sep="\t")
    combined_records = merge_simulation_records(track_a_records, track_b_records)

    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    paths = {
        "edges": staging / "track_a_differential_edges.tsv",
        "summary": staging / "track_a_differential_summary.tsv",
        "simulation_truth": staging / "track_a_differential_simulation_truth.tsv",
        "simulation_records": staging / "simulation_records.tsv",
        "report": staging / "track_a_differential_report.md",
    }
    edges_table.to_csv(paths["edges"], sep="\t", index=False)
    summary_table.to_csv(paths["summary"], sep="\t", index=False)
    track_a_records.to_csv(paths["simulation_truth"], sep="\t", index=False)
    combined_records.to_csv(paths["simulation_records"], sep="\t", index=False)
    paths["report"].write_text(_report_markdown(summary_table, spec), encoding="utf-8")
    table_by_name: dict[str, pd.DataFrame | None] = {
        "edges": edges_table,
        "summary": summary_table,
        "simulation_truth": track_a_records,
        "simulation_records": combined_records,
        "report": None,
    }
    output_rows = {
        name: None if table is None else len(table)
        for name, table in table_by_name.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": TRACK_A_OUTPUT_SCHEMA,
        "specification": str(spec_path),
        "specification_sha256": sha256_file(spec_path),
        "track_a_truth": {
            "path": str(spec.truth_path),
            "sha256": sha256_file(spec.truth_path),
            "rows": len(truth),
        },
        "track_b_simulation_records": {
            "path": str(spec.track_b_records_path),
            "sha256": sha256_file(spec.track_b_records_path),
            "rows": len(track_b_records),
        },
        "settings": {
            "reference": spec.reference,
            "target": spec.target,
            "contrast": spec.contrast,
            "min_pairs": spec.min_pairs,
            "top_k": spec.top_k,
            "truth_universe_size": spec.truth_universe_size,
            "primary_estimand": PRIMARY_ESTIMAND,
            "sensitivity_estimand": SENSITIVITY_ESTIMAND,
            "native_missing_policy": (
                "paired_observed_native_scores_only_no_zero_imputation"
            ),
            "negative_control_classification": "single_class_not_estimable",
            "formal_inference": "not_computed",
            "crychic_primary_contrast_candidate": (
                spec.crychic_primary_contrast_candidate
            ),
        },
        "inputs": input_records,
        "outputs": {
            name: {
                "path": path.name,
                "rows": output_rows[name],
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    write_json(staging / "manifest.json", manifest)
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute paired Track-A differential synthetic truth metrics"
    )
    parser.add_argument("specification", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run_track_a_differential_suite(
        args.specification,
        args.output_dir,
        overwrite=args.overwrite,
        repo_root=args.repo_root,
    )
    print(
        _canonical_json(
            {
                "status": "complete",
                "output": str(Path(args.output_dir).resolve()),
                "rows": manifest["outputs"],
            }
        )
    )


if __name__ == "__main__":
    main()
