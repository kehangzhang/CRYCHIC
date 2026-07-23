"""Cross CRYCHIC score components with common sample-level effect engines."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.evaluate_three_group import (
    EDGE_KEYS,
    _multigroup_metrics,
)
from benchmarks.metrics.multicondition import (
    external_long_to_score_table,
    unpaired_edge_effects,
)
from crychic.core import stable_id

SCHEMA_VERSION = "crychic-three-group-component-crossover-v1"
SOURCE_EVALUATION_SCHEMAS = frozenset(
    {"crychic-three-group-evaluation-v1", "crychic-three-group-evaluation-v2"}
)
SCORE_LAYERS = (
    "strict_geometric",
    "sender_downstream_blend_90_10",
    "availability_only",
    "mechanistic_geometric",
    "availability_downstream_geometric",
    "downstream_only",
    "sender_only",
)
ENGINES = ("within_sample_rank_mean", "native_raw_mean")
FROZEN_CANDIDATE_ARM = "sender_downstream_blend_90_10__native_raw_mean"
REFERENCE_ARMS = (
    "strict_geometric__within_sample_rank_mean",
    "strict_geometric__native_raw_mean",
)
VALIDATION_METRICS = {
    "active": {
        "omnibus_auprc": "higher",
        "omnibus_auroc": "higher",
        "localization_macro_auprc": "higher",
        "direction_accuracy_all_active": "higher",
        "positive_direction_ap": "higher",
        "negative_direction_ap": "higher",
        "effect_all_zero_fraction": "lower",
    },
    "global_null": {
        "effect_standard_deviation": "lower",
        "effect_dynamic_range": "lower",
        "effect_all_zero_fraction": "lower",
    },
}
COMPONENT_COLUMNS = (
    "availability",
    "downstream",
    "sender_component",
    "prior_quality",
    "comm_strength",
)
TRUTH_COLUMNS = (
    "schema_version",
    "dataset_id",
    "scenario",
    "seed",
    "contrast",
    "target",
    "reference",
    *EDGE_KEYS,
    "truth_effect",
    "truth_label",
    "truth_direction",
)


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _bound_output(directory: Path, manifest: Mapping[str, Any], name: str) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get(name), Mapping):
        raise ValueError(f"manifest does not bind output {name!r}: {directory}")
    record = outputs[name]
    path = directory / name
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"output checksum mismatch: {path}")
    return path


def _source_truth(evaluation_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _read_json(evaluation_dir / "manifest.json")
    if (
        manifest.get("schema_version") not in SOURCE_EVALUATION_SCHEMAS
        or manifest.get("status") != "complete"
    ):
        raise ValueError("source must be a complete three-group evaluation")
    effects_path = _bound_output(evaluation_dir, manifest, "event_effects.tsv.gz")
    effects = pd.read_csv(effects_path, sep="\t")
    missing = {*TRUTH_COLUMNS, "method", "run_directory"}.difference(effects.columns)
    if missing:
        raise ValueError(f"source event effects are missing: {sorted(missing)}")
    crychic = effects.loc[effects["method"].astype(str).eq("crychic")].copy()
    if crychic.empty:
        raise ValueError("source evaluation contains no CRYCHIC event effects")
    identity = ["dataset_id", "contrast", *EDGE_KEYS]
    if crychic.duplicated(identity).any():
        raise ValueError("source evaluation duplicates CRYCHIC event truth")
    truth = crychic.loc[:, [*TRUTH_COLUMNS, "run_directory"]].copy()
    return truth, manifest


def _edge_map(long_table: pd.DataFrame) -> pd.DataFrame:
    edges = long_table.loc[
        :, ["sender", "receiver", "interaction_id"]
    ].drop_duplicates()
    edges["edge_id"] = [
        stable_id(
            "communication_edge",
            {
                "interaction_id": str(interaction_id),
                "receiver": str(receiver),
                "sender": str(sender),
            },
        )
        for sender, receiver, interaction_id in edges.itertuples(
            index=False, name=None
        )
    ]
    if edges["edge_id"].duplicated().any():
        raise ValueError("communication edge IDs are not one-to-one")
    return edges


def _view_map(adapter_manifest: Mapping[str, Any]) -> pd.DataFrame:
    source = adapter_manifest.get("source_result")
    views = source.get("score_views") if isinstance(source, Mapping) else None
    if not isinstance(views, list) or not views:
        raise ValueError("CRYCHIC source score-view provenance is absent")
    records: list[dict[str, str]] = []
    for view in views:
        if not isinstance(view, Mapping):
            raise ValueError("CRYCHIC score-view record is invalid")
        candidates = view.get("contrast_candidates")
        children = view.get("child_scoring_functional_ids")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 1
            or not isinstance(children, list)
            or not children
        ):
            raise ValueError("CRYCHIC score view lacks one contrast and children")
        for functional_id in children:
            records.append(
                {
                    "scoring_functional_id": str(functional_id),
                    "run_id": str(view["run_id"]),
                    "contrast_view": str(candidates[0]),
                }
            )
    result = pd.DataFrame.from_records(records)
    if result["scoring_functional_id"].duplicated().any():
        raise ValueError("one scoring functional maps to multiple score views")
    return result


def _read_component_long(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter_manifest = _read_json(run_dir / "manifest.json")
    if adapter_manifest.get("status") != "complete":
        raise ValueError(f"CRYCHIC adapter run is incomplete: {run_dir}")
    output = adapter_manifest.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("CRYCHIC adapter output record is absent")
    long_path = run_dir / str(output.get("table"))
    if not long_path.is_file() or sha256_file(long_path) != output.get("sha256"):
        raise ValueError(f"CRYCHIC long-table checksum mismatch: {long_path}")
    long_table = pd.read_parquet(long_path)

    source = adapter_manifest.get("source_result")
    if not isinstance(source, Mapping):
        raise ValueError("CRYCHIC source result record is absent")
    result_dir = run_dir / str(source.get("directory"))
    result_manifest_path = result_dir / "run_manifest.json"
    result_manifest = _read_json(result_manifest_path)
    if result_manifest.get("status") != "complete":
        raise ValueError(f"CRYCHIC result is incomplete: {result_dir}")
    tables = result_manifest.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError("CRYCHIC result table provenance is absent")
    sample_record = tables.get("sample_scores")
    if not isinstance(sample_record, Mapping):
        raise ValueError("CRYCHIC sample-score record is absent")
    sample_path = result_dir / str(sample_record.get("filename"))
    if not sample_path.is_file() or sha256_file(sample_path) != sample_record.get(
        "sha256"
    ):
        raise ValueError(f"sample-score checksum mismatch: {sample_path}")
    sample_scores = pd.read_parquet(sample_path)
    if set(sample_scores["mode"].astype(str)) != {"state"}:
        raise ValueError("component crossover currently requires state scores")

    scores = sample_scores.merge(
        _edge_map(long_table), on="edge_id", how="left", validate="many_to_one"
    ).merge(
        _view_map(adapter_manifest),
        on="scoring_functional_id",
        how="left",
        validate="many_to_one",
    )
    if scores[["sender", "receiver", "interaction_id", "run_id"]].isna().any().any():
        raise ValueError("component ledger does not map to every external score row")
    join_keys = [
        "run_id",
        "sample_id",
        "subject_id",
        "context_json",
        "sender",
        "receiver",
        "interaction_id",
    ]
    if scores.duplicated(join_keys).any() or long_table.duplicated(join_keys).any():
        raise ValueError("component-to-external score join keys are not unique")
    component_long = long_table.merge(
        scores.loc[:, [*join_keys, "contrast_view", *COMPONENT_COLUMNS]],
        on=join_keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_ledger"),
    )
    if len(component_long) != len(long_table):
        raise RuntimeError("component join changed the external score row count")
    numeric = component_long.loc[:, list(COMPONENT_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("complete source run contains missing component values")
    if ((numeric < 0.0) | (numeric > 1.0)).any().any():
        raise ValueError("source score components must lie in [0, 1]")
    component_long.loc[:, list(COMPONENT_COLUMNS)] = numeric
    if not np.allclose(
        component_long["score"].to_numpy(dtype=float),
        component_long["comm_strength"].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("persisted strict score disagrees with the component ledger")
    provenance = {
        "run_directory": run_dir.name,
        "adapter_manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "long_table_sha256": sha256_file(long_path),
        "result_manifest_sha256": sha256_file(result_manifest_path),
        "sample_scores_sha256": sha256_file(sample_path),
    }
    return component_long, provenance


def _geometric_product(table: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = table.loc[:, list(columns)].astype(float)
    return values.prod(axis=1) ** (1.0 / len(columns))


def score_layers(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Return preregistered component layers without refitting any model."""

    missing = set(COMPONENT_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"component score table is missing: {sorted(missing)}")
    return {
        "strict_geometric": table["comm_strength"].astype(float),
        "sender_downstream_blend_90_10": (
            0.90 * table["sender_component"].astype(float)
            + 0.10 * table["downstream"].astype(float)
        ),
        "availability_only": table["availability"].astype(float),
        "mechanistic_geometric": _geometric_product(
            table, ("availability", "sender_component", "prior_quality")
        ),
        "availability_downstream_geometric": _geometric_product(
            table, ("availability", "downstream", "prior_quality")
        ),
        "downstream_only": table["downstream"].astype(float),
        "sender_only": table["sender_component"].astype(float),
    }


def _layer_diagnostics(
    table: pd.DataFrame,
    *,
    dataset_id: str,
    contrast_view: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer, score in score_layers(table).items():
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "contrast_view": contrast_view,
                "score_layer": layer,
                "rows": len(score),
                "zero_fraction": float(score.eq(0.0).mean()),
                "unique_values": int(score.nunique()),
                "standard_deviation": float(score.std(ddof=0)),
                "interquartile_range": float(
                    score.quantile(0.75) - score.quantile(0.25)
                ),
                "dynamic_range": float(score.max() - score.min()),
                **{
                    f"{component}_zero_fraction": float(
                        table[component].astype(float).eq(0.0).mean()
                    )
                    for component in COMPONENT_COLUMNS
                },
            }
        )
    return records


def _effect_arm(
    layer_long: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    layer: str,
    engine: str,
    target: str,
    reference: str,
    contrast: str,
    dataset_id: str,
) -> pd.DataFrame:
    mapped = external_long_to_score_table(
        layer_long,
        context_key="condition",
        contrast=contrast,
        dataset=dataset_id,
    )
    if engine == "native_raw_mean":
        mapped["comparison_strength"] = mapped["score"]
    elif engine != "within_sample_rank_mean":
        raise ValueError(f"unknown crossover engine: {engine}")
    effects = unpaired_edge_effects(
        mapped,
        reference=reference,
        target=target,
        min_subjects=4,
        contrast=contrast,
        validated=True,
    )
    selected = effects.loc[:, [*EDGE_KEYS, "effect", "status", "reason_code"]]
    result = truth.merge(
        selected, on=list(EDGE_KEYS), how="left", validate="one_to_one"
    )
    missing = result["status"].isna()
    result.loc[missing, "status"] = "not_estimable"
    result.loc[missing, "reason_code"] = "event_not_returned"
    result["method"] = f"{layer}__{engine}"
    result["method_label"] = result["method"]
    result["score_layer"] = layer
    result["differential_engine"] = engine
    return result


def _arm_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    active = metrics.loc[
        metrics["scenario"].eq("active") & metrics["omnibus_status"].eq("observed")
    ]
    expected = metrics.loc[metrics["scenario"].eq("active"), "dataset_id"].nunique()
    columns = (
        "omnibus_prevalence_adjusted_ap",
        "omnibus_auprc",
        "omnibus_auroc",
        "localization_macro_auprc",
        "localization_micro_auprc",
        "positive_direction_ap",
        "negative_direction_ap",
        "direction_accuracy_all_active",
        "effect_spearman",
        "effect_all_zero_fraction",
        "effect_dynamic_range",
        "event_coverage",
    )
    aggregations = {f"mean_{column}": (column, "mean") for column in columns}
    summary = (
        active.groupby("method", observed=True)
        .agg(**aggregations, active_datasets_scored=("dataset_id", "size"))
        .reset_index()
    )
    parts = summary["method"].str.rsplit("__", n=1, expand=True)
    summary.insert(1, "score_layer", parts[0])
    summary.insert(2, "differential_engine", parts[1])
    summary["active_datasets_expected"] = expected
    summary["rank_eligible"] = (
        summary["active_datasets_scored"].eq(expected)
        & summary["mean_event_coverage"].ge(0.80)
    )
    rank_columns = (
        "mean_omnibus_prevalence_adjusted_ap",
        "mean_omnibus_auprc",
        "mean_omnibus_auroc",
        "mean_localization_macro_auprc",
        "mean_positive_direction_ap",
        "mean_negative_direction_ap",
        "mean_direction_accuracy_all_active",
        "mean_effect_spearman",
    )
    for column in rank_columns:
        summary[f"{column}_rank"] = (
            summary[column]
            .where(summary["rank_eligible"])
            .rank(ascending=False, method="average")
        )
    summary["primary_rank"] = summary["mean_omnibus_prevalence_adjusted_ap_rank"]
    return summary.sort_values(
        ["primary_rank", "method"], na_position="last", ignore_index=True
    )


def _candidate_validation(
    metrics: pd.DataFrame,
    *,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 20260723,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    methods = set(metrics["method"].astype(str))
    required = {FROZEN_CANDIDATE_ARM, *REFERENCE_ARMS}
    missing = required.difference(methods)
    if missing:
        raise ValueError(f"candidate validation arms are absent: {sorted(missing)}")
    rng = np.random.default_rng(bootstrap_seed)
    paired_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for scenario, metric_directions in VALIDATION_METRICS.items():
        selected = metrics.loc[metrics["scenario"].astype(str).eq(scenario)]
        for metric, direction in metric_directions.items():
            pivot = selected.pivot(index="seed", columns="method", values=metric)
            for reference in REFERENCE_ARMS:
                pair = pivot.loc[:, [FROZEN_CANDIDATE_ARM, reference]].dropna()
                candidate = pair[FROZEN_CANDIDATE_ARM].to_numpy(dtype=float)
                baseline = pair[reference].to_numpy(dtype=float)
                raw_difference = candidate - baseline
                improvement = (
                    raw_difference if direction == "higher" else -raw_difference
                )
                if len(improvement):
                    sampled = rng.choice(
                        improvement,
                        size=(bootstrap_replicates, len(improvement)),
                        replace=True,
                    ).mean(axis=1)
                    lower, upper = np.quantile(sampled, (0.025, 0.975))
                else:
                    lower = upper = math.nan
                summary_records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "scenario": scenario,
                        "metric": metric,
                        "direction": direction,
                        "candidate_arm": FROZEN_CANDIDATE_ARM,
                        "reference_arm": reference,
                        "n_paired_seeds": len(improvement),
                        "candidate_mean": (
                            float(candidate.mean()) if len(candidate) else math.nan
                        ),
                        "reference_mean": (
                            float(baseline.mean()) if len(baseline) else math.nan
                        ),
                        "raw_candidate_minus_reference": (
                            float(raw_difference.mean())
                            if len(raw_difference)
                            else math.nan
                        ),
                        "oriented_mean_improvement": (
                            float(improvement.mean())
                            if len(improvement)
                            else math.nan
                        ),
                        "oriented_improvement_ci_lower": float(lower),
                        "oriented_improvement_ci_upper": float(upper),
                        "candidate_wins": int((improvement > 0.0).sum()),
                        "ties": int((improvement == 0.0).sum()),
                        "candidate_losses": int((improvement < 0.0).sum()),
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_seed": bootstrap_seed,
                    }
                )
                for seed, candidate_value, reference_value, raw, oriented in zip(
                    pair.index,
                    candidate,
                    baseline,
                    raw_difference,
                    improvement,
                    strict=True,
                ):
                    paired_records.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "scenario": scenario,
                            "seed": int(seed),
                            "metric": metric,
                            "direction": direction,
                            "candidate_arm": FROZEN_CANDIDATE_ARM,
                            "reference_arm": reference,
                            "candidate_value": float(candidate_value),
                            "reference_value": float(reference_value),
                            "raw_candidate_minus_reference": float(raw),
                            "oriented_improvement": float(oriented),
                        }
                    )
    summary = pd.DataFrame.from_records(summary_records)
    paired = pd.DataFrame.from_records(paired_records)
    primary = summary.loc[
        summary["scenario"].eq("active")
        & summary["metric"].eq("omnibus_auprc")
        & summary["reference_arm"].eq(REFERENCE_ARMS[0])
    ]
    null_sd = summary.loc[
        summary["scenario"].eq("global_null")
        & summary["metric"].eq("effect_standard_deviation")
        & summary["reference_arm"].eq(REFERENCE_ARMS[0])
    ]
    candidate_active = metrics.loc[
        metrics["scenario"].eq("active")
        & metrics["method"].eq(FROZEN_CANDIDATE_ARM)
    ]
    complete = len(primary) == 1 and len(null_sd) == 1
    primary_pass = bool(
        complete
        and int(primary["n_paired_seeds"].iloc[0]) >= 20
        and float(primary["oriented_improvement_ci_lower"].iloc[0]) > 0.0
    )
    coverage_pass = bool(
        not candidate_active.empty
        and candidate_active["event_coverage"].ge(0.80).all()
    )
    null_pass = bool(
        complete and float(null_sd["oriented_mean_improvement"].iloc[0]) >= 0.0
    )
    gate = {
        "candidate_arm": FROZEN_CANDIDATE_ARM,
        "primary_reference_arm": REFERENCE_ARMS[0],
        "minimum_paired_validation_seeds": 20,
        "primary_metric": "active.omnibus_auprc",
        "primary_rule": "paired_seed_bootstrap_95pct_CI_lower_gt_0",
        "primary_pass": primary_pass,
        "coverage_rule": "every_active_seed_event_coverage_ge_0.80",
        "coverage_pass": coverage_pass,
        "null_rule": "mean_null_effect_SD_not_greater_than_reference",
        "null_pass": null_pass,
        "status": (
            "ACCEPT" if primary_pass and coverage_pass and null_pass else "REJECT"
        ),
    }
    return paired, summary, gate


def _report(
    summary: pd.DataFrame,
    *,
    validation: pd.DataFrame,
    gate: Mapping[str, Any],
) -> str:
    lines = [
        "# Three-group score generator x differential engine crossover",
        "",
        (
            "All arms reuse the same frozen CRYCHIC component ledger and truth. "
            "No model is refit. Native-scale RMSE is not cross-arm ranked."
        ),
        "",
        (
            "| Rank | Score layer | Engine | Omnibus AUPRC | Omnibus AUROC | "
            "Localization AP | AP+ | AP- | Direction | Zero effects |"
        ),
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        rank = "NE" if pd.isna(row.primary_rank) else f"{float(row.primary_rank):g}"
        values = (
            rank,
            str(row.score_layer),
            str(row.differential_engine),
            f"{row.mean_omnibus_auprc:.4f}",
            f"{row.mean_omnibus_auroc:.4f}",
            f"{row.mean_localization_macro_auprc:.4f}",
            f"{row.mean_positive_direction_ap:.4f}",
            f"{row.mean_negative_direction_ap:.4f}",
            f"{row.mean_direction_accuracy_all_active:.4f}",
            f"{row.mean_effect_all_zero_fraction:.4f}",
        )
        lines.append("| " + " | ".join(values) + " |")
    primary = validation.loc[
        validation["scenario"].eq("active")
        & validation["metric"].eq("omnibus_auprc")
        & validation["reference_arm"].eq(REFERENCE_ARMS[0])
    ].iloc[0]
    lines.extend(
        (
            "",
            "## Frozen candidate validation",
            "",
            f"Gate status: **{gate['status']}**",
            "",
            (
                f"Against `{REFERENCE_ARMS[0]}`, paired omnibus AUPRC "
                f"improvement was {primary.oriented_mean_improvement:.4f} "
                f"(95% bootstrap CI "
                f"[{primary.oriented_improvement_ci_lower:.4f}, "
                f"{primary.oriented_improvement_ci_upper:.4f}]); "
                f"wins/ties/losses = {primary.candidate_wins}/"
                f"{primary.ties}/{primary.candidate_losses}."
            ),
            "",
        )
    )
    return "\n".join(lines) + "\n"


def evaluate_crossover(
    source_evaluation: Path,
    runs_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate frozen component layers and atomically publish diagnostics."""

    source_evaluation = source_evaluation.resolve()
    runs_dir = runs_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output_dir}")
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    stage_start = time.perf_counter()
    truth, source_manifest = _source_truth(source_evaluation)
    stage_seconds["read_truth"] = time.perf_counter() - stage_start

    cache: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    effect_frames: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, Any]] = []
    stage_start = time.perf_counter()
    for (dataset_id, contrast), selected_truth in truth.groupby(
        ["dataset_id", "contrast"], observed=True, sort=True
    ):
        run_names = tuple(selected_truth["run_directory"].astype(str).unique())
        if len(run_names) != 1:
            raise ValueError("one dataset must bind one CRYCHIC run directory")
        run_name = run_names[0]
        if run_name not in cache:
            cache[run_name] = _read_component_long(runs_dir / run_name)
            provenance.append(cache[run_name][1])
        component_long = cache[run_name][0]
        target = str(selected_truth["target"].iloc[0])
        reference = str(selected_truth["reference"].iloc[0])
        contrast_view = f"global:'{target}'"
        view = component_long.loc[
            component_long["contrast_view"].astype(str).eq(contrast_view)
        ].copy()
        if view.empty:
            raise ValueError(f"score view is absent: {run_name}/{contrast_view}")
        diagnostic_records.extend(
            _layer_diagnostics(
                view, dataset_id=str(dataset_id), contrast_view=contrast_view
            )
        )
        layer_values = score_layers(view)
        truth_table = selected_truth.drop(columns="run_directory")
        for layer in SCORE_LAYERS:
            layer_long = view.copy()
            layer_long["score"] = layer_values[layer]
            layer_long["score_name"] = layer
            layer_long["status"] = "ok"
            layer_long["reason_code"] = "component_crossover_observed"
            for engine in ENGINES:
                result = _effect_arm(
                    layer_long,
                    truth_table,
                    layer=layer,
                    engine=engine,
                    target=target,
                    reference=reference,
                    contrast=str(contrast),
                    dataset_id=str(dataset_id),
                )
                result["run_directory"] = run_name
                effect_frames.append(result)
    stage_seconds["component_effects"] = time.perf_counter() - stage_start

    effects = pd.concat(effect_frames, ignore_index=True)
    stage_start = time.perf_counter()
    metrics, confusion = _multigroup_metrics(effects)
    summary = _arm_summary(metrics)
    paired_validation, validation_summary, validation_gate = (
        _candidate_validation(metrics)
    )
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    stage_seconds["metrics"] = time.perf_counter() - stage_start

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    published = False
    try:
        stage_start = time.perf_counter()
        paths = {
            "event_effects.tsv.gz": effects,
            "multigroup_metrics.tsv": metrics,
            "contrast_confusion.tsv": confusion,
            "arm_summary.tsv": summary,
            "component_diagnostics.tsv": diagnostics,
            "candidate_seed_pairs.tsv": paired_validation,
            "candidate_validation.tsv": validation_summary,
        }
        for name, table in paths.items():
            table.to_csv(
                staged / name,
                sep="\t",
                index=False,
                compression="gzip" if name.endswith(".gz") else None,
            )
        (staged / "REPORT.md").write_text(
            _report(
                summary,
                validation=validation_summary,
                gate=validation_gate,
            ),
            encoding="utf-8",
        )
        stage_seconds["export"] = time.perf_counter() - stage_start
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "source_evaluation": {
                "directory": str(source_evaluation),
                "manifest_sha256": sha256_file(source_evaluation / "manifest.json"),
                "schema_version": source_manifest["schema_version"],
            },
            "source_runs": sorted(provenance, key=lambda item: item["run_directory"]),
            "score_layers": list(SCORE_LAYERS),
            "differential_engines": list(ENGINES),
            "frozen_candidate_validation": validation_gate,
            "score_layer_contract": {
                "strict_geometric": "persisted four-component geometric score",
                "sender_downstream_blend_90_10": (
                    "frozen development candidate: 0.90*sender_component + "
                    "0.10*downstream"
                ),
                "availability_only": "sample LR availability",
                "mechanistic_geometric": (
                    "geometric(availability,sender_component,prior_quality)"
                ),
                "availability_downstream_geometric": (
                    "geometric(availability,downstream,prior_quality)"
                ),
                "downstream_only": "receiver downstream activity",
                "sender_only": "context-specific sender assignment component",
            },
            "engine_contract": {
                "within_sample_rank_mean": (
                    "target-minus-reference subject mean of within-sample rank strength"
                ),
                "native_raw_mean": (
                    "target-minus-reference subject mean on the native score scale"
                ),
            },
            "stage_seconds": stage_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "outputs": {
                name: {
                    "bytes": (staged / name).stat().st_size,
                    "sha256": sha256_file(staged / name),
                }
                for name in (*paths, "REPORT.md")
            },
        }
        _write_json(staged / "manifest.json", manifest)
        os.replace(staged, output_dir)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evaluation", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = evaluate_crossover(
        args.source_evaluation, args.runs_dir, args.output_dir
    )
    print(json.dumps(json_safe(manifest), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
