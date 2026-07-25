"""Summarize checksum-bound PR10 global-null full-refit calibration campaigns."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json

SUMMARY_SCHEMA_VERSION = "crychic-suggest-next2-v7-pr10-calibration-summary-v1"
CHANNEL_TABLES = {
    "continuous_raw": "full_refit_continuous_effects.parquet",
    "occurrence_effect": "full_refit_occurrence_effects.parquet",
    "hypergraph_posterior": "full_refit_hypergraph_effects.parquet",
}
_REQUIRED_STAGES = {
    "fold_split",
    "availability_fitting",
    "absolute_activity_v2_fitting",
    "signed_program_v2_fitting",
    "sender_attribution_v2_fitting",
    "eb_shrunken_coupling_v2_fitting",
    "heldout_score_application",
    "design_aware_effect_fitting",
    "two_part_occurrence_v2_fitting",
    "hypergraph_shrinkage_v2_fitting",
}
DATASET_METRIC_COLUMNS = (
    "dataset_id",
    "dgp_family",
    "design_kind",
    "replicate_index",
    "channel",
    "n_point_total",
    "n_point_observed",
    "n_distribution_complete",
    "distribution_complete_fraction",
    "n_p_observed",
    "type_i_rejections_alpha_0_05",
    "type_i_rate_alpha_0_05",
    "n_q_observed",
    "discoveries_q_0_10",
    "false_discoveries_q_0_10",
    "false_discovery_proportion_q_0_10",
    "n_ci_observed",
    "ci_covered_zero",
    "ci_coverage_zero",
    "formal_rows",
)
SCENARIO_METRIC_COLUMNS = (
    "scenario_id",
    "dgp_family",
    "design_kind",
    "channel",
    "n_replicates",
    "n_complete_replicates",
    "complete_replicate_fraction",
    "n_hypotheses_with_p",
    "type_i_rejections_alpha_0_05",
    "empirical_type_i",
    "empirical_type_i_wilson_lower_95",
    "empirical_type_i_wilson_upper_95",
    "replicates_with_discovery_q_0_10",
    "empirical_fdr",
    "empirical_fdr_wilson_lower_95",
    "empirical_fdr_wilson_upper_95",
    "n_hypotheses_with_ci",
    "ci_covered_zero",
    "ci_coverage",
    "ci_coverage_wilson_lower_95",
    "ci_coverage_wilson_upper_95",
    "formal_rows",
)
GATE_INPUT_COLUMNS = (
    "scenario_id",
    "design_kind",
    "n_replicates",
    "empirical_type_i",
    "empirical_fdr",
    "ci_coverage",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _finite_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric), np.nan)


def _expected_resamples_by_design(
    experiment: Mapping[str, object],
) -> dict[str, int]:
    designs = experiment.get("design_kinds")
    raw = experiment.get("expected_resamples_per_dataset_by_design")
    if not isinstance(designs, list) or not isinstance(raw, Mapping):
        raise ValueError("frozen calibration config lacks per-design resample counts")
    design_names = tuple(map(str, designs))
    if len(set(design_names)) != len(design_names) or set(raw) != set(design_names):
        raise ValueError("per-design resample count keys differ from design_kinds")
    expected: dict[str, int] = {}
    for design in design_names:
        value = raw[design]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("per-design resample counts must be positive integers")
        expected[design] = value
    return expected


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
        or total < 1
        or successes < 0
        or successes > total
    ):
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total")
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _dataset_channel_metrics(
    table: pd.DataFrame,
    *,
    dataset_id: str,
    dgp_family: str,
    design_kind: str,
    replicate_index: int,
    channel: str,
    expected_bootstraps: int,
    expected_permutations: int,
    expected_loso: int,
) -> dict[str, object]:
    required = {
        "point_status",
        "n_bootstrap_total",
        "n_bootstrap_observed",
        "n_permutation_total",
        "n_permutation_observed",
        "n_loso_total",
        "n_loso_observed",
        "diagnostic_ci_lower",
        "diagnostic_ci_upper",
        "diagnostic_empirical_p_value",
        "diagnostic_q_value",
        "formal_inference_allowed",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{channel} table lacks calibration fields: {sorted(missing)}")
    if table.empty:
        raise ValueError(f"{channel} table cannot be empty")
    point_observed = table["point_status"].eq("observed")
    complete = (
        point_observed
        & table["n_bootstrap_total"].eq(expected_bootstraps)
        & table["n_bootstrap_observed"].eq(expected_bootstraps)
        & table["n_permutation_total"].eq(expected_permutations)
        & table["n_permutation_observed"].eq(expected_permutations)
        & table["n_loso_total"].eq(expected_loso)
        & table["n_loso_observed"].eq(expected_loso)
    )
    p_values = _finite_numeric(table["diagnostic_empirical_p_value"])
    q_values = _finite_numeric(table["diagnostic_q_value"])
    lower = _finite_numeric(table["diagnostic_ci_lower"])
    upper = _finite_numeric(table["diagnostic_ci_upper"])
    p_observed = point_observed & p_values.notna()
    q_observed = point_observed & q_values.notna()
    ci_observed = point_observed & lower.notna() & upper.notna()
    p_total = int(p_observed.sum())
    q_total = int(q_observed.sum())
    ci_total = int(ci_observed.sum())
    if not p_total or not q_total or not ci_total:
        raise ValueError(f"{channel} has no observed diagnostic calibration values")
    type_i = int((p_observed & p_values.le(0.05)).sum())
    discoveries = int((q_observed & q_values.le(0.10)).sum())
    covered = int((ci_observed & lower.le(0.0) & upper.ge(0.0)).sum())
    observed_points = int(point_observed.sum())
    formal_rows = int(
        table["formal_inference_allowed"].fillna(False).astype(bool).sum()
    )
    return {
        "dataset_id": dataset_id,
        "dgp_family": dgp_family,
        "design_kind": design_kind,
        "replicate_index": replicate_index,
        "channel": channel,
        "n_point_total": len(table),
        "n_point_observed": observed_points,
        "n_distribution_complete": int(complete.sum()),
        "distribution_complete_fraction": (
            int(complete.sum()) / observed_points if observed_points else math.nan
        ),
        "n_p_observed": p_total,
        "type_i_rejections_alpha_0_05": type_i,
        "type_i_rate_alpha_0_05": type_i / p_total,
        "n_q_observed": q_total,
        "discoveries_q_0_10": discoveries,
        "false_discoveries_q_0_10": discoveries,
        "false_discovery_proportion_q_0_10": 1.0 if discoveries else 0.0,
        "n_ci_observed": ci_total,
        "ci_covered_zero": covered,
        "ci_coverage_zero": covered / ci_total,
        "formal_rows": formal_rows,
    }


def _scenario_metrics(dataset_metrics: pd.DataFrame) -> pd.DataFrame:
    if tuple(dataset_metrics.columns) != DATASET_METRIC_COLUMNS:
        raise ValueError("dataset calibration metric columns are invalid")
    rows: list[dict[str, object]] = []
    grouped = dataset_metrics.groupby(
        ["dgp_family", "design_kind", "channel"],
        observed=True,
        sort=True,
    )
    for (family, design, channel), group in grouped:
        replicates = len(group)
        complete_replicates = int(group["distribution_complete_fraction"].eq(1.0).sum())
        p_total = int(group["n_p_observed"].sum())
        type_i = int(group["type_i_rejections_alpha_0_05"].sum())
        replicate_discoveries = int(group["discoveries_q_0_10"].gt(0).sum())
        ci_total = int(group["n_ci_observed"].sum())
        covered = int(group["ci_covered_zero"].sum())
        type_i_interval = _wilson_interval(type_i, p_total)
        fdr_interval = _wilson_interval(replicate_discoveries, replicates)
        coverage_interval = _wilson_interval(covered, ci_total)
        rows.append(
            {
                "scenario_id": f"{family}::{design}::{channel}",
                "dgp_family": family,
                "design_kind": design,
                "channel": channel,
                "n_replicates": replicates,
                "n_complete_replicates": complete_replicates,
                "complete_replicate_fraction": complete_replicates / replicates,
                "n_hypotheses_with_p": p_total,
                "type_i_rejections_alpha_0_05": type_i,
                "empirical_type_i": type_i / p_total,
                "empirical_type_i_wilson_lower_95": type_i_interval[0],
                "empirical_type_i_wilson_upper_95": type_i_interval[1],
                "replicates_with_discovery_q_0_10": replicate_discoveries,
                "empirical_fdr": replicate_discoveries / replicates,
                "empirical_fdr_wilson_lower_95": fdr_interval[0],
                "empirical_fdr_wilson_upper_95": fdr_interval[1],
                "n_hypotheses_with_ci": ci_total,
                "ci_covered_zero": covered,
                "ci_coverage": covered / ci_total,
                "ci_coverage_wilson_lower_95": coverage_interval[0],
                "ci_coverage_wilson_upper_95": coverage_interval[1],
                "formal_rows": int(group["formal_rows"].sum()),
            }
        )
    return pd.DataFrame.from_records(rows, columns=SCENARIO_METRIC_COLUMNS)


def _verify_dataset_artifacts(path: Path, manifest: Mapping[str, object]) -> None:
    for collection in ("outputs", "resampling_artifacts"):
        records = manifest.get(collection)
        if not isinstance(records, Mapping):
            raise ValueError(f"dataset manifest lacks {collection}: {path}")
        for record in records.values():
            if not isinstance(record, Mapping):
                raise ValueError(f"dataset artifact record is malformed: {path}")
            filename = record.get("filename")
            expected = record.get("sha256")
            if not isinstance(filename, str) or not isinstance(expected, str):
                raise ValueError(f"dataset checksum record is malformed: {path}")
            candidate = path / filename
            if not candidate.is_file() or sha256_file(candidate) != expected:
                raise ValueError(f"dataset artifact checksum differs: {candidate}")


def summarize_v7_full_refit_calibration(
    campaign_dir: Path,
    *,
    output_dir: Path,
) -> dict[str, object]:
    campaign = campaign_dir.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"calibration summary output already exists: {output}")
    manifest_path = campaign / "campaign_manifest.json"
    runs_path = campaign / "runs.tsv"
    campaign_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(campaign_manifest, dict):
        raise ValueError("campaign manifest must contain one JSON object")
    if campaign_manifest.get("status") != "completed":
        raise ValueError("calibration campaign must be completed before summary")
    frozen = campaign_manifest.get("frozen_campaign_config")
    request = campaign_manifest.get("resampling_request")
    if not isinstance(frozen, Mapping) or not isinstance(request, Mapping):
        raise ValueError("calibration campaign must be frozen and declare resampling")
    config_path = Path(str(frozen["config_path"]))
    if sha256_file(config_path) != frozen.get("config_sha256"):
        raise ValueError("frozen calibration config checksum differs")
    config: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("frozen calibration config must contain one JSON object")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("frozen calibration experiment must be a mapping")
    if experiment.get("dgp_families") != ["global_null"]:
        raise ValueError("calibration summarizer currently requires global_null only")
    expected_bootstraps = int(experiment["n_bootstraps"])
    expected_permutations = int(experiment["n_permutations"])
    expected_resamples = _expected_resamples_by_design(experiment)
    if (
        int(request.get("n_bootstraps", -1)) != expected_bootstraps
        or int(request.get("n_permutations", -1)) != expected_permutations
        or request.get("run_loso") is not True
        or int(request.get("resample_jobs", -1)) != 64
        or request.get("resample_backend") != "process"
    ):
        raise ValueError("campaign request differs from frozen calibration config")
    runs = pd.read_csv(runs_path, sep="\t")
    if runs.empty or not runs["status"].eq("completed").all():
        raise ValueError("every calibration dataset must be completed")
    if not runs["dgp_family"].eq("global_null").all():
        raise ValueError("calibration runs contain a non-global-null family")
    expected_run_resamples = runs["design_kind"].map(expected_resamples)
    if (
        expected_run_resamples.isna().any()
        or not runs["planned_resamples"].eq(expected_run_resamples).all()
    ):
        raise ValueError("calibration run has an unexpected resample count")
    records: list[dict[str, object]] = []
    dataset_manifest_hashes: dict[str, str] = {}
    source_commits: set[str] = set()
    for run in runs.itertuples(index=False):
        dataset = Path(str(run.result_directory))
        dataset_manifest_path = dataset / "manifest.json"
        dataset_manifest: object = json.loads(
            dataset_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(dataset_manifest, dict):
            raise ValueError(f"dataset manifest is malformed: {dataset}")
        if dataset_manifest.get("status") != "completed":
            raise ValueError(f"dataset is not complete: {dataset}")
        _verify_dataset_artifacts(dataset, dataset_manifest)
        if dataset_manifest.get("frozen_campaign_config") != dict(frozen):
            raise ValueError(f"dataset frozen config differs from campaign: {dataset}")
        resampling = dataset_manifest.get("resampling")
        runtime = dataset_manifest.get("runtime")
        if not isinstance(resampling, Mapping) or not isinstance(runtime, Mapping):
            raise ValueError(
                f"dataset runtime/resampling manifest is malformed: {dataset}"
            )
        git = runtime.get("git")
        if not isinstance(git, Mapping) or git.get("dirty") is not False:
            raise ValueError(
                f"calibration dataset did not use a clean commit: {dataset}"
            )
        source_commits.add(str(git.get("commit")))
        successful = resampling.get("successful_counts")
        if not isinstance(successful, Mapping):
            raise ValueError(f"dataset lacks successful resample counts: {dataset}")
        configured_stages = set(resampling.get("configured_rerun_stages_observed", ()))
        if not _REQUIRED_STAGES.issubset(configured_stages):
            raise ValueError(f"dataset lacks required full-refit stages: {dataset}")
        if int(run.successful_resamples) != int(run.planned_resamples):
            raise ValueError(f"dataset has failed resamples: {dataset}")
        expected_loso = int(successful["leave_one_subject_out"])
        dataset_manifest_hashes[str(run.dataset_id)] = sha256_file(
            dataset_manifest_path
        )
        for channel, filename in CHANNEL_TABLES.items():
            table = pd.read_parquet(dataset / filename)
            records.append(
                _dataset_channel_metrics(
                    table,
                    dataset_id=str(run.dataset_id),
                    dgp_family=str(run.dgp_family),
                    design_kind=str(run.design_kind),
                    replicate_index=int(run.replicate_index),
                    channel=channel,
                    expected_bootstraps=expected_bootstraps,
                    expected_permutations=expected_permutations,
                    expected_loso=expected_loso,
                )
            )
    dataset_metrics = pd.DataFrame.from_records(
        records,
        columns=DATASET_METRIC_COLUMNS,
    ).sort_values(
        ["design_kind", "replicate_index", "channel"],
        kind="stable",
        ignore_index=True,
    )
    scenario_metrics = _scenario_metrics(dataset_metrics)
    if len(source_commits) != 1:
        raise ValueError("calibration campaign must use exactly one source commit")
    gate_input = scenario_metrics.loc[:, list(GATE_INPUT_COLUMNS)].copy()
    output.mkdir(parents=True, exist_ok=False)
    dataset_path = output / "dataset_channel_metrics.tsv"
    scenario_path = output / "scenario_metrics.tsv"
    gate_path = output / "calibration_gate_input.tsv"
    dataset_metrics.to_csv(dataset_path, sep="\t", index=False)
    scenario_metrics.to_csv(scenario_path, sep="\t", index=False)
    gate_input.to_csv(gate_path, sep="\t", index=False)
    minimum_replicates = int(scenario_metrics["n_replicates"].min())
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "campaign_dir": str(campaign),
        "campaign_manifest_sha256": sha256_file(manifest_path),
        "runs_sha256": sha256_file(runs_path),
        "frozen_config_path": str(config_path),
        "frozen_config_sha256": sha256_file(config_path),
        "source_commits": sorted(source_commits),
        "dataset_manifest_sha256": dataset_manifest_hashes,
        "datasets": len(runs),
        "scenario_rows": len(scenario_metrics),
        "channels": list(CHANNEL_TABLES),
        "expected_resamples_per_dataset_by_design": expected_resamples,
        "formal_rows": int(dataset_metrics["formal_rows"].sum()),
        "all_distributions_complete": bool(
            dataset_metrics["distribution_complete_fraction"].eq(1.0).all()
        ),
        "release_decision_evaluated": False,
        "release_boundary": (
            "descriptive_calibration_only_below_1000_independent_null_replicates"
            if minimum_replicates < 1_000
            else "gate_input_complete_but_release_decision_not_evaluated"
        ),
        "outputs": {
            "dataset_channel_metrics": {
                "filename": dataset_path.name,
                "rows": len(dataset_metrics),
                "sha256": sha256_file(dataset_path),
            },
            "scenario_metrics": {
                "filename": scenario_path.name,
                "rows": len(scenario_metrics),
                "sha256": sha256_file(scenario_path),
            },
            "calibration_gate_input": {
                "filename": gate_path.name,
                "rows": len(gate_input),
                "sha256": sha256_file(gate_path),
            },
        },
    }
    write_json(output / "summary_manifest.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = summarize_v7_full_refit_calibration(
        arguments.campaign_dir,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
