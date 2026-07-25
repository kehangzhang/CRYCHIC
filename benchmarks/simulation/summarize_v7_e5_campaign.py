"""Publish compact checksum-bound evidence from a completed v7 E5 campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as t_distribution

from benchmarks.adapters.common import json_safe, sha256_file, write_json
from benchmarks.simulation.run_v7_e5_derived_campaign import (
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
)
from benchmarks.simulation.v7_hypergraph_swaps import E5_ARMS, E5_METRICS

SCHEMA_VERSION = "crychic-suggest-next2-v7-e5-smoke-report-v1"
LOCKED_FAMILY = "hypergraph_weak_effects"
SCOPES = {
    "locked_hypergraph": (LOCKED_FAMILY,),
    "development_graph": ("graph_smooth",),
    "topology_stress": (
        "prior_corruption",
        "wrong_topology",
        "disconnected_graph",
        "topology_jump",
        "prior_replacement",
        "annotation_perturbation",
    ),
}
COMPARISONS = (
    ("full_hypergraph", "no_prior"),
    ("full_hypergraph", "degree_matched_permuted_hypergraph"),
    ("rewired_25pct", "degree_matched_permuted_hypergraph"),
    ("tensor_factorization", "full_hypergraph"),
    ("pairwise_clique_expansion", "full_hypergraph"),
)
_LOWER_IS_BETTER = {"effect_mse"}


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _descriptive_ci(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(numeric) < 2:
        return np.nan, np.nan
    mean = float(numeric.mean())
    radius = float(
        t_distribution.ppf(0.975, len(numeric) - 1)
        * numeric.std(ddof=1)
        / math.sqrt(len(numeric))
    )
    return mean - radius, mean + radius


def _dataset_metric_values(metrics: pd.DataFrame) -> pd.DataFrame:
    observed = metrics.loc[metrics["status"].eq("observed")].copy()
    return (
        observed.groupby(
            ["dataset_id", "dgp_family", "design_kind", "score_view", "metric"],
            observed=True,
            sort=True,
        )["value"]
        .mean()
        .reset_index()
    )


def _scope_summary(dataset_values: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scope, families in SCOPES.items():
        selected = dataset_values.loc[dataset_values["dgp_family"].isin(families)]
        for (arm, metric), group in selected.groupby(
            ["score_view", "metric"], observed=True, sort=True
        ):
            low, high = _descriptive_ci(group["value"])
            records.append(
                {
                    "scope": scope,
                    "score_view": arm,
                    "metric": metric,
                    "datasets": group["dataset_id"].nunique(),
                    "mean": group["value"].mean(),
                    "std": group["value"].std(),
                    "median": group["value"].median(),
                    "descriptive_ci_low": low,
                    "descriptive_ci_high": high,
                }
            )
    return pd.DataFrame.from_records(records)


def _dgp_summary(dataset_values: pd.DataFrame) -> pd.DataFrame:
    return (
        dataset_values.groupby(
            ["dgp_family", "design_kind", "score_view", "metric"],
            observed=True,
            sort=True,
        )["value"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .rename(columns={"count": "datasets"})
    )


def _paired_comparisons(dataset_values: pd.DataFrame) -> pd.DataFrame:
    locked = dataset_values.loc[dataset_values["dgp_family"].eq(LOCKED_FAMILY)]
    pivot = locked.pivot(
        index=["dataset_id", "metric"], columns="score_view", values="value"
    )
    records: list[dict[str, object]] = []
    for candidate, baseline in COMPARISONS:
        for metric in E5_METRICS:
            if metric == "topology_permutation_advantage":
                continue
            try:
                local = pivot.xs(metric, level="metric")
            except KeyError:
                continue
            paired = local.loc[:, [candidate, baseline]].dropna()
            gain = (
                paired[baseline] - paired[candidate]
                if metric in _LOWER_IS_BETTER
                else paired[candidate] - paired[baseline]
            )
            low, high = _descriptive_ci(gain)
            records.append(
                {
                    "scope": "locked_hypergraph",
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "gain_orientation": (
                        "baseline_minus_candidate"
                        if metric in _LOWER_IS_BETTER
                        else "candidate_minus_baseline"
                    ),
                    "paired_datasets": len(gain),
                    "mean_gain": gain.mean(),
                    "std_gain": gain.std(),
                    "median_gain": gain.median(),
                    "descriptive_ci_low": low,
                    "descriptive_ci_high": high,
                    "win_fraction": gain.gt(0.0).mean(),
                }
            )
    return pd.DataFrame.from_records(records)


def _locked_ranks(scope_summary: pd.DataFrame) -> pd.DataFrame:
    selected = scope_summary.loc[scope_summary["scope"].eq("locked_hypergraph")]
    records: list[dict[str, object]] = []
    for metric, group in selected.groupby("metric", observed=True, sort=True):
        group = group.copy()
        if metric == "ci_coverage":
            group["rank_value"] = (group["mean"] - 0.95).abs().round(12)
            group["rank"] = group["rank_value"].rank(method="min", ascending=True)
            orientation = "absolute_distance_to_0.95_lower_is_better"
        else:
            ascending = metric in _LOWER_IS_BETTER
            group["rank_value"] = group["mean"].round(12)
            group["rank"] = group["rank_value"].rank(method="min", ascending=ascending)
            orientation = "lower_is_better" if ascending else "higher_is_better"
        for row in group.itertuples(index=False):
            records.append(
                {
                    "metric": metric,
                    "score_view": row.score_view,
                    "mean": row.mean,
                    "rank": int(row.rank),
                    "methods": len(group),
                    "rank_orientation": orientation,
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["metric", "rank", "score_view"], kind="stable", ignore_index=True
    )


def _comparison(
    comparisons: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metric: str,
) -> pd.Series:
    selected = comparisons.loc[
        comparisons["candidate"].eq(candidate)
        & comparisons["baseline"].eq(baseline)
        & comparisons["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(f"missing paired comparison {candidate}/{baseline}/{metric}")
    return selected.iloc[0]


def _acceptance(
    *,
    comparisons: pd.DataFrame,
    scope_summary: pd.DataFrame,
    fits: pd.DataFrame,
    topologies: pd.DataFrame,
) -> dict[str, object]:
    full_raw_mse = _comparison(
        comparisons,
        candidate="full_hypergraph",
        baseline="no_prior",
        metric="effect_mse",
    )
    full_permuted_mse = _comparison(
        comparisons,
        candidate="full_hypergraph",
        baseline="degree_matched_permuted_hypergraph",
        metric="effect_mse",
    )
    rewired_permuted_mse = _comparison(
        comparisons,
        candidate="rewired_25pct",
        baseline="degree_matched_permuted_hypergraph",
        metric="effect_mse",
    )
    full_permuted_ap = _comparison(
        comparisons,
        candidate="full_hypergraph",
        baseline="degree_matched_permuted_hypergraph",
        metric="exact_hyperedge_ap",
    )
    full_raw_ap = _comparison(
        comparisons,
        candidate="full_hypergraph",
        baseline="no_prior",
        metric="exact_hyperedge_ap",
    )
    coverage_row = scope_summary.loc[
        scope_summary["scope"].eq("locked_hypergraph")
        & scope_summary["score_view"].eq("full_hypergraph")
        & scope_summary["metric"].eq("ci_coverage")
    ].iloc[0]
    estimable_nonconverged = fits.loc[
        fits["observed_edge_count"].ge(4)
        & ~fits["score_view"].eq("no_prior")
        & ~fits["converged"].astype(bool)
    ]
    degree_controls = topologies.loc[
        topologies["score_view"].isin(
            {
                "degree_matched_permuted_hypergraph",
                "rewired_10pct",
                "rewired_25pct",
                "rewired_50pct",
            }
        )
    ]
    coverage = float(coverage_row["mean"])
    checks = {
        "m5_full_mse_gain_vs_no_prior_ci_positive": bool(
            full_raw_mse["descriptive_ci_low"] > 0.0
        ),
        "m5_full_mse_gain_vs_permuted_ci_positive": bool(
            full_permuted_mse["descriptive_ci_low"] > 0.0
        ),
        "m5_rewired25_mse_gain_vs_permuted_ci_positive": bool(
            rewired_permuted_mse["descriptive_ci_low"] > 0.0
        ),
        "m5_full_exact_ap_gain_vs_permuted_ci_positive": bool(
            full_permuted_ap["descriptive_ci_low"] > 0.0
        ),
        "m5_full_exact_ap_gain_vs_no_prior_ci_positive": bool(
            full_raw_ap["descriptive_ci_low"] > 0.0
        ),
        "m5_locked_conditional_ci_coverage_between_0_92_and_0_97": bool(
            0.92 <= coverage <= 0.97
        ),
        "all_estimable_structural_fits_converged": estimable_nonconverged.empty,
        "all_degree_matched_controls_preserve_degrees": bool(
            degree_controls["degree_profiles_equal"].astype(bool).all()
        ),
    }
    return {
        "status": "SMOKE_COMPLETE_WITH_RELEASE_GAPS",
        "checks": checks,
        "estimates": {
            "locked_full_minus_raw_mse_gain": float(full_raw_mse["mean_gain"]),
            "locked_full_minus_permuted_mse_gain": float(
                full_permuted_mse["mean_gain"]
            ),
            "locked_rewired25_minus_permuted_mse_gain": float(
                rewired_permuted_mse["mean_gain"]
            ),
            "locked_full_minus_raw_exact_ap_gain": float(full_raw_ap["mean_gain"]),
            "locked_full_minus_permuted_exact_ap_gain": float(
                full_permuted_ap["mean_gain"]
            ),
            "locked_full_conditional_ci_coverage": coverage,
        },
        "claim_boundary": (
            "hypergraph representation and descriptive shrinkage only; topology "
            "estimation and calibrated interval claims are not released"
        ),
        "formal_inference_allowed": False,
    }


def _runtime_tables(
    campaign: Path,
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = _read_json(campaign / "campaign_manifest.json")
    source = Path(str(manifest["source_campaign"]["directory"]))
    plan = pd.read_csv(source / "run_plan.tsv", sep="\t").drop_duplicates("dataset_id")
    merged = runs.merge(
        plan.loc[:, ["dataset_id", "candidate_sender_count"]],
        on="dataset_id",
        validate="one_to_one",
    )
    by_candidate = (
        merged.groupby("candidate_sender_count", observed=True, sort=True)[
            "elapsed_seconds"
        ]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    records: list[dict[str, object]] = []
    for row in runs.itertuples(index=False):
        dataset_manifest = _read_json(Path(row.result_directory) / "manifest.json")
        for stage in dataset_manifest["stage_timings"]:
            records.append(
                {
                    "dataset_id": row.dataset_id,
                    "stage": stage["stage"],
                    "seconds": stage["seconds"],
                }
            )
    stage = (
        pd.DataFrame.from_records(records)
        .groupby("stage", observed=True, sort=True)["seconds"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    return by_candidate, stage


def _publish(staged: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    if not output.exists():
        os.replace(staged, output)
        return
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def summarize(
    campaign_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate one completed E5 smoke campaign and publish compact evidence."""

    campaign = campaign_dir.resolve()
    campaign_manifest_path = campaign / "campaign_manifest.json"
    campaign_manifest = _read_json(campaign_manifest_path)
    if campaign_manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("source is not a v7 E5 derived campaign")
    if (
        campaign_manifest.get("status") != "completed"
        or campaign_manifest.get("completed_datasets") != 900
        or campaign_manifest.get("failed_datasets") != 0
    ):
        raise ValueError("E5 smoke report requires 900/900 failure-free datasets")
    runs_path = campaign / str(campaign_manifest["runs"]["filename"])
    if sha256_file(runs_path) != campaign_manifest["runs"]["sha256"]:
        raise ValueError("E5 runs checksum differs from the campaign manifest")
    aggregate_paths = {
        "metrics": campaign / "all_e5_metrics.parquet",
        "fits": campaign / "all_e5_fit_diagnostics.parquet",
        "topologies": campaign / "all_e5_topology_diagnostics.parquet",
    }
    if any(not path.is_file() for path in aggregate_paths.values()):
        raise FileNotFoundError("E5 campaign lacks aggregate tables")
    runs = pd.read_csv(runs_path, sep="\t")
    metrics = pd.read_parquet(aggregate_paths["metrics"])
    fits = pd.read_parquet(aggregate_paths["fits"])
    topologies = pd.read_parquet(aggregate_paths["topologies"])
    expected_ids = set(runs["dataset_id"].astype(str))
    for name, table in (
        ("metrics", metrics),
        ("fits", fits),
        ("topologies", topologies),
    ):
        if set(table["dataset_id"].astype(str)) != expected_ids:
            raise ValueError(f"E5 {name} dataset coverage differs from runs")
        if set(table["score_view"].astype(str)) != set(E5_ARMS):
            raise ValueError(f"E5 {name} arm coverage differs")
    if set(metrics["metric"].astype(str)) != set(E5_METRICS):
        raise ValueError("E5 metric coverage differs")

    dataset_values = _dataset_metric_values(metrics)
    scope = _scope_summary(dataset_values)
    dgp = _dgp_summary(dataset_values)
    paired = _paired_comparisons(dataset_values)
    ranks = _locked_ranks(scope)
    acceptance = _acceptance(
        comparisons=paired,
        scope_summary=scope,
        fits=fits,
        topologies=topologies,
    )
    runtime, stages = _runtime_tables(campaign, runs)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.stage-", dir=output_dir.parent
    ) as temporary:
        staged = Path(temporary) / output_dir.name
        staged.mkdir()
        tables = {
            "scope_method_summary.tsv": scope,
            "dgp_method_summary.tsv": dgp,
            "locked_paired_comparisons.tsv": paired,
            "locked_method_ranks.tsv": ranks,
            "runtime_candidate_cardinality.tsv": runtime,
            "runtime_stage_summary.tsv": stages,
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False)
        write_json(staged / "acceptance.json", acceptance)
        checks = acceptance["checks"]
        estimates = acceptance["estimates"]
        gate_lines = "\n".join(
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in checks.items()
        )
        raw_mse_gain = estimates["locked_full_minus_raw_mse_gain"]
        permuted_mse_gain = estimates["locked_full_minus_permuted_mse_gain"]
        rewired_mse_gain = estimates["locked_rewired25_minus_permuted_mse_gain"]
        raw_ap_gain = estimates["locked_full_minus_raw_exact_ap_gain"]
        permuted_ap_gain = estimates["locked_full_minus_permuted_exact_ap_gain"]
        full_coverage = estimates["locked_full_conditional_ci_coverage"]
        readme = f"""# Suggest-next2 v7 E5 smoke report

This report is descriptive evidence from the checksum-bound 20-seed v7 smoke
campaign. All 900 datasets completed, including typed non-estimable outputs for
40 fully confounded datasets. It is not formal release inference.

## Locked hypergraph family

- Full versus raw MSE gain: {raw_mse_gain:.4f}
- Full versus permuted MSE gain: {permuted_mse_gain:.4f}
- 25% rewired versus permuted MSE gain: {rewired_mse_gain:.4f}
- Full versus raw exact-hyperedge AP gain: {raw_ap_gain:.4f}
- Full versus permuted exact-hyperedge AP gain: {permuted_ap_gain:.4f}
- Full conditional CI coverage: {full_coverage:.4f}

Full topology improves MSE over raw effects, but its paired MSE advantage over
the degree-matched permutation is not established. Exact-hyperedge AP does not
improve over raw, and conditional posterior intervals are severely
under-covered. Tensor factorization has the best locked exact-hyperedge AP;
full hypergraph has the best locked MSE.

## Gate status

{gate_lines}

The accepted claim is limited to hypergraph representation and descriptive
shrinkage. Topology estimation and calibrated interval claims remain withheld.
"""
        (staged / "README.md").write_text(readme, encoding="utf-8")
        report_files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staged.iterdir())
        }
        source_commits = set()
        maximum_memory = 0.0
        for row in runs.itertuples(index=False):
            manifest = _read_json(Path(row.result_directory) / "manifest.json")
            source_commits.add(str(manifest["runtime"]["git"]["commit"]))
            maximum_memory = max(
                maximum_memory,
                float(manifest["maximum_system_memory_fraction"]),
            )
        if len(source_commits) != 1:
            raise ValueError("E5 source datasets were produced by mixed commits")
        report_manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "source_campaign": campaign.name,
            "source_campaign_manifest_sha256": sha256_file(campaign_manifest_path),
            "source_estimator_commit": next(iter(source_commits)),
            "planned_datasets": 900,
            "completed_datasets": 900,
            "failed_datasets": 0,
            "maximum_system_memory_fraction": maximum_memory,
            "source_aggregates": {
                name: {"filename": path.name, "sha256": sha256_file(path)}
                for name, path in aggregate_paths.items()
            },
            "acceptance": acceptance,
            "files": report_files,
            "formal_inference_allowed": False,
        }
        write_json(staged / "manifest.json", json_safe(report_manifest))
        _publish(staged, output_dir, overwrite=overwrite)
    return report_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = summarize(
        arguments.campaign_dir,
        arguments.output_dir,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "main", "summarize"]
