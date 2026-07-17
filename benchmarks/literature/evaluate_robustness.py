"""Evaluate top-250 recovery for the Dimitrov PBMC3k perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.literature.robustness import PERTURBATION_KINDS, top_edge_overlap
from benchmarks.openproblems.common import sha256_file, write_json


def _method_metrics(top: pd.DataFrame) -> pd.DataFrame:
    required = {
        "perturbation",
        "proportion",
        "replicate",
        "seed",
        "method",
        "source",
        "target",
        "ligand",
        "receptor",
    }
    missing = required.difference(top.columns)
    if missing:
        raise ValueError(f"robustness top table is missing: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for method_name, method in top.groupby("method", sort=True, observed=True):
        baseline = method.loc[method["perturbation"].eq("baseline")]
        if baseline.empty:
            raise ValueError(f"method {method_name!r} has no baseline top ranking")
        variants = method.loc[method["perturbation"].ne("baseline")]
        for (kind, proportion, replicate, seed), variant in variants.groupby(
            ["perturbation", "proportion", "replicate", "seed"],
            sort=True,
            observed=True,
        ):
            rows.append(
                {
                    "method": method_name,
                    "perturbation": kind,
                    "proportion": float(proportion),
                    "replicate": int(replicate),
                    "seed": int(seed),
                    **top_edge_overlap(baseline, variant),
                }
            )
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        raise ValueError("robustness inputs contain no perturbed rankings")
    return result


def _baseline_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    replicates = sorted(metrics["replicate"].unique())
    methods = sorted(metrics["method"].unique())
    rows = []
    for method_name in methods:
        baseline_edges = int(
            metrics.loc[metrics["method"].eq(method_name), "baseline_edges"].iloc[0]
        )
        for kind in PERTURBATION_KINDS:
            for replicate in replicates:
                rows.append(
                    {
                        "method": method_name,
                        "perturbation": kind,
                        "proportion": 0.0,
                        "replicate": int(replicate),
                        "seed": 0,
                        "baseline_edges": baseline_edges,
                        "perturbed_edges": baseline_edges,
                        "intersection_edges": baseline_edges,
                        "baseline_recovery": 1.0,
                        "paper_tpr": 1.0,
                        "jaccard": 1.0,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    measures = ("baseline_recovery", "paper_tpr", "jaccard")
    rows: list[dict[str, object]] = []
    for keys, group in metrics.groupby(
        ["method", "perturbation", "proportion"], sort=True, observed=True
    ):
        row: dict[str, object] = {
            "method": keys[0],
            "perturbation": keys[1],
            "proportion": float(keys[2]),
            "replicates": len(group),
        }
        for measure in measures:
            values = group[measure].to_numpy(float)
            row[f"{measure}_mean"] = float(np.mean(values))
            row[f"{measure}_median"] = float(np.median(values))
            row[f"{measure}_q1"] = float(np.quantile(values, 0.25))
            row[f"{measure}_q3"] = float(np.quantile(values, 0.75))
            row[f"{measure}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method_name, kind), group in summary.groupby(
        ["method", "perturbation"], sort=True, observed=True
    ):
        group = group.sort_values("proportion", kind="stable")
        x = group["proportion"].to_numpy(float)
        y = group["baseline_recovery_mean"].to_numpy(float)
        endpoint = group.loc[np.isclose(group["proportion"], 0.4)]
        if endpoint.empty or x[0] != 0.0 or x[-1] != 0.4:
            raise ValueError("robustness curve does not span the frozen 0--40% range")
        rows.append(
            {
                "method": method_name,
                "perturbation": kind,
                "normalized_recovery_auc_0_40": float(np.trapezoid(y, x) / 0.4),
                "recovery_mean_at_40": float(
                    endpoint["baseline_recovery_mean"].iloc[0]
                ),
                "recovery_median_at_40": float(
                    endpoint["baseline_recovery_median"].iloc[0]
                ),
                "paper_tpr_mean_at_40": float(endpoint["paper_tpr_mean"].iloc[0]),
                "jaccard_mean_at_40": float(endpoint["jaccard_mean"].iloc[0]),
            }
        )
    result = pd.DataFrame.from_records(rows)
    macro = (
        result.groupby("method", sort=True, observed=True)
        .agg(
            perturbations=("perturbation", "nunique"),
            normalized_recovery_auc_macro=(
                "normalized_recovery_auc_0_40",
                "mean",
            ),
            recovery_mean_at_40_macro=("recovery_mean_at_40", "mean"),
            jaccard_mean_at_40_macro=("jaccard_mean_at_40", "mean"),
        )
        .reset_index()
    )
    return result.merge(macro, on="method", how="left", validate="many_to_one")


def run(
    crychic_top: Path,
    liana_top: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"robustness evaluation exists: {manifest_path}")
    crychic = pd.read_parquet(crychic_top)
    liana = pd.read_parquet(liana_top)
    metrics = _method_metrics(pd.concat((crychic, liana), ignore_index=True))
    metrics = pd.concat((_baseline_rows(metrics), metrics), ignore_index=True)
    summary = _summarize(metrics)
    method_summary = _method_summary(summary)
    metrics_path = output_dir / "top250_overlap_replicates.tsv"
    summary_path = output_dir / "top250_overlap_summary.tsv"
    method_path = output_dir / "method_robustness_summary.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    method_summary.to_csv(method_path, sep="\t", index=False)
    manifest: dict[str, object] = {
        "schema_version": "crychic-dimitrov-robustness-evaluation-v1",
        "status": "complete",
        "interpretation": (
            "Recovery is reproducibility against each method's own unperturbed "
            "top-250 operational reference, not biological accuracy."
        ),
        "inputs": {
            "crychic": {
                "filename": crychic_top.name,
                "sha256": sha256_file(crychic_top),
            },
            "liana": {"filename": liana_top.name, "sha256": sha256_file(liana_top)},
        },
        "methods": sorted(metrics["method"].unique()),
        "protocol": {
            "top_n": 250,
            "primary": "baseline_recovery = intersection / baseline top edges",
            "paper_tpr": "intersection / perturbed top edges",
            "secondary": "Jaccard",
            "replicates": int(metrics["replicate"].max()),
            "proportions": sorted(metrics["proportion"].unique().tolist()),
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "rows": len(table)}
            for path, table in (
                (metrics_path, metrics),
                (summary_path, summary),
                (method_path, method_summary),
            )
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crychic_top", type=Path)
    parser.add_argument("liana_top", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.crychic_top, args.liana_top, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
