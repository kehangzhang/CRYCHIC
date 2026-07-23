"""Summarize frozen primary methods across BRCA semi-synthetic replicates."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-brca-semisynthetic-summary-v1"
PRIMARY_POLICY = {
    "crychic": {
        "view_label": "mechanistic_sender_lr_score",
        "differential_engine": "native_raw_mean",
    },
    "cellchat": {
        "view_label": "canonical_sample_score",
        "differential_engine": "native_raw_mean",
    },
    "liana_rank_aggregate": {
        "view_label": "canonical_sample_score",
        "differential_engine": "native_raw_mean",
    },
    "scseqcommdiff": {
        "view_label": "native (OnE-PreE) - (OnNE-PreNE)",
        "differential_engine": "native_pairwise_difference_in_differences",
    },
}
SUMMARY_METRICS = (
    "omnibus_auprc",
    "omnibus_auroc",
    "direction_accuracy_active",
    "coverage",
    "active_coverage",
    "effect_all_zero_fraction",
    "main_to_active_abs_ratio",
    "empirical_fdr",
    "power",
    "process_wall_seconds",
    "peak_rss_kib",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


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


def _mean_ci(values: Sequence[float], *, confidence: float = 0.95) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    mean = float(array.mean())
    if len(array) == 1:
        return {"n": 1, "mean": mean, "ci_low": math.nan, "ci_high": math.nan}
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    critical = float(t.ppf(0.5 + confidence / 2.0, df=len(array) - 1))
    radius = critical * standard_error
    return {
        "n": len(array),
        "mean": mean,
        "ci_low": mean - radius,
        "ci_high": mean + radius,
    }


def _primary_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset_id",
        "base_method_id",
        "view_label",
        "differential_engine",
        *SUMMARY_METRICS,
    }
    missing = required.difference(metrics.columns)
    if metrics.empty or missing:
        raise ValueError(f"evaluation metrics are invalid: missing={sorted(missing)}")
    unknown = set(PRIMARY_POLICY).difference(metrics["base_method_id"].astype(str))
    if unknown:
        raise ValueError(f"primary methods are absent: {sorted(unknown)}")
    parts: list[pd.DataFrame] = []
    for method, policy in PRIMARY_POLICY.items():
        selected = metrics.loc[
            metrics["base_method_id"].astype(str).eq(method)
            & metrics["view_label"].astype(str).eq(policy["view_label"])
            & metrics["differential_engine"]
            .astype(str)
            .eq(policy["differential_engine"])
        ].copy()
        if selected.empty or selected.duplicated("dataset_id").any():
            raise ValueError(f"primary policy is not one row per dataset: {method}")
        parts.append(selected)
    result = pd.concat(parts, ignore_index=True, sort=False)
    datasets = {
        method: set(group["dataset_id"].astype(str))
        for method, group in result.groupby("base_method_id", observed=True)
    }
    reference = next(iter(datasets.values()))
    if not reference or any(value != reference for value in datasets.values()):
        raise ValueError("primary methods do not share the same replicate set")
    return result.sort_values(
        ["dataset_id", "base_method_id"], kind="stable", ignore_index=True
    )


def _method_summary(primary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, group in primary.groupby("base_method_id", observed=True, sort=True):
        record: dict[str, Any] = {
            "base_method_id": str(method),
            "n_replicates": int(group["dataset_id"].nunique()),
            "formal_did_p_value": bool(group["formal_did_p_value"].all()),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            interval = _mean_ci(values)
            record[f"{metric}_mean"] = interval["mean"]
            record[f"{metric}_ci_low"] = interval["ci_low"]
            record[f"{metric}_ci_high"] = interval["ci_high"]
            finite = values[np.isfinite(values)]
            record[f"{metric}_min"] = float(finite.min()) if len(finite) else math.nan
            record[f"{metric}_max"] = float(finite.max()) if len(finite) else math.nan
        records.append(record)
    result = pd.DataFrame.from_records(records)
    result["omnibus_auprc_rank"] = result["omnibus_auprc_mean"].rank(
        method="min", ascending=False
    )
    return result.sort_values(
        ["omnibus_auprc_rank", "base_method_id"], kind="stable", ignore_index=True
    )


def _paired_differences(
    primary: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    noninferiority_margin: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    crychic = primary.loc[
        primary["base_method_id"].astype(str).eq("crychic"),
        ["dataset_id", "omnibus_auprc"],
    ].rename(columns={"omnibus_auprc": "crychic_auprc"})
    records: list[dict[str, Any]] = []
    for baseline in sorted(set(PRIMARY_POLICY).difference({"crychic"})):
        comparator = primary.loc[
            primary["base_method_id"].astype(str).eq(baseline),
            ["dataset_id", "omnibus_auprc"],
        ].rename(columns={"omnibus_auprc": "baseline_auprc"})
        paired = crychic.merge(
            comparator, on="dataset_id", how="inner", validate="one_to_one"
        )
        differences = (
            pd.to_numeric(paired["crychic_auprc"], errors="raise")
            - pd.to_numeric(paired["baseline_auprc"], errors="raise")
        ).to_numpy(float)
        interval = _mean_ci(differences)
        records.append(
            {
                "baseline_method_id": baseline,
                "n_replicates": interval["n"],
                "mean_auprc_difference": interval["mean"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "crychic_wins": int(np.sum(differences > 0.0)),
                "ties": int(np.sum(differences == 0.0)),
                "noninferiority_margin": noninferiority_margin,
                "noninferior": bool(
                    np.isfinite(interval["ci_low"])
                    and interval["ci_low"] > -noninferiority_margin
                ),
                "superior": bool(
                    np.isfinite(interval["ci_low"]) and interval["ci_low"] > 0.0
                ),
            }
        )
    differences = pd.DataFrame.from_records(records).sort_values(
        "mean_auprc_difference", ascending=False, kind="stable", ignore_index=True
    )
    baseline_summary = summary.loc[
        ~summary["base_method_id"].astype(str).eq("crychic")
    ]
    strongest = str(
        baseline_summary.sort_values(
            "omnibus_auprc_mean", ascending=False, kind="stable"
        ).iloc[0]["base_method_id"]
    )
    gate_row = differences.loc[differences["baseline_method_id"].eq(strongest)].iloc[0]
    gate = {
        "name": "canonical_event_auprc_noninferiority",
        "status": "PASS" if bool(gate_row["noninferior"]) else "REJECT",
        "strongest_baseline": strongest,
        "noninferiority_margin": noninferiority_margin,
        "mean_paired_difference": float(gate_row["mean_auprc_difference"]),
        "ci_low": float(gate_row["ci_low"]),
        "ci_high": float(gate_row["ci_high"]),
    }
    return differences, gate


def summarize(
    metrics_path: Path,
    evaluation_manifest_path: Path,
    fixture_manifest_path: Path,
    output_dir: Path,
    *,
    noninferiority_margin: float = 0.05,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not 0.0 < noninferiority_margin < 1.0:
        raise ValueError("noninferiority margin must lie in (0, 1)")
    paths = [metrics_path, evaluation_manifest_path, fixture_manifest_path]
    paths = [path.expanduser().resolve() for path in paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics_path, evaluation_manifest_path, fixture_manifest_path = paths
    evaluation = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    fixture = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "complete" or fixture.get("status") != "complete":
        raise ValueError("evaluation and fixture manifests must be complete")
    output_record = evaluation.get("outputs", {}).get(metrics_path.name, {})
    if output_record.get("sha256") != sha256_file(metrics_path):
        raise ValueError("metrics checksum is not bound by the evaluation manifest")
    seed_map = {
        str(record["dataset_id"]): int(record["seed"])
        for record in fixture.get("records", [])
    }
    metrics = pd.read_csv(metrics_path, sep="\t")
    primary = _primary_rows(metrics)
    if set(primary["dataset_id"].astype(str)) != set(seed_map):
        raise ValueError("fixture seeds do not match evaluated datasets")
    primary.insert(
        1,
        "seed",
        primary["dataset_id"].astype(str).map(seed_map).astype(int),
    )
    summary = _method_summary(primary)
    paired, gate = _paired_differences(
        primary,
        summary,
        noninferiority_margin=noninferiority_margin,
    )

    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        primary_path = staged / "replicate_primary_metrics.tsv"
        summary_path = staged / "method_summary.tsv"
        paired_path = staged / "paired_auprc_differences.tsv"
        gate_path = staged / "gate.json"
        report_path = staged / "REPORT.md"
        primary.to_csv(primary_path, sep="\t", index=False)
        summary.to_csv(summary_path, sep="\t", index=False)
        paired.to_csv(paired_path, sep="\t", index=False)
        _write_json(gate_path, gate)
        lines = [
            "# BRCA mechanistic holdout summary",
            "",
            "Primary score and engine choices are frozen by method; sensitivity "
            "views are excluded from ranking.",
            "",
            "| Rank | Method | AUPRC mean (95% CI) | AUROC | Direction | Wall s |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.omnibus_auprc_rank:.0f} | {row.base_method_id} | "
                f"{row.omnibus_auprc_mean:.4f} "
                f"[{row.omnibus_auprc_ci_low:.4f}, "
                f"{row.omnibus_auprc_ci_high:.4f}] | "
                f"{row.omnibus_auroc_mean:.4f} | "
                f"{row.direction_accuracy_active_mean:.4f} | "
                f"{row.process_wall_seconds_mean:.2f} |"
            )
        lines.extend(
            [
                "",
                "## Gate C",
                "",
                f"Status: **{gate['status']}** against {gate['strongest_baseline']} "
                f"with margin {noninferiority_margin:.3f}.",
                "",
                "Paired AUPRC difference (CRYCHIC - baseline): "
                f"{gate['mean_paired_difference']:.4f} "
                f"[{gate['ci_low']:.4f}, {gate['ci_high']:.4f}].",
            ]
        )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        output_paths = (
            primary_path,
            summary_path,
            paired_path,
            gate_path,
            report_path,
        )
        outputs = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "primary_policy": PRIMARY_POLICY,
            "confidence_interval": "two-sided Student t, 95%",
            "paired_unit": "semi-synthetic replicate seed",
            "noninferiority_margin": noninferiority_margin,
            "gate": gate,
            "inputs": {
                "metrics": {
                    "path": str(metrics_path),
                    "sha256": sha256_file(metrics_path),
                },
                "evaluation_manifest": {
                    "path": str(evaluation_manifest_path),
                    "sha256": sha256_file(evaluation_manifest_path),
                },
                "fixture_manifest": {
                    "path": str(fixture_manifest_path),
                    "sha256": sha256_file(fixture_manifest_path),
                },
            },
            "outputs": outputs,
            "code": git_metadata(Path(__file__).resolve().parents[2]),
        }
        _write_json(staged / "manifest.json", manifest)
        _publish(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = summarize(
        args.metrics,
        args.evaluation_manifest,
        args.fixture_manifest,
        args.output_dir,
        noninferiority_margin=args.noninferiority_margin,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest["gate"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
