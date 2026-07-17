"""Summarize the Xie IPF cohort with patients as the equal-weight unit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json

COHORT_SCHEMA = "xie-ipf-cohort-evaluation-summary-v1"
PRIMARY_METRICS = (
    "auroc",
    "average_precision",
    "balanced_auprc",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "raw_positive_return_fraction",
)
EVALUATION_OUTPUTS = (
    "bootstrap_summary.tsv",
    "materialized_scores.parquet",
    "negative_sampling_summary.tsv",
    "point_estimates.tsv",
)


def validate_evaluation_task_provenance(
    task_ledger: str | Path,
    samples: pd.DataFrame,
) -> dict[str, object]:
    """Verify every rostered evaluation task and bind its output checksums."""

    ledger_path = Path(task_ledger).expanduser().resolve()
    ledger = pd.read_csv(ledger_path, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "task_id",
        "study_id",
        "sample_id",
        "stage",
        "arm",
        "command_sha256",
        "publish_dir",
        "expected_outputs_json",
        "status",
    }
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"task ledger is missing columns: {sorted(missing)}")
    evaluation = ledger.loc[
        ledger["stage"].eq("evaluate") & ledger["arm"].eq("hcommon_representable")
    ].copy()
    roster_keys = set(
        samples.loc[:, ["study_id", "sample_id"]].itertuples(index=False, name=None)
    )
    task_keys = set(
        evaluation.loc[:, ["study_id", "sample_id"]].itertuples(index=False, name=None)
    )
    if evaluation.duplicated(["study_id", "sample_id"]).any():
        raise ValueError("task ledger has duplicate IPF evaluation tasks")
    if task_keys != roster_keys:
        raise ValueError("task ledger evaluation tasks differ from the sample roster")
    if not evaluation["status"].eq("complete").all():
        raise ValueError("one or more IPF evaluation tasks are not complete")

    digest_lines: list[str] = []
    for row in evaluation.sort_values(
        ["study_id", "sample_id"], kind="stable"
    ).itertuples(index=False):
        output_dir = Path(row.publish_dir).expanduser().resolve()
        task_manifest_path = output_dir / "cohort_task_manifest.json"
        if not task_manifest_path.is_file():
            raise FileNotFoundError(
                f"evaluation task manifest is missing: {task_manifest_path}"
            )
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        identity_matches = (
            task_manifest.get("task_id") == row.task_id
            and task_manifest.get("study_id") == row.study_id
            and task_manifest.get("sample_id") == row.sample_id
            and task_manifest.get("stage") == "evaluate"
            and task_manifest.get("arm") == "hcommon_representable"
            and task_manifest.get("command_sha256") == row.command_sha256
        )
        if task_manifest.get("status") != "complete" or not identity_matches:
            raise ValueError(
                f"evaluation task manifest identity mismatch: {task_manifest_path}"
            )
        expected_outputs = tuple(sorted(json.loads(row.expected_outputs_json)))
        if expected_outputs != EVALUATION_OUTPUTS:
            raise ValueError(f"unexpected evaluation outputs for task {row.task_id}")
        output_records = task_manifest.get("outputs")
        if (
            not isinstance(output_records, dict)
            or tuple(sorted(output_records)) != EVALUATION_OUTPUTS
        ):
            raise ValueError(f"task manifest output set differs for {row.task_id}")
        for relative in EVALUATION_OUTPUTS:
            path = output_dir / relative
            record = output_records[relative]
            if (
                not path.is_file()
                or not isinstance(record, dict)
                or record.get("bytes") != path.stat().st_size
                or record.get("sha256") != sha256_file(path)
            ):
                raise ValueError(f"evaluation output checksum mismatch: {path}")
        digest_lines.append(
            "\t".join(
                (
                    row.study_id,
                    row.sample_id,
                    row.task_id,
                    sha256_file(task_manifest_path),
                )
            )
        )
    aggregate = hashlib.sha256(
        ("\n".join(digest_lines) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "path": str(ledger_path),
        "sha256": sha256_file(ledger_path),
        "evaluation_tasks": len(evaluation),
        "evaluation_task_manifests": len(digest_lines),
        "evaluation_manifest_aggregate_sha256": aggregate,
        "aggregate_canonicalization": (
            "sha256 of study_id, sample_id, task_id, task_manifest_sha256 "
            "TSV lines sorted by study_id/sample_id"
        ),
    }


def load_sample_metrics(
    sample_manifest: str | Path,
    results_root: str | Path,
    *,
    resource_mode: str = "H-common",
    evaluation_arm: str = "evaluation_representable",
    require_all_samples: bool = True,
) -> pd.DataFrame:
    """Load checksum-independent evaluation tables under the frozen sample roster."""

    manifest_path = Path(sample_manifest).expanduser().resolve()
    root = Path(results_root).expanduser().resolve()
    samples = pd.read_csv(manifest_path, sep="\t", dtype=str)
    required = {"study_id", "sample_id", "dataset_id", "status"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"sample manifest is missing columns: {sorted(missing)}")
    if samples.duplicated(["study_id", "sample_id"]).any():
        raise ValueError("sample manifest contains duplicate study/sample keys")
    if not samples["status"].eq("complete").all():
        raise ValueError("sample manifest contains incomplete inputs")
    frames: list[pd.DataFrame] = []
    absent: list[str] = []
    for row in samples.itertuples(index=False):
        evaluation = (
            root / row.study_id / row.sample_id / resource_mode / evaluation_arm
        )
        point_path = evaluation / "point_estimates.tsv"
        negative_path = evaluation / "negative_sampling_summary.tsv"
        if not point_path.is_file() or not negative_path.is_file():
            absent.append(f"{row.study_id}/{row.sample_id}")
            continue
        point = pd.read_csv(point_path, sep="\t")
        point_required = {"dataset", "method", "status", *PRIMARY_METRICS[:-1]}
        # raw_positive_return_fraction is present for this IPF evaluator but is
        # checked separately to keep the error message focused.
        point_required.discard("balanced_auprc")
        missing_point = point_required.difference(point.columns)
        if missing_point or "raw_positive_return_fraction" not in point.columns:
            sample_key = f"{row.study_id}/{row.sample_id}"
            raise ValueError(
                f"IPF point estimates are incomplete for {sample_key}: "
                f"{sorted(missing_point)}"
            )
        if not point["dataset"].astype(str).eq(row.dataset_id).all():
            raise ValueError(
                f"evaluation dataset ID mismatch for {row.study_id}/{row.sample_id}"
            )
        if point["method"].duplicated().any():
            raise ValueError(
                f"duplicate method rows for {row.study_id}/{row.sample_id}"
            )
        negative = pd.read_csv(negative_path, sep="\t")
        negative_required = {"method", "metric", "replicate_mean", "status"}
        if missing_negative := negative_required.difference(negative.columns):
            raise ValueError(
                f"negative-sampling summary is missing: {sorted(missing_negative)}"
            )
        balanced = negative.loc[
            negative["metric"].eq("average_precision")
            & negative["status"].eq("observed"),
            ["method", "replicate_mean"],
        ].rename(columns={"replicate_mean": "balanced_auprc"})
        if balanced["method"].duplicated().any():
            raise ValueError(
                f"duplicate balanced AUPRC rows for {row.study_id}/{row.sample_id}"
            )
        selected = point.merge(balanced, on="method", how="left", validate="one_to_one")
        selected.insert(0, "sample_id", row.sample_id)
        selected.insert(0, "study_id", row.study_id)
        selected.insert(2, "dataset_id", row.dataset_id)
        selected.insert(3, "evaluation_dir", str(evaluation))
        frames.append(selected)
    if absent and require_all_samples:
        preview = ", ".join(absent[:8])
        suffix = "..." if len(absent) > 8 else ""
        raise FileNotFoundError(
            f"missing {len(absent)} sample evaluations: {preview}{suffix}"
        )
    if not frames:
        raise ValueError("no IPF sample evaluations were available")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["study_id", "sample_id", "method"]).any():
        raise ValueError("cohort sample metrics contain duplicate patient/method rows")
    return result


def _metric_long(sample_metrics: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    requested = tuple(metrics)
    missing = set(requested).difference(sample_metrics.columns)
    if missing:
        raise ValueError(f"sample metrics are missing columns: {sorted(missing)}")
    identifiers = ["study_id", "sample_id", "dataset_id", "method", "status"]
    long = sample_metrics.loc[:, [*identifiers, *requested]].melt(
        id_vars=identifiers,
        value_vars=list(requested),
        var_name="metric",
        value_name="estimate",
    )
    long["estimate"] = pd.to_numeric(long["estimate"], errors="coerce")
    return long.loc[
        long["status"].eq("observed") & np.isfinite(long["estimate"])
    ].copy()


def summarize_patient_equal(
    sample_metrics: pd.DataFrame,
    *,
    expected_samples: int,
    metrics: Iterable[str] = PRIMARY_METRICS,
    n_bootstrap: int = 2000,
    seed: int = 20260715,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return patient-equal point, study-stratified, and bootstrap summaries."""

    if n_bootstrap < 1 or not 0 <= seed < 2**32:
        raise ValueError("n_bootstrap and seed are invalid")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    long = _metric_long(sample_metrics, metrics)
    if long.empty:
        raise ValueError("no finite observed patient-level metrics were available")
    grouped = long.groupby(["method", "metric"], sort=True, observed=True)
    point = (
        grouped["estimate"]
        .agg(
            n_evaluable="size",
            patient_equal_mean="mean",
            patient_median="median",
            patient_standard_deviation="std",
            patient_minimum="min",
            patient_maximum="max",
        )
        .reset_index()
    )
    quartiles = grouped["estimate"].quantile([0.25, 0.75]).unstack().reset_index()
    quartiles = quartiles.rename(columns={0.25: "patient_q1", 0.75: "patient_q3"})
    point = point.merge(quartiles, on=["method", "metric"], validate="one_to_one")
    point["patient_iqr"] = point["patient_q3"] - point["patient_q1"]
    point["n_expected"] = int(expected_samples)
    point["patient_coverage_fraction"] = point["n_evaluable"] / expected_samples
    point["status"] = np.where(
        point["n_evaluable"].eq(expected_samples), "complete", "partial_method_coverage"
    )
    point["aggregation_unit"] = "patient_equal_weight"

    study = (
        long.groupby(["method", "metric", "study_id"], sort=True, observed=True)[
            "estimate"
        ]
        .agg(
            n_evaluable="size",
            patient_equal_mean="mean",
            patient_median="median",
            patient_standard_deviation="std",
            patient_minimum="min",
            patient_maximum="max",
        )
        .reset_index()
    )
    study["aggregation_unit"] = "patient_within_study_equal_weight"

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence_level) / 2.0
    bootstrap_records: list[dict[str, object]] = []
    for (method, metric), values in long.groupby(
        ["method", "metric"], sort=True, observed=True
    ):
        strata = [
            group["estimate"].to_numpy(dtype=float)
            for _, group in values.groupby("study_id", sort=True, observed=True)
        ]
        estimates = np.empty(n_bootstrap, dtype=float)
        for index in range(n_bootstrap):
            sampled = [
                rng.choice(stratum, size=len(stratum), replace=True)
                for stratum in strata
            ]
            estimates[index] = float(np.concatenate(sampled).mean())
        bootstrap_records.append(
            {
                "method": method,
                "metric": metric,
                "resampling_scheme": "patient_bootstrap_stratified_by_study",
                "aggregation_unit": "patient_equal_weight",
                "full_sample_estimate": float(values["estimate"].mean()),
                "bootstrap_mean": float(estimates.mean()),
                "bootstrap_standard_deviation": float(estimates.std(ddof=1)),
                "ci_lower": float(np.quantile(estimates, alpha)),
                "ci_upper": float(np.quantile(estimates, 1.0 - alpha)),
                "confidence_level": confidence_level,
                "n_bootstrap": n_bootstrap,
                "n_patients": len(values),
                "n_studies": values["study_id"].nunique(),
                "seed": seed,
                "interval_semantics": (
                    "descriptive_patient_heterogeneity_not_biological_inference"
                ),
            }
        )
    bootstrap = pd.DataFrame.from_records(bootstrap_records)
    return point, study, bootstrap


def write_cohort_summary(
    sample_manifest: str | Path,
    results_root: str | Path,
    output_dir: str | Path,
    *,
    resource_mode: str = "H-common",
    evaluation_arm: str = "evaluation_representable",
    require_all_samples: bool = True,
    n_bootstrap: int = 2000,
    seed: int = 20260715,
    task_ledger: str | Path | None = None,
) -> dict[str, object]:
    sample_manifest_path = Path(sample_manifest).expanduser().resolve()
    result_root = Path(results_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cohort summary: {output}")
    roster = pd.read_csv(sample_manifest_path, sep="\t", dtype=str)
    task_provenance = (
        None
        if task_ledger is None
        else validate_evaluation_task_provenance(task_ledger, roster)
    )
    sample_metrics = load_sample_metrics(
        sample_manifest_path,
        result_root,
        resource_mode=resource_mode,
        evaluation_arm=evaluation_arm,
        require_all_samples=require_all_samples,
    )
    point, study, bootstrap = summarize_patient_equal(
        sample_metrics,
        expected_samples=len(roster),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    output.mkdir(parents=True, exist_ok=False)
    tables = {
        "sample_metrics.tsv": sample_metrics,
        "cohort_patient_equal.tsv": point,
        "metrics_by_study.tsv": study,
        "patient_bootstrap.tsv": bootstrap,
    }
    for name, table in tables.items():
        table.to_csv(output / name, sep="\t", index=False)
    manifest: dict[str, object] = {
        "schema_version": COHORT_SCHEMA,
        "status": "complete"
        if len(sample_metrics[["study_id", "sample_id"]].drop_duplicates())
        == len(roster)
        else "partial",
        "benchmark_scope": (
            "disease-level static interaction ranking against an open-world "
            "curated gold standard"
        ),
        "primary_aggregation": "patient_equal_weight",
        "bootstrap": "patient resampling with replacement within each GEO study",
        "samples_expected": len(roster),
        "samples_evaluated": len(
            sample_metrics[["study_id", "sample_id"]].drop_duplicates()
        ),
        "methods": sorted(sample_metrics["method"].astype(str).unique()),
        "metrics": list(PRIMARY_METRICS),
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "caveats": [
            "gold standard is disease-level and not patient-specific",
            "unlabeled complement is not an experimentally verified negative class",
            "specificity and MCC use operational pseudo-negatives",
            "intervals summarize patient heterogeneity and are not causal inference",
        ],
        "inputs": {
            "sample_manifest": {
                "path": str(sample_manifest_path),
                "sha256": sha256_file(sample_manifest_path),
            }
        },
        "outputs": {
            name: {"rows": len(table), "sha256": sha256_file(output / name)}
            for name, table in tables.items()
        },
    }
    if task_provenance is not None:
        inputs = manifest["inputs"]
        if not isinstance(inputs, dict):
            raise RuntimeError("cohort summary manifest inputs are invalid")
        inputs["task_ledger"] = task_provenance
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_manifest", type=Path)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--resource-mode", default="H-common")
    parser.add_argument("--evaluation-arm", default="evaluation_representable")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--task-ledger", type=Path)
    args = parser.parse_args()
    manifest = write_cohort_summary(
        args.sample_manifest,
        args.results_root,
        args.output_dir,
        resource_mode=args.resource_mode,
        evaluation_arm=args.evaluation_arm,
        require_all_samples=not args.allow_incomplete,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        task_ledger=args.task_ledger,
    )
    print(
        json.dumps(
            {"status": manifest["status"], "samples": manifest["samples_evaluated"]}
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "PRIMARY_METRICS",
    "load_sample_metrics",
    "summarize_patient_equal",
    "validate_evaluation_task_provenance",
    "write_cohort_summary",
]
