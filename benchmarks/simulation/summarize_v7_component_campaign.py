"""Create a compact, checksum-bound report for a v7 component campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import t as t_distribution

from benchmarks.adapters.common import json_safe, sha256_file, write_json
from benchmarks.simulation.run_v7_campaign import CAMPAIGN_SCHEMA_VERSION

SCHEMA_VERSION = "crychic-suggest-next2-v7-component-report-v1"
_SUPPORTED_CAMPAIGN_SCHEMAS = {
    "crychic-suggest-next2-v7-campaign-v2",
    CAMPAIGN_SCHEMA_VERSION,
}
_KEY_METRICS = (
    "event_auprc",
    "event_auroc",
    "event_partial_auroc_fpr_0_10",
    "effect_spearman",
    "direction_accuracy",
    "diagnostic_false_positive_rate_alpha_0_05",
    "score_zero_fraction",
    "score_tie_fraction",
    "score_na_fraction",
)
_E2_HARD_ARMS = (
    "receptor_eligibility_gate",
    "ligand_contrast_gate",
    "family_selection_gate",
    "downstream_support_gate",
)
_E2_ANNOTATION_ARMS = (
    "receptor_eligibility_annotation",
    "ligand_contrast_annotation",
    "family_selection_annotation",
    "downstream_support_annotation",
)


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _descriptive_ci(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) < 2:
        return np.nan, np.nan
    mean = float(numeric.mean())
    half_width = float(
        t_distribution.ppf(0.975, len(numeric) - 1)
        * numeric.std(ddof=1)
        / math.sqrt(len(numeric))
    )
    return mean - half_width, mean + half_width


def _e2_arm_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    selected = metrics.loc[metrics["metric"].isin(_KEY_METRICS)]
    for (arm, metric), group in selected.groupby(
        ["score_view", "metric"], observed=True, sort=True
    ):
        observed = group.loc[group["status"].eq("observed")]
        low, high = _descriptive_ci(observed["value"])
        records.append(
            {
                "score_view": arm,
                "metric": metric,
                "rows": len(group),
                "observed": len(observed),
                "observed_fraction": len(observed) / len(group),
                "mean": observed["value"].mean(),
                "std": observed["value"].std(),
                "median": observed["value"].median(),
                "descriptive_ci_low": low,
                "descriptive_ci_high": high,
            }
        )
    return pd.DataFrame.from_records(records)


def _e2_paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    join = ["dataset_id", "dgp_family", "design_kind", "contrast_name", "metric"]
    base = metrics.loc[metrics["score_view"].eq("base_g3_i1")]
    records: list[dict[str, object]] = []
    for arm in _E2_HARD_ARMS:
        paired = metrics.loc[metrics["score_view"].eq(arm)].merge(
            base,
            on=join,
            suffixes=("_arm", "_base"),
            validate="one_to_one",
        )
        paired = paired.loc[paired["metric"].isin(_KEY_METRICS)]
        for metric, group in paired.groupby("metric", observed=True, sort=True):
            observed = group.loc[
                group["status_arm"].eq("observed") & group["status_base"].eq("observed")
            ].copy()
            delta = observed["value_arm"] - observed["value_base"]
            low, high = _descriptive_ci(delta)
            records.append(
                {
                    "score_view": arm,
                    "metric": metric,
                    "all_pairs": len(group),
                    "observed_pairs": len(delta),
                    "delta_mean": delta.mean(),
                    "delta_std": delta.std(),
                    "delta_median": delta.median(),
                    "descriptive_ci_low": low,
                    "descriptive_ci_high": high,
                }
            )
    return pd.DataFrame.from_records(records)


def _annotation_equality(metrics: pd.DataFrame) -> dict[str, dict[str, object]]:
    join = ["dataset_id", "dgp_family", "design_kind", "contrast_name", "metric"]
    base = metrics.loc[metrics["score_view"].eq("base_g3_i1")]
    result: dict[str, dict[str, object]] = {}
    for arm in _E2_ANNOTATION_ARMS:
        paired = metrics.loc[metrics["score_view"].eq(arm)].merge(
            base,
            on=join,
            suffixes=("_arm", "_base"),
            validate="one_to_one",
        )
        equal = (
            paired["status_arm"].eq(paired["status_base"])
            & paired["reason_code_arm"]
            .fillna("")
            .eq(paired["reason_code_base"].fillna(""))
            & (
                paired["value_arm"].eq(paired["value_base"])
                | (paired["value_arm"].isna() & paired["value_base"].isna())
            )
        )
        result[str(arm)] = {
            "equal_rows": int(equal.sum()),
            "rows": len(equal),
            "exact": bool(equal.all()),
        }
    return result


def _e3_summary(metrics: pd.DataFrame, *, family: str) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["dgp_family"].eq(family) & metrics["status"].eq("observed")
    ]
    return (
        selected.groupby(
            ["candidate_sender_count", "score_view", "metric"],
            observed=True,
            sort=True,
        )["value"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )


def _e3_m2_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    join = [
        "dataset_id",
        "dgp_family",
        "design_kind",
        "contrast_name",
        "candidate_sender_count",
        "metric",
    ]
    m2 = metrics.loc[metrics["score_view"].eq("null_sender_attribution_plus_m2")]
    no_m2 = metrics.loc[metrics["score_view"].eq("null_sender_attribution")]
    paired = m2.merge(
        no_m2,
        on=join,
        suffixes=("_m2", "_no_m2"),
        validate="one_to_one",
    )
    paired = paired.loc[
        paired["status_m2"].eq("observed") & paired["status_no_m2"].eq("observed")
    ].copy()
    paired["delta"] = paired["value_m2"] - paired["value_no_m2"]
    records: list[dict[str, object]] = []
    scopes = {
        "all": paired,
        "candidate_cardinality": paired.loc[
            paired["dgp_family"].eq("candidate_cardinality")
        ],
        "sender_decoy": paired.loc[paired["dgp_family"].eq("sender_decoy")],
    }
    for scope, table in scopes.items():
        for (count, metric), group in table.groupby(
            ["candidate_sender_count", "metric"], observed=True, sort=True
        ):
            low, high = _descriptive_ci(group["delta"])
            records.append(
                {
                    "scope": scope,
                    "candidate_sender_count": count,
                    "metric": metric,
                    "pairs": len(group),
                    "delta_mean": group["delta"].mean(),
                    "delta_std": group["delta"].std(),
                    "delta_median": group["delta"].median(),
                    "descriptive_ci_low": low,
                    "descriptive_ci_high": high,
                    "minimum": group["delta"].min(),
                    "maximum": group["delta"].max(),
                }
            )
    return pd.DataFrame.from_records(records)


def _main_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["status"].eq("observed")
        & metrics["metric"].isin(_KEY_METRICS)
        & (
            (
                metrics["generator_id"].eq("G0")
                & metrics["score_view"].eq("primary_sender_detection")
                & metrics["inference_id"].eq("I1")
            )
            | (
                metrics["generator_id"].eq("G3")
                & metrics["score_view"].isin(
                    ["primary_sender_detection", "parent_mean"]
                )
                & metrics["inference_id"].isin(["I1", "I2"])
            )
            | (
                metrics["generator_id"].eq("G4")
                & metrics["score_view"].isin(
                    ["primary_fixed_effect_z_blend", "program_signed_component"]
                )
                & metrics["inference_id"].eq("I1")
            )
        )
    ]
    return (
        selected.groupby(
            ["generator_id", "score_view", "inference_id", "metric"],
            observed=True,
            sort=True,
        )["value"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )


def _runtime_tables(
    runs: pd.DataFrame,
    plan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = runs.merge(
        plan.loc[
            :,
            [
                "dataset_id",
                "candidate_sender_count",
                "cells_per_type",
                "subjects_per_level",
            ],
        ].drop_duplicates(),
        on=[
            "dataset_id",
            "candidate_sender_count",
            "cells_per_type",
            "subjects_per_level",
        ],
        validate="one_to_one",
    )
    candidate = merged.loc[merged["dgp_family"].eq("candidate_cardinality")]
    runtime = (
        candidate.groupby("candidate_sender_count", observed=True, sort=True)[
            "elapsed_seconds"
        ]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    records: list[dict[str, object]] = []
    for result_directory, count in candidate.loc[
        :, ["result_directory", "candidate_sender_count"]
    ].itertuples(index=False, name=None):
        manifest = _read_json(Path(str(result_directory)) / "manifest.json")
        for stage in cast(list[Mapping[str, object]], manifest["stage_timings"]):
            records.append(
                {
                    "candidate_sender_count": count,
                    "stage": stage["stage"],
                    "seconds": stage["seconds"],
                }
            )
    stage_frame = pd.DataFrame.from_records(records)
    stages = (
        stage_frame.groupby(
            ["candidate_sender_count", "stage"], observed=True, sort=True
        )["seconds"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    return runtime, stages


def _lookup_mean(
    summary: pd.DataFrame,
    *,
    generator: str,
    score_view: str,
    inference: str,
    metric: str,
) -> float:
    selected = summary.loc[
        summary["generator_id"].eq(generator)
        & summary["score_view"].eq(score_view)
        & summary["inference_id"].eq(inference)
        & summary["metric"].eq(metric),
        "mean",
    ]
    if len(selected) != 1:
        raise ValueError(f"main summary lacks {generator}/{score_view}/{metric}")
    return float(selected.iloc[0])


def _gate_summary(
    *,
    main: pd.DataFrame,
    e2_deltas: pd.DataFrame,
    annotation_equality: Mapping[str, Mapping[str, object]],
    e3_cardinality: pd.DataFrame,
    main_metrics: pd.DataFrame,
) -> dict[str, object]:
    g0_ap = _lookup_mean(
        main,
        generator="G0",
        score_view="primary_sender_detection",
        inference="I1",
        metric="event_auprc",
    )
    g3_ap = _lookup_mean(
        main,
        generator="G3",
        score_view="primary_sender_detection",
        inference="I1",
        metric="event_auprc",
    )
    g3_spearman = _lookup_mean(
        main,
        generator="G3",
        score_view="primary_sender_detection",
        inference="I1",
        metric="effect_spearman",
    )
    top1 = e3_cardinality.loc[
        e3_cardinality["score_view"].eq("null_sender_attribution_plus_m2")
        & e3_cardinality["metric"].eq("top1_sender_accuracy")
    ]
    m1_direction = main_metrics.loc[
        main_metrics["dgp_family"].isin(["program_only", "inhibitory_program"])
        & main_metrics["generator_id"].eq("G4")
        & main_metrics["score_view"].eq("program_signed_component")
        & main_metrics["inference_id"].eq("I1")
        & main_metrics["metric"].eq("direction_accuracy")
        & main_metrics["status"].eq("observed"),
        "value",
    ]
    generic_fpr = main_metrics.loc[
        main_metrics["dgp_family"].eq("generic_state_null")
        & main_metrics["generator_id"].eq("G4")
        & main_metrics["score_view"].eq("program_signed_component")
        & main_metrics["inference_id"].eq("I1")
        & main_metrics["metric"].eq("diagnostic_false_positive_rate_alpha_0_05")
        & main_metrics["status"].eq("observed"),
        "value",
    ]
    downstream = e2_deltas.loc[
        e2_deltas["score_view"].eq("downstream_support_gate")
        & e2_deltas["metric"].eq("event_auprc")
    ].iloc[0]
    checks = {
        "e2_annotation_arms_exact": all(
            bool(value["exact"]) for value in annotation_equality.values()
        ),
        "e2_downstream_hard_gate_harms_auprc": bool(
            float(downstream["descriptive_ci_high"]) < 0.0
        ),
        "m0_auprc_regression_within_0_02": bool(g3_ap >= g0_ap - 0.02),
        "m0_effect_spearman_at_least_0_60": bool(g3_spearman >= 0.60),
        "m1_direction_accuracy_at_least_0_85": bool(
            len(m1_direction) and float(m1_direction.mean()) >= 0.85
        ),
        "m1_generic_state_diagnostic_fpr_not_above_0_05": bool(
            len(generic_fpr) and float(generic_fpr.mean()) <= 0.05
        ),
        "m2_top1_at_least_0_80_at_every_candidate_count": bool(
            len(top1) == 4 and top1["mean"].ge(0.80).all()
        ),
    }
    return {
        "status": "SMOKE_COMPLETE_WITH_RELEASE_GAPS",
        "checks": checks,
        "estimates": {
            "g0_i1_event_auprc": g0_ap,
            "g3_i1_event_auprc": g3_ap,
            "g3_minus_g0_event_auprc": g3_ap - g0_ap,
            "g3_i1_effect_spearman": g3_spearman,
            "m1_program_direction_accuracy": float(m1_direction.mean()),
            "m1_generic_state_diagnostic_fpr": float(generic_fpr.mean()),
            "m2_minimum_candidate_count_top1": float(top1["mean"].min()),
        },
        "not_evaluated": [
            "Kuppe and locked MS real-data endpoints",
            "M4 formal occurrence calibration",
            "E5 integrated topology controls",
            "full-pipeline 1000-replicate null calibration",
            "development and locked family holdouts",
        ],
    }


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
    """Validate one completed campaign and publish compact tracked evidence."""

    campaign = campaign_dir.resolve()
    manifest_path = campaign / "campaign_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") not in _SUPPORTED_CAMPAIGN_SCHEMAS:
        raise ValueError("campaign schema is not a supported component contract")
    if manifest.get("status") != "completed" or manifest.get("failed_datasets") != 0:
        raise ValueError("component report requires a failure-free campaign")
    if manifest.get("planned_datasets") != 900:
        raise ValueError("component smoke report requires the frozen 900 datasets")
    runs_path = campaign / "runs.tsv"
    run_record = cast(Mapping[str, object], manifest["runs"])
    if sha256_file(runs_path) != run_record["sha256"]:
        raise ValueError("runs.tsv checksum differs from the campaign manifest")
    required = {
        "main": campaign / "all_dataset_metrics.parquet",
        "e2": campaign / "all_e2_metrics.parquet",
        "e3": campaign / "all_e3_sender_metrics.parquet",
        "plan": campaign / "run_plan.tsv",
    }
    if any(not path.is_file() for path in required.values()):
        raise FileNotFoundError("campaign lacks one or more aggregate metric tables")

    main_metrics = pd.read_parquet(required["main"])
    e2_metrics = pd.read_parquet(required["e2"])
    e3_metrics = pd.read_parquet(required["e3"])
    runs = pd.read_csv(runs_path, sep="\t")
    plan = pd.read_csv(required["plan"], sep="\t")
    e2_summary = _e2_arm_summary(e2_metrics)
    e2_deltas = _e2_paired_deltas(e2_metrics)
    annotation = _annotation_equality(e2_metrics)
    e3_cardinality = _e3_summary(e3_metrics, family="candidate_cardinality")
    e3_decoy = _e3_summary(e3_metrics, family="sender_decoy")
    e3_deltas = _e3_m2_deltas(e3_metrics)
    main = _main_summary(main_metrics)
    runtime, stages = _runtime_tables(runs, plan)
    gates = _gate_summary(
        main=main,
        e2_deltas=e2_deltas,
        annotation_equality=annotation,
        e3_cardinality=e3_cardinality,
        main_metrics=main_metrics,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.stage-", dir=output_dir.parent
    ) as temporary:
        staged = Path(temporary) / output_dir.name
        staged.mkdir()
        tables = {
            "main_score_summary.tsv": main,
            "e2_arm_summary.tsv": e2_summary,
            "e2_paired_deltas.tsv": e2_deltas,
            "e3_candidate_cardinality.tsv": e3_cardinality,
            "e3_sender_decoy.tsv": e3_decoy,
            "e3_m2_paired_deltas.tsv": e3_deltas,
            "runtime_candidate_cardinality.tsv": runtime,
            "runtime_stage_candidate_cardinality.tsv": stages,
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False)
        write_json(staged / "acceptance.json", gates)
        estimate = cast(Mapping[str, float], gates["estimates"])
        checks = cast(Mapping[str, bool], gates["checks"])
        gate_lines = "\n".join(
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in checks.items()
        )
        readme = f"""# Suggest-next2 v7 component smoke report

This report is compact descriptive evidence from the frozen 20-seed smoke
campaign: 900/900 datasets completed with zero process failures. It is not
formal release evidence; analytic I1/I2 p-values remain diagnostic.

## Main estimator

- G3-I1 sender effect Spearman: {estimate["g3_i1_effect_spearman"]:.4f}
- G3-I1 event AUPRC: {estimate["g3_i1_event_auprc"]:.4f}
- G0-I1 event AUPRC: {estimate["g0_i1_event_auprc"]:.4f}
- G3 minus G0 event AUPRC: {estimate["g3_minus_g0_event_auprc"]:.4f}

G3 improves effect recovery substantially over G0 and remains within the
pre-registered 0.02 AUPRC non-inferiority margin, but its mean effect Spearman
does not yet clear the 0.60 release target.

## E2 hard-gate swaps

All four annotation-only arms are exact copies of G3 for every persisted
metric row. Ligand-contrast, family-selection, and downstream hard gates cause
large zero/tie inflation and loss of estimability. The downstream gate has a
negative paired AUPRC interval; receptor eligibility changes coverage geometry
without changing observed effect rankings in this smoke suite.

## E3 sender swaps

Nonconserving detection is the strongest sender-identity score. Null-sender
attribution controls diagnostic false positives, but its top-1 accuracy drops
at 10 and 20 candidates. Adding EB-shrunken M2 coupling does not materially
change rankings and fails the all-cardinality top-1 >= 0.80 gate.

## Gate status

{gate_lines}

See the TSV files for counts, descriptive intervals, paired deltas, and stage
timings. Raw per-dataset outputs remain outside Git.
"""
        (staged / "README.md").write_text(readme, encoding="utf-8")
        first_dataset = next((campaign / "datasets").glob("*/manifest.json"))
        dataset_manifest = _read_json(first_dataset)
        files = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staged.iterdir())
        }
        report_manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "source_campaign": campaign.name,
            "source_campaign_manifest_sha256": sha256_file(manifest_path),
            "source_estimator_commit": dataset_manifest["runtime"]["git"]["commit"],
            "source_protocol_digest": manifest["protocol"]["protocol_digest"],
            "planned_datasets": manifest["planned_datasets"],
            "completed_datasets": manifest["completed_datasets"],
            "failed_datasets": manifest["failed_datasets"],
            "annotation_equality": annotation,
            "acceptance": gates,
            "source_aggregates": {
                name: {"filename": path.name, "sha256": sha256_file(path)}
                for name, path in required.items()
            },
            "files": files,
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
