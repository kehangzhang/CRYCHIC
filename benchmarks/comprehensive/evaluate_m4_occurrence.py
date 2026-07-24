"""Evaluate the isolated M4 occurrence head against continuous and exact baselines."""

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
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.generate_m4_occurrence_fixture import (
    SCHEMA_VERSION as FIXTURE_SCHEMA,
)
from crychic.inference import (
    OCCURRENCE_CONTRAST_VERSION,
    OccurrenceContrastSpec,
    fit_subject_occurrence_contrasts,
)

SCHEMA_VERSION = "crychic-m4-occurrence-evaluation-v1"
CONFIG_SCHEMA_VERSION = "crychic-suggestions-next-m4-config-v1"
M0_METHOD = "crychic_m0_continuous_activity_contrast"
RAW_METHOD = "raw_subject_prevalence_difference"
FISHER_METHOD = "dcst_style_fisher_exact_occurrence"
M4_METHOD = "crychic_m4_posterior_standardized_prevalence"
METHOD_SCORE_COLUMNS: Mapping[str, str] = {
    M0_METHOD: "m0_ranking_score",
    RAW_METHOD: "raw_prevalence_ranking_score",
    FISHER_METHOD: "fisher_ranking_score",
    M4_METHOD: "m4_ranking_score",
}
METHOD_DIRECTION_COLUMNS: Mapping[str, str] = {
    M0_METHOD: "m0_direction",
    RAW_METHOD: "raw_prevalence_direction",
    FISHER_METHOD: "raw_prevalence_direction",
    M4_METHOD: "m4_direction",
}


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


def _validated_config(
    path: Path, *, role: str, fixture_manifest: Path
) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("M4 configuration schema is unsupported")
    expected_candidate = {
        "name": M4_METHOD,
        "score_version": OCCURRENCE_CONTRAST_VERSION,
        "design": "independent_groups",
        "beta_prior": 0.5,
        "minimum_subjects_per_group": 8,
        "ranking_score": "absolute_posterior_standardized_prevalence_difference",
        "missing_occurrence_policy": "missing_not_absent",
        "tuned_parameters": 0,
        "formal_inference_allowed": False,
    }
    candidate = config.get("candidate")
    if not isinstance(candidate, Mapping) or any(
        candidate.get(key) != value for key, value in expected_candidate.items()
    ):
        raise ValueError("M4 candidate identity or fixed policy changed")
    boundary = config.get("claim_boundary")
    if not isinstance(boundary, Mapping) or not all(
        boundary.get(field) is True
        for field in (
            "occurrence_is_separate_from_continuous_magnitude",
            "adr013_active_probability_is_not_reused",
            "fisher_p_values_are_baseline_only",
            "m4_emits_no_p_or_q_values",
            "m4_emits_no_event_activity_probability",
            "synthetic_occurrence_fixture_is_not_general_sota_evidence",
        )
    ):
        raise ValueError("M4 claim boundary changed")
    if role not in {"development", "validation"}:
        raise ValueError("role must be development or validation")
    section = config.get(role)
    if not isinstance(section, Mapping):
        raise ValueError(f"M4 configuration lacks {role} section")
    expected_sha = section.get("fixture_manifest_sha256")
    if not isinstance(expected_sha, str) or expected_sha.startswith("PENDING"):
        raise ValueError(f"{role} fixture checksum is not frozen")
    if sha256_file(fixture_manifest) != expected_sha:
        raise ValueError(f"{role} fixture manifest checksum differs")
    return config


def _group_summary(table: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        table.groupby(["event_id", "condition"], observed=True, sort=True)[value]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )


def _raw_log_odds(
    reference_occurrences: int,
    reference_subjects: int,
    target_occurrences: int,
    target_subjects: int,
) -> float:
    reference = reference_occurrences / reference_subjects
    target = target_occurrences / target_subjects
    reference_epsilon = 0.5 / reference_subjects
    target_epsilon = 0.5 / target_subjects
    reference = float(np.clip(reference, reference_epsilon, 1.0 - reference_epsilon))
    target = float(np.clip(target, target_epsilon, 1.0 - target_epsilon))
    return math.log(target / (1.0 - target)) - math.log(reference / (1.0 - reference))


def _evaluate_dataset(
    dataset_id: str,
    root_seed: int,
    input_path: Path,
    input_sha256: str,
    truth: pd.DataFrame,
    spec: OccurrenceContrastSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sha256_file(input_path) != input_sha256:
        raise ValueError(f"M4 input checksum mismatch: {dataset_id}")
    events = pd.read_csv(input_path, sep="\t")
    occurrence = fit_subject_occurrence_contrasts(
        events,
        reference="ctrl",
        target="stim",
        spec=spec,
    )
    activity = _group_summary(events, "m0_activity")
    activity_wide = activity.pivot(index="event_id", columns="condition", values="mean")
    m0_effect = activity_wide["stim"] - activity_wide["ctrl"]

    observed = events.dropna(subset=["occurrence"])
    binary = _group_summary(observed, "occurrence")
    count_wide = binary.pivot(index="event_id", columns="condition", values="count")
    sum_wide = binary.pivot(index="event_id", columns="condition", values="sum")
    raw_prevalence = binary.pivot(index="event_id", columns="condition", values="mean")
    rows: list[dict[str, object]] = []
    for event_id in occurrence["event_id"]:
        reference_n = int(count_wide.loc[event_id, "ctrl"])
        target_n = int(count_wide.loc[event_id, "stim"])
        reference_k = int(sum_wide.loc[event_id, "ctrl"])
        target_k = int(sum_wide.loc[event_id, "stim"])
        fisher = fisher_exact(
            [
                [target_k, target_n - target_k],
                [reference_k, reference_n - reference_k],
            ],
            alternative="two-sided",
        )
        rows.append(
            {
                "event_id": event_id,
                "m0_activity_effect": float(m0_effect.loc[event_id]),
                "raw_reference_prevalence": float(raw_prevalence.loc[event_id, "ctrl"]),
                "raw_target_prevalence": float(raw_prevalence.loc[event_id, "stim"]),
                "raw_prevalence_difference": float(
                    raw_prevalence.loc[event_id, "stim"]
                    - raw_prevalence.loc[event_id, "ctrl"]
                ),
                "raw_log_odds_ratio": _raw_log_odds(
                    reference_k, reference_n, target_k, target_n
                ),
                "fisher_odds_ratio": float(fisher.statistic),
                "fisher_p_value": float(fisher.pvalue),
            }
        )
    baseline = pd.DataFrame.from_records(rows)
    scores = (
        truth.merge(occurrence, on="event_id", validate="one_to_one")
        .merge(baseline, on="event_id", validate="one_to_one")
        .sort_values("event_id", kind="stable", ignore_index=True)
    )
    scores.insert(0, "root_seed", root_seed)
    scores.insert(0, "dataset_id", dataset_id)
    scores["m0_ranking_score"] = scores["m0_activity_effect"].abs()
    scores["raw_prevalence_ranking_score"] = scores["raw_prevalence_difference"].abs()
    scores["fisher_ranking_score"] = -np.log10(
        np.clip(scores["fisher_p_value"].to_numpy(dtype=float), 1.0e-300, 1.0)
    )
    scores["m4_ranking_score"] = scores[
        "posterior_standardized_prevalence_difference"
    ].abs()
    scores["m0_direction"] = np.sign(scores["m0_activity_effect"]).astype(int)
    scores["raw_prevalence_direction"] = np.sign(
        scores["raw_prevalence_difference"]
    ).astype(int)
    scores["m4_direction"] = scores["direction"]

    calibration_rows: list[dict[str, object]] = []
    for row in scores.itertuples(index=False):
        for condition, truth_column, raw_column, m4_column in (
            (
                "ctrl",
                "true_reference_prevalence",
                "raw_reference_prevalence",
                "reference_posterior_prevalence",
            ),
            (
                "stim",
                "true_target_prevalence",
                "raw_target_prevalence",
                "target_posterior_prevalence",
            ),
        ):
            true_value = float(getattr(row, truth_column))
            for method, prediction_column in (
                (RAW_METHOD, raw_column),
                (M4_METHOD, m4_column),
            ):
                prediction = float(getattr(row, prediction_column))
                calibration_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "root_seed": root_seed,
                        "event_id": str(row.event_id),
                        "condition": condition,
                        "method": method,
                        "truth": true_value,
                        "prediction": prediction,
                        "squared_error": (prediction - true_value) ** 2,
                    }
                )
    calibration = pd.DataFrame.from_records(calibration_rows)
    odds = scores.loc[
        :,
        [
            "dataset_id",
            "root_seed",
            "event_id",
            "expected_occurrence_differential",
            "true_log_odds_ratio",
            "raw_log_odds_ratio",
            "posterior_log_odds_ratio",
        ],
    ].copy()
    return scores, calibration, odds


def _seed_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed, table in scores.groupby("root_seed", observed=True, sort=True):
        labels = table["expected_occurrence_differential"].astype(int).to_numpy()
        positive = labels == 1
        expected_direction = table["expected_direction"].to_numpy(dtype=int)
        for method, score_column in METHOD_SCORE_COLUMNS.items():
            values = pd.to_numeric(table[score_column], errors="coerce")
            coverage = float(values.notna().mean())
            valid = values.notna().to_numpy()
            if len(np.unique(labels[valid])) != 2:
                ap = np.nan
                auroc = np.nan
            else:
                ap = float(average_precision_score(labels[valid], values[valid]))
                auroc = float(roc_auc_score(labels[valid], values[valid]))
            direction = pd.to_numeric(
                table[METHOD_DIRECTION_COLUMNS[method]], errors="coerce"
            ).to_numpy(dtype=float)
            direction_valid = positive & np.isfinite(direction)
            rows.append(
                {
                    "root_seed": int(seed),
                    "method": method,
                    "event_coverage": coverage,
                    "average_precision": ap,
                    "auroc": auroc,
                    "direction_accuracy": float(
                        np.mean(
                            direction[direction_valid]
                            == expected_direction[direction_valid]
                        )
                    ),
                    "null_mean_score": float(
                        values.loc[~table["expected_occurrence_differential"]].mean()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["method", "root_seed"], kind="stable", ignore_index=True
    )


def _calibration_metrics(calibration: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in sorted(calibration["root_seed"].unique()):
        seed_calibration = calibration.loc[calibration["root_seed"].eq(seed)]
        seed_odds = odds.loc[odds["root_seed"].eq(seed)]
        for method, odds_column in (
            (RAW_METHOD, "raw_log_odds_ratio"),
            (M4_METHOD, "posterior_log_odds_ratio"),
        ):
            selected = seed_calibration.loc[seed_calibration["method"].eq(method)]
            error = seed_odds[odds_column].to_numpy(dtype=float) - seed_odds[
                "true_log_odds_ratio"
            ].to_numpy(dtype=float)
            finite_error = error[np.isfinite(error)]
            if not len(finite_error):
                raise ValueError("M4 log-odds diagnostics have no finite events")
            rows.append(
                {
                    "root_seed": int(seed),
                    "method": method,
                    "prevalence_brier": float(selected["squared_error"].mean()),
                    "log_odds_rmse": float(np.sqrt(np.mean(finite_error**2))),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["method", "root_seed"], kind="stable", ignore_index=True
    )


def _method_summary(
    seed_metrics: pd.DataFrame, calibration_metrics: pd.DataFrame
) -> pd.DataFrame:
    ranking = (
        seed_metrics.groupby("method", observed=True, sort=True)
        .agg(
            seeds=("root_seed", "nunique"),
            event_coverage=("event_coverage", "mean"),
            average_precision=("average_precision", "mean"),
            auroc=("auroc", "mean"),
            direction_accuracy=("direction_accuracy", "mean"),
            null_mean_score=("null_mean_score", "mean"),
        )
        .reset_index()
    )
    calibration = (
        calibration_metrics.groupby("method", observed=True, sort=True)
        .agg(
            prevalence_brier=("prevalence_brier", "mean"),
            log_odds_rmse=("log_odds_rmse", "mean"),
        )
        .reset_index()
    )
    result = ranking.merge(calibration, on="method", how="left", validate="one_to_one")
    result["average_precision_rank"] = result["average_precision"].rank(
        method="min", ascending=False
    )
    result["auroc_rank"] = result["auroc"].rank(method="min", ascending=False)
    return result.sort_values(
        ["average_precision_rank", "auroc_rank", "method"],
        kind="stable",
        ignore_index=True,
    )


def _paired_bootstrap(
    table: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    left = table.loc[table["method"].eq(candidate), ["root_seed", metric]]
    right = table.loc[table["method"].eq(baseline), ["root_seed", metric]]
    paired = left.merge(
        right,
        on="root_seed",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    ).dropna()
    differences = (
        paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
    ).to_numpy(dtype=float)
    if len(differences) < 2:
        raise ValueError("M4 paired bootstrap requires at least two seeds")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(replicates, len(differences)))
    bootstrap = differences[draws].mean(axis=1)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "paired_seeds": int(len(differences)),
        "mean_delta": float(differences.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "win_fraction": float((differences > 0.0).mean()),
    }


def _gate(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    role: str,
) -> dict[str, object]:
    gates = cast(Mapping[str, Any], config["gates"])
    section = cast(Mapping[str, Any], config[role])
    indexed = summary.set_index("method")
    m4 = indexed.loc[M4_METHOD]

    def row(baseline: str, metric: str) -> pd.Series:
        selected = comparisons.loc[
            comparisons["baseline"].eq(baseline) & comparisons["metric"].eq(metric)
        ]
        if len(selected) != 1:
            raise ValueError(f"missing M4 comparison: {baseline}, {metric}")
        return selected.iloc[0]

    ap_m0 = row(M0_METHOD, "average_precision")
    auroc_m0 = row(M0_METHOD, "auroc")
    ap_fisher = row(FISHER_METHOD, "average_precision")
    brier_raw = row(RAW_METHOD, "prevalence_brier")
    odds_raw = row(RAW_METHOD, "log_odds_rmse")
    checks = {
        "minimum_paired_seeds": int(ap_m0["paired_seeds"])
        >= int(section["minimum_paired_seeds"]),
        "ap_gain_vs_continuous_m0": float(ap_m0["ci_low"])
        > float(gates["ap_delta_vs_m0_ci_lower_exclusive"]),
        "auroc_gain_vs_continuous_m0": float(auroc_m0["ci_low"])
        > float(gates["auroc_delta_vs_m0_ci_lower_exclusive"]),
        "fisher_noninferiority": float(ap_fisher["ci_low"])
        >= float(gates["ap_delta_vs_fisher_ci_lower_minimum"]),
        "prevalence_calibration_gain": float(brier_raw["ci_high"])
        < float(gates["brier_delta_vs_raw_ci_upper_exclusive"]),
        "log_odds_noninferiority": float(odds_raw["ci_high"])
        <= float(gates["log_odds_rmse_delta_vs_raw_ci_upper_maximum"]),
        "coverage": float(m4["event_coverage"])
        >= float(gates["minimum_event_coverage"]),
        "direction": float(m4["direction_accuracy"])
        >= float(gates["minimum_direction_accuracy"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "status": ("DEVELOPMENT_PASS" if role == "development" else "ACCEPT")
        if all(checks.values())
        else "REJECT",
        "checks": checks,
        "claim_boundary": "synthetic independent-subject occurrence estimand only",
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


def evaluate(
    fixture_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    role: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the locked M4 occurrence benchmark and acceptance gates."""

    started = time.perf_counter()
    fixture_dir = fixture_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    fixture_manifest_path = fixture_dir / "manifest.json"
    fixture = _read_json(fixture_manifest_path)
    if fixture.get("schema_version") != FIXTURE_SCHEMA:
        raise ValueError("M4 fixture schema is unsupported")
    config = _validated_config(
        config_path, role=role, fixture_manifest=fixture_manifest_path
    )
    section = cast(Mapping[str, Any], config[role])
    if list(map(int, fixture.get("seeds", []))) != list(
        map(int, cast(Sequence[int], section["seeds"]))
    ):
        raise ValueError("M4 fixture seeds differ from frozen role seeds")
    truth_record = cast(Mapping[str, Any], fixture["truth"])
    truth_path = fixture_dir / str(truth_record["filename"])
    if sha256_file(truth_path) != truth_record["sha256"]:
        raise ValueError("M4 truth checksum differs")
    truth = pd.read_csv(truth_path, sep="\t")
    candidate = cast(Mapping[str, Any], config["candidate"])
    spec = OccurrenceContrastSpec(
        beta_prior=float(candidate["beta_prior"]),
        minimum_subjects_per_group=int(candidate["minimum_subjects_per_group"]),
        design=str(candidate["design"]),
    )
    score_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    odds_frames: list[pd.DataFrame] = []
    for record in cast(Sequence[Mapping[str, Any]], fixture["records"]):
        dataset_id = str(record["dataset_id"])
        dataset_truth = truth.loc[truth["dataset_id"].eq(dataset_id)].drop(
            columns=["dataset_id", "root_seed"]
        )
        scores, calibration, odds = _evaluate_dataset(
            dataset_id,
            int(record["root_seed"]),
            fixture_dir / str(record["input"]),
            str(record["input_sha256"]),
            dataset_truth,
            spec,
        )
        score_frames.append(scores)
        calibration_frames.append(calibration)
        odds_frames.append(odds)
    scores = pd.concat(score_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    odds = pd.concat(odds_frames, ignore_index=True)
    seed_metrics = _seed_metrics(scores)
    calibration_metrics = _calibration_metrics(calibration, odds)
    summary = _method_summary(seed_metrics, calibration_metrics)
    comparisons = pd.DataFrame.from_records(
        [
            _paired_bootstrap(
                table,
                candidate=M4_METHOD,
                baseline=baseline,
                metric=metric,
                replicates=int(section["bootstrap_replicates"]),
                seed=int(section["bootstrap_seed"]) + index,
            )
            for index, (table, baseline, metric) in enumerate(
                (
                    (seed_metrics, M0_METHOD, "average_precision"),
                    (seed_metrics, M0_METHOD, "auroc"),
                    (seed_metrics, FISHER_METHOD, "average_precision"),
                    (calibration_metrics, RAW_METHOD, "prevalence_brier"),
                    (calibration_metrics, RAW_METHOD, "log_odds_rmse"),
                )
            )
        ]
    )
    acceptance = _gate(summary, comparisons, config=config, role=role)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        tables = {
            "event_scores.tsv": scores,
            "prevalence_calibration.tsv": calibration,
            "odds_diagnostics.tsv": odds,
            "seed_metrics.tsv": seed_metrics,
            "calibration_metrics.tsv": calibration_metrics,
            "method_summary.tsv": summary,
            "paired_comparisons.tsv": comparisons,
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False, lineterminator="\n")
        _write_json(staged / "acceptance.json", acceptance)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "role": role,
            "acceptance": acceptance,
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "fixture": {
                "path": str(fixture_dir),
                "manifest_sha256": sha256_file(fixture_manifest_path),
            },
            "configuration": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "candidate": dict(candidate),
            "datasets": len(score_frames),
            "events": int(len(scores)),
            "elapsed_seconds": time.perf_counter() - started,
            "tables": {
                filename: {
                    "sha256": sha256_file(staged / filename),
                    "rows": int(len(table)),
                }
                for filename, table in tables.items()
            },
            "acceptance_sha256": sha256_file(staged / "acceptance.json"),
            "claim_boundary": {
                "formal_inference_allowed": False,
                "general_sota_claim_allowed": False,
                "actual_dcst_package_run": False,
                "fisher_baseline_semantics": "DCST-style two-sided exact occurrence",
            },
        }
        _write_json(staged / "manifest.json", manifest)
        _publish(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "validation"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = evaluate(
        args.fixture_dir,
        args.config,
        args.output_dir,
        role=args.role,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
