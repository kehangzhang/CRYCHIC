"""Independent pair-level simulation benchmark for RC10 rank contrast."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)
from benchmarks.literature.paired_rank_contrast import paired_rank_contrast


def _read_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RC10 paired rank config must contain an object")
    return cast(dict[str, Any], value)


def _portable_config_path(path: Path) -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError("RC10 config must be stored inside the repository") from error


def simulate_pair_problem(
    config: Mapping[str, Any], *, scenario: str, seed: int
) -> pd.DataFrame:
    """Generate paired condition scores with known condition-specific truth."""

    simulation = cast(Mapping[str, Any], config["simulation"])
    allowed = {str(value) for value in simulation["scenarios"]}
    if scenario not in allowed:
        raise ValueError(f"unknown RC10 pair scenario: {scenario}")
    pair_count = (
        int(simulation["small_axis_pair_count"])
        if scenario == "small_axis_shared"
        else int(simulation["pair_count"])
    )
    rng = np.random.default_rng(seed)
    noise_sd = float(simulation["observation_noise_sd"])
    low_truth = rng.gamma(shape=0.7, scale=0.35, size=pair_count)
    high_truth = rng.gamma(shape=2.5, scale=2.0, size=pair_count)
    orientation = rng.random(pair_count) < 0.5

    if scenario in {
        "mutually_exclusive",
        "shared_opportunity",
        "small_axis_shared",
        "structural_missingness",
    }:
        target_truth = np.where(orientation, high_truth, low_truth)
        reference_truth = np.where(orientation, low_truth, high_truth)
    elif scenario in {
        "independent_bidirectional",
        "low_noise_independent",
    }:
        target_truth = rng.gamma(shape=2.0, scale=1.5, size=pair_count)
        reference_truth = rng.gamma(shape=2.0, scale=1.5, size=pair_count)
    elif scenario == "balanced_bidirectional":
        latent = rng.gamma(shape=2.0, scale=1.8, size=pair_count)
        target_truth = latent + rng.gamma(shape=0.5, scale=0.15, size=pair_count)
        reference_truth = latent + rng.gamma(shape=0.5, scale=0.15, size=pair_count)
    elif scenario == "asymmetric_bidirectional":
        latent = rng.gamma(shape=2.0, scale=1.8, size=pair_count)
        target_truth = 3.0 * latent + rng.gamma(
            shape=0.5, scale=0.2, size=pair_count
        )
        reference_truth = 0.6 * latent + rng.gamma(
            shape=0.5, scale=0.2, size=pair_count
        )
    else:
        target_truth = np.zeros(pair_count, dtype=float)
        reference_truth = np.zeros(pair_count, dtype=float)

    if scenario in {"shared_opportunity", "small_axis_shared"}:
        shared = 2.5 * rng.lognormal(mean=0.0, sigma=0.9, size=pair_count)
    elif scenario == "structural_missingness":
        shared = 1.5 * rng.lognormal(mean=0.0, sigma=0.7, size=pair_count)
    else:
        shared = np.zeros(pair_count, dtype=float)
    effective_noise = 0.15 if scenario == "low_noise_independent" else noise_sd
    target_score = np.maximum(
        target_truth + shared + rng.normal(scale=effective_noise, size=pair_count),
        0.0,
    )
    reference_score = np.maximum(
        reference_truth + shared + rng.normal(scale=effective_noise, size=pair_count),
        0.0,
    )
    if scenario == "global_null":
        target_score = np.abs(rng.normal(scale=noise_sd, size=pair_count))
        reference_score = np.abs(rng.normal(scale=noise_sd, size=pair_count))

    target_observed = np.ones(pair_count, dtype=bool)
    reference_observed = np.ones(pair_count, dtype=bool)
    if scenario == "structural_missingness":
        missing = rng.random(pair_count) < float(
            simulation["structural_missing_fraction"]
        )
        target_observed[missing] = False
        reference_observed[missing] = False
        target_score[missing] = np.nan
        reference_score[missing] = np.nan

    return pd.DataFrame(
        {
            "scenario": scenario,
            "seed": seed,
            "pair_id": [f"pair_{index:03d}" for index in range(pair_count)],
            "target_score": target_score,
            "reference_score": reference_score,
            "target_truth": target_truth,
            "reference_truth": reference_truth,
            "target_observed": target_observed,
            "reference_observed": reference_observed,
        }
    )


def _safe_spearman(score: np.ndarray, truth: np.ndarray) -> float:
    finite = np.isfinite(score) & np.isfinite(truth)
    if (
        finite.sum() < 3
        or np.unique(score[finite]).size < 2
        or np.unique(truth[finite]).size < 2
    ):
        return float("nan")
    return float(spearmanr(score[finite], truth[finite]).statistic)


def _metric_records(
    table: pd.DataFrame,
    target_score: np.ndarray,
    reference_score: np.ndarray,
    *,
    split: str,
    method: str,
    subtraction_fraction: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for direction, score in (
        ("target", target_score),
        ("reference", reference_score),
    ):
        truth = table[f"{direction}_truth"].to_numpy(dtype=float)
        finite = np.isfinite(score) & np.isfinite(truth)
        threshold = float(np.quantile(truth[finite], 0.75))
        label = truth[finite] >= threshold
        records.append(
            {
                "split": split,
                "scenario": str(table["scenario"].iloc[0]),
                "seed": int(table["seed"].iloc[0]),
                "method": method,
                "direction": direction,
                "subtraction_fraction": subtraction_fraction,
                "pair_rank_spearman": _safe_spearman(score, truth),
                "top_quartile_auroc": (
                    float(roc_auc_score(label, score[finite]))
                    if np.unique(label).size == 2
                    else np.nan
                ),
                "pair_score_mean": float(np.nanmean(score)),
                "tie_fraction": float(
                    pd.Series(score[finite]).duplicated(keep=False).mean()
                ),
                "observed_pairs": int(finite.sum()),
            }
        )
    return records


def evaluate_problem(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    split: str,
) -> pd.DataFrame:
    target = table["target_score"].to_numpy(dtype=float)
    reference = table["reference_score"].to_numpy(dtype=float)
    target_observed = table["target_observed"].to_numpy(dtype=bool)
    reference_observed = table["reference_observed"].to_numpy(dtype=bool)
    records: list[dict[str, object]] = []
    baseline_target, baseline_reference, _ = paired_rank_contrast(
        target,
        reference,
        subtraction_fraction=0.0,
        target_observed=target_observed,
        reference_observed=reference_observed,
    )
    records.extend(
        _metric_records(
            table,
            baseline_target,
            baseline_reference,
            split=split,
            method="rc9_rank_reference",
            subtraction_fraction=0.0,
        )
    )
    for candidate_value in config["candidates"]:
        candidate = cast(Mapping[str, Any], candidate_value)
        fraction = float(candidate["subtraction_fraction"])
        contrasted_target, contrasted_reference, _ = paired_rank_contrast(
            target,
            reference,
            subtraction_fraction=fraction,
            target_observed=target_observed,
            reference_observed=reference_observed,
        )
        records.extend(
            _metric_records(
                table,
                contrasted_target,
                contrasted_reference,
                split=split,
                method=str(candidate["name"]),
                subtraction_fraction=fraction,
            )
        )
    return pd.DataFrame.from_records(records)


def _scenario_delta(
    wide: pd.DataFrame, candidate: str, scenarios: set[str]
) -> float:
    delta = wide[candidate] - wide["rc9_rank_reference"]
    return float(
        delta.loc[delta.index.get_level_values("scenario").isin(scenarios)].mean()
    )


def _null_ratio(metrics: pd.DataFrame, *, split: str, candidate: str) -> float:
    null = metrics.loc[
        metrics["split"].eq(split) & metrics["scenario"].eq("global_null")
    ]
    wide = null.pivot(
        index=["seed", "direction"],
        columns="method",
        values="pair_score_mean",
    )
    supported = wide["rc9_rank_reference"] > 0.0
    ratio = (
        wide.loc[supported, candidate]
        / wide.loc[supported, "rc9_rank_reference"]
    )
    return float(ratio.quantile(0.95))


def _candidate_summary(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    split: str,
) -> pd.DataFrame:
    policy = cast(Mapping[str, Any], config["selection"])
    selected = metrics.loc[metrics["split"].eq(split)]
    wide = selected.pivot(
        index=["scenario", "seed", "direction"],
        columns="method",
        values="pair_rank_spearman",
    )
    scenario_sets = {
        "primary": {str(value) for value in policy["primary_scenarios"]},
        "safety": {str(value) for value in policy["safety_scenarios"]},
        "independent": {str(policy["independent_scenario"])},
        "asymmetric": {str(policy["asymmetric_scenario"])},
        "structural": {str(policy["structural_scenario"])},
    }
    records: list[dict[str, object]] = []
    for candidate_value in config["candidates"]:
        candidate = cast(Mapping[str, Any], candidate_value)
        name = str(candidate["name"])
        records.append(
            {
                "candidate": name,
                "subtraction_fraction": float(candidate["subtraction_fraction"]),
                **{
                    f"{label}_mean_delta_vs_rc9": _scenario_delta(
                        wide, name, scenarios
                    )
                    for label, scenarios in scenario_sets.items()
                },
                "global_null_pair_score_ratio_q95": _null_ratio(
                    metrics, split=split, candidate=name
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _select_candidate(
    metrics: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[str, pd.DataFrame]:
    policy = cast(Mapping[str, Any], config["selection"])
    selection = _candidate_summary(metrics, config, split="development")
    selection["primary_pass"] = selection["primary_mean_delta_vs_rc9"].ge(
        float(policy["minimum_development_primary_mean_delta_vs_rc9"])
    )
    selection["safety_pass"] = selection["safety_mean_delta_vs_rc9"].ge(
        -float(policy["maximum_development_safety_mean_regression"])
    )
    selection["independent_pass"] = selection[
        "independent_mean_delta_vs_rc9"
    ].ge(-float(policy["maximum_development_independent_regression"]))
    selection["asymmetric_pass"] = selection[
        "asymmetric_mean_delta_vs_rc9"
    ].ge(-float(policy["maximum_development_asymmetric_regression"]))
    selection["global_null_pass"] = selection[
        "global_null_pair_score_ratio_q95"
    ].le(float(policy["global_null_maximum_pair_score_ratio_q95"]))
    gates = [
        "primary_pass",
        "safety_pass",
        "independent_pass",
        "asymmetric_pass",
        "global_null_pass",
    ]
    selection["eligible"] = selection[gates].all(axis=1)
    selection = selection.sort_values(
        [
            "eligible",
            "primary_mean_delta_vs_rc9",
            "safety_mean_delta_vs_rc9",
            "subtraction_fraction",
        ],
        ascending=[False, False, False, True],
        kind="stable",
        ignore_index=True,
    )
    selection.insert(0, "development_rank", np.arange(1, len(selection) + 1))
    selection["selected"] = False
    selection.loc[0, "selected"] = True
    return str(selection.loc[0, "candidate"]), selection


def _acceptance(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    selected: str,
    development_candidate_eligible: bool,
) -> dict[str, Any]:
    policy = cast(Mapping[str, Any], config["selection"])
    row = _candidate_summary(metrics, config, split="holdout").set_index(
        "candidate"
    ).loc[selected]
    checks: dict[str, Any] = {
        "selected_candidate": selected,
        "development_candidate_eligible": development_candidate_eligible,
        "development_candidate_eligible_pass": development_candidate_eligible,
        "holdout_primary_mean_delta_vs_rc9": float(
            row["primary_mean_delta_vs_rc9"]
        ),
        "holdout_safety_mean_delta_vs_rc9": float(row["safety_mean_delta_vs_rc9"]),
        "holdout_independent_mean_delta_vs_rc9": float(
            row["independent_mean_delta_vs_rc9"]
        ),
        "holdout_asymmetric_mean_delta_vs_rc9": float(
            row["asymmetric_mean_delta_vs_rc9"]
        ),
        "holdout_structural_mean_delta_vs_rc9": float(
            row["structural_mean_delta_vs_rc9"]
        ),
        "global_null_pair_score_ratio_q95": float(
            row["global_null_pair_score_ratio_q95"]
        ),
    }
    checks["holdout_primary_gain_pass"] = checks[
        "holdout_primary_mean_delta_vs_rc9"
    ] >= float(policy["minimum_holdout_primary_mean_delta_vs_rc9"])
    checks["holdout_safety_pass"] = checks[
        "holdout_safety_mean_delta_vs_rc9"
    ] >= -float(policy["maximum_holdout_safety_mean_regression_vs_rc9"])
    checks["holdout_independent_pass"] = checks[
        "holdout_independent_mean_delta_vs_rc9"
    ] >= -float(policy["maximum_holdout_independent_regression_vs_rc9"])
    checks["holdout_asymmetric_pass"] = checks[
        "holdout_asymmetric_mean_delta_vs_rc9"
    ] >= -float(policy["maximum_holdout_asymmetric_regression_vs_rc9"])
    checks["holdout_structural_pass"] = checks[
        "holdout_structural_mean_delta_vs_rc9"
    ] >= -float(policy["maximum_holdout_structural_regression_vs_rc9"])
    checks["global_null_pass"] = checks[
        "global_null_pair_score_ratio_q95"
    ] <= float(policy["global_null_maximum_pair_score_ratio_q95"])
    checks["accepted"] = bool(
        checks["development_candidate_eligible_pass"]
        and checks["holdout_primary_gain_pass"]
        and checks["holdout_safety_pass"]
        and checks["holdout_independent_pass"]
        and checks["holdout_asymmetric_pass"]
        and checks["holdout_structural_pass"]
        and checks["global_null_pass"]
    )
    return checks


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["split", "scenario", "method"], observed=True, sort=True)
        .agg(
            pair_rank_spearman_mean=("pair_rank_spearman", "mean"),
            top_quartile_auroc_mean=("top_quartile_auroc", "mean"),
            pair_score_mean=("pair_score_mean", "mean"),
            tie_fraction_mean=("tie_fraction", "mean"),
            evaluations=("pair_rank_spearman", "size"),
        )
        .reset_index()
    )


def run(
    config_path: Path, output_dir: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_config(config_path)
    output = prepare_output(output_dir, overwrite=overwrite)
    tables: list[pd.DataFrame] = []
    for split, field in (
        ("development", "development_seeds"),
        ("holdout", "holdout_seeds"),
    ):
        for scenario in config["simulation"]["scenarios"]:
            for seed in config[field]:
                problem = simulate_pair_problem(
                    config, scenario=str(scenario), seed=int(seed)
                )
                tables.append(evaluate_problem(problem, config, split=split))
    metrics = pd.concat(tables, ignore_index=True)
    selected, selection = _select_candidate(metrics, config)
    development_candidate_eligible = bool(selection.loc[0, "eligible"])
    acceptance = _acceptance(
        metrics,
        config,
        selected=selected,
        development_candidate_eligible=development_candidate_eligible,
    )
    aggregate = _aggregate(metrics)
    candidate = next(
        cast(Mapping[str, Any], value)
        for value in config["candidates"]
        if cast(Mapping[str, Any], value)["name"] == selected
    )
    frozen = {
        "schema_version": "crychic-paired-rank-contrast-frozen-v1",
        "selected_candidate": selected,
        "candidate": dict(candidate),
        "selected_on": "development_only",
        "holdout_used_for_candidate_selection": False,
        "real_cohorts_used_for_selection": False,
        "accepted_for_real_benchmark": bool(acceptance["accepted"]),
        "formal_release_allowed": False,
        "config_sha256": sha256_file(config_path),
    }
    outputs = {
        "replicate_metrics.tsv": metrics,
        "aggregate_metrics.tsv": aggregate,
        "candidate_selection.tsv": selection,
    }
    for filename, table in outputs.items():
        table.to_csv(output / filename, sep="\t", index=False, lineterminator="\n")
    write_json(output / "acceptance.json", acceptance)
    write_json(output / "frozen_candidate.json", frozen)
    manifest = {
        "schema_version": "crychic-paired-rank-contrast-result-v1",
        "status": "complete",
        "accepted": bool(acceptance["accepted"]),
        "config": {
            "path": _portable_config_path(config_path),
            "sha256": sha256_file(config_path),
        },
        "selection": frozen,
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "code": git_metadata(Path(__file__).resolve().parents[2]),
        "limitations": [
            "pair_summary_simulation_not_full_edge_pipeline",
            "spatial_des_is_an_indirect_pair_level_proxy",
            "candidate_is_not_released_scientific_inference",
        ],
        "outputs": {
            filename: {"rows": len(table), "sha256": sha256_file(output / filename)}
            for filename, table in outputs.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.config, args.output_dir, overwrite=args.overwrite)
    print(json.dumps(result["acceptance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
