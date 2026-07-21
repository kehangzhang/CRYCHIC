"""Independent simulation benchmark for RC5 two-sided program calibration."""

from __future__ import annotations

import argparse
import copy
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
from benchmarks.literature.liana_hypergraph_residual import (
    conservative_sign_statistic,
    fit_cross_validated_residual,
)
from benchmarks.literature.receiver_program_soft import (
    fit_receiver_program_reliability,
    two_sided_calibrated_program_weights,
)
from benchmarks.simulation.receiver_program_soft_benchmark import (
    simulate_program_problem,
)


def _read_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RC5 config must contain an object")
    return cast(dict[str, Any], value)


def simulate_two_sided_program_problem(
    config: Mapping[str, Any], *, scenario: str, seed: int
) -> pd.DataFrame:
    """Generate balanced or sign-biased program evidence independently."""

    simulation = cast(Mapping[str, Any], config["simulation"])
    allowed = {str(value) for value in simulation["scenarios"]}
    if scenario not in allowed:
        raise ValueError(f"unknown RC5 program scenario: {scenario}")
    base_by_scenario = {
        "balanced_helpful_program": "helpful_program",
        "weak_program": "weak_program",
        "missing_helpful_program": "missing_helpful_program",
        "positive_biased_helpful_program": "helpful_program",
        "negative_biased_helpful_program": "helpful_program",
        "balanced_noisy_program": "noisy_program",
        "positive_biased_noisy_program": "noisy_program",
        "negative_biased_noisy_program": "noisy_program",
        "antagonistic_program": "antagonistic_program",
        "global_null": "global_null",
    }
    base_scenario = base_by_scenario[scenario]
    base_config = copy.deepcopy(dict(config))
    base_simulation = cast(dict[str, Any], base_config["simulation"])
    base_simulation["program_scenarios"] = [base_scenario]
    table = simulate_program_problem(base_config, scenario=base_scenario, seed=seed)
    bias = float(simulation["program_sign_bias"])
    if scenario.startswith("positive_biased"):
        table.loc[table["program_z"].notna(), "program_z"] += bias
    elif scenario.startswith("negative_biased"):
        table.loc[table["program_z"].notna(), "program_z"] -= bias
    table["scenario"] = scenario
    return table


def _pair_scores(
    table: pd.DataFrame,
    sign_statistic: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    selected = table["interaction_pvalue"].to_numpy(dtype=float) < 0.05
    sender = table["sender"].astype(str).to_numpy()
    receiver = table["receiver"].astype(str).to_numpy()
    working = pd.DataFrame(
        {
            "pair_sender": np.minimum(sender, receiver),
            "pair_receiver": np.maximum(sender, receiver),
            "target_score": weights * selected * (sign_statistic > 0.0),
            "reference_score": weights * selected * (sign_statistic < 0.0),
            "target_truth": table["true_effect"].to_numpy(dtype=float) > 0.0,
            "reference_truth": table["true_effect"].to_numpy(dtype=float) < 0.0,
        }
    )
    return working.groupby(
        ["pair_sender", "pair_receiver"], observed=True, sort=True
    ).sum()


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def evaluate_problem(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    split: str,
) -> pd.DataFrame:
    residual = cast(Mapping[str, Any], config["residual"])
    theta, residual_fit, _ = fit_cross_validated_residual(
        table,
        table["baseline_stat"].to_numpy(dtype=float),
        table["crychic_anchor"].to_numpy(dtype=float),
        views=tuple(str(value) for value in residual["views"]),
        key_columns=tuple(str(value) for value in residual["key_columns"]),
        lambda_grid=tuple(float(value) for value in residual["lambda_grid"]),
        anchor_grid=tuple(float(value) for value in residual["anchor_grid"]),
        folds=int(residual["folds"]),
        seed=int(residual["seed"]),
        minimum_cv_improvement=float(residual["minimum_cv_improvement"]),
        full_gate_improvement=float(residual["full_gate_improvement"]),
    )
    baseline = table["baseline_stat"].to_numpy(dtype=float)
    sign_statistic = conservative_sign_statistic(
        baseline,
        theta,
        residual_fit,
        maximum_abs_baseline=float(residual["sign_correction_max_abs_baseline"]),
    )
    program_policy = cast(Mapping[str, Any], config["program_policy"])
    fit = fit_receiver_program_reliability(
        baseline,
        table["program_z"].to_numpy(dtype=float),
        maximum_alpha=float(program_policy["maximum_alpha"]),
        alpha_cap=float(program_policy["alpha_cap"]),
        minimum_edges=int(program_policy["minimum_concordance_edges"]),
    )
    direction = np.where(sign_statistic >= 0.0, 1.0, -1.0)
    scenario = str(table["scenario"].iloc[0])
    seed = int(table["seed"].iloc[0])
    records = []
    for priority, candidate_value in enumerate(config["candidates"]):
        candidate = cast(Mapping[str, Any], candidate_value)
        weights, _, calibration = two_sided_calibrated_program_weights(
            direction,
            table["program_z"].to_numpy(dtype=float),
            fit,
            required_two_sided_fraction=float(candidate["required_two_sided_fraction"]),
        )
        pairs = _pair_scores(table, sign_statistic, weights)
        for label in ("target", "reference"):
            score = pairs[f"{label}_score"]
            truth = pairs[f"{label}_truth"]
            threshold = truth.quantile(0.75)
            binary = truth.ge(threshold).astype(int)
            records.append(
                {
                    "split": split,
                    "scenario": scenario,
                    "seed": seed,
                    "direction": label,
                    "candidate": str(candidate["name"]),
                    "candidate_priority": priority,
                    "required_two_sided_fraction": float(
                        candidate["required_two_sided_fraction"]
                    ),
                    "pair_rank_spearman": _safe_spearman(score, truth),
                    "top_quartile_auroc": (
                        float(roc_auc_score(binary, score))
                        if binary.nunique() == 2
                        else np.nan
                    ),
                    "tie_fraction": float(score.duplicated(keep=False).mean()),
                    **{f"program_{key}": value for key, value in fit.to_dict().items()},
                    **{
                        f"calibration_{key}": value
                        for key, value in calibration.to_dict().items()
                    },
                }
            )
    return pd.DataFrame.from_records(records)


def _candidate_deltas(metrics: pd.DataFrame, scenarios: set[str]) -> pd.Series:
    wide = metrics.pivot(
        index=["scenario", "seed", "direction"],
        columns="candidate",
        values="pair_rank_spearman",
    )
    selected = wide.index.get_level_values("scenario").isin(scenarios)
    return (
        wide.loc[selected]
        .subtract(wide.loc[selected, "signed_rc3_reference"], axis=0)
        .mean()
    )


def _select_candidate(
    metrics: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[str, pd.DataFrame]:
    policy = cast(Mapping[str, Any], config["selection"])
    development = metrics.loc[metrics["split"].eq("development")].copy()
    primary = {
        str(value)
        for field in ("balanced_primary_scenarios", "biased_primary_scenarios")
        for value in policy[field]
    }
    safety = {str(value) for value in policy["safety_scenarios"]}
    primary_delta = _candidate_deltas(development, primary)
    safety_delta = _candidate_deltas(development, safety)
    records = []
    for candidate in development["candidate"].drop_duplicates():
        rows = development.loc[development["candidate"].eq(candidate)]
        null_alpha_q95 = float(
            rows.loc[
                rows["scenario"].eq("global_null"), "program_effective_alpha"
            ].quantile(0.95)
        )
        safety_pass = float(safety_delta[candidate]) >= -float(
            policy["maximum_development_safety_mean_regression"]
        )
        null_pass = null_alpha_q95 <= float(
            policy["global_null_maximum_effective_alpha_q95"]
        )
        records.append(
            {
                "candidate": candidate,
                "candidate_priority": int(rows["candidate_priority"].iloc[0]),
                "primary_mean_delta_vs_signed": float(primary_delta[candidate]),
                "safety_mean_delta_vs_signed": float(safety_delta[candidate]),
                "global_null_effective_alpha_q95": null_alpha_q95,
                "safety_pass": safety_pass,
                "global_null_pass": null_pass,
                "eligible": safety_pass and null_pass,
            }
        )
    selection = pd.DataFrame.from_records(records).sort_values(
        [
            "eligible",
            "primary_mean_delta_vs_signed",
            "safety_mean_delta_vs_signed",
            "candidate_priority",
        ],
        ascending=[False, False, False, True],
        kind="stable",
        ignore_index=True,
    )
    if not selection["eligible"].any():
        raise ValueError("no RC5 candidate passed development safety gates")
    selection.insert(0, "development_rank", np.arange(1, len(selection) + 1))
    selection["selected"] = False
    selection.loc[0, "selected"] = True
    return str(selection.loc[0, "candidate"]), selection


def _acceptance(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    selected: str,
) -> dict[str, object]:
    policy = cast(Mapping[str, Any], config["selection"])
    holdout = metrics.loc[metrics["split"].eq("holdout")].copy()
    balanced = {str(value) for value in policy["balanced_primary_scenarios"]}
    biased = {str(value) for value in policy["biased_primary_scenarios"]}
    safety = {str(value) for value in policy["safety_scenarios"]}
    balanced_delta = float(_candidate_deltas(holdout, balanced)[selected])
    biased_delta = float(_candidate_deltas(holdout, biased)[selected])
    safety_delta = float(_candidate_deltas(holdout, safety)[selected])
    novel = selected != "signed_rc3_reference"
    checks = {
        "selected_candidate": selected,
        "novel_candidate": novel,
        "novel_candidate_pass": novel or not bool(policy["novel_candidate_required"]),
        "holdout_biased_mean_delta_vs_signed": biased_delta,
        "holdout_biased_gain_pass": biased_delta
        >= float(policy["minimum_holdout_biased_mean_delta_vs_signed"]),
        "holdout_balanced_mean_delta_vs_signed": balanced_delta,
        "holdout_balanced_safety_pass": balanced_delta
        >= -float(policy["maximum_holdout_balanced_mean_regression_vs_signed"]),
        "holdout_safety_mean_delta_vs_signed": safety_delta,
        "holdout_safety_pass": safety_delta
        >= -float(policy["maximum_holdout_safety_mean_regression_vs_signed"]),
    }
    checks["accepted"] = bool(
        checks["novel_candidate_pass"]
        and checks["holdout_biased_gain_pass"]
        and checks["holdout_balanced_safety_pass"]
        and checks["holdout_safety_pass"]
    )
    return checks


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["split", "scenario", "candidate"], observed=True, sort=True)
        .agg(
            pair_rank_spearman_mean=("pair_rank_spearman", "mean"),
            top_quartile_auroc_mean=("top_quartile_auroc", "mean"),
            tie_fraction_mean=("tie_fraction", "mean"),
            minority_sign_fraction_mean=("calibration_minority_sign_fraction", "mean"),
            contradiction_scale_mean=("calibration_contradiction_scale", "mean"),
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
    tables = []
    for split, field in (
        ("development", "development_seeds"),
        ("holdout", "holdout_seeds"),
    ):
        for scenario in config["simulation"]["scenarios"]:
            for seed in config[field]:
                problem = simulate_two_sided_program_problem(
                    config, scenario=str(scenario), seed=int(seed)
                )
                tables.append(evaluate_problem(problem, config, split=split))
    metrics = pd.concat(tables, ignore_index=True)
    selected, selection = _select_candidate(metrics, config)
    acceptance = _acceptance(metrics, config, selected)
    aggregate = _aggregate(metrics)
    outputs = {
        "replicate_metrics.tsv": metrics,
        "aggregate_metrics.tsv": aggregate,
        "candidate_selection.tsv": selection,
    }
    for filename, table in outputs.items():
        table.to_csv(output / filename, sep="\t", index=False, lineterminator="\n")
    write_json(output / "acceptance.json", acceptance)
    frozen = {
        "schema_version": "crychic-two-sided-program-frozen-v1",
        "selected_candidate": selected,
        "candidate": next(
            item for item in config["candidates"] if item["name"] == selected
        ),
        "program_policy": config["program_policy"],
        "selected_on": "development_only",
        "holdout_used_for_candidate_selection": False,
        "real_cohorts_used_for_selection": False,
        "accepted_for_real_benchmark": acceptance["accepted"],
        "config_sha256": sha256_file(config_path),
        "formal_release_allowed": False,
    }
    write_json(output / "frozen_candidate.json", frozen)
    manifest = {
        "schema_version": "crychic-two-sided-program-result-v1",
        "status": "complete",
        "accepted": acceptance["accepted"],
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "selection": frozen,
        "acceptance": acceptance,
        "limitations": [
            "effect_summary_simulation_not_full_pseudobulk_pipeline",
            "two_sided_coverage_is_global_not_receiver_specific",
            "candidate_is_not_released_scientific_inference",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "code": git_metadata(Path(__file__).resolve().parents[2]),
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
