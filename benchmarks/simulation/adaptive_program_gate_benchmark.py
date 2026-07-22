"""Independent simulation benchmark for RC8 adaptive receiver-program gates."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)
from benchmarks.literature.adaptive_program_gate import (
    adaptive_program_gate_weights,
    fit_adaptive_program_gate,
)
from benchmarks.literature.liana_hypergraph_residual import (
    conservative_sign_statistic,
    fit_cross_validated_residual,
    stable_edge_folds,
)
from benchmarks.literature.receiver_program_soft import (
    fit_receiver_program_reliability,
    receiver_program_weights,
)
from benchmarks.simulation.liana_hypergraph_residual_benchmark import (
    simulate_residual_problem,
)


def _read_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RC8 adaptive program config must contain an object")
    return cast(dict[str, Any], value)


def _portable_config_path(path: Path) -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError("RC8 config must be stored inside the repository") from error


def simulate_adaptive_gate_problem(
    config: Mapping[str, Any], *, scenario: str, seed: int
) -> pd.DataFrame:
    """Simulate flat, tail-separable, missing, and unsafe program evidence."""

    simulation = cast(Mapping[str, Any], config["simulation"])
    allowed = {str(value) for value in simulation["program_scenarios"]}
    if scenario not in allowed:
        raise ValueError(f"unknown RC8 program scenario: {scenario}")
    base_scenario = (
        "global_null" if scenario == "global_null" else "smooth_helpful_anchor"
    )
    base_config = {
        "simulation": {
            "cell_type_count": int(simulation["cell_type_count"]),
            "interaction_count": int(simulation["interaction_count"]),
            "scenarios": [base_scenario],
        }
    }
    table = simulate_residual_problem(base_config, scenario=base_scenario, seed=seed)
    rng = np.random.default_rng(seed + 15_000_011)
    if scenario != "global_null":
        call_evidence = float(simulation["lr_call_signal_scale"]) * np.abs(
            table["true_effect"].to_numpy(dtype=float)
        ) + rng.normal(scale=float(simulation["lr_call_noise_sd"]), size=len(table))
        table["interaction_pvalue"] = 2.0 * norm.sf(np.abs(call_evidence))

    group_columns = ["receiver", "interaction_id"]
    grouped = (
        table.groupby(group_columns, observed=True, sort=True)["true_effect"]
        .mean()
        .rename("program_truth")
        .reset_index()
    )
    truth = grouped["program_truth"].to_numpy(dtype=float)
    truth_scale = max(float(np.std(truth)), 1e-8)
    standardized = truth / truth_scale
    truth_sign = np.where(standardized >= 0.0, 1.0, -1.0)
    functional = np.ones(len(grouped), dtype=bool)
    if scenario == "dense_flat_program":
        functional = rng.random(len(grouped)) < 0.82
        observed_sign = np.where(functional, truth_sign, -truth_sign)
        program_z = observed_sign * 4.0
    elif scenario in {"tail_separable_program", "missing_tail_program"}:
        cutoff = float(np.quantile(np.abs(standardized), 0.7))
        functional = np.abs(standardized) >= cutoff
        nuisance_sign = rng.choice((-1.0, 1.0), size=len(grouped))
        program_z = np.where(
            functional,
            truth_sign * (5.0 + np.abs(rng.normal(scale=0.5, size=len(grouped)))),
            nuisance_sign * (1.0 + np.abs(rng.normal(scale=0.35, size=len(grouped)))),
        )
    elif scenario == "weak_tail_program":
        cutoff = float(np.quantile(np.abs(standardized), 0.65))
        functional = np.abs(standardized) >= cutoff
        nuisance_sign = rng.choice((-1.0, 1.0), size=len(grouped))
        program_z = np.where(
            functional,
            truth_sign * (3.0 + np.abs(rng.normal(scale=0.8, size=len(grouped)))),
            nuisance_sign * (1.5 + np.abs(rng.normal(scale=0.6, size=len(grouped)))),
        )
    elif scenario == "diffuse_program":
        functional = rng.random(len(grouped)) < 0.75
        nuisance_sign = rng.choice((-1.0, 1.0), size=len(grouped))
        observed_sign = np.where(functional, truth_sign, nuisance_sign)
        program_z = observed_sign * (
            1.5 + np.abs(rng.normal(scale=1.0, size=len(grouped)))
        )
    elif scenario == "antagonistic_program":
        program_z = -2.0 * standardized + rng.normal(scale=0.9, size=len(grouped))
    elif scenario == "isolated_program":
        program_z = 2.0 * standardized + rng.normal(scale=0.9, size=len(grouped))
        program_z = rng.permutation(program_z)
    else:
        program_z = rng.normal(scale=1.5, size=len(grouped))
    if scenario == "missing_tail_program":
        missing = rng.random(len(grouped)) < float(
            simulation["missing_program_fraction"]
        )
        program_z[missing] = np.nan
    grouped["program_z"] = program_z
    grouped["functional_program"] = functional
    table = table.merge(
        grouped.loc[:, [*group_columns, "program_z", "functional_program"]],
        on=group_columns,
        how="left",
        validate="many_to_one",
    )
    true_effect = table["true_effect"].to_numpy(dtype=float)
    if scenario in {
        "dense_flat_program",
        "tail_separable_program",
        "weak_tail_program",
        "missing_tail_program",
        "diffuse_program",
    }:
        communication_truth = np.where(
            table["functional_program"].to_numpy(dtype=bool), true_effect, 0.0
        )
    elif scenario == "isolated_program":
        communication_truth = np.where(
            rng.random(len(table)) < 0.5, true_effect, 0.0
        )
    else:
        communication_truth = true_effect
    table["communication_truth_effect"] = communication_truth
    table["scenario"] = scenario
    return table


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _pair_scores(
    table: pd.DataFrame, sign_statistic: np.ndarray, weights: np.ndarray
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
            "target_truth": table["communication_truth_effect"].to_numpy(dtype=float)
            > 0.0,
            "reference_truth": table["communication_truth_effect"].to_numpy(
                dtype=float
            )
            < 0.0,
        }
    )
    return working.groupby(
        ["pair_sender", "pair_receiver"], observed=True, sort=True
    ).sum()


def _metric_records(
    pairs: pd.DataFrame,
    *,
    split: str,
    scenario: str,
    seed: int,
    method: str,
    diagnostics: Mapping[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for direction in ("target", "reference"):
        score = pairs[f"{direction}_score"]
        truth = pairs[f"{direction}_truth"]
        threshold = truth.quantile(0.75)
        label = truth.ge(threshold).astype(int)
        records.append(
            {
                "split": split,
                "scenario": scenario,
                "seed": seed,
                "method": method,
                "direction": direction,
                "pair_rank_spearman": _safe_spearman(score, truth),
                "top_quartile_auroc": (
                    float(roc_auc_score(label, score))
                    if label.nunique() == 2
                    else np.nan
                ),
                "tie_fraction": float(score.duplicated(keep=False).mean()),
                **diagnostics,
            }
        )
    return records


def evaluate_problem(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    split: str,
) -> pd.DataFrame:
    residual = cast(Mapping[str, Any], config["residual"])
    baseline = table["baseline_stat"].to_numpy(dtype=float)
    theta, residual_fit, _ = fit_cross_validated_residual(
        table,
        baseline,
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
    sign_statistic = conservative_sign_statistic(
        baseline,
        theta,
        residual_fit,
        maximum_abs_baseline=float(residual["sign_correction_max_abs_baseline"]),
    )
    direction = np.where(sign_statistic >= 0.0, 1.0, -1.0)
    program = table["program_z"].to_numpy(dtype=float)
    call_mask = table["interaction_pvalue"].to_numpy(dtype=float) < 0.05
    scenario = str(table["scenario"].iloc[0])
    seed = int(table["seed"].iloc[0])
    records: list[dict[str, object]] = []
    records.extend(
        _metric_records(
            _pair_scores(table, sign_statistic, np.ones(len(table))),
            split=split,
            scenario=scenario,
            seed=seed,
            method="rc2_unweighted",
            diagnostics={"gate_mode": "not_applicable"},
        )
    )

    soft_policy = cast(Mapping[str, Any], config["soft_program_policy"])
    soft_fit = fit_receiver_program_reliability(
        baseline,
        program,
        maximum_alpha=float(soft_policy["maximum_alpha"]),
        alpha_cap=float(soft_policy["alpha_cap"]),
        minimum_edges=int(soft_policy["minimum_concordance_edges"]),
    )
    soft_weights, _ = receiver_program_weights(direction, program, soft_fit)
    records.extend(
        _metric_records(
            _pair_scores(table, sign_statistic, soft_weights),
            split=split,
            scenario=scenario,
            seed=seed,
            method="rc3_soft",
            diagnostics={
                "gate_mode": "not_applicable",
                "soft_effective_alpha": soft_fit.effective_alpha,
                "soft_sign_concordance": soft_fit.sign_concordance,
            },
        )
    )

    gate_policy = cast(Mapping[str, Any], config["gate_policy"])
    validation_fold = stable_edge_folds(
        table,
        key_columns=tuple(str(value) for value in residual["key_columns"]),
        folds=int(gate_policy["validation_folds"]),
        seed=int(gate_policy["validation_seed"]),
    )
    for candidate_value in config["gate_candidates"]:
        candidate = cast(Mapping[str, Any], candidate_value)
        fit = fit_adaptive_program_gate(
            direction,
            program,
            call_mask,
            relative_fdp_reduction=float(candidate["relative_fdp_reduction"]),
            minimum_calls=int(gate_policy["minimum_calls"]),
            minimum_sign_concordance=float(
                gate_policy["minimum_sign_concordance"]
            ),
            minimum_positive_tail_count=int(
                gate_policy["minimum_positive_tail_count"]
            ),
            minimum_positive_tail_fraction=float(
                gate_policy["minimum_positive_tail_fraction"]
            ),
            tail_confidence_z=float(gate_policy["tail_confidence_z"]),
            minimum_tail_concordance_margin=float(
                gate_policy["minimum_tail_concordance_margin"]
            ),
            validation_fold=validation_fold,
            minimum_validation_tail_count=int(
                gate_policy["minimum_validation_tail_count"]
            ),
            minimum_validation_concordance_gain=float(
                gate_policy["minimum_validation_concordance_gain"]
            ),
        )
        weights = adaptive_program_gate_weights(direction, program, fit)
        diagnostics = {
            f"gate_{key}": value
            for key, value in fit.to_dict().items()
            if key not in {"fit_status", "formal_release_allowed"}
        }
        diagnostics["gate_mode"] = fit.gate_mode
        records.extend(
            _metric_records(
                _pair_scores(table, sign_statistic, weights),
                split=split,
                scenario=scenario,
                seed=seed,
                method=str(candidate["name"]),
                diagnostics=diagnostics,
            )
        )
    return pd.DataFrame.from_records(records)


def _scenario_delta(
    wide: pd.DataFrame,
    candidate: str,
    baseline: str,
    scenarios: set[str],
) -> float:
    delta = wide[candidate] - wide[baseline]
    return float(
        delta.loc[delta.index.get_level_values("scenario").isin(scenarios)].mean()
    )


def _gate_activation_rate(
    metrics: pd.DataFrame, *, split: str, scenario: str, method: str
) -> float:
    rows = metrics.loc[
        metrics["split"].eq(split)
        & metrics["scenario"].eq(scenario)
        & metrics["method"].eq(method)
    ].drop_duplicates(["scenario", "seed", "method"])
    if rows.empty:
        return float("nan")
    return float(rows["gate_mode"].ne("no_gate").mean())


def _select_candidate(
    metrics: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[str, pd.DataFrame]:
    policy = cast(Mapping[str, Any], config["selection"])
    development = metrics.loc[metrics["split"].eq("development")].copy()
    wide = development.pivot(
        index=["scenario", "seed", "direction"],
        columns="method",
        values="pair_rank_spearman",
    )
    primary = {str(value) for value in policy["primary_scenarios"]}
    tail = {str(value) for value in policy["tail_scenarios"]}
    safety = {str(value) for value in policy["safety_scenarios"]}
    flat = {str(policy["flat_scenario"])}
    sign_reference = "mirror_reduction_100"
    records: list[dict[str, object]] = []
    for candidate_value in config["gate_candidates"]:
        candidate = cast(Mapping[str, Any], candidate_value)
        name = str(candidate["name"])
        primary_delta = _scenario_delta(wide, name, "rc3_soft", primary)
        tail_delta = _scenario_delta(wide, name, "rc3_soft", tail)
        safety_delta = _scenario_delta(wide, name, "rc3_soft", safety)
        flat_delta = _scenario_delta(wide, name, sign_reference, flat)
        null_activation = _gate_activation_rate(
            development,
            split="development",
            scenario="global_null",
            method=name,
        )
        safety_pass = safety_delta >= -float(
            policy["maximum_development_safety_mean_regression"]
        )
        primary_pass = primary_delta >= float(
            policy["minimum_development_primary_mean_delta_vs_rc3"]
        )
        tail_pass = tail_delta >= float(
            policy["minimum_development_tail_mean_delta_vs_rc3"]
        )
        flat_pass = flat_delta >= -float(
            policy["maximum_development_flat_regression_vs_sign_only"]
        )
        null_pass = null_activation <= float(
            policy["global_null_maximum_gate_activation_rate"]
        )
        novel = name != sign_reference
        novel_pass = novel or not bool(policy["novel_candidate_required"])
        records.append(
            {
                "candidate": name,
                "relative_fdp_reduction": float(
                    candidate["relative_fdp_reduction"]
                ),
                "primary_mean_delta_vs_rc3": primary_delta,
                "tail_mean_delta_vs_rc3": tail_delta,
                "safety_mean_delta_vs_rc3": safety_delta,
                "flat_mean_delta_vs_sign_only": flat_delta,
                "global_null_gate_activation_rate": null_activation,
                "primary_pass": primary_pass,
                "tail_pass": tail_pass,
                "safety_pass": safety_pass,
                "flat_pass": flat_pass,
                "global_null_pass": null_pass,
                "novel": novel,
                "novel_pass": novel_pass,
                "eligible": (
                    primary_pass
                    and tail_pass
                    and safety_pass
                    and flat_pass
                    and null_pass
                    and novel_pass
                ),
            }
        )
    selection = pd.DataFrame.from_records(records).sort_values(
        [
            "eligible",
            "primary_mean_delta_vs_rc3",
            "tail_mean_delta_vs_rc3",
            "safety_mean_delta_vs_rc3",
            "candidate",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
        ignore_index=True,
    )
    if not selection["eligible"].any():
        raise ValueError("no RC8 candidate passed development gates")
    selection.insert(0, "development_rank", np.arange(1, len(selection) + 1))
    selection["selected"] = False
    selection.loc[0, "selected"] = True
    return str(selection.loc[0, "candidate"]), selection


def _acceptance(
    metrics: pd.DataFrame, config: Mapping[str, Any], *, selected: str
) -> dict[str, Any]:
    policy = cast(Mapping[str, Any], config["selection"])
    holdout = metrics.loc[metrics["split"].eq("holdout")].copy()
    wide = holdout.pivot(
        index=["scenario", "seed", "direction"],
        columns="method",
        values="pair_rank_spearman",
    )
    primary = {str(value) for value in policy["primary_scenarios"]}
    tail = {str(value) for value in policy["tail_scenarios"]}
    safety = {str(value) for value in policy["safety_scenarios"]}
    flat = {str(policy["flat_scenario"])}
    missing = {str(policy["missing_scenario"])}
    primary_delta = _scenario_delta(wide, selected, "rc3_soft", primary)
    tail_delta = _scenario_delta(wide, selected, "rc3_soft", tail)
    safety_delta = _scenario_delta(wide, selected, "rc3_soft", safety)
    flat_delta = _scenario_delta(
        wide, selected, "mirror_reduction_100", flat
    )
    missing_delta = _scenario_delta(wide, selected, "rc3_soft", missing)
    null_activation = _gate_activation_rate(
        holdout,
        split="holdout",
        scenario="global_null",
        method=selected,
    )
    checks: dict[str, Any] = {
        "selected_candidate": selected,
        "holdout_primary_mean_delta_vs_rc3": primary_delta,
        "holdout_primary_gain_pass": primary_delta
        >= float(policy["minimum_holdout_primary_mean_delta_vs_rc3"]),
        "holdout_tail_mean_delta_vs_rc3": tail_delta,
        "holdout_tail_gain_pass": tail_delta
        >= float(policy["minimum_holdout_tail_mean_delta_vs_rc3"]),
        "holdout_safety_mean_delta_vs_rc3": safety_delta,
        "holdout_safety_pass": safety_delta
        >= -float(policy["maximum_holdout_safety_mean_regression_vs_rc3"]),
        "holdout_flat_mean_delta_vs_sign_only": flat_delta,
        "holdout_flat_pass": flat_delta
        >= -float(policy["maximum_holdout_flat_regression_vs_sign_only"]),
        "holdout_missing_mean_delta_vs_rc3": missing_delta,
        "holdout_missing_pass": missing_delta
        >= -float(policy["maximum_holdout_missing_regression_vs_rc3"]),
        "global_null_gate_activation_rate": null_activation,
        "global_null_gate_pass": null_activation
        <= float(policy["global_null_maximum_gate_activation_rate"]),
        "novel_candidate": selected != "mirror_reduction_100",
    }
    checks["novel_candidate_pass"] = bool(
        checks["novel_candidate"] or not policy["novel_candidate_required"]
    )
    checks["accepted"] = bool(
        checks["holdout_primary_gain_pass"]
        and checks["holdout_tail_gain_pass"]
        and checks["holdout_safety_pass"]
        and checks["holdout_flat_pass"]
        and checks["holdout_missing_pass"]
        and checks["global_null_gate_pass"]
        and checks["novel_candidate_pass"]
    )
    return checks


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["split", "scenario", "method"], observed=True, sort=True)
        .agg(
            pair_rank_spearman_mean=("pair_rank_spearman", "mean"),
            top_quartile_auroc_mean=("top_quartile_auroc", "mean"),
            tie_fraction_mean=("tie_fraction", "mean"),
            gate_threshold_mean=("gate_threshold", "mean"),
            gate_activation_rate=(
                "gate_mode",
                lambda values: float(values.ne("no_gate").mean()),
            ),
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
        for scenario in config["simulation"]["program_scenarios"]:
            for seed in config[field]:
                problem = simulate_adaptive_gate_problem(
                    config, scenario=str(scenario), seed=int(seed)
                )
                tables.append(evaluate_problem(problem, config, split=split))
    metrics = pd.concat(tables, ignore_index=True)
    selected, selection = _select_candidate(metrics, config)
    acceptance = _acceptance(metrics, config, selected=selected)
    aggregate = _aggregate(metrics)
    candidate = next(
        cast(Mapping[str, Any], item)
        for item in config["gate_candidates"]
        if cast(Mapping[str, Any], item)["name"] == selected
    )
    frozen = {
        "schema_version": "crychic-adaptive-program-gate-frozen-v1",
        "selected_candidate": selected,
        "candidate": dict(candidate),
        "gate_policy": config["gate_policy"],
        "soft_program_policy": config["soft_program_policy"],
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
        "schema_version": "crychic-adaptive-program-gate-result-v1",
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
            "effect_summary_simulation_not_full_pseudobulk_pipeline",
            "receiver_program_source_is_synthetic_group_shared_evidence",
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
