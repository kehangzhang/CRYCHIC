"""Independent simulation benchmark for RC3 receiver-program soft evidence."""

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
from benchmarks.literature.liana_hypergraph_residual import (
    conservative_sign_statistic,
    fit_cross_validated_residual,
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
        raise ValueError("RC3 program config must contain an object")
    return cast(dict[str, Any], value)


def simulate_program_problem(
    config: Mapping[str, Any], *, scenario: str, seed: int
) -> pd.DataFrame:
    """Add receiver-program evidence to an independently simulated LR problem."""

    simulation = cast(Mapping[str, Any], config["simulation"])
    allowed = set(str(value) for value in simulation["program_scenarios"])
    if scenario not in allowed:
        raise ValueError(f"unknown RC3 program scenario: {scenario}")
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
    rng = np.random.default_rng(seed + 9_000_001)
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
    scale = max(float(grouped["program_truth"].std()), 1e-8)
    standardized = grouped["program_truth"].to_numpy(dtype=float) / scale
    if scenario == "helpful_program" or scenario == "missing_helpful_program":
        program_z = 1.5 * standardized + rng.normal(scale=0.7, size=len(grouped))
    elif scenario == "weak_program":
        program_z = 0.4 * standardized + rng.normal(scale=1.2, size=len(grouped))
    elif scenario == "antagonistic_program":
        program_z = -1.5 * standardized + rng.normal(scale=0.7, size=len(grouped))
    else:
        program_z = rng.normal(scale=1.2, size=len(grouped))
    if scenario == "missing_helpful_program":
        program_z[rng.random(len(grouped)) < 0.45] = np.nan
    grouped["program_z"] = program_z
    table = table.merge(
        grouped.loc[:, [*group_columns, "program_z"]],
        on=group_columns,
        how="left",
        validate="many_to_one",
    )
    table["scenario"] = scenario
    return table


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _weighted_pair_scores(
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


def _metrics(
    pairs: pd.DataFrame,
    *,
    split: str,
    scenario: str,
    seed: int,
    candidate: str,
    fit: Mapping[str, object],
) -> list[dict[str, object]]:
    records = []
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
                "candidate": candidate,
                "direction": direction,
                "pair_rank_spearman": _safe_spearman(score, truth),
                "top_quartile_auroc": (
                    float(roc_auc_score(label, score))
                    if label.nunique() == 2
                    else np.nan
                ),
                "tie_fraction": float(score.duplicated(keep=False).mean()),
                **{f"program_{key}": value for key, value in fit.items()},
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
    sign_statistic = conservative_sign_statistic(
        table["baseline_stat"].to_numpy(dtype=float),
        theta,
        residual_fit,
        maximum_abs_baseline=float(residual["sign_correction_max_abs_baseline"]),
    )
    program_policy = cast(Mapping[str, Any], config["program_policy"])
    scenario = str(table["scenario"].iloc[0])
    seed = int(table["seed"].iloc[0])
    records = []
    for candidate in config["program_candidates"]:
        candidate = cast(Mapping[str, Any], candidate)
        fit = fit_receiver_program_reliability(
            table["baseline_stat"].to_numpy(dtype=float),
            table["program_z"].to_numpy(dtype=float),
            maximum_alpha=float(candidate["maximum_alpha"]),
            alpha_cap=float(program_policy["alpha_cap"]),
            minimum_edges=int(program_policy["minimum_concordance_edges"]),
        )
        direction = np.where(sign_statistic >= 0.0, 1.0, -1.0)
        weights, _ = receiver_program_weights(
            direction,
            table["program_z"].to_numpy(dtype=float),
            fit,
        )
        pairs = _weighted_pair_scores(table, sign_statistic, weights)
        records.extend(
            _metrics(
                pairs,
                split=split,
                scenario=scenario,
                seed=seed,
                candidate=str(candidate["name"]),
                fit=fit.to_dict(),
            )
        )
    return pd.DataFrame.from_records(records)


def _select_candidate(
    metrics: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[str, pd.DataFrame]:
    policy = cast(Mapping[str, Any], config["selection"])
    development = metrics.loc[metrics["split"].eq("development")].copy()
    primary = set(str(value) for value in policy["primary_scenarios"])
    safety = set(str(value) for value in policy["safety_scenarios"])
    baseline_name = "program_max_alpha_0"
    wide = development.pivot(
        index=["scenario", "seed", "direction"],
        columns="candidate",
        values="pair_rank_spearman",
    )
    records = []
    for candidate in sorted(development["candidate"].unique()):
        delta = wide[candidate] - wide[baseline_name]
        primary_gain = float(
            delta.loc[delta.index.get_level_values("scenario").isin(primary)].mean()
        )
        safety_gain = float(
            delta.loc[delta.index.get_level_values("scenario").isin(safety)].mean()
        )
        null_alpha = development.loc[
            development["candidate"].eq(candidate)
            & development["scenario"].eq("global_null"),
            "program_effective_alpha",
        ]
        null_q95 = float(null_alpha.quantile(0.95))
        safety_pass = safety_gain >= -float(policy["maximum_safety_mean_regression"])
        null_pass = null_q95 <= float(policy["global_null_maximum_effective_alpha_q95"])
        records.append(
            {
                "candidate": candidate,
                "primary_mean_delta": primary_gain,
                "safety_mean_delta": safety_gain,
                "global_null_effective_alpha_q95": null_q95,
                "safety_pass": safety_pass,
                "global_null_pass": null_pass,
                "eligible": safety_pass and null_pass,
            }
        )
    selection = pd.DataFrame.from_records(records).sort_values(
        ["eligible", "primary_mean_delta", "safety_mean_delta", "candidate"],
        ascending=[False, False, False, True],
        kind="stable",
        ignore_index=True,
    )
    if not selection["eligible"].any():
        raise ValueError("no RC3 program candidate passed development safety gates")
    selection.insert(0, "development_rank", np.arange(1, len(selection) + 1))
    selection["selected"] = False
    selection.loc[0, "selected"] = True
    return str(selection.loc[0, "candidate"]), selection


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["split", "scenario", "candidate"], observed=True, sort=True)
        .agg(
            pair_rank_spearman_mean=("pair_rank_spearman", "mean"),
            top_quartile_auroc_mean=("top_quartile_auroc", "mean"),
            effective_alpha_mean=("program_effective_alpha", "mean"),
            reliability_mean=("program_reliability", "mean"),
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
        for scenario in config["simulation"]["program_scenarios"]:
            for seed in config[field]:
                problem = simulate_program_problem(
                    config, scenario=str(scenario), seed=int(seed)
                )
                tables.append(evaluate_problem(problem, config, split=split))
    metrics = pd.concat(tables, ignore_index=True)
    selected, selection = _select_candidate(metrics, config)
    aggregate = _aggregate(metrics)
    outputs = {
        "replicate_metrics.tsv": metrics,
        "aggregate_metrics.tsv": aggregate,
        "candidate_selection.tsv": selection,
    }
    for filename, table in outputs.items():
        table.to_csv(output / filename, sep="\t", index=False, lineterminator="\n")
    frozen = {
        "schema_version": "crychic-receiver-program-soft-frozen-v1",
        "selected_candidate": selected,
        "candidate": next(
            item for item in config["program_candidates"] if item["name"] == selected
        ),
        "program_policy": config["program_policy"],
        "selected_on": "development_only",
        "holdout_used_for_selection": False,
        "real_cohorts_used_for_selection": False,
        "formal_release_allowed": False,
        "config_sha256": sha256_file(config_path),
    }
    write_json(output / "frozen_candidate.json", frozen)
    manifest = {
        "schema_version": "crychic-receiver-program-soft-result-v1",
        "status": "complete",
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "selection": frozen,
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
    print(json.dumps(result["selection"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
