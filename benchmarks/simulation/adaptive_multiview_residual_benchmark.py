"""Independent simulation benchmark for RC4 adaptive signed residual views."""

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
from benchmarks.literature.adaptive_multiview_residual import (
    estimate_oof_view_weights,
    fit_adaptive_cross_validated_residual,
    json_view_weights,
)
from benchmarks.literature.liana_hypergraph_residual import (
    conservative_sign_statistic,
    fit_cross_validated_residual,
)


def _read_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RC4 adaptive config must contain an object")
    return cast(dict[str, Any], value)


def _profiles(
    table: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[dict[str, tuple[float, ...]], pd.DataFrame]:
    residual = cast(Mapping[str, Any], config["residual"])
    policy = cast(Mapping[str, Any], residual["profile_policy"])
    profile_names = tuple(str(value) for value in policy["profiles"])
    if profile_names != ("all_equal", "reliability_weighted"):
        raise ValueError("RC4 frozen profile policy was altered")
    views = tuple(str(value) for value in residual["views"])
    weights, diagnostics = estimate_oof_view_weights(
        table,
        table["baseline_stat"].to_numpy(dtype=float),
        views=views,
        key_columns=tuple(str(value) for value in residual["key_columns"]),
        folds=int(residual["folds"]),
        seed=int(residual["seed"]),
        full_reliability_improvement=float(policy["full_reliability_improvement"]),
    )
    return {
        "all_equal": tuple(np.ones(len(views))),
        "reliability_weighted": weights,
    }, diagnostics


def simulate_adaptive_problem(
    config: Mapping[str, Any], *, scenario: str, seed: int
) -> pd.DataFrame:
    """Generate view-specific signed structure without using either solver."""

    simulation = cast(Mapping[str, Any], config["simulation"])
    allowed = {str(value) for value in simulation["scenarios"]}
    if scenario not in allowed:
        raise ValueError(f"unknown RC4 adaptive scenario: {scenario}")
    cell_count = int(simulation["cell_type_count"])
    interaction_count = int(simulation["interaction_count"])
    rng = np.random.default_rng(seed)
    cell_types = np.array([f"cell_{index}" for index in range(cell_count)])
    sender_index = np.repeat(np.arange(cell_count), cell_count * interaction_count)
    receiver_index = np.tile(
        np.repeat(np.arange(cell_count), interaction_count), cell_count
    )
    interaction_index = np.tile(np.arange(interaction_count), cell_count**2)
    ligand_index = interaction_index % 15
    receptor_index = (interaction_index * 7 + 3) % 17
    senders = cell_types[sender_index]
    receivers = cell_types[receiver_index]
    n_edges = len(interaction_index)

    sender_effect = rng.normal(scale=0.9, size=cell_count)
    receiver_effect = rng.normal(scale=0.9, size=cell_count)
    ligand_effect = rng.normal(scale=0.8, size=15)
    receptor_effect = rng.normal(scale=0.8, size=17)
    interaction_effect = rng.normal(scale=0.8, size=interaction_count)
    cell_signal = sender_effect[sender_index] - 0.7 * receiver_effect[receiver_index]
    molecular_signal = (
        0.7 * ligand_effect[ligand_index]
        - 0.6 * receptor_effect[receptor_index]
        + interaction_effect[interaction_index]
    )
    noise = rng.normal(scale=0.4, size=n_edges)
    if scenario == "cell_dominant":
        latent = 1.25 * cell_signal + 0.15 * molecular_signal + noise
    elif scenario == "molecular_dominant":
        latent = 0.15 * cell_signal + 1.25 * molecular_signal + noise
    else:
        latent = 0.8 * cell_signal + 0.8 * molecular_signal + noise
    if scenario == "topology_jump":
        jump = (interaction_index % 11 == 0) & (receiver_index >= cell_count // 2)
        latent[jump] *= -1.0
    threshold = np.quantile(np.abs(latent), 0.72)
    active = np.abs(latent) >= threshold
    true_effect = np.where(
        active,
        np.sign(latent) * (1.4 + 0.7 * np.abs(latent)),
        0.0,
    )
    if scenario == "global_null":
        true_effect[:] = 0.0

    observation_rng = np.random.default_rng(seed + 41_000_003)
    baseline = true_effect + observation_rng.normal(
        scale=float(simulation["baseline_noise_sd"]), size=n_edges
    )
    if scenario == "global_null":
        call_evidence = rng.normal(scale=1.0, size=n_edges)
    else:
        call_evidence = float(simulation["lr_call_signal_scale"]) * np.abs(
            true_effect
        ) + observation_rng.normal(
            scale=float(simulation["lr_call_noise_sd"]), size=n_edges
        )
    p_value = 2.0 * norm.sf(np.abs(call_evidence))
    if scenario == "noisy_anchor" or scenario == "wrong_topology":
        anchor = rng.normal(scale=1.5, size=n_edges)
    elif scenario == "antagonistic_anchor":
        anchor = -true_effect + rng.normal(scale=0.75, size=n_edges)
    elif scenario == "global_null":
        anchor = rng.normal(size=n_edges)
    else:
        anchor = true_effect + rng.normal(scale=0.75, size=n_edges)

    view_sender = senders.copy()
    view_receiver = receivers.copy()
    view_ligand = np.array([f"ligand_{index}" for index in ligand_index])
    view_receptor = np.array([f"receptor_{index}" for index in receptor_index])
    view_interaction = np.array([f"interaction_{index}" for index in interaction_index])
    if scenario == "corrupted_sender":
        view_sender = rng.permutation(view_sender)
    elif scenario == "corrupted_interaction":
        view_interaction = rng.permutation(view_interaction)
    elif scenario == "wrong_topology":
        view_sender = rng.permutation(view_sender)
        view_receiver = rng.permutation(view_receiver)
        view_ligand = rng.permutation(view_ligand)
        view_receptor = rng.permutation(view_receptor)
        view_interaction = rng.permutation(view_interaction)

    return pd.DataFrame(
        {
            "scenario": scenario,
            "seed": seed,
            "sender": senders,
            "receiver": receivers,
            "ligand": [f"ligand_{index}" for index in ligand_index],
            "receptor": [f"receptor_{index}" for index in receptor_index],
            "interaction_id": [f"interaction_{index}" for index in interaction_index],
            "view_sender": view_sender,
            "view_receiver": view_receiver,
            "view_ligand": view_ligand,
            "view_receptor": view_receptor,
            "view_interaction": view_interaction,
            "true_effect": true_effect,
            "baseline_stat": baseline,
            "interaction_pvalue": p_value,
            "crychic_anchor": anchor,
        }
    )


def _pair_scores(table: pd.DataFrame, sign_statistic: np.ndarray) -> pd.DataFrame:
    selected = table["interaction_pvalue"].to_numpy(dtype=float) < 0.05
    sender = table["sender"].astype(str).to_numpy()
    receiver = table["receiver"].astype(str).to_numpy()
    working = pd.DataFrame(
        {
            "pair_sender": np.minimum(sender, receiver),
            "pair_receiver": np.maximum(sender, receiver),
            "target_score": selected & (sign_statistic > 0.0),
            "reference_score": selected & (sign_statistic < 0.0),
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


def _method_metrics(
    pairs: pd.DataFrame,
    *,
    method: str,
    split: str,
    scenario: str,
    seed: int,
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
                "method": method,
                "direction": direction,
                "pair_rank_spearman": _safe_spearman(score, truth),
                "top_quartile_auroc": (
                    float(roc_auc_score(label, score))
                    if label.nunique() == 2
                    else np.nan
                ),
                "tie_fraction": float(score.duplicated(keep=False).mean()),
                **fit,
            }
        )
    return records


def evaluate_problem(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual = cast(Mapping[str, Any], config["residual"])
    common = {
        "views": tuple(str(value) for value in residual["views"]),
        "key_columns": tuple(str(value) for value in residual["key_columns"]),
        "lambda_grid": tuple(float(value) for value in residual["lambda_grid"]),
        "anchor_grid": tuple(float(value) for value in residual["anchor_grid"]),
        "folds": int(residual["folds"]),
        "seed": int(residual["seed"]),
        "minimum_cv_improvement": float(residual["minimum_cv_improvement"]),
        "full_gate_improvement": float(residual["full_gate_improvement"]),
    }
    baseline = table["baseline_stat"].to_numpy(dtype=float)
    anchor = table["crychic_anchor"].to_numpy(dtype=float)
    equal_theta, equal_fit, _ = fit_cross_validated_residual(
        table, baseline, anchor, **common
    )
    profiles, reliability = _profiles(table, config)
    adaptive_theta, adaptive_fit, cv = fit_adaptive_cross_validated_residual(
        table,
        baseline,
        anchor,
        profiles=profiles,
        minimum_profile_relative_improvement=float(
            residual["minimum_profile_relative_improvement"]
        ),
        **common,
    )
    maximum_abs = float(residual["sign_correction_max_abs_baseline"])
    equal_sign = conservative_sign_statistic(
        baseline, equal_theta, equal_fit, maximum_abs_baseline=maximum_abs
    )
    adaptive_sign = conservative_sign_statistic(
        baseline, adaptive_theta, adaptive_fit, maximum_abs_baseline=maximum_abs
    )
    scenario = str(table["scenario"].iloc[0])
    seed = int(table["seed"].iloc[0])
    records = []
    records.extend(
        _method_metrics(
            _pair_scores(table, baseline),
            method="liana_baseline",
            split=split,
            scenario=scenario,
            seed=seed,
            fit={},
        )
    )
    records.extend(
        _method_metrics(
            _pair_scores(table, equal_sign),
            method="equal_multiview_rc2",
            split=split,
            scenario=scenario,
            seed=seed,
            fit={
                "fit_profile_name": "all_equal",
                "fit_fallback_gate": equal_fit.fallback_gate,
                "fit_cv_relative_improvement": equal_fit.cv_relative_improvement,
            },
        )
    )
    adaptive_fields = {
        f"fit_{key}": value
        for key, value in adaptive_fit.to_dict().items()
        if key not in {"views", "view_weights"}
    }
    records.extend(
        _method_metrics(
            _pair_scores(table, adaptive_sign),
            method="adaptive_multiview_rc4",
            split=split,
            scenario=scenario,
            seed=seed,
            fit=adaptive_fields,
        )
    )
    cv = cv.copy()
    cv["view_oof_relative_improvements"] = json_view_weights(
        reliability["oof_relative_improvement"]
    )
    cv["reliability_weighted_profile"] = json_view_weights(
        reliability["normalized_view_weight"]
    )
    cv.insert(0, "split", split)
    cv.insert(1, "scenario", scenario)
    cv.insert(2, "seed", seed)
    return pd.DataFrame.from_records(records), cv


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["split", "scenario", "method"], observed=True, sort=True)
        .agg(
            pair_rank_spearman_mean=("pair_rank_spearman", "mean"),
            top_quartile_auroc_mean=("top_quartile_auroc", "mean"),
            tie_fraction_mean=("tie_fraction", "mean"),
            fallback_gate_mean=("fit_fallback_gate", "mean"),
            evaluations=("pair_rank_spearman", "size"),
        )
        .reset_index()
    )


def _acceptance(metrics: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    contract = cast(Mapping[str, Any], config["acceptance"])
    holdout = metrics.loc[metrics["split"].eq("holdout")].copy()
    wide = holdout.pivot(
        index=["scenario", "seed", "direction"],
        columns="method",
        values="pair_rank_spearman",
    )
    wide["delta"] = wide["adaptive_multiview_rc4"] - wide["equal_multiview_rc2"]
    heterogeneous = {str(value) for value in contract["heterogeneous_scenarios"]}
    coherent = {str(value) for value in contract["coherent_safety_scenarios"]}
    heterogeneous_delta = float(
        wide.loc[
            wide.index.get_level_values("scenario").isin(heterogeneous), "delta"
        ].mean()
    )
    coherent_delta = float(
        wide.loc[wide.index.get_level_values("scenario").isin(coherent), "delta"].mean()
    )
    topology_jump_delta = float(
        wide.loc[
            wide.index.get_level_values("scenario") == "topology_jump", "delta"
        ].mean()
    )
    wrong_delta = float(
        wide.loc[
            wide.index.get_level_values("scenario") == "wrong_topology", "delta"
        ].mean()
    )
    adaptive_rows = (
        holdout.loc[holdout["method"].eq("adaptive_multiview_rc4")]
        .drop_duplicates(["scenario", "seed"])
        .copy()
    )
    heterogeneous_selection_rate = float(
        adaptive_rows.loc[
            adaptive_rows["scenario"].isin(heterogeneous), "fit_profile_name"
        ]
        .ne("all_equal")
        .mean()
    )
    null_gate_q95 = float(
        adaptive_rows.loc[
            adaptive_rows["scenario"].eq("global_null"), "fit_fallback_gate"
        ].quantile(0.95)
    )
    checks = {
        "heterogeneous_holdout_mean_delta": heterogeneous_delta,
        "heterogeneous_gain_pass": heterogeneous_delta
        >= float(contract["minimum_heterogeneous_holdout_mean_delta"]),
        "heterogeneous_adaptive_selection_rate": heterogeneous_selection_rate,
        "heterogeneous_selection_pass": heterogeneous_selection_rate
        >= float(contract["minimum_heterogeneous_adaptive_selection_rate"]),
        "coherent_safety_mean_delta": coherent_delta,
        "coherent_safety_pass": coherent_delta
        >= -float(contract["maximum_coherent_mean_regression"]),
        "topology_jump_mean_delta": topology_jump_delta,
        "topology_jump_safety_pass": topology_jump_delta
        >= -float(contract["maximum_topology_jump_mean_regression"]),
        "wrong_topology_mean_delta": wrong_delta,
        "wrong_topology_safety_pass": wrong_delta
        >= -float(contract["wrong_topology_maximum_mean_regression"]),
        "global_null_gate_q95": null_gate_q95,
        "global_null_gate_pass": null_gate_q95
        <= float(contract["global_null_maximum_gate_q95"]),
    }
    checks["accepted"] = bool(
        checks["heterogeneous_gain_pass"]
        and checks["heterogeneous_selection_pass"]
        and checks["coherent_safety_pass"]
        and checks["topology_jump_safety_pass"]
        and checks["wrong_topology_safety_pass"]
        and checks["global_null_gate_pass"]
    )
    return checks


def run(
    config_path: Path, output_dir: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_config(config_path)
    output = prepare_output(output_dir, overwrite=overwrite)
    metric_tables = []
    cv_tables = []
    for split, field in (
        ("development", "development_seeds"),
        ("holdout", "holdout_seeds"),
    ):
        for scenario in config["simulation"]["scenarios"]:
            for seed in config[field]:
                problem = simulate_adaptive_problem(
                    config, scenario=str(scenario), seed=int(seed)
                )
                metrics, cv = evaluate_problem(problem, config, split=split)
                metric_tables.append(metrics)
                cv_tables.append(cv)
    metrics = pd.concat(metric_tables, ignore_index=True)
    cv = pd.concat(cv_tables, ignore_index=True)
    aggregate = _aggregate(metrics)
    acceptance = _acceptance(metrics, config)
    profile_selection = (
        metrics.loc[metrics["method"].eq("adaptive_multiview_rc4")]
        .drop_duplicates(["split", "scenario", "seed"])
        .groupby(["split", "scenario", "fit_profile_name"], observed=True, sort=True)
        .size()
        .rename("replicates")
        .reset_index()
    )
    outputs = {
        "replicate_metrics.tsv": metrics,
        "aggregate_metrics.tsv": aggregate,
        "profile_selection.tsv": profile_selection,
        "cv_diagnostics.tsv": cv,
    }
    for filename, table in outputs.items():
        table.to_csv(output / filename, sep="\t", index=False, lineterminator="\n")
    write_json(output / "acceptance.json", acceptance)
    manifest = {
        "schema_version": "crychic-adaptive-multiview-residual-result-v1",
        "status": "complete",
        "accepted": acceptance["accepted"],
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "acceptance": acceptance,
        "selection": {
            "within_replicate_edge_cv_profile_fitting": True,
            "holdout_seeds_used_for_policy_selection": False,
            "real_cohorts_used_for_selection": False,
        },
        "limitations": [
            "effect_summary_simulation_not_full_pseudobulk_pipeline",
            "view_reliability_is_group_mean_predictive_not_outcome_calibrated",
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
