"""Independent simulation benchmark for the RC2 LIANA residual fallback."""

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


def _read_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RC2 residual config must contain an object")
    return cast(dict[str, Any], value)


def simulate_residual_problem(
    config: Mapping[str, Any], *, scenario: str, seed: int
) -> pd.DataFrame:
    """Generate signed edge statistics without using the fitted residual model."""

    simulation = cast(Mapping[str, Any], config["simulation"])
    cell_count = int(simulation["cell_type_count"])
    interaction_count = int(simulation["interaction_count"])
    allowed = set(str(value) for value in simulation["scenarios"])
    if scenario not in allowed:
        raise ValueError(f"unknown RC2 simulation scenario: {scenario}")
    rng = np.random.default_rng(seed)
    cell_types = [f"cell_{index}" for index in range(cell_count)]
    senders = np.repeat(cell_types, cell_count * interaction_count)
    receivers = np.tile(
        np.repeat(cell_types, interaction_count), cell_count
    )
    interaction_index = np.tile(np.arange(interaction_count), cell_count**2)
    ligand_index = interaction_index % 15
    receptor_index = (interaction_index * 7 + 3) % 17
    n_edges = len(interaction_index)

    sender_program = rng.normal(scale=0.9, size=cell_count)
    receiver_program = rng.normal(scale=0.9, size=cell_count)
    interaction_program = rng.normal(scale=0.8, size=interaction_count)
    sender_index = np.repeat(np.arange(cell_count), cell_count * interaction_count)
    receiver_index = np.tile(
        np.repeat(np.arange(cell_count), interaction_count), cell_count
    )
    latent = (
        sender_program[sender_index]
        - 0.6 * receiver_program[receiver_index]
        + interaction_program[interaction_index]
        + rng.normal(scale=0.45, size=n_edges)
    )
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

    baseline_stat = true_effect + rng.normal(scale=1.15, size=n_edges)
    evidence_stat = np.abs(true_effect) + rng.normal(scale=1.0, size=n_edges)
    interaction_pvalue = 2.0 * norm.sf(np.abs(evidence_stat))
    if scenario == "smooth_helpful_anchor" or scenario == "topology_jump":
        anchor = true_effect + rng.normal(scale=0.75, size=n_edges)
    elif scenario == "smooth_noisy_anchor" or scenario == "wrong_topology":
        anchor = rng.normal(scale=1.5, size=n_edges)
    elif scenario == "antagonistic_anchor":
        anchor = -true_effect + rng.normal(scale=0.75, size=n_edges)
    else:
        anchor = rng.normal(size=n_edges)

    view_sender = senders.copy()
    view_receiver = receivers.copy()
    view_ligand = np.array([f"ligand_{index}" for index in ligand_index])
    view_receptor = np.array([f"receptor_{index}" for index in receptor_index])
    view_interaction = np.array(
        [f"interaction_{index}" for index in interaction_index]
    )
    if scenario == "wrong_topology":
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
            "interaction_id": [
                f"interaction_{index}" for index in interaction_index
            ],
            "view_sender": view_sender,
            "view_receiver": view_receiver,
            "view_ligand": view_ligand,
            "view_receptor": view_receptor,
            "view_interaction": view_interaction,
            "true_effect": true_effect,
            "baseline_stat": baseline_stat,
            "interaction_pvalue": interaction_pvalue,
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
    theta, fit, cv = fit_cross_validated_residual(
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
        fit,
        maximum_abs_baseline=float(
            residual["sign_correction_max_abs_baseline"]
        ),
    )
    scenario = str(table["scenario"].iloc[0])
    seed = int(table["seed"].iloc[0])
    records = []
    records.extend(
        _method_metrics(
            _pair_scores(table, table["baseline_stat"].to_numpy(dtype=float)),
            method="liana_baseline",
            split=split,
            scenario=scenario,
            seed=seed,
        )
    )
    records.extend(
        _method_metrics(
            _pair_scores(table, sign_statistic),
            method="liana_hypergraph_residual",
            split=split,
            scenario=scenario,
            seed=seed,
        )
    )
    metrics = pd.DataFrame.from_records(records)
    for field, value in fit.to_dict().items():
        if field not in {"views"}:
            metrics[f"fit_{field}"] = value
    cv = cv.copy()
    cv.insert(0, "split", split)
    cv.insert(1, "scenario", scenario)
    cv.insert(2, "seed", seed)
    return metrics, cv


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["split", "scenario", "method"], observed=True, sort=True
        )
        .agg(
            pair_rank_spearman_mean=("pair_rank_spearman", "mean"),
            pair_rank_spearman_sd=("pair_rank_spearman", "std"),
            top_quartile_auroc_mean=("top_quartile_auroc", "mean"),
            tie_fraction_mean=("tie_fraction", "mean"),
            fallback_gate_mean=("fit_fallback_gate", "mean"),
            fallback_gate_q95=(
                "fit_fallback_gate",
                lambda values: float(values.quantile(0.95)),
            ),
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
    ).reset_index()
    wide["delta"] = (
        wide["liana_hypergraph_residual"] - wide["liana_baseline"]
    )
    structured = set(str(value) for value in contract["structured_scenarios"])
    structured_delta = float(
        wide.loc[wide["scenario"].isin(structured), "delta"].mean()
    )
    wrong_delta = float(
        wide.loc[wide["scenario"].eq("wrong_topology"), "delta"].mean()
    )
    null_gate = holdout.loc[
        holdout["scenario"].eq("global_null"), "fit_fallback_gate"
    ]
    null_gate_q95 = float(null_gate.quantile(0.95))
    checks = {
        "structured_holdout_mean_delta": structured_delta,
        "structured_gain_pass": structured_delta
        >= float(contract["minimum_holdout_mean_delta"]),
        "wrong_topology_mean_delta": wrong_delta,
        "wrong_topology_safety_pass": wrong_delta
        >= -float(contract["wrong_topology_maximum_mean_regression"]),
        "global_null_gate_q95": null_gate_q95,
        "global_null_gate_pass": null_gate_q95
        <= float(contract["global_null_maximum_gate_q95"]),
    }
    checks["accepted"] = bool(
        checks["structured_gain_pass"]
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
    metrics_tables = []
    cv_tables = []
    for split, field in (
        ("development", "development_seeds"),
        ("holdout", "holdout_seeds"),
    ):
        for scenario in config["simulation"]["scenarios"]:
            for seed in config[field]:
                table = simulate_residual_problem(
                    config, scenario=str(scenario), seed=int(seed)
                )
                metrics, cv = evaluate_problem(table, config, split=split)
                metrics_tables.append(metrics)
                cv_tables.append(cv)
    metrics = pd.concat(metrics_tables, ignore_index=True)
    cv = pd.concat(cv_tables, ignore_index=True)
    aggregate = _aggregate(metrics)
    acceptance = _acceptance(metrics, config)
    outputs = {
        "replicate_metrics.tsv": metrics,
        "aggregate_metrics.tsv": aggregate,
        "cv_diagnostics.tsv": cv,
    }
    for filename, table in outputs.items():
        table.to_csv(output / filename, sep="\t", index=False, lineterminator="\n")
    write_json(output / "acceptance.json", acceptance)
    manifest = {
        "schema_version": "crychic-liana-hypergraph-residual-result-v1",
        "status": "complete",
        "accepted": acceptance["accepted"],
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "acceptance": acceptance,
        "limitations": [
            "effect_summary_simulation_not_full_pseudobulk_pipeline",
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
