"""Preregistered simulation benchmark for the RC1 signed-cardinality head."""

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
from benchmarks.literature.signed_expected_cardinality import (
    fit_spike_normal_working_prior,
    signed_working_probabilities,
)


def _read_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("signed-cardinality config must contain an object")
    return cast(dict[str, Any], value)


def _scenario_parameters(scenario: str) -> dict[str, float]:
    parameters = {
        "sparse_low_n": {"density": 0.025, "noise": 0.075, "missing": 0.0},
        "dense_low_n": {"density": 0.16, "noise": 0.075, "missing": 0.0},
        "heteroskedastic": {"density": 0.08, "noise": 0.07, "missing": 0.0},
        "opposite_signs": {"density": 0.10, "noise": 0.07, "missing": 0.0},
        "structural_missing": {"density": 0.08, "noise": 0.07, "missing": 0.28},
        "global_null_imbalanced_opportunity": {
            "density": 0.0,
            "noise": 0.075,
            "missing": 0.0,
        },
    }
    if scenario not in parameters:
        raise ValueError(f"unknown signed-cardinality scenario: {scenario}")
    return parameters[scenario]


def simulate_effect_summary(
    config: Mapping[str, Any], *, scenario: str, seed: int
) -> pd.DataFrame:
    """Simulate heterogeneous effect/SE summaries without using the EB prior."""

    simulation = cast(Mapping[str, Any], config["simulation"])
    pair_count = int(simulation["pair_count"])
    opportunities = tuple(int(value) for value in simulation["edge_opportunities"])
    effect_floor = float(simulation["effect_floor"])
    effect_scale = float(simulation["effect_scale"])
    parameters = _scenario_parameters(scenario)
    rng = np.random.default_rng(seed)
    target_latent = rng.normal(size=pair_count)
    reference_latent = (
        -0.65 * target_latent + np.sqrt(1.0 - 0.65**2) * rng.normal(size=pair_count)
        if scenario == "opposite_signs"
        else rng.normal(size=pair_count)
    )

    records: list[dict[str, object]] = []
    for pair in range(pair_count):
        n_edges = opportunities[pair % len(opportunities)]
        target_propensity = 1.0 / (1.0 + np.exp(-target_latent[pair]))
        reference_propensity = 1.0 / (1.0 + np.exp(-reference_latent[pair]))
        density = parameters["density"]
        p_target = density * (0.2 + 1.6 * target_propensity)
        p_reference = density * (0.2 + 1.6 * reference_propensity)
        if p_target + p_reference > 0.8:
            rescale = 0.8 / (p_target + p_reference)
            p_target *= rescale
            p_reference *= rescale
        states = rng.choice(
            np.array([-1, 0, 1], dtype=int),
            size=n_edges,
            p=(p_reference, 1.0 - p_target - p_reference, p_target),
        )
        magnitudes = effect_floor + rng.gamma(
            shape=1.6, scale=effect_scale, size=n_edges
        )
        beta = states * magnitudes
        if scenario == "heteroskedastic":
            se = rng.lognormal(
                mean=np.log(parameters["noise"]), sigma=0.85, size=n_edges
            )
        else:
            se = rng.lognormal(
                mean=np.log(parameters["noise"]), sigma=0.35, size=n_edges
            )
        estimate = beta + rng.normal(scale=se)
        observed = np.ones(n_edges, dtype=bool)
        if parameters["missing"] > 0.0:
            pair_missing = np.clip(
                parameters["missing"] + 0.12 * (pair % 3 - 1), 0.05, 0.6
            )
            observed = rng.random(n_edges) >= pair_missing
        for edge in range(n_edges):
            records.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "pair_id": f"pair_{pair:02d}",
                    "edge_id": f"pair_{pair:02d}_edge_{edge:03d}",
                    "n_opportunities": n_edges,
                    "true_effect": beta[edge],
                    "effect": estimate[edge] if observed[edge] else np.nan,
                    "standard_error": se[edge] if observed[edge] else np.nan,
                    "observed": observed[edge],
                    "true_target_active": bool(observed[edge] and beta[edge] > 0.0),
                    "true_reference_active": bool(observed[edge] and beta[edge] < 0.0),
                }
            )
    return pd.DataFrame.from_records(records)


def _candidate_weights(
    table: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_slab_scale_fraction: float = 0.01,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    effect = table["effect"].to_numpy(dtype=float)
    se = table["standard_error"].to_numpy(dtype=float)
    observed = table["observed"].to_numpy(dtype=bool)
    z = np.divide(
        effect,
        se,
        out=np.zeros(len(table), dtype=float),
        where=observed & (se > 0.0),
    )
    needs_eb = any(candidate["kind"] == "spike_normal" for candidate in candidates)
    fit = (
        fit_spike_normal_working_prior(
            effect,
            se,
            min_fit_edges=200,
            min_slab_scale_fraction=min_slab_scale_fraction,
        )
        if needs_eb
        else None
    )
    outputs: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    for candidate in candidates:
        name = str(candidate["name"])
        kind = str(candidate["kind"])
        metadata: dict[str, object] = {"kind": kind}
        if kind == "hard_z":
            threshold = float(candidate["threshold"])
            selected = observed & (np.abs(z) >= threshold)
            target = (selected & (effect > 0.0)).astype(float)
            reference = (selected & (effect < 0.0)).astype(float)
            metadata["threshold"] = threshold
        elif kind == "continuous_sign":
            evidence = np.zeros(len(table), dtype=float)
            evidence[observed] = 2.0 * (norm.cdf(np.abs(z[observed])) - 0.5)
            target = np.where(observed & (effect > 0.0), evidence, 0.0)
            reference = np.where(observed & (effect < 0.0), evidence, 0.0)
        elif kind == "spike_normal":
            if fit is None:
                raise AssertionError("spike-normal fit was not constructed")
            delta_fraction = float(candidate["delta_fraction"])
            probabilities = signed_working_probabilities(
                effect, se, fit, delta_fraction=delta_fraction
            )
            target = np.nan_to_num(probabilities["working_p_target_active"])
            reference = np.nan_to_num(
                probabilities["working_p_reference_active"]
            )
            metadata.update(fit.to_dict())
            metadata["delta_fraction"] = delta_fraction
        else:
            raise ValueError(f"unsupported candidate kind: {kind}")
        outputs[name] = (target, reference, metadata)
    return outputs


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def evaluate_candidates(
    table: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]],
    *,
    split: str,
    min_slab_scale_fraction: float = 0.01,
) -> pd.DataFrame:
    weights = _candidate_weights(
        table,
        candidates,
        min_slab_scale_fraction=min_slab_scale_fraction,
    )
    records: list[dict[str, object]] = []
    for candidate, (target, reference, metadata) in weights.items():
        working = table.loc[:, ["pair_id", "n_opportunities"]].copy()
        working["target_score"] = target
        working["reference_score"] = reference
        working["target_truth"] = table["true_target_active"].astype(int)
        working["reference_truth"] = table["true_reference_active"].astype(int)
        pairs = working.groupby("pair_id", sort=True, observed=True).agg(
            n_opportunities=("n_opportunities", "first"),
            target_score=("target_score", "sum"),
            reference_score=("reference_score", "sum"),
            target_truth=("target_truth", "sum"),
            reference_truth=("reference_truth", "sum"),
        )
        for direction in ("target", "reference"):
            score = pairs[f"{direction}_score"]
            truth = pairs[f"{direction}_truth"]
            threshold = truth.quantile(0.75)
            label = truth.ge(threshold).astype(int)
            auroc = (
                float(roc_auc_score(label, score)) if label.nunique() == 2 else np.nan
            )
            records.append(
                {
                    "split": split,
                    "scenario": str(table["scenario"].iloc[0]),
                    "seed": int(table["seed"].iloc[0]),
                    "candidate": candidate,
                    "direction": direction,
                    "pair_rank_spearman": _safe_spearman(score, truth),
                    "top_quartile_auroc": auroc,
                    "tie_fraction": float(score.duplicated(keep=False).mean()),
                    "mean_predicted_count": float(score.mean()),
                    "mean_true_count": float(truth.mean()),
                    "mean_opportunities": float(pairs["n_opportunities"].mean()),
                    "opportunity_spearman": _safe_spearman(
                        score, pairs["n_opportunities"]
                    ),
                    "fit_null_weight": metadata.get("null_weight"),
                    "fit_slab_sd": metadata.get("slab_sd"),
                    "fit_converged": metadata.get("converged"),
                }
            )
    return pd.DataFrame.from_records(records)


def _select_candidate(
    records: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[str, pd.DataFrame]:
    selection = cast(Mapping[str, Any], config["selection"])
    eligible = set(str(value) for value in selection["eligible_scenarios"])
    development = records.loc[
        records["split"].eq("development") & records["scenario"].isin(eligible)
    ].copy()
    summary = (
        development.groupby("candidate", observed=True, sort=True)
        .agg(
            pair_rank_spearman=("pair_rank_spearman", "mean"),
            top_quartile_auroc=("top_quartile_auroc", "mean"),
            tie_fraction=("tie_fraction", "mean"),
            evaluations=("pair_rank_spearman", "size"),
        )
        .reset_index()
    )
    null_gate = selection.get("null_gate")
    if isinstance(null_gate, Mapping):
        null_scenario = str(null_gate["scenario"])
        quantile = float(null_gate["quantile"])
        maximum_rate = float(null_gate["maximum_false_count_per_opportunity"])
        null = records.loc[
            records["split"].eq("development")
            & records["scenario"].eq(null_scenario)
        ].copy()
        null["false_count_per_opportunity"] = (
            null["mean_predicted_count"] / null["mean_opportunities"]
        )
        gate = (
            null.groupby("candidate", observed=True, sort=True)[
                "false_count_per_opportunity"
            ]
            .agg(
                null_false_count_rate_mean="mean",
                null_false_count_rate_quantile=lambda values: float(
                    values.quantile(quantile)
                ),
            )
            .reset_index()
        )
        summary = summary.merge(gate, on="candidate", how="left", validate="one_to_one")
        summary["null_gate_pass"] = summary[
            "null_false_count_rate_quantile"
        ].le(maximum_rate)
    else:
        summary["null_false_count_rate_mean"] = np.nan
        summary["null_false_count_rate_quantile"] = np.nan
        summary["null_gate_pass"] = True
    summary = summary.sort_values(
            [
                "null_gate_pass",
                "pair_rank_spearman",
                "top_quartile_auroc",
                "tie_fraction",
                "candidate",
            ],
            ascending=[False, False, False, True, True],
            kind="stable",
            ignore_index=True,
        )
    if not summary["null_gate_pass"].any():
        raise ValueError(
            "no signed-cardinality candidate passed the development null gate"
        )
    summary.insert(0, "development_rank", np.arange(1, len(summary) + 1))
    summary["selected"] = False
    summary.loc[0, "selected"] = True
    return str(summary.loc[0, "candidate"]), summary


def _aggregate(records: pd.DataFrame) -> pd.DataFrame:
    return (
        records.groupby(
            ["split", "scenario", "candidate"], observed=True, sort=True
        )
        .agg(
            pair_rank_spearman_mean=("pair_rank_spearman", "mean"),
            pair_rank_spearman_sd=("pair_rank_spearman", "std"),
            top_quartile_auroc_mean=("top_quartile_auroc", "mean"),
            tie_fraction_mean=("tie_fraction", "mean"),
            opportunity_spearman_mean=("opportunity_spearman", "mean"),
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
    scenarios = [str(value) for value in config["simulation"]["scenarios"]]
    candidates = [cast(Mapping[str, Any], value) for value in config["candidates"]]
    working_prior = cast(Mapping[str, Any], config.get("working_prior", {}))
    min_slab_scale_fraction = float(
        working_prior.get("min_slab_scale_fraction", 0.01)
    )
    tables: list[pd.DataFrame] = []
    for split, field in (
        ("development", "development_seeds"),
        ("holdout", "holdout_seeds"),
    ):
        for scenario in scenarios:
            for seed in config[field]:
                simulation = simulate_effect_summary(
                    config, scenario=scenario, seed=int(seed)
                )
                tables.append(
                    evaluate_candidates(
                        simulation,
                        candidates,
                        split=split,
                        min_slab_scale_fraction=min_slab_scale_fraction,
                    )
                )
    records = pd.concat(tables, ignore_index=True)
    selected, selection = _select_candidate(records, config)
    aggregate = _aggregate(records)
    paths = {
        "replicate_metrics.tsv": records,
        "aggregate_metrics.tsv": aggregate,
        "candidate_selection.tsv": selection,
    }
    for filename, table in paths.items():
        table.to_csv(output / filename, sep="\t", index=False, lineterminator="\n")
    frozen = {
        "schema_version": "crychic-signed-cardinality-frozen-candidate-v1",
        "selected_candidate": selected,
        "selected_on": "development_only",
        "holdout_used_for_selection": False,
        "candidate": next(
            value for value in config["candidates"] if value["name"] == selected
        ),
        "working_prior": dict(working_prior),
        "development_selection_diagnostics": selection.loc[
            selection["candidate"].eq(selected)
        ].iloc[0].to_dict(),
        "probability_status": "candidate_unreleased",
        "formal_release_allowed": False,
        "config_sha256": sha256_file(config_path),
    }
    write_json(output / "frozen_candidate.json", frozen)
    manifest = {
        "schema_version": "crychic-signed-cardinality-benchmark-result-v1",
        "status": "complete",
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "selection": frozen,
        "simulation_level": "effect_and_standard_error_summary_only",
        "limitations": [
            "not_a_full_pipeline_resampling_calibration",
            "working_model_probabilities_are_not_released_scientific_posteriors",
            "real_cohorts_are_not_used_for_candidate_selection",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "code": git_metadata(Path(__file__).resolve().parents[2]),
        "outputs": {
            filename: {
                "rows": len(table),
                "sha256": sha256_file(output / filename),
            }
            for filename, table in paths.items()
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
