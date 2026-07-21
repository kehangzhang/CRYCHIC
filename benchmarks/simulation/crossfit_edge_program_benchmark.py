"""Independent simulation benchmark for RC7 cross-fitted edge programs."""

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
from benchmarks.literature.crossfit_edge_program import (
    crossfit_program_candidates,
)
from benchmarks.literature.liana_hypergraph_residual import (
    conservative_sign_statistic,
    fit_cross_validated_residual,
)
from benchmarks.literature.receiver_program_soft import (
    fit_receiver_program_reliability,
    receiver_program_weights,
)
from benchmarks.literature.two_part_occurrence import (
    fit_hurdle_channel_reliability,
    fit_subject_hurdle_effects,
    hurdle_directional_weights,
)
from benchmarks.simulation.two_part_occurrence_benchmark import (
    simulate_two_part_problem,
)


def _read_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RC7 config must contain an object")
    return cast(dict[str, Any], value)


def _portable_config_path(path: Path) -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError("RC7 config must be stored inside the repository") from error


def simulate_crossfit_program_problem(
    config: Mapping[str, Any], *, scenario: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate RC6 hurdle observations plus a sparse isolated-edge safety arm."""

    simulation = cast(Mapping[str, Any], config["simulation"])
    if scenario not in {str(value) for value in simulation["scenarios"]}:
        raise ValueError(f"unknown RC7 program scenario: {scenario}")
    base_scenario = (
        "occurrence_helpful" if scenario == "isolated_occurrence" else scenario
    )
    base_config = copy.deepcopy(dict(config))
    base_simulation = cast(dict[str, Any], base_config["simulation"])
    base_simulation["scenarios"] = [base_scenario]
    table, design = simulate_two_part_problem(
        base_config, scenario=base_scenario, seed=seed
    )
    table["scenario"] = scenario
    if scenario != "isolated_occurrence":
        return table, design

    rng = np.random.default_rng(seed + 31_000_009)
    truth = np.abs(table["true_effect"].to_numpy(dtype=float))
    keep_count = max(
        1, int(np.ceil(float(simulation["isolated_edge_fraction"]) * len(table)))
    )
    keep = np.zeros(len(table), dtype=bool)
    keep[np.argsort(truth, kind="stable")[-keep_count:]] = True
    subject_ids = design["subject_id"].astype(str).tolist()
    permutation = rng.permutation(len(subject_ids))
    presence = table.loc[:, [f"presence_{value}" for value in subject_ids]].to_numpy(
        dtype=float
    )
    magnitude = table.loc[:, [f"magnitude_{value}" for value in subject_ids]].to_numpy(
        dtype=float
    )
    presence[~keep] = presence[~keep][:, permutation]
    magnitude[~keep] = magnitude[~keep][:, permutation]
    table.loc[:, [f"presence_{value}" for value in subject_ids]] = presence
    table.loc[:, [f"magnitude_{value}" for value in subject_ids]] = magnitude
    table["isolated_signal_edge"] = keep
    return table, design


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
            "target_truth": table["true_effect"].to_numpy(dtype=float) > 0.0,
            "reference_truth": table["true_effect"].to_numpy(dtype=float) < 0.0,
        }
    )
    return working.groupby(
        ["pair_sender", "pair_receiver"], observed=True, sort=True
    ).sum()


def evaluate_problem(
    table: pd.DataFrame,
    design: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    direction = np.where(sign_statistic >= 0.0, 1.0, -1.0)
    program_policy = cast(Mapping[str, Any], config["program_policy"])
    program_fit = fit_receiver_program_reliability(
        baseline,
        table["program_z"].to_numpy(dtype=float),
        maximum_alpha=float(program_policy["maximum_alpha"]),
        alpha_cap=float(program_policy["alpha_cap"]),
        minimum_edges=int(program_policy["minimum_concordance_edges"]),
    )
    program_weights, _ = receiver_program_weights(
        direction, table["program_z"].to_numpy(dtype=float), program_fit
    )

    subject_ids = design["subject_id"].astype(str).tolist()
    presence = table.loc[:, [f"presence_{value}" for value in subject_ids]].to_numpy(
        dtype=float
    )
    magnitude = table.loc[:, [f"magnitude_{value}" for value in subject_ids]].to_numpy(
        dtype=float
    )
    reconstruction_policy = cast(Mapping[str, Any], config["reconstruction_policy"])
    reconstructed, reconstruction_fits, fold_diagnostics = crossfit_program_candidates(
        presence,
        config["candidates"],
        minimum_training_subjects=int(
            reconstruction_policy["minimum_training_subjects"]
        ),
        minimum_edges=int(reconstruction_policy["minimum_edges"]),
        minimum_relative_improvement=float(
            reconstruction_policy["minimum_relative_improvement"]
        ),
        full_reliability_improvement=float(
            reconstruction_policy["full_reliability_improvement"]
        ),
    )
    fit_lookup = reconstruction_fits.set_index("candidate")
    hurdle_policy = cast(Mapping[str, Any], config["hurdle_policy"])
    scenario = str(table["scenario"].iloc[0])
    seed = int(table["seed"].iloc[0])
    records: list[dict[str, object]] = []
    for priority, candidate_value in enumerate(config["candidates"]):
        candidate = cast(Mapping[str, Any], candidate_value)
        name = str(candidate["name"])
        hurdle = fit_subject_hurdle_effects(
            reconstructed[name],
            magnitude,
            design["condition"].astype(str).to_numpy(),
            reference="reference",
            target="target",
            beta_prior=float(hurdle_policy["beta_prior"]),
            minimum_subjects=int(hurdle_policy["minimum_subjects"]),
            minimum_active_subjects=int(hurdle_policy["minimum_active_subjects"]),
            conditional_presence_threshold=float(
                hurdle_policy["conditional_presence_threshold"]
            ),
            magnitude_log_sd_floor=float(hurdle_policy["magnitude_log_sd_floor"]),
        )
        occurrence_fit = fit_hurdle_channel_reliability(
            baseline,
            hurdle["occurrence_effect"].to_numpy(dtype=float),
            channel="occurrence",
            maximum_alpha=float(hurdle_policy["occurrence_alpha"]),
            alpha_cap=float(hurdle_policy["channel_alpha_cap"]),
            minimum_edges=int(hurdle_policy["minimum_concordance_edges"]),
        )
        magnitude_fit = fit_hurdle_channel_reliability(
            baseline,
            hurdle["magnitude_effect"].to_numpy(dtype=float),
            channel="magnitude",
            maximum_alpha=float(hurdle_policy["magnitude_alpha"]),
            alpha_cap=float(hurdle_policy["channel_alpha_cap"]),
            minimum_edges=int(hurdle_policy["minimum_concordance_edges"]),
        )
        hurdle_weight, _, _ = hurdle_directional_weights(
            direction,
            hurdle["occurrence_z"].to_numpy(dtype=float),
            hurdle["magnitude_z"].to_numpy(dtype=float),
            occurrence_fit,
            magnitude_fit,
            log_weight_cap=float(hurdle_policy["log_weight_cap"]),
        )
        pairs = _pair_scores(table, sign_statistic, program_weights * hurdle_weight)
        reconstruction = fit_lookup.loc[name]
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
                    "candidate": name,
                    "candidate_priority": priority,
                    "pair_rank_spearman": _safe_spearman(score, truth),
                    "top_quartile_auroc": (
                        float(roc_auc_score(binary, score))
                        if binary.nunique() == 2
                        else np.nan
                    ),
                    "tie_fraction": float(score.duplicated(keep=False).mean()),
                    "reconstruction_rank": int(reconstruction["rank"]),
                    "reconstruction_shrinkage": float(
                        reconstruction["projection_shrinkage"]
                    ),
                    "reconstruction_relative_improvement": float(
                        reconstruction["relative_improvement"]
                    ),
                    "reconstruction_reliability": float(reconstruction["reliability"]),
                    "occurrence_effective_alpha": occurrence_fit.effective_alpha,
                    "magnitude_effective_alpha": magnitude_fit.effective_alpha,
                    "occurrence_estimable_fraction": float(
                        hurdle["occurrence_status"].eq("observed").mean()
                    ),
                }
            )
    fold_diagnostics.insert(0, "split", split)
    fold_diagnostics.insert(1, "scenario", scenario)
    fold_diagnostics.insert(2, "seed", seed)
    return pd.DataFrame.from_records(records), fold_diagnostics


def _candidate_deltas(metrics: pd.DataFrame, scenarios: set[str]) -> pd.Series:
    wide = metrics.pivot(
        index=["scenario", "seed", "direction"],
        columns="candidate",
        values="pair_rank_spearman",
    )
    selected = wide.index.get_level_values("scenario").isin(scenarios)
    return (
        wide.loc[selected]
        .subtract(wide.loc[selected, "raw_rc6_reference"], axis=0)
        .mean()
    )


def _select_candidate(
    metrics: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[str, pd.DataFrame]:
    policy = cast(Mapping[str, Any], config["selection"])
    development = metrics.loc[metrics["split"].eq("development")].copy()
    primary = {str(value) for value in policy["primary_scenarios"]}
    coordinated = {str(value) for value in policy["coordinated_scenarios"]}
    safety = {str(value) for value in policy["safety_scenarios"]}
    primary_delta = _candidate_deltas(development, primary)
    coordinated_delta = _candidate_deltas(development, coordinated)
    safety_delta = _candidate_deltas(development, safety)
    records = []
    for candidate in development["candidate"].drop_duplicates():
        rows = development.loc[development["candidate"].eq(candidate)]
        null_q95 = float(
            rows.loc[
                rows["scenario"].eq("global_null"),
                "reconstruction_reliability",
            ].quantile(0.95)
        )
        safety_pass = float(safety_delta[candidate]) >= -float(
            policy["maximum_development_safety_mean_regression"]
        )
        null_pass = null_q95 <= float(
            policy["global_null_maximum_reconstruction_reliability_q95"]
        )
        records.append(
            {
                "candidate": candidate,
                "candidate_priority": int(rows["candidate_priority"].iloc[0]),
                "primary_mean_delta_vs_rc6": float(primary_delta[candidate]),
                "coordinated_mean_delta_vs_rc6": float(coordinated_delta[candidate]),
                "safety_mean_delta_vs_rc6": float(safety_delta[candidate]),
                "global_null_reconstruction_reliability_q95": null_q95,
                "safety_pass": safety_pass,
                "global_null_pass": null_pass,
                "eligible": safety_pass and null_pass,
            }
        )
    selection = pd.DataFrame.from_records(records).sort_values(
        [
            "eligible",
            "primary_mean_delta_vs_rc6",
            "safety_mean_delta_vs_rc6",
            "candidate_priority",
        ],
        ascending=[False, False, False, True],
        kind="stable",
        ignore_index=True,
    )
    if not selection["eligible"].any():
        raise ValueError("no RC7 candidate passed development safety gates")
    selection.insert(0, "development_rank", np.arange(1, len(selection) + 1))
    selection["selected"] = False
    selection.loc[0, "selected"] = True
    return str(selection.loc[0, "candidate"]), selection


def _acceptance(
    metrics: pd.DataFrame, config: Mapping[str, Any], selected: str
) -> dict[str, object]:
    policy = cast(Mapping[str, Any], config["selection"])
    holdout = metrics.loc[metrics["split"].eq("holdout")]
    primary = {str(value) for value in policy["primary_scenarios"]}
    coordinated = {str(value) for value in policy["coordinated_scenarios"]}
    safety = {str(value) for value in policy["safety_scenarios"]}
    primary_delta = float(_candidate_deltas(holdout, primary)[selected])
    coordinated_delta = float(_candidate_deltas(holdout, coordinated)[selected])
    safety_delta = float(_candidate_deltas(holdout, safety)[selected])
    structural_delta = float(
        _candidate_deltas(holdout, {"structural_missingness"})[selected]
    )
    isolated_delta = float(
        _candidate_deltas(holdout, {"isolated_occurrence"})[selected]
    )
    selected_rows = holdout.loc[holdout["candidate"].eq(selected)]
    null_q95 = float(
        selected_rows.loc[
            selected_rows["scenario"].eq("global_null"),
            "reconstruction_reliability",
        ].quantile(0.95)
    )
    novel = selected != "raw_rc6_reference"
    checks = {
        "selected_candidate": selected,
        "novel_candidate": novel,
        "novel_candidate_pass": novel or not bool(policy["novel_candidate_required"]),
        "holdout_primary_mean_delta_vs_rc6": primary_delta,
        "holdout_primary_gain_pass": primary_delta
        >= float(policy["minimum_holdout_primary_mean_delta_vs_rc6"]),
        "holdout_coordinated_mean_delta_vs_rc6": coordinated_delta,
        "holdout_coordinated_gain_pass": coordinated_delta
        >= float(policy["minimum_holdout_coordinated_mean_delta_vs_rc6"]),
        "holdout_safety_mean_delta_vs_rc6": safety_delta,
        "holdout_safety_pass": safety_delta
        >= -float(policy["maximum_holdout_safety_mean_regression_vs_rc6"]),
        "holdout_structural_missingness_delta_vs_rc6": structural_delta,
        "holdout_structural_missingness_pass": structural_delta
        >= -float(policy["maximum_holdout_structural_missingness_regression_vs_rc6"]),
        "holdout_isolated_occurrence_delta_vs_rc6": isolated_delta,
        "holdout_isolated_occurrence_pass": isolated_delta
        >= -float(policy["maximum_holdout_isolated_occurrence_regression_vs_rc6"]),
        "global_null_reconstruction_reliability_q95": null_q95,
        "global_null_gate_pass": null_q95
        <= float(policy["global_null_maximum_reconstruction_reliability_q95"]),
    }
    checks["accepted"] = bool(
        checks["novel_candidate_pass"]
        and checks["holdout_primary_gain_pass"]
        and checks["holdout_coordinated_gain_pass"]
        and checks["holdout_safety_pass"]
        and checks["holdout_structural_missingness_pass"]
        and checks["holdout_isolated_occurrence_pass"]
        and checks["global_null_gate_pass"]
    )
    return checks


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["split", "scenario", "candidate"], observed=True, sort=True)
        .agg(
            pair_rank_spearman_mean=("pair_rank_spearman", "mean"),
            top_quartile_auroc_mean=("top_quartile_auroc", "mean"),
            tie_fraction_mean=("tie_fraction", "mean"),
            reconstruction_relative_improvement_mean=(
                "reconstruction_relative_improvement",
                "mean",
            ),
            reconstruction_reliability_mean=("reconstruction_reliability", "mean"),
            occurrence_effective_alpha_mean=(
                "occurrence_effective_alpha",
                "mean",
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
    metric_tables = []
    fold_tables = []
    for split, field in (
        ("development", "development_seeds"),
        ("holdout", "holdout_seeds"),
    ):
        for scenario in config["simulation"]["scenarios"]:
            for seed in config[field]:
                table, design = simulate_crossfit_program_problem(
                    config, scenario=str(scenario), seed=int(seed)
                )
                metrics, folds = evaluate_problem(table, design, config, split=split)
                metric_tables.append(metrics)
                fold_tables.append(folds)
    metrics = pd.concat(metric_tables, ignore_index=True)
    folds = pd.concat(fold_tables, ignore_index=True)
    selected, selection = _select_candidate(metrics, config)
    acceptance = _acceptance(metrics, config, selected)
    aggregate = _aggregate(metrics)
    outputs = {
        "replicate_metrics.tsv": metrics,
        "aggregate_metrics.tsv": aggregate,
        "candidate_selection.tsv": selection,
        "fold_diagnostics.tsv": folds,
    }
    for filename, table in outputs.items():
        table.to_csv(output / filename, sep="\t", index=False, lineterminator="\n")
    write_json(output / "acceptance.json", acceptance)
    frozen = {
        "schema_version": "crychic-crossfit-edge-program-frozen-v1",
        "selected_candidate": selected,
        "candidate": next(
            item for item in config["candidates"] if item["name"] == selected
        ),
        "reconstruction_policy": config["reconstruction_policy"],
        "hurdle_policy": config["hurdle_policy"],
        "selected_on": "development_only",
        "holdout_used_for_candidate_selection": False,
        "real_cohorts_used_for_selection": False,
        "accepted_for_real_benchmark": acceptance["accepted"],
        "config_sha256": sha256_file(config_path),
        "formal_release_allowed": False,
    }
    write_json(output / "frozen_candidate.json", frozen)
    manifest = {
        "schema_version": "crychic-crossfit-edge-program-result-v1",
        "status": "complete",
        "accepted": acceptance["accepted"],
        "config": {
            "path": _portable_config_path(config_path),
            "sha256": sha256_file(config_path),
        },
        "selection": frozen,
        "acceptance": acceptance,
        "limitations": [
            "effect_summary_simulation_not_full_pseudobulk_pipeline",
            "program_loadings_are_label_free_but_not_external_atlas_priors",
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
