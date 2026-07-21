"""Independent simulation benchmark for the RC6 two-part hurdle head."""

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
from scipy.special import expit
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
    receiver_program_weights,
)
from benchmarks.literature.two_part_occurrence import (
    fit_hurdle_channel_reliability,
    fit_subject_hurdle_effects,
    hurdle_directional_weights,
)
from benchmarks.simulation.receiver_program_soft_benchmark import (
    simulate_program_problem,
)


def _read_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RC6 config must contain an object")
    return cast(dict[str, Any], value)


def _portable_config_path(path: Path) -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError("RC6 config must be stored inside the repository") from error


def _standardized_truth(table: pd.DataFrame) -> np.ndarray:
    truth = table["true_effect"].to_numpy(dtype=float)
    scale = max(float(np.std(truth)), 1e-8)
    return truth / scale


def simulate_two_part_problem(
    config: Mapping[str, Any], *, scenario: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate subject-level occurrence and conditional positive magnitude."""

    simulation = cast(Mapping[str, Any], config["simulation"])
    allowed = {str(value) for value in simulation["scenarios"]}
    if scenario not in allowed:
        raise ValueError(f"unknown RC6 hurdle scenario: {scenario}")
    program_by_scenario = {
        "occurrence_helpful": "helpful_program",
        "magnitude_helpful": "helpful_program",
        "concordant_helpful": "helpful_program",
        "partial_responders": "helpful_program",
        "weak_two_part": "weak_program",
        "structural_missingness": "helpful_program",
        "abundance_only": "helpful_program",
        "noisy_two_part": "helpful_program",
        "antagonistic_two_part": "helpful_program",
        "global_null": "global_null",
    }
    program_scenario = program_by_scenario[scenario]
    base_config = copy.deepcopy(dict(config))
    base_simulation = cast(dict[str, Any], base_config["simulation"])
    base_simulation["program_scenarios"] = [program_scenario]
    table = simulate_program_problem(
        base_config, scenario=program_scenario, seed=seed
    ).copy()
    table["scenario"] = scenario

    reference_count = int(simulation["reference_subjects"])
    target_count = int(simulation["target_subjects"])
    design = pd.DataFrame(
        {
            "subject_id": [
                *(f"R{index:02d}" for index in range(reference_count)),
                *(f"T{index:02d}" for index in range(target_count)),
            ],
            "condition": ["reference"] * reference_count + ["target"] * target_count,
        }
    )
    rng = np.random.default_rng(seed + 17_000_003)
    truth = _standardized_truth(table)
    edge_count = len(table)
    occurrence_scale = float(simulation["occurrence_log_odds_scale"])
    magnitude_scale = float(simulation["magnitude_log_scale"])
    occurrence_delta = np.zeros(edge_count, dtype=float)
    magnitude_delta = np.zeros(edge_count, dtype=float)
    if scenario == "occurrence_helpful":
        occurrence_delta = occurrence_scale * truth
    elif scenario == "magnitude_helpful":
        magnitude_delta = magnitude_scale * truth
    elif scenario in {
        "concordant_helpful",
        "partial_responders",
        "structural_missingness",
    }:
        occurrence_delta = occurrence_scale * truth
        magnitude_delta = magnitude_scale * truth
    elif scenario == "weak_two_part":
        occurrence_delta = 0.35 * occurrence_scale * truth
        magnitude_delta = 0.35 * magnitude_scale * truth
    elif scenario == "noisy_two_part":
        occurrence_delta = rng.normal(scale=0.8, size=edge_count)
        magnitude_delta = rng.normal(scale=0.35, size=edge_count)
    elif scenario == "antagonistic_two_part":
        occurrence_delta = -occurrence_scale * truth
        magnitude_delta = -magnitude_scale * truth

    cell_types = sorted(set(table["sender"]).union(set(table["receiver"])))
    cell_index = {name: index for index, name in enumerate(cell_types)}
    sender_index = table["sender"].map(cell_index).to_numpy(dtype=int)
    receiver_index = table["receiver"].map(cell_index).to_numpy(dtype=int)
    if scenario == "abundance_only":
        abundance_shift = rng.normal(scale=0.9, size=len(cell_types))
        occurrence_delta = 0.5 * (
            abundance_shift[sender_index] + abundance_shift[receiver_index]
        )
        magnitude_delta = np.zeros(edge_count, dtype=float)

    base_log_odds = rng.normal(loc=0.35, scale=0.9, size=edge_count)
    base_log_magnitude = rng.normal(loc=0.5, scale=0.45, size=edge_count)
    cell_random = rng.normal(scale=0.3, size=(len(design), len(cell_types)))
    responder = np.ones(len(design), dtype=float)
    if scenario == "partial_responders":
        target_indices = np.flatnonzero(design["condition"].eq("target"))
        responder[target_indices] = 0.0
        responder_count = max(
            1,
            int(
                np.ceil(
                    float(simulation["partial_responder_fraction"])
                    * len(target_indices)
                )
            ),
        )
        selected = rng.choice(target_indices, size=responder_count, replace=False)
        responder[selected] = 1.0

    missing_cell_type: list[str | None] = [None] * len(design)
    if scenario == "structural_missingness":
        for subject in range(len(design)):
            if rng.random() < float(simulation["structural_missing_probability"]):
                missing_cell_type[subject] = str(rng.choice(cell_types))

    for subject, row in enumerate(design.itertuples(index=False)):
        target_subject = row.condition == "target"
        subject_cell = 0.5 * (
            cell_random[subject, sender_index] + cell_random[subject, receiver_index]
        )
        condition_multiplier = responder[subject] if target_subject else 0.0
        probability = expit(
            base_log_odds + subject_cell + condition_multiplier * occurrence_delta
        )
        presence = rng.binomial(1, probability, size=edge_count).astype(float)
        log_magnitude = (
            base_log_magnitude
            + 0.2 * subject_cell
            + condition_multiplier * magnitude_delta
            + rng.normal(scale=0.35, size=edge_count)
        )
        magnitude = np.where(presence > 0.0, np.exp(log_magnitude), np.nan)
        missing = missing_cell_type[subject]
        if missing is not None:
            structurally_absent = (
                table["sender"].eq(missing).to_numpy()
                | table["receiver"].eq(missing).to_numpy()
            )
            presence[structurally_absent] = np.nan
            magnitude[structurally_absent] = np.nan
        table[f"presence_{row.subject_id}"] = presence
        table[f"magnitude_{row.subject_id}"] = magnitude
    design["responder"] = responder
    design["structurally_missing_cell_type"] = missing_cell_type
    return table, design


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


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
            "opportunity": np.ones(len(table), dtype=int),
            "selected_target": selected & (sign_statistic > 0.0),
            "selected_reference": selected & (sign_statistic < 0.0),
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
        direction,
        table["program_z"].to_numpy(dtype=float),
        program_fit,
    )

    subject_ids = design["subject_id"].astype(str).tolist()
    presence = table.loc[:, [f"presence_{value}" for value in subject_ids]].to_numpy(
        dtype=float
    )
    magnitude = table.loc[:, [f"magnitude_{value}" for value in subject_ids]].to_numpy(
        dtype=float
    )
    hurdle_policy = cast(Mapping[str, Any], config["hurdle_policy"])
    hurdle = fit_subject_hurdle_effects(
        presence,
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

    scenario = str(table["scenario"].iloc[0])
    seed = int(table["seed"].iloc[0])
    records: list[dict[str, object]] = []
    reference_pairs: pd.DataFrame | None = None
    for priority, candidate_value in enumerate(config["candidates"]):
        candidate = cast(Mapping[str, Any], candidate_value)
        occurrence_fit = fit_hurdle_channel_reliability(
            baseline,
            hurdle["occurrence_effect"].to_numpy(dtype=float),
            channel="occurrence",
            maximum_alpha=float(candidate["occurrence_alpha"]),
            alpha_cap=float(hurdle_policy["channel_alpha_cap"]),
            minimum_edges=int(hurdle_policy["minimum_concordance_edges"]),
        )
        magnitude_fit = fit_hurdle_channel_reliability(
            baseline,
            hurdle["magnitude_effect"].to_numpy(dtype=float),
            channel="magnitude",
            maximum_alpha=float(candidate["magnitude_alpha"]),
            alpha_cap=float(hurdle_policy["channel_alpha_cap"]),
            minimum_edges=int(hurdle_policy["minimum_concordance_edges"]),
        )
        hurdle_weights, _, _ = hurdle_directional_weights(
            direction,
            hurdle["occurrence_z"].to_numpy(dtype=float),
            hurdle["magnitude_z"].to_numpy(dtype=float),
            occurrence_fit,
            magnitude_fit,
            log_weight_cap=float(hurdle_policy["log_weight_cap"]),
        )
        pairs = _pair_scores(table, sign_statistic, program_weights * hurdle_weights)
        if str(candidate["name"]) == "rc3_reference":
            reference_pairs = pairs
        if reference_pairs is None:
            raise ValueError("rc3_reference must be the first RC6 candidate")
        for label in ("target", "reference"):
            score = pairs[f"{label}_score"]
            reference_score = reference_pairs[f"{label}_score"]
            truth = pairs[f"{label}_truth"]
            threshold = truth.quantile(0.75)
            binary = truth.ge(threshold).astype(int)
            null_diagnostic = scenario == "global_null"
            reference_mean = float(reference_score.mean())
            null_mean_ratio = (
                float(score.mean() / reference_mean)
                if null_diagnostic and reference_mean > 0.0
                else np.nan
            )
            null_opportunity = (
                _safe_spearman(score, pairs["opportunity"])
                if null_diagnostic
                else np.nan
            )
            reference_null_opportunity = (
                _safe_spearman(reference_score, pairs["opportunity"])
                if null_diagnostic
                else np.nan
            )
            records.append(
                {
                    "split": split,
                    "scenario": scenario,
                    "seed": seed,
                    "direction": label,
                    "candidate": str(candidate["name"]),
                    "candidate_priority": priority,
                    "occurrence_alpha": float(candidate["occurrence_alpha"]),
                    "magnitude_alpha": float(candidate["magnitude_alpha"]),
                    "pair_rank_spearman": _safe_spearman(score, truth),
                    "top_quartile_auroc": (
                        float(roc_auc_score(binary, score))
                        if binary.nunique() == 2
                        else np.nan
                    ),
                    "tie_fraction": float(score.duplicated(keep=False).mean()),
                    "null_mean_score_ratio_vs_rc3": null_mean_ratio,
                    "null_opportunity_spearman": null_opportunity,
                    "null_opportunity_spearman_delta_vs_rc3": (
                        null_opportunity - reference_null_opportunity
                        if np.isfinite(null_opportunity)
                        and np.isfinite(reference_null_opportunity)
                        else np.nan
                    ),
                    "occurrence_estimable_fraction": float(
                        hurdle["occurrence_status"].eq("observed").mean()
                    ),
                    "magnitude_estimable_fraction": float(
                        hurdle["magnitude_status"].eq("observed").mean()
                    ),
                    **{
                        f"program_{key}": value
                        for key, value in program_fit.to_dict().items()
                    },
                    **{
                        f"occurrence_{key}": value
                        for key, value in occurrence_fit.to_dict().items()
                    },
                    **{
                        f"magnitude_{key}": value
                        for key, value in magnitude_fit.to_dict().items()
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
        wide.loc[selected].subtract(wide.loc[selected, "rc3_reference"], axis=0).mean()
    )


def _scenario_sets(policy: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        "occurrence": {str(value) for value in policy["occurrence_primary_scenarios"]},
        "magnitude": {str(value) for value in policy["magnitude_primary_scenarios"]},
        "joint": {str(value) for value in policy["joint_primary_scenarios"]},
        "safety": {str(value) for value in policy["safety_scenarios"]},
    }


def _select_candidate(
    metrics: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[str, pd.DataFrame]:
    policy = cast(Mapping[str, Any], config["selection"])
    scenarios = _scenario_sets(policy)
    development = metrics.loc[metrics["split"].eq("development")].copy()
    primary = scenarios["occurrence"] | scenarios["magnitude"] | scenarios["joint"]
    deltas = {
        name: _candidate_deltas(development, values)
        for name, values in {**scenarios, "primary": primary}.items()
    }
    records = []
    for candidate in development["candidate"].drop_duplicates():
        rows = development.loc[development["candidate"].eq(candidate)].copy()
        rows["total_effective_alpha"] = (
            rows["occurrence_effective_alpha"] + rows["magnitude_effective_alpha"]
        )
        null_q95 = float(
            rows.loc[
                rows["scenario"].eq("global_null"), "total_effective_alpha"
            ].quantile(0.95)
        )
        safety_pass = float(deltas["safety"][candidate]) >= -float(
            policy["maximum_development_safety_mean_regression"]
        )
        null_pass = null_q95 <= float(
            policy["global_null_maximum_total_effective_alpha_q95"]
        )
        records.append(
            {
                "candidate": candidate,
                "candidate_priority": int(rows["candidate_priority"].iloc[0]),
                "primary_mean_delta_vs_rc3": float(deltas["primary"][candidate]),
                "occurrence_mean_delta_vs_rc3": float(deltas["occurrence"][candidate]),
                "magnitude_mean_delta_vs_rc3": float(deltas["magnitude"][candidate]),
                "joint_mean_delta_vs_rc3": float(deltas["joint"][candidate]),
                "safety_mean_delta_vs_rc3": float(deltas["safety"][candidate]),
                "global_null_total_effective_alpha_q95": null_q95,
                "safety_pass": safety_pass,
                "global_null_pass": null_pass,
                "eligible": safety_pass and null_pass,
            }
        )
    selection = pd.DataFrame.from_records(records).sort_values(
        [
            "eligible",
            "primary_mean_delta_vs_rc3",
            "safety_mean_delta_vs_rc3",
            "candidate_priority",
        ],
        ascending=[False, False, False, True],
        kind="stable",
        ignore_index=True,
    )
    if not selection["eligible"].any():
        raise ValueError("no RC6 candidate passed development safety gates")
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
    scenarios = _scenario_sets(policy)
    holdout = metrics.loc[metrics["split"].eq("holdout")].copy()
    primary = scenarios["occurrence"] | scenarios["magnitude"] | scenarios["joint"]
    primary_delta = float(_candidate_deltas(holdout, primary)[selected])
    occurrence_delta = float(
        _candidate_deltas(holdout, scenarios["occurrence"])[selected]
    )
    magnitude_delta = float(
        _candidate_deltas(holdout, scenarios["magnitude"])[selected]
    )
    safety_delta = float(_candidate_deltas(holdout, scenarios["safety"])[selected])
    structural_delta = float(
        _candidate_deltas(holdout, {"structural_missingness"})[selected]
    )
    selected_rows = holdout.loc[holdout["candidate"].eq(selected)].copy()
    selected_rows["total_effective_alpha"] = (
        selected_rows["occurrence_effective_alpha"]
        + selected_rows["magnitude_effective_alpha"]
    )
    null_q95 = float(
        selected_rows.loc[
            selected_rows["scenario"].eq("global_null"),
            "total_effective_alpha",
        ].quantile(0.95)
    )
    novel = selected != "rc3_reference"
    checks = {
        "selected_candidate": selected,
        "novel_candidate": novel,
        "novel_candidate_pass": novel or not bool(policy["novel_candidate_required"]),
        "holdout_primary_mean_delta_vs_rc3": primary_delta,
        "holdout_primary_gain_pass": primary_delta
        >= float(policy["minimum_holdout_primary_mean_delta_vs_rc3"]),
        "holdout_occurrence_mean_delta_vs_rc3": occurrence_delta,
        "holdout_occurrence_gain_pass": occurrence_delta
        >= float(policy["minimum_holdout_occurrence_mean_delta_vs_rc3"]),
        "holdout_magnitude_mean_delta_vs_rc3": magnitude_delta,
        "holdout_magnitude_gain_pass": magnitude_delta
        >= float(policy["minimum_holdout_magnitude_mean_delta_vs_rc3"]),
        "holdout_safety_mean_delta_vs_rc3": safety_delta,
        "holdout_safety_pass": safety_delta
        >= -float(policy["maximum_holdout_safety_mean_regression_vs_rc3"]),
        "holdout_structural_missingness_delta_vs_rc3": structural_delta,
        "holdout_structural_missingness_pass": structural_delta
        >= -float(policy["maximum_holdout_structural_missingness_regression_vs_rc3"]),
        "global_null_total_effective_alpha_q95": null_q95,
        "global_null_gate_pass": null_q95
        <= float(policy["global_null_maximum_total_effective_alpha_q95"]),
    }
    checks["accepted"] = bool(
        checks["novel_candidate_pass"]
        and checks["holdout_primary_gain_pass"]
        and checks["holdout_occurrence_gain_pass"]
        and checks["holdout_magnitude_gain_pass"]
        and checks["holdout_safety_pass"]
        and checks["holdout_structural_missingness_pass"]
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
            occurrence_effective_alpha_mean=(
                "occurrence_effective_alpha",
                "mean",
            ),
            magnitude_effective_alpha_mean=("magnitude_effective_alpha", "mean"),
            occurrence_estimable_fraction_mean=(
                "occurrence_estimable_fraction",
                "mean",
            ),
            magnitude_estimable_fraction_mean=(
                "magnitude_estimable_fraction",
                "mean",
            ),
            null_mean_score_ratio_vs_rc3_mean=(
                "null_mean_score_ratio_vs_rc3",
                "mean",
            ),
            null_opportunity_spearman_mean=(
                "null_opportunity_spearman",
                "mean",
            ),
            null_opportunity_spearman_delta_vs_rc3_mean=(
                "null_opportunity_spearman_delta_vs_rc3",
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
    tables = []
    for split, field in (
        ("development", "development_seeds"),
        ("holdout", "holdout_seeds"),
    ):
        for scenario in config["simulation"]["scenarios"]:
            for seed in config[field]:
                problem, design = simulate_two_part_problem(
                    config, scenario=str(scenario), seed=int(seed)
                )
                tables.append(evaluate_problem(problem, design, config, split=split))
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
        "schema_version": "crychic-two-part-occurrence-frozen-v1",
        "selected_candidate": selected,
        "candidate": next(
            item for item in config["candidates"] if item["name"] == selected
        ),
        "hurdle_policy": config["hurdle_policy"],
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
        "schema_version": "crychic-two-part-occurrence-result-v1",
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
            "working_beta_and_normal_probabilities_not_formal_inference",
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
