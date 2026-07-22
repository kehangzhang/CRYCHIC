"""Independent simulation benchmark for RC9 bounded program-gate blends."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
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
from benchmarks.literature.bounded_program_blend import (
    bounded_program_blend_weights,
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
from benchmarks.simulation.adaptive_program_gate_benchmark import (
    _pair_scores,
    _safe_spearman,
    simulate_adaptive_gate_problem,
)


def _read_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RC9 bounded blend config must contain an object")
    return cast(dict[str, Any], value)


def _portable_config_path(path: Path) -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError("RC9 config must be stored inside the repository") from error


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
                "pair_score_mean": float(score.mean()),
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

    soft_policy = cast(Mapping[str, Any], config["soft_program_policy"])
    soft_fit = fit_receiver_program_reliability(
        baseline,
        program,
        maximum_alpha=float(soft_policy["maximum_alpha"]),
        alpha_cap=float(soft_policy["alpha_cap"]),
        minimum_edges=int(soft_policy["minimum_concordance_edges"]),
    )
    soft_weights, _ = receiver_program_weights(direction, program, soft_fit)
    normalized_soft, _ = bounded_program_blend_weights(
        soft_weights,
        np.ones(len(table), dtype=float),
        call_mask,
        blend_fraction=0.0,
    )
    records = _metric_records(
        _pair_scores(table, sign_statistic, normalized_soft),
        split=split,
        scenario=scenario,
        seed=seed,
        method="rc3_soft",
        diagnostics={
            "gate_mode": "not_applicable",
            "soft_effective_alpha": soft_fit.effective_alpha,
        },
    )

    gate_policy = cast(Mapping[str, Any], config["gate_policy"])
    validation_fold = stable_edge_folds(
        table,
        key_columns=tuple(str(value) for value in residual["key_columns"]),
        folds=int(gate_policy["validation_folds"]),
        seed=int(gate_policy["validation_seed"]),
    )
    gate_cache: dict[float, tuple[np.ndarray, Mapping[str, object]]] = {}
    for candidate_value in config["blend_candidates"]:
        candidate = cast(Mapping[str, Any], candidate_value)
        reduction = float(candidate["relative_fdp_reduction"])
        if reduction not in gate_cache:
            gate_fit = fit_adaptive_program_gate(
                direction,
                program,
                call_mask,
                relative_fdp_reduction=reduction,
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
            gate_weights = adaptive_program_gate_weights(
                direction, program, gate_fit
            )
            gate_diagnostics = {
                f"gate_{key}": value
                for key, value in gate_fit.to_dict().items()
                if key not in {"fit_status", "formal_release_allowed"}
            }
            gate_diagnostics["gate_mode"] = gate_fit.gate_mode
            gate_cache[reduction] = (gate_weights, gate_diagnostics)
        gate_weights, gate_diagnostics = gate_cache[reduction]
        blended, blend_fit = bounded_program_blend_weights(
            soft_weights,
            gate_weights,
            call_mask,
            blend_fraction=float(candidate["blend_fraction"]),
        )
        diagnostics = dict(gate_diagnostics)
        diagnostics.update(
            {
                f"blend_{key}": value
                for key, value in blend_fit.to_dict().items()
                if key not in {"fit_status", "formal_release_allowed"}
            }
        )
        records.extend(
            _metric_records(
                _pair_scores(table, sign_statistic, blended),
                split=split,
                scenario=scenario,
                seed=seed,
                method=str(candidate["name"]),
                diagnostics=diagnostics,
            )
        )
    return pd.DataFrame.from_records(records)


def _scenario_delta(
    wide: pd.DataFrame, candidate: str, scenarios: set[str]
) -> float:
    delta = wide[candidate] - wide["rc3_soft"]
    return float(
        delta.loc[delta.index.get_level_values("scenario").isin(scenarios)].mean()
    )


def _null_score_ratio_q95(
    metrics: pd.DataFrame, *, split: str, candidate: str
) -> float:
    null = metrics.loc[
        metrics["split"].eq(split) & metrics["scenario"].eq("global_null")
    ]
    wide = null.pivot(
        index=["seed", "direction"],
        columns="method",
        values="pair_score_mean",
    )
    supported = wide["rc3_soft"] > 0.0
    if not supported.any():
        return float("nan")
    ratio = wide.loc[supported, candidate] / wide.loc[supported, "rc3_soft"]
    return float(ratio.quantile(0.95))


def _candidate_summary(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    split: str,
) -> pd.DataFrame:
    policy = cast(Mapping[str, Any], config["selection"])
    selected_split = metrics.loc[metrics["split"].eq(split)].copy()
    wide = selected_split.pivot(
        index=["scenario", "seed", "direction"],
        columns="method",
        values="pair_rank_spearman",
    )
    scenario_sets = {
        "primary": {str(value) for value in policy["primary_scenarios"]},
        "tail": {str(value) for value in policy["tail_scenarios"]},
        "safety": {str(value) for value in policy["safety_scenarios"]},
        "diffuse": {str(policy["diffuse_scenario"])},
        "isolated": {str(policy["isolated_scenario"])},
        "missing": {str(policy["missing_scenario"])},
    }
    records: list[dict[str, object]] = []
    for candidate_value in config["blend_candidates"]:
        candidate = cast(Mapping[str, Any], candidate_value)
        name = str(candidate["name"])
        records.append(
            {
                "candidate": name,
                "relative_fdp_reduction": float(
                    candidate["relative_fdp_reduction"]
                ),
                "blend_fraction": float(candidate["blend_fraction"]),
                **{
                    f"{label}_mean_delta_vs_rc3": _scenario_delta(
                        wide, name, scenarios
                    )
                    for label, scenarios in scenario_sets.items()
                },
                "global_null_pair_score_ratio_q95": _null_score_ratio_q95(
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
    selection["primary_pass"] = selection["primary_mean_delta_vs_rc3"].ge(
        float(policy["minimum_development_primary_mean_delta_vs_rc3"])
    )
    selection["tail_pass"] = selection["tail_mean_delta_vs_rc3"].ge(
        float(policy["minimum_development_tail_mean_delta_vs_rc3"])
    )
    selection["safety_pass"] = selection["safety_mean_delta_vs_rc3"].ge(
        -float(policy["maximum_development_safety_mean_regression"])
    )
    selection["diffuse_pass"] = selection["diffuse_mean_delta_vs_rc3"].ge(
        -float(policy["maximum_development_diffuse_regression"])
    )
    selection["isolated_pass"] = selection["isolated_mean_delta_vs_rc3"].ge(
        -float(policy["maximum_development_isolated_regression"])
    )
    selection["global_null_pass"] = selection[
        "global_null_pair_score_ratio_q95"
    ].le(float(policy["global_null_maximum_pair_score_ratio_q95"]))
    gates = [
        "primary_pass",
        "tail_pass",
        "safety_pass",
        "diffuse_pass",
        "isolated_pass",
        "global_null_pass",
    ]
    selection["eligible"] = selection[gates].all(axis=1)
    selection = selection.sort_values(
        [
            "eligible",
            "primary_mean_delta_vs_rc3",
            "tail_mean_delta_vs_rc3",
            "safety_mean_delta_vs_rc3",
            "blend_fraction",
            "candidate",
        ],
        ascending=[False, False, False, False, True, True],
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
        "holdout_primary_mean_delta_vs_rc3": float(
            row["primary_mean_delta_vs_rc3"]
        ),
        "holdout_tail_mean_delta_vs_rc3": float(row["tail_mean_delta_vs_rc3"]),
        "holdout_safety_mean_delta_vs_rc3": float(
            row["safety_mean_delta_vs_rc3"]
        ),
        "holdout_diffuse_mean_delta_vs_rc3": float(
            row["diffuse_mean_delta_vs_rc3"]
        ),
        "holdout_isolated_mean_delta_vs_rc3": float(
            row["isolated_mean_delta_vs_rc3"]
        ),
        "holdout_missing_mean_delta_vs_rc3": float(
            row["missing_mean_delta_vs_rc3"]
        ),
        "global_null_pair_score_ratio_q95": float(
            row["global_null_pair_score_ratio_q95"]
        ),
    }
    checks["holdout_primary_gain_pass"] = checks[
        "holdout_primary_mean_delta_vs_rc3"
    ] >= float(policy["minimum_holdout_primary_mean_delta_vs_rc3"])
    checks["holdout_tail_gain_pass"] = checks[
        "holdout_tail_mean_delta_vs_rc3"
    ] >= float(policy["minimum_holdout_tail_mean_delta_vs_rc3"])
    checks["holdout_safety_pass"] = checks[
        "holdout_safety_mean_delta_vs_rc3"
    ] >= -float(policy["maximum_holdout_safety_mean_regression_vs_rc3"])
    checks["holdout_diffuse_pass"] = checks[
        "holdout_diffuse_mean_delta_vs_rc3"
    ] >= -float(policy["maximum_holdout_diffuse_regression_vs_rc3"])
    checks["holdout_isolated_pass"] = checks[
        "holdout_isolated_mean_delta_vs_rc3"
    ] >= -float(policy["maximum_holdout_isolated_regression_vs_rc3"])
    checks["holdout_missing_pass"] = checks[
        "holdout_missing_mean_delta_vs_rc3"
    ] >= -float(policy["maximum_holdout_missing_regression_vs_rc3"])
    checks["global_null_pass"] = checks[
        "global_null_pair_score_ratio_q95"
    ] <= float(policy["global_null_maximum_pair_score_ratio_q95"])
    checks["accepted"] = bool(
        checks["development_candidate_eligible_pass"]
        and checks["holdout_primary_gain_pass"]
        and checks["holdout_tail_gain_pass"]
        and checks["holdout_safety_pass"]
        and checks["holdout_diffuse_pass"]
        and checks["holdout_isolated_pass"]
        and checks["holdout_missing_pass"]
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
            gate_threshold_mean=("gate_threshold", "mean"),
            blend_fraction_mean=("blend_blend_fraction", "mean"),
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
        for value in config["blend_candidates"]
        if cast(Mapping[str, Any], value)["name"] == selected
    )
    frozen = {
        "schema_version": "crychic-bounded-program-blend-frozen-v1",
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
        "schema_version": "crychic-bounded-program-blend-result-v1",
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
            "binary_gate_is_used_only_as_a_bounded_benchmark_expert",
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
