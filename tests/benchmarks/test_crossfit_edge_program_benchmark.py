from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks.simulation.crossfit_edge_program_benchmark import (
    _acceptance,
    _portable_config_path,
    _select_candidate,
    simulate_crossfit_program_problem,
)


def _config() -> dict[str, object]:
    return {
        "simulation": {
            "cell_type_count": 4,
            "interaction_count": 20,
            "reference_subjects": 4,
            "target_subjects": 4,
            "lr_call_signal_scale": 0.45,
            "lr_call_noise_sd": 1.4,
            "occurrence_log_odds_scale": 1.4,
            "magnitude_log_scale": 0.65,
            "partial_responder_fraction": 0.5,
            "structural_missing_probability": 0.25,
            "isolated_edge_fraction": 0.1,
            "scenarios": ["isolated_occurrence"],
        },
        "selection": {
            "primary_scenarios": ["occurrence_helpful", "structural_missingness"],
            "coordinated_scenarios": ["occurrence_helpful"],
            "safety_scenarios": ["isolated_occurrence", "abundance_only"],
            "maximum_development_safety_mean_regression": 0.01,
            "global_null_maximum_reconstruction_reliability_q95": 0.2,
            "minimum_holdout_primary_mean_delta_vs_rc6": 0.01,
            "minimum_holdout_coordinated_mean_delta_vs_rc6": 0.01,
            "maximum_holdout_safety_mean_regression_vs_rc6": 0.01,
            "maximum_holdout_structural_missingness_regression_vs_rc6": 0.01,
            "maximum_holdout_isolated_occurrence_regression_vs_rc6": 0.01,
            "novel_candidate_required": True,
        },
    }


def test_isolated_program_simulation_is_deterministic_and_sparse() -> None:
    first_table, first_design = simulate_crossfit_program_problem(
        _config(), scenario="isolated_occurrence", seed=17
    )
    second_table, second_design = simulate_crossfit_program_problem(
        _config(), scenario="isolated_occurrence", seed=17
    )
    pd.testing.assert_frame_equal(first_table, second_table)
    pd.testing.assert_frame_equal(first_design, second_design)
    assert first_table["isolated_signal_edge"].mean() == 0.1


def test_selection_and_acceptance_use_frozen_scenario_families() -> None:
    rows = []
    scenarios = (
        "occurrence_helpful",
        "structural_missingness",
        "isolated_occurrence",
        "abundance_only",
        "global_null",
    )
    for split in ("development", "holdout"):
        for scenario in scenarios:
            for priority, candidate, delta in (
                (0, "raw_rc6_reference", 0.0),
                (1, "safe", 0.1),
                (2, "unsafe", 0.2),
            ):
                value = 0.6 + delta
                if scenario == "abundance_only" and candidate == "unsafe":
                    value = 0.3
                rows.append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "seed": 1,
                        "direction": "target",
                        "candidate": candidate,
                        "candidate_priority": priority,
                        "pair_rank_spearman": value,
                        "reconstruction_reliability": 0.1,
                    }
                )
    metrics = pd.DataFrame(rows)
    selected, summary = _select_candidate(metrics, _config())
    assert selected == "safe"
    assert not bool(summary.loc[summary["candidate"].eq("unsafe"), "eligible"].iloc[0])
    acceptance = _acceptance(metrics, _config(), selected)
    assert acceptance["accepted"] is True


def test_config_provenance_is_repository_relative() -> None:
    assert (
        _portable_config_path(
            Path("benchmarks/configs/crossfit_edge_program_rc7_v1.json")
        )
        == "benchmarks/configs/crossfit_edge_program_rc7_v1.json"
    )
