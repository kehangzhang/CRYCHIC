from __future__ import annotations

import pandas as pd

from benchmarks.simulation.paired_rank_contrast_benchmark import (
    _select_candidate,
    simulate_pair_problem,
)


def _config() -> dict[str, object]:
    return {
        "simulation": {
            "pair_count": 20,
            "small_axis_pair_count": 10,
            "observation_noise_sd": 0.8,
            "structural_missing_fraction": 0.2,
            "scenarios": ["mutually_exclusive", "structural_missingness"],
        },
        "candidates": [
            {"name": "safe", "subtraction_fraction": 0.25},
            {"name": "unsafe", "subtraction_fraction": 0.75},
        ],
        "selection": {
            "primary_scenarios": ["primary"],
            "safety_scenarios": ["independent", "asymmetric"],
            "independent_scenario": "independent",
            "asymmetric_scenario": "asymmetric",
            "structural_scenario": "structural",
            "minimum_development_primary_mean_delta_vs_rc9": 0.001,
            "maximum_development_safety_mean_regression": 0.01,
            "maximum_development_independent_regression": 0.01,
            "maximum_development_asymmetric_regression": 0.01,
            "global_null_maximum_pair_score_ratio_q95": 1.02,
        },
    }


def test_pair_simulation_is_deterministic_and_keeps_missingness() -> None:
    config = _config()
    first = simulate_pair_problem(config, scenario="structural_missingness", seed=7)
    second = simulate_pair_problem(config, scenario="structural_missingness", seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["target_score"].isna().any()
    assert first["reference_score"].isna().any()


def test_selection_uses_development_safety_and_ignores_holdout() -> None:
    rows = []
    for split in ("development", "holdout"):
        for scenario in (
            "primary",
            "independent",
            "asymmetric",
            "structural",
            "global_null",
        ):
            for method in ("rc9_rank_reference", "safe", "unsafe"):
                value = 0.7
                if scenario == "primary" and method == "safe":
                    value = 0.8
                if scenario == "primary" and method == "unsafe":
                    value = 0.85
                if scenario == "independent" and method == "unsafe":
                    value = 0.5
                if split == "holdout" and method == "safe":
                    value = 0.0
                rows.append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "seed": 1,
                        "direction": "target",
                        "method": method,
                        "pair_rank_spearman": value,
                        "pair_score_mean": 1.0,
                    }
                )
    selected, summary = _select_candidate(pd.DataFrame(rows), _config())
    assert selected == "safe"
    assert bool(summary.loc[summary["candidate"].eq("safe"), "eligible"].iloc[0])
    assert not bool(
        summary.loc[summary["candidate"].eq("unsafe"), "eligible"].iloc[0]
    )
