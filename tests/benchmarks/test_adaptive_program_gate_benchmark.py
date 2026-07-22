from __future__ import annotations

import pandas as pd

from benchmarks.simulation.adaptive_program_gate_benchmark import (
    _select_candidate,
    simulate_adaptive_gate_problem,
)


def _config() -> dict[str, object]:
    return {
        "simulation": {
            "cell_type_count": 4,
            "interaction_count": 20,
            "lr_call_signal_scale": 0.45,
            "lr_call_noise_sd": 1.4,
            "missing_program_fraction": 0.4,
            "program_scenarios": ["tail_separable_program", "global_null"],
        },
        "gate_candidates": [
            {"name": "mirror_reduction_100", "relative_fdp_reduction": 1.0},
            {"name": "safe", "relative_fdp_reduction": 0.5},
            {"name": "unsafe", "relative_fdp_reduction": 0.25},
        ],
        "selection": {
            "primary_scenarios": ["tail_separable_program"],
            "tail_scenarios": ["tail_separable_program"],
            "safety_scenarios": ["noisy_program"],
            "flat_scenario": "dense_flat_program",
            "missing_scenario": "missing_tail_program",
            "minimum_development_primary_mean_delta_vs_rc3": 0.001,
            "minimum_development_tail_mean_delta_vs_rc3": 0.001,
            "maximum_development_safety_mean_regression": 0.01,
            "maximum_development_flat_regression_vs_sign_only": 0.01,
            "global_null_maximum_gate_activation_rate": 0.1,
            "novel_candidate_required": True,
        },
    }


def test_simulation_is_deterministic_and_program_is_group_shared() -> None:
    config = _config()
    first = simulate_adaptive_gate_problem(
        config, scenario="tail_separable_program", seed=7
    )
    second = simulate_adaptive_gate_problem(
        config, scenario="tail_separable_program", seed=7
    )
    pd.testing.assert_frame_equal(first, second)
    assert (
        first.groupby(["receiver", "interaction_id"])["program_z"].nunique().max()
        == 1
    )


def test_selection_uses_development_only_and_rejects_unsafe_candidate() -> None:
    rows = []
    scenarios = (
        "tail_separable_program",
        "dense_flat_program",
        "missing_tail_program",
        "noisy_program",
        "global_null",
    )
    for split in ("development", "holdout"):
        for scenario in scenarios:
            for method in (
                "rc3_soft",
                "mirror_reduction_100",
                "safe",
                "unsafe",
            ):
                value = 0.7
                if scenario == "tail_separable_program" and method == "safe":
                    value = 0.8
                if scenario == "tail_separable_program" and method == "unsafe":
                    value = 0.85
                if scenario == "noisy_program" and method == "unsafe":
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
                        "gate_mode": "no_gate",
                    }
                )
    selected, summary = _select_candidate(pd.DataFrame(rows), _config())
    assert selected == "safe"
    assert not bool(summary.loc[summary["candidate"].eq("unsafe"), "eligible"].iloc[0])


def test_selection_persists_best_candidate_when_all_development_gates_fail() -> None:
    rows = []
    for scenario in (
        "tail_separable_program",
        "dense_flat_program",
        "missing_tail_program",
        "noisy_program",
        "global_null",
    ):
        for method in ("rc3_soft", "mirror_reduction_100", "safe", "unsafe"):
            rows.append(
                {
                    "split": "development",
                    "scenario": scenario,
                    "seed": 1,
                    "direction": "target",
                    "method": method,
                    "pair_rank_spearman": 0.7,
                    "gate_mode": "no_gate",
                }
            )
    selected, summary = _select_candidate(pd.DataFrame(rows), _config())
    assert selected == "mirror_reduction_100"
    assert not bool(summary.loc[0, "eligible"])
    assert bool(summary.loc[0, "selected"])
