from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks.simulation.two_sided_program_benchmark import (
    _portable_config_path,
    _select_candidate,
    simulate_two_sided_program_problem,
)


def _config() -> dict[str, object]:
    return {
        "simulation": {
            "cell_type_count": 4,
            "interaction_count": 20,
            "lr_call_signal_scale": 0.45,
            "lr_call_noise_sd": 1.4,
            "program_sign_bias": 2.0,
            "scenarios": ["positive_biased_helpful_program"],
        },
        "selection": {
            "balanced_primary_scenarios": ["balanced_helpful_program"],
            "biased_primary_scenarios": ["positive_biased_helpful_program"],
            "safety_scenarios": ["balanced_noisy_program"],
            "maximum_development_safety_mean_regression": 0.01,
            "global_null_maximum_effective_alpha_q95": 0.25,
        },
    }


def test_biased_program_simulation_is_deterministic_and_less_balanced() -> None:
    config = _config()
    first = simulate_two_sided_program_problem(
        config, scenario="positive_biased_helpful_program", seed=17
    )
    second = simulate_two_sided_program_problem(
        config, scenario="positive_biased_helpful_program", seed=17
    )
    pd.testing.assert_frame_equal(first, second)
    finite = first["program_z"].dropna()
    assert finite.gt(0.0).mean() > 0.75


def test_config_provenance_is_repository_relative() -> None:
    assert (
        _portable_config_path(Path("benchmarks/configs/two_sided_program_rc5_v1.json"))
        == "benchmarks/configs/two_sided_program_rc5_v1.json"
    )


def test_selection_uses_development_and_safety() -> None:
    rows = []
    for split in ("development", "holdout"):
        for scenario in (
            "balanced_helpful_program",
            "positive_biased_helpful_program",
            "balanced_noisy_program",
            "global_null",
        ):
            for priority, candidate, delta in (
                (0, "signed_rc3_reference", 0.0),
                (1, "safe", 0.1),
                (2, "unsafe", 0.2),
            ):
                value = 0.6 + delta
                if scenario == "balanced_noisy_program" and candidate == "unsafe":
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
                        "program_effective_alpha": 0.1,
                    }
                )
    selected, summary = _select_candidate(pd.DataFrame(rows), _config())
    assert selected == "safe"
    assert not bool(summary.loc[summary["candidate"].eq("unsafe"), "eligible"].iloc[0])
