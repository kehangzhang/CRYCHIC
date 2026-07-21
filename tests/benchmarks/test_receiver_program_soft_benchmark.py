from __future__ import annotations

import pandas as pd

from benchmarks.simulation.receiver_program_soft_benchmark import (
    _select_candidate,
    simulate_program_problem,
)


def _config() -> dict[str, object]:
    return {
        "simulation": {
            "cell_type_count": 4,
            "interaction_count": 20,
            "lr_call_signal_scale": 0.45,
            "lr_call_noise_sd": 1.4,
            "program_scenarios": ["helpful_program", "global_null"],
        },
        "selection": {
            "primary_scenarios": ["helpful_program"],
            "safety_scenarios": ["noisy_program"],
            "maximum_safety_mean_regression": 0.01,
            "global_null_maximum_effective_alpha_q95": 0.25,
        },
    }


def test_program_simulation_is_deterministic_and_group_shared() -> None:
    config = _config()
    first = simulate_program_problem(config, scenario="helpful_program", seed=7)
    second = simulate_program_problem(config, scenario="helpful_program", seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert (
        first.groupby(["receiver", "interaction_id"])["program_z"].nunique().max() == 1
    )


def test_selection_uses_development_safety_and_ignores_holdout() -> None:
    rows = []
    for split in ("development", "holdout"):
        for scenario in ("helpful_program", "noisy_program", "global_null"):
            for candidate, gain, alpha in (
                ("program_max_alpha_0", 0.0, 0.0),
                ("safe", 0.1, 0.1),
                ("unsafe", 0.2, 0.1),
            ):
                value = 0.7 + (0.0 if candidate == "program_max_alpha_0" else gain)
                if scenario == "noisy_program" and candidate == "unsafe":
                    value = 0.5
                if split == "holdout" and candidate == "safe":
                    value = 0.0
                rows.append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "seed": 1,
                        "direction": "target",
                        "candidate": candidate,
                        "pair_rank_spearman": value,
                        "program_effective_alpha": alpha,
                    }
                )
    selected, summary = _select_candidate(pd.DataFrame(rows), _config())
    assert selected == "safe"
    assert not bool(summary.loc[summary["candidate"].eq("unsafe"), "eligible"].iloc[0])
