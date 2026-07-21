from __future__ import annotations

import pandas as pd

from benchmarks.simulation.adaptive_multiview_residual_benchmark import (
    _acceptance,
    simulate_adaptive_problem,
)


def _config() -> dict[str, object]:
    return {
        "simulation": {
            "cell_type_count": 4,
            "interaction_count": 20,
            "baseline_noise_sd": 1.8,
            "lr_call_signal_scale": 0.45,
            "lr_call_noise_sd": 1.4,
            "scenarios": [
                "coherent_mixed",
                "corrupted_sender",
                "global_null",
            ],
        },
        "acceptance": {
            "heterogeneous_scenarios": ["corrupted_sender"],
            "coherent_safety_scenarios": ["coherent_mixed"],
            "minimum_heterogeneous_holdout_mean_delta": 0.01,
            "minimum_heterogeneous_adaptive_selection_rate": 0.2,
            "maximum_coherent_mean_regression": 0.01,
            "maximum_topology_jump_mean_regression": 0.01,
            "wrong_topology_maximum_mean_regression": 0.01,
            "global_null_maximum_gate_q95": 0.25,
        },
    }


def test_adaptive_simulation_is_deterministic_and_corrupts_only_view() -> None:
    config = _config()
    first = simulate_adaptive_problem(config, scenario="corrupted_sender", seed=9)
    second = simulate_adaptive_problem(config, scenario="corrupted_sender", seed=9)
    pd.testing.assert_frame_equal(first, second)
    assert first["sender"].ne(first["view_sender"]).any()
    assert (
        first["interaction_id"]
        .str.replace("interaction_", "", regex=False)
        .eq(first["view_interaction"].str.replace("interaction_", "", regex=False))
        .all()
    )


def test_acceptance_uses_holdout_and_all_safety_gates() -> None:
    rows = []
    scenarios = [
        "corrupted_sender",
        "coherent_mixed",
        "topology_jump",
        "wrong_topology",
        "global_null",
    ]
    for split in ("development", "holdout"):
        for scenario in scenarios:
            for method, value in (
                ("equal_multiview_rc2", 0.5),
                ("adaptive_multiview_rc4", 0.52),
            ):
                rows.append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "seed": 1,
                        "direction": "target",
                        "method": method,
                        "pair_rank_spearman": value,
                        "fit_profile_name": (
                            "drop_sender"
                            if method == "adaptive_multiview_rc4"
                            else "all_equal"
                        ),
                        "fit_fallback_gate": 0.0,
                    }
                )
    result = _acceptance(pd.DataFrame(rows), _config())
    assert result["accepted"] is True
