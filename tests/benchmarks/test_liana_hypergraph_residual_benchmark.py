from __future__ import annotations

import pandas as pd

from benchmarks.simulation.liana_hypergraph_residual_benchmark import (
    _acceptance,
    simulate_residual_problem,
)


def _config() -> dict[str, object]:
    return {
        "simulation": {
            "cell_type_count": 4,
            "interaction_count": 20,
            "scenarios": ["smooth_helpful_anchor", "global_null"],
        },
        "acceptance": {
            "structured_scenarios": ["smooth_helpful_anchor"],
            "minimum_holdout_mean_delta": 0.0,
            "wrong_topology_maximum_mean_regression": 0.01,
            "global_null_maximum_gate_q95": 0.25,
        },
    }


def test_residual_simulation_is_deterministic_and_has_unique_edge_keys() -> None:
    config = _config()
    first = simulate_residual_problem(
        config, scenario="smooth_helpful_anchor", seed=5
    )
    second = simulate_residual_problem(
        config, scenario="smooth_helpful_anchor", seed=5
    )
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["sender", "receiver", "interaction_id"]).any()


def test_acceptance_requires_gain_safety_and_null_fallback() -> None:
    rows = []
    for scenario, baseline, residual, gate in (
        ("smooth_helpful_anchor", 0.7, 0.8, 1.0),
        ("wrong_topology", 0.7, 0.7, 0.0),
        ("global_null", float("nan"), float("nan"), 0.0),
    ):
        for method, value in (
            ("liana_baseline", baseline),
            ("liana_hypergraph_residual", residual),
        ):
            rows.append(
                {
                    "split": "holdout",
                    "scenario": scenario,
                    "seed": 1,
                    "direction": "target",
                    "method": method,
                    "pair_rank_spearman": value,
                    "fit_fallback_gate": gate,
                }
            )
    result = _acceptance(pd.DataFrame(rows), _config())
    assert result["accepted"]
