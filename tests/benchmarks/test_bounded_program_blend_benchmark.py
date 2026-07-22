from __future__ import annotations

import pandas as pd

from benchmarks.simulation.bounded_program_blend_benchmark import (
    _select_candidate,
)


def _config() -> dict[str, object]:
    return {
        "blend_candidates": [
            {
                "name": "safe",
                "relative_fdp_reduction": 0.5,
                "blend_fraction": 0.2,
            },
            {
                "name": "unsafe",
                "relative_fdp_reduction": 0.5,
                "blend_fraction": 0.5,
            },
        ],
        "selection": {
            "primary_scenarios": ["tail"],
            "tail_scenarios": ["tail"],
            "safety_scenarios": ["diffuse", "isolated", "noisy"],
            "diffuse_scenario": "diffuse",
            "isolated_scenario": "isolated",
            "missing_scenario": "missing",
            "minimum_development_primary_mean_delta_vs_rc3": 0.001,
            "minimum_development_tail_mean_delta_vs_rc3": 0.001,
            "maximum_development_safety_mean_regression": 0.01,
            "maximum_development_diffuse_regression": 0.01,
            "maximum_development_isolated_regression": 0.01,
            "global_null_maximum_pair_score_ratio_q95": 1.02,
        },
    }


def test_selection_uses_development_safety_and_ignores_holdout() -> None:
    rows = []
    for split in ("development", "holdout"):
        for scenario in (
            "tail",
            "diffuse",
            "isolated",
            "noisy",
            "missing",
            "global_null",
        ):
            for method in ("rc3_soft", "safe", "unsafe"):
                value = 0.7
                if scenario == "tail" and method == "safe":
                    value = 0.8
                if scenario == "tail" and method == "unsafe":
                    value = 0.85
                if scenario == "diffuse" and method == "unsafe":
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
