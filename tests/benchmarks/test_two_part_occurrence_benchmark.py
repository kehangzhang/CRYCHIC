from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.simulation.two_part_occurrence_benchmark import (
    _acceptance,
    _portable_config_path,
    _select_candidate,
    simulate_two_part_problem,
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
            "structural_missing_probability": 1.0,
            "scenarios": ["structural_missingness"],
        },
        "selection": {
            "occurrence_primary_scenarios": ["occurrence_helpful"],
            "magnitude_primary_scenarios": ["magnitude_helpful"],
            "joint_primary_scenarios": ["concordant_helpful"],
            "safety_scenarios": ["abundance_only"],
            "maximum_development_safety_mean_regression": 0.01,
            "global_null_maximum_total_effective_alpha_q95": 0.25,
            "minimum_holdout_primary_mean_delta_vs_rc3": 0.01,
            "minimum_holdout_occurrence_mean_delta_vs_rc3": 0.01,
            "minimum_holdout_magnitude_mean_delta_vs_rc3": 0.01,
            "maximum_holdout_safety_mean_regression_vs_rc3": 0.01,
            "maximum_holdout_structural_missingness_regression_vs_rc3": 0.01,
            "novel_candidate_required": True,
        },
    }


def test_two_part_simulation_is_deterministic_and_keeps_structural_missingness() -> (
    None
):
    first_table, first_design = simulate_two_part_problem(
        _config(), scenario="structural_missingness", seed=17
    )
    second_table, second_design = simulate_two_part_problem(
        _config(), scenario="structural_missingness", seed=17
    )
    pd.testing.assert_frame_equal(first_table, second_table)
    pd.testing.assert_frame_equal(first_design, second_design)
    assert first_design["structurally_missing_cell_type"].notna().all()
    presence_columns = first_table.filter(like="presence_")
    assert presence_columns.isna().any().any()
    reference = first_design.loc[
        first_design["condition"].eq("reference"), "subject_id"
    ]
    target = first_design.loc[first_design["condition"].eq("target"), "subject_id"]
    occurrence_effect = first_table[[f"presence_{value}" for value in target]].mean(
        axis=1
    ) - first_table[[f"presence_{value}" for value in reference]].mean(axis=1)
    finite = occurrence_effect.notna()
    assert (
        np.corrcoef(
            occurrence_effect.loc[finite], first_table.loc[finite, "true_effect"]
        )[0, 1]
        > 0.2
    )


def test_selection_uses_development_primary_and_safety_only() -> None:
    rows = []
    scenarios = (
        "occurrence_helpful",
        "magnitude_helpful",
        "concordant_helpful",
        "structural_missingness",
        "abundance_only",
        "global_null",
    )
    for split in ("development", "holdout"):
        for scenario in scenarios:
            for priority, candidate, delta in (
                (0, "rc3_reference", 0.0),
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
                        "occurrence_effective_alpha": 0.1,
                        "magnitude_effective_alpha": 0.1,
                    }
                )
    selected, summary = _select_candidate(pd.DataFrame(rows), _config())
    assert selected == "safe"
    assert not bool(summary.loc[summary["candidate"].eq("unsafe"), "eligible"].iloc[0])
    acceptance = _acceptance(pd.DataFrame(rows), _config(), selected)
    assert acceptance["accepted"] is True
    assert acceptance["holdout_structural_missingness_pass"] is True


def test_config_provenance_is_repository_relative() -> None:
    assert (
        _portable_config_path(
            Path("benchmarks/configs/two_part_occurrence_rc6_v1.json")
        )
        == "benchmarks/configs/two_part_occurrence_rc6_v1.json"
    )
