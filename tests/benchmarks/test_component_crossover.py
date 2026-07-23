from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.comprehensive.evaluate_component_crossover import (
    _arm_summary,
    score_layers,
)


def test_score_layers_reconstruct_strict_and_isolate_hard_zero() -> None:
    table = pd.DataFrame(
        {
            "availability": [0.8, 0.4],
            "downstream": [0.0, 0.5],
            "sender_component": [0.25, 0.5],
            "prior_quality": [1.0, 0.5],
            "comm_strength": [0.0, (0.4 * 0.5 * 0.5 * 0.5) ** 0.25],
        }
    )

    layers = score_layers(table)

    assert tuple(layers) == (
        "strict_geometric",
        "availability_only",
        "mechanistic_geometric",
        "availability_downstream_geometric",
        "downstream_only",
        "sender_only",
    )
    assert layers["strict_geometric"].tolist() == pytest.approx(
        table["comm_strength"].tolist()
    )
    assert layers["mechanistic_geometric"].iloc[0] == pytest.approx(
        (0.8 * 0.25 * 1.0) ** (1.0 / 3.0)
    )
    assert layers["mechanistic_geometric"].iloc[0] > 0.0
    assert layers["availability_downstream_geometric"].iloc[0] == 0.0
    assert layers["downstream_only"].iloc[0] == 0.0


def test_arm_summary_ranks_only_complete_crossover_arms() -> None:
    rows = []
    for method, values in (
        ("strict_geometric__within_sample_rank_mean", [0.4, 0.5]),
        ("availability_only__native_raw_mean", [0.7, 0.8]),
        ("sender_only__native_raw_mean", [0.9, np.nan]),
    ):
        for seed, value in enumerate(values):
            rows.append(
                {
                    "dataset_id": f"active-{seed}",
                    "scenario": "active",
                    "method": method,
                    "omnibus_status": "observed" if np.isfinite(value) else "NE",
                    "omnibus_prevalence_adjusted_ap": value,
                    "omnibus_auprc": value,
                    "omnibus_auroc": value,
                    "localization_macro_auprc": value,
                    "localization_micro_auprc": value,
                    "positive_direction_ap": value,
                    "negative_direction_ap": value,
                    "direction_accuracy_all_active": value,
                    "effect_spearman": value,
                    "effect_all_zero_fraction": 0.0,
                    "effect_dynamic_range": 1.0,
                    "event_coverage": 1.0,
                }
            )
    summary = _arm_summary(pd.DataFrame.from_records(rows)).set_index("method")

    winner = "availability_only__native_raw_mean"
    assert summary.loc[winner, "primary_rank"] == 1
    assert bool(summary.loc[winner, "rank_eligible"])
    incomplete = "sender_only__native_raw_mean"
    assert not bool(summary.loc[incomplete, "rank_eligible"])
    assert pd.isna(summary.loc[incomplete, "primary_rank"])
