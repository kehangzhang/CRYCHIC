from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.comprehensive.evaluate_component_crossover import (
    FROZEN_CANDIDATE_ARM,
    REFERENCE_ARMS,
    _arm_summary,
    _candidate_validation,
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
        "sender_downstream_blend_90_10",
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
    assert layers["sender_downstream_blend_90_10"].tolist() == pytest.approx(
        [0.90 * 0.25, 0.90 * 0.5 + 0.10 * 0.5]
    )
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


def test_candidate_validation_uses_paired_seeds_and_orients_null_metrics() -> None:
    rows: list[dict[str, object]] = []
    for scenario in ("active", "global_null"):
        for seed in range(20):
            for method in (FROZEN_CANDIDATE_ARM, *REFERENCE_ARMS):
                candidate = method == FROZEN_CANDIDATE_ARM
                row: dict[str, object] = {
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "event_coverage": 1.0,
                }
                if scenario == "active":
                    value = 0.8 if candidate else 0.4
                    row.update(
                        {
                            "omnibus_auprc": value,
                            "omnibus_auroc": value,
                            "localization_macro_auprc": value,
                            "direction_accuracy_all_active": value,
                            "positive_direction_ap": value,
                            "negative_direction_ap": value,
                            "effect_all_zero_fraction": 0.0 if candidate else 0.5,
                        }
                    )
                else:
                    row.update(
                        {
                            "effect_standard_deviation": 0.01 if candidate else 0.1,
                            "effect_dynamic_range": 0.02 if candidate else 0.2,
                            "effect_all_zero_fraction": 0.0 if candidate else 0.5,
                        }
                    )
                rows.append(row)

    paired, summary, gate = _candidate_validation(
        pd.DataFrame.from_records(rows),
        bootstrap_replicates=500,
        bootstrap_seed=17,
    )

    assert len(paired) == 400
    primary = summary.loc[
        summary["scenario"].eq("active")
        & summary["metric"].eq("omnibus_auprc")
        & summary["reference_arm"].eq(REFERENCE_ARMS[0])
    ].iloc[0]
    assert primary["oriented_mean_improvement"] == pytest.approx(0.4)
    assert primary["oriented_improvement_ci_lower"] == pytest.approx(0.4)
    null = summary.loc[
        summary["scenario"].eq("global_null")
        & summary["metric"].eq("effect_standard_deviation")
        & summary["reference_arm"].eq(REFERENCE_ARMS[0])
    ].iloc[0]
    assert null["raw_candidate_minus_reference"] == pytest.approx(-0.09)
    assert null["oriented_mean_improvement"] == pytest.approx(0.09)
    assert gate["status"] == "ACCEPT"
