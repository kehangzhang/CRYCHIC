from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.comprehensive.evaluate_m0_absolute_activity import (
    CANONICAL_ARM,
    M0_ARM,
    RC12_ARM,
    STRICT_ARM,
    _gate,
    _layer_geometry,
    _paired_comparison,
)


def _config() -> dict[str, object]:
    return {
        "validation": {
            "seeds": [1, 2, 3],
            "minimum_paired_seeds": 3,
        },
        "gates": {
            "auprc_delta_ci_lower_minimum": -0.01,
            "auroc_delta_ci_lower_minimum_exclusive": 0.0,
            "direction_accuracy_maximum_degradation": 0.02,
            "minimum_event_coverage": 0.8,
            "require_sender_invariant_tests": True,
        },
    }


def _summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "method": M0_ARM,
                "direction_accuracy": 0.90,
                "minimum_event_coverage": 1.0,
                "mean_zero_fraction": 0.0,
                "mean_tie_fraction": 0.10,
            },
            {
                "method": RC12_ARM,
                "direction_accuracy": 0.70,
                "minimum_event_coverage": 1.0,
                "mean_zero_fraction": 0.0,
                "mean_tie_fraction": 0.20,
            },
            {
                "method": CANONICAL_ARM,
                "direction_accuracy": 0.91,
                "minimum_event_coverage": 1.0,
                "mean_zero_fraction": 0.0,
                "mean_tie_fraction": 0.20,
            },
            {
                "method": STRICT_ARM,
                "direction_accuracy": 0.80,
                "minimum_event_coverage": 1.0,
                "mean_zero_fraction": 0.40,
                "mean_tie_fraction": 0.50,
            },
        ]
    )


def _paired() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {"metric": "omnibus_auprc", "paired_seeds": 3, "ci_low": -0.005},
            {"metric": "omnibus_auroc", "paired_seeds": 3, "ci_low": 0.10},
            {
                "metric": "direction_accuracy_all_active",
                "paired_seeds": 3,
                "ci_low": -0.01,
            },
        ]
    )


def test_m0_gate_requires_noninferior_auprc_and_superior_auroc() -> None:
    accepted = _gate(_summary(), _paired(), config=_config(), role="validation")
    assert accepted["status"] == "ACCEPT"
    assert all(accepted["checks"].values())

    failed = _paired()
    failed.loc[failed["metric"].eq("omnibus_auroc"), "ci_low"] = 0.0
    rejected = _gate(_summary(), failed, config=_config(), role="validation")
    assert rejected["status"] == "REJECT"
    assert rejected["checks"]["auroc_superiority"] is False


def test_paired_comparison_bootstraps_over_seed() -> None:
    rows = []
    for seed in (1, 2, 3):
        rows.extend(
            (
                {
                    "scenario": "active",
                    "seed": seed,
                    "method": M0_ARM,
                    "omnibus_auroc": 0.8 + seed / 100,
                },
                {
                    "scenario": "active",
                    "seed": seed,
                    "method": RC12_ARM,
                    "omnibus_auroc": 0.6 + seed / 100,
                },
            )
        )
    result = _paired_comparison(
        pd.DataFrame.from_records(rows),
        metric="omnibus_auroc",
        candidate=M0_ARM,
        reference=RC12_ARM,
        replicates=1000,
        seed=17,
    )
    assert result["paired_seeds"] == 3
    assert result["candidate_minus_reference"] == pytest.approx(0.2)
    assert result["ci_low"] > 0.0
    assert result["wins"] == 3


def test_layer_geometry_reports_structural_zero_and_ties() -> None:
    result = _layer_geometry(
        pd.Series([0.0, 0.0, 0.2, 0.7]),
        dataset_id="dataset",
        contrast="B_vs_A",
        method=STRICT_ARM,
    )
    assert result["zero_fraction"] == 0.5
    assert result["tie_fraction"] == 0.5
    assert result["unique_scores"] == 3
