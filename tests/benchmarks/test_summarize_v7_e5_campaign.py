from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.simulation.summarize_v7_e5_campaign import (
    LOCKED_FAMILY,
    _descriptive_ci,
    _locked_ranks,
    _paired_comparisons,
)


def test_descriptive_ci_requires_independent_dataset_replication() -> None:
    low, high = _descriptive_ci(pd.Series([0.1, 0.2, 0.3, np.nan]))

    assert low < 0.2 < high
    assert all(np.isnan(value) for value in _descriptive_ci(pd.Series([0.2])))


def _dataset_values() -> pd.DataFrame:
    values = {
        "no_prior": {"effect_mse": 0.50, "exact_hyperedge_ap": 0.40},
        "full_hypergraph": {"effect_mse": 0.30, "exact_hyperedge_ap": 0.55},
        "degree_matched_permuted_hypergraph": {
            "effect_mse": 0.45,
            "exact_hyperedge_ap": 0.42,
        },
        "rewired_25pct": {"effect_mse": 0.38, "exact_hyperedge_ap": 0.48},
        "tensor_factorization": {
            "effect_mse": 0.32,
            "exact_hyperedge_ap": 0.60,
        },
        "pairwise_clique_expansion": {
            "effect_mse": 0.48,
            "exact_hyperedge_ap": 0.50,
        },
    }
    records = []
    for dataset_index, offset in enumerate((0.00, 0.01, -0.01), start=1):
        for arm, metrics in values.items():
            for metric, value in metrics.items():
                records.append(
                    {
                        "dataset_id": f"dataset-{dataset_index}",
                        "dgp_family": LOCKED_FAMILY,
                        "design_kind": "independent_multi_group",
                        "score_view": arm,
                        "metric": metric,
                        "value": value + offset,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_paired_gain_orientation_is_positive_when_candidate_is_better() -> None:
    paired = _paired_comparisons(_dataset_values())
    full_raw = paired.loc[
        paired["candidate"].eq("full_hypergraph") & paired["baseline"].eq("no_prior")
    ].set_index("metric")

    assert np.isclose(full_raw.loc["effect_mse", "mean_gain"], 0.20)
    assert full_raw.loc["effect_mse", "gain_orientation"] == (
        "baseline_minus_candidate"
    )
    assert np.isclose(full_raw.loc["exact_hyperedge_ap", "mean_gain"], 0.15)
    assert full_raw.loc["exact_hyperedge_ap", "gain_orientation"] == (
        "candidate_minus_baseline"
    )
    assert full_raw["paired_datasets"].eq(3).all()


def test_locked_ranks_use_coverage_distance_instead_of_larger_is_better() -> None:
    summary = pd.DataFrame.from_records(
        [
            {
                "scope": "locked_hypergraph",
                "score_view": "near_nominal",
                "metric": "ci_coverage",
                "mean": 0.94,
            },
            {
                "scope": "locked_hypergraph",
                "score_view": "too_high",
                "metric": "ci_coverage",
                "mean": 1.00,
            },
            {
                "scope": "locked_hypergraph",
                "score_view": "low_mse",
                "metric": "effect_mse",
                "mean": 0.10,
            },
            {
                "scope": "locked_hypergraph",
                "score_view": "high_mse",
                "metric": "effect_mse",
                "mean": 0.20,
            },
        ]
    )

    ranks = _locked_ranks(summary).set_index(["metric", "score_view"])

    assert ranks.loc[("ci_coverage", "near_nominal"), "rank"] == 1
    assert ranks.loc[("effect_mse", "low_mse"), "rank"] == 1
