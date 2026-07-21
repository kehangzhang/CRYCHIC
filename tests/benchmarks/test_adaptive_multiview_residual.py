from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.literature.adaptive_multiview_residual import (
    estimate_oof_view_weights,
    fit_adaptive_cross_validated_residual,
    normalize_view_weights,
    solve_adaptive_multiview_residual,
)
from benchmarks.literature.liana_hypergraph_residual import (
    multiview_codes,
    solve_multiview_residual,
)

VIEWS = ("sender", "receiver", "ligand", "receptor", "interaction_id")


def _edges(n: int = 600) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_key": [f"edge_{index}" for index in range(n)],
            "sender": [f"s{index % 4}" for index in range(n)],
            "receiver": [f"r{(index // 4) % 5}" for index in range(n)],
            "ligand": [f"l{(index // 20) % 7}" for index in range(n)],
            "receptor": [f"q{(index // 140) % 3}" for index in range(n)],
            "interaction_id": [f"i{index % 30}" for index in range(n)],
        }
    )


def test_equal_profile_is_exactly_the_rc2_solver() -> None:
    edges = _edges()
    rng = np.random.default_rng(9)
    baseline = rng.normal(size=len(edges))
    anchor = rng.normal(size=len(edges))
    codes = multiview_codes(edges, VIEWS)
    original = solve_multiview_residual(
        baseline,
        anchor,
        codes,
        lambda_hypergraph=0.25,
        anchor_weight=0.1,
    )
    adaptive = solve_adaptive_multiview_residual(
        baseline,
        anchor,
        codes,
        view_weights=np.ones(len(VIEWS)),
        lambda_hypergraph=0.25,
        anchor_weight=0.1,
    )
    np.testing.assert_array_equal(adaptive[0], original[0])
    assert adaptive[1:] == original[1:]


def test_sender_profile_is_selected_for_sender_only_signal() -> None:
    edges = _edges()
    rng = np.random.default_rng(21)
    truth = edges["sender"].map({"s0": -2.0, "s1": -1.0, "s2": 1.0, "s3": 2.0})
    baseline = truth.to_numpy() + rng.normal(scale=1.3, size=len(edges))
    theta, fit, diagnostics = fit_adaptive_cross_validated_residual(
        edges,
        baseline,
        np.full(len(edges), np.nan),
        profiles={"all_equal": [1, 1, 1, 1, 1], "sender_only": [1, 0, 0, 0, 0]},
        views=VIEWS,
        key_columns=("edge_key",),
        lambda_grid=(0.05, 0.25, 1.0),
        anchor_grid=(0.0,),
        minimum_profile_relative_improvement=0.005,
    )
    assert fit.profile_name == "sender_only"
    assert diagnostics["selected"].sum() == 1
    assert np.mean(np.square(theta - truth)) < np.mean(np.square(baseline - truth))


def test_unpredictable_views_fall_back_to_baseline_exactly() -> None:
    edges = _edges()
    rng = np.random.default_rng(44)
    baseline = rng.normal(size=len(edges))
    theta, fit, _ = fit_adaptive_cross_validated_residual(
        edges,
        baseline,
        rng.normal(size=len(edges)),
        profiles={"all_equal": [1, 1, 1, 1, 1], "sender_only": [1, 0, 0, 0, 0]},
        views=VIEWS,
        key_columns=("edge_key",),
        minimum_cv_improvement=0.05,
        full_gate_improvement=0.1,
    )
    assert fit.fallback_gate == 0.0
    np.testing.assert_array_equal(theta, baseline)


def test_profiles_keep_fixed_total_penalty_mass() -> None:
    weights = normalize_view_weights([1, 1, 0, 0, 0], view_count=5)
    assert weights == (2.5, 2.5, 0.0, 0.0, 0.0)
    assert sum(weights) == 5.0


def test_oof_reliability_downweights_unpredictable_views() -> None:
    edges = _edges()
    rng = np.random.default_rng(72)
    baseline = edges["sender"].map(
        {"s0": -2.0, "s1": -1.0, "s2": 1.0, "s3": 2.0}
    ).to_numpy() + rng.normal(scale=0.8, size=len(edges))
    weights, diagnostics = estimate_oof_view_weights(
        edges,
        baseline,
        views=VIEWS,
        key_columns=("edge_key",),
        folds=3,
        seed=17,
    )
    by_view = diagnostics.set_index("view")
    assert (
        by_view.loc["sender", "normalized_view_weight"]
        > by_view.loc["receptor", "normalized_view_weight"]
    )
    assert np.isclose(sum(weights), len(VIEWS))
