from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.literature.liana_hypergraph_residual import (
    HypergraphResidualFit,
    conservative_sign_statistic,
    fit_cross_validated_residual,
    multiview_codes,
    solve_multiview_residual,
    stable_edge_folds,
)


def _edges(n: int = 240) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sender": [f"s{index % 4}" for index in range(n)],
            "receiver": [f"r{(index // 4) % 4}" for index in range(n)],
            "ligand": [f"l{(index // 16) % 5}" for index in range(n)],
            "receptor": [f"q{(index // 80) % 3}" for index in range(n)],
            "interaction_id": [f"i{index}" for index in range(n)],
        }
    )


def test_solver_preserves_signed_values_and_converges() -> None:
    edges = _edges()
    baseline = np.linspace(-2.0, 2.0, len(edges))
    anchor = baseline * 0.8
    theta, converged, iterations = solve_multiview_residual(
        baseline,
        anchor,
        multiview_codes(edges),
        lambda_hypergraph=0.1,
        anchor_weight=0.1,
    )
    assert converged
    assert iterations < 200
    assert theta.min() < 0.0 < theta.max()
    assert np.var(theta) < np.var(baseline)


def test_edge_folds_are_deterministic_and_nonempty() -> None:
    edges = _edges()
    first = stable_edge_folds(
        edges,
        key_columns=("sender", "receiver", "interaction_id"),
        folds=3,
        seed=19,
    )
    second = stable_edge_folds(
        edges,
        key_columns=("sender", "receiver", "interaction_id"),
        folds=3,
        seed=19,
    )
    np.testing.assert_array_equal(first, second)
    assert set(first) == {0, 1, 2}


def test_cv_gate_falls_back_exactly_when_structure_is_unpredictable() -> None:
    rng = np.random.default_rng(33)
    edges = _edges(600)
    baseline = rng.normal(size=len(edges))
    anchor = rng.normal(size=len(edges))
    theta, fit, diagnostics = fit_cross_validated_residual(
        edges,
        baseline,
        anchor,
        minimum_cv_improvement=0.05,
        full_gate_improvement=0.1,
    )
    assert not diagnostics.empty
    assert fit.fallback_gate == 0.0
    np.testing.assert_array_equal(theta, baseline)


def test_cv_gate_opens_for_group_predictable_signal() -> None:
    rng = np.random.default_rng(91)
    edges = _edges(600)
    sender_effect = {f"s{index}": value for index, value in enumerate([-2, -1, 1, 2])}
    truth = edges["sender"].map(sender_effect).to_numpy(dtype=float)
    baseline = truth + rng.normal(scale=0.7, size=len(edges))
    anchor = truth + rng.normal(scale=0.5, size=len(edges))
    theta, fit, _ = fit_cross_validated_residual(
        edges,
        baseline,
        anchor,
        minimum_cv_improvement=0.01,
        full_gate_improvement=0.05,
    )
    assert fit.fallback_gate > 0.0
    assert np.mean(np.square(theta - truth)) < np.mean(np.square(baseline - truth))


def test_conservative_sign_correction_preserves_confident_and_fallback_edges() -> None:
    fit = HypergraphResidualFit(
        lambda_hypergraph=0.1,
        anchor_weight=0.0,
        cv_mse=0.5,
        cv_baseline_mse=1.0,
        cv_relative_improvement=0.5,
        fallback_gate=1.0,
        folds=3,
        views=("sender",),
        converged=True,
        iterations=2,
    )
    baseline = np.array([-2.0, -0.2, 0.2, 2.0])
    residual = -baseline
    result = conservative_sign_statistic(
        baseline, residual, fit, maximum_abs_baseline=1.0
    )
    np.testing.assert_array_equal(result, np.array([-2.0, 0.2, -0.2, 2.0]))
    fallback_fit = HypergraphResidualFit(
        **{**fit.to_dict(), "fallback_gate": 0.0}
    )
    np.testing.assert_array_equal(
        conservative_sign_statistic(
            baseline, residual, fallback_fit, maximum_abs_baseline=1.0
        ),
        baseline,
    )
