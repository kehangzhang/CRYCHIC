from __future__ import annotations

import numpy as np

from benchmarks.literature.crossfit_edge_program import (
    crossfit_program_candidates,
)


def _candidates() -> list[dict[str, object]]:
    return [
        {"name": "raw", "rank": 0, "projection_shrinkage": 0.0},
        {"name": "rank1", "rank": 1, "projection_shrinkage": 1.0},
    ]


def test_crossfit_program_denoises_low_rank_subject_edges() -> None:
    rng = np.random.default_rng(17)
    loadings = rng.normal(size=600)
    activity = np.linspace(-1.0, 1.0, 8)
    truth = 0.5 + 0.18 * np.outer(loadings, activity)
    observed = np.clip(truth + rng.normal(scale=0.08, size=truth.shape), 0.0, 1.0)
    outputs, fits, folds = crossfit_program_candidates(
        observed,
        _candidates(),
        minimum_edges=200,
        minimum_relative_improvement=0.0,
        full_reliability_improvement=0.05,
    )
    assert fits.set_index("candidate").loc["rank1", "reliability"] > 0.0
    assert np.nanmin(outputs["rank1"]) >= 0.0
    assert np.nanmax(outputs["rank1"]) <= 1.0
    raw_mse = np.mean((outputs["raw"] - truth) ** 2)
    rank_mse = np.mean((outputs["rank1"] - truth) ** 2)
    assert rank_mse < raw_mse
    assert len(folds) == 16


def test_crossfit_program_preserves_missingness_and_falls_back_when_too_small() -> None:
    matrix = np.full((20, 5), 0.5)
    matrix[0, 0] = np.nan
    outputs, fits, folds = crossfit_program_candidates(
        matrix,
        _candidates(),
        minimum_edges=200,
    )
    assert np.isnan(outputs["rank1"][0, 0])
    np.testing.assert_array_equal(outputs["rank1"], matrix)
    assert fits.set_index("candidate").loc["rank1", "reliability"] == 0.0
    assert set(folds["reason_code"]) == {"insufficient_supported_edges"}


def test_rank_zero_is_exact_identity() -> None:
    rng = np.random.default_rng(23)
    matrix = rng.uniform(size=(250, 6))
    outputs, fits, _ = crossfit_program_candidates(
        matrix,
        _candidates(),
        minimum_edges=200,
    )
    np.testing.assert_array_equal(outputs["raw"], matrix)
    assert fits.set_index("candidate").loc["raw", "reliability"] == 0.0
