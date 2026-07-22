from __future__ import annotations

import numpy as np

from benchmarks.literature.paired_rank_contrast import paired_rank_contrast


def test_zero_subtraction_returns_condition_percentiles() -> None:
    target, reference, fit = paired_rank_contrast(
        np.array([10.0, 20.0, 30.0]),
        np.array([3.0, 1.0, 2.0]),
        subtraction_fraction=0.0,
    )
    np.testing.assert_allclose(target, np.array([1 / 3, 2 / 3, 1.0]))
    np.testing.assert_allclose(reference, np.array([1.0, 1 / 3, 2 / 3]))
    assert fit.common_observed_pairs == 3


def test_equal_bidirectional_ranks_are_scaled_without_reordering() -> None:
    target, reference, _ = paired_rank_contrast(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([10.0, 20.0, 30.0, 40.0]),
        subtraction_fraction=0.75,
    )
    np.testing.assert_allclose(target, reference)
    np.testing.assert_array_equal(np.argsort(target), np.arange(4))


def test_structural_missingness_is_not_converted_to_zero() -> None:
    target, reference, fit = paired_rank_contrast(
        np.array([1.0, np.nan, 3.0]),
        np.array([3.0, 2.0, np.nan]),
        subtraction_fraction=0.5,
    )
    assert np.isnan(target[1])
    assert np.isnan(reference[2])
    assert np.isfinite(target[2])
    assert np.isfinite(reference[1])
    assert fit.common_observed_pairs == 1
