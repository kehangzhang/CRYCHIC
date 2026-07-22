from __future__ import annotations

import numpy as np

from benchmarks.literature.bounded_program_blend import (
    bounded_program_blend_weights,
)


def test_blend_preserves_call_set_mean_and_bounds_each_expert() -> None:
    soft = np.array([2.0, 1.0, 0.5, 4.0])
    gate = np.array([1.0, 0.0, 1.0, 0.0])
    calls = np.array([True, True, True, False])
    blend, fit = bounded_program_blend_weights(
        soft, gate, calls, blend_fraction=0.25
    )
    assert np.isclose(blend[calls].mean(), 1.0)
    assert np.isclose(fit.blended_call_mean, 1.0)
    expected = 0.75 * soft / soft[calls].mean() + 0.25 * gate / gate[calls].mean()
    np.testing.assert_allclose(blend, expected)


def test_zero_fraction_is_normalized_soft_reference() -> None:
    soft = np.array([2.0, 1.0, 0.5])
    calls = np.array([True, True, False])
    blend, _ = bounded_program_blend_weights(
        soft, np.ones(3), calls, blend_fraction=0.0
    )
    np.testing.assert_allclose(blend, soft / soft[calls].mean())


def test_empty_call_set_returns_soft_weights_unchanged() -> None:
    soft = np.array([2.0, 1.0])
    blend, fit = bounded_program_blend_weights(
        soft, np.ones(2), np.zeros(2, dtype=bool), blend_fraction=0.5
    )
    np.testing.assert_array_equal(blend, soft)
    assert fit.call_count == 0
