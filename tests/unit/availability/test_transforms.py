from __future__ import annotations

import pytest

from crychic.availability import (
    DetectionShrinkage,
    generalized_harmonic_softmin,
    hill_transform,
    shrink_detection_fraction,
)


def test_hill_transform_is_bounded_and_monotone() -> None:
    values = [hill_transform(value) for value in (0.0, 0.1, 1.0, 10.0)]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] < 1.0
    assert hill_transform(1.0) == pytest.approx(0.5, abs=1e-12)


def test_detection_shrinkage_stays_inside_boundaries() -> None:
    parameters = DetectionShrinkage(alpha=0.5, beta=0.5)
    low = shrink_detection_fraction(0.0, 10, parameters)
    high = shrink_detection_fraction(1.0, 10, parameters)
    assert 0.0 < low < high < 1.0
    assert shrink_detection_fraction(0.0, 1000, parameters) < low
    assert shrink_detection_fraction(1.0, 1000, parameters) > high


def test_softmin_is_limited_by_weak_complex_subunit_and_monotone() -> None:
    balanced = generalized_harmonic_softmin((0.8, 0.8), power=8)
    limited = generalized_harmonic_softmin((0.8, 0.05), power=8)
    raised_limit = generalized_harmonic_softmin((0.8, 0.2), power=8)
    assert balanced == pytest.approx(0.8, rel=1e-10)
    assert limited < 0.06
    assert limited < raised_limit < balanced
    assert generalized_harmonic_softmin((0.8, 0.0), power=8) == 0.0
