from __future__ import annotations

import numpy as np
import pytest

from crychic.attribution import winsorized_normalized_precision


def test_precision_is_winsorized_and_positive_median_normalized() -> None:
    result = winsorized_normalized_precision(
        np.asarray([1.0, 4.0, 100.0, np.nan, 0.0]),
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    np.testing.assert_allclose(result.values, [0.25, 1.0, 25.0, 0.0, 0.0])
    assert np.median(result.values[result.values > 0]) == pytest.approx(1.0)
    assert result.n_positive_features == 3
    assert result.estimable
    assert result.reason_code is None


def test_precision_transform_records_insufficient_support() -> None:
    result = winsorized_normalized_precision(
        np.asarray([np.nan, 2.0, 0.0]), min_positive_features=2
    )

    np.testing.assert_allclose(result.values, [0.0, 1.0, 0.0])
    assert not result.estimable
    assert result.reason_code == "insufficient_response_precision_support"


def test_precision_transform_id_changes_with_fitted_bounds() -> None:
    first = winsorized_normalized_precision(np.asarray([1.0, 2.0, 100.0]))
    second = winsorized_normalized_precision(np.asarray([1.0, 2.0, 10.0]))

    assert first.precision_transform_id != second.precision_transform_id
