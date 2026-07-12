from __future__ import annotations

import numpy as np
import pytest

from crychic.attribution import (
    AttributionSupportMethod,
    attribute_target_prior,
    downstream_attribution_support,
)


def test_explained_share_allocates_bounded_model_gain(prior_factory) -> None:
    prior = prior_factory({"L1": {"G1": 1.0}, "L2": {"G2": 1.0}})
    basis, result = attribute_target_prior(
        prior,
        ("G1", "G2"),
        {"L1": 1.0, "L2": 1.0},
        np.asarray([2.0, 1.0]),
        precision_weights=np.asarray([1.0, 1.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    support = downstream_attribution_support(
        basis,
        result,
        method=AttributionSupportMethod.EXPLAINED_SHARE_V3,
    )

    assert support.version == 3
    assert support.model_explained_gain == pytest.approx(1.0)
    np.testing.assert_allclose(support.values, [2.0 / 3.0, 1.0 / 3.0])
    assert support.values.sum() == pytest.approx(support.model_explained_gain)
    assert support.gate_aware


def test_tiny_noise_coefficient_is_not_full_support(prior_factory) -> None:
    prior = prior_factory({"L": {"G": 1.0}})
    basis, result = attribute_target_prior(
        prior,
        ("G",),
        {"L": 1.0},
        np.asarray([1e-12]),
        tolerance=1e-20,
        kkt_tolerance=1e-20,
    )

    relative = downstream_attribution_support(
        basis,
        result,
        method=AttributionSupportMethod.RELATIVE_COEFFICIENT_V1,
    )
    explained = downstream_attribution_support(
        basis,
        result,
        method=AttributionSupportMethod.EXPLAINED_SHARE_V3,
        response_norm_floor=1e-8,
    )

    assert relative.values[0] == pytest.approx(1.0)
    assert explained.values[0] == 0.0
    assert explained.model_explained_gain == 0.0
