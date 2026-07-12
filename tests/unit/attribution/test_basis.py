from __future__ import annotations

import numpy as np
import pytest

from crychic.attribution import (
    build_gated_target_basis,
    cluster_driver_families,
)
from crychic.core import ContractError


def test_prior_columns_are_l2_normalized_before_receptor_gate(prior_factory) -> None:
    prior = prior_factory(
        {
            "L1": {"G1": 3.0, "G2": 4.0},
            "L2": {"G1": 6.0, "G2": 8.0},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"L1": 1.0, "L2": 0.5},
    )

    np.testing.assert_allclose(basis.pre_normalization_norms, [5.0, 10.0])
    np.testing.assert_allclose(
        basis.normalized_profiles.toarray(),
        [[0.6, 0.6], [0.8, 0.8]],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        basis.matrix.toarray(),
        [[0.6, 0.3], [0.8, 0.4]],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.sqrt(np.asarray(basis.matrix.power(2).sum(axis=0)).ravel()),
        [1.0, 0.5],
    )


def test_missing_target_overlap_is_reported_and_zero_column_preserved(
    prior_factory,
) -> None:
    prior = prior_factory({"L1": {"G1": 1.0}, "L2": {"OTHER": 2.0}})
    basis = build_gated_target_basis(
        prior, ("G1",), {"L1": 1.0, "L2": 1.0}
    )

    assert basis.report.unmatched_prior_targets == ("OTHER",)
    assert basis.report.zero_norm_drivers == ("L2",)
    np.testing.assert_allclose(basis.matrix.toarray(), [[1.0, 0.0]])


def test_cosine_family_is_stable_for_scaled_collinear_profiles(
    prior_factory,
) -> None:
    prior = prior_factory(
        {
            "A": {"G1": 1.0, "G2": 2.0},
            "B": {"G1": 3.0, "G2": 6.0},
            "C": {"G3": 1.0},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2", "G3"),
        {"A": 1.0, "B": 0.0, "C": 1.0},
    )
    first = cluster_driver_families(basis, cosine_threshold=0.999)
    second = cluster_driver_families(basis, cosine_threshold=0.999)

    assert first == second
    collinear = next(family for family in first if len(family.driver_ids) == 2)
    assert collinear.driver_ids == ("A", "B")
    assert collinear.mean_pairwise_cosine == pytest.approx(1.0, abs=1e-12)
    assert collinear.assignment_uncertainty == pytest.approx(1.0, abs=1e-12)


def test_receptor_gate_must_cover_exact_driver_universe(prior_factory) -> None:
    prior = prior_factory({"L1": {"G1": 1.0}, "L2": {"G1": 1.0}})
    with pytest.raises(ContractError, match="exactly match"):
        build_gated_target_basis(prior, ("G1",), {"L1": 1.0})
