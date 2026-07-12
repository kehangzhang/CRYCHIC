from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crychic.attribution import (
    AttributionSupportMethod,
    ReceptorGatePolicy,
    build_gated_target_basis,
    downstream_attribution_support,
    fit_positive_attribution,
)
from crychic.core import ContractError


@pytest.mark.parametrize(
    "support_method",
    (
        AttributionSupportMethod.RELATIVE_COEFFICIENT_V1,
        AttributionSupportMethod.GATED_RESPONSE_NORM_V2,
    ),
)
def test_lower_eligible_gate_cannot_inflate_fit_or_support(
    prior_factory,
    support_method: AttributionSupportMethod,
) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G2": 1.0}})
    high = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"A": 1.0, "B": 1.0},
        gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
        receptor_gate_threshold=0.1,
    )
    lower_but_eligible = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"A": 0.2, "B": 1.0},
        gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
        receptor_gate_threshold=0.1,
    )

    np.testing.assert_array_equal(high.receptor_eligible, [True, True])
    np.testing.assert_array_equal(lower_but_eligible.receptor_eligible, [True, True])
    np.testing.assert_array_equal(
        lower_but_eligible.matrix.toarray(), high.matrix.toarray()
    )
    high_fit = fit_positive_attribution(
        high,
        np.asarray([1.0, 2.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    lower_fit = fit_positive_attribution(
        lower_but_eligible,
        np.asarray([1.0, 2.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    np.testing.assert_allclose(
        lower_fit.coefficients, high_fit.coefficients, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        lower_fit.predicted, high_fit.predicted, rtol=0, atol=1e-12
    )

    high_support = downstream_attribution_support(high, high_fit, method=support_method)
    lower_support = downstream_attribution_support(
        lower_but_eligible,
        lower_fit,
        method=support_method,
    )
    np.testing.assert_allclose(
        lower_support.values, high_support.values, rtol=0, atol=1e-12
    )
    assert np.all(lower_support.values <= high_support.values + 1e-12)


@pytest.mark.parametrize("excluded_gate", (0.0, 0.09))
def test_hard_policy_strictly_zeros_ineligible_columns_and_support(
    prior_factory,
    excluded_gate: float,
) -> None:
    prior = prior_factory(
        {
            "A": {"G1": 1.0},
            "B": {"G2": 1.0},
            "C": {"G3": 1.0},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2", "G3"),
        {"A": excluded_gate, "B": 0.1, "C": 1.0},
        gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
        receptor_gate_threshold=0.1,
    )

    np.testing.assert_array_equal(basis.receptor_eligible, [False, True, True])
    assert basis.matrix.getcol(0).nnz == 0
    np.testing.assert_array_equal(basis.matrix.getcol(0).toarray(), 0.0)
    np.testing.assert_array_equal(basis.matrix.getcol(1).toarray(), [[0], [1], [0]])
    result = fit_positive_attribution(
        basis,
        np.asarray([10.0, 2.0, 1.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    assert result.coefficients[0] == 0.0
    support = downstream_attribution_support(basis, result)
    assert support.numerator_values[0] == 0.0
    assert support.values[0] == 0.0
    assert result.predicted[0] == 0.0
    assert result.residual[0] == pytest.approx(10.0)


def test_gate_policy_provenance_and_mapping_order_are_deterministic(
    prior_factory,
) -> None:
    first_prior = prior_factory(
        {
            "C": {"G3": 1.0},
            "A": {"G1": 1.0},
            "B": {"G2": 1.0},
        }
    )
    reordered_prior = prior_factory(
        {
            "B": {"G2": 1.0},
            "C": {"G3": 1.0},
            "A": {"G1": 1.0},
        }
    )
    first = build_gated_target_basis(
        first_prior,
        ("G1", "G2", "G3"),
        {"C": 0.9, "A": 0.4, "B": 0.1},
        gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
        receptor_gate_threshold=0.25,
    )
    reordered = build_gated_target_basis(
        reordered_prior,
        ("G1", "G2", "G3"),
        {"B": 0.1, "A": 0.4, "C": 0.9},
        gate_policy="hard_eligibility_v2",
        receptor_gate_threshold=0.25,
    )

    assert first.driver_ids == ("A", "B", "C")
    assert first.basis_id == reordered.basis_id
    np.testing.assert_array_equal(first.receptor_gates, reordered.receptor_gates)
    np.testing.assert_array_equal(first.receptor_eligible, [True, False, True])
    np.testing.assert_array_equal(first.matrix.toarray(), reordered.matrix.toarray())
    assert (
        first.gate_provenance()
        == reordered.gate_provenance()
        == {
            "policy": "hard_eligibility_v2",
            "version": 2,
            "matrix_semantics": "unit_normalized_profile_if_eligible_else_zero",
            "threshold_operator": "receptor_gate >= threshold",
            "threshold": 0.25,
            "driver_ids": ["A", "B", "C"],
            "eligible_mask": [True, False, True],
            "eligible_driver_ids": ["A", "C"],
        }
    )
    assert not first.receptor_eligible.flags.writeable


def test_basis_contract_rejects_policy_matrix_or_mask_mismatch(prior_factory) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G2": 1.0}})
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"A": 0.5, "B": 1.0},
        gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
        receptor_gate_threshold=0.1,
    )

    continuously_scaled = basis.normalized_profiles.multiply([0.5, 1.0])
    with pytest.raises(ContractError, match="does not implement"):
        replace(basis, matrix=continuously_scaled)
    with pytest.raises(ContractError, match="does not match"):
        replace(basis, receptor_eligible=np.asarray([False, True]))


@pytest.mark.parametrize("threshold", (None, True, 0.0, -0.1, 1.1, np.nan))
def test_hard_policy_requires_explicit_valid_threshold(
    prior_factory,
    threshold: float | bool | None,
) -> None:
    prior = prior_factory({"A": {"G1": 1.0}})
    with pytest.raises(ContractError, match="threshold"):
        build_gated_target_basis(
            prior,
            ("G1",),
            {"A": 1.0},
            gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
            receptor_gate_threshold=threshold,  # type: ignore[arg-type]
        )


def test_legacy_continuous_policy_remains_the_default(prior_factory) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G2": 1.0}})
    implicit = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"A": 0.5, "B": 0.0},
    )
    explicit = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"A": 0.5, "B": 0.0},
        gate_policy=ReceptorGatePolicy.LEGACY_CONTINUOUS_V1,
    )

    assert implicit.basis_id == explicit.basis_id
    assert implicit.gate_policy is ReceptorGatePolicy.LEGACY_CONTINUOUS_V1
    assert implicit.receptor_gate_threshold is None
    np.testing.assert_array_equal(implicit.receptor_eligible, [True, False])
    np.testing.assert_array_equal(implicit.matrix.toarray(), [[0.5, 0.0], [0.0, 0.0]])
    assert implicit.gate_provenance()["version"] == 1
    assert implicit.gate_provenance()["threshold"] is None
