from __future__ import annotations

import numpy as np
import pytest

from crychic.attribution import (
    AttributionSupportMethod,
    SolverStatus,
    attribute_target_prior,
    build_gated_target_basis,
    downstream_attribution_support,
    fit_positive_attribution,
)


def test_negative_response_remains_in_complete_signed_residual(
    prior_factory,
) -> None:
    prior = prior_factory({"L": {"G2": 1.0}})
    basis, result = attribute_target_prior(
        prior,
        ("G1", "G2"),
        {"L": 1.0},
        np.asarray([-2.0, 3.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    assert basis.matrix.shape == (2, 1)
    np.testing.assert_allclose(result.positive_response, [0.0, 3.0])
    np.testing.assert_allclose(result.predicted, [0.0, 3.0])
    np.testing.assert_allclose(result.residual, [-2.0, 0.0])
    np.testing.assert_allclose(result.predicted + result.residual, [-2.0, 3.0])
    assert result.succeeded
    assert result.response_channel == "positive"
    assert not hasattr(result, "p_value")
    assert not hasattr(result.driver_estimates[0], "p_value")


def test_collinear_ligands_keep_specific_coefficients_and_family_sum(
    prior_factory,
) -> None:
    prior = prior_factory(
        {
            "L1": {"G1": 1.0, "G2": 2.0},
            "L2": {"G1": 2.0, "G2": 4.0},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"L1": 1.0, "L2": 1.0},
    )
    result = fit_positive_attribution(
        basis,
        np.asarray([1.0, 2.0]),
        lambda1=0.05,
        lambda2=0.2,
        cosine_threshold=0.999,
        tolerance=1e-10,
        kkt_tolerance=1e-8,
    )

    assert len(result.driver_estimates) == 2
    assert len(result.family_estimates) == 1
    family = result.family_estimates[0]
    assert family.driver_ids == ("L1", "L2")
    assert family.assignment_uncertainty == pytest.approx(1.0, abs=1e-12)
    assert family.coefficient_sum == pytest.approx(result.coefficients.sum())
    assert sum(
        estimate.within_family_fraction or 0.0
        for estimate in result.driver_estimates
    ) == pytest.approx(1.0)


def test_zero_receptor_gate_forces_zero_specific_driver(prior_factory) -> None:
    prior = prior_factory({"OPEN": {"G1": 1.0}, "SHUT": {"G2": 1.0}})
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"OPEN": 1.0, "SHUT": 0.0},
    )
    result = fit_positive_attribution(
        basis,
        np.asarray([1.0, 10.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    coefficients = dict(zip(result.driver_ids, result.coefficients, strict=True))
    assert coefficients["OPEN"] == pytest.approx(1.0)
    assert coefficients["SHUT"] == 0.0
    assert result.residual[1] == pytest.approx(10.0)


def test_nonconverged_result_never_reports_success(prior_factory) -> None:
    prior = prior_factory({"L": {"G": 1.0}})
    basis = build_gated_target_basis(prior, ("G",), {"L": 1.0})
    result = fit_positive_attribution(
        basis,
        np.asarray([2.0]),
        tolerance=1e-15,
        max_iterations=1,
    )

    assert result.diagnostics.status is SolverStatus.MAX_ITERATIONS
    assert not result.succeeded


def test_downstream_support_v2_uses_gated_contribution_over_positive_norm(
    prior_factory,
) -> None:
    prior = prior_factory({"L1": {"G1": 1.0}, "L2": {"G2": 1.0}})
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"L1": 0.5, "L2": 1.0},
    )
    result = fit_positive_attribution(
        basis,
        np.asarray([2.0, 1.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    v1 = downstream_attribution_support(
        basis,
        result,
        method=AttributionSupportMethod.RELATIVE_COEFFICIENT_V1,
    )
    v2 = downstream_attribution_support(
        basis,
        result,
        method=AttributionSupportMethod.GATED_RESPONSE_NORM_V2,
    )

    np.testing.assert_allclose(v1.values, [1.0, 0.25], atol=1e-10)
    np.testing.assert_allclose(v2.numerator_values, [2.0, 1.0], atol=1e-10)
    assert v2.denominator_value == pytest.approx(np.sqrt(5.0))
    np.testing.assert_allclose(
        v2.values,
        np.asarray([2.0, 1.0]) / np.sqrt(5.0),
        atol=1e-10,
    )
    assert v2.gate_aware
    assert v2.to_dict()["application"] == "multiply_sample_target_activity"
