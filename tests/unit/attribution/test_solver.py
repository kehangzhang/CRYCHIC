from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from crychic.attribution import (
    SolverStatus,
    solve_nonnegative_elastic_net,
    solve_nonnegative_residual_elastic_net,
)
from crychic.core import ContractError


def test_weighted_one_coordinate_matches_analytic_solution_and_kkt() -> None:
    matrix = sparse.csc_matrix([[1.0]])
    solution = solve_nonnegative_elastic_net(
        matrix,
        np.asarray([3.0]),
        precision_weights=np.asarray([2.0]),
        lambda1=2.0,
        lambda2=1.0,
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    expected = (2.0 * 3.0 - 2.0 / 2.0) / (2.0 + 1.0)
    assert solution.coefficients[0] == pytest.approx(expected, abs=1e-12)
    assert solution.diagnostics.status is SolverStatus.CONVERGED
    assert solution.diagnostics.kkt_violation <= 1e-12
    objective = solution.diagnostics.objective
    expected_loss = 2.0 * (3.0 - expected) ** 2
    assert objective.weighted_loss == pytest.approx(expected_loss)
    assert objective.total == pytest.approx(
        expected_loss + 2.0 * expected + expected**2
    )


def test_orthogonal_hand_solution_respects_nonnegative_boundary() -> None:
    matrix = sparse.eye(2, format="csc")
    solution = solve_nonnegative_elastic_net(
        matrix,
        np.asarray([2.0, 0.05]),
        lambda1=0.2,
        lambda2=0.5,
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    np.testing.assert_allclose(solution.coefficients, [(2.0 - 0.1) / 1.5, 0.0])
    assert solution.diagnostics.kkt_violation <= 1e-12


def test_solver_is_bitwise_deterministic() -> None:
    matrix = sparse.csc_matrix([[1.0, 0.8], [0.2, 1.0], [0.5, 0.5]])
    kwargs = {
        "precision_weights": np.asarray([1.0, 2.0, 0.5]),
        "lambda1": 0.1,
        "lambda2": 0.2,
        "tolerance": 1e-10,
        "kkt_tolerance": 1e-9,
    }
    first = solve_nonnegative_elastic_net(
        matrix, np.asarray([1.0, 2.0, 0.5]), **kwargs
    )
    second = solve_nonnegative_elastic_net(
        matrix, np.asarray([1.0, 2.0, 0.5]), **kwargs
    )

    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.predicted, second.predicted)
    assert first.diagnostics.objective_history == second.diagnostics.objective_history


def test_zero_columns_match_the_exact_reduced_problem() -> None:
    active = sparse.csc_matrix(
        [[1.0, 0.8], [0.2, 1.0], [0.5, 0.5]],
    )
    active_coo = active.tocoo()
    expanded = sparse.csc_matrix(
        (
            active_coo.data,
            (
                active_coo.row,
                np.where(active_coo.col == 0, 17, 481),
            ),
        ),
        shape=(3, 502),
    )
    response = np.asarray([1.0, 2.0, 0.5])
    kwargs = {
        "precision_weights": np.asarray([1.0, 2.0, 0.5]),
        "lambda1": 0.1,
        "lambda2": 0.2,
        "tolerance": 1e-10,
        "kkt_tolerance": 1e-9,
    }

    reduced = solve_nonnegative_elastic_net(active, response, **kwargs)
    observed = solve_nonnegative_elastic_net(expanded, response, **kwargs)

    np.testing.assert_array_equal(
        observed.coefficients[[17, 481]], reduced.coefficients
    )
    assert np.count_nonzero(observed.coefficients) == np.count_nonzero(
        reduced.coefficients
    )
    np.testing.assert_array_equal(observed.predicted, reduced.predicted)
    assert observed.diagnostics.iterations == reduced.diagnostics.iterations
    assert (
        observed.diagnostics.objective_history
        == reduced.diagnostics.objective_history
    )


def test_iteration_limit_is_explicit_failure_not_success() -> None:
    solution = solve_nonnegative_elastic_net(
        sparse.csc_matrix([[1.0]]),
        np.asarray([2.0]),
        tolerance=1e-15,
        max_iterations=1,
    )

    assert solution.diagnostics.status is SolverStatus.MAX_ITERATIONS
    assert not solution.diagnostics.converged
    assert solution.diagnostics.failure_reason == (
        "coordinate_or_kkt_tolerance_not_met"
    )


def test_signed_residual_space_requires_explicit_opt_in() -> None:
    matrix = sparse.csc_matrix([[1.0], [-1.0]])
    response = np.asarray([2.0, -2.0])

    with pytest.raises(ContractError) as basis_error:
        solve_nonnegative_elastic_net(matrix, np.abs(response))
    assert basis_error.value.details.code == "invalid_solver_basis"

    with pytest.raises(ContractError) as response_error:
        solve_nonnegative_elastic_net(sparse.eye(2), response)
    assert response_error.value.details.code == "invalid_directional_response"

    solution = solve_nonnegative_residual_elastic_net(
        matrix,
        response,
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    assert solution.coefficients[0] == pytest.approx(2.0)
    np.testing.assert_allclose(solution.predicted, response, atol=1e-12)
    assert solution.diagnostics.status is SolverStatus.CONVERGED
