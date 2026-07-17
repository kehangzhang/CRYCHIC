from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from crychic.attribution import (
    SolverStatus,
    build_family_first_basis,
    build_gated_target_basis,
    cluster_driver_families,
    fit_family_first_attribution,
    fit_graph_fused_family_attribution,
    solve_graph_fused_nonnegative_elastic_net,
    solve_nonnegative_elastic_net,
)
from crychic.core import ContractError
from crychic.design import ContextGraph


def _one_driver_inputs(values: dict[str, float]):
    matrices = {node: sparse.csc_matrix([[1.0]]) for node in values}
    responses = {node: np.asarray([value]) for node, value in values.items()}
    return matrices, responses


def test_lambda_f_zero_exactly_matches_existing_context_solvers() -> None:
    graph = ContextGraph.chain(("a", "b"))
    matrices = {
        "a": sparse.csc_matrix([[1.0, 0.2], [0.0, 1.0]]),
        "b": sparse.csc_matrix([[0.8, 0.1], [0.1, 1.0]]),
    }
    responses = {
        "a": np.asarray([1.0, 0.5]),
        "b": np.asarray([0.2, 1.5]),
    }
    observed = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        graph,
        lambda1=0.1,
        lambda2=0.2,
        lambda_f=0.0,
        tolerance=1e-10,
    )
    expected = tuple(
        solve_nonnegative_elastic_net(
            matrices[node],
            responses[node],
            lambda1=0.1,
            lambda2=0.2,
            tolerance=1e-10,
            kkt_tolerance=1e-10,
        )
        for node in observed.context_nodes
    )

    np.testing.assert_array_equal(
        observed.coefficients,
        np.vstack([solution.coefficients for solution in expected]),
    )
    assert observed.diagnostics.backend == ("nonnegative_elastic_net_lambda_f_zero_v1")
    assert observed.diagnostics.objective.fusion_penalty == 0.0


def test_two_context_one_driver_matches_piecewise_analytic_solution() -> None:
    graph = ContextGraph.chain(("control", "treated"))
    matrices, responses = _one_driver_inputs({"control": 1.0, "treated": 3.0})

    solution = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        graph,
        lambda_f=1.0,
        tolerance=1e-8,
    )

    assert solution.diagnostics.status is SolverStatus.CONVERGED
    np.testing.assert_allclose(solution.coefficients[:, 0], [1.5, 2.5], atol=2e-5)
    assert solution.diagnostics.objective.fusion_penalty == pytest.approx(1.0, abs=5e-5)


def test_large_fusion_penalty_converges_to_shared_component_coefficient() -> None:
    graph = ContextGraph.chain(("a", "b"))
    matrices, responses = _one_driver_inputs({"a": 1.0, "b": 3.0})

    solution = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        graph,
        lambda_f=10.0,
        tolerance=1e-8,
    )

    assert solution.diagnostics.converged
    np.testing.assert_allclose(solution.coefficients[:, 0], [2.0, 2.0], atol=2e-4)


def test_positive_fusion_preserves_exact_zero_optimum() -> None:
    graph = ContextGraph.chain(("a", "b"))
    matrices, responses = _one_driver_inputs({"a": 0.0, "b": 0.0})

    solution = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        graph,
        lambda1=0.1,
        lambda_f=1.0,
        tolerance=1e-8,
    )

    assert solution.diagnostics.converged
    np.testing.assert_array_equal(solution.coefficients, np.zeros((2, 1)))
    assert solution.diagnostics.objective.total == 0.0
    assert all(np.count_nonzero(values) == 0 for values in solution.predicted)


def test_positive_fusion_preserves_exact_fused_plateau() -> None:
    graph = ContextGraph.chain(("a", "b"))
    matrices, responses = _one_driver_inputs({"a": 1.0, "b": 1.0})

    solution = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        graph,
        lambda_f=1.0,
        tolerance=1e-8,
    )

    assert solution.diagnostics.converged
    np.testing.assert_array_equal(solution.coefficients[:, 0], [1.0, 1.0])
    assert solution.diagnostics.objective.fusion_penalty == 0.0


def test_positive_fusion_keeps_kkt_supported_small_positive_optimum() -> None:
    graph = ContextGraph.chain(("a", "b"))
    optimum = 5.0e-8
    matrices, responses = _one_driver_inputs({"a": 0.05 + optimum, "b": 0.05 + optimum})

    solution = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        graph,
        lambda1=0.1,
        lambda_f=1.0,
        tolerance=1e-8,
    )

    assert solution.diagnostics.converged
    np.testing.assert_allclose(
        solution.coefficients[:, 0], [optimum, optimum], rtol=1e-7, atol=1e-14
    )


def test_edge_weight_is_equivalent_to_scaling_lambda_f() -> None:
    unit_graph = ContextGraph.chain(("a", "b"), weights=(1.0,))
    weighted_graph = ContextGraph.chain(("a", "b"), weights=(2.0,))
    matrices, responses = _one_driver_inputs({"a": 1.0, "b": 3.0})

    unit = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        unit_graph,
        lambda_f=1.0,
        tolerance=1e-8,
    )
    weighted = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        weighted_graph,
        lambda_f=0.5,
        tolerance=1e-8,
    )

    np.testing.assert_allclose(unit.coefficients, weighted.coefficients, atol=2e-6)


def test_mapping_insertion_order_does_not_change_joint_solution() -> None:
    graph = ContextGraph.chain(("a", "b"))
    matrices, responses = _one_driver_inputs({"a": 1.0, "b": 3.0})
    reversed_matrices = dict(reversed(tuple(matrices.items())))
    reversed_responses = dict(reversed(tuple(responses.items())))

    first = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        graph,
        lambda_f=1.0,
        tolerance=1e-8,
    )
    second = solve_graph_fused_nonnegative_elastic_net(
        reversed_matrices,
        reversed_responses,
        graph,
        lambda_f=1.0,
        tolerance=1e-8,
    )

    assert first.context_nodes == second.context_nodes
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert first.diagnostics.objective == second.diagnostics.objective


def test_disconnected_context_is_identical_to_independent_fit() -> None:
    graph = ContextGraph.from_edges((("a", "b"),), nodes=("a", "b", "c"))
    matrices, responses = _one_driver_inputs({"a": 1.0, "b": 3.0, "c": 7.0})

    solution = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        graph,
        lambda_f=1.0,
        tolerance=1e-8,
    )
    c_index = solution.context_nodes.index("c")
    independent = solve_nonnegative_elastic_net(
        matrices["c"], responses["c"], tolerance=1e-8, kkt_tolerance=1e-8
    )

    assert solution.coefficients[c_index, 0] == independent.coefficients[0]
    assert solution.diagnostics.component_count == 2


def test_context_mapping_must_exactly_match_graph_nodes() -> None:
    graph = ContextGraph.chain(("a", "b"))
    matrices, responses = _one_driver_inputs({"a": 1.0})

    with pytest.raises(ContractError) as error:
        solve_graph_fused_nonnegative_elastic_net(matrices, responses, graph)

    assert error.value.details.code == "graph_fused_context_mismatch"


def _family_basis(prior, *, a_gate: float = 1.0, b_gate: float = 1.0):
    source = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"A": a_gate, "B": b_gate},
    )
    families = cluster_driver_families(source, cosine_threshold=0.999)
    return build_family_first_basis(
        source,
        families,
        strict_cosine_threshold=0.999,
    )


def test_family_graph_lambda_f_zero_matches_family_first_and_preserves_residual(
    prior_factory,
) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G2": 1.0}})
    basis = _family_basis(prior)
    graph = ContextGraph.chain(("control", "treated"))
    responses = {
        "control": np.asarray([1.0, -0.5]),
        "treated": np.asarray([3.0, 2.0]),
    }

    observed = fit_graph_fused_family_attribution(
        {"control": basis, "treated": basis},
        responses,
        graph,
        lambda_f=0.0,
        tolerance=1e-10,
    )
    expected = tuple(
        fit_family_first_attribution(
            basis,
            responses[node],
            tolerance=1e-10,
            kkt_tolerance=1e-10,
        )
        for node in observed.context_nodes
    )

    np.testing.assert_array_equal(
        observed.coefficients,
        np.vstack([result.coefficients for result in expected]),
    )
    for signed, predicted, residual in zip(
        observed.signed_responses,
        observed.predicted,
        observed.residuals,
        strict=True,
    ):
        np.testing.assert_allclose(predicted + residual, signed)
    assert observed.succeeded


def test_family_graph_positive_fusion_preserves_structural_zero(prior_factory) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G2": 1.0}})
    basis = _family_basis(prior)
    graph = ContextGraph.chain(("control", "treated"))

    result = fit_graph_fused_family_attribution(
        {"control": basis, "treated": basis},
        {
            "control": np.zeros(len(basis.feature_ids)),
            "treated": np.zeros(len(basis.feature_ids)),
        },
        graph,
        lambda1=0.1,
        lambda_f=1.0,
        tolerance=1e-8,
    )

    assert result.succeeded
    np.testing.assert_array_equal(result.coefficients, np.zeros((2, 2)))
    np.testing.assert_array_equal(result.contributions, np.zeros((2, 2)))
    assert all(np.count_nonzero(values) == 0 for values in result.predicted)


def test_family_graph_allows_context_specific_frozen_eligibility(prior_factory) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G2": 1.0}})
    full = _family_basis(prior)
    gated = _family_basis(prior, a_gate=0.0)
    graph = ContextGraph.chain(("a", "b"))

    result = fit_graph_fused_family_attribution(
        {"a": gated, "b": full},
        {"a": np.asarray([3.0, 1.0]), "b": np.asarray([3.0, 1.0])},
        graph,
        lambda_f=0.5,
        tolerance=1e-8,
    )

    assert result.family_ids == full.family_ids == gated.family_ids
    assert result.diagnostics.converged
    a_index = result.context_nodes.index("a")
    a_family = gated.medoid_driver_ids.index("A")
    assert result.coefficients[a_index, a_family] == pytest.approx(0.0, abs=1e-7)
