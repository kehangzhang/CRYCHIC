"""Correctness-first graph-fused non-negative attribution solver."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from crychic.core import ContractError, stable_id
from crychic.design import ContextGraph

from .contracts import FamilyFirstBasis, SolverStatus
from .solver import solve_nonnegative_elastic_net

_BACKEND = "scipy_trust_constr_active_set_graph_fused_qp_v2"
_INDEPENDENT_BACKEND = "nonnegative_elastic_net_lambda_f_zero_v1"


@dataclass(frozen=True, slots=True)
class GraphFusedObjectiveTerms:
    """Final graph-fused objective decomposition."""

    weighted_loss: float
    l1_penalty: float
    l2_penalty: float
    fusion_penalty: float
    total: float

    def __post_init__(self) -> None:
        values = (
            self.weighted_loss,
            self.l1_penalty,
            self.l2_penalty,
            self.fusion_penalty,
            self.total,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ContractError(
                "Graph-fused objective terms must be finite and non-negative",
                code="invalid_graph_fused_objective",
                field="objective",
                remediation="Do not report a numerically failed solve as successful",
            )
        expected = (
            self.weighted_loss + self.l1_penalty + self.l2_penalty + self.fusion_penalty
        )
        if not math.isclose(self.total, expected, rel_tol=1e-9, abs_tol=1e-11):
            raise ContractError(
                "Graph-fused objective total does not equal its components",
                code="invalid_graph_fused_objective",
                field="total",
                remediation="Recompute the objective from final coefficients",
            )


@dataclass(frozen=True, slots=True)
class GraphFusedSolverDiagnostics:
    """Terminal diagnostics for the correctness-first convex prototype."""

    status: SolverStatus
    objective: GraphFusedObjectiveTerms
    backend: str
    iterations: int
    max_iterations: int
    tolerance: float
    optimality: float
    constraint_violation: float
    component_count: int
    objective_history: tuple[float, ...]
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SolverStatus(self.status))
        if not self.backend:
            raise ValueError("graph-fused backend must be non-empty")
        if self.iterations < 1 or self.iterations > self.max_iterations:
            raise ValueError("iterations must lie in [1, max_iterations]")
        if self.component_count < 1:
            raise ValueError("component_count must be positive")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.optimality, self.constraint_violation)
        ):
            raise ValueError("solver residuals must be finite and non-negative")
        if not self.objective_history or any(
            not math.isfinite(value) or value < 0 for value in self.objective_history
        ):
            raise ValueError("objective_history must be finite and non-empty")
        if self.status is SolverStatus.CONVERGED:
            if self.failure_reason is not None:
                raise ValueError(
                    "converged graph-fused solve cannot have a failure reason"
                )
            if (
                self.optimality > self.tolerance
                or self.constraint_violation > self.tolerance
            ):
                raise ValueError("converged graph-fused solve exceeds its tolerance")
        elif not self.failure_reason:
            raise ValueError("non-converged graph-fused solve requires a reason")

    @property
    def converged(self) -> bool:
        return self.status is SolverStatus.CONVERGED


@dataclass(frozen=True, slots=True)
class GraphFusedSolution:
    """Context-aligned coefficients and predictions from one joint solve."""

    context_nodes: tuple[Hashable, ...]
    coefficients: np.ndarray
    predicted: tuple[np.ndarray, ...]
    diagnostics: GraphFusedSolverDiagnostics

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64).copy()
        if coefficients.ndim != 2 or coefficients.shape[0] != len(self.context_nodes):
            raise ValueError("coefficients must have context x driver shape")
        if np.any(~np.isfinite(coefficients)) or np.any(coefficients < 0):
            raise ValueError("graph-fused coefficients must be finite and non-negative")
        predictions = tuple(
            np.asarray(values, dtype=np.float64).copy() for values in self.predicted
        )
        if len(predictions) != len(self.context_nodes) or any(
            values.ndim != 1 or np.any(~np.isfinite(values)) for values in predictions
        ):
            raise ValueError("predictions must contain one finite vector per context")
        coefficients.setflags(write=False)
        for values in predictions:
            values.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "predicted", predictions)


@dataclass(frozen=True, slots=True)
class GraphFusedFamilyAttributionResult:
    """Family-grain graph-fused attribution with complete signed residuals."""

    attribution_id: str
    context_nodes: tuple[Hashable, ...]
    family_basis_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    coefficients: np.ndarray
    contributions: np.ndarray
    signed_responses: tuple[np.ndarray, ...]
    positive_responses: tuple[np.ndarray, ...]
    predicted: tuple[np.ndarray, ...]
    residuals: tuple[np.ndarray, ...]
    precision_weights: tuple[np.ndarray, ...]
    diagnostics: GraphFusedSolverDiagnostics
    lambda1: float
    lambda2: float
    lambda_f: float
    experimental: bool = True

    def __post_init__(self) -> None:
        if not self.attribution_id or len(self.family_basis_ids) != len(
            self.context_nodes
        ):
            raise ValueError("graph-fused family attribution identity is incomplete")
        expected_coefficients = (len(self.context_nodes), len(self.family_ids))
        coefficients: np.ndarray = np.asarray(
            self.coefficients, dtype=np.float64
        ).copy()
        contributions: np.ndarray = np.asarray(
            self.contributions, dtype=np.float64
        ).copy()
        if coefficients.shape != expected_coefficients or contributions.shape != (
            expected_coefficients
        ):
            raise ValueError("family coefficients must have context x family shape")
        if (
            np.any(~np.isfinite(coefficients))
            or np.any(coefficients < 0)
            or np.any(~np.isfinite(contributions))
            or np.any(contributions < 0)
        ):
            raise ValueError(
                "family coefficients and contributions must be non-negative"
            )
        vector_groups = (
            self.signed_responses,
            self.positive_responses,
            self.predicted,
            self.residuals,
            self.precision_weights,
        )
        if any(len(group) != len(self.context_nodes) for group in vector_groups):
            raise ValueError("family attribution vectors must cover every context")
        frozen_groups: list[tuple[np.ndarray, ...]] = []
        for group in vector_groups:
            frozen: list[np.ndarray] = []
            for values in group:
                vector = np.asarray(values, dtype=np.float64).copy()
                if vector.shape != (len(self.feature_ids),) or np.any(
                    ~np.isfinite(vector)
                ):
                    raise ValueError(
                        "family attribution vectors must align with feature_ids"
                    )
                vector.setflags(write=False)
                frozen.append(vector)
            frozen_groups.append(tuple(frozen))
        for signed, predicted, residual in zip(
            frozen_groups[0], frozen_groups[2], frozen_groups[3], strict=True
        ):
            if not np.allclose(signed, predicted + residual, rtol=1e-9, atol=1e-11):
                raise ValueError(
                    "predicted plus residual must reconstruct signed response"
                )
        if not self.experimental:
            raise ValueError("graph-fused family attribution remains experimental")
        coefficients.setflags(write=False)
        contributions.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "contributions", contributions)
        for field_name, output_values in zip(
            (
                "signed_responses",
                "positive_responses",
                "predicted",
                "residuals",
                "precision_weights",
            ),
            frozen_groups,
            strict=True,
        ):
            object.__setattr__(self, field_name, output_values)

    @property
    def succeeded(self) -> bool:
        return self.diagnostics.converged


@dataclass(frozen=True, slots=True)
class _ValidatedInputs:
    nodes: tuple[Hashable, ...]
    designs: tuple[sparse.csc_matrix, ...]
    responses: tuple[np.ndarray, ...]
    precision: tuple[np.ndarray, ...]
    driver_count: int


@dataclass(frozen=True, slots=True)
class _ComponentResult:
    node_indices: tuple[int, ...]
    coefficients: np.ndarray
    predicted: tuple[np.ndarray, ...]
    iterations: int
    optimality: float
    constraint_violation: float
    objective_history: tuple[float, ...]
    status: SolverStatus
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class _PolishedComponent:
    coefficients: np.ndarray
    optimality: float
    constraint_violation: float


def _penalty(value: float, *, field_name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise ContractError(
            f"{field_name} must be finite and non-negative",
            code="invalid_graph_fused_penalty",
            field=field_name,
            remediation="Choose pre-registered non-negative penalties",
        )
    return resolved


def _validate_inputs(
    matrices: Mapping[Hashable, sparse.spmatrix],
    responses: Mapping[Hashable, np.ndarray],
    graph: ContextGraph,
    precision_weights: Mapping[Hashable, np.ndarray] | None,
) -> _ValidatedInputs:
    if not isinstance(graph, ContextGraph):
        raise TypeError("graph must be a ContextGraph")
    nodes = tuple(graph.nodes)
    if set(matrices) != set(nodes) or set(responses) != set(nodes):
        raise ContractError(
            "Graph-fused inputs must exactly cover the context graph nodes",
            code="graph_fused_context_mismatch",
            field="context_nodes",
            remediation="Provide one basis and response for every graph node",
        )
    if precision_weights is not None and set(precision_weights) != set(nodes):
        raise ContractError(
            "Precision weights must exactly cover the context graph nodes",
            code="graph_fused_context_mismatch",
            field="precision_weights",
            remediation="Provide one precision vector for every graph node",
        )
    designs: list[sparse.csc_matrix] = []
    response_values: list[np.ndarray] = []
    precision_values: list[np.ndarray] = []
    driver_count: int | None = None
    for node in nodes:
        design = sparse.csc_matrix(matrices[node], dtype=np.float64, copy=True)
        design.sum_duplicates()
        design.sort_indices()
        response = np.asarray(responses[node], dtype=np.float64)
        if response.ndim != 1 or design.shape[0] != response.size:
            raise ContractError(
                "Every graph-fused response must align with its basis rows",
                code="invalid_graph_fused_shape",
                field="response",
                remediation="Align each context response to its target basis",
            )
        if driver_count is None:
            driver_count = design.shape[1]
        if design.shape[1] != driver_count or driver_count == 0:
            raise ContractError(
                "Every context must share one non-empty driver axis",
                code="invalid_graph_fused_shape",
                field="matrix",
                remediation="Freeze one common driver family universe",
            )
        if (
            np.any(~np.isfinite(design.data))
            or np.any(design.data < 0)
            or np.any(~np.isfinite(response))
            or np.any(response < 0)
        ):
            raise ContractError(
                "Graph-fused inputs must be finite and direction-compatible",
                code="invalid_graph_fused_values",
                field="matrix,response",
                remediation="Use the positive direction-compatible response channel",
            )
        precision = (
            np.ones(response.size, dtype=np.float64)
            if precision_weights is None
            else np.asarray(precision_weights[node], dtype=np.float64)
        )
        if (
            precision.shape != response.shape
            or np.any(~np.isfinite(precision))
            or np.any(precision < 0)
            or not np.any(precision > 0)
        ):
            raise ContractError(
                "Graph-fused precision must align, be non-negative, and not all zero",
                code="invalid_graph_fused_precision",
                field="precision_weights",
                remediation="Use one supported precision vector per context",
            )
        designs.append(design)
        response_values.append(response.copy())
        precision_values.append(precision.copy())
    return _ValidatedInputs(
        nodes=nodes,
        designs=tuple(designs),
        responses=tuple(response_values),
        precision=tuple(precision_values),
        driver_count=cast(int, driver_count),
    )


def _objective_terms(
    inputs: _ValidatedInputs,
    coefficients: np.ndarray,
    graph: ContextGraph,
    *,
    lambda1: float,
    lambda2: float,
    lambda_f: float,
) -> GraphFusedObjectiveTerms:
    predicted = tuple(
        np.asarray(design.dot(coefficients[index])).ravel()
        for index, design in enumerate(inputs.designs)
    )
    weighted_loss = float(
        sum(
            np.dot(weight, (response - fitted) ** 2)
            for weight, response, fitted in zip(
                inputs.precision, inputs.responses, predicted, strict=True
            )
        )
    )
    l1_penalty = float(lambda1 * coefficients.sum())
    l2_penalty = float(lambda2 * np.square(coefficients).sum())
    node_index = {node: index for index, node in enumerate(inputs.nodes)}
    fusion_penalty = float(
        lambda_f
        * sum(
            edge.weight
            * np.abs(
                coefficients[node_index[edge.left]]
                - coefficients[node_index[edge.right]]
            ).sum()
            for edge in graph.edges
        )
    )
    return GraphFusedObjectiveTerms(
        weighted_loss=weighted_loss,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
        fusion_penalty=fusion_penalty,
        total=weighted_loss + l1_penalty + l2_penalty + fusion_penalty,
    )


def _independent_solution(
    inputs: _ValidatedInputs,
    graph: ContextGraph,
    *,
    lambda1: float,
    lambda2: float,
    tolerance: float,
    max_iterations: int,
) -> GraphFusedSolution:
    children = tuple(
        solve_nonnegative_elastic_net(
            design,
            response,
            precision_weights=precision,
            lambda1=lambda1,
            lambda2=lambda2,
            tolerance=tolerance,
            kkt_tolerance=tolerance,
            max_iterations=max_iterations,
        )
        for design, response, precision in zip(
            inputs.designs, inputs.responses, inputs.precision, strict=True
        )
    )
    coefficients = np.vstack([child.coefficients for child in children])
    objective = _objective_terms(
        inputs,
        coefficients,
        graph,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda_f=0.0,
    )
    converged = all(child.diagnostics.converged for child in children)
    return GraphFusedSolution(
        context_nodes=inputs.nodes,
        coefficients=coefficients,
        predicted=tuple(child.predicted for child in children),
        diagnostics=GraphFusedSolverDiagnostics(
            status=(
                SolverStatus.CONVERGED if converged else SolverStatus.MAX_ITERATIONS
            ),
            objective=objective,
            backend=_INDEPENDENT_BACKEND,
            iterations=max(child.diagnostics.iterations for child in children),
            max_iterations=max_iterations,
            tolerance=tolerance,
            optimality=max(child.diagnostics.kkt_violation for child in children),
            constraint_violation=0.0,
            component_count=len(graph.components),
            objective_history=(objective.total,),
            failure_reason=(
                None if converged else "independent_context_solver_not_converged"
            ),
        ),
    )


def _component_edges(
    graph: ContextGraph,
    component: tuple[Hashable, ...],
) -> tuple[tuple[int, int, float], ...]:
    local = {node: index for index, node in enumerate(component)}
    return tuple(
        (local[edge.left], local[edge.right], edge.weight)
        for edge in graph.edges
        if edge.left in local and edge.right in local
    )


def _component_objective(
    inputs: _ValidatedInputs,
    node_indices: tuple[int, ...],
    edges: tuple[tuple[int, int, float], ...],
    coefficients: np.ndarray,
    *,
    lambda1: float,
    lambda2: float,
    lambda_f: float,
) -> float:
    total = 0.0
    for local_index, index in enumerate(node_indices):
        fitted = np.asarray(
            inputs.designs[index].dot(coefficients[local_index])
        ).ravel()
        residual = inputs.responses[index] - fitted
        total += float(np.dot(inputs.precision[index], residual * residual))
    total += lambda1 * float(coefficients.sum())
    total += lambda2 * float(np.square(coefficients).sum())
    total += lambda_f * sum(
        weight * float(np.abs(coefficients[left] - coefficients[right]).sum())
        for left, right, weight in edges
    )
    return total


def _graph_kkt_violation(
    inputs: _ValidatedInputs,
    node_indices: tuple[int, ...],
    edges: tuple[tuple[int, int, float], ...],
    coefficients: np.ndarray,
    fixed_zero: np.ndarray,
    *,
    lambda1: float,
    lambda2: float,
    lambda_f: float,
    active_tolerance: float,
) -> float:
    """Return the exact nonsmooth KKT residual after active-set polishing."""

    context_count, driver_count = coefficients.shape
    smooth = np.empty_like(coefficients)
    for local_index, index in enumerate(node_indices):
        fitted = np.asarray(
            inputs.designs[index].dot(coefficients[local_index])
        ).ravel()
        gradient = np.asarray(
            2.0
            * inputs.designs[index].T.dot(
                inputs.precision[index] * (fitted - inputs.responses[index])
            )
        ).ravel()
        smooth[local_index] = (
            gradient + lambda1 + 2.0 * lambda2 * coefficients[local_index]
        )

    worst = 0.0
    for driver in range(driver_count):
        base = smooth[:, driver].copy()
        tied: list[tuple[int, int, float]] = []
        for left, right, weight in edges:
            difference = coefficients[left, driver] - coefficients[right, driver]
            penalty = lambda_f * weight
            if difference > active_tolerance:
                base[left] += penalty
                base[right] -= penalty
            elif difference < -active_tolerance:
                base[left] -= penalty
                base[right] += penalty
            else:
                tied.append((left, right, penalty))

        checked = ~fixed_zero[:, driver]
        active = (coefficients[:, driver] > active_tolerance) & checked
        inactive = checked & ~active
        if not tied:
            violation = np.zeros(context_count, dtype=np.float64)
            violation[active] = np.abs(base[active])
            violation[inactive] = np.maximum(0.0, -base[inactive])
            worst = max(worst, float(violation.max(initial=0.0)))
            continue

        incidence = np.zeros((context_count, len(tied)), dtype=np.float64)
        bounds: list[tuple[float | None, float | None]] = []
        for edge_index, (left, right, penalty) in enumerate(tied):
            incidence[left, edge_index] = 1.0
            incidence[right, edge_index] = -1.0
            bounds.append((-penalty, penalty))

        inequalities: list[np.ndarray] = []
        limits: list[float] = []
        for index in np.flatnonzero(active):
            inequalities.append(np.append(incidence[index], -1.0))
            limits.append(float(-base[index]))
            inequalities.append(np.append(-incidence[index], -1.0))
            limits.append(float(base[index]))
        for index in np.flatnonzero(inactive):
            inequalities.append(np.append(-incidence[index], -1.0))
            limits.append(float(base[index]))
        if not inequalities:
            continue
        result = linprog(
            np.append(np.zeros(len(tied), dtype=np.float64), 1.0),
            A_ub=np.vstack(inequalities),
            b_ub=np.asarray(limits, dtype=np.float64),
            bounds=(*bounds, (0.0, None)),
            method="highs",
        )
        if not result.success or np.any(~np.isfinite(result.x)):
            return math.inf
        worst = max(worst, float(result.x[-1]))
    return worst


def _polish_component_active_set(
    inputs: _ValidatedInputs,
    node_indices: tuple[int, ...],
    edges: tuple[tuple[int, int, float], ...],
    hessian: np.ndarray,
    constraint_matrix: np.ndarray,
    bounds: Bounds,
    constraints: LinearConstraint,
    initial: np.ndarray,
    *,
    lambda1: float,
    lambda2: float,
    lambda_f: float,
    tolerance: float,
    max_iterations: int,
) -> _PolishedComponent | None:
    """Polish the barrier solution and certify its original nonsmooth KKT system."""

    context_count = len(node_indices)
    driver_count = inputs.driver_count
    beta_size = context_count * driver_count
    active_tolerance: float = max(float(tolerance), float(np.finfo(np.float64).eps))
    edge_weights: np.ndarray = np.repeat(
        np.asarray([edge[2] for edge in edges], dtype=np.float64),
        driver_count,
    )

    def objective(values: np.ndarray) -> float:
        beta = values[:beta_size].reshape(context_count, driver_count)
        total = _component_objective(
            inputs,
            node_indices,
            (),
            beta,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda_f=0.0,
        )
        return total + lambda_f * float(np.dot(edge_weights, values[beta_size:]))

    def gradient(values: np.ndarray) -> np.ndarray:
        beta = values[:beta_size].reshape(context_count, driver_count)
        result = np.empty_like(values)
        for local_index, index in enumerate(node_indices):
            fitted = np.asarray(inputs.designs[index].dot(beta[local_index])).ravel()
            local = np.asarray(
                2.0
                * inputs.designs[index].T.dot(
                    inputs.precision[index] * (fitted - inputs.responses[index])
                )
            ).ravel()
            local += lambda1 + 2.0 * lambda2 * beta[local_index]
            start = local_index * driver_count
            result[start : start + driver_count] = local
        result[beta_size:] = lambda_f * edge_weights
        return cast(np.ndarray, result)

    boundary = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=(constraints,),
        options={
            "disp": False,
            "ftol": max(
                np.finfo(np.float64).eps,
                min(1.0e-12, tolerance * tolerance),
            ),
            "maxiter": max_iterations,
        },
    )
    boundary_values = np.asarray(boundary.x, dtype=np.float64)
    if (
        boundary_values.shape != initial.shape
        or np.any(~np.isfinite(boundary_values))
        or np.min(constraint_matrix @ boundary_values, initial=0.0) < -active_tolerance
    ):
        return None
    boundary_beta = np.maximum(
        boundary_values[:beta_size].reshape(context_count, driver_count), 0.0
    )

    parent = list(range(beta_size))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left, right, _ in edges:
        for driver in range(driver_count):
            if (
                abs(boundary_beta[left, driver] - boundary_beta[right, driver])
                <= active_tolerance
            ):
                union(
                    left * driver_count + driver,
                    right * driver_count + driver,
                )

    groups: dict[int, list[int]] = {}
    for index in range(beta_size):
        groups.setdefault(find(index), []).append(index)
    fixed_zero = np.isfinite(bounds.ub[:beta_size]).reshape(context_count, driver_count)
    zero_roots = {
        find(index)
        for index, value in enumerate(boundary_beta.ravel())
        if value <= active_tolerance or fixed_zero.ravel()[index]
    }

    beta_hessian = hessian[:beta_size, :beta_size]
    linear: np.ndarray = np.empty(beta_size, dtype=np.float64)
    for local_index, index in enumerate(node_indices):
        start = local_index * driver_count
        linear[start : start + driver_count] = (
            np.asarray(
                -2.0
                * inputs.designs[index].T.dot(
                    inputs.precision[index] * inputs.responses[index]
                )
            ).ravel()
            + lambda1
        )
    for left, right, weight in edges:
        for driver in range(driver_count):
            difference = boundary_beta[left, driver] - boundary_beta[right, driver]
            if abs(difference) <= active_tolerance:
                continue
            signed_penalty = math.copysign(lambda_f * weight, difference)
            linear[left * driver_count + driver] += signed_penalty
            linear[right * driver_count + driver] -= signed_penalty

    polished_beta: np.ndarray
    while True:
        free_roots = tuple(root for root in sorted(groups) if root not in zero_roots)
        if not free_roots:
            polished_beta = np.zeros(beta_size, dtype=np.float64)
            break
        projection: np.ndarray = np.zeros(
            (beta_size, len(free_roots)), dtype=np.float64
        )
        for column, root in enumerate(free_roots):
            projection[groups[root], column] = 1.0
        reduced_hessian = projection.T @ beta_hessian @ projection
        reduced_linear = projection.T @ linear
        reduced = np.linalg.lstsq(
            reduced_hessian,
            -reduced_linear,
            rcond=None,
        )[0]
        newly_zero = {
            root
            for root, value in zip(free_roots, reduced, strict=True)
            if value <= active_tolerance
        }
        if newly_zero:
            zero_roots.update(newly_zero)
            continue
        polished_beta = projection @ reduced
        break

    coefficients = polished_beta.reshape(context_count, driver_count)
    if np.any(~np.isfinite(coefficients)) or np.any(coefficients < 0.0):
        return None
    for left, right, _ in edges:
        for driver in range(driver_count):
            before = boundary_beta[left, driver] - boundary_beta[right, driver]
            after = coefficients[left, driver] - coefficients[right, driver]
            if abs(before) > active_tolerance and before * after <= 0.0:
                return None
    coefficients[coefficients <= active_tolerance] = 0.0
    fixed_violation = float(np.abs(coefficients[fixed_zero]).max(initial=0.0))
    negative_violation = max(0.0, float(-coefficients.min(initial=0.0)))
    constraint_violation = max(fixed_violation, negative_violation)
    optimality = _graph_kkt_violation(
        inputs,
        node_indices,
        edges,
        coefficients,
        fixed_zero,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda_f=lambda_f,
        active_tolerance=active_tolerance,
    )
    initial_beta = initial[:beta_size].reshape(context_count, driver_count)
    initial_objective = _component_objective(
        inputs,
        node_indices,
        edges,
        initial_beta,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda_f=lambda_f,
    )
    polished_objective = _component_objective(
        inputs,
        node_indices,
        edges,
        coefficients,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda_f=lambda_f,
    )
    allowed_increase = tolerance * max(1.0, abs(initial_objective))
    if (
        not math.isfinite(optimality)
        or optimality > tolerance
        or constraint_violation > tolerance
        or polished_objective > initial_objective + allowed_increase
    ):
        return None
    return _PolishedComponent(
        coefficients=coefficients,
        optimality=optimality,
        constraint_violation=constraint_violation,
    )


def _solve_component(
    inputs: _ValidatedInputs,
    graph: ContextGraph,
    component: tuple[Hashable, ...],
    *,
    lambda1: float,
    lambda2: float,
    lambda_f: float,
    tolerance: float,
    max_iterations: int,
) -> _ComponentResult:
    global_index = {node: index for index, node in enumerate(inputs.nodes)}
    node_indices = tuple(global_index[node] for node in component)
    if len(component) == 1:
        index = node_indices[0]
        child = solve_nonnegative_elastic_net(
            inputs.designs[index],
            inputs.responses[index],
            precision_weights=inputs.precision[index],
            lambda1=lambda1,
            lambda2=lambda2,
            tolerance=tolerance,
            kkt_tolerance=tolerance,
            max_iterations=max_iterations,
        )
        return _ComponentResult(
            node_indices=node_indices,
            coefficients=child.coefficients[np.newaxis, :],
            predicted=(child.predicted,),
            iterations=child.diagnostics.iterations,
            optimality=child.diagnostics.kkt_violation,
            constraint_violation=0.0,
            objective_history=child.diagnostics.objective_history,
            status=child.diagnostics.status,
            failure_reason=child.diagnostics.failure_reason,
        )

    context_count = len(component)
    driver_count = inputs.driver_count
    edges = _component_edges(graph, component)
    beta_size = context_count * driver_count
    difference_size = len(edges) * driver_count
    variable_count = beta_size + difference_size
    initial_beta = np.vstack(
        [
            solve_nonnegative_elastic_net(
                inputs.designs[index],
                inputs.responses[index],
                precision_weights=inputs.precision[index],
                lambda1=lambda1,
                lambda2=lambda2,
                tolerance=max(tolerance, 1e-10),
                kkt_tolerance=max(tolerance, 1e-10),
                max_iterations=max_iterations,
            ).coefficients
            for index in node_indices
        ]
    )
    initial_difference = np.concatenate(
        [np.abs(initial_beta[left] - initial_beta[right]) for left, right, _ in edges]
    )
    initial = np.concatenate((initial_beta.ravel(), initial_difference))

    hessian: np.ndarray = np.zeros((variable_count, variable_count), dtype=np.float64)
    for local_index, index in enumerate(node_indices):
        design = inputs.designs[index]
        weighted = design.multiply(inputs.precision[index][:, np.newaxis])
        block = 2.0 * np.asarray((design.T @ weighted).toarray())
        block += 2.0 * lambda2 * np.eye(driver_count)
        start = local_index * driver_count
        hessian[start : start + driver_count, start : start + driver_count] = block

    edge_weights: np.ndarray = np.repeat(
        np.asarray([edge[2] for edge in edges]), driver_count
    )

    def objective(values: np.ndarray) -> float:
        beta = values[:beta_size].reshape(context_count, driver_count)
        total = 0.0
        for local_index, index in enumerate(node_indices):
            fitted = np.asarray(inputs.designs[index].dot(beta[local_index])).ravel()
            residual = inputs.responses[index] - fitted
            total += float(np.dot(inputs.precision[index], residual * residual))
        total += lambda1 * float(beta.sum())
        total += lambda2 * float(np.square(beta).sum())
        total += lambda_f * float(np.dot(edge_weights, values[beta_size:]))
        return total

    def gradient(values: np.ndarray) -> np.ndarray:
        beta = values[:beta_size].reshape(context_count, driver_count)
        result: np.ndarray = np.empty(variable_count, dtype=np.float64)
        for local_index, index in enumerate(node_indices):
            fitted = np.asarray(inputs.designs[index].dot(beta[local_index])).ravel()
            grad = np.asarray(
                2.0
                * inputs.designs[index].T.dot(
                    inputs.precision[index] * (fitted - inputs.responses[index])
                )
            ).ravel()
            grad += lambda1 + 2.0 * lambda2 * beta[local_index]
            start = local_index * driver_count
            result[start : start + driver_count] = grad
        result[beta_size:] = lambda_f * edge_weights
        return result

    constraint_matrix: np.ndarray = np.zeros(
        (2 * difference_size, variable_count), dtype=np.float64
    )
    for edge_index, (left, right, _) in enumerate(edges):
        for driver in range(driver_count):
            difference_index = edge_index * driver_count + driver
            positive_row = 2 * difference_index
            negative_row = positive_row + 1
            left_index = left * driver_count + driver
            right_index = right * driver_count + driver
            auxiliary_index = beta_size + difference_index
            constraint_matrix[positive_row, auxiliary_index] = 1.0
            constraint_matrix[positive_row, left_index] = -1.0
            constraint_matrix[positive_row, right_index] = 1.0
            constraint_matrix[negative_row, auxiliary_index] = 1.0
            constraint_matrix[negative_row, left_index] = 1.0
            constraint_matrix[negative_row, right_index] = -1.0
    constraints = LinearConstraint(constraint_matrix, 0.0, np.inf)
    upper_bounds: np.ndarray = np.full(variable_count, np.inf, dtype=np.float64)
    for local_index, index in enumerate(node_indices):
        design = inputs.designs[index]
        weighted_norm = np.asarray(
            design.power(2).T.dot(inputs.precision[index])
        ).ravel()
        start = local_index * driver_count
        upper_bounds[start : start + driver_count] = np.where(
            weighted_norm > 0.0, np.inf, 0.0
        )
    history = [objective(initial)]

    def callback(values: np.ndarray, state: Any) -> bool:
        history.append(objective(values))
        return False

    optimized = minimize(
        objective,
        initial,
        method="trust-constr",
        jac=gradient,
        hess=lambda _: hessian,
        bounds=Bounds(np.zeros(variable_count), upper_bounds),
        constraints=(constraints,),
        callback=callback,
        options={
            "barrier_tol": tolerance,
            "gtol": tolerance,
            "maxiter": max_iterations,
            "verbose": 0,
            "xtol": tolerance,
        },
    )
    optimized_values = np.asarray(optimized.x, dtype=np.float64)
    polished = (
        _polish_component_active_set(
            inputs,
            node_indices,
            edges,
            hessian,
            constraint_matrix,
            Bounds(np.zeros(variable_count), upper_bounds),
            constraints,
            optimized_values,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda_f=lambda_f,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        if optimized.success
        else None
    )
    coefficients = (
        polished.coefficients.copy()
        if polished is not None
        else np.maximum(
            optimized_values[:beta_size].reshape(context_count, driver_count), 0.0
        )
    )
    coefficients[upper_bounds[:beta_size].reshape(context_count, driver_count) == 0] = (
        0.0
    )
    predictions = tuple(
        np.asarray(inputs.designs[index].dot(coefficients[local_index])).ravel()
        for local_index, index in enumerate(node_indices)
    )
    optimality = (
        polished.optimality
        if polished is not None
        else float(cast(Any, optimized).optimality)
    )
    constraint_violation = (
        polished.constraint_violation
        if polished is not None
        else float(cast(Any, optimized).constr_violation)
    )
    converged = bool(
        optimized.success
        and polished is not None
        and optimality <= tolerance
        and constraint_violation <= tolerance
    )
    if converged:
        status = SolverStatus.CONVERGED
        reason = None
    elif optimized.success and polished is None:
        status = SolverStatus.NUMERICAL_FAILURE
        reason = "graph_fused_active_set_polish_not_certified"
    elif int(optimized.nit) >= max_iterations:
        status = SolverStatus.MAX_ITERATIONS
        reason = "graph_fused_optimality_or_constraint_tolerance_not_met"
    else:
        status = SolverStatus.NUMERICAL_FAILURE
        reason = f"graph_fused_backend_failure:{optimized.message}"
    return _ComponentResult(
        node_indices=node_indices,
        coefficients=coefficients,
        predicted=predictions,
        iterations=max(1, int(optimized.nit)),
        optimality=optimality,
        constraint_violation=constraint_violation,
        objective_history=tuple(float(value) for value in history),
        status=status,
        failure_reason=reason,
    )


def solve_graph_fused_nonnegative_elastic_net(
    matrices: Mapping[Hashable, sparse.spmatrix],
    responses: Mapping[Hashable, np.ndarray],
    graph: ContextGraph,
    *,
    precision_weights: Mapping[Hashable, np.ndarray] | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    lambda_f: float = 0.0,
    tolerance: float = 1e-8,
    max_iterations: int = 1_000,
) -> GraphFusedSolution:
    """Jointly fit non-negative context coefficients with graph total variation.

    This is the correctness-first G2 prototype. It uses a constrained convex QP
    backend for positive ``lambda_f`` and exactly delegates to the existing
    coordinate-descent solver when graph fusion is disabled.
    """

    penalties = {
        "lambda1": _penalty(lambda1, field_name="lambda1"),
        "lambda2": _penalty(lambda2, field_name="lambda2"),
        "lambda_f": _penalty(lambda_f, field_name="lambda_f"),
    }
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be an integer >= 1")
    inputs = _validate_inputs(matrices, responses, graph, precision_weights)
    if penalties["lambda_f"] == 0.0 or not graph.edges:
        return _independent_solution(
            inputs,
            graph,
            lambda1=penalties["lambda1"],
            lambda2=penalties["lambda2"],
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    components = tuple(
        _solve_component(
            inputs,
            graph,
            component,
            lambda1=penalties["lambda1"],
            lambda2=penalties["lambda2"],
            lambda_f=penalties["lambda_f"],
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        for component in graph.components
    )
    coefficients: np.ndarray = np.zeros(
        (len(inputs.nodes), inputs.driver_count), dtype=np.float64
    )
    predicted: list[np.ndarray | None] = [None] * len(inputs.nodes)
    for component in components:
        for local_index, global_index in enumerate(component.node_indices):
            coefficients[global_index] = component.coefficients[local_index]
            predicted[global_index] = component.predicted[local_index]
    objective = _objective_terms(
        inputs,
        coefficients,
        graph,
        lambda1=penalties["lambda1"],
        lambda2=penalties["lambda2"],
        lambda_f=penalties["lambda_f"],
    )
    converged = all(
        component.status is SolverStatus.CONVERGED for component in components
    )
    statuses = {component.status for component in components}
    if converged:
        status = SolverStatus.CONVERGED
        failure_reason = None
    elif SolverStatus.NUMERICAL_FAILURE in statuses:
        status = SolverStatus.NUMERICAL_FAILURE
        failure_reason = "one_or_more_graph_components_failed"
    else:
        status = SolverStatus.MAX_ITERATIONS
        failure_reason = "one_or_more_graph_components_did_not_converge"
    return GraphFusedSolution(
        context_nodes=inputs.nodes,
        coefficients=coefficients,
        predicted=tuple(cast(np.ndarray, values) for values in predicted),
        diagnostics=GraphFusedSolverDiagnostics(
            status=status,
            objective=objective,
            backend=_BACKEND,
            iterations=max(component.iterations for component in components),
            max_iterations=max_iterations,
            tolerance=tolerance,
            optimality=max(component.optimality for component in components),
            constraint_violation=max(
                component.constraint_violation for component in components
            ),
            component_count=len(components),
            objective_history=(
                sum(component.objective_history[0] for component in components),
                objective.total,
            ),
            failure_reason=failure_reason,
        ),
    )


def fit_graph_fused_family_attribution(
    bases: Mapping[Hashable, FamilyFirstBasis],
    signed_responses: Mapping[Hashable, np.ndarray],
    graph: ContextGraph,
    *,
    precision_weights: Mapping[Hashable, np.ndarray] | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    lambda_f: float = 0.0,
    tolerance: float = 1e-8,
    max_iterations: int = 1_000,
) -> GraphFusedFamilyAttributionResult:
    """Fit strict-family coefficients jointly across a frozen context graph."""

    nodes = tuple(graph.nodes)
    if set(bases) != set(nodes) or set(signed_responses) != set(nodes):
        raise ContractError(
            "Family attribution inputs must exactly cover the context graph",
            code="graph_fused_context_mismatch",
            field="context_nodes",
            remediation="Provide one family basis and response for every context",
        )
    if not all(isinstance(bases[node], FamilyFirstBasis) for node in nodes):
        raise TypeError("bases must contain FamilyFirstBasis values")
    reference = bases[nodes[0]]
    if any(
        basis.feature_ids != reference.feature_ids
        or basis.family_ids != reference.family_ids
        or basis.family_definitions != reference.family_definitions
        for basis in (bases[node] for node in nodes[1:])
    ):
        raise ContractError(
            "Graph-fused family attribution requires one common feature/family axis",
            code="graph_fused_family_universe_mismatch",
            field="family_basis",
            remediation="Freeze one family universe before context-specific gating",
        )
    signed = {
        node: np.asarray(signed_responses[node], dtype=np.float64) for node in nodes
    }
    if any(
        values.shape != (len(reference.feature_ids),) or np.any(~np.isfinite(values))
        for values in signed.values()
    ):
        raise ContractError(
            "Signed responses must be finite and align with the common feature axis",
            code="invalid_family_response",
            field="signed_responses",
            remediation="Provide one complete signed response per context",
        )
    positive = {node: np.maximum(signed[node], 0.0) for node in nodes}
    resolved_precision = (
        {node: np.ones(len(reference.feature_ids), dtype=np.float64) for node in nodes}
        if precision_weights is None
        else {
            node: np.asarray(precision_weights[node], dtype=np.float64)
            for node in nodes
        }
    )
    solution = solve_graph_fused_nonnegative_elastic_net(
        {node: bases[node].matrix for node in nodes},
        positive,
        graph,
        precision_weights=resolved_precision,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda_f=lambda_f,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    column_norms = np.vstack(
        [
            np.sqrt(np.asarray(bases[node].matrix.power(2).sum(axis=0)).ravel())
            for node in nodes
        ]
    )
    contributions = solution.coefficients * column_norms
    residuals = tuple(
        signed[node] - fitted
        for node, fitted in zip(nodes, solution.predicted, strict=True)
    )
    attribution_id = stable_id(
        "graph_fused_family_attribution",
        {
            "context_graph": graph.to_dict(),
            "context_nodes": list(nodes),
            "family_basis_ids": [bases[node].family_basis_id for node in nodes],
            "family_ids": list(reference.family_ids),
            "coefficients": solution.coefficients.tolist(),
            "precision_weights": [resolved_precision[node].tolist() for node in nodes],
            "signed_responses": [signed[node].tolist() for node in nodes],
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "lambda_f": float(lambda_f),
            "tolerance": float(tolerance),
            "max_iterations": max_iterations,
            "backend": solution.diagnostics.backend,
        },
        schema_version="1",
    )
    return GraphFusedFamilyAttributionResult(
        attribution_id=attribution_id,
        context_nodes=nodes,
        family_basis_ids=tuple(bases[node].family_basis_id for node in nodes),
        feature_ids=reference.feature_ids,
        family_ids=reference.family_ids,
        coefficients=solution.coefficients,
        contributions=contributions,
        signed_responses=tuple(signed[node] for node in nodes),
        positive_responses=tuple(positive[node] for node in nodes),
        predicted=solution.predicted,
        residuals=residuals,
        precision_weights=tuple(resolved_precision[node] for node in nodes),
        diagnostics=solution.diagnostics,
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        lambda_f=float(lambda_f),
    )


__all__ = [
    "GraphFusedFamilyAttributionResult",
    "GraphFusedObjectiveTerms",
    "GraphFusedSolution",
    "GraphFusedSolverDiagnostics",
    "fit_graph_fused_family_attribution",
    "solve_graph_fused_nonnegative_elastic_net",
]
