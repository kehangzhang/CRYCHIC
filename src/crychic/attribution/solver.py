"""Deterministic non-negative elastic-net coordinate descent."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from crychic.core import ContractError

from .contracts import ObjectiveTerms, SolverDiagnostics, SolverStatus


@dataclass(frozen=True, slots=True)
class ElasticNetSolution:
    """Numerical solution consumed by the attribution producer."""

    coefficients: np.ndarray
    predicted: np.ndarray
    diagnostics: SolverDiagnostics


def _objective(
    response: np.ndarray,
    predicted: np.ndarray,
    precision: np.ndarray,
    coefficients: np.ndarray,
    lambda1: float,
    lambda2: float,
) -> ObjectiveTerms:
    residual = response - predicted
    weighted_loss = float(np.dot(precision, residual * residual))
    l1_penalty = float(lambda1 * coefficients.sum())
    l2_penalty = float(lambda2 * np.dot(coefficients, coefficients))
    return ObjectiveTerms(
        weighted_loss=weighted_loss,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
        total=weighted_loss + l1_penalty + l2_penalty,
    )


def _kkt_violation(
    matrix: sparse.csc_matrix,
    response: np.ndarray,
    precision: np.ndarray,
    coefficients: np.ndarray,
    predicted: np.ndarray,
    lambda1: float,
    lambda2: float,
    *,
    active_tolerance: float,
) -> float:
    gradient = np.asarray(
        2.0 * matrix.T.dot(precision * (predicted - response))
    ).ravel()
    gradient += 2.0 * lambda2 * coefficients + lambda1
    violation = np.where(
        coefficients > active_tolerance,
        np.abs(gradient),
        np.maximum(0.0, -gradient),
    )
    return 0.0 if violation.size == 0 else float(violation.max())


def _try_sklearn_coordinate_descent(
    design: sparse.csc_matrix,
    response: np.ndarray,
    precision: np.ndarray,
    *,
    signed_residual_space: bool,
    lambda1: float,
    lambda2: float,
    tolerance: float,
    kkt_tolerance: float,
    max_iterations: int,
) -> ElasticNetSolution | None:
    """Use sklearn's compiled cyclic coordinate descent when explicitly enabled.

    The weighted objective is mapped exactly to sklearn's elastic-net scaling.
    A strict KKT check remains the acceptance gate; unsupported or insufficiently
    converged results return ``None`` and use the reference Python solver.
    """

    backend = os.environ.get("CRYCHIC_SOLVER_BACKEND", "python").strip().lower()
    if backend not in {"sklearn", "fast", "auto"}:
        return None
    penalty = lambda1 + 2.0 * lambda2
    if penalty <= 0.0 or design.shape[0] < 1:
        return None
    try:
        from sklearn.linear_model import ElasticNet  # type: ignore[import-untyped]
    except ImportError:
        if backend in {"sklearn", "fast"}:
            return None
        return None
    try:
        sqrt_precision = np.sqrt(precision)
        weighted_design = sparse.diags(sqrt_precision, format="csc") @ design
        weighted_response = sqrt_precision * response
        n_rows = design.shape[0]
        alpha = penalty / (2.0 * n_rows)
        l1_ratio = lambda1 / penalty
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            positive=True,
            selection="cyclic",
            tol=max(tolerance * 0.1, 1e-12),
            max_iter=max_iterations,
            random_state=0,
            precompute=False,
        )
        model.fit(weighted_design, weighted_response)
        coefficients = np.asarray(model.coef_, dtype=float)
        predicted = np.asarray(design @ coefficients, dtype=float).ravel()
        if (
            coefficients.shape != (design.shape[1],)
            or np.any(~np.isfinite(coefficients))
            or np.any(coefficients < -tolerance)
            or np.any(~np.isfinite(predicted))
        ):
            return None
        coefficients = np.maximum(coefficients, 0.0)
        kkt = _kkt_violation(
            design,
            response,
            precision,
            coefficients,
            predicted,
            lambda1,
            lambda2,
            active_tolerance=tolerance,
        )
        if not math.isfinite(kkt) or kkt > kkt_tolerance:
            return None
        initial = _objective(
            response,
            np.zeros_like(response),
            precision,
            np.zeros_like(coefficients),
            lambda1,
            lambda2,
        )
        final = _objective(
            response, predicted, precision, coefficients, lambda1, lambda2
        )
        if final.total > initial.total + 1e-10 * max(1.0, abs(initial.total)):
            return None
        coefficients.setflags(write=False)
        predicted.setflags(write=False)
        diagnostics = SolverDiagnostics(
            status=SolverStatus.CONVERGED,
            objective=final,
            iterations=1,
            max_iterations=max_iterations,
            tolerance=tolerance,
            kkt_tolerance=kkt_tolerance,
            max_coordinate_change=0.0,
            kkt_violation=float(kkt),
            initialization=(
                "sklearn_cyclic_coordinate_descent_signed"
                if signed_residual_space
                else "sklearn_cyclic_coordinate_descent"
            ),
            objective_history=(initial.total, final.total),
        )
        return ElasticNetSolution(
            coefficients=coefficients,
            predicted=predicted,
            diagnostics=diagnostics,
        )
    except (ArithmeticError, RuntimeError, ValueError):
        return None


def _solve_nonnegative_elastic_net(
    matrix: sparse.spmatrix,
    response: np.ndarray,
    *,
    signed_residual_space: bool,
    precision_weights: np.ndarray | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    tolerance: float = 1e-8,
    kkt_tolerance: float | None = None,
    max_iterations: int = 10_000,
) -> ElasticNetSolution:
    """Minimize weighted squared loss plus L1/L2 under non-negativity.

    The optimized objective is
    ``sum(w * (y - B @ beta)^2) + lambda1*sum(beta) + lambda2*sum(beta^2)``.
    Coordinates are visited in canonical matrix-column order from a zero
    initialization, making repeated runs deterministic.
    """

    design = sparse.csc_matrix(matrix, dtype=float, copy=True)
    design.sum_duplicates()
    design.sort_indices()
    y = np.asarray(response, dtype=float)
    if y.ndim != 1 or design.shape[0] != len(y):
        raise ContractError(
            "Elastic-net response must be one-dimensional and align with rows",
            code="invalid_solver_shape",
            field="response",
            remediation="Align the direction-compatible channel to the basis",
        )
    if design.shape[1] == 0:
        raise ContractError(
            "Elastic-net basis must contain at least one driver",
            code="empty_solver_basis",
            field="matrix",
            remediation="Retain at least one eligible TargetPrior driver",
        )
    if np.any(~np.isfinite(y)) or (not signed_residual_space and np.any(y < 0)):
        raise ContractError(
            "Elastic-net response must be finite and direction-compatible",
            code="invalid_directional_response",
            field="response",
            remediation=(
                "Fit the positive direction-compatible channel, or explicitly "
                "enable a reviewed signed residual-space fit"
            ),
        )
    if np.any(~np.isfinite(design.data)) or (
        not signed_residual_space and np.any(design.data < 0)
    ):
        raise ContractError(
            "Elastic-net basis must be finite and direction-compatible",
            code="invalid_solver_basis",
            field="matrix",
            remediation=(
                "Use the normalized positive TargetPrior basis, or explicitly "
                "enable a reviewed signed residual-space basis"
            ),
        )
    if precision_weights is None:
        precision: np.ndarray = np.ones(len(y), dtype=float)
    else:
        precision = np.asarray(precision_weights, dtype=float)
    if precision.shape != y.shape:
        raise ContractError(
            "Precision weights must align with the response",
            code="invalid_precision_shape",
            field="precision_weights",
            remediation="Provide one precision weight per response gene",
        )
    if (
        np.any(~np.isfinite(precision))
        or np.any(precision < 0)
        or not np.any(precision > 0)
    ):
        raise ContractError(
            "Precision weights must be finite, non-negative, and not all zero",
            code="invalid_precision_weights",
            field="precision_weights",
            remediation="Use positive response-model precision for supported genes",
        )
    for field, value in (("lambda1", lambda1), ("lambda2", lambda2)):
        if not math.isfinite(value) or value < 0:
            raise ContractError(
                f"{field} must be finite and non-negative",
                code="invalid_solver_penalty",
                field=field,
                remediation="Choose pre-registered non-negative penalties",
            )
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ContractError(
            "Solver tolerance must be finite and positive",
            code="invalid_solver_tolerance",
            field="tolerance",
            remediation="Use a positive convergence tolerance",
        )
    resolved_kkt_tolerance = tolerance if kkt_tolerance is None else kkt_tolerance
    if not math.isfinite(resolved_kkt_tolerance) or resolved_kkt_tolerance <= 0:
        raise ContractError(
            "KKT tolerance must be finite and positive",
            code="invalid_solver_tolerance",
            field="kkt_tolerance",
            remediation="Use a positive KKT residual tolerance",
        )
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ContractError(
            "max_iterations must be an integer >= 1",
            code="invalid_solver_iterations",
            field="max_iterations",
            remediation="Allow at least one complete coordinate sweep",
        )

    fast_solution = _try_sklearn_coordinate_descent(
        design,
        y,
        precision,
        signed_residual_space=signed_residual_space,
        lambda1=lambda1,
        lambda2=lambda2,
        tolerance=tolerance,
        kkt_tolerance=resolved_kkt_tolerance,
        max_iterations=max_iterations,
    )
    if fast_solution is not None:
        return fast_solution

    coefficients = np.zeros(design.shape[1], dtype=float)
    predicted = np.zeros(design.shape[0], dtype=float)
    residual = y.copy()
    initial = _objective(y, predicted, precision, coefficients, lambda1, lambda2)
    history = [initial.total]
    status = SolverStatus.MAX_ITERATIONS
    failure_reason: str | None = None
    max_change = math.inf
    kkt = math.inf
    completed_iterations = 0
    for iteration in range(1, max_iterations + 1):
        max_change = 0.0
        for column in range(design.shape[1]):
            start, stop = design.indptr[column : column + 2]
            rows = design.indices[start:stop]
            values = design.data[start:stop]
            old = coefficients[column]
            denominator = float(np.dot(precision[rows], values * values)) + lambda2
            if denominator <= 0:
                new = 0.0
            else:
                partial_residual = residual[rows] + values * old
                correlation = float(np.dot(precision[rows] * values, partial_residual))
                new = max(0.0, (correlation - 0.5 * lambda1) / denominator)
            if not math.isfinite(new):
                status = SolverStatus.NUMERICAL_FAILURE
                failure_reason = f"non_finite_coordinate:{column}"
                break
            change = new - old
            if change != 0:
                coefficients[column] = new
                predicted[rows] += values * change
                residual[rows] -= values * change
                max_change = max(max_change, abs(change))
        completed_iterations = iteration
        current = _objective(y, predicted, precision, coefficients, lambda1, lambda2)
        history.append(current.total)
        if status is SolverStatus.NUMERICAL_FAILURE:
            break
        allowed_increase = 1e-10 * max(1.0, abs(history[-2]))
        if current.total > history[-2] + allowed_increase:
            status = SolverStatus.NUMERICAL_FAILURE
            failure_reason = "objective_increased"
            break
        kkt = _kkt_violation(
            design,
            y,
            precision,
            coefficients,
            predicted,
            lambda1,
            lambda2,
            active_tolerance=tolerance,
        )
        if max_change <= tolerance and kkt <= resolved_kkt_tolerance:
            status = SolverStatus.CONVERGED
            break

    final_objective = _objective(
        y, predicted, precision, coefficients, lambda1, lambda2
    )
    if status is SolverStatus.MAX_ITERATIONS:
        failure_reason = "coordinate_or_kkt_tolerance_not_met"
    diagnostics = SolverDiagnostics(
        status=status,
        objective=final_objective,
        iterations=completed_iterations,
        max_iterations=max_iterations,
        tolerance=tolerance,
        kkt_tolerance=resolved_kkt_tolerance,
        max_coordinate_change=float(max_change),
        kkt_violation=float(kkt),
        initialization="zeros",
        objective_history=tuple(float(value) for value in history),
        failure_reason=failure_reason,
    )
    coefficients.setflags(write=False)
    predicted.setflags(write=False)
    return ElasticNetSolution(
        coefficients=coefficients,
        predicted=predicted,
        diagnostics=diagnostics,
    )


def solve_nonnegative_elastic_net(
    matrix: sparse.spmatrix,
    response: np.ndarray,
    *,
    precision_weights: np.ndarray | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    tolerance: float = 1e-8,
    kkt_tolerance: float | None = None,
    max_iterations: int = 10_000,
) -> ElasticNetSolution:
    """Fit the direction-compatible non-negative basis/response contract."""

    return _solve_nonnegative_elastic_net(
        matrix,
        response,
        signed_residual_space=False,
        precision_weights=precision_weights,
        lambda1=lambda1,
        lambda2=lambda2,
        tolerance=tolerance,
        kkt_tolerance=kkt_tolerance,
        max_iterations=max_iterations,
    )


def solve_nonnegative_residual_elastic_net(
    matrix: sparse.spmatrix,
    response: np.ndarray,
    *,
    precision_weights: np.ndarray | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    tolerance: float = 1e-8,
    kkt_tolerance: float | None = None,
    max_iterations: int = 10_000,
) -> ElasticNetSolution:
    """Fit a signed residual basis/response with non-negative coefficients."""

    return _solve_nonnegative_elastic_net(
        matrix,
        response,
        signed_residual_space=True,
        precision_weights=precision_weights,
        lambda1=lambda1,
        lambda2=lambda2,
        tolerance=tolerance,
        kkt_tolerance=kkt_tolerance,
        max_iterations=max_iterations,
    )
