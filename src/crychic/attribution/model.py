"""Experimental positive-channel TargetPrior attribution orchestration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from crychic.core import ContractError
from crychic.resources import TargetPrior

from .basis import build_gated_target_basis
from .contracts import (
    AttributionResult,
    AttributionSupportMethod,
    DownstreamAttributionSupport,
    DriverEstimate,
    FamilyEstimate,
    GatedTargetBasis,
)
from .families import cluster_driver_families
from .solver import solve_nonnegative_elastic_net


def fit_positive_attribution(
    basis: GatedTargetBasis,
    signed_response: np.ndarray,
    *,
    precision_weights: np.ndarray | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    cosine_threshold: float = 0.95,
    tolerance: float = 1e-8,
    kkt_tolerance: float | None = None,
    max_iterations: int = 10_000,
) -> AttributionResult:
    """Fit the positive response channel and preserve the full signed residual."""

    signed = np.asarray(signed_response, dtype=float)
    if signed.shape != (len(basis.feature_ids),) or np.any(~np.isfinite(signed)):
        raise ContractError(
            "Signed response must be finite and align with basis feature IDs",
            code="invalid_signed_response",
            field="signed_response",
            remediation="Align one complete continuous response per basis gene",
        )
    positive = np.maximum(signed, 0.0)
    if precision_weights is None:
        precision: np.ndarray = np.ones(len(signed), dtype=float)
    else:
        precision = np.asarray(precision_weights, dtype=float)
    families = cluster_driver_families(
        basis, cosine_threshold=cosine_threshold
    )
    solution = solve_nonnegative_elastic_net(
        basis.matrix,
        positive,
        precision_weights=precision,
        lambda1=lambda1,
        lambda2=lambda2,
        tolerance=tolerance,
        kkt_tolerance=kkt_tolerance,
        max_iterations=max_iterations,
    )
    family_by_driver = {
        driver: family
        for family in families
        for driver in family.driver_ids
    }
    coefficient_by_driver = dict(
        zip(basis.driver_ids, solution.coefficients, strict=True)
    )
    family_estimates: list[FamilyEstimate] = []
    family_sums: dict[str, float] = {}
    for family in families:
        coefficient_sum = float(
            sum(coefficient_by_driver[driver] for driver in family.driver_ids)
        )
        family_sums[family.family_id] = coefficient_sum
        family_estimates.append(
            FamilyEstimate(
                family_id=family.family_id,
                driver_ids=family.driver_ids,
                coefficient_sum=coefficient_sum,
                assignment_uncertainty=family.assignment_uncertainty,
                mean_pairwise_cosine=family.mean_pairwise_cosine,
            )
        )
    driver_estimates: list[DriverEstimate] = []
    for driver, coefficient in zip(
        basis.driver_ids, solution.coefficients, strict=True
    ):
        family = family_by_driver[driver]
        family_sum = family_sums[family.family_id]
        fraction = None if family_sum == 0 else float(coefficient / family_sum)
        driver_estimates.append(
            DriverEstimate(
                driver_id=driver,
                family_id=family.family_id,
                coefficient=float(coefficient),
                within_family_fraction=fraction,
                assignment_uncertainty=family.assignment_uncertainty,
            )
        )
    residual = signed - solution.predicted
    return AttributionResult(
        basis_id=basis.basis_id,
        feature_ids=basis.feature_ids,
        driver_ids=basis.driver_ids,
        coefficients=solution.coefficients,
        signed_response=signed,
        positive_response=positive,
        predicted=solution.predicted,
        residual=residual,
        precision_weights=precision,
        driver_estimates=tuple(driver_estimates),
        family_estimates=tuple(family_estimates),
        diagnostics=solution.diagnostics,
        lambda1=lambda1,
        lambda2=lambda2,
    )


def downstream_attribution_support(
    basis: GatedTargetBasis,
    attribution: AttributionResult,
    *,
    method: AttributionSupportMethod | str = (
        AttributionSupportMethod.RELATIVE_COEFFICIENT_V1
    ),
    response_norm_floor: float = 1e-8,
) -> DownstreamAttributionSupport:
    """Calculate the declared bounded support for every fitted driver."""

    if not isinstance(basis, GatedTargetBasis):
        raise TypeError("basis must be a GatedTargetBasis")
    if not isinstance(attribution, AttributionResult):
        raise TypeError("attribution must be an AttributionResult")
    if basis.basis_id != attribution.basis_id:
        raise ContractError(
            "Attribution support requires the exact fitted basis",
            code="attribution_support_basis_mismatch",
            field="basis_id",
            remediation="Pair the result with the basis used by its solver",
        )
    if basis.driver_ids != attribution.driver_ids:
        raise ContractError(
            "Attribution support drivers must align with the fitted basis",
            code="attribution_support_driver_mismatch",
            field="driver_ids",
            remediation="Preserve canonical driver order from attribution",
        )
    selected = AttributionSupportMethod(method)
    if not np.isfinite(response_norm_floor) or response_norm_floor < 0:
        raise ValueError("response_norm_floor must be finite and non-negative")
    coefficients = np.asarray(attribution.coefficients, dtype=float)
    model_explained_gain: float | None = None
    declared_response_floor: float | None = None
    if selected is AttributionSupportMethod.RELATIVE_COEFFICIENT_V1:
        numerators = coefficients.copy()
        denominator = float(np.max(coefficients, initial=0.0))
        values = (
            np.zeros_like(numerators)
            if denominator == 0
            else np.clip(numerators / denominator, 0.0, 1.0)
        )
    elif selected is AttributionSupportMethod.GATED_RESPONSE_NORM_V2:
        numerators = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(basis.matrix.getcol(column).toarray()).ravel()
                    * coefficient
                )
                for column, coefficient in enumerate(coefficients)
            ],
            dtype=float,
        )
        denominator = float(np.linalg.norm(attribution.positive_response))
        values = (
            np.zeros_like(numerators)
            if denominator == 0
            else np.clip(numerators / denominator, 0.0, 1.0)
        )
    else:
        positive = np.asarray(attribution.positive_response, dtype=float)
        fitted = np.asarray(attribution.predicted, dtype=float)
        precision = np.asarray(attribution.precision_weights, dtype=float)
        null_loss = float(np.dot(precision, positive * positive))
        response_norm = math.sqrt(max(0.0, null_loss))
        declared_response_floor = response_norm_floor
        if null_loss <= 0 or response_norm <= response_norm_floor:
            model_explained_gain = 0.0
        else:
            residual = positive - fitted
            fitted_loss = float(np.dot(precision, residual * residual))
            model_explained_gain = float(
                np.clip((null_loss - fitted_loss) / null_loss, 0.0, 1.0)
            )
        sqrt_precision = np.sqrt(precision)
        numerators = np.asarray(
            [
                np.linalg.norm(
                    sqrt_precision
                    * np.asarray(basis.matrix.getcol(column).toarray()).ravel()
                    * coefficient
                )
                for column, coefficient in enumerate(coefficients)
            ],
            dtype=float,
        )
        denominator = float(numerators.sum())
        values = (
            np.zeros_like(numerators)
            if denominator == 0 or model_explained_gain == 0
            else model_explained_gain * numerators / denominator
        )
    return DownstreamAttributionSupport(
        basis_id=basis.basis_id,
        driver_ids=basis.driver_ids,
        method=selected,
        values=values,
        numerator_values=numerators,
        denominator_value=denominator,
        model_explained_gain=model_explained_gain,
        response_norm_floor=declared_response_floor,
    )


def attribute_target_prior(
    prior: TargetPrior,
    feature_ids: Sequence[str],
    receptor_gates: Mapping[str, float],
    signed_response: np.ndarray,
    *,
    precision_weights: np.ndarray | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    cosine_threshold: float = 0.95,
    tolerance: float = 1e-8,
    kkt_tolerance: float | None = None,
    max_iterations: int = 10_000,
) -> tuple[GatedTargetBasis, AttributionResult]:
    """Build a gated TargetPrior basis and fit experimental attribution."""

    basis = build_gated_target_basis(prior, feature_ids, receptor_gates)
    result = fit_positive_attribution(
        basis,
        signed_response,
        precision_weights=precision_weights,
        lambda1=lambda1,
        lambda2=lambda2,
        cosine_threshold=cosine_threshold,
        tolerance=tolerance,
        kkt_tolerance=kkt_tolerance,
        max_iterations=max_iterations,
    )
    return basis, result
