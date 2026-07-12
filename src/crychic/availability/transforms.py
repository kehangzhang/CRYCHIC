"""Hand-computable transforms used by availability estimation."""

from __future__ import annotations

import math
from collections.abc import Iterable

from crychic.core import ContractError

from .contracts import DetectionShrinkage, HillParameters

_DEFAULT_HILL = HillParameters()
_DEFAULT_DETECTION = DetectionShrinkage()


def hill_transform(
    value: float, parameters: HillParameters = _DEFAULT_HILL
) -> float:
    """Map non-negative expression to [0, 1) with a Hill transform."""

    if not math.isfinite(value) or value < 0:
        raise ContractError(
            "Hill-transform input must be finite and non-negative",
            code="invalid_hill_input",
            field="value",
            remediation="Provide validated non-negative state expression",
        )
    if value == 0:
        return 0.0
    ratio = parameters.half_saturation / value
    return float(1.0 / (1.0 + ratio**parameters.coefficient))


def shrink_detection_fraction(
    detection_fraction: float,
    n_cells: int,
    parameters: DetectionShrinkage = _DEFAULT_DETECTION,
) -> float:
    """Return the beta-binomial posterior mean of a detection fraction."""

    if not math.isfinite(detection_fraction) or not 0 <= detection_fraction <= 1:
        raise ContractError(
            "Detection fraction must be finite and lie in [0, 1]",
            code="invalid_detection_fraction",
            field="detection_fraction",
            remediation="Use detected cells divided by captured cells",
        )
    if isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells < 0:
        raise ContractError(
            "Detection n_cells must be a non-negative integer",
            code="invalid_detection_cell_count",
            field="n_cells",
            remediation="Use the captured cell count for the aggregate unit",
        )
    successes = detection_fraction * n_cells
    return (successes + parameters.alpha) / (
        n_cells + parameters.alpha + parameters.beta
    )


def single_gene_availability(
    expression: float,
    detection_fraction: float,
    n_cells: int,
    *,
    hill: HillParameters = _DEFAULT_HILL,
    detection: DetectionShrinkage = _DEFAULT_DETECTION,
) -> tuple[float, float, float]:
    """Return availability, Hill expression, and shrunk detection components."""

    hill_value = hill_transform(expression, hill)
    shrunk = shrink_detection_fraction(detection_fraction, n_cells, detection)
    value = hill_value * shrunk**detection.exponent
    return value, hill_value, shrunk


def generalized_harmonic_softmin(
    values: Iterable[float],
    *,
    power: float = 4.0,
    epsilon: float = 1e-12,
) -> float:
    """Aggregate required components with a smooth limiting-subunit mean.

    This implements the negative-order generalized mean on ``value + epsilon``
    and removes the numerical offset afterwards, so an unavailable zero-valued
    required subunit yields exactly zero.
    """

    collected = tuple(values)
    if not collected:
        raise ContractError(
            "Complex availability requires at least one component",
            code="empty_complex",
            field="values",
            remediation="Preserve every required complex subunit",
        )
    if power <= 0 or epsilon <= 0:
        raise ContractError(
            "Soft-min power and epsilon must be positive",
            code="invalid_complex_parameters",
            field="power",
            remediation="Use positive generalized-harmonic parameters",
        )
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in collected):
        raise ContractError(
            "Complex component availability must be finite and lie in [0, 1]",
            code="invalid_complex_component",
            field="values",
            remediation="Validate subunit availability before aggregation",
        )
    if any(value == 0 for value in collected):
        return 0.0
    shifted_inverse = sum(
        (value + epsilon) ** (-power) for value in collected
    ) / len(collected)
    return float(
        min(1.0, max(0.0, shifted_inverse ** (-1.0 / power) - epsilon))
    )
