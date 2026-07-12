"""Producer-owned contracts for experimental receiver attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import sparse

from crychic.core import ContractError


def _readonly_vector(values: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def _readonly_csc(matrix: sparse.spmatrix) -> sparse.csc_matrix:
    result = sparse.csc_matrix(matrix, dtype=float, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.data.setflags(write=False)
    result.indices.setflags(write=False)
    result.indptr.setflags(write=False)
    return result


class SolverStatus(StrEnum):
    """Terminal state of the deterministic coordinate-descent solver."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    NUMERICAL_FAILURE = "numerical_failure"


class AttributionSupportMethod(StrEnum):
    """Explicit downstream scaling contract for fitted driver attribution."""

    RELATIVE_COEFFICIENT_V1 = "gated_prior_attribution_v1"
    GATED_RESPONSE_NORM_V2 = "gated_response_norm_attribution_support_v2"

    def to_dict(self) -> dict[str, object]:
        """Return manifest-ready provenance for the selected formula."""

        if self is AttributionSupportMethod.RELATIVE_COEFFICIENT_V1:
            numerator = "nonnegative_driver_coefficient"
            denominator = "maximum_nonnegative_driver_coefficient"
            formula = "beta_j/max_k_beta_k"
            version = 1
        else:
            numerator = "l2_norm_gated_basis_column_times_coefficient"
            denominator = "l2_norm_positive_response_channel"
            formula = "norm2(B_j*beta_j)/norm2(y_positive)"
            version = 2
        return {
            "method": self.value,
            "version": version,
            "formula": formula,
            "numerator": numerator,
            "denominator": denominator,
            "clip": [0.0, 1.0],
            "gate_aware": (
                self is AttributionSupportMethod.GATED_RESPONSE_NORM_V2
            ),
            "application": "multiply_sample_target_activity",
            "experimental": True,
        }


@dataclass(frozen=True, slots=True)
class BasisBuildReport:
    """Auditable target alignment, normalization, and gate summary."""

    prior_targets: int
    response_features: int
    matched_targets: int
    unmatched_prior_targets: tuple[str, ...]
    zero_norm_drivers: tuple[str, ...]
    zero_gate_drivers: tuple[str, ...]

    def __post_init__(self) -> None:
        if min(self.prior_targets, self.response_features, self.matched_targets) < 0:
            raise ContractError(
                "Basis build counts must be non-negative",
                code="invalid_basis_report",
                field="basis_report",
                remediation="Recompute target alignment diagnostics",
            )
        for field_name in (
            "unmatched_prior_targets",
            "zero_norm_drivers",
            "zero_gate_drivers",
        ):
            object.__setattr__(
                self, field_name, tuple(sorted(set(getattr(self, field_name))))
            )


@dataclass(frozen=True, slots=True)
class GatedTargetBasis:
    """Response-aligned normalized prior profiles and receptor-gated basis."""

    basis_id: str
    feature_ids: tuple[str, ...]
    driver_ids: tuple[str, ...]
    normalized_profiles: sparse.csc_matrix
    matrix: sparse.csc_matrix
    receptor_gates: np.ndarray
    pre_normalization_norms: np.ndarray
    prior_resource_id: str
    prior_version: str
    report: BasisBuildReport

    def __post_init__(self) -> None:
        expected = (len(self.feature_ids), len(self.driver_ids))
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ContractError(
                "Basis feature IDs must be unique",
                code="duplicate_basis_feature",
                field="feature_ids",
                remediation="Collapse duplicate response genes before attribution",
            )
        if tuple(sorted(set(self.driver_ids))) != self.driver_ids:
            raise ContractError(
                "Basis driver IDs must be unique and sorted",
                code="invalid_basis_drivers",
                field="driver_ids",
                remediation="Use the canonical TargetPrior driver order",
            )
        normalized = _readonly_csc(self.normalized_profiles)
        gated = _readonly_csc(self.matrix)
        if normalized.shape != expected or gated.shape != expected:
            raise ContractError(
                "Basis matrices must have feature by driver shape",
                code="invalid_basis_shape",
                field="matrix",
                remediation="Align the TargetPrior to the response feature IDs",
            )
        gates = _readonly_vector(self.receptor_gates)
        norms = _readonly_vector(self.pre_normalization_norms)
        if gates.shape != (len(self.driver_ids),) or norms.shape != gates.shape:
            raise ContractError(
                "Basis gates and norms must align with driver IDs",
                code="invalid_basis_shape",
                field="receptor_gates",
                remediation="Provide one receptor gate per prior driver",
            )
        if np.any(~np.isfinite(gates)) or np.any((gates < 0) | (gates > 1)):
            raise ContractError(
                "Receptor gates must be finite and lie in [0, 1]",
                code="invalid_receptor_gate",
                field="receptor_gates",
                remediation="Use fold-frozen receptor availability gates",
            )
        if np.any(~np.isfinite(norms)) or np.any(norms < 0):
            raise ContractError(
                "Prior column norms must be finite and non-negative",
                code="invalid_prior_norm",
                field="pre_normalization_norms",
                remediation="Validate prior weights before building the basis",
            )
        object.__setattr__(self, "normalized_profiles", normalized)
        object.__setattr__(self, "matrix", gated)
        object.__setattr__(self, "receptor_gates", gates)
        object.__setattr__(self, "pre_normalization_norms", norms)


@dataclass(frozen=True, slots=True)
class DownstreamAttributionSupport:
    """Bounded per-driver support used to scale sample target activity."""

    basis_id: str
    driver_ids: tuple[str, ...]
    method: AttributionSupportMethod
    values: np.ndarray
    numerator_values: np.ndarray
    denominator_value: float
    experimental: bool = True

    def __post_init__(self) -> None:
        if not self.basis_id:
            raise ContractError(
                "Attribution support requires a basis ID",
                code="invalid_attribution_support",
                field="basis_id",
                remediation="Retain the exact gated target basis identity",
            )
        if not self.driver_ids or len(set(self.driver_ids)) != len(self.driver_ids):
            raise ContractError(
                "Attribution support driver IDs must be non-empty and unique",
                code="invalid_attribution_support",
                field="driver_ids",
                remediation="Align support values to canonical basis drivers",
            )
        method = AttributionSupportMethod(self.method)
        values = _readonly_vector(self.values)
        numerators = _readonly_vector(self.numerator_values)
        expected = (len(self.driver_ids),)
        if values.shape != expected or numerators.shape != expected:
            raise ContractError(
                "Attribution support vectors must align with driver IDs",
                code="invalid_attribution_support",
                field="values",
                remediation="Emit one support and numerator per basis driver",
            )
        if (
            np.any(~np.isfinite(values))
            or np.any((values < 0) | (values > 1))
            or np.any(~np.isfinite(numerators))
            or np.any(numerators < 0)
        ):
            raise ContractError(
                "Attribution support must be finite in [0, 1] with "
                "non-negative numerators",
                code="invalid_attribution_support",
                field="values",
                remediation="Validate and clip only the declared support ratio",
            )
        denominator = float(self.denominator_value)
        if not math.isfinite(denominator) or denominator < 0:
            raise ContractError(
                "Attribution support denominator must be finite and non-negative",
                code="invalid_attribution_support",
                field="denominator_value",
                remediation="Use the declared finite v1 or v2 normalizer",
            )
        if denominator == 0 and np.any(values != 0):
            raise ContractError(
                "Zero-denominator attribution support must be exactly zero",
                code="invalid_attribution_support",
                field="values",
                remediation="Return zero support when the normalizer is zero",
            )
        if not self.experimental:
            raise ContractError(
                "Downstream attribution support remains experimental",
                code="invalid_attribution_support",
                field="experimental",
                remediation="Do not promote candidate support scaling to inference",
            )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "numerator_values", numerators)
        object.__setattr__(self, "denominator_value", denominator)

    @property
    def version(self) -> int:
        return (
            1
            if self.method is AttributionSupportMethod.RELATIVE_COEFFICIENT_V1
            else 2
        )

    @property
    def gate_aware(self) -> bool:
        return self.method is AttributionSupportMethod.GATED_RESPONSE_NORM_V2

    def to_dict(self) -> dict[str, object]:
        """Return manifest-ready method provenance without fitted values."""

        return self.method.to_dict()


@dataclass(frozen=True, slots=True)
class DriverFamilyDefinition:
    """Stable equivalence family derived from ligand target-profile cosine."""

    family_id: str
    driver_ids: tuple[str, ...]
    mean_pairwise_cosine: float
    assignment_uncertainty: float

    def __post_init__(self) -> None:
        if not self.family_id or not self.driver_ids:
            raise ContractError(
                "Driver family ID and members cannot be empty",
                code="invalid_driver_family",
                field="driver_family",
                remediation="Cluster canonical non-empty prior driver profiles",
            )
        if tuple(sorted(set(self.driver_ids))) != self.driver_ids:
            raise ContractError(
                "Driver family members must be unique and sorted",
                code="invalid_driver_family",
                field="driver_ids",
                remediation="Canonicalize family members before assigning an ID",
            )
        for field_name in ("mean_pairwise_cosine", "assignment_uncertainty"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ContractError(
                    f"{field_name} must lie in [0, 1]",
                    code="invalid_driver_family",
                    field=field_name,
                    remediation="Recompute cosine-based family diagnostics",
                )


@dataclass(frozen=True, slots=True)
class ObjectiveTerms:
    """Final positive-channel elastic-net objective decomposition."""

    weighted_loss: float
    l1_penalty: float
    l2_penalty: float
    total: float

    def __post_init__(self) -> None:
        values = (
            self.weighted_loss,
            self.l1_penalty,
            self.l2_penalty,
            self.total,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ContractError(
                "Objective terms must be finite and non-negative",
                code="invalid_solver_objective",
                field="objective",
                remediation="Do not report a numerically failed solve as successful",
            )
        expected = self.weighted_loss + self.l1_penalty + self.l2_penalty
        if not math.isclose(self.total, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ContractError(
                "Objective total does not equal its components",
                code="invalid_solver_objective",
                field="total",
                remediation="Recompute objective diagnostics from final coefficients",
            )


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    """Coordinate-descent terminal diagnostics without inferential claims."""

    status: SolverStatus
    objective: ObjectiveTerms
    iterations: int
    max_iterations: int
    tolerance: float
    kkt_tolerance: float
    max_coordinate_change: float
    kkt_violation: float
    initialization: str
    objective_history: tuple[float, ...]
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SolverStatus(self.status))
        if self.iterations < 1 or self.iterations > self.max_iterations:
            raise ContractError(
                "Solver iterations must lie in [1, max_iterations]",
                code="invalid_solver_diagnostics",
                field="iterations",
                remediation="Report completed full coordinate sweeps",
            )
        if len(self.objective_history) != self.iterations + 1:
            raise ContractError(
                "Objective history must contain initialization and every sweep",
                code="invalid_solver_diagnostics",
                field="objective_history",
                remediation="Persist the initial and per-iteration objective",
            )
        if any(
            not math.isfinite(value) or value < 0
            for value in self.objective_history
        ):
            raise ContractError(
                "Objective history must be finite and non-negative",
                code="invalid_solver_diagnostics",
                field="objective_history",
                remediation="Mark numerical failure before persisting invalid values",
            )
        if self.status is SolverStatus.CONVERGED:
            if self.failure_reason is not None:
                raise ContractError(
                    "Converged solver diagnostics cannot include a failure reason",
                    code="invalid_solver_diagnostics",
                    field="failure_reason",
                    remediation="Keep success and failure terminal states distinct",
                )
            if (
                not math.isfinite(self.max_coordinate_change)
                or not math.isfinite(self.kkt_violation)
                or self.max_coordinate_change > self.tolerance
                or self.kkt_violation > self.kkt_tolerance
            ):
                raise ContractError(
                    "Converged status requires coordinate and KKT tolerances",
                    code="false_solver_convergence",
                    field="status",
                    remediation="Return a non-converged terminal status",
                )
        elif not self.failure_reason:
            raise ContractError(
                "Non-converged solver diagnostics require a failure reason",
                code="invalid_solver_diagnostics",
                field="failure_reason",
                remediation="Expose why the solver did not converge",
            )

    @property
    def converged(self) -> bool:
        """Whether both coordinate and KKT criteria were satisfied."""

        return self.status is SolverStatus.CONVERGED


@dataclass(frozen=True, slots=True)
class DriverEstimate:
    """One concrete ligand/driver coefficient and its family assignment."""

    driver_id: str
    family_id: str
    coefficient: float
    within_family_fraction: float | None
    assignment_uncertainty: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ContractError(
                "Driver coefficient must be finite and non-negative",
                code="invalid_driver_estimate",
                field="coefficient",
                remediation="Use the unmodified non-negative solver coefficient",
            )
        if self.within_family_fraction is not None and not (
            0 <= self.within_family_fraction <= 1
        ):
            raise ContractError(
                "Within-family fraction must lie in [0, 1]",
                code="invalid_driver_estimate",
                field="within_family_fraction",
                remediation="Normalize specific coefficients by the family sum",
            )
        if not 0 <= self.assignment_uncertainty <= 1:
            raise ContractError(
                "Assignment uncertainty must lie in [0, 1]",
                code="invalid_driver_estimate",
                field="assignment_uncertainty",
                remediation="Use the cosine-family ambiguity measure",
            )


@dataclass(frozen=True, slots=True)
class FamilyEstimate:
    """Family-level coefficient sum for collinear target profiles."""

    family_id: str
    driver_ids: tuple[str, ...]
    coefficient_sum: float
    assignment_uncertainty: float
    mean_pairwise_cosine: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient_sum) or self.coefficient_sum < 0:
            raise ContractError(
                "Family coefficient sum must be finite and non-negative",
                code="invalid_family_estimate",
                field="coefficient_sum",
                remediation="Sum the concrete non-negative driver coefficients",
            )
        for field_name in ("assignment_uncertainty", "mean_pairwise_cosine"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ContractError(
                    f"{field_name} must lie in [0, 1]",
                    code="invalid_family_estimate",
                    field=field_name,
                    remediation="Use the cosine family diagnostics",
                )


@dataclass(frozen=True, slots=True)
class AttributionResult:
    """Experimental positive-channel attribution with complete signed residual."""

    basis_id: str
    feature_ids: tuple[str, ...]
    driver_ids: tuple[str, ...]
    coefficients: np.ndarray
    signed_response: np.ndarray
    positive_response: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    precision_weights: np.ndarray
    driver_estimates: tuple[DriverEstimate, ...]
    family_estimates: tuple[FamilyEstimate, ...]
    diagnostics: SolverDiagnostics
    lambda1: float
    lambda2: float
    response_channel: str = "positive"
    experimental: bool = True

    def __post_init__(self) -> None:
        coefficients = _readonly_vector(self.coefficients)
        vectors = {
            name: _readonly_vector(getattr(self, name))
            for name in (
                "signed_response",
                "positive_response",
                "predicted",
                "residual",
                "precision_weights",
            )
        }
        if coefficients.shape != (len(self.driver_ids),):
            raise ContractError(
                "Attribution coefficients must align with driver IDs",
                code="invalid_attribution_shape",
                field="coefficients",
                remediation="Fit the coefficient vector against this basis",
            )
        expected = (len(self.feature_ids),)
        if any(vector.shape != expected for vector in vectors.values()):
            raise ContractError(
                "Attribution response vectors must align with feature IDs",
                code="invalid_attribution_shape",
                field="signed_response",
                remediation="Align response and precision to the gated basis",
            )
        if np.any(coefficients < 0) or np.any(~np.isfinite(coefficients)):
            raise ContractError(
                "Attribution coefficients must be finite and non-negative",
                code="invalid_attribution_coefficient",
                field="coefficients",
                remediation="Use the non-negative solver result without mutation",
            )
        if (
            np.any(~np.isfinite(vectors["signed_response"]))
            or np.any(~np.isfinite(vectors["positive_response"]))
            or np.any(~np.isfinite(vectors["predicted"]))
            or np.any(~np.isfinite(vectors["residual"]))
            or np.any(~np.isfinite(vectors["precision_weights"]))
            or np.any(vectors["precision_weights"] < 0)
            or not np.any(vectors["precision_weights"] > 0)
        ):
            raise ContractError(
                "Attribution response and precision vectors must be finite",
                code="invalid_attribution_vector",
                field="precision_weights",
                remediation="Use complete response and supported precision weights",
            )
        if not np.array_equal(
            vectors["positive_response"],
            np.maximum(vectors["signed_response"], 0.0),
        ):
            raise ContractError(
                "Positive response must be the direction-compatible signed channel",
                code="invalid_directional_response",
                field="positive_response",
                remediation="Fit max(signed_response, 0) in the v0.1 model",
            )
        if np.any(vectors["predicted"] < -1e-12):
            raise ContractError(
                "Positive-prior prediction cannot be negative",
                code="invalid_attribution_prediction",
                field="predicted",
                remediation="Use a non-negative basis and coefficients",
            )
        if not np.allclose(
            vectors["signed_response"],
            vectors["predicted"] + vectors["residual"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ContractError(
                "Predicted response plus residual must reconstruct signed response",
                code="invalid_attribution_reconstruction",
                field="residual",
                remediation="Compute residual against the complete signed response",
            )
        if self.response_channel != "positive" or not self.experimental:
            raise ContractError(
                "V0.1 attribution must remain positive-channel and experimental",
                code="invalid_attribution_semantics",
                field="response_channel",
                remediation="Do not promote the v0.1 directional baseline to inference",
            )
        if not math.isfinite(self.lambda1) or not math.isfinite(self.lambda2):
            raise ContractError(
                "Attribution penalties must be finite",
                code="invalid_solver_penalty",
                field="lambda1",
                remediation="Persist the validated solver penalties",
            )
        if self.lambda1 < 0 or self.lambda2 < 0:
            raise ContractError(
                "Attribution penalties must be non-negative",
                code="invalid_solver_penalty",
                field="lambda1",
                remediation="Persist the validated solver penalties",
            )
        if tuple(item.driver_id for item in self.driver_estimates) != self.driver_ids:
            raise ContractError(
                "Driver estimates must align with driver IDs",
                code="invalid_attribution_drivers",
                field="driver_estimates",
                remediation="Preserve canonical basis driver order",
            )
        if not np.allclose(
            coefficients,
            np.asarray([item.coefficient for item in self.driver_estimates]),
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ContractError(
                "Concrete driver estimates must equal solver coefficients",
                code="invalid_attribution_drivers",
                field="driver_estimates",
                remediation="Build estimates directly from the solver coefficients",
            )
        family_by_id = {item.family_id: item for item in self.family_estimates}
        for family_id, family in family_by_id.items():
            driver_sum = sum(
                item.coefficient
                for item in self.driver_estimates
                if item.family_id == family_id
            )
            if not math.isclose(
                driver_sum,
                family.coefficient_sum,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ContractError(
                    "Family sum must reconstruct concrete driver coefficients",
                    code="invalid_family_estimate",
                    field="coefficient_sum",
                    remediation="Aggregate every family member exactly once",
                )
        object.__setattr__(self, "coefficients", coefficients)
        for name, vector in vectors.items():
            object.__setattr__(self, name, vector)

    @property
    def succeeded(self) -> bool:
        """Return true only for an explicitly converged solve."""

        return self.diagnostics.converged
