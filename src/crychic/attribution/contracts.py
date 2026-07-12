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


class ReceptorGatePolicy(StrEnum):
    """Versioned interpretation of receptor evidence in attribution bases."""

    LEGACY_CONTINUOUS_V1 = "legacy_continuous_v1"
    HARD_ELIGIBILITY_V2 = "hard_eligibility_v2"

    @property
    def version(self) -> int:
        """Return the policy contract version."""

        return {
            ReceptorGatePolicy.LEGACY_CONTINUOUS_V1: 1,
            ReceptorGatePolicy.HARD_ELIGIBILITY_V2: 2,
        }[self]

    def to_dict(self) -> dict[str, object]:
        """Return policy-level provenance independent of a fitted basis."""

        if self is ReceptorGatePolicy.LEGACY_CONTINUOUS_V1:
            matrix_semantics = "normalized_profile_times_continuous_gate"
            threshold_operator = None
        else:
            matrix_semantics = "unit_normalized_profile_if_eligible_else_zero"
            threshold_operator = "receptor_gate >= threshold"
        return {
            "policy": self.value,
            "version": self.version,
            "matrix_semantics": matrix_semantics,
            "threshold_operator": threshold_operator,
        }


class FamilyBasisMethod(StrEnum):
    """Versioned construction rule for one solver column per strict family."""

    STRICT_MEDOID_V1 = "strict_medoid_family_basis_v1"

    @property
    def version(self) -> int:
        """Return the family-basis contract version."""

        return 1

    def to_dict(self) -> dict[str, object]:
        """Return manifest-ready family-basis provenance."""

        return {
            "method": self.value,
            "version": self.version,
            "profile": "deterministic_maximum_mean_cosine_medoid",
            "normalization": "unit_l2",
            "fit_grain": "strict_driver_family",
            "experimental": True,
        }


class FamilyMemberAllocationMethod(StrEnum):
    """Versioned evidence-only allocation within fitted families."""

    COMPLETE_MULTIPLICATIVE_EVIDENCE_V1 = "complete_multiplicative_member_evidence_v1"

    @property
    def version(self) -> int:
        """Return the member-allocation contract version."""

        return 1

    def to_dict(self) -> dict[str, object]:
        """Return manifest-ready allocation provenance."""

        return {
            "method": self.value,
            "version": self.version,
            "components": [
                "receptor_availability",
                "ligand_availability",
                "prior_quality",
                "subject_prevalence",
                "resource_evidence",
            ],
            "combination": "product_then_family_normalize",
            "missingness": "complete_case_family_or_unresolved",
            "uses_receiver_response": False,
            "experimental": True,
        }


class LRIdentifiabilityStatus(StrEnum):
    """Whether evidence supports a within-family member allocation."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class AttributionSupportMethod(StrEnum):
    """Explicit downstream scaling contract for fitted driver attribution."""

    RELATIVE_COEFFICIENT_V1 = "gated_prior_attribution_v1"
    GATED_RESPONSE_NORM_V2 = "gated_response_norm_attribution_support_v2"
    EXPLAINED_SHARE_V3 = "explained_share_attribution_support_v3"

    def to_dict(self) -> dict[str, object]:
        """Return manifest-ready provenance for the selected formula."""

        if self is AttributionSupportMethod.RELATIVE_COEFFICIENT_V1:
            numerator = "nonnegative_driver_coefficient"
            denominator = "maximum_nonnegative_driver_coefficient"
            formula = "beta_j/max_k_beta_k"
            version = 1
        elif self is AttributionSupportMethod.GATED_RESPONSE_NORM_V2:
            numerator = "l2_norm_gated_basis_column_times_coefficient"
            denominator = "l2_norm_positive_response_channel"
            formula = "norm2(B_j*beta_j)/norm2(y_positive)"
            version = 2
        else:
            numerator = "weighted_driver_contribution_norm"
            denominator = "sum_weighted_driver_contribution_norms"
            formula = "explained_gain*contribution_j/sum_k_contribution_k"
            version = 3
        return {
            "method": self.value,
            "version": version,
            "formula": formula,
            "numerator": numerator,
            "denominator": denominator,
            "clip": [0.0, 1.0],
            "gate_aware": self
            in {
                AttributionSupportMethod.GATED_RESPONSE_NORM_V2,
                AttributionSupportMethod.EXPLAINED_SHARE_V3,
            },
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
    receptor_eligible: np.ndarray
    gate_policy: ReceptorGatePolicy
    receptor_gate_threshold: float | None
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
        raw_eligible = np.asarray(self.receptor_eligible)
        norms = _readonly_vector(self.pre_normalization_norms)
        if (
            gates.shape != (len(self.driver_ids),)
            or raw_eligible.shape != gates.shape
            or norms.shape != gates.shape
        ):
            raise ContractError(
                "Basis gates, eligibility, and norms must align with driver IDs",
                code="invalid_basis_shape",
                field="receptor_gates",
                remediation="Provide one receptor gate per prior driver",
            )
        if raw_eligible.dtype.kind != "b":
            raise ContractError(
                "Receptor eligibility must be an explicit boolean mask",
                code="invalid_receptor_eligibility",
                field="receptor_eligible",
                remediation="Persist one boolean eligibility value per driver",
            )
        eligible = raw_eligible.astype(bool, copy=True)
        eligible.setflags(write=False)
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
        try:
            policy = ReceptorGatePolicy(self.gate_policy)
        except (TypeError, ValueError) as error:
            raise ContractError(
                "Attribution basis requires a recognized receptor gate policy",
                code="invalid_receptor_gate_policy",
                field="gate_policy",
                remediation="Use an explicit versioned ReceptorGatePolicy",
            ) from error
        threshold = self.receptor_gate_threshold
        if policy is ReceptorGatePolicy.LEGACY_CONTINUOUS_V1:
            if threshold is not None:
                raise ContractError(
                    "Legacy continuous receptor gating does not use a threshold",
                    code="invalid_receptor_gate_threshold",
                    field="receptor_gate_threshold",
                    remediation="Leave the threshold unset or select hard eligibility",
                )
            expected_eligible = gates > 0
            column_scale = gates
        else:
            if isinstance(threshold, bool) or threshold is None:
                raise ContractError(
                    "Hard receptor eligibility requires an explicit threshold",
                    code="invalid_receptor_gate_threshold",
                    field="receptor_gate_threshold",
                    remediation="Provide a finite threshold in (0, 1]",
                )
            threshold = float(threshold)
            if not math.isfinite(threshold) or not 0 < threshold <= 1:
                raise ContractError(
                    "Hard receptor eligibility threshold must lie in (0, 1]",
                    code="invalid_receptor_gate_threshold",
                    field="receptor_gate_threshold",
                    remediation="Choose a pre-registered threshold in (0, 1]",
                )
            expected_eligible = gates >= threshold
            column_scale = expected_eligible.astype(float)
        if not np.array_equal(eligible, expected_eligible):
            raise ContractError(
                "Receptor eligibility mask does not match gates and policy",
                code="invalid_receptor_eligibility",
                field="receptor_eligible",
                remediation="Recompute eligibility from the recorded gate policy",
            )
        expected_matrix = normalized.multiply(column_scale).tocsc()
        difference = (gated - expected_matrix).tocsc()
        if difference.nnz and not np.allclose(
            difference.data, 0.0, rtol=1e-12, atol=1e-14
        ):
            raise ContractError(
                "Attribution basis matrix does not implement its gate policy",
                code="invalid_receptor_gate_matrix",
                field="matrix",
                remediation="Rebuild the solver matrix from normalized profiles",
            )
        object.__setattr__(self, "normalized_profiles", normalized)
        object.__setattr__(self, "matrix", gated)
        object.__setattr__(self, "receptor_gates", gates)
        object.__setattr__(self, "receptor_eligible", eligible)
        object.__setattr__(self, "gate_policy", policy)
        object.__setattr__(self, "receptor_gate_threshold", threshold)
        object.__setattr__(self, "pre_normalization_norms", norms)

    def gate_provenance(self) -> dict[str, object]:
        """Return manifest-ready gate policy, threshold, and mask provenance."""

        return {
            **self.gate_policy.to_dict(),
            "threshold": self.receptor_gate_threshold,
            "driver_ids": list(self.driver_ids),
            "eligible_mask": self.receptor_eligible.tolist(),
            "eligible_driver_ids": [
                driver
                for driver, eligible in zip(
                    self.driver_ids, self.receptor_eligible, strict=True
                )
                if eligible
            ],
        }


@dataclass(frozen=True, slots=True)
class DownstreamAttributionSupport:
    """Bounded per-driver support used to scale sample target activity."""

    basis_id: str
    driver_ids: tuple[str, ...]
    method: AttributionSupportMethod
    values: np.ndarray
    numerator_values: np.ndarray
    denominator_value: float
    model_explained_gain: float | None = None
    response_norm_floor: float | None = None
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
        explained_gain = self.model_explained_gain
        response_norm_floor = self.response_norm_floor
        if method is AttributionSupportMethod.EXPLAINED_SHARE_V3:
            if (
                explained_gain is None
                or not math.isfinite(explained_gain)
                or not 0 <= explained_gain <= 1
            ):
                raise ContractError(
                    "Explained-share support requires a model gain in [0, 1]",
                    code="invalid_attribution_support",
                    field="model_explained_gain",
                    remediation="Compute weighted model-level explained gain",
                )
            if (
                response_norm_floor is None
                or not math.isfinite(response_norm_floor)
                or response_norm_floor < 0
            ):
                raise ContractError(
                    "Explained-share support requires a non-negative response floor",
                    code="invalid_attribution_support",
                    field="response_norm_floor",
                    remediation="Declare the tiny-response suppression threshold",
                )
            if float(values.sum()) > explained_gain + 1e-10:
                raise ContractError(
                    "Explained-share support cannot exceed model explained gain",
                    code="invalid_attribution_support",
                    field="values",
                    remediation="Allocate the model gain across driver contributions",
                )
        elif explained_gain is not None or response_norm_floor is not None:
            raise ContractError(
                "Model gain diagnostics are specific to explained-share support",
                code="invalid_attribution_support",
                field="model_explained_gain",
                remediation="Leave v1/v2 explained-share diagnostics unset",
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
        object.__setattr__(self, "model_explained_gain", explained_gain)
        object.__setattr__(self, "response_norm_floor", response_norm_floor)

    @property
    def version(self) -> int:
        return {
            AttributionSupportMethod.RELATIVE_COEFFICIENT_V1: 1,
            AttributionSupportMethod.GATED_RESPONSE_NORM_V2: 2,
            AttributionSupportMethod.EXPLAINED_SHARE_V3: 3,
        }[self.method]

    @property
    def gate_aware(self) -> bool:
        return self.method in {
            AttributionSupportMethod.GATED_RESPONSE_NORM_V2,
            AttributionSupportMethod.EXPLAINED_SHARE_V3,
        }

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
            not math.isfinite(value) or value < 0 for value in self.objective_history
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
class FamilyFirstBasis:
    """One deterministic normalized solver column per strict driver family."""

    family_basis_id: str
    source_basis_id: str
    feature_ids: tuple[str, ...]
    family_definitions: tuple[DriverFamilyDefinition, ...]
    matrix: sparse.csc_matrix
    medoid_driver_ids: tuple[str, ...]
    family_eligible: np.ndarray
    strict_cosine_threshold: float
    method: FamilyBasisMethod = FamilyBasisMethod.STRICT_MEDOID_V1
    experimental: bool = True

    def __post_init__(self) -> None:
        if not self.family_basis_id or not self.source_basis_id:
            raise ContractError(
                "Family-first basis IDs must be non-empty",
                code="invalid_family_first_basis",
                field="family_basis_id",
                remediation="Retain the source and derived basis identities",
            )
        if not self.feature_ids or len(set(self.feature_ids)) != len(self.feature_ids):
            raise ContractError(
                "Family-first basis feature IDs must be non-empty and unique",
                code="invalid_family_first_basis",
                field="feature_ids",
                remediation="Align one canonical response feature universe",
            )
        definitions = tuple(self.family_definitions)
        if (
            not definitions
            or tuple(sorted(definitions, key=lambda family: family.family_id))
            != definitions
        ):
            raise ContractError(
                "Family definitions must be non-empty and sorted by family ID",
                code="invalid_family_first_basis",
                field="family_definitions",
                remediation="Canonicalize strict families before basis construction",
            )
        family_ids = tuple(family.family_id for family in definitions)
        if len(set(family_ids)) != len(family_ids):
            raise ContractError(
                "Family-first basis family IDs must be unique",
                code="invalid_family_first_basis",
                field="family_definitions",
                remediation="Retain each strict family exactly once",
            )
        members = [driver for family in definitions for driver in family.driver_ids]
        if len(set(members)) != len(members):
            raise ContractError(
                "Family-first basis members must belong to exactly one family",
                code="invalid_family_first_basis",
                field="family_definitions",
                remediation="Provide a disjoint strict-family partition",
            )
        medoids = tuple(self.medoid_driver_ids)
        if len(medoids) != len(definitions) or any(
            medoid not in family.driver_ids
            for medoid, family in zip(medoids, definitions, strict=True)
        ):
            raise ContractError(
                "Every family medoid must name one member of its family",
                code="invalid_family_first_basis",
                field="medoid_driver_ids",
                remediation="Select medoids from the corresponding strict family",
            )
        matrix = _readonly_csc(self.matrix)
        expected_shape = (len(self.feature_ids), len(definitions))
        raw_eligible = np.asarray(self.family_eligible)
        if matrix.shape != expected_shape or raw_eligible.shape != (len(definitions),):
            raise ContractError(
                "Family-first matrix and eligibility must align with families",
                code="invalid_family_first_basis",
                field="matrix",
                remediation="Emit one feature vector and eligibility per family",
            )
        if raw_eligible.dtype.kind != "b":
            raise ContractError(
                "Family eligibility must be an explicit boolean mask",
                code="invalid_family_first_basis",
                field="family_eligible",
                remediation="Persist one boolean eligibility value per family",
            )
        eligible = raw_eligible.astype(bool, copy=True)
        eligible.setflags(write=False)
        if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0):
            raise ContractError(
                "Family-first basis must be finite and non-negative",
                code="invalid_family_first_basis",
                field="matrix",
                remediation="Use normalized non-negative TargetPrior profiles",
            )
        column_norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
        if np.any(
            eligible & ~np.isclose(column_norms, 1.0, rtol=1e-10, atol=1e-12)
        ) or np.any((~eligible) & (column_norms != 0)):
            raise ContractError(
                "Eligible family columns must be unit norm and ineligible columns zero",
                code="invalid_family_first_basis",
                field="matrix",
                remediation="Rebuild medoid columns under family eligibility",
            )
        threshold = float(self.strict_cosine_threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ContractError(
                "Strict family cosine threshold must lie in [0, 1]",
                code="invalid_family_first_basis",
                field="strict_cosine_threshold",
                remediation="Use the pre-registered strict-family threshold",
            )
        method = FamilyBasisMethod(self.method)
        if not self.experimental:
            raise ContractError(
                "Family-first attribution remains experimental",
                code="invalid_family_first_basis",
                field="experimental",
                remediation="Keep the legacy attribution path as the default",
            )
        object.__setattr__(self, "family_definitions", definitions)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "medoid_driver_ids", medoids)
        object.__setattr__(self, "family_eligible", eligible)
        object.__setattr__(self, "strict_cosine_threshold", threshold)
        object.__setattr__(self, "method", method)

    @property
    def family_ids(self) -> tuple[str, ...]:
        """Return canonical family IDs aligned to matrix columns."""

        return tuple(family.family_id for family in self.family_definitions)

    @property
    def driver_ids(self) -> tuple[str, ...]:
        """Return the canonical member universe represented by the basis."""

        return tuple(
            sorted(
                driver
                for family in self.family_definitions
                for driver in family.driver_ids
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return manifest-ready family basis provenance."""

        return {
            **self.method.to_dict(),
            "family_basis_id": self.family_basis_id,
            "source_basis_id": self.source_basis_id,
            "strict_cosine_threshold": self.strict_cosine_threshold,
            "family_ids": list(self.family_ids),
            "medoid_driver_ids": list(self.medoid_driver_ids),
            "family_eligible": self.family_eligible.tolist(),
        }


@dataclass(frozen=True, slots=True)
class FamilyMemberEvidence:
    """Response-independent evidence used only for within-family allocation."""

    driver_id: str
    receptor_availability: float | None
    ligand_availability: float | None
    prior_quality: float | None
    subject_prevalence: float | None
    resource_evidence: float | None

    def __post_init__(self) -> None:
        if not self.driver_id:
            raise ContractError(
                "Family member evidence requires a driver ID",
                code="invalid_family_member_evidence",
                field="driver_id",
                remediation="Align evidence to one canonical family member",
            )
        for field_name in self.component_names:
            raw = getattr(self, field_name)
            if raw is None:
                continue
            if isinstance(raw, bool):
                raise ContractError(
                    f"{field_name} must be numeric or missing, not boolean",
                    code="invalid_family_member_evidence",
                    field=field_name,
                    remediation="Provide explicit unit-interval member evidence",
                )
            value = float(raw)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ContractError(
                    f"{field_name} must be missing or lie in [0, 1]",
                    code="invalid_family_member_evidence",
                    field=field_name,
                    remediation="Validate independent allocation evidence",
                )
            object.__setattr__(self, field_name, value)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return the fixed evidence components used by allocation v1."""

        return (
            "receptor_availability",
            "ligand_availability",
            "prior_quality",
            "subject_prevalence",
            "resource_evidence",
        )

    @property
    def all_missing(self) -> bool:
        """Whether every allocation component is missing."""

        return all(getattr(self, name) is None for name in self.component_names)

    @property
    def complete(self) -> bool:
        """Whether every allocation component is observed."""

        return all(getattr(self, name) is not None for name in self.component_names)

    @property
    def evidence_score(self) -> float | None:
        """Return the complete-case multiplicative score, otherwise missing."""

        if not self.complete:
            return None
        return float(
            math.prod(float(getattr(self, name)) for name in self.component_names)
        )

    def to_dict(self) -> dict[str, object]:
        """Return canonical evidence provenance."""

        return {
            "driver_id": self.driver_id,
            **{name: getattr(self, name) for name in self.component_names},
        }


@dataclass(frozen=True, slots=True)
class FamilyFirstAttributionResult:
    """Family-grain positive-channel attribution without member coefficients."""

    attribution_id: str
    family_basis_id: str
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    coefficients: np.ndarray
    contributions: np.ndarray
    signed_response: np.ndarray
    positive_response: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    precision_weights: np.ndarray
    diagnostics: SolverDiagnostics
    lambda1: float
    lambda2: float
    response_channel: str = "positive"
    experimental: bool = True

    def __post_init__(self) -> None:
        if not self.attribution_id or not self.family_basis_id:
            raise ContractError(
                "Family-first attribution IDs must be non-empty",
                code="invalid_family_first_attribution",
                field="attribution_id",
                remediation="Retain the fitted family basis identity",
            )
        if tuple(sorted(set(self.family_ids))) != self.family_ids:
            raise ContractError(
                "Family-first result family IDs must be unique and sorted",
                code="invalid_family_first_attribution",
                field="family_ids",
                remediation="Preserve canonical family basis column order",
            )
        coefficients = _readonly_vector(self.coefficients)
        contributions = _readonly_vector(self.contributions)
        if coefficients.shape != (len(self.family_ids),) or contributions.shape != (
            len(self.family_ids),
        ):
            raise ContractError(
                "Family coefficients and contributions must align with family IDs",
                code="invalid_family_first_attribution",
                field="coefficients",
                remediation="Emit one fitted value per family solver column",
            )
        if (
            np.any(~np.isfinite(coefficients))
            or np.any(coefficients < 0)
            or np.any(~np.isfinite(contributions))
            or np.any(contributions < 0)
        ):
            raise ContractError(
                "Family coefficients and contributions must be finite and non-negative",
                code="invalid_family_first_attribution",
                field="coefficients",
                remediation="Use the unmodified non-negative family solve",
            )
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
        expected = (len(self.feature_ids),)
        if any(vector.shape != expected for vector in vectors.values()):
            raise ContractError(
                "Family-first response vectors must align with feature IDs",
                code="invalid_family_first_attribution",
                field="signed_response",
                remediation="Align response, prediction, residual, and precision",
            )
        if (
            any(np.any(~np.isfinite(vector)) for vector in vectors.values())
            or np.any(vectors["precision_weights"] < 0)
            or not np.any(vectors["precision_weights"] > 0)
        ):
            raise ContractError(
                "Family-first response and precision vectors must be finite",
                code="invalid_family_first_attribution",
                field="precision_weights",
                remediation="Use supported response features and finite precision",
            )
        if not np.array_equal(
            vectors["positive_response"],
            np.maximum(vectors["signed_response"], 0.0),
        ) or np.any(vectors["predicted"] < -1e-12):
            raise ContractError(
                "Family-first attribution must fit the non-negative response channel",
                code="invalid_family_first_attribution",
                field="positive_response",
                remediation="Fit max(signed_response, 0) with a non-negative basis",
            )
        if not np.allclose(
            vectors["signed_response"],
            vectors["predicted"] + vectors["residual"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ContractError(
                "Family prediction plus residual must reconstruct signed response",
                code="invalid_family_first_attribution",
                field="residual",
                remediation="Preserve the complete signed receiver response",
            )
        for field_name in ("lambda1", "lambda2"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ContractError(
                    "Family-first penalties must be finite and non-negative",
                    code="invalid_family_first_attribution",
                    field=field_name,
                    remediation="Use pre-registered non-negative penalties",
                )
            object.__setattr__(self, field_name, value)
        if self.response_channel != "positive" or not self.experimental:
            raise ContractError(
                "Family-first attribution must remain positive-channel "
                "and experimental",
                code="invalid_family_first_attribution",
                field="experimental",
                remediation="Keep the legacy attribution path as the default",
            )
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "contributions", contributions)
        for name, vector in vectors.items():
            object.__setattr__(self, name, vector)

    @property
    def succeeded(self) -> bool:
        """Return true only for an explicitly converged family solve."""

        return self.diagnostics.converged


@dataclass(frozen=True, slots=True)
class FamilyAllocationSummary:
    """Family coefficient, contribution, entropy, and identifiability status."""

    family_id: str
    family_coefficient: float
    family_contribution: float
    within_family_entropy: float | None
    identifiability_status: LRIdentifiabilityStatus
    reason_code: str | None

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ContractError(
                "Family allocation summary requires a family ID",
                code="invalid_family_allocation",
                field="family_id",
                remediation="Align allocation to the fitted family result",
            )
        for field_name in ("family_coefficient", "family_contribution"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ContractError(
                    "Family allocation values must be finite and non-negative",
                    code="invalid_family_allocation",
                    field=field_name,
                    remediation="Use the fitted family coefficient and contribution",
                )
            object.__setattr__(self, field_name, value)
        status = LRIdentifiabilityStatus(self.identifiability_status)
        entropy = self.within_family_entropy
        if status is LRIdentifiabilityStatus.RESOLVED:
            if (
                entropy is None
                or not math.isfinite(entropy)
                or not 0 <= entropy <= 1
                or self.reason_code is not None
            ):
                raise ContractError(
                    "Resolved family allocation requires bounded entropy and no reason",
                    code="invalid_family_allocation",
                    field="within_family_entropy",
                    remediation="Compute entropy from resolved member weights",
                )
            entropy = float(entropy)
        elif entropy is not None or not self.reason_code:
            raise ContractError(
                "Unresolved family allocation requires NA entropy and a reason",
                code="invalid_family_allocation",
                field="reason_code",
                remediation="Do not synthesize weights for unresolved evidence",
            )
        object.__setattr__(self, "identifiability_status", status)
        object.__setattr__(self, "within_family_entropy", entropy)


@dataclass(frozen=True, slots=True)
class FamilyMemberAllocation:
    """One evidence-only member weight and conserved family contribution share."""

    family_id: str
    driver_id: str
    evidence_score: float | None
    within_family_weight: float | None
    member_contribution: float | None
    identifiability_status: LRIdentifiabilityStatus
    reason_code: str | None

    def __post_init__(self) -> None:
        if not self.family_id or not self.driver_id:
            raise ContractError(
                "Member allocation requires family and driver IDs",
                code="invalid_family_allocation",
                field="driver_id",
                remediation="Align each allocation row to one family member",
            )
        score = self.evidence_score
        if score is not None:
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ContractError(
                    "Member evidence score must be missing or lie in [0, 1]",
                    code="invalid_family_allocation",
                    field="evidence_score",
                    remediation="Use complete multiplicative unit-interval evidence",
                )
        status = LRIdentifiabilityStatus(self.identifiability_status)
        weight = self.within_family_weight
        contribution = self.member_contribution
        if status is LRIdentifiabilityStatus.RESOLVED:
            if weight is None or contribution is None or self.reason_code is not None:
                raise ContractError(
                    "Resolved member allocation requires weight and contribution",
                    code="invalid_family_allocation",
                    field="within_family_weight",
                    remediation="Allocate the complete family contribution",
                )
            weight = float(weight)
            contribution = float(contribution)
            if (
                not math.isfinite(weight)
                or not 0 <= weight <= 1
                or not math.isfinite(contribution)
                or contribution < 0
            ):
                raise ContractError(
                    "Resolved member weights and contributions must be bounded",
                    code="invalid_family_allocation",
                    field="within_family_weight",
                    remediation="Normalize non-negative complete evidence",
                )
        elif weight is not None or contribution is not None or not self.reason_code:
            raise ContractError(
                "Unresolved member allocation requires NA weight and contribution",
                code="invalid_family_allocation",
                field="within_family_weight",
                remediation=(
                    "Do not replace missing member evidence with uniform weights"
                ),
            )
        object.__setattr__(self, "evidence_score", score)
        object.__setattr__(self, "within_family_weight", weight)
        object.__setattr__(self, "member_contribution", contribution)
        object.__setattr__(self, "identifiability_status", status)


@dataclass(frozen=True, slots=True)
class FamilyFirstAllocationResult:
    """Auditable family and member allocation with exact conservation."""

    allocation_id: str
    family_basis_id: str
    attribution_id: str
    family_summaries: tuple[FamilyAllocationSummary, ...]
    member_allocations: tuple[FamilyMemberAllocation, ...]
    method: FamilyMemberAllocationMethod = (
        FamilyMemberAllocationMethod.COMPLETE_MULTIPLICATIVE_EVIDENCE_V1
    )
    experimental: bool = True

    def __post_init__(self) -> None:
        if (
            not self.allocation_id
            or not self.family_basis_id
            or not self.attribution_id
        ):
            raise ContractError(
                "Family allocation IDs must be non-empty",
                code="invalid_family_allocation",
                field="allocation_id",
                remediation="Retain basis, fit, and allocation identities",
            )
        summaries = tuple(self.family_summaries)
        members = tuple(self.member_allocations)
        if (
            not summaries
            or tuple(sorted(summaries, key=lambda row: row.family_id)) != summaries
        ):
            raise ContractError(
                "Family allocation summaries must be non-empty and sorted",
                code="invalid_family_allocation",
                field="family_summaries",
                remediation="Emit one canonical summary per fitted family",
            )
        if (
            tuple(sorted(members, key=lambda row: (row.family_id, row.driver_id)))
            != members
        ):
            raise ContractError(
                "Member allocations must be sorted by family and driver",
                code="invalid_family_allocation",
                field="member_allocations",
                remediation="Canonicalize member allocation output order",
            )
        summary_ids = tuple(summary.family_id for summary in summaries)
        if len(set(summary_ids)) != len(summary_ids):
            raise ContractError(
                "Family allocation summaries must have unique family IDs",
                code="invalid_family_allocation",
                field="family_summaries",
                remediation="Emit each fitted family exactly once",
            )
        member_keys = [(row.family_id, row.driver_id) for row in members]
        if len(set(member_keys)) != len(member_keys) or {
            row.family_id for row in members
        } != set(summary_ids):
            raise ContractError(
                "Member allocations must uniquely cover every summarized family",
                code="invalid_family_allocation",
                field="member_allocations",
                remediation="Emit one allocation row per strict family member",
            )
        for summary in summaries:
            family_members = [
                row for row in members if row.family_id == summary.family_id
            ]
            if any(
                row.identifiability_status is not summary.identifiability_status
                or row.reason_code != summary.reason_code
                for row in family_members
            ):
                raise ContractError(
                    "Family and member allocation statuses must agree",
                    code="invalid_family_allocation",
                    field="identifiability_status",
                    remediation="Propagate one family-level resolution decision",
                )
            if summary.identifiability_status is LRIdentifiabilityStatus.RESOLVED:
                weights: list[float] = []
                contributions: list[float] = []
                for row in family_members:
                    assert row.within_family_weight is not None
                    assert row.member_contribution is not None
                    weights.append(float(row.within_family_weight))
                    contributions.append(float(row.member_contribution))
                if not math.isclose(sum(weights), 1.0, rel_tol=1e-10, abs_tol=1e-12):
                    raise ContractError(
                        "Resolved within-family weights must sum to one",
                        code="invalid_family_allocation",
                        field="within_family_weight",
                        remediation="Normalize complete non-negative member evidence",
                    )
                if not math.isclose(
                    sum(contributions),
                    summary.family_contribution,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise ContractError(
                        "Member contributions must conserve family contribution",
                        code="invalid_family_allocation",
                        field="member_contribution",
                        remediation="Multiply conserved weights by family contribution",
                    )
        method = FamilyMemberAllocationMethod(self.method)
        if not self.experimental:
            raise ContractError(
                "Family member allocation remains experimental",
                code="invalid_family_allocation",
                field="experimental",
                remediation="Keep individual LR allocation opt-in",
            )
        object.__setattr__(self, "family_summaries", summaries)
        object.__setattr__(self, "member_allocations", members)
        object.__setattr__(self, "method", method)

    def to_dict(self) -> dict[str, object]:
        """Return manifest-ready allocation method provenance."""

        return {
            **self.method.to_dict(),
            "allocation_id": self.allocation_id,
            "family_basis_id": self.family_basis_id,
            "attribution_id": self.attribution_id,
        }


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
