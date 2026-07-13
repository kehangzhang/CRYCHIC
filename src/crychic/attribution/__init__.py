"""Experimental direction-compatible receiver attribution."""

from .basis import build_gated_target_basis
from .contracts import (
    AttributionResult,
    AttributionSupportMethod,
    BasisBuildReport,
    DownstreamAttributionSupport,
    DriverEstimate,
    DriverFamilyDefinition,
    FamilyAllocationSummary,
    FamilyBasisMethod,
    FamilyEstimate,
    FamilyFirstAllocationResult,
    FamilyFirstAttributionResult,
    FamilyFirstBasis,
    FamilyMemberAllocation,
    FamilyMemberAllocationMethod,
    FamilyMemberEvidence,
    GatedTargetBasis,
    LRIdentifiabilityStatus,
    ObjectiveTerms,
    ReceptorGatePolicy,
    SolverDiagnostics,
    SolverStatus,
)
from .directional import (
    DirectionalResponseChannel,
    ResponseDirection,
    directional_response_channel,
)
from .families import cluster_driver_families
from .family_first import (
    allocate_family_members,
    attribute_target_prior_family_first,
    build_family_first_basis,
    fit_family_first_attribution,
)
from .model import (
    attribute_target_prior,
    downstream_attribution_support,
    fit_positive_attribution,
)
from .precision import PrecisionTransformResult, winsorized_normalized_precision
from .solver import ElasticNetSolution, solve_nonnegative_elastic_net

__all__ = [
    "AttributionResult",
    "AttributionSupportMethod",
    "BasisBuildReport",
    "DirectionalResponseChannel",
    "DownstreamAttributionSupport",
    "DriverEstimate",
    "DriverFamilyDefinition",
    "ElasticNetSolution",
    "FamilyAllocationSummary",
    "FamilyBasisMethod",
    "FamilyEstimate",
    "FamilyFirstAllocationResult",
    "FamilyFirstAttributionResult",
    "FamilyFirstBasis",
    "FamilyMemberAllocation",
    "FamilyMemberAllocationMethod",
    "FamilyMemberEvidence",
    "GatedTargetBasis",
    "LRIdentifiabilityStatus",
    "ObjectiveTerms",
    "PrecisionTransformResult",
    "ReceptorGatePolicy",
    "ResponseDirection",
    "SolverDiagnostics",
    "SolverStatus",
    "allocate_family_members",
    "attribute_target_prior",
    "attribute_target_prior_family_first",
    "build_family_first_basis",
    "build_gated_target_basis",
    "cluster_driver_families",
    "directional_response_channel",
    "downstream_attribution_support",
    "fit_family_first_attribution",
    "fit_positive_attribution",
    "solve_nonnegative_elastic_net",
    "winsorized_normalized_precision",
]
