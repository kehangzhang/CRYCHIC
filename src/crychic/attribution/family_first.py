"""Opt-in family-first receiver attribution and evidence-only LR allocation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse

from crychic.core import ContractError, stable_id
from crychic.resources import TargetPrior

from .basis import build_gated_target_basis
from .contracts import (
    DriverFamilyDefinition,
    FamilyAllocationSummary,
    FamilyBasisMethod,
    FamilyFirstAllocationResult,
    FamilyFirstAttributionResult,
    FamilyFirstBasis,
    FamilyMemberAllocation,
    FamilyMemberAllocationMethod,
    FamilyMemberEvidence,
    GatedTargetBasis,
    LRIdentifiabilityStatus,
    ReceptorGatePolicy,
)
from .families import cluster_driver_families
from .solver import solve_nonnegative_elastic_net

_COSINE_TOLERANCE = 1e-12


def _canonical_families(
    source_basis: GatedTargetBasis,
    families: Sequence[DriverFamilyDefinition],
) -> tuple[DriverFamilyDefinition, ...]:
    definitions = tuple(families)
    if not definitions or any(
        not isinstance(family, DriverFamilyDefinition) for family in definitions
    ):
        raise ContractError(
            "Family-first attribution requires strict DriverFamilyDefinition values",
            code="invalid_family_partition",
            field="families",
            remediation="Cluster the source basis into strict complete-link families",
        )
    canonical = tuple(sorted(definitions, key=lambda family: family.family_id))
    members = [driver for family in canonical for driver in family.driver_ids]
    member_counts: dict[str, int] = {}
    for driver in members:
        member_counts[driver] = member_counts.get(driver, 0) + 1
    observed = set(members)
    expected = set(source_basis.driver_ids)
    duplicates = sorted(driver for driver, count in member_counts.items() if count > 1)
    missing = sorted(expected.difference(observed))
    unknown = sorted(observed.difference(expected))
    if duplicates or missing or unknown:
        raise ContractError(
            "Strict families must partition the complete source driver universe",
            code="invalid_family_partition",
            field="families",
            remediation=(
                f"Resolve duplicates={duplicates!r}, missing={missing!r}, "
                f"unknown={unknown!r}"
            ),
        )
    return canonical


def _family_medoid(
    source_basis: GatedTargetBasis,
    family: DriverFamilyDefinition,
    *,
    driver_index: dict[str, int],
    strict_cosine_threshold: float,
) -> tuple[str, sparse.csc_matrix, float]:
    indices = [driver_index[driver] for driver in family.driver_ids]
    profiles = source_basis.normalized_profiles[:, indices].tocsc()
    cosine = np.asarray((profiles.T @ profiles).toarray(), dtype=float)
    cosine = np.clip(cosine, 0.0, 1.0)
    if len(indices) > 1:
        off_diagonal = cosine[np.triu_indices(len(indices), k=1)]
        minimum = float(off_diagonal.min())
        if minimum + _COSINE_TOLERANCE < strict_cosine_threshold:
            raise ContractError(
                f"Family {family.family_id!r} violates strict pairwise cosine",
                code="non_strict_driver_family",
                field="families",
                remediation="Split chained or weakly similar family members",
            )
    mean_cosine = cosine.mean(axis=1)
    best = float(mean_cosine.max())
    medoid_offset = next(
        offset
        for offset, value in enumerate(mean_cosine)
        if best - float(value) <= _COSINE_TOLERANCE
    )
    medoid = family.driver_ids[medoid_offset]
    profile = source_basis.normalized_profiles.getcol(driver_index[medoid]).tocsc()
    norm = math.sqrt(float(profile.power(2).sum()))
    if norm > 0:
        profile = (profile / norm).tocsc()
    return medoid, profile, norm


def build_family_first_basis(
    source_basis: GatedTargetBasis,
    families: Sequence[DriverFamilyDefinition],
    *,
    strict_cosine_threshold: float = 0.95,
    method: FamilyBasisMethod | str = FamilyBasisMethod.STRICT_MEDOID_V1,
) -> FamilyFirstBasis:
    """Build one deterministic normalized medoid column per strict family."""

    if not isinstance(source_basis, GatedTargetBasis):
        raise TypeError("source_basis must be a GatedTargetBasis")
    if (
        isinstance(strict_cosine_threshold, bool)
        or not math.isfinite(strict_cosine_threshold)
        or not 0 <= strict_cosine_threshold <= 1
    ):
        raise ContractError(
            "strict_cosine_threshold must be finite and lie in [0, 1]",
            code="invalid_family_threshold",
            field="strict_cosine_threshold",
            remediation="Use the threshold that produced the strict families",
        )
    selected_method = FamilyBasisMethod(method)
    definitions = _canonical_families(source_basis, families)
    driver_index = {
        driver: index for index, driver in enumerate(source_basis.driver_ids)
    }
    medoids: list[str] = []
    columns: list[sparse.csc_matrix] = []
    family_eligible: list[bool] = []
    for family in definitions:
        medoid, profile, norm = _family_medoid(
            source_basis,
            family,
            driver_index=driver_index,
            strict_cosine_threshold=strict_cosine_threshold,
        )
        eligible = norm > 0 and any(
            bool(source_basis.receptor_eligible[driver_index[driver]])
            for driver in family.driver_ids
        )
        medoids.append(medoid)
        family_eligible.append(eligible)
        columns.append(
            profile
            if eligible
            else sparse.csc_matrix((len(source_basis.feature_ids), 1), dtype=float)
        )
    matrix = sparse.hstack(columns, format="csc")
    eligible_mask = np.asarray(family_eligible, dtype=bool)
    family_basis_id = stable_id(
        "family_first_basis",
        {
            "source_basis_id": source_basis.basis_id,
            "method": selected_method.value,
            "method_version": selected_method.version,
            "strict_cosine_threshold": float(strict_cosine_threshold),
            "families": [
                {
                    "family_id": family.family_id,
                    "driver_ids": family.driver_ids,
                    "medoid_driver_id": medoid,
                    "eligible": bool(eligible),
                }
                for family, medoid, eligible in zip(
                    definitions, medoids, eligible_mask, strict=True
                )
            ],
        },
    )
    return FamilyFirstBasis(
        family_basis_id=family_basis_id,
        source_basis_id=source_basis.basis_id,
        feature_ids=source_basis.feature_ids,
        family_definitions=definitions,
        matrix=matrix,
        medoid_driver_ids=tuple(medoids),
        family_eligible=eligible_mask,
        strict_cosine_threshold=float(strict_cosine_threshold),
        method=selected_method,
    )


def fit_family_first_attribution(
    basis: FamilyFirstBasis,
    signed_response: np.ndarray,
    *,
    precision_weights: np.ndarray | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    tolerance: float = 1e-8,
    kkt_tolerance: float | None = None,
    max_iterations: int = 10_000,
) -> FamilyFirstAttributionResult:
    """Fit non-negative coefficients directly at strict-family grain."""

    if not isinstance(basis, FamilyFirstBasis):
        raise TypeError("basis must be a FamilyFirstBasis")
    signed = np.asarray(signed_response, dtype=float)
    if signed.shape != (len(basis.feature_ids),) or np.any(~np.isfinite(signed)):
        raise ContractError(
            "Signed response must be finite and align with family basis features",
            code="invalid_family_response",
            field="signed_response",
            remediation="Align one complete response value per family-basis feature",
        )
    positive = np.maximum(signed, 0.0)
    precision = (
        np.ones(len(signed), dtype=float)
        if precision_weights is None
        else np.asarray(precision_weights, dtype=float)
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
    column_norms = np.sqrt(np.asarray(basis.matrix.power(2).sum(axis=0)).ravel())
    contributions = solution.coefficients * column_norms
    attribution_id = stable_id(
        "family_first_attribution",
        {
            "family_basis_id": basis.family_basis_id,
            "family_ids": basis.family_ids,
            "coefficients": solution.coefficients.tolist(),
            "signed_response": signed.tolist(),
            "precision_weights": precision.tolist(),
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "tolerance": float(tolerance),
            "kkt_tolerance": kkt_tolerance,
            "max_iterations": max_iterations,
        },
    )
    return FamilyFirstAttributionResult(
        attribution_id=attribution_id,
        family_basis_id=basis.family_basis_id,
        feature_ids=basis.feature_ids,
        family_ids=basis.family_ids,
        coefficients=solution.coefficients,
        contributions=contributions,
        signed_response=signed,
        positive_response=positive,
        predicted=solution.predicted,
        residual=signed - solution.predicted,
        precision_weights=precision,
        diagnostics=solution.diagnostics,
        lambda1=lambda1,
        lambda2=lambda2,
    )


def attribute_target_prior_family_first(
    prior: TargetPrior,
    feature_ids: Sequence[str],
    receptor_gates: Mapping[str, float],
    signed_response: np.ndarray,
    *,
    precision_weights: np.ndarray | None = None,
    receptor_gate_threshold: float = 0.1,
    cosine_threshold: float = 0.95,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    tolerance: float = 1e-8,
    kkt_tolerance: float | None = None,
    max_iterations: int = 10_000,
) -> tuple[GatedTargetBasis, FamilyFirstBasis, FamilyFirstAttributionResult]:
    """Build and fit the default family-grain attribution path.

    This is the family-first counterpart of :func:`attribute_target_prior`.
    Receptor evidence is used only as a hard eligibility decision, so changing
    an already eligible gate cannot rescale a family coefficient. Families are
    strict complete-link equivalence classes and only one medoid column per
    family enters the solver. Individual LR allocation remains a separate,
    evidence-only step via :func:`allocate_family_members`.
    """

    source_basis = build_gated_target_basis(
        prior,
        feature_ids,
        receptor_gates,
        gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
        receptor_gate_threshold=receptor_gate_threshold,
    )
    families = cluster_driver_families(
        source_basis,
        cosine_threshold=cosine_threshold,
    )
    family_basis = build_family_first_basis(
        source_basis,
        families,
        strict_cosine_threshold=cosine_threshold,
    )
    attribution = fit_family_first_attribution(
        family_basis,
        signed_response,
        precision_weights=precision_weights,
        lambda1=lambda1,
        lambda2=lambda2,
        tolerance=tolerance,
        kkt_tolerance=kkt_tolerance,
        max_iterations=max_iterations,
    )
    return source_basis, family_basis, attribution


def _normalized_entropy(weights: np.ndarray) -> float:
    if len(weights) <= 1:
        return 0.0
    positive = weights[weights > 0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(len(weights))
    return min(1.0, max(0.0, entropy))


def allocate_family_members(
    basis: FamilyFirstBasis,
    attribution: FamilyFirstAttributionResult,
    member_evidence: Sequence[FamilyMemberEvidence],
    *,
    method: FamilyMemberAllocationMethod | str = (
        FamilyMemberAllocationMethod.COMPLETE_MULTIPLICATIVE_EVIDENCE_V1
    ),
) -> FamilyFirstAllocationResult:
    """Conservatively allocate family contribution using independent evidence."""

    if not isinstance(basis, FamilyFirstBasis):
        raise TypeError("basis must be a FamilyFirstBasis")
    if not isinstance(attribution, FamilyFirstAttributionResult):
        raise TypeError("attribution must be a FamilyFirstAttributionResult")
    if basis.family_basis_id != attribution.family_basis_id or (
        basis.family_ids != attribution.family_ids
    ):
        raise ContractError(
            "Member allocation requires the exact fitted family basis",
            code="family_allocation_basis_mismatch",
            field="family_basis_id",
            remediation="Pair allocation with its family-level attribution result",
        )
    evidence = tuple(member_evidence)
    if any(not isinstance(item, FamilyMemberEvidence) for item in evidence):
        raise TypeError("member_evidence must contain FamilyMemberEvidence values")
    evidence_by_driver = {item.driver_id: item for item in evidence}
    observed = {item.driver_id for item in evidence}
    expected = set(basis.driver_ids)
    if len(evidence_by_driver) != len(evidence) or observed != expected:
        raise ContractError(
            "Member evidence must cover every family member exactly once",
            code="family_member_evidence_mismatch",
            field="member_evidence",
            remediation=(
                f"Resolve missing={sorted(expected - observed)!r}, "
                f"unknown={sorted(observed - expected)!r}, or duplicates"
            ),
        )
    selected_method = FamilyMemberAllocationMethod(method)
    coefficient_by_family = dict(
        zip(basis.family_ids, attribution.coefficients, strict=True)
    )
    contribution_by_family = dict(
        zip(basis.family_ids, attribution.contributions, strict=True)
    )
    summaries: list[FamilyAllocationSummary] = []
    allocations: list[FamilyMemberAllocation] = []
    for family in basis.family_definitions:
        family_evidence = tuple(
            evidence_by_driver[driver] for driver in family.driver_ids
        )
        if all(item.all_missing for item in family_evidence):
            status = LRIdentifiabilityStatus.UNRESOLVED
            reason_code = "all_member_evidence_missing"
            weights: np.ndarray | None = None
        elif not all(item.complete for item in family_evidence):
            status = LRIdentifiabilityStatus.UNRESOLVED
            reason_code = "incomplete_member_evidence"
            weights = None
        else:
            score_values: list[float] = []
            for item in family_evidence:
                score = item.evidence_score
                assert score is not None
                score_values.append(float(score))
            scores = np.asarray(score_values, dtype=float)
            total = float(scores.sum())
            if total <= 0:
                status = LRIdentifiabilityStatus.UNRESOLVED
                reason_code = "no_positive_member_evidence"
                weights = None
            else:
                status = LRIdentifiabilityStatus.RESOLVED
                reason_code = None
                weights = scores / total
        family_coefficient = float(coefficient_by_family[family.family_id])
        family_contribution = float(contribution_by_family[family.family_id])
        entropy = None if weights is None else _normalized_entropy(weights)
        summaries.append(
            FamilyAllocationSummary(
                family_id=family.family_id,
                family_coefficient=family_coefficient,
                family_contribution=family_contribution,
                within_family_entropy=entropy,
                identifiability_status=status,
                reason_code=reason_code,
            )
        )
        for offset, item in enumerate(family_evidence):
            weight = None if weights is None else float(weights[offset])
            allocations.append(
                FamilyMemberAllocation(
                    family_id=family.family_id,
                    driver_id=item.driver_id,
                    evidence_score=item.evidence_score,
                    within_family_weight=weight,
                    member_contribution=(
                        None if weight is None else family_contribution * weight
                    ),
                    identifiability_status=status,
                    reason_code=reason_code,
                )
            )
    canonical_evidence = tuple(sorted(evidence, key=lambda item: item.driver_id))
    allocation_id = stable_id(
        "family_member_allocation",
        {
            "family_basis_id": basis.family_basis_id,
            "attribution_id": attribution.attribution_id,
            "method": selected_method.value,
            "method_version": selected_method.version,
            "member_evidence": [item.to_dict() for item in canonical_evidence],
        },
    )
    return FamilyFirstAllocationResult(
        allocation_id=allocation_id,
        family_basis_id=basis.family_basis_id,
        attribution_id=attribution.attribution_id,
        family_summaries=tuple(sorted(summaries, key=lambda row: row.family_id)),
        member_allocations=tuple(
            sorted(allocations, key=lambda row: (row.family_id, row.driver_id))
        ),
        method=selected_method,
    )
