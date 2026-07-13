from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crychic.attribution import (
    DriverFamilyDefinition,
    FamilyFirstAllocationResult,
    FamilyMemberEvidence,
    LRIdentifiabilityStatus,
    ReceptorGatePolicy,
    allocate_family_members,
    attribute_target_prior_family_first,
    build_family_first_basis,
    build_gated_target_basis,
    cluster_driver_families,
    fit_family_first_attribution,
)
from crychic.core import ContractError


def _evidence(
    driver_id: str,
    *,
    receptor: float | None = 1.0,
    ligand: float | None = 1.0,
    quality: float | None = 1.0,
    prevalence: float | None = 1.0,
    resource: float | None = 1.0,
) -> FamilyMemberEvidence:
    return FamilyMemberEvidence(
        driver_id=driver_id,
        receptor_availability=receptor,
        ligand_availability=ligand,
        prior_quality=quality,
        subject_prevalence=prevalence,
        resource_evidence=resource,
    )


def _family_id_with_driver(basis, driver_id: str) -> str:
    return next(
        family.family_id
        for family in basis.family_definitions
        if driver_id in family.driver_ids
    )


def _family_column(basis, driver_id: str) -> np.ndarray:
    family_id = _family_id_with_driver(basis, driver_id)
    column = basis.family_ids.index(family_id)
    return np.asarray(basis.matrix.getcol(column).toarray()).ravel()


def _family_value(basis, values: np.ndarray, driver_id: str) -> float:
    family_id = _family_id_with_driver(basis, driver_id)
    return float(values[basis.family_ids.index(family_id)])


def test_family_first_fits_one_medoid_column_per_strict_family(prior_factory) -> None:
    prior = prior_factory(
        {
            "A": {"G1": 1.0},
            "B": {"G1": 4.0},
            "C": {"G2": 1.0},
        }
    )
    source = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"A": 1.0, "B": 0.5, "C": 1.0},
        gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
        receptor_gate_threshold=0.1,
    )
    families = cluster_driver_families(source, cosine_threshold=0.999)

    basis = build_family_first_basis(source, families, strict_cosine_threshold=0.999)
    result = fit_family_first_attribution(
        basis,
        np.asarray([2.0, 1.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    assert basis.matrix.shape == (2, 2)
    assert set(basis.medoid_driver_ids) == {"A", "C"}
    np.testing.assert_array_equal(_family_column(basis, "A"), [1.0, 0.0])
    assert _family_value(basis, result.coefficients, "A") == pytest.approx(2.0)
    assert _family_value(basis, result.coefficients, "C") == pytest.approx(1.0)
    assert _family_value(basis, result.contributions, "A") == pytest.approx(2.0)
    assert result.succeeded
    assert not hasattr(result, "driver_estimates")


def test_adding_collinear_members_does_not_change_family_core(prior_factory) -> None:
    single_prior = prior_factory({"A": {"G1": 1.0}, "C": {"G2": 1.0}})
    expanded_prior = prior_factory(
        {
            "A": {"G1": 1.0},
            "B": {"G1": 10.0},
            "C": {"G2": 1.0},
        }
    )
    single_source = build_gated_target_basis(
        single_prior, ("G1", "G2"), {"A": 1.0, "C": 1.0}
    )
    expanded_source = build_gated_target_basis(
        expanded_prior,
        ("G1", "G2"),
        {"A": 1.0, "B": 0.2, "C": 1.0},
    )
    single = build_family_first_basis(
        single_source,
        cluster_driver_families(single_source, cosine_threshold=0.999),
        strict_cosine_threshold=0.999,
    )
    expanded = build_family_first_basis(
        expanded_source,
        cluster_driver_families(expanded_source, cosine_threshold=0.999),
        strict_cosine_threshold=0.999,
    )

    np.testing.assert_array_equal(
        _family_column(single, "A"), _family_column(expanded, "A")
    )
    single_fit = fit_family_first_attribution(
        single,
        np.asarray([3.0, 1.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    expanded_fit = fit_family_first_attribution(
        expanded,
        np.asarray([3.0, 1.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    assert _family_value(single, single_fit.coefficients, "A") == pytest.approx(
        _family_value(expanded, expanded_fit.coefficients, "A")
    )
    assert _family_value(single, single_fit.contributions, "A") == pytest.approx(
        _family_value(expanded, expanded_fit.contributions, "A")
    )


def test_high_level_family_first_path_uses_hard_gate_and_preserves_signed_residual(
    prior_factory,
) -> None:
    prior = prior_factory(
        {
            "A": {"G1": 1.0},
            "B": {"G1": 4.0},
            "C": {"G2": 1.0},
        }
    )

    low_source, low_basis, low_fit = attribute_target_prior_family_first(
        prior,
        ("G1", "G2"),
        {"A": 0.2, "B": 0.3, "C": 0.4},
        np.asarray([2.0, -1.0]),
        receptor_gate_threshold=0.1,
        cosine_threshold=0.999,
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    high_source, high_basis, high_fit = attribute_target_prior_family_first(
        prior,
        ("G1", "G2"),
        {"A": 0.8, "B": 0.9, "C": 1.0},
        np.asarray([2.0, -1.0]),
        receptor_gate_threshold=0.1,
        cosine_threshold=0.999,
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    assert low_source.gate_policy is ReceptorGatePolicy.HARD_ELIGIBILITY_V2
    assert low_source.receptor_gate_threshold == pytest.approx(0.1)
    np.testing.assert_array_equal(
        low_source.matrix.toarray(), high_source.matrix.toarray()
    )
    assert low_basis.family_ids == high_basis.family_ids
    np.testing.assert_array_equal(low_fit.coefficients, high_fit.coefficients)
    np.testing.assert_array_equal(
        low_fit.predicted + low_fit.residual,
        low_fit.signed_response,
    )
    assert low_fit.residual[1] == pytest.approx(-1.0)


def test_high_level_family_first_path_excludes_subthreshold_family(
    prior_factory,
) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G2": 1.0}})

    source, basis, fit = attribute_target_prior_family_first(
        prior,
        ("G1", "G2"),
        {"A": 0.09, "B": 0.8},
        np.asarray([3.0, 2.0]),
        receptor_gate_threshold=0.1,
        cosine_threshold=0.999,
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    assert source.receptor_eligible.tolist() == [False, True]
    assert _family_value(basis, fit.coefficients, "A") == pytest.approx(0.0)
    assert _family_value(basis, fit.coefficients, "B") == pytest.approx(2.0)


def test_member_evidence_allocation_conserves_family_contribution_and_entropy(
    prior_factory,
) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G1": 2.0}, "C": {"G2": 1.0}})
    source = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"A": 1.0, "B": 1.0, "C": 1.0},
    )
    basis = build_family_first_basis(
        source,
        cluster_driver_families(source, cosine_threshold=0.999),
        strict_cosine_threshold=0.999,
    )
    fit = fit_family_first_attribution(
        basis,
        np.asarray([3.0, 1.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )

    allocation = allocate_family_members(
        basis,
        fit,
        (
            _evidence("A", ligand=0.8, quality=0.5),
            _evidence("B", receptor=0.5, ligand=0.8, quality=0.5),
            _evidence("C"),
        ),
    )
    family_id = _family_id_with_driver(basis, "A")
    summary = next(
        row for row in allocation.family_summaries if row.family_id == family_id
    )
    members = {
        row.driver_id: row
        for row in allocation.member_allocations
        if row.family_id == family_id
    }

    assert summary.identifiability_status is LRIdentifiabilityStatus.RESOLVED
    assert summary.reason_code is None
    assert members["A"].within_family_weight == pytest.approx(2 / 3)
    assert members["B"].within_family_weight == pytest.approx(1 / 3)
    expected_entropy = -((2 / 3) * np.log(2 / 3) + (1 / 3) * np.log(1 / 3)) / np.log(2)
    assert summary.within_family_entropy == pytest.approx(expected_entropy)
    assert sum(
        float(row.member_contribution) for row in members.values()
    ) == pytest.approx(summary.family_contribution)
    assert summary.family_coefficient == pytest.approx(
        _family_value(basis, fit.coefficients, "A")
    )


@pytest.mark.parametrize("partial", (False, True))
def test_missing_member_evidence_is_unresolved_and_never_uniform(
    prior_factory,
    partial: bool,
) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G1": 2.0}})
    source = build_gated_target_basis(prior, ("G1",), {"A": 1.0, "B": 1.0})
    basis = build_family_first_basis(
        source,
        cluster_driver_families(source, cosine_threshold=0.999),
        strict_cosine_threshold=0.999,
    )
    fit = fit_family_first_attribution(
        basis,
        np.asarray([2.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    missing = _evidence(
        "B",
        receptor=None,
        ligand=None,
        quality=None,
        prevalence=None,
        resource=None,
    )
    first = _evidence("A") if partial else replace(missing, driver_id="A")

    allocation = allocate_family_members(basis, fit, (first, missing))

    summary = allocation.family_summaries[0]
    assert summary.identifiability_status is LRIdentifiabilityStatus.UNRESOLVED
    assert summary.within_family_entropy is None
    assert summary.reason_code == (
        "incomplete_member_evidence" if partial else "all_member_evidence_missing"
    )
    assert all(
        row.within_family_weight is None and row.member_contribution is None
        for row in allocation.member_allocations
    )


def test_family_first_is_deterministic_under_family_driver_and_evidence_order(
    prior_factory,
) -> None:
    first_prior = prior_factory({"C": {"G2": 1.0}, "B": {"G1": 2.0}, "A": {"G1": 1.0}})
    reordered_prior = prior_factory(
        {"A": {"G1": 1.0}, "C": {"G2": 1.0}, "B": {"G1": 2.0}}
    )
    first_source = build_gated_target_basis(
        first_prior,
        ("G1", "G2"),
        {"C": 1.0, "A": 1.0, "B": 1.0},
    )
    reordered_source = build_gated_target_basis(
        reordered_prior,
        ("G1", "G2"),
        {"B": 1.0, "C": 1.0, "A": 1.0},
    )
    first_families = cluster_driver_families(first_source, cosine_threshold=0.999)
    reordered_families = cluster_driver_families(
        reordered_source, cosine_threshold=0.999
    )
    first = build_family_first_basis(
        first_source,
        tuple(reversed(first_families)),
        strict_cosine_threshold=0.999,
    )
    reordered = build_family_first_basis(
        reordered_source,
        reordered_families,
        strict_cosine_threshold=0.999,
    )
    first_fit = fit_family_first_attribution(
        first,
        np.asarray([2.0, 1.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    reordered_fit = fit_family_first_attribution(
        reordered,
        np.asarray([2.0, 1.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    first_allocation = allocate_family_members(
        first,
        first_fit,
        (_evidence("C"), _evidence("A", ligand=0.8), _evidence("B", ligand=0.2)),
    )
    reordered_allocation = allocate_family_members(
        reordered,
        reordered_fit,
        (_evidence("B", ligand=0.2), _evidence("A", ligand=0.8), _evidence("C")),
    )

    assert first.family_basis_id == reordered.family_basis_id
    assert first.medoid_driver_ids == reordered.medoid_driver_ids
    np.testing.assert_array_equal(first.matrix.toarray(), reordered.matrix.toarray())
    assert first_fit.attribution_id == reordered_fit.attribution_id
    assert first_allocation.allocation_id == reordered_allocation.allocation_id
    assert first_allocation.family_summaries == reordered_allocation.family_summaries
    assert (
        first_allocation.member_allocations == reordered_allocation.member_allocations
    )


def test_family_first_rejects_non_strict_or_incomplete_family_definitions(
    prior_factory,
) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G2": 1.0}})
    source = build_gated_target_basis(prior, ("G1", "G2"), {"A": 1.0, "B": 1.0})
    non_strict = DriverFamilyDefinition(
        family_id="family-ab",
        driver_ids=("A", "B"),
        mean_pairwise_cosine=0.0,
        assignment_uncertainty=0.0,
    )
    incomplete = DriverFamilyDefinition(
        family_id="family-a",
        driver_ids=("A",),
        mean_pairwise_cosine=0.0,
        assignment_uncertainty=0.0,
    )

    with pytest.raises(ContractError, match="strict pairwise cosine"):
        build_family_first_basis(source, (non_strict,), strict_cosine_threshold=0.95)
    with pytest.raises(ContractError, match="partition"):
        build_family_first_basis(source, (incomplete,))


def test_allocation_contract_rejects_nonconserved_member_contribution(
    prior_factory,
) -> None:
    prior = prior_factory({"A": {"G1": 1.0}, "B": {"G1": 2.0}})
    source = build_gated_target_basis(prior, ("G1",), {"A": 1.0, "B": 1.0})
    basis = build_family_first_basis(
        source,
        cluster_driver_families(source, cosine_threshold=0.999),
        strict_cosine_threshold=0.999,
    )
    fit = fit_family_first_attribution(
        basis,
        np.asarray([2.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    allocation = allocate_family_members(basis, fit, (_evidence("A"), _evidence("B")))
    corrupted = (
        replace(
            allocation.member_allocations[0],
            member_contribution=float(
                allocation.member_allocations[0].member_contribution
            )
            + 1.0,
        ),
        *allocation.member_allocations[1:],
    )

    with pytest.raises(ContractError, match="conserve"):
        FamilyFirstAllocationResult(
            allocation_id=allocation.allocation_id,
            family_basis_id=allocation.family_basis_id,
            attribution_id=allocation.attribution_id,
            family_summaries=allocation.family_summaries,
            member_allocations=corrupted,
        )
