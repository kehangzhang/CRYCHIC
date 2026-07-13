from __future__ import annotations

import crychic.attribution as attribution


def test_family_first_api_is_explicitly_versioned_and_opt_in() -> None:
    expected = {
        "FamilyBasisMethod",
        "FamilyFirstAllocationResult",
        "FamilyFirstAttributionResult",
        "FamilyFirstBasis",
        "FamilyMemberAllocation",
        "FamilyMemberAllocationMethod",
        "FamilyMemberEvidence",
        "LRIdentifiabilityStatus",
        "allocate_family_members",
        "attribute_target_prior_family_first",
        "build_family_first_basis",
        "fit_family_first_attribution",
    }
    assert expected.issubset(attribution.__all__)
    assert (
        attribution.FamilyBasisMethod.STRICT_MEDOID_V1.value
        == "strict_medoid_family_basis_v1"
    )
    allocation = (
        attribution.FamilyMemberAllocationMethod.COMPLETE_MULTIPLICATIVE_EVIDENCE_V1
    )
    assert allocation.value == "complete_multiplicative_member_evidence_v1"
    assert allocation.to_dict()["uses_receiver_response"] is False
    assert "fit_positive_attribution" in attribution.__all__
