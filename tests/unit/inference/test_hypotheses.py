from __future__ import annotations

from dataclasses import replace

import pytest

from crychic.core import ContractError
from crychic.inference import (
    FrozenHypothesisCoverage,
    FrozenHypothesisUniverse,
    HypothesisCoverageRecord,
    HypothesisCoverageStatus,
    HypothesisDeclaration,
    HypothesisFilterStage,
    HypothesisPrefilterPolicy,
    HypothesisPrefilterStatus,
    HypothesisRole,
    freeze_hypothesis_universe,
    validate_frozen_hypothesis_coverage,
)


def _primary(
    *,
    receiver: str = "Receiver",
    family_id: str = "family-1",
    multiplicity_family: str = "gene-state-primary",
) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint="driver_family_receiver_context_omnibus_v1",
        contrast_name="stim_vs_control",
        receiver=receiver,
        family_id=family_id,
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family=multiplicity_family,
    )


def _secondary(
    parent: HypothesisDeclaration,
    *,
    endpoint: str = "driver_family_receiver_context_posthoc_v1",
    filtered: bool = False,
) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint=endpoint,
        contrast_name="stim_vs_control",
        receiver=parent.receiver,
        family_id=parent.family_id,
        mode=parent.mode,
        role=HypothesisRole.SECONDARY,
        multiplicity_family="gene-state-secondary",
        parent_key=parent.hypothesis_key,
        prefilter_policy=(
            HypothesisPrefilterPolicy.POOLED_CONTEXT_BLIND_SUPPORT
            if filtered
            else HypothesisPrefilterPolicy.NONE
        ),
        prefilter_status=(
            HypothesisPrefilterStatus.FILTERED
            if filtered
            else HypothesisPrefilterStatus.INCLUDED
        ),
        filter_reason_code=("pooled_support_below_threshold" if filtered else None),
    )


def _universe() -> tuple[
    FrozenHypothesisUniverse,
    HypothesisDeclaration,
    HypothesisDeclaration,
    HypothesisDeclaration,
]:
    primary = _primary()
    filtered = _secondary(primary, filtered=True)
    other = _primary(
        receiver="Receiver-2",
        family_id="family-2",
        multiplicity_family="gene-state-primary",
    )
    universe = freeze_hypothesis_universe(
        (primary, filtered, other),
        universe_name="gene-state-v1",
    )
    return universe, primary, filtered, other


def test_universe_is_producer_owned_order_invariant_and_keeps_full_sizes() -> None:
    universe, primary, filtered, other = _universe()
    reordered = freeze_hypothesis_universe(
        (other, filtered, primary),
        universe_name="gene-state-v1",
    )

    assert universe.universe_id == reordered.universe_id
    assert universe.hypothesis_ids == tuple(sorted(universe.hypothesis_ids))
    assert universe.multiplicity_denominator == 3
    assert universe.multiplicity_family_sizes == (
        ("gene-state-primary", 2),
        ("gene-state-secondary", 1),
    )
    assert universe.q_value_release_allowed is False
    assert universe.hierarchical_procedure_status == (
        "frozen_treebh_benjamini_bogomolov_v1_g3_release_required"
    )
    assert universe.declaration_for(primary.hypothesis_id) is primary
    assert universe.to_dict()["q_value_release_allowed"] is False
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenHypothesisUniverse()


def test_hypothesis_identity_binds_scope_role_family_and_parent() -> None:
    primary = _primary()
    secondary = _secondary(primary)
    variants = (
        replace(primary, endpoint="different-endpoint"),
        replace(primary, contrast_name="different-contrast"),
        replace(primary, receiver="different-receiver"),
        replace(primary, family_id="different-family"),
        replace(primary, mode="ecosystem"),
        replace(primary, multiplicity_family="different-multiplicity-family"),
    )

    assert all(item.hypothesis_id != primary.hypothesis_id for item in variants)
    different_parent = _primary(receiver="Other", family_id="other-family")
    rebound = HypothesisDeclaration(
        endpoint=secondary.endpoint,
        contrast_name=secondary.contrast_name,
        receiver=secondary.receiver,
        family_id=secondary.family_id,
        mode=secondary.mode,
        role=secondary.role,
        multiplicity_family=secondary.multiplicity_family,
        parent_key=different_parent.hypothesis_key,
    )
    assert rebound.hypothesis_id != secondary.hypothesis_id


def test_duplicate_and_invalid_hierarchy_fail_closed() -> None:
    primary = _primary()
    secondary = _secondary(primary)
    with pytest.raises(ContractError) as duplicate_error:
        freeze_hypothesis_universe(
            (primary, primary),
            universe_name="duplicates",
        )
    assert duplicate_error.value.details.code == "duplicate_frozen_hypothesis"

    dangling = replace(secondary, parent_key="hypothesis_key_missing")
    with pytest.raises(ContractError) as dangling_error:
        freeze_hypothesis_universe(
            (primary, dangling),
            universe_name="dangling",
        )
    assert dangling_error.value.details.code == "dangling_hypothesis_parent"

    grandchild = replace(
        secondary,
        endpoint="nested-posthoc-endpoint",
        parent_key=secondary.hypothesis_key,
    )
    with pytest.raises(ContractError) as role_error:
        freeze_hypothesis_universe(
            (primary, secondary, grandchild),
            universe_name="secondary-parent",
        )
    assert role_error.value.details.code == "invalid_hypothesis_parent_role"

    mismatched = replace(secondary, receiver="Different")
    with pytest.raises(ContractError) as scope_error:
        freeze_hypothesis_universe(
            (primary, mismatched),
            universe_name="scope-mismatch",
        )
    assert scope_error.value.details.code == "hypothesis_parent_scope_mismatch"


def test_role_and_prefilter_contracts_reject_post_fit_decisions() -> None:
    primary = _primary()
    with pytest.raises(ContractError) as primary_parent_error:
        replace(primary, parent_key=primary.hypothesis_key)
    assert primary_parent_error.value.details.code == (
        "invalid_primary_hypothesis_parent"
    )
    with pytest.raises(ContractError) as secondary_parent_error:
        replace(primary, role=HypothesisRole.SECONDARY)
    assert secondary_parent_error.value.details.code == (
        "secondary_hypothesis_parent_missing"
    )
    with pytest.raises(ContractError) as stage_error:
        replace(primary, filter_stage="post_fit_p_value_filtering_v1")
    assert stage_error.value.details.code == "post_fit_hypothesis_filtering_forbidden"
    with pytest.raises(ContractError) as policy_error:
        replace(primary, prefilter_policy="fitted_effect_threshold_v1")
    assert policy_error.value.details.code == (
        "post_fit_hypothesis_filtering_forbidden"
    )
    with pytest.raises(ValueError, match="cannot have a filter reason"):
        replace(primary, filter_reason_code="unexpected")
    with pytest.raises(ValueError, match="require a reason"):
        replace(
            primary,
            prefilter_policy=HypothesisPrefilterPolicy.EXTERNAL_RESOURCE,
            prefilter_status=HypothesisPrefilterStatus.FILTERED,
        )

    filtered_primary = replace(
        primary,
        prefilter_policy=HypothesisPrefilterPolicy.EXTERNAL_RESOURCE,
        prefilter_status=HypothesisPrefilterStatus.FILTERED,
        filter_reason_code="primary_resource_not_supported",
    )
    included_child = _secondary(filtered_primary)
    with pytest.raises(ContractError) as hierarchy_error:
        freeze_hypothesis_universe(
            (filtered_primary, included_child),
            universe_name="filtered-parent-with-included-child",
        )
    assert hierarchy_error.value.details.code == (
        "hypothesis_prefilter_hierarchy_violation"
    )


def test_exact_coverage_preserves_filtered_rows_and_denominator() -> None:
    universe, primary, filtered, other = _universe()
    records = (
        HypothesisCoverageRecord(
            hypothesis_id=other.hypothesis_id,
            status=HypothesisCoverageStatus.FAILED,
            reason_code="effect_backend_failed",
        ),
        HypothesisCoverageRecord(
            hypothesis_id=filtered.hypothesis_id,
            status=HypothesisCoverageStatus.NOT_ESTIMABLE,
            reason_code=filtered.filter_reason_code,
        ),
        HypothesisCoverageRecord(
            hypothesis_id=primary.hypothesis_id,
            status=HypothesisCoverageStatus.OBSERVED,
        ),
    )
    audit = validate_frozen_hypothesis_coverage(universe, records)
    repeated = validate_frozen_hypothesis_coverage(universe, tuple(reversed(records)))

    assert audit.complete
    assert audit.coverage_id == repeated.coverage_id
    assert (audit.n_observed, audit.n_not_estimable, audit.n_failed) == (1, 1, 1)
    assert audit.multiplicity_family_sizes == universe.multiplicity_family_sizes
    assert sum(count for _, count in audit.multiplicity_family_sizes) == 3
    assert audit.q_value_release_allowed is False
    assert audit.to_dict()["multiplicity_denominator_policy"] == (
        "all_frozen_hypotheses_including_prefiltered_v1"
    )
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenHypothesisCoverage()


def test_missing_extra_and_duplicate_coverage_fail_closed() -> None:
    universe, primary, filtered, other = _universe()
    observed = HypothesisCoverageRecord(
        hypothesis_id=primary.hypothesis_id,
        status=HypothesisCoverageStatus.OBSERVED,
    )
    filtered_record = HypothesisCoverageRecord(
        hypothesis_id=filtered.hypothesis_id,
        status=HypothesisCoverageStatus.NOT_ESTIMABLE,
        reason_code=filtered.filter_reason_code,
    )
    failed = HypothesisCoverageRecord(
        hypothesis_id=other.hypothesis_id,
        status=HypothesisCoverageStatus.FAILED,
        reason_code="failed",
    )

    with pytest.raises(ContractError) as missing_error:
        validate_frozen_hypothesis_coverage(
            universe,
            (observed, filtered_record),
        )
    assert missing_error.value.details.code == "missing_hypothesis_coverage"
    extra = HypothesisCoverageRecord(
        hypothesis_id="hypothesis_extra",
        status=HypothesisCoverageStatus.FAILED,
        reason_code="not-in-universe",
    )
    with pytest.raises(ContractError) as extra_error:
        validate_frozen_hypothesis_coverage(
            universe,
            (observed, filtered_record, failed, extra),
        )
    assert extra_error.value.details.code == "extra_hypothesis_coverage"
    with pytest.raises(ContractError) as duplicate_error:
        validate_frozen_hypothesis_coverage(
            universe,
            (observed, observed, filtered_record, failed),
        )
    assert duplicate_error.value.details.code == "duplicate_hypothesis_coverage"


def test_prefiltered_coverage_requires_exact_ne_reason() -> None:
    universe, primary, filtered, other = _universe()
    common = (
        HypothesisCoverageRecord(
            hypothesis_id=primary.hypothesis_id,
            status=HypothesisCoverageStatus.OBSERVED,
        ),
        HypothesisCoverageRecord(
            hypothesis_id=other.hypothesis_id,
            status=HypothesisCoverageStatus.FAILED,
            reason_code="failed",
        ),
    )
    wrong_status = HypothesisCoverageRecord(
        hypothesis_id=filtered.hypothesis_id,
        status=HypothesisCoverageStatus.OBSERVED,
    )
    wrong_reason = HypothesisCoverageRecord(
        hypothesis_id=filtered.hypothesis_id,
        status=HypothesisCoverageStatus.NOT_ESTIMABLE,
        reason_code="different-filter",
    )

    for record in (wrong_status, wrong_reason):
        with pytest.raises(ContractError) as error:
            validate_frozen_hypothesis_coverage(universe, (*common, record))
        assert error.value.details.code == (
            "prefiltered_hypothesis_coverage_mismatch"
        )


def test_declaration_universe_record_and_coverage_reject_tampering() -> None:
    universe, primary, _, _ = _universe()
    object.__setattr__(primary, "hypothesis_id", "hypothesis_forged")
    with pytest.raises(ContractError) as universe_error:
        universe.to_dict()
    assert universe_error.value.details.code == (
        "frozen_hypothesis_universe_integrity_violation"
    )

    intact_universe, intact_primary, intact_filtered, intact_other = _universe()
    records = (
        HypothesisCoverageRecord(
            hypothesis_id=intact_primary.hypothesis_id,
            status=HypothesisCoverageStatus.OBSERVED,
        ),
        HypothesisCoverageRecord(
            hypothesis_id=intact_filtered.hypothesis_id,
            status=HypothesisCoverageStatus.NOT_ESTIMABLE,
            reason_code=intact_filtered.filter_reason_code,
        ),
        HypothesisCoverageRecord(
            hypothesis_id=intact_other.hypothesis_id,
            status=HypothesisCoverageStatus.FAILED,
            reason_code="failed",
        ),
    )
    audit = validate_frozen_hypothesis_coverage(intact_universe, records)
    object.__setattr__(records[0], "status", HypothesisCoverageStatus.FAILED)
    with pytest.raises(ContractError) as audit_error:
        audit.to_dict()
    assert audit_error.value.details.code == (
        "frozen_hypothesis_coverage_integrity_violation"
    )


def test_unknown_hypothesis_lookup_and_invalid_coverage_reason_fail_closed() -> None:
    universe, _, _, _ = _universe()
    with pytest.raises(ContractError) as lookup_error:
        universe.declaration_for("hypothesis_unknown")
    assert lookup_error.value.details.code == "hypothesis_not_in_frozen_universe"
    with pytest.raises(ValueError, match="cannot have a reason"):
        HypothesisCoverageRecord(
            hypothesis_id=universe.hypothesis_ids[0],
            status=HypothesisCoverageStatus.OBSERVED,
            reason_code="unexpected",
        )
    with pytest.raises(ValueError, match="requires a reason"):
        HypothesisCoverageRecord(
            hypothesis_id=universe.hypothesis_ids[0],
            status=HypothesisCoverageStatus.NOT_ESTIMABLE,
        )


def test_filter_stage_is_explicitly_pre_fit() -> None:
    declaration = _primary()
    assert declaration.filter_stage is (
        HypothesisFilterStage.PRE_FIT_OUTCOME_INDEPENDENT
    )
