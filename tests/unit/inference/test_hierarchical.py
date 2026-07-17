from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support.g3f import insufficient_g3f_gate, passed_g3f_gate

from crychic.core import CommunicationMode, ContractError
from crychic.inference.full_pipeline import G3FrequencyCalibrationGate
from crychic.inference.hierarchical import (
    FrozenHierarchicalFDRCollection,
    FrozenHierarchicalFDRSpec,
    HierarchicalFDRHypothesisResult,
    HierarchicalFDRReleaseStatus,
    HypothesisPValueRecord,
    evaluate_hierarchical_fdr,
    freeze_hierarchical_fdr_spec,
)
from crychic.inference.hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisCoverageStatus,
    HypothesisDeclaration,
    HypothesisPrefilterPolicy,
    HypothesisPrefilterStatus,
    HypothesisRole,
    freeze_hypothesis_universe,
)


def _primary(
    index: int,
    *,
    filtered: bool = False,
    endpoint: str = "driver_family_receiver_context_omnibus_v1",
    mode: CommunicationMode = CommunicationMode.STATE,
    multiplicity_family: str = "gene-state-primary",
) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint=endpoint,
        contrast_name="all_contexts_v1",
        receiver=f"Receiver-{index}",
        family_id=f"family-{index}",
        mode=mode,
        role=HypothesisRole.PRIMARY,
        multiplicity_family=multiplicity_family,
        prefilter_policy=(
            HypothesisPrefilterPolicy.EXTERNAL_RESOURCE
            if filtered
            else HypothesisPrefilterPolicy.NONE
        ),
        prefilter_status=(
            HypothesisPrefilterStatus.FILTERED
            if filtered
            else HypothesisPrefilterStatus.INCLUDED
        ),
        filter_reason_code=(f"primary-{index}-filtered" if filtered else None),
    )


def _secondary(
    parent: HypothesisDeclaration,
    index: int,
    *,
    filtered: bool = False,
    endpoint: str = "family_common_integrated_lr_context_effect_v1",
    multiplicity_family: str = "gene-state-secondary",
) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint=endpoint,
        contrast_name=f"posthoc-{index}",
        receiver=parent.receiver,
        family_id=parent.family_id,
        mode=parent.mode,
        role=HypothesisRole.SECONDARY,
        multiplicity_family=multiplicity_family,
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
        filter_reason_code=(
            f"{parent.family_id}-posthoc-{index}-filtered" if filtered else None
        ),
    )


def _universe(
    family_sizes: tuple[int, ...],
    *,
    filtered_family: int | None = None,
) -> tuple[
    FrozenHypothesisUniverse,
    tuple[HypothesisDeclaration, ...],
    tuple[tuple[HypothesisDeclaration, ...], ...],
]:
    primary = tuple(
        _primary(index, filtered=index == filtered_family)
        for index in range(len(family_sizes))
    )
    children = tuple(
        tuple(
            _secondary(
                parent,
                child_index,
                filtered=family_index == filtered_family,
            )
            for child_index in range(size)
        )
        for family_index, (parent, size) in enumerate(
            zip(primary, family_sizes, strict=True)
        )
    )
    declarations = (*primary, *(item for family in children for item in family))
    universe = freeze_hypothesis_universe(
        declarations,
        universe_name="two-level-hierarchical-fdr-fixture",
    )
    return universe, primary, children


def _records(
    universe: FrozenHypothesisUniverse,
    p_values: dict[str, float],
    *,
    unavailable: dict[str, HypothesisCoverageStatus] | None = None,
) -> tuple[HypothesisPValueRecord, ...]:
    unavailable = unavailable or {}
    rows: list[HypothesisPValueRecord] = []
    for declaration in universe.declarations:
        status = unavailable.get(declaration.hypothesis_id)
        if declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED:
            rows.append(
                HypothesisPValueRecord(
                    hypothesis_id=declaration.hypothesis_id,
                    source_result_id=f"source-{declaration.hypothesis_id}",
                    status=HypothesisCoverageStatus.NOT_ESTIMABLE,
                    reason_code=declaration.filter_reason_code,
                )
            )
        elif status is not None:
            rows.append(
                HypothesisPValueRecord(
                    hypothesis_id=declaration.hypothesis_id,
                    source_result_id=f"source-{declaration.hypothesis_id}",
                    status=status,
                    reason_code=f"fixture-{status.value}",
                )
            )
        else:
            rows.append(
                HypothesisPValueRecord(
                    hypothesis_id=declaration.hypothesis_id,
                    source_result_id=f"source-{declaration.hypothesis_id}",
                    status=HypothesisCoverageStatus.OBSERVED,
                    p_value=p_values[declaration.hypothesis_id],
                )
            )
    return tuple(rows)


def _calibration_gate(
    universe: FrozenHypothesisUniverse,
    spec: FrozenHierarchicalFDRSpec,
) -> G3FrequencyCalibrationGate:
    return passed_g3f_gate(universe, spec)


def _evaluate_with_passing_gate(
    universe: FrozenHypothesisUniverse,
) -> FrozenHierarchicalFDRCollection:
    spec = freeze_hierarchical_fdr_spec()
    p_values = {item.hypothesis_id: 0.01 for item in universe.declarations}
    return evaluate_hierarchical_fdr(
        universe,
        _records(universe, p_values),
        spec=spec,
        calibration_gate=_calibration_gate(universe, spec),
    )


def test_fixed_spec_and_hand_calculated_two_level_q_values() -> None:
    universe, primary, children = _universe((2, 1))
    p_values = {
        primary[0].hypothesis_id: 0.01,
        primary[1].hypothesis_id: 0.04,
        children[0][0].hypothesis_id: 0.01,
        children[0][1].hypothesis_id: 0.04,
        children[1][0].hypothesis_id: 0.03,
    }
    spec = freeze_hierarchical_fdr_spec()

    candidate = evaluate_hierarchical_fdr(
        universe,
        _records(universe, p_values),
        spec=spec,
    )

    assert spec.alpha == 0.05
    assert spec.primary_procedure == "benjamini_hochberg_v1"
    assert spec.secondary_procedure == ("benjamini_bogomolov_selected_family_bh_v1")
    assert spec.q_value_scope == "selective_fdr_not_pooled_leaf_fdr"
    assert candidate.n_primary == 2
    assert candidate.n_primary_selected == 2
    assert candidate.secondary_test_level == pytest.approx(0.05)
    assert candidate.release_status is HierarchicalFDRReleaseStatus.CANDIDATE_ONLY
    assert candidate.release_reason_code == "hierarchical_fdr_calibration_gate_missing"
    assert candidate.q_value_release_allowed is False
    assert candidate.result_for(primary[0].hypothesis_id).candidate_q_value == (
        pytest.approx(0.02)
    )
    assert candidate.result_for(primary[1].hypothesis_id).candidate_q_value == (
        pytest.approx(0.04)
    )
    for family in children:
        for child in family:
            row = candidate.result_for(child.hypothesis_id)
            assert row.candidate_q_value == pytest.approx(0.04)
            assert row.candidate_rejected_at_alpha is True
            assert row.rejected_at_alpha is None
            assert row.candidate_parent_selected_at_alpha is True
            assert row.parent_selected_at_alpha is None
            assert row.q_value is None

    passing_gate = _calibration_gate(universe, spec)
    released = evaluate_hierarchical_fdr(
        universe,
        _records(universe, p_values),
        spec=spec,
        calibration_gate=passing_gate,
    )
    assert released.release_status is HierarchicalFDRReleaseStatus.RELEASED
    assert released.q_value_release_allowed is True
    assert released.calibration_gate_id == passing_gate.gate_id
    assert released.calibration_evidence_id == passing_gate.evidence_id
    assert released.calibration_protocol_id == passing_gate.protocol_id
    assert all(row.q_value == row.candidate_q_value for row in released.records)
    assert all(
        row.rejected_at_alpha is row.candidate_rejected_at_alpha
        for row in released.records
    )


@pytest.mark.parametrize(
    "wrong_role",
    [HypothesisRole.PRIMARY, HypothesisRole.SECONDARY],
)  # type: ignore[untyped-decorator]
def test_v1_scope_rejects_wrong_layer_endpoint(
    wrong_role: HypothesisRole,
) -> None:
    primary = _primary(
        0,
        endpoint=(
            "generic-primary-endpoint"
            if wrong_role is HypothesisRole.PRIMARY
            else "driver_family_receiver_context_omnibus_v1"
        ),
    )
    secondary = _secondary(
        primary,
        0,
        endpoint=(
            "generic-secondary-endpoint"
            if wrong_role is HypothesisRole.SECONDARY
            else "family_common_integrated_lr_context_effect_v1"
        ),
    )
    universe = freeze_hypothesis_universe(
        (primary, secondary),
        universe_name="wrong-endpoint-hierarchical-fixture",
    )

    with pytest.raises(ContractError) as error:
        _evaluate_with_passing_gate(universe)
    assert error.value.details.code == "hierarchical_fdr_v1_endpoint_scope_mismatch"


def test_v1_scope_rejects_ecosystem_mode_with_passing_gate() -> None:
    primary = _primary(0, mode=CommunicationMode.ECOSYSTEM)
    secondary = _secondary(primary, 0)
    universe = freeze_hypothesis_universe(
        (primary, secondary),
        universe_name="ecosystem-hierarchical-fixture",
    )

    with pytest.raises(ContractError) as error:
        _evaluate_with_passing_gate(universe)
    assert error.value.details.code == "hierarchical_fdr_v1_mode_scope_mismatch"


@pytest.mark.parametrize(
    "split_role",
    [HypothesisRole.PRIMARY, HypothesisRole.SECONDARY],
)  # type: ignore[untyped-decorator]
def test_v1_scope_rejects_multiple_multiplicity_families_per_layer(
    split_role: HypothesisRole,
) -> None:
    primary = tuple(
        _primary(
            index,
            multiplicity_family=(
                f"primary-family-{index}"
                if split_role is HypothesisRole.PRIMARY
                else "primary-family"
            ),
        )
        for index in range(2)
    )
    secondary = tuple(
        _secondary(
            parent,
            0,
            multiplicity_family=(
                f"secondary-family-{index}"
                if split_role is HypothesisRole.SECONDARY
                else "secondary-family"
            ),
        )
        for index, parent in enumerate(primary)
    )
    universe = freeze_hypothesis_universe(
        (*primary, *secondary),
        universe_name="multiple-multiplicity-family-fixture",
    )

    with pytest.raises(ContractError) as error:
        _evaluate_with_passing_gate(universe)
    assert error.value.details.code == (
        "hierarchical_fdr_v1_multiplicity_scope_mismatch"
    )


def test_bb_level_rejects_less_than_ordinary_selected_family_bh() -> None:
    universe, primary, children = _universe((1, 1))
    p_values = {
        primary[0].hypothesis_id: 0.001,
        primary[1].hypothesis_id: 0.20,
        children[0][0].hypothesis_id: 0.03,
        children[1][0].hypothesis_id: 0.001,
    }

    result = evaluate_hierarchical_fdr(
        universe,
        _records(universe, p_values),
        spec=freeze_hierarchical_fdr_spec(),
    )

    assert result.n_primary_selected == 1
    assert result.secondary_test_level == pytest.approx(0.025)
    selected_parent_child = result.result_for(children[0][0].hypothesis_id)
    assert p_values[children[0][0].hypothesis_id] < 0.05
    assert selected_parent_child.candidate_rejected_at_alpha is False
    assert selected_parent_child.rejected_at_alpha is None
    assert selected_parent_child.candidate_q_value == pytest.approx(0.06)
    unselected_parent_child = result.result_for(children[1][0].hypothesis_id)
    assert p_values[children[1][0].hypothesis_id] < 0.05
    assert unselected_parent_child.candidate_parent_selected_at_alpha is False
    assert unselected_parent_child.parent_selected_at_alpha is None
    assert unselected_parent_child.candidate_rejected_at_alpha is False
    assert unselected_parent_child.rejected_at_alpha is None
    assert unselected_parent_child.candidate_q_value == pytest.approx(0.20)


def test_ties_and_input_order_are_deterministic() -> None:
    universe, primary, children = _universe((1, 1, 1))
    p_values = {
        primary[0].hypothesis_id: 0.01,
        primary[1].hypothesis_id: 0.01,
        primary[2].hypothesis_id: 0.20,
        children[0][0].hypothesis_id: 0.01,
        children[1][0].hypothesis_id: 0.01,
        children[2][0].hypothesis_id: 0.01,
    }
    records = _records(universe, p_values)
    spec = freeze_hierarchical_fdr_spec()

    first = evaluate_hierarchical_fdr(universe, records, spec=spec)
    second = evaluate_hierarchical_fdr(
        universe,
        tuple(reversed(records)),
        spec=spec,
    )

    assert first.collection_id == second.collection_id
    assert tuple(item.result_id for item in first.records) == tuple(
        item.result_id for item in second.records
    )
    assert first.result_for(primary[0].hypothesis_id).candidate_q_value == (
        pytest.approx(0.015)
    )
    assert first.result_for(primary[1].hypothesis_id).candidate_q_value == (
        pytest.approx(0.015)
    )
    assert (
        first.result_for(children[2][0].hypothesis_id).candidate_rejected_at_alpha
        is False
    )


def test_all_one_effective_p_values_select_nothing_at_alpha() -> None:
    universe, primary, children = _universe((2, 2))
    p_values = {declaration.hypothesis_id: 1.0 for declaration in universe.declarations}

    result = evaluate_hierarchical_fdr(
        universe,
        _records(universe, p_values),
        spec=freeze_hierarchical_fdr_spec(),
    )

    assert result.n_primary == len(primary)
    assert result.n_primary_selected == 0
    assert result.secondary_test_level == 0.0
    assert all(row.candidate_q_value == 1.0 for row in result.records)
    assert not any(row.candidate_rejected_at_alpha for row in result.records)
    assert all(row.rejected_at_alpha is None for row in result.records)
    assert all(
        result.result_for(child.hypothesis_id).candidate_parent_selected_at_alpha
        is False
        for family in children
        for child in family
    )
    assert all(
        result.result_for(child.hypothesis_id).parent_selected_at_alpha is None
        for family in children
        for child in family
    )


def test_filtered_rows_keep_denominators_but_do_not_block_release() -> None:
    universe, primary, children = _universe((1, 1), filtered_family=1)
    p_values = {
        primary[0].hypothesis_id: 0.001,
        children[0][0].hypothesis_id: 0.03,
    }
    records = _records(universe, p_values)
    spec = freeze_hierarchical_fdr_spec()

    result = evaluate_hierarchical_fdr(
        universe,
        records,
        spec=spec,
        calibration_gate=_calibration_gate(universe, spec),
    )

    assert result.q_value_release_allowed is True
    assert result.n_primary == 2
    assert result.n_primary_selected == 1
    assert result.secondary_test_level == pytest.approx(0.025)
    observed_child = result.result_for(children[0][0].hypothesis_id)
    assert observed_child.candidate_rejected_at_alpha is False
    assert observed_child.rejected_at_alpha is False
    assert observed_child.candidate_parent_selected_at_alpha is True
    assert observed_child.parent_selected_at_alpha is True
    assert observed_child.candidate_q_value == pytest.approx(0.06)
    assert observed_child.q_value == pytest.approx(0.06)
    for declaration in (primary[1], children[1][0]):
        row = result.result_for(declaration.hypothesis_id)
        source = next(
            item for item in records if item.hypothesis_id == declaration.hypothesis_id
        )
        assert source.p_value is None
        assert source.effective_p_value == 1.0
        assert row.effective_p_value == 1.0
        assert row.candidate_q_value == 1.0
        assert row.q_value is None
        assert row.rejected_at_alpha is None
        assert row.parent_selected_at_alpha is None


@pytest.mark.parametrize(
    "status",
    [HypothesisCoverageStatus.NOT_ESTIMABLE, HypothesisCoverageStatus.FAILED],
)  # type: ignore[untyped-decorator]
def test_included_unavailable_row_globally_blocks_release(
    status: HypothesisCoverageStatus,
) -> None:
    universe, primary, children = _universe((1, 1))
    p_values = {
        primary[0].hypothesis_id: 0.001,
        primary[1].hypothesis_id: 0.02,
        children[0][0].hypothesis_id: 0.01,
        children[1][0].hypothesis_id: 0.01,
    }
    unavailable_id = children[1][0].hypothesis_id
    records = _records(
        universe,
        p_values,
        unavailable={unavailable_id: status},
    )
    spec = freeze_hierarchical_fdr_spec()

    result = evaluate_hierarchical_fdr(
        universe,
        records,
        spec=spec,
        calibration_gate=_calibration_gate(universe, spec),
    )

    assert result.release_status is HierarchicalFDRReleaseStatus.BLOCKED
    assert result.release_reason_code == "included_hypothesis_p_value_unavailable"
    assert result.q_value_release_allowed is False
    assert all(row.q_value is None for row in result.records)
    assert all(row.rejected_at_alpha is None for row in result.records)
    assert all(row.parent_selected_at_alpha is None for row in result.records)
    unavailable = result.result_for(unavailable_id)
    assert unavailable.status is status
    assert unavailable.effective_p_value == 1.0
    assert unavailable.candidate_q_value == 1.0


def test_exact_coverage_and_filtered_reason_fail_closed() -> None:
    universe, primary, _ = _universe((1, 1), filtered_family=1)
    p_values = {primary[0].hypothesis_id: 0.01}
    for declaration in universe.declarations:
        if declaration.prefilter_status is HypothesisPrefilterStatus.INCLUDED:
            p_values.setdefault(declaration.hypothesis_id, 0.01)
    records = _records(universe, p_values)
    spec = freeze_hierarchical_fdr_spec()

    with pytest.raises(ContractError) as missing_error:
        evaluate_hierarchical_fdr(universe, records[:-1], spec=spec)
    assert missing_error.value.details.code == "missing_hypothesis_p_value"
    with pytest.raises(ContractError) as duplicate_error:
        evaluate_hierarchical_fdr(universe, (*records, records[0]), spec=spec)
    assert duplicate_error.value.details.code == "duplicate_hypothesis_p_value"
    extra = HypothesisPValueRecord(
        hypothesis_id="hypothesis-extra",
        source_result_id="source-extra",
        status=HypothesisCoverageStatus.FAILED,
        reason_code="outside-universe",
    )
    with pytest.raises(ContractError) as extra_error:
        evaluate_hierarchical_fdr(universe, (*records, extra), spec=spec)
    assert extra_error.value.details.code == "extra_hypothesis_p_value"

    filtered_index = next(
        index
        for index, declaration in enumerate(universe.declarations)
        if declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED
    )
    filtered = records[filtered_index]
    wrong = replace(filtered, reason_code="wrong-frozen-reason")
    invalid = (*records[:filtered_index], wrong, *records[filtered_index + 1 :])
    with pytest.raises(ContractError) as filtered_error:
        evaluate_hierarchical_fdr(universe, invalid, spec=spec)
    assert filtered_error.value.details.code == (
        "prefiltered_hypothesis_p_value_mismatch"
    )


def test_calibration_gate_is_producer_owned_and_exactly_bound() -> None:
    universe, primary, children = _universe((1,))
    p_values = {
        primary[0].hypothesis_id: 0.01,
        children[0][0].hypothesis_id: 0.01,
    }
    records = _records(universe, p_values)
    spec = freeze_hierarchical_fdr_spec()

    with pytest.raises(TypeError, match="never a boolean"):
        evaluate_hierarchical_fdr(
            universe,
            records,
            spec=spec,
            calibration_gate=True,
        )
    other_universe, _, _ = _universe((2,))
    wrong_universe = _calibration_gate(other_universe, spec)
    with pytest.raises(ContractError) as universe_error:
        evaluate_hierarchical_fdr(
            universe,
            records,
            spec=spec,
            calibration_gate=wrong_universe,
        )
    assert (
        universe_error.value.details.code == "hierarchical_fdr_gate_universe_mismatch"
    )
    wrong_procedure = _calibration_gate(universe, spec)
    original_procedure = wrong_procedure.hierarchical_procedure_id
    object.__setattr__(
        wrong_procedure,
        "hierarchical_procedure_id",
        "different-procedure",
    )
    try:
        with pytest.raises(ContractError) as procedure_error:
            evaluate_hierarchical_fdr(
                universe,
                records,
                spec=spec,
                calibration_gate=wrong_procedure,
            )
    finally:
        object.__setattr__(
            wrong_procedure,
            "hierarchical_procedure_id",
            original_procedure,
        )
    assert procedure_error.value.details.code == (
        "g3_frequency_gate_integrity_violation"
    )

    failed_gate = insufficient_g3f_gate(universe, spec)
    blocked = evaluate_hierarchical_fdr(
        universe,
        records,
        spec=spec,
        calibration_gate=failed_gate,
    )
    assert blocked.release_status is HierarchicalFDRReleaseStatus.BLOCKED
    assert blocked.release_reason_code == "g3_frequency_scenario_replicates_below_1000"
    assert blocked.calibration_gate_id == failed_gate.gate_id
    assert all(row.q_value is None for row in blocked.records)
    assert all(row.rejected_at_alpha is None for row in blocked.records)
    assert all(row.parent_selected_at_alpha is None for row in blocked.records)
    assert all(row.candidate_rejected_at_alpha for row in blocked.records)


def test_spec_input_gate_results_and_collection_reject_tampering() -> None:
    universe, primary, children = _universe((1,))
    p_values = {
        primary[0].hypothesis_id: 0.01,
        children[0][0].hypothesis_id: 0.01,
    }

    tampered_spec = freeze_hierarchical_fdr_spec()
    object.__setattr__(tampered_spec, "alpha", 0.10)
    with pytest.raises(ContractError) as spec_error:
        tampered_spec.to_dict()
    assert spec_error.value.details.code == "hierarchical_fdr_spec_integrity_violation"

    spec = freeze_hierarchical_fdr_spec()
    tampered_records = list(_records(universe, p_values))
    object.__setattr__(tampered_records[0], "p_value", 0.9)
    with pytest.raises(ContractError) as input_error:
        evaluate_hierarchical_fdr(universe, tampered_records, spec=spec)
    assert input_error.value.details.code == (
        "hypothesis_p_value_record_integrity_violation"
    )

    gate = _calibration_gate(universe, spec)
    original_gate_id = gate.gate_id
    object.__setattr__(gate, "gate_id", "forged-gate")
    try:
        with pytest.raises(ContractError) as gate_error:
            evaluate_hierarchical_fdr(
                universe,
                _records(universe, p_values),
                spec=spec,
                calibration_gate=gate,
            )
    finally:
        object.__setattr__(gate, "gate_id", original_gate_id)
    assert gate_error.value.details.code == "g3_frequency_gate_integrity_violation"

    result = evaluate_hierarchical_fdr(
        universe,
        _records(universe, p_values),
        spec=spec,
    )
    object.__setattr__(result.records[0], "candidate_q_value", 0.9)
    with pytest.raises(ContractError) as record_error:
        result.to_dict()
    assert record_error.value.details.code == (
        "hierarchical_fdr_collection_integrity_violation"
    )

    intact = evaluate_hierarchical_fdr(
        universe,
        _records(universe, p_values),
        spec=spec,
    )
    object.__setattr__(intact, "n_primary_selected", 99)
    with pytest.raises(ContractError) as collection_error:
        intact.to_dict()
    assert collection_error.value.details.code == (
        "hierarchical_fdr_collection_integrity_violation"
    )

    with pytest.raises(TypeError, match="producer-owned"):
        FrozenHierarchicalFDRSpec()
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenHierarchicalFDRCollection()
    with pytest.raises(TypeError, match="producer-owned"):
        HierarchicalFDRHypothesisResult()
