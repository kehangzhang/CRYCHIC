from __future__ import annotations

from collections.abc import Callable

import pytest

import crychic.workflow.frozen_selection_frequency_universe as module
from crychic.core import ContractError
from crychic.inference.bootstrap_selection import SelectionFrequencyStatus
from crychic.inference.hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisDeclaration,
    HypothesisPrefilterPolicy,
    HypothesisPrefilterStatus,
    HypothesisRole,
    freeze_hypothesis_universe,
)
from crychic.resampling import SubjectBootstrapPlan
from crychic.scoring import FAMILY_COMMON_SCORE_VERSION
from crychic.workflow.full_pipeline_effects import FrozenFamilyEffectTarget
from crychic.workflow.full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
)
from crychic.workflow.resampled_attribution import (
    FrozenResampledAttributionCollection,
    ResampledAttributionEventStatus,
    ResampledFamilySelectionEvent,
)

_ENDPOINT = "driver_family_receiver_context_omnibus_v1"


def _universe() -> tuple[
    FrozenHypothesisUniverse,
    HypothesisDeclaration,
    HypothesisDeclaration,
    HypothesisDeclaration,
]:
    included = HypothesisDeclaration(
        endpoint=_ENDPOINT,
        contrast_name="driver-context-omnibus",
        receiver="Receiver-included",
        family_id="family-included",
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="driver-family-primary",
    )
    filtered = HypothesisDeclaration(
        endpoint=_ENDPOINT,
        contrast_name="driver-context-omnibus",
        receiver="Receiver-filtered",
        family_id="family-filtered",
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="driver-family-primary",
        prefilter_policy=HypothesisPrefilterPolicy.EXTERNAL_RESOURCE,
        prefilter_status=HypothesisPrefilterStatus.FILTERED,
        filter_reason_code="family_absent_from_frozen_resource",
    )
    unrelated = HypothesisDeclaration(
        endpoint="unrelated_primary_endpoint_v1",
        contrast_name="unrelated",
        receiver="Receiver-unrelated",
        family_id="family-unrelated",
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="other-primary",
    )
    universe = freeze_hypothesis_universe(
        (included, filtered, unrelated),
        universe_name="selection-frequency-universe-v1",
    )
    return universe, included, filtered, unrelated


def _plan(index: int) -> SubjectBootstrapPlan:
    plan = object.__new__(SubjectBootstrapPlan)
    object.__setattr__(plan, "bootstrap_id", f"bootstrap-plan-{index:04d}")
    object.__setattr__(plan, "resample_index", index)
    return plan


def _workflow_record(index: int) -> FullPipelineResampleRecord:
    record = object.__new__(FullPipelineResampleRecord)
    object.__setattr__(
        record,
        "operation",
        FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP,
    )
    object.__setattr__(record, "plan_id", f"bootstrap-plan-{index:04d}")
    object.__setattr__(record, "resample_index", index)
    object.__setattr__(record, "record_id", f"workflow-record-{index:04d}")
    return record


def _resampling(n_plans: int) -> FullPipelineResamplingResult:
    result = object.__new__(FullPipelineResamplingResult)
    object.__setattr__(result, "result_id", f"resampling-{n_plans}")
    object.__setattr__(result, "crossfit_spec_id", "crossfit-spec-v7")
    object.__setattr__(result, "retain_children", True)
    object.__setattr__(result, "plans", tuple(_plan(i) for i in range(n_plans)))
    object.__setattr__(
        result,
        "records",
        tuple(_workflow_record(i) for i in range(n_plans)),
    )
    return result


def _event(
    index: int,
    target: FrozenFamilyEffectTarget,
    *,
    status: ResampledAttributionEventStatus = (
        ResampledAttributionEventStatus.OBSERVED
    ),
) -> ResampledFamilySelectionEvent:
    event = object.__new__(ResampledFamilySelectionEvent)
    selected = (
        index % 2 == 0 if status is ResampledAttributionEventStatus.OBSERVED else None
    )
    reason = (
        None
        if status is ResampledAttributionEventStatus.OBSERVED
        else f"source_{status.value}"
    )
    values: dict[str, object] = {
        "plan_id": f"bootstrap-plan-{index:04d}",
        "resample_index": index,
        "fold_id": "fold-0",
        "target_id": target.target_id,
        "universe_id": target.universe_id,
        "hypothesis_id": target.hypothesis_id,
        "crossfit_spec_id": "crossfit-spec-v7",
        "selection_rule_id": "outer-training-positive-coefficient-v1",
        "score_version": FAMILY_COMMON_SCORE_VERSION,
        "family_selected": selected,
        "status": status,
        "reason_code": reason,
        "event_id": f"selection-event-{index:04d}",
    }
    for name, value in values.items():
        object.__setattr__(event, name, value)
    return event


def _source_collection(
    resampling: FullPipelineResamplingResult,
    target: FrozenFamilyEffectTarget,
    n_plans: int,
) -> FrozenResampledAttributionCollection:
    source = object.__new__(FrozenResampledAttributionCollection)
    events = tuple(_event(index, target) for index in range(n_plans))
    values: dict[str, object] = {
        "resampling_result_id": resampling.result_id,
        "target_id": target.target_id,
        "universe_id": target.universe_id,
        "hypothesis_id": target.hypothesis_id,
        "crossfit_spec_id": resampling.crossfit_spec_id,
        "bootstrap_plan_ids": tuple(event.plan_id for event in events),
        "full_pipeline_record_ids": tuple(
            record.record_id for record in resampling.records
        ),
        "event_ids": tuple(event.event_id for event in events),
        "events": events,
        "collection_id": f"source-selection-{target.hypothesis_id}-{n_plans}",
        "_target": target,
        "_resampling": resampling,
    }
    for name, value in values.items():
        object.__setattr__(source, name, value)
    return source


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _stub_source_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        FullPipelineResamplingResult,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        FrozenResampledAttributionCollection,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        ResampledFamilySelectionEvent,
        "_require_intact",
        lambda self, **kwargs: None,
    )


def _install_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_plans: int,
) -> list[str]:
    calls: list[str] = []

    def summarize(
        resampling: FullPipelineResamplingResult,
        target: FrozenFamilyEffectTarget,
    ) -> FrozenResampledAttributionCollection:
        calls.append(target.hypothesis_id)
        return _source_collection(resampling, target, n_plans)

    monkeypatch.setattr(module, "summarize_resampled_family_selection", summarize)
    return calls


def _run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_plans: int,
) -> tuple[
    module.FrozenSelectionFrequencyUniverseCollection,
    HypothesisDeclaration,
    HypothesisDeclaration,
    list[str],
]:
    universe, included, filtered, _ = _universe()
    calls = _install_source(monkeypatch, n_plans=n_plans)
    collection = module.summarize_frozen_selection_frequency_universe(
        universe,
        _resampling(n_plans),
    )
    return collection, included, filtered, calls


def test_complete_universe_releases_only_observed_included_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection, included, filtered, calls = _run(monkeypatch, n_plans=1_000)
    included_row = collection.result_for(included.hypothesis_id)
    filtered_row = collection.result_for(filtered.hypothesis_id)

    assert calls == [included.hypothesis_id]
    assert collection.exact_universe_coverage is True
    assert collection.n_records_total == 2
    assert collection.n_observed == 1
    assert collection.n_not_estimable == 1
    assert collection.n_failed == 0
    assert len(collection.universe_declaration_ids) == 3
    assert collection.applicable_hypothesis_ids == tuple(
        item.hypothesis_id
        for item in collection._universe.declarations
        if item.endpoint == _ENDPOINT and item.role is HypothesisRole.PRIMARY
    )
    assert included_row.status is SelectionFrequencyStatus.OBSERVED
    assert included_row.selection_frequency == 0.5
    assert included_row.selection_frequency_release_allowed is True
    assert included_row._child is not None
    assert included_row._child.selection_frequency_release_allowed is False
    assert filtered_row.status is SelectionFrequencyStatus.NOT_ESTIMABLE
    assert filtered_row.reason_code == "family_absent_from_frozen_resource"
    assert filtered_row.child_result_id is None
    assert filtered_row.selection_frequency_release_allowed is False

    payload = collection.to_dict()
    assert payload["performance_status"] == (
        "serial_universe_batch_parallel_execution_deferred_v1"
    )
    assert payload["excluded_output_kinds"] == [
        "p_value",
        "q_value",
        "posterior_probability",
        "communication_probability",
    ]
    forbidden = {
        "p_value",
        "q_value",
        "posterior_probability",
        "comm_probability",
        "communication_probability",
    }
    assert forbidden.isdisjoint(payload)
    assert all(forbidden.isdisjoint(row) for row in payload["records"])


def test_less_than_1000_remains_not_estimable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection, included, filtered, _ = _run(monkeypatch, n_plans=999)
    included_row = collection.result_for(included.hypothesis_id)
    filtered_row = collection.result_for(filtered.hypothesis_id)

    assert included_row.status is SelectionFrequencyStatus.NOT_ESTIMABLE
    assert included_row.selection_frequency is None
    assert included_row.reason_code == (
        "selection_frequency_bootstrap_count_below_1000"
    )
    assert included_row.selection_frequency_release_allowed is False
    assert filtered_row.status is SelectionFrequencyStatus.NOT_ESTIMABLE
    assert collection.n_observed == 0
    assert collection.n_not_estimable == 2


def test_included_adapter_failure_is_retained_as_typed_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, included, filtered, _ = _universe()
    resampling = _resampling(3)

    def fail(
        resampling: FullPipelineResamplingResult,
        target: FrozenFamilyEffectTarget,
    ) -> FrozenResampledAttributionCollection:
        raise ContractError(
            "synthetic source failure",
            code="synthetic_v03_11_failure",
        )

    monkeypatch.setattr(module, "summarize_resampled_family_selection", fail)
    collection = module.summarize_frozen_selection_frequency_universe(
        universe,
        resampling,
    )

    failed = collection.result_for(included.hypothesis_id)
    prefiltered = collection.result_for(filtered.hypothesis_id)
    assert failed.status is SelectionFrequencyStatus.FAILED
    assert failed.reason_code == "synthetic_v03_11_failure"
    assert failed.child_result_id is None
    assert failed.selection_frequency_release_allowed is False
    assert prefiltered.status is SelectionFrequencyStatus.NOT_ESTIMABLE
    assert prefiltered.reason_code == "family_absent_from_frozen_resource"
    assert collection.n_failed == 1


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutate",
    [
        lambda records: records[:-1],
        lambda records: (*records, records[0]),
    ],
)
def test_missing_or_extra_applicable_record_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[
        [tuple[module.FrozenSelectionFrequencyUniverseRecord, ...]],
        tuple[module.FrozenSelectionFrequencyUniverseRecord, ...],
    ],
) -> None:
    collection, _, _, _ = _run(monkeypatch, n_plans=3)
    original = collection.records
    object.__setattr__(collection, "records", mutate(original))

    with pytest.raises(ContractError) as caught:
        collection.to_dict()

    assert caught.value.details.code == (
        "selection_frequency_universe_collection_integrity_violation"
    )


def test_record_and_collection_tampering_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection, _, filtered, _ = _run(monkeypatch, n_plans=3)
    filtered_row = next(
        item
        for item in collection.records
        if item.hypothesis_id == filtered.hypothesis_id
    )
    object.__setattr__(filtered_row, "status", SelectionFrequencyStatus.OBSERVED)
    with pytest.raises(ContractError) as caught:
        filtered_row.to_dict()
    assert caught.value.details.code == (
        "selection_frequency_universe_record_integrity_violation"
    )

    collection, _, _, _ = _run(monkeypatch, n_plans=3)
    object.__setattr__(collection, "child_result_ids", ("forged-child", None))
    with pytest.raises(ContractError) as caught:
        collection.to_dict()
    assert caught.value.details.code == (
        "selection_frequency_universe_collection_integrity_violation"
    )


def test_public_result_types_are_producer_owned() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        module.FrozenFamilySelectionFrequency()
    with pytest.raises(TypeError, match="producer-owned"):
        module.FrozenSelectionFrequencyUniverseRecord()
    with pytest.raises(TypeError, match="producer-owned"):
        module.FrozenSelectionFrequencyUniverseCollection()
