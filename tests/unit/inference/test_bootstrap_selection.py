from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from crychic.core import ContractError
from crychic.inference.bootstrap_selection import (
    BootstrapSelectionOpportunity,
    BootstrapSelectionRecord,
    SelectionFrequencyResult,
    SelectionFrequencySpec,
    SelectionFrequencyStatus,
    summarize_selection_frequency,
)

_TARGET_ID = "target-receiver-R-family-F"
_UNIVERSE_ID = "frozen-hypothesis-universe"
_HYPOTHESIS_ID = "receiver-R-family-F"
_CROSSFIT_SPEC_ID = "crossfit-spec-v7"
_SELECTION_RULE_ID = "outer-training-positive-coefficient-v1"
_SCORE_VERSION = "family-first-mechanistic-v2"
_GRAIN = "equal_bootstrap_mean_of_outer_fold_selection_v1"


def _plan_ids(n_plans: int) -> tuple[str, ...]:
    return tuple(f"bootstrap-plan-{index:04d}" for index in range(n_plans))


def _opportunity(
    index: int,
    *,
    fold_id: str | None = "fold-0",
    plan_id: str | None = None,
) -> BootstrapSelectionOpportunity:
    return BootstrapSelectionOpportunity(
        plan_id=plan_id or _plan_ids(index + 1)[index],
        resample_index=index,
        fold_id=fold_id,
    )


def _opportunities(n_plans: int) -> tuple[BootstrapSelectionOpportunity, ...]:
    return tuple(_opportunity(index) for index in range(n_plans))


def _spec(n_plans: int = 1_000, **changes: Any) -> SelectionFrequencySpec:
    values: dict[str, Any] = {
        "target_id": _TARGET_ID,
        "hypothesis_universe_id": _UNIVERSE_ID,
        "hypothesis_id": _HYPOTHESIS_ID,
        "crossfit_spec_id": _CROSSFIT_SPEC_ID,
        "selection_rule_id": _SELECTION_RULE_ID,
        "score_version": _SCORE_VERSION,
        "subject_bootstrap_plan_ids": _plan_ids(n_plans),
        "expected_opportunities": _opportunities(n_plans),
    }
    values.update(changes)
    return SelectionFrequencySpec(**values)


def _record(
    index: int,
    *,
    plan_id: str | None = None,
    fold_id: str | None = "fold-0",
    selected: bool | None = False,
    status: SelectionFrequencyStatus = SelectionFrequencyStatus.OBSERVED,
    reason_code: str | None = None,
    target_id: str = _TARGET_ID,
    hypothesis_universe_id: str = _UNIVERSE_ID,
    hypothesis_id: str = _HYPOTHESIS_ID,
    crossfit_spec_id: str = _CROSSFIT_SPEC_ID,
    selection_rule_id: str = _SELECTION_RULE_ID,
    score_version: str = _SCORE_VERSION,
    resampling_kind: str = "subject_bootstrap",
) -> BootstrapSelectionRecord:
    if status is not SelectionFrequencyStatus.OBSERVED and reason_code is None:
        reason_code = f"source_{status.value}"
    return BootstrapSelectionRecord(
        plan_id=plan_id or _plan_ids(index + 1)[index],
        resample_index=index,
        fold_id=fold_id,
        target_id=target_id,
        hypothesis_universe_id=hypothesis_universe_id,
        hypothesis_id=hypothesis_id,
        crossfit_spec_id=crossfit_spec_id,
        selection_rule_id=selection_rule_id,
        score_version=score_version,
        selected=selected,
        status=status,
        reason_code=reason_code,
        resampling_kind=resampling_kind,
    )


def _observed_records(n_plans: int) -> tuple[BootstrapSelectionRecord, ...]:
    return tuple(_record(index) for index in range(n_plans))


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def complete_case() -> tuple[
    SelectionFrequencySpec,
    tuple[BootstrapSelectionRecord, ...],
]:
    return _spec(), _observed_records(1_000)


def test_equal_bootstrap_mean_is_not_event_or_exposure_weighted() -> None:
    opportunities = (
        _opportunity(0, fold_id="fold-a"),
        _opportunity(0, fold_id="fold-b"),
        _opportunity(1, fold_id="fold-a"),
        _opportunity(1, fold_id="fold-b"),
        _opportunity(1, fold_id="fold-c"),
        _opportunity(1, fold_id="fold-d"),
        *(_opportunity(index) for index in range(2, 1_000)),
    )
    spec = _spec(expected_opportunities=opportunities)
    records = (
        _record(0, fold_id="fold-a", selected=True),
        _record(0, fold_id="fold-b", selected=False),
        _record(1, fold_id="fold-a", selected=True),
        _record(1, fold_id="fold-b", selected=True),
        _record(1, fold_id="fold-c", selected=True),
        _record(1, fold_id="fold-d", selected=True),
        *(_record(index) for index in range(2, 1_000)),
    )

    result = summarize_selection_frequency(spec, tuple(reversed(records)))
    repeated = summarize_selection_frequency(spec, records)

    assert result.status is SelectionFrequencyStatus.OBSERVED
    assert result.selection_frequency == pytest.approx((0.5 + 1.0) / 1_000)
    assert result.selection_frequency != pytest.approx(5.0 / 1_004)
    assert result.n_bootstrap_plans_total == 1_000
    assert result.n_bootstrap_plans_observed == 1_000
    assert result.n_bootstrap_plans_not_estimable == 0
    assert result.n_bootstrap_plans_failed == 0
    assert result.n_event_rows_total == 1_004
    assert result.n_observed_event_rows == 1_004
    assert result.n_not_estimable_event_rows == 0
    assert result.n_failed_event_rows == 0
    assert result.n_selected_event_rows == 5
    assert result.opportunity_grain == _GRAIN
    assert result.result_id == repeated.result_id

    payload = result.to_dict()
    assert payload["source_binding_status"] == (
        "unbound_numeric_primitive_requires_frozen_workflow_adapter"
    )
    assert payload["public_release_allowed"] is False
    forbidden = {
        "p",
        "p_value",
        "q",
        "q_value",
        "probability",
        "posterior_probability",
        "comm_probability",
        "communication_probability",
    }
    assert forbidden.isdisjoint(payload)
    assert not hasattr(result, "p_value")
    assert not hasattr(result, "q_value")
    assert not hasattr(result, "comm_probability")


def test_999_is_not_estimable_but_1000_false_events_are_observed(
    complete_case: tuple[
        SelectionFrequencySpec,
        tuple[BootstrapSelectionRecord, ...],
    ],
) -> None:
    short = summarize_selection_frequency(_spec(999), _observed_records(999))
    spec, records = complete_case
    complete = summarize_selection_frequency(spec, records)

    assert short.status is SelectionFrequencyStatus.NOT_ESTIMABLE
    assert short.selection_frequency is None
    assert short.reason_code == "selection_frequency_bootstrap_count_below_1000"
    assert short.n_bootstrap_plans_total == 999
    assert short.n_observed_event_rows == 999
    assert short.n_selected_event_rows == 0

    assert complete.status is SelectionFrequencyStatus.OBSERVED
    assert complete.selection_frequency == 0.0
    assert complete.reason_code is None
    assert complete.n_observed_event_rows == 1_000


def test_failed_event_has_priority_over_not_estimable(
    complete_case: tuple[
        SelectionFrequencySpec,
        tuple[BootstrapSelectionRecord, ...],
    ],
) -> None:
    _, observed = complete_case
    opportunities = (
        _opportunity(0, fold_id=None),
        _opportunity(1, fold_id=None),
        *(_opportunity(index) for index in range(2, 1_000)),
    )
    spec = _spec(expected_opportunities=opportunities)
    mixed = (
        _record(
            0,
            fold_id=None,
            selected=None,
            status=SelectionFrequencyStatus.NOT_ESTIMABLE,
        ),
        _record(
            1,
            fold_id=None,
            selected=None,
            status=SelectionFrequencyStatus.FAILED,
        ),
        *observed[2:],
    )

    failed = summarize_selection_frequency(spec, mixed)
    not_estimable = summarize_selection_frequency(
        spec,
        (
            mixed[0],
            _record(
                1,
                fold_id=None,
                selected=None,
                status=SelectionFrequencyStatus.NOT_ESTIMABLE,
            ),
            *observed[2:],
        ),
    )

    assert failed.status is SelectionFrequencyStatus.FAILED
    assert failed.selection_frequency is None
    assert failed.reason_code == "selection_frequency_bootstrap_event_failed"
    assert failed.n_bootstrap_plans_failed == 1
    assert failed.n_bootstrap_plans_not_estimable == 1
    assert failed.n_failed_event_rows == 1
    assert failed.n_not_estimable_event_rows == 1

    assert not_estimable.status is SelectionFrequencyStatus.NOT_ESTIMABLE
    assert not_estimable.selection_frequency is None
    assert not_estimable.reason_code == (
        "selection_frequency_bootstrap_event_not_estimable"
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("records", "expected_code"),
    [
        (_observed_records(3)[:-1], "selection_frequency_plan_set_mismatch"),
        (
            (*_observed_records(3), _record(3)),
            "selection_frequency_plan_set_mismatch",
        ),
        (
            (*_observed_records(3), _observed_records(3)[0]),
            "selection_frequency_duplicate_record",
        ),
        (
            (*_observed_records(3), _record(0, selected=True)),
            "selection_frequency_duplicate_plan_fold",
        ),
    ],
)
def test_exact_plan_and_unique_fold_coverage_is_required(
    records: tuple[BootstrapSelectionRecord, ...],
    expected_code: str,
) -> None:
    with pytest.raises(ContractError) as caught:
        summarize_selection_frequency(_spec(3), records)

    assert caught.value.details.code == expected_code


def test_expected_fold_manifest_rejects_missing_and_extra_observed_fold() -> None:
    opportunities = (
        _opportunity(0, fold_id="fold-a"),
        _opportunity(0, fold_id="fold-b"),
        _opportunity(1),
    )
    spec = _spec(2, expected_opportunities=opportunities)
    complete = (
        _record(0, fold_id="fold-a"),
        _record(0, fold_id="fold-b"),
        _record(1),
    )

    for records in (
        (complete[0], complete[2]),
        (*complete, _record(0, fold_id="fold-c")),
    ):
        with pytest.raises(ContractError) as caught:
            summarize_selection_frequency(spec, records)
        assert caught.value.details.code == (
            "selection_frequency_opportunity_coverage_mismatch"
        )


def test_unique_multiple_folds_are_valid_and_foldless_placeholder_is_exclusive() -> (
    None
):
    opportunities = (
        _opportunity(0, fold_id="fold-a"),
        _opportunity(0, fold_id="fold-b"),
        _opportunity(1),
        _opportunity(2),
    )
    spec = _spec(3, expected_opportunities=opportunities)
    multiple = (
        _record(0, fold_id="fold-a"),
        _record(0, fold_id="fold-b", selected=True),
        _record(1),
        _record(2),
    )
    result = summarize_selection_frequency(spec, multiple)

    assert result.status is SelectionFrequencyStatus.NOT_ESTIMABLE
    assert result.reason_code == "selection_frequency_bootstrap_count_below_1000"
    assert result.n_event_rows_total == 4
    assert result.n_selected_event_rows == 1

    ambiguous = (
        _record(
            0,
            fold_id=None,
            selected=None,
            status=SelectionFrequencyStatus.NOT_ESTIMABLE,
        ),
        _record(0, fold_id="fold-a"),
        _record(1),
        _record(2),
    )
    with pytest.raises(ContractError) as caught:
        summarize_selection_frequency(spec, ambiguous)
    assert caught.value.details.code == "selection_frequency_foldless_plan_ambiguous"


def test_plan_and_resample_lineage_must_be_one_to_one() -> None:
    spec = _spec(2)
    split_plan = (
        _record(0, fold_id="fold-a"),
        BootstrapSelectionRecord(
            plan_id=_plan_ids(2)[0],
            resample_index=1,
            fold_id="fold-b",
            target_id=_TARGET_ID,
            hypothesis_universe_id=_UNIVERSE_ID,
            hypothesis_id=_HYPOTHESIS_ID,
            crossfit_spec_id=_CROSSFIT_SPEC_ID,
            selection_rule_id=_SELECTION_RULE_ID,
            score_version=_SCORE_VERSION,
            selected=False,
            status=SelectionFrequencyStatus.OBSERVED,
            reason_code=None,
        ),
        BootstrapSelectionRecord(
            plan_id=_plan_ids(2)[1],
            resample_index=2,
            fold_id="fold-0",
            target_id=_TARGET_ID,
            hypothesis_universe_id=_UNIVERSE_ID,
            hypothesis_id=_HYPOTHESIS_ID,
            crossfit_spec_id=_CROSSFIT_SPEC_ID,
            selection_rule_id=_SELECTION_RULE_ID,
            score_version=_SCORE_VERSION,
            selected=False,
            status=SelectionFrequencyStatus.OBSERVED,
            reason_code=None,
        ),
    )
    with pytest.raises(ContractError) as caught:
        summarize_selection_frequency(spec, split_plan)
    assert caught.value.details.code == "selection_frequency_plan_resample_mismatch"

    duplicate_index = (
        _record(0),
        BootstrapSelectionRecord(
            plan_id=_plan_ids(2)[1],
            resample_index=0,
            fold_id="fold-0",
            target_id=_TARGET_ID,
            hypothesis_universe_id=_UNIVERSE_ID,
            hypothesis_id=_HYPOTHESIS_ID,
            crossfit_spec_id=_CROSSFIT_SPEC_ID,
            selection_rule_id=_SELECTION_RULE_ID,
            score_version=_SCORE_VERSION,
            selected=False,
            status=SelectionFrequencyStatus.OBSERVED,
            reason_code=None,
        ),
    )
    with pytest.raises(ContractError) as caught:
        summarize_selection_frequency(spec, duplicate_index)
    assert caught.value.details.code == ("selection_frequency_duplicate_resample_index")


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field_name", "value"),
    [
        ("target_id", "other-target"),
        ("hypothesis_universe_id", "other-universe"),
        ("hypothesis_id", "other-hypothesis"),
        ("crossfit_spec_id", "other-crossfit-spec"),
        ("selection_rule_id", "other-selection-rule"),
        ("score_version", "other-score-version"),
    ],
)
def test_record_target_rule_and_spec_lineage_must_match(
    field_name: str,
    value: str,
) -> None:
    spec = _spec(2)
    records = list(_observed_records(2))
    records[0] = replace(records[0], **{field_name: value})

    with pytest.raises(ContractError) as caught:
        summarize_selection_frequency(spec, tuple(records))

    assert caught.value.details.code == "selection_frequency_record_lineage_mismatch"


def test_records_reject_nonbootstrap_and_foldless_observed() -> None:
    with pytest.raises(ValueError, match="subject_bootstrap only"):
        _record(0, resampling_kind="context_permutation")
    with pytest.raises(ValueError, match="require fold_id"):
        _record(0, fold_id=None)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "kwargs",
    [
        {
            "status": SelectionFrequencyStatus.OBSERVED,
            "selected": True,
            "reason_code": "unexpected_reason",
        },
        {
            "status": SelectionFrequencyStatus.NOT_ESTIMABLE,
            "selected": False,
        },
        {
            "status": SelectionFrequencyStatus.FAILED,
            "selected": True,
        },
    ],
)
def test_record_status_contract_is_typed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _record(0, **kwargs)  # type: ignore[arg-type]


def test_spec_freezes_minimum_grain_and_unique_plan_ids() -> None:
    with pytest.raises(ValueError, match="fixed at 1000"):
        _spec(2, minimum_bootstraps=999)
    with pytest.raises(ValueError, match="opportunity_grain is fixed"):
        _spec(2, opportunity_grain="event_weighted")
    with pytest.raises(ValueError, match="must be unique"):
        _spec(2, subject_bootstrap_plan_ids=("plan-a", "plan-a"))
    with pytest.raises(ValueError, match="must be unique"):
        _spec(
            2,
            expected_opportunities=(
                _opportunity(0),
                _opportunity(0),
                _opportunity(1),
            ),
        )
    with pytest.raises(ValueError, match="foldless expected opportunity"):
        _spec(
            2,
            expected_opportunities=(
                _opportunity(0, fold_id=None),
                _opportunity(0, fold_id="fold-a"),
                _opportunity(1),
            ),
        )


def test_record_spec_and_result_tampering_fail_closed(
    complete_case: tuple[
        SelectionFrequencySpec,
        tuple[BootstrapSelectionRecord, ...],
    ],
) -> None:
    record = _record(0)
    object.__setattr__(record, "selected", True)
    with pytest.raises(ContractError) as caught:
        record.to_dict()
    assert caught.value.details.code == (
        "bootstrap_selection_record_integrity_violation"
    )

    spec = _spec(2)
    object.__setattr__(spec, "selection_rule_id", "tampered-rule")
    with pytest.raises(ContractError) as caught:
        spec.to_dict()
    assert caught.value.details.code == "selection_frequency_spec_integrity_violation"

    intact_spec, records = complete_case
    result = summarize_selection_frequency(intact_spec, records)
    object.__setattr__(result, "selection_frequency", 0.75)
    with pytest.raises(ContractError) as caught:
        result.to_dict()
    assert caught.value.details.code == (
        "selection_frequency_result_integrity_violation"
    )


def test_result_is_producer_owned() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        SelectionFrequencyResult()
