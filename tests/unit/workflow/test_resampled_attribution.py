from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.inference.hypotheses import (
    HypothesisDeclaration,
    HypothesisRole,
    freeze_hypothesis_universe,
)
from crychic.resampling import ContextPermutationPlan, SubjectBootstrapPlan
from crychic.scoring import (
    FamilyCommonScoringApplication,
    FamilyCommonScoringFunctional,
    IncrementalDownstreamFunctional,
)
from crychic.workflow.crossfit import CrossFitArtifacts, CrossFitFoldArtifacts
from crychic.workflow.full_pipeline_effects import (
    FrozenFamilyEffectTarget,
    build_frozen_family_effect_target,
)
from crychic.workflow.full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResampleStatus,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
)
from crychic.workflow.resampled_attribution import (
    FrozenResampledAttributionCollection,
    ResampledAttributionEventStatus,
    ResampledFamilySelectionEvent,
    summarize_resampled_family_selection,
)

_PRIMARY_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_EFFECT_ENDPOINT = "family_common_integrated_lr_context_effect_v1"


@pytest.fixture(autouse=True)
def _stub_large_parent_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        FullPipelineResamplingResult,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        FullPipelineResampleRecord,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", lambda self: None)
    monkeypatch.setattr(
        FamilyCommonScoringFunctional,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        FamilyCommonScoringApplication,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        IncrementalDownstreamFunctional,
        "_require_intact",
        lambda self: None,
    )


def _target(*, endpoint: str = _PRIMARY_ENDPOINT) -> FrozenFamilyEffectTarget:
    declaration = HypothesisDeclaration(
        endpoint=endpoint,
        contrast_name="all_contexts_v1",
        receiver="Receiver",
        family_id="family-1",
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="gene-state-primary",
    )
    universe = freeze_hypothesis_universe(
        (declaration,),
        universe_name=f"resampled-attribution-{endpoint}",
    )
    return build_frozen_family_effect_target(
        universe,
        declaration.hypothesis_id,
    )


def _attribution_row(
    functional_id: str,
    *,
    estimable: object,
    selected: object,
    selection_status: object,
    reason_code: object = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "family_common_functional_id": [functional_id],
            "family_id": ["family-1"],
            "family_estimable": [estimable],
            "family_selected": [selected],
            "selection_status": [selection_status],
            "reason_code": [reason_code],
        }
    )


def _chain(
    identifier: str,
    attribution: object,
    *,
    family_ids: tuple[str, ...] = ("family-1",),
    active_family_ids: tuple[str, ...] | None = None,
    coefficient: float = 1.0,
    estimable: bool = True,
    training_reason_code: str | None = None,
    incremental_missing: bool = False,
    score_version: str = "family-common-score-v1",
) -> tuple[FamilyCommonScoringFunctional, FamilyCommonScoringApplication]:
    active = family_ids if active_family_ids is None else active_family_ids
    incremental: IncrementalDownstreamFunctional | None
    if incremental_missing:
        incremental = None
    else:
        incremental = object.__new__(IncrementalDownstreamFunctional)
        object.__setattr__(incremental, "family_ids", active)
        object.__setattr__(
            incremental,
            "family_estimable",
            np.asarray([estimable] * len(active), dtype=bool),
        )
        object.__setattr__(
            incremental,
            "family_coefficients",
            np.asarray([coefficient] * len(active), dtype=float),
        )
        object.__setattr__(
            incremental,
            "family_reason_codes",
            tuple(training_reason_code for _ in active),
        )
        object.__setattr__(
            incremental,
            "incremental_functional_id",
            f"incremental-{identifier}",
        )
    functional = object.__new__(FamilyCommonScoringFunctional)
    object.__setattr__(functional, "contrast_name", "all_contexts_v1")
    object.__setattr__(functional, "receiver", "Receiver")
    object.__setattr__(functional, "family_ids", family_ids)
    object.__setattr__(functional, "active_family_ids", active)
    object.__setattr__(functional, "incremental_functional", incremental)
    object.__setattr__(
        functional,
        "incremental_reason_code",
        "incremental_parent_missing" if incremental_missing else None,
    )
    object.__setattr__(functional, "score_version", score_version)
    object.__setattr__(functional, "tuning_manifest_id", f"tuning-{identifier}")
    object.__setattr__(functional, "selected_penalty_id", f"penalty-{identifier}")
    object.__setattr__(
        functional,
        "family_common_functional_id",
        f"functional-{identifier}",
    )
    application = object.__new__(FamilyCommonScoringApplication)
    object.__setattr__(application, "functional", functional)
    object.__setattr__(application, "application_id", f"application-{identifier}")
    object.__setattr__(application, "_family_attribution", attribution)
    return functional, application


def _fold(
    identifier: str,
    heldout_subject_ids: tuple[str, ...],
    *,
    attribution: object | None = None,
    family_ids: tuple[str, ...] = ("family-1",),
    include_chain: bool = True,
    active_family_ids: tuple[str, ...] | None = None,
    coefficient: float = 1.0,
    estimable: bool = True,
    training_reason_code: str | None = None,
    incremental_missing: bool = False,
    score_version: str = "family-common-score-v1",
) -> CrossFitFoldArtifacts:
    fold = object.__new__(CrossFitFoldArtifacts)
    object.__setattr__(fold, "fold_id", identifier)
    object.__setattr__(
        fold,
        "application",
        SimpleNamespace(heldout_subject_ids=heldout_subject_ids),
    )
    if include_chain:
        functional, application = _chain(
            identifier,
            attribution,
            family_ids=family_ids,
            active_family_ids=active_family_ids,
            coefficient=coefficient,
            estimable=estimable,
            training_reason_code=training_reason_code,
            incremental_missing=incremental_missing,
            score_version=score_version,
        )
        object.__setattr__(fold, "family_common_functionals", (functional,))
        object.__setattr__(fold, "family_common_applications", (application,))
    else:
        object.__setattr__(fold, "family_common_functionals", ())
        object.__setattr__(fold, "family_common_applications", ())
    return fold


def _child(
    identifier: str,
    folds: tuple[CrossFitFoldArtifacts, ...],
    *,
    crossfit_spec_id: str = "crossfit-spec",
) -> CrossFitArtifacts:
    child = object.__new__(CrossFitArtifacts)
    object.__setattr__(child, "crossfit_id", identifier)
    object.__setattr__(child, "folds", folds)
    object.__setattr__(child, "spec", SimpleNamespace(spec_id=crossfit_spec_id))
    return child


def _bootstrap_plan(index: int, *, n_draws: int = 1) -> SubjectBootstrapPlan:
    plan = object.__new__(SubjectBootstrapPlan)
    object.__setattr__(plan, "bootstrap_id", f"bootstrap-{index}")
    object.__setattr__(plan, "exchangeability_id", "exchangeability-v1")
    object.__setattr__(plan, "resample_index", index)
    object.__setattr__(plan, "seed_lineage", f"seed-{index}")
    object.__setattr__(
        plan,
        "draws",
        tuple(SimpleNamespace(draw_index=draw) for draw in range(n_draws)),
    )
    return plan


def _record(
    plan: SubjectBootstrapPlan,
    *,
    child: CrossFitArtifacts | None,
    failed: bool = False,
) -> FullPipelineResampleRecord:
    record = object.__new__(FullPipelineResampleRecord)
    object.__setattr__(
        record,
        "operation",
        FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP,
    )
    object.__setattr__(record, "plan_id", plan.bootstrap_id)
    object.__setattr__(record, "resample_index", plan.resample_index)
    object.__setattr__(record, "exchangeability_id", plan.exchangeability_id)
    object.__setattr__(record, "plan_seed_lineage", plan.seed_lineage)
    object.__setattr__(
        record,
        "status",
        (
            FullPipelineResampleStatus.FAILED
            if failed
            else FullPipelineResampleStatus.SUCCEEDED
        ),
    )
    object.__setattr__(
        record, "crossfit_id", None if child is None else child.crossfit_id
    )
    object.__setattr__(
        record,
        "failure_code",
        "synthetic_workflow_failure" if failed else None,
    )
    object.__setattr__(record, "_child", child)
    object.__setattr__(record, "record_id", f"record-{plan.bootstrap_id}")
    return record


def _resampling(
    pairs: tuple[tuple[object, FullPipelineResampleRecord], ...],
    *,
    retain_children: bool = True,
) -> FullPipelineResamplingResult:
    result = object.__new__(FullPipelineResamplingResult)
    object.__setattr__(result, "plans", tuple(plan for plan, _ in pairs))
    object.__setattr__(result, "records", tuple(record for _, record in pairs))
    object.__setattr__(result, "retain_children", retain_children)
    object.__setattr__(result, "result_id", "resampling-result-v1")
    object.__setattr__(result, "crossfit_spec_id", "crossfit-spec")
    return result


def test_exact_events_and_two_conditional_frequencies() -> None:
    first_plan = _bootstrap_plan(0, n_draws=3)
    second_plan = _bootstrap_plan(1, n_draws=5)
    first_child = _child(
        "child-0",
        (
            _fold(
                "fold-a",
                ("s1", "s2"),
                attribution=_attribution_row(
                    "functional-fold-a",
                    estimable=True,
                    selected=False,
                    selection_status="not_selected",
                ),
            ),
            _fold(
                "fold-b",
                ("s3",),
                attribution=_attribution_row(
                    "functional-fold-b",
                    estimable=True,
                    selected=True,
                    selection_status="selected",
                ),
                coefficient=0.0,
            ),
        ),
    )
    second_child = _child(
        "child-1",
        (
            _fold(
                "fold-a",
                ("s4", "s5"),
                attribution=_attribution_row(
                    "functional-fold-a",
                    estimable=False,
                    selected=False,
                    selection_status="structural_zero",
                    reason_code="receptor_family_ineligible",
                ),
                coefficient=0.0,
                estimable=False,
                training_reason_code="training_family_not_identifiable",
            ),
            _fold(
                "fold-b",
                ("s6", "s7", "s8"),
                attribution=_attribution_row(
                    "functional-fold-b",
                    estimable=True,
                    selected=True,
                    selection_status="selected",
                ),
            ),
        ),
    )
    result = summarize_resampled_family_selection(
        _resampling(
            (
                (first_plan, _record(first_plan, child=first_child)),
                (second_plan, _record(second_plan, child=second_child)),
            )
        ),
        _target(),
    )

    assert result.n_bootstrap_plans_total == 2
    assert result.n_bootstrap_plans_succeeded == 2
    assert result.n_event_rows_total == 4
    assert result.n_observed_outer_fold_events == 3
    assert result.n_not_estimable_outer_fold_events == 1
    assert result.n_failed_event_rows == 0
    assert result.n_selected_observed_outer_fold_events == 2
    assert result.outer_fold_denominator_complete is True
    assert (
        result.diagnostic_conditional_observed_outer_fold_selection_frequency
        == pytest.approx(2 / 3)
    )
    assert result.n_subject_exposures_total == 8
    assert result.n_subject_exposures_estimable == 6
    assert result.n_subject_exposures_not_estimable == 2
    assert result.n_subject_exposures_failed == 0
    assert result.n_subject_exposures_selected == 5
    assert (
        result.diagnostic_subject_exposure_weighted_selection_frequency
        == pytest.approx(5 / 6)
    )
    unavailable = next(
        event
        for event in result.events
        if event.status is ResampledAttributionEventStatus.NOT_ESTIMABLE
    )
    assert unavailable.family_estimable is False
    assert unavailable.family_selected is None
    assert unavailable.reason_code == "training_family_not_identifiable"
    observed = result.events[0]
    assert observed.resampling_result_id == "resampling-result-v1"
    assert observed.full_pipeline_record_id == "record-bootstrap-0"
    assert observed.crossfit_id == "child-0"
    assert observed.target_id == result.target_id
    assert observed.scoring_functional_id == "functional-fold-a"
    assert observed.family_selected is True
    assert observed.family_coefficient == pytest.approx(1.0)
    assert observed.selection_threshold == 0.0
    assert observed.score_version == "family-common-score-v1"
    training_unselected = next(
        event
        for event in result.events_for_plan(first_plan.bootstrap_id)
        if event.fold_id == "fold-b"
    )
    assert training_unselected.family_selected is False
    assert training_unselected.family_coefficient == 0.0
    assert observed.p_value is None
    assert observed.q_value is None
    assert observed.communication_probability is None
    assert result.formal_inference_allowed is False
    assert result.diagnostic_status == "diagnostic_partial"
    with pytest.raises(TypeError, match="producer-owned"):
        ResampledFamilySelectionEvent()
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenResampledAttributionCollection()


def test_missing_chain_and_family_are_not_estimable_not_nonselection() -> None:
    plan = _bootstrap_plan(0, n_draws=3)
    child = _child(
        "child-ne",
        (
            _fold("fold-chain-missing", ("s1",), include_chain=False),
            _fold(
                "fold-family-missing",
                ("s2", "s3"),
                attribution=pd.DataFrame(),
                family_ids=("family-other",),
            ),
        ),
    )
    result = summarize_resampled_family_selection(
        _resampling(((plan, _record(plan, child=child)),)),
        _target(),
    )

    assert result.n_event_rows_total == 2
    assert result.n_observed_outer_fold_events == 0
    assert result.n_not_estimable_outer_fold_events == 2
    assert result.n_selected_observed_outer_fold_events == 0
    assert result.diagnostic_conditional_observed_outer_fold_selection_frequency is None
    assert result.n_subject_exposures_not_estimable == 3
    assert all(event.family_selected is None for event in result.events)
    assert {event.reason_code for event in result.events} == {
        "family_common_target_chain_missing",
        "family_not_in_training_fold_universe",
    }


def test_failed_workflow_plan_retains_one_foldless_failed_event() -> None:
    plan = _bootstrap_plan(0, n_draws=4)
    result = summarize_resampled_family_selection(
        _resampling(((plan, _record(plan, child=None, failed=True)),)),
        _target(),
    )

    assert result.bootstrap_plan_ids == (plan.bootstrap_id,)
    assert result.n_bootstrap_plans_failed == 1
    assert result.n_event_rows_total == 1
    assert result.n_failed_event_rows == 1
    assert result.n_foldless_failed_plan_events == 1
    assert result.outer_fold_denominator_complete is False
    event = result.events_for_plan(plan.bootstrap_id)[0]
    assert event.fold_id is None
    assert event.crossfit_id is None
    assert event.status is ResampledAttributionEventStatus.FAILED
    assert event.reason_code == "synthetic_workflow_failure"
    assert event.n_subject_exposures == 4
    assert result.n_subject_exposures_total == 4
    assert result.n_subject_exposures_failed == 4


def test_adapter_exception_is_failed_and_preserves_known_fold_exposure() -> None:
    plan = _bootstrap_plan(0, n_draws=2)
    child = _child(
        "child-broken-table",
        (
            _fold(
                "fold-broken",
                ("s1", "s2"),
                attribution=pd.DataFrame(),
                coefficient=float("nan"),
            ),
        ),
    )
    result = summarize_resampled_family_selection(
        _resampling(((plan, _record(plan, child=child)),)),
        _target(),
    )

    assert result.n_failed_event_rows == 1
    assert result.n_observed_outer_fold_events == 0
    assert result.n_subject_exposures_failed == 2
    event = result.events[0]
    assert event.fold_id == "fold-broken"
    assert event.status is ResampledAttributionEventStatus.FAILED
    assert event.reason_code == "resampled_attribution_adapter_failed_ValueError"
    assert event.family_selected is None


def test_bootstrap_only_retention_and_primary_target_contracts() -> None:
    plan = _bootstrap_plan(0)
    child = _child(
        "child-valid",
        (
            _fold(
                "fold-valid",
                ("s1",),
                attribution=_attribution_row(
                    "functional-fold-valid",
                    estimable=True,
                    selected=True,
                    selection_status="selected",
                ),
            ),
        ),
    )
    record = _record(plan, child=child)
    with pytest.raises(ContractError) as retention_error:
        summarize_resampled_family_selection(
            _resampling(((plan, record),), retain_children=False),
            _target(),
        )
    assert retention_error.value.details.code == (
        "resampled_attribution_children_not_retained"
    )

    context_plan = object.__new__(ContextPermutationPlan)
    object.__setattr__(context_plan, "permutation_id", "permutation-0")
    object.__setattr__(context_plan, "resample_index", 0)
    object.__setattr__(context_plan, "exchangeability_id", "exchangeability-v1")
    object.__setattr__(context_plan, "seed_lineage", "seed-context")
    context_record = object.__new__(FullPipelineResampleRecord)
    object.__setattr__(
        context_record,
        "operation",
        FullPipelineResamplingOperation.CONTEXT_PERMUTATION,
    )
    object.__setattr__(context_record, "plan_id", "permutation-0")
    object.__setattr__(context_record, "resample_index", 0)
    object.__setattr__(context_record, "exchangeability_id", "exchangeability-v1")
    object.__setattr__(context_record, "plan_seed_lineage", "seed-context")
    object.__setattr__(context_record, "record_id", "record-permutation-0")
    object.__setattr__(context_record, "_child", child)
    with pytest.raises(ContractError) as context_error:
        summarize_resampled_family_selection(
            _resampling(((context_plan, context_record),)),
            _target(),
        )
    assert context_error.value.details.code == (
        "resampled_attribution_nonbootstrap_plan_forbidden"
    )

    with pytest.raises(ContractError) as target_error:
        summarize_resampled_family_selection(
            _resampling(((plan, record),)),
            _target(endpoint=_EFFECT_ENDPOINT),
        )
    assert target_error.value.details.code == (
        "resampled_attribution_primary_target_mismatch"
    )


def test_collection_integrity_detects_summary_tampering() -> None:
    plan = _bootstrap_plan(0, n_draws=2)
    child = _child(
        "child-tamper",
        (
            _fold(
                "fold-tamper",
                ("s1", "s2"),
                attribution=_attribution_row(
                    "functional-fold-tamper",
                    estimable=True,
                    selected=True,
                    selection_status="selected",
                ),
            ),
        ),
    )
    result = summarize_resampled_family_selection(
        _resampling(((plan, _record(plan, child=child)),)),
        _target(),
    )

    object.__setattr__(
        result,
        "diagnostic_conditional_observed_outer_fold_selection_frequency",
        0.0,
    )
    with pytest.raises(ContractError) as error:
        result.to_dict()
    assert error.value.details.code == (
        "resampled_attribution_collection_integrity_violation"
    )


def test_child_crossfit_spec_and_observed_score_rule_mismatch_fail_closed() -> None:
    mismatched_plan = _bootstrap_plan(0)
    mismatched_child = _child(
        "child-wrong-spec",
        (
            _fold(
                "fold-wrong-spec",
                ("s1",),
                attribution=pd.DataFrame(),
            ),
        ),
        crossfit_spec_id="different-crossfit-spec",
    )
    mismatched = summarize_resampled_family_selection(
        _resampling(
            ((mismatched_plan, _record(mismatched_plan, child=mismatched_child)),)
        ),
        _target(),
    )
    assert mismatched.crossfit_spec_id == "crossfit-spec"
    assert mismatched.events[0].status is ResampledAttributionEventStatus.FAILED
    assert mismatched.events[0].reason_code == (
        "resampled_attribution_child_crossfit_spec_mismatch"
    )

    rule_plan = _bootstrap_plan(1, n_draws=2)
    rule_child = _child(
        "child-rule-mismatch",
        (
            _fold(
                "fold-rule-a",
                ("s1",),
                attribution=pd.DataFrame(),
                score_version="score-version-a",
            ),
            _fold(
                "fold-rule-b",
                ("s2",),
                attribution=pd.DataFrame(),
                score_version="score-version-b",
            ),
        ),
    )
    with pytest.raises(ContractError) as rule_error:
        summarize_resampled_family_selection(
            _resampling(((rule_plan, _record(rule_plan, child=rule_child)),)),
            _target(),
        )
    assert rule_error.value.details.code == (
        "resampled_attribution_observed_rule_mismatch"
    )


def test_event_integrity_binds_target_heldout_and_workflow_failure_reason() -> None:
    def observed_result() -> FrozenResampledAttributionCollection:
        plan = _bootstrap_plan(0, n_draws=2)
        child = _child(
            "child-event-integrity",
            (
                _fold(
                    "fold-event-integrity",
                    ("s1", "s2"),
                    attribution=_attribution_row(
                        "functional-fold-event-integrity",
                        estimable=True,
                        selected=True,
                        selection_status="selected",
                    ),
                ),
            ),
        )
        return summarize_resampled_family_selection(
            _resampling(((plan, _record(plan, child=child)),)),
            _target(),
        )

    target_tampered = observed_result().events[0]
    object.__setattr__(target_tampered, "contrast_name", "forged-contrast")
    with pytest.raises(ContractError) as target_error:
        target_tampered.to_dict()
    assert target_error.value.details.code == (
        "resampled_attribution_event_integrity_violation"
    )

    heldout_tampered = observed_result().events[0]
    object.__setattr__(heldout_tampered, "heldout_subject_ids", ("forged", "s2"))
    with pytest.raises(ContractError) as heldout_error:
        heldout_tampered.to_dict()
    assert heldout_error.value.details.code == (
        "resampled_attribution_event_integrity_violation"
    )

    rule_tampered = observed_result().events[0]
    object.__setattr__(rule_tampered, "selection_rule_id", "forged-rule")
    with pytest.raises(ContractError) as rule_error:
        rule_tampered.to_dict()
    assert rule_error.value.details.code == (
        "resampled_attribution_event_integrity_violation"
    )

    score_tampered = observed_result().events[0]
    object.__setattr__(score_tampered, "score_version", "forged-score-version")
    with pytest.raises(ContractError) as score_error:
        score_tampered.to_dict()
    assert score_error.value.details.code == (
        "resampled_attribution_event_integrity_violation"
    )

    failed_plan = _bootstrap_plan(0)
    failed = summarize_resampled_family_selection(
        _resampling(((failed_plan, _record(failed_plan, child=None, failed=True)),)),
        _target(),
    ).events[0]
    object.__setattr__(failed, "reason_code", "forged-workflow-reason")
    with pytest.raises(ContractError) as reason_error:
        failed.to_dict()
    assert reason_error.value.details.code == (
        "resampled_attribution_event_integrity_violation"
    )
