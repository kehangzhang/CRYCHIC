from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.scoring import (
    PlannedScoringCollectionManifest,
    ReceiverScoringFunctionalManifest,
    ScoringCollectionDocument,
    ScoringCollectionManifest,
)
from crychic.workflow.certification import (
    CROSSFIT_OOF_DESCRIPTIVE_SCOPE,
    FORMAL_INFERENCE_DISABLED_REASON,
    CrossFitOOFCertificationAudit,
    CrossFitOOFRequirementRecord,
    audit_crossfit_oof_readiness,
)
from crychic.workflow.crossfit import CrossFitArtifacts
from crychic.workflow.receiver_universe import ReceiverTrainingSupportStatus


def _registry(
    *,
    functional_status: str = "observed",
    reason_code: str | None = None,
) -> ScoringCollectionDocument:
    child = ReceiverScoringFunctionalManifest.registered(
        receiver="Receiver",
        receiver_family_model_id="family-parent",
        receiver_incremental_model_id="incremental-model",
        filter_universe_id="universe-1",
        scoring_functional_id="common-functional",
        score_version="family-common-v3",
        functional_status=functional_status,
        reason_code=reason_code,
        receiver_training_support_id="receiver-support",
        receiver_training_support_status="observed",
    )
    collection = ScoringCollectionManifest.planned_receiver_registry(
        contrast="stim_vs_control",
        repeat_id="repeat-0",
        fold_id="fold-0",
        planned_receivers=("Receiver",),
        children=(child,),
    )
    return ScoringCollectionDocument(
        collections=(collection,),
        planned_collections=(
            PlannedScoringCollectionManifest.from_collection(
                collection,
                filter_universe_id="universe-1",
            ),
        ),
    )


def _fake_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    functional_status: str = "observed",
    functional_reason: str | None = None,
    tuning_present: bool = True,
    resource_trusted: bool = True,
    family_observed: bool = True,
    program_observed: bool = True,
    incremental_certified: bool = True,
    calibration_observed: bool = True,
    common_application_observed: bool = True,
    include_binding: bool = True,
    global_common_present: bool = True,
    subject_overlap: bool = False,
    functional_contexts: tuple[str, ...] = ("control", "stim"),
    family_score_contexts: tuple[str, ...] = ("control", "stim"),
    member_score_contexts: tuple[str, ...] = ("control", "stim"),
) -> CrossFitArtifacts:
    registry = _registry(
        functional_status=functional_status,
        reason_code=functional_reason,
    )
    training_subjects = ("p1", "p2")
    heldout_subjects = ("p1", "p3") if subject_overlap else ("p3", "p4")

    family_parent = SimpleNamespace(
        receiver="Receiver",
        training_artifact_id="family-parent",
    )
    family_model = SimpleNamespace(
        contrast_name="stim_vs_control",
        receiver_family_artifact=family_parent,
        training_artifact_id="family-model",
        training_subject_ids=training_subjects,
        downstream_functional=object() if family_observed else None,
        reason_code=(None if family_observed else "receiver_family_not_estimable"),
    )
    family_application = SimpleNamespace(
        training_artifact=family_model,
        training_artifact_id="family-model",
        heldout_subject_ids=heldout_subjects,
        application_id="family-application",
        downstream_application=object() if family_observed else None,
        reason_code=(
            None if family_observed else "receiver_family_application_not_estimable"
        ),
    )
    program_model = SimpleNamespace(
        contrast_name="stim_vs_control",
        receiver="Receiver",
        status="observed" if program_observed else "not_estimable",
        reason_code=(None if program_observed else "receiver_program_not_estimable"),
        training_subject_ids=training_subjects,
        training_artifact_id="program-model",
    )
    program_application = SimpleNamespace(
        training_artifact=program_model,
        status="observed" if program_observed else "not_estimable",
        reason_code=(
            None if program_observed else "receiver_program_application_not_estimable"
        ),
        heldout_subject_ids=heldout_subjects,
        application_id="program-application",
    )
    diagnostic_functional = SimpleNamespace(
        incremental_functional_id="outer-incremental-functional"
    )
    tuning_artifact = SimpleNamespace(tuning_id="inner-penalty-tuning")
    gain_calibration_spec = SimpleNamespace(spec_id="gain-calibration-spec")
    gain_calibration = SimpleNamespace(
        artifact_id="gain-calibration-artifact",
        spec=gain_calibration_spec,
        receiver="Receiver",
        contrast_name="stim_vs_control",
        outer_fold_id="fold-0",
        training_subject_ids=training_subjects,
        family_ids=("family-1",),
        feature_ids=("gene-1",),
        null_loss_floor=1e-8,
        tuning_id=tuning_artifact.tuning_id,
        outer_incremental_functional_id=(
            diagnostic_functional.incremental_functional_id
        ),
        outer_selected_resolved_penalty_id="resolved-penalty",
        status="observed" if calibration_observed else "not_estimable",
        is_estimable=calibration_observed,
        reason_code=(
            None if calibration_observed else "insufficient_inner_oof_subjects"
        ),
    )
    incremental_model = SimpleNamespace(
        contrast_name="stim_vs_control",
        receiver="Receiver",
        training_artifact_id="incremental-model",
        is_oof_certified=incremental_certified,
        official_incremental_status=(
            "observed" if incremental_certified else "not_estimable"
        ),
        reason_code=(
            None
            if incremental_certified
            else "trusted_autonomous_incremental_not_oof_certified"
        ),
        training_subject_ids=training_subjects,
        family_ids=("family-1",),
        feature_ids=("gene-1",),
        null_loss_floor=1e-8,
        fold_id="fold-0",
        diagnostic_reason_code=None,
        diagnostic_functional=diagnostic_functional,
        penalty_tuning_artifact=tuning_artifact,
        selected_resolved_penalty_id="resolved-penalty",
        gain_calibration_spec_id=gain_calibration_spec.spec_id,
        gain_calibration_artifact=gain_calibration,
    )
    diagnostic_application = SimpleNamespace(application_id="diagnostic-application")
    incremental_application = SimpleNamespace(
        is_oof_certified=incremental_certified,
        official_incremental_status=(
            "observed" if incremental_certified else "not_estimable"
        ),
        reason_code=(
            None if incremental_certified else "heldout_incremental_diagnostic_only"
        ),
        heldout_subject_ids=heldout_subjects,
        application_id="incremental-application",
        diagnostic_application=diagnostic_application,
    )
    registered_sender_functional = SimpleNamespace(
        contrast_name="stim_vs_control",
        contrast_manifest_id="contrast-manifest",
        context_ids=("control", "stim"),
        sender_functional_id="sender-functional",
    )
    common_functional = SimpleNamespace(
        contrast_name="stim_vs_control",
        receiver="Receiver",
        family_common_functional_id="common-functional",
        contrast_manifest_id="contrast-manifest",
        context_ids=functional_contexts,
        sender_functional=registered_sender_functional,
        incremental_functional=object(),
        incremental_reason_code=None,
        training_subject_ids=training_subjects,
    )
    common_application = SimpleNamespace(
        functional=common_functional,
        heldout_reason_code=(
            None
            if common_application_observed
            else "family_common_heldout_not_estimable"
        ),
        incremental_application_id=(
            "diagnostic-application" if common_application_observed else None
        ),
        heldout_subject_ids=heldout_subjects,
        application_id="common-application",
        family_scores=pd.DataFrame({"context_id": family_score_contexts}),
        member_scores=pd.DataFrame({"context_id": member_score_contexts}),
    )
    binding = SimpleNamespace(
        family_common_functional_id="common-functional",
        family_common_application_id="common-application",
        binding_id="common-binding",
    )
    global_common_functional = SimpleNamespace(
        contrast_name="stim_vs_control",
        fold_id="fold-0",
        receiver_ids=("Receiver",),
        child_functional_ids=("common-functional",),
        common_functional_across_receivers=False,
        receiver_balanced_descriptive_collection=True,
        spec=SimpleNamespace(
            schema_version="4.0.0",
            spec_id="global-common-spec",
        ),
        receiver_gain_calibrations=(("Receiver", gain_calibration),),
        all_receivers_gain_calibrated=calibration_observed,
        cross_receiver_percentile_rank_eligible=calibration_observed,
        gain_calibration_binding=lambda receiver: {
            "gain_calibration_binding_id": "gain-binding",
            "gain_calibration_artifact_id": gain_calibration.artifact_id,
            "gain_calibration_status": gain_calibration.status,
        },
        training_subject_ids=training_subjects,
        global_common_functional_id="global-common-functional",
    )
    global_common_application = SimpleNamespace(
        functional=global_common_functional,
        heldout_subject_ids=heldout_subjects,
        application_id="global-common-application",
        lr_scores_digest="global-lr-digest",
        sender_scores_digest="global-sender-digest",
    )
    fold = SimpleNamespace(
        fold_id="fold-0",
        training=SimpleNamespace(
            training_subject_ids=training_subjects,
            sender_functionals=(registered_sender_functional,),
        ),
        application=SimpleNamespace(heldout_subject_ids=heldout_subjects),
        receiver_training_support=(
            SimpleNamespace(
                receiver_id="Receiver",
                support_record_id="receiver-support",
                status=ReceiverTrainingSupportStatus.OBSERVED,
                reason_code=None,
            ),
        ),
        receiver_family_models=(family_model,),
        receiver_family_applications=(family_application,),
        receiver_program_models=(program_model,),
        receiver_program_applications=(program_application,),
        receiver_incremental_models=(incremental_model,),
        receiver_incremental_applications=(incremental_application,),
        family_common_functionals=(common_functional,),
        family_common_applications=(common_application,),
        family_common_bindings=((binding,) if include_binding else ()),
        cross_receiver_common_functionals=(
            (global_common_functional,) if global_common_present else ()
        ),
        cross_receiver_common_applications=(
            (global_common_application,) if global_common_present else ()
        ),
    )
    tuning = (
        SimpleNamespace(spec_id="tuning-spec", _require_intact=lambda: None)
        if tuning_present
        else None
    )
    resource = SimpleNamespace(
        artifact_id="autonomous-resource",
        is_manifest_verified_trusted=resource_trusted,
    )
    spec = SimpleNamespace(
        repeat_id="repeat-0",
        penalty_tuning_spec=tuning,
        autonomous_program_resource=resource,
    )
    artifacts = object.__new__(CrossFitArtifacts)
    object.__setattr__(artifacts, "crossfit_id", "crossfit-source")
    object.__setattr__(artifacts, "spec", spec)
    object.__setattr__(artifacts, "folds", (fold,))
    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", lambda self: None)
    monkeypatch.setattr(
        CrossFitArtifacts,
        "receiver_scoring_registry",
        property(lambda self: registry),
    )
    return artifacts


def _failed_by_code(audit: CrossFitOOFCertificationAudit) -> dict[str, str]:
    return {
        record.requirement_code: record.reason_code or ""
        for record in audit.failed_requirements
    }


def _parser_audit() -> CrossFitOOFCertificationAudit:
    observed = CrossFitOOFRequirementRecord(
        requirement_code="observed_requirement",
        status="satisfied",
        reason_code=None,
        repeat_id="repeat-0",
        fold_id="fold-1",
        contrast="stim_vs_control",
        receiver="Receiver",
        evidence_ids=("evidence-z", "evidence-a"),
    )
    unavailable = CrossFitOOFRequirementRecord(
        requirement_code="unavailable_requirement",
        status="not_satisfied",
        reason_code="missing_evidence",
        repeat_id="repeat-0",
        fold_id="fold-2",
    )
    return CrossFitOOFCertificationAudit._from_requirements(
        source_crossfit_id="crossfit-source",
        source_registry_id="registry-source",
        requirements=(unavailable, observed),
    )


def test_complete_chain_is_descriptive_oof_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = audit_crossfit_oof_readiness(_fake_artifacts(monkeypatch))

    assert audit.complete
    assert audit.is_oof_descriptive_certified
    assert audit.certification_scope == CROSSFIT_OOF_DESCRIPTIVE_SCOPE
    assert audit.formal_inference_allowed is False
    assert audit.formal_inference_reason_code == FORMAL_INFERENCE_DISABLED_REASON
    assert audit.failed_requirements == ()
    assert len(audit.requirement_ledger) == 13
    assert len({row.requirement_id for row in audit.requirement_ledger}) == 13
    context_requirement = next(
        row
        for row in audit.requirement_ledger
        if row.requirement_code == "family_common_contrast_context_contract_complete"
    )
    assert context_requirement.satisfied
    assert audit.to_dict()["formal_inference_allowed"] is False


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("changes", "reason"),
    [
        (
            {"functional_contexts": ("control",)},
            "family_common_functional_context_mismatch",
        ),
        (
            {"family_score_contexts": ("control",)},
            "family_common_application_family_context_mismatch",
        ),
        (
            {"member_score_contexts": ("control",)},
            "family_common_application_member_context_mismatch",
        ),
    ],
)
def test_family_common_context_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    reason: str,
) -> None:
    audit = audit_crossfit_oof_readiness(_fake_artifacts(monkeypatch, **changes))

    assert not audit.complete
    assert not audit.is_oof_descriptive_certified
    assert audit.formal_inference_allowed is False
    assert _failed_by_code(audit)[
        "family_common_contrast_context_contract_complete"
    ] == reason


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("changes", "requirement", "reason"),
    [
        (
            {"tuning_present": False},
            "subject_blocked_penalty_tuning_policy_present",
            "penalty_tuning_spec_missing",
        ),
        (
            {"resource_trusted": False},
            "approved_receiver_autonomous_nuisance_policy_present",
            "autonomous_resource_untrusted",
        ),
        (
            {"family_observed": False},
            "receiver_family_train_apply_chain_complete",
            "receiver_family_not_estimable",
        ),
        (
            {"program_observed": False},
            "receiver_program_train_apply_chain_observed",
            "receiver_program_not_estimable",
        ),
        (
            {"incremental_certified": False},
            "receiver_incremental_train_apply_chain_oof_observed",
            "trusted_autonomous_incremental_not_oof_certified",
        ),
        (
            {"calibration_observed": False},
            "selected_penalty_inner_oof_gain_calibration_observed",
            "insufficient_inner_oof_subjects",
        ),
        (
            {
                "functional_status": "not_estimable",
                "functional_reason": "incremental_training_not_estimable",
            },
            "planned_receiver_child_observed",
            "incremental_training_not_estimable",
        ),
        (
            {"include_binding": False},
            "family_common_train_apply_binding_chain_complete",
            "family_common_binding_missing",
        ),
        (
            {"global_common_present": False},
            "cross_receiver_common_train_apply_chain_complete",
            "cross_receiver_common_functional_missing",
        ),
        (
            {"common_application_observed": False},
            "family_common_train_apply_binding_chain_complete",
            "family_common_heldout_not_estimable",
        ),
        (
            {"subject_overlap": True},
            "training_apply_subjects_disjoint",
            "training_heldout_subject_overlap",
        ),
    ],
)
def test_incomplete_chains_fail_closed_with_stable_reasons(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    requirement: str,
    reason: str,
) -> None:
    audit = audit_crossfit_oof_readiness(_fake_artifacts(monkeypatch, **changes))

    assert not audit.complete
    assert not audit.is_oof_descriptive_certified
    assert audit.formal_inference_allowed is False
    assert _failed_by_code(audit)[requirement] == reason


def test_requirement_order_does_not_change_audit_identity() -> None:
    first = CrossFitOOFRequirementRecord(
        requirement_code="z_requirement",
        status="satisfied",
        reason_code=None,
        repeat_id="repeat-0",
    )
    second = CrossFitOOFRequirementRecord(
        requirement_code="a_requirement",
        status="not_satisfied",
        reason_code="missing_evidence",
        repeat_id="repeat-0",
    )

    forward = CrossFitOOFCertificationAudit._from_requirements(
        source_crossfit_id="crossfit-source",
        source_registry_id="registry-source",
        requirements=(first, second),
    )
    reverse = CrossFitOOFCertificationAudit._from_requirements(
        source_crossfit_id="crossfit-source",
        source_registry_id="registry-source",
        requirements=(second, first),
    )

    assert forward.audit_id == reverse.audit_id
    assert forward.requirement_ledger == reverse.requirement_ledger
    assert not forward.complete


def test_requirement_and_audit_strict_parser_round_trip() -> None:
    audit = _parser_audit()
    record = audit.requirement_ledger[0]

    assert CrossFitOOFRequirementRecord.from_dict(record.to_dict()) == record
    assert CrossFitOOFCertificationAudit.from_dict(audit.to_dict()) == audit
    assert audit.requirement_ledger[0].evidence_ids == (
        "evidence-a",
        "evidence-z",
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field_name", "replacement"),
    [
        ("requirement_code", "forged_requirement"),
        ("requirement_id", "forged-requirement-id"),
        ("record_id", "forged-record-id"),
        ("repeat_id", "forged-repeat"),
        ("receiver", "forged-receiver"),
    ],
)
def test_requirement_parser_rejects_payload_and_id_tampering(
    field_name: str,
    replacement: object,
) -> None:
    serialized = _parser_audit().requirement_ledger[0].to_dict()
    serialized[field_name] = replacement

    with pytest.raises(ValueError, match=r"payload|identity"):
        CrossFitOOFRequirementRecord.from_dict(serialized)


def test_requirement_parser_rejects_noncanonical_evidence_and_fields() -> None:
    serialized = _parser_audit().requirement_ledger[0].to_dict()
    reversed_evidence = copy.deepcopy(serialized)
    reversed_evidence["evidence_ids"] = list(
        reversed(cast(list[object], serialized["evidence_ids"]))
    )
    duplicate_evidence = copy.deepcopy(serialized)
    duplicate_evidence["evidence_ids"] = ["evidence-a", "evidence-a", "evidence-z"]
    extra_field = copy.deepcopy(serialized)
    extra_field["unexpected"] = True
    missing_field = copy.deepcopy(serialized)
    missing_field.pop("requirement_id")

    for poisoned in (
        reversed_evidence,
        duplicate_evidence,
        extra_field,
        missing_field,
    ):
        with pytest.raises(ValueError):
            CrossFitOOFRequirementRecord.from_dict(poisoned)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field_name", "replacement"),
    [
        ("audit_id", "forged-audit-id"),
        ("complete", True),
        ("formal_inference_allowed", True),
        ("formal_inference_reason_code", "forged-formal-reason"),
        ("certification_scope", "forged-scope"),
        ("schema_version", "2.0.0"),
        ("source_crossfit_id", "forged-crossfit"),
        ("source_registry_id", "forged-registry"),
    ],
)
def test_audit_parser_rejects_scope_version_source_and_id_tampering(
    field_name: str,
    replacement: object,
) -> None:
    serialized = _parser_audit().to_dict()
    serialized[field_name] = replacement

    with pytest.raises(ValueError):
        CrossFitOOFCertificationAudit.from_dict(serialized)


def test_audit_parser_rejects_ledger_order_nested_payload_and_field_drift() -> None:
    serialized = _parser_audit().to_dict()
    reversed_ledger = copy.deepcopy(serialized)
    reversed_ledger["requirement_ledger"] = list(
        reversed(cast(list[object], serialized["requirement_ledger"]))
    )
    nested_payload = copy.deepcopy(serialized)
    nested_payload["requirement_ledger"][0][  # type: ignore[index]
        "requirement_code"
    ] = "forged-nested-requirement"
    extra_field = copy.deepcopy(serialized)
    extra_field["unexpected"] = True
    missing_field = copy.deepcopy(serialized)
    missing_field.pop("audit_id")

    for poisoned in reversed_ledger, nested_payload, extra_field, missing_field:
        with pytest.raises(ValueError):
            CrossFitOOFCertificationAudit.from_dict(poisoned)


def test_audit_and_source_tampering_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _fake_artifacts(monkeypatch)
    audit = audit_crossfit_oof_readiness(artifacts)
    object.__setattr__(audit, "complete", False)
    with pytest.raises(ContractError, match="integrity"):
        audit.to_dict()

    def reject_source(self: CrossFitArtifacts) -> None:
        raise ContractError(
            "source changed",
            code="source_changed",
            field="crossfit_id",
            remediation="rerun",
        )

    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", reject_source)
    with pytest.raises(ContractError, match="source changed"):
        audit_crossfit_oof_readiness(artifacts)


def test_nonproducer_object_is_rejected() -> None:
    with pytest.raises(TypeError, match="producer-owned CrossFitArtifacts"):
        audit_crossfit_oof_readiness(SimpleNamespace())
