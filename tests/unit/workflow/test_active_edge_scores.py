from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from crychic.core import CommunicationMode, ContractError, stable_id
from crychic.design import balanced_contrast
from crychic.inference import ActiveEdgeScoreStatus
from crychic.scoring import (
    GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    CrossReceiverCommonScoringApplication,
    CrossReceiverCommonScoringFunctional,
    CrossReceiverCommonScoringSpec,
)
from crychic.sender import SenderPrevalencePrior, SenderPrevalenceStatus
from crychic.workflow.active_edge_scores import (
    adapt_crossfit_active_edge_point_records,
    adapt_crossfit_active_edge_records_against_universe,
    freeze_crossfit_active_edge_universe,
)
from crychic.workflow.certification import (
    CrossFitOOFCertificationAudit,
    CrossFitOOFRequirementRecord,
)
from crychic.workflow.crossfit import CrossFitArtifacts

_SCORE_VERSION = "receiver_gain_percentile_mechanistic_conserved_sender_v4"
_CONTRAST_NAME = "treated-vs-control"
_DEFAULT_COVERAGE = (
    (
        ("control-tech-1", "subject-1", "control"),
        ("treated-tech-1a", "subject-1", "treated"),
        ("treated-tech-1b", "subject-1", "treated"),
    ),
    (
        ("control-tech-2", "subject-2", "control"),
        ("treated-tech-2", "subject-2", "treated"),
    ),
)
_DEFAULT_FOLD_CANDIDATES = (
    (("lr-1", "Sender", "driver-lr-1"),),
    (("lr-1", "Sender", "driver-lr-1"),),
)


def _sender_score_row(
    *,
    functional_id: str,
    sender_functional_id: str,
    fold_id: str,
    sample_id: str,
    subject_id: str,
    context_id: str,
    interaction_id: str,
    sender: str,
    driver_id: str,
    mode: str,
    score: float,
) -> dict[str, object]:
    values: dict[str, object] = {
        column: None for column in GLOBAL_COMMON_SENDER_SCORE_COLUMNS
    }
    values.update(
        {
            "global_common_functional_id": functional_id,
            "source_family_common_functional_id": f"family-functional-{fold_id}",
            "source_family_common_application_id": f"family-application-{fold_id}",
            "source_sender_functional_id": sender_functional_id,
            "source_sender_application_id": f"sender-application-{fold_id}",
            "sample_id": sample_id,
            "subject_id": subject_id,
            "context_id": context_id,
            "receiver": "Receiver",
            "family_id": "family-1",
            "driver_id": driver_id,
            "interaction_id": interaction_id,
            "mode": mode,
            "sender": sender,
            "global_lr_score": score,
            "gain_calibration_status": "observed",
            "ligand_availability": score,
            "training_prevalence_prior": 0.5,
            "raw_sender_evidence": score,
            "global_sender_lr_score": score,
            "status": "observed",
            "reason_code": None,
            "score_version": _SCORE_VERSION,
        }
    )
    return values


@pytest.fixture
def artifact_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., CrossFitArtifacts]:
    audits: dict[str, CrossFitOOFCertificationAudit] = {}
    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", lambda self: None)
    monkeypatch.setattr(
        CrossReceiverCommonScoringFunctional,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        CrossReceiverCommonScoringApplication,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        CrossFitArtifacts,
        "oof_certification_audit",
        property(lambda self: audits[self.crossfit_id]),
    )

    def build(
        *,
        tag: str,
        fold_candidates: tuple[
            tuple[tuple[str, str, str], ...], ...
        ] = _DEFAULT_FOLD_CANDIDATES,
        score_overrides: dict[tuple[str, str, str, str], float] | None = None,
        missing_rows: set[tuple[str, str, str, str]] | None = None,
        audit_complete: bool = True,
    ) -> CrossFitArtifacts:
        contrast = balanced_contrast(
            ("treated",),
            ("control",),
            name=_CONTRAST_NAME,
        )
        contrast_manifest_id = stable_id("contrast_manifest", contrast.to_dict())
        design_contrast_id = stable_id("contrast", contrast.to_dict())
        scoring_spec = CrossReceiverCommonScoringSpec()
        overrides = {} if score_overrides is None else score_overrides
        omitted = set() if missing_rows is None else missing_rows
        folds: list[SimpleNamespace] = []
        coverage_rows: list[dict[str, object]] = []

        for fold_index, candidates in enumerate(fold_candidates, start=1):
            fold_id = f"fold-{fold_index}"
            sender_functional_id = f"sender-functional-{tag}-{fold_id}"
            functional_id = f"global-functional-{tag}-{fold_id}"
            priors = tuple(
                SenderPrevalencePrior(
                    receiver="Receiver",
                    interaction_id=interaction_id,
                    sender=sender,
                    prevalence_prior=0.5,
                    n_subjects=2,
                    status=SenderPrevalenceStatus.SUPPORTED,
                    reason_code=None,
                )
                for interaction_id, sender, _ in candidates
            )
            sender_functional = SimpleNamespace(
                sender_functional_id=sender_functional_id,
                candidate_priors=priors,
                contrast_context_ids=(
                    ("control", "control"),
                    ("treated", "treated"),
                ),
            )
            child = SimpleNamespace(sender_functional=sender_functional)
            functional = object.__new__(CrossReceiverCommonScoringFunctional)
            functional_values: dict[str, object] = {
                "spec": scoring_spec,
                "child_functionals": (child,),
                "contrast_name": _CONTRAST_NAME,
                "contrast_manifest_id": contrast_manifest_id,
                "fold_id": fold_id,
                "context_ids": ("control", "treated"),
                "sender_functional_id": sender_functional_id,
                "interaction_mapping": tuple(
                    (interaction_id, "family-1", driver_id)
                    for interaction_id, _, driver_id in candidates
                ),
                "score_version": _SCORE_VERSION,
                "global_common_functional_id": functional_id,
            }
            for field_name, value in functional_values.items():
                object.__setattr__(functional, field_name, value)

            table_rows: list[dict[str, object]] = []
            for sample_id, subject_id, context_id in _DEFAULT_COVERAGE[fold_index - 1]:
                coverage_rows.append(
                    {
                        "fold_id": fold_id,
                        "contrast_id": design_contrast_id,
                        "contrast_context": context_id,
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                    }
                )
                for interaction_id, sender, driver_id in candidates:
                    for mode in (
                        CommunicationMode.STATE.value,
                        CommunicationMode.ECOSYSTEM.value,
                    ):
                        source_key = (fold_id, sample_id, interaction_id, mode)
                        if source_key in omitted:
                            continue
                        table_rows.append(
                            _sender_score_row(
                                functional_id=functional_id,
                                sender_functional_id=sender_functional_id,
                                fold_id=fold_id,
                                sample_id=sample_id,
                                subject_id=subject_id,
                                context_id=context_id,
                                interaction_id=interaction_id,
                                sender=sender,
                                driver_id=driver_id,
                                mode=mode,
                                score=overrides.get(source_key, 0.25),
                            )
                        )
            application = object.__new__(CrossReceiverCommonScoringApplication)
            application_values: dict[str, object] = {
                "functional": functional,
                "heldout_subject_ids": tuple(
                    sorted(
                        {
                            subject_id
                            for _, subject_id, _ in _DEFAULT_COVERAGE[fold_index - 1]
                        }
                    )
                ),
                "sender_scores_digest": f"sender-digest-{tag}-{fold_id}",
                "application_id": f"global-application-{tag}-{fold_id}",
                "_sender_scores": pd.DataFrame(
                    table_rows,
                    columns=GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
                ),
            }
            for field_name, value in application_values.items():
                object.__setattr__(application, field_name, value)
            folds.append(
                SimpleNamespace(
                    fold_id=fold_id,
                    cross_receiver_common_functionals=(functional,),
                    cross_receiver_common_applications=(application,),
                )
            )

        artifacts = object.__new__(CrossFitArtifacts)
        crossfit_id = f"crossfit-{tag}"
        artifact_values: dict[str, object] = {
            "spec": SimpleNamespace(
                contrasts=(contrast,),
                repeat_id="repeat-0",
            ),
            "folds": tuple(folds),
            "_oof_coverage": pd.DataFrame(coverage_rows),
            "crossfit_id": crossfit_id,
            "receiver_scoring_registry_id": f"registry-{tag}",
        }
        for field_name, value in artifact_values.items():
            object.__setattr__(artifacts, field_name, value)
        requirement = CrossFitOOFRequirementRecord(
            requirement_code="complete_v3_oof_chain",
            status="satisfied" if audit_complete else "not_satisfied",
            reason_code=None if audit_complete else "test_incomplete_oof_chain",
            repeat_id="repeat-0",
            evidence_ids=(crossfit_id,),
        )
        audits[crossfit_id] = CrossFitOOFCertificationAudit._from_requirements(
            source_crossfit_id=crossfit_id,
            source_registry_id=f"registry-{tag}",
            requirements=(requirement,),
        )
        return artifacts

    return build


def _record_for(
    collection: Any,
    universe: Any,
    *,
    context_id: str,
    interaction_id: str,
    mode: CommunicationMode,
) -> Any:
    candidate = next(
        item
        for item in universe.candidates
        if item.context_id == context_id
        and item.interaction_id == interaction_id
        and item.mode is mode
    )
    return next(
        record
        for record in collection.records
        if record.candidate_edge_id == candidate.candidate_edge_id
    )


def test_strict_adapter_uses_repeat_then_subject_equal_oof_reduction(
    artifact_factory: Callable[..., CrossFitArtifacts],
) -> None:
    overrides = {
        ("fold-1", "treated-tech-1a", "lr-1", "state"): 1.0,
        ("fold-1", "treated-tech-1b", "lr-1", "state"): 0.0,
        ("fold-2", "treated-tech-2", "lr-1", "state"): 1.0,
    }
    artifacts = artifact_factory(tag="technical", score_overrides=overrides)

    universe = freeze_crossfit_active_edge_universe(
        artifacts,
        contrast_id_or_name=_CONTRAST_NAME,
    )
    collection = adapt_crossfit_active_edge_point_records(artifacts, universe)
    record = _record_for(
        collection,
        universe,
        context_id="treated",
        interaction_id="lr-1",
        mode=CommunicationMode.STATE,
    )

    assert len(universe.candidates) == 4
    assert record.score == pytest.approx(0.75)
    assert record.n_subjects == 2
    assert record.status is ActiveEdgeScoreStatus.OBSERVED
    assert collection.repeat_id == "repeat-0"
    assert collection.score_spec_id == CrossReceiverCommonScoringSpec().spec_id
    assert {item.source_score_collection_id for item in collection.records} == {
        collection.source_score_collection_id
    }
    collection._require_intact()


def test_fold_candidate_union_and_authoritative_subset_adapter_are_fail_closed(
    artifact_factory: Callable[..., CrossFitArtifacts],
) -> None:
    point = artifact_factory(
        tag="point-union",
        fold_candidates=(
            (
                ("lr-1", "Sender", "driver-lr-1"),
                ("lr-2", "Sender", "driver-lr-2"),
            ),
            (("lr-1", "Sender", "driver-lr-1"),),
        ),
    )
    universe = freeze_crossfit_active_edge_universe(
        point,
        contrast_id_or_name=_CONTRAST_NAME,
    )
    point_collection = adapt_crossfit_active_edge_point_records(point, universe)
    absent_in_fold = _record_for(
        point_collection,
        universe,
        context_id="treated",
        interaction_id="lr-2",
        mode=CommunicationMode.STATE,
    )
    assert len(universe.candidates) == 8
    assert absent_in_fold.status is ActiveEdgeScoreStatus.NOT_ESTIMABLE
    assert absent_in_fold.n_subjects == 2

    child = artifact_factory(tag="subset-child")
    child_collection = adapt_crossfit_active_edge_records_against_universe(
        child,
        universe,
    )
    child_missing = _record_for(
        child_collection,
        universe,
        context_id="treated",
        interaction_id="lr-2",
        mode=CommunicationMode.STATE,
    )
    assert child_missing.status is ActiveEdgeScoreStatus.NOT_ESTIMABLE
    assert child_collection.active_edge_universe_id == universe.universe_id
    with pytest.raises(ContractError) as raised:
        adapt_crossfit_active_edge_point_records(child, universe)
    assert raised.value.details.code == "active_edge_point_universe_mismatch"


def test_missing_expected_sample_row_is_not_silently_dropped(
    artifact_factory: Callable[..., CrossFitArtifacts],
) -> None:
    artifacts = artifact_factory(
        tag="missing-row",
        missing_rows={("fold-2", "treated-tech-2", "lr-1", "state")},
    )
    universe = freeze_crossfit_active_edge_universe(
        artifacts,
        contrast_id_or_name=_CONTRAST_NAME,
    )

    collection = adapt_crossfit_active_edge_point_records(artifacts, universe)
    record = _record_for(
        collection,
        universe,
        context_id="treated",
        interaction_id="lr-1",
        mode=CommunicationMode.STATE,
    )

    assert record.status is ActiveEdgeScoreStatus.NOT_ESTIMABLE
    assert record.reason_code == "active_edge_point_source_not_estimable"
    assert record.n_subjects == 2


def test_authoritative_adapter_rejects_extra_candidates_and_wrong_lineage(
    artifact_factory: Callable[..., CrossFitArtifacts],
) -> None:
    point = artifact_factory(tag="authoritative-point")
    universe = freeze_crossfit_active_edge_universe(
        point,
        contrast_id_or_name=_CONTRAST_NAME,
    )
    extra_child = artifact_factory(
        tag="extra-child",
        fold_candidates=(
            (
                ("lr-1", "Sender", "driver-lr-1"),
                ("lr-2", "Sender", "driver-lr-2"),
            ),
            (
                ("lr-1", "Sender", "driver-lr-1"),
                ("lr-2", "Sender", "driver-lr-2"),
            ),
        ),
    )
    with pytest.raises(ContractError) as raised:
        adapt_crossfit_active_edge_records_against_universe(extra_child, universe)
    assert raised.value.details.code == "active_edge_point_universe_mismatch"

    wrong_lineage = artifact_factory(tag="wrong-lineage")
    foreign = artifact_factory(tag="foreign-lineage")
    application = wrong_lineage.folds[0].cross_receiver_common_applications[0]
    foreign_application = foreign.folds[0].cross_receiver_common_applications[0]
    object.__setattr__(application, "functional", foreign_application.functional)
    object.__setattr__(
        application, "_sender_scores", foreign_application._sender_scores
    )
    wrong_universe = freeze_crossfit_active_edge_universe(
        wrong_lineage,
        contrast_id_or_name=_CONTRAST_NAME,
    )
    with pytest.raises(ContractError) as raised:
        adapt_crossfit_active_edge_point_records(wrong_lineage, wrong_universe)
    assert raised.value.details.code == "active_edge_point_source_mismatch"


def test_adapter_rejects_incomplete_oof_audit_and_non_v3_source_table(
    artifact_factory: Callable[..., CrossFitArtifacts],
) -> None:
    incomplete = artifact_factory(tag="incomplete", audit_complete=False)
    incomplete_universe = freeze_crossfit_active_edge_universe(
        incomplete,
        contrast_id_or_name=_CONTRAST_NAME,
    )
    with pytest.raises(ContractError) as raised:
        adapt_crossfit_active_edge_point_records(incomplete, incomplete_universe)
    assert raised.value.details.code == "active_edge_point_oof_not_certified"

    legacy = artifact_factory(tag="legacy-table")
    legacy_universe = freeze_crossfit_active_edge_universe(
        legacy,
        contrast_id_or_name=_CONTRAST_NAME,
    )
    application = legacy.folds[0].cross_receiver_common_applications[0]
    legacy_table = application._sender_scores.drop(
        columns=["global_sender_lr_score"]
    ).assign(comm_strength=0.5)
    object.__setattr__(application, "_sender_scores", legacy_table)
    with pytest.raises(ContractError) as raised:
        adapt_crossfit_active_edge_point_records(legacy, legacy_universe)
    assert raised.value.details.code == "active_edge_point_source_table_mismatch"


def test_freeze_rejects_cross_fold_interaction_driver_conflict(
    artifact_factory: Callable[..., CrossFitArtifacts],
) -> None:
    artifacts = artifact_factory(
        tag="driver-conflict",
        fold_candidates=(
            (("lr-1", "Sender-A", "driver-a"),),
            (("lr-1", "Sender-B", "driver-b"),),
        ),
    )

    with pytest.raises(ContractError) as raised:
        freeze_crossfit_active_edge_universe(
            artifacts,
            contrast_id_or_name=_CONTRAST_NAME,
        )
    assert raised.value.details.code == "active_edge_point_driver_mapping_conflict"
