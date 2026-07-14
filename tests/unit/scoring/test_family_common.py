from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

from crychic.attribution import (
    ReceiverFamilyTrainingArtifact,
    fit_receiver_family_training_artifact,
)
from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)
from crychic.core import ContractError
from crychic.design import balanced_contrast
from crychic.resources import (
    GeneNamespace,
    MappingReport,
    Species,
    TargetPrior,
)
from crychic.scoring import (
    DownstreamRowManifest,
    FamilyCommonScoringApplication,
    FamilyCommonScoringFunctional,
    IncrementalDownstreamApplication,
    IncrementalDownstreamFunctional,
    apply_family_common_scoring_functional,
    apply_incremental_downstream_functional,
    apply_receiver_program_training_artifact,
    family_common_edge_evidence_digest,
    family_common_sender_application_digest,
    fit_family_common_scoring_functional,
    fit_incremental_downstream_functional,
    fit_receiver_program_training_artifact,
    mark_family_common_scoring_application_not_estimable,
    mark_family_common_scoring_not_estimable,
)
from crychic.sender import (
    COMMON_SENDER_APPLICATION_COLUMNS,
    CommonSenderApplication,
    ContrastCommonSenderFunctional,
    ContrastCommonSenderParameters,
    apply_contrast_common_sender_functional,
    fit_contrast_common_sender_functional,
    freeze_common_sender_candidate_manifest,
    interaction_ligand_contrast_gate,
)


def _prior(columns: Mapping[str, Mapping[str, float]]) -> TargetPrior:
    driver_ids = tuple(sorted(columns))
    target_ids = tuple(
        sorted({target for links in columns.values() for target in links})
    )
    target_index = {target: index for index, target in enumerate(target_ids)}
    indptr = [0]
    target_indices: list[int] = []
    weights: list[float] = []
    ranks: list[int] = []
    for driver in driver_ids:
        links = sorted(columns[driver].items(), key=lambda item: (-item[1], item[0]))
        for rank, (target, weight) in enumerate(links, start=1):
            target_indices.append(target_index[target])
            weights.append(weight)
            ranks.append(rank)
        indptr.append(len(weights))
    return TargetPrior(
        resource_id="tiny_prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=target_ids,
        driver_ids=driver_ids,
        indptr=tuple(indptr),
        target_indices=tuple(target_indices),
        weights=tuple(weights),
        ranks=tuple(ranks),
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(
            source_rows=len(weights),
            loaded_rows=len(weights),
            mapped_entities=len(driver_ids) + len(target_ids),
        ),
        manifest_digest="tiny-prior-manifest",
    )


def _receiver_family() -> ReceiverFamilyTrainingArtifact:
    receptor_values = {"iA": 0.8, "iB": 0.6, "iC": 0.02}
    rows = [
        {
            "sample_id": f"p{index}",
            "subject_id": f"p{index}",
            "receiver": "Receiver",
            "interaction_id": interaction_id,
            "receptor_availability": value,
        }
        for index in range(1, 4)
        for interaction_id, value in receptor_values.items()
    ]
    universe = FrozenInteractionUniverse(
        interaction_ids=("iA", "iB", "iC"),
        training_subject_ids=("p1", "p2", "p3"),
        resource_id="tiny_lr",
        resource_version="1",
        resource_manifest_digest="tiny-lr-manifest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )
    availability = BatchAvailability(
        sample_interactions=pd.DataFrame(rows),
        mapping_summary=pd.DataFrame(),
        resource_id="tiny_lr",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=universe,
        filter_application=InteractionFilterApplication.TRAINING_SELECTION_V1,
        application_subject_ids=("p1", "p2", "p3"),
    )
    return fit_receiver_family_training_artifact(
        availability,
        _prior({"A": {"G1": 1.0}, "B": {"G1": 4.0}, "C": {"G2": 1.0}}),
        receiver="Receiver",
        fold_id="fold-1",
        feature_ids=("G1", "G2"),
        driver_by_interaction={"iA": "A", "iB": "B", "iC": "C"},
        receptor_gate_threshold=0.1,
        cosine_threshold=0.999,
    )


def _manifest(*, prefix: str, subjects: tuple[str, ...]) -> DownstreamRowManifest:
    return DownstreamRowManifest(
        sample_ids=tuple(
            f"{prefix}-{subject}-{context}"
            for context in ("reference", "target")
            for subject in subjects
        ),
        subject_ids=tuple(subject for _ in range(2) for subject in subjects),
        context_ids=tuple(
            context for context in ("reference", "target") for _ in subjects
        ),
    )


def _incremental_functional(
    receiver_family: ReceiverFamilyTrainingArtifact,
) -> IncrementalDownstreamFunctional:
    subjects = ("p1", "p2", "p3")
    manifest = _manifest(prefix="training", subjects=subjects)
    regressor = np.asarray([-1.0] * 3 + [1.0] * 3)
    response = np.column_stack(
        [
            np.asarray([0.0, 0.1, -0.1, 2.0, 2.1, 1.9]),
            np.asarray([3.0, 3.1, 2.9, 3.0, 3.1, 2.9]),
        ]
    )
    basis = receiver_family.family_basis
    eligible = np.flatnonzero(basis.family_eligible)
    family_ids = tuple(basis.family_ids[index] for index in eligible)
    return fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        reference_mask=regressor < 0,
        nuisance_matrix=np.ones((6, 1)),
        context_regressor=regressor,
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        fold_id="fold-1",
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("G1", "G2"),
        family_ids=family_ids,
        nuisance_column_ids=("intercept",),
        training_subject_ids=subjects,
        family_basis=basis.matrix[:, eligible],
        precision_weights=np.ones(2),
        minimum_scale=0.25,
        null_loss_floor=1e-8,
    )


def _incremental_application(
    functional: IncrementalDownstreamFunctional,
    *,
    second_subject_active: bool = True,
) -> IncrementalDownstreamApplication:
    subjects = ("h1", "h2")
    manifest = _manifest(prefix="heldout", subjects=subjects)
    target_second = 2.0 if second_subject_active else 1.0
    response = np.column_stack(
        [
            np.asarray([0.05, 0.0, 2.05, target_second]),
            np.asarray([3.05, 3.0, 3.05, 3.0]),
        ]
    )
    return apply_incremental_downstream_functional(
        functional,
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=np.asarray([-1.0, -1.0, 1.0, 1.0]),
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("G1", "G2"),
        nuisance_column_ids=("intercept",),
    )


def _sender_training(
    *,
    unsupported_interactions: tuple[str, ...] = (),
    not_estimable_interactions: tuple[str, ...] = (),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_index in range(1, 4):
        for context in ("reference", "target"):
            for interaction_index, interaction_id in enumerate(("iA", "iB", "iC")):
                for sender_index, sender in enumerate(("S1", "S2")):
                    rows.append(
                        {
                            "sample_id": f"p{subject_index}-{context}",
                            "subject_id": f"p{subject_index}",
                            "context_id": context,
                            "sender": sender,
                            "receiver": "Receiver",
                            "interaction_id": interaction_id,
                            "ligand_availability": (
                                0.8
                                - 0.1 * sender_index
                                - 0.05 * interaction_index
                                + (
                                    0.0
                                    if interaction_id in unsupported_interactions
                                    else 0.2
                                    if context == "target"
                                    else 0.0
                                )
                                if not (
                                    interaction_id in not_estimable_interactions
                                    and context == "target"
                                    and subject_index > 1
                                )
                                else None
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _sender_functional(
    receiver_family: ReceiverFamilyTrainingArtifact,
    *,
    unsupported_interactions: tuple[str, ...] = (),
    not_estimable_interactions: tuple[str, ...] = (),
) -> ContrastCommonSenderFunctional:
    contrast = balanced_contrast(
        ("target",),
        ("reference",),
        name="stim_vs_ctrl",
        family="test",
    )
    training = _sender_training(
        unsupported_interactions=unsupported_interactions,
        not_estimable_interactions=not_estimable_interactions,
    )
    frozen_interaction_ids = ("iA", "iB", "iC")
    universe = FrozenInteractionUniverse(
        interaction_ids=frozen_interaction_ids,
        training_subject_ids=("p1", "p2", "p3"),
        resource_id="tiny_lr",
        resource_version="1",
        resource_manifest_digest="tiny-lr-manifest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )
    return fit_contrast_common_sender_functional(
        training,
        contrast=contrast,
        context_keys=("context_id",),
        frozen_interaction_universe=universe,
        frozen_candidate_sender_manifest=freeze_common_sender_candidate_manifest(
            training,
            frozen_interaction_ids=frozen_interaction_ids,
        ),
        training_input_digest="family-common-unit-training-input",
        parameters=ContrastCommonSenderParameters(
            min_subjects=2,
            prevalence_threshold=0.0,
            softmax_temperature=0.5,
        ),
    )


def _sender_heldout() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    manifest = _manifest(prefix="heldout", subjects=("h1", "h2"))
    for sample_id, subject_id, context_id in zip(
        manifest.sample_ids,
        manifest.subject_ids,
        manifest.context_ids,
        strict=True,
    ):
        for interaction_id in ("iA", "iB", "iC"):
            rows.extend(
                [
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "context_id": context_id,
                        "sender": "S1",
                        "receiver": "Receiver",
                        "interaction_id": interaction_id,
                        "ligand_availability": 0.9,
                    },
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "context_id": context_id,
                        "sender": "S2",
                        "receiver": "Receiver",
                        "interaction_id": interaction_id,
                        "ligand_availability": 0.1,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _edge_evidence(
    sender_functional: ContrastCommonSenderFunctional | None = None,
) -> pd.DataFrame:
    sender = (
        _sender_functional(_receiver_family())
        if sender_functional is None
        else sender_functional
    )
    gates = {
        interaction_id: interaction_ligand_contrast_gate(
            sender, "Receiver", interaction_id
        )
        for interaction_id in ("iA", "iB", "iC")
    }
    rows: list[dict[str, object]] = []
    availability = {"iA": 0.8, "iB": 0.4, "iC": 1.0}
    manifest = _manifest(prefix="heldout", subjects=("h1", "h2"))
    for sample_id, subject_id, context_id in zip(
        manifest.sample_ids,
        manifest.subject_ids,
        manifest.context_ids,
        strict=True,
    ):
        for interaction_id in ("iA", "iB", "iC"):
            gate = gates[interaction_id]
            for mode in ("state", "ecosystem"):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "context_id": context_id,
                        "receiver": "Receiver",
                        "interaction_id": interaction_id,
                        "mode": mode,
                        "ligand_contrast_gate": gate.gate,
                        "ligand_contrast_gate_id": gate.gate_id,
                        "ligand_contrast_gate_status": gate.status.value,
                        "ligand_contrast_gate_reason_code": gate.reason_code,
                        "availability": availability[interaction_id],
                        "ligand_availability": 0.7,
                        "prior_quality": 1.0,
                        "subject_prevalence": 0.8,
                        "resource_evidence": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _parents() -> tuple[
    ReceiverFamilyTrainingArtifact,
    IncrementalDownstreamFunctional,
    ContrastCommonSenderFunctional,
    FamilyCommonScoringFunctional,
    CommonSenderApplication,
]:
    receiver_family = _receiver_family()
    incremental = _incremental_functional(receiver_family)
    sender = _sender_functional(receiver_family)
    functional = fit_family_common_scoring_functional(
        receiver_family,
        incremental,
        sender,
        receiver_incremental_training_artifact_id="receiver-incremental-parent-1",
        tuning_manifest_id="subject-blocked-tuning-manifest-1",
        selected_penalty_id="selected-penalty-1",
        autonomous_program_resource_id=None,
    )
    sender_application = apply_contrast_common_sender_functional(
        sender, _sender_heldout()
    )
    return receiver_family, incremental, sender, functional, sender_application


def test_functional_binds_common_contrast_fold_and_explicit_lineage() -> None:
    _, incremental, _, functional, _ = _parents()
    manifest = functional.to_dict()
    changed_tuning = fit_family_common_scoring_functional(
        functional.receiver_family,
        incremental,
        functional.sender_functional,
        receiver_incremental_training_artifact_id="receiver-incremental-parent-1",
        tuning_manifest_id="subject-blocked-tuning-manifest-2",
        selected_penalty_id="selected-penalty-1",
        autonomous_program_resource_id=None,
    )

    assert functional.context_ids == ("reference", "target")
    assert manifest["common_across_contexts"] is True
    assert manifest["family_first"] is True
    assert manifest["sender_is_allocation_only"] is True
    assert manifest["receiver_incremental_training_artifact_id"] == (
        "receiver-incremental-parent-1"
    )
    assert manifest["tuning_manifest_id"] == "subject-blocked-tuning-manifest-1"
    assert manifest["selected_penalty_id"] == "selected-penalty-1"
    expected_gates = [
        interaction_ligand_contrast_gate(
            functional.sender_functional,
            functional.receiver,
            interaction_id,
        ).gate_id
        for interaction_id in functional.interaction_ids
    ]
    assert [
        gate["gate_id"]
        for gate in manifest["interaction_ligand_contrast_gates"]
    ] == expected_gates
    assert manifest["member_allocation_method"].endswith(
        "then_supported_family_normalize_v2"
    )
    assert str(manifest["score_version"]).endswith("softmin_v2")
    assert functional.family_common_functional_id != (
        changed_tuning.family_common_functional_id
    )
    assert not functional.is_oof_certified


def test_heldout_subject_family_gain_drives_family_first_conserved_scores() -> None:
    _, incremental, _, functional, sender_application = _parents()
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        _edge_evidence(),
        sender_application,
    )
    attribution = application.family_attribution
    subject = application.subject_differential
    family = application.family_scores
    members = application.member_scores
    senders = application.sender_scores

    assert application.edge_evidence_digest == family_common_edge_evidence_digest(
        _edge_evidence()
    )
    assert application.sender_application_digest == (
        family_common_sender_application_digest(sender_application)
    )

    active_family = functional.active_family_ids[0]
    active_attribution = attribution.loc[attribution["family_id"].eq(active_family)]
    assert active_attribution["selection_status"].iloc[0] == "selected"
    assert (
        subject.loc[subject["family_id"].eq(active_family), "bounded_incremental_gain"]
        .gt(0.9)
        .all()
    )
    active_family_rows = family.loc[family["family_id"].eq(active_family)]
    assert active_family_rows["availability_score"].notna().all()
    assert active_family_rows["integrated_lr_score"].gt(0).all()
    assert active_family_rows["receiver_program_score"].isna().all()
    assert set(active_family_rows["receiver_program_status"]) == {"not_estimable"}
    assert set(active_family_rows["family_common_functional_id"]) == {
        functional.family_common_functional_id
    }
    family_key = ["sample_id", "subject_id", "context_id", "family_id", "mode"]
    for key, group in members.loc[members["family_id"].eq(active_family)].groupby(
        family_key, observed=True
    ):
        expected = family.set_index(family_key).loc[key, "integrated_lr_score"]
        assert group["sender_unresolved_strength"].sum() == pytest.approx(expected)
    sender_key = [
        "sample_id",
        "subject_id",
        "context_id",
        "interaction_id",
        "mode",
    ]
    for key, group in senders.groupby(sender_key, observed=True):
        unresolved = members.set_index(sender_key).loc[
            key, "sender_unresolved_strength"
        ]
        assert group["sender_resolved_strength"].sum() == pytest.approx(unresolved)


def test_receiver_program_stays_observed_when_incremental_parent_is_unavailable() -> (
    None
):
    receiver_family = _receiver_family()
    sender = _sender_functional(receiver_family)
    program = fit_receiver_program_training_artifact(
        receiver_family,
        np.asarray([[0.0, 1.0], [0.1, 1.0], [0.0, 1.0]]),
        contrast_name="stim_vs_ctrl",
        sample_ids=("p1-reference", "p2-reference", "p3-reference"),
        sample_subject_ids=("p1", "p2", "p3"),
        reference_context_ids=("reference", "reference", "reference"),
    )
    manifest = _manifest(prefix="heldout", subjects=("h1", "h2"))
    program_application = apply_receiver_program_training_artifact(
        program,
        np.asarray([[2.0, 3.0]] * len(manifest.sample_ids)),
        feature_ids=("G1", "G2"),
        sample_ids=manifest.sample_ids,
        sample_subject_ids=manifest.subject_ids,
        sample_context_ids=manifest.context_ids,
    )
    functional = mark_family_common_scoring_not_estimable(
        receiver_family,
        sender,
        fold_id="fold-1",
        receiver_incremental_training_artifact_id="incremental-ne-1",
        tuning_manifest_id="tuning-ne-1",
        selected_penalty_id=None,
        autonomous_program_resource_id=None,
        reason_code="incremental_parent_not_estimable",
        receiver_program_artifact=program,
    )
    sender_application = apply_contrast_common_sender_functional(
        sender, _sender_heldout()
    )
    application = mark_family_common_scoring_application_not_estimable(
        functional,
        _edge_evidence(),
        sender_application,
        heldout_reason_code="incremental_application_not_estimable",
        receiver_program_application=program_application,
    )
    family = application.family_scores
    active = family.loc[
        lambda frame: frame["family_id"].isin(functional.active_family_ids)
    ]

    assert len(program_application.to_table()) == (
        len(manifest.sample_ids) * len(functional.family_ids)
    )
    assert set(family["receiver_program_status"]) == {"observed"}
    assert family["receiver_program_score"].gt(0).all()
    assert active["integrated_lr_score"].isna().all()
    assert set(active["status"]) == {"not_estimable"}
    assert "receiver_program_score" not in application.sender_scores.columns
    assert (
        family.groupby(
            ["sample_id", "family_id"], observed=True
        )["receiver_program_score"].nunique()
        == 1
    ).all()


def test_application_input_digests_are_order_stable_and_value_sensitive() -> None:
    _, incremental, _, functional, sender_application = _parents()
    evidence = _edge_evidence()
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        evidence,
        sender_application,
    )
    reversed_evidence = evidence.iloc[::-1].reset_index(drop=True)
    integer_evidence = evidence.copy(deep=True)
    integer_evidence["resource_evidence"] = np.int64(1)
    altered_evidence = evidence.copy(deep=True)
    altered_evidence.loc[0, "prior_quality"] = 0.123
    altered_sender_input = _sender_heldout()
    altered_sender_input["ligand_availability"] = 0.5
    altered_sender = apply_contrast_common_sender_functional(
        functional.sender_functional,
        altered_sender_input,
    )

    assert application.edge_evidence_digest == family_common_edge_evidence_digest(
        reversed_evidence
    )
    assert application.edge_evidence_digest == family_common_edge_evidence_digest(
        integer_evidence
    )
    assert application.edge_evidence_digest != family_common_edge_evidence_digest(
        altered_evidence
    )
    assert application.sender_application_digest != (
        family_common_sender_application_digest(altered_sender)
    )


def test_subject_specific_gain_is_not_replaced_by_fold_broadcast() -> None:
    _, incremental, _, functional, sender_application = _parents()
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental, second_subject_active=False),
        _edge_evidence(),
        sender_application,
    )
    active = functional.active_family_ids[0]
    subject = application.subject_differential.loc[
        lambda frame: frame["family_id"].eq(active)
    ].set_index("subject_id")
    family = application.family_scores.loc[lambda frame: frame["family_id"].eq(active)]

    assert subject.loc["h1", "bounded_incremental_gain"] > 0.9
    assert subject.loc["h2", "bounded_incremental_gain"] == 0.0
    assert family.loc[family["subject_id"].eq("h1"), "integrated_lr_score"].gt(0).all()
    assert family.loc[family["subject_id"].eq("h2"), "integrated_lr_score"].eq(0).all()


def test_missing_member_evidence_is_explicit_lr_ne_without_erasing_family() -> None:
    _, incremental, _, functional, sender_application = _parents()
    evidence = _edge_evidence()
    evidence.loc[evidence["interaction_id"].eq("iB"), "prior_quality"] = np.nan
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        evidence,
        sender_application,
    )
    active = functional.active_family_ids[0]
    family = application.family_scores.loc[lambda frame: frame["family_id"].eq(active)]
    members = application.member_scores.loc[lambda frame: frame["family_id"].eq(active)]
    senders = application.sender_scores.loc[lambda frame: frame["family_id"].eq(active)]

    assert family["integrated_lr_score"].gt(0).all()
    assert set(family["lr_identifiability_status"]) == {"unresolved"}
    assert members["within_family_lr_weight"].isna().all()
    assert members["sender_unresolved_strength"].isna().all()
    assert set(members["reason_code"]) == {"incomplete_member_evidence"}
    assert senders["sender_resolved_strength"].isna().all()


def test_invalid_receptor_availability_or_downstream_cannot_be_revived_by_sender() -> (
    None
):
    _, incremental, _, functional, sender_application = _parents()
    evidence = _edge_evidence()
    evidence.loc[evidence["interaction_id"].isin(["iA", "iB"]), "availability"] = 0.0
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        evidence,
        sender_application,
    )
    members = application.member_scores.set_index("interaction_id")
    senders = application.sender_scores

    assert members.loc[["iA", "iB"], "sender_unresolved_strength"].eq(0).all()
    assert members.loc["iC", "sender_unresolved_strength"].eq(0).all()
    assert senders["sender_resolved_strength"].eq(0).all()
    assert set(senders["status"]) == {"structural_zero"}


def test_not_estimable_producers_preserve_every_planned_heldout_row() -> None:
    receiver_family, _, sender, _, sender_application = _parents()
    functional = mark_family_common_scoring_not_estimable(
        receiver_family,
        sender,
        fold_id="fold-1",
        receiver_incremental_training_artifact_id="receiver-incremental-ne-1",
        tuning_manifest_id="tuning-ne-1",
        selected_penalty_id=None,
        autonomous_program_resource_id=None,
        reason_code="no_training_eligible_receiver_family",
    )
    application = mark_family_common_scoring_application_not_estimable(
        functional,
        _edge_evidence(),
        sender_application,
        heldout_reason_code="no_training_eligible_receiver_family",
    )
    active = functional.active_family_ids[0]
    active_rows = application.family_scores.loc[
        lambda frame: frame["family_id"].eq(active)
    ]

    assert len(application.member_scores) == len(_edge_evidence())
    assert len(active_rows) == 8
    assert active_rows["integrated_lr_score"].isna().all()
    assert set(active_rows["status"]) == {"not_estimable"}
    assert set(active_rows["reason_code"]) == {"no_training_eligible_receiver_family"}
    assert application.to_dict()["incremental_application_id"] is None
    assert functional.to_dict()["selected_penalty_id"] is None


def test_observed_training_functional_can_emit_heldout_ne_without_mutation() -> None:
    _, _, _, functional, sender_application = _parents()
    before = functional.family_common_functional_id
    application = mark_family_common_scoring_application_not_estimable(
        functional,
        _edge_evidence(),
        sender_application,
        heldout_reason_code="heldout_receiver_response_missing",
    )
    active = application.family_scores.loc[
        lambda frame: frame["family_id"].isin(functional.active_family_ids)
    ]

    assert functional.family_common_functional_id == before
    assert functional.incremental_functional is not None
    assert application.incremental_application_id is None
    assert application.heldout_reason_code == "heldout_receiver_response_missing"
    assert active["integrated_lr_score"].isna().all()
    assert set(active["reason_code"]) == {"heldout_receiver_response_missing"}
    assert set(
        application.subject_differential.loc[
            lambda frame: frame["family_id"].isin(functional.active_family_ids),
            "reason_code",
        ]
    ) == {"heldout_receiver_response_missing"}


def test_missing_sender_groups_emit_explicit_ne_rows_instead_of_aborting() -> None:
    _, _, sender, functional, _ = _parents()
    empty_sender = CommonSenderApplication(
        pd.DataFrame(columns=COMMON_SENDER_APPLICATION_COLUMNS), sender
    )
    application = mark_family_common_scoring_application_not_estimable(
        functional,
        _edge_evidence(),
        empty_sender,
        heldout_reason_code="heldout_receiver_response_missing",
    )
    senders = application.sender_scores
    active = senders.loc[
        lambda frame: frame["family_id"].isin(functional.active_family_ids)
    ]
    ineligible = senders.loc[
        lambda frame: ~frame["family_id"].isin(functional.active_family_ids)
    ]

    assert len(senders) == len(_edge_evidence()) * 2
    assert active["sender_resolved_strength"].isna().all()
    assert set(active["reason_code"]) == {"heldout_receiver_response_missing"}
    assert ineligible["sender_resolved_strength"].eq(0).all()


def test_tables_are_defensive_and_forced_internal_mutation_is_detected() -> None:
    _, incremental, _, functional, sender_application = _parents()
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        _edge_evidence(),
        sender_application,
    )
    public = application.family_scores
    public.loc[0, "integrated_lr_score"] = 0.0
    assert application.family_scores.loc[0, "integrated_lr_score"] != 0.0

    internal: pd.DataFrame = object.__getattribute__(application, "_family_scores")
    internal.loc[0, "integrated_lr_score"] = 0.0
    with pytest.raises(ContractError) as error:
        application.to_dict()
    assert error.value.details.code == "family_common_application_integrity_violation"


def test_producer_owned_types_and_edge_contract_fail_closed() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        FamilyCommonScoringFunctional()
    with pytest.raises(TypeError, match="producer-owned"):
        FamilyCommonScoringApplication()

    _, incremental, _, functional, sender_application = _parents()
    with pytest.raises(ContractError, match="columns do not match"):
        apply_family_common_scoring_functional(
            functional,
            _incremental_application(incremental),
            _edge_evidence().assign(receiver_activity=1.0),
            sender_application,
        )
    with pytest.raises(ContractError, match="released scoring modes"):
        apply_family_common_scoring_functional(
            functional,
            _incremental_application(incremental),
            _edge_evidence().loc[lambda frame: frame["mode"].eq("state")],
            sender_application,
        )


@pytest.mark.parametrize(
    ("column", "forged_value"),
    (
        ("ligand_contrast_gate", 0.0),
        ("ligand_contrast_gate_id", "forged-gate-id"),
        ("ligand_contrast_gate_status", "unsupported"),
        (
            "ligand_contrast_gate_reason_code",
            "ligand_contrast_lower_bound_not_positive",
        ),
    ),
)
def test_edge_evidence_rejects_every_forged_ligand_contrast_gate_field(
    column: str,
    forged_value: object,
) -> None:
    _, incremental, sender, functional, sender_application = _parents()
    evidence = _edge_evidence(sender)
    evidence.loc[evidence.index[0], column] = forged_value

    with pytest.raises(ContractError) as error:
        apply_family_common_scoring_functional(
            functional,
            _incremental_application(incremental),
            evidence,
            sender_application,
        )

    assert error.value.details.code == "family_common_application_parent_mismatch"
    assert error.value.details.field == "ligand_contrast_gate_id"


def test_all_receptor_eligible_unsupported_gates_are_exact_family_zero() -> None:
    receiver_family = _receiver_family()
    incremental = _incremental_functional(receiver_family)
    sender = _sender_functional(
        receiver_family,
        unsupported_interactions=("iA", "iB"),
    )
    functional = fit_family_common_scoring_functional(
        receiver_family,
        incremental,
        sender,
        receiver_incremental_training_artifact_id="incremental-unsupported-1",
        tuning_manifest_id="tuning-unsupported-1",
        selected_penalty_id="penalty-unsupported-1",
        autonomous_program_resource_id=None,
    )
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        _edge_evidence(sender),
        apply_contrast_common_sender_functional(sender, _sender_heldout()),
    )
    active_family = functional.active_family_ids[0]
    family = application.family_scores.loc[
        lambda frame: frame["family_id"].eq(active_family)
    ]
    members = application.member_scores.loc[
        lambda frame: frame["family_id"].eq(active_family)
    ]
    assert family["integrated_lr_score"].eq(0.0).all()
    assert set(family["status"]) == {"structural_zero"}
    assert set(family["reason_code"]) == {"ligand_contrast_not_supported"}
    assert set(family["ligand_contrast_gate_status"]) == {"unsupported"}
    assert family["ligand_contrast_supported_interaction_count"].eq(0).all()
    assert family["ligand_contrast_not_estimable_interaction_count"].eq(0).all()
    assert set(members["ligand_contrast_gate_status"]) == {"unsupported"}
    assert members["ligand_contrast_gate"].eq(0.0).all()
    assert members["ligand_contrast_gate_id"].str.len().gt(0).all()
    assert set(members["ligand_contrast_gate_reason_code"]) == {
        "ligand_contrast_holm_adjusted_p_not_below_familywise_alpha"
    }
    assert members["within_family_lr_weight"].eq(0.0).all()
    assert members["sender_unresolved_strength"].eq(0.0).all()


def test_unsupported_member_cannot_borrow_from_supported_same_family_member() -> None:
    receiver_family = _receiver_family()
    incremental = _incremental_functional(receiver_family)
    sender = _sender_functional(
        receiver_family,
        unsupported_interactions=("iB",),
    )
    functional = fit_family_common_scoring_functional(
        receiver_family,
        incremental,
        sender,
        receiver_incremental_training_artifact_id="incremental-mixed-gate-1",
        tuning_manifest_id="tuning-mixed-gate-1",
        selected_penalty_id="penalty-mixed-gate-1",
        autonomous_program_resource_id=None,
    )
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        _edge_evidence(sender),
        apply_contrast_common_sender_functional(sender, _sender_heldout()),
    )
    active_family = functional.active_family_ids[0]
    family = application.family_scores.loc[
        lambda frame: frame["family_id"].eq(active_family)
    ]
    members = application.member_scores.loc[
        lambda frame: frame["family_id"].eq(active_family)
    ]
    supported = members.loc[lambda frame: frame["interaction_id"].eq("iA")]
    unsupported = members.loc[lambda frame: frame["interaction_id"].eq("iB")]

    assert family["integrated_lr_score"].gt(0.0).all()
    assert set(family["ligand_contrast_gate_status"]) == {"supported"}
    assert family["ligand_contrast_supported_interaction_count"].eq(1).all()
    assert family["ligand_contrast_not_estimable_interaction_count"].eq(0).all()
    assert supported["within_family_lr_weight"].eq(1.0).all()
    assert set(supported["ligand_contrast_gate_status"]) == {"supported"}
    assert supported["ligand_contrast_gate"].eq(1.0).all()
    assert unsupported["member_evidence_score"].eq(0.0).all()
    assert unsupported["within_family_lr_weight"].eq(0.0).all()
    assert unsupported["sender_unresolved_strength"].eq(0.0).all()
    assert set(unsupported["reason_code"]) == {"ligand_contrast_not_supported"}


def test_not_estimable_member_prevents_partial_supported_member_allocation() -> None:
    receiver_family = _receiver_family()
    incremental = _incremental_functional(receiver_family)
    sender = _sender_functional(
        receiver_family,
        not_estimable_interactions=("iB",),
    )
    functional = fit_family_common_scoring_functional(
        receiver_family,
        incremental,
        sender,
        receiver_incremental_training_artifact_id="incremental-mixed-ne-1",
        tuning_manifest_id="tuning-mixed-ne-1",
        selected_penalty_id="penalty-mixed-ne-1",
        autonomous_program_resource_id=None,
    )
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        _edge_evidence(sender),
        apply_contrast_common_sender_functional(sender, _sender_heldout()),
    )
    active_family = functional.active_family_ids[0]
    family = application.family_scores.loc[
        lambda frame: frame["family_id"].eq(active_family)
    ]
    members = application.member_scores.loc[
        lambda frame: frame["family_id"].eq(active_family)
    ]
    senders = application.sender_scores.loc[
        lambda frame: frame["family_id"].eq(active_family)
    ]
    supported = members.loc[lambda frame: frame["interaction_id"].eq("iA")]
    not_estimable = members.loc[lambda frame: frame["interaction_id"].eq("iB")]

    assert family["integrated_lr_score"].gt(0.0).all()
    assert set(family["status"]) == {"ok"}
    assert set(family["lr_identifiability_status"]) == {"unresolved"}
    assert family["ligand_contrast_supported_interaction_count"].eq(1).all()
    assert family["ligand_contrast_not_estimable_interaction_count"].eq(1).all()
    assert supported["within_family_lr_weight"].isna().all()
    assert supported["sender_unresolved_strength"].isna().all()
    assert set(supported["reason_code"]) == {"ligand_contrast_not_estimable"}
    assert not_estimable["within_family_lr_weight"].isna().all()
    assert not_estimable["sender_unresolved_strength"].isna().all()
    assert senders["sender_resolved_strength"].isna().all()
    assert set(senders["status"]) == {"not_estimable"}


def test_no_supported_gate_with_receptor_eligible_ne_is_family_ne() -> None:
    receiver_family = _receiver_family()
    incremental = _incremental_functional(receiver_family)
    sender = _sender_functional(
        receiver_family,
        unsupported_interactions=("iB",),
        not_estimable_interactions=("iA",),
    )
    functional = fit_family_common_scoring_functional(
        receiver_family,
        incremental,
        sender,
        receiver_incremental_training_artifact_id="incremental-gate-ne-1",
        tuning_manifest_id="tuning-gate-ne-1",
        selected_penalty_id="penalty-gate-ne-1",
        autonomous_program_resource_id=None,
    )
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        _edge_evidence(sender),
        apply_contrast_common_sender_functional(sender, _sender_heldout()),
    )
    active_family = functional.active_family_ids[0]
    family = application.family_scores.loc[
        lambda frame: frame["family_id"].eq(active_family)
    ]
    members = application.member_scores.loc[
        lambda frame: frame["family_id"].eq(active_family)
    ]
    ne_member = members.loc[lambda frame: frame["interaction_id"].eq("iA")]
    unsupported = members.loc[lambda frame: frame["interaction_id"].eq("iB")]

    assert family["integrated_lr_score"].isna().all()
    assert set(family["status"]) == {"not_estimable"}
    assert set(family["reason_code"]) == {"ligand_contrast_not_estimable"}
    assert set(family["ligand_contrast_gate_status"]) == {"not_estimable"}
    assert family["ligand_contrast_supported_interaction_count"].eq(0).all()
    assert family["ligand_contrast_not_estimable_interaction_count"].eq(1).all()
    assert set(ne_member["ligand_contrast_gate_status"]) == {"not_estimable"}
    assert ne_member["ligand_contrast_gate"].isna().all()
    assert ne_member["sender_unresolved_strength"].isna().all()
    assert unsupported["sender_unresolved_strength"].eq(0.0).all()


def test_receptor_family_ineligible_precedes_ligand_gate_ne() -> None:
    receiver_family = _receiver_family()
    incremental = _incremental_functional(receiver_family)
    sender = _sender_functional(
        receiver_family,
        not_estimable_interactions=("iC",),
    )
    functional = fit_family_common_scoring_functional(
        receiver_family,
        incremental,
        sender,
        receiver_incremental_training_artifact_id="incremental-receptor-first-1",
        tuning_manifest_id="tuning-receptor-first-1",
        selected_penalty_id="penalty-receptor-first-1",
        autonomous_program_resource_id=None,
    )
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        _edge_evidence(sender),
        apply_contrast_common_sender_functional(sender, _sender_heldout()),
    )
    ineligible_family = next(
        family_id
        for family_id in functional.family_ids
        if family_id not in functional.active_family_ids
    )
    family = application.family_scores.loc[
        lambda frame: frame["family_id"].eq(ineligible_family)
    ]
    members = application.member_scores.loc[
        lambda frame: frame["family_id"].eq(ineligible_family)
    ]

    assert family["integrated_lr_score"].eq(0.0).all()
    assert set(family["reason_code"]) == {"receptor_family_ineligible"}
    assert set(family["ligand_contrast_gate_status"]) == {
        "not_applicable_receptor_ineligible"
    }
    assert members["sender_unresolved_strength"].eq(0.0).all()


def test_functional_rejects_mismatched_explicit_autonomous_lineage() -> None:
    receiver_family = _receiver_family()
    incremental = _incremental_functional(receiver_family)
    sender = _sender_functional(receiver_family)

    with pytest.raises(ContractError) as error:
        fit_family_common_scoring_functional(
            receiver_family,
            incremental,
            sender,
            receiver_incremental_training_artifact_id="receiver-incremental-parent-1",
            tuning_manifest_id="tuning-1",
            selected_penalty_id="penalty-1",
            autonomous_program_resource_id="forged-resource",
        )
    assert error.value.details.code == "family_common_parent_mismatch"


def test_observed_semantic_fields_never_silently_become_nan() -> None:
    _, incremental, _, functional, sender_application = _parents()
    application = apply_family_common_scoring_functional(
        functional,
        _incremental_application(incremental),
        _edge_evidence(),
        sender_application,
    )
    observed = application.family_scores.loc[lambda frame: frame["status"].eq("ok")]

    assert observed["availability_score"].notna().all()
    assert observed["incremental_downstream_gain"].notna().all()
    assert observed["differential_effect"].notna().all()
    assert observed["integrated_lr_score"].notna().all()
    assert observed["receiver_program_score"].isna().all()
    assert set(observed["receiver_program_reason_code"]) == {
        "source_agnostic_receiver_program_parent_not_connected"
    }


def test_all_lineage_fields_are_strings_or_explicit_missing_resource() -> None:
    *_, functional, _ = _parents()
    manifest: dict[str, Any] = functional.to_dict()

    for field_name in (
        "receiver_incremental_training_artifact_id",
        "tuning_manifest_id",
        "selected_penalty_id",
    ):
        assert isinstance(manifest[field_name], str)
    assert manifest["autonomous_program_resource_id"] is None
