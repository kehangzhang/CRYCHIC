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
    family_common_edge_evidence_digest,
    family_common_sender_application_digest,
    fit_family_common_scoring_functional,
    fit_incremental_downstream_functional,
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


def _sender_training() -> pd.DataFrame:
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
                                0.8 - 0.1 * sender_index - 0.05 * interaction_index
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _sender_functional(
    receiver_family: ReceiverFamilyTrainingArtifact,
) -> ContrastCommonSenderFunctional:
    contrast = balanced_contrast(
        ("target",),
        ("reference",),
        name="stim_vs_ctrl",
        family="test",
    )
    return fit_contrast_common_sender_functional(
        _sender_training(),
        contrast=contrast,
        context_keys=("context_id",),
        filter_universe_id=receiver_family.filter_universe_id,
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


def _edge_evidence() -> pd.DataFrame:
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
            for mode in ("state", "ecosystem"):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "context_id": context_id,
                        "receiver": "Receiver",
                        "interaction_id": interaction_id,
                        "mode": mode,
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
