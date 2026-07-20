from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

import crychic.scoring.global_common as global_common_module
from crychic.attribution import (
    GainCalibrationSpec,
    PenaltyTuningSpec,
    PenaltyValidationLossEstimand,
    ReceiverFamilyTrainingArtifact,
    RelativePenaltyCandidate,
    SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact,
    fit_receiver_family_training_artifact,
    select_penalty_candidate,
)
from crychic.attribution.gain_calibration import (
    _WORKFLOW_GAIN_CALIBRATION_PRODUCER_TOKEN,
    _finalize_selected_penalty_inner_oof_gain_calibration,
)
from crychic.attribution.tuning import (
    _WORKFLOW_SUBJECT_BLOCKED_PRODUCER_TOKEN,
    _record_subject_blocked_penalty_fold_evaluation,
)
from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)
from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import balanced_contrast
from crychic.resources import GeneNamespace, MappingReport, Species, TargetPrior
from crychic.scoring import (
    CrossReceiverCommonScoringApplication,
    CrossReceiverCommonScoringFunctional,
    CrossReceiverCommonScoringSpec,
    DownstreamRowManifest,
    FamilyCommonScoringApplication,
    FamilyCommonScoringFunctional,
    IncrementalDownstreamApplication,
    IncrementalDownstreamFunctional,
    apply_cross_receiver_common_scoring_functional,
    apply_family_common_scoring_functional,
    apply_incremental_downstream_functional,
    fit_cross_receiver_common_scoring_functional,
    fit_family_common_scoring_functional,
    fit_incremental_downstream_functional,
)
from crychic.scoring.global_common import _lr_row_values
from crychic.sender import (
    CommonSenderApplication,
    ContrastCommonSenderFunctional,
    ContrastCommonSenderParameters,
    SenderContrastSupportStatus,
    apply_contrast_common_sender_functional,
    fit_contrast_common_sender_functional,
    freeze_common_sender_candidate_manifest,
    interaction_ligand_contrast_gate,
)

_RECEIVERS = ("R1", "R2")
_INTERACTIONS = ("iA", "iB", "iC")
_TRAINING_SUBJECTS = ("p1", "p2", "p3")
_HELDOUT_SUBJECTS = ("h1", "h2")


def test_streamed_global_table_digest_matches_legacy_sorted_stable_id() -> None:
    columns = ("identifier", "score", "reason")
    table = pd.DataFrame(
        [
            ["row-b", np.float64(0.25), None],
            ["row-a", np.nan, "not_estimable"],
            ["row-c", 0.0, "structural_zero"],
        ],
        columns=columns,
    )
    rows = [
        [global_common_module._canonical_scalar(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    rows.sort(key=canonical_json)
    expected = stable_id(
        "global_common_digest_fixture",
        {"columns": list(columns), "rows": rows},
        schema_version="1",
        digest_length=64,
    )

    observed = global_common_module._table_digest(
        "global_common_digest_fixture",
        table,
        columns,
        unique_text_prefix=("identifier",),
    )
    reordered = global_common_module._table_digest(
        "global_common_digest_fixture",
        table.iloc[::-1],
        columns,
        unique_text_prefix=("identifier",),
    )

    assert observed == expected
    assert reordered == expected


def test_batched_global_digest_sorts_canonical_encoded_prefixes() -> None:
    columns = ("identifier", "score")
    table = pd.DataFrame(
        [
            ["a", 1.0],
            ["a!", 2.0],
            ["a space", 3.0],
            ['a"quote', 4.0],
            ["unicode-\u03b1", 5.0],
        ],
        columns=columns,
    )
    rows = [
        [global_common_module._canonical_scalar(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    rows.sort(key=canonical_json)
    expected = stable_id(
        "global_common_prefix_order_fixture",
        {"columns": list(columns), "rows": rows},
        schema_version="1",
        digest_length=64,
    )

    assert (
        global_common_module._table_digest(
            "global_common_prefix_order_fixture",
            table,
            columns,
            unique_text_prefix=("identifier",),
        )
        == expected
    )


def test_streamed_global_digest_prefix_must_be_unique_text() -> None:
    columns = ("identifier", "score")
    duplicate = pd.DataFrame([["same", 1.0], ["same", 2.0]], columns=columns)

    with pytest.raises(ValueError, match="must be unique"):
        global_common_module._table_digest(
            "global_common_digest_fixture",
            duplicate,
            columns,
            unique_text_prefix=("identifier",),
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


def _interaction_universe() -> FrozenInteractionUniverse:
    return FrozenInteractionUniverse(
        interaction_ids=_INTERACTIONS,
        training_subject_ids=_TRAINING_SUBJECTS,
        resource_id="tiny_lr",
        resource_version="1",
        resource_manifest_digest="tiny-lr-manifest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )


def _receiver_family(
    receiver: str, *, fold_id: str = "fold-1"
) -> ReceiverFamilyTrainingArtifact:
    receptor_values = {"iA": 0.8, "iB": 0.6, "iC": 0.02}
    rows = [
        {
            "sample_id": f"{subject}-reference",
            "subject_id": subject,
            "context_id": "reference",
            "receiver": receiver,
            "interaction_id": interaction_id,
            "receptor_availability": value,
        }
        for subject in _TRAINING_SUBJECTS
        for interaction_id, value in receptor_values.items()
    ]
    availability = BatchAvailability(
        sample_interactions=pd.DataFrame(rows),
        mapping_summary=pd.DataFrame(),
        resource_id="tiny_lr",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=_interaction_universe(),
        filter_application=InteractionFilterApplication.TRAINING_SELECTION_V1,
        application_subject_ids=_TRAINING_SUBJECTS,
    )
    return fit_receiver_family_training_artifact(
        availability,
        _prior({"A": {"G1": 1.0}, "B": {"G1": 4.0}, "C": {"G2": 1.0}}),
        receiver=receiver,
        fold_id=fold_id,
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
    *,
    effect_scale: float = 1.0,
    training_subjects: tuple[str, ...] = _TRAINING_SUBJECTS,
    fold_id: str | None = None,
    penalty_candidate: RelativePenaltyCandidate | None = None,
) -> IncrementalDownstreamFunctional:
    manifest = _manifest(prefix="training", subjects=training_subjects)
    positions = np.asarray(
        [_TRAINING_SUBJECTS.index(subject) for subject in training_subjects],
        dtype=np.float64,
    )
    reference = np.column_stack((0.1 * positions, 3.0 + 0.1 * positions))
    target = reference + np.column_stack(
        (
            effect_scale * (1.8 + 0.2 * positions),
            np.zeros(len(positions), dtype=np.float64),
        )
    )
    response = np.vstack((reference, target))
    basis = receiver_family.family_basis
    eligible = np.flatnonzero(basis.family_eligible)
    family_ids = tuple(basis.family_ids[index] for index in eligible)
    return fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        reference_mask=np.asarray(
            [True] * len(training_subjects) + [False] * len(training_subjects)
        ),
        nuisance_matrix=np.ones((2 * len(training_subjects), 1)),
        context_regressor=np.asarray(
            [-1.0] * len(training_subjects) + [1.0] * len(training_subjects)
        ),
        receiver=receiver_family.receiver,
        contrast_name="stim_vs_ctrl",
        fold_id=receiver_family.fold_id if fold_id is None else fold_id,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("G1", "G2"),
        family_ids=family_ids,
        nuisance_column_ids=("intercept",),
        training_subject_ids=training_subjects,
        family_basis=basis.matrix[:, eligible],
        precision_weights=np.ones(2),
        minimum_scale=0.25,
        null_loss_floor=1e-8,
        penalty_candidate=penalty_candidate,
    )


def _incremental_application(
    functional: IncrementalDownstreamFunctional,
    *,
    prefix: str = "heldout",
) -> IncrementalDownstreamApplication:
    manifest = _manifest(prefix=prefix, subjects=_HELDOUT_SUBJECTS)
    response = np.column_stack(
        [
            np.asarray([0.05, 0.0, 2.05, 2.0]),
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


def _controlled_calibration_application(
    functional: IncrementalDownstreamFunctional,
    *,
    subject: str,
    gain: float,
) -> IncrementalDownstreamApplication:
    manifest = _manifest(
        prefix=f"calibration-{functional.fold_id}", subjects=(subject,)
    )
    n_families = len(functional.family_ids)
    subject_null = np.ones(1, dtype=np.float64)
    subject_family = np.asarray(
        [[1.0 - min(0.95, gain + 0.03 * index) for index in range(n_families)]],
        dtype=np.float64,
    )
    subject_full = np.mean(subject_family, axis=1)
    family_losses = subject_family[0]
    family_gains = 1.0 - family_losses
    full_loss = float(subject_full[0])
    model_gain = 1.0 - full_loss
    return IncrementalDownstreamApplication._from_application(
        functional=functional,
        status="observed",
        reason_code=None,
        sample_ids=manifest.sample_ids,
        sample_subject_ids=manifest.subject_ids,
        sample_context_ids=manifest.context_ids,
        heldout_row_manifest_id=manifest.manifest_id,
        heldout_input_digest=f"calibration-input-{functional.fold_id}",
        null_loss=1.0,
        full_loss=full_loss,
        model_gain=model_gain,
        raw_model_gain=model_gain,
        family_losses=family_losses,
        family_gains=family_gains,
        raw_family_gains=family_gains,
        sample_null_losses=np.ones(2, dtype=np.float64),
        sample_full_losses=np.repeat(subject_full, 2),
        sample_family_losses=np.repeat(subject_family, 2, axis=0),
        subject_ids=(subject,),
        subject_null_losses=subject_null,
        subject_full_losses=subject_full,
        subject_family_losses=subject_family,
    )


def _selected_gain_calibration(
    receiver_family: ReceiverFamilyTrainingArtifact,
    outer: IncrementalDownstreamFunctional,
    *,
    effect_scale: float,
    tuning_spec: PenaltyTuningSpec,
    calibration_spec: GainCalibrationSpec | None = None,
) -> tuple[str, SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact]:
    candidate = tuning_spec.candidates[0]
    functionals: list[IncrementalDownstreamFunctional] = []
    applications: list[IncrementalDownstreamApplication] = []
    evaluations = []
    fold_ids: list[str] = []
    for index, validation_subject in enumerate(_TRAINING_SUBJECTS, start=1):
        fold_id = f"{receiver_family.fold_id}-{receiver_family.receiver}-inner-{index}"
        fold_ids.append(fold_id)
        training_subjects = tuple(
            subject for subject in _TRAINING_SUBJECTS if subject != validation_subject
        )
        functional = _incremental_functional(
            receiver_family,
            effect_scale=effect_scale,
            training_subjects=training_subjects,
            fold_id=fold_id,
            penalty_candidate=candidate,
        )
        application = _controlled_calibration_application(
            functional,
            subject=validation_subject,
            gain=0.12 + 0.14 * index,
        )
        functionals.append(functional)
        applications.append(application)
        evaluations.append(
            _record_subject_blocked_penalty_fold_evaluation(
                candidate,
                _producer_token=_WORKFLOW_SUBJECT_BLOCKED_PRODUCER_TOKEN,
                inner_fold_id=fold_id,
                inner_fold_manifest_id=fold_id,
                training_subject_ids=training_subjects,
                validation_subject_ids=(validation_subject,),
                validation_loss_estimand=(
                    PenaltyValidationLossEstimand.PAIRED_SUBJECT_CONTRAST
                ),
                subject_losses=application.subject_full_losses,
                scale_resolution_id=functional.penalty_scale_resolution_id,
                resolved_penalty_id=functional.resolved_penalty_id,
                resolved_lambda1=functional.lambda1,
                resolved_lambda2=functional.lambda2,
                training_functional_id=functional.incremental_functional_id,
                heldout_application_id=application.application_id,
            )
        )
    tuning = select_penalty_candidate(
        tuning_spec,
        evaluations,
        tuning_scope_id=f"global-common-calibration-{receiver_family.receiver}",
        training_subject_ids=_TRAINING_SUBJECTS,
        inner_fold_ids=tuple(fold_ids),
        inner_fold_plan_id=f"global-common-plan-{receiver_family.receiver}",
    )
    resolved_calibration_spec = (
        GainCalibrationSpec(
            min_subjects=2,
            min_supported_families=1,
            min_subjects_per_family=1,
            min_positive_observations=1,
            min_distinct_positive_gains=2,
        )
        if calibration_spec is None
        else calibration_spec
    )
    calibration = _finalize_selected_penalty_inner_oof_gain_calibration(
        _producer_token=_WORKFLOW_GAIN_CALIBRATION_PRODUCER_TOKEN,
        spec=resolved_calibration_spec,
        tuning_artifact=tuning,
        inner_functionals=functionals,
        inner_applications=applications,
        outer_final_functional=outer,
    )
    assert tuning.tuning_id == calibration.tuning_id
    if calibration_spec is None:
        assert calibration.is_estimable
    return tuning.tuning_id, calibration


def _candidate_senders(receiver: str) -> tuple[str, ...]:
    return ("S1", "S2") if receiver == "R1" else ("S1", "S2", "S3")


def _sender_training() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in _TRAINING_SUBJECTS:
        for context in ("reference", "target"):
            for receiver in _RECEIVERS:
                for interaction_index, interaction_id in enumerate(_INTERACTIONS):
                    for sender_index, sender in enumerate(_candidate_senders(receiver)):
                        rows.append(
                            {
                                "sample_id": f"{subject}-{context}",
                                "subject_id": subject,
                                "context_id": context,
                                "sender": sender,
                                "receiver": receiver,
                                "interaction_id": interaction_id,
                                "ligand_availability": (
                                    0.65
                                    - 0.1 * sender_index
                                    - 0.05 * interaction_index
                                    + (0.2 if context == "target" else 0.0)
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def _sender_functional() -> ContrastCommonSenderFunctional:
    training = _sender_training()
    return fit_contrast_common_sender_functional(
        training,
        contrast=balanced_contrast(
            ("target",),
            ("reference",),
            name="stim_vs_ctrl",
            family="test",
        ),
        context_keys=("context_id",),
        frozen_interaction_universe=_interaction_universe(),
        frozen_candidate_sender_manifest=freeze_common_sender_candidate_manifest(
            training,
            frozen_interaction_ids=_INTERACTIONS,
        ),
        training_input_digest="global-common-unit-training-input",
        parameters=ContrastCommonSenderParameters(
            min_subjects=2,
            prevalence_threshold=0.0,
            softmax_temperature=0.5,
        ),
    )


def _sender_heldout(
    receiver: str,
    *,
    overrides: Mapping[tuple[str, str], float | None] | None = None,
    prefix: str = "heldout",
) -> pd.DataFrame:
    changed = {} if overrides is None else dict(overrides)
    rows: list[dict[str, object]] = []
    manifest = _manifest(prefix=prefix, subjects=_HELDOUT_SUBJECTS)
    defaults = {"S1": 0.8, "S2": 0.2, "S3": 0.7}
    for sample_id, subject_id, context_id in zip(
        manifest.sample_ids,
        manifest.subject_ids,
        manifest.context_ids,
        strict=True,
    ):
        for interaction_id in _INTERACTIONS:
            for sender in _candidate_senders(receiver):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "context_id": context_id,
                        "sender": sender,
                        "receiver": receiver,
                        "interaction_id": interaction_id,
                        "ligand_availability": changed.get(
                            (sender, interaction_id), defaults[sender]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _edge_evidence(
    receiver: str,
    sender_functional: ContrastCommonSenderFunctional,
    *,
    i_b_availability: float,
    prior_overrides: Mapping[str, float | None] | None = None,
    prefix: str = "heldout",
) -> pd.DataFrame:
    changed_priors = {} if prior_overrides is None else dict(prior_overrides)
    gates = {
        interaction_id: interaction_ligand_contrast_gate(
            sender_functional, receiver, interaction_id
        )
        for interaction_id in _INTERACTIONS
    }
    availability = {"iA": 0.8, "iB": i_b_availability, "iC": 1.0}
    rows: list[dict[str, object]] = []
    manifest = _manifest(prefix=prefix, subjects=_HELDOUT_SUBJECTS)
    for sample_id, subject_id, context_id in zip(
        manifest.sample_ids,
        manifest.subject_ids,
        manifest.context_ids,
        strict=True,
    ):
        for interaction_id in _INTERACTIONS:
            gate = gates[interaction_id]
            for mode in ("state", "ecosystem"):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "context_id": context_id,
                        "receiver": receiver,
                        "interaction_id": interaction_id,
                        "mode": mode,
                        "ligand_contrast_gate": gate.gate,
                        "ligand_contrast_gate_id": gate.gate_id,
                        "ligand_contrast_gate_status": gate.status.value,
                        "ligand_contrast_gate_reason_code": gate.reason_code,
                        "availability": availability[interaction_id],
                        "ligand_availability": 0.7,
                        "prior_quality": changed_priors.get(interaction_id, 1.0),
                        "subject_prevalence": 0.8,
                        "resource_evidence": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _child_functional(
    receiver: str,
    sender_functional: ContrastCommonSenderFunctional,
    *,
    fold_id: str = "fold-1",
    effect_scale: float = 1.0,
    family_selection_threshold: float = 0.0,
    calibration_spec: GainCalibrationSpec | None = None,
) -> tuple[
    FamilyCommonScoringFunctional,
    IncrementalDownstreamFunctional,
    SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact,
]:
    receiver_family = _receiver_family(receiver, fold_id=fold_id)
    tuning_spec = PenaltyTuningSpec(
        lambda1_fractions=(0.0,),
        lambda2_fractions=(0.0,),
        inner_allowed_n_splits=(3,),
    )
    incremental = _incremental_functional(
        receiver_family,
        effect_scale=effect_scale,
        penalty_candidate=tuning_spec.candidates[0],
    )
    tuning_id, calibration = _selected_gain_calibration(
        receiver_family,
        incremental,
        effect_scale=effect_scale,
        tuning_spec=tuning_spec,
        calibration_spec=calibration_spec,
    )
    child = fit_family_common_scoring_functional(
        receiver_family,
        incremental,
        sender_functional,
        receiver_incremental_training_artifact_id=f"incremental-parent-{receiver}",
        tuning_manifest_id=tuning_id,
        selected_penalty_id=incremental.resolved_penalty_id,
        autonomous_program_resource_id=None,
        family_selection_threshold=family_selection_threshold,
    )
    return child, incremental, calibration


def _world(
    *,
    prior_overrides: Mapping[str, Mapping[str, float | None]] | None = None,
    sender_overrides: (
        Mapping[str, Mapping[tuple[str, str], float | None]] | None
    ) = None,
    effect_scales: Mapping[str, float] | None = None,
    calibration_specs: Mapping[str, GainCalibrationSpec] | None = None,
) -> tuple[
    CrossReceiverCommonScoringFunctional,
    CrossReceiverCommonScoringApplication,
    tuple[FamilyCommonScoringFunctional, ...],
    tuple[FamilyCommonScoringApplication, ...],
    tuple[CommonSenderApplication, ...],
]:
    sender = _sender_functional()
    scales = {} if effect_scales is None else dict(effect_scales)
    resolved_calibration_specs = (
        {} if calibration_specs is None else dict(calibration_specs)
    )
    children_and_incrementals = tuple(
        _child_functional(
            receiver,
            sender,
            effect_scale=scales.get(receiver, 1.0),
            calibration_spec=resolved_calibration_specs.get(receiver),
        )
        for receiver in _RECEIVERS
    )
    children = tuple(item[0] for item in children_and_incrementals)
    global_functional = fit_cross_receiver_common_scoring_functional(
        children,
        planned_receiver_ids=_RECEIVERS,
        gain_calibration_artifacts={
            child.receiver: item[2]
            for child, item in zip(children, children_and_incrementals, strict=True)
        },
    )
    prior_changes = {} if prior_overrides is None else dict(prior_overrides)
    sender_changes = {} if sender_overrides is None else dict(sender_overrides)
    child_applications: list[FamilyCommonScoringApplication] = []
    sender_applications: list[CommonSenderApplication] = []
    for index, receiver in enumerate(_RECEIVERS):
        child, incremental, _ = children_and_incrementals[index]
        sender_application = apply_contrast_common_sender_functional(
            sender,
            _sender_heldout(
                receiver,
                overrides=sender_changes.get(receiver),
            ),
        )
        sender_applications.append(sender_application)
        child_applications.append(
            apply_family_common_scoring_functional(
                child,
                _incremental_application(incremental),
                _edge_evidence(
                    receiver,
                    sender,
                    i_b_availability=0.4 if receiver == "R1" else 0.1,
                    prior_overrides=prior_changes.get(receiver),
                ),
                sender_application,
            )
        )
    applications = tuple(child_applications)
    sender_results = tuple(sender_applications)
    global_application = apply_cross_receiver_common_scoring_functional(
        global_functional,
        applications,
        sender_results,
    )
    return (
        global_functional,
        global_application,
        children,
        applications,
        sender_results,
    )


def _one_row(
    table: pd.DataFrame,
    *,
    receiver: str,
    interaction_id: str,
    sender: str | None = None,
) -> pd.Series:
    selected = table.loc[
        table["receiver"].eq(receiver)
        & table["interaction_id"].eq(interaction_id)
        & table["subject_id"].eq("h1")
        & table["context_id"].eq("target")
        & table["mode"].eq("state")
    ]
    if sender is not None:
        selected = selected.loc[selected["sender"].eq(sender)]
    assert len(selected) == 1
    return selected.iloc[0]


def test_spec_and_functional_are_receiver_order_invariant() -> None:
    sender = _sender_functional()
    child_parts = tuple(_child_functional(receiver, sender) for receiver in _RECEIVERS)
    children = tuple(item[0] for item in child_parts)
    calibrations = {
        child.receiver: item[2]
        for child, item in zip(children, child_parts, strict=True)
    }
    spec = CrossReceiverCommonScoringSpec(softmin_power=4, epsilon=1e-12)
    equivalent_spec = CrossReceiverCommonScoringSpec(softmin_power=4.0, epsilon=1e-12)

    forward = fit_cross_receiver_common_scoring_functional(
        children,
        planned_receiver_ids=_RECEIVERS,
        gain_calibration_artifacts=calibrations,
        spec=spec,
    )
    reverse = fit_cross_receiver_common_scoring_functional(
        children[::-1],
        planned_receiver_ids=_RECEIVERS[::-1],
        gain_calibration_artifacts=calibrations,
        spec=equivalent_spec,
    )

    assert spec.spec_id == equivalent_spec.spec_id
    assert forward.receiver_ids == _RECEIVERS
    assert forward.child_functional_ids == reverse.child_functional_ids
    assert forward.global_common_functional_id == reverse.global_common_functional_id
    assert forward.to_dict() == reverse.to_dict()


def test_training_only_receiver_gain_calibrations_are_exactly_bound() -> None:
    sender = _sender_functional()
    r1, _, c1 = _child_functional("R1", sender, effect_scale=1.0)
    r2, _, c2 = _child_functional("R2", sender, effect_scale=4.0)

    functional = fit_cross_receiver_common_scoring_functional(
        (r1, r2),
        planned_receiver_ids=_RECEIVERS,
        gain_calibration_artifacts={"R1": c1, "R2": c2},
    )

    assert functional.gain_calibration("R1") is c1
    assert functional.gain_calibration("R2") is c2
    assert functional.all_receivers_gain_calibrated
    assert functional.cross_receiver_percentile_rank_eligible
    assert (
        functional.gain_calibration_binding("R1")["gain_calibration_artifact_id"]
        == c1.artifact_id
    )
    assert functional.to_dict()["training_only_receiver_calibration"] is True
    assert functional.to_dict()["receiver_scale_amplification"] is False


def test_swapped_receiver_gain_calibrations_fail_closed() -> None:
    sender = _sender_functional()
    r1, _, c1 = _child_functional("R1", sender)
    r2, _, c2 = _child_functional("R2", sender)

    with pytest.raises(ContractError) as mismatch:
        fit_cross_receiver_common_scoring_functional(
            (r1, r2),
            planned_receiver_ids=_RECEIVERS,
            gain_calibration_artifacts={"R1": c2, "R2": c1},
        )

    assert mismatch.value.details.code == (
        "global_common_gain_calibration_parent_mismatch"
    )


def test_frozen_percentile_gain_is_used_without_coefficient_rescaling() -> None:
    functional, application, _, _, _ = _world(effect_scales={"R1": 1.0, "R2": 4.0})
    lr_scores = application.global_lr_scores
    observed = lr_scores.loc[lr_scores["status"].eq("observed")]

    assert not observed.empty
    assert "training_receiver_scale_factor" not in lr_scores
    assert observed["calibrated_family_gain_percentile"].notna().all()
    np.testing.assert_allclose(
        observed["global_lr_score"].to_numpy(dtype=float),
        observed["global_lr_core_strength"].to_numpy(dtype=float)
        * observed["prior_quality"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-14,
    )
    for receiver, rows in observed.groupby("receiver", observed=True):
        calibration = functional.gain_calibration(str(receiver))
        assert calibration is not None
        assert set(rows["gain_calibration_artifact_id"]) == {calibration.artifact_id}


def test_positive_gain_with_typed_ne_calibration_does_not_fall_back() -> None:
    typed_ne_spec = GainCalibrationSpec(
        min_subjects=4,
        min_supported_families=1,
        min_subjects_per_family=1,
        min_positive_observations=1,
        min_distinct_positive_gains=2,
    )
    functional, application, _, _, _ = _world(calibration_specs={"R1": typed_ne_spec})
    calibration = functional.gain_calibration("R1")
    assert calibration is not None
    assert calibration.status == "not_estimable"
    assert calibration.reason_code == "insufficient_inner_oof_subjects"

    row = _one_row(
        application.global_lr_scores,
        receiver="R1",
        interaction_id="iA",
    )
    assert row["receiver_relative_family_gain"] > 0.0
    assert pd.isna(row["calibrated_family_gain_percentile"])
    assert pd.isna(row["global_lr_core_strength"])
    assert pd.isna(row["global_lr_score"])
    assert row["status"] == "not_estimable"
    assert row["reason_code"] == "insufficient_inner_oof_subjects"
    assert row["gain_calibration_status"] == "not_estimable"
    assert row["gain_calibration_reason_code"] == ("insufficient_inner_oof_subjects")


def test_training_coefficient_magnitude_does_not_rescale_global_score() -> None:
    spec = CrossReceiverCommonScoringSpec()
    shared = {
        "receptor_eligible": True,
        "gate_status": SenderContrastSupportStatus.SUPPORTED.value,
        "training_selected": True,
        "training_reason": None,
        "availability": 0.8,
        "family_gain": 0.6,
        "calibrated_family_gain": 0.75,
        "gain_calibration_status": "observed",
        "gain_calibration_reason": None,
        "prior_quality": 0.9,
        "spec": spec,
    }

    unit_coefficient = _lr_row_values(training_coefficient=1.0, **shared)
    large_coefficient = _lr_row_values(training_coefficient=100.0, **shared)

    assert unit_coefficient == large_coefficient
    assert unit_coefficient[2:] == ("observed", None)


def test_missing_receiver_and_child_contract_mismatch_fail_closed() -> None:
    sender = _sender_functional()
    r1, _, _ = _child_functional("R1", sender)
    r2, _, _ = _child_functional("R2", sender)

    with pytest.raises(ContractError) as missing:
        fit_cross_receiver_common_scoring_functional(
            (r1,), planned_receiver_ids=_RECEIVERS
        )
    assert missing.value.details.code == "global_common_receiver_coverage_mismatch"

    r2_wrong_fold, _, _ = _child_functional("R2", sender, fold_id="fold-2")
    with pytest.raises(ContractError) as mismatch:
        fit_cross_receiver_common_scoring_functional(
            (r1, r2_wrong_fold), planned_receiver_ids=_RECEIVERS
        )
    assert mismatch.value.details.code == "global_common_child_contract_mismatch"

    r2_thresholded, _, _ = _child_functional(
        "R2",
        sender,
        family_selection_threshold=0.1,
    )
    with pytest.raises(ContractError) as thresholded:
        fit_cross_receiver_common_scoring_functional(
            (r1, r2_thresholded), planned_receiver_ids=_RECEIVERS
        )
    assert thresholded.value.details.code == (
        "global_common_family_selection_threshold_unsupported"
    )

    functional = fit_cross_receiver_common_scoring_functional(
        (r1, r2), planned_receiver_ids=_RECEIVERS
    )
    assert not functional.common_functional_across_receivers
    assert functional.receiver_balanced_descriptive_collection


def test_lr_score_ignores_local_allocation_but_sender_uses_frozen_weight() -> None:
    _, global_application, _, child_applications, sender_applications = _world()
    global_lr = global_application.global_lr_scores
    global_sender = global_application.global_sender_lr_scores

    child_weights = []
    sender_weights = []
    for receiver, child_application, sender_application in zip(
        _RECEIVERS, child_applications, sender_applications, strict=True
    ):
        child_weights.append(
            _one_row(
                child_application.member_scores,
                receiver=receiver,
                interaction_id="iA",
            )["within_family_lr_weight"]
        )
        sender_weights.append(
            sender_application.table.loc[
                sender_application.table["subject_id"].eq("h1")
                & sender_application.table["context_id"].eq("target")
                & sender_application.table["interaction_id"].eq("iA")
                & sender_application.table["sender"].eq("S1"),
                "assignment_weight",
            ].item()
        )

    r1_lr = _one_row(global_lr, receiver="R1", interaction_id="iA")
    r2_lr = _one_row(global_lr, receiver="R2", interaction_id="iA")
    r1_sender = _one_row(global_sender, receiver="R1", interaction_id="iA", sender="S1")
    r2_sender = _one_row(global_sender, receiver="R2", interaction_id="iA", sender="S1")

    assert child_weights[0] != pytest.approx(child_weights[1])
    assert sender_weights[0] != pytest.approx(sender_weights[1])
    assert r1_lr["global_lr_score"] == pytest.approx(r2_lr["global_lr_score"])
    assert r1_sender["raw_sender_evidence"] == pytest.approx(
        r2_sender["raw_sender_evidence"]
    )
    assert r1_sender["assignment_weight"] == pytest.approx(sender_weights[0])
    assert r2_sender["assignment_weight"] == pytest.approx(sender_weights[1])
    assert r1_sender["global_sender_lr_score"] == pytest.approx(
        r1_sender["global_lr_score"] * sender_weights[0]
    )
    assert r2_sender["global_sender_lr_score"] == pytest.approx(
        r2_sender["global_lr_score"] * sender_weights[1]
    )


def test_structural_zero_and_not_estimable_states_are_not_revived() -> None:
    _, application, _, _, _ = _world(
        prior_overrides={"R1": {"iB": None}},
        sender_overrides={"R1": {("S1", "iA"): None, ("S2", "iA"): 0.0}},
    )
    lr = application.global_lr_scores
    sender = application.global_sender_lr_scores

    receptor_zero = _one_row(lr, receiver="R1", interaction_id="iC")
    missing_prior = _one_row(lr, receiver="R1", interaction_id="iB")
    missing_sender = _one_row(sender, receiver="R1", interaction_id="iA", sender="S1")
    zero_sender = _one_row(sender, receiver="R1", interaction_id="iA", sender="S2")

    assert receptor_zero["global_lr_score"] == 0.0
    assert receptor_zero["status"] == "structural_zero"
    assert receptor_zero["reason_code"] == "receptor_interaction_ineligible"
    assert pd.isna(missing_prior["global_lr_score"])
    assert missing_prior["status"] == "not_estimable"
    assert missing_prior["reason_code"] == "prior_quality_missing"
    assert pd.isna(missing_sender["global_sender_lr_score"])
    assert missing_sender["status"] == "not_estimable"
    assert pd.isna(zero_sender["global_sender_lr_score"])
    assert zero_sender["status"] == "not_estimable"
    assert zero_sender["reason_code"] == "incomplete_candidate_evidence"


def test_known_family_zero_short_circuits_unknown_ligand_gate() -> None:
    spec = CrossReceiverCommonScoringSpec()
    selected_zero = _lr_row_values(
        receptor_eligible=True,
        gate_status="not_estimable",
        training_coefficient=0.0,
        training_selected=False,
        training_reason="family_not_selected_in_training",
        availability=None,
        family_gain=None,
        calibrated_family_gain=None,
        gain_calibration_status="not_estimable",
        gain_calibration_reason="calibration_missing",
        prior_quality=None,
        spec=spec,
    )
    heldout_zero = _lr_row_values(
        receptor_eligible=True,
        gate_status="not_estimable",
        training_coefficient=1.0,
        training_selected=True,
        training_reason=None,
        availability=None,
        family_gain=0.0,
        calibrated_family_gain=0.0,
        gain_calibration_status="not_estimable",
        gain_calibration_reason="calibration_missing",
        prior_quality=None,
        spec=spec,
    )
    unresolved = _lr_row_values(
        receptor_eligible=True,
        gate_status="not_estimable",
        training_coefficient=1.0,
        training_selected=True,
        training_reason=None,
        availability=None,
        family_gain=None,
        calibrated_family_gain=None,
        gain_calibration_status="not_estimable",
        gain_calibration_reason="calibration_missing",
        prior_quality=None,
        spec=spec,
    )

    assert selected_zero == (
        0.0,
        0.0,
        "structural_zero",
        "family_not_selected_in_training",
    )
    assert heldout_zero == (
        0.0,
        0.0,
        "structural_zero",
        "incremental_downstream_gain_zero",
    )
    assert unresolved == (
        None,
        None,
        "not_estimable",
        "ligand_contrast_not_estimable",
    )


def test_sender_score_uses_assignment_weight_and_conserves_parent() -> None:
    _, application, _, _, _ = _world()
    sender_scores = application.global_sender_lr_scores
    observed = sender_scores.loc[sender_scores["status"].eq("observed")]

    assert not observed.empty
    np.testing.assert_allclose(
        observed["global_sender_lr_score"].to_numpy(dtype=float),
        observed["global_lr_score"].to_numpy(dtype=float)
        * observed["assignment_weight"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-14,
    )
    group_keys = [
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "mode",
    ]
    grouped = observed.groupby(group_keys, observed=True, sort=False)
    for _, group in grouped:
        assert group["assignment_weight"].sum() == pytest.approx(1.0)
        assert group["global_sender_lr_score"].sum() == pytest.approx(
            group["global_lr_score"].iloc[0]
        )


def test_sender_sources_reject_rows_outside_heldout_lr_scope() -> None:
    functional, _, _, child_applications, sender_applications = _world()
    extra_rows = pd.DataFrame(
        [
            {
                "sample_id": "heldout-h3-target",
                "subject_id": "h3",
                "context_id": "target",
                "sender": sender,
                "receiver": "R1",
                "interaction_id": "iA",
                "ligand_availability": 0.5,
            }
            for sender in _candidate_senders("R1")
        ]
    )
    poisoned_source = apply_contrast_common_sender_functional(
        functional.child_functionals[0].sender_functional,
        pd.concat([_sender_heldout("R1"), extra_rows], ignore_index=True),
    )

    with pytest.raises(ValueError, match="exactly cover held-out LR rows"):
        apply_cross_receiver_common_scoring_functional(
            functional,
            child_applications,
            (poisoned_source, sender_applications[1]),
        )


def test_receiver_children_require_one_exact_heldout_score_grid() -> None:
    functional, _, children, child_applications, sender_applications = _world()
    r2_child = children[1]
    r2_incremental = r2_child.incremental_functional
    assert r2_incremental is not None
    alternate_sender = apply_contrast_common_sender_functional(
        r2_child.sender_functional,
        _sender_heldout("R2", prefix="alternate"),
    )
    alternate_application = apply_family_common_scoring_functional(
        r2_child,
        _incremental_application(r2_incremental, prefix="alternate"),
        _edge_evidence(
            "R2",
            r2_child.sender_functional,
            i_b_availability=0.1,
            prefix="alternate",
        ),
        alternate_sender,
    )

    with pytest.raises(ContractError) as mismatch:
        apply_cross_receiver_common_scoring_functional(
            functional,
            (child_applications[0], alternate_application),
            (sender_applications[0], alternate_sender),
        )
    assert mismatch.value.details.code == "global_common_heldout_grid_mismatch"


def test_spec_functional_and_application_tampering_is_detected() -> None:
    spec = CrossReceiverCommonScoringSpec()
    object.__setattr__(spec, "epsilon", 0.5)
    with pytest.raises(ContractError) as spec_error:
        spec.to_dict()
    assert spec_error.value.details.code == "global_common_spec_integrity_violation"

    functional, _application, _, _, _ = _world()
    object.__setattr__(functional, "receiver_ids", ("R1",))
    with pytest.raises(ContractError) as functional_error:
        functional.to_dict()
    assert functional_error.value.details.code == (
        "global_common_functional_integrity_violation"
    )

    intact_functional, intact_application, _, _, _ = _world()
    tampered = intact_application._lr_scores.copy(deep=True)
    observed_index = tampered.index[tampered["status"].eq("observed")][0]
    tampered.loc[observed_index, "global_lr_score"] = 0.123
    object.__setattr__(intact_application, "_lr_scores", tampered)
    with pytest.raises(ContractError) as application_error:
        _ = intact_application.global_lr_scores
    assert application_error.value.details.code == (
        "global_common_application_integrity_violation"
    )
    assert intact_functional.to_dict()["common_functional_across_receivers"] is False
    assert (
        intact_functional.to_dict()["receiver_balanced_descriptive_collection"] is True
    )
