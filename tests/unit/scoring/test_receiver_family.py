from __future__ import annotations

import inspect
from collections.abc import Mapping
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from crychic.attribution import (
    ReceiverFamilyTrainingArtifact,
    ReceptorGatePolicy,
    fit_receiver_family_training_artifact,
    fit_receiver_family_training_artifacts,
)
from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)
from crychic.resources import (
    GeneNamespace,
    MappingReport,
    Species,
    TargetPrior,
)
from crychic.scoring import (
    ReceiverFamilyScoringArtifact,
    apply_receiver_family_scoring_artifact,
    fit_receiver_family_scoring_artifact,
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


def _availability(
    *,
    receptor_values: Mapping[str, tuple[float, float, float]] | None = None,
    reverse_rows: bool = False,
    application: InteractionFilterApplication = (
        InteractionFilterApplication.TRAINING_SELECTION_V1
    ),
) -> BatchAvailability:
    values = receptor_values or {
        "iA": (0.4, 0.6, 0.8),
        "iB": (0.7, 0.9, 0.5),
        "iC": (0.05, 0.02, 0.08),
    }
    rows = [
        {
            "sample_id": f"s{subject_index}",
            "subject_id": f"p{subject_index}",
            "receiver": "Receiver",
            "interaction_id": interaction_id,
            "receptor_availability": interaction_values[subject_index - 1],
        }
        for subject_index in range(1, 4)
        for interaction_id, interaction_values in values.items()
    ]
    if reverse_rows:
        rows.reverse()
    training_subjects = (
        ("p1", "p2", "p3")
        if application is InteractionFilterApplication.TRAINING_SELECTION_V1
        else ("upstream-training",)
    )
    universe = FrozenInteractionUniverse(
        interaction_ids=("iA", "iB", "iC"),
        training_subject_ids=training_subjects,
        resource_id="tiny_lr",
        resource_version="1",
        resource_manifest_digest="tiny-lr-manifest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )
    return BatchAvailability(
        sample_interactions=pd.DataFrame(rows),
        mapping_summary=pd.DataFrame(),
        resource_id="tiny_lr",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=universe,
        filter_application=application,
        application_subject_ids=("p1", "p2", "p3"),
    )


def _receiver_family(
    availability: BatchAvailability | None = None,
) -> ReceiverFamilyTrainingArtifact:
    return fit_receiver_family_training_artifact(
        availability or _availability(),
        _prior({"A": {"G1": 1.0}, "B": {"G1": 4.0}, "C": {"G2": 1.0}}),
        receiver="Receiver",
        fold_id="fold-1",
        feature_ids=("G1", "G2"),
        driver_by_interaction={"iA": "A", "iB": "B", "iC": "C"},
        receptor_gate_threshold=0.1,
        cosine_threshold=0.999,
    )


def _multi_receiver_availability() -> BatchAvailability:
    first = _availability()
    second = first.sample_interactions.copy(deep=True)
    second["receiver"] = "ReceiverB"
    second["receptor_availability"] = (
        1.0 - second["receptor_availability"].astype(float)
    )
    return BatchAvailability(
        sample_interactions=pd.concat(
            [first.sample_interactions, second], ignore_index=True
        ),
        mapping_summary=first.mapping_summary,
        resource_id=first.resource_id,
        resource_version=first.resource_version,
        detection_available=first.detection_available,
        frozen_interaction_universe=first.frozen_interaction_universe,
        filter_application=first.filter_application,
        application_subject_ids=first.application_subject_ids,
    )


def _scoring_artifact(
    reference_expression: np.ndarray | None = None,
) -> ReceiverFamilyScoringArtifact:
    expression = (
        np.asarray([[1.0, 3.0], [2.0, 3.0], [3.0, 3.0]])
        if reference_expression is None
        else reference_expression
    )
    return fit_receiver_family_scoring_artifact(
        _receiver_family(),
        expression,
        contrast_name="stim_vs_ctrl",
        sample_ids=("s1", "s2", "s3"),
        sample_subject_ids=("p1", "p2", "p3"),
        minimum_scale=0.25,
    )


def test_training_freezes_hard_gates_strict_families_and_downstream() -> None:
    receiver_family = _receiver_family()

    assert (
        receiver_family.source_basis.gate_policy
        is ReceptorGatePolicy.HARD_ELIGIBILITY_V2
    )
    assert dict(receiver_family.receptor_gates) == pytest.approx(
        {"A": 0.6, "B": 0.7, "C": 0.05}
    )
    assert receiver_family.source_basis.receptor_eligible.tolist() == [
        True,
        True,
        False,
    ]
    assert len(receiver_family.eligible_family_ids) == 1
    assert receiver_family.certification_status == (
        "training_only_partial_receiver_family_v1"
    )
    assert not receiver_family.is_oof_certified

    scoring = fit_receiver_family_scoring_artifact(
        receiver_family,
        np.asarray([[1.0, 3.0], [2.0, 3.0], [3.0, 3.0]]),
        contrast_name="stim_vs_ctrl",
        sample_ids=("s1", "s2", "s3"),
        sample_subject_ids=("p1", "p2", "p3"),
    )
    assert scoring.downstream_functional is not None
    assert scoring.active_family_ids == receiver_family.eligible_family_ids
    assert scoring.downstream_functional.family_ids == scoring.active_family_ids
    assert scoring.certification_status == (
        "training_only_partial_receiver_family_downstream_v1"
    )
    assert "common_scoring_functional" in scoring.remaining_stages
    assert not scoring.is_oof_certified


def test_heldout_apply_never_calls_fit_and_poison_cannot_change_artifact() -> None:
    artifact = _scoring_artifact()
    assert artifact.downstream_functional is not None
    center = artifact.downstream_functional.feature_center.copy()
    scale = artifact.downstream_functional.feature_scale.copy()
    artifact_id = artifact.training_artifact_id
    functional_id = artifact.downstream_functional.downstream_functional_id

    with (
        patch(
            "crychic.scoring.receiver_family.fit_downstream_functional",
            side_effect=AssertionError("fit reached heldout apply"),
        ),
        patch(
            "crychic.attribution.frozen_family.build_gated_target_basis",
            side_effect=AssertionError("gate fit reached heldout apply"),
        ),
    ):
        clean = apply_receiver_family_scoring_artifact(
            artifact,
            np.asarray([[2.0, 3.0], [4.0, 3.0]]),
            feature_ids=("G1", "G2"),
            sample_subject_ids=("h1", "h2"),
        )
        poisoned = apply_receiver_family_scoring_artifact(
            artifact,
            np.asarray([[20_000.0, -20_000.0], [40_000.0, 30_000.0]]),
            feature_ids=("G1", "G2"),
            sample_subject_ids=("h1", "h2"),
        )

    assert clean.application_status == "frozen_application_partial_not_oof"
    assert not clean.is_oof_certified
    assert clean.downstream_application is not None
    assert poisoned.downstream_application is not None
    assert not np.array_equal(
        clean.downstream_application.receiver_program_score,
        poisoned.downstream_application.receiver_program_score,
    )
    assert artifact.training_artifact_id == artifact_id
    assert artifact.downstream_functional.downstream_functional_id == functional_id
    np.testing.assert_array_equal(artifact.downstream_functional.feature_center, center)
    np.testing.assert_array_equal(artifact.downstream_functional.feature_scale, scale)


def test_training_poison_changes_only_training_owned_identities() -> None:
    clean_receiver = _receiver_family()
    poisoned_receiver = _receiver_family(
        _availability(
            receptor_values={
                "iA": (0.9, 0.9, 0.9),
                "iB": (0.7, 0.9, 0.5),
                "iC": (0.05, 0.02, 0.08),
            }
        )
    )

    assert clean_receiver.receptor_gate_manifest_id != (
        poisoned_receiver.receptor_gate_manifest_id
    )
    assert clean_receiver.training_artifact_id != poisoned_receiver.training_artifact_id
    np.testing.assert_array_equal(
        clean_receiver.family_basis.matrix.toarray(),
        poisoned_receiver.family_basis.matrix.toarray(),
    )

    clean_scoring = _scoring_artifact()
    poisoned_scoring = _scoring_artifact(
        np.asarray([[10.0, 3.0], [20.0, 3.0], [30.0, 3.0]])
    )
    assert clean_scoring.downstream_functional is not None
    assert poisoned_scoring.downstream_functional is not None
    assert clean_scoring.training_artifact_id != poisoned_scoring.training_artifact_id
    assert clean_scoring.downstream_functional.downstream_functional_id != (
        poisoned_scoring.downstream_functional.downstream_functional_id
    )


def test_receiver_artifact_identity_covers_reference_rows_not_only_summary() -> None:
    receiver_family = _receiver_family()
    first_expression = np.asarray([[0.0, 3.0], [1.0, 3.0], [2.0, 3.0]])
    same_summary = np.asarray([[0.0, 3.0], [1.0, 3.0], [5.0, 3.0]])
    first = fit_receiver_family_scoring_artifact(
        receiver_family,
        first_expression,
        contrast_name="stim_vs_ctrl",
        sample_ids=("s1", "s2", "s3"),
        sample_subject_ids=("p1", "p2", "p3"),
    )
    changed = fit_receiver_family_scoring_artifact(
        receiver_family,
        same_summary,
        contrast_name="stim_vs_ctrl",
        sample_ids=("s1", "s2", "s3"),
        sample_subject_ids=("p1", "p2", "p3"),
    )

    assert first.downstream_functional is not None
    assert changed.downstream_functional is not None
    np.testing.assert_array_equal(
        first.downstream_functional.feature_center,
        changed.downstream_functional.feature_center,
    )
    np.testing.assert_array_equal(
        first.downstream_functional.feature_scale,
        changed.downstream_functional.feature_scale,
    )
    assert first.training_artifact_id != changed.training_artifact_id


def test_receiver_artifact_identity_canonicalizes_sample_row_order() -> None:
    receiver_family = _receiver_family()
    expression = np.asarray([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    first = fit_receiver_family_scoring_artifact(
        receiver_family,
        expression,
        contrast_name="stim_vs_ctrl",
        sample_ids=("s1", "s2", "s3"),
        sample_subject_ids=("p1", "p2", "p3"),
    )
    order = np.asarray([2, 0, 1])
    reordered = fit_receiver_family_scoring_artifact(
        receiver_family,
        expression[order],
        contrast_name="stim_vs_ctrl",
        sample_ids=("s3", "s1", "s2"),
        sample_subject_ids=("p3", "p1", "p2"),
    )

    assert reordered.reference_sample_ids == first.reference_sample_ids
    assert reordered.training_artifact_id == first.training_artifact_id


def test_subject_overlap_feature_drift_and_nontraining_availability_are_blocked() -> (
    None
):
    artifact = _scoring_artifact()

    with pytest.raises(ValueError, match="overlaps training subjects"):
        apply_receiver_family_scoring_artifact(
            artifact,
            np.asarray([[2.0, 3.0]]),
            feature_ids=("G1", "G2"),
            sample_subject_ids=("p1",),
        )
    with pytest.raises(ValueError, match="exactly match frozen features"):
        apply_receiver_family_scoring_artifact(
            artifact,
            np.asarray([[2.0, 3.0]]),
            feature_ids=("G2", "G1"),
            sample_subject_ids=("h1",),
        )
    with pytest.raises(ValueError, match="training-selection availability"):
        _receiver_family(
            _availability(
                application=InteractionFilterApplication.FROZEN_APPLICATION_V1
            )
        )


def test_no_eligible_family_is_explicitly_partial_and_not_estimable() -> None:
    unavailable = _availability(
        receptor_values={
            "iA": (0.01, 0.01, 0.01),
            "iB": (0.01, 0.01, 0.01),
            "iC": (0.01, 0.01, 0.01),
        }
    )
    receiver_family = _receiver_family(unavailable)
    artifact = fit_receiver_family_scoring_artifact(
        receiver_family,
        np.asarray([[1.0, 3.0], [2.0, 3.0], [3.0, 3.0]]),
        contrast_name="stim_vs_ctrl",
        sample_ids=("s1", "s2", "s3"),
        sample_subject_ids=("p1", "p2", "p3"),
    )
    application = apply_receiver_family_scoring_artifact(
        artifact,
        np.asarray([[2.0, 3.0]]),
        feature_ids=("G1", "G2"),
        sample_subject_ids=("h1",),
    )

    assert artifact.downstream_functional is None
    assert artifact.reason_code == "no_training_eligible_receiver_family"
    assert artifact.certification_status == (
        "training_only_partial_receiver_family_not_estimable_v1"
    )
    assert application.downstream_application is None
    assert application.application_status == (
        "frozen_application_partial_not_estimable"
    )
    assert application.reason_code == artifact.reason_code


def test_row_order_and_mapping_order_do_not_change_training_artifact() -> None:
    first = _receiver_family(_availability())
    reordered = fit_receiver_family_training_artifact(
        _availability(reverse_rows=True),
        _prior({"A": {"G1": 1.0}, "B": {"G1": 4.0}, "C": {"G2": 1.0}}),
        receiver="Receiver",
        fold_id="fold-1",
        feature_ids=("G1", "G2"),
        driver_by_interaction={"iC": "C", "iB": "B", "iA": "A"},
        receptor_gate_threshold=0.1,
        cosine_threshold=0.999,
    )

    assert first.receptor_evidence_digest == reordered.receptor_evidence_digest
    assert first.receptor_gate_manifest_id == reordered.receptor_gate_manifest_id
    assert first.training_artifact_id == reordered.training_artifact_id


def test_batch_receiver_fit_reuses_one_static_partition_and_matches_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    availability = _multi_receiver_availability()
    prior = _prior({"A": {"G1": 1.0}, "B": {"G1": 4.0}, "C": {"G2": 1.0}})
    common = {
        "fold_id": "fold-1",
        "feature_ids": ("G1", "G2"),
        "driver_by_interaction": {"iA": "A", "iB": "B", "iC": "C"},
        "receptor_gate_threshold": 0.1,
        "cosine_threshold": 0.999,
    }
    independent = {
        receiver: fit_receiver_family_training_artifact(
            availability,
            prior,
            receiver=receiver,
            **common,
        )
        for receiver in ("Receiver", "ReceiverB")
    }

    import crychic.attribution.frozen_family as frozen_family_module

    original_cluster = frozen_family_module.cluster_driver_families
    calls = 0

    def counted_cluster(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original_cluster(*args, **kwargs)

    monkeypatch.setattr(
        frozen_family_module,
        "cluster_driver_families",
        counted_cluster,
    )
    bulk = fit_receiver_family_training_artifacts(
        availability,
        prior,
        receivers=("ReceiverB", "Receiver"),
        **common,
    )

    assert calls == 1
    assert tuple(artifact.receiver for artifact in bulk) == ("Receiver", "ReceiverB")
    for artifact in bulk:
        expected = independent[artifact.receiver]
        assert artifact.training_artifact_id == expected.training_artifact_id
        assert artifact.receptor_gate_manifest_id == expected.receptor_gate_manifest_id
        assert artifact.receptor_evidence_digest == expected.receptor_evidence_digest
        assert artifact.receptor_gates == expected.receptor_gates
        assert artifact.family_basis.family_basis_id == (
            expected.family_basis.family_basis_id
        )
        np.testing.assert_array_equal(
            artifact.source_basis.normalized_profiles.toarray(),
            expected.source_basis.normalized_profiles.toarray(),
        )
        np.testing.assert_array_equal(
            artifact.family_basis.matrix.toarray(),
            expected.family_basis.matrix.toarray(),
        )


def test_batch_receiver_fit_is_stable_to_receiver_order() -> None:
    availability = _multi_receiver_availability()
    prior = _prior({"A": {"G1": 1.0}, "B": {"G1": 4.0}, "C": {"G2": 1.0}})
    common = {
        "fold_id": "fold-1",
        "feature_ids": ("G1", "G2"),
        "driver_by_interaction": {"iA": "A", "iB": "B", "iC": "C"},
        "receptor_gate_threshold": 0.1,
        "cosine_threshold": 0.999,
    }

    forward = fit_receiver_family_training_artifacts(
        availability,
        prior,
        receivers=("Receiver", "ReceiverB"),
        **common,
    )
    reversed_order = fit_receiver_family_training_artifacts(
        availability,
        prior,
        receivers=("ReceiverB", "Receiver"),
        **common,
    )

    assert tuple(item.receiver for item in forward) == tuple(
        item.receiver for item in reversed_order
    )
    assert tuple(item.training_artifact_id for item in forward) == tuple(
        item.training_artifact_id for item in reversed_order
    )


def test_batch_receiver_fit_rejects_ambiguous_or_duplicate_receiver_input() -> None:
    availability = _multi_receiver_availability()
    prior = _prior({"A": {"G1": 1.0}, "B": {"G1": 4.0}, "C": {"G2": 1.0}})
    common = {
        "fold_id": "fold-1",
        "feature_ids": ("G1", "G2"),
        "driver_by_interaction": {"iA": "A", "iB": "B", "iC": "C"},
    }

    with pytest.raises(TypeError, match="sequence of receiver names"):
        fit_receiver_family_training_artifacts(
            availability,
            prior,
            receivers="Receiver",  # type: ignore[arg-type]
            **common,
        )
    with pytest.raises(ValueError, match="must be unique"):
        fit_receiver_family_training_artifacts(
            availability,
            prior,
            receivers=("Receiver", " Receiver "),
            **common,
        )


def test_contracts_are_producer_owned_and_apply_has_no_availability_input() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        ReceiverFamilyTrainingArtifact()
    with pytest.raises(TypeError, match="producer-owned"):
        ReceiverFamilyScoringArtifact()

    parameters = inspect.signature(apply_receiver_family_scoring_artifact).parameters
    assert "availability" not in parameters
    assert set(parameters) == {
        "artifact",
        "sample_expression",
        "feature_ids",
        "sample_subject_ids",
    }
    fit_parameters = inspect.signature(
        fit_receiver_family_scoring_artifact
    ).parameters
    assert "reference_input_digest" not in fit_parameters
    assert "sample_ids" in fit_parameters
