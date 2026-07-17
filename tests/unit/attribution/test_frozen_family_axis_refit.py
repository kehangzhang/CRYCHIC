from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence

import pandas as pd
import pytest

from crychic.attribution import (
    ReceiverFamilyAxisRefitResult,
    ReceiverFamilyTrainingArtifact,
    fit_receiver_family_training_artifact,
    refit_receiver_family_training_artifact_on_frozen_axis,
)
from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)
from crychic.core import ContractError
from crychic.resources import (
    GeneNamespace,
    MappingReport,
    Species,
    TargetPrior,
)

_OUTER_SUBJECTS = ("p1", "p2", "p3", "p4")
_INTERACTIONS = ("iA", "iB", "iC")
FrozenInputs = tuple[
    TargetPrior, FrozenInteractionUniverse, ReceiverFamilyTrainingArtifact
]


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


def _universe(*, min_pooled_availability: float = 0.0) -> FrozenInteractionUniverse:
    return FrozenInteractionUniverse(
        interaction_ids=_INTERACTIONS,
        training_subject_ids=_OUTER_SUBJECTS,
        resource_id="tiny_lr",
        resource_version="1",
        resource_manifest_digest="tiny-lr-manifest",
        min_pooled_availability=min_pooled_availability,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )


def _availability(
    universe: FrozenInteractionUniverse,
    *,
    subjects: Sequence[str],
    interactions: Sequence[str] = _INTERACTIONS,
    receiver_by_subject: Mapping[str, str] | None = None,
    receptor_value: float | None = None,
    training_selection: bool = False,
    reverse_rows: bool = False,
) -> BatchAvailability:
    values = {"iA": 0.8, "iB": 0.6, "iC": 0.4}
    rows = [
        {
            "sample_id": f"sample-{subject}",
            "subject_id": subject,
            "context_id": "reference",
            "receiver": (receiver_by_subject or {}).get(subject, "Receiver"),
            "interaction_id": interaction,
            "receptor_availability": (
                values[interaction] if receptor_value is None else receptor_value
            ),
        }
        for subject in subjects
        for interaction in interactions
    ]
    if reverse_rows:
        rows.reverse()
    return BatchAvailability(
        sample_interactions=pd.DataFrame(rows),
        mapping_summary=pd.DataFrame(),
        resource_id="tiny_lr",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=universe,
        filter_application=(
            InteractionFilterApplication.TRAINING_SELECTION_V1
            if training_selection
            else InteractionFilterApplication.FROZEN_APPLICATION_V1
        ),
        application_subject_ids=tuple(subjects),
    )


@pytest.fixture  # type: ignore[untyped-decorator]
def frozen_inputs() -> FrozenInputs:
    prior = _prior({"A": {"G1": 1.0}, "B": {"G1": 4.0}, "C": {"G2": 1.0}})
    universe = _universe()
    parent = fit_receiver_family_training_artifact(
        _availability(
            universe,
            subjects=_OUTER_SUBJECTS,
            training_selection=True,
        ),
        prior,
        receiver="Receiver",
        fold_id="outer-fold",
        feature_ids=("G1", "G2"),
        driver_by_interaction={"iA": "A", "iB": "B", "iC": "C"},
        receptor_gate_threshold=0.5,
        cosine_threshold=0.999,
    )
    return prior, universe, parent


def _refit(
    frozen_inputs: FrozenInputs,
    availability: BatchAvailability,
    *,
    training_subjects: Sequence[str] = ("p1", "p2"),
    validation_subjects: Sequence[str] = ("p3", "p4"),
) -> ReceiverFamilyAxisRefitResult:
    prior, _, parent = frozen_inputs
    return refit_receiver_family_training_artifact_on_frozen_axis(
        availability,
        prior,
        parent,
        fold_id="inner-fold-1",
        inner_training_subject_ids=training_subjects,
        inner_validation_subject_ids=validation_subjects,
    )


def test_inner_dropout_refits_gates_without_changing_frozen_family_axis(
    frozen_inputs: FrozenInputs,
) -> None:
    _, universe, parent = frozen_inputs
    result = _refit(
        frozen_inputs,
        _availability(universe, subjects=("p1", "p2"), interactions=("iA",)),
    )

    assert result.status == "observed"
    assert result.reason_code is None
    assert result.artifact is not None
    artifact = result.artifact
    assert parent.frozen_family_axis_parent_id is None
    assert artifact.frozen_family_axis_parent_id == parent.training_artifact_id
    assert artifact.training_subject_ids == ("p1", "p2")
    assert artifact.driver_by_interaction == parent.driver_by_interaction
    assert artifact.source_basis.feature_ids == parent.source_basis.feature_ids
    assert artifact.source_basis.driver_ids == parent.source_basis.driver_ids
    assert artifact.family_basis.family_ids == parent.family_basis.family_ids
    assert artifact.family_basis.family_definitions == (
        parent.family_basis.family_definitions
    )
    assert artifact.family_basis.medoid_driver_ids == (
        parent.family_basis.medoid_driver_ids
    )
    assert dict(artifact.receptor_gates) == {"A": 0.8, "B": 0.0, "C": 0.0}
    assert result.outer_parent_id == parent.training_artifact_id
    assert result.validation_subject_ids == ("p3", "p4")
    assert result.to_dict()["method"].endswith("v1")


def test_refit_is_stable_to_inner_evidence_row_order(
    frozen_inputs: FrozenInputs,
) -> None:
    _, universe, _ = frozen_inputs
    forward = _refit(
        frozen_inputs,
        _availability(universe, subjects=("p1", "p2")),
    )
    reversed_rows = _refit(
        frozen_inputs,
        _availability(
            universe,
            subjects=("p1", "p2"),
            reverse_rows=True,
        ),
    )

    assert forward.refit_id == reversed_rows.refit_id
    assert forward.artifact is not None
    assert reversed_rows.artifact is not None
    assert (
        forward.artifact.training_artifact_id
        == reversed_rows.artifact.training_artifact_id
    )


def test_validation_values_have_no_api_and_validation_rows_fail_closed(
    frozen_inputs: FrozenInputs,
) -> None:
    parameters = inspect.signature(
        refit_receiver_family_training_artifact_on_frozen_axis
    ).parameters
    assert "inner_validation_subject_ids" in parameters
    assert all(
        name == "inner_validation_subject_ids" or "validation" not in name
        for name in parameters
    )
    _, universe, _ = frozen_inputs
    contaminated = _availability(universe, subjects=("p1", "p2", "p3"))
    object.__setattr__(contaminated, "application_subject_ids", ("p1", "p2"))

    with pytest.raises(ContractError) as caught:
        _refit(frozen_inputs, contaminated)

    assert caught.value.details.code == "receiver_family_frozen_axis_subject_leakage"


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("training_subjects", "validation_subjects"),
    [
        (("p1", "p2"), ("p2", "p3", "p4")),
        (("p1", "p2"), ("p3",)),
    ],
)
def test_subject_overlap_or_incomplete_parent_partition_fails_closed(
    frozen_inputs: FrozenInputs,
    training_subjects: tuple[str, ...],
    validation_subjects: tuple[str, ...],
) -> None:
    _, universe, _ = frozen_inputs
    availability = _availability(universe, subjects=training_subjects)

    with pytest.raises(ContractError) as caught:
        _refit(
            frozen_inputs,
            availability,
            training_subjects=training_subjects,
            validation_subjects=validation_subjects,
        )

    assert caught.value.details.code in {
        "receiver_family_frozen_axis_subject_leakage",
        "receiver_family_frozen_axis_parent_scope_mismatch",
    }


def test_changed_interaction_universe_is_a_parent_mismatch(
    frozen_inputs: FrozenInputs,
) -> None:
    changed = _universe(min_pooled_availability=0.1)

    with pytest.raises(ContractError) as caught:
        _refit(
            frozen_inputs,
            _availability(changed, subjects=("p1", "p2")),
        )

    assert caught.value.details.code == "receiver_family_frozen_axis_parent_mismatch"


def test_statistical_subject_support_failures_are_typed_not_estimable(
    frozen_inputs: FrozenInputs,
) -> None:
    _, universe, _ = frozen_inputs
    one_training_subject = _refit(
        frozen_inputs,
        _availability(universe, subjects=("p1",)),
        training_subjects=("p1",),
        validation_subjects=("p2", "p3", "p4"),
    )
    one_receiver_subject = _refit(
        frozen_inputs,
        _availability(
            universe,
            subjects=("p1", "p2"),
            receiver_by_subject={"p2": "OtherReceiver"},
        ),
    )

    assert one_training_subject.status == "not_estimable"
    assert one_training_subject.reason_code == (
        "receiver_family_frozen_axis_insufficient_training_subjects"
    )
    assert one_training_subject.artifact is None
    assert one_receiver_subject.status == "not_estimable"
    assert one_receiver_subject.reason_code == (
        "receiver_family_frozen_axis_insufficient_receiver_subjects"
    )
    assert one_receiver_subject.artifact is None


def test_observed_zero_receptor_evidence_is_biological_absence_not_missing(
    frozen_inputs: FrozenInputs,
) -> None:
    _, universe, _ = frozen_inputs
    result = _refit(
        frozen_inputs,
        _availability(
            universe,
            subjects=("p1", "p2"),
            receptor_value=0.0,
        ),
    )

    assert result.status == "observed"
    assert result.artifact is not None
    assert result.artifact.eligible_family_ids == ()


def test_refit_result_and_parent_lineage_are_producer_owned_and_tamper_evident(
    frozen_inputs: FrozenInputs,
) -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        ReceiverFamilyAxisRefitResult()
    _, universe, _ = frozen_inputs
    result = _refit(
        frozen_inputs,
        _availability(universe, subjects=("p1", "p2")),
    )
    assert result.artifact is not None
    object.__setattr__(result.artifact, "frozen_family_axis_parent_id", "poisoned")

    with pytest.raises(ContractError) as caught:
        result.to_dict()

    assert caught.value.details.code == (
        "receiver_family_frozen_axis_refit_integrity_violation"
    )
