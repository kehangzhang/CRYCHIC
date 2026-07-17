from __future__ import annotations

from collections.abc import Hashable, Mapping

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import crychic.workflow.graph_fused as graph_fused_module
from crychic.attribution import (
    GraphPenaltyTuningSpec,
    ReceiverFamilyTrainingArtifact,
    fit_family_first_attribution,
    fit_receiver_family_training_artifact,
)
from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)
from crychic.core import ContractError, stable_id
from crychic.design import (
    ContextGraph,
    FrozenDesignApplication,
    FrozenDesignEncoder,
    apply_frozen_design_encoder,
    balanced_contrast,
    fit_frozen_design_encoder,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.resources import (
    GeneNamespace,
    MappingReport,
    Species,
    TargetPrior,
)
from crychic.response import (
    FoldGeneResponseApplication,
    FoldGeneResponseArtifact,
    apply_fold_gene_response,
    fit_fold_gene_response,
)
from crychic.workflow.graph_fused import (
    GraphFusedInnerPartition,
    GraphFusedWorkflowApplication,
    GraphFusedWorkflowArtifact,
    GraphFusedWorkflowSpec,
    apply_graph_fused_workflow,
    build_graph_fused_training_problem,
    fit_graph_fused_workflow,
    freeze_graph_fused_inner_fold_parents,
)


def _metadata(subjects: tuple[str, ...], *, prefix: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"{prefix}{subject}:{context}",
                "subject_id": f"{prefix}{subject}",
                "condition": context,
            }
            for subject in subjects
            for context in ("ctrl", "stim")
        ]
    )


def _independent_metadata(n_subjects_per_context: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"c{index}:ctrl",
                "subject_id": f"c{index}",
                "condition": "ctrl",
            }
            for index in range(1, n_subjects_per_context + 1)
        ]
        + [
            {
                "sample_id": f"s{index}:stim",
                "subject_id": f"s{index}",
                "condition": "stim",
            }
            for index in range(1, n_subjects_per_context + 1)
        ]
    )


def _aggregate(
    metadata: pd.DataFrame,
    *,
    poison_subject: str | None = None,
) -> PseudobulkDataset:
    counts: list[tuple[int, int]] = []
    units: list[dict[str, object]] = []
    matrix_ids: list[str] = []
    subject_order = {
        subject: index
        for index, subject in enumerate(sorted(metadata["subject_id"].unique()))
    }
    for _, row in metadata.iterrows():
        sample_id = str(row["sample_id"])
        subject_id = str(row["subject_id"])
        context = str(row["condition"])
        index = subject_order[subject_id]
        first = 20 + index + (35 if context == "stim" else 0)
        if subject_id == poison_subject:
            first = 99 if context == "ctrl" else 1
        values = (first, 120 - first)
        unit_id = f"unit:{sample_id}"
        matrix_row = len(counts)
        counts.append(values)
        matrix_ids.append(unit_id)
        units.append(
            {
                "unit_id": unit_id,
                "sample_id": sample_id,
                "subject_id": subject_id,
                "cell_type": "Receiver",
                "context": (("condition", context),),
                "matrix_row": matrix_row,
                "n_cells": 20,
                "cell_proportion": 1.0,
                "state_eligible": True,
                "abundance_eligible": True,
                "missingness_reason": "observed",
            }
        )
    matrix = sparse.csr_matrix(np.asarray(counts, dtype=np.int64))
    return PseudobulkDataset(
        counts=matrix,
        detection_fraction=sparse.csr_matrix(matrix.toarray() > 0),
        unit_metadata=pd.DataFrame(units),
        feature_ids=("G1", "G2"),
        matrix_unit_ids=tuple(matrix_ids),
        source_location="synthetic",
    )


def _prior(columns: Mapping[str, Mapping[str, float]]) -> TargetPrior:
    drivers = tuple(sorted(columns))
    targets = tuple(
        sorted({target for values in columns.values() for target in values})
    )
    target_index = {target: index for index, target in enumerate(targets)}
    indices: list[int] = []
    weights: list[float] = []
    indptr = [0]
    for driver in drivers:
        for target, weight in sorted(columns[driver].items()):
            indices.append(target_index[target])
            weights.append(weight)
        indptr.append(len(weights))
    return TargetPrior(
        resource_id="tiny-prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=targets,
        driver_ids=drivers,
        indptr=tuple(indptr),
        target_indices=tuple(indices),
        weights=tuple(weights),
        ranks=None,
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(len(weights), len(weights), len(targets)),
        manifest_digest="tiny-prior-manifest",
    )


def _family(
    subjects: tuple[str, ...],
    *,
    fold_id: str,
    columns: Mapping[str, Mapping[str, float]] | None = None,
    receptor_value: float = 0.9,
) -> ReceiverFamilyTrainingArtifact:
    prior_columns = columns or {"D1": {"G1": 1.0}, "D2": {"G2": 1.0}}
    interactions = tuple(f"i{index + 1}" for index in range(len(prior_columns)))
    drivers = tuple(sorted(prior_columns))
    universe = FrozenInteractionUniverse(
        interaction_ids=interactions,
        training_subject_ids=subjects,
        resource_id="tiny-lr",
        resource_version="1",
        resource_manifest_digest="tiny-lr-manifest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )
    availability = BatchAvailability(
        sample_interactions=pd.DataFrame(
            [
                {
                    "sample_id": f"{subject}:ctrl",
                    "subject_id": subject,
                    "context_id": "ctrl",
                    "receiver": "Receiver",
                    "interaction_id": interaction,
                    "receptor_availability": receptor_value,
                }
                for subject in subjects
                for interaction in interactions
            ]
        ),
        mapping_summary=pd.DataFrame(),
        resource_id="tiny-lr",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=universe,
        filter_application=InteractionFilterApplication.TRAINING_SELECTION_V1,
        application_subject_ids=subjects,
    )
    return fit_receiver_family_training_artifact(
        availability,
        _prior(prior_columns),
        receiver="Receiver",
        fold_id=fold_id,
        feature_ids=("G1", "G2"),
        driver_by_interaction=dict(zip(interactions, drivers, strict=True)),
        receptor_gate_threshold=0.1,
        cosine_threshold=0.99,
    )


def _encoder(metadata: pd.DataFrame) -> FrozenDesignEncoder:
    return fit_frozen_design_encoder(
        metadata,
        contrast=balanced_contrast(("stim",), ("ctrl",), name="stim_vs_ctrl"),
        context_keys=("condition",),
        formula="~ condition",
    )


def _family_mapping(
    graph: ContextGraph, family: ReceiverFamilyTrainingArtifact
) -> dict[Hashable, ReceiverFamilyTrainingArtifact]:
    return {node: family for node in graph.nodes}


def _spec(*, max_iterations: int = 1_000) -> GraphFusedWorkflowSpec:
    return GraphFusedWorkflowSpec(
        tuning_spec=GraphPenaltyTuningSpec(
            lambda1_values=(0.0,),
            lambda2_values=(0.0,),
            lambda_f_values=(0.0,),
        ),
        max_iterations=max_iterations,
        solver_tolerance=1.0e-9,
    )


def _partition(
    subjects: tuple[str, ...], graph: ContextGraph
) -> GraphFusedInnerPartition:
    midpoint = len(subjects) // 2
    validation_groups = (subjects[:midpoint], subjects[midpoint:])
    folds = []
    for index, validation in enumerate(validation_groups):
        training = tuple(subject for subject in subjects if subject not in validation)
        fold_id = f"inner-{index}"
        family = _family(training, fold_id=fold_id)
        folds.append(
            freeze_graph_fused_inner_fold_parents(
                fold_id=fold_id,
                graph=graph,
                receiver="Receiver",
                training_subject_ids=training,
                validation_subject_ids=validation,
                family_parents=_family_mapping(graph, family),
            )
        )
    return GraphFusedInnerPartition(
        outer_training_subject_ids=subjects,
        folds=tuple(folds),
    )


def _inner_frozen_availability(
    outer_subjects: tuple[str, ...],
    inner_subjects: tuple[str, ...],
    *,
    receptor_values: Mapping[str, float] | None = None,
    receiver_subjects: tuple[str, ...] | None = None,
) -> BatchAvailability:
    interactions = ("i1", "i2")
    values = {"i1": 0.9, "i2": 0.9} | dict(receptor_values or {})
    receiver_evidence = set(
        inner_subjects if receiver_subjects is None else receiver_subjects
    )
    universe = FrozenInteractionUniverse(
        interaction_ids=interactions,
        training_subject_ids=outer_subjects,
        resource_id="tiny-lr",
        resource_version="1",
        resource_manifest_digest="tiny-lr-manifest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )
    return BatchAvailability(
        sample_interactions=pd.DataFrame(
            [
                {
                    "sample_id": f"{subject}:ctrl",
                    "subject_id": subject,
                    "context_id": "ctrl",
                    "receiver": (
                        "Receiver" if subject in receiver_evidence else "OtherReceiver"
                    ),
                    "interaction_id": interaction,
                    "receptor_availability": values[interaction],
                }
                for subject in inner_subjects
                for interaction in interactions
            ]
        ),
        mapping_summary=pd.DataFrame(),
        resource_id="tiny-lr",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=universe,
        filter_application=InteractionFilterApplication.FROZEN_APPLICATION_V1,
        application_subject_ids=inner_subjects,
    )


def _frozen_axis_partition(
    subjects: tuple[str, ...],
    graph: ContextGraph,
    outer_family: ReceiverFamilyTrainingArtifact,
    *,
    insufficient_evidence_fold: int | None = None,
) -> GraphFusedInnerPartition:
    midpoint = len(subjects) // 2
    validation_groups = (subjects[:midpoint], subjects[midpoint:])
    folds = []
    prior = _prior({"D1": {"G1": 1.0}, "D2": {"G2": 1.0}})
    for index, validation in enumerate(validation_groups):
        training = tuple(subject for subject in subjects if subject not in validation)
        fold_id = f"frozen-axis-inner-{index}"
        receiver_subjects = (
            training[:1] if insufficient_evidence_fold == index else None
        )
        availability = {
            node: _inner_frozen_availability(
                subjects,
                training,
                receptor_values=(
                    {"i1": 0.0, "i2": 0.9}
                    if node == graph.nodes[0]
                    else {"i1": 0.9, "i2": 0.9}
                ),
                receiver_subjects=receiver_subjects,
            )
            for node in graph.nodes
        }
        folds.append(
            freeze_graph_fused_inner_fold_parents(
                fold_id=fold_id,
                graph=graph,
                receiver="Receiver",
                training_subject_ids=training,
                validation_subject_ids=validation,
                family_parents=_family_mapping(graph, outer_family),
                inner_training_availability=availability,
                target_prior=prior,
            )
        )
    return GraphFusedInnerPartition(
        outer_training_subject_ids=subjects,
        folds=tuple(folds),
    )


def _independent_partition(
    graph: ContextGraph,
) -> tuple[tuple[str, ...], GraphFusedInnerPartition]:
    subjects = tuple(
        sorted(
            (*tuple(f"c{i}" for i in range(1, 7)), *tuple(f"s{i}" for i in range(1, 7)))
        )
    )
    validation_groups = (
        ("c1", "c2", "c3", "s1", "s2", "s3"),
        ("c4", "c5", "c6", "s4", "s5", "s6"),
    )
    folds = []
    for index, validation in enumerate(validation_groups):
        training = tuple(subject for subject in subjects if subject not in validation)
        fold_id = f"independent-inner-{index}"
        family = _family(training, fold_id=fold_id)
        folds.append(
            freeze_graph_fused_inner_fold_parents(
                fold_id=fold_id,
                graph=graph,
                receiver="Receiver",
                training_subject_ids=training,
                validation_subject_ids=validation,
                family_parents=_family_mapping(graph, family),
            )
        )
    return subjects, GraphFusedInnerPartition(
        outer_training_subject_ids=subjects,
        folds=tuple(folds),
    )


def _outer_inputs() -> tuple[
    tuple[str, ...],
    pd.DataFrame,
    ContextGraph,
    ReceiverFamilyTrainingArtifact,
]:
    subjects = tuple(f"p{index}" for index in range(1, 9))
    metadata = _metadata(subjects)
    graph = ContextGraph.chain(("ctrl", "stim"))
    family = _family(subjects, fold_id="outer-fold")
    return subjects, metadata, graph, family


def _application_parents(
    *,
    poison_subject: str | None = None,
    heldout_subjects: tuple[str, ...] = ("q1", "q2"),
) -> tuple[
    GraphFusedWorkflowArtifact,
    FoldGeneResponseArtifact,
    FoldGeneResponseApplication,
    FrozenDesignApplication,
]:
    subjects, metadata, graph, family = _outer_inputs()
    heldout_metadata = _metadata(heldout_subjects)
    combined_metadata = pd.concat([metadata, heldout_metadata], ignore_index=True)
    aggregate = _aggregate(
        combined_metadata,
        poison_subject=poison_subject,
    )
    encoder = _encoder(metadata)
    artifact = fit_graph_fused_workflow(
        aggregate,
        metadata,
        encoder,
        _family_mapping(graph, family),
        graph,
        _partition(subjects, graph),
        receiver="Receiver",
        fold_id="outer-fold",
        training_input_digest="outer-training-input",
        outer_training_subject_ids=subjects,
        heldout_subject_ids=heldout_subjects,
        spec=_spec(),
    )
    training_response = fit_fold_gene_response(
        aggregate,
        encoder,
        receiver="Receiver",
        fold_id="outer-fold",
        training_input_digest="outer-training-input",
        min_subjects_per_context=2,
    )
    design_application = apply_frozen_design_encoder(encoder, heldout_metadata)
    response_application = apply_fold_gene_response(
        aggregate,
        training_response,
        design_application,
    )
    assert training_response.artifact_id == artifact.problem.response_artifact_id
    return artifact, training_response, response_application, design_application


def test_compact_problem_uses_training_pseudobulk_not_contrast_effect() -> None:
    subjects, metadata, graph, family = _outer_inputs()

    problem = build_graph_fused_training_problem(
        _aggregate(metadata),
        metadata,
        _encoder(metadata),
        _family_mapping(graph, family),
        graph,
        receiver="Receiver",
        fold_id="outer-fold",
        training_input_digest="outer-training-input",
        outer_training_subject_ids=subjects,
        heldout_subject_ids=("heldout-1", "heldout-2"),
        spec=_spec(),
    )

    ctrl, stim = problem.signed_responses
    np.testing.assert_allclose(ctrl, -stim, rtol=1e-10, atol=1e-10)
    assert problem.context_subject_counts == (8, 8)
    assert not ctrl.flags.writeable
    assert not problem.precision_weights[0].flags.writeable
    manifest = problem.to_dict()
    assert manifest["response_method"] == (
        "adjusted_balanced_one_vs_rest_context_effect_v1"
    )
    assert manifest["formal_inference_allowed"] is False
    assert manifest["p_value"] is None
    assert "signed_responses" not in manifest


def test_lambda_f_zero_bridge_matches_independent_family_first_fits() -> None:
    subjects, metadata, graph, family = _outer_inputs()
    partition = _partition(subjects, graph)

    artifact = fit_graph_fused_workflow(
        _aggregate(metadata),
        metadata,
        _encoder(metadata),
        _family_mapping(graph, family),
        graph,
        partition,
        receiver="Receiver",
        fold_id="outer-fold",
        training_input_digest="outer-training-input",
        outer_training_subject_ids=subjects,
        heldout_subject_ids=("heldout-1", "heldout-2"),
        spec=_spec(),
    )

    assert artifact.status == "observed"
    assert artifact.fit is not None
    expected = tuple(
        fit_family_first_attribution(
            basis,
            response,
            precision_weights=precision,
            tolerance=1.0e-9,
            kkt_tolerance=1.0e-9,
        )
        for basis, response, precision in zip(
            artifact.problem.family_bases,
            artifact.problem.signed_responses,
            artifact.problem.precision_weights,
            strict=True,
        )
    )
    np.testing.assert_array_equal(
        artifact.fit.attribution.coefficients,
        np.vstack([result.coefficients for result in expected]),
    )
    manifest = artifact.to_dict()
    assert manifest["inner_partition_id"]
    tuning_fold_data_ids = manifest["tuning_fold_data_ids"]
    assert isinstance(tuning_fold_data_ids, list)
    assert len(tuning_fold_data_ids) == 2
    assert manifest["formal_inference_allowed"] is False
    assert manifest["q_value"] is None
    assert tuple(fold.family_parent_set_id for fold in artifact._tuning_folds) == tuple(
        fold.parent_set_id for fold in partition.folds
    )


def test_workflow_rejects_rehashed_foreign_inner_partition_id() -> None:
    subjects, metadata, graph, family = _outer_inputs()
    artifact = fit_graph_fused_workflow(
        _aggregate(metadata),
        metadata,
        _encoder(metadata),
        _family_mapping(graph, family),
        graph,
        _partition(subjects, graph),
        receiver="Receiver",
        fold_id="outer-fold",
        training_input_digest="outer-training-input",
        outer_training_subject_ids=subjects,
        heldout_subject_ids=("heldout-1", "heldout-2"),
        spec=_spec(),
    )
    object.__setattr__(artifact, "inner_partition_id", "foreign-intact-partition")
    object.__setattr__(
        artifact,
        "workflow_id",
        stable_id(
            "graph_fused_workflow_artifact",
            artifact._identity_payload(),
            schema_version="1",
        ),
    )

    with pytest.raises(ContractError) as caught:
        artifact.to_dict()

    assert caught.value.details.code == "graph_fused_workflow_integrity_violation"


def test_workflow_rejects_rehashed_tuning_fold_parent_lineage() -> None:
    subjects, metadata, graph, family = _outer_inputs()
    artifact = fit_graph_fused_workflow(
        _aggregate(metadata),
        metadata,
        _encoder(metadata),
        _family_mapping(graph, family),
        graph,
        _partition(subjects, graph),
        receiver="Receiver",
        fold_id="outer-fold",
        training_input_digest="outer-training-input",
        outer_training_subject_ids=subjects,
        heldout_subject_ids=("heldout-1", "heldout-2"),
        spec=_spec(),
    )
    tuning_fold = artifact._tuning_folds[0]
    object.__setattr__(tuning_fold, "family_parent_set_id", "foreign-parent-set")
    object.__setattr__(
        tuning_fold,
        "fold_data_id",
        stable_id(
            "graph_tuning_fold",
            tuning_fold._identity_payload(),
            schema_version="1",
        ),
    )
    object.__setattr__(
        artifact,
        "tuning_fold_data_ids",
        tuple(fold.fold_data_id for fold in artifact._tuning_folds),
    )
    assert artifact.fit is not None
    tuning = artifact.fit.tuning
    object.__setattr__(
        tuning,
        "fold_data_ids",
        tuple(fold.fold_data_id for fold in artifact._tuning_folds),
    )
    object.__setattr__(
        artifact,
        "workflow_id",
        stable_id(
            "graph_fused_workflow_artifact",
            artifact._identity_payload(),
            schema_version="1",
        ),
    )

    with pytest.raises(ContractError) as caught:
        artifact.to_dict()

    assert caught.value.details.code == "graph_fused_workflow_integrity_violation"


def test_inner_family_gates_refit_on_the_exact_frozen_outer_axis() -> None:
    subjects, metadata, graph, family = _outer_inputs()
    partition = _frozen_axis_partition(subjects, graph, family)

    first_fold = partition.folds[0]
    assert first_fold.uses_frozen_family_axis_refits is True
    assert first_fold.status == "observed"
    assert len(first_fold.family_axis_refits) == len(graph.nodes)
    assert all(
        refit.outer_parent_id == family.training_artifact_id
        for refit in first_fold.family_axis_refits
    )
    assert all(
        parent.frozen_family_axis_parent_id == family.training_artifact_id
        for parent in first_fold.family_parents
    )
    ctrl_parent, stim_parent = first_fold.family_parents
    assert ctrl_parent.family_basis.family_definitions == (
        family.family_basis.family_definitions
    )
    assert ctrl_parent.family_basis.medoid_driver_ids == (
        family.family_basis.medoid_driver_ids
    )
    assert dict(ctrl_parent.receptor_gates) == {"D1": 0.0, "D2": 0.9}
    assert dict(stim_parent.receptor_gates) == {"D1": 0.9, "D2": 0.9}

    artifact = fit_graph_fused_workflow(
        _aggregate(metadata),
        metadata,
        _encoder(metadata),
        _family_mapping(graph, family),
        graph,
        partition,
        receiver="Receiver",
        fold_id="outer-fold",
        training_input_digest="outer-training-input",
        outer_training_subject_ids=subjects,
        heldout_subject_ids=("heldout-1", "heldout-2"),
        spec=_spec(),
    )

    assert artifact.status == "observed"
    assert artifact.fit is not None


def test_legacy_inner_parent_identity_payload_remains_stable() -> None:
    subjects, _, graph, _ = _outer_inputs()
    validation = subjects[:4]
    training = subjects[4:]
    family = _family(training, fold_id="legacy-inner")

    frozen = freeze_graph_fused_inner_fold_parents(
        fold_id="legacy-inner",
        graph=graph,
        receiver="Receiver",
        training_subject_ids=training,
        validation_subject_ids=validation,
        family_parents=_family_mapping(graph, family),
    )

    assert frozen.family_axis_refits == ()
    assert frozen.parent_set_id == stable_id(
        "graph_fused_inner_fold_parents",
        {
            "family_parent_ids": [family.training_artifact_id for _ in graph.nodes],
            "fold_id": "legacy-inner",
            "graph_id": stable_id("context_graph", graph.to_dict(), schema_version="1"),
            "training_subject_ids": list(training),
            "validation_subject_ids": list(validation),
        },
        schema_version="1",
    )


def test_inner_partition_cannot_mix_legacy_and_frozen_axis_modes() -> None:
    subjects, _, graph, family = _outer_inputs()
    legacy = _partition(subjects, graph)
    strict = _frozen_axis_partition(subjects, graph, family)

    with pytest.raises(ContractError) as caught:
        GraphFusedInnerPartition(
            outer_training_subject_ids=subjects,
            folds=(legacy.folds[0], strict.folds[1]),
        )

    assert caught.value.details.code == ("graph_fused_inner_family_refit_mode_mismatch")


def test_inner_family_refit_insufficient_evidence_is_typed_not_estimable() -> None:
    subjects, metadata, graph, family = _outer_inputs()
    partition = _frozen_axis_partition(
        subjects,
        graph,
        family,
        insufficient_evidence_fold=0,
    )

    unavailable = next(
        fold for fold in partition.folds if fold.status == "not_estimable"
    )
    assert unavailable.family_parents == ()
    assert unavailable.reason_code == (
        "graph_fused_inner_family_refit_not_estimable:"
        "receiver_family_frozen_axis_insufficient_receiver_subjects"
    )

    artifact = fit_graph_fused_workflow(
        _aggregate(metadata),
        metadata,
        _encoder(metadata),
        _family_mapping(graph, family),
        graph,
        partition,
        receiver="Receiver",
        fold_id="outer-fold",
        training_input_digest="outer-training-input",
        outer_training_subject_ids=subjects,
        heldout_subject_ids=("heldout-1", "heldout-2"),
        spec=_spec(),
    )

    assert artifact.status == "not_estimable"
    assert artifact.fit is None
    assert artifact.tuning_fold_data_ids == ()
    assert artifact.reason_code == unavailable.reason_code


def test_single_inner_training_subject_is_typed_ne_only_on_strict_path() -> None:
    subjects = ("p1", "p2", "p3", "p4")
    graph = ContextGraph.chain(("ctrl", "stim"))
    family = _family(subjects, fold_id="small-outer")
    availability = _inner_frozen_availability(subjects, ("p4",))

    strict = freeze_graph_fused_inner_fold_parents(
        fold_id="small-inner",
        graph=graph,
        receiver="Receiver",
        training_subject_ids=("p4",),
        validation_subject_ids=("p1", "p2", "p3"),
        family_parents=_family_mapping(graph, family),
        inner_training_availability={node: availability for node in graph.nodes},
        target_prior=_prior({"D1": {"G1": 1.0}, "D2": {"G2": 1.0}}),
    )

    assert strict.status == "not_estimable"
    assert strict.reason_code == (
        "graph_fused_inner_family_refit_not_estimable:"
        "receiver_family_frozen_axis_insufficient_training_subjects"
    )
    with pytest.raises(ValueError, match="at least 2"):
        freeze_graph_fused_inner_fold_parents(
            fold_id="legacy-small-inner",
            graph=graph,
            receiver="Receiver",
            training_subject_ids=("p4",),
            validation_subject_ids=("p1", "p2", "p3"),
            family_parents=_family_mapping(
                graph,
                _family(("p4",), fold_id="legacy-small-inner"),
            ),
        )


def test_inner_family_refit_binds_the_exact_outer_parent_ids() -> None:
    subjects, metadata, graph, family = _outer_inputs()
    alternate = _family(
        subjects,
        fold_id="outer-fold",
        receptor_value=0.8,
    )
    assert alternate.family_basis.family_definitions == (
        family.family_basis.family_definitions
    )
    assert alternate.training_artifact_id != family.training_artifact_id
    partition = _frozen_axis_partition(subjects, graph, alternate)

    with pytest.raises(ContractError) as caught:
        fit_graph_fused_workflow(
            _aggregate(metadata),
            metadata,
            _encoder(metadata),
            _family_mapping(graph, family),
            graph,
            partition,
            receiver="Receiver",
            fold_id="outer-fold",
            training_input_digest="outer-training-input",
            outer_training_subject_ids=subjects,
            heldout_subject_ids=("heldout-1", "heldout-2"),
            spec=_spec(),
        )

    assert caught.value.details.code == (
        "graph_fused_inner_family_refit_parent_mismatch"
    )


def test_fold_external_expression_cannot_change_outer_training_problem() -> None:
    subjects, metadata, graph, family = _outer_inputs()
    heldout = _metadata(("q1",), prefix="heldout-")
    combined = pd.concat([metadata, heldout], ignore_index=True)
    common = {
        "sample_metadata": metadata,
        "design_encoder": _encoder(metadata),
        "family_parents": _family_mapping(graph, family),
        "graph": graph,
        "receiver": "Receiver",
        "fold_id": "outer-fold",
        "training_input_digest": "outer-training-input",
        "outer_training_subject_ids": subjects,
        "heldout_subject_ids": ("heldout-q1",),
        "spec": _spec(),
    }

    ordinary = build_graph_fused_training_problem(
        _aggregate(combined),
        **common,
    )
    poisoned = build_graph_fused_training_problem(
        _aggregate(combined, poison_subject="heldout-q1"),
        **common,
    )

    assert ordinary.problem_id == poisoned.problem_id
    for left, right in zip(
        ordinary.signed_responses, poisoned.signed_responses, strict=True
    ):
        np.testing.assert_array_equal(left, right)


def test_inner_validation_projection_supports_independent_subject_groups() -> None:
    metadata = _independent_metadata(6)
    graph = ContextGraph.chain(("ctrl", "stim"))
    subjects, partition = _independent_partition(graph)
    family = _family(subjects, fold_id="outer-independent")

    artifact = fit_graph_fused_workflow(
        _aggregate(metadata),
        metadata,
        _encoder(metadata),
        _family_mapping(graph, family),
        graph,
        partition,
        receiver="Receiver",
        fold_id="outer-independent",
        training_input_digest="outer-independent-input",
        outer_training_subject_ids=subjects,
        heldout_subject_ids=("heldout",),
        spec=_spec(),
    )

    assert artifact.status == "observed"
    assert artifact.fit is not None
    assert artifact.fit.tuning.validation_subject_ids == subjects


def test_family_axis_and_graph_coverage_mismatches_fail_closed() -> None:
    subjects, metadata, graph, family = _outer_inputs()
    alternate = _family(
        subjects,
        fold_id="outer-fold",
        columns={"D1": {"G1": 1.0}},
    )
    mismatched: dict[Hashable, ReceiverFamilyTrainingArtifact] = {
        "ctrl": family,
        "stim": alternate,
    }

    with pytest.raises(ContractError) as axis_error:
        build_graph_fused_training_problem(
            _aggregate(metadata),
            metadata,
            _encoder(metadata),
            mismatched,
            graph,
            receiver="Receiver",
            fold_id="outer-fold",
            training_input_digest="outer-training-input",
            outer_training_subject_ids=subjects,
            heldout_subject_ids=("heldout",),
            spec=_spec(),
        )
    assert axis_error.value.details.code == "graph_fused_family_axis_mismatch"

    expanded = ContextGraph.chain(("ctrl", "middle", "stim"))
    with pytest.raises(ContractError) as graph_error:
        build_graph_fused_training_problem(
            _aggregate(metadata),
            metadata,
            _encoder(metadata),
            _family_mapping(expanded, family),
            expanded,
            receiver="Receiver",
            fold_id="outer-fold",
            training_input_digest="outer-training-input",
            outer_training_subject_ids=subjects,
            heldout_subject_ids=("heldout",),
            spec=_spec(),
        )
    assert graph_error.value.details.code == "graph_fused_graph_node_missing"


def test_solver_failure_is_a_typed_nonreleasing_artifact() -> None:
    subjects, metadata, graph, family = _outer_inputs()

    artifact = fit_graph_fused_workflow(
        _aggregate(metadata),
        metadata,
        _encoder(metadata),
        _family_mapping(graph, family),
        graph,
        _partition(subjects, graph),
        receiver="Receiver",
        fold_id="outer-fold",
        training_input_digest="outer-training-input",
        outer_training_subject_ids=subjects,
        heldout_subject_ids=("heldout",),
        spec=_spec(max_iterations=1),
    )

    assert artifact.status == "failed"
    assert artifact.fit is None
    assert artifact.reason_code == "graph_tuning_no_estimable_candidate"
    assert artifact.formal_inference_allowed is False


def test_frozen_application_is_auditable_and_never_refits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        artifact,
        _,
        response_application,
        design_application,
    ) = _application_parents()

    def _forbidden_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("held-out graph application must not fit")

    for name in (
        "fit_fold_gene_response",
        "fit_frozen_design_encoder",
        "fit_tuned_graph_fused_family_attribution",
    ):
        monkeypatch.setattr(graph_fused_module, name, _forbidden_fit)

    application = apply_graph_fused_workflow(
        artifact,
        response_application,
        design_application,
    )

    assert application.status == "observed"
    assert application.reason_code is None
    assert artifact.fit is not None
    assert application.fit_id == artifact.fit.tuned_attribution_id
    assert application.tuning_id == artifact.fit.tuning.tuning_id
    assert application.application_functional_id == (
        artifact.problem.application_functional_id
    )
    assert not set(application.training_subject_ids).intersection(
        application.heldout_subject_ids
    )
    assert application.fixed_predictions is not None
    assert application.fixed_precision_weights is not None
    assert application.heldout_positive_responses is not None
    assert application.residuals is not None
    assert application.context_losses is not None
    assert application.subject_losses is not None
    np.testing.assert_allclose(
        application.fixed_predictions,
        np.vstack(artifact.fit.attribution.predicted),
    )
    np.testing.assert_allclose(
        application.heldout_positive_responses,
        application.fixed_predictions[np.newaxis, :, :] + application.residuals,
    )
    weight_sums = application.fixed_precision_weights.sum(axis=1)
    expected_subject_losses = (
        application.context_losses * weight_sums[np.newaxis, :]
    ).sum(axis=1) / weight_sums.sum()
    np.testing.assert_allclose(application.subject_losses, expected_subject_losses)
    assert application.mean_subject_loss == pytest.approx(
        float(np.mean(application.subject_losses))
    )
    application.require_compatible(artifact, response_application, design_application)
    manifest = application.to_dict()
    assert manifest["heldout_input_digest"]
    assert manifest["fixed_prediction_digest"]
    assert manifest["residual_digest"]
    assert manifest["loss_estimand"] == (
        "subject_equal_full_context_weighted_prediction_loss_v1"
    )
    with pytest.raises(TypeError):
        GraphFusedWorkflowApplication()


def test_heldout_poison_changes_only_application_lineage_and_loss() -> None:
    ordinary_artifact, _, ordinary_response, ordinary_design = _application_parents()
    poisoned_artifact, _, poisoned_response, poisoned_design = _application_parents(
        poison_subject="q1"
    )

    assert ordinary_artifact.workflow_id == poisoned_artifact.workflow_id
    assert ordinary_artifact.problem.problem_id == poisoned_artifact.problem.problem_id
    assert ordinary_artifact.fit is not None
    assert poisoned_artifact.fit is not None
    assert (
        ordinary_artifact.fit.tuning.tuning_id == poisoned_artifact.fit.tuning.tuning_id
    )
    assert (
        ordinary_artifact.fit.selected_candidate_id
        == poisoned_artifact.fit.selected_candidate_id
    )
    ordinary = apply_graph_fused_workflow(
        ordinary_artifact, ordinary_response, ordinary_design
    )
    poisoned = apply_graph_fused_workflow(
        poisoned_artifact, poisoned_response, poisoned_design
    )

    assert ordinary.response_application_id != poisoned.response_application_id
    assert ordinary.heldout_input_digest != poisoned.heldout_input_digest
    assert ordinary.application_id != poisoned.application_id
    assert ordinary.mean_subject_loss != poisoned.mean_subject_loss


def test_application_rejects_scope_and_integrity_mismatches() -> None:
    artifact, _, response_application, design_application = _application_parents()
    _, _, wrong_response, wrong_design = _application_parents(
        heldout_subjects=("q1", "q3")
    )

    with pytest.raises(ContractError) as scope_error:
        apply_graph_fused_workflow(artifact, wrong_response, wrong_design)
    assert scope_error.value.details.code == ("graph_fused_application_parent_mismatch")

    application = apply_graph_fused_workflow(
        artifact, response_application, design_application
    )
    object.__setattr__(application, "application_id", "tampered")
    with pytest.raises(ContractError) as application_error:
        application.to_dict()
    assert application_error.value.details.code == (
        "graph_fused_application_integrity_violation"
    )

    object.__setattr__(response_application, "sample_values_digest", "tampered")
    with pytest.raises(ContractError) as response_error:
        apply_graph_fused_workflow(artifact, response_application, design_application)
    assert response_error.value.details.code == (
        "fold_gene_response_application_integrity_violation"
    )


def test_not_estimable_design_releases_no_application_scores() -> None:
    artifact, training_response, _, _ = _application_parents()
    subjects, metadata, _, _ = _outer_inputs()
    assert subjects == artifact.problem.training_subject_ids
    unseen_metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{subject}:novel",
                "subject_id": subject,
                "condition": "novel",
            }
            for subject in artifact.problem.heldout_subject_ids
        ]
    )
    design_application = apply_frozen_design_encoder(
        _encoder(metadata), unseen_metadata
    )
    response_application = apply_fold_gene_response(
        _aggregate(unseen_metadata),
        training_response,
        design_application,
    )

    application = apply_graph_fused_workflow(
        artifact, response_application, design_application
    )

    assert design_application.status == "not_estimable"
    assert response_application.status == "not_estimable"
    assert application.status == "not_estimable"
    assert application.reason_code is not None
    assert application.fixed_predictions is None
    assert application.heldout_positive_responses is None
    assert application.subject_losses is None
    assert application.mean_subject_loss is None
    assert application.to_dict()["subject_losses"] is None
