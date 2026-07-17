from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from crychic.attribution import (
    DirectionalContrastPairSpec,
    ReceiverFamilyTrainingArtifact,
    fit_receiver_family_training_artifact,
    fit_response_precision,
)
from crychic.attribution.precision import fit_repeated_response_precision
from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)
from crychic.core import ContractError
from crychic.design import (
    FrozenDesignEncoder,
    apply_frozen_design_encoder,
    balanced_contrast,
    fit_frozen_design_encoder,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.resources import GeneNamespace, MappingReport, Species, TargetPrior
from crychic.response import (
    FoldGeneResponseArtifact,
    RepeatedMeasuresCR2Status,
    apply_fold_gene_response,
    fit_fold_gene_response,
)
from crychic.response.repeated_fold import (
    RepeatedMeasuresFoldResponseArtifact,
    apply_repeated_measures_fold_response,
    fit_repeated_measures_cr2_fold_response,
)
from crychic.workflow import (
    DirectionalCrossFitBinding,
    ReceiverIncrementalApplication,
    ReceiverIncrementalTrainingArtifact,
    apply_receiver_incremental_training_artifact,
    build_directional_crossfit_binding,
    fit_receiver_incremental_training_artifact,
)
from crychic.workflow.directional import _require_response_pair_scope


def _metadata(subjects: tuple[str, ...], *, prefix: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"{prefix}{subject}:{condition}",
                "subject_id": f"{prefix}{subject}",
                "condition": condition,
            }
            for subject in subjects
            for condition in ("ctrl", "stim")
        ]
    )


def _aggregate(
    metadata: pd.DataFrame,
    *,
    offset: int = 0,
) -> PseudobulkDataset:
    counts: list[tuple[int, int]] = []
    units: list[dict[str, object]] = []
    matrix_ids: list[str] = []
    for position, row in enumerate(metadata.itertuples(index=False)):
        sample_id = str(row.sample_id)
        condition = str(row.condition)
        first = 12 + position + offset + (18 if condition == "stim" else 0)
        unit_id = f"unit:{sample_id}"
        counts.append((first, 120 - first))
        matrix_ids.append(unit_id)
        units.append(
            {
                "unit_id": unit_id,
                "sample_id": sample_id,
                "subject_id": str(row.subject_id),
                "cell_type": "Receiver",
                "context": (("condition", condition),),
                "matrix_row": position,
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
    targets = tuple(sorted({key for values in columns.values() for key in values}))
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


def _receiver_family(
    subjects: tuple[str, ...],
    *,
    receptor_value: float,
) -> ReceiverFamilyTrainingArtifact:
    universe = FrozenInteractionUniverse(
        interaction_ids=("i1", "i2"),
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
                for interaction in ("i1", "i2")
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
        _prior({"D1": {"G1": 1.0}, "D2": {"G2": 1.0}}),
        receiver="Receiver",
        fold_id="fold-1",
        feature_ids=("G1", "G2"),
        driver_by_interaction={"i1": "D1", "i2": "D2"},
        receptor_gate_threshold=0.1,
        cosine_threshold=0.99,
    )


def _encoders(
    metadata: pd.DataFrame,
) -> tuple[
    DirectionalContrastPairSpec,
    FrozenDesignEncoder,
    FrozenDesignEncoder,
]:
    forward = balanced_contrast(
        ("stim",), ("ctrl",), name="stim_vs_ctrl", family="treatment"
    )
    reverse = balanced_contrast(
        ("ctrl",), ("stim",), name="ctrl_vs_stim", family="treatment"
    )
    return (
        DirectionalContrastPairSpec(
            forward_contrast=forward,
            reverse_contrast=reverse,
        ),
        fit_frozen_design_encoder(
            metadata,
            contrast=forward,
            context_keys=("condition",),
            formula="~ condition",
        ),
        fit_frozen_design_encoder(
            metadata,
            contrast=reverse,
            context_keys=("condition",),
            formula="~ condition",
        ),
    )


def _chains(
    *,
    receptor_value: float = 0.9,
    reverse_receptor_value: float | None = None,
    heldout_subjects: tuple[str, ...] = ("q1", "q2"),
) -> tuple[
    DirectionalContrastPairSpec,
    FoldGeneResponseArtifact,
    FoldGeneResponseArtifact,
    ReceiverIncrementalTrainingArtifact,
    ReceiverIncrementalTrainingArtifact,
    ReceiverIncrementalApplication,
    ReceiverIncrementalApplication,
]:
    training_metadata = _metadata(("p1", "p2", "p3", "p4"))
    pair_spec, forward_encoder, reverse_encoder = _encoders(training_metadata)
    training_aggregate = _aggregate(training_metadata)
    forward_response = fit_fold_gene_response(
        training_aggregate,
        forward_encoder,
        receiver="Receiver",
        fold_id="fold-1",
        training_input_digest="training-input",
    )
    reverse_response = fit_fold_gene_response(
        training_aggregate,
        reverse_encoder,
        receiver="Receiver",
        fold_id="fold-1",
        training_input_digest="training-input",
    )
    family = _receiver_family(
        forward_response.training_subject_ids,
        receptor_value=receptor_value,
    )
    reverse_family = (
        family
        if reverse_receptor_value is None
        else _receiver_family(
            forward_response.training_subject_ids,
            receptor_value=reverse_receptor_value,
        )
    )
    forward_model = fit_receiver_incremental_training_artifact(
        forward_encoder,
        forward_response,
        fit_response_precision(forward_response, min_positive_features=2),
        family,
    )
    reverse_model = fit_receiver_incremental_training_artifact(
        reverse_encoder,
        reverse_response,
        fit_response_precision(reverse_response, min_positive_features=2),
        reverse_family,
    )
    heldout_metadata = _metadata(heldout_subjects)
    heldout_aggregate = _aggregate(heldout_metadata, offset=7)
    forward_design = apply_frozen_design_encoder(forward_encoder, heldout_metadata)
    reverse_design = apply_frozen_design_encoder(reverse_encoder, heldout_metadata)
    forward_response_application = apply_fold_gene_response(
        heldout_aggregate, forward_response, forward_design
    )
    reverse_response_application = apply_fold_gene_response(
        heldout_aggregate, reverse_response, reverse_design
    )
    forward_application = apply_receiver_incremental_training_artifact(
        forward_model, forward_response_application, forward_design
    )
    reverse_application = apply_receiver_incremental_training_artifact(
        reverse_model, reverse_response_application, reverse_design
    )
    return (
        pair_spec,
        forward_response,
        reverse_response,
        forward_model,
        reverse_model,
        forward_application,
        reverse_application,
    )


def _binding(*, receptor_value: float = 0.9) -> DirectionalCrossFitBinding:
    return build_directional_crossfit_binding(*_chains(receptor_value=receptor_value))


def _repeated_metadata() -> pd.DataFrame:
    allocations = {
        "p1": ("ctrl", "stim"),
        "p2": ("ctrl", "stim"),
        "c1": ("ctrl",),
        "c2": ("ctrl",),
        "c3": ("ctrl",),
        "s1": ("stim",),
        "s2": ("stim",),
        "s3": ("stim",),
    }
    return pd.DataFrame(
        [
            {
                "sample_id": f"{subject}:{condition}",
                "subject_id": subject,
                "condition": condition,
            }
            for subject, conditions in allocations.items()
            for condition in conditions
        ]
    )


def _repeated_chains(
    *,
    reverse_min_subject_clusters: int = 6,
) -> tuple[
    DirectionalContrastPairSpec,
    RepeatedMeasuresFoldResponseArtifact,
    RepeatedMeasuresFoldResponseArtifact,
    ReceiverIncrementalTrainingArtifact,
    ReceiverIncrementalTrainingArtifact,
    ReceiverIncrementalApplication,
    ReceiverIncrementalApplication,
]:
    training_metadata = _repeated_metadata()
    pair_spec, forward_encoder, reverse_encoder = _encoders(training_metadata)
    aggregate = _aggregate(training_metadata)
    shared_arguments: dict[str, object] = {
        "receiver": "Receiver",
        "fold_id": "fold-1",
        "training_input_digest": "training-input",
        "min_subjects_per_context": 2,
    }
    forward_response = fit_repeated_measures_cr2_fold_response(
        aggregate,
        forward_encoder,
        training_metadata,
        min_subject_clusters=6,
        **shared_arguments,  # type: ignore[arg-type]
    )
    reverse_response = fit_repeated_measures_cr2_fold_response(
        aggregate,
        reverse_encoder,
        training_metadata,
        min_subject_clusters=reverse_min_subject_clusters,
        **shared_arguments,  # type: ignore[arg-type]
    )
    family = _receiver_family(
        forward_response.training_subject_ids,
        receptor_value=0.9,
    )
    forward_model = fit_receiver_incremental_training_artifact(
        forward_encoder,
        forward_response,
        fit_repeated_response_precision(forward_response),
        family,
    )
    reverse_model = fit_receiver_incremental_training_artifact(
        reverse_encoder,
        reverse_response,
        fit_repeated_response_precision(reverse_response),
        family,
    )
    heldout_metadata = _metadata(("q1", "q2"))
    heldout_aggregate = _aggregate(heldout_metadata, offset=7)
    forward_design = apply_frozen_design_encoder(forward_encoder, heldout_metadata)
    reverse_design = apply_frozen_design_encoder(reverse_encoder, heldout_metadata)
    forward_response_application = apply_repeated_measures_fold_response(
        heldout_aggregate,
        forward_response,
        forward_design,
    )
    reverse_response_application = apply_repeated_measures_fold_response(
        heldout_aggregate,
        reverse_response,
        reverse_design,
    )
    forward_application = apply_receiver_incremental_training_artifact(
        forward_model,
        forward_response_application,
        forward_design,
    )
    reverse_application = apply_receiver_incremental_training_artifact(
        reverse_model,
        reverse_response_application,
        reverse_design,
    )
    return (
        pair_spec,
        forward_response,
        reverse_response,
        forward_model,
        reverse_model,
        forward_application,
        reverse_application,
    )


def test_observed_binding_retains_explicit_nonadditive_directional_semantics() -> None:
    first = _binding()
    repeated = _binding()

    assert first.status == "observed"
    assert first.reason_code is None
    assert first.response_pair is not None
    assert first.forward_channel_name == "increased_activation_compatible"
    assert first.reverse_channel_name == "reduced_activation_compatible"
    assert first.formal_inference_allowed is False
    assert first.active_inhibition_allowed is False
    assert first.supports_active_inhibition_claim is False
    assert first.paired_score_comparison_allowed is False
    assert first.combination_rule == "independent_contrast_views_not_additive_v1"
    assert first.training_subject_ids == ("p1", "p2", "p3", "p4")
    assert first.heldout_subject_ids == ("q1", "q2")
    assert set(first.training_subject_ids).isdisjoint(first.heldout_subject_ids)
    assert first.binding_id == repeated.binding_id
    payload = first.to_dict()
    assert payload["response_pair"] is not None
    assert payload["pair_spec_id"] == first.pair_spec.pair_spec_id


def test_legal_unavailable_incremental_components_produce_complete_typed_ne() -> None:
    binding = _binding(receptor_value=0.01)

    assert binding.status == "not_estimable"
    assert binding.response_pair is None
    assert binding.reason_code == (
        "directional_components_not_estimable["
        "forward_model:no_training_eligible_receiver_family;"
        "reverse_model:no_training_eligible_receiver_family;"
        "forward_application:no_training_eligible_receiver_family;"
        "reverse_application:no_training_eligible_receiver_family]"
    )
    assert binding.receiver == "Receiver"
    assert binding.forward_response_id
    assert binding.reverse_response_id
    assert binding.forward_training_artifact_id
    assert binding.reverse_training_artifact_id
    assert binding.forward_application_id
    assert binding.reverse_application_id
    binding.to_dict()


def test_builder_maps_direction_by_registered_contrast_name_not_tuple_position() -> (
    None
):
    parents = _chains()
    with pytest.raises(ContractError) as error:
        build_directional_crossfit_binding(
            parents[0],
            parents[2],
            parents[1],
            parents[3],
            parents[4],
            parents[5],
            parents[6],
        )
    assert error.value.details.code == "directional_crossfit_parent_mismatch"
    assert error.value.details.field == "forward_response"


def test_builder_rejects_swapped_models_and_mismatched_heldout_scope() -> None:
    parents = _chains()
    with pytest.raises(ContractError) as model_error:
        build_directional_crossfit_binding(
            parents[0],
            parents[1],
            parents[2],
            parents[4],
            parents[3],
            parents[5],
            parents[6],
        )
    assert model_error.value.details.code == "directional_crossfit_parent_mismatch"
    assert model_error.value.details.field == "forward_model"

    other = _chains(heldout_subjects=("r1", "r2"))
    with pytest.raises(ContractError) as heldout_error:
        build_directional_crossfit_binding(
            parents[0],
            parents[1],
            parents[2],
            parents[3],
            parents[4],
            parents[5],
            other[6],
        )
    assert heldout_error.value.details.code == "directional_crossfit_parent_mismatch"
    assert heldout_error.value.details.field == "reverse_application"


def test_builder_rejects_distinct_frozen_family_parents() -> None:
    parents = _chains(reverse_receptor_value=0.8)
    with pytest.raises(ContractError) as error:
        build_directional_crossfit_binding(*parents)
    assert error.value.details.code == "directional_crossfit_parent_mismatch"
    assert error.value.details.field == "reverse_model"


def test_repeated_builder_rejects_asymmetric_design_policy() -> None:
    parents = _repeated_chains(reverse_min_subject_clusters=7)

    assert parents[1].status == "ok"
    assert parents[2].status == "ok"
    with pytest.raises(ContractError) as error:
        build_directional_crossfit_binding(*parents)

    assert error.value.details.code == "directional_crossfit_parent_mismatch"
    assert error.value.details.field == "reverse_response.repeated_design.spec"


def test_repeated_scope_rejects_asymmetric_feature_status_and_support() -> None:
    pair_spec, forward, reverse, *_ = _repeated_chains()
    reverse_feature = reverse.repeated_effect.feature_effects[0]
    object.__setattr__(
        reverse_feature,
        "status",
        RepeatedMeasuresCR2Status.NOT_ESTIMABLE,
    )
    object.__setattr__(reverse_feature, "reason_code", "forged_feature_status")

    with pytest.raises(ContractError) as error:
        _require_response_pair_scope(pair_spec, forward, reverse)

    assert error.value.details.code == "directional_crossfit_parent_mismatch"
    assert error.value.details.field == "reverse_response.feature_effects"


def test_binding_revalidates_parent_and_identity_tampering() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        DirectionalCrossFitBinding()

    binding = _binding()
    object.__setattr__(binding, "forward_channel_name", "forged-channel")
    with pytest.raises(ContractError) as binding_error:
        binding.to_dict()
    assert binding_error.value.details.code == (
        "directional_crossfit_binding_integrity_violation"
    )

    parent_binding = _binding()
    object.__setattr__(parent_binding._forward_model, "contrast_name", "forged")
    with pytest.raises(ContractError) as parent_error:
        parent_binding.to_dict()
    assert parent_error.value.details.code == (
        "directional_crossfit_binding_integrity_violation"
    )
