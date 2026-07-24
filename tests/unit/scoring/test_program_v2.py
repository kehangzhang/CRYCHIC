from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.scoring import (
    AbsoluteActivityV2Spec,
    SignedMechanismDirection,
    SignedProgramDefinitionV2,
    SignedProgramV2Spec,
    apply_absolute_activity_v2_transform,
    apply_signed_program_v2_to_sample_edges,
    build_signed_program_v2_training_table,
    fit_absolute_activity_v2_transform,
    fit_signed_program_v2,
    signed_program_definitions_from_resources_v2,
)

FEATURES = ("BG1", "BG2", "FILL", "GENERIC", "L", "R", "TN", "TP")
CELL_TYPES = ("receiver", "sender-a", "sender-b")
INTERACTIONS = ("act", "att", "bi", "unknown")


def _interaction(interaction_id: str) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=interaction_id,
        ligand_name="L",
        receptor_name="R",
        ligand_subunits=("L",),
        receptor_subunits=("R",),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="ligand_to_receptor",
        source="fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _resource() -> ResourceBundle:
    return ResourceBundle(
        resource_id="program-fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=tuple(_interaction(value) for value in INTERACTIONS),
        mapping_report=MappingReport(
            source_rows=len(INTERACTIONS),
            loaded_rows=len(INTERACTIONS),
            mapped_entities=len(INTERACTIONS),
        ),
        manifest_digest="program-fixture-digest",
        source_files=("fixture.tsv",),
        license="CC0",
        citation="Synthetic fixture",
    )


def _universe(training_subjects: tuple[str, ...]) -> FrozenInteractionUniverse:
    return FrozenInteractionUniverse(
        interaction_ids=INTERACTIONS,
        training_subject_ids=training_subjects,
        resource_id="program-fixture",
        resource_version="1",
        resource_manifest_digest="program-fixture-digest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )


def _aggregate(
    subjects: tuple[str, ...],
    *,
    index_offset: int = 0,
    target_deltas: tuple[float, ...] | None = None,
    batch_override: str | None = None,
    condition: str = "training-label",
) -> PseudobulkDataset:
    rows: list[dict[str, object]] = []
    matrix: list[list[int]] = []
    unit_ids: list[str] = []
    deltas = target_deltas or tuple(0.0 for _ in subjects)
    for local_index, (subject, target_delta) in enumerate(
        zip(subjects, deltas, strict=True)
    ):
        index = local_index + index_offset
        sample_id = f"sample-{subject}"
        batch = batch_override or ("A" if index % 2 == 0 else "B")
        generic = 25.0 + 4.0 * ((index * 3) % 7)
        receiver_counts = {
            "BG1": 80.0 + 3.0 * ((index * 5) % 9),
            "BG2": 55.0 + 2.0 * ((index * 7) % 11),
            "FILL": 800.0 + 13.0 * index,
            "GENERIC": generic,
            "L": 4.0,
            "R": 90.0 + 2.0 * (index % 3),
            "TN": 75.0 - 3.0 * np.sin(index * 0.8),
            "TP": 45.0 + 1.8 * generic + 5.0 * (batch == "B") + target_delta,
        }
        for cell_type in CELL_TYPES:
            unit_id = f"unit:{sample_id}:{cell_type}"
            unit_ids.append(unit_id)
            if cell_type == "receiver":
                counts = receiver_counts
            else:
                sender_multiplier = 1.5 if cell_type == "sender-a" else 0.8
                counts = {
                    "BG1": 60.0 + index,
                    "BG2": 40.0 + 2.0 * index,
                    "FILL": 900.0 + 7.0 * index,
                    "GENERIC": 20.0,
                    "L": 70.0 * sender_multiplier + index,
                    "R": 2.0,
                    "TN": 10.0,
                    "TP": 10.0,
                }
            matrix.append([max(1, round(counts[name])) for name in FEATURES])
            rows.append(
                {
                    "unit_id": unit_id,
                    "sample_id": sample_id,
                    "subject_id": subject,
                    "cell_type": cell_type,
                    "context": (("condition", condition),),
                    "condition": condition,
                    "batch": batch,
                    "matrix_row": len(matrix) - 1,
                    "n_cells": 80 + ((index * 11 + len(cell_type)) % 37),
                    "cell_proportion": 1.0 / len(CELL_TYPES),
                    "state_eligible": True,
                    "abundance_eligible": True,
                    "missingness_reason": "observed",
                }
            )
    values = np.asarray(matrix, dtype=np.int64)
    return PseudobulkDataset(
        counts=sparse.csr_matrix(values),
        detection_fraction=sparse.csr_matrix(values > 0, dtype=float),
        unit_metadata=pd.DataFrame(rows),
        feature_ids=FEATURES,
        matrix_unit_ids=tuple(unit_ids),
        source_location="X",
    )


def _availability(
    aggregate: PseudobulkDataset,
    universe: FrozenInteractionUniverse,
) -> BatchAvailability:
    rows = [
        {
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "context_id": f"context-{sample.condition}",
            "condition": sample.condition,
            "sender": sender,
            "receiver": "receiver",
            "interaction_id": interaction_id,
            "availability_state": 0.5,
        }
        for sample in aggregate.unit_metadata.drop_duplicates("sample_id").itertuples(
            index=False
        )
        for interaction_id in INTERACTIONS
        for sender in ("sender-a", "sender-b")
    ]
    return BatchAvailability(
        sample_interactions=pd.DataFrame(rows),
        mapping_summary=pd.DataFrame(),
        resource_id="program-fixture",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=universe,
        filter_application=InteractionFilterApplication.FROZEN_APPLICATION_V1,
        application_subject_ids=tuple(
            sorted(aggregate.unit_metadata["subject_id"].astype(str).unique())
        ),
    )


def _definitions() -> tuple[SignedProgramDefinitionV2, ...]:
    return (
        SignedProgramDefinitionV2(
            interaction_id="act",
            positive_targets=(("TP", 1.0),),
            mechanism_direction="activation",
        ),
        SignedProgramDefinitionV2(
            interaction_id="att",
            positive_targets=(("TP", 1.0),),
            mechanism_direction="attenuation",
        ),
        SignedProgramDefinitionV2(
            interaction_id="bi",
            positive_targets=(("TP", 1.0),),
            negative_targets=(("TN", 1.0),),
            mechanism_direction="activation",
        ),
        SignedProgramDefinitionV2(
            interaction_id="unknown",
            positive_targets=(("TP", 1.0),),
            mechanism_direction="unknown",
        ),
    )


def _fit_program():
    subjects = tuple(f"train-{index}" for index in range(12))
    aggregate = _aggregate(subjects)
    universe = _universe(subjects)
    activity = fit_absolute_activity_v2_transform(
        aggregate,
        _resource(),
        universe,
        fold_id="fold-1",
        training_input_digest="training-input-digest",
        spec=AbsoluteActivityV2Spec(),
        additional_feature_ids=FEATURES,
    )
    functional = fit_signed_program_v2(
        activity,
        aggregate,
        _definitions(),
        training_input_digest="training-input-digest",
        spec=SignedProgramV2Spec(
            categorical_covariates=("batch",),
            generic_state_feature_ids=("GENERIC",),
            minimum_training_samples=8,
            minimum_training_subjects=6,
        ),
    )
    return aggregate, universe, activity, functional


def _heldout_scores(activity, universe, aggregate):
    return apply_absolute_activity_v2_transform(
        activity,
        aggregate,
        _availability(aggregate, universe),
        condition_columns=("condition",),
        repeat_id="repeat-1",
        application_input_digest="heldout-input-digest",
        config_digest="config-digest",
        seed_lineage_id="seed-lineage",
        package_version="0.0.test",
    )


def test_signed_program_fit_is_condition_label_independent() -> None:
    aggregate, _, activity, _ = _fit_program()
    relabeled = replace(
        aggregate,
        unit_metadata=aggregate.unit_metadata.assign(
            condition="different-label",
            context=[(("condition", "different-label"),)]
            * len(aggregate.unit_metadata),
        ),
    )
    spec = SignedProgramV2Spec(
        categorical_covariates=("batch",),
        generic_state_feature_ids=("GENERIC",),
        minimum_training_samples=8,
        minimum_training_subjects=6,
    )
    first = fit_signed_program_v2(
        activity,
        aggregate,
        _definitions(),
        training_input_digest="training-input-digest",
        spec=spec,
    )
    second = fit_signed_program_v2(
        activity,
        relabeled,
        _definitions(),
        training_input_digest="training-input-digest",
        spec=spec,
    )

    assert first.functional_id == second.functional_id


def test_signed_program_drops_invariant_numeric_nuisance() -> None:
    subjects = tuple(f"train-{index}" for index in range(12))
    aggregate = _aggregate(subjects)
    metadata = aggregate.unit_metadata.assign(n_cells=100)
    aggregate = replace(aggregate, unit_metadata=metadata)
    universe = _universe(subjects)
    activity = fit_absolute_activity_v2_transform(
        aggregate,
        _resource(),
        universe,
        fold_id="fold-1",
        training_input_digest="training-input-digest",
        spec=AbsoluteActivityV2Spec(),
        additional_feature_ids=FEATURES,
    )
    functional = fit_signed_program_v2(
        activity,
        aggregate,
        _definitions(),
        training_input_digest="training-input-digest",
        spec=SignedProgramV2Spec(
            categorical_covariates=("batch",),
            generic_state_feature_ids=("GENERIC",),
            minimum_training_samples=8,
            minimum_training_subjects=6,
        ),
    )

    receiver = next(
        item for item in functional.receiver_transforms if item.receiver == "receiver"
    )
    assert receiver.status == "observed"
    assert "log_cell_count" not in {name for name, _, _ in receiver.numeric_encodings}


def test_program_output_preserves_negative_values_and_direction_semantics() -> None:
    _, universe, activity, functional = _fit_program()
    heldout = _aggregate(
        ("heldout-low", "heldout-high"),
        index_offset=20,
        target_deltas=(-80.0, 120.0),
    )
    scores = _heldout_scores(activity, universe, heldout)

    result = apply_signed_program_v2_to_sample_edges(
        functional, activity, heldout, scores
    ).table
    parent = result.drop_duplicates(["sample_id", "receiver", "interaction_id"])
    pivot = parent.pivot(
        index="sample_id", columns="interaction_id", values="program_signed"
    )

    assert pivot.loc["sample-heldout-low", "act"] < 0.0
    assert pivot.loc["sample-heldout-high", "act"] > 0.0
    assert pivot.loc["sample-heldout-high", "act"] > 1.0
    assert pivot.loc["sample-heldout-high", "att"] == pytest.approx(
        -pivot.loc["sample-heldout-high", "act"]
    )
    unknown = parent.loc[parent["interaction_id"].eq("unknown")].iloc[0]
    assert pd.notna(unknown["program_unaligned_raw"])
    assert pd.isna(unknown["program_signed"])
    assert unknown["program_direction"] == "unknown"
    assert unknown["program_status"] == "not_estimable"
    assert set(result["program_functional_id"]) == {functional.functional_id}


def test_training_program_residual_is_orthogonal_to_declared_nuisance() -> None:
    aggregate, _, activity, functional = _fit_program()
    table = build_signed_program_v2_training_table(functional, activity, aggregate)
    activation = table.loc[
        table["receiver"].eq("receiver")
        & table["interaction_id"].eq("act")
        & table["program_status"].eq("observed")
    ]
    sample_metadata = aggregate.unit_metadata.drop_duplicates("sample_id").loc[
        :, ["sample_id", "batch"]
    ]
    merged = activation.merge(sample_metadata, on="sample_id", validate="one_to_one")

    batch_indicator = merged["batch"].eq("B").to_numpy(dtype=float)
    correlation = np.corrcoef(
        merged["program_unaligned_raw"].to_numpy(dtype=float), batch_indicator
    )[0, 1]
    assert abs(correlation) < 1.0e-10


def test_unseen_heldout_batch_is_not_silently_mapped_to_reference() -> None:
    _, universe, activity, functional = _fit_program()
    heldout = _aggregate(
        ("heldout-1",),
        index_offset=30,
        target_deltas=(30.0,),
        batch_override="unseen-site",
    )
    scores = _heldout_scores(activity, universe, heldout)

    result = apply_signed_program_v2_to_sample_edges(
        functional, activity, heldout, scores
    ).table

    assert set(result["program_status"]) == {"not_estimable"}
    assert set(result["program_reason_code"]) == {
        "heldout_program_nuisance_not_estimable"
    }
    assert result["program_signed"].isna().all()


def test_generic_state_cannot_include_mechanism_target() -> None:
    subjects = tuple(f"train-{index}" for index in range(12))
    aggregate = _aggregate(subjects)
    universe = _universe(subjects)
    activity = fit_absolute_activity_v2_transform(
        aggregate,
        _resource(),
        universe,
        fold_id="fold-1",
        training_input_digest="training-input-digest",
        spec=AbsoluteActivityV2Spec(),
        additional_feature_ids=FEATURES,
    )

    with pytest.raises(ValueError, match="cannot overlap mechanism targets"):
        fit_signed_program_v2(
            activity,
            aggregate,
            _definitions(),
            training_input_digest="training-input-digest",
            spec=SignedProgramV2Spec(generic_state_feature_ids=("TP",)),
        )


def test_resource_mapping_freezes_target_direction_without_expression() -> None:
    prior = TargetPrior(
        resource_id="prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="interaction",
        target_ids=("TN", "TP"),
        driver_ids=("act",),
        indptr=(0, 2),
        target_indices=(0, 1),
        weights=(0.0, 2.0),
        ranks=(2, 1),
        direction=1,
        evidence="fixture",
        mapping_report=MappingReport(source_rows=2, loaded_rows=2, mapped_entities=2),
        manifest_digest="prior-digest",
    )

    definitions = signed_program_definitions_from_resources_v2(_resource(), prior)

    assert len(definitions) == 1
    assert definitions[0].interaction_id == "act"
    assert definitions[0].positive_targets == (("TP", 1.0),)
    assert definitions[0].mechanism_direction is SignedMechanismDirection.ACTIVATION
