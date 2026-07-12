from __future__ import annotations

import pandas as pd
import pytest
from scipy import sparse

from crychic.availability import (
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
    estimate_bundle_availability,
)
from crychic.core import ContractError
from crychic.pseudobulk import ExploratoryAggregate
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
)


def _interaction(interaction_id: str, ligand: str, receptor: str) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=interaction_id,
        ligand_name=ligand,
        receptor_name=receptor,
        ligand_subunits=(ligand,),
        receptor_subunits=(receptor,),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _bundle() -> ResourceBundle:
    return ResourceBundle(
        resource_id="filter_fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(
            _interaction("i1", "L1", "R1"),
            _interaction("i2", "L2", "R2"),
        ),
        mapping_report=MappingReport(2, 2, 4),
        manifest_digest="f" * 64,
        source_files=("fixture.tsv",),
        license="CC0",
        citation="Synthetic filter fixture",
    )


def _aggregate(*, subject: str, first_high: bool) -> ExploratoryAggregate:
    high, low = (10.0, 0.1)
    first, second = (high, low) if first_high else (low, high)
    unit_ids = (f"{subject}:sender", f"{subject}:receiver")
    metadata = pd.DataFrame(
        [
            {
                "unit_id": unit_ids[0],
                "sample_id": subject,
                "subject_id": subject,
                "cell_type": "Sender",
                "context": (("condition", "stim"),),
                "condition": "stim",
                "matrix_row": 0,
                "n_cells": 20,
                "cell_proportion": 0.5,
                "state_eligible": True,
                "abundance_eligible": True,
                "missingness_reason": "observed",
            },
            {
                "unit_id": unit_ids[1],
                "sample_id": subject,
                "subject_id": subject,
                "cell_type": "Receiver",
                "context": (("condition", "stim"),),
                "condition": "stim",
                "matrix_row": 1,
                "n_cells": 20,
                "cell_proportion": 0.5,
                "state_eligible": True,
                "abundance_eligible": True,
                "missingness_reason": "observed",
            },
        ]
    )
    return ExploratoryAggregate(
        mean_expression=sparse.csr_matrix(
            [
                [first, 0.0, second, 0.0],
                [0.0, first, 0.0, second],
            ]
        ),
        detection_fraction=None,
        unit_metadata=metadata,
        feature_ids=("L1", "R1", "L2", "R2"),
        matrix_unit_ids=unit_ids,
        source_location="X",
        expression_source="synthetic normalized expression",
        expression_transform="linear_normalized",
    )


def _manifest(*, interaction_ids: tuple[str, ...]) -> FrozenInteractionUniverse:
    bundle = _bundle()
    return FrozenInteractionUniverse(
        interaction_ids=interaction_ids,
        training_subject_ids=("train-2", "train-1"),
        resource_id=bundle.resource_id,
        resource_version=bundle.version,
        resource_manifest_digest=bundle.manifest_digest,
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )


def test_frozen_universe_prevents_test_signal_from_reselecting_top_k() -> None:
    bundle = _bundle()
    training = estimate_bundle_availability(
        _aggregate(subject="train", first_high=True),
        bundle,
        context_keys=("condition",),
        min_pooled_availability=0.0,
        max_interactions=1,
    )

    assert training.frozen_interaction_universe.interaction_ids == ("i1",)
    assert training.frozen_interaction_universe.training_subject_ids == ("train",)
    assert training.frozen_interaction_universe.selection_policy is (
        InteractionFilterPolicy.EXPLORATORY_POOLED_TOP_K_V1
    )
    assert training.filter_application is (
        InteractionFilterApplication.TRAINING_SELECTION_V1
    )

    reversed_test = _aggregate(subject="test", first_high=False)
    frozen = estimate_bundle_availability(
        reversed_test,
        bundle,
        context_keys=("condition",),
        min_pooled_availability=0.99,
        frozen_interaction_universe=training.frozen_interaction_universe,
    )
    independently_selected = estimate_bundle_availability(
        reversed_test,
        bundle,
        context_keys=("condition",),
        min_pooled_availability=0.0,
        max_interactions=1,
    )

    assert set(frozen.sample_interactions["interaction_id"].astype(str)) == {"i1"}
    assert set(
        independently_selected.sample_interactions["interaction_id"].astype(str)
    ) == {"i2"}
    assert frozen.filter_universe_id == training.filter_universe_id
    assert frozen.frozen_interaction_universe.training_subject_ids == ("train",)
    assert frozen.application_subject_ids == ("test",)
    assert frozen.filter_application is (
        InteractionFilterApplication.FROZEN_APPLICATION_V1
    )
    summary = frozen.mapping_summary.set_index("metric")["value"]
    assert summary["filter_candidate_interactions"] == 1
    assert summary["dropped_no_pooled_support_interactions"] == 0


def test_filter_universe_id_is_order_independent_and_versioned() -> None:
    first = _manifest(interaction_ids=("i2", "i1"))
    second = FrozenInteractionUniverse(
        interaction_ids=("i1", "i2"),
        training_subject_ids=("train-1", "train-2"),
        resource_id=first.resource_id,
        resource_version=first.resource_version,
        resource_manifest_digest=first.resource_manifest_digest,
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )

    assert first.filter_universe_id == second.filter_universe_id
    assert first.interaction_ids == ("i1", "i2")
    assert first.training_subject_ids == ("train-1", "train-2")
    assert first.to_dict()["schema_version"] == "1.0.0"


def test_duplicate_and_unknown_frozen_interactions_are_rejected() -> None:
    with pytest.raises(ContractError, match="interaction_ids must contain unique"):
        _manifest(interaction_ids=("i1", "i1"))

    unknown = _manifest(interaction_ids=("not_in_resource",))
    with pytest.raises(ValueError, match="unknown interaction IDs"):
        estimate_bundle_availability(
            _aggregate(subject="test", first_high=False),
            _bundle(),
            context_keys=("condition",),
            frozen_interaction_universe=unknown,
        )


def test_frozen_universe_rejects_data_driven_cap_and_resource_mismatch() -> None:
    universe = _manifest(interaction_ids=("i1",))
    aggregate = _aggregate(subject="test", first_high=False)

    with pytest.raises(ValueError, match="cannot be combined with max_interactions"):
        estimate_bundle_availability(
            aggregate,
            _bundle(),
            context_keys=("condition",),
            max_interactions=1,
            frozen_interaction_universe=universe,
        )

    base = _bundle()
    mismatched = ResourceBundle(
        resource_id=base.resource_id,
        version="2",
        species=base.species,
        gene_namespace=base.gene_namespace,
        interactions=base.interactions,
        mapping_report=base.mapping_report,
        manifest_digest=base.manifest_digest,
        source_files=base.source_files,
        license=base.license,
        citation=base.citation,
    )
    with pytest.raises(ValueError, match="resource provenance"):
        estimate_bundle_availability(
            aggregate,
            mismatched,
            context_keys=("condition",),
            frozen_interaction_universe=universe,
        )
