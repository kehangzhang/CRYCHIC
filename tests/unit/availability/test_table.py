from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import crychic.availability.table as table_module
from crychic.availability import (
    AvailabilityParameters,
    BatchAvailability,
    DetectionShrinkage,
    HillParameters,
    estimate_bundle_availability,
)
from crychic.availability.table import _entity_values
from crychic.pseudobulk import ExploratoryAggregate, PseudobulkDataset
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
)


def _aggregate() -> PseudobulkDataset:
    matrix_unit_ids = ("u0", "u1", "u2", "u3", "u4", "u5", "u6")
    rows = [
        [10, 0],
        [0, 10],
        [20, 0],
        [0, 20],
        [10, 0],
        [0, 10],
        [20, 0],
    ]
    metadata_rows: list[dict[str, object]] = []
    row_index = 0
    for subject in ("S1", "S2"):
        for context in ("ctrl", "stim"):
            for cell_type in ("Sender", "Receiver"):
                missing = (
                    subject == "S2" and context == "stim" and cell_type == "Receiver"
                )
                metadata_rows.append(
                    {
                        "unit_id": f"missing-{subject}-{context}-{cell_type}"
                        if missing
                        else matrix_unit_ids[row_index],
                        "sample_id": f"{subject}:{context}",
                        "subject_id": subject,
                        "cell_type": cell_type,
                        "context": (("condition", context),),
                        "condition": context,
                        "matrix_row": pd.NA if missing else row_index,
                        "n_cells": 0 if missing else 10,
                        "cell_proportion": 0.0 if missing else 0.5,
                        "state_eligible": not missing,
                        "abundance_eligible": True,
                        "missingness_reason": "sampling_zero"
                        if missing
                        else "observed",
                    }
                )
                if not missing:
                    row_index += 1
    return PseudobulkDataset(
        counts=sparse.csr_matrix(rows),
        detection_fraction=sparse.csr_matrix(
            [[1 if value else 0 for value in row] for row in rows], dtype=float
        ),
        unit_metadata=pd.DataFrame(metadata_rows),
        feature_ids=("L", "R"),
        matrix_unit_ids=matrix_unit_ids,
        source_location="layers[counts]",
    )


def _bundle() -> ResourceBundle:
    interaction = Interaction(
        interaction_id="i1",
        source_interaction_id="L_R",
        ligand_name="L",
        receptor_name="R",
        ligand_subunits=("L",),
        receptor_subunits=("R",),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        pathway="test",
    )
    return ResourceBundle(
        resource_id="fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(interaction,),
        mapping_report=MappingReport(source_rows=1, loaded_rows=1, mapped_entities=2),
        manifest_digest="digest",
        source_files=("fixture.csv",),
        license="CC0",
        citation="Synthetic fixture",
    )


def _copy_availability(
    source: BatchAvailability,
    *,
    sample_interactions: pd.DataFrame | None = None,
    application_subject_ids: tuple[object, ...] | None = None,
) -> BatchAvailability:
    return BatchAvailability(
        sample_interactions=(
            source.sample_interactions
            if sample_interactions is None
            else sample_interactions
        ),
        mapping_summary=source.mapping_summary,
        resource_id=source.resource_id,
        resource_version=source.resource_version,
        detection_available=source.detection_available,
        frozen_interaction_universe=source.frozen_interaction_universe,
        filter_application=source.filter_application,
        application_subject_ids=(
            source.application_subject_ids
            if application_subject_ids is None
            else application_subject_ids
        ),
    )


@pytest.mark.parametrize("column", ["sample_id", "subject_id", "context_id"])
@pytest.mark.parametrize("invalid", [pd.NA, np.nan, None])
def test_batch_availability_rejects_missing_row_identifiers(
    column: str, invalid: object
) -> None:
    source = estimate_bundle_availability(
        _aggregate(),
        _bundle(),
        context_keys=("condition",),
        min_pooled_availability=0.0,
    )
    rows = source.sample_interactions.copy(deep=True)
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = invalid

    with pytest.raises(
        ValueError,
        match=rf"sample_interactions\.{column} must not contain missing identifiers",
    ):
        _copy_availability(source, sample_interactions=rows)


@pytest.mark.parametrize("column", ["sample_id", "subject_id", "context_id"])
@pytest.mark.parametrize("invalid", ["", "   "])
def test_batch_availability_rejects_empty_row_identifiers(
    column: str, invalid: str
) -> None:
    source = estimate_bundle_availability(
        _aggregate(),
        _bundle(),
        context_keys=("condition",),
        min_pooled_availability=0.0,
    )
    rows = source.sample_interactions.copy(deep=True)
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = invalid

    with pytest.raises(
        ValueError,
        match=rf"sample_interactions\.{column} must contain non-empty identifiers",
    ):
        _copy_availability(source, sample_interactions=rows)


@pytest.mark.parametrize("invalid", [pd.NA, np.nan, None])
def test_batch_availability_rejects_missing_application_subject_ids(
    invalid: object,
) -> None:
    source = estimate_bundle_availability(
        _aggregate(),
        _bundle(),
        context_keys=("condition",),
        min_pooled_availability=0.0,
    )

    with pytest.raises(
        ValueError,
        match="application_subject_ids must not contain missing identifiers",
    ):
        _copy_availability(source, application_subject_ids=(invalid,))


@pytest.mark.parametrize("invalid", ["", "   "])
def test_batch_availability_rejects_empty_application_subject_ids(
    invalid: str,
) -> None:
    source = estimate_bundle_availability(
        _aggregate(),
        _bundle(),
        context_keys=("condition",),
        min_pooled_availability=0.0,
    )

    with pytest.raises(
        ValueError,
        match="application_subject_ids must contain non-empty identifiers",
    ):
        _copy_availability(source, application_subject_ids=(invalid,))


@pytest.mark.parametrize(
    ("column", "invalid"),
    [
        ("sample_id", pd.NA),
        ("sample_id", "  "),
        ("subject_id", np.nan),
        ("subject_id", ""),
    ],
)
def test_estimation_rejects_invalid_aggregate_identifiers_before_stringification(
    column: str, invalid: object
) -> None:
    source = _aggregate()
    metadata = source.unit_metadata.copy(deep=True)
    metadata[column] = metadata[column].astype(object)
    metadata.loc[0, column] = invalid
    aggregate = PseudobulkDataset(
        counts=source.counts,
        detection_fraction=source.detection_fraction,
        unit_metadata=metadata,
        feature_ids=source.feature_ids,
        matrix_unit_ids=source.matrix_unit_ids,
        source_location=source.source_location,
    )

    with pytest.raises(
        ValueError,
        match=rf"aggregate\.unit_metadata\.{column} must",
    ):
        estimate_bundle_availability(
            aggregate,
            _bundle(),
            context_keys=("condition",),
            min_pooled_availability=0.0,
        )


def test_batch_availability_preserves_state_and_ecosystem_components() -> None:
    result = estimate_bundle_availability(
        _aggregate(),
        _bundle(),
        context_keys=("condition",),
        parameters=AvailabilityParameters(
            hill=HillParameters(half_saturation=1.0),
            detection=DetectionShrinkage(alpha=1.0, beta=1.0, exponent=0.0),
        ),
        min_pooled_availability=0.0,
    )

    table = result.sample_interactions
    assert len(table) == 13
    active = table.loc[
        (table["sender"] == "Sender") & (table["receiver"] == "Receiver")
    ]
    assert len(active) == 3
    assert (active["availability_state"] > 0).all()
    assert (active["availability_ecosystem"] < active["availability_state"]).all()
    assert {
        "ligand_absolute_evidence",
        "receptor_absolute_evidence",
        "absolute_lr_activity",
    }.issubset(table.columns)
    assert np.allclose(
        active["absolute_lr_activity"],
        0.5
        * (active["ligand_absolute_evidence"] + active["receptor_absolute_evidence"]),
    )
    assert active["absolute_lr_activity"].between(0.0, 1.0).all()
    assert set(table["state_status"].astype(str)) == {"observed"}
    assert table["state_reason_code"].isna().all()
    assert set(table["ecosystem_status"].astype(str)) == {"observed"}
    assert table["ecosystem_reason_code"].isna().all()
    missing_sample = table.loc[table["sample_id"] == "S2:stim"]
    assert set(missing_sample["sender"].astype(str)) == {"Sender"}
    assert set(missing_sample["receiver"].astype(str)) == {"Sender"}
    assert (
        result.mapping_summary.set_index("metric").loc[
            "pooled_supported_interactions", "value"
        ]
        == 1
    )


def test_unmapped_interaction_is_reported_and_not_scored() -> None:
    base = _bundle()
    unmapped = Interaction(
        interaction_id="i2",
        source_interaction_id="X_R",
        ligand_name="X",
        receptor_name="R",
        ligand_subunits=("X",),
        receptor_subunits=("R",),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    bundle = ResourceBundle(
        resource_id=base.resource_id,
        version=base.version,
        species=base.species,
        gene_namespace=base.gene_namespace,
        interactions=(*base.interactions, unmapped),
        mapping_report=MappingReport(source_rows=2, loaded_rows=2, mapped_entities=3),
        manifest_digest=base.manifest_digest,
        source_files=base.source_files,
        license=base.license,
        citation=base.citation,
    )

    result = estimate_bundle_availability(
        _aggregate(), bundle, context_keys=("condition",), min_pooled_availability=0.0
    )
    summary = result.mapping_summary.set_index("metric")["value"]
    assert summary["source_interactions"] == 2
    assert summary["dropped_unmapped_interactions"] == 1
    assert set(result.sample_interactions["interaction_id"].astype(str)) == {"i1"}


def test_context_key_named_context_is_recovered_from_canonical_tuple() -> None:
    aggregate = _aggregate()
    metadata = aggregate.unit_metadata.drop(columns="condition").copy()
    metadata["context"] = metadata["context"].map(
        lambda value: (("context", value[0][1]),)
    )
    renamed = PseudobulkDataset(
        counts=aggregate.counts,
        detection_fraction=aggregate.detection_fraction,
        unit_metadata=metadata,
        feature_ids=aggregate.feature_ids,
        matrix_unit_ids=aggregate.matrix_unit_ids,
        source_location=aggregate.source_location,
    )

    result = estimate_bundle_availability(
        renamed, _bundle(), context_keys=("context",), min_pooled_availability=0.0
    )

    assert set(result.sample_interactions["context"]) == {"ctrl", "stim"}


def test_normalized_without_detection_is_invariant_to_cell_count() -> None:
    matrix_unit_ids = ("s1-sender", "s1-receiver", "s2-sender", "s2-receiver")
    metadata = pd.DataFrame(
        [
            {
                "unit_id": unit_id,
                "sample_id": sample,
                "subject_id": sample,
                "cell_type": cell_type,
                "context": (("condition", condition),),
                "condition": condition,
                "matrix_row": index,
                "n_cells": n_cells,
                "cell_proportion": 0.5,
                "state_eligible": True,
                "abundance_eligible": True,
                "missingness_reason": "observed",
            }
            for index, (unit_id, sample, condition, cell_type, n_cells) in enumerate(
                (
                    ("s1-sender", "s1", "ctrl", "Sender", 2),
                    ("s1-receiver", "s1", "ctrl", "Receiver", 2),
                    ("s2-sender", "s2", "stim", "Sender", 20),
                    ("s2-receiver", "s2", "stim", "Receiver", 20),
                )
            )
        ]
    )
    aggregate = ExploratoryAggregate(
        mean_expression=sparse.csr_matrix(
            [[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]]
        ),
        detection_fraction=None,
        unit_metadata=metadata,
        feature_ids=("L", "R"),
        matrix_unit_ids=matrix_unit_ids,
        source_location="X",
        expression_source="published normalized matrix",
        expression_transform="log1p_normalized",
    )

    result = estimate_bundle_availability(
        aggregate,
        _bundle(),
        context_keys=("condition",),
        min_pooled_availability=0.0,
    )
    selected = result.sample_interactions.loc[
        (result.sample_interactions["sender"] == "Sender")
        & (result.sample_interactions["receiver"] == "Receiver")
    ]

    assert not result.detection_available
    assert selected["availability_state"].nunique() == 1
    assert selected["availability_ecosystem"].nunique() == 1


def test_complex_values_preserve_exact_zero_and_missingness() -> None:
    values = np.asarray(
        [
            [0.0, 0.8],
            [np.nan, 0.8],
            [0.0, np.nan],
            [0.4, 0.8],
        ]
    )

    result = _entity_values(values, (0, 1), power=4.0, epsilon=1e-12)

    assert result[0] == 0.0
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert 0.4 < result[3] < 0.8


def test_bundle_availability_reuses_identical_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _bundle()
    first = base.interactions[0]
    second = Interaction(
        interaction_id="i2",
        source_interaction_id="L_R_second",
        ligand_name=first.ligand_name,
        receptor_name=first.receptor_name,
        ligand_subunits=first.ligand_subunits,
        receptor_subunits=first.receptor_subunits,
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction=first.direction,
        source=first.source,
        version=first.version,
        species=first.species,
        gene_namespace=first.gene_namespace,
        pathway=first.pathway,
    )
    bundle = ResourceBundle(
        resource_id=base.resource_id,
        version=base.version,
        species=base.species,
        gene_namespace=base.gene_namespace,
        interactions=(first, second),
        mapping_report=MappingReport(2, 2, 2),
        manifest_digest=base.manifest_digest,
        source_files=base.source_files,
        license=base.license,
        citation=base.citation,
    )
    original = table_module._entity_values
    calls: list[tuple[int, ...]] = []

    def counted_entity_values(
        gene_values: np.ndarray,
        indices: tuple[int, ...] | list[int],
        *,
        power: float,
        epsilon: float,
    ) -> np.ndarray:
        calls.append(tuple(indices))
        return original(gene_values, indices, power=power, epsilon=epsilon)

    monkeypatch.setattr(table_module, "_entity_values", counted_entity_values)

    result = estimate_bundle_availability(
        _aggregate(),
        bundle,
        context_keys=("condition",),
        min_pooled_availability=0.0,
    )

    assert len(result.frozen_interaction_universe.interaction_ids) == 2
    assert len(calls) == 2


def test_state_score_does_not_require_abundance_eligibility() -> None:
    aggregate = _aggregate()
    metadata = aggregate.unit_metadata.copy()
    unavailable = (metadata["sample_id"] == "S1:ctrl") & (
        metadata["cell_type"] == "Sender"
    )
    metadata.loc[unavailable, "abundance_eligible"] = False
    modified = PseudobulkDataset(
        counts=aggregate.counts,
        detection_fraction=aggregate.detection_fraction,
        unit_metadata=metadata,
        feature_ids=aggregate.feature_ids,
        matrix_unit_ids=aggregate.matrix_unit_ids,
        source_location=aggregate.source_location,
    )

    result = estimate_bundle_availability(
        modified,
        _bundle(),
        context_keys=("condition",),
        parameters=AvailabilityParameters(
            hill=HillParameters(half_saturation=1.0),
            detection=DetectionShrinkage(alpha=1.0, beta=1.0, exponent=0.0),
        ),
        min_pooled_availability=0.0,
    )

    sample = result.sample_interactions.loc[
        result.sample_interactions["sample_id"] == "S1:ctrl"
    ]
    assert len(sample) == 4
    assert sample["availability_state"].notna().all()
    assert set(sample["state_status"].astype(str)) == {"observed"}
    assert sample["state_reason_code"].isna().all()

    sender_to_receiver = sample.loc[
        (sample["sender"] == "Sender") & (sample["receiver"] == "Receiver")
    ].iloc[0]
    assert pd.isna(sender_to_receiver["availability_ecosystem"])
    assert sender_to_receiver["ecosystem_status"] == "abundance_not_estimable"
    assert (
        sender_to_receiver["ecosystem_reason_code"] == "sender_abundance_not_eligible"
    )

    receiver_to_sender = sample.loc[
        (sample["sender"] == "Receiver") & (sample["receiver"] == "Sender")
    ].iloc[0]
    assert pd.isna(receiver_to_sender["availability_ecosystem"])
    assert (
        receiver_to_sender["ecosystem_reason_code"] == "receiver_abundance_not_eligible"
    )

    receiver_to_receiver = sample.loc[
        (sample["sender"] == "Receiver") & (sample["receiver"] == "Receiver")
    ].iloc[0]
    assert receiver_to_receiver["ecosystem_status"] == "observed"
    assert pd.isna(receiver_to_receiver["ecosystem_reason_code"])
    assert pd.notna(receiver_to_receiver["availability_ecosystem"])
