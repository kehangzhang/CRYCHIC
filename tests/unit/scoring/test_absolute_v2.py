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
)
from crychic.scoring import (
    ABSOLUTE_ACTIVITY_V2_SCORE_VERSION,
    AbsoluteActivityV2Spec,
    apply_absolute_activity_v2_transform,
    fit_absolute_activity_v2_transform,
)

FEATURES = ("FILL", "L1", "L2", "R")
CELL_TYPES = ("receiver", "sender-a", "sender-b", "sender-c")


def _interaction(
    interaction_id: str,
    ligand_subunits: tuple[str, ...],
    *,
    ligand_is_complex: bool,
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=interaction_id,
        ligand_name="+".join(ligand_subunits),
        receptor_name="R",
        ligand_subunits=ligand_subunits,
        receptor_subunits=("R",),
        ligand_is_complex=ligand_is_complex,
        receptor_is_complex=False,
        direction="ligand_to_receptor",
        source="fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _resource() -> ResourceBundle:
    interactions = (
        _interaction("lr-simple", ("L1",), ligand_is_complex=False),
        _interaction("lr-and", ("L1", "L2"), ligand_is_complex=True),
        _interaction("lr-or", ("L1", "L2"), ligand_is_complex=False),
    )
    return ResourceBundle(
        resource_id="fixture-resource",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=interactions,
        mapping_report=MappingReport(
            source_rows=3,
            loaded_rows=3,
            mapped_entities=3,
        ),
        manifest_digest="fixture-resource-digest",
        source_files=("fixture.tsv",),
        license="CC0",
        citation="Synthetic fixture",
    )


def _universe() -> FrozenInteractionUniverse:
    return FrozenInteractionUniverse(
        interaction_ids=("lr-simple", "lr-and", "lr-or"),
        training_subject_ids=("train-1", "train-2"),
        resource_id="fixture-resource",
        resource_version="1",
        resource_manifest_digest="fixture-resource-digest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )


def _cell_counts(cell_type: str, multiplier: int) -> list[int]:
    by_type = {
        "receiver": [898, 1, 1, 100],
        "sender-a": [899, 100, 0, 1],
        "sender-b": [989, 10, 0, 1],
        "sender-c": [994, 5, 0, 1],
    }
    return [value * multiplier for value in by_type[cell_type]]


def _aggregate(
    subjects: tuple[str, ...],
    *,
    count_multiplier: int = 1,
    condition: str = "treated",
) -> PseudobulkDataset:
    rows: list[dict[str, object]] = []
    matrix_rows: list[list[int]] = []
    matrix_unit_ids: list[str] = []
    for subject_index, subject in enumerate(subjects):
        sample = f"sample-{subject}"
        per_subject_multiplier = count_multiplier * (subject_index + 1)
        for cell_type in CELL_TYPES:
            unit_id = f"unit:{sample}:{cell_type}"
            matrix_row = len(matrix_rows)
            matrix_unit_ids.append(unit_id)
            matrix_rows.append(_cell_counts(cell_type, per_subject_multiplier))
            rows.append(
                {
                    "unit_id": unit_id,
                    "sample_id": sample,
                    "subject_id": subject,
                    "cell_type": cell_type,
                    "context": (("condition", condition),),
                    "condition": condition,
                    "matrix_row": matrix_row,
                    "n_cells": 100 * per_subject_multiplier,
                    "cell_proportion": 0.25,
                    "state_eligible": True,
                    "abundance_eligible": True,
                    "missingness_reason": "observed",
                }
            )
    return PseudobulkDataset(
        counts=sparse.csr_matrix(np.asarray(matrix_rows, dtype=np.int64)),
        detection_fraction=sparse.csr_matrix(
            np.asarray(matrix_rows, dtype=float) > 0.0, dtype=float
        ),
        unit_metadata=pd.DataFrame(rows),
        feature_ids=FEATURES,
        matrix_unit_ids=tuple(matrix_unit_ids),
        source_location="X",
    )


def _availability(
    aggregate: PseudobulkDataset,
    *,
    include_sender_c: bool = True,
) -> BatchAvailability:
    senders = ("sender-a", "sender-b", "sender-c") if include_sender_c else (
        "sender-a",
        "sender-b",
    )
    rows: list[dict[str, object]] = []
    sample_metadata = aggregate.unit_metadata.drop_duplicates("sample_id")
    for sample in sample_metadata.itertuples(index=False):
        for interaction_id in ("lr-simple", "lr-and", "lr-or"):
            for sender in senders:
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "subject_id": sample.subject_id,
                        "context_id": f"context-{sample.condition}",
                        "condition": sample.condition,
                        "sender": sender,
                        "receiver": "receiver",
                        "interaction_id": interaction_id,
                        "availability_state": (
                            0.0
                            if sender == "sender-a" and interaction_id == "lr-simple"
                            else 0.5
                        ),
                    }
                )
    subjects = tuple(sorted(set(aggregate.unit_metadata["subject_id"].astype(str))))
    return BatchAvailability(
        sample_interactions=pd.DataFrame(rows),
        mapping_summary=pd.DataFrame(),
        resource_id="fixture-resource",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=_universe(),
        filter_application=InteractionFilterApplication.FROZEN_APPLICATION_V1,
        application_subject_ids=subjects,
    )


def _fit(spec: AbsoluteActivityV2Spec | None = None):
    return fit_absolute_activity_v2_transform(
        _aggregate(("train-1", "train-2"), condition="training-label"),
        _resource(),
        _universe(),
        fold_id="fold-1",
        training_input_digest="training-input-digest",
        spec=spec,
    )


def _apply(transform, aggregate, availability):
    return apply_absolute_activity_v2_transform(
        transform,
        aggregate,
        availability,
        condition_columns=("condition",),
        repeat_id="repeat-1",
        application_input_digest="application-input-digest",
        config_digest="config-digest",
        seed_lineage_id="seed-lineage",
        package_version="0.0.test",
    )


def test_fit_is_fold_frozen_and_does_not_consume_condition_labels() -> None:
    first = _aggregate(("train-1", "train-2"), condition="a")
    relabeled = replace(
        first,
        unit_metadata=first.unit_metadata.assign(condition="b"),
    )

    fit_a = fit_absolute_activity_v2_transform(
        first,
        _resource(),
        _universe(),
        fold_id="fold-1",
        training_input_digest="same-input-digest",
    )
    fit_b = fit_absolute_activity_v2_transform(
        relabeled,
        _resource(),
        _universe(),
        fold_id="fold-1",
        training_input_digest="same-input-digest",
    )

    assert fit_a.transform_manifest_id == fit_b.transform_manifest_id
    assert np.array_equal(fit_a.feature_medians, fit_b.feature_medians)
    assert np.array_equal(fit_a.feature_mads, fit_b.feature_mads)
    assert not fit_a.feature_medians.flags.writeable
    assert (fit_a.feature_mads >= fit_a.spec.robust_scale_floor).all()


def test_m0_v2_is_library_scale_invariant_and_not_mechanism_gated() -> None:
    transform = _fit()
    heldout = _aggregate(("heldout-1",), count_multiplier=1)
    doubled = _aggregate(("heldout-1",), count_multiplier=2)

    base = _apply(transform, heldout, _availability(heldout)).table
    scaled = _apply(transform, doubled, _availability(doubled)).table

    assert set(base["score_version"]) == {ABSOLUTE_ACTIVITY_V2_SCORE_VERSION}
    assert np.allclose(base["sender_detection_raw"], scaled["sender_detection_raw"])
    ungated = base.loc[
        base["sender"].eq("sender-a")
        & base["interaction_id"].eq("lr-simple")
    ].iloc[0]
    assert ungated["mechanism_support"] == 0.0
    assert ungated["sender_detection_raw"] > 0.0
    assert ungated["status"] == "observed"


def test_log_geometric_formula_and_frozen_bottleneck_penalty() -> None:
    heldout = _aggregate(("heldout-1",))
    availability = _availability(heldout)
    plain = _apply(
        _fit(AbsoluteActivityV2Spec(bottleneck_penalty=0.0)),
        heldout,
        availability,
    )
    bottleneck = _apply(
        _fit(AbsoluteActivityV2Spec(bottleneck_penalty=0.25)),
        heldout,
        availability,
    )
    key = plain.table["interaction_id"].eq("lr-simple") & plain.table[
        "sender"
    ].eq("sender-b")
    row = plain.table.loc[key].iloc[0]
    expected = 0.5 * (row["ligand_activity_raw"] + row["receptor_activity_raw"])

    assert row["sender_detection_raw"] == pytest.approx(expected)
    assert bottleneck.table.loc[key, "sender_detection_raw"].iloc[0] < expected
    with pytest.raises(ValueError, match=r"one of 0 or 0\.25"):
        AbsoluteActivityV2Spec(bottleneck_penalty=0.1)


def test_complex_and_alternative_or_use_distinct_rules() -> None:
    heldout = _aggregate(("heldout-1",))
    result = _apply(_fit(), heldout, _availability(heldout)).table
    sender_a = result[result["sender"].eq("sender-a")].set_index("interaction_id")

    assert sender_a.loc["lr-and", "ligand_activity_raw"] < sender_a.loc[
        "lr-or", "ligand_activity_raw"
    ]
    assert sender_a.loc["lr-and", "sender_detection_raw"] < sender_a.loc[
        "lr-or", "sender_detection_raw"
    ]


def test_sender_detection_is_invariant_to_candidate_cardinality() -> None:
    heldout = _aggregate(("heldout-1",))
    transform = _fit()
    two = _apply(
        transform,
        heldout,
        _availability(heldout, include_sender_c=False),
    ).table
    three = _apply(transform, heldout, _availability(heldout)).table
    shared = three[three["sender"].isin(("sender-a", "sender-b"))]
    keys = ["sample_id", "sender", "receiver", "interaction_id"]
    comparison = two.merge(shared, on=keys, suffixes=("_two", "_three"))

    assert np.allclose(
        comparison["sender_detection_raw_two"],
        comparison["sender_detection_raw_three"],
    )
    assert set(two["candidate_sender_count"]) == {2}
    assert set(three["candidate_sender_count"]) == {3}
