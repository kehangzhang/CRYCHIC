from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.adapters.crychic.resource import (
    MOLECULAR_LR_CROSSWALK_COLUMNS,
    attach_molecular_lr_equivalence_ids,
    bundle_molecular_lr_crosswalk,
)

from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
)


def _interaction(
    interaction_id: str,
    *,
    direction: str = "Ligand-Receptor",
    agonists: tuple[str, ...] = (),
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=f"source-{interaction_id}",
        ligand_name="L1",
        receptor_name="R1_R2",
        ligand_subunits=("L1",),
        receptor_subunits=("R2", "R1"),
        ligand_is_complex=False,
        receptor_is_complex=True,
        direction=direction,
        source="toy",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        agonist_subunits=agonists,
    )


def _bundle() -> ResourceBundle:
    interactions = (
        _interaction("record-b", agonists=("A1",)),
        _interaction("record-a"),
        _interaction("record-unsupported", direction="Cell-Cell Contact"),
    )
    return ResourceBundle(
        resource_id="toy-resource",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=interactions,
        mapping_report=MappingReport(
            source_rows=len(interactions),
            loaded_rows=len(interactions),
            mapped_entities=6,
        ),
        manifest_digest="d" * 64,
        source_files=("toy.tsv",),
        license="CC0",
        citation="Synthetic fixture",
    )


def test_bundle_crosswalk_preserves_source_rows_and_orthogonal_ids() -> None:
    table = bundle_molecular_lr_crosswalk(_bundle())

    assert tuple(table.columns) == MOLECULAR_LR_CROSSWALK_COLUMNS
    assert table["interaction_id"].tolist() == [
        "record-a",
        "record-b",
        "record-unsupported",
    ]
    mapped = table.loc[table["mapping_status"].eq("mapped")]
    assert mapped["molecular_lr_equivalence_id"].nunique() == 1
    assert mapped["mechanistic_variant_id"].nunique() == 2
    assert mapped["reason_code"].isna().all()

    unsupported = table.loc[
        table["interaction_id"].eq("record-unsupported")
    ].iloc[0]
    assert unsupported["mapping_status"] == "unsupported_direction"
    assert unsupported["reason_code"] == "unsupported_interaction_direction"
    assert pd.isna(unsupported["molecular_lr_equivalence_id"])
    assert pd.isna(unsupported["mechanistic_variant_id"])

    for column in (
        "resource_bundle_content_id",
        "molecular_lr_equivalence_universe_id",
        "molecular_lr_axis_id",
        "mechanistic_variant_axis_id",
        "mapping_axis_id",
    ):
        assert table[column].nunique() == 1


def test_bundle_crosswalk_is_deterministic_and_defensive() -> None:
    first = bundle_molecular_lr_crosswalk(_bundle())
    repeated = bundle_molecular_lr_crosswalk(_bundle())

    pd.testing.assert_frame_equal(first, repeated)
    first.loc[0, "molecular_lr_equivalence_id"] = "forged"
    assert "forged" not in set(
        bundle_molecular_lr_crosswalk(_bundle())["molecular_lr_equivalence_id"]
    )


def test_attach_crosswalk_requires_complete_exact_resource_keys() -> None:
    crosswalk = bundle_molecular_lr_crosswalk(_bundle())
    scores = pd.DataFrame(
        {
            "resource": ["toy-resource", "toy-resource"],
            "resource_version": ["1", "1"],
            "interaction_id": ["record-a", "record-b"],
            "sample_id": ["s1", "s1"],
            "score": [0.2, 0.1],
        }
    )

    attached = attach_molecular_lr_equivalence_ids(scores, crosswalk)
    assert len(attached) == len(scores)
    assert attached["molecular_lr_equivalence_id"].nunique() == 1
    assert attached["mechanistic_variant_id"].nunique() == 2
    assert "mapping_status" not in attached
    assert "reason_code" not in attached

    unsupported = scores.copy()
    unsupported.loc[1, "interaction_id"] = "record-unsupported"
    with pytest.raises(ValueError, match="does not completely map"):
        attach_molecular_lr_equivalence_ids(unsupported, crosswalk)

    missing = scores.copy()
    missing.loc[1, "interaction_id"] = "absent-record"
    with pytest.raises(ValueError, match="does not completely map"):
        attach_molecular_lr_equivalence_ids(missing, crosswalk)


def test_attach_crosswalk_rejects_duplicate_or_existing_identity() -> None:
    crosswalk = bundle_molecular_lr_crosswalk(_bundle())
    scores = pd.DataFrame(
        {
            "resource": ["toy-resource"],
            "resource_version": ["1"],
            "interaction_id": ["record-a"],
        }
    )
    duplicate = pd.concat([crosswalk, crosswalk.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate resource-edge"):
        attach_molecular_lr_equivalence_ids(scores, duplicate)

    scores["molecular_lr_equivalence_id"] = "preexisting"
    with pytest.raises(ValueError, match="already contains"):
        attach_molecular_lr_equivalence_ids(scores, crosswalk)
