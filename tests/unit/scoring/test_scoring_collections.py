from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from crychic.scoring import (
    ReceiverScoringFunctionalManifest,
    ScoringCollectionDocument,
    ScoringCollectionManifest,
    scoring_source_key_digest,
)

ROOT = Path(__file__).parents[3]


def _source_keys(functional_id: str = "functional-r1") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "subject-2",
                "sample_id": "sample-2",
                "context_id": "context-treated",
                "design_row_id": "design-2",
                "edge_id": "edge-2",
                "scoring_functional_id": functional_id,
                "repeat_id": "repeat-0",
                "fold_id": "in_sample",
                "mode": "state",
            },
            {
                "subject_id": "subject-1",
                "sample_id": "sample-1",
                "context_id": "context-control",
                "design_row_id": "design-1",
                "edge_id": "edge-1",
                "scoring_functional_id": functional_id,
                "repeat_id": "repeat-0",
                "fold_id": "in_sample",
                "mode": "state",
            },
        ]
    )


def _child(
    receiver: str,
    functional_id: str,
) -> ReceiverScoringFunctionalManifest:
    keys = _source_keys(functional_id)
    return ReceiverScoringFunctionalManifest(
        receiver=receiver,
        scoring_functional_id=functional_id,
        source_score_key_digest=scoring_source_key_digest(keys),
        source_score_row_count=len(keys),
    )


def test_source_key_digest_is_order_independent_and_content_bound() -> None:
    source = _source_keys()

    assert scoring_source_key_digest(source) == scoring_source_key_digest(
        source.iloc[::-1].reset_index(drop=True)
    )
    poisoned = source.copy(deep=True)
    poisoned.loc[0, "edge_id"] = "edge-poison"
    assert scoring_source_key_digest(poisoned) != scoring_source_key_digest(source)


def test_receiver_collection_is_stable_under_child_reordering() -> None:
    r1 = _child("Receiver-1", "functional-r1")
    r2 = _child("Receiver-2", "functional-r2")

    first = ScoringCollectionManifest(
        contrast="treated-v-control",
        repeat_id="repeat-0",
        fold_id="in_sample",
        emitted_receivers=("Receiver-2", "Receiver-1"),
        children=(r2, r1),
    )
    second = ScoringCollectionManifest(
        contrast="treated-v-control",
        repeat_id="repeat-0",
        fold_id="in_sample",
        emitted_receivers=("Receiver-1", "Receiver-2"),
        children=(r1, r2),
    )

    assert first.scoring_collection_id == second.scoring_collection_id
    assert first.source_score_key_digest == second.source_score_key_digest
    assert first.source_score_row_count == 4
    assert not first.common_functional_across_receivers
    assert first.partition_key == "receiver"
    assert first.composition_status == "partial_emitted_only"
    assert [child.receiver for child in first.children] == [
        "Receiver-1",
        "Receiver-2",
    ]


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("override", "message"),
    [
        ({"common_functional_across_receivers": True}, "must declare"),
        ({"emitted_receivers": ("Receiver-1",)}, "exactly match"),
        ({"composition_status": "complete"}, "must be 'partial_emitted_only'"),
        ({"partition_key": "context"}, "must be 'receiver'"),
    ],
)
def test_receiver_collection_rejects_false_composition_claims(
    override: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "contrast": "treated-v-control",
        "repeat_id": "repeat-0",
        "fold_id": "in_sample",
        "emitted_receivers": ("Receiver-1", "Receiver-2"),
        "children": (
            _child("Receiver-1", "functional-r1"),
            _child("Receiver-2", "functional-r2"),
        ),
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        ScoringCollectionManifest(**values)  # type: ignore[arg-type]


def test_receiver_collection_rejects_reused_child_functional() -> None:
    first = _child("Receiver-1", "functional-r1")
    shared_functional = _child("Receiver-2", "functional-r1")

    with pytest.raises(ValueError, match="unique scoring_functional_id"):
        ScoringCollectionManifest(
            contrast="treated-v-control",
            repeat_id="repeat-0",
            fold_id="in_sample",
            emitted_receivers=("Receiver-1", "Receiver-2"),
            children=(first, shared_functional),
        )


def test_serialized_collection_rejects_derived_id_and_digest_poison() -> None:
    collection = ScoringCollectionManifest(
        contrast="treated-v-control",
        repeat_id="repeat-0",
        fold_id="in_sample",
        emitted_receivers=("Receiver-1",),
        children=(_child("Receiver-1", "functional-r1"),),
    )
    document = ScoringCollectionDocument(collections=(collection,))
    serialized = document.to_dict()
    schema = json.loads(
        (ROOT / "schemas" / "scoring_collections.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(serialized)
    round_trip = ScoringCollectionDocument.from_dict(serialized)
    assert round_trip == document

    poisoned_collection = dict(serialized["collections"][0])  # type: ignore[index]
    poisoned_collection["source_score_key_digest"] = "0" * 64
    poisoned = dict(serialized)
    poisoned["collections"] = [poisoned_collection]
    with pytest.raises(ValueError, match="source_score_key_digest"):
        ScoringCollectionDocument.from_dict(poisoned)

    poisoned_child = dict(poisoned_collection["children"][0])
    poisoned_child["child_manifest_id"] = "receiver_scoring_functional_poison"
    poisoned_collection = dict(serialized["collections"][0])  # type: ignore[index]
    poisoned_collection["children"] = [poisoned_child]
    poisoned = dict(serialized)
    poisoned["collections"] = [poisoned_collection]
    with pytest.raises(ValueError, match="child manifest ID"):
        ScoringCollectionDocument.from_dict(poisoned)


def test_receiver_child_rejects_forged_unregistered_metadata() -> None:
    serialized = _child("Receiver-1", "functional-r1").to_dict()
    serialized["model_manifest_id"] = "forged-model"

    with pytest.raises(ValueError, match="fields are invalid"):
        ReceiverScoringFunctionalManifest.from_dict(serialized)
