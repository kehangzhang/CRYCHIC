from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crychic.core import stable_id
from crychic.results import (
    CrychicResult,
    ResultValidationError,
    ResultWriteError,
    write_result,
)
from crychic.scoring import (
    ReceiverScoringFunctionalManifest,
    ScoringCollectionManifest,
    scoring_source_key_digest,
)


def _collection(
    result_payload: dict[str, Any],
    *,
    receiver: str = "B cell",
    repeat_id: str = "repeat-0",
    source_digest: str | None = None,
) -> ScoringCollectionManifest:
    source = result_payload["tables"]["sample_scores"]
    source = source.loc[source["scoring_functional_id"].eq("functional-1")]
    child = ReceiverScoringFunctionalManifest(
        receiver=receiver,
        scoring_functional_id="functional-1",
        source_score_key_digest=source_digest or scoring_source_key_digest(source),
        source_score_row_count=len(source),
    )
    return ScoringCollectionManifest(
        contrast="stim_vs_ctrl",
        repeat_id=repeat_id,
        fold_id="fold-0",
        emitted_receivers=(receiver,),
        children=(child,),
    )


def test_scoring_collection_round_trip_and_table_linkage(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    destination = tmp_path / "result"
    collection = _collection(result_payload)

    result = write_result(
        destination,
        **result_payload,
        scoring_collections=(collection,),
    )

    assert result.has_scoring_collections
    assert not result.has_edge_evidence
    assert (destination / "scoring_collections.json").is_file()
    extension = result.manifest["extensions"]["scoring_collections"]
    assert extension["collections"] == 1
    assert (
        extension["linked_tables"]["sample_scores"]
        == (result.manifest["tables"]["sample_scores"]["sha256"])
    )
    assert (
        extension["linked_tables"]["interactions"]
        == (result.manifest["tables"]["interactions"]["sha256"])
    )
    observed = result.read_scoring_collections()
    assert observed == (collection,)
    assert not observed[0].common_functional_across_receivers
    assert observed[0].emitted_receivers == ("B cell",)
    assert CrychicResult.load(destination).has_scoring_collections


def test_original_result_without_scoring_collection_remains_readable(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    result = write_result(tmp_path / "legacy", **result_payload)

    assert not result.has_scoring_collections
    with pytest.raises(KeyError, match="scoring_collections"):
        result.read_scoring_collections()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("poison", "message"),
    [
        ("receiver", "contrast/receiver edge partition"),
        ("repeat", "repeat does not match"),
        ("source_digest", "source-key digest"),
    ],
)
def test_scoring_collection_poison_is_atomically_rejected(
    tmp_path: Path,
    result_payload: dict[str, Any],
    poison: str,
    message: str,
) -> None:
    destination = tmp_path / f"failed-{poison}"
    collection = _collection(
        result_payload,
        receiver="T cell" if poison == "receiver" else "B cell",
        repeat_id="repeat-1" if poison == "repeat" else "repeat-0",
        source_digest="0" * 64 if poison == "source_digest" else None,
    )

    with pytest.raises(ResultWriteError, match="marked incomplete") as error:
        write_result(
            destination,
            **result_payload,
            scoring_collections=(collection,),
        )

    assert error.value.__cause__ is not None
    assert message in str(error.value.__cause__)
    status = json.loads((destination / "_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "incomplete"
    assert not (destination / "scoring_collections.json").exists()


def test_scoring_collection_file_digest_is_enforced(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    destination = tmp_path / "result"
    write_result(
        destination,
        **result_payload,
        scoring_collections=(_collection(result_payload),),
    )
    collection_path = destination / "scoring_collections.json"
    value = json.loads(collection_path.read_text(encoding="utf-8"))
    value["collections"][0]["emitted_receivers"] = ["poison"]
    collection_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        ResultValidationError,
        match=r"missing or corrupted|does not match its manifest",
    ):
        CrychicResult.load(destination)


def test_scoring_collection_rejects_an_unregistered_functional_member(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    payload = {
        **result_payload,
        "tables": {
            name: table.copy(deep=True)
            for name, table in result_payload["tables"].items()
        },
    }
    sample_scores = payload["tables"]["sample_scores"]
    second = sample_scores.iloc[[0]].copy(deep=True)
    second["subject_id"] = "donor-2"
    second["sample_id"] = "donor-2-stim"
    second["design_row_id"] = stable_id(
        "design_row",
        {
            "context_id": str(second.iloc[0]["context_id"]),
            "sample_id": "donor-2-stim",
        },
    )
    second["scoring_functional_id"] = "functional-2"
    payload["tables"]["sample_scores"] = pd.concat(
        [sample_scores, second], ignore_index=True
    )
    collection = _collection(payload)

    with pytest.raises(ResultWriteError, match="marked incomplete") as error:
        write_result(
            tmp_path / "missing-child",
            **payload,
            scoring_collections=(collection,),
        )

    assert error.value.__cause__ is not None
    assert "exactly cover" in str(error.value.__cause__)


def test_scoring_collection_rejects_source_score_key_poison(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    collection = _collection(result_payload)
    payload = {
        **result_payload,
        "tables": {
            name: table.copy(deep=True)
            for name, table in result_payload["tables"].items()
        },
    }
    sample_scores = payload["tables"]["sample_scores"]
    sample_scores.loc[0, "sample_id"] = "poisoned-sample"
    sample_scores.loc[0, "design_row_id"] = stable_id(
        "design_row",
        {
            "context_id": str(sample_scores.loc[0, "context_id"]),
            "sample_id": "poisoned-sample",
        },
    )

    with pytest.raises(ResultWriteError, match="marked incomplete") as error:
        write_result(
            tmp_path / "poisoned-score",
            **payload,
            scoring_collections=(collection,),
        )

    assert error.value.__cause__ is not None
    assert "source-key digest" in str(error.value.__cause__)
