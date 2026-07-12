import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crychic.results import (
    EDGE_EVIDENCE_EXTENSION_VERSION,
    CrychicResult,
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
    write_result,
)


def test_edge_evidence_extension_round_trip_and_manifest_linkage(
    tmp_path: Path,
    result_payload: dict[str, Any],
    edge_evidence_frame: pd.DataFrame,
) -> None:
    destination = tmp_path / "result"

    result = write_result(
        destination,
        **result_payload,
        edge_evidence=edge_evidence_frame,
    )

    extension = result.manifest["extensions"]["edge_evidence"]
    assert result.has_edge_evidence
    assert (destination / "edge_evidence.parquet").is_file()
    assert extension["extension_schema_version"] == EDGE_EVIDENCE_EXTENSION_VERSION
    assert (
        extension["linked_tables"]["sample_scores"]
        == (result.manifest["tables"]["sample_scores"]["sha256"])
    )
    assert not tuple(tmp_path.glob(".result.tmp-*"))
    selected = result.read_edge_evidence(
        filters={"sender": "Monocyte"},
        columns=["sample_id", "interaction_id", "legacy_integrated_strength"],
    )
    assert selected.to_dict(orient="records") == [
        {
            "sample_id": "donor-1-stim",
            "interaction_id": "CXCL10_CXCR3",
            "legacy_integrated_strength": 0.82,
        }
    ]
    assert CrychicResult.load(destination).has_edge_evidence


def test_original_v0_1_result_without_extension_remains_readable(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    result = write_result(tmp_path / "legacy", **result_payload)

    assert not result.has_edge_evidence
    assert "extensions" not in result.manifest
    with pytest.raises(KeyError, match="edge_evidence"):
        result.read_edge_evidence()


def test_edge_evidence_manifest_rejects_bad_table_linkage(
    tmp_path: Path,
    result_payload: dict[str, Any],
    edge_evidence_frame: pd.DataFrame,
) -> None:
    destination = tmp_path / "result"
    write_result(
        destination,
        **result_payload,
        edge_evidence=edge_evidence_frame,
    )
    manifest_path = destination / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extensions"]["edge_evidence"]["linked_tables"]["sample_scores"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ResultValidationError, match="linkage does not match"):
        CrychicResult.load(destination)


def test_edge_evidence_file_digest_is_enforced(
    tmp_path: Path,
    result_payload: dict[str, Any],
    edge_evidence_frame: pd.DataFrame,
) -> None:
    destination = tmp_path / "result"
    write_result(
        destination,
        **result_payload,
        edge_evidence=edge_evidence_frame,
    )
    extension_path = destination / "edge_evidence.parquet"
    extension_path.write_bytes(extension_path.read_bytes() + b"corruption")

    with pytest.raises(
        ResultValidationError, match=r"missing or corrupted|does not match its manifest"
    ):
        CrychicResult.load(destination)


def test_invalid_edge_evidence_keeps_atomic_failure_semantics(
    tmp_path: Path,
    result_payload: dict[str, Any],
    edge_evidence_frame: pd.DataFrame,
) -> None:
    destination = tmp_path / "failed"
    invalid = pd.concat([edge_evidence_frame, edge_evidence_frame], ignore_index=True)

    with pytest.raises(ResultWriteError, match="marked incomplete"):
        write_result(destination, **result_payload, edge_evidence=invalid)

    status = json.loads((destination / "_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "incomplete"
    assert not (destination / "edge_evidence.parquet").exists()
    with pytest.raises(IncompleteResultError):
        CrychicResult.load(destination)


@pytest.mark.parametrize("poison", ["interaction_id", "legacy_integrated_strength"])
def test_cross_table_edge_evidence_poison_is_atomically_rejected(
    tmp_path: Path,
    result_payload: dict[str, Any],
    edge_evidence_frame: pd.DataFrame,
    poison: str,
) -> None:
    destination = tmp_path / f"failed-{poison}"
    invalid = edge_evidence_frame.copy(deep=True)
    invalid.loc[0, poison] = "CD40LG_CD40" if poison == "interaction_id" else 0.81

    with pytest.raises(ResultWriteError, match="marked incomplete"):
        write_result(destination, **result_payload, edge_evidence=invalid)

    status = json.loads((destination / "_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "incomplete"
    assert not (destination / "edge_evidence.parquet").exists()
