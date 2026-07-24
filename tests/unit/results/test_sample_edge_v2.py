from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.support.sample_edge_v2 import sample_edge_scores

from crychic.results import (
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
    SampleEdgeScoreV2Artifact,
    write_sample_edge_score_v2,
)


def test_sample_edge_v2_atomic_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "sample-edge"

    result = write_sample_edge_score_v2(destination, sample_edge_scores())
    loaded = SampleEdgeScoreV2Artifact.load(destination)

    assert result.path == destination.resolve()
    assert loaded.scores.table.equals(result.scores.table)
    assert loaded.scores.provenance == result.scores.provenance
    assert loaded.manifest["table"]["rows"] == 2
    assert json.loads((destination / "_status.json").read_text())["status"] == (
        "complete"
    )
    assert not tuple(tmp_path.glob(".sample-edge.tmp-*"))


def test_sample_edge_v2_corruption_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "sample-edge"
    write_sample_edge_score_v2(destination, sample_edge_scores())
    table_path = destination / "sample_edge_scores_v2.parquet"
    table_path.write_bytes(table_path.read_bytes() + b"corruption")

    with pytest.raises(ResultValidationError, match="missing or corrupted"):
        SampleEdgeScoreV2Artifact.load(destination)


def test_sample_edge_v2_failed_write_is_marked_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "sample-edge"

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("injected parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(ResultWriteError, match="marked incomplete"):
        write_sample_edge_score_v2(destination, sample_edge_scores())

    status = json.loads((destination / "_status.json").read_text())
    assert status["status"] == "incomplete"
    with pytest.raises(IncompleteResultError):
        SampleEdgeScoreV2Artifact.load(destination)
