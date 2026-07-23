import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from crychic.results import (
    CrychicResult,
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
    write_result,
)


def _write(path: Path, payload: dict[str, Any]) -> CrychicResult:
    return write_result(path, **payload)


def test_atomic_round_trip_and_immutable_metadata(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    destination = tmp_path / "result"

    result = _write(destination, result_payload)

    assert result.path == destination.resolve()
    assert (
        json.loads((destination / "_status.json").read_text())["status"] == "complete"
    )
    assert CrychicResult.load(destination).manifest["run_id"] == "run-fixture"
    assert not tuple(tmp_path.glob(".result.tmp-*"))
    with pytest.raises(FrozenInstanceError):
        result.path = tmp_path  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.manifest["tables"]["interactions"]["rows"] = 4


def test_profiling_metadata_round_trips(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    result_payload["run_manifest"]["profiling"] = {
        "clock": "time.perf_counter",
        "scope": "fit_baseline_excludes_result_persistence",
        "stage_seconds": {"response": 1.25, "fit_total": 2.5},
    }

    result = _write(tmp_path / "profiled", result_payload)

    assert result.manifest["profiling"]["stage_seconds"] == {
        "fit_total": 2.5,
        "response": 1.25,
    }


@pytest.mark.parametrize(
    "profiling",
    [
        {
            "clock": "wall",
            "scope": "fit_baseline_excludes_result_persistence",
            "stage_seconds": {"fit_total": 1.0},
        },
        {
            "clock": "time.perf_counter",
            "scope": "fit_baseline_excludes_result_persistence",
            "stage_seconds": {"fit_total": -1.0},
        },
        {
            "clock": "time.perf_counter",
            "scope": "fit_baseline_excludes_result_persistence",
            "stage_seconds": {"response": 1.0},
        },
    ],
)
def test_invalid_profiling_metadata_is_rejected(
    tmp_path: Path,
    result_payload: dict[str, Any],
    profiling: dict[str, Any],
) -> None:
    result_payload["run_manifest"]["profiling"] = profiling

    with pytest.raises(ResultWriteError) as exc_info:
        _write(tmp_path / "invalid-profile", result_payload)
    assert "profiling" in str(exc_info.value.__cause__)


def test_failed_write_is_marked_incomplete_and_cannot_load(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    destination = tmp_path / "failed"
    interactions = result_payload["tables"]["interactions"]
    result_payload["tables"]["interactions"] = interactions.iloc[[0, 0]].copy()

    with pytest.raises(ResultWriteError, match="marked incomplete"):
        _write(destination, result_payload)

    marker = json.loads((destination / "_status.json").read_text())
    assert marker["status"] == "incomplete"
    with pytest.raises(IncompleteResultError):
        CrychicResult.load(destination)


def test_corrupted_table_is_rejected(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    destination = tmp_path / "result"
    _write(destination, result_payload)
    table_path = destination / "interactions.parquet"
    table_path.write_bytes(table_path.read_bytes() + b"corruption")

    with pytest.raises(ResultValidationError, match="does not match its manifest"):
        CrychicResult.load(destination)
