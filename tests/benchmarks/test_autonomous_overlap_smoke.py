from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.simulation.run_autonomous_overlap_smoke import (
    AUTONOMOUS_SMOKE_SOURCE_PATHS,
    autonomous_smoke_source_sha256,
    build_artifact_metadata,
    build_compact_summary,
    canonical_payload_sha256,
    run_smoke,
)

from crychic.scoring import INCREMENTAL_DOWNSTREAM_ALGORITHM_CONTRACT

_SUMMARY_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/autonomous_overlap_smoke_v1_summary.json"
)
_FULL_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmark_work/algorithm_smoke/autonomous_overlap_v1.json"
)
_REQUIRED_SOURCE_CLOSURE = {
    "src/crychic/attribution/tuning.py",
    "src/crychic/resources/autonomous_registry.py",
    "src/crychic/resources/manifest.py",
    "src/crychic/response/autonomous.py",
    "src/crychic/scoring/downstream.py",
}


def test_autonomous_smoke_provenance_tracks_algorithm_contract_and_closure() -> None:
    payload = run_smoke()

    assert payload["provenance"]["algorithm_contract"] == (
        INCREMENTAL_DOWNSTREAM_ALGORITHM_CONTRACT
    )
    assert _REQUIRED_SOURCE_CLOSURE.issubset(AUTONOMOUS_SMOKE_SOURCE_PATHS)
    assert set(autonomous_smoke_source_sha256()) == set(AUTONOMOUS_SMOKE_SOURCE_PATHS)


def test_autonomous_overlap_smoke_suppresses_false_gain_and_retains_unique_gain() -> (
    None
):
    payload = run_smoke()

    assert all(payload["checks"].values())
    records = payload["records"]
    generic = [row for row in records if row["scenario"] == "generic_only"]
    active = [row for row in records if row["scenario"] == "active_unique"]
    assert all(row["status"] == "observed" for row in records)
    assert all(row["model_gain"] == pytest.approx(0.0, abs=1e-12) for row in generic)
    assert all(row["model_gain"] > 0.9 for row in active)
    assert all(
        row["retained_norm_fraction"] == pytest.approx(1.0 / 2.0**0.5)
        for row in records
    )
    assert payload["claims"] == {
        "official_incremental_certification": False,
        "method_superiority": False,
        "biological_discovery": False,
    }


def test_tracked_summary_matches_current_semantic_smoke_payload() -> None:
    summary = json.loads(_SUMMARY_PATH.read_text(encoding="utf-8"))
    payload = run_smoke()
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    expected = build_compact_summary(
        payload,
        build_artifact_metadata(
            payload,
            serialized=serialized,
            relative_path="benchmark_work/algorithm_smoke/autonomous_overlap_v1.json",
        ),
    )

    assert summary == expected


def test_workspace_full_autonomous_artifact_matches_live_payload_and_summary() -> None:
    if not _FULL_ARTIFACT_PATH.exists():
        pytest.skip("workspace smoke artifact is intentionally not tracked")
    summary = json.loads(_SUMMARY_PATH.read_text(encoding="utf-8"))
    raw = _FULL_ARTIFACT_PATH.read_bytes()
    full = json.loads(raw)
    live = run_smoke()

    assert full == live
    assert canonical_payload_sha256(full) == summary["artifact"]["canonical_sha256"]
    assert hashlib.sha256(raw).hexdigest() == summary["artifact"]["file_sha256"]
    assert len(raw) == summary["artifact"]["size_bytes"]
