from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = (
    REPO_ROOT / "benchmarks/results/algorithm_crossfit_smoke_v3_summary.json"
)
FULL_ARTIFACT_PATH = (
    REPO_ROOT.parent / "benchmark_work/algorithm_smoke/crossfit_summary_v3.json"
)


def test_algorithm_crossfit_smoke_summary_is_fail_closed() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())

    assert summary["scope"] == (
        "partial_public_crossfit_algorithm_smoke_not_full_benchmark"
    )
    assert summary["exact_oof_coverage_audit_scope"] == [
        "availability_filter",
        "contrast_common_sender_functional",
    ]
    assert all(value is False for value in summary["guardrails"].values())
    assert all(
        record["completed_stage_oof_verified"] is True
        and record["complete_pipeline_oof_certified"] is False
        for record in summary["records"]
    )
    kang = next(
        record
        for record in summary["records"]
        if record["dataset"] == "kang2018_4donor_3celltype"
    )
    assert kang["receiver_family_application_reason_counts"] == {"observed": 6}


def test_algorithm_crossfit_smoke_full_artifact_matches_tracked_digest() -> None:
    if not FULL_ARTIFACT_PATH.exists():
        pytest.skip("workspace smoke artifact is intentionally not tracked")
    summary = json.loads(SUMMARY_PATH.read_text())
    full = json.loads(FULL_ARTIFACT_PATH.read_text())
    canonical = json.dumps(
        full,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    assert full["schema_version"] == "crychic-crossfit-smoke-v3"
    assert hashlib.sha256(canonical).hexdigest() == summary["full_artifact"][
        "canonical_sha256"
    ]
    assert full["process_peak_rss_kib"] == summary["process_performance"][
        "peak_rss_kib"
    ]
