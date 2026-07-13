from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "benchmarks/results/algorithm_crossfit_smoke_v3_summary.json"
FULL_ARTIFACT_PATH = (
    REPO_ROOT.parent / "benchmark_work/algorithm_smoke/crossfit_summary_v3.json"
)
V4_SUMMARY_PATH = (
    REPO_ROOT / "benchmarks/results/algorithm_crossfit_smoke_v4_summary.json"
)
V4_FULL_ARTIFACT_PATH = (
    REPO_ROOT.parent / "benchmark_work/algorithm_smoke/crossfit_summary_v4.json"
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
    assert (
        hashlib.sha256(canonical).hexdigest()
        == summary["full_artifact"]["canonical_sha256"]
    )
    assert (
        full["process_peak_rss_kib"] == summary["process_performance"]["peak_rss_kib"]
    )


def test_typed_crossfit_smoke_v4_is_fail_closed() -> None:
    summary = json.loads(V4_SUMMARY_PATH.read_text())

    assert summary["scope"] == (
        "typed_public_crossfit_algorithm_smoke_not_full_benchmark"
    )
    assert summary["exact_oof_coverage_audit_scope"] == [
        "availability_filter",
        "contrast_common_sender_functional",
        "receiver_incremental_application_diagnostic",
    ]
    assert all(value is False for value in summary["guardrails"].values())
    assert "response_precision" not in summary["remaining_stages"]
    assert "receiver_autonomous_nuisance" in summary["remaining_stages"]
    assert "subject_blocked_inner_tuning" in summary["remaining_stages"]
    assert all(
        record["n_receiver_response_artifacts"]
        == record["n_response_precision_artifacts"]
        == record["n_receiver_incremental_training_artifacts"]
        == record["n_receiver_incremental_applications"]
        == 6
        and record["n_oof_receiver_coverage_rows"] == 48
        and record["response_precision_status_counts"] == {"estimable": 6}
        and record["official_incremental_status_counts"] == {"not_estimable": 48}
        for record in summary["records"]
    )
    assert all(
        record["mean_raw_incremental_diagnostic_gain"] < 0
        for record in summary["records"]
    )


def test_typed_crossfit_smoke_v4_full_artifact_matches_tracked_digest() -> None:
    if not V4_FULL_ARTIFACT_PATH.exists():
        pytest.skip("workspace smoke artifact is intentionally not tracked")
    summary = json.loads(V4_SUMMARY_PATH.read_text())
    raw = V4_FULL_ARTIFACT_PATH.read_bytes()
    full = json.loads(raw)
    canonical = json.dumps(
        full,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    assert full["schema_version"] == "crychic-crossfit-smoke-v4"
    assert (
        hashlib.sha256(canonical).hexdigest()
        == summary["full_artifact"]["canonical_sha256"]
    )
    assert hashlib.sha256(raw).hexdigest() == summary["full_artifact"]["file_sha256"]
    assert len(raw) == summary["full_artifact"]["size_bytes"]
    assert (
        full["process_peak_rss_kib"] == summary["process_performance"]["peak_rss_kib"]
    )
