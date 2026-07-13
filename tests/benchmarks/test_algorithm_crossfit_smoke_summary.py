from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.simulation.run_crossfit_smoke import (
    CROSSFIT_SMOKE_SOURCE_PATHS,
    crossfit_smoke_source_sha256,
)

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
V5_SUMMARY_PATH = (
    REPO_ROOT / "benchmarks/results/algorithm_crossfit_smoke_v5_summary.json"
)
V5_FULL_ARTIFACT_PATH = (
    REPO_ROOT.parent / "benchmark_work/algorithm_smoke/crossfit_summary_v5.json"
)
REQUIRED_V5_SOURCE_CLOSURE = {
    "src/crychic/attribution/tuning.py",
    "src/crychic/resources/autonomous_registry.py",
    "src/crychic/resources/manifest.py",
    "src/crychic/response/autonomous.py",
    "src/crychic/scoring/downstream.py",
    "src/crychic/scoring/family_common.py",
    "src/crychic/workflow/crossfit.py",
    "src/crychic/workflow/receiver_incremental.py",
}


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


def test_typed_crossfit_smoke_v5_records_current_diagnostic_boundary() -> None:
    summary = json.loads(V5_SUMMARY_PATH.read_text())
    by_dataset = {record["dataset"]: record for record in summary["records"]}

    assert all(value is False for value in summary["guardrails"].values())
    assert (
        by_dataset["synthetic_active"]["mean_raw_incremental_diagnostic_gain"]
        > by_dataset["synthetic_ligand_only"][
            "mean_raw_incremental_diagnostic_gain"
        ]
    )
    assert (
        by_dataset["synthetic_active"]["mean_bounded_incremental_diagnostic_gain"]
        > by_dataset["synthetic_ligand_only"][
            "mean_bounded_incremental_diagnostic_gain"
        ]
    )
    assert all(
        record["gain_denominator_status_counts"]
        == {"positive_receiver_null_loss_ratio_v1": 2}
        and record["incremental_loss_design_counts"]
        == {"fully_paired_subject_contrasts_v1": 2}
        and record["official_incremental_status_counts"] == {"not_estimable": 48}
        for record in summary["records"]
    )


def test_typed_crossfit_smoke_v5_binds_the_current_source_closure() -> None:
    summary = json.loads(V5_SUMMARY_PATH.read_text())

    assert REQUIRED_V5_SOURCE_CLOSURE.issubset(CROSSFIT_SMOKE_SOURCE_PATHS)
    assert summary["source_sha256"] == crossfit_smoke_source_sha256()


def test_typed_crossfit_smoke_v5_full_artifact_matches_tracked_digest() -> None:
    if not V5_FULL_ARTIFACT_PATH.exists():
        pytest.skip("workspace smoke artifact is intentionally not tracked")
    summary = json.loads(V5_SUMMARY_PATH.read_text())
    raw = V5_FULL_ARTIFACT_PATH.read_bytes()
    full = json.loads(raw)
    canonical = json.dumps(
        full,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    assert full["schema_version"] == "crychic-crossfit-smoke-v5"
    assert (
        hashlib.sha256(canonical).hexdigest()
        == (summary["full_artifact"]["canonical_sha256"])
    )
    assert hashlib.sha256(raw).hexdigest() == summary["full_artifact"]["file_sha256"]
    assert len(raw) == summary["full_artifact"]["size_bytes"]
    assert full["source_sha256"] == summary["source_sha256"]
    full_by_dataset = {record["dataset"]: record for record in full["records"]}
    for compact in summary["records"]:
        source = full_by_dataset[compact["dataset"]]
        assert compact == {key: source[key] for key in compact}
