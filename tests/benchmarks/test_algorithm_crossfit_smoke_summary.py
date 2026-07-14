from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.simulation.run_crossfit_smoke import (
    CROSSFIT_SMOKE_SOURCE_PATHS,
    DEFAULT_SCENARIOS,
    build_artifact_metadata,
    build_compact_summary,
    build_parser,
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
    "src/crychic/workflow/repeated_crossfit.py",
}


def _compact_source_record(
    scenario: str,
    *,
    elapsed_seconds: float,
    bounded_gain: float,
    raw_gain: float,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "crossfit_id": f"crossfit-{scenario}",
        "dataset": f"synthetic_{scenario}",
        "elapsed_seconds": elapsed_seconds,
        "gain_denominator_status_counts": {"positive": 2},
        "incremental_loss_design_counts": {"paired": 2},
        "mean_bounded_incremental_diagnostic_gain": bounded_gain,
        "mean_raw_incremental_diagnostic_gain": raw_gain,
        "n_cells": 100,
        "n_folds": 2,
        "n_genes": 10,
        "n_subjects": 8,
        "observed_incremental_diagnostic_count": 2,
        "official_incremental_status_counts": {"not_estimable": 48},
        "receiver_coverage_audit_id": f"coverage-{scenario}",
        "remaining_stages": ["attribution_tuning", "incremental_downstream"],
    }


def _compact_payload() -> dict[str, object]:
    return {
        "schema_version": "crychic-crossfit-smoke-v5",
        "scope": "typed_public_crossfit_algorithm_smoke_not_full_benchmark",
        "source_sha256": {"source.py": "1" * 64},
        "process_peak_rss_kib": 1234,
        "records": [
            _compact_source_record(
                "ligand_only",
                elapsed_seconds=2.5,
                bounded_gain=0.0,
                raw_gain=-0.125,
            ),
            _compact_source_record(
                "active",
                elapsed_seconds=1.5,
                bounded_gain=0.25,
                raw_gain=0.25,
            ),
        ],
    }


def test_crossfit_smoke_cli_defaults_remain_backward_compatible() -> None:
    args = build_parser().parse_args([])

    assert args.output == Path("benchmark_work/algorithm_smoke/crossfit_summary.json")
    assert tuple(args.scenarios) == DEFAULT_SCENARIOS
    assert args.include_kang_subset is False
    assert args.summary_output is None
    assert args.base_revision is None
    assert args.artifact_relative_workspace_path == (
        "benchmark_work/algorithm_smoke/crossfit_summary.json"
    )


def test_crossfit_smoke_cli_accepts_compact_summary_metadata() -> None:
    args = build_parser().parse_args(
        [
            "--summary-output",
            "benchmarks/results/summary.json",
            "--base-revision",
            "abc123",
            "--generated-on",
            "2026-07-14",
            "--artifact-relative-workspace-path",
            "benchmark_work/algorithm_smoke/crossfit_summary_v5.json",
        ]
    )

    assert args.summary_output == Path("benchmarks/results/summary.json")
    assert args.base_revision == "abc123"
    assert args.generated_on == "2026-07-14"
    assert args.artifact_relative_workspace_path.endswith("crossfit_summary_v5.json")


def test_v5_compact_summary_is_derived_from_payload_and_serialized_bytes() -> None:
    payload = _compact_payload()
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    artifact = build_artifact_metadata(
        payload,
        serialized=serialized,
        relative_workspace_path="benchmark_work/algorithm_smoke/test.json",
    )

    summary = build_compact_summary(
        payload,
        artifact,
        base_revision="base-revision",
        generated_on="2026-07-14",
    )

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert artifact == {
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "file_sha256": hashlib.sha256(serialized).hexdigest(),
        "relative_workspace_path": "benchmark_work/algorithm_smoke/test.json",
        "size_bytes": len(serialized),
    }
    assert [record["dataset"] for record in summary["records"]] == [
        "synthetic_active",
        "synthetic_ligand_only",
    ]
    assert [record["crossfit_id"] for record in summary["records"]] == [
        "crossfit-active",
        "crossfit-ligand_only",
    ]
    assert summary["process_performance"] == {
        "peak_rss_kib": 1234,
        "wall_clock_seconds": 4.0,
    }
    assert "(0.2500000)" in summary["interpretation"]
    assert "(-0.1250000)" in summary["interpretation"]
    assert (
        "all 48 official rows per dataset remain not_estimable"
        in summary["interpretation"]
    )


def test_v5_compact_summary_rejects_noncanonical_record_sets() -> None:
    payload = _compact_payload()
    payload["records"] = [payload["records"][0]]  # type: ignore[index]

    with pytest.raises(ValueError, match="canonical active and ligand_only"):
        build_compact_summary(
            payload,
            {},
            base_revision="base-revision",
            generated_on="2026-07-14",
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


def test_typed_crossfit_smoke_v5_records_current_diagnostic_boundary() -> None:
    summary = json.loads(V5_SUMMARY_PATH.read_text())
    by_dataset = {record["dataset"]: record for record in summary["records"]}

    assert all(value is False for value in summary["guardrails"].values())
    assert (
        by_dataset["synthetic_active"]["mean_raw_incremental_diagnostic_gain"]
        > by_dataset["synthetic_ligand_only"]["mean_raw_incremental_diagnostic_gain"]
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
    expected = build_compact_summary(
        full,
        build_artifact_metadata(
            full,
            serialized=raw,
            relative_workspace_path=summary["full_artifact"]["relative_workspace_path"],
        ),
        base_revision=summary["base_revision"],
        generated_on=summary["generated_on"],
    )
    assert summary == expected
