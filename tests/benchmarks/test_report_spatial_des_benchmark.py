from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.literature.report_spatial_des_benchmark import (
    LEADERBOARD_FILENAME,
    SUMMARY_MANIFEST_SOURCE_FILENAME,
    SUMMARY_SOURCE_FILENAME,
    run,
    validate_publication_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_summary(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    summary = pd.DataFrame(
        {
            "comparison_panel_id": ["panel", "panel"],
            "dataset": ["fixture", "fixture"],
            "scenario": ["multi_sample", "multi_sample"],
            "analysis_unit": ["subject_id", "subject_id"],
            "expected_variant": ["primary", "primary"],
            "method": ["crychic", "scseqcommdiff"],
            "method_version": ["0.1.0", "2.0.0"],
            "resource": ["ConnectomeDB2020", "ConnectomeDB2020"],
            "ranking_semantics": ["subject effect", "subject native"],
            "des_strata_expected": [8, 8],
            "des_strata_observed": [8, 8],
            "des_median": [0.8, 0.4],
            "des_mean": [0.75, 0.45],
            "rank_eligible": [True, True],
            "median_rank": [1, 2],
            "mean_rank": [1, 2],
            "rank_eligible_fraction_min": [0.9, 0.8],
            "rank_eligible_fraction_mean": [0.95, 0.85],
            "expected_set_coverage_min": [1.0, 0.75],
            "expected_set_coverage_mean": [1.0, 0.9],
        }
    )
    summary_path = root / "spatial_des_benchmark_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False, lineterminator="\n")
    manifest = {
        "schema_version": "crychic-spatial-des-benchmark-summary-v1",
        "status": "complete",
        "output": {
            "filename": summary_path.name,
            "rows": len(summary),
            "sha256": _sha256(summary_path),
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return summary_path, manifest_path


def test_report_writes_bound_leaderboard_figures_and_markdown(tmp_path: Path) -> None:
    summary, manifest = _write_summary(tmp_path / "summary")
    output = tmp_path / "report"

    payload = run(summary, manifest, output)

    assert payload["status"] == "complete"
    assert payload["protocol"]["no_cross_panel_ranking"] is True
    assert (output / SUMMARY_SOURCE_FILENAME).read_bytes() == summary.read_bytes()
    assert (
        output / SUMMARY_MANIFEST_SOURCE_FILENAME
    ).read_bytes() == manifest.read_bytes()
    assert payload["input"]["summary_sha256"] == _sha256(
        output / SUMMARY_SOURCE_FILENAME
    )
    assert payload["input"]["summary_manifest_sha256"] == _sha256(
        output / SUMMARY_MANIFEST_SOURCE_FILENAME
    )
    leaderboard = pd.read_csv(output / LEADERBOARD_FILENAME, sep="\t")
    assert leaderboard["method_display"].tolist() == ["CRYCHIC", "scSeqCommDiff"]
    assert leaderboard["median_rank"].tolist() == [1, 2]
    assert leaderboard["mean_rank"].tolist() == [1, 2]
    assert (output / "figure_des_median.png").stat().st_size > 1000
    assert (output / "figure_des_mean.png").stat().st_size > 1000
    assert (output / "figure_rank_coverage.png").stat().st_size > 1000
    assert "Panels never mix" in (output / "README.md").read_text(encoding="utf-8")
    assert "primary" in (output / "README.md").read_text(encoding="utf-8")
    assert "panel" in (output / "README.md").read_text(encoding="utf-8")
    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == payload
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(summary, manifest, output)


def test_report_rejects_tampered_summary(tmp_path: Path) -> None:
    summary, manifest = _write_summary(tmp_path / "summary")
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256"):
        run(summary, manifest, tmp_path / "report")


def test_report_rejects_rank_mismatch(tmp_path: Path) -> None:
    summary, manifest = _write_summary(tmp_path / "summary")
    table = pd.read_csv(summary, sep="\t")
    table["median_rank"] = [2, 1]
    table.to_csv(summary, sep="\t", index=False, lineterminator="\n")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["output"]["sha256"] = _sha256(summary)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rank mismatch"):
        run(summary, manifest, tmp_path / "report")


def test_report_groups_and_labels_distinct_truth_panels(tmp_path: Path) -> None:
    summary, manifest = _write_summary(tmp_path / "summary")
    primary = pd.read_csv(summary, sep="\t")
    sensitivity = primary.copy()
    sensitivity["comparison_panel_id"] = "panel_sensitivity"
    sensitivity["expected_variant"] = "sensitivity"
    interleaved = pd.concat(
        [
            primary.iloc[[0]],
            sensitivity.iloc[[0]],
            primary.iloc[[1]],
            sensitivity.iloc[[1]],
        ],
        ignore_index=True,
    )
    interleaved.to_csv(summary, sep="\t", index=False, lineterminator="\n")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["output"]["rows"] = len(interleaved)
    payload["output"]["sha256"] = _sha256(summary)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "report"
    run(summary, manifest, output)

    leaderboard = pd.read_csv(output / LEADERBOARD_FILENAME, sep="\t")
    assert leaderboard["comparison_panel_id"].tolist() == [
        "panel",
        "panel",
        "panel_sensitivity",
        "panel_sensitivity",
    ]
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "Truth variant" in readme
    assert "panel_sensitivity" in readme


def test_primary_report_rejects_declared_sensitivity_methods(tmp_path: Path) -> None:
    summary, manifest = _write_summary(tmp_path / "summary")
    table = pd.read_csv(summary, sep="\t")
    table.loc[0, "method"] = "liana_rank_aggregate"
    table.to_csv(summary, sep="\t", index=False, lineterminator="\n")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["output"]["sha256"] = _sha256(summary)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="declared sensitivity"):
        run(summary, manifest, tmp_path / "report")


def test_publication_validator_rejects_large_or_private_artifacts(
    tmp_path: Path,
) -> None:
    summary, manifest = _write_summary(tmp_path / "summary")
    output = tmp_path / "report"
    run(summary, manifest, output)

    private = output / "raw.h5ad"
    private.write_bytes(b"private")
    with pytest.raises(ValueError, match="forbidden publication artifact"):
        validate_publication_bundle(output)
    private.unlink()

    readme = output / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\nsource=/media/private/run.tsv\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absolute machine path"):
        validate_publication_bundle(output)
