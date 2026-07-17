from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from benchmarks.literature.publish_extension_bundle import (
    COPIED_TABLES,
    SOURCE_MANIFESTS,
    publish_bundle,
    validate_bundle,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_source(root: Path) -> None:
    reports = {
        "reports/literature_extension_20260717/REPORT.md": (
            "# Results\n\n[old](../../results/ipf/full_cohort/full_task_ledger.tsv)\n"
        ),
        "reports/citeseq_extended_7datasets/REPORT.md": "# CITE-seq\n",
        "cytokine/her2/REPORT.md": "# HER2\n\n## Checksum linkage\n[raw](x.h5ad)\n",
        "results/ipf/full_cohort/REPORT.md": "# IPF\n\n## 可复现产物\nraw\n",
    }
    for relative, text in reports.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    for source in COPIED_TABLES.values():
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("metric\n1\n", encoding="utf-8")
    sample_metrics = root / "results/ipf/full_cohort/cohort_summary/sample_metrics.tsv"
    sample_metrics.write_text(
        "sample_id\tevaluation_dir\tcommand\tauroc\n"
        "s1\t/media/private/run/s1\tpython score.py\t0.7\n",
        encoding="utf-8",
    )

    payloads: dict[str, dict[str, object]] = {
        "joint": {
            "generated_on": "2026-07-17",
            "headline": {"citeseq_datasets_complete": 7},
        },
        "citeseq": {
            "code": {"commit": "a" * 40, "dirty": True},
            "comparison_contract": {"primary": "H-common/resource_fixed"},
            "dataset_count": 7,
            "datasets": ["d1"],
        },
        "her2_preparation": {
            "dataset_id": "Wu_GSE176078_HER2",
            "truth_status": "reconstructed",
        },
        "her2_evaluation": {"limitations": ["descriptive"]},
        "ipf": {
            "source_record": "10.5281/zenodo.6497091",
            "cohort": {"samples": 56},
            "execution": {"tasks_complete": 168},
            "evaluation_attestation": {
                "evaluation_manifest_aggregate_sha256": "b" * 64
            },
            "repository_state": {"git_head": "a" * 40},
            "frozen_inputs": {"gold": {"path": "/private/x", "sha256": "c" * 64}},
            "artifacts": {"full_task_ledger.tsv": "d" * 64},
            "limitations": ["static"],
        },
        "ipf_summary": {"status": "complete"},
    }
    for name, relative in SOURCE_MANIFESTS.items():
        _write_json(root / relative, payloads[name])


def test_publish_bundle_sanitizes_machine_specific_fields(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "published"
    _fake_source(source)

    manifest = publish_bundle(source, output)

    with (output / "ipf/sample_metrics.tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == ["sample_id", "auroc"]
        assert list(reader) == [{"sample_id": "s1", "auroc": "0.7"}]
    assert manifest["safety"]["ipf_sample_metrics_dropped_columns"] == [
        "command",
        "evaluation_dir",
    ]
    assert not any(
        path.suffix in {".h5ad", ".parquet", ".rds", ".xlsx", ".log"}
        for path in output.rglob("*")
    )
    assert "ipf/manifest.json" in (output / "README.md").read_text()
    validate_bundle(output)


def test_validate_bundle_rejects_absolute_paths_and_broken_links(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    readme = bundle / "README.md"
    readme.write_text("[missing](missing.tsv)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved Markdown link"):
        validate_bundle(bundle)

    readme.write_text("source=/home/user/private.tsv\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absolute machine path"):
        validate_bundle(bundle)
