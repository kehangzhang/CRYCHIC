from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import benchmarks.datasets.audit_kuppe_repeated_measures as module
import pandas as pd
import pytest


def _realistic_sample_design() -> pd.DataFrame:
    subject_regions = {
        "P1": ("CTRL",),
        "P7": ("CTRL",),
        "P8": ("CTRL",),
        "P17": ("CTRL",),
        "P2": ("RZ", "BZ", "IZ"),
        "P3": ("RZ", "BZ", "IZ"),
        "P9": ("RZ", "IZ"),
        "P6": ("RZ",),
        "P11": ("RZ",),
        "P12": ("BZ",),
        "P10": ("IZ",),
        "P13": ("IZ",),
        "P15": ("IZ",),
        "P16": ("IZ",),
        "P4": ("FZ",),
        "P5": ("FZ",),
        "P14": ("FZ",),
        "P18": ("FZ",),
        "P19": ("FZ",),
        "P20": ("FZ",),
    }
    rows: list[dict[str, object]] = []
    index = 0
    for subject, regions in subject_regions.items():
        for region in regions:
            condition = (
                "ischemic"
                if region == "IZ"
                else "fibrotic"
                if region == "FZ" and subject != "P5"
                else "myogenic"
            )
            rows.append(
                {
                    "sample_id": f"S{index}",
                    "subject_id": subject,
                    "region": region,
                    "condition": condition,
                    "n_cells": 10,
                }
            )
            index += 1
    for subject, copies in (("P9", 2), ("P15", 1), ("P16", 1)):
        for _ in range(copies):
            rows.append(
                {
                    "sample_id": f"S{index}",
                    "subject_id": subject,
                    "region": "IZ",
                    "condition": "ischemic",
                    "n_cells": 10,
                }
            )
            index += 1
    return pd.DataFrame(rows)


def test_realistic_metadata_audit_freezes_all_region_contrasts() -> None:
    result = module.audit_kuppe_repeated_measures(
        _realistic_sample_design(),
        lineage={"input_kind": "fixture", "expression_matrix_accessed": False},
    )

    assert result["support"]["n_samples"] == 29
    assert result["support"]["n_subjects"] == 20
    region_model = cast(dict[str, object], result["region_model"])
    assert region_model["contrast_count"] == 10
    assert region_model["status_counts"] == {
        "estimable_under_complete_score_coverage": 10
    }
    contrasts = cast(list[dict[str, object]], region_model["contrasts"])
    rz_bz = next(row for row in contrasts if row["contrast"] == "BZ_vs_RZ")
    assert rz_bz["design_kind"] == "mixed_paired_unpaired_subject_cluster_ols"
    assert rz_bz["n_paired_subjects"] == 2
    assert rz_bz["n_contrast_subject_clusters"] == 6
    assert rz_bz["n_replicated_design_cells"] == 3


def test_condition_adjusted_for_region_fails_closed_as_confounded() -> None:
    result = module.audit_kuppe_repeated_measures(
        _realistic_sample_design(),
        lineage={"input_kind": "fixture", "expression_matrix_accessed": False},
    )
    model = cast(dict[str, object], result["condition_region_adjusted_model"])
    assert model["contrast_count"] == 3
    assert model["status_counts"] == {"not_estimable": 3}
    for contrast in cast(list[dict[str, object]], model["contrasts"]):
        assert contrast["design_status"] == "not_estimable"
        assert contrast["reason_code"] == "rank_deficient_declared_design"


def test_sample_design_tsv_without_condition_skips_condition_audit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "design.tsv"
    _realistic_sample_design().drop(columns="condition").to_csv(
        path, sep="\t", index=False
    )
    design = module.read_sample_design(path)
    result = module.audit_kuppe_repeated_measures(
        design,
        lineage={"input_kind": "sample_design_tsv"},
    )

    model = cast(dict[str, object], result["condition_region_adjusted_model"])
    assert model["status"] == "not_run"
    assert model["reason_code"] == "condition_field_not_available_in_sample_design"
    assert result["condition_manifest_sha256"] is None


def test_core_manifest_is_stable_when_backed_obs_adds_condition() -> None:
    complete = _realistic_sample_design()
    without_condition = complete.drop(columns="condition")
    complete_result = module.audit_kuppe_repeated_measures(
        complete,
        lineage={"input_kind": "backed_obs"},
    )
    tsv_result = module.audit_kuppe_repeated_measures(
        without_condition,
        lineage={"input_kind": "sample_design_tsv"},
    )

    assert complete_result["sample_manifest_sha256"] == (
        tsv_result["sample_manifest_sha256"]
    )
    assert complete_result["condition_manifest_sha256"] is not None
    assert tsv_result["condition_manifest_sha256"] is None


def test_backed_loader_reads_obs_and_closes_without_matrix_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.h5ad"
    source_path.write_bytes(b"fixture")
    obs = _realistic_sample_design().rename(
        columns={
            "sample_id": "sample",
            "subject_id": "patient",
            "region": "major_labl",
            "condition": "patient_group",
        }
    )
    obs = obs.loc[obs.index.repeat(obs["n_cells"])].drop(columns="n_cells")

    class FileHandle:
        closed = False

        def close(self) -> None:
            self.closed = True

    class BackedFixture:
        def __init__(self, metadata: pd.DataFrame) -> None:
            self.isbacked = True
            self.n_obs = len(metadata)
            self.n_vars = 29_126
            self.file = FileHandle()
            self._obs = metadata

        @property
        def obs(self) -> pd.DataFrame:
            return self._obs

        @property
        def X(self) -> object:
            raise AssertionError("expression matrix must not be accessed")

    fixture = BackedFixture(obs)

    def fake_read(path: Path, *, backed: str) -> BackedFixture:
        assert path == source_path
        assert backed == "r"
        return fixture

    monkeypatch.setattr(module.ad, "read_h5ad", fake_read)
    design, lineage = module.read_backed_obs_sample_design(source_path)

    assert len(design) == 29
    assert fixture.file.closed
    assert lineage["input_kind"] == "h5ad_obs_backed_r"
    assert lineage["expression_matrix_accessed"] is False


def test_json_output_is_strict_and_contains_no_effect_or_pq(tmp_path: Path) -> None:
    result = module.audit_kuppe_repeated_measures(
        _realistic_sample_design(),
        lineage={"input_kind": "fixture", "expression_matrix_accessed": False},
    )
    output = tmp_path / "audit.json"
    module.write_audit(result, output)
    parsed = json.loads(output.read_text())
    serialized = json.dumps(parsed, sort_keys=True)

    assert parsed["guardrails"]["effect_estimated"] is False
    assert '"effect"' not in serialized
    assert '"p_value"' not in serialized
    assert '"q_value"' not in serialized


def test_compact_summary_retains_contrast_support_and_guardrails() -> None:
    result = module.audit_kuppe_repeated_measures(
        _realistic_sample_design(),
        lineage={"input_kind": "fixture", "expression_matrix_accessed": False},
    )
    summary = module.compact_summary(result)
    region = cast(dict[str, object], summary["region_model"])
    condition = cast(dict[str, object], summary["condition_region_adjusted_model"])

    assert summary["schema_version"] == module.SUMMARY_SCHEMA
    assert len(str(summary["full_audit_canonical_sha256"])) == 64
    assert region["status_counts"] == {
        "estimable_under_complete_score_coverage": 10
    }
    assert condition["status_counts"] == {"not_estimable": 3}
    limits = cast(dict[str, object], summary["interpretation_limits"])
    assert limits["effect_or_biological_claim"] is False
