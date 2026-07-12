from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.adapters.common import (
    LONG_TABLE_SCHEMA,
    MANIFEST_SCHEMA,
    materialize_fixed_universe,
    sha256_file,
)
from benchmarks.validation.validate_external_bundle import validate_external_bundle


def _write_bundle(tmp_path: Path) -> Path:
    sample_metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "subject_id": ["p1", "p2"],
            "context_json": ['{"condition":"A"}', '{"condition":"B"}'],
        }
    )
    support = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "cell_type": ["A", "B", "A", "B"],
            "n_cells": [20, 20, 20, 20],
        }
    )
    resource = pd.DataFrame(
        {
            "interaction_id": ["i1"],
            "native_interaction_id": ["n1"],
            "ligand": ["L1"],
            "receptor": ["R1"],
            "method_covered": [True],
        }
    )
    observed = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "sender": ["A"],
            "receiver": ["B"],
            "interaction_id": ["i1"],
            "target": [pd.NA],
            "score": [0.8],
        }
    )
    table = materialize_fixed_universe(
        observed,
        sample_metadata=sample_metadata,
        support=support,
        resource=resource,
        dataset_id="toy",
        run_id="run",
        method_id="method",
        method_version="1",
        analysis_track="lr_stlr",
        resource_mode="H-common",
        resource_id="resource",
        resource_version="1",
        score_name="native",
        score_direction="higher",
        specificity_score_name=None,
        min_cells=10,
    )
    table_path = tmp_path / "interactions_long.parquet"
    table.to_parquet(table_path, index=False)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "run_id": "run",
        "dataset_id": "toy",
        "output": {
            "table": table_path.name,
            "rows": len(table),
            "schema_version": LONG_TABLE_SCHEMA,
            "sha256": sha256_file(table_path),
        },
        "sample_failures": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_validate_external_bundle_accepts_frozen_universe(tmp_path: Path) -> None:
    summary = validate_external_bundle(_write_bundle(tmp_path))

    assert summary["status"] == "valid"
    assert summary["rows"] == 8
    assert summary["run_ids"] == ["run"]
    assert summary["run_views"] == 1
    assert summary["samples"] == 2
    assert summary["universe_size"] == 4
    assert summary["status_counts"] == {"not_returned": 7, "ok": 1}


def test_validate_external_bundle_rejects_manifest_hash_mismatch(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256"):
        validate_external_bundle(bundle)


def test_validate_external_bundle_accepts_manifested_multi_run_views(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path)
    table_path = bundle / "interactions_long.parquet"
    table = pd.read_parquet(table_path)
    second = table.copy(deep=True)
    second["run_id"] = "run-2"
    combined = pd.concat([table, second], ignore_index=True)
    combined.to_parquet(table_path, index=False)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["rows"] = len(combined)
    manifest["output"]["sha256"] = sha256_file(table_path)
    manifest["source_result"] = {
        "score_views": [
            {"run_id": "run", "rows": len(table)},
            {"run_id": "run-2", "rows": len(second)},
        ]
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = validate_external_bundle(bundle)

    assert summary["rows"] == 16
    assert summary["run_ids"] == ["run", "run-2"]
    assert summary["run_views"] == 2
