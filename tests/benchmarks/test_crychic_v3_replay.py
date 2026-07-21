from __future__ import annotations

import json

import pytest
from benchmarks.adapters.common import sha256_file
from benchmarks.adapters.crychic.replay_v3_differential import (
    _validated_crossfit_semantic_source,
)


def test_replay_authenticates_crossfit_manifest_and_semantic_table(tmp_path) -> None:
    source = tmp_path / "run"
    crossfit = source / "crossfit_result"
    crossfit.mkdir(parents=True)
    semantic = crossfit / "semantic_availability_scores.parquet"
    semantic.write_bytes(b"authenticated semantic bytes")
    crossfit_manifest = {
        "status": "complete",
        "schema_version": "9.0.0",
        "crossfit_result_id": "crossfit-result-test",
        "tables": {
            "semantic_availability_scores": {
                "filename": semantic.name,
                "sha256": sha256_file(semantic),
            }
        },
    }
    crossfit_manifest_path = crossfit / "crossfit_manifest.json"
    crossfit_manifest_path.write_text(
        json.dumps(crossfit_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_manifest = {
        "crossfit_result": {
            "directory": crossfit.name,
            "manifest_sha256": sha256_file(crossfit_manifest_path),
            "schema_version": "9.0.0",
            "crossfit_result_id": "crossfit-result-test",
        }
    }

    _, returned_semantic, _, returned_semantic_sha = (
        _validated_crossfit_semantic_source(source, source_manifest)
    )
    assert returned_semantic == semantic
    assert returned_semantic_sha == sha256_file(semantic)

    semantic.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="semantic availability checksum"):
        _validated_crossfit_semantic_source(source, source_manifest)


def test_replay_rejects_unbound_crossfit_manifest(tmp_path) -> None:
    source = tmp_path / "run"
    crossfit = source / "crossfit_result"
    crossfit.mkdir(parents=True)
    (crossfit / "crossfit_manifest.json").write_text("{}\n", encoding="utf-8")
    source_manifest = {
        "crossfit_result": {
            "directory": crossfit.name,
            "manifest_sha256": "0" * 64,
            "schema_version": "9.0.0",
            "crossfit_result_id": "crossfit-result-test",
        }
    }

    with pytest.raises(ValueError, match="cross-fit manifest checksum"):
        _validated_crossfit_semantic_source(source, source_manifest)
