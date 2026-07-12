from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from crychic.resources import ResourceIntegrityError, ResourceManifest


def _manifest(payload_path: str, checksum: str) -> dict[str, object]:
    return {
        "resource_id": "tiny_resource",
        "version": "1.0",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "source_url": "https://example.invalid/tiny",
        "license": "CC0-1.0",
        "citation": "Tiny synthetic resource.",
        "retrieved_at": "2026-07-12",
        "adapter_version": "test-v1",
        "payloads": [{"path": payload_path, "sha256": checksum}],
        "transformation_log": ["Created by a unit test."],
    }


def test_manifest_checksum_failure_is_hard_error(tmp_path: Path) -> None:
    payload = tmp_path / "tiny.tsv"
    payload.write_text("gene\tvalue\nA\t1\n", encoding="utf-8")
    checksum = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest("tiny.tsv", checksum)), encoding="utf-8"
    )
    manifest = ResourceManifest.from_json(manifest_path)
    assert manifest.verify(tmp_path) == ("tiny.tsv",)

    payload.write_text("gene\tvalue\nA\t2\n", encoding="utf-8")
    with pytest.raises(ResourceIntegrityError, match="checksum mismatch") as error:
        manifest.verify(tmp_path)
    assert error.value.details.code == "resource_checksum_mismatch"


def test_manifest_rejects_license_mismatch(tmp_path: Path) -> None:
    payload = tmp_path / "tiny.tsv"
    payload.write_text("x", encoding="utf-8")
    checksum = hashlib.sha256(payload.read_bytes()).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest("tiny.tsv", checksum)), encoding="utf-8")
    manifest = ResourceManifest.from_json(path)

    with pytest.raises(ResourceIntegrityError, match="license mismatch"):
        manifest.require(
            species="human", gene_namespace="HGNC symbol", license="GPL-3"
        )


def test_repository_resource_manifests_parse() -> None:
    resource_dir = Path(__file__).parents[3] / "resources"
    manifests = sorted(resource_dir.glob("*.json"))
    assert {path.name for path in manifests} == {
        "cellchatdb_human_v2.json",
        "cellchatdb_mouse_v2.json",
        "cellphonedb_human_v5.0.0.json",
        "nichenet_human_v2_2021.json",
    }
    for path in manifests:
        assert ResourceManifest.from_json(path).payloads
