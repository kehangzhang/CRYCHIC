from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from crychic.resources import load_nichenet_target_prior

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def test_nichenet_long_parquet_builds_immutable_sparse_prior(tmp_path: Path) -> None:
    directory = tmp_path / "nichenet" / "v2_2021"
    directory.mkdir(parents=True)
    table = pd.DataFrame(
        {
            "ligand": ["L2", "L1", "L1"],
            "target": ["T1", "T2", "T1"],
            "weight": pd.Series([0.4, 0.2, 0.8], dtype="float32"),
            "rank": pd.Series([1, 2, 1], dtype="int16"),
        }
    )
    payload = directory / "ligand_target_top250.parquet"
    table.to_parquet(payload, index=False)
    manifest = {
        "resource_id": "nichenet_v2_ligand_target_human",
        "release": "21122021",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "source_url": "https://example.invalid/nichenet",
        "license": "CC-BY-4.0",
        "citation": "NicheNet test fixture.",
        "retrieved_at": "2026-07-12",
        "adapter_version": "test-v1",
        "files": [
            {
                "path": payload.name,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ],
        "transformation_log": ["Tiny deterministic fixture."],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    prior = load_nichenet_target_prior(tmp_path)

    assert prior.driver_ids == ("L1", "L2")
    assert prior.target_ids == ("T1", "T2")
    assert prior.shape == (2, 2)
    assert prior.nnz == 3
    assert [(link.target, link.rank) for link in prior.links_for_driver("L1")] == [
        ("T1", 1),
        ("T2", 2),
    ]
