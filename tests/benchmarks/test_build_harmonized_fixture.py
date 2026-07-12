from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from benchmarks.adapters.common import sha256_file
from benchmarks.simulation.build_harmonized_fixture import build_fixture
from benchmarks.simulation.generate import DECOY_INTERACTIONS


def test_build_fixture_selects_active_and_decoys(tmp_path: Path) -> None:
    pairs = [("CXCL10", "CXCR3"), *DECOY_INTERACTIONS, ("A", "B")]
    source = pd.DataFrame(
        {
            "harmonized_interaction_id": [f"I{index}" for index in range(6)],
            "ligand": [pair[0] for pair in pairs],
            "receptor": [pair[1] for pair in pairs],
            "cellchat_source_interaction_id": [f"CC{index}" for index in range(6)],
            "cellphonedb_source_interaction_id": [
                f"CP{index}" for index in range(6)
            ],
        }
    )
    source_path = tmp_path / "source.tsv"
    source.to_csv(source_path, sep="\t", index=False)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "resource_id": "source",
                "license": "test",
                "citation": ["test"],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "fixture"
    manifest = build_fixture(source_path, source_manifest, output)

    fixture = pd.read_csv(output / "harmonized_lr.tsv", sep="\t")
    assert len(fixture) == 5
    assert manifest["payload"]["sha256"] == sha256_file(  # type: ignore[index]
        output / "harmonized_lr.tsv"
    )
