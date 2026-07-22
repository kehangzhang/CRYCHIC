from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from benchmarks.comprehensive.prepare_misc_benchmark import _common_resource


def test_common_resource_keeps_exact_directed_lr_intersection(tmp_path: Path) -> None:
    harmonized_path = tmp_path / "harmonized.tsv"
    connectome_path = tmp_path / "connectome.tsv"
    harmonized_manifest = tmp_path / "harmonized.json"
    connectome_manifest = tmp_path / "connectome.json"
    pd.DataFrame(
        {
            "harmonized_interaction_id": ["h1", "h2"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "cellchat_source_interaction_id": ["c1", "c2"],
        }
    ).to_csv(harmonized_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "harmonized_interaction_id": ["x1", "x2"],
            "ligand": ["L1", "L9"],
            "receptor": ["R1", "R9"],
        }
    ).to_csv(connectome_path, sep="\t", index=False)
    harmonized_manifest.write_text(
        json.dumps({"citation": ["source-a"]}), encoding="utf-8"
    )
    connectome_manifest.write_text(
        json.dumps({"citation": "source-b"}), encoding="utf-8"
    )

    table, manifest = _common_resource(
        harmonized_path,
        harmonized_manifest,
        connectome_path,
        connectome_manifest,
    )

    assert table[["ligand", "receptor"]].to_records(index=False).tolist() == [
        ("L1", "R1")
    ]
    assert table.loc[0, "harmonized_interaction_id"] == "h1"
    assert table.loc[0, "scseqcommdiff_source_interaction_id"] == "x1"
    assert manifest["construction"]["retained_interactions"] == 1
