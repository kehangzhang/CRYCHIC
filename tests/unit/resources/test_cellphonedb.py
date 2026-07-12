from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from crychic.resources import load_cellphonedb_resource

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _write_cpdb_fixture(root: Path) -> None:
    directory = root / "cellphonedb" / "v5.0.0"
    directory.mkdir(parents=True)
    tables = {
        "genes.parquet": pd.DataFrame(
            {
                "protein_multidata_id": [1, 2, 3],
                "hgnc_symbol": ["LIG", "R1", "R2"],
            }
        ),
        "complex_composition.parquet": pd.DataFrame(
            {
                "complex_multidata_id": [10, 10],
                "protein_multidata_id": [2, 3],
                "total_protein": [2, 2],
            }
        ),
        "complex_expanded.parquet": pd.DataFrame(
            {"complex_multidata_id": [10], "name": ["R_COMPLEX"]}
        ),
        "interactions.parquet": pd.DataFrame(
            {
                "id_cp_interaction": ["CPI-TEST"],
                "multidata_1_id": [1],
                "multidata_2_id": [10],
                "name_1": ["P_LIG"],
                "name_2": ["R_COMPLEX"],
                "receptor_1": [False],
                "receptor_2": [True],
                "is_complex_1": [False],
                "is_complex_2": [True],
                "directionality": ["Ligand-Receptor"],
                "classification": ["Signaling by Test"],
                "source": ["PMID:1;uniprot"],
                "annotation_strategy": ["curated"],
                "curator": ["tester"],
            }
        ),
    }
    for name, table in tables.items():
        table.to_parquet(directory / name, index=False)
    files = [
        {
            "path": name,
            "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
        }
        for name in sorted(tables)
    ]
    (directory / "manifest.json").write_text(
        json.dumps({"files": files}), encoding="utf-8"
    )


def test_cellphonedb_maps_multidata_and_expands_complex(tmp_path: Path) -> None:
    _write_cpdb_fixture(tmp_path)
    bundle = load_cellphonedb_resource(tmp_path)

    assert len(bundle.interactions) == 1
    interaction = bundle.interactions[0]
    assert interaction.ligand_subunits == ("LIG",)
    assert interaction.receptor_subunits == ("R1", "R2")
    assert interaction.receptor_is_complex
    assert interaction.direction == "Ligand-Receptor"
    assert interaction.pathway == "Signaling by Test"
    assert interaction.evidence == ("PMID:1", "uniprot")
    assert bundle.mapping_report.unmapped_entities == ()
