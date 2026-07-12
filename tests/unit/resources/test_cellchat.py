from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from crychic.resources import (
    ResourceIntegrityError,
    Species,
    load_cellchat_resource,
)


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _cellchat_fixture(root: Path) -> Path:
    directory = root / "cellchat"
    directory.mkdir()
    (directory / "CELLCHATDB_VERSION.txt").write_text(
        "CellChat_package_version: 2.2.0.9001\n", encoding="ascii"
    )
    _write_csv(
        directory / "human_complex.csv",
        ["", "subunit_1", "subunit_2"],
        [["R_COMPLEX", "R1", "R2"]],
    )
    _write_csv(
        directory / "human_cofactor.csv",
        ["", "cofactor1", "cofactor2"],
        [["L agonists", "AGO1", "AGO2"]],
    )
    _write_csv(
        directory / "human_interaction.csv",
        [
            "",
            "interaction_name",
            "pathway_name",
            "ligand",
            "receptor",
            "agonist",
            "antagonist",
            "co_A_receptor",
            "co_I_receptor",
            "annotation",
            "evidence",
            "version",
        ],
        [
            [
                "LIG_R_COMPLEX",
                "LIG_R_COMPLEX",
                "TEST",
                "LIG",
                "R_COMPLEX",
                "L agonists",
                "",
                "",
                "",
                "Secreted Signaling",
                "PMID:1",
                "CellChatDB v2",
            ]
        ],
    )
    names = (
        "CELLCHATDB_VERSION.txt",
        "human_cofactor.csv",
        "human_complex.csv",
        "human_interaction.csv",
    )
    lines = [
        f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}"
        for name in names
    ]
    (directory / "CHECKSUMS.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )
    return directory


def test_cellchat_expands_complexes_and_preserves_annotations(tmp_path: Path) -> None:
    _cellchat_fixture(tmp_path)
    bundle = load_cellchat_resource(tmp_path, Species.HUMAN)

    assert len(bundle.interactions) == 1
    interaction = bundle.interactions[0]
    assert interaction.ligand_subunits == ("LIG",)
    assert interaction.receptor_subunits == ("R1", "R2")
    assert interaction.receptor_is_complex
    assert interaction.agonist_subunits == ("AGO1", "AGO2")
    assert interaction.direction == "Ligand-Receptor"
    assert interaction.pathway == "TEST"
    assert interaction.evidence == ("PMID:1",)


def test_cellchat_checksum_failure_precedes_parsing(tmp_path: Path) -> None:
    directory = _cellchat_fixture(tmp_path)
    with (directory / "human_complex.csv").open("a", encoding="utf-8") as handle:
        handle.write("corruption\n")

    with pytest.raises(ResourceIntegrityError, match="checksum mismatch"):
        load_cellchat_resource(tmp_path, Species.HUMAN)
