"""Deterministic frozen-resource tables shared by external method runners."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.adapters.common import stable_interaction_id


def _import_crychic(repo_root: Path) -> Any:
    source = str(repo_root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import crychic

    return crychic


def _complex_id(parts: tuple[str, ...]) -> str:
    return "&".join(sorted(parts))


def _identifier(value: object) -> int:
    return int(float(str(value)))


def _collapse_source_rows(
    source: pd.DataFrame, *, resource_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resource_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, str]] = []
    for (ligand, receptor), group in source.groupby(
        ["ligand", "receptor"], sort=True, observed=True
    ):
        native_ids = tuple(sorted(group["source_interaction_id"].astype(str).unique()))
        canonical_id = stable_interaction_id(
            resource_id=resource_id,
            ligand=str(ligand),
            receptor=str(receptor),
            native_id=";".join(native_ids),
        )
        resource_rows.append(
            {
                "interaction_id": canonical_id,
                "native_interaction_id": ";".join(native_ids),
                "ligand": str(ligand),
                "receptor": str(receptor),
                "method_covered": True,
            }
        )
        source_rows.extend(
            {
                "source_interaction_id": native_id,
                "interaction_id": canonical_id,
                "ligand": str(ligand),
                "receptor": str(receptor),
            }
            for native_id in native_ids
        )
    return pd.DataFrame(resource_rows), pd.DataFrame(source_rows)


def _cellphonedb_native_resource(
    database_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Read the CPDB v5 archive without importing the Python-3.11 package."""
    resource_dir = database_root / "cellphonedb/v5.0.0"
    database_zip = resource_dir / "cellphonedb_07_12_2026_053629.zip"
    with zipfile.ZipFile(database_zip) as archive:
        protein = pd.read_csv(archive.open("protein_table.csv"))
        genes = pd.read_csv(archive.open("gene_table.csv"))
        composition = pd.read_csv(archive.open("complex_composition_table.csv"))
        multidata = pd.read_csv(archive.open("multidata_table.csv"))
        interactions = pd.read_csv(archive.open("interaction_table.csv"))

    protein_to_multidata = dict(
        zip(
            protein["id_protein"].astype(int),
            protein["protein_multidata_id"].astype(int),
            strict=True,
        )
    )
    symbol_by_multidata: dict[int, str] = {}
    for row in genes.dropna(subset=["hgnc_symbol", "protein_id"]).itertuples(
        index=False
    ):
        multidata_id = protein_to_multidata.get(_identifier(row.protein_id))
        if multidata_id is not None:
            symbol_by_multidata[multidata_id] = str(row.hgnc_symbol)
    complex_map: dict[int, tuple[str, ...]] = {}
    for complex_id, group in composition.groupby(
        "complex_multidata_id", sort=True, observed=True
    ):
        subunits = sorted(
            {
                symbol_by_multidata[_identifier(protein_id)]
                for protein_id in group["protein_multidata_id"]
                if _identifier(protein_id) in symbol_by_multidata
            }
        )
        if len(subunits) == len(group):
            complex_map[_identifier(complex_id)] = tuple(subunits)
    receptor_by_multidata = {
        _identifier(row.id_multidata): bool(row.receptor)
        for row in multidata.itertuples(index=False)
    }

    def resolve(identifier: object) -> tuple[str, ...] | None:
        key = _identifier(identifier)
        if key in complex_map:
            return complex_map[key]
        symbol = symbol_by_multidata.get(key)
        return None if symbol is None else (symbol,)

    rows: list[dict[str, str]] = []
    selected = interactions.loc[interactions["directionality"].eq("Ligand-Receptor")]
    for row in selected.itertuples(index=False):
        first = resolve(row.multidata_1_id)
        second = resolve(row.multidata_2_id)
        if first is None or second is None:
            continue
        swap = receptor_by_multidata.get(
            _identifier(row.multidata_1_id), False
        ) and not (receptor_by_multidata.get(_identifier(row.multidata_2_id), False))
        ligand, receptor = (second, first) if swap else (first, second)
        rows.append(
            {
                "source_interaction_id": str(row.id_cp_interaction),
                "ligand": _complex_id(ligand),
                "receptor": _complex_id(receptor),
            }
        )
    source = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    resource, mapping = _collapse_source_rows(source, resource_id="cellphonedb")
    native_manifest = json.loads(
        (resource_dir / "manifest.json").read_text(encoding="utf-8")
    )
    metadata = {
        "resource_id": "cellphonedb",
        "resource_version": "v5.0.0",
        "manifest_digest": str(native_manifest.get("adapter_version", "v5-readback")),
        "license": "MIT",
        "citation": "CellPhoneDB v5 resource",
    }
    return resource, mapping, metadata


def native_lr_resource(
    *,
    method: str,
    database_root: Path,
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load a native LR resource and collapse duplicate expanded gene pairs.

    The returned source map retains every upstream interaction ID so native
    output can be mapped back to a unique benchmark edge without silently
    duplicating identical expanded ligand/receptor pairs.
    """
    if method == "cellphonedb":
        return _cellphonedb_native_resource(database_root)
    crychic = _import_crychic(repo_root)
    if method == "cellchat":
        bundle = crychic.load_cellchat_resource(
            database_root,
            "human",
            manifest_path=repo_root / "resources/cellchatdb_human_v2.json",
        )
    else:
        raise ValueError(f"unknown native LR method: {method!r}")

    rows = []
    for interaction in bundle.interactions:
        if interaction.direction != "Ligand-Receptor":
            continue
        rows.append(
            {
                "source_interaction_id": interaction.source_interaction_id,
                "ligand": _complex_id(interaction.ligand_subunits),
                "receptor": _complex_id(interaction.receptor_subunits),
            }
        )
    source = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    if source.empty:
        raise ValueError(f"{method} native resource contains no LR interactions")

    resource, mapping = _collapse_source_rows(
        source, resource_id=str(bundle.resource_id)
    )
    metadata = {
        "resource_id": str(bundle.resource_id),
        "resource_version": str(bundle.version),
        "manifest_digest": str(bundle.manifest_digest),
        "license": str(bundle.license),
        "citation": str(bundle.citation),
    }
    return resource, mapping, metadata


def liana_native_resource(consensus: pd.DataFrame, *, version: str) -> pd.DataFrame:
    """Build the unique simple LR universe exposed by LIANA consensus."""
    required = {"ligand", "receptor"}
    missing = required.difference(consensus.columns)
    if missing:
        raise ValueError(f"LIANA resource lacks columns: {sorted(missing)}")
    pairs = (
        consensus.loc[:, ["ligand", "receptor"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(["ligand", "receptor"], kind="stable", ignore_index=True)
    )
    pairs["native_interaction_id"] = pairs["ligand"] + "|" + pairs["receptor"]
    pairs["interaction_id"] = [
        stable_interaction_id(
            resource_id="liana_consensus",
            ligand=ligand,
            receptor=receptor,
            native_id=native,
        )
        for ligand, receptor, native in pairs[
            ["ligand", "receptor", "native_interaction_id"]
        ].itertuples(index=False, name=None)
    ]
    pairs["method_covered"] = True
    return pairs[
        [
            "interaction_id",
            "native_interaction_id",
            "ligand",
            "receptor",
            "method_covered",
        ]
    ].assign(resource_version=version)


def nichenet_ligand_program_resource(prior: pd.DataFrame) -> pd.DataFrame:
    """Define one fixed target-program edge per ligand in the frozen prior."""
    ligands = sorted(prior["ligand"].dropna().astype(str).unique())
    rows = []
    for ligand in ligands:
        native_id = f"{ligand}|weighted_top_target_program"
        rows.append(
            {
                "interaction_id": stable_interaction_id(
                    resource_id="nichenet_v2_ligand_target_human",
                    ligand=ligand,
                    receptor="__target_program__",
                    native_id=native_id,
                ),
                "native_interaction_id": native_id,
                "ligand": ligand,
                "receptor": "__target_program__",
                "target": "__weighted_target_program__",
                "method_covered": True,
            }
        )
    return pd.DataFrame(rows)
