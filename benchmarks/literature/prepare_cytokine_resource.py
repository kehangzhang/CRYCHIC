"""Freeze the current human LIANA consensus for cytokine validation."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import anndata as ad
import liana as li

from benchmarks.openproblems.common import sha256_file, write_json


def _subunits(value: object) -> tuple[str, ...]:
    return tuple(part for part in str(value).split("_") if part)


def run(input_h5ad: Path, output_dir: Path, *, overwrite: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "liana_consensus_human.parquet"
    manifest_path = output_dir / "resource_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"resource output exists: {manifest_path}")
    resource = (
        li.rs.select_resource("consensus")
        .loc[:, ["ligand", "receptor"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["ligand", "receptor"], kind="stable", ignore_index=True)
    )
    resource.insert(
        0,
        "interaction_id",
        [
            f"cytokine-v1::{ligand}::{receptor}"
            for ligand, receptor in resource.itertuples(index=False)
        ],
    )
    data = ad.read_h5ad(input_h5ad, backed="r")
    try:
        genes = set(map(str, data.var_names))
    finally:
        data.file.close()
    resource["input_available"] = [
        set((*_subunits(ligand), *_subunits(receptor))).issubset(genes)
        for ligand, receptor in resource.loc[
            :, ["ligand", "receptor"]
        ].itertuples(index=False)
    ]
    resource.to_parquet(table_path, index=False)
    tsv_path = output_dir / "liana_consensus_human.tsv"
    resource.to_csv(tsv_path, sep="\t", index=False)
    manifest: dict[str, object] = {
        "schema_version": "crychic-cytokine-common-resource-v1",
        "resource_id": "liana_consensus_human_current",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "source_resource": "LIANA consensus",
        "interactions": len(resource),
        "input_available_interactions": int(resource["input_available"].sum()),
        "input": {"filename": input_h5ad.name, "sha256": sha256_file(input_h5ad)},
        "files": {
            "resource": {
                "filename": table_path.name,
                "sha256": sha256_file(table_path),
            },
            "resource_tsv": {
                "filename": tsv_path.name,
                "sha256": sha256_file(tsv_path),
            },
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("liana", "anndata", "pandas")
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.input_h5ad, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
