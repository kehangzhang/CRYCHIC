"""Prepare the mouse LIANA consensus resource used by all common-resource arms."""

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
    resource_path = output_dir / "liana_consensus_mouse.parquet"
    ortholog_path = output_dir / "hcop_human_mouse_min3.parquet"
    raw_hcop_path = output_dir / "hcop_human_mouse.txt.gz"
    manifest_path = output_dir / "resource_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"resource manifest already exists: {manifest_path}")

    orthologs = li.rs.get_hcop_orthologs(
        "mouse",
        filename=str(raw_hcop_path),
        min_evidence=3,
    )
    mapping = (
        orthologs.loc[:, ["human_symbol", "mouse_symbol", "evidence", "support"]]
        .dropna(subset=["human_symbol", "mouse_symbol"])
        .drop_duplicates()
        .sort_values(["human_symbol", "mouse_symbol"], kind="stable")
        .reset_index(drop=True)
    )
    mapping.to_parquet(ortholog_path, index=False)
    translate_map = mapping.loc[:, ["human_symbol", "mouse_symbol"]].rename(
        columns={"human_symbol": "source", "mouse_symbol": "target"}
    )
    human = li.rs.select_resource("consensus").loc[:, ["ligand", "receptor"]]
    mouse = li.rs.translate_resource(
        human,
        translate_map,
        columns=["ligand", "receptor"],
    )
    mouse = (
        mouse.loc[:, ["ligand", "receptor"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["ligand", "receptor"], kind="stable")
        .reset_index(drop=True)
    )
    mouse.insert(
        0,
        "interaction_id",
        [
            f"opv1::{ligand}::{receptor}"
            for ligand, receptor in mouse.itertuples(index=False)
        ],
    )
    data = ad.read_h5ad(input_h5ad, backed="r")
    try:
        genes = set(map(str, data.var_names))
    finally:
        data.file.close()
    mouse["input_available"] = [
        set((*_subunits(ligand), *_subunits(receptor))).issubset(genes)
        for ligand, receptor in mouse.loc[:, ["ligand", "receptor"]].itertuples(
            index=False
        )
    ]
    mouse.to_parquet(resource_path, index=False)
    mouse.to_csv(output_dir / "liana_consensus_mouse.tsv", sep="\t", index=False)
    manifest: dict[str, object] = {
        "schema_version": "crychic-openproblems-source-target-resource-v1",
        "task_version": "v1.0.0",
        "resource_id": "liana_consensus_mouse_hcop_min3",
        "source_resource": "LIANA consensus",
        "orthology": "HCOP human-to-mouse, minimum evidence 3",
        "human_interactions": len(human),
        "mouse_interactions": len(mouse),
        "input_available_interactions": int(mouse["input_available"].sum()),
        "ortholog_rows": len(mapping),
        "files": {
            "resource": {
                "filename": resource_path.name,
                "sha256": sha256_file(resource_path),
            },
            "resource_tsv": {
                "filename": "liana_consensus_mouse.tsv",
                "sha256": sha256_file(output_dir / "liana_consensus_mouse.tsv"),
            },
            "orthologs": {
                "filename": ortholog_path.name,
                "sha256": sha256_file(ortholog_path),
            },
            "raw_hcop": {
                "filename": raw_hcop_path.name,
                "sha256": sha256_file(raw_hcop_path),
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
