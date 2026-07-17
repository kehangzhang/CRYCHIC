"""Build the frozen mouse LIANA-consensus x CellChatDB CITE-seq resource."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from benchmarks.adapters.common import canonical_digest, sha256_file, write_json
from crychic.resources import Species, load_cellchat_resource


def build(
    *,
    liana_mouse: Path,
    liana_manifest: Path,
    database_root: Path,
    cellchat_manifest: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "harmonized_lr.tsv"
    manifest_path = output_dir / "manifest.json"
    if (table_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"mouse H-common output exists: {output_dir}")

    source_manifest = json.loads(liana_manifest.read_text(encoding="utf-8"))
    expected_liana_sha = source_manifest["files"]["resource_tsv"]["sha256"]
    if sha256_file(liana_mouse) != expected_liana_sha:
        raise ValueError("LIANA mouse consensus checksum mismatch")
    liana = pd.read_csv(liana_mouse, sep="\t", dtype=str, keep_default_na=False)
    required = {"interaction_id", "ligand", "receptor"}
    missing = required.difference(liana.columns)
    if missing:
        raise ValueError(f"LIANA mouse resource lacks: {sorted(missing)}")

    cellchat = load_cellchat_resource(
        database_root,
        Species.MOUSE,
        manifest_path=cellchat_manifest,
    )
    simple = [
        interaction
        for interaction in cellchat.interactions
        if len(interaction.ligand_subunits) == 1
        and len(interaction.receptor_subunits) == 1
    ]
    cellchat_pair_counts = Counter(
        (item.ligand_subunits[0], item.receptor_subunits[0]) for item in simple
    )
    cellchat_by_pair = {
        (item.ligand_subunits[0], item.receptor_subunits[0]): item
        for item in simple
        if cellchat_pair_counts[(item.ligand_subunits[0], item.receptor_subunits[0])]
        == 1
    }
    liana_pair_counts = Counter(zip(liana["ligand"], liana["receptor"], strict=True))
    liana_by_pair = {
        (row.ligand, row.receptor): row
        for row in liana.itertuples(index=False)
        if liana_pair_counts[(row.ligand, row.receptor)] == 1
    }

    rows: list[dict[str, str]] = []
    for ligand, receptor in sorted(set(cellchat_by_pair).intersection(liana_by_pair)):
        cellchat_item = cellchat_by_pair[(ligand, receptor)]
        liana_item = liana_by_pair[(ligand, receptor)]
        rows.append(
            {
                "harmonized_interaction_id": canonical_digest(
                    {
                        "schema_version": "crychic-harmonized-lr-v1",
                        "species": "mouse",
                        "ligand": ligand,
                        "receptor": receptor,
                    },
                    prefix="harmonized_mouse_lr",
                ),
                "ligand": ligand,
                "receptor": receptor,
                "liana_source_interaction_id": str(liana_item.interaction_id),
                "cellchat_source_interaction_id": str(
                    cellchat_item.source_interaction_id
                ),
            }
        )
    table = pd.DataFrame.from_records(rows)
    if table.empty or table.duplicated(["ligand", "receptor"]).any():
        raise RuntimeError("mouse H-common did not produce unique LR pairs")
    table.to_csv(table_path, sep="\t", index=False)
    manifest: dict[str, object] = {
        "schema_version": "crychic-harmonized-lr-v1",
        "resource_id": "crychic_citeseq_mouse_liana_cellchat_simple_lr",
        "version": "2026-07-17",
        "species": "mouse",
        "gene_namespace": "MGI symbol",
        "license": "intersection-only; LIANA BSD-3-Clause and CellChat GPL-3",
        "citation": [
            "Dimitrov et al. LIANA, Nature Communications 2022.",
            cellchat.citation,
        ],
        "construction": {
            "rule": (
                "exact simple MGI-symbol ligand/receptor intersection; one source "
                "row per LIANA mouse consensus and CellChatDB mouse pair"
            ),
            "liana_source_rows": len(liana),
            "cellchat_source_interactions": len(cellchat.interactions),
            "cellchat_simple_unique_pairs": len(cellchat_by_pair),
            "retained_interactions": len(table),
            "unavailable_method": "CellPhoneDB mouse",
            "unavailable_reason": "first-party CellPhoneDB adapter supports human only",
        },
        "upstream_manifests": {
            "liana_mouse": {
                "filename": liana_manifest.name,
                "sha256": sha256_file(liana_manifest),
                "payload_filename": liana_mouse.name,
                "payload_sha256": expected_liana_sha,
            },
            "cellchat_mouse": {
                "filename": cellchat_manifest.name,
                "sha256": sha256_file(cellchat_manifest),
                "resource_digest": cellchat.manifest_digest,
            },
        },
        "payload": {
            "filename": table_path.name,
            "bytes": table_path.stat().st_size,
            "sha256": sha256_file(table_path),
            "rows": len(table),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liana-mouse", type=Path, required=True)
    parser.add_argument("--liana-manifest", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--cellchat-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(
        liana_mouse=args.liana_mouse.resolve(),
        liana_manifest=args.liana_manifest.resolve(),
        database_root=args.database_root.resolve(),
        cellchat_manifest=args.cellchat_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
