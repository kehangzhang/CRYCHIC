"""Build the checksum-pinned simple LR intersection used by external adapters."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from benchmarks.adapters.common import canonical_digest, sha256_file, write_json
from crychic import load_cellchat_resource, load_cellphonedb_resource
from crychic.resources import Interaction

SCHEMA_VERSION = "crychic-harmonized-lr-v1"


def build(
    *, database_root: Path, repo_root: Path, output_dir: Path, overwrite: bool
) -> dict[str, object]:
    """Build an exact, direction-preserving monomeric LR intersection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "harmonized_lr.tsv"
    covered_path = output_dir / "covered_lr.tsv"
    manifest_path = output_dir / "manifest.json"
    if (
        any(path.exists() for path in (table_path, covered_path, manifest_path))
        and not overwrite
    ):
        raise FileExistsError(
            f"harmonized output exists in {output_dir}; pass --overwrite to replace it"
        )
    cellchat_manifest = repo_root / "resources/cellchatdb_human_v2.json"
    cellphonedb_manifest = repo_root / "resources/cellphonedb_human_v5.0.0.json"
    cellchat = load_cellchat_resource(
        database_root,
        "human",
        manifest_path=cellchat_manifest,
    )
    cellphonedb = load_cellphonedb_resource(
        database_root,
        version="v5.0.0",
        manifest_path=cellphonedb_manifest,
    )
    by_cellchat: dict[tuple[tuple[str, ...], tuple[str, ...]], list[Interaction]] = (
        defaultdict(list)
    )
    by_cellphonedb: dict[tuple[tuple[str, ...], tuple[str, ...]], list[Interaction]] = (
        defaultdict(list)
    )
    for interaction in cellchat.interactions:
        by_cellchat[
            (interaction.ligand_subunits, interaction.receptor_subunits)
        ].append(interaction)
    for interaction in cellphonedb.interactions:
        by_cellphonedb[
            (interaction.ligand_subunits, interaction.receptor_subunits)
        ].append(interaction)
    covered_rows: list[dict[str, object]] = []
    for key in sorted(set(by_cellchat).union(by_cellphonedb)):
        left = by_cellchat[key]
        right = by_cellphonedb[key]
        ligand, receptor = key
        if len(ligand) != 1 or len(receptor) != 1:
            continue
        if len(left) > 1 or len(right) > 1:
            continue
        if right and right[0].direction != "Ligand-Receptor":
            continue
        ligand_gene, receptor_gene = ligand[0], receptor[0]
        harmonized_id = canonical_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "ligand": ligand_gene,
                "receptor": receptor_gene,
            },
            prefix="harmonized_lr",
        )
        covered_rows.append(
            {
                "harmonized_interaction_id": harmonized_id,
                "ligand": ligand_gene,
                "receptor": receptor_gene,
                "cellchat_source_interaction_id": (
                    left[0].source_interaction_id if left else ""
                ),
                "cellphonedb_source_interaction_id": (
                    right[0].source_interaction_id if right else ""
                ),
                "cellchat_covered": bool(left),
                "cellphonedb_covered": bool(right),
            }
        )
    covered = pd.DataFrame(covered_rows).sort_values(
        ["ligand", "receptor"], kind="stable", ignore_index=True
    )
    table = covered.loc[
        covered["cellchat_covered"] & covered["cellphonedb_covered"],
        [
            "harmonized_interaction_id",
            "ligand",
            "receptor",
            "cellchat_source_interaction_id",
            "cellphonedb_source_interaction_id",
        ],
    ].reset_index(drop=True)
    if table.empty or table.duplicated(["ligand", "receptor"]).any():
        raise RuntimeError("harmonized LR construction did not produce unique rows")
    if covered.empty or covered.duplicated(["ligand", "receptor"]).any():
        raise RuntimeError("covered LR construction did not produce unique rows")
    table.to_csv(table_path, sep="\t", index=False)
    covered.to_csv(covered_path, sep="\t", index=False)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "resource_id": "crychic_harmonized_cellchat_cellphonedb_simple_lr",
        "version": "2026-07-12",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "license": "intersection-only; upstream CellChat GPL-3 and CellPhoneDB MIT",
        "citation": [cellchat.citation, cellphonedb.citation],
        "construction": {
            "rule": (
                "exact expanded ligand/receptor HGNC tuple match; one source row per "
                "resource; one ligand gene; one receptor gene; CellPhoneDB "
                "directionality=Ligand-Receptor"
            ),
            "cellchat_source_interactions": len(cellchat.interactions),
            "cellphonedb_source_interactions": len(cellphonedb.interactions),
            "retained_interactions": len(table),
            "covered_union_interactions": len(covered),
        },
        "upstream_manifests": {
            "cellchat": {
                "filename": cellchat_manifest.name,
                "sha256": sha256_file(cellchat_manifest),
                "resource_digest": cellchat.manifest_digest,
            },
            "cellphonedb": {
                "filename": cellphonedb_manifest.name,
                "sha256": sha256_file(cellphonedb_manifest),
                "resource_digest": cellphonedb.manifest_digest,
            },
        },
        "payload": {
            "filename": table_path.name,
            "bytes": table_path.stat().st_size,
            "sha256": sha256_file(table_path),
            "rows": len(table),
        },
        "covered_payload": {
            "filename": covered_path.name,
            "bytes": covered_path.stat().st_size,
            "sha256": sha256_file(covered_path),
            "rows": len(covered),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", required=True, type=Path)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("resources") / "harmonized_simple_lr",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(
        database_root=args.database_root.resolve(),
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
