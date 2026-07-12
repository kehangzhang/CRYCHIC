"""Build the five-edge H-common fixture used by multimethod simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json
from benchmarks.simulation.generate import DECOY_INTERACTIONS

ACTIVE_INTERACTION = ("CXCL10", "CXCR3")


def build_fixture(
    source_table: Path, source_manifest: Path, output_dir: Path
) -> dict[str, object]:
    """Select one active and four expressed decoy pairs from frozen H-common."""
    source = pd.read_csv(source_table, sep="\t", dtype=str)
    expected = {ACTIVE_INTERACTION, *DECOY_INTERACTIONS}
    selected = source.loc[
        source[["ligand", "receptor"]]
        .apply(tuple, axis=1)
        .isin(expected)
    ].copy()
    observed = set(
        selected[["ligand", "receptor"]].itertuples(index=False, name=None)
    )
    if observed != expected or len(selected) != len(expected):
        raise ValueError(
            f"synthetic H-common fixture mismatch: observed={sorted(observed)}"
        )
    selected = selected.sort_values(
        ["ligand", "receptor"], kind="stable", ignore_index=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "harmonized_lr.tsv"
    selected.to_csv(table_path, sep="\t", index=False)
    source_metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest: dict[str, object] = {
        "schema_version": "crychic-harmonized-lr-v1-synthetic-fixture",
        "resource_id": "crychic_harmonized_simple_lr_synthetic_fixture",
        "version": "2026-07-12",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "license": source_metadata["license"],
        "citation": source_metadata["citation"],
        "construction": {
            "rule": "active CXCL10-CXCR3 plus four expressed no-change decoys",
            "source_resource_id": source_metadata["resource_id"],
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_payload_sha256": sha256_file(source_table),
            "retained_interactions": len(selected),
        },
        "payload": {
            "filename": table_path.name,
            "bytes": table_path.stat().st_size,
            "sha256": sha256_file(table_path),
            "rows": len(selected),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_table", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = build_fixture(
        args.source_table, args.source_manifest, args.output_dir
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
