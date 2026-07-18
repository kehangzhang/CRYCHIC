"""Export and checksum-pin the paper's ConnectomeDB2020 LR resource."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "crychic-connectomedb2020-resource-v1"
EXPECTED_ROWS = 2_293
REQUIRED_COLUMNS = (
    "ligand",
    "receptor",
    "interaction_label",
    "source",
    "pmid_support",
    "ligand_hgnc_id",
    "ligand_location",
    "receptor_hgnc_id",
    "hgnc_pair",
    "secondary_source",
)
PAPER_METHODS = (
    "cellchat",
    "crychic",
    "liana",
    "multinichenet",
    "scdiffcom",
    "scseqcommdiff",
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resource_table(
    table: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS
) -> pd.DataFrame:
    """Validate and C-locale-sort a directed ConnectomeDB2020 table."""
    if tuple(table.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            "ConnectomeDB2020 columns do not match the frozen schema: "
            f"{list(table.columns)}"
        )
    if len(table) != expected_rows:
        raise ValueError(
            f"ConnectomeDB2020 has {len(table)} rows; expected {expected_rows}"
        )
    result = table.copy(deep=True)
    for column in ("ligand", "receptor"):
        values = result[column]
        if values.isna().any():
            raise ValueError(f"ConnectomeDB2020 {column} contains missing values")
        canonical = values.astype(str).str.strip()
        if canonical.eq("").any() or not canonical.equals(values.astype(str)):
            raise ValueError(
                f"ConnectomeDB2020 {column} contains empty or non-canonical values"
            )
        result[column] = canonical
    if result.duplicated(["ligand", "receptor"]).any():
        raise ValueError(
            "ConnectomeDB2020 contains duplicate directed ligand-receptor pairs"
        )
    return result.sort_values(
        ["ligand", "receptor"], kind="stable", ignore_index=True
    )


def add_adapter_identifiers(table: pd.DataFrame) -> pd.DataFrame:
    """Add stable paper-panel IDs without altering the biological LR axis."""
    result = table.copy(deep=True)
    pair_ids = [
        "connectomedb2020_"
        + hashlib.sha256(f"{ligand}\0{receptor}".encode()).hexdigest()[:20]
        for ligand, receptor in result[["ligand", "receptor"]].itertuples(
            index=False, name=None
        )
    ]
    result.insert(0, "harmonized_interaction_id", pair_ids)
    for method in PAPER_METHODS:
        result[f"{method}_source_interaction_id"] = pair_ids
        result[f"{method}_covered"] = True
    return result


def _manifest_payload_sha256(manifest: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_payload_sha256"
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def export_connectomedb2020(
    source_rdata: Path,
    output_dir: Path,
    *,
    source_commit: str,
    rscript: str = "Rscript",
    retrieval_date: date | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export the scSeqComm v2 resource and write a checksum manifest."""
    if not source_rdata.is_file():
        raise FileNotFoundError(source_rdata)
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("source_commit must be a lowercase 40-character git SHA")
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "connectomedb2020.tsv"
    manifest_path = output_dir / "manifest.json"
    if not overwrite and (table_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"ConnectomeDB2020 output exists in {output_dir}; pass --overwrite"
        )

    exporter = Path(__file__).with_name("scripts") / "export_connectomedb2020.R"
    with tempfile.TemporaryDirectory(prefix="connectomedb2020-") as temp_dir:
        raw_path = Path(temp_dir) / "resource.tsv"
        subprocess.run(
            [rscript, str(exporter), str(source_rdata), str(raw_path)],
            check=True,
        )
        resource = add_adapter_identifiers(
            validate_resource_table(
                pd.read_csv(raw_path, sep="\t", dtype=str, keep_default_na=False)
            )
        )

    resource.to_csv(table_path, sep="\t", index=False, lineterminator="\n")
    retrieved = retrieval_date or date.today()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "resource_id": "ConnectomeDB2020_Hou_2020_human",
        "version": "2020-scSeqComm-2.0.0",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "directionality": "ligand_to_receptor",
        "rows": len(resource),
        "retrieval_date": retrieved.isoformat(),
        "source": {
            "repository": "https://gitlab.com/sysbiobig/scseqcomm",
            "commit": source_commit,
            "object": "LR_pairs_ConnectomeDB_2020",
            "filename": source_rdata.name,
            "bytes": source_rdata.stat().st_size,
            "sha256": sha256_file(source_rdata),
        },
        "payload": {
            "filename": table_path.name,
            "bytes": table_path.stat().st_size,
            "sha256": sha256_file(table_path),
        },
        "citation": (
            "Hou R, Denisenko E, Ong HT, et al. Predicting cell-to-cell "
            "communication networks using NATMI. Nat Commun 2020;11:5011. "
            "doi:10.1038/s41467-020-18873-z"
        ),
        "license": {
            "scseqcomm": "GPL-3",
            "upstream_resource": "verify before redistribution",
        },
        "construction": (
            "exact 2293-row scSeqComm v2.0.0 package object; directed pairs; "
            "no complex expansion or gene filtering; stable adapter IDs derived "
            "only from the directed ligand/receptor symbols"
        ),
    }
    manifest["manifest_payload_sha256"] = _manifest_payload_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_rdata", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--retrieval-date", type=date.fromisoformat)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = export_connectomedb2020(
        args.source_rdata,
        args.output_dir,
        source_commit=args.source_commit,
        rscript=args.rscript,
        retrieval_date=args.retrieval_date,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
