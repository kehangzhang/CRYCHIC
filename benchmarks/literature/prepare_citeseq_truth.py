"""Normalize author-protocol CITE-seq receptor labels for benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmarks.adapters.common import sha256_file

CANONICAL_COLUMNS = (
    "dataset",
    "receiver",
    "receptor",
    "is_positive",
    "truth_status",
)


def normalize_author_receptor_truth(source: pd.DataFrame) -> pd.DataFrame:
    """Map the author-style target/receptor label table to the shared schema."""

    required = {"dataset_id", "target_cell_type", "receptor_gene", "label"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"author receptor truth is missing: {sorted(missing)}")
    result = (
        source.loc[:, sorted(required)]
        .rename(
            columns={
                "dataset_id": "dataset",
                "target_cell_type": "receiver",
                "receptor_gene": "receptor",
                "label": "is_positive",
            }
        )
        .loc[:, ["dataset", "receiver", "receptor", "is_positive"]]
    )
    for column in ("dataset", "receiver", "receptor"):
        result[column] = result[column].astype(str).str.strip()
    # Upper-case is the evaluation namespace for both HGNC and MGI symbols.
    # Original display case remains available in the author truth artifact.
    result["receptor"] = result["receptor"].str.upper()
    result["is_positive"] = pd.to_numeric(result["is_positive"], errors="raise").astype(
        int
    )
    if not set(result["is_positive"]).issubset({0, 1}):
        raise ValueError("author receptor truth labels must be binary 0/1")
    if result.loc[:, ["dataset", "receiver", "receptor"]].eq("").any().any():
        raise ValueError("author receptor truth identifiers must not be empty")
    if result.duplicated(["dataset", "receiver", "receptor"]).any():
        raise ValueError("author receptor truth contains duplicate canonical keys")
    result["truth_status"] = "observed"
    return result.loc[:, list(CANONICAL_COLUMNS)].sort_values(
        ["dataset", "receiver", "receptor"], kind="stable", ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("author_truth", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    result = normalize_author_receptor_truth(pd.read_csv(args.author_truth, sep="\t"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "crychic-citeseq-receptor-truth-v1",
            "source": str(args.author_truth.resolve()),
            "source_sha256": sha256_file(args.author_truth),
            "output": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
            "rows": len(result),
            "positive_rows": int(result["is_positive"].sum()),
            "datasets": sorted(result["dataset"].unique()),
            "threshold_source": "Dimitrov protocol author-style table; z >= 1.645",
        }
        args.manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()


__all__ = ["CANONICAL_COLUMNS", "normalize_author_receptor_truth"]
