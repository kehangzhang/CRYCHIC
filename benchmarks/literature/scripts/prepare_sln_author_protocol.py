#!/usr/bin/env python3
"""Prepare public SLN111/SLN208 CITE-seq objects with the author protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r_make_names(values: list[str]) -> list[str]:
    """Implement the subset of base R make.names used by the released script."""

    result: list[str] = []
    counts: dict[str, int] = {}
    reserved = {
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "in",
        "next",
        "break",
    }
    for value in values:
        name = re.sub(r"[^A-Za-z0-9._]", ".", value)
        if not name or not re.match(r"^[A-Za-z]|^\.(?![0-9])", name):
            name = f"X{name}"
        if name in reserved:
            name = f"{name}."
        occurrence = counts.get(name, 0)
        counts[name] = occurrence + 1
        result.append(name if occurrence == 0 else f"{name}.{occurrence}")
    return result


def author_adt_names(raw_names: list[str]) -> tuple[list[int], list[str]]:
    kept_indices: list[int] = []
    stripped: list[str] = []
    for index, raw in enumerate(raw_names):
        if "Ctrl" in raw or "Ligand" in raw:
            continue
        pieces = raw.split("_")
        if len(pieces) < 2:
            raise ValueError(f"SLN ADT name does not follow author encoding: {raw!r}")
        kept_indices.append(index)
        stripped.append(re.sub(r"\(.*", "", pieces[1]))
    return kept_indices, r_make_names(stripped)


def seurat_clr_featurewise(counts: np.ndarray) -> np.ndarray:
    if counts.ndim != 2 or counts.shape[0] == 0:
        raise ValueError("ADT counts must be a non-empty cells-by-proteins matrix")
    positive_logs = np.where(counts > 0, np.log1p(counts), 0.0)
    denominator = np.exp(positive_logs.sum(axis=0) / counts.shape[0])
    return np.log1p(counts / denominator).astype(np.float64, copy=False)


def _auxiliary_protein_columns(adata: ad.AnnData) -> list[str]:
    result: list[str] = []
    for key in adata.obsm.keys():
        if key == "protein_expression":
            continue
        value = adata.obsm[key]
        if isinstance(value, pd.DataFrame):
            result.extend(map(str, value.columns))
    return result


def prepare(input_h5ad: Path, output_dir: Path, dataset_id: str) -> dict[str, Any]:
    input_h5ad = input_h5ad.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(input_h5ad)
    required_obs = {"cell_types"}
    if not required_obs.issubset(adata.obs.columns):
        raise ValueError("SLN object lacks cell_types")
    if "protein_expression" not in adata.obsm or "protein_names" not in adata.uns:
        raise ValueError("SLN object lacks protein_expression or protein_names")
    proteins = adata.obsm["protein_expression"]
    if not isinstance(proteins, pd.DataFrame):
        raise TypeError("protein_expression must retain its encoded DataFrame columns")
    matrix_names = list(map(str, proteins.columns))
    uns_names = list(map(str, adata.uns["protein_names"]))
    if matrix_names != uns_names[: len(matrix_names)]:
        raise ValueError(
            "protein_expression columns do not match the protein_names prefix"
        )
    extra_names = uns_names[len(matrix_names) :]
    auxiliary_names = _auxiliary_protein_columns(adata)
    if extra_names != auxiliary_names:
        raise ValueError(
            "protein_names suffix is not exactly explained by auxiliary "
            "HTO/isotype matrices"
        )

    raw_cell_types = adata.obs["cell_types"].astype(str)
    excluded = raw_cell_types.str.contains(
        "quality", regex=False
    ) | raw_cell_types.str.contains("doublet", regex=False)
    keep = (~excluded).to_numpy()
    cleaned_types = (
        raw_cell_types.loc[~excluded]
        .str.replace("+", "", regex=False)
        .str.replace("/", " ", n=1, regex=False)
    )
    selected = adata[keep].copy()
    counts = (
        selected.X.tocsr()
        if sp.issparse(selected.X)
        else sp.csr_matrix(selected.X)
    )
    if counts.data.size and (
        np.any(counts.data < 0) or not np.array_equal(counts.data, np.rint(counts.data))
    ):
        raise ValueError("SLN RNA matrix is not non-negative integer counts")
    counts.data = np.rint(counts.data).astype(np.int32, copy=False)
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    if np.any(totals <= 0):
        raise ValueError("SLN retained cells include a zero RNA library")
    normalized = (
        counts.astype(np.float32)
        .multiply((10_000.0 / totals)[:, None])
        .tocsr()
    )
    normalized.data = np.log1p(normalized.data)

    protein_counts = proteins.loc[~excluded].to_numpy(dtype=np.float64, copy=True)
    if np.any(protein_counts < 0) or not np.array_equal(
        protein_counts, np.rint(protein_counts)
    ):
        raise ValueError("SLN protein_expression is not non-negative integer counts")
    kept_adt_indices, adt_names = author_adt_names(matrix_names)
    protein_counts = protein_counts[:, kept_adt_indices]
    adt_clr = seurat_clr_featurewise(protein_counts)

    cluster_ids = cleaned_types.to_numpy(dtype=str)
    cluster_levels = sorted(set(cluster_ids))
    mean_records: list[dict[str, Any]] = []
    for cluster_id in cluster_levels:
        mask = cluster_ids == cluster_id
        values = adt_clr[mask].mean(axis=0)
        for feature, mean_clr in zip(adt_names, values, strict=True):
            mean_records.append(
                {
                    "dataset_id": dataset_id,
                    "cluster_id": cluster_id,
                    "adt_feature": feature,
                    "mean_clr": float(mean_clr),
                    "n_cells": int(mask.sum()),
                }
            )
    means = pd.DataFrame.from_records(mean_records)
    means["z_across_clusters"] = means.groupby("adt_feature", observed=True)[
        "mean_clr"
    ].transform(lambda values: (values - values.mean()) / values.std(ddof=1))
    if means["z_across_clusters"].isna().any():
        raise ValueError("SLN has an undefined across-cluster ADT z score")

    obs = pd.DataFrame(
        {
            "sample_id": dataset_id,
            "subject_id": dataset_id,
            "cell_type": pd.Categorical(cluster_ids),
            "context": "CITE-seq",
            "cluster_id": pd.Categorical(cluster_ids),
        },
        index=pd.Index(selected.obs_names.astype(str), name="barcode"),
    )
    var = selected.var.copy()
    var.index = pd.Index(selected.var_names.astype(str), name="gene_symbol")
    prepared = ad.AnnData(X=normalized, obs=obs, var=var)
    prepared.layers["counts"] = counts
    prepared.uns["normalization"] = {
        "X": "per-cell total count 10000 followed by log1p",
        "counts_layer": "raw integer RNA counts",
    }
    prepared.uns["source_dataset_id"] = dataset_id
    output_h5ad = output_dir / f"{dataset_id}.h5ad"
    prepared.write_h5ad(output_h5ad, compression="gzip")

    cells = pd.DataFrame(
        {"dataset_id": dataset_id, "barcode": obs.index, "cluster_id": cluster_ids}
    )
    cells.to_csv(output_dir / "cell_clusters.tsv", sep="\t", index=False)
    means.to_csv(output_dir / "adt_cluster_means.tsv", sep="\t", index=False)
    metadata = pd.DataFrame(
        {
            "key": [
                "dataset_id",
                "input_h5ad",
                "input_sha256",
                "raw_cells",
                "author_qc_cells",
                "n_rna_features",
                "protein_expression_columns",
                "protein_names_entries",
                "auxiliary_name_entries",
                "adt_features_after_author_filter",
                "n_cell_types",
                "protein_column_recovery",
                "author_qc_rule",
                "adt_normalization",
                "protein_used_as_algorithm_input",
            ],
            "value": [
                dataset_id,
                str(input_h5ad),
                sha256_file(input_h5ad),
                adata.n_obs,
                selected.n_obs,
                selected.n_vars,
                len(matrix_names),
                len(uns_names),
                len(extra_names),
                len(adt_names),
                len(cluster_levels),
                "protein_expression DataFrame columns equal protein_names prefix; "
                "suffix equals auxiliary obsm columns",
                "exclude case-sensitive cell_types containing quality or doublet",
                "Seurat CLR margin=1 reproduced from retained-cell raw ADT counts",
                "false",
            ],
        }
    )
    metadata.to_csv(output_dir / "run_metadata.tsv", sep="\t", index=False)
    manifest = {
        "schema_version": "liana-citeseq-sln-preparation-v1",
        "generated_on": date.today().isoformat(),
        "dataset_id": dataset_id,
        "species": "mouse",
        "input": {
            "path": str(input_h5ad),
            "bytes": input_h5ad.stat().st_size,
            "sha256": sha256_file(input_h5ad),
            "shape": [int(adata.n_obs), int(adata.n_vars)],
        },
        "protein_column_recovery": {
            "matrix_columns": len(matrix_names),
            "uns_names": len(uns_names),
            "matrix_columns_equal_uns_prefix": True,
            "uns_suffix_equal_auxiliary_columns": True,
            "auxiliary_columns": extra_names,
        },
        "output": {
            "h5ad": output_h5ad.name,
            "h5ad_sha256": sha256_file(output_h5ad),
            "shape": [int(prepared.n_obs), int(prepared.n_vars)],
            "cell_types": len(cluster_levels),
            "adt_features": len(adt_names),
            "positive_labels_before_receptor_mapping": int(
                means["z_across_clusters"].ge(1.645).sum()
            ),
        },
    }
    (output_dir / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("dataset_id")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.input_h5ad, args.output_dir, args.dataset_id), indent=2
        )
    )


if __name__ == "__main__":
    main()
