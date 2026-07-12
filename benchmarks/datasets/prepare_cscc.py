"""Convert the paired Ji cSCC matrix to a benchmark-ready AnnData object."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

COUNTS_NAME = "GSE144236_cSCC_counts.txt.gz"
METADATA_NAME = "GSE144236_patient_metadata_new.txt.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_field(value: bytes) -> str:
    return value.strip().strip(b'"').decode("utf-8")


def _read_header(handle: gzip.GzipFile) -> list[str]:
    line = handle.readline()
    if not line:
        raise ValueError("cSCC counts file is empty")
    return [_decode_field(value) for value in line.rstrip(b"\r\n").split(b"\t")]


def _parse_counts(
    counts_path: Path,
    source_cell_ids: pd.Index,
    keep: np.ndarray,
) -> tuple[sparse.csr_matrix, list[str]]:
    """Stream the gene-by-cell text matrix without materializing its dense form."""
    kept_positions = np.flatnonzero(keep).astype(np.int32)
    index_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    indptr = [0]
    genes: list[str] = []

    with gzip.open(counts_path, "rb") as handle:
        observed_ids = _read_header(handle)
        if observed_ids != source_cell_ids.astype(str).tolist():
            mismatch = next(
                (
                    index
                    for index, (observed, expected) in enumerate(
                        zip(observed_ids, source_cell_ids.astype(str), strict=False)
                    )
                    if observed != expected
                ),
                None,
            )
            raise ValueError(
                "counts header and corrected metadata are not exactly aligned; "
                f"first mismatch index={mismatch}, counts={len(observed_ids)}, "
                f"metadata={len(source_cell_ids)}"
            )

        for expected_label in ("Patient", "Tissue: 0=Normal, 1=Tumor"):
            line = handle.readline()
            label, separator, _ = line.partition(b"\t")
            if not separator or _decode_field(label) != expected_label:
                raise ValueError(
                    f"missing expected counts annotation row {expected_label!r}"
                )

        for line_number, line in enumerate(handle, start=4):
            label, separator, payload = line.rstrip(b"\r\n").partition(b"\t")
            if not separator:
                raise ValueError(f"malformed counts row at line {line_number}")
            gene = _decode_field(label)
            if not gene:
                raise ValueError(f"empty gene identifier at line {line_number}")
            values = np.fromstring(payload, dtype=np.int32, sep="\t")
            if values.size != len(source_cell_ids):
                raise ValueError(
                    f"counts width mismatch at line {line_number}: "
                    f"{values.size} != {len(source_cell_ids)}"
                )
            if np.any(values < 0):
                raise ValueError(f"negative count at line {line_number}")
            retained = values[kept_positions]
            nonzero = np.flatnonzero(retained).astype(np.int32)
            index_chunks.append(nonzero)
            data_chunks.append(retained[nonzero])
            indptr.append(indptr[-1] + len(nonzero))
            genes.append(gene)

    if not genes:
        raise ValueError("cSCC counts file contains no gene rows")
    indices = np.concatenate(index_chunks).astype(np.int32, copy=False)
    data = np.concatenate(data_chunks).astype(np.int32, copy=False)
    gene_by_cell = sparse.csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int64)),
        shape=(len(genes), int(keep.sum())),
        dtype=np.int32,
    )
    return gene_by_cell, genes


def _collapse_gene_symbols(
    gene_by_cell: sparse.csr_matrix, genes: list[str]
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    codes, unique = pd.factorize(pd.Index(genes, dtype=object), sort=True)
    source_counts = np.bincount(codes, minlength=len(unique)).astype(np.int32)
    if len(unique) != len(genes):
        projection = sparse.csr_matrix(
            (
                np.ones(len(genes), dtype=np.int8),
                (codes.astype(np.int32), np.arange(len(genes), dtype=np.int32)),
            ),
            shape=(len(unique), len(genes)),
        )
        gene_by_cell = (projection @ gene_by_cell).tocsr().astype(np.int32)
    else:
        order = np.argsort(codes)
        gene_by_cell = gene_by_cell[order, :].tocsr()
    var = pd.DataFrame(
        {
            "gene_symbol": np.asarray(unique, dtype=object),
            "n_source_features": source_counts,
        },
        index=pd.Index(np.asarray(unique, dtype=object), name="gene_symbol"),
    )
    return gene_by_cell, var


def prepare_cscc(dataset_dir: Path, output_path: Path) -> ad.AnnData:
    """Prepare paired Normal/Tumor samples using level-1 cell annotations."""
    counts_path = dataset_dir / COUNTS_NAME
    metadata_path = dataset_dir / METADATA_NAME
    for path in (counts_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
    required = {
        "nCount_RNA",
        "nFeature_RNA",
        "patient",
        "tum.norm",
        "level1_celltype",
        "level2_celltype",
        "level3_celltype",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"corrected cSCC metadata is missing {sorted(missing)}")
    if metadata.index.has_duplicates:
        raise ValueError("corrected cSCC metadata contains duplicate cell IDs")
    if metadata[list(required)].isna().any().any():
        raise ValueError("corrected cSCC metadata contains missing required values")

    source_cell_ids = pd.Index(metadata.index.astype(str), dtype=object)
    keep = metadata["level1_celltype"].astype(str).ne("Multiplet").to_numpy()
    gene_by_cell, genes = _parse_counts(counts_path, source_cell_ids, keep)
    gene_by_cell, var = _collapse_gene_symbols(gene_by_cell, genes)
    counts = gene_by_cell.T.tocsr().astype(np.int32)

    retained = metadata.loc[keep].copy()
    observed_library = np.asarray(counts.sum(axis=1)).ravel()
    expected_library = retained["nCount_RNA"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_library, expected_library):
        raise ValueError("matrix library sizes do not match corrected metadata")
    observed_features = np.asarray(counts.getnnz(axis=1)).ravel()
    expected_features = retained["nFeature_RNA"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_features, expected_features):
        raise ValueError("matrix detected-gene counts do not match corrected metadata")

    patient = retained["patient"].astype(str)
    condition = retained["tum.norm"].astype(str)
    design = pd.crosstab(patient, condition)
    if set(design.columns) != {"Normal", "Tumor"} or len(design) != 10:
        raise ValueError(
            f"expected 10 paired Normal/Tumor patients, observed:\n{design}"
        )
    if (design == 0).any().any():
        raise ValueError("at least one cSCC patient lacks a Normal or Tumor sample")

    obs = retained.copy()
    obs.index = pd.Index(source_cell_ids[keep], dtype=object, name="cell_id")
    obs["sample_id"] = (patient + "_" + condition).to_numpy()
    obs["subject_id"] = patient.to_numpy()
    obs["condition"] = condition.to_numpy()
    obs["cell_type"] = retained["level1_celltype"].astype(str).to_numpy()
    for column in obs.columns:
        if isinstance(obs[column].dtype, pd.StringDtype):
            obs[column] = obs[column].astype(object)

    normalized = counts.astype(np.float32).copy()
    scale = np.divide(
        10_000.0,
        observed_library,
        out=np.zeros_like(observed_library, dtype=np.float32),
        where=observed_library > 0,
    )
    normalized = (sparse.diags(scale) @ normalized).tocsr()
    normalized.data = np.log1p(normalized.data)

    result = ad.AnnData(X=normalized, obs=obs, var=var)
    result.layers["counts"] = counts
    result.uns["crychic_conversion"] = {
        "dataset_id": "GSE144236_Ji_cSCC",
        "source_sha256": {
            COUNTS_NAME: _sha256(counts_path),
            METADATA_NAME: _sha256(metadata_path),
        },
        "metadata_source": "corrected GSE144236 patient metadata",
        "filters": ["level1_celltype != Multiplet"],
        "cell_type_level": "level1_celltype",
        "sample_id_definition": "patient + '_' + condition",
        "subject_key": "subject_id",
        "sample_key": "sample_id",
        "context_keys": ["condition"],
        "design": "10-patient paired Normal/Tumor",
        "counts_semantics": "raw non-negative integer counts",
        "x_semantics": "log1p library-normalized to 10000 counts per cell",
        "gene_mapping": "duplicate symbols summed; output symbols sorted",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(
        output_path,
        compression="gzip",
        compression_opts=4,
    )
    summary = {
        "output": output_path.name,
        "shape": [result.n_obs, result.n_vars],
        "source_cells": len(metadata),
        "excluded_multiplets": int((~keep).sum()),
        "subjects": int(obs["subject_id"].nunique()),
        "samples": int(obs["sample_id"].nunique()),
        "contexts": sorted(obs["condition"].unique().tolist()),
        "cell_types": sorted(obs["cell_type"].unique().tolist()),
        "counts_nnz": int(counts.nnz),
        "provenance": result.uns["crychic_conversion"],
    }
    output_path.with_suffix(".conversion.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    args = parser.parse_args()
    prepare_cscc(args.dataset_dir, args.output_h5ad)


if __name__ == "__main__":
    main()
