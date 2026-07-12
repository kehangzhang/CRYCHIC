"""Convert the official UCSC MS snRNA atlas to benchmark-ready AnnData."""

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
from scipy.io import mmread

EXPECTED = {
    "matrix.mtx.gz": {
        "size": 658_908_791,
        "md5_prefix": "cc1d9c9f49",
    },
    "barcodes.tsv.gz": {
        "size": 564_282,
        "md5_prefix": "4c5b12ae52",
    },
    "features.tsv.gz": {
        "size": 81_090,
        "md5_prefix": "761e6997d0",
    },
    "meta.tsv": {
        "size": 10_828_584,
        "md5_prefix": "74e189fcd1",
    },
}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(path: Path) -> dict[str, object]:
    expected = EXPECTED[path.name]
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = path.stat().st_size
    if observed_size != expected["size"]:
        raise ValueError(
            f"unexpected {path.name} size: {observed_size} != {expected['size']}"
        )
    md5 = _digest(path, "md5")
    if not md5.startswith(str(expected["md5_prefix"])):
        raise ValueError(
            f"unexpected {path.name} MD5 prefix: {md5[:10]} != "
            f"{expected['md5_prefix']}"
        )
    return {
        "size": observed_size,
        "md5": md5,
        "sha256": _digest(path, "sha256"),
    }


def _read_single_column(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        values = [line.rstrip("\r\n") for line in handle]
    if not values or any(not value for value in values):
        raise ValueError(f"{path.name} contains an empty identifier")
    if len(set(values)) != len(values):
        raise ValueError(f"{path.name} contains duplicate identifiers")
    return values


def _integer_counts(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix.tocoo(copy=False)
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("MS matrix contains non-finite or negative entries")
    chunk_size = 5_000_000
    for start in range(0, matrix.nnz, chunk_size):
        values = matrix.data[start : start + chunk_size]
        if not np.equal(values, np.rint(values)).all():
            raise ValueError("MS matrix contains non-integer values")
    if matrix.data.size and matrix.data.max() > np.iinfo(np.int32).max:
        raise ValueError("MS matrix count exceeds int32 range")
    matrix.data = matrix.data.astype(np.int32, copy=False)
    return matrix.tocsr().astype(np.int32, copy=False)


def prepare_ms_ucsc(source_dir: Path, output_path: Path) -> ad.AnnData:
    """Prepare the exact UCSC matrix/metadata release used by the benchmark."""
    paths = {name: source_dir / name for name in EXPECTED}
    source_manifest = {
        name: _validate_source(path) for name, path in paths.items()
    }

    barcodes = _read_single_column(paths["barcodes.tsv.gz"])
    genes = _read_single_column(paths["features.tsv.gz"])
    metadata = pd.read_csv(paths["meta.tsv"], sep="\t")
    required = {
        "cellId",
        "patient_id",
        "sample_id",
        "condition",
        "lesion_type",
        "batch_sn",
        "celltype",
        "subtype",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"UCSC metadata is missing {sorted(missing)}")
    if metadata["cellId"].astype(str).tolist() != barcodes:
        raise ValueError("UCSC matrix barcodes and metadata are not exactly aligned")

    gene_by_cell = _integer_counts(mmread(paths["matrix.mtx.gz"], spmatrix=True))
    expected_shape = (len(genes), len(barcodes))
    if gene_by_cell.shape != expected_shape:
        raise ValueError(
            f"MS matrix shape {gene_by_cell.shape} != expected {expected_shape}"
        )
    counts = gene_by_cell.T.tocsr().astype(np.int32, copy=False)

    obs_columns = [
        "patient_id",
        "sample_id",
        "condition",
        "lesion_type",
        "batch_sn",
        "celltype",
        "subtype",
    ]
    obs = metadata[obs_columns].copy()
    obs.index = pd.Index(barcodes, dtype=object, name="cell_id")
    obs["subject_id"] = obs["patient_id"].astype(str)
    obs["cell_type"] = obs["celltype"].astype(str)
    batch_values = pd.to_numeric(obs["batch_sn"], errors="raise")
    if not np.equal(batch_values, np.rint(batch_values)).all():
        raise ValueError("MS batch_sn contains a non-integer batch code")
    obs["batch"] = batch_values.astype(int).astype(str)
    for column in obs.columns:
        if isinstance(obs[column].dtype, pd.StringDtype):
            obs[column] = obs[column].astype(object)
    var = pd.DataFrame(
        {"gene_symbol": genes},
        index=pd.Index(genes, dtype=object, name="feature_id"),
    )

    library_size = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(library_size <= 0):
        raise ValueError("MS matrix contains an empty cell library")
    normalized = counts.astype(np.float32).copy()
    scale = np.divide(10_000.0, library_size, dtype=np.float32)
    normalized = (sparse.diags(scale) @ normalized).tocsr()
    normalized.data = np.log1p(normalized.data)

    result = ad.AnnData(X=normalized, obs=obs, var=var)
    result.layers["counts"] = counts
    result.uns["crychic_conversion"] = {
        "dataset_id": "UCSC_Lerma_Martin_MS_snRNA",
        "source_url": (
            "https://cells.ucsc.edu/ms-subcortical-lesions/snrna-atlas"
        ),
        "source_files": source_manifest,
        "subject_key": "subject_id",
        "sample_key": "sample_id",
        "context_keys": ["lesion_type"],
        "cell_type_key": "cell_type",
        "covariates": ["batch"],
        "design": "Ctrl/CA/CI; CA_vs_Ctrl primary; CI contrasts secondary",
        "counts_semantics": "raw non-negative integer counts",
        "x_semantics": "log1p library-normalized to 10000 counts per nucleus",
        "barcode_alignment": "exact UCSC order equality",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path, compression="gzip", compression_opts=4)
    summary = {
        "output": output_path.name,
        "shape": [result.n_obs, result.n_vars],
        "counts_nnz": int(counts.nnz),
        "subjects": int(obs["subject_id"].nunique()),
        "samples": int(obs["sample_id"].nunique()),
        "contexts": sorted(obs["lesion_type"].unique().tolist()),
        "cell_types": sorted(obs["cell_type"].unique().tolist()),
        "provenance": result.uns["crychic_conversion"],
    }
    output_path.with_suffix(".conversion.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    args = parser.parse_args()
    prepare_ms_ucsc(args.source_dir, args.output_h5ad)


if __name__ == "__main__":
    main()
