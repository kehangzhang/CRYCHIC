"""Convert the paired Kang 2018 GEO matrices to a validated AnnData object."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import BinaryIO

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

CTRL_MATRIX = "GSM2560248_2.1.mtx.gz"
STIM_MATRIX = "GSM2560249_2.2.mtx.gz"
CTRL_BARCODES = "GSM2560248_barcodes.tsv.gz"
STIM_BARCODES = "GSM2560249_barcodes.tsv.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member(tar: tarfile.TarFile, name: str) -> BinaryIO:
    handle = tar.extractfile(name)
    if handle is None:
        raise ValueError(f"archive member is missing: {name}")
    return handle


def _read_matrix(tar: tarfile.TarFile, name: str) -> sparse.csr_matrix:
    with gzip.GzipFile(fileobj=_member(tar, name)) as handle:
        matrix = mmread(handle)
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    result = matrix.T.tocsr()
    if np.any(result.data < 0) or np.any(result.data != np.floor(result.data)):
        raise ValueError(f"{name} does not contain non-negative integer counts")
    return result.astype(np.int32)


def _read_barcodes(tar: tarfile.TarFile, name: str) -> pd.Index:
    with gzip.GzipFile(fileobj=_member(tar, name), mode="rb") as handle:
        values = [line.decode("utf-8").strip() for line in handle if line.strip()]
    return pd.Index(values, dtype="string")


def _validate_positional_barcodes(
    raw: pd.Index, metadata: pd.Index, *, condition: str
) -> None:
    if len(raw) != len(metadata):
        raise ValueError(
            f"{condition} barcode count mismatch: matrix={len(raw)}, "
            f"metadata={len(metadata)}"
        )
    mismatches = [
        (source, observed)
        for source, observed in zip(raw, metadata, strict=True)
        if observed != source and observed != f"{source}1"
    ]
    if mismatches:
        example = mismatches[0]
        raise ValueError(
            f"{condition} metadata is not positionally aligned; "
            f"first mismatch={example!r}"
        )


def _collapse_gene_symbols(
    matrix: sparse.csr_matrix, symbols: pd.Series
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    valid = symbols.notna() & symbols.astype("string").str.strip().ne("")
    clean = symbols.loc[valid].astype("string").str.strip()
    source = matrix[:, valid.to_numpy()]
    codes, unique = pd.factorize(clean, sort=True)
    projection = sparse.csr_matrix(
        (
            np.ones(len(codes), dtype=np.int8),
            (np.arange(len(codes), dtype=np.int32), codes.astype(np.int32)),
        ),
        shape=(len(codes), len(unique)),
    )
    collapsed = (source @ projection).tocsr().astype(np.int32)
    counts = pd.Series(clean).value_counts().reindex(unique).to_numpy(dtype=np.int32)
    var = pd.DataFrame(
        {"gene_symbol": np.asarray(unique, dtype=object), "n_source_features": counts},
        index=pd.Index(np.asarray(unique, dtype=object), name="gene_symbol"),
    )
    return collapsed, var


def prepare_kang(dataset_dir: Path, output_path: Path) -> ad.AnnData:
    """Prepare batch 2, retaining only labelled singlet cells."""
    archive = dataset_dir / "GSE96583_RAW.tar"
    metadata_path = dataset_dir / "GSE96583_batch2.total.tsne.df.tsv.gz"
    genes_path = dataset_dir / "GSE96583_batch2.genes.tsv.gz"
    for path in (archive, metadata_path, genes_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
    required = {"ind", "stim", "cell", "multiplets"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Kang metadata is missing columns: {sorted(missing)}")

    with tarfile.open(archive) as tar:
        ctrl = _read_matrix(tar, CTRL_MATRIX)
        stim = _read_matrix(tar, STIM_MATRIX)
        ctrl_barcodes = _read_barcodes(tar, CTRL_BARCODES)
        stim_barcodes = _read_barcodes(tar, STIM_BARCODES)

    n_ctrl, n_stim = ctrl.shape[0], stim.shape[0]
    if len(metadata) != n_ctrl + n_stim:
        raise ValueError(
            "metadata row count does not equal the two matrix column counts: "
            f"{len(metadata)} != {n_ctrl} + {n_stim}"
        )
    ctrl_meta = metadata.iloc[:n_ctrl].copy()
    stim_meta = metadata.iloc[n_ctrl:].copy()
    if set(ctrl_meta["stim"].astype(str)) != {"ctrl"}:
        raise ValueError("the first metadata block is not exclusively ctrl")
    if set(stim_meta["stim"].astype(str)) != {"stim"}:
        raise ValueError("the second metadata block is not exclusively stim")
    _validate_positional_barcodes(ctrl_barcodes, ctrl_meta.index, condition="ctrl")
    _validate_positional_barcodes(stim_barcodes, stim_meta.index, condition="stim")

    matrix = sparse.vstack([ctrl, stim], format="csr")
    metadata["raw_barcode"] = np.concatenate(
        [ctrl_barcodes.to_numpy(), stim_barcodes.to_numpy()]
    )
    keep = metadata["multiplets"].eq("singlet") & metadata["cell"].notna()
    metadata = metadata.loc[keep].copy()
    matrix = matrix[keep.to_numpy(), :]

    genes = pd.read_csv(
        genes_path,
        sep="\t",
        header=None,
        names=["ensembl_id", "gene_symbol"],
        dtype="string",
    )
    if len(genes) != matrix.shape[1]:
        raise ValueError(
            f"gene table/matrix mismatch: {len(genes)} != {matrix.shape[1]}"
        )
    matrix, var = _collapse_gene_symbols(matrix, genes["gene_symbol"])

    condition = metadata["stim"].astype("string")
    subject = metadata["ind"].astype("string")
    cell_ids = (condition + ":" + metadata["raw_barcode"]).astype(str).to_numpy()
    obs = pd.DataFrame(index=pd.Index(cell_ids, dtype=object))
    obs.index.name = "cell_id"
    obs["sample_id"] = (subject + ":" + condition).to_numpy()
    obs["subject_id"] = subject.to_numpy()
    obs["condition"] = condition.to_numpy()
    obs["cell_type"] = metadata["cell"].astype("string").to_numpy()
    obs["multiplets"] = metadata["multiplets"].astype("string").to_numpy()
    obs["technical_batch"] = "GSE96583_batch2"
    obs["raw_barcode"] = metadata["raw_barcode"].astype("string").to_numpy()
    for column in obs.columns:
        if isinstance(obs[column].dtype, pd.StringDtype):
            obs[column] = obs[column].astype(object)

    counts = matrix.tocsr()
    normalized = counts.astype(np.float32).copy()
    library = np.asarray(normalized.sum(axis=1)).ravel()
    scale = np.divide(
        10_000.0,
        library,
        out=np.zeros_like(library, dtype=np.float32),
        where=library > 0,
    )
    normalized = sparse.diags(scale) @ normalized
    normalized.data = np.log1p(normalized.data)

    adata = ad.AnnData(X=normalized.tocsr(), obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.uns["crychic_conversion"] = {
        "dataset_id": "GSE96583_Kang2018_batch2",
        "source_archive_sha256": _sha256(archive),
        "metadata_sha256": _sha256(metadata_path),
        "genes_sha256": _sha256(genes_path),
        "filters": ["multiplets == singlet", "cell_type is not missing"],
        "barcode_alignment": "condition-block positional alignment",
        "gene_mapping": "duplicate HGNC-like symbols summed",
        "counts_semantics": "raw non-negative integer UMI",
        "x_semantics": "log1p library-normalized to 10000 counts per cell",
        "subject_key": "subject_id",
        "sample_key": "sample_id",
        "context_keys": ["condition"],
    }

    expected = {"ctrl": 8, "stim": 8}
    observed = (
        obs[["sample_id", "subject_id", "condition"]]
        .drop_duplicates()
        .groupby("condition", observed=True)["subject_id"]
        .nunique()
        .to_dict()
    )
    if observed != expected:
        raise ValueError(f"expected 8 paired donors per condition, observed {observed}")
    if adata.n_obs != 24_673:
        raise ValueError(f"expected 24,673 labelled singlets, observed {adata.n_obs}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression="gzip")
    summary = {
        "output": output_path.name,
        "shape": [adata.n_obs, adata.n_vars],
        "subjects": int(obs["subject_id"].nunique()),
        "samples": int(obs["sample_id"].nunique()),
        "contexts": sorted(obs["condition"].unique().tolist()),
        "cell_types": sorted(obs["cell_type"].unique().tolist()),
        "counts_nnz": int(counts.nnz),
        "provenance": adata.uns["crychic_conversion"],
    }
    output_path.with_suffix(".conversion.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    args = parser.parse_args()
    prepare_kang(args.dataset_dir, args.output_h5ad)


if __name__ == "__main__":
    main()
