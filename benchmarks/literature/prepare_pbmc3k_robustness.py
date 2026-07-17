"""Prepare the 10x PBMC3k robustness input used by Dimitrov et al. 2022."""

from __future__ import annotations

import argparse
import importlib.metadata
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from benchmarks.openproblems.common import sha256_file, write_json


def _safe_extract(archive: Path, output: Path) -> None:
    root = output.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (output / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError("PBMC3k archive contains an unsafe path")
        handle.extractall(output, members=members)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def prepare(
    archive: Path,
    output_h5ad: Path,
    *,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    """Apply the paper's Seurat-tutorial QC with a Scanpy graph analogue."""

    if output_h5ad.exists() and not overwrite:
        raise FileExistsError(f"prepared PBMC3k input exists: {output_h5ad}")
    if not archive.is_file():
        raise FileNotFoundError(f"PBMC3k archive is missing: {archive}")
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crychic-pbmc3k-") as temporary:
        extracted = Path(temporary)
        _safe_extract(archive, extracted)
        matrix_dir = extracted / "filtered_gene_bc_matrices" / "hg19"
        data = sc.read_10x_mtx(
            matrix_dir,
            var_names="gene_symbols",
            make_unique=True,
            cache=False,
        )
    initial_shape = [int(data.n_obs), int(data.n_vars)]
    sc.pp.filter_cells(data, min_genes=200)
    sc.pp.filter_genes(data, min_cells=3)
    data.var["mt"] = data.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        data,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    keep = (
        data.obs["n_genes_by_counts"].gt(200)
        & data.obs["n_genes_by_counts"].lt(2500)
        & data.obs["pct_counts_mt"].lt(5)
    )
    data = data[keep].copy()
    data.layers["counts"] = sparse.csr_matrix(data.X, dtype=np.int32)
    sc.pp.normalize_total(data, target_sum=1.0e4)
    sc.pp.log1p(data)
    if None in data.layers:
        del data.layers[None]
    sc.pp.highly_variable_genes(
        data,
        flavor="seurat",
        n_top_genes=min(2000, data.n_vars),
        inplace=True,
    )
    work = data[:, data.var["highly_variable"]].copy()
    sc.pp.scale(work, max_value=10)
    n_comps = min(50, work.n_obs - 1, work.n_vars - 1)
    sc.tl.pca(work, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(
        work,
        n_neighbors=10,
        n_pcs=min(10, n_comps),
        random_state=seed,
    )
    sc.tl.leiden(
        work,
        resolution=0.5,
        random_state=seed,
        key_added="cluster",
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    labels = work.obs["cluster"].astype(str).map(lambda value: f"cluster.{value}")
    data.obs["cell_type"] = pd.Categorical(labels)
    data.obs["sample_id"] = "pbmc3k"
    data.obs["subject_id"] = "pbmc3k"
    data.obs["context"] = "baseline"
    data.uns["robustness_protocol"] = {
        "paper": "Dimitrov et al. Nature Communications 2022;13:3224",
        "qc": "200 < genes < 2500; mitochondrial percent < 5; genes in >=3 cells",
        "normalization": "library size 10000 followed by natural log1p",
        "clustering": "Scanpy Leiden graph analogue of Seurat resolution 0.5",
        "seed": int(seed),
    }
    data.write_h5ad(output_h5ad, compression="gzip")
    cluster_sizes = (
        data.obs["cell_type"].astype(str).value_counts().sort_index().to_dict()
    )
    manifest: dict[str, object] = {
        "schema_version": "crychic-dimitrov-pbmc3k-input-v1",
        "status": "complete",
        "source": {
            "filename": archive.name,
            "sha256": sha256_file(archive),
            "initial_shape": initial_shape,
        },
        "prepared": {
            "filename": output_h5ad.name,
            "sha256": sha256_file(output_h5ad),
            "shape": [int(data.n_obs), int(data.n_vars)],
            "clusters": len(cluster_sizes),
            "cluster_sizes": {key: int(value) for key, value in cluster_sizes.items()},
            "highly_variable_genes": int(data.var["highly_variable"].sum()),
            "counts_layer": "counts",
            "expression_matrix": "log1p normalized",
        },
        "parameters": {"seed": int(seed), "resolution": 0.5},
        "versions": {
            name: _version(name)
            for name in ("anndata", "scanpy", "numpy", "pandas", "scipy")
        },
    }
    write_json(output_h5ad.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(
        args.archive,
        args.output_h5ad,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
