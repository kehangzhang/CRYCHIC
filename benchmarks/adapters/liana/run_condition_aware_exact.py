"""Exact-version worker for the LIANA+ 1.5.0 multi-sample benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

METHOD_VERSION = "1.5.0"
METHOD_TAG = "v1.5.0"
METHOD_COMMIT = "8f8f3d6617b190aaaf0d50fdff68aa16426abaf8"
DECOUPLER_VERSION = "1.8.0"
PYDESEQ2_VERSION = "0.5.0"
VIGNETTE_GIT_BLOB = "105c49abcab2a278f0f5e88ade5a57d0e85a7d48"
VIGNETTE_SHA256 = "56712d7c35ce9fb558f3adfa066dad904b0bbdbce728fe006696454f7eac621e"
VIGNETTE_URL = (
    "https://raw.githubusercontent.com/saezlab/liana-py/v1.5.0/"
    "docs/source/notebooks/targeted.ipynb"
)
EXPR_PROP = 0.1
ALPHA = 0.05
MIN_CELLS_PER_PSEUDOBULK = 10
MIN_COUNTS_PER_PSEUDOBULK = 10_000
MIN_GENE_COUNT = 5
MIN_GENE_TOTAL_COUNT = 10
MIN_REPLICATES_PER_CONDITION = 2
SCHEMA_VERSION = "crychic-liana-condition-aware-exact-worker-v1"
PACKAGE_NAMES = (
    "liana",
    "decoupler",
    "pydeseq2",
    "anndata",
    "numba",
    "numpy",
    "pandas",
    "scanpy",
    "scipy",
    "h5py",
)
COMPATIBILITY_BRIDGES = ("decoupler_1.8.0_sparse_pandas_boolean_mask_to_numpy",)


def _versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def _probe() -> None:
    import decoupler  # type: ignore[import-not-found]  # noqa: F401
    import liana  # type: ignore[import-not-found]  # noqa: F401
    import pydeseq2  # type: ignore[import-not-found]  # noqa: F401

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "executable": str(Path(sys.executable).resolve()),
                "platform": platform.platform(),
                "packages": _versions(),
            },
            sort_keys=True,
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_script_digest(path: Path, expected: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"script checksum mismatch for {path}: {observed} != {expected}"
        )
    return observed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_tsv(table: Any, path: Path) -> None:
    if path.suffix == ".gz":
        table.to_csv(
            path,
            sep="\t",
            index=False,
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        )
    else:
        table.to_csv(path, sep="\t", index=False)


def _load_counts(
    input_h5ad: Path,
    counts_layer: str,
    expected_shape: tuple[int, int],
) -> Any:
    import h5py  # type: ignore[import-untyped]
    import numpy as np
    from scipy import sparse

    with h5py.File(input_h5ad, "r") as handle:
        path = f"layers/{counts_layer}"
        if path not in handle:
            raise ValueError("raw counts layer is missing")
        group = handle[path]
        if group.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("raw counts layer must use CSR encoding")
        shape = tuple(int(value) for value in group.attrs["shape"])
        if shape != expected_shape:
            raise ValueError("raw counts shape disagrees with exported axes")
        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
    if data.size and (np.any(data < 0) or not np.issubdtype(data.dtype, np.integer)):
        raise ValueError("raw counts must be non-negative integers")
    matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
    if not matrix.has_sorted_indices:
        matrix.sort_indices()
    return matrix


@contextmanager
def _decoupler_sparse_index_compatibility() -> Any:
    """Bridge decoupler 1.8 pandas masks to identical NumPy CSR masks."""
    import pandas as pd
    from scipy.sparse import csr_matrix

    original = csr_matrix.__getitem__

    def compatible_getitem(matrix: Any, key: Any) -> Any:
        if isinstance(key, pd.Series):
            key = key.to_numpy()
        elif isinstance(key, tuple):
            key = tuple(
                item.to_numpy() if isinstance(item, pd.Series) else item for item in key
            )
        return original(matrix, key)

    csr_matrix.__getitem__ = compatible_getitem
    try:
        yield
    finally:
        csr_matrix.__getitem__ = original


def _select_condition_specific(lr_results: Any, target: str, reference: str) -> Any:
    import numpy as np
    import pandas as pd

    required = {"interaction_pvalue", "interaction_stat"}
    missing = required.difference(lr_results.columns)
    if missing:
        raise ValueError(f"LIANA+ results lack statistics: {sorted(missing)}")
    pvalues = pd.to_numeric(lr_results["interaction_pvalue"], errors="coerce")
    stats = pd.to_numeric(lr_results["interaction_stat"], errors="coerce")
    selected = lr_results.loc[
        np.isfinite(pvalues) & np.isfinite(stats) & pvalues.lt(ALPHA) & stats.ne(0)
    ].copy()
    selected["condition"] = np.where(
        pd.to_numeric(selected["interaction_stat"], errors="coerce").gt(0),
        target,
        reference,
    )
    return selected


def _coefficient_name(columns: Sequence[str], target: str, reference: str) -> str:
    expected_names = (
        f"condition_{target}_vs_{reference}",
        f"condition[T.{target}]",
    )
    for expected in expected_names:
        if expected in columns:
            return expected
    candidates = [
        value
        for value in columns
        if value.startswith("condition_") and value.endswith(f"_vs_{reference}")
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"cannot identify PyDESeq2 target coefficient; observed={list(columns)}"
        )
    return candidates[0]


def _validate_inference_cores(inference: Any, requested: int) -> int:
    observed = getattr(inference, "n_cpus", None)
    if not isinstance(observed, int) or observed != requested:
        raise RuntimeError(
            f"PyDESeq2 effective n_cpus {observed!r} != requested {requested}"
        )
    return observed


def _raise_on_cell_type_failures(analysis: Any) -> None:
    failures = analysis.loc[analysis["status"].eq("failed")]
    if failures.empty:
        return
    details = "; ".join(
        f"{row.cell_type}: {row.error}" for row in failures.itertuples(index=False)
    )
    raise RuntimeError(f"cell-type PyDESeq2 computation failed: {details}")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import anndata as ad
    import decoupler as dc
    import liana as li
    import numpy as np
    import pandas as pd
    import scanpy as sc  # type: ignore[import-not-found]
    from pydeseq2.dds import DeseqDataSet  # type: ignore[import-not-found]
    from pydeseq2.ds import DeseqStats  # type: ignore[import-not-found]

    started = time.time()
    np.random.seed(args.seed)
    exact_script = Path(__file__).resolve()
    exact_script_sha256 = _validate_script_digest(
        exact_script, args.exact_script_sha256
    )
    driver_script = args.driver_script.resolve()
    driver_script_sha256 = _validate_script_digest(driver_script, args.driver_sha256)
    versions = _versions()
    expected_versions = {
        "liana": METHOD_VERSION,
        "decoupler": DECOUPLER_VERSION,
        "pydeseq2": PYDESEQ2_VERSION,
    }
    mismatches = {
        key: {"expected": value, "observed": versions.get(key)}
        for key, value in expected_versions.items()
        if versions.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"paper-pinned versions mismatch: {mismatches}")

    output_dir = args.output_dir
    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str)
    genes = pd.read_csv(args.genes, sep="\t", dtype=str)["gene"].astype(str)
    resource = pd.read_csv(args.resource, sep="\t", dtype=str)
    required_metadata = {
        "cell_id",
        "replicate_id",
        "subject_id",
        "condition",
        "cell_type",
    }
    missing_metadata = required_metadata.difference(metadata.columns)
    if missing_metadata:
        raise ValueError(f"metadata columns are missing: {sorted(missing_metadata)}")
    if len(metadata) == 0 or len(genes) == 0:
        raise ValueError("exported h5ad axes cannot be empty")
    if metadata["cell_id"].duplicated().any() or genes.duplicated().any():
        raise ValueError("exported cell and gene identifiers must be unique")
    if set(metadata["condition"]) != {args.target, args.reference}:
        raise ValueError("exported conditions do not match the contrast")
    counts = _load_counts(
        args.input_h5ad,
        args.counts_layer,
        (len(metadata), len(genes)),
    )
    obs = (
        metadata.set_index("cell_id")
        .loc[:, ["replicate_id", "subject_id", "condition", "cell_type"]]
        .copy()
    )
    obs["replicate_id"] = pd.Categorical(obs["replicate_id"])
    obs["subject_id"] = pd.Categorical(obs["subject_id"])
    obs["condition"] = pd.Categorical(
        obs["condition"], categories=[args.reference, args.target]
    )
    obs["cell_type"] = pd.Categorical(obs["cell_type"])
    adata = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    with _decoupler_sparse_index_compatibility():
        pdata = dc.get_pseudobulk(
            adata,
            sample_col="replicate_id",
            groups_col="cell_type",
            mode="sum",
            min_cells=MIN_CELLS_PER_PSEUDOBULK,
            min_counts=MIN_COUNTS_PER_PSEUDOBULK,
        )
    if pdata.n_obs == 0 or pdata.n_vars == 0:
        raise RuntimeError("decoupler pseudobulk filtering removed all data")
    pdata.obs["condition"] = pd.Categorical(
        pdata.obs["condition"].astype(str), categories=[args.reference, args.target]
    )
    pdata.obs["cell_type"] = pd.Categorical(pdata.obs["cell_type"].astype(str))
    pdata.write_h5ad(output_dir / "pseudobulk_counts.h5ad", compression="gzip")
    support_columns = [
        column
        for column in (
            "replicate_id",
            "subject_id",
            "condition",
            "cell_type",
            "psbulk_n_cells",
            "psbulk_counts",
        )
        if column in pdata.obs.columns
    ]
    support = pdata.obs.loc[:, support_columns].reset_index(names="pseudobulk_id")
    _write_tsv(support, output_dir / "pseudobulk_support.tsv")

    gene_index = pd.Index(genes)
    resource_genes = pd.Index(
        sorted(set(resource["ligand"]).union(set(resource["receptor"])))
    )
    resource_positions = gene_index.get_indexer(resource_genes)
    present_mask = resource_positions >= 0
    present_resource_genes = resource_genes[present_mask]
    resource_positions = resource_positions[present_mask]
    target_mask = metadata["condition"].eq(args.target).to_numpy()
    target_counts = counts[target_mask, :][:, resource_positions].copy()
    target_obs = obs.iloc[np.flatnonzero(target_mask)].copy()
    del adata, counts
    gc.collect()

    source_cell_types = sorted(metadata["cell_type"].unique())
    support_counts = (
        support.groupby(["cell_type", "condition"], observed=True, sort=True)[
            "replicate_id"
        ]
        .nunique()
        .unstack(fill_value=0)
    )
    target_cell_counts = metadata.loc[
        metadata["condition"].eq(args.target), "cell_type"
    ].value_counts()
    analysis_rows: list[dict[str, Any]] = []
    dea_tables: list[Any] = []
    for cell_type in source_cell_types:
        n_reference = int(
            support_counts.loc[cell_type, args.reference]
            if cell_type in support_counts.index
            and args.reference in support_counts.columns
            else 0
        )
        n_target = int(
            support_counts.loc[cell_type, args.target]
            if cell_type in support_counts.index
            and args.target in support_counts.columns
            else 0
        )
        row: dict[str, Any] = {
            "cell_type": cell_type,
            "status": "not_estimable",
            "reason_code": "",
            "error": "",
            "n_reference_pseudobulks": n_reference,
            "n_target_pseudobulks": n_target,
            "n_target_cells": int(target_cell_counts.get(cell_type, 0)),
            "n_genes_tested": 0,
            "pydeseq_coefficient": "",
            "effective_n_cpus": 0,
        }
        if min(n_reference, n_target) < MIN_REPLICATES_PER_CONDITION:
            row["reason_code"] = "insufficient_pseudobulk_replicates_per_condition"
            analysis_rows.append(row)
            continue
        if row["n_target_cells"] < 5:
            row["reason_code"] = "insufficient_target_cells_for_df_to_lr"
            analysis_rows.append(row)
            continue
        try:
            ctdata = pdata[pdata.obs["cell_type"].astype(str).eq(cell_type)].copy()
            ctdata.obs["condition"] = pd.Categorical(
                ctdata.obs["condition"].astype(str),
                categories=[args.reference, args.target],
            )
            selected_genes = dc.filter_by_expr(
                ctdata,
                group="condition",
                min_count=MIN_GENE_COUNT,
                min_total_count=MIN_GENE_TOTAL_COUNT,
            )
            if len(selected_genes) == 0:
                row["reason_code"] = "no_genes_after_filter_by_expr"
                analysis_rows.append(row)
                continue
            ctdata = ctdata[:, selected_genes].copy()
            dds = DeseqDataSet(
                adata=ctdata,
                design_factors="condition",
                ref_level=["condition", args.reference],
                refit_cooks=True,
                n_cpus=args.cores,
                quiet=True,
            )
            effective_n_cpus = _validate_inference_cores(dds.inference, args.cores)
            dds.deseq2()
            stat_res = DeseqStats(
                dds,
                contrast=["condition", args.target, args.reference],
                inference=dds.inference,
                quiet=True,
            )
            if stat_res.inference is not dds.inference:
                raise RuntimeError("DeseqStats did not reuse the DDS inference object")
            _validate_inference_cores(stat_res.inference, args.cores)
            stat_res.summary()
            coefficient = _coefficient_name(
                list(dds.varm["LFC"].columns), args.target, args.reference
            )
            stat_res.lfc_shrink(coeff=coefficient)
            result = stat_res.results_df.copy()
            result.index.name = "gene"
            result = result.reset_index()
            result.insert(0, "cell_type", cell_type)
            dea_tables.append(result)
            row["status"] = "analyzed"
            row["n_genes_tested"] = len(result)
            row["pydeseq_coefficient"] = coefficient
            row["effective_n_cpus"] = effective_n_cpus
        except Exception as error:  # keep method-level support explicit
            row["status"] = "failed"
            row["reason_code"] = "pydeseq2_cell_type_failure"
            row["error"] = f"{type(error).__name__}: {error}"
        analysis_rows.append(row)
    analysis = pd.DataFrame.from_records(analysis_rows).sort_values(
        "cell_type", kind="stable", ignore_index=True
    )
    _write_tsv(analysis, output_dir / "analysis_cell_types.tsv")
    _raise_on_cell_type_failures(analysis)
    eligible = analysis.loc[analysis["status"].eq("analyzed"), "cell_type"].tolist()
    if not eligible or not dea_tables:
        raise RuntimeError("no cell type produced estimable PyDESeq2 results")
    dea_results = pd.concat(dea_tables, ignore_index=True)
    _write_tsv(dea_results, output_dir / "dea_results.tsv.gz")

    target_keep = target_obs["cell_type"].astype(str).isin(eligible).to_numpy()
    target_adata = ad.AnnData(
        X=target_counts[target_keep, :],
        obs=target_obs.iloc[np.flatnonzero(target_keep)].copy(),
        var=pd.DataFrame(index=present_resource_genes),
    )
    target_adata.obs["cell_type"] = pd.Categorical(
        target_adata.obs["cell_type"].astype(str), categories=sorted(eligible)
    )
    sc.pp.normalize_total(target_adata, target_sum=1e4)
    sc.pp.log1p(target_adata)
    dea_for_liana = dea_results.set_index("gene")
    lr_results = li.multi.df_to_lr(
        target_adata,
        dea_df=dea_for_liana,
        resource=resource.loc[:, ["ligand", "receptor"]],
        resource_name=None,
        expr_prop=EXPR_PROP,
        groupby="cell_type",
        stat_keys=["stat", "pvalue", "padj"],
        use_raw=False,
        complex_col="stat",
        verbose=True,
        return_all_lrs=False,
    )
    if lr_results.empty:
        raise RuntimeError("LIANA+ df_to_lr returned no interactions")
    required_lr = {
        "source",
        "target",
        "ligand_complex",
        "receptor_complex",
        "interaction_stat",
        "interaction_pvalue",
        "interaction_padj",
    }
    missing_lr = required_lr.difference(lr_results.columns)
    if missing_lr:
        raise RuntimeError(f"LIANA+ output columns are missing: {sorted(missing_lr)}")
    resource_lookup = resource.rename(
        columns={"ligand": "ligand_complex", "receptor": "receptor_complex"}
    )
    lr_results = lr_results.merge(
        resource_lookup.loc[
            :, ["interaction_id", "ligand_complex", "receptor_complex"]
        ],
        on=["ligand_complex", "receptor_complex"],
        how="left",
        validate="many_to_one",
    )
    if lr_results["interaction_id"].isna().any():
        raise RuntimeError("LIANA+ output could not be mapped to the frozen resource")
    lr_results = lr_results.sort_values(
        ["source", "target", "interaction_id"], kind="stable", ignore_index=True
    )
    if lr_results.duplicated(["source", "target", "interaction_id"]).any():
        raise RuntimeError("LIANA+ output contains duplicate directed LR keys")
    calls = _select_condition_specific(lr_results, args.target, args.reference)
    calls = calls.sort_values(
        ["condition", "source", "target", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )
    _write_tsv(lr_results, output_dir / "liana_lr_results.tsv.gz")
    _write_tsv(calls, output_dir / "condition_specific_calls.tsv.gz")
    freeze = "".join(f"{name}=={value}\n" for name, value in sorted(versions.items()))
    (output_dir / "environment_freeze.txt").write_text(freeze, encoding="utf-8")
    output_files = (
        "pseudobulk_counts.h5ad",
        "pseudobulk_support.tsv",
        "dea_results.tsv.gz",
        "liana_lr_results.tsv.gz",
        "condition_specific_calls.tsv.gz",
        "analysis_cell_types.tsv",
        "environment_freeze.txt",
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset_id": args.dataset_id,
        "code": {
            "driver": {
                "path": str(driver_script),
                "sha256": driver_script_sha256,
            },
            "exact_worker": {
                "path": str(exact_script),
                "sha256": exact_script_sha256,
            },
        },
        "method_provenance": {
            "version": METHOD_VERSION,
            "git_tag": METHOD_TAG,
            "git_commit": METHOD_COMMIT,
            "targeted_vignette": {
                "path": "docs/source/notebooks/targeted.ipynb",
                "git_blob": VIGNETTE_GIT_BLOB,
                "sha256": VIGNETTE_SHA256,
                "url": VIGNETTE_URL,
            },
        },
        "packages": versions,
        "compatibility_bridges": list(COMPATIBILITY_BRIDGES),
        "parameters": {
            "expr_prop": EXPR_PROP,
            "alpha": ALPHA,
            "selection_p_value": "interaction_pvalue",
            "additional_multiple_testing_correction": False,
            "min_cells_per_pseudobulk": MIN_CELLS_PER_PSEUDOBULK,
            "min_counts_per_pseudobulk": MIN_COUNTS_PER_PSEUDOBULK,
            "min_gene_count": MIN_GENE_COUNT,
            "min_gene_total_count": MIN_GENE_TOTAL_COUNT,
            "min_replicates_per_condition": MIN_REPLICATES_PER_CONDITION,
            "cores": args.cores,
            "seed": args.seed,
        },
        "parallelism": {
            "requested_cores": args.cores,
            "effective_n_cpus": sorted(
                {
                    int(value)
                    for value in analysis.loc[
                        analysis["status"].eq("analyzed"), "effective_n_cpus"
                    ]
                }
            ),
            "deseqstats_reuses_dds_inference": True,
            "all_analyzed_cell_types_match_requested": bool(
                analysis.loc[analysis["status"].eq("analyzed"), "effective_n_cpus"]
                .eq(args.cores)
                .all()
            ),
        },
        "results": {
            "pseudobulks": int(pdata.n_obs),
            "pseudobulk_genes": int(pdata.n_vars),
            "analyzed_cell_types": sorted(eligible),
            "dea_rows": len(dea_results),
            "lr_rows": len(lr_results),
            "condition_specific_calls": len(calls),
            "calls_by_condition": {
                str(key): int(value)
                for key, value in calls.groupby("condition", observed=True)
                .size()
                .items()
            },
        },
        "outputs": {
            name: {
                "sha256": _sha256(output_dir / name),
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in output_files
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_dir / "exact_analysis_manifest.json", manifest)
    return manifest


def main() -> None:
    if "--probe" in sys.argv:
        _probe()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--genes", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--target", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--cores", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--driver-script", required=True, type=Path)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--exact-script-sha256", required=True)
    args = parser.parse_args()
    result = _run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
