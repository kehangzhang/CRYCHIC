"""Run LIANA methods for the Open Problems v1.0.0 source-target task."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import time
from pathlib import Path
from types import ModuleType

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.openproblems.common import (
    aggregate_lr_scores,
    sha256_file,
    write_json,
    write_predictions,
)


def _validate_counts(data: ad.AnnData) -> None:
    matrix = sparse.csr_matrix(data.X)
    if matrix.data.size and (
        not np.isfinite(matrix.data).all()
        or (matrix.data < 0).any()
        or not np.allclose(matrix.data, np.rint(matrix.data), atol=1e-8, rtol=0.0)
    ):
        raise ValueError("Open Problems X must contain finite non-negative counts")


def _load_runtime() -> tuple[ModuleType, ModuleType]:
    try:
        return importlib.import_module("liana"), importlib.import_module("scanpy")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LIANA source-target benchmark requires liana and scanpy"
        ) from exc


def _label_axis(data: ad.AnnData) -> tuple[str, ...]:
    if "label" not in data.obs:
        raise ValueError("Open Problems input lacks obs['label']")
    labels = data.obs["label"]
    if isinstance(labels.dtype, pd.CategoricalDtype):
        return tuple(map(str, labels.cat.categories))
    return tuple(map(str, pd.unique(labels.astype(str))))


def _method_predictions(
    rank_result: pd.DataFrame,
    cellchat_result: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    resource_id: str,
) -> list[pd.DataFrame]:
    rank = rank_result.copy()
    rank["cellphonedb_filtered"] = np.where(
        pd.to_numeric(rank["cellphone_pvals"], errors="coerce").le(0.05),
        pd.to_numeric(rank["lr_means"], errors="coerce"),
        0.0,
    )
    rank["liana_magnitude"] = 1.0 - pd.to_numeric(
        rank["magnitude_rank"], errors="coerce"
    )
    rank["liana_specificity"] = 1.0 - pd.to_numeric(
        rank["specificity_rank"], errors="coerce"
    )
    chat = cellchat_result.copy()
    chat["cellchat_filtered"] = np.where(
        pd.to_numeric(chat["cellchat_pvals"], errors="coerce").le(0.05),
        pd.to_numeric(chat["lr_probs"], errors="coerce"),
        0.0,
    )
    definitions = (
        (rank, "cellphonedb_filtered", "cellphonedb_liana", "CellPhoneDB via LIANA"),
        (rank, "scaled_weight", "connectome", "Connectome"),
        (rank, "lr_logfc", "log2fc", "LIANA log2FC"),
        (rank, "spec_weight", "natmi", "NATMI"),
        (rank, "lrscore", "singlecellsignalr", "SingleCellSignalR"),
        (
            rank,
            "liana_magnitude",
            "liana_magnitude_rank_aggregate",
            "LIANA magnitude rank aggregate",
        ),
        (
            rank,
            "liana_specificity",
            "liana_specificity_rank_aggregate",
            "LIANA specificity rank aggregate",
        ),
        (chat, "cellchat_filtered", "cellchat_liana", "CellChat via LIANA"),
    )
    predictions: list[pd.DataFrame] = []
    for source, value_column, base_id, name in definitions:
        for aggregation in ("max", "sum"):
            predictions.append(
                aggregate_lr_scores(
                    source,
                    value_column=value_column,
                    method_id=f"{base_id}_{aggregation}",
                    method_name=f"{name} ({aggregation})",
                    method_scope=(
                        "steady_state_pooled_liana_consensus_mouse_common_resource"
                    ),
                    resource_id=resource_id,
                    aggregation=aggregation,
                    labels=labels,
                )
            )
    return predictions


def run(
    input_h5ad: Path,
    resource_dir: Path,
    output_dir: Path,
    *,
    n_perms: int,
    n_jobs: int,
    overwrite: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "liana_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"LIANA output already exists: {manifest_path}")
    resource_path = resource_dir / "liana_consensus_mouse.parquet"
    resource_manifest_path = resource_dir / "resource_manifest.json"
    resource_manifest = json.loads(resource_manifest_path.read_text(encoding="utf-8"))
    expected_resource_hash = resource_manifest["files"]["resource"]["sha256"]
    if sha256_file(resource_path) != expected_resource_hash:
        raise ValueError("LIANA mouse consensus resource checksum mismatch")
    resource = pd.read_parquet(resource_path).loc[:, ["ligand", "receptor"]]
    data = ad.read_h5ad(input_h5ad)
    labels = _label_axis(data)
    removed_uns_keys = tuple(sorted(map(str, data.uns.keys())))
    data.uns.clear()
    _validate_counts(data)
    data.obs["label"] = data.obs["label"].astype("category")
    data.layers["counts"] = sparse.csr_matrix(data.X, dtype=np.float64)
    data.raw = None
    li, sc = _load_runtime()
    sc.pp.normalize_total(data, target_sum=1.0e4)
    sc.pp.log1p(data)

    started = time.perf_counter()
    rank_result = li.mt.rank_aggregate(
        data,
        groupby="label",
        resource=resource,
        expr_prop=0.1,
        min_cells=5,
        return_all_lrs=False,
        use_raw=False,
        n_perms=n_perms,
        seed=20260717,
        n_jobs=n_jobs,
        inplace=False,
        verbose=True,
    )
    if not isinstance(rank_result, pd.DataFrame) or rank_result.empty:
        raise RuntimeError("LIANA rank aggregate returned no interactions")
    cellchat_result = li.mt.cellchat(
        data,
        groupby="label",
        resource=resource,
        expr_prop=0.1,
        min_cells=5,
        return_all_lrs=False,
        use_raw=False,
        n_perms=n_perms,
        seed=20260717,
        n_jobs=n_jobs,
        inplace=False,
        verbose=True,
    )
    if not isinstance(cellchat_result, pd.DataFrame) or cellchat_result.empty:
        raise RuntimeError("LIANA CellChat returned no interactions")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    rank_path = raw_dir / "liana_rank_aggregate_lr.parquet"
    chat_path = raw_dir / "liana_cellchat_lr.parquet"
    rank_result.to_parquet(rank_path, index=False)
    cellchat_result.to_parquet(chat_path, index=False)
    prediction_dir = output_dir / "predictions"
    prediction_paths = [
        write_predictions(prediction_dir, table)
        for table in _method_predictions(
            rank_result,
            cellchat_result,
            labels=labels,
            resource_id=str(resource_manifest["resource_id"]),
        )
    ]
    manifest: dict[str, object] = {
        "schema_version": "crychic-openproblems-liana-run-v1",
        "status": "complete",
        "task_version": "v1.0.0",
        "input": {
            "filename": input_h5ad.name,
            "sha256": sha256_file(input_h5ad),
            "shape": list(data.shape),
            "cell_types": len(labels),
            "truth_and_proxy_isolation": {
                "all_uns_removed_before_method_execution": True,
                "removed_key_names": list(removed_uns_keys),
            },
        },
        "resource": {
            "resource_id": resource_manifest["resource_id"],
            "filename": resource_path.name,
            "sha256": expected_resource_hash,
            "rows": len(resource),
        },
        "parameters": {
            "normalization": "total_1e4_log1p",
            "expr_prop": 0.1,
            "min_cells": 5,
            "n_perms": n_perms,
            "n_jobs": n_jobs,
            "seed": 20260717,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("liana", "anndata", "scanpy", "pandas", "numpy")
        },
        "elapsed_seconds": time.perf_counter() - started,
        "raw": {
            rank_path.name: sha256_file(rank_path),
            chat_path.name: sha256_file(chat_path),
        },
        "predictions": {
            path.name: sha256_file(path) for path in prediction_paths
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("resource_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.input_h5ad,
        args.resource_dir,
        args.output_dir,
        n_perms=args.n_perms,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
