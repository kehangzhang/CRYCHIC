"""Run current LIANA methods for the CytoSig cytokine benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import anndata as ad
import liana as li
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from benchmarks.openproblems.common import sha256_file, write_json


def _validate_counts(data: ad.AnnData) -> None:
    matrix = sparse.csr_matrix(data.X)
    if matrix.data.size and (
        not np.isfinite(matrix.data).all()
        or (matrix.data < 0).any()
        or not np.allclose(matrix.data, np.rint(matrix.data), rtol=0.0, atol=1e-8)
    ):
        raise ValueError("cytokine benchmark X must contain raw integer counts")


def _score_table(rank: pd.DataFrame, chat: pd.DataFrame) -> pd.DataFrame:
    rank = rank.copy()
    chat = chat.copy()
    rank_definitions = (
        (
            "cellphonedb_composite",
            "CellPhoneDB composite",
            "lr_means",
            pd.to_numeric(rank["cellphone_pvals"], errors="coerce").le(0.05),
        ),
        ("cellphonedb_pvalue", "CellPhoneDB p-value", "cellphone_pvals", None),
        ("connectome_specificity", "Connectome specificity", "scaled_weight", None),
        ("logfc_specificity", "logFC specificity", "lr_logfc", None),
        ("natmi_specificity", "NATMI specificity", "spec_weight", None),
        (
            "singlecellsignalr_lrscore",
            "SingleCellSignalR LRscore",
            "lrscore",
            pd.to_numeric(rank["lrscore"], errors="coerce").ge(0.5),
        ),
        (
            "liana_magnitude_consensus",
            "LIANA magnitude consensus",
            "magnitude_rank",
            None,
        ),
        (
            "liana_specificity_consensus",
            "LIANA specificity consensus",
            "specificity_rank",
            None,
        ),
    )
    rows: list[pd.DataFrame] = []
    for method_id, method_name, column, keep in rank_definitions:
        selected = rank if keep is None else rank.loc[keep]
        values = pd.to_numeric(selected[column], errors="coerce")
        if column in {"cellphone_pvals", "magnitude_rank", "specificity_rank"}:
            values = -values
        table = pd.DataFrame(
            {
                "method_id": method_id,
                "method_name": method_name,
                "source": selected["source"].astype(str),
                "target": selected["target"].astype(str),
                "ligand": selected["ligand_complex"].astype(str),
                "receptor": selected["receptor_complex"].astype(str),
                "score": values,
            }
        )
        rows.append(table.loc[table["score"].notna()])
    chat_definitions = (
        (
            "cellchat_composite",
            "CellChat composite",
            "lr_probs",
            pd.to_numeric(chat["cellchat_pvals"], errors="coerce").le(0.05),
        ),
        ("cellchat_pvalue", "CellChat p-value", "cellchat_pvals", None),
    )
    for method_id, method_name, column, keep in chat_definitions:
        selected = chat if keep is None else chat.loc[keep]
        values = pd.to_numeric(selected[column], errors="coerce")
        if column == "cellchat_pvals":
            values = -values
        table = pd.DataFrame(
            {
                "method_id": method_id,
                "method_name": method_name,
                "source": selected["source"].astype(str),
                "target": selected["target"].astype(str),
                "ligand": selected["ligand_complex"].astype(str),
                "receptor": selected["receptor_complex"].astype(str),
                "score": values,
            }
        )
        rows.append(table.loc[table["score"].notna()])
    result = pd.concat(rows, ignore_index=True)
    result["score_direction"] = "higher"
    result["status"] = "observed"
    return result.sort_values(
        ["method_id", "source", "target", "ligand", "receptor"],
        kind="stable",
        ignore_index=True,
    )


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
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"LIANA cytokine output exists: {manifest_path}")
    resource_path = resource_dir / "liana_consensus_human.parquet"
    resource_manifest_path = resource_dir / "resource_manifest.json"
    resource_manifest = json.loads(resource_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(resource_path) != resource_manifest["files"]["resource"]["sha256"]:
        raise ValueError("cytokine common resource checksum mismatch")
    resource = pd.read_parquet(resource_path).loc[:, ["ligand", "receptor"]]
    data = ad.read_h5ad(input_h5ad)
    removed_uns_keys = tuple(sorted(map(str, data.uns.keys())))
    data.uns.clear()
    _validate_counts(data)
    data.obs["label"] = data.obs["label"].astype("category")
    data.layers["counts"] = sparse.csr_matrix(data.X, dtype=np.float64)
    data.raw = None
    sc.pp.normalize_total(data, target_sum=1.0e4)
    sc.pp.log1p(data)
    started = time.perf_counter()
    rank = li.mt.rank_aggregate(
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
    chat = li.mt.cellchat(
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
    if not isinstance(rank, pd.DataFrame) or rank.empty:
        raise RuntimeError("LIANA rank aggregate returned no cytokine interactions")
    if not isinstance(chat, pd.DataFrame) or chat.empty:
        raise RuntimeError("LIANA CellChat returned no cytokine interactions")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    rank_path = raw_dir / "rank_aggregate.parquet"
    chat_path = raw_dir / "cellchat.parquet"
    score_path = output_dir / "standardized_lr_scores.parquet"
    rank.to_parquet(rank_path, index=False)
    chat.to_parquet(chat_path, index=False)
    scores = _score_table(rank, chat)
    scores.to_parquet(score_path, index=False)
    manifest: dict[str, object] = {
        "schema_version": "crychic-cytokine-liana-run-v1",
        "status": "complete",
        "input": {
            "filename": input_h5ad.name,
            "sha256": sha256_file(input_h5ad),
            "shape": list(data.shape),
            "truth_and_proxy_isolation": {
                "all_uns_removed_before_method_execution": True,
                "removed_key_names": list(removed_uns_keys),
            },
        },
        "resource": {
            "id": resource_manifest["resource_id"],
            "sha256": sha256_file(resource_path),
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
        "methods": sorted(scores["method_id"].unique()),
        "outputs": {
            path.name: {"sha256": sha256_file(path), "rows": len(table)}
            for path, table in (
                (rank_path, rank),
                (chat_path, chat),
                (score_path, scores),
            )
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
