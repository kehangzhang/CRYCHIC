"""Run reproducible LIANA consensus and CellChat literature benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    canonical_digest,
    git_metadata,
    load_harmonized_resource,
    prepare_output,
    python_environment,
    sha256_file,
    write_json,
)

SCHEMA_VERSION = "crychic-liana-cellchat-literature-run-v1"
RANK_OUTPUT = "rank_aggregate_raw.parquet"
CELLCHAT_OUTPUT = "cellchat_raw.parquet"
RANK_REQUIRED_COLUMNS = frozenset(
    {
        "source",
        "target",
        "ligand_complex",
        "receptor_complex",
        "lr_means",
        "cellphone_pvals",
        "expr_prod",
        "scaled_weight",
        "lr_logfc",
        "spec_weight",
        "lrscore",
        "specificity_rank",
        "magnitude_rank",
    }
)
CELLCHAT_REQUIRED_COLUMNS = frozenset(
    {
        "source",
        "target",
        "ligand_complex",
        "receptor_complex",
        "lr_probs",
        "cellchat_pvals",
    }
)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _resource_sha256(resource: pd.DataFrame) -> str:
    canonical = (
        resource.loc[:, ["ligand", "receptor"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["ligand", "receptor"], kind="stable", ignore_index=True)
    )
    return hashlib.sha256(
        canonical.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _validate_resource(resource: pd.DataFrame) -> pd.DataFrame:
    missing = {"ligand", "receptor"}.difference(resource.columns)
    if missing or resource.empty:
        raise ValueError(f"LIANA resource is invalid: missing={sorted(missing)}")
    result = resource.loc[:, ["ligand", "receptor"]].copy()
    if result.isna().any().any():
        raise ValueError("LIANA resource ligand/receptor values must be complete")
    result = result.astype(str).drop_duplicates(ignore_index=True)
    if (result["ligand"].str.len() == 0).any() or (
        result["receptor"].str.len() == 0
    ).any():
        raise ValueError("LIANA resource ligand/receptor values must not be empty")
    return result.sort_values(["ligand", "receptor"], kind="stable", ignore_index=True)


def _select_resource(
    li: Any,
    *,
    resource_mode: str,
    harmonized_resource: Path | None,
    harmonized_manifest: Path | None,
    custom_resource: Path | None,
    custom_resource_manifest: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if resource_mode == "native":
        if harmonized_resource is not None or harmonized_manifest is not None:
            raise ValueError("native mode cannot receive harmonized resource inputs")
        if (custom_resource is None) != (custom_resource_manifest is None):
            raise ValueError(
                "native custom resource requires both table and manifest"
            )
        if custom_resource is not None and custom_resource_manifest is not None:
            manifest = json.loads(custom_resource_manifest.read_text(encoding="utf-8"))
            files = manifest.get("files", {})
            if not isinstance(files, dict):
                raise ValueError("native custom resource manifest has invalid files")
            matches = [
                value
                for value in files.values()
                if isinstance(value, dict)
                and value.get("filename") == custom_resource.name
            ]
            if len(matches) != 1:
                raise ValueError("native custom resource is not pinned by manifest")
            expected = str(matches[0].get("sha256", ""))
            if sha256_file(custom_resource) != expected:
                raise ValueError("native custom resource checksum mismatch")
            resource = _validate_resource(pd.read_csv(custom_resource, sep="\t"))
            return resource, {
                "mode": "native",
                "id": str(manifest.get("resource_id", "custom_native")),
                "version": str(
                    manifest.get("task_version", manifest.get("version", "unknown"))
                ),
                "interactions": len(resource),
                "resource_sha256": _resource_sha256(resource),
                "payload_filename": custom_resource.name,
                "payload_sha256": expected,
                "manifest_filename": custom_resource_manifest.name,
                "manifest_sha256": sha256_file(custom_resource_manifest),
                "selection": "checksum-pinned custom native resource",
                "species": str(manifest.get("species", "mouse")),
                "gene_namespace": str(manifest.get("gene_namespace", "MGI symbol")),
            }
        resource = _validate_resource(li.rs.select_resource("consensus"))
        return resource, {
            "mode": "native",
            "id": "liana_consensus",
            "version": _distribution_version("liana"),
            "interactions": len(resource),
            "resource_sha256": _resource_sha256(resource),
            "selection": "liana.rs.select_resource('consensus')",
        }
    if resource_mode != "H-common":
        raise ValueError("resource_mode must be native or H-common")
    if custom_resource is not None or custom_resource_manifest is not None:
        raise ValueError("H-common cannot receive native custom resource inputs")
    if harmonized_resource is None or harmonized_manifest is None:
        raise ValueError("H-common requires resource table and manifest")
    table, frozen_manifest = load_harmonized_resource(
        harmonized_resource, harmonized_manifest
    )
    resource = _validate_resource(table)
    return resource, {
        "mode": "H-common",
        "id": str(frozen_manifest["resource_id"]),
        "version": str(frozen_manifest["version"]),
        "interactions": len(resource),
        "resource_sha256": _resource_sha256(resource),
        "payload_filename": harmonized_resource.name,
        "payload_sha256": sha256_file(harmonized_resource),
        "manifest_filename": harmonized_manifest.name,
        "manifest_sha256": sha256_file(harmonized_manifest),
    }


def _validate_input(adata: ad.AnnData, *, groupby: str) -> dict[str, object]:
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("input h5ad must contain cells and genes")
    if groupby not in adata.obs:
        raise ValueError(f"input h5ad is missing groupby column {groupby!r}")
    if adata.obs[groupby].isna().any():
        raise ValueError("groupby column must not contain missing labels")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("input h5ad cell and gene identifiers must be unique")
    labels = adata.obs[groupby].astype(str)
    if (labels.str.len() == 0).any():
        raise ValueError("groupby labels must not be empty")
    adata.obs = adata.obs.copy()
    adata.obs[groupby] = labels.astype("category")
    counts = labels.value_counts(sort=False).sort_index()
    return {
        "groups": len(counts),
        "group_sizes": {str(key): int(value) for key, value in counts.items()},
    }


def _validate_raw_output(
    result: object,
    *,
    method: str,
    required_columns: frozenset[str],
) -> pd.DataFrame:
    if result is None:
        raise RuntimeError(f"LIANA {method} returned None")
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"LIANA {method} did not return a DataFrame")
    missing = required_columns.difference(result.columns)
    if missing:
        raise RuntimeError(
            f"LIANA {method} output is missing columns: {sorted(missing)}"
        )
    if result.empty:
        raise RuntimeError(f"LIANA {method} returned no interactions")
    edge_keys = ["source", "target", "ligand_complex", "receptor_complex"]
    if result.loc[:, edge_keys].isna().any().any():
        raise RuntimeError(f"LIANA {method} output has missing edge identifiers")
    if result.duplicated(edge_keys).any():
        raise RuntimeError(f"LIANA {method} output has duplicate interaction keys")
    return result.reset_index(drop=True)


def _run_method(
    method: Any,
    adata: ad.AnnData,
    *,
    groupby: str,
    resource: pd.DataFrame,
    n_perms: int,
    seed: int,
    n_jobs: int,
    min_cells: int,
    expr_prop: float,
    layer: str | None,
) -> tuple[object, float]:
    np.random.seed(seed)
    started = time.perf_counter()
    result = method(
        adata.copy(),
        groupby=groupby,
        resource_name="consensus",
        resource=resource,
        expr_prop=expr_prop,
        min_cells=min_cells,
        use_raw=False,
        layer=layer,
        return_all_lrs=True,
        n_perms=n_perms,
        seed=seed,
        n_jobs=n_jobs,
        inplace=False,
        verbose=False,
    )
    return result, time.perf_counter() - started


def run_literature_benchmark(
    input_h5ad: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str,
    groupby: str = "cell_type",
    resource_mode: str = "native",
    harmonized_resource: str | Path | None = None,
    harmonized_manifest: str | Path | None = None,
    custom_resource: str | Path | None = None,
    custom_resource_manifest: str | Path | None = None,
    n_perms: int = 100,
    seed: int = 20260715,
    n_jobs: int = 1,
    min_cells: int = 10,
    expr_prop: float = 0.1,
    layer: str | None = None,
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run rank aggregation and CellChat on one frozen expression input."""

    import liana as li  # type: ignore[import-not-found]

    if not dataset_id or dataset_id != dataset_id.strip():
        raise ValueError("dataset_id must be a canonical non-empty identifier")
    if not groupby or groupby != groupby.strip():
        raise ValueError("groupby must be a canonical non-empty column name")
    for name, value in (
        ("n_perms", n_perms),
        ("n_jobs", n_jobs),
        ("min_cells", min_cells),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    if (
        isinstance(expr_prop, bool)
        or not np.isfinite(expr_prop)
        or not 0 <= expr_prop <= 1
    ):
        raise ValueError("expr_prop must lie in [0, 1]")
    input_path = Path(input_h5ad).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input h5ad is missing: {input_path}")
    harmonized_path = (
        None
        if harmonized_resource is None
        else Path(harmonized_resource).expanduser().resolve()
    )
    harmonized_manifest_path = (
        None
        if harmonized_manifest is None
        else Path(harmonized_manifest).expanduser().resolve()
    )
    custom_resource_path = (
        None
        if custom_resource is None
        else Path(custom_resource).expanduser().resolve()
    )
    custom_resource_manifest_path = (
        None
        if custom_resource_manifest is None
        else Path(custom_resource_manifest).expanduser().resolve()
    )
    output = prepare_output(output_dir, overwrite=overwrite)
    root = (
        Path(__file__).resolve().parents[3]
        if repo_root is None
        else Path(repo_root).expanduser().resolve()
    )
    started = time.perf_counter()
    adata: ad.AnnData | None = None
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "dataset_id": dataset_id,
        "failure": None,
        "output": None,
    }
    write_json(output / "manifest.json", manifest)
    try:
        adata = ad.read_h5ad(input_path)
        input_summary = _validate_input(adata, groupby=groupby)
        resource, resource_provenance = _select_resource(
            li,
            resource_mode=resource_mode,
            harmonized_resource=harmonized_path,
            harmonized_manifest=harmonized_manifest_path,
            custom_resource=custom_resource_path,
            custom_resource_manifest=custom_resource_manifest_path,
        )
        versions = {
            name: _distribution_version(name)
            for name in ("liana", "anndata", "scanpy", "pandas", "numpy", "pyarrow")
        }
        parameters: dict[str, object] = {
            "groupby": groupby,
            "n_perms": n_perms,
            "seed": seed,
            "n_jobs": n_jobs,
            "min_cells": min_cells,
            "expr_prop": expr_prop,
            "use_raw": False,
            "layer": layer,
            "return_all_lrs": True,
        }
        input_sha = sha256_file(input_path)
        run_id = canonical_digest(
            {
                "dataset_id": dataset_id,
                "input_sha256": input_sha,
                "liana_version": versions["liana"],
                "parameters": parameters,
                "resource": resource_provenance,
            },
            prefix="liana_literature_run",
        )
        manifest.update(
            {
                "run_id": run_id,
                "method": {
                    "framework": "LIANA",
                    "framework_version": versions["liana"],
                    "entrypoints": [
                        "liana.mt.rank_aggregate",
                        "liana.mt.cellchat",
                    ],
                    "license": "BSD-3-Clause",
                },
                "input": {
                    "filename": input_path.name,
                    "sha256": input_sha,
                    "bytes": input_path.stat().st_size,
                    "shape": [int(adata.n_obs), int(adata.n_vars)],
                    "groupby": groupby,
                    **input_summary,
                },
                "resource": resource_provenance,
                "parameters": parameters,
                "reproducibility": {
                    "same_seed_for_both_entrypoints": True,
                    "numpy_seed_reset_before_each_entrypoint": True,
                    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                    "recommended_n_jobs_for_exact_repeat": 1,
                },
                "environment": python_environment(
                    environment_name="liana_env",
                    packages=versions,
                    threads=n_jobs,
                ),
                "versions": versions,
                "code": git_metadata(root),
            }
        )
        write_json(output / "manifest.json", manifest)

        rank_raw, rank_seconds = _run_method(
            li.mt.rank_aggregate,
            adata,
            groupby=groupby,
            resource=resource,
            n_perms=n_perms,
            seed=seed,
            n_jobs=n_jobs,
            min_cells=min_cells,
            expr_prop=expr_prop,
            layer=layer,
        )
        rank = _validate_raw_output(
            rank_raw,
            method="rank_aggregate",
            required_columns=RANK_REQUIRED_COLUMNS,
        )
        cellchat_raw, cellchat_seconds = _run_method(
            li.mt.cellchat,
            adata,
            groupby=groupby,
            resource=resource,
            n_perms=n_perms,
            seed=seed,
            n_jobs=n_jobs,
            min_cells=min_cells,
            expr_prop=expr_prop,
            layer=layer,
        )
        cellchat = _validate_raw_output(
            cellchat_raw,
            method="cellchat",
            required_columns=CELLCHAT_REQUIRED_COLUMNS,
        )
        rank_path = output / RANK_OUTPUT
        cellchat_path = output / CELLCHAT_OUTPUT
        rank.to_parquet(rank_path, index=False)
        cellchat.to_parquet(cellchat_path, index=False)
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - started,
                "method_elapsed_seconds": {
                    "rank_aggregate": rank_seconds,
                    "cellchat": cellchat_seconds,
                },
                "output": {
                    "rank_aggregate": {
                        "filename": rank_path.name,
                        "rows": len(rank),
                        "columns": list(map(str, rank.columns)),
                        "sha256": sha256_file(rank_path),
                    },
                    "cellchat": {
                        "filename": cellchat_path.name,
                        "rows": len(cellchat),
                        "columns": list(map(str, cellchat.columns)),
                        "sha256": sha256_file(cellchat_path),
                    },
                },
            }
        )
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if adata is not None and adata.isbacked:
            adata.file.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--groupby", default="cell_type")
    parser.add_argument(
        "--resource-mode", choices=("native", "H-common"), default="native"
    )
    parser.add_argument("--harmonized-resource", type=Path)
    parser.add_argument("--harmonized-manifest", type=Path)
    parser.add_argument("--custom-resource", type=Path)
    parser.add_argument("--custom-resource-manifest", type=Path)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--expr-prop", type=float, default=0.1)
    parser.add_argument("--layer")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run_literature_benchmark(
        args.input_h5ad,
        args.output_dir,
        dataset_id=args.dataset_id,
        groupby=args.groupby,
        resource_mode=args.resource_mode,
        harmonized_resource=args.harmonized_resource,
        harmonized_manifest=args.harmonized_manifest,
        custom_resource=args.custom_resource,
        custom_resource_manifest=args.custom_resource_manifest,
        n_perms=args.n_perms,
        seed=args.seed,
        n_jobs=args.n_jobs,
        min_cells=args.min_cells,
        expr_prop=args.expr_prop,
        layer=args.layer,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CELLCHAT_OUTPUT",
    "RANK_OUTPUT",
    "SCHEMA_VERSION",
    "run_literature_benchmark",
]
