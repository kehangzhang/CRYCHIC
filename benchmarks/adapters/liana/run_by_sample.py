"""Run LIANA rank aggregation independently for each biological sample."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    begin_manifest,
    cell_type_support,
    fail_manifest,
    finalize_manifest,
    load_harmonized_resource,
    materialize_fixed_universe,
    prepare_output,
    python_environment,
    sha256_file,
    validate_prepared_input,
)
from benchmarks.adapters.resource_tables import liana_native_resource

METHOD_ID = "liana_rank_aggregate"


def _empty_observed() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id",
            "sender",
            "receiver",
            "interaction_id",
            "target",
            "score",
            "specificity_score",
            "within_dataset_p_value",
            "within_dataset_p_value_semantics",
        ]
    )


def _normalize(
    result: pd.DataFrame,
    *,
    pair_map: pd.DataFrame,
) -> pd.DataFrame:
    candidates = ("sample", "sample_id", "_adapter_sample")
    sample_column = next(
        (column for column in candidates if column in result.columns), None
    )
    if sample_column is None:
        raise RuntimeError("LIANA output is missing its sample identifier column")
    result = result.rename(
        columns={sample_column: "sample_id", "source": "sender", "target": "receiver"}
    ).copy()
    required = {
        "sample_id",
        "sender",
        "receiver",
        "ligand_complex",
        "receptor_complex",
        "magnitude_rank",
        "specificity_rank",
    }
    missing = required.difference(result.columns)
    if missing:
        raise RuntimeError(f"LIANA output is missing columns: {sorted(missing)}")
    result["ligand"] = result["ligand_complex"].astype(str)
    result["receptor"] = result["receptor_complex"].astype(str)
    result = result.merge(
        pair_map[["ligand", "receptor", "interaction_id"]],
        on=["ligand", "receptor"],
        how="inner",
        validate="many_to_one",
    )
    result["score"] = pd.to_numeric(result["magnitude_rank"], errors="coerce")
    result["specificity_score"] = pd.to_numeric(
        result["specificity_rank"], errors="coerce"
    )
    if "cellphone_pvals" in result:
        result["within_dataset_p_value"] = pd.to_numeric(
            result["cellphone_pvals"], errors="coerce"
        )
        semantics = (
            "LIANA component cell-label specificity; not between-condition inference"
        )
    else:
        result["within_dataset_p_value"] = np.nan
        semantics = "not_emitted"
    result = result.loc[result["score"].notna()].copy()
    if result.empty:
        return _empty_observed()
    result = result.groupby(
        ["sample_id", "sender", "receiver", "interaction_id"],
        sort=True,
        observed=True,
        as_index=False,
    ).agg(
        score=("score", "min"),
        specificity_score=("specificity_score", "min"),
        within_dataset_p_value=("within_dataset_p_value", "min"),
    )
    result["target"] = pd.NA
    result["within_dataset_p_value_semantics"] = semantics
    return result


def run(
    input_h5ad: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    sample_key: str,
    subject_key: str,
    cell_type_key: str,
    context_keys: tuple[str, ...],
    resource_mode: str,
    harmonized_resource: Path | None,
    harmonized_manifest: Path | None,
    n_perms: int,
    n_jobs: int,
    min_cells: int,
    expr_prop: float,
    seed: int,
    layer: str | None,
    overwrite: bool,
) -> dict[str, object]:
    """Run LIANA using either consensus or a frozen harmonized LR table."""
    import liana as li

    output_dir = prepare_output(output_dir, overwrite=overwrite)
    repo_root = Path(__file__).resolve().parents[3]
    adata = ad.read_h5ad(input_h5ad)
    sample_metadata = validate_prepared_input(
        adata,
        sample_key=sample_key,
        subject_key=subject_key,
        cell_type_key=cell_type_key,
        context_keys=context_keys,
    )
    support = cell_type_support(
        adata, sample_key=sample_key, cell_type_key=cell_type_key
    )
    adata.obs = adata.obs.copy()
    adata.obs["_adapter_sample"] = adata.obs[sample_key].astype(str).astype("category")
    adata.obs["_adapter_cell_type"] = adata.obs[cell_type_key].astype(str)
    method_version = importlib.metadata.version("liana")

    consensus = li.rs.select_resource("consensus")[["ligand", "receptor"]]
    if resource_mode == "native":
        frozen_resource = liana_native_resource(consensus, version=method_version).drop(
            columns="resource_version"
        )
        method_resource = consensus
        resource_id = "liana_consensus"
        resource_version = method_version
        resource_payload: dict[str, object] = {
            "mode": resource_mode,
            "resource_id": resource_id,
            "version": resource_version,
            "license": "OmniPath/LIANA integrated resources; see LIANA upstream",
            "interactions": len(frozen_resource),
        }
    else:
        if harmonized_resource is None or harmonized_manifest is None:
            raise ValueError("harmonized arms require resource table and manifest")
        harmonized, frozen_manifest = load_harmonized_resource(
            harmonized_resource, harmonized_manifest
        )
        frozen_resource = pd.DataFrame(
            {
                "interaction_id": harmonized["harmonized_interaction_id"].astype(str),
                "native_interaction_id": (
                    harmonized["ligand"].astype(str)
                    + "|"
                    + harmonized["receptor"].astype(str)
                ),
                "ligand": harmonized["ligand"].astype(str),
                "receptor": harmonized["receptor"].astype(str),
                "method_covered": True,
            }
        )
        method_resource = harmonized[["ligand", "receptor"]].copy()
        resource_id = str(frozen_manifest["resource_id"])
        resource_version = str(frozen_manifest["version"])
        resource_payload = {
            "mode": resource_mode,
            "resource_id": resource_id,
            "version": resource_version,
            "payload_filename": harmonized_resource.name,
            "payload_sha256": sha256_file(harmonized_resource),
            "manifest_filename": harmonized_manifest.name,
            "manifest_sha256": sha256_file(harmonized_manifest),
            "interactions": len(frozen_resource),
            "method_covered": len(frozen_resource),
            "custom_resource_supported": True,
        }

    parameters = {
        "n_perms_per_sample": n_perms,
        "n_jobs": n_jobs,
        "min_cells": min_cells,
        "expr_prop": expr_prop,
        "seed": seed,
        "use_raw": False,
        "layer": layer,
        "sample_level_execution": True,
    }
    manifest = begin_manifest(
        repo_root=repo_root,
        dataset_id=dataset_id,
        method={
            "id": METHOD_ID,
            "name": "LIANA rank_aggregate with sample isolation",
            "version": method_version,
            "license": "BSD-3-Clause",
            "entrypoint": "liana.mt.rank_aggregate (per-sample loop)",
        },
        environment=python_environment(
            environment_name="liana_env",
            packages=("liana", "anndata", "numpy", "pandas", "scanpy"),
            threads=n_jobs,
        ),
        input_path=input_h5ad,
        input_shape=adata.shape,
        sample_metadata=sample_metadata,
        input_keys={
            "sample": sample_key,
            "subject": subject_key,
            "cell_type": cell_type_key,
            "contexts": context_keys,
        },
        resource=resource_payload,
        parameters=parameters,
        score_semantics={
            "primary_score": "magnitude_rank",
            "direction": "lower_is_stronger",
            "within_dataset_p_value": (
                "LIANA component cell-label specificity; not between-condition "
                "inference"
            ),
        },
    )
    started = time.perf_counter()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    try:
        observed: list[pd.DataFrame] = []
        sample_status: dict[str, str] = {}
        sample_failures: dict[str, dict[str, str]] = {}
        for ordinal, sample_id in enumerate(sample_metadata["sample_id"]):
            sample_id = str(sample_id)
            selected = adata[
                adata.obs["_adapter_sample"].astype(str) == sample_id
            ].copy()
            try:
                result = li.mt.rank_aggregate(
                    selected,
                    groupby="_adapter_cell_type",
                    resource_name="consensus",
                    resource=method_resource,
                    expr_prop=expr_prop,
                    min_cells=min_cells,
                    use_raw=False,
                    layer=layer,
                    return_all_lrs=True,
                    n_perms=n_perms,
                    seed=seed + ordinal,
                    n_jobs=n_jobs,
                    inplace=False,
                    verbose=True,
                )
                if result is None or result.empty:
                    continue
                result = result.copy()
                result.insert(0, "sample_id", sample_id)
                result.to_parquet(
                    raw_dir / f"sample_{ordinal:04d}.parquet", index=False
                )
                observed.append(_normalize(result, pair_map=frozen_resource))
            except BaseException as exc:
                sample_status[sample_id] = "method_failed"
                sample_failures[sample_id] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        sparse_observed = (
            pd.concat(observed, ignore_index=True) if observed else _empty_observed()
        )
        table = materialize_fixed_universe(
            sparse_observed,
            sample_metadata=sample_metadata,
            support=support,
            resource=frozen_resource,
            dataset_id=dataset_id,
            run_id=str(manifest["run_id"]),
            method_id=METHOD_ID,
            method_version=method_version,
            analysis_track="lr_stlr",
            resource_mode=resource_mode,
            resource_id=resource_id,
            resource_version=resource_version,
            score_name="magnitude_rank",
            score_direction="lower",
            specificity_score_name="specificity_rank",
            min_cells=min_cells,
            sample_status=sample_status,
        )
        manifest["sample_failures"] = sample_failures
        return finalize_manifest(manifest, table, output_dir, started=started)
    except BaseException as exc:
        fail_manifest(manifest, output_dir, exc, started=started)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--subject-key", default="subject_id")
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--context-key", action="append", required=True)
    parser.add_argument(
        "--resource-mode", choices=("H-common", "H-covered", "native"), default="native"
    )
    parser.add_argument("--harmonized-resource", type=Path)
    parser.add_argument("--harmonized-manifest", type=Path)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--expr-prop", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--layer")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.input_h5ad,
        args.output_dir,
        dataset_id=args.dataset_id,
        sample_key=args.sample_key,
        subject_key=args.subject_key,
        cell_type_key=args.cell_type_key,
        context_keys=tuple(args.context_key),
        resource_mode=args.resource_mode,
        harmonized_resource=args.harmonized_resource,
        harmonized_manifest=args.harmonized_manifest,
        n_perms=args.n_perms,
        n_jobs=args.n_jobs,
        min_cells=args.min_cells,
        expr_prop=args.expr_prop,
        seed=args.seed,
        layer=args.layer,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
