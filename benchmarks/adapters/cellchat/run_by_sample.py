"""Run CellChat independently for each biological sample."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from benchmarks.adapters.common import (
    begin_manifest,
    cell_type_support,
    fail_manifest,
    finalize_manifest,
    load_harmonized_resource,
    materialize_fixed_universe,
    method_frozen_resource,
    prepare_output,
    sha256_file,
    validate_prepared_input,
)
from benchmarks.adapters.resource_tables import native_lr_resource

METHOD_ID = "cellchat"
R_SEED_MODULUS = 2_147_483_647
UINT32_MAX = 2**32 - 1


def _r_query(environment: str, expression: str) -> str:
    completed = subprocess.run(
        ["conda", "run", "-n", environment, "Rscript", "-e", expression],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _environment(environment: str, threads: int) -> tuple[str, dict[str, object]]:
    version = _r_query(
        environment,
        "suppressPackageStartupMessages(library(CellChat)); "
        "cat(as.character(packageVersion('CellChat')))",
    )
    r_version = _r_query(environment, "cat(as.character(getRversion()))")
    return version, {
        "environment_name": environment,
        "runtime": "R",
        "r": r_version,
        "packages": {"CellChat": version},
        "threads": threads,
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def _r_seed(seed: int, ordinal: int) -> int:
    """Map a requested uint32 seed into R's positive signed-integer range."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= UINT32_MAX
    ):
        raise ValueError("CellChat requested seed must be a uint32 integer")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("CellChat sample ordinal must be a non-negative integer")
    effective = seed + ordinal
    if 1 <= effective <= R_SEED_MODULUS:
        return effective
    return 1 + ((effective - 1) % R_SEED_MODULUS)


def _seed_provenance(sample_ids: list[str], requested_seed: int) -> dict[str, Any]:
    effective = {
        sample_id: _r_seed(requested_seed, ordinal)
        for ordinal, sample_id in enumerate(sample_ids)
    }
    if len(effective) != len(sample_ids):
        raise ValueError("CellChat sample IDs must be unique for seed assignment")
    return {
        "requested_seed": requested_seed,
        "requested_seed_type": "uint32",
        "effective_seed_range": [1, R_SEED_MODULUS],
        "effective_sample_seeds": effective,
        "sample_seed_mapping": (
            "preserve requested_seed + sample_ordinal when in 1..2147483647; "
            "otherwise 1 + ((effective - 1) modulo 2147483647)"
        ),
    }


def _expression(adata: ad.AnnData, layer: str | None) -> sparse.csr_matrix:
    values = adata.X if layer is None else adata.layers[layer]
    matrix = sparse.csr_matrix(values)
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or (matrix.data < 0).any()
    ):
        raise ValueError("CellChat input expression must be finite and non-negative")
    return matrix


def _write_sample(
    adata: ad.AnnData,
    directory: Path,
    *,
    cell_type_key: str,
    layer: str | None,
) -> None:
    directory.mkdir(parents=True)
    matrix = _expression(adata, layer).transpose().tocsr()
    mmwrite(directory / "matrix.mtx", matrix)
    (directory / "genes.txt").write_text(
        "\n".join(adata.var_names.astype(str)) + "\n", encoding="utf-8"
    )
    (directory / "cells.txt").write_text(
        "\n".join(adata.obs_names.astype(str)) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "cell": adata.obs_names.astype(str),
            "cell_type": adata.obs[cell_type_key].astype(str).to_numpy(),
        }
    ).to_csv(directory / "metadata.tsv", sep="\t", index=False)


def _normalize_sample(
    table: pd.DataFrame,
    *,
    sample_id: str,
    source_map: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
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
    required = {"source", "target", "interaction_name", "prob", "pval"}
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"CellChat output lacks columns: {sorted(missing)}")
    result = table.rename(columns={"source": "sender", "target": "receiver"})
    result = result.merge(
        source_map[["source_interaction_id", "interaction_id"]],
        left_on="interaction_name",
        right_on="source_interaction_id",
        how="inner",
        validate="many_to_one",
    )
    result["score"] = pd.to_numeric(result["prob"], errors="coerce")
    result["within_dataset_p_value"] = pd.to_numeric(result["pval"], errors="coerce")
    result = result.loc[result["score"].notna()].copy()
    if result.empty:
        return _normalize_sample(
            pd.DataFrame(), sample_id=sample_id, source_map=source_map
        )
    grouped = result.groupby(
        ["sender", "receiver", "interaction_id"],
        sort=True,
        observed=True,
        as_index=False,
    ).agg(
        score=("score", "max"), within_dataset_p_value=("within_dataset_p_value", "min")
    )
    grouped["sample_id"] = sample_id
    grouped["target"] = pd.NA
    grouped["specificity_score"] = pd.NA
    grouped["within_dataset_p_value_semantics"] = (
        "CellChat within-sample bootstrap probability p-value; not a condition test"
    )
    return grouped


def run(
    input_h5ad: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    database_root: Path,
    sample_key: str,
    subject_key: str,
    cell_type_key: str,
    context_keys: tuple[str, ...],
    resource_mode: str,
    harmonized_resource: Path | None,
    harmonized_manifest: Path | None,
    layer: str | None,
    min_cells: int,
    nboot: int,
    trim: float,
    population_size: bool,
    seed: int,
    threads: int,
    environment: str,
    overwrite: bool,
) -> dict[str, object]:
    """Execute CellChat sample by sample and materialize its frozen universe."""
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
    method_version, environment_metadata = _environment(environment, threads)

    native_resource, native_map, native_metadata = native_lr_resource(
        method="cellchat",
        database_root=database_root,
        repo_root=repo_root,
    )
    if resource_mode == "native":
        frozen_resource = native_resource
        source_map = native_map
        resource_id = native_metadata["resource_id"]
        resource_version = native_metadata["resource_version"]
        resource_payload: dict[str, object] = {
            "mode": resource_mode,
            **native_metadata,
            "interactions": len(frozen_resource),
        }
    else:
        if harmonized_resource is None or harmonized_manifest is None:
            raise ValueError("harmonized arms require resource table and manifest")
        harmonized, frozen_manifest = load_harmonized_resource(
            harmonized_resource, harmonized_manifest
        )
        frozen_resource = method_frozen_resource(
            harmonized, method="cellchat", resource_mode=resource_mode
        )
        source_map = frozen_resource.loc[
            frozen_resource["method_covered"],
            ["native_interaction_id", "interaction_id"],
        ].rename(columns={"native_interaction_id": "source_interaction_id"})
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
            "method_covered": int(frozen_resource["method_covered"].sum()),
        }

    seed_provenance = _seed_provenance(
        sample_metadata["sample_id"].astype(str).tolist(), seed
    )
    effective_sample_seeds = cast(
        dict[str, int], seed_provenance["effective_sample_seeds"]
    )
    parameters = {
        "layer": layer,
        "min_cells": min_cells,
        "nboot": nboot,
        "trim": trim,
        "population_size": population_size,
        "seed": seed,
        **seed_provenance,
        "threads": threads,
        "sample_level_execution": True,
    }
    manifest = begin_manifest(
        repo_root=repo_root,
        dataset_id=dataset_id,
        method={
            "id": METHOD_ID,
            "name": "CellChat",
            "version": method_version,
            "license": "GPL-3",
            "entrypoint": "CellChat::computeCommunProb",
        },
        environment=environment_metadata,
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
            "primary_score": "communication_probability",
            "direction": "higher_is_stronger",
            "within_dataset_p_value": (
                "CellChat bootstrap probability p-value; not between-condition "
                "inference"
            ),
        },
    )
    started = time.perf_counter()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    sample_status: dict[str, str] = {}
    sample_failures: dict[str, dict[str, object]] = {}
    observed: list[pd.DataFrame] = []
    try:
        with tempfile.TemporaryDirectory(prefix="crychic_cellchat_") as temporary:
            work_root = Path(temporary)
            source_ids_path = work_root / "source_ids.txt"
            source_ids_path.write_text(
                "\n".join(source_map["source_interaction_id"].dropna().astype(str))
                + "\n",
                encoding="utf-8",
            )
            for ordinal, sample_id in enumerate(sample_metadata["sample_id"]):
                sample_id = str(sample_id)
                selected = adata[adata.obs[sample_key].astype(str) == sample_id].copy()
                sample_work = work_root / f"{ordinal:04d}_{_safe_name(sample_id)}"
                _write_sample(
                    selected,
                    sample_work,
                    cell_type_key=cell_type_key,
                    layer=layer,
                )
                raw_path = raw_dir / f"{ordinal:04d}_{_safe_name(sample_id)}.csv"
                command = [
                    "conda",
                    "run",
                    "-n",
                    environment,
                    "Rscript",
                    str(Path(__file__).with_name("run_sample.R")),
                    str(sample_work),
                    str(database_root / "cellchat/CellChatDB_human.rds"),
                    str(source_ids_path),
                    str(raw_path),
                    str(min_cells),
                    str(nboot),
                    str(effective_sample_seeds[sample_id]),
                    str(trim),
                    str(population_size).upper(),
                ]
                completed = subprocess.run(
                    command, capture_output=True, text=True, check=False
                )
                if completed.returncode != 0:
                    sample_status[sample_id] = "method_failed"
                    sample_failures[sample_id] = {
                        "returncode": completed.returncode,
                        "stdout": completed.stdout[-4000:],
                        "stderr": completed.stderr[-4000:],
                    }
                    continue
                raw = pd.read_csv(raw_path)
                observed.append(
                    _normalize_sample(raw, sample_id=sample_id, source_map=source_map)
                )
        sparse_observed = (
            pd.concat(observed, ignore_index=True)
            if observed
            else _normalize_sample(pd.DataFrame(), sample_id="", source_map=source_map)
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
            score_name="communication_probability",
            score_direction="higher",
            specificity_score_name=None,
            min_cells=min_cells,
            sample_status=sample_status,
        )
        manifest["sample_failures"] = sample_failures
        return cast(
            dict[str, object],
            finalize_manifest(manifest, table, output_dir, started=started),
        )
    except BaseException as exc:
        fail_manifest(manifest, output_dir, exc, started=started)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--database-root", required=True, type=Path)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--subject-key", default="subject_id")
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--context-key", action="append", required=True)
    parser.add_argument(
        "--resource-mode", choices=("H-common", "H-covered", "native"), default="native"
    )
    parser.add_argument("--harmonized-resource", type=Path)
    parser.add_argument("--harmonized-manifest", type=Path)
    parser.add_argument("--layer")
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=100)
    parser.add_argument("--trim", type=float, default=0.1)
    parser.add_argument("--population-size", action="store_true")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--environment", default="r_cellchat")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.input_h5ad,
        args.output_dir,
        dataset_id=args.dataset_id,
        database_root=args.database_root,
        sample_key=args.sample_key,
        subject_key=args.subject_key,
        cell_type_key=args.cell_type_key,
        context_keys=tuple(args.context_key),
        resource_mode=args.resource_mode,
        harmonized_resource=args.harmonized_resource,
        harmonized_manifest=args.harmonized_manifest,
        layer=args.layer,
        min_cells=args.min_cells,
        nboot=args.nboot,
        trim=args.trim,
        population_size=args.population_size,
        seed=args.seed,
        threads=args.threads,
        environment=args.environment,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
