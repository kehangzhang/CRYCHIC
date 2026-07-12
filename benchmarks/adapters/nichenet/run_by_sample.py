"""Score NicheNet ligand-target programs independently by biological sample."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.adapters.common import (
    begin_manifest,
    cell_type_support,
    fail_manifest,
    finalize_manifest,
    materialize_fixed_universe,
    prepare_output,
    python_environment,
    sha256_file,
    validate_prepared_input,
)
from benchmarks.adapters.resource_tables import nichenet_ligand_program_resource

METHOD_ID = "nichenet_prior_activity"
ADAPTER_VERSION = "crychic-nichenet-prior-activity-v1"


def _expression(adata: ad.AnnData, layer: str | None) -> tuple[sparse.csr_matrix, str]:
    values = adata.X if layer is None else adata.layers[layer]
    matrix = sparse.csr_matrix(values, dtype=np.float64)
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or (matrix.data < 0).any()
    ):
        raise ValueError("NicheNet adapter expression must be finite and non-negative")
    integer_like = not matrix.data.size or np.allclose(
        matrix.data,
        np.rint(matrix.data),
        rtol=0.0,
        atol=1e-8,
    )
    if not integer_like:
        return matrix, "input_continuous_expression_preserved"
    library_sizes = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(
        1.0e4,
        library_sizes,
        out=np.zeros_like(library_sizes, dtype=float),
        where=library_sizes > 0,
    )
    normalized = sparse.diags(scale) @ matrix
    normalized = sparse.csr_matrix(normalized)
    normalized.data = np.log1p(normalized.data)
    return normalized, "counts_library_size_1e4_log1p"


def _group_means(
    adata: ad.AnnData,
    *,
    sample_key: str,
    cell_type_key: str,
    layer: str | None,
) -> tuple[dict[tuple[str, str], np.ndarray], str]:
    matrix, transform = _expression(adata, layer)
    samples = adata.obs[sample_key].astype(str).to_numpy()
    cell_types = adata.obs[cell_type_key].astype(str).to_numpy()
    result: dict[tuple[str, str], np.ndarray] = {}
    for sample_id, cell_type in sorted(set(zip(samples, cell_types, strict=True))):
        selected = (samples == sample_id) & (cell_types == cell_type)
        result[(sample_id, cell_type)] = np.asarray(
            matrix[selected].mean(axis=0)
        ).ravel()
    return result, transform


def _prior_operator(
    prior: pd.DataFrame,
    *,
    var_names: pd.Index,
    ligand_order: list[str],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    gene_index = {str(gene): index for index, gene in enumerate(var_names)}
    ligand_index = {ligand: index for index, ligand in enumerate(ligand_order)}
    selected = prior.loc[
        prior["target"].astype(str).isin(gene_index)
        & prior["ligand"].astype(str).isin(ligand_index)
    ].copy()
    target_rows = selected["target"].astype(str).map(gene_index).to_numpy(dtype=int)
    ligand_columns = (
        selected["ligand"].astype(str).map(ligand_index).to_numpy(dtype=int)
    )
    weights = pd.to_numeric(selected["weight"], errors="raise").to_numpy(dtype=float)
    operator = sparse.coo_matrix(
        (weights, (target_rows, ligand_columns)),
        shape=(len(var_names), len(ligand_order)),
    ).tocsr()
    weight_sum = np.asarray(operator.sum(axis=0)).ravel()
    target_count = np.bincount(ligand_columns, minlength=len(ligand_order))
    ligand_gene_index = np.array(
        [gene_index.get(ligand, -1) for ligand in ligand_order], dtype=int
    )
    return operator, weight_sum, target_count, ligand_gene_index


def _score_samples(
    means: dict[tuple[str, str], np.ndarray],
    *,
    sample_ids: list[str],
    cell_types: list[str],
    resource: pd.DataFrame,
    operator: sparse.csr_matrix,
    weight_sum: np.ndarray,
    support: pd.DataFrame,
    min_cells: int,
) -> pd.DataFrame:
    counts = {
        (str(row.sample_id), str(row.cell_type)): int(float(str(row.n_cells)))
        for row in support.itertuples(index=False)
    }
    interaction_ids = resource["interaction_id"].astype(str).to_numpy()
    rows: list[pd.DataFrame] = []
    for sample_id in sample_ids:
        for receiver in cell_types:
            receiver_mean = means.get((sample_id, receiver))
            if (
                receiver_mean is None
                or counts.get((sample_id, receiver), 0) < min_cells
            ):
                continue
            target_activity = np.asarray(receiver_mean @ operator).ravel()
            target_activity = np.divide(
                target_activity,
                weight_sum,
                out=np.full_like(target_activity, np.nan),
                where=weight_sum > 0,
            )
            emitted = np.isfinite(target_activity)
            if not emitted.any():
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "sample_id": sample_id,
                        "sender": "__source_agnostic__",
                        "receiver": receiver,
                        "interaction_id": interaction_ids[emitted],
                        "target": "__weighted_target_program__",
                        "score": target_activity[emitted],
                        "specificity_score": target_activity[emitted],
                        "within_dataset_p_value": np.nan,
                        "within_dataset_p_value_semantics": "not_emitted",
                    }
                )
            )
    if not rows:
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
    return pd.concat(rows, ignore_index=True)


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
    layer: str | None,
    min_cells: int,
    min_targets: int,
    threads: int,
    overwrite: bool,
) -> dict[str, object]:
    """Run the frozen local NicheNet prior without requiring its R package."""
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
    prior_path = database_root / "nichenet/v2_2021/ligand_target_top250.parquet"
    manifest_path = database_root / "nichenet/v2_2021/manifest.json"
    prior = pd.read_parquet(prior_path)
    resource_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resource = nichenet_ligand_program_resource(prior)
    ligand_order = resource["ligand"].astype(str).tolist()
    operator, weight_sum, target_count, ligand_gene_index = _prior_operator(
        prior, var_names=adata.var_names, ligand_order=ligand_order
    )
    resource["input_available"] = (ligand_gene_index >= 0) & (
        target_count >= min_targets
    )
    resource_id = str(resource_manifest["resource_id"])
    resource_version = str(resource_manifest["release"])
    resource_payload = {
        "mode": "native",
        "resource_id": resource_id,
        "version": resource_version,
        "license": str(resource_manifest["license"]),
        "citation": str(resource_manifest["citation"]),
        "payload_filename": prior_path.name,
        "payload_sha256": sha256_file(prior_path),
        "manifest_filename": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "ligands": len(resource),
        "input_available_ligands": int(resource["input_available"].sum()),
    }
    parameters = {
        "layer": layer,
        "min_cells": min_cells,
        "min_targets": min_targets,
        "target_program": "weighted mean of matched top-250 target expression",
        "expression_transform": "auto_detect_counts_else_preserve_continuous",
        "sender_term": "source agnostic; sender prioritization is out of Track B",
        "score": "receiver weighted target-program mean for each candidate ligand",
        "threads": threads,
        "sample_level_execution": True,
    }
    manifest = begin_manifest(
        repo_root=repo_root,
        dataset_id=dataset_id,
        method={
            "id": METHOD_ID,
            "name": "NicheNet frozen-prior sample activity proxy",
            "version": ADAPTER_VERSION,
            "license": "adapter code under project license; prior CC-BY-4.0",
            "entrypoint": "local sparse ligand-target matrix multiplication",
            "native_nichenet_claim": False,
        },
        environment=python_environment(
            environment_name="crychic",
            packages=("anndata", "numpy", "pandas", "scipy", "pyarrow"),
            threads=threads,
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
            "primary_score": "ligand_target_program_proxy",
            "direction": "higher_is_stronger",
            "within_dataset_p_value": "not_emitted",
            "warning": (
                "This is not NicheNet predict_ligand_activities: sample-level inputs "
                "do not define a receiver DE gene set. It is a preregistered Track-B "
                "proxy using the exact frozen NicheNet prior."
            ),
        },
    )
    started = time.perf_counter()
    try:
        means, expression_transform = _group_means(
            adata,
            sample_key=sample_key,
            cell_type_key=cell_type_key,
            layer=layer,
        )
        manifest["parameters"]["resolved_expression_transform"] = expression_transform
        observed = _score_samples(
            means,
            sample_ids=sample_metadata["sample_id"].astype(str).tolist(),
            cell_types=sorted(support["cell_type"].astype(str).unique()),
            resource=resource,
            operator=operator,
            weight_sum=weight_sum,
            support=support,
            min_cells=min_cells,
        )
        table = materialize_fixed_universe(
            observed,
            sample_metadata=sample_metadata,
            support=support,
            resource=resource,
            dataset_id=dataset_id,
            run_id=str(manifest["run_id"]),
            method_id=METHOD_ID,
            method_version=ADAPTER_VERSION,
            analysis_track="ligand_target_program",
            resource_mode="native",
            resource_id=resource_id,
            resource_version=resource_version,
            score_name="ligand_target_program_proxy",
            score_direction="higher",
            specificity_score_name="receiver_weighted_target_program",
            min_cells=min_cells,
            sender_types=("__source_agnostic__",),
            sender_requires_cells=False,
            interaction_direction="ligand_to_target_program",
        )
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
    parser.add_argument("--layer")
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--min-targets", type=int, default=25)
    parser.add_argument("--threads", type=int, default=1)
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
        layer=args.layer,
        min_cells=args.min_cells,
        min_targets=args.min_targets,
        threads=args.threads,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
