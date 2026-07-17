"""Run the CRYCHIC steady-state diagnostic for Open Problems v1.0.0."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from benchmarks.openproblems.common import (
    aggregate_lr_scores,
    sha256_file,
    write_json,
    write_predictions,
)
from crychic.availability import AvailabilityParameters, estimate_bundle_availability
from crychic.data import InputSchema, validate_anndata
from crychic.pseudobulk import aggregate_pseudobulk
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
)

METHOD_SCOPE = "steady_state_pooled_static_availability_no_subject_crossfit"
SPECIFICITY_SCOPE = (
    "steady_state_pooled_availability_natmi_style_specificity_posthoc_diagnostic"
)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _subunits(value: object) -> tuple[str, ...]:
    result = tuple(part for part in str(value).split("_") if part)
    if not result:
        raise ValueError("resource complex has no gene subunits")
    return result


def load_task_resource(
    resource_path: str | Path,
    manifest_path: str | Path,
) -> tuple[ResourceBundle, dict[str, Any]]:
    """Build a checksum-bound mouse bundle from the common LIANA resource."""

    table_path = Path(resource_path)
    metadata_path = Path(manifest_path)
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "crychic-openproblems-source-target-resource-v1"
        or manifest.get("task_version") != "v1.0.0"
    ):
        raise ValueError("unsupported Open Problems resource manifest")
    declared = manifest.get("files", {}).get("resource", {})
    if declared.get("filename") != table_path.name:
        raise ValueError("resource filename differs from its manifest")
    actual_hash = sha256_file(table_path)
    if declared.get("sha256") != actual_hash:
        raise ValueError("resource checksum differs from its manifest")
    table = pd.read_parquet(table_path)
    required = {"interaction_id", "ligand", "receptor", "input_available"}
    if required.difference(table.columns) or table.empty:
        raise ValueError("Open Problems resource table has an invalid schema")
    if table["interaction_id"].astype(str).duplicated().any():
        raise ValueError("Open Problems resource has duplicate interaction IDs")

    interactions: list[Interaction] = []
    mapped_genes: set[str] = set()
    resource_id = str(manifest["resource_id"])
    for row in table.loc[
        :, ["interaction_id", "ligand", "receptor", "input_available"]
    ].itertuples(index=False):
        ligand = _subunits(row.ligand)
        receptor = _subunits(row.receptor)
        mapped_genes.update((*ligand, *receptor))
        interaction_id = str(row.interaction_id)
        interactions.append(
            Interaction(
                interaction_id=interaction_id,
                source_interaction_id=interaction_id,
                ligand_name=str(row.ligand),
                receptor_name=str(row.receptor),
                ligand_subunits=ligand,
                receptor_subunits=receptor,
                ligand_is_complex=len(ligand) > 1,
                receptor_is_complex=len(receptor) > 1,
                direction="Ligand-Receptor",
                source=resource_id,
                version="v1.0.0-current-liana-bridge",
                species=Species.MOUSE,
                gene_namespace=GeneNamespace.MGI_SYMBOL,
                evidence=("LIANA consensus via HCOP min-evidence 3",),
                metadata=(("input_available", str(bool(row.input_available)).lower()),),
            )
        )
    bundle = ResourceBundle(
        resource_id=resource_id,
        version="v1.0.0-current-liana-bridge",
        species=Species.MOUSE,
        gene_namespace=GeneNamespace.MGI_SYMBOL,
        interactions=tuple(interactions),
        mapping_report=MappingReport(
            source_rows=len(table),
            loaded_rows=len(table),
            mapped_entities=len(mapped_genes),
            notes=("current LIANA consensus translated through HCOP",),
        ),
        manifest_digest=sha256_file(metadata_path),
        source_files=(table_path.name, metadata_path.name),
        license="Benchmark use subject to LIANA and upstream resource licenses",
        citation=(
            "Dimitrov D et al. Comparison of methods and resources for cell-cell "
            "communication inference from single-cell RNA-Seq data. Nat Commun 2022."
        ),
    )
    return bundle, manifest


def _label_axis(data: ad.AnnData) -> tuple[str, ...]:
    if "label" not in data.obs:
        raise ValueError("Open Problems input lacks obs['label']")
    labels = data.obs["label"]
    if isinstance(labels.dtype, pd.CategoricalDtype):
        return tuple(map(str, labels.cat.categories))
    return tuple(map(str, pd.unique(labels.astype(str))))


def availability_specificity(raw: pd.DataFrame) -> pd.DataFrame:
    """Factorize state availability into sender and receiver specificity."""

    required = {
        "interaction_id",
        "source",
        "target",
        "ligand_availability",
        "receptor_availability",
    }
    if required.difference(raw.columns):
        raise ValueError("availability table cannot derive specificity")
    ligand = raw.loc[
        :, ["interaction_id", "source", "ligand_availability"]
    ].drop_duplicates(["interaction_id", "source"])
    receptor = raw.loc[
        :, ["interaction_id", "target", "receptor_availability"]
    ].drop_duplicates(["interaction_id", "target"])
    ligand_total = ligand.groupby("interaction_id", observed=True)[
        "ligand_availability"
    ].transform("sum")
    receptor_total = receptor.groupby("interaction_id", observed=True)[
        "receptor_availability"
    ].transform("sum")
    ligand["ligand_specificity"] = ligand["ligand_availability"].div(
        ligand_total.where(ligand_total.gt(0.0))
    )
    receptor["receptor_specificity"] = receptor["receptor_availability"].div(
        receptor_total.where(receptor_total.gt(0.0))
    )
    result = raw.merge(
        ligand.loc[:, ["interaction_id", "source", "ligand_specificity"]],
        on=["interaction_id", "source"],
        how="left",
        validate="many_to_one",
    ).merge(
        receptor.loc[:, ["interaction_id", "target", "receptor_specificity"]],
        on=["interaction_id", "target"],
        how="left",
        validate="many_to_one",
    )
    result["availability_specificity"] = (
        result["ligand_specificity"] * result["receptor_specificity"]
    ).fillna(0.0)
    return result


def run(
    input_h5ad: Path,
    resource_dir: Path,
    output_dir: Path,
    *,
    min_cells: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Run static molecular availability without reading benchmark truth."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "crychic_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"CRYCHIC output already exists: {manifest_path}")
    resource_path = resource_dir / "liana_consensus_mouse.parquet"
    resource_manifest_path = resource_dir / "resource_manifest.json"
    bundle, resource_manifest = load_task_resource(
        resource_path, resource_manifest_path
    )

    started = time.perf_counter()
    data = ad.read_h5ad(input_h5ad)
    labels = _label_axis(data)
    removed_uns_keys = tuple(sorted(map(str, data.uns.keys())))
    data.uns.clear()
    data.obs["op_sample_id"] = "mouse_brain_atlas"
    data.obs["op_subject_id"] = "mouse_brain_atlas"
    data.obs["op_context"] = "steady_state"
    data.obs["op_cell_type"] = data.obs["label"].astype(str).astype("object")
    data.layers["op_counts"] = data.X
    schema = InputSchema(
        context_keys=("op_context",),
        counts_layer="op_counts",
        sample_key="op_sample_id",
        subject_key="op_subject_id",
        cell_type_key="op_cell_type",
        species=Species.MOUSE.value,
        gene_namespace=GeneNamespace.MGI_SYMBOL.value,
    )
    validated = validate_anndata(data, schema)
    pseudobulk = aggregate_pseudobulk(validated, min_cells=min_cells)
    availability = estimate_bundle_availability(
        pseudobulk,
        bundle,
        context_keys=("op_context",),
        parameters=AvailabilityParameters(),
        min_pooled_availability=0.0,
        max_interactions=None,
    )
    raw = availability.sample_interactions.copy()
    required = {
        "sender",
        "receiver",
        "interaction_id",
        "availability_state",
        "state_status",
        "state_reason_code",
    }
    if required.difference(raw.columns):
        raise RuntimeError("CRYCHIC availability output has an invalid schema")
    raw = raw.rename(columns={"sender": "source", "receiver": "target"})
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    raw_path = raw_dir / "crychic_availability_lr.parquet"
    raw.to_parquet(raw_path, index=False)
    observed = raw.loc[
        raw["state_status"].astype(str).eq("observed")
        & pd.to_numeric(raw["availability_state"], errors="coerce").notna()
    ].copy()
    if observed.empty:
        raise RuntimeError("CRYCHIC produced no observed availability scores")

    ecosystem = raw.loc[
        raw["ecosystem_status"].astype(str).eq("observed")
        & pd.to_numeric(raw["availability_ecosystem"], errors="coerce").notna()
    ].copy()
    specificity = availability_specificity(observed)
    score_definitions = (
        (
            observed,
            "availability_state",
            "crychic_availability",
            "CRYCHIC steady-state availability",
            METHOD_SCOPE,
        ),
        (
            ecosystem,
            "availability_ecosystem",
            "crychic_availability_ecosystem",
            "CRYCHIC ecosystem availability",
            METHOD_SCOPE,
        ),
        (
            specificity,
            "availability_specificity",
            "crychic_availability_specificity",
            "CRYCHIC availability specificity",
            SPECIFICITY_SCOPE,
        ),
    )
    predictions = []
    for source, value_column, base_id, base_name, scope in score_definitions:
        for aggregation in ("max", "sum"):
            predictions.append(
                aggregate_lr_scores(
                    source,
                    value_column=value_column,
                    method_id=f"{base_id}_{aggregation}",
                    method_name=f"{base_name} ({aggregation})",
                    method_scope=scope,
                    resource_id=bundle.resource_id,
                    aggregation=aggregation,
                    labels=labels,
                )
            )
    prediction_dir = output_dir / "predictions"
    prediction_paths = [
        write_predictions(prediction_dir, table) for table in predictions
    ]
    manifest: dict[str, Any] = {
        "schema_version": "crychic-openproblems-source-target-run-v1",
        "status": "complete",
        "task_version": "v1.0.0",
        "input": {
            "filename": input_h5ad.name,
            "sha256": sha256_file(input_h5ad),
            "shape": list(data.shape),
            "cell_types": len(labels),
            "truth_and_proxy_isolation": {
                "all_uns_removed_before_core_execution": True,
                "removed_key_names": list(removed_uns_keys),
            },
        },
        "resource": {
            "resource_id": bundle.resource_id,
            "filename": resource_path.name,
            "sha256": sha256_file(resource_path),
            "rows": len(bundle.interactions),
            "input_available_rows": int(
                resource_manifest["input_available_interactions"]
            ),
            "bridge_manifest_sha256": sha256_file(resource_manifest_path),
        },
        "parameters": {
            "min_cells": min_cells,
            "aggregation": ["max", "sum"],
            "score_arms": {
                "availability_state": "preexisting_core_mode",
                "availability_ecosystem": "preexisting_core_mode",
                "availability_specificity": (
                    "posthoc_supportive_natmi_style_diagnostic"
                ),
            },
            "formal_inference": False,
            "subject_crossfit": False,
            "single_context_static_diagnostic": True,
        },
        "core_execution": {
            "entrypoints": [
                "crychic.data.validate_anndata",
                "crychic.pseudobulk.aggregate_pseudobulk",
                "crychic.availability.estimate_bundle_availability",
            ],
            "input_mode": validated.mode.value,
            "aggregate_units": len(pseudobulk.matrix_unit_ids),
            "raw_rows": len(raw),
            "observed_rows": len(observed),
            "detection_available": availability.detection_available,
        },
        "versions": {
            name: _version(name)
            for name in ("CRYCHIC", "anndata", "pandas", "numpy", "scipy")
        },
        "elapsed_seconds": time.perf_counter() - started,
        "raw": {raw_path.name: sha256_file(raw_path)},
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
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.input_h5ad,
        args.resource_dir,
        args.output_dir,
        min_cells=args.min_cells,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
