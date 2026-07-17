"""Run CRYCHIC static availability for CytoSig cytokine validation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import anndata as ad
import pandas as pd

from benchmarks.openproblems.common import sha256_file, write_json
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


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _subunits(value: object) -> tuple[str, ...]:
    return tuple(part for part in str(value).split("_") if part)


def _bundle(resource_dir: Path) -> tuple[ResourceBundle, dict[str, object], Path]:
    table_path = resource_dir / "liana_consensus_human.parquet"
    manifest_path = resource_dir / "resource_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "crychic-cytokine-common-resource-v1"
        or sha256_file(table_path) != manifest["files"]["resource"]["sha256"]
    ):
        raise ValueError("cytokine common resource manifest is invalid")
    table = pd.read_parquet(table_path)
    interactions = []
    genes: set[str] = set()
    for row in table.loc[
        :, ["interaction_id", "ligand", "receptor", "input_available"]
    ].itertuples(index=False):
        ligand = _subunits(row.ligand)
        receptor = _subunits(row.receptor)
        genes.update((*ligand, *receptor))
        interactions.append(
            Interaction(
                interaction_id=str(row.interaction_id),
                source_interaction_id=str(row.interaction_id),
                ligand_name=str(row.ligand),
                receptor_name=str(row.receptor),
                ligand_subunits=ligand,
                receptor_subunits=receptor,
                ligand_is_complex=len(ligand) > 1,
                receptor_is_complex=len(receptor) > 1,
                direction="Ligand-Receptor",
                source=str(manifest["resource_id"]),
                version="current-liana-consensus",
                species=Species.HUMAN,
                gene_namespace=GeneNamespace.HGNC_SYMBOL,
                evidence=("LIANA consensus",),
                metadata=(("input_available", str(bool(row.input_available)).lower()),),
            )
        )
    bundle = ResourceBundle(
        resource_id=str(manifest["resource_id"]),
        version="current-liana-consensus",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=tuple(interactions),
        mapping_report=MappingReport(
            source_rows=len(table),
            loaded_rows=len(table),
            mapped_entities=len(genes),
            notes=("cytokine benchmark common resource",),
        ),
        manifest_digest=sha256_file(manifest_path),
        source_files=(table_path.name, manifest_path.name),
        license="Benchmark use subject to LIANA and upstream resource licenses",
        citation="Dimitrov D et al. Nature Communications 2022;13:3224.",
    )
    return bundle, manifest, table_path


def run(
    input_h5ad: Path,
    resource_dir: Path,
    output_dir: Path,
    *,
    min_cells: int,
    dataset_id: str = "TNBC",
    label_key: str = "label",
    overwrite: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"CRYCHIC cytokine output exists: {manifest_path}")
    bundle, resource_manifest, resource_path = _bundle(resource_dir)
    started = time.perf_counter()
    data = ad.read_h5ad(input_h5ad)
    if not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")
    if label_key not in data.obs:
        raise ValueError(f"cytokine input lacks label key: {label_key}")
    removed_uns_keys = tuple(sorted(map(str, data.uns.keys())))
    data.uns.clear()
    data.obs["cyto_sample"] = dataset_id
    data.obs["cyto_subject"] = dataset_id
    data.obs["cyto_context"] = dataset_id
    data.obs["cyto_cell_type"] = data.obs[label_key].astype(str).astype("object")
    data.layers["cyto_counts"] = data.X
    validated = validate_anndata(
        data,
        InputSchema(
            context_keys=("cyto_context",),
            counts_layer="cyto_counts",
            sample_key="cyto_sample",
            subject_key="cyto_subject",
            cell_type_key="cyto_cell_type",
            species=Species.HUMAN.value,
            gene_namespace=GeneNamespace.HGNC_SYMBOL.value,
        ),
    )
    aggregate = aggregate_pseudobulk(validated, min_cells=min_cells)
    availability = estimate_bundle_availability(
        aggregate,
        bundle,
        context_keys=("cyto_context",),
        parameters=AvailabilityParameters(),
        min_pooled_availability=0.0,
        max_interactions=None,
    )
    raw = availability.sample_interactions.rename(
        columns={"sender": "source", "receiver": "target"}
    )
    raw_path = output_dir / "availability_lr_scores.parquet"
    raw.to_parquet(raw_path, index=False)
    manifest: dict[str, object] = {
        "schema_version": "crychic-cytokine-availability-run-v1",
        "status": "complete",
        "input": {
            "filename": input_h5ad.name,
            "sha256": sha256_file(input_h5ad),
            "shape": list(data.shape),
            "truth_and_proxy_isolation": {
                "all_uns_removed_before_core_execution": True,
                "removed_key_names": list(removed_uns_keys),
            },
        },
        "resource": {
            "id": bundle.resource_id,
            "sha256": sha256_file(resource_path),
            "rows": len(bundle.interactions),
            "input_available_rows": int(
                resource_manifest["input_available_interactions"]
            ),
        },
        "parameters": {
            "min_cells": min_cells,
            "dataset_id": dataset_id,
            "label_key": label_key,
            "formal_inference": False,
            "subject_crossfit": False,
            "single_context_static_diagnostic": True,
            "score_arms": ["availability_state", "availability_ecosystem"],
        },
        "core_execution": {
            "entrypoints": [
                "crychic.data.validate_anndata",
                "crychic.pseudobulk.aggregate_pseudobulk",
                "crychic.availability.estimate_bundle_availability",
            ],
            "input_mode": validated.mode.value,
            "aggregate_units": len(aggregate.matrix_unit_ids),
            "rows": len(raw),
        },
        "versions": {
            name: _version(name)
            for name in ("CRYCHIC", "anndata", "numpy", "pandas", "scipy")
        },
        "elapsed_seconds": time.perf_counter() - started,
        "output": {
            "filename": raw_path.name,
            "sha256": sha256_file(raw_path),
            "rows": len(raw),
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
    parser.add_argument("--dataset-id", default="TNBC")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.input_h5ad,
        args.resource_dir,
        args.output_dir,
        min_cells=args.min_cells,
        dataset_id=args.dataset_id,
        label_key=args.label_key,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
