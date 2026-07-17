"""Run the static, sample-level CRYCHIC availability diagnostic.

This adapter deliberately stops before response modeling, sender attribution,
or differential communication scoring. It delegates the numerical work to the
public CRYCHIC validation, pseudobulk, and availability APIs and only adds
benchmark resource bridges, tidy output, and provenance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    begin_manifest,
    canonical_context_json,
    fail_manifest,
    prepare_output,
    python_environment,
    sha256_file,
    validate_prepared_input,
    write_json,
)
from crychic.availability import (
    AvailabilityParameters,
    BatchAvailability,
    estimate_bundle_availability,
)
from crychic.data import ExpressionTransform, InputSchema, validate_anndata
from crychic.pseudobulk import aggregate_pseudobulk
from crychic.resources import ResourceBundle, Species

from .resource import harmonized_resource_bundle, load_native_resource_bundle

REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_ID = "crychic_availability_state_diagnostic"
OUTPUT_SCHEMA = "crychic-availability-state-diagnostic-v1"
SCORE_NAME = "availability_state"
SCORE_DIRECTION = "higher"

SCORE_COLUMNS = (
    "schema_version",
    "run_id",
    "dataset_id",
    "method_id",
    "method_version",
    "resource_mode",
    "resource_id",
    "resource_version",
    "sample_id",
    "subject_id",
    "context_id",
    "context_json",
    "sender",
    "receiver",
    "interaction_id",
    "native_interaction_id",
    "ligand",
    "receptor",
    "ligand_availability",
    "receptor_availability",
    "availability_state",
    "status",
    "reason_code",
)


def _package_version() -> str:
    try:
        return importlib.metadata.version("CRYCHIC")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _canonical_context_keys(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        raise ValueError("context_keys must contain at least one field")
    if any(not value or value != value.strip() for value in values):
        raise ValueError("context_keys must contain canonical non-empty names")
    if len(set(values)) != len(values):
        raise ValueError("context_keys must not contain duplicates")
    return values


def _normalize_declared_metadata(
    adata: ad.AnnData, columns: tuple[str, ...]
) -> tuple[str, ...]:
    """Make h5ad categorical metadata safe for the public validation path."""

    converted: list[str] = []
    for column in columns:
        if column not in adata.obs:
            continue
        if isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
            adata.obs[column] = adata.obs[column].astype("object")
            converted.append(column)
    return tuple(converted)


def _resource_bundle(
    *,
    resource_mode: str,
    harmonized_resource: str | Path | None,
    harmonized_manifest: str | Path | None,
    native_adapter: str | None,
    database_root: str | Path | None,
    resource_manifest: str | Path | None,
    species: Species | str,
) -> tuple[ResourceBundle, dict[str, Any]]:
    if resource_mode == "H-common":
        if harmonized_resource is None or harmonized_manifest is None:
            raise ValueError(
                "H-common requires harmonized_resource and harmonized_manifest"
            )
        if any(
            value is not None
            for value in (native_adapter, database_root, resource_manifest)
        ):
            raise ValueError("H-common cannot be combined with native resource inputs")
        table_path = Path(harmonized_resource).expanduser().resolve()
        manifest_path = Path(harmonized_manifest).expanduser().resolve()
        bundle = harmonized_resource_bundle(table_path, manifest_path)
        provenance = {
            "mode": resource_mode,
            "id": bundle.resource_id,
            "version": bundle.version,
            "manifest_digest": bundle.manifest_digest,
            "interactions": len(bundle.interactions),
            "table_filename": table_path.name,
            "table_sha256": sha256_file(table_path),
            "manifest_filename": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "source_files": list(bundle.source_files),
            "species": bundle.species.value,
            "gene_namespace": bundle.gene_namespace.value,
        }
        return bundle, provenance
    if resource_mode != "native":
        raise ValueError("resource_mode must be 'H-common' or 'native'")
    if native_adapter not in {"cellchat", "cellphonedb"}:
        raise ValueError("native_adapter must be cellchat or cellphonedb")
    if database_root is None or resource_manifest is None:
        raise ValueError("native mode requires database_root and resource_manifest")
    if harmonized_resource is not None or harmonized_manifest is not None:
        raise ValueError("native mode cannot be combined with H-common inputs")
    database_path = Path(database_root).expanduser().resolve()
    manifest_path = Path(resource_manifest).expanduser().resolve()
    selected_species = Species(species)
    bundle = load_native_resource_bundle(
        native_adapter,
        database_root=database_path,
        manifest_path=manifest_path,
        species=selected_species,
    )
    provenance = {
        "mode": resource_mode,
        "id": bundle.resource_id,
        "version": bundle.version,
        "manifest_digest": bundle.manifest_digest,
        "interactions": len(bundle.interactions),
        "native_adapter": native_adapter,
        "database_root_name": database_path.name,
        "manifest_filename": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "source_files": list(bundle.source_files),
        "species": bundle.species.value,
        "gene_namespace": bundle.gene_namespace.value,
    }
    return bundle, provenance


def validate_availability_score_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate the dedicated static-availability output contract."""

    missing = set(SCORE_COLUMNS).difference(table.columns)
    extra = set(table.columns).difference(SCORE_COLUMNS)
    if missing or extra:
        raise ValueError(
            "availability score columns differ: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result = cast(pd.DataFrame, table.loc[:, SCORE_COLUMNS].copy())
    if result.empty:
        return result
    required_identifiers = [
        "schema_version",
        "run_id",
        "dataset_id",
        "method_id",
        "method_version",
        "resource_mode",
        "resource_id",
        "resource_version",
        "sample_id",
        "subject_id",
        "context_id",
        "context_json",
        "sender",
        "receiver",
        "interaction_id",
        "native_interaction_id",
        "ligand",
        "receptor",
        "status",
    ]
    if result.loc[:, required_identifiers].isna().any().any():
        raise ValueError("availability scores contain null required identifiers")
    if set(result["schema_version"].astype(str)) != {OUTPUT_SCHEMA}:
        raise ValueError("availability scores have an invalid schema_version")
    if set(result["method_id"].astype(str)) != {METHOD_ID}:
        raise ValueError("availability scores have an invalid method_id")
    if not set(result["resource_mode"].astype(str)).issubset({"H-common", "native"}):
        raise ValueError("availability scores have an invalid resource_mode")
    for value in result["context_json"].astype(str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("context_json must encode an object")
    numeric_columns = (
        "ligand_availability",
        "receptor_availability",
        "availability_state",
    )
    for column in numeric_columns:
        numeric = pd.to_numeric(result[column], errors="coerce")
        supplied = result[column].notna()
        if ((supplied & numeric.isna()) | np.isinf(numeric.fillna(0.0))).any():
            raise ValueError(f"{column} must contain finite numeric values or NA")
        if ((numeric.dropna() < 0) | (numeric.dropna() > 1)).any():
            raise ValueError(f"{column} must lie in [0, 1]")
        result[column] = numeric
    observed = result["status"].astype(str).eq("observed")
    if result.loc[observed, list(numeric_columns)].isna().any().any():
        raise ValueError("observed availability rows require all component values")
    if result.loc[~observed, list(numeric_columns)].notna().any().any():
        raise ValueError("non-observed availability rows require missing components")
    if (observed & result["reason_code"].notna()).any():
        raise ValueError("observed availability rows must not carry reason_code")
    if ((~observed) & result["reason_code"].isna()).any():
        raise ValueError("non-observed availability rows require reason_code")
    key = ["run_id", "sample_id", "sender", "receiver", "interaction_id"]
    if result.duplicated(key).any():
        raise ValueError("availability scores contain duplicate result keys")
    return result


def _tidy_scores(
    availability: BatchAvailability,
    *,
    bundle: ResourceBundle,
    run_id: str,
    dataset_id: str,
    method_version: str,
    resource_mode: str,
    context_keys: tuple[str, ...],
) -> pd.DataFrame:
    source = availability.sample_interactions
    if source.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    required = {
        "sample_id",
        "subject_id",
        "context_id",
        *context_keys,
        "sender",
        "receiver",
        "interaction_id",
        "source_interaction_id",
        "ligand",
        "receptor",
        "ligand_availability",
        "receptor_availability",
        "availability_state",
        "state_status",
        "state_reason_code",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            f"core availability output is missing columns: {sorted(missing)}"
        )
    entity_labels = pd.DataFrame(
        {
            "interaction_id": [item.interaction_id for item in bundle.interactions],
            "ligand_gene_label": [
                "&".join(item.ligand_subunits) for item in bundle.interactions
            ],
            "receptor_gene_label": [
                "&".join(item.receptor_subunits) for item in bundle.interactions
            ],
        }
    )
    if entity_labels["interaction_id"].duplicated().any():
        raise ValueError("resource bundle contains duplicate interaction identifiers")
    canonical_entities = source.loc[:, ["interaction_id"]].merge(
        entity_labels,
        on="interaction_id",
        how="left",
        validate="many_to_one",
    )
    if (
        canonical_entities[["ligand_gene_label", "receptor_gene_label"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("core availability rows are absent from the resource bundle")
    context_json = [
        canonical_context_json(row, context_keys)
        for row in source.loc[:, list(context_keys)].to_dict(orient="records")
    ]
    result = pd.DataFrame(
        {
            "schema_version": OUTPUT_SCHEMA,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "method_id": METHOD_ID,
            "method_version": method_version,
            "resource_mode": resource_mode,
            "resource_id": availability.resource_id,
            "resource_version": availability.resource_version,
            "sample_id": source["sample_id"].astype(str),
            "subject_id": source["subject_id"].astype(str),
            "context_id": source["context_id"].astype(str),
            "context_json": context_json,
            "sender": source["sender"].astype(str),
            "receiver": source["receiver"].astype(str),
            "interaction_id": source["interaction_id"].astype(str),
            "native_interaction_id": source["source_interaction_id"].astype(str),
            "ligand": canonical_entities["ligand_gene_label"].astype(str),
            "receptor": canonical_entities["receptor_gene_label"].astype(str),
            "ligand_availability": source["ligand_availability"],
            "receptor_availability": source["receptor_availability"],
            "availability_state": source["availability_state"],
            "status": source["state_status"].astype(str),
            "reason_code": source["state_reason_code"].astype("object"),
        }
    )
    return validate_availability_score_table(
        result.sort_values(
            ["sample_id", "sender", "receiver", "interaction_id"],
            kind="stable",
            ignore_index=True,
        )
    )


def _finalize_manifest(
    manifest: dict[str, Any],
    scores: pd.DataFrame,
    mapping_summary: pd.DataFrame,
    output_dir: Path,
    *,
    started: float,
) -> dict[str, Any]:
    scores = validate_availability_score_table(scores)
    parquet_path = output_dir / "availability_scores.parquet"
    tsv_path = output_dir / "availability_scores.tsv"
    mapping_path = output_dir / "availability_mapping_summary.tsv"
    scores.to_parquet(parquet_path, index=False)
    scores.to_csv(tsv_path, sep="\t", index=False, na_rep="")
    mapping_summary.to_csv(mapping_path, sep="\t", index=False)
    manifest.update(
        {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "output": {
                "schema_version": OUTPUT_SCHEMA,
                "table": parquet_path.name,
                "table_sha256": sha256_file(parquet_path),
                "tsv": tsv_path.name,
                "tsv_sha256": sha256_file(tsv_path),
                "mapping_summary": mapping_path.name,
                "mapping_summary_sha256": sha256_file(mapping_path),
                "rows": len(scores),
            },
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def run_availability(
    input_h5ad: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str,
    resource_mode: str,
    context_keys: tuple[str, ...],
    sample_key: str = "sample_id",
    subject_key: str = "subject_id",
    cell_type_key: str = "cell_type",
    counts_layer: str | None = "counts",
    expression_layer: str | None = None,
    expression_source: str | None = None,
    expression_transform: ExpressionTransform | None = None,
    normalized_zero_is_nondetection: bool = False,
    min_cells: int = 10,
    min_pooled_availability: float = 0.0,
    availability_parameters: AvailabilityParameters | None = None,
    harmonized_resource: str | Path | None = None,
    harmonized_manifest: str | Path | None = None,
    native_adapter: str | None = None,
    database_root: str | Path | None = None,
    resource_manifest: str | Path | None = None,
    species: Species | str = Species.HUMAN,
    overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run availability for each sample independently and persist tidy scores."""

    if not dataset_id or dataset_id != dataset_id.strip():
        raise ValueError("dataset_id must be a canonical non-empty identifier")
    selected_context_keys = _canonical_context_keys(tuple(context_keys))
    if isinstance(min_cells, bool) or not isinstance(min_cells, int) or min_cells < 1:
        raise ValueError("min_cells must be an integer >= 1")
    if (
        isinstance(min_pooled_availability, bool)
        or not math.isfinite(min_pooled_availability)
        or not 0 <= min_pooled_availability <= 1
    ):
        raise ValueError("min_pooled_availability must lie in [0, 1]")
    parameters = availability_parameters or AvailabilityParameters()
    if not isinstance(parameters, AvailabilityParameters):
        raise TypeError("availability_parameters must be AvailabilityParameters")
    input_path = Path(input_h5ad).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input h5ad is missing: {input_path}")
    root = Path(repo_root).expanduser().resolve()
    bundle, resource_provenance = _resource_bundle(
        resource_mode=resource_mode,
        harmonized_resource=harmonized_resource,
        harmonized_manifest=harmonized_manifest,
        native_adapter=native_adapter,
        database_root=database_root,
        resource_manifest=resource_manifest,
        species=species,
    )
    output = prepare_output(output_dir, overwrite=overwrite)
    started = time.perf_counter()
    manifest: dict[str, Any] | None = None
    adata: ad.AnnData | None = None
    try:
        adata = ad.read_h5ad(input_path)
        normalized_metadata = _normalize_declared_metadata(
            adata,
            (
                sample_key,
                subject_key,
                cell_type_key,
                *selected_context_keys,
            ),
        )
        sample_metadata = validate_prepared_input(
            adata,
            sample_key=sample_key,
            subject_key=subject_key,
            cell_type_key=cell_type_key,
            context_keys=selected_context_keys,
        )
        method_version = _package_version()
        manifest = begin_manifest(
            repo_root=root,
            dataset_id=dataset_id,
            method={"id": METHOD_ID, "version": method_version},
            environment=python_environment(
                environment_name="crychic_project_uv",
                packages=("CRYCHIC", "anndata", "numpy", "pandas", "pyarrow"),
                threads=1,
            ),
            input_path=input_path,
            input_shape=adata.shape,
            sample_metadata=sample_metadata,
            input_keys={
                "sample_key": sample_key,
                "subject_key": subject_key,
                "cell_type_key": cell_type_key,
                "context_keys": list(selected_context_keys),
                "counts_layer": counts_layer,
                "expression_layer": expression_layer,
                "expression_source": expression_source,
                "expression_transform": expression_transform,
                "normalized_zero_is_nondetection": normalized_zero_is_nondetection,
            },
            resource=resource_provenance,
            parameters={
                "min_cells": min_cells,
                "min_pooled_availability": min_pooled_availability,
                "max_interactions": None,
                "availability": asdict(parameters),
            },
            score_semantics={
                "name": SCORE_NAME,
                "direction": SCORE_DIRECTION,
                "interpretation": "static_sample_level_availability_diagnostic",
                "probability": False,
                "comm_strength": False,
                "full_differential_comm_strength": False,
                "differential_communication": False,
                "single_sample_estimable": True,
                "observed_zero_is_valid": True,
                "unreturned_rows_imputed_as_zero": False,
                "row_scope": "core_state_eligible_mapped_supported_rows",
                "p_value": None,
                "q_value": None,
                "within_dataset_p_value": "not_emitted",
            },
        )
        manifest["statistical_scope"] = {
            "unit": "sample_sender_receiver_interaction",
            "single_sample_estimable": True,
            "between_condition_inference": False,
            "differential_effect": None,
            "differential_p_value": None,
            "differential_q_value": None,
            "reason_code": "static_availability_diagnostic_no_group_inference",
        }
        manifest["input"]["categorical_metadata_cast_to_object"] = list(
            normalized_metadata
        )
        schema = InputSchema(
            context_keys=selected_context_keys,
            counts_layer=counts_layer,
            sample_key=sample_key,
            subject_key=subject_key,
            cell_type_key=cell_type_key,
            expression_layer=expression_layer,
            expression_source=expression_source,
            expression_transform=expression_transform,
            normalized_zero_is_nondetection=normalized_zero_is_nondetection,
            species=bundle.species.value,
            gene_namespace=bundle.gene_namespace.value,
        )
        validated = validate_anndata(adata, schema)
        aggregate = aggregate_pseudobulk(validated, min_cells=min_cells)
        availability = estimate_bundle_availability(
            aggregate,
            bundle,
            context_keys=selected_context_keys,
            parameters=parameters,
            min_pooled_availability=min_pooled_availability,
            max_interactions=None,
        )
        scores = _tidy_scores(
            availability,
            bundle=bundle,
            run_id=str(manifest["run_id"]),
            dataset_id=dataset_id,
            method_version=method_version,
            resource_mode=resource_mode,
            context_keys=selected_context_keys,
        )
        manifest["core_execution"] = {
            "entrypoints": [
                "crychic.data.validate_anndata",
                "crychic.pseudobulk.aggregate_pseudobulk",
                "crychic.availability.estimate_bundle_availability",
            ],
            "input_mode": validated.mode.value,
            "expression_location": validated.report.expression_location,
            "detection_eligible": validated.report.detection_eligible,
            "warnings": list(validated.report.warnings),
            "aggregate_units": len(aggregate.matrix_unit_ids),
            "detection_available": availability.detection_available,
            "filter_application": availability.filter_application.value,
            "filter_universe": availability.frozen_interaction_universe.to_dict(),
            "application_subject_count": len(availability.application_subject_ids),
            "mapping_summary": availability.mapping_summary.to_dict(orient="records"),
        }
        return _finalize_manifest(
            manifest,
            scores,
            availability.mapping_summary,
            output,
            started=started,
        )
    except Exception as exc:
        if manifest is not None:
            fail_manifest(manifest, output, exc, started=started)
        raise
    finally:
        if adata is not None and adata.isbacked:
            adata.file.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--resource-mode", choices=("H-common", "native"), required=True
    )
    parser.add_argument("--context-key", action="append", required=True)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--subject-key", default="subject_id")
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--min-pooled-availability", type=float, default=0.0)
    parser.add_argument("--harmonized-resource", type=Path)
    parser.add_argument("--harmonized-manifest", type=Path)
    parser.add_argument("--native-adapter", choices=("cellchat", "cellphonedb"))
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--resource-manifest", type=Path)
    parser.add_argument("--species", choices=("human", "mouse"), default="human")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run_availability(
        args.input_h5ad,
        args.output_dir,
        dataset_id=args.dataset_id,
        resource_mode=args.resource_mode,
        context_keys=tuple(args.context_key),
        sample_key=args.sample_key,
        subject_key=args.subject_key,
        cell_type_key=args.cell_type_key,
        counts_layer=args.counts_layer,
        min_cells=args.min_cells,
        min_pooled_availability=args.min_pooled_availability,
        harmonized_resource=args.harmonized_resource,
        harmonized_manifest=args.harmonized_manifest,
        native_adapter=args.native_adapter,
        database_root=args.database_root,
        resource_manifest=args.resource_manifest,
        species=args.species,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": manifest["status"], "output": manifest["output"]}))


if __name__ == "__main__":
    main()


__all__ = [
    "METHOD_ID",
    "OUTPUT_SCHEMA",
    "SCORE_COLUMNS",
    "run_availability",
    "validate_availability_score_table",
]
