"""Export the frozen Ji cSCC subset shared by all benchmark methods."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    load_harmonized_resource,
    sha256_file,
    write_json,
)
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.run_cscc_paired_gate_smoke import (
    DEFAULT_CONFIG,
    DEFAULT_WORKSPACE_ROOT,
    load_smoke_config,
    prepare_input,
)
from crychic.core import canonical_digest

SCHEMA_VERSION = "crychic-cscc-crossmethod-inputs-v1"
DEFAULT_OUTPUT_DIR = Path("benchmark_work/cscc_crossmethod_smoke_inputs")
H5AD_FILENAME = "cscc_paired_gate_capped.h5ad"
RESOURCE_FILENAME = "harmonized_lr.tsv"
MANIFEST_FILENAME = "manifest.json"
REQUIRED_RESOURCE_COLUMNS = (
    "harmonized_interaction_id",
    "ligand",
    "receptor",
    "cellchat_source_interaction_id",
    "cellphonedb_source_interaction_id",
)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def diagnostic_interaction_ids(
    definitions: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return non-empty, unique interaction IDs in frozen config order."""

    interaction_ids = tuple(
        str(definition.get("interaction_id", "")).strip() for definition in definitions
    )
    if not interaction_ids or any(not item for item in interaction_ids):
        raise ValueError("diagnostic interaction IDs must be non-empty")
    if len(set(interaction_ids)) != len(interaction_ids):
        raise ValueError("diagnostic interaction IDs must be unique")
    return interaction_ids


def select_harmonized_rows(
    source: pd.DataFrame,
    interaction_ids: Sequence[str],
) -> pd.DataFrame:
    """Select exactly one source row per ID in requested, source-independent order."""

    missing_columns = set(REQUIRED_RESOURCE_COLUMNS).difference(source.columns)
    if missing_columns:
        raise ValueError(
            f"harmonized source is missing columns: {sorted(missing_columns)}"
        )
    requested = tuple(str(item).strip() for item in interaction_ids)
    if not requested or any(not item for item in requested):
        raise ValueError("requested interaction IDs must be non-empty")
    if len(set(requested)) != len(requested):
        raise ValueError("requested interaction IDs contain duplicates")

    id_column = source["harmonized_interaction_id"].astype(str)
    duplicate_ids = sorted(id_column[id_column.duplicated(keep=False)].unique())
    if duplicate_ids:
        raise ValueError(f"harmonized source contains duplicate IDs: {duplicate_ids}")
    indexed = source.copy()
    indexed["harmonized_interaction_id"] = id_column
    indexed = indexed.set_index("harmonized_interaction_id", drop=False)
    missing_ids = sorted(set(requested).difference(indexed.index))
    if missing_ids:
        raise ValueError(f"harmonized source lacks requested IDs: {missing_ids}")

    selected = indexed.loc[list(requested)].reset_index(drop=True)
    if len(selected) != len(requested):
        raise RuntimeError("harmonized selection is not one-to-one")
    return selected.loc[:, source.columns].copy()


def validate_selected_definitions(
    selected: pd.DataFrame,
    definitions: Sequence[Mapping[str, object]],
) -> None:
    """Check that config IDs still name the frozen molecular/source identities."""

    expected = [
        (
            str(item["interaction_id"]),
            str(item["ligand"]),
            str(item["receptor"]),
            str(item["cellchat_source_interaction_id"]),
            str(item["cellphonedb_source_interaction_id"]),
        )
        for item in definitions
    ]
    observed = list(
        selected.loc[:, REQUIRED_RESOURCE_COLUMNS].itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError("selected harmonized rows do not match config identities")


def build_export_manifest(
    *,
    source_manifest: Mapping[str, object],
    source_table_sha256: str,
    source_manifest_sha256: str,
    table_payload: Mapping[str, object],
    dataset_payload: Mapping[str, object],
    selected_interaction_ids: Sequence[str],
) -> dict[str, object]:
    """Build a loader-compatible manifest for the exact cSCC resource subset."""

    required_metadata = (
        "resource_id",
        "version",
        "species",
        "gene_namespace",
        "license",
        "citation",
    )
    missing = [key for key in required_metadata if not source_manifest.get(key)]
    if missing:
        raise ValueError(f"source manifest lacks metadata: {missing}")
    selected_ids = tuple(selected_interaction_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected interaction IDs must be unique")
    if _integer(table_payload.get("rows"), field="table payload rows") != len(
        selected_ids
    ):
        raise ValueError("table payload row count does not match selected IDs")

    return {
        "schema_version": "crychic-harmonized-lr-v1",
        "resource_id": source_manifest["resource_id"],
        "version": source_manifest["version"],
        "species": source_manifest["species"],
        "gene_namespace": source_manifest["gene_namespace"],
        "license": source_manifest["license"],
        "citation": source_manifest["citation"],
        "construction": {
            "rule": "exact config-ordered cSCC diagnostic interaction ID subset",
            "source_manifest_sha256": source_manifest_sha256,
            "source_payload_sha256": source_table_sha256,
            "retained_interactions": len(selected_ids),
            "selected_interaction_id_digest": canonical_digest(sorted(selected_ids)),
        },
        "payload": dict(table_payload),
        "analysis_input": dict(dataset_payload),
    }


def _validate_adata(adata: ad.AnnData) -> None:
    if adata.X is None:
        raise ValueError("cross-method input must retain X")
    if adata.obs_names.has_duplicates or adata.var_names.has_duplicates:
        raise ValueError("cross-method input axes must be unique")
    if "counts" not in adata.layers:
        raise ValueError("cross-method input must retain the counts layer")
    counts = adata.layers["counts"]
    if getattr(counts, "format", None) != "csr" or counts.dtype != np.dtype("int32"):
        raise ValueError("cross-method counts must remain CSR int32")


def _validate_h5ad_readback(path: Path, source: ad.AnnData) -> None:
    readback = ad.read_h5ad(path, backed="r")
    try:
        if readback.X is None or readback.shape != source.shape:
            raise RuntimeError("H5AD readback changed X or shape")
        if tuple(map(str, readback.obs_names)) != tuple(map(str, source.obs_names)):
            raise RuntimeError("H5AD readback changed the observation axis")
        if tuple(map(str, readback.var_names)) != tuple(map(str, source.var_names)):
            raise RuntimeError("H5AD readback changed the variable axis")
        if tuple(readback.obs.columns) != tuple(source.obs.columns):
            raise RuntimeError("H5AD readback changed observation columns")
        if tuple(readback.var.columns) != tuple(source.var.columns):
            raise RuntimeError("H5AD readback changed variable columns")
        counts = readback.layers["counts"]
        if getattr(counts, "format", None) != "csr" or counts.dtype != np.dtype(
            "int32"
        ):
            raise RuntimeError("H5AD readback changed counts storage")
    finally:
        readback.file.close()


def write_crossmethod_inputs(
    *,
    adata: ad.AnnData,
    selected_resource: pd.DataFrame,
    source_manifest: Mapping[str, object],
    source_table_sha256: str,
    source_manifest_sha256: str,
    selected_interaction_ids: Sequence[str],
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write and immediately read back one small cross-method input bundle."""

    _validate_adata(adata)
    expected_ids = tuple(selected_interaction_ids)
    if len(selected_resource) != len(expected_ids):
        raise ValueError("selected resource row count does not match requested IDs")
    validated_resource = select_harmonized_rows(selected_resource, expected_ids)
    observed_input_ids = tuple(
        selected_resource["harmonized_interaction_id"].astype(str)
    )
    if observed_input_ids != expected_ids:
        raise ValueError("selected resource is not in requested config order")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = output_dir / H5AD_FILENAME
    table_path = output_dir / RESOURCE_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    adata.write_h5ad(h5ad_path, compression="gzip", compression_opts=4)
    _validate_h5ad_readback(h5ad_path, adata)
    validated_resource.to_csv(table_path, sep="\t", index=False)

    table_payload = {
        "filename": table_path.name,
        "bytes": table_path.stat().st_size,
        "sha256": sha256_file(table_path),
        "rows": len(validated_resource),
    }
    dataset_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "GSE144236_Ji_cSCC",
        "filename": h5ad_path.name,
        "bytes": h5ad_path.stat().st_size,
        "sha256": sha256_file(h5ad_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "obs_axis_digest": canonical_digest(list(map(str, adata.obs_names))),
        "var_axis_digest": canonical_digest(list(map(str, adata.var_names))),
        "x_retained": True,
        "counts_layer": "counts",
        "counts_storage": "csr",
        "counts_dtype": "int32",
    }
    manifest = build_export_manifest(
        source_manifest=source_manifest,
        source_table_sha256=source_table_sha256,
        source_manifest_sha256=source_manifest_sha256,
        table_payload=table_payload,
        dataset_payload=dataset_payload,
        selected_interaction_ids=selected_interaction_ids,
    )
    write_json(manifest_path, manifest)

    loaded_table, loaded_manifest = load_harmonized_resource(table_path, manifest_path)
    bundle = harmonized_resource_bundle(table_path, manifest_path)
    observed_ids = tuple(loaded_table["harmonized_interaction_id"].astype(str))
    bundle_ids = tuple(item.interaction_id for item in bundle.interactions)
    if observed_ids != expected_ids:
        raise RuntimeError("harmonized table readback changed interaction order")
    if len(bundle_ids) != len(expected_ids) or set(bundle_ids) != set(expected_ids):
        raise RuntimeError("harmonized bundle readback changed interaction IDs")
    if loaded_manifest != manifest:
        raise RuntimeError("harmonized manifest readback changed content")
    return {
        "output_dir": str(output_dir.resolve()),
        "h5ad": dataset_payload,
        "harmonized_resource": table_payload,
        "manifest_sha256": sha256_file(manifest_path),
    }


def export_cscc_crossmethod_inputs(
    *,
    workspace_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, object]:
    """Prepare the frozen cSCC cells and exact 14-edge H-common input bundle."""

    workspace_root = workspace_root.resolve()
    config = load_smoke_config(config_path)
    dataset = _mapping(config["dataset"], field="dataset")
    resource = _mapping(config["harmonized_resource"], field="harmonized_resource")
    definitions = tuple(
        _mapping(item, field="diagnostic_interactions[]")
        for item in cast(Sequence[object], config["diagnostic_interactions"])
    )
    interaction_ids = diagnostic_interaction_ids(definitions)

    source_table_path = workspace_root / str(resource["table_relative_path"])
    source_manifest_path = workspace_root / str(resource["manifest_relative_path"])
    if source_table_path.stat().st_size != _integer(
        resource["table_bytes"], field="harmonized_resource.table_bytes"
    ):
        raise ValueError("source harmonized table byte size changed")
    if source_manifest_path.stat().st_size != _integer(
        resource["manifest_bytes"], field="harmonized_resource.manifest_bytes"
    ):
        raise ValueError("source harmonized manifest byte size changed")
    source_table_sha256 = sha256_file(source_table_path)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    if source_table_sha256 != resource["table_sha256"]:
        raise ValueError("source harmonized table checksum changed")
    if source_manifest_sha256 != resource["manifest_sha256"]:
        raise ValueError("source harmonized manifest checksum changed")

    source_table, source_manifest = load_harmonized_resource(
        source_table_path, source_manifest_path
    )
    if len(source_table) != _integer(
        resource["expected_source_interaction_count"],
        field="harmonized_resource.expected_source_interaction_count",
    ):
        raise ValueError("source harmonized interaction count changed")
    if canonical_digest(source_manifest) != resource["manifest_digest"]:
        raise ValueError("source harmonized manifest identity changed")
    if (
        source_manifest.get("resource_id") != resource["resource_id"]
        or source_manifest.get("version") != resource["version"]
    ):
        raise ValueError("source harmonized resource identity changed")

    selected = select_harmonized_rows(source_table, interaction_ids)
    validate_selected_definitions(selected, definitions)
    if len(selected) != _integer(
        resource["expected_selected_interaction_count"],
        field="harmonized_resource.expected_selected_interaction_count",
    ):
        raise ValueError("selected harmonized interaction count changed")
    if (
        canonical_digest(sorted(interaction_ids))
        != resource["expected_selected_id_digest"]
    ):
        raise ValueError("selected harmonized interaction IDs changed")

    input_path = workspace_root / str(dataset["relative_path"])
    adata, _, _ = prepare_input(input_path, dataset_config=dataset)
    resolved_output = output_dir
    if not resolved_output.is_absolute():
        resolved_output = workspace_root / resolved_output
    return write_crossmethod_inputs(
        adata=adata,
        selected_resource=selected,
        source_manifest=cast(Mapping[str, object], source_manifest),
        source_table_sha256=source_table_sha256,
        source_manifest_sha256=source_manifest_sha256,
        selected_interaction_ids=interaction_ids,
        output_dir=resolved_output,
        overwrite=overwrite,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_cscc_crossmethod_inputs(
        workspace_root=args.workspace_root,
        config_path=args.config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
