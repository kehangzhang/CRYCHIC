"""Prepare checksum-bound MIS-C multi-group inputs and the four-method LR axis."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-misc-multigroup-benchmark-v1"
CONTRASTS = (
    ("M_vs_S", "MIS-C", "healthy_sibling"),
    ("C_vs_S", "adult_severe_COVID19", "healthy_sibling"),
    ("M_vs_C", "MIS-C", "adult_severe_COVID19"),
)


def _json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _source_h5ad_record(
    manifest: Mapping[str, Any], filename: str
) -> Mapping[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("source preparation manifest has no outputs mapping")
    record = outputs.get("h5ad")
    if not isinstance(record, Mapping) or record.get("filename") != filename:
        raise ValueError("source preparation manifest does not bind the H5AD")
    return record


def _common_resource(
    harmonized_path: Path,
    harmonized_manifest_path: Path,
    connectome_path: Path,
    connectome_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    harmonized = pd.read_csv(harmonized_path, sep="\t", dtype=str).fillna("")
    connectome = pd.read_csv(connectome_path, sep="\t", dtype=str).fillna("")
    required = {"harmonized_interaction_id", "ligand", "receptor"}
    for label, table in (("harmonized", harmonized), ("connectome", connectome)):
        missing = required.difference(table.columns)
        if missing or table.empty:
            raise ValueError(f"{label} resource is invalid: missing={sorted(missing)}")
        if table.duplicated(["ligand", "receptor"]).any():
            raise ValueError(f"{label} resource contains duplicate molecular LR pairs")
    connectome_ids = connectome.loc[
        :, ["ligand", "receptor", "harmonized_interaction_id"]
    ].rename(columns={"harmonized_interaction_id": "connectome_interaction_id"})
    common = harmonized.merge(
        connectome_ids,
        on=["ligand", "receptor"],
        how="inner",
        validate="one_to_one",
    )
    if common.empty:
        raise ValueError("the harmonized and ConnectomeDB resources do not overlap")
    common["liana_source_interaction_id"] = common["connectome_interaction_id"]
    common["liana_covered"] = "True"
    common["scseqcommdiff_source_interaction_id"] = common["connectome_interaction_id"]
    common["scseqcommdiff_covered"] = "True"
    common = common.drop(columns="connectome_interaction_id").sort_values(
        ["ligand", "receptor"], kind="stable", ignore_index=True
    )
    source_manifest = _json_object(harmonized_manifest_path)
    connectome_manifest = _json_object(connectome_manifest_path)
    manifest = {
        "schema_version": "crychic-harmonized-lr-v1",
        "resource_id": "crychic_misc_four_method_harmonized_simple_lr",
        "version": "2026-07-23",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "license": "intersection-only; retain upstream method licenses",
        "citation": [
            *list(source_manifest.get("citation", [])),
            str(connectome_manifest.get("citation", "")),
        ],
        "construction": {
            "rule": (
                "exact directed ligand/receptor HGNC match between the frozen "
                "CellChat/CellPhoneDB simple H-common axis and ConnectomeDB2020"
            ),
            "source_harmonized_rows": len(harmonized),
            "source_connectome_rows": len(connectome),
            "retained_interactions": len(common),
        },
        "sources": {
            "harmonized_table_sha256": sha256_file(harmonized_path),
            "harmonized_manifest_sha256": sha256_file(harmonized_manifest_path),
            "connectome_table_sha256": sha256_file(connectome_path),
            "connectome_manifest_sha256": sha256_file(connectome_manifest_path),
        },
    }
    return common, manifest


def _crychic_spec(
    *, dataset_id: str, input_path: Path, input_sha256: str
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "input": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "output_name": dataset_id,
        "input_mode": "counts",
        "lr_resource": "misc_four_method_common",
        "target_prior": "nichenet_human",
        "benchmark_scope": "independent_subject_multigroup_hcommon",
        "config": {
            "context_keys": ["condition"],
            "counts_layer": "counts",
            "sample_key": "sample_id",
            "subject_key": "subject_id",
            "cell_type_key": "cell_type",
            "species": "human",
            "gene_namespace": "hgnc_symbol",
            "design": "~ condition",
            "communication_modes": ["state"],
            "random_seed": 20260723,
        },
        "workflow": {
            "min_cells": 10,
            "min_samples_per_context": 4,
            "min_subjects_per_context": 4,
            "min_pooled_availability": 0.0,
            "max_interactions": None,
            "subject_fixed_effects": False,
            "lambda1": 0.0,
            "lambda2": 0.0,
            "cosine_threshold": 0.95,
            "prior_quality": 1.0,
        },
    }


def prepare(
    input_h5ad: Path,
    input_manifest: Path,
    harmonized_resource: Path,
    harmonized_manifest: Path,
    connectome_resource: Path,
    connectome_manifest: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = tuple(
        path.resolve()
        for path in (
            input_h5ad,
            input_manifest,
            harmonized_resource,
            harmonized_manifest,
            connectome_resource,
            connectome_manifest,
        )
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    (
        input_h5ad,
        input_manifest,
        harmonized_resource,
        harmonized_manifest,
        connectome_resource,
        connectome_manifest,
    ) = paths
    source_manifest = _json_object(input_manifest)
    source_record = _source_h5ad_record(source_manifest, input_h5ad.name)
    if sha256_file(input_h5ad) != source_record.get("sha256"):
        raise ValueError("source H5AD checksum mismatch")
    common, resource_manifest = _common_resource(
        harmonized_resource,
        harmonized_manifest,
        connectome_resource,
        connectome_manifest,
    )
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    source: ad.AnnData | None = None
    try:
        resource_dir = staged / "resource"
        inputs_dir = staged / "inputs"
        resource_dir.mkdir()
        inputs_dir.mkdir()
        common_path = resource_dir / "harmonized_lr.tsv"
        common.to_csv(common_path, sep="\t", index=False, lineterminator="\n")
        resource_manifest["payload"] = {
            "filename": common_path.name,
            "rows": len(common),
            "bytes": common_path.stat().st_size,
            "sha256": sha256_file(common_path),
        }
        resource_manifest_path = resource_dir / "manifest.json"
        _write_json(resource_manifest_path, resource_manifest)

        source = ad.read_h5ad(input_h5ad)
        required_obs = {
            "sample_id",
            "subject_id",
            "family_id",
            "condition",
            "cell_type",
        }
        missing_obs = required_obs.difference(source.obs.columns)
        if missing_obs:
            raise ValueError(f"MIS-C H5AD metadata is missing: {sorted(missing_obs)}")
        if "counts" not in source.layers:
            raise ValueError("MIS-C H5AD must retain raw counts in layers['counts']")
        x_values = source.X.data if hasattr(source.X, "data") else np.asarray(source.X)
        count_matrix = source.layers["counts"]
        count_values = (
            count_matrix.data
            if hasattr(count_matrix, "data")
            else np.asarray(count_matrix)
        )
        if (
            not np.isfinite(x_values).all()
            or (x_values < 0).any()
            or np.allclose(x_values, np.round(x_values))
        ):
            raise ValueError("MIS-C X must contain finite non-negative logcounts")
        if (
            not np.isfinite(count_values).all()
            or (count_values < 0).any()
            or not np.allclose(count_values, np.round(count_values))
        ):
            raise ValueError("MIS-C counts layer must contain raw integer values")
        if (
            source.obs["subject_id"]
            .astype(str)
            .groupby(source.obs["condition"].astype(str), observed=True)
            .nunique()
            .sum()
            != source.obs["subject_id"].nunique()
        ):
            raise ValueError("MIS-C subjects cannot occur in multiple conditions")

        records: list[dict[str, Any]] = []
        config_datasets: dict[str, Any] = {
            "misc_olink": _crychic_spec(
                dataset_id="misc_olink",
                input_path=input_h5ad,
                input_sha256=sha256_file(input_h5ad),
            )
        }
        for contrast, target, reference in CONTRASTS:
            selected = source[
                source.obs["condition"].astype(str).isin((target, reference))
            ].copy()
            path = inputs_dir / f"misc_olink.{contrast}.h5ad"
            selected.write_h5ad(path, compression="gzip")
            digest = sha256_file(path)
            pair_manifest = {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": "misc_olink",
                "contrast": contrast,
                "target": target,
                "reference": reference,
                "inferential_unit": "subject_id",
                "family_block": "family_id",
                "output": {
                    "filename": path.name,
                    "sha256": digest,
                    "shape": [int(selected.n_obs), int(selected.n_vars)],
                },
            }
            manifest_path = inputs_dir / f"{contrast}.manifest.json"
            _write_json(manifest_path, pair_manifest)
            records.append(
                {
                    "contrast": contrast,
                    "target": target,
                    "reference": reference,
                    "h5ad": str(path.relative_to(staged)),
                    "manifest": str(manifest_path.relative_to(staged)),
                    "sha256": digest,
                    "shape": [int(selected.n_obs), int(selected.n_vars)],
                }
            )
            pair_dataset_id = f"misc_olink__{contrast}"
            config_datasets[pair_dataset_id] = _crychic_spec(
                dataset_id=pair_dataset_id,
                input_path=(output_dir / path.relative_to(staged)).resolve(),
                input_sha256=digest,
            )

        repo_root = Path(__file__).resolve().parents[2]
        config = {
            "schema_version": "canonical-v0.1",
            "database_root": str((repo_root / "../databases").resolve()),
            "output_root": str((output_dir.parent / "runs/crychic_misc").resolve()),
            "resources": {
                "misc_four_method_common": {
                    "adapter": "harmonized_fixture",
                    "manifest": str(
                        (
                            output_dir / resource_manifest_path.relative_to(staged)
                        ).resolve()
                    ),
                    "species": "human",
                },
                "nichenet_human": {
                    "manifest": str(
                        (repo_root / "resources/nichenet_human_v2_2021.json").resolve()
                    ),
                    "release": "v2_2021",
                },
            },
            "datasets": config_datasets,
        }
        config_path = staged / "crychic_config.json"
        _write_json(config_path, config)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "code": git_metadata(repo_root),
            "dataset_id": "misc_olink",
            "source_h5ad": {
                "filename": input_h5ad.name,
                "sha256": sha256_file(input_h5ad),
                "shape": [int(source.n_obs), int(source.n_vars)],
            },
            "expression": {
                "external_method_matrix": "X_logcounts",
                "crychic_matrix": "layers/counts_raw_integer",
            },
            "contrasts": records,
            "resource": {
                "table": str(common_path.relative_to(staged)),
                "manifest": str(resource_manifest_path.relative_to(staged)),
                "rows": len(common),
                "ligands": int(common["ligand"].nunique()),
                "sha256": sha256_file(common_path),
            },
            "crychic_config": {
                "filename": config_path.name,
                "sha256": sha256_file(config_path),
            },
        }
        _write_json(staged / "manifest.json", manifest)
        if output_dir.exists() and not overwrite:
            raise FileExistsError(f"output exists: {output_dir}; pass --overwrite")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staged, output_dir)
        published = True
        return manifest
    finally:
        if source is not None and source.isbacked:
            source.file.close()
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--harmonized-resource", required=True, type=Path)
    parser.add_argument("--harmonized-manifest", required=True, type=Path)
    parser.add_argument("--connectome-resource", required=True, type=Path)
    parser.add_argument("--connectome-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prepare(
        args.input_h5ad,
        args.input_manifest,
        args.harmonized_resource,
        args.harmonized_manifest,
        args.connectome_resource,
        args.connectome_manifest,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
