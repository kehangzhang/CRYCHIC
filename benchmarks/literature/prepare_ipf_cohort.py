"""Build validated per-sample AnnData inputs for the Xie IPF cohort.

The companion R exporter reads the four published Seurat objects and writes a
small, language-neutral sparse bundle for every patient.  This module performs
the benchmark-facing normalization, validates the frozen 56-sample roster, and
records one checksum-bound row per sample.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from benchmarks.adapters.common import sha256_file, write_json

IPF_CELL_TYPES = (
    "AT1",
    "AT2",
    "Endothelial",
    "Fibroblast",
    "Macrophage",
    "Mast",
    "Monocyte",
    "Tcell",
)
SAMPLE_MANIFEST_SCHEMA = "xie-ipf-cohort-sample-manifest-v1"
SAMPLE_MANIFEST_COLUMNS = (
    "study_id",
    "sample_id",
    "subject_id",
    "condition",
    "dataset_id",
    "input_h5ad",
    "input_sha256",
    "n_cells",
    "n_genes",
    "n_cell_types",
    "cell_types_json",
    "source_bundle_manifest",
    "source_bundle_manifest_sha256",
    "status",
)


def _canonical_roster(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype=str)
    required = {"geo_accession", "sample_id"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"IPF roster is missing columns: {sorted(missing)}")
    roster = table.loc[:, ["geo_accession", "sample_id"]].rename(
        columns={"geo_accession": "study_id"}
    )
    if roster.isna().any().any():
        raise ValueError("IPF roster identifiers must not be missing")
    for column in ("study_id", "sample_id"):
        roster[column] = roster[column].astype(str).str.strip()
        if roster[column].eq("").any():
            raise ValueError("IPF roster identifiers must not be empty")
    if roster.duplicated(["study_id", "sample_id"]).any():
        raise ValueError("IPF roster contains duplicate study/sample keys")
    return roster.sort_values(
        ["study_id", "sample_id"], kind="stable", ignore_index=True
    )


def _read_bundle(
    bundle_dir: Path,
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = {
        "counts": bundle_dir / "counts.mtx.gz",
        "features": bundle_dir / "features.tsv",
        "cells": bundle_dir / "cells.tsv",
        "manifest": bundle_dir / "manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("sample bundle is incomplete: " + ", ".join(missing))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"sample bundle is not complete: {paths['manifest']}")
    records = manifest.get("outputs")
    if not isinstance(records, dict):
        raise ValueError("sample bundle manifest lacks output checksums")
    for key in ("counts", "features", "cells"):
        record = records.get(key)
        if not isinstance(record, dict) or record.get("sha256") != sha256_file(
            paths[key]
        ):
            raise ValueError(f"sample bundle checksum mismatch for {key}")
    with gzip.open(paths["counts"], "rb") as handle:
        feature_by_cell = mmread(handle, spmatrix=True)
    if not sparse.issparse(feature_by_cell):
        raise TypeError("sample bundle counts must be sparse")
    counts = feature_by_cell.transpose().tocsr()
    if counts.data.size and (
        np.any(~np.isfinite(counts.data))
        or np.any(counts.data < 0)
        or np.any(counts.data != np.rint(counts.data))
    ):
        raise ValueError("sample bundle counts must be finite non-negative integers")
    counts = counts.astype(np.int64, copy=False)
    features = pd.read_csv(paths["features"], sep="\t", dtype=str)
    cells = pd.read_csv(paths["cells"], sep="\t", dtype=str)
    return counts, features, cells, manifest


def build_sample_h5ad(
    bundle_dir: str | Path,
    output_h5ad: str | Path,
    *,
    expected_study: str | None = None,
    expected_sample: str | None = None,
) -> dict[str, Any]:
    """Convert one checksum-bound sparse bundle to the benchmark H5AD contract."""

    source = Path(bundle_dir).expanduser().resolve()
    output = Path(output_h5ad).expanduser().resolve()
    manifest_path = output.with_name("manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing sample input: {output}")
    counts, features, cells, bundle_manifest = _read_bundle(source)
    if list(features.columns) != ["gene_symbol"]:
        raise ValueError("features.tsv must contain exactly gene_symbol")
    required_cells = {"barcode", "study_id", "sample_id", "cell_type"}
    if set(cells.columns) != required_cells:
        raise ValueError("cells.tsv has an unexpected schema")
    if len(features) != counts.shape[1] or len(cells) != counts.shape[0]:
        raise ValueError("sample bundle matrix dimensions disagree with metadata")
    if features["gene_symbol"].isna().any() or features["gene_symbol"].eq("").any():
        raise ValueError("gene symbols must not be missing or empty")
    if features["gene_symbol"].duplicated().any():
        raise ValueError("gene symbols must be unique; aggregation was not predeclared")
    if cells["barcode"].isna().any() or cells["barcode"].duplicated().any():
        raise ValueError("cell barcodes must be complete and unique")
    studies = cells["study_id"].drop_duplicates().tolist()
    samples = cells["sample_id"].drop_duplicates().tolist()
    if len(studies) != 1 or len(samples) != 1:
        raise ValueError("each sparse bundle must contain exactly one study and sample")
    study_id, sample_id = str(studies[0]), str(samples[0])
    if expected_study is not None and study_id != expected_study:
        raise ValueError(f"bundle study {study_id!r} != expected {expected_study!r}")
    if expected_sample is not None and sample_id != expected_sample:
        raise ValueError(f"bundle sample {sample_id!r} != expected {expected_sample!r}")
    observed_types = set(cells["cell_type"].astype(str))
    if not observed_types or not observed_types.issubset(IPF_CELL_TYPES):
        raise ValueError(f"unexpected IPF cell types: {sorted(observed_types)}")
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    if np.any(totals <= 0):
        raise ValueError("all retained cells must have positive library sizes")
    normalized = (
        counts.astype(np.float32).multiply((10_000.0 / totals)[:, None]).tocsr()
    )
    normalized.data = np.log1p(normalized.data)
    obs = cells.set_index("barcode", drop=True).copy()
    obs["subject_id"] = sample_id
    obs["condition"] = "IPF"
    for column in obs.columns:
        # AnnData gates pandas nullable-string serialization behind a global
        # opt-in.  Plain object strings preserve the declared metadata without
        # depending on that process-global setting or categorical conversion.
        obs[column] = obs[column].astype(str).astype(object)
    genes = features["gene_symbol"].astype(str)
    var = pd.DataFrame(index=pd.Index(genes, name="gene_symbol"))
    adata = ad.AnnData(X=normalized, obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    dataset_id = f"{study_id}_{sample_id}_IPF_8cell"
    adata.uns.update(
        {
            "dataset_id": dataset_id,
            "source_geo_accession": study_id,
            "source_condition": "IPF",
            "benchmark_scope": "single-sample static interaction scoring",
            "source_bundle_manifest_sha256": sha256_file(source / "manifest.json"),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    try:
        adata.write_h5ad(
            temporary, compression="gzip", convert_strings_to_categoricals=False
        )
        check = ad.read_h5ad(temporary, backed="r")
        try:
            if check.shape != adata.shape or "counts" not in check.layers:
                raise RuntimeError("sample H5AD readback validation failed")
            if check.obs_names.tolist() != adata.obs_names.tolist():
                raise RuntimeError("sample H5AD cell order changed during write")
            if check.var_names.tolist() != adata.var_names.tolist():
                raise RuntimeError("sample H5AD feature order changed during write")
        finally:
            check.file.close()
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "schema_version": "xie-ipf-cohort-sample-input-v1",
        "status": "complete",
        "study_id": study_id,
        "sample_id": sample_id,
        "subject_id": sample_id,
        "condition": "IPF",
        "dataset_id": dataset_id,
        "shape_cells_by_genes": [adata.n_obs, adata.n_vars],
        "counts_nnz": int(counts.nnz),
        "cell_type_counts": {
            str(key): int(value)
            for key, value in obs["cell_type"].value_counts(sort=False).items()
        },
        "source_bundle": {
            "manifest": str((source / "manifest.json").resolve()),
            "manifest_sha256": sha256_file(source / "manifest.json"),
            "schema_version": bundle_manifest.get("schema_version"),
        },
        "output": {
            "filename": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "normalization": {
            "X": "per-cell total count 10000 followed by log1p",
            "counts_layer": "raw integer counts",
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def prepare_ipf_cohort(
    export_root: str | Path,
    roster_path: str | Path,
    output_root: str | Path,
    *,
    studies: Iterable[str] | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Build all rostered sample inputs and write the canonical sample manifest."""

    exports = Path(export_root).expanduser().resolve()
    roster_file = Path(roster_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    roster = _canonical_roster(roster_file)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if studies is not None:
        selected_studies = tuple(dict.fromkeys(str(value) for value in studies))
        if not selected_studies or any(
            not value or value != value.strip() for value in selected_studies
        ):
            raise ValueError("studies must contain canonical non-empty identifiers")
        unknown = set(selected_studies).difference(roster["study_id"])
        if unknown:
            raise ValueError(
                f"requested studies are absent from roster: {sorted(unknown)}"
            )
        roster = roster.loc[roster["study_id"].isin(selected_studies)].reset_index(
            drop=True
        )
    jobs = [
        (
            exports / str(row.study_id) / str(row.sample_id),
            output
            / str(row.study_id)
            / str(row.sample_id)
            / "input"
            / f"{row.study_id}_{row.sample_id}_IPF_8cell_hgnc_log1p.h5ad",
            str(row.study_id),
            str(row.sample_id),
        )
        for row in roster.itertuples(index=False)
    ]
    if workers == 1:
        manifests = [_prepare_sample_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            manifests = list(executor.map(_prepare_sample_job, jobs))
    records: list[dict[str, object]] = []
    for manifest in manifests:
        study_id = str(manifest["study_id"])
        sample_id = str(manifest["sample_id"])
        h5ad_path = (
            output
            / study_id
            / sample_id
            / "input"
            / f"{study_id}_{sample_id}_IPF_8cell_hgnc_log1p.h5ad"
        )
        cell_counts = manifest["cell_type_counts"]
        records.append(
            {
                "study_id": study_id,
                "sample_id": sample_id,
                "subject_id": sample_id,
                "condition": "IPF",
                "dataset_id": manifest["dataset_id"],
                "input_h5ad": str(h5ad_path.resolve()),
                "input_sha256": manifest["output"]["sha256"],
                "n_cells": manifest["shape_cells_by_genes"][0],
                "n_genes": manifest["shape_cells_by_genes"][1],
                "n_cell_types": len(cell_counts),
                "cell_types_json": json.dumps(
                    sorted(cell_counts), separators=(",", ":")
                ),
                "source_bundle_manifest": manifest["source_bundle"]["manifest"],
                "source_bundle_manifest_sha256": manifest["source_bundle"][
                    "manifest_sha256"
                ],
                "status": "complete",
            }
        )
    result = pd.DataFrame.from_records(records, columns=SAMPLE_MANIFEST_COLUMNS)
    if len(result) != len(roster):
        raise RuntimeError("prepared sample count differs from frozen roster")
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "sample_manifest.tsv"
    temporary = table_path.with_suffix(".tsv.tmp")
    result.to_csv(temporary, sep="\t", index=False)
    temporary.replace(table_path)
    write_json(
        output / "sample_manifest.json",
        {
            "schema_version": SAMPLE_MANIFEST_SCHEMA,
            "status": "complete",
            "roster": {"path": str(roster_file), "sha256": sha256_file(roster_file)},
            "samples": len(result),
            "studies": {
                str(key): int(value)
                for key, value in result["study_id"].value_counts().sort_index().items()
            },
            "table": {"filename": table_path.name, "sha256": sha256_file(table_path)},
        },
    )
    return result


def _prepare_sample_job(
    job: tuple[Path, Path, str, str],
) -> dict[str, Any]:
    bundle, h5ad_path, study_id, sample_id = job
    manifest_path = h5ad_path.with_name("manifest.json")
    if h5ad_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest.get("output", {})
        expected_dataset = f"{study_id}_{sample_id}_IPF_8cell"
        identity_matches = (
            manifest.get("study_id") == study_id
            and manifest.get("sample_id") == sample_id
            and manifest.get("subject_id") == sample_id
            and manifest.get("dataset_id") == expected_dataset
        )
        if (
            manifest.get("status") != "complete"
            or not identity_matches
            or record.get("sha256") != sha256_file(h5ad_path)
        ):
            raise ValueError(
                f"existing sample input failed validation: {h5ad_path.parent}"
            )
        return manifest
    if h5ad_path.exists() or manifest_path.exists():
        raise FileExistsError(f"partial sample input exists: {h5ad_path.parent}")
    return build_sample_h5ad(
        bundle,
        h5ad_path,
        expected_study=study_id,
        expected_sample=sample_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_root", type=Path)
    parser.add_argument("roster", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--study", action="append")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = prepare_ipf_cohort(
        args.export_root,
        args.roster,
        args.output_root,
        studies=args.study,
        workers=args.workers,
    )
    print(json.dumps({"status": "complete", "samples": len(result)}))


if __name__ == "__main__":
    main()


__all__ = [
    "IPF_CELL_TYPES",
    "SAMPLE_MANIFEST_COLUMNS",
    "build_sample_h5ad",
    "prepare_ipf_cohort",
]
