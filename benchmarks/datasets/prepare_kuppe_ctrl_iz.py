"""Prepare the Kuppe CTRL-versus-IZ differential-CCC benchmark cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

DATASET_ID = "Kuppe_MI_CTRL_vs_IZ"
SOURCE_DATASET_ID = "Kuppe_MI_Zenodo6578047"
CONDITIONS = ("CTRL", "IZ")
AUDIT_SCHEMA = "crychic-kuppe-ctrl-iz-audit-v1"
MANIFEST_SCHEMA = "crychic-kuppe-ctrl-iz-preparation-v1"

CELL_TYPES = (
    "Adipocyte",
    "Cardiomyocyte",
    "Cycling cells",
    "Endothelial",
    "Fibroblast",
    "Lymphoid",
    "Mast",
    "Myeloid",
    "Neuronal",
    "Pericyte",
    "vSMCs",
)

EXPECTED_KUPPE_CTRL_IZ: dict[str, Any] = {
    "source_shape": [191_795, 29_126],
    "n_obs": 76_141,
    "n_vars": 29_126,
    "n_subjects": 11,
    "n_samples": 15,
    "n_cell_types": 11,
    "cell_types": list(CELL_TYPES),
    "cells_by_condition": {"CTRL": 41_663, "IZ": 34_478},
    "samples_by_condition": {"CTRL": 4, "IZ": 11},
    "subjects_by_condition": {"CTRL": 4, "IZ": 7},
}

SOURCE_TO_CANONICAL = {
    "sample": "sample_id",
    "patient": "subject_id",
    "cell_type_original": "cell_type",
    "major_labl": "condition",
}
REQUIRED_OBS = {*SOURCE_TO_CANONICAL, "n_counts", "n_genes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _manifest_payload_sha256(manifest: dict[str, object]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_payload_sha256"
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _artifact_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}.{suffix}"


def _validate_source_structure(source: ad.AnnData) -> None:
    expected_shape = tuple(EXPECTED_KUPPE_CTRL_IZ["source_shape"])
    if source.shape != expected_shape:
        raise ValueError(
            f"unexpected Kuppe source shape: {source.shape} != {expected_shape}"
        )
    if source.raw is None:
        raise ValueError("Kuppe source has no raw count matrix")
    if source.raw.shape != source.shape:
        raise ValueError(
            f"Kuppe raw shape {source.raw.shape} does not match X shape {source.shape}"
        )
    if not source.raw.var_names.equals(source.var_names):
        raise ValueError("Kuppe raw and normalized feature identifiers are not aligned")
    if source.var_names.has_duplicates:
        raise ValueError("Kuppe source contains duplicate feature identifiers")

    missing = REQUIRED_OBS.difference(source.obs.columns)
    if missing:
        raise ValueError(f"Kuppe metadata is missing {sorted(missing)}")
    if source.obs_names.has_duplicates:
        raise ValueError("Kuppe metadata contains duplicate cell IDs")


def _canonical_obs(source_obs: pd.DataFrame) -> pd.DataFrame:
    if source_obs.empty:
        raise ValueError("Kuppe CTRL/IZ selection contains no cells")
    required = list(REQUIRED_OBS)
    if source_obs[required].isna().any().any():
        raise ValueError("Kuppe CTRL/IZ metadata contains missing required values")

    obs = source_obs.copy()
    for source_key, canonical_key in SOURCE_TO_CANONICAL.items():
        values = source_obs[source_key].astype(str)
        if canonical_key in source_obs and not source_obs[canonical_key].astype(
            str
        ).equals(values):
            raise ValueError(
                f"existing {canonical_key!r} values disagree with source column "
                f"{source_key!r}"
            )
        obs[canonical_key] = values.to_numpy(dtype=object)
    obs["region"] = obs["condition"].to_numpy(dtype=object)
    obs["broad_cell_type"] = obs["cell_type"].to_numpy(dtype=object)
    obs.index = pd.Index(source_obs.index.astype(str), dtype=object, name="cell_id")
    return obs


def _sample_design(obs: pd.DataFrame) -> pd.DataFrame:
    grouped = obs.groupby("sample_id", observed=True, sort=True)
    for field in ("subject_id", "condition"):
        ambiguous = grouped[field].nunique().gt(1)
        if ambiguous.any():
            samples = ambiguous.index[ambiguous].astype(str).tolist()
            raise ValueError(f"sample IDs map to multiple {field} values: {samples}")

    design = grouped.agg(
        subject_id=("subject_id", "first"),
        condition=("condition", "first"),
        n_cells=("subject_id", "size"),
        n_cell_types=("cell_type", "nunique"),
    ).reset_index()
    for field in ("sample_id", "subject_id", "condition"):
        design[field] = design[field].astype(str)
    return design


def _cell_type_support(obs: pd.DataFrame) -> pd.DataFrame:
    return (
        obs.groupby(["condition", "cell_type"], observed=True, sort=True)
        .agg(
            n_cells=("subject_id", "size"),
            n_samples=("sample_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
        .sort_values(["condition", "cell_type"], kind="stable")
        .reset_index(drop=True)
    )


def _obs_audit(obs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for source_key, canonical_key in SOURCE_TO_CANONICAL.items():
        source_values = obs[source_key].astype(str)
        canonical_values = obs[canonical_key].astype(str)
        records.append(
            {
                "source_column": source_key,
                "canonical_column": canonical_key,
                "n_cells": len(obs),
                "n_missing": int(obs[source_key].isna().sum()),
                "n_unique": int(source_values.nunique()),
                "mapping_equal": bool(source_values.equals(canonical_values)),
            }
        )
    return pd.DataFrame.from_records(records)


def _mapping(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.items()}


def _validate_selected_cohort(
    obs: pd.DataFrame,
    design: pd.DataFrame,
    *,
    n_vars: int,
) -> None:
    expected = EXPECTED_KUPPE_CTRL_IZ
    observed_counts = {
        "n_obs": len(obs),
        "n_vars": n_vars,
        "n_subjects": obs["subject_id"].nunique(),
        "n_samples": obs["sample_id"].nunique(),
        "n_cell_types": obs["cell_type"].nunique(),
    }
    for key, observed in observed_counts.items():
        if int(observed) != int(expected[key]):
            raise ValueError(
                f"unexpected Kuppe CTRL/IZ {key}: {observed} != {expected[key]}"
            )

    observed_conditions = set(obs["condition"].astype(str).unique())
    if observed_conditions != set(CONDITIONS):
        raise ValueError(
            "unexpected Kuppe CTRL/IZ conditions: "
            f"{sorted(observed_conditions)} != {sorted(CONDITIONS)}"
        )
    observed_cell_types = set(obs["cell_type"].astype(str).unique())
    expected_cell_types = set(map(str, expected["cell_types"]))
    if observed_cell_types != expected_cell_types:
        raise ValueError(
            "unexpected Kuppe CTRL/IZ broad cell types: "
            f"{sorted(observed_cell_types)} != {sorted(expected_cell_types)}"
        )

    unique_subject_conditions = design.drop_duplicates(["subject_id", "condition"])
    observed_by_condition: dict[str, dict[str, int]] = {
        "cells_by_condition": _mapping(obs.groupby("condition", observed=True).size()),
        "samples_by_condition": _mapping(
            design.groupby("condition", observed=True)["sample_id"].nunique()
        ),
        "subjects_by_condition": _mapping(
            unique_subject_conditions.groupby("condition", observed=True)[
                "subject_id"
            ].nunique()
        ),
    }
    for key, observed_summary in observed_by_condition.items():
        if observed_summary != expected[key]:
            raise ValueError(
                f"unexpected Kuppe CTRL/IZ {key}: {observed_summary} != {expected[key]}"
            )

    subject_sets = {
        condition: set(
            design.loc[design["condition"].eq(condition), "subject_id"].astype(str)
        )
        for condition in CONDITIONS
    }
    overlap = subject_sets["CTRL"].intersection(subject_sets["IZ"])
    if overlap:
        raise ValueError(
            "Kuppe CTRL and IZ must be independent subject groups; shared subjects: "
            f"{sorted(overlap)}"
        )


def _metadata_audit(
    source: ad.AnnData,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_source_structure(source)
    source_condition = source.obs["major_labl"].astype(str)
    positions = np.flatnonzero(source_condition.isin(CONDITIONS).to_numpy())
    selected_source_obs = source.obs.iloc[positions].copy()
    obs = _canonical_obs(selected_source_obs)
    design = _sample_design(obs)
    support = _cell_type_support(obs)
    obs_audit = _obs_audit(obs)
    _validate_selected_cohort(obs, design, n_vars=source.n_vars)
    return positions, obs, design, support, obs_audit


def _validate_payload(values: np.ndarray, *, label: str, integer: bool) -> None:
    for start in range(0, values.size, 5_000_000):
        chunk = values[start : start + 5_000_000]
        if not np.isfinite(chunk).all() or np.any(chunk < 0):
            raise ValueError(f"Kuppe {label} contains non-finite or negative values")
        if integer and not np.equal(chunk, np.rint(chunk)).all():
            raise ValueError(f"Kuppe {label} contains non-integer values")


def _integer_counts(
    matrix: sparse.spmatrix | np.ndarray,
) -> sparse.csr_matrix | np.ndarray:
    if sparse.issparse(matrix):
        source = cast(sparse.spmatrix, matrix)
        _validate_payload(np.asarray(source.data), label="raw.X", integer=True)
        if source.data.size and source.data.max() > np.iinfo(np.int32).max:
            raise ValueError("Kuppe raw.X count exceeds int32 range")
        result = source.astype(np.int32).tocsr()
        result.eliminate_zeros()
        return result
    values = np.asarray(matrix)
    _validate_payload(values.ravel(), label="raw.X", integer=True)
    if values.size and values.max() > np.iinfo(np.int32).max:
        raise ValueError("Kuppe raw.X count exceeds int32 range")
    return values.astype(np.int32)


def _normalized_matrix(
    matrix: sparse.spmatrix | np.ndarray,
) -> sparse.csr_matrix | np.ndarray:
    if sparse.issparse(matrix):
        source = cast(sparse.spmatrix, matrix)
        _validate_payload(np.asarray(source.data), label="X", integer=False)
        result = source.tocsr()
        result.eliminate_zeros()
        return result
    values = np.asarray(matrix)
    _validate_payload(values.ravel(), label="X", integer=False)
    return values.copy()


def _selected_matrices(
    source: ad.AnnData,
    positions: np.ndarray,
) -> tuple[sparse.csr_matrix | np.ndarray, sparse.csr_matrix | np.ndarray]:
    normalized = _normalized_matrix(source.X[positions, :])
    assert source.raw is not None
    counts = _integer_counts(source.raw.X[positions, :])
    return normalized, counts


def _cohort_summary(
    obs: pd.DataFrame,
    design: pd.DataFrame,
    n_vars: int,
) -> dict[str, object]:
    unique_subject_conditions = design.drop_duplicates(["subject_id", "condition"])
    return {
        "shape": [len(obs), int(n_vars)],
        "n_cells": len(obs),
        "n_genes": int(n_vars),
        "n_subjects": int(obs["subject_id"].nunique()),
        "n_samples": int(obs["sample_id"].nunique()),
        "n_cell_types": int(obs["cell_type"].nunique()),
        "cell_types": sorted(obs["cell_type"].astype(str).unique().tolist()),
        "cells_by_condition": _mapping(obs.groupby("condition", observed=True).size()),
        "samples_by_condition": _mapping(
            design.groupby("condition", observed=True)["sample_id"].nunique()
        ),
        "subjects_by_condition": _mapping(
            unique_subject_conditions.groupby("condition", observed=True)[
                "subject_id"
            ].nunique()
        ),
    }


def _base_manifest(
    source_path: Path,
    obs: pd.DataFrame,
    design: pd.DataFrame,
    *,
    n_vars: int,
    source_sha256: str | None,
    mode: str,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "audit_schema_version": AUDIT_SCHEMA,
        "dataset_id": DATASET_ID,
        "source_dataset_id": SOURCE_DATASET_ID,
        "mode": mode,
        "source": {
            "filename": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": source_sha256,
            "read_mode": "backed_r",
            "shape": list(EXPECTED_KUPPE_CTRL_IZ["source_shape"]),
        },
        "selection": {
            "source_field": "major_labl",
            "values": list(CONDITIONS),
            "reference": "CTRL",
            "target": "IZ",
            "preserve_observation_order": True,
            "preserve_variable_axis": True,
        },
        "obs_mapping": SOURCE_TO_CANONICAL,
        "cohort": _cohort_summary(obs, design, n_vars),
        "design": {
            "analysis_unit": "subject_id",
            "sample_unit": "sample_id",
            "context_key": "condition",
            "design_class": "independent_subjects_with_nested_samples",
            "paired_subjects": [],
            "technical_replicates": (
                "samples from the same subject remain in one resampling block"
            ),
        },
        "matrices": {
            "expression_matrix_accessed": mode == "counts_ready_h5ad",
            "normalized_source": "X",
            "counts_source": "raw.X",
            "counts_layer": "counts",
            "counts_semantics": "raw non-negative integer counts",
        },
    }


def _write_audit_tables(
    prefix: Path,
    *,
    design: pd.DataFrame,
    support: pd.DataFrame,
    obs_audit: pd.DataFrame,
) -> dict[str, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "sample_design": _artifact_path(prefix, "sample_design.tsv"),
        "cell_type_support": _artifact_path(prefix, "cell_type_support.tsv"),
        "obs_audit": _artifact_path(prefix, "obs_audit.tsv"),
    }
    design.to_csv(paths["sample_design"], sep="\t", index=False)
    support.to_csv(paths["cell_type_support"], sep="\t", index=False)
    obs_audit.to_csv(paths["obs_audit"], sep="\t", index=False)
    return paths


def _write_manifest(
    path: Path,
    manifest: dict[str, object],
    *,
    artifact_paths: dict[str, Path],
) -> dict[str, object]:
    manifest["artifacts"] = {
        key: {
            "filename": artifact.name,
            "size_bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        }
        for key, artifact in sorted(artifact_paths.items())
    }
    manifest["manifest_payload_sha256"] = _manifest_payload_sha256(manifest)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def audit_kuppe_ctrl_iz(
    source_path: Path,
    audit_dir: Path,
    *,
    compute_source_sha256: bool = True,
) -> dict[str, object]:
    """Write a backed-``obs`` audit without reading either expression matrix."""
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = ad.read_h5ad(source_path, backed="r")
    try:
        _, obs, design, support, obs_audit = _metadata_audit(source)
        n_vars = source.n_vars
    finally:
        source.file.close()

    source_sha256 = _sha256(source_path) if compute_source_sha256 else None
    prefix = audit_dir / DATASET_ID
    artifact_paths = _write_audit_tables(
        prefix,
        design=design,
        support=support,
        obs_audit=obs_audit,
    )
    manifest = _base_manifest(
        source_path,
        obs,
        design,
        n_vars=n_vars,
        source_sha256=source_sha256,
        mode="obs_design_audit",
    )
    manifest["matrices"]["output_h5ad_written"] = False  # type: ignore[index]
    manifest_path = _artifact_path(prefix, "manifest.json")
    _write_manifest(manifest_path, manifest, artifact_paths=artifact_paths)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }


def prepare_kuppe_ctrl_iz(source_path: Path, output_path: Path) -> ad.AnnData:
    """Write the official CTRL/IZ subset with raw counts in ``layers['counts']``."""
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Kuppe source and output paths must differ")

    source = ad.read_h5ad(source_path, backed="r")
    try:
        positions, obs, design, support, obs_audit = _metadata_audit(source)
        var = source.var.copy()
        n_vars = source.n_vars
        normalized, counts = _selected_matrices(source, positions)
    finally:
        source.file.close()

    observed_library = np.asarray(counts.sum(axis=1)).ravel()
    expected_library = obs["n_counts"].to_numpy(dtype=np.float64)
    if not np.array_equal(observed_library, expected_library):
        raise ValueError("Kuppe CTRL/IZ raw.X library sizes disagree with n_counts")
    if sparse.issparse(counts):
        observed_features = np.asarray(
            cast(sparse.spmatrix, counts).getnnz(axis=1)
        ).ravel()
    else:
        observed_features = np.count_nonzero(counts, axis=1)
    expected_features = obs["n_genes"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_features, expected_features):
        raise ValueError("Kuppe CTRL/IZ raw.X detected genes disagree with n_genes")

    source_sha256 = _sha256(source_path)
    manifest = _base_manifest(
        source_path,
        obs,
        design,
        n_vars=n_vars,
        source_sha256=source_sha256,
        mode="counts_ready_h5ad",
    )
    conversion = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_sha256": source_sha256,
        "filters": ["major_labl in {CTRL, IZ}"],
        "subject_key": "subject_id",
        "sample_key": "sample_id",
        "context_keys": ["condition"],
        "cell_type_key": "cell_type",
        "counts_lineage": "source raw.X selected by frozen row positions",
        "x_lineage": "source normalized X selected by the same frozen row positions",
        "design": "independent CTRL and IZ subjects with nested samples",
    }
    result = ad.AnnData(X=normalized, obs=obs, var=var)
    result.layers["counts"] = counts
    result.uns["crychic_conversion"] = conversion
    result.uns["crychic_design"] = manifest["design"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path, compression="gzip", compression_opts=4)
    output_sha256 = _sha256(output_path)
    manifest["output"] = {
        "filename": output_path.name,
        "size_bytes": output_path.stat().st_size,
        "sha256": output_sha256,
        "shape": [result.n_obs, result.n_vars],
        "counts_nnz": (
            int(cast(sparse.spmatrix, counts).nnz)
            if sparse.issparse(counts)
            else int(np.count_nonzero(counts))
        ),
    }

    prefix = output_path.with_suffix("")
    artifact_paths = _write_audit_tables(
        prefix,
        design=design,
        support=support,
        obs_audit=obs_audit,
    )
    manifest_path = _artifact_path(prefix, "manifest.json")
    _write_manifest(manifest_path, manifest, artifact_paths=artifact_paths)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_h5ad", type=Path)
    parser.add_argument("output_h5ad", type=Path, nargs="?")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="write backed-obs design artifacts without reading expression matrices",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        help="audit output directory (required with --audit-only)",
    )
    parser.add_argument(
        "--skip-source-sha256",
        action="store_true",
        help="defer the source hash during a metadata-only audit",
    )
    args = parser.parse_args()

    if args.audit_only:
        if args.output_h5ad is not None:
            parser.error("output_h5ad is not used with --audit-only")
        if args.audit_dir is None:
            parser.error("--audit-dir is required with --audit-only")
        result = audit_kuppe_ctrl_iz(
            args.source_h5ad,
            args.audit_dir,
            compute_source_sha256=not args.skip_source_sha256,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.output_h5ad is None:
        parser.error("output_h5ad is required unless --audit-only is set")
    if args.audit_dir is not None:
        parser.error("--audit-dir is only valid with --audit-only")
    if args.skip_source_sha256:
        parser.error("--skip-source-sha256 is only valid with --audit-only")
    prepare_kuppe_ctrl_iz(args.source_h5ad, args.output_h5ad)


if __name__ == "__main__":
    main()
