"""Prepare and design-audit the Kuppe myocardial-infarction snRNA atlas."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

DATASET_ID = "Kuppe_MI_Zenodo6578047"
RESPONSE_REASON = "mixed_paired_unpaired_multi_region_design_unsupported"

EXPECTED_KUPPE: dict[str, Any] = {
    "n_obs": 191_795,
    "n_vars": 29_126,
    "n_subjects": 20,
    "n_samples": 29,
    "regions": ["CTRL", "RZ", "BZ", "IZ", "FZ"],
    "n_cell_types": 11,
    "cell_types": [
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
    ],
    "subjects_by_region": {"CTRL": 4, "RZ": 5, "BZ": 3, "IZ": 7, "FZ": 6},
    "samples_by_region": {"CTRL": 4, "RZ": 5, "BZ": 3, "IZ": 11, "FZ": 6},
    "multi_region_subjects": {"P2": 3, "P3": 3, "P9": 2},
    "replicate_samples": {"P9|IZ": 3, "P15|IZ": 2, "P16|IZ": 2},
}

SOURCE_TO_CANONICAL = {
    "sample": "sample_id",
    "patient": "subject_id",
    "major_labl": "region",
    "cell_type_original": "cell_type",
}
REQUIRED_OBS = {
    *SOURCE_TO_CANONICAL,
    "n_counts",
    "n_genes",
    "patient_region_id",
    "patient_group",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}.{suffix}"


def _canonical_obs(source_obs: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_OBS.difference(source_obs.columns)
    if missing:
        raise ValueError(f"Kuppe metadata is missing {sorted(missing)}")
    if source_obs.index.has_duplicates:
        raise ValueError("Kuppe metadata contains duplicate cell IDs")
    if source_obs[list(REQUIRED_OBS)].isna().any().any():
        raise ValueError("Kuppe metadata contains missing required values")

    obs = source_obs.copy()
    for source, canonical in SOURCE_TO_CANONICAL.items():
        values = source_obs[source].astype(str)
        existing_matches = canonical not in source_obs or source_obs[canonical].astype(
            str
        ).equals(values)
        if not existing_matches:
            raise ValueError(
                f"existing {canonical!r} values disagree with source column {source!r}"
            )
        obs[canonical] = values.to_numpy(dtype=object)
    obs["broad_cell_type"] = obs["cell_type"].to_numpy(dtype=object)
    obs.index = pd.Index(source_obs.index.astype(str), dtype=object, name="cell_id")
    return obs


def _sample_design(obs: pd.DataFrame) -> pd.DataFrame:
    grouped = obs.groupby("sample_id", observed=True, sort=True)
    for column in ("subject_id", "region"):
        ambiguous = grouped[column].nunique().gt(1)
        if ambiguous.any():
            samples = ambiguous.index[ambiguous].astype(str).tolist()
            raise ValueError(f"sample IDs map to multiple {column} values: {samples}")

    sample_table = grouped.agg(
        subject_id=("subject_id", "first"),
        region=("region", "first"),
        n_cells=("subject_id", "size"),
        n_cell_types=("cell_type", "nunique"),
    ).reset_index()
    for column in ("sample_id", "subject_id", "region"):
        sample_table[column] = sample_table[column].astype(str)
    return sample_table


def _observed_mapping(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.items()}


def _subject_region_mapping(series: pd.Series) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in series.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("expected a two-level subject-region index")
        subject, region = key
        result[f"{subject}|{region}"] = int(value)
    return result


def _validate_expected(
    obs: pd.DataFrame,
    sample_table: pd.DataFrame,
    *,
    n_vars: int,
) -> None:
    expected = EXPECTED_KUPPE
    observed_counts = {
        "n_obs": len(obs),
        "n_vars": n_vars,
        "n_subjects": obs["subject_id"].nunique(),
        "n_samples": obs["sample_id"].nunique(),
        "n_cell_types": obs["cell_type"].nunique(),
    }
    for key, observed in observed_counts.items():
        if int(observed) != int(expected[key]):
            raise ValueError(f"unexpected Kuppe {key}: {observed} != {expected[key]}")

    observed_regions = set(obs["region"].astype(str).unique())
    expected_regions = set(map(str, expected["regions"]))
    if observed_regions != expected_regions:
        raise ValueError(
            f"unexpected Kuppe regions: {sorted(observed_regions)} != "
            f"{sorted(expected_regions)}"
        )
    observed_cell_types = set(obs["cell_type"].astype(str).unique())
    expected_cell_types = set(map(str, expected["cell_types"]))
    if observed_cell_types != expected_cell_types:
        raise ValueError(
            "unexpected Kuppe broad cell types: "
            f"{sorted(observed_cell_types)} != {sorted(expected_cell_types)}"
        )

    unique_subject_regions = sample_table.drop_duplicates(["subject_id", "region"])
    subjects_by_region = _observed_mapping(
        unique_subject_regions.groupby("region", observed=True)["subject_id"].nunique()
    )
    samples_by_region = _observed_mapping(
        sample_table.groupby("region", observed=True)["sample_id"].nunique()
    )
    if subjects_by_region != expected["subjects_by_region"]:
        raise ValueError(
            "unexpected Kuppe subject support by region: "
            f"{subjects_by_region} != {expected['subjects_by_region']}"
        )
    if samples_by_region != expected["samples_by_region"]:
        raise ValueError(
            "unexpected Kuppe sample support by region: "
            f"{samples_by_region} != {expected['samples_by_region']}"
        )

    regions_per_subject = unique_subject_regions.groupby("subject_id")[
        "region"
    ].nunique()
    multi_region_subjects = _observed_mapping(
        regions_per_subject[regions_per_subject > 1]
    )
    if multi_region_subjects != expected["multi_region_subjects"]:
        raise ValueError(
            "unexpected Kuppe cross-region subjects: "
            f"{multi_region_subjects} != {expected['multi_region_subjects']}"
        )
    sample_counts = sample_table.groupby(["subject_id", "region"]).size()
    repeated = sample_counts[sample_counts > 1]
    replicate_samples = _subject_region_mapping(repeated)
    if replicate_samples != expected["replicate_samples"]:
        raise ValueError(
            "unexpected Kuppe within-region replicate samples: "
            f"{replicate_samples} != {expected['replicate_samples']}"
        )


def _pairwise_contrasts(
    sample_table: pd.DataFrame,
    regions: list[str],
) -> pd.DataFrame:
    subjects = {
        region: set(
            sample_table.loc[sample_table["region"] == region, "subject_id"].astype(str)
        )
        for region in regions
    }
    records: list[dict[str, object]] = []
    for reference, comparison in itertools.combinations(regions, 2):
        reference_subjects = subjects[reference]
        comparison_subjects = subjects[comparison]
        overlap = reference_subjects & comparison_subjects
        if not overlap:
            pairwise_design = "independent_between_subjects"
            subset_status = "candidate_after_preregistered_pairwise_subsetting"
        elif reference_subjects == comparison_subjects:
            pairwise_design = "fully_paired"
            subset_status = "candidate_after_preregistered_pairwise_subsetting"
        else:
            pairwise_design = "mixed_paired_unpaired"
            subset_status = "not_estimable_by_current_response_backend"
        records.append(
            {
                "reference_region": reference,
                "comparison_region": comparison,
                "n_reference_subjects": len(reference_subjects),
                "n_comparison_subjects": len(comparison_subjects),
                "n_paired_subjects": len(overlap),
                "paired_subjects": ";".join(sorted(overlap)),
                "n_reference_only_subjects": len(reference_subjects - overlap),
                "n_comparison_only_subjects": len(comparison_subjects - overlap),
                "pairwise_design": pairwise_design,
                "pairwise_subset_status": subset_status,
                "full_dataset_response_status": "not_estimable",
                "reason_code": RESPONSE_REASON,
                "availability_status": "descriptive_only",
            }
        )
    return pd.DataFrame.from_records(records)


def _cell_type_support(obs: pd.DataFrame) -> pd.DataFrame:
    return (
        obs.groupby(["region", "cell_type"], observed=True, sort=True)
        .agg(
            n_cells=("subject_id", "size"),
            n_samples=("sample_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
        .sort_values(["region", "cell_type"], kind="stable")
        .reset_index(drop=True)
    )


def audit_kuppe_metadata(
    source_obs: pd.DataFrame,
    *,
    n_vars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Validate canonical metadata and classify the repeated-region design."""
    obs = _canonical_obs(source_obs)
    samples = _sample_design(obs)
    _validate_expected(obs, samples, n_vars=n_vars)

    region_order = [str(value) for value in EXPECTED_KUPPE["regions"]]
    contrasts = _pairwise_contrasts(samples, region_order)
    support = _cell_type_support(obs)
    unique_subject_regions = samples.drop_duplicates(["subject_id", "region"])
    regions_per_subject = unique_subject_regions.groupby("subject_id")[
        "region"
    ].nunique()
    cross_region = regions_per_subject[regions_per_subject > 1]
    within_region_counts = samples.groupby(["subject_id", "region"]).size()
    within_region_replicates = within_region_counts[within_region_counts > 1]

    audit: dict[str, object] = {
        "audit_version": "2026-07-12.v1",
        "dataset_id": DATASET_ID,
        "shape": [len(obs), int(n_vars)],
        "independent_unit": "subject_id",
        "sample_unit": "sample_id",
        "context_key": "region",
        "design_class": "mixed_paired_unpaired_multi_region",
        "subjects": int(obs["subject_id"].nunique()),
        "samples": int(obs["sample_id"].nunique()),
        "regions": region_order,
        "subjects_by_region": _observed_mapping(
            unique_subject_regions.groupby("region")["subject_id"].nunique()
        ),
        "samples_by_region": _observed_mapping(
            samples.groupby("region")["sample_id"].nunique()
        ),
        "cross_region_subjects": {
            str(subject): sorted(
                unique_subject_regions.loc[
                    unique_subject_regions["subject_id"] == subject, "region"
                ].astype(str)
            )
            for subject in cross_region.index
        },
        "within_region_replicate_samples": _subject_region_mapping(
            within_region_replicates
        ),
        "response_analysis": {
            "status": "not_estimable",
            "reason_code": RESPONSE_REASON,
            "formal_inference_allowed": False,
            "explanation": (
                "Only P2, P3, and P9 span regions, while most subjects occur in "
                "one region and P9/P15/P16 also have within-region replicate samples. "
                "The current response backend cannot combine paired, unpaired, and "
                "nested replicate contributions without changing the estimand."
            ),
        },
        "availability_analysis": {
            "status": "eligible_descriptive_only",
            "formal_inference_allowed": False,
            "estimand": "per-sample and per-region expression-supported availability",
            "required_block": (
                "keep every sample from one subject in one resampling block"
            ),
        },
    }
    return samples, contrasts, support, audit


def _validate_structure(source: ad.AnnData) -> None:
    expected_shape = (
        int(EXPECTED_KUPPE["n_obs"]),
        int(EXPECTED_KUPPE["n_vars"]),
    )
    if source.shape != expected_shape:
        raise ValueError(f"unexpected Kuppe shape: {source.shape} != {expected_shape}")
    if source.raw is None:
        raise ValueError("Kuppe source has no raw matrix")
    if source.raw.shape != source.shape:
        raise ValueError(
            f"Kuppe raw shape {source.raw.shape} does not match X shape {source.shape}"
        )
    if not source.var_names.equals(source.raw.var_names):
        raise ValueError("Kuppe raw and normalized feature identifiers are not aligned")
    if source.var_names.has_duplicates:
        raise ValueError("Kuppe source contains duplicate feature identifiers")


def _validate_payload(values: np.ndarray, *, label: str, integer: bool) -> None:
    chunk_size = 5_000_000
    for start in range(0, values.size, chunk_size):
        chunk = values[start : start + chunk_size]
        if not np.isfinite(chunk).all() or np.any(chunk < 0):
            raise ValueError(f"Kuppe {label} contains non-finite or negative values")
        if integer and not np.equal(chunk, np.rint(chunk)).all():
            raise ValueError(f"Kuppe {label} contains non-integer values")


def _integer_counts(
    matrix: sparse.spmatrix | np.ndarray,
) -> sparse.csr_matrix | np.ndarray:
    if sparse.issparse(matrix):
        sparse_matrix = cast(sparse.spmatrix, matrix)
        data = np.asarray(sparse_matrix.data)
        _validate_payload(data, label="raw.X", integer=True)
        if data.size and data.max() > np.iinfo(np.int32).max:
            raise ValueError("Kuppe raw.X count exceeds int32 range")
        return sparse_matrix.astype(np.int32).tocsr()

    values = np.asarray(matrix)
    _validate_payload(values.ravel(), label="raw.X", integer=True)
    if values.size and values.max() > np.iinfo(np.int32).max:
        raise ValueError("Kuppe raw.X count exceeds int32 range")
    return values.astype(np.int32)


def _normalized_matrix(
    matrix: sparse.spmatrix | np.ndarray,
) -> sparse.csr_matrix | np.ndarray:
    if sparse.issparse(matrix):
        sparse_matrix = cast(sparse.spmatrix, matrix)
        _validate_payload(np.asarray(sparse_matrix.data), label="X", integer=False)
        return sparse_matrix.tocsr()
    values = np.asarray(matrix)
    _validate_payload(values.ravel(), label="X", integer=False)
    return values.copy()


def _write_audit_artifacts(
    prefix: Path,
    *,
    audit: dict[str, object],
    samples: pd.DataFrame,
    contrasts: pd.DataFrame,
    support: pd.DataFrame,
) -> dict[str, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "design_audit": _artifact_path(prefix, "design_audit.json"),
        "sample_design": _artifact_path(prefix, "sample_design.tsv"),
        "contrast_design": _artifact_path(prefix, "contrast_design.tsv"),
        "cell_type_support": _artifact_path(prefix, "cell_type_support.tsv"),
    }
    paths["design_audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    samples.to_csv(paths["sample_design"], sep="\t", index=False)
    contrasts.to_csv(paths["contrast_design"], sep="\t", index=False)
    support.to_csv(paths["cell_type_support"], sep="\t", index=False)
    return paths


def audit_kuppe(source_path: Path, audit_dir: Path) -> dict[str, object]:
    """Run a metadata-only audit without loading or writing the full matrices."""
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = ad.read_h5ad(source_path, backed="r")
    try:
        _validate_structure(source)
        samples, contrasts, support, audit = audit_kuppe_metadata(
            source.obs, n_vars=source.n_vars
        )
        audit["lineage"] = {
            "source_path": str(source_path.resolve()),
            "source_size_bytes": source_path.stat().st_size,
            "source_sha256": None,
            "hash_status": "deferred_to_full_conversion",
            "counts_source": "raw.X",
            "normalized_source": "X",
        }
    finally:
        source.file.close()

    prefix = audit_dir / source_path.stem
    paths = _write_audit_artifacts(
        prefix,
        audit=audit,
        samples=samples,
        contrasts=contrasts,
        support=support,
    )
    return {
        "audit": audit,
        "artifacts": {key: str(path) for key, path in paths.items()},
    }


def prepare_kuppe(source_path: Path, output_path: Path) -> ad.AnnData:
    """Copy raw counts into ``layers['counts']`` and preserve normalized ``X``."""
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Kuppe source and output paths must differ")

    source = ad.read_h5ad(source_path, backed="r")
    try:
        _validate_structure(source)
        samples, contrasts, support, audit = audit_kuppe_metadata(
            source.obs, n_vars=source.n_vars
        )
        obs = _canonical_obs(source.obs)
        var = source.var.copy()
        source_uns = {
            key: str(source.uns[key])
            for key in (
                "title",
                "schema_version",
                "X_normalization",
                "X_approximate_distribution",
            )
            if key in source.uns
        }
        normalized = _normalized_matrix(source.X[:])
        assert source.raw is not None
        counts = _integer_counts(source.raw.X[:])
    finally:
        source.file.close()

    observed_library = np.asarray(counts.sum(axis=1)).ravel()
    expected_library = obs["n_counts"].to_numpy(dtype=np.float64)
    if not np.array_equal(observed_library, expected_library):
        raise ValueError("Kuppe raw.X library sizes do not match obs['n_counts']")
    if sparse.issparse(counts):
        sparse_counts = cast(sparse.spmatrix, counts)
        observed_features = np.asarray(sparse_counts.getnnz(axis=1)).ravel()
    else:
        observed_features = np.count_nonzero(counts, axis=1)
    expected_features = obs["n_genes"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_features, expected_features):
        raise ValueError("Kuppe raw.X detected genes do not match obs['n_genes']")

    source_manifest = {
        "path": str(source_path.resolve()),
        "filename": source_path.name,
        "size_bytes": source_path.stat().st_size,
        "sha256": _sha256(source_path),
    }
    audit["lineage"] = {
        "source": source_manifest,
        "counts_source": "raw.X",
        "normalized_source": "X",
        "counts_validation": (
            "all stored values checked as finite non-negative integers"
        ),
    }
    conversion = {
        "dataset_id": DATASET_ID,
        "converter": "benchmarks.datasets.prepare_kuppe",
        "source": source_manifest,
        "source_release_metadata": source_uns,
        "subject_key_source": "patient",
        "sample_key_source": "sample",
        "context_key_source": "major_labl",
        "cell_type_key_source": "cell_type_original",
        "subject_key": "subject_id",
        "sample_key": "sample_id",
        "context_keys": ["region"],
        "cell_type_key": "cell_type",
        "broad_cell_type_key": "broad_cell_type",
        "counts_lineage": "source raw.X copied exactly then losslessly cast to int32",
        "counts_semantics": "raw non-negative integer counts",
        "x_lineage": "source normalized X preserved without numerical transformation",
        "x_semantics": "source log-normalized expression",
        "matrix_layout": "CSR",
        "design": "mixed paired/unpaired five-region design",
        "response_status": "not_estimable",
        "response_reason_code": RESPONSE_REASON,
        "availability_status": "eligible_descriptive_only",
    }

    result = ad.AnnData(X=normalized, obs=obs, var=var)
    result.layers["counts"] = counts
    result.uns["crychic_conversion"] = conversion
    result.uns["crychic_design_audit"] = {
        "design_class": audit["design_class"],
        "response_status": "not_estimable",
        "response_reason_code": RESPONSE_REASON,
        "availability_status": "eligible_descriptive_only",
        "formal_inference_allowed": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path, compression="gzip", compression_opts=4)
    output_manifest = {
        "path": str(output_path.resolve()),
        "filename": output_path.name,
        "size_bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
    }
    prefix = output_path.with_suffix("")
    audit_paths = _write_audit_artifacts(
        prefix,
        audit=audit,
        samples=samples,
        contrasts=contrasts,
        support=support,
    )
    counts_nnz = (
        int(cast(sparse.spmatrix, counts).nnz)
        if sparse.issparse(counts)
        else int(np.count_nonzero(counts))
    )
    summary = {
        "dataset_id": DATASET_ID,
        "shape": [result.n_obs, result.n_vars],
        "subjects": int(obs["subject_id"].nunique()),
        "samples": int(obs["sample_id"].nunique()),
        "regions": [str(value) for value in EXPECTED_KUPPE["regions"]],
        "cell_types": sorted(obs["cell_type"].unique().tolist()),
        "counts_nnz": counts_nnz,
        "source": source_manifest,
        "output": output_manifest,
        "provenance": conversion,
        "design_audit": audit,
        "audit_artifacts": {key: path.name for key, path in audit_paths.items()},
    }
    output_path.with_suffix(".conversion.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_h5ad", type=Path)
    parser.add_argument("output_h5ad", type=Path, nargs="?")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="read backed metadata and write design tables without loading matrices",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        help="audit output directory (required with --audit-only)",
    )
    args = parser.parse_args()

    if args.audit_only:
        if args.output_h5ad is not None:
            parser.error("output_h5ad is not used with --audit-only")
        if args.audit_dir is None:
            parser.error("--audit-dir is required with --audit-only")
        result = audit_kuppe(args.source_h5ad, args.audit_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.output_h5ad is None:
        parser.error("output_h5ad is required unless --audit-only is set")
    if args.audit_dir is not None:
        parser.error("--audit-dir is only valid with --audit-only")
    prepare_kuppe(args.source_h5ad, args.output_h5ad)


if __name__ == "__main__":
    main()
