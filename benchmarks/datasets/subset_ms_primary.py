"""Freeze the independent-subject CA-versus-Ctrl MS primary analysis object."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad

PRIMARY_CONTEXTS = ("CA", "Ctrl")
EXPECTED_PRIMARY_SHAPE = (75_004, 32_115)
EXPECTED_SUBJECTS = {"CA": 5, "Ctrl": 6}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset_ms_primary(
    input_path: Path,
    output_path: Path,
    *,
    expected_sha256: str,
) -> ad.AnnData:
    """Select CA and Ctrl without changing cells, genes, matrices, or order."""
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    source_sha256 = _sha256(input_path)
    if source_sha256 != expected_sha256:
        raise ValueError(
            f"MS prepared input SHA256 mismatch: {source_sha256} != {expected_sha256}"
        )
    source = ad.read_h5ad(input_path)
    conversion = source.uns.get("crychic_conversion", {})
    if conversion.get("dataset_id") != "UCSC_Lerma_Martin_MS_snRNA":
        raise ValueError("input is not the frozen UCSC MS snRNA object")
    required = {"sample_id", "subject_id", "lesion_type", "cell_type", "batch"}
    missing = required.difference(source.obs.columns)
    if missing:
        raise ValueError(f"MS prepared input is missing {sorted(missing)}")

    mask = source.obs["lesion_type"].astype(str).isin(PRIMARY_CONTEXTS)
    result = source[mask].copy()
    contexts = set(result.obs["lesion_type"].astype(str))
    if contexts != set(PRIMARY_CONTEXTS):
        raise ValueError(f"MS primary contexts are incomplete: {sorted(contexts)}")
    samples = result.obs[
        ["sample_id", "subject_id", "lesion_type", "batch"]
    ].drop_duplicates()
    if samples["sample_id"].duplicated().any():
        raise ValueError("MS primary sample IDs do not map uniquely")
    by_context = {
        context: set(group["subject_id"].astype(str))
        for context, group in samples.groupby("lesion_type", observed=True)
    }
    if by_context["CA"].intersection(by_context["Ctrl"]):
        raise ValueError("CA and Ctrl unexpectedly share a subject")
    observed_subjects = {
        context: len(subjects) for context, subjects in by_context.items()
    }
    if observed_subjects != EXPECTED_SUBJECTS:
        raise ValueError(
            "MS primary subject support differs: "
            f"{observed_subjects} != {EXPECTED_SUBJECTS}"
        )
    if result.shape != EXPECTED_PRIMARY_SHAPE:
        raise ValueError(
            f"unexpected MS primary shape: {result.shape}"
        )

    result.uns["crychic_analysis_subset"] = {
        "schema_version": "crychic-prepared-subset-v1",
        "dataset_id": "UCSC_Lerma_Martin_MS_snRNA_CA_vs_Ctrl",
        "source_filename": input_path.name,
        "source_sha256": source_sha256,
        "selection_field": "lesion_type",
        "selection_values": list(PRIMARY_CONTEXTS),
        "preserve_observation_order": True,
        "preserve_variable_axis": True,
        "design": "independent subjects; ~ batch + lesion_type",
        "primary_contrast": "CA_vs_Ctrl",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path, compression="gzip", compression_opts=4)
    output_sha256 = _sha256(output_path)
    summary = {
        "output": output_path.name,
        "output_sha256": output_sha256,
        "shape": [result.n_obs, result.n_vars],
        "subjects_by_context": {
            context: len(subjects) for context, subjects in by_context.items()
        },
        "samples_by_context": {
            str(context): int(group["sample_id"].nunique())
            for context, group in samples.groupby("lesion_type", observed=True)
        },
        "lineage": result.uns["crychic_analysis_subset"],
    }
    output_path.with_suffix(".subset.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    subset_ms_primary(
        args.input_h5ad,
        args.output_h5ad,
        expected_sha256=args.expected_sha256,
    )


if __name__ == "__main__":
    main()
