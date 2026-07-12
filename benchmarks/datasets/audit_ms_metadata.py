"""Audit the official UCSC Lerma-Martin MS snRNA cell metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_MD5 = "74e189fcd12d0e3ec6b1449e3dd810f6"
EXPECTED_COLUMNS = {
    "cellId",
    "patient_id",
    "sample_id",
    "condition",
    "lesion_type",
    "celltype",
    "subtype",
}
EXPECTED_SUBJECTS = {"Ctrl": 6, "CA": 5, "CI": 2}
EXPECTED_SAMPLES = {"Ctrl": 6, "CA": 6, "CI": 4}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_ms_metadata(
    metadata_path: Path, output_design: Path, output_summary: Path
) -> None:
    """Validate identifiers and write a compact non-clinical design summary."""
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    observed_md5 = _digest(metadata_path, "md5")
    if observed_md5 != EXPECTED_MD5:
        raise ValueError(
            f"unexpected UCSC metadata MD5: {observed_md5} != {EXPECTED_MD5}"
        )
    metadata = pd.read_csv(metadata_path, sep="\t")
    missing = EXPECTED_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"UCSC MS metadata is missing {sorted(missing)}")
    if len(metadata) != 103_794 or metadata["cellId"].duplicated().any():
        raise ValueError("unexpected MS metadata row count or duplicate cell IDs")

    sample_columns = ["patient_id", "sample_id", "condition", "lesion_type"]
    sample_design = (
        metadata.groupby(sample_columns, dropna=False)
        .size()
        .rename("n_nuclei")
        .reset_index()
        .sort_values(sample_columns)
        .reset_index(drop=True)
    )
    if metadata.groupby("sample_id")[sample_columns].nunique().to_numpy().max() != 1:
        raise ValueError("sample ID does not map uniquely to subject and context")

    subjects = (
        metadata.groupby("lesion_type")["patient_id"].nunique().sort_index().to_dict()
    )
    samples = (
        metadata.groupby("lesion_type")["sample_id"].nunique().sort_index().to_dict()
    )
    if subjects != EXPECTED_SUBJECTS or samples != EXPECTED_SAMPLES:
        raise ValueError(
            f"unexpected subject/sample support: subjects={subjects}, samples={samples}"
        )

    major_support = (
        metadata.groupby(["lesion_type", "celltype"], observed=True)
        .agg(
            n_nuclei=("cellId", "size"),
            n_subjects=("patient_id", "nunique"),
            n_samples=("sample_id", "nunique"),
        )
        .reset_index()
        .sort_values(["lesion_type", "celltype"])
    )
    output_design.parent.mkdir(parents=True, exist_ok=True)
    sample_design.to_csv(output_design, sep="\t", index=False)
    major_support_path = output_design.with_name("ms_major_celltype_support.tsv")
    major_support.to_csv(major_support_path, sep="\t", index=False)

    summary = {
        "dataset_id": "UCSC_Lerma_Martin_MS_snRNA",
        "source_url": (
            "https://cells.ucsc.edu/ms-subcortical-lesions/snrna-atlas/meta.tsv"
        ),
        "source_md5": observed_md5,
        "source_sha256": _digest(metadata_path, "sha256"),
        "n_nuclei": len(metadata),
        "n_subjects": int(metadata["patient_id"].nunique()),
        "n_samples": int(metadata["sample_id"].nunique()),
        "n_major_cell_types": int(metadata["celltype"].nunique()),
        "n_subtypes": int(metadata["subtype"].nunique(dropna=True)),
        "subjects_by_context": subjects,
        "samples_by_context": samples,
        "primary_contrast": "CA_vs_Ctrl",
        "secondary_contrasts": ["CI_vs_Ctrl", "CA_vs_CI"],
        "inferential_warning": (
            "CI has only two independent subjects; three-group and CI contrasts "
            "are descriptive/secondary"
        ),
        "local_alignment_warning": (
            "GSE279183 sample-level 10x matrices under dataset/ are GEO snRNA-seq "
            "raw matrices, but their barcodes overlap this UCSC processed atlas only "
            "partially; do not join annotations without a validated alignment policy"
        ),
        "clinical_columns_excluded_from_design_outputs": [
            "age",
            "sex",
            "rin",
            "pmi_hrs",
            "duration_y",
            "ms_class",
            "cause_death",
        ],
        "outputs": [output_design.name, major_support_path.name],
    }
    output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_tsv", type=Path)
    parser.add_argument("output_design_tsv", type=Path)
    parser.add_argument("output_summary_json", type=Path)
    args = parser.parse_args()
    audit_ms_metadata(
        args.metadata_tsv,
        args.output_design_tsv,
        args.output_summary_json,
    )


if __name__ == "__main__":
    main()
