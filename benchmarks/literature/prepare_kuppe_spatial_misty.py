"""Prepare compact Kuppe spatial inputs for a pinned MISTy recomputation.

The public CELLxGENE H5AD files contain the cell2location proportions in
``obs``.  This preparer intentionally never reads the gene-expression matrix.
It exports Visium array coordinates, not image-pixel coordinates, because the
original Kuppe helper requested the Seurat ``row``/``col`` geometry.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd

DATASET_ID = "Kuppe_MI_spatial_CTRL_vs_IZ"
PREPARATION_SCHEMA = "crychic-kuppe-spatial-misty-input-v1"
COHORT_MANIFEST_NAME = "kuppe_spatial_misty_input.manifest.json"

ABUNDANCE_COLUMNS = (
    "Adipocyte",
    "Cardiomyocyte",
    "Endothelial",
    "Fibroblast",
    "Lymphoid",
    "Mast",
    "Myeloid",
    "Neuronal",
    "Pericyte",
    "Cycling.cells",
    "vSMCs",
)

# The 4 CTRL and 9 IZ slides used by the paper are frozen here.  Do not infer a
# condition from a filename: a typo or an extra file must fail preparation.
FROZEN_SLIDES: tuple[tuple[str, str], ...] = (
    ("control_P1.cellxgene.h5ad", "CTRL"),
    ("control_P17.cellxgene.h5ad", "CTRL"),
    ("control_P7.cellxgene.h5ad", "CTRL"),
    ("control_P8.cellxgene.h5ad", "CTRL"),
    ("GT_IZ_P13.cellxgene.h5ad", "IZ"),
    ("GT_IZ_P15.cellxgene.h5ad", "IZ"),
    ("GT_IZ_P9.cellxgene.h5ad", "IZ"),
    ("GT_IZ_P9_rep2.cellxgene.h5ad", "IZ"),
    ("IZ_BZ_P2.cellxgene.h5ad", "IZ"),
    ("IZ_P10.cellxgene.h5ad", "IZ"),
    ("IZ_P15.cellxgene.h5ad", "IZ"),
    ("IZ_P16.cellxgene.h5ad", "IZ"),
    ("IZ_P3.cellxgene.h5ad", "IZ"),
)


@dataclass(frozen=True)
class PreparedSlide:
    """Checksum-bound summary of one prepared spatial slide."""

    sample_id: str
    condition: str
    source_filename: str
    source_sha256: str
    output_filename: str
    output_sha256: str
    checksum_filename: str
    n_spots_total: int
    n_spots_in_tissue: int
    n_spots_complete: int
    n_spots_retained: int

    def to_manifest(self) -> dict[str, object]:
        """Return a JSON-serializable record consumed by the R runner."""

        return {
            "sample_id": self.sample_id,
            "condition": self.condition,
            "source": {
                "filename": self.source_filename,
                "sha256": self.source_sha256,
            },
            "input": {
                "filename": self.output_filename,
                "sha256": self.output_sha256,
                "checksum_filename": self.checksum_filename,
            },
            "spots": {
                "total": self.n_spots_total,
                "in_tissue": self.n_spots_in_tissue,
                "complete_abundance": self.n_spots_complete,
                "retained": self.n_spots_retained,
                "excluded_out_of_tissue": (self.n_spots_total - self.n_spots_in_tissue),
                "excluded_in_tissue_incomplete_abundance": (
                    self.n_spots_in_tissue - self.n_spots_retained
                ),
            },
            "status": "complete",
        }


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


def _payload_sha256(manifest: dict[str, object]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_payload_sha256"
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sample_id(source_filename: str) -> str:
    suffix = ".cellxgene.h5ad"
    if not source_filename.endswith(suffix):
        raise ValueError(
            f"frozen Kuppe slide filename must end with {suffix!r}: {source_filename!r}"
        )
    return source_filename[: -len(suffix)]


def _validate_roster(source_dir: Path) -> None:
    expected = {name for name, _ in FROZEN_SLIDES}
    observed = {path.name for path in source_dir.glob("*.h5ad")}
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if missing or unexpected:
        raise ValueError(
            "Kuppe spatial slide roster mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _coerce_binary_in_tissue(values: pd.Series, *, sample_id: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        observed = sorted(values.drop_duplicates().astype(str).tolist())
        raise ValueError(
            f"slide {sample_id!r} has invalid in_tissue values: {observed}"
        )
    return numeric.astype(bool)


def _validated_spot_table(source: ad.AnnData, *, sample_id: str) -> pd.DataFrame:
    required = {"array_row", "array_col", "in_tissue", *ABUNDANCE_COLUMNS}
    missing = required.difference(source.obs.columns)
    if missing:
        raise ValueError(
            f"slide {sample_id!r} obs is missing required fields: {sorted(missing)}"
        )
    if source.obs_names.has_duplicates:
        raise ValueError(f"slide {sample_id!r} has duplicate spot identifiers")

    obs = source.obs.loc[
        :, ["array_row", "array_col", "in_tissue", *ABUNDANCE_COLUMNS]
    ].copy()
    in_tissue = _coerce_binary_in_tissue(obs["in_tissue"], sample_id=sample_id)

    numeric_columns = ["array_row", "array_col", *ABUNDANCE_COLUMNS]
    numeric = obs.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    coordinate_complete = numeric[["array_row", "array_col"]].notna().all(axis=1)
    if not coordinate_complete.all():
        bad = obs.index[~coordinate_complete].astype(str).tolist()[:5]
        raise ValueError(
            f"slide {sample_id!r} has missing/non-numeric array coordinates; "
            f"examples={bad}"
        )
    coordinates = numeric[["array_row", "array_col"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"slide {sample_id!r} has non-finite array coordinates")
    if not np.equal(coordinates, np.floor(coordinates)).all():
        raise ValueError(f"slide {sample_id!r} has non-integer array coordinates")

    abundance = numeric.loc[:, list(ABUNDANCE_COLUMNS)]
    complete = abundance.notna().all(axis=1)
    complete_values = abundance.loc[complete].to_numpy(dtype=float)
    if not np.isfinite(complete_values).all():
        raise ValueError(f"slide {sample_id!r} has non-finite abundance values")
    if (complete_values < 0).any():
        raise ValueError(f"slide {sample_id!r} has negative abundance values")

    keep = in_tissue & complete
    if not keep.any():
        raise ValueError(
            f"slide {sample_id!r} has no in-tissue spots with complete abundance"
        )
    retained = numeric.loc[keep, numeric_columns].copy()
    retained.insert(0, "spot_id", obs.index[keep].astype(str))
    retained["array_row"] = retained["array_row"].astype(np.int64)
    retained["array_col"] = retained["array_col"].astype(np.int64)
    if retained.duplicated(["array_row", "array_col"]).any():
        duplicates = retained.loc[
            retained.duplicated(["array_row", "array_col"], keep=False),
            ["spot_id", "array_row", "array_col"],
        ].head(5)
        raise ValueError(
            f"slide {sample_id!r} has duplicate array geometry: "
            f"{duplicates.to_dict(orient='records')}"
        )
    retained.attrs.update(
        {
            "n_spots_total": len(obs),
            "n_spots_in_tissue": int(in_tissue.sum()),
            "n_spots_complete": int(complete.sum()),
            "n_spots_retained": int(keep.sum()),
        }
    )
    return cast(pd.DataFrame, retained)


def _write_deterministic_gzip_tsv(table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                    table.to_csv(text, sep="\t", index=False, lineterminator="\n")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_checksum(path: Path, checksum_path: Path) -> str:
    checksum = _sha256(path)
    temporary = checksum_path.with_name(f".{checksum_path.name}.tmp-{os.getpid()}")
    temporary.write_text(f"{checksum}  {path.name}\n", encoding="ascii")
    temporary.replace(checksum_path)
    return checksum


def _prepare_slide(
    source_path: Path,
    output_dir: Path,
    *,
    condition: str,
) -> PreparedSlide:
    sample_id = _sample_id(source_path.name)
    source_sha256 = _sha256(source_path)
    backed = ad.read_h5ad(source_path, backed="r")
    try:
        table = _validated_spot_table(backed, sample_id=sample_id)
    finally:
        if backed.file is not None:
            backed.file.close()

    output = output_dir / f"{sample_id}.misty_input.tsv.gz"
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    _write_deterministic_gzip_tsv(table, output)
    output_sha256 = _write_checksum(output, checksum_path)

    readback = pd.read_csv(output, sep="\t")
    expected_columns = ["spot_id", "array_row", "array_col", *ABUNDANCE_COLUMNS]
    if readback.columns.tolist() != expected_columns or len(readback) != len(table):
        raise RuntimeError(f"slide {sample_id!r} compact TSV readback failed")

    return PreparedSlide(
        sample_id=sample_id,
        condition=condition,
        source_filename=source_path.name,
        source_sha256=source_sha256,
        output_filename=output.name,
        output_sha256=output_sha256,
        checksum_filename=checksum_path.name,
        n_spots_total=int(table.attrs["n_spots_total"]),
        n_spots_in_tissue=int(table.attrs["n_spots_in_tissue"]),
        n_spots_complete=int(table.attrs["n_spots_complete"]),
        n_spots_retained=int(table.attrs["n_spots_retained"]),
    )


def prepare_kuppe_spatial_misty(
    source_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Prepare all frozen Kuppe spatial slides and return the cohort manifest."""

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Kuppe spatial source directory not found: {source_dir}"
        )
    _validate_roster(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slides = [
        _prepare_slide(source_dir / filename, output_dir, condition=condition)
        for filename, condition in FROZEN_SLIDES
    ]
    manifest: dict[str, Any] = {
        "schema_version": PREPARATION_SCHEMA,
        "dataset_id": DATASET_ID,
        "status": "complete",
        "provenance": {
            "source": "public Kuppe CELLxGENE spatial H5AD files",
            "author_protocol_repository": ("https://github.com/saezlab/visium_heart"),
            "author_protocol_commit": ("5b30c7e497e06688a8448afd8d069d2fa70ebcd2"),
            "result_scope": "protocol-level MISTy recomputation input",
            "author_importance_table": False,
            "warning": (
                "Recomputed results must not be labeled as the authors' published "
                "MISTy importance table."
            ),
        },
        "protocol": {
            "abundance_source": "public obs cell2location columns",
            "abundance_columns": list(ABUNDANCE_COLUMNS),
            "abundance_transform": "none",
            "assay_compatibility_note": (
                "The author workflow ran both c2l density and c2l_props assays. "
                "The public CELLxGENE H5AD exposes one 11-column panel whose "
                "complete in-tissue spot sums are approximately one; this "
                "reconstruction uses that public normalized panel unchanged and "
                "does not claim to recover the private raw c2l density assay."
            ),
            "feature_compatibility_note": (
                "The versioned author script excluded a feature named 'prolif'. "
                "The public CELLxGENE export instead exposes the frozen 11-column "
                "benchmark panel including 'Cycling.cells'; no unverified alias "
                "mapping was applied."
            ),
            "geometry_source": ["obs.array_row", "obs.array_col"],
            "geometry_output_columns": ["array_row", "array_col"],
            "pixel_coordinates_used": False,
            "spot_filter": "in_tissue == 1 and all abundance columns complete",
            "mistyR_version": "1.3.5",
            "mistyR_tag_commit": ("19248ea7e02803063d1e1112a8af6c3f06c59e03"),
            "views": {
                "intra": {},
                "juxta_5": {"neighbor.thr": 5},
                "para_15": {"l": 15, "family": "gaussian", "approx": 1},
            },
        },
        "software": {
            "python": platform.python_version(),
            "anndata": importlib.metadata.version("anndata"),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
        },
        "cohort": {
            "conditions": ["CTRL", "IZ"],
            "slides_by_condition": {
                condition: sum(slide.condition == condition for slide in slides)
                for condition in ("CTRL", "IZ")
            },
            "n_slides": len(slides),
            "n_retained_spots": sum(slide.n_spots_retained for slide in slides),
        },
        "slides": [slide.to_manifest() for slide in slides],
    }
    manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
    manifest_path = output_dir / COHORT_MANIFEST_NAME
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = prepare_kuppe_spatial_misty(args.source_dir, args.output_dir)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str((args.output_dir / COHORT_MANIFEST_NAME).resolve()),
                "n_slides": manifest["cohort"]["n_slides"],
                "n_retained_spots": manifest["cohort"]["n_retained_spots"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ABUNDANCE_COLUMNS",
    "COHORT_MANIFEST_NAME",
    "DATASET_ID",
    "FROZEN_SLIDES",
    "PREPARATION_SCHEMA",
    "PreparedSlide",
    "prepare_kuppe_spatial_misty",
]
