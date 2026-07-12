"""Prepare the CellPhoneDB trophoblast tutorial for single-context smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_from_barcode(value: str) -> str:
    parts = value.split("_", maxsplit=1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"cannot parse source sample from barcode {value!r}")
    return parts[1]


def prepare_trophoblast(zip_path: Path, output_path: Path) -> ad.AnnData:
    """Use the embedded raw counts without inventing a differential context."""
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    with tempfile.TemporaryDirectory(prefix="crychic-trophoblast-") as temporary:
        with zipfile.ZipFile(zip_path) as archive:
            member = "data/normalised_log_counts.h5ad"
            if member not in archive.namelist():
                raise ValueError(f"tutorial archive is missing {member}")
            archive.extract(member, path=temporary)
        source = ad.read_h5ad(Path(temporary) / member)

    if source.raw is None:
        raise ValueError("tutorial AnnData has no raw counts snapshot")
    counts = source.raw.X
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(np.asarray(counts))
    else:
        counts = counts.tocsr()
    if np.any(counts.data < 0) or np.any(counts.data != np.floor(counts.data)):
        raise ValueError("embedded raw matrix is not non-negative integer counts")
    counts = counts.astype(np.int32)

    obs = source.obs.copy()
    obs["cell_type"] = obs["cell_labels"].astype(str)
    obs["sample_id"] = [_sample_from_barcode(str(value)) for value in obs.index]
    # The tutorial supplies source-sample labels but no repeated-subject map.
    obs["subject_id"] = obs["sample_id"]
    obs["context"] = "maternal_fetal_interface"
    for column in obs.columns:
        if isinstance(obs[column].dtype, pd.StringDtype):
            obs[column] = obs[column].astype(object)
    obs.index = pd.Index(obs.index.astype(str), dtype=object, name="cell_id")

    var = pd.DataFrame(index=pd.Index(source.var_names.astype(str), dtype=object))
    var.index.name = "gene_symbol"
    result = ad.AnnData(X=source.X.copy(), obs=obs, var=var)
    result.layers["counts"] = counts
    result.uns["crychic_conversion"] = {
        "dataset_id": "CellPhoneDB_trophoblast_tutorial",
        "source_archive_sha256": _sha256(zip_path),
        "counts_semantics": "integer counts from embedded AnnData.raw.X",
        "x_semantics": "upstream log-normalized expression",
        "subject_semantics": (
            "source sample used as subject; no repeated-context mapping"
        ),
        "analysis_scope": "single-context exploratory smoke and signature validation",
        "formal_differential_eligible": False,
        "reason_code": "no_validated_condition_or_repeated_subject_mapping",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path, compression="gzip")
    summary = {
        "output": output_path.name,
        "shape": [result.n_obs, result.n_vars],
        "source_samples": int(obs["sample_id"].nunique()),
        "cell_types": int(obs["cell_type"].nunique()),
        "counts_nnz": int(counts.nnz),
        "provenance": result.uns["crychic_conversion"],
    }
    output_path.with_suffix(".conversion.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tutorial_zip", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    args = parser.parse_args()
    prepare_trophoblast(args.tutorial_zip, args.output_h5ad)


if __name__ == "__main__":
    main()
