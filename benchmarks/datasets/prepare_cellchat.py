"""Convert official CellChat objects to normalized-only benchmark AnnData."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import anndata as ad
import pandas as pd
from scipy import sparse
from scipy.io import mmread

DatasetKind = Literal["ad_skin", "embryonic_skin"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_object(
    rscript: Path, export_script: Path, source: Path, output: Path
) -> ad.AnnData:
    subprocess.run(
        [str(rscript), str(export_script), str(source), str(output)],
        check=True,
    )
    matrix = mmread(output / "expression.mtx")
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    matrix = matrix.T.tocsr().astype("float32")
    genes = (output / "genes.txt").read_text(encoding="utf-8").splitlines()
    metadata = pd.read_csv(output / "metadata.tsv", sep="\t")
    if matrix.shape != (len(metadata), len(genes)):
        raise ValueError(
            f"CellChat export shape mismatch: {matrix.shape} vs "
            f"{len(metadata)} cells x {len(genes)} genes"
        )
    metadata.index = pd.Index(metadata.pop("cell_id").astype(str), dtype=object)
    var = pd.DataFrame(index=pd.Index(genes, dtype=object, name="gene_symbol"))
    return ad.AnnData(X=matrix, obs=metadata, var=var)


def prepare_cellchat(
    sources: tuple[Path, Path],
    output_path: Path,
    *,
    kind: DatasetKind,
    rscript: Path,
    export_script: Path,
) -> ad.AnnData:
    """Prepare AD or embryonic skin without inventing count semantics."""
    for path in (*sources, rscript, export_script):
        if not path.is_file():
            raise FileNotFoundError(path)
    labels = ("NL", "LS") if kind == "ad_skin" else ("E13.5", "E14.5")
    objects: list[ad.AnnData] = []
    with tempfile.TemporaryDirectory(prefix="crychic-cellchat-") as temporary:
        base = Path(temporary)
        for index, (source, label) in enumerate(zip(sources, labels, strict=True)):
            current = _export_object(
                rscript, export_script, source, base / f"object-{index}"
            )
            current.obs["context"] = label
            if kind == "ad_skin":
                required = {"patient.id", "condition", "labels"}
                if not required.issubset(current.obs.columns):
                    raise ValueError(f"AD CellChat metadata is missing {required}")
                current.obs["sample_id"] = (
                    current.obs["patient.id"].astype(str)
                    + ":"
                    + current.obs["condition"].astype(str)
                )
                current.obs["subject_id"] = current.obs["patient.id"].astype(str)
                current.obs["cell_type"] = current.obs["labels"].astype(str)
            else:
                if "labels" not in current.obs.columns:
                    raise ValueError("embryonic CellChat metadata lacks labels")
                replicate = current.obs_names.to_series().str.split("_", n=1).str[0]
                current.obs["sample_id"] = label + ":" + replicate.astype(str)
                current.obs["subject_id"] = current.obs["sample_id"].astype(str)
                current.obs["cell_type"] = current.obs["labels"].astype(str)
            objects.append(current)

    result = ad.concat(
        objects,
        join="outer",
        merge="same",
        label="source_context",
        keys=labels,
        index_unique=None,
        fill_value=0.0,
    )
    for column in result.obs.columns:
        if isinstance(result.obs[column].dtype, pd.StringDtype):
            result.obs[column] = result.obs[column].astype(object)
    result.obs_names = pd.Index(result.obs_names.astype(str), dtype=object)
    result.var_names = pd.Index(result.var_names.astype(str), dtype=object)
    result.uns["crychic_conversion"] = {
        "dataset_id": (
            "CellChat_AD_skin_NL_LS"
            if kind == "ad_skin"
            else "CellChat_embryonic_skin_E13_E14"
        ),
        "source_sha256": {
            label: _sha256(path)
            for label, path in zip(labels, sources, strict=True)
        },
        "expression_semantics": "upstream CellChat normalized @data",
        "expression_transform": "log1p_normalized",
        "counts_available": False,
        "formal_differential_eligible": False,
        "reason_code": (
            "normalized_only_and_unpaired_condition_specific_subjects"
            if kind == "ad_skin"
            else "normalized_only_and_no_independent_stage_replicates"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path, compression="gzip")
    summary = {
        "output": output_path.name,
        "shape": [result.n_obs, result.n_vars],
        "samples": int(result.obs["sample_id"].nunique()),
        "subjects": int(result.obs["subject_id"].nunique()),
        "cell_types": int(result.obs["cell_type"].nunique()),
        "contexts": labels,
        "provenance": result.uns["crychic_conversion"],
    }
    output_path.with_suffix(".conversion.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["ad_skin", "embryonic_skin"])
    parser.add_argument("source_a", type=Path)
    parser.add_argument("source_b", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument(
        "--export-script",
        type=Path,
        default=Path(__file__).with_name("export_cellchat_expression.R"),
    )
    args = parser.parse_args()
    prepare_cellchat(
        (args.source_a, args.source_b),
        args.output_h5ad,
        kind=args.kind,
        rscript=args.rscript,
        export_script=args.export_script,
    )


if __name__ == "__main__":
    main()
