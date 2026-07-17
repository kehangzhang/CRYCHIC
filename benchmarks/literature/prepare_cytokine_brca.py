"""Prepare Wu BRCA subtypes and reconstruct Dimitrov CytoSig activity truth."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import io
import json
import resource
import tarfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from benchmarks.openproblems.common import sha256_file, write_json

FIGURE6A_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41467-022-30755-0/MediaObjects/"
    "41467_2022_30755_MOESM4_ESM.xlsx"
)
TRUTH_STATUS = "reconstructed_clean_activity_truth_not_historical_exact"
EXPECTED_FIGURE6A_DATASETS = {
    "HER2+ Breast Cancer",
    "Triple N. Breast Cancer",
}
EXPECTED_FIGURE6A_METHODS = {
    "CellChat",
    "CellPhoneDB",
    "Connectome",
    "Consensus*",
    "Crosstalk scores",
    "LogFC Mean",
    "NATMI",
    "SingleCellSignalR",
}
EXPECTED_FIGURE6A_CUTOFFS = {100, 250, 500, 1000, 2500, 5000, 10000}
ALIASES_AND_FAMILIES = {
    "CD40L": ("CD40LG",),
    # This spelling follows the released author code. GCSF is therefore not remapped.
    "GSFC": ("CSF3",),
    "IFN1": (
        "IFNA1",
        "IFNA2",
        "IFNA10",
        "IFNA7",
        "IFNA21",
        "IFNA5",
        "IFNA14",
        "IFNA17",
        "IFNA6",
        "IFNA4",
        "IFNA16",
        "IFNA8",
    ),
    "IFNL": ("IFNL1", "IFNL2", "IFNL3", "IFNL4"),
    "IL12": ("IL12A", "IL12B"),
    "IL36": ("IL36A", "IL36B", "IL36G", "IL36RN"),
    "MCSF": ("CSF1",),
    "TNFA": ("TNF", "TNFA"),
    "TRAIL": ("TNFSF10",),
    "TWEAK": ("TNFSF12",),
}


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _member(handle: tarfile.TarFile, suffix: str) -> BinaryIO:
    matches = [item for item in handle.getmembers() if item.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one archive member ending in {suffix!r}, found {len(matches)}"
        )
    stream = handle.extractfile(matches[0])
    if stream is None:
        raise ValueError(f"unable to read archive member: {matches[0].name}")
    return stream


def _normalise_author_label(value: object) -> str:
    result = str(value)
    for old, new in (("/", "."), (" ", "."), ("+", ""), ("-", "."), ("_", ".")):
        result = result.replace(old, new)
    return result


def _read_combined_metadata(archive: Path) -> pd.DataFrame:
    with tarfile.open(archive, "r:gz") as handle:
        with _member(handle, "/metadata.csv") as stream:
            metadata = pd.read_csv(stream, index_col=0)
    required = {"orig.ident", "subtype", "celltype_minor"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Wu metadata is missing columns: {sorted(missing)}")
    if metadata.index.has_duplicates:
        raise ValueError("Wu metadata contains duplicate cell identifiers")
    metadata.index = metadata.index.astype(str)
    return metadata


def _sample_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result: dict[str, tarfile.TarInfo] = {}
    for item in archive.getmembers():
        if (
            not item.isfile()
            or not item.name.endswith(".tar.gz")
            or "_CID" not in item.name
        ):
            continue
        sample_id = item.name.removesuffix(".tar.gz").rsplit("_", maxsplit=1)[-1]
        if sample_id in result:
            raise ValueError(f"duplicate raw archive for sample {sample_id}")
        result[sample_id] = item
    return result


def _read_lines(stream: BinaryIO) -> list[str]:
    return [line.decode("utf-8").strip() for line in stream if line.strip()]


def _read_sample(
    outer: tarfile.TarFile,
    item: tarfile.TarInfo,
    sample_id: str,
) -> tuple[sparse.csr_matrix, pd.Index, pd.Index]:
    stream = outer.extractfile(item)
    if stream is None:
        raise ValueError(f"unable to read raw archive: {item.name}")
    payload = stream.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as inner:
        with _member(inner, "/count_matrix_sparse.mtx") as matrix_stream:
            kwargs = (
                {"spmatrix": True}
                if "spmatrix" in inspect.signature(mmread).parameters
                else {}
            )
            matrix = mmread(matrix_stream, **kwargs)
        with _member(inner, "/count_matrix_genes.tsv") as gene_stream:
            genes = pd.Index(_read_lines(gene_stream), dtype=object)
        with _member(inner, "/count_matrix_barcodes.tsv") as barcode_stream:
            barcodes = pd.Index(_read_lines(barcode_stream), dtype=object)
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    matrix = matrix.T.tocsr()
    if matrix.shape != (len(barcodes), len(genes)):
        raise ValueError(
            f"{sample_id} matrix dimensions {matrix.shape} do not match "
            f"{len(barcodes)} barcodes x {len(genes)} genes"
        )
    if matrix.data.size and (
        (matrix.data < 0).any()
        or not np.equal(matrix.data, np.floor(matrix.data)).all()
    ):
        raise ValueError(f"{sample_id} matrix is not non-negative integer counts")
    return matrix.astype(np.int32), genes, barcodes


def build_brca_input(
    raw_archive: Path,
    combined_archive: Path,
    *,
    subtype: str,
    min_celltype_cells: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    if min_celltype_cells < 1:
        raise ValueError("min_celltype_cells must be positive")
    metadata = _read_combined_metadata(combined_archive)
    normalised_subtype = metadata["subtype"].map(_normalise_author_label)
    selected = metadata.loc[normalised_subtype.eq(subtype)].copy()
    if selected.empty:
        available = sorted(set(normalised_subtype))
        raise ValueError(f"subtype {subtype!r} is absent; available={available}")
    sample_ids = sorted(selected["orig.ident"].astype(str).unique())
    matrices: list[sparse.csr_matrix] = []
    observations: list[pd.DataFrame] = []
    expected_genes: pd.Index | None = None
    with tarfile.open(raw_archive, "r:") as outer:
        members = _sample_members(outer)
        missing_samples = set(sample_ids).difference(members)
        if missing_samples:
            raise ValueError(
                f"raw archive is missing samples: {sorted(missing_samples)}"
            )
        for sample_id in sample_ids:
            matrix, genes, barcodes = _read_sample(outer, members[sample_id], sample_id)
            if expected_genes is None:
                expected_genes = genes
                if expected_genes.has_duplicates:
                    raise ValueError("Wu gene symbols are not unique")
            elif not genes.equals(expected_genes):
                raise ValueError(f"gene order differs in sample {sample_id}")
            prefixed = pd.Index(
                [
                    value
                    if str(value).startswith(f"{sample_id}_")
                    else f"{sample_id}_{value}"
                    for value in barcodes
                ],
                dtype=object,
            )
            sample_metadata = selected.loc[
                selected["orig.ident"].astype(str).eq(sample_id)
            ]
            keep = prefixed.isin(sample_metadata.index)
            observed_ids = prefixed[keep]
            if len(observed_ids) != len(sample_metadata) or set(observed_ids) != set(
                sample_metadata.index
            ):
                raise ValueError(
                    f"{sample_id} raw barcodes do not match combined metadata: "
                    f"matrix matches={len(observed_ids)}, "
                    f"metadata={len(sample_metadata)}"
                )
            matrices.append(matrix[keep, :])
            observations.append(sample_metadata.loc[observed_ids].copy())
    if expected_genes is None:
        raise RuntimeError("no BRCA sample matrices were read")
    counts = sparse.vstack(matrices, format="csr", dtype=np.int32)
    obs = pd.concat(observations, axis=0)
    obs["label"] = obs["celltype_minor"].map(_normalise_author_label)
    raw_cluster_sizes = obs["label"].value_counts().sort_index()
    retained_labels = raw_cluster_sizes.loc[
        raw_cluster_sizes.ge(min_celltype_cells)
    ].index
    keep_cells = obs["label"].isin(retained_labels).to_numpy()
    obs = obs.loc[keep_cells].copy()
    counts = counts[keep_cells, :]
    for column in ("subtype", "celltype_subset", "celltype_minor", "celltype_major"):
        if column in obs:
            obs[column] = obs[column].map(_normalise_author_label)
    obs["label"] = pd.Categorical(obs["label"], categories=sorted(retained_labels))
    for column in obs.columns:
        if isinstance(obs[column].dtype, pd.StringDtype):
            obs[column] = obs[column].astype(object)
    var = pd.DataFrame(index=pd.Index(expected_genes.astype(str), name="gene_symbol"))
    data = ad.AnnData(X=counts, obs=obs, var=var)
    data.uns["cytosig_preparation"] = {
        "dataset_id": f"GSE176078_Wu2021_{subtype}",
        "subtype": subtype,
        "cell_type_key": "label",
        "counts_semantics": "raw non-negative integer UMI",
        "author_label_formatting": "slash/space/hyphen/underscore to dot; plus removed",
        "author_cell_type_filter": f"at least {min_celltype_cells} cells",
    }
    audit = {
        "subtype": subtype,
        "sample_ids": sample_ids,
        "raw_cells": len(selected),
        "prepared_cells": int(data.n_obs),
        "genes": int(data.n_vars),
        "raw_cell_types": len(raw_cluster_sizes),
        "prepared_cell_types": len(retained_labels),
        "raw_cluster_sizes": {
            str(key): int(value) for key, value in raw_cluster_sizes.items()
        },
        "prepared_cluster_sizes": {
            str(key): int(value)
            for key, value in obs["label"].value_counts().sort_index().items()
        },
        "removed_cell_types": sorted(
            set(raw_cluster_sizes.index) - set(retained_labels)
        ),
    }
    return data, audit


def build_cytosig_network(
    signature_path: Path,
    *,
    top_targets: int,
) -> pd.DataFrame:
    if top_targets < 1:
        raise ValueError("top_targets must be positive")
    signatures = pd.read_csv(signature_path, sep="\t", index_col=0)
    signatures.index = signatures.index.astype(str)
    if signatures.index.has_duplicates or signatures.columns.has_duplicates:
        raise ValueError("CytoSig signature matrix has duplicate genes or signatures")
    rows: list[pd.DataFrame] = []
    for raw_source in signatures.columns:
        values = pd.to_numeric(signatures[raw_source], errors="raise")
        order = values.abs().sort_values(ascending=False, kind="stable").index
        selected = values.loc[order[:top_targets]]
        source = "Activin_A" if str(raw_source) == "Activin A" else str(raw_source)
        rows.append(
            pd.DataFrame(
                {
                    "source": source,
                    "target": selected.index.astype(str),
                    "weight": selected.to_numpy(float),
                    "absolute_rank": np.arange(1, len(selected) + 1),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["source", "target"]).any():
        raise ValueError(
            "formatted CytoSig network contains duplicate source-target rows"
        )
    return result


def _bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if not valid.any():
        return result
    ordered = numeric.loc[valid].sort_values(kind="stable")
    count = len(ordered)
    adjusted = (ordered.to_numpy() * count / np.arange(1, count + 1))[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


MlmCallable = Callable[
    [pd.DataFrame, pd.DataFrame, int], tuple[pd.DataFrame, pd.DataFrame]
]


def _default_mlm(
    expression: pd.DataFrame,
    network: pd.DataFrame,
    tmin: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import decoupler as dc

    return dc.mt.mlm(
        expression,
        network.loc[:, ["source", "target", "weight"]],
        tmin=tmin,
        raw=False,
        empty=True,
        verbose=False,
    )


def infer_cytosig_activity(
    data: ad.AnnData,
    network: pd.DataFrame,
    *,
    detection_fraction: float,
    minimum_sum: int,
    tmin: int,
    mlm: MlmCallable | None = None,
) -> pd.DataFrame:
    if not 0.0 <= detection_fraction <= 1.0:
        raise ValueError("detection_fraction must be between zero and one")
    if minimum_sum < 1 or tmin < 1:
        raise ValueError("minimum_sum and tmin must be positive")
    if "label" not in data.obs:
        raise ValueError("prepared BRCA input lacks label")
    counts = sparse.csr_matrix(data.X)
    labels = data.obs["label"].astype(str)
    mlm = mlm or _default_mlm
    expected_sources = sorted(network["source"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for label in sorted(labels.unique()):
        selected = counts[labels.eq(label).to_numpy(), :]
        detected = np.asarray(selected.getnnz(axis=0)).ravel() / selected.shape[0]
        summed = np.asarray(selected.sum(axis=0)).ravel()
        keep = (detected >= detection_fraction) & (summed >= minimum_sum)
        genes = data.var_names[keep].astype(str)
        expression = pd.DataFrame([np.log2(summed[keep])], index=[label], columns=genes)
        available = set(genes)
        local_network = network.loc[network["target"].isin(available)]
        score, p_value = mlm(expression, local_network, tmin)
        for source in expected_sources:
            observed = source in score.columns and source in p_value.columns
            rows.append(
                {
                    "target": label,
                    "signature": source,
                    "activity_score": float(score.at[label, source])
                    if observed
                    else np.nan,
                    "p_value": float(p_value.at[label, source]) if observed else np.nan,
                    "status": "observed" if observed else "not_estimable",
                    "cells": int(selected.shape[0]),
                    "retained_genes": int(keep.sum()),
                }
            )
    result = pd.DataFrame.from_records(rows)
    result["p_value_bh_clean_global"] = _bh(result["p_value"])
    result["response"] = (
        result["activity_score"].gt(0) & result["p_value_bh_clean_global"].le(0.05)
    ).astype(int)
    if result.duplicated(["signature", "target"]).any():
        raise RuntimeError(
            "CytoSig activity output is not unique by signature and target"
        )
    return result.sort_values(["target", "signature"], kind="stable", ignore_index=True)


def build_activity_truth(
    activity: pd.DataFrame,
    resource_path: Path,
) -> pd.DataFrame:
    resource = pd.read_parquet(resource_path, columns=["ligand"])
    resource_ligands = set(resource["ligand"].astype(str))
    rows: list[dict[str, Any]] = []
    for row in activity.itertuples(index=False):
        aliases = ALIASES_AND_FAMILIES.get(row.signature, (row.signature,))
        for ligand in aliases:
            if ligand in resource_ligands:
                rows.append(
                    {
                        "ligand": ligand,
                        "target": row.target,
                        "response": int(row.response),
                    }
                )
    truth = pd.DataFrame.from_records(rows).drop_duplicates()
    if truth.empty:
        raise RuntimeError("no CytoSig aliases overlap the benchmark LR resource")
    conflicts = truth.groupby(["ligand", "target"], observed=True)["response"].nunique()
    if conflicts.gt(1).any():
        raise RuntimeError("CytoSig aliases create conflicting binary truth rows")
    truth = truth.drop_duplicates(["ligand", "target"])
    return truth.sort_values(["target", "ligand"], kind="stable", ignore_index=True)


def validate_figure6a(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "method_name",
        "pval",
        "odds_ratio",
        "padj",
        "n_rank",
        "dataset",
        "max_rank",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Figure6A is missing columns: {sorted(missing)}")
    result = table.loc[:, sorted(required)].copy()
    if set(result["dataset"].astype(str)) != EXPECTED_FIGURE6A_DATASETS:
        raise ValueError(
            "Figure6A does not contain the expected HER2 and TNBC datasets"
        )
    if set(result["method_name"].astype(str)) != EXPECTED_FIGURE6A_METHODS:
        raise ValueError("Figure6A method set does not match the published eight arms")
    cutoffs = set(pd.to_numeric(result["n_rank"], errors="raise").astype(int))
    if cutoffs != EXPECTED_FIGURE6A_CUTOFFS:
        raise ValueError("Figure6A rank cutoffs do not match the publication")
    keys = ["dataset", "method_name", "n_rank"]
    if len(result) != 112 or result.duplicated(keys).any():
        raise ValueError("Figure6A must contain 112 unique dataset-method-cutoff rows")
    return result.sort_values(keys, kind="stable", ignore_index=True)


def _official_figure6a(
    *,
    source_data_xlsx: Path | None,
    url: str,
    output_xlsx: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if source_data_xlsx is None:
        request = urllib.request.Request(
            url, headers={"User-Agent": "CRYCHIC-benchmark/1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        source = url
    else:
        payload = source_data_xlsx.read_bytes()
        source = str(source_data_xlsx.resolve())
    if not payload.startswith(b"PK"):
        raise ValueError("Dimitrov source data is not an XLSX archive")
    output_xlsx.write_bytes(payload)
    table = pd.read_excel(io.BytesIO(payload), sheet_name="Figure6A")
    return validate_figure6a(table), {
        "source": source,
        "filename": output_xlsx.name,
        "sha256": sha256_file(output_xlsx),
        "bytes": len(payload),
        "sheet": "Figure6A",
    }


def prepare(
    raw_archive: Path,
    combined_archive: Path,
    signature_path: Path,
    resource_dir: Path,
    output_dir: Path,
    *,
    subtype: str,
    min_celltype_cells: int,
    detection_fraction: float,
    minimum_sum: int,
    top_targets: int,
    tmin: int,
    source_data_xlsx: Path | None,
    figure6a_url: str,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "h5ad": output_dir / f"brca_{subtype.lower()}.h5ad",
        "network": output_dir / "cytosig_top500_network.tsv",
        "activity": output_dir / "cytosig_activity.tsv",
        "truth": output_dir / "cytosig_truth.tsv",
        "source_xlsx": output_dir / "dimitrov_2022_source_data.xlsx",
        "figure6a": output_dir / "official_figure6a_reference.tsv",
        "manifest": output_dir / "manifest.json",
    }
    if paths["manifest"].exists() and not overwrite:
        raise FileExistsError(f"BRCA cytokine preparation exists: {paths['manifest']}")
    for path in (raw_archive, combined_archive, signature_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    resource_path = resource_dir / "liana_consensus_human.parquet"
    resource_manifest_path = resource_dir / "resource_manifest.json"
    resource_manifest = json.loads(resource_manifest_path.read_text(encoding="utf-8"))
    if (
        resource_manifest.get("schema_version") != "crychic-cytokine-common-resource-v1"
        or sha256_file(resource_path)
        != resource_manifest["files"]["resource"]["sha256"]
    ):
        raise ValueError("cytokine common resource manifest is invalid")
    started = time.perf_counter()
    data, input_audit = build_brca_input(
        raw_archive,
        combined_archive,
        subtype=subtype,
        min_celltype_cells=min_celltype_cells,
    )
    data.write_h5ad(paths["h5ad"], compression="gzip")
    network = build_cytosig_network(signature_path, top_targets=top_targets)
    network.to_csv(paths["network"], sep="\t", index=False)
    activity = infer_cytosig_activity(
        data,
        network,
        detection_fraction=detection_fraction,
        minimum_sum=minimum_sum,
        tmin=tmin,
    )
    activity.to_csv(paths["activity"], sep="\t", index=False, na_rep="")
    truth = build_activity_truth(activity, resource_path)
    truth.to_csv(paths["truth"], sep="\t", index=False)
    figure6a, figure6a_source = _official_figure6a(
        source_data_xlsx=source_data_xlsx,
        url=figure6a_url,
        output_xlsx=paths["source_xlsx"],
    )
    figure6a.to_csv(paths["figure6a"], sep="\t", index=False)
    elapsed = time.perf_counter() - started
    output_records = {
        key: {
            "filename": path.name,
            "sha256": sha256_file(path),
            **(
                {"rows": len(table)}
                if (
                    table := {
                        "network": network,
                        "activity": activity,
                        "truth": truth,
                        "figure6a": figure6a,
                    }.get(key)
                )
                is not None
                else {}
            ),
        }
        for key, path in paths.items()
        if key != "manifest"
    }
    manifest: dict[str, Any] = {
        "schema_version": "crychic-cytokine-brca-preparation-v1",
        "status": "complete",
        "dataset_id": f"GSE176078_Wu2021_{subtype}",
        "truth_status": TRUTH_STATUS,
        "sources": {
            "raw_archive": {
                "filename": raw_archive.name,
                "sha256": sha256_file(raw_archive),
            },
            "combined_archive": {
                "filename": combined_archive.name,
                "sha256": sha256_file(combined_archive),
            },
            "cytosig_signature": {
                "filename": signature_path.name,
                "sha256": sha256_file(signature_path),
                "expected_upstream_commit": "b321adae9b4d1a0499f628854a9682d01491fd87",
            },
            "resource": {
                "id": resource_manifest["resource_id"],
                "sha256": sha256_file(resource_path),
            },
            "official_figure6a": figure6a_source,
        },
        "input_audit": input_audit,
        "activity_protocol": {
            "pseudobulk": "sum raw counts independently within each cell type",
            "gene_filter": (
                f"detected in >= {detection_fraction:.3g} of cells and summed counts "
                f">= {minimum_sum}"
            ),
            "transform": "log2 summed counts",
            "model": "decoupler.mt.mlm multivariate linear model",
            "signature_targets": top_targets,
            "minimum_network_targets": tmin,
            "multiplicity": (
                "BH once over unique signature x target-cell-type activities"
            ),
            "positive_definition": "activity_score > 0 and clean-global BH q <= 0.05",
            "aliases": "released Dimitrov cytosig_src.R aliases, including GSFC typo",
        },
        "truth": {
            "status": TRUTH_STATUS,
            "rows": len(truth),
            "ligands": int(truth["ligand"].nunique()),
            "targets": int(truth["target"].nunique()),
            "positive_rows": int(truth["response"].sum()),
        },
        "historical_comparability": {
            "official_figure6a_is_reference_only": True,
            "skipped_unavailable_arms": ["Crosstalk scores"],
            "limitations": [
                "The paper LIANA 0.0.5 OmniPath resource snapshot is unavailable.",
                "The original preprocessed Seurat object is unavailable.",
                (
                    "The released author code adjusts activity p-values after "
                    "LR-row expansion; this reconstruction uses a leakage-free "
                    "unique activity universe."
                ),
                (
                    "Current method scores must not be compared as exact "
                    "replications of the historical Figure6A arms."
                ),
            ],
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "versions": {
            name: _version(name)
            for name in ("anndata", "decoupler", "numpy", "pandas", "scipy")
        },
        "outputs": output_records,
    }
    write_json(paths["manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_archive", type=Path)
    parser.add_argument("combined_archive", type=Path)
    parser.add_argument("signature_path", type=Path)
    parser.add_argument("resource_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--subtype", default="HER2")
    parser.add_argument("--min-celltype-cells", type=int, default=25)
    parser.add_argument("--detection-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-sum", type=int, default=5)
    parser.add_argument("--top-targets", type=int, default=500)
    parser.add_argument("--tmin", type=int, default=5)
    parser.add_argument("--source-data-xlsx", type=Path)
    parser.add_argument("--figure6a-url", default=FIGURE6A_URL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(
        args.raw_archive,
        args.combined_archive,
        args.signature_path,
        args.resource_dir,
        args.output_dir,
        subtype=args.subtype,
        min_celltype_cells=args.min_celltype_cells,
        detection_fraction=args.detection_fraction,
        minimum_sum=args.minimum_sum,
        top_targets=args.top_targets,
        tmin=args.tmin,
        source_data_xlsx=args.source_data_xlsx,
        figure6a_url=args.figure6a_url,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
