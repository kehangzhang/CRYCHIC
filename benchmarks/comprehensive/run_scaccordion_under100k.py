"""Evaluate the eligible scACCorDiON patient-graph cohorts.

The runner consumes one LIANA/CellPhoneDB-style table per biological sample.
It reproduces the paper's patient-graph representation and evaluates all
distance baselines on one frozen sample universe.  The CRYCHIC-labelled arm is
explicitly a CRYCHIC-compatible canonical event projection, not native
CRYCHIC differential inference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.suggest_v5_under100k_contract import load_contract

SCHEMA_VERSION = "crychic-scaccordion-under100k-v1"
METHODS = (
    "scaccordion_dw_ot",
    "corr_ot",
    "got",
    "tabular_pca",
    "crychic_canonical_event_correlation",
)
PAPER_TRUE_K_ARI: dict[str, dict[str, float]] = {
    "scaccordion_pdac": {
        "kbarycenter_corr_ot": -0.020883,
        "kbarycenter_dw_ot": 0.034453,
        "kmeans_tabular_pca": 0.069089,
        "kmedoids_corr_ot": -0.069666,
        "kmedoids_scaccordion_dw_ot": 1.0,
        "kmedoids_got": 0.109974,
        "kmedoids_tabular_pca": 0.066567,
        "leiden_corr_ot": -0.034972,
        "leiden_scaccordion_dw_ot": -0.034972,
        "leiden_got": -0.030510,
        "leiden_tabular_pca": -0.046670,
    },
    "scaccordion_kidney_aki": {
        "kbarycenter_corr_ot": 0.303182,
        "kbarycenter_dw_ot": 0.134250,
        "kmeans_tabular_pca": 0.015940,
        "kmedoids_corr_ot": 0.087983,
        "kmedoids_scaccordion_dw_ot": 0.076468,
        "kmedoids_got": -0.038452,
        "kmedoids_tabular_pca": 0.018215,
        "leiden_corr_ot": -0.014061,
        "leiden_scaccordion_dw_ot": -0.017802,
        "leiden_got": -0.023747,
        "leiden_tabular_pca": -0.057877,
    },
    "scaccordion_rcc": {
        "kbarycenter_corr_ot": 1.0,
        "kbarycenter_dw_ot": 1.0,
        "kmeans_tabular_pca": -0.022337,
        "kmedoids_corr_ot": 1.0,
        "kmedoids_scaccordion_dw_ot": 1.0,
        "kmedoids_got": -0.013245,
        "kmedoids_tabular_pca": -0.013245,
        "leiden_corr_ot": 0.114583,
        "leiden_scaccordion_dw_ot": 0.114583,
        "leiden_got": -0.062500,
        "leiden_tabular_pca": 0.060302,
    },
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _contract_record(benchmark_id: str) -> Mapping[str, Any]:
    contract = load_contract()
    matches = [
        item for item in contract["benchmarks"] if item["benchmark_id"] == benchmark_id
    ]
    if len(matches) != 1 or not benchmark_id.startswith("scaccordion_"):
        raise ValueError(f"unknown scACCorDiON benchmark: {benchmark_id}")
    return matches[0]


def _sample_id(path: Path) -> str:
    name = path.name
    for suffix in (".tsv.gz", ".csv.gz", ".tsv", ".csv"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            return stem.rsplit("|", maxsplit=1)[-1]
    raise ValueError(f"unsupported graph-table suffix: {path}")


def load_graph_tables(graph_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate one patient graph table per sample."""

    paths = sorted(
        path
        for path in graph_dir.iterdir()
        if path.is_file()
        and any(
            path.name.endswith(suffix)
            for suffix in (".csv", ".csv.gz", ".tsv", ".tsv.gz")
        )
    )
    if not paths:
        raise ValueError(f"no graph tables found in {graph_dir}")
    tables: dict[str, pd.DataFrame] = {}
    for path in paths:
        sample_id = _sample_id(path)
        if sample_id in tables:
            raise ValueError(f"duplicate sample table: {sample_id}")
        sep = "\t" if ".tsv" in path.name else ","
        table = pd.read_csv(path, sep=sep)
        required = {"source", "target", "lr_means"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        table = table.copy()
        table["source"] = table["source"].astype(str)
        table["target"] = table["target"].astype(str)
        table["lr_means"] = pd.to_numeric(table["lr_means"], errors="coerce")
        table = table.loc[np.isfinite(table["lr_means"]) & (table["lr_means"] >= 0)]
        if table.empty:
            raise ValueError(f"{path} contains no estimable nonnegative score")
        tables[sample_id] = table.reset_index(drop=True)
    return tables


def load_metadata(path: Path) -> pd.DataFrame:
    """Load sample labels from TSV/CSV or the paper's PDAC HDF object."""

    if path.suffix.lower() in {".h5", ".hdf", ".hdf5", ".h5ad"}:
        frame = pd.read_hdf(path)
        frame = frame.copy()
        frame["sample_id"] = frame.index.astype(str)
        if "label" not in frame and "accLabel" in frame:
            frame["label"] = frame["accLabel"].astype(str)
    else:
        sep = "\t" if ".tsv" in path.name else ","
        frame = pd.read_csv(path, sep=sep)
    required = {"sample_id", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"metadata lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["label"] = frame["label"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError("metadata sample IDs are not unique")
    if frame["label"].nunique() < 2:
        raise ValueError("at least two phenotype labels are required")
    return frame.set_index("sample_id", drop=False)


def align_sample_universe(
    tables: Mapping[str, pd.DataFrame], metadata: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Require exact graph/metadata sample equality and canonical ordering."""

    graph_ids = set(tables)
    metadata_ids = set(metadata.index.astype(str))
    if graph_ids != metadata_ids:
        raise ValueError(
            "graph/metadata sample mismatch; "
            f"graphs_only={sorted(graph_ids - metadata_ids)}, "
            f"metadata_only={sorted(metadata_ids - graph_ids)}"
        )
    ids = sorted(graph_ids)
    return ({sample_id: tables[sample_id] for sample_id in ids}, metadata.loc[ids])


def _finite_distance(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(np.real_if_close(matrix), dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError("distance matrix must be square")
    result = (result + result.T) / 2.0
    finite = result[np.isfinite(result)]
    replacement = float(np.max(finite)) if finite.size else 1.0
    if replacement <= 0:
        replacement = 1.0
    result[~np.isfinite(result)] = replacement
    result[result < 0] = 0.0
    np.fill_diagonal(result, 0.0)
    return result


def _tmm(frame: pd.DataFrame) -> pd.DataFrame:
    import conorm

    return conorm.tmm(frame + 1.0e-6, trim_lfc=0, trim_mag=0)


def build_accordion(tables: Mapping[str, pd.DataFrame]) -> Any:
    """Build the paper-native Accordion object with its documented settings."""

    from scaccordion import tl as actl

    return actl.Accordion(
        tbls=dict(tables), weight="lr_means", normf=_tmm, filter=0.2
    )


def accordion_distances(accordion: Any) -> dict[str, np.ndarray]:
    """Compute directed DW-OT and correlation-cost OT distances."""

    result: dict[str, np.ndarray] = {}
    accordion.compute_cost(mode="HTD", beta=0.5)
    accordion.compute_wassestein(cost="HTD_0.5")
    result["scaccordion_dw_ot"] = _finite_distance(
        accordion.wdist["HTD_0.5"].loc[
            accordion.p.columns, accordion.p.columns
        ].to_numpy()
    )
    accordion.compute_cost(mode="distance", metric="correlation")
    accordion.compute_wassestein(cost="correlation")
    result["corr_ot"] = _finite_distance(
        accordion.wdist["correlation"].loc[
            accordion.p.columns, accordion.p.columns
        ].to_numpy()
    )
    return result


def got_distance(accordion: Any) -> np.ndarray:
    """Compute the paper's undirected GOT baseline on a common node universe."""

    import networkx as nx
    from scaccordion import tl as actl

    cell_types = sorted(accordion.nodes)
    sample_ids = list(accordion.p.columns)
    laplacians: list[np.ndarray] = []
    for sample_id in sample_ids:
        directed = np.zeros((len(cell_types), len(cell_types)), dtype=float)
        index = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
        for event, value in accordion.p[sample_id].items():
            source, target = str(event).split("$", maxsplit=1)
            if source in index and target in index:
                directed[index[source], index[target]] = float(value)
        undirected = (directed + directed.T) / 2.0
        graph = nx.from_numpy_array(undirected)
        laplacians.append(nx.laplacian_matrix(graph).toarray().astype(float))
    distance = np.zeros((len(sample_ids), len(sample_ids)), dtype=float)
    for left in range(len(sample_ids)):
        for right in range(left + 1, len(sample_ids)):
            value = actl.GOT.wass_dist_(laplacians[left], laplacians[right])
            value = float(np.real_if_close(value))
            distance[left, right] = distance[right, left] = max(value, 0.0)
    return _finite_distance(distance)


def tabular_pca_distance(accordion: Any) -> np.ndarray:
    """Return Euclidean sample distance after the paper's tabular PCA step."""

    sample_by_edge = accordion.p.T.to_numpy(dtype=float)
    components = min(sample_by_edge.shape)
    embedding = PCA(n_components=components, svd_solver="full").fit_transform(
        sample_by_edge
    )
    return _finite_distance(squareform(pdist(embedding, metric="euclidean")))


def _event_entity(table: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    if preferred in table:
        preferred_values = table[preferred].astype("string")
        fallback_values = table[fallback].astype("string")
        return preferred_values.fillna(fallback_values).astype(str)
    if fallback not in table:
        raise ValueError(f"canonical event table lacks {preferred}/{fallback}")
    return table[fallback].astype(str)


def canonical_event_matrix(
    tables: Mapping[str, pd.DataFrame],
    sample_ids: Sequence[str],
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Project graph rows to source-ligand-receptor-target event vectors."""

    records: list[pd.DataFrame] = []
    for sample_id in sample_ids:
        table = tables[sample_id]
        if "ligand" not in table or "receptor" not in table:
            raise ValueError(
                "canonical event projection requires ligand and receptor columns"
            )
        part = pd.DataFrame(
            {
                "sample_id": sample_id,
                "source": table["source"].astype(str),
                "ligand": _event_entity(table, "ligand_complex", "ligand"),
                "receptor": _event_entity(table, "receptor_complex", "receptor"),
                "target": table["target"].astype(str),
                "score": table["lr_means"].astype(float),
            }
        )
        records.append(part)
    long = pd.concat(records, ignore_index=True)
    long["event"] = (
        long["source"]
        + "$"
        + long["ligand"]
        + "$"
        + long["receptor"]
        + "$"
        + long["target"]
    )
    matrix = long.pivot_table(
        index="event",
        columns="sample_id",
        values="score",
        aggfunc="mean",
        fill_value=0.0,
    )
    matrix = matrix.reindex(columns=list(sample_ids), fill_value=0.0)
    return _tmm(matrix) if normalize else matrix


def canonical_event_distance(
    tables: Mapping[str, pd.DataFrame], sample_ids: Sequence[str]
) -> tuple[np.ndarray, int]:
    matrix = canonical_event_matrix(tables, sample_ids)
    distance = squareform(pdist(matrix.T.to_numpy(dtype=float), metric="correlation"))
    return _finite_distance(distance), int(matrix.shape[0])


def _exact_small_kmedoids(distance: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Dependency-free exact fallback used only for small unit-test matrices."""

    if len(distance) > 20 or math.comb(len(distance), k) > 100_000:
        raise ModuleNotFoundError(
            "kmedoids is required for formal cohort evaluation"
        )
    best_labels: np.ndarray | None = None
    best_inertia = math.inf
    for medoids in itertools.combinations(range(len(distance)), k):
        selected = distance[:, medoids]
        labels = np.argmin(selected, axis=1).astype(int)
        inertia = float(selected[np.arange(len(distance)), labels].sum())
        if inertia < best_inertia - 1.0e-12:
            best_labels, best_inertia = labels, inertia
    if best_labels is None:
        raise RuntimeError("exact k-medoids fallback produced no fit")
    return best_labels, best_inertia


def _best_kmedoids(
    distance: np.ndarray, k: int, starts: int, seed: int
) -> tuple[np.ndarray, float, int]:
    try:
        import kmedoids
    except ModuleNotFoundError:
        labels, inertia = _exact_small_kmedoids(distance, k)
        return labels, inertia, seed

    best: tuple[np.ndarray, float, int] | None = None
    for offset in range(starts):
        current_seed = seed + offset
        fitted = kmedoids.KMedoids(
            n_clusters=k,
            method="fasterpam",
            random_state=current_seed,
        ).fit(distance)
        labels = np.asarray(fitted.labels_, dtype=int)
        inertia = float(fitted.inertia_)
        candidate = (labels, inertia, current_seed)
        if best is None or inertia < best[1] - 1.0e-12:
            best = candidate
        elif math.isclose(inertia, best[1], rel_tol=0.0, abs_tol=1.0e-12):
            if tuple(labels.tolist()) < tuple(best[0].tolist()):
                best = candidate
    if best is None:
        raise RuntimeError("k-medoids produced no fit")
    return best


def evaluate_distance(
    *,
    benchmark_id: str,
    method: str,
    distance: np.ndarray,
    labels: pd.Series,
    cluster_counts: Sequence[int],
    starts: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate a fixed distance matrix without using labels for fitting."""

    encoded = labels.astype("category").cat.codes.to_numpy()
    true_k = int(labels.nunique())
    true_label_silhouette = (
        float(silhouette_score(distance, encoded, metric="precomputed"))
        if true_k > 1 and len(labels) > true_k
        else None
    )
    rows: list[dict[str, Any]] = []
    for k in cluster_counts:
        if k < 2 or k >= len(labels):
            continue
        fitted, inertia, fit_seed = _best_kmedoids(distance, k, starts, seed)
        predicted_silhouette = (
            float(silhouette_score(distance, fitted, metric="precomputed"))
            if len(set(fitted)) > 1
            else None
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark_id": benchmark_id,
                "method": method,
                "clustering_backend": "kmedoids_fasterpam_multistart",
                "k": int(k),
                "true_k": true_k,
                "is_true_k": bool(k == true_k),
                "ari": float(adjusted_rand_score(encoded, fitted)),
                "rand_index": float(rand_score(encoded, fitted)),
                "predicted_cluster_silhouette": predicted_silhouette,
                "true_label_distance_silhouette": true_label_silhouette,
                "inertia": inertia,
                "selected_start_seed": fit_seed,
                "starts": starts,
            }
        )
    return rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (benchmark_id, method), group in metrics.groupby(
        ["benchmark_id", "method"], observed=True
    ):
        at_true = group.loc[group["is_true_k"]]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark_id": benchmark_id,
                "method": method,
                "ari_at_true_k": (
                    float(at_true.iloc[0]["ari"]) if len(at_true) == 1 else None
                ),
                "rand_at_true_k": (
                    float(at_true.iloc[0]["rand_index"])
                    if len(at_true) == 1
                    else None
                ),
                "maximum_ari_over_k": float(group["ari"].max()),
                "maximum_rand_over_k": float(group["rand_index"].max()),
                "distance_label_silhouette": float(
                    group.iloc[0]["true_label_distance_silhouette"]
                ),
                "best_ari_k": int(group.loc[group["ari"].idxmax(), "k"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def _source_fingerprint(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], str]:
    records = [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]
    import hashlib

    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return records, digest


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    benchmark = _contract_record(args.benchmark_id)
    graph_paths = sorted(
        path
        for path in args.graph_dir.iterdir()
        if path.is_file()
        and any(
            path.name.endswith(suffix)
            for suffix in (".csv", ".csv.gz", ".tsv", ".tsv.gz")
        )
    )
    source_records, source_digest = _source_fingerprint(
        [*graph_paths, args.metadata, args.contract]
    )
    tables, metadata = align_sample_universe(
        load_graph_tables(args.graph_dir), load_metadata(args.metadata)
    )
    expected_samples = int(benchmark["design"]["samples"])
    expected_labels = int(benchmark["design"]["labels"])
    if (
        len(metadata) != expected_samples
        or metadata["label"].nunique() != expected_labels
    ):
        raise ValueError(
            "cohort dimensions disagree with frozen contract: "
            f"observed {len(metadata)} samples/{metadata['label'].nunique()} labels, "
            f"expected {expected_samples}/{expected_labels}"
        )

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "distances").mkdir()
    metadata.reset_index(drop=True).to_csv(
        args.output / "sample_metadata.tsv", sep="\t", index=False
    )
    sample_ids = list(tables)
    stage_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    distances: dict[str, np.ndarray] = {}

    def stage(name: str, function: Any) -> Any:
        before = time.perf_counter()
        value = function()
        stage_rows.append(
            {
                "stage": name,
                "wall_seconds": time.perf_counter() - before,
                "peak_process_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            }
        )
        return value

    accordion = stage("build_accordion", lambda: build_accordion(tables))
    if list(accordion.p.columns) != sample_ids:
        raise ValueError("Accordion changed the canonical sample order")
    for method, function in (
        ("accordion_ot", lambda: accordion_distances(accordion)),
        ("got", lambda: {"got": got_distance(accordion)}),
        ("tabular_pca", lambda: {"tabular_pca": tabular_pca_distance(accordion)}),
        (
            "crychic_canonical_event_correlation",
            lambda: {
                "crychic_canonical_event_correlation": canonical_event_distance(
                    tables, sample_ids
                )[0]
            },
        ),
    ):
        try:
            distances.update(stage(f"distance_{method}", function))
        except Exception as exc:  # NE is persisted; it is never replaced by zero.
            represented = (
                ("scaccordion_dw_ot", "corr_ot")
                if method == "accordion_ot"
                else (method,)
            )
            for represented_method in represented:
                status_rows.append(
                    {
                        "benchmark_id": args.benchmark_id,
                        "method": represented_method,
                        "status": "NE",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

    metric_rows: list[dict[str, Any]] = []
    for method in METHODS:
        if method not in distances:
            if not any(row["method"] == method for row in status_rows):
                status_rows.append(
                    {
                        "benchmark_id": args.benchmark_id,
                        "method": method,
                        "status": "NE",
                        "reason": "distance_not_produced",
                    }
                )
            continue
        distance = distances[method]
        if distance.shape != (len(sample_ids), len(sample_ids)):
            raise ValueError(f"{method} distance shape mismatch")
        np.savez_compressed(
            args.output / "distances" / f"{method}.npz",
            distance=distance,
            sample_ids=np.asarray(sample_ids, dtype=str),
        )
        metric_rows.extend(
            stage(
                f"cluster_{method}",
                lambda method=method, distance=distance: evaluate_distance(
                    benchmark_id=args.benchmark_id,
                    method=method,
                    distance=distance,
                    labels=metadata.loc[sample_ids, "label"],
                    cluster_counts=benchmark["design"]["cluster_counts"],
                    starts=args.kmedoids_starts,
                    seed=args.seed,
                ),
            )
        )
        status_rows.append(
            {
                "benchmark_id": args.benchmark_id,
                "method": method,
                "status": "complete",
                "reason": "",
            }
        )

    metrics = pd.DataFrame.from_records(metric_rows)
    if metrics.empty:
        raise RuntimeError("no scACCorDiON method was estimable")
    summary = summarize_metrics(metrics)
    metrics.to_csv(args.output / "method_metrics_by_k.tsv", sep="\t", index=False)
    summary.to_csv(args.output / "summary_metrics.tsv", sep="\t", index=False)
    pd.DataFrame.from_records(status_rows).to_csv(
        args.output / "method_status.tsv", sep="\t", index=False
    )
    pd.DataFrame.from_records(stage_rows).to_csv(
        args.output / "stage_timings.tsv", sep="\t", index=False
    )
    paper_rows = [
        {
            "benchmark_id": args.benchmark_id,
            "source": "Nagai_et_al_2025_Supplementary_Table_1",
            "metric": "ARI_at_true_k",
            "paper_method": method,
            "value": value,
        }
        for method, value in PAPER_TRUE_K_ARI[args.benchmark_id].items()
    ]
    pd.DataFrame.from_records(paper_rows).to_csv(
        args.output / "paper_reference_metrics.tsv", sep="\t", index=False
    )

    artifact_names = [
        "sample_metadata.tsv",
        "method_metrics_by_k.tsv",
        "summary_metrics.tsv",
        "method_status.tsv",
        "stage_timings.tsv",
        "paper_reference_metrics.tsv",
    ]
    distance_names = [f"distances/{method}.npz" for method in distances]
    repo = git_metadata(args.repo_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark_id": args.benchmark_id,
        "protocol_status": args.protocol_status,
        "canonical_projection_interpretation": (
            "CRYCHIC-compatible source-ligand-receptor-target vector distance; "
            "not native CRYCHIC differential inference"
        ),
        "cohort": {
            "samples": len(metadata),
            "labels": metadata["label"].value_counts().sort_index().to_dict(),
            "declared_single_cells": int(benchmark["observed_single_cells"]),
            "graph_tables": len(tables),
            "paper_cell_types": int(benchmark["design"]["cell_types"]),
        },
        "execution": {
            "wall_seconds": time.perf_counter() - started,
            "kmedoids_starts": args.kmedoids_starts,
            "seed": args.seed,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "source_repository": repo,
        "inputs": {"records": source_records, "aggregate_sha256": source_digest},
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256_file(args.contract),
        },
        "artifacts": {
            name: {"sha256": sha256_file(args.output / name)}
            for name in [*artifact_names, *distance_names]
        },
    }
    _write_json(args.output / "manifest.json", manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-id",
        required=True,
        choices=tuple(PAPER_TRUE_K_ARI),
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "suggest_v5_under100k_v1.json",
    )
    parser.add_argument("--protocol-status", required=True)
    parser.add_argument("--kmedoids-starts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=859)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.kmedoids_starts < 1:
        raise ValueError("--kmedoids-starts must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
