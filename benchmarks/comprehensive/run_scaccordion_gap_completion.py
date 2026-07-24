"""Complete scACCorDiON distance, clustering, and cross-cohort rank endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import platform
import resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from sklearn.metrics import adjusted_rand_score, rand_score

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.run_scaccordion_under100k import (
    _finite_distance,
    accordion_distances,
    align_sample_universe,
    build_accordion,
    canonical_event_matrix,
    evaluate_distance,
    load_graph_tables,
    load_metadata,
)

SCHEMA = "crychic-scaccordion-gap-completion-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def canonical_spearman_distance(
    tables: Mapping[str, pd.DataFrame], sample_ids: Sequence[str]
) -> tuple[np.ndarray, int]:
    matrix = canonical_event_matrix(tables, sample_ids)
    ranked = np.apply_along_axis(rankdata, 1, matrix.T.to_numpy(dtype=float))
    return _finite_distance(squareform(pdist(ranked, metric="correlation"))), int(
        matrix.shape[0]
    )


def _kbarycenter_once(
    distance: np.ndarray,
    distributions: np.ndarray,
    cost: np.ndarray,
    *,
    k: int,
    seed: int,
    max_iterations: int,
    regularization: float,
) -> tuple[np.ndarray, float, int]:
    import ot

    rng = np.random.default_rng(seed)
    n_samples = distance.shape[0]
    initial = rng.choice(n_samples, size=k, replace=False)
    centroid_distance = distance[:, initial].copy()
    labels = np.argmin(centroid_distance, axis=1)
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        updated = np.empty((n_samples, k), dtype=float)
        for cluster in range(k):
            members = np.flatnonzero(labels == cluster)
            if not len(members):
                farthest = int(np.argmax(np.min(centroid_distance, axis=1)))
                members = np.asarray([farthest])
            selected = distributions[:, members]
            selected = selected / np.maximum(
                selected.sum(axis=0, keepdims=True), 1.0e-15
            )
            barycenter = ot.barycenter(
                A=selected, M=cost, reg=regularization, numItermax=10_000
            )
            barycenter = np.asarray(barycenter, dtype=float)
            barycenter = np.clip(barycenter, 0.0, None)
            barycenter /= max(float(barycenter.sum()), 1.0e-15)
            for sample in range(n_samples):
                sample_distribution = distributions[:, sample]
                sample_distribution = sample_distribution / max(
                    float(sample_distribution.sum()), 1.0e-15
                )
                updated[sample, cluster] = float(
                    ot.emd2(barycenter, sample_distribution, cost)
                )
        new_labels = np.argmin(updated, axis=1)
        centroid_distance = updated
        if np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels = new_labels
    inertia = float(centroid_distance[np.arange(n_samples), labels].sum())
    return labels, inertia, iterations


def best_kbarycenter(
    distance: np.ndarray,
    distributions: np.ndarray,
    cost: np.ndarray,
    *,
    k: int,
    starts: int,
    seed: int,
    max_iterations: int,
    regularization: float,
) -> tuple[np.ndarray, float, int, int]:
    best: tuple[np.ndarray, float, int, int] | None = None
    for offset in range(starts):
        current_seed = seed + offset
        labels, inertia, iterations = _kbarycenter_once(
            distance,
            distributions,
            cost,
            k=k,
            seed=current_seed,
            max_iterations=max_iterations,
            regularization=regularization,
        )
        candidate = labels, inertia, current_seed, iterations
        if best is None or inertia < best[1] - 1.0e-12:
            best = candidate
        elif best is not None and math.isclose(inertia, best[1], abs_tol=1.0e-12):
            if tuple(labels.tolist()) < tuple(best[0].tolist()):
                best = candidate
    if best is None:
        raise RuntimeError("k-barycenter produced no fit")
    return best


def _kbary_start_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    labels, inertia, iterations = _kbarycenter_once(
        np.asarray(payload["distance"]),
        np.asarray(payload["distributions"]),
        np.asarray(payload["cost"]),
        k=int(payload["k"]),
        seed=int(payload["seed"]),
        max_iterations=int(payload["max_iterations"]),
        regularization=float(payload["regularization"]),
    )
    return {
        "method": str(payload["method"]),
        "k": int(payload["k"]),
        "labels": labels,
        "inertia": inertia,
        "seed": int(payload["seed"]),
        "iterations": iterations,
    }


def _knn_graph(distance: np.ndarray, neighbors: int) -> tuple[Any, list[float]]:
    import igraph as ig

    n = len(distance)
    k = min(neighbors, n - 1)
    edges: dict[tuple[int, int], float] = {}
    positive = distance[np.isfinite(distance) & (distance > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    for left in range(n):
        order = np.argsort(distance[left], kind="mergesort")
        for right in order[order != left][:k]:
            edge = tuple(sorted((left, int(right))))
            weight = math.exp(-float(distance[left, right]) / max(scale, 1.0e-12))
            edges[edge] = max(edges.get(edge, 0.0), weight)
    graph = ig.Graph(n=n, edges=list(edges), directed=False)
    weights = [edges[edge] for edge in edges]
    return graph, weights


def leiden_grid(
    distance: np.ndarray,
    labels: pd.Series,
    *,
    resolutions: Sequence[float],
    neighbors: int,
    seed: int,
    method: str,
    benchmark_id: str,
) -> list[dict[str, Any]]:
    import leidenalg

    truth = labels.astype("category").cat.codes.to_numpy()
    graph, weights = _knn_graph(distance, neighbors)
    rows: list[dict[str, Any]] = []
    for resolution in resolutions:
        if resolution == 0.0:
            predicted = np.zeros(len(labels), dtype=int)
            quality = 0.0
        else:
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                resolution_parameter=float(resolution),
                seed=seed,
            )
            predicted = np.asarray(partition.membership, dtype=int)
            quality = float(partition.quality())
        rows.append(
            {
                "schema_version": SCHEMA,
                "benchmark_id": benchmark_id,
                "method": method,
                "clustering_backend": "leiden_knn_distance",
                "resolution": float(resolution),
                "k": int(len(np.unique(predicted))),
                "true_k": int(labels.nunique()),
                "ari": float(adjusted_rand_score(truth, predicted)),
                "rand_index": float(rand_score(truth, predicted)),
                "quality": quality,
                "neighbors": min(neighbors, len(labels) - 1),
                "seed": seed,
            }
        )
    return rows


def run_cohort(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / "distances").mkdir()
    config = json.loads(args.contract.read_text(encoding="utf-8"))["scaccordion"]
    started = time.perf_counter()
    tables, metadata = align_sample_universe(
        load_graph_tables(args.graph_dir), load_metadata(args.metadata)
    )
    sample_ids = list(tables)
    labels = metadata.loc[sample_ids, "label"]
    spearman, event_count = canonical_spearman_distance(tables, sample_ids)
    np.savez_compressed(
        args.output / "distances" / "crychic_canonical_event_spearman.npz",
        distance=spearman,
        sample_ids=np.asarray(sample_ids, dtype=str),
    )
    base_distances: dict[str, np.ndarray] = {}
    for path in sorted((args.base_result / "distances").glob("*.npz")):
        payload = np.load(path)
        if list(payload["sample_ids"].astype(str)) != sample_ids:
            raise ValueError(f"base sample order mismatch: {path}")
        base_distances[path.stem] = _finite_distance(payload["distance"])
    base_distances["crychic_canonical_event_spearman"] = spearman
    if args.native_distance is not None:
        payload = np.load(args.native_distance)
        native_ids = list(payload["sample_ids"].astype(str))
        if native_ids != sample_ids:
            raise ValueError("native CRYCHIC distance sample order mismatch")
        base_distances["crychic_native_availability_hyperedge_spearman"] = (
            _finite_distance(payload["distance"])
        )
        np.savez_compressed(
            args.output
            / "distances"
            / "crychic_native_availability_hyperedge_spearman.npz",
            distance=base_distances["crychic_native_availability_hyperedge_spearman"],
            sample_ids=np.asarray(sample_ids, dtype=str),
        )
    kmedoid_rows = evaluate_distance(
        benchmark_id=args.benchmark_id,
        method="crychic_canonical_event_spearman",
        distance=spearman,
        labels=labels,
        cluster_counts=range(2, 8),
        starts=args.kmedoids_starts,
        seed=int(config["seed"]),
    )
    if "crychic_native_availability_hyperedge_spearman" in base_distances:
        kmedoid_rows.extend(
            evaluate_distance(
                benchmark_id=args.benchmark_id,
                method="crychic_native_availability_hyperedge_spearman",
                distance=base_distances[
                    "crychic_native_availability_hyperedge_spearman"
                ],
                labels=labels,
                cluster_counts=range(2, 8),
                starts=args.kmedoids_starts,
                seed=int(config["seed"]),
            )
        )
    accordion = build_accordion(tables)
    accordion_values = accordion_distances(accordion)
    distributions = accordion.p.loc[:, sample_ids].to_numpy(dtype=float)
    kbary_rows: list[dict[str, Any]] = []
    kbary_jobs: list[dict[str, Any]] = []
    true_k = int(labels.nunique())
    for method, cost_name in (
        ("scaccordion_dw_ot", "HTD_0.5"),
        ("corr_ot", "correlation"),
    ):
        cost = np.asarray(accordion.Cs[cost_name], dtype=float)
        distance = accordion_values[method]
        for offset in range(int(config["kbarycenter_starts"])):
            kbary_jobs.append(
                {
                    "method": method,
                    "k": true_k,
                    "distance": distance,
                    "distributions": distributions,
                    "cost": cost,
                    "seed": int(config["seed"]) + offset,
                    "max_iterations": int(config["kbarycenter_max_iterations"]),
                    "regularization": float(config["kbarycenter_regularization"]),
                }
            )
    kbary_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(_kbary_start_job, kbary_jobs):
            kbary_results.append(result)
    truth = labels.astype("category").cat.codes.to_numpy()
    for method in ("scaccordion_dw_ot", "corr_ot"):
        candidates = [row for row in kbary_results if row["method"] == method]
        selected = min(
            candidates,
            key=lambda row: (row["inertia"], tuple(row["labels"].tolist())),
        )
        predicted = selected["labels"]
        kbary_rows.append(
            {
                "schema_version": SCHEMA,
                "benchmark_id": args.benchmark_id,
                "method": method,
                "clustering_backend": "k_barycenter_entropic_ot",
                "k": true_k,
                "true_k": true_k,
                "ari": float(adjusted_rand_score(truth, predicted)),
                "rand_index": float(rand_score(truth, predicted)),
                "inertia": float(selected["inertia"]),
                "selected_start_seed": int(selected["seed"]),
                "starts": int(config["kbarycenter_starts"]),
                "iterations": int(selected["iterations"]),
                "regularization": float(config["kbarycenter_regularization"]),
            }
        )
    resolution_config = config["leiden_resolutions"]
    resolutions = np.round(
        np.arange(
            float(resolution_config["start"]),
            float(resolution_config["stop"]) + float(resolution_config["step"]) / 2,
            float(resolution_config["step"]),
        ),
        8,
    )
    leiden_rows: list[dict[str, Any]] = []
    for method, distance in base_distances.items():
        leiden_rows.extend(
            leiden_grid(
                distance,
                labels,
                resolutions=resolutions,
                neighbors=int(config["leiden_knn"]),
                seed=int(config["seed"]),
                method=method,
                benchmark_id=args.benchmark_id,
            )
        )
    kmedoids = pd.DataFrame.from_records(kmedoid_rows)
    kbary = pd.DataFrame.from_records(kbary_rows)
    leiden = pd.DataFrame.from_records(leiden_rows)
    kmedoids.to_csv(args.output / "new_kmedoids_metrics.tsv", sep="\t", index=False)
    kbary.to_csv(args.output / "kbarycenter_metrics.tsv", sep="\t", index=False)
    leiden.to_csv(args.output / "leiden_metrics.tsv", sep="\t", index=False)
    summaries: list[dict[str, Any]] = []
    for frame, backend in (
        (kmedoids, "kmedoids"),
        (kbary, "k_barycenter"),
        (leiden, "leiden"),
    ):
        for method, group in frame.groupby("method", observed=True):
            true_k = int(group["true_k"].iloc[0])
            if backend == "leiden":
                candidates = group.loc[group["k"].eq(true_k)]
                selected = (
                    (
                        candidates
                        if not candidates.empty
                        else group.loc[
                            (group["k"] - true_k)
                            .abs()
                            .eq((group["k"] - true_k).abs().min())
                        ]
                    )
                    .sort_values(["quality", "resolution"], ascending=[False, True])
                    .iloc[0]
                )
            else:
                selected = group.loc[group["k"].eq(true_k)].iloc[0]
            summaries.append(
                {
                    "benchmark_id": args.benchmark_id,
                    "method": method,
                    "clustering_backend": backend,
                    "ari_at_protocol_selected_true_k": float(selected["ari"]),
                    "rand_at_protocol_selected_true_k": float(selected["rand_index"]),
                    "maximum_ari_over_grid": float(group["ari"].max()),
                    "selected_k": int(selected["k"]),
                    "selected_resolution": float(selected["resolution"])
                    if "resolution" in selected
                    else None,
                }
            )
    pd.DataFrame(summaries).to_csv(
        args.output / "summary_metrics.tsv", sep="\t", index=False
    )
    coverage = pd.DataFrame.from_records(
        [
            {"endpoint": "canonical_spearman", "status": "complete", "reason_code": ""},
            {"endpoint": "k_barycenter", "status": "complete", "reason_code": ""},
            {
                "endpoint": "leiden_resolution_0_to_1",
                "status": "complete",
                "reason_code": "",
            },
            {
                "endpoint": "native_CRYCHIC_hyperedge_distance",
                "status": "complete" if args.native_distance else "NE",
                "reason_code": ""
                if args.native_distance
                else "native_distance_not_supplied",
            },
        ]
    )
    coverage.to_csv(args.output / "endpoint_coverage.tsv", sep="\t", index=False)
    artifacts = [
        "new_kmedoids_metrics.tsv",
        "kbarycenter_metrics.tsv",
        "leiden_metrics.tsv",
        "summary_metrics.tsv",
        "endpoint_coverage.tsv",
        "distances/crychic_canonical_event_spearman.npz",
    ]
    if args.native_distance:
        artifacts.append("distances/crychic_native_availability_hyperedge_spearman.npz")
    _write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "complete_with_declared_NE",
            "benchmark_id": args.benchmark_id,
            "canonical_event_count": event_count,
            "source_repository": git_metadata(args.repo_root),
            "contract": {
                "path": str(args.contract.resolve()),
                "sha256": sha256_file(args.contract),
            },
            "inputs": {
                "base_result_manifest_sha256": sha256_file(
                    args.base_result / "manifest.json"
                ),
                "graph_files": len(tables),
                "native_distance_sha256": sha256_file(args.native_distance)
                if args.native_distance
                else None,
            },
            "execution": {
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "python": platform.python_version(),
            },
            "artifacts": {
                name: {"sha256": sha256_file(args.output / name)} for name in artifacts
            },
        },
    )


def summarize(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    records: list[pd.DataFrame] = []
    for item in args.cohort:
        benchmark_id, base_text, gap_text = item.split("=", maxsplit=2)
        base = pd.read_csv(Path(base_text) / "method_metrics_by_k.tsv", sep="\t")
        base = base.loc[
            base["is_true_k"].astype(bool), ["benchmark_id", "method", "ari"]
        ]
        gap = pd.read_csv(Path(gap_text) / "new_kmedoids_metrics.tsv", sep="\t")
        gap = gap.loc[gap["is_true_k"].astype(bool), ["benchmark_id", "method", "ari"]]
        records.append(
            pd.concat([base, gap], ignore_index=True).assign(benchmark_id=benchmark_id)
        )
    table = pd.concat(records, ignore_index=True)
    pivot = table.pivot(index="benchmark_id", columns="method", values="ari")
    complete_methods = list(pivot.columns[pivot.notna().all(axis=0)])
    pivot = pivot.loc[:, complete_methods]
    rank_matrix = pivot.rank(axis=1, method="average", ascending=False)
    average = rank_matrix.mean(axis=0).sort_values()
    statistic, p_value = friedmanchisquare(
        *[pivot[column].to_numpy() for column in complete_methods]
    )
    rank_rows = pd.DataFrame(
        {"method": average.index, "average_rank": average.values}
    ).sort_values("average_rank")
    rank_rows["rank"] = np.arange(1, len(rank_rows) + 1)
    rank_rows["cohorts"] = len(pivot)
    rank_rows["friedman_statistic"] = float(statistic)
    rank_rows["friedman_p_value"] = float(p_value)
    comparisons: list[dict[str, Any]] = []
    k, n = len(complete_methods), len(pivot)
    standard_error = math.sqrt(k * (k + 1) / (6.0 * n))
    for left_index, left in enumerate(complete_methods):
        for right in complete_methods[left_index + 1 :]:
            difference = abs(float(average[left] - average[right]))
            q_value = difference / standard_error
            comparisons.append(
                {
                    "method_a": left,
                    "method_b": right,
                    "average_rank_difference": difference,
                    "nemenyi_q": q_value,
                    "nemenyi_p_value": float(
                        studentized_range.sf(q_value * math.sqrt(2.0), k, np.inf)
                    ),
                    "cohorts": n,
                }
            )
    pivot.to_csv(args.output / "ari_at_true_k.tsv", sep="\t")
    rank_matrix.to_csv(args.output / "within_cohort_ranks.tsv", sep="\t")
    rank_rows.to_csv(args.output / "friedman_average_ranks.tsv", sep="\t", index=False)
    pd.DataFrame(comparisons).to_csv(
        args.output / "nemenyi_pairwise.tsv", sep="\t", index=False
    )
    _write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "complete_descriptive_low_n",
            "warning": (
                "Friedman-Nemenyi uses only three eligible primary cohorts and "
                "is underpowered"
            ),
            "cohorts": list(pivot.index),
            "methods": complete_methods,
            "source_repository": git_metadata(args.repo_root),
            "artifacts": {
                name: {"sha256": sha256_file(args.output / name)}
                for name in (
                    "ari_at_true_k.tsv",
                    "within_cohort_ranks.tsv",
                    "friedman_average_ranks.tsv",
                    "nemenyi_pairwise.tsv",
                )
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cohort = sub.add_parser("cohort")
    cohort.add_argument("--benchmark-id", required=True)
    cohort.add_argument("--graph-dir", type=Path, required=True)
    cohort.add_argument("--metadata", type=Path, required=True)
    cohort.add_argument("--base-result", type=Path, required=True)
    cohort.add_argument("--native-distance", type=Path)
    cohort.add_argument("--output", type=Path, required=True)
    cohort.add_argument("--kmedoids-starts", type=int, default=100)
    cohort.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 20))
    cohort.add_argument("--repo-root", type=Path, default=Path.cwd())
    cohort.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "suggest_v5_gap_completion_v1.json",
    )
    summary = sub.add_parser("summarize")
    summary.add_argument(
        "--cohort",
        action="append",
        required=True,
        help="benchmark_id=base_result=gap_result",
    )
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cohort":
        run_cohort(args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
