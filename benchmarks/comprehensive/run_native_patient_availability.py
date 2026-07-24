"""Build native CRYCHIC patient availability-hyperedge distances from H5ADs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from crychic.availability import AvailabilityParameters, estimate_bundle_availability
from crychic.data import InputSchema, validate_anndata
from crychic.pseudobulk import aggregate_pseudobulk

SCHEMA = "crychic-native-patient-availability-hyperedge-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _score_sample(payload: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    input_path = Path(str(payload["input_path"]))
    output_path = Path(str(payload["output_path"]))
    data = ad.read_h5ad(input_path)
    sample_id = str(payload["sample_id"])
    annotation = str(payload["annotation_column"])
    if annotation not in data.obs:
        raise ValueError(f"{input_path} lacks {annotation}")
    if data.obs["sample_id"].astype(str).nunique() != 1:
        raise ValueError(f"{input_path} is not one biological sample")
    data.obs["subject_id"] = sample_id
    data.obs["context"] = data.obs["label"].astype(str).astype(object)
    data.obs["cell_type"] = data.obs[annotation].astype(str).astype(object)
    data.obs["sample_id"] = data.obs["sample_id"].astype(str).astype(object)
    mode = str(payload["input_mode"])
    if mode == "counts_X":
        data.layers["benchmark_counts"] = data.X
        schema = InputSchema(
            context_keys=("context",),
            counts_layer="benchmark_counts",
            sample_key="sample_id",
            subject_key="subject_id",
            cell_type_key="cell_type",
            species="human",
            gene_namespace="HGNC symbol",
        )
    elif mode == "log1p_X":
        schema = InputSchema(
            context_keys=("context",),
            counts_layer=None,
            sample_key="sample_id",
            subject_key="subject_id",
            cell_type_key="cell_type",
            expression_source="X",
            expression_transform="log1p_normalized",
            normalized_zero_is_nondetection=False,
            species="human",
            gene_namespace="HGNC symbol",
        )
    else:
        raise ValueError(f"unsupported input mode: {mode}")
    bundle = harmonized_resource_bundle(
        Path(str(payload["resource_table"])), Path(str(payload["resource_manifest"]))
    )
    validated = validate_anndata(data, schema)
    aggregate = aggregate_pseudobulk(validated, min_cells=int(payload["min_cells"]))
    availability = estimate_bundle_availability(
        aggregate,
        bundle,
        context_keys=("context",),
        parameters=AvailabilityParameters(),
        min_pooled_availability=0.0,
        max_interactions=None,
    )
    scores = availability.sample_interactions.loc[
        :,
        [
            "sample_id",
            "subject_id",
            "sender",
            "receiver",
            "interaction_id",
            "ligand",
            "receptor",
            "availability_state",
            "state_status",
            "state_reason_code",
        ],
    ].copy()
    scores.to_parquet(output_path, index=False)
    return {
        "sample_id": sample_id,
        "status": "complete",
        "reason_code": "",
        "cells": int(data.n_obs),
        "cell_types": int(data.obs["cell_type"].nunique()),
        "rows": len(scores),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "input_sha256": str(payload["input_sha256"]),
        "output_sha256": sha256_file(output_path),
    }


def _patient_distance(
    score_paths: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    frames: list[pd.DataFrame] = []
    for path in score_paths:
        frame = pd.read_parquet(path)
        frame["event"] = (
            frame["sender"].astype(str)
            + "$"
            + frame["ligand"].astype(str)
            + "$"
            + frame["receptor"].astype(str)
            + "$"
            + frame["receiver"].astype(str)
        )
        frames.append(frame.loc[:, ["sample_id", "event", "availability_state"]])
    long = pd.concat(frames, ignore_index=True)
    matrix = long.pivot_table(
        index="event",
        columns="sample_id",
        values="availability_state",
        aggfunc="mean",
    )
    matrix = matrix.loc[matrix.notna().sum(axis=1).ge(2)]
    sample_ids = sorted(matrix.columns.astype(str))
    matrix = matrix.reindex(columns=sample_ids)
    n = len(sample_ids)
    distance = np.zeros((n, n), dtype=float)
    overlap = np.zeros((n, n), dtype=int)
    for left in range(n):
        overlap[left, left] = int(matrix.iloc[:, left].notna().sum())
        for right in range(left + 1, n):
            observed = matrix.iloc[:, [left, right]].dropna()
            overlap[left, right] = overlap[right, left] = len(observed)
            if len(observed) < 3:
                raise ValueError(
                    f"patient pair {sample_ids[left]}/{sample_ids[right]} "
                    "has <3 shared hyperedges"
                )
            rho = float(spearmanr(observed.iloc[:, 0], observed.iloc[:, 1]).statistic)
            if not np.isfinite(rho):
                raise ValueError(
                    f"patient pair {sample_ids[left]}/{sample_ids[right]} "
                    "has undefined Spearman"
                )
            distance[left, right] = distance[right, left] = max(0.0, 1.0 - rho)
    return distance, overlap, sample_ids, int(len(matrix))


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    score_dir = args.output / "scores"
    score_dir.mkdir()
    sample_paths = sorted(args.sample_dir.glob("*.h5ad"))
    metadata = pd.read_csv(args.metadata, sep="\t")
    sample_ids = [path.stem for path in sample_paths]
    if set(sample_ids) != set(metadata["sample_id"].astype(str)):
        raise ValueError("sample files and metadata differ")
    payloads = []
    for path in sample_paths:
        payloads.append(
            {
                "sample_id": path.stem,
                "input_path": str(path.resolve()),
                "input_sha256": sha256_file(path),
                "output_path": str((score_dir / f"{path.stem}.parquet").resolve()),
                "annotation_column": args.annotation_column,
                "input_mode": args.input_mode,
                "min_cells": args.min_cells,
                "resource_table": str(args.resource_table.resolve()),
                "resource_manifest": str(args.resource_manifest.resolve()),
            }
        )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_score_sample, payload): payload for payload in payloads
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(payloads)}] {row['sample_id']} "
                f"{row['wall_seconds']:.1f}s",
                flush=True,
            )
    task_metrics = pd.DataFrame(rows).sort_values("sample_id")
    task_metrics.to_csv(args.output / "task_metrics.tsv", sep="\t", index=False)
    distance, overlap, distance_ids, event_count = _patient_distance(
        [score_dir / f"{sample_id}.parquet" for sample_id in sorted(sample_ids)]
    )
    np.savez_compressed(
        args.output / "native_availability_hyperedge_spearman.npz",
        distance=distance,
        overlap=overlap,
        sample_ids=np.asarray(distance_ids, dtype=str),
    )
    pd.DataFrame(overlap, index=distance_ids, columns=distance_ids).to_csv(
        args.output / "pairwise_shared_hyperedges.tsv", sep="\t"
    )
    metadata.sort_values("sample_id").to_csv(
        args.output / "sample_metadata.tsv", sep="\t", index=False
    )
    _write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "interpretation": (
                "native CRYCHIC static availability on sender-ligand-receptor-"
                "receiver hyperedges; not differential inference or downstream-target "
                "hypergraph"
            ),
            "missing_structure_policy": (
                "pairwise complete hyperedges; absent cell types are missing, not zero"
            ),
            "source_repository": git_metadata(args.repo_root),
            "parameters": {
                "annotation_column": args.annotation_column,
                "input_mode": args.input_mode,
                "min_cells": args.min_cells,
                "distance": "1 - pairwise-complete Spearman",
            },
            "cohort": {
                "samples": len(sample_ids),
                "retained_union_hyperedges": event_count,
            },
            "execution": {
                "workers": args.workers,
                "wall_seconds": time.perf_counter() - started,
                "peak_worker_rss_kib": int(task_metrics["peak_rss_kib"].max()),
                "python": platform.python_version(),
            },
            "resource": {
                "table_sha256": sha256_file(args.resource_table),
                "manifest_sha256": sha256_file(args.resource_manifest),
            },
            "artifacts": {
                name: {"sha256": sha256_file(args.output / name)}
                for name in (
                    "task_metrics.tsv",
                    "native_availability_hyperedge_spearman.npz",
                    "pairwise_shared_hyperedges.tsv",
                    "sample_metadata.tsv",
                )
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--annotation-column", required=True)
    parser.add_argument("--input-mode", choices=("counts_X", "log1p_X"), required=True)
    parser.add_argument("--resource-table", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
