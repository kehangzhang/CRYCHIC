"""Generate patient LR graphs with LIANA CellPhoneDB for scACCorDiON."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-liana-cellphonedb-cohort-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sample_id(path: Path) -> str:
    if not path.name.endswith(".h5ad"):
        raise ValueError(f"sample input is not H5AD: {path}")
    return path.name[: -len(".h5ad")]


def _stable_seed(first_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{first_seed}|{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _memory_state() -> tuple[int, int]:
    fields: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", maxsplit=1)
        fields[key] = int(value.strip().split()[0]) * 1024
    return fields["MemTotal"], fields["MemAvailable"]


def bounded_worker_count(
    requested: int,
    *,
    memory_limit_fraction: float,
    memory_per_worker_gib: float,
) -> int:
    """Bound workers by current memory and the hard total-memory ceiling."""

    if requested < 1:
        raise ValueError("requested worker count must be positive")
    if not 0 < memory_limit_fraction <= 0.8:
        raise ValueError("memory limit fraction must be in (0, 0.8]")
    total, available = _memory_state()
    used = total - available
    budget = max(0.0, total * memory_limit_fraction - used)
    per_worker = memory_per_worker_gib * 1024**3
    memory_workers = max(1, int(budget // per_worker))
    return min(requested, memory_workers)


def standardize_liana_result(
    result: pd.DataFrame, *, p_value_threshold: float
) -> pd.DataFrame:
    """Filter the paper's significant CellPhoneDB rows and retain provenance."""

    required = {
        "source",
        "target",
        "ligand",
        "receptor",
        "lr_means",
        "cellphone_pvals",
    }
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"LIANA result lacks columns: {sorted(missing)}")
    frame = result.copy()
    frame["lr_means"] = pd.to_numeric(frame["lr_means"], errors="coerce")
    frame["cellphone_pvals"] = pd.to_numeric(
        frame["cellphone_pvals"], errors="coerce"
    )
    frame = frame.loc[
        frame["lr_means"].notna()
        & frame["cellphone_pvals"].notna()
        & frame["cellphone_pvals"].le(p_value_threshold)
    ].copy()
    frame = frame.sort_values(
        ["source", "target", "ligand", "receptor"], kind="mergesort"
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no CellPhoneDB row passed the registered P-value threshold")
    return frame


def _run_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run one independent patient graph and return timing/provenance."""

    import anndata as ad
    import liana as li

    started = time.perf_counter()
    input_path = Path(str(payload["input_path"]))
    output_path = Path(str(payload["output_path"]))
    sample_id = str(payload["sample_id"])
    adata = ad.read_h5ad(input_path)
    annotation_column = str(payload["annotation_column"])
    if annotation_column not in adata.obs:
        raise ValueError(f"{input_path} lacks {annotation_column}")
    adata.obs["cell_type"] = adata.obs[annotation_column].astype("category")
    if adata.obs["sample_id"].astype(str).nunique() != 1:
        raise ValueError(f"{input_path} is not one biological sample")
    before_types = int(adata.obs["cell_type"].nunique())
    result = li.mt.cellphonedb(
        adata,
        groupby="cell_type",
        resource_name=str(payload["resource_name"]),
        expr_prop=float(payload["expr_prop"]),
        min_cells=int(payload["min_cells"]),
        return_all_lrs=False,
        use_raw=False,
        n_perms=int(payload["n_perms"]),
        seed=int(payload["seed"]),
        n_jobs=1,
        inplace=False,
        verbose=False,
    )
    graph = standardize_liana_result(
        result, p_value_threshold=float(payload["p_value_threshold"])
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    graph.to_csv(temporary, index=False)
    temporary.replace(output_path)
    return {
        "sample_id": sample_id,
        "status": "complete",
        "reason": "",
        "cells": int(adata.n_obs),
        "input_cell_types": before_types,
        "output_cell_types": int(
            len(set(graph["source"].astype(str)) | set(graph["target"].astype(str)))
        ),
        "significant_lr_rows": int(len(graph)),
        "directed_cell_pairs": int(
            graph.loc[:, ["source", "target"]].drop_duplicates().shape[0]
        ),
        "wall_seconds": time.perf_counter() - started,
        "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "seed": int(payload["seed"]),
        "input_sha256": str(payload["input_sha256"]),
        "output_sha256": sha256_file(output_path),
    }


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    sample_paths = sorted(args.sample_dir.glob("*.h5ad"))
    if not sample_paths:
        raise ValueError(f"no sample H5AD files in {args.sample_dir}")
    metadata = pd.read_csv(args.metadata, sep="\t")
    if {"sample_id", "label"}.difference(metadata.columns):
        raise ValueError("sample metadata must contain sample_id and label")
    sample_ids = [_sample_id(path) for path in sample_paths]
    if set(sample_ids) != set(metadata["sample_id"].astype(str)):
        raise ValueError("sample H5AD files and metadata do not have equal IDs")

    args.output.mkdir(parents=True)
    graph_dir = args.output / "graphs"
    graph_dir.mkdir()
    effective_workers = bounded_worker_count(
        args.workers,
        memory_limit_fraction=args.memory_limit_fraction,
        memory_per_worker_gib=args.memory_per_worker_gib,
    )
    input_sha = {path: sha256_file(path) for path in sample_paths}
    payloads = [
        {
            "sample_id": _sample_id(path),
            "input_path": str(path.resolve()),
            "input_sha256": input_sha[path],
            "output_path": str((graph_dir / f"{_sample_id(path)}.csv").resolve()),
            "annotation_column": args.annotation_column,
            "resource_name": args.resource_name,
            "expr_prop": args.expr_prop,
            "min_cells": args.min_cells,
            "n_perms": args.n_perms,
            "p_value_threshold": args.p_value_threshold,
            "seed": _stable_seed(args.first_seed, _sample_id(path)),
        }
        for path in sample_paths
    ]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=effective_workers
    ) as executor:
        future_map = {
            executor.submit(_run_worker, payload): payload for payload in payloads
        }
        for future in concurrent.futures.as_completed(future_map):
            payload = future_map[future]
            try:
                row = future.result()
                rows.append(row)
                print(
                    f"[{len(rows) + len(failures)}/{len(payloads)}] "
                    f"{row['sample_id']} complete {row['wall_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "sample_id": payload["sample_id"],
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                print(
                    f"[{len(rows) + len(failures)}/{len(payloads)}] "
                    f"{payload['sample_id']} FAILED: {failure['reason']}",
                    flush=True,
                )
    timing_columns = [
        "sample_id",
        "status",
        "reason",
        "cells",
        "input_cell_types",
        "output_cell_types",
        "significant_lr_rows",
        "directed_cell_pairs",
        "wall_seconds",
        "peak_worker_rss_kib",
        "seed",
        "input_sha256",
        "output_sha256",
    ]
    pd.DataFrame.from_records(rows, columns=timing_columns).sort_values(
        "sample_id"
    ).to_csv(args.output / "task_metrics.tsv", sep="\t", index=False)
    pd.DataFrame.from_records(
        failures, columns=["sample_id", "status", "reason"]
    ).to_csv(args.output / "failures.tsv", sep="\t", index=False)
    metadata.sort_values("sample_id").to_csv(
        args.output / "sample_metadata.tsv", sep="\t", index=False
    )

    artifact_paths = [
        args.output / "task_metrics.tsv",
        args.output / "failures.tsv",
        args.output / "sample_metadata.tsv",
        *sorted(graph_dir.glob("*.csv")),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not failures else "failed",
        "protocol_status": args.protocol_status,
        "method": "LIANA_1.5.0_CellPhoneDB",
        "parameters": {
            "resource_name": args.resource_name,
            "expr_prop": args.expr_prop,
            "min_cells": args.min_cells,
            "n_perms": args.n_perms,
            "p_value_threshold": args.p_value_threshold,
            "annotation_column": args.annotation_column,
            "first_seed": args.first_seed,
        },
        "execution": {
            "requested_workers": args.workers,
            "effective_workers": effective_workers,
            "threads_per_worker": 1,
            "memory_limit_fraction": args.memory_limit_fraction,
            "memory_per_worker_gib": args.memory_per_worker_gib,
            "tasks": len(payloads),
            "successful_tasks": len(rows),
            "failed_tasks": len(failures),
            "campaign_wall_seconds": time.perf_counter() - started,
            "peak_worker_rss_kib": max(
                (int(row["peak_worker_rss_kib"]) for row in rows), default=None
            ),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "preparation_manifest": {
            "path": str(args.preparation_manifest.resolve()),
            "sha256": sha256_file(args.preparation_manifest),
        },
        "source_repository": git_metadata(args.repo_root),
        "artifacts": {
            str(path.relative_to(args.output)): {"sha256": sha256_file(path)}
            for path in artifact_paths
        },
    }
    _write_json(args.output / "manifest.json", manifest)
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--annotation-column", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol-status", required=True)
    parser.add_argument("--resource-name", default="consensus")
    parser.add_argument("--expr-prop", type=float, default=0.15)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--p-value-threshold", type=float, default=0.01)
    parser.add_argument("--first-seed", type=int, default=20260724)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--memory-limit-fraction", type=float, default=0.8)
    parser.add_argument("--memory-per-worker-gib", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
