"""Run native STACCato's 60-subject condition/batch simulation benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import pandas as pd

SCHEMA_VERSION = "crychic-staccato-under100k-v1"
DEFAULT_REPLICATES = 100


def sha256_file(path: Path) -> str:
    """Return a content digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON with a terminal newline."""

    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_metadata(repo_root: Path) -> dict[str, Any]:
    """Read commit and dirty status without changing the worktree."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _run_worker(command: Sequence[str], label: str) -> tuple[str, float, int, str]:
    """Run one R worker and capture its diagnostic output."""

    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    log = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"STACCato worker failed for {label}: {log}")
    return label, elapsed, completed.returncode, log


def _r_package_versions(rscript: Path, r_library: Path) -> dict[str, str]:
    """Read versions from the isolated R library."""

    expression = (
        "for (pkg in c('tensorregress','rTensor','pracma','CJIVE')) "
        "cat(pkg, as.character(packageVersion(pkg)), '\\n')"
    )
    completed = subprocess.run(
        [str(rscript), "-e", expression],
        check=True,
        capture_output=True,
        env={**os.environ, "R_LIBS_USER": str(r_library)},
        text=True,
    )
    versions = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            versions[fields[0]] = fields[1]
    if set(versions) != {"tensorregress", "rTensor", "pracma", "CJIVE"}:
        raise ValueError("could not resolve all native STACCato package versions")
    return versions


def run_campaign(args: argparse.Namespace) -> None:
    """Run all paper designs and aggregate their native effect-estimation metrics."""

    if args.output.exists():
        raise FileExistsError(f"output must not already exist: {args.output}")
    config = json.loads(args.contract.read_text(encoding="utf-8"))
    record = next(
        item
        for item in config["benchmarks"]
        if item["benchmark_id"] == "staccato_condition_batch_simulation"
    )
    if args.replicates != record["design"]["replicates"]:
        raise ValueError("replicate count differs from the frozen under-100k contract")
    for path in (args.rscript, args.r_worker, args.staccato_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.r_library.is_dir():
        raise FileNotFoundError(args.r_library)

    args.output.mkdir(parents=True)
    worker_dir = args.output / "workers"
    log_dir = args.output / "logs"
    timing_dir = args.output / "timing"
    worker_dir.mkdir()
    log_dir.mkdir()
    timing_dir.mkdir()
    commands: list[tuple[list[str], str]] = []
    command_rows: list[dict[str, str]] = []
    for index in range(1, args.replicates + 1):
        seed = args.first_seed + index
        label = f"rep{index:03d}_seed{seed}"
        result_path = worker_dir / f"{label}.tsv"
        timing_path = timing_dir / f"{label}.tsv"
        command = [
            "/usr/bin/time",
            "-f",
            "%e\\t%M",
            "-o",
            str(timing_path),
            "env",
            f"R_LIBS_USER={args.r_library}",
            "OPENBLAS_NUM_THREADS=1",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            str(args.rscript),
            str(args.r_worker),
            "--seed",
            str(seed),
            "--output",
            str(result_path),
            "--staccato-source",
            str(args.staccato_source),
        ]
        commands.append((command, label))
        command_rows.append({"label": label, "command": " ".join(command)})
    pd.DataFrame(command_rows).to_csv(
        args.output / "commands.tsv", sep="\t", index=False
    )

    started = time.monotonic()
    worker_runtime: dict[str, tuple[float, str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_worker, command, label): label
            for command, label in commands
        }
        for future in concurrent.futures.as_completed(futures):
            label, elapsed, _, log = future.result()
            worker_runtime[label] = (elapsed, log)
            (log_dir / f"{label}.log").write_text(log + "\n", encoding="utf-8")

    tables: list[pd.DataFrame] = []
    for index in range(1, args.replicates + 1):
        seed = args.first_seed + index
        label = f"rep{index:03d}_seed{seed}"
        table = pd.read_csv(worker_dir / f"{label}.tsv", sep="\t")
        timing = (timing_dir / f"{label}.tsv").read_text(encoding="utf-8").strip()
        seconds, peak_rss_kib = timing.split("\t")
        table["worker_wall_seconds"] = worker_runtime[label][0]
        table["worker_peak_rss_kib"] = int(peak_rss_kib)
        table["native_rscript_seconds"] = float(seconds)
        tables.append(table)
    metrics = pd.concat(tables, axis=0, ignore_index=True)
    metrics.to_csv(args.output / "replicate_metrics.tsv", sep="\t", index=False)
    estimable = metrics.loc[metrics["status"].eq("complete")].copy()
    aggregate = (
        estimable.groupby(["design", "method"], as_index=False)
        .agg(
            replicate_count=("seed", "size"),
            mse_mean=("mse", "mean"),
            mse_sd=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            sign_accuracy_mean=("sign_accuracy_nonzero", "mean"),
            opposite_direction_rate_mean=("opposite_direction_rate", "mean"),
            top100_recovery_mean=("top100_recovery", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            worker_peak_rss_kib_max=("worker_peak_rss_kib", "max"),
        )
        .sort_values(["design", "method"])
    )
    aggregate.to_csv(args.output / "aggregate_metrics.tsv", sep="\t", index=False)
    status = (
        metrics.groupby(["design", "method", "status"], as_index=False)
        .size()
        .sort_values(["design", "method", "status"])
    )
    status.to_csv(args.output / "status_counts.tsv", sep="\t", index=False)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "protocol_status": "paper_design_public_low_rank_generator_reconstruction",
            "contract": {
                "path": str(args.contract),
                "sha256": sha256_file(args.contract),
                "benchmark_id": record["benchmark_id"],
            },
            "native_software": {
                "rscript": str(args.rscript),
                "r_worker": {
                    "path": str(args.r_worker),
                    "sha256": sha256_file(args.r_worker),
                },
                "staccato_source": {
                    "path": str(args.staccato_source),
                    "sha256": sha256_file(args.staccato_source),
                },
                "r_packages": _r_package_versions(args.rscript, args.r_library),
            },
            "source_repository": git_metadata(args.repo_root),
            "execution": {
                "orchestrator_python": sys.version,
                "platform": platform.platform(),
                "workers": args.workers,
                "replicates": args.replicates,
                "first_seed": args.first_seed,
                "campaign_wall_seconds": time.monotonic() - started,
            },
            "crychic_applicability": {
                "method": "CRYCHIC common-functional OOF",
                "status": "NE",
                "reason": (
                    "paper design has unpaired subjects across conditions; public "
                    "OOF contrast requires every subject in every context"
                ),
            },
            "artifacts": {
                name: {"sha256": sha256_file(args.output / name)}
                for name in (
                    "commands.tsv",
                    "replicate_metrics.tsv",
                    "aggregate_metrics.tsv",
                    "status_counts.tsv",
                )
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the native-STACCato campaign parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "suggest_v5_under100k_v1.json",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--rscript",
        type=Path,
        default=Path("/home/agent1/miniforge3/bin/Rscript"),
    )
    parser.add_argument(
        "--r-worker",
        type=Path,
        default=Path(__file__).with_name("staccato_under100k_worker.R"),
    )
    parser.add_argument("--staccato-source", type=Path, required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--first-seed", type=int, default=20_260_724)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP007
    args = build_parser().parse_args(argv)
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
