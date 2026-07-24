"""Execute and aggregate all eligible paper-native DCST sweep realizations."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

SCHEMA_VERSION = "crychic-dcst-under100k-campaign-v1"
DEFAULT_REPLICATES = 25
DEFAULT_WORKERS = 8
MEMORY_POLL_SECONDS = 0.25


@dataclass(frozen=True)
class Task:
    """One isolated realization from one of the two paper-native sweeps."""

    sweep: str
    axis_name: str
    axis_value: int
    subjects_per_condition: int
    receiver_cells_in_condition_2: int
    replicate: int
    seed: int

    @property
    def label(self) -> str:
        return (
            f"{self.sweep}_{self.axis_name}{self.axis_value:03d}"
            f"_rep{self.replicate:02d}_seed{self.seed}"
        )


def sha256_file(path: Path) -> str:
    """Return a content digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON with a final newline."""

    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_metadata(repo_root: Path) -> dict[str, Any]:
    """Read source commit and dirtiness without changing repository state."""

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


def system_used_fraction() -> float:
    """Read current memory pressure from Linux procfs."""

    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(value.strip().split()[0])
    return 1.0 - values["MemAvailable"] / values["MemTotal"]


def planned_tasks(
    contract: Mapping[str, Any],
    *,
    replicates: int,
    first_seed: int,
    requested_sweeps: str,
) -> list[Task]:
    """Build exactly the independent sweep settings frozen in the contract."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    records = {item["benchmark_id"]: item for item in contract["benchmarks"]}
    subject = records["dcst_subject_count_sweep"]["design"]
    receiver = records["dcst_receiver_cell_count_sweep"]["design"]
    selected = (
        {"subject_count", "receiver_cell_count"}
        if requested_sweeps == "all"
        else {requested_sweeps}
    )
    tasks: list[Task] = []
    seed_offset = 0
    if "subject_count" in selected:
        for subjects in subject["subjects_per_condition"]:
            for replicate in range(1, replicates + 1):
                tasks.append(
                    Task(
                        sweep="subject_count",
                        axis_name="subjects_per_condition",
                        axis_value=int(subjects),
                        subjects_per_condition=int(subjects),
                        receiver_cells_in_condition_2=500,
                        replicate=replicate,
                        seed=first_seed + seed_offset,
                    )
                )
                seed_offset += 1
    if "receiver_cell_count" in selected:
        for receiver_cells in receiver["receiver_cells_in_condition_2"]:
            for replicate in range(1, replicates + 1):
                tasks.append(
                    Task(
                        sweep="receiver_cell_count",
                        axis_name="receiver_cells_in_condition_2",
                        axis_value=int(receiver_cells),
                        subjects_per_condition=int(receiver["subjects_per_condition"]),
                        receiver_cells_in_condition_2=int(receiver_cells),
                        replicate=replicate,
                        seed=first_seed + seed_offset,
                    )
                )
                seed_offset += 1
    if not tasks:
        raise ValueError("no selected DCST sweeps")
    return tasks


def _run_task(
    task: Task,
    command: Sequence[str],
    timing_path: Path,
    memory_limit_fraction: float,
) -> dict[str, Any]:
    """Run one isolated realization and terminate it if the memory guard trips."""

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    memory_limit_breached = False
    while process.poll() is None:
        if system_used_fraction() >= memory_limit_fraction:
            memory_limit_breached = True
            os.killpg(process.pid, signal.SIGTERM)
            break
        time.sleep(MEMORY_POLL_SECONDS)
    stdout, stderr = process.communicate()
    elapsed = time.monotonic() - started
    record: dict[str, Any] = {
        "label": task.label,
        "sweep": task.sweep,
        "axis_name": task.axis_name,
        "axis_value": task.axis_value,
        "subjects_per_condition": task.subjects_per_condition,
        "receiver_cells_in_condition_2": task.receiver_cells_in_condition_2,
        "replicate": task.replicate,
        "seed": task.seed,
        "wall_seconds": elapsed,
        "returncode": process.returncode,
        "memory_limit_breached": memory_limit_breached,
        "timing_path": str(timing_path),
        "stdout": stdout,
        "stderr": stderr,
    }
    if timing_path.is_file():
        fields = timing_path.read_text(encoding="utf-8").strip().split("\t")
        if len(fields) == 2:
            record["native_wall_seconds"] = float(fields[0])
            record["peak_rss_kib"] = int(fields[1])
    return record


def _command(
    args: argparse.Namespace,
    task: Task,
    output: Path,
    timing_path: Path,
) -> list[str]:
    return [
        "/usr/bin/time",
        "-f",
        "%e\\t%M",
        "-o",
        str(timing_path),
        "env",
        "PYTHONPATH=" + str(args.repo_root / "src") + ":" + str(args.repo_root),
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        sys.executable,
        str(args.runner),
        "run-full",
        "--output",
        str(output),
        "--vendor-root",
        str(args.vendor_root),
        "--repo-root",
        str(args.repo_root),
        "--sweep",
        task.sweep,
        "--subjects-per-condition",
        str(task.subjects_per_condition),
        "--receiver-cells-in-condition-2",
        str(task.receiver_cells_in_condition_2),
        "--seed",
        str(task.seed),
    ]


def run_campaign(args: argparse.Namespace) -> None:
    """Execute all requested tasks and write only checksum-bound aggregates."""

    if args.output.exists():
        raise FileExistsError(f"output must not already exist: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "crychic-suggest-v5-under100k-contract-v1":
        raise ValueError("unexpected frozen benchmark contract")
    records = {item["benchmark_id"]: item for item in contract["benchmarks"]}
    expected_replicates = records["dcst_subject_count_sweep"]["design"]["replicates"]
    if args.replicates != expected_replicates:
        raise ValueError("replicate count differs from the frozen DCST contract")
    for path in (args.runner, args.vendor_root, args.repo_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0.0 < args.memory_limit_fraction < 1.0:
        raise ValueError("memory_limit_fraction must be between zero and one")
    if system_used_fraction() >= args.memory_limit_fraction:
        raise RuntimeError("memory guard already exceeded before campaign start")

    tasks = planned_tasks(
        contract,
        replicates=args.replicates,
        first_seed=args.first_seed,
        requested_sweeps=args.sweeps,
    )
    args.output.mkdir(parents=True)
    realization_root = args.output / "realizations"
    timing_root = args.output / "timing"
    log_root = args.output / "logs"
    realization_root.mkdir()
    timing_root.mkdir()
    log_root.mkdir()
    command_rows: list[dict[str, str]] = []
    commands: list[tuple[Task, list[str], Path, Path]] = []
    for task in tasks:
        output = (
            realization_root
            / task.sweep
            / f"{task.axis_name}_{task.axis_value:03d}"
            / f"rep{task.replicate:02d}_seed{task.seed}"
        )
        timing_path = timing_root / f"{task.label}.tsv"
        command = _command(args, task, output, timing_path)
        commands.append((task, command, timing_path, output))
        command_rows.append({"label": task.label, "command": shlex.join(command)})
    pd.DataFrame(command_rows).to_csv(
        args.output / "commands.tsv", sep="\t", index=False
    )

    campaign_started = time.monotonic()
    records_out: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_task,
                task,
                command,
                timing_path,
                args.memory_limit_fraction,
            ): (task, output)
            for task, command, timing_path, output in commands
        }
        for future in concurrent.futures.as_completed(futures):
            task, output = futures[future]
            record = future.result()
            record["output"] = str(output)
            records_out.append(record)
            (log_root / f"{task.label}.stdout.log").write_text(
                str(record.pop("stdout")) + "\n", encoding="utf-8"
            )
            (log_root / f"{task.label}.stderr.log").write_text(
                str(record.pop("stderr")) + "\n", encoding="utf-8"
            )
    timings = pd.DataFrame.from_records(records_out).sort_values(
        ["sweep", "axis_value", "replicate"]
    )
    timings.to_csv(args.output / "task_timings.tsv", sep="\t", index=False)
    failures = timings.loc[
        timings["returncode"].ne(0) | timings["memory_limit_breached"].astype(bool)
    ].copy()
    failures.to_csv(args.output / "failures.tsv", sep="\t", index=False)
    if not failures.empty:
        write_json(
            args.output / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "partial",
                "reason": "one_or_more_realizations_failed",
                "tasks_requested": len(tasks),
                "tasks_completed": int(timings["returncode"].eq(0).sum()),
                "failures_sha256": sha256_file(args.output / "failures.tsv"),
            },
        )
        raise RuntimeError(f"{len(failures)} DCST realizations failed")

    metric_tables: list[pd.DataFrame] = []
    for task, _, _, output in commands:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError(f"realization manifest is not complete: {task.label}")
        metrics = pd.read_csv(output / "evaluation" / "method_metrics.tsv", sep="\t")
        metrics.insert(0, "seed", task.seed)
        metrics.insert(0, "replicate", task.replicate)
        metrics.insert(0, "axis_value", task.axis_value)
        metrics.insert(0, "axis_name", task.axis_name)
        metrics.insert(0, "sweep", task.sweep)
        metrics.insert(0, "cells", manifest["dimensions"]["cells"])
        metric_tables.append(metrics)
    all_metrics = pd.concat(metric_tables, axis=0, ignore_index=True)
    all_metrics.to_csv(
        args.output / "all_replicate_method_metrics.tsv", sep="\t", index=False
    )
    numeric = [
        "coverage",
        "auroc",
        "auprc",
        "prevalence_adjusted_ap_10pct",
        "direction_accuracy",
        "sensitivity_at_005",
        "specificity_at_005",
        "native_type1_error_005",
        "native_empirical_fdr_005",
    ]
    aggregate = (
        all_metrics.groupby(
            ["sweep", "axis_name", "axis_value", "method"], as_index=False
        )
        .agg(
            cells=("cells", "first"),
            replicates=("replicate", "size"),
            **{f"{column}_mean": (column, "mean") for column in numeric},
            **{f"{column}_sd": (column, "std") for column in numeric},
        )
        .sort_values(["sweep", "axis_value", "method"])
    )
    aggregate.to_csv(args.output / "aggregate_metrics.tsv", sep="\t", index=False)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "protocol_status": (
                "paper_native_independent_sweeps_protocol_reimplementation"
            ),
            "contract": {
                "path": str(args.contract),
                "sha256": sha256_file(args.contract),
            },
            "source_repository": git_metadata(args.repo_root),
            "runner": {
                "path": str(args.runner),
                "sha256": sha256_file(args.runner),
            },
            "execution": {
                "platform": platform.platform(),
                "workers": args.workers,
                "memory_limit_fraction": args.memory_limit_fraction,
                "replicates": args.replicates,
                "first_seed": args.first_seed,
                "requested_sweeps": args.sweeps,
                "campaign_wall_seconds": time.monotonic() - campaign_started,
                "tasks": len(tasks),
            },
            "artifacts": {
                name: {"sha256": sha256_file(args.output / name)}
                for name in (
                    "commands.tsv",
                    "task_timings.tsv",
                    "failures.tsv",
                    "all_replicate_method_metrics.tsv",
                    "aggregate_metrics.tsv",
                )
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_dcst_under100k.py"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "suggest_v5_under100k_v1.json",
    )
    parser.add_argument(
        "--sweeps",
        choices=("all", "subject_count", "receiver_cell_count"),
        default="all",
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--first-seed", type=int, default=20_260_724)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--memory-limit-fraction", type=float, default=0.80)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP007
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
