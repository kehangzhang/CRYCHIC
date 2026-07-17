"""Plan and run a resumable, memory-gated IPF cohort benchmark queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json

TASK_SCHEMA = "xie-ipf-cohort-task-v1"
TASK_MANIFEST = "cohort_task_manifest.json"
OUTPUT_PLACEHOLDER = "{output_dir}"
LEDGER_COLUMNS = (
    "task_id",
    "study_id",
    "sample_id",
    "stage",
    "arm",
    "command_json",
    "command_sha256",
    "publish_dir",
    "expected_outputs_json",
    "dependencies_json",
    "threads",
    "status",
    "attempts",
    "exit_code",
    "elapsed_seconds",
    "peak_rss_bytes",
    "memory_fraction_at_start",
    "failure_reason",
    "started_unix_seconds",
    "finished_unix_seconds",
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "RCPP_PARALLEL_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    study_id: str
    sample_id: str
    stage: str
    arm: str
    command: tuple[str, ...]
    publish_dir: Path
    expected_outputs: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    threads: int = 1

    def __post_init__(self) -> None:
        canonical = (self.task_id, self.study_id, self.sample_id, self.stage, self.arm)
        if any(not value or value != value.strip() for value in canonical):
            raise ValueError("task identifiers must be canonical non-empty strings")
        if not self.command or OUTPUT_PLACEHOLDER not in self.command:
            raise ValueError(
                "task command must contain the output directory placeholder"
            )
        if self.publish_dir.name in {"", ".", ".."}:
            raise ValueError("task publish_dir must identify a concrete directory")
        if not self.expected_outputs or any(
            Path(value).is_absolute() or ".." in Path(value).parts
            for value in self.expected_outputs
        ):
            raise ValueError("expected outputs must be non-empty relative paths")
        if len(set(self.expected_outputs)) != len(self.expected_outputs):
            raise ValueError("expected task outputs must not contain duplicates")
        if isinstance(self.threads, bool) or self.threads != 1:
            raise ValueError("IPF sample tasks must declare exactly one thread")

    @property
    def command_sha256(self) -> str:
        payload = json.dumps(self.command, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _RunningTask:
    spec: TaskSpec
    process: subprocess.Popen[bytes]
    staging_dir: Path
    stdout_handle: Any
    stderr_handle: Any
    started_monotonic: float
    started_unix: float
    memory_fraction_at_start: float
    peak_rss_bytes: int = 0


def read_memory_fraction(meminfo_path: str | Path = "/proc/meminfo") -> float:
    """Return (MemTotal - MemAvailable) / MemTotal from Linux procfs."""

    values: dict[str, int] = {}
    for line in Path(meminfo_path).read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in {"MemTotal:", "MemAvailable:"}:
            values[fields[0][:-1]] = int(fields[1]) * 1024
    if values.keys() != {"MemTotal", "MemAvailable"} or values["MemTotal"] <= 0:
        raise RuntimeError("/proc/meminfo lacks MemTotal or MemAvailable")
    fraction = (values["MemTotal"] - values["MemAvailable"]) / values["MemTotal"]
    return min(1.0, max(0.0, float(fraction)))


def _task_record(spec: TaskSpec) -> dict[str, object]:
    return {
        "task_id": spec.task_id,
        "study_id": spec.study_id,
        "sample_id": spec.sample_id,
        "stage": spec.stage,
        "arm": spec.arm,
        "command_json": json.dumps(spec.command, separators=(",", ":")),
        "command_sha256": spec.command_sha256,
        "publish_dir": str(spec.publish_dir),
        "expected_outputs_json": json.dumps(
            spec.expected_outputs, separators=(",", ":")
        ),
        "dependencies_json": json.dumps(spec.dependencies, separators=(",", ":")),
        "threads": spec.threads,
        "status": "pending",
        "attempts": 0,
        "exit_code": pd.NA,
        "elapsed_seconds": pd.NA,
        "peak_rss_bytes": pd.NA,
        "memory_fraction_at_start": pd.NA,
        "failure_reason": pd.NA,
        "started_unix_seconds": pd.NA,
        "finished_unix_seconds": pd.NA,
    }


def write_task_plan(tasks: Sequence[TaskSpec], path: str | Path) -> pd.DataFrame:
    """Write a deterministic task ledger in its initial pending state."""

    if not tasks:
        raise ValueError("IPF task plan must not be empty")
    identifiers = [task.task_id for task in tasks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("IPF task IDs must be unique")
    known = set(identifiers)
    for task in tasks:
        missing = set(task.dependencies).difference(known)
        if missing:
            raise ValueError(
                f"task {task.task_id} has unknown dependencies: {sorted(missing)}"
            )
        if task.task_id in task.dependencies:
            raise ValueError(f"task {task.task_id} depends on itself")
    table = pd.DataFrame.from_records(
        [_task_record(task) for task in tasks], columns=LEDGER_COLUMNS
    )
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_ledger(table, output)
    return table


def load_task_plan(path: str | Path) -> list[TaskSpec]:
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(LEDGER_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"IPF task ledger is missing columns: {sorted(missing)}")
    tasks: list[TaskSpec] = []
    for row in table.itertuples(index=False):
        command = json.loads(row.command_json)
        expected = json.loads(row.expected_outputs_json)
        dependencies = json.loads(row.dependencies_json)
        if not all(
            isinstance(value, str) for value in command + expected + dependencies
        ):
            raise ValueError(f"task {row.task_id} JSON fields must contain strings")
        task = TaskSpec(
            task_id=row.task_id,
            study_id=row.study_id,
            sample_id=row.sample_id,
            stage=row.stage,
            arm=row.arm,
            command=tuple(command),
            publish_dir=Path(row.publish_dir).expanduser().resolve(),
            expected_outputs=tuple(expected),
            dependencies=tuple(dependencies),
            threads=int(row.threads),
        )
        if row.command_sha256 != task.command_sha256:
            raise ValueError(f"task {row.task_id} command digest is inconsistent")
        tasks.append(task)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("IPF task ledger contains duplicate task IDs")
    return tasks


def _atomic_write_ledger(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    table.loc[:, LEDGER_COLUMNS].to_csv(temporary, sep="\t", index=False, na_rep="")
    temporary.replace(path)


def _validate_completion(spec: TaskSpec) -> bool:
    manifest_path = spec.publish_dir / TASK_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("schema_version") != TASK_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("task_id") != spec.task_id
        or manifest.get("command_sha256") != spec.command_sha256
    ):
        return False
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(spec.expected_outputs):
        return False
    for relative in spec.expected_outputs:
        record = outputs.get(relative)
        path = spec.publish_dir / relative
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or record.get("sha256") != sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            return False
    return True


def _publish_task(running: _RunningTask, *, peak_rss_bytes: int) -> None:
    spec = running.spec
    outputs: dict[str, dict[str, object]] = {}
    for relative in spec.expected_outputs:
        path = running.staging_dir / relative
        if not path.is_file():
            raise RuntimeError(f"task {spec.task_id} did not create {relative}")
        outputs[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    algorithm_manifest = running.staging_dir / "manifest.json"
    if algorithm_manifest.is_file():
        payload = json.loads(algorithm_manifest.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(
                f"task {spec.task_id} algorithm manifest is not complete"
            )
    write_json(
        running.staging_dir / TASK_MANIFEST,
        {
            "schema_version": TASK_SCHEMA,
            "status": "complete",
            "task_id": spec.task_id,
            "study_id": spec.study_id,
            "sample_id": spec.sample_id,
            "stage": spec.stage,
            "arm": spec.arm,
            "command": list(spec.command),
            "command_sha256": spec.command_sha256,
            "threads": spec.threads,
            "dependencies": list(spec.dependencies),
            "elapsed_seconds": time.monotonic() - running.started_monotonic,
            "peak_rss_bytes": peak_rss_bytes,
            "memory_fraction_at_start": running.memory_fraction_at_start,
            "started_unix_seconds": running.started_unix,
            "finished_unix_seconds": time.time(),
            "outputs": outputs,
        },
    )
    if spec.publish_dir.exists():
        raise FileExistsError(
            f"refusing to replace existing result directory: {spec.publish_dir}"
        )
    spec.publish_dir.parent.mkdir(parents=True, exist_ok=True)
    running.staging_dir.replace(spec.publish_dir)


def _process_tree_rss(root_pid: int) -> int:
    processes: dict[int, int] = {}
    rss: dict[int, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat_path.read_text(encoding="ascii").split()
            pid, parent = int(fields[0]), int(fields[3])
            status = stat_path.with_name("status").read_text(encoding="ascii")
        except (OSError, ValueError, IndexError):
            continue
        processes[pid] = parent
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss[pid] = int(line.split()[1]) * 1024
                break
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in processes.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss.get(pid, 0) for pid in descendants)


def _terminate_task(running: _RunningTask, *, grace_seconds: float = 5.0) -> None:
    if running.process.poll() is not None:
        return
    try:
        os.killpg(running.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        running.process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(running.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        running.process.wait()


def _replace_output_placeholder(command: Sequence[str], staging: Path) -> list[str]:
    return [str(staging) if value == OUTPUT_PLACEHOLDER else value for value in command]


def _start_task(
    spec: TaskSpec,
    *,
    logs_root: Path,
    memory_fraction: float,
    cwd: Path | None,
) -> _RunningTask:
    if spec.publish_dir.exists():
        raise FileExistsError(
            f"result directory exists without a valid task manifest: {spec.publish_dir}"
        )
    staging = spec.publish_dir.with_name(
        f".{spec.publish_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    staging.mkdir(parents=True, exist_ok=False)
    logs_root.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_root / f"{spec.task_id}.stdout.log"
    stderr_path = logs_root / f"{spec.task_id}.stderr.log"
    stdout_handle = stdout_path.open("ab")
    stderr_handle = stderr_path.open("ab")
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    command = _replace_output_placeholder(spec.command, staging)
    try:
        process = subprocess.Popen(
            command,
            cwd=None if cwd is None else str(cwd),
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
    except BaseException:
        stdout_handle.close()
        stderr_handle.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    now = time.time()
    return _RunningTask(
        spec=spec,
        process=process,
        staging_dir=staging,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        started_monotonic=time.monotonic(),
        started_unix=now,
        memory_fraction_at_start=memory_fraction,
    )


def run_task_queue(
    ledger_path: str | Path,
    *,
    max_workers: int = 8,
    soft_memory_fraction: float = 0.65,
    hard_memory_fraction: float = 0.70,
    poll_seconds: float = 1.0,
    max_attempts: int = 2,
    cwd: str | Path | None = None,
    memory_reader: Callable[[], float] = read_memory_fraction,
) -> pd.DataFrame:
    """Run ready tasks while enforcing process-level and system-memory bounds."""

    if isinstance(max_workers, bool) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if not 0 < soft_memory_fraction < hard_memory_fraction <= 0.70:
        raise ValueError("memory thresholds must satisfy 0 < soft < hard <= 0.70")
    if poll_seconds < 0 or max_attempts < 1:
        raise ValueError("poll_seconds and max_attempts are invalid")
    path = Path(ledger_path).expanduser().resolve()
    tasks = load_task_plan(path)
    specs = {task.task_id: task for task in tasks}
    ledger = pd.read_csv(path, sep="\t", keep_default_na=False)
    # A resumed TSV has empty numeric fields, so pandas may infer Arrow string
    # columns.  The ledger deliberately mixes pending blanks with later numeric
    # observations; object storage keeps those state transitions explicit.
    ledger = ledger.loc[:, LEDGER_COLUMNS].astype(object)
    if set(ledger["task_id"]) != set(specs):
        raise ValueError("task plan and ledger identifiers differ")
    ledger = ledger.set_index("task_id", drop=False)
    for task_id, spec in specs.items():
        if _validate_completion(spec):
            ledger.at[task_id, "status"] = "complete"
            ledger.at[task_id, "failure_reason"] = ""
        elif ledger.at[task_id, "status"] in {"running", "complete", "blocked"}:
            ledger.at[task_id, "status"] = "pending"
    _atomic_write_ledger(ledger.reset_index(drop=True), path)
    logs_root = path.parent / "logs"
    working_directory = None if cwd is None else Path(cwd).expanduser().resolve()
    running: dict[str, _RunningTask] = {}

    def update(task_id: str, **values: object) -> None:
        for key, value in values.items():
            ledger.at[task_id, key] = value
        _atomic_write_ledger(ledger.reset_index(drop=True), path)

    def cleanup_running(reason: str) -> None:
        for task_id, active in list(running.items()):
            try:
                active.peak_rss_bytes = max(
                    active.peak_rss_bytes, _process_tree_rss(active.process.pid)
                )
                _terminate_task(active)
            finally:
                active.stdout_handle.close()
                active.stderr_handle.close()
                shutil.rmtree(active.staging_dir, ignore_errors=True)
            update(
                task_id,
                status="failed",
                exit_code=active.process.returncode,
                elapsed_seconds=time.monotonic() - active.started_monotonic,
                peak_rss_bytes=active.peak_rss_bytes,
                failure_reason=reason,
                finished_unix_seconds=time.time(),
            )
            del running[task_id]

    previous_sigterm: Any = None
    handler_installed = threading.current_thread() is threading.main_thread()
    if handler_installed:
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def handle_sigterm(_signum: int, _frame: Any) -> None:
            raise SystemExit(128 + signal.SIGTERM)

        signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        return _run_queue_loop(
            ledger=ledger,
            specs=specs,
            running=running,
            path=path,
            logs_root=logs_root,
            working_directory=working_directory,
            update=update,
            max_workers=max_workers,
            soft_memory_fraction=soft_memory_fraction,
            hard_memory_fraction=hard_memory_fraction,
            poll_seconds=poll_seconds,
            max_attempts=max_attempts,
            memory_reader=memory_reader,
            cleanup_running=cleanup_running,
        )
    except BaseException:
        cleanup_running("scheduler_interrupted")
        raise
    finally:
        if handler_installed:
            signal.signal(signal.SIGTERM, previous_sigterm)


def _run_queue_loop(
    *,
    ledger: pd.DataFrame,
    specs: dict[str, TaskSpec],
    running: dict[str, _RunningTask],
    path: Path,
    logs_root: Path,
    working_directory: Path | None,
    update: Callable[..., None],
    max_workers: int,
    soft_memory_fraction: float,
    hard_memory_fraction: float,
    poll_seconds: float,
    max_attempts: int,
    memory_reader: Callable[[], float],
    cleanup_running: Callable[[str], None],
) -> pd.DataFrame:
    while True:
        completed = set(ledger.index[ledger["status"].eq("complete")])
        exhausted = set(
            ledger.index[
                ledger["status"].eq("failed")
                & pd.to_numeric(ledger["attempts"], errors="coerce")
                .fillna(0)
                .ge(max_attempts)
            ]
        )
        pending = [
            task_id
            for task_id in ledger.index
            if ledger.at[task_id, "status"] in {"pending", "failed"}
            and int(ledger.at[task_id, "attempts"] or 0) < max_attempts
        ]
        if not pending and not running:
            break
        memory_fraction = float(memory_reader())
        if not 0 <= memory_fraction <= 1:
            raise RuntimeError("memory reader returned a fraction outside [0, 1]")
        if memory_fraction >= hard_memory_fraction and running:
            cleanup_running("system_memory_hard_limit_reached")
            continue

        for task_id, active in list(running.items()):
            active.peak_rss_bytes = max(
                active.peak_rss_bytes, _process_tree_rss(active.process.pid)
            )
            exit_code = active.process.poll()
            if exit_code is None:
                continue
            active.stdout_handle.close()
            active.stderr_handle.close()
            elapsed = time.monotonic() - active.started_monotonic
            failure_reason = ""
            status = "complete"
            if exit_code != 0:
                status = "failed"
                failure_reason = f"subprocess_exit_{exit_code}"
                shutil.rmtree(active.staging_dir, ignore_errors=True)
            else:
                try:
                    _publish_task(active, peak_rss_bytes=active.peak_rss_bytes)
                except BaseException as exc:
                    status = "failed"
                    failure_reason = (
                        f"publish_validation_failed:{type(exc).__name__}:{exc}"
                    )
                    shutil.rmtree(active.staging_dir, ignore_errors=True)
            update(
                task_id,
                status=status,
                exit_code=exit_code,
                elapsed_seconds=elapsed,
                peak_rss_bytes=active.peak_rss_bytes,
                failure_reason=failure_reason,
                finished_unix_seconds=time.time(),
            )
            del running[task_id]

        if memory_fraction < soft_memory_fraction:
            ready = [
                task_id
                for task_id in pending
                if task_id not in running
                and set(specs[task_id].dependencies).issubset(completed)
            ]
            while ready and len(running) < max_workers:
                task_id = ready.pop(0)
                spec = specs[task_id]
                memory_fraction = float(memory_reader())
                if memory_fraction >= soft_memory_fraction:
                    break
                attempts = int(ledger.at[task_id, "attempts"] or 0) + 1
                try:
                    active = _start_task(
                        spec,
                        logs_root=logs_root,
                        memory_fraction=memory_fraction,
                        cwd=working_directory,
                    )
                except BaseException as exc:
                    update(
                        task_id,
                        status="failed",
                        attempts=attempts,
                        failure_reason=f"launch_failed:{type(exc).__name__}:{exc}",
                        started_unix_seconds=time.time(),
                        finished_unix_seconds=time.time(),
                    )
                    continue
                running[task_id] = active
                update(
                    task_id,
                    status="running",
                    attempts=attempts,
                    memory_fraction_at_start=memory_fraction,
                    failure_reason="",
                    started_unix_seconds=active.started_unix,
                    finished_unix_seconds="",
                )

        if not running:
            pending_with_failed_dependencies = [
                task_id
                for task_id in pending
                if set(specs[task_id].dependencies).intersection(exhausted)
            ]
            if pending_with_failed_dependencies:
                for task_id in pending_with_failed_dependencies:
                    update(
                        task_id,
                        status="blocked",
                        failure_reason="dependency_exhausted",
                        finished_unix_seconds=time.time(),
                    )
                continue
        if poll_seconds:
            time.sleep(poll_seconds)

    result = ledger.reset_index(drop=True).loc[:, LEDGER_COLUMNS]
    _atomic_write_ledger(result, path)
    return result


def _sample_manifest(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype=str)
    required = {
        "study_id",
        "sample_id",
        "dataset_id",
        "input_h5ad",
        "input_sha256",
        "status",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"sample manifest is missing columns: {sorted(missing)}")
    if not table["status"].eq("complete").all():
        raise ValueError("all sample inputs must be complete before task planning")
    for row in table.itertuples(index=False):
        input_path = Path(row.input_h5ad)
        if not input_path.is_file() or sha256_file(input_path) != row.input_sha256:
            raise ValueError(f"sample input checksum mismatch: {input_path}")
    return table.sort_values(
        ["study_id", "sample_id"], kind="stable", ignore_index=True
    )


def plan_hcommon_tasks(
    sample_manifest: str | Path,
    results_root: str | Path,
    *,
    resource: str | Path,
    resource_manifest: str | Path,
    gold: str | Path,
    liana_python_executable: str | Path = sys.executable,
    crychic_python_executable: str | Path = sys.executable,
    n_perms: int = 100,
    seed: int = 20260715,
    sample_ids: Sequence[str] | None = None,
) -> list[TaskSpec]:
    """Plan LIANA, CRYCHIC availability, and evaluation for every IPF patient."""

    if n_perms < 1 or not 0 <= seed < 2**32:
        raise ValueError("n_perms and seed are invalid")
    samples = _sample_manifest(Path(sample_manifest).expanduser().resolve())
    if sample_ids is not None:
        selected_samples = tuple(dict.fromkeys(str(value) for value in sample_ids))
        if not selected_samples or any(
            not value or value != value.strip() for value in selected_samples
        ):
            raise ValueError("sample_ids must contain canonical non-empty identifiers")
        unknown = set(selected_samples).difference(samples["sample_id"])
        if unknown:
            raise ValueError(
                f"requested samples are absent from manifest: {sorted(unknown)}"
            )
        samples = samples.loc[samples["sample_id"].isin(selected_samples)].reset_index(
            drop=True
        )
    root = Path(results_root).expanduser().resolve()
    resource_path = Path(resource).expanduser().resolve()
    resource_manifest_path = Path(resource_manifest).expanduser().resolve()
    gold_path = Path(gold).expanduser().resolve()
    for path in (resource_path, resource_manifest_path, gold_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    # Do not resolve interpreter symlinks: a venv's python often links to the
    # base binary, but invocation through the venv path is what selects its
    # package environment.
    liana_python = str(Path(liana_python_executable).expanduser().absolute())
    crychic_python = str(Path(crychic_python_executable).expanduser().absolute())
    for executable in (liana_python, crychic_python):
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"Python executable is unavailable: {executable}")
    tasks: list[TaskSpec] = []
    for row in samples.itertuples(index=False):
        prefix = f"{row.study_id}__{row.sample_id}"
        sample_root = root / row.study_id / row.sample_id / "H-common"
        liana_dir = sample_root / "liana_cellchat_literature"
        crychic_dir = sample_root / "crychic_availability"
        evaluation_dir = sample_root / "evaluation_representable"
        liana_id = f"{prefix}__liana_hcommon"
        crychic_id = f"{prefix}__crychic_hcommon"
        tasks.extend(
            [
                TaskSpec(
                    task_id=liana_id,
                    study_id=row.study_id,
                    sample_id=row.sample_id,
                    stage="score",
                    arm="liana_hcommon",
                    command=(
                        liana_python,
                        "-m",
                        "benchmarks.adapters.liana.run_literature",
                        row.input_h5ad,
                        OUTPUT_PLACEHOLDER,
                        "--dataset-id",
                        row.dataset_id,
                        "--resource-mode",
                        "H-common",
                        "--harmonized-resource",
                        str(resource_path),
                        "--harmonized-manifest",
                        str(resource_manifest_path),
                        "--n-perms",
                        str(n_perms),
                        "--seed",
                        str(seed),
                        "--n-jobs",
                        "1",
                        "--overwrite",
                    ),
                    publish_dir=liana_dir,
                    expected_outputs=(
                        "manifest.json",
                        "rank_aggregate_raw.parquet",
                        "cellchat_raw.parquet",
                    ),
                ),
                TaskSpec(
                    task_id=crychic_id,
                    study_id=row.study_id,
                    sample_id=row.sample_id,
                    stage="score",
                    arm="crychic_hcommon",
                    command=(
                        crychic_python,
                        "-m",
                        "benchmarks.adapters.crychic.run_availability",
                        row.input_h5ad,
                        OUTPUT_PLACEHOLDER,
                        "--dataset-id",
                        row.dataset_id,
                        "--resource-mode",
                        "H-common",
                        "--context-key",
                        "condition",
                        "--harmonized-resource",
                        str(resource_path),
                        "--harmonized-manifest",
                        str(resource_manifest_path),
                        "--overwrite",
                    ),
                    publish_dir=crychic_dir,
                    expected_outputs=(
                        "manifest.json",
                        "availability_scores.parquet",
                        "availability_scores.tsv",
                        "availability_mapping_summary.tsv",
                    ),
                ),
                TaskSpec(
                    task_id=f"{prefix}__evaluate_hcommon",
                    study_id=row.study_id,
                    sample_id=row.sample_id,
                    stage="evaluate",
                    arm="hcommon_representable",
                    command=(
                        crychic_python,
                        "-m",
                        "benchmarks.literature.evaluate_ipf",
                        str(liana_dir / "rank_aggregate_raw.parquet"),
                        str(gold_path),
                        OUTPUT_PLACEHOLDER,
                        "--dataset",
                        row.dataset_id,
                        "--resource-mode",
                        "H-common",
                        "--universe-mode",
                        "H-common_representable",
                        "--resource",
                        str(resource_path),
                        "--crychic",
                        str(crychic_dir / "availability_scores.parquet"),
                        "--cellchat",
                        str(liana_dir / "cellchat_raw.parquet"),
                        "--seed",
                        str(seed),
                    ),
                    publish_dir=evaluation_dir,
                    expected_outputs=(
                        "materialized_scores.parquet",
                        "point_estimates.tsv",
                        "bootstrap_summary.tsv",
                        "negative_sampling_summary.tsv",
                    ),
                    dependencies=(liana_id, crychic_id),
                ),
            ]
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    plan = subparsers.add_parser("plan", help="create H-common sample task ledger")
    plan.add_argument("sample_manifest", type=Path)
    plan.add_argument("results_root", type=Path)
    plan.add_argument("ledger", type=Path)
    plan.add_argument("--resource", type=Path, required=True)
    plan.add_argument("--resource-manifest", type=Path, required=True)
    plan.add_argument("--gold", type=Path, required=True)
    plan.add_argument("--liana-python", type=Path, default=Path(sys.executable))
    plan.add_argument("--crychic-python", type=Path, default=Path(sys.executable))
    plan.add_argument("--n-perms", type=int, default=100)
    plan.add_argument("--seed", type=int, default=20260715)
    plan.add_argument("--sample", action="append")
    run = subparsers.add_parser("run", help="run or resume a task ledger")
    run.add_argument("ledger", type=Path)
    run.add_argument("--max-workers", type=int, default=8)
    run.add_argument("--soft-memory-fraction", type=float, default=0.65)
    run.add_argument("--hard-memory-fraction", type=float, default=0.70)
    run.add_argument("--poll-seconds", type=float, default=1.0)
    run.add_argument("--max-attempts", type=int, default=2)
    run.add_argument("--cwd", type=Path)
    args = parser.parse_args()
    if args.action == "plan":
        tasks = plan_hcommon_tasks(
            args.sample_manifest,
            args.results_root,
            resource=args.resource,
            resource_manifest=args.resource_manifest,
            gold=args.gold,
            liana_python_executable=args.liana_python,
            crychic_python_executable=args.crychic_python,
            n_perms=args.n_perms,
            seed=args.seed,
            sample_ids=args.sample,
        )
        write_task_plan(tasks, args.ledger)
        print(json.dumps({"status": "planned", "tasks": len(tasks)}))
        return
    result = run_task_queue(
        args.ledger,
        max_workers=args.max_workers,
        soft_memory_fraction=args.soft_memory_fraction,
        hard_memory_fraction=args.hard_memory_fraction,
        poll_seconds=args.poll_seconds,
        max_attempts=args.max_attempts,
        cwd=args.cwd,
    )
    counts = result["status"].value_counts().sort_index().to_dict()
    print(json.dumps({"status_counts": counts}, sort_keys=True))
    if counts != {"complete": len(result)}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "OUTPUT_PLACEHOLDER",
    "TASK_MANIFEST",
    "TaskSpec",
    "plan_hcommon_tasks",
    "read_memory_fraction",
    "run_task_queue",
    "write_task_plan",
]
