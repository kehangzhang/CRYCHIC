"""Run the frozen CellChat, LIANA, and scSeqCommDiff three-group panel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]

from benchmarks.adapters.common import json_safe, sha256_file

SCHEMA_VERSION = "crychic-three-group-external-panel-v1"
FIXTURE_SCHEMA_VERSION = "crychic-three-group-fixture-v2"
METHODS = ("cellchat", "liana", "scseqcommdiff")


@dataclass(frozen=True, slots=True)
class PanelSettings:
    adapter_root: Path
    database_root: Path
    python: Path
    liana_python: Path
    cellchat_environment: str
    scseq_rscript: Path
    max_workers: int = 16
    memory_limit_percent: float = 70.0
    task_threads: int = 8


@dataclass(frozen=True, slots=True)
class PanelTask:
    method: str
    dataset_id: str
    input_path: Path
    input_manifest: Path
    input_sha256: str
    output_dir: Path
    log_path: Path
    target: str | None = None
    reference: str | None = None
    contrast: str | None = None

    @property
    def task_id(self) -> str:
        suffix = f"__{self.contrast}" if self.contrast is not None else ""
        return f"{self.method}__{self.dataset_id}{suffix}"


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_state(root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _bound_fixture_input(fixture_dir: Path, relative_manifest: str) -> tuple[Path, str]:
    manifest_path = (fixture_dir / relative_manifest).resolve()
    manifest = _read_json(manifest_path)
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError(
            f"fixture input manifest lacks output binding: {manifest_path}"
        )
    filename = output.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"fixture input filename is invalid: {manifest_path}")
    input_path = manifest_path.parent / filename
    expected = output.get("sha256")
    if not input_path.is_file() or sha256_file(input_path) != expected:
        raise ValueError(f"fixture input checksum mismatch: {input_path}")
    return input_path, str(expected)


def build_tasks(
    fixture_dir: Path,
    output_dir: Path,
    *,
    methods: tuple[str, ...] = METHODS,
) -> tuple[list[PanelTask], dict[str, Any]]:
    """Build and validate the immutable external-method execution matrix."""

    invalid = set(methods).difference(METHODS)
    if invalid or not methods:
        raise ValueError(f"unsupported or empty method set: {sorted(invalid)}")
    fixture_dir = fixture_dir.resolve()
    fixture = _read_json(fixture_dir / "manifest.json")
    if fixture.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError("three-group external panel requires fixture v2")
    records = fixture.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("three-group fixture contains no records")
    resource = fixture.get("resource")
    if not isinstance(resource, dict):
        raise ValueError("three-group fixture lacks resource provenance")
    resource_path = fixture_dir / str(resource.get("filename"))
    resource_manifest = fixture_dir / str(resource.get("manifest"))
    if (
        not resource_path.is_file()
        or sha256_file(resource_path) != resource.get("sha256")
        or not resource_manifest.is_file()
    ):
        raise ValueError("three-group fixture resource binding is invalid")
    tasks: list[PanelTask] = []
    logs = output_dir / "_logs"
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("three-group fixture record must be an object")
        dataset_id = str(raw.get("dataset_id", ""))
        if not dataset_id:
            raise ValueError("three-group fixture record lacks dataset_id")
        full_input, full_sha = _bound_fixture_input(
            fixture_dir, str(raw["manifest"])
        )
        for method in methods:
            if method == "scseqcommdiff":
                pairs = raw.get("pairs")
                if not isinstance(pairs, list) or len(pairs) != 3:
                    raise ValueError(f"fixture pair axis is invalid: {dataset_id}")
                for pair in pairs:
                    if not isinstance(pair, dict):
                        raise ValueError("fixture pair record must be an object")
                    contrast = str(pair["contrast"])
                    pair_input, pair_sha = _bound_fixture_input(
                        fixture_dir, str(pair["manifest"])
                    )
                    name = f"scseq__{dataset_id}__{contrast}"
                    tasks.append(
                        PanelTask(
                            method=method,
                            dataset_id=dataset_id,
                            input_path=pair_input,
                            input_manifest=(
                                fixture_dir / str(pair["manifest"])
                            ).resolve(),
                            input_sha256=pair_sha,
                            output_dir=output_dir / name,
                            log_path=logs / f"{name}.log",
                            target=str(pair["target"]),
                            reference=str(pair["reference"]),
                            contrast=contrast,
                        )
                    )
            else:
                name = f"{method}__{dataset_id}"
                tasks.append(
                    PanelTask(
                        method=method,
                        dataset_id=dataset_id,
                        input_path=full_input,
                        input_manifest=(fixture_dir / str(raw["manifest"])).resolve(),
                        input_sha256=full_sha,
                        output_dir=output_dir / name,
                        log_path=logs / f"{name}.log",
                    )
                )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("three-group panel task identifiers are not unique")
    fixture["resolved_resource_path"] = str(resource_path.resolve())
    fixture["resolved_resource_manifest"] = str(resource_manifest.resolve())
    return tasks, fixture


def _task_manifest_path(task: PanelTask) -> Path:
    filename = (
        "run_manifest.json" if task.method == "scseqcommdiff" else "manifest.json"
    )
    return task.output_dir / filename


def _completed_task(task: PanelTask, *, expected_commit: str) -> bool:
    path = _task_manifest_path(task)
    if not path.is_file():
        return False
    try:
        manifest = _read_json(path)
        if manifest.get("status") != "complete":
            return False
        code = manifest.get("code")
        if not isinstance(code, dict) or code != {
            "commit": expected_commit,
            "dirty": False,
        }:
            return False
        if task.method == "scseqcommdiff":
            preflight = manifest.get("preflight")
            source = preflight.get("input") if isinstance(preflight, dict) else None
        else:
            source = manifest.get("input")
        return isinstance(source, dict) and source.get("sha256") == task.input_sha256
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _build_command(
    task: PanelTask,
    fixture: dict[str, Any],
    settings: PanelSettings,
) -> list[str]:
    resource = str(fixture["resolved_resource_path"])
    resource_manifest = str(fixture["resolved_resource_manifest"])
    common = [
        str(task.input_path),
        str(task.output_dir),
        "--dataset-id",
        task.dataset_id,
        "--context-key",
        "condition",
    ]
    harmonized = [
        "--resource-mode",
        "H-common",
        "--harmonized-resource",
        resource,
        "--harmonized-manifest",
        resource_manifest,
    ]
    if task.method == "cellchat":
        return [
            str(settings.python),
            "-m",
            "benchmarks.adapters.cellchat.run_by_sample",
            *common,
            "--database-root",
            str(settings.database_root),
            *harmonized,
            "--min-cells",
            "10",
            "--nboot",
            "100",
            "--threads",
            "1",
            "--sample-jobs",
            str(settings.task_threads),
            "--environment",
            settings.cellchat_environment,
        ]
    if task.method == "liana":
        return [
            str(settings.liana_python),
            "-m",
            "benchmarks.adapters.liana.run_by_sample",
            *common,
            *harmonized,
            "--n-perms",
            "100",
            "--n-jobs",
            str(settings.task_threads),
            "--min-cells",
            "10",
        ]
    if task.target is None or task.reference is None:
        raise ValueError(f"scSeqCommDiff task lacks a contrast: {task.task_id}")
    return [
        str(settings.python),
        "-m",
        "benchmarks.adapters.scseqcommdiff.run",
        str(task.input_path),
        str(task.output_dir),
        "--input-manifest",
        str(task.input_manifest),
        "--resource",
        resource,
        "--resource-manifest",
        resource_manifest,
        "--resource-mode",
        "H-common",
        "--rscript",
        str(settings.scseq_rscript),
        "--python-executable",
        str(settings.python),
        "--dataset-id",
        task.dataset_id,
        "--scenario",
        "multi-sample",
        "--condition-key",
        "condition",
        "--sample-unit-key",
        "sample_id",
        "--target",
        task.target,
        "--reference",
        task.reference,
        "--cores",
        str(settings.task_threads),
        "--nrep",
        "1000",
        "--min-cells",
        "10",
    ]


def _peak_rss(process: psutil.Process) -> int:
    total = 0
    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.Error, OSError):
        return 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return total


def _run_task(
    task: PanelTask,
    fixture: dict[str, Any],
    settings: PanelSettings,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    if _completed_task(task, expected_commit=expected_commit):
        return {"task_id": task.task_id, "status": "resumed", "elapsed_seconds": 0.0}
    if task.output_dir.exists():
        raise FileExistsError(
            "incomplete output exists; inspect or remove it explicitly: "
            f"{task.output_dir}"
        )
    while psutil.virtual_memory().percent >= settings.memory_limit_percent:
        time.sleep(2.0)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(task, fixture, settings)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(settings.adapter_root),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = time.perf_counter()
    peak = 0
    with task.log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"task_id": task.task_id, "command": command}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=settings.adapter_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        tracked = psutil.Process(process.pid)
        while process.poll() is None:
            peak = max(peak, _peak_rss(tracked))
            time.sleep(0.5)
        peak = max(peak, _peak_rss(tracked))
    elapsed = time.perf_counter() - started
    if process.returncode != 0 or not _completed_task(
        task, expected_commit=expected_commit
    ):
        raise RuntimeError(
            f"adapter task failed validation: {task.task_id}; log={task.log_path}"
        )
    return {
        "task_id": task.task_id,
        "method": task.method,
        "dataset_id": task.dataset_id,
        "contrast": task.contrast,
        "status": "complete",
        "elapsed_seconds": elapsed,
        "peak_rss_mb": peak / (1024**2),
        "output_manifest": str(_task_manifest_path(task).resolve()),
        "output_manifest_sha256": sha256_file(_task_manifest_path(task)),
        "log": str(task.log_path.resolve()),
    }


def _link_crychic_runs(
    source: Path,
    output_dir: Path,
    *,
    expected_commit: str,
    expected_datasets: set[str],
) -> list[str]:
    linked: list[str] = []
    observed: set[str] = set()
    for run in sorted(path for path in source.iterdir() if path.is_dir()):
        manifest_path = run / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"CRYCHIC run is incomplete: {run}")
        code = manifest.get("code")
        if not isinstance(code, dict) or code.get("commit") != expected_commit:
            raise ValueError(f"CRYCHIC run code does not match fixture: {run}")
        dataset_id = str(manifest.get("dataset_id", ""))
        observed.add(dataset_id)
        destination = output_dir / run.name
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != run.resolve():
                raise FileExistsError(f"CRYCHIC link target differs: {destination}")
        else:
            destination.symlink_to(run.resolve(), target_is_directory=True)
        linked.append(destination.name)
    if observed != expected_datasets:
        raise ValueError(
            "CRYCHIC run dataset coverage disagrees with fixture: "
            f"missing={sorted(expected_datasets - observed)}, "
            f"extra={sorted(observed - expected_datasets)}"
        )
    return linked


def run_panel(
    fixture_dir: Path,
    output_dir: Path,
    crychic_runs: Path,
    *,
    settings: PanelSettings,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, Any]:
    """Execute or resume the external panel and link frozen CRYCHIC runs."""

    if not 1 <= settings.max_workers <= 32:
        raise ValueError("max_workers must lie in [1, 32]")
    if not 1 <= settings.task_threads <= 16:
        raise ValueError("task_threads must lie in [1, 16]")
    if not 20.0 <= settings.memory_limit_percent <= 80.0:
        raise ValueError("memory_limit_percent must lie in [20, 80]")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, fixture = build_tasks(fixture_dir, output_dir, methods=methods)
    code = fixture.get("code")
    if not isinstance(code, dict) or code.get("dirty") is not False:
        raise ValueError("fixture code provenance is not clean")
    expected_commit = str(code.get("commit", ""))
    adapter_code = _git_state(settings.adapter_root.resolve())
    if adapter_code != {"commit": expected_commit, "dirty": False}:
        raise ValueError(
            f"adapter worktree must be clean at {expected_commit}: {adapter_code}"
        )
    for executable in (
        settings.python,
        settings.liana_python,
        settings.scseq_rscript,
    ):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"required executable is unavailable: {executable}")
    if not settings.database_root.is_dir():
        raise FileNotFoundError(settings.database_root)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    progress_path = output_dir / "panel_progress.json"

    def record(result: dict[str, Any]) -> None:
        with lock:
            results.append(result)
            completed = len(results)
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(tasks) - completed) if completed else None
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "running" if completed < len(tasks) else "tasks_complete",
                "tasks_total": len(tasks),
                "tasks_finished": completed,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "memory_percent": psutil.virtual_memory().percent,
                "results": sorted(results, key=lambda row: str(row["task_id"])),
            }
            _write_json(progress_path, payload)
            print(
                f"[{completed}/{len(tasks)}] {result['task_id']} "
                f"{result['status']} elapsed={elapsed:.1f}s eta={eta or 0.0:.1f}s",
                flush=True,
            )

    futures: dict[Future[dict[str, Any]], PanelTask] = {}
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        for task in tasks:
            futures[
                executor.submit(
                    _run_task,
                    task,
                    fixture,
                    settings,
                    expected_commit=expected_commit,
                )
            ] = task
        for future in as_completed(futures):
            record(future.result())

    expected_datasets = {
        str(record["dataset_id"])
        for record in fixture["records"]
        if isinstance(record, dict)
    }
    links = _link_crychic_runs(
        crychic_runs.resolve(),
        output_dir,
        expected_commit=expected_commit,
        expected_datasets=expected_datasets,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "fixture_manifest": {
            "path": str((fixture_dir / "manifest.json").resolve()),
            "sha256": sha256_file(fixture_dir / "manifest.json"),
        },
        "adapter_code": adapter_code,
        "settings": {
            **asdict(settings),
            "adapter_root": str(settings.adapter_root.resolve()),
            "database_root": str(settings.database_root.resolve()),
            "python": str(settings.python.resolve()),
            "liana_python": str(settings.liana_python.resolve()),
            "scseq_rscript": str(settings.scseq_rscript.resolve()),
        },
        "methods": list(methods),
        "task_count": len(tasks),
        "crychic_links": links,
        "elapsed_seconds": time.perf_counter() - started,
        "results": sorted(results, key=lambda row: str(row["task_id"])),
    }
    _write_json(output_dir / "panel_manifest.json", manifest)
    return manifest


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = set(methods).difference(METHODS)
    if invalid or not methods:
        raise argparse.ArgumentTypeError(f"invalid methods: {sorted(invalid)}")
    return methods


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crychic-runs", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--liana-python", type=Path, required=True)
    parser.add_argument("--cellchat-environment", required=True)
    parser.add_argument("--scseq-rscript", type=Path, required=True)
    parser.add_argument("--methods", type=_parse_methods, default=METHODS)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--memory-limit-percent", type=float, default=70.0)
    parser.add_argument("--task-threads", type=int, default=8)
    return parser


def main() -> int:
    args = _parser().parse_args()
    settings = PanelSettings(
        adapter_root=args.adapter_root,
        database_root=args.database_root,
        python=args.python,
        liana_python=args.liana_python,
        cellchat_environment=args.cellchat_environment,
        scseq_rscript=args.scseq_rscript,
        max_workers=args.max_workers,
        memory_limit_percent=args.memory_limit_percent,
        task_threads=args.task_threads,
    )
    manifest = run_panel(
        args.fixture_dir,
        args.output_dir,
        args.crychic_runs,
        settings=settings,
        methods=args.methods,
    )
    print(json.dumps(json_safe(manifest), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
