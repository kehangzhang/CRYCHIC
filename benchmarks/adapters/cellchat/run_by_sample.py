"""Run CellChat independently for each biological sample."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from benchmarks.adapters.common import (
    begin_manifest,
    canonical_digest,
    cell_type_support,
    fail_manifest,
    finalize_manifest,
    load_harmonized_resource,
    materialize_fixed_universe,
    method_frozen_resource,
    prepare_output,
    sha256_file,
    validate_prepared_input,
)
from benchmarks.adapters.resource_tables import native_lr_resource

METHOD_ID = "cellchat"
R_SEED_MODULUS = 2_147_483_647
UINT32_MAX = 2**32 - 1
FUTURE_GLOBALS_MAX_SIZE_BYTES = 8 * 1024**3
PROCESS_TERMINATION_GRACE_SECONDS = 10.0
PROCESS_KILL_GRACE_SECONDS = 2.0
SINGLE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "RCPP_PARALLEL_NUM_THREADS": "1",
}
_NO_SIGNIFICANT_INTERACTIONS = (
    "No significant signaling interactions are inferred based on the input!"
)
_INFERENCE_COMPLETE_MARKER = "CellChat inference is done"
_SUBSET_ERROR_MARKER = "Error in subsetCommunication_internal"


@dataclass(frozen=True)
class _SampleRunResult:
    ordinal: int
    sample_id: str
    observed: pd.DataFrame | None
    status: str
    failure: dict[str, object] | None
    empty_result: bool
    reused_raw: bool
    raw_path: Path | None


@dataclass(frozen=True)
class _RawCacheContext:
    run_fingerprint: str
    run_spec_sha256: str


class _SampleExecutionCancelled(RuntimeError):
    """Stop a worker before it starts another external process."""


def _single_thread_environment(
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(SINGLE_THREAD_ENVIRONMENT)
    environment["R_FUTURE_PLAN"] = "sequential"
    if extra is not None:
        environment.update(extra)
    return environment


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    )
    _atomic_write_text(path, payload + "\n")


def _raw_sidecar_path(raw_path: Path) -> Path:
    return raw_path.with_suffix(f"{raw_path.suffix}.cache.json")


def _sample_cache_payload(
    *,
    raw_path: Path,
    ordinal: int,
    sample_id: str,
    effective_seed: int,
    cache_context: _RawCacheContext,
) -> dict[str, object]:
    return {
        "schema_version": "crychic-cellchat-raw-cache-v1",
        "run_fingerprint": cache_context.run_fingerprint,
        "run_spec_sha256": cache_context.run_spec_sha256,
        "ordinal": ordinal,
        "sample_id": sample_id,
        "effective_seed": effective_seed,
        "raw_filename": raw_path.name,
        "raw_sha256": sha256_file(raw_path),
        "raw_bytes": raw_path.stat().st_size,
    }


def _write_raw_sidecar(
    *,
    raw_path: Path,
    ordinal: int,
    sample_id: str,
    effective_seed: int,
    cache_context: _RawCacheContext,
) -> None:
    _atomic_write_json(
        _raw_sidecar_path(raw_path),
        _sample_cache_payload(
            raw_path=raw_path,
            ordinal=ordinal,
            sample_id=sample_id,
            effective_seed=effective_seed,
            cache_context=cache_context,
        ),
    )


def _validate_raw_sidecar(
    *,
    raw_path: Path,
    ordinal: int,
    sample_id: str,
    effective_seed: int,
    cache_context: _RawCacheContext,
) -> None:
    sidecar_path = _raw_sidecar_path(raw_path)
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"CellChat raw cache sidecar is unreadable: {error}"
        ) from error
    expected = _sample_cache_payload(
        raw_path=raw_path,
        ordinal=ordinal,
        sample_id=sample_id,
        effective_seed=effective_seed,
        cache_context=cache_context,
    )
    if sidecar != expected:
        raise ValueError("CellChat raw cache sidecar does not match this run")


def _signal_process_group(
    process: subprocess.Popen[str], signal_number: signal.Signals
) -> str | None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        return None
    except OSError as error:
        return f"killpg({process.pid}, {signal_number.name}) failed: {error}"
    return None


def _process_group_exists(process: subprocess.Popen[str]) -> bool:
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_groups(
    processes: Sequence[subprocess.Popen[str]],
    *,
    grace_seconds: float,
    kill_grace_seconds: float = PROCESS_KILL_GRACE_SECONDS,
) -> tuple[str, ...]:
    errors: list[str] = []
    owned = list({process.pid: process for process in processes}.values())
    for process in owned:
        if error := _signal_process_group(process, signal.SIGTERM):
            errors.append(error)

    deadline = time.monotonic() + grace_seconds
    while any(_process_group_exists(process) for process in owned):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))

    survivors = [process for process in owned if _process_group_exists(process)]
    for process in survivors:
        if error := _signal_process_group(process, signal.SIGKILL):
            errors.append(error)

    kill_deadline = time.monotonic() + kill_grace_seconds
    while any(_process_group_exists(process) for process in survivors):
        remaining = kill_deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))

    for process in owned:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.0, kill_deadline - time.monotonic()))
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"failed to reap process {process.pid}: {error}")
    for process in survivors:
        if _process_group_exists(process):
            errors.append(f"process group {process.pid} survived SIGKILL")
    return tuple(errors)


def _close_process_pipes(process: subprocess.Popen[str]) -> tuple[str, ...]:
    errors: list[str] = []
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is None or stream.closed:
            continue
        try:
            stream.close()
        except OSError as error:
            errors.append(f"failed to close process {process.pid} {name}: {error}")
    return tuple(errors)


def _add_cleanup_notes(error: BaseException, messages: Sequence[str]) -> None:
    for message in messages:
        error.add_note(f"CellChat cleanup: {message}")


class _ProcessRegistry:
    """Own sample process groups so interruption can terminate every descendant."""

    def __init__(
        self, *, grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS
    ) -> None:
        self._grace_seconds = grace_seconds
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[str]] = set()
        self._cancelled = threading.Event()

    def raise_if_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise _SampleExecutionCancelled("CellChat sample execution was cancelled")

    def register(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            cancelled = self._cancelled.is_set()
            if not cancelled:
                self._processes.add(process)
        if cancelled:
            cleanup_errors = _terminate_process_groups(
                [process], grace_seconds=self._grace_seconds
            )
            error = _SampleExecutionCancelled(
                "CellChat sample execution was cancelled"
            )
            _add_cleanup_notes(error, cleanup_errors)
            raise error

    def unregister(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.discard(process)

    def cancel(self) -> None:
        self._cancelled.set()

    def terminate(self, process: subprocess.Popen[str]) -> tuple[str, ...]:
        return _terminate_process_groups(
            [process], grace_seconds=self._grace_seconds
        )

    def terminate_all(self) -> tuple[str, ...]:
        self.cancel()
        with self._lock:
            processes = tuple(self._processes)
        return _terminate_process_groups(
            processes, grace_seconds=self._grace_seconds
        )


def _run_registered_process(
    command: Sequence[str],
    *,
    process_registry: _ProcessRegistry,
    timeout_seconds: float | None,
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_registry.raise_if_cancelled()
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_single_thread_environment(environment_overrides),
        start_new_session=True,
    )
    registered = False
    try:
        process_registry.register(process)
        registered = True
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + timeout_seconds
        )
        while True:
            process_registry.raise_if_cancelled()
            remaining = (
                None if deadline is None else deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                raise TimeoutError(
                    f"CellChat sample process exceeded {timeout_seconds} seconds"
                )
            try:
                stdout, stderr = process.communicate(
                    timeout=(
                        0.2 if remaining is None else min(0.2, remaining)
                    )
                )
            except subprocess.TimeoutExpired:
                continue
            break
        if process.returncode is None:
            raise RuntimeError("CellChat sample process exited without a return code")
        completed = subprocess.CompletedProcess(
            args=list(command),
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except BaseException as error:
        failure_cleanup_errors = [
            *process_registry.terminate(process),
            *_close_process_pipes(process),
        ]
        _add_cleanup_notes(error, failure_cleanup_errors)
        raise
    else:
        success_cleanup_errors = process_registry.terminate(process)
        if success_cleanup_errors:
            raise RuntimeError("; ".join(success_cleanup_errors))
        return completed
    finally:
        if registered and process.poll() is not None:
            process_registry.unregister(process)


def _r_query(environment: str, expression: str) -> str:
    completed = _run_registered_process(
        [
            "conda",
            "run",
            "-n",
            environment,
            "Rscript",
            "--vanilla",
            "-e",
            expression,
        ],
        process_registry=_ProcessRegistry(),
        timeout_seconds=None,
    )
    completed.check_returncode()
    return completed.stdout.strip()


def _environment(environment: str, threads: int) -> tuple[str, dict[str, object]]:
    fields = _r_query(
        environment,
        "packages <- c('CellChat', 'future', 'future.apply', 'parallelly', "
        "'Matrix'); versions <- vapply(packages, function(package) "
        "as.character(packageVersion(package)), character(1)); "
        "cat(paste(c(as.character(getRversion()), versions, "
        "paste(RNGkind(), collapse='|')), collapse='\\t'))",
    ).split("\t")
    if len(fields) != 7:
        raise RuntimeError("unexpected CellChat R environment query output")
    r_version, *package_versions, rng_kind = fields
    packages = dict(
        zip(
            ("CellChat", "future", "future.apply", "parallelly", "Matrix"),
            package_versions,
            strict=True,
        )
    )
    return packages["CellChat"], {
        "environment_name": environment,
        "runtime": "R",
        "r": r_version,
        "packages": packages,
        "rng_kind": rng_kind.split("|"),
        "future_rng_on_misuse": "error",
        "future_plan": "sequential" if threads == 1 else "multisession",
        "future_workers_per_sample": 0 if threads == 1 else threads,
        "configured_blas_openmp_threads_per_r_session": 1,
        "threads": threads,
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def _run_database(
    *,
    database_root: Path,
    resource_mode: str,
    custom_database_rds: Path | None,
) -> tuple[Path, dict[str, object]]:
    native = database_root / "cellchat/CellChatDB_human.rds"
    if custom_database_rds is None:
        if not native.is_file():
            raise FileNotFoundError(f"CellChat database is missing: {native}")
        return native, {
            "database_role": "native_CellChatDB",
            "database_filename": native.name,
            "database_sha256": sha256_file(native),
        }
    if resource_mode not in {"H-common", "H-covered"}:
        raise ValueError("custom_database_rds requires a harmonized resource mode")
    custom = custom_database_rds.expanduser().resolve()
    if not custom.is_file():
        raise FileNotFoundError(f"custom CellChat database is missing: {custom}")
    return custom, {
        "database_role": "custom_resource_matched_CellChatDB",
        "database_filename": custom.name,
        "database_sha256": sha256_file(custom),
    }


def _r_seed(seed: int, ordinal: int) -> int:
    """Map a requested uint32 seed into R's positive signed-integer range."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= UINT32_MAX
    ):
        raise ValueError("CellChat requested seed must be a uint32 integer")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("CellChat sample ordinal must be a non-negative integer")
    effective = seed + ordinal
    if 1 <= effective <= R_SEED_MODULUS:
        return effective
    return 1 + ((effective - 1) % R_SEED_MODULUS)


def _seed_provenance(sample_ids: list[str], requested_seed: int) -> dict[str, Any]:
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("CellChat sample IDs must be unique for seed assignment")
    ordered_sample_ids = sorted(sample_ids)
    effective = {
        sample_id: _r_seed(requested_seed, ordinal)
        for ordinal, sample_id in enumerate(ordered_sample_ids)
    }
    return {
        "requested_seed": requested_seed,
        "requested_seed_type": "uint32",
        "effective_seed_range": [1, R_SEED_MODULUS],
        "effective_sample_seeds": effective,
        "seed_scope": "cohort",
        "cohort_sample_set_fingerprint": canonical_digest(
            {"sample_ids": sorted(sample_ids)}, prefix="cellchat_cohort"
        ),
        "cohort_sample_count": len(sample_ids),
        "seed_sample_order": ordered_sample_ids,
        "cohort_membership_can_change_ordinal_seeds": True,
        "cohort_seed_stability": (
            "adding or removing a lexicographically preceding sample changes "
            "the effective seed of later samples"
        ),
        "sample_seed_mapping": (
            "preserve requested_seed + sample_ordinal when in 1..2147483647; "
            "otherwise 1 + ((effective - 1) modulo 2147483647)"
        ),
        "sample_ordinal_order": "lexicographically sorted sample_id",
    }


def _is_valid_empty_result(completed: subprocess.CompletedProcess[str]) -> bool:
    """Recognize CellChat's post-inference exception for a valid empty result."""

    return bool(
        completed.returncode != 0
        and _INFERENCE_COMPLETE_MARKER in completed.stdout
        and _SUBSET_ERROR_MARKER in completed.stderr
        and _NO_SIGNIFICANT_INTERACTIONS in completed.stderr
    )


def _expression(adata: ad.AnnData, layer: str | None) -> sparse.csr_matrix:
    values = adata.X if layer is None else adata.layers[layer]
    matrix = sparse.csr_matrix(values)
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or (matrix.data < 0).any()
    ):
        raise ValueError("CellChat input expression must be finite and non-negative")
    return matrix


def _write_sample(
    adata: ad.AnnData,
    directory: Path,
    *,
    cell_type_key: str,
    layer: str | None,
) -> None:
    directory.mkdir(parents=True)
    matrix = _expression(adata, layer).transpose().tocsr()
    mmwrite(directory / "matrix.mtx", matrix)
    (directory / "genes.txt").write_text(
        "\n".join(adata.var_names.astype(str)) + "\n", encoding="utf-8"
    )
    (directory / "cells.txt").write_text(
        "\n".join(adata.obs_names.astype(str)) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "cell": adata.obs_names.astype(str),
            "cell_type": adata.obs[cell_type_key].astype(str).to_numpy(),
        }
    ).to_csv(directory / "metadata.tsv", sep="\t", index=False)


def _validate_parallelism(
    *,
    threads: int,
    sample_jobs: int,
    sample_timeout_seconds: float | None = None,
) -> None:
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if (
        isinstance(sample_jobs, bool)
        or not isinstance(sample_jobs, int)
        or sample_jobs < 1
    ):
        raise ValueError("sample_jobs must be a positive integer")
    if sample_timeout_seconds is not None and (
        isinstance(sample_timeout_seconds, bool) or sample_timeout_seconds <= 0
    ):
        raise ValueError("sample_timeout_seconds must be positive when provided")


def _parallelism_provenance(
    *, sample_count: int, threads: int, sample_jobs: int
) -> dict[str, object]:
    concurrent_samples = min(sample_count, sample_jobs)
    future_workers_per_master = 0 if threads == 1 else threads
    future_workers = concurrent_samples * future_workers_per_master
    return {
        "threads": threads,
        "sample_jobs": sample_jobs,
        "maximum_concurrent_samples": concurrent_samples,
        "maximum_concurrent_r_masters": concurrent_samples,
        "future_workers_per_r_master": future_workers_per_master,
        "maximum_concurrent_future_workers": future_workers,
        "maximum_concurrent_r_sessions": concurrent_samples + future_workers,
        "configured_blas_openmp_threads_per_r_session": 1,
        "concurrency_accounting": (
            "configured upper bounds for R sessions; conda wrapper processes "
            "are excluded"
        ),
        "future_globals_max_size_bytes": FUTURE_GLOBALS_MAX_SIZE_BYTES,
        "future_globals_guard_semantics": (
            "per-future serialized-globals size guard; not an RSS or total-memory "
            "limit"
        ),
    }


def _raw_cache_run_spec(
    *,
    dataset_id: str,
    input_h5ad: Path,
    input_shape: tuple[int, int],
    sample_metadata: pd.DataFrame,
    input_keys: Mapping[str, object],
    resource_payload: Mapping[str, object],
    parameters: Mapping[str, object],
    environment_metadata: Mapping[str, object],
) -> dict[str, object]:
    adapter_path = Path(__file__).resolve()
    r_runner_path = adapter_path.with_name("run_sample.R")
    return {
        "schema_version": "crychic-cellchat-raw-cache-run-v1",
        "dataset_id": dataset_id,
        "input": {
            "sha256": sha256_file(input_h5ad),
            "bytes": input_h5ad.stat().st_size,
            "shape": list(input_shape),
            "keys": dict(input_keys),
            "sample_ids": sample_metadata["sample_id"].astype(str).tolist(),
        },
        "resource": dict(resource_payload),
        "parameters": dict(parameters),
        "environment": dict(environment_metadata),
        "code": {
            "python_adapter_filename": adapter_path.name,
            "python_adapter_sha256": sha256_file(adapter_path),
            "r_runner_filename": r_runner_path.name,
            "r_runner_sha256": sha256_file(r_runner_path),
        },
    }


def _write_cache_run_spec(
    raw_dir: Path, run_spec: Mapping[str, object]
) -> _RawCacheContext:
    run_spec_path = raw_dir / "cache_run_spec.json"
    _atomic_write_json(run_spec_path, run_spec)
    return _RawCacheContext(
        run_fingerprint=canonical_digest(run_spec, prefix="cellchat_raw_cache"),
        run_spec_sha256=sha256_file(run_spec_path),
    )


def _normalize_sample(
    table: pd.DataFrame,
    *,
    sample_id: str,
    source_map: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "sender",
                "receiver",
                "interaction_id",
                "target",
                "score",
                "specificity_score",
                "within_dataset_p_value",
                "within_dataset_p_value_semantics",
            ]
        )
    required = {"source", "target", "interaction_name", "prob", "pval"}
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"CellChat output lacks columns: {sorted(missing)}")
    result = table.rename(columns={"source": "sender", "target": "receiver"})
    result = result.merge(
        source_map[["source_interaction_id", "interaction_id"]],
        left_on="interaction_name",
        right_on="source_interaction_id",
        how="inner",
        validate="many_to_one",
    )
    result["score"] = pd.to_numeric(result["prob"], errors="coerce")
    result["within_dataset_p_value"] = pd.to_numeric(result["pval"], errors="coerce")
    result = result.loc[result["score"].notna()].copy()
    if result.empty:
        return _normalize_sample(
            pd.DataFrame(), sample_id=sample_id, source_map=source_map
        )
    grouped = result.groupby(
        ["sender", "receiver", "interaction_id"],
        sort=True,
        observed=True,
        as_index=False,
    ).agg(
        score=("score", "max"), within_dataset_p_value=("within_dataset_p_value", "min")
    )
    grouped["sample_id"] = sample_id
    grouped["target"] = pd.NA
    grouped["specificity_score"] = pd.NA
    grouped["within_dataset_p_value_semantics"] = (
        "CellChat within-sample bootstrap probability p-value; not a condition test"
    )
    return grouped


def _execute_sample(
    *,
    ordinal: int,
    sample_id: str,
    adata: ad.AnnData,
    sample_key: str,
    cell_type_key: str,
    layer: str | None,
    work_root: Path,
    raw_dir: Path,
    source_map: pd.DataFrame,
    run_database: Path,
    source_ids_path: Path,
    min_cells: int,
    nboot: int,
    effective_seed: int,
    trim: float,
    population_size: bool,
    threads: int,
    environment: str,
    resume_raw: bool,
    process_registry: _ProcessRegistry,
    sample_timeout_seconds: float | None,
    cache_context: _RawCacheContext,
) -> _SampleRunResult:
    process_registry.raise_if_cancelled()
    raw_path = raw_dir / f"{ordinal:04d}_{_safe_name(sample_id)}.csv"
    if resume_raw and raw_path.is_file():
        try:
            _validate_raw_sidecar(
                raw_path=raw_path,
                ordinal=ordinal,
                sample_id=sample_id,
                effective_seed=effective_seed,
                cache_context=cache_context,
            )
            raw = pd.read_csv(raw_path)
            normalized = _normalize_sample(
                raw, sample_id=sample_id, source_map=source_map
            )
        except (OSError, ValueError, pd.errors.ParserError, RuntimeError):
            raw_path.unlink(missing_ok=True)
            _raw_sidecar_path(raw_path).unlink(missing_ok=True)
        else:
            return _SampleRunResult(
                ordinal=ordinal,
                sample_id=sample_id,
                observed=normalized,
                status="complete",
                failure=None,
                empty_result=normalized.empty,
                reused_raw=True,
                raw_path=raw_path,
            )

    selected = adata[adata.obs[sample_key].astype(str) == sample_id].copy()
    sample_work = work_root / f"{ordinal:04d}_{_safe_name(sample_id)}"
    _write_sample(
        selected,
        sample_work,
        cell_type_key=cell_type_key,
        layer=layer,
    )
    command = [
        "conda",
        "run",
        "-n",
        environment,
        "Rscript",
        "--vanilla",
        str(Path(__file__).with_name("run_sample.R")),
        str(sample_work),
        str(run_database),
        str(source_ids_path),
        str(raw_path),
        str(min_cells),
        str(nboot),
        str(effective_seed),
        str(trim),
        str(population_size).upper(),
        str(threads),
    ]
    r_metadata_path = sample_work / "r_runtime_metadata.json"
    completed = _run_registered_process(
        command,
        process_registry=process_registry,
        timeout_seconds=sample_timeout_seconds,
        environment_overrides={
            "CRYCHIC_CELLCHAT_RUN_METADATA": str(r_metadata_path)
        },
    )
    if completed.returncode != 0:
        return _SampleRunResult(
            ordinal=ordinal,
            sample_id=sample_id,
            observed=None,
            status="method_failed",
            failure={
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            },
            empty_result=False,
            reused_raw=False,
            raw_path=None,
        )
    raw = pd.read_csv(raw_path)
    _write_raw_sidecar(
        raw_path=raw_path,
        ordinal=ordinal,
        sample_id=sample_id,
        effective_seed=effective_seed,
        cache_context=cache_context,
    )
    return _SampleRunResult(
        ordinal=ordinal,
        sample_id=sample_id,
        observed=_normalize_sample(raw, sample_id=sample_id, source_map=source_map),
        status="complete",
        failure=None,
        empty_result=raw.empty,
        reused_raw=False,
        raw_path=raw_path,
    )


def _cancel_sample_futures(
    futures: Sequence[Future[_SampleRunResult]],
    *,
    process_registry: _ProcessRegistry,
) -> tuple[str, ...]:
    process_registry.cancel()
    for future in futures:
        future.cancel()
    return process_registry.terminate_all()


def _run_sample_tasks(
    tasks: Sequence[dict[str, object]],
    *,
    sample_jobs: int,
    process_registry: _ProcessRegistry,
    execute_sample: Callable[..., _SampleRunResult] = _execute_sample,
) -> list[_SampleRunResult]:
    executor = ThreadPoolExecutor(max_workers=sample_jobs)
    pending: set[Future[_SampleRunResult]] = set()
    task_iterator = iter(tasks)
    results: list[_SampleRunResult] = []

    def submit_next() -> bool:
        try:
            task = next(task_iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(execute_sample, **task))
        return True

    try:
        for _ in range(min(sample_jobs, len(tasks))):
            submit_next()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed_batch = [future.result() for future in done]
            results.extend(completed_batch)
            for _ in completed_batch:
                submit_next()
    except BaseException as error:
        cleanup_errors: tuple[str, ...] = ()
        try:
            cleanup_errors = _cancel_sample_futures(
                tuple(pending), process_registry=process_registry
            )
        except BaseException as cleanup_error:
            error.add_note(
                "CellChat cleanup raised "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        finally:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as shutdown_error:
                error.add_note(
                    "CellChat executor shutdown raised "
                    f"{type(shutdown_error).__name__}: {shutdown_error}"
                )
        _add_cleanup_notes(error, cleanup_errors)
        raise
    executor.shutdown(wait=True, cancel_futures=False)
    return sorted(results, key=lambda result: result.ordinal)


def run(
    input_h5ad: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    database_root: Path,
    sample_key: str,
    subject_key: str,
    cell_type_key: str,
    context_keys: tuple[str, ...],
    resource_mode: str,
    harmonized_resource: Path | None,
    harmonized_manifest: Path | None,
    layer: str | None,
    min_cells: int,
    nboot: int,
    trim: float,
    population_size: bool,
    seed: int,
    threads: int,
    sample_jobs: int,
    environment: str,
    custom_database_rds: Path | None,
    overwrite: bool,
    resume_raw: bool,
    sample_timeout_seconds: float | None = None,
) -> dict[str, object]:
    """Execute CellChat sample by sample and materialize its frozen universe."""
    _validate_parallelism(
        threads=threads,
        sample_jobs=sample_jobs,
        sample_timeout_seconds=sample_timeout_seconds,
    )
    if overwrite and resume_raw:
        raise ValueError("overwrite and resume_raw are mutually exclusive")
    if resume_raw:
        output_dir = output_dir.expanduser().resolve()
        if not output_dir.is_dir() or not (output_dir / "raw").is_dir():
            raise FileNotFoundError(
                "resume_raw requires an existing adapter output with raw/"
            )
        if (output_dir / "interactions_long.parquet").exists():
            raise FileExistsError(
                "resume_raw refuses a finalized interactions_long.parquet"
            )
    else:
        output_dir = prepare_output(output_dir, overwrite=overwrite)
    repo_root = Path(__file__).resolve().parents[3]
    adata = ad.read_h5ad(input_h5ad)
    sample_metadata = validate_prepared_input(
        adata,
        sample_key=sample_key,
        subject_key=subject_key,
        cell_type_key=cell_type_key,
        context_keys=context_keys,
    )
    support = cell_type_support(
        adata, sample_key=sample_key, cell_type_key=cell_type_key
    )
    method_version, environment_metadata = _environment(environment, threads)
    run_database, database_provenance = _run_database(
        database_root=database_root,
        resource_mode=resource_mode,
        custom_database_rds=custom_database_rds,
    )

    native_resource, native_map, native_metadata = native_lr_resource(
        method="cellchat",
        database_root=database_root,
        repo_root=repo_root,
    )
    if resource_mode == "native":
        frozen_resource = native_resource
        source_map = native_map
        resource_id = native_metadata["resource_id"]
        resource_version = native_metadata["resource_version"]
        resource_payload: dict[str, object] = {
            "mode": resource_mode,
            **native_metadata,
            "interactions": len(frozen_resource),
        }
    else:
        if harmonized_resource is None or harmonized_manifest is None:
            raise ValueError("harmonized arms require resource table and manifest")
        harmonized, frozen_manifest = load_harmonized_resource(
            harmonized_resource, harmonized_manifest
        )
        frozen_resource = method_frozen_resource(
            harmonized, method="cellchat", resource_mode=resource_mode
        )
        source_map = frozen_resource.loc[
            frozen_resource["method_covered"],
            ["native_interaction_id", "interaction_id"],
        ].rename(columns={"native_interaction_id": "source_interaction_id"})
        resource_id = str(frozen_manifest["resource_id"])
        resource_version = str(frozen_manifest["version"])
        resource_payload = {
            "mode": resource_mode,
            "resource_id": resource_id,
            "version": resource_version,
            "payload_filename": harmonized_resource.name,
            "payload_sha256": sha256_file(harmonized_resource),
            "manifest_filename": harmonized_manifest.name,
            "manifest_sha256": sha256_file(harmonized_manifest),
            "interactions": len(frozen_resource),
            "method_covered": int(frozen_resource["method_covered"].sum()),
        }
    resource_payload.update(database_provenance)

    seed_provenance = _seed_provenance(
        sample_metadata["sample_id"].astype(str).tolist(), seed
    )
    effective_sample_seeds = cast(
        dict[str, int], seed_provenance["effective_sample_seeds"]
    )
    parameters = {
        "layer": layer,
        "min_cells": min_cells,
        "nboot": nboot,
        "trim": trim,
        "population_size": population_size,
        "seed": seed,
        **seed_provenance,
        **_parallelism_provenance(
            sample_count=len(sample_metadata),
            threads=threads,
            sample_jobs=sample_jobs,
        ),
        "sample_timeout_seconds": sample_timeout_seconds,
        "sample_level_execution": True,
        "resume_raw": resume_raw,
    }
    input_keys: dict[str, object] = {
        "sample": sample_key,
        "subject": subject_key,
        "cell_type": cell_type_key,
        "contexts": context_keys,
    }
    manifest = begin_manifest(
        repo_root=repo_root,
        dataset_id=dataset_id,
        method={
            "id": METHOD_ID,
            "name": "CellChat",
            "version": method_version,
            "license": "GPL-3",
            "entrypoint": "CellChat::computeCommunProb",
        },
        environment=environment_metadata,
        input_path=input_h5ad,
        input_shape=adata.shape,
        sample_metadata=sample_metadata,
        input_keys=input_keys,
        resource=resource_payload,
        parameters=parameters,
        score_semantics={
            "primary_score": "communication_probability",
            "direction": "higher_is_stronger",
            "within_dataset_p_value": (
                "CellChat bootstrap probability p-value; not between-condition "
                "inference"
            ),
        },
    )
    started = time.perf_counter()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=resume_raw)
    sample_status: dict[str, str] = {}
    sample_failures: dict[str, dict[str, object]] = {}
    sample_empty_results: list[str] = []
    resumed_raw_files: list[dict[str, object]] = []
    observed: list[pd.DataFrame] = []
    process_registry = _ProcessRegistry()
    try:
        cache_parameters = {
            key: value for key, value in parameters.items() if key != "resume_raw"
        }
        cache_run_spec = _raw_cache_run_spec(
            dataset_id=dataset_id,
            input_h5ad=input_h5ad,
            input_shape=adata.shape,
            sample_metadata=sample_metadata,
            input_keys=input_keys,
            resource_payload=resource_payload,
            parameters=cache_parameters,
            environment_metadata=environment_metadata,
        )
        cache_run_fingerprint = canonical_digest(
            cache_run_spec, prefix="cellchat_raw_cache"
        )
        cache_run_spec_path = raw_dir / "cache_run_spec.json"
        if resume_raw:
            try:
                existing_cache_spec = json.loads(
                    cache_run_spec_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    "resume_raw requires a valid raw/cache_run_spec.json"
                ) from error
            existing_fingerprint = canonical_digest(
                existing_cache_spec, prefix="cellchat_raw_cache"
            )
            if existing_fingerprint != cache_run_fingerprint:
                raise ValueError("resume_raw cache run spec is incompatible")
            cache_context = _RawCacheContext(
                run_fingerprint=existing_fingerprint,
                run_spec_sha256=sha256_file(cache_run_spec_path),
            )
        else:
            cache_context = _write_cache_run_spec(raw_dir, cache_run_spec)
        manifest["raw_cache"] = {
            "schema_version": "crychic-cellchat-raw-cache-run-v1",
            "run_spec_filename": cache_run_spec_path.name,
            "run_spec_sha256": cache_context.run_spec_sha256,
            "run_fingerprint": cache_context.run_fingerprint,
        }
        with tempfile.TemporaryDirectory(prefix="crychic_cellchat_") as temporary:
            work_root = Path(temporary)
            source_ids_path = work_root / "source_ids.txt"
            source_ids_path.write_text(
                "\n".join(source_map["source_interaction_id"].dropna().astype(str))
                + "\n",
                encoding="utf-8",
            )
            tasks = [
                {
                    "ordinal": ordinal,
                    "sample_id": str(sample_id),
                    "adata": adata,
                    "sample_key": sample_key,
                    "cell_type_key": cell_type_key,
                    "layer": layer,
                    "work_root": work_root,
                    "raw_dir": raw_dir,
                    "source_map": source_map,
                    "run_database": run_database,
                    "source_ids_path": source_ids_path,
                    "min_cells": min_cells,
                    "nboot": nboot,
                    "effective_seed": effective_sample_seeds[str(sample_id)],
                    "trim": trim,
                    "population_size": population_size,
                    "threads": threads,
                    "environment": environment,
                    "resume_raw": resume_raw,
                    "process_registry": process_registry,
                    "sample_timeout_seconds": sample_timeout_seconds,
                    "cache_context": cache_context,
                }
                for ordinal, sample_id in enumerate(sample_metadata["sample_id"])
            ]
            results = _run_sample_tasks(
                tasks,
                sample_jobs=sample_jobs,
                process_registry=process_registry,
            )
            for result in results:
                if result.status == "method_failed":
                    sample_status[result.sample_id] = result.status
                    if result.failure is not None:
                        sample_failures[result.sample_id] = result.failure
                    continue
                if result.observed is not None:
                    observed.append(result.observed)
                if result.empty_result:
                    sample_empty_results.append(result.sample_id)
                if result.reused_raw and result.raw_path is not None:
                    resumed_raw_files.append(
                        {
                            "filename": result.raw_path.name,
                            "sha256": sha256_file(result.raw_path),
                        }
                    )
        sparse_observed = (
            pd.concat(observed, ignore_index=True)
            if observed
            else _normalize_sample(pd.DataFrame(), sample_id="", source_map=source_map)
        )
        table = materialize_fixed_universe(
            sparse_observed,
            sample_metadata=sample_metadata,
            support=support,
            resource=frozen_resource,
            dataset_id=dataset_id,
            run_id=str(manifest["run_id"]),
            method_id=METHOD_ID,
            method_version=method_version,
            analysis_track="lr_stlr",
            resource_mode=resource_mode,
            resource_id=resource_id,
            resource_version=resource_version,
            score_name="communication_probability",
            score_direction="higher",
            specificity_score_name=None,
            min_cells=min_cells,
            sample_status=sample_status,
        )
        manifest["sample_failures"] = sample_failures
        manifest["sample_empty_results"] = sorted(sample_empty_results)
        manifest["resumed_raw_files"] = sorted(
            resumed_raw_files, key=lambda record: str(record["filename"])
        )
        return cast(
            dict[str, object],
            finalize_manifest(manifest, table, output_dir, started=started),
        )
    except BaseException as exc:
        _add_cleanup_notes(exc, process_registry.terminate_all())
        fail_manifest(manifest, output_dir, exc, started=started)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--database-root", required=True, type=Path)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--subject-key", default="subject_id")
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--context-key", action="append", required=True)
    parser.add_argument(
        "--resource-mode", choices=("H-common", "H-covered", "native"), default="native"
    )
    parser.add_argument("--harmonized-resource", type=Path)
    parser.add_argument("--harmonized-manifest", type=Path)
    parser.add_argument(
        "--custom-database-rds",
        type=Path,
        help=("resource-matched CellChatDB RDS; valid only for a harmonized arm"),
    )
    parser.add_argument("--layer")
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=100)
    parser.add_argument("--trim", type=float, default=0.1)
    parser.add_argument("--population-size", action="store_true")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--sample-jobs",
        type=int,
        default=1,
        help="number of biological samples evaluated concurrently",
    )
    parser.add_argument(
        "--sample-timeout-seconds",
        type=float,
        help="optional wall-time limit for one biological sample",
    )
    parser.add_argument("--environment", default="r_cellchat")
    parser.add_argument(
        "--resume-raw",
        action="store_true",
        help="reuse validated raw CSVs from an interrupted compatible run",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.input_h5ad,
        args.output_dir,
        dataset_id=args.dataset_id,
        database_root=args.database_root,
        sample_key=args.sample_key,
        subject_key=args.subject_key,
        cell_type_key=args.cell_type_key,
        context_keys=tuple(args.context_key),
        resource_mode=args.resource_mode,
        harmonized_resource=args.harmonized_resource,
        harmonized_manifest=args.harmonized_manifest,
        layer=args.layer,
        min_cells=args.min_cells,
        nboot=args.nboot,
        trim=args.trim,
        population_size=args.population_size,
        seed=args.seed,
        threads=args.threads,
        sample_jobs=args.sample_jobs,
        environment=args.environment,
        custom_database_rds=args.custom_database_rds,
        overwrite=args.overwrite,
        resume_raw=args.resume_raw,
        sample_timeout_seconds=args.sample_timeout_seconds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
