from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.cellchat.run_by_sample import (
    SINGLE_THREAD_ENVIRONMENT,
    _execute_sample,
    _is_valid_empty_result,
    _parallelism_provenance,
    _ProcessRegistry,
    _r_seed,
    _RawCacheContext,
    _run_database,
    _run_registered_process,
    _run_sample_tasks,
    _SampleRunResult,
    _seed_provenance,
    _validate_parallelism,
    _validate_raw_sidecar,
    _write_raw_sidecar,
)
from benchmarks.adapters.cellchat.run_by_sample import (
    _normalize_sample as normalize_cellchat,
)
from benchmarks.adapters.cellphonedb.run_by_sample import (
    _filtered_database,
)
from benchmarks.adapters.cellphonedb.run_by_sample import (
    _normalize_sample as normalize_cellphonedb,
)
from benchmarks.adapters.common import (
    LONG_TABLE_COLUMNS,
    materialize_fixed_universe,
    method_frozen_resource,
    sha256_file,
    validate_long_table,
)
from benchmarks.adapters.liana.run_by_sample import _normalize as normalize_liana
from benchmarks.adapters.nichenet.run_by_sample import (
    _expression,
    _prior_operator,
    _score_samples,
)
from scipy import sparse


def _sample_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "subject_id": ["p1", "p2"],
            "context_json": ['{"condition":"A"}', '{"condition":"B"}'],
        }
    )


def _support() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "cell_type": ["A", "B", "A", "B"],
            "n_cells": [20, 20, 20, 2],
        }
    )


def _resource() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interaction_id": ["i1", "i2"],
            "native_interaction_id": ["n1", pd.NA],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "method_covered": [True, False],
        }
    )


def test_cellchat_seed_is_mapped_into_r_integer_range() -> None:
    assert _r_seed(20260712, 0) == 20260712
    assert _r_seed(4_285_072_227, 0) == 2_137_588_580
    assert _r_seed(2_147_483_646, 0) == 2_147_483_646
    assert _r_seed(2_147_483_647, 0) == 2_147_483_647
    assert _r_seed(2_147_483_647, 1) == 1
    assert _r_seed(0, 0) == 2_147_483_647
    with pytest.raises(ValueError, match="uint32"):
        _r_seed(-1, 0)
    with pytest.raises(ValueError, match="uint32"):
        _r_seed(2**32, 0)


def test_cellchat_custom_database_is_checksum_bound_to_harmonized_arm(
    tmp_path: Path,
) -> None:
    database_root = tmp_path / "databases"
    native = database_root / "cellchat" / "CellChatDB_human.rds"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"native")
    custom = tmp_path / "connectomedb.rds"
    custom.write_bytes(b"custom")

    selected, provenance = _run_database(
        database_root=database_root,
        resource_mode="H-common",
        custom_database_rds=custom,
    )

    assert selected == custom.resolve()
    assert provenance["database_role"] == "custom_resource_matched_CellChatDB"
    assert provenance["database_sha256"] != sha256_file(native)
    with pytest.raises(ValueError, match="harmonized"):
        _run_database(
            database_root=database_root,
            resource_mode="native",
            custom_database_rds=custom,
        )


def test_cellchat_seed_provenance_records_requested_and_effective_values() -> None:
    provenance = _seed_provenance(["s1", "s2"], 4_285_072_227)

    assert provenance["requested_seed"] == 4_285_072_227
    assert provenance["requested_seed_type"] == "uint32"
    assert provenance["effective_seed_range"] == [1, 2_147_483_647]
    assert provenance["effective_sample_seeds"] == {
        "s1": 2_137_588_580,
        "s2": 2_137_588_581,
    }
    assert provenance["seed_scope"] == "cohort"
    assert provenance["cohort_sample_count"] == 2
    assert str(provenance["cohort_sample_set_fingerprint"]).startswith(
        "cellchat_cohort_"
    )


def test_cellchat_seed_fingerprint_is_set_stable_but_seeds_remain_ordinal() -> None:
    first = _seed_provenance(["s1", "s2"], 17)
    reordered = _seed_provenance(["s2", "s1"], 17)
    expanded = _seed_provenance(["a0", "s1", "s2"], 17)

    assert (
        first["cohort_sample_set_fingerprint"]
        == reordered["cohort_sample_set_fingerprint"]
    )
    assert first["effective_sample_seeds"] == reordered["effective_sample_seeds"]
    assert (
        first["effective_sample_seeds"]["s1"]
        != expanded["effective_sample_seeds"]["s1"]
    )
    assert first["seed_sample_order"] == ["s1", "s2"]


def test_cellchat_distinguishes_valid_empty_result_from_method_failure() -> None:
    valid_empty = subprocess.CompletedProcess(
        args=["Rscript"],
        returncode=1,
        stdout="CellChat inference is done. Parameter values are stored.\n",
        stderr=(
            "Error in subsetCommunication_internal(...): "
            "No significant signaling interactions are inferred based on the input!\n"
        ),
    )
    unrelated_failure = subprocess.CompletedProcess(
        args=["Rscript"],
        returncode=1,
        stdout=valid_empty.stdout,
        stderr="Error in computeCommunProb: numerical failure\n",
    )
    premature_subset_failure = subprocess.CompletedProcess(
        args=["Rscript"],
        returncode=1,
        stdout="",
        stderr=valid_empty.stderr,
    )

    assert _is_valid_empty_result(valid_empty)
    assert not _is_valid_empty_result(unrelated_failure)
    assert not _is_valid_empty_result(premature_subset_failure)


def test_cellchat_parallelism_requires_positive_worker_counts() -> None:
    _validate_parallelism(threads=4, sample_jobs=3)
    with pytest.raises(ValueError, match="threads"):
        _validate_parallelism(threads=0, sample_jobs=1)
    with pytest.raises(ValueError, match="sample_jobs"):
        _validate_parallelism(threads=1, sample_jobs=0)
    with pytest.raises(ValueError, match="sample_timeout_seconds"):
        _validate_parallelism(
            threads=1, sample_jobs=1, sample_timeout_seconds=0
        )


def test_cellchat_parallelism_manifest_counts_r_sessions_truthfully() -> None:
    parallel = _parallelism_provenance(
        sample_count=3, threads=4, sample_jobs=8
    )
    sequential = _parallelism_provenance(
        sample_count=3, threads=1, sample_jobs=2
    )

    assert parallel["maximum_concurrent_r_masters"] == 3
    assert parallel["maximum_concurrent_future_workers"] == 12
    assert parallel["maximum_concurrent_r_sessions"] == 15
    assert sequential["maximum_concurrent_r_masters"] == 2
    assert sequential["maximum_concurrent_future_workers"] == 0
    assert sequential["maximum_concurrent_r_sessions"] == 2
    assert parallel["configured_blas_openmp_threads_per_r_session"] == 1
    assert "not an RSS" in str(parallel["future_globals_guard_semantics"])


def test_cellchat_resume_reuses_a_valid_checksum_ready_raw_csv(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "0000_s1.csv"
    pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "interaction_name": ["native1"],
            "prob": [0.7],
            "pval": [0.1],
        }
    ).to_csv(raw_path, index=False)
    cache_context = _RawCacheContext(
        run_fingerprint="cellchat_raw_cache_test",
        run_spec_sha256="a" * 64,
    )
    _write_raw_sidecar(
        raw_path=raw_path,
        ordinal=0,
        sample_id="s1",
        effective_seed=42,
        cache_context=cache_context,
    )

    result = _execute_sample(
        ordinal=0,
        sample_id="s1",
        adata=ad.AnnData(),
        sample_key="sample_id",
        cell_type_key="cell_type",
        layer=None,
        work_root=tmp_path / "work",
        raw_dir=raw_dir,
        source_map=pd.DataFrame(
            {
                "source_interaction_id": ["native1"],
                "interaction_id": ["edge1"],
            }
        ),
        run_database=tmp_path / "database.rds",
        source_ids_path=tmp_path / "source_ids.txt",
        min_cells=10,
        nboot=100,
        effective_seed=42,
        trim=0.1,
        population_size=False,
        threads=2,
        environment="unused",
        resume_raw=True,
        process_registry=_ProcessRegistry(grace_seconds=0.01),
        sample_timeout_seconds=None,
        cache_context=cache_context,
    )

    assert result.status == "complete"
    assert result.reused_raw is True
    assert result.raw_path == raw_path
    assert result.observed is not None
    assert result.observed.loc[0, "score"] == pytest.approx(0.7)


def test_cellchat_resume_rejects_schema_valid_raw_with_wrong_run_fingerprint(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "0000_s1.csv"
    pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "interaction_name": ["native1"],
            "prob": [0.7],
            "pval": [0.1],
        }
    ).to_csv(raw_path, index=False)
    original_context = _RawCacheContext(
        run_fingerprint="cellchat_raw_cache_old",
        run_spec_sha256="a" * 64,
    )
    _write_raw_sidecar(
        raw_path=raw_path,
        ordinal=0,
        sample_id="s1",
        effective_seed=42,
        cache_context=original_context,
    )

    with pytest.raises(KeyError, match="sample_id"):
        _execute_sample(
            ordinal=0,
            sample_id="s1",
            adata=ad.AnnData(),
            sample_key="sample_id",
            cell_type_key="cell_type",
            layer=None,
            work_root=tmp_path / "work",
            raw_dir=raw_dir,
            source_map=pd.DataFrame(
                {
                    "source_interaction_id": ["native1"],
                    "interaction_id": ["edge1"],
                }
            ),
            run_database=tmp_path / "database.rds",
            source_ids_path=tmp_path / "source_ids.txt",
            min_cells=10,
            nboot=100,
            effective_seed=42,
            trim=0.1,
            population_size=False,
            threads=2,
            environment="unused",
            resume_raw=True,
            process_registry=_ProcessRegistry(grace_seconds=0.01),
            sample_timeout_seconds=None,
            cache_context=_RawCacheContext(
                run_fingerprint="cellchat_raw_cache_new",
                run_spec_sha256="b" * 64,
            ),
        )

    assert not raw_path.exists()
    assert not raw_path.with_suffix(".csv.cache.json").exists()


def test_cellchat_resume_rejects_raw_modified_after_sidecar(tmp_path: Path) -> None:
    raw_path = tmp_path / "0000_s1.csv"
    raw_path.write_text("source,target,interaction_name,prob,pval\n", encoding="utf-8")
    cache_context = _RawCacheContext(
        run_fingerprint="cellchat_raw_cache_test",
        run_spec_sha256="a" * 64,
    )
    _write_raw_sidecar(
        raw_path=raw_path,
        ordinal=0,
        sample_id="s1",
        effective_seed=42,
        cache_context=cache_context,
    )
    raw_path.write_text(
        "source,target,interaction_name,prob,pval\nA,B,native1,0.5,0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        _validate_raw_sidecar(
            raw_path=raw_path,
            ordinal=0,
            sample_id="s1",
            effective_seed=42,
            cache_context=cache_context,
        )


def test_cellchat_registered_process_uses_own_group_and_single_thread_env(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "process.txt"
    script = (
        "import os,sys; from pathlib import Path; "
        "Path(sys.argv[1]).write_text("
        "f'{os.getpid()}|{os.getpgrp()}|{os.environ[\"OMP_NUM_THREADS\"]}|' "
        "f'{os.environ[\"RCPP_PARALLEL_NUM_THREADS\"]}')"
    )

    completed = _run_registered_process(
        [sys.executable, "-c", script, str(payload)],
        process_registry=_ProcessRegistry(grace_seconds=0.1),
        timeout_seconds=5,
    )

    process_id, group_id, omp_threads, rcpp_threads = payload.read_text().split("|")
    assert completed.returncode == 0
    assert process_id == group_id
    assert omp_threads == "1"
    assert rcpp_threads == "1"
    assert set(SINGLE_THREAD_ENVIRONMENT.values()) == {"1"}


def test_cellchat_registered_process_timeout_kills_process_group(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = (
        "import signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "child=subprocess.Popen([sys.executable, '-c', "
        "'import signal,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);'"
        "'time.sleep(60)']); "
        "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )

    with pytest.raises(TimeoutError, match="exceeded"):
        _run_registered_process(
            [sys.executable, "-c", script, str(child_pid_path)],
            process_registry=_ProcessRegistry(grace_seconds=0.05),
            timeout_seconds=1,
        )

    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 3
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


def test_cellchat_timeout_does_not_wait_for_detached_pipe_holder(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "detached.pid"
    script = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], "
        "start_new_session=True); Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="exceeded"):
        _run_registered_process(
            [sys.executable, "-c", script, str(child_pid_path)],
            process_registry=_ProcessRegistry(grace_seconds=0.05),
            timeout_seconds=1,
        )

    assert time.monotonic() - started < 5
    detached_pid = int(child_pid_path.read_text())
    try:
        os.killpg(detached_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_cellchat_task_exception_cancels_pending_and_running_work() -> None:
    registry = _ProcessRegistry(grace_seconds=0.01)
    running = threading.Event()
    released = threading.Event()
    started: list[int] = []

    def execute(*, ordinal: int) -> _SampleRunResult:
        started.append(ordinal)
        if ordinal == 0:
            running.set()
            while True:
                registry.raise_if_cancelled()
                time.sleep(0.001)
        if ordinal == 1:
            assert running.wait(timeout=1)
            raise RuntimeError("injected sample failure")
        released.set()
        raise AssertionError("pending sample should have been cancelled")

    with pytest.raises(RuntimeError, match="injected sample failure"):
        _run_sample_tasks(
            [{"ordinal": ordinal} for ordinal in range(3)],
            sample_jobs=2,
            process_registry=registry,
            execute_sample=execute,
        )

    assert released.is_set() is False
    assert 2 not in started


def test_cellchat_keyboard_interrupt_terminates_registered_process_group(
    tmp_path: Path,
) -> None:
    registry = _ProcessRegistry(grace_seconds=0.05)
    process_pid_path = tmp_path / "process.pid"
    running = threading.Event()

    def execute(*, ordinal: int) -> _SampleRunResult:
        if ordinal == 0:
            script = (
                "import os,signal,sys,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
            )
            running.set()
            _run_registered_process(
                [sys.executable, "-c", script, str(process_pid_path)],
                process_registry=registry,
                timeout_seconds=None,
            )
        else:
            assert running.wait(timeout=1)
            deadline = time.monotonic() + 1
            while not process_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.001)
            raise KeyboardInterrupt
        raise AssertionError("cancelled process should not return normally")

    with pytest.raises(KeyboardInterrupt):
        _run_sample_tasks(
            [{"ordinal": 0}, {"ordinal": 1}],
            sample_jobs=2,
            process_registry=registry,
            execute_sample=execute,
        )

    process_pid = int(process_pid_path.read_text())
    assert not Path(f"/proc/{process_pid}").exists()


def test_cellchat_task_results_are_sorted_by_sample_ordinal() -> None:
    registry = _ProcessRegistry(grace_seconds=0.01)

    def execute(*, ordinal: int, delay: float) -> _SampleRunResult:
        time.sleep(delay)
        return _SampleRunResult(
            ordinal=ordinal,
            sample_id=f"s{ordinal}",
            observed=None,
            status="complete",
            failure=None,
            empty_result=True,
            reused_raw=False,
            raw_path=None,
        )

    results = _run_sample_tasks(
        [
            {"ordinal": 0, "delay": 0.03},
            {"ordinal": 1, "delay": 0.01},
            {"ordinal": 2, "delay": 0.0},
        ],
        sample_jobs=3,
        process_registry=registry,
        execute_sample=execute,
    )

    assert [result.ordinal for result in results] == [0, 1, 2]


def test_cellchat_r_runner_uses_local_future_plan_cleanup_and_rng_guard() -> None:
    script = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "adapters"
        / "cellchat"
        / "run_sample.R"
    ).read_text(encoding="utf-8")

    assert "main <- function(arguments)" in script
    assert 'Sys.setenv(R_FUTURE_PLAN = "sequential")' in script
    assert 'options(future.plan = "sequential")' in script
    assert "future::plan(future::sequential)" in script
    assert "on.exit({" in script
    assert 'future.rng.onMisuse = "error"' in script


def _assert_cellchat_r_future_cleanup_on_error(
    tmp_path: Path, *, threads: int
) -> None:
    input_dir = tmp_path / "missing-input"
    metadata_path = tmp_path / "runtime.json"
    configured_path = Path(f"{metadata_path}.configured")
    script_path = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "adapters"
        / "cellchat"
        / "run_sample.R"
    )
    command = [
        "conda",
        "run",
        "-n",
        "r_cellchat",
        "Rscript",
        "--vanilla",
        str(script_path),
        str(input_dir),
        str(tmp_path / "database.rds"),
        str(tmp_path / "source_ids.txt"),
        str(tmp_path / "output.csv"),
        "10",
        "1",
        "42",
        "0.1",
        "FALSE",
        str(threads),
    ]
    environment = os.environ.copy()
    environment.update(SINGLE_THREAD_ENVIRONMENT)
    environment.update(
        {
            "R_FUTURE_PLAN": "multisession",
            "CRYCHIC_CELLCHAT_RUN_METADATA": str(metadata_path),
        }
    )

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=30,
    )

    assert completed.returncode != 0
    configured = json.loads(configured_path.read_text(encoding="utf-8"))
    exited = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert configured["future_workers"] == threads
    assert exited["stage"] == "exit"
    assert exited["future_workers"] == 1
    assert exited["future_plan"] == "FutureStrategy"


def test_cellchat_r_sequential_cleanup_runs_on_error(tmp_path: Path) -> None:
    _assert_cellchat_r_future_cleanup_on_error(tmp_path, threads=1)


def test_cellchat_r_multisession_cleanup_runs_on_error(tmp_path: Path) -> None:
    _assert_cellchat_r_future_cleanup_on_error(tmp_path, threads=2)


def test_fixed_universe_materializes_absence_and_support_states() -> None:
    observed = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "sender": ["A"],
            "receiver": ["B"],
            "interaction_id": ["i1"],
            "target": [pd.NA],
            "score": [0.8],
        }
    )
    table = materialize_fixed_universe(
        observed,
        sample_metadata=_sample_metadata(),
        support=_support(),
        resource=_resource(),
        dataset_id="toy",
        run_id="run",
        method_id="method",
        method_version="1",
        analysis_track="lr_stlr",
        resource_mode="H-covered",
        resource_id="resource",
        resource_version="1",
        score_name="native",
        score_direction="higher",
        specificity_score_name=None,
        min_cells=10,
    )

    assert tuple(table.columns) == LONG_TABLE_COLUMNS
    assert table.groupby("sample_id").size().to_dict() == {"s1": 8, "s2": 8}
    assert table["universe_size"].unique().tolist() == [8]
    assert table["universe_id"].nunique() == 1
    status = table.groupby(["sample_id", "status"]).size().to_dict()
    assert status[("s1", "ok")] == 1
    assert status[("s1", "not_returned")] == 3
    assert status[("s1", "resource_unavailable")] == 4
    assert status[("s2", "not_returned")] == 1
    assert status[("s2", "insufficient_cells")] == 3
    assert status[("s2", "resource_unavailable")] == 4
    assert table.loc[~table["status"].eq("ok"), "score"].isna().all()


def test_target_program_universe_uses_one_source_agnostic_sender() -> None:
    observed = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "sender": ["__source_agnostic__"],
            "receiver": ["B"],
            "interaction_id": ["i1"],
            "target": [pd.NA],
            "score": [0.8],
        }
    )
    table = materialize_fixed_universe(
        observed,
        sample_metadata=_sample_metadata(),
        support=_support(),
        resource=_resource(),
        dataset_id="toy",
        run_id="run",
        method_id="target_method",
        method_version="1",
        analysis_track="ligand_target_program",
        resource_mode="native",
        resource_id="resource",
        resource_version="1",
        score_name="target_activity",
        score_direction="higher",
        specificity_score_name=None,
        min_cells=10,
        sender_types=("__source_agnostic__",),
        sender_requires_cells=False,
        interaction_direction="ligand_to_target_program",
    )

    assert table.groupby("sample_id").size().to_dict() == {"s1": 4, "s2": 4}
    assert set(table["sender"]) == {"__source_agnostic__"}
    assert set(table["interaction_direction"]) == {"ligand_to_target_program"}
    receiver_status = table.loc[
        (table["sample_id"] == "s2") & (table["receiver"] == "B"),
        ["interaction_id", "status"],
    ].set_index("interaction_id")["status"]
    assert receiver_status.to_dict() == {
        "i1": "insufficient_cells",
        "i2": "resource_unavailable",
    }


def test_validate_long_table_rejects_score_on_absent_result() -> None:
    observed = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "sender": ["A"],
            "receiver": ["B"],
            "interaction_id": ["i1"],
            "target": [pd.NA],
            "score": [0.8],
        }
    )
    table = materialize_fixed_universe(
        observed,
        sample_metadata=_sample_metadata(),
        support=_support(),
        resource=_resource(),
        dataset_id="toy",
        run_id="run",
        method_id="method",
        method_version="1",
        analysis_track="lr_stlr",
        resource_mode="H-covered",
        resource_id="resource",
        resource_version="1",
        score_name="native",
        score_direction="higher",
        specificity_score_name=None,
        min_cells=10,
    )
    index = table.index[table["status"].eq("not_returned")][0]
    table.loc[index, "score"] = 0.0
    with pytest.raises(ValueError, match="non-ok statuses"):
        validate_long_table(table)


def test_validate_long_table_rejects_sample_specific_universe() -> None:
    table = materialize_fixed_universe(
        pd.DataFrame(
            columns=["sample_id", "sender", "receiver", "interaction_id", "target"]
        ),
        sample_metadata=_sample_metadata(),
        support=_support(),
        resource=_resource(),
        dataset_id="toy",
        run_id="run",
        method_id="method",
        method_version="1",
        analysis_track="lr_stlr",
        resource_mode="H-covered",
        resource_id="resource",
        resource_version="1",
        score_name="native",
        score_direction="higher",
        specificity_score_name=None,
        min_cells=10,
    )
    index = table.index[table["sample_id"].eq("s2")][0]
    table.loc[index, "interaction_id"] = "sample_specific_edge"
    with pytest.raises(ValueError, match="same frozen edge universe"):
        validate_long_table(table)


def test_h_covered_resource_retains_method_specific_coverage() -> None:
    harmonized = pd.DataFrame(
        {
            "harmonized_interaction_id": ["i1", "i2"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "cellchat_source_interaction_id": ["cc1", ""],
            "cellchat_covered": ["true", "false"],
        }
    )
    covered = method_frozen_resource(
        harmonized, method="cellchat", resource_mode="H-covered"
    )
    assert covered["method_covered"].tolist() == [True, False]
    with pytest.raises(ValueError, match="H-common"):
        method_frozen_resource(harmonized, method="cellchat", resource_mode="H-common")


def test_cellchat_normalizer_maps_and_collapses_native_duplicates() -> None:
    native = pd.DataFrame(
        {
            "source": ["A", "A"],
            "target": ["B", "B"],
            "interaction_name": ["native1", "native2"],
            "prob": [0.2, 0.7],
            "pval": [0.4, 0.1],
        }
    )
    source_map = pd.DataFrame(
        {
            "source_interaction_id": ["native1", "native2"],
            "interaction_id": ["edge", "edge"],
        }
    )
    result = normalize_cellchat(native, sample_id="s1", source_map=source_map)
    assert len(result) == 1
    assert result.loc[0, "score"] == pytest.approx(0.7)
    assert result.loc[0, "within_dataset_p_value"] == pytest.approx(0.1)


def test_cellphonedb_normalizer_preserves_score_and_cell_label_pvalue() -> None:
    identifiers = {
        "id_cp_interaction": ["native1"],
        "interacting_pair": ["L_R"],
        "A|B": [7.5],
    }
    native = {
        "interaction_scores": pd.DataFrame(identifiers),
        "pvalues": pd.DataFrame({**identifiers, "A|B": [0.03]}),
    }
    source_map = pd.DataFrame(
        {"source_interaction_id": ["native1"], "interaction_id": ["edge"]}
    )
    result = normalize_cellphonedb(
        native,
        sample_id="s1",
        source_map=source_map,
        statistical=True,
        separator="|",
    )
    assert result.loc[0, "sender"] == "A"
    assert result.loc[0, "receiver"] == "B"
    assert result.loc[0, "score"] == pytest.approx(7.5)
    assert result.loc[0, "within_dataset_p_value"] == pytest.approx(0.03)


def test_liana_normalizer_maps_custom_pair_and_keeps_lower_score() -> None:
    native = pd.DataFrame(
        {
            "sample": ["s1", "s1"],
            "source": ["A", "A"],
            "target": ["B", "B"],
            "ligand_complex": ["L", "L"],
            "receptor_complex": ["R", "R"],
            "magnitude_rank": [0.4, 0.2],
            "specificity_rank": [0.3, 0.1],
        }
    )
    pair_map = pd.DataFrame(
        {"ligand": ["L"], "receptor": ["R"], "interaction_id": ["edge"]}
    )
    result = normalize_liana(native, pair_map=pair_map)
    assert len(result) == 1
    assert result.loc[0, "score"] == pytest.approx(0.2)
    assert result.loc[0, "specificity_score"] == pytest.approx(0.1)


def test_cellphonedb_database_filter_keeps_archive_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    output = tmp_path / "filtered.zip"
    repeated = tmp_path / "filtered_repeated.zip"
    interactions = pd.DataFrame(
        {
            "id_cp_interaction": ["keep", "drop"],
            "multidata_1_id": [1, 2],
            "multidata_2_id": [3, 4],
        }
    )
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("interaction_table.csv", interactions.to_csv(index=False))
        archive.writestr("gene_table.csv", "gene\nG\n")
    _filtered_database(source, output, source_interaction_ids={"keep"})
    _filtered_database(source, repeated, source_interaction_ids={"keep"})
    with zipfile.ZipFile(output) as archive:
        filtered = pd.read_csv(io.BytesIO(archive.read("interaction_table.csv")))
        assert archive.read("gene_table.csv") == b"gene\nG\n"
    assert filtered["id_cp_interaction"].tolist() == ["keep"]
    assert output.read_bytes() == repeated.read_bytes()


def test_nichenet_prior_operator_preserves_ligand_order_and_coverage() -> None:
    prior = pd.DataFrame(
        {
            "ligand": ["L1", "L1", "L2"],
            "target": ["T1", "T2", "T1"],
            "weight": [1.0, 3.0, 2.0],
        }
    )
    operator, weight_sum, target_count, ligand_index = _prior_operator(
        prior,
        var_names=pd.Index(["L1", "T1", "T2"]),
        ligand_order=["L1", "L2"],
    )
    assert operator.shape == (3, 2)
    np.testing.assert_allclose(weight_sum, [4.0, 2.0])
    np.testing.assert_array_equal(target_count, [2, 1])
    np.testing.assert_array_equal(ligand_index, [0, -1])


def test_nichenet_target_program_score_does_not_multiply_sender_expression() -> None:
    resource = pd.DataFrame({"interaction_id": ["i1"]})
    operator = pd.DataFrame([[0.0], [1.0]])
    observed = _score_samples(
        {
            ("s1", "Sender"): np.array([100.0, 0.0]),
            ("s1", "Receiver"): np.array([0.0, 4.0]),
        },
        sample_ids=["s1"],
        cell_types=["Sender", "Receiver"],
        resource=resource,
        operator=sparse.csr_matrix(operator),
        weight_sum=np.array([1.0]),
        support=pd.DataFrame(
            {
                "sample_id": ["s1", "s1"],
                "cell_type": ["Sender", "Receiver"],
                "n_cells": [20, 20],
            }
        ),
        min_cells=10,
    )

    assert set(observed["sender"]) == {"__source_agnostic__"}
    assert observed.loc[observed["receiver"] == "Receiver", "score"].iloc[0] == 4.0


def test_nichenet_count_input_is_library_normalized_and_log_transformed() -> None:
    counts = sparse.csr_matrix([[1, 1], [1, 3]], dtype=np.int32)
    adata = ad.AnnData(X=counts)

    expression, transform = _expression(adata, None)

    assert transform == "counts_library_size_1e4_log1p"
    np.testing.assert_allclose(
        expression.toarray(),
        np.log1p([[5000.0, 5000.0], [2500.0, 7500.0]]),
    )


def test_nichenet_continuous_expression_is_preserved() -> None:
    continuous = sparse.csr_matrix([[0.0, 0.25], [1.5, 2.75]])
    adata = ad.AnnData(X=continuous)

    expression, transform = _expression(adata, None)

    assert transform == "input_continuous_expression_preserved"
    np.testing.assert_allclose(expression.toarray(), continuous.toarray())
