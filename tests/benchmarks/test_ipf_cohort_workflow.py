from __future__ import annotations

import gzip
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import sha256_file, write_json
from benchmarks.literature.prepare_ipf_cohort import (
    build_sample_h5ad,
    prepare_ipf_cohort,
)
from benchmarks.literature.run_ipf_cohort import (
    OUTPUT_PLACEHOLDER,
    TaskSpec,
    plan_hcommon_tasks,
    read_memory_fraction,
    run_task_queue,
    write_task_plan,
)
from benchmarks.literature.summarize_ipf_cohort import (
    summarize_patient_equal,
    validate_evaluation_task_provenance,
)
from scipy import sparse
from scipy.io import mmwrite


def _write_sparse_bundle(
    root: Path, *, study_id: str = "GSE1", sample_id: str = "s1"
) -> Path:
    root.mkdir(parents=True)
    counts = sparse.csc_matrix(
        np.array(
            [
                [5, 4, 0, 0],
                [0, 0, 6, 5],
                [2, 1, 2, 1],
            ],
            dtype=np.int64,
        )
    )
    uncompressed = root / "counts.mtx"
    mmwrite(uncompressed, counts)
    with (
        uncompressed.open("rb") as source,
        gzip.open(root / "counts.mtx.gz", "wb") as destination,
    ):
        destination.write(source.read())
    uncompressed.unlink()
    pd.DataFrame({"gene_symbol": ["L1", "R1", "G1"]}).to_csv(
        root / "features.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "barcode": ["c1", "c2", "c3", "c4"],
            "study_id": [study_id] * 4,
            "sample_id": [sample_id] * 4,
            "cell_type": ["Macrophage", "Macrophage", "AT2", "AT2"],
        }
    ).to_csv(root / "cells.tsv", sep="\t", index=False)
    write_json(
        root / "manifest.json",
        {
            "schema_version": "xie-ipf-processed-rds-sample-bundle-v1",
            "status": "complete",
            "outputs": {
                key: {
                    "filename": filename,
                    "sha256": sha256_file(root / filename),
                }
                for key, filename in {
                    "counts": "counts.mtx.gz",
                    "features": "features.tsv",
                    "cells": "cells.tsv",
                }.items()
            },
        },
    )
    return root


def test_parallel_cohort_prepare_preserves_roster_order(tmp_path: Path) -> None:
    export = tmp_path / "exports"
    _write_sparse_bundle(export / "GSE1" / "s2", sample_id="s2")
    _write_sparse_bundle(export / "GSE1" / "s1", sample_id="s1")
    roster = tmp_path / "roster.tsv"
    pd.DataFrame({"geo_accession": ["GSE1", "GSE1"], "sample_id": ["s2", "s1"]}).to_csv(
        roster, sep="\t", index=False
    )

    result = prepare_ipf_cohort(export, roster, tmp_path / "cohort", workers=2)

    # Canonical roster sorting is deterministic regardless of process completion order.
    assert result["sample_id"].tolist() == ["s1", "s2"]
    assert result["status"].tolist() == ["complete", "complete"]
    assert (tmp_path / "cohort" / "sample_manifest.json").is_file()


def test_build_sample_h5ad_validates_sparse_bundle_and_metadata(tmp_path: Path) -> None:
    bundle = _write_sparse_bundle(tmp_path / "bundle")
    output = tmp_path / "input" / "sample.h5ad"

    manifest = build_sample_h5ad(
        bundle, output, expected_study="GSE1", expected_sample="s1"
    )

    assert manifest["status"] == "complete"
    assert manifest["shape_cells_by_genes"] == [4, 3]
    assert manifest["cell_type_counts"] == {"Macrophage": 2, "AT2": 2}
    assert manifest["output"]["sha256"] == sha256_file(output)
    import anndata as ad

    data = ad.read_h5ad(output)
    assert data.shape == (4, 3)
    assert set(data.obs["subject_id"].astype(str)) == {"s1"}
    assert set(data.obs["condition"].astype(str)) == {"IPF"}
    assert data.layers["counts"].dtype == np.int64
    assert np.isfinite(data.X.data).all()


def test_reused_h5ad_manifest_must_match_roster_identity(tmp_path: Path) -> None:
    export = tmp_path / "exports"
    _write_sparse_bundle(export / "GSE1" / "s1")
    roster = tmp_path / "roster.tsv"
    pd.DataFrame({"geo_accession": ["GSE1"], "sample_id": ["s1"]}).to_csv(
        roster, sep="\t", index=False
    )
    output = tmp_path / "cohort"
    prepare_ipf_cohort(export, roster, output)
    manifest_path = output / "GSE1" / "s1" / "input" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sample_id"] = "wrong"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="existing sample input failed"):
        prepare_ipf_cohort(export, roster, output)


def test_evaluation_task_provenance_verifies_all_output_hashes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    expected = (
        "bootstrap_summary.tsv",
        "materialized_scores.parquet",
        "negative_sampling_summary.tsv",
        "point_estimates.tsv",
    )
    for relative in expected:
        (output / relative).write_text(relative)
    task_id = "GSE1__s1__evaluate_hcommon"
    command_sha = "a" * 64
    outputs = {
        relative: {
            "bytes": (output / relative).stat().st_size,
            "sha256": sha256_file(output / relative),
        }
        for relative in expected
    }
    write_json(
        output / "cohort_task_manifest.json",
        {
            "status": "complete",
            "task_id": task_id,
            "study_id": "GSE1",
            "sample_id": "s1",
            "stage": "evaluate",
            "arm": "hcommon_representable",
            "command_sha256": command_sha,
            "outputs": outputs,
        },
    )
    ledger = tmp_path / "ledger.tsv"
    pd.DataFrame(
        {
            "task_id": [task_id],
            "study_id": ["GSE1"],
            "sample_id": ["s1"],
            "stage": ["evaluate"],
            "arm": ["hcommon_representable"],
            "command_sha256": [command_sha],
            "publish_dir": [str(output)],
            "expected_outputs_json": [json.dumps(expected)],
            "status": ["complete"],
        }
    ).to_csv(ledger, sep="\t", index=False)
    samples = pd.DataFrame({"study_id": ["GSE1"], "sample_id": ["s1"]})

    result = validate_evaluation_task_provenance(ledger, samples)

    assert result["evaluation_tasks"] == 1
    assert result["evaluation_task_manifests"] == 1
    assert len(result["evaluation_manifest_aggregate_sha256"]) == 64
    (output / "point_estimates.tsv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_evaluation_task_provenance(ledger, samples)


def _task(tmp_path: Path, name: str, script: str, *, dependencies=()) -> TaskSpec:
    return TaskSpec(
        task_id=name,
        study_id="GSE1",
        sample_id="s1",
        stage="test",
        arm=name,
        command=(
            sys.executable,
            "-c",
            script,
            OUTPUT_PLACEHOLDER,
        ),
        publish_dir=tmp_path / "results" / name,
        expected_outputs=("result.txt",),
        dependencies=tuple(dependencies),
    )


def _writer_script(value: str) -> str:
    return (
        "from pathlib import Path; import sys; "
        f"Path(sys.argv[1], 'result.txt').write_text({value!r})"
    )


def test_resumable_queue_publishes_atomically_and_skips_verified_task(
    tmp_path: Path,
) -> None:
    first = _task(
        tmp_path,
        "first",
        _writer_script("one"),
    )
    second = _task(
        tmp_path,
        "second",
        _writer_script("two"),
        dependencies=("first",),
    )
    ledger = tmp_path / "tasks.tsv"
    write_task_plan([first, second], ledger)

    result = run_task_queue(
        ledger,
        max_workers=2,
        poll_seconds=0.01,
        memory_reader=lambda: 0.2,
    )

    assert set(result["status"]) == {"complete"}
    assert result.set_index("task_id").loc["first", "attempts"] == 1
    assert (first.publish_dir / "result.txt").read_text() == "one"
    assert (
        json.loads((first.publish_dir / "cohort_task_manifest.json").read_text())[
            "threads"
        ]
        == 1
    )

    resumed = run_task_queue(
        ledger,
        max_workers=2,
        poll_seconds=0,
        memory_reader=lambda: 0.2,
    )
    assert resumed.set_index("task_id").loc["first", "attempts"] == 1
    assert resumed.set_index("task_id").loc["second", "attempts"] == 1


def test_queue_soft_memory_gate_delays_dispatch(tmp_path: Path) -> None:
    task = _task(
        tmp_path,
        "soft",
        _writer_script("ok"),
    )
    ledger = tmp_path / "tasks.tsv"
    write_task_plan([task], ledger)
    readings = iter([0.66, 0.64, 0.64, 0.64, 0.64])

    result = run_task_queue(
        ledger,
        max_workers=1,
        poll_seconds=0.01,
        memory_reader=lambda: next(readings, 0.64),
    )

    row = result.iloc[0]
    assert row["status"] == "complete"
    assert float(row["memory_fraction_at_start"]) == pytest.approx(0.64)


def test_keyboard_interrupt_terminates_task_group_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "release"
    script = (
        "from pathlib import Path; import sys,time; "
        f"marker=Path({str(marker)!r}); "
        "time.sleep(60) if not marker.exists() else "
        "Path(sys.argv[1], 'result.txt').write_text('recovered')"
    )
    task = _task(tmp_path, "interrupt", script)
    ledger = tmp_path / "tasks.tsv"
    write_task_plan([task], ledger)
    pids: list[int] = []
    from benchmarks.literature import run_ipf_cohort as scheduler

    original_start = scheduler._start_task

    def start_spy(*args, **kwargs):
        running = original_start(*args, **kwargs)
        pids.append(running.process.pid)
        return running

    monkeypatch.setattr(scheduler, "_start_task", start_spy)
    calls = 0

    def interrupt_after_dispatch() -> float:
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise KeyboardInterrupt
        return 0.2

    with pytest.raises(KeyboardInterrupt):
        run_task_queue(
            ledger,
            max_workers=1,
            poll_seconds=0,
            memory_reader=interrupt_after_dispatch,
        )
    assert len(pids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)
    interrupted = pd.read_csv(ledger, sep="\t").iloc[0]
    assert interrupted["status"] == "failed"
    assert interrupted["failure_reason"] == "scheduler_interrupted"
    assert interrupted["attempts"] == 1
    assert not list((tmp_path / "results").glob(".*.tmp-*"))

    marker.touch()
    resumed = run_task_queue(
        ledger,
        max_workers=1,
        poll_seconds=0.01,
        memory_reader=lambda: 0.2,
    )
    row = resumed.iloc[0]
    assert row["status"] == "complete"
    assert row["attempts"] == 2
    assert (task.publish_dir / "result.txt").read_text() == "recovered"


def test_sigterm_handler_enters_scheduler_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(
        tmp_path,
        "sigterm",
        "import time; time.sleep(60)",
    )
    ledger = tmp_path / "tasks.tsv"
    write_task_plan([task], ledger)
    previous = signal.getsignal(signal.SIGTERM)
    calls = 0

    def invoke_installed_handler() -> float:
        nonlocal calls
        calls += 1
        if calls >= 3:
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
        return 0.2

    with pytest.raises(SystemExit, match="143"):
        run_task_queue(
            ledger,
            max_workers=1,
            poll_seconds=0,
            memory_reader=invoke_installed_handler,
        )
    assert signal.getsignal(signal.SIGTERM) is previous
    row = pd.read_csv(ledger, sep="\t").iloc[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "scheduler_interrupted"
    assert not list((tmp_path / "results").glob(".*.tmp-*"))


def test_memory_reader_and_threshold_validation(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1000 kB\nMemAvailable: 300 kB\n")
    assert read_memory_fraction(meminfo) == pytest.approx(0.7)
    task = _task(
        tmp_path,
        "limit",
        _writer_script("ok"),
    )
    ledger = tmp_path / "tasks.tsv"
    write_task_plan([task], ledger)
    with pytest.raises(ValueError, match=r"hard <= 0\.70"):
        run_task_queue(
            ledger,
            hard_memory_fraction=0.71,
            memory_reader=lambda: 0.2,
        )


def test_hcommon_plan_declares_single_thread_and_dependencies(tmp_path: Path) -> None:
    input_path = tmp_path / "input.h5ad"
    input_path.write_bytes(b"fixture")
    samples = tmp_path / "samples.tsv"
    pd.DataFrame(
        {
            "study_id": ["GSE1"],
            "sample_id": ["s1"],
            "dataset_id": ["GSE1_s1_IPF_8cell"],
            "input_h5ad": [str(input_path)],
            "input_sha256": [sha256_file(input_path)],
            "status": ["complete"],
        }
    ).to_csv(samples, sep="\t", index=False)
    resource = tmp_path / "resource.tsv"
    resource.write_text("ligand\treceptor\nL1\tR1\n")
    resource_manifest = tmp_path / "resource.json"
    resource_manifest.write_text("{}\n")
    gold = tmp_path / "gold.tsv"
    gold.write_text("source\ttarget\tligand\treceptor\nA\tB\tL1\tR1\n")

    tasks = plan_hcommon_tasks(
        samples,
        tmp_path / "results",
        resource=resource,
        resource_manifest=resource_manifest,
        gold=gold,
        liana_python_executable=sys.executable,
        crychic_python_executable=Path(sys.executable).parent / "python",
        sample_ids=("s1",),
    )

    assert len(tasks) == 3
    assert {task.threads for task in tasks} == {1}
    liana = next(task for task in tasks if task.arm == "liana_hcommon")
    crychic = next(task for task in tasks if task.arm == "crychic_hcommon")
    assert liana.command[0] == str(Path(sys.executable).absolute())
    assert crychic.command[0] == str(
        (Path(sys.executable).parent / "python").absolute()
    )
    assert any("--n-jobs" in task.command and "1" in task.command for task in tasks)
    evaluation = next(task for task in tasks if task.stage == "evaluate")
    assert evaluation.command[0] == crychic.command[0]
    assert len(evaluation.dependencies) == 2


def test_patient_equal_summary_does_not_weight_study_or_sample_size() -> None:
    rows = []
    for study, sample, auroc in (
        ("A", "a1", 0.2),
        ("A", "a2", 0.4),
        ("B", "b1", 0.9),
    ):
        rows.append(
            {
                "study_id": study,
                "sample_id": sample,
                "dataset_id": sample,
                "method": "m",
                "status": "observed",
                "auroc": auroc,
                "average_precision": auroc,
                "balanced_auprc": auroc,
                "precision": auroc,
                "sensitivity": auroc,
                "specificity": auroc,
                "f1": auroc,
                "mcc": auroc,
                "raw_positive_return_fraction": auroc,
            }
        )

    point, by_study, bootstrap = summarize_patient_equal(
        pd.DataFrame(rows), expected_samples=3, n_bootstrap=20, seed=7
    )

    auroc = point.loc[point["metric"].eq("auroc")].iloc[0]
    assert auroc["patient_equal_mean"] == pytest.approx(0.5)
    assert auroc["n_evaluable"] == 3
    assert auroc["patient_coverage_fraction"] == 1
    assert set(by_study["study_id"]) == {"A", "B"}
    assert set(bootstrap["resampling_scheme"]) == {
        "patient_bootstrap_stratified_by_study"
    }
