from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.adapters.liana import run_condition_aware as runner
from benchmarks.adapters.liana import run_condition_aware_exact as exact_runner


def test_input_manifest_sha256_accepts_both_prepared_shapes() -> None:
    assert (
        runner._input_manifest_sha256(
            {"output": {"filename": "input.h5ad", "sha256": "abc"}},
            "input.h5ad",
        )
        == "abc"
    )
    assert (
        runner._input_manifest_sha256(
            {"output": "input.h5ad", "output_sha256": "def"},
            "input.h5ad",
        )
        == "def"
    )


def test_probe_environment_rejects_version_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    observed = dict(runner.EXPECTED_ENVIRONMENT)
    observed["liana"] = "1.7.3"

    class Completed:
        returncode = 0
        stdout = json.dumps({"packages": observed})
        stderr = ""

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises(runner.ExactEnvironmentUnavailable, match="versions"):
        runner._probe_environment(python)


def test_build_rankings_counts_calls_and_preserves_not_estimable_pairs() -> None:
    calls = pd.DataFrame(
        {
            "condition": ["case", "case", "ctrl"],
            "source": ["A", "B", "A"],
            "target": ["B", "A", "A"],
            "interaction_id": ["lr1", "lr2", "lr3"],
        }
    )
    lr_results = pd.DataFrame(
        {
            "source": ["A", "B", "A", "A"],
            "target": ["B", "A", "A", "B"],
            "interaction_id": ["lr1", "lr2", "lr3", "lr4"],
        }
    )
    result = runner._build_rankings(
        calls,
        lr_results,
        source_cell_types=("A", "B", "C"),
        eligible_cell_types=("A", "B"),
        target="case",
        reference="ctrl",
        dataset_id="toy",
        replicate_key="sample_id",
        subject_key="subject_id",
    )
    case_ab = result.loc[
        result["condition"].eq("case")
        & result["sender"].eq("A")
        & result["receiver"].eq("B")
    ].iloc[0]
    assert case_ab["ranked_strength"] == 2
    assert case_ab["estimable_directed_lr"] == 3
    ctrl_ab = result.loc[
        result["condition"].eq("ctrl")
        & result["sender"].eq("A")
        & result["receiver"].eq("B")
    ].iloc[0]
    assert ctrl_ab["ranked_strength"] == 0
    pair_ac = result.loc[
        result["condition"].eq("case")
        & result["sender"].eq("A")
        & result["receiver"].eq("C")
    ].iloc[0]
    assert pair_ac["status"] == "not_estimable"
    assert pd.isna(pair_ac["ranked_strength"])
    assert ";sample_pseudobulk;" in case_ab["ranking_semantics"]
    assert pair_ac["reason_code"] == "cell_type_not_estimable_for_sample_pseudobulk_de"


def test_build_rankings_rejects_duplicate_directed_calls() -> None:
    calls = pd.DataFrame(
        {
            "condition": ["case", "case"],
            "source": ["A", "A"],
            "target": ["B", "B"],
            "interaction_id": ["lr1", "lr1"],
        }
    )
    lr_results = calls.drop(columns="condition").drop_duplicates()
    with pytest.raises(ValueError, match="duplicate"):
        runner._build_rankings(
            calls,
            lr_results,
            source_cell_types=("A", "B"),
            eligible_cell_types=("A", "B"),
            target="case",
            reference="ctrl",
            dataset_id="toy",
            replicate_key="subject_id",
            subject_key="subject_id",
        )


def test_validate_resource_requires_frozen_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resource = tmp_path / "resource.tsv"
    resource.write_text("ligand\treceptor\nA\tB\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "sha256_file", lambda path: "wrong")
    with pytest.raises(ValueError, match="checksum"):
        runner._validate_resource(resource, manifest)


def test_subject_metadata_columns_are_unique() -> None:
    columns = list(
        dict.fromkeys(["subject_id", "subject_id", "condition", "cell_type"])
    )
    assert columns == ["subject_id", "condition", "cell_type"]


def test_pydeseq2_formulaic_coefficient_name_is_supported() -> None:
    assert (
        exact_runner._coefficient_name(
            ["Intercept", "condition[T.case]"], "case", "ctrl"
        )
        == "condition[T.case]"
    )


def test_effective_inference_cores_must_match_request() -> None:
    inference = type("Inference", (), {"n_cpus": 3})()
    assert exact_runner._validate_inference_cores(inference, 3) == 3
    with pytest.raises(RuntimeError, match="effective n_cpus"):
        exact_runner._validate_inference_cores(inference, 2)


def test_pydeseq_failure_aborts_exact_run() -> None:
    analysis = pd.DataFrame(
        {
            "cell_type": ["A", "B"],
            "status": ["analyzed", "failed"],
            "error": ["", "ValueError: injected failure"],
        }
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        exact_runner._raise_on_cell_type_failures(analysis)


def test_external_worker_failure_marks_outer_manifest_failed(tmp_path: Path) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text("injected worker failure\n", encoding="utf-8")
    manifest: dict[str, object] = {"status": "initializing"}
    outcome = runner._ProcessOutcome(
        returncode=1,
        peak_memory_percent=20.0,
        memory_guard_triggered=False,
    )
    assert runner._record_failed_outcome(manifest, outcome, stderr)
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "ExternalProcessError",
        "message": "LIANA+ exact runner exited with code 1",
        "returncode": 1,
        "stderr_tail": "injected worker failure",
    }


def test_exact_worker_validates_script_digest(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text("print('fixed')\n", encoding="utf-8")
    expected = hashlib.sha256(script.read_bytes()).hexdigest()
    assert exact_runner._validate_script_digest(script, expected) == expected
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        exact_runner._validate_script_digest(script, "0" * 64)
