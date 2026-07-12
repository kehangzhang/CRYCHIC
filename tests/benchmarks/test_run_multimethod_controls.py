from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from benchmarks.adapters.common import sha256_file
from benchmarks.simulation import run_multimethod_controls as runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, runner.RunnerSettings]:
    source = tmp_path / "source"
    source.mkdir()
    input_path = source / "synthetic_active.h5ad"
    input_path.write_bytes(b"synthetic-placeholder")
    manifest_path = source / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "crychic-multimethod-synthetic-v1",
                "records": [
                    {
                        "scenario": "active",
                        "dataset_id": "synthetic_active",
                        "scenario_seed": 17,
                        "expected_receiver_response": True,
                        "path": input_path.name,
                        "sha256": _sha256(input_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    resource_root = tmp_path / "resource"
    resource_root.mkdir()
    resource_path = resource_root / "harmonized_lr.tsv"
    resource_path.write_text("ligand\treceptor\nCXCL10\tCXCR3\n", encoding="utf-8")
    resource_manifest = resource_root / "manifest.json"
    resource_manifest.write_text(
        json.dumps(
            {
                "resource_id": "test_resource",
                "version": "1",
                "payload": {
                    "filename": resource_path.name,
                    "sha256": _sha256(resource_path),
                    "rows": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    executable = Path(sys.executable)
    settings = runner.RunnerSettings(
        database_root=tmp_path,
        harmonized_resource=resource_path,
        harmonized_manifest=resource_manifest,
        cellchat_environment="unused",
        cellphonedb_python=executable,
        liana_python=executable,
        crychic_python=executable,
    )
    return manifest_path, tmp_path / "output", settings


def _write_success(command: list[str]) -> None:
    output_dir = Path(command[4])
    output_dir.mkdir(parents=True)
    table_path = output_dir / "interactions_long.parquet"
    pd.DataFrame({"status": ["ok", "not_returned"]}).to_parquet(
        table_path, index=False
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sample_failures": {},
                "parameters": {
                    "resolved_expression_transform": (
                        "counts_library_size_1e4_log1p"
                    )
                },
                "output": {
                    "table": table_path.name,
                    "rows": 2,
                    "sha256": sha256_file(table_path),
                },
            }
        ),
        encoding="utf-8",
    )


def test_runner_records_success_and_resumes_without_reexecution(
    tmp_path: Path, monkeypatch: object
) -> None:
    manifest, output, settings = _fixture(tmp_path)
    calls = 0
    in_progress_statuses: list[str] = []

    def execute(command: list[str], **_: object) -> runner.ProcessResult:
        nonlocal calls
        calls += 1
        current = json.loads((output / "suite_manifest.json").read_text())
        in_progress_statuses.append(current["executions"][0]["status"])
        _write_success(command)
        return runner.ProcessResult(0, 1.25, 64.0)

    monkeypatch.setattr(runner, "_execute_process", execute)  # type: ignore[attr-defined]
    first = runner.run_multimethod_controls(
        manifest,
        output,
        settings=settings,
        methods=("nichenet",),
        scenarios=("active",),
    )
    second = runner.run_multimethod_controls(
        manifest,
        output,
        settings=settings,
        methods=("nichenet",),
        scenarios=("active",),
    )
    overwritten = runner.run_multimethod_controls(
        manifest,
        output,
        settings=settings,
        methods=("nichenet",),
        scenarios=("active",),
        overwrite=True,
    )

    assert calls == 2
    assert in_progress_statuses == ["running", "running"]
    assert first["executions"][0]["status"] == "complete"
    assert first["executions"][0]["ok_rows"] == 1
    assert (
        first["executions"][0]["resolved_expression_transform"]
        == "counts_library_size_1e4_log1p"
    )
    assert second["executions"][0]["execution_action"] == "resumed"
    assert second["executions"][0]["wall_time_seconds"] == 1.25
    assert overwritten["executions"][0]["execution_action"] == "executed"
    track_b = pd.read_csv(output / "track_b_scenario_truth.tsv", sep="\t")
    assert track_b.loc[0, "expected_receiver_response"]
    assert not track_b.loc[0, "lr_edge_truth_available"]


def test_process_failure_is_missing_measurement_not_zero(
    tmp_path: Path, monkeypatch: object
) -> None:
    manifest, output, settings = _fixture(tmp_path)

    def execute(command: list[str], **_: object) -> runner.ProcessResult:
        del command
        return runner.ProcessResult(9, 0.5, 12.0)

    monkeypatch.setattr(runner, "_execute_process", execute)  # type: ignore[attr-defined]
    suite = runner.run_multimethod_controls(
        manifest,
        output,
        settings=settings,
        methods=("nichenet",),
        scenarios=("active",),
    )

    execution = suite["executions"][0]
    assert execution["status"] == "failed"
    assert execution["reason_code"] == "adapter_process_failed"
    assert execution["exit_code"] == 9
    assert execution["rows"] is None
    assert execution["ok_rows"] is None


def test_sample_level_adapter_failure_invalidates_run(
    tmp_path: Path, monkeypatch: object
) -> None:
    manifest, output, settings = _fixture(tmp_path)

    def execute(command: list[str], **_: object) -> runner.ProcessResult:
        _write_success(command)
        output_dir = Path(command[4])
        manifest_path = output_dir / "manifest.json"
        adapter_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        adapter_manifest["sample_failures"] = {"subject_1_ctrl": {"message": "boom"}}
        manifest_path.write_text(json.dumps(adapter_manifest), encoding="utf-8")
        return runner.ProcessResult(0, 1.0, 32.0)

    monkeypatch.setattr(runner, "_execute_process", execute)  # type: ignore[attr-defined]
    suite = runner.run_multimethod_controls(
        manifest,
        output,
        settings=settings,
        methods=("nichenet",),
        scenarios=("active",),
    )

    execution = suite["executions"][0]
    assert execution["status"] == "failed"
    assert execution["reason_code"] == "adapter_sample_failures"
    assert execution["failed_samples"] == 1
    assert execution["ok_rows"] == 1


def test_track_a_truth_labels_only_active_sender_receiver_edge(
    tmp_path: Path,
) -> None:
    resource = [
        ("I_active", "CXCL10", "CXCR3"),
        ("I_2", "CCL5", "CCR5"),
        ("I_3", "VEGFA", "FLT1"),
        ("I_4", "CXCL12", "CXCR4"),
        ("I_5", "EGF", "EGFR"),
    ]
    records = [
        {
            "scenario": scenario,
            "dataset_id": f"synthetic_{scenario}",
            "expected_receiver_response": scenario == "active",
        }
        for scenario in ("active", "global_null")
    ]
    executions: dict[tuple[str, str], dict[str, object]] = {}
    for scenario in ("active", "global_null"):
        rows = [
            {
                "universe_id": f"universe_{scenario}",
                "sender": sender,
                "receiver": receiver,
                "interaction_id": interaction,
                "ligand": ligand,
                "receptor": receptor,
            }
            for sender in ("Bystander", "Receiver", "Sender")
            for receiver in ("Bystander", "Receiver", "Sender")
            for interaction, ligand, receptor in resource
        ]
        method_dir = tmp_path / scenario / "liana"
        method_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(
            method_dir / "interactions_long.parquet", index=False
        )
        executions[(scenario, "liana")] = {"status": "complete"}

    outputs = runner._write_truth_tables(  # type: ignore[attr-defined]
        tmp_path,
        source_records=records,
        executions=executions,
    )

    truth = pd.read_csv(tmp_path / "track_a_edge_truth.tsv", sep="\t")
    active = truth[truth["dataset"].eq("synthetic_active")]
    global_null = truth[truth["dataset"].eq("synthetic_global_null")]
    assert len(active) == 45
    assert active["is_positive"].sum() == 1
    positive = active[active["is_positive"].eq(1)].iloc[0]
    assert (positive["sender"], positive["receiver"]) == ("Sender", "Receiver")
    assert (positive["ligand"], positive["receptor"]) == ("CXCL10", "CXCR3")
    assert global_null["is_positive"].sum() == 0
    assert set(global_null["truth_status"]) == {"not_estimable"}
    assert set(global_null["reason_code"]) == {
        "truth_has_single_class_no_positive_integrated_edge"
    }
    assert outputs["track_a_edge_truth"]["rows"] == 90
