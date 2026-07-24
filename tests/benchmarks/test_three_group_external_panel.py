from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.comprehensive.run_three_group_external_panel import (
    _completed_task,
    _provenance_commits,
    build_tasks,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(root: Path, name: str) -> tuple[str, str]:
    directory = root / "inputs" / name
    directory.mkdir(parents=True)
    h5ad = directory / f"{name}.h5ad"
    h5ad.write_bytes(name.encode())
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps({"output": {"filename": h5ad.name, "sha256": _sha(h5ad)}}),
        encoding="utf-8",
    )
    return str(manifest.relative_to(root)), _sha(h5ad)


def _fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    resource = fixture / "resource"
    resource.mkdir()
    table = resource / "harmonized_lr.tsv"
    pd.DataFrame({"ligand": ["L"], "receptor": ["R"]}).to_csv(
        table, sep="\t", index=False
    )
    (resource / "manifest.json").write_text("{}", encoding="utf-8")
    full_manifest, _ = _input(fixture, "three_group_r01_arm01")
    pairs = []
    for contrast, target, reference in (
        ("B_vs_A", "B", "A"),
        ("C_vs_A", "C", "A"),
        ("C_vs_B", "C", "B"),
    ):
        manifest, _ = _input(fixture, f"pair_{contrast}")
        pairs.append(
            {
                "contrast": contrast,
                "target": target,
                "reference": reference,
                "manifest": manifest,
            }
        )
    (fixture / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "crychic-three-group-fixture-v2",
                "code": {"commit": "abc", "dirty": False},
                "resource": {
                    "filename": "resource/harmonized_lr.tsv",
                    "manifest": "resource/manifest.json",
                    "sha256": _sha(table),
                },
                "records": [
                    {
                        "dataset_id": "three_group_r01_arm01",
                        "manifest": full_manifest,
                        "pairs": pairs,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return fixture


def test_build_tasks_expands_full_and_pairwise_methods(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    tasks, _ = build_tasks(fixture, tmp_path / "runs")

    assert len(tasks) == 5
    assert [task.method for task in tasks].count("cellchat") == 1
    assert [task.method for task in tasks].count("liana") == 1
    assert [task.method for task in tasks].count("scseqcommdiff") == 3
    assert {task.contrast for task in tasks if task.method == "scseqcommdiff"} == {
        "B_vs_A",
        "C_vs_A",
        "C_vs_B",
    }


def test_build_tasks_rejects_unknown_method(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        build_tasks(fixture, tmp_path / "runs", methods=("unknown",))


def test_completed_task_requires_clean_code_and_input_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    tasks, _ = build_tasks(fixture, tmp_path / "runs", methods=("cellchat",))
    task = tasks[0]
    task.output_dir.mkdir(parents=True)
    manifest_path = task.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "code": {"commit": "abc", "dirty": False},
                "input": {"sha256": task.input_sha256},
            }
        ),
        encoding="utf-8",
    )

    assert _completed_task(task, expected_commit="abc")
    assert not _completed_task(task, expected_commit="different")


def test_fixture_and_adapter_may_have_distinct_clean_commits(tmp_path: Path) -> None:
    fixture = json.loads((_fixture(tmp_path) / "manifest.json").read_text())

    assert _provenance_commits(
        fixture, {"commit": "new-adapter", "dirty": False}
    ) == ("abc", "new-adapter")


def test_provenance_rejects_dirty_adapter(tmp_path: Path) -> None:
    fixture = json.loads((_fixture(tmp_path) / "manifest.json").read_text())

    with pytest.raises(ValueError, match="adapter worktree must be clean"):
        _provenance_commits(fixture, {"commit": "new-adapter", "dirty": True})
