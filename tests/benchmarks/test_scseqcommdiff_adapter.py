from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.scseqcommdiff.run import (
    RESOURCE_ID,
    RESOURCE_ROWS,
    _filter_multi_condition_metadata,
    _filter_multi_sample_metadata,
    _input_manifest_sha256,
    _prepare_metadata,
    _validate_resource,
)
from benchmarks.adapters.scseqcommdiff.run import (
    run as run_benchmark,
)
from scipy import sparse


def test_input_manifest_digest_supports_both_prepared_schemas() -> None:
    digest = "a" * 64
    assert (
        _input_manifest_sha256(
            {"output": {"filename": "input.h5ad", "sha256": digest}},
            "input.h5ad",
        )
        == digest
    )
    assert (
        _input_manifest_sha256(
            {"output": "input.h5ad", "output_sha256": digest}, "input.h5ad"
        )
        == digest
    )
    with pytest.raises(ValueError, match="does not bind"):
        _input_manifest_sha256({}, "input.h5ad")


def test_prepare_metadata_supports_sample_and_subject_units(tmp_path: Path) -> None:
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "B"] * 8,
            "condition": ["case"] * 8 + ["control"] * 8,
            "sample_id": [f"s{i // 2}" for i in range(16)],
            "subject_id": [f"p{i // 2}" for i in range(16)],
        },
        index=[f"cell{i}" for i in range(16)],
    )
    source = ad.AnnData(
        X=sparse.csr_matrix(np.ones((16, 2))),
        obs=obs,
        var=pd.DataFrame(index=["L", "R"]),
    )
    path = tmp_path / "input.h5ad"
    source.write_h5ad(path)

    metadata, audit = _prepare_metadata(
        path,
        cell_type_key="cell_type",
        condition_key="condition",
        sample_unit_key="sample_id",
        target="case",
        reference="control",
    )

    assert list(metadata.columns) == [
        "Cell_ID",
        "Cluster_ID",
        "Condition_ID",
        "Sample_ID",
    ]
    assert audit["units_by_condition"] == {"case": 4, "control": 4}
    assert audit["n_cell_types"] == 2


def test_validate_resource_requires_the_frozen_complete_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "connectomedb2020.tsv"
    table = pd.DataFrame(
        {
            "ligand": [f"L{i}" for i in range(RESOURCE_ROWS)],
            "receptor": [f"R{i}" for i in range(RESOURCE_ROWS)],
            "scseqcommdiff_covered": [True] * RESOURCE_ROWS,
        }
    )
    table.to_csv(resource, sep="\t", index=False)
    digest = hashlib.sha256(resource.read_bytes()).hexdigest()
    monkeypatch.setattr("benchmarks.adapters.scseqcommdiff.run.RESOURCE_SHA256", digest)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"resource_id": RESOURCE_ID, "payload": {"sha256": digest}}),
        encoding="utf-8",
    )

    observed = _validate_resource(resource, manifest)

    assert observed["resource_id"] == RESOURCE_ID
    table.loc[0, "scseqcommdiff_covered"] = False
    table.to_csv(resource, sep="\t", index=False)
    monkeypatch.setattr(
        "benchmarks.adapters.scseqcommdiff.run.RESOURCE_SHA256",
        hashlib.sha256(resource.read_bytes()).hexdigest(),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["payload"]["sha256"] = hashlib.sha256(resource.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cover all"):
        _validate_resource(resource, manifest)


def test_validate_resource_rejects_non_object_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "connectomedb2020.tsv"
    table = pd.DataFrame(
        {
            "ligand": [f"L{i}" for i in range(RESOURCE_ROWS)],
            "receptor": [f"R{i}" for i in range(RESOURCE_ROWS)],
            "scseqcommdiff_covered": [True] * RESOURCE_ROWS,
        }
    )
    table.to_csv(resource, sep="\t", index=False)
    monkeypatch.setattr(
        "benchmarks.adapters.scseqcommdiff.run.RESOURCE_SHA256",
        hashlib.sha256(resource.read_bytes()).hexdigest(),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        _validate_resource(resource, manifest)


def test_multi_sample_filter_excludes_cell_type_missing_from_reference() -> None:
    records: list[dict[str, str]] = []
    for condition in ("case", "control"):
        for sample_number in range(4):
            for cell_type in ("shared", "case_only"):
                if condition == "control" and cell_type == "case_only":
                    continue
                for cell_number in range(2):
                    records.append(
                        {
                            "Cell_ID": (
                                f"{condition}-{sample_number}-{cell_type}-{cell_number}"
                            ),
                            "Cluster_ID": cell_type,
                            "Condition_ID": condition,
                            "Sample_ID": f"{condition}-{sample_number}",
                        }
                    )
    metadata = pd.DataFrame.from_records(records)

    filtered, support, audit = _filter_multi_sample_metadata(
        metadata,
        target="case",
        reference="control",
    )

    assert set(filtered["Cluster_ID"]) == {"shared"}
    assert audit["excluded_cell_types"] == ["case_only"]
    case_only = support.loc[support["cell_type"].eq("case_only")]
    assert not case_only["analysis_eligible"].any()
    assert set(case_only["reason_code"]) == {"insufficient_common_pseudobulk_support"}


def test_multi_condition_filter_excludes_cluster_with_fewer_than_two_cells() -> None:
    records: list[dict[str, str]] = []
    for condition in ("case", "control"):
        for cell_type, n_cells in (
            ("shared", 3),
            ("case_only", 2 if condition == "case" else 0),
            ("singleton", 2 if condition == "case" else 1),
        ):
            for cell_number in range(n_cells):
                records.append(
                    {
                        "Cell_ID": f"{condition}-{cell_type}-{cell_number}",
                        "Cluster_ID": cell_type,
                        "Condition_ID": condition,
                        "Sample_ID": condition,
                    }
                )
    metadata = pd.DataFrame.from_records(records)

    filtered, support, audit = _filter_multi_condition_metadata(
        metadata,
        target="case",
        reference="control",
    )

    assert set(filtered["Cluster_ID"]) == {"shared"}
    assert audit["excluded_cell_types"] == ["case_only", "singleton"]
    assert audit["minimum_cells_per_condition"] == 2
    excluded = support.loc[support["cell_type"].ne("shared")]
    assert not excluded["analysis_eligible"].any()
    assert set(excluded["reason_code"]) == {"insufficient_common_condition_support"}


def test_failed_external_process_is_recorded_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = pd.DataFrame.from_records(
        [
            {
                "Cell_ID": f"{condition}-{sample_number}-{cell_number}",
                "Cluster_ID": "shared",
                "Condition_ID": condition,
                "Sample_ID": f"{condition}-{sample_number}",
            }
            for condition in ("case", "control")
            for sample_number in range(4)
            for cell_number in range(2)
        ]
    )
    audit: dict[str, Any] = {
        "input": {"units_by_condition": {"case": 4, "control": 4}},
        "resource": {},
        "environment": {},
    }
    monkeypatch.setattr(
        "benchmarks.adapters.scseqcommdiff.run.preflight",
        lambda *args, **kwargs: (metadata, audit),
    )
    completed = subprocess.CompletedProcess(
        args=["Rscript"],
        returncode=9,
        stdout="partial output\n",
        stderr="first line\nprecise failure\n",
    )
    monkeypatch.setattr(
        "benchmarks.adapters.scseqcommdiff.run.subprocess.run",
        lambda *args, **kwargs: completed,
    )
    input_path = tmp_path / "input.h5ad"
    input_path.touch()
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="R runner failed"):
        run_benchmark(
            input_path,
            output_dir,
            input_manifest=tmp_path / "input.json",
            resource_path=tmp_path / "resource.tsv",
            resource_manifest=tmp_path / "resource.json",
            rscript=tmp_path / "Rscript",
            python_executable=Path(sys.executable),
            dataset_id="failure-test",
            scenario="multi-sample",
            cell_type_key="cell_type",
            condition_key="condition",
            sample_unit_key="sample_id",
            target="case",
            reference="control",
            cores=2,
            nrep=2,
            min_cells=2,
            preflight_only=False,
            overwrite=False,
        )

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["returncode"] == 9
    assert manifest["failure"] == {
        "message": "scSeqCommDiff R runner exited with code 9; inspect stderr.log",
        "returncode": 9,
        "stderr_log": "stderr.log",
        "stderr_tail": "first line\nprecise failure",
        "type": "ExternalProcessError",
    }
