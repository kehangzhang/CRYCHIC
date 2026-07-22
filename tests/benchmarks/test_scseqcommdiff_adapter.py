from __future__ import annotations

import hashlib
import importlib
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
    _validate_event_scores,
    _validate_native_pair_outputs,
    _validate_resource,
)
from benchmarks.adapters.scseqcommdiff.run import (
    run as run_benchmark,
)
from scipy import sparse


def test_r_driver_casts_integer_sparse_counts_to_double() -> None:
    module = importlib.import_module("benchmarks.adapters.scseqcommdiff.run")
    driver = Path(module.__file__).with_name("run.R").read_text(encoding="utf-8")

    assert 'adata$X$astype("float64")$tocsc()' in driver


def test_r_driver_types_empty_selected_result_keys_as_character() -> None:
    module = importlib.import_module("benchmarks.adapters.scseqcommdiff.run")
    driver = Path(module.__file__).with_name("run.R").read_text(encoding="utf-8")

    assert "sender = as.character(sender_unordered)" in driver
    assert "receiver = as.character(receiver_unordered)" in driver


def test_validate_event_scores_requires_unique_finite_native_effects(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.tsv.gz"
    table = pd.DataFrame(
        {
            "ligand": ["L", "L2"],
            "receptor": ["R", "R2"],
            "cluster_L": ["A", "A"],
            "cluster_R": ["B", "B"],
            "score_target": [0.7, np.nan],
            "score_reference": [0.2, np.nan],
            "effect": [0.5, np.nan],
            "logFC": [1.0, np.nan],
            "p_value": [0.01, np.nan],
            "native_rows_collapsed": [2, 1],
            "status": ["observed", "not_estimable"],
            "reason_code": ["", "non_finite_native_intercellular_score"],
            "target": ["case", "case"],
            "reference": ["control", "control"],
            "native_p_column": ["pvalue_S_inter", "pvalue_S_inter"],
        }
    )
    table.to_csv(path, sep="\t", index=False)

    assert _validate_event_scores(path, target="case", reference="control") == {
        "rows": 2,
        "observed_rows": 1,
        "not_estimable_rows": 1,
    }
    table.loc[0, "effect"] = 0.4
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="disagree"):
        _validate_event_scores(path, target="case", reference="control")
    table.loc[0, "score_target"] = np.inf
    table.loc[0, "effect"] = np.inf
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="finite"):
        _validate_event_scores(path, target="case", reference="control")


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


def test_validate_resource_accepts_checksum_bound_hcommon_subset(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "harmonized_lr.tsv"
    table = pd.DataFrame(
        {
            "harmonized_interaction_id": ["h1", "h2"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "scseqcommdiff_source_interaction_id": ["n1", "n2"],
            "scseqcommdiff_covered": [True, True],
        }
    )
    table.to_csv(resource, sep="\t", index=False)
    digest = hashlib.sha256(resource.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "resource_id": "test_hcommon",
                "version": "v1",
                "payload": {
                    "filename": resource.name,
                    "sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )

    observed = _validate_resource(
        resource,
        manifest,
        resource_mode="H-common",
    )

    assert observed == {
        "resource_id": "test_hcommon",
        "version": "v1",
        "mode": "H-common",
        "sha256": digest,
        "rows": 2,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


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


def test_native_pair_output_validation_rejects_zero_imputed_untested_pair(
    tmp_path: Path,
) -> None:
    rankings = pd.DataFrame(
        {
            "condition": ["case", "control", "case", "control"],
            "sender": ["A", "A", "A", "A"],
            "receiver": ["A", "A", "B", "B"],
            "ranked_strength": [2.0, 0.0, np.nan, np.nan],
            "status": ["observed", "observed", "not_estimable", "not_estimable"],
            "reason_code": [
                "",
                "",
                "no_finite_native_intercellular_test_for_cell_pair",
                "no_finite_native_intercellular_test_for_cell_pair",
            ],
        }
    )
    support = pd.DataFrame(
        {
            "sender": ["A", "A"],
            "receiver": ["A", "B"],
            "finite_native_p_rows": [5, 0],
            "finite_native_interactions": [3, 0],
            "pair_native_tested": [True, False],
            "cell_type_eligible": [True, True],
            "status": ["observed", "not_estimable"],
            "reason_code": [
                "",
                "no_finite_native_intercellular_test_for_cell_pair",
            ],
        }
    )
    ranking_path = tmp_path / "rankings.tsv"
    support_path = tmp_path / "support.tsv"
    rankings.to_csv(ranking_path, sep="\t", index=False)
    support.to_csv(support_path, sep="\t", index=False)

    paper_rankings = rankings.copy()
    paper_rankings.loc[
        paper_rankings["receiver"].eq("B"), ["ranked_strength", "status", "reason_code"]
    ] = [0.0, "observed", ""]
    paper_ranking_path = tmp_path / "paper_rankings.tsv"
    paper_rankings.to_csv(paper_ranking_path, sep="\t", index=False)

    audit = _validate_native_pair_outputs(
        ranking_path, paper_ranking_path, support_path
    )

    assert audit == {
        "pairs": 2,
        "tested_pairs": 1,
        "not_estimable_pairs": 1,
        "finite_native_p_rows": 5,
    }
    rankings.loc[rankings["receiver"].eq("B"), "ranked_strength"] = 0.0
    rankings.to_csv(ranking_path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="require missing strength"):
        _validate_native_pair_outputs(ranking_path, paper_ranking_path, support_path)
    rankings.loc[rankings["receiver"].eq("B"), "ranked_strength"] = np.nan
    rankings.to_csv(ranking_path, sep="\t", index=False)
    support.loc[support["receiver"].eq("B"), "status"] = "observed"
    support.to_csv(support_path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="eligibility and finite tests"):
        _validate_native_pair_outputs(ranking_path, paper_ranking_path, support_path)


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
        "resource": {"resource_id": RESOURCE_ID},
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
            resource_mode="native",
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
