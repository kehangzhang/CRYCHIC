from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.openproblems.common import sha256_file, validate_predictions, write_json
from benchmarks.openproblems.run_crychic_source_target import (
    METHOD_SCOPE,
    load_task_resource,
    run,
)


def _resource(path: Path) -> Path:
    path.mkdir()
    table_path = path / "liana_consensus_mouse.parquet"
    pd.DataFrame(
        {
            "interaction_id": ["opv1::L::R", "opv1::L2::R2"],
            "ligand": ["L", "L2"],
            "receptor": ["R", "R2"],
            "input_available": [True, True],
        }
    ).to_parquet(table_path, index=False)
    write_json(
        path / "resource_manifest.json",
        {
            "schema_version": "crychic-openproblems-source-target-resource-v1",
            "task_version": "v1.0.0",
            "resource_id": "toy_mouse_consensus",
            "input_available_interactions": 2,
            "files": {
                "resource": {
                    "filename": table_path.name,
                    "sha256": sha256_file(table_path),
                }
            },
        },
    )
    return path


def _input(path: Path) -> Path:
    labels = pd.Categorical(
        ["B", "B", "A", "A", "C", "C"],
        categories=["B", "A", "C"],
    )
    counts = np.array(
        [
            [9, 1, 7, 1],
            [8, 1, 6, 1],
            [2, 8, 1, 7],
            [1, 7, 1, 6],
            [5, 5, 4, 4],
            [4, 4, 3, 3],
        ],
        dtype=np.int64,
    )
    data = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"label": labels}, index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame(index=["L", "R", "L2", "R2"]),
    )
    data.uns["ccc_target"] = pd.DataFrame(
        {
            "source": ["B", "A"],
            "target": ["A", "B"],
            "response": [1, 0],
        }
    )
    data.uns["secret_spatial_proxy"] = np.eye(3)
    data.write_h5ad(path)
    return path


def test_crychic_runner_uses_core_and_isolates_truth(tmp_path: Path) -> None:
    resource_dir = _resource(tmp_path / "resource")
    input_path = _input(tmp_path / "input.h5ad")
    output = tmp_path / "output"

    manifest = run(
        input_path,
        resource_dir,
        output,
        min_cells=2,
        overwrite=False,
    )

    assert manifest["status"] == "complete"
    assert manifest["parameters"] == {
        "min_cells": 2,
        "aggregation": ["max", "sum"],
        "score_arms": {
            "availability_state": "preexisting_core_mode",
            "availability_ecosystem": "preexisting_core_mode",
            "availability_specificity": (
                "posthoc_supportive_natmi_style_diagnostic"
            ),
        },
        "formal_inference": False,
        "subject_crossfit": False,
        "single_context_static_diagnostic": True,
    }
    isolation = manifest["input"]["truth_and_proxy_isolation"]
    assert isolation["all_uns_removed_before_core_execution"] is True
    assert "ccc_target" in isolation["removed_key_names"]
    assert "secret_spatial_proxy" in isolation["removed_key_names"]
    assert manifest["core_execution"]["entrypoints"] == [
        "crychic.data.validate_anndata",
        "crychic.pseudobulk.aggregate_pseudobulk",
        "crychic.availability.estimate_bundle_availability",
    ]
    assert manifest["core_execution"]["input_mode"] == "counts"
    assert manifest["core_execution"]["observed_rows"] == 18
    prediction_paths = tuple((output / "predictions").glob("*.parquet"))
    assert len(prediction_paths) == 6
    for path in prediction_paths:
        table = pd.read_parquet(path)
        validate_predictions(table, labels=("B", "A", "C"))
        assert set(table["aggregation"]) == {
            path.stem.rsplit("_", maxsplit=1)[-1]
        }
        if "specificity" not in path.stem:
            assert set(table["method_scope"]) == {METHOD_SCOPE}


def test_resource_bridge_rejects_checksum_drift(tmp_path: Path) -> None:
    resource_dir = _resource(tmp_path / "resource")
    table_path = resource_dir / "liana_consensus_mouse.parquet"
    table = pd.read_parquet(table_path)
    table.loc[0, "ligand"] = "CHANGED"
    table.to_parquet(table_path, index=False)

    with pytest.raises(ValueError, match="checksum"):
        load_task_resource(table_path, resource_dir / "resource_manifest.json")
