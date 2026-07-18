from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import anndata as ad
import benchmarks.datasets.prepare_kuppe_ctrl_iz as module
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


def _fixture_expected() -> dict[str, object]:
    return {
        "source_shape": [12, 4],
        "n_obs": 11,
        "n_vars": 4,
        "n_subjects": 3,
        "n_samples": 3,
        "n_cell_types": 11,
        "cell_types": list(module.CELL_TYPES),
        "cells_by_condition": {"CTRL": 5, "IZ": 6},
        "samples_by_condition": {"CTRL": 1, "IZ": 2},
        "subjects_by_condition": {"CTRL": 1, "IZ": 2},
    }


def _write_fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(
        [
            [2, 0, 1, 0],
            [0, 3, 0, 1],
            [4, 0, 0, 0],
            [0, 2, 2, 0],
            [1, 1, 0, 3],
            [0, 0, 5, 1],
            [2, 1, 0, 0],
            [0, 4, 0, 1],
            [1, 0, 2, 2],
            [0, 1, 3, 0],
            [2, 0, 0, 4],
            [9, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    normalized = np.log1p(counts.astype(np.float64))
    conditions = ["CTRL"] * 5 + ["IZ"] * 6 + ["RZ"]
    samples = ["CTRL_1"] * 5 + ["IZ_1"] * 3 + ["IZ_2"] * 3 + ["RZ_1"]
    patients = ["P1"] * 5 + ["P2"] * 3 + ["P3"] * 3 + ["P4"]
    cell_types = [*module.CELL_TYPES, module.CELL_TYPES[0]]
    obs = pd.DataFrame(
        {
            "sample": samples,
            "patient": patients,
            "cell_type_original": cell_types,
            "major_labl": conditions,
            "n_counts": counts.sum(axis=1).astype(float),
            "n_genes": np.count_nonzero(counts, axis=1).astype(np.int32),
        },
        index=pd.Index([f"cell_{i}" for i in range(12)], name="cell_id"),
    )
    var = pd.DataFrame(index=pd.Index(["G1", "G2", "G3", "G4"]))
    source = ad.AnnData(X=sparse.csc_matrix(normalized), obs=obs, var=var)
    source.raw = ad.AnnData(
        X=sparse.csc_matrix(counts.astype(np.float32)),
        obs=obs.copy(),
        var=var.copy(),
    )
    source.write_h5ad(path)
    return counts, normalized


def test_prepare_kuppe_ctrl_iz_writes_counts_ready_subset_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.h5ad"
    counts, normalized = _write_fixture(source_path)
    monkeypatch.setattr(module, "EXPECTED_KUPPE_CTRL_IZ", _fixture_expected())

    output_path = tmp_path / "kuppe_ctrl_iz.h5ad"
    result = module.prepare_kuppe_ctrl_iz(source_path, output_path)

    assert result.shape == (11, 4)
    assert result.obs_names.tolist() == [f"cell_{i}" for i in range(11)]
    assert set(result.obs["condition"].astype(str)) == {"CTRL", "IZ"}
    assert result.obs["sample_id"].tolist() == result.obs["sample"].tolist()
    assert result.obs["subject_id"].tolist() == result.obs["patient"].tolist()
    assert result.obs["cell_type"].tolist() == (
        result.obs["cell_type_original"].tolist()
    )
    assert result.obs["condition"].tolist() == result.obs["major_labl"].tolist()
    assert result.obs["region"].tolist() == result.obs["condition"].tolist()
    assert sparse.isspmatrix_csr(result.X)
    assert sparse.isspmatrix_csr(result.layers["counts"])
    assert result.layers["counts"].dtype == np.int32
    assert np.array_equal(result.layers["counts"].toarray(), counts[:11])
    assert np.array_equal(result.X.toarray(), normalized[:11])

    manifest_path = tmp_path / "kuppe_ctrl_iz.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source"]["sha256"] == module._sha256(source_path)
    assert manifest["output"]["sha256"] == module._sha256(output_path)
    assert manifest["cohort"]["n_cells"] == 11
    assert manifest["cohort"]["n_samples"] == 3
    assert manifest["cohort"]["n_cell_types"] == 11
    assert manifest["cohort"]["cell_types"] == sorted(module.CELL_TYPES)
    assert manifest["design"]["analysis_unit"] == "subject_id"
    assert manifest["manifest_payload_sha256"] == (
        module._manifest_payload_sha256(manifest)
    )
    for suffix in ("sample_design.tsv", "cell_type_support.tsv", "obs_audit.tsv"):
        assert (tmp_path / f"kuppe_ctrl_iz.{suffix}").is_file()

    round_trip = ad.read_h5ad(output_path)
    assert np.array_equal(round_trip.layers["counts"].toarray(), counts[:11])


def test_audit_kuppe_ctrl_iz_reads_only_backed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.h5ad"
    _write_fixture(source_path)
    monkeypatch.setattr(module, "EXPECTED_KUPPE_CTRL_IZ", _fixture_expected())

    def fail_if_expression_is_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("audit-only path accessed an expression matrix")

    monkeypatch.setattr(module, "_selected_matrices", fail_if_expression_is_read)
    result = module.audit_kuppe_ctrl_iz(
        source_path,
        tmp_path / "audit",
        compute_source_sha256=False,
    )

    manifest = cast(dict[str, Any], result["manifest"])
    assert manifest["mode"] == "obs_design_audit"
    assert manifest["source"]["read_mode"] == "backed_r"
    assert manifest["source"]["sha256"] is None
    assert manifest["matrices"]["expression_matrix_accessed"] is False
    assert manifest["matrices"]["output_h5ad_written"] is False
    assert manifest["cohort"]["n_cells"] == 11
    assert manifest["cohort"]["n_samples"] == 3
    assert manifest["cohort"]["n_cell_types"] == 11
    assert not list((tmp_path / "audit").glob("*.h5ad"))

    artifacts = cast(dict[str, str], result["artifacts"])
    design = pd.read_csv(artifacts["sample_design"], sep="\t")
    assert design[["sample_id", "subject_id", "condition"]].to_dict(
        orient="records"
    ) == [
        {"sample_id": "CTRL_1", "subject_id": "P1", "condition": "CTRL"},
        {"sample_id": "IZ_1", "subject_id": "P2", "condition": "IZ"},
        {"sample_id": "IZ_2", "subject_id": "P3", "condition": "IZ"},
    ]


def test_prepare_kuppe_ctrl_iz_rejects_fractional_raw_counts() -> None:
    with pytest.raises(ValueError, match="non-integer"):
        module._integer_counts(sparse.csc_matrix([[1.0, 0.5]]))
