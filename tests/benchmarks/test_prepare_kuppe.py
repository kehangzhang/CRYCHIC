from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import anndata as ad
import benchmarks.datasets.prepare_kuppe as module
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


def _expected_fixture() -> dict[str, object]:
    return {
        "n_obs": 6,
        "n_vars": 4,
        "n_subjects": 4,
        "n_samples": 5,
        "regions": ["CTRL", "RZ", "BZ", "IZ", "FZ"],
        "n_cell_types": 3,
        "cell_types": ["Cardiomyocyte", "Fibroblast", "Myeloid"],
        "subjects_by_region": {"CTRL": 1, "RZ": 1, "BZ": 1, "IZ": 1, "FZ": 1},
        "samples_by_region": {"CTRL": 1, "RZ": 1, "BZ": 1, "IZ": 1, "FZ": 1},
        "multi_region_subjects": {"P2": 2},
        "replicate_samples": {},
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
        ],
        dtype=np.int32,
    )
    normalized = np.log1p(counts.astype(np.float64))
    sample = ["S_CTRL", "S_CTRL", "S_RZ", "S_BZ", "S_IZ", "S_FZ"]
    patient = ["P1", "P1", "P2", "P2", "P3", "P4"]
    region = ["CTRL", "CTRL", "RZ", "BZ", "IZ", "FZ"]
    cell_type = [
        "Cardiomyocyte",
        "Fibroblast",
        "Cardiomyocyte",
        "Fibroblast",
        "Myeloid",
        "Myeloid",
    ]
    obs = pd.DataFrame(
        {
            "sample": sample,
            "n_counts": counts.sum(axis=1).astype(float),
            "n_genes": np.count_nonzero(counts, axis=1).astype(np.int32),
            "patient_region_id": [
                f"{p}_{r}" for p, r in zip(patient, region, strict=True)
            ],
            "patient": patient,
            "patient_group": [
                "control",
                "myogenic",
                "myogenic",
                "ischemic",
                "fibrotic",
                "fibrotic",
            ],
            "major_labl": region,
            "cell_type_original": cell_type,
        },
        index=pd.Index([f"cell_{index}" for index in range(6)], name="cell_id"),
    )
    var = pd.DataFrame(
        {"feature_biotype": ["gene"] * 4, "feature_is_filtered": [False] * 4},
        index=pd.Index(["G1", "G2", "G3", "G4"], name="feature_id"),
    )
    source = ad.AnnData(X=sparse.csc_matrix(normalized), obs=obs, var=var)
    source.raw = ad.AnnData(
        X=sparse.csc_matrix(counts.astype(np.float32)),
        obs=obs.copy(),
        var=var.copy(),
    )
    source.uns["title"] = "Kuppe fixture"
    source.uns["schema_version"] = "fixture"
    source.uns["X_normalization"] = "log1p"
    source.write_h5ad(path)
    return counts, normalized


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_kuppe_preserves_counts_aliases_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.h5ad"
    counts, normalized = _write_fixture(source_path)
    monkeypatch.setattr(module, "EXPECTED_KUPPE", _expected_fixture())

    output_path = tmp_path / "prepared.h5ad"
    result = module.prepare_kuppe(source_path, output_path)

    assert result.shape == (6, 4)
    assert sparse.isspmatrix_csr(result.X)
    assert sparse.isspmatrix_csr(result.layers["counts"])
    assert result.layers["counts"].dtype == np.int32
    assert np.array_equal(result.layers["counts"].toarray(), counts)
    assert np.array_equal(result.X.toarray(), normalized)
    assert result.obs["sample_id"].tolist() == result.obs["sample"].astype(str).tolist()
    assert result.obs["subject_id"].tolist() == (
        result.obs["patient"].astype(str).tolist()
    )
    assert result.obs["region"].tolist() == (
        result.obs["major_labl"].astype(str).tolist()
    )
    assert result.obs["cell_type"].tolist() == (
        result.obs["cell_type_original"].astype(str).tolist()
    )
    assert result.obs["broad_cell_type"].tolist() == result.obs["cell_type"].tolist()
    assert result.uns["crychic_design_audit"]["response_status"] == "not_estimable"
    assert result.uns["crychic_design_audit"]["availability_status"] == (
        "eligible_descriptive_only"
    )

    conversion_path = output_path.with_suffix(".conversion.json")
    summary = json.loads(conversion_path.read_text())
    assert summary["source"]["sha256"] == _sha256(source_path)
    assert summary["output"]["sha256"] == _sha256(output_path)
    assert summary["provenance"]["counts_lineage"].startswith("source raw.X")
    for suffix in (
        "design_audit.json",
        "sample_design.tsv",
        "contrast_design.tsv",
        "cell_type_support.tsv",
    ):
        assert (tmp_path / f"prepared.{suffix}").is_file()

    round_trip = ad.read_h5ad(output_path)
    assert np.array_equal(round_trip.layers["counts"].toarray(), counts)
    assert round_trip.obs["broad_cell_type"].tolist() == cell_types_as_list(result)


def test_audit_kuppe_reports_mixed_design_without_loading_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.h5ad"
    _write_fixture(source_path)
    monkeypatch.setattr(module, "EXPECTED_KUPPE", _expected_fixture())

    audit_dir = tmp_path / "audit"
    result = module.audit_kuppe(source_path, audit_dir)

    audit = cast(dict[str, Any], result["audit"])
    assert audit["design_class"] == "mixed_paired_unpaired_multi_region"
    assert audit["response_analysis"]["status"] == "not_estimable"
    assert audit["availability_analysis"]["status"] == ("eligible_descriptive_only")
    assert audit["lineage"]["hash_status"] == "deferred_to_full_conversion"

    contrasts = pd.read_csv(
        audit_dir / "source.contrast_design.tsv", sep="\t", keep_default_na=False
    )
    rz_bz = contrasts.loc[
        (contrasts["reference_region"] == "RZ")
        & (contrasts["comparison_region"] == "BZ")
    ].iloc[0]
    assert rz_bz["pairwise_design"] == "fully_paired"
    assert rz_bz["n_paired_subjects"] == 1
    assert set(contrasts["full_dataset_response_status"]) == {"not_estimable"}


def test_integer_counts_rejects_fractional_values() -> None:
    matrix = sparse.csc_matrix(np.asarray([[1.0, 0.5]]))
    with pytest.raises(ValueError, match="non-integer"):
        module._integer_counts(matrix)


def cell_types_as_list(adata: ad.AnnData) -> list[str]:
    return [str(value) for value in adata.obs["cell_type"].tolist()]
