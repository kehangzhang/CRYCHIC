from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import load_harmonized_resource, sha256_file
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.export_cscc_crossmethod_smoke_inputs import (
    H5AD_FILENAME,
    MANIFEST_FILENAME,
    RESOURCE_FILENAME,
    build_export_manifest,
    select_harmonized_rows,
    write_crossmethod_inputs,
)
from scipy import sparse


def _resource_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "harmonized_interaction_id": ["id_c", "id_a", "id_b"],
            "ligand": ["L3", "L1", "L2"],
            "receptor": ["R3", "R1", "R2"],
            "cellchat_source_interaction_id": ["cc3", "cc1", "cc2"],
            "cellphonedb_source_interaction_id": ["cp3", "cp1", "cp2"],
        }
    )


def _source_manifest() -> dict[str, object]:
    return {
        "schema_version": "crychic-harmonized-lr-v1",
        "resource_id": "toy_h_common",
        "version": "1",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "license": "test-only",
        "citation": ["Toy resource"],
    }


def _adata() -> ad.AnnData:
    counts = sparse.csr_matrix(np.asarray([[1, 0, 2], [0, 3, 1]], dtype=np.int32))
    obs = pd.DataFrame(
        {"subject_id": ["s1", "s2"], "condition": ["Normal", "Tumor"]},
        index=["cell_b", "cell_a"],
    )
    var = pd.DataFrame({"feature_type": ["gene"] * 3}, index=["L1", "R1", "X"])
    result = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=var)
    result.layers["counts"] = counts
    return result


def test_exact_selection_uses_requested_order_not_source_order() -> None:
    source = _resource_table()

    selected = select_harmonized_rows(source, ("id_b", "id_a"))
    shuffled = select_harmonized_rows(
        source.sample(frac=1.0, random_state=7), ("id_b", "id_a")
    )

    assert selected["harmonized_interaction_id"].tolist() == ["id_b", "id_a"]
    pd.testing.assert_frame_equal(selected, shuffled)
    assert source["harmonized_interaction_id"].tolist() == ["id_c", "id_a", "id_b"]


def test_selection_rejects_missing_interaction() -> None:
    with pytest.raises(ValueError, match="lacks requested IDs"):
        select_harmonized_rows(_resource_table(), ("id_a", "missing"))


def test_selection_rejects_duplicate_request() -> None:
    with pytest.raises(
        ValueError, match="requested interaction IDs contain duplicates"
    ):
        select_harmonized_rows(_resource_table(), ("id_a", "id_a"))


def test_selection_rejects_duplicate_source_id() -> None:
    source = pd.concat([_resource_table(), _resource_table().iloc[[1]]])
    with pytest.raises(ValueError, match="source contains duplicate IDs"):
        select_harmonized_rows(source, ("id_a",))


def test_manifest_builder_rejects_inconsistent_row_count() -> None:
    with pytest.raises(ValueError, match="row count"):
        build_export_manifest(
            source_manifest=_source_manifest(),
            source_table_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            table_payload={"filename": RESOURCE_FILENAME, "rows": 1},
            dataset_payload={"filename": H5AD_FILENAME},
            selected_interaction_ids=("id_a", "id_b"),
        )


def test_small_bundle_preserves_h5ad_and_resource_checksum_readback(
    tmp_path: Path,
) -> None:
    selected = select_harmonized_rows(_resource_table(), ("id_b", "id_a"))
    adata = _adata()

    result = write_crossmethod_inputs(
        adata=adata,
        selected_resource=selected,
        source_manifest=_source_manifest(),
        source_table_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        selected_interaction_ids=("id_b", "id_a"),
        output_dir=tmp_path,
    )

    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["manifest_sha256"] == sha256_file(manifest_path)
    assert manifest["payload"]["sha256"] == sha256_file(tmp_path / RESOURCE_FILENAME)
    assert manifest["analysis_input"]["sha256"] == sha256_file(tmp_path / H5AD_FILENAME)

    table, loaded_manifest = load_harmonized_resource(
        tmp_path / RESOURCE_FILENAME, manifest_path
    )
    bundle = harmonized_resource_bundle(tmp_path / RESOURCE_FILENAME, manifest_path)
    assert loaded_manifest == manifest
    assert table["harmonized_interaction_id"].tolist() == ["id_b", "id_a"]
    assert [item.interaction_id for item in bundle.interactions] == ["id_a", "id_b"]

    readback = ad.read_h5ad(tmp_path / H5AD_FILENAME)
    assert readback.X is not None
    assert sparse.isspmatrix_csr(readback.X)
    assert (readback.X != adata.X).nnz == 0
    assert tuple(readback.obs_names) == tuple(adata.obs_names)
    assert tuple(readback.var_names) == tuple(adata.var_names)
    assert tuple(readback.obs.columns) == tuple(adata.obs.columns)
    assert tuple(readback.var.columns) == tuple(adata.var.columns)
    pd.testing.assert_frame_equal(readback.obs, adata.obs)
    pd.testing.assert_frame_equal(readback.var, adata.var)
    assert sparse.isspmatrix_csr(readback.layers["counts"])
    assert readback.layers["counts"].dtype == np.dtype("int32")

    tampered = tmp_path / "tampered.tsv"
    tampered.write_bytes((tmp_path / RESOURCE_FILENAME).read_bytes() + b"\n")
    tampered_manifest = dict(manifest)
    tampered_manifest["payload"] = dict(manifest["payload"])
    tampered_manifest["payload"]["filename"] = tampered.name
    tampered_manifest_path = tmp_path / "tampered_manifest.json"
    tampered_manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_harmonized_resource(tampered, tampered_manifest_path)

    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        == result["manifest_sha256"]
    )
