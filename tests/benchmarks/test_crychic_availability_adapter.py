from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import sha256_file, write_json
from benchmarks.adapters.crychic import run_availability as adapter

from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
)


def _input(path: Path) -> Path:
    obs = pd.DataFrame(
        {
            "sample_id": ["s1"] * 4,
            "subject_id": ["p1"] * 4,
            "cell_type": ["Sender", "Sender", "Receiver", "Receiver"],
            "condition": ["stim"] * 4,
        },
        index=[f"cell-{index}" for index in range(4)],
    )
    counts = np.array(
        [
            [12, 1, 8, 1],
            [10, 1, 7, 1],
            [1, 11, 1, 9],
            [1, 10, 1, 8],
        ],
        dtype=np.int64,
    )
    adata = ad.AnnData(
        X=np.log1p(counts.astype(float)),
        obs=obs,
        var=pd.DataFrame(index=["L1", "R1", "L2", "R2"]),
    )
    adata.layers["counts"] = counts
    adata.write_h5ad(path)
    return path


def _harmonized(tmp_path: Path) -> tuple[Path, Path]:
    table_path = tmp_path / "harmonized_lr.tsv"
    pd.DataFrame(
        {
            "harmonized_interaction_id": ["h1", "h2"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "cellchat_source_interaction_id": ["cc1", "cc2"],
            "cellphonedb_source_interaction_id": ["cp1", "cp2"],
        }
    ).to_csv(table_path, sep="\t", index=False)
    manifest_path = tmp_path / "harmonized_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "crychic-harmonized-lr-v1-synthetic-fixture",
            "resource_id": "toy_hcommon",
            "version": "1",
            "species": "human",
            "gene_namespace": "HGNC symbol",
            "license": "CC0",
            "citation": "Synthetic fixture",
            "payload": {
                "filename": table_path.name,
                "sha256": sha256_file(table_path),
            },
        },
    )
    return table_path, manifest_path


def _interaction() -> Interaction:
    return Interaction(
        interaction_id="native-i1",
        source_interaction_id="source-i1",
        ligand_name="native-display-ligand",
        receptor_name="native-display-receptor",
        ligand_subunits=("L1",),
        receptor_subunits=("R1",),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="native-toy",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _native_bundle() -> ResourceBundle:
    return ResourceBundle(
        resource_id="native-toy",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(_interaction(),),
        mapping_report=MappingReport(1, 1, 2),
        manifest_digest="d" * 64,
        source_files=("native.tsv",),
        license="CC0",
        citation="Synthetic native fixture",
    )


def test_hcommon_adapter_calls_public_core_and_records_static_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _input(tmp_path / "input.h5ad")
    resource_path, resource_manifest = _harmonized(tmp_path)
    calls: list[str] = []
    original_validate = adapter.validate_anndata
    original_aggregate = adapter.aggregate_pseudobulk
    original_estimate = adapter.estimate_bundle_availability

    def validate_spy(*args, **kwargs):
        calls.append("validate_anndata")
        return original_validate(*args, **kwargs)

    def aggregate_spy(*args, **kwargs):
        calls.append("aggregate_pseudobulk")
        return original_aggregate(*args, **kwargs)

    def estimate_spy(*args, **kwargs):
        calls.append("estimate_bundle_availability")
        return original_estimate(*args, **kwargs)

    monkeypatch.setattr(adapter, "validate_anndata", validate_spy)
    monkeypatch.setattr(adapter, "aggregate_pseudobulk", aggregate_spy)
    monkeypatch.setattr(adapter, "estimate_bundle_availability", estimate_spy)
    output = tmp_path / "out-hcommon"
    manifest = adapter.run_availability(
        input_path,
        output,
        dataset_id="toy-static",
        resource_mode="H-common",
        context_keys=("condition",),
        min_cells=2,
        harmonized_resource=resource_path,
        harmonized_manifest=resource_manifest,
    )

    assert calls == [
        "validate_anndata",
        "aggregate_pseudobulk",
        "estimate_bundle_availability",
    ]
    assert manifest["status"] == "complete"
    assert manifest["method"]["id"] == adapter.METHOD_ID
    assert manifest["score_semantics"] == {
        "name": "availability_state",
        "direction": "higher",
        "interpretation": "static_sample_level_availability_diagnostic",
        "probability": False,
        "comm_strength": False,
        "full_differential_comm_strength": False,
        "differential_communication": False,
        "single_sample_estimable": True,
        "observed_zero_is_valid": True,
        "unreturned_rows_imputed_as_zero": False,
        "row_scope": "core_state_eligible_mapped_supported_rows",
        "p_value": None,
        "q_value": None,
        "within_dataset_p_value": "not_emitted",
    }
    assert manifest["statistical_scope"]["single_sample_estimable"] is True
    assert manifest["statistical_scope"]["between_condition_inference"] is False
    assert manifest["resource"]["mode"] == "H-common"
    assert manifest["resource"]["table_sha256"] == sha256_file(resource_path)
    assert manifest["core_execution"]["entrypoints"] == [
        "crychic.data.validate_anndata",
        "crychic.pseudobulk.aggregate_pseudobulk",
        "crychic.availability.estimate_bundle_availability",
    ]
    table = pd.read_parquet(output / "availability_scores.parquet")
    tsv = pd.read_csv(output / "availability_scores.tsv", sep="\t")
    assert tuple(table.columns) == adapter.SCORE_COLUMNS
    assert len(table) == 8
    assert set(table["sample_id"]) == {"s1"}
    assert set(table["sender"]) == {"Sender", "Receiver"}
    assert set(table["receiver"]) == {"Sender", "Receiver"}
    assert set(table["ligand"]) == {"L1", "L2"}
    assert set(table["receptor"]) == {"R1", "R2"}
    assert set(table["status"]) == {"observed"}
    assert table["availability_state"].between(0, 1).all()
    assert table["receptor_availability"].between(0, 1).all()
    assert "comm_strength" not in table.columns
    pd.testing.assert_series_equal(
        table["availability_state"],
        tsv["availability_state"],
        check_names=False,
        check_exact=False,
        rtol=1e-12,
    )
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stored["output"]["table_sha256"] == sha256_file(
        output / "availability_scores.parquet"
    )


def test_native_resource_branch_records_adapter_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _input(tmp_path / "input-native.h5ad")
    database_root = tmp_path / "database"
    database_root.mkdir()
    resource_manifest = tmp_path / "native-manifest.json"
    resource_manifest.write_text("{}\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_loader(adapter_name: str, **kwargs) -> ResourceBundle:
        observed["adapter"] = adapter_name
        observed.update(kwargs)
        return _native_bundle()

    monkeypatch.setattr(adapter, "load_native_resource_bundle", fake_loader)
    output = tmp_path / "out-native"
    manifest = adapter.run_availability(
        input_path,
        output,
        dataset_id="toy-native",
        resource_mode="native",
        context_keys=("condition",),
        min_cells=2,
        native_adapter="cellchat",
        database_root=database_root,
        resource_manifest=resource_manifest,
    )

    assert observed["adapter"] == "cellchat"
    assert observed["database_root"] == database_root.resolve()
    assert observed["manifest_path"] == resource_manifest.resolve()
    assert observed["species"] is Species.HUMAN
    assert manifest["resource"]["mode"] == "native"
    assert manifest["resource"]["native_adapter"] == "cellchat"
    assert manifest["resource"]["manifest_sha256"] == sha256_file(resource_manifest)
    table = pd.read_parquet(output / "availability_scores.parquet")
    assert len(table) == 4
    assert set(table["resource_id"]) == {"native-toy"}
    assert set(table["native_interaction_id"]) == {"source-i1"}
    assert set(table["ligand"]) == {"L1"}
    assert set(table["receptor"]) == {"R1"}


def test_resource_arguments_fail_closed(tmp_path: Path) -> None:
    input_path = _input(tmp_path / "input-invalid.h5ad")
    with pytest.raises(ValueError, match="H-common requires"):
        adapter.run_availability(
            input_path,
            tmp_path / "out-invalid",
            dataset_id="toy",
            resource_mode="H-common",
            context_keys=("condition",),
        )
