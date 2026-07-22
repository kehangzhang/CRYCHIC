from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from benchmarks.comprehensive.audit import (
    REGISTRY_COLUMNS,
    RegistryError,
    build_execution_matrix,
    inspect_asset,
    load_and_validate_registries,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = REPO_ROOT / "benchmarks" / "comprehensive"


def _copy_registry(tmp_path: Path) -> Path:
    destination = tmp_path / "registry"
    destination.mkdir()
    for name in REGISTRY_COLUMNS:
        shutil.copy2(REGISTRY_ROOT / f"{name}.tsv", destination)
    return destination


def test_frozen_comprehensive_registry_is_internally_consistent() -> None:
    registries = load_and_validate_registries(REGISTRY_ROOT)

    assert len(registries["panels"]) == 13
    assert "misc_olink" in registries["datasets"]
    assert "multinichenet" in registries["methods"]
    assert registries["methods"]["nichenet_prior"]["historical_exact"] == "false"
    assert (
        registries["methods"]["nichenet_native"]["implementation_status"] == "planned"
    )
    assert "cytosig_2021" in registries["literature"]
    assert "liana_historic" in registries["vendor_sources"]
    required_negative_controls = {
        "global_null",
        "abundance_only",
        "receiver_autonomous",
        "ligand_only",
        "target_only",
        "composition_imbalance",
        "graph_smooth",
        "topology_jump",
        "wrong_topology",
        "disconnected_graph",
        "collinear_lr",
        "structural_absence",
        "batch_context_confounding",
        "annotation_perturbation",
        "prior_replacement",
        "legal_context_permutation",
        "fixed_subject_increasing_cells_null",
        "context_correlated_sampling_missingness",
        "scoring_functional_null",
    }
    assert required_negative_controls <= set(registries["simulation_scenarios"])
    assert (
        registries["simulation_scenarios"]["batch_context_confounding"]["estimable"]
        == "false"
    )
    sample_figure3 = set(
        registries["panels"]["P06_legacy_figure3_sample"]["methods"].split(";")
    )
    pooled_figure3 = set(
        registries["panels"]["P07_legacy_figure3_pooled"]["methods"].split(";")
    )
    assert {"liana_plus_de", "multinichenet", "scseqcommdiff_sample"} <= (
        sample_figure3
    )
    assert {"cellchat_pooled", "scdiffcom_pooled", "scseqcommdiff_pooled"} <= (
        pooled_figure3
    )
    for panel in registries["panels"].values():
        primary_metrics = panel["primary_metrics"].split(";")
        assert all(
            registries["metrics"][metric_id]["track"] == panel["track"]
            for metric_id in primary_metrics
        )


def test_registry_rejects_unknown_panel_method_reference(tmp_path: Path) -> None:
    registry = _copy_registry(tmp_path)
    panels = registry / "panels.tsv"
    contents = panels.read_text(encoding="utf-8")
    panels.write_text(
        contents.replace(
            "crychic_rc9;cellchat_sample",
            "unknown_method;cellchat_sample",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match=r"unknown method IDs.*unknown_method"):
        load_and_validate_registries(registry)


def test_asset_inspection_detects_aria2_sidecar_as_partial(tmp_path: Path) -> None:
    target = tmp_path / "cohort.h5ad"
    target.write_bytes(b"incomplete payload")
    Path(f"{target}.aria2").write_bytes(b"aria2 control")

    audit = inspect_asset(tmp_path, "cohort.h5ad")

    assert audit["exists"] == "true"
    assert audit["asset_state"] == "partial"
    assert audit["aria2_sidecars"] == "1"


def test_execution_matrix_keeps_unavailable_states_separate_from_zero() -> None:
    registries = load_and_validate_registries(REGISTRY_ROOT)
    dataset_assets = {dataset_id: "complete" for dataset_id in registries["datasets"]}
    method_readiness = {
        method_id: method["implementation_status"]
        for method_id, method in registries["methods"].items()
    }
    method_readiness["scdiffcom_pooled"] = "reusable"
    rows, readiness = build_execution_matrix(
        registries,
        dataset_assets=dataset_assets,
        method_readiness=method_readiness,
    )

    assert {row["score_as_zero"] for row in rows} == {"false"}
    multinichenet = [
        row
        for row in rows
        if row["panel_id"] == "P04_misc_olink"
        and row["method_id"] == "multinichenet"
        and row["resource_mode"] == "native"
    ]
    assert len(multinichenet) == 1
    assert multinichenet[0]["execution_status"] == "skipped"
    assert multinichenet[0]["reason_code"] == "method_adapter_or_environment_planned"

    reusable = [
        row
        for row in rows
        if row["panel_id"] == "P07_legacy_figure3_pooled"
        and row["method_id"] == "scdiffcom_pooled"
    ]
    assert len(reusable) == 2
    assert {row["dataset_id"] for row in reusable} == {"kuppe_mi", "lerma_ms"}
    assert {row["execution_status"] for row in reusable} == {"reusable"}
    assert {row["reason_code"] for row in reusable} == {"frozen_output_only"}

    p04 = next(row for row in readiness if row["panel_id"] == "P04_misc_olink")
    assert int(p04["matrix_rows"]) > int(p04["method_dataset_pairs"])
    assert int(p04["eligible_pairs"]) < int(p04["method_dataset_pairs"])
