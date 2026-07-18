from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.cellchat.run_condition_aware import (
    METHOD_COMMIT,
    METHOD_VERSION,
    RANKING_COLUMNS,
    RESOURCE_ID,
    RESOURCE_ROWS,
    _input_manifest_sha256,
    _prepare_metadata,
    _validate_environment_manifest,
    _validate_rankings,
    _validate_resource,
    _validate_selected_and_rankings,
)
from scipy import sparse


def _ranking_table(
    *,
    cell_types: list[str],
    target: str = "case",
    reference: str = "control",
    not_estimable: set[str] | None = None,
) -> pd.DataFrame:
    excluded = set() if not_estimable is None else not_estimable
    records: list[dict[str, object]] = []
    for condition in (target, reference):
        for left, sender in enumerate(cell_types):
            for receiver in cell_types[left:]:
                estimable = sender not in excluded and receiver not in excluded
                records.append(
                    {
                        "dataset": "toy",
                        "method": "cellchat_condition_aware",
                        "method_version": METHOD_VERSION,
                        "resource": RESOURCE_ID,
                        "ranking_semantics": "cardinality_test",
                        "condition": condition,
                        "sender": sender,
                        "receiver": receiver,
                        "ranked_strength": 0 if estimable else np.nan,
                        "status": "observed" if estimable else "not_estimable",
                        "reason_code": (
                            "" if estimable else "cell_type_not_supported"
                        ),
                    }
                )
    return pd.DataFrame.from_records(records, columns=RANKING_COLUMNS)


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
            {"output": "input.h5ad", "output_sha256": digest},
            "input.h5ad",
        )
        == digest
    )
    with pytest.raises(ValueError, match="does not bind"):
        _input_manifest_sha256({}, "input.h5ad")


def test_prepare_metadata_preserves_condition_specific_cell_types(
    tmp_path: Path,
) -> None:
    obs = pd.DataFrame(
        {
            "cell_type": ["A"] * 6 + ["B"] * 3 + ["A"] * 6 + ["B"],
            "condition": ["case"] * 9 + ["control"] * 7,
        },
        index=[f"cell-{index}" for index in range(16)],
    )
    source = ad.AnnData(
        X=sparse.csr_matrix(np.ones((16, 3))),
        obs=obs,
        var=pd.DataFrame(index=["L", "R", "G"]),
    )
    path = tmp_path / "input.h5ad"
    source.write_h5ad(path)

    metadata, support, audit = _prepare_metadata(
        path,
        cell_type_key="cell_type",
        condition_key="condition",
        target="case",
        reference="control",
        min_cells=3,
    )

    assert list(metadata.columns) == ["cell", "cell_type", "condition"]
    assert set(metadata["cell_type"]) == {"A", "B"}
    assert len(metadata) == 16
    assert len(support) == 4
    b = support.loc[support["cell_type"].eq("B")].set_index("condition")
    assert bool(b.loc["case", "condition_present"])
    assert bool(b.loc["control", "condition_present"])
    assert not bool(b.loc["case", "native_network_supported"])
    assert not bool(b.loc["control", "native_network_supported"])
    assert b["de_comparable"].all()
    assert audit["de_comparable_cell_types"] == ["A", "B"]
    assert audit["absent_cell_types_by_condition"] == {
        "case": [],
        "control": [],
    }
    assert audit["cells_by_condition"] == {"case": 9, "control": 7}


def test_prepare_metadata_distinguishes_absent_and_low_support(
    tmp_path: Path,
) -> None:
    rows = [
        (condition, cell_type)
        for condition, counts in {
            "case": {"A": 12, "BC": 12, "SC": 12},
            "control": {"A": 12, "BC": 0, "SC": 8},
        }.items()
        for cell_type, count in counts.items()
        for _ in range(count)
    ]
    obs = pd.DataFrame(
        rows,
        columns=["condition", "cell_type"],
        index=[f"cell-{index}" for index in range(len(rows))],
    )
    source = ad.AnnData(
        X=sparse.csr_matrix(np.ones((len(obs), 3))),
        obs=obs,
        var=pd.DataFrame(index=["L", "R", "G"]),
    )
    path = tmp_path / "asymmetric.h5ad"
    source.write_h5ad(path)

    metadata, support, audit = _prepare_metadata(
        path,
        cell_type_key="cell_type",
        condition_key="condition",
        target="case",
        reference="control",
        min_cells=10,
    )

    indexed = support.set_index(["condition", "cell_type"])
    assert len(metadata) == len(obs)
    assert not bool(indexed.loc[("control", "BC"), "condition_present"])
    assert not bool(
        indexed.loc[("control", "BC"), "native_network_supported"]
    )
    assert bool(indexed.loc[("control", "SC"), "condition_present"])
    assert not bool(
        indexed.loc[("control", "SC"), "native_network_supported"]
    )
    assert bool(indexed.loc[("case", "SC"), "native_network_supported"])
    assert not bool(indexed.loc[("case", "BC"), "de_comparable"])
    assert bool(indexed.loc[("control", "SC"), "de_comparable"])
    assert audit["absent_cell_types_by_condition"] == {
        "case": [],
        "control": ["BC"],
    }
    assert audit["native_structural_zero_cell_types_by_condition"] == {
        "case": [],
        "control": ["SC"],
    }


def test_validate_environment_manifest_pins_paper_commit(tmp_path: Path) -> None:
    path = tmp_path / "environment.json"
    payload = {
        "cellchat": {"version": METHOD_VERSION, "git_commit": METHOD_COMMIT},
        "paper_protocol": {"supplementary_section": "S4"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _validate_environment_manifest(path) == payload
    payload["cellchat"]["git_commit"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="paper commit"):
        _validate_environment_manifest(path)


def test_validate_resource_requires_frozen_complete_cellchat_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "connectomedb2020.tsv"
    table = pd.DataFrame(
        {
            "harmonized_interaction_id": [f"id-{i}" for i in range(RESOURCE_ROWS)],
            "ligand": [f"L{i}" for i in range(RESOURCE_ROWS)],
            "receptor": [f"R{i}" for i in range(RESOURCE_ROWS)],
            "cellchat_source_interaction_id": [
                f"id-{i}" for i in range(RESOURCE_ROWS)
            ],
            "cellchat_covered": [True] * RESOURCE_ROWS,
        }
    )
    table.to_csv(resource, sep="\t", index=False)
    digest = hashlib.sha256(resource.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "benchmarks.adapters.cellchat.run_condition_aware.RESOURCE_SHA256",
        digest,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"resource_id": RESOURCE_ID, "payload": {"sha256": digest}}),
        encoding="utf-8",
    )

    assert _validate_resource(resource, manifest)["resource_id"] == RESOURCE_ID
    table.loc[0, "cellchat_covered"] = False
    table.to_csv(resource, sep="\t", index=False)
    changed_digest = hashlib.sha256(resource.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "benchmarks.adapters.cellchat.run_condition_aware.RESOURCE_SHA256",
        changed_digest,
    )
    manifest.write_text(
        json.dumps(
            {
                "resource_id": RESOURCE_ID,
                "payload": {"sha256": changed_digest},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cover all"):
        _validate_resource(resource, manifest)


def test_validate_rankings_requires_exact_full_unordered_axis(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rankings.tsv"
    table = _ranking_table(cell_types=["A", "B", "C"], not_estimable={"C"})
    table.to_csv(path, sep="\t", index=False)

    observed = _validate_rankings(
        path,
        dataset_id="toy",
        target="case",
        reference="control",
        cell_types=["A", "B", "C"],
    )

    assert len(observed) == 12
    assert observed["status"].eq("not_estimable").sum() == 6
    table.loc[0, "receiver"] = "missing"
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="pair labels"):
        _validate_rankings(
            path,
            dataset_id="toy",
            target="case",
            reference="control",
            cell_types=["A", "B", "C"],
        )


def test_validate_selected_rows_recomputes_ranking_cardinality(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "resource.tsv"
    pd.DataFrame(
        {
            "harmonized_interaction_id": ["lr-1", "lr-2"],
            "cellchat_source_interaction_id": ["lr-1", "lr-2"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
        }
    ).to_csv(resource, sep="\t", index=False)
    selected = tmp_path / "selected.tsv.gz"
    pd.DataFrame(
        {
            "condition": ["case", "case", "control"],
            "source": ["A", "B", "A"],
            "target": ["B", "A", "B"],
            "harmonized_interaction_id": ["lr-1", "lr-2", "lr-1"],
            "interaction_name": ["lr-1", "lr-2", "lr-1"],
            "ligand": ["L1", "L2", "L1"],
            "receptor": ["R1", "R2", "R1"],
            "datasets": ["case", "case", "control"],
        }
    ).to_csv(selected, sep="\t", index=False, compression="gzip")
    rankings = _ranking_table(cell_types=["A", "B"])
    pair = rankings["sender"].eq("A") & rankings["receiver"].eq("B")
    rankings.loc[pair & rankings["condition"].eq("case"), "ranked_strength"] = 2
    rankings.loc[
        pair & rankings["condition"].eq("control"), "ranked_strength"
    ] = 1
    support = pd.DataFrame(
        [
            {
                "condition": condition,
                "cell_type": cell_type,
                "condition_present": True,
                "native_network_supported": True,
                "de_comparable": True,
            }
            for condition in ("case", "control")
            for cell_type in ("A", "B")
        ]
    )

    observed = _validate_selected_and_rankings(
        selected,
        rankings,
        resource_path=resource,
        target="case",
        reference="control",
        support=support,
        sensitivity=False,
    )

    assert len(observed) == 3
    rankings.loc[pair & rankings["condition"].eq("case"), "ranked_strength"] = 1
    with pytest.raises(ValueError, match="cardinality"):
        _validate_selected_and_rankings(
            selected,
            rankings,
            resource_path=resource,
            target="case",
            reference="control",
            support=support,
            sensitivity=False,
        )


def test_condition_specific_status_distinguishes_absence_and_native_zero(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "resource.tsv"
    pd.DataFrame(
        {
            "harmonized_interaction_id": ["lr-1"],
            "cellchat_source_interaction_id": ["lr-1"],
            "ligand": ["L1"],
            "receptor": ["R1"],
        }
    ).to_csv(resource, sep="\t", index=False)
    selected = tmp_path / "selected.tsv.gz"
    pd.DataFrame(
        columns=[
            "condition",
            "source",
            "target",
            "harmonized_interaction_id",
            "interaction_name",
            "ligand",
            "receptor",
            "datasets",
        ]
    ).to_csv(selected, sep="\t", index=False, compression="gzip")
    support = pd.DataFrame(
        [
            ("case", "A", True, True, True),
            ("case", "BC", True, True, False),
            ("case", "SC", True, True, True),
            ("control", "A", True, True, True),
            ("control", "BC", False, False, False),
            ("control", "SC", True, False, True),
        ],
        columns=[
            "condition",
            "cell_type",
            "condition_present",
            "native_network_supported",
            "de_comparable",
        ],
    )
    records: list[dict[str, object]] = []
    for condition in ("case", "control"):
        condition_support = support.loc[
            support["condition"].eq(condition)
        ].set_index("cell_type")
        cell_types = ["A", "BC", "SC"]
        for left, sender in enumerate(cell_types):
            for receiver in cell_types[left:]:
                sender_row = condition_support.loc[sender]
                receiver_row = condition_support.loc[receiver]
                observed = bool(
                    sender_row["condition_present"]
                    and receiver_row["condition_present"]
                    and (
                        sender_row["de_comparable"]
                        or receiver_row["de_comparable"]
                    )
                )
                structural = bool(
                    observed
                    and (
                        not sender_row["native_network_supported"]
                        or not receiver_row["native_network_supported"]
                    )
                )
                if not observed:
                    reason = (
                        "cell_type_absent_in_condition"
                        if not sender_row["condition_present"]
                        or not receiver_row["condition_present"]
                        else "no_de_comparable_ligand_direction"
                    )
                elif structural:
                    reason = "native_structural_zero_low_cell_support"
                elif bool(sender_row["de_comparable"]) != bool(
                    receiver_row["de_comparable"]
                ):
                    reason = "partial_directional_de_support"
                else:
                    reason = ""
                records.append(
                    {
                        "dataset": "toy",
                        "method": "cellchat_condition_aware",
                        "method_version": METHOD_VERSION,
                        "resource": RESOURCE_ID,
                        "ranking_semantics": "condition_specific_test",
                        "condition": condition,
                        "sender": sender,
                        "receiver": receiver,
                        "ranked_strength": 0 if observed else np.nan,
                        "status": "observed" if observed else "not_estimable",
                        "reason_code": reason,
                    }
                )
    rankings = pd.DataFrame.from_records(records, columns=RANKING_COLUMNS)

    _validate_selected_and_rankings(
        selected,
        rankings,
        resource_path=resource,
        target="case",
        reference="control",
        support=support,
        sensitivity=False,
    )

    indexed = rankings.set_index(["condition", "sender", "receiver"])
    assert indexed.loc[("case", "A", "BC"), "status"] == "observed"
    assert indexed.loc[("case", "BC", "BC"), "status"] == "not_estimable"
    assert indexed.loc[("control", "A", "BC"), "status"] == "not_estimable"
    assert indexed.loc[("control", "A", "SC"), "status"] == "observed"
    assert (
        indexed.loc[("control", "A", "SC"), "reason_code"]
        == "native_structural_zero_low_cell_support"
    )


def test_r_runner_contains_exact_cellchat_s4_protocol() -> None:
    source = (
        Path(__file__).parents[2]
        / "benchmarks/adapters/cellchat/run_condition_aware.R"
    ).read_text(encoding="utf-8")

    assert 'computeCommunProb(object, type = "triMean")' in source
    assert (
        "identifyOverExpressedGenes(object, min.cells = min_cells)" in source
    )
    assert "thresh.p = 1" not in source
    assert "mergeCellChat(" in source
    assert 'group.dataset = "datasets"' in source
    assert "pos.dataset = target" in source
    assert "only.pos = FALSE" in source
    assert "thresh.pc = 0.1" in source
    assert "thresh.fc = 0.05" in source
    assert "thresh.p = 0.05" in source
    assert "group.DE.combined = FALSE" in source
    assert "variable.all = TRUE" in source
    assert "ligand_logfc = 0.05" in source
    assert "ligand_logfc = -0.05" in source
    assert "receptor_logfc = NULL" in source
    assert "receptor_logfc = 0.05" in source
    assert "receptor_logfc = -0.05" in source
    assert "liftCellChat(object, group.new = all_cell_types)" in source
    assert "idents.use = de_comparable_cell_types" in source
