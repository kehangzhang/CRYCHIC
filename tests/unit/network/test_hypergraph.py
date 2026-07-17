from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from crychic.core import ContractError, canonical_json
from crychic.network import (
    EDGE_RECORD_COLUMNS,
    CommunicationEdgeRecord,
    CommunicationHypergraph,
    HyperedgeStatus,
    HypergraphTables,
    NodeType,
    TargetKind,
    build_communication_hypergraph,
)


def _record(
    source_record_id: str,
    *,
    context_id: str = "control",
    sender: str = "T_cell",
    ligand: str = "IL12_complex",
    ligand_members: tuple[str, ...] = ("IL12A", "IL12B"),
    receptor: str = "IL12RB1",
    receptor_members: tuple[str, ...] = ("IL12RB1",),
    receiver: str = "NK_cell",
    target: str = "cytotoxic_program",
    target_kind: TargetKind = TargetKind.PROGRAM,
    weight: float | None = 2.0,
    uncertainty: float | None = 0.2,
    uncertainty_kind: str | None = "standard_error",
    status: HyperedgeStatus = HyperedgeStatus.OBSERVED,
    reason_code: str | None = None,
) -> CommunicationEdgeRecord:
    return CommunicationEdgeRecord(
        source_record_id=source_record_id,
        context_id=context_id,
        context_json=canonical_json({"condition": context_id, "tissue": "blood"}),
        sender=sender,
        ligand=ligand,
        ligand_members=ligand_members,
        receptor=receptor,
        receptor_members=receptor_members,
        receiver=receiver,
        target=target,
        target_kind=target_kind,
        interaction_id=f"{ligand}:{receptor}",
        mode="state",
        weight=weight,
        weight_semantics="communication_strength",
        uncertainty=uncertainty,
        uncertainty_kind=uncertainty_kind,
        status=status,
        reason_code=reason_code,
        source_artifact_id="edge_evidence_v1",
        scoring_functional_id="receiver_functional_v2",
        fold_id="fold_1",
        repeat_id="repeat_1",
        components_json=canonical_json({"availability": 0.7, "sender_abundance": 0.4}),
        provenance_json=canonical_json(
            {"config_digest": "cfg_123", "run_id": "run_123"}
        ),
    )


def test_stable_ids_are_independent_of_record_and_complex_member_order() -> None:
    first = _record("edge_a")
    second = _record(
        "edge_b",
        context_id="treated",
        ligand="CXCL10",
        ligand_members=("CXCL10",),
        receptor="CXCR3_complex",
        receptor_members=("CXCR3B", "CXCR3A"),
    )
    forward = build_communication_hypergraph([first, second])
    reversed_input = build_communication_hypergraph(
        [replace(second, receptor_members=("CXCR3A", "CXCR3B")), first]
    )

    assert (
        forward.hyperedges["hyperedge_id"].tolist()
        == reversed_input.hyperedges["hyperedge_id"].tolist()
    )
    assert forward.nodes["node_id"].tolist() == reversed_input.nodes["node_id"].tolist()
    assert (
        forward.complex_members["membership_id"].tolist()
        == reversed_input.complex_members["membership_id"].tolist()
    )


def test_direction_and_complex_members_are_explicit() -> None:
    graph = build_communication_hypergraph([_record("edge_a")])
    edge = graph.hyperedges.iloc[0]

    assert edge["direction"] == "sender_ligand_to_receptor_receiver_target"
    node_types = graph.nodes.set_index("node_id")["node_type"]
    assert node_types[edge["sender_node_id"]] == NodeType.SENDER.value
    assert node_types[edge["ligand_node_id"]] == NodeType.LIGAND_COMPLEX.value
    assert node_types[edge["receptor_node_id"]] == NodeType.RECEPTOR.value
    assert node_types[edge["receiver_node_id"]] == NodeType.RECEIVER.value
    assert node_types[edge["target_node_id"]] == NodeType.TARGET_PROGRAM.value

    members = graph.complex_members
    assert members["complex_role"].tolist() == [
        "ligand_subunit",
        "ligand_subunit",
    ]
    member_labels = graph.nodes.set_index("node_id").loc[
        members["member_node_id"], "label"
    ]
    assert sorted(member_labels.tolist()) == ["IL12A", "IL12B"]


def test_not_estimable_remains_missing_and_is_not_counted_by_default() -> None:
    unavailable = _record(
        "edge_ne",
        status=HyperedgeStatus.NOT_ESTIMABLE,
        weight=None,
        uncertainty=None,
        uncertainty_kind=None,
        reason_code="receiver_absent",
    )
    graph = build_communication_hypergraph([unavailable])

    assert pd.isna(graph.hyperedges.iloc[0]["weight"])
    assert graph.hyperedges.iloc[0]["status"] == "not_estimable"
    assert graph.degree_table()["total_degree"].sum() == 0
    included = graph.degree_table(statuses=[HyperedgeStatus.NOT_ESTIMABLE])
    assert included["total_degree"].sum() == 6
    assert included["total_strength"].sum() == pytest.approx(0.0)


def test_not_estimable_zero_and_inferential_fields_are_rejected() -> None:
    with pytest.raises(ContractError, match="do not encode not-estimable as zero"):
        _record(
            "edge_ne",
            status=HyperedgeStatus.NOT_ESTIMABLE,
            weight=0.0,
            uncertainty=None,
            uncertainty_kind=None,
            reason_code="receiver_absent",
        )

    record = _record("edge_a")
    frame = pd.DataFrame(
        [{field: getattr(record, field) for field in EDGE_RECORD_COLUMNS}]
    )
    frame["p_value"] = 0.01
    with pytest.raises(ContractError) as error:
        build_communication_hypergraph(frame)
    assert error.value.details.code == "forbidden_hypergraph_inference"

    with pytest.raises(ContractError) as nested_error:
        replace(
            _record("edge_b"),
            components_json=canonical_json({"posterior_probability": 0.9}),
        )
    assert nested_error.value.details.code == "forbidden_hypergraph_inference"


def test_dataframe_input_preserves_components_uncertainty_and_provenance() -> None:
    record = _record("edge_a")
    row = {field: getattr(record, field) for field in EDGE_RECORD_COLUMNS}
    row["context_json"] = {"tissue": "blood", "condition": "control"}
    row["components_json"] = {"availability": 0.7, "sender_abundance": 0.4}
    row["provenance_json"] = {"run_id": "run_123", "config_digest": "cfg_123"}
    graph = build_communication_hypergraph(pd.DataFrame([row]))
    edge = graph.hyperedges.iloc[0]

    assert edge["uncertainty"] == pytest.approx(0.2)
    assert edge["uncertainty_kind"] == "standard_error"
    assert edge["components_json"] == record.components_json
    assert edge["provenance_json"] == record.provenance_json


def test_context_query_known_directional_degree_and_hub_ranking() -> None:
    records = [
        _record("edge_a", weight=-2.0),
        _record(
            "edge_b",
            sender="B_cell",
            ligand="IFNG",
            ligand_members=("IFNG",),
            receptor="IFNGR1",
            receptor_members=("IFNGR1",),
            weight=1.0,
        ),
        _record(
            "edge_c",
            context_id="treated",
            ligand="CXCL10",
            ligand_members=("CXCL10",),
            receptor="CXCR3",
            receptor_members=("CXCR3",),
            weight=5.0,
        ),
    ]
    graph = build_communication_hypergraph(records)

    assert sorted(
        graph.query_edges(context_id="control")["source_record_id"].tolist()
    ) == [
        "edge_a",
        "edge_b",
    ]
    receiver_id = graph.query_nodes(
        node_type=NodeType.RECEIVER, entity_id="NK_cell"
    ).iloc[0]["node_id"]
    receiver_degree = graph.degree_table().set_index("node_id").loc[receiver_id]
    assert receiver_degree["in_degree"] == 3
    assert receiver_degree["in_strength"] == pytest.approx(8.0)
    assert "sum_absolute_fitted_weight" in receiver_degree["weight_policy"]

    top_receiver = graph.hubs(
        top_k=1,
        node_type=NodeType.RECEIVER,
        direction="in",
        weighted=True,
    )
    assert top_receiver.iloc[0]["label"] == "NK_cell"
    assert top_receiver.iloc[0]["ranking_metric"] == "in_strength"


def test_disconnected_contexts_form_separate_weak_components() -> None:
    graph = build_communication_hypergraph(
        [
            _record("edge_a"),
            _record(
                "edge_b",
                context_id="tumor",
                sender="Fibroblast",
                ligand="COL1A1",
                ligand_members=("COL1A1",),
                receptor="ITGB1",
                receptor_members=("ITGB1",),
                receiver="Endothelial",
                target="angiogenesis",
            ),
        ]
    )
    components = graph.connected_components()

    assert components["component_id"].nunique() == 2
    assert components.groupby("component_id")["component_size"].nunique().eq(1).all()
    assert set(components["node_id"]) == set(graph.nodes["node_id"])


def test_table_and_parquet_roundtrip_preserve_canonical_tables(tmp_path: Path) -> None:
    graph = build_communication_hypergraph(
        [_record("edge_a"), _record("edge_b", context_id="treated")]
    )
    tables = graph.to_tables()
    restored = CommunicationHypergraph.from_tables(tables)
    pdt.assert_frame_equal(restored.nodes, graph.nodes)
    pdt.assert_frame_equal(restored.hyperedges, graph.hyperedges)
    pdt.assert_frame_equal(restored.complex_members, graph.complex_members)

    nodes_path = tmp_path / "nodes.parquet"
    edges_path = tmp_path / "hyperedges.parquet"
    members_path = tmp_path / "complex_members.parquet"
    tables.nodes.to_parquet(nodes_path, index=False)
    tables.hyperedges.to_parquet(edges_path, index=False)
    tables.complex_members.to_parquet(members_path, index=False)
    parquet_restored = CommunicationHypergraph.from_tables(
        HypergraphTables(
            nodes=pd.read_parquet(nodes_path),
            hyperedges=pd.read_parquet(edges_path),
            complex_members=pd.read_parquet(members_path),
        )
    )
    pdt.assert_frame_equal(parquet_restored.nodes, graph.nodes)
    pdt.assert_frame_equal(parquet_restored.hyperedges, graph.hyperedges)
    pdt.assert_frame_equal(parquet_restored.complex_members, graph.complex_members)


def test_persisted_direction_and_ids_are_revalidated() -> None:
    graph = build_communication_hypergraph([_record("edge_a")])
    tables = graph.to_tables()
    corrupted = tables.hyperedges.copy()
    corrupted.loc[0, "direction"] = "receiver_to_sender"

    with pytest.raises(ContractError) as error:
        CommunicationHypergraph.from_tables(
            HypergraphTables(tables.nodes, corrupted, tables.complex_members)
        )
    assert error.value.details.code == "invalid_hypergraph_direction"
