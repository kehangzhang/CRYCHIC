from __future__ import annotations

import pytest

from crychic.design import ContextEdge, ContextGraph


def test_chain_preserves_declared_adjacency_and_weights() -> None:
    graph = ContextGraph.chain(["adjacent", "border", "core"], weights=[0.5, 2.0])

    assert len(graph.edges) == 2
    assert graph.neighbors("adjacent") == (("border", 0.5),)
    assert dict(graph.neighbors("border")) == {"adjacent": 0.5, "core": 2.0}
    assert graph.edge_weight("border", "core") == 2.0
    assert graph.components == (("adjacent", "border", "core"),)


def test_complete_graph_contains_each_pair_once() -> None:
    graph = ContextGraph.complete(["a", "b", "c"], weight=0.25)

    assert len(graph.edges) == 3
    assert all(edge.weight == 0.25 for edge in graph.edges)
    assert len(graph.neighbors("a")) == 2


def test_custom_graph_retains_isolated_and_disconnected_nodes() -> None:
    graph = ContextGraph.from_edges([("a", "b", 2.0)], nodes=["a", "b", "c", "d"])

    assert graph.components == (("a", "b"), ("c",), ("d",))
    assert graph.neighbors("c") == ()


def test_product_changes_exactly_one_named_factor() -> None:
    region = ContextGraph.chain(["adjacent", "core"])
    treatment = ContextGraph.complete(["control", "treated"])
    product = ContextGraph.product(
        {"treatment": treatment, "region": region},
        edge_weights={"region": 2.0, "treatment": 0.5},
    )

    assert len(product.nodes) == 4
    assert len(product.edges) == 4
    expected = (
        ("region", "adjacent"),
        ("treatment", "control"),
    )
    assert expected in product.nodes
    for edge in product.edges:
        changed = sum(
            left != right
            for (_, left), (_, right) in zip(edge.left, edge.right, strict=True)
        )
        assert changed == 1
        changed_factor = next(
            factor
            for (factor, left), (_, right) in zip(edge.left, edge.right, strict=True)
            if left != right
        )
        assert edge.weight == pytest.approx(2.0 if changed_factor == "region" else 0.5)


def test_product_does_not_connect_disconnected_factor_components() -> None:
    disconnected = ContextGraph.from_edges([("a", "b")], nodes=["a", "b", "c"])
    binary = ContextGraph.chain([0, 1])
    product = ContextGraph.product({"x": disconnected, "y": binary})

    assert len(product.components) == 2
    assert sorted(map(len, product.components)) == [2, 4]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ContextGraph.chain(["a", "a"]),
        lambda: ContextGraph.from_edges([("a", "a")]),
        lambda: ContextGraph.from_edges([("a", "b"), ("b", "a")]),
        lambda: ContextGraph(nodes=("a", "b"), edges=(ContextEdge("a", "b", 0),)),
    ],
)
def test_invalid_topologies_are_rejected(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()
