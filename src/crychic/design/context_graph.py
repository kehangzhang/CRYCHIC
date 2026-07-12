"""Context-specific topology with no molecular or result-graph semantics."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

ContextNode: TypeAlias = Hashable
CanonicalContext: TypeAlias = tuple[tuple[str, Hashable], ...]


def _node_key(node: ContextNode) -> tuple[str, str]:
    type_name = f"{type(node).__module__}.{type(node).__qualname__}"
    encoded = json.dumps(node, sort_keys=True, default=str, ensure_ascii=True)
    return type_name, encoded


def _ordered_pair(
    left: ContextNode, right: ContextNode
) -> tuple[ContextNode, ContextNode]:
    return (left, right) if _node_key(left) < _node_key(right) else (right, left)


@dataclass(frozen=True, slots=True)
class ContextEdge:
    """One positive-weight undirected edge in a context graph."""

    left: ContextNode
    right: ContextNode
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise ValueError(f"context graph self-edge is not allowed: {self.left!r}")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(
                f"context edge weight must be finite and positive: {self.weight!r}"
            )
        left, right = _ordered_pair(self.left, self.right)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True, slots=True)
class ContextGraph:
    """An immutable weighted undirected graph over experimental contexts."""

    nodes: tuple[ContextNode, ...]
    edges: tuple[ContextEdge, ...] = ()
    kind: str = "custom"

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("context graph must contain at least one node")
        try:
            node_set = set(self.nodes)
        except TypeError as error:
            raise TypeError("context graph nodes must be hashable") from error
        if len(node_set) != len(self.nodes):
            raise ValueError("context graph nodes must be unique")
        normalized_nodes = tuple(sorted(node_set, key=_node_key))

        normalized_edges: list[ContextEdge] = []
        seen: set[frozenset[ContextNode]] = set()
        for edge in self.edges:
            if not isinstance(edge, ContextEdge):
                raise TypeError("ContextGraph.edges must contain ContextEdge values")
            if edge.left not in node_set or edge.right not in node_set:
                raise ValueError(
                    f"context edge {(edge.left, edge.right)!r} references "
                    "an unknown node"
                )
            key = frozenset((edge.left, edge.right))
            if key in seen:
                raise ValueError(
                    f"duplicate context edge between {edge.left!r} and {edge.right!r}"
                )
            seen.add(key)
            normalized_edges.append(edge)
        normalized_edges.sort(
            key=lambda edge: (_node_key(edge.left), _node_key(edge.right))
        )
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "edges", tuple(normalized_edges))
        if not self.kind:
            raise ValueError("context graph kind must be non-empty")

    @classmethod
    def chain(
        cls,
        nodes: Sequence[ContextNode],
        *,
        weights: Sequence[float] | None = None,
    ) -> ContextGraph:
        """Connect consecutive nodes in the declared biological order."""

        ordered = tuple(nodes)
        if not ordered:
            raise ValueError("chain requires at least one context node")
        edge_weights = (
            tuple(weights) if weights is not None else (1.0,) * (len(ordered) - 1)
        )
        if len(edge_weights) != max(0, len(ordered) - 1):
            raise ValueError("chain weights must have len(nodes) - 1 entries")
        edges = tuple(
            ContextEdge(left, right, weight)
            for left, right, weight in zip(
                ordered[:-1], ordered[1:], edge_weights, strict=True
            )
        )
        return cls(nodes=ordered, edges=edges, kind="chain")

    @classmethod
    def complete(
        cls,
        nodes: Sequence[ContextNode],
        *,
        weight: float = 1.0,
    ) -> ContextGraph:
        """Connect every pair of supplied contexts with a common weight."""

        supplied = tuple(nodes)
        if not supplied:
            raise ValueError("complete graph requires at least one context node")
        edges = tuple(
            ContextEdge(left, right, weight)
            for left, right in itertools.combinations(supplied, 2)
        )
        return cls(nodes=supplied, edges=edges, kind="complete")

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[
            ContextEdge
            | tuple[ContextNode, ContextNode]
            | tuple[ContextNode, ContextNode, float]
        ],
        *,
        nodes: Sequence[ContextNode] | None = None,
    ) -> ContextGraph:
        """Build a custom graph, optionally retaining isolated nodes."""

        parsed: list[ContextEdge] = []
        discovered: list[ContextNode] = list(nodes or ())
        for raw_edge in edges:
            if isinstance(raw_edge, ContextEdge):
                edge = raw_edge
            else:
                values = tuple(raw_edge)
                if len(values) == 2:
                    edge = ContextEdge(values[0], values[1])
                elif len(values) == 3:
                    edge = ContextEdge(
                        values[0], values[1], float(cast(Any, values[2]))
                    )
                else:
                    raise ValueError("custom edges must have (left, right[, weight])")
            parsed.append(edge)
            discovered.extend((edge.left, edge.right))
        unique_nodes = tuple(dict.fromkeys(discovered))
        if not unique_nodes:
            raise ValueError("from_edges requires an edge or an explicit isolated node")
        return cls(nodes=unique_nodes, edges=tuple(parsed), kind="custom")

    @classmethod
    def product(
        cls,
        graphs: Mapping[str, ContextGraph],
        *,
        edge_weights: Mapping[str, float] | None = None,
    ) -> ContextGraph:
        """Return the Cartesian product of named factor graphs.

        Product nodes are canonical tuples sorted by factor name. An edge changes
        exactly one factor along an edge declared in that factor's graph.
        """

        if not graphs:
            raise ValueError("product requires at least one named factor graph")
        if any(not isinstance(name, str) or not name for name in graphs):
            raise ValueError("product factor names must be non-empty strings")
        if any(not isinstance(graph, ContextGraph) for graph in graphs.values()):
            raise TypeError("product values must be ContextGraph instances")
        scales = dict(edge_weights or {})
        unknown_scales = set(scales).difference(graphs)
        if unknown_scales:
            raise ValueError(
                f"edge_weights contains unknown factor(s): {sorted(unknown_scales)}"
            )
        for factor, scale in scales.items():
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"product edge scale for {factor!r} must be finite and positive"
                )

        factors = tuple(sorted(graphs))
        combinations = tuple(
            itertools.product(*(graphs[name].nodes for name in factors))
        )
        product_nodes: tuple[CanonicalContext, ...] = tuple(
            tuple(zip(factors, values, strict=True)) for values in combinations
        )
        node_by_values = dict(zip(combinations, product_nodes, strict=True))

        product_edges: list[ContextEdge] = []
        seen: set[frozenset[ContextNode]] = set()
        for values, node in node_by_values.items():
            for factor_index, factor in enumerate(factors):
                for neighbor, base_weight in graphs[factor].neighbors(
                    values[factor_index]
                ):
                    neighbor_values = list(values)
                    neighbor_values[factor_index] = neighbor
                    other = node_by_values[tuple(neighbor_values)]
                    key = frozenset((node, other))
                    if key in seen:
                        continue
                    seen.add(key)
                    product_edges.append(
                        ContextEdge(node, other, base_weight * scales.get(factor, 1.0))
                    )
        return cls(nodes=product_nodes, edges=tuple(product_edges), kind="product")

    def neighbors(self, node: ContextNode) -> tuple[tuple[ContextNode, float], ...]:
        """Return neighboring contexts and edge weights in canonical order."""

        if node not in self.nodes:
            raise KeyError(node)
        adjacent: list[tuple[ContextNode, float]] = []
        for edge in self.edges:
            if edge.left == node:
                adjacent.append((edge.right, edge.weight))
            elif edge.right == node:
                adjacent.append((edge.left, edge.weight))
        adjacent.sort(key=lambda item: _node_key(item[0]))
        return tuple(adjacent)

    def edge_weight(self, left: ContextNode, right: ContextNode) -> float:
        """Return a declared edge weight or raise ``KeyError``."""

        wanted = frozenset((left, right))
        for edge in self.edges:
            if frozenset((edge.left, edge.right)) == wanted:
                return edge.weight
        raise KeyError((left, right))

    @property
    def components(self) -> tuple[tuple[ContextNode, ...], ...]:
        """Connected components without information flow across components."""

        remaining = set(self.nodes)
        components: list[tuple[ContextNode, ...]] = []
        while remaining:
            seed = min(remaining, key=_node_key)
            stack = [seed]
            component: set[ContextNode] = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(neighbor for neighbor, _ in self.neighbors(node))
            remaining.difference_update(component)
            components.append(tuple(sorted(component, key=_node_key)))
        components.sort(key=lambda component: _node_key(component[0]))
        return tuple(components)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible graph representation."""

        return {
            "kind": self.kind,
            "nodes": list(self.nodes),
            "edges": [
                {"left": edge.left, "right": edge.right, "weight": edge.weight}
                for edge in self.edges
            ],
        }
