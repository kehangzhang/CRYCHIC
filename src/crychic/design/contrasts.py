"""Balanced contrasts over declared context topology."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .context_graph import ContextGraph, ContextNode, _node_key


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    """A named, balanced linear contrast over context marginal means."""

    name: str
    weights: Mapping[ContextNode, float]
    family: str
    mode: str
    estimable: bool = True
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("contrast name must be non-empty")
        if not self.family:
            raise ValueError("contrast family must be non-empty")
        normalized: dict[ContextNode, float] = {}
        for node, value in self.weights.items():
            weight = float(value)
            if not math.isfinite(weight):
                raise ValueError(f"contrast weight for {node!r} must be finite")
            if weight != 0:
                normalized[node] = weight
        if not normalized:
            raise ValueError("contrast must contain at least one non-zero weight")
        if not math.isclose(sum(normalized.values()), 0.0, abs_tol=1e-12):
            raise ValueError("balanced contrast weights must sum to zero")
        ordered = dict(sorted(normalized.items(), key=lambda item: _node_key(item[0])))
        object.__setattr__(self, "weights", MappingProxyType(ordered))
        if not self.estimable and not self.reason_code:
            raise ValueError("a non-estimable contrast requires reason_code")

    def vector(self, nodes: Sequence[ContextNode]) -> np.ndarray:
        """Return weights in an explicit context-node order."""

        if len(set(nodes)) != len(nodes):
            raise ValueError("contrast vector node order must be unique")
        missing = set(self.weights).difference(nodes)
        if missing:
            raise ValueError(
                f"node order omits contrast context(s): {sorted(map(repr, missing))}"
            )
        result: np.ndarray = np.asarray(
            [self.weights.get(node, 0.0) for node in nodes], dtype=float
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready contrast definition."""

        return {
            "name": self.name,
            "family": self.family,
            "mode": self.mode,
            "estimable": self.estimable,
            "reason_code": self.reason_code,
            "weights": [
                {"context": node, "weight": weight}
                for node, weight in self.weights.items()
            ],
        }


def _unique_nodes(nodes: Iterable[ContextNode]) -> tuple[ContextNode, ...]:
    values = tuple(nodes)
    if not values:
        raise ValueError("contrast requires at least one context")
    if len(set(values)) != len(values):
        raise ValueError("contrast context collections must not contain duplicates")
    return tuple(sorted(values, key=_node_key))


def balanced_contrast(
    positive: Iterable[ContextNode],
    negative: Iterable[ContextNode],
    *,
    name: str = "balanced",
    family: str = "user_balanced",
) -> ContrastSpec:
    """Contrast two context sets with equal total mass on each side."""

    positive_nodes = _unique_nodes(positive)
    negative_nodes = _unique_nodes(negative)
    overlap = set(positive_nodes).intersection(negative_nodes)
    if overlap:
        raise ValueError(
            f"positive and negative contexts overlap: {sorted(map(repr, overlap))}"
        )
    weights = {node: 1.0 / len(positive_nodes) for node in positive_nodes}
    weights.update({node: -1.0 / len(negative_nodes) for node in negative_nodes})
    return ContrastSpec(name=name, weights=weights, family=family, mode="balanced")


def global_one_vs_rest(
    graph_or_nodes: ContextGraph | Iterable[ContextNode],
    focal: ContextNode,
    *,
    name: str | None = None,
    family: str = "global_one_vs_rest",
) -> ContrastSpec:
    """Compare one context with the balanced mean of every other context."""

    nodes = (
        graph_or_nodes.nodes
        if isinstance(graph_or_nodes, ContextGraph)
        else _unique_nodes(graph_or_nodes)
    )
    if focal not in nodes:
        raise ValueError(f"focal context {focal!r} is not in the declared context set")
    rest = tuple(node for node in nodes if node != focal)
    if not rest:
        raise ValueError("global one-vs-rest requires at least two contexts")
    weights = {focal: 1.0, **{node: -1.0 / len(rest) for node in rest}}
    return ContrastSpec(
        name=name or f"global:{focal!r}",
        weights=weights,
        family=family,
        mode="global_one_vs_rest",
    )


def local_neighbor_contrast(
    graph: ContextGraph,
    focal: ContextNode,
    *,
    name: str | None = None,
    family: str = "local_neighbor",
) -> ContrastSpec:
    """Compare one context with its edge-weighted neighbor mean."""

    neighbors = graph.neighbors(focal)
    if not neighbors:
        raise ValueError(f"local contrast for {focal!r} has no declared graph neighbor")
    total_weight = sum(weight for _, weight in neighbors)
    weights = {focal: 1.0}
    weights.update({neighbor: -weight / total_weight for neighbor, weight in neighbors})
    return ContrastSpec(
        name=name or f"local:{focal!r}",
        weights=weights,
        family=family,
        mode="local_neighbor",
    )


def global_contrasts(graph: ContextGraph) -> tuple[ContrastSpec, ...]:
    """Build one balanced global contrast for every context node."""

    return tuple(global_one_vs_rest(graph, node) for node in graph.nodes)


def local_contrasts(graph: ContextGraph) -> tuple[ContrastSpec, ...]:
    """Build local contrasts for nodes with at least one declared neighbor."""

    return tuple(
        local_neighbor_contrast(graph, node)
        for node in graph.nodes
        if graph.neighbors(node)
    )


def _factor_value(node: ContextNode, factor: str) -> Hashable:
    if not isinstance(node, tuple) or not all(
        isinstance(item, tuple) and len(item) == 2 for item in node
    ):
        raise ValueError(
            "factorial contrasts require canonical multi-factor context nodes"
        )
    mapping = dict(node)
    if factor not in mapping:
        raise ValueError(f"context node {node!r} has no factor {factor!r}")
    value = mapping[factor]
    if not isinstance(value, Hashable):  # pragma: no cover - ContextNode contract
        raise TypeError(f"factor value for {factor!r} must be hashable")
    return value


def marginal_factor_contrast(
    graph_or_nodes: ContextGraph | Iterable[ContextNode],
    factor: str,
    positive_level: Hashable,
    negative_level: Hashable,
    *,
    name: str | None = None,
    family: str = "factorial_main_effect",
) -> ContrastSpec:
    """Compare two factor levels, equally averaging all declared context cells."""

    nodes = (
        graph_or_nodes.nodes
        if isinstance(graph_or_nodes, ContextGraph)
        else _unique_nodes(graph_or_nodes)
    )
    if positive_level == negative_level:
        raise ValueError("factor contrast levels must differ")
    factor_maps = [dict(node) for node in nodes if isinstance(node, tuple)]
    if len(factor_maps) != len(nodes):
        raise ValueError(
            "factorial contrasts require canonical multi-factor context nodes"
        )
    nuisance_factors = sorted(set(factor_maps[0]).difference({factor}))
    positive_support = {
        tuple(mapping[name] for name in nuisance_factors)
        for mapping in factor_maps
        if mapping.get(factor) == positive_level
    }
    negative_support = {
        tuple(mapping[name] for name in nuisance_factors)
        for mapping in factor_maps
        if mapping.get(factor) == negative_level
    }
    if positive_support != negative_support:
        raise ValueError(
            "factor levels lack common nuisance-factor support for a balanced EMM"
        )
    positive = tuple(
        node for node in nodes if _factor_value(node, factor) == positive_level
    )
    negative = tuple(
        node for node in nodes if _factor_value(node, factor) == negative_level
    )
    if not positive or not negative:
        raise ValueError(
            f"factor {factor!r} must contain both requested levels in the context graph"
        )
    return balanced_contrast(
        positive,
        negative,
        name=name or f"main:{factor}:{positive_level!r}-vs-{negative_level!r}",
        family=family,
    )


def factorial_interaction_contrast(
    graph_or_nodes: ContextGraph | Iterable[ContextNode],
    factor_a: str,
    positive_a: Hashable,
    negative_a: Hashable,
    factor_b: str,
    positive_b: Hashable,
    negative_b: Hashable,
    *,
    name: str | None = None,
    family: str = "factorial_interaction",
) -> ContrastSpec:
    """Build a balanced difference-in-differences over two context factors."""

    if factor_a == factor_b:
        raise ValueError("factorial interaction requires two distinct factors")
    if positive_a == negative_a or positive_b == negative_b:
        raise ValueError("factorial interaction levels must differ within each factor")
    nodes = (
        graph_or_nodes.nodes
        if isinstance(graph_or_nodes, ContextGraph)
        else _unique_nodes(graph_or_nodes)
    )
    cells: dict[tuple[Hashable, Hashable], tuple[ContextNode, ...]] = {}
    for level_a in (positive_a, negative_a):
        for level_b in (positive_b, negative_b):
            matched = tuple(
                node
                for node in nodes
                if _factor_value(node, factor_a) == level_a
                and _factor_value(node, factor_b) == level_b
            )
            if not matched:
                raise ValueError(
                    "factorial interaction context graph lacks cell "
                    f"({factor_a}={level_a!r}, {factor_b}={level_b!r})"
                )
            cells[(level_a, level_b)] = matched
    nuisance_support: list[set[tuple[Hashable, ...]]] = []
    for matched in cells.values():
        support: set[tuple[Hashable, ...]] = set()
        for node in matched:
            if not isinstance(node, tuple):  # protected by _factor_value above
                raise ValueError(
                    "factorial contrasts require canonical multi-factor context nodes"
                )
            support.add(
                tuple(value for key, value in node if key not in {factor_a, factor_b})
            )
        nuisance_support.append(support)
    if any(support != nuisance_support[0] for support in nuisance_support[1:]):
        raise ValueError(
            "factorial interaction cells lack common nuisance-factor support"
        )
    weights: dict[ContextNode, float] = {}
    for levels, sign in (
        ((positive_a, positive_b), 1.0),
        ((positive_a, negative_b), -1.0),
        ((negative_a, positive_b), -1.0),
        ((negative_a, negative_b), 1.0),
    ):
        matched = cells[levels]
        for node in matched:
            weights[node] = sign / len(matched)
    return ContrastSpec(
        name=name
        or (
            f"interaction:{factor_a}[{positive_a!r}-{negative_a!r}]x"
            f"{factor_b}[{positive_b!r}-{negative_b!r}]"
        ),
        weights=weights,
        family=family,
        mode="factorial_difference_in_differences",
    )
