from __future__ import annotations

import numpy as np
import pytest

from crychic.design import (
    ContextGraph,
    balanced_contrast,
    global_contrasts,
    global_one_vs_rest,
    local_contrasts,
    local_neighbor_contrast,
)


def test_balanced_group_contrast_matches_hand_calculation() -> None:
    contrast = balanced_contrast(
        ["treated_a", "treated_b"],
        ["control_a", "control_b", "control_c"],
        name="treated-v-control",
    )

    assert contrast.weights["treated_a"] == pytest.approx(0.5)
    assert contrast.weights["control_a"] == pytest.approx(-1 / 3)
    assert sum(contrast.weights.values()) == pytest.approx(0.0)


def test_global_one_vs_rest_is_balanced_not_cell_weighted() -> None:
    graph = ContextGraph.complete(["a", "b", "c", "d"])
    contrast = global_one_vs_rest(graph, "a")

    assert contrast.weights == {
        "a": 1.0,
        "b": pytest.approx(-1 / 3),
        "c": pytest.approx(-1 / 3),
        "d": pytest.approx(-1 / 3),
    }
    np.testing.assert_allclose(
        contrast.vector(["a", "b", "c", "d"]), [1, -1 / 3, -1 / 3, -1 / 3]
    )
    assert len(global_contrasts(graph)) == 4


def test_local_contrast_uses_only_weighted_graph_neighbors() -> None:
    graph = ContextGraph.from_edges(
        [("a", "b", 1.0), ("a", "c", 3.0)], nodes=["a", "b", "c", "d"]
    )
    contrast = local_neighbor_contrast(graph, "a")

    assert contrast.weights == {
        "a": 1.0,
        "b": pytest.approx(-0.25),
        "c": pytest.approx(-0.75),
    }
    assert "d" not in contrast.weights
    assert sum(contrast.weights.values()) == pytest.approx(0.0)
    assert len(local_contrasts(graph)) == 3
    with pytest.raises(ValueError, match="no declared graph neighbor"):
        local_neighbor_contrast(graph, "d")


def test_contrast_vector_requires_all_referenced_contexts() -> None:
    contrast = balanced_contrast(["treated"], ["control"])
    with pytest.raises(ValueError, match="omits contrast context"):
        contrast.vector(["treated"])


def test_context_relabeling_preserves_weight_pattern() -> None:
    first = ContextGraph.chain(["a", "b", "c"])
    second = ContextGraph.chain(["z", "y", "x"])

    first_weights = sorted(local_neighbor_contrast(first, "b").weights.values())
    second_weights = sorted(local_neighbor_contrast(second, "y").weights.values())
    np.testing.assert_allclose(first_weights, second_weights)


def test_invalid_balanced_groups_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        balanced_contrast(["a", "b"], ["b", "c"])
    with pytest.raises(ValueError, match="at least one context"):
        balanced_contrast([], ["a"])
