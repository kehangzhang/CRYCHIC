from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd
import pytest
from benchmarks.metrics.multicondition import (
    EDGE_KEYS,
    FAMILY_KEYS,
    _family_loso_metrics,
)
from scipy.stats import spearmanr


def _effect_top_keys_reference(
    effect: pd.Series, *, k: int
) -> set[tuple[object, ...]]:
    if effect.empty:
        return set()
    magnitude = effect.abs().sort_values(ascending=False, kind="stable")
    cutoff = float(magnitude.iloc[min(k, len(magnitude)) - 1])
    selected = magnitude.loc[magnitude >= cutoff]
    return set(selected.index.tolist())


def _family_loso_metrics_reference(
    held: pd.Series,
    training: pd.Series,
    universe: pd.DataFrame,
    *,
    top_k: int,
) -> dict[str, object]:
    """Scalar family-scan implementation retained as a regression oracle."""
    held_frame = held.rename("held_effect").reset_index()
    training_frame = training.rename("training_effect").reset_index()
    shared = held_frame.merge(
        training_frame,
        on=list(EDGE_KEYS),
        validate="one_to_one",
    )
    rho_values: list[float] = []
    direction_values: list[float] = []
    jaccard_values: list[float] = []
    family_coverages: list[float] = []
    direction_edges = 0
    eligible_families = 0
    for family, family_universe in universe.groupby(
        list(FAMILY_KEYS), sort=False, observed=True
    ):
        sender_value, receiver_value = cast(tuple[str | int, str | int], family)
        family_shared = shared.loc[
            shared["sender"].eq(sender_value)
            & shared["receiver"].eq(receiver_value)
        ]
        family_size = len(family_universe)
        family_coverages.append(len(family_shared) / family_size)
        if family_size < 3:
            continue
        eligible_families += 1
        enough_edges = len(family_shared) >= 3
        constant = bool(
            enough_edges
            and (
                family_shared["held_effect"].nunique() < 2
                or family_shared["training_effect"].nunique() < 2
            )
        )
        if enough_edges and not constant:
            rho = spearmanr(
                family_shared["held_effect"],
                family_shared["training_effect"],
            ).statistic
            if np.isfinite(rho):
                rho_values.append(float(rho))

        nonzero = ~(
            family_shared["held_effect"].eq(0)
            & family_shared["training_effect"].eq(0)
        )
        n_direction = int(nonzero.sum())
        direction_edges += n_direction
        if n_direction:
            direction_values.append(
                float(
                    np.mean(
                        np.sign(family_shared.loc[nonzero, "held_effect"])
                        == np.sign(family_shared.loc[nonzero, "training_effect"])
                    )
                )
            )

        held_family = family_shared.set_index(list(EDGE_KEYS))["held_effect"]
        training_family = family_shared.set_index(list(EDGE_KEYS))["training_effect"]
        held_top = _effect_top_keys_reference(held_family, k=top_k)
        training_top = _effect_top_keys_reference(training_family, k=top_k)
        top_union = held_top | training_top
        if top_union:
            jaccard_values.append(len(held_top & training_top) / len(top_union))

    return {
        "shared_edges": len(shared),
        "n_eligible_families": eligible_families,
        "n_estimable_families": len(rho_values),
        "effect_spearman": (float(np.mean(rho_values)) if rho_values else math.nan),
        "direction_comparable_edges": direction_edges,
        "direction_agreement": (
            float(np.mean(direction_values)) if direction_values else math.nan
        ),
        "top_k_jaccard": (
            float(np.mean(jaccard_values)) if jaccard_values else math.nan
        ),
        "macro_family_shared_edge_coverage": (
            float(np.mean(family_coverages)) if family_coverages else math.nan
        ),
    }


def _edge(sender: object, receiver: object, index: int) -> tuple[object, ...]:
    return (sender, receiver, f"I{index}", f"L{index}", f"R{index}")


def _series(rows: list[tuple[tuple[object, ...], float]]) -> pd.Series:
    index = pd.MultiIndex.from_tuples(
        [edge for edge, _ in rows], names=list(EDGE_KEYS)
    )
    return pd.Series([value for _, value in rows], index=index, dtype=float)


def _universe(edges: list[tuple[object, ...]]) -> pd.DataFrame:
    return pd.DataFrame(edges, columns=list(EDGE_KEYS))


def _float_metric(metrics: dict[str, object], key: str) -> float:
    value = metrics[key]
    assert isinstance(value, float)
    return value


@pytest.fixture(  # type: ignore[untyped-decorator]
    params=(
        "mixed_families",
        "all_zero_direction",
        "no_shared_edges",
        "string_shared_numeric_universe",
        "numeric_shared_numeric_universe",
    )
)
def family_metric_case(
    request: pytest.FixtureRequest,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, int]:
    if request.param == "mixed_families":
        ties = [_edge("ties", "R-ties", index) for index in range(4)]
        constant = [_edge("constant", "R-constant", 10 + index) for index in range(3)]
        zero = [_edge("zero", "R-zero", 20 + index) for index in range(3)]
        no_shared = [_edge("none", "R-none", 30 + index) for index in range(3)]
        partial = [_edge("partial", "R-partial", 40 + index) for index in range(5)]
        small = [_edge("small", "R-small", 50 + index) for index in range(2)]
        held_rows = [
            *zip(ties, (4.0, 4.0, 1.0, -1.0), strict=True),
            *zip(constant, (1.0, 2.0, 3.0), strict=True),
            *zip(zero, (0.0, 0.0, 0.0), strict=True),
            (no_shared[0], 2.0),
            *zip(partial[:4], (3.0, -2.0, 0.0, 1.0), strict=True),
            *zip(small, (2.0, 1.0), strict=True),
        ]
        training_rows = [
            *zip(ties, (4.0, 3.0, 3.0, -1.0), strict=True),
            *zip(constant, (2.0, 2.0, 2.0), strict=True),
            *zip(zero, (0.0, 0.0, 0.0), strict=True),
            (no_shared[1], 2.0),
            *zip(
                (partial[0], partial[1], partial[2], partial[4]),
                (1.0, 2.0, 0.0, 4.0),
                strict=True,
            ),
            *zip(small, (1.0, 2.0), strict=True),
        ]
        universe = _universe(
            [*ties, *constant, *zero, *no_shared, *partial, *small]
        )
        return _series(held_rows), _series(training_rows), universe, 1

    if request.param == "all_zero_direction":
        edges = [_edge("zero", "R-zero", index) for index in range(4)]
        rows = list(zip(edges, (0.0, 0.0, 0.0, 0.0), strict=True))
        return _series(rows), _series(rows), _universe(edges), 2

    if request.param == "no_shared_edges":
        edges = [_edge("none", "R-none", index) for index in range(3)]
        return (
            _series([(edges[0], 1.0)]),
            _series([(edges[1], 1.0)]),
            _universe(edges),
            2,
        )

    if request.param == "string_shared_numeric_universe":
        shared_edges = [_edge("1", "2", index) for index in range(3)]
        universe_edges = [_edge(1, 2, index) for index in range(3)]
        return (
            _series(list(zip(shared_edges, (1.0, 2.0, 3.0), strict=True))),
            _series(list(zip(shared_edges, (3.0, 2.0, 1.0), strict=True))),
            _universe(universe_edges),
            1,
        )

    shared_edges = [_edge(1, 2, index) for index in range(3)]
    rows = list(zip(shared_edges, (1.0, 2.0, 3.0), strict=True))
    return _series(rows), _series(rows), _universe(shared_edges), 1


def test_family_loso_lookup_matches_scalar_reference(
    family_metric_case: tuple[pd.Series, pd.Series, pd.DataFrame, int],
) -> None:
    held, training, universe, top_k = family_metric_case

    expected = _family_loso_metrics_reference(
        held, training, universe, top_k=top_k
    )
    actual = _family_loso_metrics(held, training, universe, top_k=top_k)

    for key in (
        "shared_edges",
        "n_eligible_families",
        "n_estimable_families",
        "direction_comparable_edges",
    ):
        assert actual[key] == expected[key]
    for key in (
        "effect_spearman",
        "direction_agreement",
        "top_k_jaccard",
        "macro_family_shared_edge_coverage",
    ):
        np.testing.assert_allclose(
            _float_metric(actual, key),
            _float_metric(expected, key),
            rtol=1e-14,
            atol=1e-14,
            equal_nan=True,
        )


def test_family_loso_matches_exact_key_types_without_collisions() -> None:
    numeric_edges = [_edge(1, 2, index) for index in range(3)]
    string_edges = [_edge("1", "2", 10 + index) for index in range(3)]
    held = _series(
        [
            *zip(numeric_edges, (1.0, 2.0, 3.0), strict=True),
            *zip(string_edges, (1.0, 2.0, 3.0), strict=True),
        ]
    )
    training = _series(
        [
            *zip(numeric_edges, (1.0, 2.0, 3.0), strict=True),
            *zip(string_edges, (3.0, 2.0, 1.0), strict=True),
        ]
    )

    metrics = _family_loso_metrics(
        held,
        training,
        _universe(numeric_edges),
        top_k=1,
    )

    assert metrics["shared_edges"] == 6
    assert metrics["n_eligible_families"] == 1
    assert metrics["n_estimable_families"] == 1
    assert _float_metric(metrics, "effect_spearman") == pytest.approx(1.0)
    assert _float_metric(
        metrics, "macro_family_shared_edge_coverage"
    ) == pytest.approx(1.0)
    assert _float_metric(metrics, "macro_family_shared_edge_coverage") <= 1.0


def test_family_loso_preserves_macro_weighting_ties_and_partial_coverage() -> None:
    ties = [_edge("ties", "R-ties", index) for index in range(4)]
    constant = [_edge("constant", "R-constant", 10 + index) for index in range(3)]
    zero = [_edge("zero", "R-zero", 20 + index) for index in range(3)]
    no_shared = [_edge("none", "R-none", 30 + index) for index in range(3)]
    partial = [_edge("partial", "R-partial", 40 + index) for index in range(5)]
    small = [_edge("small", "R-small", 50 + index) for index in range(2)]
    held = _series(
        [
            *zip(ties, (4.0, 4.0, 1.0, -1.0), strict=True),
            *zip(constant, (1.0, 2.0, 3.0), strict=True),
            *zip(zero, (0.0, 0.0, 0.0), strict=True),
            *zip(partial[:3], (3.0, -2.0, 0.0), strict=True),
            *zip(small, (2.0, 1.0), strict=True),
        ]
    )
    training = _series(
        [
            *zip(ties, (4.0, 3.0, 3.0, -1.0), strict=True),
            *zip(constant, (2.0, 2.0, 2.0), strict=True),
            *zip(zero, (0.0, 0.0, 0.0), strict=True),
            *zip(partial[:3], (1.0, 2.0, 0.0), strict=True),
            *zip(small, (1.0, 2.0), strict=True),
        ]
    )
    universe = _universe([*ties, *constant, *zero, *no_shared, *partial, *small])

    metrics = _family_loso_metrics(held, training, universe, top_k=1)

    assert metrics["shared_edges"] == 15
    assert metrics["n_eligible_families"] == 5
    assert metrics["n_estimable_families"] == 2
    assert metrics["direction_comparable_edges"] == 9
    assert metrics["direction_agreement"] == pytest.approx(5 / 6)
    # Ties expand top-1 sets; each eligible family contributes equally.
    assert metrics["top_k_jaccard"] == pytest.approx((1 / 2 + 1 / 3 + 1) / 4)
    # Coverage includes the size-two family even though correlations exclude it.
    assert metrics["macro_family_shared_edge_coverage"] == pytest.approx(23 / 30)


def test_family_loso_all_zero_family_has_no_direction_or_correlation() -> None:
    edges = [_edge("zero", "R-zero", index) for index in range(4)]
    effects = _series(list(zip(edges, (0.0, 0.0, 0.0, 0.0), strict=True)))

    metrics = _family_loso_metrics(effects, effects, _universe(edges), top_k=1)

    assert metrics["n_eligible_families"] == 1
    assert metrics["n_estimable_families"] == 0
    assert metrics["direction_comparable_edges"] == 0
    assert math.isnan(_float_metric(metrics, "effect_spearman"))
    assert math.isnan(_float_metric(metrics, "direction_agreement"))
    assert metrics["top_k_jaccard"] == pytest.approx(1.0)
