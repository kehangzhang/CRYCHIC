from __future__ import annotations

import math

import pandas as pd
import pytest
from benchmarks.metrics.spatial_des import SpatialDESSpec, evaluate_spatial_des


def _ranked(
    strengths: list[float | None],
    pairs: list[tuple[str, str]],
    *,
    statuses: list[str] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["fixture"] * len(pairs),
            "method": ["method_a"] * len(pairs),
            "condition": ["case"] * len(pairs),
            "sender": [pair[0] for pair in pairs],
            "receiver": [pair[1] for pair in pairs],
            "ranked_strength": strengths,
            "status": statuses or ["observed"] * len(pairs),
        }
    )


def _expected(
    members: dict[float, list[tuple[str, str]]],
) -> pd.DataFrame:
    rows = []
    for fraction, pairs in members.items():
        rows.extend(
            {
                "dataset": "fixture",
                "condition": "case",
                "top_fraction": fraction,
                "sender": sender,
                "receiver": receiver,
            }
            for sender, receiver in pairs
        )
    return pd.DataFrame.from_records(rows)


def test_perfect_top_enrichment_at_all_preregistered_fractions() -> None:
    pairs = [(f"S{index}", "R") for index in range(1, 11)]
    ranked = _ranked([float(value) for value in range(10, 0, -1)], pairs)
    expected = _expected(
        {
            0.1: pairs[:1],
            0.2: pairs[:2],
            0.3: pairs[:3],
            0.4: pairs[:4],
        }
    )

    result = evaluate_spatial_des(ranked, expected)

    assert result.scores["top_fraction"].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert result.scores["des"].tolist() == pytest.approx([1.0] * 4)
    assert set(result.scores["status"]) == {"observed"}
    assert set(result.scores["score_type"]) == {"pos"}
    assert set(result.scores["fgsea_analogue"]) == {"fgsea_scoreType=pos;gseaParam=0"}
    assert set(result.scores["cell_pair_direction"]) == {"ordered_sender_to_receiver"}
    assert result.coverage["expected_pair_coverage_fraction"].tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]


def test_pos_and_custom_abs_have_explicitly_different_bottom_hit_semantics() -> None:
    pairs = [("S1", "R"), ("S2", "R"), ("S3", "R"), ("S4", "R")]
    ranked = _ranked([4.0, 3.0, 2.0, 1.0], pairs)
    expected = _expected({0.1: pairs[-1:]})
    common = {"top_fractions": (0.1,)}

    positive = evaluate_spatial_des(
        ranked, expected, SpatialDESSpec(**common, score_type="pos")
    )
    absolute = evaluate_spatial_des(
        ranked, expected, SpatialDESSpec(**common, score_type="abs")
    )

    assert positive.scores.iloc[0]["des"] == pytest.approx(0.0)
    assert positive.scores.iloc[0]["signed_peak_es"] == pytest.approx(0.0)
    assert absolute.scores.iloc[0]["des"] == pytest.approx(1.0)
    assert absolute.scores.iloc[0]["signed_peak_es"] == pytest.approx(-1.0)
    assert absolute.scores.iloc[0]["fgsea_analogue"] == (
        "custom_abs_excursion;not_a_literal_fgsea_scoreType;gseaParam=0"
    )


def test_equal_strength_blocks_are_order_invariant_and_never_name_broken() -> None:
    pairs = [("hit", "R"), ("miss", "R"), ("other1", "R"), ("other2", "R")]
    ranked = _ranked([5.0, 5.0, 2.0, 1.0], pairs)
    expected = _expected({0.1: [pairs[0]]})
    spec = SpatialDESSpec(top_fractions=(0.1,))

    first = evaluate_spatial_des(ranked, expected, spec)
    second = evaluate_spatial_des(
        ranked.iloc[[1, 0, 3, 2]].reset_index(drop=True), expected, spec
    )

    pd.testing.assert_frame_equal(first.scores, second.scores)
    pd.testing.assert_frame_equal(first.coverage, second.coverage)
    assert first.scores.iloc[0]["des"] == pytest.approx(2 / 3)
    assert first.scores.iloc[0]["tie_policy"] == ("simultaneous_equal_strength_blocks")
    assert first.coverage.iloc[0]["tied_strength_blocks"] == 1
    assert first.coverage.iloc[0]["ranked_pairs_in_ties"] == 2


def test_sender_receiver_pairs_are_directional() -> None:
    pairs = [("A", "B"), ("B", "A"), ("C", "D")]
    ranked = _ranked([3.0, 2.0, 1.0], pairs)
    expected = _expected({0.1: [("A", "B")]})

    result = evaluate_spatial_des(
        ranked, expected, SpatialDESSpec(top_fractions=(0.1,))
    )

    assert result.scores.iloc[0]["des"] == pytest.approx(1.0)
    assert result.coverage.iloc[0]["expected_pairs"] == 1
    assert result.coverage.iloc[0]["ranked_background_pairs"] == 2


def test_paper_unordered_mode_requires_precollapsed_canonical_pairs() -> None:
    ranked = _ranked(
        [3.0, 2.0, 1.0],
        [("A", "A"), ("A", "B"), ("C", "D")],
    )
    expected = _expected({0.1: [("A", "B")]})
    spec = SpatialDESSpec(top_fractions=(0.1,), cell_pair_mode="unordered")

    result = evaluate_spatial_des(ranked, expected, spec)

    assert result.scores.iloc[0]["cell_pair_direction"] == (
        "unordered_directions_collapsed"
    )
    reversed_rank = ranked.copy()
    reversed_rank.loc[1, ["sender", "receiver"]] = ["B", "A"]
    with pytest.raises(ValueError, match="pre-collapsed and canonicalized"):
        evaluate_spatial_des(reversed_rank, expected, spec)


def test_missing_ranked_rows_reduce_coverage_and_are_not_zero_imputed() -> None:
    pairs = [("S1", "R"), ("S2", "R"), ("S3", "R")]
    ranked = _ranked(
        [3.0, None, 1.0],
        pairs,
        statuses=["observed", "not_estimable", "observed"],
    )
    expected = _expected({0.1: [pairs[1]]})

    result = evaluate_spatial_des(
        ranked, expected, SpatialDESSpec(top_fractions=(0.1,))
    )
    score = result.scores.iloc[0]
    coverage = result.coverage.iloc[0]

    assert score["status"] == "not_estimable"
    assert score["reason_code"] == "no_expected_pairs_covered"
    assert math.isnan(float(score["des"]))
    assert coverage["expected_pair_coverage_fraction"] == pytest.approx(0.0)
    assert coverage["rank_pairs_eligible"] == 2
    assert coverage["rank_pairs_noneligible"] == 1
    assert coverage["rank_status_counts_json"] == ('{"not_estimable":1,"observed":2}')


def test_missing_fraction_and_empty_materialized_set_emit_distinct_ne_rows() -> None:
    pairs = [("S1", "R"), ("S2", "R")]
    ranked = _ranked([2.0, 1.0], pairs)
    expected = pd.DataFrame(
        {
            "dataset": ["fixture", "fixture"],
            "condition": ["case", "case"],
            "top_fraction": [0.1, 0.1],
            "sender": ["S1", "S2"],
            "receiver": ["R", "R"],
            "is_expected": [False, False],
        }
    )
    spec = SpatialDESSpec(
        top_fractions=(0.1, 0.2), expected_member_column="is_expected"
    )

    result = evaluate_spatial_des(ranked, expected, spec)
    scores = result.scores.set_index("top_fraction")

    assert scores.loc[0.1, "reason_code"] == "expected_spatial_set_empty"
    assert scores.loc[0.2, "reason_code"] == "expected_spatial_set_missing"
    assert not bool(
        result.coverage.set_index("top_fraction").loc[0.2, "expected_set_available"]
    )


def test_all_ranked_pairs_are_hits_is_explicitly_not_estimable() -> None:
    pairs = [("S1", "R"), ("S2", "R")]
    ranked = _ranked([2.0, 1.0], pairs)
    expected = _expected({0.1: pairs})

    result = evaluate_spatial_des(
        ranked, expected, SpatialDESSpec(top_fractions=(0.1,))
    )

    assert result.scores.iloc[0]["reason_code"] == "no_ranked_background_pairs"
    assert result.coverage.iloc[0]["expected_pair_coverage_fraction"] == 1.0


def test_lower_is_better_and_direction_column_are_supported() -> None:
    pairs = [("S1", "R"), ("S2", "R"), ("S3", "R")]
    ranked = _ranked([0.1, 0.5, 0.9], pairs).assign(score_direction="lower")
    expected = _expected({0.1: [pairs[0]]})
    spec = SpatialDESSpec(top_fractions=(0.1,), direction_column="score_direction")

    result = evaluate_spatial_des(ranked, expected, spec)

    assert result.scores.iloc[0]["des"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("which", "message"),
    [
        ("rank", "duplicate method-condition ordered cell-pair"),
        ("expected", "duplicate condition-fraction ordered cell-pair"),
    ],
)
def test_duplicate_pairs_fail_closed(which: str, message: str) -> None:
    pairs = [("S1", "R"), ("S2", "R")]
    ranked = _ranked([2.0, 1.0], pairs)
    expected = _expected({0.1: [pairs[0]]})
    if which == "rank":
        ranked = pd.concat([ranked, ranked.iloc[[0]]], ignore_index=True)
    else:
        expected = pd.concat([expected, expected.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=message):
        evaluate_spatial_des(ranked, expected, SpatialDESSpec(top_fractions=(0.1,)))


@pytest.mark.parametrize(
    ("strengths", "statuses", "message"),
    [
        ([None, 1.0], ["observed", "observed"], "eligible.*finite"),
        ([2.0, 1.0], ["missing", "observed"], "non-eligible.*missing"),
        ([float("inf"), 1.0], ["observed", "observed"], "finite numeric"),
    ],
)
def test_invalid_score_missingness_contract_fails_closed(
    strengths: list[float | None], statuses: list[str], message: str
) -> None:
    pairs = [("S1", "R"), ("S2", "R")]
    ranked = _ranked(strengths, pairs, statuses=statuses)
    expected = _expected({0.1: [pairs[0]]})

    with pytest.raises(ValueError, match=message):
        evaluate_spatial_des(ranked, expected, SpatialDESSpec(top_fractions=(0.1,)))


def test_expected_sets_must_be_nested_across_top_fractions() -> None:
    pairs = [("S1", "R"), ("S2", "R"), ("S3", "R")]
    ranked = _ranked([3.0, 2.0, 1.0], pairs)
    expected = _expected({0.1: [pairs[0]], 0.2: [pairs[1]]})

    with pytest.raises(ValueError, match="must be nested"):
        evaluate_spatial_des(
            ranked,
            expected,
            SpatialDESSpec(top_fractions=(0.1, 0.2)),
        )


def test_full_membership_tables_require_a_fixed_spatial_pair_universe() -> None:
    pairs = [("S1", "R"), ("S2", "R"), ("S3", "R")]
    ranked = _ranked([3.0, 2.0, 1.0], pairs)
    expected = pd.DataFrame(
        {
            "dataset": ["fixture", "fixture", "fixture"],
            "condition": ["case", "case", "case"],
            "top_fraction": [0.1, 0.1, 0.2],
            "sender": ["S1", "S2", "S1"],
            "receiver": ["R", "R", "R"],
            "is_expected": [True, False, True],
        }
    )
    spec = SpatialDESSpec(
        top_fractions=(0.1, 0.2), expected_member_column="is_expected"
    )

    with pytest.raises(ValueError, match="same spatial cell-pair universe"):
        evaluate_spatial_des(ranked, expected, spec)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_fractions": (0.5,)},
        {"top_fractions": (0.1, 0.1)},
        {"score_type": "std"},
    ],
)
def test_invalid_scoring_policy_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SpatialDESSpec(**kwargs)  # type: ignore[arg-type]
