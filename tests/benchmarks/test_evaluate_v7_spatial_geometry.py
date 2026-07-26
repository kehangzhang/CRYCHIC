from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.literature.evaluate_v7_spatial_geometry import (
    TRUTH_SUPPORT_COLUMNS,
    annotate_event_mechanisms,
    build_mechanism_pair_rankings,
    evaluate_geometry_des,
    geometry_pair_axes,
    geometry_rank_concordance,
    geometry_truth_scenarios,
    mechanism_distance_summary,
)

BANDS = ("contact", "short", "local", "diffuse")
CONDITION_MAP = {"ref": "control", "target": "treated"}


def _alignment() -> dict[str, object]:
    return {
        "generators": ["G0"],
        "mechanisms": ["all", "contact", "secreted"],
        "event_budgets": [1],
        "continuous_weight": "abs_effect",
        "top_k_evidence": "event_evidence",
        "top_k_scope": "global_across_both_effect_directions",
        "score_type": "pos",
        "weight_exponent": 1.0,
        "tie_policy": "fgsea_native",
        "mechanism_band_hypotheses": {
            "all": list(BANDS),
            "contact": ["contact"],
            "secreted": ["local", "diffuse"],
        },
    }


def _resource() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand": ["L1", "L2", "L3", "L4"],
            "receptor": ["R1", "R2", "R3", "R4"],
            "ligand_location": [
                "plasma membrane",
                "plasma membrane",
                "secreted",
                "plasma membrane; secreted",
            ],
        }
    )


def _ledger() -> pd.DataFrame:
    rows = (
        ("i1", "A", "B", "L1", "R1", 5.0, "target", True),
        ("i2", "B", "A", "L2", "R2", 1.0, "target", False),
        ("i3", "A", "C", "L3", "R3", -4.0, "ref", True),
        ("i4", "B", "C", "L4", "R4", 2.0, "target", True),
    )
    return pd.DataFrame.from_records(
        {
            "generator_id": "G0",
            "condition": condition,
            "sender": sender,
            "receiver": receiver,
            "pair_sender": min(sender, receiver),
            "pair_receiver": max(sender, receiver),
            "interaction_id": interaction,
            "ligand": ligand,
            "receptor": receptor,
            "effect_target_minus_reference": effect,
            "abs_effect": abs(effect),
            "event_evidence": abs(effect),
            "native_selected": native,
            "status": "observed",
        }
        for (
            interaction,
            sender,
            receiver,
            ligand,
            receptor,
            effect,
            condition,
            native,
        ) in rows
    )


def _geometry_effects() -> pd.DataFrame:
    pairs = (
        ("A", "B", 3.0, 0.01, 0.01),
        ("A", "C", -2.0, 0.01, 0.20),
        ("B", "C", 1.0, 0.20, 0.01),
    )
    return pd.DataFrame.from_records(
        {
            "dataset": "geometry-dataset",
            "analysis_unit": "subject_id",
            "band": band,
            "sender": sender,
            "receiver": receiver,
            "status": "observed",
            "absolute_effect": abs(effect),
            "effect_target_minus_reference": effect,
            "reference": "control",
            "target": "treated",
            "coordinate_max_t_p_value": coordinate_p,
            "cell_label_max_t_p_value": label_p,
        }
        for band in BANDS
        for sender, receiver, effect, coordinate_p, label_p in pairs
    )


def _expected_sets() -> pd.DataFrame:
    effects = _geometry_effects()
    records: list[dict[str, object]] = []
    for row in effects.to_dict(orient="records"):
        expected_condition = (
            "treated" if float(row["effect_target_minus_reference"]) > 0 else "control"
        )
        for fraction in (0.1, 0.2, 0.3, 0.4):
            records.append(
                {
                    **row,
                    "top_fraction": fraction,
                    "is_expected": (row["sender"], row["receiver"])
                    in {("A", "B"), ("A", "C")},
                    "expected_condition": expected_condition,
                }
            )
    return pd.DataFrame.from_records(records)


def _rankings_and_effects() -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = _geometry_effects()
    axes, primary = geometry_pair_axes(
        effects,
        dataset_id="geometry-dataset",
        analysis_unit="subject_id",
        bands=BANDS,
        algorithm_conditions=tuple(CONDITION_MAP),
    )
    ledger = annotate_event_mechanisms(_ledger(), _resource(), generators=("G0",))
    rankings = build_mechanism_pair_rankings(
        ledger,
        axes,
        dataset_id="algorithm-dataset",
        resource_id="fixture-resource",
        alignment=_alignment(),
    )
    return rankings, primary


def test_event_mechanisms_and_direction_collapse_are_exact() -> None:
    rankings, _ = _rankings_and_effects()

    continuous = rankings.loc[
        rankings["des_variant"].eq("continuous_weighted_des")
    ]
    all_target = continuous.loc[
        continuous["mechanism"].eq("all") & continuous["condition"].eq("target")
    ].set_index(["sender", "receiver"])
    contact_target = continuous.loc[
        continuous["mechanism"].eq("contact")
        & continuous["condition"].eq("target")
    ].set_index(["sender", "receiver"])
    secreted_reference = continuous.loc[
        continuous["mechanism"].eq("secreted")
        & continuous["condition"].eq("ref")
    ].set_index(["sender", "receiver"])

    assert all_target.loc[("A", "B"), "ranked_strength"] == pytest.approx(6.0)
    assert contact_target.loc[("A", "B"), "ranked_strength"] == pytest.approx(6.0)
    assert secreted_reference.loc[("A", "C"), "ranked_strength"] == pytest.approx(
        4.0
    )
    assert set(rankings["des_variant"]) == {
        "continuous_weighted_des",
        "diagnostic_one_se_native_count_des",
        "top_k_count_des",
    }


def test_geometry_truth_variants_apply_global_max_t_support() -> None:
    truth, scenarios = geometry_truth_scenarios(
        _expected_sets(),
        dataset_id="geometry-dataset",
        analysis_unit="subject_id",
        bands=BANDS,
        truth_variants=tuple(TRUTH_SUPPORT_COLUMNS),
        geometry_conditions=tuple(CONDITION_MAP.values()),
        diagnostic_alpha=0.05,
    )

    coordinate = truth.loc[
        truth["scenario"].eq(
            "geometry_contact__coordinate_max_t_supported_rank_top"
        )
        & truth["top_fraction"].eq(0.1)
        & truth["is_expected"]
    ]
    joint = truth.loc[
        truth["scenario"].eq(
            "geometry_contact__coordinate_and_cell_label_max_t_supported_rank_top"
        )
        & truth["top_fraction"].eq(0.1)
        & truth["is_expected"]
    ]
    coordinate_pairs = set(
        coordinate[["sender", "receiver"]].itertuples(index=False, name=None)
    )
    assert coordinate_pairs == {
        ("A", "B"),
        ("A", "C"),
    }
    assert set(joint[["sender", "receiver"]].itertuples(index=False, name=None)) == {
        ("A", "B")
    }
    assert len(scenarios) == len(BANDS) * len(TRUTH_SUPPORT_COLUMNS)


def test_des_matrix_and_concordance_keep_claim_boundaries() -> None:
    rankings, effects = _rankings_and_effects()
    truth, scenarios = geometry_truth_scenarios(
        _expected_sets(),
        dataset_id="geometry-dataset",
        analysis_unit="subject_id",
        bands=BANDS,
        truth_variants=tuple(TRUTH_SUPPORT_COLUMNS),
        geometry_conditions=tuple(CONDITION_MAP.values()),
        diagnostic_alpha=0.05,
    )

    scores, coverage = evaluate_geometry_des(
        rankings,
        truth,
        scenarios,
        algorithm_dataset_id="algorithm-dataset",
        geometry_dataset_id="geometry-dataset",
        condition_map=CONDITION_MAP,
        alignment=_alignment(),
    )
    concordance = geometry_rank_concordance(
        rankings,
        effects,
        condition_map=CONDITION_MAP,
        alignment=_alignment(),
    )
    summary = mechanism_distance_summary(scores)

    assert set(scores["band"]) == set(BANDS)
    assert set(scores["truth_variant"]) == set(TRUTH_SUPPORT_COLUMNS)
    assert {1, pd.NA} == set(scores["event_budget"].unique())
    assert not scores["formal_inference_allowed"].any()
    assert not coverage["formal_inference_allowed"].any()
    all_target = concordance.loc[
        concordance["generator_id"].eq("G0")
        & concordance["mechanism"].eq("all")
        & concordance["condition"].eq("treated")
        & concordance["band"].eq("contact")
    ].iloc[0]
    assert all_target["spearman"] == pytest.approx(1.0)
    assert set(summary["generator_rank"].dropna().astype(int)) == {1}


def test_geometry_pair_axes_reject_incomplete_cell_pair_universe() -> None:
    incomplete = _geometry_effects().loc[
        lambda table: ~(
            table["sender"].eq("B") & table["receiver"].eq("C")
        )
    ]
    with pytest.raises(ValueError, match="complete non-self universe"):
        geometry_pair_axes(
            incomplete,
            dataset_id="geometry-dataset",
            analysis_unit="subject_id",
            bands=BANDS,
            algorithm_conditions=tuple(CONDITION_MAP),
        )


def test_mechanism_alignment_requires_exact_generator_axis() -> None:
    with pytest.raises(ValueError, match="generator axis changed"):
        annotate_event_mechanisms(_ledger(), _resource(), generators=("G0", "G2"))
