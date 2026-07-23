from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.literature.event_level_des import (
    assert_original_count_parity,
    bounded_pair_gate,
    mechanism_annotations,
    original_count_version_audit,
    pair_rankings_from_events,
    prepare_crychic_event_ledger,
    prepare_scseqcommdiff_event_ledger,
    select_top_k_events,
    select_top_k_events_by_scope,
)


def _resource() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand": ["L1", "L2", "L3", "L4"],
            "receptor": ["R1", "R2", "R3", "R4"],
            "ligand_location": [
                "plasma membrane",
                "secreted",
                "ECM",
                "plasma membrane; secreted",
            ],
        }
    )


def _pair_axes() -> pd.DataFrame:
    rows = []
    for condition in ("A", "B"):
        for sender, receiver, score in (("X", "Y", 1.0), ("X", "Z", 3.0)):
            rows.append(
                {
                    "condition": condition,
                    "sender": sender,
                    "receiver": receiver,
                    "ranked_strength": score,
                    "status": "observed",
                }
            )
    return pd.DataFrame.from_records(rows)


def _crychic_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sender": ["X", "Y", "X", "Z"],
            "receiver": ["Y", "X", "Z", "X"],
            "interaction_id": ["e1", "e2", "e3", "e4"],
            "ligand": ["L1", "L2", "L3", "L4"],
            "receptor": ["R1", "R2", "R3", "R4"],
            "reference_condition": ["A"] * 4,
            "target_condition": ["B"] * 4,
            "effect_target_minus_reference": [2.0, 1.0, -4.0, 0.0],
            "effect_signal_to_noise": [2.0, 3.0, -4.0, 0.0],
            "one_standard_error_stable": [True, True, True, False],
            "status": ["observed"] * 4,
            "reason_code": [None] * 4,
        }
    )


def test_mechanism_annotations_keep_ambiguous_location_separate() -> None:
    lookup = mechanism_annotations(_resource())

    assert lookup.set_index("ligand").loc["L1", "mechanism"] == "contact"
    assert lookup.set_index("ligand").loc["L2", "mechanism"] == "secreted"
    assert lookup.set_index("ligand").loc["L3", "mechanism"] == "ecm_receptor"
    assert (
        lookup.set_index("ligand").loc["L4", "mechanism"]
        == "ambiguous_contact_secreted"
    )


def test_bounded_pair_gate_preserves_order_and_bounds() -> None:
    gate = bounded_pair_gate(_pair_axes(), floor=0.75)

    assert gate["pair_gate"].between(0.75, 1.0).all()
    for _, group in gate.groupby("condition"):
        values = group.set_index("receiver")["pair_gate"]
        assert values["Z"] > values["Y"]


def test_crychic_event_ledger_assigns_condition_and_bounded_evidence() -> None:
    ledger = prepare_crychic_event_ledger(
        _crychic_events(),
        _pair_axes(),
        mechanism_annotations(_resource()),
        pair_gate_floor=0.75,
    ).set_index("interaction_id")

    assert ledger.loc["e1", "condition"] == "B"
    assert ledger.loc["e3", "condition"] == "A"
    assert ledger.loc["e4", "condition"] == "tied"
    assert ledger.loc["e1", "pair_sender"] == "X"
    assert ledger.loc["e2", "pair_receiver"] == "Y"
    assert ledger.loc["e3", "bounded_abs_effect"] == pytest.approx(4.0)
    assert ledger.loc["e1", "bounded_event_evidence"] <= 2.0


def test_top_k_is_global_exact_and_deterministic() -> None:
    ledger = prepare_crychic_event_ledger(
        _crychic_events(),
        _pair_axes(),
        mechanism_annotations(_resource()),
        pair_gate_floor=0.75,
    )
    eligible = ledger["status"].eq("observed") & ledger["abs_effect"].gt(0.0)

    first = select_top_k_events(
        ledger,
        budget=2,
        evidence_column="bounded_event_evidence",
        eligible=eligible,
    )
    second = select_top_k_events(
        ledger.sample(frac=1.0, random_state=17).sort_index(),
        budget=2,
        evidence_column="bounded_event_evidence",
        eligible=eligible,
    )

    assert int(first.sum()) == 2
    assert first.equals(second)
    selected_ids = set(ledger.loc[first, "interaction_id"])
    assert selected_ids == {"e2", "e3"}


def test_top_k_per_condition_keeps_an_exact_budget_in_each_direction() -> None:
    ledger = prepare_crychic_event_ledger(
        _crychic_events(),
        _pair_axes(),
        mechanism_annotations(_resource()),
        pair_gate_floor=0.0,
    )
    eligible = ledger["status"].eq("observed") & ledger["abs_effect"].gt(0.0)

    selected = select_top_k_events_by_scope(
        ledger,
        budget=1,
        evidence_column="bounded_event_evidence",
        eligible=eligible,
        scope="per_condition",
    )

    assert int(selected.sum()) == 2
    assert ledger.loc[selected].groupby("condition").size().to_dict() == {
        "A": 1,
        "B": 1,
    }


def test_pair_collapse_sums_both_sender_directions() -> None:
    ledger = prepare_crychic_event_ledger(
        _crychic_events(),
        _pair_axes(),
        mechanism_annotations(_resource()),
        pair_gate_floor=0.75,
    )
    ranking = pair_rankings_from_events(
        ledger,
        _pair_axes(),
        selected=ledger["native_selected"],
        weight_column=None,
        metadata={
            "dataset": "fixture",
            "method": "crychic",
            "method_version": "fixture",
            "resource": "fixture",
            "ranking_semantics": "original_count",
        },
    )

    b_xy = ranking.loc[
        ranking["condition"].eq("B")
        & ranking["sender"].eq("X")
        & ranking["receiver"].eq("Y")
    ].iloc[0]
    a_xz = ranking.loc[
        ranking["condition"].eq("A")
        & ranking["sender"].eq("X")
        & ranking["receiver"].eq("Z")
    ].iloc[0]
    assert b_xy["ranked_strength"] == 2.0
    assert b_xy["condition_specific_directed_lr"] == 2
    assert a_xz["ranked_strength"] == 1.0


def test_scseq_original_count_rebuild_can_be_checked_exactly() -> None:
    events = pd.DataFrame(
        {
            "sender": ["X", "Y", "X"],
            "receiver": ["Y", "X", "Z"],
            "ligand": ["L1", "L2", "L3"],
            "receptor": ["R1", "R2", "R3"],
            "score_target": [3.0, 2.0, 1.0],
            "score_reference": [1.0, 1.0, 3.0],
            "effect_target_minus_reference": [2.0, 1.0, -2.0],
            "logFC": [1.0, 0.5, -1.0],
            "p_value": [0.01, 0.02, 0.01],
            "max_S_intra": [np.nan, 0.8, 0.2],
            "reference_condition": ["A"] * 3,
            "target_condition": ["B"] * 3,
            "status": ["observed"] * 3,
            "reason_code": [None] * 3,
        }
    )
    ledger = prepare_scseqcommdiff_event_ledger(
        events, mechanism_annotations(_resource())
    )
    rebuilt = pair_rankings_from_events(
        ledger,
        _pair_axes(),
        selected=ledger["native_selected"],
        weight_column=None,
        metadata={
            "dataset": "fixture",
            "method": "scseqcommdiff",
            "method_version": "2.0.0",
            "resource": "fixture",
            "ranking_semantics": "native_original_count",
        },
    )
    native = rebuilt.copy()
    assert_original_count_parity(rebuilt, native)

    altered = native.copy()
    altered.loc[0, "ranked_strength"] += 1.0
    with pytest.raises(ValueError, match="values disagree"):
        assert_original_count_parity(rebuilt, altered)


def test_scseq_infinite_logfc_is_countable_but_not_continuously_weighted() -> None:
    events = pd.DataFrame(
        {
            "sender": ["X"],
            "receiver": ["Y"],
            "ligand": ["L1"],
            "receptor": ["R1"],
            "score_target": [np.nan],
            "score_reference": [0.0],
            "effect_target_minus_reference": [np.nan],
            "logFC": [np.inf],
            "p_value": [0.01],
            "max_S_intra": [np.nan],
            "reference_condition": ["A"],
            "target_condition": ["B"],
            "status": ["observed"],
            "reason_code": [None],
        }
    )

    ledger = prepare_scseqcommdiff_event_ledger(
        events, mechanism_annotations(_resource())
    )

    assert ledger.loc[0, "condition"] == "B"
    assert bool(ledger.loc[0, "native_selected"])
    assert bool(ledger.loc[0, "top_k_eligible"])
    assert pd.isna(ledger.loc[0, "continuous_weight"])


def test_original_count_version_audit_reports_but_does_not_reject_change() -> None:
    current = _pair_axes()
    historical = _pair_axes()
    current.loc[0, "ranked_strength"] = 7.0

    audit, summary = original_count_version_audit(current, historical)

    assert not summary["exact_parity"]
    assert summary["changed_count_axes"] == 1
    assert summary["exact_count_axes"] == 3
    changed = audit.loc[
        audit["count_delta_current_minus_historical"].ne(0.0)
    ].iloc[0]
    assert changed["count_delta_current_minus_historical"] == 6.0


def test_top_k_rejects_an_unavailable_budget() -> None:
    ledger = prepare_crychic_event_ledger(
        _crychic_events(),
        _pair_axes(),
        mechanism_annotations(_resource()),
        pair_gate_floor=0.75,
    )
    with pytest.raises(ValueError, match="fewer than K=10"):
        select_top_k_events(
            ledger,
            budget=10,
            evidence_column="bounded_event_evidence",
            eligible=ledger["abs_effect"].gt(0.0),
        )
