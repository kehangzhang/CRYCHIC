from __future__ import annotations

import pandas as pd

from benchmarks.literature.prepare_scseqcommdiff_subject_tracks import (
    build_subject_event_track_rankings,
)


def test_subject_event_tracks_use_one_global_budget_across_directions() -> None:
    ledger = pd.DataFrame(
        {
            "condition": ["case", "case", "case", "control", "control"],
            "sender": ["A", "A", "B", "A", "B"],
            "receiver": ["B", "C", "C", "B", "C"],
            "ligand": [f"L{i}" for i in range(5)],
            "receptor": [f"R{i}" for i in range(5)],
            "pair_sender": ["A", "A", "B", "A", "B"],
            "pair_receiver": ["B", "C", "C", "B", "C"],
            "top_k_eligible": [True] * 5,
            "event_evidence": [5.0, 4.0, 3.0, 2.0, 1.0],
            "abs_effect": [0.5, 0.4, 0.3, 0.2, 0.1],
            "continuous_weight": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    axes = pd.DataFrame(
        {
            "condition": ["case"] * 3 + ["control"] * 3,
            "sender": [pair[0] for pair in pairs] * 2,
            "receiver": [pair[1] for pair in pairs] * 2,
            "ranked_strength": [1.0] * 6,
            "status": ["observed"] * 6,
        }
    )

    rankings, diagnostics = build_subject_event_track_rankings(
        ledger,
        axes,
        dataset_id="fixture",
        resource_id="resource",
        event_budgets=(1, 2),
    )

    fixed = rankings.loc[rankings["des_variant"].eq("top_k_count_des")]
    selected_totals = fixed.groupby("event_budget")["ranked_strength"].sum()
    assert selected_totals.to_dict() == {1: 1.0, 2: 2.0}
    assert diagnostics.loc[
        diagnostics["des_variant"].eq("top_k_count_des"), "selected_events"
    ].tolist() == [1, 2]
    continuous = rankings.loc[
        rankings["des_variant"].eq("continuous_weighted_des")
    ]
    assert continuous["ranked_strength"].sum() == 1.5
    assert set(rankings["method"]) == {"scseqcommdiff"}
