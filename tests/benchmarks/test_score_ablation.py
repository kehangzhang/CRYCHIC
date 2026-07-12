from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.metrics.score_ablation import evaluate_score_ablation


def _records(*, split_senders: bool = True) -> pd.DataFrame:
    component_values = {
        "active": (0.8, 0.7, 0.7),
        "global_null": (0.0, 0.0, 0.0),
        "ligand_only": (0.8, 0.3, 0.0),
        "target_only": (0.0, 0.7, 0.7),
        "receiver_autonomous": (0.0, 0.5, 0.0),
        "receptor_knockout": (0.0, 0.7, 0.7),
    }
    senders = (("S1", 0.6), ("S2", 0.4)) if split_senders else (("S", 1.0),)
    rows: list[dict[str, object]] = []
    for seed in range(4):
        for edge in ("E1", "E2"):
            for scenario, (
                availability,
                legacy_downstream,
                incremental,
            ) in component_values.items():
                for sender, weight in senders:
                    rows.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "known_edge_id": edge,
                            "sender_id": sender,
                            "availability": availability,
                            "legacy_downstream": legacy_downstream,
                            "incremental_downstream": incremental,
                            "sender_weight": weight,
                            "prior_quality": 1.0,
                            "status": "observed",
                            "reason_code": None,
                        }
                    )
    return pd.DataFrame(rows)


def _metric(result, formula: str, metric: str, scenario: str) -> pd.Series:
    selected = result.summary.loc[
        result.summary["formula"].eq(formula)
        & result.summary["metric"].eq(metric)
        & result.summary["scenario"].eq(scenario)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_mechanistic_formula_eliminates_ligand_only_false_activation() -> None:
    result = evaluate_score_ablation(_records())
    legacy = _metric(
        result,
        "legacy_geometric_sender_total",
        "negative_false_activation_rate",
        "ligand_only",
    )
    candidate = _metric(
        result,
        "mechanistic_sender_unresolved",
        "negative_false_activation_rate",
        "ligand_only",
    )

    assert legacy["estimate"] == pytest.approx(1.0)
    assert candidate["estimate"] == pytest.approx(0.0)
    margin = _metric(
        result,
        "mechanistic_sender_unresolved",
        "active_minus_ligand_only_paired_margin",
        "ligand_only",
    )
    assert margin["estimate"] > 0.5


def test_sender_resolved_strength_conserves_unresolved_core() -> None:
    result = evaluate_score_ablation(_records())
    observed = result.edge_scores.loc[result.edge_scores["status"].eq("observed")]

    assert observed["sender_conservation_error"].abs().max() < 1e-12
    conservation = result.summary.loc[
        result.summary["metric"].eq("maximum_absolute_sender_conservation_error")
    ].iloc[0]
    assert conservation["estimate"] < 1e-12


def test_sender_count_changes_legacy_total_but_not_mechanistic_lr_core() -> None:
    split = evaluate_score_ablation(_records(split_senders=True)).edge_scores
    single = evaluate_score_ablation(_records(split_senders=False)).edge_scores
    key = ["seed", "scenario", "known_edge_id"]
    merged = split.merge(single, on=key, suffixes=("_split", "_single"))

    assert (
        merged["mechanistic_sender_unresolved_split"]
        == merged["mechanistic_sender_unresolved_single"]
    ).all()
    active = merged["scenario"].eq("active")
    assert (
        merged.loc[active, "legacy_geometric_sender_total_split"]
        != merged.loc[active, "legacy_geometric_sender_total_single"]
    ).all()


def test_prior_quality_one_is_exactly_neutral_for_mechanistic_score() -> None:
    result = evaluate_score_ablation(_records())
    sender = result.sender_scores.loc[result.sender_scores["status"].eq("observed")]

    assert (sender["mechanistic_lr_core"] == sender["mechanistic_prior_adjusted"]).all()


def test_missing_edge_components_remain_not_estimable() -> None:
    records = _records()
    selected = (
        records["seed"].eq(0)
        & records["scenario"].eq("ligand_only")
        & records["known_edge_id"].eq("E1")
    )
    records.loc[selected, "status"] = "not_estimable"
    records.loc[selected, "reason_code"] = "component_missing"
    for column in (
        "availability",
        "legacy_downstream",
        "incremental_downstream",
        "sender_weight",
        "prior_quality",
    ):
        records.loc[selected, column] = None

    result = evaluate_score_ablation(records)
    affected = result.edge_scores.loc[
        result.edge_scores["seed"].eq(0)
        & result.edge_scores["scenario"].eq("ligand_only")
        & result.edge_scores["known_edge_id"].eq("E1")
    ].iloc[0]
    assert affected["status"] == "not_estimable"
    assert pd.isna(affected["mechanistic_sender_unresolved"])
