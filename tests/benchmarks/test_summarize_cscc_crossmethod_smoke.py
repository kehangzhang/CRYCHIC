from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
from benchmarks.summarize_cscc_crossmethod_smoke import (
    EDGE_KEYS,
    FOCUS_EDGES,
    TRACK_A_MAIN_ORDER,
    _crychic_effects_and_coverage,
    _figure_edge_selection,
    pairwise_effect_concordance,
)


def _crychic_artifact() -> dict[str, object]:
    effects: list[dict[str, object]] = []
    edges = (
        ("CD1C", "CD1C", "interaction-1", "CCL3", "CCR5"),
        ("Epithelial", "Epithelial", "interaction-2", "HBEGF", "EGFR"),
    )
    for mode in ("state", "ecosystem"):
        for index, (sender, receiver, interaction, ligand, receptor) in enumerate(
            edges
        ):
            effects.append(
                {
                    "mode": mode,
                    "sender": sender,
                    "receiver": receiver,
                    "interaction_id": interaction,
                    "ligand": ligand,
                    "receptor": receptor,
                    "effect": 0.2 - index * 0.3,
                    "median_effect": 0.1 - index * 0.2,
                    "direction_consistency": 1.0,
                    "direction_comparable_pairs": 4,
                    "n_pairs": 4,
                    "status": "exploratory",
                    "reason_code": None,
                }
            )
    return {
        "schema_version": "crychic-cscc-paired-gate-smoke-v2",
        "scope": "bounded_real_data_algorithm_smoke_not_biological_validation",
        "runtime": {"total_elapsed_seconds": 12.5},
        "diagnostic_score_summary": {
            "scope": ("exploratory_unadjusted_noncertified_paired_rank_effects"),
            "effect_semantics": "tumor_minus_normal_comparison_strength",
            "inferential_fields_available": [],
            "raw_sample_and_subject_rows_exported": False,
            "modes": [
                {
                    "mode": mode,
                    "n_samples": 8,
                    "n_subjects": 4,
                    "universe_size": 2,
                    "input_status_counts": {"ok": 4, "structural_zero": 12},
                    "effect_status_counts": {"exploratory": 2},
                }
                for mode in ("state", "ecosystem")
            ],
            "paired_effects": effects,
            "selected_penalties": [],
        },
    }


def _track_a_effects() -> pd.DataFrame:
    edges = (
        ("CD1C", "CD1C", "i1", "CCL3", "CCR5"),
        ("Epithelial", "CD1C", "i2", "CCL3", "CCR5"),
        ("CD1C", "Epithelial", "i3", "HBEGF", "EGFR"),
        ("Epithelial", "Epithelial", "i4", "TGFA", "EGFR"),
    )
    vectors = {
        "cellchat": [0.1, 0.2, 0.3, 0.4],
        "cellphonedb": [0.4, 0.3, 0.2, 0.1],
        "liana_rank_aggregate": [0.2, 0.4, 0.1, 0.3],
        "crychic_state": [0.1, 0.3, 0.2, 0.4],
    }
    rows = []
    for method, values in vectors.items():
        for edge, effect in zip(edges, values, strict=True):
            rows.append(
                {
                    "method": method,
                    **dict(zip(EDGE_KEYS, edge, strict=True)),
                    "effect": effect,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_crychic_effects_remain_deidentified_and_noncertified() -> None:
    effects, coverage = _crychic_effects_and_coverage(_crychic_artifact())

    assert len(effects) == 4
    assert set(effects["method"]) == {"crychic_state", "crychic_ecosystem"}
    assert not effects["is_official"].any()
    assert not effects["is_oof_certified"].any()
    assert set(effects["status"]) == {"exploratory"}
    assert set(coverage["n_estimable_effect_rows"]) == {2}
    assert set(coverage["native_observed_fraction"]) == {0.25}


def test_crychic_summary_rejects_inferential_field_claims() -> None:
    artifact = deepcopy(_crychic_artifact())
    summary = artifact["diagnostic_score_summary"]
    assert isinstance(summary, dict)
    summary["inferential_fields_available"] = ["p_value"]

    with pytest.raises(ValueError, match="must not expose inferential fields"):
        _crychic_effects_and_coverage(artifact)


def test_pairwise_concordance_handles_same_method_without_duplicate_columns() -> None:
    concordance = pairwise_effect_concordance(_track_a_effects())
    matrix = concordance.pivot(
        index="method_left", columns="method_right", values="spearman"
    ).reindex(index=TRACK_A_MAIN_ORDER, columns=TRACK_A_MAIN_ORDER)

    assert matrix.loc["cellchat", "cellchat"] == pytest.approx(1.0)
    assert matrix.loc["cellchat", "cellphonedb"] == pytest.approx(-1.0)
    assert matrix.equals(matrix.T)
    assert set(concordance["n_shared_edges"]) == {4}


def test_figure_selection_keeps_frozen_biology_focus_edges_first() -> None:
    selected = _figure_edge_selection(_track_a_effects(), limit=4)
    observed = list(
        selected.loc[:, ["sender", "receiver", "ligand", "receptor"]].itertuples(
            index=False, name=None
        )
    )

    expected = [edge for edge in FOCUS_EDGES if edge in set(observed)]
    assert observed[: len(expected)] == expected
    assert selected["figure_order"].tolist() == list(range(len(selected)))
