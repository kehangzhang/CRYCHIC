from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.literature.evaluate_citeseq import (
    component_scores_from_cellchat,
    component_scores_from_liana,
    evaluate_citeseq_components,
    scores_from_crychic_availability,
)


def _raw_liana() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["s", "s", "s"],
            "target": ["a", "b", "c"],
            "ligand_complex": ["L"] * 3,
            "receptor_complex": ["R"] * 3,
            "lr_means": [3.0, 2.0, 1.0],
            "cellphone_pvals": [0.01, 0.20, 0.03],
            "expr_prod": [3.0, 2.0, 1.0],
            "scaled_weight": [3.0, 2.0, 1.0],
            "lr_logfc": [3.0, 2.0, 1.0],
            "spec_weight": [3.0, 2.0, 1.0],
            "lrscore": [0.9, 0.4, 0.6],
            "magnitude_rank": [0.1, 0.5, 0.9],
            "specificity_rank": [0.1, 0.5, 0.9],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["cite"] * 3,
            "receiver": ["a", "b", "c"],
            "receptor": ["R"] * 3,
            "is_positive": [1, 0, 0],
            "truth_status": ["observed"] * 3,
        }
    )


def test_component_conversion_applies_published_filters_and_directions() -> None:
    result = component_scores_from_liana(
        _raw_liana(), dataset="cite", resource_mode="H-common"
    )

    counts = result.groupby("method").size()
    assert counts["CellPhoneDB composite"] == 2
    assert counts["CellPhoneDB p-value"] == 3
    assert counts["SingleCellSignalR LRscore"] == 2
    assert set(
        result.loc[result["method"].eq("LIANA magnitude consensus"), "score_direction"]
    ) == {"lower"}


def test_cellchat_conversion_uses_complex_keys_without_duplicate_columns() -> None:
    raw = pd.DataFrame(
        {
            "source": ["s", "s"],
            "target": ["a", "b"],
            "ligand": ["L-subunit", "L-subunit"],
            "ligand_complex": ["L", "L"],
            "receptor": ["R-subunit", "R-subunit"],
            "receptor_complex": ["R", "R"],
            "lr_probs": [0.8, 0.2],
            "cellchat_pvals": [0.01, 0.20],
        }
    )

    result = component_scores_from_cellchat(raw, dataset="cite", resource_mode="native")

    assert set(result["ligand"]) == {"L"}
    assert set(result["receptor"]) == {"R"}
    assert result.groupby("method").size().to_dict() == {
        "CellChat composite": 1,
        "CellChat p-value": 2,
    }


def test_independent_and_intersection_evaluations_are_explicit() -> None:
    scores = component_scores_from_liana(
        _raw_liana(), dataset="cite", resource_mode="H-common"
    )
    _, independent = evaluate_citeseq_components(
        scores,
        _truth(),
        universe_mode="independent",
        n_bootstrap=10,
        n_negative_samples=10,
    )
    retained, intersection = evaluate_citeseq_components(
        scores,
        _truth(),
        universe_mode="intersection",
        n_bootstrap=10,
        n_negative_samples=10,
    )

    assert set(independent.point_estimates["universe_mode"]) == {"independent"}
    assert set(intersection.point_estimates["universe_mode"]) == {"intersection"}
    assert retained.groupby("method").size().nunique() == 1
    perfect = independent.point_estimates.loc[
        independent.point_estimates["method"].eq("LIANA magnitude consensus")
    ].iloc[0]
    assert perfect["auroc"] == pytest.approx(1.0)


def test_crychic_conversion_collapses_duplicate_edges_by_strongest_score() -> None:
    raw = pd.DataFrame(
        {
            "sender": ["s", "s"],
            "receiver": ["r", "r"],
            "ligand": ["L", "L"],
            "receptor": ["R", "R"],
            "availability_state": [0.2, 0.8],
            "status": ["observed", "observed"],
        }
    )

    result = scores_from_crychic_availability(
        raw, dataset="cite", resource_mode="H-common"
    )

    assert len(result) == 1
    assert result.iloc[0]["score"] == pytest.approx(0.8)


def test_resource_fixed_evaluation_uses_identical_truth_for_every_method() -> None:
    liana = component_scores_from_liana(
        _raw_liana(), dataset="cite", resource_mode="H-common"
    )
    crychic = scores_from_crychic_availability(
        pd.DataFrame(
            {
                "sender": ["a"],
                "receiver": ["a"],
                "ligand": ["L"],
                "receptor": ["R"],
                "availability_state": [0.8],
                "status": ["observed"],
            }
        ),
        dataset="cite",
        resource_mode="H-common",
    )
    retained, result = evaluate_citeseq_components(
        pd.concat((liana, crychic), ignore_index=True),
        _truth(),
        universe_mode="resource_fixed",
        resource=pd.DataFrame({"ligand": ["L"], "receptor": ["R"]}),
        n_bootstrap=10,
        n_negative_samples=10,
    )

    counts = result.point_estimates.set_index("method")[
        ["truth_universe_edges", "truth_positive_edges", "truth_negative_edges"]
    ]
    assert counts.nunique().eq(1).all()
    assert counts.iloc[0].to_dict() == {
        "truth_universe_edges": 9,
        "truth_positive_edges": 3,
        "truth_negative_edges": 6,
    }
    assert retained.groupby("method").size().nunique() == 1
