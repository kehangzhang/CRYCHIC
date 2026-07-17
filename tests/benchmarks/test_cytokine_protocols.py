from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.literature.evaluate_cytokine_protocols import (
    _balanced_auprc,
    fisher_rank_curves,
    ranking_metrics,
)


def _toy_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "target": ["T", "T"],
            "response": [1, 0],
        }
    )
    scores = pd.DataFrame(
        {
            "method_id": ["a", "a", "b"],
            "method_name": ["A", "A", "B"],
            "implementation": ["test"] * 3,
            "source": ["S1", "S2", "S1"],
            "target": ["T", "T", "T"],
            "ligand": ["L1", "L2", "L1"],
            "receptor": ["R1", "R2", "R1"],
            "score": [2.0, 1.0, 3.0],
        }
    )
    return scores, truth


def test_union_protocol_imputes_missing_stlr_rows() -> None:
    scores, truth = _toy_scores()
    curves = fisher_rank_curves(scores, truth)
    union = curves.loc[
        curves["protocol"].eq("paper_union_max_imputed")
        & curves["fisher_definition"].eq("paper_text_top_vs_remainder")
        & curves["rank_cutoff"].eq(100)
    ]

    assert set(union["evaluation_rows"]) == {2}
    assert "q_value_bh_primary_8arm" in curves
    assert dict(zip(union["method_id"], union["missing_rows"], strict=True)) == {
        "a": 0,
        "b": 1,
    }


def test_controlled_metrics_use_full_unique_truth() -> None:
    scores, truth = _toy_scores()
    metrics = ranking_metrics(scores, truth)
    controlled = metrics.loc[metrics["protocol"].eq("controlled_unique_ligand_target")]

    assert set(controlled["evaluation_rows"]) == {2}
    assert set(controlled["aggregation"]) == {"max", "sum"}


def test_balanced_auprc_is_reproducible() -> None:
    response = np.array([1, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])

    first = _balanced_auprc(response, score, iterations=10)
    second = _balanced_auprc(response, score, iterations=10)

    assert first == second
