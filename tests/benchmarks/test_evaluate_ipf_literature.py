from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.literature.evaluate_ipf import (
    evaluate_ipf_components,
    restrict_ipf_universe_to_resource,
)
from benchmarks.literature.ipf import build_ipf_truth_universe


def _gold() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
        }
    )


def test_ipf_evaluation_expands_complexes_and_ranks_absent_edges_last() -> None:
    scores = pd.DataFrame(
        {
            "dataset": ["ipf", "ipf"],
            "resource_mode": ["H-common", "H-common"],
            "method": ["method", "method"],
            "sender": ["A", "A"],
            "receiver": ["B", "A"],
            "ligand": ["L1_LX", "L2"],
            "receptor": ["R1", "R2"],
            "score": [0.9, 0.1],
            "score_direction": ["higher", "higher"],
            "status": ["observed", "observed"],
        }
    )

    materialized, result = evaluate_ipf_components(
        scores,
        _gold(),
        universe_mode="paper_base_full",
        n_bootstrap=10,
        n_negative_samples=10,
    )

    point = result.point_estimates.iloc[0]
    assert point["truth_universe_edges"] == 8
    assert point["raw_returned_edges"] == 2
    assert point["raw_returned_positive_edges"] == 1
    assert point["raw_positive_return_fraction"] == pytest.approx(0.5)
    assert point["auroc"] > 0.5
    absent = materialized.loc[~materialized["raw_returned"], "score"]
    returned = materialized.loc[materialized["raw_returned"], "score"]
    assert absent.max() < returned.min()


def test_ipf_resource_restriction_preserves_intact_pairs() -> None:
    universe = build_ipf_truth_universe(_gold())
    resource = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]})

    result = restrict_ipf_universe_to_resource(universe, resource)

    assert len(result) == 4
    assert result["is_positive"].sum() == 1
    assert set(result["ligand"]) == {"L1"}
    assert set(result["receptor"]) == {"R1"}
