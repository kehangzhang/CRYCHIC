from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.literature.run_two_part_occurrence_real import (
    _assert_rank_parity,
    _weighted_rankings,
    derive_subject_hurdle_inputs,
)


def test_subject_hurdle_collapses_replicates_and_preserves_structural_absence() -> None:
    obs = pd.DataFrame(
        {
            "replicate_id": ["r1", "r2", "r1", "t1"],
            "subject_id": ["S1", "S1", "S1", "S2"],
            "condition": ["R", "R", "R", "T"],
            "cell_type": ["A", "A", "B", "A"],
            "psbulk_n_cells": [10.0, 30.0, 20.0, 25.0],
        },
        index=["r1_A", "r2_A", "r1_B", "t1_A"],
    )
    counts = np.array([[20.0, 0.0], [90.0, 0.0], [0.0, 80.0], [60.0, 0.0]])
    props = np.array([[0.2, 0.0], [0.4, 0.0], [0.0, 0.5], [0.3, 0.0]])
    pdata = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=["L", "R"]),
    )
    pdata.layers["psbulk_props"] = props
    edges = pd.DataFrame(
        {
            "sender": ["A"],
            "receiver": ["B"],
            "interaction_id": ["i1"],
            "ligand": ["L"],
            "receptor": ["R"],
        }
    )
    presence, magnitude, design, support, edge_subject = derive_subject_hurdle_inputs(
        pdata, edges
    )
    assert design["subject_id"].tolist() == ["S1", "S2"]
    assert presence[0, 0] > 0.5
    assert magnitude[0, 0] > 0.0
    assert np.isnan(presence[0, 1])
    assert np.isnan(magnitude[0, 1])
    s1_sender = support.loc[
        support["subject_id"].eq("S1") & support["cell_type"].eq("A")
    ].iloc[0]
    assert s1_sender["pseudobulk_samples"] == 2
    assert s1_sender["subject_cell_count"] == 40.0
    assert edge_subject["status"].tolist() == ["observed", "structural_absence"]


def test_unit_weights_reproduce_rc3_rankings() -> None:
    edges = pd.DataFrame(
        {
            "sender": ["A", "B", "A"],
            "receiver": ["B", "A", "B"],
            "interaction_pvalue": [0.01, 0.02, 0.03],
            "final_sign_statistic": [1.0, 2.0, -1.0],
        }
    )
    template = pd.DataFrame(
        {
            "dataset": ["d", "d", "d"],
            "method": ["old"] * 3,
            "method_version": ["old"] * 3,
            "resource": ["r"] * 3,
            "ranking_semantics": ["old"] * 3,
            "condition": ["T", "R", "T"],
            "sender": ["A", "A", "A"],
            "receiver": ["B", "B", "C"],
            "ranked_strength": pd.array([2.0, 1.0, pd.NA], dtype="Float64"),
            "condition_specific_directed_lr": pd.array(
                [2.0, 1.0, pd.NA], dtype="Float64"
            ),
            "estimable_directed_lr": pd.array([3.0, 3.0, pd.NA], dtype="Float64"),
            "status": ["observed", "observed", "not_estimable"],
            "reason_code": [None, None, "structural_absence"],
        }
    )
    rebuilt, calls = _weighted_rankings(
        edges,
        template,
        np.ones(len(edges)),
        target="T",
        reference="R",
        method="new",
        semantics="test",
    )
    assert calls == 3
    assert _assert_rank_parity(rebuilt, template)["rankings_exact"] is True
    absent = rebuilt.loc[rebuilt["receiver"].eq("C")].iloc[0]
    assert pd.isna(absent["ranked_strength"])
