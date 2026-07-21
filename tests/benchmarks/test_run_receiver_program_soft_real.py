from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.literature.run_receiver_program_soft_real import (
    _assert_alpha_zero_parity,
    _weighted_strict_rankings,
    derive_receiver_program_effects,
)


def test_welch_program_effect_uses_sample_conditions() -> None:
    program = pd.DataFrame(
        {
            "sample_id": ["r1", "r2", "t1", "t2", "r1", "t1"],
            "receiver": ["B"] * 6,
            "family_id": ["f1"] * 4 + ["f2"] * 2,
            "receiver_program_score": [1.0, 1.0, 3.0, 5.0, 1.0, 2.0],
            "status": ["observed"] * 6,
            "formal_inference_allowed": [False] * 6,
        }
    )
    conditions = pd.DataFrame(
        {
            "sample_id": ["r1", "r2", "t1", "t2"],
            "group": ["R", "R", "T", "T"],
        }
    )
    result = derive_receiver_program_effects(
        program,
        conditions,
        condition_column="group",
        target="T",
        reference="R",
    ).set_index("family_id")
    assert result.loc["f1", "program_z"] == 3.0
    assert result.loc["f1", "status"] == "observed"
    assert result.loc["f2", "status"] == "not_estimable"
    assert not bool(result["formal_inference_allowed"].any())


def test_unit_weights_reproduce_rc2_cardinality_and_preserve_absence() -> None:
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
    rebuilt, calls = _weighted_strict_rankings(
        edges,
        template,
        np.ones(len(edges)),
        target="T",
        reference="R",
        method="new",
        semantics="test",
    )
    assert calls == 3
    assert _assert_alpha_zero_parity(rebuilt, template)["rankings_exact"] is True
    absent = rebuilt.loc[rebuilt["receiver"].eq("C")].iloc[0]
    assert pd.isna(absent["ranked_strength"])
