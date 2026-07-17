from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.metrics.repeated_measures_cr2 import repeated_measures_cr2_effects

from crychic.design import RepeatedMeasuresDesignSpec, balanced_contrast

EDGE_KEYS = ("sender", "receiver", "interaction_id", "ligand", "receptor")


def _sample_design() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"p{subject}-{condition}",
                "subject_id": f"p{subject}",
                "condition": condition,
            }
            for subject in range(8)
            for condition in ("control", "case")
        ]
    )


def _score_table() -> pd.DataFrame:
    deviations = (-0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07)
    rows: list[dict[str, object]] = []
    for subject in range(8):
        for condition in ("control", "case"):
            sample_id = f"p{subject}-{condition}"
            baseline = subject * 0.04
            rows.extend(
                [
                    {
                        "method": "m1",
                        "contrast": "case_vs_control",
                        "sample_id": sample_id,
                        "sender": "S",
                        "receiver": "R",
                        "interaction_id": "I1",
                        "ligand": "L1",
                        "receptor": "R1",
                        "value": baseline
                        + (0.4 + deviations[subject] if condition == "case" else 0.0),
                        "status": "observed",
                    },
                    {
                        "method": "m1",
                        "contrast": "case_vs_control",
                        "sample_id": sample_id,
                        "sender": "S",
                        "receiver": "R",
                        "interaction_id": "I2",
                        "ligand": "L2",
                        "receptor": "R2",
                        "value": baseline if condition == "control" else np.nan,
                        "status": "observed" if condition == "control" else "missing",
                    },
                ]
            )
    return pd.DataFrame(rows)


def _fit(table: pd.DataFrame) -> pd.DataFrame:
    return repeated_measures_cr2_effects(
        table,
        _sample_design(),
        design_spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition",),
            min_subjects_per_context=3,
            min_subject_clusters=6,
        ),
        contrast=balanced_contrast(
            ("case",),
            ("control",),
            name="case_vs_control",
        ),
        identity_keys=("method", "contrast"),
        edge_keys=EDGE_KEYS,
        value_key="value",
        status_key="status",
        observed_statuses=("observed",),
    )


def test_grouped_metric_batches_edges_and_preserves_cr2_diagnostics() -> None:
    result = _fit(_score_table())

    eligible = result.loc[result["interaction_id"].eq("I1")].iloc[0]
    blocked = result.loc[result["interaction_id"].eq("I2")].iloc[0]
    assert eligible["status"] == "eligible"
    assert eligible["effect"] == pytest.approx(0.4)
    assert eligible["standard_error"] > 0.0
    assert eligible["n_subject_clusters"] == 8
    assert eligible["n_effective_clusters"] == 8
    assert eligible["condition_number"] >= 1.0
    assert 0.0 <= eligible["maximum_cluster_leverage"] < 1.0
    assert eligible["minimum_cr2_adjustment_eigenvalue"] > 0.0
    assert bool(eligible["formal_backend_eligible"])
    assert not bool(eligible["formal_inference_allowed"])
    assert blocked["status"] == "not_estimable"
    assert blocked["reason_code"] == "insufficient_subjects_per_context"
    assert pd.isna(blocked["effect"])
    assert pd.isna(blocked["standard_error"])
    assert result["repeated_measures_cr2_artifact_id"].nunique() == 1
    assert result["repeated_measures_cr2_design_id"].nunique() == 1
    assert not {"p", "p_value", "q", "q_value"}.intersection(result.columns)


def test_grouped_metric_is_row_order_invariant() -> None:
    table = _score_table()
    first = _fit(table).sort_values("interaction_id", ignore_index=True)
    second = _fit(table.sample(frac=1.0, random_state=17)).sort_values(
        "interaction_id", ignore_index=True
    )

    columns = (
        "interaction_id",
        "effect",
        "standard_error",
        "status",
        "reason_code",
        "repeated_measures_cr2_feature_id",
        "repeated_measures_cr2_artifact_id",
    )
    pd.testing.assert_frame_equal(first.loc[:, columns], second.loc[:, columns])


def test_grouped_metric_rejects_repeated_edge_sample_rows() -> None:
    table = _score_table()
    duplicated = pd.concat([table, table.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="repeated edge/sample"):
        _fit(duplicated)
