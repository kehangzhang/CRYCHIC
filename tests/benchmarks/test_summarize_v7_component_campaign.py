from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.simulation.summarize_v7_component_campaign import (
    _annotation_equality,
    _descriptive_ci,
    _e2_paired_deltas,
)


def test_descriptive_ci_requires_replication_and_contains_the_mean() -> None:
    low, high = _descriptive_ci(pd.Series([0.2, 0.4, 0.6, np.nan]))

    assert low < 0.4 < high
    assert all(np.isnan(value) for value in _descriptive_ci(pd.Series([0.4])))


def _e2_row(
    score_view: str,
    *,
    value: float,
    status: str = "observed",
    reason_code: str = "ok",
    dataset_id: str = "dataset-1",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "dgp_family": "sender_decoy",
        "design_kind": "binary",
        "contrast_name": "B_vs_A",
        "metric": "event_auprc",
        "score_view": score_view,
        "value": value,
        "status": status,
        "reason_code": reason_code,
    }


def test_annotation_equality_checks_values_statuses_and_reason_codes() -> None:
    rows = [_e2_row("base_g3_i1", value=0.5)]
    for arm in (
        "receptor_eligibility_annotation",
        "ligand_contrast_annotation",
        "family_selection_annotation",
        "downstream_support_annotation",
    ):
        rows.append(_e2_row(arm, value=0.5))
    metrics = pd.DataFrame(rows)

    equal = _annotation_equality(metrics)

    expected = {"equal_rows": 1, "rows": 1, "exact": True}
    assert all(result == expected for result in equal.values())
    metrics.loc[
        metrics["score_view"].eq("downstream_support_annotation"), "reason_code"
    ] = "changed"
    assert not _annotation_equality(metrics)["downstream_support_annotation"]["exact"]


def test_e2_paired_deltas_exclude_nonobserved_arm_rows() -> None:
    rows = [
        _e2_row("base_g3_i1", value=0.4, dataset_id="dataset-1"),
        _e2_row("base_g3_i1", value=0.6, dataset_id="dataset-2"),
    ]
    hard_arms = (
        "receptor_eligibility_gate",
        "ligand_contrast_gate",
        "family_selection_gate",
        "downstream_support_gate",
    )
    for arm in hard_arms:
        rows.extend(
            [
                _e2_row(arm, value=0.5, dataset_id="dataset-1"),
                _e2_row(
                    arm,
                    value=np.nan,
                    status="not_estimable",
                    reason_code="gate_failed",
                    dataset_id="dataset-2",
                ),
            ]
        )

    result = _e2_paired_deltas(pd.DataFrame(rows))

    assert set(result["score_view"]) == set(hard_arms)
    assert result["all_pairs"].eq(2).all()
    assert result["observed_pairs"].eq(1).all()
    assert np.allclose(result["delta_mean"], 0.1)
    assert result["descriptive_ci_low"].isna().all()
    assert result["descriptive_ci_high"].isna().all()
