from __future__ import annotations

import math

import numpy as np
import pandas as pd
from benchmarks.comprehensive.run_score_engine_crossover_under100k import (
    EDGE_KEYS,
    _bh_adjust,
    _dataset_metrics,
    _eligibility_matrix,
    _fit_engine_arm,
    _fit_pairwise_event,
)


def test_bh_adjust_preserves_missing_and_monotonicity() -> None:
    adjusted = _bh_adjust([0.01, 0.02, 0.04, math.nan])

    np.testing.assert_allclose(adjusted[:3], [0.03, 0.03, 0.04])
    assert math.isnan(adjusted[3])


def test_pairwise_glm_and_singleton_cr2_recover_effect_direction() -> None:
    values = [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
    indicator = [0, 0, 0, 0, 1, 1, 1, 1]

    ordinary = _fit_pairwise_event(values, indicator, robust=False)
    robust = _fit_pairwise_event(values, indicator, robust=True)

    assert ordinary[0] == robust[0] == 2.0
    assert ordinary[1] > 0.0
    assert robust[1] > 0.0
    assert 0.0 <= ordinary[2] < 0.05
    assert 0.0 <= robust[2] < 0.05


def _score_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_index, interaction_id in enumerate(("LR1", "LR2")):
        for condition, offset in (("A", 0.0), ("B", 2.0)):
            for subject_index in range(4):
                status = "observed"
                score = offset + subject_index
                if interaction_id == "LR2" and condition == "B" and subject_index == 3:
                    status = "not_estimable"
                    score = math.nan
                rows.append(
                    {
                        "dataset_id": "fixture",
                        "seed": 1,
                        "score_generator": "crychic_native",
                        "contrast": "B_vs_A",
                        "target": "B",
                        "reference": "A",
                        "sample_id": f"{condition}:{subject_index}",
                        "subject_id": f"{condition}:{subject_index}",
                        "condition": condition,
                        "sender": "S",
                        "receiver": "R",
                        "interaction_id": interaction_id,
                        "score": score,
                        "status": status,
                        "reason_code": "fixture",
                        "event_index": event_index,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_common_engine_keeps_structural_missingness_not_estimable() -> None:
    fitted, _ = _fit_engine_arm(_score_rows(), engine="sample_glm")
    by_event = fitted.set_index("interaction_id")

    assert by_event.loc["LR1", "status"] == "observed"
    assert by_event.loc["LR1", "effect"] == 2.0
    assert by_event.loc["LR2", "status"] == "not_estimable"
    assert math.isnan(float(by_event.loc["LR2", "effect"]))
    assert (
        by_event.loc["LR2", "reason_code"]
        == "fewer_than_four_subjects_per_condition"
    )


def test_eligibility_matrix_marks_noninjectable_engines_ne() -> None:
    effect = {
        "score_generator": "crychic_native",
        "differential_engine": "sample_glm",
        "status": "observed",
    }
    matrix = _eligibility_matrix(pd.DataFrame([effect]))
    key = matrix.set_index(["score_generator", "differential_engine"])

    assert key.loc[("crychic_native", "sample_glm"), "status"] == "observed"
    assert (
        key.loc[
            ("crychic_native", "crychic_common_functional_oof"), "status"
        ]
        == "not_estimable"
    )
    assert (
        key.loc[
            ("crychic_native", "scseqcommdiff_multi_sample"), "reason_code"
        ]
        == "public_engine_has_no_external_score_injection_interface"
    )
    assert tuple(EDGE_KEYS) == ("sender", "receiver", "interaction_id")


def test_dataset_metrics_do_not_turn_missing_fdr_into_zero() -> None:
    effects = pd.DataFrame(
        [
            {
                "dataset_id": "fixture",
                "scenario": "active",
                "seed": 1,
                "score_generator": "cellchat_probability",
                "differential_engine": "staccato",
                "status": "not_estimable",
                "effect": math.nan,
                "p_value": math.nan,
                "q_value": math.nan,
                "truth_effect": 1.0,
                "truth_label": 1,
                "truth_direction": 1,
                "engine_elapsed_seconds": 0.1,
            }
        ]
    )

    metrics = _dataset_metrics(effects).iloc[0]
    assert metrics.coverage == 0.0
    assert math.isnan(float(metrics.empirical_fdr_q05))
