from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.comprehensive.evaluate_brca_semisynthetic import (
    EVENT_KEYS,
    _metrics,
    _paired_effects,
)


def _event(index: int) -> dict[str, str]:
    return {
        "sender": "S",
        "receiver": "R",
        "interaction_id": f"I{index}",
        "ligand": f"L{index}",
        "receptor": f"R{index}",
    }


def test_paired_effect_is_difference_of_within_subject_changes() -> None:
    records = []
    for expansion, delta in (("E", 2.0), ("NE", 0.5)):
        for subject in range(4):
            records.append(
                {
                    "subject_id": f"{expansion}{subject}",
                    "expansion": expansion,
                    **_event(0),
                    "paired_delta": delta + 0.01 * subject,
                }
            )
    result = _paired_effects(
        pd.DataFrame.from_records(records),
        min_subjects_per_group=4,
    )

    assert len(result) == 1
    assert result.loc[0, "status"] == "observed"
    assert np.isclose(result.loc[0, "difference_in_differences"], 1.5)
    assert result.loc[0, "n_subjects_E"] == 4
    assert result.loc[0, "n_subjects_NE"] == 4


def test_metrics_separate_signed_truth_and_main_effect_leakage() -> None:
    effects = pd.DataFrame.from_records(
        [
            {
                **_event(0),
                "event_class": "positive_did",
                "truth_label": 1,
                "truth_direction": 1,
                "planted_main_effect_control": 0,
                "difference_in_differences": 2.0,
                "q_value": 0.01,
                "status": "observed",
            },
            {
                **_event(1),
                "event_class": "negative_did",
                "truth_label": 1,
                "truth_direction": -1,
                "planted_main_effect_control": 0,
                "difference_in_differences": -2.0,
                "q_value": 0.01,
                "status": "observed",
            },
            {
                **_event(2),
                "event_class": "time_main_only",
                "truth_label": 0,
                "truth_direction": 0,
                "planted_main_effect_control": 1,
                "difference_in_differences": 0.1,
                "q_value": 0.8,
                "status": "observed",
            },
            {
                **_event(3),
                "event_class": "no_effect",
                "truth_label": 0,
                "truth_direction": 0,
                "planted_main_effect_control": 0,
                "difference_in_differences": 0.0,
                "q_value": 0.9,
                "status": "observed",
            },
        ],
        columns=[
            *EVENT_KEYS,
            "event_class",
            "truth_label",
            "truth_direction",
            "planted_main_effect_control",
            "difference_in_differences",
            "q_value",
            "status",
        ],
    )

    result = _metrics(effects)

    assert result["omnibus_auprc"] == 1.0
    assert result["omnibus_auroc"] == 1.0
    assert result["direction_accuracy_active"] == 1.0
    assert np.isclose(result["main_to_active_abs_ratio"], 0.05)
    assert result["power"] == 1.0
    assert result["empirical_fdr"] == 0.0
