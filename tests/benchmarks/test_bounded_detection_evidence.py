from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.comprehensive.evaluate_bounded_detection_evidence import (
    BASE_ARM,
    CANDIDATE_ARM,
    CANDIDATE_LAYER,
    UPPER_DIAGNOSTIC_LAYER,
    _paired_metric,
    _score_layers,
)


def test_soft_guard_score_is_detection_only_and_keeps_dynamic_range() -> None:
    table = pd.DataFrame(
        {
            "availability": [0.0, 1.0],
            "prior_quality": [1.0, 1.0],
            "sender_component": [0.4, 0.4],
            "downstream": [0.8, 0.8],
        }
    )
    layers = _score_layers(table, mechanism_floor=0.75, downstream_weight=0.05)

    assert layers["canonical_mechanistic"].tolist() == pytest.approx([0.0, 0.4])
    assert layers[CANDIDATE_LAYER].tolist() == pytest.approx([0.315, 0.42])
    assert layers[UPPER_DIAGNOSTIC_LAYER].tolist() == pytest.approx([0.44, 0.44])


def test_soft_guard_requires_complete_validation_components() -> None:
    table = pd.DataFrame(
        {
            "availability": [1.0],
            "prior_quality": [1.0],
            "sender_component": [0.4],
            "downstream": [None],
        }
    )
    with pytest.raises(ValueError, match="complete components"):
        _score_layers(table, mechanism_floor=0.75, downstream_weight=0.05)


def test_paired_metric_orients_lower_is_better() -> None:
    rows: list[dict[str, object]] = []
    for seed, baseline, candidate in ((1, 0.4, 0.2), (2, 0.3, 0.2)):
        rows.extend(
            (
                {
                    "scenario": "global_null",
                    "seed": seed,
                    "method": BASE_ARM,
                    "effect_standard_deviation": baseline,
                },
                {
                    "scenario": "global_null",
                    "seed": seed,
                    "method": CANDIDATE_ARM,
                    "effect_standard_deviation": candidate,
                },
            )
        )
    record, detail = _paired_metric(
        pd.DataFrame.from_records(rows),
        scenario="global_null",
        metric="effect_standard_deviation",
        direction="lower",
        replicates=100,
        seed=17,
    )

    assert record["paired_seeds"] == 2
    assert record["raw_candidate_minus_baseline"] == pytest.approx(-0.15)
    assert record["oriented_mean_improvement"] == pytest.approx(0.15)
    assert detail["oriented_improvement"].tolist() == pytest.approx([0.2, 0.1])


def test_paired_metric_accepts_a_frozen_alternative_candidate() -> None:
    alternative = f"{UPPER_DIAGNOSTIC_LAYER}__native_raw_mean"
    metrics = pd.DataFrame(
        [
            {
                "scenario": "active",
                "seed": 1,
                "method": BASE_ARM,
                "omnibus_auprc": 0.1,
            },
            {
                "scenario": "active",
                "seed": 1,
                "method": alternative,
                "omnibus_auprc": 0.2,
            },
        ]
    )

    record, _ = _paired_metric(
        metrics,
        scenario="active",
        metric="omnibus_auprc",
        direction="higher",
        replicates=100,
        seed=18,
        candidate_arm=alternative,
    )

    assert record["oriented_mean_improvement"] == pytest.approx(0.1)
