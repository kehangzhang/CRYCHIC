from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.comprehensive.compare_rc12_external_panel import (
    CANDIDATE_METHOD,
    _paired_comparison,
    _summarize,
)


def _metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, offset in ((CANDIDATE_METHOD, 0.2), ("scseqcommdiff", 0.1)):
        for index in range(20):
            common = {
                "seed": index,
                "dataset_id": f"active_{index}",
                "method": method,
                "omnibus_prevalence_adjusted_ap": offset + 0.01,
                "omnibus_auprc": offset,
                "omnibus_auroc": offset + 0.4,
                "localization_macro_auprc": offset - 0.02,
                "positive_direction_ap": 0.1,
                "negative_direction_ap": 0.1,
                "direction_accuracy_all_active": 0.5,
                "event_coverage": 1.0,
                "effect_dynamic_range": np.nan,
                "effect_all_zero_fraction": 0.0,
            }
            rows.append(
                {**common, "scenario": "active", "effect_standard_deviation": 0.2}
            )
            rows.append(
                {
                    **common,
                    "scenario": "global_null",
                    "dataset_id": f"null_{index}",
                    "effect_standard_deviation": 0.01
                    if method == CANDIDATE_METHOD
                    else 0.02,
                    "effect_dynamic_range": 0.03,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_summary_ranks_detection_and_null_metrics_in_expected_directions() -> None:
    summary = _summarize(_metrics()).set_index("method")

    assert summary.loc[CANDIDATE_METHOD, "omnibus_auprc_rank"] == 1
    assert summary.loc[CANDIDATE_METHOD, "global_null_effect_sd_rank"] == 1
    assert summary.loc["scseqcommdiff", "omnibus_auprc_rank"] == 2


def test_paired_comparison_is_seed_paired_and_deterministic() -> None:
    metrics = _metrics()
    first, detail = _paired_comparison(
        metrics,
        comparator="scseqcommdiff",
        scenario="active",
        metric="omnibus_auprc",
        direction="higher",
        replicates=100,
        seed=7,
    )
    second, _ = _paired_comparison(
        metrics,
        comparator="scseqcommdiff",
        scenario="active",
        metric="omnibus_auprc",
        direction="higher",
        replicates=100,
        seed=7,
    )

    assert first == second
    assert first["raw_candidate_minus_comparator"] == pytest.approx(0.1)
    assert first["wins"] == 20
    assert len(detail) == 20


def test_lower_is_better_reorients_null_improvement() -> None:
    record, _ = _paired_comparison(
        _metrics(),
        comparator="scseqcommdiff",
        scenario="global_null",
        metric="effect_standard_deviation",
        direction="lower",
        replicates=100,
        seed=8,
    )

    assert record["raw_candidate_minus_comparator"] == pytest.approx(-0.01)
    assert record["oriented_mean_improvement"] == pytest.approx(0.01)
    assert record["wins"] == 20
