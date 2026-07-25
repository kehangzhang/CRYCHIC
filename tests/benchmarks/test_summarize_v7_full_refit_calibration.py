from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.simulation.summarize_v7_full_refit_calibration import (
    DATASET_METRIC_COLUMNS,
    _dataset_channel_metrics,
    _expected_resamples_by_design,
    _scenario_metrics,
    _wilson_interval,
)


def _effect_table(*, discovery: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_status": ["observed", "observed", "not_estimable"],
            "n_bootstrap_total": [19, 19, 19],
            "n_bootstrap_observed": [19, 19, 0],
            "n_permutation_total": [19, 19, 19],
            "n_permutation_observed": [19, 19, 0],
            "n_loso_total": [12, 12, 12],
            "n_loso_observed": [12, 12, 0],
            "diagnostic_ci_lower": [-1.0, 0.1, None],
            "diagnostic_ci_upper": [1.0, 1.0, None],
            "diagnostic_empirical_p_value": [0.05 if discovery else 0.2, 0.2, None],
            "diagnostic_q_value": [0.1 if discovery else 0.4, 0.4, None],
            "formal_inference_allowed": [False, False, False],
        }
    )


def _record(*, dataset_id: str, discovery: bool) -> dict[str, object]:
    return _dataset_channel_metrics(
        _effect_table(discovery=discovery),
        dataset_id=dataset_id,
        dgp_family="global_null",
        design_kind="paired",
        replicate_index=int(dataset_id.removeprefix("dataset-")),
        channel="continuous_raw",
        expected_bootstraps=19,
        expected_permutations=19,
        expected_loso=12,
    )


def test_dataset_and_scenario_global_null_metrics_are_hand_computable() -> None:
    first = _record(dataset_id="dataset-1", discovery=True)
    second = _record(dataset_id="dataset-2", discovery=False)

    assert first["n_point_observed"] == 2
    assert first["n_distribution_complete"] == 2
    assert first["type_i_rate_alpha_0_05"] == 0.5
    assert first["false_discovery_proportion_q_0_10"] == 1.0
    assert first["ci_coverage_zero"] == 0.5
    assert first["formal_rows"] == 0

    dataset = pd.DataFrame.from_records(
        [first, second],
        columns=DATASET_METRIC_COLUMNS,
    )
    scenario = _scenario_metrics(dataset).iloc[0]
    assert scenario["n_replicates"] == 2
    assert scenario["n_hypotheses_with_p"] == 4
    assert scenario["empirical_type_i"] == 0.25
    assert scenario["empirical_fdr"] == 0.5
    assert scenario["ci_coverage"] == 0.5
    assert scenario["formal_rows"] == 0


def test_wilson_interval_validates_counts_and_contains_observed_rate() -> None:
    lower, upper = _wilson_interval(5, 100)
    assert lower < 0.05 < upper
    with pytest.raises(ValueError, match="Wilson counts"):
        _wilson_interval(2, 1)


def test_expected_resamples_are_design_specific_and_strict() -> None:
    experiment: dict[str, object] = {
        "design_kinds": ["paired", "continuous"],
        "expected_resamples_per_dataset_by_design": {
            "paired": 50,
            "continuous": 62,
        },
    }

    assert _expected_resamples_by_design(experiment) == {
        "paired": 50,
        "continuous": 62,
    }
    experiment["expected_resamples_per_dataset_by_design"] = 50
    with pytest.raises(ValueError, match="per-design"):
        _expected_resamples_by_design(experiment)
