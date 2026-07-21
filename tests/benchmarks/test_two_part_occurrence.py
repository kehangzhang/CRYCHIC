from __future__ import annotations

import numpy as np

from benchmarks.literature.two_part_occurrence import (
    fit_hurdle_channel_reliability,
    fit_subject_hurdle_effects,
    hurdle_directional_weights,
)


def test_hurdle_effects_separate_occurrence_and_positive_magnitude() -> None:
    conditions = np.array(["R"] * 4 + ["T"] * 4)
    presence = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    magnitude = np.array(
        [
            [np.nan, np.nan, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        ]
    )
    result = fit_subject_hurdle_effects(
        presence,
        magnitude,
        conditions,
        reference="R",
        target="T",
        minimum_active_subjects=3,
    )
    assert result.loc[0, "occurrence_effect"] > 0.0
    assert result.loc[0, "magnitude_status"] == "not_estimable"
    assert result.loc[1, "occurrence_effect"] == 0.0
    assert result.loc[1, "magnitude_effect"] > 0.0


def test_structural_missingness_stays_not_estimable() -> None:
    conditions = np.array(["R"] * 3 + ["T"] * 3)
    presence = np.array([[1.0, 1.0, 1.0, np.nan, np.nan, 1.0]])
    magnitude = np.where(np.isfinite(presence), 2.0, np.nan)
    result = fit_subject_hurdle_effects(
        presence,
        magnitude,
        conditions,
        reference="R",
        target="T",
        minimum_subjects=3,
    )
    assert result.loc[0, "occurrence_status"] == "not_estimable"
    assert np.isnan(result.loc[0, "occurrence_effect"])


def test_zero_alpha_and_missing_channels_are_exactly_neutral() -> None:
    baseline = np.r_[np.ones(300), -np.ones(300)]
    occurrence_effect = baseline.copy()
    magnitude_effect = np.full(600, np.nan)
    occurrence_fit = fit_hurdle_channel_reliability(
        baseline,
        occurrence_effect,
        channel="occurrence",
        maximum_alpha=0.0,
        alpha_cap=1.0,
        minimum_edges=200,
    )
    magnitude_fit = fit_hurdle_channel_reliability(
        baseline,
        magnitude_effect,
        channel="magnitude",
        maximum_alpha=1.0,
        alpha_cap=1.0,
        minimum_edges=200,
    )
    weights, occurrence_evidence, magnitude_evidence = hurdle_directional_weights(
        np.sign(baseline),
        occurrence_effect,
        magnitude_effect,
        occurrence_fit,
        magnitude_fit,
        log_weight_cap=1.5,
    )
    np.testing.assert_array_equal(weights, np.ones(600))
    assert (occurrence_evidence > 0.0).all()
    np.testing.assert_array_equal(magnitude_evidence, np.zeros(600))


def test_reliability_closes_an_antagonistic_channel() -> None:
    baseline = np.r_[np.ones(300), -np.ones(300)]
    fit = fit_hurdle_channel_reliability(
        baseline,
        -baseline,
        channel="occurrence",
        maximum_alpha=4.0,
        alpha_cap=1.0,
        minimum_edges=200,
    )
    assert fit.reliability == 0.0
    assert fit.effective_alpha == 0.0
