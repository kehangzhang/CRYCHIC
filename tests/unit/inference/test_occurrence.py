from __future__ import annotations

import math

import pandas as pd
import pytest

from crychic.inference import (
    OCCURRENCE_CONTRAST_VERSION,
    OccurrenceContrastSpec,
    OccurrenceContrastStatus,
    fit_subject_occurrence_contrasts,
)


def _event(
    reference: list[float | None], target: list[float | None], *, event_id: str = "e1"
) -> pd.DataFrame:
    rows = []
    for condition, prefix, values in (
        ("ctrl", "C", reference),
        ("stim", "T", target),
    ):
        rows.extend(
            {
                "event_id": event_id,
                "subject_id": f"{prefix}{index:02d}",
                "condition": condition,
                "occurrence": value,
            }
            for index, value in enumerate(values)
        )
    return pd.DataFrame.from_records(rows)


def test_occurrence_contrast_uses_jeffreys_prevalence_and_no_inference() -> None:
    table = _event([0, 0, 1, 0], [1, 1, 1, 0])
    result = fit_subject_occurrence_contrasts(
        table,
        reference="ctrl",
        target="stim",
        spec=OccurrenceContrastSpec(minimum_subjects_per_group=4),
    ).iloc[0]

    assert result["status"] == OccurrenceContrastStatus.OBSERVED.value
    assert result["reference_posterior_prevalence"] == pytest.approx(1.5 / 5.0)
    assert result["target_posterior_prevalence"] == pytest.approx(3.5 / 5.0)
    assert result["prevalence_difference"] == pytest.approx(0.4)
    expected_variance = (1.5 * 3.5) / (5.0**2 * 6.0)
    assert result["posterior_standardized_prevalence_difference"] == pytest.approx(
        0.4 / math.sqrt(2.0 * expected_variance)
    )
    expected_log_or = math.log(3.5 / 1.5) - math.log(1.5 / 3.5)
    assert result["posterior_log_odds_ratio"] == pytest.approx(expected_log_or)
    assert result["direction"] == 1
    assert not bool(result["formal_inference_allowed"])
    assert result["score_version"] == OCCURRENCE_CONTRAST_VERSION
    assert not any(
        column in result.index
        for column in ("p_value", "q_value", "active_probability")
    )


def test_missing_occurrence_reduces_coverage_instead_of_becoming_absence() -> None:
    table = _event([0, 1, None, 1], [1, None, 1, 1])
    result = fit_subject_occurrence_contrasts(
        table,
        reference="ctrl",
        target="stim",
        spec=OccurrenceContrastSpec(minimum_subjects_per_group=3),
    ).iloc[0]

    assert result["reference_subjects"] == 3
    assert result["target_subjects"] == 3
    assert result["reference_occurrences"] == 2
    assert result["target_occurrences"] == 3


def test_occurrence_contrast_is_row_order_deterministic() -> None:
    first_event = _event([0, 0, 1, 0], [1, 1, 1, 0], event_id="e1")
    second_event = _event([1, 1, 0, 1], [0, 0, 0, 1], event_id="e2")
    table = pd.concat([first_event, second_event], ignore_index=True)
    spec = OccurrenceContrastSpec(minimum_subjects_per_group=4)
    first = fit_subject_occurrence_contrasts(
        table, reference="ctrl", target="stim", spec=spec
    )
    second = fit_subject_occurrence_contrasts(
        table.sample(frac=1.0, random_state=9),
        reference="ctrl",
        target="stim",
        spec=spec,
    )

    pd.testing.assert_frame_equal(first, second)


def test_insufficient_subjects_are_typed_not_estimable() -> None:
    result = fit_subject_occurrence_contrasts(
        _event([0, None, 1, None], [1, 1, 0, 1]),
        reference="ctrl",
        target="stim",
        spec=OccurrenceContrastSpec(minimum_subjects_per_group=3),
    ).iloc[0]

    assert result["status"] == OccurrenceContrastStatus.NOT_ESTIMABLE.value
    assert result["reason_code"] == "insufficient_observed_subjects_per_group"
    assert pd.isna(result["prevalence_difference"])


def test_occurrence_contract_rejects_duplicates_overlap_and_nonbinary_values() -> None:
    table = _event([0, 0, 1, 0], [1, 1, 1, 0])
    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per event"):
        fit_subject_occurrence_contrasts(duplicate, reference="ctrl", target="stim")

    overlap = table.copy()
    overlap.loc[overlap["condition"].eq("stim"), "subject_id"] = [
        f"C{index:02d}" for index in range(4)
    ]
    with pytest.raises(ValueError, match="cannot reuse subject IDs"):
        fit_subject_occurrence_contrasts(overlap, reference="ctrl", target="stim")

    nonbinary = table.astype({"occurrence": float}).copy()
    nonbinary.loc[0, "occurrence"] = 0.5
    with pytest.raises(ValueError, match="zero or one"):
        fit_subject_occurrence_contrasts(nonbinary, reference="ctrl", target="stim")
