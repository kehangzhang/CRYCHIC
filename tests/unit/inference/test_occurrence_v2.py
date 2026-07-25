from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.support.sample_edge_v2 import sample_edge_scores

from crychic.inference import (
    TWO_PART_OCCURRENCE_VERSION,
    DifferentialContrastSpec,
    DifferentialDesignKind,
    DifferentialDesignSpec,
    TwoPartOccurrenceV2Spec,
    build_sample_edge_two_part_input,
    fit_two_part_occurrence_v2,
)


def _contrast() -> DifferentialContrastSpec:
    return DifferentialContrastSpec(
        name="treated-vs-control",
        weights=(("control", -1.0), ("treated", 1.0)),
    )


def _design(
    kind: DifferentialDesignKind | str,
    *,
    continuous_covariates: tuple[str, ...] = (),
) -> DifferentialDesignSpec:
    return DifferentialDesignSpec(
        design_kind=kind,
        condition_column="condition",
        condition_levels=("control", "treated"),
        continuous_covariates=continuous_covariates,
        contrasts=(_contrast(),),
        precision_weight_column=None,
        minimum_subjects_per_level=4,
        minimum_clusters_for_cr2=6,
    )


def _row(
    *,
    event_id: str,
    sample_id: str,
    subject_id: str,
    condition: str,
    active: bool,
    age: float | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "event_id": event_id,
        "sample_id": sample_id,
        "subject_id": subject_id,
        "condition": condition,
        "score": 2.0 + (0.1 if condition == "treated" else 0.0) if active else 0.2,
        "score_status": "observed",
        "out_of_fold": True,
    }
    if age is not None:
        result["age"] = age
    return result


def test_independent_two_group_uses_fisher_and_formal_bh() -> None:
    rows = [
        _row(
            event_id="e1",
            sample_id=f"control-{index}",
            subject_id=f"control-{index}",
            condition="control",
            active=index < 2,
        )
        for index in range(16)
    ] + [
        _row(
            event_id="e1",
            sample_id=f"treated-{index}",
            subject_id=f"treated-{index}",
            condition="treated",
            active=index < 12,
        )
        for index in range(16)
    ]
    spec = TwoPartOccurrenceV2Spec(
        design=_design("independent_two_group"),
        activity_threshold_raw=1.0,
    )

    result = fit_two_part_occurrence_v2(pd.DataFrame(rows), spec)
    effect = result.effects.iloc[0]

    assert effect["status"] == "observed"
    assert effect["occurrence_method"] == "fisher_exact_two_sided"
    assert effect["reference_occurrences"] == 2
    assert effect["target_occurrences"] == 12
    assert effect["prevalence_difference"] == pytest.approx(10.0 / 16.0)
    assert effect["p_value"] < 0.001
    assert effect["q_value"] == pytest.approx(effect["p_value"])
    assert bool(effect["formal_inference_allowed"])
    assert effect["formal_inference_scope"] == "fixed_threshold_oof_occurrence_only"
    assert result.subject_events["active_probability"].between(0.0, 1.0).all()
    assert result.spec.to_dict()["version"] == TWO_PART_OCCURRENCE_VERSION


def test_paired_design_uses_exact_mcnemar_and_subject_offsets_do_not_matter() -> None:
    control = np.asarray([0] * 8 + [1] * 4, dtype=bool)
    treated = np.asarray([1] * 8 + [0] + [1] * 3, dtype=bool)
    rows = [
        _row(
            event_id="e1",
            sample_id=f"S{index}-{condition}",
            subject_id=f"S{index}",
            condition=condition,
            active=bool(values[index]),
        )
        for index in range(12)
        for condition, values in (("control", control), ("treated", treated))
    ]
    spec = TwoPartOccurrenceV2Spec(
        design=_design("paired"),
        activity_threshold_raw=1.0,
    )

    effect = fit_two_part_occurrence_v2(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["status"] == "observed"
    assert effect["occurrence_method"] == "mcnemar_exact_binomial"
    assert effect["complete_pairs"] == 12
    assert effect["target_occurrences"] - effect["reference_occurrences"] == 7
    assert effect["p_value"] == pytest.approx(20.0 / 512.0)


def test_paired_no_discordance_is_an_estimable_null_result() -> None:
    rows = [
        _row(
            event_id="e1",
            sample_id=f"S{index}-{condition}",
            subject_id=f"S{index}",
            condition=condition,
            active=index % 2 == 0,
        )
        for index in range(8)
        for condition in ("control", "treated")
    ]
    spec = TwoPartOccurrenceV2Spec(
        design=_design("paired"),
        activity_threshold_raw=1.0,
    )

    effect = fit_two_part_occurrence_v2(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["status"] == "observed"
    assert effect["p_value"] == 1.0
    assert effect["odds_ratio"] == 1.0


def test_repeated_covariate_design_uses_subject_cluster_logistic() -> None:
    rows = []
    for index in range(20):
        age = 35.0 + index
        control_active = index % 5 == 0 or index in {7, 13}
        treated_active = index % 2 == 0 or index in {3, 9, 15}
        for condition, active in (
            ("control", control_active),
            ("treated", treated_active),
        ):
            rows.append(
                _row(
                    event_id="e1",
                    sample_id=f"R{index}-{condition}",
                    subject_id=f"R{index}",
                    condition=condition,
                    active=active,
                    age=age,
                )
            )
    spec = TwoPartOccurrenceV2Spec(
        design=_design("repeated", continuous_covariates=("age",)),
        activity_threshold_raw=1.0,
    )

    effect = fit_two_part_occurrence_v2(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["status"] == "observed"
    assert effect["occurrence_method"] == "logistic_subject_cluster_sandwich"
    assert effect["complete_pairs"] == 20
    assert pd.notna(effect["standard_error"])


def test_independent_multigroup_emits_pairwise_fisher_tests_with_global_bh() -> None:
    active_counts = {"A": 1, "B": 4, "C": 7}
    rows = [
        {
            **_row(
                event_id="e1",
                sample_id=f"{level}-{index}",
                subject_id=f"{level}-{index}",
                condition=level,
                active=index < active_count,
            )
        }
        for level, active_count in active_counts.items()
        for index in range(8)
    ]
    design = DifferentialDesignSpec(
        design_kind="independent_multi_group",
        condition_column="condition",
        condition_levels=("A", "B", "C"),
        contrasts=(
            DifferentialContrastSpec(name="B-vs-A", weights=(("A", -1.0), ("B", 1.0))),
            DifferentialContrastSpec(name="C-vs-A", weights=(("A", -1.0), ("C", 1.0))),
        ),
        precision_weight_column=None,
        minimum_subjects_per_level=4,
    )
    spec = TwoPartOccurrenceV2Spec(design=design, activity_threshold_raw=1.0)

    effects = fit_two_part_occurrence_v2(pd.DataFrame(rows), spec).effects

    assert len(effects) == 2
    assert set(effects["occurrence_method"]) == {"fisher_exact_two_sided"}
    assert effects["q_value"].ge(effects["p_value"]).all()
    assert effects.set_index("contrast_name").loc[
        "C-vs-A", "prevalence_difference"
    ] == pytest.approx(0.75)


def test_multicohort_adjustment_uses_logistic_hc1_sandwich() -> None:
    rows = []
    for cohort in ("site-a", "site-b"):
        for condition in ("control", "treated"):
            for index in range(12):
                active = (
                    index % 4 == 0
                    if condition == "control"
                    else index % 2 == 0 or index in {1, 3}
                )
                row = _row(
                    event_id="e1",
                    sample_id=f"{cohort}-{condition}-{index}",
                    subject_id=f"{cohort}-{condition}-{index}",
                    condition=condition,
                    active=active,
                )
                row["cohort"] = cohort
                rows.append(row)
    design = DifferentialDesignSpec(
        design_kind="multi_cohort",
        condition_column="condition",
        condition_levels=("control", "treated"),
        cohort_column="cohort",
        contrasts=(_contrast(),),
        precision_weight_column=None,
        minimum_subjects_per_level=4,
    )
    spec = TwoPartOccurrenceV2Spec(design=design, activity_threshold_raw=1.0)

    effect = fit_two_part_occurrence_v2(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["status"] == "observed"
    assert effect["occurrence_method"] == "logistic_hc1_sandwich"
    assert effect["target_prevalence"] > effect["reference_prevalence"]


def test_missing_measurement_is_excluded_instead_of_becoming_absence() -> None:
    rows = [
        _row(
            event_id="e1",
            sample_id=f"control-{index}",
            subject_id=f"control-{index}",
            condition="control",
            active=index < 2,
        )
        for index in range(8)
    ] + [
        _row(
            event_id="e1",
            sample_id=f"treated-{index}",
            subject_id=f"treated-{index}",
            condition="treated",
            active=index < 5,
        )
        for index in range(8)
    ]
    table = pd.DataFrame(rows)
    table.loc[0, "score"] = np.nan
    table.loc[0, "score_status"] = "not_estimable"
    spec = TwoPartOccurrenceV2Spec(
        design=_design("independent_two_group"),
        activity_threshold_raw=1.0,
    )

    result = fit_two_part_occurrence_v2(table, spec)
    effect = result.effects.iloc[0]

    assert len(result.subject_events) == 15
    assert effect["reference_subjects"] == 7
    assert effect["reference_occurrences"] == 1
    assert effect["observed_fraction"] == pytest.approx(15.0 / 16.0)


def test_result_is_row_order_deterministic_and_adapter_uses_configured_head() -> None:
    rows = [
        _row(
            event_id=event_id,
            sample_id=f"{event_id}-{condition}-{index}",
            subject_id=f"{event_id}-{condition}-{index}",
            condition=condition,
            active=(index < (2 if condition == "control" else active_target)),
        )
        for event_id, active_target in (("e1", 6), ("e2", 4))
        for condition in ("control", "treated")
        for index in range(8)
    ]
    table = pd.DataFrame(rows)
    spec = TwoPartOccurrenceV2Spec(
        design=_design("independent_two_group"),
        activity_threshold_raw=1.0,
    )

    first = fit_two_part_occurrence_v2(table, spec)
    second = fit_two_part_occurrence_v2(table.sample(frac=1.0, random_state=19), spec)

    pd.testing.assert_frame_equal(first.subject_events, second.subject_events)
    pd.testing.assert_frame_equal(first.effects, second.effects)
    projected = build_sample_edge_two_part_input(sample_edge_scores(), spec=spec)
    assert projected["score"].unique().tolist() == [1.5]
    assert projected["event_id"].nunique() == 1
