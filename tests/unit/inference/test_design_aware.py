from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.support.sample_edge_v2 import sample_edge_scores

from crychic.inference import (
    DifferentialContrastSpec,
    DifferentialDesignKind,
    DifferentialDesignSpec,
    build_sample_edge_differential_input,
    fit_design_aware_differential,
)
from crychic.scoring import SampleEdgeScoreV2


def _contrast(
    name: str = "treated-vs-control",
    *,
    reference: str = "control",
    target: str = "treated",
) -> DifferentialContrastSpec:
    return DifferentialContrastSpec(
        name=name,
        weights=((reference, -1.0), (target, 1.0)),
    )


def _row(
    *,
    sample_id: str,
    subject_id: str,
    condition: object,
    score: float,
    event_id: str = "event-1",
    reliability_weight: float = 1.0,
    **metadata: object,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "sample_id": sample_id,
        "subject_id": subject_id,
        "condition": condition,
        "score": score,
        "score_status": "observed",
        "out_of_fold": True,
        "reliability_weight": reliability_weight,
        **metadata,
    }


def _independent_two_group_table(*, effect: float = 2.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    residuals = (-0.30, -0.20, -0.10, 0.00, 0.00, 0.10, 0.20, 0.30)
    for condition, prefix, shift in (
        ("control", "C", 0.0),
        ("treated", "T", effect),
    ):
        for index, residual in enumerate(residuals):
            rows.append(
                _row(
                    sample_id=f"{prefix}-sample-{index}",
                    subject_id=f"{prefix}-subject-{index}",
                    condition=condition,
                    score=4.0 + shift + residual,
                    reliability_weight=0.25 if index % 2 else 1.0,
                )
            )
    return pd.DataFrame(rows)


def test_independent_two_group_uses_raw_score_and_hc3() -> None:
    spec = DifferentialDesignSpec(
        design_kind=DifferentialDesignKind.INDEPENDENT_TWO_GROUP,
        condition_column="condition",
        condition_levels=("control", "treated"),
        contrasts=(_contrast(),),
        minimum_subjects_per_level=4,
    )

    result = fit_design_aware_differential(
        _independent_two_group_table(), spec
    ).effects.iloc[0]

    assert result["status"] == "observed"
    assert result["effect"] == pytest.approx(2.0, abs=1e-12)
    assert result["covariance_method"] == "hc3_diagnostic"
    assert result["n_subjects"] == 16
    assert result["observed_fraction"] == 1.0
    assert result["diagnostic_p_value"] < 0.001


def test_independent_multi_group_reports_contrasts_and_omnibus() -> None:
    offsets = {"A": 0.0, "B": 1.0, "C": 3.0}
    rows = [
        _row(
            sample_id=f"{level}-{index}",
            subject_id=f"{level}-subject-{index}",
            condition=level,
            score=offset + (index - 3.5) * 0.05,
        )
        for level, offset in offsets.items()
        for index in range(8)
    ]
    spec = DifferentialDesignSpec(
        design_kind="independent_multi_group",
        condition_column="condition",
        condition_levels=("A", "B", "C"),
        contrasts=(
            _contrast("B-vs-A", reference="A", target="B"),
            _contrast("C-vs-A", reference="A", target="C"),
        ),
        minimum_subjects_per_level=4,
    )

    result = fit_design_aware_differential(pd.DataFrame(rows), spec)
    effects = result.effects.set_index("contrast_name")

    assert effects.loc["B-vs-A", "effect"] == pytest.approx(1.0)
    assert effects.loc["C-vs-A", "effect"] == pytest.approx(3.0)
    assert result.omnibus.iloc[0]["status"] == "observed"
    assert result.omnibus.iloc[0]["numerator_df"] == 2.0
    assert result.omnibus.iloc[0]["diagnostic_p_value"] < 0.001


def test_paired_difference_removes_subject_offsets() -> None:
    differences = (0.8, 0.9, 1.0, 1.1, 1.2, 0.9, 1.1, 1.0)
    rows: list[dict[str, object]] = []
    for index, difference in enumerate(differences):
        offset = 50.0 * index
        for condition, value in (
            ("control", offset),
            ("treated", offset + difference),
        ):
            rows.append(
                _row(
                    sample_id=f"S{index}-{condition}",
                    subject_id=f"S{index}",
                    condition=condition,
                    score=value,
                )
            )
    spec = DifferentialDesignSpec(
        design_kind="paired",
        condition_column="condition",
        condition_levels=("control", "treated"),
        contrasts=(_contrast(),),
        minimum_subjects_per_level=4,
    )

    effect = fit_design_aware_differential(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["status"] == "observed"
    assert effect["effect"] == pytest.approx(np.mean(differences))
    assert effect["n_subjects"] == 8
    assert effect["covariance_method"] == "hc3_diagnostic"


def test_repeated_design_uses_subject_cluster_cr2() -> None:
    rows: list[dict[str, object]] = []
    subject_deviations = (-0.4, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)
    for index, deviation in enumerate(subject_deviations):
        for condition, shift in (("control", 0.0), ("treated", 1.0)):
            rows.append(
                _row(
                    sample_id=f"R{index}-{condition}",
                    subject_id=f"R{index}",
                    condition=condition,
                    score=5.0 + deviation + shift + 0.02 * index * shift,
                )
            )
    spec = DifferentialDesignSpec(
        design_kind="repeated",
        condition_column="condition",
        condition_levels=("control", "treated"),
        contrasts=(_contrast(),),
        minimum_subjects_per_level=4,
        minimum_clusters_for_cr2=6,
    )

    effect = fit_design_aware_differential(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["status"] == "observed"
    assert effect["effect"] == pytest.approx(1.07)
    assert effect["covariance_method"] == "subject_cluster_cr2_diagnostic"
    assert effect["n_clusters"] == 8
    assert effect["residual_df"] == 7.0


def test_repeated_design_does_not_fall_back_to_independent_hc3() -> None:
    rows = [
        _row(
            sample_id=f"R{index}-{condition}",
            subject_id=f"R{index}",
            condition=condition,
            score=float(index) + shift,
        )
        for index in range(4)
        for condition, shift in (("control", 0.0), ("treated", 1.0))
    ]
    spec = DifferentialDesignSpec(
        design_kind="repeated",
        condition_column="condition",
        condition_levels=("control", "treated"),
        contrasts=(_contrast(),),
        minimum_subjects_per_level=4,
        minimum_clusters_for_cr2=6,
    )

    effect = fit_design_aware_differential(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["effect"] == pytest.approx(1.0)
    assert effect["covariance_method"] == "insufficient_clusters_for_cr2_diagnostic"
    assert pd.isna(effect["standard_error"])
    assert pd.isna(effect["diagnostic_p_value"])


def test_multi_cohort_fixed_effect_removes_cohort_shift() -> None:
    rows = []
    for cohort, cohort_shift in (("site-a", 0.0), ("site-b", 10.0)):
        for condition, condition_shift in (("control", 0.0), ("treated", 1.5)):
            for index in range(6):
                rows.append(
                    _row(
                        sample_id=f"{cohort}-{condition}-{index}",
                        subject_id=f"{cohort}-{condition}-subject-{index}",
                        condition=condition,
                        score=cohort_shift + condition_shift + index * 0.03,
                        cohort=cohort,
                    )
                )
    spec = DifferentialDesignSpec(
        design_kind="multi_cohort",
        condition_column="condition",
        condition_levels=("control", "treated"),
        cohort_column="cohort",
        contrasts=(_contrast(),),
        minimum_subjects_per_level=4,
    )

    effect = fit_design_aware_differential(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["status"] == "observed"
    assert effect["effect"] == pytest.approx(1.5)
    assert effect["design_rank"] == 3


def test_continuous_design_estimates_slope() -> None:
    residuals = np.tile(np.asarray([-0.1, 0.0, 0.1, 0.0]), 6)
    rows = [
        _row(
            sample_id=f"dose-{index}",
            subject_id=f"dose-subject-{index}",
            condition=float(index),
            score=2.0 + 1.25 * index + float(residuals[index]),
        )
        for index in range(24)
    ]
    spec = DifferentialDesignSpec(
        design_kind="continuous",
        condition_column="condition",
        minimum_subjects_per_level=4,
    )

    effect = fit_design_aware_differential(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["status"] == "observed"
    assert effect["contrast_name"] == "slope:condition"
    assert effect["effect"] == pytest.approx(1.25, abs=0.005)
    assert effect["covariance_method"] == "hc3_diagnostic"


def test_confounded_multi_cohort_is_typed_not_estimable() -> None:
    rows = [
        _row(
            sample_id=f"{condition}-{index}",
            subject_id=f"{condition}-subject-{index}",
            condition=condition,
            score=float(index) + shift,
            cohort=cohort,
        )
        for condition, cohort, shift in (
            ("control", "site-a", 0.0),
            ("treated", "site-b", 1.0),
        )
        for index in range(6)
    ]
    spec = DifferentialDesignSpec(
        design_kind="multi_cohort",
        condition_column="condition",
        condition_levels=("control", "treated"),
        cohort_column="cohort",
        contrasts=(_contrast(),),
        minimum_subjects_per_level=4,
    )

    effect = fit_design_aware_differential(pd.DataFrame(rows), spec).effects.iloc[0]

    assert effect["status"] == "not_estimable"
    assert effect["reason_code"] == "rank_deficient_or_ill_conditioned_design"
    assert pd.isna(effect["effect"])


def test_analytic_results_never_claim_formal_inference() -> None:
    spec = DifferentialDesignSpec(
        design_kind="independent_two_group",
        condition_column="condition",
        condition_levels=("control", "treated"),
        contrasts=(_contrast(),),
        minimum_subjects_per_level=4,
    )

    result = fit_design_aware_differential(_independent_two_group_table(), spec)

    assert not result.effects["formal_inference_allowed"].any()
    assert not result.omnibus["formal_inference_allowed"].any()
    assert (
        result.effects[["p_value", "q_value", "ci_lower", "ci_upper"]]
        .isna()
        .all(axis=None)
    )
    assert result.omnibus[["p_value", "q_value"]].isna().all(axis=None)


def test_unsupported_score_status_is_rejected() -> None:
    spec = DifferentialDesignSpec(
        design_kind="independent_two_group",
        condition_column="condition",
        condition_levels=("control", "treated"),
        contrasts=(_contrast(),),
        minimum_subjects_per_level=4,
    )
    table = _independent_two_group_table()
    table.loc[0, "score_status"] = "invented_status"

    with pytest.raises(ValueError, match="score_status contains unsupported"):
        fit_design_aware_differential(table, spec)


def test_design_spec_rejects_untyped_contrasts_cleanly() -> None:
    with pytest.raises(TypeError, match="DifferentialContrastSpec"):
        DifferentialDesignSpec(
            design_kind="independent_two_group",
            condition_column="condition",
            condition_levels=("control", "treated"),
            contrasts=("not-a-contrast",),  # type: ignore[arg-type]
        )


def test_result_identity_is_row_order_deterministic() -> None:
    spec = DifferentialDesignSpec(
        design_kind="independent_two_group",
        condition_column="condition",
        condition_levels=("control", "treated"),
        contrasts=(_contrast(),),
        minimum_subjects_per_level=4,
    )
    table = _independent_two_group_table()

    first = fit_design_aware_differential(table, spec)
    second = fit_design_aware_differential(
        table.sample(frac=1.0, random_state=19), spec
    )

    pd.testing.assert_frame_equal(first.effects, second.effects)
    pd.testing.assert_frame_equal(first.omnibus, second.omnibus)


def test_sample_edge_adapter_uses_parent_measurement_status() -> None:
    original = sample_edge_scores()
    table = original.table.copy(deep=True)
    table.loc[0, "sender_detection_raw"] = np.nan
    table.loc[0, "status"] = "not_estimable"
    table.loc[0, "coverage_status"] = "not_measured_or_not_estimable"
    table.loc[0, "reason_code"] = "sender_not_measured"
    table["effective_candidate_count"] = 1
    for column in ("parent_peak_raw", "parent_total_raw", "parent_mean_raw"):
        table[column] = 1.0
    scores = SampleEdgeScoreV2(table=table, provenance=original.provenance)

    parent = build_sample_edge_differential_input(scores, score_head="parent_mean_raw")
    sender = build_sample_edge_differential_input(
        scores, score_head="sender_detection_raw"
    )

    assert len(parent) == 1
    assert parent.iloc[0]["score"] == 1.0
    assert parent.iloc[0]["score_status"] == "observed"
    assert len(sender) == 2
    assert set(sender["score_status"]) == {"not_estimable", "observed"}
