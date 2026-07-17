from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.design import (
    RepeatedMeasuresDesignSpec,
    balanced_contrast,
    freeze_repeated_measures_design,
)
from crychic.response import (
    RepeatedMeasuresCR2ReceiverEffect,
    RepeatedMeasuresCR2Status,
    fit_repeated_measures_cr2_receiver_effect,
    fit_repeated_measures_receiver_effect,
)
from crychic.response.repeated_cr2 import _CR2Failure, _subject_equal_cr2


def _mixed_metadata() -> pd.DataFrame:
    allocations = {
        "p1": ("control", "case"),
        "p2": ("control", "case"),
        "p3": ("control",),
        "p4": ("control",),
        "p5": ("case",),
        "p6": ("case",),
    }
    return pd.DataFrame(
        [
            {
                "sample_id": f"{subject}-{context}",
                "subject_id": subject,
                "condition": context,
            }
            for subject, contexts in allocations.items()
            for context in contexts
        ]
    )


def _design(
    metadata: pd.DataFrame,
    *,
    subject_fixed_effects: bool = False,
    min_clusters: int = 6,
):
    return freeze_repeated_measures_design(
        metadata,
        contrast=balanced_contrast(
            ("case",), ("control",), name="case_vs_control"
        ),
        spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition",),
            subject_fixed_effects=subject_fixed_effects,
            min_subjects_per_context=2,
            min_subject_clusters=min_clusters,
        ),
    )


def _mixed_values(metadata: pd.DataFrame) -> np.ndarray:
    values = {
        "p1-control": 0.0,
        "p1-case": 0.4,
        "p2-control": 0.1,
        "p2-case": 0.5,
        "p3-control": -0.1,
        "p4-control": 0.2,
        "p5-case": 0.6,
        "p6-case": 0.3,
    }
    return np.asarray([[values[sample]] for sample in metadata["sample_id"]])


def test_mixed_design_uses_subject_equal_cr2_and_preserves_cr1_api() -> None:
    metadata = _mixed_metadata()
    design = _design(metadata)
    values = _mixed_values(metadata)

    formal = fit_repeated_measures_cr2_receiver_effect(
        design,
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    )
    exploratory = fit_repeated_measures_receiver_effect(
        design,
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    )

    effect = formal.effect_for("G")
    assert effect.status is RepeatedMeasuresCR2Status.ELIGIBLE
    assert effect.effect == pytest.approx(0.4)
    assert effect.standard_error is not None and effect.standard_error > 0.0
    assert effect.n_effective_clusters == 6
    assert effect.cluster_df == 5
    assert effect.degrees_of_freedom_method == (
        "contrast_support_clusters_minus_one_guard_only_v1"
    )
    assert effect.maximum_cluster_leverage is not None
    assert effect.maximum_cluster_leverage < 1.0
    assert effect.minimum_cr2_adjustment_eigenvalue is not None
    assert effect.minimum_cr2_adjustment_eigenvalue > 0.0
    assert effect.formal_backend_eligible
    assert not effect.formal_inference_allowed
    assert not effect.exploratory_only
    assert exploratory.effect_for("G").status == "exploratory"
    assert exploratory.method == "formula_ols_subject_cluster_cr1_v1"

    frame = formal.to_frame()
    assert frame["formal_backend_eligible"].eq(True).all()
    assert frame["formal_inference_allowed"].eq(False).all()
    assert not {"p", "p_value", "q", "q_value"}.intersection(frame.columns)


def test_complete_paired_subject_fixed_effect_uses_subject_contrasts() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"p{index}-{condition}",
                "subject_id": f"p{index}",
                "condition": condition,
            }
            for index in range(8)
            for condition in ("control", "case")
        ]
    )
    offsets = {f"p{index}": index * 0.3 for index in range(8)}
    deviations = (-0.12, -0.08, -0.04, 0.0, 0.0, 0.04, 0.08, 0.12)
    values = np.asarray(
        [
            [
                offsets[row.subject_id]
                + (
                    0.7 + deviations[int(row.subject_id[1:])]
                    if row.condition == "case"
                    else 0.0
                )
            ]
            for row in metadata.itertuples()
        ]
    )

    result = fit_repeated_measures_cr2_receiver_effect(
        _design(metadata, subject_fixed_effects=True, min_clusters=6),
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    )

    effect = result.effect_for("G")
    assert effect.status is RepeatedMeasuresCR2Status.ELIGIBLE
    assert effect.backend == "subject_equal_paired_contrast_cr2_v1"
    assert effect.effect == pytest.approx(0.7)
    assert effect.n_effective_clusters == 8
    assert effect.cluster_df == 7
    assert effect.diagnostic_codes == ("paired_subject_contrast",)


def test_cr2_eligibility_and_uncertainty_are_response_scale_equivariant() -> None:
    metadata = _mixed_metadata()
    design = _design(metadata)
    values = _mixed_values(metadata)

    larger = fit_repeated_measures_cr2_receiver_effect(
        design,
        values * 1.0e-4,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    ).effect_for("G")
    smaller = fit_repeated_measures_cr2_receiver_effect(
        design,
        values * 1.0e-5,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    ).effect_for("G")

    assert larger.status is RepeatedMeasuresCR2Status.ELIGIBLE
    assert smaller.status is RepeatedMeasuresCR2Status.ELIGIBLE
    assert smaller.effect == pytest.approx(larger.effect * 0.1)
    assert smaller.standard_error == pytest.approx(larger.standard_error * 0.1)
    assert smaller.raw_precision == pytest.approx(larger.raw_precision * 100.0)


def test_subject_fixed_effect_shortcut_rejects_custom_multifactor_formula() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"p{subject}-{condition}-{region}",
                "subject_id": f"p{subject}",
                "condition": condition,
                "region": region,
            }
            for subject in range(8)
            for condition in ("control", "case")
            for region in ("R1", "R2")
        ]
    )
    positive = (
        (("condition", "case"), ("region", "R1")),
        (("condition", "case"), ("region", "R2")),
    )
    negative = (
        (("condition", "control"), ("region", "R1")),
        (("condition", "control"), ("region", "R2")),
    )
    design = freeze_repeated_measures_design(
        metadata,
        contrast=balanced_contrast(
            positive,
            negative,
            name="case_vs_control_across_regions",
        ),
        spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition", "region"),
            formula="~ condition + region",
            subject_fixed_effects=True,
            min_subjects_per_context=3,
            min_subject_clusters=6,
        ),
    )
    values = np.asarray(
        [
            [
                subject * 0.2
                + (1.0 if condition == "case" else 0.0)
                + (0.3 if region == "R2" else 0.0)
            ]
            for subject in range(8)
            for condition in ("control", "case")
            for region in ("R1", "R2")
        ]
    )

    effect = fit_repeated_measures_cr2_receiver_effect(
        design,
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    ).effect_for("G")

    assert design.estimable
    assert effect.status is RepeatedMeasuresCR2Status.NOT_ESTIMABLE
    assert effect.reason_code == "subject_fixed_effect_cr2_formula_unsupported"


def test_small_cluster_support_is_typed_not_estimable() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"p{index}-{condition}",
                "subject_id": f"p{index}",
                "condition": condition,
            }
            for index in range(3)
            for condition in ("control", "case")
        ]
    )
    values = np.asarray(
        [[index * 0.1 + (0.2 if condition == "case" else 0.0)]
         for index in range(3)
         for condition in ("control", "case")]
    )

    result = fit_repeated_measures_cr2_receiver_effect(
        _design(metadata, min_clusters=6),
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    )

    effect = result.effect_for("G")
    assert effect.status is RepeatedMeasuresCR2Status.NOT_ESTIMABLE
    assert effect.reason_code == "insufficient_subject_clusters"
    assert effect.effect is None
    assert effect.standard_error is None
    assert not effect.formal_backend_eligible


def test_formal_backend_enforces_six_cluster_floor_above_design_minimum() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"p{index}-{condition}",
                "subject_id": f"p{index}",
                "condition": condition,
            }
            for index in range(5)
            for condition in ("control", "case")
        ]
    )
    design = _design(metadata, min_clusters=2)
    values = np.asarray(
        [
            [index * 0.1 + (0.4 if condition == "case" else 0.0)]
            for index in range(5)
            for condition in ("control", "case")
        ]
    )

    effect = fit_repeated_measures_cr2_receiver_effect(
        design,
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    ).effect_for("G")

    assert design.estimable
    assert effect.status is RepeatedMeasuresCR2Status.NOT_ESTIMABLE
    assert effect.reason_code == "insufficient_subject_clusters"
    assert effect.n_effective_clusters == 5


def test_feature_missing_one_context_fails_closed_without_dropping_peer() -> None:
    metadata = _mixed_metadata()
    first = _mixed_values(metadata)[:, 0]
    second = first.copy()
    second[metadata["condition"].eq("case").to_numpy()] = np.nan
    values = np.column_stack((first, second))

    result = fit_repeated_measures_cr2_receiver_effect(
        _design(metadata),
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1", "G2"),
        receiver="Receiver",
    )

    assert result.effect_for("G1").formal_backend_eligible
    blocked = result.effect_for("G2")
    assert blocked.status is RepeatedMeasuresCR2Status.NOT_ESTIMABLE
    assert blocked.reason_code == "insufficient_subjects_per_context"


def test_cluster_leverage_singularity_fails_closed() -> None:
    matrix = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    with pytest.raises(_CR2Failure) as error:
        _subject_equal_cr2(
            matrix,
            np.asarray([2.0, 0.1, -0.1, 0.2, 0.0, 0.3]),
            np.asarray([0.0, 1.0]),
            np.asarray(["p1", "p2", "p3", "p4", "p5", "p6"]),
            n_effective_clusters=6,
            max_condition_number=1.0e8,
        )
    assert error.value.reason_code == "cr2_cluster_leverage_not_estimable"


def test_feature_specific_rank_deficiency_fails_closed() -> None:
    matrix = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    with pytest.raises(_CR2Failure) as error:
        _subject_equal_cr2(
            matrix,
            np.asarray([0.0, 0.1, -0.1, 0.2, 0.0, 0.3]),
            np.asarray([-1.0, 1.0]),
            np.asarray(["p1", "p2", "p3", "p4", "p5", "p6"]),
            n_effective_clusters=6,
            max_condition_number=1.0e8,
        )
    assert error.value.reason_code == "rank_deficient_feature_design"


def test_public_fit_reports_feature_specific_rank_deficiency() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": [f"p{index}" for index in range(8)],
            "subject_id": [f"p{index}" for index in range(8)],
            "condition": ["control"] * 4 + ["case"] * 4,
            "age": [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 3.0],
        }
    )
    design = freeze_repeated_measures_design(
        metadata,
        contrast=balanced_contrast(
            ("case",), ("control",), name="case_vs_control"
        ),
        spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition",),
            covariates=("age",),
            min_subjects_per_context=3,
            min_subject_clusters=6,
        ),
    )
    assert design.estimable
    values = np.asarray([[index * 0.1] for index in range(8)])
    values[[3, 7]] = np.nan

    result = fit_repeated_measures_cr2_receiver_effect(
        design,
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    )

    effect = result.effect_for("G")
    assert effect.status is RepeatedMeasuresCR2Status.NOT_ESTIMABLE
    assert effect.reason_code == "rank_deficient_feature_design"
    assert effect.n_effective_clusters == 6


def test_incomplete_paired_subject_count_is_reported_exactly() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"p{index}-{condition}",
                "subject_id": f"p{index}",
                "condition": condition,
            }
            for index in range(8)
            for condition in ("control", "case")
        ]
    )
    values = np.asarray(
        [
            [index * 0.1 + (0.4 if condition == "case" else 0.0)]
            for index in range(8)
            for condition in ("control", "case")
        ]
    )
    missing = metadata["sample_id"].isin(("p5-case", "p6-case", "p7-case"))
    values[missing.to_numpy()] = np.nan

    result = fit_repeated_measures_cr2_receiver_effect(
        _design(metadata, subject_fixed_effects=True, min_clusters=6),
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    )

    effect = result.effect_for("G")
    assert effect.status is RepeatedMeasuresCR2Status.NOT_ESTIMABLE
    assert effect.reason_code == "insufficient_complete_subject_clusters"
    assert effect.n_contrast_subject_clusters == 8
    assert effect.n_complete_contrast_subjects == 5
    assert effect.n_effective_clusters == 5
    assert effect.cluster_df == 4


def test_result_and_feature_are_producer_owned_and_tamper_evident() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        RepeatedMeasuresCR2ReceiverEffect()

    metadata = _mixed_metadata()
    result = fit_repeated_measures_cr2_receiver_effect(
        _design(metadata),
        _mixed_values(metadata),
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G",),
        receiver="Receiver",
    )
    object.__setattr__(result.feature_effects[0], "effect", 99.0)
    with pytest.raises(ContractError) as error:
        result.to_frame()
    assert error.value.details.code == (
        "repeated_measures_cr2_feature_integrity_violation"
    )
