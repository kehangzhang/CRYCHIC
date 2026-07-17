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
    RepeatedMeasuresReceiverEffect,
    fit_repeated_measures_receiver_effect,
)


def _mixed_metadata() -> pd.DataFrame:
    allocations = {
        "p1": ("control", "case"),
        "p2": ("control", "case"),
        "p3": ("control",),
        "p4": ("control",),
        "p5": ("case",),
        "p6": ("case",),
    }
    rows = [
        {
            "sample_id": f"{subject}-{context}",
            "subject_id": subject,
            "condition": context,
        }
        for subject, contexts in allocations.items()
        for context in contexts
    ]
    rows.append(
        {
            "sample_id": "p1-control-replicate",
            "subject_id": "p1",
            "condition": "control",
        }
    )
    return pd.DataFrame(rows)


def _design(*, subject_fixed_effects: bool = False):
    metadata = _mixed_metadata()
    return freeze_repeated_measures_design(
        metadata,
        contrast=balanced_contrast(
            ("case",), ("control",), name="case_vs_control"
        ),
        spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition",),
            subject_fixed_effects=subject_fixed_effects,
            min_subjects_per_context=3,
            min_subject_clusters=6,
        ),
    )


def _values(metadata: pd.DataFrame) -> np.ndarray:
    subject_offset = {
        subject: index * 0.04
        for index, subject in enumerate(sorted(metadata["subject_id"].unique()))
    }
    first = np.asarray(
        [
            0.2
            + subject_offset[row.subject_id]
            + (0.25 if row.condition == "case" else 0.0)
            for row in metadata.itertuples()
        ],
        dtype=np.float64,
    )
    second = np.asarray(
        [
            1.0
            - subject_offset[row.subject_id]
            - (0.1 if row.condition == "case" else 0.0)
            for row in metadata.itertuples()
        ],
        dtype=np.float64,
    )
    return np.column_stack((first, second))


def test_mixed_repeated_receiver_fit_recovers_effects_and_cr1_diagnostics() -> None:
    metadata = _mixed_metadata()
    design = _design()
    result = fit_repeated_measures_receiver_effect(
        design,
        _values(metadata),
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1", "G2"),
        receiver="Receiver",
    )

    g1 = result.effect_for("G1")
    g2 = result.effect_for("G2")
    assert g1.status == "exploratory"
    assert g2.status == "exploratory"
    # Marginal cluster-OLS retains singleton subjects; their balanced-context
    # mean offset is 0.04 in this fixture, in addition to the planted effect.
    assert g1.effect == pytest.approx(0.29)
    assert g2.effect == pytest.approx(-0.14)
    assert g1.diagnostic_standard_error == pytest.approx(0.042661458015403136)
    assert g1.n_input_samples == 9
    assert g1.n_model_cells == 8
    assert g1.n_subject_clusters == 6
    assert g1.n_repeated_subject_clusters == 2
    assert g1.n_complete_contrast_subjects == 2
    assert g1.cluster_df == 5
    assert not result.formal_inference_allowed
    frame = result.to_frame()
    assert frame["formal_inference_allowed"].eq(False).all()
    assert not {"p", "p_value", "q", "q_value"}.intersection(frame.columns)


def test_technical_replicate_is_averaged_within_subject_context_cell() -> None:
    metadata = _mixed_metadata()
    values = _values(metadata)
    replicate = metadata["sample_id"].eq("p1-control-replicate").to_numpy()
    original = metadata["sample_id"].eq("p1-control").to_numpy()
    baseline = float(values[original, 0][0])
    values[original, 0] = baseline - 1.0
    values[replicate, 0] = baseline + 1.0

    result = fit_repeated_measures_receiver_effect(
        _design(),
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1", "G2"),
        receiver="Receiver",
    )

    assert result.effect_for("G1").effect == pytest.approx(0.29)
    assert result.effect_for("G1").n_model_cells == 8


def test_feature_specific_missing_context_fails_closed_without_affecting_peer() -> None:
    metadata = _mixed_metadata()
    values = _values(metadata)
    values[metadata["condition"].eq("case").to_numpy(), 1] = np.nan

    result = fit_repeated_measures_receiver_effect(
        _design(),
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1", "G2"),
        receiver="Receiver",
    )

    assert result.effect_for("G1").status == "exploratory"
    g2 = result.effect_for("G2")
    assert g2.status == "not_estimable"
    assert g2.reason_code == "insufficient_subjects_per_context"
    assert np.isnan(g2.effect)
    assert np.isnan(g2.diagnostic_standard_error)


def test_declared_design_failure_propagates_to_every_feature() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{context}-{index}",
                "subject_id": f"{context}-{index}",
                "condition": context,
            }
            for context in ("control", "case")
            for index in range(4)
        ]
    )
    design = freeze_repeated_measures_design(
        metadata,
        contrast=balanced_contrast(
            ("case",), ("control",), name="case_vs_control"
        ),
        spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition",),
            subject_fixed_effects=True,
            min_subjects_per_context=3,
            min_subject_clusters=6,
        ),
    )

    result = fit_repeated_measures_receiver_effect(
        design,
        np.ones((len(metadata), 2)),
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1", "G2"),
        receiver="Receiver",
    )

    assert {
        effect.reason_code for effect in result.feature_effects
    } == {"subject_fixed_effects_absorb_contrast"}
    assert {effect.status for effect in result.feature_effects} == {"not_estimable"}


def test_fully_paired_subject_fixed_effect_fit_is_estimable() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"p{index}-{condition}",
                "subject_id": f"p{index}",
                "condition": condition,
            }
            for index in range(6)
            for condition in ("control", "case")
        ]
    )
    design = freeze_repeated_measures_design(
        metadata,
        contrast=balanced_contrast(
            ("case",), ("control",), name="case_vs_control"
        ),
        spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition",),
            subject_fixed_effects=True,
            min_subjects_per_context=3,
            min_subject_clusters=6,
        ),
    )
    offsets = {f"p{index}": index * 0.2 for index in range(6)}
    values = np.asarray(
        [
            [offsets[row.subject_id] + (0.7 if row.condition == "case" else 0.0)]
            for row in metadata.itertuples()
        ]
    )

    result = fit_repeated_measures_receiver_effect(
        design,
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1",),
        receiver="Receiver",
    )

    effect = result.effect_for("G1")
    assert effect.status == "exploratory"
    assert effect.effect == pytest.approx(0.7)
    assert effect.n_complete_contrast_subjects == 6


def test_input_row_order_does_not_change_result_identity() -> None:
    metadata = _mixed_metadata()
    values = _values(metadata)
    first = fit_repeated_measures_receiver_effect(
        _design(),
        values,
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1", "G2"),
        receiver="Receiver",
    )
    order = np.asarray([4, 0, 8, 2, 7, 1, 6, 3, 5])
    second = fit_repeated_measures_receiver_effect(
        _design(),
        values[order],
        sample_ids=tuple(metadata.iloc[order]["sample_id"]),
        feature_ids=("G1", "G2"),
        receiver="Receiver",
    )

    assert first.response_input_id == second.response_input_id
    assert first.artifact_id == second.artifact_id


def test_low_cluster_support_is_feature_level_not_estimable() -> None:
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
    design = freeze_repeated_measures_design(
        metadata,
        contrast=balanced_contrast(
            ("case",), ("control",), name="case_vs_control"
        ),
        spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition",),
            min_subjects_per_context=2,
            min_subject_clusters=6,
        ),
    )
    result = fit_repeated_measures_receiver_effect(
        design,
        np.ones((len(metadata), 1)),
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1",),
        receiver="Receiver",
    )

    effect = result.effect_for("G1")
    assert effect.status == "not_estimable"
    assert effect.reason_code == "insufficient_subject_clusters"
    assert effect.n_subject_clusters == 3


def test_result_integrity_mutation_fails_closed() -> None:
    metadata = _mixed_metadata()
    result = fit_repeated_measures_receiver_effect(
        _design(),
        _values(metadata),
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1", "G2"),
        receiver="Receiver",
    )
    poisoned = tuple(reversed(result.feature_effects))
    object.__setattr__(result, "feature_effects", poisoned)

    with pytest.raises(ContractError) as error:
        result.to_frame()
    assert error.value.details.code == (
        "repeated_measures_receiver_result_integrity_violation"
    )


def test_result_receiver_mutation_fails_closed() -> None:
    metadata = _mixed_metadata()
    result = fit_repeated_measures_receiver_effect(
        _design(),
        _values(metadata),
        sample_ids=tuple(metadata["sample_id"]),
        feature_ids=("G1", "G2"),
        receiver="Receiver",
    )
    object.__setattr__(result, "receiver", "Poisoned")

    with pytest.raises(ContractError) as error:
        result.to_frame()
    assert error.value.details.code == (
        "repeated_measures_receiver_result_integrity_violation"
    )


def test_receiver_effect_constructor_is_producer_owned() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        RepeatedMeasuresReceiverEffect()
