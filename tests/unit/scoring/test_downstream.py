from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from crychic.scoring import (
    DownstreamFunctional,
    apply_downstream_functional,
    apply_incremental_downstream_functional,
    fit_downstream_functional,
    fit_incremental_downstream_functional,
)


def _functional() -> DownstreamFunctional:
    return fit_downstream_functional(
        np.asarray(
            [
                [1.0, 3.0],
                [1.0, 5.0],
                [1.0, 7.0],
            ]
        ),
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        fold_id="fold-1",
        feature_ids=("G1", "G2"),
        family_ids=("F1", "F2"),
        reference_sample_ids=("s3", "s1", "s2"),
        reference_subject_ids=("p3", "p1", "p2"),
        training_subject_ids=("p3", "p1", "p2"),
        target_weight_matrix=sparse.eye(2, format="csc"),
        family_support=np.asarray([1.0, 0.5]),
        minimum_scale=0.5,
    )


def test_fit_uses_reference_median_mad_and_apply_is_hand_computable() -> None:
    functional = _functional()

    np.testing.assert_allclose(functional.feature_center, [1.0, 5.0])
    np.testing.assert_allclose(functional.feature_scale, [0.5, 2.9652])
    assert functional.training_subject_ids == ("p1", "p2", "p3")

    expression = np.asarray(
        [
            [1.0, 5.0],
            [1.5, 7.9652],
            [0.0, 2.0],
        ]
    )
    result = apply_downstream_functional(
        functional,
        expression,
        feature_ids=("G1", "G2"),
    )

    np.testing.assert_allclose(result.raw_program, [[0, 0], [1, 1], [0, 0]])
    np.testing.assert_allclose(
        result.receiver_program_score, [[0, 0], [0.5, 0.5], [0, 0]]
    )
    np.testing.assert_allclose(
        result.supported_program_score, [[0, 0], [0.5, 0.25], [0, 0]]
    )


def test_same_functional_applies_to_all_contexts_without_refitting() -> None:
    functional = _functional()
    control = apply_downstream_functional(
        functional,
        np.asarray([1.0, 5.0]),
        feature_ids=("G1", "G2"),
    )
    treated = apply_downstream_functional(
        functional,
        np.asarray([1.5, 7.9652]),
        feature_ids=("G1", "G2"),
    )

    assert control.downstream_functional_id == treated.downstream_functional_id
    assert treated.downstream_functional_id == functional.downstream_functional_id
    assert treated.receiver_program_score[0, 0] > control.receiver_program_score[0, 0]


def test_feature_order_and_training_support_are_strict() -> None:
    functional = _functional()

    with pytest.raises(ValueError, match="exactly match"):
        apply_downstream_functional(
            functional,
            np.asarray([1.0, 5.0]),
            feature_ids=("G2", "G1"),
        )
    with pytest.raises(ValueError, match="at least two"):
        fit_downstream_functional(
            np.asarray([[1.0, 2.0]]),
            receiver="R",
            contrast_name="c",
            fold_id="f",
            feature_ids=("G1", "G2"),
            family_ids=("F",),
            reference_sample_ids=("s1",),
            reference_subject_ids=("p1",),
            training_subject_ids=("p1",),
            target_weight_matrix=np.asarray([[1.0], [0.0]]),
            family_support=np.asarray([1.0]),
        )


def test_target_weights_are_normalized_and_invalid_columns_are_rejected() -> None:
    functional = fit_downstream_functional(
        np.asarray([[0.0, 1.0], [1.0, 2.0]]),
        receiver="R",
        contrast_name="c",
        fold_id="f",
        feature_ids=("G1", "G2"),
        family_ids=("F",),
        reference_sample_ids=("s1", "s2"),
        reference_subject_ids=("p1", "p2"),
        training_subject_ids=("p1", "p2"),
        target_weight_matrix=np.asarray([[1.0], [3.0]]),
        family_support=np.asarray([0.4]),
    )
    np.testing.assert_allclose(
        np.asarray(functional.target_weight_matrix.sum(axis=0)).ravel(), [1.0]
    )

    with pytest.raises(ValueError, match="positive target-weight mass"):
        fit_downstream_functional(
            np.asarray([[0.0, 1.0], [1.0, 2.0]]),
            receiver="R",
            contrast_name="c",
            fold_id="f",
            feature_ids=("G1", "G2"),
            family_ids=("F",),
            reference_sample_ids=("s1", "s2"),
            reference_subject_ids=("p1", "p2"),
            training_subject_ids=("p1", "p2"),
            target_weight_matrix=np.zeros((2, 1)),
            family_support=np.asarray([0.4]),
        )


def test_functional_identity_changes_with_learned_artifacts() -> None:
    first = _functional()
    changed = fit_downstream_functional(
        np.asarray([[1.0, 4.0], [1.0, 6.0], [1.0, 8.0]]),
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        fold_id="fold-1",
        feature_ids=("G1", "G2"),
        family_ids=("F1", "F2"),
        reference_sample_ids=("s1", "s2", "s3"),
        reference_subject_ids=("p1", "p2", "p3"),
        training_subject_ids=("p1", "p2", "p3"),
        target_weight_matrix=sparse.eye(2, format="csc"),
        family_support=np.asarray([1.0, 0.5]),
        minimum_scale=0.5,
    )

    assert changed.downstream_functional_id != first.downstream_functional_id


def test_reference_identity_covers_full_matrix_and_row_manifest() -> None:
    common = {
        "receiver": "Receiver",
        "contrast_name": "stim_vs_ctrl",
        "fold_id": "fold-1",
        "feature_ids": ("G1", "G2"),
        "family_ids": ("F1", "F2"),
        "training_subject_ids": ("p1", "p2", "p3"),
        "target_weight_matrix": sparse.eye(2, format="csc"),
        "family_support": np.asarray([1.0, 0.5]),
        "minimum_scale": 0.5,
    }
    first_expression = np.asarray([[0.0, 3.0], [1.0, 3.0], [2.0, 3.0]])
    same_summary = np.asarray([[0.0, 3.0], [1.0, 3.0], [5.0, 3.0]])
    first = fit_downstream_functional(
        first_expression,
        reference_sample_ids=("s1", "s2", "s3"),
        reference_subject_ids=("p1", "p2", "p3"),
        **common,
    )
    changed_matrix = fit_downstream_functional(
        same_summary,
        reference_sample_ids=("s1", "s2", "s3"),
        reference_subject_ids=("p1", "p2", "p3"),
        **common,
    )
    changed_mapping = fit_downstream_functional(
        first_expression,
        reference_sample_ids=("s1", "s2", "s3"),
        reference_subject_ids=("p3", "p2", "p1"),
        **common,
    )

    np.testing.assert_array_equal(first.feature_center, changed_matrix.feature_center)
    np.testing.assert_array_equal(first.feature_scale, changed_matrix.feature_scale)
    assert first.reference_input_digest != changed_matrix.reference_input_digest
    assert first.downstream_functional_id != changed_matrix.downstream_functional_id
    assert first.reference_input_digest != changed_mapping.reference_input_digest
    assert first.downstream_functional_id != changed_mapping.downstream_functional_id


def test_reference_identity_is_invariant_to_input_row_order() -> None:
    expression = np.asarray([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    common = {
        "receiver": "Receiver",
        "contrast_name": "stim_vs_ctrl",
        "fold_id": "fold-1",
        "feature_ids": ("G1", "G2"),
        "family_ids": ("F1", "F2"),
        "training_subject_ids": ("p1", "p2", "p3"),
        "target_weight_matrix": sparse.eye(2, format="csc"),
        "family_support": np.asarray([1.0, 0.5]),
        "minimum_scale": 0.5,
    }
    first = fit_downstream_functional(
        expression,
        reference_sample_ids=("s1", "s2", "s3"),
        reference_subject_ids=("p1", "p2", "p3"),
        **common,
    )
    order = np.asarray([2, 0, 1])
    reordered = fit_downstream_functional(
        expression[order],
        reference_sample_ids=("s3", "s1", "s2"),
        reference_subject_ids=("p3", "p1", "p2"),
        **common,
    )

    assert first.reference_sample_ids == ("s1", "s2", "s3")
    assert reordered.reference_sample_ids == first.reference_sample_ids
    assert reordered.reference_input_digest == first.reference_input_digest
    assert reordered.downstream_functional_id == first.downstream_functional_id


def test_missing_target_feature_propagates_only_to_affected_family() -> None:
    functional = _functional()
    result = apply_downstream_functional(
        functional,
        np.asarray([np.nan, 7.9652]),
        feature_ids=("G1", "G2"),
    )

    assert np.isnan(result.receiver_program_score[0, 0])
    assert result.receiver_program_score[0, 1] == pytest.approx(0.5)


def _incremental_functional():
    nuisance = np.ones((8, 1), dtype=float)
    regressor = np.asarray([-1.0] * 4 + [1.0] * 4)
    response = np.column_stack(
        [
            np.asarray([0.0, 0.1, -0.1, 0.0, 2.0, 2.1, 1.9, 2.0]),
            np.asarray([3.0, 3.1, 2.9, 3.0, 3.0, 3.1, 2.9, 3.0]),
        ]
    )
    return fit_incremental_downstream_functional(
        response,
        reference_mask=regressor < 0,
        nuisance_matrix=nuisance,
        context_regressor=regressor,
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        fold_id="fold-1",
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        family_ids=("LR_family",),
        nuisance_column_ids=("intercept",),
        training_subject_ids=tuple(f"p{index}" for index in range(8)),
        family_basis=np.asarray([[1.0], [0.0]]),
        precision_weights=np.ones(2),
        minimum_scale=0.25,
        null_loss_floor=1e-8,
    )


def test_heldout_active_response_has_positive_incremental_family_gain() -> None:
    functional = _incremental_functional()
    regressor = np.asarray([-1.0, -1.0, 1.0, 1.0])
    active = np.column_stack(
        [
            np.asarray([0.05, -0.05, 2.05, 1.95]),
            np.asarray([3.05, 2.95, 3.05, 2.95]),
        ]
    )
    result = apply_incremental_downstream_functional(
        functional,
        active,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=regressor,
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert result.status == "observed"
    assert result.model_gain is not None and result.model_gain > 0.95
    assert result.family_gains[0] > 0.95
    assert result.full_loss is not None and result.null_loss is not None
    assert result.full_loss < result.null_loss


def test_ligand_only_and_receiver_autonomous_have_no_family_gain() -> None:
    functional = _incremental_functional()
    regressor = np.asarray([-1.0, -1.0, 1.0, 1.0])
    ligand_only = np.asarray(
        [
            [0.0, 3.0],
            [0.1, 3.1],
            [0.0, 3.0],
            [0.1, 3.1],
        ]
    )
    autonomous = np.asarray(
        [
            [0.0, 2.0],
            [0.1, 2.1],
            [0.0, 4.0],
            [0.1, 4.1],
        ]
    )

    ligand_result = apply_incremental_downstream_functional(
        functional,
        ligand_only,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=regressor,
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )
    autonomous_result = apply_incremental_downstream_functional(
        functional,
        autonomous,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=regressor,
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert ligand_result.family_gains[0] == pytest.approx(0.0)
    assert autonomous_result.family_gains[0] == pytest.approx(0.0)


def test_target_only_can_have_gain_without_an_integrated_edge_claim() -> None:
    functional = _incremental_functional()
    target_only = np.asarray(
        [
            [0.0, 3.0],
            [0.1, 3.1],
            [2.0, 3.0],
            [2.1, 3.1],
        ]
    )
    result = apply_incremental_downstream_functional(
        functional,
        target_only,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=np.asarray([-1.0, -1.0, 1.0, 1.0]),
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert result.family_gains[0] > 0.9


def test_incremental_gain_is_not_estimable_when_null_loss_is_tiny() -> None:
    functional = _incremental_functional()
    nuisance = np.ones((2, 1))
    null_standardized = nuisance @ functional.null_nuisance_coefficients
    response = functional.feature_center + null_standardized * functional.feature_scale
    result = apply_incremental_downstream_functional(
        functional,
        response,
        nuisance_matrix=nuisance,
        context_regressor=np.asarray([-1.0, 1.0]),
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert result.status == "not_estimable"
    assert result.reason_code == "heldout_receiver_null_loss_below_floor"
    assert result.null_loss is None
    assert np.isnan(result.family_gains).all()


def test_incremental_application_rejects_changed_nuisance_columns() -> None:
    functional = _incremental_functional()

    with pytest.raises(ValueError, match="nuisance columns"):
        apply_incremental_downstream_functional(
            functional,
            np.asarray([[0.0, 3.0], [2.0, 3.0]]),
            nuisance_matrix=np.ones((2, 1)),
            context_regressor=np.asarray([-1.0, 1.0]),
            feature_ids=("TARGET", "AUTO"),
            nuisance_column_ids=("batch",),
        )
