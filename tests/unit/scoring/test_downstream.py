from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from crychic.core import ContractError
from crychic.resources import GeneNamespace, Species
from crychic.response import build_receiver_autonomous_program_resource
from crychic.scoring import (
    DownstreamFunctional,
    DownstreamRowManifest,
    IncrementalDownstreamApplication,
    IncrementalDownstreamFunctional,
    apply_downstream_functional,
    apply_incremental_downstream_functional,
    fit_downstream_functional,
    fit_incremental_downstream_functional,
)
from crychic.scoring.downstream import _project_feature_values


def _paired_manifest(regressor: np.ndarray, *, prefix: str) -> DownstreamRowManifest:
    values = np.asarray(regressor, dtype=float)
    if (
        len(values) % 2
        or not np.array_equal(values[: len(values) // 2], -np.ones(len(values) // 2))
        or not np.array_equal(values[len(values) // 2 :], np.ones(len(values) // 2))
    ):
        raise ValueError("test manifest helper requires paired -1/+1 rows")
    n_subjects = len(values) // 2
    subject_ids = tuple(
        f"{prefix}-subject-{index % n_subjects}" for index in range(len(values))
    )
    context_ids = tuple("reference" if value < 0 else "target" for value in values)
    sample_ids = tuple(
        f"{prefix}-sample-{index % n_subjects}-{context_ids[index]}"
        for index in range(len(values))
    )
    return DownstreamRowManifest(
        sample_ids=sample_ids,
        subject_ids=subject_ids,
        context_ids=context_ids,
    )


def _independent_manifest(
    context_ids: tuple[str, ...], *, prefix: str
) -> DownstreamRowManifest:
    return DownstreamRowManifest(
        sample_ids=tuple(
            f"{prefix}-sample-{index}-{context}"
            for index, context in enumerate(context_ids)
        ),
        subject_ids=tuple(
            f"{prefix}-subject-{index}" for index in range(len(context_ids))
        ),
        context_ids=context_ids,
    )


def _apply_incremental(
    functional,
    response: np.ndarray,
    regressor: np.ndarray,
):
    manifest = _paired_manifest(regressor, prefix="heldout")
    return apply_incremental_downstream_functional(
        functional,
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((len(response), 1)),
        context_regressor=regressor,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
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


def _autonomous_resource(
    matrix: np.ndarray,
    *,
    feature_ids: tuple[str, ...] = ("TARGET", "AUTO", "BACKGROUND"),
):
    return build_receiver_autonomous_program_resource(
        matrix,
        feature_ids=feature_ids,
        program_ids=("generic_program",),
        resource_id="test-generic-programs",
        version="1",
        manifest_digest="a" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def test_heldout_projection_removes_ill_conditioned_exact_span() -> None:
    rng = np.random.default_rng(4)
    left, _ = np.linalg.qr(rng.normal(size=(10, 3)))
    right, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    smallest = 10.0 ** rng.uniform(-11.9, -9.5)
    programs = left @ np.diag([1.0, 0.1, smallest]) @ right.T
    coefficients = rng.normal(size=(3, 2))
    exact_span_rows = (programs @ coefficients).T

    projected = _project_feature_values(
        exact_span_rows,
        autonomous_basis=programs,
        precision_weights=np.ones(10),
    )

    np.testing.assert_allclose(projected, 0.0, atol=1e-10)


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
    manifest = _paired_manifest(regressor, prefix="training")
    return fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
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
        training_subject_ids=tuple(sorted(set(manifest.subject_ids))),
        family_basis=np.asarray([[1.0], [0.0]]),
        precision_weights=np.ones(2),
        minimum_scale=0.25,
        null_loss_floor=1e-8,
    )


def _independent_incremental_functional():
    regressor = np.asarray([-1.0] * 4 + [1.0] * 4)
    response = np.column_stack(
        [
            np.asarray([0.0, 0.1, -0.1, 0.0, 2.0, 2.1, 1.9, 2.0]),
            np.asarray([3.0, 3.1, 2.9, 3.0, 3.0, 3.1, 2.9, 3.0]),
        ]
    )
    manifest = _independent_manifest(
        ("reference",) * 4 + ("target",) * 4,
        prefix="independent-training",
    )
    return fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        reference_mask=regressor < 0,
        nuisance_matrix=np.ones((8, 1)),
        context_regressor=regressor,
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        fold_id="fold-independent",
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        family_ids=("LR_family",),
        nuisance_column_ids=("intercept",),
        training_subject_ids=manifest.subject_ids,
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
    result = _apply_incremental(functional, active, regressor)

    assert result.status == "observed"
    assert result.model_gain is not None and result.model_gain > 0.95
    assert result.family_gains[0] > 0.95
    assert result.full_loss is not None and result.null_loss is not None
    assert result.full_loss < result.null_loss


def _projected_overlap_functional(*, active_unique_effect: float = 0.0):
    regressor = np.asarray([-1.0] * 4 + [1.0] * 4)
    generic = np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    response = np.zeros((8, 3), dtype=np.float64)
    subject_noise = np.asarray([-0.12, -0.04, 0.05, 0.11])
    response[:4, 2] = subject_noise
    response[4:, 2] = subject_noise + 0.1
    response[4:] += 2.0 * generic
    response[4:, 0] += active_unique_effect
    manifest = _paired_manifest(regressor, prefix="projected-training")
    return fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        reference_mask=regressor < 0,
        nuisance_matrix=np.ones((8, 1)),
        context_regressor=regressor,
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        fold_id="fold-1",
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO", "BACKGROUND"),
        family_ids=("LR_family",),
        nuisance_column_ids=("intercept",),
        training_subject_ids=tuple(sorted(set(manifest.subject_ids))),
        family_basis=np.asarray([[1.0], [0.0], [0.0]]),
        autonomous_program_resource=_autonomous_resource(generic[:, None]),
        precision_weights=np.ones(3),
        minimum_scale=0.25,
    )


def _apply_projected(functional, *, active_unique_effect: float = 0.0):
    regressor = np.asarray([-1.0, -1.0, 1.0, 1.0])
    generic = np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    response = np.zeros((4, 3), dtype=np.float64)
    response[:, 2] = np.asarray([-0.1, 0.1, 0.0, 0.2])
    response[2:] += 2.0 * generic
    response[2:, 0] += active_unique_effect
    manifest = _paired_manifest(regressor, prefix="projected-heldout")
    return apply_incremental_downstream_functional(
        functional,
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=regressor,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO", "BACKGROUND"),
        nuisance_column_ids=("intercept",),
    )


def test_overlapping_autonomous_program_does_not_create_false_family_gain() -> None:
    functional = _projected_overlap_functional()
    result = _apply_projected(functional)

    assert functional.autonomous_program_ids == ("generic_program",)
    assert functional.family_estimable.tolist() == [True]
    assert functional.family_retained_norm_fraction[0] == pytest.approx(
        1.0 / np.sqrt(2.0)
    )
    assert result.status == "observed"
    assert result.model_gain == pytest.approx(0.0, abs=1e-12)
    assert result.family_gains[0] == pytest.approx(0.0, abs=1e-12)
    assert functional.family_coefficients[0] == pytest.approx(0.0, abs=1e-12)


def test_unique_lr_effect_survives_overlapping_autonomous_projection() -> None:
    functional = _projected_overlap_functional(active_unique_effect=1.0)
    result = _apply_projected(functional, active_unique_effect=1.0)

    assert result.status == "observed"
    assert result.model_gain is not None and result.model_gain > 0.9
    assert result.family_gains[0] > 0.9


def test_exact_autonomous_family_overlap_fails_closed() -> None:
    regressor = np.asarray([-1.0] * 4 + [1.0] * 4)
    response = np.column_stack(
        [np.asarray([0.0] * 4 + [2.0] * 4), np.linspace(0.0, 0.2, 8)]
    )
    manifest = _paired_manifest(regressor, prefix="exact-overlap")

    with pytest.raises(ContractError) as error:
        fit_incremental_downstream_functional(
            response,
            row_manifest=manifest,
            design_sample_ids=manifest.sample_ids,
            reference_mask=regressor < 0,
            nuisance_matrix=np.ones((8, 1)),
            context_regressor=regressor,
            receiver="Receiver",
            contrast_name="stim_vs_ctrl",
            fold_id="fold-1",
            context_regressor_id="stim_vs_ctrl_regressor_v1",
            nuisance_design_id="intercept_only_v1",
            feature_ids=("TARGET", "AUTO"),
            family_ids=("LR_family",),
            nuisance_column_ids=("intercept",),
            training_subject_ids=tuple(sorted(set(manifest.subject_ids))),
            family_basis=np.asarray([[1.0], [0.0]]),
            autonomous_program_resource=_autonomous_resource(
                np.asarray([[1.0], [0.0]]), feature_ids=("TARGET", "AUTO")
            ),
            precision_weights=np.ones(2),
        )
    assert error.value.details.code == (
        "all_family_bases_not_identifiable_after_autonomous_projection"
    )


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

    ligand_result = _apply_incremental(functional, ligand_only, regressor)
    autonomous_result = _apply_incremental(functional, autonomous, regressor)

    assert ligand_result.status == "observed"
    assert ligand_result.reason_code is None
    assert ligand_result.gain_denominator_status == (
        "zero_receiver_contrast_structural_zero_v1"
    )
    assert ligand_result.family_gains[0] == 0.0
    assert autonomous_result.status == "observed"
    assert autonomous_result.family_gains[0] == pytest.approx(0.0)
    assert autonomous_result.raw_family_gains[0] < 0
    assert ligand_result.raw_family_gains[0] == 0.0


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
    result = _apply_incremental(
        functional,
        target_only,
        np.asarray([-1.0, -1.0, 1.0, 1.0]),
    )

    assert result.family_gains[0] > 0.9


def test_incremental_gain_is_structural_zero_when_null_loss_is_tiny() -> None:
    functional = _incremental_functional()
    nuisance = np.ones((2, 1))
    null_standardized = nuisance @ functional.null_nuisance_coefficients
    response = functional.feature_center + null_standardized * functional.feature_scale
    result = _apply_incremental(functional, response, np.asarray([-1.0, 1.0]))

    assert result.status == "observed"
    assert result.reason_code is None
    assert result.gain_denominator_status == (
        "zero_receiver_contrast_structural_zero_v1"
    )
    assert result.null_loss is not None
    assert result.null_loss <= functional.null_loss_floor
    assert result.model_gain == 0.0
    np.testing.assert_array_equal(result.family_gains, 0.0)


def test_incremental_gain_is_not_estimable_for_changed_context_universe() -> None:
    functional = _incremental_functional()
    manifest = DownstreamRowManifest(
        sample_ids=("h1", "h2"),
        subject_ids=("heldout-a", "heldout-b"),
        context_ids=("reference", "reference"),
    )
    result = apply_incremental_downstream_functional(
        functional,
        np.asarray([[0.0, 3.0], [2.0, 3.0]]),
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((2, 1)),
        context_regressor=np.asarray([-1.0, 1.0]),
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert result.status == "not_estimable"
    assert result.reason_code == "heldout_contrast_context_universe_mismatch"
    assert np.isnan(result.family_gains).all()


def test_incremental_gain_is_not_estimable_without_regressor_variation() -> None:
    functional = _incremental_functional()
    manifest = DownstreamRowManifest(
        sample_ids=("h-ref", "h-target"),
        subject_ids=("heldout-a", "heldout-b"),
        context_ids=("reference", "target"),
    )
    result = apply_incremental_downstream_functional(
        functional,
        np.asarray([[0.0, 3.0], [2.0, 3.0]]),
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((2, 1)),
        context_regressor=np.asarray([1.0, 1.0]),
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert result.status == "not_estimable"
    assert result.reason_code == "heldout_context_regressor_lacks_variation"
    assert np.isnan(result.family_gains).all()


def test_incremental_application_rejects_changed_nuisance_columns() -> None:
    functional = _incremental_functional()

    with pytest.raises(ValueError, match="nuisance columns"):
        regressor = np.asarray([-1.0, 1.0])
        manifest = _paired_manifest(regressor, prefix="heldout")
        apply_incremental_downstream_functional(
            functional,
            np.asarray([[0.0, 3.0], [2.0, 3.0]]),
            row_manifest=manifest,
            design_sample_ids=manifest.sample_ids,
            nuisance_matrix=np.ones((2, 1)),
            context_regressor=regressor,
            context_regressor_id="stim_vs_ctrl_regressor_v1",
            nuisance_design_id="intercept_only_v1",
            feature_ids=("TARGET", "AUTO"),
            nuisance_column_ids=("batch",),
        )


@pytest.mark.parametrize(
    ("context_regressor_id", "nuisance_design_id", "message"),
    [
        ("wrong-regressor", "intercept_only_v1", "context_regressor_id"),
        ("stim_vs_ctrl_regressor_v1", "wrong-design", "nuisance_design_id"),
    ],
)
def test_incremental_application_rejects_changed_design_lineage(
    context_regressor_id: str,
    nuisance_design_id: str,
    message: str,
) -> None:
    functional = _incremental_functional()
    regressor = np.asarray([-1.0, 1.0])
    manifest = _paired_manifest(regressor, prefix="heldout")

    with pytest.raises(ValueError, match=message):
        apply_incremental_downstream_functional(
            functional,
            np.asarray([[0.0, 3.0], [2.0, 3.0]]),
            row_manifest=manifest,
            design_sample_ids=manifest.sample_ids,
            nuisance_matrix=np.ones((2, 1)),
            context_regressor=regressor,
            context_regressor_id=context_regressor_id,
            nuisance_design_id=nuisance_design_id,
            feature_ids=("TARGET", "AUTO"),
            nuisance_column_ids=("intercept",),
        )


def test_incremental_artifacts_are_producer_owned() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        IncrementalDownstreamFunctional()
    with pytest.raises(TypeError, match="producer-owned"):
        IncrementalDownstreamApplication()


def test_incremental_fit_is_invariant_to_joint_row_reordering() -> None:
    nuisance = np.ones((8, 1), dtype=float)
    regressor = np.asarray([-1.0] * 4 + [1.0] * 4)
    response = np.column_stack(
        [
            np.asarray([0.0, 0.1, -0.1, 0.0, 2.0, 2.1, 1.9, 2.0]),
            np.asarray([3.0, 3.1, 2.9, 3.0, 3.0, 3.1, 2.9, 3.0]),
        ]
    )
    manifest = _paired_manifest(regressor, prefix="training")
    common = {
        "receiver": "Receiver",
        "contrast_name": "stim_vs_ctrl",
        "fold_id": "fold-1",
        "context_regressor_id": "stim_vs_ctrl_regressor_v1",
        "nuisance_design_id": "intercept_only_v1",
        "feature_ids": ("TARGET", "AUTO"),
        "family_ids": ("LR_family",),
        "nuisance_column_ids": ("intercept",),
        "training_subject_ids": tuple(sorted(set(manifest.subject_ids))),
        "family_basis": np.asarray([[1.0], [0.0]]),
        "precision_weights": np.ones(2),
        "minimum_scale": 0.25,
        "null_loss_floor": 1e-8,
    }
    first = fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        reference_mask=regressor < 0,
        nuisance_matrix=nuisance,
        context_regressor=regressor,
        **common,
    )
    order = np.asarray([7, 0, 5, 2, 6, 3, 1, 4])
    reordered_manifest = DownstreamRowManifest(
        sample_ids=tuple(manifest.sample_ids[index] for index in order),
        subject_ids=tuple(manifest.subject_ids[index] for index in order),
        context_ids=tuple(manifest.context_ids[index] for index in order),
    )
    reordered = fit_incremental_downstream_functional(
        response[order],
        row_manifest=reordered_manifest,
        design_sample_ids=tuple(manifest.sample_ids[index] for index in order),
        reference_mask=(regressor < 0)[order],
        nuisance_matrix=nuisance[order],
        context_regressor=regressor[order],
        **common,
    )

    assert reordered.training_input_digest == first.training_input_digest
    assert reordered.incremental_functional_id == first.incremental_functional_id
    np.testing.assert_array_equal(
        reordered.family_coefficients, first.family_coefficients
    )


def test_incremental_design_is_joined_by_sample_id() -> None:
    functional = _incremental_functional()
    regressor = np.asarray([-1.0, -1.0, 1.0, 1.0])
    response = np.asarray([[0.05, 3.05], [-0.05, 2.95], [2.05, 3.05], [1.95, 2.95]])
    manifest = _paired_manifest(regressor, prefix="heldout")
    first = apply_incremental_downstream_functional(
        functional,
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=regressor,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )
    design_order = np.asarray([2, 0, 3, 1])
    joined = apply_incremental_downstream_functional(
        functional,
        response,
        row_manifest=manifest,
        design_sample_ids=tuple(manifest.sample_ids[index] for index in design_order),
        nuisance_matrix=np.ones((4, 1))[design_order],
        context_regressor=regressor[design_order],
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert joined.application_id == first.application_id
    np.testing.assert_array_equal(joined.family_gains, first.family_gains)


def test_incremental_fit_joins_reference_mask_with_design_rows() -> None:
    nuisance = np.ones((8, 1), dtype=float)
    regressor = np.asarray([-1.0] * 4 + [1.0] * 4)
    response = np.column_stack(
        [
            np.asarray([0.0, 0.1, -0.1, 0.0, 2.0, 2.1, 1.9, 2.0]),
            np.asarray([3.0, 3.1, 2.9, 3.0, 3.0, 3.1, 2.9, 3.0]),
        ]
    )
    manifest = _paired_manifest(regressor, prefix="training")
    common = {
        "row_manifest": manifest,
        "receiver": "Receiver",
        "contrast_name": "stim_vs_ctrl",
        "fold_id": "fold-1",
        "context_regressor_id": "stim_vs_ctrl_regressor_v1",
        "nuisance_design_id": "intercept_only_v1",
        "feature_ids": ("TARGET", "AUTO"),
        "family_ids": ("LR_family",),
        "nuisance_column_ids": ("intercept",),
        "training_subject_ids": tuple(sorted(set(manifest.subject_ids))),
        "family_basis": np.asarray([[1.0], [0.0]]),
        "precision_weights": np.ones(2),
        "minimum_scale": 0.25,
        "null_loss_floor": 1e-8,
    }
    first = fit_incremental_downstream_functional(
        response,
        design_sample_ids=manifest.sample_ids,
        reference_mask=regressor < 0,
        nuisance_matrix=nuisance,
        context_regressor=regressor,
        **common,
    )
    design_order = np.asarray([7, 0, 5, 2, 6, 3, 1, 4])
    joined = fit_incremental_downstream_functional(
        response,
        design_sample_ids=tuple(manifest.sample_ids[index] for index in design_order),
        reference_mask=(regressor < 0)[design_order],
        nuisance_matrix=nuisance[design_order],
        context_regressor=regressor[design_order],
        **common,
    )

    assert joined.training_input_digest == first.training_input_digest
    assert joined.incremental_functional_id == first.incremental_functional_id


@pytest.mark.parametrize(
    "reference_mask",
    [np.asarray([1, 1, 1, 1, 0, 0, 0, 0]), np.asarray(["yes"] * 8)],
)
def test_incremental_fit_rejects_non_boolean_reference_mask(
    reference_mask: np.ndarray,
) -> None:
    regressor = np.asarray([-1.0] * 4 + [1.0] * 4)
    manifest = _paired_manifest(regressor, prefix="training")

    with pytest.raises(ValueError, match="boolean vector"):
        fit_incremental_downstream_functional(
            np.ones((8, 2)),
            row_manifest=manifest,
            design_sample_ids=manifest.sample_ids,
            reference_mask=reference_mask,
            nuisance_matrix=np.ones((8, 1)),
            context_regressor=regressor,
            receiver="Receiver",
            contrast_name="stim_vs_ctrl",
            fold_id="fold-1",
            context_regressor_id="regressor-v1",
            nuisance_design_id="nuisance-v1",
            feature_ids=("TARGET", "AUTO"),
            family_ids=("LR_family",),
            nuisance_column_ids=("intercept",),
            training_subject_ids=tuple(sorted(set(manifest.subject_ids))),
            family_basis=np.asarray([[1.0], [0.0]]),
        )


def test_incremental_application_rejects_training_subject_overlap() -> None:
    functional = _incremental_functional()
    regressor = np.asarray([-1.0, 1.0])
    manifest = DownstreamRowManifest(
        sample_ids=("overlap-reference", "new-target"),
        subject_ids=(functional.training_subject_ids[0], "heldout-subject"),
        context_ids=("reference", "target"),
    )

    with pytest.raises(ValueError, match="overlap training"):
        apply_incremental_downstream_functional(
            functional,
            np.asarray([[0.0, 3.0], [2.0, 3.0]]),
            row_manifest=manifest,
            design_sample_ids=manifest.sample_ids,
            nuisance_matrix=np.ones((2, 1)),
            context_regressor=regressor,
            context_regressor_id="stim_vs_ctrl_regressor_v1",
            nuisance_design_id="intercept_only_v1",
            feature_ids=("TARGET", "AUTO"),
            nuisance_column_ids=("intercept",),
        )


def test_incremental_application_rejects_training_sample_id_reuse() -> None:
    functional = _incremental_functional()
    manifest = DownstreamRowManifest(
        sample_ids=(functional.training_sample_ids[0], "new-sample"),
        subject_ids=("heldout-a", "heldout-b"),
        context_ids=("reference", "target"),
    )

    with pytest.raises(ValueError, match="sample IDs overlap"):
        apply_incremental_downstream_functional(
            functional,
            np.asarray([[0.0, 3.0], [2.0, 3.0]]),
            row_manifest=manifest,
            design_sample_ids=manifest.sample_ids,
            nuisance_matrix=np.ones((2, 1)),
            context_regressor=np.asarray([-1.0, 1.0]),
            context_regressor_id="stim_vs_ctrl_regressor_v1",
            nuisance_design_id="intercept_only_v1",
            feature_ids=("TARGET", "AUTO"),
            nuisance_column_ids=("intercept",),
        )


def test_incremental_functional_detects_mutated_coefficients() -> None:
    functional = _incremental_functional()
    object.__setattr__(
        functional,
        "family_coefficients",
        functional.family_coefficients.copy(),
    )
    functional.family_coefficients[0] += 1.0

    with pytest.raises(ContractError) as error:
        functional.to_dict()

    assert error.value.details.code == "incremental_functional_integrity_violation"


def test_incremental_sparse_basis_has_immutable_backing_buffers() -> None:
    functional = _incremental_functional()

    for values in (
        functional.family_basis.data,
        functional.family_basis.indices,
        functional.family_basis.indptr,
    ):
        with pytest.raises(ValueError, match="cannot set WRITEABLE"):
            values.setflags(write=True)
        assert isinstance(values.base, np.ndarray)
        assert not values.base.flags.writeable


def test_incremental_application_detects_mutated_loss_vector() -> None:
    functional = _incremental_functional()
    result = _apply_incremental(
        functional,
        np.asarray([[0.0, 3.0], [0.1, 3.1], [2.0, 3.0], [2.1, 3.1]]),
        np.asarray([-1.0, -1.0, 1.0, 1.0]),
    )
    object.__setattr__(result, "family_gains", result.family_gains.copy())
    result.family_gains[0] = 0.0

    with pytest.raises(ContractError) as error:
        result.to_dict()

    assert error.value.details.code == "incremental_application_integrity_violation"


def test_incremental_application_identity_binds_subject_loss_labels() -> None:
    functional = _incremental_functional()
    result = _apply_incremental(
        functional,
        np.asarray([[0.0, 3.0], [0.1, 3.1], [2.0, 3.0], [2.1, 3.1]]),
        np.asarray([-1.0, -1.0, 1.0, 1.0]),
    )
    object.__setattr__(result, "subject_ids", tuple(reversed(result.subject_ids)))

    with pytest.raises(ContractError) as error:
        result.to_dict()

    assert error.value.details.code == "incremental_application_integrity_violation"


def test_downstream_row_manifest_detects_mapping_mutation() -> None:
    manifest = DownstreamRowManifest(
        sample_ids=("s1", "s2"),
        subject_ids=("p1", "p2"),
        context_ids=("reference", "target"),
    )
    object.__setattr__(manifest, "subject_ids", ("p2", "p1"))

    with pytest.raises(ContractError) as error:
        manifest._require_intact()

    assert error.value.details.code == "downstream_row_manifest_integrity_violation"


def test_incremental_reports_negative_raw_gain_without_negative_bounded_gain() -> None:
    functional = _incremental_functional()
    regressor = np.asarray([-1.0, -1.0, 1.0, 1.0])
    reversed_effect = np.asarray([[0.0, 3.0], [0.1, 3.1], [-2.0, 3.0], [-1.9, 3.1]])
    result = _apply_incremental(functional, reversed_effect, regressor)

    assert result.status == "observed"
    assert result.raw_model_gain is not None and result.raw_model_gain < 0
    assert result.model_gain == 0.0
    assert result.raw_family_gains[0] < 0
    assert result.family_gains[0] == 0.0


def test_incremental_loss_requires_every_subject_in_every_context() -> None:
    functional = _incremental_functional()
    manifest = DownstreamRowManifest(
        sample_ids=("a1", "a2", "a3", "b1"),
        subject_ids=("heldout-a", "heldout-a", "heldout-a", "heldout-b"),
        context_ids=("reference", "reference", "target", "target"),
    )
    response = np.asarray([[0.0, 3.0], [0.0, 3.0], [2.0, 3.0], [3.0, 3.0]])
    result = apply_incremental_downstream_functional(
        functional,
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=np.asarray([-1.0, -1.0, 1.0, 1.0]),
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert result.status == "not_estimable"
    assert result.reason_code == "paired_contrast_loss_required"
    assert result.null_loss is None


def test_incremental_loss_supports_independent_subject_groups() -> None:
    functional = _independent_incremental_functional()
    manifest = _independent_manifest(
        ("reference", "reference", "target", "target"),
        prefix="independent-heldout",
    )
    response = np.asarray([[0.05, 3.05], [-0.05, 2.95], [2.05, 3.05], [1.95, 2.95]])
    result = apply_incremental_downstream_functional(
        functional,
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=np.asarray([-1.0, -1.0, 1.0, 1.0]),
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert functional.loss_design == "independent_subject_pseudocontrasts_v1"
    np.testing.assert_allclose(functional.loss_context_weights, [-0.5, 0.5])
    assert result.status == "observed"
    assert result.model_gain is not None and result.model_gain > 0.9
    assert result.family_gains[0] > 0.9
    assert result.subject_ids == manifest.subject_ids
    assert "independent_pseudocontrast" in result.loss_aggregation


def test_incremental_loss_supports_independent_multicontext_contrast() -> None:
    contexts = (
        "reference-a",
        "reference-a",
        "reference-b",
        "reference-b",
        "target",
        "target",
    )
    regressor = np.asarray([-1.0, -1.0, -1.0, -1.0, 2.0, 2.0])
    response = np.column_stack(
        [
            np.asarray([0.0, 0.1, -0.1, 0.0, 3.0, 3.1]),
            np.asarray([2.9, 3.1, 3.0, 3.0, 3.1, 2.9]),
        ]
    )
    training_manifest = _independent_manifest(contexts, prefix="multicontext-training")
    functional = fit_incremental_downstream_functional(
        response,
        row_manifest=training_manifest,
        design_sample_ids=training_manifest.sample_ids,
        reference_mask=regressor < 0,
        nuisance_matrix=np.ones((6, 1)),
        context_regressor=regressor,
        receiver="Receiver",
        contrast_name="target_vs_two_references",
        fold_id="fold-multicontext",
        context_regressor_id="multicontext_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        family_ids=("LR_family",),
        nuisance_column_ids=("intercept",),
        training_subject_ids=training_manifest.subject_ids,
        family_basis=np.asarray([[1.0], [0.0]]),
        precision_weights=np.ones(2),
        minimum_scale=0.25,
    )
    heldout_manifest = _independent_manifest(contexts, prefix="multicontext-heldout")
    heldout = apply_incremental_downstream_functional(
        functional,
        response + np.asarray([0.02, -0.02]),
        row_manifest=heldout_manifest,
        design_sample_ids=heldout_manifest.sample_ids,
        nuisance_matrix=np.ones((6, 1)),
        context_regressor=regressor,
        context_regressor_id="multicontext_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    np.testing.assert_allclose(
        functional.loss_context_weights,
        [-1.0 / 6.0, -1.0 / 6.0, 1.0 / 3.0],
    )
    assert heldout.status == "observed"
    assert heldout.model_gain is not None and heldout.model_gain > 0.9


def test_incremental_loss_averages_technical_rows_within_subject_context() -> None:
    functional = _incremental_functional()
    base_manifest = DownstreamRowManifest(
        sample_ids=("a-ref", "a-target", "b-ref", "b-target"),
        subject_ids=("heldout-a", "heldout-a", "heldout-b", "heldout-b"),
        context_ids=("reference", "target", "reference", "target"),
    )
    base_response = np.asarray([[0.0, 3.0], [2.0, 3.0], [0.1, 3.1], [2.1, 3.1]])
    base_regressor = np.asarray([-1.0, 1.0, -1.0, 1.0])
    base = apply_incremental_downstream_functional(
        functional,
        base_response,
        row_manifest=base_manifest,
        design_sample_ids=base_manifest.sample_ids,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=base_regressor,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )
    replicated_manifest = DownstreamRowManifest(
        sample_ids=("a-ref-1", "a-ref-2", "a-target", "b-ref", "b-target"),
        subject_ids=(
            "heldout-a",
            "heldout-a",
            "heldout-a",
            "heldout-b",
            "heldout-b",
        ),
        context_ids=("reference", "reference", "target", "reference", "target"),
    )
    replicated_response = np.vstack(
        [base_response[0], base_response[0], base_response[1:]]
    )
    replicated_regressor = np.asarray([-1.0, -1.0, 1.0, -1.0, 1.0])
    replicated = apply_incremental_downstream_functional(
        functional,
        replicated_response,
        row_manifest=replicated_manifest,
        design_sample_ids=replicated_manifest.sample_ids,
        nuisance_matrix=np.ones((5, 1)),
        context_regressor=replicated_regressor,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert replicated.null_loss == pytest.approx(base.null_loss)
    assert replicated.full_loss == pytest.approx(base.full_loss)
    np.testing.assert_allclose(replicated.family_gains, base.family_gains)


def test_incremental_loss_averages_variable_technical_values_before_squaring() -> None:
    functional = _incremental_functional()
    base_manifest = DownstreamRowManifest(
        sample_ids=("a-ref", "a-target", "b-ref", "b-target"),
        subject_ids=("heldout-a", "heldout-a", "heldout-b", "heldout-b"),
        context_ids=("reference", "target", "reference", "target"),
    )
    base_response = np.asarray([[1.0, 3.0], [2.0, 3.0], [0.1, 3.1], [2.1, 3.1]])
    base_regressor = np.asarray([-1.0, 1.0, -1.0, 1.0])
    base = apply_incremental_downstream_functional(
        functional,
        base_response,
        row_manifest=base_manifest,
        design_sample_ids=base_manifest.sample_ids,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=base_regressor,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )
    split_manifest = DownstreamRowManifest(
        sample_ids=("a-ref-0", "a-ref-2", "a-target", "b-ref", "b-target"),
        subject_ids=(
            "heldout-a",
            "heldout-a",
            "heldout-a",
            "heldout-b",
            "heldout-b",
        ),
        context_ids=("reference", "reference", "target", "reference", "target"),
    )
    split_response = np.vstack(
        [
            np.asarray([0.0, 3.0]),
            np.asarray([2.0, 3.0]),
            base_response[1:],
        ]
    )
    split_regressor = np.asarray([-1.0, -1.0, 1.0, -1.0, 1.0])
    split = apply_incremental_downstream_functional(
        functional,
        split_response,
        row_manifest=split_manifest,
        design_sample_ids=split_manifest.sample_ids,
        nuisance_matrix=np.ones((5, 1)),
        context_regressor=split_regressor,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        nuisance_column_ids=("intercept",),
    )

    assert split.null_loss == pytest.approx(base.null_loss)
    assert split.full_loss == pytest.approx(base.full_loss)
    np.testing.assert_allclose(split.family_gains, base.family_gains)


def test_incremental_contrast_loss_is_invariant_to_subject_constant_baselines() -> None:
    functional = _incremental_functional()
    regressor = np.asarray([-1.0, -1.0, 1.0, 1.0])
    response = np.asarray([[0.0, 3.0], [0.1, 3.1], [2.0, 3.0], [2.1, 3.1]])
    shifted = response.copy()
    subject_offsets = np.asarray([-25.0, 40.0])
    shifted[:2, 0] += subject_offsets
    shifted[2:, 0] += subject_offsets

    base = _apply_incremental(functional, response, regressor)
    with_baselines = _apply_incremental(functional, shifted, regressor)

    assert with_baselines.null_loss == pytest.approx(base.null_loss)
    assert with_baselines.full_loss == pytest.approx(base.full_loss)
    assert with_baselines.model_gain == pytest.approx(base.model_gain)
    np.testing.assert_allclose(
        with_baselines.subject_null_losses, base.subject_null_losses
    )
    np.testing.assert_allclose(with_baselines.family_gains, base.family_gains)
    assert not np.allclose(with_baselines.sample_null_losses, base.sample_null_losses)


def test_incremental_fit_ignores_exact_technical_replication_weight() -> None:
    base = _incremental_functional()
    regressor = np.asarray([-1.0] * 4 + [1.0] * 4)
    response = np.column_stack(
        [
            np.asarray([0.0, 0.1, -0.1, 0.0, 2.0, 2.1, 1.9, 2.0]),
            np.asarray([3.0, 3.1, 2.9, 3.0, 3.0, 3.1, 2.9, 3.0]),
        ]
    )
    manifest = _paired_manifest(regressor, prefix="training")
    replicated_manifest = DownstreamRowManifest(
        sample_ids=(f"{manifest.sample_ids[0]}-rep", *manifest.sample_ids),
        subject_ids=(manifest.subject_ids[0], *manifest.subject_ids),
        context_ids=(manifest.context_ids[0], *manifest.context_ids),
    )
    replicated_response = np.vstack([response[0], response])
    replicated_regressor = np.concatenate([[regressor[0]], regressor])
    replicated = fit_incremental_downstream_functional(
        replicated_response,
        row_manifest=replicated_manifest,
        design_sample_ids=replicated_manifest.sample_ids,
        reference_mask=replicated_regressor < 0,
        nuisance_matrix=np.ones((9, 1)),
        context_regressor=replicated_regressor,
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        fold_id="fold-1",
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("TARGET", "AUTO"),
        family_ids=("LR_family",),
        nuisance_column_ids=("intercept",),
        training_subject_ids=tuple(sorted(set(manifest.subject_ids))),
        family_basis=np.asarray([[1.0], [0.0]]),
        precision_weights=np.ones(2),
        minimum_scale=0.25,
        null_loss_floor=1e-8,
    )

    np.testing.assert_allclose(replicated.feature_center, base.feature_center)
    np.testing.assert_allclose(replicated.feature_scale, base.feature_scale)
    np.testing.assert_allclose(
        replicated.family_coefficients, base.family_coefficients, atol=1e-12
    )
