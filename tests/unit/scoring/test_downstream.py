from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from crychic.attribution import RelativePenaltyCandidate
from crychic.core import ContractError
from crychic.resources import GeneNamespace, Species
from crychic.response import build_receiver_autonomous_program_resource
from crychic.scoring import (
    DownstreamApplication,
    DownstreamFunctional,
    DownstreamRowManifest,
    IncrementalDownstreamApplication,
    IncrementalDownstreamFunctional,
    apply_downstream_functional,
    apply_incremental_downstream_functional,
    fit_downstream_functional,
    fit_incremental_downstream_functional,
)
from crychic.scoring.downstream import _matrix_digest, _project_feature_values


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
        reference_context_ids=("reference", "reference", "reference"),
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


def test_downstream_application_is_producer_owned_and_checks_bound_algebra() -> None:
    functional = _functional()
    expression = np.asarray([[1.5, 7.9652]])

    with pytest.raises(TypeError):
        DownstreamApplication(
            downstream_functional_id=functional.downstream_functional_id,
            raw_program=np.asarray([[999.0, 999.0]]),
            receiver_program_score=np.asarray([[0.999, 0.999]]),
            supported_program_score=np.asarray([[0.123, 0.123]]),
        )

    application = apply_downstream_functional(
        functional,
        expression,
        feature_ids=("G1", "G2"),
        input_row_manifest_id="heldout-row-manifest",
    )
    application._require_intact()
    assert application.input_row_manifest_id == "heldout-row-manifest"
    assert application.input_expression_digest
    np.testing.assert_allclose(
        application.receiver_program_score,
        application.raw_program / (1.0 + application.raw_program),
    )
    np.testing.assert_allclose(
        application.supported_program_score,
        application.receiver_program_score * functional.family_support,
    )

    poisoned = application.receiver_program_score.copy()
    poisoned[:] = 0.123
    poisoned.setflags(write=False)
    object.__setattr__(application, "receiver_program_score", poisoned)
    with pytest.raises(ContractError) as error:
        application._require_intact()
    assert error.value.details.code == "downstream_application_integrity_violation"


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
            reference_context_ids=("reference",),
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
        reference_context_ids=("reference", "reference"),
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
            reference_context_ids=("reference", "reference"),
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
        reference_context_ids=("reference", "reference", "reference"),
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
        reference_context_ids=("reference", "reference", "reference"),
        **common,
    )
    changed_matrix = fit_downstream_functional(
        same_summary,
        reference_sample_ids=("s1", "s2", "s3"),
        reference_subject_ids=("p1", "p2", "p3"),
        reference_context_ids=("reference", "reference", "reference"),
        **common,
    )
    changed_mapping = fit_downstream_functional(
        first_expression,
        reference_sample_ids=("s1", "s2", "s3"),
        reference_subject_ids=("p3", "p2", "p1"),
        reference_context_ids=("reference", "reference", "reference"),
        **common,
    )
    changed_context = fit_downstream_functional(
        first_expression,
        reference_sample_ids=("s1", "s2", "s3"),
        reference_subject_ids=("p1", "p2", "p3"),
        reference_context_ids=("reference", "reference", "forged-context"),
        **common,
    )

    np.testing.assert_array_equal(first.feature_center, changed_matrix.feature_center)
    np.testing.assert_array_equal(first.feature_scale, changed_matrix.feature_scale)
    assert first.reference_input_digest != changed_matrix.reference_input_digest
    assert first.downstream_functional_id != changed_matrix.downstream_functional_id
    assert first.reference_input_digest != changed_mapping.reference_input_digest
    assert first.downstream_functional_id != changed_mapping.downstream_functional_id
    assert first.reference_row_manifest_id != changed_context.reference_row_manifest_id
    assert first.reference_input_digest != changed_context.reference_input_digest
    assert first.downstream_functional_id != changed_context.downstream_functional_id


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
        reference_context_ids=("reference", "reference", "reference"),
        **common,
    )
    order = np.asarray([2, 0, 1])
    reordered = fit_downstream_functional(
        expression[order],
        reference_sample_ids=("s3", "s1", "s2"),
        reference_subject_ids=("p3", "p1", "p2"),
        reference_context_ids=("reference", "reference", "reference"),
        **common,
    )

    assert first.reference_sample_ids == ("s1", "s2", "s3")
    assert reordered.reference_sample_ids == first.reference_sample_ids
    assert reordered.reference_input_digest == first.reference_input_digest
    assert reordered.reference_row_manifest_id == first.reference_row_manifest_id
    assert reordered.reference_subject_summary_digest == (
        first.reference_subject_summary_digest
    )
    assert reordered.reference_transform_id == first.reference_transform_id
    assert reordered.downstream_functional_id == first.downstream_functional_id


def test_reference_transform_is_subject_equal_under_technical_row_duplication() -> None:
    common = {
        "receiver": "Receiver",
        "contrast_name": "stim_vs_ctrl",
        "fold_id": "fold-1",
        "feature_ids": ("G1",),
        "family_ids": ("F1",),
        "training_subject_ids": ("p1", "p2", "p3"),
        "target_weight_matrix": np.ones((1, 1)),
        "family_support": np.ones(1),
        "minimum_scale": 0.25,
    }
    expression = np.asarray([[0.0], [2.0], [4.0], [2.0], [6.0], [4.0], [8.0]])
    first = fit_downstream_functional(
        expression,
        reference_sample_ids=("p1-a1", "p1-a2", "p1-b", "p2-a", "p2-b", "p3-a", "p3-b"),
        reference_subject_ids=("p1", "p1", "p1", "p2", "p2", "p3", "p3"),
        reference_context_ids=("a", "a", "b", "a", "b", "a", "b"),
        **common,
    )
    duplicated = fit_downstream_functional(
        np.vstack([expression, [[0.0], [2.0]]]),
        reference_sample_ids=(
            "p1-a1",
            "p1-a2",
            "p1-b",
            "p2-a",
            "p2-b",
            "p3-a",
            "p3-b",
            "p1-a3",
            "p1-a4",
        ),
        reference_subject_ids=(
            "p1",
            "p1",
            "p1",
            "p2",
            "p2",
            "p3",
            "p3",
            "p1",
            "p1",
        ),
        reference_context_ids=("a", "a", "b", "a", "b", "a", "b", "a", "a"),
        **common,
    )

    np.testing.assert_allclose(first.feature_center, [4.0])
    np.testing.assert_allclose(duplicated.feature_center, first.feature_center)
    np.testing.assert_allclose(duplicated.feature_scale, first.feature_scale)
    assert duplicated.reference_subject_summary_digest == (
        first.reference_subject_summary_digest
    )
    assert duplicated.reference_input_digest != first.reference_input_digest
    assert duplicated.reference_row_manifest_id != first.reference_row_manifest_id
    assert duplicated.downstream_functional_id != first.downstream_functional_id


def test_missing_target_feature_propagates_only_to_affected_family() -> None:
    functional = _functional()
    result = apply_downstream_functional(
        functional,
        np.asarray([np.nan, 7.9652]),
        feature_ids=("G1", "G2"),
    )

    assert np.isnan(result.receiver_program_score[0, 0])
    assert result.receiver_program_score[0, 1] == pytest.approx(0.5)


def _incremental_functional(
    *,
    family_basis: np.ndarray | None = None,
    family_ids: tuple[str, ...] = ("LR_family",),
    precision_weights: np.ndarray | None = None,
):
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
        family_ids=family_ids,
        nuisance_column_ids=("intercept",),
        training_subject_ids=tuple(sorted(set(manifest.subject_ids))),
        family_basis=(
            np.asarray([[1.0], [0.0]]) if family_basis is None else family_basis
        ),
        precision_weights=(
            np.ones(2) if precision_weights is None else precision_weights
        ),
        minimum_scale=0.25,
        null_loss_floor=1e-8,
    )


def _factorized_nuisance_case(*, with_autonomous: bool):
    n_subjects = 6
    regressor = np.asarray([-1.0] * n_subjects + [1.0] * n_subjects)
    subject_covariate = np.linspace(-1.2, 1.1, n_subjects)
    shifted_covariate = np.concatenate(
        [subject_covariate - 0.2, subject_covariate + 0.6]
    )
    nuisance = np.column_stack(
        [
            np.ones(2 * n_subjects),
            shifted_covariate,
            np.square(shifted_covariate),
        ]
    )
    family_basis = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.0, 0.7],
        ]
    )
    nuisance_effects = np.asarray(
        [
            [1.2, 0.8, 1.5, 0.7, 1.1, 0.9],
            [0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
            [0.1, 0.05, -0.08, 0.03, 0.07, -0.04],
        ]
    )
    autonomous_basis = np.asarray([1.0, 0.5, 0.2, 0.0, 0.0, 0.0])
    context_effect = family_basis @ np.asarray([0.8, 1.1, 0.7])
    if with_autonomous:
        context_effect = context_effect + 0.9 * autonomous_basis
    row_noise = np.linspace(-0.04, 0.04, 2 * n_subjects)[:, np.newaxis]
    feature_noise = np.asarray([1.0, -0.5, 0.25, -0.2, 0.4, -0.3])
    response = (
        nuisance @ nuisance_effects
        + np.outer(regressor, context_effect)
        + row_noise * feature_noise
    )
    context_ids = tuple("reference" if value < 0 else "target" for value in regressor)
    manifest = DownstreamRowManifest(
        sample_ids=tuple(
            f"factorized-train-{index:02d}" for index in range(len(regressor))
        ),
        subject_ids=tuple(
            f"factorized-subject-{index % n_subjects:02d}"
            for index in range(len(regressor))
        ),
        context_ids=context_ids,
    )
    feature_ids = tuple(f"GENE_{index}" for index in range(family_basis.shape[0]))
    autonomous_resource = (
        _autonomous_resource(
            autonomous_basis[:, np.newaxis],
            feature_ids=feature_ids,
        )
        if with_autonomous
        else None
    )
    functional = fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        reference_mask=regressor < 0,
        nuisance_matrix=nuisance,
        context_regressor=regressor,
        receiver="Receiver",
        contrast_name="target_vs_reference",
        fold_id="factorized-fold",
        context_regressor_id="factorized-regressor-v1",
        nuisance_design_id="factorized-nuisance-v1",
        feature_ids=feature_ids,
        family_ids=("FAMILY_0", "FAMILY_1", "FAMILY_2"),
        nuisance_column_ids=("intercept", "covariate", "covariate_squared"),
        training_subject_ids=tuple(sorted(set(manifest.subject_ids))),
        family_basis=family_basis,
        autonomous_program_resource=autonomous_resource,
        precision_weights=np.linspace(0.6, 1.4, family_basis.shape[0]),
        minimum_scale=0.05,
        lambda2=1e-6,
    )
    return functional, response, nuisance, regressor, feature_ids


@pytest.mark.parametrize("with_autonomous", [False, True])
def test_factorized_nuisance_predictions_match_direct_wls_oracle(
    with_autonomous: bool,
) -> None:
    functional, training_response, training_nuisance, training_regressor, features = (
        _factorized_nuisance_case(with_autonomous=with_autonomous)
    )
    projected_training = _project_feature_values(
        (training_response - functional.feature_center) / functional.feature_scale,
        autonomous_basis=functional.autonomous_basis,
        precision_weights=functional.precision_weights,
    )
    direct_null = np.linalg.lstsq(training_nuisance, projected_training, rcond=None)[0]
    direct_context = np.linalg.lstsq(training_nuisance, training_regressor, rcond=None)[
        0
    ]
    family_effects = (
        functional.family_basis.toarray()
        * functional.family_coefficients[np.newaxis, :]
    )
    total_effect = np.sum(family_effects, axis=1)
    direct_full = np.linalg.lstsq(
        training_nuisance,
        projected_training - np.outer(training_regressor, total_effect),
        rcond=None,
    )[0]
    direct_families = np.stack(
        [
            np.linalg.lstsq(
                training_nuisance,
                projected_training
                - np.outer(training_regressor, family_effects[:, family_index]),
                rcond=None,
            )[0]
            for family_index in range(len(functional.family_ids))
        ]
    )

    heldout_subjects = 4
    heldout_regressor = np.asarray([-1.0] * heldout_subjects + [1.0] * heldout_subjects)
    heldout_covariate = np.concatenate(
        [
            np.linspace(-0.9, 0.8, heldout_subjects) - 0.1,
            np.linspace(-0.9, 0.8, heldout_subjects) + 0.4,
        ]
    )
    heldout_nuisance = np.column_stack(
        [
            np.ones(2 * heldout_subjects),
            heldout_covariate,
            np.square(heldout_covariate),
        ]
    )
    heldout_response = heldout_nuisance @ np.asarray(
        [
            [1.0, 0.9, 1.3, 0.8, 1.0, 0.7],
            [0.2, -0.1, 0.05, 0.3, -0.05, 0.15],
            [0.08, 0.04, -0.05, 0.02, 0.06, -0.03],
        ]
    ) + np.outer(heldout_regressor, total_effect)
    heldout_manifest = DownstreamRowManifest(
        sample_ids=tuple(
            f"factorized-heldout-{index:02d}" for index in range(len(heldout_regressor))
        ),
        subject_ids=tuple(
            f"factorized-heldout-subject-{index % heldout_subjects:02d}"
            for index in range(len(heldout_regressor))
        ),
        context_ids=tuple(
            "reference" if value < 0 else "target" for value in heldout_regressor
        ),
    )

    direct_null_prediction = heldout_nuisance @ direct_null
    direct_full_prediction = heldout_nuisance @ direct_full + np.outer(
        heldout_regressor, total_effect
    )
    direct_family_predictions = np.stack(
        [
            heldout_nuisance @ direct_families[family_index]
            + np.outer(heldout_regressor, family_effects[:, family_index])
            for family_index in range(len(functional.family_ids))
        ]
    )
    residualized_context = (
        heldout_regressor
        - heldout_nuisance @ functional.context_regressor_nuisance_coefficients
    )
    factorized_null_prediction = (
        heldout_nuisance @ functional.null_nuisance_coefficients
    )
    factorized_full_prediction = factorized_null_prediction + np.outer(
        residualized_context, total_effect
    )
    factorized_family_predictions = np.stack(
        [
            factorized_null_prediction
            + np.outer(residualized_context, family_effects[:, family_index])
            for family_index in range(len(functional.family_ids))
        ]
    )

    np.testing.assert_allclose(
        functional.context_regressor_nuisance_coefficients,
        direct_context,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        factorized_null_prediction,
        direct_null_prediction,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        factorized_full_prediction,
        direct_full_prediction,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        factorized_family_predictions,
        direct_family_predictions,
        atol=1e-12,
        rtol=1e-12,
    )

    application = apply_incremental_downstream_functional(
        functional,
        heldout_response,
        row_manifest=heldout_manifest,
        design_sample_ids=heldout_manifest.sample_ids,
        nuisance_matrix=heldout_nuisance,
        context_regressor=heldout_regressor,
        context_regressor_id="factorized-regressor-v1",
        nuisance_design_id="factorized-nuisance-v1",
        feature_ids=features,
        nuisance_column_ids=("intercept", "covariate", "covariate_squared"),
    )
    projected_heldout = _project_feature_values(
        (heldout_response - functional.feature_center) / functional.feature_scale,
        autonomous_basis=functional.autonomous_basis,
        precision_weights=functional.precision_weights,
    )
    expected_null_losses = np.sum(
        np.square(projected_heldout - direct_null_prediction)
        * functional.precision_weights,
        axis=1,
    )
    expected_full_losses = np.sum(
        np.square(projected_heldout - direct_full_prediction)
        * functional.precision_weights,
        axis=1,
    )
    expected_family_losses = np.column_stack(
        [
            np.sum(
                np.square(projected_heldout - direct_family_prediction)
                * functional.precision_weights,
                axis=1,
            )
            for direct_family_prediction in direct_family_predictions
        ]
    )
    np.testing.assert_allclose(
        application.sample_null_losses, expected_null_losses, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        application.sample_full_losses, expected_full_losses, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        application.sample_family_losses,
        expected_family_losses,
        atol=1e-12,
        rtol=1e-12,
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


def test_no_autonomous_fit_marks_zero_precision_family_not_estimable() -> None:
    functional = _incremental_functional(
        family_basis=np.eye(2),
        family_ids=("supported", "unsupported"),
        precision_weights=np.asarray([1.0, 0.0]),
    )

    assert functional.family_estimable.tolist() == [True, False]
    np.testing.assert_allclose(functional.family_retained_norm_fraction, [1.0, 0.0])
    assert functional.family_reason_codes == (
        None,
        "family_basis_not_identifiable_after_autonomous_projection",
    )
    np.testing.assert_array_equal(functional.family_basis.getcol(1).toarray(), 0.0)


def test_no_autonomous_fit_fails_closed_when_all_families_lack_precision() -> None:
    with pytest.raises(ContractError) as error:
        _incremental_functional(
            family_basis=np.asarray([[0.0], [1.0]]),
            precision_weights=np.asarray([1.0, 0.0]),
        )

    assert error.value.details.code == (
        "all_family_bases_not_identifiable_after_autonomous_projection"
    )


def _projected_overlap_functional(
    *,
    active_unique_effect: float = 0.0,
    penalty_candidate: RelativePenaltyCandidate | None = None,
):
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
        penalty_candidate=penalty_candidate,
    )


def test_relative_candidate_is_resolved_inside_signed_residual_training() -> None:
    candidate = RelativePenaltyCandidate(
        lambda1_fraction=0.1,
        lambda2_fraction=0.2,
    )

    functional = _projected_overlap_functional(
        active_unique_effect=1.5,
        penalty_candidate=candidate,
    )

    assert functional.penalty_candidate_id == candidate.candidate_id
    assert functional.penalty_scale_resolution_id is not None
    assert functional.resolved_penalty_id is not None
    assert functional.lambda1 >= 0
    assert functional.lambda2 > 0
    functional.to_dict()


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


_COORDINATE_CONTRACT_FEATURES = ("AUTO_SLOW", "AUTO_FAST", "UNIQUE")


def _coordinate_contract_fit(
    *,
    autonomous_effect: float,
    unique_effect: float,
    feature_units: np.ndarray | None = None,
    family_basis_multiplier: float = 1.0,
):
    units = (
        np.ones(3, dtype=np.float64)
        if feature_units is None
        else np.asarray(feature_units, dtype=np.float64)
    )
    reference = np.asarray(
        [
            [-3.0, -0.3, -1.5],
            [-1.0, -0.1, -0.5],
            [1.0, 0.1, 0.5],
            [3.0, 0.3, 1.5],
        ]
    )
    autonomous_basis = np.asarray([1.0, 1.0, 0.0])
    family_basis = np.asarray([0.0, 1.0, 1.0])
    context_effect = autonomous_effect * autonomous_basis + unique_effect * family_basis
    response = np.vstack([reference, reference + 2.0 * context_effect]) * units
    regressor = np.asarray([-1.0] * 4 + [1.0] * 4)
    manifest = _paired_manifest(regressor, prefix="coordinate-contract-training")
    functional = fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        reference_mask=regressor < 0,
        nuisance_matrix=np.ones((8, 1)),
        context_regressor=regressor,
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        fold_id="fold-coordinate-contract",
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=_COORDINATE_CONTRACT_FEATURES,
        family_ids=("LR_family",),
        nuisance_column_ids=("intercept",),
        training_subject_ids=tuple(sorted(set(manifest.subject_ids))),
        family_basis=(family_basis * units * family_basis_multiplier)[:, np.newaxis],
        autonomous_program_resource=_autonomous_resource(
            (autonomous_basis * units)[:, np.newaxis],
            feature_ids=_COORDINATE_CONTRACT_FEATURES,
        ),
        precision_weights=np.ones(3),
        minimum_scale=0.01,
    )
    return functional


def _apply_coordinate_contract(
    functional,
    *,
    autonomous_effect: float,
    unique_effect: float,
    feature_units: np.ndarray | None = None,
):
    units = (
        np.ones(3, dtype=np.float64)
        if feature_units is None
        else np.asarray(feature_units, dtype=np.float64)
    )
    reference = np.asarray([[-0.8, -0.08, -0.4], [0.8, 0.08, 0.4]])
    autonomous_basis = np.asarray([1.0, 1.0, 0.0])
    family_basis = np.asarray([0.0, 1.0, 1.0])
    context_effect = autonomous_effect * autonomous_basis + unique_effect * family_basis
    response = np.vstack([reference, reference + 2.0 * context_effect]) * units
    regressor = np.asarray([-1.0, -1.0, 1.0, 1.0])
    manifest = _paired_manifest(regressor, prefix="coordinate-contract-heldout")
    return apply_incremental_downstream_functional(
        functional,
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((4, 1)),
        context_regressor=regressor,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=_COORDINATE_CONTRACT_FEATURES,
        nuisance_column_ids=("intercept",),
    )


@pytest.mark.parametrize("autonomous_effect", [0.0, 2.0])
def test_frozen_coordinate_projection_rejects_global_and_autonomous_nulls(
    autonomous_effect: float,
) -> None:
    functional = _coordinate_contract_fit(
        autonomous_effect=autonomous_effect,
        unique_effect=0.0,
    )
    result = _apply_coordinate_contract(
        functional,
        autonomous_effect=autonomous_effect,
        unique_effect=0.0,
    )

    assert functional.feature_scale[0] > 5.0 * functional.feature_scale[1]
    assert functional.basis_coordinate_transform == (
        "raw_family_basis_l2_normalize_then_divide_by_frozen_reference_mad_scale_v1"
    )
    assert functional.family_coefficients[0] == pytest.approx(0.0, abs=1e-12)
    assert result.model_gain == pytest.approx(0.0, abs=1e-12)
    assert result.family_gains[0] == pytest.approx(0.0, abs=1e-12)
    provenance = functional.to_dict()
    assert provenance["basis_coordinate_transform_id"] == (
        functional.basis_coordinate_transform_id
    )


def test_unique_lr_effect_survives_frozen_coordinate_projection() -> None:
    functional = _coordinate_contract_fit(
        autonomous_effect=2.0,
        unique_effect=1.0,
    )
    result = _apply_coordinate_contract(
        functional,
        autonomous_effect=2.0,
        unique_effect=1.0,
    )

    assert functional.family_coefficients[0] > 0.0
    assert result.model_gain is not None and result.model_gain > 0.99
    assert result.family_gains[0] > 0.99


def test_basis_coordinate_transform_is_invariant_to_feature_units() -> None:
    feature_units = np.asarray([4.0, 0.5, 2.0])
    base = _coordinate_contract_fit(autonomous_effect=2.0, unique_effect=1.0)
    rescaled = _coordinate_contract_fit(
        autonomous_effect=2.0,
        unique_effect=1.0,
        feature_units=feature_units,
    )
    base_result = _apply_coordinate_contract(
        base,
        autonomous_effect=2.0,
        unique_effect=1.0,
    )
    rescaled_result = _apply_coordinate_contract(
        rescaled,
        autonomous_effect=2.0,
        unique_effect=1.0,
        feature_units=feature_units,
    )

    np.testing.assert_allclose(
        base.family_basis.toarray(), rescaled.family_basis.toarray(), atol=1e-12
    )
    np.testing.assert_allclose(
        base.family_coefficients, rescaled.family_coefficients, atol=1e-12
    )
    np.testing.assert_allclose(
        base_result.family_gains, rescaled_result.family_gains, atol=1e-12
    )
    assert base_result.model_gain == pytest.approx(
        rescaled_result.model_gain, abs=1e-12
    )


def test_raw_family_basis_lineage_precedes_internal_normalization() -> None:
    functional = _coordinate_contract_fit(
        autonomous_effect=2.0,
        unique_effect=1.0,
        family_basis_multiplier=7.0,
    )
    expected_raw_id = _matrix_digest(
        sparse.csc_matrix(np.asarray([[0.0], [7.0], [7.0]]))
    )

    assert functional.raw_input_family_basis_id == expected_raw_id
    assert functional.original_family_basis_id == expected_raw_id
    assert functional.coordinate_family_basis_id != expected_raw_id
    provenance = functional.to_dict()
    assert provenance["raw_input_family_basis_id"] == expected_raw_id
    assert provenance["coordinate_family_basis_id"] == (
        functional.coordinate_family_basis_id
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("basis_coordinate_transform", "unfrozen_raw_basis_v0"),
        ("basis_coordinate_transform_id", "tampered-coordinate-transform"),
    ],
)
def test_incremental_functional_binds_basis_coordinate_provenance(
    field_name: str,
    replacement: str,
) -> None:
    functional = _coordinate_contract_fit(
        autonomous_effect=2.0,
        unique_effect=1.0,
    )
    object.__setattr__(functional, field_name, replacement)

    with pytest.raises(ContractError) as error:
        functional.to_dict()

    assert error.value.details.code == "incremental_functional_integrity_violation"


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


def test_incremental_functional_detects_mutated_context_nuisance_factor() -> None:
    functional = _incremental_functional()
    object.__setattr__(
        functional,
        "context_regressor_nuisance_coefficients",
        functional.context_regressor_nuisance_coefficients.copy(),
    )
    functional.context_regressor_nuisance_coefficients[0] += 1.0

    with pytest.raises(ContractError) as error:
        functional.to_dict()

    assert error.value.details.code == "incremental_functional_integrity_violation"


def test_incremental_functional_detects_mutated_nuisance_factorization() -> None:
    functional = _incremental_functional()
    object.__setattr__(
        functional,
        "nuisance_factorization",
        "tampered_nuisance_factorization",
    )

    with pytest.raises(ContractError) as error:
        functional.to_dict()

    assert error.value.details.code == "incremental_functional_integrity_violation"


def test_incremental_functional_does_not_store_dense_family_nuisance_models() -> None:
    functional = _incremental_functional()
    slots = set(IncrementalDownstreamFunctional.__slots__)
    provenance = functional.to_dict()

    assert "full_nuisance_coefficients" not in slots
    assert "family_nuisance_coefficients" not in slots
    assert "full_nuisance_digest" not in provenance
    assert "family_nuisance_digest" not in provenance
    assert functional.context_regressor_nuisance_coefficients.shape == (1,)
    assert functional.nuisance_factorization == (
        "null_plus_residualized_context_outer_family_effect_v1"
    )


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
