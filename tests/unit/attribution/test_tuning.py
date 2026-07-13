from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from crychic.attribution import (
    PenaltyCandidateSummary,
    PenaltyFoldEvaluation,
    PenaltyScaleResolution,
    PenaltyTuningArtifact,
    PenaltyTuningSpec,
    RelativePenaltyCandidate,
    ResolvedPenalty,
    record_penalty_fold_evaluation,
    resolve_penalty_scale,
    resolve_residualized_penalty_scale,
    select_penalty_candidate,
)
from crychic.core import ContractError


def _evaluations(
    spec: PenaltyTuningSpec,
    losses: dict[tuple[float, float], tuple[float, float, float, float]],
    *,
    unavailable: tuple[float, float] | None = None,
) -> tuple[PenaltyFoldEvaluation, ...]:
    results: list[PenaltyFoldEvaluation] = []
    for candidate in spec.candidates:
        key = (candidate.lambda1_fraction, candidate.lambda2_fraction)
        if key == unavailable:
            for fold_id, subjects in (
                ("inner-0", ("s1", "s2")),
                ("inner-1", ("s3", "s4")),
            ):
                results.append(
                    record_penalty_fold_evaluation(
                        candidate,
                        inner_fold_id=fold_id,
                        validation_subject_ids=subjects,
                        status="failed",
                        reason_code="solver_not_converged",
                    )
                )
            continue
        values = losses[key]
        results.extend(
            (
                record_penalty_fold_evaluation(
                    candidate,
                    inner_fold_id="inner-0",
                    validation_subject_ids=("s1", "s2"),
                    subject_losses=np.asarray(values[:2]),
                ),
                record_penalty_fold_evaluation(
                    candidate,
                    inner_fold_id="inner-1",
                    validation_subject_ids=("s3", "s4"),
                    subject_losses=np.asarray(values[2:]),
                ),
            )
        )
    return tuple(results)


def _select(
    spec: PenaltyTuningSpec,
    evaluations: tuple[PenaltyFoldEvaluation, ...],
) -> PenaltyTuningArtifact:
    return select_penalty_candidate(
        spec,
        evaluations,
        tuning_scope_id="outer-fold:receiver:contrast",
        training_subject_ids=("s1", "s2", "s3", "s4"),
        inner_fold_ids=("inner-0", "inner-1"),
    )


def test_relative_grid_is_canonical_cartesian_and_rejects_boolean() -> None:
    first = PenaltyTuningSpec(
        lambda1_fractions=(0.1, 1.0, 0.1),
        lambda2_fractions=(0.0, 1.0),
    )
    second = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.1),
        lambda2_fractions=(1.0, 0.0),
    )

    assert first.spec_id == second.spec_id
    assert tuple(
        (candidate.lambda1_fraction, candidate.lambda2_fraction)
        for candidate in first.candidates
    ) == ((1.0, 1.0), (1.0, 0.0), (0.1, 1.0), (0.1, 0.0))
    with pytest.raises(ValueError, match="not boolean"):
        RelativePenaltyCandidate(lambda1_fraction=True, lambda2_fraction=0.0)


def test_penalty_identity_canonicalizes_signed_zero() -> None:
    positive = RelativePenaltyCandidate(
        lambda1_fraction=0.0,
        lambda2_fraction=0.0,
    )
    negative = RelativePenaltyCandidate(
        lambda1_fraction=-0.0,
        lambda2_fraction=-0.0,
    )
    first_spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0, -0.0),
        lambda2_fractions=(0.0,),
    )
    second_spec = PenaltyTuningSpec(
        lambda1_fractions=(0.0, 1.0),
        lambda2_fractions=(-0.0,),
    )

    assert negative.to_dict() == positive.to_dict()
    assert first_spec.spec_id == second_spec.spec_id


def test_penalty_scale_identity_canonicalizes_sparse_and_dense_zero_storage() -> None:
    implicit = sparse.csc_matrix(np.eye(2))
    explicit = sparse.csc_matrix(
        (
            np.asarray([1.0, -0.0, 1.0]),
            np.asarray([0, 1, 1]),
            np.asarray([0, 2, 3]),
        ),
        shape=(2, 2),
    )
    first = resolve_residualized_penalty_scale(
        implicit,
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]),
        feature_ids=("g1", "g2"),
        family_ids=("f1", "f2"),
    )
    second = resolve_residualized_penalty_scale(
        explicit,
        np.asarray([-0.0, 1.0]),
        np.asarray([1.0, -0.0]),
        feature_ids=("g1", "g2"),
        family_ids=("f1", "f2"),
    )

    assert first.basis_digest == second.basis_digest
    assert first.response_digest == second.response_digest
    assert first.precision_digest == second.precision_digest
    assert first.scale_resolution_id == second.scale_resolution_id


def test_penalty_scale_matches_hand_calculation_and_resolves_candidate() -> None:
    basis = sparse.csc_matrix(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [1.0, 1.0],
            ]
        )
    )
    response = np.asarray([2.0, 1.0, 3.0])
    precision = np.asarray([1.0, 2.0, 0.5])

    scale = resolve_penalty_scale(
        basis,
        response,
        precision,
        feature_ids=("g1", "g2", "g3"),
        family_ids=("f1", "f2"),
    )
    candidate = RelativePenaltyCandidate(
        lambda1_fraction=0.3,
        lambda2_fraction=0.1,
    )
    resolved = scale.resolve(candidate)

    # 2 * max(B.T @ (Q*y)) = 2 * max(3.5, 5.5) = 11.
    assert scale.lambda1_max == pytest.approx(11.0, abs=1e-12, rel=0.0)
    # diag(B.T Q B) = (1.5, 8.5); median = 5.
    assert scale.lambda2_scale == pytest.approx(5.0, abs=1e-12, rel=0.0)
    assert resolved.lambda1 == pytest.approx(3.3, abs=1e-12, rel=0.0)
    assert resolved.lambda2 == pytest.approx(0.5, abs=1e-12, rel=0.0)
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        scale.weighted_correlations.setflags(write=True)


def test_zero_response_keeps_estimable_zero_lambda1_max() -> None:
    scale = resolve_penalty_scale(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        feature_ids=("g1", "g2"),
        family_ids=("f1", "f2"),
    )

    assert scale.estimable
    assert scale.reason_code is None
    assert scale.lambda1_max == 0.0
    assert scale.lambda2_scale == 1.0


def test_signed_residual_scale_matches_nonnegative_coefficient_kkt() -> None:
    basis = np.asarray(
        [
            [1.0, -1.0],
            [-2.0, 1.0],
            [0.5, 0.5],
        ]
    )
    response = np.asarray([2.0, -1.0, -3.0])
    precision = np.asarray([1.0, 2.0, 0.5])

    with pytest.raises(ValueError, match="non-negative"):
        resolve_penalty_scale(
            basis,
            response,
            precision,
            feature_ids=("g1", "g2", "g3"),
            family_ids=("f1", "f2"),
        )

    scale = resolve_residualized_penalty_scale(
        basis,
        response,
        precision,
        feature_ids=("g1", "g2", "g3"),
        family_ids=("f1", "f2"),
    )

    np.testing.assert_allclose(scale.weighted_correlations, [5.25, -4.75])
    assert scale.lambda1_max == pytest.approx(10.5, abs=1e-12, rel=0.0)
    assert scale.lambda2_scale == pytest.approx(6.125, abs=1e-12, rel=0.0)
    assert scale.problem_space == "signed_residual_nonnegative_coefficients_v1"


def test_scale_identity_distinguishes_directional_and_residual_contracts() -> None:
    directional = resolve_penalty_scale(
        np.eye(2),
        np.ones(2),
        np.ones(2),
        feature_ids=("g1", "g2"),
        family_ids=("f1", "f2"),
    )
    residual = resolve_residualized_penalty_scale(
        np.eye(2),
        np.ones(2),
        np.ones(2),
        feature_ids=("g1", "g2"),
        family_ids=("f1", "f2"),
    )

    assert directional.scale_resolution_id != residual.scale_resolution_id
    assert directional.problem_space == "direction_compatible_nonnegative_v1"


def test_scale_records_no_positive_weighted_gram_as_not_estimable() -> None:
    scale = resolve_penalty_scale(
        np.zeros((2, 1)),
        np.ones(2),
        np.ones(2),
        feature_ids=("g1", "g2"),
        family_ids=("f1",),
    )

    assert not scale.estimable
    assert scale.reason_code == "no_positive_weighted_gram_scale"
    with pytest.raises(ContractError) as error:
        scale.resolve(
            RelativePenaltyCandidate(lambda1_fraction=1.0, lambda2_fraction=0.0)
        )
    assert error.value.details.code == "penalty_scale_not_estimable"


def test_one_se_selects_strongest_regularization_subject_equally() -> None:
    spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.1),
        lambda2_fractions=(1.0, 0.0),
    )
    losses = {
        # Mean 1.05: within one SE of the best and strongest on both axes.
        (1.0, 1.0): (0.9, 1.0, 1.1, 1.2),
        (1.0, 0.0): (1.0, 1.0, 1.1, 1.1),
        # Unique best mean 1.00, SE ~= 0.11547.
        (0.1, 1.0): (0.8, 0.8, 1.2, 1.2),
        (0.1, 0.0): (1.3, 1.3, 1.3, 1.3),
    }

    tuning = _select(spec, _evaluations(spec, losses))

    assert tuning.status == "selected"
    assert tuning.reason_code is None
    assert tuning.best_mean_loss == pytest.approx(1.0, abs=1e-12, rel=0.0)
    assert tuning.one_se_threshold == pytest.approx(
        1.1154700538379252, abs=1e-12, rel=0.0
    )
    assert tuning.selected_candidate is not None
    assert tuning.certification_status == (
        "caller_recorded_inner_losses_selection_only"
    )
    assert tuning.is_oof_certified is False
    assert all(
        evaluation.verification_status
        == "caller_recorded_fold_losses_unverified"
        for evaluation in tuning.evaluations
    )
    assert (
        tuning.selected_candidate.lambda1_fraction,
        tuning.selected_candidate.lambda2_fraction,
    ) == (1.0, 1.0)
    manifest = tuning.to_dict()
    assert len(manifest["evaluations"]) == len(spec.candidates) * 2


def test_failed_candidate_is_excluded_but_retained_in_complete_grid() -> None:
    spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.1),
        lambda2_fractions=(0.0,),
    )
    losses = {
        (1.0, 0.0): (1.0, 1.0, 1.0, 1.0),
        (0.1, 0.0): (0.1, 0.1, 0.1, 0.1),
    }

    tuning = _select(
        spec,
        _evaluations(spec, losses, unavailable=(0.1, 0.0)),
    )

    assert tuning.status == "selected"
    assert tuning.selected_candidate is not None
    assert tuning.selected_candidate.lambda1_fraction == 1.0
    assert len(tuning.evaluations) == len(spec.candidates) * 2
    failed = [summary for summary in tuning.summaries if summary.status == "failed"]
    assert len(failed) == 1
    assert failed[0].mean_loss is None


def test_all_unavailable_candidates_preserve_failed_status_without_fallback() -> None:
    spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0,),
        lambda2_fractions=(0.0,),
    )
    evaluations = _evaluations(
        spec,
        {(1.0, 0.0): (1.0, 1.0, 1.0, 1.0)},
        unavailable=(1.0, 0.0),
    )

    tuning = _select(spec, evaluations)

    assert tuning.status == "failed"
    assert tuning.reason_code == "no_candidate_complete_inner_coverage"
    assert tuning.selected_candidate is None
    assert tuning.selected_candidate_id is None


def test_evaluation_and_selection_are_order_stable() -> None:
    spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.1),
        lambda2_fractions=(0.0,),
    )
    losses = {
        (1.0, 0.0): (1.0, 1.2, 1.1, 1.3),
        (0.1, 0.0): (0.9, 1.0, 1.1, 1.2),
    }
    original = _evaluations(spec, losses)
    candidate = spec.candidates[0]
    reordered_evaluation = record_penalty_fold_evaluation(
        candidate,
        inner_fold_id="inner-0",
        validation_subject_ids=("s2", "s1"),
        subject_losses=np.asarray([1.2, 1.0]),
    )
    assert reordered_evaluation.evaluation_id == original[0].evaluation_id

    first = _select(spec, original)
    second = select_penalty_candidate(
        spec,
        tuple(reversed(original)),
        tuning_scope_id="outer-fold:receiver:contrast",
        training_subject_ids=("s4", "s2", "s1", "s3"),
        inner_fold_ids=("inner-1", "inner-0"),
    )
    assert first.tuning_id == second.tuning_id


def test_selection_rejects_missing_candidate_fold_and_partition_drift() -> None:
    spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.1),
        lambda2_fractions=(0.0,),
    )
    losses = {
        (1.0, 0.0): (1.0, 1.0, 1.0, 1.0),
        (0.1, 0.0): (1.0, 1.0, 1.0, 1.0),
    }
    evaluations = _evaluations(spec, losses)
    with pytest.raises(ValueError, match="complete candidate x fold grid"):
        _select(spec, evaluations[:-1])

    drifted = list(evaluations)
    drifted[2] = record_penalty_fold_evaluation(
        spec.candidates[1],
        inner_fold_id="inner-0",
        validation_subject_ids=("s1", "s3"),
        subject_losses=np.asarray([1.0, 1.0]),
    )
    with pytest.raises(ValueError, match="identical validation partitions"):
        _select(spec, tuple(drifted))


def test_producer_owned_arrays_and_tamper_detection() -> None:
    spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0,),
        lambda2_fractions=(0.0,),
    )
    tuning = _select(
        spec,
        _evaluations(spec, {(1.0, 0.0): (1.0, 1.0, 1.0, 1.0)}),
    )
    evaluation = tuning.evaluations[0]

    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        evaluation.subject_losses.setflags(write=True)
    with pytest.raises(TypeError, match="producer-owned"):
        PenaltyFoldEvaluation()
    with pytest.raises(TypeError, match="producer-owned"):
        PenaltyScaleResolution()
    with pytest.raises(TypeError, match="producer-owned"):
        PenaltyTuningArtifact()
    with pytest.raises(TypeError, match="producer-owned"):
        PenaltyCandidateSummary()
    with pytest.raises(TypeError, match="producer-owned"):
        ResolvedPenalty()

    object.__setattr__(evaluation, "subject_losses", evaluation.subject_losses.copy())
    evaluation.subject_losses[0] = 99.0
    with pytest.raises(ContractError) as error:
        tuning.to_dict()
    assert error.value.details.code == "penalty_tuning_integrity_violation"


def test_summary_and_resolved_penalty_detect_scalar_poison() -> None:
    scale = resolve_penalty_scale(
        np.eye(2),
        np.ones(2),
        np.ones(2),
        feature_ids=("g1", "g2"),
        family_ids=("f1", "f2"),
    )
    resolved = scale.resolve(
        RelativePenaltyCandidate(lambda1_fraction=0.3, lambda2_fraction=0.1)
    )
    object.__setattr__(resolved, "lambda1", resolved.lambda1 + 1.0)
    with pytest.raises(ContractError) as error:
        resolved.to_dict()
    assert error.value.details.code == "resolved_penalty_integrity_violation"

    spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0,),
        lambda2_fractions=(0.0,),
    )
    tuning = _select(
        spec,
        _evaluations(spec, {(1.0, 0.0): (1.0, 1.0, 1.0, 1.0)}),
    )
    object.__setattr__(tuning.summaries[0], "mean_loss", 99.0)
    with pytest.raises(ContractError) as error:
        tuning.to_dict()
    assert error.value.details.code == "penalty_tuning_integrity_violation"
