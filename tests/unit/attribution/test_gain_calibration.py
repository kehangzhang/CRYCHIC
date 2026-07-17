from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from crychic.attribution import (
    GainCalibrationSpec,
    PenaltyTuningArtifact,
    PenaltyTuningSpec,
    PenaltyValidationLossEstimand,
    RelativePenaltyCandidate,
    SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact,
    select_penalty_candidate,
)
from crychic.attribution.gain_calibration import (
    _WORKFLOW_GAIN_CALIBRATION_PRODUCER_TOKEN,
    _finalize_selected_penalty_inner_oof_gain_calibration,
)
from crychic.attribution.tuning import (
    _WORKFLOW_SUBJECT_BLOCKED_PRODUCER_TOKEN,
    _record_subject_blocked_penalty_fold_evaluation,
)
from crychic.core import ContractError
from crychic.scoring.downstream import (
    DownstreamRowManifest,
    IncrementalDownstreamApplication,
    IncrementalDownstreamFunctional,
    _fit_incremental_downstream_functional,
)

_FAMILY_IDS = ("family-1", "family-2", "family-3")
_FEATURE_IDS = ("G1", "G2", "G3")
_SUBJECT_IDS = tuple(f"s{index}" for index in range(1, 9))


def _manifest(subjects: Sequence[str], *, prefix: str) -> DownstreamRowManifest:
    normalized = tuple(subjects)
    return DownstreamRowManifest(
        sample_ids=tuple(
            f"{prefix}-{subject}-{context}"
            for context in ("reference", "target")
            for subject in normalized
        ),
        subject_ids=tuple(subject for _ in range(2) for subject in normalized),
        context_ids=tuple(
            context for context in ("reference", "target") for _ in normalized
        ),
    )


def _fit_functional(
    subjects: Sequence[str],
    *,
    fold_id: str,
    candidate: RelativePenaltyCandidate,
) -> IncrementalDownstreamFunctional:
    normalized = tuple(sorted(subjects))
    manifest = _manifest(normalized, prefix=f"training-{fold_id}")
    subject_position = {subject: _SUBJECT_IDS.index(subject) for subject in normalized}
    reference_rows = np.asarray(
        [
            [
                0.05 * subject_position[subject],
                1.0 + 0.07 * subject_position[subject],
                2.0 - 0.04 * subject_position[subject],
            ]
            for subject in normalized
        ],
        dtype=np.float64,
    )
    target_rows = reference_rows + np.asarray([1.2, 0.8, 0.4])
    response = np.vstack((reference_rows, target_rows))
    n_subjects = len(normalized)
    return _fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        reference_mask=np.asarray([True] * n_subjects + [False] * n_subjects),
        nuisance_matrix=np.ones((2 * n_subjects, 1), dtype=np.float64),
        context_regressor=np.asarray([-1.0] * n_subjects + [1.0] * n_subjects),
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        fold_id=fold_id,
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=_FEATURE_IDS,
        family_ids=_FAMILY_IDS,
        nuisance_column_ids=("intercept",),
        training_subject_ids=normalized,
        family_basis=np.eye(len(_FAMILY_IDS)),
        precision_weights=np.ones(len(_FEATURE_IDS)),
        minimum_scale=0.25,
        null_loss_floor=1e-8,
        penalty_candidate=candidate,
    )


def _application(
    functional: IncrementalDownstreamFunctional,
    subjects: Sequence[str],
    gains: np.ndarray,
    *,
    prefix: str,
) -> IncrementalDownstreamApplication:
    normalized = tuple(sorted(subjects))
    observed_gains = np.asarray(gains, dtype=np.float64)
    assert observed_gains.shape == (len(normalized), len(_FAMILY_IDS))
    manifest = _manifest(normalized, prefix=prefix)
    subject_null_losses: np.ndarray = np.ones(len(normalized), dtype=np.float64)
    subject_family_losses = 1.0 - observed_gains
    subject_full_losses = np.mean(subject_family_losses, axis=1)
    null_loss = float(np.mean(subject_null_losses))
    full_loss = float(np.mean(subject_full_losses))
    family_losses = np.mean(subject_family_losses, axis=0)
    raw_model_gain = (null_loss - full_loss) / null_loss
    raw_family_gains = (null_loss - family_losses) / null_loss
    subject_index = {subject: index for index, subject in enumerate(normalized)}
    sample_null_losses = np.asarray(
        [subject_null_losses[subject_index[item]] for item in manifest.subject_ids]
    )
    sample_full_losses = np.asarray(
        [subject_full_losses[subject_index[item]] for item in manifest.subject_ids]
    )
    sample_family_losses = np.vstack(
        [subject_family_losses[subject_index[item]] for item in manifest.subject_ids]
    )
    return IncrementalDownstreamApplication._from_application(
        functional=functional,
        status="observed",
        reason_code=None,
        sample_ids=manifest.sample_ids,
        sample_subject_ids=manifest.subject_ids,
        sample_context_ids=manifest.context_ids,
        heldout_row_manifest_id=manifest.manifest_id,
        heldout_input_digest=f"heldout-input-{prefix}",
        null_loss=null_loss,
        full_loss=full_loss,
        model_gain=float(np.clip(raw_model_gain, 0.0, 1.0)),
        raw_model_gain=raw_model_gain,
        family_losses=family_losses,
        family_gains=np.clip(raw_family_gains, 0.0, 1.0),
        raw_family_gains=raw_family_gains,
        sample_null_losses=sample_null_losses,
        sample_full_losses=sample_full_losses,
        sample_family_losses=sample_family_losses,
        subject_ids=normalized,
        subject_null_losses=subject_null_losses,
        subject_full_losses=subject_full_losses,
        subject_family_losses=subject_family_losses,
    )


def _world() -> tuple[
    PenaltyTuningArtifact,
    tuple[IncrementalDownstreamFunctional, ...],
    tuple[IncrementalDownstreamApplication, ...],
    IncrementalDownstreamFunctional,
    np.ndarray,
]:
    tuning_spec = PenaltyTuningSpec(
        lambda1_fractions=(0.0,),
        lambda2_fractions=(0.0,),
        inner_allowed_n_splits=(2,),
    )
    candidate = tuning_spec.candidates[0]
    validation_scopes = (_SUBJECT_IDS[:4], _SUBJECT_IDS[4:])
    gain_parts = (
        np.asarray(
            [
                [0.0, 0.10, 0.20],
                [0.15, 0.25, 0.35],
                [0.30, 0.40, 0.50],
                [0.45, 0.55, 0.65],
            ]
        ),
        np.asarray(
            [
                [0.05, 0.12, 0.22],
                [0.18, 0.28, 0.38],
                [0.32, 0.42, 0.52],
                [0.48, 0.58, 0.68],
            ]
        ),
    )
    functionals: list[IncrementalDownstreamFunctional] = []
    applications: list[IncrementalDownstreamApplication] = []
    evaluations = []
    for fold_index, (validation_subjects, gains) in enumerate(
        zip(validation_scopes, gain_parts, strict=True), start=1
    ):
        fold_id = f"inner-{fold_index}"
        training_subjects = tuple(
            subject for subject in _SUBJECT_IDS if subject not in validation_subjects
        )
        functional = _fit_functional(
            training_subjects,
            fold_id=fold_id,
            candidate=candidate,
        )
        application = _application(
            functional,
            validation_subjects,
            gains,
            prefix=f"validation-{fold_id}",
        )
        functionals.append(functional)
        applications.append(application)
        evaluations.append(
            _record_subject_blocked_penalty_fold_evaluation(
                candidate,
                _producer_token=_WORKFLOW_SUBJECT_BLOCKED_PRODUCER_TOKEN,
                inner_fold_id=fold_id,
                inner_fold_manifest_id=fold_id,
                training_subject_ids=training_subjects,
                validation_subject_ids=validation_subjects,
                validation_loss_estimand=(
                    PenaltyValidationLossEstimand.PAIRED_SUBJECT_CONTRAST
                ),
                subject_losses=application.subject_full_losses,
                scale_resolution_id=functional.penalty_scale_resolution_id,
                resolved_penalty_id=functional.resolved_penalty_id,
                resolved_lambda1=functional.lambda1,
                resolved_lambda2=functional.lambda2,
                training_functional_id=functional.incremental_functional_id,
                heldout_application_id=application.application_id,
            )
        )
    tuning = select_penalty_candidate(
        tuning_spec,
        evaluations,
        tuning_scope_id="receiver-selected-penalty-scope",
        training_subject_ids=_SUBJECT_IDS,
        inner_fold_ids=("inner-1", "inner-2"),
        inner_fold_plan_id="inner-plan",
    )
    outer = _fit_functional(
        _SUBJECT_IDS,
        fold_id="outer-1",
        candidate=candidate,
    )
    expected = np.vstack(gain_parts)
    return tuning, tuple(functionals), tuple(applications), outer, expected


def _finalize(
    *, spec: GainCalibrationSpec | None = None
) -> SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact:
    tuning, functionals, applications, outer, _ = _world()
    return _finalize_selected_penalty_inner_oof_gain_calibration(
        _producer_token=_WORKFLOW_GAIN_CALIBRATION_PRODUCER_TOKEN,
        spec=GainCalibrationSpec() if spec is None else spec,
        tuning_artifact=tuning,
        inner_functionals=functionals,
        inner_applications=applications,
        outer_final_functional=outer,
    )


def test_exact_selected_penalty_oof_gain_reconstruction_and_lineage() -> None:
    tuning, functionals, applications, outer, expected = _world()

    artifact = _finalize_selected_penalty_inner_oof_gain_calibration(
        _producer_token=_WORKFLOW_GAIN_CALIBRATION_PRODUCER_TOKEN,
        spec=GainCalibrationSpec(),
        tuning_artifact=tuning,
        inner_functionals=functionals,
        inner_applications=applications,
        outer_final_functional=outer,
    )

    assert artifact.status == "observed"
    assert artifact.reason_code is None
    assert artifact.is_estimable
    assert not artifact.formal_inference_allowed
    assert artifact.penalty_validation_loss_estimand == (
        "paired_subject_contrast_prediction_loss_v1"
    )
    assert "subject_family_observation_equal" in artifact.percentile_policy
    assert len(artifact.positive_gain_source_knots_digest) == 64
    assert len(artifact.positive_gain_percentile_knots_digest) == 64
    assert artifact.training_subject_ids == _SUBJECT_IDS
    assert artifact.selected_candidate_id == tuning.selected_candidate_id
    assert artifact.outer_incremental_functional_id == (outer.incremental_functional_id)
    assert artifact.selected_evaluation_ids == tuple(
        evaluation.evaluation_id
        for evaluation in sorted(
            tuning.evaluations, key=lambda item: item.inner_fold_id
        )
    )
    assert set(artifact.validation_inner_fold_ids) == {"inner-1", "inner-2"}
    np.testing.assert_allclose(
        artifact.bounded_subject_family_gains,
        expected,
        rtol=0.0,
        atol=1e-14,
    )
    assert artifact.structural_zero_mask[0, 0]
    assert artifact.estimable_mask.all()
    for array in (
        artifact.subject_null_losses,
        artifact.subject_family_losses,
        artifact.bounded_subject_family_gains,
        artifact.estimable_mask,
        artifact.structural_zero_mask,
        artifact.positive_gain_source_knots,
        artifact.positive_gain_percentile_knots,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    artifact.to_dict()


def test_percentile_mapping_is_monotone_and_preserves_zero_and_missing() -> None:
    artifact = _finalize()
    source = np.linspace(0.0, 1.0, 201)

    calibrated = artifact.calibrate_gains(source)

    assert calibrated[0] == 0.0
    assert calibrated[-1] == 1.0
    assert np.all(np.diff(calibrated) >= -1e-15)
    assert artifact.calibrate_gain(0.0) == 0.0
    assert artifact.calibrate_gain(None) is None
    assert artifact.calibrate_gain(float("nan")) is None
    mapped_missing = artifact.calibrate_gains(np.asarray([0.0, np.nan, 0.4]))
    assert mapped_missing[0] == 0.0
    assert np.isnan(mapped_missing[1])
    assert 0.0 < mapped_missing[2] <= 1.0
    assert np.all(np.diff(artifact.positive_gain_source_knots) > 0)
    assert np.all(np.diff(artifact.positive_gain_percentile_knots) > 0)


def test_low_support_is_typed_not_estimable_without_reviving_positive_gain() -> None:
    artifact = _finalize(spec=GainCalibrationSpec(min_positive_observations=100))

    assert artifact.status == "not_estimable"
    assert artifact.reason_code == "insufficient_positive_gain_observations"
    assert not artifact.is_estimable
    assert artifact.positive_gain_source_knots.size == 0
    assert artifact.positive_gain_percentile_knots.size == 0
    assert artifact.calibrate_gain(0.0) == 0.0
    assert artifact.calibrate_gain(0.4) is None
    result = artifact.calibrate_gains(np.asarray([0.0, 0.4, np.nan]))
    assert result[0] == 0.0
    assert np.isnan(result[1:]).all()
    artifact.to_dict()


def test_selected_parent_lineage_and_workflow_ownership_fail_closed() -> None:
    tuning, functionals, applications, outer, _ = _world()

    with pytest.raises(TypeError, match="producer-owned"):
        SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact()
    with pytest.raises(TypeError, match="workflow-producer-owned"):
        _finalize_selected_penalty_inner_oof_gain_calibration(
            _producer_token=object(),
            spec=GainCalibrationSpec(),
            tuning_artifact=tuning,
            inner_functionals=functionals,
            inner_applications=applications,
            outer_final_functional=outer,
        )
    with pytest.raises(ContractError) as mismatch:
        _finalize_selected_penalty_inner_oof_gain_calibration(
            _producer_token=_WORKFLOW_GAIN_CALIBRATION_PRODUCER_TOKEN,
            spec=GainCalibrationSpec(),
            tuning_artifact=tuning,
            inner_functionals=functionals,
            inner_applications=applications[:1],
            outer_final_functional=outer,
        )
    assert mismatch.value.details.code == "gain_calibration_parent_mismatch"


def test_spec_and_artifact_tampering_is_detected() -> None:
    spec = GainCalibrationSpec()
    object.__setattr__(spec, "min_subjects", 2)
    with pytest.raises(ContractError) as spec_error:
        spec.to_dict()
    assert spec_error.value.details.code == "gain_calibration_spec_integrity_violation"

    artifact = _finalize()
    changed = np.asarray(artifact.bounded_subject_family_gains).copy()
    changed[0, 1] = 0.99
    object.__setattr__(artifact, "bounded_subject_family_gains", changed)
    with pytest.raises(ContractError) as artifact_error:
        artifact.to_dict()
    assert artifact_error.value.details.code == (
        "gain_calibration_artifact_integrity_violation"
    )
