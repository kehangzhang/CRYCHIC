"""Typed orchestration for receiver incremental-downstream diagnostics.

The low-level incremental model is intentionally exposed here only through
producer-owned design, response, precision, and receiver-family parents. A
trusted autonomous nuisance basis plus producer-owned paired inner tuning can
cross the incremental-child official gate; the complete scoring pipeline
remains separate and noncertifying.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from crychic.attribution import (
    PenaltyFoldEvaluation,
    PenaltyTuningArtifact,
    PenaltyTuningSpec,
    PrecisionTransformResult,
    ReceiverFamilyTrainingArtifact,
    RelativePenaltyCandidate,
    not_estimable_penalty_tuning,
    select_penalty_candidate,
)
from crychic.attribution.tuning import (
    _WORKFLOW_SUBJECT_BLOCKED_PRODUCER_TOKEN,
    _record_subject_blocked_penalty_fold_evaluation,
)
from crychic.core import ContractError, SeedLineage, stable_id
from crychic.design import (
    FrozenDesignApplication,
    FrozenDesignEncoder,
    node_context_fields,
)
from crychic.resampling import (
    FoldEstimabilityResult,
    FoldManifest,
    FoldPlanningError,
    SubjectFoldPlan,
    plan_subject_folds,
)
from crychic.response import (
    AutonomousProgramSupportError,
    FoldGeneResponseApplication,
    FoldGeneResponseArtifact,
    ReceiverAutonomousProgramResource,
)
from crychic.scoring import (
    DownstreamRowManifest,
    IncrementalDownstreamApplication,
    IncrementalDownstreamFunctional,
    apply_incremental_downstream_functional,
    fit_incremental_downstream_functional,
    incremental_heldout_input_digest,
)
from crychic.scoring.downstream import _subject_reference_summary

_PRODUCER_MARKER = "crychic.workflow.receiver_incremental_training.v2"
_APPLICATION_PRODUCER_MARKER = "crychic.workflow.receiver_incremental_application.v2"
_OFFICIAL_STATUS = "not_estimable"
_FORMULA_CERTIFICATION_STATUS = "formula_nuisance_incremental_diagnostic_only"
_FORMULA_REASON = "receiver_autonomous_nuisance_not_frozen"
_TRUSTED_NOT_CERTIFIED_STATUS = "trusted_autonomous_incremental_not_oof_certified_v1"
_TUNING_NOT_CONNECTED_REASON = "subject_blocked_inner_tuning_not_connected"
_CERTIFIED_STATUS = "observed"
_CERTIFIED_TRAINING_STATUS = (
    "trusted_autonomous_outer_frozen_representation_inner_tuned_oof_ready_v1"
)
_CERTIFIED_APPLICATION_STATUS = (
    "trusted_autonomous_outer_frozen_representation_inner_tuned_oof_observed_v1"
)
_CERTIFIED_APPLICATION_NOT_ESTIMABLE_STATUS = (
    "trusted_autonomous_outer_frozen_representation_inner_tuned_heldout_ne_v1"
)
_INNER_STATISTICAL_CONTRACT_FAILURES = frozenset(
    {
        "all_family_bases_not_identifiable_after_autonomous_projection",
        "autonomous_program_support_not_estimable",
        "penalty_scale_not_estimable",
    }
)
_INNER_NOT_ESTIMABLE_VALUE_ERRORS = {
    "context regressor has no variation after nuisance projection": (
        "inner_context_regressor_not_estimable"
    ),
    "independent context regressor has no frozen contrast variation": (
        "inner_independent_context_contrast_not_estimable"
    ),
    "incremental training requires fully paired or independent subjects": (
        "inner_subject_allocation_not_estimable"
    ),
    "reference_mask must select at least two training samples": (
        "inner_reference_sample_support_not_estimable"
    ),
    "reference_mask must select at least two training subjects": (
        "inner_reference_subject_support_not_estimable"
    ),
    "response_matrix requires at least three complete samples": (
        "inner_complete_sample_support_not_estimable"
    ),
    "training nuisance_matrix must have full column rank": (
        "inner_nuisance_design_not_estimable"
    ),
}
_MAD_GAUSSIAN_CONSISTENCY = 1.4826


def _parent_mismatch(message: str, *, field: str) -> ContractError:
    return ContractError(
        message,
        code="receiver_incremental_parent_mismatch",
        field=field,
        remediation="Use typed artifacts produced inside the same physical fold",
    )


def _required_id(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _receiver_tuning_scope_id(
    *,
    encoder_id: str,
    response_artifact_id: str,
    precision_transform_id: str,
    receiver_family_training_artifact_id: str,
    autonomous_program_resource_id: str | None,
    penalty_tuning_spec_id: str,
) -> str:
    scope_id: str = stable_id(
        "receiver_incremental_tuning_scope",
        {
            "autonomous_program_resource_id": autonomous_program_resource_id,
            "encoder_id": encoder_id,
            "penalty_tuning_spec_id": penalty_tuning_spec_id,
            "precision_transform_id": precision_transform_id,
            "receiver_family_training_artifact_id": (
                receiver_family_training_artifact_id
            ),
            "response_artifact_id": response_artifact_id,
        },
        schema_version="1",
    )
    return scope_id


def _validated_hyperparameters(
    *,
    minimum_scale: float,
    null_loss_floor: float,
    lambda1: float,
    lambda2: float,
) -> tuple[float, float, float, float]:
    normalized: dict[str, float] = {}
    for field_name, raw, strictly_positive in (
        ("minimum_scale", minimum_scale, True),
        ("null_loss_floor", null_loss_floor, True),
        ("lambda1", lambda1, False),
        ("lambda2", lambda2, False),
    ):
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{field_name} must be numeric, not boolean")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{field_name} must be a finite numeric scalar") from error
        valid_bound = value > 0 if strictly_positive else value >= 0
        if not math.isfinite(value) or not valid_bound:
            qualifier = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{field_name} must be finite and {qualifier}")
        normalized[field_name] = value
    return (
        normalized["minimum_scale"],
        normalized["null_loss_floor"],
        normalized["lambda1"],
        normalized["lambda2"],
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _InnerFoldDesignChecker:
    """Audit one inner-training subset against outer-frozen design columns."""

    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    reference_mask: np.ndarray
    nuisance_matrix: np.ndarray
    context_regressor: np.ndarray
    contrast_id: str

    @property
    def contrast_ids(self) -> tuple[str, ...]:
        return (self.contrast_id,)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ()

    def __call__(self, training_metadata: pd.DataFrame) -> FoldEstimabilityResult:
        requested = set(training_metadata["sample_id"].astype(str))
        positions = tuple(
            index
            for index, sample_id in enumerate(self.sample_ids)
            if sample_id in requested
        )
        selected_samples = tuple(self.sample_ids[index] for index in positions)
        nuisance = np.asarray(self.nuisance_matrix[list(positions)], dtype=float)
        regressor = np.asarray(self.context_regressor[list(positions)], dtype=float)
        reference = np.asarray(self.reference_mask[list(positions)], dtype=bool)
        reference_subjects = {
            self.sample_subject_ids[index]
            for index, selected in zip(positions, reference, strict=True)
            if bool(selected)
        }
        reason: str | None = None
        if len(positions) < 3:
            reason = "inner_training_requires_three_samples"
        elif len(reference_subjects) < 2:
            reason = "inner_training_requires_two_reference_subjects"
        elif np.linalg.matrix_rank(nuisance) != nuisance.shape[1]:
            reason = "inner_training_nuisance_rank_deficient"
        elif np.linalg.matrix_rank(np.column_stack((nuisance, regressor))) <= (
            nuisance.shape[1]
        ):
            reason = "inner_training_context_regressor_not_identifiable"
        design_matrix_id = stable_id(
            "receiver_incremental_inner_design",
            {
                "contrast_id": self.contrast_id,
                "nuisance_matrix": nuisance.tolist(),
                "regressor": regressor.tolist(),
                "sample_ids": list(selected_samples),
            },
            schema_version="1",
        )
        return FoldEstimabilityResult(
            design_matrix_id=design_matrix_id,
            contrast_ids=self.contrast_ids,
            estimable=reason is None,
            reason_code=reason,
        )


@dataclass(frozen=True, slots=True, init=False)
class ReceiverIncrementalTrainingArtifact:
    """One typed-parent incremental diagnostic fitted on training subjects."""

    encoder_id: str
    context_regressor_id: str
    nuisance_design_id: str
    nuisance_column_ids: tuple[str, ...]
    training_design_application_id: str
    response_artifact_id: str
    precision_transform_id: str
    receiver_family_training_artifact_id: str
    autonomous_program_resource_id: str | None
    autonomous_program_verification_status: str | None
    autonomous_projection_id: str | None
    receiver: str
    contrast_name: str
    fold_id: str
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    training_sample_ids: tuple[str, ...]
    training_sample_context_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    penalty_tuning_spec_id: str | None
    inner_fold_plan: SubjectFoldPlan | None
    penalty_tuning_artifact: PenaltyTuningArtifact | None
    selected_penalty_candidate_id: str | None
    selected_penalty_scale_resolution_id: str | None
    selected_resolved_penalty_id: str | None
    minimum_scale: float
    null_loss_floor: float
    lambda1: float
    lambda2: float
    diagnostic_status: str
    diagnostic_reason_code: str | None
    diagnostic_functional: IncrementalDownstreamFunctional | None
    certification_status: str
    official_incremental_status: str
    reason_code: str | None
    training_artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverIncrementalTrainingArtifact is producer-owned; "
            "use fit_receiver_incremental_training_artifact()"
        )

    @classmethod
    def _from_fit(
        cls,
        *,
        encoder: FrozenDesignEncoder,
        training_design: FrozenDesignApplication,
        response: FoldGeneResponseArtifact,
        precision: PrecisionTransformResult,
        receiver_family: ReceiverFamilyTrainingArtifact,
        autonomous_program_resource: ReceiverAutonomousProgramResource | None,
        penalty_tuning_spec: PenaltyTuningSpec | None,
        inner_fold_plan: SubjectFoldPlan | None,
        penalty_tuning_artifact: PenaltyTuningArtifact | None,
        family_ids: tuple[str, ...],
        minimum_scale: float,
        null_loss_floor: float,
        lambda1: float,
        lambda2: float,
        diagnostic_functional: IncrementalDownstreamFunctional | None,
        diagnostic_reason_code: str | None,
    ) -> ReceiverIncrementalTrainingArtifact:
        diagnostic_status = (
            "observed" if diagnostic_functional is not None else "not_estimable"
        )
        if (diagnostic_functional is None) == (diagnostic_reason_code is None):
            raise ValueError(
                "diagnostic_reason_code is required exactly when fitting is unavailable"
            )
        if penalty_tuning_spec is None:
            if inner_fold_plan is not None or penalty_tuning_artifact is not None:
                raise ValueError("fixed penalties cannot retain tuning artifacts")
        else:
            penalty_tuning_spec._require_intact()
            if penalty_tuning_artifact is None:
                raise ValueError("tuned training requires a tuning artifact")
            penalty_tuning_artifact._require_intact()
            if inner_fold_plan is not None:
                inner_fold_plan._require_intact()
        autonomous_id = (
            None
            if autonomous_program_resource is None
            else autonomous_program_resource.artifact_id
        )
        autonomous_verification_status = (
            None
            if autonomous_program_resource is None
            else autonomous_program_resource.verification_status
        )
        projection_id = (
            None
            if autonomous_program_resource is None or diagnostic_functional is None
            else diagnostic_functional.autonomous_projection_id
        )
        tuning_spec_id = (
            None if penalty_tuning_spec is None else penalty_tuning_spec.spec_id
        )
        selected_candidate_id = (
            None
            if penalty_tuning_artifact is None
            else penalty_tuning_artifact.selected_candidate_id
        )
        selected_scale_resolution_id = (
            None
            if diagnostic_functional is None
            else diagnostic_functional.penalty_scale_resolution_id
        )
        selected_resolved_penalty_id = (
            None
            if diagnostic_functional is None
            else diagnostic_functional.resolved_penalty_id
        )
        certified = bool(
            diagnostic_functional is not None
            and autonomous_program_resource is not None
            and autonomous_program_resource.is_manifest_verified_trusted
            and penalty_tuning_artifact is not None
            and penalty_tuning_artifact.is_oof_certified
            and diagnostic_functional.penalty_candidate_id
            == penalty_tuning_artifact.selected_candidate_id
        )
        trusted_resource = bool(
            autonomous_program_resource is not None
            and autonomous_program_resource.is_manifest_verified_trusted
        )
        if certified:
            certification_status = _CERTIFIED_TRAINING_STATUS
            official_status = _CERTIFIED_STATUS
            official_reason: str | None = None
        elif not trusted_resource:
            certification_status = _FORMULA_CERTIFICATION_STATUS
            official_status = _OFFICIAL_STATUS
            official_reason = _FORMULA_REASON
        else:
            certification_status = _TRUSTED_NOT_CERTIFIED_STATUS
            official_status = _OFFICIAL_STATUS
            if penalty_tuning_spec is None:
                official_reason = _TUNING_NOT_CONNECTED_REASON
            else:
                official_reason = diagnostic_reason_code or (
                    None
                    if penalty_tuning_artifact is None
                    else penalty_tuning_artifact.reason_code
                )
                official_reason = (
                    official_reason
                    or "subject_blocked_inner_tuning_not_estimable"
                )
        self = object.__new__(cls)
        values: dict[str, Any] = {
            "encoder_id": encoder.encoder_id,
            "context_regressor_id": training_design.context_regressor_id,
            "nuisance_design_id": training_design.nuisance_design_id,
            "nuisance_column_ids": encoder.nuisance_column_ids,
            "training_design_application_id": training_design.application_id,
            "response_artifact_id": response.artifact_id,
            "precision_transform_id": precision.precision_transform_id,
            "receiver_family_training_artifact_id": (
                receiver_family.training_artifact_id
            ),
            "autonomous_program_resource_id": autonomous_id,
            "autonomous_program_verification_status": (autonomous_verification_status),
            "autonomous_projection_id": projection_id,
            "receiver": response.receiver,
            "contrast_name": response.contrast_name,
            "fold_id": response.fold_id,
            "feature_ids": response.feature_ids,
            "family_ids": family_ids,
            "training_sample_ids": response.sample_ids,
            "training_sample_context_ids": response.sample_context_ids,
            "training_subject_ids": response.training_subject_ids,
            "penalty_tuning_spec_id": tuning_spec_id,
            "inner_fold_plan": inner_fold_plan,
            "penalty_tuning_artifact": penalty_tuning_artifact,
            "selected_penalty_candidate_id": selected_candidate_id,
            "selected_penalty_scale_resolution_id": selected_scale_resolution_id,
            "selected_resolved_penalty_id": selected_resolved_penalty_id,
            "minimum_scale": float(minimum_scale),
            "null_loss_floor": float(null_loss_floor),
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "diagnostic_status": diagnostic_status,
            "diagnostic_reason_code": diagnostic_reason_code,
            "diagnostic_functional": diagnostic_functional,
            "certification_status": certification_status,
            "official_incremental_status": official_status,
            "reason_code": official_reason,
            "_producer_marker": _PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "training_artifact_id",
            stable_id(
                "receiver_incremental_training_artifact",
                self._identity_payload(),
                schema_version="2",
            ),
        )
        return self

    @property
    def is_oof_certified(self) -> bool:
        """Return whether trusted nuisance and tuning make the model OOF-ready."""

        return (
            self.certification_status == _CERTIFIED_TRAINING_STATUS
            and self.official_incremental_status == _CERTIFIED_STATUS
            and self.reason_code is None
        )

    def _identity_payload(self) -> dict[str, object]:
        functional_id = (
            None
            if self.diagnostic_functional is None
            else self.diagnostic_functional.incremental_functional_id
        )
        inner_fold_plan_id = (
            None if self.inner_fold_plan is None else self.inner_fold_plan.plan_id
        )
        tuning_id = (
            None
            if self.penalty_tuning_artifact is None
            else self.penalty_tuning_artifact.tuning_id
        )
        return {
            "certification_status": self.certification_status,
            "autonomous_program_resource_id": self.autonomous_program_resource_id,
            "autonomous_program_verification_status": (
                self.autonomous_program_verification_status
            ),
            "autonomous_projection_id": self.autonomous_projection_id,
            "contrast_name": self.contrast_name,
            "context_regressor_id": self.context_regressor_id,
            "diagnostic_functional_id": functional_id,
            "diagnostic_reason_code": self.diagnostic_reason_code,
            "diagnostic_status": self.diagnostic_status,
            "encoder_id": self.encoder_id,
            "family_ids": list(self.family_ids),
            "feature_ids": list(self.feature_ids),
            "fold_id": self.fold_id,
            "inner_fold_plan_id": inner_fold_plan_id,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "minimum_scale": self.minimum_scale,
            "nuisance_column_ids": list(self.nuisance_column_ids),
            "nuisance_design_id": self.nuisance_design_id,
            "null_loss_floor": self.null_loss_floor,
            "official_incremental_status": self.official_incremental_status,
            "precision_transform_id": self.precision_transform_id,
            "penalty_tuning_artifact_id": tuning_id,
            "penalty_tuning_spec_id": self.penalty_tuning_spec_id,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "receiver_family_training_artifact_id": (
                self.receiver_family_training_artifact_id
            ),
            "response_artifact_id": self.response_artifact_id,
            "selected_penalty_candidate_id": self.selected_penalty_candidate_id,
            "selected_penalty_scale_resolution_id": (
                self.selected_penalty_scale_resolution_id
            ),
            "selected_resolved_penalty_id": self.selected_resolved_penalty_id,
            "training_design_application_id": self.training_design_application_id,
            "training_sample_ids": list(self.training_sample_ids),
            "training_sample_context_ids": list(self.training_sample_context_ids),
            "training_subject_ids": list(self.training_subject_ids),
        }

    def _require_intact(self) -> None:
        try:
            _validated_hyperparameters(
                minimum_scale=self.minimum_scale,
                null_loss_floor=self.null_loss_floor,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
            )
            for field_name in (
                "encoder_id",
                "context_regressor_id",
                "nuisance_design_id",
                "training_design_application_id",
                "response_artifact_id",
                "precision_transform_id",
                "receiver_family_training_artifact_id",
                "receiver",
                "contrast_name",
                "fold_id",
                "training_artifact_id",
            ):
                _required_id(getattr(self, field_name), field_name=field_name)
            if len(self.nuisance_column_ids) < 1 or any(
                _required_id(value, field_name="nuisance_column_ids") != value
                for value in self.nuisance_column_ids
            ):
                raise ValueError("nuisance_column_ids are invalid")
            if len(set(self.nuisance_column_ids)) != len(self.nuisance_column_ids):
                raise ValueError("nuisance_column_ids must be unique")
            if len(self.training_sample_context_ids) != len(
                self.training_sample_ids
            ) or any(
                _required_id(value, field_name="training_sample_context_ids") != value
                for value in self.training_sample_context_ids
            ):
                raise ValueError("training sample contexts are invalid")
            if (self.autonomous_program_resource_id is None) != (
                self.autonomous_program_verification_status is None
            ):
                raise ValueError(
                    "autonomous resource ID and verification status must co-occur"
                )
            if self.autonomous_program_verification_status not in {
                None,
                "caller_declared_static_unverified",
                "manifest_verified_static_trusted_v1",
            }:
                raise ValueError("unsupported autonomous resource verification status")
            tuning = self.penalty_tuning_artifact
            plan = self.inner_fold_plan
            if self.penalty_tuning_spec_id is None:
                if tuning is not None or plan is not None:
                    raise ValueError("fixed penalties cannot retain tuning artifacts")
                if any(
                    value is not None
                    for value in (
                        self.selected_penalty_candidate_id,
                        self.selected_penalty_scale_resolution_id,
                        self.selected_resolved_penalty_id,
                    )
                ):
                    raise ValueError("fixed penalties cannot claim selected lineage")
            else:
                _required_id(
                    self.penalty_tuning_spec_id,
                    field_name="penalty_tuning_spec_id",
                )
                if tuning is None:
                    raise ValueError("tuned training requires a tuning artifact")
                tuning._require_intact()
                expected_scope_id = _receiver_tuning_scope_id(
                    encoder_id=self.encoder_id,
                    response_artifact_id=self.response_artifact_id,
                    precision_transform_id=self.precision_transform_id,
                    receiver_family_training_artifact_id=(
                        self.receiver_family_training_artifact_id
                    ),
                    autonomous_program_resource_id=(
                        self.autonomous_program_resource_id
                    ),
                    penalty_tuning_spec_id=self.penalty_tuning_spec_id,
                )
                if (
                    tuning.spec.spec_id != self.penalty_tuning_spec_id
                    or tuning.tuning_scope_id != expected_scope_id
                    or tuning.training_subject_ids != self.training_subject_ids
                    or tuning.selected_candidate_id
                    != self.selected_penalty_candidate_id
                ):
                    raise ValueError("penalty tuning lineage does not match training")
                if plan is None:
                    if tuning.inner_fold_plan_id is not None:
                        raise ValueError("tuning claims an absent inner fold plan")
                else:
                    plan._require_intact()
                    if (
                        plan.plan_id != tuning.inner_fold_plan_id
                        or plan.subject_ids != self.training_subject_ids
                        or tuple(sorted(fold.fold_id for fold in plan.folds))
                        != tuning.inner_fold_ids
                    ):
                        raise ValueError(
                            "inner fold plan does not match tuning lineage"
                        )
            if self.diagnostic_functional is not None:
                self.diagnostic_functional._require_intact()
                functional = self.diagnostic_functional
                if (
                    self.diagnostic_status != "observed"
                    or self.diagnostic_reason_code is not None
                    or functional.incremental_functional_id
                    != self._identity_payload()["diagnostic_functional_id"]
                ):
                    raise ValueError("diagnostic functional status is inconsistent")
                expected_functional_lineage = (
                    self.receiver,
                    self.contrast_name,
                    self.fold_id,
                    self.feature_ids,
                    self.family_ids,
                    self.training_sample_ids,
                    self.training_sample_context_ids,
                    self.training_subject_ids,
                    self.context_regressor_id,
                    self.nuisance_design_id,
                    self.nuisance_column_ids,
                    self.minimum_scale,
                    self.null_loss_floor,
                    self.lambda1,
                    self.lambda2,
                    self.autonomous_program_resource_id,
                )
                observed_functional_lineage = (
                    functional.receiver,
                    functional.contrast_name,
                    functional.fold_id,
                    functional.feature_ids,
                    functional.family_ids,
                    functional.training_sample_ids,
                    functional.training_sample_context_ids,
                    functional.training_subject_ids,
                    functional.context_regressor_id,
                    functional.nuisance_design_id,
                    functional.nuisance_column_ids,
                    functional.minimum_scale,
                    functional.null_loss_floor,
                    functional.lambda1,
                    functional.lambda2,
                    functional.autonomous_basis_id,
                )
                if observed_functional_lineage != expected_functional_lineage:
                    raise ValueError(
                        "diagnostic functional does not match its wrapper lineage"
                    )
                expected_projection_id = (
                    None
                    if self.autonomous_program_resource_id is None
                    else functional.autonomous_projection_id
                )
                if self.autonomous_projection_id != expected_projection_id:
                    raise ValueError(
                        "autonomous projection does not match the diagnostic functional"
                    )
                functional_penalty_lineage = (
                    functional.penalty_candidate_id,
                    functional.penalty_scale_resolution_id,
                    functional.resolved_penalty_id,
                )
                wrapper_penalty_lineage = (
                    self.selected_penalty_candidate_id,
                    self.selected_penalty_scale_resolution_id,
                    self.selected_resolved_penalty_id,
                )
                if functional_penalty_lineage != wrapper_penalty_lineage:
                    raise ValueError(
                        "selected penalty lineage does not match final functional"
                    )
            elif (
                self.diagnostic_status != "not_estimable"
                or not self.diagnostic_reason_code
            ):
                raise ValueError("not-estimable diagnostic status is inconsistent")
            elif self.autonomous_projection_id is not None:
                raise ValueError(
                    "unavailable diagnostics cannot claim an autonomous projection"
                )
            elif self.selected_penalty_scale_resolution_id is not None or (
                self.selected_resolved_penalty_id is not None
            ):
                raise ValueError(
                    "unavailable diagnostics cannot claim resolved final penalties"
                )
            certified = bool(
                self.diagnostic_functional is not None
                and self.autonomous_program_verification_status
                == "manifest_verified_static_trusted_v1"
                and tuning is not None
                and tuning.is_oof_certified
                and self.selected_penalty_candidate_id
                == self.diagnostic_functional.penalty_candidate_id
            )
            trusted_resource = (
                self.autonomous_program_verification_status
                == "manifest_verified_static_trusted_v1"
            )
            if certified:
                expected_certification_status = _CERTIFIED_TRAINING_STATUS
                expected_official_status = _CERTIFIED_STATUS
                expected_reason: str | None = None
            elif not trusted_resource:
                expected_certification_status = _FORMULA_CERTIFICATION_STATUS
                expected_official_status = _OFFICIAL_STATUS
                expected_reason = _FORMULA_REASON
            else:
                expected_certification_status = _TRUSTED_NOT_CERTIFIED_STATUS
                expected_official_status = _OFFICIAL_STATUS
                if self.penalty_tuning_spec_id is None:
                    expected_reason = _TUNING_NOT_CONNECTED_REASON
                else:
                    expected_reason = self.diagnostic_reason_code or (
                        None if tuning is None else tuning.reason_code
                    )
                    expected_reason = (
                        expected_reason
                        or "subject_blocked_inner_tuning_not_estimable"
                    )
            expected = stable_id(
                "receiver_incremental_training_artifact",
                self._identity_payload(),
                schema_version="2",
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and self.certification_status == expected_certification_status
                and self.official_incremental_status == expected_official_status
                and self.reason_code == expected_reason
                and self.is_oof_certified == certified
                and expected == self.training_artifact_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver incremental training artifact failed integrity validation",
                code="receiver_incremental_training_integrity_violation",
                field="training_artifact_id",
                remediation="Refit from intact typed training parents",
            ) from error
        if not valid:
            raise ContractError(
                "Receiver incremental training artifact failed integrity validation",
                code="receiver_incremental_training_integrity_violation",
                field="training_artifact_id",
                remediation="Refit from intact typed training parents",
            )

    def to_dict(self) -> dict[str, object]:
        """Return complete parent lineage without expanding learned arrays."""

        self._require_intact()
        return {
            "training_artifact_id": self.training_artifact_id,
            **self._identity_payload(),
            "inner_fold_plan": (
                None
                if self.inner_fold_plan is None
                else self.inner_fold_plan.to_dict()
            ),
            "penalty_tuning_artifact": (
                None
                if self.penalty_tuning_artifact is None
                else self.penalty_tuning_artifact.to_dict()
            ),
        }


def _validate_heldout_parents(
    model: ReceiverIncrementalTrainingArtifact,
    response_application: FoldGeneResponseApplication,
    design_application: FrozenDesignApplication,
) -> None:
    model._require_intact()
    response_application.to_dict()
    design_application.to_dict()
    expected = (
        model.response_artifact_id,
        design_application.application_id,
        model.encoder_id,
        model.receiver,
        model.contrast_name,
        model.fold_id,
        model.feature_ids,
        design_application.sample_ids,
        design_application.sample_subject_ids,
        design_application.sample_context_ids,
    )
    observed = (
        response_application.training_response_id,
        response_application.design_application_id,
        response_application.encoder_id,
        response_application.receiver,
        response_application.contrast_name,
        response_application.fold_id,
        response_application.feature_ids,
        response_application.sample_ids,
        response_application.sample_subject_ids,
        response_application.sample_context_ids,
    )
    if observed != expected or design_application.encoder_id != model.encoder_id:
        raise _parent_mismatch(
            "Held-out response and design do not match the training artifact",
            field="response_application_id",
        )
    if design_application.application_scope != "heldout":
        raise _parent_mismatch(
            "Receiver incremental application requires a held-out design",
            field="design_application_id",
        )
    sample_overlap = set(response_application.sample_ids).intersection(
        model.training_sample_ids
    )
    if sample_overlap:
        raise ValueError(
            "heldout sample IDs overlap training samples: "
            + ", ".join(sorted(sample_overlap))
        )
    subject_overlap = set(response_application.subject_ids).intersection(
        model.training_subject_ids
    )
    if subject_overlap:
        raise ValueError(
            "heldout subjects overlap training subjects: "
            + ", ".join(sorted(subject_overlap))
        )


@dataclass(frozen=True, slots=True, init=False)
class ReceiverIncrementalApplication:
    """Held-out incremental diagnostic applied without any refitting."""

    training_artifact_id: str
    diagnostic_functional_id: str | None
    response_application_id: str
    design_application_id: str
    heldout_sample_ids: tuple[str, ...]
    heldout_context_ids: tuple[str, ...]
    heldout_subject_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    diagnostic_status: str
    diagnostic_reason_code: str | None
    diagnostic_application: IncrementalDownstreamApplication | None
    certification_status: str
    official_incremental_status: str
    reason_code: str | None
    application_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverIncrementalApplication is producer-owned; "
            "use apply_receiver_incremental_training_artifact()"
        )

    @classmethod
    def _from_application(
        cls,
        *,
        model: ReceiverIncrementalTrainingArtifact,
        response_application: FoldGeneResponseApplication,
        design_application: FrozenDesignApplication,
        diagnostic_application: IncrementalDownstreamApplication | None,
        diagnostic_reason_code: str | None,
    ) -> ReceiverIncrementalApplication:
        _validate_heldout_parents(model, response_application, design_application)
        diagnostic_status = (
            "observed"
            if diagnostic_application is not None
            and diagnostic_application.status == "observed"
            else "not_estimable"
        )
        if diagnostic_status == "observed":
            if diagnostic_reason_code is not None:
                raise ValueError("observed diagnostic application cannot have a reason")
        elif not diagnostic_reason_code:
            raise ValueError("not-estimable diagnostic application requires a reason")
        if diagnostic_application is not None:
            if model.diagnostic_functional is None:
                raise ValueError(
                    "diagnostic application requires a fitted diagnostic functional"
                )
            expected_diagnostic_lineage = (
                model.diagnostic_functional.incremental_functional_id,
                model.family_ids,
                response_application.sample_ids,
                response_application.sample_subject_ids,
                response_application.sample_context_ids,
                response_application.subject_ids,
            )
            observed_diagnostic_lineage = (
                diagnostic_application.incremental_functional_id,
                diagnostic_application.family_ids,
                diagnostic_application.sample_ids,
                diagnostic_application.sample_subject_ids,
                diagnostic_application.sample_context_ids,
                diagnostic_application.heldout_subject_ids,
            )
            if observed_diagnostic_lineage != expected_diagnostic_lineage:
                raise ValueError(
                    "diagnostic application does not match its held-out parents"
                )
            row_manifest = DownstreamRowManifest(
                sample_ids=response_application.sample_ids,
                subject_ids=response_application.sample_subject_ids,
                context_ids=response_application.sample_context_ids,
            )
            expected_input_digest = incremental_heldout_input_digest(
                response_application.sample_values,
                row_manifest=row_manifest,
                design_sample_ids=design_application.sample_ids,
                nuisance_matrix=design_application.nuisance_matrix,
                context_regressor=design_application.context_regressor,
                context_regressor_id=design_application.context_regressor_id,
                nuisance_design_id=design_application.nuisance_design_id,
                feature_ids=response_application.feature_ids,
                nuisance_column_ids=model.diagnostic_functional.nuisance_column_ids,
            )
            if (
                diagnostic_application.heldout_row_manifest_id
                != row_manifest.manifest_id
                or diagnostic_application.heldout_input_digest != expected_input_digest
            ):
                raise _parent_mismatch(
                    "Diagnostic application does not derive from the exact held-out "
                    "response and design values",
                    field="diagnostic_application",
                )
        certified_application = model.is_oof_certified and (
            diagnostic_status == "observed"
        )
        certification_status = (
            _CERTIFIED_APPLICATION_STATUS
            if certified_application
            else (
                _CERTIFIED_APPLICATION_NOT_ESTIMABLE_STATUS
                if model.is_oof_certified
                else model.certification_status
            )
        )
        official_status = (
            _CERTIFIED_STATUS if certified_application else _OFFICIAL_STATUS
        )
        official_reason = (
            None
            if certified_application
            else (
                diagnostic_reason_code
                if model.is_oof_certified
                else model.reason_code
            )
        )
        self = object.__new__(cls)
        values: dict[str, Any] = {
            "training_artifact_id": model.training_artifact_id,
            "diagnostic_functional_id": (
                None
                if model.diagnostic_functional is None
                else model.diagnostic_functional.incremental_functional_id
            ),
            "response_application_id": response_application.application_id,
            "design_application_id": design_application.application_id,
            "heldout_sample_ids": response_application.sample_ids,
            "heldout_context_ids": response_application.sample_context_ids,
            "heldout_subject_ids": response_application.subject_ids,
            "training_subject_ids": model.training_subject_ids,
            "diagnostic_status": diagnostic_status,
            "diagnostic_reason_code": diagnostic_reason_code,
            "diagnostic_application": diagnostic_application,
            "certification_status": certification_status,
            "official_incremental_status": official_status,
            "reason_code": official_reason,
            "_producer_marker": _APPLICATION_PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "application_id",
            stable_id(
                "receiver_incremental_application",
                self._identity_payload(),
                schema_version="2",
            ),
        )
        return self

    @property
    def is_oof_certified(self) -> bool:
        """Return whether this held-out result passed the certified path."""

        return (
            self.certification_status == _CERTIFIED_APPLICATION_STATUS
            and self.official_incremental_status == _CERTIFIED_STATUS
            and self.reason_code is None
            and self.diagnostic_status == "observed"
        )

    def _identity_payload(self) -> dict[str, object]:
        diagnostic_id = (
            None
            if self.diagnostic_application is None
            else self.diagnostic_application.application_id
        )
        return {
            "certification_status": self.certification_status,
            "design_application_id": self.design_application_id,
            "diagnostic_application_id": diagnostic_id,
            "diagnostic_functional_id": self.diagnostic_functional_id,
            "diagnostic_reason_code": self.diagnostic_reason_code,
            "diagnostic_status": self.diagnostic_status,
            "heldout_sample_ids": list(self.heldout_sample_ids),
            "heldout_context_ids": list(self.heldout_context_ids),
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "official_incremental_status": self.official_incremental_status,
            "reason_code": self.reason_code,
            "response_application_id": self.response_application_id,
            "training_artifact_id": self.training_artifact_id,
            "training_subject_ids": list(self.training_subject_ids),
        }

    def _require_intact(self) -> None:
        try:
            if set(self.heldout_subject_ids).intersection(self.training_subject_ids):
                raise ValueError("held-out subjects overlap training subjects")
            if self.diagnostic_application is not None:
                self.diagnostic_application._require_intact()
                if (
                    self.diagnostic_application.status != self.diagnostic_status
                    or self.diagnostic_application.incremental_functional_id
                    != self.diagnostic_functional_id
                    or self.diagnostic_application.sample_ids != self.heldout_sample_ids
                    or self.diagnostic_application.sample_context_ids
                    != self.heldout_context_ids
                    or self.diagnostic_application.heldout_subject_ids
                    != self.heldout_subject_ids
                ):
                    raise ValueError("diagnostic application status changed")
            elif self.diagnostic_status != "not_estimable":
                raise ValueError("missing diagnostic application must be unavailable")
            if self.diagnostic_functional_id is not None:
                _required_id(
                    self.diagnostic_functional_id,
                    field_name="diagnostic_functional_id",
                )
            expected = stable_id(
                "receiver_incremental_application",
                self._identity_payload(),
                schema_version="2",
            )
            certified = self.is_oof_certified
            trusted_but_unavailable = (
                self.certification_status
                == _CERTIFIED_APPLICATION_NOT_ESTIMABLE_STATUS
            )
            valid = (
                self._producer_marker == _APPLICATION_PRODUCER_MARKER
                and (
                    (
                        certified
                        and self.official_incremental_status == _CERTIFIED_STATUS
                        and self.reason_code is None
                    )
                    or (
                        trusted_but_unavailable
                        and self.official_incremental_status == _OFFICIAL_STATUS
                        and bool(self.reason_code)
                        and self.diagnostic_status == "not_estimable"
                    )
                    or (
                        self.certification_status
                        in {
                            _FORMULA_CERTIFICATION_STATUS,
                            _TRUSTED_NOT_CERTIFIED_STATUS,
                        }
                        and self.official_incremental_status == _OFFICIAL_STATUS
                        and bool(self.reason_code)
                        and not certified
                    )
                )
                and expected == self.application_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver incremental application failed integrity validation",
                code="receiver_incremental_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact training artifact to held-out parents",
            ) from error
        if not valid:
            raise ContractError(
                "Receiver incremental application failed integrity validation",
                code="receiver_incremental_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact training artifact to held-out parents",
            )

    def to_dict(self) -> dict[str, object]:
        """Return held-out parent lineage and diagnostic status."""

        self._require_intact()
        return {"application_id": self.application_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class _IncrementalFitInputs:
    response_matrix: np.ndarray
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_context_ids: tuple[str, ...]
    nuisance_matrix: np.ndarray
    context_regressor: np.ndarray
    reference_mask: np.ndarray
    receiver: str
    contrast_name: str
    context_regressor_id: str
    nuisance_design_id: str
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    nuisance_column_ids: tuple[str, ...]
    family_basis: sparse.spmatrix | np.ndarray
    autonomous_program_resource: ReceiverAutonomousProgramResource | None
    precision_weights: np.ndarray
    minimum_scale: float
    null_loss_floor: float


def _subject_indices(
    inputs: _IncrementalFitInputs, subject_ids: tuple[str, ...]
) -> np.ndarray:
    requested = set(subject_ids)
    indices = np.asarray(
        [
            index
            for index, subject_id in enumerate(inputs.sample_subject_ids)
            if subject_id in requested
        ],
        dtype=np.int64,
    )
    observed = {
        inputs.sample_subject_ids[index] for index in indices.tolist()
    }
    if observed != requested:
        raise ValueError("subject subset is absent from incremental training inputs")
    return cast(np.ndarray, indices)


def _fit_incremental_subset(
    inputs: _IncrementalFitInputs,
    *,
    subject_ids: tuple[str, ...],
    fold_id: str,
    penalty_candidate: RelativePenaltyCandidate | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
) -> IncrementalDownstreamFunctional:
    indices = _subject_indices(inputs, subject_ids)
    sample_ids = tuple(inputs.sample_ids[index] for index in indices)
    sample_subject_ids = tuple(
        inputs.sample_subject_ids[index] for index in indices
    )
    sample_context_ids = tuple(
        inputs.sample_context_ids[index] for index in indices
    )
    return fit_incremental_downstream_functional(
        inputs.response_matrix[indices],
        row_manifest=DownstreamRowManifest(
            sample_ids=sample_ids,
            subject_ids=sample_subject_ids,
            context_ids=sample_context_ids,
        ),
        design_sample_ids=sample_ids,
        reference_mask=inputs.reference_mask[indices],
        nuisance_matrix=inputs.nuisance_matrix[indices],
        context_regressor=inputs.context_regressor[indices],
        receiver=inputs.receiver,
        contrast_name=inputs.contrast_name,
        fold_id=fold_id,
        context_regressor_id=inputs.context_regressor_id,
        nuisance_design_id=inputs.nuisance_design_id,
        feature_ids=inputs.feature_ids,
        family_ids=inputs.family_ids,
        nuisance_column_ids=inputs.nuisance_column_ids,
        training_subject_ids=tuple(sorted(subject_ids)),
        family_basis=inputs.family_basis,
        autonomous_program_resource=inputs.autonomous_program_resource,
        precision_weights=inputs.precision_weights,
        minimum_scale=inputs.minimum_scale,
        null_loss_floor=inputs.null_loss_floor,
        lambda1=lambda1,
        lambda2=lambda2,
        penalty_candidate=penalty_candidate,
    )


def _apply_incremental_subset(
    functional: IncrementalDownstreamFunctional,
    inputs: _IncrementalFitInputs,
    *,
    subject_ids: tuple[str, ...],
) -> IncrementalDownstreamApplication:
    indices = _subject_indices(inputs, subject_ids)
    sample_ids = tuple(inputs.sample_ids[index] for index in indices)
    return apply_incremental_downstream_functional(
        functional,
        inputs.response_matrix[indices],
        row_manifest=DownstreamRowManifest(
            sample_ids=sample_ids,
            subject_ids=tuple(inputs.sample_subject_ids[index] for index in indices),
            context_ids=tuple(inputs.sample_context_ids[index] for index in indices),
        ),
        design_sample_ids=sample_ids,
        nuisance_matrix=inputs.nuisance_matrix[indices],
        context_regressor=inputs.context_regressor[indices],
        context_regressor_id=inputs.context_regressor_id,
        nuisance_design_id=inputs.nuisance_design_id,
        feature_ids=inputs.feature_ids,
        nuisance_column_ids=inputs.nuisance_column_ids,
    )


def _inner_failure_reason(error: Exception, *, stage: str) -> str:
    if isinstance(error, AutonomousProgramSupportError):
        reason: str = error.reason_code
        return reason
    if isinstance(error, ContractError):
        code: str = error.details.code
        return code
    return f"inner_penalty_candidate_{stage}_not_estimable"


def _classify_inner_value_error(error: ValueError) -> tuple[str, str] | None:
    message = str(error)
    reason = _INNER_NOT_ESTIMABLE_VALUE_ERRORS.get(message)
    if reason is not None:
        return "not_estimable", reason
    if message.startswith("incremental downstream family solver did not converge:"):
        return "failed", "inner_penalty_solver_not_converged"
    return None


def _unavailable_inner_evaluation(
    candidate: RelativePenaltyCandidate,
    inner_fold: FoldManifest,
    *,
    reason_code: str,
    functional: IncrementalDownstreamFunctional | None,
    application: IncrementalDownstreamApplication | None,
    status: str = "not_estimable",
) -> PenaltyFoldEvaluation:
    return _record_subject_blocked_penalty_fold_evaluation(
        candidate,
        _producer_token=_WORKFLOW_SUBJECT_BLOCKED_PRODUCER_TOKEN,
        inner_fold_id=inner_fold.fold_id,
        inner_fold_manifest_id=inner_fold.fold_id,
        training_subject_ids=inner_fold.train_subject_ids,
        validation_subject_ids=inner_fold.test_subject_ids,
        scale_resolution_id=(
            None if functional is None else functional.penalty_scale_resolution_id
        ),
        resolved_penalty_id=(
            None if functional is None else functional.resolved_penalty_id
        ),
        resolved_lambda1=(None if functional is None else functional.lambda1),
        resolved_lambda2=(None if functional is None else functional.lambda2),
        training_functional_id=(
            None if functional is None else functional.incremental_functional_id
        ),
        heldout_application_id=(
            None if application is None else application.application_id
        ),
        status=status,
        reason_code=reason_code,
    )


def _fit_subject_blocked_penalty_tuning(
    inputs: _IncrementalFitInputs,
    *,
    spec: PenaltyTuningSpec,
    tuning_scope_id: str,
    outer_fold_id: str,
    inner_partition_seed_lineage: SeedLineage | None,
) -> tuple[SubjectFoldPlan | None, PenaltyTuningArtifact]:
    spec._require_intact()
    outer_subjects = tuple(sorted(set(inputs.sample_subject_ids)))
    contexts = set(inputs.sample_context_ids)
    contexts_by_subject = {
        subject: {
            context
            for row_subject, context in zip(
                inputs.sample_subject_ids,
                inputs.sample_context_ids,
                strict=True,
            )
            if row_subject == subject
        }
        for subject in outer_subjects
    }
    fully_paired = bool(contexts) and all(
        subject_contexts == contexts
        for subject_contexts in contexts_by_subject.values()
    )
    independent = len(contexts) >= 2 and all(
        len(subject_contexts) == 1
        for subject_contexts in contexts_by_subject.values()
    )
    if independent:
        return None, not_estimable_penalty_tuning(
            spec,
            tuning_scope_id=tuning_scope_id,
            training_subject_ids=outer_subjects,
            reason_code="independent_inner_tuning_one_se_not_supported",
        )
    if not fully_paired:
        return None, not_estimable_penalty_tuning(
            spec,
            tuning_scope_id=tuning_scope_id,
            training_subject_ids=outer_subjects,
            reason_code="mixed_subject_allocation_inner_tuning_not_supported",
        )
    if inputs.autonomous_program_resource is None:
        return None, not_estimable_penalty_tuning(
            spec,
            tuning_scope_id=tuning_scope_id,
            training_subject_ids=outer_subjects,
            reason_code=(
                "signed_residual_inner_tuning_requires_autonomous_program_resource"
            ),
        )
    split_scope_id = stable_id(
        "receiver_incremental_inner_split_scope",
        {
            "contrast_name": inputs.contrast_name,
            "outer_fold_id": outer_fold_id,
            "penalty_tuning_spec_id": spec.spec_id,
            "receiver": inputs.receiver,
            "training_subject_ids": list(outer_subjects),
        },
        schema_version="1",
    )
    contrast_id = stable_id(
        "receiver_incremental_inner_contrast",
        {
            "contrast_name": inputs.contrast_name,
            "outer_fold_id": outer_fold_id,
            "receiver": inputs.receiver,
            "split_scope_id": split_scope_id,
        },
        schema_version="1",
    )
    metadata = pd.DataFrame(
        {
            "sample_id": inputs.sample_ids,
            "subject_id": inputs.sample_subject_ids,
            "context_id": inputs.sample_context_ids,
        }
    )
    repeat_id = stable_id(
        "receiver_incremental_inner_repeat",
        {"split_scope_id": split_scope_id},
        schema_version="1",
    )
    try:
        plan = plan_subject_folds(
            metadata,
            design_checker=_InnerFoldDesignChecker(
                sample_ids=inputs.sample_ids,
                sample_subject_ids=inputs.sample_subject_ids,
                reference_mask=inputs.reference_mask,
                nuisance_matrix=inputs.nuisance_matrix,
                context_regressor=inputs.context_regressor,
                contrast_id=contrast_id,
            ),
            context_keys=("context_id",),
            allowed_n_splits=spec.inner_allowed_n_splits,
            min_train_subjects_per_context=(
                spec.min_inner_train_subjects_per_context
            ),
            min_test_subjects_per_context=(
                spec.min_inner_validation_subjects_per_context
            ),
            repeat_id=repeat_id,
            seed_lineage=SeedLineage(spec.root_seed).derive(
                "receiver_incremental_inner_tuning", split_scope_id
            ),
            partition_seed_lineage=inner_partition_seed_lineage,
        )
    except FoldPlanningError as error:
        return None, not_estimable_penalty_tuning(
            spec,
            tuning_scope_id=tuning_scope_id,
            training_subject_ids=outer_subjects,
            reason_code=error.details.code,
        )

    evaluations: list[PenaltyFoldEvaluation] = []
    for candidate in spec.candidates:
        for inner_fold in plan.folds:
            functional: IncrementalDownstreamFunctional | None = None
            application: IncrementalDownstreamApplication | None = None
            try:
                functional = _fit_incremental_subset(
                    inputs,
                    subject_ids=inner_fold.train_subject_ids,
                    fold_id=inner_fold.fold_id,
                    penalty_candidate=candidate,
                )
                application = _apply_incremental_subset(
                    functional,
                    inputs,
                    subject_ids=inner_fold.test_subject_ids,
                )
                if application.status != "observed":
                    evaluations.append(
                        _unavailable_inner_evaluation(
                            candidate,
                            inner_fold,
                            reason_code=(
                                application.reason_code
                                or "inner_penalty_candidate_apply_not_estimable"
                            ),
                            functional=functional,
                            application=application,
                        )
                    )
                    continue
                if application.subject_ids != inner_fold.test_subject_ids:
                    raise ValueError(
                        "inner application subject losses do not match its fold"
                    )
                evaluations.append(
                    _record_subject_blocked_penalty_fold_evaluation(
                        candidate,
                        _producer_token=_WORKFLOW_SUBJECT_BLOCKED_PRODUCER_TOKEN,
                        inner_fold_id=inner_fold.fold_id,
                        inner_fold_manifest_id=inner_fold.fold_id,
                        training_subject_ids=inner_fold.train_subject_ids,
                        validation_subject_ids=application.subject_ids,
                        subject_losses=application.subject_full_losses,
                        scale_resolution_id=functional.penalty_scale_resolution_id,
                        resolved_penalty_id=functional.resolved_penalty_id,
                        resolved_lambda1=functional.lambda1,
                        resolved_lambda2=functional.lambda2,
                        training_functional_id=functional.incremental_functional_id,
                        heldout_application_id=application.application_id,
                    )
                )
            except ContractError as error:
                if error.details.code not in _INNER_STATISTICAL_CONTRACT_FAILURES:
                    raise
                evaluations.append(
                    _unavailable_inner_evaluation(
                        candidate,
                        inner_fold,
                        reason_code=_inner_failure_reason(
                            error,
                            stage=("fit" if functional is None else "apply"),
                        ),
                        functional=functional,
                        application=application,
                    )
                )
            except AutonomousProgramSupportError as error:
                evaluations.append(
                    _unavailable_inner_evaluation(
                        candidate,
                        inner_fold,
                        reason_code=_inner_failure_reason(
                            error,
                            stage=("fit" if functional is None else "apply"),
                        ),
                        functional=functional,
                        application=application,
                    )
                )
            except ValueError as error:
                classification = _classify_inner_value_error(error)
                if classification is None:
                    raise
                status, reason = classification
                evaluations.append(
                    _unavailable_inner_evaluation(
                        candidate,
                        inner_fold,
                        reason_code=reason,
                        functional=functional,
                        application=application,
                        status=status,
                    )
                )
    tuning = select_penalty_candidate(
        spec,
        evaluations,
        tuning_scope_id=tuning_scope_id,
        training_subject_ids=outer_subjects,
        inner_fold_ids=tuple(fold.fold_id for fold in plan.folds),
        inner_fold_plan_id=plan.plan_id,
    )
    return plan, tuning


def _validate_training_parents(
    encoder: FrozenDesignEncoder,
    response: FoldGeneResponseArtifact,
    precision: PrecisionTransformResult,
    receiver_family: ReceiverFamilyTrainingArtifact,
) -> FrozenDesignApplication:
    if not isinstance(encoder, FrozenDesignEncoder):
        raise TypeError("encoder must be a FrozenDesignEncoder")
    if not isinstance(response, FoldGeneResponseArtifact):
        raise TypeError("response must be a FoldGeneResponseArtifact")
    if not isinstance(precision, PrecisionTransformResult):
        raise TypeError("precision must be a PrecisionTransformResult")
    if not isinstance(receiver_family, ReceiverFamilyTrainingArtifact):
        raise TypeError("receiver_family must be a ReceiverFamilyTrainingArtifact")
    training_design = encoder.training_application()
    response.require_compatible(encoder)
    precision.require_response_compatible(response)
    receiver_family._require_producer_owned()
    family_basis = receiver_family.family_basis
    expected = (
        response.receiver,
        response.fold_id,
        response.feature_ids,
        response.training_subject_ids,
    )
    observed = (
        receiver_family.receiver,
        receiver_family.fold_id,
        family_basis.feature_ids,
        receiver_family.training_subject_ids,
    )
    if observed != expected:
        raise _parent_mismatch(
            "Receiver-family parent does not match the fold response",
            field="receiver_family_training_artifact_id",
        )
    if response.contrast_name != encoder.contrast.name:
        raise _parent_mismatch(
            "Response contrast does not match the frozen encoder",
            field="response_artifact_id",
        )
    return training_design


def _receiver_incremental_feature_scale(
    encoder: FrozenDesignEncoder,
    response: FoldGeneResponseArtifact,
    *,
    minimum_scale: float,
) -> np.ndarray:
    """Derive the exact outer-training scale used by incremental fitting."""

    if not math.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("minimum_scale must be finite and positive")
    negative_context_ids = {
        node_context_fields(node, encoder.context_keys)[0]
        for node, weight in encoder.contrast.weights.items()
        if weight < 0
    }
    reference = np.asarray(
        [
            context_id in negative_context_ids
            for context_id in response.sample_context_ids
        ],
        dtype=bool,
    )
    if int(np.count_nonzero(reference)) < 2:
        raise ValueError("reference response scale requires at least two samples")
    reference_by_subject = _subject_reference_summary(
        response.sample_values,
        reference=reference,
        subject_ids=response.sample_subject_ids,
        context_ids=response.sample_context_ids,
    )
    if len(reference_by_subject) < 2:
        raise ValueError("reference response scale requires at least two subjects")
    center = np.median(reference_by_subject, axis=0)
    mad = np.median(np.abs(reference_by_subject - center), axis=0)
    scale: np.ndarray = np.maximum(
        _MAD_GAUSSIAN_CONSISTENCY * mad, minimum_scale
    )
    return scale


def fit_receiver_incremental_training_artifact(
    encoder: FrozenDesignEncoder,
    response: FoldGeneResponseArtifact,
    precision: PrecisionTransformResult,
    receiver_family: ReceiverFamilyTrainingArtifact,
    autonomous_program_resource: ReceiverAutonomousProgramResource | None = None,
    *,
    minimum_scale: float = 0.25,
    null_loss_floor: float = 1e-8,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    penalty_tuning_spec: PenaltyTuningSpec | None = None,
    inner_partition_seed_lineage: SeedLineage | None = None,
) -> ReceiverIncrementalTrainingArtifact:
    """Fit a receiver-null model and optional conditional subject-blocked tuning.

    For fully paired allocations, inner folds refit response centering, scaling,
    nuisance coefficients and the signed residual solver, and resolve relative
    penalties from inner-training rows only. Precision, family construction and
    the encoder remain frozen from the complete outer-training fold;
    outer-heldout values are unavailable. Independent-group one-SE tuning fails
    closed until correlated pseudocontrast losses have a valid uncertainty rule.
    An explicit inner partition lineage changes only the donor allocation; the
    tuning and model identities remain bound to their exact outer-fold parents.
    """

    minimum_scale, null_loss_floor, lambda1, lambda2 = _validated_hyperparameters(
        minimum_scale=minimum_scale,
        null_loss_floor=null_loss_floor,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    if penalty_tuning_spec is not None:
        if not isinstance(penalty_tuning_spec, PenaltyTuningSpec):
            raise TypeError("penalty_tuning_spec must be a PenaltyTuningSpec")
        penalty_tuning_spec._require_intact()
        if lambda1 != 0.0 or lambda2 != 0.0:
            raise ValueError(
                "explicit lambda values cannot be combined with penalty_tuning_spec"
            )
    if inner_partition_seed_lineage is not None and not isinstance(
        inner_partition_seed_lineage, SeedLineage
    ):
        raise TypeError(
            "inner_partition_seed_lineage must be a SeedLineage or None"
        )
    if penalty_tuning_spec is None and inner_partition_seed_lineage is not None:
        raise ValueError(
            "inner_partition_seed_lineage requires an explicit penalty_tuning_spec"
        )
    training_design = _validate_training_parents(
        encoder, response, precision, receiver_family
    )
    declared_feature_scale: np.ndarray | None = None
    if precision.feature_scale.size:
        declared_feature_scale = _receiver_incremental_feature_scale(
            encoder,
            response,
            minimum_scale=minimum_scale,
        )
    precision.require_standardized_space_compatible(
        feature_scale=declared_feature_scale,
        residual_df=response.residual_df,
    )
    if autonomous_program_resource is not None:
        if not isinstance(
            autonomous_program_resource, ReceiverAutonomousProgramResource
        ):
            raise TypeError(
                "autonomous_program_resource must be a "
                "ReceiverAutonomousProgramResource"
            )
        autonomous_program_resource._require_producer_owned()
    basis = receiver_family.family_basis
    eligible_indices = np.flatnonzero(basis.family_eligible)
    family_ids = tuple(basis.family_ids[index] for index in eligible_indices)
    reason_code: str | None = None
    functional: IncrementalDownstreamFunctional | None = None
    inner_fold_plan: SubjectFoldPlan | None = None
    tuning_artifact: PenaltyTuningArtifact | None = None
    tuning_scope_id = (
        None
        if penalty_tuning_spec is None
        else _receiver_tuning_scope_id(
            encoder_id=encoder.encoder_id,
            response_artifact_id=response.artifact_id,
            precision_transform_id=precision.precision_transform_id,
            receiver_family_training_artifact_id=(
                receiver_family.training_artifact_id
            ),
            autonomous_program_resource_id=(
                None
                if autonomous_program_resource is None
                else autonomous_program_resource.artifact_id
            ),
            penalty_tuning_spec_id=penalty_tuning_spec.spec_id,
        )
    )
    if response.missing_sample_ids or (
        response.subject_ids != response.training_subject_ids
    ):
        reason_code = "training_receiver_response_incomplete_sample_coverage"
    elif response.status != "ok":
        reason_code = response.reason_code or "receiver_response_not_estimable"
    elif not precision.estimable:
        reason_code = precision.reason_code or "insufficient_response_precision_support"
    elif training_design.status != "observed":
        reason_code = training_design.reason_code or "training_design_not_estimable"
    elif not family_ids:
        reason_code = "no_training_eligible_receiver_family"
    if reason_code is not None and penalty_tuning_spec is not None:
        assert tuning_scope_id is not None
        tuning_artifact = not_estimable_penalty_tuning(
            penalty_tuning_spec,
            tuning_scope_id=tuning_scope_id,
            training_subject_ids=response.training_subject_ids,
            reason_code=reason_code,
        )
    elif reason_code is None:
        design_by_sample = {
            sample_id: index
            for index, sample_id in enumerate(training_design.sample_ids)
        }
        try:
            design_order = np.asarray(
                [design_by_sample[sample_id] for sample_id in response.sample_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise _parent_mismatch(
                "Response rows are absent from the training design",
                field="sample_ids",
            ) from error
        selected_design_ids = tuple(
            training_design.sample_ids[index] for index in design_order
        )
        if selected_design_ids != response.sample_ids:
            raise _parent_mismatch(
                "Response rows do not preserve the frozen design sample identity",
                field="sample_ids",
            )
        negative_context_ids = {
            node_context_fields(node, encoder.context_keys)[0]
            for node, weight in encoder.contrast.weights.items()
            if weight < 0
        }
        reference_mask = np.asarray(
            [
                training_design.sample_context_ids[index] in negative_context_ids
                for index in design_order
            ],
            dtype=bool,
        )
        inputs = _IncrementalFitInputs(
            response_matrix=response.sample_values,
            sample_ids=response.sample_ids,
            sample_subject_ids=response.sample_subject_ids,
            sample_context_ids=response.sample_context_ids,
            nuisance_matrix=training_design.nuisance_matrix[design_order],
            context_regressor=training_design.context_regressor[design_order],
            reference_mask=reference_mask,
            receiver=response.receiver,
            contrast_name=response.contrast_name,
            context_regressor_id=training_design.context_regressor_id,
            nuisance_design_id=training_design.nuisance_design_id,
            feature_ids=response.feature_ids,
            family_ids=family_ids,
            nuisance_column_ids=encoder.nuisance_column_ids,
            family_basis=basis.matrix[:, eligible_indices],
            autonomous_program_resource=autonomous_program_resource,
            precision_weights=precision.values,
            minimum_scale=minimum_scale,
            null_loss_floor=null_loss_floor,
        )
        try:
            if penalty_tuning_spec is None:
                functional = _fit_incremental_subset(
                    inputs,
                    subject_ids=response.training_subject_ids,
                    fold_id=response.fold_id,
                    lambda1=lambda1,
                    lambda2=lambda2,
                )
            else:
                assert tuning_scope_id is not None
                inner_fold_plan, tuning_artifact = (
                    _fit_subject_blocked_penalty_tuning(
                        inputs,
                        spec=penalty_tuning_spec,
                        tuning_scope_id=tuning_scope_id,
                        outer_fold_id=response.fold_id,
                        inner_partition_seed_lineage=(
                            inner_partition_seed_lineage
                        ),
                    )
                )
                selected = tuning_artifact.selected_candidate
                if selected is None:
                    reason_code = (
                        tuning_artifact.reason_code
                        or "penalty_tuning_not_estimable"
                    )
                else:
                    functional = _fit_incremental_subset(
                        inputs,
                        subject_ids=response.training_subject_ids,
                        fold_id=response.fold_id,
                        penalty_candidate=selected,
                    )
                    lambda1 = functional.lambda1
                    lambda2 = functional.lambda2
        except AutonomousProgramSupportError as error:
            reason_code = error.reason_code
        except ContractError as error:
            if error.details.code not in {
                "all_family_bases_not_identifiable_after_autonomous_projection",
                "penalty_scale_not_estimable",
            }:
                raise
            reason_code = error.details.code
        except ValueError as error:
            if penalty_tuning_spec is None:
                raise
            classification = _classify_inner_value_error(error)
            if classification is None:
                raise
            _, classified_reason = classification
            functional = None
            reason_code = f"selected_penalty_final_{classified_reason}"
    if penalty_tuning_spec is not None and tuning_artifact is None:
        assert tuning_scope_id is not None
        tuning_artifact = not_estimable_penalty_tuning(
            penalty_tuning_spec,
            tuning_scope_id=tuning_scope_id,
            training_subject_ids=response.training_subject_ids,
            reason_code=reason_code or "penalty_tuning_not_estimable",
        )
    return ReceiverIncrementalTrainingArtifact._from_fit(
        encoder=encoder,
        training_design=training_design,
        response=response,
        precision=precision,
        receiver_family=receiver_family,
        autonomous_program_resource=autonomous_program_resource,
        penalty_tuning_spec=penalty_tuning_spec,
        inner_fold_plan=inner_fold_plan,
        penalty_tuning_artifact=tuning_artifact,
        family_ids=family_ids,
        minimum_scale=minimum_scale,
        null_loss_floor=null_loss_floor,
        lambda1=lambda1,
        lambda2=lambda2,
        diagnostic_functional=functional,
        diagnostic_reason_code=reason_code,
    )


def apply_receiver_incremental_training_artifact(
    model: ReceiverIncrementalTrainingArtifact,
    response_application: FoldGeneResponseApplication,
    design_application: FrozenDesignApplication,
) -> ReceiverIncrementalApplication:
    """Apply a frozen diagnostic to exact typed held-out parents without fitting."""

    if not isinstance(model, ReceiverIncrementalTrainingArtifact):
        raise TypeError("model must be a ReceiverIncrementalTrainingArtifact")
    if not isinstance(response_application, FoldGeneResponseApplication):
        raise TypeError("response_application must be a FoldGeneResponseApplication")
    if not isinstance(design_application, FrozenDesignApplication):
        raise TypeError("design_application must be a FrozenDesignApplication")
    _validate_heldout_parents(model, response_application, design_application)

    diagnostic: IncrementalDownstreamApplication | None = None
    diagnostic_reason: str | None = None
    if model.diagnostic_functional is None:
        diagnostic_reason = (
            model.diagnostic_reason_code or "incremental_training_not_estimable"
        )
    elif design_application.status != "observed":
        diagnostic_reason = (
            design_application.reason_code or "heldout_design_not_estimable"
        )
    elif response_application.status != "ok":
        diagnostic_reason = (
            response_application.reason_code or "heldout_response_not_estimable"
        )
    else:
        row_manifest = DownstreamRowManifest(
            sample_ids=response_application.sample_ids,
            subject_ids=response_application.sample_subject_ids,
            context_ids=response_application.sample_context_ids,
        )
        diagnostic = apply_incremental_downstream_functional(
            model.diagnostic_functional,
            response_application.sample_values,
            row_manifest=row_manifest,
            design_sample_ids=design_application.sample_ids,
            nuisance_matrix=design_application.nuisance_matrix,
            context_regressor=design_application.context_regressor,
            context_regressor_id=design_application.context_regressor_id,
            nuisance_design_id=design_application.nuisance_design_id,
            feature_ids=response_application.feature_ids,
            nuisance_column_ids=model.diagnostic_functional.nuisance_column_ids,
        )
        if diagnostic.status != "observed":
            diagnostic_reason = (
                diagnostic.reason_code or "incremental_application_not_estimable"
            )
    return ReceiverIncrementalApplication._from_application(
        model=model,
        response_application=response_application,
        design_application=design_application,
        diagnostic_application=diagnostic,
        diagnostic_reason_code=diagnostic_reason,
    )


__all__ = [
    "ReceiverIncrementalApplication",
    "ReceiverIncrementalTrainingArtifact",
    "apply_receiver_incremental_training_artifact",
    "fit_receiver_incremental_training_artifact",
]
