"""Typed orchestration for receiver incremental-downstream diagnostics.

The low-level incremental model is intentionally exposed here only through
producer-owned design, response, precision, and receiver-family parents.  The
result remains diagnostic until an autonomous nuisance basis is frozen inside
the training fold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from crychic.attribution import PrecisionTransformResult, ReceiverFamilyTrainingArtifact
from crychic.core import ContractError, stable_id
from crychic.design import (
    FrozenDesignApplication,
    FrozenDesignEncoder,
    node_context_fields,
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

_PRODUCER_MARKER = "crychic.workflow.receiver_incremental_training.v1"
_APPLICATION_PRODUCER_MARKER = "crychic.workflow.receiver_incremental_application.v1"
_OFFICIAL_STATUS = "not_estimable"
_FORMULA_CERTIFICATION_STATUS = "formula_nuisance_incremental_diagnostic_only"
_FORMULA_REASON = "receiver_autonomous_nuisance_not_frozen"


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
    minimum_scale: float
    null_loss_floor: float
    lambda1: float
    lambda2: float
    diagnostic_status: str
    diagnostic_reason_code: str | None
    diagnostic_functional: IncrementalDownstreamFunctional | None
    certification_status: str
    official_incremental_status: str
    reason_code: str
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
            "minimum_scale": float(minimum_scale),
            "null_loss_floor": float(null_loss_floor),
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "diagnostic_status": diagnostic_status,
            "diagnostic_reason_code": diagnostic_reason_code,
            "diagnostic_functional": diagnostic_functional,
            "certification_status": _FORMULA_CERTIFICATION_STATUS,
            "official_incremental_status": _OFFICIAL_STATUS,
            "reason_code": _FORMULA_REASON,
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
                schema_version="1",
            ),
        )
        return self

    @property
    def is_oof_certified(self) -> bool:
        """Return false until the autonomous nuisance model is fold-frozen."""

        return False

    def _identity_payload(self) -> dict[str, object]:
        functional_id = (
            None
            if self.diagnostic_functional is None
            else self.diagnostic_functional.incremental_functional_id
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
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "minimum_scale": self.minimum_scale,
            "nuisance_column_ids": list(self.nuisance_column_ids),
            "nuisance_design_id": self.nuisance_design_id,
            "null_loss_floor": self.null_loss_floor,
            "official_incremental_status": self.official_incremental_status,
            "precision_transform_id": self.precision_transform_id,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "receiver_family_training_artifact_id": (
                self.receiver_family_training_artifact_id
            ),
            "response_artifact_id": self.response_artifact_id,
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
            }:
                raise ValueError("unsupported autonomous resource verification status")
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
            elif (
                self.diagnostic_status != "not_estimable"
                or not self.diagnostic_reason_code
            ):
                raise ValueError("not-estimable diagnostic status is inconsistent")
            elif self.autonomous_projection_id is not None:
                raise ValueError(
                    "unavailable diagnostics cannot claim an autonomous projection"
                )
            expected = stable_id(
                "receiver_incremental_training_artifact",
                self._identity_payload(),
                schema_version="1",
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and self.certification_status == _FORMULA_CERTIFICATION_STATUS
                and self.official_incremental_status == _OFFICIAL_STATUS
                and self.reason_code == _FORMULA_REASON
                and not self.is_oof_certified
                and expected == self.training_artifact_id
            )
        except (AttributeError, TypeError, ValueError) as error:
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
    reason_code: str
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
            "certification_status": model.certification_status,
            "official_incremental_status": _OFFICIAL_STATUS,
            "reason_code": model.reason_code,
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
                schema_version="1",
            ),
        )
        return self

    @property
    def is_oof_certified(self) -> bool:
        """Return false because the fitted nuisance basis is not autonomous."""

        return False

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
                schema_version="1",
            )
            valid = (
                self._producer_marker == _APPLICATION_PRODUCER_MARKER
                and self.certification_status == _FORMULA_CERTIFICATION_STATUS
                and self.official_incremental_status == _OFFICIAL_STATUS
                and self.reason_code == _FORMULA_REASON
                and not self.is_oof_certified
                and expected == self.application_id
            )
        except (AttributeError, TypeError, ValueError) as error:
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
) -> ReceiverIncrementalTrainingArtifact:
    """Fit a receiver-null diagnostic from exact typed training parents."""

    minimum_scale, null_loss_floor, lambda1, lambda2 = _validated_hyperparameters(
        minimum_scale=minimum_scale,
        null_loss_floor=null_loss_floor,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    training_design = _validate_training_parents(
        encoder, response, precision, receiver_family
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
    else:
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
        row_manifest = DownstreamRowManifest(
            sample_ids=response.sample_ids,
            subject_ids=response.sample_subject_ids,
            context_ids=response.sample_context_ids,
        )
        try:
            functional = fit_incremental_downstream_functional(
                response.sample_values,
                row_manifest=row_manifest,
                design_sample_ids=selected_design_ids,
                reference_mask=reference_mask,
                nuisance_matrix=training_design.nuisance_matrix[design_order],
                context_regressor=training_design.context_regressor[design_order],
                receiver=response.receiver,
                contrast_name=response.contrast_name,
                fold_id=response.fold_id,
                context_regressor_id=training_design.context_regressor_id,
                nuisance_design_id=training_design.nuisance_design_id,
                feature_ids=response.feature_ids,
                family_ids=family_ids,
                nuisance_column_ids=encoder.nuisance_column_ids,
                training_subject_ids=response.training_subject_ids,
                family_basis=basis.matrix[:, eligible_indices],
                autonomous_program_resource=autonomous_program_resource,
                precision_weights=precision.values,
                minimum_scale=minimum_scale,
                null_loss_floor=null_loss_floor,
                lambda1=lambda1,
                lambda2=lambda2,
            )
        except AutonomousProgramSupportError as error:
            reason_code = error.reason_code
        except ContractError as error:
            if error.details.code != (
                "all_family_bases_not_identifiable_after_autonomous_projection"
            ):
                raise
            reason_code = error.details.code
    return ReceiverIncrementalTrainingArtifact._from_fit(
        encoder=encoder,
        training_design=training_design,
        response=response,
        precision=precision,
        receiver_family=receiver_family,
        autonomous_program_resource=autonomous_program_resource,
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
