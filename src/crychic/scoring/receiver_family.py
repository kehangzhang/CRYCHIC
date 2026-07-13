"""Frozen receiver-family programs for held-out expression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crychic.attribution.frozen_family import ReceiverFamilyTrainingArtifact
from crychic.core import ContractError, stable_id

from .downstream import (
    DownstreamApplication,
    DownstreamFunctional,
    apply_downstream_functional,
    fit_downstream_functional,
)

_PRODUCER_MARKER = "crychic.receiver_family_scoring.v1"
_TRAINING_STATUS = "training_only_partial_receiver_family_downstream_v1"
_NOT_ESTIMABLE_STATUS = "training_only_partial_receiver_family_not_estimable_v1"
_APPLICATION_STATUS = "frozen_application_partial_not_oof"
_APPLICATION_NOT_ESTIMABLE = "frozen_application_partial_not_estimable"
_REMAINING_STAGES = (
    "response_precision",
    "family_attribution",
    "attribution_tuning",
    "incremental_downstream",
    "common_scoring_functional",
)


def _sample_subjects(
    values: tuple[str, ...], *, n_samples: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(values) != n_samples:
        raise ValueError(
            "sample_subject_ids must align one-to-one with expression rows"
        )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("sample_subject_ids must contain non-empty strings")
    aligned = tuple(value.strip() for value in values)
    return aligned, tuple(sorted(set(aligned)))


@dataclass(frozen=True, slots=True, init=False)
class ReceiverFamilyScoringArtifact:
    """Producer-owned frozen receiver program with explicitly partial status."""

    receiver_family_artifact: ReceiverFamilyTrainingArtifact
    contrast_name: str
    reference_sample_ids: tuple[str, ...]
    reference_subject_ids: tuple[str, ...]
    active_family_ids: tuple[str, ...]
    downstream_functional: DownstreamFunctional | None
    completed_stages: tuple[str, ...]
    remaining_stages: tuple[str, ...]
    reason_code: str | None
    certification_status: str
    training_artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverFamilyScoringArtifact is producer-owned; "
            "use fit_receiver_family_scoring_artifact()"
        )

    @classmethod
    def _from_training(
        cls,
        *,
        receiver_family_artifact: ReceiverFamilyTrainingArtifact,
        contrast_name: str,
        reference_sample_ids: tuple[str, ...],
        reference_subject_ids: tuple[str, ...],
        active_family_ids: tuple[str, ...],
        downstream_functional: DownstreamFunctional | None,
        reason_code: str | None,
    ) -> ReceiverFamilyScoringArtifact:
        receiver_family_artifact._require_producer_owned()
        has_functional = downstream_functional is not None
        if has_functional == (reason_code is not None):
            raise ValueError(
                "reason_code must be present exactly when downstream is unavailable"
            )
        if has_functional:
            assert downstream_functional is not None
            if downstream_functional.family_ids != active_family_ids:
                raise ValueError("downstream families do not match eligible families")
            if (
                downstream_functional.receiver != receiver_family_artifact.receiver
                or downstream_functional.fold_id != receiver_family_artifact.fold_id
            ):
                raise ValueError("downstream provenance does not match receiver family")
            completed = (
                *receiver_family_artifact.completed_stages,
                "receiver_program_reference_transform",
            )
            status = _TRAINING_STATUS
            downstream_id: str | None = downstream_functional.downstream_functional_id
        else:
            completed = receiver_family_artifact.completed_stages
            status = _NOT_ESTIMABLE_STATUS
            downstream_id = None
        artifact_id = stable_id(
            "receiver_family_scoring_artifact",
            {
                "active_family_ids": list(active_family_ids),
                "certification_status": status,
                "contrast_name": contrast_name,
                "downstream_functional_id": downstream_id,
                "reason_code": reason_code,
                "receiver_family_training_artifact_id": (
                    receiver_family_artifact.training_artifact_id
                ),
                "reference_subject_ids": list(reference_subject_ids),
                "reference_sample_ids": list(reference_sample_ids),
                "reference_input_digest": (
                    None
                    if downstream_functional is None
                    else downstream_functional.reference_input_digest
                ),
            },
        )
        self = object.__new__(cls)
        values: dict[str, Any] = {
            "receiver_family_artifact": receiver_family_artifact,
            "contrast_name": contrast_name,
            "reference_sample_ids": reference_sample_ids,
            "reference_subject_ids": reference_subject_ids,
            "active_family_ids": active_family_ids,
            "downstream_functional": downstream_functional,
            "completed_stages": completed,
            "remaining_stages": _REMAINING_STAGES,
            "reason_code": reason_code,
            "certification_status": status,
            "training_artifact_id": artifact_id,
            "_producer_marker": _PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    @property
    def training_subject_ids(self) -> tuple[str, ...]:
        """Return every subject consulted by the receptor-family stage."""

        subjects: tuple[str, ...] = self.receiver_family_artifact.training_subject_ids
        return subjects

    @property
    def is_oof_certified(self) -> bool:
        """Return false until response, attribution, and common scoring are frozen."""

        return False

    def _require_producer_owned(self) -> None:
        if self._producer_marker != _PRODUCER_MARKER:
            raise TypeError("receiver-family scoring artifact is not producer-owned")
        self._require_intact()

    def _identity_payload(self) -> dict[str, object]:
        return {
            "active_family_ids": list(self.active_family_ids),
            "certification_status": self.certification_status,
            "contrast_name": self.contrast_name,
            "downstream_functional_id": (
                None
                if self.downstream_functional is None
                else self.downstream_functional.downstream_functional_id
            ),
            "reason_code": self.reason_code,
            "receiver_family_training_artifact_id": (
                self.receiver_family_artifact.training_artifact_id
            ),
            "reference_subject_ids": list(self.reference_subject_ids),
            "reference_sample_ids": list(self.reference_sample_ids),
            "reference_input_digest": (
                None
                if self.downstream_functional is None
                else self.downstream_functional.reference_input_digest
            ),
        }

    def _require_intact(self) -> None:
        """Reject forced mutation of receiver-family scoring lineage."""

        try:
            self.receiver_family_artifact._require_producer_owned()
            if not isinstance(self.contrast_name, str) or not self.contrast_name:
                raise ValueError("invalid contrast name")
            active = tuple(self.active_family_ids)
            samples = tuple(self.reference_sample_ids)
            subjects = tuple(self.reference_subject_ids)
            if (
                not isinstance(self.active_family_ids, tuple)
                or not isinstance(self.reference_sample_ids, tuple)
                or not isinstance(self.reference_subject_ids, tuple)
                or len(active) != len(set(active))
                or any(not value for value in (*active, *samples, *subjects))
            ):
                raise ValueError("invalid receiver-family identifiers")
            has_functional = self.downstream_functional is not None
            if has_functional == (self.reason_code is not None):
                raise ValueError("invalid receiver-family scoring status")
            if has_functional:
                assert self.downstream_functional is not None
                self.downstream_functional._require_intact()
                if (
                    self.downstream_functional.family_ids != active
                    or self.downstream_functional.receiver
                    != self.receiver_family_artifact.receiver
                    or self.downstream_functional.fold_id
                    != self.receiver_family_artifact.fold_id
                ):
                    raise ValueError("downstream functional lineage mismatch")
                expected_completed = (
                    *self.receiver_family_artifact.completed_stages,
                    "receiver_program_reference_transform",
                )
                expected_status = _TRAINING_STATUS
            else:
                expected_completed = self.receiver_family_artifact.completed_stages
                expected_status = _NOT_ESTIMABLE_STATUS
            valid = (
                self.completed_stages == expected_completed
                and self.remaining_stages == _REMAINING_STAGES
                and self.certification_status == expected_status
                and stable_id(
                    "receiver_family_scoring_artifact",
                    self._identity_payload(),
                )
                == self.training_artifact_id
                and not self.is_oof_certified
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver-family scoring artifact failed integrity validation",
                code="receiver_family_scoring_integrity_violation",
                field="training_artifact_id",
                remediation="Refit receiver-family scoring from intact parents",
            ) from error
        if not valid:
            raise ContractError(
                "Receiver-family scoring artifact failed integrity validation",
                code="receiver_family_scoring_integrity_violation",
                field="training_artifact_id",
                remediation="Refit receiver-family scoring from intact parents",
            )


def mark_receiver_family_scoring_not_estimable(
    receiver_family_artifact: ReceiverFamilyTrainingArtifact,
    *,
    contrast_name: str,
    reason_code: str,
) -> ReceiverFamilyScoringArtifact:
    """Preserve a planned receiver model that lacks training reference support."""

    if not isinstance(receiver_family_artifact, ReceiverFamilyTrainingArtifact):
        raise TypeError(
            "receiver_family_artifact must be a ReceiverFamilyTrainingArtifact"
        )
    receiver_family_artifact._require_producer_owned()
    if not isinstance(contrast_name, str) or not contrast_name.strip():
        raise ValueError("contrast_name must be a non-empty string")
    if not isinstance(reason_code, str) or not reason_code.strip():
        raise ValueError("reason_code must be a non-empty string")
    return ReceiverFamilyScoringArtifact._from_training(
        receiver_family_artifact=receiver_family_artifact,
        contrast_name=contrast_name.strip(),
        reference_sample_ids=(),
        reference_subject_ids=receiver_family_artifact.training_subject_ids,
        active_family_ids=receiver_family_artifact.eligible_family_ids,
        downstream_functional=None,
        reason_code=reason_code.strip(),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReceiverFamilyScoringApplication:
    """Held-out receiver-family programs produced without any refitting."""

    training_artifact_id: str
    heldout_subject_ids: tuple[str, ...]
    active_family_ids: tuple[str, ...]
    downstream_application: DownstreamApplication | None
    reason_code: str | None
    application_status: str
    application_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.training_artifact_id:
            raise ValueError("training_artifact_id must not be empty")
        if not self.heldout_subject_ids:
            raise ValueError("heldout_subject_ids must not be empty")
        observed = self.downstream_application is not None
        if observed == (self.reason_code is not None):
            raise ValueError(
                "reason_code must be present exactly when application is unavailable"
            )
        expected_status = (
            _APPLICATION_STATUS if observed else _APPLICATION_NOT_ESTIMABLE
        )
        if self.application_status != expected_status:
            raise ValueError(
                "application status does not match downstream availability"
            )
        subjects = tuple(sorted(set(self.heldout_subject_ids)))
        active = tuple(self.active_family_ids)
        if (
            not subjects
            or len(active) != len(set(active))
            or any(not isinstance(value, str) or not value.strip() for value in active)
        ):
            raise ValueError("receiver-family application identifiers are invalid")
        if observed:
            assert self.downstream_application is not None
            self.downstream_application._require_intact()
            if self.downstream_application.raw_program.shape[1] != len(active):
                raise ValueError(
                    "downstream application columns do not match active families"
                )
        object.__setattr__(self, "heldout_subject_ids", subjects)
        object.__setattr__(self, "active_family_ids", active)
        object.__setattr__(
            self,
            "application_id",
            stable_id("receiver_family_scoring_application", self._identity_payload()),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "active_family_ids": list(self.active_family_ids),
            "application_status": self.application_status,
            "downstream_application_id": (
                None
                if self.downstream_application is None
                else self.downstream_application.application_id
            ),
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "reason_code": self.reason_code,
            "training_artifact_id": self.training_artifact_id,
        }

    def _require_intact(self) -> None:
        """Reject forced mutation of held-out receiver-family values."""

        try:
            observed = self.downstream_application is not None
            if observed:
                assert self.downstream_application is not None
                self.downstream_application._require_intact()
            expected_status = (
                _APPLICATION_STATUS if observed else _APPLICATION_NOT_ESTIMABLE
            )
            valid = (
                bool(self.training_artifact_id)
                and isinstance(self.heldout_subject_ids, tuple)
                and isinstance(self.active_family_ids, tuple)
                and self.heldout_subject_ids
                == tuple(sorted(set(self.heldout_subject_ids)))
                and len(self.active_family_ids) == len(set(self.active_family_ids))
                and observed == (self.reason_code is None)
                and self.application_status == expected_status
                and (
                    not observed
                    or (
                        self.downstream_application is not None
                        and self.downstream_application.raw_program.shape[1]
                        == len(self.active_family_ids)
                    )
                )
                and stable_id(
                    "receiver_family_scoring_application",
                    self._identity_payload(),
                )
                == self.application_id
                and not self.is_oof_certified
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver-family application failed integrity validation",
                code="receiver_family_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact receiver-family scoring artifact",
            ) from error
        if not valid:
            raise ContractError(
                "Receiver-family application failed integrity validation",
                code="receiver_family_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact receiver-family scoring artifact",
            )

    @property
    def is_oof_certified(self) -> bool:
        """Return false because this application covers only a partial stage."""

        return False


def fit_receiver_family_scoring_artifact(
    receiver_family_artifact: ReceiverFamilyTrainingArtifact,
    reference_expression: np.ndarray,
    *,
    contrast_name: str,
    sample_ids: tuple[str, ...],
    sample_subject_ids: tuple[str, ...],
    minimum_scale: float = 0.25,
) -> ReceiverFamilyScoringArtifact:
    """Fit a family program transform using training-reference expression only."""

    if not isinstance(receiver_family_artifact, ReceiverFamilyTrainingArtifact):
        raise TypeError(
            "receiver_family_artifact must be a ReceiverFamilyTrainingArtifact"
        )
    receiver_family_artifact._require_producer_owned()
    if not isinstance(contrast_name, str) or not contrast_name.strip():
        raise ValueError("contrast_name must be a non-empty string")
    expression = np.asarray(reference_expression, dtype=np.float64)
    if expression.ndim != 2:
        raise ValueError("reference_expression must be a two-dimensional matrix")
    if expression.shape[1] != len(receiver_family_artifact.source_basis.feature_ids):
        raise ValueError("reference_expression must align with frozen features")
    if expression.shape[0] < 2 or np.any(~np.isfinite(expression)):
        raise ValueError(
            "reference_expression requires at least two complete finite samples"
        )
    aligned_subjects, reference_subjects = _sample_subjects(
        sample_subject_ids, n_samples=expression.shape[0]
    )
    if len(sample_ids) != expression.shape[0]:
        raise ValueError("sample_ids must align one-to-one with expression rows")
    unknown = set(reference_subjects).difference(
        receiver_family_artifact.training_subject_ids
    )
    if unknown:
        raise ValueError(
            "reference expression contains subjects outside the training fold: "
            + ", ".join(sorted(unknown))
        )
    basis = receiver_family_artifact.family_basis
    active_indices = np.flatnonzero(basis.family_eligible)
    active_family_ids = tuple(basis.family_ids[index] for index in active_indices)
    if len(active_indices) == 0:
        return ReceiverFamilyScoringArtifact._from_training(
            receiver_family_artifact=receiver_family_artifact,
            contrast_name=contrast_name.strip(),
            reference_sample_ids=tuple(sample_ids),
            reference_subject_ids=reference_subjects,
            active_family_ids=(),
            downstream_functional=None,
            reason_code="no_training_eligible_receiver_family",
        )
    target_weights = basis.matrix[:, active_indices]
    downstream = fit_downstream_functional(
        expression,
        receiver=receiver_family_artifact.receiver,
        contrast_name=contrast_name.strip(),
        fold_id=receiver_family_artifact.fold_id,
        feature_ids=basis.feature_ids,
        family_ids=active_family_ids,
        reference_sample_ids=tuple(sample_ids),
        reference_subject_ids=aligned_subjects,
        training_subject_ids=reference_subjects,
        target_weight_matrix=target_weights,
        family_support=np.ones(len(active_family_ids), dtype=np.float64),
        minimum_scale=minimum_scale,
    )
    return ReceiverFamilyScoringArtifact._from_training(
        receiver_family_artifact=receiver_family_artifact,
        contrast_name=contrast_name.strip(),
        reference_sample_ids=downstream.reference_sample_ids,
        reference_subject_ids=reference_subjects,
        active_family_ids=active_family_ids,
        downstream_functional=downstream,
        reason_code=None,
    )


def apply_receiver_family_scoring_artifact(
    artifact: ReceiverFamilyScoringArtifact,
    sample_expression: np.ndarray,
    *,
    feature_ids: tuple[str, ...],
    sample_subject_ids: tuple[str, ...],
) -> ReceiverFamilyScoringApplication:
    """Apply one frozen receiver-family program to held-out subjects."""

    if not isinstance(artifact, ReceiverFamilyScoringArtifact):
        raise TypeError("artifact must be a ReceiverFamilyScoringArtifact")
    artifact._require_producer_owned()
    expression = np.asarray(sample_expression, dtype=np.float64)
    if expression.ndim == 1:
        expression = expression.reshape(1, -1)
    if expression.ndim != 2 or expression.shape[1] != len(feature_ids):
        raise ValueError("sample_expression must be samples x declared features")
    _, heldout_subjects = _sample_subjects(
        sample_subject_ids, n_samples=expression.shape[0]
    )
    overlap = set(heldout_subjects).intersection(artifact.training_subject_ids)
    if overlap:
        raise ValueError(
            "held-out expression overlaps training subjects: "
            + ", ".join(sorted(overlap))
        )
    if tuple(feature_ids) != artifact.receiver_family_artifact.source_basis.feature_ids:
        raise ValueError("test feature_ids must exactly match frozen features")
    if artifact.downstream_functional is None:
        return ReceiverFamilyScoringApplication(
            training_artifact_id=artifact.training_artifact_id,
            heldout_subject_ids=heldout_subjects,
            active_family_ids=artifact.active_family_ids,
            downstream_application=None,
            reason_code=artifact.reason_code,
            application_status=_APPLICATION_NOT_ESTIMABLE,
        )
    application = apply_downstream_functional(
        artifact.downstream_functional,
        expression,
        feature_ids=feature_ids,
    )
    return ReceiverFamilyScoringApplication(
        training_artifact_id=artifact.training_artifact_id,
        heldout_subject_ids=heldout_subjects,
        active_family_ids=artifact.active_family_ids,
        downstream_application=application,
        reason_code=None,
        application_status=_APPLICATION_STATUS,
    )


def mark_receiver_family_application_not_estimable(
    artifact: ReceiverFamilyScoringArtifact,
    *,
    heldout_subject_ids: tuple[str, ...],
    reason_code: str,
) -> ReceiverFamilyScoringApplication:
    """Preserve a planned held-out receiver lacking sample-expression support."""

    if not isinstance(artifact, ReceiverFamilyScoringArtifact):
        raise TypeError("artifact must be a ReceiverFamilyScoringArtifact")
    artifact._require_producer_owned()
    subjects = tuple(sorted(set(heldout_subject_ids)))
    if not subjects or any(not value.strip() for value in subjects):
        raise ValueError("heldout_subject_ids must contain non-empty strings")
    overlap = set(subjects).intersection(artifact.training_subject_ids)
    if overlap:
        raise ValueError(
            "held-out subjects overlap training subjects: " + ", ".join(sorted(overlap))
        )
    if not isinstance(reason_code, str) or not reason_code.strip():
        raise ValueError("reason_code must be a non-empty string")
    return ReceiverFamilyScoringApplication(
        training_artifact_id=artifact.training_artifact_id,
        heldout_subject_ids=subjects,
        active_family_ids=artifact.active_family_ids,
        downstream_application=None,
        reason_code=reason_code.strip(),
        application_status=_APPLICATION_NOT_ESTIMABLE,
    )


__all__ = [
    "ReceiverFamilyScoringApplication",
    "ReceiverFamilyScoringArtifact",
    "apply_receiver_family_scoring_artifact",
    "fit_receiver_family_scoring_artifact",
    "mark_receiver_family_application_not_estimable",
    "mark_receiver_family_scoring_not_estimable",
]
