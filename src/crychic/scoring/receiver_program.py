"""Source-agnostic receiver target-program scoring.

Receiver programs describe whether a receiver expresses a frozen family target
program.  They are deliberately independent of receptor eligibility, sender
assignment, and incremental edge evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from crychic.attribution import ReceiverFamilyTrainingArtifact
from crychic.core import ContractError, stable_id

from .contracts import float64_array_digest
from .downstream import (
    DownstreamApplication,
    DownstreamFunctional,
    apply_downstream_functional,
    downstream_input_expression_digest,
    fit_downstream_functional,
)

RECEIVER_PROGRAM_SCORE_COLUMNS = (
    "receiver_program_training_artifact_id",
    "receiver_program_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "receiver_program_score",
    "status",
    "reason_code",
)

_TRAINING_PRODUCER = "crychic.receiver_program_training.v2"
_APPLICATION_PRODUCER = "crychic.receiver_program_application.v2"
_TRAINING_OBSERVED = "training_fold_frozen_source_agnostic_receiver_program_v1"
_TRAINING_NOT_ESTIMABLE = (
    "training_fold_source_agnostic_receiver_program_not_estimable_v1"
)
_APPLICATION_OBSERVED = "heldout_source_agnostic_receiver_program_observed_v1"
_APPLICATION_NOT_ESTIMABLE = "heldout_source_agnostic_receiver_program_not_estimable_v1"
_TARGET_PROFILE_REASON = "target_profile_not_estimable"


def _required_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _aligned_names(
    values: tuple[str, ...], *, length: int, field_name: str
) -> tuple[str, ...]:
    if len(values) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    return tuple(_required_name(value, field_name=field_name) for value in values)


def _row_manifest_id(
    sample_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> str:
    result: str = stable_id(
        "receiver_program_row_manifest",
        {
            "rows": [
                {
                    "context_id": context_id,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                }
                for sample_id, subject_id, context_id in zip(
                    sample_ids, subject_ids, context_ids, strict=True
                )
            ]
        },
        schema_version="1",
    )
    return result


def _heldout_input_digest(
    expression_digest: str,
    *,
    feature_ids: tuple[str, ...],
    row_manifest_id: str,
) -> str:
    normalized_expression_digest = _required_name(
        expression_digest, field_name="expression_digest"
    )
    result: str = stable_id(
        "receiver_program_heldout_input",
        {
            "expression_digest": normalized_expression_digest,
            "feature_ids": list(feature_ids),
            "row_manifest_id": row_manifest_id,
        },
        schema_version="1",
    )
    return result


def _canonical_heldout_rows(
    sample_expression: np.ndarray,
    *,
    sample_ids: tuple[str, ...],
    sample_subject_ids: tuple[str, ...],
    sample_context_ids: tuple[str, ...],
    n_features: int,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    str,
]:
    expression = np.asarray(sample_expression, dtype=np.float64)
    if expression.ndim == 1:
        expression = expression.reshape(1, -1)
    if (
        expression.ndim != 2
        or expression.shape[1] != n_features
        or np.any(~np.isfinite(expression))
    ):
        raise ValueError(
            "sample_expression must be a complete finite samples x features matrix"
        )
    samples = _aligned_names(
        tuple(sample_ids), length=expression.shape[0], field_name="sample_ids"
    )
    if len(set(samples)) != len(samples):
        raise ValueError("sample_ids must be unique")
    subjects = _aligned_names(
        tuple(sample_subject_ids),
        length=len(samples),
        field_name="sample_subject_ids",
    )
    contexts = _aligned_names(
        tuple(sample_context_ids),
        length=len(samples),
        field_name="sample_context_ids",
    )
    order = np.asarray(sorted(range(len(samples)), key=samples.__getitem__), dtype=int)
    canonical_samples = tuple(samples[index] for index in order)
    canonical_subjects = tuple(subjects[index] for index in order)
    canonical_contexts = tuple(contexts[index] for index in order)
    canonical_expression = np.asarray(expression[order], dtype=np.float64, order="C")
    manifest_id = _row_manifest_id(
        canonical_samples, canonical_subjects, canonical_contexts
    )
    return (
        canonical_expression,
        canonical_samples,
        canonical_subjects,
        canonical_contexts,
        manifest_id,
    )


def _canonical_declared_rows(
    *,
    sample_ids: tuple[str, ...],
    sample_subject_ids: tuple[str, ...],
    sample_context_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    samples = _aligned_names(
        tuple(sample_ids), length=len(sample_ids), field_name="sample_ids"
    )
    if not samples or len(set(samples)) != len(samples):
        raise ValueError("sample_ids must be non-empty and unique")
    subjects = _aligned_names(
        tuple(sample_subject_ids),
        length=len(samples),
        field_name="sample_subject_ids",
    )
    contexts = _aligned_names(
        tuple(sample_context_ids),
        length=len(samples),
        field_name="sample_context_ids",
    )
    order = sorted(range(len(samples)), key=samples.__getitem__)
    canonical_samples = tuple(samples[index] for index in order)
    canonical_subjects = tuple(subjects[index] for index in order)
    canonical_contexts = tuple(contexts[index] for index in order)
    return (
        canonical_samples,
        canonical_subjects,
        canonical_contexts,
        _row_manifest_id(canonical_samples, canonical_subjects, canonical_contexts),
    )


def _target_profile_matrix(
    receiver_family: ReceiverFamilyTrainingArtifact,
) -> tuple[
    tuple[str, ...],
    tuple[bool, ...],
    tuple[str | None, ...],
    sparse.csc_matrix,
]:
    source = receiver_family.source_basis
    family_basis = receiver_family.family_basis
    driver_index = {
        driver_id: index for index, driver_id in enumerate(source.driver_ids)
    }
    estimable: list[bool] = []
    reasons: list[str | None] = []
    columns: list[sparse.csc_matrix] = []
    estimable_family_ids: list[str] = []
    for family_id, medoid in zip(
        family_basis.family_ids, family_basis.medoid_driver_ids, strict=True
    ):
        profile = source.normalized_profiles.getcol(driver_index[medoid]).tocsc()
        profile.sum_duplicates()
        profile.sort_indices()
        has_profile = bool(profile.nnz) and float(profile.sum()) > 0.0
        estimable.append(has_profile)
        reasons.append(None if has_profile else _TARGET_PROFILE_REASON)
        if has_profile:
            estimable_family_ids.append(family_id)
            columns.append(profile)
    matrix = (
        sparse.hstack(columns, format="csc")
        if columns
        else sparse.csc_matrix((len(source.feature_ids), 0), dtype=np.float64)
    )
    return (
        tuple(estimable_family_ids),
        tuple(estimable),
        tuple(reasons),
        matrix,
    )


def _normalized_target_profile_matrix(
    matrix: sparse.csc_matrix,
) -> sparse.csc_matrix:
    result = sparse.csc_matrix(matrix, dtype=np.float64).copy()
    column_sums = np.asarray(result.sum(axis=0)).ravel()
    if np.any(~np.isfinite(column_sums)) or np.any(column_sums <= 0):
        raise ValueError("estimable target profiles must have positive finite mass")
    if column_sums.size:
        result = result @ sparse.diags(1.0 / column_sums, format="csc")
    result = result.tocsc()
    result.sum_duplicates()
    result.sort_indices()
    return result


def _target_profile_matrix_id(matrix: sparse.csc_matrix) -> str:
    canonical = sparse.csc_matrix(matrix, dtype=np.float64).copy()
    canonical.sum_duplicates()
    canonical.sort_indices()
    result: str = stable_id(
        "receiver_program_target_profile_matrix",
        {
            "data_digest": float64_array_digest(canonical.data),
            "indices": canonical.indices.astype(int).tolist(),
            "indptr": canonical.indptr.astype(int).tolist(),
            "shape": list(canonical.shape),
        },
        schema_version="1",
    )
    return result


def _same_sparse_matrix(left: sparse.spmatrix, right: sparse.spmatrix) -> bool:
    left_csc = sparse.csc_matrix(left, dtype=np.float64).copy()
    right_csc = sparse.csc_matrix(right, dtype=np.float64).copy()
    for matrix in (left_csc, right_csc):
        matrix.sum_duplicates()
        matrix.sort_indices()
    return (
        left_csc.shape == right_csc.shape
        and np.array_equal(left_csc.data, right_csc.data)
        and np.array_equal(left_csc.indices, right_csc.indices)
        and np.array_equal(left_csc.indptr, right_csc.indptr)
    )


@dataclass(frozen=True, slots=True, init=False)
class ReceiverProgramTrainingArtifact:
    """Fold-frozen receptor- and sender-agnostic family target programs."""

    receiver_family_artifact: ReceiverFamilyTrainingArtifact
    receiver: str
    contrast_name: str
    fold_id: str
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    estimable_family_ids: tuple[str, ...]
    family_estimable: tuple[bool, ...]
    family_reason_codes: tuple[str | None, ...]
    training_subject_ids: tuple[str, ...]
    target_profile_matrix_id: str
    downstream_functional: DownstreamFunctional | None
    status: str
    reason_code: str | None
    certification_status: str
    training_artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverProgramTrainingArtifact is producer-owned; use "
            "fit_receiver_program_training_artifact()"
        )

    @classmethod
    def _from_training(
        cls,
        *,
        receiver_family_artifact: ReceiverFamilyTrainingArtifact,
        contrast_name: str,
        estimable_family_ids: tuple[str, ...],
        family_estimable: tuple[bool, ...],
        family_reason_codes: tuple[str | None, ...],
        downstream_functional: DownstreamFunctional | None,
        reason_code: str | None,
    ) -> ReceiverProgramTrainingArtifact:
        receiver_family_artifact._require_producer_owned()
        normalized_contrast = _required_name(contrast_name, field_name="contrast_name")
        family_ids = receiver_family_artifact.family_basis.family_ids
        (
            expected_estimable_family_ids,
            expected_family_estimable,
            expected_family_reason_codes,
            raw_target_profiles,
        ) = _target_profile_matrix(receiver_family_artifact)
        if (
            tuple(estimable_family_ids) != expected_estimable_family_ids
            or tuple(family_estimable) != expected_family_estimable
            or tuple(family_reason_codes) != expected_family_reason_codes
        ):
            raise ValueError("receiver-program family estimability is inconsistent")
        normalized_target_profiles = _normalized_target_profile_matrix(
            raw_target_profiles
        )
        target_profile_matrix_id = _target_profile_matrix_id(normalized_target_profiles)
        if downstream_functional is None:
            normalized_reason = _required_name(reason_code, field_name="reason_code")
            status = "not_estimable"
            certification = _TRAINING_NOT_ESTIMABLE
        else:
            if reason_code is not None:
                raise ValueError("observed receiver program cannot have a reason")
            downstream_functional._require_intact()
            expected = (
                receiver_family_artifact.receiver,
                normalized_contrast,
                receiver_family_artifact.fold_id,
                receiver_family_artifact.source_basis.feature_ids,
                estimable_family_ids,
                receiver_family_artifact.training_subject_ids,
            )
            observed = (
                downstream_functional.receiver,
                downstream_functional.contrast_name,
                downstream_functional.fold_id,
                downstream_functional.feature_ids,
                downstream_functional.family_ids,
                downstream_functional.training_subject_ids,
            )
            if observed != expected:
                raise ValueError("receiver-program functional lineage is inconsistent")
            if not _same_sparse_matrix(
                downstream_functional.target_weight_matrix,
                normalized_target_profiles,
            ):
                raise ValueError(
                    "receiver-program target profiles do not match frozen medoids"
                )
            if not np.array_equal(
                np.asarray(downstream_functional.family_support, dtype=np.float64),
                np.ones(len(estimable_family_ids), dtype=np.float64),
            ):
                raise ValueError("receiver-program family support must be exactly one")
            normalized_reason = None
            status = "observed"
            certification = _TRAINING_OBSERVED
        self = object.__new__(cls)
        values: dict[str, Any] = {
            "receiver_family_artifact": receiver_family_artifact,
            "receiver": receiver_family_artifact.receiver,
            "contrast_name": normalized_contrast,
            "fold_id": receiver_family_artifact.fold_id,
            "feature_ids": receiver_family_artifact.source_basis.feature_ids,
            "family_ids": family_ids,
            "estimable_family_ids": estimable_family_ids,
            "family_estimable": family_estimable,
            "family_reason_codes": family_reason_codes,
            "training_subject_ids": receiver_family_artifact.training_subject_ids,
            "target_profile_matrix_id": target_profile_matrix_id,
            "downstream_functional": downstream_functional,
            "status": status,
            "reason_code": normalized_reason,
            "certification_status": certification,
            "_producer_marker": _TRAINING_PRODUCER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "training_artifact_id",
            stable_id(
                "receiver_program_training_artifact",
                self._identity_payload(),
                schema_version="2",
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "certification_status": self.certification_status,
            "contrast_name": self.contrast_name,
            "downstream_functional_id": (
                None
                if self.downstream_functional is None
                else self.downstream_functional.downstream_functional_id
            ),
            "estimable_family_ids": list(self.estimable_family_ids),
            "family_estimable": list(self.family_estimable),
            "family_ids": list(self.family_ids),
            "family_reason_codes": list(self.family_reason_codes),
            "feature_ids": list(self.feature_ids),
            "fold_id": self.fold_id,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "receiver_family_training_artifact_id": (
                self.receiver_family_artifact.training_artifact_id
            ),
            "status": self.status,
            "target_profile_matrix_id": self.target_profile_matrix_id,
            "training_subject_ids": list(self.training_subject_ids),
        }

    def _require_intact(self) -> None:
        try:
            self.receiver_family_artifact._require_producer_owned()
            repeated = ReceiverProgramTrainingArtifact._from_training(
                receiver_family_artifact=self.receiver_family_artifact,
                contrast_name=self.contrast_name,
                estimable_family_ids=self.estimable_family_ids,
                family_estimable=self.family_estimable,
                family_reason_codes=self.family_reason_codes,
                downstream_functional=self.downstream_functional,
                reason_code=self.reason_code,
            )
            valid = (
                self._producer_marker == _TRAINING_PRODUCER
                and self.target_profile_matrix_id == repeated.target_profile_matrix_id
                and stable_id(
                    "receiver_program_training_artifact",
                    self._identity_payload(),
                    schema_version="2",
                )
                == self.training_artifact_id
                and repeated.training_artifact_id == self.training_artifact_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver-program training artifact failed integrity validation",
                code="receiver_program_training_integrity_violation",
                field="training_artifact_id",
                remediation="Refit from intact receiver-family and reference inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Receiver-program training artifact failed integrity validation",
                code="receiver_program_training_integrity_violation",
                field="training_artifact_id",
                remediation="Refit from intact receiver-family and reference inputs",
            )

    def to_dict(self) -> dict[str, object]:
        """Return source-agnostic semantics and complete frozen lineage."""

        self._require_intact()
        downstream = self.downstream_functional
        return {
            "training_artifact_id": self.training_artifact_id,
            **self._identity_payload(),
            "reference_transform_id": (
                None if downstream is None else downstream.reference_transform_id
            ),
            "reference_row_manifest_id": (
                None if downstream is None else downstream.reference_row_manifest_id
            ),
            "reference_subject_summary_digest": (
                None
                if downstream is None
                else downstream.reference_subject_summary_digest
            ),
            "reference_summary_method": (
                None if downstream is None else downstream.reference_summary_method
            ),
            "center_method": (None if downstream is None else downstream.center_method),
            "scale_method": (None if downstream is None else downstream.scale_method),
            "source_agnostic": True,
            "receptor_agnostic": True,
            "sender_agnostic": True,
            "incremental_edge_evidence": False,
            "integrated_edge_evidence": False,
        }


@dataclass(frozen=True, slots=True, init=False)
class ReceiverProgramApplication:
    """Held-out family programs with exact sample/subject/context lineage."""

    training_artifact: ReceiverProgramTrainingArtifact
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_context_ids: tuple[str, ...]
    heldout_subject_ids: tuple[str, ...]
    heldout_row_manifest_id: str
    heldout_expression_digest: str | None
    heldout_input_digest: str | None
    downstream_application: DownstreamApplication | None
    status: str
    reason_code: str | None
    certification_status: str
    application_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverProgramApplication is producer-owned; use "
            "apply_receiver_program_training_artifact()"
        )

    @classmethod
    def _from_application(
        cls,
        *,
        training_artifact: ReceiverProgramTrainingArtifact,
        sample_ids: tuple[str, ...],
        sample_subject_ids: tuple[str, ...],
        sample_context_ids: tuple[str, ...],
        heldout_row_manifest_id: str,
        heldout_expression_digest: str | None,
        heldout_input_digest: str | None,
        downstream_application: DownstreamApplication | None,
        reason_code: str | None,
    ) -> ReceiverProgramApplication:
        training_artifact._require_intact()
        samples = _aligned_names(
            sample_ids, length=len(sample_ids), field_name="sample_ids"
        )
        subjects = _aligned_names(
            sample_subject_ids,
            length=len(samples),
            field_name="sample_subject_ids",
        )
        contexts = _aligned_names(
            sample_context_ids,
            length=len(samples),
            field_name="sample_context_ids",
        )
        if not samples or len(set(samples)) != len(samples):
            raise ValueError("receiver-program sample IDs must be non-empty and unique")
        if tuple(sorted(samples)) != samples:
            raise ValueError("receiver-program sample IDs must be canonicalized")
        expected_manifest = _row_manifest_id(samples, subjects, contexts)
        if heldout_row_manifest_id != expected_manifest:
            raise ValueError(
                "receiver-program row manifest does not match heldout rows"
            )
        if heldout_expression_digest is None:
            normalized_expression_digest = None
            if heldout_input_digest is not None:
                raise ValueError("heldout input digest requires an expression digest")
        else:
            normalized_expression_digest = _required_name(
                heldout_expression_digest,
                field_name="heldout_expression_digest",
            )
            expected_input_digest = _heldout_input_digest(
                normalized_expression_digest,
                feature_ids=training_artifact.feature_ids,
                row_manifest_id=expected_manifest,
            )
            if heldout_input_digest != expected_input_digest:
                raise ValueError(
                    "heldout input digest does not match expression and rows"
                )
        heldout_subjects = tuple(sorted(set(subjects)))
        if set(heldout_subjects).intersection(training_artifact.training_subject_ids):
            raise ValueError("receiver-program heldout subjects overlap training")
        if downstream_application is None:
            normalized_reason = _required_name(reason_code, field_name="reason_code")
            status = "not_estimable"
            certification = _APPLICATION_NOT_ESTIMABLE
        else:
            if reason_code is not None:
                raise ValueError("observed receiver program cannot have a reason")
            if training_artifact.downstream_functional is None:
                raise ValueError("unavailable training cannot produce heldout programs")
            downstream_application._require_intact()
            if (
                downstream_application.downstream_functional_id
                != training_artifact.downstream_functional.downstream_functional_id
                or downstream_application.input_expression_digest
                != normalized_expression_digest
                or downstream_application.input_row_manifest_id != expected_manifest
                or downstream_application.receiver_program_score.shape
                != (len(samples), len(training_artifact.estimable_family_ids))
                or np.any(~np.isfinite(downstream_application.receiver_program_score))
            ):
                raise ValueError("heldout receiver-program values do not match parent")
            normalized_reason = None
            status = "observed"
            certification = _APPLICATION_OBSERVED
        self = object.__new__(cls)
        values: dict[str, Any] = {
            "training_artifact": training_artifact,
            "sample_ids": samples,
            "sample_subject_ids": subjects,
            "sample_context_ids": contexts,
            "heldout_subject_ids": heldout_subjects,
            "heldout_row_manifest_id": heldout_row_manifest_id,
            "heldout_expression_digest": normalized_expression_digest,
            "heldout_input_digest": heldout_input_digest,
            "downstream_application": downstream_application,
            "status": status,
            "reason_code": normalized_reason,
            "certification_status": certification,
            "_producer_marker": _APPLICATION_PRODUCER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "application_id",
            stable_id(
                "receiver_program_application",
                self._identity_payload(),
                schema_version="2",
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "certification_status": self.certification_status,
            "downstream_application_id": (
                None
                if self.downstream_application is None
                else self.downstream_application.application_id
            ),
            "heldout_input_digest": self.heldout_input_digest,
            "heldout_expression_digest": self.heldout_expression_digest,
            "heldout_row_manifest_id": self.heldout_row_manifest_id,
            "reason_code": self.reason_code,
            "rows": [
                {
                    "context_id": context_id,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                }
                for sample_id, subject_id, context_id in zip(
                    self.sample_ids,
                    self.sample_subject_ids,
                    self.sample_context_ids,
                    strict=True,
                )
            ],
            "status": self.status,
            "training_artifact_id": self.training_artifact.training_artifact_id,
        }

    def _require_intact(self) -> None:
        try:
            repeated = ReceiverProgramApplication._from_application(
                training_artifact=self.training_artifact,
                sample_ids=self.sample_ids,
                sample_subject_ids=self.sample_subject_ids,
                sample_context_ids=self.sample_context_ids,
                heldout_row_manifest_id=self.heldout_row_manifest_id,
                heldout_expression_digest=self.heldout_expression_digest,
                heldout_input_digest=self.heldout_input_digest,
                downstream_application=self.downstream_application,
                reason_code=self.reason_code,
            )
            valid = (
                self._producer_marker == _APPLICATION_PRODUCER
                and self.heldout_subject_ids == repeated.heldout_subject_ids
                and self.status == repeated.status
                and self.certification_status == repeated.certification_status
                and stable_id(
                    "receiver_program_application",
                    self._identity_payload(),
                    schema_version="2",
                )
                == self.application_id
                and repeated.application_id == self.application_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver-program application failed integrity validation",
                code="receiver_program_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact frozen receiver-program parent",
            ) from error
        if not valid:
            raise ContractError(
                "Receiver-program application failed integrity validation",
                code="receiver_program_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact frozen receiver-program parent",
            )

    def to_table(self) -> pd.DataFrame:
        """Return one source-agnostic row per sample and frozen family."""

        self._require_intact()
        training = self.training_artifact
        family_index = {
            family_id: index
            for index, family_id in enumerate(training.estimable_family_ids)
        }
        rows: list[dict[str, object]] = []
        for row_index, (sample_id, subject_id, context_id) in enumerate(
            zip(
                self.sample_ids,
                self.sample_subject_ids,
                self.sample_context_ids,
                strict=True,
            )
        ):
            for family_offset, family_id in enumerate(training.family_ids):
                estimable_index = family_index.get(family_id)
                if self.downstream_application is None:
                    score: float | None = None
                    status = "not_estimable"
                    reason = self.reason_code
                elif estimable_index is None:
                    score = None
                    status = "not_estimable"
                    reason = training.family_reason_codes[family_offset]
                else:
                    score = float(
                        self.downstream_application.receiver_program_score[
                            row_index, estimable_index
                        ]
                    )
                    status = "observed"
                    reason = None
                rows.append(
                    {
                        "receiver_program_training_artifact_id": (
                            training.training_artifact_id
                        ),
                        "receiver_program_application_id": self.application_id,
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "context_id": context_id,
                        "receiver": training.receiver,
                        "family_id": family_id,
                        "receiver_program_score": score,
                        "status": status,
                        "reason_code": reason,
                    }
                )
        return pd.DataFrame(rows, columns=RECEIVER_PROGRAM_SCORE_COLUMNS)

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "application_id": self.application_id,
            **self._identity_payload(),
            "source_agnostic": True,
            "receptor_agnostic": True,
            "sender_agnostic": True,
            "incremental_edge_evidence": False,
            "integrated_edge_evidence": False,
        }


def fit_receiver_program_training_artifact(
    receiver_family_artifact: ReceiverFamilyTrainingArtifact,
    reference_expression: np.ndarray,
    *,
    contrast_name: str,
    sample_ids: tuple[str, ...],
    sample_subject_ids: tuple[str, ...],
    reference_context_ids: tuple[str, ...],
    minimum_scale: float = 0.25,
) -> ReceiverProgramTrainingArtifact:
    """Freeze target-only family programs from training reference expression."""

    if not isinstance(receiver_family_artifact, ReceiverFamilyTrainingArtifact):
        raise TypeError(
            "receiver_family_artifact must be ReceiverFamilyTrainingArtifact"
        )
    receiver_family_artifact._require_producer_owned()
    (
        estimable_family_ids,
        family_estimable,
        family_reason_codes,
        target_profiles,
    ) = _target_profile_matrix(receiver_family_artifact)
    if not estimable_family_ids:
        return ReceiverProgramTrainingArtifact._from_training(
            receiver_family_artifact=receiver_family_artifact,
            contrast_name=contrast_name,
            estimable_family_ids=(),
            family_estimable=family_estimable,
            family_reason_codes=family_reason_codes,
            downstream_functional=None,
            reason_code="no_estimable_family_target_profile",
        )
    expression = np.asarray(reference_expression, dtype=np.float64)
    downstream = fit_downstream_functional(
        expression,
        receiver=receiver_family_artifact.receiver,
        contrast_name=contrast_name,
        fold_id=receiver_family_artifact.fold_id,
        feature_ids=receiver_family_artifact.source_basis.feature_ids,
        family_ids=estimable_family_ids,
        reference_sample_ids=sample_ids,
        reference_subject_ids=sample_subject_ids,
        reference_context_ids=reference_context_ids,
        training_subject_ids=receiver_family_artifact.training_subject_ids,
        target_weight_matrix=target_profiles,
        family_support=np.ones(len(estimable_family_ids), dtype=np.float64),
        minimum_scale=minimum_scale,
    )
    return ReceiverProgramTrainingArtifact._from_training(
        receiver_family_artifact=receiver_family_artifact,
        contrast_name=contrast_name,
        estimable_family_ids=estimable_family_ids,
        family_estimable=family_estimable,
        family_reason_codes=family_reason_codes,
        downstream_functional=downstream,
        reason_code=None,
    )


def mark_receiver_program_training_not_estimable(
    receiver_family_artifact: ReceiverFamilyTrainingArtifact,
    *,
    contrast_name: str,
    reason_code: str,
) -> ReceiverProgramTrainingArtifact:
    """Retain all target families when reference expression is unavailable."""

    if not isinstance(receiver_family_artifact, ReceiverFamilyTrainingArtifact):
        raise TypeError(
            "receiver_family_artifact must be ReceiverFamilyTrainingArtifact"
        )
    receiver_family_artifact._require_producer_owned()
    _, family_estimable, family_reason_codes, _ = _target_profile_matrix(
        receiver_family_artifact
    )
    return ReceiverProgramTrainingArtifact._from_training(
        receiver_family_artifact=receiver_family_artifact,
        contrast_name=contrast_name,
        estimable_family_ids=tuple(
            family_id
            for family_id, estimable in zip(
                receiver_family_artifact.family_basis.family_ids,
                family_estimable,
                strict=True,
            )
            if estimable
        ),
        family_estimable=family_estimable,
        family_reason_codes=family_reason_codes,
        downstream_functional=None,
        reason_code=reason_code,
    )


def apply_receiver_program_training_artifact(
    training_artifact: ReceiverProgramTrainingArtifact,
    sample_expression: np.ndarray,
    *,
    feature_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
    sample_subject_ids: tuple[str, ...],
    sample_context_ids: tuple[str, ...],
) -> ReceiverProgramApplication:
    """Apply one training-fold transform unchanged to held-out contexts."""

    if not isinstance(training_artifact, ReceiverProgramTrainingArtifact):
        raise TypeError("training_artifact must be ReceiverProgramTrainingArtifact")
    training_artifact._require_intact()
    if tuple(feature_ids) != training_artifact.feature_ids:
        raise ValueError("feature_ids must exactly match the frozen program parent")
    (
        expression,
        samples,
        subjects,
        contexts,
        row_manifest_id,
    ) = _canonical_heldout_rows(
        sample_expression,
        sample_ids=sample_ids,
        sample_subject_ids=sample_subject_ids,
        sample_context_ids=sample_context_ids,
        n_features=len(feature_ids),
    )
    overlap = set(subjects).intersection(training_artifact.training_subject_ids)
    if overlap:
        raise ValueError(
            "receiver-program heldout subjects overlap training: "
            + ", ".join(sorted(overlap))
        )
    expression_digest = downstream_input_expression_digest(expression)
    input_digest = _heldout_input_digest(
        expression_digest,
        feature_ids=tuple(feature_ids),
        row_manifest_id=row_manifest_id,
    )
    if training_artifact.downstream_functional is None:
        return ReceiverProgramApplication._from_application(
            training_artifact=training_artifact,
            sample_ids=samples,
            sample_subject_ids=subjects,
            sample_context_ids=contexts,
            heldout_row_manifest_id=row_manifest_id,
            heldout_expression_digest=expression_digest,
            heldout_input_digest=input_digest,
            downstream_application=None,
            reason_code=training_artifact.reason_code,
        )
    downstream = apply_downstream_functional(
        training_artifact.downstream_functional,
        expression,
        feature_ids=tuple(feature_ids),
        input_row_manifest_id=row_manifest_id,
    )
    return ReceiverProgramApplication._from_application(
        training_artifact=training_artifact,
        sample_ids=samples,
        sample_subject_ids=subjects,
        sample_context_ids=contexts,
        heldout_row_manifest_id=row_manifest_id,
        heldout_expression_digest=expression_digest,
        heldout_input_digest=input_digest,
        downstream_application=downstream,
        reason_code=None,
    )


def mark_receiver_program_application_not_estimable(
    training_artifact: ReceiverProgramTrainingArtifact,
    *,
    sample_ids: tuple[str, ...],
    sample_subject_ids: tuple[str, ...],
    sample_context_ids: tuple[str, ...],
    reason_code: str,
) -> ReceiverProgramApplication:
    """Emit exact held-out rows with an explicit unavailable program status."""

    if not isinstance(training_artifact, ReceiverProgramTrainingArtifact):
        raise TypeError("training_artifact must be ReceiverProgramTrainingArtifact")
    training_artifact._require_intact()
    samples, subjects, contexts, manifest_id = _canonical_declared_rows(
        sample_ids=sample_ids,
        sample_subject_ids=sample_subject_ids,
        sample_context_ids=sample_context_ids,
    )
    return ReceiverProgramApplication._from_application(
        training_artifact=training_artifact,
        sample_ids=samples,
        sample_subject_ids=subjects,
        sample_context_ids=contexts,
        heldout_row_manifest_id=manifest_id,
        heldout_expression_digest=None,
        heldout_input_digest=None,
        downstream_application=None,
        reason_code=reason_code,
    )


__all__ = [
    "RECEIVER_PROGRAM_SCORE_COLUMNS",
    "ReceiverProgramApplication",
    "ReceiverProgramTrainingArtifact",
    "apply_receiver_program_training_artifact",
    "fit_receiver_program_training_artifact",
    "mark_receiver_program_application_not_estimable",
    "mark_receiver_program_training_not_estimable",
]
