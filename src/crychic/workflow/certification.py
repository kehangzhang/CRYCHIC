"""Fail-closed certification of the complete descriptive cross-fit chain."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from crychic.core import ContractError, stable_id
from crychic.core._validation import validation_scope
from crychic.scoring import (
    SCORING_COLLECTION_AUTHORITATIVE_PLAN_STATUS,
    SCORING_COLLECTION_EXTENSION_VERSION,
)

from .crossfit import CrossFitArtifacts

CROSSFIT_OOF_CERTIFICATION_SCHEMA_VERSION = "1.0.0"
CROSSFIT_OOF_DESCRIPTIVE_SCOPE = (
    "complete_train_apply_oof_descriptive_pipeline_v1"
)
FORMAL_INFERENCE_DISABLED_REASON = (
    "full_pipeline_resampling_and_multiplicity_calibration_required"
)

_AUDIT_PRODUCER_MARKER = "crychic.workflow.crossfit_oof_certification.v1"
_SATISFIED: Literal["satisfied"] = "satisfied"
_NOT_SATISFIED: Literal["not_satisfied"] = "not_satisfied"

RequirementStatus = Literal["satisfied", "not_satisfied"]


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_identifier(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, field_name=field_name)


def _reason(value: object, *, fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossFitOOFRequirementRecord:
    """One stable global or planned receiver-child certification requirement."""

    requirement_code: str
    status: RequirementStatus
    reason_code: str | None
    repeat_id: str
    fold_id: str | None = None
    contrast: str | None = None
    receiver: str | None = None
    evidence_ids: tuple[str, ...] = ()
    requirement_id: str = field(init=False)
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        requirement_code = _identifier(
            self.requirement_code, field_name="requirement_code"
        )
        repeat_id = _identifier(self.repeat_id, field_name="repeat_id")
        fold_id = _optional_identifier(self.fold_id, field_name="fold_id")
        contrast = _optional_identifier(self.contrast, field_name="contrast")
        receiver = _optional_identifier(self.receiver, field_name="receiver")
        if self.status not in {_SATISFIED, _NOT_SATISFIED}:
            raise ValueError("requirement status is invalid")
        if self.status == _SATISFIED:
            if self.reason_code is not None:
                raise ValueError("satisfied requirements cannot have a reason")
            reason_code: str | None = None
        else:
            reason_code = _identifier(self.reason_code, field_name="reason_code")
        evidence_ids = tuple(
            sorted(
                {
                    _identifier(value, field_name="evidence_ids")
                    for value in self.evidence_ids
                }
            )
        )
        scope_payload = {
            "contrast": contrast,
            "fold_id": fold_id,
            "receiver": receiver,
            "repeat_id": repeat_id,
            "requirement_code": requirement_code,
        }
        requirement_id = stable_id(
            "crossfit_oof_requirement", scope_payload, schema_version="1"
        )
        record_payload = {
            **scope_payload,
            "evidence_ids": list(evidence_ids),
            "reason_code": reason_code,
            "requirement_id": requirement_id,
            "status": self.status,
        }
        object.__setattr__(self, "requirement_code", requirement_code)
        object.__setattr__(self, "repeat_id", repeat_id)
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(self, "contrast", contrast)
        object.__setattr__(self, "receiver", receiver)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "crossfit_oof_requirement_record",
                record_payload,
                schema_version="1",
            ),
        )

    @property
    def satisfied(self) -> bool:
        """Whether this exact requirement is satisfied."""

        return self.status == _SATISFIED

    def _identity_payload(self) -> dict[str, object]:
        return {
            "contrast": self.contrast,
            "evidence_ids": list(self.evidence_ids),
            "fold_id": self.fold_id,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "repeat_id": self.repeat_id,
            "requirement_code": self.requirement_code,
            "requirement_id": self.requirement_id,
            "status": self.status,
        }

    def _require_intact(self) -> None:
        repeated = CrossFitOOFRequirementRecord(
            requirement_code=self.requirement_code,
            status=self.status,
            reason_code=self.reason_code,
            repeat_id=self.repeat_id,
            fold_id=self.fold_id,
            contrast=self.contrast,
            receiver=self.receiver,
            evidence_ids=self.evidence_ids,
        )
        if (
            repeated.requirement_id != self.requirement_id
            or repeated.record_id != self.record_id
        ):
            raise ContractError(
                "Cross-fit OOF requirement record failed integrity validation",
                code="crossfit_oof_requirement_integrity_violation",
                field="record_id",
                remediation="Recompute the certification audit from intact artifacts",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical requirement ledger row."""

        self._require_intact()
        return {"record_id": self.record_id, **self._identity_payload()}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> CrossFitOOFRequirementRecord:
        """Load one exact current-version ledger row and verify both IDs."""

        expected = {
            "record_id",
            "contrast",
            "evidence_ids",
            "fold_id",
            "reason_code",
            "receiver",
            "repeat_id",
            "requirement_code",
            "requirement_id",
            "status",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("cross-fit OOF requirement fields are invalid")
        raw_evidence = value["evidence_ids"]
        if not isinstance(raw_evidence, list) or any(
            not isinstance(item, str) for item in raw_evidence
        ):
            raise ValueError("cross-fit OOF requirement evidence_ids are invalid")
        status = value["status"]
        if status not in {_SATISFIED, _NOT_SATISFIED}:
            raise ValueError("cross-fit OOF requirement status is invalid")
        result = cls(
            requirement_code=cast(str, value["requirement_code"]),
            status=status,
            reason_code=cast(str | None, value["reason_code"]),
            repeat_id=cast(str, value["repeat_id"]),
            fold_id=cast(str | None, value["fold_id"]),
            contrast=cast(str | None, value["contrast"]),
            receiver=cast(str | None, value["receiver"]),
            evidence_ids=tuple(raw_evidence),
        )
        if dict(value) != result.to_dict():
            raise ValueError(
                "cross-fit OOF requirement payload or stable identity is invalid"
            )
        return result


def _record_sort_key(
    record: CrossFitOOFRequirementRecord,
) -> tuple[str, str, str, str, str]:
    return (
        record.repeat_id,
        record.fold_id or "",
        record.contrast or "",
        record.receiver or "",
        record.requirement_code,
    )


@dataclass(frozen=True, slots=True, init=False)
class CrossFitOOFCertificationAudit:
    """Producer-owned audit for a complete OOF descriptive train/apply chain.

    Completion is deliberately not a p-value, FDR, probability, or formal
    inference certificate. Those claims require separate full-pipeline
    resampling and multiplicity calibration.
    """

    source_crossfit_id: str
    source_registry_id: str | None
    requirement_ledger: tuple[CrossFitOOFRequirementRecord, ...]
    complete: bool
    certification_scope: str
    formal_inference_allowed: bool
    formal_inference_reason_code: str
    schema_version: str
    audit_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "CrossFitOOFCertificationAudit is producer-owned; use "
            "audit_crossfit_oof_readiness()"
        )

    @classmethod
    def _from_requirements(
        cls,
        *,
        source_crossfit_id: str,
        source_registry_id: str | None,
        requirements: Iterable[CrossFitOOFRequirementRecord],
    ) -> CrossFitOOFCertificationAudit:
        crossfit_id = _identifier(
            source_crossfit_id, field_name="source_crossfit_id"
        )
        registry_id = _optional_identifier(
            source_registry_id, field_name="source_registry_id"
        )
        ledger = tuple(sorted(tuple(requirements), key=_record_sort_key))
        if not ledger or any(
            not isinstance(record, CrossFitOOFRequirementRecord) for record in ledger
        ):
            raise ValueError("certification requirement ledger must not be empty")
        for record in ledger:
            record._require_intact()
        requirement_ids = [record.requirement_id for record in ledger]
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("certification requirements must be scope-unique")
        complete = all(record.satisfied for record in ledger)
        payload = {
            "certification_scope": CROSSFIT_OOF_DESCRIPTIVE_SCOPE,
            "complete": complete,
            "formal_inference_allowed": False,
            "formal_inference_reason_code": FORMAL_INFERENCE_DISABLED_REASON,
            "requirement_record_ids": [record.record_id for record in ledger],
            "schema_version": CROSSFIT_OOF_CERTIFICATION_SCHEMA_VERSION,
            "source_crossfit_id": crossfit_id,
            "source_registry_id": registry_id,
        }
        self = object.__new__(cls)
        values: dict[str, object] = {
            "source_crossfit_id": crossfit_id,
            "source_registry_id": registry_id,
            "requirement_ledger": ledger,
            "complete": complete,
            "certification_scope": CROSSFIT_OOF_DESCRIPTIVE_SCOPE,
            "formal_inference_allowed": False,
            "formal_inference_reason_code": FORMAL_INFERENCE_DISABLED_REASON,
            "schema_version": CROSSFIT_OOF_CERTIFICATION_SCHEMA_VERSION,
            "audit_id": stable_id(
                "crossfit_oof_certification_audit", payload, schema_version="1"
            ),
            "_producer_marker": _AUDIT_PRODUCER_MARKER,
        }
        for field_name, value in values.items():
            object.__setattr__(self, field_name, value)
        return self

    @property
    def is_oof_descriptive_certified(self) -> bool:
        """Whether every requirement for the descriptive OOF chain passed."""

        self._require_intact()
        return self.complete

    @property
    def failed_requirements(self) -> tuple[CrossFitOOFRequirementRecord, ...]:
        """Return every failed requirement in canonical ledger order."""

        self._require_intact()
        return tuple(
            record for record in self.requirement_ledger if not record.satisfied
        )

    def _require_intact(self) -> None:
        try:
            repeated = CrossFitOOFCertificationAudit._from_requirements(
                source_crossfit_id=self.source_crossfit_id,
                source_registry_id=self.source_registry_id,
                requirements=self.requirement_ledger,
            )
            valid = (
                self._producer_marker == _AUDIT_PRODUCER_MARKER
                and self.schema_version
                == CROSSFIT_OOF_CERTIFICATION_SCHEMA_VERSION
                and self.certification_scope == CROSSFIT_OOF_DESCRIPTIVE_SCOPE
                and self.formal_inference_allowed is False
                and self.formal_inference_reason_code
                == FORMAL_INFERENCE_DISABLED_REASON
                and self.complete == repeated.complete
                and self.audit_id == repeated.audit_id
                and self.requirement_ledger == repeated.requirement_ledger
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Cross-fit OOF certification audit failed integrity validation",
                code="crossfit_oof_certification_integrity_violation",
                field="audit_id",
                remediation="Recompute the audit from intact cross-fit artifacts",
            ) from error
        if not valid:
            raise ContractError(
                "Cross-fit OOF certification audit failed integrity validation",
                code="crossfit_oof_certification_integrity_violation",
                field="audit_id",
                remediation="Recompute the audit from intact cross-fit artifacts",
            )

    def to_dict(self) -> dict[str, object]:
        """Return a canonical, inference-safe audit document."""

        self._require_intact()
        return {
            "audit_id": self.audit_id,
            "certification_scope": self.certification_scope,
            "complete": self.complete,
            "formal_inference_allowed": self.formal_inference_allowed,
            "formal_inference_reason_code": self.formal_inference_reason_code,
            "requirement_ledger": [
                record.to_dict() for record in self.requirement_ledger
            ],
            "schema_version": self.schema_version,
            "source_crossfit_id": self.source_crossfit_id,
            "source_registry_id": self.source_registry_id,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> CrossFitOOFCertificationAudit:
        """Load an exact v1 audit without upgrading weaker or newer payloads."""

        expected = {
            "audit_id",
            "certification_scope",
            "complete",
            "formal_inference_allowed",
            "formal_inference_reason_code",
            "requirement_ledger",
            "schema_version",
            "source_crossfit_id",
            "source_registry_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("cross-fit OOF certification audit fields are invalid")
        if value["schema_version"] != CROSSFIT_OOF_CERTIFICATION_SCHEMA_VERSION:
            raise ValueError("cross-fit OOF certification audit version is unsupported")
        if value["certification_scope"] != CROSSFIT_OOF_DESCRIPTIVE_SCOPE:
            raise ValueError("cross-fit OOF certification scope is invalid")
        if type(value["complete"]) is not bool:
            raise ValueError("cross-fit OOF certification complete flag is invalid")
        if value["formal_inference_allowed"] is not False:
            raise ValueError(
                "cross-fit OOF certification cannot allow formal inference"
            )
        if (
            value["formal_inference_reason_code"]
            != FORMAL_INFERENCE_DISABLED_REASON
        ):
            raise ValueError(
                "cross-fit OOF certification formal inference reason is invalid"
            )
        raw_ledger = value["requirement_ledger"]
        if not isinstance(raw_ledger, list) or not raw_ledger:
            raise ValueError("cross-fit OOF certification ledger is invalid")
        requirements = tuple(
            CrossFitOOFRequirementRecord.from_dict(record)
            for record in raw_ledger
            if isinstance(record, Mapping)
        )
        if len(requirements) != len(raw_ledger):
            raise ValueError("cross-fit OOF certification ledger rows are invalid")
        result = cls._from_requirements(
            source_crossfit_id=cast(str, value["source_crossfit_id"]),
            source_registry_id=cast(str | None, value["source_registry_id"]),
            requirements=requirements,
        )
        if dict(value) != result.to_dict():
            raise ValueError(
                "cross-fit OOF certification payload, order, or identity is invalid"
            )
        return result


def _requirement(
    *,
    code: str,
    passed: bool,
    reason_code: str,
    repeat_id: str,
    fold_id: str | None = None,
    contrast: str | None = None,
    receiver: str | None = None,
    evidence_ids: Sequence[str] = (),
) -> CrossFitOOFRequirementRecord:
    return CrossFitOOFRequirementRecord(
        requirement_code=code,
        status=_SATISFIED if passed else _NOT_SATISFIED,
        reason_code=None if passed else reason_code,
        repeat_id=repeat_id,
        fold_id=fold_id,
        contrast=contrast,
        receiver=receiver,
        evidence_ids=tuple(evidence_ids),
    )


def _unique_index(
    values: Iterable[Any],
    key: Callable[[Any], tuple[str, str]],
    *,
    label: str,
) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    for value in values:
        item_key = key(value)
        if item_key in result:
            raise ValueError(f"{label} contains duplicate receiver/contrast keys")
        result[item_key] = value
    return result


def _first_failed_reason(
    checks: Sequence[tuple[bool, str]],
) -> tuple[bool, str]:
    for passed, reason_code in checks:
        if not passed:
            return False, reason_code
    return True, "requirement_satisfied"


def _ids(*values: object) -> tuple[str, ...]:
    return tuple(
        value for value in values if isinstance(value, str) and bool(value.strip())
    )


def _canonical_context_ids(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, tuple) or not value:
        return None
    contexts = tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )
    if (
        len(contexts) != len(value)
        or len(set(contexts)) != len(contexts)
        or contexts != tuple(sorted(contexts))
    ):
        return None
    return contexts


def _table_context_ids(table: object) -> tuple[str, ...] | None:
    columns = getattr(table, "columns", ())
    if "context_id" not in columns:
        return None
    values = tuple(cast(Any, table)["context_id"].tolist())
    if not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        return None
    return tuple(sorted(set(value.strip() for value in values)))


def _child_requirements(
    *,
    fold: Any,
    collection: Any,
    child: Any,
) -> tuple[CrossFitOOFRequirementRecord, ...]:
    key = (collection.contrast, child.receiver)
    family_models = _unique_index(
        fold.receiver_family_models,
        lambda value: (
            value.contrast_name,
            value.receiver_family_artifact.receiver,
        ),
        label="receiver family models",
    )
    family_applications = _unique_index(
        fold.receiver_family_applications,
        lambda value: (
            value.training_artifact.contrast_name,
            value.training_artifact.receiver_family_artifact.receiver,
        ),
        label="receiver family applications",
    )
    program_models = _unique_index(
        fold.receiver_program_models,
        lambda value: (value.contrast_name, value.receiver),
        label="receiver program models",
    )
    program_applications = _unique_index(
        fold.receiver_program_applications,
        lambda value: (
            value.training_artifact.contrast_name,
            value.training_artifact.receiver,
        ),
        label="receiver program applications",
    )
    incremental_models = _unique_index(
        fold.receiver_incremental_models,
        lambda value: (value.contrast_name, value.receiver),
        label="receiver incremental models",
    )
    incremental_applications = {
        (model.contrast_name, model.receiver): application
        for model, application in zip(
            fold.receiver_incremental_models,
            fold.receiver_incremental_applications,
            strict=True,
        )
    }
    common_functionals = _unique_index(
        fold.family_common_functionals,
        lambda value: (value.contrast_name, value.receiver),
        label="family common functionals",
    )
    common_applications = _unique_index(
        fold.family_common_applications,
        lambda value: (
            value.functional.contrast_name,
            value.functional.receiver,
        ),
        label="family common applications",
    )

    family_model = family_models.get(key)
    family_application = family_applications.get(key)
    program_model = program_models.get(key)
    program_application = program_applications.get(key)
    incremental_model = incremental_models.get(key)
    incremental_application = incremental_applications.get(key)
    common_functional = common_functionals.get(key)
    common_application = common_applications.get(key)
    common_binding = None
    if common_functional is not None and common_application is not None:
        common_binding = next(
            (
                binding
                for binding in fold.family_common_bindings
                if binding.family_common_functional_id
                == common_functional.family_common_functional_id
                and binding.family_common_application_id
                == common_application.application_id
            ),
            None,
        )
    registered_sender_functionals = tuple(
        functional
        for functional in getattr(fold.training, "sender_functionals", ())
        if functional.contrast_name == collection.contrast
    )
    registered_sender_functional = (
        registered_sender_functionals[0]
        if len(registered_sender_functionals) == 1
        else None
    )

    scope = {
        "repeat_id": collection.repeat_id,
        "fold_id": collection.fold_id,
        "contrast": collection.contrast,
        "receiver": child.receiver,
    }
    support_matches = tuple(
        record
        for record in fold.receiver_training_support
        if record.receiver_id == child.receiver
    )
    support = support_matches[0] if len(support_matches) == 1 else None
    support_ok = bool(
        support is not None
        and child.receiver_training_support_id == support.support_record_id
        and child.receiver_training_support_status == support.status.value
        and child.receiver_training_support_reason_code == support.reason_code
    )
    if child.receiver_training_support_status == "not_estimable":
        absence_reason = _reason(
            child.receiver_training_support_reason_code,
            fallback="receiver_absent_in_outer_training",
        )
        no_model_chain = all(
            value is None
            for value in (
                family_model,
                family_application,
                program_model,
                program_application,
                incremental_model,
                incremental_application,
                common_functional,
                common_application,
                common_binding,
                child.receiver_family_model_id,
                child.receiver_incremental_model_id,
                child.scoring_functional_id,
            )
        )
        evidence = _ids(
            None if support is None else support.support_record_id,
            child.child_manifest_id,
        )
        records = [
            _requirement(
                code="receiver_training_support_observed",
                passed=False,
                reason_code=absence_reason,
                evidence_ids=evidence,
                **scope,
            )
        ]
        for code in (
            "planned_receiver_child_observed",
            "receiver_family_train_apply_chain_complete",
            "receiver_program_train_apply_chain_observed",
            "receiver_incremental_train_apply_chain_oof_observed",
            "selected_penalty_inner_oof_gain_calibration_observed",
            "family_common_train_apply_binding_chain_complete",
            "family_common_contrast_context_contract_complete",
        ):
            records.append(
                _requirement(
                    code=code,
                    passed=False,
                    reason_code=(
                        absence_reason
                        if support_ok and no_model_chain
                        else "receiver_training_support_lineage_mismatch"
                    ),
                    evidence_ids=evidence,
                    **scope,
                )
            )
        records.append(
            _requirement(
                code="training_apply_subjects_disjoint",
                passed=not bool(
                    set(fold.training.training_subject_ids).intersection(
                        fold.application.heldout_subject_ids
                    )
                ),
                reason_code="training_heldout_subject_overlap",
                evidence_ids=evidence,
                **scope,
            )
        )
        return tuple(records)
    child_ok, child_reason = _first_failed_reason(
        (
            (
                child.registry_status == "functional_registered",
                _reason(
                    child.reason_code,
                    fallback="scoring_functional_not_produced",
                ),
            ),
            (
                child.functional_status == "observed",
                _reason(
                    child.reason_code,
                    fallback="scoring_functional_not_estimable",
                ),
            ),
            (
                child.scoring_functional_id is not None
                and child.score_version is not None,
                "scoring_functional_identity_incomplete",
            ),
        )
    )
    family_ok, family_reason = _first_failed_reason(
        (
            (family_model is not None, "receiver_family_model_missing"),
            (
                family_model is not None
                and family_model.receiver_family_artifact.training_artifact_id
                == child.receiver_family_model_id,
                "receiver_family_model_registry_mismatch",
            ),
            (
                family_model is not None
                and family_model.downstream_functional is not None
                and family_model.reason_code is None,
                _reason(
                    None if family_model is None else family_model.reason_code,
                    fallback="receiver_family_training_not_estimable",
                ),
            ),
            (family_application is not None, "receiver_family_application_missing"),
            (
                family_application is not None
                and family_model is not None
                and family_application.training_artifact_id
                == family_model.training_artifact_id,
                "receiver_family_application_lineage_mismatch",
            ),
            (
                family_application is not None
                and family_application.downstream_application is not None
                and family_application.reason_code is None,
                _reason(
                    None
                    if family_application is None
                    else family_application.reason_code,
                    fallback="receiver_family_application_not_estimable",
                ),
            ),
        )
    )
    program_ok, program_reason = _first_failed_reason(
        (
            (program_model is not None, "receiver_program_model_missing"),
            (
                program_model is not None and program_model.status == "observed",
                _reason(
                    None if program_model is None else program_model.reason_code,
                    fallback="receiver_program_training_not_estimable",
                ),
            ),
            (program_application is not None, "receiver_program_application_missing"),
            (
                program_application is not None
                and program_application.status == "observed",
                _reason(
                    None
                    if program_application is None
                    else program_application.reason_code,
                    fallback="receiver_program_application_not_estimable",
                ),
            ),
        )
    )
    incremental_ok, incremental_reason = _first_failed_reason(
        (
            (incremental_model is not None, "receiver_incremental_model_missing"),
            (
                incremental_model is not None
                and incremental_model.training_artifact_id
                == child.receiver_incremental_model_id,
                "receiver_incremental_model_registry_mismatch",
            ),
            (
                incremental_model is not None and incremental_model.is_oof_certified,
                _reason(
                    None
                    if incremental_model is None
                    else incremental_model.reason_code,
                    fallback="receiver_incremental_training_diagnostic_only",
                ),
            ),
            (
                incremental_model is not None
                and incremental_model.official_incremental_status == "observed",
                "receiver_incremental_training_not_officially_observed",
            ),
            (
                incremental_application is not None,
                "receiver_incremental_application_missing",
            ),
            (
                incremental_application is not None
                and incremental_application.is_oof_certified,
                _reason(
                    None
                    if incremental_application is None
                    else incremental_application.reason_code,
                    fallback="receiver_incremental_application_diagnostic_only",
                ),
            ),
            (
                incremental_application is not None
                and incremental_application.official_incremental_status == "observed",
                "receiver_incremental_application_not_officially_observed",
            ),
        )
    )
    calibration = (
        None
        if incremental_model is None
        else incremental_model.gain_calibration_artifact
    )
    diagnostic_functional = (
        None
        if incremental_model is None
        else incremental_model.diagnostic_functional
    )
    tuning_artifact = (
        None
        if incremental_model is None
        else incremental_model.penalty_tuning_artifact
    )
    calibration_ok, calibration_reason = _first_failed_reason(
        (
            (incremental_model is not None, "receiver_incremental_model_missing"),
            (
                calibration is not None,
                _reason(
                    None
                    if incremental_model is None
                    else incremental_model.diagnostic_reason_code,
                    fallback="selected_penalty_inner_oof_gain_calibration_missing",
                ),
            ),
            (
                calibration is not None
                and calibration.status == "observed"
                and calibration.is_estimable
                and calibration.reason_code is None,
                _reason(
                    None if calibration is None else calibration.reason_code,
                    fallback=(
                        "selected_penalty_inner_oof_gain_calibration_not_estimable"
                    ),
                ),
            ),
            (
                calibration is not None
                and incremental_model is not None
                and diagnostic_functional is not None
                and tuning_artifact is not None
                and calibration.spec.spec_id
                == incremental_model.gain_calibration_spec_id
                and calibration.receiver == incremental_model.receiver
                and calibration.contrast_name == incremental_model.contrast_name
                and calibration.outer_fold_id == incremental_model.fold_id
                and calibration.training_subject_ids
                == incremental_model.training_subject_ids
                and calibration.family_ids == incremental_model.family_ids
                and calibration.feature_ids == incremental_model.feature_ids
                and calibration.null_loss_floor
                == incremental_model.null_loss_floor
                and calibration.tuning_id == tuning_artifact.tuning_id
                and calibration.outer_incremental_functional_id
                == diagnostic_functional.incremental_functional_id
                and calibration.outer_selected_resolved_penalty_id
                == incremental_model.selected_resolved_penalty_id,
                "selected_penalty_inner_oof_gain_calibration_lineage_mismatch",
            ),
        )
    )
    common_ok, common_reason = _first_failed_reason(
        (
            (common_functional is not None, "family_common_functional_missing"),
            (
                common_functional is not None
                and common_functional.family_common_functional_id
                == child.scoring_functional_id,
                "family_common_functional_registry_mismatch",
            ),
            (
                common_functional is not None
                and common_functional.incremental_functional is not None
                and common_functional.incremental_reason_code is None,
                _reason(
                    None
                    if common_functional is None
                    else common_functional.incremental_reason_code,
                    fallback="family_common_functional_not_estimable",
                ),
            ),
            (common_application is not None, "family_common_application_missing"),
            (
                common_application is not None
                and common_application.heldout_reason_code is None
                and common_application.incremental_application_id is not None,
                _reason(
                    None
                    if common_application is None
                    else common_application.heldout_reason_code,
                    fallback="family_common_application_not_estimable",
                ),
            ),
            (common_binding is not None, "family_common_binding_missing"),
            (
                common_binding is not None
                and common_functional is not None
                and common_application is not None
                and common_binding.family_common_functional_id
                == common_functional.family_common_functional_id
                and common_binding.family_common_application_id
                == common_application.application_id,
                "family_common_binding_lineage_mismatch",
            ),
        )
    )
    registered_contexts = (
        None
        if registered_sender_functional is None
        else _canonical_context_ids(registered_sender_functional.context_ids)
    )
    functional_contexts = (
        None
        if common_functional is None
        else _canonical_context_ids(common_functional.context_ids)
    )
    family_score_contexts = (
        None
        if common_application is None
        else _table_context_ids(common_application.family_scores)
    )
    member_score_contexts = (
        None
        if common_application is None
        else _table_context_ids(common_application.member_scores)
    )
    common_context_ok, common_context_reason = _first_failed_reason(
        (
            (
                len(registered_sender_functionals) == 1,
                (
                    "family_common_registered_contrast_missing"
                    if not registered_sender_functionals
                    else "family_common_registered_contrast_duplicate"
                ),
            ),
            (common_functional is not None, "family_common_functional_missing"),
            (common_application is not None, "family_common_application_missing"),
            (common_binding is not None, "family_common_binding_missing"),
            (
                registered_contexts is not None,
                "family_common_registered_contexts_noncanonical",
            ),
            (
                common_functional is not None
                and registered_sender_functional is not None
                and functional_contexts == registered_contexts
                and common_functional.contrast_manifest_id
                == registered_sender_functional.contrast_manifest_id
                and common_functional.sender_functional.sender_functional_id
                == registered_sender_functional.sender_functional_id,
                "family_common_functional_context_mismatch",
            ),
            (
                common_application is not None
                and common_functional is not None
                and common_application.functional.family_common_functional_id
                == common_functional.family_common_functional_id,
                "family_common_application_lineage_mismatch",
            ),
            (
                common_binding is not None
                and common_functional is not None
                and common_application is not None
                and common_binding.family_common_functional_id
                == common_functional.family_common_functional_id
                and common_binding.family_common_application_id
                == common_application.application_id,
                "family_common_binding_lineage_mismatch",
            ),
            (
                family_score_contexts == functional_contexts,
                "family_common_application_family_context_mismatch",
            ),
            (
                member_score_contexts == functional_contexts,
                "family_common_application_member_context_mismatch",
            ),
        )
    )

    subject_pairs: list[tuple[object | None, object | None]] = [
        (fold.training.training_subject_ids, fold.application.heldout_subject_ids),
        (
            None if family_model is None else family_model.training_subject_ids,
            None
            if family_application is None
            else family_application.heldout_subject_ids,
        ),
        (
            None if program_model is None else program_model.training_subject_ids,
            None
            if program_application is None
            else program_application.heldout_subject_ids,
        ),
        (
            None
            if incremental_model is None
            else incremental_model.training_subject_ids,
            None
            if incremental_application is None
            else incremental_application.heldout_subject_ids,
        ),
        (
            None if calibration is None else calibration.training_subject_ids,
            None
            if incremental_application is None
            else incremental_application.heldout_subject_ids,
        ),
        (
            None
            if common_functional is None
            else common_functional.training_subject_ids,
            None
            if common_application is None
            else common_application.heldout_subject_ids,
        ),
    ]
    subject_lineage_complete = all(
        isinstance(training, tuple) and isinstance(heldout, tuple)
        for training, heldout in subject_pairs
    )
    subject_overlap = any(
        bool(set(training).intersection(heldout))
        for training, heldout in subject_pairs
        if isinstance(training, tuple) and isinstance(heldout, tuple)
    )
    subject_ok = subject_lineage_complete and not subject_overlap
    subject_reason = (
        "training_heldout_subject_overlap"
        if subject_overlap
        else "training_heldout_subject_lineage_incomplete"
    )

    return (
        _requirement(
            code="receiver_training_support_observed",
            passed=support_ok
            and child.receiver_training_support_status == "observed",
            reason_code=(
                "receiver_training_support_lineage_mismatch"
                if not support_ok
                else "receiver_training_support_not_observed"
            ),
            evidence_ids=_ids(
                None if support is None else support.support_record_id,
            ),
            **scope,
        ),
        _requirement(
            code="planned_receiver_child_observed",
            passed=child_ok,
            reason_code=child_reason,
            evidence_ids=_ids(child.child_manifest_id),
            **scope,
        ),
        _requirement(
            code="receiver_family_train_apply_chain_complete",
            passed=family_ok,
            reason_code=family_reason,
            evidence_ids=_ids(
                None if family_model is None else family_model.training_artifact_id,
                None
                if family_application is None
                else family_application.application_id,
            ),
            **scope,
        ),
        _requirement(
            code="receiver_program_train_apply_chain_observed",
            passed=program_ok,
            reason_code=program_reason,
            evidence_ids=_ids(
                None if program_model is None else program_model.training_artifact_id,
                None
                if program_application is None
                else program_application.application_id,
            ),
            **scope,
        ),
        _requirement(
            code="receiver_incremental_train_apply_chain_oof_observed",
            passed=incremental_ok,
            reason_code=incremental_reason,
            evidence_ids=_ids(
                None
                if incremental_model is None
                else incremental_model.training_artifact_id,
                None
                if incremental_model is None
                else getattr(
                    incremental_model,
                    "autonomous_program_source_id",
                    None,
                ),
                None
                if incremental_model is None
                else getattr(incremental_model, "latent_nuisance_spec_id", None),
                None
                if incremental_model is None
                else getattr(
                    incremental_model,
                    "latent_nuisance_artifact_id",
                    None,
                ),
                None
                if incremental_application is None
                else incremental_application.application_id,
            ),
            **scope,
        ),
        _requirement(
            code="selected_penalty_inner_oof_gain_calibration_observed",
            passed=calibration_ok,
            reason_code=calibration_reason,
            evidence_ids=_ids(
                None if calibration is None else calibration.artifact_id,
                None if calibration is None else calibration.spec.spec_id,
                None if calibration is None else calibration.tuning_id,
                None
                if calibration is None
                else calibration.outer_incremental_functional_id,
            ),
            **scope,
        ),
        _requirement(
            code="family_common_train_apply_binding_chain_complete",
            passed=common_ok,
            reason_code=common_reason,
            evidence_ids=_ids(
                None
                if common_functional is None
                else common_functional.family_common_functional_id,
                None
                if common_application is None
                else common_application.application_id,
                None if common_binding is None else common_binding.binding_id,
            ),
            **scope,
        ),
        _requirement(
            code="family_common_contrast_context_contract_complete",
            passed=common_context_ok,
            reason_code=common_context_reason,
            evidence_ids=_ids(
                None
                if registered_sender_functional is None
                else registered_sender_functional.sender_functional_id,
                None
                if registered_sender_functional is None
                else registered_sender_functional.contrast_manifest_id,
                None
                if common_functional is None
                else common_functional.family_common_functional_id,
                None
                if common_application is None
                else common_application.application_id,
                None if common_binding is None else common_binding.binding_id,
            ),
            **scope,
        ),
        _requirement(
            code="training_apply_subjects_disjoint",
            passed=subject_ok,
            reason_code=subject_reason,
            evidence_ids=(),
            **scope,
        ),
    )


def _cross_receiver_common_requirement(
    *,
    fold: Any,
    collection: Any,
) -> CrossFitOOFRequirementRecord:
    absent_child = next(
        (
            child
            for child in collection.children
            if child.receiver_training_support_status == "not_estimable"
        ),
        None,
    )
    if absent_child is not None:
        return _requirement(
            code="cross_receiver_common_train_apply_chain_complete",
            passed=False,
            reason_code=_reason(
                absent_child.receiver_training_support_reason_code,
                fallback="receiver_absent_in_outer_training",
            ),
            repeat_id=collection.repeat_id,
            fold_id=collection.fold_id,
            contrast=collection.contrast,
            evidence_ids=_ids(
                absent_child.receiver_training_support_id,
                absent_child.child_manifest_id,
            ),
        )
    functionals = tuple(
        functional
        for functional in fold.cross_receiver_common_functionals
        if functional.contrast_name == collection.contrast
    )
    applications = tuple(
        application
        for application in fold.cross_receiver_common_applications
        if application.functional.contrast_name == collection.contrast
    )
    functional = functionals[0] if len(functionals) == 1 else None
    application = applications[0] if len(applications) == 1 else None
    expected_child_ids = tuple(
        child.scoring_functional_id for child in collection.children
    )
    functional_ok, functional_reason = _first_failed_reason(
        (
            (
                len(functionals) == 1,
                (
                    "cross_receiver_common_functional_missing"
                    if not functionals
                    else "cross_receiver_common_functional_duplicate"
                ),
            ),
            (
                functional is not None
                and functional.fold_id == collection.fold_id
                and functional.contrast_name == collection.contrast,
                "cross_receiver_common_functional_scope_mismatch",
            ),
            (
                functional is not None
                and functional.receiver_ids == collection.planned_receivers,
                "cross_receiver_common_receiver_coverage_mismatch",
            ),
            (
                functional is not None
                and all(value is not None for value in expected_child_ids)
                and functional.child_functional_ids == expected_child_ids,
                "cross_receiver_common_child_lineage_mismatch",
            ),
            (
                functional is not None
                and not functional.common_functional_across_receivers
                and functional.receiver_balanced_descriptive_collection,
                "cross_receiver_common_functional_policy_mismatch",
            ),
            (
                functional is not None
                and functional.spec.schema_version == "4.0.0"
                and tuple(
                    receiver for receiver, _ in functional.receiver_gain_calibrations
                )
                == collection.planned_receivers
                and all(
                    calibration is not None
                    and calibration.status == "observed"
                    and calibration.is_estimable
                    for _, calibration in functional.receiver_gain_calibrations
                )
                and functional.all_receivers_gain_calibrated
                and functional.cross_receiver_percentile_rank_eligible
                and all(
                    functional.gain_calibration_binding(receiver)[
                        "gain_calibration_status"
                    ]
                    == "observed"
                    and functional.gain_calibration_binding(receiver)[
                        "gain_calibration_artifact_id"
                    ]
                    == calibration.artifact_id
                    for receiver, calibration in functional.receiver_gain_calibrations
                    if calibration is not None
                ),
                "cross_receiver_common_training_calibration_invalid",
            ),
            (
                functional is not None
                and functional.training_subject_ids
                == fold.training.training_subject_ids,
                "cross_receiver_common_training_scope_mismatch",
            ),
        )
    )
    application_ok, application_reason = _first_failed_reason(
        (
            (
                len(applications) == 1,
                (
                    "cross_receiver_common_application_missing"
                    if not applications
                    else "cross_receiver_common_application_duplicate"
                ),
            ),
            (
                application is not None
                and functional is not None
                and application.functional.global_common_functional_id
                == functional.global_common_functional_id,
                "cross_receiver_common_application_lineage_mismatch",
            ),
            (
                application is not None
                and application.heldout_subject_ids
                == fold.application.heldout_subject_ids,
                "cross_receiver_common_heldout_scope_mismatch",
            ),
            (
                application is not None
                and isinstance(application.lr_scores_digest, str)
                and bool(application.lr_scores_digest)
                and isinstance(application.sender_scores_digest, str)
                and bool(application.sender_scores_digest),
                "cross_receiver_common_score_digest_missing",
            ),
            (
                functional is not None
                and application is not None
                and not set(functional.training_subject_ids).intersection(
                    application.heldout_subject_ids
                ),
                "cross_receiver_common_subject_overlap",
            ),
        )
    )
    passed = functional_ok and application_ok
    return _requirement(
        code="cross_receiver_common_train_apply_chain_complete",
        passed=passed,
        reason_code=(functional_reason if not functional_ok else application_reason),
        repeat_id=collection.repeat_id,
        fold_id=collection.fold_id,
        contrast=collection.contrast,
        evidence_ids=_ids(
            None
            if functional is None
            else functional.global_common_functional_id,
            None if functional is None else functional.spec.spec_id,
            None if application is None else application.application_id,
            None if application is None else application.lr_scores_digest,
            None if application is None else application.sender_scores_digest,
        ),
    )


@validation_scope()
def audit_crossfit_oof_readiness(
    artifacts: CrossFitArtifacts,
) -> CrossFitOOFCertificationAudit:
    """Audit whether every planned receiver has a complete OOF descriptive chain.

    Invalid or mutated source artifacts are rejected rather than converted into
    a result. Valid but incomplete, untrusted, diagnostic-only, or
    not-estimable chains produce a stable failed-requirement ledger.
    """

    if type(artifacts) is not CrossFitArtifacts:
        raise TypeError("artifacts must be producer-owned CrossFitArtifacts")
    try:
        artifacts._require_intact()
        registry = artifacts.receiver_scoring_registry
        repeat_id = artifacts.spec.repeat_id
        requirements: list[CrossFitOOFRequirementRecord] = []
        registry_ok = bool(
            registry.extension_schema_version
            == SCORING_COLLECTION_EXTENSION_VERSION
            and registry.is_authoritative_registry
            and registry.planning_status
            == SCORING_COLLECTION_AUTHORITATIVE_PLAN_STATUS
        )
        requirements.append(
            _requirement(
                code="authoritative_v4_receiver_registry",
                passed=registry_ok,
                reason_code="receiver_registry_not_v4_authoritative",
                repeat_id=repeat_id,
                evidence_ids=_ids(registry.registry_id),
            )
        )
        tuning_spec = artifacts.spec.penalty_tuning_spec
        tuning_ok = tuning_spec is not None
        if tuning_spec is not None:
            tuning_spec._require_intact()
        requirements.append(
            _requirement(
                code="subject_blocked_penalty_tuning_policy_present",
                passed=tuning_ok,
                reason_code="penalty_tuning_spec_missing",
                repeat_id=repeat_id,
                evidence_ids=_ids(None if tuning_spec is None else tuning_spec.spec_id),
            )
        )
        resource = artifacts.spec.autonomous_program_resource
        latent_nuisance_spec = getattr(
            artifacts.spec,
            "latent_nuisance_spec",
            None,
        )
        nuisance_policy_ok = bool(
            (
                latent_nuisance_spec is None
                and resource is not None
                and resource.is_manifest_verified_trusted
            )
            or (
                latent_nuisance_spec is not None
                and (
                    resource is None
                    or resource.is_manifest_verified_trusted
                )
            )
        )
        requirements.append(
            _requirement(
                code="approved_receiver_autonomous_nuisance_policy_present",
                passed=nuisance_policy_ok,
                reason_code=(
                    "receiver_autonomous_nuisance_policy_missing"
                    if resource is None and latent_nuisance_spec is None
                    else "autonomous_resource_untrusted"
                ),
                repeat_id=repeat_id,
                evidence_ids=_ids(
                    None if resource is None else resource.artifact_id,
                    (
                        None
                        if latent_nuisance_spec is None
                        else latent_nuisance_spec.spec_id
                    ),
                ),
            )
        )

        folds = {fold.fold_id: fold for fold in artifacts.folds}
        for collection in registry.collections:
            fold = folds.get(collection.fold_id)
            if fold is None:
                requirements.append(
                    _requirement(
                        code="cross_receiver_common_train_apply_chain_complete",
                        passed=False,
                        reason_code="planned_fold_artifacts_missing",
                        repeat_id=collection.repeat_id,
                        fold_id=collection.fold_id,
                        contrast=collection.contrast,
                        evidence_ids=(collection.scoring_collection_id,),
                    )
                )
                for child in collection.children:
                    requirements.append(
                        _requirement(
                            code="planned_receiver_fold_artifacts_present",
                            passed=False,
                            reason_code="planned_fold_artifacts_missing",
                            repeat_id=collection.repeat_id,
                            fold_id=collection.fold_id,
                            contrast=collection.contrast,
                            receiver=child.receiver,
                            evidence_ids=(collection.scoring_collection_id,),
                        )
                    )
                continue
            requirements.append(
                _cross_receiver_common_requirement(
                    fold=fold,
                    collection=collection,
                )
            )
            for child in collection.children:
                requirements.extend(
                    _child_requirements(
                        fold=fold,
                        collection=collection,
                        child=child,
                    )
                )
        return CrossFitOOFCertificationAudit._from_requirements(
            source_crossfit_id=artifacts.crossfit_id,
            source_registry_id=registry.registry_id,
            requirements=requirements,
        )
    except ContractError:
        raise
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ContractError(
            "Cross-fit OOF readiness source failed integrity validation",
            code="crossfit_oof_certification_source_invalid",
            field="crossfit_id",
            remediation="Rerun subject cross-fit from intact producer artifacts",
        ) from error


__all__ = [
    "CROSSFIT_OOF_CERTIFICATION_SCHEMA_VERSION",
    "CROSSFIT_OOF_DESCRIPTIVE_SCOPE",
    "FORMAL_INFERENCE_DISABLED_REASON",
    "CrossFitOOFCertificationAudit",
    "CrossFitOOFRequirementRecord",
    "audit_crossfit_oof_readiness",
]
