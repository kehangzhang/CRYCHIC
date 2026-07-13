"""Producer-owned contracts for exploratory sender evidence assignment."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import ContrastSpec

SENDER_EVIDENCE_COMPONENTS = (
    "ligand_availability",
    "cell_type_specificity",
    "subject_prevalence",
    "adjusted_coupling",
)
ASSIGNMENT_GROUP_COLUMNS = ("context_id", "receiver", "interaction_id")
SENDER_ASSIGNMENT_COLUMNS = (
    "assignment_id",
    "assignment_functional_id",
    "context_id",
    "receiver",
    "interaction_id",
    "sender",
    "ligand_availability",
    "cell_type_specificity",
    "subject_prevalence",
    "adjusted_coupling",
    "coupling_weight",
    "evidence_score",
    "assignment_weight",
    "normalized_entropy",
    "n_subjects",
    "n_group_subjects",
    "status",
    "reason_code",
    "coupling_status",
    "coupling_reason_code",
    "assignment_mode",
)
COMMON_SENDER_APPLICATION_COLUMNS = (
    "sender_application_id",
    "sender_functional_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "interaction_id",
    "sender",
    "ligand_availability",
    "training_prevalence_prior",
    "raw_sender_evidence",
    "assignment_weight",
    "normalized_entropy",
    "training_n_subjects",
    "status",
    "reason_code",
    "assignment_mode",
)
COMMON_SENDER_GROUP_COLUMNS = (
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "interaction_id",
)

_COUPLING_REASON = "v0_1_adjusted_coupling_disabled"
_ASSIGNMENT_MODE = "exploratory_in_sample"
_FORBIDDEN_INFERENCE_COLUMNS = {
    "p",
    "p_value",
    "q",
    "q_value",
    "posterior",
    "posterior_probability",
}


class SenderAssignmentStatus(StrEnum):
    """Estimability of one candidate sender's descriptive evidence."""

    OK = "ok"
    LOW_SUPPORT = "low_support"
    PARTIAL_EVIDENCE = "partial_evidence"
    MISSING_EVIDENCE = "missing_evidence"


class SenderCouplingStatus(StrEnum):
    """Availability of covariate-adjusted sender/receiver coupling."""

    NOT_ESTIMABLE_V0_1 = "not_estimable_v0_1"


class SenderPrevalenceStatus(StrEnum):
    """Training-fold support for one frozen sender candidate prior."""

    SUPPORTED = "supported"
    LOW_SUPPORT = "low_support"
    MISSING_EVIDENCE = "missing_evidence"


class CommonSenderApplicationStatus(StrEnum):
    """Application status for one frozen sender candidate."""

    OK = "ok"
    PARTIAL_EVIDENCE = "partial_evidence"
    MISSING_LOCAL_EVIDENCE = "missing_local_evidence"
    TRAINING_PRIOR_NOT_ESTIMABLE = "training_prior_not_estimable"
    NOT_ESTIMABLE = "not_estimable"


def _default_component_weights() -> dict[str, float]:
    return {
        "ligand_availability": 1.0,
        "cell_type_specificity": 1.0,
        "subject_prevalence": 1.0,
        "adjusted_coupling": 0.0,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class SenderEvidenceParameters:
    """Frozen v0.1 parameters for descriptive sender evidence and softmax."""

    min_subjects: int = 3
    prevalence_threshold: float = 0.0
    softmax_temperature: float = 1.0
    component_weights: Mapping[str, float] = field(
        default_factory=_default_component_weights
    )
    schema_version: str = "0.1.0"
    frozen: bool = True
    assignment_functional_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_subjects, bool)
            or not isinstance(self.min_subjects, int)
            or self.min_subjects < 1
        ):
            raise ContractError(
                "min_subjects must be a positive integer",
                code="invalid_sender_parameters",
                field="min_subjects",
                remediation="Choose a positive subject-support threshold",
            )
        threshold = float(self.prevalence_threshold)
        temperature = float(self.softmax_temperature)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ContractError(
                "prevalence_threshold must lie in [0, 1]",
                code="invalid_sender_parameters",
                field="prevalence_threshold",
                remediation="Use a ligand-availability threshold on the unit interval",
            )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ContractError(
                "softmax_temperature must be finite and positive",
                code="invalid_sender_parameters",
                field="softmax_temperature",
                remediation="Use a fixed positive exploratory temperature",
            )
        if set(self.component_weights) != set(SENDER_EVIDENCE_COMPONENTS):
            raise ContractError(
                "component_weights must define every sender evidence component",
                code="invalid_sender_parameters",
                field="component_weights",
                remediation=f"Define exactly {list(SENDER_EVIDENCE_COMPONENTS)}",
            )
        weights: dict[str, float] = {}
        for component in SENDER_EVIDENCE_COMPONENTS:
            weight = float(self.component_weights[component])
            if not math.isfinite(weight) or weight < 0:
                raise ContractError(
                    "sender component weights must be finite and non-negative",
                    code="invalid_sender_parameters",
                    field="component_weights",
                    remediation="Use fixed non-negative evidence weights",
                )
            weights[component] = weight
        if weights["adjusted_coupling"] != 0:
            raise ContractError(
                "v0.1 adjusted coupling must have weight zero",
                code="coupling_not_available_v0_1",
                field="component_weights",
                remediation=(
                    "Set adjusted_coupling to 0 until a validated fold model exists"
                ),
            )
        if not any(
            weights[component] > 0
            for component in SENDER_EVIDENCE_COMPONENTS
            if component != "adjusted_coupling"
        ):
            raise ContractError(
                "at least one descriptive sender component must have positive weight",
                code="invalid_sender_parameters",
                field="component_weights",
                remediation="Enable ligand, specificity, or prevalence evidence",
            )
        if self.schema_version != "0.1.0" or not self.frozen:
            raise ContractError(
                "v0.1 sender parameters must use schema 0.1.0 and be frozen",
                code="invalid_sender_parameters",
                field="schema_version",
                remediation="Use the released immutable v0.1 parameter contract",
            )
        functional_id = stable_id(
            "sender_assignment_functional",
            {
                "component_weights": [
                    [component, weights[component]]
                    for component in SENDER_EVIDENCE_COMPONENTS
                ],
                "min_subjects": self.min_subjects,
                "prevalence_threshold": threshold,
                "schema_version": self.schema_version,
                "softmax_temperature": temperature,
            },
            schema_version=self.schema_version,
        )
        object.__setattr__(self, "prevalence_threshold", threshold)
        object.__setattr__(self, "softmax_temperature", temperature)
        object.__setattr__(self, "component_weights", MappingProxyType(weights))
        object.__setattr__(self, "assignment_functional_id", functional_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready parameter definition."""

        return {
            "assignment_functional_id": self.assignment_functional_id,
            "min_subjects": self.min_subjects,
            "prevalence_threshold": self.prevalence_threshold,
            "softmax_temperature": self.softmax_temperature,
            "component_weights": dict(self.component_weights),
            "schema_version": self.schema_version,
            "frozen": self.frozen,
            "assignment_mode": _ASSIGNMENT_MODE,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ContrastCommonSenderParameters:
    """Caller-declared parameters for a contrast-common sender functional."""

    min_subjects: int = 3
    prevalence_threshold: float = 0.0
    softmax_temperature: float = 1.0
    schema_version: str = "1.0.0"
    parameter_manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_subjects, bool)
            or not isinstance(self.min_subjects, int)
            or self.min_subjects < 1
        ):
            raise ContractError(
                "min_subjects must be a positive integer",
                code="invalid_common_sender_parameters",
                field="min_subjects",
                remediation="Freeze a positive training-support threshold",
            )
        threshold = float(self.prevalence_threshold)
        temperature = float(self.softmax_temperature)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ContractError(
                "prevalence_threshold must lie in [0, 1]",
                code="invalid_common_sender_parameters",
                field="prevalence_threshold",
                remediation="Use a frozen unit-interval availability threshold",
            )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ContractError(
                "softmax_temperature must be finite and positive",
                code="invalid_common_sender_parameters",
                field="softmax_temperature",
                remediation="Freeze a positive temperature before application",
            )
        if self.schema_version != "1.0.0":
            raise ContractError(
                "common sender parameter schema_version must be 1.0.0",
                code="invalid_common_sender_parameters",
                field="schema_version",
                remediation="Use the released common-sender parameter schema",
            )
        payload = {
            "min_subjects": self.min_subjects,
            "prevalence_threshold": threshold,
            "schema_version": self.schema_version,
            "softmax_temperature": temperature,
        }
        object.__setattr__(self, "prevalence_threshold", threshold)
        object.__setattr__(self, "softmax_temperature", temperature)
        object.__setattr__(
            self,
            "parameter_manifest_id",
            stable_id(
                "common_sender_parameters",
                payload,
                schema_version=self.schema_version,
            ),
        )

    def _require_intact(self) -> None:
        """Reject forced mutation of the frozen parameter manifest."""

        try:
            repeated = ContrastCommonSenderParameters(
                min_subjects=self.min_subjects,
                prevalence_threshold=self.prevalence_threshold,
                softmax_temperature=self.softmax_temperature,
                schema_version=self.schema_version,
            )
            valid = (
                self.min_subjects == repeated.min_subjects
                and self.prevalence_threshold == repeated.prevalence_threshold
                and self.softmax_temperature == repeated.softmax_temperature
                and self.schema_version == repeated.schema_version
                and self.parameter_manifest_id == repeated.parameter_manifest_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Common-sender parameters failed integrity validation",
                code="common_sender_parameter_integrity_violation",
                field="parameter_manifest_id",
                remediation="Rebuild the parameters from the declared values",
            ) from error
        if not valid:
            raise ContractError(
                "Common-sender parameters failed integrity validation",
                code="common_sender_parameter_integrity_violation",
                field="parameter_manifest_id",
                remediation="Rebuild the parameters from the declared values",
            )

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-ready frozen parameter manifest."""

        self._require_intact()
        return {
            "parameter_manifest_id": self.parameter_manifest_id,
            "min_subjects": self.min_subjects,
            "prevalence_threshold": self.prevalence_threshold,
            "softmax_temperature": self.softmax_temperature,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SenderPrevalencePrior:
    """One training-fold sender candidate and its pooled prevalence prior."""

    receiver: str
    interaction_id: str
    sender: str
    prevalence_prior: float | None
    n_subjects: int
    status: SenderPrevalenceStatus | str
    reason_code: str | None
    sender_prior_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in ("receiver", "interaction_id", "sender"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(
                    f"{field_name} must be a non-empty identifier",
                    code="invalid_sender_prevalence_prior",
                    field=field_name,
                    remediation="Use stable receiver, interaction, and sender IDs",
                )
            object.__setattr__(self, field_name, value.strip())
        if (
            isinstance(self.n_subjects, bool)
            or not isinstance(self.n_subjects, int)
            or self.n_subjects < 0
        ):
            raise ContractError(
                "n_subjects must be a non-negative integer",
                code="invalid_sender_prevalence_prior",
                field="n_subjects",
                remediation="Count observed training subjects exactly once",
            )
        status = SenderPrevalenceStatus(self.status)
        prior = self.prevalence_prior
        if prior is not None:
            prior = float(prior)
            if not math.isfinite(prior) or not 0 <= prior <= 1:
                raise ContractError(
                    "prevalence_prior must lie in [0, 1] or be missing",
                    code="invalid_sender_prevalence_prior",
                    field="prevalence_prior",
                    remediation="Use a frozen training-fold prevalence fraction",
                )
        if status is SenderPrevalenceStatus.SUPPORTED:
            if prior is None or self.reason_code is not None:
                raise ContractError(
                    "supported sender prior requires a value and no reason",
                    code="invalid_sender_prevalence_prior",
                    field="status",
                    remediation="Keep support status and prior availability aligned",
                )
        elif prior is not None or not self.reason_code:
            raise ContractError(
                "unsupported sender prior requires NA and an explicit reason",
                code="invalid_sender_prevalence_prior",
                field="reason_code",
                remediation="Do not fabricate a low-support prevalence prior",
            )
        payload = {
            "interaction_id": self.interaction_id,
            "n_subjects": self.n_subjects,
            "prevalence_prior": prior,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "sender": self.sender,
            "status": status.value,
        }
        object.__setattr__(self, "prevalence_prior", prior)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "sender_prior_id",
            stable_id("sender_prevalence_prior", payload),
        )

    def _require_intact(self) -> None:
        """Reject forced mutation of a frozen sender prevalence prior."""

        try:
            repeated = SenderPrevalencePrior(
                receiver=self.receiver,
                interaction_id=self.interaction_id,
                sender=self.sender,
                prevalence_prior=self.prevalence_prior,
                n_subjects=self.n_subjects,
                status=self.status,
                reason_code=self.reason_code,
            )
            valid = (
                self.receiver == repeated.receiver
                and self.interaction_id == repeated.interaction_id
                and self.sender == repeated.sender
                and self.prevalence_prior == repeated.prevalence_prior
                and self.n_subjects == repeated.n_subjects
                and self.status == repeated.status
                and self.reason_code == repeated.reason_code
                and self.sender_prior_id == repeated.sender_prior_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Sender prevalence prior failed integrity validation",
                code="sender_prevalence_prior_integrity_violation",
                field="sender_prior_id",
                remediation="Refit the sender prior from training availability",
            ) from error
        if not valid:
            raise ContractError(
                "Sender prevalence prior failed integrity validation",
                code="sender_prevalence_prior_integrity_violation",
                field="sender_prior_id",
                remediation="Refit the sender prior from training availability",
            )

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-ready sender-prior record."""

        self._require_intact()
        return {
            "sender_prior_id": self.sender_prior_id,
            "receiver": self.receiver,
            "interaction_id": self.interaction_id,
            "sender": self.sender,
            "prevalence_prior": self.prevalence_prior,
            "n_subjects": self.n_subjects,
            "status": SenderPrevalenceStatus(self.status).value,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ContrastCommonSenderFunctional:
    """Frozen sender universe and priors shared across contrast contexts."""

    contrast: ContrastSpec
    contrast_context_ids: tuple[tuple[Hashable, str], ...]
    training_subject_ids: tuple[str, ...]
    filter_universe_id: str
    parameters: ContrastCommonSenderParameters
    candidate_priors: tuple[SenderPrevalencePrior, ...]
    schema_version: str = "1.0.0"
    contrast_name: str = field(init=False)
    contrast_manifest_id: str = field(init=False)
    contrast_weights: tuple[tuple[str, float], ...] = field(init=False)
    context_ids: tuple[str, ...] = field(init=False)
    sender_functional_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, ContrastCommonSenderParameters):
            raise TypeError("parameters must be ContrastCommonSenderParameters")
        self.parameters._require_intact()
        if not isinstance(self.contrast, ContrastSpec):
            raise TypeError("contrast must be a ContrastSpec")
        if not self.contrast.estimable or len(self.contrast.weights) < 2:
            raise ContractError(
                "common sender functional requires an estimable contrast",
                code="invalid_common_sender_functional",
                field="contrast",
                remediation="Use an estimable multi-context ContrastSpec",
            )
        raw_contexts = tuple(self.contrast_context_ids)
        if any(
            not isinstance(context_id, str) or not context_id.strip()
            for _, context_id in raw_contexts
        ):
            raise ContractError(
                "contrast context IDs must be non-empty identifiers",
                code="invalid_common_sender_functional",
                field="contrast_context_ids",
                remediation="Map every ContrastSpec context to one context ID",
            )
        contrast_contexts = tuple(
            sorted(
                ((node, context_id.strip()) for node, context_id in raw_contexts),
                key=lambda item: canonical_json(item[0]),
            )
        )
        nodes = tuple(node for node, _ in contrast_contexts)
        contexts = tuple(sorted(context_id for _, context_id in contrast_contexts))
        if (
            len(nodes) != len(set(nodes))
            or set(nodes) != set(self.contrast.weights)
            or len(contexts) != len(set(contexts))
        ):
            raise ContractError(
                "contrast context mapping must exactly match the ContrastSpec",
                code="invalid_common_sender_functional",
                field="contrast_context_ids",
                remediation="Map every contrast context exactly once",
            )
        contrast_weights = tuple(
            sorted(
                (
                    (context_id, float(self.contrast.weights[node]))
                    for node, context_id in contrast_contexts
                )
            )
        )
        contrast_manifest_id = stable_id(
            "contrast_manifest", self.contrast.to_dict()
        )
        subjects = tuple(sorted(value.strip() for value in self.training_subject_ids))
        if len(contexts) < 2:
            raise ContractError(
                "context_ids must contain at least two unique identifiers",
                code="invalid_common_sender_functional",
                field="context_ids",
                remediation="Freeze the complete training contrast context set",
            )
        if (
            not subjects
            or len(subjects) != len(set(subjects))
            or any(not value for value in subjects)
        ):
            raise ContractError(
                "training_subject_ids must be non-empty and unique",
                code="invalid_common_sender_functional",
                field="training_subject_ids",
                remediation="Derive training provenance from the raw fold scope",
            )
        if not isinstance(self.filter_universe_id, str) or not self.filter_universe_id:
            raise ContractError(
                "filter_universe_id must be a non-empty identifier",
                code="invalid_common_sender_functional",
                field="filter_universe_id",
                remediation="Bind sender candidates to the frozen interaction universe",
            )
        priors = tuple(
            sorted(
                self.candidate_priors,
                key=lambda item: (item.receiver, item.interaction_id, item.sender),
            )
        )
        if not priors or any(
            not isinstance(prior, SenderPrevalencePrior) for prior in priors
        ):
            raise ContractError(
                "candidate_priors must contain at least one sender prior",
                code="invalid_common_sender_functional",
                field="candidate_priors",
                remediation="Fit candidates from training-fold availability only",
            )
        keys = [
            (prior.receiver, prior.interaction_id, prior.sender) for prior in priors
        ]
        if len(keys) != len(set(keys)):
            raise ContractError(
                "sender candidate keys must be unique",
                code="invalid_common_sender_functional",
                field="candidate_priors",
                remediation="Emit one prior per receiver/interaction/sender",
            )
        if any(prior.n_subjects > len(subjects) for prior in priors):
            raise ContractError(
                "sender prior support exceeds the training subject universe",
                code="invalid_common_sender_functional",
                field="n_subjects",
                remediation="Pool each subject across contrast contexts once",
            )
        if self.schema_version != "1.0.0":
            raise ContractError(
                "common sender functional schema_version must be 1.0.0",
                code="invalid_common_sender_functional",
                field="schema_version",
                remediation="Use the released common-sender functional schema",
            )
        payload = {
            "candidate_priors": [prior.to_dict() for prior in priors],
            "contrast_manifest_id": contrast_manifest_id,
            "contrast_spec": self.contrast.to_dict(),
            "contrast_name": self.contrast.name,
            "contrast_weights": [
                [context_id, weight] for context_id, weight in contrast_weights
            ],
            "context_ids": list(contexts),
            "filter_universe_id": self.filter_universe_id,
            "parameter_manifest_id": self.parameters.parameter_manifest_id,
            "schema_version": self.schema_version,
            "training_subject_ids": list(subjects),
        }
        object.__setattr__(self, "contrast_context_ids", contrast_contexts)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "contrast_name", self.contrast.name)
        object.__setattr__(self, "contrast_manifest_id", contrast_manifest_id)
        object.__setattr__(self, "contrast_weights", contrast_weights)
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "candidate_priors", priors)
        object.__setattr__(
            self,
            "sender_functional_id",
            stable_id(
                "contrast_common_sender_functional",
                payload,
                schema_version=self.schema_version,
            ),
        )

    def _require_intact(self) -> None:
        """Reject forced mutation of the frozen sender functional."""

        try:
            self.parameters._require_intact()
            for prior in self.candidate_priors:
                prior._require_intact()
            repeated = ContrastCommonSenderFunctional(
                contrast=self.contrast,
                contrast_context_ids=self.contrast_context_ids,
                training_subject_ids=self.training_subject_ids,
                filter_universe_id=self.filter_universe_id,
                parameters=self.parameters,
                candidate_priors=self.candidate_priors,
                schema_version=self.schema_version,
            )
            valid = (
                isinstance(self.contrast_context_ids, tuple)
                and isinstance(self.training_subject_ids, tuple)
                and isinstance(self.candidate_priors, tuple)
                and self.contrast_context_ids == repeated.contrast_context_ids
                and self.training_subject_ids == repeated.training_subject_ids
                and self.filter_universe_id == repeated.filter_universe_id
                and self.candidate_priors == repeated.candidate_priors
                and self.schema_version == repeated.schema_version
                and self.contrast_name == repeated.contrast_name
                and self.contrast_manifest_id == repeated.contrast_manifest_id
                and self.contrast_weights == repeated.contrast_weights
                and self.context_ids == repeated.context_ids
                and self.sender_functional_id == repeated.sender_functional_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Common sender functional failed integrity validation",
                code="common_sender_functional_integrity_violation",
                field="sender_functional_id",
                remediation="Refit the sender functional from training availability",
            ) from error
        if not valid:
            raise ContractError(
                "Common sender functional failed integrity validation",
                code="common_sender_functional_integrity_violation",
                field="sender_functional_id",
                remediation="Refit the sender functional from training availability",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete frozen sender functional manifest."""

        self._require_intact()
        return {
            "sender_functional_id": self.sender_functional_id,
            "schema_version": self.schema_version,
            "contrast_name": self.contrast_name,
            "contrast_manifest_id": self.contrast_manifest_id,
            "contrast_spec": self.contrast.to_dict(),
            "contrast_context_ids": [
                list(value) for value in self.contrast_context_ids
            ],
            "contrast_weights": [list(value) for value in self.contrast_weights],
            "context_ids": list(self.context_ids),
            "training_subject_ids": list(self.training_subject_ids),
            "filter_universe_id": self.filter_universe_id,
            "parameters": self.parameters.to_dict(),
            "candidate_priors": [prior.to_dict() for prior in self.candidate_priors],
            "common_across_contexts": True,
            "causal_interpretation": "evidence_based_non_causal",
        }


@dataclass(frozen=True, slots=True)
class CommonSenderApplication:
    """Sample-local application of one frozen contrast-common functional."""

    table: pd.DataFrame
    functional: ContrastCommonSenderFunctional

    def __post_init__(self) -> None:
        if not isinstance(self.functional, ContrastCommonSenderFunctional):
            raise TypeError("functional must be ContrastCommonSenderFunctional")
        self.functional._require_intact()
        if not isinstance(self.table, pd.DataFrame):
            raise TypeError("common sender application must be a pandas DataFrame")
        table = self.table.copy(deep=True)
        missing = set(COMMON_SENDER_APPLICATION_COLUMNS).difference(table.columns)
        extra = set(table.columns).difference(COMMON_SENDER_APPLICATION_COLUMNS)
        if missing or extra:
            raise ContractError(
                "common sender application columns do not match the contract",
                code="invalid_common_sender_application",
                field="columns",
                remediation=f"Use exactly {list(COMMON_SENDER_APPLICATION_COLUMNS)}",
            )
        identifier_columns = (
            "sender_application_id",
            "sender_functional_id",
            "sample_id",
            "subject_id",
            "context_id",
            "receiver",
            "interaction_id",
            "sender",
        )
        for column in identifier_columns:
            if any(
                not isinstance(value, str) or not value.strip()
                for value in table[column].tolist()
            ):
                raise ContractError(
                    f"{column} must contain non-empty string identifiers",
                    code="invalid_common_sender_application",
                    field=column,
                    remediation="Use the frozen functional and sample identifiers",
                )
        if set(table["sender_functional_id"]) not in (
            set(),
            {self.functional.sender_functional_id},
        ):
            raise ContractError(
                "application rows do not match their sender functional",
                code="invalid_common_sender_application",
                field="sender_functional_id",
                remediation="Apply exactly one frozen sender functional",
            )
        if not set(table["context_id"]).issubset(self.functional.context_ids):
            raise ContractError(
                "application contains a context outside the bound contrast",
                code="invalid_common_sender_application",
                field="context_id",
                remediation="Apply only contexts bound to the ContrastSpec",
            )
        key = [*COMMON_SENDER_GROUP_COLUMNS, "sender"]
        if (
            table.duplicated(key).any()
            or table["sender_application_id"].duplicated().any()
        ):
            raise ContractError(
                "common sender application keys must be unique",
                code="invalid_common_sender_application",
                field="sender_application_id",
                remediation="Emit one row per sample/group/frozen sender candidate",
            )
        numeric_columns = (
            "ligand_availability",
            "training_prevalence_prior",
            "raw_sender_evidence",
            "assignment_weight",
            "normalized_entropy",
        )
        for column in numeric_columns:
            _numeric_interval(table, column, nullable=True)
        if set(table["assignment_mode"]) not in (
            set(),
            {"frozen_contrast_common_partial_not_oof"},
        ):
            raise ContractError(
                "common sender application has an invalid assignment mode",
                code="invalid_common_sender_application",
                field="assignment_mode",
                remediation="Do not relabel a partial application as certified OOF",
            )
        statuses = {status.value for status in CommonSenderApplicationStatus}
        if not set(table["status"]).issubset(statuses):
            raise ContractError(
                "common sender application status is not recognized",
                code="invalid_common_sender_application",
                field="status",
                remediation="Use a released common-sender application status",
            )
        prior_lookup = {
            (prior.receiver, prior.interaction_id, prior.sender): prior
            for prior in self.functional.candidate_priors
        }
        for row in table.itertuples(index=False):
            candidate = (str(row.receiver), str(row.interaction_id), str(row.sender))
            if candidate not in prior_lookup:
                raise ContractError(
                    "application contains a sender outside the frozen universe",
                    code="invalid_common_sender_application",
                    field="sender",
                    remediation="Ignore held-out senders absent from training",
                )
            prior = prior_lookup[candidate]
            expected_application_id = stable_id(
                "common_sender_application",
                {
                    "context_id": str(row.context_id),
                    "interaction_id": str(row.interaction_id),
                    "receiver": str(row.receiver),
                    "sample_id": str(row.sample_id),
                    "sender": str(row.sender),
                    "sender_functional_id": self.functional.sender_functional_id,
                    "subject_id": str(row.subject_id),
                },
            )
            if row.sender_application_id != expected_application_id:
                raise ContractError(
                    "sender application ID does not match its stable grain",
                    code="invalid_common_sender_application",
                    field="sender_application_id",
                    remediation="Build application IDs from the frozen functional",
                )
            if row.training_n_subjects != prior.n_subjects:
                raise ContractError(
                    "application training support does not match the functional",
                    code="invalid_common_sender_application",
                    field="training_n_subjects",
                    remediation="Carry frozen training support unchanged",
                )
            observed_prior = (
                None
                if pd.isna(row.training_prevalence_prior)
                else float(cast(Any, row.training_prevalence_prior))
            )
            if observed_prior != prior.prevalence_prior:
                raise ContractError(
                    "application prevalence prior differs from the functional",
                    code="invalid_common_sender_application",
                    field="training_prevalence_prior",
                    remediation="Carry the training prevalence prior unchanged",
                )
            ligand = (
                None
                if pd.isna(row.ligand_availability)
                else float(cast(Any, row.ligand_availability))
            )
            expected_raw = (
                None
                if ligand is None or prior.prevalence_prior is None
                else ligand * prior.prevalence_prior
            )
            observed_raw = (
                None
                if pd.isna(row.raw_sender_evidence)
                else float(cast(Any, row.raw_sender_evidence))
            )
            if observed_raw != expected_raw:
                raise ContractError(
                    "raw sender evidence must equal local ligand times frozen prior",
                    code="invalid_common_sender_application",
                    field="raw_sender_evidence",
                    remediation="Apply the released sample-local sender formula",
                )
            reason_missing = pd.isna(row.reason_code)
            if (str(row.status) == CommonSenderApplicationStatus.OK.value) != (
                reason_missing
            ):
                raise ContractError(
                    "application status and reason_code disagree",
                    code="invalid_common_sender_application",
                    field="reason_code",
                    remediation="Explain every degraded sender application row",
                )
        for _, group in table.groupby(
            list(COMMON_SENDER_GROUP_COLUMNS), observed=True, sort=False
        ):
            first = group.iloc[0]
            expected_senders = {
                prior.sender
                for prior in self.functional.candidate_priors
                if prior.receiver == first["receiver"]
                and prior.interaction_id == first["interaction_id"]
            }
            if set(group["sender"]) != expected_senders:
                raise ContractError(
                    "application group does not contain the frozen sender universe",
                    code="invalid_common_sender_application",
                    field="sender",
                    remediation="Emit every frozen candidate for each sample group",
                )
            weights = group["assignment_weight"]
            entropy = group["normalized_entropy"]
            raw_evidence = [
                None if pd.isna(value) else float(value)
                for value in group["raw_sender_evidence"]
            ]
            if weights.isna().all():
                if entropy.notna().any() or set(group["status"]) != {
                    CommonSenderApplicationStatus.NOT_ESTIMABLE.value
                }:
                    raise ContractError(
                        "all-missing sender evidence requires NA weights and entropy",
                        code="invalid_common_sender_application",
                        field="assignment_weight",
                        remediation=(
                            "Do not create fallback weights for missing evidence"
                        ),
                    )
                expected_reason = (
                    "all_candidate_evidence_missing"
                    if all(value is None for value in raw_evidence)
                    else "incomplete_candidate_evidence"
                )
                if set(group["reason_code"]) != {expected_reason}:
                    raise ContractError(
                        "not-estimable sender group has an invalid reason code",
                        code="invalid_common_sender_application",
                        field="reason_code",
                        remediation="Preserve all-missing versus incomplete evidence",
                    )
                continue
            if weights.isna().any() or entropy.isna().any():
                raise ContractError(
                    "sender weights must be entirely numeric or entirely missing",
                    code="invalid_common_sender_application",
                    field="assignment_weight",
                    remediation=(
                        "Normalize observed candidates or mark the group missing"
                    ),
                )
            if set(group["status"]) != {
                CommonSenderApplicationStatus.OK.value
            } or group["reason_code"].notna().any():
                raise ContractError(
                    "complete sender evidence requires status=ok and no reason",
                    code="invalid_common_sender_application",
                    field="status",
                    remediation="Derive status from the complete evidence group",
                )
            numeric_weights = [float(value) for value in weights]
            observed = {
                index: value
                for index, value in enumerate(raw_evidence)
                if value is not None
            }
            maximum = max(observed.values())
            exponentials = {
                index: math.exp(
                    (value - maximum)
                    / self.functional.parameters.softmax_temperature
                )
                for index, value in observed.items()
            }
            denominator = sum(exponentials.values())
            expected_weights = [
                exponentials.get(index, 0.0) / denominator
                for index in range(len(raw_evidence))
            ]
            if any(
                not math.isclose(
                    observed_weight,
                    expected_weight,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                for observed_weight, expected_weight in zip(
                    numeric_weights, expected_weights, strict=True
                )
            ):
                raise ContractError(
                    "sender weights do not match the frozen-temperature softmax",
                    code="invalid_common_sender_application",
                    field="assignment_weight",
                    remediation="Apply the frozen functional without retuning",
                )
            if not math.isclose(
                sum(numeric_weights), 1.0, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ContractError(
                    "common sender weights must sum to one",
                    code="invalid_common_sender_application",
                    field="assignment_weight",
                    remediation="Use one frozen-temperature softmax per sample group",
                )
            expected_entropy = 0.0
            if len(numeric_weights) > 1:
                expected_entropy = -sum(
                    weight * math.log(weight)
                    for weight in numeric_weights
                    if weight > 0
                ) / math.log(len(numeric_weights))
            if any(
                not math.isclose(
                    float(value), expected_entropy, rel_tol=1e-10, abs_tol=1e-12
                )
                for value in entropy
            ):
                raise ContractError(
                    "common sender entropy does not match assignment weights",
                    code="invalid_common_sender_application",
                    field="normalized_entropy",
                    remediation="Use normalized Shannon entropy",
                )
        table = table.loc[:, list(COMMON_SENDER_APPLICATION_COLUMNS)].sort_values(
            [*COMMON_SENDER_GROUP_COLUMNS, "sender"],
            kind="stable",
            ignore_index=True,
        )
        object.__setattr__(self, "table", table)

    @property
    def is_oof_certified(self) -> bool:
        """Remain false until every learned stage is train/apply separated."""

        return False

    @property
    def causal_interpretation(self) -> str:
        """Sender weights are non-causal expression allocations."""

        return "evidence_based_non_causal"


def sender_assignment_id(
    *,
    functional_id: str,
    context_id: str,
    receiver: str,
    interaction_id: str,
    sender: str,
    schema_version: str = "0.1.0",
) -> str:
    """Build the stable ID for one sender candidate at its declared grain."""

    return str(
        stable_id(
            "sender_assignment",
            {
                "assignment_functional_id": functional_id,
                "context_id": context_id,
                "interaction_id": interaction_id,
                "receiver": receiver,
                "sender": sender,
            },
            schema_version=schema_version,
        )
    )


def _numeric_interval(table: pd.DataFrame, column: str, *, nullable: bool) -> pd.Series:
    numeric = pd.to_numeric(table[column], errors="coerce")
    invalid_type = table[column].notna() & numeric.isna()
    if invalid_type.any() or (not nullable and numeric.isna().any()):
        raise ContractError(
            f"{column} must contain unit-interval numeric values"
            + (" or NA" if nullable else ""),
            code="invalid_sender_assignment",
            field=column,
            remediation="Build the table through assign_senders",
        )
    present = [float(value) for value in numeric.dropna().tolist()]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in present):
        raise ContractError(
            f"{column} values must lie in [0, 1]",
            code="invalid_sender_assignment",
            field=column,
            remediation="Preserve the released sender evidence scale",
        )
    return numeric


@dataclass(frozen=True, slots=True)
class SenderAssignment:
    """Validated, descriptive soft assignment of candidate sender cell types."""

    table: pd.DataFrame
    parameters: SenderEvidenceParameters

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, SenderEvidenceParameters):
            raise TypeError("parameters must be SenderEvidenceParameters")
        table = self.table.copy(deep=True)
        forbidden = {
            str(column)
            for column in table.columns
            if str(column).lower() in _FORBIDDEN_INFERENCE_COLUMNS
            or "probability" in str(column).lower()
        }
        if forbidden:
            raise ContractError(
                "sender assignment cannot contain inferential fields: "
                f"{sorted(forbidden)}",
                code="forbidden_sender_inference",
                field="columns",
                remediation=(
                    "Keep p/q/posterior quantities out of exploratory assignment"
                ),
            )
        missing = set(SENDER_ASSIGNMENT_COLUMNS).difference(table.columns)
        extra = set(table.columns).difference(SENDER_ASSIGNMENT_COLUMNS)
        if missing or extra:
            raise ContractError(
                "sender assignment columns do not match the producer contract",
                code="invalid_sender_assignment",
                field="columns",
                remediation=f"Use exactly {list(SENDER_ASSIGNMENT_COLUMNS)}",
            )
        identifier_columns = (
            "assignment_id",
            "assignment_functional_id",
            "context_id",
            "receiver",
            "interaction_id",
            "sender",
        )
        for column in identifier_columns:
            if any(
                not isinstance(value, str) or not value.strip()
                for value in table[column].tolist()
            ):
                raise ContractError(
                    f"{column} must contain non-empty string identifiers",
                    code="invalid_sender_assignment",
                    field=column,
                    remediation="Use stable identifiers at the sender assignment grain",
                )
        key = [*ASSIGNMENT_GROUP_COLUMNS, "sender"]
        if table.duplicated(key).any() or table["assignment_id"].duplicated().any():
            raise ContractError(
                "sender assignment primary keys must be unique",
                code="duplicate_sender_assignment",
                field="assignment_id",
                remediation=(
                    "Emit one candidate row per context/receiver/interaction/sender"
                ),
            )
        if set(table["assignment_functional_id"]) not in (
            set(),
            {self.parameters.assignment_functional_id},
        ):
            raise ContractError(
                "sender rows do not match the supplied assignment functional",
                code="invalid_sender_assignment",
                field="assignment_functional_id",
                remediation="Keep parameters and their produced table together",
            )
        for row in table.itertuples(index=False):
            expected_id = sender_assignment_id(
                functional_id=self.parameters.assignment_functional_id,
                context_id=cast(str, row.context_id),
                receiver=cast(str, row.receiver),
                interaction_id=cast(str, row.interaction_id),
                sender=cast(str, row.sender),
                schema_version=self.parameters.schema_version,
            )
            if row.assignment_id != expected_id:
                raise ContractError(
                    "sender assignment ID does not match its stable grain",
                    code="invalid_sender_assignment",
                    field="assignment_id",
                    remediation="Generate assignment IDs with the producer constructor",
                )
        numeric_columns = {
            column: _numeric_interval(
                table,
                column,
                nullable=column
                in {
                    "ligand_availability",
                    "cell_type_specificity",
                    "subject_prevalence",
                    "evidence_score",
                    "assignment_weight",
                    "normalized_entropy",
                },
            )
            for column in (
                "ligand_availability",
                "cell_type_specificity",
                "subject_prevalence",
                "coupling_weight",
                "evidence_score",
                "assignment_weight",
                "normalized_entropy",
            )
        }
        if table["adjusted_coupling"].notna().any():
            raise ContractError(
                "v0.1 adjusted_coupling must be entirely NA",
                code="coupling_not_available_v0_1",
                field="adjusted_coupling",
                remediation="Do not infer coupling from context/batch co-variation",
            )
        if (numeric_columns["coupling_weight"] != 0).any():
            raise ContractError(
                "v0.1 coupling_weight must be zero",
                code="coupling_not_available_v0_1",
                field="coupling_weight",
                remediation="Exclude unvalidated coupling from the evidence score",
            )
        expected_coupling_status = SenderCouplingStatus.NOT_ESTIMABLE_V0_1.value
        if set(table["coupling_status"]) not in (set(), {expected_coupling_status}):
            raise ContractError(
                "v0.1 coupling status must be not_estimable_v0_1",
                code="coupling_not_available_v0_1",
                field="coupling_status",
                remediation="Preserve the explicit v0.1 coupling sentinel",
            )
        if set(table["coupling_reason_code"]) not in (set(), {_COUPLING_REASON}):
            raise ContractError(
                "v0.1 coupling requires its disabled reason code",
                code="coupling_not_available_v0_1",
                field="coupling_reason_code",
                remediation="Record why coupling is not estimable",
            )
        if set(table["assignment_mode"]) not in (set(), {_ASSIGNMENT_MODE}):
            raise ContractError(
                "v0.1 sender assignment must be exploratory in-sample",
                code="invalid_sender_assignment",
                field="assignment_mode",
                remediation="Do not label the descriptive producer as cross-fitted",
            )
        allowed_status = {status.value for status in SenderAssignmentStatus}
        if not set(table["status"]).issubset(allowed_status):
            raise ContractError(
                "sender assignment status is not recognized",
                code="invalid_sender_assignment",
                field="status",
                remediation="Use a released SenderAssignmentStatus value",
            )
        for row in table.itertuples(index=False):
            status = SenderAssignmentStatus(cast(str, row.status))
            reason_missing = pd.isna(row.reason_code)
            evidence_missing = pd.isna(row.evidence_score)
            if status is SenderAssignmentStatus.OK and not reason_missing:
                raise ContractError(
                    "status=ok must not carry a reason_code",
                    code="invalid_sender_assignment",
                    field="reason_code",
                    remediation="Reserve reasons for degraded assignment rows",
                )
            if status is not SenderAssignmentStatus.OK and reason_missing:
                raise ContractError(
                    "degraded sender status requires an explicit reason_code",
                    code="invalid_sender_assignment",
                    field="reason_code",
                    remediation="Record missing, partial, or low-support evidence",
                )
            if (status is SenderAssignmentStatus.MISSING_EVIDENCE) != evidence_missing:
                raise ContractError(
                    "missing_evidence status must agree with evidence_score=NA",
                    code="invalid_sender_assignment",
                    field="evidence_score",
                    remediation="Keep missing evidence distinct from observed zero",
                )
        for column in ("n_subjects", "n_group_subjects"):
            numeric = pd.to_numeric(table[column], errors="coerce")
            if numeric.isna().any() or any(
                not float(value).is_integer() or float(value) < 0
                for value in numeric.tolist()
            ):
                raise ContractError(
                    f"{column} must contain non-negative integer support",
                    code="invalid_sender_assignment",
                    field=column,
                    remediation="Count unique subject IDs at the declared grain",
                )
        grouped = table.groupby(
            list(ASSIGNMENT_GROUP_COLUMNS), sort=False, observed=True
        )
        for _, group in grouped:
            raw_weights = group["assignment_weight"]
            if raw_weights.isna().all():
                if not (
                    group["status"] == SenderAssignmentStatus.MISSING_EVIDENCE.value
                ).all() or group["normalized_entropy"].notna().any():
                    raise ContractError(
                        "all-missing sender evidence requires NA weights and entropy",
                        code="invalid_sender_normalization",
                        field="assignment_weight",
                        remediation="Do not emit fallback weights without evidence",
                    )
                continue
            if raw_weights.isna().any() or group["normalized_entropy"].isna().any():
                raise ContractError(
                    "partially missing sender weights are not a valid assignment",
                    code="invalid_sender_normalization",
                    field="assignment_weight",
                    remediation="Normalize observed evidence or mark the group missing",
                )
            weights = [float(value) for value in raw_weights]
            if not math.isclose(sum(weights), 1.0, rel_tol=1e-10, abs_tol=1e-12):
                raise ContractError(
                    "sender assignment weights must sum to one within every group",
                    code="invalid_sender_normalization",
                    field="assignment_weight",
                    remediation="Apply one stable softmax across all candidate senders",
                )
            candidate_count = len(weights)
            entropy = 0.0
            if candidate_count > 1:
                entropy = -sum(
                    weight * math.log(weight) for weight in weights if weight > 0
                ) / math.log(candidate_count)
            observed_entropy = [float(value) for value in group["normalized_entropy"]]
            if any(
                not math.isclose(value, entropy, rel_tol=1e-10, abs_tol=1e-12)
                for value in observed_entropy
            ):
                raise ContractError(
                    "normalized_entropy must match the group's assignment weights",
                    code="invalid_sender_entropy",
                    field="normalized_entropy",
                    remediation="Use normalized Shannon entropy of assignment weights",
                )
            group_support = {int(value) for value in group["n_group_subjects"]}
            if len(group_support) != 1 or any(
                int(cast(Any, row.n_subjects)) > int(cast(Any, row.n_group_subjects))
                for row in group.itertuples(index=False)
            ):
                raise ContractError(
                    "subject support is inconsistent within a sender group",
                    code="invalid_sender_assignment",
                    field="n_group_subjects",
                    remediation=(
                        "Use the group's unique subject universe as denominator"
                    ),
                )
        table = table.loc[:, list(SENDER_ASSIGNMENT_COLUMNS)].sort_values(
            [*ASSIGNMENT_GROUP_COLUMNS, "sender"],
            kind="stable",
            ignore_index=True,
        )
        object.__setattr__(self, "table", table)

    @property
    def inference_eligible(self) -> bool:
        """Sender evidence is descriptive and never a formal test in v0.1."""

        return False

    @property
    def causal_interpretation(self) -> str:
        """State the interpretation boundary for downstream consumers."""

        return "evidence_based_non_causal"

    @property
    def coupling_estimable(self) -> bool:
        """Adjusted coupling is deliberately disabled in v0.1."""

        return False

    def query(
        self,
        *,
        context_id: str | None = None,
        receiver: str | None = None,
        interaction_id: str | None = None,
        sender: str | None = None,
    ) -> pd.DataFrame:
        """Return a defensive copy filtered at sender-assignment grain."""

        selected = pd.Series(True, index=self.table.index, dtype=bool)
        for column, value in (
            ("context_id", context_id),
            ("receiver", receiver),
            ("interaction_id", interaction_id),
            ("sender", sender),
        ):
            if value is not None:
                selected &= self.table[column] == value
        return self.table.loc[selected].copy(deep=True).reset_index(drop=True)
