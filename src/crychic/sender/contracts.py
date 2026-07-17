"""Producer-owned contracts for exploratory sender evidence assignment."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import InitVar, dataclass, field
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import pandas as pd
from scipy.stats import t as student_t

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
_SENDER_CONTRAST_SUPPORT_POLICY = "subject_equal_paired_one_sided_t_holm_fwer_v3"
_SENDER_CONTRAST_MULTIPLICITY_METHOD = "holm_step_down"
_SENDER_CONTRAST_MULTIPLICITY_SCOPE = (
    "outer_training_fold_receiver_contrast_frozen_interactions"
)
_SENDER_CONTRAST_SUPPORT_PRODUCER = "common_sender_fit_v3"
_SENDER_CONTRAST_SUPPORT_PRODUCER_TOKEN = object()
_SENDER_FUNCTIONAL_PRODUCER = "common_sender_functional_fit_v3"
_SENDER_FUNCTIONAL_PRODUCER_TOKEN = object()
_FORBIDDEN_INFERENCE_COLUMNS = {
    "p",
    "p_value",
    "q",
    "q_value",
    "posterior",
    "posterior_probability",
}


def _canonical_contrast_weights(
    contrast_weights: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    """Return one order-invariant context-ID-to-weight mapping."""

    normalized: list[tuple[str, float]] = []
    for raw_context_id, raw_weight in tuple(contrast_weights):
        if not isinstance(raw_context_id, str) or not raw_context_id.strip():
            raise ValueError("contrast context IDs must be non-empty strings")
        weight = float(raw_weight)
        if not math.isfinite(weight):
            raise ValueError("contrast weights must be finite")
        normalized.append((raw_context_id.strip(), weight))
    if len(normalized) < 2 or len({item[0] for item in normalized}) != len(
        normalized
    ):
        raise ValueError("contrast weights require unique context IDs")
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _sender_contrast_family_id(
    *,
    receiver: str,
    interaction_ids: tuple[str, ...],
    candidate_sender_ids_by_interaction: tuple[tuple[str, tuple[str, ...]], ...],
    training_subject_ids: tuple[str, ...],
    contrast_weights: tuple[tuple[str, float], ...],
    filter_universe_id: str,
    minimum_complete_subjects: int,
    ligand_contrast_confidence_level: float,
    ligand_contrast_minimum_effect: float,
) -> str:
    """Return the immutable receiver-wise Holm family identity."""

    canonical_weights = _canonical_contrast_weights(contrast_weights)
    family_id: str = stable_id(
        "interaction_ligand_contrast_multiplicity_family",
        {
            "aggregation_policy": _SENDER_CONTRAST_SUPPORT_POLICY,
            "contrast_weights": [list(value) for value in canonical_weights],
            "candidate_sender_ids_by_interaction": [
                [interaction_id, list(sender_ids)]
                for interaction_id, sender_ids in candidate_sender_ids_by_interaction
            ],
            "filter_universe_id": filter_universe_id,
            "interaction_ids": list(interaction_ids),
            "ligand_contrast_confidence_level": ligand_contrast_confidence_level,
            "ligand_contrast_minimum_effect": ligand_contrast_minimum_effect,
            "multiplicity_method": _SENDER_CONTRAST_MULTIPLICITY_METHOD,
            "multiplicity_scope": _SENDER_CONTRAST_MULTIPLICITY_SCOPE,
            "minimum_complete_subjects": minimum_complete_subjects,
            "receiver": receiver,
            "training_subject_ids": list(training_subject_ids),
        },
        schema_version="2",
    )
    return family_id


def _holm_step_down_adjustments(
    raw_p_values: Mapping[str, float | None],
) -> dict[str, tuple[int, float]]:
    """Return deterministic Holm ranks and adjusted p-values.

    Not-estimable interactions use an effective p-value of one so they remain
    inside the frozen multiplicity family without acquiring an observed test.
    """

    ordered = sorted(
        (
            (1.0 if raw_p is None else float(raw_p), interaction_id)
            for interaction_id, raw_p in raw_p_values.items()
        ),
        key=lambda item: (item[0], item[1]),
    )
    family_size = len(ordered)
    running_adjusted = 0.0
    result: dict[str, tuple[int, float]] = {}
    for rank, (effective_p, interaction_id) in enumerate(ordered, start=1):
        adjusted = min(
            1.0,
            max(running_adjusted, (family_size - rank + 1) * effective_p),
        )
        result[interaction_id] = (rank, adjusted)
        running_adjusted = adjusted
    return result


def _sender_contrast_familywise_alpha(confidence_level: float) -> float:
    """Derive alpha without rounding decimal 0.95 above decimal 0.05."""

    return float(Decimal("1") - Decimal(str(confidence_level)))


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


class SenderContrastSupportStatus(StrEnum):
    """Training-fold evidence for a positive sender ligand contrast."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    NOT_ESTIMABLE = "not_estimable"


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
    ligand_contrast_confidence_level: float = 0.95
    ligand_contrast_minimum_effect: float = 0.0
    schema_version: str = "3.0.0"
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
        confidence_level = float(self.ligand_contrast_confidence_level)
        minimum_effect = float(self.ligand_contrast_minimum_effect)
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
        if not math.isfinite(confidence_level) or not 0.5 < confidence_level < 1:
            raise ContractError(
                "ligand_contrast_confidence_level must lie in (0.5, 1)",
                code="invalid_common_sender_parameters",
                field="ligand_contrast_confidence_level",
                remediation="Freeze a valid one-sided confidence level",
            )
        if not math.isfinite(minimum_effect) or not 0 <= minimum_effect <= 1:
            raise ContractError(
                "ligand_contrast_minimum_effect must lie in [0, 1]",
                code="invalid_common_sender_parameters",
                field="ligand_contrast_minimum_effect",
                remediation="Freeze an absolute effect on the availability scale",
            )
        if self.schema_version != "3.0.0":
            raise ContractError(
                "common sender parameter schema_version must be 3.0.0",
                code="invalid_common_sender_parameters",
                field="schema_version",
                remediation="Use the released common-sender v3 parameter schema",
            )
        payload = {
            "ligand_contrast_confidence_level": confidence_level,
            "ligand_contrast_minimum_effect": minimum_effect,
            "ligand_contrast_multiplicity_method": (
                _SENDER_CONTRAST_MULTIPLICITY_METHOD
            ),
            "ligand_contrast_multiplicity_scope": (_SENDER_CONTRAST_MULTIPLICITY_SCOPE),
            "min_subjects": self.min_subjects,
            "prevalence_threshold": threshold,
            "schema_version": self.schema_version,
            "softmax_temperature": temperature,
        }
        object.__setattr__(self, "prevalence_threshold", threshold)
        object.__setattr__(self, "softmax_temperature", temperature)
        object.__setattr__(self, "ligand_contrast_confidence_level", confidence_level)
        object.__setattr__(self, "ligand_contrast_minimum_effect", minimum_effect)
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
                ligand_contrast_confidence_level=(
                    self.ligand_contrast_confidence_level
                ),
                ligand_contrast_minimum_effect=(self.ligand_contrast_minimum_effect),
                schema_version=self.schema_version,
            )
            valid = (
                self.min_subjects == repeated.min_subjects
                and self.prevalence_threshold == repeated.prevalence_threshold
                and self.softmax_temperature == repeated.softmax_temperature
                and self.ligand_contrast_confidence_level
                == repeated.ligand_contrast_confidence_level
                and self.ligand_contrast_minimum_effect
                == repeated.ligand_contrast_minimum_effect
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
            "ligand_contrast_confidence_level": (self.ligand_contrast_confidence_level),
            "ligand_contrast_minimum_effect": (self.ligand_contrast_minimum_effect),
            "ligand_contrast_multiplicity_method": (
                _SENDER_CONTRAST_MULTIPLICITY_METHOD
            ),
            "ligand_contrast_multiplicity_scope": (_SENDER_CONTRAST_MULTIPLICITY_SCOPE),
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


@dataclass(frozen=True, slots=True, init=False)
class InteractionLigandContrastSupport:
    """Producer-owned receiver-family-adjusted interaction evidence."""

    receiver: str
    interaction_id: str
    complete_subject_ids: tuple[str, ...]
    subject_effects: tuple[float, ...]
    n_complete: int
    minimum_complete_subjects: int
    contrast_weights: tuple[tuple[str, float], ...]
    aggregation_policy: str
    ligand_contrast_confidence_level: float
    ligand_contrast_minimum_effect: float
    degrees_of_freedom: int | None
    mean_effect: float | None
    sample_standard_error: float | None
    critical_value: float | None
    lower_confidence_bound: float | None
    raw_one_sided_p_value: float | None
    multiplicity_method: str
    multiplicity_scope: str
    multiplicity_family_id: str
    multiplicity_family_size: int
    holm_rank: int
    holm_adjusted_p_value: float
    status: SenderContrastSupportStatus
    reason_code: str | None
    support_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "InteractionLigandContrastSupport is producer-owned; use "
            "fit_contrast_common_sender_functional()"
        )

    @classmethod
    def _from_training(
        cls,
        *,
        _producer_token: object,
        receiver: str,
        interaction_id: str,
        complete_subject_ids: tuple[str, ...],
        subject_effects: tuple[float, ...],
        minimum_complete_subjects: int,
        contrast_weights: tuple[tuple[str, float], ...],
        ligand_contrast_confidence_level: float,
        ligand_contrast_minimum_effect: float,
        degrees_of_freedom: int | None,
        mean_effect: float | None,
        sample_standard_error: float | None,
        critical_value: float | None,
        lower_confidence_bound: float | None,
        raw_one_sided_p_value: float | None,
        multiplicity_method: str,
        multiplicity_scope: str,
        multiplicity_family_id: str,
        multiplicity_family_size: int,
        holm_rank: int,
        holm_adjusted_p_value: float,
        status: SenderContrastSupportStatus | str,
        reason_code: str | None,
    ) -> InteractionLigandContrastSupport:
        if _producer_token is not _SENDER_CONTRAST_SUPPORT_PRODUCER_TOKEN:
            raise TypeError(
                "interaction ligand contrast support is sender-producer-owned"
            )
        identifiers: dict[str, str] = {}
        for field_name, raw_value in (
            ("receiver", receiver),
            ("interaction_id", interaction_id),
            ("multiplicity_family_id", multiplicity_family_id),
        ):
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ContractError(
                    f"{field_name} must be a non-empty identifier",
                    code="invalid_sender_contrast_support",
                    field=field_name,
                    remediation="Use the frozen receiver-family identifiers",
                )
            identifiers[field_name] = raw_value.strip()
        if (
            isinstance(minimum_complete_subjects, bool)
            or not isinstance(minimum_complete_subjects, int)
            or minimum_complete_subjects < 2
        ):
            raise ContractError(
                "minimum_complete_subjects must be an integer of at least two",
                code="invalid_sender_contrast_support",
                field="minimum_complete_subjects",
                remediation="Use max(2, common sender min_subjects)",
            )
        subjects = tuple(complete_subject_ids)
        effects = tuple(float(value) for value in subject_effects)
        if len(subjects) != len(effects) or any(
            not isinstance(value, str) or not value.strip() for value in subjects
        ):
            raise ContractError(
                "complete subject IDs and effects must align one-to-one",
                code="invalid_sender_contrast_support",
                field="complete_subject_ids",
                remediation="Retain every complete subject and paired effect",
            )
        normalized_subjects = tuple(value.strip() for value in subjects)
        if len(set(normalized_subjects)) != len(normalized_subjects) or any(
            not math.isfinite(value) for value in effects
        ):
            raise ContractError(
                "complete subjects must be unique with finite effects",
                code="invalid_sender_contrast_support",
                field="subject_effects",
                remediation="Aggregate one finite contrast effect per subject",
            )
        ordered = tuple(sorted(zip(normalized_subjects, effects, strict=True)))
        canonical_subjects = tuple(subject for subject, _ in ordered)
        canonical_effects = tuple(effect for _, effect in ordered)
        weights: list[tuple[str, float]] = []
        for context_id, raw_weight in tuple(contrast_weights):
            if not isinstance(context_id, str) or not context_id.strip():
                raise ContractError(
                    "contrast context IDs must be non-empty identifiers",
                    code="invalid_sender_contrast_support",
                    field="contrast_weights",
                    remediation="Bind support to the frozen contrast contexts",
                )
            weight = float(raw_weight)
            if not math.isfinite(weight):
                raise ContractError(
                    "contrast weights must be finite",
                    code="invalid_sender_contrast_support",
                    field="contrast_weights",
                    remediation="Bind support to the frozen contrast weights",
                )
            weights.append((context_id.strip(), weight))
        canonical_weights = tuple(sorted(weights))
        if (
            len(canonical_weights) < 2
            or len({context for context, _ in canonical_weights})
            != len(canonical_weights)
            or not any(weight > 0 for _, weight in canonical_weights)
            or not any(weight < 0 for _, weight in canonical_weights)
        ):
            raise ContractError(
                "contrast support requires unique positive and negative contexts",
                code="invalid_sender_contrast_support",
                field="contrast_weights",
                remediation="Use the complete estimable frozen contrast",
            )
        normalized_confidence = float(ligand_contrast_confidence_level)
        normalized_minimum_effect = float(ligand_contrast_minimum_effect)
        if (
            not math.isfinite(normalized_confidence)
            or not 0.5 < normalized_confidence < 1
        ):
            raise ContractError(
                "ligand_contrast_confidence_level must lie in (0.5, 1)",
                code="invalid_sender_contrast_support",
                field="ligand_contrast_confidence_level",
                remediation="Use the frozen familywise confidence level",
            )
        if (
            not math.isfinite(normalized_minimum_effect)
            or not 0 <= normalized_minimum_effect <= 1
        ):
            raise ContractError(
                "ligand_contrast_minimum_effect must lie in [0, 1]",
                code="invalid_sender_contrast_support",
                field="ligand_contrast_minimum_effect",
                remediation="Use the frozen availability-scale null effect",
            )
        if multiplicity_method != _SENDER_CONTRAST_MULTIPLICITY_METHOD:
            raise ContractError(
                "multiplicity_method must use the released Holm policy",
                code="invalid_sender_contrast_support",
                field="multiplicity_method",
                remediation="Use receiver-wise Holm step-down adjustment",
            )
        if multiplicity_scope != _SENDER_CONTRAST_MULTIPLICITY_SCOPE:
            raise ContractError(
                "multiplicity_scope must use the released receiver family",
                code="invalid_sender_contrast_support",
                field="multiplicity_scope",
                remediation="Count every frozen receiver interaction",
            )
        if (
            isinstance(multiplicity_family_size, bool)
            or not isinstance(multiplicity_family_size, int)
            or multiplicity_family_size < 1
        ):
            raise ContractError(
                "multiplicity_family_size must be a positive integer",
                code="invalid_sender_contrast_support",
                field="multiplicity_family_size",
                remediation="Count every frozen interaction in the receiver family",
            )
        if (
            isinstance(holm_rank, bool)
            or not isinstance(holm_rank, int)
            or not 1 <= holm_rank <= multiplicity_family_size
        ):
            raise ContractError(
                "holm_rank must lie within the receiver family",
                code="invalid_sender_contrast_support",
                field="holm_rank",
                remediation="Rank raw p-values with interaction-ID tie breaking",
            )
        normalized_adjusted_p = float(holm_adjusted_p_value)
        if (
            not math.isfinite(normalized_adjusted_p)
            or not 0 <= normalized_adjusted_p <= 1
        ):
            raise ContractError(
                "holm_adjusted_p_value must lie in [0, 1]",
                code="invalid_sender_contrast_support",
                field="holm_adjusted_p_value",
                remediation="Recompute the complete receiver-wise Holm family",
            )

        normalized_status = SenderContrastSupportStatus(status)
        n_complete = len(canonical_subjects)
        numeric_stats = (
            mean_effect,
            sample_standard_error,
            critical_value,
            lower_confidence_bound,
            raw_one_sided_p_value,
        )
        if normalized_status is SenderContrastSupportStatus.NOT_ESTIMABLE:
            if (
                n_complete >= minimum_complete_subjects
                or degrees_of_freedom is not None
                or any(value is not None for value in numeric_stats)
                or normalized_adjusted_p != 1.0
                or not reason_code
            ):
                raise ContractError(
                    "not-estimable support must retain low support and Holm p=1",
                    code="invalid_sender_contrast_support",
                    field="status",
                    remediation="Keep NE interactions inside the frozen family",
                )
            normalized_df: int | None = None
            normalized_stats: tuple[float | None, ...] = (
                None,
                None,
                None,
                None,
                None,
            )
        else:
            if n_complete < minimum_complete_subjects:
                raise ContractError(
                    "observed contrast support lacks complete subjects",
                    code="invalid_sender_contrast_support",
                    field="n_complete",
                    remediation="Mark low-support contrasts not estimable",
                )
            if (
                isinstance(degrees_of_freedom, bool)
                or not isinstance(degrees_of_freedom, int)
                or degrees_of_freedom != n_complete - 1
                or any(value is None for value in numeric_stats)
            ):
                raise ContractError(
                    "observed support requires complete paired test statistics",
                    code="invalid_sender_contrast_support",
                    field="degrees_of_freedom",
                    remediation="Compute the subject-equal one-sided Student-t test",
                )
            normalized_df = degrees_of_freedom
            supplied_stats = tuple(float(cast(float, value)) for value in numeric_stats)
            observed_mean = math.fsum(canonical_effects) / n_complete
            sample_variance = (
                math.fsum((effect - observed_mean) ** 2 for effect in canonical_effects)
                / normalized_df
            )
            observed_se = math.sqrt(sample_variance / n_complete)
            observed_critical = float(
                student_t.ppf(normalized_confidence, normalized_df)
            )
            observed_lower = (
                observed_mean
                if sample_variance == 0.0
                else observed_mean - observed_critical * observed_se
            )
            observed_raw_p = (
                0.0
                if observed_se == 0.0 and observed_mean > normalized_minimum_effect
                else 1.0
                if observed_se == 0.0
                else float(
                    student_t.sf(
                        (observed_mean - normalized_minimum_effect) / observed_se,
                        normalized_df,
                    )
                )
            )
            normalized_stats = (
                observed_mean,
                observed_se,
                observed_critical,
                observed_lower,
                observed_raw_p,
            )
            if (
                any(not math.isfinite(value) for value in normalized_stats)
                or observed_se < 0
                or observed_critical <= 0
                or not 0 <= observed_raw_p <= 1
            ):
                raise ContractError(
                    "observed contrast statistics must be finite and valid",
                    code="invalid_sender_contrast_support",
                    field="mean_effect",
                    remediation="Recompute the paired one-sided Student-t test",
                )
            if any(
                not math.isclose(
                    supplied,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                for supplied, expected in zip(
                    supplied_stats, normalized_stats, strict=True
                )
            ):
                raise ContractError(
                    "supplied contrast statistics disagree with subject effects",
                    code="invalid_sender_contrast_support",
                    field="subject_effects",
                    remediation="Recompute all statistics from complete subjects",
                )
            if normalized_adjusted_p + 1e-15 < observed_raw_p:
                raise ContractError(
                    "Holm adjusted p-value cannot be below the raw p-value",
                    code="invalid_sender_contrast_support",
                    field="holm_adjusted_p_value",
                    remediation="Jointly recompute the receiver family",
                )
            familywise_alpha = _sender_contrast_familywise_alpha(normalized_confidence)
            expected_status = (
                SenderContrastSupportStatus.SUPPORTED
                if normalized_adjusted_p < familywise_alpha
                else SenderContrastSupportStatus.UNSUPPORTED
            )
            if normalized_status is not expected_status:
                raise ContractError(
                    "support status disagrees with strict Holm FWER decision",
                    code="invalid_sender_contrast_support",
                    field="status",
                    remediation="Use adjusted p strictly below familywise alpha",
                )
            if (
                normalized_status is SenderContrastSupportStatus.SUPPORTED
                and reason_code is not None
            ) or (
                normalized_status is SenderContrastSupportStatus.UNSUPPORTED
                and reason_code
                != "ligand_contrast_holm_adjusted_p_not_below_familywise_alpha"
            ):
                raise ContractError(
                    "contrast support status and reason code are inconsistent",
                    code="invalid_sender_contrast_support",
                    field="reason_code",
                    remediation="Use the released Holm decision reason",
                )

        payload = {
            "aggregation_policy": _SENDER_CONTRAST_SUPPORT_POLICY,
            "complete_subject_ids": list(canonical_subjects),
            "contrast_weights": [list(value) for value in canonical_weights],
            "critical_value": normalized_stats[2],
            "degrees_of_freedom": normalized_df,
            "holm_adjusted_p_value": normalized_adjusted_p,
            "holm_rank": holm_rank,
            "interaction_id": identifiers["interaction_id"],
            "ligand_contrast_confidence_level": normalized_confidence,
            "ligand_contrast_minimum_effect": normalized_minimum_effect,
            "lower_confidence_bound": normalized_stats[3],
            "mean_effect": normalized_stats[0],
            "minimum_complete_subjects": minimum_complete_subjects,
            "multiplicity_family_id": identifiers["multiplicity_family_id"],
            "multiplicity_family_size": multiplicity_family_size,
            "multiplicity_method": multiplicity_method,
            "multiplicity_scope": multiplicity_scope,
            "n_complete": n_complete,
            "raw_one_sided_p_value": normalized_stats[4],
            "reason_code": reason_code,
            "receiver": identifiers["receiver"],
            "sample_standard_error": normalized_stats[1],
            "status": normalized_status.value,
            "subject_effects": list(canonical_effects),
        }
        self = object.__new__(cls)
        values: dict[str, object] = {
            "receiver": identifiers["receiver"],
            "interaction_id": identifiers["interaction_id"],
            "complete_subject_ids": canonical_subjects,
            "subject_effects": canonical_effects,
            "n_complete": n_complete,
            "minimum_complete_subjects": minimum_complete_subjects,
            "contrast_weights": canonical_weights,
            "aggregation_policy": _SENDER_CONTRAST_SUPPORT_POLICY,
            "ligand_contrast_confidence_level": normalized_confidence,
            "ligand_contrast_minimum_effect": normalized_minimum_effect,
            "degrees_of_freedom": normalized_df,
            "mean_effect": normalized_stats[0],
            "sample_standard_error": normalized_stats[1],
            "critical_value": normalized_stats[2],
            "lower_confidence_bound": normalized_stats[3],
            "raw_one_sided_p_value": normalized_stats[4],
            "multiplicity_method": multiplicity_method,
            "multiplicity_scope": multiplicity_scope,
            "multiplicity_family_id": identifiers["multiplicity_family_id"],
            "multiplicity_family_size": multiplicity_family_size,
            "holm_rank": holm_rank,
            "holm_adjusted_p_value": normalized_adjusted_p,
            "status": normalized_status,
            "reason_code": reason_code,
            "support_id": stable_id(
                "interaction_ligand_contrast_support", payload, schema_version="3"
            ),
            "_producer_marker": _SENDER_CONTRAST_SUPPORT_PRODUCER,
        }
        for field_name, value in values.items():
            object.__setattr__(self, field_name, value)
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "aggregation_policy": self.aggregation_policy,
            "complete_subject_ids": list(self.complete_subject_ids),
            "contrast_weights": [list(value) for value in self.contrast_weights],
            "critical_value": self.critical_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "holm_adjusted_p_value": self.holm_adjusted_p_value,
            "holm_rank": self.holm_rank,
            "interaction_id": self.interaction_id,
            "ligand_contrast_confidence_level": (self.ligand_contrast_confidence_level),
            "ligand_contrast_minimum_effect": (self.ligand_contrast_minimum_effect),
            "lower_confidence_bound": self.lower_confidence_bound,
            "mean_effect": self.mean_effect,
            "minimum_complete_subjects": self.minimum_complete_subjects,
            "multiplicity_family_id": self.multiplicity_family_id,
            "multiplicity_family_size": self.multiplicity_family_size,
            "multiplicity_method": self.multiplicity_method,
            "multiplicity_scope": self.multiplicity_scope,
            "n_complete": self.n_complete,
            "raw_one_sided_p_value": self.raw_one_sided_p_value,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "sample_standard_error": self.sample_standard_error,
            "status": SenderContrastSupportStatus(self.status).value,
            "subject_effects": list(self.subject_effects),
        }

    def _require_intact(self) -> None:
        try:
            repeated = InteractionLigandContrastSupport._from_training(
                _producer_token=_SENDER_CONTRAST_SUPPORT_PRODUCER_TOKEN,
                receiver=self.receiver,
                interaction_id=self.interaction_id,
                complete_subject_ids=self.complete_subject_ids,
                subject_effects=self.subject_effects,
                minimum_complete_subjects=self.minimum_complete_subjects,
                contrast_weights=self.contrast_weights,
                ligand_contrast_confidence_level=(
                    self.ligand_contrast_confidence_level
                ),
                ligand_contrast_minimum_effect=(self.ligand_contrast_minimum_effect),
                degrees_of_freedom=self.degrees_of_freedom,
                mean_effect=self.mean_effect,
                sample_standard_error=self.sample_standard_error,
                critical_value=self.critical_value,
                lower_confidence_bound=self.lower_confidence_bound,
                raw_one_sided_p_value=self.raw_one_sided_p_value,
                multiplicity_method=self.multiplicity_method,
                multiplicity_scope=self.multiplicity_scope,
                multiplicity_family_id=self.multiplicity_family_id,
                multiplicity_family_size=self.multiplicity_family_size,
                holm_rank=self.holm_rank,
                holm_adjusted_p_value=self.holm_adjusted_p_value,
                status=self.status,
                reason_code=self.reason_code,
            )
            valid = (
                self._producer_marker == _SENDER_CONTRAST_SUPPORT_PRODUCER
                and self._identity_payload() == repeated._identity_payload()
                and self.support_id == repeated.support_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Sender contrast support failed integrity validation",
                code="sender_contrast_support_integrity_violation",
                field="support_id",
                remediation="Refit support from outer-training availability",
            ) from error
        if not valid:
            raise ContractError(
                "Sender contrast support failed integrity validation",
                code="sender_contrast_support_integrity_violation",
                field="support_id",
                remediation="Refit support from outer-training availability",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the immutable raw and receiver-family-adjusted evidence."""

        self._require_intact()
        return {"support_id": self.support_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionLigandContrastGate:
    """Immutable interaction-level view of receiver-family-adjusted support."""

    sender_functional_id: str
    receiver: str
    interaction_id: str
    gate: float | None
    status: SenderContrastSupportStatus | str
    reason_code: str | None
    support_ids: tuple[str, ...]
    gate_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers: dict[str, str] = {}
        for field_name in ("sender_functional_id", "receiver", "interaction_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(
                    f"{field_name} must be a non-empty identifier",
                    code="invalid_interaction_ligand_contrast_gate",
                    field=field_name,
                    remediation="Aggregate an intact common-sender functional",
                )
            identifiers[field_name] = value.strip()
        support_ids = tuple(sorted(self.support_ids))
        if (
            len(support_ids) > 1
            or len(support_ids) != len(set(support_ids))
            or any(
                not isinstance(value, str) or not value.strip() for value in support_ids
            )
        ):
            raise ContractError(
                "support_ids must contain at most one immutable identifier",
                code="invalid_interaction_ligand_contrast_gate",
                field="support_ids",
                remediation="Bind the exact contributing sender supports",
            )
        status = SenderContrastSupportStatus(self.status)
        gate = None if self.gate is None else float(self.gate)
        expected_gate = {
            SenderContrastSupportStatus.SUPPORTED: 1.0,
            SenderContrastSupportStatus.UNSUPPORTED: 0.0,
            SenderContrastSupportStatus.NOT_ESTIMABLE: None,
        }[status]
        observed_gate_without_support = (
            status is not SenderContrastSupportStatus.NOT_ESTIMABLE
            and len(support_ids) != 1
        )
        absent_support_with_observed_status = (
            not support_ids and status is not SenderContrastSupportStatus.NOT_ESTIMABLE
        )
        if (
            gate != expected_gate
            or observed_gate_without_support
            or absent_support_with_observed_status
            or (
                status is SenderContrastSupportStatus.SUPPORTED
                and self.reason_code is not None
            )
            or (
                status is not SenderContrastSupportStatus.SUPPORTED
                and not self.reason_code
            )
        ):
            raise ContractError(
                "gate, status, and reason code are inconsistent",
                code="invalid_interaction_ligand_contrast_gate",
                field="gate",
                remediation="Use the released sender contrast aggregation policy",
            )
        payload = {
            "gate": gate,
            "interaction_id": identifiers["interaction_id"],
            "reason_code": self.reason_code,
            "receiver": identifiers["receiver"],
            "sender_functional_id": identifiers["sender_functional_id"],
            "status": status.value,
            "support_ids": list(support_ids),
        }
        object.__setattr__(
            self,
            "sender_functional_id",
            identifiers["sender_functional_id"],
        )
        object.__setattr__(self, "receiver", identifiers["receiver"])
        object.__setattr__(self, "interaction_id", identifiers["interaction_id"])
        object.__setattr__(self, "support_ids", support_ids)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "gate", gate)
        object.__setattr__(
            self,
            "gate_id",
            stable_id("interaction_ligand_contrast_gate", payload, schema_version="3"),
        )

    def _require_intact(self) -> None:
        try:
            repeated = InteractionLigandContrastGate(
                sender_functional_id=self.sender_functional_id,
                receiver=self.receiver,
                interaction_id=self.interaction_id,
                gate=self.gate,
                status=self.status,
                reason_code=self.reason_code,
                support_ids=self.support_ids,
            )
            valid = self.gate_id == repeated.gate_id
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Interaction ligand contrast gate failed integrity validation",
                code="interaction_ligand_contrast_gate_integrity_violation",
                field="gate_id",
                remediation="Reaggregate the intact common-sender functional",
            ) from error
        if not valid:
            raise ContractError(
                "Interaction ligand contrast gate failed integrity validation",
                code="interaction_ligand_contrast_gate_integrity_violation",
                field="gate_id",
                remediation="Reaggregate the intact common-sender functional",
            )

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-ready immutable gate summary."""

        self._require_intact()
        return {
            "gate_id": self.gate_id,
            "sender_functional_id": self.sender_functional_id,
            "receiver": self.receiver,
            "interaction_id": self.interaction_id,
            "gate": self.gate,
            "status": SenderContrastSupportStatus(self.status).value,
            "reason_code": self.reason_code,
            "support_ids": list(self.support_ids),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ContrastCommonSenderFunctional:
    """Frozen sender universe and priors shared across contrast contexts."""

    _producer_token: InitVar[object]
    contrast: ContrastSpec
    contrast_context_ids: tuple[tuple[Hashable, str], ...]
    training_subject_ids: tuple[str, ...]
    filter_universe_id: str
    frozen_interaction_ids: tuple[str, ...]
    candidate_sender_manifest: tuple[tuple[str, str, tuple[str, ...]], ...]
    training_availability_digest: str
    training_input_digest: str
    parameters: ContrastCommonSenderParameters
    candidate_priors: tuple[SenderPrevalencePrior, ...]
    contrast_supports: tuple[InteractionLigandContrastSupport, ...]
    schema_version: str = "3.0.0"
    contrast_name: str = field(init=False)
    contrast_manifest_id: str = field(init=False)
    contrast_weights: tuple[tuple[str, float], ...] = field(init=False)
    context_ids: tuple[str, ...] = field(init=False)
    sender_functional_id: str = field(init=False)
    _producer_marker: str = field(init=False, repr=False)

    def __post_init__(self, _producer_token: object) -> None:
        if _producer_token is not _SENDER_FUNCTIONAL_PRODUCER_TOKEN:
            raise TypeError(
                "ContrastCommonSenderFunctional is producer-owned; use "
                "fit_contrast_common_sender_functional()"
            )
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
        contrast_weights = _canonical_contrast_weights(
            tuple(
                (context_id, float(self.contrast.weights[node]))
                for node, context_id in contrast_contexts
            )
        )
        contrast_manifest_id = stable_id("contrast_manifest", self.contrast.to_dict())
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
        if not isinstance(self.frozen_interaction_ids, tuple):
            raise ContractError(
                "frozen_interaction_ids must be an immutable identifier tuple",
                code="invalid_common_sender_functional",
                field="frozen_interaction_ids",
                remediation="Bind the complete outer-training interaction universe",
            )
        raw_interactions = tuple(self.frozen_interaction_ids)
        if not raw_interactions or any(
            not isinstance(value, str) or not value.strip()
            for value in raw_interactions
        ):
            raise ContractError(
                "frozen_interaction_ids must contain non-empty identifiers",
                code="invalid_common_sender_functional",
                field="frozen_interaction_ids",
                remediation="Bind the complete outer-training interaction universe",
            )
        frozen_interactions = tuple(
            sorted(value.strip() for value in raw_interactions)
        )
        if len(frozen_interactions) != len(set(frozen_interactions)):
            raise ContractError(
                "frozen_interaction_ids must be unique",
                code="invalid_common_sender_functional",
                field="frozen_interaction_ids",
                remediation="Deduplicate the frozen interaction manifest",
            )
        if (
            not isinstance(self.training_availability_digest, str)
            or not self.training_availability_digest.strip()
        ):
            raise ContractError(
                "training_availability_digest must be a non-empty identifier",
                code="invalid_common_sender_functional",
                field="training_availability_digest",
                remediation="Fit the functional from canonical training availability",
            )
        training_availability_digest = self.training_availability_digest.strip()
        if (
            not isinstance(self.training_input_digest, str)
            or not self.training_input_digest.strip()
        ):
            raise ContractError(
                "training_input_digest must be a non-empty identifier",
                code="invalid_common_sender_functional",
                field="training_input_digest",
                remediation="Bind the functional to the sanitized raw fold input",
            )
        training_input_digest = self.training_input_digest.strip()
        candidate_manifest: list[tuple[str, str, tuple[str, ...]]] = []
        for raw_entry in tuple(self.candidate_sender_manifest):
            if not isinstance(raw_entry, tuple) or len(raw_entry) != 3:
                raise ContractError(
                    "candidate sender manifest entries must have three fields",
                    code="invalid_common_sender_functional",
                    field="candidate_sender_manifest",
                    remediation="Freeze receiver, interaction, and sender identifiers",
                )
            raw_receiver, raw_interaction, raw_senders = raw_entry
            if (
                not isinstance(raw_receiver, str)
                or not raw_receiver.strip()
                or not isinstance(raw_interaction, str)
                or not raw_interaction.strip()
            ):
                raise ContractError(
                    "candidate sender manifest identifiers must be non-empty",
                    code="invalid_common_sender_functional",
                    field="candidate_sender_manifest",
                    remediation="Use stable receiver and interaction identifiers",
                )
            if not isinstance(raw_senders, tuple):
                raise ContractError(
                    "candidate senders must be an immutable identifier tuple",
                    code="invalid_common_sender_functional",
                    field="candidate_sender_manifest",
                    remediation="Freeze candidates before fitting ligand contrasts",
                )
            senders = tuple(raw_senders)
            if not senders or any(
                not isinstance(sender, str) or not sender.strip()
                for sender in senders
            ):
                raise ContractError(
                    "each frozen interaction must contain candidate senders",
                    code="invalid_common_sender_functional",
                    field="candidate_sender_manifest",
                    remediation="Freeze candidates before fitting ligand contrasts",
                )
            normalized_senders = tuple(sorted(sender.strip() for sender in senders))
            if len(normalized_senders) != len(set(normalized_senders)):
                raise ContractError(
                    "candidate sender identifiers must be unique per interaction",
                    code="invalid_common_sender_functional",
                    field="candidate_sender_manifest",
                    remediation="Deduplicate each frozen candidate-sender universe",
                )
            candidate_manifest.append(
                (
                    raw_receiver.strip(),
                    raw_interaction.strip(),
                    normalized_senders,
                )
            )
        manifest = tuple(sorted(candidate_manifest))
        manifest_keys = [
            (receiver, interaction) for receiver, interaction, _ in manifest
        ]
        if not manifest or len(manifest_keys) != len(set(manifest_keys)):
            raise ContractError(
                "candidate sender manifest keys must be non-empty and unique",
                code="invalid_common_sender_functional",
                field="candidate_sender_manifest",
                remediation="Emit one candidate universe per receiver and interaction",
            )
        for receiver in sorted({receiver for receiver, _, _ in manifest}):
            receiver_interactions = tuple(
                interaction
                for candidate_receiver, interaction, _ in manifest
                if candidate_receiver == receiver
            )
            if receiver_interactions != frozen_interactions:
                raise ContractError(
                    "every receiver must bind the complete frozen interaction universe",
                    code="invalid_common_sender_functional",
                    field="candidate_sender_manifest",
                    remediation="Materialize zero-row interactions as not estimable",
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
        expected_prior_keys = {
            (receiver, interaction, sender)
            for receiver, interaction, sender_ids in manifest
            for sender in sender_ids
        }
        if len(keys) != len(set(keys)) or set(keys) != expected_prior_keys:
            raise ContractError(
                "sender prior keys must exactly equal the frozen candidate manifest",
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
        supports = tuple(
            sorted(
                self.contrast_supports,
                key=lambda item: (item.receiver, item.interaction_id),
            )
        )
        if any(
            not isinstance(support, InteractionLigandContrastSupport)
            for support in supports
        ):
            raise ContractError(
                "contrast_supports must contain producer-owned support artifacts",
                code="invalid_common_sender_functional",
                field="contrast_supports",
                remediation="Fit interaction supports from training availability",
            )
        for support in supports:
            support._require_intact()
        expected_support_keys = {
            (receiver, interaction) for receiver, interaction, _ in manifest
        }
        observed_support_keys = {
            (support.receiver, support.interaction_id) for support in supports
        }
        if (
            len(observed_support_keys) != len(supports)
            or observed_support_keys != expected_support_keys
        ):
            raise ContractError(
                "contrast support keys must equal unique sender-prior interactions",
                code="invalid_common_sender_functional",
                field="contrast_supports",
                remediation="Emit one support per receiver and interaction",
            )
        minimum_complete = max(2, self.parameters.min_subjects)
        for support in supports:
            if (
                not set(support.complete_subject_ids).issubset(subjects)
                or support.minimum_complete_subjects != minimum_complete
                or support.contrast_weights != contrast_weights
                or support.ligand_contrast_confidence_level
                != self.parameters.ligand_contrast_confidence_level
                or support.ligand_contrast_minimum_effect
                != self.parameters.ligand_contrast_minimum_effect
                or support.multiplicity_method != _SENDER_CONTRAST_MULTIPLICITY_METHOD
                or support.multiplicity_scope != _SENDER_CONTRAST_MULTIPLICITY_SCOPE
            ):
                raise ContractError(
                    "contrast support lineage does not match its functional",
                    code="invalid_common_sender_functional",
                    field="contrast_supports",
                    remediation="Refit supports from the exact functional inputs",
                )

        for receiver in sorted({receiver for receiver, _ in expected_support_keys}):
            interaction_ids = tuple(
                sorted(
                    interaction_id
                    for candidate_receiver, interaction_id in expected_support_keys
                    if candidate_receiver == receiver
                )
            )
            receiver_supports = tuple(
                support for support in supports if support.receiver == receiver
            )
            receiver_candidate_manifest = tuple(
                (interaction, sender_ids)
                for candidate_receiver, interaction, sender_ids in manifest
                if candidate_receiver == receiver
            )
            expected_family_id = _sender_contrast_family_id(
                receiver=receiver,
                interaction_ids=interaction_ids,
                candidate_sender_ids_by_interaction=receiver_candidate_manifest,
                training_subject_ids=subjects,
                contrast_weights=contrast_weights,
                filter_universe_id=self.filter_universe_id,
                minimum_complete_subjects=minimum_complete,
                ligand_contrast_confidence_level=(
                    self.parameters.ligand_contrast_confidence_level
                ),
                ligand_contrast_minimum_effect=(
                    self.parameters.ligand_contrast_minimum_effect
                ),
            )
            expected_adjustments = _holm_step_down_adjustments(
                {
                    support.interaction_id: support.raw_one_sided_p_value
                    for support in receiver_supports
                }
            )
            familywise_alpha = _sender_contrast_familywise_alpha(
                self.parameters.ligand_contrast_confidence_level
            )
            for support in receiver_supports:
                expected_rank, expected_adjusted_p = expected_adjustments[
                    support.interaction_id
                ]
                expected_status = (
                    SenderContrastSupportStatus.NOT_ESTIMABLE
                    if support.raw_one_sided_p_value is None
                    else SenderContrastSupportStatus.SUPPORTED
                    if expected_adjusted_p < familywise_alpha
                    else SenderContrastSupportStatus.UNSUPPORTED
                )
                expected_reason = (
                    "insufficient_complete_subject_ligand_contrasts"
                    if expected_status is SenderContrastSupportStatus.NOT_ESTIMABLE
                    else None
                    if expected_status is SenderContrastSupportStatus.SUPPORTED
                    else ("ligand_contrast_holm_adjusted_p_not_below_familywise_alpha")
                )
                if (
                    support.multiplicity_family_id != expected_family_id
                    or support.multiplicity_family_size != len(interaction_ids)
                    or support.holm_rank != expected_rank
                    or not math.isclose(
                        support.holm_adjusted_p_value,
                        expected_adjusted_p,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or SenderContrastSupportStatus(support.status)
                    is not expected_status
                    or support.reason_code != expected_reason
                ):
                    raise ContractError(
                        "contrast support Holm family is not jointly reproducible",
                        code="invalid_common_sender_functional",
                        field="contrast_supports",
                        remediation=("Jointly refit every frozen receiver interaction"),
                    )
        if self.schema_version != "3.0.0":
            raise ContractError(
                "common sender functional schema_version must be 3.0.0",
                code="invalid_common_sender_functional",
                field="schema_version",
                remediation="Use the released common-sender v3 functional schema",
            )
        payload = {
            "candidate_priors": [prior.to_dict() for prior in priors],
            "contrast_supports": [support.to_dict() for support in supports],
            "contrast_manifest_id": contrast_manifest_id,
            "contrast_spec": self.contrast.to_dict(),
            "contrast_name": self.contrast.name,
            "contrast_weights": [
                [context_id, weight] for context_id, weight in contrast_weights
            ],
            "context_ids": list(contexts),
            "candidate_sender_manifest": [
                [receiver, interaction, list(sender_ids)]
                for receiver, interaction, sender_ids in manifest
            ],
            "filter_universe_id": self.filter_universe_id,
            "frozen_interaction_ids": list(frozen_interactions),
            "parameter_manifest_id": self.parameters.parameter_manifest_id,
            "schema_version": self.schema_version,
            "training_availability_digest": training_availability_digest,
            "training_input_digest": training_input_digest,
            "training_subject_ids": list(subjects),
        }
        if len(contrast_contexts) > 2:
            payload["contrast_context_ids"] = [
                [node, context_id] for node, context_id in contrast_contexts
            ]
        object.__setattr__(self, "contrast_context_ids", contrast_contexts)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "contrast_name", self.contrast.name)
        object.__setattr__(self, "contrast_manifest_id", contrast_manifest_id)
        object.__setattr__(self, "contrast_weights", contrast_weights)
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "frozen_interaction_ids", frozen_interactions)
        object.__setattr__(self, "candidate_sender_manifest", manifest)
        object.__setattr__(
            self, "training_availability_digest", training_availability_digest
        )
        object.__setattr__(self, "training_input_digest", training_input_digest)
        object.__setattr__(self, "candidate_priors", priors)
        object.__setattr__(self, "contrast_supports", supports)
        object.__setattr__(self, "_producer_marker", _SENDER_FUNCTIONAL_PRODUCER)
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
            for support in self.contrast_supports:
                support._require_intact()
            repeated = ContrastCommonSenderFunctional(
                _producer_token=_SENDER_FUNCTIONAL_PRODUCER_TOKEN,
                contrast=self.contrast,
                contrast_context_ids=self.contrast_context_ids,
                training_subject_ids=self.training_subject_ids,
                filter_universe_id=self.filter_universe_id,
                frozen_interaction_ids=self.frozen_interaction_ids,
                candidate_sender_manifest=self.candidate_sender_manifest,
                training_availability_digest=self.training_availability_digest,
                training_input_digest=self.training_input_digest,
                parameters=self.parameters,
                candidate_priors=self.candidate_priors,
                contrast_supports=self.contrast_supports,
                schema_version=self.schema_version,
            )
            valid = (
                self._producer_marker == _SENDER_FUNCTIONAL_PRODUCER
                and isinstance(self.contrast_context_ids, tuple)
                and isinstance(self.training_subject_ids, tuple)
                and isinstance(self.frozen_interaction_ids, tuple)
                and isinstance(self.candidate_sender_manifest, tuple)
                and isinstance(self.candidate_priors, tuple)
                and isinstance(self.contrast_supports, tuple)
                and self.contrast_context_ids == repeated.contrast_context_ids
                and self.training_subject_ids == repeated.training_subject_ids
                and self.filter_universe_id == repeated.filter_universe_id
                and self.frozen_interaction_ids == repeated.frozen_interaction_ids
                and self.candidate_sender_manifest
                == repeated.candidate_sender_manifest
                and self.training_availability_digest
                == repeated.training_availability_digest
                and self.training_input_digest == repeated.training_input_digest
                and self.candidate_priors == repeated.candidate_priors
                and self.contrast_supports == repeated.contrast_supports
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
            "frozen_interaction_ids": list(self.frozen_interaction_ids),
            "candidate_sender_manifest": [
                [receiver, interaction, list(sender_ids)]
                for receiver, interaction, sender_ids in self.candidate_sender_manifest
            ],
            "training_availability_digest": self.training_availability_digest,
            "training_input_digest": self.training_input_digest,
            "parameters": self.parameters.to_dict(),
            "candidate_priors": [prior.to_dict() for prior in self.candidate_priors],
            "contrast_supports": [
                support.to_dict() for support in self.contrast_supports
            ],
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
            if (
                set(group["status"]) != {CommonSenderApplicationStatus.OK.value}
                or group["reason_code"].notna().any()
            ):
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
                    (value - maximum) / self.functional.parameters.softmax_temperature
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
                if (
                    not (
                        group["status"] == SenderAssignmentStatus.MISSING_EVIDENCE.value
                    ).all()
                    or group["normalized_entropy"].notna().any()
                ):
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
