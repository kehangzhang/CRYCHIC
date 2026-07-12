"""Immutable availability inputs, components, and estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from crychic.core import ContractError, stable_id


class AvailabilityStatus(StrEnum):
    """Status values that keep biological absence separate from missing data."""

    OBSERVED = "observed"
    LOW_CELL_COUNT = "low_cell_count"
    SAMPLING_ZERO = "sampling_zero"
    QC_FAILURE = "qc_failure"
    CONFIRMED_ABSENCE = "confirmed_absence"
    UNAVAILABLE = "unavailable"


class InteractionFilterPolicy(StrEnum):
    """Training-only policies used to define an LR interaction universe."""

    POOLED_SUPPORT_V1 = "training_pooled_support_v1"
    EXPLORATORY_POOLED_TOP_K_V1 = "exploratory_training_pooled_top_k_v1"


class InteractionFilterApplication(StrEnum):
    """Whether availability selected or only applied an interaction universe."""

    TRAINING_SELECTION_V1 = "training_selection_v1"
    FROZEN_APPLICATION_V1 = "frozen_application_v1"


def _stable_identifiers(
    values: tuple[str, ...], *, field_name: str, allow_empty: bool
) -> tuple[str, ...]:
    if not allow_empty and not values:
        raise ContractError(
            f"{field_name} must not be empty",
            code="invalid_frozen_interaction_universe",
            field=field_name,
            remediation="Record the biological subjects used to fit the filter",
        )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ContractError(
            f"{field_name} must contain non-empty strings",
            code="invalid_frozen_interaction_universe",
            field=field_name,
            remediation="Use stable resource and subject identifiers",
        )
    normalized = tuple(value.strip() for value in values)
    if len(normalized) != len(set(normalized)):
        raise ContractError(
            f"{field_name} must contain unique identifiers",
            code="duplicate_frozen_interaction_universe",
            field=field_name,
            remediation="Deduplicate the training filter manifest before application",
        )
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenInteractionUniverse:
    """Fold-frozen interaction IDs and training provenance for test application."""

    interaction_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    resource_id: str
    resource_version: str
    resource_manifest_digest: str
    min_pooled_availability: float
    max_interactions: int | None
    selection_policy: InteractionFilterPolicy | str
    schema_version: str = "1.0.0"
    filter_universe_id: str = field(init=False)

    def __post_init__(self) -> None:
        interactions = _stable_identifiers(
            tuple(self.interaction_ids),
            field_name="interaction_ids",
            allow_empty=True,
        )
        subjects = _stable_identifiers(
            tuple(self.training_subject_ids),
            field_name="training_subject_ids",
            allow_empty=False,
        )
        resource_fields = (
            "resource_id",
            "resource_version",
            "resource_manifest_digest",
        )
        for field_name in resource_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(
                    f"{field_name} must be a non-empty string",
                    code="invalid_frozen_interaction_universe",
                    field=field_name,
                    remediation="Freeze the resource identity with the filter universe",
                )
            object.__setattr__(self, field_name, value.strip())
        threshold = float(self.min_pooled_availability)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ContractError(
                "min_pooled_availability must lie in [0, 1]",
                code="invalid_frozen_interaction_universe",
                field="min_pooled_availability",
                remediation="Record the exact training-fold pooled threshold",
            )
        maximum = self.max_interactions
        if maximum is not None and (
            isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
        ):
            raise ContractError(
                "max_interactions must be a positive integer or None",
                code="invalid_frozen_interaction_universe",
                field="max_interactions",
                remediation="Record the exact exploratory training cap",
            )
        policy = InteractionFilterPolicy(self.selection_policy)
        expected_policy = (
            InteractionFilterPolicy.POOLED_SUPPORT_V1
            if maximum is None
            else InteractionFilterPolicy.EXPLORATORY_POOLED_TOP_K_V1
        )
        if policy is not expected_policy:
            raise ContractError(
                "selection_policy does not match max_interactions",
                code="invalid_frozen_interaction_universe",
                field="selection_policy",
                remediation=(
                    "Use pooled_support_v1 without a cap or the explicit "
                    "exploratory top-k policy with a cap"
                ),
            )
        if self.schema_version != "1.0.0":
            raise ContractError(
                "FrozenInteractionUniverse schema_version must be 1.0.0",
                code="invalid_frozen_interaction_universe",
                field="schema_version",
                remediation="Use the released availability filter manifest schema",
            )
        payload = {
            "interaction_ids": list(interactions),
            "max_interactions": maximum,
            "min_pooled_availability": threshold,
            "resource_id": self.resource_id,
            "resource_manifest_digest": self.resource_manifest_digest,
            "resource_version": self.resource_version,
            "schema_version": self.schema_version,
            "selection_policy": policy.value,
            "training_subject_ids": list(subjects),
        }
        object.__setattr__(self, "interaction_ids", interactions)
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "min_pooled_availability", threshold)
        object.__setattr__(self, "selection_policy", policy)
        object.__setattr__(
            self,
            "filter_universe_id",
            stable_id(
                "availability_filter_universe",
                payload,
                schema_version=self.schema_version,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready frozen filter manifest."""

        return {
            "filter_universe_id": self.filter_universe_id,
            "schema_version": self.schema_version,
            "selection_policy": InteractionFilterPolicy(self.selection_policy).value,
            "interaction_ids": list(self.interaction_ids),
            "training_subject_ids": list(self.training_subject_ids),
            "resource_id": self.resource_id,
            "resource_version": self.resource_version,
            "resource_manifest_digest": self.resource_manifest_digest,
            "min_pooled_availability": self.min_pooled_availability,
            "max_interactions": self.max_interactions,
        }


@dataclass(frozen=True, slots=True)
class HillParameters:
    """Parameters of a non-negative Hill saturation transform."""

    half_saturation: float = 1.0
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.half_saturation <= 0 or self.coefficient <= 0:
            raise ContractError(
                "Hill parameters must be positive",
                code="invalid_hill_parameters",
                field="half_saturation",
                remediation="Use positive saturation and coefficient values",
            )


@dataclass(frozen=True, slots=True)
class DetectionShrinkage:
    """Beta-prior detection shrinkage and its availability exponent."""

    alpha: float = 0.5
    beta: float = 0.5
    exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0 or self.exponent < 0:
            raise ContractError(
                "Detection prior parameters must be positive and exponent non-negative",
                code="invalid_detection_shrinkage",
                field="detection_shrinkage",
                remediation="Use alpha,beta > 0 and exponent >= 0",
            )


@dataclass(frozen=True, slots=True)
class AvailabilityParameters:
    """Fold-frozen numerical parameters for availability estimation."""

    hill: HillParameters = HillParameters()
    detection: DetectionShrinkage = DetectionShrinkage()
    complex_power: float = 4.0
    complex_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.complex_power <= 0 or self.complex_epsilon <= 0:
            raise ContractError(
                "Complex soft-min power and epsilon must be positive",
                code="invalid_complex_parameters",
                field="complex_power",
                remediation="Use a positive generalized-harmonic power and epsilon",
            )


@dataclass(frozen=True, slots=True)
class GeneObservation:
    """Conditional expression and detection evidence for one sample/cell type."""

    expression: float | None
    detection_fraction: float | None
    n_cells: int
    status: AvailabilityStatus = AvailabilityStatus.OBSERVED
    qc_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", AvailabilityStatus(self.status))
        object.__setattr__(self, "qc_flags", tuple(sorted(set(self.qc_flags))))
        if self.n_cells < 0:
            raise ContractError(
                "Gene observation n_cells cannot be negative",
                code="invalid_availability_input",
                field="n_cells",
                remediation="Use the captured cell count for this sample/cell type",
            )
        if self.expression is not None and self.expression < 0:
            raise ContractError(
                "Availability expression cannot be negative",
                code="invalid_availability_input",
                field="expression",
                remediation="Provide a non-negative normalized expression measure",
            )
        if self.detection_fraction is not None and not (
            0 <= self.detection_fraction <= 1
        ):
            raise ContractError(
                "Detection fraction must lie in [0, 1]",
                code="invalid_availability_input",
                field="detection_fraction",
                remediation="Provide detected cells divided by captured cells",
            )
        if self.status is AvailabilityStatus.OBSERVED and (
            self.n_cells == 0
            or self.expression is None
            or self.detection_fraction is None
        ):
            raise ContractError(
                "Observed availability requires cells, expression, and detection",
                code="incomplete_availability_input",
                field="status",
                remediation=(
                    "Use an explicit missingness status when state is unavailable"
                ),
            )


@dataclass(frozen=True, slots=True)
class AbundanceObservation:
    """Two-part capture-weighted cell-type abundance evidence."""

    captured_proportion: float | None
    presence_probability: float | None = 1.0
    capture_weight: float | None = 1.0
    status: AvailabilityStatus = AvailabilityStatus.OBSERVED
    absolute_calibrated: bool = False
    qc_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", AvailabilityStatus(self.status))
        object.__setattr__(self, "qc_flags", tuple(sorted(set(self.qc_flags))))
        for field_name in (
            "captured_proportion",
            "presence_probability",
            "capture_weight",
        ):
            value = getattr(self, field_name)
            if value is not None and not 0 <= value <= 1:
                raise ContractError(
                    f"{field_name} must lie in [0, 1]",
                    code="invalid_abundance_input",
                    field=field_name,
                    remediation=(
                        "Provide an explicit capture-weighted abundance component"
                    ),
                )
        if self.status is AvailabilityStatus.OBSERVED and any(
            value is None
            for value in (
                self.captured_proportion,
                self.presence_probability,
                self.capture_weight,
            )
        ):
            raise ContractError(
                "Observed abundance requires proportion, presence, and capture weight",
                code="incomplete_abundance_input",
                field="status",
                remediation=(
                    "Use an explicit missingness status when abundance is unavailable"
                ),
            )


@dataclass(frozen=True, slots=True)
class SubunitAvailability:
    """One required gene's state availability and shrinkage diagnostics."""

    gene: str
    value: float | None
    hill_value: float | None
    shrunk_detection: float | None
    status: AvailabilityStatus
    qc_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityAvailability:
    """State availability of one ligand or receptor entity."""

    name: str
    subunits: tuple[str, ...]
    state: float | None
    status: AvailabilityStatus
    components: tuple[SubunitAvailability, ...]
    qc_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AvailabilityEstimate:
    """State and ecosystem availability for one sender-receiver interaction."""

    interaction_id: str
    ligand: EntityAvailability
    receptor: EntityAvailability
    state: float | None
    ecosystem: float | None
    state_status: AvailabilityStatus
    ecosystem_status: AvailabilityStatus
    sender_abundance_proxy: float | None
    receiver_abundance_proxy: float | None
    ecosystem_label: str
    qc_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "state",
            "ecosystem",
            "sender_abundance_proxy",
            "receiver_abundance_proxy",
        ):
            value = getattr(self, field_name)
            if value is not None and not 0 <= value <= 1:
                raise ContractError(
                    f"{field_name} must lie in [0, 1]",
                    code="invalid_availability_estimate",
                    field=field_name,
                    remediation="Inspect availability transform and abundance inputs",
                )
        object.__setattr__(self, "qc_flags", tuple(sorted(set(self.qc_flags))))
