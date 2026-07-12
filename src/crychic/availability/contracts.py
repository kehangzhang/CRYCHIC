"""Immutable availability inputs, components, and estimates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from crychic.core import ContractError


class AvailabilityStatus(StrEnum):
    """Status values that keep biological absence separate from missing data."""

    OBSERVED = "observed"
    LOW_CELL_COUNT = "low_cell_count"
    SAMPLING_ZERO = "sampling_zero"
    QC_FAILURE = "qc_failure"
    CONFIRMED_ABSENCE = "confirmed_absence"
    UNAVAILABLE = "unavailable"


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
