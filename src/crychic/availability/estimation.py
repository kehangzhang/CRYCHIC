"""State and capture-weighted ecosystem availability estimation."""

from __future__ import annotations

import math
from collections.abc import Mapping

from crychic.resources import Interaction

from .contracts import (
    AbundanceObservation,
    AvailabilityEstimate,
    AvailabilityParameters,
    AvailabilityStatus,
    EntityAvailability,
    GeneObservation,
    SubunitAvailability,
)
from .transforms import generalized_harmonic_softmin, single_gene_availability

_MISSING_PRIORITY = (
    AvailabilityStatus.QC_FAILURE,
    AvailabilityStatus.SAMPLING_ZERO,
    AvailabilityStatus.LOW_CELL_COUNT,
    AvailabilityStatus.UNAVAILABLE,
)
_DEFAULT_PARAMETERS = AvailabilityParameters()


def _dominant_missing(statuses: tuple[AvailabilityStatus, ...]) -> AvailabilityStatus:
    for candidate in _MISSING_PRIORITY:
        if candidate in statuses:
            return candidate
    return AvailabilityStatus.UNAVAILABLE


def estimate_entity_availability(
    name: str,
    subunits: tuple[str, ...],
    observations: Mapping[str, GeneObservation],
    *,
    parameters: AvailabilityParameters = _DEFAULT_PARAMETERS,
) -> EntityAvailability:
    """Estimate a single- or multi-subunit entity without imputing missing state."""

    components: list[SubunitAvailability] = []
    for gene in sorted(set(subunits)):
        observation = observations.get(gene)
        if observation is None:
            components.append(
                SubunitAvailability(
                    gene=gene,
                    value=None,
                    hill_value=None,
                    shrunk_detection=None,
                    status=AvailabilityStatus.UNAVAILABLE,
                    qc_flags=("gene_not_mapped",),
                )
            )
            continue
        status = AvailabilityStatus(observation.status)
        if status is AvailabilityStatus.CONFIRMED_ABSENCE:
            components.append(
                SubunitAvailability(
                    gene=gene,
                    value=0.0,
                    hill_value=0.0,
                    shrunk_detection=0.0,
                    status=status,
                    qc_flags=observation.qc_flags,
                )
            )
            continue
        if status is not AvailabilityStatus.OBSERVED:
            components.append(
                SubunitAvailability(
                    gene=gene,
                    value=None,
                    hill_value=None,
                    shrunk_detection=None,
                    status=status,
                    qc_flags=observation.qc_flags,
                )
            )
            continue
        if observation.expression is None or observation.detection_fraction is None:
            raise RuntimeError(
                "GeneObservation validation failed to protect estimation"
            )
        value, hill_value, shrunk = single_gene_availability(
            observation.expression,
            observation.detection_fraction,
            observation.n_cells,
            hill=parameters.hill,
            detection=parameters.detection,
        )
        components.append(
            SubunitAvailability(
                gene=gene,
                value=value,
                hill_value=hill_value,
                shrunk_detection=shrunk,
                status=status,
                qc_flags=observation.qc_flags,
            )
        )
    statuses = tuple(component.status for component in components)
    qc_flags = tuple(
        sorted({flag for component in components for flag in component.qc_flags})
    )
    if AvailabilityStatus.CONFIRMED_ABSENCE in statuses:
        state = 0.0
        status = AvailabilityStatus.CONFIRMED_ABSENCE
    elif all(value is AvailabilityStatus.OBSERVED for value in statuses):
        state = generalized_harmonic_softmin(
            (
                component.value
                for component in components
                if component.value is not None
            ),
            power=parameters.complex_power,
            epsilon=parameters.complex_epsilon,
        )
        status = AvailabilityStatus.OBSERVED
    else:
        state = None
        status = _dominant_missing(statuses)
    return EntityAvailability(
        name=name,
        subunits=tuple(sorted(set(subunits))),
        state=state,
        status=status,
        components=tuple(components),
        qc_flags=qc_flags,
    )


def _abundance_proxy(
    abundance: AbundanceObservation,
) -> tuple[float | None, AvailabilityStatus]:
    status = AvailabilityStatus(abundance.status)
    if status is AvailabilityStatus.CONFIRMED_ABSENCE:
        return 0.0, status
    if status is not AvailabilityStatus.OBSERVED:
        return None, status
    if (
        abundance.captured_proportion is None
        or abundance.presence_probability is None
        or abundance.capture_weight is None
    ):
        raise RuntimeError(
            "AbundanceObservation validation failed to protect estimation"
        )
    return (
        abundance.captured_proportion
        * abundance.presence_probability
        * abundance.capture_weight,
        status,
    )


def _state_status(
    ligand: EntityAvailability, receptor: EntityAvailability
) -> AvailabilityStatus:
    statuses = (ligand.status, receptor.status)
    if AvailabilityStatus.CONFIRMED_ABSENCE in statuses:
        return AvailabilityStatus.CONFIRMED_ABSENCE
    if all(value is AvailabilityStatus.OBSERVED for value in statuses):
        return AvailabilityStatus.OBSERVED
    return _dominant_missing(statuses)


def estimate_interaction_availability(
    interaction: Interaction,
    sender_observations: Mapping[str, GeneObservation],
    receiver_observations: Mapping[str, GeneObservation],
    sender_abundance: AbundanceObservation,
    receiver_abundance: AbundanceObservation,
    *,
    parameters: AvailabilityParameters = _DEFAULT_PARAMETERS,
) -> AvailabilityEstimate:
    """Estimate interaction state and two-part ecosystem availability.

    Inputs are baseline or training-fold expression/detection and abundance
    only. No receiver response or outcome contrast is accepted by this API.
    """

    ligand = estimate_entity_availability(
        interaction.ligand_name,
        interaction.ligand_subunits,
        sender_observations,
        parameters=parameters,
    )
    receptor = estimate_entity_availability(
        interaction.receptor_name,
        interaction.receptor_subunits,
        receiver_observations,
        parameters=parameters,
    )
    state_status = _state_status(ligand, receptor)
    if state_status is AvailabilityStatus.CONFIRMED_ABSENCE:
        state = 0.0
    elif state_status is AvailabilityStatus.OBSERVED:
        if ligand.state is None or receptor.state is None:
            raise RuntimeError("Entity status/value contract is inconsistent")
        state = ligand.state * receptor.state
    else:
        state = None

    sender_proxy, sender_status = _abundance_proxy(sender_abundance)
    receiver_proxy, receiver_status = _abundance_proxy(receiver_abundance)
    abundance_statuses = (sender_status, receiver_status)
    if state_status is AvailabilityStatus.CONFIRMED_ABSENCE or (
        AvailabilityStatus.CONFIRMED_ABSENCE in abundance_statuses
    ):
        ecosystem = 0.0
        ecosystem_status = AvailabilityStatus.CONFIRMED_ABSENCE
    elif state is None:
        ecosystem = None
        ecosystem_status = state_status
    elif sender_proxy is None or receiver_proxy is None:
        ecosystem = None
        ecosystem_status = _dominant_missing(abundance_statuses)
    else:
        ecosystem = state * math.sqrt(sender_proxy * receiver_proxy)
        ecosystem_status = AvailabilityStatus.OBSERVED

    calibrated = (
        sender_abundance.absolute_calibrated
        and receiver_abundance.absolute_calibrated
    )
    label = (
        "absolute_abundance_calibrated"
        if calibrated
        else "capture_weighted_ecosystem_proxy"
    )
    qc_flags = {
        *ligand.qc_flags,
        *receptor.qc_flags,
        *sender_abundance.qc_flags,
        *receiver_abundance.qc_flags,
    }
    if ecosystem_status is not AvailabilityStatus.OBSERVED:
        qc_flags.add(f"ecosystem_{ecosystem_status.value}")
    return AvailabilityEstimate(
        interaction_id=interaction.interaction_id,
        ligand=ligand,
        receptor=receptor,
        state=state,
        ecosystem=ecosystem,
        state_status=state_status,
        ecosystem_status=ecosystem_status,
        sender_abundance_proxy=sender_proxy,
        receiver_abundance_proxy=receiver_proxy,
        ecosystem_label=label,
        qc_flags=tuple(sorted(qc_flags)),
    )
