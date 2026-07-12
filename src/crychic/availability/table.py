"""Vectorized resource availability over sample-cell-type aggregates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from crychic.design import context_id
from crychic.pseudobulk import ExploratoryAggregate, PseudobulkDataset
from crychic.resources import Interaction, ResourceBundle

from .contracts import AvailabilityParameters

Aggregate: TypeAlias = PseudobulkDataset | ExploratoryAggregate
_DEFAULT_PARAMETERS = AvailabilityParameters()


@dataclass(frozen=True, slots=True)
class BatchAvailability:
    """Mapped interactions and sample-level state/ecosystem components."""

    sample_interactions: pd.DataFrame
    mapping_summary: pd.DataFrame
    resource_id: str
    resource_version: str
    detection_available: bool


def _entity_values(
    gene_values: np.ndarray,
    indices: Sequence[int],
    *,
    power: float,
    epsilon: float,
) -> npt.NDArray[np.float64]:
    values = gene_values[:, indices]
    if values.shape[1] == 1:
        return np.asarray(values[:, 0], dtype=np.float64).copy()
    shifted_inverse = np.mean((values + epsilon) ** (-power), axis=1)
    result = shifted_inverse ** (-1.0 / power) - epsilon
    return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float64)


def _context_value(row: pd.Series, key: str) -> object:
    if key != "context":
        return row[key]
    canonical = row["context"]
    if isinstance(canonical, tuple):
        for context_key, value in canonical:
            if context_key == key:
                return value
    raise ValueError("canonical context tuple does not contain the declared key")


def _context_id(row: pd.Series, context_keys: Sequence[str]) -> str:
    payload = {key: _context_value(row, key) for key in context_keys}
    return context_id(payload, context_keys)


def _empty_sample_table(context_keys: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id",
            "subject_id",
            "context_id",
            *context_keys,
            "sender",
            "receiver",
            "interaction_id",
            "source_interaction_id",
            "ligand",
            "receptor",
            "pathway",
            "ligand_availability",
            "receptor_availability",
            "sender_proportion",
            "receiver_proportion",
            "availability_state",
            "availability_ecosystem",
            "state_status",
            "ecosystem_status",
            "ecosystem_label",
        ]
    )


def estimate_bundle_availability(
    aggregate: Aggregate,
    bundle: ResourceBundle,
    *,
    context_keys: Sequence[str],
    parameters: AvailabilityParameters = _DEFAULT_PARAMETERS,
    min_pooled_availability: float = 0.01,
    max_interactions: int | None = None,
) -> BatchAvailability:
    """Estimate all mapped LR components without using response outcomes.

    The pooled-support filter is independent of context labels. Only units with
    eligible state expression enter the long score table; missing grid units
    remain explicit in ``aggregate.unit_metadata`` and are never materialized
    as measured zero rows.
    """

    if not 0 <= min_pooled_availability <= 1:
        raise ValueError("min_pooled_availability must lie in [0, 1]")
    if max_interactions is not None and max_interactions < 1:
        raise ValueError("max_interactions must be positive when provided")
    if not context_keys:
        raise ValueError("context_keys must contain at least one field")
    missing_context = set(context_keys).difference(aggregate.unit_metadata.columns)
    missing_context.discard("context")
    if missing_context:
        raise ValueError(
            f"aggregate metadata lacks contexts: {sorted(missing_context)}"
        )

    feature_index = {gene: index for index, gene in enumerate(aggregate.feature_ids)}
    mapped: list[Interaction] = []
    dropped_unmapped: list[str] = []
    needed_genes: set[str] = set()
    for interaction in bundle.interactions:
        required = (*interaction.ligand_subunits, *interaction.receptor_subunits)
        if any(gene not in feature_index for gene in required):
            dropped_unmapped.append(interaction.interaction_id)
            continue
        mapped.append(interaction)
        needed_genes.update(required)

    ordered_genes = tuple(sorted(needed_genes))
    selected_columns = np.array(
        [feature_index[gene] for gene in ordered_genes], dtype=int
    )
    local_index = {gene: index for index, gene in enumerate(ordered_genes)}
    matrix = (
        aggregate.counts
        if isinstance(aggregate, PseudobulkDataset)
        else aggregate.mean_expression
    )
    selected = matrix[:, selected_columns].toarray().astype(np.float64, copy=False)

    metadata = (
        aggregate.unit_metadata.set_index("unit_id", drop=False)
        .loc[list(aggregate.matrix_unit_ids)]
        .reset_index(drop=True)
    )
    eligible = metadata["state_eligible"].to_numpy(dtype=bool)
    n_cells = metadata["n_cells"].to_numpy(dtype=float)
    if isinstance(aggregate, PseudobulkDataset):
        library = np.asarray(aggregate.counts.sum(axis=1)).ravel().astype(float)
        selected = np.divide(
            selected * 1_000_000.0,
            library[:, None],
            out=np.zeros_like(selected),
            where=library[:, None] > 0,
        )
        detection = aggregate.detection_fraction[:, selected_columns].toarray()
        detection_available = True
    else:
        if aggregate.detection_fraction is None:
            detection = np.ones_like(selected)
            detection_available = False
        else:
            detection = aggregate.detection_fraction[:, selected_columns].toarray()
            detection_available = True

    coefficient = parameters.hill.coefficient
    half = parameters.hill.half_saturation
    powered = selected**coefficient
    hill = np.divide(
        powered,
        powered + half**coefficient,
        out=np.zeros_like(powered),
        where=(powered + half**coefficient) > 0,
    )
    if detection_available:
        shrunk_detection = (
            detection * n_cells[:, None] + parameters.detection.alpha
        ) / (
            n_cells[:, None]
            + parameters.detection.alpha
            + parameters.detection.beta
        )
    else:
        # Without declared zero/detection semantics, detection is neutral.
        shrunk_detection = np.ones_like(selected)
    gene_values = hill * shrunk_detection**parameters.detection.exponent
    gene_values[~eligible, :] = np.nan

    ligand_columns: list[np.ndarray] = []
    receptor_columns: list[np.ndarray] = []
    supported: list[Interaction] = []
    dropped_no_support: list[str] = []
    for interaction in mapped:
        ligand = _entity_values(
            gene_values,
            [local_index[gene] for gene in interaction.ligand_subunits],
            power=parameters.complex_power,
            epsilon=parameters.complex_epsilon,
        )
        receptor = _entity_values(
            gene_values,
            [local_index[gene] for gene in interaction.receptor_subunits],
            power=parameters.complex_power,
            epsilon=parameters.complex_epsilon,
        )
        ligand_max = float(np.nanmax(ligand)) if np.isfinite(ligand).any() else 0.0
        receptor_max = (
            float(np.nanmax(receptor)) if np.isfinite(receptor).any() else 0.0
        )
        if min(ligand_max, receptor_max) < min_pooled_availability:
            dropped_no_support.append(interaction.interaction_id)
            continue
        supported.append(interaction)
        ligand_columns.append(ligand)
        receptor_columns.append(receptor)

    if max_interactions is not None and len(supported) > max_interactions:
        pooled = np.array(
            [
                math.sqrt(float(np.nanmax(ligand)) * float(np.nanmax(receptor)))
                for ligand, receptor in zip(
                    ligand_columns, receptor_columns, strict=True
                )
            ]
        )
        order = sorted(
            range(len(supported)),
            key=lambda index: (-pooled[index], supported[index].interaction_id),
        )[:max_interactions]
        removed = set(range(len(supported))).difference(order)
        dropped_no_support.extend(supported[index].interaction_id for index in removed)
        supported = [supported[index] for index in order]
        ligand_columns = [ligand_columns[index] for index in order]
        receptor_columns = [receptor_columns[index] for index in order]

    mapping_summary = pd.DataFrame(
        {
            "metric": [
                "source_interactions",
                "gene_mapped_interactions",
                "pooled_supported_interactions",
                "dropped_unmapped_interactions",
                "dropped_no_pooled_support_interactions",
            ],
            "value": [
                len(bundle.interactions),
                len(mapped),
                len(supported),
                len(dropped_unmapped),
                len(dropped_no_support),
            ],
        }
    )
    if not supported:
        return BatchAvailability(
            sample_interactions=_empty_sample_table(context_keys),
            mapping_summary=mapping_summary,
            resource_id=bundle.resource_id,
            resource_version=bundle.version,
            detection_available=detection_available,
        )

    ligand_matrix = np.column_stack(ligand_columns)
    receptor_matrix = np.column_stack(receptor_columns)
    usable_rows = metadata.index[
        eligible & metadata["abundance_eligible"].to_numpy(dtype=bool)
    ]
    records: list[pd.DataFrame] = []
    interaction_fields = {
        "interaction_id": [item.interaction_id for item in supported],
        "source_interaction_id": [item.source_interaction_id for item in supported],
        "ligand": [item.ligand_name for item in supported],
        "receptor": [item.receptor_name for item in supported],
        "pathway": [item.pathway for item in supported],
    }
    for sample_id, sample_rows in metadata.loc[usable_rows].groupby(
        "sample_id", sort=False, observed=True
    ):
        row_indices = sample_rows.index.to_numpy(dtype=int)
        for sender_row in row_indices:
            sender = metadata.loc[sender_row]
            ligand = ligand_matrix[sender_row]
            for receiver_row in row_indices:
                receiver = metadata.loc[receiver_row]
                receptor = receptor_matrix[receiver_row]
                state = ligand * receptor
                sender_proportion = float(sender["cell_proportion"])
                receiver_proportion = float(receiver["cell_proportion"])
                ecosystem = state * math.sqrt(sender_proportion * receiver_proportion)
                frame = pd.DataFrame(interaction_fields)
                frame.insert(0, "receiver", receiver["cell_type"])
                frame.insert(0, "sender", sender["cell_type"])
                for key in reversed(context_keys):
                    frame.insert(
                        0, key, [_context_value(sender, key)] * len(frame)
                    )
                frame.insert(0, "context_id", _context_id(sender, context_keys))
                frame.insert(0, "subject_id", sender["subject_id"])
                frame.insert(0, "sample_id", sample_id)
                frame["ligand_availability"] = ligand
                frame["receptor_availability"] = receptor
                frame["sender_proportion"] = sender_proportion
                frame["receiver_proportion"] = receiver_proportion
                frame["availability_state"] = state
                frame["availability_ecosystem"] = ecosystem
                frame["state_status"] = "observed"
                frame["ecosystem_status"] = "observed"
                frame["ecosystem_label"] = "capture_weighted_ecosystem_proxy"
                records.append(frame)
    sample_interactions = pd.concat(records, ignore_index=True)
    for column in (
        "sender",
        "receiver",
        "interaction_id",
        "source_interaction_id",
        "ligand",
        "receptor",
        "pathway",
        "state_status",
        "ecosystem_status",
        "ecosystem_label",
    ):
        sample_interactions[column] = sample_interactions[column].astype("category")
    return BatchAvailability(
        sample_interactions=sample_interactions,
        mapping_summary=mapping_summary,
        resource_id=bundle.resource_id,
        resource_version=bundle.version,
        detection_available=detection_available,
    )
