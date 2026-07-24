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

from .contracts import (
    AvailabilityParameters,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)

Aggregate: TypeAlias = PseudobulkDataset | ExploratoryAggregate
_DEFAULT_PARAMETERS = AvailabilityParameters()
_BATCH_IDENTIFIER_COLUMNS = ("sample_id", "subject_id", "context_id")


def _validate_identifier_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    if table.empty:
        return
    missing_columns = set(columns).difference(table.columns)
    if missing_columns:
        raise ValueError(
            f"{table_name} is missing identifier columns: {sorted(missing_columns)}"
        )
    for column in columns:
        values = table[column]
        if bool(values.isna().any()):
            raise ValueError(
                f"{table_name}.{column} must not contain missing identifiers"
            )
        if any(not str(value).strip() for value in values.tolist()):
            raise ValueError(
                f"{table_name}.{column} must contain non-empty identifiers"
            )


def _normalize_application_subject_ids(values: Sequence[object]) -> tuple[str, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError("application_subject_ids must be non-empty and unique")
    source = pd.Series(raw, dtype="object")
    if bool(source.isna().any()):
        raise ValueError("application_subject_ids must not contain missing identifiers")
    normalized = tuple(str(value) for value in raw)
    if any(not value.strip() for value in normalized):
        raise ValueError("application_subject_ids must contain non-empty identifiers")
    if len(normalized) != len(set(normalized)):
        raise ValueError("application_subject_ids must be non-empty and unique")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class BatchAvailability:
    """Mapped interactions and sample-level state/ecosystem components."""

    sample_interactions: pd.DataFrame
    mapping_summary: pd.DataFrame
    resource_id: str
    resource_version: str
    detection_available: bool
    frozen_interaction_universe: FrozenInteractionUniverse
    filter_application: InteractionFilterApplication
    application_subject_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_interaction_universe, FrozenInteractionUniverse):
            raise TypeError(
                "frozen_interaction_universe must be a FrozenInteractionUniverse"
            )
        application = InteractionFilterApplication(self.filter_application)
        subjects = _normalize_application_subject_ids(self.application_subject_ids)
        _validate_identifier_columns(
            self.sample_interactions,
            _BATCH_IDENTIFIER_COLUMNS,
            table_name="sample_interactions",
        )
        observed_ids = set(
            self.sample_interactions.get("interaction_id", pd.Series(dtype="object"))
            .dropna()
            .astype(str)
        )
        unknown = observed_ids.difference(
            self.frozen_interaction_universe.interaction_ids
        )
        if unknown:
            raise ValueError(
                "availability rows fall outside the frozen interaction universe: "
                f"{sorted(unknown)}"
            )
        if (
            application is InteractionFilterApplication.TRAINING_SELECTION_V1
            and subjects != self.frozen_interaction_universe.training_subject_ids
        ):
            raise ValueError(
                "training filter application subjects must match training provenance"
            )
        object.__setattr__(self, "filter_application", application)
        object.__setattr__(self, "application_subject_ids", subjects)

    @property
    def filter_universe_id(self) -> str:
        """Stable identity of the selected or applied interaction universe."""

        return self.frozen_interaction_universe.filter_universe_id


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
    missing = np.isnan(values).any(axis=1)
    has_zero = np.equal(values, 0.0).any(axis=1)
    shifted_inverse = np.mean((values + epsilon) ** (-power), axis=1)
    result = np.asarray(
        np.clip(shifted_inverse ** (-1.0 / power) - epsilon, 0.0, 1.0),
        dtype=np.float64,
    )
    result[has_zero] = 0.0
    result[missing] = np.nan
    return result


def _absolute_entity_values(
    gene_values: np.ndarray,
    indices: Sequence[int],
    *,
    power: float,
    epsilon: float,
) -> npt.NDArray[np.float64]:
    """Aggregate required absolute-evidence subunits without sharing caches."""

    values = gene_values[:, indices]
    if values.shape[1] == 1:
        return np.asarray(values[:, 0], dtype=np.float64).copy()
    missing = np.isnan(values).any(axis=1)
    has_zero = np.equal(values, 0.0).any(axis=1)
    shifted_inverse = np.mean((values + epsilon) ** (-power), axis=1)
    result = np.asarray(
        np.clip(shifted_inverse ** (-1.0 / power) - epsilon, 0.0, 1.0),
        dtype=np.float64,
    )
    result[has_zero] = 0.0
    result[missing] = np.nan
    return result


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
    return str(context_id(payload, context_keys))


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
            "ligand_absolute_evidence",
            "receptor_absolute_evidence",
            "absolute_lr_activity",
            "sender_proportion",
            "receiver_proportion",
            "availability_state",
            "availability_ecosystem",
            "state_status",
            "state_reason_code",
            "ecosystem_status",
            "ecosystem_reason_code",
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
    frozen_interaction_universe: FrozenInteractionUniverse | None = None,
) -> BatchAvailability:
    """Estimate all mapped LR components without using response outcomes.

    Without a frozen universe, pooled support and the legacy optional top-k cap
    are training-data operations. The cap is retained only as an explicitly
    exploratory policy. With ``frozen_interaction_universe``, resource IDs are
    applied directly and neither pooled support nor top-k is relearned from the
    application data.
    """

    if not 0 <= min_pooled_availability <= 1:
        raise ValueError("min_pooled_availability must lie in [0, 1]")
    if max_interactions is not None and (
        isinstance(max_interactions, bool)
        or not isinstance(max_interactions, int)
        or max_interactions < 1
    ):
        raise ValueError("max_interactions must be a positive integer when provided")
    if frozen_interaction_universe is not None and not isinstance(
        frozen_interaction_universe, FrozenInteractionUniverse
    ):
        raise TypeError(
            "frozen_interaction_universe must be a FrozenInteractionUniverse or None"
        )
    if frozen_interaction_universe is not None and max_interactions is not None:
        raise ValueError(
            "frozen_interaction_universe cannot be combined with max_interactions"
        )
    if not context_keys:
        raise ValueError("context_keys must contain at least one field")
    missing_context = set(context_keys).difference(aggregate.unit_metadata.columns)
    missing_context.discard("context")
    if missing_context:
        raise ValueError(
            f"aggregate metadata lacks contexts: {sorted(missing_context)}"
        )

    _validate_identifier_columns(
        aggregate.unit_metadata,
        ("sample_id", "subject_id"),
        table_name="aggregate.unit_metadata",
    )

    application_subject_ids = tuple(
        sorted(set(aggregate.unit_metadata["subject_id"].astype(str)))
    )
    if not application_subject_ids:
        raise ValueError("availability requires at least one application subject")
    if frozen_interaction_universe is None:
        candidate_interactions = bundle.interactions
        filter_application = InteractionFilterApplication.TRAINING_SELECTION_V1
    else:
        expected_resource = (
            frozen_interaction_universe.resource_id,
            frozen_interaction_universe.resource_version,
            frozen_interaction_universe.resource_manifest_digest,
        )
        observed_resource = (
            bundle.resource_id,
            bundle.version,
            bundle.manifest_digest,
        )
        if expected_resource != observed_resource:
            raise ValueError(
                "frozen interaction universe resource provenance does not match bundle"
            )
        by_id = {item.interaction_id: item for item in bundle.interactions}
        unknown = set(frozen_interaction_universe.interaction_ids).difference(by_id)
        if unknown:
            raise ValueError(
                "frozen interaction universe contains unknown interaction IDs: "
                f"{sorted(unknown)}"
            )
        candidate_interactions = tuple(
            by_id[interaction_id]
            for interaction_id in frozen_interaction_universe.interaction_ids
        )
        filter_application = InteractionFilterApplication.FROZEN_APPLICATION_V1

    feature_index = {gene: index for index, gene in enumerate(aggregate.feature_ids)}
    mapped: list[Interaction] = []
    dropped_unmapped: list[str] = []
    needed_genes: set[str] = set()
    for interaction in candidate_interactions:
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
    state_eligible = metadata["state_eligible"].to_numpy(dtype=bool)
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
        ) / (n_cells[:, None] + parameters.detection.alpha + parameters.detection.beta)
    else:
        # Without declared zero/detection semantics, detection is neutral.
        shrunk_detection = np.ones_like(selected)
    gene_values = hill * shrunk_detection**parameters.detection.exponent
    gene_values[~state_eligible, :] = np.nan
    log_reference = np.log1p(selected) / math.log1p(
        parameters.absolute_expression_reference
    )
    log_reference = np.clip(log_reference, 0.0, 1.0)
    log_reference[~state_eligible, :] = np.nan

    ligand_columns: list[np.ndarray] = []
    receptor_columns: list[np.ndarray] = []
    ligand_absolute_columns: list[np.ndarray] = []
    receptor_absolute_columns: list[np.ndarray] = []
    supported: list[Interaction] = []
    dropped_no_support: list[str] = []
    entity_values: dict[tuple[str, ...], npt.NDArray[np.float64]] = {}
    absolute_entity_values: dict[tuple[str, ...], npt.NDArray[np.float64]] = {}

    def values_for(subunits: tuple[str, ...]) -> npt.NDArray[np.float64]:
        values = entity_values.get(subunits)
        if values is None:
            values = _entity_values(
                gene_values,
                [local_index[gene] for gene in subunits],
                power=parameters.complex_power,
                epsilon=parameters.complex_epsilon,
            )
            entity_values[subunits] = values
        return values

    def absolute_values_for(subunits: tuple[str, ...]) -> npt.NDArray[np.float64]:
        values = absolute_entity_values.get(subunits)
        if values is None:
            values = _absolute_entity_values(
                log_reference,
                [local_index[gene] for gene in subunits],
                power=parameters.complex_power,
                epsilon=parameters.complex_epsilon,
            )
            absolute_entity_values[subunits] = values
        return values

    for interaction in mapped:
        ligand = values_for(interaction.ligand_subunits)
        receptor = values_for(interaction.receptor_subunits)
        ligand_max = float(np.nanmax(ligand)) if np.isfinite(ligand).any() else 0.0
        receptor_max = (
            float(np.nanmax(receptor)) if np.isfinite(receptor).any() else 0.0
        )
        if frozen_interaction_universe is None and (
            min(ligand_max, receptor_max) < min_pooled_availability
        ):
            dropped_no_support.append(interaction.interaction_id)
            continue
        supported.append(interaction)
        ligand_columns.append(ligand)
        receptor_columns.append(receptor)
        ligand_absolute_columns.append(absolute_values_for(interaction.ligand_subunits))
        receptor_absolute_columns.append(
            absolute_values_for(interaction.receptor_subunits)
        )

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
        ligand_absolute_columns = [ligand_absolute_columns[index] for index in order]
        receptor_absolute_columns = [
            receptor_absolute_columns[index] for index in order
        ]

    if frozen_interaction_universe is None:
        selection_policy = (
            InteractionFilterPolicy.POOLED_SUPPORT_V1
            if max_interactions is None
            else InteractionFilterPolicy.EXPLORATORY_POOLED_TOP_K_V1
        )
        resolved_universe = FrozenInteractionUniverse(
            interaction_ids=tuple(item.interaction_id for item in supported),
            training_subject_ids=application_subject_ids,
            resource_id=bundle.resource_id,
            resource_version=bundle.version,
            resource_manifest_digest=bundle.manifest_digest,
            min_pooled_availability=min_pooled_availability,
            max_interactions=max_interactions,
            selection_policy=selection_policy,
        )
    else:
        resolved_universe = frozen_interaction_universe

    mapping_summary = pd.DataFrame(
        {
            "metric": [
                "source_interactions",
                "filter_candidate_interactions",
                "excluded_by_frozen_universe_interactions",
                "gene_mapped_interactions",
                "pooled_supported_interactions",
                "dropped_unmapped_interactions",
                "dropped_no_pooled_support_interactions",
            ],
            "value": [
                len(bundle.interactions),
                len(candidate_interactions),
                len(bundle.interactions) - len(candidate_interactions)
                if frozen_interaction_universe is not None
                else 0,
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
            frozen_interaction_universe=resolved_universe,
            filter_application=filter_application,
            application_subject_ids=application_subject_ids,
        )

    ligand_matrix = np.column_stack(ligand_columns)
    receptor_matrix = np.column_stack(receptor_columns)
    ligand_absolute_matrix = np.column_stack(ligand_absolute_columns)
    receptor_absolute_matrix = np.column_stack(receptor_absolute_columns)
    state_rows = metadata.index[state_eligible]
    records: list[pd.DataFrame] = []
    interaction_fields = {
        "interaction_id": [item.interaction_id for item in supported],
        "source_interaction_id": [item.source_interaction_id for item in supported],
        "ligand": [item.ligand_name for item in supported],
        "receptor": [item.receptor_name for item in supported],
        "pathway": [item.pathway for item in supported],
    }
    for sample_id, sample_rows in metadata.loc[state_rows].groupby(
        "sample_id", sort=False, observed=True
    ):
        row_indices = sample_rows.index.to_numpy(dtype=int)
        for sender_row in row_indices:
            sender = metadata.loc[sender_row]
            ligand = ligand_matrix[sender_row]
            ligand_absolute = ligand_absolute_matrix[sender_row]
            for receiver_row in row_indices:
                receiver = metadata.loc[receiver_row]
                receptor = receptor_matrix[receiver_row]
                receptor_absolute = receptor_absolute_matrix[receiver_row]
                state = ligand * receptor
                absolute_lr_activity = 0.5 * (ligand_absolute + receptor_absolute)
                sender_proportion = float(sender["cell_proportion"])
                receiver_proportion = float(receiver["cell_proportion"])
                sender_abundance_eligible = bool(sender["abundance_eligible"])
                receiver_abundance_eligible = bool(receiver["abundance_eligible"])
                ecosystem_eligible = (
                    sender_abundance_eligible and receiver_abundance_eligible
                )
                if ecosystem_eligible:
                    ecosystem = state * math.sqrt(
                        sender_proportion * receiver_proportion
                    )
                    ecosystem_status = "observed"
                    ecosystem_reason_code: str | None = None
                else:
                    ecosystem = np.full_like(state, np.nan)
                    ecosystem_status = "abundance_not_estimable"
                    if (
                        not sender_abundance_eligible
                        and not receiver_abundance_eligible
                    ):
                        ecosystem_reason_code = (
                            "sender_and_receiver_abundance_not_eligible"
                        )
                    elif not sender_abundance_eligible:
                        ecosystem_reason_code = "sender_abundance_not_eligible"
                    else:
                        ecosystem_reason_code = "receiver_abundance_not_eligible"
                frame = pd.DataFrame(interaction_fields)
                frame.insert(0, "receiver", receiver["cell_type"])
                frame.insert(0, "sender", sender["cell_type"])
                for key in reversed(context_keys):
                    frame.insert(0, key, [_context_value(sender, key)] * len(frame))
                frame.insert(0, "context_id", _context_id(sender, context_keys))
                frame.insert(0, "subject_id", sender["subject_id"])
                frame.insert(0, "sample_id", sample_id)
                frame["ligand_availability"] = ligand
                frame["receptor_availability"] = receptor
                frame["ligand_absolute_evidence"] = ligand_absolute
                frame["receptor_absolute_evidence"] = receptor_absolute
                frame["absolute_lr_activity"] = absolute_lr_activity
                frame["sender_proportion"] = sender_proportion
                frame["receiver_proportion"] = receiver_proportion
                frame["availability_state"] = state
                frame["availability_ecosystem"] = ecosystem
                frame["state_status"] = "observed"
                frame["state_reason_code"] = None
                frame["ecosystem_status"] = ecosystem_status
                frame["ecosystem_reason_code"] = ecosystem_reason_code
                frame["ecosystem_label"] = "capture_weighted_ecosystem_proxy"
                records.append(frame)
    sample_interactions = (
        pd.concat(records, ignore_index=True)
        if records
        else _empty_sample_table(context_keys)
    )
    for column in (
        "sender",
        "receiver",
        "interaction_id",
        "source_interaction_id",
        "ligand",
        "receptor",
        "pathway",
        "state_status",
        "state_reason_code",
        "ecosystem_status",
        "ecosystem_reason_code",
        "ecosystem_label",
    ):
        sample_interactions[column] = sample_interactions[column].astype("category")
    return BatchAvailability(
        sample_interactions=sample_interactions,
        mapping_summary=mapping_summary,
        resource_id=bundle.resource_id,
        resource_version=bundle.version,
        detection_available=detection_available,
        frozen_interaction_universe=resolved_universe,
        filter_application=filter_application,
        application_subject_ids=application_subject_ids,
    )
