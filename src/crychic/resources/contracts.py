"""Immutable contracts for static molecular communication resources."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from crychic.core import ContractError, stable_id


class Species(StrEnum):
    """Species supported by the first-party resource adapters."""

    HUMAN = "human"
    MOUSE = "mouse"


class GeneNamespace(StrEnum):
    """Canonical gene identifiers emitted by resource adapters."""

    HGNC_SYMBOL = "HGNC symbol"
    MGI_SYMBOL = "MGI symbol"


def _nonempty_tuple(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    cleaned = tuple(value.strip() for value in values if value.strip())
    if not cleaned:
        raise ContractError(
            f"{field} must contain at least one gene",
            code="empty_resource_entity",
            field=field,
            remediation="Preserve and map every required interaction component",
        )
    if len(cleaned) != len(set(cleaned)):
        raise ContractError(
            f"{field} contains duplicate genes",
            code="duplicate_resource_component",
            field=field,
            remediation="Deduplicate source mappings before constructing the contract",
        )
    return tuple(sorted(cleaned))


@dataclass(frozen=True, slots=True)
class Interaction:
    """One directed or symmetric molecular interaction with expanded components."""

    interaction_id: str
    source_interaction_id: str
    ligand_name: str
    receptor_name: str
    ligand_subunits: tuple[str, ...]
    receptor_subunits: tuple[str, ...]
    ligand_is_complex: bool
    receptor_is_complex: bool
    direction: str
    source: str
    version: str
    species: Species
    gene_namespace: GeneNamespace
    pathway: str | None = None
    annotation: str | None = None
    evidence: tuple[str, ...] = ()
    agonist_subunits: tuple[str, ...] = ()
    antagonist_subunits: tuple[str, ...] = ()
    activating_coreceptor_subunits: tuple[str, ...] = ()
    inhibiting_coreceptor_subunits: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.interaction_id or not self.source_interaction_id:
            raise ContractError(
                "Interaction identifiers cannot be empty",
                code="empty_interaction_id",
                field="interaction_id",
                remediation="Use the source ID and canonical interaction ID builder",
            )
        if not self.direction.strip():
            raise ContractError(
                "Interaction direction cannot be empty",
                code="empty_interaction_direction",
                field="direction",
                remediation="Preserve the source direction annotation",
            )
        object.__setattr__(
            self,
            "ligand_subunits",
            _nonempty_tuple(self.ligand_subunits, field="ligand_subunits"),
        )
        object.__setattr__(
            self,
            "receptor_subunits",
            _nonempty_tuple(self.receptor_subunits, field="receptor_subunits"),
        )
        for field_name in (
            "agonist_subunits",
            "antagonist_subunits",
            "activating_coreceptor_subunits",
            "inhibiting_coreceptor_subunits",
        ):
            values = tuple(sorted(set(getattr(self, field_name))))
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "evidence", tuple(sorted(set(self.evidence))))
        object.__setattr__(self, "metadata", tuple(sorted(set(self.metadata))))


def build_interaction_id(
    *,
    resource_id: str,
    version: str,
    species: Species,
    source_interaction_id: str,
    ligand_subunits: tuple[str, ...],
    receptor_subunits: tuple[str, ...],
    direction: str,
) -> str:
    """Return an order-independent stable identifier for one interaction."""

    return stable_id(
        "interaction",
        {
            "direction": direction,
            "ligand": sorted(ligand_subunits),
            "receptor": sorted(receptor_subunits),
            "resource_id": resource_id,
            "source_interaction_id": source_interaction_id,
            "species": species.value,
            "version": version,
        },
    )


@dataclass(frozen=True, slots=True)
class MappingReport:
    """Auditable summary of source-to-canonical identifier mapping."""

    source_rows: int
    loaded_rows: int
    mapped_entities: int
    unmapped_entities: tuple[str, ...] = ()
    ambiguous_entities: tuple[str, ...] = ()
    skipped_source_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.source_rows, self.loaded_rows, self.mapped_entities) < 0:
            raise ContractError(
                "Mapping report counts must be non-negative",
                code="invalid_mapping_report",
                field="mapping_report",
                remediation="Report source and loaded row counts explicitly",
            )
        if self.loaded_rows > self.source_rows:
            raise ContractError(
                "loaded_rows cannot exceed source_rows",
                code="invalid_mapping_report",
                field="loaded_rows",
                remediation="Check adapter row accounting",
            )
        for field_name in (
            "unmapped_entities",
            "ambiguous_entities",
            "skipped_source_ids",
            "notes",
        ):
            object.__setattr__(
                self, field_name, tuple(sorted(set(getattr(self, field_name))))
            )


@dataclass(frozen=True, slots=True)
class ResourceBundle:
    """An immutable, versioned collection of molecular interactions."""

    resource_id: str
    version: str
    species: Species
    gene_namespace: GeneNamespace
    interactions: tuple[Interaction, ...]
    mapping_report: MappingReport
    manifest_digest: str
    source_files: tuple[str, ...]
    license: str
    citation: str

    def __post_init__(self) -> None:
        if not self.resource_id or not self.version:
            raise ContractError(
                "Resource ID and version cannot be empty",
                code="invalid_resource_identity",
                field="resource_id",
                remediation="Provide the upstream resource name and frozen release",
            )
        if not self.license.strip() or not self.citation.strip():
            raise ContractError(
                "Resource license and citation are required",
                code="missing_resource_governance",
                field="license",
                remediation="Add reviewed license and citation metadata",
            )
        ordered = tuple(sorted(self.interactions, key=lambda item: item.interaction_id))
        if len({item.interaction_id for item in ordered}) != len(ordered):
            raise ContractError(
                "Resource interaction IDs must be unique",
                code="duplicate_interaction_id",
                field="interactions",
                remediation="Include the source interaction ID in canonical IDs",
            )
        if any(item.species is not self.species for item in ordered):
            raise ContractError(
                "Every interaction must match the bundle species",
                code="mixed_resource_species",
                field="interactions",
                remediation="Load human and mouse resources separately",
            )
        object.__setattr__(self, "interactions", ordered)
        object.__setattr__(self, "source_files", tuple(sorted(set(self.source_files))))


@dataclass(frozen=True, slots=True)
class PriorLink:
    """One materialized sparse prior link."""

    target: str
    driver: str
    weight: float
    rank: int | None = None


@dataclass(frozen=True, slots=True)
class TargetPrior:
    """Immutable sparse target-by-driver prior in compressed-column form."""

    resource_id: str
    version: str
    species: Species
    gene_namespace: GeneNamespace
    driver_kind: Literal["ligand", "interaction"]
    target_ids: tuple[str, ...]
    driver_ids: tuple[str, ...]
    indptr: tuple[int, ...]
    target_indices: tuple[int, ...]
    weights: tuple[float, ...]
    ranks: tuple[int, ...] | None
    direction: int
    evidence: str
    mapping_report: MappingReport
    manifest_digest: str

    def __post_init__(self) -> None:
        if self.direction not in {-1, 1}:
            raise ContractError(
                "Target prior direction must be -1 or 1",
                code="invalid_prior_direction",
                field="direction",
                remediation="Store unsigned priors as activation direction +1",
            )
        if tuple(sorted(set(self.target_ids))) != self.target_ids:
            raise ContractError(
                "target_ids must be unique and sorted",
                code="invalid_prior_targets",
                field="target_ids",
                remediation="Canonicalize target genes before building the prior",
            )
        if tuple(sorted(set(self.driver_ids))) != self.driver_ids:
            raise ContractError(
                "driver_ids must be unique and sorted",
                code="invalid_prior_drivers",
                field="driver_ids",
                remediation="Canonicalize driver IDs before building the prior",
            )
        if len(self.indptr) != len(self.driver_ids) + 1:
            raise ContractError(
                "indptr length must equal number of drivers plus one",
                code="invalid_sparse_prior",
                field="indptr",
                remediation="Build compressed columns in driver ID order",
            )
        if not self.indptr or self.indptr[0] != 0:
            raise ContractError(
                "indptr must start at zero",
                code="invalid_sparse_prior",
                field="indptr",
                remediation="Build compressed columns in driver ID order",
            )
        if any(
            left > right
            for left, right in zip(self.indptr, self.indptr[1:], strict=False)
        ):
            raise ContractError(
                "indptr must be non-decreasing",
                code="invalid_sparse_prior",
                field="indptr",
                remediation="Sort links by driver before constructing the prior",
            )
        nnz = len(self.target_indices)
        if self.indptr[-1] != nnz or len(self.weights) != nnz:
            raise ContractError(
                "Sparse prior coordinate arrays have inconsistent lengths",
                code="invalid_sparse_prior",
                field="weights",
                remediation="Regenerate the derived sparse prior",
            )
        if self.ranks is not None and len(self.ranks) != nnz:
            raise ContractError(
                "Prior ranks must align with non-zero weights",
                code="invalid_sparse_prior",
                field="ranks",
                remediation="Preserve one rank per derived prior link",
            )
        if any(
            index < 0 or index >= len(self.target_ids)
            for index in self.target_indices
        ):
            raise ContractError(
                "Target index is outside target_ids",
                code="invalid_sparse_prior",
                field="target_indices",
                remediation="Rebuild target indices from the canonical target list",
            )
        if any(not math.isfinite(weight) or weight < 0 for weight in self.weights):
            raise ContractError(
                "Target prior weights must be finite and non-negative",
                code="invalid_prior_weight",
                field="weights",
                remediation="Remove invalid links during resource derivation",
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return target rows by driver columns."""

        return (len(self.target_ids), len(self.driver_ids))

    @property
    def nnz(self) -> int:
        """Return the number of stored prior links."""

        return len(self.weights)

    def links_for_driver(self, driver: str) -> tuple[PriorLink, ...]:
        """Materialize links for one driver without exposing mutable storage."""

        try:
            column = self.driver_ids.index(driver)
        except ValueError as exc:
            raise KeyError(driver) from exc
        start, stop = self.indptr[column : column + 2]
        return tuple(
            PriorLink(
                target=self.target_ids[self.target_indices[position]],
                driver=driver,
                weight=self.weights[position],
                rank=None if self.ranks is None else self.ranks[position],
            )
            for position in range(start, stop)
        )
