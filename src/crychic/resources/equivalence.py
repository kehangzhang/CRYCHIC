"""Frozen, resource-independent molecular ligand-receptor equivalence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Final

from crychic.core import ContractError, stable_id

from .contracts import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
)

_SCHEMA_VERSION: Final = "1"
_CANONICAL_DIRECTION: Final = "ligand_to_receptor"
_COMPONENT_SEMANTICS: Final = (
    "exact_unordered_unique_component_sets_no_stoichiometry_v1"
)
_MECHANISTIC_VARIANT_SEMANTICS: Final = "exact_modifier_component_sets_v1"
_SUPPORTED_SOURCE_DIRECTIONS: Final = frozenset({"Ligand-Receptor"})
_UNSUPPORTED_DIRECTION_REASON: Final = "unsupported_interaction_direction"

MOLECULAR_LR_EQUIVALENCE_POLICY_ID: Final[str] = stable_id(
    "molecular_lr_equivalence_policy",
    {
        "component_semantics": _COMPONENT_SEMANTICS,
        "direction": _CANONICAL_DIRECTION,
        "source_directions": sorted(_SUPPORTED_SOURCE_DIRECTIONS),
    },
    schema_version=_SCHEMA_VERSION,
)

MECHANISTIC_VARIANT_POLICY_ID: Final[str] = stable_id(
    "molecular_lr_mechanistic_variant_policy",
    {
        "core_policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
        "modifier_semantics": _MECHANISTIC_VARIANT_SEMANTICS,
    },
    schema_version=_SCHEMA_VERSION,
)


def _canonical_name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _canonical_components(
    values: Sequence[str], *, field_name: str, allow_empty: bool
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    canonical = tuple(
        sorted(_canonical_name(value, field_name=field_name) for value in values)
    )
    if not allow_empty and not canonical:
        raise ValueError(f"{field_name} must contain at least one component")
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{field_name} must contain unique components")
    return canonical


def _mapping_report_payload(report: MappingReport) -> dict[str, object]:
    return {
        "source_rows": report.source_rows,
        "loaded_rows": report.loaded_rows,
        "mapped_entities": report.mapped_entities,
        "unmapped_entities": list(report.unmapped_entities),
        "ambiguous_entities": list(report.ambiguous_entities),
        "skipped_source_ids": list(report.skipped_source_ids),
        "notes": list(report.notes),
    }


def _require_mapping_report_intact(report: MappingReport) -> None:
    try:
        repeated = MappingReport(
            source_rows=report.source_rows,
            loaded_rows=report.loaded_rows,
            mapped_entities=report.mapped_entities,
            unmapped_entities=report.unmapped_entities,
            ambiguous_entities=report.ambiguous_entities,
            skipped_source_ids=report.skipped_source_ids,
            notes=report.notes,
        )
        valid = report == repeated
    except (AttributeError, TypeError, ValueError, ContractError) as error:
        raise ContractError(
            "Resource mapping report failed molecular-equivalence integrity validation",
            code="molecular_lr_source_mapping_report_integrity_violation",
            field="mapping_report",
            remediation="Reload the immutable ResourceBundle from its frozen manifest",
        ) from error
    if not valid:
        raise ContractError(
            "Resource mapping report failed molecular-equivalence integrity validation",
            code="molecular_lr_source_mapping_report_integrity_violation",
            field="mapping_report",
            remediation="Reload the immutable ResourceBundle from its frozen manifest",
        )


@dataclass(frozen=True, slots=True)
class MolecularLREquivalenceClass:
    """One exact directed ligand-complex to receptor-complex molecular event."""

    species: Species
    gene_namespace: GeneNamespace
    ligand_subunits: tuple[str, ...]
    receptor_subunits: tuple[str, ...]
    molecular_lr_equivalence_id: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            species = Species(self.species)
            namespace = GeneNamespace(self.gene_namespace)
        except ValueError as error:
            raise ValueError("species or gene_namespace is unsupported") from error
        ligand = _canonical_components(
            self.ligand_subunits,
            field_name="ligand_subunits",
            allow_empty=False,
        )
        receptor = _canonical_components(
            self.receptor_subunits,
            field_name="receptor_subunits",
            allow_empty=False,
        )
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "gene_namespace", namespace)
        object.__setattr__(self, "ligand_subunits", ligand)
        object.__setattr__(self, "receptor_subunits", receptor)
        object.__setattr__(
            self,
            "molecular_lr_equivalence_id",
            stable_id(
                "molecular_lr_equivalence",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    @property
    def direction(self) -> str:
        """Return the only directed semantic supported by policy v1."""

        return _CANONICAL_DIRECTION

    @property
    def policy_id(self) -> str:
        """Return the exact molecular-equivalence policy identity."""

        return MOLECULAR_LR_EQUIVALENCE_POLICY_ID

    def _identity_payload(self) -> dict[str, object]:
        return {
            "component_semantics": _COMPONENT_SEMANTICS,
            "direction": _CANONICAL_DIRECTION,
            "gene_namespace": self.gene_namespace.value,
            "ligand_subunits": list(self.ligand_subunits),
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "receptor_subunits": list(self.receptor_subunits),
            "species": self.species.value,
        }

    def _require_intact(self) -> None:
        try:
            repeated = MolecularLREquivalenceClass(
                species=self.species,
                gene_namespace=self.gene_namespace,
                ligand_subunits=self.ligand_subunits,
                receptor_subunits=self.receptor_subunits,
            )
            valid = self == repeated
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Molecular LR equivalence class failed integrity validation",
                code="molecular_lr_equivalence_integrity_violation",
                field="molecular_lr_equivalence_id",
                remediation="Rebuild the class from canonical resource components",
            ) from error
        if not valid:
            raise ContractError(
                "Molecular LR equivalence class failed integrity validation",
                code="molecular_lr_equivalence_integrity_violation",
                field="molecular_lr_equivalence_id",
                remediation="Rebuild the class from canonical resource components",
            )

    def to_dict(self) -> dict[str, object]:
        """Return an integrity-checked portable molecular class."""

        self._require_intact()
        return {
            "molecular_lr_equivalence_id": self.molecular_lr_equivalence_id,
            **self._identity_payload(),
        }


@dataclass(frozen=True, slots=True)
class MolecularLRMechanisticVariant:
    """One core molecular LR event with exact optional modifier sets."""

    molecular_lr_equivalence_id: str
    agonist_subunits: tuple[str, ...] = ()
    antagonist_subunits: tuple[str, ...] = ()
    activating_coreceptor_subunits: tuple[str, ...] = ()
    inhibiting_coreceptor_subunits: tuple[str, ...] = ()
    mechanistic_variant_id: str = field(init=False)

    def __post_init__(self) -> None:
        core_id = _canonical_name(
            self.molecular_lr_equivalence_id,
            field_name="molecular_lr_equivalence_id",
        )
        object.__setattr__(self, "molecular_lr_equivalence_id", core_id)
        for field_name in (
            "agonist_subunits",
            "antagonist_subunits",
            "activating_coreceptor_subunits",
            "inhibiting_coreceptor_subunits",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_components(
                    getattr(self, field_name),
                    field_name=field_name,
                    allow_empty=True,
                ),
            )
        object.__setattr__(
            self,
            "mechanistic_variant_id",
            stable_id(
                "molecular_lr_mechanistic_variant",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    @property
    def policy_id(self) -> str:
        """Return the exact modifier-set policy identity."""

        return MECHANISTIC_VARIANT_POLICY_ID

    def _identity_payload(self) -> dict[str, object]:
        return {
            "activating_coreceptor_subunits": list(self.activating_coreceptor_subunits),
            "agonist_subunits": list(self.agonist_subunits),
            "antagonist_subunits": list(self.antagonist_subunits),
            "inhibiting_coreceptor_subunits": list(self.inhibiting_coreceptor_subunits),
            "mechanistic_variant_semantics": _MECHANISTIC_VARIANT_SEMANTICS,
            "molecular_lr_equivalence_id": self.molecular_lr_equivalence_id,
            "policy_id": MECHANISTIC_VARIANT_POLICY_ID,
        }

    def _require_intact(self) -> None:
        try:
            repeated = MolecularLRMechanisticVariant(
                molecular_lr_equivalence_id=self.molecular_lr_equivalence_id,
                agonist_subunits=self.agonist_subunits,
                antagonist_subunits=self.antagonist_subunits,
                activating_coreceptor_subunits=(self.activating_coreceptor_subunits),
                inhibiting_coreceptor_subunits=self.inhibiting_coreceptor_subunits,
            )
            valid = self == repeated
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Molecular LR mechanistic variant failed integrity validation",
                code="molecular_lr_mechanistic_variant_integrity_violation",
                field="mechanistic_variant_id",
                remediation="Rebuild the variant from the canonical interaction",
            ) from error
        if not valid:
            raise ContractError(
                "Molecular LR mechanistic variant failed integrity validation",
                code="molecular_lr_mechanistic_variant_integrity_violation",
                field="mechanistic_variant_id",
                remediation="Rebuild the variant from the canonical interaction",
            )

    def to_dict(self) -> dict[str, object]:
        """Return an integrity-checked portable mechanistic variant."""

        self._require_intact()
        return {
            "mechanistic_variant_id": self.mechanistic_variant_id,
            **self._identity_payload(),
        }


class MolecularLRMappingStatus(StrEnum):
    """Outcome of mapping one source interaction to policy v1."""

    MAPPED = "mapped"
    UNSUPPORTED_DIRECTION = "unsupported_direction"


@dataclass(frozen=True, slots=True)
class MolecularLRMappingRecord:
    """Auditable mapping of one source interaction without direction guessing."""

    resource_bundle_content_id: str
    interaction_id: str
    source_interaction_id: str
    source_direction: str
    status: MolecularLRMappingStatus | str
    molecular_lr_equivalence_id: str | None
    mechanistic_variant_id: str | None
    reason_code: str | None
    mapping_record_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "resource_bundle_content_id",
            "interaction_id",
            "source_interaction_id",
            "source_direction",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_name(getattr(self, field_name), field_name=field_name),
            )
        status = MolecularLRMappingStatus(self.status)
        object.__setattr__(self, "status", status)
        if status is MolecularLRMappingStatus.MAPPED:
            if (
                self.molecular_lr_equivalence_id is None
                or self.mechanistic_variant_id is None
                or self.reason_code is not None
            ):
                raise ValueError(
                    "mapped records require core and variant IDs and no reason_code"
                )
            _canonical_name(
                self.molecular_lr_equivalence_id,
                field_name="molecular_lr_equivalence_id",
            )
            _canonical_name(
                self.mechanistic_variant_id,
                field_name="mechanistic_variant_id",
            )
        elif (
            self.molecular_lr_equivalence_id is not None
            or self.mechanistic_variant_id is not None
            or self.reason_code != _UNSUPPORTED_DIRECTION_REASON
        ):
            raise ValueError(
                "unsupported-direction records require only the released reason_code"
            )
        object.__setattr__(
            self,
            "mapping_record_id",
            stable_id(
                "molecular_lr_mapping_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "interaction_id": self.interaction_id,
            "mechanistic_variant_id": self.mechanistic_variant_id,
            "molecular_lr_equivalence_id": self.molecular_lr_equivalence_id,
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "reason_code": self.reason_code,
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "source_direction": self.source_direction,
            "source_interaction_id": self.source_interaction_id,
            "status": MolecularLRMappingStatus(self.status).value,
        }

    def _require_intact(self) -> None:
        try:
            repeated = MolecularLRMappingRecord(
                resource_bundle_content_id=self.resource_bundle_content_id,
                interaction_id=self.interaction_id,
                source_interaction_id=self.source_interaction_id,
                source_direction=self.source_direction,
                status=self.status,
                molecular_lr_equivalence_id=self.molecular_lr_equivalence_id,
                mechanistic_variant_id=self.mechanistic_variant_id,
                reason_code=self.reason_code,
            )
            valid = self == repeated
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Molecular LR mapping record failed integrity validation",
                code="molecular_lr_mapping_record_integrity_violation",
                field="mapping_record_id",
                remediation="Refreeze mappings from the complete ResourceBundle",
            ) from error
        if not valid:
            raise ContractError(
                "Molecular LR mapping record failed integrity validation",
                code="molecular_lr_mapping_record_integrity_violation",
                field="mapping_record_id",
                remediation="Refreeze mappings from the complete ResourceBundle",
            )

    def to_dict(self) -> dict[str, object]:
        """Return an integrity-checked portable mapping record."""

        self._require_intact()
        return {"mapping_record_id": self.mapping_record_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True)
class MolecularLRSourceBundleBinding:
    """Frozen source-bundle identity and its original adapter mapping report."""

    resource_bundle_content_id: str
    resource_id: str
    version: str
    manifest_digest: str
    species: Species
    gene_namespace: GeneNamespace
    interaction_ids: tuple[str, ...]
    mapping_report: MappingReport
    source_binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "resource_bundle_content_id",
            "resource_id",
            "version",
            "manifest_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_name(getattr(self, field_name), field_name=field_name),
            )
        species = Species(self.species)
        namespace = GeneNamespace(self.gene_namespace)
        interactions = _canonical_components(
            self.interaction_ids,
            field_name="interaction_ids",
            allow_empty=True,
        )
        if not isinstance(self.mapping_report, MappingReport):
            raise TypeError("mapping_report must be a MappingReport")
        _require_mapping_report_intact(self.mapping_report)
        if self.mapping_report.loaded_rows != len(interactions):
            raise ValueError(
                "mapping_report.loaded_rows must equal the frozen interaction count"
            )
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "gene_namespace", namespace)
        object.__setattr__(self, "interaction_ids", interactions)
        object.__setattr__(
            self,
            "source_binding_id",
            stable_id(
                "molecular_lr_source_bundle_binding",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "gene_namespace": self.gene_namespace.value,
            "interaction_ids": list(self.interaction_ids),
            "manifest_digest": self.manifest_digest,
            "mapping_report": _mapping_report_payload(self.mapping_report),
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "resource_id": self.resource_id,
            "species": self.species.value,
            "version": self.version,
        }

    def _require_intact(self) -> None:
        try:
            _require_mapping_report_intact(self.mapping_report)
            repeated = MolecularLRSourceBundleBinding(
                resource_bundle_content_id=self.resource_bundle_content_id,
                resource_id=self.resource_id,
                version=self.version,
                manifest_digest=self.manifest_digest,
                species=self.species,
                gene_namespace=self.gene_namespace,
                interaction_ids=self.interaction_ids,
                mapping_report=self.mapping_report,
            )
            valid = self == repeated
        except (AttributeError, TypeError, ValueError, ContractError) as error:
            raise ContractError(
                "Molecular LR source binding failed integrity validation",
                code="molecular_lr_source_binding_integrity_violation",
                field="source_binding_id",
                remediation="Refreeze the universe from intact ResourceBundles",
            ) from error
        if not valid:
            raise ContractError(
                "Molecular LR source binding failed integrity validation",
                code="molecular_lr_source_binding_integrity_violation",
                field="source_binding_id",
                remediation="Refreeze the universe from intact ResourceBundles",
            )

    def to_dict(self) -> dict[str, object]:
        """Return an integrity-checked source binding."""

        self._require_intact()
        return {"source_binding_id": self.source_binding_id, **self._identity_payload()}


def _molecular_lr_axis_id(
    *,
    species: Species,
    gene_namespace: GeneNamespace,
    equivalence_ids: Sequence[str],
) -> str:
    identifier: str = stable_id(
        "molecular_lr_equivalence_axis",
        {
            "gene_namespace": gene_namespace.value,
            "molecular_lr_equivalence_ids": list(equivalence_ids),
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "species": species.value,
        },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


def _mechanistic_variant_axis_id(
    *, molecular_lr_axis_id: str, variant_ids: Sequence[str]
) -> str:
    identifier: str = stable_id(
        "molecular_lr_mechanistic_variant_axis",
        {
            "mechanistic_variant_ids": list(variant_ids),
            "molecular_lr_axis_id": molecular_lr_axis_id,
            "policy_id": MECHANISTIC_VARIANT_POLICY_ID,
        },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


def _mapping_axis_id(
    *,
    source_binding_ids: Sequence[str],
    mapping_record_ids: Sequence[str],
    molecular_lr_axis_id: str,
    mechanistic_variant_axis_id: str,
) -> str:
    identifier: str = stable_id(
        "molecular_lr_mapping_axis",
        {
            "mapping_record_ids": list(mapping_record_ids),
            "mechanistic_variant_axis_id": mechanistic_variant_axis_id,
            "molecular_lr_axis_id": molecular_lr_axis_id,
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "source_binding_ids": list(source_binding_ids),
        },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


@dataclass(frozen=True, slots=True, init=False)
class FrozenMolecularLREquivalenceUniverse:
    """Complete source-record crosswalk onto deduplicated molecular LR classes."""

    species: Species
    gene_namespace: GeneNamespace
    source_bindings: tuple[MolecularLRSourceBundleBinding, ...]
    equivalence_classes: tuple[MolecularLREquivalenceClass, ...]
    mechanistic_variants: tuple[MolecularLRMechanisticVariant, ...]
    mapping_records: tuple[MolecularLRMappingRecord, ...]
    molecular_lr_equivalence_ids: tuple[str, ...]
    molecular_lr_axis_id: str
    mechanistic_variant_ids: tuple[str, ...]
    mechanistic_variant_axis_id: str
    mapping_record_ids: tuple[str, ...]
    mapping_axis_id: str
    universe_id: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenMolecularLREquivalenceUniverse is producer-owned; use "
            "freeze_molecular_lr_equivalence_universe()"
        )

    @property
    def policy_id(self) -> str:
        """Return the exact molecular-equivalence policy identity."""

        return MOLECULAR_LR_EQUIVALENCE_POLICY_ID

    def _identity_payload(self) -> dict[str, object]:
        return {
            "gene_namespace": self.gene_namespace.value,
            "mapping_axis_id": self.mapping_axis_id,
            "mechanistic_variant_axis_id": self.mechanistic_variant_axis_id,
            "molecular_lr_axis_id": self.molecular_lr_axis_id,
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "source_binding_ids": [
                binding.source_binding_id for binding in self.source_bindings
            ],
            "species": self.species.value,
        }

    def _require_intact(self) -> None:
        try:
            for binding in self.source_bindings:
                binding._require_intact()
            for equivalence in self.equivalence_classes:
                equivalence._require_intact()
            for variant in self.mechanistic_variants:
                variant._require_intact()
            for mapping in self.mapping_records:
                mapping._require_intact()

            binding_ids = tuple(
                binding.source_binding_id for binding in self.source_bindings
            )
            bundle_content_ids = tuple(
                binding.resource_bundle_content_id for binding in self.source_bindings
            )
            equivalence_ids = tuple(
                item.molecular_lr_equivalence_id for item in self.equivalence_classes
            )
            variant_ids = tuple(
                item.mechanistic_variant_id for item in self.mechanistic_variants
            )
            mapping_ids = tuple(item.mapping_record_id for item in self.mapping_records)
            expected_pairs = {
                (binding.resource_bundle_content_id, interaction_id)
                for binding in self.source_bindings
                for interaction_id in binding.interaction_ids
            }
            actual_pairs = {
                (record.resource_bundle_content_id, record.interaction_id)
                for record in self.mapping_records
            }
            mapped_core_ids = {
                record.molecular_lr_equivalence_id
                for record in self.mapping_records
                if record.status is MolecularLRMappingStatus.MAPPED
            }
            mapped_variant_ids = {
                record.mechanistic_variant_id
                for record in self.mapping_records
                if record.status is MolecularLRMappingStatus.MAPPED
            }
            molecular_axis = _molecular_lr_axis_id(
                species=self.species,
                gene_namespace=self.gene_namespace,
                equivalence_ids=equivalence_ids,
            )
            variant_axis = _mechanistic_variant_axis_id(
                molecular_lr_axis_id=molecular_axis,
                variant_ids=variant_ids,
            )
            mapping_axis = _mapping_axis_id(
                source_binding_ids=binding_ids,
                mapping_record_ids=mapping_ids,
                molecular_lr_axis_id=molecular_axis,
                mechanistic_variant_axis_id=variant_axis,
            )
            universe_id = stable_id(
                "frozen_molecular_lr_equivalence_universe",
                {
                    "gene_namespace": self.gene_namespace.value,
                    "mapping_axis_id": mapping_axis,
                    "mechanistic_variant_axis_id": variant_axis,
                    "molecular_lr_axis_id": molecular_axis,
                    "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
                    "source_binding_ids": list(binding_ids),
                    "species": self.species.value,
                },
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                bool(self.source_bindings)
                and binding_ids == tuple(sorted(binding_ids))
                and len(binding_ids) == len(set(binding_ids))
                and len(bundle_content_ids) == len(set(bundle_content_ids))
                and equivalence_ids == tuple(sorted(equivalence_ids))
                and len(equivalence_ids) == len(set(equivalence_ids))
                and variant_ids == tuple(sorted(variant_ids))
                and len(variant_ids) == len(set(variant_ids))
                and mapping_ids == tuple(sorted(mapping_ids))
                and len(mapping_ids) == len(set(mapping_ids))
                and len(actual_pairs) == len(self.mapping_records)
                and actual_pairs == expected_pairs
                and mapped_core_ids == set(equivalence_ids)
                and mapped_variant_ids == set(variant_ids)
                and all(
                    binding.species is self.species
                    and binding.gene_namespace is self.gene_namespace
                    for binding in self.source_bindings
                )
                and all(
                    item.species is self.species
                    and item.gene_namespace is self.gene_namespace
                    for item in self.equivalence_classes
                )
                and all(
                    item.molecular_lr_equivalence_id in set(equivalence_ids)
                    for item in self.mechanistic_variants
                )
                and self.molecular_lr_equivalence_ids == equivalence_ids
                and self.molecular_lr_axis_id == molecular_axis
                and self.mechanistic_variant_ids == variant_ids
                and self.mechanistic_variant_axis_id == variant_axis
                and self.mapping_record_ids == mapping_ids
                and self.mapping_axis_id == mapping_axis
                and self.universe_id == universe_id
            )
        except (AttributeError, TypeError, ValueError, ContractError) as error:
            raise ContractError(
                "Frozen molecular LR equivalence universe failed integrity validation",
                code="frozen_molecular_lr_equivalence_universe_integrity_violation",
                field="universe_id",
                remediation=(
                    "Refreeze the universe from complete intact ResourceBundles"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Frozen molecular LR equivalence universe failed integrity validation",
                code="frozen_molecular_lr_equivalence_universe_integrity_violation",
                field="universe_id",
                remediation=(
                    "Refreeze the universe from complete intact ResourceBundles"
                ),
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete integrity-checked frozen universe."""

        self._require_intact()
        return {
            "universe_id": self.universe_id,
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "species": self.species.value,
            "gene_namespace": self.gene_namespace.value,
            "direction": _CANONICAL_DIRECTION,
            "component_semantics": _COMPONENT_SEMANTICS,
            "molecular_lr_axis_id": self.molecular_lr_axis_id,
            "molecular_lr_equivalence_ids": list(self.molecular_lr_equivalence_ids),
            "mechanistic_variant_axis_id": self.mechanistic_variant_axis_id,
            "mechanistic_variant_ids": list(self.mechanistic_variant_ids),
            "mapping_axis_id": self.mapping_axis_id,
            "mapping_record_ids": list(self.mapping_record_ids),
            "source_bindings": [item.to_dict() for item in self.source_bindings],
            "equivalence_classes": [
                item.to_dict() for item in self.equivalence_classes
            ],
            "mechanistic_variants": [
                item.to_dict() for item in self.mechanistic_variants
            ],
            "mapping_records": [item.to_dict() for item in self.mapping_records],
            "mapped_count": sum(
                item.status is MolecularLRMappingStatus.MAPPED
                for item in self.mapping_records
            ),
            "unsupported_direction_count": sum(
                item.status is MolecularLRMappingStatus.UNSUPPORTED_DIRECTION
                for item in self.mapping_records
            ),
        }


def _resource_bundle_content_id(bundle: ResourceBundle) -> str:
    identifier: str = stable_id("resource_bundle_content", asdict(bundle))
    return identifier


def _source_binding(bundle: ResourceBundle) -> MolecularLRSourceBundleBinding:
    return MolecularLRSourceBundleBinding(
        resource_bundle_content_id=_resource_bundle_content_id(bundle),
        resource_id=bundle.resource_id,
        version=bundle.version,
        manifest_digest=bundle.manifest_digest,
        species=bundle.species,
        gene_namespace=bundle.gene_namespace,
        interaction_ids=tuple(item.interaction_id for item in bundle.interactions),
        mapping_report=bundle.mapping_report,
    )


def _equivalence_class(interaction: Interaction) -> MolecularLREquivalenceClass:
    return MolecularLREquivalenceClass(
        species=interaction.species,
        gene_namespace=interaction.gene_namespace,
        ligand_subunits=interaction.ligand_subunits,
        receptor_subunits=interaction.receptor_subunits,
    )


def _mechanistic_variant(
    interaction: Interaction,
    *,
    molecular_lr_equivalence_id: str,
) -> MolecularLRMechanisticVariant:
    return MolecularLRMechanisticVariant(
        molecular_lr_equivalence_id=molecular_lr_equivalence_id,
        agonist_subunits=interaction.agonist_subunits,
        antagonist_subunits=interaction.antagonist_subunits,
        activating_coreceptor_subunits=(interaction.activating_coreceptor_subunits),
        inhibiting_coreceptor_subunits=interaction.inhibiting_coreceptor_subunits,
    )


def freeze_molecular_lr_equivalence_universe(
    resource_bundles: ResourceBundle | Sequence[ResourceBundle],
) -> FrozenMolecularLREquivalenceUniverse:
    """Freeze complete source mappings onto exact molecular LR classes.

    Only the exact reviewed ``Ligand-Receptor`` source direction is mapped by
    policy v1. Other direction labels remain explicit unsupported records.
    """

    bundles: tuple[ResourceBundle, ...]
    if isinstance(resource_bundles, ResourceBundle):
        bundles = (resource_bundles,)
    else:
        raw_bundles: object = resource_bundles
        if isinstance(raw_bundles, (str, bytes, bytearray)):
            raise TypeError("resource_bundles must contain ResourceBundle objects")
        bundles = tuple(resource_bundles)
    if not bundles:
        raise ValueError("resource_bundles must contain at least one bundle")
    if any(not isinstance(bundle, ResourceBundle) for bundle in bundles):
        raise TypeError("resource_bundles must contain only ResourceBundle objects")

    species = bundles[0].species
    namespace = bundles[0].gene_namespace
    if any(
        bundle.species is not species or bundle.gene_namespace is not namespace
        for bundle in bundles
    ):
        raise ContractError(
            "Molecular LR universe bundles must share species and gene namespace",
            code="mixed_molecular_lr_universe_identity",
            field="resource_bundles",
            remediation="Freeze each species and canonical gene namespace separately",
        )
    for bundle in bundles:
        if any(
            interaction.species is not bundle.species
            or interaction.gene_namespace is not bundle.gene_namespace
            for interaction in bundle.interactions
        ):
            raise ContractError(
                "Resource interaction species or namespace differs from its bundle",
                code="mixed_molecular_lr_bundle_identity",
                field="resource_bundle.interactions",
                remediation="Correct the adapter mapping before freezing equivalence",
            )

    source_bindings = tuple(
        sorted(
            (_source_binding(bundle) for bundle in bundles),
            key=lambda item: item.source_binding_id,
        )
    )
    if len({item.resource_bundle_content_id for item in source_bindings}) != len(
        source_bindings
    ):
        raise ValueError("resource_bundles contains duplicate bundle content")

    classes: dict[str, MolecularLREquivalenceClass] = {}
    variants: dict[str, MolecularLRMechanisticVariant] = {}
    mappings: list[MolecularLRMappingRecord] = []
    binding_by_content_id = {
        binding.resource_bundle_content_id: binding for binding in source_bindings
    }
    bundles_by_content_id = {
        _resource_bundle_content_id(bundle): bundle for bundle in bundles
    }
    for content_id in sorted(bundles_by_content_id):
        bundle = bundles_by_content_id[content_id]
        binding = binding_by_content_id[content_id]
        for interaction in bundle.interactions:
            if interaction.direction not in _SUPPORTED_SOURCE_DIRECTIONS:
                mappings.append(
                    MolecularLRMappingRecord(
                        resource_bundle_content_id=content_id,
                        interaction_id=interaction.interaction_id,
                        source_interaction_id=interaction.source_interaction_id,
                        source_direction=interaction.direction,
                        status=MolecularLRMappingStatus.UNSUPPORTED_DIRECTION,
                        molecular_lr_equivalence_id=None,
                        mechanistic_variant_id=None,
                        reason_code=_UNSUPPORTED_DIRECTION_REASON,
                    )
                )
                continue
            equivalence = _equivalence_class(interaction)
            variant = _mechanistic_variant(
                interaction,
                molecular_lr_equivalence_id=(equivalence.molecular_lr_equivalence_id),
            )
            classes[equivalence.molecular_lr_equivalence_id] = equivalence
            variants[variant.mechanistic_variant_id] = variant
            mappings.append(
                MolecularLRMappingRecord(
                    resource_bundle_content_id=binding.resource_bundle_content_id,
                    interaction_id=interaction.interaction_id,
                    source_interaction_id=interaction.source_interaction_id,
                    source_direction=interaction.direction,
                    status=MolecularLRMappingStatus.MAPPED,
                    molecular_lr_equivalence_id=(
                        equivalence.molecular_lr_equivalence_id
                    ),
                    mechanistic_variant_id=variant.mechanistic_variant_id,
                    reason_code=None,
                )
            )

    equivalence_classes = tuple(classes[key] for key in sorted(classes))
    mechanistic_variants = tuple(variants[key] for key in sorted(variants))
    mapping_records = tuple(sorted(mappings, key=lambda item: item.mapping_record_id))
    equivalence_ids = tuple(classes)
    equivalence_ids = tuple(sorted(equivalence_ids))
    variant_ids = tuple(sorted(variants))
    mapping_ids = tuple(item.mapping_record_id for item in mapping_records)
    molecular_axis_id = _molecular_lr_axis_id(
        species=species,
        gene_namespace=namespace,
        equivalence_ids=equivalence_ids,
    )
    variant_axis_id = _mechanistic_variant_axis_id(
        molecular_lr_axis_id=molecular_axis_id,
        variant_ids=variant_ids,
    )
    mapping_axis_id = _mapping_axis_id(
        source_binding_ids=tuple(item.source_binding_id for item in source_bindings),
        mapping_record_ids=mapping_ids,
        molecular_lr_axis_id=molecular_axis_id,
        mechanistic_variant_axis_id=variant_axis_id,
    )
    universe = object.__new__(FrozenMolecularLREquivalenceUniverse)
    object.__setattr__(universe, "species", species)
    object.__setattr__(universe, "gene_namespace", namespace)
    object.__setattr__(universe, "source_bindings", source_bindings)
    object.__setattr__(universe, "equivalence_classes", equivalence_classes)
    object.__setattr__(universe, "mechanistic_variants", mechanistic_variants)
    object.__setattr__(universe, "mapping_records", mapping_records)
    object.__setattr__(universe, "molecular_lr_equivalence_ids", equivalence_ids)
    object.__setattr__(universe, "molecular_lr_axis_id", molecular_axis_id)
    object.__setattr__(universe, "mechanistic_variant_ids", variant_ids)
    object.__setattr__(universe, "mechanistic_variant_axis_id", variant_axis_id)
    object.__setattr__(universe, "mapping_record_ids", mapping_ids)
    object.__setattr__(universe, "mapping_axis_id", mapping_axis_id)
    object.__setattr__(
        universe,
        "universe_id",
        stable_id(
            "frozen_molecular_lr_equivalence_universe",
            universe._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    universe._require_intact()
    return universe


__all__ = [
    "MECHANISTIC_VARIANT_POLICY_ID",
    "MOLECULAR_LR_EQUIVALENCE_POLICY_ID",
    "FrozenMolecularLREquivalenceUniverse",
    "MolecularLREquivalenceClass",
    "MolecularLRMappingRecord",
    "MolecularLRMappingStatus",
    "MolecularLRMechanisticVariant",
    "MolecularLRSourceBundleBinding",
    "freeze_molecular_lr_equivalence_universe",
]
