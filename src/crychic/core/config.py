"""Serializable Phase 0 configuration contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from .enums import CommunicationMode
from .errors import ConfigurationError
from .serialization import canonical_digest, canonical_json
from .version import CONFIG_SCHEMA_VERSION

_MAX_SEED = 2**63 - 1
_NORMALIZED_TRANSFORMS = frozenset({"linear_normalized", "log1p_normalized"})


def _normalise_names(
    value: Sequence[str],
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigurationError(
            f"{field_name} must be a sequence of field names, not a string",
            code="invalid_field_sequence",
            field=field_name,
            remediation="Provide a list or tuple of names",
        )
    names = tuple(value)
    if not allow_empty and not names:
        raise ConfigurationError(
            f"{field_name} must contain at least one field name",
            code="missing_required_field_sequence",
            field=field_name,
            remediation="Declare at least one biological context field",
        )
    for name in names:
        _validate_name(name, field_name=field_name)
    if len(set(names)) != len(names):
        raise ConfigurationError(
            f"{field_name} contains duplicate field names",
            code="duplicate_field_name",
            field=field_name,
            remediation="Remove duplicate metadata field names",
        )
    return names


def _validate_name(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigurationError(
            f"{field_name} must contain non-empty names without outer whitespace",
            code="invalid_field_name",
            field=field_name,
            remediation="Use the exact AnnData field name",
        )


def _normalise_transform(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in _NORMALIZED_TRANSFORMS:
        allowed = ", ".join(sorted(_NORMALIZED_TRANSFORMS))
        raise ConfigurationError(
            "expression_transform is not supported",
            code="unsupported_expression_transform",
            field="expression_transform",
            remediation=f"Use one of: {allowed}",
        )
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class CrychicConfig:
    """Top-level, immutable configuration for a CRYCHIC analysis."""

    context_keys: Sequence[str]
    counts_layer: str | None = "counts"
    sample_key: str = "sample_id"
    subject_key: str = "subject_id"
    cell_type_key: str = "cell_type"
    covariates: Sequence[str] = ()
    species: str | None = "human"
    gene_namespace: str | None = "hgnc_symbol"
    expression_layer: str | None = None
    expression_source: str | None = None
    expression_transform: str | None = None
    normalized_zero_is_nondetection: bool = False
    allow_duplicate_genes: bool = False
    design: str | None = None
    communication_modes: Sequence[CommunicationMode | str] = (
        CommunicationMode.STATE,
        CommunicationMode.ECOSYSTEM,
    )
    random_seed: int = 20260712

    def __post_init__(self) -> None:
        context_keys = _normalise_names(
            self.context_keys,
            field_name="context_keys",
            allow_empty=False,
        )
        covariates = _normalise_names(
            self.covariates,
            field_name="covariates",
            allow_empty=True,
        )
        for field_name, name in (
            ("sample_key", self.sample_key),
            ("subject_key", self.subject_key),
            ("cell_type_key", self.cell_type_key),
        ):
            _validate_name(name, field_name=field_name)
        for field_name, optional_name in (
            ("counts_layer", self.counts_layer),
            ("species", self.species),
            ("gene_namespace", self.gene_namespace),
            ("expression_layer", self.expression_layer),
            ("expression_source", self.expression_source),
        ):
            if optional_name is not None:
                _validate_name(optional_name, field_name=field_name)

        transform = _normalise_transform(self.expression_transform)
        if (self.expression_source is None) != (transform is None):
            raise ConfigurationError(
                "expression_source and expression_transform must be declared together",
                code="incomplete_expression_declaration",
                field="expression_source",
                remediation="Declare both normalized expression source and transform",
            )
        if self.counts_layer is None and self.expression_source is None:
            raise ConfigurationError(
                "Input must declare either a counts layer or normalized expression",
                code="missing_expression_source",
                field="counts_layer",
                remediation=(
                    "Set counts_layer or provide normalized expression metadata"
                ),
            )

        identifiers = (self.sample_key, self.subject_key, self.cell_type_key)
        if len(set(identifiers)) != len(identifiers):
            raise ConfigurationError(
                "sample, subject, and cell-type keys must be distinct",
                code="conflicting_input_keys",
                field="sample_key",
                remediation="Map each semantic role to a distinct AnnData field",
            )
        reserved = set(identifiers)
        overlap = reserved.intersection(context_keys)
        overlap.update(reserved.intersection(covariates))
        overlap.update(set(context_keys).intersection(covariates))
        if overlap:
            raise ConfigurationError(
                "Identifier, context, and covariate roles must use distinct fields",
                code="conflicting_input_keys",
                field="context_keys",
                remediation="Remove fields assigned to more than one semantic role",
            )

        if self.design is not None and (
            not isinstance(self.design, str)
            or not self.design
            or self.design != self.design.strip()
        ):
            raise ConfigurationError(
                "design must be a non-empty formula without outer whitespace",
                code="invalid_design_formula",
                field="design",
                remediation="Provide a formula such as ~ batch + treatment",
            )
        if isinstance(self.communication_modes, str) or not isinstance(
            self.communication_modes, Sequence
        ):
            raise ConfigurationError(
                "communication_modes must be a sequence, not a string",
                code="invalid_communication_modes",
                field="communication_modes",
                remediation="Provide a list containing state and/or ecosystem",
            )
        try:
            modes = tuple(CommunicationMode(mode) for mode in self.communication_modes)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in CommunicationMode)
            raise ConfigurationError(
                "communication_modes contains an unsupported value",
                code="invalid_communication_modes",
                field="communication_modes",
                remediation=f"Use one or more of: {allowed}",
            ) from exc
        if not modes or len(set(modes)) != len(modes):
            raise ConfigurationError(
                "communication_modes must contain unique supported modes",
                code="invalid_communication_modes",
                field="communication_modes",
                remediation="Declare each requested mode exactly once",
            )
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed <= _MAX_SEED
        ):
            raise ConfigurationError(
                f"random_seed must be an integer between 0 and {_MAX_SEED}",
                code="invalid_random_seed",
                field="random_seed",
                remediation="Provide a fixed non-negative 63-bit integer seed",
            )
        for field_name, value in (
            ("normalized_zero_is_nondetection", self.normalized_zero_is_nondetection),
            ("allow_duplicate_genes", self.allow_duplicate_genes),
        ):
            if not isinstance(value, bool):
                raise ConfigurationError(
                    f"{field_name} must be a boolean",
                    code="invalid_configuration_field",
                    field=field_name,
                    remediation="Use true or false explicitly",
                )

        object.__setattr__(self, "context_keys", context_keys)
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "expression_transform", transform)
        object.__setattr__(self, "communication_modes", modes)

    @property
    def normalized_only(self) -> bool:
        """Whether input validation must force exploratory behavior."""

        return self.counts_layer is None

    @property
    def digest(self) -> str:
        """SHA-256 digest of the complete canonical configuration."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the persisted configuration representation."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "counts_layer": self.counts_layer,
            "sample_key": self.sample_key,
            "subject_key": self.subject_key,
            "cell_type_key": self.cell_type_key,
            "context_keys": list(self.context_keys),
            "covariates": list(self.covariates),
            "species": self.species,
            "gene_namespace": self.gene_namespace,
            "expression_layer": self.expression_layer,
            "expression_source": self.expression_source,
            "expression_transform": self.expression_transform,
            "normalized_zero_is_nondetection": (self.normalized_zero_is_nondetection),
            "allow_duplicate_genes": self.allow_duplicate_genes,
            "design": self.design,
            "communication_modes": [
                CommunicationMode(mode).value for mode in self.communication_modes
            ],
            "random_seed": self.random_seed,
        }

    def to_json(self) -> str:
        """Return canonical JSON suitable for hashing and persistence."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CrychicConfig:
        """Construct a configuration while enforcing its schema version."""

        payload = dict(value)
        schema_version = payload.pop("schema_version", CONFIG_SCHEMA_VERSION)
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(
                "Configuration schema version is not supported",
                code="unsupported_configuration_schema",
                field="schema_version",
                remediation=f"Migrate the configuration to {CONFIG_SCHEMA_VERSION}",
            )
        known = {field.name for field in fields(cls)}
        unknown = set(payload).difference(known)
        if unknown:
            raise ConfigurationError(
                "Configuration contains unknown fields",
                code="unknown_configuration_field",
                field=sorted(unknown)[0],
                remediation="Remove fields not defined by the current schema",
            )
        return cls(**payload)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, value: str) -> CrychicConfig:
        """Parse a persisted JSON configuration."""

        try:
            payload: Any = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                "Configuration is not valid JSON",
                code="invalid_configuration_json",
                field=None,
                remediation="Provide a JSON object matching config.schema.json",
            ) from exc
        if not isinstance(payload, dict):
            raise ConfigurationError(
                "Configuration JSON must contain an object",
                code="invalid_configuration_json",
                field=None,
                remediation="Provide a JSON object matching config.schema.json",
            )
        return cls.from_dict(payload)
