"""Schema-version identifiers independent of the package version."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ContractError


@dataclass(frozen=True, order=True, slots=True)
class SchemaVersion:
    """A small semantic-version value used by persisted contracts."""

    major: int
    minor: int = 0
    patch: int = 0

    def __post_init__(self) -> None:
        components = (self.major, self.minor, self.patch)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in components
        ):
            raise ContractError(
                "Schema version components must be non-negative integers",
                code="invalid_schema_version",
                field="schema_version",
                remediation="Use a version such as 0.1.0",
            )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, value: str) -> SchemaVersion:
        """Parse a three-component semantic schema version."""

        try:
            parts = tuple(int(part) for part in value.split("."))
        except ValueError as exc:
            raise ContractError(
                "Schema version must contain only integer components",
                code="invalid_schema_version",
                field="schema_version",
                remediation="Use a version such as 0.1.0",
            ) from exc
        if len(parts) != 3:
            raise ContractError(
                "Schema version must have major, minor, and patch components",
                code="invalid_schema_version",
                field="schema_version",
                remediation="Use a version such as 0.1.0",
            )
        return cls(*parts)


CONFIG_SCHEMA_VERSION = str(SchemaVersion(0, 1, 0))
PROVENANCE_SCHEMA_VERSION = str(SchemaVersion(0, 1, 0))
