"""In-memory provenance primitives with canonical serialization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType

from .errors import ContractError
from .seed import SeedLineage
from .serialization import canonical_digest, canonical_json
from .version import PROVENANCE_SCHEMA_VERSION, SchemaVersion

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_digest(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ContractError(
            f"{field_name} must be a lowercase SHA-256 digest",
            code="invalid_digest",
            field=field_name,
            remediation="Compute the full 64-character SHA-256 checksum",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RunProvenance:
    """Minimum provenance required for a Phase 0 persisted artifact."""

    package_version: str
    config_digest: str
    seed_lineage: SeedLineage
    git_commit: str | None = None
    git_dirty: bool = False
    input_digest: str | None = None
    resource_digests: Mapping[str, str] = field(default_factory=dict)
    result_schema_version: str | None = None
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.package_version, str) or not self.package_version:
            raise ContractError(
                "package_version cannot be empty",
                code="missing_provenance_field",
                field="package_version",
                remediation="Record the installed CRYCHIC package version",
            )
        _validate_digest(self.config_digest, field_name="config_digest")
        if self.input_digest is not None:
            _validate_digest(self.input_digest, field_name="input_digest")
        if self.git_commit is not None and (
            not isinstance(self.git_commit, str)
            or _GIT_COMMIT_PATTERN.fullmatch(self.git_commit) is None
        ):
            raise ContractError(
                "git_commit must be a hexadecimal Git object identifier",
                code="invalid_git_commit",
                field="git_commit",
                remediation="Record the full or unambiguous abbreviated commit ID",
            )
        if not isinstance(self.git_dirty, bool):
            raise ContractError(
                "git_dirty must be a boolean",
                code="invalid_provenance_field",
                field="git_dirty",
                remediation="Record whether the source worktree had modifications",
            )
        if not isinstance(self.created_at, str):
            raise ContractError(
                "created_at must be an ISO-8601 timestamp",
                code="invalid_timestamp",
                field="created_at",
                remediation="Use an explicit timezone, preferably UTC",
            )
        try:
            parsed_time = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractError(
                "created_at must be an ISO-8601 timestamp",
                code="invalid_timestamp",
                field="created_at",
                remediation="Use an explicit timezone, preferably UTC",
            ) from exc
        if parsed_time.tzinfo is None:
            raise ContractError(
                "created_at must include an explicit timezone",
                code="invalid_timestamp",
                field="created_at",
                remediation="Use an explicit timezone, preferably UTC",
            )
        if not isinstance(self.seed_lineage, SeedLineage):
            raise ContractError(
                "seed_lineage must be a SeedLineage contract",
                code="invalid_provenance_field",
                field="seed_lineage",
                remediation="Construct seed lineage from the configured root seed",
            )
        if not isinstance(self.resource_digests, Mapping):
            raise ContractError(
                "resource_digests must map resource IDs to SHA-256 digests",
                code="invalid_provenance_field",
                field="resource_digests",
                remediation="Provide a mapping from stable resource IDs to checksums",
            )
        if self.result_schema_version is not None:
            if not isinstance(self.result_schema_version, str):
                raise ContractError(
                    "result_schema_version must be a semantic schema version",
                    code="invalid_provenance_field",
                    field="result_schema_version",
                    remediation="Use a version such as 0.1.0",
                )
            SchemaVersion.parse(self.result_schema_version)
        resources: dict[str, str] = {}
        for resource_id, digest in self.resource_digests.items():
            if (
                not isinstance(resource_id, str)
                or not resource_id
                or resource_id != resource_id.strip()
            ):
                raise ContractError(
                    "Resource identifiers cannot be empty or padded",
                    code="invalid_resource_id",
                    field="resource_digests",
                    remediation="Use the stable resource manifest identifier",
                )
            _validate_digest(digest, field_name="resource_digests")
            resources[resource_id] = digest
        object.__setattr__(
            self,
            "resource_digests",
            MappingProxyType(dict(sorted(resources.items()))),
        )

    @property
    def digest(self) -> str:
        """Return the canonical digest of this complete record."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the persisted provenance representation."""

        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "created_at": self.created_at,
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config_digest": self.config_digest,
            "input_digest": self.input_digest,
            "resource_digests": dict(self.resource_digests),
            "result_schema_version": self.result_schema_version,
            "seed_lineage": self.seed_lineage.to_dict(),
        }

    def to_json(self) -> str:
        """Return canonical JSON suitable for persistence."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RunProvenance:
        """Restore a provenance record and reject incompatible fields."""

        payload = dict(value)
        schema_version = payload.pop("schema_version", None)
        if schema_version != PROVENANCE_SCHEMA_VERSION:
            raise ContractError(
                "Provenance schema version is not supported",
                code="unsupported_provenance_schema",
                field="schema_version",
                remediation=f"Migrate provenance to {PROVENANCE_SCHEMA_VERSION}",
            )
        expected = {
            "created_at",
            "package_version",
            "git_commit",
            "git_dirty",
            "config_digest",
            "input_digest",
            "resource_digests",
            "result_schema_version",
            "seed_lineage",
        }
        unknown = set(payload).difference(expected)
        if unknown:
            raise ContractError(
                "Provenance contains unknown fields",
                code="unknown_provenance_field",
                field=sorted(unknown)[0],
                remediation="Migrate the artifact with a supported schema migration",
            )
        seed_value = payload.get("seed_lineage")
        if not isinstance(seed_value, Mapping):
            raise ContractError(
                "Provenance seed_lineage must be an object",
                code="invalid_provenance_field",
                field="seed_lineage",
                remediation="Persist the complete root, path, and derived seed",
            )
        payload["seed_lineage"] = SeedLineage.from_dict(seed_value)
        return cls(**payload)  # type: ignore[arg-type]
