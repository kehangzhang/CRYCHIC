"""Machine-readable resource manifests and checksum enforcement."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from crychic.core import ContractError, canonical_digest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ResourceIntegrityError(ContractError):
    """Raised when resource governance or content integrity cannot be verified."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without materializing it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_payload_path(value: str) -> str:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ResourceIntegrityError(
            "Resource payload paths must be relative and cannot traverse parents",
            code="invalid_resource_path",
            field="path",
            remediation=(
                "Store payload paths relative to the caller-provided database root"
            ),
        )
    return candidate.as_posix()


@dataclass(frozen=True, slots=True)
class ResourcePayload:
    """One checksum-pinned file referenced by a resource manifest."""

    path: str
    sha256: str
    role: str = "payload"
    bytes: int | None = None
    upstream_md5: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_payload_path(self.path))
        normalized = self.sha256.lower()
        if _SHA256.fullmatch(normalized) is None:
            raise ResourceIntegrityError(
                "Resource payload SHA-256 must contain 64 hexadecimal characters",
                code="invalid_resource_checksum",
                field="sha256",
                remediation="Regenerate the manifest from the downloaded payload",
            )
        object.__setattr__(self, "sha256", normalized)
        if self.bytes is not None and self.bytes < 0:
            raise ResourceIntegrityError(
                "Resource payload byte size cannot be negative",
                code="invalid_resource_size",
                field="bytes",
                remediation="Record the exact non-negative payload size",
            )


@dataclass(frozen=True, slots=True)
class ResourceManifest:
    """Governance and provenance metadata required before loading a resource."""

    resource_id: str
    version: str
    species: str
    gene_namespace: str
    source_url: str
    license: str
    citation: str
    retrieved_at: str
    adapter_version: str
    payloads: tuple[ResourcePayload, ...]
    transformation_log: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = {
            "resource_id": self.resource_id,
            "version": self.version,
            "species": self.species,
            "gene_namespace": self.gene_namespace,
            "source_url": self.source_url,
            "license": self.license,
            "citation": self.citation,
            "retrieved_at": self.retrieved_at,
            "adapter_version": self.adapter_version,
        }
        empty = sorted(key for key, value in required.items() if not value.strip())
        if empty:
            raise ResourceIntegrityError(
                f"Resource manifest has empty required fields: {', '.join(empty)}",
                code="incomplete_resource_manifest",
                field=empty[0],
                remediation="Record all governance and provenance metadata",
            )
        if not self.payloads:
            raise ResourceIntegrityError(
                "Resource manifest must pin at least one payload",
                code="empty_resource_manifest",
                field="payloads",
                remediation="Add each runtime payload and its SHA-256 checksum",
            )
        paths = tuple(payload.path for payload in self.payloads)
        if len(paths) != len(set(paths)):
            raise ResourceIntegrityError(
                "Resource manifest payload paths must be unique",
                code="duplicate_resource_payload",
                field="payloads",
                remediation="Keep one checksum record per relative payload path",
            )
        object.__setattr__(
            self,
            "payloads",
            tuple(sorted(self.payloads, key=lambda payload: payload.path)),
        )
        object.__setattr__(
            self,
            "transformation_log",
            tuple(entry.strip() for entry in self.transformation_log if entry.strip()),
        )

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the complete manifest."""

        return canonical_digest(self)

    @classmethod
    def from_json(cls, path: str | Path) -> ResourceManifest:
        """Load and validate the CRYCHIC manifest JSON contract."""

        manifest_path = Path(path)
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResourceIntegrityError(
                f"Cannot read resource manifest {manifest_path.name}",
                code="unreadable_resource_manifest",
                field="manifest",
                remediation="Provide a readable CRYCHIC resource manifest JSON file",
            ) from exc
        if not isinstance(raw, dict):
            raise ResourceIntegrityError(
                "Resource manifest root must be a JSON object",
                code="invalid_resource_manifest",
                field="manifest",
                remediation="Regenerate the manifest from the documented schema",
            )
        files = raw.get("payloads", raw.get("files"))
        if not isinstance(files, list):
            raise ResourceIntegrityError(
                "Resource manifest payloads must be a JSON array",
                code="invalid_resource_manifest",
                field="payloads",
                remediation="List runtime payload records under payloads",
            )
        try:
            payloads = tuple(
                ResourcePayload(
                    path=str(item["path"]),
                    sha256=str(item["sha256"]),
                    role=str(item.get("role", "payload")),
                    bytes=None if item.get("bytes") is None else int(item["bytes"]),
                    upstream_md5=(
                        None
                        if item.get("upstream_md5") is None
                        else str(item["upstream_md5"])
                    ),
                )
                for item in files
            )
            return cls(
                resource_id=str(raw["resource_id"]),
                version=str(raw.get("version", raw.get("release", ""))),
                species=str(raw["species"]),
                gene_namespace=str(raw["gene_namespace"]),
                source_url=str(raw["source_url"]),
                license=str(raw["license"]),
                citation=str(raw["citation"]),
                retrieved_at=str(raw["retrieved_at"]),
                adapter_version=str(raw["adapter_version"]),
                payloads=payloads,
                transformation_log=tuple(map(str, raw.get("transformation_log", ()))),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResourceIntegrityError(
                "Resource manifest is missing or has an invalid required field",
                code="invalid_resource_manifest",
                field="manifest",
                remediation="Validate the manifest against the resource contract",
            ) from exc

    def require(
        self,
        *,
        species: str,
        gene_namespace: str,
        license: str | None = None,
    ) -> None:
        """Hard-fail if the manifest describes incompatible resource semantics."""

        expected = {
            "species": (self.species, species),
            "gene_namespace": (self.gene_namespace, gene_namespace),
        }
        if license is not None:
            expected["license"] = (self.license, license)
        for field, (observed, required) in expected.items():
            if observed != required:
                raise ResourceIntegrityError(
                    f"Resource manifest {field} mismatch: {observed!r} != {required!r}",
                    code="resource_metadata_mismatch",
                    field=field,
                    remediation=(
                        "Select a compatible, checksum-pinned resource manifest"
                    ),
                )

    def verify(
        self,
        database_root: str | Path,
        *,
        paths: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """Verify selected payloads below the caller-provided database root."""

        root = Path(database_root)
        selected = None if paths is None else set(paths)
        known = {payload.path for payload in self.payloads}
        if selected is not None and not selected.issubset(known):
            missing = sorted(selected.difference(known))
            raise ResourceIntegrityError(
                f"Manifest does not pin requested payload(s): {', '.join(missing)}",
                code="unpinned_resource_payload",
                field="payloads",
                remediation="Update the manifest before loading new derived files",
            )
        verified: list[str] = []
        for payload in self.payloads:
            if selected is not None and payload.path not in selected:
                continue
            path = root.joinpath(*PurePosixPath(payload.path).parts)
            if not path.is_file():
                raise ResourceIntegrityError(
                    f"Resource payload is missing: {payload.path}",
                    code="missing_resource_payload",
                    field="path",
                    remediation=(
                        "Download the frozen payload or select offline cache data"
                    ),
                )
            observed = sha256_file(path)
            if observed != payload.sha256:
                raise ResourceIntegrityError(
                    f"Resource checksum mismatch for {payload.path}",
                    code="resource_checksum_mismatch",
                    field="sha256",
                    remediation=(
                        "Remove the corrupted payload and retrieve the pinned version"
                    ),
                )
            if payload.bytes is not None and path.stat().st_size != payload.bytes:
                raise ResourceIntegrityError(
                    f"Resource byte-size mismatch for {payload.path}",
                    code="resource_size_mismatch",
                    field="bytes",
                    remediation="Remove the incomplete payload and retrieve it again",
                )
            verified.append(payload.path)
        return tuple(verified)


def verify_gnu_checksum_file(
    directory: str | Path,
    checksum_file: str | Path,
    *,
    required_paths: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Verify a GNU-style SHA-256 list without accepting path traversal."""

    root = Path(directory)
    checksum_path = Path(checksum_file)
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ResourceIntegrityError(
            "Cannot read resource checksum list",
            code="unreadable_resource_manifest",
            field="checksum_file",
            remediation="Provide the checksum file shipped with the resource export",
        ) from exc
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or _SHA256.fullmatch(parts[0].lower()) is None:
            raise ResourceIntegrityError(
                f"Malformed checksum record at line {line_number}",
                code="invalid_resource_checksum",
                field="checksum_file",
                remediation="Regenerate the GNU-style SHA-256 list",
            )
        relative = _relative_payload_path(parts[1].lstrip("*"))
        entries[relative] = parts[0].lower()
    selected = set(entries) if required_paths is None else set(required_paths)
    missing = selected.difference(entries)
    if missing:
        raise ResourceIntegrityError(
            "Checksum list is missing required payload(s): "
            + ", ".join(sorted(missing)),
            code="unpinned_resource_payload",
            field="checksum_file",
            remediation="Use the checksum list generated with this resource version",
        )
    verified: list[str] = []
    for relative in sorted(selected):
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise ResourceIntegrityError(
                f"Resource payload is missing: {relative}",
                code="missing_resource_payload",
                field="path",
                remediation="Restore the checksum-pinned resource export",
            )
        if sha256_file(path) != entries[relative]:
            raise ResourceIntegrityError(
                f"Resource checksum mismatch for {relative}",
                code="resource_checksum_mismatch",
                field="sha256",
                remediation="Restore the unmodified checksum-pinned resource payload",
            )
        verified.append(relative)
    return tuple(verified)
