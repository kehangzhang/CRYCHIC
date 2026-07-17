"""Replay-backed generator attestations for G3 calibration campaigns.

Content hashes make calibration ledgers tamper evident; they do not establish
that a declared generator produced those ledgers.  This module supplies the
separate boundary: package-owned registrations bind a replay callable to its
configuration and source files, and an attestation is emitted only after every
supplied replicate is reproduced exactly.

The object capabilities used here prevent accidental construction through the
public API.  They are not a cryptographic defence against an attacker who can
modify the installed Python package.  Release provenance must additionally pin
the wheel or source revision containing the approved registration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from crychic.core import (
    ContractError,
    SeedLineage,
    canonical_digest,
    canonical_json,
    stable_id,
)

_MANIFEST_SCHEMA_VERSION = "1.0.0"
_ATTESTATION_SCHEMA_VERSION = "1.0.0"
_REGISTRY_TOKEN = object()
_ATTESTATION_TOKEN = object()


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _sha256(value: object, *, field_name: str) -> str:
    text = _name(value, field_name=field_name)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _positive_jobs(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("n_jobs must be an integer >= 1")
    return value


def _contract_error(
    message: str,
    *,
    code: str,
    field: str,
    remediation: str,
) -> ContractError:
    return ContractError(
        message,
        code=code,
        field=field,
        remediation=remediation,
    )


class CalibrationCampaignKind(StrEnum):
    """Calibration campaigns supported by the replay registry."""

    G3_FREQUENCY = "g3_frequency"
    G3_PROBABILITY = "g3_probability"


class CalibrationGeneratorProfile(StrEnum):
    """Whether a registered generator may authorize a release gate."""

    DIAGNOSTIC_ONLY = "diagnostic_only"
    RELEASE_APPROVED = "release_approved"


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationReplayRequest:
    """One deterministic scenario cell requested from a registered generator."""

    campaign_kind: CalibrationCampaignKind | str
    protocol_id: str
    dependence_structure: str
    non_null_prevalence: float | None
    replicate_index: int
    seed_lineage: SeedLineage
    request_id: str = field(init=False)

    def __post_init__(self) -> None:
        campaign = CalibrationCampaignKind(self.campaign_kind)
        protocol = _name(self.protocol_id, field_name="protocol_id")
        dependence = _name(
            self.dependence_structure, field_name="dependence_structure"
        )
        prevalence = self.non_null_prevalence
        if prevalence is not None:
            if isinstance(prevalence, bool):
                raise ValueError("non_null_prevalence must be finite and in [0, 1]")
            prevalence = float(prevalence)
            if not math.isfinite(prevalence) or not 0.0 <= prevalence <= 1.0:
                raise ValueError("non_null_prevalence must be finite and in [0, 1]")
            prevalence = 0.0 if prevalence == 0.0 else prevalence
        index = self.replicate_index
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("replicate_index must be an integer >= 0")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        object.__setattr__(self, "campaign_kind", campaign)
        object.__setattr__(self, "protocol_id", protocol)
        object.__setattr__(self, "dependence_structure", dependence)
        object.__setattr__(self, "non_null_prevalence", prevalence)
        object.__setattr__(
            self,
            "request_id",
            stable_id(
                "calibration_replay_request",
                self._identity_payload(),
                schema_version="1",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "campaign_kind": CalibrationCampaignKind(self.campaign_kind).value,
            "protocol_id": self.protocol_id,
            "dependence_structure": self.dependence_structure,
            "non_null_prevalence": self.non_null_prevalence,
            "replicate_index": self.replicate_index,
            "seed_lineage": self.seed_lineage.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {"request_id": self.request_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationGeneratorManifest:
    """Portable identity of one exact replay implementation."""

    campaign_kind: CalibrationCampaignKind | str
    generator_kind: str
    generator_version: str
    code_version: str
    replay_entrypoint: str
    config_digest: str
    implementation_digest: str
    profile: CalibrationGeneratorProfile | str
    schema_version: str = _MANIFEST_SCHEMA_VERSION
    generator_id: str = field(init=False)

    def __post_init__(self) -> None:
        campaign = CalibrationCampaignKind(self.campaign_kind)
        profile = CalibrationGeneratorProfile(self.profile)
        values = {
            "generator_kind": _name(
                self.generator_kind, field_name="generator_kind"
            ),
            "generator_version": _name(
                self.generator_version, field_name="generator_version"
            ),
            "code_version": _name(self.code_version, field_name="code_version"),
            "replay_entrypoint": _name(
                self.replay_entrypoint, field_name="replay_entrypoint"
            ),
            "config_digest": _sha256(
                self.config_digest, field_name="config_digest"
            ),
            "implementation_digest": _sha256(
                self.implementation_digest, field_name="implementation_digest"
            ),
        }
        if self.schema_version != _MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "calibration generator manifest schema_version must be "
                f"{_MANIFEST_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "campaign_kind", campaign)
        object.__setattr__(self, "profile", profile)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "generator_id",
            stable_id(
                "calibration_generator",
                self._identity_payload(),
                schema_version="1",
            ),
        )

    @property
    def release_approved(self) -> bool:
        return self.profile is CalibrationGeneratorProfile.RELEASE_APPROVED

    def _identity_payload(self) -> dict[str, object]:
        return {
            "campaign_kind": CalibrationCampaignKind(self.campaign_kind).value,
            "generator_kind": self.generator_kind,
            "generator_version": self.generator_version,
            "code_version": self.code_version,
            "replay_entrypoint": self.replay_entrypoint,
            "config_digest": self.config_digest,
            "implementation_digest": self.implementation_digest,
            "profile": CalibrationGeneratorProfile(self.profile).value,
            "schema_version": self.schema_version,
        }

    def _require_intact(self) -> None:
        repeated = CalibrationGeneratorManifest(
            campaign_kind=self.campaign_kind,
            generator_kind=self.generator_kind,
            generator_version=self.generator_version,
            code_version=self.code_version,
            replay_entrypoint=self.replay_entrypoint,
            config_digest=self.config_digest,
            implementation_digest=self.implementation_digest,
            profile=self.profile,
            schema_version=self.schema_version,
        )
        if repeated.generator_id != self.generator_id:
            raise _contract_error(
                "Calibration generator manifest failed integrity validation",
                code="calibration_generator_manifest_integrity_violation",
                field="generator_id",
                remediation="Recreate the manifest from the registered generator",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"generator_id": self.generator_id, **self._identity_payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CalibrationGeneratorManifest:
        """Restore and verify one persisted generator declaration."""

        required = {
            "generator_id",
            "campaign_kind",
            "generator_kind",
            "generator_version",
            "code_version",
            "replay_entrypoint",
            "config_digest",
            "implementation_digest",
            "profile",
            "schema_version",
        }
        if set(value) != required:
            raise _contract_error(
                "Persisted calibration generator manifest has invalid fields",
                code="invalid_calibration_generator_manifest",
                field="generator_manifest",
                remediation="Regenerate the protocol with the current producer",
            )
        repeated = cls(
            campaign_kind=cast(str, value["campaign_kind"]),
            generator_kind=cast(str, value["generator_kind"]),
            generator_version=cast(str, value["generator_version"]),
            code_version=cast(str, value["code_version"]),
            replay_entrypoint=cast(str, value["replay_entrypoint"]),
            config_digest=cast(str, value["config_digest"]),
            implementation_digest=cast(str, value["implementation_digest"]),
            profile=cast(str, value["profile"]),
            schema_version=cast(str, value["schema_version"]),
        )
        if value["generator_id"] != repeated.generator_id:
            raise _contract_error(
                "Persisted calibration generator identity does not match its fields",
                code="calibration_generator_manifest_integrity_violation",
                field="generator_id",
                remediation="Reject the artifact and regenerate the protocol",
            )
        return repeated


@runtime_checkable
class CalibrationReplayLedger(Protocol):
    """Internal structural contract shared by G3-F and G3-P replicate ledgers."""

    @property
    def protocol_id(self) -> str: ...

    @property
    def dependence_structure(self) -> object: ...

    @property
    def non_null_prevalence(self) -> float | None: ...

    @property
    def replicate_index(self) -> int: ...

    @property
    def seed_lineage(self) -> SeedLineage: ...

    @property
    def replicate_id(self) -> str: ...

    def _identity_payload(self) -> dict[str, object]: ...

    def _require_intact(self) -> None: ...


CalibrationReplayFunction = Callable[
    [object, CalibrationReplayRequest, Mapping[str, object]],
    CalibrationReplayLedger,
]


def _source_digest(source_paths: Sequence[Path]) -> str:
    records: list[dict[str, str]] = []
    paths = tuple(item.resolve() for item in source_paths)
    if len({item.name for item in paths}) != len(paths):
        raise ValueError("calibration generator source filenames must be unique")
    for index, path in enumerate(paths):
        if not path.is_file():
            raise ValueError(f"calibration generator source does not exist: {path}")
        records.append(
            {
                "source_index": str(index),
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not records:
        raise ValueError("calibration generator requires at least one source file")
    return canonical_digest(records)


@dataclass(frozen=True, slots=True)
class _CalibrationGeneratorRegistration:
    manifest: CalibrationGeneratorManifest
    config_json: str
    source_paths: tuple[Path, ...]
    replay: CalibrationReplayFunction

    @property
    def config(self) -> Mapping[str, object]:
        value = json.loads(self.config_json)
        return cast(Mapping[str, object], MappingProxyType(value))

    def _require_intact(self) -> None:
        self.manifest._require_intact()
        if canonical_digest(self.config) != self.manifest.config_digest:
            raise _contract_error(
                "Registered calibration generator configuration drifted",
                code="calibration_generator_config_drift",
                field="config_digest",
                remediation="Recreate the protocol and rerun the complete campaign",
            )
        if _source_digest(self.source_paths) != self.manifest.implementation_digest:
            raise _contract_error(
                "Registered calibration generator implementation drifted",
                code="calibration_generator_code_drift",
                field="implementation_digest",
                remediation="Recreate the protocol and rerun the complete campaign",
            )


def _register_calibration_generator(
    *,
    campaign_kind: CalibrationCampaignKind | str,
    generator_kind: str,
    generator_version: str,
    code_version: str,
    config: Mapping[str, object],
    source_paths: Sequence[str | Path],
    replay: CalibrationReplayFunction,
    profile: CalibrationGeneratorProfile | str,
) -> _CalibrationGeneratorRegistration:
    """Create a package-owned registration; intentionally not a public API."""

    if not callable(replay):
        raise TypeError("replay must be callable")
    config_json = canonical_json(dict(config))
    canonical_config = cast(Mapping[str, object], json.loads(config_json))
    paths = tuple(Path(item).resolve() for item in source_paths)
    entrypoint = f"{replay.__module__}:{replay.__qualname__}"
    manifest = CalibrationGeneratorManifest(
        campaign_kind=campaign_kind,
        generator_kind=generator_kind,
        generator_version=generator_version,
        code_version=code_version,
        replay_entrypoint=entrypoint,
        config_digest=canonical_digest(canonical_config),
        implementation_digest=_source_digest(paths),
        profile=profile,
    )
    registration = _CalibrationGeneratorRegistration(
        manifest=manifest,
        config_json=config_json,
        source_paths=paths,
        replay=replay,
    )
    registration._require_intact()
    return registration


@dataclass(frozen=True, slots=True, init=False)
class CalibrationReplayRegistry:
    """Immutable package-owned set of exact generator registrations."""

    manifests: tuple[CalibrationGeneratorManifest, ...]
    registry_id: str
    _registrations: Mapping[str, _CalibrationGeneratorRegistration]
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "CalibrationReplayRegistry is package-owned; use an approved registry"
        )

    def _require_intact(self) -> None:
        if self._producer_token is not _REGISTRY_TOKEN:
            raise _contract_error(
                "Calibration replay registry is not producer-owned",
                code="calibration_registry_integrity_violation",
                field="registry_id",
                remediation="Use the registry shipped with this CRYCHIC build",
            )
        for registration in self._registrations.values():
            registration._require_intact()
        expected_manifests = tuple(
            registration.manifest
            for _, registration in sorted(self._registrations.items())
        )
        expected_id = stable_id(
            "calibration_replay_registry",
            {"generator_ids": [item.generator_id for item in expected_manifests]},
            schema_version="1",
        )
        if self.manifests != expected_manifests or self.registry_id != expected_id:
            raise _contract_error(
                "Calibration replay registry failed integrity validation",
                code="calibration_registry_integrity_violation",
                field="registry_id",
                remediation="Use the registry shipped with this CRYCHIC build",
            )

    def resolve(self, generator_id: str) -> _CalibrationGeneratorRegistration:
        self._require_intact()
        identifier = _name(generator_id, field_name="generator_id")
        registration = self._registrations.get(identifier)
        if registration is None:
            raise _contract_error(
                "Calibration protocol names an unregistered generator",
                code="calibration_generator_not_registered",
                field="generator_id",
                remediation=(
                    "Use a generator registered by this CRYCHIC build and rerun the "
                    "complete campaign"
                ),
            )
        registration._require_intact()
        return registration

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "registry_id": self.registry_id,
            "generators": [item.to_dict() for item in self.manifests],
        }


def _freeze_calibration_replay_registry(
    registrations: Sequence[_CalibrationGeneratorRegistration],
) -> CalibrationReplayRegistry:
    """Freeze package registrations; intentionally not a public API."""

    supplied = tuple(registrations)
    if not supplied:
        raise ValueError("calibration replay registry cannot be empty")
    by_id = {item.manifest.generator_id: item for item in supplied}
    if len(by_id) != len(supplied):
        raise ValueError("calibration replay registry contains duplicate generators")
    ordered = dict(sorted(by_id.items()))
    self = object.__new__(CalibrationReplayRegistry)
    manifests = tuple(item.manifest for item in ordered.values())
    for name, value in {
        "manifests": manifests,
        "registry_id": stable_id(
            "calibration_replay_registry",
            {"generator_ids": [item.generator_id for item in manifests]},
            schema_version="1",
        ),
        "_registrations": MappingProxyType(ordered),
        "_producer_token": _REGISTRY_TOKEN,
    }.items():
        object.__setattr__(self, name, value)
    self._require_intact()
    return self


@dataclass(frozen=True, slots=True, init=False)
class CalibrationReplayAttestation:
    """Producer-owned proof that an exact raw ledger was replayed."""

    campaign_kind: CalibrationCampaignKind
    protocol_id: str
    generator_id: str
    registry_id: str
    generator_config_digest: str
    generator_implementation_digest: str
    generator_code_version: str
    release_approved: bool
    replicate_ids: tuple[str, ...]
    request_ids: tuple[str, ...]
    ledger_digest: str
    replayed_ledger_digest: str
    schema_version: str
    attestation_id: str
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "CalibrationReplayAttestation is producer-owned; replay the campaign"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "campaign_kind": self.campaign_kind.value,
            "protocol_id": self.protocol_id,
            "generator_id": self.generator_id,
            "registry_id": self.registry_id,
            "generator_config_digest": self.generator_config_digest,
            "generator_implementation_digest": (
                self.generator_implementation_digest
            ),
            "generator_code_version": self.generator_code_version,
            "release_approved": self.release_approved,
            "replicate_ids": list(self.replicate_ids),
            "request_ids": list(self.request_ids),
            "ledger_digest": self.ledger_digest,
            "replayed_ledger_digest": self.replayed_ledger_digest,
            "schema_version": self.schema_version,
        }

    def _require_intact(self) -> None:
        expected = stable_id(
            "calibration_replay_attestation",
            self._identity_payload(),
            schema_version="1",
        )
        if (
            self._producer_token is not _ATTESTATION_TOKEN
            or self.schema_version != _ATTESTATION_SCHEMA_VERSION
            or self.ledger_digest != self.replayed_ledger_digest
            or self.attestation_id != expected
        ):
            raise _contract_error(
                "Calibration replay attestation failed integrity validation",
                code="calibration_attestation_integrity_violation",
                field="attestation_id",
                remediation="Replay the complete campaign with the approved registry",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"attestation_id": self.attestation_id, **self._identity_payload()}


def _request_from_ledger(
    campaign_kind: CalibrationCampaignKind,
    protocol_id: str,
    ledger: CalibrationReplayLedger,
) -> CalibrationReplayRequest:
    dependence = getattr(
        ledger.dependence_structure,
        "value",
        ledger.dependence_structure,
    )
    return CalibrationReplayRequest(
        campaign_kind=campaign_kind,
        protocol_id=protocol_id,
        dependence_structure=cast(str, dependence),
        non_null_prevalence=ledger.non_null_prevalence,
        replicate_index=ledger.replicate_index,
        seed_lineage=ledger.seed_lineage,
    )


def _ledger_digest(ledgers: Sequence[CalibrationReplayLedger]) -> str:
    return canonical_digest(
        [
            {
                "replicate_id": item.replicate_id,
                "identity": item._identity_payload(),
            }
            for item in ledgers
        ]
    )


def attest_calibration_replay(
    registry: CalibrationReplayRegistry,
    *,
    campaign_kind: CalibrationCampaignKind | str,
    protocol: object,
    ledgers: Sequence[CalibrationReplayLedger],
    n_jobs: int = 1,
) -> CalibrationReplayAttestation:
    """Replay and compare every supplied ledger under one registered generator.

    ``n_jobs`` controls only bounded execution.  Results are restored to the
    canonical request order, so serial and parallel runs have identical
    attestation IDs.
    """

    if not isinstance(registry, CalibrationReplayRegistry):
        raise TypeError("registry must be a producer-owned CalibrationReplayRegistry")
    registry._require_intact()
    campaign = CalibrationCampaignKind(campaign_kind)
    jobs = _positive_jobs(n_jobs)
    protocol_id = _name(
        getattr(protocol, "protocol_id", None), field_name="protocol_id"
    )
    generator_id = _name(
        getattr(protocol, "generator_id", None), field_name="generator_id"
    )
    registration = registry.resolve(generator_id)
    if registration.manifest.campaign_kind is not campaign:
        raise _contract_error(
            "Calibration generator is registered for a different campaign kind",
            code="calibration_generator_kind_mismatch",
            field="campaign_kind",
            remediation="Use the generator registered for this calibration protocol",
        )
    supplied = tuple(ledgers)
    if not supplied:
        raise ValueError("calibration replay requires at least one ledger")
    for ledger in supplied:
        if not isinstance(ledger, CalibrationReplayLedger):
            raise TypeError("ledgers must satisfy CalibrationReplayLedger")
        ledger._require_intact()
        if ledger.protocol_id != protocol_id:
            raise _contract_error(
                "Calibration ledger belongs to a different protocol",
                code="calibration_replay_protocol_mismatch",
                field="protocol_id",
                remediation="Replay only ledgers produced for this protocol",
            )
    ordered = tuple(
        sorted(
            supplied,
            key=lambda item: (
                str(
                    getattr(
                        item.dependence_structure,
                        "value",
                        item.dependence_structure,
                    )
                ),
                -1.0
                if item.non_null_prevalence is None
                else float(item.non_null_prevalence),
                item.replicate_index,
            ),
        )
    )
    requests = tuple(
        _request_from_ledger(campaign, protocol_id, item) for item in ordered
    )
    keys = tuple(
        (
            item.dependence_structure,
            item.non_null_prevalence,
            item.replicate_index,
        )
        for item in requests
    )
    if len(keys) != len(set(keys)):
        raise _contract_error(
            "Calibration replay contains duplicate scenario replicate cells",
            code="duplicate_calibration_replay_cell",
            field="dependence_structure,non_null_prevalence,replicate_index",
            remediation="Retain exactly one ledger for every attempted replicate",
        )

    def replay_one(request: CalibrationReplayRequest) -> CalibrationReplayLedger:
        value = registration.replay(protocol, request, registration.config)
        if not isinstance(value, CalibrationReplayLedger):
            raise _contract_error(
                "Registered generator returned an invalid replay ledger",
                code="calibration_replay_invalid_output",
                field="replay_entrypoint",
                remediation="Fix the registered generator and rerun the campaign",
            )
        value._require_intact()
        return value

    if jobs == 1 or len(requests) == 1:
        replayed = tuple(replay_one(item) for item in requests)
    else:
        with ThreadPoolExecutor(max_workers=min(jobs, len(requests))) as executor:
            replayed = tuple(executor.map(replay_one, requests))
    observed_digest = _ledger_digest(ordered)
    replayed_digest = _ledger_digest(replayed)
    mismatches = tuple(
        request.request_id
        for request, observed, expected in zip(
            requests, ordered, replayed, strict=True
        )
        if observed.replicate_id != expected.replicate_id
        or observed._identity_payload() != expected._identity_payload()
    )
    if mismatches or observed_digest != replayed_digest:
        raise _contract_error(
            "Calibration ledger does not match registered generator replay",
            code="calibration_replay_mismatch",
            field="replicate_id",
            remediation="Discard the ledger and rerun the registered generator",
        )
    manifest = registration.manifest
    self = object.__new__(CalibrationReplayAttestation)
    values: dict[str, object] = {
        "campaign_kind": campaign,
        "protocol_id": protocol_id,
        "generator_id": manifest.generator_id,
        "registry_id": registry.registry_id,
        "generator_config_digest": manifest.config_digest,
        "generator_implementation_digest": manifest.implementation_digest,
        "generator_code_version": manifest.code_version,
        "release_approved": manifest.release_approved,
        "replicate_ids": tuple(item.replicate_id for item in ordered),
        "request_ids": tuple(item.request_id for item in requests),
        "ledger_digest": observed_digest,
        "replayed_ledger_digest": replayed_digest,
        "schema_version": _ATTESTATION_SCHEMA_VERSION,
        "_producer_token": _ATTESTATION_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "attestation_id",
        stable_id(
            "calibration_replay_attestation",
            self._identity_payload(),
            schema_version="1",
        ),
    )
    self._require_intact()
    return self


__all__ = [
    "CalibrationCampaignKind",
    "CalibrationGeneratorManifest",
    "CalibrationGeneratorProfile",
    "CalibrationReplayAttestation",
    "CalibrationReplayRegistry",
    "CalibrationReplayRequest",
]
