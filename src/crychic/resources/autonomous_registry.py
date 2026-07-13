"""Code-reviewed registrations for receiver-autonomous program resources."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from .contracts import GeneNamespace, Species
from .manifest import ResourceIntegrityError

AutonomousProgramReviewScope = Literal[
    "synthetic_benchmark_only",
    "biological_reference",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_ADAPTER: Final = "crychic-receiver-autonomous-feature-program-tsv-v1"


@dataclass(frozen=True, slots=True)
class ReceiverAutonomousProgramRegistration:
    """One immutable code-reviewed manifest identity and permitted use scope."""

    registration_id: str
    manifest_digest: str
    resource_id: str
    version: str
    species: Species
    gene_namespace: GeneNamespace
    expected_license: str
    adapter_version: str
    review_scope: AutonomousProgramReviewScope
    payload_path: str
    payload_role: str
    payload_sha256: str
    payload_bytes: int
    expected_matrix_digest: str

    def __post_init__(self) -> None:
        required = {
            "registration_id": self.registration_id,
            "resource_id": self.resource_id,
            "version": self.version,
            "expected_license": self.expected_license,
            "adapter_version": self.adapter_version,
            "payload_path": self.payload_path,
            "payload_role": self.payload_role,
        }
        if any(not value or value != value.strip() for value in required.values()):
            raise ValueError("autonomous program registration fields must be unpadded")
        if any(
            _SHA256.fullmatch(digest) is None
            for digest in (
                self.manifest_digest,
                self.payload_sha256,
                self.expected_matrix_digest,
            )
        ):
            raise ValueError("autonomous program registration digests must be SHA-256")
        if self.payload_bytes < 0:
            raise ValueError("autonomous program payload bytes cannot be negative")
        if self.adapter_version != _SUPPORTED_ADAPTER:
            raise ValueError("autonomous program registration adapter is unsupported")
        if self.review_scope not in {
            "synthetic_benchmark_only",
            "biological_reference",
        }:
            raise ValueError("autonomous program review scope is unsupported")


_SYNTHETIC_BENCHMARK_V1 = ReceiverAutonomousProgramRegistration(
    registration_id="crychic.synthetic_receiver_autonomous_program.v1",
    manifest_digest="0e393a3227aee6ed5644c4e211248ca509f9651fd02c49c29f551c2ba62f4746",
    resource_id="crychic_synthetic_receiver_autonomous_program",
    version="1.0.0",
    species=Species.HUMAN,
    gene_namespace=GeneNamespace.HGNC_SYMBOL,
    expected_license="CC0-1.0",
    adapter_version=_SUPPORTED_ADAPTER,
    review_scope="synthetic_benchmark_only",
    payload_path="programs.tsv",
    payload_role="receiver_autonomous_feature_by_program_matrix_v1",
    payload_sha256="2b65b3d6199f3229fe40343ede3973a494be9e557f9488e3b62e5a040eed310e",
    payload_bytes=55,
    expected_matrix_digest=(
        "084f3ce4c24865d29732ed83663c53904b6a281cccf3015fc616e5975ed7d0ad"
    ),
)

_REGISTRATIONS: Final[Mapping[str, ReceiverAutonomousProgramRegistration]] = (
    MappingProxyType(
        {
            _SYNTHETIC_BENCHMARK_V1.registration_id: _SYNTHETIC_BENCHMARK_V1,
        }
    )
)


def require_receiver_autonomous_program_registration(
    registration_id: str,
) -> ReceiverAutonomousProgramRegistration:
    """Return a reviewed registration or fail before reading caller files."""

    if (
        not isinstance(registration_id, str)
        or not registration_id
        or registration_id != registration_id.strip()
    ):
        raise ResourceIntegrityError(
            "Receiver-autonomous registration ID must be a non-empty string",
            code="invalid_resource_registration",
            field="registration_id",
            remediation="Select a code-reviewed receiver-autonomous registration",
        )
    registration = _REGISTRATIONS.get(registration_id)
    if registration is None:
        raise ResourceIntegrityError(
            "Receiver-autonomous resource registration is not approved: "
            f"{registration_id}",
            code="unregistered_receiver_autonomous_resource",
            field="registration_id",
            remediation="Add the reviewed manifest identity to the code registry",
        )
    return registration


def receiver_autonomous_program_registration_ids() -> tuple[str, ...]:
    """List registration IDs without exposing a mutable registry mapping."""

    return tuple(sorted(_REGISTRATIONS))
