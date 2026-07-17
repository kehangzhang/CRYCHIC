from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import pytest

import crychic.inference.calibration_attestation as _attestation
from crychic.core import ContractError, SeedLineage, stable_id
from crychic.inference.calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationGeneratorProfile,
    CalibrationReplayAttestation,
    CalibrationReplayRegistry,
    CalibrationReplayRequest,
    attest_calibration_replay,
)


@dataclass(frozen=True, slots=True)
class _Protocol:
    protocol_id: str
    generator_id: str


@dataclass(frozen=True, slots=True)
class _Ledger:
    protocol_id: str
    dependence_structure: str
    non_null_prevalence: float | None
    replicate_index: int
    seed_lineage: SeedLineage
    value: int
    replicate_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replicate_id",
            stable_id(
                "unit_calibration_replay_ledger",
                self._identity_payload(),
                schema_version="1",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "dependence_structure": self.dependence_structure,
            "non_null_prevalence": self.non_null_prevalence,
            "replicate_index": self.replicate_index,
            "seed_lineage": self.seed_lineage.to_dict(),
            "value": self.value,
        }

    def _require_intact(self) -> None:
        expected = stable_id(
            "unit_calibration_replay_ledger",
            self._identity_payload(),
            schema_version="1",
        )
        if expected != self.replicate_id:
            raise ValueError("unit ledger integrity violation")


def _replay(
    protocol: object,
    request: CalibrationReplayRequest,
    config: Mapping[str, object],
) -> _Ledger:
    assert isinstance(protocol, _Protocol)
    offset_value = config["offset"]
    assert isinstance(offset_value, int) and not isinstance(offset_value, bool)
    return _Ledger(
        protocol_id=protocol.protocol_id,
        dependence_structure=request.dependence_structure,
        non_null_prevalence=request.non_null_prevalence,
        replicate_index=request.replicate_index,
        seed_lineage=request.seed_lineage,
        value=offset_value + request.seed_lineage.seed % 97,
    )


def _registry(
    *,
    profile: CalibrationGeneratorProfile = (
        CalibrationGeneratorProfile.RELEASE_APPROVED
    ),
) -> tuple[CalibrationReplayRegistry, _Protocol]:
    registration = _attestation._register_calibration_generator(
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        generator_kind="unit-deterministic-replay",
        generator_version="1.0.0",
        code_version="unit-test-source-v1",
        config={"offset": 11},
        source_paths=(Path(__file__),),
        replay=_replay,
        profile=profile,
    )
    registry = _attestation._freeze_calibration_replay_registry((registration,))
    protocol = _Protocol(
        protocol_id="unit-replay-protocol",
        generator_id=registration.manifest.generator_id,
    )
    return registry, protocol


def _ledgers(
    registry: CalibrationReplayRegistry,
    protocol: _Protocol,
) -> tuple[_Ledger, ...]:
    registration = registry.resolve(protocol.generator_id)
    output: list[_Ledger] = []
    for index in range(4):
        request = CalibrationReplayRequest(
            campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
            protocol_id=protocol.protocol_id,
            dependence_structure="independent",
            non_null_prevalence=0.01,
            replicate_index=index,
            seed_lineage=SeedLineage(20260716).derive(f"replicate={index}"),
        )
        output.append(
            cast(
                _Ledger,
                registration.replay(protocol, request, registration.config),
            )
        )
    return tuple(output)


def test_registry_and_attestation_are_producer_owned() -> None:
    with pytest.raises(TypeError, match="package-owned"):
        CalibrationReplayRegistry()
    with pytest.raises(TypeError, match="producer-owned"):
        CalibrationReplayAttestation()


def test_exact_replay_attests_and_parallelism_does_not_change_identity() -> None:
    registry, protocol = _registry()
    ledgers = _ledgers(registry, protocol)

    serial = attest_calibration_replay(
        registry,
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        protocol=protocol,
        ledgers=tuple(reversed(ledgers)),
    )
    parallel = attest_calibration_replay(
        registry,
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        protocol=protocol,
        ledgers=ledgers,
        n_jobs=2,
    )

    assert serial.attestation_id == parallel.attestation_id
    assert serial.release_approved is True
    assert serial.ledger_digest == serial.replayed_ledger_digest
    assert serial.replicate_ids == tuple(item.replicate_id for item in ledgers)
    assert serial.to_dict()["generator_code_version"] == "unit-test-source-v1"


def test_diagnostic_registration_replays_but_cannot_authorize_release() -> None:
    registry, protocol = _registry(
        profile=CalibrationGeneratorProfile.DIAGNOSTIC_ONLY
    )
    result = attest_calibration_replay(
        registry,
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        protocol=protocol,
        ledgers=_ledgers(registry, protocol),
    )

    assert result.release_approved is False


def test_tampered_or_duplicate_ledgers_fail_closed() -> None:
    registry, protocol = _registry()
    ledgers = _ledgers(registry, protocol)
    tampered = replace(ledgers[0], value=ledgers[0].value + 1)

    with pytest.raises(ContractError) as mismatch:
        attest_calibration_replay(
            registry,
            campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
            protocol=protocol,
            ledgers=(tampered, *ledgers[1:]),
        )
    assert mismatch.value.details.code == "calibration_replay_mismatch"

    with pytest.raises(ContractError) as duplicate:
        attest_calibration_replay(
            registry,
            campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
            protocol=protocol,
            ledgers=(ledgers[0], ledgers[0]),
        )
    assert duplicate.value.details.code == "duplicate_calibration_replay_cell"


def test_unknown_generator_kind_mismatch_and_source_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, protocol = _registry()
    ledgers = _ledgers(registry, protocol)
    with pytest.raises(ContractError) as unknown:
        attest_calibration_replay(
            registry,
            campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
            protocol=replace(protocol, generator_id="unknown-generator"),
            ledgers=ledgers,
        )
    assert unknown.value.details.code == "calibration_generator_not_registered"

    with pytest.raises(ContractError) as kind:
        attest_calibration_replay(
            registry,
            campaign_kind=CalibrationCampaignKind.G3_FREQUENCY,
            protocol=protocol,
            ledgers=ledgers,
        )
    assert kind.value.details.code == "calibration_generator_kind_mismatch"

    monkeypatch.setattr(_attestation, "_source_digest", lambda _: "0" * 64)
    with pytest.raises(ContractError) as drift:
        registry.resolve(protocol.generator_id)
    assert drift.value.details.code == "calibration_generator_code_drift"
