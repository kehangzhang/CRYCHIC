from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import crychic.response.autonomous as autonomous_module
from crychic.core import ContractError
from crychic.resources import (
    GeneNamespace,
    ResourceIntegrityError,
    ResourceManifest,
    Species,
)
from crychic.resources.autonomous_registry import (
    ReceiverAutonomousProgramRegistration,
    receiver_autonomous_program_registration_ids,
    require_receiver_autonomous_program_registration,
)
from crychic.response import (
    build_receiver_autonomous_program_resource,
    load_receiver_autonomous_program_resource,
)

_REGISTRATION_ID = "crychic.synthetic_receiver_autonomous_program.v1"
_MANIFEST_DIGEST = "0e393a3227aee6ed5644c4e211248ca509f9651fd02c49c29f551c2ba62f4746"
_PAYLOAD_SHA256 = "2b65b3d6199f3229fe40343ede3973a494be9e557f9488e3b62e5a040eed310e"
_MATRIX_DIGEST = "084f3ce4c24865d29732ed83663c53904b6a281cccf3015fc616e5975ed7d0ad"
_ARTIFACT_ID = "receiver_autonomous_program_resource_83ab1eb0b30386b66f0aab2225c1096a"
_FIXTURE_ROOT = (
    Path(__file__).parents[3]
    / "benchmarks"
    / "fixtures"
    / "synthetic_receiver_autonomous_program"
)


def _load(root: Path, manifest_path: Path | None = None):
    resolved_manifest = (
        root / "manifest.json" if manifest_path is None else manifest_path
    )
    return load_receiver_autonomous_program_resource(
        root,
        manifest_path=resolved_manifest,
        registration_id=_REGISTRATION_ID,
    )


def _copied_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "registered_fixture"
    shutil.copytree(_FIXTURE_ROOT, destination)
    return destination


def _rewrite_manifest(root: Path, **changes: object) -> str:
    path = root / "manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(changes)
    path.write_text(json.dumps(record), encoding="utf-8")
    return ResourceManifest.from_json(path).digest


def _patch_registration_digest(
    monkeypatch: pytest.MonkeyPatch,
    digest: str,
    **changes: object,
) -> ReceiverAutonomousProgramRegistration:
    registration = replace(
        require_receiver_autonomous_program_registration(_REGISTRATION_ID),
        manifest_digest=digest,
        **changes,
    )
    monkeypatch.setattr(
        autonomous_module,
        "require_receiver_autonomous_program_registration",
        lambda registration_id: registration,
    )
    return registration


def test_checked_in_fixture_loads_only_through_frozen_registration() -> None:
    manifest = ResourceManifest.from_json(_FIXTURE_ROOT / "manifest.json")
    payload = _FIXTURE_ROOT / "programs.tsv"

    assert manifest.digest == _MANIFEST_DIGEST
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == _PAYLOAD_SHA256
    assert payload.stat().st_size == 55
    assert receiver_autonomous_program_registration_ids() == (_REGISTRATION_ID,)
    registration = require_receiver_autonomous_program_registration(_REGISTRATION_ID)
    assert registration.manifest_digest == _MANIFEST_DIGEST
    assert registration.payload_path == "programs.tsv"
    assert registration.payload_role == (
        "receiver_autonomous_feature_by_program_matrix_v1"
    )
    assert registration.payload_sha256 == _PAYLOAD_SHA256
    assert registration.payload_bytes == 55
    assert registration.expected_matrix_digest == _MATRIX_DIGEST

    resource = _load(_FIXTURE_ROOT)

    assert resource.artifact_id == _ARTIFACT_ID
    assert resource.manifest_digest == _MANIFEST_DIGEST
    assert resource.matrix_digest == _MATRIX_DIGEST
    assert resource.verification_status == "manifest_verified_static_trusted_v1"
    assert resource.registration_id == _REGISTRATION_ID
    assert resource.review_scope == "synthetic_benchmark_only"
    assert resource.expected_license == "CC0-1.0"
    assert resource.registered_payload_path == "programs.tsv"
    assert resource.registered_payload_role == (
        "receiver_autonomous_feature_by_program_matrix_v1"
    )
    assert resource.registered_payload_sha256 == _PAYLOAD_SHA256
    assert resource.registered_payload_bytes == 55
    assert resource.is_manifest_verified_trusted
    assert not resource.is_biological_reference_trusted
    assert resource.feature_ids == ("DUSP1", "FOS", "JUN")
    assert resource.program_ids == ("generic_immediate_early",)
    np.testing.assert_array_equal(resource.matrix, np.ones((3, 1)))
    assert not resource.matrix.flags.writeable
    assert resource.to_dict()["registration_id"] == _REGISTRATION_ID


def test_public_builder_remains_unverified_and_unregistered() -> None:
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [-1.0]]),
        feature_ids=("G1", "G2"),
        program_ids=("stress",),
        resource_id="caller_programs",
        version="1",
        manifest_digest="a" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )

    assert resource.verification_status == "caller_declared_static_unverified"
    assert resource.registration_id is None
    assert resource.review_scope is None
    assert resource.expected_license is None
    assert resource.registered_payload_path is None
    assert resource.registered_payload_role is None
    assert resource.registered_payload_sha256 is None
    assert resource.registered_payload_bytes is None
    assert not resource.is_manifest_verified_trusted
    assert not resource.is_biological_reference_trusted


def test_unknown_registration_fails_before_reading_manifest(tmp_path: Path) -> None:
    with pytest.raises(ResourceIntegrityError) as error:
        load_receiver_autonomous_program_resource(
            tmp_path,
            manifest_path=tmp_path / "missing.json",
            registration_id="caller.self_registered.v1",
        )
    assert error.value.details.code == "unregistered_receiver_autonomous_resource"


def test_registered_manifest_digest_tampering_is_a_hard_error(
    tmp_path: Path,
) -> None:
    root = _copied_fixture(tmp_path)
    _rewrite_manifest(root, citation="Unreviewed replacement citation.")

    with pytest.raises(ResourceIntegrityError) as error:
        _load(root)
    assert error.value.details.code == "resource_manifest_digest_mismatch"


def test_registered_license_is_enforced_after_digest_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copied_fixture(tmp_path)
    digest = _rewrite_manifest(root, license="LicenseRef-unreviewed")
    _patch_registration_digest(monkeypatch, digest)

    with pytest.raises(ResourceIntegrityError) as error:
        _load(root)
    assert error.value.details.code == "resource_metadata_mismatch"
    assert error.value.details.field == "license"


@pytest.mark.parametrize(
    ("manifest_change", "registration_change", "field"),
    [
        ({"resource_id": "other"}, {}, "resource_id"),
        ({"version": "2.0.0"}, {}, "version"),
        ({"species": "mouse"}, {}, "species"),
        ({"gene_namespace": "MGI symbol"}, {}, "gene_namespace"),
        ({"adapter_version": "generic-tsv-v1"}, {}, "adapter_version"),
    ],
)
def test_all_registered_manifest_metadata_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_change: dict[str, str],
    registration_change: dict[str, object],
    field: str,
) -> None:
    root = _copied_fixture(tmp_path)
    digest = _rewrite_manifest(root, **manifest_change)
    _patch_registration_digest(monkeypatch, digest, **registration_change)

    with pytest.raises(ResourceIntegrityError) as error:
        _load(root)
    assert error.value.details.code == "resource_metadata_mismatch"
    assert error.value.details.field == field


def test_registered_payload_content_tampering_is_a_hard_error(tmp_path: Path) -> None:
    root = _copied_fixture(tmp_path)
    (root / "programs.tsv").write_text(
        "feature_id\tgeneric_immediate_early\nDUSP1\t9\nFOS\t1\nJUN\t1\n",
        encoding="utf-8",
    )

    with pytest.raises(ResourceIntegrityError) as error:
        _load(root)
    assert error.value.details.code == "resource_checksum_mismatch"


@pytest.mark.parametrize(
    "registration_change",
    [
        {"payload_path": "other.tsv"},
        {"payload_role": "other_role"},
        {"payload_sha256": "0" * 64},
        {"payload_bytes": 999},
    ],
)
def test_registered_payload_record_is_enforced_independently(
    monkeypatch: pytest.MonkeyPatch,
    registration_change: dict[str, object],
) -> None:
    _patch_registration_digest(
        monkeypatch,
        _MANIFEST_DIGEST,
        **registration_change,
    )

    with pytest.raises(ResourceIntegrityError) as error:
        _load(_FIXTURE_ROOT)
    assert error.value.details.code == "resource_registration_payload_mismatch"


def test_registered_payload_symlink_is_rejected_even_with_identical_bytes(
    tmp_path: Path,
) -> None:
    root = _copied_fixture(tmp_path)
    payload = root / "programs.tsv"
    external = tmp_path / "outside.tsv"
    external.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(external)

    with pytest.raises(ResourceIntegrityError) as error:
        _load(root)
    assert error.value.details.code == "resource_payload_symlink"


def test_exact_bytes_are_rechecked_after_manifest_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copied_fixture(tmp_path)
    payload = root / "programs.tsv"
    original_verify = ResourceManifest.verify

    def verify_then_change(
        manifest: ResourceManifest,
        database_root: str | Path,
        *,
        paths: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        result = original_verify(manifest, database_root, paths=paths)
        payload.write_text(
            "feature_id\tgeneric_immediate_early\nDUSP1\t9\nFOS\t1\nJUN\t1\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(ResourceManifest, "verify", verify_then_change)

    with pytest.raises(ResourceIntegrityError) as error:
        _load(root)
    assert error.value.details.code == "resource_checksum_mismatch"


def test_private_factory_cannot_validate_an_unregistered_matrix() -> None:
    registration = require_receiver_autonomous_program_registration(_REGISTRATION_ID)
    forged = autonomous_module._build_receiver_autonomous_program_resource(
        np.asarray([[99.0], [1.0], [1.0]]),
        feature_ids=("DUSP1", "FOS", "JUN"),
        program_ids=("generic_immediate_early",),
        resource_id=registration.resource_id,
        version=registration.version,
        manifest_digest=registration.manifest_digest,
        species=registration.species,
        gene_namespace=registration.gene_namespace,
        verification_status="manifest_verified_static_trusted_v1",
        registration_id=registration.registration_id,
        review_scope=registration.review_scope,
        expected_license=registration.expected_license,
        registered_payload_path=registration.payload_path,
        registered_payload_role=registration.payload_role,
        registered_payload_sha256=registration.payload_sha256,
        registered_payload_bytes=registration.payload_bytes,
        trusted_loader_token=autonomous_module._TRUSTED_LOADER_TOKEN,
    )

    with pytest.raises(ContractError) as error:
        _ = forged.is_manifest_verified_trusted
    assert (
        error.value.details.code
        == "receiver_autonomous_program_integrity_violation"
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("registration_id", "other.registration"),
        ("review_scope", "biological_reference"),
        ("expected_license", "LicenseRef-other"),
        ("registered_payload_path", "other.tsv"),
        ("registered_payload_role", "other_role"),
        ("registered_payload_sha256", "0" * 64),
        ("registered_payload_bytes", 999),
        ("matrix_digest", "0" * 64),
    ],
)
def test_verified_artifact_detects_registration_evidence_tampering(
    field: str,
    replacement: object,
) -> None:
    resource = _load(_FIXTURE_ROOT)
    object.__setattr__(resource, field, replacement)

    with pytest.raises(ContractError) as error:
        resource.to_dict()
    assert (
        error.value.details.code
        == "receiver_autonomous_program_integrity_violation"
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"gene\tstress\nG1\t1\n",
        b"feature_id\tstress\nG1\tnan\n",
        b"feature_id\tstress\nG1\t1\nG1\t2\n",
        b"feature_id\tstress\nG1\t1\t2\n",
        b"feature_id\tstress\n",
    ],
)
def test_static_tsv_parser_rejects_invalid_schema(payload: bytes) -> None:
    with pytest.raises(ResourceIntegrityError) as error:
        autonomous_module._parse_receiver_autonomous_tsv(payload)
    assert error.value.details.code == "invalid_receiver_autonomous_program_payload"
