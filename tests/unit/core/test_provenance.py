from types import MappingProxyType

import pytest

from crychic.core import ContractError, CrychicConfig, RunProvenance, SeedLineage


def _provenance() -> RunProvenance:
    config = CrychicConfig(context_keys=["condition"])
    return RunProvenance(
        package_version="0.0.0",
        git_commit="ccc6795",
        git_dirty=False,
        config_digest=config.digest,
        input_digest="a" * 64,
        resource_digests={"cellphonedb-v5": "b" * 64},
        result_schema_version=None,
        seed_lineage=SeedLineage(config.random_seed).derive("run"),
        created_at="2026-07-12T00:00:00Z",
    )


def test_provenance_round_trip_and_canonical_digest() -> None:
    provenance = _provenance()
    restored = RunProvenance.from_dict(provenance.to_dict())

    assert restored == provenance
    assert restored.to_json() == provenance.to_json()
    assert restored.digest == provenance.digest
    assert isinstance(restored.resource_digests, MappingProxyType)


def test_provenance_resource_mapping_is_immutable() -> None:
    provenance = _provenance()

    with pytest.raises(TypeError):
        provenance.resource_digests["other"] = "c" * 64  # type: ignore[index]


def test_provenance_rejects_malformed_digest() -> None:
    with pytest.raises(ContractError, match="SHA-256"):
        RunProvenance(
            package_version="0.0.0",
            config_digest="short",
            seed_lineage=SeedLineage(1),
        )


def test_provenance_rejects_unversioned_result_schema() -> None:
    with pytest.raises(ContractError, match="Schema version"):
        RunProvenance(
            package_version="0.0.0",
            config_digest="a" * 64,
            result_schema_version="v1",
            seed_lineage=SeedLineage(1),
        )
