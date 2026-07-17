from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from tests.support.calibration import diagnostic_generator_manifest

from crychic.core import SeedLineage, canonical_json, stable_id
from crychic.inference import (
    G3PCalibrationCampaign,
    G3PCalibrationProtocol,
    G3PCandidateStatus,
    G3PGateStatus,
    G3PReplicatePredictions,
    G3PReplicateStatus,
    summarize_attested_g3p_calibration_campaign,
    summarize_g3p_calibration_campaign,
)
from crychic.inference.calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationGeneratorProfile,
    CalibrationReplayRequest,
    _freeze_calibration_replay_registry,
    _register_calibration_generator,
)
from crychic.results import (
    G3P_CALIBRATION_CANDIDATE_TABLE,
    G3P_CALIBRATION_REPLICATE_TABLE,
    G3P_CALIBRATION_SCENARIO_TABLE,
    G3PCalibrationResult,
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
    write_g3p_calibration_result,
)

ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = "g3p_calibration_manifest.json"
_STATUS = "_status.json"


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def campaign() -> G3PCalibrationCampaign:
    protocol = G3PCalibrationProtocol(
        campaign_name="three-replicate-persistence-test",
        generator_manifest=diagnostic_generator_manifest(
            CalibrationCampaignKind.G3_PROBABILITY,
            "g3p-test-generator-v1",
        ),
        active_null_spec_id="active-null-spec-test-v1",
        active_probability_spec_id="active-probability-spec-test-v1",
        score_spec_id="score-spec-test-v4",
        candidate_universe_policy_id="candidate-universe-policy-test-v1",
        estimator_id="bum-test-v1",
        stratum_policy_id="receiver-test-v1",
        score_version="global-score-test-v4",
        seed_lineage=SeedLineage(31).derive("g3p-persistence-test"),
    )
    replicates = tuple(
        G3PReplicatePredictions(
            protocol_id=protocol.protocol_id,
            dependence_structure="independent_candidate_edges",
            non_null_prevalence=0.005,
            replicate_index=index,
            seed_lineage=protocol.replicate_seed_lineage(
                "independent_candidate_edges",
                0.005,
                index,
            ),
            source_active_null_id=f"active-null-{index}",
            source_candidate_universe_id=f"candidate-universe-test-{index}",
            source_probability_collection_id=f"probability-collection-{index}",
            candidate_edge_ids=("edge-c", "edge-a", "edge-b"),
            stratum_ids=("receiver-stratum",) * 3,
            truth_active=np.asarray([False, True, False], dtype=bool),
            candidate_probabilities=np.asarray(
                [0.0, 0.8, 0.2 + index * 0.01],
                dtype=float,
            ),
            candidate_statuses=(
                G3PCandidateStatus.STRUCTURAL_ZERO,
                G3PCandidateStatus.ELIGIBLE,
                G3PCandidateStatus.ELIGIBLE,
            ),
            candidate_reason_codes=("receptor_ineligible", None, None),
        )
        for index in range(3)
    )
    return summarize_g3p_calibration_campaign(protocol, replicates)


def _replay_persistence_replicate(
    protocol: object,
    request: CalibrationReplayRequest,
    config: Mapping[str, object],
) -> G3PReplicatePredictions:
    if not isinstance(protocol, G3PCalibrationProtocol) or config.get("rows") != 3:
        raise TypeError("invalid persistence replay request")
    index = request.replicate_index
    return G3PReplicatePredictions(
        protocol_id=protocol.protocol_id,
        dependence_structure=request.dependence_structure,
        non_null_prevalence=request.non_null_prevalence,
        replicate_index=index,
        seed_lineage=request.seed_lineage,
        source_active_null_id=f"attested-active-null-{index}",
        source_candidate_universe_id=(
            f"attested-candidate-universe-test-{index}"
        ),
        source_probability_collection_id=f"attested-probability-collection-{index}",
        candidate_edge_ids=("edge-c", "edge-a", "edge-b"),
        stratum_ids=("receiver-stratum",) * 3,
        truth_active=np.asarray([False, True, False], dtype=bool),
        candidate_probabilities=np.asarray(
            [0.0, 0.8, 0.2 + index * 0.01],
            dtype=float,
        ),
        candidate_statuses=(
            G3PCandidateStatus.STRUCTURAL_ZERO,
            G3PCandidateStatus.ELIGIBLE,
            G3PCandidateStatus.ELIGIBLE,
        ),
        candidate_reason_codes=("receptor_ineligible", None, None),
    )


def _attested_persistence_campaign():  # type: ignore[no-untyped-def]
    registration = _register_calibration_generator(
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        generator_kind="unit-g3p-persistence-replay",
        generator_version="1.0.0",
        code_version="unit-g3p-persistence-replay-v1",
        config={"rows": 3},
        source_paths=(Path(__file__),),
        replay=_replay_persistence_replicate,
        profile=CalibrationGeneratorProfile.RELEASE_APPROVED,
    )
    registry = _freeze_calibration_replay_registry((registration,))
    protocol = G3PCalibrationProtocol(
        campaign_name="attested-persistence-test",
        generator_manifest=registration.manifest,
        active_null_spec_id="active-null-spec-test-v1",
        active_probability_spec_id="active-probability-spec-test-v1",
        score_spec_id="score-spec-test-v4",
        candidate_universe_policy_id="candidate-universe-policy-test-v1",
        estimator_id="bum-test-v1",
        stratum_policy_id="receiver-test-v1",
        score_version="global-score-test-v4",
        seed_lineage=SeedLineage(47).derive("attested-g3p-persistence-test"),
    )
    ledgers = tuple(
        _replay_persistence_replicate(
            protocol,
            CalibrationReplayRequest(
                campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
                protocol_id=protocol.protocol_id,
                dependence_structure="independent_candidate_edges",
                non_null_prevalence=0.005,
                replicate_index=index,
                seed_lineage=protocol.replicate_seed_lineage(
                    "independent_candidate_edges", 0.005, index
                ),
            ),
            registration.config,
        )
        for index in range(3)
    )
    return (
        summarize_attested_g3p_calibration_campaign(
            protocol,
            ledgers,
            registry=registry,
            n_jobs=2,
        ),
        registry,
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def artifact(
    tmp_path_factory: pytest.TempPathFactory,
    campaign: G3PCalibrationCampaign,
) -> Path:
    destination = tmp_path_factory.mktemp("g3p-result") / "campaign"
    write_g3p_calibration_result(destination, campaign=campaign)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _rehash(destination: Path) -> None:
    manifest_path = destination / _MANIFEST
    status_path = destination / _STATUS
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_id"] = stable_id(
        "g3p_calibration_result",
        {key: value for key, value in manifest.items() if key != "artifact_id"},
        schema_version="1",
    )
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["artifact_id"] = manifest["artifact_id"]
    status_path.write_text(f"{canonical_json(status)}\n", encoding="utf-8")


def _replace_table(destination: Path, table_name: str, frame: pd.DataFrame) -> None:
    manifest_path = destination / _MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["tables"][table_name]
    path = destination / record["filename"]
    frame.to_parquet(path, index=False, engine="pyarrow")
    record["rows"] = len(frame)
    record["sha256"] = _sha256(path)
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    _rehash(destination)


def _copy_artifact(artifact: Path, tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(artifact, destination)
    return destination


def test_g3p_calibration_schemas_are_valid_and_independent() -> None:
    filenames = (
        "g3p_calibration_result.schema.json",
        "g3p_calibration_replicates.schema.json",
        "g3p_calibration_candidates.schema.json",
        "g3p_calibration_scenarios.schema.json",
    )
    for filename in filenames:
        value = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(value)


def test_three_replicate_campaign_round_trips_with_not_estimable_gate(
    artifact: Path,
    campaign: G3PCalibrationCampaign,
) -> None:
    result = G3PCalibrationResult.load(artifact)

    assert result.manifest["campaign_id"] == campaign.campaign_id
    assert result.manifest["evidence_id"] == campaign.evidence.evidence_id
    assert result.manifest["evidence"]["generator_verified"] is False
    assert result.manifest["calibration_gate"]["generator_verified"] is False
    assert result.calibration_gate.status is G3PGateStatus.NOT_ESTIMABLE
    assert result.calibration_gate.comm_probability_release_allowed is False
    assert result.calibration_gate.source_artifact_id == campaign.campaign_id
    assert len(result.read_replicates()) == 3
    assert len(result.read_candidates()) == 9
    assert len(result.read_scenarios()) == 12
    filtered = result.read_candidates(
        filters={"candidate_edge_id": "edge-a"},
        columns=["replicate_id", "candidate_edge_id", "candidate_probability"],
    )
    assert len(filtered) == 3
    assert set(filtered["candidate_probability"]) == {0.8}
    candidates = result.read_candidates()
    structural = candidates.loc[candidates["candidate_status"] == "structural_zero"]
    assert len(structural) == 3
    assert structural["candidate_probability"].eq(0.0).all()
    assert not structural["truth_active"].any()
    populated = result.read_scenarios(
        filters={
            "dependence_structure": "independent_candidate_edges",
            "non_null_prevalence": 0.005,
        }
    )
    assert populated["minimum_eligible_candidates_per_stratum"].tolist() == [2]
    strata = json.loads(str(populated.iloc[0]["strata_json"]))
    stratum_result_ids = json.loads(str(populated.iloc[0]["stratum_result_ids_json"]))
    assert len(strata) == 1
    assert strata[0]["stratum_id"] == "receiver-stratum"
    assert strata[0]["minimum_eligible_candidates"] == 2
    assert stratum_result_ids == [strata[0]["stratum_result_id"]]


def test_not_estimable_candidate_round_trips_and_keeps_gate_closed(
    tmp_path: Path,
) -> None:
    protocol = G3PCalibrationProtocol(
        campaign_name="candidate-ne-persistence-test",
        generator_manifest=diagnostic_generator_manifest(
            CalibrationCampaignKind.G3_PROBABILITY,
            "g3p-test-generator-v1",
        ),
        active_null_spec_id="active-null-spec-test-v1",
        active_probability_spec_id="active-probability-spec-test-v1",
        score_spec_id="score-spec-test-v4",
        candidate_universe_policy_id="candidate-universe-policy-test-v1",
        estimator_id="bum-test-v1",
        stratum_policy_id="receiver-test-v1",
        score_version="global-score-test-v4",
        seed_lineage=SeedLineage(47).derive("g3p-candidate-ne-test"),
    )
    replicate = G3PReplicatePredictions(
        protocol_id=protocol.protocol_id,
        dependence_structure="independent_candidate_edges",
        non_null_prevalence=0.005,
        replicate_index=0,
        seed_lineage=protocol.replicate_seed_lineage(
            "independent_candidate_edges", 0.005, 0
        ),
        source_active_null_id="active-null-ne-0",
        source_candidate_universe_id="candidate-universe-ne-v1",
        source_probability_collection_id=None,
        candidate_edge_ids=("edge-a", "edge-b"),
        stratum_ids=("receiver-stratum", "receiver-stratum"),
        truth_active=np.asarray([False, True], dtype=bool),
        candidate_probabilities=np.asarray([0.2, np.nan], dtype=float),
        candidate_statuses=(
            G3PCandidateStatus.ELIGIBLE,
            G3PCandidateStatus.NOT_ESTIMABLE,
        ),
        candidate_reason_codes=(None, "candidate_pipeline_not_estimable"),
        status=G3PReplicateStatus.NOT_ESTIMABLE,
        reason_code="candidate_pipeline_not_estimable",
    )
    campaign = summarize_g3p_calibration_campaign(protocol, (replicate,))
    result = write_g3p_calibration_result(tmp_path / "candidate-ne", campaign=campaign)

    assert result.calibration_gate.status is G3PGateStatus.NOT_ESTIMABLE
    candidates = result.read_candidates()
    assert candidates["candidate_status"].tolist() == [
        "eligible",
        "not_estimable",
    ]
    assert pd.isna(candidates.loc[1, "candidate_probability"])
    assert result.read_replicates()["status"].tolist() == ["not_estimable"]


def test_protocol_declaration_cannot_be_mutated_into_verified(
    campaign: G3PCalibrationCampaign,
) -> None:
    assert campaign.protocol.generator_release_verified is False
    with pytest.raises(AttributeError):
        object.__setattr__(
            campaign.protocol,
            "_generator_authorization_token",
            object(),
        )


def test_attested_campaign_requires_registry_and_round_trips_by_replay(
    tmp_path: Path,
) -> None:
    attested, registry = _attested_persistence_campaign()
    destination = tmp_path / "attested-campaign"

    assert attested.evidence.generator_verified is True
    assert attested.generator_attestation is not None
    with pytest.raises(ResultValidationError) as missing_write_registry:
        write_g3p_calibration_result(destination, campaign=attested)
    assert (
        missing_write_registry.value.details.code
        == "g3p_calibration_attestation_unavailable"
    )

    unrelated_registration = _register_calibration_generator(
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        generator_kind="unrelated-unit-g3p-persistence-replay",
        generator_version="1.0.0",
        code_version="unrelated-unit-g3p-persistence-replay-v1",
        config={"rows": 3},
        source_paths=(Path(__file__),),
        replay=_replay_persistence_replicate,
        profile=CalibrationGeneratorProfile.RELEASE_APPROVED,
    )
    unrelated_registry = _freeze_calibration_replay_registry(
        (unrelated_registration,)
    )
    failed_destination = tmp_path / "wrong-registry"
    with pytest.raises(ResultValidationError) as wrong_registry:
        write_g3p_calibration_result(
            failed_destination,
            campaign=attested,
            replay_registry=unrelated_registry,
        )
    assert (
        wrong_registry.value.details.code
        == "g3p_calibration_prewrite_replay_failed"
    )
    assert not failed_destination.exists()

    result = write_g3p_calibration_result(
        destination,
        campaign=attested,
        replay_registry=registry,
        n_jobs=2,
    )
    assert result.manifest["generator_attestation"] == (
        attested.generator_attestation.to_dict()
    )
    assert result.manifest["evidence"]["generator_verified"] is True
    with pytest.raises(ResultValidationError) as missing_load_registry:
        G3PCalibrationResult.load(destination)
    assert (
        missing_load_registry.value.details.code
        == "g3p_calibration_attestation_unavailable"
    )
    reloaded = G3PCalibrationResult.load(
        destination,
        replay_registry=registry,
        n_jobs=2,
    )
    assert reloaded.manifest["artifact_id"] == result.manifest["artifact_id"]


def test_load_rejects_corrupted_table_bytes(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "corrupted")
    manifest = json.loads((destination / _MANIFEST).read_text(encoding="utf-8"))
    path = destination / manifest["tables"][G3P_CALIBRATION_CANDIDATE_TABLE]["filename"]
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises(ResultValidationError) as raised:
        G3PCalibrationResult.load(destination)
    assert raised.value.details.code == "g3p_calibration_digest_mismatch"


def test_load_rejects_missing_candidate_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "missing-candidate")
    candidates = pd.read_parquet(destination / "g3p_calibration_candidates.parquet")
    _replace_table(
        destination,
        G3P_CALIBRATION_CANDIDATE_TABLE,
        candidates.iloc[:-1].copy(),
    )

    with pytest.raises(ResultValidationError) as raised:
        G3PCalibrationResult.load(destination)
    assert raised.value.details.code == "g3p_calibration_candidate_coverage_mismatch"


def test_load_rejects_duplicate_candidate_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "duplicate-candidate")
    candidates = pd.read_parquet(destination / "g3p_calibration_candidates.parquet")
    forged = pd.concat([candidates, candidates.iloc[[0]]], ignore_index=True)
    _replace_table(destination, G3P_CALIBRATION_CANDIDATE_TABLE, forged)

    with pytest.raises(ResultValidationError) as raised:
        G3PCalibrationResult.load(destination)
    assert raised.value.details.code == "duplicate_g3p_calibration_primary_key"


def test_load_rejects_forged_candidate_row_id_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "forged-row-id")
    candidates = pd.read_parquet(destination / "g3p_calibration_candidates.parquet")
    candidates.loc[0, "candidate_row_id"] = "forged-candidate-row"
    _replace_table(destination, G3P_CALIBRATION_CANDIDATE_TABLE, candidates)

    with pytest.raises(ResultValidationError) as raised:
        G3PCalibrationResult.load(destination)
    assert raised.value.details.code == "g3p_calibration_reconstruction_mismatch"


def test_load_rejects_forged_gate_registry_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "forged-gate")
    manifest_path = destination / _MANIFEST
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration_gate"]["reason_code"] = "forged-reason"
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    _rehash(destination)

    with pytest.raises(ResultValidationError) as raised:
        G3PCalibrationResult.load(destination)
    assert raised.value.details.code == "g3p_calibration_linkage_mismatch"


def test_load_rejects_forged_stratum_registry_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "forged-stratum")
    scenarios = pd.read_parquet(destination / "g3p_calibration_scenarios.parquet")
    populated = scenarios["strata_json"].map(lambda value: value != "[]")
    index = int(scenarios.index[populated][0])
    strata = json.loads(str(scenarios.loc[index, "strata_json"]))
    strata[0]["minimum_eligible_candidates"] = 999
    scenarios.loc[index, "strata_json"] = canonical_json(strata)
    _replace_table(destination, G3P_CALIBRATION_SCENARIO_TABLE, scenarios)

    with pytest.raises(ResultValidationError) as raised:
        G3PCalibrationResult.load(destination)
    assert raised.value.details.code == "g3p_calibration_reconstruction_mismatch"


def test_load_rejects_noncanonical_stratum_json_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "noncanonical-stratum-json")
    scenarios = pd.read_parquet(destination / "g3p_calibration_scenarios.parquet")
    populated = scenarios["strata_json"].map(lambda value: value != "[]")
    index = int(scenarios.index[populated][0])
    scenarios.loc[index, "strata_json"] = "[ ]"
    _replace_table(destination, G3P_CALIBRATION_SCENARIO_TABLE, scenarios)

    with pytest.raises(ResultValidationError) as raised:
        G3PCalibrationResult.load(destination)
    assert raised.value.details.code == "invalid_g3p_calibration_scenario_registry"


def test_interrupted_write_is_atomic_and_loads_only_as_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    campaign: G3PCalibrationCampaign,
) -> None:
    destination = tmp_path / "interrupted"

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(ResultWriteError) as raised:
        write_g3p_calibration_result(destination, campaign=campaign)
    assert raised.value.details.code == "g3p_calibration_write_failed"
    assert destination.exists()
    with pytest.raises(IncompleteResultError) as incomplete:
        G3PCalibrationResult.load(destination)
    assert incomplete.value.details.code == "incomplete_g3p_calibration_result"


def test_query_rejects_unknown_columns(
    artifact: Path,
) -> None:
    result = G3PCalibrationResult.load(artifact)
    with pytest.raises(KeyError, match="not_a_column"):
        result.read_replicates(columns=["not_a_column"])
    with pytest.raises(KeyError, match="not_a_filter"):
        result.read_scenarios(filters={"not_a_filter": "value"})


def test_public_table_constants_match_manifest(
    artifact: Path,
) -> None:
    result = G3PCalibrationResult.load(artifact)
    assert set(result.manifest["tables"]) == {
        G3P_CALIBRATION_REPLICATE_TABLE,
        G3P_CALIBRATION_CANDIDATE_TABLE,
        G3P_CALIBRATION_SCENARIO_TABLE,
    }
