from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from tests.unit.inference.test_g3f_calibration import (
    _protocol,
    _replay_noninferiority_case,
    _replicate,
    _replicates,
)

from crychic.core import canonical_json, stable_id
from crychic.inference.calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationGeneratorProfile,
    CalibrationReplayRequest,
    _freeze_calibration_replay_registry,
    _register_calibration_generator,
)
from crychic.inference.g3f_calibration import (
    G3FrequencyCalibrationCampaign,
    G3FrequencyCalibrationProtocol,
    G3FrequencyDependenceStructure,
    G3FrequencyGateStatus,
    G3FrequencyReplicateStatus,
    summarize_attested_g3_frequency_calibration_campaign,
    summarize_g3_frequency_calibration_campaign,
)
from crychic.results import (
    G3F_CALIBRATION_HYPOTHESIS_TABLE,
    G3F_CALIBRATION_PERMUTATION_TABLE,
    G3F_CALIBRATION_REPLICATE_TABLE,
    G3F_CALIBRATION_SCENARIO_TABLE,
    G3FCalibrationResult,
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
    write_g3f_calibration_result,
)

ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = "g3f_calibration_manifest.json"
_STATUS = "_status.json"
_PREVALENCES = (0.005, 0.01, 0.05, 0.10)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def campaign() -> G3FrequencyCalibrationCampaign:
    return summarize_g3_frequency_calibration_campaign(_protocol(), _replicates())


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def artifact(
    tmp_path_factory: pytest.TempPathFactory,
    campaign: G3FrequencyCalibrationCampaign,
) -> Path:
    destination = tmp_path_factory.mktemp("g3f-result") / "campaign"
    write_g3f_calibration_result(destination, campaign=campaign)
    return destination


def _attested_campaign():  # type: ignore[no-untyped-def]
    registration = _register_calibration_generator(
        campaign_kind=CalibrationCampaignKind.G3_FREQUENCY,
        generator_kind="unit-g3f-persistence-replay",
        generator_version="1.0.0",
        code_version="unit-g3f-persistence-replay-v1",
        config={"mode": "width"},
        source_paths=(Path(__file__),),
        replay=_replay_noninferiority_case,
        profile=CalibrationGeneratorProfile.RELEASE_APPROVED,
    )
    base = _protocol()
    protocol = G3FrequencyCalibrationProtocol(
        campaign_name="g3f-attested-persistence-campaign",
        generator_manifest=registration.manifest,
        noninferiority_baseline_id=base.noninferiority_baseline_id,
        hypothesis_universe=base.hypothesis_universe,
        hierarchical_procedure=base.hierarchical_procedure,
        seed_lineage=base.seed_lineage,
    )
    registry = _freeze_calibration_replay_registry((registration,))
    ledgers = tuple(
        _replay_noninferiority_case(
            protocol,
            CalibrationReplayRequest(
                campaign_kind=CalibrationCampaignKind.G3_FREQUENCY,
                protocol_id=protocol.protocol_id,
                dependence_structure=dependence.value,
                non_null_prevalence=prevalence,
                replicate_index=index,
                seed_lineage=protocol.replicate_seed_lineage(
                    dependence, prevalence, index
                ),
            ),
            registration.config,
        )
        for dependence in G3FrequencyDependenceStructure
        for prevalence in (None, *_PREVALENCES)
        for index in range(2)
    )
    return (
        summarize_attested_g3_frequency_calibration_campaign(
            protocol,
            ledgers,
            registry=registry,
            n_jobs=2,
        ),
        registry,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _rehash(destination: Path) -> None:
    manifest_path = destination / _MANIFEST
    status_path = destination / _STATUS
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_id"] = stable_id(
        "g3f_calibration_result",
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


def test_g3f_calibration_schemas_are_valid() -> None:
    filenames = (
        "g3f_calibration_result.schema.json",
        "g3f_calibration_replicates.schema.json",
        "g3f_calibration_hypotheses.schema.json",
        "g3f_calibration_scenarios.schema.json",
        "g3f_calibration_permutation_diagnostics.schema.json",
    )
    for filename in filenames:
        value = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(value)


def test_diagnostic_campaign_round_trips_from_authoritative_ledgers(
    artifact: Path,
    campaign: G3FrequencyCalibrationCampaign,
) -> None:
    result = G3FCalibrationResult.load(artifact)
    result_schema = json.loads(
        (ROOT / "schemas" / "g3f_calibration_result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(result_schema).validate(result.manifest)

    assert result.manifest["campaign_id"] == campaign.campaign_id
    assert result.manifest["evidence_id"] == campaign.evidence.evidence_id
    assert result.manifest["hypothesis_universe"] == (
        campaign.protocol.hypothesis_universe.to_dict()
    )
    assert result.manifest["hierarchical_procedure"] == (
        campaign.protocol.hierarchical_procedure.to_dict()
    )
    assert result.calibration_gate.status is G3FrequencyGateStatus.NOT_ESTIMABLE
    assert result.calibration_gate.formal_release_allowed is False
    assert len(result.read_replicates()) == 30
    assert len(result.read_hypotheses()) == 6_000
    assert len(result.read_scenarios()) == 81
    assert len(result.read_permutation_diagnostics()) == 3
    filtered = result.read_hypotheses(
        filters={"hypothesis_role": "primary"},
        columns=["replicate_id", "hypothesis_id", "truth_non_null"],
    )
    assert len(filtered) == 3_000
    assert set(result.manifest["tables"]) == {
        G3F_CALIBRATION_REPLICATE_TABLE,
        G3F_CALIBRATION_HYPOTHESIS_TABLE,
        G3F_CALIBRATION_SCENARIO_TABLE,
        G3F_CALIBRATION_PERMUTATION_TABLE,
    }


def test_attested_campaign_requires_registry_and_replays_on_load(
    tmp_path: Path,
) -> None:
    attested, registry = _attested_campaign()
    destination = tmp_path / "attested"

    with pytest.raises(ResultValidationError) as missing_write:
        write_g3f_calibration_result(destination, campaign=attested)
    assert missing_write.value.details.code == "g3f_calibration_attestation_unavailable"
    wrong_registration = _register_calibration_generator(
        campaign_kind=CalibrationCampaignKind.G3_FREQUENCY,
        generator_kind="unit-g3f-unrelated-persistence-replay",
        generator_version="1.0.0",
        code_version="unit-g3f-persistence-replay-v1",
        config={"mode": "width"},
        source_paths=(Path(__file__),),
        replay=_replay_noninferiority_case,
        profile=CalibrationGeneratorProfile.RELEASE_APPROVED,
    )
    wrong_registry = _freeze_calibration_replay_registry((wrong_registration,))
    with pytest.raises(ResultValidationError) as wrong_write:
        write_g3f_calibration_result(
            destination,
            campaign=attested,
            replay_registry=wrong_registry,
        )
    assert (
        wrong_write.value.details.code
        == "g3f_calibration_attestation_registry_mismatch"
    )
    assert not destination.exists()
    with pytest.raises(ValueError, match="n_jobs"):
        write_g3f_calibration_result(
            destination,
            campaign=attested,
            replay_registry=registry,
            n_jobs=0,
        )
    assert not destination.exists()

    drifted_campaign, drifted_registry = _attested_campaign()
    drifted_registration = drifted_registry.resolve(
        drifted_campaign.protocol.generator_id
    )

    def broken_replay(*args: object, **kwargs: object) -> object:
        raise ValueError("drifted replay implementation")

    object.__setattr__(drifted_registration, "replay", broken_replay)
    with pytest.raises(ResultValidationError) as drifted_write:
        write_g3f_calibration_result(
            destination,
            campaign=drifted_campaign,
            replay_registry=drifted_registry,
        )
    assert (
        drifted_write.value.details.code
        == "g3f_calibration_prewrite_replay_failed"
    )
    assert not destination.exists()

    written = write_g3f_calibration_result(
        destination,
        campaign=attested,
        replay_registry=registry,
        n_jobs=2,
    )
    assert written.manifest["generator_attestation"] == (
        attested.generator_attestation.to_dict()
    )
    with pytest.raises(ResultValidationError) as missing_load:
        G3FCalibrationResult.load(destination)
    assert missing_load.value.details.code == "g3f_calibration_attestation_unavailable"
    reloaded = G3FCalibrationResult.load(
        destination,
        replay_registry=registry,
        n_jobs=2,
    )
    assert reloaded.manifest["artifact_id"] == written.manifest["artifact_id"]


def test_nonobserved_replicate_round_trips_with_null_interval_ledgers(
    tmp_path: Path,
) -> None:
    ledgers = list(_replicates())
    target = ledgers[0]
    ledgers[0] = _replicate(
        target.dependence_structure,
        target.non_null_prevalence,
        target.replicate_index,
        status=G3FrequencyReplicateStatus.NOT_ESTIMABLE,
    )
    campaign = summarize_g3_frequency_calibration_campaign(_protocol(), ledgers)
    result = write_g3f_calibration_result(
        tmp_path / "nonobserved",
        campaign=campaign,
    )

    row = result.read_replicates(
        filters={"replicate_id": ledgers[0].replicate_id}
    ).iloc[0]
    hypotheses = result.read_hypotheses(
        filters={"replicate_id": ledgers[0].replicate_id}
    )
    assert row["status"] == "not_estimable"
    assert row["reason_code"] == "synthetic_replicate_not_observed"
    assert hypotheses["candidate_ci_interval_width"].isna().all()
    assert hypotheses["baseline_ci_interval_width"].isna().all()


def test_single_replicate_dkw_bound_above_one_round_trips(tmp_path: Path) -> None:
    ledgers = tuple(item for item in _replicates() if item.replicate_index == 0)
    campaign = summarize_g3_frequency_calibration_campaign(_protocol(), ledgers)
    result = write_g3f_calibration_result(tmp_path / "single", campaign=campaign)

    diagnostics = result.read_permutation_diagnostics()
    assert diagnostics["dkw_limit"].gt(1.0).all()


def test_empty_campaign_round_trips_as_not_estimable(tmp_path: Path) -> None:
    campaign = summarize_g3_frequency_calibration_campaign(_protocol(), ())
    result = write_g3f_calibration_result(tmp_path / "empty", campaign=campaign)

    assert result.calibration_gate.status is G3FrequencyGateStatus.NOT_ESTIMABLE
    assert result.calibration_gate.reason_code == (
        "g3_frequency_calibration_scenario_not_estimable"
    )
    assert result.calibration_gate.permutation_ks_distance == 1.0
    assert result.read_replicates().empty
    assert result.read_hypotheses().empty


def test_load_rejects_corrupted_table_bytes(artifact: Path, tmp_path: Path) -> None:
    destination = _copy_artifact(artifact, tmp_path, "corrupted")
    path = destination / "g3f_calibration_hypotheses.parquet"
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises(ResultValidationError) as raised:
        G3FCalibrationResult.load(destination)
    assert raised.value.details.code == "g3f_calibration_digest_mismatch"


def test_load_rejects_missing_hypothesis_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "missing-hypothesis")
    frame = pd.read_parquet(destination / "g3f_calibration_hypotheses.parquet")
    _replace_table(destination, G3F_CALIBRATION_HYPOTHESIS_TABLE, frame.iloc[:-1])

    with pytest.raises(ResultValidationError) as raised:
        G3FCalibrationResult.load(destination)
    assert raised.value.details.code == "g3f_calibration_hypothesis_coverage_mismatch"


def test_load_rejects_duplicate_hypothesis_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "duplicate-hypothesis")
    frame = pd.read_parquet(destination / "g3f_calibration_hypotheses.parquet")
    forged = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    _replace_table(destination, G3F_CALIBRATION_HYPOTHESIS_TABLE, forged)

    with pytest.raises(ResultValidationError) as raised:
        G3FCalibrationResult.load(destination)
    assert raised.value.details.code == "duplicate_g3f_calibration_primary_key"


def test_load_rejects_raw_ledger_tampering_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "forged-raw-ledger")
    frame = pd.read_parquet(destination / "g3f_calibration_hypotheses.parquet")
    frame.loc[0, "ci_contains_truth"] = False
    _replace_table(destination, G3F_CALIBRATION_HYPOTHESIS_TABLE, frame)

    with pytest.raises(ResultValidationError) as raised:
        G3FCalibrationResult.load(destination)
    assert raised.value.details.code == "g3f_calibration_replicate_identity_mismatch"


@pytest.mark.parametrize(
    ("table_name", "filename", "column"),
    [
        (
            G3F_CALIBRATION_SCENARIO_TABLE,
            "g3f_calibration_scenarios.parquet",
            "point_estimate",
        ),
        (
            G3F_CALIBRATION_PERMUTATION_TABLE,
            "g3f_calibration_permutation_diagnostics.parquet",
            "ks_distance",
        ),
    ],
)
def test_load_rejects_forged_derived_tables_after_full_rehash(
    artifact: Path,
    tmp_path: Path,
    table_name: str,
    filename: str,
    column: str,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, f"forged-{table_name}")
    frame = pd.read_parquet(destination / filename)
    observed = frame[column].notna()
    index = int(frame.index[observed][0])
    frame.loc[index, column] = float(frame.loc[index, column]) + 0.001
    _replace_table(destination, table_name, frame)

    with pytest.raises(ResultValidationError) as raised:
        G3FCalibrationResult.load(destination)
    assert raised.value.details.code == "g3f_calibration_reconstruction_mismatch"


def test_load_rejects_forged_gate_after_manifest_rehash(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "forged-gate")
    path = destination / _MANIFEST
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    manifest["calibration_gate"]["reason_code"] = "forged-reason"
    path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    _rehash(destination)

    with pytest.raises(ResultValidationError) as raised:
        G3FCalibrationResult.load(destination)
    assert raised.value.details.code == "g3f_calibration_linkage_mismatch"


def test_load_rejects_noncanonical_derived_json(
    artifact: Path,
    tmp_path: Path,
) -> None:
    destination = _copy_artifact(artifact, tmp_path, "noncanonical-json")
    frame = pd.read_parquet(destination / "g3f_calibration_scenarios.parquet")
    frame.loc[0, "replicate_ids_json"] = "[ ]"
    _replace_table(destination, G3F_CALIBRATION_SCENARIO_TABLE, frame)

    with pytest.raises(ResultValidationError) as raised:
        G3FCalibrationResult.load(destination)
    assert raised.value.details.code == "invalid_g3f_calibration_derived_registry"


def test_interrupted_write_is_marked_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    campaign: G3FrequencyCalibrationCampaign,
) -> None:
    destination = tmp_path / "interrupted"

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(ResultWriteError) as raised:
        write_g3f_calibration_result(destination, campaign=campaign)
    assert raised.value.details.code == "g3f_calibration_write_failed"
    with pytest.raises(IncompleteResultError) as incomplete:
        G3FCalibrationResult.load(destination)
    assert incomplete.value.details.code == "incomplete_g3f_calibration_result"


def test_query_rejects_unknown_columns(artifact: Path) -> None:
    result = G3FCalibrationResult.load(artifact)
    with pytest.raises(KeyError, match="not_a_column"):
        result.read_replicates(columns=["not_a_column"])
    with pytest.raises(KeyError, match="not_a_filter"):
        result.read_scenarios(filters={"not_a_filter": "value"})
