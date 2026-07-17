from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from tests.support.calibration import diagnostic_generator_manifest
from tests.support.g3p import (
    TEST_SCORE_SPEC_ID,
    compact_g3p_campaign,
    release_test_replay_registry,
)

import crychic.inference.g3p_calibration as _g3p_calibration
from crychic.core import SeedLineage, canonical_json, stable_id
from crychic.inference import (
    ActiveEdgeScoreStatus,
    ActiveProbabilityCollection,
    ActiveProbabilitySpec,
    G3PCalibrationGate,
    G3PCalibrationProtocol,
    NullActiveEdgeScoreRecord,
    NullScoreDistribution,
    PointActiveEdgeScoreRecord,
    build_g3p_calibration_gate,
    estimate_active_probabilities,
    summarize_g3p_calibration_campaign,
)
from crychic.inference.calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationReplayRegistry,
)
from crychic.resampling import ActiveNullSpec
from crychic.results._schema import validate_table
from crychic.results.active_probability import (
    ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE,
    ACTIVE_PROBABILITY_NULL_SOURCE_TABLE,
    ACTIVE_PROBABILITY_TABLE,
    ActiveProbabilityResult,
    _sha256_file,
    write_active_probability_result,
)
from crychic.results.errors import ResultValidationError
from crychic.results.g3p_calibration import (
    G3PCalibrationResult,
    write_g3p_calibration_result,
)
from crychic.scoring import (
    ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
    ActiveEdgeCandidate,
    FrozenActiveEdgeUniverse,
    freeze_active_edge_universe,
)
from crychic.workflow.crossfit_persistence import (
    CROSSFIT_COMPONENT_TABLE,
    _table_columns,
)
from crychic.workflow.crossfit_persistence import (
    _validate_table as validate_crossfit_table,
)

ROOT = Path(__file__).parents[3]
_SCORE_VERSION = "receiver_gain_percentile_mechanistic_conserved_sender_v4"


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def calibration_parent(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[G3PCalibrationResult, CalibrationReplayRegistry]]:
    patch = pytest.MonkeyPatch()
    patch.setattr(_g3p_calibration, "_MIN_CALIBRATION_REPLICATES", 200)
    patch.setattr(_g3p_calibration, "_MIN_CANDIDATES_PER_STRATUM", 20)
    patch.setattr(_g3p_calibration, "_ECE_BOOTSTRAP_RESAMPLES", 200)
    patch.setattr(_g3p_calibration, "_BRIER_IMPROVEMENT_MINIMUM", 0.02)
    patch.setattr(_g3p_calibration, "_ECE_MAXIMUM", 0.07)
    patch.setattr(
        _g3p_calibration,
        "_CALIBRATION_IN_THE_LARGE_ABS_MAXIMUM",
        0.12,
    )
    spec = ActiveProbabilitySpec()
    registry = release_test_replay_registry()
    campaign = compact_g3p_campaign(
        active_probability_spec_id=spec.spec_id,
        estimator_id=spec.estimator_id,
        stratum_policy_id=spec.stratum_policy_id,
        score_spec_id=TEST_SCORE_SPEC_ID,
        score_version=_SCORE_VERSION,
    )
    destination = tmp_path_factory.mktemp("active-probability-g3p") / "calibration"
    result = write_g3p_calibration_result(
        destination,
        campaign=campaign,
        replay_registry=registry,
        n_jobs=2,
    )
    assert result.calibration_gate.comm_probability_release_allowed
    yield result, registry
    patch.undo()


def _not_estimable_gate(
    distribution: NullScoreDistribution,
    spec: ActiveProbabilitySpec,
) -> G3PCalibrationGate:
    protocol = G3PCalibrationProtocol(
        campaign_name="empty-result-writer-test-campaign",
        generator_manifest=diagnostic_generator_manifest(
            CalibrationCampaignKind.G3_PROBABILITY,
            "empty-test-generator-v1",
        ),
        active_null_spec_id=distribution.active_null_spec_id,
        active_probability_spec_id=spec.spec_id,
        score_spec_id=distribution.score_spec_id,
        candidate_universe_policy_id=distribution.candidate_universe_policy_id,
        estimator_id=spec.estimator_id,
        stratum_policy_id=spec.stratum_policy_id,
        score_version=distribution.score_version,
        seed_lineage=SeedLineage(7).derive("empty-g3p-test"),
    )
    campaign = summarize_g3p_calibration_campaign(protocol, ())
    return build_g3p_calibration_gate(campaign.evidence)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def probability_case(
    calibration_parent: tuple[G3PCalibrationResult, CalibrationReplayRegistry],
) -> tuple[
    FrozenActiveEdgeUniverse,
    NullScoreDistribution,
    ActiveProbabilitySpec,
    G3PCalibrationGate,
    ActiveProbabilityCollection,
    ActiveProbabilityCollection,
]:
    spec = ActiveProbabilitySpec()
    parent, _ = calibration_parent
    gate = parent.calibration_gate
    assert gate.active_probability_spec_id == spec.spec_id
    candidates = tuple(
        ActiveEdgeCandidate(
            contrast_id="treated-vs-control",
            context_id="treated",
            sender="Sender",
            receiver="Receiver",
            interaction_id=f"lr-{index:03d}",
            driver_id=f"driver-{index:03d}",
            mode="state",
        )
        for index in range(201)
    )
    universe = freeze_active_edge_universe(
        candidates,
        universe_name="active-probability-persistence-test",
        contrast_id="treated-vs-control",
        score_version=_SCORE_VERSION,
    )
    plans = tuple(f"plan-{index:03d}" for index in range(200))
    uniform = (np.arange(170, dtype=float) + 0.5) / 170
    alternative = ((np.arange(30, dtype=float) + 0.5) / 30) ** (1.0 / 0.3)
    target_p = np.concatenate([uniform, alternative])
    extreme_counts = np.clip(np.rint(target_p * 201 - 1), 0, 200).astype(int)
    point_scores = (200 - extreme_counts) / 200
    points = tuple(
        PointActiveEdgeScoreRecord(
            candidate_edge_id=candidate.candidate_edge_id,
            stratum_id=candidate.stratum_id(score_version=_SCORE_VERSION),
            score_version=_SCORE_VERSION,
            source_score_collection_id="point-score-collection-v1",
            n_subjects=8,
            score=(0.0 if index == 200 else float(point_scores[index])),
            status=(
                ActiveEdgeScoreStatus.STRUCTURAL_ZERO
                if index == 200
                else ActiveEdgeScoreStatus.OBSERVED
            ),
            reason_code=("receptor_ineligible" if index == 200 else None),
        )
        for index, candidate in enumerate(universe.candidates)
    )
    null_grid = (np.arange(200, dtype=float) + 0.5) / 200
    nulls = tuple(
        NullActiveEdgeScoreRecord(
            candidate_edge_id=point.candidate_edge_id,
            plan_id=plan,
            stratum_id=point.stratum_id,
            score_version=_SCORE_VERSION,
            null_rerun_record_id=f"rerun-{plan}",
            source_score_collection_id=f"null-score-collection-{plan}",
            n_subjects=8,
            score=float(null_grid[plan_index]),
            status=ActiveEdgeScoreStatus.OBSERVED,
            reason_code=None,
        )
        for point in points
        for plan_index, plan in enumerate(plans)
    )
    distribution = NullScoreDistribution(
        active_null_id="active-null-v1",
        active_null_spec_id=ActiveNullSpec().active_null_spec_id,
        candidate_universe_id=universe.universe_id,
        candidate_universe_policy_id=ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
        score_spec_id=TEST_SCORE_SPEC_ID,
        point_records=points,
        plan_ids=plans,
        null_records=nulls,
    )
    candidate = estimate_active_probabilities(distribution, spec=spec)
    released = estimate_active_probabilities(
        distribution,
        spec=spec,
        calibration_gate=gate,
    )
    return universe, distribution, spec, gate, candidate, released


def _rehash_manifest(destination: Path) -> None:
    manifest_path = destination / "active_probability_manifest.json"
    status_path = destination / "_status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_id"] = stable_id(
        "active_probability_result",
        {key: value for key, value in manifest.items() if key != "artifact_id"},
        schema_version="2",
    )
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["artifact_id"] = manifest["artifact_id"]
    status_path.write_text(f"{canonical_json(status)}\n", encoding="utf-8")


def test_active_probability_schemas_are_valid_and_independent() -> None:
    filenames = (
        "active_probability_result.schema.json",
        "active_probabilities.schema.json",
        "active_probability_candidate_diagnostics.schema.json",
        "active_probability_stratum_diagnostics.schema.json",
        "active_probability_null_sources.schema.json",
    )
    for filename in filenames:
        value = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(value)


def test_diagnostic_only_artifact_never_creates_public_probability_table(
    tmp_path: Path,
    probability_case: tuple[
        FrozenActiveEdgeUniverse,
        NullScoreDistribution,
        ActiveProbabilitySpec,
        G3PCalibrationGate,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    universe, distribution, spec, _, candidate, _ = probability_case
    destination = tmp_path / "diagnostic-only"

    result = write_active_probability_result(
        destination,
        collection=candidate,
        distribution=distribution,
        universe=universe,
        spec=spec,
        calibration_gate=None,
    )

    assert result.has_released_probabilities is False
    assert result.manifest["release_mode"] == "diagnostic_only"
    assert not (destination / "active_probabilities.parquet").exists()
    with pytest.raises(KeyError):
        result.read_active_probabilities()
    diagnostics = result.read_candidate_diagnostics()
    assert "candidate_comm_probability" in diagnostics
    assert "comm_probability" not in diagnostics
    assert diagnostics["candidate_comm_probability"].notna().all()
    assert len(result.read_null_sources()) == (
        len(universe.candidates) * len(distribution.plan_ids)
    )


def test_passed_gate_round_trips_exact_public_projection(
    tmp_path: Path,
    calibration_parent: tuple[G3PCalibrationResult, CalibrationReplayRegistry],
    probability_case: tuple[
        FrozenActiveEdgeUniverse,
        NullScoreDistribution,
        ActiveProbabilitySpec,
        G3PCalibrationGate,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    universe, distribution, spec, gate, _, released = probability_case
    parent, registry = calibration_parent
    destination = tmp_path / "released"

    with pytest.raises(ResultValidationError) as missing_parent:
        write_active_probability_result(
            tmp_path / "released-without-parent",
            collection=released,
            distribution=distribution,
            universe=universe,
            spec=spec,
            calibration_gate=gate,
        )
    assert (
        missing_parent.value.details.code
        == "active_probability_calibration_result_unavailable"
    )
    result = write_active_probability_result(
        destination,
        collection=released,
        distribution=distribution,
        universe=universe,
        spec=spec,
        calibration_gate=gate,
        calibration_result=parent,
        replay_registry=registry,
        n_jobs=2,
    )

    assert result.has_released_probabilities is True
    assert result.manifest["calibration_gate_status"] == "passed"
    assert (
        result.manifest["calibration_result_artifact_id"]
        == (parent.manifest["artifact_id"])
    )
    public = result.read_active_probabilities()
    expected = {
        item.candidate_edge_id: item.comm_probability for item in released.records
    }
    assert tuple(public["candidate_edge_id"]) == universe.candidate_edge_ids
    assert public["comm_probability"].tolist() == pytest.approx(
        [expected[edge_id] for edge_id in public["candidate_edge_id"]]
    )
    assert {
        "active_null_empirical_p_value",
        "candidate_local_fdr",
        "candidate_comm_probability",
    }.isdisjoint(public.columns)
    filtered = result.read_active_probabilities(
        filters={"interaction_id": "lr-007"},
        columns=["interaction_id", "comm_probability"],
    )
    assert filtered["interaction_id"].tolist() == ["lr-007"]
    with pytest.raises(ResultValidationError) as missing_result:
        ActiveProbabilityResult.load(destination)
    assert (
        missing_result.value.details.code
        == "active_probability_calibration_result_unavailable"
    )
    with pytest.raises(ResultValidationError) as missing_registry:
        ActiveProbabilityResult.load(
            destination,
            calibration_result_path=parent.path,
        )
    assert (
        missing_registry.value.details.code
        == "active_probability_calibration_replay_unavailable"
    )
    assert (
        ActiveProbabilityResult.load(
            destination,
            calibration_result_path=parent.path,
            replay_registry=registry,
            n_jobs=2,
        ).manifest
        == result.manifest
    )


def test_load_rejects_calibration_lineage_tampering_after_outer_rehash(
    tmp_path: Path,
    calibration_parent: tuple[G3PCalibrationResult, CalibrationReplayRegistry],
    probability_case: tuple[
        FrozenActiveEdgeUniverse,
        NullScoreDistribution,
        ActiveProbabilitySpec,
        G3PCalibrationGate,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    universe, distribution, spec, gate, _, released = probability_case
    parent, registry = calibration_parent
    original = tmp_path / "lineage-original"
    write_active_probability_result(
        original,
        collection=released,
        distribution=distribution,
        universe=universe,
        spec=spec,
        calibration_gate=gate,
        calibration_result=parent,
        replay_registry=registry,
        n_jobs=2,
    )

    destination = tmp_path / "tampered-calibration-result-artifact-id"
    shutil.copytree(original, destination)
    manifest_path = destination / "active_probability_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration_result_artifact_id"] = "tampered-parent-artifact-id"
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    _rehash_manifest(destination)

    with pytest.raises(ResultValidationError) as raised:
        ActiveProbabilityResult.load(
            destination,
            calibration_result_path=parent.path,
            replay_registry=registry,
            n_jobs=2,
        )
    assert (
        raised.value.details.code
        == "active_probability_calibration_result_binding_mismatch"
    )


def test_load_recomputes_candidate_probability_after_synchronized_rehash(
    tmp_path: Path,
    calibration_parent: tuple[G3PCalibrationResult, CalibrationReplayRegistry],
    probability_case: tuple[
        FrozenActiveEdgeUniverse,
        NullScoreDistribution,
        ActiveProbabilitySpec,
        G3PCalibrationGate,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    universe, distribution, spec, gate, _, released = probability_case
    parent, registry = calibration_parent
    destination = tmp_path / "rehashed-probability"
    write_active_probability_result(
        destination,
        collection=released,
        distribution=distribution,
        universe=universe,
        spec=spec,
        calibration_gate=gate,
        calibration_result=parent,
        replay_registry=registry,
        n_jobs=2,
    )
    candidate_path = destination / "active_probability_candidate_diagnostics.parquet"
    public_path = destination / "active_probabilities.parquet"
    candidate = pd.read_parquet(candidate_path)
    public = pd.read_parquet(public_path)
    index = int(
        candidate.index[candidate["candidate_comm_probability"].between(0.01, 0.99)][0]
    )
    edge_id = str(candidate.loc[index, "candidate_edge_id"])
    forged_probability = float(candidate.loc[index, "candidate_comm_probability"]) * 0.9
    candidate.loc[index, "candidate_comm_probability"] = forged_probability
    candidate.loc[index, "candidate_local_fdr"] = 1.0 - forged_probability
    public.loc[public["candidate_edge_id"] == edge_id, "comm_probability"] = (
        forged_probability
    )
    candidate.to_parquet(candidate_path, index=False, engine="pyarrow")
    public.to_parquet(public_path, index=False, engine="pyarrow")
    manifest_path = destination / "active_probability_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"][ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE]["sha256"] = (
        _sha256_file(candidate_path)
    )
    manifest["tables"][ACTIVE_PROBABILITY_TABLE]["sha256"] = _sha256_file(public_path)
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    _rehash_manifest(destination)

    with pytest.raises(ResultValidationError) as raised:
        ActiveProbabilityResult.load(
            destination,
            calibration_result_path=parent.path,
            replay_registry=registry,
            n_jobs=2,
        )
    assert raised.value.details.code == "active_probability_recomputation_mismatch"


def test_failed_gate_remains_diagnostic_only(
    tmp_path: Path,
    probability_case: tuple[
        FrozenActiveEdgeUniverse,
        NullScoreDistribution,
        ActiveProbabilitySpec,
        G3PCalibrationGate,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    universe, distribution, spec, _, _, _ = probability_case
    failed_gate = _not_estimable_gate(distribution, spec)
    failed = estimate_active_probabilities(
        distribution,
        spec=spec,
        calibration_gate=failed_gate,
    )

    result = write_active_probability_result(
        tmp_path / "failed-gate",
        collection=failed,
        distribution=distribution,
        universe=universe,
        spec=spec,
        calibration_gate=failed_gate,
    )

    assert result.has_released_probabilities is False
    assert result.manifest["calibration_gate_status"] == "not_estimable"
    assert ACTIVE_PROBABILITY_TABLE not in result.manifest["tables"]


def test_writer_rejects_parent_mismatch_before_projection(
    tmp_path: Path,
    probability_case: tuple[
        FrozenActiveEdgeUniverse,
        NullScoreDistribution,
        ActiveProbabilitySpec,
        G3PCalibrationGate,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    universe, distribution, spec, gate, _, released = probability_case
    foreign = freeze_active_edge_universe(
        universe.candidates[:-1],
        universe_name="foreign-universe",
        contrast_id=universe.contrast_id,
        score_version=universe.score_version,
    )

    with pytest.raises(ResultValidationError) as raised:
        write_active_probability_result(
            tmp_path / "mismatched",
            collection=released,
            distribution=distribution,
            universe=foreign,
            spec=spec,
            calibration_gate=gate,
        )
    assert raised.value.details.code == "active_probability_parent_binding_mismatch"
    assert not (tmp_path / "mismatched").exists()


def test_load_rejects_missing_null_rectangle_even_after_rehash(
    tmp_path: Path,
    probability_case: tuple[
        FrozenActiveEdgeUniverse,
        NullScoreDistribution,
        ActiveProbabilitySpec,
        G3PCalibrationGate,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    universe, distribution, spec, _, candidate, _ = probability_case
    destination = tmp_path / "missing-null"
    write_active_probability_result(
        destination,
        collection=candidate,
        distribution=distribution,
        universe=universe,
        spec=spec,
        calibration_gate=None,
    )
    null_path = destination / "active_probability_null_sources.parquet"
    nulls = pd.read_parquet(null_path).iloc[:-1].copy()
    nulls.to_parquet(null_path, index=False, engine="pyarrow")
    manifest_path = destination / "active_probability_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["tables"][ACTIVE_PROBABILITY_NULL_SOURCE_TABLE]
    record["rows"] = len(nulls)
    record["sha256"] = _sha256_file(null_path)
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    _rehash_manifest(destination)

    with pytest.raises(ResultValidationError) as raised:
        ActiveProbabilityResult.load(destination)
    assert raised.value.details.code == "active_probability_null_rectangle_mismatch"


def test_load_rejects_corrupted_bytes(
    tmp_path: Path,
    probability_case: tuple[
        FrozenActiveEdgeUniverse,
        NullScoreDistribution,
        ActiveProbabilitySpec,
        G3PCalibrationGate,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    universe, distribution, spec, _, candidate, _ = probability_case
    destination = tmp_path / "corrupted"
    write_active_probability_result(
        destination,
        collection=candidate,
        distribution=distribution,
        universe=universe,
        spec=spec,
        calibration_gate=None,
    )
    path = destination / "active_probability_candidate_diagnostics.parquet"
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises(ResultValidationError) as raised:
        ActiveProbabilityResult.load(destination)
    assert raised.value.details.code == "active_probability_digest_mismatch"


def test_old_result_schemas_still_reject_probability_release(
    result_payload: dict[str, object],
) -> None:
    tables = result_payload["tables"]
    assert isinstance(tables, dict)
    interactions = tables["interactions"]
    assert isinstance(interactions, pd.DataFrame)
    forged = interactions.copy(deep=True)
    forged["comm_probability"] = 0.9
    with pytest.raises(ResultValidationError) as raised:
        validate_table("interactions", forged)
    assert raised.value.details.code == "v0_1_inferential_field_enabled"

    for version in ("1.0.0", "2.0.0", "3.0.0", "4.0.0", "5.0.0"):
        columns = _table_columns(version, CROSSFIT_COMPONENT_TABLE)
        crossfit = pd.DataFrame(columns=columns)
        crossfit["comm_probability"] = pd.Series(dtype=float)
        with pytest.raises(ValueError, match="columns do not match"):
            validate_crossfit_table(
                CROSSFIT_COMPONENT_TABLE,
                crossfit,
                schema_version=version,
            )
