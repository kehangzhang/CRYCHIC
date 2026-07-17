from __future__ import annotations

import math
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from crychic.core import SeedLineage
from crychic.inference import (
    ActiveProbabilitySpec,
    G3PCalibrationCampaign,
    G3PCalibrationGate,
    G3PCalibrationProtocol,
    G3PCandidateStatus,
    G3PDependenceStructure,
    G3PReplicatePredictions,
    build_g3p_calibration_gate,
    summarize_attested_g3p_calibration_campaign,
)
from crychic.inference.calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationGeneratorProfile,
    CalibrationReplayRequest,
    _freeze_calibration_replay_registry,
    _register_calibration_generator,
)
from crychic.resampling import ActiveNullSpec
from crychic.scoring import ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID

TEST_SCORE_SPEC_ID = "global-common-score-spec-v4-test"
TEST_SCORE_VERSION = "receiver_gain_percentile_mechanistic_conserved_sender_v4"
_CANDIDATE_IDS = tuple(f"calibration-edge-{index:03d}" for index in range(200))
_COMPACT_CANDIDATE_IDS = tuple(
    f"compact-calibration-edge-{index:03d}" for index in range(20)
)


def _cell_values(
    *,
    dependence: G3PDependenceStructure,
    prevalence: float,
    seed: int,
    replicate_index: int,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    high_denominator = 10 if prevalence == 0.1 else 20
    n_high = max(2, n_candidates // high_denominator)
    n_low = n_candidates - n_high
    low_probability = prevalence * 0.2
    high_probability = (prevalence * n_candidates - low_probability * n_low) / n_high
    probabilities = np.concatenate(
        (
            np.full(n_low, low_probability, dtype=float),
            np.full(n_high, high_probability, dtype=float),
        )
    )

    def balanced_truth(probability: float, size: int, block_size: int) -> np.ndarray:
        units = size // block_size
        before = math.floor(replicate_index * units * probability)
        after = math.floor((replicate_index + 1) * units * probability)
        count = after - before
        values: NDArray[np.bool_] = np.zeros(units, dtype=bool)
        start = (replicate_index * 37 + seed % max(units, 1)) % max(units, 1)
        indexes: NDArray[np.int64] = (start + np.arange(count, dtype=np.int64)) % units
        values[indexes] = True
        return np.repeat(values, block_size)

    outcome_block = (
        2
        if dependence is G3PDependenceStructure.SENDER_LR_BLOCK_CORRELATED_EDGES
        else 1
    )
    truth = np.concatenate(
        (
            balanced_truth(low_probability, n_low, outcome_block),
            balanced_truth(high_probability, n_high, outcome_block),
        )
    )
    return truth.astype(bool), probabilities.astype(float)


def _replay_test_replicate(
    protocol: object,
    request: CalibrationReplayRequest,
    config: Mapping[str, object],
) -> G3PReplicatePredictions:
    if not isinstance(protocol, G3PCalibrationProtocol):
        raise TypeError("test G3-P replay requires G3PCalibrationProtocol")
    dependence = G3PDependenceStructure(request.dependence_structure)
    prevalence = request.non_null_prevalence
    candidate_count = config.get("candidate_count")
    if (
        prevalence is None
        or not isinstance(candidate_count, int)
        or candidate_count not in {len(_CANDIDATE_IDS), len(_COMPACT_CANDIDATE_IDS)}
    ):
        raise ValueError("invalid test G3-P replay request")
    candidate_ids = (
        _CANDIDATE_IDS
        if candidate_count == len(_CANDIDATE_IDS)
        else _COMPACT_CANDIDATE_IDS
    )
    truth, probabilities = _cell_values(
        dependence=dependence,
        prevalence=prevalence,
        seed=request.seed_lineage.seed,
        replicate_index=request.replicate_index,
        n_candidates=len(candidate_ids),
    )
    return G3PReplicatePredictions(
        protocol_id=protocol.protocol_id,
        dependence_structure=dependence,
        non_null_prevalence=prevalence,
        replicate_index=request.replicate_index,
        seed_lineage=request.seed_lineage,
        source_active_null_id=(
            f"test-null-{dependence.value}-{prevalence:g}-{request.replicate_index:04d}"
        ),
        source_candidate_universe_id=(
            f"test-universe-{dependence.value}-{prevalence:g}-"
            f"{request.replicate_index:04d}"
        ),
        source_probability_collection_id=(
            f"test-probability-{dependence.value}-{prevalence:g}-"
            f"{request.replicate_index:04d}"
        ),
        candidate_edge_ids=candidate_ids,
        stratum_ids=("test-stratum",) * len(candidate_ids),
        truth_active=truth,
        candidate_probabilities=probabilities,
        candidate_statuses=(G3PCandidateStatus.ELIGIBLE,) * len(candidate_ids),
        candidate_reason_codes=(None,) * len(candidate_ids),
    )


@cache
def _test_generator_registration():  # type: ignore[no-untyped-def]
    return _register_calibration_generator(
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        generator_kind="test-only-beta-bernoulli-dependence-generator",
        generator_version="1.0.0",
        code_version="test-support-g3p-v1",
        config={"candidate_count": len(_CANDIDATE_IDS)},
        source_paths=(Path(__file__),),
        replay=_replay_test_replicate,
        profile=CalibrationGeneratorProfile.RELEASE_APPROVED,
    )


@cache
def _test_replay_registry():  # type: ignore[no-untyped-def]
    return _freeze_calibration_replay_registry((_test_generator_registration(),))


@cache
def _compact_generator_registration():  # type: ignore[no-untyped-def]
    return _register_calibration_generator(
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        generator_kind="test-only-compact-beta-bernoulli-generator",
        generator_version="1.0.0",
        code_version="test-support-compact-g3p-v1",
        config={"candidate_count": len(_COMPACT_CANDIDATE_IDS)},
        source_paths=(Path(__file__),),
        replay=_replay_test_replicate,
        profile=CalibrationGeneratorProfile.RELEASE_APPROVED,
    )


@cache
def _compact_replay_registry():  # type: ignore[no-untyped-def]
    return _freeze_calibration_replay_registry((_compact_generator_registration(),))


def release_test_replay_registry():  # type: ignore[no-untyped-def]
    """Return the immutable compact test-only G3-P replay registry."""

    return _compact_replay_registry()


def _attested_g3p_campaign(
    *,
    active_probability_spec_id: str,
    estimator_id: str,
    stratum_policy_id: str,
    n_replicates: int,
    campaign_name: str,
    compact: bool,
    score_spec_id: str = TEST_SCORE_SPEC_ID,
    score_version: str = TEST_SCORE_VERSION,
) -> G3PCalibrationCampaign:
    registration = (
        _compact_generator_registration() if compact else _test_generator_registration()
    )
    registry = _compact_replay_registry() if compact else _test_replay_registry()
    protocol = G3PCalibrationProtocol(
        campaign_name=campaign_name,
        generator_manifest=registration.manifest,
        active_null_spec_id=ActiveNullSpec().active_null_spec_id,
        active_probability_spec_id=active_probability_spec_id,
        score_spec_id=score_spec_id,
        candidate_universe_policy_id=ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
        estimator_id=estimator_id,
        stratum_policy_id=stratum_policy_id,
        score_version=score_version,
        seed_lineage=SeedLineage(20260716).derive("test-g3p-release-grid"),
    )
    replicates: list[G3PReplicatePredictions] = []
    for dependence in G3PDependenceStructure:
        for prevalence in protocol.required_prevalences:
            for replicate_index in range(n_replicates):
                lineage = protocol.replicate_seed_lineage(
                    dependence, prevalence, replicate_index
                )
                replicates.append(
                    _replay_test_replicate(
                        protocol,
                        CalibrationReplayRequest(
                            campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
                            protocol_id=protocol.protocol_id,
                            dependence_structure=dependence.value,
                            non_null_prevalence=prevalence,
                            replicate_index=replicate_index,
                            seed_lineage=lineage,
                        ),
                        registration.config,
                    )
                )
    return summarize_attested_g3p_calibration_campaign(
        protocol,
        replicates,
        registry=registry,
        n_jobs=2,
    )


@cache
def complete_g3p_campaign(
    *,
    active_probability_spec_id: str,
    estimator_id: str,
    stratum_policy_id: str,
    score_spec_id: str = TEST_SCORE_SPEC_ID,
    score_version: str = TEST_SCORE_VERSION,
) -> G3PCalibrationCampaign:
    """Run one fixed 1,000-replicate producer campaign for release-path tests."""

    return _attested_g3p_campaign(
        active_probability_spec_id=active_probability_spec_id,
        estimator_id=estimator_id,
        stratum_policy_id=stratum_policy_id,
        n_replicates=1_000,
        campaign_name="test-only-complete-g3p-release-grid",
        compact=False,
        score_spec_id=score_spec_id,
        score_version=score_version,
    )


@cache
def compact_g3p_campaign(
    *,
    active_probability_spec_id: str,
    estimator_id: str,
    stratum_policy_id: str,
    score_spec_id: str = TEST_SCORE_SPEC_ID,
    score_version: str = TEST_SCORE_VERSION,
) -> G3PCalibrationCampaign:
    """Run a 200-replicate full-grid campaign for persistence contract tests."""

    return _attested_g3p_campaign(
        active_probability_spec_id=active_probability_spec_id,
        estimator_id=estimator_id,
        stratum_policy_id=stratum_policy_id,
        n_replicates=200,
        campaign_name="test-only-compact-g3p-release-grid",
        compact=True,
        score_spec_id=score_spec_id,
        score_version=score_version,
    )


@cache
def passed_g3p_gate(
    *,
    active_probability_spec_id: str,
    estimator_id: str,
    stratum_policy_id: str,
    score_spec_id: str = TEST_SCORE_SPEC_ID,
    score_version: str = TEST_SCORE_VERSION,
) -> G3PCalibrationGate:
    campaign = complete_g3p_campaign(
        active_probability_spec_id=active_probability_spec_id,
        estimator_id=estimator_id,
        stratum_policy_id=stratum_policy_id,
        score_spec_id=score_spec_id,
        score_version=score_version,
    )
    return build_g3p_calibration_gate(campaign.evidence)


def default_passed_g3p_gate(
    spec: ActiveProbabilitySpec | None = None,
) -> G3PCalibrationGate:
    resolved = ActiveProbabilitySpec() if spec is None else spec
    return passed_g3p_gate(
        active_probability_spec_id=resolved.spec_id,
        estimator_id=resolved.estimator_id,
        stratum_policy_id=resolved.stratum_policy_id,
    )
