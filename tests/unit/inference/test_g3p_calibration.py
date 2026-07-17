from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.calibration import diagnostic_generator_manifest

import crychic.inference.g3p_calibration as _g3p_calibration
from crychic.core import ContractError, SeedLineage
from crychic.inference import (
    ActiveProbabilitySpec,
    G3PCalibrationEvidence,
    G3PCalibrationGate,
    G3PCalibrationProtocol,
    G3PCalibrationScenarioResult,
    G3PCalibrationScenarioStatus,
    G3PCalibrationStratumResult,
    G3PCandidateStatus,
    G3PDependenceStructure,
    G3PGateStatus,
    G3PReplicatePredictions,
    build_g3p_calibration_gate,
    summarize_g3p_calibration_campaign,
)
from crychic.inference.calibration_attestation import CalibrationCampaignKind
from crychic.resampling import ActiveNullSpec
from crychic.scoring import ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID


def _protocol() -> G3PCalibrationProtocol:
    probability_spec = ActiveProbabilitySpec()
    return G3PCalibrationProtocol(
        campaign_name="unit-development-campaign",
        generator_manifest=diagnostic_generator_manifest(
            CalibrationCampaignKind.G3_PROBABILITY,
            "unit-g3p-generator-v1",
        ),
        active_null_spec_id=ActiveNullSpec().active_null_spec_id,
        active_probability_spec_id=probability_spec.spec_id,
        score_spec_id="score-spec-v4",
        candidate_universe_policy_id=ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
        estimator_id=probability_spec.estimator_id,
        stratum_policy_id=probability_spec.stratum_policy_id,
        score_version="receiver_gain_percentile_mechanistic_conserved_sender_v4",
        seed_lineage=SeedLineage(20260716).derive("unit-g3p-campaign"),
    )


def _truth(prevalence: float, replicate_index: int, n_candidates: int) -> np.ndarray:
    count = max(1, round(prevalence * n_candidates))
    values = np.zeros(n_candidates, dtype=bool)
    # Positives span the probability range; neither class separates perfectly.
    indexes = (
        np.linspace(20, n_candidates - 21, count, dtype=int) + replicate_index
    ) % n_candidates
    values[indexes] = True
    return values


def _replicate(
    protocol: G3PCalibrationProtocol,
    dependence: G3PDependenceStructure,
    prevalence: float,
    replicate_index: int,
    *,
    n_candidates: int = 400,
) -> G3PReplicatePredictions:
    candidate_ids = tuple(f"edge-{index:03d}" for index in range(n_candidates))
    truth = _truth(prevalence, replicate_index, n_candidates)
    phase = np.linspace(-0.75, 0.75, n_candidates)
    probabilities = np.clip(prevalence * np.exp(phase), 1e-5, 0.95)
    return G3PReplicatePredictions(
        protocol_id=protocol.protocol_id,
        dependence_structure=dependence,
        non_null_prevalence=prevalence,
        replicate_index=replicate_index,
        seed_lineage=protocol.replicate_seed_lineage(
            dependence, prevalence, replicate_index
        ),
        source_active_null_id=(
            f"active-null-{dependence.value}-{prevalence:g}-{replicate_index}"
        ),
        source_candidate_universe_id=(
            f"universe-{dependence.value}-{prevalence:g}-{replicate_index}"
        ),
        source_probability_collection_id=(
            f"probability-{dependence.value}-{prevalence:g}-{replicate_index}"
        ),
        candidate_edge_ids=candidate_ids,
        stratum_ids=("stratum-1",) * n_candidates,
        truth_active=truth,
        candidate_probabilities=probabilities,
        candidate_statuses=(G3PCandidateStatus.ELIGIBLE,) * n_candidates,
        candidate_reason_codes=(None,) * n_candidates,
    )


def _small_grid(
    protocol: G3PCalibrationProtocol,
) -> tuple[G3PReplicatePredictions, ...]:
    return tuple(
        _replicate(protocol, dependence, prevalence, replicate_index)
        for dependence in G3PDependenceStructure
        for prevalence in protocol.required_prevalences
        for replicate_index in range(3)
    )


def test_protocol_freezes_complete_dependence_and_prevalence_grid() -> None:
    protocol = _protocol()

    assert protocol.required_dependence_structures == tuple(
        item.value for item in G3PDependenceStructure
    )
    assert protocol.required_prevalences == (0.005, 0.01, 0.05, 0.1)
    assert protocol.to_dict()["minimum_replicates_per_cell"] == 1_000
    assert protocol.to_dict()["minimum_candidates_per_stratum"] == 200
    assert protocol.generator_release_verified is False


def test_scenario_evidence_and_gate_cannot_be_caller_constructed() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        G3PCalibrationStratumResult()
    with pytest.raises(TypeError, match="producer-owned"):
        G3PCalibrationScenarioResult()
    with pytest.raises(TypeError, match="producer-owned"):
        G3PCalibrationEvidence()
    with pytest.raises(TypeError, match="build_g3p"):
        G3PCalibrationGate()


def test_small_complete_grid_is_auditable_but_cannot_release_probability() -> None:
    protocol = _protocol()
    campaign = summarize_g3p_calibration_campaign(protocol, _small_grid(protocol))
    gate = build_g3p_calibration_gate(campaign.evidence)

    assert len(campaign.scenarios) == 12
    assert all(
        item.status is G3PCalibrationScenarioStatus.OBSERVED
        for item in campaign.scenarios
    )
    assert {item.n_replicates for item in campaign.scenarios} == {3}
    assert campaign.evidence.source_artifact_id == campaign.campaign_id
    assert gate.status is G3PGateStatus.NOT_ESTIMABLE
    assert gate.reason_code == "g3p_calibration_insufficient_replicates"
    assert gate.comm_probability_release_allowed is False


def test_public_raw_ledger_cannot_self_attest_its_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    campaign = summarize_g3p_calibration_campaign(protocol, _small_grid(protocol))
    monkeypatch.setattr(_g3p_calibration, "_MIN_CALIBRATION_REPLICATES", 3)

    gate = build_g3p_calibration_gate(campaign.evidence)

    assert campaign.evidence.generator_verified is False
    assert gate.generator_verified is False
    assert gate.status is G3PGateStatus.NOT_ESTIMABLE
    assert gate.reason_code == "g3p_calibration_generator_unverified"


def test_campaign_and_replicate_ids_are_input_order_invariant() -> None:
    protocol = _protocol()
    replicates = _small_grid(protocol)
    reversed_candidates = tuple(
        G3PReplicatePredictions(
            protocol_id=item.protocol_id,
            dependence_structure=item.dependence_structure,
            non_null_prevalence=item.non_null_prevalence,
            replicate_index=item.replicate_index,
            seed_lineage=item.seed_lineage,
            source_active_null_id=item.source_active_null_id,
            source_candidate_universe_id=item.source_candidate_universe_id,
            source_probability_collection_id=(item.source_probability_collection_id),
            candidate_edge_ids=tuple(reversed(item.candidate_edge_ids)),
            stratum_ids=tuple(reversed(item.stratum_ids)),
            truth_active=item.truth_active[::-1],
            candidate_probabilities=item.candidate_probabilities[::-1],
            candidate_statuses=tuple(reversed(item.candidate_statuses)),
            candidate_reason_codes=tuple(reversed(item.candidate_reason_codes)),
            status=item.status,
            reason_code=item.reason_code,
        )
        for item in reversed(replicates)
    )

    first = summarize_g3p_calibration_campaign(protocol, replicates)
    second = summarize_g3p_calibration_campaign(protocol, reversed_candidates)

    assert tuple(item.replicate_id for item in first.replicates) == tuple(
        item.replicate_id for item in second.replicates
    )
    assert first.campaign_id == second.campaign_id
    assert first.evidence.evidence_id == second.evidence.evidence_id


def test_missing_grid_cell_and_small_stratum_fail_closed() -> None:
    protocol = _protocol()
    replicates = _small_grid(protocol)
    missing = tuple(
        item
        for item in replicates
        if not (
            item.dependence_structure
            is G3PDependenceStructure.TARGET_DEGREE_CORRELATED_EDGES
            and item.non_null_prevalence == 0.1
        )
    )
    missing_campaign = summarize_g3p_calibration_campaign(protocol, missing)
    small_replicate = _replicate(
        protocol,
        G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES,
        0.005,
        0,
        n_candidates=199,
    )
    remaining = tuple(
        item
        for item in replicates
        if not (
            item.dependence_structure
            is G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES
            and item.non_null_prevalence == 0.005
            and item.replicate_index == 0
        )
    )
    small_campaign = summarize_g3p_calibration_campaign(
        protocol, (small_replicate, *remaining)
    )

    missing_scenario = next(
        item
        for item in missing_campaign.scenarios
        if item.dependence_structure
        == G3PDependenceStructure.TARGET_DEGREE_CORRELATED_EDGES.value
        and item.non_null_prevalence == 0.1
    )
    assert missing_scenario.reason_code == "g3p_calibration_missing_replicates"
    assert (
        build_g3p_calibration_gate(missing_campaign.evidence).status
        is G3PGateStatus.NOT_ESTIMABLE
    )
    small_scenario = next(
        item
        for item in small_campaign.scenarios
        if item.dependence_structure
        == G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES.value
        and item.non_null_prevalence == 0.005
    )
    assert small_scenario.reason_code == "g3p_calibration_candidate_coverage_mismatch"


def test_duplicate_replicate_and_foreign_protocol_are_rejected() -> None:
    protocol = _protocol()
    replicates = _small_grid(protocol)

    with pytest.raises(ContractError) as duplicate:
        summarize_g3p_calibration_campaign(protocol, (replicates[0], *replicates))
    assert duplicate.value.details.code == "duplicate_g3p_calibration_replicate"
    with pytest.raises(ContractError) as foreign:
        summarize_g3p_calibration_campaign(
            protocol, (replace(replicates[0], protocol_id="foreign-protocol"),)
        )
    assert foreign.value.details.code == "g3p_replicate_protocol_mismatch"


def test_wrong_seed_is_retained_as_not_estimable_evidence() -> None:
    protocol = _protocol()
    replicates = _small_grid(protocol)
    wrong = replace(replicates[0], seed_lineage=SeedLineage(99).derive("wrong"))
    campaign = summarize_g3p_calibration_campaign(protocol, (wrong, *replicates[1:]))
    scenario = next(
        item
        for item in campaign.scenarios
        if item.dependence_structure
        == G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES.value
        and item.non_null_prevalence == 0.005
    )

    assert scenario.status is G3PCalibrationScenarioStatus.NOT_ESTIMABLE
    assert scenario.reason_code == "g3p_calibration_seed_lineage_mismatch"


def test_reused_source_artifact_cannot_inflate_replicate_count() -> None:
    protocol = _protocol()
    replicates = _small_grid(protocol)
    first, second = replicates[:2]
    reused = replace(
        second,
        source_active_null_id=first.source_active_null_id,
        source_probability_collection_id=first.source_probability_collection_id,
    )

    with pytest.raises(ContractError) as raised:
        summarize_g3p_calibration_campaign(protocol, (first, reused, *replicates[2:]))
    assert raised.value.details.code == "duplicate_g3p_calibration_source_artifact"

    reused_universe = replace(
        second,
        source_candidate_universe_id=first.source_candidate_universe_id,
    )
    with pytest.raises(ContractError) as universe:
        summarize_g3p_calibration_campaign(
            protocol,
            (first, reused_universe, *replicates[2:]),
        )
    assert universe.value.details.code == (
        "duplicate_g3p_calibration_source_artifact"
    )


def test_structural_zeros_are_recorded_but_do_not_satisfy_eligible_minimum() -> None:
    protocol = _protocol()
    original = _small_grid(protocol)
    replacement: list[G3PReplicatePredictions] = []
    for replicate_index in range(3):
        base = _replicate(
            protocol,
            G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES,
            0.005,
            replicate_index,
            n_candidates=250,
        )
        truth = base.truth_active.copy()
        truth[199:] = False
        probabilities = base.candidate_probabilities.copy()
        probabilities[199:] = 0.0
        replacement.append(
            replace(
                base,
                truth_active=truth,
                candidate_probabilities=probabilities,
                candidate_statuses=(G3PCandidateStatus.ELIGIBLE,) * 199
                + (G3PCandidateStatus.STRUCTURAL_ZERO,) * 51,
                candidate_reason_codes=(None,) * 199
                + ("simulated_receptor_ineligible",) * 51,
            )
        )
    remaining = tuple(
        item
        for item in original
        if not (
            item.dependence_structure
            is G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES
            and item.non_null_prevalence == 0.005
        )
    )

    campaign = summarize_g3p_calibration_campaign(protocol, (*replacement, *remaining))
    scenario = next(
        item
        for item in campaign.scenarios
        if item.dependence_structure
        == G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES.value
        and item.non_null_prevalence == 0.005
    )

    assert replacement[0].n_candidates == 250
    assert replacement[0].n_eligible_candidates == 199
    assert replacement[0].n_structural_zero_candidates == 51
    assert scenario.minimum_eligible_candidates_per_stratum == 199
    assert scenario.status is G3PCalibrationScenarioStatus.NOT_ESTIMABLE
    assert scenario.reason_code == "g3p_calibration_stratum_too_small"


def test_candidate_failure_makes_the_replicate_and_scenario_not_estimable() -> None:
    protocol = _protocol()
    replicates = _small_grid(protocol)
    base = replicates[0]
    probabilities = base.candidate_probabilities.copy()
    probabilities[0] = np.nan
    statuses = list(base.candidate_statuses)
    statuses[0] = G3PCandidateStatus.NOT_ESTIMABLE
    reasons = list(base.candidate_reason_codes)
    reasons[0] = "candidate_probability_missing"
    unavailable = replace(
        base,
        candidate_probabilities=probabilities,
        candidate_statuses=tuple(statuses),
        candidate_reason_codes=tuple(reasons),
        status="not_estimable",
        reason_code="candidate_probability_missing",
    )

    campaign = summarize_g3p_calibration_campaign(
        protocol, (unavailable, *replicates[1:])
    )
    scenario = next(
        item
        for item in campaign.scenarios
        if item.dependence_structure
        == G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES.value
        and item.non_null_prevalence == 0.005
    )

    assert scenario.status is G3PCalibrationScenarioStatus.NOT_ESTIMABLE
    assert scenario.reason_code == "g3p_calibration_nonobserved_replicate"


def test_producer_capability_token_cannot_be_replaced_by_a_public_marker() -> None:
    protocol = _protocol()
    campaign = summarize_g3p_calibration_campaign(protocol, _small_grid(protocol))
    gate = build_g3p_calibration_gate(campaign.evidence)

    object.__setattr__(campaign.evidence, "_producer_token", "known-producer-marker")
    object.__setattr__(gate, "_producer_token", "known-gate-marker")

    with pytest.raises(ContractError) as evidence_error:
        build_g3p_calibration_gate(campaign.evidence)
    assert evidence_error.value.details.code == "g3p_evidence_integrity_violation"
    with pytest.raises(ContractError) as gate_error:
        gate.to_dict()
    assert gate_error.value.details.code == "g3p_gate_integrity_violation"


def test_opposing_stratum_miscalibration_cannot_cancel_in_the_gate() -> None:
    protocol = _protocol()
    original = _small_grid(protocol)
    candidate_ids = tuple(f"edge-{index:03d}" for index in range(400))
    probabilities = np.concatenate(
        (
            np.full(100, 0.2),
            np.full(100, 0.4),
            np.full(100, 0.2),
            np.full(100, 0.4),
        )
    )
    strata = ("receiver-a",) * 200 + ("receiver-b",) * 200
    replacements: list[G3PReplicatePredictions] = []
    for replicate_index in range(3):
        truth = np.zeros(400, dtype=bool)
        for offset, count in ((0, 30), (100, 50), (200, 10), (300, 30)):
            indexes = (
                offset + (np.arange(count, dtype=int) + replicate_index * 37) % 100
            )
            truth[indexes] = True
        replacements.append(
            G3PReplicatePredictions(
                protocol_id=protocol.protocol_id,
                dependence_structure=(
                    G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES
                ),
                non_null_prevalence=0.1,
                replicate_index=replicate_index,
                seed_lineage=protocol.replicate_seed_lineage(
                    G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES,
                    0.1,
                    replicate_index,
                ),
                source_active_null_id=f"stratified-null-{replicate_index}",
                source_candidate_universe_id=f"stratified-universe-{replicate_index}",
                source_probability_collection_id=(
                    f"stratified-probability-{replicate_index}"
                ),
                candidate_edge_ids=candidate_ids,
                stratum_ids=strata,
                truth_active=truth,
                candidate_probabilities=probabilities,
                candidate_statuses=(G3PCandidateStatus.ELIGIBLE,) * 400,
                candidate_reason_codes=(None,) * 400,
            )
        )
    remaining = tuple(
        item
        for item in original
        if not (
            item.dependence_structure
            is G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES
            and item.non_null_prevalence == 0.1
        )
    )

    campaign = summarize_g3p_calibration_campaign(protocol, (*replacements, *remaining))
    scenario = next(
        item
        for item in campaign.scenarios
        if item.dependence_structure
        == G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES.value
        and item.non_null_prevalence == 0.1
    )
    gate = build_g3p_calibration_gate(campaign.evidence)

    assert scenario.status is G3PCalibrationScenarioStatus.OBSERVED
    assert tuple(item.stratum_id for item in scenario.strata) == (
        "receiver-a",
        "receiver-b",
    )
    assert [item.ece for item in scenario.strata] == pytest.approx([0.1, 0.1])
    assert scenario.ece == pytest.approx(0.1)
    assert gate.maximum_ece == pytest.approx(0.1)
