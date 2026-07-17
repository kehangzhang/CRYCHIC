from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.calibration import diagnostic_generator_manifest

from crychic.core import CommunicationMode, ContractError, SeedLineage, canonical_digest
from crychic.inference import (
    ActiveEdgeScoreStatus,
    ActiveProbabilityCollection,
    ActiveProbabilityRecord,
    ActiveProbabilitySpec,
    CalibrationReplayRequest,
    G3PCalibrationProtocol,
    G3PCandidateStatus,
    G3PDependenceStructure,
    G3PReplicateStatus,
    G3PSimulationTruth,
    LocalFDRFitStatus,
    LocalFDRStratumDiagnostics,
    PointActiveEdgeScoreRecord,
    build_g3p_replicate_from_pipeline,
    freeze_g3p_simulation_truth,
)
from crychic.inference.calibration_attestation import CalibrationCampaignKind
from crychic.resampling import ActiveNullSpec
from crychic.scoring import (
    ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
    ActiveEdgeCandidate,
    FrozenActiveEdgeUniverse,
    freeze_active_edge_universe,
)
from crychic.workflow import ActiveProbabilityPipelineResult

_SCORE_VERSION = "receiver_gain_percentile_mechanistic_conserved_sender_v4"
_SCORE_SPEC_ID = "g3p-pipeline-adapter-score-spec"


@dataclass(frozen=True, slots=True)
class _Fixture:
    protocol: G3PCalibrationProtocol
    request: CalibrationReplayRequest
    universe: FrozenActiveEdgeUniverse
    pipeline: ActiveProbabilityPipelineResult


@pytest.fixture(autouse=True)
def _bypass_composed_fixture_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unit fixtures retain typed leaf records but avoid constructing a 200 x 200
    # active-null campaign. Production calls still execute the real integrity audit.
    monkeypatch.setattr(
        ActiveProbabilityPipelineResult,
        "_require_intact",
        lambda self: None,
    )


def _protocol_and_request() -> tuple[
    G3PCalibrationProtocol,
    CalibrationReplayRequest,
]:
    probability_spec = ActiveProbabilitySpec()
    protocol = G3PCalibrationProtocol(
        campaign_name="g3p-pipeline-adapter-unit",
        generator_manifest=diagnostic_generator_manifest(
            CalibrationCampaignKind.G3_PROBABILITY,
            "g3p-pipeline-adapter-unit-generator",
        ),
        active_null_spec_id=ActiveNullSpec().active_null_spec_id,
        active_probability_spec_id=probability_spec.spec_id,
        score_spec_id=_SCORE_SPEC_ID,
        candidate_universe_policy_id=ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
        estimator_id=probability_spec.estimator_id,
        stratum_policy_id=probability_spec.stratum_policy_id,
        score_version=_SCORE_VERSION,
        seed_lineage=SeedLineage(20260716).derive("g3p-pipeline-adapter-unit"),
    )
    dependence = G3PDependenceStructure.INDEPENDENT_CANDIDATE_EDGES
    prevalence = 0.1
    request = CalibrationReplayRequest(
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        protocol_id=protocol.protocol_id,
        dependence_structure=dependence.value,
        non_null_prevalence=prevalence,
        replicate_index=3,
        seed_lineage=protocol.replicate_seed_lineage(dependence, prevalence, 3),
    )
    return protocol, request


def _candidate(index: int) -> ActiveEdgeCandidate:
    return ActiveEdgeCandidate(
        contrast_id="contrast-stim-control",
        context_id="condition=stim",
        sender=f"Sender{index}",
        receiver="Receiver",
        interaction_id=f"L{index}_R",
        driver_id=f"L{index}",
        mode=CommunicationMode.STATE,
    )


def _diagnostic(stratum_id: str) -> LocalFDRStratumDiagnostics:
    return LocalFDRStratumDiagnostics._from_values(
        stratum_id=stratum_id,
        status=LocalFDRFitStatus.OBSERVED,
        reason_code=None,
        n_candidate_edges=200,
        n_null_plans=200,
        complete_score_matrix=True,
        pi0=0.8,
        a=0.5,
        log_likelihood=1.0,
        projected_gradient_norm=0.0,
        curvature_minimum_eigenvalue=1.0,
        n_starts=11,
        n_successful_starts=11,
        log_likelihood_spread=0.0,
    )


def _fixture(
    states: tuple[tuple[ActiveEdgeScoreStatus, float | None], ...],
    *,
    reverse_probability_rows: bool = False,
) -> _Fixture:
    protocol, request = _protocol_and_request()
    candidates = tuple(_candidate(index) for index in range(len(states)))
    universe = freeze_active_edge_universe(
        candidates,
        universe_name="g3p-adapter-unit-universe",
        contrast_id="contrast-stim-control",
        score_version=_SCORE_VERSION,
    )
    points: list[PointActiveEdgeScoreRecord] = []
    records: list[ActiveProbabilityRecord] = []
    diagnostic_by_stratum: dict[str, LocalFDRStratumDiagnostics] = {}
    for candidate, (status, probability) in zip(
        universe.candidates,
        states,
        strict=True,
    ):
        reason = None
        score: float | None = 0.8
        if status is ActiveEdgeScoreStatus.STRUCTURAL_ZERO:
            score = 0.0
            reason = "candidate_structural_zero"
        elif status is ActiveEdgeScoreStatus.NOT_ESTIMABLE:
            score = None
            reason = "candidate_not_estimable"
        elif status is ActiveEdgeScoreStatus.FAILED:
            score = None
            reason = "candidate_failed"
        stratum_id = candidate.stratum_id(score_version=_SCORE_VERSION)
        diagnostic = diagnostic_by_stratum.setdefault(
            stratum_id,
            _diagnostic(stratum_id),
        )
        point = PointActiveEdgeScoreRecord(
            candidate_edge_id=candidate.candidate_edge_id,
            stratum_id=stratum_id,
            score_version=_SCORE_VERSION,
            source_score_collection_id="point-score-collection",
            n_subjects=4 if score is not None else 0,
            score=score,
            status=status,
            reason_code=reason,
        )
        points.append(point)
        records.append(
            ActiveProbabilityRecord._from_values(
                candidate_edge_id=candidate.candidate_edge_id,
                point_record_id=point.record_id,
                stratum_id=stratum_id,
                score_version=_SCORE_VERSION,
                point_score=score,
                point_status=status,
                point_reason_code=reason,
                active_null_empirical_p_value=(
                    None if probability is None else 0.1
                ),
                candidate_local_fdr=(
                    None if probability is None else 1.0 - probability
                ),
                candidate_comm_probability=probability,
                comm_probability=None,
                probability_reason_code=(
                    reason if probability is None else "g3p_gate_not_passed"
                ),
                local_fdr_diagnostic_id=diagnostic.diagnostic_id,
                calibration_gate_id=None,
            )
        )
    if reverse_probability_rows:
        records.reverse()
    distribution = SimpleNamespace(
        distribution_id="typed-null-distribution",
        point_records=tuple(points),
    )
    null_rerun = SimpleNamespace(
        active_null_spec_id=ActiveNullSpec().active_null_spec_id,
        score_spec_id=_SCORE_SPEC_ID,
        active_null_id="typed-active-null",
        distribution=distribution,
        records=(SimpleNamespace(status=SimpleNamespace(value="succeeded")),),
    )
    collection = object.__new__(ActiveProbabilityCollection)
    for name, value in {
        "candidate_universe_id": universe.universe_id,
        "active_null_id": null_rerun.active_null_id,
        "distribution_id": distribution.distribution_id,
        "records": tuple(records),
        "diagnostics": tuple(diagnostic_by_stratum.values()),
        "collection_id": "typed-probability-collection",
    }.items():
        object.__setattr__(collection, name, value)
    pipeline = object.__new__(ActiveProbabilityPipelineResult)
    for name, value in {
        "universe": universe,
        "active_probability_spec": ActiveProbabilitySpec(),
        "null_rerun": null_rerun,
        "probability_collection": collection,
    }.items():
        object.__setattr__(pipeline, name, value)
    return _Fixture(protocol, request, universe, pipeline)


def _truth(
    fixture: _Fixture,
    values: tuple[bool, ...],
    *,
    reverse_rows: bool = False,
) -> G3PSimulationTruth:
    items = list(zip(fixture.universe.candidates, values, strict=True))
    if reverse_rows:
        items.reverse()
    return freeze_g3p_simulation_truth(
        fixture.protocol,
        fixture.request,
        fixture.universe,
        truth_by_candidate=dict(items),
        generation_id="simulated-dataset-000003",
        source_digest=canonical_digest({"replicate": 3, "truth": list(values)}),
    )


def test_builds_observed_ledger_from_typed_pipeline_children() -> None:
    fixture = _fixture(
        (
            (ActiveEdgeScoreStatus.OBSERVED, 0.75),
            (ActiveEdgeScoreStatus.STRUCTURAL_ZERO, 0.0),
        )
    )
    truth = _truth(fixture, (True, False))

    replicate = build_g3p_replicate_from_pipeline(
        fixture.protocol,
        fixture.request,
        fixture.pipeline,
        truth,
    )

    by_edge = {
        edge_id: (probability, status, reason, active, stratum)
        for edge_id, probability, status, reason, active, stratum in zip(
            replicate.candidate_edge_ids,
            replicate.candidate_probabilities,
            replicate.candidate_statuses,
            replicate.candidate_reason_codes,
            replicate.truth_active,
            replicate.stratum_ids,
            strict=True,
        )
    }
    observed_id = fixture.universe.candidates[0].candidate_edge_id
    zero_id = fixture.universe.candidates[1].candidate_edge_id
    assert replicate.status is G3PReplicateStatus.OBSERVED
    assert by_edge[observed_id][:4] == (0.75, G3PCandidateStatus.ELIGIBLE, None, True)
    assert by_edge[zero_id][:4] == (
        0.0,
        G3PCandidateStatus.STRUCTURAL_ZERO,
        "candidate_structural_zero",
        False,
    )
    observed_record = next(
        item
        for item in fixture.pipeline.probability_collection.records
        if item.candidate_edge_id == observed_id
    )
    assert by_edge[observed_id][4] == observed_record.stratum_id
    assert replicate.source_active_null_id == fixture.pipeline.null_rerun.active_null_id
    assert replicate.source_candidate_universe_id == fixture.universe.universe_id
    assert replicate.source_probability_collection_id == (
        fixture.pipeline.probability_collection.collection_id
    )


@pytest.mark.parametrize(
    ("point_status", "expected_candidate", "expected_replicate"),
    (
        (
            ActiveEdgeScoreStatus.NOT_ESTIMABLE,
            G3PCandidateStatus.NOT_ESTIMABLE,
            G3PReplicateStatus.NOT_ESTIMABLE,
        ),
        (
            ActiveEdgeScoreStatus.FAILED,
            G3PCandidateStatus.FAILED,
            G3PReplicateStatus.FAILED,
        ),
    ),
)
def test_maps_nonobserved_pipeline_states(
    point_status: ActiveEdgeScoreStatus,
    expected_candidate: G3PCandidateStatus,
    expected_replicate: G3PReplicateStatus,
) -> None:
    fixture = _fixture(((point_status, None),))
    replicate = build_g3p_replicate_from_pipeline(
        fixture.protocol,
        fixture.request,
        fixture.pipeline,
        _truth(fixture, (False,)),
    )

    assert replicate.status is expected_replicate
    assert replicate.candidate_statuses == (expected_candidate,)
    assert np.isnan(replicate.candidate_probabilities[0])
    assert replicate.candidate_reason_codes[0] is not None


def test_truth_requires_exact_typed_biological_candidate_coverage() -> None:
    fixture = _fixture(
        (
            (ActiveEdgeScoreStatus.OBSERVED, 0.6),
            (ActiveEdgeScoreStatus.OBSERVED, 0.4),
        )
    )
    candidates = fixture.universe.candidates
    common = {
        "generation_id": "simulated-dataset-000003",
        "source_digest": canonical_digest({"replicate": 3}),
    }
    with pytest.raises(ContractError, match="exactly cover"):
        freeze_g3p_simulation_truth(
            fixture.protocol,
            fixture.request,
            fixture.universe,
            truth_by_candidate={candidates[0]: True},
            **common,
        )
    foreign = ActiveEdgeCandidate(
        contrast_id=candidates[1].contrast_id,
        context_id=candidates[1].context_id,
        sender=candidates[1].sender,
        receiver=candidates[1].receiver,
        interaction_id=candidates[1].interaction_id,
        driver_id="wrong-driver",
        mode=candidates[1].mode,
    )
    with pytest.raises(ContractError, match="exactly cover"):
        freeze_g3p_simulation_truth(
            fixture.protocol,
            fixture.request,
            fixture.universe,
            truth_by_candidate={candidates[0]: True, foreign: False},
            **common,
        )
    with pytest.raises(ContractError, match="exactly cover"):
        freeze_g3p_simulation_truth(
            fixture.protocol,
            fixture.request,
            fixture.universe,
            truth_by_candidate={
                candidates[0]: True,
                candidates[1]: False,
                foreign: False,
            },
            **common,
        )


def test_rejects_probability_parent_identity_mismatch() -> None:
    fixture = _fixture(((ActiveEdgeScoreStatus.OBSERVED, 0.75),))
    object.__setattr__(
        fixture.pipeline.probability_collection,
        "candidate_universe_id",
        "foreign-universe",
    )

    with pytest.raises(ContractError, match="incompatible typed parents"):
        build_g3p_replicate_from_pipeline(
            fixture.protocol,
            fixture.request,
            fixture.pipeline,
            _truth(fixture, (True,)),
        )


def test_row_order_does_not_change_truth_or_replicate_identity() -> None:
    states = (
        (ActiveEdgeScoreStatus.OBSERVED, 0.75),
        (ActiveEdgeScoreStatus.STRUCTURAL_ZERO, 0.0),
    )
    first = _fixture(states)
    second = _fixture(states, reverse_probability_rows=True)
    first_replicate = build_g3p_replicate_from_pipeline(
        first.protocol,
        first.request,
        first.pipeline,
        _truth(first, (True, False)),
    )
    second_truth = _truth(second, (True, False), reverse_rows=True)
    second_replicate = build_g3p_replicate_from_pipeline(
        second.protocol,
        second.request,
        second.pipeline,
        second_truth,
    )

    assert first_replicate.replicate_id == second_replicate.replicate_id
    assert first_replicate.candidate_edge_ids == second_replicate.candidate_edge_ids


def test_adapter_has_no_array_or_caller_source_id_escape_hatch() -> None:
    fixture = _fixture(((ActiveEdgeScoreStatus.OBSERVED, 0.75),))
    with pytest.raises(TypeError, match="producer-owned G3PSimulationTruth"):
        build_g3p_replicate_from_pipeline(
            fixture.protocol,
            fixture.request,
            fixture.pipeline,
            np.asarray([True]),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="producer-owned"):
        G3PSimulationTruth()
