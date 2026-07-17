from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from tests.support.calibration import diagnostic_generator_manifest

import crychic.inference.g3f_calibration as _g3f_calibration
from crychic.core import ContractError, SeedLineage
from crychic.inference.calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationGeneratorProfile,
    CalibrationReplayRequest,
    _freeze_calibration_replay_registry,
    _register_calibration_generator,
)
from crychic.inference.g3f_calibration import (
    G3CalibrationMetric,
    G3CalibrationScenarioResult,
    G3CalibrationScenarioStatus,
    G3FrequencyCalibrationEvidence,
    G3FrequencyCalibrationGate,
    G3FrequencyCalibrationProtocol,
    G3FrequencyCalibrationReplicate,
    G3FrequencyDependenceStructure,
    G3FrequencyGateStatus,
    G3FrequencyReplicateStatus,
    build_g3_frequency_calibration_gate,
    summarize_attested_g3_frequency_calibration_campaign,
    summarize_g3_frequency_calibration_campaign,
)
from crychic.inference.hierarchical import freeze_hierarchical_fdr_spec
from crychic.inference.hypotheses import (
    HypothesisDeclaration,
    HypothesisRole,
    freeze_hypothesis_universe,
)

_PREVALENCES = (0.005, 0.01, 0.05, 0.10)


@lru_cache(maxsize=1)
def _protocol() -> G3FrequencyCalibrationProtocol:
    primaries = tuple(
        HypothesisDeclaration(
            endpoint="driver_family_receiver_context_omnibus_v1",
            contrast_name="all-contexts",
            receiver=f"receiver-{index:03d}",
            family_id=f"family-{index:03d}",
            mode="state",
            role=HypothesisRole.PRIMARY,
            multiplicity_family="g3f-primary",
        )
        for index in range(100)
    )
    children = tuple(
        HypothesisDeclaration(
            endpoint="family_common_integrated_lr_context_effect_v1",
            contrast_name="B-vs-A",
            receiver=parent.receiver,
            family_id=parent.family_id,
            mode="state",
            role=HypothesisRole.SECONDARY,
            multiplicity_family="g3f-secondary",
            parent_key=parent.hypothesis_key,
        )
        for parent in primaries
    )
    universe = freeze_hypothesis_universe(
        (*primaries, *children), universe_name="g3f-unit-universe"
    )
    return G3FrequencyCalibrationProtocol(
        campaign_name="g3f-unit-campaign",
        generator_manifest=diagnostic_generator_manifest(
            CalibrationCampaignKind.G3_FREQUENCY,
            "g3f-unit-generator",
        ),
        noninferiority_baseline_id="g3f-unit-baseline",
        hypothesis_universe=universe,
        hierarchical_procedure=freeze_hierarchical_fdr_spec(),
        seed_lineage=SeedLineage(20260716),
    )


def _columns() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str | None, ...]]:
    declarations = _protocol().hypothesis_universe.declarations
    parents = {
        item.hypothesis_key: item.hypothesis_id
        for item in declarations
        if item.role is HypothesisRole.PRIMARY
    }
    rows = tuple(
        sorted(
            (
                item.hypothesis_id,
                item.role.value,
                None
                if item.role is HypothesisRole.PRIMARY
                else parents[item.parent_key],
            )
            for item in declarations
        )
    )
    return (
        tuple(item[0] for item in rows),
        tuple(item[1] for item in rows),
        tuple(item[2] for item in rows),
    )


def _replicate(
    dependence: G3FrequencyDependenceStructure,
    prevalence: float | None,
    index: int,
    *,
    reverse_columns: bool = False,
    status: G3FrequencyReplicateStatus = G3FrequencyReplicateStatus.OBSERVED,
    protocol: G3FrequencyCalibrationProtocol | None = None,
) -> G3FrequencyCalibrationReplicate:
    resolved_protocol = _protocol() if protocol is None else protocol
    ids, roles, parents = _columns()
    truth = np.zeros(len(ids), dtype=bool)
    if prevalence is not None:
        primary_positions = tuple(
            position
            for position, role in enumerate(roles)
            if role == HypothesisRole.PRIMARY.value
        )
        truth[np.asarray(primary_positions[: int(len(ids) * prevalence)])] = True
    rejected = np.zeros(len(ids), dtype=bool)
    baseline_rejected = np.zeros(len(ids), dtype=bool)
    coverage = np.ones(len(ids), dtype=bool)
    interval_width = np.ones(len(ids), dtype=float)
    baseline_interval_width = np.ones(len(ids), dtype=float)
    permutation = None if prevalence is not None else (index + 0.5) / 2
    reason = None
    if status is not G3FrequencyReplicateStatus.OBSERVED:
        coverage[:] = False
        interval_width[:] = np.nan
        baseline_interval_width[:] = np.nan
        permutation = None
        reason = "synthetic_replicate_not_observed"
    if reverse_columns:
        order = np.arange(len(ids) - 1, -1, -1)
        ids = tuple(ids[int(item)] for item in order)
        roles = tuple(roles[int(item)] for item in order)
        parents = tuple(parents[int(item)] for item in order)
        truth = truth[order]
        rejected = rejected[order]
        baseline_rejected = baseline_rejected[order]
        coverage = coverage[order]
        interval_width = interval_width[order]
        baseline_interval_width = baseline_interval_width[order]
    return G3FrequencyCalibrationReplicate(
        protocol_id=resolved_protocol.protocol_id,
        dependence_structure=dependence,
        non_null_prevalence=prevalence,
        replicate_index=index,
        seed_lineage=resolved_protocol.replicate_seed_lineage(
            dependence, prevalence, index
        ),
        source_artifact_id=(f"source:{dependence.value}:{prevalence}:{index:04d}"),
        hypothesis_ids=ids,
        hypothesis_roles=roles,
        parent_hypothesis_ids=parents,
        truth_non_null=truth,
        rejected_at_alpha=rejected,
        baseline_rejected_at_alpha=baseline_rejected,
        ci_contains_truth=coverage,
        ci_interval_width=interval_width,
        baseline_ci_interval_width=baseline_interval_width,
        permutation_null_p_value=permutation,
        status=status,
        reason_code=reason,
    )


def _replicates(
    *,
    reverse: bool = False,
    protocol: G3FrequencyCalibrationProtocol | None = None,
):
    output = tuple(
        _replicate(
            dependence,
            prevalence,
            index,
            reverse_columns=reverse,
            protocol=protocol,
        )
        for dependence in G3FrequencyDependenceStructure
        for prevalence in (None, *_PREVALENCES)
        for index in range(2)
    )
    return tuple(reversed(output)) if reverse else output


def _replay_noninferiority_case(
    protocol: object,
    request: CalibrationReplayRequest,
    config: Mapping[str, object],
) -> G3FrequencyCalibrationReplicate:
    if not isinstance(protocol, G3FrequencyCalibrationProtocol):
        raise TypeError("unit replay requires G3FrequencyCalibrationProtocol")
    base = _replicate(
        G3FrequencyDependenceStructure(request.dependence_structure),
        request.non_null_prevalence,
        request.replicate_index,
        protocol=protocol,
    )
    mode = config.get("mode")
    if mode == "power":
        return replace(base, baseline_rejected_at_alpha=base.truth_non_null)
    if mode == "width":
        return replace(
            base,
            ci_interval_width=np.full(len(base.hypothesis_ids), 1.2),
        )
    raise ValueError("unknown unit noninferiority replay mode")


def _attested_noninferiority_campaign(mode: str):  # type: ignore[no-untyped-def]
    registration = _register_calibration_generator(
        campaign_kind=CalibrationCampaignKind.G3_FREQUENCY,
        generator_kind=f"unit-g3f-{mode}-noninferiority-generator",
        generator_version="1.0.0",
        code_version="unit-g3f-ni-replay-v1",
        config={"mode": mode},
        source_paths=(Path(__file__),),
        replay=_replay_noninferiority_case,
        profile=CalibrationGeneratorProfile.RELEASE_APPROVED,
    )
    base = _protocol()
    protocol = G3FrequencyCalibrationProtocol(
        campaign_name=f"g3f-attested-{mode}-ni-unit-campaign",
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
    return summarize_attested_g3_frequency_calibration_campaign(
        protocol,
        ledgers,
        registry=registry,
        n_jobs=2,
    )


def test_scenario_evidence_campaign_and_gate_are_producer_owned() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        G3CalibrationScenarioResult()
    with pytest.raises(TypeError, match="producer-owned"):
        G3FrequencyCalibrationEvidence()
    with pytest.raises(TypeError, match="producer-owned"):
        G3FrequencyCalibrationGate()


def test_protocol_identity_freezes_noninferiority_baseline_and_thresholds() -> None:
    alternative = replace(
        _protocol(),
        noninferiority_baseline_id="g3f-alternative-baseline",
    )

    assert alternative.protocol_id != _protocol().protocol_id
    payload = _protocol().to_dict()
    assert payload["noninferiority_baseline_id"] == "g3f-unit-baseline"
    assert payload["power_noninferiority_margin"] == -0.05
    assert payload["interval_width_noninferiority_margin"] == -0.10
    assert payload["power_noninferiority_direction"] == (
        "larger_is_better_lower_bound_at_least_margin"
    )


def test_campaign_derives_the_exact_fixed_grid_from_raw_ledgers() -> None:
    campaign = summarize_g3_frequency_calibration_campaign(_protocol(), _replicates())
    gate = build_g3_frequency_calibration_gate(campaign.evidence)

    assert len(campaign.replicates) == 30
    assert len(campaign.scenarios) == 81
    assert all(
        item.status is G3CalibrationScenarioStatus.OBSERVED
        for item in campaign.scenarios
    )
    assert len(campaign.permutation_diagnostics) == 3
    assert gate.status is G3FrequencyGateStatus.NOT_ESTIMABLE
    assert gate.reason_code == "g3_frequency_scenario_replicates_below_1000"
    assert gate.hypothesis_universe_id == _protocol().hypothesis_universe_id
    assert gate.hierarchical_procedure_id == (_protocol().hierarchical_procedure_id)
    assert gate.noninferiority_baseline_id == "g3f-unit-baseline"


def test_power_and_interval_width_are_derived_from_hypothesis_ledgers() -> None:
    replicates = list(_replicates())
    target_index = next(
        index
        for index, item in enumerate(replicates)
        if item.dependence_structure
        is G3FrequencyDependenceStructure.INDEPENDENT_HYPOTHESES
        and item.non_null_prevalence == 0.005
        and item.replicate_index == 0
    )
    target = replicates[target_index]
    candidate_rejected = target.rejected_at_alpha.copy()
    candidate_rejected[target.truth_non_null] = True
    replicates[target_index] = replace(
        target,
        rejected_at_alpha=candidate_rejected,
        ci_interval_width=np.full(len(target.hypothesis_ids), 1.2),
    )

    campaign = summarize_g3_frequency_calibration_campaign(_protocol(), replicates)

    def result(metric: G3CalibrationMetric):
        return next(
            item
            for item in campaign.scenarios
            if item.dependence_structure
            == G3FrequencyDependenceStructure.INDEPENDENT_HYPOTHESES.value
            and item.non_null_prevalence == 0.005
            and item.metric is metric
        )

    power = result(G3CalibrationMetric.POWER_NONINFERIORITY_LOWER)
    width = result(G3CalibrationMetric.INTERVAL_WIDTH_NONINFERIORITY_LOWER)
    assert power.point_estimate == pytest.approx(0.5)
    assert power.n_metric_replicates == 2
    assert power.noninferiority_baseline_id == "g3f-unit-baseline"
    assert power.noninferiority_margin == -0.05
    assert width.point_estimate == pytest.approx(-0.1)
    assert width.noninferiority_margin == -0.10


def test_interval_width_and_baseline_decision_ledgers_are_strict() -> None:
    replicate = _replicate(
        G3FrequencyDependenceStructure.INDEPENDENT_HYPOTHESES,
        0.005,
        0,
    )
    with pytest.raises(ValueError, match="must be paired"):
        replace(
            replicate,
            ci_interval_width=np.full(len(replicate.hypothesis_ids), np.nan),
        )
    with pytest.raises(ValueError, match="baseline > 0"):
        replace(
            replicate,
            baseline_ci_interval_width=np.zeros(len(replicate.hypothesis_ids)),
        )

    child_index = next(
        index
        for index, role in enumerate(replicate.hypothesis_roles)
        if role is HypothesisRole.SECONDARY
    )
    baseline_rejected = replicate.baseline_rejected_at_alpha.copy()
    baseline_rejected[child_index] = True
    with pytest.raises(ValueError, match="baseline-rejected parent"):
        replace(replicate, baseline_rejected_at_alpha=baseline_rejected)


def test_public_raw_ledgers_cannot_self_attest_the_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = summarize_g3_frequency_calibration_campaign(_protocol(), _replicates())
    monkeypatch.setattr(_g3f_calibration, "_MIN_CALIBRATION_REPLICATES", 2)

    gate = build_g3_frequency_calibration_gate(campaign.evidence)

    assert _protocol().generator_release_verified is False
    assert campaign.evidence.generator_verified is False
    assert gate.generator_verified is False
    assert gate.status is G3FrequencyGateStatus.NOT_ESTIMABLE
    assert gate.reason_code == "g3_frequency_calibration_generator_unverified"


def test_campaign_is_invariant_to_replicate_and_hypothesis_row_order() -> None:
    canonical = summarize_g3_frequency_calibration_campaign(_protocol(), _replicates())
    reordered = summarize_g3_frequency_calibration_campaign(
        _protocol(), _replicates(reverse=True)
    )

    assert reordered.campaign_id == canonical.campaign_id
    assert reordered.evidence.evidence_id == canonical.evidence.evidence_id


def test_metrics_are_computed_from_truth_and_rejection_ledgers() -> None:
    replicates = list(_replicates())
    global_index = next(
        index
        for index, item in enumerate(replicates)
        if item.dependence_structure
        is G3FrequencyDependenceStructure.INDEPENDENT_HYPOTHESES
        and item.non_null_prevalence is None
        and item.replicate_index == 0
    )
    global_row = replicates[global_index]
    global_rejected = global_row.rejected_at_alpha.copy()
    primary_index = next(
        index
        for index, role in enumerate(global_row.hypothesis_roles)
        if role is HypothesisRole.PRIMARY
    )
    global_rejected[primary_index] = True
    replicates[global_index] = replace(global_row, rejected_at_alpha=global_rejected)

    mixed_index = next(
        index
        for index, item in enumerate(replicates)
        if item.dependence_structure
        is G3FrequencyDependenceStructure.INDEPENDENT_HYPOTHESES
        and item.non_null_prevalence == 0.005
        and item.replicate_index == 0
    )
    mixed_row = replicates[mixed_index]
    true_primary_index = int(np.flatnonzero(mixed_row.truth_non_null)[0])
    true_primary_id = mixed_row.hypothesis_ids[true_primary_index]
    null_child_index = next(
        index
        for index, parent in enumerate(mixed_row.parent_hypothesis_ids)
        if parent == true_primary_id
    )
    mixed_rejected = mixed_row.rejected_at_alpha.copy()
    mixed_rejected[[true_primary_index, null_child_index]] = True
    replicates[mixed_index] = replace(mixed_row, rejected_at_alpha=mixed_rejected)
    campaign = summarize_g3_frequency_calibration_campaign(_protocol(), replicates)

    def point(metric: G3CalibrationMetric, prevalence: float | None) -> float:
        result = next(
            item
            for item in campaign.scenarios
            if item.dependence_structure
            == G3FrequencyDependenceStructure.INDEPENDENT_HYPOTHESES.value
            and item.non_null_prevalence == prevalence
            and item.metric is metric
        )
        assert result.point_estimate is not None
        return result.point_estimate

    assert point(G3CalibrationMetric.NULL_TYPE_I_UPPER, None) == 0.5
    assert point(G3CalibrationMetric.MIXED_FDR_UPPER, 0.005) == 0.25
    assert point(G3CalibrationMetric.PRIMARY_FDR_UPPER, 0.005) == 0.0
    assert point(G3CalibrationMetric.SELECTIVE_CHILD_FDR_UPPER, 0.005) == 0.5


def test_power_and_width_noninferiority_margins_block_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_g3f_calibration, "_MIN_CALIBRATION_REPLICATES", 2)
    monkeypatch.setattr(_g3f_calibration, "_TYPE_I_UPPER_LIMIT", 1.0)
    monkeypatch.setattr(_g3f_calibration, "_MIXED_FDR_UPPER_LIMIT", 1.0)
    monkeypatch.setattr(_g3f_calibration, "_PRIMARY_FDR_UPPER_LIMIT", 1.0)
    monkeypatch.setattr(_g3f_calibration, "_SELECTIVE_CHILD_FDR_UPPER_LIMIT", 1.0)
    monkeypatch.setattr(_g3f_calibration, "_COVERAGE_LOWER_LIMIT", 0.0)
    power_gate = build_g3_frequency_calibration_gate(
        _attested_noninferiority_campaign("power").evidence
    )
    assert power_gate.status is G3FrequencyGateStatus.FAILED
    assert power_gate.reason_code == (
        "g3_frequency_power_noninferiority_lower_below_margin"
    )
    assert power_gate.power_noninferiority_lower_minimum == pytest.approx(-1.0)

    width_gate = build_g3_frequency_calibration_gate(
        _attested_noninferiority_campaign("width").evidence
    )
    assert width_gate.status is G3FrequencyGateStatus.FAILED
    assert width_gate.reason_code == (
        "g3_frequency_interval_width_noninferiority_lower_below_margin"
    )
    assert width_gate.interval_width_noninferiority_lower_minimum == pytest.approx(
        -0.2
    )


def test_missing_interval_width_fails_closed() -> None:
    replicates = list(_replicates())
    target = replicates[0]
    replicates[0] = replace(
        target,
        ci_interval_width=np.full(len(target.hypothesis_ids), np.nan),
        baseline_ci_interval_width=np.full(len(target.hypothesis_ids), np.nan),
    )

    campaign = summarize_g3_frequency_calibration_campaign(_protocol(), replicates)
    affected = tuple(
        item
        for item in campaign.scenarios
        if item.scenario_id
        == _protocol().scenario_id(
            target.dependence_structure, target.non_null_prevalence
        )
    )
    assert affected
    assert all(
        item.status is G3CalibrationScenarioStatus.NOT_ESTIMABLE
        and item.reason_code == "g3_frequency_interval_width_coverage_incomplete"
        for item in affected
    )
    gate = build_g3_frequency_calibration_gate(campaign.evidence)
    assert gate.status is G3FrequencyGateStatus.NOT_ESTIMABLE
    assert gate.reason_code == "g3_frequency_calibration_scenario_not_estimable"


def test_missing_cell_and_nonobserved_replicate_fail_closed() -> None:
    complete = _replicates()
    missing = summarize_g3_frequency_calibration_campaign(
        _protocol(),
        tuple(
            item
            for item in complete
            if not (
                item.dependence_structure
                is G3FrequencyDependenceStructure.NEGATIVE_WITHIN_FAMILY_BLOCK
                and item.non_null_prevalence == 0.10
            )
        ),
    )
    missing_gate = build_g3_frequency_calibration_gate(missing.evidence)
    assert missing_gate.status is G3FrequencyGateStatus.NOT_ESTIMABLE
    assert missing_gate.reason_code == (
        "g3_frequency_calibration_scenario_not_estimable"
    )

    replaced = tuple(complete)
    target = next(
        item
        for item in replaced
        if item.dependence_structure
        is G3FrequencyDependenceStructure.INDEPENDENT_HYPOTHESES
        and item.non_null_prevalence == 0.01
        and item.replicate_index == 0
    )
    failed = _replicate(
        G3FrequencyDependenceStructure.INDEPENDENT_HYPOTHESES,
        0.01,
        0,
        status=G3FrequencyReplicateStatus.FAILED,
    )
    nonobserved = summarize_g3_frequency_calibration_campaign(
        _protocol(), tuple(failed if item is target else item for item in replaced)
    )
    assert (
        build_g3_frequency_calibration_gate(nonobserved.evidence).status
        is G3FrequencyGateStatus.NOT_ESTIMABLE
    )


def test_wrong_seed_and_duplicate_replicate_are_rejected_or_ne() -> None:
    replicates = _replicates()
    target = replicates[0]
    wrong_seed = replace(target, seed_lineage=SeedLineage(999))
    campaign = summarize_g3_frequency_calibration_campaign(
        _protocol(), (wrong_seed, *replicates[1:])
    )
    assert (
        build_g3_frequency_calibration_gate(campaign.evidence).status
        is G3FrequencyGateStatus.NOT_ESTIMABLE
    )

    with pytest.raises(ContractError) as error:
        summarize_g3_frequency_calibration_campaign(
            _protocol(), (*replicates, replicates[0])
        )
    assert error.value.details.code == ("duplicate_g3_frequency_calibration_replicate")


def test_nested_truth_and_evidence_tampering_are_rejected() -> None:
    replicate = _replicate(
        G3FrequencyDependenceStructure.INDEPENDENT_HYPOTHESES, 0.005, 0
    )
    truth = replicate.truth_non_null.copy()
    child_index = next(
        index
        for index, role in enumerate(replicate.hypothesis_roles)
        if role is HypothesisRole.SECONDARY
    )
    truth[:] = False
    truth[child_index] = True
    with pytest.raises(ValueError, match="non-null parent"):
        replace(replicate, truth_non_null=truth)

    campaign = summarize_g3_frequency_calibration_campaign(_protocol(), _replicates())
    object.__setattr__(campaign.evidence.scenarios[0], "one_sided_bound", 0.99)
    with pytest.raises(ContractError) as error:
        build_g3_frequency_calibration_gate(campaign.evidence)
    assert error.value.details.code == "g3_calibration_evidence_integrity_violation"
