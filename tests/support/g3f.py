"""Test-only producer-derived G3-F release gates."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from crychic.core import SeedLineage
from crychic.inference.calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationGeneratorProfile,
    CalibrationReplayRequest,
    _freeze_calibration_replay_registry,
    _register_calibration_generator,
)
from crychic.inference.g3f_calibration import (
    G3FrequencyCalibrationGate,
    G3FrequencyCalibrationProtocol,
    G3FrequencyCalibrationReplicate,
    G3FrequencyDependenceStructure,
    build_g3_frequency_calibration_gate,
    summarize_attested_g3_frequency_calibration_campaign,
)
from crychic.inference.hierarchical import (
    FrozenHierarchicalFDRSpec,
    freeze_hierarchical_fdr_spec,
)
from crychic.inference.hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisPrefilterStatus,
    HypothesisRole,
)

_PREVALENCES = (0.005, 0.01, 0.05, 0.10)
_N_REPLICATES = 1_000
_INSUFFICIENT_N_REPLICATES = 500
_CACHE: dict[tuple[str, str, int], G3FrequencyCalibrationGate] = {}
_TRUTH_CACHE: dict[tuple[str, float, int], tuple[NDArray[np.bool_], ...]] = {}


def _columns(
    universe: FrozenHypothesisUniverse,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str | None, ...],
    tuple[bool, ...],
]:
    parents = {
        item.hypothesis_key: item.hypothesis_id
        for item in universe.declarations
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
                item.prefilter_status is HypothesisPrefilterStatus.INCLUDED,
            )
            for item in universe.declarations
        )
    )
    return (
        tuple(item[0] for item in rows),
        tuple(item[1] for item in rows),
        tuple(item[2] for item in rows),
        tuple(item[3] for item in rows),
    )


def _truth_ledgers(
    *,
    roles: tuple[str, ...],
    parents: tuple[str | None, ...],
    hypothesis_ids: tuple[str, ...],
    included: tuple[bool, ...],
    prevalence: float,
    n_replicates: int,
) -> tuple[NDArray[np.bool_], ...]:
    output: list[NDArray[np.bool_]] = [
        np.zeros(len(hypothesis_ids), dtype=bool) for _ in range(n_replicates)
    ]
    primary_positions = tuple(
        index
        for index, role in enumerate(roles)
        if role == HypothesisRole.PRIMARY.value and included[index]
    )
    children_by_parent = {
        hypothesis_ids[position]: tuple(
            index
            for index, parent in enumerate(parents)
            if parent == hypothesis_ids[position] and included[index]
        )
        for position in primary_positions
    }
    increments: list[tuple[int, int]] = []
    for replicate_index in range(n_replicates):
        for primary_position in primary_positions:
            increments.append((replicate_index, primary_position))
            increments.extend(
                (replicate_index, child_position)
                for child_position in children_by_parent[
                    hypothesis_ids[primary_position]
                ]
            )
    target = round(prevalence * n_replicates * len(hypothesis_ids))
    if target > len(increments):
        raise ValueError("test universe cannot realize the requested prevalence")
    for replicate_index, position in increments[:target]:
        output[replicate_index][position] = True
        parent_id = parents[position]
        if parent_id is not None:
            output[replicate_index][hypothesis_ids.index(parent_id)] = True
    return tuple(output)


def _truth_for_protocol(
    protocol: G3FrequencyCalibrationProtocol,
    prevalence: float,
    n_replicates: int,
) -> tuple[NDArray[np.bool_], ...]:
    key = (protocol.hypothesis_universe_id, prevalence, n_replicates)
    cached = _TRUTH_CACHE.get(key)
    if cached is not None:
        return cached
    ids, roles, parents, included = _columns(protocol.hypothesis_universe)
    result = _truth_ledgers(
        roles=roles,
        parents=parents,
        hypothesis_ids=ids,
        included=included,
        prevalence=prevalence,
        n_replicates=n_replicates,
    )
    _TRUTH_CACHE[key] = result
    return result


def _replay_test_replicate(
    protocol: object,
    request: CalibrationReplayRequest,
    config: Mapping[str, object],
) -> G3FrequencyCalibrationReplicate:
    if not isinstance(protocol, G3FrequencyCalibrationProtocol):
        raise TypeError("test G3-F replay requires G3FrequencyCalibrationProtocol")
    n_replicates = config.get("n_replicates")
    if (
        isinstance(n_replicates, bool)
        or not isinstance(n_replicates, int)
        or request.replicate_index >= n_replicates
    ):
        raise ValueError("invalid test G3-F replay configuration")
    dependence = G3FrequencyDependenceStructure(request.dependence_structure)
    prevalence = request.non_null_prevalence
    ids, roles, parents, included = _columns(protocol.hypothesis_universe)
    truth = (
        np.zeros(len(ids), dtype=bool)
        if prevalence is None
        else _truth_for_protocol(protocol, prevalence, n_replicates)[
            request.replicate_index
        ]
    )
    coverage = np.asarray(included, dtype=bool)
    interval_width = np.where(coverage, 1.0, np.nan)
    return G3FrequencyCalibrationReplicate(
        protocol_id=protocol.protocol_id,
        dependence_structure=dependence,
        non_null_prevalence=prevalence,
        replicate_index=request.replicate_index,
        seed_lineage=request.seed_lineage,
        source_artifact_id=(
            f"test-source:{dependence.value}:{prevalence}:"
            f"{request.replicate_index:04d}"
        ),
        hypothesis_ids=ids,
        hypothesis_roles=roles,
        parent_hypothesis_ids=parents,
        truth_non_null=truth,
        rejected_at_alpha=truth,
        baseline_rejected_at_alpha=truth,
        ci_contains_truth=coverage,
        ci_interval_width=interval_width,
        baseline_ci_interval_width=interval_width,
        permutation_null_p_value=(
            (request.replicate_index + 0.5) / n_replicates
            if prevalence is None
            else None
        ),
    )


@cache
def _test_generator_registration(n_replicates: int):  # type: ignore[no-untyped-def]
    return _register_calibration_generator(
        campaign_kind=CalibrationCampaignKind.G3_FREQUENCY,
        generator_kind="test-perfect-frequency-generator",
        generator_version="1.0.0",
        code_version="test-support-g3f-v1",
        config={"n_replicates": n_replicates},
        source_paths=(Path(__file__),),
        replay=_replay_test_replicate,
        profile=CalibrationGeneratorProfile.RELEASE_APPROVED,
    )


@cache
def _test_replay_registry(n_replicates: int):  # type: ignore[no-untyped-def]
    return _freeze_calibration_replay_registry(
        (_test_generator_registration(n_replicates),)
    )


def _g3f_gate(
    universe: FrozenHypothesisUniverse,
    procedure: FrozenHierarchicalFDRSpec,
    *,
    n_replicates: int,
) -> G3FrequencyCalibrationGate:
    key = (universe.universe_id, procedure.procedure_id, n_replicates)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    registration = _test_generator_registration(n_replicates)
    protocol = G3FrequencyCalibrationProtocol(
        campaign_name=f"test-g3f:{universe.universe_id}",
        generator_manifest=registration.manifest,
        noninferiority_baseline_id="test-matched-frequency-baseline-v1",
        hypothesis_universe=universe,
        hierarchical_procedure=procedure,
        seed_lineage=SeedLineage(20260716),
    )
    replicates: list[G3FrequencyCalibrationReplicate] = []
    for dependence in G3FrequencyDependenceStructure:
        for prevalence in (None, *_PREVALENCES):
            for index in range(n_replicates):
                replicates.append(
                    _replay_test_replicate(
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
                )
    campaign = summarize_attested_g3_frequency_calibration_campaign(
        protocol,
        replicates,
        registry=_test_replay_registry(n_replicates),
        n_jobs=2,
    )
    gate = build_g3_frequency_calibration_gate(campaign.evidence)
    _CACHE[key] = gate
    return gate


def passed_g3f_gate(
    universe: FrozenHypothesisUniverse,
    procedure: FrozenHierarchicalFDRSpec | None = None,
) -> G3FrequencyCalibrationGate:
    """Build and cache a genuinely producer-derived passed test gate."""

    resolved_procedure = (
        freeze_hierarchical_fdr_spec() if procedure is None else procedure
    )
    gate = _g3f_gate(
        universe,
        resolved_procedure,
        n_replicates=_N_REPLICATES,
    )
    if not gate.formal_release_allowed or not gate.hierarchical_q_release_allowed:
        raise RuntimeError("test G3-F producer failed to construct a passed gate")
    return gate


def insufficient_g3f_gate(
    universe: FrozenHypothesisUniverse,
    procedure: FrozenHierarchicalFDRSpec | None = None,
) -> G3FrequencyCalibrationGate:
    """Build a producer-derived sub-1,000-replicate gate that must remain NE."""

    resolved_procedure = (
        freeze_hierarchical_fdr_spec() if procedure is None else procedure
    )
    return _g3f_gate(
        universe,
        resolved_procedure,
        n_replicates=_INSUFFICIENT_N_REPLICATES,
    )


__all__ = ["insufficient_g3f_gate", "passed_g3f_gate"]
