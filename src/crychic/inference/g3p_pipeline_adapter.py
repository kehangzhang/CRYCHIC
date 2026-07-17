"""Strict adapter from the active-probability workflow to a G3-P ledger.

The adapter deliberately has no array or source-ID arguments.  Candidate
probabilities and their lineage come from the producer-owned runtime result;
simulation truth is frozen separately against the exact biological candidate
objects in the runtime universe.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from crychic.core import CommunicationMode, ContractError, SeedLineage, stable_id
from crychic.scoring import ActiveEdgeCandidate, FrozenActiveEdgeUniverse

from .active_probability import ActiveEdgeScoreStatus, LocalFDRFitStatus
from .calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationReplayRequest,
)
from .g3p_calibration import (
    G3PCalibrationProtocol,
    G3PCandidateStatus,
    G3PDependenceStructure,
    G3PReplicatePredictions,
    G3PReplicateStatus,
)

if TYPE_CHECKING:
    from crychic.workflow.active_probability_pipeline import (
        ActiveProbabilityPipelineResult,
    )

_SCHEMA_VERSION = "1.0.0"
_TRUTH_PRODUCER = "crychic.inference.g3p_simulation_truth.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _contract_error(
    message: str,
    *,
    code: str,
    field: str,
    remediation: str,
) -> ContractError:
    return ContractError(
        message,
        code=code,
        field=field,
        remediation=remediation,
    )


def _candidate_key(candidate: ActiveEdgeCandidate) -> tuple[str, ...]:
    """Return every field that defines one frozen biological candidate."""

    return (
        candidate.contrast_id,
        candidate.context_id,
        candidate.sender,
        candidate.receiver,
        candidate.interaction_id,
        candidate.driver_id,
        CommunicationMode(candidate.mode).value,
    )


def _require_request_binding(
    protocol: G3PCalibrationProtocol,
    request: CalibrationReplayRequest,
) -> tuple[G3PDependenceStructure, float]:
    if not isinstance(protocol, G3PCalibrationProtocol):
        raise TypeError("protocol must be G3PCalibrationProtocol")
    if not isinstance(request, CalibrationReplayRequest):
        raise TypeError("request must be CalibrationReplayRequest")
    protocol._require_intact()
    repeated = CalibrationReplayRequest(
        campaign_kind=request.campaign_kind,
        protocol_id=request.protocol_id,
        dependence_structure=request.dependence_structure,
        non_null_prevalence=request.non_null_prevalence,
        replicate_index=request.replicate_index,
        seed_lineage=request.seed_lineage,
    )
    if repeated.request_id != request.request_id:
        raise _contract_error(
            "G3-P replay request failed integrity validation",
            code="g3p_replay_request_integrity_violation",
            field="request_id",
            remediation="Recreate the request from the intact G3-P protocol",
        )
    if (
        request.campaign_kind is not CalibrationCampaignKind.G3_PROBABILITY
        or request.protocol_id != protocol.protocol_id
        or request.non_null_prevalence is None
    ):
        raise _contract_error(
            "G3-P replay request does not bind the supplied protocol",
            code="g3p_replay_request_binding_mismatch",
            field="campaign_kind,protocol_id,non_null_prevalence",
            remediation="Use one request emitted for this exact G3-P protocol cell",
        )
    try:
        dependence = G3PDependenceStructure(request.dependence_structure)
    except ValueError as error:
        raise _contract_error(
            "G3-P replay request uses an unsupported dependence structure",
            code="g3p_replay_request_binding_mismatch",
            field="dependence_structure",
            remediation="Use a dependence cell declared by the frozen protocol",
        ) from error
    prevalence = float(request.non_null_prevalence)
    if (
        dependence.value not in protocol.required_dependence_structures
        or prevalence not in protocol.required_prevalences
    ):
        raise _contract_error(
            "G3-P replay request is outside the frozen protocol grid",
            code="g3p_replay_request_binding_mismatch",
            field="dependence_structure,non_null_prevalence",
            remediation="Use a dependence/prevalence cell from the frozen protocol",
        )
    expected_seed = protocol.replicate_seed_lineage(
        dependence,
        prevalence,
        request.replicate_index,
    )
    if request.seed_lineage.to_dict() != expected_seed.to_dict():
        raise _contract_error(
            "G3-P replay request seed does not match the protocol lineage",
            code="g3p_replay_seed_mismatch",
            field="seed_lineage",
            remediation="Derive the replicate seed from the exact protocol cell",
        )
    return dependence, prevalence


@dataclass(frozen=True, slots=True, init=False)
class G3PSimulationTruth:
    """Producer-owned active truth for one exact simulated candidate universe."""

    protocol_id: str
    request_id: str
    dependence_structure: G3PDependenceStructure
    non_null_prevalence: float
    replicate_index: int
    seed_lineage: SeedLineage
    generation_id: str
    source_digest: str
    source_candidate_universe_id: str
    candidate_edge_ids: tuple[str, ...]
    biological_candidate_keys: tuple[tuple[str, ...], ...]
    truth_active: tuple[bool, ...]
    truth_id: str
    _protocol: G3PCalibrationProtocol = field(repr=False)
    _request: CalibrationReplayRequest = field(repr=False)
    _universe: FrozenActiveEdgeUniverse = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "G3PSimulationTruth is producer-owned; use "
            "freeze_g3p_simulation_truth()"
        )

    @classmethod
    def _from_universe(
        cls,
        *,
        protocol: G3PCalibrationProtocol,
        request: CalibrationReplayRequest,
        universe: FrozenActiveEdgeUniverse,
        truth_by_candidate: Mapping[ActiveEdgeCandidate, bool | np.bool_],
        generation_id: str,
        source_digest: str,
    ) -> G3PSimulationTruth:
        dependence, prevalence = _require_request_binding(protocol, request)
        if not isinstance(universe, FrozenActiveEdgeUniverse):
            raise TypeError("universe must be FrozenActiveEdgeUniverse")
        universe._require_intact()
        if (
            universe.candidate_universe_policy_id
            != protocol.candidate_universe_policy_id
            or universe.score_version != protocol.score_version
        ):
            raise _contract_error(
                "Simulation truth universe does not bind the G3-P protocol",
                code="g3p_truth_protocol_binding_mismatch",
                field="candidate_universe_policy_id,score_version",
                remediation="Generate truth for the exact protocol-bound universe",
            )
        if not isinstance(truth_by_candidate, Mapping):
            raise TypeError(
                "truth_by_candidate must map typed ActiveEdgeCandidate values to bool"
            )
        truth_items = tuple(truth_by_candidate.items())
        if any(
            not isinstance(candidate, ActiveEdgeCandidate)
            for candidate, _ in truth_items
        ):
            raise TypeError("truth keys must be typed ActiveEdgeCandidate values")
        if any(
            not isinstance(value, (bool, np.bool_)) for _, value in truth_items
        ):
            raise TypeError("simulation truth values must be boolean")
        supplied_by_id: dict[str, tuple[ActiveEdgeCandidate, bool]] = {}
        for candidate, active in truth_items:
            candidate._require_intact()
            if candidate.candidate_edge_id in supplied_by_id:
                raise _contract_error(
                    "Simulation truth contains a duplicate candidate identity",
                    code="g3p_truth_candidate_coverage_mismatch",
                    field="truth_by_candidate",
                    remediation="Emit one boolean truth value per universe candidate",
                )
            supplied_by_id[candidate.candidate_edge_id] = (candidate, bool(active))
        expected_ids = set(universe.candidate_edge_ids)
        if set(supplied_by_id) != expected_ids:
            raise _contract_error(
                "Simulation truth does not exactly cover the frozen universe",
                code="g3p_truth_candidate_coverage_mismatch",
                field="truth_by_candidate",
                remediation="Emit one boolean truth value for every exact candidate",
            )
        truth: list[bool] = []
        keys: list[tuple[str, ...]] = []
        for expected in universe.candidates:
            supplied, active = supplied_by_id[expected.candidate_edge_id]
            if _candidate_key(supplied) != _candidate_key(expected):
                raise _contract_error(
                    "Simulation truth candidate key differs from the frozen universe",
                    code="g3p_truth_candidate_key_mismatch",
                    field="biological_candidate_key",
                    remediation="Use the exact typed candidates from the universe",
                )
            keys.append(_candidate_key(expected))
            truth.append(active)
        self = object.__new__(cls)
        values: Mapping[str, object] = {
            "protocol_id": protocol.protocol_id,
            "request_id": request.request_id,
            "dependence_structure": dependence,
            "non_null_prevalence": prevalence,
            "replicate_index": request.replicate_index,
            "seed_lineage": request.seed_lineage,
            "generation_id": _name(generation_id, field_name="generation_id"),
            "source_digest": _sha256(source_digest, field_name="source_digest"),
            "source_candidate_universe_id": universe.universe_id,
            "candidate_edge_ids": universe.candidate_edge_ids,
            "biological_candidate_keys": tuple(keys),
            "truth_active": tuple(truth),
            "_protocol": protocol,
            "_request": request,
            "_universe": universe,
            "_producer_marker": _TRUTH_PRODUCER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "truth_id",
            stable_id(
                "g3p_simulation_truth",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "request_id": self.request_id,
            "dependence_structure": self.dependence_structure.value,
            "non_null_prevalence": self.non_null_prevalence,
            "replicate_index": self.replicate_index,
            "seed_lineage": self.seed_lineage.to_dict(),
            "generation_id": self.generation_id,
            "source_digest": self.source_digest,
            "source_candidate_universe_id": self.source_candidate_universe_id,
            "candidate_edge_ids": list(self.candidate_edge_ids),
            "biological_candidate_keys": [
                list(item) for item in self.biological_candidate_keys
            ],
            "truth_active": list(self.truth_active),
            "producer_marker": _TRUTH_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            repeated = G3PSimulationTruth._from_universe(
                protocol=self._protocol,
                request=self._request,
                universe=self._universe,
                truth_by_candidate={
                    candidate: active
                    for candidate, active in zip(
                        self._universe.candidates,
                        self.truth_active,
                        strict=True,
                    )
                },
                generation_id=self.generation_id,
                source_digest=self.source_digest,
            )
            valid = (
                self._producer_marker == _TRUTH_PRODUCER
                and repeated._identity_payload() == self._identity_payload()
                and repeated.truth_id == self.truth_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-P simulation truth failed integrity validation",
                code="g3p_simulation_truth_integrity_violation",
                field="truth_id",
                remediation="Refreeze truth from the exact simulation universe",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-P simulation truth failed integrity validation",
                code="g3p_simulation_truth_integrity_violation",
                field="truth_id",
                remediation="Refreeze truth from the exact simulation universe",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"truth_id": self.truth_id, **payload}


def freeze_g3p_simulation_truth(
    protocol: G3PCalibrationProtocol,
    request: CalibrationReplayRequest,
    universe: FrozenActiveEdgeUniverse,
    *,
    truth_by_candidate: Mapping[ActiveEdgeCandidate, bool | np.bool_],
    generation_id: str,
    source_digest: str,
) -> G3PSimulationTruth:
    """Freeze complete truth without accepting edge IDs or probabilities."""

    return G3PSimulationTruth._from_universe(
        protocol=protocol,
        request=request,
        universe=universe,
        truth_by_candidate=truth_by_candidate,
        generation_id=generation_id,
        source_digest=source_digest,
    )


def _require_runtime_binding(
    protocol: G3PCalibrationProtocol,
    pipeline_result: ActiveProbabilityPipelineResult,
) -> None:
    universe = pipeline_result.universe
    probability_spec = pipeline_result.active_probability_spec
    null_rerun = pipeline_result.null_rerun
    collection = pipeline_result.probability_collection
    if (
        protocol.active_null_spec_id != null_rerun.active_null_spec_id
        or protocol.active_probability_spec_id != probability_spec.spec_id
        or protocol.score_spec_id != null_rerun.score_spec_id
        or protocol.candidate_universe_policy_id
        != universe.candidate_universe_policy_id
        or protocol.estimator_id != probability_spec.estimator_id
        or protocol.stratum_policy_id != probability_spec.stratum_policy_id
        or protocol.score_version != universe.score_version
    ):
        raise _contract_error(
            "Active-probability pipeline does not bind the G3-P protocol contracts",
            code="g3p_pipeline_protocol_binding_mismatch",
            field=(
                "active_null_spec_id,active_probability_spec_id,score_spec_id,"
                "candidate_universe_policy_id,estimator_id,stratum_policy_id,"
                "score_version"
            ),
            remediation="Run the exact protocol-bound active-probability workflow",
        )
    if (
        collection.candidate_universe_id != universe.universe_id
        or collection.active_null_id != null_rerun.active_null_id
        or collection.distribution_id != null_rerun.distribution.distribution_id
    ):
        raise _contract_error(
            "Active-probability collection has incompatible typed parents",
            code="g3p_pipeline_probability_parent_mismatch",
            field="candidate_universe_id,active_null_id,distribution_id",
            remediation="Use the intact collection returned by this pipeline",
        )


def _nonobserved_reason(
    *,
    point_reason: str | None,
    probability_reason: str | None,
    diagnostic_reason: str | None,
    fallback: str,
) -> str:
    return point_reason or diagnostic_reason or probability_reason or fallback


def build_g3p_replicate_from_pipeline(
    protocol: G3PCalibrationProtocol,
    request: CalibrationReplayRequest,
    pipeline_result: ActiveProbabilityPipelineResult,
    simulation_truth: G3PSimulationTruth,
) -> G3PReplicatePredictions:
    """Derive one G3-P replicate ledger from intact workflow-owned children."""

    # Delayed to avoid an inference/workflow package import cycle.
    from crychic.workflow.active_probability_pipeline import (
        ActiveProbabilityPipelineResult,
    )

    dependence, prevalence = _require_request_binding(protocol, request)
    if not isinstance(pipeline_result, ActiveProbabilityPipelineResult):
        raise TypeError(
            "pipeline_result must be producer-owned ActiveProbabilityPipelineResult"
        )
    if not isinstance(simulation_truth, G3PSimulationTruth):
        raise TypeError("simulation_truth must be producer-owned G3PSimulationTruth")
    pipeline_result._require_intact()
    simulation_truth._require_intact()
    _require_runtime_binding(protocol, pipeline_result)
    universe = pipeline_result.universe
    if (
        simulation_truth.protocol_id != protocol.protocol_id
        or simulation_truth.request_id != request.request_id
        or simulation_truth.dependence_structure is not dependence
        or simulation_truth.non_null_prevalence != prevalence
        or simulation_truth.replicate_index != request.replicate_index
        or simulation_truth.seed_lineage.to_dict() != request.seed_lineage.to_dict()
        or simulation_truth.source_candidate_universe_id != universe.universe_id
        or simulation_truth.candidate_edge_ids != universe.candidate_edge_ids
        or simulation_truth.biological_candidate_keys
        != tuple(_candidate_key(item) for item in universe.candidates)
    ):
        raise _contract_error(
            "Simulation truth does not bind this protocol request and universe",
            code="g3p_pipeline_truth_binding_mismatch",
            field="protocol_id,request_id,seed_lineage,candidate_universe_id",
            remediation="Use truth frozen for this exact pipeline replicate",
        )

    collection = pipeline_result.probability_collection
    records_by_edge = {item.candidate_edge_id: item for item in collection.records}
    if (
        len(records_by_edge) != len(collection.records)
        or set(records_by_edge) != set(universe.candidate_edge_ids)
    ):
        raise _contract_error(
            "Probability records do not exactly cover the frozen universe",
            code="g3p_pipeline_candidate_coverage_mismatch",
            field="candidate_edge_id",
            remediation="Recompute the complete producer-owned probability collection",
        )
    point_by_edge = {
        item.candidate_edge_id: item
        for item in pipeline_result.null_rerun.distribution.point_records
    }
    diagnostics_by_id = {
        item.diagnostic_id: item for item in collection.diagnostics
    }
    has_null_failure = any(
        item.status.value == "failed" for item in pipeline_result.null_rerun.records
    )
    truth_by_edge = dict(
        zip(
            simulation_truth.candidate_edge_ids,
            simulation_truth.truth_active,
            strict=True,
        )
    )
    probabilities: list[float] = []
    statuses: list[G3PCandidateStatus] = []
    reasons: list[str | None] = []
    strata: list[str] = []
    for candidate in universe.candidates:
        edge_id = candidate.candidate_edge_id
        record = records_by_edge[edge_id]
        point = point_by_edge.get(edge_id)
        diagnostic = diagnostics_by_id.get(record.local_fdr_diagnostic_id)
        expected_stratum = candidate.stratum_id(score_version=universe.score_version)
        if (
            point is None
            or diagnostic is None
            or record.point_record_id != point.record_id
            or record.stratum_id != point.stratum_id
            or record.stratum_id != expected_stratum
            or record.score_version != universe.score_version
        ):
            raise _contract_error(
                "Probability record does not bind its typed point and stratum parents",
                code="g3p_pipeline_probability_parent_mismatch",
                field="point_record_id,stratum_id,score_version",
                remediation=(
                    "Use records from the intact pipeline probability collection"
                ),
            )
        strata.append(record.stratum_id)
        probability = record.candidate_comm_probability
        diagnostic_failed = diagnostic.status is LocalFDRFitStatus.FAILED
        if record.point_status is ActiveEdgeScoreStatus.FAILED:
            candidate_status = G3PCandidateStatus.FAILED
        elif probability is None:
            candidate_status = (
                G3PCandidateStatus.FAILED
                if has_null_failure or diagnostic_failed
                else G3PCandidateStatus.NOT_ESTIMABLE
            )
        elif record.point_status is ActiveEdgeScoreStatus.NOT_ESTIMABLE:
            raise _contract_error(
                "Not-estimable point record unexpectedly carries a probability",
                code="g3p_pipeline_probability_state_mismatch",
                field="candidate_comm_probability",
                remediation=(
                    "Recompute probability records from the intact distribution"
                ),
            )
        else:
            numeric_probability = float(probability)
            if (
                not math.isfinite(numeric_probability)
                or not 0.0 <= numeric_probability <= 1.0
            ):
                raise _contract_error(
                    "Candidate probability is outside the unit interval",
                    code="g3p_pipeline_probability_state_mismatch",
                    field="candidate_comm_probability",
                    remediation="Recompute the producer-owned probability collection",
                )
            if record.point_status is ActiveEdgeScoreStatus.STRUCTURAL_ZERO:
                if numeric_probability != 0.0:
                    raise _contract_error(
                        "Structural-zero point record has nonzero probability",
                        code="g3p_pipeline_probability_state_mismatch",
                        field="candidate_comm_probability",
                        remediation=(
                            "Recompute from the intact structural-zero point row"
                        ),
                    )
                candidate_status = (
                    G3PCandidateStatus.ELIGIBLE
                    if truth_by_edge[edge_id]
                    else G3PCandidateStatus.STRUCTURAL_ZERO
                )
            else:
                candidate_status = G3PCandidateStatus.ELIGIBLE

        if candidate_status is G3PCandidateStatus.ELIGIBLE:
            assert probability is not None
            probabilities.append(float(probability))
            reasons.append(None)
        elif candidate_status is G3PCandidateStatus.STRUCTURAL_ZERO:
            probabilities.append(0.0)
            reasons.append(
                _nonobserved_reason(
                    point_reason=record.point_reason_code,
                    probability_reason=record.probability_reason_code,
                    diagnostic_reason=diagnostic.reason_code,
                    fallback="active_edge_structural_zero",
                )
            )
        else:
            probabilities.append(float("nan"))
            reasons.append(
                _nonobserved_reason(
                    point_reason=record.point_reason_code,
                    probability_reason=record.probability_reason_code,
                    diagnostic_reason=diagnostic.reason_code,
                    fallback=(
                        "active_probability_pipeline_failed"
                        if candidate_status is G3PCandidateStatus.FAILED
                        else "active_probability_not_estimable"
                    ),
                )
            )
        statuses.append(candidate_status)

    if G3PCandidateStatus.FAILED in statuses:
        replicate_status = G3PReplicateStatus.FAILED
        replicate_reason = "active_probability_pipeline_failed"
    elif G3PCandidateStatus.NOT_ESTIMABLE in statuses:
        replicate_status = G3PReplicateStatus.NOT_ESTIMABLE
        replicate_reason = "active_probability_not_estimable"
    else:
        replicate_status = G3PReplicateStatus.OBSERVED
        replicate_reason = None
    return G3PReplicatePredictions(
        protocol_id=protocol.protocol_id,
        dependence_structure=dependence,
        non_null_prevalence=prevalence,
        replicate_index=request.replicate_index,
        seed_lineage=request.seed_lineage,
        source_active_null_id=pipeline_result.null_rerun.active_null_id,
        source_candidate_universe_id=universe.universe_id,
        source_probability_collection_id=collection.collection_id,
        candidate_edge_ids=universe.candidate_edge_ids,
        stratum_ids=tuple(strata),
        truth_active=np.asarray(simulation_truth.truth_active, dtype=bool),
        candidate_probabilities=np.asarray(probabilities, dtype=float),
        candidate_statuses=tuple(statuses),
        candidate_reason_codes=tuple(reasons),
        status=replicate_status,
        reason_code=replicate_reason,
    )


__all__ = [
    "G3PSimulationTruth",
    "build_g3p_replicate_from_pipeline",
    "freeze_g3p_simulation_truth",
]
