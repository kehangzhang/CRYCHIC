"""Producer-owned G3-P probability-calibration campaign contracts.

The release gate is derived from complete replicate-level truth and candidate
probability ledgers.  Callers can declare a protocol and submit replicate
outputs, but cannot directly construct scenario metrics, evidence, or a gate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np

from crychic.core import ContractError, SeedLineage, stable_id

from .calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationGeneratorManifest,
    CalibrationReplayAttestation,
    CalibrationReplayRegistry,
    attest_calibration_replay,
)
from .calibration_metrics import (
    BinaryCalibrationRecord,
    CalibrationNotEstimableError,
    compute_binary_calibration_metrics,
    replicate_cluster_bootstrap_ece_upper_bound,
)

_SCHEMA_VERSION = "3.0.0"
_PROTOCOL_SCHEMA_VERSION = "2.0.0"
_PRODUCER = "crychic.inference.g3p_calibration_campaign.v3"
_GATE_PRODUCER = "crychic.inference.active_probability.g3p_gate.v3"
# These object capabilities prevent accidental API construction; generator
# authenticity still requires the separate replayable attestation boundary.
_PRODUCER_TOKEN = object()
_GATE_PRODUCER_TOKEN = object()
_UNVERIFIED_GENERATOR_TOKEN = object()
_VERIFIED_GENERATOR_TOKEN = object()

_REQUIRED_PREVALENCES = (0.005, 0.01, 0.05, 0.10)
_MIN_CALIBRATION_REPLICATES = 1_000
_MIN_CANDIDATES_PER_STRATUM = 200
_ECE_BINS = 10
_ECE_BOOTSTRAP_RESAMPLES = 10_000
_CONFIDENCE_LEVEL = 0.95
_PROBABILITY_CLIP = 1e-6
_METRIC_WEIGHTING = "candidate_equal_within_replicate_then_replicate_equal_v1"

_BRIER_IMPROVEMENT_MINIMUM = 0.05
_ECE_MAXIMUM = 0.05
_ECE_UPPER_MAXIMUM = 0.07
_CALIBRATION_IN_THE_LARGE_ABS_MAXIMUM = 0.02
_CALIBRATION_SLOPE_BOUNDS = (0.8, 1.2)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_name(value: object, *, field_name: str) -> str | None:
    return None if value is None else _name(value, field_name=field_name)


def _prevalence(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("non_null_prevalence must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "non_null_prevalence must be in the frozen G3-P grid"
        ) from error
    if result not in _REQUIRED_PREVALENCES:
        raise ValueError("non_null_prevalence must be in the frozen G3-P grid")
    return result


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{field_name} must be an integer >= 0")
    return int(value)


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be finite")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if result == 0.0 else result


def _unit(value: object, *, field_name: str) -> float:
    result = _finite(value, field_name=field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1]")
    return result


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


def _array_digest(values: np.ndarray, *, dtype: str) -> str:
    canonical = np.asarray(values, dtype=dtype, order="C")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(dtype.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_float_array(values: object, *, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype="<f8", order="C")
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    canonical = np.array(array, dtype="<f8", copy=True, order="C")
    canonical[canonical == 0.0] = 0.0
    result = np.frombuffer(canonical.tobytes(order="C"), dtype="<f8")
    result.setflags(write=False)
    return cast(np.ndarray, result)


def _immutable_bool_array(values: object, *, field_name: str) -> np.ndarray:
    supplied = tuple(cast(Sequence[object], values))
    if any(not isinstance(item, (bool, np.bool_)) for item in supplied):
        raise ValueError(f"{field_name} must contain boolean values")
    canonical = np.asarray(supplied, dtype="|b1", order="C")
    if canonical.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    result = np.frombuffer(canonical.tobytes(order="C"), dtype="|b1")
    result.setflags(write=False)
    return cast(np.ndarray, result)


class G3PDependenceStructure(StrEnum):
    """The complete ADR-013 v1 dependence grid."""

    INDEPENDENT_CANDIDATE_EDGES = "independent_candidate_edges"
    SENDER_LR_BLOCK_CORRELATED_EDGES = "sender_lr_block_correlated_edges"
    TARGET_DEGREE_CORRELATED_EDGES = "target_degree_correlated_edges"


_REQUIRED_DEPENDENCE_STRUCTURES = tuple(item.value for item in G3PDependenceStructure)


class G3PReplicateStatus(StrEnum):
    """Availability of one complete simulated-dataset candidate ledger."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


class G3PCandidateStatus(StrEnum):
    """Availability of one frozen candidate in a simulated replicate."""

    ELIGIBLE = "eligible"
    STRUCTURAL_ZERO = "structural_zero"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


class G3PCalibrationScenarioStatus(StrEnum):
    """Availability of metrics for one required campaign cell."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"


class G3PGateStatus(StrEnum):
    """Producer-derived G3-P release decision."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_ESTIMABLE = "not_estimable"


@dataclass(frozen=True, slots=True, kw_only=True)
class G3PCalibrationProtocol:
    """Pre-inspection campaign grid and exact portable runtime bindings."""

    campaign_name: str
    generator_manifest: CalibrationGeneratorManifest
    active_null_spec_id: str
    active_probability_spec_id: str
    score_spec_id: str
    candidate_universe_policy_id: str
    estimator_id: str
    stratum_policy_id: str
    score_version: str
    seed_lineage: SeedLineage
    schema_version: str = _PROTOCOL_SCHEMA_VERSION
    protocol_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = {
            name: _name(getattr(self, name), field_name=name)
            for name in (
                "campaign_name",
                "active_null_spec_id",
                "active_probability_spec_id",
                "score_spec_id",
                "candidate_universe_policy_id",
                "estimator_id",
                "stratum_policy_id",
                "score_version",
            )
        }
        if not isinstance(self.generator_manifest, CalibrationGeneratorManifest):
            raise TypeError(
                "generator_manifest must be a CalibrationGeneratorManifest"
            )
        self.generator_manifest._require_intact()
        if (
            self.generator_manifest.campaign_kind
            is not CalibrationCampaignKind.G3_PROBABILITY
        ):
            raise ValueError("G3-P protocol requires a G3 probability generator")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        if self.schema_version != _PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"G3-P protocol schema_version must be {_PROTOCOL_SCHEMA_VERSION}"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "protocol_id",
            stable_id(
                "g3p_calibration_protocol",
                self._identity_payload(),
                schema_version="1",
            ),
        )

    @property
    def generator_id(self) -> str:
        return self.generator_manifest.generator_id

    @property
    def generator_release_verified(self) -> bool:
        """A declaration alone is never release evidence."""

        return False

    @property
    def required_dependence_structures(self) -> tuple[str, ...]:
        return _REQUIRED_DEPENDENCE_STRUCTURES

    @property
    def required_prevalences(self) -> tuple[float, ...]:
        return _REQUIRED_PREVALENCES

    def scenario_id(
        self,
        dependence_structure: G3PDependenceStructure | str,
        non_null_prevalence: float,
    ) -> str:
        dependence = G3PDependenceStructure(dependence_structure)
        prevalence = _prevalence(non_null_prevalence)
        identifier: str = stable_id(
            "g3p_calibration_scenario_cell",
            {
                "protocol_id": self.protocol_id,
                "dependence_structure": dependence.value,
                "non_null_prevalence": prevalence,
            },
            schema_version="1",
        )
        return identifier

    def replicate_seed_lineage(
        self,
        dependence_structure: G3PDependenceStructure | str,
        non_null_prevalence: float,
        replicate_index: int,
    ) -> SeedLineage:
        index = _nonnegative_integer(replicate_index, field_name="replicate_index")
        scenario = self.scenario_id(dependence_structure, non_null_prevalence)
        return self.seed_lineage.derive(
            "g3p_calibration_v1",
            f"scenario_id={scenario}",
            f"replicate_index={index:06d}",
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "campaign_name": self.campaign_name,
            "generator_id": self.generator_id,
            "generator_manifest": self.generator_manifest.to_dict(),
            "active_null_spec_id": self.active_null_spec_id,
            "active_probability_spec_id": self.active_probability_spec_id,
            "score_spec_id": self.score_spec_id,
            "candidate_universe_policy_id": self.candidate_universe_policy_id,
            "estimator_id": self.estimator_id,
            "stratum_policy_id": self.stratum_policy_id,
            "score_version": self.score_version,
            "seed_lineage": self.seed_lineage.to_dict(),
            "required_dependence_structures": list(_REQUIRED_DEPENDENCE_STRUCTURES),
            "required_prevalences": list(_REQUIRED_PREVALENCES),
            "minimum_replicates_per_cell": _MIN_CALIBRATION_REPLICATES,
            "minimum_candidates_per_stratum": _MIN_CANDIDATES_PER_STRATUM,
            "ece_bins": _ECE_BINS,
            "ece_bootstrap_resamples": _ECE_BOOTSTRAP_RESAMPLES,
            "confidence_level": _CONFIDENCE_LEVEL,
            "probability_clip": _PROBABILITY_CLIP,
            "metric_weighting": _METRIC_WEIGHTING,
            "schema_version": self.schema_version,
        }

    def _require_intact(self) -> None:
        try:
            repeated = G3PCalibrationProtocol(
                campaign_name=self.campaign_name,
                generator_manifest=self.generator_manifest,
                active_null_spec_id=self.active_null_spec_id,
                active_probability_spec_id=self.active_probability_spec_id,
                score_spec_id=self.score_spec_id,
                candidate_universe_policy_id=self.candidate_universe_policy_id,
                estimator_id=self.estimator_id,
                stratum_policy_id=self.stratum_policy_id,
                score_version=self.score_version,
                seed_lineage=self.seed_lineage,
                schema_version=self.schema_version,
            )
            valid = (
                repeated.protocol_id == self.protocol_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-P protocol failed integrity validation",
                code="g3p_protocol_integrity_violation",
                field="protocol_id",
                remediation="Recreate the frozen G3-P calibration protocol",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-P protocol failed integrity validation",
                code="g3p_protocol_integrity_violation",
                field="protocol_id",
                remediation="Recreate the frozen G3-P calibration protocol",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"protocol_id": self.protocol_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class G3PReplicatePredictions:
    """Complete eligible-candidate truth/probability ledger for one replicate."""

    protocol_id: str
    dependence_structure: G3PDependenceStructure | str
    non_null_prevalence: float
    replicate_index: int
    seed_lineage: SeedLineage
    source_active_null_id: str
    source_candidate_universe_id: str
    source_probability_collection_id: str | None
    candidate_edge_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    truth_active: np.ndarray
    candidate_probabilities: np.ndarray
    candidate_statuses: tuple[G3PCandidateStatus | str, ...]
    candidate_reason_codes: tuple[str | None, ...]
    status: G3PReplicateStatus | str = G3PReplicateStatus.OBSERVED
    reason_code: str | None = None
    replicate_id: str = field(init=False)

    def __post_init__(self) -> None:
        protocol_id = _name(self.protocol_id, field_name="protocol_id")
        dependence = G3PDependenceStructure(self.dependence_structure)
        prevalence = _prevalence(self.non_null_prevalence)
        index = _nonnegative_integer(self.replicate_index, field_name="replicate_index")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        source_null = _name(
            self.source_active_null_id, field_name="source_active_null_id"
        )
        source_universe = _name(
            self.source_candidate_universe_id,
            field_name="source_candidate_universe_id",
        )
        source_collection = _optional_name(
            self.source_probability_collection_id,
            field_name="source_probability_collection_id",
        )
        candidate_ids = tuple(
            _name(item, field_name="candidate_edge_ids")
            for item in self.candidate_edge_ids
        )
        strata = tuple(
            _name(item, field_name="stratum_ids") for item in self.stratum_ids
        )
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_edge_ids must contain unique identifiers")
        if len(strata) != len(candidate_ids):
            raise ValueError("stratum_ids must align with candidate_edge_ids")
        truth = _immutable_bool_array(self.truth_active, field_name="truth_active")
        probabilities = _immutable_float_array(
            self.candidate_probabilities,
            field_name="candidate_probabilities",
        )
        if len(truth) != len(candidate_ids) or len(probabilities) != len(candidate_ids):
            raise ValueError("truth and probability vectors must cover every candidate")
        candidate_statuses = tuple(
            G3PCandidateStatus(item) for item in self.candidate_statuses
        )
        candidate_reasons = tuple(
            _optional_name(item, field_name="candidate_reason_codes")
            for item in self.candidate_reason_codes
        )
        if len(candidate_statuses) != len(candidate_ids) or len(
            candidate_reasons
        ) != len(candidate_ids):
            raise ValueError(
                "candidate status and reason vectors must cover every candidate"
            )
        for candidate_index, candidate_status in enumerate(candidate_statuses):
            candidate_reason = candidate_reasons[candidate_index]
            probability = float(probabilities[candidate_index])
            if candidate_status is G3PCandidateStatus.ELIGIBLE:
                if candidate_reason is not None or not math.isfinite(probability):
                    raise ValueError(
                        "eligible G3-P candidates require a finite probability "
                        "and no reason"
                    )
                _unit(probability, field_name="candidate_probabilities")
            elif candidate_status is G3PCandidateStatus.STRUCTURAL_ZERO:
                if (
                    candidate_reason is None
                    or probability != 0.0
                    or bool(truth[candidate_index])
                ):
                    raise ValueError(
                        "structural-zero G3-P candidates require false truth, zero "
                        "probability, and a reason"
                    )
            elif candidate_reason is None or not math.isnan(probability):
                raise ValueError(
                    "non-observed G3-P candidates require a NaN probability "
                    "and a reason"
                )
        status = G3PReplicateStatus(self.status)
        reason = _optional_name(self.reason_code, field_name="reason_code")
        candidate_status_set = set(candidate_statuses)
        if status is G3PReplicateStatus.OBSERVED:
            if (
                source_collection is None
                or reason is not None
                or G3PCandidateStatus.NOT_ESTIMABLE in candidate_status_set
                or G3PCandidateStatus.FAILED in candidate_status_set
            ):
                raise ValueError(
                    "observed G3-P replicates require a complete source collection"
                )
        else:
            if reason is None:
                raise ValueError("non-observed G3-P replicates require a reason_code")
            if status is G3PReplicateStatus.NOT_ESTIMABLE and (
                G3PCandidateStatus.NOT_ESTIMABLE not in candidate_status_set
                or G3PCandidateStatus.FAILED in candidate_status_set
            ):
                raise ValueError(
                    "not-estimable G3-P replicates require a not-estimable candidate"
                )
            if status is G3PReplicateStatus.FAILED and (
                G3PCandidateStatus.FAILED not in candidate_status_set
            ):
                raise ValueError(
                    "failed G3-P replicates require at least one failed candidate"
                )
        order = np.argsort(np.asarray(candidate_ids, dtype=str), kind="stable")
        candidate_ids = tuple(candidate_ids[int(item)] for item in order)
        strata = tuple(strata[int(item)] for item in order)
        candidate_statuses = tuple(candidate_statuses[int(item)] for item in order)
        candidate_reasons = tuple(candidate_reasons[int(item)] for item in order)
        truth = _immutable_bool_array(truth[order], field_name="truth_active")
        probabilities = _immutable_float_array(
            probabilities[order], field_name="candidate_probabilities"
        )
        values: Mapping[str, object] = {
            "protocol_id": protocol_id,
            "dependence_structure": dependence,
            "non_null_prevalence": prevalence,
            "replicate_index": index,
            "source_active_null_id": source_null,
            "source_candidate_universe_id": source_universe,
            "source_probability_collection_id": source_collection,
            "candidate_edge_ids": candidate_ids,
            "stratum_ids": strata,
            "truth_active": truth,
            "candidate_probabilities": probabilities,
            "candidate_statuses": candidate_statuses,
            "candidate_reason_codes": candidate_reasons,
            "status": status,
            "reason_code": reason,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "replicate_id",
            stable_id(
                "g3p_calibration_replicate",
                self._identity_payload(),
                schema_version="1",
            ),
        )

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_edge_ids)

    @property
    def n_eligible_candidates(self) -> int:
        return sum(
            item is G3PCandidateStatus.ELIGIBLE for item in self.candidate_statuses
        )

    @property
    def n_structural_zero_candidates(self) -> int:
        return sum(
            item is G3PCandidateStatus.STRUCTURAL_ZERO
            for item in self.candidate_statuses
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "dependence_structure": G3PDependenceStructure(
                self.dependence_structure
            ).value,
            "non_null_prevalence": self.non_null_prevalence,
            "replicate_index": self.replicate_index,
            "seed_lineage": self.seed_lineage.to_dict(),
            "source_active_null_id": self.source_active_null_id,
            "source_candidate_universe_id": self.source_candidate_universe_id,
            "source_probability_collection_id": self.source_probability_collection_id,
            "candidate_edge_ids": list(self.candidate_edge_ids),
            "stratum_ids": list(self.stratum_ids),
            "truth_active_digest": _array_digest(self.truth_active, dtype="|b1"),
            "candidate_probabilities_digest": _array_digest(
                self.candidate_probabilities, dtype="<f8"
            ),
            "candidate_statuses": [
                G3PCandidateStatus(item).value for item in self.candidate_statuses
            ],
            "candidate_reason_codes": list(self.candidate_reason_codes),
            "status": G3PReplicateStatus(self.status).value,
            "reason_code": self.reason_code,
        }

    def _require_intact(self) -> None:
        try:
            repeated = G3PReplicatePredictions(
                protocol_id=self.protocol_id,
                dependence_structure=self.dependence_structure,
                non_null_prevalence=self.non_null_prevalence,
                replicate_index=self.replicate_index,
                seed_lineage=self.seed_lineage,
                source_active_null_id=self.source_active_null_id,
                source_candidate_universe_id=self.source_candidate_universe_id,
                source_probability_collection_id=self.source_probability_collection_id,
                candidate_edge_ids=self.candidate_edge_ids,
                stratum_ids=self.stratum_ids,
                truth_active=self.truth_active,
                candidate_probabilities=self.candidate_probabilities,
                candidate_statuses=self.candidate_statuses,
                candidate_reason_codes=self.candidate_reason_codes,
                status=self.status,
                reason_code=self.reason_code,
            )
            valid = repeated.replicate_id == self.replicate_id
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-P replicate ledger failed integrity validation",
                code="g3p_replicate_integrity_violation",
                field="replicate_id",
                remediation="Regenerate the complete replicate ledger",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-P replicate ledger failed integrity validation",
                code="g3p_replicate_integrity_violation",
                field="replicate_id",
                remediation="Regenerate the complete replicate ledger",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "replicate_id": self.replicate_id,
            **self._identity_payload(),
            "n_candidates": self.n_candidates,
            "n_eligible_candidates": self.n_eligible_candidates,
            "n_structural_zero_candidates": self.n_structural_zero_candidates,
        }


@dataclass(frozen=True, slots=True, init=False)
class G3PCalibrationStratumResult:
    """Producer-owned metrics for one runtime probability stratum."""

    scenario_id: str
    protocol_id: str
    stratum_id: str
    status: G3PCalibrationScenarioStatus
    reason_code: str | None
    n_replicates: int
    minimum_eligible_candidates: int
    observed_prevalence: float | None
    brier_score: float | None
    prevalence_only_brier_score: float | None
    ece: float | None
    ece_upper_bound: float | None
    calibration_in_the_large: float | None
    calibration_slope: float | None
    bootstrap_seed: int | None
    stratum_result_id: str
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "G3PCalibrationStratumResult is producer-owned; use "
            "summarize_g3p_calibration_campaign()"
        )

    @property
    def brier_relative_improvement(self) -> float | None:
        if self.brier_score is None or self.prevalence_only_brier_score is None:
            return None
        return (self.prevalence_only_brier_score - self.brier_score) / (
            self.prevalence_only_brier_score
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "protocol_id": self.protocol_id,
            "stratum_id": self.stratum_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_replicates": self.n_replicates,
            "minimum_eligible_candidates": self.minimum_eligible_candidates,
            "observed_prevalence": self.observed_prevalence,
            "brier_score": self.brier_score,
            "prevalence_only_brier_score": self.prevalence_only_brier_score,
            "brier_relative_improvement": self.brier_relative_improvement,
            "ece": self.ece,
            "ece_upper_bound": self.ece_upper_bound,
            "calibration_in_the_large": self.calibration_in_the_large,
            "calibration_slope": self.calibration_slope,
            "bootstrap_seed": self.bootstrap_seed,
            "metric_weighting": _METRIC_WEIGHTING,
            "producer_marker": _PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            expected = stable_id(
                "g3p_calibration_stratum_result",
                self._identity_payload(),
                schema_version="1",
            )
            valid = (
                self._producer_token is _PRODUCER_TOKEN
                and expected == self.stratum_result_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-P stratum result failed integrity validation",
                code="g3p_stratum_integrity_violation",
                field="stratum_result_id",
                remediation="Rebuild the stratum from complete replicate ledgers",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-P stratum result failed integrity validation",
                code="g3p_stratum_integrity_violation",
                field="stratum_result_id",
                remediation="Rebuild the stratum from complete replicate ledgers",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"stratum_result_id": self.stratum_result_id, **payload}


@dataclass(frozen=True, slots=True, init=False)
class G3PCalibrationScenarioResult:
    """Producer-owned metrics for one dependence/prevalence campaign cell."""

    scenario_id: str
    protocol_id: str
    dependence_structure: str
    non_null_prevalence: float
    status: G3PCalibrationScenarioStatus
    reason_code: str | None
    n_replicates: int
    n_observed_replicates: int
    minimum_eligible_candidates_per_stratum: int
    replicate_ids: tuple[str, ...]
    strata: tuple[G3PCalibrationStratumResult, ...]
    observed_prevalence: float | None
    brier_score: float | None
    prevalence_only_brier_score: float | None
    ece: float | None
    ece_upper_bound: float | None
    calibration_in_the_large: float | None
    calibration_slope: float | None
    bootstrap_seed: int | None
    scenario_result_id: str
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "G3PCalibrationScenarioResult is producer-owned; use "
            "summarize_g3p_calibration_campaign()"
        )

    @property
    def brier_relative_improvement(self) -> float | None:
        if self.brier_score is None or self.prevalence_only_brier_score is None:
            return None
        return (self.prevalence_only_brier_score - self.brier_score) / (
            self.prevalence_only_brier_score
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "protocol_id": self.protocol_id,
            "dependence_structure": self.dependence_structure,
            "non_null_prevalence": self.non_null_prevalence,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_replicates": self.n_replicates,
            "n_observed_replicates": self.n_observed_replicates,
            "minimum_eligible_candidates_per_stratum": (
                self.minimum_eligible_candidates_per_stratum
            ),
            "replicate_ids": list(self.replicate_ids),
            "stratum_result_ids": [item.stratum_result_id for item in self.strata],
            "observed_prevalence": self.observed_prevalence,
            "brier_score": self.brier_score,
            "prevalence_only_brier_score": self.prevalence_only_brier_score,
            "brier_relative_improvement": self.brier_relative_improvement,
            "ece": self.ece,
            "ece_upper_bound": self.ece_upper_bound,
            "calibration_in_the_large": self.calibration_in_the_large,
            "calibration_slope": self.calibration_slope,
            "bootstrap_seed": self.bootstrap_seed,
            "metric_weighting": _METRIC_WEIGHTING,
            "producer_marker": _PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            for stratum in self.strata:
                stratum._require_intact()
            expected = stable_id(
                "g3p_calibration_scenario_result",
                self._identity_payload(),
                schema_version="2",
            )
            valid = (
                self._producer_token is _PRODUCER_TOKEN
                and tuple(sorted(self.strata, key=lambda item: item.stratum_id))
                == self.strata
                and len({item.stratum_id for item in self.strata}) == len(self.strata)
                and all(
                    item.scenario_id == self.scenario_id
                    and item.protocol_id == self.protocol_id
                    for item in self.strata
                )
                and expected == self.scenario_result_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-P scenario failed integrity validation",
                code="g3p_scenario_integrity_violation",
                field="scenario_result_id",
                remediation="Rebuild the scenario from complete replicate ledgers",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-P scenario failed integrity validation",
                code="g3p_scenario_integrity_violation",
                field="scenario_result_id",
                remediation="Rebuild the scenario from complete replicate ledgers",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "scenario_result_id": self.scenario_result_id,
            **payload,
            "strata": [item.to_dict() for item in self.strata],
        }


@dataclass(frozen=True, slots=True, init=False)
class G3PCalibrationEvidence:
    """Producer-owned, portable runtime-contract-bound G3-P evidence."""

    source_artifact_id: str
    protocol_id: str
    active_null_spec_id: str
    active_probability_spec_id: str
    score_spec_id: str
    candidate_universe_policy_id: str
    estimator_id: str
    stratum_policy_id: str
    score_version: str
    generator_id: str
    generator_attestation_id: str | None
    generator_verified: bool
    required_dependence_structures: tuple[str, ...]
    scenarios: tuple[G3PCalibrationScenarioResult, ...]
    evidence_id: str
    _producer_token: object
    _generator_authorization_token: object

    def __init__(self) -> None:
        raise TypeError(
            "G3PCalibrationEvidence is producer-owned; use "
            "summarize_g3p_calibration_campaign()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "source_artifact_id": self.source_artifact_id,
            "protocol_id": self.protocol_id,
            "active_null_spec_id": self.active_null_spec_id,
            "active_probability_spec_id": self.active_probability_spec_id,
            "score_spec_id": self.score_spec_id,
            "candidate_universe_policy_id": self.candidate_universe_policy_id,
            "estimator_id": self.estimator_id,
            "stratum_policy_id": self.stratum_policy_id,
            "score_version": self.score_version,
            "generator_id": self.generator_id,
            "generator_attestation_id": self.generator_attestation_id,
            "generator_verified": self.generator_verified,
            "required_dependence_structures": list(self.required_dependence_structures),
            "scenario_result_ids": [item.scenario_result_id for item in self.scenarios],
            "producer_marker": _PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            for scenario in self.scenarios:
                scenario._require_intact()
            expected = stable_id(
                "g3p_calibration_evidence",
                self._identity_payload(),
                schema_version="3",
            )
            valid = (
                self._producer_token is _PRODUCER_TOKEN
                and _name(self.generator_id, field_name="generator_id")
                == self.generator_id
                and (
                    self.generator_attestation_id is None
                    or _name(
                        self.generator_attestation_id,
                        field_name="generator_attestation_id",
                    )
                    == self.generator_attestation_id
                )
                and self._generator_authorization_token
                in {_UNVERIFIED_GENERATOR_TOKEN, _VERIFIED_GENERATOR_TOKEN}
                and self.generator_verified
                is (self._generator_authorization_token is _VERIFIED_GENERATOR_TOKEN)
                and (
                    self.generator_attestation_id is not None
                    if self.generator_verified
                    else True
                )
                and self.required_dependence_structures
                == _REQUIRED_DEPENDENCE_STRUCTURES
                and expected == self.evidence_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-P calibration evidence failed integrity validation",
                code="g3p_evidence_integrity_violation",
                field="evidence_id",
                remediation="Rebuild evidence from the intact campaign",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-P calibration evidence failed integrity validation",
                code="g3p_evidence_integrity_violation",
                field="evidence_id",
                remediation="Rebuild evidence from the intact campaign",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "evidence_id": self.evidence_id,
            **payload,
            "scenarios": [item.to_dict() for item in self.scenarios],
        }


@dataclass(frozen=True, slots=True, init=False)
class G3PCalibrationCampaign:
    """Authenticated protocol, replicate ledgers, metrics, and evidence."""

    protocol: G3PCalibrationProtocol
    replicates: tuple[G3PReplicatePredictions, ...]
    scenarios: tuple[G3PCalibrationScenarioResult, ...]
    campaign_id: str
    evidence: G3PCalibrationEvidence
    generator_attestation: CalibrationReplayAttestation | None
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "G3PCalibrationCampaign is producer-owned; use "
            "summarize_g3p_calibration_campaign()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol.protocol_id,
            "replicate_ids": [item.replicate_id for item in self.replicates],
            "scenario_result_ids": [item.scenario_result_id for item in self.scenarios],
            "producer_marker": _PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            self.protocol._require_intact()
            for replicate in self.replicates:
                replicate._require_intact()
            for scenario in self.scenarios:
                scenario._require_intact()
            self.evidence._require_intact()
            if self.generator_attestation is not None:
                self.generator_attestation._require_intact()
            expected = stable_id(
                "g3p_calibration_campaign",
                self._identity_payload(),
                schema_version="1",
            )
            valid = (
                self._producer_token is _PRODUCER_TOKEN
                and expected == self.campaign_id
                and self.evidence.source_artifact_id == self.campaign_id
                and self.evidence.protocol_id == self.protocol.protocol_id
                and tuple(
                    item.scenario_result_id for item in self.evidence.scenarios
                )
                == tuple(item.scenario_result_id for item in self.scenarios)
                and (
                    (
                        self.generator_attestation is None
                        and self.evidence.generator_attestation_id is None
                        and not self.evidence.generator_verified
                    )
                    or (
                        self.generator_attestation is not None
                        and self.evidence.generator_attestation_id
                        == self.generator_attestation.attestation_id
                        and self.evidence.generator_verified
                        is self.generator_attestation.release_approved
                        and self.generator_attestation.replicate_ids
                        == tuple(item.replicate_id for item in self.replicates)
                    )
                )
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-P campaign failed integrity validation",
                code="g3p_campaign_integrity_violation",
                field="campaign_id",
                remediation="Rebuild the campaign from the original replicate ledgers",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-P campaign failed integrity validation",
                code="g3p_campaign_integrity_violation",
                field="campaign_id",
                remediation="Rebuild the campaign from the original replicate ledgers",
            )

    def to_manifest(self) -> dict[str, object]:
        self._require_intact()
        return {
            "schema_version": _SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "protocol": self.protocol.to_dict(),
            "replicate_ids": [item.replicate_id for item in self.replicates],
            "scenarios": [item.to_dict() for item in self.scenarios],
            "evidence": self.evidence.to_dict(),
        }


def _stratum_result(
    *,
    protocol: G3PCalibrationProtocol,
    scenario_id: str,
    prevalence: float,
    stratum_id: str,
    replicates: tuple[G3PReplicatePredictions, ...],
) -> G3PCalibrationStratumResult:
    minimum_eligible = min(
        (
            sum(
                candidate_stratum == stratum_id
                and candidate_status is G3PCandidateStatus.ELIGIBLE
                for candidate_stratum, candidate_status in zip(
                    replicate.stratum_ids,
                    replicate.candidate_statuses,
                    strict=True,
                )
            )
            for replicate in replicates
        ),
        default=0,
    )
    reason: str | None = None
    values: dict[str, float | int | None] = {
        "observed_prevalence": None,
        "brier_score": None,
        "prevalence_only_brier_score": None,
        "ece": None,
        "ece_upper_bound": None,
        "calibration_in_the_large": None,
        "calibration_slope": None,
        "bootstrap_seed": None,
    }
    if minimum_eligible < _MIN_CANDIDATES_PER_STRATUM:
        reason = "g3p_calibration_stratum_too_small"
    else:
        records = tuple(
            BinaryCalibrationRecord(
                replicate_id=replicate.replicate_id,
                observation_id=candidate_id,
                outcome=int(replicate.truth_active[index]),
                probability=float(replicate.candidate_probabilities[index]),
            )
            for replicate in replicates
            for index, (candidate_id, candidate_stratum) in enumerate(
                zip(
                    replicate.candidate_edge_ids,
                    replicate.stratum_ids,
                    strict=True,
                )
            )
            if candidate_stratum == stratum_id
            and replicate.candidate_statuses[index] is G3PCandidateStatus.ELIGIBLE
        )
        bootstrap_seed = protocol.seed_lineage.derive(
            "g3p_ece_bootstrap_v2",
            f"scenario_id={scenario_id}",
            f"stratum_id={stratum_id}",
        ).seed
        try:
            point = compute_binary_calibration_metrics(
                records,
                baseline_probability=prevalence,
                n_bins=_ECE_BINS,
                probability_clip=_PROBABILITY_CLIP,
            )
            interval = replicate_cluster_bootstrap_ece_upper_bound(
                records,
                n_bins=_ECE_BINS,
                confidence_level=_CONFIDENCE_LEVEL,
                n_resamples=_ECE_BOOTSTRAP_RESAMPLES,
                seed=bootstrap_seed,
            )
        except CalibrationNotEstimableError as error:
            reason = error.reason_code
        else:
            values = {
                "observed_prevalence": point.observed_prevalence,
                "brier_score": point.brier_score,
                "prevalence_only_brier_score": point.prevalence_only_brier_score,
                "ece": point.fixed_bin_ece.ece,
                "ece_upper_bound": interval.upper_bound,
                "calibration_in_the_large": (
                    point.logistic_recalibration.calibration_in_the_large
                ),
                "calibration_slope": point.logistic_recalibration.slope,
                "bootstrap_seed": bootstrap_seed,
            }
    status = (
        G3PCalibrationScenarioStatus.OBSERVED
        if reason is None
        else G3PCalibrationScenarioStatus.NOT_ESTIMABLE
    )
    self = object.__new__(G3PCalibrationStratumResult)
    for name, value in {
        "scenario_id": scenario_id,
        "protocol_id": protocol.protocol_id,
        "stratum_id": stratum_id,
        "status": status,
        "reason_code": reason,
        "n_replicates": len(replicates),
        "minimum_eligible_candidates": minimum_eligible,
        **values,
        "_producer_token": _PRODUCER_TOKEN,
    }.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "stratum_result_id",
        stable_id(
            "g3p_calibration_stratum_result",
            self._identity_payload(),
            schema_version="1",
        ),
    )
    self._require_intact()
    return self


def _scenario_summary_values(
    strata: tuple[G3PCalibrationStratumResult, ...],
) -> dict[str, float | int | None]:
    worst_brier = min(
        strata,
        key=lambda item: cast(float, item.brier_relative_improvement),
    )
    maximum_ece = max(strata, key=lambda item: cast(float, item.ece))
    maximum_ece_upper = max(strata, key=lambda item: cast(float, item.ece_upper_bound))
    maximum_citl = max(
        strata,
        key=lambda item: abs(cast(float, item.calibration_in_the_large)),
    )
    worst_slope = max(
        strata,
        key=lambda item: abs(cast(float, item.calibration_slope) - 1.0),
    )
    return {
        "observed_prevalence": float(
            np.mean([cast(float, item.observed_prevalence) for item in strata])
        ),
        "brier_score": worst_brier.brier_score,
        "prevalence_only_brier_score": worst_brier.prevalence_only_brier_score,
        "ece": maximum_ece.ece,
        "ece_upper_bound": maximum_ece_upper.ece_upper_bound,
        "calibration_in_the_large": maximum_citl.calibration_in_the_large,
        "calibration_slope": worst_slope.calibration_slope,
        "bootstrap_seed": strata[0].bootstrap_seed if len(strata) == 1 else None,
    }


def _scenario_result(
    *,
    protocol: G3PCalibrationProtocol,
    dependence: G3PDependenceStructure,
    prevalence: float,
    replicates: tuple[G3PReplicatePredictions, ...],
) -> G3PCalibrationScenarioResult:
    scenario_id = protocol.scenario_id(dependence, prevalence)
    ordered = tuple(sorted(replicates, key=lambda item: item.replicate_index))
    reason: str | None = None
    observed = tuple(
        item for item in ordered if item.status is G3PReplicateStatus.OBSERVED
    )
    strata: tuple[G3PCalibrationStratumResult, ...] = ()
    minimum_eligible = 0
    metrics_values: dict[str, float | int | None] = {
        "observed_prevalence": None,
        "brier_score": None,
        "prevalence_only_brier_score": None,
        "ece": None,
        "ece_upper_bound": None,
        "calibration_in_the_large": None,
        "calibration_slope": None,
        "bootstrap_seed": None,
    }
    expected_indexes = tuple(range(len(ordered)))
    actual_indexes = tuple(item.replicate_index for item in ordered)
    if not ordered:
        reason = "g3p_calibration_missing_replicates"
    elif actual_indexes != expected_indexes:
        reason = "g3p_calibration_noncontiguous_replicates"
    elif any(
        item.seed_lineage
        != protocol.replicate_seed_lineage(dependence, prevalence, item.replicate_index)
        for item in ordered
    ):
        reason = "g3p_calibration_seed_lineage_mismatch"
    elif len(observed) != len(ordered):
        reason = "g3p_calibration_nonobserved_replicate"
    elif any(item.protocol_id != protocol.protocol_id for item in ordered):
        reason = "g3p_calibration_protocol_mismatch"
    elif len({item.source_active_null_id for item in ordered}) != len(ordered) or len(
        {item.source_probability_collection_id for item in observed}
    ) != len(observed):
        reason = "g3p_calibration_duplicate_source_artifact"
    else:
        reference_candidates = ordered[0].candidate_edge_ids
        reference_strata = ordered[0].stratum_ids
        if any(
            item.candidate_edge_ids != reference_candidates
            or item.stratum_ids != reference_strata
            for item in ordered[1:]
        ):
            reason = "g3p_calibration_candidate_coverage_mismatch"
        else:
            strata = tuple(
                _stratum_result(
                    protocol=protocol,
                    scenario_id=scenario_id,
                    prevalence=prevalence,
                    stratum_id=stratum_id,
                    replicates=ordered,
                )
                for stratum_id in sorted(set(reference_strata))
            )
            minimum_eligible = min(
                (item.minimum_eligible_candidates for item in strata),
                default=0,
            )
            unavailable_strata = tuple(
                item
                for item in strata
                if item.status is not G3PCalibrationScenarioStatus.OBSERVED
            )
            if unavailable_strata:
                reason = (
                    unavailable_strata[0].reason_code
                    if len(unavailable_strata) == 1
                    else "g3p_calibration_stratum_not_estimable"
                )
            else:
                metrics_values = _scenario_summary_values(strata)
    status = (
        G3PCalibrationScenarioStatus.OBSERVED
        if reason is None
        else G3PCalibrationScenarioStatus.NOT_ESTIMABLE
    )
    self = object.__new__(G3PCalibrationScenarioResult)
    values: dict[str, object] = {
        "scenario_id": scenario_id,
        "protocol_id": protocol.protocol_id,
        "dependence_structure": dependence.value,
        "non_null_prevalence": prevalence,
        "status": status,
        "reason_code": reason,
        "n_replicates": len(ordered),
        "n_observed_replicates": len(observed),
        "minimum_eligible_candidates_per_stratum": minimum_eligible,
        "replicate_ids": tuple(item.replicate_id for item in ordered),
        "strata": strata,
        **metrics_values,
        "_producer_token": _PRODUCER_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "scenario_result_id",
        stable_id(
            "g3p_calibration_scenario_result",
            self._identity_payload(),
            schema_version="2",
        ),
    )
    self._require_intact()
    return self


def _summarize_g3p_calibration_campaign(
    protocol: G3PCalibrationProtocol,
    replicates: Sequence[G3PReplicatePredictions],
    *,
    generator_attestation: CalibrationReplayAttestation | None,
) -> G3PCalibrationCampaign:
    """Summarize the exact grid with an optional producer-owned replay proof."""

    if not isinstance(protocol, G3PCalibrationProtocol):
        raise TypeError("protocol must be a G3PCalibrationProtocol")
    protocol._require_intact()
    supplied = tuple(replicates)
    if any(not isinstance(item, G3PReplicatePredictions) for item in supplied):
        raise TypeError("replicates must contain G3PReplicatePredictions values")
    if generator_attestation is not None:
        if not isinstance(generator_attestation, CalibrationReplayAttestation):
            raise TypeError(
                "generator_attestation must be a CalibrationReplayAttestation"
            )
        generator_attestation._require_intact()
        if (
            generator_attestation.campaign_kind
            is not CalibrationCampaignKind.G3_PROBABILITY
            or generator_attestation.protocol_id != protocol.protocol_id
            or generator_attestation.generator_id != protocol.generator_id
        ):
            raise _contract_error(
                "G3-P generator attestation belongs to a different protocol",
                code="g3p_generator_attestation_mismatch",
                field="generator_attestation_id",
                remediation="Replay the complete campaign under this protocol",
            )
    for replicate in supplied:
        replicate._require_intact()
        if replicate.protocol_id != protocol.protocol_id:
            raise _contract_error(
                "G3-P replicate belongs to a different protocol",
                code="g3p_replicate_protocol_mismatch",
                field="protocol_id",
                remediation="Submit only replicates generated under this protocol",
            )
    ordered = tuple(
        sorted(
            supplied,
            key=lambda item: (
                G3PDependenceStructure(item.dependence_structure).value,
                item.non_null_prevalence,
                item.replicate_index,
            ),
        )
    )
    keys = tuple(
        (
            G3PDependenceStructure(item.dependence_structure).value,
            item.non_null_prevalence,
            item.replicate_index,
        )
        for item in ordered
    )
    if len(keys) != len(set(keys)):
        raise _contract_error(
            "G3-P campaign contains duplicate scenario replicate cells",
            code="duplicate_g3p_calibration_replicate",
            field="dependence_structure,non_null_prevalence,replicate_index",
            remediation="Retain exactly one ledger for every attempted replicate",
        )
    if generator_attestation is not None and (
        generator_attestation.replicate_ids
        != tuple(item.replicate_id for item in ordered)
    ):
        raise _contract_error(
            "G3-P generator attestation does not cover the supplied ledgers",
            code="g3p_generator_attestation_coverage_mismatch",
            field="replicate_ids",
            remediation="Replay the complete supplied campaign without adding rows",
        )
    active_null_ids = tuple(item.source_active_null_id for item in ordered)
    candidate_universe_ids = tuple(
        item.source_candidate_universe_id for item in ordered
    )
    probability_collection_ids = tuple(
        item.source_probability_collection_id
        for item in ordered
        if item.source_probability_collection_id is not None
    )
    if (
        len(active_null_ids) != len(set(active_null_ids))
        or len(candidate_universe_ids) != len(set(candidate_universe_ids))
        or len(probability_collection_ids)
        != len(set(probability_collection_ids))
    ):
        raise _contract_error(
            "G3-P campaign reuses a source artifact across replicated datasets",
            code="duplicate_g3p_calibration_source_artifact",
            field=(
                "source_active_null_id,source_candidate_universe_id,"
                "source_probability_collection_id"
            ),
            remediation="Run and retain one distinct source artifact per replicate",
        )
    scenarios = tuple(
        _scenario_result(
            protocol=protocol,
            dependence=dependence,
            prevalence=prevalence,
            replicates=tuple(
                item
                for item in ordered
                if item.dependence_structure is dependence
                and item.non_null_prevalence == prevalence
            ),
        )
        for dependence in G3PDependenceStructure
        for prevalence in _REQUIRED_PREVALENCES
    )
    campaign_payload = {
        "protocol_id": protocol.protocol_id,
        "replicate_ids": [item.replicate_id for item in ordered],
        "scenario_result_ids": [item.scenario_result_id for item in scenarios],
        "producer_marker": _PRODUCER,
    }
    campaign_id = stable_id(
        "g3p_calibration_campaign",
        campaign_payload,
        schema_version="1",
    )
    evidence = object.__new__(G3PCalibrationEvidence)
    evidence_values: dict[str, object] = {
        "source_artifact_id": campaign_id,
        "protocol_id": protocol.protocol_id,
        "active_null_spec_id": protocol.active_null_spec_id,
        "active_probability_spec_id": protocol.active_probability_spec_id,
        "score_spec_id": protocol.score_spec_id,
        "candidate_universe_policy_id": protocol.candidate_universe_policy_id,
        "estimator_id": protocol.estimator_id,
        "stratum_policy_id": protocol.stratum_policy_id,
        "score_version": protocol.score_version,
        "generator_id": protocol.generator_id,
        "generator_attestation_id": (
            None
            if generator_attestation is None
            else generator_attestation.attestation_id
        ),
        "generator_verified": (
            False
            if generator_attestation is None
            else generator_attestation.release_approved
        ),
        "required_dependence_structures": _REQUIRED_DEPENDENCE_STRUCTURES,
        "scenarios": scenarios,
        "_producer_token": _PRODUCER_TOKEN,
        "_generator_authorization_token": (
            _VERIFIED_GENERATOR_TOKEN
            if generator_attestation is not None
            and generator_attestation.release_approved
            else _UNVERIFIED_GENERATOR_TOKEN
        ),
    }
    for name, value in evidence_values.items():
        object.__setattr__(evidence, name, value)
    object.__setattr__(
        evidence,
        "evidence_id",
        stable_id(
            "g3p_calibration_evidence",
            evidence._identity_payload(),
            schema_version="3",
        ),
    )
    evidence._require_intact()
    campaign = object.__new__(G3PCalibrationCampaign)
    for name, value in {
        "protocol": protocol,
        "replicates": ordered,
        "scenarios": scenarios,
        "campaign_id": campaign_id,
        "evidence": evidence,
        "generator_attestation": generator_attestation,
        "_producer_token": _PRODUCER_TOKEN,
    }.items():
        object.__setattr__(campaign, name, value)
    campaign._require_intact()
    return campaign


def summarize_g3p_calibration_campaign(
    protocol: G3PCalibrationProtocol,
    replicates: Sequence[G3PReplicatePredictions],
) -> G3PCalibrationCampaign:
    """Summarize caller-supplied raw ledgers as diagnostic-only evidence."""

    return _summarize_g3p_calibration_campaign(
        protocol,
        replicates,
        generator_attestation=None,
    )


def summarize_attested_g3p_calibration_campaign(
    protocol: G3PCalibrationProtocol,
    replicates: Sequence[G3PReplicatePredictions],
    *,
    registry: CalibrationReplayRegistry,
    n_jobs: int = 1,
) -> G3PCalibrationCampaign:
    """Replay every G3-P ledger, then derive evidence bound to that proof."""

    supplied = tuple(replicates)
    attestation = attest_calibration_replay(
        registry,
        campaign_kind=CalibrationCampaignKind.G3_PROBABILITY,
        protocol=protocol,
        ledgers=supplied,
        n_jobs=n_jobs,
    )
    return _summarize_g3p_calibration_campaign(
        protocol,
        supplied,
        generator_attestation=attestation,
    )


@dataclass(frozen=True, slots=True, init=False)
class G3PCalibrationGate:
    """Producer-owned portable G3-P release gate."""

    gate_id: str
    evidence_id: str
    source_artifact_id: str
    protocol_id: str
    active_null_spec_id: str
    active_probability_spec_id: str
    score_spec_id: str
    candidate_universe_policy_id: str
    estimator_id: str
    stratum_policy_id: str
    score_version: str
    generator_id: str
    generator_attestation_id: str | None
    generator_verified: bool
    status: G3PGateStatus
    reason_code: str | None
    minimum_replicates: int
    minimum_brier_relative_improvement: float | None
    maximum_ece: float | None
    maximum_ece_upper_bound: float | None
    maximum_absolute_calibration_in_the_large: float | None
    minimum_calibration_slope: float | None
    maximum_calibration_slope: float | None
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError("Use build_g3p_calibration_gate()")

    @property
    def comm_probability_release_allowed(self) -> bool:
        return self.status is G3PGateStatus.PASSED

    def _identity_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "source_artifact_id": self.source_artifact_id,
            "protocol_id": self.protocol_id,
            "active_null_spec_id": self.active_null_spec_id,
            "active_probability_spec_id": self.active_probability_spec_id,
            "score_spec_id": self.score_spec_id,
            "candidate_universe_policy_id": self.candidate_universe_policy_id,
            "estimator_id": self.estimator_id,
            "stratum_policy_id": self.stratum_policy_id,
            "score_version": self.score_version,
            "generator_id": self.generator_id,
            "generator_attestation_id": self.generator_attestation_id,
            "generator_verified": self.generator_verified,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "minimum_replicates": self.minimum_replicates,
            "minimum_brier_relative_improvement": (
                self.minimum_brier_relative_improvement
            ),
            "maximum_ece": self.maximum_ece,
            "maximum_ece_upper_bound": self.maximum_ece_upper_bound,
            "maximum_absolute_calibration_in_the_large": (
                self.maximum_absolute_calibration_in_the_large
            ),
            "minimum_calibration_slope": self.minimum_calibration_slope,
            "maximum_calibration_slope": self.maximum_calibration_slope,
            "producer_marker": _GATE_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            expected = stable_id(
                "g3p_calibration_gate",
                self._identity_payload(),
                schema_version="3",
            )
            valid = (
                self._producer_token is _GATE_PRODUCER_TOKEN
                and (
                    not self.generator_verified
                    or self.generator_attestation_id is not None
                )
                and expected == self.gate_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-P gate failed integrity validation",
                code="g3p_gate_integrity_violation",
                field="gate_id",
                remediation="Rebuild the gate from intact campaign evidence",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-P gate failed integrity validation",
                code="g3p_gate_integrity_violation",
                field="gate_id",
                remediation="Rebuild the gate from intact campaign evidence",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "gate_id": self.gate_id,
            **payload,
            "comm_probability_release_allowed": self.comm_probability_release_allowed,
            "thresholds": {
                "minimum_replicates": _MIN_CALIBRATION_REPLICATES,
                "required_dependence_structures": list(_REQUIRED_DEPENDENCE_STRUCTURES),
                "required_prevalences": list(_REQUIRED_PREVALENCES),
                "minimum_candidates_per_stratum": _MIN_CANDIDATES_PER_STRATUM,
                "brier_relative_improvement_minimum": _BRIER_IMPROVEMENT_MINIMUM,
                "ece_maximum": _ECE_MAXIMUM,
                "ece_upper_maximum": _ECE_UPPER_MAXIMUM,
                "calibration_in_the_large_abs_maximum": (
                    _CALIBRATION_IN_THE_LARGE_ABS_MAXIMUM
                ),
                "calibration_slope_bounds": list(_CALIBRATION_SLOPE_BOUNDS),
            },
        }


def build_g3p_calibration_gate(
    evidence: G3PCalibrationEvidence,
) -> G3PCalibrationGate:
    """Derive PASS/FAIL/NE from producer-owned complete-grid evidence."""

    if not isinstance(evidence, G3PCalibrationEvidence):
        raise TypeError("evidence must be producer-owned G3PCalibrationEvidence")
    evidence._require_intact()
    expected = {
        (dependence, prevalence)
        for dependence in _REQUIRED_DEPENDENCE_STRUCTURES
        for prevalence in _REQUIRED_PREVALENCES
    }
    observed_grid = {
        (item.dependence_structure, item.non_null_prevalence)
        for item in evidence.scenarios
    }
    observed_scenarios = tuple(
        item
        for item in evidence.scenarios
        if item.status is G3PCalibrationScenarioStatus.OBSERVED
    )
    observed_strata = tuple(
        stratum
        for scenario in observed_scenarios
        for stratum in scenario.strata
        if stratum.status is G3PCalibrationScenarioStatus.OBSERVED
    )
    minimum_replicates = min(
        (item.n_replicates for item in evidence.scenarios), default=0
    )
    metrics_complete = (
        len(observed_scenarios) == len(evidence.scenarios)
        and all(item.strata for item in observed_scenarios)
        and len(observed_strata) == sum(len(item.strata) for item in observed_scenarios)
    )
    if observed_grid != expected:
        status = G3PGateStatus.NOT_ESTIMABLE
        reason: str | None = "g3p_calibration_missing_scenarios"
    elif not metrics_complete:
        status = G3PGateStatus.NOT_ESTIMABLE
        reason = "g3p_calibration_scenario_not_estimable"
    elif minimum_replicates < _MIN_CALIBRATION_REPLICATES:
        status = G3PGateStatus.NOT_ESTIMABLE
        reason = "g3p_calibration_insufficient_replicates"
    elif not evidence.generator_verified:
        status = G3PGateStatus.NOT_ESTIMABLE
        reason = "g3p_calibration_generator_unverified"
    else:
        improvements = cast(
            tuple[float, ...],
            tuple(item.brier_relative_improvement for item in observed_strata),
        )
        eces = cast(tuple[float, ...], tuple(item.ece for item in observed_strata))
        ece_uppers = cast(
            tuple[float, ...],
            tuple(item.ece_upper_bound for item in observed_strata),
        )
        citl = cast(
            tuple[float, ...],
            tuple(
                abs(cast(float, item.calibration_in_the_large))
                for item in observed_strata
            ),
        )
        slopes = cast(
            tuple[float, ...],
            tuple(item.calibration_slope for item in observed_strata),
        )
        if (
            min(improvements) < _BRIER_IMPROVEMENT_MINIMUM
            or max(eces) > _ECE_MAXIMUM
            or max(ece_uppers) > _ECE_UPPER_MAXIMUM
            or max(citl) > _CALIBRATION_IN_THE_LARGE_ABS_MAXIMUM
            or min(slopes) < _CALIBRATION_SLOPE_BOUNDS[0]
            or max(slopes) > _CALIBRATION_SLOPE_BOUNDS[1]
        ):
            status = G3PGateStatus.FAILED
            reason = "g3p_calibration_failed"
        else:
            status = G3PGateStatus.PASSED
            reason = None
    if observed_strata:
        improvement_values = cast(
            tuple[float, ...],
            tuple(item.brier_relative_improvement for item in observed_strata),
        )
        ece_values = cast(
            tuple[float, ...], tuple(item.ece for item in observed_strata)
        )
        ece_upper_values = cast(
            tuple[float, ...],
            tuple(item.ece_upper_bound for item in observed_strata),
        )
        citl_values = tuple(
            abs(cast(float, item.calibration_in_the_large)) for item in observed_strata
        )
        slope_values = cast(
            tuple[float, ...],
            tuple(item.calibration_slope for item in observed_strata),
        )
    else:
        improvement_values = ()
        ece_values = ()
        ece_upper_values = ()
        citl_values = ()
        slope_values = ()
    self = object.__new__(G3PCalibrationGate)
    values: dict[str, object] = {
        "evidence_id": evidence.evidence_id,
        "source_artifact_id": evidence.source_artifact_id,
        "protocol_id": evidence.protocol_id,
        "active_null_spec_id": evidence.active_null_spec_id,
        "active_probability_spec_id": evidence.active_probability_spec_id,
        "score_spec_id": evidence.score_spec_id,
        "candidate_universe_policy_id": evidence.candidate_universe_policy_id,
        "estimator_id": evidence.estimator_id,
        "stratum_policy_id": evidence.stratum_policy_id,
        "score_version": evidence.score_version,
        "generator_id": evidence.generator_id,
        "generator_attestation_id": evidence.generator_attestation_id,
        "generator_verified": evidence.generator_verified,
        "status": status,
        "reason_code": reason,
        "minimum_replicates": minimum_replicates,
        "minimum_brier_relative_improvement": min(improvement_values, default=None),
        "maximum_ece": max(ece_values, default=None),
        "maximum_ece_upper_bound": max(ece_upper_values, default=None),
        "maximum_absolute_calibration_in_the_large": max(citl_values, default=None),
        "minimum_calibration_slope": min(slope_values, default=None),
        "maximum_calibration_slope": max(slope_values, default=None),
        "_producer_token": _GATE_PRODUCER_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "gate_id",
        stable_id(
            "g3p_calibration_gate",
            self._identity_payload(),
            schema_version="3",
        ),
    )
    self._require_intact()
    return self


__all__ = [
    "G3PCalibrationCampaign",
    "G3PCalibrationEvidence",
    "G3PCalibrationGate",
    "G3PCalibrationProtocol",
    "G3PCalibrationScenarioResult",
    "G3PCalibrationScenarioStatus",
    "G3PCalibrationStratumResult",
    "G3PCandidateStatus",
    "G3PDependenceStructure",
    "G3PGateStatus",
    "G3PReplicatePredictions",
    "G3PReplicateStatus",
    "build_g3p_calibration_gate",
    "summarize_attested_g3p_calibration_campaign",
    "summarize_g3p_calibration_campaign",
]
