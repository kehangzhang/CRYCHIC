"""Producer-owned G3-F frequency-calibration campaign contracts.

Release evidence is computed from complete simulated-dataset ledgers.  A
caller may provide raw truth, candidate/baseline decisions, interval coverage
and widths, and permutation p-values, but cannot provide scenario metrics,
Monte Carlo bounds, or a gate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import numpy as np

from crychic.core import ContractError, SeedLineage, stable_id

from .calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationGeneratorManifest,
    CalibrationReplayAttestation,
    CalibrationReplayRegistry,
    attest_calibration_replay,
)
from .calibration_metrics import BinomialBoundSide, exact_binomial_one_sided_bound
from .hierarchical import FrozenHierarchicalFDRSpec
from .hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisPrefilterStatus,
    HypothesisRole,
)

_PROTOCOL_SCHEMA_VERSION = "3.0.0"
_PRODUCER = "crychic.inference.g3f_calibration_campaign.v4"
_GATE_PRODUCER = "crychic.inference.g3_frequency_gate.v5"
# These object capabilities prevent accidental API construction; generator
# authenticity still requires the separate replayable attestation boundary.
_PRODUCER_TOKEN = object()
_GATE_PRODUCER_TOKEN = object()
_UNVERIFIED_GENERATOR_TOKEN = object()
_VERIFIED_GENERATOR_TOKEN = object()

_MIN_CALIBRATION_REPLICATES = 1_000
_MONTE_CARLO_BOOTSTRAP_RESAMPLES = 10_000
_CONFIDENCE_LEVEL = 0.95
_TYPE_I_UPPER_LIMIT = 0.06
_MIXED_FDR_UPPER_LIMIT = 0.07
_PRIMARY_FDR_UPPER_LIMIT = 0.07
_SELECTIVE_CHILD_FDR_UPPER_LIMIT = 0.07
_COVERAGE_LOWER_LIMIT = 0.92
_POWER_NONINFERIORITY_MARGIN = -0.05
_INTERVAL_WIDTH_NONINFERIORITY_MARGIN = -0.10
_POWER_NONINFERIORITY_ESTIMAND = (
    "candidate_minus_baseline_true_positive_rejection_rate_v1"
)
_INTERVAL_WIDTH_NONINFERIORITY_ESTIMAND = (
    "mean_baseline_minus_candidate_width_over_baseline_width_v1"
)
_NONINFERIORITY_DIRECTION = "larger_is_better_lower_bound_at_least_margin"
_PERMUTATION_DKW_ALPHA = 0.05
_REQUIRED_PREVALENCES = (0.005, 0.01, 0.05, 0.10)
_METRIC_WEIGHTING = (
    "hypothesis_equal_within_replicate_then_informative_replicate_equal_v2"
)
_PREVALENCE_MONTE_CARLO_Z = 4.0


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_name(value: object | None, *, field_name: str) -> str | None:
    return None if value is None else _name(value, field_name=field_name)


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{field_name} must be an integer >= 0")
    return int(value)


def _prevalence(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("non_null_prevalence must be in the frozen G3-F grid")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "non_null_prevalence must be in the frozen G3-F grid"
        ) from error
    if result not in _REQUIRED_PREVALENCES:
        raise ValueError("non_null_prevalence must be in the frozen G3-F grid")
    return result


def _unit(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must lie in [0, 1]") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1]")
    return 0.0 if result == 0.0 else result


def _strict_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain boolean values")
    return bool(value)


def _immutable_bool_array(values: object, *, field_name: str) -> np.ndarray:
    supplied = tuple(cast(Sequence[object], values))
    canonical = np.asarray(
        tuple(_strict_bool(item, field_name=field_name) for item in supplied),
        dtype="|b1",
    )
    if canonical.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    result = np.frombuffer(canonical.tobytes(order="C"), dtype="|b1")
    result.setflags(write=False)
    return cast(np.ndarray, result)


def _immutable_float_array(values: object, *, field_name: str) -> np.ndarray:
    canonical = np.asarray(values, dtype="<f8", order="C")
    if canonical.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    copied = np.array(canonical, dtype="<f8", copy=True, order="C")
    copied[copied == 0.0] = 0.0
    result = np.frombuffer(copied.tobytes(order="C"), dtype="<f8")
    result.setflags(write=False)
    return cast(np.ndarray, result)


def _array_digest(values: np.ndarray, *, dtype: str) -> str:
    canonical = np.asarray(values, dtype=dtype, order="C")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(dtype.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


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


class G3FrequencyDependenceStructure(StrEnum):
    """Frozen G3-F v1 hypothesis-dependence structures."""

    INDEPENDENT_HYPOTHESES = "independent_hypotheses"
    POSITIVE_WITHIN_FAMILY_BLOCK = "positive_within_family_block"
    NEGATIVE_WITHIN_FAMILY_BLOCK = "negative_within_family_block"


_REQUIRED_DEPENDENCE_STRUCTURES = tuple(
    item.value for item in G3FrequencyDependenceStructure
)


class G3FrequencyReplicateStatus(StrEnum):
    """Availability of one complete simulated-dataset ledger."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


class G3CalibrationScenarioStatus(StrEnum):
    """Availability of one producer-computed metric in one campaign cell."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"


class G3CalibrationMetric(StrEnum):
    """One preregistered one-sided G3-F operating characteristic."""

    NULL_TYPE_I_UPPER = "null_type_i_upper"
    MIXED_FDR_UPPER = "mixed_fdr_upper"
    PRIMARY_FDR_UPPER = "primary_fdr_upper"
    SELECTIVE_CHILD_FDR_UPPER = "selective_child_fdr_upper"
    COVERAGE_LOWER = "coverage_lower"
    POWER_NONINFERIORITY_LOWER = "power_noninferiority_lower"
    INTERVAL_WIDTH_NONINFERIORITY_LOWER = (
        "interval_width_noninferiority_lower"
    )


_GLOBAL_NULL_METRICS = (
    G3CalibrationMetric.NULL_TYPE_I_UPPER,
    G3CalibrationMetric.COVERAGE_LOWER,
    G3CalibrationMetric.INTERVAL_WIDTH_NONINFERIORITY_LOWER,
)
_MIXED_METRICS = (
    G3CalibrationMetric.MIXED_FDR_UPPER,
    G3CalibrationMetric.PRIMARY_FDR_UPPER,
    G3CalibrationMetric.SELECTIVE_CHILD_FDR_UPPER,
    G3CalibrationMetric.COVERAGE_LOWER,
    G3CalibrationMetric.POWER_NONINFERIORITY_LOWER,
    G3CalibrationMetric.INTERVAL_WIDTH_NONINFERIORITY_LOWER,
)


class G3FrequencyGateStatus(StrEnum):
    """Producer-derived G3-F decision."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_ESTIMABLE = "not_estimable"


@dataclass(frozen=True, slots=True, kw_only=True)
class G3FrequencyCalibrationProtocol:
    """Pre-inspection campaign grid bound to one universe and procedure."""

    campaign_name: str
    generator_manifest: CalibrationGeneratorManifest
    noninferiority_baseline_id: str
    hypothesis_universe: FrozenHypothesisUniverse
    hierarchical_procedure: FrozenHierarchicalFDRSpec
    seed_lineage: SeedLineage
    schema_version: str = _PROTOCOL_SCHEMA_VERSION
    protocol_id: str = field(init=False)

    def __post_init__(self) -> None:
        campaign = _name(self.campaign_name, field_name="campaign_name")
        if not isinstance(self.generator_manifest, CalibrationGeneratorManifest):
            raise TypeError(
                "generator_manifest must be a CalibrationGeneratorManifest"
            )
        self.generator_manifest._require_intact()
        if (
            self.generator_manifest.campaign_kind
            is not CalibrationCampaignKind.G3_FREQUENCY
        ):
            raise ValueError("G3-F protocol requires a G3 frequency generator")
        baseline = _name(
            self.noninferiority_baseline_id,
            field_name="noninferiority_baseline_id",
        )
        if not isinstance(self.hypothesis_universe, FrozenHypothesisUniverse):
            raise TypeError("hypothesis_universe must be a FrozenHypothesisUniverse")
        self.hypothesis_universe._require_intact()
        if not isinstance(self.hierarchical_procedure, FrozenHierarchicalFDRSpec):
            raise TypeError(
                "hierarchical_procedure must be a FrozenHierarchicalFDRSpec"
            )
        self.hierarchical_procedure._require_intact()
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        if self.schema_version != _PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"G3-F protocol schema_version must be {_PROTOCOL_SCHEMA_VERSION}"
            )
        declarations = self.hypothesis_universe.declarations
        roles = tuple(item.role for item in declarations)
        if HypothesisRole.PRIMARY not in roles or HypothesisRole.SECONDARY not in roles:
            raise ValueError("G3-F universe requires primary and secondary hypotheses")
        primary_ids = {
            item.hypothesis_id
            for item in declarations
            if item.role is HypothesisRole.PRIMARY
        }
        children_by_parent = {item: 0 for item in primary_ids}
        for item in declarations:
            if item.role is HypothesisRole.SECONDARY:
                parent_id = next(
                    (
                        parent.hypothesis_id
                        for parent in declarations
                        if parent.hypothesis_key == item.parent_key
                    ),
                    None,
                )
                if parent_id is None:
                    raise ValueError("G3-F universe contains a dangling secondary")
                children_by_parent[parent_id] += 1
        if min(children_by_parent.values(), default=0) < 1:
            raise ValueError("every G3-F primary must have at least one child")
        object.__setattr__(self, "campaign_name", campaign)
        object.__setattr__(self, "noninferiority_baseline_id", baseline)
        object.__setattr__(
            self,
            "protocol_id",
            stable_id(
                "g3_frequency_calibration_protocol",
                self._identity_payload(),
                schema_version="3",
            ),
        )

    @property
    def generator_id(self) -> str:
        return self.generator_manifest.generator_id

    @property
    def generator_release_verified(self) -> bool:
        """A generator declaration alone is never release evidence."""

        return False

    @property
    def hypothesis_universe_id(self) -> str:
        return self.hypothesis_universe.universe_id

    @property
    def hierarchical_procedure_id(self) -> str:
        return self.hierarchical_procedure.procedure_id

    @property
    def required_dependence_structures(self) -> tuple[str, ...]:
        return _REQUIRED_DEPENDENCE_STRUCTURES

    @property
    def required_prevalences(self) -> tuple[float, ...]:
        return _REQUIRED_PREVALENCES

    def scenario_id(
        self,
        dependence_structure: G3FrequencyDependenceStructure | str,
        non_null_prevalence: float | None,
    ) -> str:
        dependence = G3FrequencyDependenceStructure(dependence_structure)
        prevalence = _prevalence(non_null_prevalence)
        return stable_id(
            "g3_frequency_calibration_scenario_cell",
            {
                "protocol_id": self.protocol_id,
                "dependence_structure": dependence.value,
                "non_null_prevalence": prevalence,
            },
            schema_version="2",
        )

    def replicate_seed_lineage(
        self,
        dependence_structure: G3FrequencyDependenceStructure | str,
        non_null_prevalence: float | None,
        replicate_index: int,
    ) -> SeedLineage:
        index = _nonnegative_integer(replicate_index, field_name="replicate_index")
        scenario_id = self.scenario_id(dependence_structure, non_null_prevalence)
        return self.seed_lineage.derive(
            "g3f_calibration_v2",
            f"scenario_id={scenario_id}",
            f"replicate_index={index:06d}",
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "campaign_name": self.campaign_name,
            "generator_id": self.generator_id,
            "generator_manifest": self.generator_manifest.to_dict(),
            "noninferiority_baseline_id": self.noninferiority_baseline_id,
            "power_noninferiority_estimand": _POWER_NONINFERIORITY_ESTIMAND,
            "power_noninferiority_direction": _NONINFERIORITY_DIRECTION,
            "power_noninferiority_margin": _POWER_NONINFERIORITY_MARGIN,
            "interval_width_noninferiority_estimand": (
                _INTERVAL_WIDTH_NONINFERIORITY_ESTIMAND
            ),
            "interval_width_noninferiority_direction": _NONINFERIORITY_DIRECTION,
            "interval_width_noninferiority_margin": (
                _INTERVAL_WIDTH_NONINFERIORITY_MARGIN
            ),
            "hypothesis_universe_id": self.hypothesis_universe.universe_id,
            "hierarchical_procedure_id": self.hierarchical_procedure.procedure_id,
            "seed_lineage": self.seed_lineage.to_dict(),
            "required_dependence_structures": list(_REQUIRED_DEPENDENCE_STRUCTURES),
            "required_prevalences": list(_REQUIRED_PREVALENCES),
            "global_null_metrics": [item.value for item in _GLOBAL_NULL_METRICS],
            "mixed_metrics": [item.value for item in _MIXED_METRICS],
            "minimum_replicates_per_cell": _MIN_CALIBRATION_REPLICATES,
            "monte_carlo_bootstrap_resamples": (_MONTE_CARLO_BOOTSTRAP_RESAMPLES),
            "confidence_level": _CONFIDENCE_LEVEL,
            "metric_weighting": _METRIC_WEIGHTING,
            "realized_prevalence_audit_z": _PREVALENCE_MONTE_CARLO_Z,
            "schema_version": self.schema_version,
        }

    def _require_intact(self) -> None:
        try:
            self.hypothesis_universe._require_intact()
            self.hierarchical_procedure._require_intact()
            repeated = G3FrequencyCalibrationProtocol(
                campaign_name=self.campaign_name,
                generator_manifest=self.generator_manifest,
                noninferiority_baseline_id=self.noninferiority_baseline_id,
                hypothesis_universe=self.hypothesis_universe,
                hierarchical_procedure=self.hierarchical_procedure,
                seed_lineage=self.seed_lineage,
                schema_version=self.schema_version,
            )
            valid = repeated.protocol_id == self.protocol_id
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-F protocol failed integrity validation",
                code="g3_frequency_protocol_integrity_violation",
                field="protocol_id",
                remediation="Recreate the frozen G3-F protocol",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-F protocol failed integrity validation",
                code="g3_frequency_protocol_integrity_violation",
                field="protocol_id",
                remediation="Recreate the frozen G3-F protocol",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"protocol_id": self.protocol_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class G3FrequencyCalibrationReplicate:
    """Raw truth/decision/interval ledger for one simulated dataset."""

    protocol_id: str
    dependence_structure: G3FrequencyDependenceStructure | str
    non_null_prevalence: float | None
    replicate_index: int
    seed_lineage: SeedLineage
    source_artifact_id: str
    hypothesis_ids: tuple[str, ...]
    hypothesis_roles: tuple[HypothesisRole | str, ...]
    parent_hypothesis_ids: tuple[str | None, ...]
    truth_non_null: np.ndarray
    rejected_at_alpha: np.ndarray
    baseline_rejected_at_alpha: np.ndarray
    ci_contains_truth: np.ndarray
    ci_interval_width: np.ndarray
    baseline_ci_interval_width: np.ndarray
    permutation_null_p_value: float | None
    status: G3FrequencyReplicateStatus | str = G3FrequencyReplicateStatus.OBSERVED
    reason_code: str | None = None
    replicate_id: str = field(init=False)

    def __post_init__(self) -> None:
        protocol_id = _name(self.protocol_id, field_name="protocol_id")
        dependence = G3FrequencyDependenceStructure(self.dependence_structure)
        prevalence = _prevalence(self.non_null_prevalence)
        index = _nonnegative_integer(self.replicate_index, field_name="replicate_index")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        source = _name(self.source_artifact_id, field_name="source_artifact_id")
        hypothesis_ids = tuple(
            _name(item, field_name="hypothesis_ids") for item in self.hypothesis_ids
        )
        if not hypothesis_ids or len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("hypothesis_ids must contain unique identifiers")
        roles = tuple(HypothesisRole(item) for item in self.hypothesis_roles)
        parents = tuple(
            _optional_name(item, field_name="parent_hypothesis_ids")
            for item in self.parent_hypothesis_ids
        )
        bool_vectors = (
            _immutable_bool_array(self.truth_non_null, field_name="truth_non_null"),
            _immutable_bool_array(
                self.rejected_at_alpha, field_name="rejected_at_alpha"
            ),
            _immutable_bool_array(
                self.baseline_rejected_at_alpha,
                field_name="baseline_rejected_at_alpha",
            ),
            _immutable_bool_array(
                self.ci_contains_truth, field_name="ci_contains_truth"
            ),
        )
        width_vectors = (
            _immutable_float_array(
                self.ci_interval_width,
                field_name="ci_interval_width",
            ),
            _immutable_float_array(
                self.baseline_ci_interval_width,
                field_name="baseline_ci_interval_width",
            ),
        )
        if any(
            len(item) != len(hypothesis_ids)
            for item in (*bool_vectors, *width_vectors, roles, parents)
        ):
            raise ValueError("G3-F ledger columns must align with hypothesis_ids")
        primary_ids = {
            hypothesis_id
            for hypothesis_id, role in zip(hypothesis_ids, roles, strict=True)
            if role is HypothesisRole.PRIMARY
        }
        if not primary_ids or HypothesisRole.SECONDARY not in roles:
            raise ValueError("G3-F ledger requires primary and secondary hypotheses")
        for hypothesis_id, role, parent in zip(
            hypothesis_ids, roles, parents, strict=True
        ):
            if role is HypothesisRole.PRIMARY and parent is not None:
                raise ValueError(
                    f"primary hypothesis {hypothesis_id} cannot have a parent"
                )
            if role is HypothesisRole.SECONDARY and parent not in primary_ids:
                raise ValueError(
                    f"secondary hypothesis {hypothesis_id} has an invalid parent"
                )
        truth, rejected, baseline_rejected, coverage = bool_vectors
        interval_width, baseline_interval_width = width_vectors
        paired_missing_width = np.isnan(interval_width) != np.isnan(
            baseline_interval_width
        )
        invalid_candidate_width = ~np.isnan(interval_width) & (
            ~np.isfinite(interval_width) | (interval_width < 0.0)
        )
        invalid_baseline_width = ~np.isnan(baseline_interval_width) & (
            ~np.isfinite(baseline_interval_width) | (baseline_interval_width <= 0.0)
        )
        if (
            np.any(paired_missing_width)
            or np.any(invalid_candidate_width)
            or np.any(invalid_baseline_width)
        ):
            raise ValueError(
                "candidate/baseline interval widths must be paired, finite, "
                "candidate >= 0, and baseline > 0"
            )
        truth_by_id = dict(zip(hypothesis_ids, truth.tolist(), strict=True))
        rejected_by_id = dict(zip(hypothesis_ids, rejected.tolist(), strict=True))
        baseline_rejected_by_id = dict(
            zip(hypothesis_ids, baseline_rejected.tolist(), strict=True)
        )
        for position, role in enumerate(roles):
            if role is not HypothesisRole.SECONDARY:
                continue
            parent = cast(str, parents[position])
            if bool(truth[position]) and not bool(truth_by_id[parent]):
                raise ValueError("non-null secondary requires a non-null parent")
            if bool(rejected[position]) and not bool(rejected_by_id[parent]):
                raise ValueError("rejected secondary requires a rejected parent")
            if bool(baseline_rejected[position]) and not bool(
                baseline_rejected_by_id[parent]
            ):
                raise ValueError(
                    "baseline-rejected secondary requires a baseline-rejected parent"
                )
        if prevalence is None:
            if np.any(truth):
                raise ValueError(
                    "global-null G3-F replicate cannot contain non-null truth"
                )
            permutation = (
                None
                if self.permutation_null_p_value is None
                else _unit(
                    self.permutation_null_p_value,
                    field_name="permutation_null_p_value",
                )
            )
        else:
            if self.permutation_null_p_value is not None:
                raise ValueError("mixed G3-F replicate cannot carry a null p-value")
            permutation = None
        status = G3FrequencyReplicateStatus(self.status)
        reason = _optional_name(self.reason_code, field_name="reason_code")
        if status is G3FrequencyReplicateStatus.OBSERVED:
            if reason is not None:
                raise ValueError("observed G3-F replicate cannot have a reason")
            if prevalence is None and permutation is None:
                raise ValueError("observed global-null replicate requires a p-value")
        else:
            if reason is None:
                raise ValueError("non-observed G3-F replicate requires a reason_code")
            if (
                np.any(rejected)
                or np.any(baseline_rejected)
                or np.any(coverage)
                or np.any(~np.isnan(interval_width))
                or np.any(~np.isnan(baseline_interval_width))
                or permutation is not None
            ):
                raise ValueError(
                    "non-observed G3-F replicate cannot carry decisions or metrics"
                )
        order = np.argsort(np.asarray(hypothesis_ids, dtype=str), kind="stable")
        hypothesis_ids = tuple(hypothesis_ids[int(item)] for item in order)
        roles = tuple(roles[int(item)] for item in order)
        parents = tuple(parents[int(item)] for item in order)
        truth = _immutable_bool_array(truth[order], field_name="truth_non_null")
        rejected = _immutable_bool_array(
            rejected[order], field_name="rejected_at_alpha"
        )
        baseline_rejected = _immutable_bool_array(
            baseline_rejected[order], field_name="baseline_rejected_at_alpha"
        )
        coverage = _immutable_bool_array(
            coverage[order], field_name="ci_contains_truth"
        )
        interval_width = _immutable_float_array(
            interval_width[order], field_name="ci_interval_width"
        )
        baseline_interval_width = _immutable_float_array(
            baseline_interval_width[order],
            field_name="baseline_ci_interval_width",
        )
        values: Mapping[str, object] = {
            "protocol_id": protocol_id,
            "dependence_structure": dependence,
            "non_null_prevalence": prevalence,
            "replicate_index": index,
            "source_artifact_id": source,
            "hypothesis_ids": hypothesis_ids,
            "hypothesis_roles": roles,
            "parent_hypothesis_ids": parents,
            "truth_non_null": truth,
            "rejected_at_alpha": rejected,
            "baseline_rejected_at_alpha": baseline_rejected,
            "ci_contains_truth": coverage,
            "ci_interval_width": interval_width,
            "baseline_ci_interval_width": baseline_interval_width,
            "permutation_null_p_value": permutation,
            "status": status,
            "reason_code": reason,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "replicate_id",
            stable_id(
                "g3_frequency_calibration_replicate",
                self._identity_payload(),
                schema_version="2",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "dependence_structure": G3FrequencyDependenceStructure(
                self.dependence_structure
            ).value,
            "non_null_prevalence": self.non_null_prevalence,
            "replicate_index": self.replicate_index,
            "seed_lineage": self.seed_lineage.to_dict(),
            "source_artifact_id": self.source_artifact_id,
            "hypothesis_ids": list(self.hypothesis_ids),
            "hypothesis_roles": [
                HypothesisRole(item).value for item in self.hypothesis_roles
            ],
            "parent_hypothesis_ids": list(self.parent_hypothesis_ids),
            "truth_non_null_digest": _array_digest(self.truth_non_null, dtype="|b1"),
            "rejected_at_alpha_digest": _array_digest(
                self.rejected_at_alpha, dtype="|b1"
            ),
            "baseline_rejected_at_alpha_digest": _array_digest(
                self.baseline_rejected_at_alpha, dtype="|b1"
            ),
            "ci_contains_truth_digest": _array_digest(
                self.ci_contains_truth, dtype="|b1"
            ),
            "ci_interval_width_digest": _array_digest(
                self.ci_interval_width, dtype="<f8"
            ),
            "baseline_ci_interval_width_digest": _array_digest(
                self.baseline_ci_interval_width, dtype="<f8"
            ),
            "permutation_null_p_value": self.permutation_null_p_value,
            "status": G3FrequencyReplicateStatus(self.status).value,
            "reason_code": self.reason_code,
        }

    def _require_intact(self) -> None:
        try:
            repeated = G3FrequencyCalibrationReplicate(
                protocol_id=self.protocol_id,
                dependence_structure=self.dependence_structure,
                non_null_prevalence=self.non_null_prevalence,
                replicate_index=self.replicate_index,
                seed_lineage=self.seed_lineage,
                source_artifact_id=self.source_artifact_id,
                hypothesis_ids=self.hypothesis_ids,
                hypothesis_roles=self.hypothesis_roles,
                parent_hypothesis_ids=self.parent_hypothesis_ids,
                truth_non_null=self.truth_non_null,
                rejected_at_alpha=self.rejected_at_alpha,
                baseline_rejected_at_alpha=self.baseline_rejected_at_alpha,
                ci_contains_truth=self.ci_contains_truth,
                ci_interval_width=self.ci_interval_width,
                baseline_ci_interval_width=self.baseline_ci_interval_width,
                permutation_null_p_value=self.permutation_null_p_value,
                status=self.status,
                reason_code=self.reason_code,
            )
            valid = repeated.replicate_id == self.replicate_id
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-F replicate ledger failed integrity validation",
                code="g3_frequency_replicate_integrity_violation",
                field="replicate_id",
                remediation="Regenerate the complete replicate ledger",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-F replicate ledger failed integrity validation",
                code="g3_frequency_replicate_integrity_violation",
                field="replicate_id",
                remediation="Regenerate the complete replicate ledger",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"replicate_id": self.replicate_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, init=False)
class G3CalibrationScenarioResult:
    """Producer-owned point estimate and one-sided Monte Carlo bound."""

    scenario_result_id: str
    scenario_id: str
    protocol_id: str
    dependence_structure: str
    non_null_prevalence: float | None
    metric: G3CalibrationMetric
    status: G3CalibrationScenarioStatus
    reason_code: str | None
    n_replicates: int
    n_observed_replicates: int
    n_metric_replicates: int
    replicate_ids: tuple[str, ...]
    point_estimate: float | None
    one_sided_bound: float | None
    bootstrap_seed: int | None
    noninferiority_baseline_id: str | None
    noninferiority_estimand: str | None
    noninferiority_direction: str | None
    noninferiority_margin: float | None
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "G3CalibrationScenarioResult is producer-owned; use "
            "summarize_g3_frequency_calibration_campaign()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "protocol_id": self.protocol_id,
            "dependence_structure": self.dependence_structure,
            "non_null_prevalence": self.non_null_prevalence,
            "metric": self.metric.value,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_replicates": self.n_replicates,
            "n_observed_replicates": self.n_observed_replicates,
            "n_metric_replicates": self.n_metric_replicates,
            "replicate_ids": list(self.replicate_ids),
            "point_estimate": self.point_estimate,
            "one_sided_bound": self.one_sided_bound,
            "bootstrap_seed": self.bootstrap_seed,
            "noninferiority_baseline_id": self.noninferiority_baseline_id,
            "noninferiority_estimand": self.noninferiority_estimand,
            "noninferiority_direction": self.noninferiority_direction,
            "noninferiority_margin": self.noninferiority_margin,
            "metric_weighting": _METRIC_WEIGHTING,
            "producer_marker": _PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            expected = stable_id(
                "g3_frequency_calibration_scenario_result",
                self._identity_payload(),
                schema_version="3",
            )
            valid = (
                self._producer_token is _PRODUCER_TOKEN
                and expected == self.scenario_result_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-F scenario result failed integrity validation",
                code="g3_calibration_scenario_integrity_violation",
                field="scenario_result_id",
                remediation="Rebuild the scenario from raw replicate ledgers",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-F scenario result failed integrity validation",
                code="g3_calibration_scenario_integrity_violation",
                field="scenario_result_id",
                remediation="Rebuild the scenario from raw replicate ledgers",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"scenario_result_id": self.scenario_result_id, **payload}


@dataclass(frozen=True, slots=True, init=False)
class G3FrequencyPermutationDiagnostic:
    """Producer-owned DKW diagnostic for one global-null structure."""

    diagnostic_id: str
    protocol_id: str
    dependence_structure: str
    status: G3CalibrationScenarioStatus
    reason_code: str | None
    n_replicates: int
    p_values: np.ndarray
    ks_distance: float | None
    dkw_limit: float | None
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError("G3FrequencyPermutationDiagnostic is producer-owned")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "dependence_structure": self.dependence_structure,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_replicates": self.n_replicates,
            "p_values_digest": _array_digest(self.p_values, dtype="<f8"),
            "ks_distance": self.ks_distance,
            "dkw_limit": self.dkw_limit,
            "producer_marker": _PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            expected = stable_id(
                "g3_frequency_permutation_diagnostic",
                self._identity_payload(),
                schema_version="1",
            )
            valid = (
                self._producer_token is _PRODUCER_TOKEN
                and expected == self.diagnostic_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-F permutation diagnostic failed integrity validation",
                code="g3_frequency_permutation_diagnostic_integrity_violation",
                field="diagnostic_id",
                remediation="Rebuild diagnostics from raw global-null ledgers",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-F permutation diagnostic failed integrity validation",
                code="g3_frequency_permutation_diagnostic_integrity_violation",
                field="diagnostic_id",
                remediation="Rebuild diagnostics from raw global-null ledgers",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "diagnostic_id": self.diagnostic_id,
            "protocol_id": self.protocol_id,
            "dependence_structure": self.dependence_structure,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_replicates": self.n_replicates,
            "ks_distance": self.ks_distance,
            "dkw_limit": self.dkw_limit,
        }


@dataclass(frozen=True, slots=True, init=False)
class G3FrequencyCalibrationEvidence:
    """Producer-owned complete-grid evidence consumed by the G3-F gate."""

    evidence_id: str
    source_artifact_id: str
    protocol_id: str
    hypothesis_universe_id: str
    hierarchical_procedure_id: str
    noninferiority_baseline_id: str
    power_noninferiority_margin: float
    interval_width_noninferiority_margin: float
    noninferiority_direction: str
    generator_id: str
    generator_attestation_id: str | None
    generator_verified: bool
    required_dependence_structures: tuple[str, ...]
    scenarios: tuple[G3CalibrationScenarioResult, ...]
    permutation_diagnostics: tuple[G3FrequencyPermutationDiagnostic, ...]
    permutation_null_p_values: np.ndarray
    _producer_token: object
    _generator_authorization_token: object

    def __init__(self) -> None:
        raise TypeError(
            "G3FrequencyCalibrationEvidence is producer-owned; use "
            "summarize_g3_frequency_calibration_campaign()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "source_artifact_id": self.source_artifact_id,
            "protocol_id": self.protocol_id,
            "hypothesis_universe_id": self.hypothesis_universe_id,
            "hierarchical_procedure_id": self.hierarchical_procedure_id,
            "noninferiority_baseline_id": self.noninferiority_baseline_id,
            "power_noninferiority_margin": self.power_noninferiority_margin,
            "interval_width_noninferiority_margin": (
                self.interval_width_noninferiority_margin
            ),
            "noninferiority_direction": self.noninferiority_direction,
            "generator_id": self.generator_id,
            "generator_attestation_id": self.generator_attestation_id,
            "generator_verified": self.generator_verified,
            "required_dependence_structures": list(self.required_dependence_structures),
            "scenario_result_ids": [item.scenario_result_id for item in self.scenarios],
            "permutation_diagnostic_ids": [
                item.diagnostic_id for item in self.permutation_diagnostics
            ],
            "permutation_null_p_values_digest": _array_digest(
                self.permutation_null_p_values, dtype="<f8"
            ),
            "producer_marker": _PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            for scenario in self.scenarios:
                scenario._require_intact()
            for diagnostic in self.permutation_diagnostics:
                diagnostic._require_intact()
            expected = stable_id(
                "g3_frequency_calibration_evidence",
                self._identity_payload(),
                schema_version="4",
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
                    not self.generator_verified
                    or self.generator_attestation_id is not None
                )
                and expected == self.evidence_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "G3-F calibration evidence failed integrity validation",
                code="g3_calibration_evidence_integrity_violation",
                field="evidence_id",
                remediation="Rebuild evidence from the intact campaign",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-F calibration evidence failed integrity validation",
                code="g3_calibration_evidence_integrity_violation",
                field="evidence_id",
                remediation="Rebuild evidence from the intact campaign",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "evidence_id": self.evidence_id,
            "source_artifact_id": self.source_artifact_id,
            "protocol_id": self.protocol_id,
            "hypothesis_universe_id": self.hypothesis_universe_id,
            "hierarchical_procedure_id": self.hierarchical_procedure_id,
            "noninferiority_baseline_id": self.noninferiority_baseline_id,
            "power_noninferiority_margin": self.power_noninferiority_margin,
            "interval_width_noninferiority_margin": (
                self.interval_width_noninferiority_margin
            ),
            "noninferiority_direction": self.noninferiority_direction,
            "generator_id": self.generator_id,
            "generator_attestation_id": self.generator_attestation_id,
            "generator_verified": self.generator_verified,
            "required_dependence_structures": list(self.required_dependence_structures),
            "scenario_results": [item.to_dict() for item in self.scenarios],
            "permutation_diagnostics": [
                item.to_dict() for item in self.permutation_diagnostics
            ],
            "n_permutation_null_statistics": len(self.permutation_null_p_values),
        }


@dataclass(frozen=True, slots=True, init=False)
class G3FrequencyCalibrationCampaign:
    """Producer-owned raw campaign and its derived evidence."""

    protocol: G3FrequencyCalibrationProtocol
    replicates: tuple[G3FrequencyCalibrationReplicate, ...]
    scenarios: tuple[G3CalibrationScenarioResult, ...]
    permutation_diagnostics: tuple[G3FrequencyPermutationDiagnostic, ...]
    campaign_id: str
    evidence: G3FrequencyCalibrationEvidence
    generator_attestation: CalibrationReplayAttestation | None
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "G3FrequencyCalibrationCampaign is producer-owned; use "
            "summarize_g3_frequency_calibration_campaign()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol.protocol_id,
            "replicate_ids": [item.replicate_id for item in self.replicates],
            "scenario_result_ids": [item.scenario_result_id for item in self.scenarios],
            "permutation_diagnostic_ids": [
                item.diagnostic_id for item in self.permutation_diagnostics
            ],
            "producer_marker": _PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            self.protocol._require_intact()
            for replicate in self.replicates:
                replicate._require_intact()
            for scenario in self.scenarios:
                scenario._require_intact()
            for diagnostic in self.permutation_diagnostics:
                diagnostic._require_intact()
            self.evidence._require_intact()
            if self.generator_attestation is not None:
                self.generator_attestation._require_intact()
            expected = stable_id(
                "g3_frequency_calibration_campaign",
                self._identity_payload(),
                schema_version="2",
            )
            valid = (
                self._producer_token is _PRODUCER_TOKEN
                and expected == self.campaign_id
                and self.evidence.source_artifact_id == self.campaign_id
                and tuple(
                    item.scenario_result_id for item in self.evidence.scenarios
                )
                == tuple(item.scenario_result_id for item in self.scenarios)
                and tuple(
                    item.diagnostic_id
                    for item in self.evidence.permutation_diagnostics
                )
                == tuple(item.diagnostic_id for item in self.permutation_diagnostics)
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
                "G3-F campaign failed integrity validation",
                code="g3_frequency_campaign_integrity_violation",
                field="campaign_id",
                remediation="Rebuild the campaign from raw replicate ledgers",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-F campaign failed integrity validation",
                code="g3_frequency_campaign_integrity_violation",
                field="campaign_id",
                remediation="Rebuild the campaign from raw replicate ledgers",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "campaign_id": self.campaign_id,
            "protocol": self.protocol.to_dict(),
            "replicate_ids": [item.replicate_id for item in self.replicates],
            "scenario_results": [item.to_dict() for item in self.scenarios],
            "permutation_diagnostics": [
                item.to_dict() for item in self.permutation_diagnostics
            ],
            "evidence": self.evidence.to_dict(),
        }


def _expected_ledger_columns(
    protocol: G3FrequencyCalibrationProtocol,
) -> tuple[tuple[str, ...], tuple[HypothesisRole, ...], tuple[str | None, ...]]:
    declarations = protocol.hypothesis_universe.declarations
    parent_by_key = {
        item.hypothesis_key: item.hypothesis_id
        for item in declarations
        if item.role is HypothesisRole.PRIMARY
    }
    rows = tuple(
        sorted(
            (
                item.hypothesis_id,
                item.role,
                None
                if item.role is HypothesisRole.PRIMARY
                else parent_by_key[cast(str, item.parent_key)],
            )
            for item in declarations
        )
    )
    return (
        tuple(item[0] for item in rows),
        tuple(item[1] for item in rows),
        tuple(item[2] for item in rows),
    )


def _included_hypothesis_mask(
    protocol: G3FrequencyCalibrationProtocol,
) -> np.ndarray:
    rows = tuple(
        sorted(
            (item.hypothesis_id, item.prefilter_status)
            for item in protocol.hypothesis_universe.declarations
        )
    )
    return cast(
        np.ndarray,
        np.asarray(
            tuple(status is HypothesisPrefilterStatus.INCLUDED for _, status in rows),
            dtype=bool,
        ),
    )


def _required_metrics(prevalence: float | None) -> tuple[G3CalibrationMetric, ...]:
    return _GLOBAL_NULL_METRICS if prevalence is None else _MIXED_METRICS


def _replicate_metric_values(
    replicate: G3FrequencyCalibrationReplicate,
    *,
    included_mask: np.ndarray,
) -> dict[G3CalibrationMetric, float]:
    roles = tuple(HypothesisRole(item) for item in replicate.hypothesis_roles)
    truth: np.ndarray = replicate.truth_non_null.astype(bool, copy=False)
    rejected: np.ndarray = replicate.rejected_at_alpha.astype(bool, copy=False)
    baseline_rejected: np.ndarray = replicate.baseline_rejected_at_alpha.astype(
        bool, copy=False
    )
    false_rejected = rejected & ~truth
    primary_mask = np.asarray(
        tuple(item is HypothesisRole.PRIMARY for item in roles), dtype=bool
    )
    all_rejections = int(np.count_nonzero(rejected))
    primary_rejections = int(np.count_nonzero(rejected & primary_mask))
    values = {
        G3CalibrationMetric.COVERAGE_LOWER: float(
            np.mean(replicate.ci_contains_truth[included_mask])
        ),
        G3CalibrationMetric.NULL_TYPE_I_UPPER: float(
            np.any(false_rejected & primary_mask)
        ),
        G3CalibrationMetric.MIXED_FDR_UPPER: (
            int(np.count_nonzero(false_rejected)) / max(all_rejections, 1)
        ),
        G3CalibrationMetric.PRIMARY_FDR_UPPER: (
            int(np.count_nonzero(false_rejected & primary_mask))
            / max(primary_rejections, 1)
        ),
    }
    selected_primary_ids = {
        replicate.hypothesis_ids[index]
        for index, role in enumerate(roles)
        if role is HypothesisRole.PRIMARY and bool(rejected[index])
    }
    family_fdps: list[float] = []
    for parent_id in sorted(selected_primary_ids):
        child_indexes = tuple(
            index
            for index, role in enumerate(roles)
            if role is HypothesisRole.SECONDARY
            and replicate.parent_hypothesis_ids[index] == parent_id
        )
        child_rejections = sum(bool(rejected[index]) for index in child_indexes)
        child_false = sum(bool(false_rejected[index]) for index in child_indexes)
        family_fdps.append(child_false / max(child_rejections, 1))
    values[G3CalibrationMetric.SELECTIVE_CHILD_FDR_UPPER] = (
        float(np.mean(family_fdps)) if family_fdps else 0.0
    )
    values[G3CalibrationMetric.INTERVAL_WIDTH_NONINFERIORITY_LOWER] = float(
        np.mean(
            (
                replicate.baseline_ci_interval_width[included_mask]
                - replicate.ci_interval_width[included_mask]
            )
            / replicate.baseline_ci_interval_width[included_mask]
        )
    )
    true_included = truth & included_mask
    if np.any(true_included):
        values[G3CalibrationMetric.POWER_NONINFERIORITY_LOWER] = float(
            np.mean(rejected[true_included])
            - np.mean(baseline_rejected[true_included])
        )
    return values


def _cell_metric_values(
    replicates: tuple[G3FrequencyCalibrationReplicate, ...],
    *,
    included_mask: np.ndarray,
    required_metrics: tuple[G3CalibrationMetric, ...],
) -> dict[G3CalibrationMetric, np.ndarray]:
    values_by_replicate = tuple(
        _replicate_metric_values(item, included_mask=included_mask)
        for item in replicates
        if item.status is G3FrequencyReplicateStatus.OBSERVED
    )
    return {
        metric: np.asarray(
            tuple(values[metric] for values in values_by_replicate if metric in values),
            dtype="<f8",
        )
        for metric in required_metrics
    }


def _bootstrap_one_sided_bound(
    values: np.ndarray,
    *,
    side: BinomialBoundSide,
    seed: int,
) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("Monte Carlo bound requires at least two replicates")
    point = float(np.mean(values))
    if np.all(values == values[0]):
        return point, point
    generator = np.random.default_rng(seed)
    output: np.ndarray = np.empty(_MONTE_CARLO_BOOTSTRAP_RESAMPLES, dtype="<f8")
    chunk_size = max(1, min(512, 2_000_000 // len(values)))
    offset = 0
    while offset < len(output):
        size = min(chunk_size, len(output) - offset)
        indexes: np.ndarray = generator.integers(
            0, len(values), size=(size, len(values))
        )
        output[offset : offset + size] = np.mean(values[indexes], axis=1)
        offset += size
    ordered = np.sort(output)
    if side is BinomialBoundSide.UPPER:
        rank = min(
            len(ordered),
            max(1, math.ceil(_CONFIDENCE_LEVEL * (len(ordered) + 1))),
        )
        bound = max(point, float(ordered[rank - 1]))
    else:
        rank = min(
            len(ordered),
            max(1, math.floor((1.0 - _CONFIDENCE_LEVEL) * (len(ordered) + 1))),
        )
        bound = min(point, float(ordered[rank - 1]))
    return point, bound


def _noninferiority_contract(
    protocol: G3FrequencyCalibrationProtocol,
    metric: G3CalibrationMetric,
) -> tuple[str | None, str | None, str | None, float | None]:
    if metric is G3CalibrationMetric.POWER_NONINFERIORITY_LOWER:
        return (
            protocol.noninferiority_baseline_id,
            _POWER_NONINFERIORITY_ESTIMAND,
            _NONINFERIORITY_DIRECTION,
            _POWER_NONINFERIORITY_MARGIN,
        )
    if metric is G3CalibrationMetric.INTERVAL_WIDTH_NONINFERIORITY_LOWER:
        return (
            protocol.noninferiority_baseline_id,
            _INTERVAL_WIDTH_NONINFERIORITY_ESTIMAND,
            _NONINFERIORITY_DIRECTION,
            _INTERVAL_WIDTH_NONINFERIORITY_MARGIN,
        )
    return None, None, None, None


def _new_scenario_result(
    *,
    protocol: G3FrequencyCalibrationProtocol,
    dependence: G3FrequencyDependenceStructure,
    prevalence: float | None,
    metric: G3CalibrationMetric,
    replicates: tuple[G3FrequencyCalibrationReplicate, ...],
    metric_values: np.ndarray | None,
    reason: str | None,
) -> G3CalibrationScenarioResult:
    scenario_id = protocol.scenario_id(dependence, prevalence)
    observed = tuple(
        item
        for item in replicates
        if item.status is G3FrequencyReplicateStatus.OBSERVED
    )
    point: float | None = None
    bound: float | None = None
    bootstrap_seed: int | None = None
    n_metric_replicates = 0
    local_reason = reason
    if local_reason is None:
        if metric_values is None:
            local_reason = "g3_frequency_metric_ledger_incomplete"
        else:
            n_metric_replicates = len(metric_values)
        if local_reason is None and (
            metric is not G3CalibrationMetric.POWER_NONINFERIORITY_LOWER
            and n_metric_replicates != len(observed)
        ):
            local_reason = "g3_frequency_metric_ledger_incomplete"
        elif (
            local_reason is None
            and metric is G3CalibrationMetric.NULL_TYPE_I_UPPER
        ):
            if metric_values is None:
                raise RuntimeError("G3-F metric values unexpectedly unavailable")
            exact = exact_binomial_one_sided_bound(
                int(np.sum(metric_values)),
                len(metric_values),
                side=BinomialBoundSide.UPPER,
                confidence_level=_CONFIDENCE_LEVEL,
            )
            point = exact.estimate
            bound = exact.bound
        elif local_reason is None:
            if metric_values is None:
                raise RuntimeError("G3-F metric values unexpectedly unavailable")
            bootstrap_seed = protocol.seed_lineage.derive(
                "g3f_monte_carlo_bootstrap_v1",
                f"scenario_id={scenario_id}",
                f"metric={metric.value}",
            ).seed
            side = (
                BinomialBoundSide.LOWER
                if metric
                in {
                    G3CalibrationMetric.COVERAGE_LOWER,
                    G3CalibrationMetric.POWER_NONINFERIORITY_LOWER,
                    G3CalibrationMetric.INTERVAL_WIDTH_NONINFERIORITY_LOWER,
                }
                else BinomialBoundSide.UPPER
            )
            try:
                point, bound = _bootstrap_one_sided_bound(
                    metric_values,
                    side=side,
                    seed=bootstrap_seed,
                )
            except ValueError:
                local_reason = "g3_frequency_too_few_replicates_for_bound"
                point = None
                bound = None
                bootstrap_seed = None
    baseline_id, estimand, direction, margin = _noninferiority_contract(
        protocol, metric
    )
    status = (
        G3CalibrationScenarioStatus.OBSERVED
        if local_reason is None
        else G3CalibrationScenarioStatus.NOT_ESTIMABLE
    )
    self = object.__new__(G3CalibrationScenarioResult)
    for name, value in {
        "scenario_id": scenario_id,
        "protocol_id": protocol.protocol_id,
        "dependence_structure": dependence.value,
        "non_null_prevalence": prevalence,
        "metric": metric,
        "status": status,
        "reason_code": local_reason,
        "n_replicates": len(replicates),
        "n_observed_replicates": len(observed),
        "n_metric_replicates": n_metric_replicates,
        "replicate_ids": tuple(item.replicate_id for item in replicates),
        "point_estimate": point,
        "one_sided_bound": bound,
        "bootstrap_seed": bootstrap_seed,
        "noninferiority_baseline_id": baseline_id,
        "noninferiority_estimand": estimand,
        "noninferiority_direction": direction,
        "noninferiority_margin": margin,
        "_producer_token": _PRODUCER_TOKEN,
    }.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "scenario_result_id",
        stable_id(
            "g3_frequency_calibration_scenario_result",
            self._identity_payload(),
            schema_version="3",
        ),
    )
    self._require_intact()
    return self


def _scenario_reason(
    protocol: G3FrequencyCalibrationProtocol,
    dependence: G3FrequencyDependenceStructure,
    prevalence: float | None,
    replicates: tuple[G3FrequencyCalibrationReplicate, ...],
) -> str | None:
    if not replicates:
        return "g3_frequency_calibration_missing_replicates"
    expected_indexes = tuple(range(len(replicates)))
    if tuple(item.replicate_index for item in replicates) != expected_indexes:
        return "g3_frequency_calibration_noncontiguous_replicates"
    if any(item.protocol_id != protocol.protocol_id for item in replicates):
        return "g3_frequency_calibration_protocol_mismatch"
    if any(
        item.seed_lineage
        != protocol.replicate_seed_lineage(dependence, prevalence, item.replicate_index)
        for item in replicates
    ):
        return "g3_frequency_calibration_seed_lineage_mismatch"
    if any(
        item.status is not G3FrequencyReplicateStatus.OBSERVED for item in replicates
    ):
        return "g3_frequency_calibration_nonobserved_replicate"
    if len({item.source_artifact_id for item in replicates}) != len(replicates):
        return "g3_frequency_calibration_duplicate_source_artifact"
    expected_ids, expected_roles, expected_parents = _expected_ledger_columns(protocol)
    if any(
        item.hypothesis_ids != expected_ids
        or tuple(HypothesisRole(role) for role in item.hypothesis_roles)
        != expected_roles
        or item.parent_hypothesis_ids != expected_parents
        for item in replicates
    ):
        return "g3_frequency_calibration_universe_coverage_mismatch"
    included_mask = _included_hypothesis_mask(protocol)
    if any(
        np.any(item.rejected_at_alpha[~included_mask])
        or np.any(item.baseline_rejected_at_alpha[~included_mask])
        for item in replicates
    ):
        return "g3_frequency_prefiltered_hypothesis_rejected"
    if any(np.any(item.ci_contains_truth[~included_mask]) for item in replicates):
        return "g3_frequency_prefiltered_hypothesis_has_coverage"
    if any(
        np.any(~np.isnan(item.ci_interval_width[~included_mask]))
        or np.any(~np.isnan(item.baseline_ci_interval_width[~included_mask]))
        for item in replicates
    ):
        return "g3_frequency_prefiltered_hypothesis_has_interval_width"
    if any(
        np.any(~np.isfinite(item.ci_interval_width[included_mask]))
        or np.any(~np.isfinite(item.baseline_ci_interval_width[included_mask]))
        for item in replicates
    ):
        return "g3_frequency_interval_width_coverage_incomplete"
    if prevalence is not None:
        n_trials = len(replicates) * len(expected_ids)
        realized = (
            sum(int(np.count_nonzero(item.truth_non_null)) for item in replicates)
            / n_trials
        )
        standard_error = math.sqrt(prevalence * (1.0 - prevalence) / n_trials)
        tolerance = max(
            _PREVALENCE_MONTE_CARLO_Z * standard_error,
            1.0 / n_trials,
        )
        if abs(realized - prevalence) > tolerance:
            return "g3_frequency_calibration_realized_prevalence_mismatch"
    return None


def _permutation_ks(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(values)
    n_values = len(ordered)
    if not n_values:
        return math.inf, 0.0
    indexes: np.ndarray = np.arange(1, n_values + 1, dtype=float)
    distance = max(
        float(np.max(indexes / n_values - ordered)),
        float(np.max(ordered - (indexes - 1.0) / n_values)),
    )
    limit = math.sqrt(-math.log(_PERMUTATION_DKW_ALPHA / 2.0) / (2 * n_values))
    return distance, limit


def _new_permutation_diagnostic(
    *,
    protocol: G3FrequencyCalibrationProtocol,
    dependence: G3FrequencyDependenceStructure,
    replicates: tuple[G3FrequencyCalibrationReplicate, ...],
    reason: str | None,
) -> G3FrequencyPermutationDiagnostic:
    p_values = _immutable_float_array(
        tuple(
            item.permutation_null_p_value
            for item in replicates
            if item.status is G3FrequencyReplicateStatus.OBSERVED
            and item.permutation_null_p_value is not None
        ),
        field_name="permutation_null_p_values",
    )
    local_reason = reason
    distance: float | None = None
    limit: float | None = None
    if local_reason is None:
        if len(p_values) != len(replicates):
            local_reason = "g3_frequency_permutation_p_value_coverage_incomplete"
        else:
            distance, limit = _permutation_ks(p_values)
    status = (
        G3CalibrationScenarioStatus.OBSERVED
        if local_reason is None
        else G3CalibrationScenarioStatus.NOT_ESTIMABLE
    )
    self = object.__new__(G3FrequencyPermutationDiagnostic)
    for name, value in {
        "protocol_id": protocol.protocol_id,
        "dependence_structure": dependence.value,
        "status": status,
        "reason_code": local_reason,
        "n_replicates": len(replicates),
        "p_values": p_values,
        "ks_distance": distance,
        "dkw_limit": limit,
        "_producer_token": _PRODUCER_TOKEN,
    }.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "diagnostic_id",
        stable_id(
            "g3_frequency_permutation_diagnostic",
            self._identity_payload(),
            schema_version="1",
        ),
    )
    self._require_intact()
    return self


def _summarize_g3_frequency_calibration_campaign(
    protocol: G3FrequencyCalibrationProtocol,
    replicates: Sequence[G3FrequencyCalibrationReplicate],
    *,
    generator_attestation: CalibrationReplayAttestation | None,
) -> G3FrequencyCalibrationCampaign:
    """Derive the fixed grid with an optional producer-owned replay proof."""

    if not isinstance(protocol, G3FrequencyCalibrationProtocol):
        raise TypeError("protocol must be a G3FrequencyCalibrationProtocol")
    protocol._require_intact()
    supplied = tuple(replicates)
    if any(not isinstance(item, G3FrequencyCalibrationReplicate) for item in supplied):
        raise TypeError(
            "replicates must contain G3FrequencyCalibrationReplicate values"
        )
    if generator_attestation is not None:
        if not isinstance(generator_attestation, CalibrationReplayAttestation):
            raise TypeError(
                "generator_attestation must be a CalibrationReplayAttestation"
            )
        generator_attestation._require_intact()
        if (
            generator_attestation.campaign_kind
            is not CalibrationCampaignKind.G3_FREQUENCY
            or generator_attestation.protocol_id != protocol.protocol_id
            or generator_attestation.generator_id != protocol.generator_id
        ):
            raise _contract_error(
                "G3-F generator attestation belongs to a different protocol",
                code="g3_frequency_generator_attestation_mismatch",
                field="generator_attestation_id",
                remediation="Replay the complete campaign under this protocol",
            )
    for replicate in supplied:
        replicate._require_intact()
        if replicate.protocol_id != protocol.protocol_id:
            raise _contract_error(
                "G3-F replicate belongs to a different protocol",
                code="g3_frequency_replicate_protocol_mismatch",
                field="protocol_id",
                remediation="Submit only ledgers generated under this protocol",
            )
    ordered = tuple(
        sorted(
            supplied,
            key=lambda item: (
                G3FrequencyDependenceStructure(item.dependence_structure).value,
                -1.0 if item.non_null_prevalence is None else item.non_null_prevalence,
                item.replicate_index,
            ),
        )
    )
    keys = tuple(
        (
            G3FrequencyDependenceStructure(item.dependence_structure).value,
            item.non_null_prevalence,
            item.replicate_index,
        )
        for item in ordered
    )
    if len(keys) != len(set(keys)):
        raise _contract_error(
            "G3-F campaign contains duplicate scenario replicate cells",
            code="duplicate_g3_frequency_calibration_replicate",
            field="dependence_structure,non_null_prevalence,replicate_index",
            remediation="Retain exactly one ledger for every attempted replicate",
        )
    if generator_attestation is not None and (
        generator_attestation.replicate_ids
        != tuple(item.replicate_id for item in ordered)
    ):
        raise _contract_error(
            "G3-F generator attestation does not cover the supplied ledgers",
            code="g3_frequency_generator_attestation_coverage_mismatch",
            field="replicate_ids",
            remediation="Replay the complete supplied campaign without adding rows",
        )
    if len({item.source_artifact_id for item in ordered}) != len(ordered):
        raise _contract_error(
            "G3-F source artifact IDs must be unique across the campaign",
            code="duplicate_g3_frequency_source_artifact",
            field="source_artifact_id",
            remediation="Persist one distinct full-pipeline result per replicate",
        )
    scenario_results: list[G3CalibrationScenarioResult] = []
    diagnostics: list[G3FrequencyPermutationDiagnostic] = []
    included_mask = _included_hypothesis_mask(protocol)
    for dependence in G3FrequencyDependenceStructure:
        for prevalence in (None, *_REQUIRED_PREVALENCES):
            cell = tuple(
                item
                for item in ordered
                if item.dependence_structure is dependence
                and item.non_null_prevalence == prevalence
            )
            reason = _scenario_reason(protocol, dependence, prevalence, cell)
            required_metrics = _required_metrics(prevalence)
            metric_values = (
                {}
                if reason is not None
                else _cell_metric_values(
                    cell,
                    included_mask=included_mask,
                    required_metrics=required_metrics,
                )
            )
            scenario_results.extend(
                _new_scenario_result(
                    protocol=protocol,
                    dependence=dependence,
                    prevalence=prevalence,
                    metric=metric,
                    replicates=cell,
                    metric_values=metric_values.get(metric),
                    reason=reason,
                )
                for metric in required_metrics
            )
            if prevalence is None:
                diagnostics.append(
                    _new_permutation_diagnostic(
                        protocol=protocol,
                        dependence=dependence,
                        replicates=cell,
                        reason=reason,
                    )
                )
    scenarios = tuple(scenario_results)
    permutation_diagnostics = tuple(diagnostics)
    campaign_payload = {
        "protocol_id": protocol.protocol_id,
        "replicate_ids": [item.replicate_id for item in ordered],
        "scenario_result_ids": [item.scenario_result_id for item in scenarios],
        "permutation_diagnostic_ids": [
            item.diagnostic_id for item in permutation_diagnostics
        ],
        "producer_marker": _PRODUCER,
    }
    campaign_id = stable_id(
        "g3_frequency_calibration_campaign",
        campaign_payload,
        schema_version="2",
    )
    permutation_values = _immutable_float_array(
        tuple(
            float(value) for item in permutation_diagnostics for value in item.p_values
        ),
        field_name="permutation_null_p_values",
    )
    evidence = object.__new__(G3FrequencyCalibrationEvidence)
    for name, value in {
        "source_artifact_id": campaign_id,
        "protocol_id": protocol.protocol_id,
        "hypothesis_universe_id": protocol.hypothesis_universe_id,
        "hierarchical_procedure_id": protocol.hierarchical_procedure_id,
        "noninferiority_baseline_id": protocol.noninferiority_baseline_id,
        "power_noninferiority_margin": _POWER_NONINFERIORITY_MARGIN,
        "interval_width_noninferiority_margin": (
            _INTERVAL_WIDTH_NONINFERIORITY_MARGIN
        ),
        "noninferiority_direction": _NONINFERIORITY_DIRECTION,
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
        "permutation_diagnostics": permutation_diagnostics,
        "permutation_null_p_values": permutation_values,
        "_producer_token": _PRODUCER_TOKEN,
        "_generator_authorization_token": (
            _VERIFIED_GENERATOR_TOKEN
            if generator_attestation is not None
            and generator_attestation.release_approved
            else _UNVERIFIED_GENERATOR_TOKEN
        ),
    }.items():
        object.__setattr__(evidence, name, value)
    object.__setattr__(
        evidence,
        "evidence_id",
        stable_id(
            "g3_frequency_calibration_evidence",
            evidence._identity_payload(),
            schema_version="4",
        ),
    )
    evidence._require_intact()
    campaign = object.__new__(G3FrequencyCalibrationCampaign)
    for name, value in {
        "protocol": protocol,
        "replicates": ordered,
        "scenarios": scenarios,
        "permutation_diagnostics": permutation_diagnostics,
        "campaign_id": campaign_id,
        "evidence": evidence,
        "generator_attestation": generator_attestation,
        "_producer_token": _PRODUCER_TOKEN,
    }.items():
        object.__setattr__(campaign, name, value)
    campaign._require_intact()
    return campaign


def summarize_g3_frequency_calibration_campaign(
    protocol: G3FrequencyCalibrationProtocol,
    replicates: Sequence[G3FrequencyCalibrationReplicate],
) -> G3FrequencyCalibrationCampaign:
    """Summarize caller-supplied G3-F ledgers as diagnostic-only evidence."""

    return _summarize_g3_frequency_calibration_campaign(
        protocol,
        replicates,
        generator_attestation=None,
    )


def summarize_attested_g3_frequency_calibration_campaign(
    protocol: G3FrequencyCalibrationProtocol,
    replicates: Sequence[G3FrequencyCalibrationReplicate],
    *,
    registry: CalibrationReplayRegistry,
    n_jobs: int = 1,
) -> G3FrequencyCalibrationCampaign:
    """Replay every G3-F ledger, then derive evidence bound to that proof."""

    supplied = tuple(replicates)
    attestation = attest_calibration_replay(
        registry,
        campaign_kind=CalibrationCampaignKind.G3_FREQUENCY,
        protocol=protocol,
        ledgers=supplied,
        n_jobs=n_jobs,
    )
    return _summarize_g3_frequency_calibration_campaign(
        protocol,
        supplied,
        generator_attestation=attestation,
    )


@dataclass(frozen=True, slots=True, init=False)
class G3FrequencyCalibrationGate:
    """Producer-owned G3-F release decision from complete campaign evidence."""

    gate_id: str
    evidence_id: str
    source_artifact_id: str
    protocol_id: str
    hypothesis_universe_id: str
    hierarchical_procedure_id: str
    noninferiority_baseline_id: str
    power_noninferiority_margin: float
    interval_width_noninferiority_margin: float
    noninferiority_direction: str
    generator_id: str
    generator_attestation_id: str | None
    generator_verified: bool
    status: G3FrequencyGateStatus
    reason_code: str | None
    hierarchical_status: G3FrequencyGateStatus
    hierarchical_reason_code: str | None
    permutation_ks_distance: float
    permutation_dkw_limit: float
    minimum_scenario_replicates: int
    type_i_upper_maximum: float | None
    mixed_fdr_upper_maximum: float | None
    primary_fdr_upper_maximum: float | None
    selective_child_fdr_upper_maximum: float | None
    coverage_lower_minimum: float | None
    power_noninferiority_lower_minimum: float | None
    interval_width_noninferiority_lower_minimum: float | None
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "G3FrequencyCalibrationGate is producer-owned; use "
            "build_g3_frequency_calibration_gate()"
        )

    @property
    def formal_release_allowed(self) -> bool:
        return self.status is G3FrequencyGateStatus.PASSED

    @property
    def hierarchical_q_release_allowed(self) -> bool:
        return (
            self.formal_release_allowed
            and self.hierarchical_status is G3FrequencyGateStatus.PASSED
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "source_artifact_id": self.source_artifact_id,
            "protocol_id": self.protocol_id,
            "hypothesis_universe_id": self.hypothesis_universe_id,
            "hierarchical_procedure_id": self.hierarchical_procedure_id,
            "noninferiority_baseline_id": self.noninferiority_baseline_id,
            "power_noninferiority_margin": self.power_noninferiority_margin,
            "interval_width_noninferiority_margin": (
                self.interval_width_noninferiority_margin
            ),
            "noninferiority_direction": self.noninferiority_direction,
            "generator_id": self.generator_id,
            "generator_attestation_id": self.generator_attestation_id,
            "generator_verified": self.generator_verified,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "hierarchical_status": self.hierarchical_status.value,
            "hierarchical_reason_code": self.hierarchical_reason_code,
            "permutation_ks_distance": self.permutation_ks_distance,
            "permutation_dkw_limit": self.permutation_dkw_limit,
            "minimum_scenario_replicates": self.minimum_scenario_replicates,
            "type_i_upper_maximum": self.type_i_upper_maximum,
            "mixed_fdr_upper_maximum": self.mixed_fdr_upper_maximum,
            "primary_fdr_upper_maximum": self.primary_fdr_upper_maximum,
            "selective_child_fdr_upper_maximum": (
                self.selective_child_fdr_upper_maximum
            ),
            "coverage_lower_minimum": self.coverage_lower_minimum,
            "power_noninferiority_lower_minimum": (
                self.power_noninferiority_lower_minimum
            ),
            "interval_width_noninferiority_lower_minimum": (
                self.interval_width_noninferiority_lower_minimum
            ),
            "producer_marker": _GATE_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            expected = stable_id(
                "g3_frequency_calibration_gate",
                self._identity_payload(),
                schema_version="4",
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
                "G3-F gate failed integrity validation",
                code="g3_frequency_gate_integrity_violation",
                field="gate_id",
                remediation="Rebuild the gate from intact campaign evidence",
            ) from error
        if not valid:
            raise _contract_error(
                "G3-F gate failed integrity validation",
                code="g3_frequency_gate_integrity_violation",
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
            "hierarchical_q_release_allowed": self.hierarchical_q_release_allowed,
            "thresholds": {
                "minimum_replicates": _MIN_CALIBRATION_REPLICATES,
                "required_dependence_structures": list(_REQUIRED_DEPENDENCE_STRUCTURES),
                "required_prevalences": list(_REQUIRED_PREVALENCES),
                "monte_carlo_bootstrap_resamples": (_MONTE_CARLO_BOOTSTRAP_RESAMPLES),
                "type_i_upper_limit": _TYPE_I_UPPER_LIMIT,
                "mixed_fdr_upper_limit": _MIXED_FDR_UPPER_LIMIT,
                "primary_fdr_upper_limit": _PRIMARY_FDR_UPPER_LIMIT,
                "selective_child_fdr_upper_limit": (_SELECTIVE_CHILD_FDR_UPPER_LIMIT),
                "coverage_lower_limit": _COVERAGE_LOWER_LIMIT,
                "power_noninferiority_estimand": _POWER_NONINFERIORITY_ESTIMAND,
                "power_noninferiority_direction": _NONINFERIORITY_DIRECTION,
                "power_noninferiority_margin": _POWER_NONINFERIORITY_MARGIN,
                "interval_width_noninferiority_estimand": (
                    _INTERVAL_WIDTH_NONINFERIORITY_ESTIMAND
                ),
                "interval_width_noninferiority_direction": (
                    _NONINFERIORITY_DIRECTION
                ),
                "interval_width_noninferiority_margin": (
                    _INTERVAL_WIDTH_NONINFERIORITY_MARGIN
                ),
            },
        }


def build_g3_frequency_calibration_gate(
    evidence: G3FrequencyCalibrationEvidence,
) -> G3FrequencyCalibrationGate:
    """Derive PASS/FAIL/NE from producer-owned complete-grid evidence."""

    if not isinstance(evidence, G3FrequencyCalibrationEvidence):
        raise TypeError("evidence must be producer-owned G3-F evidence")
    evidence._require_intact()
    expected = {
        (dependence.value, prevalence, metric)
        for dependence in G3FrequencyDependenceStructure
        for prevalence in (None, *_REQUIRED_PREVALENCES)
        for metric in _required_metrics(prevalence)
    }
    observed_grid = {
        (item.dependence_structure, item.non_null_prevalence, item.metric)
        for item in evidence.scenarios
    }
    noninferiority_contract_complete = (
        evidence.power_noninferiority_margin == _POWER_NONINFERIORITY_MARGIN
        and evidence.interval_width_noninferiority_margin
        == _INTERVAL_WIDTH_NONINFERIORITY_MARGIN
        and evidence.noninferiority_direction == _NONINFERIORITY_DIRECTION
        and all(
            (
                item.noninferiority_baseline_id,
                item.noninferiority_estimand,
                item.noninferiority_direction,
                item.noninferiority_margin,
            )
            == (
                (
                    evidence.noninferiority_baseline_id,
                    _POWER_NONINFERIORITY_ESTIMAND,
                    _NONINFERIORITY_DIRECTION,
                    _POWER_NONINFERIORITY_MARGIN,
                )
                if item.metric is G3CalibrationMetric.POWER_NONINFERIORITY_LOWER
                else (
                    evidence.noninferiority_baseline_id,
                    _INTERVAL_WIDTH_NONINFERIORITY_ESTIMAND,
                    _NONINFERIORITY_DIRECTION,
                    _INTERVAL_WIDTH_NONINFERIORITY_MARGIN,
                )
                if item.metric
                is G3CalibrationMetric.INTERVAL_WIDTH_NONINFERIORITY_LOWER
                else (None, None, None, None)
            )
            for item in evidence.scenarios
        )
    )
    scenarios_observed = all(
        item.status is G3CalibrationScenarioStatus.OBSERVED
        for item in evidence.scenarios
    )
    diagnostics_complete = {
        item.dependence_structure for item in evidence.permutation_diagnostics
    } == set(_REQUIRED_DEPENDENCE_STRUCTURES) and all(
        item.status is G3CalibrationScenarioStatus.OBSERVED
        for item in evidence.permutation_diagnostics
    )
    minimum_replicates = min(
        (
            *(item.n_replicates for item in evidence.scenarios),
            *(item.n_replicates for item in evidence.permutation_diagnostics),
        ),
        default=0,
    )
    bounds_by_metric = {
        metric: tuple(
            item.one_sided_bound
            for item in evidence.scenarios
            if item.metric is metric and item.one_sided_bound is not None
        )
        for metric in G3CalibrationMetric
    }
    distances = tuple(
        item.ks_distance
        for item in evidence.permutation_diagnostics
        if item.ks_distance is not None
    )
    limits = tuple(
        item.dkw_limit
        for item in evidence.permutation_diagnostics
        if item.dkw_limit is not None
    )
    if observed_grid != expected or len(evidence.scenarios) != len(expected):
        status = G3FrequencyGateStatus.NOT_ESTIMABLE
        reason: str | None = "g3_frequency_calibration_missing_scenarios"
    elif not noninferiority_contract_complete:
        status = G3FrequencyGateStatus.NOT_ESTIMABLE
        reason = "g3_frequency_noninferiority_contract_mismatch"
    elif not scenarios_observed:
        status = G3FrequencyGateStatus.NOT_ESTIMABLE
        reason = "g3_frequency_calibration_scenario_not_estimable"
    elif not diagnostics_complete:
        status = G3FrequencyGateStatus.NOT_ESTIMABLE
        reason = "g3_frequency_permutation_diagnostic_not_estimable"
    elif minimum_replicates < _MIN_CALIBRATION_REPLICATES:
        status = G3FrequencyGateStatus.NOT_ESTIMABLE
        reason = "g3_frequency_scenario_replicates_below_1000"
    elif not evidence.generator_verified:
        status = G3FrequencyGateStatus.NOT_ESTIMABLE
        reason = "g3_frequency_calibration_generator_unverified"
    elif any(
        cast(float, item.ks_distance) > cast(float, item.dkw_limit)
        for item in evidence.permutation_diagnostics
    ):
        status = G3FrequencyGateStatus.FAILED
        reason = "g3_frequency_permutation_uniformity_diagnostic_failed"
    elif max(bounds_by_metric[G3CalibrationMetric.NULL_TYPE_I_UPPER]) > (
        _TYPE_I_UPPER_LIMIT
    ):
        status = G3FrequencyGateStatus.FAILED
        reason = "g3_frequency_type_i_upper_exceeds_0_06"
    elif max(bounds_by_metric[G3CalibrationMetric.MIXED_FDR_UPPER]) > (
        _MIXED_FDR_UPPER_LIMIT
    ):
        status = G3FrequencyGateStatus.FAILED
        reason = "g3_frequency_mixed_fdr_upper_exceeds_0_07"
    elif max(bounds_by_metric[G3CalibrationMetric.PRIMARY_FDR_UPPER]) > (
        _PRIMARY_FDR_UPPER_LIMIT
    ):
        status = G3FrequencyGateStatus.FAILED
        reason = "g3_primary_fdr_upper_exceeds_0_07"
    elif (
        max(bounds_by_metric[G3CalibrationMetric.SELECTIVE_CHILD_FDR_UPPER])
        > _SELECTIVE_CHILD_FDR_UPPER_LIMIT
    ):
        status = G3FrequencyGateStatus.FAILED
        reason = "g3_selective_child_fdr_upper_exceeds_0_07"
    elif min(bounds_by_metric[G3CalibrationMetric.COVERAGE_LOWER]) < (
        _COVERAGE_LOWER_LIMIT
    ):
        status = G3FrequencyGateStatus.FAILED
        reason = "g3_frequency_coverage_lower_below_0_92"
    elif min(
        bounds_by_metric[G3CalibrationMetric.POWER_NONINFERIORITY_LOWER]
    ) < _POWER_NONINFERIORITY_MARGIN:
        status = G3FrequencyGateStatus.FAILED
        reason = "g3_frequency_power_noninferiority_lower_below_margin"
    elif min(
        bounds_by_metric[
            G3CalibrationMetric.INTERVAL_WIDTH_NONINFERIORITY_LOWER
        ]
    ) < _INTERVAL_WIDTH_NONINFERIORITY_MARGIN:
        status = G3FrequencyGateStatus.FAILED
        reason = "g3_frequency_interval_width_noninferiority_lower_below_margin"
    else:
        status = G3FrequencyGateStatus.PASSED
        reason = None
    self = object.__new__(G3FrequencyCalibrationGate)
    values: dict[str, object] = {
        "evidence_id": evidence.evidence_id,
        "source_artifact_id": evidence.source_artifact_id,
        "protocol_id": evidence.protocol_id,
        "hypothesis_universe_id": evidence.hypothesis_universe_id,
        "hierarchical_procedure_id": evidence.hierarchical_procedure_id,
        "noninferiority_baseline_id": evidence.noninferiority_baseline_id,
        "power_noninferiority_margin": evidence.power_noninferiority_margin,
        "interval_width_noninferiority_margin": (
            evidence.interval_width_noninferiority_margin
        ),
        "noninferiority_direction": evidence.noninferiority_direction,
        "generator_id": evidence.generator_id,
        "generator_attestation_id": evidence.generator_attestation_id,
        "generator_verified": evidence.generator_verified,
        "status": status,
        "reason_code": reason,
        "hierarchical_status": status,
        "hierarchical_reason_code": reason,
        # KS distance is bounded by one.  Using that finite worst case keeps
        # empty/fully unavailable campaigns content-addressable while their
        # status remains NOT_ESTIMABLE.
        "permutation_ks_distance": max(distances, default=1.0),
        "permutation_dkw_limit": min(limits, default=0.0),
        "minimum_scenario_replicates": minimum_replicates,
        "type_i_upper_maximum": max(
            bounds_by_metric[G3CalibrationMetric.NULL_TYPE_I_UPPER], default=None
        ),
        "mixed_fdr_upper_maximum": max(
            bounds_by_metric[G3CalibrationMetric.MIXED_FDR_UPPER], default=None
        ),
        "primary_fdr_upper_maximum": max(
            bounds_by_metric[G3CalibrationMetric.PRIMARY_FDR_UPPER], default=None
        ),
        "selective_child_fdr_upper_maximum": max(
            bounds_by_metric[G3CalibrationMetric.SELECTIVE_CHILD_FDR_UPPER],
            default=None,
        ),
        "coverage_lower_minimum": min(
            bounds_by_metric[G3CalibrationMetric.COVERAGE_LOWER], default=None
        ),
        "power_noninferiority_lower_minimum": min(
            bounds_by_metric[G3CalibrationMetric.POWER_NONINFERIORITY_LOWER],
            default=None,
        ),
        "interval_width_noninferiority_lower_minimum": min(
            bounds_by_metric[
                G3CalibrationMetric.INTERVAL_WIDTH_NONINFERIORITY_LOWER
            ],
            default=None,
        ),
        "_producer_token": _GATE_PRODUCER_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "gate_id",
        stable_id(
            "g3_frequency_calibration_gate",
            self._identity_payload(),
            schema_version="4",
        ),
    )
    self._require_intact()
    return self


__all__ = [
    "G3CalibrationMetric",
    "G3CalibrationScenarioResult",
    "G3CalibrationScenarioStatus",
    "G3FrequencyCalibrationCampaign",
    "G3FrequencyCalibrationEvidence",
    "G3FrequencyCalibrationGate",
    "G3FrequencyCalibrationProtocol",
    "G3FrequencyCalibrationReplicate",
    "G3FrequencyDependenceStructure",
    "G3FrequencyGateStatus",
    "G3FrequencyPermutationDiagnostic",
    "G3FrequencyReplicateStatus",
    "build_g3_frequency_calibration_gate",
    "summarize_attested_g3_frequency_calibration_campaign",
    "summarize_g3_frequency_calibration_campaign",
]
