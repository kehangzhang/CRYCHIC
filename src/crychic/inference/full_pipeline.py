"""Diagnostic full-pipeline effect distributions and the G3-F release gate."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np

from crychic.core import ContractError, stable_id

from .effects import (
    OOFContextEffectResult,
    OOFEffectSpec,
    assess_oof_effect_formal_eligibility,
)
from .g3f_calibration import (
    G3CalibrationMetric,
    G3CalibrationScenarioResult,
    G3FrequencyCalibrationEvidence,
    G3FrequencyCalibrationGate,
    G3FrequencyGateStatus,
    build_g3_frequency_calibration_gate,
)

_SCHEMA_VERSION = "1.0.0"
_MIN_RUNTIME_RESAMPLES = 1_000
_Q_STATUS = "available_only_from_complete_hierarchical_collection_g3_gated"
_FORMAL_UNRELEASED = "g3_frequency_calibration_not_passed"
_FORMAL_RELEASED = "g3_frequency_calibrated_se_ci_p_released_q_collection_only"


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if result == 0.0 else result


def _unit_interval(value: object, *, field_name: str) -> float:
    result = _finite(value, field_name=field_name)
    if not 0 <= result <= 1:
        raise ValueError(f"{field_name} must lie in [0, 1]")
    return result


def _optional_finite(value: object | None, *, field_name: str) -> float | None:
    return None if value is None else _finite(value, field_name=field_name)


def _immutable_vector(values: np.ndarray) -> np.ndarray:
    canonical = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    canonical[canonical == 0.0] = 0.0
    result = cast(
        np.ndarray,
        np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(
            canonical.shape
        ),
    )
    result.setflags(write=False)
    return result


class FullPipelineEffectResamplingKind(StrEnum):
    """Neutral operation names shared without importing workflow."""

    SUBJECT_BOOTSTRAP = "subject_bootstrap"
    CONTEXT_PERMUTATION = "context_permutation"


class FullPipelineResampleEffectStatus(StrEnum):
    """Availability of one resampled unpenalized context effect."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


class SpecificityDirection(StrEnum):
    """Pre-registered minimum-effect exceedance direction."""

    GREATER = "greater"
    LESS = "less"
    TWO_SIDED = "two_sided"


@dataclass(frozen=True, slots=True, kw_only=True)
class FullPipelineResampleEffectRecord:
    """Workflow-neutral effect value with exact full-pipeline provenance."""

    full_pipeline_record_id: str
    plan_id: str
    crossfit_id: str | None
    resampling_kind: FullPipelineEffectResamplingKind
    resample_index: int
    effect_spec_id: str
    hypothesis_id: str
    effect_result_id: str | None
    effect: float | None
    status: FullPipelineResampleEffectStatus
    reason_code: str | None
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = {
            name: _name(cast(str, getattr(self, name)), field_name=name)
            for name in (
                "full_pipeline_record_id",
                "plan_id",
                "effect_spec_id",
                "hypothesis_id",
            )
        }
        kind = FullPipelineEffectResamplingKind(self.resampling_kind)
        status = FullPipelineResampleEffectStatus(self.status)
        crossfit_id: str | None
        effect_result_id: str | None
        effect: float | None
        reason_code: str | None
        if (
            isinstance(self.resample_index, bool)
            or not isinstance(self.resample_index, int)
            or self.resample_index < 0
        ):
            raise ValueError("resample_index must be a non-negative integer")
        if status is FullPipelineResampleEffectStatus.OBSERVED:
            crossfit_id = _name(cast(str, self.crossfit_id), field_name="crossfit_id")
            effect_result_id = _name(
                cast(str, self.effect_result_id), field_name="effect_result_id"
            )
            effect = _finite(self.effect, field_name="effect")
            if self.reason_code is not None:
                raise ValueError("observed resample effect cannot have a reason")
            reason_code = None
        elif status is FullPipelineResampleEffectStatus.NOT_ESTIMABLE:
            crossfit_id = _name(cast(str, self.crossfit_id), field_name="crossfit_id")
            effect_result_id = (
                None
                if self.effect_result_id is None
                else _name(self.effect_result_id, field_name="effect_result_id")
            )
            if self.effect is not None:
                raise ValueError("not-estimable resample effect cannot have a value")
            effect = None
            reason_code = _name(cast(str, self.reason_code), field_name="reason_code")
        else:
            crossfit_id = (
                None
                if self.crossfit_id is None
                else _name(self.crossfit_id, field_name="crossfit_id")
            )
            effect_result_id = (
                None
                if self.effect_result_id is None
                else _name(self.effect_result_id, field_name="effect_result_id")
            )
            if self.effect is not None:
                raise ValueError("failed resample effect cannot have a value")
            effect = None
            reason_code = _name(cast(str, self.reason_code), field_name="reason_code")
        payload = {
            **identifiers,
            "crossfit_id": crossfit_id,
            "resampling_kind": kind.value,
            "resample_index": self.resample_index,
            "effect_result_id": effect_result_id,
            "effect": effect,
            "status": status.value,
            "reason_code": reason_code,
        }
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "crossfit_id", crossfit_id)
        object.__setattr__(self, "effect_result_id", effect_result_id)
        object.__setattr__(self, "resampling_kind", kind)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "full_pipeline_resample_effect",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "full_pipeline_record_id": self.full_pipeline_record_id,
            "plan_id": self.plan_id,
            "crossfit_id": self.crossfit_id,
            "resampling_kind": self.resampling_kind.value,
            "resample_index": self.resample_index,
            "effect_spec_id": self.effect_spec_id,
            "hypothesis_id": self.hypothesis_id,
            "effect_result_id": self.effect_result_id,
            "effect": self.effect,
            "status": self.status.value,
            "reason_code": self.reason_code,
        }

    def _require_intact(self) -> None:
        try:
            repeated = FullPipelineResampleEffectRecord(
                full_pipeline_record_id=self.full_pipeline_record_id,
                plan_id=self.plan_id,
                crossfit_id=self.crossfit_id,
                resampling_kind=self.resampling_kind,
                resample_index=self.resample_index,
                effect_spec_id=self.effect_spec_id,
                hypothesis_id=self.hypothesis_id,
                effect_result_id=self.effect_result_id,
                effect=self.effect,
                status=self.status,
                reason_code=self.reason_code,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.record_id == repeated.record_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Full-pipeline effect record failed integrity validation",
                code="full_pipeline_effect_record_integrity_violation",
                field="record_id",
                remediation="Rebuild the record from the exact resample child",
            ) from error
        if not valid:
            raise ContractError(
                "Full-pipeline effect record failed integrity validation",
                code="full_pipeline_effect_record_integrity_violation",
                field="record_id",
                remediation="Rebuild the record from the exact resample child",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "record_id": self.record_id,
            "full_pipeline_record_id": self.full_pipeline_record_id,
            "plan_id": self.plan_id,
            "crossfit_id": self.crossfit_id,
            "resampling_kind": self.resampling_kind.value,
            "resample_index": self.resample_index,
            "effect_spec_id": self.effect_spec_id,
            "hypothesis_id": self.hypothesis_id,
            "effect_result_id": self.effect_result_id,
            "effect": self.effect,
            "status": self.status.value,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FullPipelineEffectDistributionSpec:
    """Pre-register minimum-effect frequency before viewing resamples."""

    effect_spec: OOFEffectSpec
    minimum_effect: float
    specificity_direction: SpecificityDirection
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.effect_spec, OOFEffectSpec):
            raise TypeError("effect_spec must be an OOFEffectSpec")
        self.effect_spec._require_intact()
        minimum = _finite(self.minimum_effect, field_name="minimum_effect")
        if minimum < 0:
            raise ValueError("minimum_effect must be non-negative")
        direction = SpecificityDirection(self.specificity_direction)
        payload = {
            "effect_spec_id": self.effect_spec.spec_id,
            "minimum_effect": minimum,
            "specificity_direction": direction.value,
            "specificity_semantics": (
                "frequentist_full_pipeline_bootstrap_exceedance_frequency_v1"
            ),
        }
        object.__setattr__(self, "minimum_effect", minimum)
        object.__setattr__(self, "specificity_direction", direction)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "full_pipeline_effect_distribution_spec",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "effect_spec_id": self.effect_spec.spec_id,
            "minimum_effect": self.minimum_effect,
            "specificity_direction": self.specificity_direction.value,
            "specificity_semantics": (
                "frequentist_full_pipeline_bootstrap_exceedance_frequency_v1"
            ),
        }

    def _require_intact(self) -> None:
        try:
            self.effect_spec._require_intact()
            repeated = FullPipelineEffectDistributionSpec(
                effect_spec=self.effect_spec,
                minimum_effect=self.minimum_effect,
                specificity_direction=self.specificity_direction,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.spec_id == repeated.spec_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Full-pipeline effect distribution spec failed integrity validation",
                code="full_pipeline_effect_distribution_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the distribution specification",
            ) from error
        if not valid:
            raise ContractError(
                "Full-pipeline effect distribution spec failed integrity validation",
                code="full_pipeline_effect_distribution_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the distribution specification",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "effect_spec": self.effect_spec.to_dict(),
            "minimum_effect": self.minimum_effect,
            "specificity_direction": self.specificity_direction.value,
            "specificity_semantics": (
                "frequentist_full_pipeline_bootstrap_exceedance_frequency_v1"
            ),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FullPipelineEffectDistribution:
    """Diagnostic resample distribution with a strict formal-release boundary."""

    distribution_spec_id: str
    effect_spec_id: str
    hypothesis_id: str
    point_effect_result_id: str
    point_effect: float | None
    point_n_clusters: int
    minimum_point_clusters: int
    point_uncertainty_status: str
    resample_record_ids: tuple[str, ...]
    bootstrap_effects: np.ndarray
    permutation_effects: np.ndarray
    n_bootstrap_total: int
    n_bootstrap_observed: int
    n_permutation_total: int
    n_permutation_observed: int
    diagnostic_bootstrap_standard_error: float | None
    diagnostic_ci_lower: float | None
    diagnostic_ci_upper: float | None
    diagnostic_specificity_frequency: float | None
    diagnostic_empirical_permutation_p: float | None
    minimum_effect: float
    specificity_direction: SpecificityDirection
    calibration_gate_id: str | None
    standard_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    q_value: None
    formal_inference_status: str
    formal_reason_code: str | None
    distribution_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "distribution_spec_id",
            "effect_spec_id",
            "hypothesis_id",
            "point_effect_result_id",
        ):
            object.__setattr__(
                self,
                name,
                _name(cast(str, getattr(self, name)), field_name=name),
            )
        records = tuple(self.resample_record_ids)
        if len(records) != len(set(records)) or any(not value for value in records):
            raise ValueError("resample_record_ids must be unique and non-empty")
        bootstrap = _immutable_vector(self.bootstrap_effects)
        permutation = _immutable_vector(self.permutation_effects)
        if (
            bootstrap.ndim != 1
            or permutation.ndim != 1
            or np.any(~np.isfinite(bootstrap))
            or np.any(~np.isfinite(permutation))
        ):
            raise ValueError("resample effect distributions must be finite vectors")
        direction = SpecificityDirection(self.specificity_direction)
        point_n_clusters = self.point_n_clusters
        minimum_point_clusters = self.minimum_point_clusters
        if (
            isinstance(point_n_clusters, bool)
            or not isinstance(point_n_clusters, int)
            or point_n_clusters < 0
        ):
            raise ValueError("point_n_clusters must be a non-negative integer")
        if (
            isinstance(minimum_point_clusters, bool)
            or not isinstance(minimum_point_clusters, int)
            or minimum_point_clusters < 4
        ):
            raise ValueError("minimum_point_clusters must be an integer >= 4")
        point_uncertainty_status = _name(
            self.point_uncertainty_status,
            field_name="point_uncertainty_status",
        )
        counts: dict[str, int] = {}
        for name in (
            "n_bootstrap_total",
            "n_bootstrap_observed",
            "n_permutation_total",
            "n_permutation_observed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            counts[name] = value
        if (
            counts["n_bootstrap_observed"] > counts["n_bootstrap_total"]
            or counts["n_permutation_observed"] > counts["n_permutation_total"]
            or len(bootstrap) != counts["n_bootstrap_observed"]
            or len(permutation) != counts["n_permutation_observed"]
            or len(records)
            != counts["n_bootstrap_total"] + counts["n_permutation_total"]
        ):
            raise ValueError("resample counts do not match records and effect arrays")
        point_effect = _optional_finite(self.point_effect, field_name="point_effect")
        minimum_effect = _finite(self.minimum_effect, field_name="minimum_effect")
        if minimum_effect < 0:
            raise ValueError("minimum_effect must be non-negative")
        diagnostic_se = _optional_finite(
            self.diagnostic_bootstrap_standard_error,
            field_name="diagnostic_bootstrap_standard_error",
        )
        diagnostic_lower = _optional_finite(
            self.diagnostic_ci_lower, field_name="diagnostic_ci_lower"
        )
        diagnostic_upper = _optional_finite(
            self.diagnostic_ci_upper, field_name="diagnostic_ci_upper"
        )
        diagnostic_specificity = (
            None
            if self.diagnostic_specificity_frequency is None
            else _unit_interval(
                self.diagnostic_specificity_frequency,
                field_name="diagnostic_specificity_frequency",
            )
        )
        diagnostic_p = (
            None
            if self.diagnostic_empirical_permutation_p is None
            else _unit_interval(
                self.diagnostic_empirical_permutation_p,
                field_name="diagnostic_empirical_permutation_p",
            )
        )
        if (diagnostic_lower is None) != (diagnostic_upper is None) or (
            diagnostic_lower is not None
            and diagnostic_upper is not None
            and diagnostic_lower > diagnostic_upper
        ):
            raise ValueError("diagnostic confidence interval is invalid")
        if counts["n_bootstrap_observed"] >= 2 and any(
            value is None
            for value in (diagnostic_se, diagnostic_lower, diagnostic_upper)
        ):
            raise ValueError(
                "bootstrap diagnostics are missing despite sufficient values"
            )
        if counts["n_bootstrap_observed"] and diagnostic_specificity is None:
            raise ValueError(
                "specificity frequency is missing despite bootstrap values"
            )
        if (
            counts["n_permutation_observed"]
            and point_effect is not None
            and diagnostic_p is None
        ):
            raise ValueError(
                "empirical p is missing despite point and permutation values"
            )
        standard_error = _optional_finite(
            self.standard_error, field_name="standard_error"
        )
        ci_lower = _optional_finite(self.ci_lower, field_name="ci_lower")
        ci_upper = _optional_finite(self.ci_upper, field_name="ci_upper")
        p_value = (
            None
            if self.p_value is None
            else _unit_interval(self.p_value, field_name="p_value")
        )
        if self.q_value is not None:
            raise ValueError(
                "q_value is available only from a complete hierarchical collection"
            )
        if self.formal_inference_status == _FORMAL_RELEASED:
            if (
                self.calibration_gate_id is None
                or self.formal_reason_code is not None
                or point_effect is None
                or point_n_clusters < minimum_point_clusters
                or point_uncertainty_status != "observed_cr2_diagnostic"
                or counts["n_bootstrap_total"] < _MIN_RUNTIME_RESAMPLES
                or counts["n_permutation_total"] < _MIN_RUNTIME_RESAMPLES
                or counts["n_bootstrap_observed"] != counts["n_bootstrap_total"]
                or counts["n_permutation_observed"] != counts["n_permutation_total"]
                or any(
                    value is None
                    for value in (standard_error, ci_lower, ci_upper, p_value)
                )
                or (standard_error, ci_lower, ci_upper, p_value)
                != (diagnostic_se, diagnostic_lower, diagnostic_upper, diagnostic_p)
            ):
                raise ValueError(
                    "released formal fields are incomplete or inconsistent"
                )
        elif self.formal_inference_status == _FORMAL_UNRELEASED:
            if (
                any(
                    value is not None
                    for value in (standard_error, ci_lower, ci_upper, p_value)
                )
                or not self.formal_reason_code
            ):
                raise ValueError("unreleased result cannot contain formal fields")
        else:
            raise ValueError("formal_inference_status is unsupported")
        payload = {
            "distribution_spec_id": self.distribution_spec_id,
            "effect_spec_id": self.effect_spec_id,
            "hypothesis_id": self.hypothesis_id,
            "point_effect_result_id": self.point_effect_result_id,
            "point_effect": point_effect,
            "point_n_clusters": point_n_clusters,
            "minimum_point_clusters": minimum_point_clusters,
            "point_uncertainty_status": point_uncertainty_status,
            "resample_record_ids": list(records),
            "bootstrap_effects": bootstrap.tolist(),
            "permutation_effects": permutation.tolist(),
            "diagnostic_bootstrap_standard_error": (diagnostic_se),
            "diagnostic_ci": [diagnostic_lower, diagnostic_upper],
            "diagnostic_specificity_frequency": diagnostic_specificity,
            "diagnostic_empirical_permutation_p": diagnostic_p,
            "minimum_effect": minimum_effect,
            "specificity_direction": direction.value,
            "calibration_gate_id": self.calibration_gate_id,
            "standard_error": standard_error,
            "ci": [ci_lower, ci_upper],
            "p_value": p_value,
            "q_value": None,
            "formal_inference_status": self.formal_inference_status,
            "formal_reason_code": self.formal_reason_code,
        }
        object.__setattr__(self, "resample_record_ids", records)
        object.__setattr__(self, "bootstrap_effects", bootstrap)
        object.__setattr__(self, "permutation_effects", permutation)
        object.__setattr__(self, "specificity_direction", direction)
        object.__setattr__(self, "point_effect", point_effect)
        object.__setattr__(self, "point_uncertainty_status", point_uncertainty_status)
        object.__setattr__(self, "minimum_effect", minimum_effect)
        object.__setattr__(self, "diagnostic_bootstrap_standard_error", diagnostic_se)
        object.__setattr__(self, "diagnostic_ci_lower", diagnostic_lower)
        object.__setattr__(self, "diagnostic_ci_upper", diagnostic_upper)
        object.__setattr__(
            self, "diagnostic_specificity_frequency", diagnostic_specificity
        )
        object.__setattr__(self, "diagnostic_empirical_permutation_p", diagnostic_p)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "ci_lower", ci_lower)
        object.__setattr__(self, "ci_upper", ci_upper)
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(
            self,
            "distribution_id",
            stable_id(
                "full_pipeline_effect_distribution",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "distribution_spec_id": self.distribution_spec_id,
            "effect_spec_id": self.effect_spec_id,
            "hypothesis_id": self.hypothesis_id,
            "point_effect_result_id": self.point_effect_result_id,
            "point_effect": self.point_effect,
            "point_n_clusters": self.point_n_clusters,
            "minimum_point_clusters": self.minimum_point_clusters,
            "point_uncertainty_status": self.point_uncertainty_status,
            "resample_record_ids": list(self.resample_record_ids),
            "bootstrap_effects": self.bootstrap_effects.tolist(),
            "permutation_effects": self.permutation_effects.tolist(),
            "diagnostic_bootstrap_standard_error": (
                self.diagnostic_bootstrap_standard_error
            ),
            "diagnostic_ci": [self.diagnostic_ci_lower, self.diagnostic_ci_upper],
            "diagnostic_specificity_frequency": (self.diagnostic_specificity_frequency),
            "diagnostic_empirical_permutation_p": (
                self.diagnostic_empirical_permutation_p
            ),
            "minimum_effect": self.minimum_effect,
            "specificity_direction": self.specificity_direction.value,
            "calibration_gate_id": self.calibration_gate_id,
            "standard_error": self.standard_error,
            "ci": [self.ci_lower, self.ci_upper],
            "p_value": self.p_value,
            "q_value": None,
            "formal_inference_status": self.formal_inference_status,
            "formal_reason_code": self.formal_reason_code,
        }

    def _require_intact(self) -> None:
        try:
            repeated = FullPipelineEffectDistribution(
                distribution_spec_id=self.distribution_spec_id,
                effect_spec_id=self.effect_spec_id,
                hypothesis_id=self.hypothesis_id,
                point_effect_result_id=self.point_effect_result_id,
                point_effect=self.point_effect,
                point_n_clusters=self.point_n_clusters,
                minimum_point_clusters=self.minimum_point_clusters,
                point_uncertainty_status=self.point_uncertainty_status,
                resample_record_ids=self.resample_record_ids,
                bootstrap_effects=self.bootstrap_effects,
                permutation_effects=self.permutation_effects,
                n_bootstrap_total=self.n_bootstrap_total,
                n_bootstrap_observed=self.n_bootstrap_observed,
                n_permutation_total=self.n_permutation_total,
                n_permutation_observed=self.n_permutation_observed,
                diagnostic_bootstrap_standard_error=(
                    self.diagnostic_bootstrap_standard_error
                ),
                diagnostic_ci_lower=self.diagnostic_ci_lower,
                diagnostic_ci_upper=self.diagnostic_ci_upper,
                diagnostic_specificity_frequency=(
                    self.diagnostic_specificity_frequency
                ),
                diagnostic_empirical_permutation_p=(
                    self.diagnostic_empirical_permutation_p
                ),
                minimum_effect=self.minimum_effect,
                specificity_direction=self.specificity_direction,
                calibration_gate_id=self.calibration_gate_id,
                standard_error=self.standard_error,
                ci_lower=self.ci_lower,
                ci_upper=self.ci_upper,
                p_value=self.p_value,
                q_value=None,
                formal_inference_status=self.formal_inference_status,
                formal_reason_code=self.formal_reason_code,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.distribution_id == repeated.distribution_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Full-pipeline effect distribution failed integrity validation",
                code="full_pipeline_effect_distribution_integrity_violation",
                field="distribution_id",
                remediation="Re-summarize the exact complete resample records",
            ) from error
        if not valid:
            raise ContractError(
                "Full-pipeline effect distribution failed integrity validation",
                code="full_pipeline_effect_distribution_integrity_violation",
                field="distribution_id",
                remediation="Re-summarize the exact complete resample records",
            )

    @property
    def formal_inference_allowed(self) -> bool:
        return self.formal_inference_status == _FORMAL_RELEASED

    @property
    def q_value_status(self) -> str:
        return _Q_STATUS

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "distribution_id": self.distribution_id,
            "distribution_spec_id": self.distribution_spec_id,
            "effect_spec_id": self.effect_spec_id,
            "hypothesis_id": self.hypothesis_id,
            "point_effect_result_id": self.point_effect_result_id,
            "point_effect": self.point_effect,
            "point_n_clusters": self.point_n_clusters,
            "minimum_point_clusters": self.minimum_point_clusters,
            "point_uncertainty_status": self.point_uncertainty_status,
            "resample_record_ids": list(self.resample_record_ids),
            "n_bootstrap_total": self.n_bootstrap_total,
            "n_bootstrap_observed": self.n_bootstrap_observed,
            "n_permutation_total": self.n_permutation_total,
            "n_permutation_observed": self.n_permutation_observed,
            "diagnostic_bootstrap_standard_error": (
                self.diagnostic_bootstrap_standard_error
            ),
            "diagnostic_ci_lower": self.diagnostic_ci_lower,
            "diagnostic_ci_upper": self.diagnostic_ci_upper,
            "diagnostic_specificity_frequency": (self.diagnostic_specificity_frequency),
            "diagnostic_empirical_permutation_p": (
                self.diagnostic_empirical_permutation_p
            ),
            "minimum_effect": self.minimum_effect,
            "specificity_direction": self.specificity_direction.value,
            "specificity_semantics": (
                "frequentist_full_pipeline_bootstrap_exceedance_frequency_v1"
            ),
            "calibration_gate_id": self.calibration_gate_id,
            "standard_error": self.standard_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "q_value": None,
            "q_value_status": _Q_STATUS,
            "formal_inference_status": self.formal_inference_status,
            "formal_reason_code": self.formal_reason_code,
            "formal_inference_allowed": self.formal_inference_allowed,
        }


def _specificity_frequency(
    effects: np.ndarray,
    *,
    minimum_effect: float,
    direction: SpecificityDirection,
) -> float | None:
    if not len(effects):
        return None
    if direction is SpecificityDirection.GREATER:
        selected = effects > minimum_effect
    elif direction is SpecificityDirection.LESS:
        selected = effects < -minimum_effect
    else:
        selected = np.abs(effects) > minimum_effect
    return float(np.mean(selected))


def _empirical_p(
    point_effect: float | None,
    effects: np.ndarray,
    *,
    direction: SpecificityDirection,
) -> float | None:
    if point_effect is None or not len(effects):
        return None
    tolerance = 1e-12 * max(1.0, abs(point_effect))
    if direction is SpecificityDirection.GREATER:
        extreme = effects >= point_effect - tolerance
    elif direction is SpecificityDirection.LESS:
        extreme = effects <= point_effect + tolerance
    else:
        extreme = np.abs(effects) >= abs(point_effect) - tolerance
    return float((1 + int(np.count_nonzero(extreme))) / (len(effects) + 1))


def summarize_full_pipeline_effect_distribution(
    point_effect: OOFContextEffectResult,
    spec: FullPipelineEffectDistributionSpec,
    records: tuple[FullPipelineResampleEffectRecord, ...],
    *,
    calibration_gate: G3FrequencyCalibrationGate | None = None,
) -> FullPipelineEffectDistribution:
    """Summarize diagnostic resamples and apply the producer-owned G3-F gate."""

    if not isinstance(point_effect, OOFContextEffectResult):
        raise TypeError("point_effect must be an OOFContextEffectResult")
    if not isinstance(spec, FullPipelineEffectDistributionSpec):
        raise TypeError("spec must be FullPipelineEffectDistributionSpec")
    point_effect._require_intact()
    spec._require_intact()
    if (
        point_effect.spec_id != spec.effect_spec.spec_id
        or point_effect.hypothesis_id != spec.effect_spec.hypothesis_id
    ):
        raise ContractError(
            "Point effect does not match the distribution specification",
            code="full_pipeline_effect_point_spec_mismatch",
            field="spec_id,hypothesis_id",
            remediation="Use the exact OOF effect result bound by the specification",
        )
    supplied = tuple(records)
    if not supplied or any(
        not isinstance(item, FullPipelineResampleEffectRecord) for item in supplied
    ):
        raise ValueError("records must contain typed full-pipeline effect records")
    for item in supplied:
        item._require_intact()
    values = tuple(
        sorted(
            supplied,
            key=lambda item: (
                item.resampling_kind.value,
                item.resample_index,
                item.plan_id,
            ),
        )
    )
    if len({item.record_id for item in values}) != len(values) or len(
        {item.full_pipeline_record_id for item in values}
    ) != len(values):
        raise ContractError(
            "Full-pipeline effect records must have unique provenance",
            code="duplicate_full_pipeline_effect_record",
            field="full_pipeline_record_id",
            remediation="Include every full-pipeline resample exactly once",
        )
    if len({item.plan_id for item in values}) != len(values) or len(
        {(item.resampling_kind, item.resample_index) for item in values}
    ) != len(values):
        raise ContractError(
            "Full-pipeline effect records contain duplicate resampling plans",
            code="duplicate_full_pipeline_effect_plan",
            field="plan_id,resampling_kind,resample_index",
            remediation="Include every pre-registered resampling plan exactly once",
        )
    if any(
        item.effect_spec_id != spec.effect_spec.spec_id
        or item.hypothesis_id != spec.effect_spec.hypothesis_id
        for item in values
    ):
        raise ContractError(
            "Resampled effects do not match the point-effect specification",
            code="full_pipeline_effect_record_spec_mismatch",
            field="effect_spec_id,hypothesis_id",
            remediation="Aggregate only one pre-registered hypothesis",
        )
    bootstrap_records = tuple(
        item
        for item in values
        if item.resampling_kind is FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP
    )
    permutation_records = tuple(
        item
        for item in values
        if item.resampling_kind is FullPipelineEffectResamplingKind.CONTEXT_PERMUTATION
    )
    bootstrap = _immutable_vector(
        np.asarray(
            [
                item.effect
                for item in bootstrap_records
                if item.status is FullPipelineResampleEffectStatus.OBSERVED
            ],
            dtype=float,
        )
    )
    permutation = _immutable_vector(
        np.asarray(
            [
                item.effect
                for item in permutation_records
                if item.status is FullPipelineResampleEffectStatus.OBSERVED
            ],
            dtype=float,
        )
    )
    diagnostic_se = float(np.std(bootstrap, ddof=1)) if len(bootstrap) >= 2 else None
    diagnostic_lower: float | None
    diagnostic_upper: float | None
    if len(bootstrap) >= 2:
        interval = cast(
            np.ndarray,
            np.quantile(bootstrap, [0.025, 0.975], method="linear"),
        )
        diagnostic_lower = float(interval[0])
        diagnostic_upper = float(interval[1])
    else:
        diagnostic_lower = diagnostic_upper = None
    diagnostic_specificity = _specificity_frequency(
        bootstrap,
        minimum_effect=spec.minimum_effect,
        direction=spec.specificity_direction,
    )
    point_value = (
        float(point_effect.effect)
        if point_effect.effect_status == "observed" and point_effect.effect is not None
        else None
    )
    diagnostic_p = _empirical_p(
        point_value,
        permutation,
        direction=spec.specificity_direction,
    )
    if calibration_gate is not None and not isinstance(
        calibration_gate, G3FrequencyCalibrationGate
    ):
        raise TypeError("calibration_gate must be producer-owned G3 gate or None")
    if calibration_gate is not None:
        calibration_gate._require_intact()
    point_eligibility = assess_oof_effect_formal_eligibility(
        point_effect,
        spec.effect_spec,
    )
    formal_reason: str | None
    gate_id: str | None
    standard_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    runtime_reason: str | None = None
    if point_value is None:
        runtime_reason = "full_pipeline_point_effect_not_observed"
    elif not point_eligibility.formal_backend_eligible:
        if point_eligibility.reason_code == (
            "oof_effect_insufficient_subject_clusters_for_formal_cr2"
        ):
            runtime_reason = "full_pipeline_point_clusters_below_minimum"
        elif point_eligibility.reason_code in {
            "oof_effect_cr2_covariance_not_estimable",
            "oof_effect_cr2_uncertainty_not_formal_eligible",
        }:
            runtime_reason = "full_pipeline_point_cr2_uncertainty_not_estimable"
        else:
            runtime_reason = point_eligibility.reason_code
    elif len(bootstrap_records) != len(bootstrap):
        runtime_reason = "full_pipeline_bootstrap_distribution_incomplete"
    elif len(bootstrap) < _MIN_RUNTIME_RESAMPLES:
        runtime_reason = "full_pipeline_bootstrap_observed_below_1000"
    elif len(permutation_records) != len(permutation):
        runtime_reason = "full_pipeline_permutation_distribution_incomplete"
    elif len(permutation) < _MIN_RUNTIME_RESAMPLES:
        runtime_reason = "full_pipeline_permutation_observed_below_1000"
    elif any(
        value is None
        for value in (diagnostic_se, diagnostic_lower, diagnostic_upper, diagnostic_p)
    ):
        runtime_reason = "full_pipeline_runtime_diagnostics_not_estimable"
    if (
        calibration_gate is not None
        and calibration_gate.formal_release_allowed
        and runtime_reason is None
    ):
        formal_status = _FORMAL_RELEASED
        formal_reason = None
        gate_id = calibration_gate.gate_id
        standard_error = diagnostic_se
        ci_lower = diagnostic_lower
        ci_upper = diagnostic_upper
        p_value = diagnostic_p
    else:
        formal_status = _FORMAL_UNRELEASED
        if calibration_gate is None:
            formal_reason = "g3_frequency_calibration_gate_absent"
        elif not calibration_gate.formal_release_allowed:
            formal_reason = calibration_gate.reason_code
        else:
            formal_reason = runtime_reason
        gate_id = None if calibration_gate is None else calibration_gate.gate_id
        standard_error = ci_lower = ci_upper = p_value = None
    return FullPipelineEffectDistribution(
        distribution_spec_id=spec.spec_id,
        effect_spec_id=spec.effect_spec.spec_id,
        hypothesis_id=spec.effect_spec.hypothesis_id,
        point_effect_result_id=point_effect.result_id,
        point_effect=point_value,
        point_n_clusters=point_effect.n_clusters,
        minimum_point_clusters=point_eligibility.minimum_clusters,
        point_uncertainty_status=point_effect.uncertainty_status,
        resample_record_ids=tuple(item.record_id for item in values),
        bootstrap_effects=bootstrap,
        permutation_effects=permutation,
        n_bootstrap_total=len(bootstrap_records),
        n_bootstrap_observed=len(bootstrap),
        n_permutation_total=len(permutation_records),
        n_permutation_observed=len(permutation),
        diagnostic_bootstrap_standard_error=diagnostic_se,
        diagnostic_ci_lower=diagnostic_lower,
        diagnostic_ci_upper=diagnostic_upper,
        diagnostic_specificity_frequency=diagnostic_specificity,
        diagnostic_empirical_permutation_p=diagnostic_p,
        minimum_effect=spec.minimum_effect,
        specificity_direction=spec.specificity_direction,
        calibration_gate_id=gate_id,
        standard_error=standard_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        q_value=None,
        formal_inference_status=formal_status,
        formal_reason_code=formal_reason,
    )


__all__ = [
    "FullPipelineEffectDistribution",
    "FullPipelineEffectDistributionSpec",
    "FullPipelineEffectResamplingKind",
    "FullPipelineResampleEffectRecord",
    "FullPipelineResampleEffectStatus",
    "G3CalibrationMetric",
    "G3CalibrationScenarioResult",
    "G3FrequencyCalibrationEvidence",
    "G3FrequencyCalibrationGate",
    "G3FrequencyGateStatus",
    "SpecificityDirection",
    "build_g3_frequency_calibration_gate",
    "summarize_full_pipeline_effect_distribution",
]
