"""Producer-owned specificity support from complete subject bootstraps."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np

from crychic.core import ContractError, stable_id

from .full_pipeline import (
    FullPipelineEffectDistribution,
    FullPipelineEffectDistributionSpec,
    FullPipelineEffectResamplingKind,
    FullPipelineResampleEffectRecord,
    FullPipelineResampleEffectStatus,
    SpecificityDirection,
    _specificity_frequency,
)

_SCHEMA_VERSION = "1.0.0"
_MINIMUM_BOOTSTRAPS = 1_000
_SEMANTICS = (
    "frequentist_full_pipeline_subject_bootstrap_exceedance_frequency_"
    "not_posterior_v1"
)
_SOURCE_BINDING_STATUS = (
    "unbound_numeric_primitive_requires_frozen_workflow_adapter"
)
_PRODUCER_MARKER = "crychic.inference.specificity_support.v1"


def _canonical_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


class SpecificitySupportStatus(StrEnum):
    """Availability of a complete pre-registered specificity support result."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecificitySupportSpec:
    """Bind specificity support to its estimand, scale, universe, and plans."""

    effect_distribution_spec: FullPipelineEffectDistributionSpec
    hypothesis_universe_id: str
    effect_scale_id: str
    subject_bootstrap_plan_ids: tuple[str, ...]
    minimum_bootstraps: int = _MINIMUM_BOOTSTRAPS
    bootstrap_plan_set_id: str = field(init=False)
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.effect_distribution_spec, FullPipelineEffectDistributionSpec
        ):
            raise TypeError(
                "effect_distribution_spec must be "
                "FullPipelineEffectDistributionSpec"
            )
        self.effect_distribution_spec._require_intact()
        universe_id = _canonical_name(
            self.hypothesis_universe_id,
            field_name="hypothesis_universe_id",
        )
        scale_id = _canonical_name(self.effect_scale_id, field_name="effect_scale_id")
        plans = tuple(
            sorted(
                _canonical_name(value, field_name="subject_bootstrap_plan_ids")
                for value in self.subject_bootstrap_plan_ids
            )
        )
        if not plans:
            raise ValueError("subject_bootstrap_plan_ids cannot be empty")
        if len(plans) != len(set(plans)):
            raise ValueError("subject_bootstrap_plan_ids must be unique")
        if (
            isinstance(self.minimum_bootstraps, bool)
            or not isinstance(self.minimum_bootstraps, int)
            or self.minimum_bootstraps != _MINIMUM_BOOTSTRAPS
        ):
            raise ValueError("minimum_bootstraps is fixed at 1000")
        plan_set_id = stable_id(
            "subject_bootstrap_plan_set",
            {"plan_ids": list(plans)},
            schema_version=_SCHEMA_VERSION,
        )
        object.__setattr__(self, "hypothesis_universe_id", universe_id)
        object.__setattr__(self, "effect_scale_id", scale_id)
        object.__setattr__(self, "subject_bootstrap_plan_ids", plans)
        object.__setattr__(self, "bootstrap_plan_set_id", plan_set_id)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "specificity_support_spec",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "effect_distribution_spec_id": self.effect_distribution_spec.spec_id,
            "hypothesis_universe_id": self.hypothesis_universe_id,
            "effect_scale_id": self.effect_scale_id,
            "bootstrap_plan_set_id": self.bootstrap_plan_set_id,
            "subject_bootstrap_plan_ids": list(self.subject_bootstrap_plan_ids),
            "minimum_bootstraps": self.minimum_bootstraps,
            "specificity_semantics": _SEMANTICS,
            "source_binding_status": _SOURCE_BINDING_STATUS,
            "public_release_allowed": False,
        }

    def _require_intact(self) -> None:
        try:
            self.effect_distribution_spec._require_intact()
            repeated = SpecificitySupportSpec(
                effect_distribution_spec=self.effect_distribution_spec,
                hypothesis_universe_id=self.hypothesis_universe_id,
                effect_scale_id=self.effect_scale_id,
                subject_bootstrap_plan_ids=self.subject_bootstrap_plan_ids,
                minimum_bootstraps=self.minimum_bootstraps,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.bootstrap_plan_set_id == repeated.bootstrap_plan_set_id
                and self.spec_id == repeated.spec_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Specificity support spec failed integrity validation",
                code="specificity_support_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the spec from the frozen bootstrap plans",
            ) from error
        if not valid:
            raise ContractError(
                "Specificity support spec failed integrity validation",
                code="specificity_support_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the spec from the frozen bootstrap plans",
            )

    @property
    def specificity_semantics(self) -> str:
        return _SEMANTICS

    @property
    def source_binding_status(self) -> str:
        return _SOURCE_BINDING_STATUS

    @property
    def public_release_allowed(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "effect_distribution_spec": self.effect_distribution_spec.to_dict(),
            "hypothesis_universe_id": self.hypothesis_universe_id,
            "effect_scale_id": self.effect_scale_id,
            "bootstrap_plan_set_id": self.bootstrap_plan_set_id,
            "subject_bootstrap_plan_ids": list(self.subject_bootstrap_plan_ids),
            "minimum_bootstraps": self.minimum_bootstraps,
            "specificity_semantics": _SEMANTICS,
            "source_binding_status": _SOURCE_BINDING_STATUS,
            "public_release_allowed": False,
            "is_posterior_probability": False,
            "is_comm_probability": False,
        }


@dataclass(frozen=True, slots=True)
class _SpecificityEvaluation:
    records: tuple[FullPipelineResampleEffectRecord, ...]
    n_bootstrap_total: int
    n_bootstrap_observed: int
    n_bootstrap_not_estimable: int
    n_bootstrap_failed: int
    specificity_support: float | None
    status: SpecificitySupportStatus
    reason_code: str | None


def _provenance_error(
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


def _evaluate_specificity_support(
    distribution: FullPipelineEffectDistribution,
    spec: SpecificitySupportSpec,
    records: tuple[FullPipelineResampleEffectRecord, ...],
) -> _SpecificityEvaluation:
    distribution._require_intact()
    spec._require_intact()
    distribution_spec = spec.effect_distribution_spec
    if (
        distribution.distribution_spec_id != distribution_spec.spec_id
        or distribution.effect_spec_id != distribution_spec.effect_spec.spec_id
        or distribution.hypothesis_id != distribution_spec.effect_spec.hypothesis_id
        or distribution.minimum_effect != distribution_spec.minimum_effect
        or distribution.specificity_direction
        is not distribution_spec.specificity_direction
    ):
        raise _provenance_error(
            "Specificity source distribution does not match the support spec",
            code="specificity_support_distribution_spec_mismatch",
            field="distribution_spec_id,effect_spec_id,hypothesis_id",
            remediation="Use the exact distribution produced for the bound spec",
        )
    if (
        distribution.n_permutation_total != 0
        or distribution.n_permutation_observed != 0
        or len(distribution.permutation_effects) != 0
    ):
        raise _provenance_error(
            "Specificity support requires a bootstrap-only source distribution",
            code="specificity_support_context_permutation_forbidden",
            field="n_permutation_total,permutation_effects",
            remediation=(
                "Summarize subject-bootstrap records in a separate distribution"
            ),
        )
    if not records:
        raise _provenance_error(
            "Specificity source records are missing",
            code="specificity_support_source_records_incomplete",
            field="resample_record_ids",
            remediation="Supply every original distribution record exactly once",
        )
    if any(not isinstance(item, FullPipelineResampleEffectRecord) for item in records):
        raise TypeError("records must contain FullPipelineResampleEffectRecord values")
    for item in records:
        item._require_intact()
    canonical = tuple(
        sorted(
            records,
            key=lambda item: (
                item.resampling_kind.value,
                item.resample_index,
                item.plan_id,
            ),
        )
    )
    if any(
        item.resampling_kind
        is FullPipelineEffectResamplingKind.CONTEXT_PERMUTATION
        for item in canonical
    ):
        raise _provenance_error(
            "Specificity support accepts subject-bootstrap records only",
            code="specificity_support_context_permutation_forbidden",
            field="resampling_kind",
            remediation=(
                "Remove context permutations and rebuild the source distribution"
            ),
        )
    if len({item.record_id for item in canonical}) != len(canonical) or len(
        {item.full_pipeline_record_id for item in canonical}
    ) != len(canonical):
        raise _provenance_error(
            "Specificity source records contain duplicate provenance",
            code="specificity_support_duplicate_source_record",
            field="record_id,full_pipeline_record_id",
            remediation="Supply every original distribution record exactly once",
        )
    if len({item.plan_id for item in canonical}) != len(canonical) or len(
        {(item.resampling_kind, item.resample_index) for item in canonical}
    ) != len(canonical):
        raise _provenance_error(
            "Specificity source records contain duplicate resampling plans",
            code="specificity_support_duplicate_resampling_plan",
            field="plan_id,resampling_kind,resample_index",
            remediation="Supply every pre-registered resampling plan exactly once",
        )
    record_ids = tuple(item.record_id for item in canonical)
    if record_ids != distribution.resample_record_ids:
        raise _provenance_error(
            "Specificity source records do not exactly reproduce the distribution",
            code="specificity_support_source_records_incomplete",
            field="resample_record_ids",
            remediation="Supply the original complete distribution record collection",
        )
    if any(
        item.effect_spec_id != distribution.effect_spec_id
        or item.hypothesis_id != distribution.hypothesis_id
        for item in canonical
    ):
        raise _provenance_error(
            "Specificity source record estimands do not match the distribution",
            code="specificity_support_source_record_spec_mismatch",
            field="effect_spec_id,hypothesis_id",
            remediation="Use records from only the bound hypothesis and effect spec",
        )
    bootstrap_records = tuple(
        item
        for item in canonical
        if item.resampling_kind
        is FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP
    )
    bootstrap_plan_ids = tuple(sorted(item.plan_id for item in bootstrap_records))
    if bootstrap_plan_ids != spec.subject_bootstrap_plan_ids:
        raise _provenance_error(
            "Observed subject-bootstrap plans do not match the frozen plan set",
            code="specificity_support_bootstrap_plan_set_mismatch",
            field="subject_bootstrap_plan_ids",
            remediation="Run and supply the exact pre-registered bootstrap plans",
        )
    bootstrap = np.asarray(
        [
            cast(float, item.effect)
            for item in bootstrap_records
            if item.status is FullPipelineResampleEffectStatus.OBSERVED
        ],
        dtype=float,
    )
    if (
        distribution.n_bootstrap_total != len(bootstrap_records)
        or distribution.n_bootstrap_observed != len(bootstrap)
        or not np.array_equal(distribution.bootstrap_effects, bootstrap)
    ):
        raise _provenance_error(
            "Specificity source distribution is inconsistent with its records",
            code="specificity_support_source_distribution_inconsistent",
            field="bootstrap_effects",
            remediation="Re-summarize the exact original resample records",
        )
    n_failed = sum(
        item.status is FullPipelineResampleEffectStatus.FAILED
        for item in bootstrap_records
    )
    n_not_estimable = sum(
        item.status is FullPipelineResampleEffectStatus.NOT_ESTIMABLE
        for item in bootstrap_records
    )
    support: float | None = None
    reason: str | None
    status: SpecificitySupportStatus
    if n_failed:
        status = SpecificitySupportStatus.FAILED
        reason = "specificity_bootstrap_record_failed"
    elif n_not_estimable:
        status = SpecificitySupportStatus.NOT_ESTIMABLE
        reason = "specificity_bootstrap_record_not_estimable"
    elif len(bootstrap_records) < spec.minimum_bootstraps:
        status = SpecificitySupportStatus.NOT_ESTIMABLE
        reason = "specificity_bootstrap_count_below_1000"
    elif distribution.point_effect is None:
        status = SpecificitySupportStatus.NOT_ESTIMABLE
        reason = "specificity_point_effect_not_observed"
    else:
        support = _specificity_frequency(
            bootstrap,
            minimum_effect=distribution_spec.minimum_effect,
            direction=distribution_spec.specificity_direction,
        )
        if (
            support is None
            or distribution.diagnostic_specificity_frequency is None
            or support != distribution.diagnostic_specificity_frequency
        ):
            raise _provenance_error(
                "Specificity diagnostic is inconsistent with bootstrap effects",
                code="specificity_support_source_diagnostic_inconsistent",
                field="diagnostic_specificity_frequency",
                remediation="Re-summarize the exact original bootstrap records",
            )
        status = SpecificitySupportStatus.OBSERVED
        reason = None
    return _SpecificityEvaluation(
        records=canonical,
        n_bootstrap_total=len(bootstrap_records),
        n_bootstrap_observed=len(bootstrap),
        n_bootstrap_not_estimable=n_not_estimable,
        n_bootstrap_failed=n_failed,
        specificity_support=support,
        status=status,
        reason_code=reason,
    )


@dataclass(frozen=True, slots=True, init=False)
class SpecificitySupportResult:
    """Producer-owned frequentist support; never a posterior probability."""

    specificity_support_spec_id: str
    source_distribution_id: str
    source_distribution_spec_id: str
    source_point_effect_result_id: str
    source_resample_record_ids: tuple[str, ...]
    hypothesis_universe_id: str
    hypothesis_id: str
    effect_spec_id: str
    effect_scale_id: str
    bootstrap_plan_set_id: str
    subject_bootstrap_plan_ids: tuple[str, ...]
    minimum_bootstraps: int
    n_bootstrap_total: int
    n_bootstrap_observed: int
    n_bootstrap_not_estimable: int
    n_bootstrap_failed: int
    minimum_effect: float
    specificity_direction: SpecificityDirection
    specificity_support: float | None
    status: SpecificitySupportStatus
    reason_code: str | None
    result_id: str
    _spec: SpecificitySupportSpec
    _source_distribution: FullPipelineEffectDistribution
    _source_records: tuple[FullPipelineResampleEffectRecord, ...]
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "SpecificitySupportResult is producer-owned; use "
            "summarize_specificity_support()"
        )

    @property
    def specificity_semantics(self) -> str:
        return _SEMANTICS

    @property
    def is_posterior_probability(self) -> bool:
        return False

    @property
    def is_comm_probability(self) -> bool:
        return False

    @property
    def source_binding_status(self) -> str:
        return _SOURCE_BINDING_STATUS

    @property
    def public_release_allowed(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "specificity_support_spec_id": self.specificity_support_spec_id,
            "source_distribution_id": self.source_distribution_id,
            "source_distribution_spec_id": self.source_distribution_spec_id,
            "source_point_effect_result_id": self.source_point_effect_result_id,
            "source_resample_record_ids": list(self.source_resample_record_ids),
            "hypothesis_universe_id": self.hypothesis_universe_id,
            "hypothesis_id": self.hypothesis_id,
            "effect_spec_id": self.effect_spec_id,
            "effect_scale_id": self.effect_scale_id,
            "bootstrap_plan_set_id": self.bootstrap_plan_set_id,
            "subject_bootstrap_plan_ids": list(self.subject_bootstrap_plan_ids),
            "minimum_bootstraps": self.minimum_bootstraps,
            "n_bootstrap_total": self.n_bootstrap_total,
            "n_bootstrap_observed": self.n_bootstrap_observed,
            "n_bootstrap_not_estimable": self.n_bootstrap_not_estimable,
            "n_bootstrap_failed": self.n_bootstrap_failed,
            "minimum_effect": self.minimum_effect,
            "specificity_direction": self.specificity_direction.value,
            "specificity_support": self.specificity_support,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "specificity_semantics": _SEMANTICS,
            "source_binding_status": _SOURCE_BINDING_STATUS,
            "public_release_allowed": False,
            "is_posterior_probability": False,
            "is_comm_probability": False,
            "producer_marker": _PRODUCER_MARKER,
        }

    def _require_intact(self) -> None:
        try:
            evaluation = _evaluate_specificity_support(
                self._source_distribution,
                self._spec,
                self._source_records,
            )
            expected = _result_values(
                self._source_distribution,
                self._spec,
                evaluation,
            )
            expected_payload = _result_identity_payload(expected)
            expected_id = stable_id(
                "specificity_support_result",
                expected_payload,
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and self._identity_payload() == expected_payload
                and self.result_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Specificity support result failed integrity validation",
                code="specificity_support_result_integrity_violation",
                field="result_id",
                remediation="Re-summarize the intact distribution and source records",
            ) from error
        if not valid:
            raise ContractError(
                "Specificity support result failed integrity validation",
                code="specificity_support_result_integrity_violation",
                field="result_id",
                remediation="Re-summarize the intact distribution and source records",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "result_id": self.result_id,
            **payload,
        }


def _result_values(
    distribution: FullPipelineEffectDistribution,
    spec: SpecificitySupportSpec,
    evaluation: _SpecificityEvaluation,
) -> dict[str, object]:
    return {
        "specificity_support_spec_id": spec.spec_id,
        "source_distribution_id": distribution.distribution_id,
        "source_distribution_spec_id": distribution.distribution_spec_id,
        "source_point_effect_result_id": distribution.point_effect_result_id,
        "source_resample_record_ids": tuple(
            item.record_id for item in evaluation.records
        ),
        "hypothesis_universe_id": spec.hypothesis_universe_id,
        "hypothesis_id": distribution.hypothesis_id,
        "effect_spec_id": distribution.effect_spec_id,
        "effect_scale_id": spec.effect_scale_id,
        "bootstrap_plan_set_id": spec.bootstrap_plan_set_id,
        "subject_bootstrap_plan_ids": spec.subject_bootstrap_plan_ids,
        "minimum_bootstraps": spec.minimum_bootstraps,
        "n_bootstrap_total": evaluation.n_bootstrap_total,
        "n_bootstrap_observed": evaluation.n_bootstrap_observed,
        "n_bootstrap_not_estimable": evaluation.n_bootstrap_not_estimable,
        "n_bootstrap_failed": evaluation.n_bootstrap_failed,
        "minimum_effect": spec.effect_distribution_spec.minimum_effect,
        "specificity_direction": (
            spec.effect_distribution_spec.specificity_direction
        ),
        "specificity_support": evaluation.specificity_support,
        "status": evaluation.status,
        "reason_code": evaluation.reason_code,
    }


def _result_identity_payload(values: dict[str, object]) -> dict[str, object]:
    direction = cast(SpecificityDirection, values["specificity_direction"])
    status = cast(SpecificitySupportStatus, values["status"])
    return {
        **values,
        "source_resample_record_ids": list(
            cast(tuple[str, ...], values["source_resample_record_ids"])
        ),
        "subject_bootstrap_plan_ids": list(
            cast(tuple[str, ...], values["subject_bootstrap_plan_ids"])
        ),
        "specificity_direction": direction.value,
        "status": status.value,
        "specificity_semantics": _SEMANTICS,
        "source_binding_status": _SOURCE_BINDING_STATUS,
        "public_release_allowed": False,
        "is_posterior_probability": False,
        "is_comm_probability": False,
        "producer_marker": _PRODUCER_MARKER,
    }


def summarize_specificity_support(
    distribution: FullPipelineEffectDistribution,
    spec: SpecificitySupportSpec,
    records: tuple[FullPipelineResampleEffectRecord, ...],
) -> SpecificitySupportResult:
    """Produce support only from the complete frozen subject-bootstrap plan set."""

    if not isinstance(distribution, FullPipelineEffectDistribution):
        raise TypeError("distribution must be FullPipelineEffectDistribution")
    if not isinstance(spec, SpecificitySupportSpec):
        raise TypeError("spec must be SpecificitySupportSpec")
    supplied = tuple(records)
    evaluation = _evaluate_specificity_support(distribution, spec, supplied)
    values = _result_values(distribution, spec, evaluation)
    self = object.__new__(SpecificitySupportResult)
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(self, "_spec", spec)
    object.__setattr__(self, "_source_distribution", distribution)
    object.__setattr__(self, "_source_records", evaluation.records)
    object.__setattr__(self, "_producer_marker", _PRODUCER_MARKER)
    object.__setattr__(
        self,
        "result_id",
        stable_id(
            "specificity_support_result",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    self._require_intact()
    return self


__all__ = [
    "SpecificitySupportResult",
    "SpecificitySupportSpec",
    "SpecificitySupportStatus",
    "summarize_specificity_support",
]
