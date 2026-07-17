"""Multi-context OOF Wald omnibus diagnostics and permutation summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import ContractError, stable_id

from .effects import OOFContextEffectResult, OOFEffectSpec, fit_oof_context_effect

_SCHEMA_VERSION = "1.0.0"
_BACKEND = "subject_equal_wls_cluster_cr2_wald_omnibus_v1"
_FORMAL_STATUS = "not_released_g3_frequency_gate_not_applied"
_MIN_FORMAL_PERMUTATIONS = 1_000
_MIN_FORMAL_CLUSTERS = 6
_RESULT_PRODUCER = "crychic.inference.oof_context_omnibus.v1"
_RECORD_PRODUCER = "crychic.inference.full_pipeline_omnibus_record.v1"
_DISTRIBUTION_PRODUCER = "crychic.inference.full_pipeline_omnibus_distribution.v1"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _finite_nonnegative(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite and non-negative") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return 0.0 if result == 0.0 else result


def _immutable_array(values: np.ndarray) -> np.ndarray:
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


def _array_payload(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype="<f8", order="C")
    return {
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _validated_context_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("context_ids must be a sequence, not a string")
    supplied = tuple(_name(value, field_name="context_id") for value in values)
    if len(supplied) < 2:
        raise ValueError("context_ids must contain at least two contexts")
    if len(set(supplied)) != len(supplied):
        raise ValueError("context_ids must be unique")
    return tuple(sorted(supplied))


def canonical_helmert_basis(context_ids: Sequence[str]) -> np.ndarray:
    """Return the immutable orthonormal zero-sum basis in canonical ID order."""

    contexts = _validated_context_ids(context_ids)
    n_contexts = len(contexts)
    basis: np.ndarray = np.zeros((n_contexts - 1, n_contexts), dtype="<f8")
    for row in range(n_contexts - 1):
        count = row + 1
        denominator = math.sqrt(count * (count + 1))
        basis[row, :count] = 1.0 / denominator
        basis[row, count] = -count / denominator
    rank = int(np.linalg.matrix_rank(basis))
    if rank != n_contexts - 1:
        raise RuntimeError("canonical Helmert basis is unexpectedly rank deficient")
    if not np.allclose(basis.sum(axis=1), 0.0, rtol=0.0, atol=1e-14):
        raise RuntimeError("canonical Helmert basis does not span zero-sum contrasts")
    return _immutable_array(basis)


def _support_effect_spec(
    hypothesis_id: str,
    omnibus_name: str,
    context_ids: tuple[str, ...],
    minimum_clusters: int,
) -> OOFEffectSpec:
    weights = tuple(
        (
            context,
            -1.0 if index == 0 else (1.0 if index == 1 else 0.0),
        )
        for index, context in enumerate(context_ids)
    )
    return OOFEffectSpec(
        hypothesis_id=hypothesis_id,
        contrast_name=f"{omnibus_name}:canonical_context_fit_support",
        contrast_weights=weights,
        minimum_clusters_for_diagnostic_se=minimum_clusters,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class OOFContextOmnibusSpec:
    """Pre-registered multi-context equality hypothesis."""

    hypothesis_id: str
    omnibus_name: str
    context_ids: tuple[str, ...]
    minimum_clusters_for_diagnostic_se: int = 8
    backend: str = _BACKEND
    support_effect_spec_id: str = field(init=False)
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        hypothesis = _name(self.hypothesis_id, field_name="hypothesis_id")
        omnibus_name = _name(self.omnibus_name, field_name="omnibus_name")
        contexts = _validated_context_ids(self.context_ids)
        minimum = self.minimum_clusters_for_diagnostic_se
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 4:
            raise ValueError("minimum clusters must be an integer >= 4")
        if self.backend != _BACKEND:
            raise ValueError(f"backend must be {_BACKEND!r}")
        support = _support_effect_spec(hypothesis, omnibus_name, contexts, minimum)
        payload = {
            "hypothesis_id": hypothesis,
            "omnibus_name": omnibus_name,
            "context_ids": list(contexts),
            "minimum_clusters_for_diagnostic_se": minimum,
            "backend": _BACKEND,
            "support_effect_spec_id": support.spec_id,
            "formal_inference_status": _FORMAL_STATUS,
        }
        object.__setattr__(self, "hypothesis_id", hypothesis)
        object.__setattr__(self, "omnibus_name", omnibus_name)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "support_effect_spec_id", support.spec_id)
        object.__setattr__(
            self,
            "spec_id",
            stable_id("oof_context_omnibus_spec", payload, schema_version="1"),
        )

    @property
    def degrees_of_freedom(self) -> int:
        return len(self.context_ids) - 1

    def support_effect_spec(self) -> OOFEffectSpec:
        """Return the fixed contrast used only to fit the shared WLS model."""

        self._require_intact()
        return _support_effect_spec(
            self.hypothesis_id,
            self.omnibus_name,
            self.context_ids,
            self.minimum_clusters_for_diagnostic_se,
        )

    def _require_intact(self) -> None:
        try:
            repeated = OOFContextOmnibusSpec(
                hypothesis_id=self.hypothesis_id,
                omnibus_name=self.omnibus_name,
                context_ids=self.context_ids,
                minimum_clusters_for_diagnostic_se=(
                    self.minimum_clusters_for_diagnostic_se
                ),
                backend=self.backend,
            )
            valid = (
                repeated.hypothesis_id == self.hypothesis_id
                and repeated.omnibus_name == self.omnibus_name
                and repeated.context_ids == self.context_ids
                and repeated.minimum_clusters_for_diagnostic_se
                == self.minimum_clusters_for_diagnostic_se
                and repeated.backend == self.backend
                and repeated.spec_id == self.spec_id
                and repeated.support_effect_spec_id == self.support_effect_spec_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "OOF omnibus specification failed integrity validation",
                code="oof_omnibus_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the immutable omnibus specification",
            ) from error
        if not valid:
            raise ContractError(
                "OOF omnibus specification failed integrity validation",
                code="oof_omnibus_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the immutable omnibus specification",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "hypothesis_id": self.hypothesis_id,
            "omnibus_name": self.omnibus_name,
            "context_ids": list(self.context_ids),
            "degrees_of_freedom": self.degrees_of_freedom,
            "minimum_clusters_for_diagnostic_se": (
                self.minimum_clusters_for_diagnostic_se
            ),
            "backend": self.backend,
            "support_effect_spec_id": self.support_effect_spec_id,
            "formal_inference_status": _FORMAL_STATUS,
        }


@dataclass(frozen=True, slots=True, init=False)
class OOFContextOmnibusResult:
    """Producer-owned multi-context Wald statistic with CR2 diagnostics."""

    spec_id: str
    hypothesis_id: str
    source_effect_spec_id: str
    source_effect_result_id: str
    source_table_digest: str
    context_ids: tuple[str, ...]
    helmert_basis: np.ndarray
    context_estimates: np.ndarray
    helmert_estimates: np.ndarray
    helmert_covariance: np.ndarray
    wald_statistic: float | None
    degrees_of_freedom: int
    n_samples: int
    n_subject_context_rows: int
    n_clusters: int
    n_folds: int
    design_rank: int
    status: str
    reason_code: str | None
    backend: str
    formal_inference_status: str
    result_id: str
    _spec: OOFContextOmnibusSpec
    _source_effect: OOFContextEffectResult
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "OOFContextOmnibusResult is producer-owned; use fit_oof_context_omnibus()"
        )

    @property
    def observed(self) -> bool:
        self._require_intact()
        return self.status == "observed"

    @property
    def formal_inference_allowed(self) -> bool:
        self._require_intact()
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "hypothesis_id": self.hypothesis_id,
            "source_effect_spec_id": self.source_effect_spec_id,
            "source_effect_result_id": self.source_effect_result_id,
            "source_table_digest": self.source_table_digest,
            "context_ids": list(self.context_ids),
            "helmert_basis": _array_payload(self.helmert_basis),
            "context_estimates": _array_payload(self.context_estimates),
            "helmert_estimates": _array_payload(self.helmert_estimates),
            "helmert_covariance": _array_payload(self.helmert_covariance),
            "wald_statistic": self.wald_statistic,
            "degrees_of_freedom": self.degrees_of_freedom,
            "n_samples": self.n_samples,
            "n_subject_context_rows": self.n_subject_context_rows,
            "n_clusters": self.n_clusters,
            "n_folds": self.n_folds,
            "design_rank": self.design_rank,
            "status": self.status,
            "reason_code": self.reason_code,
            "backend": self.backend,
            "formal_inference_status": self.formal_inference_status,
        }

    def _require_intact(self) -> None:
        try:
            self._spec._require_intact()
            repeated = _omnibus_from_effect(self._spec, self._source_effect)
            valid = (
                self._producer_marker == _RESULT_PRODUCER
                and self.result_id == repeated.result_id
                and self._identity_payload() == repeated._identity_payload()
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "OOF omnibus result failed integrity validation",
                code="oof_omnibus_result_integrity_violation",
                field="result_id",
                remediation="Refit the omnibus from its intact OOF score table",
            ) from error
        if not valid:
            raise ContractError(
                "OOF omnibus result failed integrity validation",
                code="oof_omnibus_result_integrity_violation",
                field="result_id",
                remediation="Refit the omnibus from its intact OOF score table",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "result_id": self.result_id,
            **self._identity_payload(),
            "formal_inference_allowed": False,
        }


def _new_omnibus_result(
    spec: OOFContextOmnibusSpec,
    source: OOFContextEffectResult,
    *,
    helmert_basis: np.ndarray,
    context_estimates: np.ndarray,
    helmert_estimates: np.ndarray,
    helmert_covariance: np.ndarray,
    wald_statistic: float | None,
    status: str,
    reason_code: str | None,
) -> OOFContextOmnibusResult:
    if status == "observed":
        if wald_statistic is None or reason_code is not None:
            raise ValueError("observed omnibus requires a statistic and no reason")
    elif status == "not_estimable":
        if wald_statistic is not None or not reason_code:
            raise ValueError("not-estimable omnibus requires a reason and no statistic")
    else:
        raise ValueError("omnibus status is unsupported")
    self = object.__new__(OOFContextOmnibusResult)
    values: dict[str, object] = {
        "spec_id": spec.spec_id,
        "hypothesis_id": spec.hypothesis_id,
        "source_effect_spec_id": source.spec_id,
        "source_effect_result_id": source.result_id,
        "source_table_digest": source.source_table_digest,
        "context_ids": spec.context_ids,
        "helmert_basis": _immutable_array(helmert_basis),
        "context_estimates": _immutable_array(context_estimates),
        "helmert_estimates": _immutable_array(helmert_estimates),
        "helmert_covariance": _immutable_array(helmert_covariance),
        "wald_statistic": wald_statistic,
        "degrees_of_freedom": spec.degrees_of_freedom,
        "n_samples": source.n_samples,
        "n_subject_context_rows": source.n_subject_context_rows,
        "n_clusters": source.n_clusters,
        "n_folds": source.n_folds,
        "design_rank": source.design_rank,
        "status": status,
        "reason_code": reason_code,
        "backend": _BACKEND,
        "formal_inference_status": _FORMAL_STATUS,
        "_spec": spec,
        "_source_effect": source,
        "_producer_marker": _RESULT_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "result_id",
        stable_id(
            "oof_context_omnibus_result",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    return self


def _omnibus_from_effect(
    spec: OOFContextOmnibusSpec,
    source: OOFContextEffectResult,
) -> OOFContextOmnibusResult:
    spec._require_intact()
    if not isinstance(source, OOFContextEffectResult):
        raise TypeError("source must be an OOFContextEffectResult")
    source._require_intact()
    if (
        source.spec_id != spec.support_effect_spec_id
        or source.hypothesis_id != spec.hypothesis_id
    ):
        raise ContractError(
            "OOF effect does not belong to the omnibus specification",
            code="oof_omnibus_source_effect_mismatch",
            field="spec_id,hypothesis_id",
            remediation="Fit the support effect from this exact omnibus spec",
        )
    basis = canonical_helmert_basis(spec.context_ids)
    empty_vector: np.ndarray = np.empty(0, dtype="<f8")
    empty_matrix: np.ndarray = np.empty((0, 0), dtype="<f8")
    if source.effect_status != "observed":
        return _new_omnibus_result(
            spec,
            source,
            helmert_basis=basis,
            context_estimates=empty_vector,
            helmert_estimates=empty_vector,
            helmert_covariance=empty_matrix,
            wald_statistic=None,
            status="not_estimable",
            reason_code=f"oof_omnibus_source_{source.reason_code}",
        )
    if source.n_clusters < max(
        spec.minimum_clusters_for_diagnostic_se,
        _MIN_FORMAL_CLUSTERS,
    ):
        return _new_omnibus_result(
            spec,
            source,
            helmert_basis=basis,
            context_estimates=empty_vector,
            helmert_estimates=empty_vector,
            helmert_covariance=empty_matrix,
            wald_statistic=None,
            status="not_estimable",
            reason_code="oof_omnibus_insufficient_cluster_support",
        )
    expected_context_columns = tuple(
        f"context:{context}" for context in spec.context_ids
    )
    n_contexts = len(spec.context_ids)
    if source.design_columns[:n_contexts] != expected_context_columns:
        raise ContractError(
            "OOF effect context columns do not match the omnibus universe",
            code="oof_omnibus_context_axis_mismatch",
            field="design_columns",
            remediation="Fit the support effect with the canonical context IDs",
        )
    coefficients = np.asarray(source.coefficients, dtype="<f8")
    covariance = np.asarray(source.covariance, dtype="<f8")
    if coefficients.shape[0] < n_contexts or np.any(~np.isfinite(coefficients)):
        raise ContractError(
            "OOF effect coefficients do not contain a finite context axis",
            code="oof_omnibus_source_effect_invalid",
            field="coefficients",
            remediation="Refit the intact subject-equal OOF effect model",
        )
    context_estimates = coefficients[:n_contexts]
    helmert_estimates = basis @ context_estimates
    if covariance.shape != (len(coefficients), len(coefficients)) or np.any(
        ~np.isfinite(covariance)
    ):
        return _new_omnibus_result(
            spec,
            source,
            helmert_basis=basis,
            context_estimates=context_estimates,
            helmert_estimates=helmert_estimates,
            helmert_covariance=empty_matrix,
            wald_statistic=None,
            status="not_estimable",
            reason_code="oof_omnibus_cr2_not_estimable",
        )
    context_covariance = covariance[:n_contexts, :n_contexts]
    covariance_scale = float(np.max(np.abs(context_covariance), initial=0.0))
    symmetry_tolerance = (
        np.finfo(np.float64).eps * max(context_covariance.shape) * covariance_scale
    )
    if not np.allclose(
        context_covariance,
        context_covariance.T,
        rtol=0.0,
        atol=symmetry_tolerance,
    ):
        raise ContractError(
            "OOF effect covariance is not symmetric",
            code="oof_omnibus_source_covariance_invalid",
            field="covariance",
            remediation="Refit the CR2 covariance from the intact OOF model",
        )
    helmert_covariance = basis @ context_covariance @ basis.T
    helmert_covariance = 0.5 * (helmert_covariance + helmert_covariance.T)
    eigenvalues = np.linalg.eigvalsh(helmert_covariance)
    tolerance = (
        np.finfo(np.float64).eps
        * max(helmert_covariance.shape)
        * covariance_scale
    )
    if (
        int(np.linalg.matrix_rank(helmert_covariance, tol=tolerance))
        != spec.degrees_of_freedom
    ):
        return _new_omnibus_result(
            spec,
            source,
            helmert_basis=basis,
            context_estimates=context_estimates,
            helmert_estimates=helmert_estimates,
            helmert_covariance=helmert_covariance,
            wald_statistic=None,
            status="not_estimable",
            reason_code="oof_omnibus_covariance_rank_deficient",
        )
    if float(np.min(eigenvalues)) <= tolerance:
        return _new_omnibus_result(
            spec,
            source,
            helmert_basis=basis,
            context_estimates=context_estimates,
            helmert_estimates=helmert_estimates,
            helmert_covariance=helmert_covariance,
            wald_statistic=None,
            status="not_estimable",
            reason_code="oof_omnibus_covariance_not_positive_definite",
        )
    try:
        solved = np.linalg.solve(helmert_covariance, helmert_estimates)
    except np.linalg.LinAlgError:
        return _new_omnibus_result(
            spec,
            source,
            helmert_basis=basis,
            context_estimates=context_estimates,
            helmert_estimates=helmert_estimates,
            helmert_covariance=helmert_covariance,
            wald_statistic=None,
            status="not_estimable",
            reason_code="oof_omnibus_covariance_solve_failed",
        )
    statistic = float(helmert_estimates @ solved)
    if not math.isfinite(statistic) or statistic < 0.0:
        return _new_omnibus_result(
            spec,
            source,
            helmert_basis=basis,
            context_estimates=context_estimates,
            helmert_estimates=helmert_estimates,
            helmert_covariance=helmert_covariance,
            wald_statistic=None,
            status="not_estimable",
            reason_code="oof_omnibus_wald_statistic_invalid",
        )
    return _new_omnibus_result(
        spec,
        source,
        helmert_basis=basis,
        context_estimates=context_estimates,
        helmert_estimates=helmert_estimates,
        helmert_covariance=helmert_covariance,
        wald_statistic=max(0.0, statistic),
        status="observed",
        reason_code=None,
    )


def fit_oof_context_omnibus(
    score_table: pd.DataFrame,
    spec: OOFContextOmnibusSpec,
) -> OOFContextOmnibusResult:
    """Fit the shared WLS/CR2 model and derive a multi-context Wald statistic."""

    if not isinstance(spec, OOFContextOmnibusSpec):
        raise TypeError("spec must be an OOFContextOmnibusSpec")
    spec._require_intact()
    source = fit_oof_context_effect(score_table, spec.support_effect_spec())
    return _omnibus_from_effect(spec, source)


class FullPipelineOmnibusRecordStatus(StrEnum):
    """Availability of one full-pipeline context-permutation statistic."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, init=False)
class FullPipelineOmnibusRecord:
    """Producer-owned workflow-neutral context-permutation Wald statistic."""

    full_pipeline_record_id: str
    plan_id: str
    crossfit_id: str | None
    resample_index: int
    omnibus_spec_id: str
    hypothesis_id: str
    omnibus_result_id: str | None
    wald_statistic: float | None
    status: FullPipelineOmnibusRecordStatus
    reason_code: str | None
    record_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FullPipelineOmnibusRecord is producer-owned; use "
            "build_full_pipeline_omnibus_record()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "full_pipeline_record_id": self.full_pipeline_record_id,
            "plan_id": self.plan_id,
            "crossfit_id": self.crossfit_id,
            "resampling_kind": "context_permutation",
            "resample_index": self.resample_index,
            "omnibus_spec_id": self.omnibus_spec_id,
            "hypothesis_id": self.hypothesis_id,
            "omnibus_result_id": self.omnibus_result_id,
            "wald_statistic": self.wald_statistic,
            "status": self.status.value,
            "reason_code": self.reason_code,
        }

    def _require_intact(self) -> None:
        try:
            repeated = build_full_pipeline_omnibus_record(
                full_pipeline_record_id=self.full_pipeline_record_id,
                plan_id=self.plan_id,
                crossfit_id=self.crossfit_id,
                resample_index=self.resample_index,
                omnibus_spec_id=self.omnibus_spec_id,
                hypothesis_id=self.hypothesis_id,
                omnibus_result_id=self.omnibus_result_id,
                wald_statistic=self.wald_statistic,
                status=self.status,
                reason_code=self.reason_code,
            )
            valid = (
                self._producer_marker == _RECORD_PRODUCER
                and self.record_id == repeated.record_id
                and self._identity_payload() == repeated._identity_payload()
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Full-pipeline omnibus record failed integrity validation",
                code="full_pipeline_omnibus_record_integrity_violation",
                field="record_id",
                remediation="Rebuild the record from the exact permutation result",
            ) from error
        if not valid:
            raise ContractError(
                "Full-pipeline omnibus record failed integrity validation",
                code="full_pipeline_omnibus_record_integrity_violation",
                field="record_id",
                remediation="Rebuild the record from the exact permutation result",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"record_id": self.record_id, **self._identity_payload()}


def build_full_pipeline_omnibus_record(
    *,
    full_pipeline_record_id: str,
    plan_id: str,
    crossfit_id: str | None,
    resample_index: int,
    omnibus_spec_id: str,
    hypothesis_id: str,
    omnibus_result_id: str | None,
    wald_statistic: float | None,
    status: FullPipelineOmnibusRecordStatus,
    reason_code: str | None,
) -> FullPipelineOmnibusRecord:
    """Build one immutable context-permutation statistic and lineage record."""

    identifiers = {
        "full_pipeline_record_id": _name(
            full_pipeline_record_id,
            field_name="full_pipeline_record_id",
        ),
        "plan_id": _name(plan_id, field_name="plan_id"),
        "omnibus_spec_id": _name(omnibus_spec_id, field_name="omnibus_spec_id"),
        "hypothesis_id": _name(hypothesis_id, field_name="hypothesis_id"),
    }
    if (
        isinstance(resample_index, bool)
        or not isinstance(resample_index, int)
        or resample_index < 0
    ):
        raise ValueError("resample_index must be a non-negative integer")
    normalized_status = FullPipelineOmnibusRecordStatus(status)
    crossfit: str | None
    result_id: str | None
    statistic: float | None
    reason: str | None
    if normalized_status is FullPipelineOmnibusRecordStatus.OBSERVED:
        crossfit = _name(crossfit_id, field_name="crossfit_id")
        result_id = _name(omnibus_result_id, field_name="omnibus_result_id")
        statistic = _finite_nonnegative(
            wald_statistic,
            field_name="wald_statistic",
        )
        if reason_code is not None:
            raise ValueError("observed omnibus record cannot have a reason")
        reason = None
    elif normalized_status is FullPipelineOmnibusRecordStatus.NOT_ESTIMABLE:
        crossfit = _name(crossfit_id, field_name="crossfit_id")
        result_id = _name(omnibus_result_id, field_name="omnibus_result_id")
        if wald_statistic is not None:
            raise ValueError("not-estimable omnibus record cannot have a statistic")
        statistic = None
        reason = _name(reason_code, field_name="reason_code")
    else:
        crossfit = (
            None
            if crossfit_id is None
            else _name(crossfit_id, field_name="crossfit_id")
        )
        result_id = (
            None
            if omnibus_result_id is None
            else _name(omnibus_result_id, field_name="omnibus_result_id")
        )
        if wald_statistic is not None:
            raise ValueError("failed omnibus record cannot have a statistic")
        statistic = None
        reason = _name(reason_code, field_name="reason_code")
    self = object.__new__(FullPipelineOmnibusRecord)
    values: dict[str, object] = {
        **identifiers,
        "crossfit_id": crossfit,
        "resample_index": resample_index,
        "omnibus_result_id": result_id,
        "wald_statistic": statistic,
        "status": normalized_status,
        "reason_code": reason,
        "_producer_marker": _RECORD_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "record_id",
        stable_id(
            "full_pipeline_omnibus_record",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    return self


@dataclass(frozen=True, slots=True, init=False)
class FullPipelineOmnibusDistribution:
    """Diagnostic empirical omnibus p-value with no formal release authority."""

    omnibus_spec_id: str
    hypothesis_id: str
    point_result_id: str
    point_wald_statistic: float | None
    record_ids: tuple[str, ...]
    permutation_plan_ids: tuple[str, ...]
    full_pipeline_record_ids: tuple[str, ...]
    permutation_statistics: np.ndarray
    n_permutation_total: int
    n_permutation_observed: int
    diagnostic_empirical_p: float | None
    diagnostic_status: str
    formal_inference_status: str
    formal_reason_code: str
    p_value: None
    q_value: None
    distribution_id: str
    _point_result: OOFContextOmnibusResult
    _spec: OOFContextOmnibusSpec
    _records: tuple[FullPipelineOmnibusRecord, ...]
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FullPipelineOmnibusDistribution is producer-owned; use "
            "summarize_full_pipeline_omnibus()"
        )

    @property
    def formal_inference_allowed(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "omnibus_spec_id": self.omnibus_spec_id,
            "hypothesis_id": self.hypothesis_id,
            "point_result_id": self.point_result_id,
            "point_wald_statistic": self.point_wald_statistic,
            "record_ids": list(self.record_ids),
            "permutation_plan_ids": list(self.permutation_plan_ids),
            "full_pipeline_record_ids": list(self.full_pipeline_record_ids),
            "permutation_statistics": _array_payload(self.permutation_statistics),
            "n_permutation_total": self.n_permutation_total,
            "n_permutation_observed": self.n_permutation_observed,
            "diagnostic_empirical_p": self.diagnostic_empirical_p,
            "diagnostic_status": self.diagnostic_status,
            "minimum_formal_permutations": _MIN_FORMAL_PERMUTATIONS,
            "formal_inference_status": self.formal_inference_status,
            "formal_reason_code": self.formal_reason_code,
            "p_value": None,
            "q_value": None,
        }

    def _require_intact(self) -> None:
        try:
            repeated = summarize_full_pipeline_omnibus(
                self._point_result,
                self._spec,
                self._records,
            )
            valid = (
                self._producer_marker == _DISTRIBUTION_PRODUCER
                and self.distribution_id == repeated.distribution_id
                and self._identity_payload() == repeated._identity_payload()
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Full-pipeline omnibus distribution failed integrity validation",
                code="full_pipeline_omnibus_distribution_integrity_violation",
                field="distribution_id",
                remediation="Re-summarize the exact complete permutation records",
            ) from error
        if not valid:
            raise ContractError(
                "Full-pipeline omnibus distribution failed integrity validation",
                code="full_pipeline_omnibus_distribution_integrity_violation",
                field="distribution_id",
                remediation="Re-summarize the exact complete permutation records",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "distribution_id": self.distribution_id,
            **self._identity_payload(),
            "formal_inference_allowed": False,
        }


def summarize_full_pipeline_omnibus(
    point_result: OOFContextOmnibusResult,
    spec: OOFContextOmnibusSpec,
    records: Sequence[FullPipelineOmnibusRecord],
) -> FullPipelineOmnibusDistribution:
    """Summarize full-pipeline permutation Wald statistics as diagnostic only."""

    if not isinstance(point_result, OOFContextOmnibusResult):
        raise TypeError("point_result must be an OOFContextOmnibusResult")
    if not isinstance(spec, OOFContextOmnibusSpec):
        raise TypeError("spec must be an OOFContextOmnibusSpec")
    point_result._require_intact()
    spec._require_intact()
    if point_result.spec_id != spec.spec_id or (
        point_result.hypothesis_id != spec.hypothesis_id
    ):
        raise ContractError(
            "Point omnibus does not match the supplied specification",
            code="full_pipeline_omnibus_point_spec_mismatch",
            field="spec_id,hypothesis_id",
            remediation="Use the exact point result produced from this spec",
        )
    supplied = tuple(records)
    if not supplied or any(
        not isinstance(record, FullPipelineOmnibusRecord) for record in supplied
    ):
        raise ValueError("records must contain typed omnibus permutation rows")
    for record in supplied:
        record._require_intact()
    for field_name, provenance_values in (
        ("record_id", tuple(record.record_id for record in supplied)),
        (
            "full_pipeline_record_id",
            tuple(record.full_pipeline_record_id for record in supplied),
        ),
        ("plan_id", tuple(record.plan_id for record in supplied)),
        ("resample_index", tuple(record.resample_index for record in supplied)),
    ):
        if len(set(provenance_values)) != len(provenance_values):
            raise ContractError(
                "Full-pipeline omnibus records contain duplicate provenance",
                code="duplicate_full_pipeline_omnibus_record",
                field=field_name,
                remediation="Include every context permutation exactly once",
            )
    if any(
        record.omnibus_spec_id != spec.spec_id
        or record.hypothesis_id != spec.hypothesis_id
        for record in supplied
    ):
        raise ContractError(
            "Permutation omnibus records do not match the point specification",
            code="full_pipeline_omnibus_record_spec_mismatch",
            field="omnibus_spec_id,hypothesis_id",
            remediation="Aggregate only one pre-registered omnibus hypothesis",
        )
    ordered = tuple(
        sorted(supplied, key=lambda record: (record.resample_index, record.plan_id))
    )
    statistics = _immutable_array(
        np.asarray(
            [
                record.wald_statistic
                for record in ordered
                if record.status is FullPipelineOmnibusRecordStatus.OBSERVED
            ],
            dtype="<f8",
        )
    )
    point_statistic = (
        point_result.wald_statistic if point_result.status == "observed" else None
    )
    diagnostic_p: float | None
    if point_statistic is None or not len(statistics):
        diagnostic_p = None
    else:
        tolerance = 1e-12 * max(1.0, point_statistic)
        extreme = statistics >= point_statistic - tolerance
        diagnostic_p = float(
            (1 + int(np.count_nonzero(extreme))) / (len(statistics) + 1)
        )
    if point_statistic is None:
        diagnostic_status = "not_estimable_point_omnibus"
        formal_reason = "full_pipeline_omnibus_point_not_observed"
    elif len(statistics) != len(ordered):
        diagnostic_status = "diagnostic_incomplete_permutation_distribution"
        formal_reason = "full_pipeline_omnibus_permutation_distribution_incomplete"
    elif len(statistics) < _MIN_FORMAL_PERMUTATIONS:
        diagnostic_status = "diagnostic_permutation_count_below_1000"
        formal_reason = "full_pipeline_omnibus_permutations_below_1000"
    else:
        diagnostic_status = "diagnostic_complete_minimum_1000"
        formal_reason = "g3_frequency_calibration_gate_not_applied"
    self = object.__new__(FullPipelineOmnibusDistribution)
    attributes: dict[str, object] = {
        "omnibus_spec_id": spec.spec_id,
        "hypothesis_id": spec.hypothesis_id,
        "point_result_id": point_result.result_id,
        "point_wald_statistic": point_statistic,
        "record_ids": tuple(record.record_id for record in ordered),
        "permutation_plan_ids": tuple(record.plan_id for record in ordered),
        "full_pipeline_record_ids": tuple(
            record.full_pipeline_record_id for record in ordered
        ),
        "permutation_statistics": statistics,
        "n_permutation_total": len(ordered),
        "n_permutation_observed": len(statistics),
        "diagnostic_empirical_p": diagnostic_p,
        "diagnostic_status": diagnostic_status,
        "formal_inference_status": _FORMAL_STATUS,
        "formal_reason_code": formal_reason,
        "p_value": None,
        "q_value": None,
        "_point_result": point_result,
        "_spec": spec,
        "_records": ordered,
        "_producer_marker": _DISTRIBUTION_PRODUCER,
    }
    for name, value in attributes.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "distribution_id",
        stable_id(
            "full_pipeline_omnibus_distribution",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    return self


__all__ = [
    "FullPipelineOmnibusDistribution",
    "FullPipelineOmnibusRecord",
    "FullPipelineOmnibusRecordStatus",
    "OOFContextOmnibusResult",
    "OOFContextOmnibusSpec",
    "build_full_pipeline_omnibus_record",
    "canonical_helmert_basis",
    "fit_oof_context_omnibus",
    "summarize_full_pipeline_omnibus",
]
