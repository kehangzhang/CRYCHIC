"""Bounded full-pipeline subject bootstrap and context permutation workflows."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from typing import TypeAlias, cast

import numpy as np
from anndata import AnnData

from crychic.core import (
    ContractError,
    CrychicConfig,
    CrychicError,
    SeedLineage,
    stable_id,
)
from crychic.data import validate_anndata
from crychic.resampling import (
    ContextPermutationPlan,
    ExchangeabilityMap,
    SubjectBootstrapPlan,
    apply_context_permutation,
    build_exchangeability_map,
    plan_context_permutations,
    plan_subject_bootstraps,
)
from crychic.resources import ResourceBundle, TargetPrior

from .crossfit import CrossFitArtifacts, CrossFitSpec, _run_subject_crossfit
from .receiver_universe import (
    FrozenReceiverUniverse,
    ReceiverUniverseReuseBinding,
    bind_reused_receiver_universe,
    freeze_receiver_universe,
)
from .training import (
    _input_schema,
    _resource_bundle_content_id,
    _sanitized_raw_input_snapshot,
    _target_prior_content_id,
)

_SCHEMA_VERSION = "1.0.0"
_INFERENCE_STATUS = "not_released_full_pipeline_resampling_diagnostic_only"
_MATERIALIZATION_POLICY = "deterministic_complete_subject_or_sample_block_v1"
_RERUN_POLICY_ID = "full_pipeline_refit_from_materialized_anndata_v1"
_RERUN_STAGES = (
    "nuisance_response_fitting",
    "feature_and_interaction_filtering",
    "availability_and_receptor_gating",
    "driver_family_clustering",
    "hyperparameter_tuning",
    "receiver_attribution",
    "sender_assignment",
    "family_common_scoring",
    "effect_fitting",
)
_INFERENTIAL_FIELDS_UNAVAILABLE = (
    "p_value",
    "q_value",
    "communication_probability",
    "posterior_probability",
)
_SERIAL_BACKEND = "serial_v1"
_THREAD_BACKEND = "bounded_shared_snapshot_thread_pool_v1"
_SHARED_SNAPSHOT_POLICY = "immutable_sanitized_root_snapshot_shared_read_only_v1"


class FullPipelineResamplingOperation(StrEnum):
    """Materialized subject-level operation run through the complete workflow."""

    SUBJECT_BOOTSTRAP = "subject_bootstrap"
    CONTEXT_PERMUTATION = "context_permutation"


class FullPipelineResampleStatus(StrEnum):
    """Execution status of one fully materialized resample."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FullPipelineResamplingStatus(StrEnum):
    """Aggregate completion status across requested resamples."""

    SUCCEEDED = "succeeded"
    PARTIAL_FAILURE = "partial_failure"
    FAILED = "failed"


ResamplingPlan: TypeAlias = ContextPermutationPlan | SubjectBootstrapPlan


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _count(value: int | None, *, field_name: str, required: bool) -> int | None:
    if value is None:
        if required:
            raise ValueError(f"successful resample requires {field_name}")
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer >= 1 when present")
    return value


def _plan_id(plan: ResamplingPlan) -> str:
    if isinstance(plan, ContextPermutationPlan):
        identifier: str = plan.permutation_id
        return identifier
    if isinstance(plan, SubjectBootstrapPlan):
        identifier = plan.bootstrap_id
        return identifier
    raise TypeError("resampling plan type is unsupported")


def _plan_operation(plan: ResamplingPlan) -> FullPipelineResamplingOperation:
    if isinstance(plan, ContextPermutationPlan):
        return FullPipelineResamplingOperation.CONTEXT_PERMUTATION
    if isinstance(plan, SubjectBootstrapPlan):
        return FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP
    raise TypeError("resampling plan type is unsupported")


def _require_retained_child_alignment(
    records: tuple[FullPipelineResampleRecord, ...],
    *,
    crossfit_spec_id: str,
    resource_bundle_content_id: str,
    target_prior_content_id: str,
) -> None:
    for record in records:
        child = record.child
        if child is None:
            continue
        resource_ids = {
            fold.training.resource_bundle_content_id for fold in child.folds
        }
        target_prior_ids = {
            fold.training.target_prior_content_id for fold in child.folds
        }
        if (
            child.spec.spec_id != crossfit_spec_id
            or child.root_input_identity.config_digest != record.crossfit_config_digest
            or resource_ids != {resource_bundle_content_id}
            or target_prior_ids != {target_prior_content_id}
        ):
            raise ContractError(
                "Retained cross-fit child does not match resampling lineage",
                code="full_pipeline_resampling_child_lineage_mismatch",
                field=(
                    "crossfit_spec_id,crossfit_config_digest,"
                    "resource_bundle_content_id,target_prior_content_id"
                ),
                remediation=(
                    "Retain only children produced by the exact resampling "
                    "configuration and versioned resources"
                ),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class FullPipelineResampleRecord:
    """Log-safe execution record for one complete cross-fit rerun."""

    operation: FullPipelineResamplingOperation
    resample_index: int
    plan_id: str
    exchangeability_id: str
    materialized_input_id: str
    plan_seed_lineage: SeedLineage
    crossfit_seed_lineage: SeedLineage
    crossfit_config_digest: str
    status: FullPipelineResampleStatus
    crossfit_id: str | None
    failure_type: str | None
    failure_code: str | None
    n_obs: int | None
    n_samples: int | None
    n_subjects: int | None
    source_receiver_universe_id: str | None = None
    source_receiver_axis_id: str | None = None
    child_receiver_universe_id: str | None = None
    child_receiver_axis_id: str | None = None
    receiver_universe_reuse_binding_id: str | None = None
    _child: CrossFitArtifacts | None = field(default=None, repr=False)
    _receiver_universe_reuse_binding: ReceiverUniverseReuseBinding | None = field(
        default=None,
        repr=False,
    )
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        operation = FullPipelineResamplingOperation(self.operation)
        status = FullPipelineResampleStatus(self.status)
        if (
            isinstance(self.resample_index, bool)
            or not isinstance(self.resample_index, int)
            or self.resample_index < 0
        ):
            raise ValueError("resample_index must be a non-negative integer")
        identifiers = {
            name: _name(cast(str, getattr(self, name)), field_name=name)
            for name in (
                "plan_id",
                "exchangeability_id",
                "materialized_input_id",
                "crossfit_config_digest",
            )
        }
        if not isinstance(self.plan_seed_lineage, SeedLineage) or not isinstance(
            self.crossfit_seed_lineage, SeedLineage
        ):
            raise TypeError("resample seed lineages must be SeedLineage values")
        if self.crossfit_seed_lineage.path[: len(self.plan_seed_lineage.path)] != (
            self.plan_seed_lineage.path
        ):
            raise ValueError(
                "cross-fit seed lineage must descend from its plan lineage"
            )
        required_counts = status is FullPipelineResampleStatus.SUCCEEDED
        counts = {
            name: _count(
                cast(int | None, getattr(self, name)),
                field_name=name,
                required=required_counts,
            )
            for name in ("n_obs", "n_samples", "n_subjects")
        }
        source_lineage = (
            self.source_receiver_universe_id,
            self.source_receiver_axis_id,
        )
        if any(value is not None for value in source_lineage) and not all(
            value is not None for value in source_lineage
        ):
            raise ValueError("source receiver universe lineage must be complete")
        source_universe_id = (
            None
            if self.source_receiver_universe_id is None
            else _name(
                self.source_receiver_universe_id,
                field_name="source_receiver_universe_id",
            )
        )
        source_axis_id = (
            None
            if self.source_receiver_axis_id is None
            else _name(
                self.source_receiver_axis_id,
                field_name="source_receiver_axis_id",
            )
        )
        child_universe_id: str | None = None
        child_axis_id: str | None = None
        reuse_binding_id: str | None = None
        if status is FullPipelineResampleStatus.SUCCEEDED:
            crossfit_id = _name(cast(str, self.crossfit_id), field_name="crossfit_id")
            if self.failure_type is not None or self.failure_code is not None:
                raise ValueError("successful resample cannot contain failure metadata")
            if self._child is not None:
                if not isinstance(self._child, CrossFitArtifacts):
                    raise TypeError("retained child must be CrossFitArtifacts")
                if self._child.crossfit_id != crossfit_id:
                    raise ValueError("retained child crossfit_id does not match record")
            if self._receiver_universe_reuse_binding is not None:
                binding = self._receiver_universe_reuse_binding
                if not isinstance(binding, ReceiverUniverseReuseBinding):
                    raise TypeError("invalid receiver-universe reuse binding type")
                binding._require_intact()
                child_universe_id = _name(
                    cast(str, self.child_receiver_universe_id),
                    field_name="child_receiver_universe_id",
                )
                child_axis_id = _name(
                    cast(str, self.child_receiver_axis_id),
                    field_name="child_receiver_axis_id",
                )
                reuse_binding_id = _name(
                    cast(str, self.receiver_universe_reuse_binding_id),
                    field_name="receiver_universe_reuse_binding_id",
                )
                if (
                    source_universe_id != binding.source_receiver_universe_id
                    or source_axis_id != binding.receiver_axis_id
                    or child_universe_id != binding.child_receiver_universe_id
                    or child_axis_id != binding.receiver_axis_id
                    or reuse_binding_id != binding.binding_id
                    or binding.operation != operation.value
                    or binding.plan_id != identifiers["plan_id"]
                    or binding.resample_index != self.resample_index
                    or binding.materialized_input_id
                    != identifiers["materialized_input_id"]
                    or binding.child_root_config_digest
                    != identifiers["crossfit_config_digest"]
                ):
                    raise ValueError("receiver-universe reuse lineage is misaligned")
                if self._child is not None and (
                    self._child.receiver_universe.universe_id != child_universe_id
                    or self._child.receiver_universe.receiver_axis_id != child_axis_id
                ):
                    raise ValueError(
                        "retained child receiver universe does not match binding"
                    )
            elif any(
                value is not None
                for value in (
                    source_universe_id,
                    source_axis_id,
                    self.child_receiver_universe_id,
                    self.child_receiver_axis_id,
                    self.receiver_universe_reuse_binding_id,
                )
            ):
                raise ValueError(
                    "successful receiver-axis lineage requires a reuse binding"
                )
        else:
            crossfit_id = None
            if self.crossfit_id is not None or self._child is not None:
                raise ValueError("failed resample cannot retain cross-fit output")
            _name(cast(str, self.failure_type), field_name="failure_type")
            _name(cast(str, self.failure_code), field_name="failure_code")
            if any(
                value is not None
                for value in (
                    self.child_receiver_universe_id,
                    self.child_receiver_axis_id,
                    self.receiver_universe_reuse_binding_id,
                    self._receiver_universe_reuse_binding,
                )
            ):
                raise ValueError(
                    "failed resample cannot contain child receiver lineage"
                )
        payload = {
            "operation": operation.value,
            "resample_index": self.resample_index,
            **identifiers,
            "plan_seed_lineage": self.plan_seed_lineage.to_dict(),
            "crossfit_seed_lineage": self.crossfit_seed_lineage.to_dict(),
            "status": status.value,
            "crossfit_id": crossfit_id,
            "failure_type": self.failure_type,
            "failure_code": self.failure_code,
            "source_receiver_universe_id": source_universe_id,
            "source_receiver_axis_id": source_axis_id,
            "child_receiver_universe_id": child_universe_id,
            "child_receiver_axis_id": child_axis_id,
            "receiver_universe_reuse_binding_id": reuse_binding_id,
            **counts,
            "rerun_policy_id": _RERUN_POLICY_ID,
            "score_table_relabeling_only": False,
            "formal_inference_status": _INFERENCE_STATUS,
        }
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "crossfit_id", crossfit_id)
        object.__setattr__(
            self,
            "source_receiver_universe_id",
            source_universe_id,
        )
        object.__setattr__(self, "source_receiver_axis_id", source_axis_id)
        object.__setattr__(self, "child_receiver_universe_id", child_universe_id)
        object.__setattr__(self, "child_receiver_axis_id", child_axis_id)
        object.__setattr__(
            self,
            "receiver_universe_reuse_binding_id",
            reuse_binding_id,
        )
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        for name, count_value in counts.items():
            object.__setattr__(self, name, count_value)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "full_pipeline_resample_record",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation.value,
            "resample_index": self.resample_index,
            "plan_id": self.plan_id,
            "exchangeability_id": self.exchangeability_id,
            "materialized_input_id": self.materialized_input_id,
            "crossfit_config_digest": self.crossfit_config_digest,
            "plan_seed_lineage": self.plan_seed_lineage.to_dict(),
            "crossfit_seed_lineage": self.crossfit_seed_lineage.to_dict(),
            "status": self.status.value,
            "crossfit_id": self.crossfit_id,
            "failure_type": self.failure_type,
            "failure_code": self.failure_code,
            "source_receiver_universe_id": self.source_receiver_universe_id,
            "source_receiver_axis_id": self.source_receiver_axis_id,
            "child_receiver_universe_id": self.child_receiver_universe_id,
            "child_receiver_axis_id": self.child_receiver_axis_id,
            "receiver_universe_reuse_binding_id": (
                self.receiver_universe_reuse_binding_id
            ),
            "n_obs": self.n_obs,
            "n_samples": self.n_samples,
            "n_subjects": self.n_subjects,
            "rerun_policy_id": _RERUN_POLICY_ID,
            "score_table_relabeling_only": False,
            "formal_inference_status": _INFERENCE_STATUS,
        }

    def _require_intact(self) -> None:
        try:
            repeated = FullPipelineResampleRecord(
                operation=self.operation,
                resample_index=self.resample_index,
                plan_id=self.plan_id,
                exchangeability_id=self.exchangeability_id,
                materialized_input_id=self.materialized_input_id,
                plan_seed_lineage=self.plan_seed_lineage,
                crossfit_seed_lineage=self.crossfit_seed_lineage,
                crossfit_config_digest=self.crossfit_config_digest,
                status=self.status,
                crossfit_id=self.crossfit_id,
                failure_type=self.failure_type,
                failure_code=self.failure_code,
                n_obs=self.n_obs,
                n_samples=self.n_samples,
                n_subjects=self.n_subjects,
                source_receiver_universe_id=self.source_receiver_universe_id,
                source_receiver_axis_id=self.source_receiver_axis_id,
                child_receiver_universe_id=self.child_receiver_universe_id,
                child_receiver_axis_id=self.child_receiver_axis_id,
                receiver_universe_reuse_binding_id=(
                    self.receiver_universe_reuse_binding_id
                ),
                _child=self._child,
                _receiver_universe_reuse_binding=(
                    self._receiver_universe_reuse_binding
                ),
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.record_id == repeated.record_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Full-pipeline resample record failed integrity validation",
                code="full_pipeline_resample_record_integrity_violation",
                field="record_id",
                remediation="Re-execute the exact frozen resampling plan",
            ) from error
        if not valid:
            raise ContractError(
                "Full-pipeline resample record failed integrity validation",
                code="full_pipeline_resample_record_integrity_violation",
                field="record_id",
                remediation="Re-execute the exact frozen resampling plan",
            )

    @property
    def child(self) -> CrossFitArtifacts | None:
        """Return the explicitly retained child, otherwise ``None``."""

        return self._child

    @property
    def receiver_universe_reuse_binding(
        self,
    ) -> ReceiverUniverseReuseBinding | None:
        """Return the small source-to-child receiver-axis proof when available."""

        return self._receiver_universe_reuse_binding

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "record_id": self.record_id,
            "operation": self.operation.value,
            "resample_index": self.resample_index,
            "plan_id": self.plan_id,
            "exchangeability_id": self.exchangeability_id,
            "materialized_input_id": self.materialized_input_id,
            "plan_seed_lineage": self.plan_seed_lineage.to_dict(),
            "crossfit_seed_lineage": self.crossfit_seed_lineage.to_dict(),
            "crossfit_config_digest": self.crossfit_config_digest,
            "status": self.status.value,
            "crossfit_id": self.crossfit_id,
            "failure_type": self.failure_type,
            "failure_code": self.failure_code,
            "source_receiver_universe_id": self.source_receiver_universe_id,
            "source_receiver_axis_id": self.source_receiver_axis_id,
            "child_receiver_universe_id": self.child_receiver_universe_id,
            "child_receiver_axis_id": self.child_receiver_axis_id,
            "receiver_universe_reuse_binding_id": (
                self.receiver_universe_reuse_binding_id
            ),
            "receiver_universe_reuse_binding": (
                None
                if self._receiver_universe_reuse_binding is None
                else self._receiver_universe_reuse_binding.to_dict()
            ),
            "n_obs": self.n_obs,
            "n_samples": self.n_samples,
            "n_subjects": self.n_subjects,
            "child_retained": self._child is not None,
            "rerun_policy_id": _RERUN_POLICY_ID,
            "score_table_relabeling_only": False,
            "formal_inference_status": _INFERENCE_STATUS,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FullPipelineResamplingResult:
    """Auditable bounded execution of complete subject-level resamples."""

    exchangeability: ExchangeabilityMap
    plans: tuple[ResamplingPlan, ...]
    records: tuple[FullPipelineResampleRecord, ...]
    source_input_identity_id: str
    source_input_digest: str
    source_snapshot_id: str
    config_digest: str
    crossfit_spec_id: str
    resource_bundle_content_id: str
    target_prior_content_id: str
    root_seed_lineage: SeedLineage
    retain_children: bool
    requested_n_jobs: int = 1
    effective_n_jobs: int = 1
    source_receiver_universe_id: str | None = None
    source_receiver_axis_id: str | None = None
    _source_receiver_universe: FrozenReceiverUniverse | None = field(
        default=None,
        repr=False,
    )
    result_id: str = field(init=False)
    execution_metadata_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.exchangeability, ExchangeabilityMap):
            raise TypeError("exchangeability must be an ExchangeabilityMap")
        plans = tuple(self.plans)
        records = tuple(self.records)
        if not plans or len(plans) != len(records):
            raise ValueError("plans and records must contain one aligned resample each")
        if any(
            not isinstance(plan, (ContextPermutationPlan, SubjectBootstrapPlan))
            for plan in plans
        ) or any(
            not isinstance(record, FullPipelineResampleRecord) for record in records
        ):
            raise TypeError(
                "resampling result contains unsupported plan or record values"
            )
        for plan, record in zip(plans, records, strict=True):
            if (
                _plan_id(plan) != record.plan_id
                or _plan_operation(plan) is not record.operation
                or plan.resample_index != record.resample_index
                or plan.exchangeability_id != self.exchangeability.exchangeability_id
                or record.exchangeability_id != self.exchangeability.exchangeability_id
                or plan.seed_lineage != record.plan_seed_lineage
            ):
                raise ValueError(
                    "resampling plans and execution records are misaligned"
                )
        plan_ids = tuple(_plan_id(plan) for plan in plans)
        plan_keys = tuple(
            (_plan_operation(plan), plan.resample_index) for plan in plans
        )
        record_ids = tuple(record.record_id for record in records)
        if (
            len(set(plan_ids)) != len(plan_ids)
            or len(set(plan_keys)) != len(plan_keys)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise ContractError(
                "Full-pipeline resampling contains duplicate plan provenance",
                code="duplicate_full_pipeline_resampling_plan",
                field="plan_id,operation,resample_index,record_id",
                remediation="Generate every frozen resampling plan exactly once",
            )
        identifiers = {
            name: _name(cast(str, getattr(self, name)), field_name=name)
            for name in (
                "source_input_identity_id",
                "source_input_digest",
                "source_snapshot_id",
                "config_digest",
                "crossfit_spec_id",
                "resource_bundle_content_id",
                "target_prior_content_id",
            )
        }
        if not isinstance(self.root_seed_lineage, SeedLineage):
            raise TypeError("root_seed_lineage must be a SeedLineage")
        if not isinstance(self.retain_children, bool):
            raise TypeError("retain_children must be boolean")
        requested_n_jobs = _positive_jobs(
            self.requested_n_jobs,
            field_name="requested_n_jobs",
        )
        effective_n_jobs = _positive_jobs(
            self.effective_n_jobs,
            field_name="effective_n_jobs",
        )
        if effective_n_jobs != min(requested_n_jobs, len(plans)):
            raise ValueError(
                "effective_n_jobs must equal min(requested_n_jobs, n_plans)"
            )
        source_receiver_values = (
            self.source_receiver_universe_id,
            self.source_receiver_axis_id,
            self._source_receiver_universe,
        )
        has_receiver_lineage = any(
            value is not None for value in source_receiver_values
        )
        if has_receiver_lineage and not all(
            value is not None for value in source_receiver_values
        ):
            raise ValueError("source receiver-universe lineage must be complete")
        if has_receiver_lineage:
            source_receiver_universe = self._source_receiver_universe
            if not isinstance(source_receiver_universe, FrozenReceiverUniverse):
                raise TypeError("invalid source receiver universe type")
            source_receiver_universe._require_intact()
            source_receiver_universe_id = _name(
                cast(str, self.source_receiver_universe_id),
                field_name="source_receiver_universe_id",
            )
            source_receiver_axis_id = _name(
                cast(str, self.source_receiver_axis_id),
                field_name="source_receiver_axis_id",
            )
            if (
                source_receiver_universe_id != source_receiver_universe.universe_id
                or source_receiver_axis_id != source_receiver_universe.receiver_axis_id
                or source_receiver_universe.root_input_identity_id
                != identifiers["source_input_identity_id"]
                or source_receiver_universe.root_input_digest
                != identifiers["source_input_digest"]
                or source_receiver_universe.root_config_digest
                != identifiers["config_digest"]
            ):
                raise ValueError("source receiver universe does not match source input")
            for record in records:
                if (
                    record.source_receiver_universe_id != source_receiver_universe_id
                    or record.source_receiver_axis_id != source_receiver_axis_id
                    or (
                        record.status is FullPipelineResampleStatus.SUCCEEDED
                        and record.receiver_universe_reuse_binding is None
                    )
                ):
                    raise ValueError(
                        "resample record does not preserve source receiver lineage"
                    )
        else:
            source_receiver_universe = None
            source_receiver_universe_id = None
            source_receiver_axis_id = None
            if any(
                record.source_receiver_universe_id is not None
                or record.source_receiver_axis_id is not None
                for record in records
            ):
                raise ValueError(
                    "record receiver lineage requires a result source universe"
                )
        if self.retain_children:
            if any(
                record.status is FullPipelineResampleStatus.SUCCEEDED
                and record.child is None
                for record in records
            ):
                raise ValueError("retained mode requires every successful child")
        elif any(record.child is not None for record in records):
            raise ValueError("non-retained mode cannot hold cross-fit children")
        _require_retained_child_alignment(
            records,
            crossfit_spec_id=identifiers["crossfit_spec_id"],
            resource_bundle_content_id=identifiers["resource_bundle_content_id"],
            target_prior_content_id=identifiers["target_prior_content_id"],
        )
        payload = {
            "exchangeability_id": self.exchangeability.exchangeability_id,
            "plan_ids": [_plan_id(plan) for plan in plans],
            "record_ids": [record.record_id for record in records],
            **identifiers,
            "source_receiver_universe_id": source_receiver_universe_id,
            "source_receiver_axis_id": source_receiver_axis_id,
            "root_seed_lineage": self.root_seed_lineage.to_dict(),
            "retain_children": self.retain_children,
            "rerun_policy_id": _RERUN_POLICY_ID,
            "formal_inference_status": _INFERENCE_STATUS,
        }
        object.__setattr__(self, "plans", plans)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "requested_n_jobs", requested_n_jobs)
        object.__setattr__(self, "effective_n_jobs", effective_n_jobs)
        object.__setattr__(
            self,
            "source_receiver_universe_id",
            source_receiver_universe_id,
        )
        object.__setattr__(
            self,
            "source_receiver_axis_id",
            source_receiver_axis_id,
        )
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "full_pipeline_resampling",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )
        object.__setattr__(
            self,
            "execution_metadata_id",
            stable_id(
                "full_pipeline_resampling_execution",
                {
                    "effective_n_jobs": effective_n_jobs,
                    "execution_backend": self.execution_backend,
                    "requested_n_jobs": requested_n_jobs,
                    "result_id": self.result_id,
                    "shared_snapshot_policy": _SHARED_SNAPSHOT_POLICY,
                },
                schema_version="1",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "exchangeability_id": self.exchangeability.exchangeability_id,
            "plan_ids": [_plan_id(plan) for plan in self.plans],
            "record_ids": [record.record_id for record in self.records],
            "source_input_identity_id": self.source_input_identity_id,
            "source_input_digest": self.source_input_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "config_digest": self.config_digest,
            "crossfit_spec_id": self.crossfit_spec_id,
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "target_prior_content_id": self.target_prior_content_id,
            "source_receiver_universe_id": self.source_receiver_universe_id,
            "source_receiver_axis_id": self.source_receiver_axis_id,
            "root_seed_lineage": self.root_seed_lineage.to_dict(),
            "retain_children": self.retain_children,
            "rerun_policy_id": _RERUN_POLICY_ID,
            "formal_inference_status": _INFERENCE_STATUS,
        }

    def _require_intact(self) -> None:
        try:
            replace(self.exchangeability)
            for plan in self.plans:
                replace(plan)
            for record in self.records:
                record._require_intact()
                if record.child is not None:
                    record.child._require_intact()
            repeated = FullPipelineResamplingResult(
                exchangeability=self.exchangeability,
                plans=self.plans,
                records=self.records,
                source_input_identity_id=self.source_input_identity_id,
                source_input_digest=self.source_input_digest,
                source_snapshot_id=self.source_snapshot_id,
                config_digest=self.config_digest,
                crossfit_spec_id=self.crossfit_spec_id,
                resource_bundle_content_id=self.resource_bundle_content_id,
                target_prior_content_id=self.target_prior_content_id,
                root_seed_lineage=self.root_seed_lineage,
                retain_children=self.retain_children,
                requested_n_jobs=self.requested_n_jobs,
                effective_n_jobs=self.effective_n_jobs,
                source_receiver_universe_id=self.source_receiver_universe_id,
                source_receiver_axis_id=self.source_receiver_axis_id,
                _source_receiver_universe=self._source_receiver_universe,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.result_id == repeated.result_id
                and self.execution_metadata_id == repeated.execution_metadata_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Full-pipeline resampling result failed integrity validation",
                code="full_pipeline_resampling_integrity_violation",
                field="result_id",
                remediation="Rerun the exact frozen full-pipeline resampling plan",
            ) from error
        if not valid:
            raise ContractError(
                "Full-pipeline resampling result failed integrity validation",
                code="full_pipeline_resampling_integrity_violation",
                field="result_id",
                remediation="Rerun the exact frozen full-pipeline resampling plan",
            )

    @property
    def status(self) -> FullPipelineResamplingStatus:
        succeeded = sum(
            record.status is FullPipelineResampleStatus.SUCCEEDED
            for record in self.records
        )
        if succeeded == len(self.records):
            return FullPipelineResamplingStatus.SUCCEEDED
        if succeeded:
            return FullPipelineResamplingStatus.PARTIAL_FAILURE
        return FullPipelineResamplingStatus.FAILED

    @property
    def children(self) -> tuple[CrossFitArtifacts, ...]:
        """Return only explicitly retained successful cross-fit children."""

        return tuple(
            record.child for record in self.records if record.child is not None
        )

    @property
    def execution_backend(self) -> str:
        """Return the runtime backend without changing scientific identity."""

        return _SERIAL_BACKEND if self.effective_n_jobs == 1 else _THREAD_BACKEND

    def to_manifest(self) -> dict[str, object]:
        """Return lineage and release boundaries without embedding large children."""

        self._require_intact()
        return {
            "schema_version": _SCHEMA_VERSION,
            "result_id": self.result_id,
            "status": self.status.value,
            "exchangeability": self.exchangeability.to_dict(),
            "source_input_identity_id": self.source_input_identity_id,
            "source_input_digest": self.source_input_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "config_digest": self.config_digest,
            "crossfit_spec_id": self.crossfit_spec_id,
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "target_prior_content_id": self.target_prior_content_id,
            "source_receiver_universe_id": self.source_receiver_universe_id,
            "source_receiver_axis_id": self.source_receiver_axis_id,
            "source_receiver_ids": (
                []
                if self._source_receiver_universe is None
                else list(self._source_receiver_universe.receiver_ids)
            ),
            "source_receiver_universe": (
                None
                if self._source_receiver_universe is None
                else self._source_receiver_universe.to_dict()
            ),
            "root_seed_lineage": self.root_seed_lineage.to_dict(),
            "execution_backend": self.execution_backend,
            "execution_metadata_id": self.execution_metadata_id,
            "requested_n_jobs": self.requested_n_jobs,
            "effective_n_jobs": self.effective_n_jobs,
            "shared_snapshot_policy": _SHARED_SNAPSHOT_POLICY,
            "memory_budget_note": (
                "peak working memory may approach effective_n_jobs times one "
                "fold-local full-pipeline fit"
            ),
            "materialization_policy": _MATERIALIZATION_POLICY,
            "rerun_policy_id": _RERUN_POLICY_ID,
            "full_pipeline_refit_per_resample": True,
            "score_table_relabeling_only": False,
            "rerun_stages": list(_RERUN_STAGES),
            "retain_children": self.retain_children,
            "child_retention_policy": (
                "retain_successful_crossfit_children_v1"
                if self.retain_children
                else "discard_child_after_recording_crossfit_id_v1"
            ),
            "formal_inference_status": _INFERENCE_STATUS,
            "is_inference_eligible": False,
            "inferential_fields_available": [],
            "inferential_fields_unavailable": list(_INFERENTIAL_FIELDS_UNAVAILABLE),
            "plans": [plan.to_dict() for plan in self.plans],
            "records": [record.to_dict() for record in self.records],
        }


def aligned_full_pipeline_resamples(
    result: FullPipelineResamplingResult,
    *,
    operation: FullPipelineResamplingOperation | None = None,
) -> tuple[tuple[ResamplingPlan, FullPipelineResampleRecord], ...]:
    """Return the exact intact plan/record manifest, optionally by operation."""

    if not isinstance(result, FullPipelineResamplingResult):
        raise TypeError("result must be a FullPipelineResamplingResult")
    result._require_intact()
    normalized = (
        None if operation is None else FullPipelineResamplingOperation(operation)
    )
    return tuple(
        (plan, record)
        for plan, record in zip(result.plans, result.records, strict=True)
        if normalized is None or record.operation is normalized
    )


@dataclass(frozen=True, slots=True)
class _MaterializedResample:
    adata: AnnData = field(repr=False)
    input_id: str
    n_obs: int
    n_samples: int
    n_subjects: int


def _materialized_input_id(
    *,
    source_snapshot_id: str,
    plan: ResamplingPlan,
) -> str:
    identifier: str = stable_id(
        "materialized_full_pipeline_resample",
        {
            "materialization_policy": _MATERIALIZATION_POLICY,
            "operation": _plan_operation(plan).value,
            "plan_id": _plan_id(plan),
            "source_snapshot_id": source_snapshot_id,
        },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


def _validated_materialized(
    adata: AnnData,
    config: CrychicConfig,
    *,
    input_id: str,
) -> _MaterializedResample:
    validated = validate_anndata(adata, _input_schema(config))
    if not validated.is_counts:
        raise ValueError("full-pipeline resampling requires raw counts")
    return _MaterializedResample(
        adata=adata,
        input_id=input_id,
        n_obs=int(adata.n_obs),
        n_samples=validated.report.n_samples,
        n_subjects=validated.report.n_subjects,
    )


def _materialize_subject_bootstrap(
    source: AnnData,
    config: CrychicConfig,
    plan: SubjectBootstrapPlan,
    *,
    source_snapshot_id: str,
) -> _MaterializedResample:
    """Copy complete subject blocks and assign unique subject/sample/cell IDs."""

    source_subjects = source.obs[config.subject_key].astype(str).to_numpy()
    source_samples = source.obs[config.sample_key].astype(str).to_numpy()
    positions: list[int] = []
    new_subject_ids: list[str] = []
    new_sample_ids: list[str] = []
    new_cell_ids: list[str] = []
    for draw in plan.draws:
        block = np.flatnonzero(source_subjects == draw.source_subject_id)
        if not len(block):
            raise ContractError(
                "Bootstrap source subject is absent from the sanitized input",
                code="bootstrap_subject_block_missing",
                field="source_subject_id",
                remediation="Apply the plan to its exact exchangeability input",
            )
        block_samples = tuple(sorted(set(source_samples[block])))
        sample_mapping = {
            sample_id: f"{draw.bootstrap_subject_id}__sample_{index:06d}"
            for index, sample_id in enumerate(block_samples)
        }
        for cell_index, position in enumerate(block):
            positions.append(int(position))
            new_subject_ids.append(draw.bootstrap_subject_id)
            new_sample_ids.append(sample_mapping[source_samples[position]])
            new_cell_ids.append(f"{draw.bootstrap_subject_id}__cell_{cell_index:08d}")
    # Duplicate source rows are intentional until deterministic bootstrap IDs
    # are installed immediately below.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Observation names are not unique.*",
            category=UserWarning,
        )
        result = source[positions, :].copy()
    result.obs[config.subject_key] = np.asarray(new_subject_ids, dtype=object)
    result.obs[config.sample_key] = np.asarray(new_sample_ids, dtype=object)
    result.obs_names = new_cell_ids
    if (
        not result.obs_names.is_unique
        or result.obs[config.subject_key].nunique() != len(plan.draws)
        or result.obs[config.sample_key].astype(str).duplicated().all()
    ):
        raise RuntimeError("bootstrap identifier materialization is not unique")
    return _validated_materialized(
        result,
        config,
        input_id=_materialized_input_id(
            source_snapshot_id=source_snapshot_id,
            plan=plan,
        ),
    )


def _materialize_context_permutation(
    source: AnnData,
    config: CrychicConfig,
    exchangeability: ExchangeabilityMap,
    plan: ContextPermutationPlan,
    *,
    source_snapshot_id: str,
) -> _MaterializedResample:
    """Apply one sample-block context map to every cell in a defensive copy."""

    result = source.copy()
    result.obs = apply_context_permutation(result.obs, exchangeability, plan)
    return _validated_materialized(
        result,
        config,
        input_id=_materialized_input_id(
            source_snapshot_id=source_snapshot_id,
            plan=plan,
        ),
    )


def _failure_metadata(error: Exception) -> tuple[str, str]:
    failure_type = f"{type(error).__module__}.{type(error).__qualname__}"
    if isinstance(error, CrychicError):
        return failure_type, error.details.code
    return failure_type, type(error).__qualname__


def _execute_plan(
    source: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    spec: CrossFitSpec,
    exchangeability: ExchangeabilityMap,
    plan: ResamplingPlan,
    source_receiver_universe: FrozenReceiverUniverse,
    *,
    source_snapshot_id: str,
    retain_child: bool,
) -> FullPipelineResampleRecord:
    operation = _plan_operation(plan)
    plan_id = _plan_id(plan)
    input_id = _materialized_input_id(
        source_snapshot_id=source_snapshot_id,
        plan=plan,
    )
    crossfit_seed = plan.seed_lineage.derive(
        "full_pipeline_crossfit",
        plan_id,
    )
    child_config = replace(config, random_seed=crossfit_seed.seed)
    materialized: _MaterializedResample | None = None
    try:
        if isinstance(plan, SubjectBootstrapPlan):
            materialized = _materialize_subject_bootstrap(
                source,
                child_config,
                plan,
                source_snapshot_id=source_snapshot_id,
            )
        else:
            materialized = _materialize_context_permutation(
                source,
                child_config,
                exchangeability,
                plan,
                source_snapshot_id=source_snapshot_id,
            )
        child_snapshot = _sanitized_raw_input_snapshot(
            materialized.adata,
            child_config,
        )
        child = _run_subject_crossfit(
            child_snapshot,
            child_config,
            resource_bundle,
            target_prior,
            spec=spec,
            _receiver_axis_source=source_receiver_universe,
        )
        if not isinstance(child, CrossFitArtifacts):
            raise TypeError("full-pipeline runner must return CrossFitArtifacts")
        reuse_binding = bind_reused_receiver_universe(
            source_receiver_universe,
            child.receiver_universe,
            operation=operation.value,
            plan_id=plan_id,
            resample_index=plan.resample_index,
            materialized_input_id=input_id,
        )
        return FullPipelineResampleRecord(
            operation=operation,
            resample_index=plan.resample_index,
            plan_id=plan_id,
            exchangeability_id=exchangeability.exchangeability_id,
            materialized_input_id=input_id,
            plan_seed_lineage=plan.seed_lineage,
            crossfit_seed_lineage=crossfit_seed,
            crossfit_config_digest=child_config.digest,
            status=FullPipelineResampleStatus.SUCCEEDED,
            crossfit_id=child.crossfit_id,
            failure_type=None,
            failure_code=None,
            n_obs=materialized.n_obs,
            n_samples=materialized.n_samples,
            n_subjects=materialized.n_subjects,
            source_receiver_universe_id=source_receiver_universe.universe_id,
            source_receiver_axis_id=source_receiver_universe.receiver_axis_id,
            child_receiver_universe_id=child.receiver_universe.universe_id,
            child_receiver_axis_id=child.receiver_universe.receiver_axis_id,
            receiver_universe_reuse_binding_id=reuse_binding.binding_id,
            _child=child if retain_child else None,
            _receiver_universe_reuse_binding=reuse_binding,
        )
    except Exception as error:
        failure_type, failure_code = _failure_metadata(error)
        return FullPipelineResampleRecord(
            operation=operation,
            resample_index=plan.resample_index,
            plan_id=plan_id,
            exchangeability_id=exchangeability.exchangeability_id,
            materialized_input_id=input_id,
            plan_seed_lineage=plan.seed_lineage,
            crossfit_seed_lineage=crossfit_seed,
            crossfit_config_digest=child_config.digest,
            status=FullPipelineResampleStatus.FAILED,
            crossfit_id=None,
            failure_type=failure_type,
            failure_code=failure_code,
            n_obs=None if materialized is None else materialized.n_obs,
            n_samples=None if materialized is None else materialized.n_samples,
            n_subjects=None if materialized is None else materialized.n_subjects,
            source_receiver_universe_id=source_receiver_universe.universe_id,
            source_receiver_axis_id=source_receiver_universe.receiver_axis_id,
        )


def _resample_count(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_jobs(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer >= 1")
    return value


def run_full_pipeline_resampling(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    spec: CrossFitSpec,
    n_bootstraps: int = 0,
    n_permutations: int = 0,
    strata_keys: Sequence[str] | None = None,
    immutable_covariates: Sequence[str] | None = None,
    seed_lineage: SeedLineage | None = None,
    retain_children: bool = False,
    n_jobs: int = 1,
) -> FullPipelineResamplingResult:
    """Materialize each legal subject resample and rerun complete cross-fit.

    ``n_jobs`` is bounded by the number of plans. Workers share the immutable
    sanitized source snapshot, but each materializes its own AnnData and creates
    fold-local matrices and models. Results retain plan order regardless of
    completion order. Size ``n_jobs`` from an external memory budget and cap
    BLAS/OpenMP threads separately.
    """

    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig")
    if not isinstance(spec, CrossFitSpec):
        raise TypeError("spec must be a CrossFitSpec")
    spec._require_intact()
    bootstraps = _resample_count(n_bootstraps, field_name="n_bootstraps")
    permutations = _resample_count(n_permutations, field_name="n_permutations")
    if bootstraps + permutations < 1:
        raise ValueError("request at least one bootstrap or permutation")
    requested_jobs = _positive_jobs(n_jobs, field_name="n_jobs")
    effective_jobs = min(requested_jobs, bootstraps + permutations)
    if not isinstance(retain_children, bool):
        raise TypeError("retain_children must be boolean")
    snapshot = _sanitized_raw_input_snapshot(adata, config)
    snapshot._require_intact()
    observed_root_cell_types = tuple(
        sorted(snapshot.adata.obs[config.cell_type_key].astype(str).unique())
    )
    source_receiver_universe = freeze_receiver_universe(
        snapshot.identity,
        observed_root_cell_types,
        predeclared_receiver_ids=spec.predeclared_receiver_ids,
    )
    validated = validate_anndata(snapshot.adata, _input_schema(config))
    resolved_strata = tuple(spec.strata_keys if strata_keys is None else strata_keys)
    resolved_immutable = tuple(
        resolved_strata if immutable_covariates is None else immutable_covariates
    )
    exchangeability = build_exchangeability_map(
        validated.report.sample_metadata,
        sample_key=config.sample_key,
        subject_key=config.subject_key,
        context_keys=config.context_keys,
        strata_keys=resolved_strata,
        immutable_covariates=resolved_immutable,
    )
    lineage = (
        SeedLineage(config.random_seed).derive(
            "full_pipeline_resampling",
            exchangeability.exchangeability_id,
            spec.spec_id,
        )
        if seed_lineage is None
        else seed_lineage
    )
    if not isinstance(lineage, SeedLineage):
        raise TypeError("seed_lineage must be a SeedLineage or None")
    plans: tuple[ResamplingPlan, ...] = (
        plan_subject_bootstraps(
            exchangeability,
            n_bootstraps=bootstraps,
            seed_lineage=lineage,
        )
        if bootstraps
        else ()
    ) + (
        plan_context_permutations(
            exchangeability,
            n_permutations=permutations,
            seed_lineage=lineage,
        )
        if permutations
        else ()
    )
    execute = partial(
        _execute_plan,
        snapshot.adata,
        config,
        resource_bundle,
        target_prior,
        spec,
        exchangeability,
        source_receiver_universe=source_receiver_universe,
        source_snapshot_id=snapshot.snapshot_id,
        retain_child=retain_children,
    )
    if effective_jobs == 1:
        records = tuple(execute(plan=plan) for plan in plans)
    else:
        with ThreadPoolExecutor(
            max_workers=effective_jobs,
            thread_name_prefix="crychic-full-resample",
        ) as executor:
            records = tuple(executor.map(execute, plans))
    return FullPipelineResamplingResult(
        exchangeability=exchangeability,
        plans=plans,
        records=records,
        source_input_identity_id=snapshot.identity.identity_id,
        source_input_digest=snapshot.identity.input_digest,
        source_snapshot_id=snapshot.snapshot_id,
        config_digest=config.digest,
        crossfit_spec_id=spec.spec_id,
        resource_bundle_content_id=_resource_bundle_content_id(resource_bundle),
        target_prior_content_id=_target_prior_content_id(target_prior),
        root_seed_lineage=lineage,
        retain_children=retain_children,
        requested_n_jobs=requested_jobs,
        effective_n_jobs=effective_jobs,
        source_receiver_universe_id=source_receiver_universe.universe_id,
        source_receiver_axis_id=source_receiver_universe.receiver_axis_id,
        _source_receiver_universe=source_receiver_universe,
    )


__all__ = [
    "FullPipelineResampleRecord",
    "FullPipelineResampleStatus",
    "FullPipelineResamplingOperation",
    "FullPipelineResamplingResult",
    "FullPipelineResamplingStatus",
    "aligned_full_pipeline_resamples",
    "run_full_pipeline_resampling",
]
