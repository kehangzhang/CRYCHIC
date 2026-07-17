"""End-to-end composition for ADR-013 active-edge probabilities.

This workflow owns orchestration only.  Point scoring, active-null planning,
full-pipeline reruns, and local-FDR estimation remain in their owning modules.
The returned artifact binds every producer-owned child by stable identity and
never accepts a caller-controlled probability-release flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anndata import AnnData

from crychic.core import ContractError, CrychicConfig, SeedLineage, stable_id
from crychic.inference import (
    ActiveProbabilityCollection,
    ActiveProbabilitySpec,
    G3PCalibrationGate,
    G3PGateStatus,
    estimate_active_probabilities,
)
from crychic.resampling import ActiveNullSpec
from crychic.resources import ResourceBundle, TargetPrior
from crychic.scoring import FrozenActiveEdgeUniverse

from .active_edge_scores import (
    CrossFitActiveEdgePointRecords,
    adapt_crossfit_active_edge_point_records,
    freeze_crossfit_active_edge_universe,
)
from .active_null_rerun import ActiveNullRerunResult, run_active_null_reruns
from .crossfit import CrossFitArtifacts, CrossFitSpec, run_subject_crossfit

_SCHEMA_VERSION = "1.0.0"
_LINEAGE_PRODUCER = "crychic.workflow.active_probability_pipeline_lineage.v1"
_RESULT_PRODUCER = "crychic.workflow.active_probability_pipeline_result.v1"
_COMPOSITION_POLICY_ID = "point_crossfit_freeze_adapt_full_active_null_bum_g3p_gate_v1"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_name(value: object, *, field_name: str) -> str | None:
    return None if value is None else _name(value, field_name=field_name)


def _positive_count(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer >= 1")
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


@dataclass(frozen=True, slots=True, init=False)
class ActiveProbabilityPipelineLineage:
    """Stable parent chain for one composed active-probability workflow."""

    source_input_identity_id: str
    source_input_digest: str
    source_snapshot_id: str
    config_digest: str
    crossfit_spec_id: str
    repeat_id: str
    fold_plan_id: str
    point_crossfit_id: str
    point_oof_audit_id: str
    point_registry_id: str
    resource_bundle_content_id: str
    source_target_prior_content_id: str
    contrast_id: str
    candidate_universe_id: str
    score_version: str
    score_spec_id: str
    source_score_collection_id: str
    point_collection_id: str
    active_null_spec_id: str
    crossfit_seed_lineage: SeedLineage
    partition_seed_lineage: SeedLineage | None
    active_null_root_seed_lineage: SeedLineage
    active_null_id: str
    active_null_result_id: str
    null_distribution_id: str
    plan_ids: tuple[str, ...]
    rerun_record_ids: tuple[str, ...]
    active_probability_spec_id: str
    estimator_id: str
    stratum_policy_id: str
    calibration_gate_id: str | None
    calibration_gate_status: G3PGateStatus | None
    calibration_evidence_id: str | None
    calibration_source_artifact_id: str | None
    calibration_protocol_id: str | None
    probability_collection_id: str
    retain_null_children: bool
    lineage_id: str
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ActiveProbabilityPipelineLineage is producer-owned; use "
            "run_active_probability_pipeline()"
        )

    @classmethod
    def _from_components(
        cls,
        *,
        point_artifacts: CrossFitArtifacts,
        universe: FrozenActiveEdgeUniverse,
        point_records: CrossFitActiveEdgePointRecords,
        null_rerun: ActiveNullRerunResult,
        active_probability_spec: ActiveProbabilitySpec,
        calibration_gate: G3PCalibrationGate | None,
        probability_collection: ActiveProbabilityCollection,
    ) -> ActiveProbabilityPipelineLineage:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "source_input_identity_id": null_rerun.source_input_identity_id,
            "source_input_digest": null_rerun.source_input_digest,
            "source_snapshot_id": null_rerun.source_snapshot_id,
            "config_digest": null_rerun.config_digest,
            "crossfit_spec_id": null_rerun.crossfit_spec_id,
            "repeat_id": null_rerun.repeat_id,
            "fold_plan_id": null_rerun.fold_plan_id,
            "point_crossfit_id": point_artifacts.crossfit_id,
            "point_oof_audit_id": point_records.source_oof_audit_id,
            "point_registry_id": point_records.source_registry_id,
            "resource_bundle_content_id": null_rerun.resource_bundle_content_id,
            "source_target_prior_content_id": (
                null_rerun.source_target_prior_content_id
            ),
            "contrast_id": universe.contrast_id,
            "candidate_universe_id": universe.universe_id,
            "score_version": point_records.score_version,
            "score_spec_id": point_records.score_spec_id,
            "source_score_collection_id": (point_records.source_score_collection_id),
            "point_collection_id": point_records.collection_id,
            "active_null_spec_id": null_rerun.active_null_spec_id,
            "crossfit_seed_lineage": null_rerun.crossfit_seed_lineage,
            "partition_seed_lineage": null_rerun.partition_seed_lineage,
            "active_null_root_seed_lineage": null_rerun.root_seed_lineage,
            "active_null_id": null_rerun.active_null_id,
            "active_null_result_id": null_rerun.result_id,
            "null_distribution_id": null_rerun.distribution.distribution_id,
            "plan_ids": tuple(request.plan_id for request in null_rerun.requests),
            "rerun_record_ids": tuple(
                record.null_rerun_record_id for record in null_rerun.records
            ),
            "active_probability_spec_id": active_probability_spec.spec_id,
            "estimator_id": active_probability_spec.estimator_id,
            "stratum_policy_id": active_probability_spec.stratum_policy_id,
            "calibration_gate_id": (
                None if calibration_gate is None else calibration_gate.gate_id
            ),
            "calibration_gate_status": (
                None if calibration_gate is None else calibration_gate.status
            ),
            "calibration_evidence_id": (
                None if calibration_gate is None else calibration_gate.evidence_id
            ),
            "calibration_source_artifact_id": (
                None
                if calibration_gate is None
                else calibration_gate.source_artifact_id
            ),
            "calibration_protocol_id": (
                None if calibration_gate is None else calibration_gate.protocol_id
            ),
            "probability_collection_id": probability_collection.collection_id,
            "retain_null_children": null_rerun.retain_children,
            "_producer_marker": _LINEAGE_PRODUCER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self._validate_state()
        object.__setattr__(
            self,
            "lineage_id",
            stable_id(
                "active_probability_pipeline_lineage",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    def _validate_state(self) -> None:
        required_names = (
            "source_input_identity_id",
            "source_input_digest",
            "source_snapshot_id",
            "config_digest",
            "crossfit_spec_id",
            "repeat_id",
            "fold_plan_id",
            "point_crossfit_id",
            "point_oof_audit_id",
            "point_registry_id",
            "resource_bundle_content_id",
            "source_target_prior_content_id",
            "contrast_id",
            "candidate_universe_id",
            "score_version",
            "score_spec_id",
            "source_score_collection_id",
            "point_collection_id",
            "active_null_spec_id",
            "active_null_id",
            "active_null_result_id",
            "null_distribution_id",
            "active_probability_spec_id",
            "estimator_id",
            "stratum_policy_id",
            "probability_collection_id",
        )
        for name in required_names:
            _name(getattr(self, name), field_name=name)
        plans = tuple(_name(item, field_name="plan_id") for item in self.plan_ids)
        reruns = tuple(
            _name(item, field_name="null_rerun_record_id")
            for item in self.rerun_record_ids
        )
        if not plans or len(plans) != len(reruns):
            raise ValueError("lineage must bind one rerun record per null plan")
        if len(plans) != len(set(plans)) or len(reruns) != len(set(reruns)):
            raise ValueError("lineage null plan and rerun identities must be unique")
        gate_names = (
            self.calibration_gate_id,
            self.calibration_evidence_id,
            self.calibration_source_artifact_id,
            self.calibration_protocol_id,
        )
        if self.calibration_gate_id is None:
            if self.calibration_gate_status is not None or any(
                value is not None for value in gate_names
            ):
                raise ValueError("absent calibration gate cannot carry gate lineage")
        else:
            for field_name, value in zip(
                (
                    "calibration_gate_id",
                    "calibration_evidence_id",
                    "calibration_source_artifact_id",
                    "calibration_protocol_id",
                ),
                gate_names,
                strict=True,
            ):
                _optional_name(value, field_name=field_name)
                if value is None:
                    raise ValueError("present calibration gate requires full lineage")
            if not isinstance(self.calibration_gate_status, G3PGateStatus):
                raise TypeError("calibration_gate_status must be G3PGateStatus")
        if not isinstance(self.retain_null_children, bool):
            raise TypeError("retain_null_children must be boolean")
        if not isinstance(self.crossfit_seed_lineage, SeedLineage):
            raise TypeError("crossfit_seed_lineage must be a SeedLineage")
        if self.partition_seed_lineage is not None and not isinstance(
            self.partition_seed_lineage, SeedLineage
        ):
            raise TypeError("partition_seed_lineage must be SeedLineage or None")
        if not isinstance(self.active_null_root_seed_lineage, SeedLineage):
            raise TypeError("active_null_root_seed_lineage must be a SeedLineage")
        if self._producer_marker != _LINEAGE_PRODUCER:
            raise TypeError("active-probability pipeline lineage is not producer-owned")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "source_input_identity_id": self.source_input_identity_id,
            "source_input_digest": self.source_input_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "config_digest": self.config_digest,
            "crossfit_spec_id": self.crossfit_spec_id,
            "repeat_id": self.repeat_id,
            "fold_plan_id": self.fold_plan_id,
            "point_crossfit_id": self.point_crossfit_id,
            "point_oof_audit_id": self.point_oof_audit_id,
            "point_registry_id": self.point_registry_id,
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "source_target_prior_content_id": self.source_target_prior_content_id,
            "contrast_id": self.contrast_id,
            "candidate_universe_id": self.candidate_universe_id,
            "score_version": self.score_version,
            "score_spec_id": self.score_spec_id,
            "source_score_collection_id": self.source_score_collection_id,
            "point_collection_id": self.point_collection_id,
            "active_null_spec_id": self.active_null_spec_id,
            "crossfit_seed_lineage": self.crossfit_seed_lineage.to_dict(),
            "partition_seed_lineage": (
                None
                if self.partition_seed_lineage is None
                else self.partition_seed_lineage.to_dict()
            ),
            "active_null_root_seed_lineage": (
                self.active_null_root_seed_lineage.to_dict()
            ),
            "active_null_id": self.active_null_id,
            "active_null_result_id": self.active_null_result_id,
            "null_distribution_id": self.null_distribution_id,
            "plan_ids": list(self.plan_ids),
            "rerun_record_ids": list(self.rerun_record_ids),
            "active_probability_spec_id": self.active_probability_spec_id,
            "estimator_id": self.estimator_id,
            "stratum_policy_id": self.stratum_policy_id,
            "calibration_gate_id": self.calibration_gate_id,
            "calibration_gate_status": (
                None
                if self.calibration_gate_status is None
                else self.calibration_gate_status.value
            ),
            "calibration_evidence_id": self.calibration_evidence_id,
            "calibration_source_artifact_id": (self.calibration_source_artifact_id),
            "calibration_protocol_id": self.calibration_protocol_id,
            "probability_collection_id": self.probability_collection_id,
            "retain_null_children": self.retain_null_children,
            "composition_policy_id": _COMPOSITION_POLICY_ID,
            "producer_marker": _LINEAGE_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            self._validate_state()
            expected = stable_id(
                "active_probability_pipeline_lineage",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = self.lineage_id == expected
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "Active-probability pipeline lineage failed integrity validation",
                code="active_probability_pipeline_lineage_integrity_violation",
                field="lineage_id",
                remediation="Rerun the exact composed active-probability workflow",
            ) from error
        if not valid:
            raise _contract_error(
                "Active-probability pipeline lineage failed integrity validation",
                code="active_probability_pipeline_lineage_integrity_violation",
                field="lineage_id",
                remediation="Rerun the exact composed active-probability workflow",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete stable parent chain."""

        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"lineage_id": self.lineage_id, **payload}


@dataclass(frozen=True, slots=True, init=False)
class ActiveProbabilityPipelineResult:
    """Producer-owned result of the complete ADR-013 runtime composition."""

    point_artifacts: CrossFitArtifacts = field(repr=False)
    universe: FrozenActiveEdgeUniverse
    point_records: CrossFitActiveEdgePointRecords
    null_rerun: ActiveNullRerunResult
    active_null_spec: ActiveNullSpec
    active_probability_spec: ActiveProbabilitySpec
    calibration_gate: G3PCalibrationGate | None
    probability_collection: ActiveProbabilityCollection
    lineage: ActiveProbabilityPipelineLineage
    pipeline_id: str
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ActiveProbabilityPipelineResult is producer-owned; use "
            "run_active_probability_pipeline()"
        )

    @classmethod
    def _from_workflow(
        cls,
        *,
        point_artifacts: CrossFitArtifacts,
        universe: FrozenActiveEdgeUniverse,
        point_records: CrossFitActiveEdgePointRecords,
        null_rerun: ActiveNullRerunResult,
        active_null_spec: ActiveNullSpec,
        active_probability_spec: ActiveProbabilitySpec,
        calibration_gate: G3PCalibrationGate | None,
        probability_collection: ActiveProbabilityCollection,
    ) -> ActiveProbabilityPipelineResult:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "point_artifacts": point_artifacts,
            "universe": universe,
            "point_records": point_records,
            "null_rerun": null_rerun,
            "active_null_spec": active_null_spec,
            "active_probability_spec": active_probability_spec,
            "calibration_gate": calibration_gate,
            "probability_collection": probability_collection,
            "lineage": ActiveProbabilityPipelineLineage._from_components(
                point_artifacts=point_artifacts,
                universe=universe,
                point_records=point_records,
                null_rerun=null_rerun,
                active_probability_spec=active_probability_spec,
                calibration_gate=calibration_gate,
                probability_collection=probability_collection,
            ),
            "_producer_marker": _RESULT_PRODUCER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self._validate_state()
        object.__setattr__(
            self,
            "pipeline_id",
            stable_id(
                "active_probability_pipeline_result",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    @property
    def comm_probability_release_allowed(self) -> bool:
        """Return the producer-gated release state; callers cannot set it."""

        return bool(self.probability_collection.comm_probability_release_allowed)

    def _validate_state(self) -> None:
        expected_types = (
            (self.point_artifacts, CrossFitArtifacts, "point_artifacts"),
            (self.universe, FrozenActiveEdgeUniverse, "universe"),
            (
                self.point_records,
                CrossFitActiveEdgePointRecords,
                "point_records",
            ),
            (self.null_rerun, ActiveNullRerunResult, "null_rerun"),
            (self.active_null_spec, ActiveNullSpec, "active_null_spec"),
            (
                self.active_probability_spec,
                ActiveProbabilitySpec,
                "active_probability_spec",
            ),
            (
                self.probability_collection,
                ActiveProbabilityCollection,
                "probability_collection",
            ),
            (
                self.lineage,
                ActiveProbabilityPipelineLineage,
                "lineage",
            ),
        )
        for value, expected, field_name in expected_types:
            if not isinstance(value, expected):
                raise TypeError(f"{field_name} must be {expected.__name__}")
        if self.calibration_gate is not None and not isinstance(
            self.calibration_gate, G3PCalibrationGate
        ):
            raise TypeError("calibration_gate must be G3PCalibrationGate or None")
        self.point_artifacts._require_intact()
        self.universe._require_intact()
        self.point_records._require_intact()
        self.null_rerun._require_intact()
        self.active_probability_spec._require_intact()
        if self.active_null_spec.to_dict() != ActiveNullSpec().to_dict():
            raise ValueError("active_null_spec is not the intact ADR-013 spec")
        if self.calibration_gate is not None:
            self.calibration_gate._require_intact()
        self.probability_collection._require_intact()
        self.lineage._require_intact()

        point_ids = tuple(
            record.candidate_edge_id for record in self.point_records.records
        )
        distribution_point_ids = tuple(
            record.record_id for record in self.null_rerun.distribution.point_records
        )
        if (
            self.point_records.source_crossfit_id != self.point_artifacts.crossfit_id
            or self.point_records.source_registry_id
            != self.point_artifacts.receiver_scoring_registry_id
            or self.point_records.active_edge_universe_id != self.universe.universe_id
            or self.point_records.contrast_id != self.universe.contrast_id
            or self.point_records.score_version != self.universe.score_version
            or point_ids != self.universe.candidate_edge_ids
        ):
            raise ValueError("point cross-fit, universe, and adapter are misbound")
        if (
            self.null_rerun.source_point_crossfit_id != self.point_artifacts.crossfit_id
            or self.null_rerun.source_point_collection_id
            != self.point_records.collection_id
            or self.null_rerun.source_point_oof_audit_id
            != self.point_records.source_oof_audit_id
            or self.null_rerun.source_point_registry_id
            != self.point_records.source_registry_id
            or self.null_rerun.source_point_score_collection_id
            != self.point_records.source_score_collection_id
            or self.null_rerun.candidate_universe_id != self.universe.universe_id
            or self.null_rerun.score_version != self.point_records.score_version
            or self.null_rerun.score_spec_id != self.point_records.score_spec_id
            or self.null_rerun.active_null_spec_id
            != self.active_null_spec.active_null_spec_id
            or self.null_rerun.source_input_identity_id
            != self.point_artifacts.root_input_identity.identity_id
            or self.null_rerun.source_input_digest
            != self.point_artifacts.root_input_identity.input_digest
            or self.null_rerun.config_digest
            != self.point_artifacts.root_input_identity.config_digest
            or self.null_rerun.crossfit_spec_id != self.point_artifacts.spec.spec_id
            or self.null_rerun.repeat_id != self.point_artifacts.spec.repeat_id
            or self.null_rerun.repeat_id != self.point_records.repeat_id
            or self.null_rerun.fold_plan_id != self.point_artifacts.fold_plan.plan_id
            or distribution_point_ids
            != tuple(record.record_id for record in self.point_records.records)
        ):
            raise ValueError("point and active-null artifacts are misbound")
        expected_gate_id = (
            None if self.calibration_gate is None else self.calibration_gate.gate_id
        )
        expected_gate_status = (
            None if self.calibration_gate is None else self.calibration_gate.status
        )
        if (
            self.probability_collection.distribution_id
            != self.null_rerun.distribution.distribution_id
            or self.probability_collection.active_null_id
            != self.null_rerun.active_null_id
            or self.probability_collection.candidate_universe_id
            != self.universe.universe_id
            or self.probability_collection.score_spec_id
            != self.point_records.score_spec_id
            or self.probability_collection.active_probability_spec_id
            != self.active_probability_spec.spec_id
            or self.probability_collection.calibration_gate_id != expected_gate_id
            or self.probability_collection.calibration_gate_status
            != expected_gate_status
        ):
            raise ValueError("active-null, probability spec, and gate are misbound")
        expected_lineage = ActiveProbabilityPipelineLineage._from_components(
            point_artifacts=self.point_artifacts,
            universe=self.universe,
            point_records=self.point_records,
            null_rerun=self.null_rerun,
            active_probability_spec=self.active_probability_spec,
            calibration_gate=self.calibration_gate,
            probability_collection=self.probability_collection,
        )
        if self.lineage.lineage_id != expected_lineage.lineage_id:
            raise ValueError("pipeline lineage does not bind the exact child artifacts")
        if self._producer_marker != _RESULT_PRODUCER:
            raise TypeError("active-probability pipeline result is not producer-owned")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage.lineage_id,
            "point_crossfit_id": self.point_artifacts.crossfit_id,
            "candidate_universe_id": self.universe.universe_id,
            "point_collection_id": self.point_records.collection_id,
            "active_null_result_id": self.null_rerun.result_id,
            "probability_collection_id": self.probability_collection.collection_id,
            "composition_policy_id": _COMPOSITION_POLICY_ID,
            "producer_marker": _RESULT_PRODUCER,
        }

    def _require_intact(self) -> None:
        try:
            self._validate_state()
            expected = stable_id(
                "active_probability_pipeline_result",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = self.pipeline_id == expected
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "Active-probability pipeline result failed integrity validation",
                code="active_probability_pipeline_integrity_violation",
                field="pipeline_id",
                remediation="Rerun the exact composed active-probability workflow",
            ) from error
        if not valid:
            raise _contract_error(
                "Active-probability pipeline result failed integrity validation",
                code="active_probability_pipeline_integrity_violation",
                field="pipeline_id",
                remediation="Rerun the exact composed active-probability workflow",
            )

    def to_manifest(self) -> dict[str, object]:
        """Return stable lineage and typed child summaries without score tables."""

        self._require_intact()
        return {
            "schema_version": _SCHEMA_VERSION,
            "pipeline_id": self.pipeline_id,
            "lineage": self.lineage.to_dict(),
            "composition_policy_id": _COMPOSITION_POLICY_ID,
            "point_crossfit_id": self.point_artifacts.crossfit_id,
            "candidate_universe_id": self.universe.universe_id,
            "point_collection_id": self.point_records.collection_id,
            "active_null_result_id": self.null_rerun.result_id,
            "active_null_status": self.null_rerun.status.value,
            "null_distribution_id": self.null_rerun.distribution.distribution_id,
            "probability_collection_id": self.probability_collection.collection_id,
            "calibration_gate_id": self.probability_collection.calibration_gate_id,
            "calibration_gate_status": (
                None
                if self.probability_collection.calibration_gate_status is None
                else self.probability_collection.calibration_gate_status.value
            ),
            "comm_probability_release_allowed": (self.comm_probability_release_allowed),
            "retain_null_children": self.null_rerun.retain_children,
            "active_null_execution_backend": (
                self.null_rerun.to_manifest()["execution_backend"]
            ),
            "active_null_requested_n_jobs": self.null_rerun.requested_n_jobs,
            "active_null_effective_n_jobs": self.null_rerun.effective_n_jobs,
            "n_candidates": len(self.universe.candidate_edge_ids),
            "n_null_plans": len(self.null_rerun.requests),
            "probability_status_counts": [
                list(item) for item in self.probability_collection.status_counts
            ],
        }


def run_active_probability_pipeline(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    crossfit_spec: CrossFitSpec,
    contrast_id_or_name: str,
    n_plans: int,
    n_jobs: int = 1,
    universe_name: str | None = None,
    active_null_spec: ActiveNullSpec | None = None,
    active_probability_spec: ActiveProbabilitySpec | None = None,
    calibration_gate: G3PCalibrationGate | None = None,
    retain_children: bool = False,
) -> ActiveProbabilityPipelineResult:
    """Run point cross-fit through exact active-null probability inference.

    Active-null opportunities use at most ``min(n_jobs, n_plans)`` shared-
    snapshot threads.  Peak fold-working memory can scale with that effective
    concurrency even though the sanitized AnnData snapshot itself is shared.
    """

    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData")
    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig")
    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be a TargetPrior")
    if not isinstance(crossfit_spec, CrossFitSpec):
        raise TypeError("crossfit_spec must be a CrossFitSpec")
    selector = _name(contrast_id_or_name, field_name="contrast_id_or_name")
    plans = _positive_count(n_plans, field_name="n_plans")
    jobs = _positive_count(n_jobs, field_name="n_jobs")
    if universe_name is not None:
        _name(universe_name, field_name="universe_name")
    if not isinstance(retain_children, bool):
        raise TypeError("retain_children must be boolean")
    resolved_null_spec = (
        ActiveNullSpec() if active_null_spec is None else active_null_spec
    )
    if not isinstance(resolved_null_spec, ActiveNullSpec):
        raise TypeError("active_null_spec must be ActiveNullSpec or None")
    if resolved_null_spec.to_dict() != ActiveNullSpec().to_dict():
        raise _contract_error(
            "Active-null specification identity is not intact",
            code="active_null_spec_integrity_violation",
            field="active_null_spec_id",
            remediation="Rebuild the accepted ADR-013 active-null specification",
        )
    resolved_probability_spec = (
        ActiveProbabilitySpec()
        if active_probability_spec is None
        else active_probability_spec
    )
    if not isinstance(resolved_probability_spec, ActiveProbabilitySpec):
        raise TypeError("active_probability_spec must be ActiveProbabilitySpec or None")
    resolved_probability_spec._require_intact()
    if calibration_gate is not None:
        if not isinstance(calibration_gate, G3PCalibrationGate):
            raise TypeError("calibration_gate must be G3PCalibrationGate or None")
        calibration_gate._require_intact()

    point_artifacts = run_subject_crossfit(
        adata,
        config,
        resource_bundle,
        target_prior,
        spec=crossfit_spec,
    )
    universe = freeze_crossfit_active_edge_universe(
        point_artifacts,
        contrast_id_or_name=selector,
        universe_name=universe_name,
    )
    point_records = adapt_crossfit_active_edge_point_records(
        point_artifacts,
        universe,
    )
    null_rerun = run_active_null_reruns(
        adata,
        config,
        resource_bundle,
        target_prior,
        point_artifacts,
        universe,
        n_plans=plans,
        n_jobs=jobs,
        active_null_spec=resolved_null_spec,
        retain_children=retain_children,
    )
    probability_collection = estimate_active_probabilities(
        null_rerun.distribution,
        spec=resolved_probability_spec,
        calibration_gate=calibration_gate,
    )
    return ActiveProbabilityPipelineResult._from_workflow(
        point_artifacts=point_artifacts,
        universe=universe,
        point_records=point_records,
        null_rerun=null_rerun,
        active_null_spec=resolved_null_spec,
        active_probability_spec=resolved_probability_spec,
        calibration_gate=calibration_gate,
        probability_collection=probability_collection,
    )


__all__ = [
    "ActiveProbabilityPipelineLineage",
    "ActiveProbabilityPipelineResult",
    "run_active_probability_pipeline",
]
