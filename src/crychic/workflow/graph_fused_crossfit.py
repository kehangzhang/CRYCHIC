"""Independent, opt-in graph-fused subject cross-fit registry.

The released cross-fit workflow deliberately does not include graph-fused
attribution in its default contract.  This module is a producer-owned bridge
for callers that explicitly request the experimental graph estimand.  It
reconstructs every physical outer/inner scope from the raw-count snapshot,
keeps the outer family axis frozen, and applies each fitted graph model only
to the matching held-out response/design parents.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from anndata import AnnData

from crychic.attribution import (
    FrozenReceiverFamilyOpportunityUniverse,
    freeze_receiver_family_opportunity_universe,
)
from crychic.availability import (
    BatchAvailability,
    InteractionFilterApplication,
    estimate_bundle_availability,
)
from crychic.core import ContractError, SeedLineage, canonical_json, stable_id
from crychic.data import validate_anndata
from crychic.design import (
    ContextGraph,
    FrozenDesignEncoder,
    apply_frozen_design_encoder,
    fit_frozen_design_encoder,
    global_one_vs_rest,
    node_context_fields,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.resampling import DesignFoldChecker, FoldPlanningError, plan_subject_folds
from crychic.response import (
    FoldGeneResponseApplication,
    apply_fold_gene_response,
    fit_fold_gene_response,
)
from crychic.workflow.crossfit import (
    CrossFitArtifacts,
    CrossFitFoldArtifacts,
    _physical_subject_scope,
)
from crychic.workflow.graph_fused import (
    GraphFusedInnerFoldParents,
    GraphFusedInnerPartition,
    GraphFusedWorkflowApplication,
    GraphFusedWorkflowArtifact,
    GraphFusedWorkflowSpec,
    apply_graph_fused_workflow,
    fit_graph_fused_workflow,
    freeze_graph_fused_inner_fold_parents,
)
from crychic.workflow.receiver_universe import (
    ReceiverTrainingSupportRecord,
    ReceiverTrainingSupportStatus,
)
from crychic.workflow.training import (
    _input_schema,
    _prepare_raw_fold,
    _sanitized_raw_input_snapshot,
)

_SCHEMA_VERSION = "2.0.0"
_REGISTRY_IDENTITY_SCHEMA_VERSION = "2"
_PRODUCER = "crychic.workflow.graph_fused_crossfit_registry.v2"
_RECORD_PRODUCER = "crychic.workflow.graph_fused_crossfit_record.v1"


def _table_digest(table: pd.DataFrame) -> str:
    """Digest a small registry table without depending on pandas internals."""

    digest = hashlib.sha256()
    digest.update(canonical_json(list(map(str, table.columns))).encode("utf-8"))
    records = table.astype(object).where(table.notna(), None).to_dict(orient="records")
    digest.update(canonical_json(records).encode("utf-8"))
    return str(digest.hexdigest())


def _graph_id(graph: ContextGraph) -> str:
    return str(stable_id("context_graph", graph.to_dict(), schema_version="1"))


def _node_ids(graph: ContextGraph, context_keys: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(
        node_context_fields(node, tuple(context_keys))[0] for node in graph.nodes
    )
    if len(ids) != len(set(ids)):
        raise ContractError(
            "Context graph nodes collapse to duplicate context IDs",
            code="graph_fused_crossfit_context_id_collision",
            field="graph.nodes",
            remediation="Use graph nodes compatible with the cross-fit context keys",
        )
    return ids


def _node_availability(
    availability: BatchAvailability,
    *,
    node_context_id: str,
    training_subject_ids: tuple[str, ...],
) -> BatchAvailability:
    """Restrict an outer-universe application to one graph node.

    The interaction universe and resource provenance are copied verbatim.  Only
    sample-level evidence from the requested node is retained, and the declared
    application subjects remain the exact inner-training subject block even when
    a node has no observed rows (which then produces a typed non-estimable gate).
    """

    rows = availability.sample_interactions
    if "context_id" not in rows.columns:
        raise ContractError(
            "Availability table lacks the normalized context ID",
            code="graph_fused_crossfit_availability_context_missing",
            field="sample_interactions.context_id",
            remediation="Recompute availability from the validated aggregate",
        )
    selected = rows.loc[rows["context_id"].astype(str).eq(node_context_id)].copy(
        deep=True
    )
    selected.reset_index(drop=True, inplace=True)
    return BatchAvailability(
        sample_interactions=selected,
        mapping_summary=availability.mapping_summary.copy(deep=True),
        resource_id=availability.resource_id,
        resource_version=availability.resource_version,
        detection_available=availability.detection_available,
        frozen_interaction_universe=availability.frozen_interaction_universe,
        filter_application=InteractionFilterApplication.FROZEN_APPLICATION_V1,
        application_subject_ids=training_subject_ids,
    )


def _outer_parent_by_receiver(
    fold: CrossFitFoldArtifacts,
) -> dict[str, Any]:
    """Resolve one pooled family parent per receiver and reject divergence."""

    grouped: dict[str, list[Any]] = {}
    for model in fold.receiver_family_models:
        parent = model.receiver_family_artifact
        grouped.setdefault(parent.receiver, []).append(parent)
    result: dict[str, Any] = {}
    for receiver, parents in grouped.items():
        first = parents[0]
        first._require_producer_owned()
        if any(
            parent.training_artifact_id != first.training_artifact_id
            for parent in parents
        ):
            raise ContractError(
                "Receiver family parents disagree across cross-fit contrasts",
                code="graph_fused_crossfit_outer_family_parent_divergence",
                field="receiver_family_models",
                remediation="Use one pooled outer family axis for every contrast",
            )
        result[receiver] = first
    return result


def _require_parent_on_run_family_axis(
    parent: Any,
    universe: FrozenReceiverFamilyOpportunityUniverse,
) -> None:
    """Reject a fold-local family partition that differs from the run axis."""

    parent._require_producer_owned()
    basis = parent.family_basis
    if (
        parent.prior_manifest_digest != universe.prior_manifest_digest
        or basis.feature_ids != universe.feature_ids
        or basis.strict_cosine_threshold != universe.cosine_threshold
        or basis.family_definitions != universe.family_definitions
        or basis.family_ids != universe.family_ids
        or parent.source_basis.driver_ids != universe.driver_ids
    ):
        raise ContractError(
            "Outer receiver family axis differs from the frozen run-level universe",
            code="graph_fused_crossfit_run_family_axis_mismatch",
            field="receiver_family_models",
            remediation=(
                "Rebuild graph-fused cross-fit from the exact target prior, root "
                "feature axis, and family threshold"
            ),
        )


def _graph_encoder(
    metadata: pd.DataFrame,
    *,
    graph: ContextGraph,
    config: Any,
    training_subject_ids: tuple[str, ...],
    contrast_name: str,
) -> FrozenDesignEncoder:
    """Fit a graph-wide design using only the current outer-training metadata."""

    contrast = global_one_vs_rest(
        graph,
        graph.nodes[0],
        name=contrast_name,
        family="graph_fused_crossfit",
    )
    subset = metadata.loc[
        metadata[config.subject_key].astype(str).isin(training_subject_ids)
    ].copy(deep=True)
    return fit_frozen_design_encoder(
        subset,
        contrast=contrast,
        context_keys=tuple(config.context_keys),
        covariates=tuple(config.covariates),
        categorical_covariates=tuple(config.categorical_covariates),
        formula=config.design,
        sample_key=config.sample_key,
        subject_key=config.subject_key,
    )


def _inner_plan(
    metadata: pd.DataFrame,
    *,
    graph: ContextGraph,
    config: Any,
    crossfit: CrossFitArtifacts,
    outer_fold_id: str,
    outer_subject_ids: tuple[str, ...],
    graph_contrast: Any,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...]:
    """Plan deterministic subject-blocked inner folds from outer metadata."""

    outer_metadata = metadata.loc[
        metadata[config.subject_key].astype(str).isin(outer_subject_ids)
    ].copy(deep=True)
    checker = DesignFoldChecker(
        context_keys=tuple(config.context_keys),
        covariates=tuple(config.covariates),
        categorical_covariates=tuple(config.categorical_covariates),
        formula=config.design,
        contrasts=(graph_contrast,),
        sample_key=config.sample_key,
    )
    repeat_id = stable_id(
        "graph_fused_crossfit_inner_repeat",
        {"crossfit_id": crossfit.crossfit_id, "outer_fold_id": outer_fold_id},
        schema_version="1",
    )
    lineage = SeedLineage(config.random_seed).derive(
        "graph_fused_crossfit_inner", crossfit.spec.spec_id, outer_fold_id
    )
    try:
        plan = plan_subject_folds(
            outer_metadata,
            design_checker=checker,
            subject_key=config.subject_key,
            sample_key=config.sample_key,
            context_keys=tuple(config.context_keys),
            strata_keys=tuple(crossfit.spec.strata_keys),
            allowed_n_splits=tuple(crossfit.spec.allowed_n_splits),
            min_train_subjects_per_context=max(
                2, crossfit.spec.min_train_subjects_per_context
            ),
            min_test_subjects_per_context=max(
                1, crossfit.spec.min_test_subjects_per_context
            ),
            repeat_id=repeat_id,
            seed_lineage=lineage,
        )
        return tuple(
            (fold.train_subject_ids, fold.test_subject_ids, fold.fold_id)
            for fold in sorted(plan.folds, key=lambda value: value.fold_id)
        )
    except FoldPlanningError:
        # A deterministic two-way fallback retains a typed-NE path for small or
        # confounded outer scopes; it never consults held-out values.
        if len(outer_subject_ids) < 4:
            return ()
        midpoint = len(outer_subject_ids) // 2
        groups = (outer_subject_ids[:midpoint], outer_subject_ids[midpoint:])
        return tuple(
            (
                tuple(
                    subject
                    for subject in outer_subject_ids
                    if subject not in validation
                ),
                tuple(validation),
                stable_id(
                    "graph_fused_crossfit_inner_fold",
                    {"outer_fold_id": outer_fold_id, "index": index},
                    schema_version="1",
                ),
            )
            for index, validation in enumerate(groups)
        )


def _make_inner_partition(
    *,
    graph: ContextGraph,
    graph_node_ids: tuple[str, ...],
    outer_fold: CrossFitFoldArtifacts,
    outer_parent_by_receiver: Mapping[str, Any],
    outer_aggregate: Any,
    raw_snapshot: Any,
    sample_metadata: pd.DataFrame,
    config: Any,
    crossfit: CrossFitArtifacts,
    receiver: str,
    training: Any,
) -> GraphFusedInnerPartition | None:
    outer_subjects = tuple(sorted(outer_fold.training.training_subject_ids))
    if len(outer_subjects) < 4:
        return None
    metadata = sample_metadata
    contrast = global_one_vs_rest(graph, graph.nodes[0], name="graph-inner")
    planned = _inner_plan(
        metadata,
        graph=graph,
        config=config,
        crossfit=crossfit,
        outer_fold_id=outer_fold.fold_id,
        outer_subject_ids=outer_subjects,
        graph_contrast=contrast,
    )
    if len(planned) < 2:
        return None
    parents = {node: outer_parent_by_receiver[receiver] for node in graph.nodes}
    frozen: list[GraphFusedInnerFoldParents] = []
    for inner_training, inner_validation, inner_fold_id in planned:
        inner_scope = _physical_subject_scope(
            raw_snapshot.adata,
            subject_key=config.subject_key,
            subject_ids=tuple(sorted(inner_training)),
        )
        prepared = _prepare_raw_fold(
            inner_scope,
            config,
            min_cells=training.spec.min_cells,
            cell_types=training.cell_type_ids,
            root_input_identity=raw_snapshot.identity,
        )
        if prepared.aggregate.feature_ids != outer_aggregate.feature_ids:
            raise ContractError(
                "Inner aggregate feature axis differs from the outer aggregate",
                code="graph_fused_crossfit_inner_feature_axis_mismatch",
                field="aggregate.feature_ids",
                remediation="Rebuild inner scopes from the exact raw snapshot",
            )
        availability = estimate_bundle_availability(
            prepared.aggregate,
            training.resource_bundle,
            context_keys=tuple(config.context_keys),
            parameters=training.spec.availability_parameters,
            min_pooled_availability=training.spec.min_pooled_availability,
            frozen_interaction_universe=training.frozen_interaction_universe,
        )
        per_node = {
            node: _node_availability(
                availability,
                node_context_id=node_id,
                training_subject_ids=tuple(sorted(inner_training)),
            )
            for node, node_id in zip(graph.nodes, graph_node_ids, strict=True)
        }
        frozen.append(
            freeze_graph_fused_inner_fold_parents(
                fold_id=inner_fold_id,
                graph=graph,
                receiver=receiver,
                training_subject_ids=tuple(sorted(inner_training)),
                validation_subject_ids=tuple(sorted(inner_validation)),
                family_parents=parents,
                inner_training_availability=per_node,
                target_prior=training.target_prior,
            )
        )
    return GraphFusedInnerPartition(
        outer_training_subject_ids=outer_subjects,
        folds=tuple(frozen),
    )


@dataclass(frozen=True, slots=True, init=False)
class GraphFusedCrossFitRecord:
    """One outer-fold/receiver graph model and its held-out application."""

    fold_id: str
    receiver: str
    receiver_universe_id: str
    receiver_training_support_id: str
    receiver_training_support_status: str
    receiver_training_support_reason_code: str | None
    training_subject_ids: tuple[str, ...]
    heldout_subject_ids: tuple[str, ...]
    inner_partition_id: str | None
    workflow: GraphFusedWorkflowArtifact | None
    application: GraphFusedWorkflowApplication | None
    status: str
    reason_code: str | None
    record_id: str
    _receiver_training_support: ReceiverTrainingSupportRecord = field(repr=False)
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError("GraphFusedCrossFitRecord is producer-owned")

    @classmethod
    def _from_producer(cls, **values: object) -> GraphFusedCrossFitRecord:
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "application_id": None
            if self.application is None
            else self.application.application_id,
            "fold_id": self.fold_id,
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "inner_partition_id": self.inner_partition_id,
            "receiver": self.receiver,
            "receiver_training_support_id": self.receiver_training_support_id,
            "receiver_training_support_reason_code": (
                self.receiver_training_support_reason_code
            ),
            "receiver_training_support_status": (
                self.receiver_training_support_status
            ),
            "receiver_universe_id": self.receiver_universe_id,
            "reason_code": self.reason_code,
            "status": self.status,
            "training_subject_ids": list(self.training_subject_ids),
            "workflow_id": None if self.workflow is None else self.workflow.workflow_id,
        }

    def _require_intact(self) -> None:
        if self._producer_marker != _RECORD_PRODUCER:
            raise ContractError(
                "Graph-fused cross-fit record is not producer-owned",
                code="graph_fused_crossfit_record_integrity_violation",
                field="record_id",
                remediation="Recreate the registry from intact raw inputs",
            )
        if not isinstance(
            self._receiver_training_support, ReceiverTrainingSupportRecord
        ):
            raise ContractError(
                "Graph-fused record lacks producer-owned receiver support",
                code="graph_fused_crossfit_record_integrity_violation",
                field="receiver_training_support_id",
                remediation="Recreate the record from intact cross-fit support",
            )
        self._receiver_training_support._require_intact()
        support = self._receiver_training_support
        support_matches = (
            self.receiver_universe_id == support.receiver_universe_id
            and self.receiver_training_support_id == support.support_record_id
            and self.receiver_training_support_status == support.status.value
            and self.receiver_training_support_reason_code == support.reason_code
            and self.fold_id == support.outer_fold_id
            and self.receiver == support.receiver_id
        )
        if not support_matches:
            raise ContractError(
                "Graph-fused record receiver support lineage does not match",
                code="graph_fused_crossfit_record_integrity_violation",
                field="receiver_training_support_id",
                remediation="Recreate the record from intact cross-fit support",
            )
        if self.status not in {"observed", "not_estimable"}:
            raise ContractError(
                "Graph-fused cross-fit record has an invalid status",
                code="graph_fused_crossfit_record_integrity_violation",
                field="status",
                remediation="Use observed or typed not_estimable records",
            )
        if support.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE and (
            self.status != "not_estimable"
            or self.reason_code != "receiver_absent_in_outer_training"
            or self.inner_partition_id is not None
            or self.workflow is not None
            or self.application is not None
        ):
            raise ContractError(
                "Training-absent receiver released graph model lineage",
                code="graph_fused_crossfit_record_integrity_violation",
                field="receiver_training_support_status",
                remediation=(
                    "Keep the complete receiver opportunity as typed not_estimable "
                    "without family, gate, model, or application parents"
                ),
            )
        if (self.status == "observed") != (
            self.workflow is not None and self.application is not None
        ):
            raise ContractError(
                "Graph-fused cross-fit record parents do not match status",
                code="graph_fused_crossfit_record_integrity_violation",
                field="workflow,application,status",
                remediation="Rebuild the complete outer-fold record",
            )
        if self.status == "observed":
            assert self.workflow is not None and self.application is not None
            self.workflow._require_intact()
            self.application._require_intact()
            problem = self.workflow.problem
            fit = self.workflow.fit
            parents_match = (
                self.reason_code is None
                and self.inner_partition_id == self.workflow.inner_partition_id
                and self.fold_id == problem.fold_id
                and self.receiver == problem.receiver
                and self.training_subject_ids == problem.training_subject_ids
                and self.heldout_subject_ids == problem.heldout_subject_ids
                and self.application.status == "observed"
                and self.application.workflow_id == self.workflow.workflow_id
                and self.application.problem_id == problem.problem_id
                and self.application.graph_id == problem.graph_id
                and self.application.application_functional_id
                == problem.application_functional_id
                and self.application.training_subject_ids
                == self.training_subject_ids
                and self.application.heldout_subject_ids
                == self.heldout_subject_ids
                and fit is not None
                and self.application.fit_id == fit.tuned_attribution_id
            )
            if not parents_match:
                raise ContractError(
                    "Graph-fused record workflow and application do not share "
                    "one exact outer-fold lineage",
                    code="graph_fused_crossfit_record_parent_mismatch",
                    field="workflow,application",
                    remediation="Rebuild the record from one matched graph workflow",
                )
        elif self.reason_code is None:
            raise ContractError(
                "Not-estimable graph-fused records require a reason code",
                code="graph_fused_crossfit_record_integrity_violation",
                field="reason_code",
                remediation="Recreate the registry from intact raw inputs",
            )
        expected = stable_id(
            "graph_fused_crossfit_record", self._identity_payload(), schema_version="1"
        )
        if self.record_id != expected:
            raise ContractError(
                "Graph-fused cross-fit record identity is not intact",
                code="graph_fused_crossfit_record_integrity_violation",
                field="record_id",
                remediation="Recreate the record from intact parents",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "record_id": self.record_id,
            "fold_id": self.fold_id,
            "receiver": self.receiver,
            "receiver_universe_id": self.receiver_universe_id,
            "receiver_training_support_id": self.receiver_training_support_id,
            "receiver_training_support_status": (
                self.receiver_training_support_status
            ),
            "receiver_training_support_reason_code": (
                self.receiver_training_support_reason_code
            ),
            "training_subject_ids": list(self.training_subject_ids),
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "inner_partition_id": self.inner_partition_id,
            "workflow_id": None if self.workflow is None else self.workflow.workflow_id,
            "application_id": None
            if self.application is None
            else self.application.application_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "experimental": True,
            "formal_inference_allowed": False,
            "p_value": None,
            "q_value": None,
        }


@dataclass(frozen=True, slots=True, init=False)
class GraphFusedCrossFitRegistry:
    """Complete producer-owned graph-fused outer-fold registry."""

    graph: ContextGraph
    spec: GraphFusedWorkflowSpec
    crossfit_id: str
    receiver_universe_id: str
    receiver_axis_id: str
    receiver_ids: tuple[str, ...]
    receiver_family_universe: FrozenReceiverFamilyOpportunityUniverse
    receiver_family_universe_id: str
    family_axis_id: str
    family_ids: tuple[str, ...]
    outer_fold_ids: tuple[str, ...]
    root_input_identity_id: str
    root_input_digest: str
    source_snapshot_id: str
    records: tuple[GraphFusedCrossFitRecord, ...]
    table_digest: str
    registry_id: str
    certification_status: str
    _table: pd.DataFrame = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError("GraphFusedCrossFitRegistry is producer-owned")

    @classmethod
    def _from_producer(cls, **values: object) -> GraphFusedCrossFitRegistry:
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    @property
    def table(self) -> pd.DataFrame:
        self._require_intact()
        return self._table.copy(deep=True)

    @property
    def observed_records(self) -> tuple[GraphFusedCrossFitRecord, ...]:
        return tuple(record for record in self.records if record.status == "observed")

    @property
    def not_estimable_records(self) -> tuple[GraphFusedCrossFitRecord, ...]:
        return tuple(
            record for record in self.records if record.status == "not_estimable"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "certification_status": self.certification_status,
            "crossfit_id": self.crossfit_id,
            "graph_id": _graph_id(self.graph),
            "outer_fold_ids": list(self.outer_fold_ids),
            "record_ids": [record.record_id for record in self.records],
            "receiver_axis_id": self.receiver_axis_id,
            "receiver_family_universe_id": self.receiver_family_universe_id,
            "receiver_ids": list(self.receiver_ids),
            "receiver_universe_id": self.receiver_universe_id,
            "family_axis_id": self.family_axis_id,
            "family_ids": list(self.family_ids),
            "root_input_digest": self.root_input_digest,
            "root_input_identity_id": self.root_input_identity_id,
            "source_snapshot_id": self.source_snapshot_id,
            "spec_id": self.spec.spec_id,
            "table_digest": self.table_digest,
        }

    def _require_intact(self) -> None:
        if self._producer_marker != _PRODUCER:
            raise ContractError(
                "Graph-fused cross-fit registry is not producer-owned",
                code="graph_fused_crossfit_registry_integrity_violation",
                field="registry_id",
                remediation="Recreate the registry from intact raw inputs",
            )
        self.spec._require_intact()
        if not isinstance(
            self.receiver_family_universe,
            FrozenReceiverFamilyOpportunityUniverse,
        ):
            raise ContractError(
                "Graph-fused registry lacks its frozen receiver-family universe",
                code="graph_fused_crossfit_registry_integrity_violation",
                field="receiver_family_universe",
                remediation="Recreate the registry from intact raw inputs",
            )
        self.receiver_family_universe._require_intact()
        if not isinstance(self.graph, ContextGraph):
            raise ContractError(
                "Graph-fused registry graph is invalid",
                code="graph_fused_crossfit_registry_integrity_violation",
                field="graph",
                remediation="Use an intact ContextGraph",
            )
        records = tuple(self.records)
        record_keys = tuple((record.fold_id, record.receiver) for record in records)
        receiver_ids = tuple(self.receiver_ids)
        fold_ids = tuple(self.outer_fold_ids)
        expected_record_keys = tuple(
            (fold_id, receiver)
            for fold_id in fold_ids
            for receiver in receiver_ids
        )
        expected_receiver_axis_id = stable_id(
            "receiver_axis",
            {"receiver_ids": list(receiver_ids)},
            schema_version="1",
        )
        family_universe = self.receiver_family_universe
        if (
            not records
            or not receiver_ids
            or not fold_ids
            or receiver_ids != tuple(sorted(set(receiver_ids)))
            or fold_ids != tuple(sorted(set(fold_ids)))
            or not isinstance(self.receiver_universe_id, str)
            or not self.receiver_universe_id
            or self.receiver_axis_id != expected_receiver_axis_id
            or self.receiver_family_universe_id != family_universe.universe_id
            or self.family_axis_id != family_universe.family_axis_id
            or self.family_ids != family_universe.family_ids
            or self.receiver_universe_id != family_universe.receiver_universe_id
            or self.receiver_axis_id != family_universe.receiver_axis_id
            or receiver_ids != family_universe.receiver_ids
            or self.root_input_identity_id
            != family_universe.root_input_identity_id
            or self.root_input_digest != family_universe.root_input_digest
            or len(record_keys) != len(set(record_keys))
            or record_keys != tuple(sorted(record_keys))
            or record_keys != expected_record_keys
            or len({record.record_id for record in records}) != len(records)
            or len(
                {record.receiver_training_support_id for record in records}
            )
            != len(records)
            or any(
                record.receiver_universe_id != self.receiver_universe_id
                for record in records
            )
        ):
            raise ContractError(
                "Graph-fused registry records are incomplete or non-canonical",
                code="graph_fused_crossfit_registry_integrity_violation",
                field="records",
                remediation="Recreate the registry from intact raw inputs",
            )
        for record in records:
            record._require_intact()
        expected_table = pd.DataFrame([record.to_dict() for record in records])
        if (
            not self._table.equals(expected_table)
            or self.table_digest != _table_digest(expected_table)
            or self.certification_status != "complete_descriptive"
        ):
            raise ContractError(
                "Graph-fused registry table does not match its records",
                code="graph_fused_crossfit_registry_integrity_violation",
                field="records,table_digest",
                remediation="Recreate the registry from intact parents",
            )
        expected = stable_id(
            "graph_fused_crossfit_registry",
            self._identity_payload(),
            schema_version=_REGISTRY_IDENTITY_SCHEMA_VERSION,
        )
        if self.registry_id != expected:
            raise ContractError(
                "Graph-fused registry identity is not intact",
                code="graph_fused_crossfit_registry_integrity_violation",
                field="registry_id",
                remediation="Recreate the registry from intact raw inputs",
            )

    def to_manifest(self) -> dict[str, object]:
        self._require_intact()
        return {
            "registry_id": self.registry_id,
            "schema_version": _SCHEMA_VERSION,
            "producer": _PRODUCER,
            "crossfit_id": self.crossfit_id,
            "receiver_universe_id": self.receiver_universe_id,
            "receiver_axis_id": self.receiver_axis_id,
            "receiver_ids": list(self.receiver_ids),
            "receiver_family_universe": self.receiver_family_universe.to_dict(),
            "receiver_family_universe_id": self.receiver_family_universe_id,
            "family_axis_id": self.family_axis_id,
            "family_ids": list(self.family_ids),
            "outer_fold_ids": list(self.outer_fold_ids),
            "spec": self.spec.to_dict(),
            "graph": self.graph.to_dict(),
            "graph_id": _graph_id(self.graph),
            "root_input_identity_id": self.root_input_identity_id,
            "root_input_digest": self.root_input_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "record_ids": [record.record_id for record in self.records],
            "table_digest": self.table_digest,
            "status": self.certification_status,
            "experimental": True,
            "formal_inference_allowed": False,
            "p_value": None,
            "q_value": None,
        }


def _record(
    *,
    fold_id: str,
    receiver: str,
    receiver_training_support: ReceiverTrainingSupportRecord,
    training_subject_ids: tuple[str, ...],
    heldout_subject_ids: tuple[str, ...],
    inner_partition: GraphFusedInnerPartition | None,
    workflow: GraphFusedWorkflowArtifact | None,
    application: GraphFusedWorkflowApplication | None,
    reason_code: str | None,
) -> GraphFusedCrossFitRecord:
    receiver_training_support._require_intact()
    if (
        receiver_training_support.outer_fold_id != fold_id
        or receiver_training_support.receiver_id != receiver
    ):
        raise ContractError(
            "Receiver support does not match the requested graph opportunity",
            code="graph_fused_crossfit_receiver_support_mismatch",
            field="receiver_training_support",
            remediation="Use the support record from this exact outer fold/receiver",
        )
    if receiver_training_support.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE:
        if (
            inner_partition is not None
            or workflow is not None
            or application is not None
        ):
            raise ContractError(
                "Training-absent receiver cannot own graph model parents",
                code="graph_fused_crossfit_receiver_absent_parent_violation",
                field="workflow,application",
                remediation="Emit only the typed receiver opportunity record",
            )
        reason_code = receiver_training_support.reason_code
    status = (
        "observed"
        if workflow is not None
        and application is not None
        and application.status == "observed"
        else "not_estimable"
    )
    if status == "observed":
        reason_code = None
    elif not reason_code:
        reason_code = "graph_fused_crossfit_not_estimable"
    values: dict[str, object] = {
        "fold_id": fold_id,
        "receiver": receiver,
        "receiver_universe_id": receiver_training_support.receiver_universe_id,
        "receiver_training_support_id": (
            receiver_training_support.support_record_id
        ),
        "receiver_training_support_status": receiver_training_support.status.value,
        "receiver_training_support_reason_code": (
            receiver_training_support.reason_code
        ),
        "training_subject_ids": training_subject_ids,
        "heldout_subject_ids": heldout_subject_ids,
        "inner_partition_id": None
        if inner_partition is None
        else inner_partition.partition_id,
        "workflow": workflow if status == "observed" else None,
        "application": application if status == "observed" else None,
        "status": status,
        "reason_code": reason_code,
        "_receiver_training_support": receiver_training_support,
        "_producer_marker": _RECORD_PRODUCER,
    }
    temporary = GraphFusedCrossFitRecord._from_producer(**values, record_id="pending")
    values["record_id"] = stable_id(
        "graph_fused_crossfit_record", temporary._identity_payload(), schema_version="1"
    )
    result = GraphFusedCrossFitRecord._from_producer(**values)
    result._require_intact()
    return result


def run_graph_fused_crossfit(
    adata: AnnData,
    crossfit: CrossFitArtifacts,
    graph: ContextGraph,
    spec: GraphFusedWorkflowSpec,
    *,
    contrast_name: str | None = None,
) -> GraphFusedCrossFitRegistry:
    """Run the opt-in graph-fused bridge on every outer fold/receiver.

    ``adata`` must be the exact raw-count input used by ``crossfit``.  The
    function never accepts caller-created folds, aggregates, matrices, or
    validation values; all such objects are reconstructed from the producer
    cross-fit manifest and the authenticated raw snapshot.
    """

    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData instance")
    if not isinstance(crossfit, CrossFitArtifacts):
        raise TypeError("crossfit must be CrossFitArtifacts")
    if not isinstance(graph, ContextGraph):
        raise TypeError("graph must be a ContextGraph")
    if not isinstance(spec, GraphFusedWorkflowSpec):
        raise TypeError("spec must be a GraphFusedWorkflowSpec")
    crossfit._require_intact()
    spec._require_intact()
    if len(graph.nodes) < 2:
        raise ValueError("graph-fused cross-fit requires at least two graph nodes")
    if not crossfit.folds:
        raise ValueError("crossfit must contain at least one outer fold")
    training0 = crossfit.folds[0].training
    config = training0.config
    snapshot = _sanitized_raw_input_snapshot(adata, config)
    if (
        snapshot.identity.identity_id != crossfit.root_input_identity.identity_id
        or snapshot.identity.input_digest != crossfit.root_input_identity.input_digest
    ):
        raise ContractError(
            "Raw input does not match the cross-fit root identity",
            code="graph_fused_crossfit_root_input_mismatch",
            field="adata",
            remediation="Provide the exact raw-count AnnData used for cross-fit",
        )
    graph_node_ids = _node_ids(graph, tuple(config.context_keys))
    records: list[GraphFusedCrossFitRecord] = []
    validated = validate_anndata(snapshot.adata, _input_schema(config))
    metadata = validated.report.sample_metadata.copy(deep=True)
    receiver_family_universe = freeze_receiver_family_opportunity_universe(
        training0.target_prior,
        feature_ids=validated.feature_ids,
        receiver_ids=crossfit.receiver_universe.receiver_ids,
        receiver_universe_id=crossfit.receiver_universe.universe_id,
        receiver_axis_id=crossfit.receiver_universe.receiver_axis_id,
        prior_content_id=training0.target_prior_content_id,
        root_input_identity_id=crossfit.root_input_identity.identity_id,
        root_input_digest=crossfit.root_input_identity.input_digest,
        cosine_threshold=crossfit.spec.family_cosine_threshold,
    )
    for outer_fold in sorted(crossfit.folds, key=lambda value: value.fold_id):
        training = outer_fold.training
        if (
            training.config.digest != config.digest
            or training.target_prior.manifest_digest
            != receiver_family_universe.prior_manifest_digest
            or training.target_prior.resource_id
            != receiver_family_universe.prior_resource_id
            or training.target_prior.version != receiver_family_universe.prior_version
            or training.target_prior_content_id
            != receiver_family_universe.prior_content_id
        ):
            raise ContractError(
                "Outer fold config or target prior disagrees with the run root",
                code="graph_fused_crossfit_outer_config_mismatch",
                field="training.config,target_prior",
                remediation="Use one intact cross-fit artifact",
            )
        outer_subjects = tuple(sorted(training.training_subject_ids))
        heldout_subjects = tuple(sorted(outer_fold.application.heldout_subject_ids))
        training_scope = _physical_subject_scope(
            snapshot.adata,
            subject_key=config.subject_key,
            subject_ids=outer_subjects,
        )
        heldout_scope = _physical_subject_scope(
            snapshot.adata,
            subject_key=config.subject_key,
            subject_ids=heldout_subjects,
        )
        prepared_training = _prepare_raw_fold(
            training_scope,
            config,
            min_cells=training.spec.min_cells,
            root_input_identity=snapshot.identity,
        )
        prepared_heldout = _prepare_raw_fold(
            heldout_scope,
            config,
            min_cells=training.spec.min_cells,
            cell_types=training.cell_type_ids,
            root_input_identity=snapshot.identity,
        )
        if not isinstance(
            prepared_training.aggregate, PseudobulkDataset
        ) or not isinstance(prepared_heldout.aggregate, PseudobulkDataset):
            raise ContractError(
                "Graph-fused cross-fit requires count-derived pseudobulk scopes",
                code="graph_fused_crossfit_pseudobulk_required",
                field="aggregate",
                remediation="Rebuild the outer scopes from raw counts",
            )
        training_aggregate = prepared_training.aggregate
        heldout_aggregate = prepared_heldout.aggregate
        if prepared_training.input_digest != training.training_input_digest:
            raise ContractError(
                "Reconstructed outer training scope does not match its artifact",
                code="graph_fused_crossfit_outer_training_scope_mismatch",
                field="training_input_digest",
                remediation="Use the exact raw snapshot and fold manifest",
            )
        parents = _outer_parent_by_receiver(outer_fold)
        for parent in parents.values():
            _require_parent_on_run_family_axis(parent, receiver_family_universe)
        if tuple(sorted(parents)) != training.cell_type_ids:
            raise ContractError(
                "Observed outer receivers do not have one exact family parent",
                code="graph_fused_crossfit_outer_family_parent_coverage_mismatch",
                field="receiver_family_models",
                remediation="Rebuild the intact outer cross-fit family registry",
            )
        support_by_receiver = {
            support.receiver_id: support
            for support in outer_fold.receiver_training_support
        }
        if tuple(sorted(support_by_receiver)) != (
            crossfit.receiver_universe.receiver_ids
        ):
            raise ContractError(
                "Outer fold support does not cover the frozen receiver universe",
                code="graph_fused_crossfit_receiver_support_coverage_mismatch",
                field="receiver_training_support",
                remediation="Use one intact CrossFitArtifacts receiver universe",
            )
        for receiver in crossfit.receiver_universe.receiver_ids:
            support = support_by_receiver[receiver]
            if support.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE:
                records.append(
                    _record(
                        fold_id=outer_fold.fold_id,
                        receiver=receiver,
                        receiver_training_support=support,
                        training_subject_ids=outer_subjects,
                        heldout_subject_ids=heldout_subjects,
                        inner_partition=None,
                        workflow=None,
                        application=None,
                        reason_code=support.reason_code,
                    )
                )
                continue
            if receiver not in parents:
                raise ContractError(
                    "Observed receiver is missing its outer family parent",
                    code="graph_fused_crossfit_outer_family_parent_coverage_mismatch",
                    field="receiver_family_models",
                    remediation="Rebuild the intact outer cross-fit family registry",
                )
            reason: str | None = None
            inner_partition: GraphFusedInnerPartition | None = None
            workflow: GraphFusedWorkflowArtifact | None = None
            application: GraphFusedWorkflowApplication | None = None
            try:
                graph_encoder = _graph_encoder(
                    metadata,
                    graph=graph,
                    config=config,
                    training_subject_ids=outer_subjects,
                    contrast_name=contrast_name
                    or f"graph_fused:{outer_fold.fold_id}:{receiver}",
                )
                graph_application_design = apply_frozen_design_encoder(
                    graph_encoder,
                    metadata.loc[
                        metadata[config.subject_key].astype(str).isin(heldout_subjects)
                    ].copy(deep=True),
                )
                inner_partition = _make_inner_partition(
                    graph=graph,
                    graph_node_ids=graph_node_ids,
                    outer_fold=outer_fold,
                    outer_parent_by_receiver=parents,
                    outer_aggregate=training_aggregate,
                    raw_snapshot=snapshot,
                    sample_metadata=metadata,
                    config=config,
                    crossfit=crossfit,
                    receiver=receiver,
                    training=training,
                )
                if inner_partition is None:
                    reason = "graph_fused_crossfit_inner_partition_not_estimable"
                else:
                    workflow = fit_graph_fused_workflow(
                        training_aggregate,
                        metadata.loc[
                            metadata[config.subject_key]
                            .astype(str)
                            .isin(outer_subjects)
                        ].copy(deep=True),
                        graph_encoder,
                        {node: parents[receiver] for node in graph.nodes},
                        graph,
                        inner_partition,
                        receiver=receiver,
                        fold_id=outer_fold.fold_id,
                        training_input_digest=training.training_input_digest,
                        outer_training_subject_ids=outer_subjects,
                        heldout_subject_ids=heldout_subjects,
                        spec=spec,
                    )
                    response = fit_fold_gene_response(
                        training_aggregate,
                        graph_encoder,
                        receiver=receiver,
                        fold_id=outer_fold.fold_id,
                        training_input_digest=training.training_input_digest,
                        min_subjects_per_context=2,
                    )
                    response_application: FoldGeneResponseApplication = (
                        apply_fold_gene_response(
                            heldout_aggregate,
                            response,
                            graph_application_design,
                        )
                    )
                    application = apply_graph_fused_workflow(
                        workflow,
                        response_application,
                        graph_application_design,
                    )
                    if (
                        workflow.status != "observed"
                        or application.status != "observed"
                    ):
                        reason = workflow.reason_code or application.reason_code
            except (ContractError, RuntimeError, ValueError, KeyError) as error:
                reason = getattr(getattr(error, "details", None), "code", None) or str(
                    error
                )
                workflow = None
                application = None
            records.append(
                _record(
                    fold_id=outer_fold.fold_id,
                    receiver=receiver,
                    receiver_training_support=support,
                    training_subject_ids=outer_subjects,
                    heldout_subject_ids=heldout_subjects,
                    inner_partition=inner_partition,
                    workflow=workflow,
                    application=application,
                    reason_code=reason,
                )
            )
    ordered_records = tuple(
        sorted(records, key=lambda value: (value.fold_id, value.receiver))
    )
    table = pd.DataFrame([record.to_dict() for record in ordered_records])
    if table.empty:
        table = pd.DataFrame(
            columns=(
                "record_id",
                "fold_id",
                "receiver",
                "receiver_universe_id",
                "receiver_training_support_id",
                "receiver_training_support_status",
                "receiver_training_support_reason_code",
                "training_subject_ids",
                "heldout_subject_ids",
                "inner_partition_id",
                "workflow_id",
                "application_id",
                "status",
                "reason_code",
                "experimental",
                "formal_inference_allowed",
                "p_value",
                "q_value",
            )
        )
    table_digest = _table_digest(table)
    values: dict[str, object] = {
        "graph": graph,
        "spec": spec,
        "crossfit_id": crossfit.crossfit_id,
        "receiver_universe_id": crossfit.receiver_universe.universe_id,
        "receiver_axis_id": crossfit.receiver_universe.receiver_axis_id,
        "receiver_ids": crossfit.receiver_universe.receiver_ids,
        "receiver_family_universe": receiver_family_universe,
        "receiver_family_universe_id": receiver_family_universe.universe_id,
        "family_axis_id": receiver_family_universe.family_axis_id,
        "family_ids": receiver_family_universe.family_ids,
        "outer_fold_ids": tuple(sorted(fold.fold_id for fold in crossfit.folds)),
        "root_input_identity_id": crossfit.root_input_identity.identity_id,
        "root_input_digest": crossfit.root_input_identity.input_digest,
        # The raw snapshot's defensive metadata digest may depend on physical
        # row order; the authenticated root identity is intentionally canonical
        # and is the registry's stable source lineage.
        "source_snapshot_id": stable_id(
            "graph_fused_crossfit_source_snapshot",
            {"root_input_identity_id": crossfit.root_input_identity.identity_id},
            schema_version="1",
        ),
        "records": ordered_records,
        "table_digest": table_digest,
        "certification_status": "complete_descriptive"
        if all(
            record.status in {"observed", "not_estimable"} for record in ordered_records
        )
        else "partial",
        "_table": table,
        "_producer_marker": _PRODUCER,
    }
    temporary = GraphFusedCrossFitRegistry._from_producer(
        **values, registry_id="pending"
    )
    values["registry_id"] = stable_id(
        "graph_fused_crossfit_registry",
        temporary._identity_payload(),
        schema_version=_REGISTRY_IDENTITY_SCHEMA_VERSION,
    )
    result = GraphFusedCrossFitRegistry._from_producer(**values)
    result._require_intact()
    return result


# Explicit aliases make the opt-in nature clear while keeping the primary API
# discoverable for callers that name the producer after its returned registry.
run_graph_fused_crossfit_registry = run_graph_fused_crossfit
fit_graph_fused_crossfit_registry = run_graph_fused_crossfit


__all__ = [
    "GraphFusedCrossFitRecord",
    "GraphFusedCrossFitRegistry",
    "fit_graph_fused_crossfit_registry",
    "run_graph_fused_crossfit",
    "run_graph_fused_crossfit_registry",
]
