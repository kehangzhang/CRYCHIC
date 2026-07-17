"""Outer-training-only bridge for experimental graph-fused attribution.

The public cross-fit object intentionally does not retain its training
``PseudobulkDataset``.  This module is therefore a producer that must run while
the physical outer-training aggregate is still in scope.  It freezes only the
compact context-by-feature response and precision problem needed downstream.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.attribution import (
    FamilyFirstBasis,
    GraphPenaltyTuningSpec,
    GraphTuningFold,
    ReceiverFamilyAxisRefitResult,
    ReceiverFamilyTrainingArtifact,
    TunedGraphFusedFamilyAttribution,
    fit_tuned_graph_fused_family_attribution,
    refit_receiver_family_training_artifact_on_frozen_axis,
)
from crychic.availability import BatchAvailability
from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import (
    ContextGraph,
    FrozenDesignApplication,
    FrozenDesignEncoder,
    apply_frozen_design_encoder,
    audit_sample_design,
    fit_frozen_design_encoder,
    global_one_vs_rest,
    node_context_fields,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.resources import TargetPrior
from crychic.response import (
    FoldGeneResponseApplication,
    FoldGeneResponseArtifact,
    fit_fold_gene_response,
)

_SCHEMA_VERSION = "1.0.0"
_PROBLEM_PRODUCER = "crychic.workflow.graph_fused_problem.v1"
_WORKFLOW_PRODUCER = "crychic.workflow.graph_fused_fit.v1"
_APPLICATION_PRODUCER = "crychic.workflow.graph_fused_application.v1"
_RESPONSE_METHOD = "adjusted_balanced_one_vs_rest_context_effect_v1"
_PRECISION_METHOD = "subject_cluster_cr1_winsorized_median_v1"
_LOW_DF_PRECISION_METHOD = "equal_feature_low_df_guardrail_v1"
_APPLICATION_FUNCTIONAL_METHOD = "frozen_node_response_coordinate_v1"
_APPLICATION_LOSS_ESTIMAND = "subject_equal_full_context_weighted_prediction_loss_v1"


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(
    values: Sequence[str], *, field_name: str, minimum: int = 1
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if len(result) < minimum or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(
            f"{field_name} must contain at least {minimum} canonical names"
        )
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique values")
    return result


def _array_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    if np.any(~np.isfinite(array)):
        raise ValueError("graph-fused array digests require finite values")
    digest = hashlib.sha256()
    digest.update(canonical_json({"shape": list(array.shape)}).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sparse_matrix_equal(left: object, right: object) -> bool:
    left_matrix = cast(Any, left).tocsc(copy=True)
    right_matrix = cast(Any, right).tocsc(copy=True)
    left_matrix.sum_duplicates()
    right_matrix.sum_duplicates()
    left_matrix.eliminate_zeros()
    right_matrix.eliminate_zeros()
    left_matrix.sort_indices()
    right_matrix.sort_indices()
    return bool(
        left_matrix.shape == right_matrix.shape
        and np.array_equal(left_matrix.indptr, right_matrix.indptr)
        and np.array_equal(left_matrix.indices, right_matrix.indices)
        and np.array_equal(left_matrix.data, right_matrix.data)
    )


def _immutable_vector(
    values: np.ndarray, *, length: int, field_name: str
) -> np.ndarray:
    owned = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if owned.shape != (length,) or np.any(~np.isfinite(owned)):
        raise ValueError(f"{field_name} must be a finite aligned vector")
    owned[owned == 0.0] = 0.0
    result = cast(
        np.ndarray,
        np.frombuffer(owned.tobytes(order="C"), dtype="<f8").reshape(owned.shape),
    )
    result.setflags(write=False)
    return result


def _immutable_array(
    values: np.ndarray, *, shape: tuple[int, ...], field_name: str
) -> np.ndarray:
    owned = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if owned.shape != shape or np.any(~np.isfinite(owned)):
        raise ValueError(f"{field_name} must be a finite aligned array")
    owned[owned == 0.0] = 0.0
    result = cast(
        np.ndarray,
        np.frombuffer(owned.tobytes(order="C"), dtype="<f8").reshape(owned.shape),
    )
    result.setflags(write=False)
    return result


def _is_immutable_byte_backed(values: np.ndarray) -> bool:
    if values.flags.writeable or not values.flags.c_contiguous:
        return False
    base: object = values
    while isinstance(base, np.ndarray):
        base = base.base
    return isinstance(base, bytes)


def _graph_id(graph: ContextGraph) -> str:
    identifier: str = stable_id("context_graph", graph.to_dict(), schema_version="1")
    return identifier


def _context_ids(graph: ContextGraph, context_keys: tuple[str, ...]) -> tuple[str, ...]:
    identifiers = tuple(
        node_context_fields(node, context_keys)[0] for node in graph.nodes
    )
    if len(set(identifiers)) != len(identifiers):
        raise ContractError(
            "Context graph nodes collapse to duplicate normalized context IDs",
            code="graph_fused_context_id_collision",
            field="graph.nodes",
            remediation="Use graph nodes compatible with the frozen context keys",
        )
    return identifiers


def _family_parent_payload(
    graph: ContextGraph,
    parents: Mapping[Hashable, ReceiverFamilyTrainingArtifact],
    *,
    receiver: str,
    fold_id: str,
    training_subject_ids: tuple[str, ...],
) -> tuple[
    tuple[ReceiverFamilyTrainingArtifact, ...],
    tuple[FamilyFirstBasis, ...],
    tuple[str, ...],
]:
    nodes = tuple(graph.nodes)
    if set(parents) != set(nodes):
        raise ContractError(
            "Receiver-family parents must exactly cover the frozen context graph",
            code="graph_fused_family_context_mismatch",
            field="family_parents",
            remediation="Provide one training-only family parent for every graph node",
        )
    ordered = tuple(parents[node] for node in nodes)
    if any(
        not isinstance(parent, ReceiverFamilyTrainingArtifact) for parent in ordered
    ):
        raise TypeError(
            "family_parents must contain ReceiverFamilyTrainingArtifact values"
        )
    for parent in ordered:
        parent._require_producer_owned()
    if any(
        parent.receiver != receiver
        or parent.fold_id != fold_id
        or parent.training_subject_ids != training_subject_ids
        for parent in ordered
    ):
        raise ContractError(
            "Receiver-family parents are outside the declared training scope",
            code="graph_fused_family_parent_scope_mismatch",
            field="family_parents",
            remediation="Refit every family parent on this exact training subject set",
        )
    bases = tuple(parent.family_basis for parent in ordered)
    reference = bases[0]
    if any(
        basis.feature_ids != reference.feature_ids
        or basis.family_ids != reference.family_ids
        or basis.family_definitions != reference.family_definitions
        for basis in bases[1:]
    ):
        raise ContractError(
            "Graph-fused family parents do not share one feature/family universe",
            code="graph_fused_family_axis_mismatch",
            field="family_parents",
            remediation=(
                "Freeze a common family universe before context-specific gating"
            ),
        )
    return ordered, bases, tuple(parent.training_artifact_id for parent in ordered)


def _outer_family_axis_parent_payload(
    graph: ContextGraph,
    parents: Mapping[Hashable, ReceiverFamilyTrainingArtifact],
    *,
    receiver: str,
    outer_training_subject_ids: tuple[str, ...],
) -> tuple[ReceiverFamilyTrainingArtifact, ...]:
    """Validate root outer parents without assuming their outer-fold ID."""

    nodes = tuple(graph.nodes)
    if set(parents) != set(nodes):
        raise ContractError(
            "Outer receiver-family parents must exactly cover the context graph",
            code="graph_fused_family_context_mismatch",
            field="family_parents",
            remediation="Provide one root outer family parent for every graph node",
        )
    ordered = tuple(parents[node] for node in nodes)
    if any(
        not isinstance(parent, ReceiverFamilyTrainingArtifact) for parent in ordered
    ):
        raise TypeError(
            "family_parents must contain ReceiverFamilyTrainingArtifact values"
        )
    for parent in ordered:
        parent._require_producer_owned()
    if any(
        parent.receiver != receiver
        or parent.training_subject_ids != outer_training_subject_ids
        or parent.frozen_family_axis_parent_id is not None
        for parent in ordered
    ):
        raise ContractError(
            "Frozen-axis refits require root parents from the exact outer scope",
            code="graph_fused_inner_family_refit_parent_mismatch",
            field="family_parents",
            remediation="Use the root family parents fitted on this outer fold",
        )
    if len({parent.fold_id for parent in ordered}) != 1:
        raise ContractError(
            "Outer receiver-family parents use different fold identities",
            code="graph_fused_inner_family_refit_parent_mismatch",
            field="family_parents.fold_id",
            remediation="Use parents produced inside one outer training fold",
        )
    bases = tuple(parent.family_basis for parent in ordered)
    reference = bases[0]
    if any(
        basis.feature_ids != reference.feature_ids
        or basis.family_ids != reference.family_ids
        or basis.family_definitions != reference.family_definitions
        or basis.medoid_driver_ids != reference.medoid_driver_ids
        for basis in bases[1:]
    ):
        raise ContractError(
            "Outer parents do not share one frozen family axis",
            code="graph_fused_family_axis_mismatch",
            field="family_parents",
            remediation="Freeze one family definition before context-specific gating",
        )
    return ordered


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphFusedWorkflowSpec:
    """Frozen tuning and numerical policy for the experimental workflow bridge."""

    tuning_spec: GraphPenaltyTuningSpec
    precision_lower_quantile: float = 0.05
    precision_upper_quantile: float = 0.95
    min_positive_precision_features: int = 2
    low_df_threshold: int = 4
    solver_tolerance: float = 1.0e-7
    max_iterations: int = 1_000
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tuning_spec, GraphPenaltyTuningSpec):
            raise TypeError("tuning_spec must be a GraphPenaltyTuningSpec")
        self.tuning_spec._require_intact()
        lower = float(self.precision_lower_quantile)
        upper = float(self.precision_upper_quantile)
        if not 0 <= lower <= upper <= 1:
            raise ValueError(
                "precision quantiles must satisfy 0 <= lower <= upper <= 1"
            )
        for field_name in (
            "min_positive_precision_features",
            "low_df_threshold",
            "max_iterations",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be an integer >= 1")
        tolerance = float(self.solver_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("solver_tolerance must be finite and positive")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        payload = {
            "experimental": True,
            "formal_inference_allowed": False,
            "low_df_threshold": self.low_df_threshold,
            "max_iterations": self.max_iterations,
            "min_positive_precision_features": self.min_positive_precision_features,
            "precision_lower_quantile": lower,
            "precision_upper_quantile": upper,
            "response_method": _RESPONSE_METHOD,
            "schema_version": self.schema_version,
            "solver_tolerance": tolerance,
            "tuning_spec_id": self.tuning_spec.spec_id,
        }
        object.__setattr__(self, "precision_lower_quantile", lower)
        object.__setattr__(self, "precision_upper_quantile", upper)
        object.__setattr__(self, "solver_tolerance", tolerance)
        object.__setattr__(
            self,
            "spec_id",
            stable_id("graph_fused_workflow_spec", payload, schema_version="1"),
        )

    def _require_intact(self) -> None:
        try:
            repeated = GraphFusedWorkflowSpec(
                tuning_spec=self.tuning_spec,
                precision_lower_quantile=self.precision_lower_quantile,
                precision_upper_quantile=self.precision_upper_quantile,
                min_positive_precision_features=self.min_positive_precision_features,
                low_df_threshold=self.low_df_threshold,
                solver_tolerance=self.solver_tolerance,
                max_iterations=self.max_iterations,
                schema_version=self.schema_version,
            )
            valid = self.spec_id == repeated.spec_id
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Graph-fused workflow specification identity is not intact",
                code="graph_fused_workflow_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the workflow specification",
            ) from error
        if not valid:
            raise ContractError(
                "Graph-fused workflow specification identity is not intact",
                code="graph_fused_workflow_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the workflow specification",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "tuning_spec": self.tuning_spec.to_dict(),
            "precision_lower_quantile": self.precision_lower_quantile,
            "precision_upper_quantile": self.precision_upper_quantile,
            "min_positive_precision_features": self.min_positive_precision_features,
            "low_df_threshold": self.low_df_threshold,
            "solver_tolerance": self.solver_tolerance,
            "max_iterations": self.max_iterations,
            "response_method": _RESPONSE_METHOD,
            "experimental": True,
            "formal_inference_allowed": False,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphFusedInnerFoldParents:
    """One inner split plus either legacy or frozen-axis family parents."""

    fold_id: str
    graph_id: str
    training_subject_ids: tuple[str, ...]
    validation_subject_ids: tuple[str, ...]
    family_parents: tuple[ReceiverFamilyTrainingArtifact, ...]
    parent_set_id: str
    family_axis_refits: tuple[ReceiverFamilyAxisRefitResult, ...] = ()
    status: str = "observed"
    reason_code: str | None = None

    @property
    def uses_frozen_family_axis_refits(self) -> bool:
        """Whether the parents were refitted on an immutable outer axis."""

        return bool(self.family_axis_refits)

    def _identity_payload(
        self,
        *,
        parent_ids: tuple[str, ...],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "family_parent_ids": list(parent_ids),
            "fold_id": self.fold_id,
            "graph_id": self.graph_id,
            "training_subject_ids": list(self.training_subject_ids),
            "validation_subject_ids": list(self.validation_subject_ids),
        }
        # Keep the legacy payload byte-for-byte stable for old callers and bundles.
        if self.family_axis_refits:
            payload.update(
                {
                    "family_axis_refit_ids": [
                        refit.refit_id for refit in self.family_axis_refits
                    ],
                    "reason_code": self.reason_code,
                    "status": self.status,
                }
            )
        return payload

    def _require_intact(
        self,
        graph: ContextGraph,
        receiver: str,
        *,
        outer_parent_ids: tuple[str, ...] | None = None,
    ) -> None:
        graph_id = _graph_id(graph)
        if graph_id != self.graph_id:
            raise ContractError(
                "Inner family parents use a different context graph",
                code="graph_fused_inner_graph_mismatch",
                field="graph_id",
                remediation="Freeze every inner parent against the outer graph",
            )
        if self.family_axis_refits:
            if len(self.family_axis_refits) != len(graph.nodes):
                raise ContractError(
                    "Frozen-axis refits do not cover the complete context graph",
                    code="graph_fused_inner_parent_integrity_violation",
                    field="family_axis_refits",
                    remediation="Repeat every node-specific inner family refit",
                )
            for refit in self.family_axis_refits:
                refit._require_producer_owned()
            expected_scope = (
                receiver,
                self.fold_id,
                self.training_subject_ids,
                self.validation_subject_ids,
            )
            if any(
                (
                    refit.receiver,
                    refit.fold_id,
                    refit.training_subject_ids,
                    refit.validation_subject_ids,
                )
                != expected_scope
                for refit in self.family_axis_refits
            ):
                raise ContractError(
                    "Frozen-axis refits use a different inner subject scope",
                    code="graph_fused_inner_parent_integrity_violation",
                    field="family_axis_refits",
                    remediation="Repeat refits on this exact inner split",
                )
            refit_parent_ids = tuple(
                refit.outer_parent_id for refit in self.family_axis_refits
            )
            if outer_parent_ids is not None and refit_parent_ids != outer_parent_ids:
                raise ContractError(
                    "Inner family refits derive from different outer parents",
                    code="graph_fused_inner_family_refit_parent_mismatch",
                    field="family_axis_refits.outer_parent_id",
                    remediation="Refit from the exact parents in the outer problem",
                )
            observed_artifacts = tuple(
                refit.artifact
                for refit in self.family_axis_refits
                if refit.artifact is not None
            )
            all_observed = len(observed_artifacts) == len(self.family_axis_refits)
            if all_observed:
                if (
                    self.status != "observed"
                    or self.reason_code is not None
                    or self.family_parents != observed_artifacts
                ):
                    raise ContractError(
                        "Observed inner family-refit state is inconsistent",
                        code="graph_fused_inner_parent_integrity_violation",
                        field="status,family_parents",
                        remediation="Refreeze the inner family parents",
                    )
                mapping = dict(zip(graph.nodes, self.family_parents, strict=True))
                _, _, parent_ids = _family_parent_payload(
                    graph,
                    mapping,
                    receiver=receiver,
                    fold_id=self.fold_id,
                    training_subject_ids=self.training_subject_ids,
                )
            else:
                expected_reason = _inner_family_refit_reason(self.family_axis_refits)
                if (
                    self.status != "not_estimable"
                    or self.reason_code != expected_reason
                    or self.family_parents
                ):
                    raise ContractError(
                        "Non-estimable inner family-refit state is inconsistent",
                        code="graph_fused_inner_parent_integrity_violation",
                        field="status,reason_code,family_parents",
                        remediation="Refreeze the inner family parents",
                    )
                parent_ids = ()
        else:
            if self.status != "observed" or self.reason_code is not None:
                raise ContractError(
                    "Legacy inner family parents have an invalid state",
                    code="graph_fused_inner_parent_integrity_violation",
                    field="status,reason_code",
                    remediation="Refreeze the inner family parents",
                )
            mapping = dict(zip(graph.nodes, self.family_parents, strict=True))
            _, _, parent_ids = _family_parent_payload(
                graph,
                mapping,
                receiver=receiver,
                fold_id=self.fold_id,
                training_subject_ids=self.training_subject_ids,
            )
        expected = stable_id(
            "graph_fused_inner_fold_parents",
            self._identity_payload(parent_ids=parent_ids),
            schema_version="1",
        )
        if expected != self.parent_set_id:
            raise ContractError(
                "Inner family-parent identity is not intact",
                code="graph_fused_inner_parent_integrity_violation",
                field="parent_set_id",
                remediation="Refreeze the inner family parents",
            )


def _inner_family_refit_reason(
    refits: tuple[ReceiverFamilyAxisRefitResult, ...],
) -> str:
    reasons = tuple(
        sorted(
            {
                str(refit.reason_code)
                for refit in refits
                if refit.status != "observed" and refit.reason_code is not None
            }
        )
    )
    if not reasons:
        raise ValueError("non-estimable family refits require a reason code")
    return "graph_fused_inner_family_refit_not_estimable:" + ",".join(reasons)


def freeze_graph_fused_inner_fold_parents(
    *,
    fold_id: str,
    graph: ContextGraph,
    receiver: str,
    training_subject_ids: Sequence[str],
    validation_subject_ids: Sequence[str],
    family_parents: Mapping[Hashable, ReceiverFamilyTrainingArtifact],
    inner_training_availability: Mapping[Hashable, BatchAvailability] | None = None,
    target_prior: TargetPrior | None = None,
) -> GraphFusedInnerFoldParents:
    """Bind an inner split, optionally refitting gates on a frozen outer axis.

    Supplying both ``inner_training_availability`` and ``target_prior`` selects
    the strict path: ``family_parents`` are root outer-fold parents and every
    graph node is refitted on only the inner-training subjects.  Omitting both
    retains the legacy pre-refitted-parent contract and its stable identity.
    """

    fold = _identifier(fold_id, field_name="fold_id")
    receiver_name = _identifier(receiver, field_name="receiver")
    training = tuple(
        sorted(
            _names(training_subject_ids, field_name="training_subject_ids", minimum=1)
        )
    )
    validation = tuple(
        sorted(_names(validation_subject_ids, field_name="validation_subject_ids"))
    )
    if set(training).intersection(validation):
        raise ContractError(
            "Inner graph-fused training and validation subjects overlap",
            code="graph_fused_inner_subject_leakage",
            field="training_subject_ids,validation_subject_ids",
            remediation="Use disjoint subject-blocked inner folds",
        )
    if (inner_training_availability is None) != (target_prior is None):
        raise ValueError(
            "inner_training_availability and target_prior must be supplied together"
        )
    if inner_training_availability is None and len(training) < 2:
        raise ValueError("training_subject_ids must contain at least 2 canonical names")
    if inner_training_availability is not None and not isinstance(
        inner_training_availability, Mapping
    ):
        raise TypeError("inner_training_availability must be a mapping")
    graph_id = _graph_id(graph)
    refits: tuple[ReceiverFamilyAxisRefitResult, ...] = ()
    status = "observed"
    reason_code: str | None = None
    if inner_training_availability is None:
        ordered, _, parent_ids = _family_parent_payload(
            graph,
            family_parents,
            receiver=receiver_name,
            fold_id=fold,
            training_subject_ids=training,
        )
    else:
        assert target_prior is not None
        if not isinstance(target_prior, TargetPrior):
            raise TypeError("target_prior must be a TargetPrior")
        nodes = tuple(graph.nodes)
        if set(inner_training_availability) != set(nodes):
            raise ContractError(
                "Inner availability must exactly cover the context graph",
                code="graph_fused_inner_family_availability_mismatch",
                field="inner_training_availability",
                remediation="Provide one inner-training availability per graph node",
            )
        outer_subjects = tuple(sorted((*training, *validation)))
        outer_parents = _outer_family_axis_parent_payload(
            graph,
            family_parents,
            receiver=receiver_name,
            outer_training_subject_ids=outer_subjects,
        )
        refits = tuple(
            refit_receiver_family_training_artifact_on_frozen_axis(
                inner_training_availability[node],
                target_prior,
                outer_parent,
                fold_id=fold,
                inner_training_subject_ids=training,
                inner_validation_subject_ids=validation,
            )
            for node, outer_parent in zip(nodes, outer_parents, strict=True)
        )
        if all(refit.artifact is not None for refit in refits):
            ordered = tuple(
                cast(ReceiverFamilyTrainingArtifact, refit.artifact) for refit in refits
            )
            parent_ids = tuple(parent.training_artifact_id for parent in ordered)
        else:
            ordered = ()
            parent_ids = ()
            status = "not_estimable"
            reason_code = _inner_family_refit_reason(refits)
    temporary = GraphFusedInnerFoldParents(
        fold_id=fold,
        graph_id=graph_id,
        training_subject_ids=training,
        validation_subject_ids=validation,
        family_parents=ordered,
        parent_set_id="pending",
        family_axis_refits=refits,
        status=status,
        reason_code=reason_code,
    )
    result = GraphFusedInnerFoldParents(
        fold_id=fold,
        graph_id=graph_id,
        training_subject_ids=training,
        validation_subject_ids=validation,
        family_parents=ordered,
        parent_set_id=stable_id(
            "graph_fused_inner_fold_parents",
            temporary._identity_payload(parent_ids=parent_ids),
            schema_version="1",
        ),
        family_axis_refits=refits,
        status=status,
        reason_code=reason_code,
    )
    result._require_intact(graph, receiver_name)
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphFusedInnerPartition:
    """Complete inner partition of one outer-training subject universe."""

    outer_training_subject_ids: tuple[str, ...]
    folds: tuple[GraphFusedInnerFoldParents, ...]
    partition_id: str = field(init=False)

    def __post_init__(self) -> None:
        outer = tuple(
            sorted(
                _names(
                    self.outer_training_subject_ids,
                    field_name="outer_training_subject_ids",
                    minimum=4,
                )
            )
        )
        folds = tuple(self.folds)
        if len(folds) < 2 or any(
            not isinstance(fold, GraphFusedInnerFoldParents) for fold in folds
        ):
            raise ValueError(
                "folds must contain at least two GraphFusedInnerFoldParents values"
            )
        validation = tuple(
            subject for fold in folds for subject in fold.validation_subject_ids
        )
        if len(set(validation)) != len(validation) or set(validation) != set(outer):
            raise ContractError(
                "Inner validation folds must partition every outer-training "
                "subject once",
                code="graph_fused_inner_partition_mismatch",
                field="validation_subject_ids",
                remediation="Create a complete disjoint inner subject partition",
            )
        if any(
            set(fold.training_subject_ids).union(fold.validation_subject_ids)
            != set(outer)
            for fold in folds
        ):
            raise ContractError(
                "Every inner fold must partition the same outer-training subjects",
                code="graph_fused_inner_partition_mismatch",
                field="training_subject_ids,validation_subject_ids",
                remediation="Build all inner folds from one outer-training scope",
            )
        if len({fold.fold_id for fold in folds}) != len(folds):
            raise ValueError("inner fold IDs must be unique")
        if len({fold.uses_frozen_family_axis_refits for fold in folds}) != 1:
            raise ContractError(
                "Inner folds mix legacy parents and frozen-axis refits",
                code="graph_fused_inner_family_refit_mode_mismatch",
                field="folds.family_axis_refits",
                remediation=(
                    "Build every inner fold with the same family-refit contract"
                ),
            )
        canonical_folds = tuple(sorted(folds, key=lambda fold: fold.fold_id))
        object.__setattr__(self, "outer_training_subject_ids", outer)
        object.__setattr__(self, "folds", canonical_folds)
        object.__setattr__(
            self,
            "partition_id",
            stable_id(
                "graph_fused_inner_partition",
                {
                    "outer_training_subject_ids": list(outer),
                    "parent_set_ids": [fold.parent_set_id for fold in canonical_folds],
                },
                schema_version="1",
            ),
        )

    def _require_intact(
        self,
        graph: ContextGraph,
        receiver: str,
        *,
        outer_parent_ids: tuple[str, ...],
    ) -> None:
        try:
            repeated = GraphFusedInnerPartition(
                outer_training_subject_ids=self.outer_training_subject_ids,
                folds=self.folds,
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Graph-fused inner partition identity is not intact",
                code="graph_fused_inner_partition_integrity_violation",
                field="partition_id",
                remediation="Rebuild the partition from intact inner family parents",
            ) from error
        for fold in self.folds:
            fold._require_intact(
                graph,
                receiver,
                outer_parent_ids=outer_parent_ids,
            )
        if (
            repeated.outer_training_subject_ids != self.outer_training_subject_ids
            or repeated.folds != self.folds
            or repeated.partition_id != self.partition_id
        ):
            raise ContractError(
                "Graph-fused inner partition identity is not intact",
                code="graph_fused_inner_partition_integrity_violation",
                field="partition_id",
                remediation="Rebuild the partition from intact inner family parents",
            )


@dataclass(frozen=True, slots=True, init=False)
class GraphFusedTrainingProblem:
    """Compact context-by-feature problem frozen inside one outer fold."""

    receiver: str
    fold_id: str
    training_input_digest: str
    graph: ContextGraph
    graph_id: str
    context_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    heldout_subject_ids: tuple[str, ...]
    design_encoder_id: str
    response_artifact_id: str
    family_parent_ids: tuple[str, ...]
    family_bases: tuple[FamilyFirstBasis, ...]
    signed_responses: tuple[np.ndarray, ...]
    precision_weights: tuple[np.ndarray, ...]
    context_subject_counts: tuple[int, ...]
    residual_df: int
    response_method: str
    precision_method: str
    response_id: str
    precision_id: str
    application_training_parent_ids: tuple[str, ...]
    application_reparameterization_inverse: np.ndarray
    application_context_directions: tuple[np.ndarray, ...]
    application_nuisance_projections: tuple[np.ndarray, ...]
    application_functional_method: str
    application_functional_id: str
    problem_id: str
    experimental: bool
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "GraphFusedTrainingProblem is producer-owned; "
            "use build_graph_fused_training_problem()"
        )

    @classmethod
    def _from_producer(cls, **values: object) -> GraphFusedTrainingProblem:
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "context_ids": list(self.context_ids),
            "context_subject_counts": list(self.context_subject_counts),
            "design_encoder_id": self.design_encoder_id,
            "experimental": True,
            "family_basis_ids": [basis.family_basis_id for basis in self.family_bases],
            "family_ids": list(self.family_ids),
            "family_parent_ids": list(self.family_parent_ids),
            "feature_ids": list(self.feature_ids),
            "fold_id": self.fold_id,
            "graph_id": self.graph_id,
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "application_functional_id": self.application_functional_id,
            "precision_id": self.precision_id,
            "precision_method": self.precision_method,
            "receiver": self.receiver,
            "residual_df": self.residual_df,
            "response_artifact_id": self.response_artifact_id,
            "response_id": self.response_id,
            "response_method": self.response_method,
            "training_input_digest": self.training_input_digest,
            "training_subject_ids": list(self.training_subject_ids),
        }

    def _require_intact(self) -> None:
        try:
            nodes = tuple(self.graph.nodes)
            shape = (len(self.feature_ids),)
            design_width = self.application_reparameterization_inverse.shape[0]
            expected_application_id = stable_id(
                "graph_fused_application_functional",
                {
                    "context_ids": list(self.context_ids),
                    "context_direction_digests": [
                        _array_digest(values)
                        for values in self.application_context_directions
                    ],
                    "design_encoder_id": self.design_encoder_id,
                    "method": self.application_functional_method,
                    "nuisance_projection_digests": [
                        _array_digest(values)
                        for values in self.application_nuisance_projections
                    ],
                    "reparameterization_inverse_digest": _array_digest(
                        self.application_reparameterization_inverse
                    ),
                    "response_artifact_id": self.response_artifact_id,
                    "training_parent_ids": list(self.application_training_parent_ids),
                },
                schema_version="1",
            )
            valid = (
                self._producer_marker == _PROBLEM_PRODUCER
                and self.experimental
                and self.graph_id == _graph_id(self.graph)
                and len(nodes) == len(self.context_ids)
                and len(nodes) == len(self.family_bases)
                and len(nodes) == len(self.signed_responses)
                and len(nodes) == len(self.precision_weights)
                and len(nodes) == len(self.context_subject_counts)
                and len(self.application_training_parent_ids) == 2 * len(nodes)
                and len(self.application_context_directions) == len(nodes)
                and len(self.application_nuisance_projections) == len(nodes)
                and design_width >= 1
                and self.application_reparameterization_inverse.shape
                == (design_width, design_width)
                and all(
                    values.shape == (design_width,)
                    for values in self.application_context_directions
                )
                and all(
                    values.shape == (design_width, len(self.feature_ids))
                    for values in self.application_nuisance_projections
                )
                and all(values.shape == shape for values in self.signed_responses)
                and all(values.shape == shape for values in self.precision_weights)
                and all(
                    _is_immutable_byte_backed(values)
                    for values in self.signed_responses
                )
                and all(
                    _is_immutable_byte_backed(values)
                    for values in self.precision_weights
                )
                and _is_immutable_byte_backed(
                    self.application_reparameterization_inverse
                )
                and all(
                    _is_immutable_byte_backed(values)
                    for values in self.application_context_directions
                )
                and all(
                    _is_immutable_byte_backed(values)
                    for values in self.application_nuisance_projections
                )
                and all(
                    np.all(values >= 0) and np.any(values > 0)
                    for values in self.precision_weights
                )
                and self.response_id
                == stable_id(
                    "graph_fused_context_response",
                    {
                        "context_ids": list(self.context_ids),
                        "method": self.response_method,
                        "response_artifact_id": self.response_artifact_id,
                        "response_digests": [
                            _array_digest(values) for values in self.signed_responses
                        ],
                    },
                    schema_version="1",
                )
                and self.precision_id
                == stable_id(
                    "graph_fused_context_precision",
                    {
                        "context_ids": list(self.context_ids),
                        "method": self.precision_method,
                        "precision_digests": [
                            _array_digest(values) for values in self.precision_weights
                        ],
                        "response_id": self.response_id,
                    },
                    schema_version="1",
                )
                and self.application_functional_method == _APPLICATION_FUNCTIONAL_METHOD
                and self.application_functional_id == expected_application_id
                and self.problem_id
                == stable_id(
                    "graph_fused_training_problem",
                    self._identity_payload(),
                    schema_version="1",
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Graph-fused training problem failed integrity validation",
                code="graph_fused_problem_integrity_violation",
                field="problem_id",
                remediation="Rebuild the compact problem from training parents",
            ) from error
        if not valid:
            raise ContractError(
                "Graph-fused training problem failed integrity validation",
                code="graph_fused_problem_integrity_violation",
                field="problem_id",
                remediation="Rebuild the compact problem from training parents",
            )

    def basis_mapping(self) -> dict[Hashable, FamilyFirstBasis]:
        self._require_intact()
        return dict(zip(self.graph.nodes, self.family_bases, strict=True))

    def response_mapping(self) -> dict[Hashable, np.ndarray]:
        self._require_intact()
        return dict(zip(self.graph.nodes, self.signed_responses, strict=True))

    def precision_mapping(self) -> dict[Hashable, np.ndarray]:
        self._require_intact()
        return dict(zip(self.graph.nodes, self.precision_weights, strict=True))

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "problem_id": self.problem_id,
            "receiver": self.receiver,
            "fold_id": self.fold_id,
            "graph_id": self.graph_id,
            "context_ids": list(self.context_ids),
            "feature_ids": list(self.feature_ids),
            "family_ids": list(self.family_ids),
            "training_subject_ids": list(self.training_subject_ids),
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "design_encoder_id": self.design_encoder_id,
            "response_artifact_id": self.response_artifact_id,
            "family_parent_ids": list(self.family_parent_ids),
            "family_basis_ids": [basis.family_basis_id for basis in self.family_bases],
            "context_subject_counts": list(self.context_subject_counts),
            "residual_df": self.residual_df,
            "response_method": self.response_method,
            "precision_method": self.precision_method,
            "response_id": self.response_id,
            "precision_id": self.precision_id,
            "application_training_parent_ids": list(
                self.application_training_parent_ids
            ),
            "application_reparameterization_inverse_digest": _array_digest(
                self.application_reparameterization_inverse
            ),
            "application_context_direction_digests": [
                _array_digest(values) for values in self.application_context_directions
            ],
            "application_nuisance_projection_digests": [
                _array_digest(values)
                for values in self.application_nuisance_projections
            ],
            "application_functional_method": self.application_functional_method,
            "application_functional_id": self.application_functional_id,
            "experimental": True,
            "formal_inference_allowed": False,
            "p_value": None,
            "q_value": None,
            "comm_probability": None,
        }


@dataclass(frozen=True, slots=True)
class _ContextResponseFit:
    signed_responses: tuple[np.ndarray, ...]
    precision_weights: tuple[np.ndarray, ...]
    context_subject_counts: tuple[int, ...]
    residual_df: int
    precision_method: str
    collapsed_values: np.ndarray
    collapsed_full_design: np.ndarray
    collapsed_nuisance_design: np.ndarray
    collapsed_subject_ids: tuple[str, ...]
    collapsed_context_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ApplicationFunctionalFit:
    training_parent_ids: tuple[str, ...]
    reparameterization_inverse: np.ndarray
    context_directions: tuple[np.ndarray, ...]
    nuisance_projections: tuple[np.ndarray, ...]
    functional_id: str


def _ordered_metadata(
    sample_metadata: pd.DataFrame, encoder: FrozenDesignEncoder
) -> pd.DataFrame:
    required = {
        encoder.sample_key,
        encoder.subject_key,
        *encoder.context_keys,
        *encoder.covariates,
    }
    missing = required.difference(sample_metadata.columns)
    if missing:
        raise ValueError(
            f"training sample metadata is missing columns: {sorted(missing)}"
        )
    table = sample_metadata.loc[:, list(required)].copy(deep=True)
    if table[encoder.sample_key].duplicated().any():
        raise ValueError("training sample metadata must contain one row per sample")
    by_sample = table.set_index(encoder.sample_key, drop=False)
    if set(by_sample.index.astype(str)) != set(encoder.training_sample_ids):
        raise ContractError(
            "Training metadata does not exactly match the frozen design samples",
            code="graph_fused_design_sample_mismatch",
            field="sample_metadata",
            remediation="Pass the exact outer-training sample metadata",
        )
    try:
        ordered = by_sample.loc[list(encoder.training_sample_ids)].reset_index(
            drop=True
        )
    except KeyError as error:
        raise ContractError(
            "Training metadata sample identifiers changed type or value",
            code="graph_fused_design_sample_mismatch",
            field="sample_metadata",
            remediation="Use canonical string sample IDs from the frozen encoder",
        ) from error
    return ordered


def _design_emm(
    sample_metadata: pd.DataFrame,
    encoder: FrozenDesignEncoder,
    graph: ContextGraph,
) -> tuple[np.ndarray, ...]:
    ordered = _ordered_metadata(sample_metadata, encoder)
    audit = audit_sample_design(
        ordered,
        context_keys=encoder.context_keys,
        covariates=encoder.covariates,
        categorical_covariates=encoder.categorical_covariates,
        formula=encoder.formula,
        sample_key=encoder.sample_key,
    )
    observed_design = audit.design_matrix.to_numpy(dtype=np.float64)
    if (
        not audit.ready
        or audit.column_names != encoder.formula_column_ids
        or observed_design.shape != encoder._training_full_design.shape
        or not np.allclose(
            observed_design,
            encoder._training_full_design,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ContractError(
            "Training metadata does not reproduce the frozen full design",
            code="graph_fused_design_parent_mismatch",
            field="design_encoder",
            remediation=(
                "Build the graph problem beside the intact outer design producer"
            ),
        )
    audit_ids = tuple(
        node_context_fields(node, encoder.context_keys)[0]
        for node in audit.context_nodes
    )
    if len(set(audit_ids)) != len(audit_ids):
        raise RuntimeError("audited EMM contexts do not have unique normalized IDs")
    row_by_context = dict(zip(audit_ids, audit.emm_matrix.to_numpy(), strict=True))
    graph_ids = _context_ids(graph, encoder.context_keys)
    missing = set(graph_ids).difference(row_by_context)
    if missing:
        raise ContractError(
            "Context graph contains nodes absent from the frozen EMM design",
            code="graph_fused_graph_node_missing",
            field="graph.nodes",
            remediation="Use a graph whose nodes are represented in outer training",
        )
    return tuple(
        np.asarray(row_by_context[value], dtype=np.float64) for value in graph_ids
    )


def _collapse_response_cells(
    response: FoldGeneResponseArtifact,
    encoder: FrozenDesignEncoder,
    graph: ContextGraph,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    response._require_intact()
    response.require_compatible(encoder)
    graph_contexts = set(_context_ids(graph, encoder.context_keys))
    design_position = {
        sample_id: index for index, sample_id in enumerate(encoder.training_sample_ids)
    }
    rows: list[tuple[str, str, str, int]] = []
    for response_index, (sample_id, subject_id, context_id) in enumerate(
        zip(
            response.sample_ids,
            response.sample_subject_ids,
            response.sample_context_ids,
            strict=True,
        )
    ):
        if context_id in graph_contexts:
            rows.append((subject_id, context_id, sample_id, response_index))
    if not rows:
        raise ContractError(
            "Receiver has no observed training response in the graph contexts",
            code="graph_fused_receiver_response_missing",
            field="receiver",
            remediation="Inspect receiver pseudobulk eligibility in outer training",
        )
    frame = pd.DataFrame(
        rows,
        columns=("subject_id", "context_id", "sample_id", "response_index"),
    )
    values: list[np.ndarray] = []
    full_design: list[np.ndarray] = []
    nuisance_design: list[np.ndarray] = []
    subjects: list[str] = []
    contexts: list[str] = []
    for (group_subject_id, group_context_id), group in frame.groupby(
        ["subject_id", "context_id"], observed=True, sort=True, dropna=False
    ):
        response_indices = group["response_index"].to_numpy(dtype=np.int64)
        try:
            design_indices = np.asarray(
                [design_position[str(value)] for value in group["sample_id"]],
                dtype=np.int64,
            )
        except KeyError as error:  # pragma: no cover - protected by response parent
            raise RuntimeError(
                "response sample is absent from its frozen design"
            ) from error
        values.append(response.sample_values[response_indices].mean(axis=0))
        full_design.append(encoder._training_full_design[design_indices].mean(axis=0))
        nuisance_design.append(
            encoder.training_nuisance_matrix[design_indices].mean(axis=0)
        )
        subjects.append(str(group_subject_id))
        contexts.append(str(group_context_id))
    return (
        np.vstack(values),
        np.vstack(full_design),
        np.vstack(nuisance_design),
        tuple(subjects),
        tuple(contexts),
    )


def _context_contrast_vectors(
    emm_rows: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    if len(emm_rows) < 2:
        raise ContractError(
            "Graph-fused attribution requires at least two context nodes",
            code="graph_fused_insufficient_contexts",
            field="graph.nodes",
            remediation="Use independent attribution for a single context",
        )
    return tuple(
        row
        - np.mean(
            np.vstack(
                [other for index, other in enumerate(emm_rows) if index != focal]
            ),
            axis=0,
        )
        for focal, row in enumerate(emm_rows)
    )


def _balanced_context_direction(
    encoder: FrozenDesignEncoder,
    sample_metadata: pd.DataFrame,
    graph: ContextGraph,
    *,
    focal_index: int,
) -> np.ndarray:
    """Choose an equivalent contrast direction with nonzero balanced leverage."""

    emm = np.vstack(_design_emm(sample_metadata, encoder, graph))
    context_count = len(graph.nodes)
    weights: np.ndarray = np.full(
        context_count, -1.0 / (context_count - 1), dtype=np.float64
    )
    weights[focal_index] = 1.0
    desired = weights / float(np.dot(weights, weights))
    direction: np.ndarray = np.asarray(np.linalg.pinv(emm) @ desired, dtype=np.float64)
    observed = emm @ direction
    coefficient_contrast = weights @ emm
    if not np.allclose(observed, desired, rtol=0.0, atol=1.0e-10) or not math.isclose(
        float(coefficient_contrast @ direction),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ContractError(
            "Balanced node-specific response direction is not estimable",
            code="graph_fused_inner_validation_design_not_estimable",
            field="graph.nodes",
            remediation="Use a full-rank context design for graph tuning",
        )
    return direction


def _vector_estimable(
    matrix: np.ndarray, vector: np.ndarray, *, tolerance: float = 1.0e-8
) -> bool:
    projection = np.linalg.pinv(matrix) @ matrix
    residual = vector - vector @ projection
    return bool(
        np.linalg.norm(vector) > tolerance
        and np.linalg.norm(residual)
        <= tolerance * max(1.0, float(np.linalg.norm(vector)))
    )


def _build_application_functional(
    aggregate: PseudobulkDataset,
    response: FoldGeneResponseArtifact,
    design_encoder: FrozenDesignEncoder,
    sample_metadata: pd.DataFrame,
    graph: ContextGraph,
) -> _ApplicationFunctionalFit:
    """Freeze the node coordinates used later for fit-free held-out scoring."""

    design_width = len(design_encoder.formula_column_ids)
    feature_count = len(response.feature_ids)
    reparameterization = np.column_stack(
        (design_encoder._null_basis, design_encoder._contrast_direction)
    )
    if reparameterization.shape != (design_width, design_width) or (
        np.linalg.matrix_rank(reparameterization) != design_width
    ):
        raise ContractError(
            "Outer graph design reparameterization is not invertible",
            code="graph_fused_application_design_not_estimable",
            field="design_encoder",
            remediation="Refit from a full-rank frozen outer design",
        )
    inverse = _immutable_array(
        np.linalg.inv(reparameterization),
        shape=(design_width, design_width),
        field_name="application_reparameterization_inverse",
    )
    directions: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    parent_ids: list[str] = []
    for node_index, node in enumerate(graph.nodes):
        context_id = node_context_fields(node, design_encoder.context_keys)[0]
        node_encoder = fit_frozen_design_encoder(
            sample_metadata,
            contrast=global_one_vs_rest(
                graph,
                node,
                name=f"graph_application_node:{context_id}",
            ),
            context_keys=design_encoder.context_keys,
            covariates=design_encoder.covariates,
            categorical_covariates=design_encoder.categorical_covariates,
            formula=design_encoder.formula,
            sample_key=design_encoder.sample_key,
            subject_key=design_encoder.subject_key,
        )
        node_response = fit_fold_gene_response(
            aggregate,
            node_encoder,
            receiver=response.receiver,
            fold_id=f"{response.fold_id}:application-node:{node_index}",
            training_input_digest=response.training_input_digest,
            min_subjects_per_context=2,
        )
        (
            node_values,
            node_full_design,
            node_nuisance_design,
            _,
            _,
        ) = _collapse_response_cells(node_response, node_encoder, graph)
        direction = _balanced_context_direction(
            node_encoder,
            sample_metadata,
            graph,
            focal_index=node_index,
        )
        node_regressor = np.asarray(node_full_design @ direction, dtype=np.float64)
        node_design = np.column_stack((node_nuisance_design, node_regressor))
        if np.linalg.matrix_rank(node_design) < node_design.shape[1]:
            raise ContractError(
                "Outer node response coordinate is not estimable",
                code="graph_fused_application_design_not_estimable",
                field="sample_metadata",
                remediation="Repair outer context/covariate support",
            )
        nuisance_coefficients = (np.linalg.pinv(node_design) @ node_values)[:-1]
        projection = np.asarray(
            node_encoder._null_basis @ nuisance_coefficients,
            dtype=np.float64,
        )
        directions.append(
            _immutable_vector(
                direction,
                length=design_width,
                field_name="application_context_directions",
            )
        )
        projections.append(
            _immutable_array(
                projection,
                shape=(design_width, feature_count),
                field_name="application_nuisance_projections",
            )
        )
        parent_ids.extend((node_encoder.encoder_id, node_response.artifact_id))
    context_ids = _context_ids(graph, design_encoder.context_keys)
    payload = {
        "context_ids": list(context_ids),
        "context_direction_digests": [_array_digest(values) for values in directions],
        "design_encoder_id": design_encoder.encoder_id,
        "method": _APPLICATION_FUNCTIONAL_METHOD,
        "nuisance_projection_digests": [
            _array_digest(values) for values in projections
        ],
        "reparameterization_inverse_digest": _array_digest(inverse),
        "response_artifact_id": response.artifact_id,
        "training_parent_ids": parent_ids,
    }
    return _ApplicationFunctionalFit(
        training_parent_ids=tuple(parent_ids),
        reparameterization_inverse=inverse,
        context_directions=tuple(directions),
        nuisance_projections=tuple(projections),
        functional_id=stable_id(
            "graph_fused_application_functional", payload, schema_version="1"
        ),
    )


def _normalize_precision(
    raw: np.ndarray,
    *,
    residual_df: int,
    spec: GraphFusedWorkflowSpec,
) -> tuple[np.ndarray, str]:
    valid = np.isfinite(raw) & (raw > 0)
    if (
        residual_df <= spec.low_df_threshold
        or int(np.count_nonzero(valid)) < spec.min_positive_precision_features
    ):
        return np.ones(raw.shape, dtype=np.float64), _LOW_DF_PRECISION_METHOD
    result = np.zeros(raw.shape, dtype=np.float64)
    bounds = np.asarray(
        np.quantile(
            raw[valid],
            [spec.precision_lower_quantile, spec.precision_upper_quantile],
        ),
        dtype=np.float64,
    )
    lower = float(bounds[0])
    upper = float(bounds[1])
    clipped = np.clip(raw[valid], lower, upper)
    median = float(np.median(clipped))
    if not math.isfinite(median) or median <= 0:
        return np.ones(raw.shape, dtype=np.float64), _LOW_DF_PRECISION_METHOD
    result[valid] = clipped / median
    return result, _PRECISION_METHOD


def _fit_context_responses(
    response: FoldGeneResponseArtifact,
    encoder: FrozenDesignEncoder,
    sample_metadata: pd.DataFrame,
    graph: ContextGraph,
    spec: GraphFusedWorkflowSpec,
) -> _ContextResponseFit:
    emm_rows = _design_emm(sample_metadata, encoder, graph)
    contrasts = _context_contrast_vectors(emm_rows)
    (
        values,
        full_design,
        nuisance_design,
        subjects,
        contexts,
    ) = _collapse_response_cells(response, encoder, graph)
    rank = int(np.linalg.matrix_rank(full_design))
    residual_df = len(full_design) - rank
    n_clusters = len(set(subjects))
    if rank < full_design.shape[1] or residual_df < 1 or n_clusters < 2:
        raise ContractError(
            "Receiver-specific graph response design is not estimable",
            code="graph_fused_response_design_not_estimable",
            field="receiver_response",
            remediation=(
                "Increase receiver subject/context support or simplify the design"
            ),
        )
    if any(not _vector_estimable(full_design, vector) for vector in contrasts):
        raise ContractError(
            "One or more context response contrasts are not estimable",
            code="graph_fused_context_response_not_estimable",
            field="graph.nodes",
            remediation="Remove unsupported graph nodes or repair design confounding",
        )
    graph_context_ids = _context_ids(graph, encoder.context_keys)
    context_subject_counts = tuple(
        len(
            {
                subject
                for subject, observed_context in zip(subjects, contexts, strict=True)
                if observed_context == context_id
            }
        )
        for context_id in graph_context_ids
    )
    if any(count < 2 for count in context_subject_counts):
        raise ContractError(
            "Every graph context requires at least two receiver subjects",
            code="graph_fused_context_subject_support_insufficient",
            field="graph.nodes",
            remediation="Increase receiver support or remove the unsupported context",
        )
    coefficients = np.linalg.pinv(full_design) @ values
    residual = values - full_design @ coefficients
    bread = np.linalg.pinv(full_design.T @ full_design)
    correction = (n_clusters / (n_clusters - 1)) * (
        (len(full_design) - 1) / residual_df
    )
    signed: list[np.ndarray] = []
    precision: list[np.ndarray] = []
    precision_methods: set[str] = set()
    subject_array = np.asarray(subjects, dtype=object)
    for contrast in contrasts:
        effect = np.asarray(contrast @ coefficients, dtype=np.float64)
        row_weights = np.asarray(contrast @ bread @ full_design.T, dtype=np.float64)
        influence = np.vstack(
            [
                np.sum(
                    row_weights[subject_array == subject, None]
                    * residual[subject_array == subject],
                    axis=0,
                )
                for subject in sorted(set(subjects))
            ]
        )
        variance = correction * np.sum(np.square(influence), axis=0)
        raw_precision = np.zeros(variance.shape, dtype=np.float64)
        valid = np.isfinite(variance) & (variance > 0)
        raw_precision[valid] = 1.0 / variance[valid]
        normalized, method = _normalize_precision(
            raw_precision,
            residual_df=residual_df,
            spec=spec,
        )
        signed.append(effect)
        precision.append(normalized)
        precision_methods.add(method)
    precision_method = (
        next(iter(precision_methods))
        if len(precision_methods) == 1
        else "mixed_context_precision_guardrail_v1"
    )
    return _ContextResponseFit(
        signed_responses=tuple(signed),
        precision_weights=tuple(precision),
        context_subject_counts=context_subject_counts,
        residual_df=residual_df,
        precision_method=precision_method,
        collapsed_values=values,
        collapsed_full_design=full_design,
        collapsed_nuisance_design=nuisance_design,
        collapsed_subject_ids=subjects,
        collapsed_context_ids=contexts,
    )


def build_graph_fused_training_problem(
    aggregate: PseudobulkDataset,
    sample_metadata: pd.DataFrame,
    design_encoder: FrozenDesignEncoder,
    family_parents: Mapping[Hashable, ReceiverFamilyTrainingArtifact],
    graph: ContextGraph,
    *,
    receiver: str,
    fold_id: str,
    training_input_digest: str,
    outer_training_subject_ids: Sequence[str],
    heldout_subject_ids: Sequence[str],
    spec: GraphFusedWorkflowSpec,
) -> GraphFusedTrainingProblem:
    """Freeze a compact graph problem from physical outer-training parents."""

    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("aggregate must be an inferential PseudobulkDataset")
    if not isinstance(design_encoder, FrozenDesignEncoder):
        raise TypeError("design_encoder must be a FrozenDesignEncoder")
    design_encoder._require_producer_owned()
    if not isinstance(graph, ContextGraph):
        raise TypeError("graph must be a ContextGraph")
    if not isinstance(spec, GraphFusedWorkflowSpec):
        raise TypeError("spec must be a GraphFusedWorkflowSpec")
    spec._require_intact()
    receiver_name = _identifier(receiver, field_name="receiver")
    outer_fold = _identifier(fold_id, field_name="fold_id")
    input_digest = _identifier(
        training_input_digest, field_name="training_input_digest"
    )
    training_subjects = tuple(
        sorted(
            _names(
                outer_training_subject_ids,
                field_name="outer_training_subject_ids",
                minimum=2,
            )
        )
    )
    heldout_subjects = tuple(
        sorted(_names(heldout_subject_ids, field_name="heldout_subject_ids"))
    )
    if set(training_subjects).intersection(heldout_subjects):
        raise ContractError(
            "Outer graph-fused training and held-out subjects overlap",
            code="graph_fused_outer_subject_leakage",
            field="outer_training_subject_ids,heldout_subject_ids",
            remediation="Use the exact disjoint outer fold manifest",
        )
    if design_encoder.training_subject_ids != training_subjects:
        raise ContractError(
            "Frozen design subjects do not match the outer-training manifest",
            code="graph_fused_design_subject_mismatch",
            field="design_encoder",
            remediation="Use the design encoder fitted inside this outer fold",
        )
    _, bases, parent_ids = _family_parent_payload(
        graph,
        family_parents,
        receiver=receiver_name,
        fold_id=outer_fold,
        training_subject_ids=training_subjects,
    )
    response = fit_fold_gene_response(
        aggregate,
        design_encoder,
        receiver=receiver_name,
        fold_id=outer_fold,
        training_input_digest=input_digest,
        min_subjects_per_context=2,
    )
    response_fit = _fit_context_responses(
        response,
        design_encoder,
        sample_metadata,
        graph,
        spec,
    )
    feature_ids = bases[0].feature_ids
    if response.feature_ids != feature_ids:
        raise ContractError(
            "Receiver response and family basis feature axes differ",
            code="graph_fused_feature_axis_mismatch",
            field="feature_ids",
            remediation="Build family parents against this aggregate feature universe",
        )
    application_functional = _build_application_functional(
        aggregate,
        response,
        design_encoder,
        sample_metadata,
        graph,
    )
    context_ids = _context_ids(graph, design_encoder.context_keys)
    signed = tuple(
        _immutable_vector(
            values, length=len(feature_ids), field_name="signed_responses"
        )
        for values in response_fit.signed_responses
    )
    precision = tuple(
        _immutable_vector(
            values, length=len(feature_ids), field_name="precision_weights"
        )
        for values in response_fit.precision_weights
    )
    response_id = stable_id(
        "graph_fused_context_response",
        {
            "context_ids": list(context_ids),
            "method": _RESPONSE_METHOD,
            "response_artifact_id": response.artifact_id,
            "response_digests": [_array_digest(values) for values in signed],
        },
        schema_version="1",
    )
    precision_id = stable_id(
        "graph_fused_context_precision",
        {
            "context_ids": list(context_ids),
            "method": response_fit.precision_method,
            "precision_digests": [_array_digest(values) for values in precision],
            "response_id": response_id,
        },
        schema_version="1",
    )
    values: dict[str, object] = {
        "receiver": receiver_name,
        "fold_id": outer_fold,
        "training_input_digest": input_digest,
        "graph": graph,
        "graph_id": _graph_id(graph),
        "context_ids": context_ids,
        "feature_ids": feature_ids,
        "family_ids": bases[0].family_ids,
        "training_subject_ids": training_subjects,
        "heldout_subject_ids": heldout_subjects,
        "design_encoder_id": design_encoder.encoder_id,
        "response_artifact_id": response.artifact_id,
        "family_parent_ids": parent_ids,
        "family_bases": bases,
        "signed_responses": signed,
        "precision_weights": precision,
        "context_subject_counts": response_fit.context_subject_counts,
        "residual_df": response_fit.residual_df,
        "response_method": _RESPONSE_METHOD,
        "precision_method": response_fit.precision_method,
        "response_id": response_id,
        "precision_id": precision_id,
        "application_training_parent_ids": (application_functional.training_parent_ids),
        "application_reparameterization_inverse": (
            application_functional.reparameterization_inverse
        ),
        "application_context_directions": (application_functional.context_directions),
        "application_nuisance_projections": (
            application_functional.nuisance_projections
        ),
        "application_functional_method": _APPLICATION_FUNCTIONAL_METHOD,
        "application_functional_id": application_functional.functional_id,
        "experimental": True,
        "_producer_marker": _PROBLEM_PRODUCER,
    }
    temporary = GraphFusedTrainingProblem._from_producer(
        **values,
        problem_id="pending",
    )
    result = GraphFusedTrainingProblem._from_producer(
        **values,
        problem_id=stable_id(
            "graph_fused_training_problem",
            temporary._identity_payload(),
            schema_version="1",
        ),
    )
    result._require_intact()
    return result


def _subset_metadata(
    sample_metadata: pd.DataFrame,
    encoder: FrozenDesignEncoder,
    subject_ids: tuple[str, ...],
) -> pd.DataFrame:
    selected: pd.DataFrame = sample_metadata.loc[
        sample_metadata[encoder.subject_key].astype(str).isin(subject_ids)
    ].copy(deep=True)
    observed = tuple(sorted(selected[encoder.subject_key].astype(str).unique()))
    if observed != subject_ids:
        raise ContractError(
            "Inner metadata does not cover its declared subject scope",
            code="graph_fused_inner_metadata_scope_mismatch",
            field="sample_metadata",
            remediation="Build inner folds from the exact outer-training metadata",
        )
    return selected


def _validation_responses(
    aggregate: PseudobulkDataset,
    outer_response: FoldGeneResponseArtifact,
    inner_response_fit: _ContextResponseFit,
    design_template: FrozenDesignEncoder,
    training_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    graph: ContextGraph,
    inner_fold_id: str,
    training_input_digest: str,
    validation_subject_ids: tuple[str, ...],
) -> tuple[
    dict[str, dict[Hashable, np.ndarray]], dict[str, dict[Hashable, np.ndarray]]
]:
    response_by_sample = {
        sample_id: outer_response.sample_values[index]
        for index, sample_id in enumerate(outer_response.sample_ids)
    }
    validation_responses: dict[str, dict[Hashable, np.ndarray]] = {
        subject: {} for subject in validation_subject_ids
    }
    validation_precision: dict[str, dict[Hashable, np.ndarray]] = {
        subject: {} for subject in validation_subject_ids
    }
    for node_index, node in enumerate(graph.nodes):
        context_id = node_context_fields(node, design_template.context_keys)[0]
        node_encoder = fit_frozen_design_encoder(
            training_metadata,
            contrast=global_one_vs_rest(
                graph,
                node,
                name=f"graph_node:{context_id}",
            ),
            context_keys=design_template.context_keys,
            covariates=design_template.covariates,
            categorical_covariates=design_template.categorical_covariates,
            formula=design_template.formula,
            sample_key=design_template.sample_key,
            subject_key=design_template.subject_key,
        )
        node_training_response = fit_fold_gene_response(
            aggregate,
            node_encoder,
            receiver=outer_response.receiver,
            fold_id=f"{inner_fold_id}:node:{node_index}",
            training_input_digest=training_input_digest,
            min_subjects_per_context=2,
        )
        (
            node_values,
            node_full_design,
            node_nuisance_design,
            _,
            _,
        ) = _collapse_response_cells(node_training_response, node_encoder, graph)
        balanced_direction = _balanced_context_direction(
            node_encoder,
            training_metadata,
            graph,
            focal_index=node_index,
        )
        node_regressor = np.asarray(
            node_full_design @ balanced_direction,
            dtype=np.float64,
        )
        node_design = np.column_stack((node_nuisance_design, node_regressor))
        if np.linalg.matrix_rank(node_design) < node_design.shape[1]:
            raise ContractError(
                "Inner node-specific response coordinate is not estimable",
                code="graph_fused_inner_validation_design_not_estimable",
                field="training_metadata",
                remediation="Repair inner context/covariate support",
            )
        nuisance_coefficients = (np.linalg.pinv(node_design) @ node_values)[:-1]
        application = apply_frozen_design_encoder(node_encoder, validation_metadata)
        if application.status != "observed":
            raise ContractError(
                "Inner validation metadata cannot be encoded by the training design",
                code="graph_fused_inner_validation_design_not_estimable",
                field="validation_metadata",
                remediation="Repair unseen levels or choose a supported partition",
            )
        reparameterization = np.column_stack(
            (node_encoder._null_basis, node_encoder._contrast_direction)
        )
        transformed_application = np.column_stack(
            (application.nuisance_matrix, application.context_regressor)
        )
        application_full_design = transformed_application @ np.linalg.inv(
            reparameterization
        )
        application_regressor = application_full_design @ balanced_direction
        subject_rows: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {}
        for row_index, (sample_id, subject_id) in enumerate(
            zip(
                application.sample_ids,
                application.sample_subject_ids,
                strict=True,
            )
        ):
            values = response_by_sample.get(sample_id)
            if values is None:
                continue
            nuisance = application.nuisance_matrix[row_index]
            predicted = nuisance @ nuisance_coefficients if nuisance.size else 0.0
            subject_rows.setdefault(subject_id, []).append(
                (
                    values,
                    np.asarray(predicted, dtype=np.float64),
                    float(application_regressor[row_index]),
                )
            )
        for subject in validation_subject_ids:
            rows = subject_rows.get(subject, [])
            if not rows:
                raise ContractError(
                    "Inner validation subject has no observed receiver response",
                    code="graph_fused_inner_validation_response_missing",
                    field="validation_subject_ids",
                    remediation="Use subjects with receiver response support",
                )
            regressors = np.asarray([row[2] for row in rows], dtype=np.float64)
            denominator = float(np.dot(regressors, regressors))
            if denominator <= 1.0e-12:
                raise ContractError(
                    "Inner validation response coordinate has zero leverage",
                    code="graph_fused_inner_validation_response_not_estimable",
                    field="validation_subject_ids",
                    remediation="Choose a validation partition with context leverage",
                )
            residuals = np.vstack([row[0] - row[1] for row in rows])
            coordinate = regressors @ residuals / denominator
            validation_responses[subject][node] = np.maximum(coordinate, 0.0)
            validation_precision[subject][node] = inner_response_fit.precision_weights[
                node_index
            ]
    return validation_responses, validation_precision


def build_graph_fused_tuning_folds(
    aggregate: PseudobulkDataset,
    sample_metadata: pd.DataFrame,
    outer_design_encoder: FrozenDesignEncoder,
    outer_problem: GraphFusedTrainingProblem,
    inner_partition: GraphFusedInnerPartition,
    *,
    spec: GraphFusedWorkflowSpec,
) -> tuple[GraphTuningFold, ...]:
    """Build subject-blocked tuning folds without consulting outer held-out data."""

    outer_problem._require_intact()
    spec._require_intact()
    if inner_partition.outer_training_subject_ids != outer_problem.training_subject_ids:
        raise ContractError(
            "Inner partition does not cover the outer-training subjects",
            code="graph_fused_inner_partition_mismatch",
            field="inner_partition",
            remediation="Plan inner folds only within this outer-training scope",
        )
    outer_response = fit_fold_gene_response(
        aggregate,
        outer_design_encoder,
        receiver=outer_problem.receiver,
        fold_id=outer_problem.fold_id,
        training_input_digest=outer_problem.training_input_digest,
        min_subjects_per_context=2,
    )
    if outer_response.artifact_id != outer_problem.response_artifact_id:
        raise ContractError(
            "Outer response parent changed after the compact problem was frozen",
            code="graph_fused_response_parent_mismatch",
            field="response_artifact_id",
            remediation=(
                "Build tuning folds beside the same physical training aggregate"
            ),
        )
    tuning_folds: list[GraphTuningFold] = []
    for inner in inner_partition.folds:
        inner._require_intact(
            outer_problem.graph,
            outer_problem.receiver,
            outer_parent_ids=outer_problem.family_parent_ids,
        )
        if inner.status != "observed":
            raise ContractError(
                "Inner family gates are not estimable on the frozen outer axis",
                code="graph_fused_inner_family_refit_not_estimable",
                field="inner_partition",
                remediation="Use inner folds with sufficient receiver evidence",
            )
        training_metadata = _subset_metadata(
            sample_metadata,
            outer_design_encoder,
            inner.training_subject_ids,
        )
        validation_metadata = _subset_metadata(
            sample_metadata,
            outer_design_encoder,
            inner.validation_subject_ids,
        )
        inner_encoder = fit_frozen_design_encoder(
            training_metadata,
            contrast=outer_design_encoder.contrast,
            context_keys=outer_design_encoder.context_keys,
            covariates=outer_design_encoder.covariates,
            categorical_covariates=outer_design_encoder.categorical_covariates,
            formula=outer_design_encoder.formula,
            sample_key=outer_design_encoder.sample_key,
            subject_key=outer_design_encoder.subject_key,
        )
        inner_response = fit_fold_gene_response(
            aggregate,
            inner_encoder,
            receiver=outer_problem.receiver,
            fold_id=inner.fold_id,
            training_input_digest=outer_problem.training_input_digest,
            min_subjects_per_context=2,
        )
        inner_fit = _fit_context_responses(
            inner_response,
            inner_encoder,
            training_metadata,
            outer_problem.graph,
            spec,
        )
        mapping = dict(
            zip(outer_problem.graph.nodes, inner.family_parents, strict=True)
        )
        _, inner_bases, _ = _family_parent_payload(
            outer_problem.graph,
            mapping,
            receiver=outer_problem.receiver,
            fold_id=inner.fold_id,
            training_subject_ids=inner.training_subject_ids,
        )
        reference = inner_bases[0]
        if (
            reference.feature_ids != outer_problem.feature_ids
            or reference.family_ids != outer_problem.family_ids
            or any(
                basis.family_definitions
                != outer_problem.family_bases[0].family_definitions
                for basis in inner_bases
            )
        ):
            raise ContractError(
                "Inner and outer graph-fused family axes differ",
                code="graph_fused_inner_family_axis_mismatch",
                field="family_parents",
                remediation="Freeze a common family definition before nested tuning",
            )
        validation_responses, validation_precision = _validation_responses(
            aggregate,
            outer_response,
            inner_fit,
            outer_design_encoder,
            training_metadata,
            validation_metadata,
            outer_problem.graph,
            inner.fold_id,
            outer_problem.training_input_digest,
            inner.validation_subject_ids,
        )
        nodes = tuple(outer_problem.graph.nodes)
        tuning_folds.append(
            GraphTuningFold.from_mappings(
                fold_id=inner.fold_id,
                graph=outer_problem.graph,
                feature_ids=outer_problem.feature_ids,
                family_ids=outer_problem.family_ids,
                training_subject_ids=inner.training_subject_ids,
                matrices={
                    node: basis.matrix
                    for node, basis in zip(nodes, inner_bases, strict=True)
                },
                training_responses={
                    node: np.maximum(values, 0.0)
                    for node, values in zip(
                        nodes, inner_fit.signed_responses, strict=True
                    )
                },
                training_precision_weights={
                    node: values
                    for node, values in zip(
                        nodes, inner_fit.precision_weights, strict=True
                    )
                },
                validation_responses=validation_responses,
                validation_precision_weights=validation_precision,
                family_parent_set_id=inner.parent_set_id,
            )
        )
    return tuple(tuning_folds)


@dataclass(frozen=True, slots=True, init=False)
class GraphFusedWorkflowArtifact:
    """Experimental outer-fold fit with complete nested-training lineage."""

    workflow_id: str
    spec: GraphFusedWorkflowSpec
    problem: GraphFusedTrainingProblem
    inner_partition_id: str
    tuning_fold_data_ids: tuple[str, ...]
    fit: TunedGraphFusedFamilyAttribution | None
    status: str
    reason_code: str | None
    experimental: bool
    _inner_partition: GraphFusedInnerPartition = field(repr=False)
    _tuning_folds: tuple[GraphTuningFold, ...] = field(repr=False)
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "GraphFusedWorkflowArtifact is producer-owned; use "
            "fit_graph_fused_workflow()"
        )

    @classmethod
    def _from_producer(cls, **values: object) -> GraphFusedWorkflowArtifact:
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    @property
    def formal_inference_allowed(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "experimental": True,
            "fit_id": None if self.fit is None else self.fit.tuned_attribution_id,
            "formal_inference_allowed": False,
            "inner_partition_id": self.inner_partition_id,
            "problem_id": self.problem.problem_id,
            "reason_code": self.reason_code,
            "spec_id": self.spec.spec_id,
            "status": self.status,
            "tuning_fold_data_ids": list(self.tuning_fold_data_ids),
        }

    def _require_intact(self) -> None:
        self.spec._require_intact()
        self.problem._require_intact()
        if not isinstance(self._inner_partition, GraphFusedInnerPartition):
            raise ContractError(
                "Graph-fused workflow lacks its inner partition parent",
                code="graph_fused_workflow_integrity_violation",
                field="inner_partition_id",
                remediation="Refit from intact outer and inner training parents",
            )
        self._inner_partition._require_intact(
            self.problem.graph,
            self.problem.receiver,
            outer_parent_ids=self.problem.family_parent_ids,
        )
        partition_folds = self._inner_partition.folds
        tuning_folds = tuple(self._tuning_folds)
        for fold in tuning_folds:
            fold._require_intact()
        tuning_by_id = {fold.fold_id: fold for fold in tuning_folds}
        partition_by_id = {fold.fold_id: fold for fold in partition_folds}
        nested_lineage_matches = (
            self.inner_partition_id == self._inner_partition.partition_id
            and self._inner_partition.outer_training_subject_ids
            == self.problem.training_subject_ids
            and self.tuning_fold_data_ids
            == tuple(fold.fold_data_id for fold in tuning_folds)
            and len(tuning_by_id) == len(tuning_folds)
        )
        if tuning_folds:
            nested_lineage_matches = nested_lineage_matches and (
                set(tuning_by_id) == set(partition_by_id)
                and all(
                    inner.status == "observed"
                    and tuning.family_parent_set_id == inner.parent_set_id
                    and tuning.training_subject_ids == inner.training_subject_ids
                    and tuning.validation_subject_ids == inner.validation_subject_ids
                    and _graph_id(tuning.graph) == self.problem.graph_id
                    and tuning.feature_ids == self.problem.feature_ids
                    and tuning.family_ids == self.problem.family_ids
                    and len(tuning.matrices) == len(inner.family_parents)
                    and all(
                        _sparse_matrix_equal(matrix, parent.family_basis.matrix)
                        for matrix, parent in zip(
                            tuning.matrices,
                            inner.family_parents,
                            strict=True,
                        )
                    )
                    for fold_id, inner in partition_by_id.items()
                    for tuning in (tuning_by_id[fold_id],)
                )
            )
        else:
            nested_lineage_matches = nested_lineage_matches and (
                self.status == "not_estimable"
                and any(inner.status != "observed" for inner in partition_folds)
            )
        fit_matches_parents = True
        if self.fit is not None:
            attribution = self.fit.attribution
            tuning = self.fit.tuning
            expected_predictions = tuple(
                np.asarray(basis.matrix.dot(attribution.coefficients[index])).ravel()
                for index, basis in enumerate(self.problem.family_bases)
            )
            column_norms = np.vstack(
                [
                    np.sqrt(np.asarray(basis.matrix.power(2).sum(axis=0)).ravel())
                    for basis in self.problem.family_bases
                ]
            )
            expected_attribution_id = stable_id(
                "graph_fused_family_attribution",
                {
                    "context_graph": self.problem.graph.to_dict(),
                    "context_nodes": list(self.problem.graph.nodes),
                    "family_basis_ids": [
                        basis.family_basis_id for basis in self.problem.family_bases
                    ],
                    "family_ids": list(self.problem.family_ids),
                    "coefficients": attribution.coefficients.tolist(),
                    "precision_weights": [
                        values.tolist() for values in self.problem.precision_weights
                    ],
                    "signed_responses": [
                        values.tolist() for values in self.problem.signed_responses
                    ],
                    "lambda1": attribution.lambda1,
                    "lambda2": attribution.lambda2,
                    "lambda_f": attribution.lambda_f,
                    "tolerance": self.spec.solver_tolerance,
                    "max_iterations": self.spec.max_iterations,
                    "backend": attribution.diagnostics.backend,
                },
                schema_version="1",
            )
            expected_tuned_id = stable_id(
                "tuned_graph_fused_family_attribution",
                {
                    "tuning_id": tuning.tuning_id,
                    "attribution_id": attribution.attribution_id,
                    "graph_id": self.problem.graph_id,
                    "selected_candidate_id": self.fit.selected_candidate_id,
                    "formal_inference_allowed": False,
                },
                schema_version="1",
            )
            fit_matches_parents = (
                self.fit.graph_id == self.problem.graph_id
                and tuning.graph_id == self.problem.graph_id
                and tuning.spec.spec_id == self.spec.tuning_spec.spec_id
                and tuning.fold_data_ids == self.tuning_fold_data_ids
                and tuning.training_subject_ids == self.problem.training_subject_ids
                and attribution.context_nodes == tuple(self.problem.graph.nodes)
                and attribution.feature_ids == self.problem.feature_ids
                and attribution.family_ids == self.problem.family_ids
                and attribution.family_basis_ids
                == tuple(basis.family_basis_id for basis in self.problem.family_bases)
                and attribution.attribution_id == expected_attribution_id
                and self.fit.tuned_attribution_id == expected_tuned_id
                and attribution.coefficients.shape
                == (len(self.problem.context_ids), len(self.problem.family_ids))
                and bool(np.all(attribution.coefficients >= 0))
                and np.allclose(
                    attribution.contributions,
                    attribution.coefficients * column_norms,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
                and all(
                    np.array_equal(observed, expected)
                    for observed, expected in zip(
                        attribution.signed_responses,
                        self.problem.signed_responses,
                        strict=True,
                    )
                )
                and all(
                    np.array_equal(observed, expected)
                    for observed, expected in zip(
                        attribution.precision_weights,
                        self.problem.precision_weights,
                        strict=True,
                    )
                )
                and all(
                    np.allclose(observed, expected, rtol=1.0e-10, atol=1.0e-12)
                    for observed, expected in zip(
                        attribution.predicted,
                        expected_predictions,
                        strict=True,
                    )
                )
                and all(
                    np.allclose(
                        signed,
                        predicted + residual,
                        rtol=1.0e-10,
                        atol=1.0e-12,
                    )
                    for signed, predicted, residual in zip(
                        attribution.signed_responses,
                        attribution.predicted,
                        attribution.residuals,
                        strict=True,
                    )
                )
            )
        valid_state = (
            (
                self.status == "observed"
                and self.reason_code is None
                and self.fit is not None
                and self.fit.attribution.succeeded
            )
            or (
                self.status == "not_estimable"
                and self.reason_code is not None
                and self.fit is None
                and not self.tuning_fold_data_ids
            )
            or (
                self.status == "failed"
                and self.reason_code is not None
                and (self.fit is None or not self.fit.attribution.succeeded)
            )
        )
        if (
            self._producer_marker != _WORKFLOW_PRODUCER
            or not self.experimental
            or not nested_lineage_matches
            or not fit_matches_parents
            or not valid_state
            or self.workflow_id
            != stable_id(
                "graph_fused_workflow_artifact",
                self._identity_payload(),
                schema_version="1",
            )
        ):
            raise ContractError(
                "Graph-fused workflow artifact failed integrity validation",
                code="graph_fused_workflow_integrity_violation",
                field="workflow_id",
                remediation="Refit from intact outer and inner training parents",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "workflow_id": self.workflow_id,
            "spec_id": self.spec.spec_id,
            "problem_id": self.problem.problem_id,
            "graph_id": self.problem.graph_id,
            "family_parent_ids": list(self.problem.family_parent_ids),
            "response_id": self.problem.response_id,
            "precision_id": self.problem.precision_id,
            "tuning_spec_id": self.spec.tuning_spec.spec_id,
            "inner_partition_id": self.inner_partition_id,
            "tuning_fold_data_ids": list(self.tuning_fold_data_ids),
            "fit_id": None if self.fit is None else self.fit.tuned_attribution_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "experimental": True,
            "formal_inference_allowed": False,
            "p_value": None,
            "q_value": None,
            "comm_probability": None,
        }


def _application_parent_mismatch(message: str, *, field: str) -> ContractError:
    return ContractError(
        message,
        code="graph_fused_application_parent_mismatch",
        field=field,
        remediation=(
            "Use the exact held-out response and design applications declared by "
            "this outer graph workflow"
        ),
    )


def _validate_application_parents(
    artifact: GraphFusedWorkflowArtifact,
    response_application: FoldGeneResponseApplication,
    design_application: FrozenDesignApplication,
) -> None:
    try:
        artifact._require_intact()
    except ContractError:
        raise
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ContractError(
            "Graph-fused workflow artifact failed integrity validation",
            code="graph_fused_workflow_integrity_violation",
            field="workflow_id",
            remediation="Refit from intact outer and inner training parents",
        ) from error
    response_application.to_dict()
    design_application.to_dict()
    problem = artifact.problem
    response_lineage = (
        response_application.design_application_id,
        response_application.encoder_id,
        response_application.context_regressor_id,
        response_application.nuisance_design_id,
        response_application.sample_ids,
        response_application.sample_subject_ids,
        response_application.sample_context_ids,
        response_application.subject_ids,
    )
    design_lineage = (
        design_application.application_id,
        design_application.encoder_id,
        design_application.context_regressor_id,
        design_application.nuisance_design_id,
        design_application.sample_ids,
        design_application.sample_subject_ids,
        design_application.sample_context_ids,
        design_application.subject_ids,
    )
    if response_lineage != design_lineage:
        raise _application_parent_mismatch(
            "Held-out graph response and design rows or lineage differ",
            field="response_application,design_application",
        )
    expected_scope = (
        problem.response_artifact_id,
        problem.design_encoder_id,
        problem.receiver,
        problem.fold_id,
        problem.feature_ids,
        problem.heldout_subject_ids,
        "heldout",
    )
    observed_scope = (
        response_application.training_response_id,
        response_application.encoder_id,
        response_application.receiver,
        response_application.fold_id,
        response_application.feature_ids,
        response_application.subject_ids,
        design_application.application_scope,
    )
    if observed_scope != expected_scope:
        raise _application_parent_mismatch(
            "Held-out graph parents are outside the frozen outer-fold scope",
            field="heldout_subject_ids",
        )
    overlap = set(problem.training_subject_ids).intersection(
        response_application.subject_ids
    )
    if overlap:
        raise _application_parent_mismatch(
            "Held-out graph subjects overlap outer training",
            field="training_subject_ids,heldout_subject_ids",
        )


def _application_input_digest(
    response_application: FoldGeneResponseApplication,
    design_application: FrozenDesignApplication,
) -> str:
    identifier: str = stable_id(
        "graph_fused_workflow_application_input",
        {
            "context_regressor_digest": (design_application.context_regressor_digest),
            "design_application_id": design_application.application_id,
            "nuisance_matrix_digest": design_application.nuisance_matrix_digest,
            "response_application_id": response_application.application_id,
            "response_input_digest": response_application.response_input_digest,
            "sample_values_digest": response_application.sample_values_digest,
            "rows": [
                {
                    "context_id": context_id,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                }
                for sample_id, subject_id, context_id in zip(
                    response_application.sample_ids,
                    response_application.sample_subject_ids,
                    response_application.sample_context_ids,
                    strict=True,
                )
            ],
        },
        schema_version="1",
        digest_length=64,
    )
    return identifier


@dataclass(frozen=True, slots=True)
class _ApplicationOutputs:
    fixed_predictions: np.ndarray
    fixed_contributions: np.ndarray
    fixed_precision_weights: np.ndarray
    heldout_positive_responses: np.ndarray
    residuals: np.ndarray
    context_losses: np.ndarray
    subject_losses: np.ndarray


def _compute_application_outputs(
    artifact: GraphFusedWorkflowArtifact,
    response_application: FoldGeneResponseApplication,
    design_application: FrozenDesignApplication,
) -> _ApplicationOutputs | str:
    fit = artifact.fit
    if fit is None:  # pragma: no cover - guarded by the public application path
        return "graph_fused_workflow_not_estimable"
    problem = artifact.problem
    transformed_design = np.column_stack(
        (design_application.nuisance_matrix, design_application.context_regressor)
    )
    full_design = np.asarray(
        transformed_design @ problem.application_reparameterization_inverse,
        dtype=np.float64,
    )
    context_count = len(problem.context_ids)
    feature_count = len(problem.feature_ids)
    heldout: np.ndarray = np.empty(
        (len(problem.heldout_subject_ids), context_count, feature_count),
        dtype=np.float64,
    )
    subject_values = np.asarray(response_application.sample_subject_ids, dtype=object)
    context_values = np.asarray(response_application.sample_context_ids, dtype=object)
    graph_contexts = set(problem.context_ids)
    for subject_index, subject_id in enumerate(problem.heldout_subject_ids):
        positions = np.flatnonzero(
            (subject_values == subject_id)
            & np.asarray(
                [value in graph_contexts for value in context_values], dtype=bool
            )
        )
        if positions.size == 0:
            return f"heldout_receiver_response_missing:{subject_id}"
        values = response_application.sample_values[positions]
        subject_design = full_design[positions]
        for context_index, context_id in enumerate(problem.context_ids):
            regressor = np.asarray(
                subject_design @ problem.application_context_directions[context_index],
                dtype=np.float64,
            )
            denominator = float(np.dot(regressor, regressor))
            if denominator <= 1.0e-12:
                return (
                    "heldout_response_coordinate_not_estimable:"
                    f"{subject_id}:{context_id}"
                )
            nuisance_prediction = (
                subject_design @ problem.application_nuisance_projections[context_index]
            )
            coordinate = regressor @ (values - nuisance_prediction) / denominator
            heldout[subject_index, context_index] = np.maximum(coordinate, 0.0)
    predictions = np.vstack(fit.attribution.predicted)
    contributions = np.asarray(fit.attribution.contributions, dtype=np.float64)
    precision = np.vstack(problem.precision_weights)
    residuals = heldout - predictions[np.newaxis, :, :]
    numerators = np.sum(precision[np.newaxis, :, :] * np.square(residuals), axis=2)
    weight_sums = precision.sum(axis=1)
    context_losses = numerators / weight_sums[np.newaxis, :]
    subject_losses = numerators.sum(axis=1) / float(weight_sums.sum())
    return _ApplicationOutputs(
        fixed_predictions=predictions,
        fixed_contributions=contributions,
        fixed_precision_weights=precision,
        heldout_positive_responses=heldout,
        residuals=residuals,
        context_losses=context_losses,
        subject_losses=subject_losses,
    )


@dataclass(frozen=True, slots=True, init=False)
class GraphFusedWorkflowApplication:
    """Frozen graph model evaluated on exact typed held-out parents."""

    application_id: str
    workflow_id: str
    problem_id: str
    graph_id: str
    application_functional_id: str
    fit_id: str | None
    tuning_id: str | None
    selected_candidate_id: str | None
    coefficient_digest: str | None
    response_application_id: str
    response_input_digest: str
    sample_values_digest: str
    design_application_id: str
    nuisance_matrix_digest: str
    context_regressor_digest: str
    heldout_input_digest: str
    training_subject_ids: tuple[str, ...]
    heldout_subject_ids: tuple[str, ...]
    heldout_sample_ids: tuple[str, ...]
    heldout_sample_subject_ids: tuple[str, ...]
    heldout_sample_context_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    fixed_predictions: np.ndarray | None
    fixed_contributions: np.ndarray | None
    fixed_precision_weights: np.ndarray | None
    heldout_positive_responses: np.ndarray | None
    residuals: np.ndarray | None
    context_losses: np.ndarray | None
    subject_losses: np.ndarray | None
    fixed_prediction_digest: str | None
    fixed_contribution_digest: str | None
    fixed_precision_digest: str | None
    heldout_response_digest: str | None
    residual_digest: str | None
    context_loss_digest: str | None
    subject_loss_digest: str | None
    mean_subject_loss: float | None
    loss_estimand: str
    status: str
    reason_code: str | None
    experimental: bool
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "GraphFusedWorkflowApplication is producer-owned; use "
            "apply_graph_fused_workflow()"
        )

    @classmethod
    def _from_application(
        cls,
        *,
        artifact: GraphFusedWorkflowArtifact,
        response_application: FoldGeneResponseApplication,
        design_application: FrozenDesignApplication,
        outputs: _ApplicationOutputs | None,
        reason_code: str | None,
    ) -> GraphFusedWorkflowApplication:
        _validate_application_parents(
            artifact, response_application, design_application
        )
        problem = artifact.problem
        status = "observed" if outputs is not None else "not_estimable"
        if (status == "observed") == (reason_code is not None):
            raise ValueError("graph application status and reason are inconsistent")
        fit = artifact.fit
        coefficient_digest = (
            None if fit is None else _array_digest(fit.attribution.coefficients)
        )
        output_values: dict[str, object]
        if outputs is None:
            output_values = {
                "fixed_predictions": None,
                "fixed_contributions": None,
                "fixed_precision_weights": None,
                "heldout_positive_responses": None,
                "residuals": None,
                "context_losses": None,
                "subject_losses": None,
                "fixed_prediction_digest": None,
                "fixed_contribution_digest": None,
                "fixed_precision_digest": None,
                "heldout_response_digest": None,
                "residual_digest": None,
                "context_loss_digest": None,
                "subject_loss_digest": None,
                "mean_subject_loss": None,
            }
        else:
            subject_count = len(problem.heldout_subject_ids)
            context_count = len(problem.context_ids)
            feature_count = len(problem.feature_ids)
            family_count = len(problem.family_ids)
            predictions = _immutable_array(
                outputs.fixed_predictions,
                shape=(context_count, feature_count),
                field_name="fixed_predictions",
            )
            contributions = _immutable_array(
                outputs.fixed_contributions,
                shape=(context_count, family_count),
                field_name="fixed_contributions",
            )
            precision = _immutable_array(
                outputs.fixed_precision_weights,
                shape=(context_count, feature_count),
                field_name="fixed_precision_weights",
            )
            heldout = _immutable_array(
                outputs.heldout_positive_responses,
                shape=(subject_count, context_count, feature_count),
                field_name="heldout_positive_responses",
            )
            residuals = _immutable_array(
                outputs.residuals,
                shape=(subject_count, context_count, feature_count),
                field_name="residuals",
            )
            context_losses = _immutable_array(
                outputs.context_losses,
                shape=(subject_count, context_count),
                field_name="context_losses",
            )
            subject_losses = _immutable_vector(
                outputs.subject_losses,
                length=subject_count,
                field_name="subject_losses",
            )
            output_values = {
                "fixed_predictions": predictions,
                "fixed_contributions": contributions,
                "fixed_precision_weights": precision,
                "heldout_positive_responses": heldout,
                "residuals": residuals,
                "context_losses": context_losses,
                "subject_losses": subject_losses,
                "fixed_prediction_digest": _array_digest(predictions),
                "fixed_contribution_digest": _array_digest(contributions),
                "fixed_precision_digest": _array_digest(precision),
                "heldout_response_digest": _array_digest(heldout),
                "residual_digest": _array_digest(residuals),
                "context_loss_digest": _array_digest(context_losses),
                "subject_loss_digest": _array_digest(subject_losses),
                "mean_subject_loss": float(np.mean(subject_losses)),
            }
        values: dict[str, object] = {
            "workflow_id": artifact.workflow_id,
            "problem_id": problem.problem_id,
            "graph_id": problem.graph_id,
            "application_functional_id": problem.application_functional_id,
            "fit_id": None if fit is None else fit.tuned_attribution_id,
            "tuning_id": None if fit is None else fit.tuning.tuning_id,
            "selected_candidate_id": (
                None if fit is None else fit.selected_candidate_id
            ),
            "coefficient_digest": coefficient_digest,
            "response_application_id": response_application.application_id,
            "response_input_digest": response_application.response_input_digest,
            "sample_values_digest": response_application.sample_values_digest,
            "design_application_id": design_application.application_id,
            "nuisance_matrix_digest": design_application.nuisance_matrix_digest,
            "context_regressor_digest": (design_application.context_regressor_digest),
            "heldout_input_digest": _application_input_digest(
                response_application, design_application
            ),
            "training_subject_ids": problem.training_subject_ids,
            "heldout_subject_ids": problem.heldout_subject_ids,
            "heldout_sample_ids": response_application.sample_ids,
            "heldout_sample_subject_ids": (response_application.sample_subject_ids),
            "heldout_sample_context_ids": (response_application.sample_context_ids),
            "context_ids": problem.context_ids,
            "feature_ids": problem.feature_ids,
            "family_ids": problem.family_ids,
            **output_values,
            "loss_estimand": _APPLICATION_LOSS_ESTIMAND,
            "status": status,
            "reason_code": reason_code,
            "experimental": True,
            "_producer_marker": _APPLICATION_PRODUCER,
        }
        temporary = cls._from_producer(**values, application_id="pending")
        result = cls._from_producer(
            **values,
            application_id=stable_id(
                "graph_fused_workflow_application",
                temporary._identity_payload(),
                schema_version="1",
            ),
        )
        result._require_intact()
        return result

    @classmethod
    def _from_producer(cls, **values: object) -> GraphFusedWorkflowApplication:
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "application_functional_id": self.application_functional_id,
            "coefficient_digest": self.coefficient_digest,
            "context_loss_digest": self.context_loss_digest,
            "design_application_id": self.design_application_id,
            "experimental": True,
            "fit_id": self.fit_id,
            "fixed_contribution_digest": self.fixed_contribution_digest,
            "fixed_precision_digest": self.fixed_precision_digest,
            "fixed_prediction_digest": self.fixed_prediction_digest,
            "graph_id": self.graph_id,
            "heldout_input_digest": self.heldout_input_digest,
            "heldout_response_digest": self.heldout_response_digest,
            "loss_estimand": self.loss_estimand,
            "mean_subject_loss": self.mean_subject_loss,
            "problem_id": self.problem_id,
            "reason_code": self.reason_code,
            "residual_digest": self.residual_digest,
            "response_application_id": self.response_application_id,
            "selected_candidate_id": self.selected_candidate_id,
            "status": self.status,
            "subject_loss_digest": self.subject_loss_digest,
            "tuning_id": self.tuning_id,
            "workflow_id": self.workflow_id,
        }

    def _require_intact(self) -> None:
        try:
            sample_ids = _names(
                self.heldout_sample_ids, field_name="heldout_sample_ids"
            )
            sample_subjects = tuple(self.heldout_sample_subject_ids)
            sample_contexts = tuple(self.heldout_sample_context_ids)
            subjects = tuple(
                sorted(
                    _names(
                        self.heldout_subject_ids,
                        field_name="heldout_subject_ids",
                    )
                )
            )
            training = tuple(
                sorted(
                    _names(
                        self.training_subject_ids,
                        field_name="training_subject_ids",
                        minimum=2,
                    )
                )
            )
            if (
                len(sample_subjects) != len(sample_ids)
                or len(sample_contexts) != len(sample_ids)
                or tuple(sorted(set(sample_subjects))) != subjects
                or set(training).intersection(subjects)
            ):
                raise ValueError("graph application held-out rows are inconsistent")
            if (
                self._producer_marker != _APPLICATION_PRODUCER
                or not self.experimental
                or self.loss_estimand != _APPLICATION_LOSS_ESTIMAND
                or self.status not in {"observed", "not_estimable"}
                or ((self.status == "observed") == (self.reason_code is not None))
            ):
                raise ValueError("graph application state is inconsistent")
            observed_arrays = (
                self.fixed_predictions,
                self.fixed_contributions,
                self.fixed_precision_weights,
                self.heldout_positive_responses,
                self.residuals,
                self.context_losses,
                self.subject_losses,
            )
            observed_digests = (
                self.fixed_prediction_digest,
                self.fixed_contribution_digest,
                self.fixed_precision_digest,
                self.heldout_response_digest,
                self.residual_digest,
                self.context_loss_digest,
                self.subject_loss_digest,
            )
            if self.status == "observed":
                if any(values is None for values in observed_arrays) or any(
                    digest is None for digest in observed_digests
                ):
                    raise ValueError("observed graph application outputs are missing")
                predictions = cast(np.ndarray, self.fixed_predictions)
                contributions = cast(np.ndarray, self.fixed_contributions)
                precision = cast(np.ndarray, self.fixed_precision_weights)
                heldout = cast(np.ndarray, self.heldout_positive_responses)
                residuals = cast(np.ndarray, self.residuals)
                context_losses = cast(np.ndarray, self.context_losses)
                subject_losses = cast(np.ndarray, self.subject_losses)
                context_count = len(self.context_ids)
                feature_count = len(self.feature_ids)
                subject_count = len(subjects)
                valid_shapes = (
                    predictions.shape == (context_count, feature_count)
                    and contributions.shape == (context_count, len(self.family_ids))
                    and precision.shape == (context_count, feature_count)
                    and heldout.shape == (subject_count, context_count, feature_count)
                    and residuals.shape == heldout.shape
                    and context_losses.shape == (subject_count, context_count)
                    and subject_losses.shape == (subject_count,)
                )
                if not valid_shapes or any(
                    not _is_immutable_byte_backed(values)
                    for values in cast(tuple[np.ndarray, ...], observed_arrays)
                ):
                    raise ValueError("graph application arrays are invalid")
                if (
                    np.any(predictions < 0)
                    or np.any(contributions < 0)
                    or np.any(precision < 0)
                    or np.any(heldout < 0)
                    or np.any(context_losses < 0)
                    or np.any(subject_losses < 0)
                    or not np.allclose(
                        heldout,
                        predictions[np.newaxis, :, :] + residuals,
                        rtol=1.0e-10,
                        atol=1.0e-12,
                    )
                ):
                    raise ValueError("graph application output algebra is invalid")
                numerators = np.sum(
                    precision[np.newaxis, :, :] * np.square(residuals), axis=2
                )
                weight_sums = precision.sum(axis=1)
                expected_context_losses = numerators / weight_sums[np.newaxis, :]
                expected_subject_losses = numerators.sum(axis=1) / float(
                    weight_sums.sum()
                )
                expected_digests = tuple(
                    _array_digest(values)
                    for values in (
                        predictions,
                        contributions,
                        precision,
                        heldout,
                        residuals,
                        context_losses,
                        subject_losses,
                    )
                )
                if (
                    tuple(cast(tuple[str, ...], observed_digests)) != expected_digests
                    or not np.allclose(
                        context_losses,
                        expected_context_losses,
                        rtol=1.0e-10,
                        atol=1.0e-12,
                    )
                    or not np.allclose(
                        subject_losses,
                        expected_subject_losses,
                        rtol=1.0e-10,
                        atol=1.0e-12,
                    )
                    or self.mean_subject_loss is None
                    or not math.isclose(
                        self.mean_subject_loss,
                        float(np.mean(subject_losses)),
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise ValueError("graph application output digests are invalid")
            elif (
                any(values is not None for values in observed_arrays)
                or any(digest is not None for digest in observed_digests)
                or self.mean_subject_loss is not None
            ):
                raise ValueError("not-estimable graph application released outputs")
            expected_id = stable_id(
                "graph_fused_workflow_application",
                self._identity_payload(),
                schema_version="1",
            )
            valid = self.application_id == expected_id
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Graph-fused workflow application failed integrity validation",
                code="graph_fused_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact frozen graph workflow",
            ) from error
        if not valid:
            raise ContractError(
                "Graph-fused workflow application failed integrity validation",
                code="graph_fused_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact frozen graph workflow",
            )

    def require_compatible(
        self,
        artifact: GraphFusedWorkflowArtifact,
        response_application: FoldGeneResponseApplication,
        design_application: FrozenDesignApplication,
    ) -> None:
        """Validate integrity and exact derivation from all three parents."""

        self._require_intact()
        expected = apply_graph_fused_workflow(
            artifact, response_application, design_application
        )
        if self.application_id != expected.application_id:
            raise _application_parent_mismatch(
                "Graph application does not derive from the supplied parents",
                field="application_id",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "application_id": self.application_id,
            "workflow_id": self.workflow_id,
            "problem_id": self.problem_id,
            "graph_id": self.graph_id,
            "application_functional_id": self.application_functional_id,
            "fit_id": self.fit_id,
            "tuning_id": self.tuning_id,
            "selected_candidate_id": self.selected_candidate_id,
            "coefficient_digest": self.coefficient_digest,
            "response_application_id": self.response_application_id,
            "response_input_digest": self.response_input_digest,
            "sample_values_digest": self.sample_values_digest,
            "design_application_id": self.design_application_id,
            "nuisance_matrix_digest": self.nuisance_matrix_digest,
            "context_regressor_digest": self.context_regressor_digest,
            "heldout_input_digest": self.heldout_input_digest,
            "training_subject_ids": list(self.training_subject_ids),
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "heldout_sample_ids": list(self.heldout_sample_ids),
            "heldout_sample_subject_ids": list(self.heldout_sample_subject_ids),
            "heldout_sample_context_ids": list(self.heldout_sample_context_ids),
            "context_ids": list(self.context_ids),
            "feature_ids": list(self.feature_ids),
            "family_ids": list(self.family_ids),
            "fixed_prediction_digest": self.fixed_prediction_digest,
            "fixed_contribution_digest": self.fixed_contribution_digest,
            "fixed_precision_digest": self.fixed_precision_digest,
            "heldout_response_digest": self.heldout_response_digest,
            "residual_digest": self.residual_digest,
            "context_loss_digest": self.context_loss_digest,
            "subject_loss_digest": self.subject_loss_digest,
            "context_losses": (
                None if self.context_losses is None else self.context_losses.tolist()
            ),
            "subject_losses": (
                None if self.subject_losses is None else self.subject_losses.tolist()
            ),
            "mean_subject_loss": self.mean_subject_loss,
            "loss_estimand": self.loss_estimand,
            "status": self.status,
            "reason_code": self.reason_code,
            "experimental": True,
            "formal_inference_allowed": False,
            "p_value": None,
            "q_value": None,
        }


def _workflow_artifact(
    *,
    spec: GraphFusedWorkflowSpec,
    problem: GraphFusedTrainingProblem,
    inner_partition: GraphFusedInnerPartition,
    tuning_folds: tuple[GraphTuningFold, ...],
    fit: TunedGraphFusedFamilyAttribution | None,
    reason_code: str | None,
    status_override: str | None = None,
) -> GraphFusedWorkflowArtifact:
    status = (
        status_override
        if status_override is not None
        else "observed"
        if fit is not None and fit.attribution.succeeded
        else "failed"
    )
    if status not in {"observed", "not_estimable", "failed"}:
        raise ValueError("status_override must be observed, not_estimable, or failed")
    reason = (
        None
        if status == "observed"
        else reason_code
        or (None if fit is None else fit.attribution.diagnostics.failure_reason)
        or "graph_fused_solver_not_converged"
    )
    values: dict[str, object] = {
        "spec": spec,
        "problem": problem,
        "inner_partition_id": inner_partition.partition_id,
        "tuning_fold_data_ids": tuple(fold.fold_data_id for fold in tuning_folds),
        "fit": fit,
        "status": status,
        "reason_code": reason,
        "experimental": True,
        "_inner_partition": inner_partition,
        "_tuning_folds": tuning_folds,
        "_producer_marker": _WORKFLOW_PRODUCER,
    }
    temporary = GraphFusedWorkflowArtifact._from_producer(
        **values,
        workflow_id="pending",
    )
    result = GraphFusedWorkflowArtifact._from_producer(
        **values,
        workflow_id=stable_id(
            "graph_fused_workflow_artifact",
            temporary._identity_payload(),
            schema_version="1",
        ),
    )
    result._require_intact()
    return result


def fit_graph_fused_workflow(
    aggregate: PseudobulkDataset,
    sample_metadata: pd.DataFrame,
    design_encoder: FrozenDesignEncoder,
    family_parents: Mapping[Hashable, ReceiverFamilyTrainingArtifact],
    graph: ContextGraph,
    inner_partition: GraphFusedInnerPartition,
    *,
    receiver: str,
    fold_id: str,
    training_input_digest: str,
    outer_training_subject_ids: Sequence[str],
    heldout_subject_ids: Sequence[str],
    spec: GraphFusedWorkflowSpec,
) -> GraphFusedWorkflowArtifact:
    """Build, tune, and refit one graph model entirely inside outer training."""

    problem = build_graph_fused_training_problem(
        aggregate,
        sample_metadata,
        design_encoder,
        family_parents,
        graph,
        receiver=receiver,
        fold_id=fold_id,
        training_input_digest=training_input_digest,
        outer_training_subject_ids=outer_training_subject_ids,
        heldout_subject_ids=heldout_subject_ids,
        spec=spec,
    )
    for inner in inner_partition.folds:
        inner._require_intact(
            problem.graph,
            problem.receiver,
            outer_parent_ids=problem.family_parent_ids,
        )
    unavailable = next(
        (inner for inner in inner_partition.folds if inner.status != "observed"),
        None,
    )
    if unavailable is not None:
        return _workflow_artifact(
            spec=spec,
            problem=problem,
            inner_partition=inner_partition,
            tuning_folds=(),
            fit=None,
            reason_code=unavailable.reason_code,
            status_override="not_estimable",
        )
    tuning_folds = build_graph_fused_tuning_folds(
        aggregate,
        sample_metadata,
        design_encoder,
        problem,
        inner_partition,
        spec=spec,
    )
    try:
        fit = fit_tuned_graph_fused_family_attribution(
            problem.basis_mapping(),
            problem.response_mapping(),
            problem.graph,
            tuning_folds,
            spec.tuning_spec,
            precision_weights=problem.precision_mapping(),
            solver_tolerance=spec.solver_tolerance,
            max_iterations=spec.max_iterations,
        )
    except ContractError as error:
        if error.details.code != "graph_tuning_no_estimable_candidate":
            raise
        return _workflow_artifact(
            spec=spec,
            problem=problem,
            inner_partition=inner_partition,
            tuning_folds=tuning_folds,
            fit=None,
            reason_code=error.details.code,
        )
    return _workflow_artifact(
        spec=spec,
        problem=problem,
        inner_partition=inner_partition,
        tuning_folds=tuning_folds,
        fit=fit,
        reason_code=fit.attribution.diagnostics.failure_reason,
    )


def apply_graph_fused_workflow(
    artifact: GraphFusedWorkflowArtifact,
    response_application: FoldGeneResponseApplication,
    design_application: FrozenDesignApplication,
) -> GraphFusedWorkflowApplication:
    """Score one frozen outer graph fit without fitting or tuning held-out data."""

    if not isinstance(artifact, GraphFusedWorkflowArtifact):
        raise TypeError("artifact must be a GraphFusedWorkflowArtifact")
    if not isinstance(response_application, FoldGeneResponseApplication):
        raise TypeError("response_application must be a FoldGeneResponseApplication")
    if not isinstance(design_application, FrozenDesignApplication):
        raise TypeError("design_application must be a FrozenDesignApplication")
    _validate_application_parents(artifact, response_application, design_application)
    reason: str | None = None
    outputs: _ApplicationOutputs | None = None
    if artifact.status != "observed" or artifact.fit is None:
        reason = "graph_fused_workflow_not_estimable:" + str(artifact.reason_code)
    elif design_application.status != "observed":
        reason = "heldout_design_not_estimable:" + str(design_application.reason_code)
    elif response_application.status != "ok":
        reason = "heldout_response_not_estimable:" + str(
            response_application.reason_code
        )
    else:
        computed = _compute_application_outputs(
            artifact, response_application, design_application
        )
        if isinstance(computed, str):
            reason = computed
        else:
            outputs = computed
    return GraphFusedWorkflowApplication._from_application(
        artifact=artifact,
        response_application=response_application,
        design_application=design_application,
        outputs=outputs,
        reason_code=reason,
    )


__all__ = [
    "GraphFusedInnerFoldParents",
    "GraphFusedInnerPartition",
    "GraphFusedTrainingProblem",
    "GraphFusedWorkflowApplication",
    "GraphFusedWorkflowArtifact",
    "GraphFusedWorkflowSpec",
    "apply_graph_fused_workflow",
    "build_graph_fused_training_problem",
    "build_graph_fused_tuning_folds",
    "fit_graph_fused_workflow",
    "freeze_graph_fused_inner_fold_parents",
]
