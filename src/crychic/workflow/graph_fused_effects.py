"""Held-out family effects from one intact graph-fused cross-fit registry.

The producer in this module does not refit a model.  It evaluates the exact
outer-fold graph coefficients by removing one frozen family contribution at a
time from each held-out subject/context prediction.  The resulting table is a
descriptive conditional-gain view and never authorizes formal inference.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import node_context_fields

from .crossfit import CrossFitArtifacts, CrossFitFoldArtifacts
from .graph_fused_crossfit import (
    GraphFusedCrossFitRecord,
    GraphFusedCrossFitRegistry,
)
from .receiver_universe import ReceiverTrainingSupportStatus

GRAPH_FUSED_FAMILY_EFFECT_SEMANTICS = "graph_joint_conditional_family_gain_v1"
GRAPH_FUSED_FAMILY_EFFECT_COLUMNS = (
    "crossfit_id",
    "registry_id",
    "receiver_family_universe_id",
    "family_axis_id",
    "record_id",
    "graph_id",
    "workflow_id",
    "application_id",
    "problem_id",
    "fit_id",
    "coefficient_digest",
    "fixed_precision_digest",
    "heldout_response_digest",
    "context_loss_digest",
    "fold_id",
    "receiver",
    "receiver_universe_id",
    "receiver_training_support_id",
    "receiver_training_support_status",
    "receiver_training_support_reason_code",
    "receiver_family_opportunity_id",
    "subject_id",
    "context_id",
    "family_id",
    "receiver_family_training_artifact_id",
    "family_basis_id",
    "family_coefficient",
    "full_loss",
    "loss_without",
    "raw_conditional_gain",
    "bounded_conditional_gain",
    "status",
    "reason_code",
    "semantics",
    "experimental",
    "formal_inference_allowed",
)
GRAPH_FUSED_FAMILY_EFFECT_KEY = (
    "fold_id",
    "receiver",
    "subject_id",
    "context_id",
    "family_id",
)

_SCHEMA_VERSION = "2.0.0"
_IDENTITY_SCHEMA_VERSION = "1"
_PRODUCER = "crychic.workflow.graph_fused_family_effects.v2"
_COLLECTION_MARKER = "crychic.graph_fused_family_effect_collection.v2"
_GAIN_EPSILON = 1.0e-12
_STRUCTURAL_ZERO_REASON = "zero_fitted_family_coefficient"
_ROW_STATUSES = frozenset({"observed", "structural_zero", "not_estimable"})


def _source_mismatch(message: str, *, field_name: str) -> ContractError:
    return ContractError(
        message,
        code="graph_fused_family_effect_source_mismatch",
        field=field_name,
        remediation=(
            "Use one intact CrossFitArtifacts object and the exact graph-fused "
            "registry derived from it"
        ),
    )


def _graph_id(registry: GraphFusedCrossFitRegistry) -> str:
    return str(
        stable_id(
            "context_graph",
            registry.graph.to_dict(),
            schema_version=_IDENTITY_SCHEMA_VERSION,
        )
    )


def _array_digest(values: object) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    if np.any(~np.isfinite(array)):
        raise ValueError("graph-fused effect arrays must be finite")
    digest = hashlib.sha256()
    digest.update(canonical_json({"shape": list(array.shape)}).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _cell_token(value: object) -> dict[str, object]:
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "missing", "value": None}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "missing", "value": None}
        if not math.isfinite(value):
            raise ValueError("family-effect tables cannot contain infinity")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": canonical_json(value),
    }


def _table_digest(table: pd.DataFrame) -> str:
    rows = [
        [_cell_token(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    rows.sort(key=canonical_json)
    return str(
        stable_id(
            "graph_fused_family_effect_table",
            {"columns": list(table.columns), "rows": rows},
            schema_version=_IDENTITY_SCHEMA_VERSION,
            digest_length=64,
        )
    )


def _optional_string(value: object) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    result = str(value)
    return result if result else None


def _fold_by_id(crossfit: CrossFitArtifacts) -> dict[str, CrossFitFoldArtifacts]:
    result = {fold.fold_id: fold for fold in crossfit.folds}
    if len(result) != len(crossfit.folds):
        raise _source_mismatch(
            "Cross-fit folds do not have unique identities",
            field_name="crossfit.folds",
        )
    return result


def _outer_family_parents(
    crossfit: CrossFitArtifacts,
) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    for fold in crossfit.folds:
        expected_receivers = set(map(str, fold.training.cell_type_ids))
        observed_receivers: set[str] = set()
        for model in fold.receiver_family_models:
            parent = model.receiver_family_artifact
            parent._require_producer_owned()
            receiver = str(parent.receiver)
            key = (fold.fold_id, receiver)
            observed_receivers.add(receiver)
            previous = result.get(key)
            if previous is not None and (
                previous.training_artifact_id != parent.training_artifact_id
            ):
                raise _source_mismatch(
                    "Outer receiver family axes diverge across contrasts",
                    field_name="crossfit.folds.receiver_family_models",
                )
            if parent.fold_id != fold.fold_id or parent.training_subject_ids != tuple(
                sorted(fold.training.training_subject_ids)
            ):
                raise _source_mismatch(
                    "Outer receiver family parent is outside its cross-fit fold",
                    field_name="receiver_family_training_artifact",
                )
            result[key] = parent
        if observed_receivers != expected_receivers:
            raise _source_mismatch(
                "Outer receiver family parents do not cover the planned receivers",
                field_name="crossfit.folds.receiver_family_models",
            )
    return result


def _context_ids(
    registry: GraphFusedCrossFitRegistry,
    folds: dict[str, CrossFitFoldArtifacts],
) -> tuple[str, ...]:
    first = next(iter(folds.values()), None)
    if first is None:
        raise _source_mismatch(
            "Cross-fit contains no outer folds",
            field_name="crossfit.folds",
        )
    context_keys = tuple(first.training.config.context_keys)
    for fold in folds.values():
        if tuple(fold.training.config.context_keys) != context_keys:
            raise _source_mismatch(
                "Outer folds disagree on context keys",
                field_name="crossfit.folds.training.config.context_keys",
            )
    values = tuple(
        str(node_context_fields(node, context_keys)[0]) for node in registry.graph.nodes
    )
    if not values or len(values) != len(set(values)):
        raise _source_mismatch(
            "Graph nodes do not map to one unique context axis",
            field_name="registry.graph.nodes",
        )
    return values


def _validate_observed_record(
    record: GraphFusedCrossFitRecord,
    *,
    fold: CrossFitFoldArtifacts,
    parent: Any,
    registry: GraphFusedCrossFitRegistry,
    context_ids: tuple[str, ...],
) -> None:
    workflow = record.workflow
    application = record.application
    if workflow is None or application is None or workflow.fit is None:
        raise _source_mismatch(
            "Observed graph-fused record is missing its fitted parent chain",
            field_name="registry.records.workflow,application",
        )
    problem = workflow.problem
    fit = workflow.fit
    attribution = fit.attribution
    family_basis = parent.family_basis
    expected_graph_id = _graph_id(registry)
    expected_parent_ids = tuple(
        parent.training_artifact_id for _ in registry.graph.nodes
    )
    expected_basis_ids = tuple(
        family_basis.family_basis_id for _ in registry.graph.nodes
    )
    application_matches_workflow = (
        application.workflow_id == workflow.workflow_id
        and application.problem_id == problem.problem_id
        and application.graph_id == problem.graph_id
        and application.fit_id == fit.tuned_attribution_id
        and application.tuning_id == fit.tuning.tuning_id
        and application.selected_candidate_id == fit.selected_candidate_id
        and application.coefficient_digest == _array_digest(attribution.coefficients)
        and application.application_functional_id == problem.application_functional_id
    )
    axes_match = (
        record.fold_id == fold.fold_id == problem.fold_id
        and record.receiver == parent.receiver == problem.receiver
        and record.training_subject_ids
        == tuple(sorted(fold.training.training_subject_ids))
        == problem.training_subject_ids
        == application.training_subject_ids
        and record.heldout_subject_ids
        == tuple(sorted(fold.application.heldout_subject_ids))
        == problem.heldout_subject_ids
        == application.heldout_subject_ids
        and problem.graph_id == expected_graph_id == application.graph_id
        and problem.graph.to_dict() == registry.graph.to_dict()
        and problem.context_ids == context_ids == application.context_ids
        and attribution.context_nodes == tuple(registry.graph.nodes)
        and problem.family_ids
        == family_basis.family_ids
        == application.family_ids
        == attribution.family_ids
        and problem.feature_ids
        == family_basis.feature_ids
        == application.feature_ids
        == attribution.feature_ids
        and problem.family_parent_ids == expected_parent_ids
        and tuple(basis.family_basis_id for basis in problem.family_bases)
        == expected_basis_ids
        and attribution.family_basis_ids == expected_basis_ids
    )
    if not application_matches_workflow or not axes_match:
        raise _source_mismatch(
            "Observed record does not bind one exact graph fit/application lineage",
            field_name="registry.records.workflow,application",
        )


@dataclass(frozen=True, slots=True)
class _ValidatedSources:
    folds: dict[str, CrossFitFoldArtifacts]
    parents: dict[tuple[str, str], Any]
    records: dict[tuple[str, str], GraphFusedCrossFitRecord]
    context_ids: tuple[str, ...]
    graph_id: str


def _validate_sources(
    crossfit: CrossFitArtifacts,
    registry: GraphFusedCrossFitRegistry,
) -> _ValidatedSources:
    if type(crossfit) is not CrossFitArtifacts:
        raise TypeError("crossfit must be producer-owned CrossFitArtifacts")
    if type(registry) is not GraphFusedCrossFitRegistry:
        raise TypeError("registry must be a producer-owned GraphFusedCrossFitRegistry")
    crossfit._require_intact()
    registry._require_intact()
    if (
        registry.crossfit_id != crossfit.crossfit_id
        or registry.root_input_identity_id != crossfit.root_input_identity.identity_id
        or registry.root_input_digest != crossfit.root_input_identity.input_digest
        or registry.receiver_universe_id != crossfit.receiver_universe.universe_id
        or registry.receiver_axis_id != crossfit.receiver_universe.receiver_axis_id
        or registry.receiver_ids != crossfit.receiver_universe.receiver_ids
        or registry.outer_fold_ids
        != tuple(sorted(fold.fold_id for fold in crossfit.folds))
        or registry.receiver_family_universe.receiver_universe_id
        != crossfit.receiver_universe.universe_id
        or registry.receiver_family_universe.receiver_axis_id
        != crossfit.receiver_universe.receiver_axis_id
        or registry.receiver_family_universe.receiver_ids
        != crossfit.receiver_universe.receiver_ids
        or registry.receiver_family_universe.root_input_identity_id
        != crossfit.root_input_identity.identity_id
        or registry.receiver_family_universe.root_input_digest
        != crossfit.root_input_identity.input_digest
        or registry.receiver_family_universe.cosine_threshold
        != crossfit.spec.family_cosine_threshold
    ):
        raise _source_mismatch(
            "Graph-fused registry does not derive from this cross-fit run",
            field_name="registry.crossfit_id",
        )
    folds = _fold_by_id(crossfit)
    parents = _outer_family_parents(crossfit)
    context_ids = _context_ids(registry, folds)
    expected_keys = {
        (fold.fold_id, str(receiver))
        for fold in crossfit.folds
        for receiver in crossfit.receiver_universe.receiver_ids
    }
    support_by_key = {
        (fold.fold_id, support.receiver_id): support
        for fold in crossfit.folds
        for support in fold.receiver_training_support
    }
    if set(support_by_key) != expected_keys:
        raise _source_mismatch(
            "Cross-fit support does not cover fold x frozen receiver universe",
            field_name="crossfit.folds.receiver_training_support",
        )
    records: dict[tuple[str, str], GraphFusedCrossFitRecord] = {}
    for record in registry.records:
        key = (record.fold_id, record.receiver)
        if key in records:
            raise _source_mismatch(
                "Graph-fused registry duplicates an outer fold/receiver record",
                field_name="registry.records",
            )
        records[key] = record
    if set(records) != expected_keys:
        raise _source_mismatch(
            "Graph-fused records do not exactly cover outer fold x receiver",
            field_name="registry.records",
        )
    for key, record in records.items():
        fold = folds[key[0]]
        support = support_by_key[key]
        expected_training = tuple(sorted(fold.training.training_subject_ids))
        expected_heldout = tuple(sorted(fold.application.heldout_subject_ids))
        if (
            record.training_subject_ids != expected_training
            or record.heldout_subject_ids != expected_heldout
            or set(expected_training).intersection(expected_heldout)
            or record.receiver_universe_id != support.receiver_universe_id
            or record.receiver_training_support_id != support.support_record_id
            or record.receiver_training_support_status != support.status.value
            or record.receiver_training_support_reason_code != support.reason_code
        ):
            raise _source_mismatch(
                "Graph-fused record axes or receiver support differ from its fold",
                field_name="registry.records",
            )
        if support.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE:
            if (
                key in parents
                or record.status != "not_estimable"
                or record.reason_code != "receiver_absent_in_outer_training"
                or record.inner_partition_id is not None
                or record.workflow is not None
                or record.application is not None
            ):
                raise _source_mismatch(
                    "Training-absent receiver released graph family/model lineage",
                    field_name="registry.records.receiver_training_support_status",
                )
            continue
        parent = parents.get(key)
        if parent is None:
            raise _source_mismatch(
                "Observed receiver lacks its frozen outer family parent",
                field_name="crossfit.folds.receiver_family_models",
            )
        basis = parent.family_basis
        universe = registry.receiver_family_universe
        if (
            parent.prior_manifest_digest != universe.prior_manifest_digest
            or basis.feature_ids != universe.feature_ids
            or basis.strict_cosine_threshold != universe.cosine_threshold
            or basis.family_definitions != universe.family_definitions
            or basis.family_ids != universe.family_ids
            or parent.source_basis.driver_ids != universe.driver_ids
        ):
            raise _source_mismatch(
                "Outer family parent differs from the frozen run-level family axis",
                field_name="crossfit.folds.receiver_family_models",
            )
        if record.status == "observed":
            _validate_observed_record(
                record,
                fold=fold,
                parent=parent,
                registry=registry,
                context_ids=context_ids,
            )
        elif (
            record.status != "not_estimable"
            or record.workflow is not None
            or record.application is not None
            or not record.reason_code
        ):
            raise _source_mismatch(
                "Graph-fused record has an invalid typed not-estimable state",
                field_name="registry.records.status",
            )
    return _ValidatedSources(
        folds=folds,
        parents=parents,
        records=records,
        context_ids=context_ids,
        graph_id=_graph_id(registry),
    )


def _lineage_row(
    *,
    crossfit: CrossFitArtifacts,
    registry: GraphFusedCrossFitRegistry,
    record: GraphFusedCrossFitRecord,
    graph_id: str,
    parent: Any | None,
    subject_id: str,
    context_id: str,
    family_id: str,
    family_basis_id: str | None,
) -> dict[str, object]:
    workflow = record.workflow
    application = record.application
    return {
        "crossfit_id": crossfit.crossfit_id,
        "registry_id": registry.registry_id,
        "receiver_family_universe_id": (
            registry.receiver_family_universe_id
        ),
        "family_axis_id": registry.family_axis_id,
        "record_id": record.record_id,
        "graph_id": graph_id,
        "workflow_id": None if workflow is None else workflow.workflow_id,
        "application_id": (None if application is None else application.application_id),
        "problem_id": None if workflow is None else workflow.problem.problem_id,
        "fit_id": (
            None
            if workflow is None or workflow.fit is None
            else workflow.fit.tuned_attribution_id
        ),
        "coefficient_digest": (
            None if application is None else application.coefficient_digest
        ),
        "fixed_precision_digest": (
            None if application is None else application.fixed_precision_digest
        ),
        "heldout_response_digest": (
            None if application is None else application.heldout_response_digest
        ),
        "context_loss_digest": (
            None if application is None else application.context_loss_digest
        ),
        "fold_id": record.fold_id,
        "receiver": record.receiver,
        "receiver_universe_id": record.receiver_universe_id,
        "receiver_training_support_id": record.receiver_training_support_id,
        "receiver_training_support_status": (
            record.receiver_training_support_status
        ),
        "receiver_training_support_reason_code": (
            record.receiver_training_support_reason_code
        ),
        "receiver_family_opportunity_id": (
            registry.receiver_family_universe.opportunity_id_for(
                record.receiver, family_id
            )
        ),
        "subject_id": subject_id,
        "context_id": context_id,
        "family_id": family_id,
        "receiver_family_training_artifact_id": (
            None if parent is None else parent.training_artifact_id
        ),
        "family_basis_id": family_basis_id,
        "semantics": GRAPH_FUSED_FAMILY_EFFECT_SEMANTICS,
        "experimental": True,
        "formal_inference_allowed": False,
    }


def _not_estimable_rows(
    *,
    crossfit: CrossFitArtifacts,
    registry: GraphFusedCrossFitRegistry,
    record: GraphFusedCrossFitRecord,
    parent: Any | None,
    context_ids: tuple[str, ...],
    graph_id: str,
) -> list[dict[str, object]]:
    basis = None if parent is None else parent.family_basis
    family_ids = (
        registry.receiver_family_universe.family_ids
        if basis is None
        else basis.family_ids
    )
    family_basis_id = None if basis is None else basis.family_basis_id
    return [
        {
            **_lineage_row(
                crossfit=crossfit,
                registry=registry,
                record=record,
                graph_id=graph_id,
                parent=parent,
                subject_id=subject_id,
                context_id=context_id,
                family_id=family_id,
                family_basis_id=family_basis_id,
            ),
            "family_coefficient": np.nan,
            "full_loss": np.nan,
            "loss_without": np.nan,
            "raw_conditional_gain": np.nan,
            "bounded_conditional_gain": np.nan,
            "status": "not_estimable",
            "reason_code": record.reason_code,
        }
        for subject_id in record.heldout_subject_ids
        for context_id in context_ids
        for family_id in family_ids
    ]


def _observed_rows(
    *,
    crossfit: CrossFitArtifacts,
    registry: GraphFusedCrossFitRegistry,
    record: GraphFusedCrossFitRecord,
    parent: Any,
    context_ids: tuple[str, ...],
    graph_id: str,
) -> list[dict[str, object]]:
    workflow = cast(Any, record.workflow)
    application = cast(Any, record.application)
    fit = workflow.fit
    if fit is None:  # pragma: no cover - guarded by source validation
        raise RuntimeError("observed graph workflow is missing its fit")
    problem = workflow.problem
    alpha = np.asarray(fit.attribution.coefficients, dtype=np.float64)
    heldout = np.asarray(application.heldout_positive_responses, dtype=np.float64)
    precision = np.asarray(application.fixed_precision_weights, dtype=np.float64)
    application_losses = np.asarray(application.context_losses, dtype=np.float64)
    full_predictions = np.vstack(
        [
            np.asarray(basis.matrix.dot(alpha[index])).ravel()
            for index, basis in enumerate(problem.family_bases)
        ]
    )
    if not np.allclose(
        full_predictions,
        np.asarray(application.fixed_predictions, dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise _source_mismatch(
            "Graph basis and coefficients do not reproduce held-out predictions",
            field_name="application.fixed_predictions",
        )
    expected_precision = np.vstack(problem.precision_weights)
    if not np.array_equal(precision, expected_precision):
        raise _source_mismatch(
            "Held-out precision differs from the fitted graph problem",
            field_name="application.fixed_precision_weights",
        )
    weight_sums = precision.sum(axis=1)
    if np.any(~np.isfinite(weight_sums)) or np.any(weight_sums <= 0):
        raise _source_mismatch(
            "Graph held-out loss has an invalid precision denominator",
            field_name="application.fixed_precision_weights",
        )
    full_residuals = heldout - full_predictions[np.newaxis, :, :]
    full_losses = (
        np.sum(precision[np.newaxis, :, :] * np.square(full_residuals), axis=2)
        / weight_sums[np.newaxis, :]
    )
    if not np.allclose(
        full_losses,
        application_losses,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise _source_mismatch(
            "Recomputed full loss differs from graph application context losses",
            field_name="application.context_losses",
        )
    rows: list[dict[str, object]] = []
    for context_index, context_id in enumerate(context_ids):
        basis = problem.family_bases[context_index]
        context_precision = precision[context_index]
        context_full_residuals = full_residuals[:, context_index, :]
        for family_index, family_id in enumerate(problem.family_ids):
            coefficient = float(alpha[context_index, family_index])
            family_component = (
                np.asarray(basis.matrix.getcol(family_index).toarray()).ravel()
                * coefficient
            )
            residuals_without = context_full_residuals + family_component[np.newaxis, :]
            losses_without = (
                np.sum(
                    context_precision[np.newaxis, :] * np.square(residuals_without),
                    axis=1,
                )
                / weight_sums[context_index]
            )
            context_full_losses = full_losses[:, context_index]
            raw_gains = (losses_without - context_full_losses) / (
                losses_without + _GAIN_EPSILON
            )
            bounded_gains = np.clip(raw_gains, 0.0, 1.0)
            structural_zero = coefficient == 0.0
            if structural_zero:
                raw_gains = np.zeros_like(raw_gains)
                bounded_gains = np.zeros_like(bounded_gains)
            for subject_index, subject_id in enumerate(record.heldout_subject_ids):
                rows.append(
                    {
                        **_lineage_row(
                            crossfit=crossfit,
                            registry=registry,
                            record=record,
                            graph_id=graph_id,
                            parent=parent,
                            subject_id=subject_id,
                            context_id=context_id,
                            family_id=family_id,
                            family_basis_id=basis.family_basis_id,
                        ),
                        "family_coefficient": coefficient,
                        "full_loss": float(context_full_losses[subject_index]),
                        "loss_without": float(losses_without[subject_index]),
                        "raw_conditional_gain": float(raw_gains[subject_index]),
                        "bounded_conditional_gain": float(bounded_gains[subject_index]),
                        "status": (
                            "structural_zero" if structural_zero else "observed"
                        ),
                        "reason_code": (
                            _STRUCTURAL_ZERO_REASON if structural_zero else None
                        ),
                    }
                )
    return rows


def _build_table(
    crossfit: CrossFitArtifacts,
    registry: GraphFusedCrossFitRegistry,
    sources: _ValidatedSources,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key in sorted(sources.records):
        record = sources.records[key]
        parent = sources.parents.get(key)
        if record.status == "observed":
            if parent is None:  # pragma: no cover - source validation owns this
                raise RuntimeError("observed graph record lacks its family parent")
            rows.extend(
                _observed_rows(
                    crossfit=crossfit,
                    registry=registry,
                    record=record,
                    parent=parent,
                    context_ids=sources.context_ids,
                    graph_id=sources.graph_id,
                )
            )
        else:
            rows.extend(
                _not_estimable_rows(
                    crossfit=crossfit,
                    registry=registry,
                    record=record,
                    parent=parent,
                    context_ids=sources.context_ids,
                    graph_id=sources.graph_id,
                )
            )
    table = pd.DataFrame(rows, columns=GRAPH_FUSED_FAMILY_EFFECT_COLUMNS)
    if table.empty:
        return table
    return table.sort_values(
        list(GRAPH_FUSED_FAMILY_EFFECT_KEY),
        kind="stable",
        ignore_index=True,
    )


def _missing(value: object) -> bool:
    return value is None or bool(pd.isna(cast(Any, value)))


def _validate_table(
    table: pd.DataFrame,
    *,
    crossfit_id: str,
    registry_id: str,
    graph_id: str,
    receiver_universe_id: str,
    receiver_family_universe_id: str,
    family_axis_id: str,
) -> None:
    if tuple(table.columns) != GRAPH_FUSED_FAMILY_EFFECT_COLUMNS:
        raise ValueError("family_effects columns do not match the contract")
    if table.empty:
        raise ValueError("family_effects must cover at least one frozen family")
    if table.duplicated(list(GRAPH_FUSED_FAMILY_EFFECT_KEY)).any():
        raise ValueError("family_effects contains duplicate primary-key rows")
    for column, expected in (
        ("crossfit_id", crossfit_id),
        ("registry_id", registry_id),
        ("graph_id", graph_id),
        ("receiver_universe_id", receiver_universe_id),
        ("receiver_family_universe_id", receiver_family_universe_id),
        ("family_axis_id", family_axis_id),
        ("semantics", GRAPH_FUSED_FAMILY_EFFECT_SEMANTICS),
    ):
        if set(table[column].astype(str)) != {expected}:
            raise ValueError(f"family_effects.{column} changed source lineage")
    if set(table["status"].astype(str)).difference(_ROW_STATUSES):
        raise ValueError("family_effects contains an unsupported status")
    if any(type(value) is not bool for value in table["experimental"].tolist()):
        raise ValueError("family_effects.experimental must be boolean")
    if not bool(table["experimental"].all()):
        raise ValueError("graph-fused family effects remain experimental")
    if any(
        type(value) is not bool for value in table["formal_inference_allowed"].tolist()
    ) or bool(table["formal_inference_allowed"].any()):
        raise ValueError("family_effects cannot claim formal inference")
    numeric = (
        "family_coefficient",
        "full_loss",
        "loss_without",
        "raw_conditional_gain",
        "bounded_conditional_gain",
    )
    for row in table.itertuples(index=False):
        status = str(row.status)
        reason = _optional_string(row.reason_code)
        support_status = _optional_string(row.receiver_training_support_status)
        support_reason = _optional_string(
            row.receiver_training_support_reason_code
        )
        training_parent_id = _optional_string(
            row.receiver_family_training_artifact_id
        )
        family_basis_id = _optional_string(row.family_basis_id)
        if (
            _optional_string(row.receiver_training_support_id) is None
            or _optional_string(row.receiver_family_opportunity_id) is None
            or support_status not in {"observed", "not_estimable"}
        ):
            raise ValueError("family effects lack frozen support/opportunity lineage")
        if support_status == "not_estimable":
            model_lineage = (
                _optional_string(row.workflow_id),
                _optional_string(row.application_id),
                _optional_string(row.problem_id),
                _optional_string(row.fit_id),
                training_parent_id,
                family_basis_id,
            )
            if (
                support_reason != "receiver_absent_in_outer_training"
                or status != "not_estimable"
                or reason != support_reason
                or any(value is not None for value in model_lineage)
            ):
                raise ValueError(
                    "training-absent family opportunities must be typed NE "
                    "without model parents"
                )
        elif (
            support_reason is not None
            or training_parent_id is None
            or family_basis_id is None
        ):
            raise ValueError(
                "training-supported family effects require their real family parents"
            )
        source_digests = (
            _optional_string(row.coefficient_digest),
            _optional_string(row.fixed_precision_digest),
            _optional_string(row.heldout_response_digest),
            _optional_string(row.context_loss_digest),
        )
        values = {name: getattr(row, name) for name in numeric}
        if status == "not_estimable":
            if (
                any(not _missing(value) for value in values.values())
                or any(value is not None for value in source_digests)
                or reason is None
            ):
                raise ValueError(
                    "not-estimable family effects require missing values and a reason"
                )
            continue
        if any(_missing(value) for value in values.values()) or any(
            value is None for value in source_digests
        ):
            raise ValueError("estimable family effects require complete numeric values")
        coefficient = float(values["family_coefficient"])
        full_loss = float(values["full_loss"])
        loss_without = float(values["loss_without"])
        raw_gain = float(values["raw_conditional_gain"])
        bounded_gain = float(values["bounded_conditional_gain"])
        if (
            not all(
                math.isfinite(value)
                for value in (
                    coefficient,
                    full_loss,
                    loss_without,
                    raw_gain,
                    bounded_gain,
                )
            )
            or coefficient < 0
            or full_loss < 0
            or loss_without < 0
            or not 0 <= bounded_gain <= 1
        ):
            raise ValueError("family-effect numeric values are outside their domain")
        if status == "structural_zero":
            if (
                coefficient != 0.0
                or raw_gain != 0.0
                or bounded_gain != 0.0
                or reason != _STRUCTURAL_ZERO_REASON
            ):
                raise ValueError(
                    "structural-zero family effects require an exact zero coefficient"
                )
        elif (
            status != "observed"
            or coefficient == 0.0
            or reason is not None
            or not math.isclose(
                bounded_gain,
                float(np.clip(raw_gain, 0.0, 1.0)),
                rel_tol=1.0e-12,
                abs_tol=1.0e-14,
            )
        ):
            raise ValueError("observed family effects have invalid value semantics")


def _status_counts(table: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    return tuple(
        (status, int(table["status"].eq(status).sum()))
        for status in ("observed", "structural_zero", "not_estimable")
    )


@dataclass(frozen=True, slots=True, init=False)
class GraphFusedFamilyEffectCollection:
    """Tamper-evident complete family-effect grid for one graph registry."""

    collection_id: str
    crossfit_id: str
    registry_id: str
    graph_id: str
    receiver_universe_id: str
    receiver_family_universe_id: str
    family_axis_id: str
    semantics: str
    table_digest: str
    row_count: int
    status_counts: tuple[tuple[str, int], ...]
    experimental: bool
    formal_inference_allowed: bool
    _family_effects: pd.DataFrame = field(repr=False)
    _source_crossfit: CrossFitArtifacts = field(repr=False)
    _source_registry: GraphFusedCrossFitRegistry = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "GraphFusedFamilyEffectCollection is producer-owned; use "
            "derive_graph_fused_family_effects()"
        )

    @classmethod
    def _from_table(
        cls,
        crossfit: CrossFitArtifacts,
        registry: GraphFusedCrossFitRegistry,
        table: pd.DataFrame,
        *,
        graph_id: str,
    ) -> GraphFusedFamilyEffectCollection:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "crossfit_id": crossfit.crossfit_id,
            "registry_id": registry.registry_id,
            "graph_id": graph_id,
            "receiver_universe_id": registry.receiver_universe_id,
            "receiver_family_universe_id": registry.receiver_family_universe_id,
            "family_axis_id": registry.family_axis_id,
            "semantics": GRAPH_FUSED_FAMILY_EFFECT_SEMANTICS,
            "table_digest": _table_digest(table),
            "row_count": len(table),
            "status_counts": _status_counts(table),
            "experimental": True,
            "formal_inference_allowed": False,
            "_family_effects": table.copy(deep=True),
            "_source_crossfit": crossfit,
            "_source_registry": registry,
            "_producer_marker": _COLLECTION_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "collection_id",
            stable_id(
                "graph_fused_family_effect_collection",
                self._identity_payload(),
                schema_version=_IDENTITY_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "crossfit_id": self.crossfit_id,
            "gain_epsilon": _GAIN_EPSILON,
            "experimental": self.experimental,
            "formal_inference_allowed": self.formal_inference_allowed,
            "graph_id": self.graph_id,
            "receiver_universe_id": self.receiver_universe_id,
            "receiver_family_universe_id": self.receiver_family_universe_id,
            "family_axis_id": self.family_axis_id,
            "registry_id": self.registry_id,
            "row_count": self.row_count,
            "semantics": self.semantics,
            "status_counts": [list(item) for item in self.status_counts],
            "table_digest": self.table_digest,
        }

    def _require_intact(self) -> None:
        try:
            sources = _validate_sources(
                self._source_crossfit,
                self._source_registry,
            )
            expected = _build_table(
                self._source_crossfit,
                self._source_registry,
                sources,
            )
            _validate_table(
                self._family_effects,
                crossfit_id=self.crossfit_id,
                registry_id=self.registry_id,
                graph_id=self.graph_id,
                receiver_universe_id=self._source_registry.receiver_universe_id,
                receiver_family_universe_id=(
                    self._source_registry.receiver_family_universe_id
                ),
                family_axis_id=self._source_registry.family_axis_id,
            )
            valid = (
                self._producer_marker == _COLLECTION_MARKER
                and self.crossfit_id == self._source_crossfit.crossfit_id
                and self.registry_id == self._source_registry.registry_id
                and self.graph_id == sources.graph_id
                and self.receiver_universe_id
                == self._source_registry.receiver_universe_id
                and self.receiver_family_universe_id
                == self._source_registry.receiver_family_universe_id
                and self.family_axis_id == self._source_registry.family_axis_id
                and self.semantics == GRAPH_FUSED_FAMILY_EFFECT_SEMANTICS
                and self.table_digest == _table_digest(self._family_effects)
                and self.table_digest == _table_digest(expected)
                and self._family_effects.equals(expected)
                and self.row_count == len(self._family_effects) == len(expected)
                and self.status_counts == _status_counts(self._family_effects)
                and self.experimental is True
                and self.formal_inference_allowed is False
                and self.collection_id
                == stable_id(
                    "graph_fused_family_effect_collection",
                    self._identity_payload(),
                    schema_version=_IDENTITY_SCHEMA_VERSION,
                )
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Graph-fused family-effect collection failed integrity validation",
                code="graph_fused_family_effect_collection_integrity_violation",
                field="collection_id",
                remediation=(
                    "Re-derive it from intact CrossFitArtifacts and its exact "
                    "graph-fused registry"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Graph-fused family-effect collection failed integrity validation",
                code="graph_fused_family_effect_collection_integrity_violation",
                field="collection_id",
                remediation=(
                    "Re-derive it from intact CrossFitArtifacts and its exact "
                    "graph-fused registry"
                ),
            )

    @property
    def family_effects(self) -> pd.DataFrame:
        """Return the complete family-effect grid as a defensive copy."""

        self._require_intact()
        return self._family_effects.copy(deep=True)

    @property
    def table(self) -> pd.DataFrame:
        """Alias for :attr:`family_effects`."""

        return self.family_effects

    def to_manifest(self) -> dict[str, object]:
        """Return the collection identity and explicit descriptive scope."""

        self._require_intact()
        return {
            "collection_id": self.collection_id,
            "schema_version": _SCHEMA_VERSION,
            "producer": _PRODUCER,
            **self._identity_payload(),
            "grain": "outer_fold_x_receiver_x_subject_x_context_x_family",
            "primary_key": list(GRAPH_FUSED_FAMILY_EFFECT_KEY),
            "columns": list(GRAPH_FUSED_FAMILY_EFFECT_COLUMNS),
            "status": "complete_descriptive",
            "p_value": None,
            "q_value": None,
            "comm_probability": None,
        }


def derive_graph_fused_family_effects(
    crossfit: CrossFitArtifacts,
    registry: GraphFusedCrossFitRegistry,
) -> GraphFusedFamilyEffectCollection:
    """Derive held-out leave-one-family effects without fitting any new model."""

    sources = _validate_sources(crossfit, registry)
    table = _build_table(crossfit, registry, sources)
    _validate_table(
        table,
        crossfit_id=crossfit.crossfit_id,
        registry_id=registry.registry_id,
        graph_id=sources.graph_id,
        receiver_universe_id=registry.receiver_universe_id,
        receiver_family_universe_id=registry.receiver_family_universe_id,
        family_axis_id=registry.family_axis_id,
    )
    return GraphFusedFamilyEffectCollection._from_table(
        crossfit,
        registry,
        table,
        graph_id=sources.graph_id,
    )


__all__ = [
    "GRAPH_FUSED_FAMILY_EFFECT_COLUMNS",
    "GRAPH_FUSED_FAMILY_EFFECT_KEY",
    "GRAPH_FUSED_FAMILY_EFFECT_SEMANTICS",
    "GraphFusedFamilyEffectCollection",
    "derive_graph_fused_family_effects",
]
