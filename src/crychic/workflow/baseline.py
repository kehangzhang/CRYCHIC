"""End-to-end orchestration for the v0.1 exploratory baseline."""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from anndata import AnnData

from crychic.attribution import (
    AttributionSupportMethod,
    attribute_target_prior,
    downstream_attribution_support,
    winsorized_normalized_precision,
)
from crychic.availability import (
    AvailabilityParameters,
    BatchAvailability,
    HillParameters,
    estimate_bundle_availability,
    hill_transform,
)
from crychic.core import CommunicationMode, CrychicConfig, CrychicError, stable_id
from crychic.data import (
    ExpressionTransform,
    InputSchema,
    ValidatedInput,
    validate_anndata,
)
from crychic.design import (
    ContextGraph,
    ContrastSpec,
    audit_sample_design,
    audit_subject_design,
    canonical_context,
    context_id,
    global_contrasts,
    local_contrasts,
)
from crychic.pseudobulk import aggregate_pseudobulk
from crychic.resources import Interaction, ResourceBundle, TargetPrior
from crychic.response import ResponseEstimate, ResponseStatus, estimate_gene_response
from crychic.scoring import (
    CORE_COMPONENTS,
    CommunicationScores,
    ScoringFunctional,
    ScoringModelManifest,
    float64_array_digest,
    score_communication,
)
from crychic.sender import (
    SenderAssignment,
    SenderEvidenceParameters,
    assign_senders,
)

from .contracts import (
    EDGE_EVIDENCE_COLUMNS,
    BaselineArtifacts,
    BaselineAttributionRun,
    BaselineDryRunPlan,
    BaselineMode,
    BaselineScoreRun,
    PlanStatus,
    RunStatus,
    StagePlan,
)

_METHOD_VERSION = "0.1.0-exploratory"
_JOIN_KEYS = ["sample_id", "subject_id", "context", "receiver", "interaction_id"]
_DEFAULT_AVAILABILITY_PARAMETERS = AvailabilityParameters()
_PRECISION_LOWER_QUANTILE = 0.05
_PRECISION_UPPER_QUANTILE = 0.95
_MIN_POSITIVE_PRECISION_FEATURES = 2
_PRECISION_METHOD = "winsorized_median_normalized_v1"
_TRACKED_GEOMETRIC_SCORE_VERSION = "geometric_v1_tracked"


def _workflow_contrasts(graph: ContextGraph) -> tuple[ContrastSpec, ...]:
    """Keep one canonical name for each distinct global/local estimand."""
    if len(graph.nodes) < 2:
        return ()
    candidates = (*global_contrasts(graph), *local_contrasts(graph))
    selected: list[ContrastSpec] = []
    vectors: list[np.ndarray] = []
    for contrast in candidates:
        vector = contrast.vector(graph.nodes)
        if any(
            np.allclose(vector, previous, rtol=0.0, atol=1e-12)
            for previous in vectors
        ):
            continue
        selected.append(contrast)
        vectors.append(vector)
    return tuple(selected)


def _input_schema(config: CrychicConfig) -> InputSchema:
    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig instance")
    return InputSchema(
        context_keys=tuple(config.context_keys),
        counts_layer=config.counts_layer,
        sample_key=config.sample_key,
        subject_key=config.subject_key,
        cell_type_key=config.cell_type_key,
        covariates=tuple(config.covariates),
        expression_layer=config.expression_layer,
        expression_source=config.expression_source,
        expression_transform=cast(
            ExpressionTransform | None, config.expression_transform
        ),
        normalized_zero_is_nondetection=config.normalized_zero_is_nondetection,
        species=config.species,
        gene_namespace=config.gene_namespace,
        allow_duplicate_genes=config.allow_duplicate_genes,
    )


def _plain(value: Any) -> Hashable:
    result = value.item() if isinstance(value, np.generic) else value
    if not isinstance(result, Hashable):
        raise TypeError(f"context value must be hashable, got {type(result)!r}")
    return result


def _canonical_context(
    row: Mapping[str, Any] | pd.Series, context_keys: Sequence[str]
) -> tuple[tuple[str, Hashable], ...]:
    return canonical_context(row, context_keys)


def _canonical_context_id(
    row: Mapping[str, Any] | pd.Series, context_keys: Sequence[str]
) -> str:
    return context_id(row, context_keys)


def _string_identifier(value: Any) -> str:
    plain = value.item() if isinstance(value, np.generic) else value
    if isinstance(plain, str):
        return plain
    return json.dumps(
        {
            "type": f"{type(plain).__module__}.{type(plain).__qualname__}",
            "value": plain,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _match_context_node(
    canonical: tuple[tuple[str, Hashable], ...], graph: ContextGraph
) -> Hashable:
    candidates: list[Hashable] = [canonical]
    if len(canonical) == 1:
        candidates.append(canonical[0][1])
    matches = [candidate for candidate in candidates if candidate in graph.nodes]
    if len(matches) > 1:
        raise ValueError(f"context {canonical!r} ambiguously matches ContextGraph")
    if not matches:
        raise ValueError(f"context {canonical!r} is absent from ContextGraph")
    return matches[0]


def _resolve_context_graph(
    validated: ValidatedInput, supplied: ContextGraph | None
) -> ContextGraph:
    contexts = tuple(
        _canonical_context(row, validated.schema.context_keys)
        for _, row in validated.report.sample_metadata.iterrows()
    )
    if supplied is None:
        if len(validated.schema.context_keys) == 1:
            nodes: tuple[Hashable, ...] = tuple(
                dict.fromkeys(context[0][1] for context in contexts)
            )
        else:
            nodes = tuple(dict.fromkeys(contexts))
        return ContextGraph.complete(nodes)
    if not isinstance(supplied, ContextGraph):
        raise TypeError("context_graph must be a ContextGraph or None")
    for context in contexts:
        _match_context_node(context, supplied)
    return supplied


def _canonical_label(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _resource_failures(
    config: CrychicConfig,
    bundle: ResourceBundle | None,
    prior: TargetPrior | None,
) -> tuple[str, ...]:
    failures: list[str] = []
    if bundle is None:
        return ("resource_bundle_missing",)
    if config.species is not None and _canonical_label(config.species) != (
        _canonical_label(bundle.species.value)
    ):
        failures.append("resource_bundle_species_mismatch")
    if config.gene_namespace is not None and _canonical_label(
        config.gene_namespace
    ) != _canonical_label(bundle.gene_namespace.value):
        failures.append("resource_bundle_namespace_mismatch")
    if prior is not None:
        if prior.species is not bundle.species:
            failures.append("target_prior_species_mismatch")
        if prior.gene_namespace is not bundle.gene_namespace:
            failures.append("target_prior_namespace_mismatch")
        if prior.direction != 1:
            failures.append("target_prior_direction_unsupported")
    return tuple(sorted(set(failures)))


def _interaction_driver(interaction: Interaction, prior: TargetPrior) -> str | None:
    drivers = set(prior.driver_ids)
    if prior.driver_kind == "interaction":
        return (
            interaction.interaction_id
            if interaction.interaction_id in drivers
            else None
        )
    candidates = {interaction.ligand_name}
    if len(interaction.ligand_subunits) == 1:
        candidates.add(interaction.ligand_subunits[0])
    matches = sorted(candidates.intersection(drivers))
    return matches[0] if len(matches) == 1 else None


def _driver_by_interaction(
    bundle: ResourceBundle, prior: TargetPrior
) -> dict[str, str]:
    result: dict[str, str] = {}
    for interaction in bundle.interactions:
        driver = _interaction_driver(interaction, prior)
        if driver is not None:
            result[interaction.interaction_id] = driver
    return result


def _has_gated_target_support(
    prior: TargetPrior,
    response: ResponseEstimate,
    gates: Mapping[str, float],
    driver_by_interaction: Mapping[str, str],
) -> bool:
    features = set(response.feature_ids)
    mapped_drivers = set(driver_by_interaction.values())
    return any(
        gates[driver] > 0
        and driver in mapped_drivers
        and any(
            link.weight > 0 and link.target in features
            for link in prior.links_for_driver(driver)
        )
        for driver in prior.driver_ids
    )


def _support_table(
    validated: ValidatedInput,
    *,
    min_cells: int,
    min_samples_per_context: int,
    min_subjects_per_context: int,
) -> pd.DataFrame:
    schema = validated.schema
    columns = [
        schema.sample_key,
        schema.subject_key,
        *schema.context_keys,
        schema.cell_type_key,
    ]
    units = (
        validated.adata.obs.loc[:, columns]
        .groupby(columns, observed=True, dropna=False, sort=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    units["state_eligible"] = units["n_cells"] >= min_cells
    group_keys = [*schema.context_keys, schema.cell_type_key]
    records: list[dict[str, Any]] = []
    for values, group in units.groupby(
        group_keys, observed=True, dropna=False, sort=False
    ):
        key_values = values if isinstance(values, tuple) else (values,)
        eligible = group.loc[group["state_eligible"]]
        record = dict(zip(group_keys, key_values, strict=True))
        record.update(
            {
                "n_cells": int(group["n_cells"].sum()),
                "n_observed_samples": int(group[schema.sample_key].nunique()),
                "n_eligible_samples": int(eligible[schema.sample_key].nunique()),
                "n_eligible_subjects": int(eligible[schema.subject_key].nunique()),
                "min_cells_per_unit": min_cells,
            }
        )
        record["response_support_ready"] = (
            record["n_eligible_samples"] >= min_samples_per_context
            and record["n_eligible_subjects"] >= min_subjects_per_context
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(
        group_keys,
        key=lambda column: column.astype(str),
        kind="stable",
        ignore_index=True,
    )


def _resource_table(
    validated: ValidatedInput,
    bundle: ResourceBundle | None,
    prior: TargetPrior | None,
    failures: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    features = set(validated.feature_ids)
    if bundle is None:
        rows.append(
            {
                "resource_kind": "lr_bundle",
                "resource_id": pd.NA,
                "version": pd.NA,
                "status": PlanStatus.BLOCKED.value,
                "source_entities": 0,
                "mapped_entities": 0,
                "reason_code": "resource_bundle_missing",
            }
        )
    else:
        mapped = sum(
            all(
                gene in features
                for gene in (*item.ligand_subunits, *item.receptor_subunits)
            )
            for item in bundle.interactions
        )
        bundle_reasons = tuple(
            reason for reason in failures if reason.startswith("resource_bundle")
        )
        if mapped == 0:
            bundle_reasons = (*bundle_reasons, "no_interactions_map_to_input")
        rows.append(
            {
                "resource_kind": "lr_bundle",
                "resource_id": bundle.resource_id,
                "version": bundle.version,
                "status": (
                    PlanStatus.BLOCKED.value
                    if bundle_reasons
                    else PlanStatus.READY.value
                ),
                "source_entities": len(bundle.interactions),
                "mapped_entities": mapped,
                "reason_code": ",".join(bundle_reasons) or None,
            }
        )
    if prior is None:
        rows.append(
            {
                "resource_kind": "target_prior",
                "resource_id": pd.NA,
                "version": pd.NA,
                "status": PlanStatus.OPTIONAL.value,
                "source_entities": 0,
                "mapped_entities": 0,
                "reason_code": "target_prior_not_provided",
            }
        )
    else:
        prior_reasons = tuple(
            reason for reason in failures if reason.startswith("target_prior")
        )
        mapped_targets = len(features.intersection(prior.target_ids))
        if mapped_targets == 0:
            prior_reasons = (*prior_reasons, "no_prior_targets_map_to_input")
        rows.append(
            {
                "resource_kind": "target_prior",
                "resource_id": prior.resource_id,
                "version": prior.version,
                "status": (
                    PlanStatus.BLOCKED.value
                    if prior_reasons
                    else PlanStatus.READY.value
                ),
                "source_entities": prior.nnz,
                "mapped_entities": mapped_targets,
                "reason_code": ",".join(prior_reasons) or None,
            }
        )
    return pd.DataFrame(rows)


def _build_dry_run_plan(
    validated: ValidatedInput,
    config: CrychicConfig,
    graph: ContextGraph,
    bundle: ResourceBundle | None,
    prior: TargetPrior | None,
    *,
    min_cells: int,
    min_samples_per_context: int,
    min_subjects_per_context: int,
) -> BaselineDryRunPlan:
    failures = _resource_failures(config, bundle, prior)
    design_audit = audit_sample_design(
        validated.report.sample_metadata,
        context_keys=config.context_keys,
        covariates=config.covariates,
        formula=config.design,
        sample_key=config.sample_key,
    )
    all_contrasts = (
        (*global_contrasts(graph), *local_contrasts(graph))
        if len(graph.nodes) > 1
        else ()
    )
    registered_contrasts = _workflow_contrasts(graph) if all_contrasts else ()
    contrast_table = design_audit.contrast_table(registered_contrasts)
    sample_design = validated.report.sample_metadata.copy(deep=True)
    sample_design["__context_node"] = [
        _match_context_node(
            _canonical_context(row, validated.schema.context_keys), graph
        )
        for _, row in sample_design.iterrows()
    ]
    subject_designs = [
        audit_subject_design(
            sample_design,
            tuple(contrast.weights),
            context_key="__context_node",
            subject_key=validated.schema.subject_key,
        )
        for contrast in registered_contrasts
    ]
    if not contrast_table.empty:
        contrast_table["formula_estimable"] = contrast_table["estimable"]
        contrast_table["formula_reason_code"] = contrast_table["reason_code"]
        contrast_table["subject_design"] = [
            audit.design.value for audit in subject_designs
        ]
        contrast_table["subject_design_estimable"] = [
            audit.ready for audit in subject_designs
        ]
        subject_reasons = pd.Series(
            [audit.reason_code for audit in subject_designs],
            dtype="object",
        )
        contrast_table["subject_design_reason_code"] = subject_reasons
        blocked_by_subjects = ~contrast_table["subject_design_estimable"]
        subject_only_failure = blocked_by_subjects & contrast_table["formula_estimable"]
        contrast_table.loc[subject_only_failure, "reason_code"] = subject_reasons.loc[
            subject_only_failure
        ]
        contrast_table["estimable"] = (
            contrast_table["formula_estimable"]
            & contrast_table["subject_design_estimable"]
        )
    contrasts_ready = bool(contrast_table.empty or contrast_table["estimable"].all())
    duplicate_fit_blocked = bool(validated.report.duplicate_genes)
    observed_nodes = {
        _match_context_node(
            _canonical_context(row, validated.schema.context_keys), graph
        )
        for _, row in validated.report.sample_metadata.iterrows()
    }
    unobserved_nodes = set(graph.nodes).difference(observed_nodes)
    warnings = list(validated.report.warnings)
    if duplicate_fit_blocked:
        warnings.append("duplicate_genes_fit_unsupported")
    if unobserved_nodes:
        warnings.append("context_graph_contains_unobserved_nodes")
    if prior is None:
        warnings.append("target_prior_not_provided")
    resource_table = _resource_table(validated, bundle, prior, failures)
    lr_row = resource_table.loc[resource_table["resource_kind"] == "lr_bundle"].iloc[0]
    lr_ready = lr_row["status"] == PlanStatus.READY.value
    multi_context = len(observed_nodes) > 1
    if not multi_context:
        warnings.append("single_context_no_response_contrast")
    if len(registered_contrasts) < len(all_contrasts):
        warnings.append("equivalent_contrast_aliases_omitted")
    warnings.extend(design_audit.reason_codes)
    warnings.extend(
        audit.reason_code for audit in subject_designs if audit.reason_code is not None
    )
    if not contrasts_ready:
        warnings.append("registered_contrast_not_estimable")
    prior_row = resource_table.loc[
        resource_table["resource_kind"] == "target_prior"
    ].iloc[0]
    if duplicate_fit_blocked:
        attribution_status = PlanStatus.BLOCKED
    elif prior is None or not multi_context:
        attribution_status = PlanStatus.SKIPPED
    elif prior_row["status"] == PlanStatus.READY.value:
        attribution_status = PlanStatus.READY
    else:
        attribution_status = PlanStatus.BLOCKED
    design_ready = bool(design_audit.ready and contrasts_ready)
    fit_ready = bool(lr_ready and design_ready and not duplicate_fit_blocked)
    stages = (
        StagePlan(
            "validate",
            PlanStatus.BLOCKED if duplicate_fit_blocked else PlanStatus.READY,
            (
                "duplicate gene identifiers are validation-only"
                if duplicate_fit_blocked
                else "AnnData contract validated"
            ),
        ),
        StagePlan(
            "pseudobulk",
            PlanStatus.BLOCKED if duplicate_fit_blocked else PlanStatus.READY,
            (
                "unique gene identifiers are required for fitting"
                if duplicate_fit_blocked
                else f"sample x cell-type aggregation with min_cells={min_cells}"
            ),
        ),
        StagePlan(
            "context_graph",
            PlanStatus.READY,
            f"{len(graph.nodes)} nodes and {len(graph.edges)} edges",
        ),
        StagePlan(
            "design_audit",
            PlanStatus.READY if design_ready else PlanStatus.BLOCKED,
            (
                f"formula={design_audit.formula}; rank={design_audit.rank}/"
                f"{design_audit.n_columns}; registered contrasts are estimable"
                if design_ready
                else "sample design is rank deficient, incomplete, or has "
                "non-estimable registered contrasts"
            ),
        ),
        StagePlan(
            "gene_response",
            (
                PlanStatus.BLOCKED
                if duplicate_fit_blocked or not design_ready
                else PlanStatus.READY
                if multi_context
                else PlanStatus.SKIPPED
            ),
            (
                "unique sample-level global and topology-local diagnostic contrasts"
                if multi_context
                else "single context: no response contrast is defined"
            ),
        ),
        StagePlan(
            "availability",
            PlanStatus.READY if fit_ready else PlanStatus.BLOCKED,
            "batch LR state and ecosystem availability",
        ),
        StagePlan(
            "sender_assignment",
            PlanStatus.READY if fit_ready else PlanStatus.BLOCKED,
            "context-level non-causal sender evidence from ligand availability",
        ),
        StagePlan(
            "attribution",
            attribution_status,
            "condition-blind receiver gates and positive TargetPrior channel",
        ),
        StagePlan(
            "scoring",
            attribution_status,
            "one exploratory in-sample functional shared by contrast contexts",
        ),
    )
    return BaselineDryRunPlan(
        config_digest=config.digest,
        input_schema=validated.schema,
        context_graph=graph,
        design_audit=design_audit,
        contrast_table=contrast_table,
        support_table=_support_table(
            validated,
            min_cells=min_cells,
            min_samples_per_context=min_samples_per_context,
            min_subjects_per_context=min_subjects_per_context,
        ),
        resource_table=resource_table,
        stages=stages,
        warnings=tuple(warnings),
        can_fit=fit_ready,
    )


def _validate_workflow_parameters(
    *,
    min_cells: int,
    min_samples_per_context: int,
    min_subjects_per_context: int,
    min_pooled_availability: float,
    prior_quality: float,
) -> None:
    for name, value, minimum in (
        ("min_cells", min_cells, 1),
        ("min_samples_per_context", min_samples_per_context, 2),
        ("min_subjects_per_context", min_subjects_per_context, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not 0 <= min_pooled_availability <= 1:
        raise ValueError("min_pooled_availability must lie in [0, 1]")
    if not math.isfinite(prior_quality) or not 0 <= prior_quality <= 1:
        raise ValueError("prior_quality must be finite and lie in [0, 1]")


def dry_run_baseline(
    adata: AnnData,
    config: CrychicConfig,
    *,
    resource_bundle: ResourceBundle | None = None,
    context_graph: ContextGraph | None = None,
    target_prior: TargetPrior | None = None,
    min_cells: int = 10,
    min_samples_per_context: int = 2,
    min_subjects_per_context: int = 2,
) -> BaselineDryRunPlan:
    """Validate input and emit the support/resource plan without fitting matrices."""

    _validate_workflow_parameters(
        min_cells=min_cells,
        min_samples_per_context=min_samples_per_context,
        min_subjects_per_context=min_subjects_per_context,
        min_pooled_availability=0.0,
        prior_quality=1.0,
    )
    schema = _input_schema(config)
    validated = validate_anndata(adata, schema)
    graph = _resolve_context_graph(validated, context_graph)
    return _build_dry_run_plan(
        validated,
        config,
        graph,
        resource_bundle,
        target_prior,
        min_cells=min_cells,
        min_samples_per_context=min_samples_per_context,
        min_subjects_per_context=min_subjects_per_context,
    )


def _sender_input(
    availability: BatchAvailability, config: CrychicConfig
) -> pd.DataFrame:
    source = availability.sample_interactions
    columns = [
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
        "ligand_availability",
    ]
    if source.empty:
        return pd.DataFrame(columns=columns)
    result = source.loc[
        :,
        [
            "sample_id",
            "subject_id",
            *config.context_keys,
            "sender",
            "receiver",
            "interaction_id",
            "ligand_availability",
        ],
    ].copy()
    result["context_id"] = [
        _canonical_context_id(row, config.context_keys) for _, row in result.iterrows()
    ]
    for column in (
        "sample_id",
        "subject_id",
        "sender",
        "receiver",
        "interaction_id",
    ):
        result[column] = result[column].map(_string_identifier)
    return result.loc[:, columns]


def _scoring_availability(
    availability: BatchAvailability,
    config: CrychicConfig,
    graph: ContextGraph,
    sender_assignment: SenderAssignment,
) -> pd.DataFrame:
    source = availability.sample_interactions.copy(deep=True)
    if source.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "subject_id",
                "context",
                "context_id",
                "sender",
                "receiver",
                "interaction_id",
                "state_availability",
                "state_availability_status",
                "state_availability_reason_code",
                "ecosystem_availability",
                "ecosystem_availability_status",
                "ecosystem_availability_reason_code",
                "sender_weight",
                "sender_status",
                "sender_reason_code",
                "sender_component",
            ]
        )
    source["context"] = [
        _match_context_node(_canonical_context(row, config.context_keys), graph)
        for _, row in source.iterrows()
    ]
    source["context_id"] = [
        _canonical_context_id(row, config.context_keys) for _, row in source.iterrows()
    ]
    key_columns = {
        "receiver": "_receiver_id",
        "interaction_id": "_interaction_id",
        "sender": "_sender_id",
    }
    for source_column, key_column in key_columns.items():
        source[key_column] = source[source_column].map(_string_identifier)
    assignment = (
        sender_assignment.table.rename(columns=key_columns)
        .rename(
            columns={
                "status": "sender_status",
                "reason_code": "sender_reason_code",
            }
        )
        .loc[
            :,
            [
                "context_id",
                *key_columns.values(),
                "assignment_weight",
                "sender_status",
                "sender_reason_code",
            ],
        ]
    )
    source = source.merge(
        assignment,
        how="left",
        on=["context_id", *key_columns.values()],
        validate="many_to_one",
        sort=False,
    )
    if source["sender_status"].isna().any():
        raise ValueError("sender assignment does not cover every availability edge")
    supported_sender = source["sender_status"] == "ok"
    source["sender_component"] = (
        source["assignment_weight"].astype(float).where(supported_sender)
    )
    source["sender_weight"] = (
        source["assignment_weight"]
        .astype(float)
        .where(source["sender_status"] != "missing_evidence")
    )
    result = source.rename(
        columns={
            "availability_state": "state_availability",
            "availability_ecosystem": "ecosystem_availability",
            "state_status": "state_availability_status",
            "state_reason_code": "state_availability_reason_code",
            "ecosystem_status": "ecosystem_availability_status",
            "ecosystem_reason_code": "ecosystem_availability_reason_code",
        }
    ).loc[
        :,
        [
            "sample_id",
            "subject_id",
            "context",
            "context_id",
            "sender",
            "receiver",
            "interaction_id",
            "state_availability",
            "state_availability_status",
            "state_availability_reason_code",
            "ecosystem_availability",
            "ecosystem_availability_status",
            "ecosystem_availability_reason_code",
            "sender_weight",
            "sender_status",
            "sender_reason_code",
            "sender_component",
        ],
    ]
    return result.sort_values(
        ["sample_id", "sender", "receiver", "interaction_id"],
        key=lambda column: column.astype(str),
        kind="stable",
        ignore_index=True,
    )


def _pooled_receptor_gates(
    availability: BatchAvailability,
    receiver: Hashable,
    prior: TargetPrior,
    driver_by_interaction: Mapping[str, str],
) -> dict[str, float]:
    gates = dict.fromkeys(prior.driver_ids, 0.0)
    table = availability.sample_interactions
    if table.empty:
        return gates
    selected = table.loc[table["receiver"].map(lambda value: value == receiver)].copy()
    if selected.empty:
        return gates
    selected["driver_id"] = (
        selected["interaction_id"].astype(str).map(driver_by_interaction)
    )
    selected = selected.dropna(subset=["driver_id"])
    if selected.empty:
        return gates
    # Receptor values repeat over candidate senders. Pool once per sample/LR,
    # then over contexts without consulting a context label.
    per_interaction = selected.groupby(
        ["sample_id", "driver_id", "interaction_id"], observed=True, sort=False
    )["receptor_availability"].max()
    per_sample = per_interaction.groupby(
        ["sample_id", "driver_id"], observed=True, sort=False
    ).max()
    pooled = per_sample.groupby("driver_id", observed=True, sort=False).mean()
    for driver, value in pooled.items():
        gates[str(driver)] = float(min(1.0, max(0.0, value)))
    return gates


def _global_specs(response: ResponseEstimate) -> tuple[ContrastSpec, ...]:
    return tuple(
        spec for spec in response.contrast_specs if spec.mode == "global_one_vs_rest"
    )


def _response_rows(
    response: ResponseEstimate, receiver: Hashable, contrast: str
) -> pd.DataFrame:
    selected = response.contrasts.loc[
        response.contrasts["receiver"].map(lambda value: value == receiver)
        & (response.contrasts["contrast"] == contrast)
    ].copy()
    if selected.empty or selected["gene"].duplicated().any():
        return selected
    return (
        selected.set_index("gene", drop=False)
        .reindex(response.feature_ids)
        .reset_index(drop=True)
    )


def _failure_code(error: Exception) -> str:
    if isinstance(error, CrychicError):
        return str(error.details.code)
    return f"{type(error).__name__}:{error!s}"


def _run_attribution(
    response: ResponseEstimate,
    availability: BatchAvailability,
    bundle: ResourceBundle,
    prior: TargetPrior | None,
    *,
    lambda1: float,
    lambda2: float,
    cosine_threshold: float,
) -> tuple[BaselineAttributionRun, ...]:
    receivers = tuple(
        sorted(set(response.contrasts["receiver"]), key=lambda value: str(value))
    )
    mapping = {} if prior is None else _driver_by_interaction(bundle, prior)
    mapping_tuple = tuple(sorted(mapping.items()))
    runs: list[BaselineAttributionRun] = []
    for spec in _global_specs(response):
        contexts = tuple(spec.weights)
        for receiver in receivers:
            selected = _response_rows(response, receiver, spec.name)
            if selected.empty:
                runs.append(
                    BaselineAttributionRun(
                        receiver=receiver,
                        contrast=spec.name,
                        contrast_contexts=contexts,
                        status=RunStatus.NOT_ESTIMABLE,
                        reason_code="response_rows_missing",
                        receptor_gates=(),
                        driver_by_interaction=mapping_tuple,
                        basis=None,
                        attribution=None,
                    )
                )
                continue
            statuses = set(selected["status"])
            effects = pd.to_numeric(selected["effect"], errors="coerce").to_numpy(
                dtype=float
            )
            if statuses != {ResponseStatus.OK.value} or not np.isfinite(effects).all():
                reasons = sorted(
                    set(selected["reason_code"].dropna().astype(str))
                    or {"response_not_estimable"}
                )
                runs.append(
                    BaselineAttributionRun(
                        receiver=receiver,
                        contrast=spec.name,
                        contrast_contexts=contexts,
                        status=RunStatus.NOT_ESTIMABLE,
                        reason_code=",".join(reasons),
                        receptor_gates=(),
                        driver_by_interaction=mapping_tuple,
                        basis=None,
                        attribution=None,
                    )
                )
                continue
            if prior is None:
                runs.append(
                    BaselineAttributionRun(
                        receiver=receiver,
                        contrast=spec.name,
                        contrast_contexts=contexts,
                        status=RunStatus.UNAVAILABLE,
                        reason_code="target_prior_not_provided",
                        receptor_gates=(),
                        driver_by_interaction=(),
                        basis=None,
                        attribution=None,
                    )
                )
                continue
            gates = _pooled_receptor_gates(availability, receiver, prior, mapping)
            gate_tuple = tuple((driver, gates[driver]) for driver in prior.driver_ids)
            if not mapping:
                runs.append(
                    BaselineAttributionRun(
                        receiver=receiver,
                        contrast=spec.name,
                        contrast_contexts=contexts,
                        status=RunStatus.UNAVAILABLE,
                        reason_code="no_prior_driver_maps_to_lr_bundle",
                        receptor_gates=gate_tuple,
                        driver_by_interaction=(),
                        basis=None,
                        attribution=None,
                    )
                )
                continue
            if not any(gates.values()):
                runs.append(
                    BaselineAttributionRun(
                        receiver=receiver,
                        contrast=spec.name,
                        contrast_contexts=contexts,
                        status=RunStatus.UNAVAILABLE,
                        reason_code="no_condition_blind_receptor_support",
                        receptor_gates=gate_tuple,
                        driver_by_interaction=mapping_tuple,
                        basis=None,
                        attribution=None,
                    )
                )
                continue
            if not _has_gated_target_support(prior, response, gates, mapping):
                runs.append(
                    BaselineAttributionRun(
                        receiver=receiver,
                        contrast=spec.name,
                        contrast_contexts=contexts,
                        status=RunStatus.UNAVAILABLE,
                        reason_code="no_gated_prior_target_matches_response",
                        receptor_gates=gate_tuple,
                        driver_by_interaction=mapping_tuple,
                        basis=None,
                        attribution=None,
                    )
                )
                continue
            raw_precision = pd.to_numeric(
                selected["precision"], errors="coerce"
            ).to_numpy(dtype=float)
            precision_transform = winsorized_normalized_precision(
                raw_precision,
                lower_quantile=_PRECISION_LOWER_QUANTILE,
                upper_quantile=_PRECISION_UPPER_QUANTILE,
                min_positive_features=_MIN_POSITIVE_PRECISION_FEATURES,
            )
            if not precision_transform.estimable:
                runs.append(
                    BaselineAttributionRun(
                        receiver=receiver,
                        contrast=spec.name,
                        contrast_contexts=contexts,
                        status=RunStatus.NOT_ESTIMABLE,
                        reason_code=precision_transform.reason_code,
                        receptor_gates=gate_tuple,
                        driver_by_interaction=mapping_tuple,
                        basis=None,
                        attribution=None,
                        precision_method=_PRECISION_METHOD,
                        precision_transform_id=(
                            precision_transform.precision_transform_id
                        ),
                        precision_lower_quantile=(
                            precision_transform.lower_quantile
                        ),
                        precision_upper_quantile=(
                            precision_transform.upper_quantile
                        ),
                        n_positive_precision_features=(
                            precision_transform.n_positive_features
                        ),
                    )
                )
                continue
            try:
                basis, result = attribute_target_prior(
                    prior,
                    response.feature_ids,
                    gates,
                    effects,
                    precision_weights=precision_transform.values,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    cosine_threshold=cosine_threshold,
                )
                status = RunStatus.OK if result.succeeded else RunStatus.FAILED
                reason = (
                    None
                    if result.succeeded
                    else f"solver_{result.diagnostics.status.value}"
                )
            except (CrychicError, ValueError, FloatingPointError) as error:
                basis = None
                result = None
                status = RunStatus.FAILED
                reason = _failure_code(error)
            runs.append(
                BaselineAttributionRun(
                    receiver=receiver,
                    contrast=spec.name,
                    contrast_contexts=contexts,
                    status=status,
                    reason_code=reason,
                    receptor_gates=gate_tuple,
                    driver_by_interaction=mapping_tuple,
                    basis=basis,
                    attribution=result,
                    precision_method=_PRECISION_METHOD,
                    precision_transform_id=precision_transform.precision_transform_id,
                    precision_lower_quantile=precision_transform.lower_quantile,
                    precision_upper_quantile=precision_transform.upper_quantile,
                    n_positive_precision_features=(
                        precision_transform.n_positive_features
                    ),
                )
            )
    return tuple(runs)


def _context_mask(series: pd.Series, contexts: Sequence[Hashable]) -> pd.Series:
    return series.map(
        lambda value: any(value == context for context in contexts)
    ).astype(bool)


def _value_mask(series: pd.Series, expected: Hashable) -> pd.Series:
    return series.map(lambda value: value == expected).astype(bool)


def _receiver_sample_rows(
    response: ResponseEstimate, receiver: Hashable
) -> dict[Hashable, int]:
    selected = response.sample_metadata.loc[
        response.sample_metadata["cell_type"].map(lambda value: value == receiver)
    ]
    return {
        _plain(row["sample_id"]): int(cast(int, index))
        for index, row in selected.iterrows()
    }


def _downstream_table(
    base: pd.DataFrame,
    run: BaselineAttributionRun,
    response: ResponseEstimate,
    prior: TargetPrior,
    *,
    hill_parameters: HillParameters,
    prior_quality: float,
    attribution_support_method: AttributionSupportMethod,
) -> pd.DataFrame:
    driver_by_interaction = dict(run.driver_by_interaction)
    sample_rows = _receiver_sample_rows(response, run.receiver)
    support = (
        None
        if run.attribution is None
        or run.basis is None
        else downstream_attribution_support(
            run.basis,
            run.attribution,
            method=attribution_support_method,
        )
    )
    records: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        interaction_id = str(row.interaction_id)
        driver = driver_by_interaction.get(interaction_id)
        activity: float | None = None
        receiver_program_score: float | None = None
        quality: float | None = None
        status = "missing_core_evidence"
        reason = run.reason_code
        receiver_program_status = "missing_core_evidence"
        receiver_program_reason = run.reason_code
        target_weight_id: str | None = None
        support_value: float | None = None
        support_numerator: float | None = None
        support_denominator: float | None = None
        if driver is None:
            reason = "interaction_not_in_target_prior"
            receiver_program_reason = reason
        elif not run.succeeded or run.basis is None or run.attribution is None:
            reason = run.reason_code or "attribution_unavailable"
            receiver_program_reason = reason
        else:
            column = run.basis.driver_ids.index(driver)
            profile = np.asarray(
                run.basis.normalized_profiles.getcol(column).toarray()
            ).ravel()
            profile_sum = float(profile.sum())
            if support is None:
                raise RuntimeError("successful attribution lacks support contract")
            support_value = float(support.values[column])
            support_numerator = float(support.numerator_values[column])
            support_denominator = support.denominator_value
            quality = prior_quality
            target_weight_payload = {
                "basis_id": run.basis.basis_id,
                "contrast": run.contrast,
                "driver_id": driver,
                "receiver": str(run.receiver),
            }
            if (
                attribution_support_method
                is not AttributionSupportMethod.RELATIVE_COEFFICIENT_V1
            ):
                target_weight_payload["attribution_support_method"] = (
                    attribution_support_method.value
                )
            target_weight_id = stable_id(
                "downstream_target_weights", target_weight_payload
            )
            if profile_sum <= 0:
                reason = "driver_basis_unavailable"
                receiver_program_reason = reason
                quality = None
            else:
                source_row = sample_rows.get(_plain(row.sample_id))
                if source_row is None or not bool(
                    response.sample_metadata.at[source_row, "response_eligible"]
                ):
                    reason = "receiver_response_unavailable"
                    receiver_program_reason = reason
                else:
                    values = response.sample_values[source_row]
                    if not np.isfinite(values).all():
                        reason = "receiver_response_unavailable"
                        receiver_program_reason = reason
                    else:
                        weights = profile / profile_sum
                        transformed = np.asarray(
                            [
                                hill_transform(float(value), hill_parameters)
                                for value in values
                            ],
                            dtype=float,
                        )
                        receiver_program_score = float(
                            np.clip(np.dot(weights, transformed), 0, 1)
                        )
                        receiver_program_status = "observed"
                        receiver_program_reason = None
                        if support_value <= 0:
                            activity = 0.0
                            status = "zero_attribution"
                            reason = None
                        else:
                            activity = float(
                                np.clip(
                                    receiver_program_score * support_value,
                                    0,
                                    1,
                                )
                            )
                            status = "observed"
                            reason = None
        records.append(
            {
                "sample_id": row.sample_id,
                "subject_id": row.subject_id,
                "context": row.context,
                "receiver": row.receiver,
                "interaction_id": row.interaction_id,
                "receiver_program_score": receiver_program_score,
                "receiver_program_status": receiver_program_status,
                "receiver_program_reason_code": receiver_program_reason,
                "incremental_downstream": None,
                "incremental_downstream_status": "not_estimable",
                "incremental_downstream_reason_code": (
                    "cross_fitted_receiver_null_not_implemented"
                ),
                "downstream_activity": activity,
                "prior_quality": quality,
                "contrast": run.contrast,
                "driver_id": driver,
                "downstream_status": status,
                "downstream_reason_code": reason,
                "target_weight_id": target_weight_id,
                "target_weight_method": attribution_support_method.value,
                "attribution_support": support_value,
                "attribution_support_numerator": support_numerator,
                "attribution_support_denominator": support_denominator,
                "prior_quality_source": (
                    f"constant={prior_quality};resource={prior.resource_id};"
                    f"version={prior.version};evidence={prior.evidence}"
                    if quality is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(records)


def _scoring_model_manifest(
    run: BaselineAttributionRun,
    response: ResponseEstimate,
    prior: TargetPrior,
    *,
    filter_universe_id: str,
    matched_targets: tuple[str, ...],
    availability_parameters: AvailabilityParameters,
    sender_functional_id: str,
    attribution_support_method: AttributionSupportMethod,
    cosine_threshold: float,
) -> ScoringModelManifest:
    if run.basis is None or run.attribution is None:
        raise ValueError("successful score run requires attribution artifacts")
    if run.precision_transform_id is None:
        raise ValueError("successful score run requires precision provenance")
    if not filter_universe_id:
        raise ValueError("successful score run requires a filter universe ID")
    receptor_gate_manifest_id = stable_id(
        "receptor_gate_manifest",
        {
            "gate_policy": "continuous_basis_scale_v1",
            "receiver": _string_identifier(run.receiver),
            "receptor_gates": [
                [driver, gate] for driver, gate in run.receptor_gates
            ],
        },
    )
    target_weight_manifest_id = stable_id(
        "target_weight_manifest",
        {
            "attribution_support_method": attribution_support_method.value,
            "basis_id": run.basis.basis_id,
            "matched_targets": list(matched_targets),
            "prior_manifest": prior.manifest_digest,
            "prior_resource_id": prior.resource_id,
            "prior_version": prior.version,
        },
    )
    downstream_functional_id = stable_id(
        "downstream_functional",
        {
            "attribution_support_method": attribution_support_method.value,
            "hill_coefficient": availability_parameters.hill.coefficient,
            "hill_half_saturation": availability_parameters.hill.half_saturation,
            "method": "absolute_receiver_program_hill_v1",
            "response_scale": response.value_scale,
            "target_weight_manifest_id": target_weight_manifest_id,
        },
    )
    availability_transform_id = stable_id(
        "availability_transform",
        {
            "complex_epsilon": availability_parameters.complex_epsilon,
            "complex_power": availability_parameters.complex_power,
            "detection_alpha": availability_parameters.detection.alpha,
            "detection_beta": availability_parameters.detection.beta,
            "detection_exponent": availability_parameters.detection.exponent,
            "hill_coefficient": availability_parameters.hill.coefficient,
            "hill_half_saturation": availability_parameters.hill.half_saturation,
        },
    )
    tuning_manifest_id = stable_id(
        "attribution_tuning_manifest",
        {
            "attribution_support_method": attribution_support_method.value,
            "cosine_threshold": cosine_threshold,
            "lambda1": run.attribution.lambda1,
            "lambda2": run.attribution.lambda2,
            "solver": "nonnegative_elastic_net_coordinate_descent_v1",
        },
    )
    return ScoringModelManifest(
        score_version=_TRACKED_GEOMETRIC_SCORE_VERSION,
        basis_id=run.basis.basis_id,
        coefficient_digest=float64_array_digest(run.attribution.coefficients),
        receptor_gate_manifest_id=receptor_gate_manifest_id,
        target_weight_manifest_id=target_weight_manifest_id,
        downstream_functional_id=downstream_functional_id,
        sender_functional_id=sender_functional_id,
        availability_transform_id=availability_transform_id,
        precision_transform_id=run.precision_transform_id,
        filter_universe_id=filter_universe_id,
        tuning_manifest_id=tuning_manifest_id,
    )


def _score_runs(
    response: ResponseEstimate,
    availability: pd.DataFrame,
    prior: TargetPrior | None,
    attribution_runs: tuple[BaselineAttributionRun, ...],
    *,
    availability_parameters: AvailabilityParameters,
    prior_quality: float,
    component_weights: Mapping[str, float] | None,
    component_scales: Mapping[str, float] | None,
    config_modes: Sequence[CommunicationMode | str],
    attribution_support_method: AttributionSupportMethod,
    sender_functional_id: str,
    filter_universe_id: str,
    cosine_threshold: float,
) -> tuple[BaselineScoreRun, ...]:
    if prior is None:
        return ()
    matched_targets = tuple(
        sorted(set(response.feature_ids).intersection(prior.target_ids))
    )
    if not matched_targets:
        return ()
    training_subjects = tuple(
        sorted(set(response.sample_metadata["subject_id"].astype(str)))
    )
    weights = (
        dict.fromkeys(CORE_COMPONENTS, 1.0)
        if component_weights is None
        else dict(component_weights)
    )
    scales = (
        dict.fromkeys(CORE_COMPONENTS, 1.0)
        if component_scales is None
        else dict(component_scales)
    )
    outputs: list[BaselineScoreRun] = []
    for run in attribution_runs:
        if not run.succeeded:
            continue
        receiver = run.receiver
        selected = availability.loc[
            _value_mask(availability["receiver"], receiver)
            & _context_mask(availability["context"], run.contrast_contexts)
        ].copy()
        if selected.empty:
            continue
        interaction_ids = tuple(sorted(set(selected["interaction_id"].astype(str))))
        if not interaction_ids:
            continue
        model_manifest = _scoring_model_manifest(
            run,
            response,
            prior,
            filter_universe_id=filter_universe_id,
            matched_targets=matched_targets,
            availability_parameters=availability_parameters,
            sender_functional_id=sender_functional_id,
            attribution_support_method=attribution_support_method,
            cosine_threshold=cosine_threshold,
        )
        functional = ScoringFunctional(
            contrast_name=run.contrast,
            contrast_contexts=run.contrast_contexts,
            training_subject_ids=training_subjects,
            interaction_ids=interaction_ids,
            target_ids=matched_targets,
            score_version=_TRACKED_GEOMETRIC_SCORE_VERSION,
            model_manifest=model_manifest,
            component_weights=weights,
            component_scales=scales,
        )
        base = selected.loc[:, _JOIN_KEYS].drop_duplicates(ignore_index=True)
        downstream = _downstream_table(
            base,
            run,
            response,
            prior,
            hill_parameters=availability_parameters.hill,
            prior_quality=prior_quality,
            attribution_support_method=attribution_support_method,
        )
        scores = score_communication(selected, downstream, functional)
        requested_modes = {CommunicationMode(mode).value for mode in config_modes}
        scores = CommunicationScores(
            table=scores.table.loc[
                scores.table["mode"].isin(requested_modes)
            ].reset_index(drop=True),
            functional=functional,
        )
        outputs.append(
            BaselineScoreRun(
                receiver=run.receiver,
                contrast=run.contrast,
                functional=functional,
                downstream_activity=downstream,
                scores=scores,
            )
        )
    return tuple(outputs)


def _empty_edge_evidence() -> pd.DataFrame:
    return pd.DataFrame(columns=EDGE_EVIDENCE_COLUMNS)


def _edge_modes(
    config_modes: Sequence[CommunicationMode | str],
) -> tuple[str, ...]:
    return tuple(CommunicationMode(mode).value for mode in config_modes)


def _expand_edge_modes(source: pd.DataFrame, modes: tuple[str, ...]) -> pd.DataFrame:
    return pd.concat(
        [source.assign(mode=mode) for mode in modes],
        ignore_index=True,
        sort=False,
    )


def _add_receptor_gate_evidence(
    table: pd.DataFrame, run: BaselineAttributionRun
) -> pd.DataFrame:
    result = table.copy(deep=True)
    mapping = dict(run.driver_by_interaction)
    gates = dict(run.receptor_gates)
    result["driver_id"] = result["interaction_id"].astype(str).map(mapping)
    known_gate = result["driver_id"].notna() & result["driver_id"].isin(gates)
    missing_driver = result["driver_id"].isna()
    result["receptor_gate"] = pd.to_numeric(
        result["driver_id"].map(gates), errors="coerce"
    ).where(known_gate)
    fallback_status = (
        RunStatus.UNAVAILABLE.value if run.succeeded else run.status.value
    )
    result["receptor_gate_status"] = fallback_status
    result["receptor_gate_reason_code"] = (
        run.reason_code or "receptor_gate_unavailable"
    )
    result.loc[missing_driver, "receptor_gate_status"] = "missing"
    result.loc[missing_driver, "receptor_gate_reason_code"] = (
        run.reason_code
        if not mapping and run.reason_code is not None
        else "interaction_not_in_target_prior"
    )
    result.loc[known_gate, "receptor_gate_status"] = "observed"
    result.loc[known_gate, "receptor_gate_reason_code"] = None
    return result


def _scored_edge_evidence(
    scoring_availability: pd.DataFrame,
    score_run: BaselineScoreRun,
    attribution_run: BaselineAttributionRun,
    *,
    emitted: bool,
) -> pd.DataFrame:
    source_columns = [
        *_JOIN_KEYS,
        "sender",
        "context_id",
        "state_availability",
        "state_availability_status",
        "state_availability_reason_code",
        "ecosystem_availability",
        "ecosystem_availability_status",
        "ecosystem_availability_reason_code",
        "sender_weight",
        "sender_status",
        "sender_reason_code",
    ]
    source = scoring_availability.loc[
        _value_mask(scoring_availability["receiver"], attribution_run.receiver)
        & _context_mask(
            scoring_availability["context"], attribution_run.contrast_contexts
        ),
        source_columns,
    ].copy()
    downstream_columns = [
        *_JOIN_KEYS,
        "receiver_program_score",
        "receiver_program_status",
        "receiver_program_reason_code",
        "incremental_downstream",
        "incremental_downstream_status",
        "incremental_downstream_reason_code",
        "downstream_status",
        "downstream_reason_code",
        "attribution_support",
        "target_weight_method",
        "prior_quality_source",
    ]
    result = score_run.scores.table.merge(
        source,
        how="left",
        on=[*_JOIN_KEYS, "sender"],
        validate="many_to_one",
        sort=False,
    ).merge(
        score_run.downstream_activity.loc[:, downstream_columns],
        how="left",
        on=_JOIN_KEYS,
        validate="many_to_one",
        sort=False,
    )
    if result["context_id"].isna().any():
        raise ValueError("edge evidence source does not cover every score row")
    result = _add_receptor_gate_evidence(result, attribution_run)
    support_observed = result["attribution_support"].notna()
    result["attribution_support_method"] = result["target_weight_method"]
    result["attribution_support_status"] = result["downstream_status"].where(
        ~support_observed, "observed"
    )
    result["attribution_support_reason_code"] = result["downstream_reason_code"].where(
        ~support_observed
    )
    quality_observed = result["prior_quality"].notna()
    result["prior_quality_status"] = result["downstream_status"].where(
        ~quality_observed, "observed"
    )
    result["prior_quality_reason_code"] = result["downstream_reason_code"].where(
        ~quality_observed
    )
    result["legacy_downstream_activity"] = result["downstream_activity"]
    result["legacy_downstream_status"] = result["downstream_status"]
    result["legacy_downstream_reason_code"] = result["downstream_reason_code"]
    result["legacy_integrated_strength"] = result["comm_strength"]
    result["legacy_integrated_status"] = result["status"]
    result["legacy_integrated_reason_code"] = result["reason_code"]
    result["scoring_function_status"] = result["functional_status"]
    result["scoring_function_reason_code"] = result["functional_reason_code"]
    result["sample_score_status"] = "linked" if emitted else "not_emitted"
    result["sample_score_reason_code"] = (
        None if emitted else "branch_has_no_complete_core_evidence"
    )
    return result.loc[:, list(EDGE_EVIDENCE_COLUMNS)]


def _unscored_edge_evidence(
    source: pd.DataFrame,
    run: BaselineAttributionRun,
    *,
    modes: tuple[str, ...],
    attribution_support_method: AttributionSupportMethod,
) -> pd.DataFrame:
    result = _expand_edge_modes(source, modes)
    result["contrast"] = run.contrast
    result["fold_id"] = "in_sample"
    result = _add_receptor_gate_evidence(result, run)
    unavailable_status = (
        run.status.value if not run.succeeded else RunStatus.UNAVAILABLE.value
    )
    unavailable_reason = run.reason_code or "scoring_function_unavailable"
    result["receiver_program_score"] = None
    result["receiver_program_status"] = unavailable_status
    result["receiver_program_reason_code"] = unavailable_reason
    result["incremental_downstream"] = None
    result["incremental_downstream_status"] = RunStatus.NOT_ESTIMABLE.value
    result["incremental_downstream_reason_code"] = (
        "cross_fitted_receiver_null_not_implemented"
    )
    result["attribution_support"] = None
    result["attribution_support_method"] = attribution_support_method.value
    result["attribution_support_status"] = unavailable_status
    result["attribution_support_reason_code"] = unavailable_reason
    result["prior_quality"] = None
    result["prior_quality_source"] = None
    result["prior_quality_status"] = unavailable_status
    result["prior_quality_reason_code"] = unavailable_reason
    result["legacy_downstream_activity"] = None
    result["legacy_downstream_status"] = unavailable_status
    result["legacy_downstream_reason_code"] = unavailable_reason
    result["legacy_integrated_strength"] = None
    result["legacy_integrated_status"] = unavailable_status
    result["legacy_integrated_reason_code"] = unavailable_reason
    result["scoring_function_status"] = unavailable_status
    result["scoring_function_reason_code"] = unavailable_reason
    result["score_version"] = None
    result["model_manifest_id"] = None
    result["scoring_function_id"] = None
    result["sample_score_status"] = "not_emitted"
    result["sample_score_reason_code"] = "scoring_function_unavailable"
    return result.loc[:, list(EDGE_EVIDENCE_COLUMNS)]


def _availability_only_edge_evidence(
    source: pd.DataFrame,
    *,
    modes: tuple[str, ...],
    attribution_support_method: AttributionSupportMethod,
    reason_code: str,
) -> pd.DataFrame:
    result = _expand_edge_modes(source, modes)
    result["driver_id"] = None
    result["contrast"] = "availability_only"
    result["fold_id"] = "in_sample"
    for value_column, status_column, reason_column in (
        ("receptor_gate", "receptor_gate_status", "receptor_gate_reason_code"),
        (
            "receiver_program_score",
            "receiver_program_status",
            "receiver_program_reason_code",
        ),
        (
            "attribution_support",
            "attribution_support_status",
            "attribution_support_reason_code",
        ),
        ("prior_quality", "prior_quality_status", "prior_quality_reason_code"),
        (
            "legacy_downstream_activity",
            "legacy_downstream_status",
            "legacy_downstream_reason_code",
        ),
        (
            "legacy_integrated_strength",
            "legacy_integrated_status",
            "legacy_integrated_reason_code",
        ),
    ):
        result[value_column] = None
        result[status_column] = RunStatus.NOT_ESTIMABLE.value
        result[reason_column] = reason_code
    result["incremental_downstream"] = None
    result["incremental_downstream_status"] = RunStatus.NOT_ESTIMABLE.value
    result["incremental_downstream_reason_code"] = (
        "cross_fitted_receiver_null_not_implemented"
    )
    result["attribution_support_method"] = attribution_support_method.value
    result["prior_quality_source"] = None
    result["scoring_function_status"] = RunStatus.UNAVAILABLE.value
    result["scoring_function_reason_code"] = reason_code
    result["score_version"] = None
    result["model_manifest_id"] = None
    result["scoring_function_id"] = None
    result["sample_score_status"] = "not_emitted"
    result["sample_score_reason_code"] = "integrated_strength_unavailable"
    return result.loc[:, list(EDGE_EVIDENCE_COLUMNS)]


def _build_edge_evidence(
    scoring_availability: pd.DataFrame,
    attribution_runs: tuple[BaselineAttributionRun, ...],
    candidate_score_runs: tuple[BaselineScoreRun, ...],
    emitted_score_runs: tuple[BaselineScoreRun, ...],
    *,
    config_modes: Sequence[CommunicationMode | str],
    attribution_support_method: AttributionSupportMethod,
) -> pd.DataFrame:
    if scoring_availability.empty:
        return _empty_edge_evidence()
    modes = _edge_modes(config_modes)
    score_lookup = {
        (_string_identifier(run.receiver), run.contrast): run
        for run in candidate_score_runs
    }
    if len(score_lookup) != len(candidate_score_runs):
        raise ValueError("candidate score-run receiver/contrast keys must be unique")
    emitted_keys = {
        (_string_identifier(run.receiver), run.contrast) for run in emitted_score_runs
    }
    frames: list[pd.DataFrame] = []
    covered_receivers: set[str] = set()
    source_columns = [
        "sample_id",
        "subject_id",
        "context",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
        "state_availability",
        "state_availability_status",
        "state_availability_reason_code",
        "ecosystem_availability",
        "ecosystem_availability_status",
        "ecosystem_availability_reason_code",
        "sender_weight",
        "sender_status",
        "sender_reason_code",
    ]
    for run in attribution_runs:
        receiver_id = _string_identifier(run.receiver)
        selected = scoring_availability.loc[
            _value_mask(scoring_availability["receiver"], run.receiver)
            & _context_mask(scoring_availability["context"], run.contrast_contexts),
            source_columns,
        ].copy()
        if selected.empty:
            continue
        covered_receivers.add(receiver_id)
        key = (receiver_id, run.contrast)
        score_run = score_lookup.get(key)
        if score_run is None:
            frames.append(
                _unscored_edge_evidence(
                    selected,
                    run,
                    modes=modes,
                    attribution_support_method=attribution_support_method,
                )
            )
        else:
            frames.append(
                _scored_edge_evidence(
                    scoring_availability,
                    score_run,
                    run,
                    emitted=key in emitted_keys,
                )
            )
    uncovered = scoring_availability.loc[
        ~scoring_availability["receiver"]
        .map(_string_identifier)
        .isin(covered_receivers),
        source_columns,
    ].copy()
    if not uncovered.empty:
        reason = (
            "no_estimable_response_contrast"
            if not attribution_runs
            else "receiver_attribution_branch_unavailable"
        )
        frames.append(
            _availability_only_edge_evidence(
                uncovered,
                modes=modes,
                attribution_support_method=attribution_support_method,
                reason_code=reason,
            )
        )
    if not frames:
        return _empty_edge_evidence()
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(
        [
            "contrast",
            "sample_id",
            "context_id",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
        ],
        key=lambda column: column.astype(str),
        kind="stable",
        ignore_index=True,
    )


def _effective_run_parameters(
    *,
    graph: ContextGraph,
    availability: BatchAvailability,
    availability_parameters: AvailabilityParameters,
    sender_assignment: SenderAssignment,
    min_cells: int,
    min_samples_per_context: int,
    min_subjects_per_context: int,
    min_pooled_availability: float,
    max_interactions: int | None,
    subject_fixed_effects: bool,
    lambda1: float,
    lambda2: float,
    cosine_threshold: float,
    prior_quality: float,
    component_weights: Mapping[str, float] | None,
    component_scales: Mapping[str, float] | None,
    communication_modes: tuple[str, ...],
    attribution_support_method: AttributionSupportMethod,
) -> dict[str, object]:
    weights = (
        dict.fromkeys(CORE_COMPONENTS, 1.0)
        if component_weights is None
        else {key: float(value) for key, value in component_weights.items()}
    )
    scales = (
        dict.fromkeys(CORE_COMPONENTS, 1.0)
        if component_scales is None
        else {key: float(value) for key, value in component_scales.items()}
    )
    parameters: dict[str, object] = {
        "method_version": _METHOD_VERSION,
        "communication_modes": list(communication_modes),
        "context_graph": {
            "kind": graph.kind,
            "nodes": [_string_identifier(node) for node in graph.nodes],
            "edges": [
                {
                    "left": _string_identifier(edge.left),
                    "right": _string_identifier(edge.right),
                    "weight": edge.weight,
                }
                for edge in graph.edges
            ],
        },
        "pseudobulk": {"min_cells": min_cells},
        "response": {
            "min_samples_per_context": min_samples_per_context,
            "min_subjects_per_context": min_subjects_per_context,
            "subject_fixed_effects": subject_fixed_effects,
        },
        "availability": {
            "hill_half_saturation": availability_parameters.hill.half_saturation,
            "hill_coefficient": availability_parameters.hill.coefficient,
            "detection_alpha": availability_parameters.detection.alpha,
            "detection_beta": availability_parameters.detection.beta,
            "detection_exponent": availability_parameters.detection.exponent,
            "complex_power": availability_parameters.complex_power,
            "complex_epsilon": availability_parameters.complex_epsilon,
            "min_pooled_availability": min_pooled_availability,
            "max_interactions": max_interactions,
            "filter_application": availability.filter_application.value,
            "application_subject_ids": list(availability.application_subject_ids),
            "frozen_interaction_universe": (
                availability.frozen_interaction_universe.to_dict()
            ),
        },
        "sender": sender_assignment.parameters.to_dict(),
        "attribution": {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "cosine_threshold": cosine_threshold,
            "precision_transform": {
                "method": _PRECISION_METHOD,
                "lower_quantile": _PRECISION_LOWER_QUANTILE,
                "upper_quantile": _PRECISION_UPPER_QUANTILE,
                "min_positive_features": _MIN_POSITIVE_PRECISION_FEATURES,
            },
        },
        "scoring": {
            "prior_quality": prior_quality,
            "component_weights": weights,
            "component_scales": scales,
            "cross_fitted": False,
        },
    }
    if (
        attribution_support_method
        is not AttributionSupportMethod.RELATIVE_COEFFICIENT_V1
    ):
        attribution = cast(dict[str, object], parameters["attribution"])
        attribution["downstream_support"] = attribution_support_method.to_dict()
    return parameters


def fit_baseline(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    *,
    context_graph: ContextGraph | None = None,
    target_prior: TargetPrior | None = None,
    availability_parameters: AvailabilityParameters = _DEFAULT_AVAILABILITY_PARAMETERS,
    sender_parameters: SenderEvidenceParameters | None = None,
    min_cells: int = 10,
    min_samples_per_context: int = 2,
    min_subjects_per_context: int = 2,
    min_pooled_availability: float = 0.01,
    max_interactions: int | None = None,
    subject_fixed_effects: bool = False,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    cosine_threshold: float = 0.95,
    prior_quality: float = 1.0,
    component_weights: Mapping[str, float] | None = None,
    component_scales: Mapping[str, float] | None = None,
    downstream_attribution_support_method: AttributionSupportMethod | str = (
        AttributionSupportMethod.RELATIVE_COEFFICIENT_V1
    ),
) -> BaselineArtifacts:
    """Fit the deterministic, explicitly non-cross-fitted v0.1 baseline."""

    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if target_prior is not None and not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be a TargetPrior or None")
    if not isinstance(availability_parameters, AvailabilityParameters):
        raise TypeError("availability_parameters must be AvailabilityParameters")
    attribution_support_method = AttributionSupportMethod(
        downstream_attribution_support_method
    )
    _validate_workflow_parameters(
        min_cells=min_cells,
        min_samples_per_context=min_samples_per_context,
        min_subjects_per_context=min_subjects_per_context,
        min_pooled_availability=min_pooled_availability,
        prior_quality=prior_quality,
    )
    schema = _input_schema(config)
    validated = validate_anndata(adata, schema)
    if validated.report.duplicate_genes:
        raise ValueError(
            "baseline fitting does not support duplicate gene identifiers; "
            "allow_duplicate_genes=True is validation-only"
        )
    graph = _resolve_context_graph(validated, context_graph)
    plan = _build_dry_run_plan(
        validated,
        config,
        graph,
        resource_bundle,
        target_prior,
        min_cells=min_cells,
        min_samples_per_context=min_samples_per_context,
        min_subjects_per_context=min_subjects_per_context,
    )
    if not plan.can_fit:
        blocking_reasons = plan.resource_table["reason_code"].dropna().astype(str)
        plan_reasons = [
            *blocking_reasons,
            *plan.design_audit.reason_codes,
        ]
        if not plan.contrast_table.empty and not plan.contrast_table["estimable"].all():
            plan_reasons.append("registered_contrast_not_estimable")
        raise ValueError(
            "baseline design/resources are incompatible: "
            + ",".join(sorted(set(plan_reasons)))
        )

    aggregate = aggregate_pseudobulk(validated, min_cells=min_cells)
    response = estimate_gene_response(
        aggregate,
        graph,
        contrasts=_workflow_contrasts(graph),
        min_samples_per_context=min_samples_per_context,
        min_subjects_per_context=min_subjects_per_context,
        subject_fixed_effects=subject_fixed_effects,
        design_audit=plan.design_audit,
    )
    availability = estimate_bundle_availability(
        aggregate,
        resource_bundle,
        context_keys=config.context_keys,
        parameters=availability_parameters,
        min_pooled_availability=min_pooled_availability,
        max_interactions=max_interactions,
    )
    sender_assignment = assign_senders(
        _sender_input(availability, config), sender_parameters
    )
    attribution_runs = _run_attribution(
        response,
        availability,
        resource_bundle,
        target_prior,
        lambda1=lambda1,
        lambda2=lambda2,
        cosine_threshold=cosine_threshold,
    )
    scoring_availability = _scoring_availability(
        availability,
        config,
        graph,
        sender_assignment,
    )
    candidate_score_runs = _score_runs(
        response,
        scoring_availability,
        target_prior,
        attribution_runs,
        availability_parameters=availability_parameters,
        prior_quality=prior_quality,
        component_weights=component_weights,
        component_scales=component_scales,
        config_modes=config.communication_modes,
        attribution_support_method=attribution_support_method,
        sender_functional_id=(
            sender_assignment.parameters.assignment_functional_id
        ),
        filter_universe_id=availability.filter_universe_id,
        cosine_threshold=cosine_threshold,
    )
    score_runs = tuple(
        run
        for run in candidate_score_runs
        if run.scores.table["comm_strength"].notna().any()
    )
    if score_runs:
        sample_scores = pd.concat(
            [run.scores.table for run in score_runs], ignore_index=True, sort=False
        ).sort_values(
            [
                "contrast",
                "sample_id",
                "sender",
                "receiver",
                "interaction_id",
                "mode",
            ],
            key=lambda column: column.astype(str),
            kind="stable",
            ignore_index=True,
        )
        mode = BaselineMode.EXPLORATORY_STRENGTH
    else:
        sample_scores = pd.DataFrame()
        mode = BaselineMode.AVAILABILITY_BASELINE
    edge_evidence = _build_edge_evidence(
        scoring_availability,
        attribution_runs,
        candidate_score_runs,
        score_runs,
        config_modes=config.communication_modes,
        attribution_support_method=attribution_support_method,
    )

    reason_codes = [
        *validated.report.reason_codes,
        *validated.report.warnings,
    ]
    if target_prior is None:
        reason_codes.append("target_prior_not_provided")
    if not score_runs:
        reason_codes.append("integrated_strength_unavailable")
    if len(score_runs) != len(candidate_score_runs):
        reason_codes.append("score_branch_no_complete_core_evidence")
    reason_codes.extend(
        run.reason_code for run in attribution_runs if run.reason_code is not None
    )
    return BaselineArtifacts(
        config=config,
        input_schema=schema,
        validated_input=validated,
        aggregate=aggregate,
        context_graph=graph,
        response=response,
        resource_bundle=resource_bundle,
        target_prior=target_prior,
        availability=availability,
        sender_assignment=sender_assignment,
        dry_run_plan=plan,
        attribution_runs=attribution_runs,
        score_runs=score_runs,
        sample_scores=sample_scores,
        edge_evidence=edge_evidence,
        mode=mode,
        reason_codes=tuple(reason_codes),
        run_parameters=_effective_run_parameters(
            graph=graph,
            availability=availability,
            availability_parameters=availability_parameters,
            sender_assignment=sender_assignment,
            min_cells=min_cells,
            min_samples_per_context=min_samples_per_context,
            min_subjects_per_context=min_subjects_per_context,
            min_pooled_availability=min_pooled_availability,
            max_interactions=max_interactions,
            subject_fixed_effects=subject_fixed_effects,
            lambda1=lambda1,
            lambda2=lambda2,
            cosine_threshold=cosine_threshold,
            prior_quality=prior_quality,
            component_weights=component_weights,
            component_scales=component_scales,
            communication_modes=tuple(
                CommunicationMode(mode).value for mode in config.communication_modes
            ),
            attribution_support_method=attribution_support_method,
        ),
        method_version=_METHOD_VERSION,
    )
