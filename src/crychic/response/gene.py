"""V0.1 sample-level continuous gene response estimation."""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from crychic.design import (
    ContextGraph,
    ContextNode,
    ContrastSpec,
    DesignAudit,
    SubjectDesignAudit,
    audit_subject_design,
    global_contrasts,
    local_contrasts,
)
from crychic.pseudobulk import ExploratoryAggregate, PseudobulkDataset

from .contracts import ResponseEstimate, ResponseMethod, ResponseStatus

Aggregate: TypeAlias = PseudobulkDataset | ExploratoryAggregate

_CONTEXT_COLUMNS = [
    "receiver",
    "context",
    "gene",
    "mean_response",
    "n_samples",
    "n_subjects",
    "status",
    "reason_code",
]
_CONTRAST_COLUMNS = [
    "receiver",
    "gene",
    "contrast",
    "family",
    "mode",
    "effect",
    "standard_error",
    "z_score",
    "precision",
    "direction",
    "n_samples",
    "n_subjects",
    "paired",
    "method",
    "status",
    "reason_code",
]


@dataclass(frozen=True, slots=True)
class _SupportPlan:
    indices: np.ndarray
    reason_code: str | None
    paired: bool
    complete_case: bool
    n_samples: int
    n_subjects: int


def _stable_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)


def _matches_node(series: pd.Series, node: ContextNode) -> np.ndarray:
    return series.map(lambda value: value == node).to_numpy(dtype=bool)


def _canonical_context(context: object) -> tuple[tuple[str, Hashable], ...]:
    if not isinstance(context, tuple):
        raise ValueError(
            f"aggregate context must be a canonical tuple, got {context!r}"
        )
    pairs: list[tuple[str, Hashable]] = []
    for entry in context:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(f"invalid aggregate context entry {entry!r}")
        key, value = entry
        if not isinstance(key, str) or not isinstance(value, Hashable):
            raise ValueError(f"invalid aggregate context entry {entry!r}")
        pairs.append((key, value))
    return tuple(sorted(pairs, key=lambda pair: pair[0]))


def _context_node(context: object, graph: ContextGraph) -> ContextNode:
    canonical = _canonical_context(context)
    candidates: list[ContextNode] = [canonical]
    if len(canonical) == 1:
        candidates.append(canonical[0][1])
    matches = [candidate for candidate in candidates if candidate in graph.nodes]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"context {canonical!r} ambiguously matches multiple context graph nodes"
        )
    return canonical


def _selected_contrasts(
    graph: ContextGraph,
    contrasts: Sequence[ContrastSpec] | None,
    *,
    include_global: bool,
    include_local: bool,
) -> tuple[ContrastSpec, ...]:
    if contrasts is None:
        selected: list[ContrastSpec] = []
        if len(graph.nodes) > 1 and include_global:
            selected.extend(global_contrasts(graph))
        if include_local:
            selected.extend(local_contrasts(graph))
    else:
        selected = list(contrasts)
    if len({contrast.name for contrast in selected}) != len(selected):
        raise ValueError("contrast names must be unique")
    graph_nodes = set(graph.nodes)
    for contrast in selected:
        unknown = set(contrast.weights).difference(graph_nodes)
        if unknown:
            raise ValueError(
                f"contrast {contrast.name!r} references unknown graph nodes: "
                f"{sorted(map(repr, unknown))}"
            )
    return tuple(selected)


def _sample_response(
    aggregate: Aggregate,
    graph: ContextGraph,
    *,
    cpm_scale: float,
) -> tuple[np.ndarray, pd.DataFrame, str, str, tuple[str, ...]]:
    metadata = aggregate.unit_metadata.copy(deep=True).reset_index(drop=True)
    metadata["context_node"] = [
        _context_node(context, graph) for context in metadata["context"]
    ]
    unexpected = set(metadata["context_node"]).difference(graph.nodes)
    if unexpected:
        raise ValueError(
            "aggregate contains context(s) absent from ContextGraph: "
            f"{sorted(map(repr, unexpected))}"
        )

    values: np.ndarray = np.full(
        (len(metadata), len(aggregate.feature_ids)), np.nan, dtype=float
    )
    response_eligible: list[bool] = []
    response_reason: list[str | None] = []
    matrix = (
        aggregate.counts
        if isinstance(aggregate, PseudobulkDataset)
        else aggregate.mean_expression
    )
    for output_row, unit in metadata.iterrows():
        if not bool(unit["state_eligible"]) or pd.isna(unit["matrix_row"]):
            response_eligible.append(False)
            response_reason.append(str(unit["missingness_reason"]))
            continue
        source_row = (
            matrix.getrow(int(unit["matrix_row"])).toarray().ravel().astype(float)
        )
        if isinstance(aggregate, PseudobulkDataset):
            library_size = float(source_row.sum())
            if library_size <= 0:
                response_eligible.append(False)
                response_reason.append("zero_library_size")
                continue
            source_row = np.log1p(source_row / library_size * cpm_scale)
        values[output_row, :] = source_row
        response_eligible.append(True)
        response_reason.append(None)
    metadata["response_eligible"] = response_eligible
    metadata["response_reason"] = response_reason

    if isinstance(aggregate, PseudobulkDataset):
        return values, metadata, "log1p_cpm", "raw_counts", ()
    return (
        values,
        metadata,
        aggregate.expression_transform,
        aggregate.expression_source,
        aggregate.reason_codes,
    )


def _context_mean_table(
    values: np.ndarray,
    metadata: pd.DataFrame,
    feature_ids: tuple[str, ...],
    graph: ContextGraph,
    receivers: tuple[Hashable, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = metadata["response_eligible"].to_numpy(dtype=bool)
    for receiver in receivers:
        receiver_mask = (metadata["cell_type"] == receiver).to_numpy()
        for context in graph.nodes:
            context_mask = _matches_node(metadata["context_node"], context)
            indices = np.flatnonzero(eligible & receiver_mask & context_mask)
            n_samples = len(indices)
            n_subjects = int(metadata.iloc[indices]["subject_id"].nunique())
            if n_samples:
                subject_means = [
                    values[group.index.to_numpy(dtype=int), :].mean(axis=0)
                    for _, group in metadata.loc[indices].groupby(
                        "subject_id", observed=True, sort=False, dropna=False
                    )
                ]
                means = np.vstack(subject_means).mean(axis=0)
                status = ResponseStatus.OK.value
                reason: str | None = None
            else:
                means = np.full(len(feature_ids), np.nan)
                status = ResponseStatus.NOT_ESTIMABLE.value
                reason = "no_eligible_samples"
            for gene, mean in zip(feature_ids, means, strict=True):
                rows.append(
                    {
                        "receiver": receiver,
                        "context": context,
                        "gene": gene,
                        "mean_response": float(mean),
                        "n_samples": n_samples,
                        "n_subjects": n_subjects,
                        "status": status,
                        "reason_code": reason,
                    }
                )
    return pd.DataFrame(rows, columns=_CONTEXT_COLUMNS)


def _audit_rows(design_audit: DesignAudit, sample_ids: Sequence[object]) -> np.ndarray:
    row_by_sample = {
        sample: index for index, sample in enumerate(design_audit.sample_ids)
    }
    missing = [sample for sample in sample_ids if sample not in row_by_sample]
    if missing:
        raise ValueError(
            "response samples are absent from the audited design matrix: "
            f"{sorted(map(repr, set(missing)))[:5]}"
        )
    positions = np.asarray([row_by_sample[sample] for sample in sample_ids], dtype=int)
    return design_audit.design_matrix.to_numpy(dtype=float)[positions, :]


def _vector_estimable(
    matrix: np.ndarray, vector: np.ndarray, *, tolerance: float = 1e-8
) -> bool:
    if np.linalg.norm(vector) <= tolerance or matrix.size == 0:
        return False
    projection = np.linalg.pinv(matrix) @ matrix
    residual = vector - vector @ projection
    return bool(
        np.linalg.norm(residual) <= tolerance * max(1.0, float(np.linalg.norm(vector)))
    )


def _collapse_subject_context(
    values: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    design_audit: DesignAudit,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    selected = metadata.loc[indices, ["sample_id", "subject_id", "context_node"]].copy()
    selected["source_index"] = indices
    design = _audit_rows(design_audit, selected["sample_id"].tolist())
    collapsed_values: list[np.ndarray] = []
    collapsed_design: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    grouping = selected.groupby(
        ["subject_id", "context_node"], observed=True, sort=False, dropna=False
    )
    for (subject, context), group in grouping:
        positions = group.index.to_numpy(dtype=int)
        local_positions = selected.index.get_indexer(pd.Index(positions))
        source_indices = group["source_index"].to_numpy(dtype=int)
        collapsed_values.append(values[source_indices, :].mean(axis=0))
        collapsed_design.append(design[local_positions, :].mean(axis=0))
        rows.append(
            {
                "subject_id": subject,
                "context_node": context,
                "n_samples": len(group),
            }
        )
    return (
        np.vstack(collapsed_values),
        np.vstack(collapsed_design),
        pd.DataFrame(rows),
    )


def _design_context_mean_table(
    values: np.ndarray,
    metadata: pd.DataFrame,
    feature_ids: tuple[str, ...],
    graph: ContextGraph,
    receivers: tuple[Hashable, ...],
    design_audit: DesignAudit,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = metadata["response_eligible"].to_numpy(dtype=bool)
    emm_by_context = {
        node: design_audit.emm_matrix.iloc[index].to_numpy(dtype=float)
        for index, node in enumerate(design_audit.context_nodes)
    }
    for receiver in receivers:
        receiver_mask = (metadata["cell_type"] == receiver).to_numpy(dtype=bool)
        receiver_indices = np.flatnonzero(eligible & receiver_mask)
        if len(receiver_indices):
            receiver_values, receiver_design, _ = _collapse_subject_context(
                values, metadata, receiver_indices, design_audit
            )
            coefficients = np.linalg.pinv(receiver_design) @ receiver_values
        else:
            receiver_values = np.empty((0, len(feature_ids)))
            receiver_design = np.empty((0, design_audit.n_columns))
            coefficients = np.empty((design_audit.n_columns, len(feature_ids)))
        for context in graph.nodes:
            context_mask = _matches_node(metadata["context_node"], context)
            indices = np.flatnonzero(eligible & receiver_mask & context_mask)
            n_samples = len(indices)
            n_subjects = int(metadata.iloc[indices]["subject_id"].nunique())
            emm = emm_by_context.get(context)
            if (
                emm is not None
                and len(receiver_values)
                and _vector_estimable(receiver_design, emm)
            ):
                means = emm @ coefficients
                status = ResponseStatus.OK.value
                reason: str | None = None
            else:
                means = np.full(len(feature_ids), np.nan)
                status = ResponseStatus.NOT_ESTIMABLE.value
                reason = (
                    "no_eligible_samples"
                    if not len(receiver_values)
                    else "receiver_emm_not_estimable"
                )
            for gene, mean in zip(feature_ids, means, strict=True):
                rows.append(
                    {
                        "receiver": receiver,
                        "context": context,
                        "gene": gene,
                        "mean_response": float(mean),
                        "n_samples": n_samples,
                        "n_subjects": n_subjects,
                        "status": status,
                        "reason_code": reason,
                    }
                )
    return pd.DataFrame(rows, columns=_CONTEXT_COLUMNS)


def _support_by_context(
    metadata: pd.DataFrame,
    receiver: Hashable,
    contrast: ContrastSpec,
) -> tuple[dict[ContextNode, np.ndarray], pd.DataFrame]:
    receiver_mask = metadata["cell_type"] == receiver
    eligible_mask = metadata["response_eligible"]
    support = metadata.loc[receiver_mask & eligible_mask]
    by_context: dict[ContextNode, np.ndarray] = {}
    for node in contrast.weights:
        by_context[node] = support.index[
            _matches_node(support["context_node"], node)
        ].to_numpy()
    referenced = support[support["context_node"].isin(contrast.weights)]
    return by_context, referenced


def _study_subject_design(
    metadata: pd.DataFrame, contrast: ContrastSpec
) -> SubjectDesignAudit:
    samples = metadata.loc[
        :, ["sample_id", "subject_id", "context_node"]
    ].drop_duplicates()
    if samples["sample_id"].duplicated().any():
        raise ValueError("a sample maps to multiple subjects or contexts")
    return audit_subject_design(
        samples,
        tuple(contrast.weights),
        context_key="context_node",
        subject_key="subject_id",
    )


def _response_support(
    metadata: pd.DataFrame,
    receiver: Hashable,
    contrast: ContrastSpec,
    *,
    min_samples_per_context: int,
    min_subjects_per_context: int,
    subject_fixed_effects: bool,
) -> _SupportPlan:
    by_context, referenced = _support_by_context(metadata, receiver, contrast)
    referenced_indices = referenced.index.to_numpy(dtype=int)
    n_samples = len(referenced)
    n_subjects = int(referenced["subject_id"].nunique())
    if not contrast.estimable:
        return _SupportPlan(
            referenced_indices,
            contrast.reason_code or "contrast_declared_not_estimable",
            False,
            False,
            n_samples,
            n_subjects,
        )

    study_design = _study_subject_design(metadata, contrast)
    if not study_design.ready:
        return _SupportPlan(
            referenced_indices,
            study_design.reason_code,
            False,
            False,
            n_samples,
            n_subjects,
        )

    for indices in by_context.values():
        if len(indices) == 0:
            return _SupportPlan(
                referenced_indices,
                "missing_context_support",
                False,
                False,
                n_samples,
                n_subjects,
            )
        if len(indices) < min_samples_per_context:
            return _SupportPlan(
                referenced_indices,
                "insufficient_sample_support",
                False,
                False,
                n_samples,
                n_subjects,
            )
        if metadata.loc[indices, "subject_id"].nunique() < min_subjects_per_context:
            return _SupportPlan(
                referenced_indices,
                "insufficient_subject_support",
                False,
                False,
                n_samples,
                n_subjects,
            )

    nodes = tuple(contrast.weights)
    if study_design.paired:
        receiver_subject_sets = [
            set(metadata.loc[by_context[node], "subject_id"]) for node in nodes
        ]
        complete_subjects = set.intersection(*receiver_subject_sets)
        if len(complete_subjects) < min_subjects_per_context:
            return _SupportPlan(
                referenced_indices,
                "insufficient_complete_pair_support",
                False,
                False,
                n_samples,
                len(complete_subjects),
            )
        selected = referenced.loc[referenced["subject_id"].isin(complete_subjects)]
        full_subjects = set(study_design.subjects_for(nodes[0]))
        return _SupportPlan(
            selected.index.to_numpy(dtype=int),
            None,
            True,
            complete_subjects != full_subjects,
            len(selected),
            len(complete_subjects),
        )

    subject_context_counts = referenced.groupby("subject_id", observed=True)[
        "context_node"
    ].nunique()
    if bool((subject_context_counts > 1).any()):  # pragma: no cover - audit invariant
        return _SupportPlan(
            referenced_indices,
            "repeated_measures_not_supported",
            False,
            False,
            n_samples,
            n_subjects,
        )
    if subject_fixed_effects:
        return _SupportPlan(
            referenced_indices,
            "rank_deficient_subject_fixed_effects",
            False,
            False,
            n_samples,
            n_subjects,
        )
    return _SupportPlan(
        referenced_indices,
        None,
        False,
        False,
        n_samples,
        n_subjects,
    )


def _support_failure(
    metadata: pd.DataFrame,
    receiver: Hashable,
    contrast: ContrastSpec,
    *,
    min_samples_per_context: int,
    min_subjects_per_context: int,
    subject_fixed_effects: bool,
) -> tuple[str | None, bool, ResponseMethod, int, int]:
    support = _response_support(
        metadata,
        receiver,
        contrast,
        min_samples_per_context=min_samples_per_context,
        min_subjects_per_context=min_subjects_per_context,
        subject_fixed_effects=subject_fixed_effects,
    )
    if support.reason_code is not None:
        method = ResponseMethod.NOT_ESTIMABLE
    elif support.paired:
        method = (
            ResponseMethod.PAIRED_COMPLETE_CASE
            if support.complete_case
            else ResponseMethod.PAIRED
        )
    else:
        method = ResponseMethod.INDEPENDENT
    return (
        support.reason_code,
        support.paired,
        method,
        support.n_samples,
        support.n_subjects,
    )


def _paired_estimate(
    values: np.ndarray,
    metadata: pd.DataFrame,
    receiver: Hashable,
    contrast: ContrastSpec,
) -> tuple[np.ndarray, np.ndarray]:
    by_context, _ = _support_by_context(metadata, receiver, contrast)
    nodes = tuple(contrast.weights)
    subjects_by_context = [
        set(metadata.loc[by_context[node], "subject_id"]) for node in nodes
    ]
    subjects = sorted(set.intersection(*subjects_by_context), key=_stable_value)
    differences: list[np.ndarray] = []
    for subject in subjects:
        subject_value = np.zeros(values.shape[1], dtype=float)
        for node in nodes:
            indices = by_context[node]
            subject_indices = [
                index
                for index in indices
                if metadata.at[index, "subject_id"] == subject
            ]
            subject_value += contrast.weights[node] * values[
                np.asarray(subject_indices, dtype=int), :
            ].mean(axis=0)
        differences.append(subject_value)
    difference_matrix = np.vstack(differences)
    effect = difference_matrix.mean(axis=0)
    standard_error = difference_matrix.std(axis=0, ddof=1) / math.sqrt(len(subjects))
    return effect, standard_error


def _independent_estimate(
    values: np.ndarray,
    metadata: pd.DataFrame,
    receiver: Hashable,
    contrast: ContrastSpec,
) -> tuple[np.ndarray, np.ndarray]:
    by_context, _ = _support_by_context(metadata, receiver, contrast)
    effect = np.zeros(values.shape[1], dtype=float)
    variance = np.zeros(values.shape[1], dtype=float)
    for node, weight in contrast.weights.items():
        context_metadata = metadata.loc[by_context[node]]
        context_values = np.vstack(
            [
                values[group.index.to_numpy(dtype=int), :].mean(axis=0)
                for _, group in context_metadata.groupby(
                    "subject_id", observed=True, sort=False, dropna=False
                )
            ]
        )
        effect += weight * context_values.mean(axis=0)
        variance += weight**2 * context_values.var(axis=0, ddof=1) / len(context_values)
    return effect, np.sqrt(variance)


def _formula_estimate(
    values: np.ndarray,
    metadata: pd.DataFrame,
    receiver: Hashable,
    contrast: ContrastSpec,
    design_audit: DesignAudit,
    *,
    min_samples_per_context: int,
    min_subjects_per_context: int,
    subject_fixed_effects: bool,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    str | None,
    bool,
    ResponseMethod,
    int,
    int,
]:
    _, referenced = _support_by_context(metadata, receiver, contrast)
    if not design_audit.ready:
        return (
            None,
            None,
            (
                design_audit.reason_codes[0]
                if design_audit.reason_codes
                else "design_blocked"
            ),
            False,
            ResponseMethod.NOT_ESTIMABLE,
            len(referenced),
            int(referenced["subject_id"].nunique()),
        )
    support = _response_support(
        metadata,
        receiver,
        contrast,
        min_samples_per_context=min_samples_per_context,
        min_subjects_per_context=min_subjects_per_context,
        subject_fixed_effects=subject_fixed_effects,
    )
    if support.reason_code is not None:
        return (
            None,
            None,
            support.reason_code,
            False,
            ResponseMethod.NOT_ESTIMABLE,
            support.n_samples,
            support.n_subjects,
        )
    paired = support.paired
    complete_case = support.complete_case
    n_samples = support.n_samples
    n_subjects = support.n_subjects

    model_values, model_design, model_metadata = _collapse_subject_context(
        values,
        metadata,
        support.indices,
        design_audit,
    )
    coefficient_contrast = design_audit.coefficient_contrast(contrast)
    if paired:
        subjects = tuple(
            sorted(model_metadata["subject_id"].unique(), key=_stable_value)
        )
        subject_codes = pd.Categorical(
            model_metadata["subject_id"], categories=list(subjects)
        ).codes
        subject_columns = np.zeros((len(model_metadata), max(0, len(subjects) - 1)))
        for column in range(1, len(subjects)):
            subject_columns[:, column - 1] = subject_codes == column
        model_design = np.column_stack((model_design, subject_columns))
        coefficient_contrast = np.concatenate(
            (coefficient_contrast, np.zeros(subject_columns.shape[1]))
        )

    if not _vector_estimable(model_design, coefficient_contrast):
        return (
            None,
            None,
            "receiver_formula_contrast_not_estimable",
            False,
            ResponseMethod.NOT_ESTIMABLE,
            n_samples,
            n_subjects,
        )
    rank = int(np.linalg.matrix_rank(model_design))
    residual_df = len(model_design) - rank
    if residual_df < 1:
        return (
            None,
            None,
            "insufficient_residual_degrees_of_freedom",
            False,
            ResponseMethod.NOT_ESTIMABLE,
            n_samples,
            n_subjects,
        )
    coefficients = np.linalg.pinv(model_design) @ model_values
    effect = coefficient_contrast @ coefficients
    residual = model_values - model_design @ coefficients
    residual_variance = np.sum(residual**2, axis=0) / residual_df
    contrast_variance = float(
        coefficient_contrast
        @ np.linalg.pinv(model_design.T @ model_design)
        @ coefficient_contrast
    )
    standard_error = np.sqrt(np.maximum(0.0, residual_variance * contrast_variance))
    if paired:
        method = (
            ResponseMethod.EMM_PAIRED_COMPLETE_CASE
            if complete_case
            else ResponseMethod.EMM_PAIRED
        )
    else:
        method = ResponseMethod.EMM_INDEPENDENT
    return effect, standard_error, None, paired, method, n_samples, n_subjects


def _failure_rows(
    receiver: Hashable,
    feature_ids: tuple[str, ...],
    contrast: ContrastSpec,
    reason: str,
    *,
    n_samples: int,
    n_subjects: int,
) -> list[dict[str, Any]]:
    return [
        {
            "receiver": receiver,
            "gene": gene,
            "contrast": contrast.name,
            "family": contrast.family,
            "mode": contrast.mode,
            "effect": np.nan,
            "standard_error": np.nan,
            "z_score": np.nan,
            "precision": np.nan,
            "direction": "not_estimable",
            "n_samples": n_samples,
            "n_subjects": n_subjects,
            "paired": False,
            "method": ResponseMethod.NOT_ESTIMABLE.value,
            "status": ResponseStatus.NOT_ESTIMABLE.value,
            "reason_code": reason,
        }
        for gene in feature_ids
    ]


def _contrast_table(
    values: np.ndarray,
    metadata: pd.DataFrame,
    feature_ids: tuple[str, ...],
    receivers: tuple[Hashable, ...],
    contrast_specs: tuple[ContrastSpec, ...],
    *,
    min_samples_per_context: int,
    min_subjects_per_context: int,
    subject_fixed_effects: bool,
    design_audit: DesignAudit | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for receiver in receivers:
        for contrast in contrast_specs:
            if design_audit is not None:
                (
                    effect,
                    standard_error,
                    reason,
                    paired,
                    method,
                    n_samples,
                    n_subjects,
                ) = _formula_estimate(
                    values,
                    metadata,
                    receiver,
                    contrast,
                    design_audit,
                    min_samples_per_context=min_samples_per_context,
                    min_subjects_per_context=min_subjects_per_context,
                    subject_fixed_effects=subject_fixed_effects,
                )
                if reason is not None or effect is None or standard_error is None:
                    rows.extend(
                        _failure_rows(
                            receiver,
                            feature_ids,
                            contrast,
                            reason or "formula_estimation_failed",
                            n_samples=n_samples,
                            n_subjects=n_subjects,
                        )
                    )
                    continue
            else:
                effect = None
                standard_error = None
                reason = None
            reason, paired, method, n_samples, n_subjects = (
                _support_failure(
                    metadata,
                    receiver,
                    contrast,
                    min_samples_per_context=min_samples_per_context,
                    min_subjects_per_context=min_subjects_per_context,
                    subject_fixed_effects=subject_fixed_effects,
                )
                if design_audit is None
                else (
                    None,
                    paired,
                    method,
                    n_samples,
                    n_subjects,
                )
            )
            if reason is not None:
                rows.extend(
                    _failure_rows(
                        receiver,
                        feature_ids,
                        contrast,
                        reason,
                        n_samples=n_samples,
                        n_subjects=n_subjects,
                    )
                )
                continue
            if design_audit is None:
                if paired:
                    effect, standard_error = _paired_estimate(
                        values, metadata, receiver, contrast
                    )
                else:
                    effect, standard_error = _independent_estimate(
                        values, metadata, receiver, contrast
                    )
            if effect is None or standard_error is None:  # pragma: no cover
                raise RuntimeError("response estimator returned no result")
            for gene_index, gene in enumerate(feature_ids):
                gene_effect = float(effect[gene_index])
                gene_se = float(standard_error[gene_index])
                if gene_se > 0 and math.isfinite(gene_se):
                    z_score = gene_effect / gene_se
                    precision = 1.0 / gene_se**2
                    gene_reason: str | None = None
                else:
                    z_score = np.nan
                    precision = np.nan
                    gene_reason = "zero_standard_error"
                direction = (
                    "positive"
                    if gene_effect > 0
                    else "negative"
                    if gene_effect < 0
                    else "zero"
                )
                rows.append(
                    {
                        "receiver": receiver,
                        "gene": gene,
                        "contrast": contrast.name,
                        "family": contrast.family,
                        "mode": contrast.mode,
                        "effect": gene_effect,
                        "standard_error": gene_se,
                        "z_score": z_score,
                        "precision": precision,
                        "direction": direction,
                        "n_samples": n_samples,
                        "n_subjects": n_subjects,
                        "paired": paired,
                        "method": method.value,
                        "status": ResponseStatus.OK.value,
                        "reason_code": gene_reason,
                    }
                )
    return pd.DataFrame(rows, columns=_CONTRAST_COLUMNS)


def estimate_gene_response(
    aggregate: Aggregate,
    context_graph: ContextGraph,
    *,
    receivers: Sequence[Hashable] | None = None,
    contrasts: Sequence[ContrastSpec] | None = None,
    include_global: bool = True,
    include_local: bool = True,
    min_samples_per_context: int = 2,
    min_subjects_per_context: int = 2,
    subject_fixed_effects: bool = False,
    design_audit: DesignAudit | None = None,
    cpm_scale: float = 1_000_000.0,
) -> ResponseEstimate:
    """Estimate continuous gene responses using sample-level diagnostics only.

    Count aggregates are converted to natural-log ``log1p(CPM)``. Explicitly
    normalized aggregates remain on their declared scale. Fully paired
    two-context contrasts use within-subject differences. A partially paired
    receiver uses only complete subject pairs when they pass the declared
    support threshold and records that complete-case method explicitly.
    Unsupported repeated designs are returned as not estimable instead of
    silently changing models.
    """

    if not isinstance(aggregate, (PseudobulkDataset, ExploratoryAggregate)):
        raise TypeError("aggregate must be PseudobulkDataset or ExploratoryAggregate")
    if not isinstance(context_graph, ContextGraph):
        raise TypeError("context_graph must be a ContextGraph")
    if design_audit is not None and not isinstance(design_audit, DesignAudit):
        raise TypeError("design_audit must be a DesignAudit or None")
    for name, value in (
        ("min_samples_per_context", min_samples_per_context),
        ("min_subjects_per_context", min_subjects_per_context),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    if not math.isfinite(cpm_scale) or cpm_scale <= 0:
        raise ValueError("cpm_scale must be finite and positive")

    selected_contrasts = _selected_contrasts(
        context_graph,
        contrasts,
        include_global=include_global,
        include_local=include_local,
    )
    sample_values, sample_metadata, value_scale, expression_source, reasons = (
        _sample_response(aggregate, context_graph, cpm_scale=cpm_scale)
    )
    observed_receivers = set(sample_metadata["cell_type"])
    if receivers is None:
        selected_receivers = tuple(sorted(observed_receivers, key=_stable_value))
    else:
        if len(set(receivers)) != len(receivers):
            raise ValueError("receivers must not contain duplicates")
        unknown = set(receivers).difference(observed_receivers)
        if unknown:
            raise ValueError(f"unknown receiver(s): {sorted(map(repr, unknown))}")
        selected_receivers = tuple(sorted(receivers, key=_stable_value))

    context_means = (
        _design_context_mean_table(
            sample_values,
            sample_metadata,
            aggregate.feature_ids,
            context_graph,
            selected_receivers,
            design_audit,
        )
        if design_audit is not None
        else _context_mean_table(
            sample_values,
            sample_metadata,
            aggregate.feature_ids,
            context_graph,
            selected_receivers,
        )
    )
    contrast_table = _contrast_table(
        sample_values,
        sample_metadata,
        aggregate.feature_ids,
        selected_receivers,
        selected_contrasts,
        min_samples_per_context=min_samples_per_context,
        min_subjects_per_context=min_subjects_per_context,
        subject_fixed_effects=subject_fixed_effects,
        design_audit=design_audit,
    )
    return ResponseEstimate(
        sample_values=sample_values,
        sample_metadata=sample_metadata,
        context_means=context_means,
        contrasts=contrast_table,
        feature_ids=aggregate.feature_ids,
        contrast_specs=selected_contrasts,
        input_mode=aggregate.mode,
        value_scale=value_scale,
        expression_source=expression_source,
        reason_codes=reasons,
    )
