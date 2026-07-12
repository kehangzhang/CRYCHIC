"""Translate v0.1 workflow artifacts into the versioned result contract."""

from __future__ import annotations

import math
from collections.abc import Hashable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import (
    CommunicationMode,
    RunProvenance,
    SeedLineage,
    canonical_json,
    stable_id,
)
from crychic.design import (
    ContrastSpec,
    context_fields,
    node_context_fields,
    node_context_mapping,
)
from crychic.results import (
    RESULT_SCHEMA_VERSION,
    CrychicResult,
    empty_table,
    write_result,
)
from crychic.sender import SenderEvidenceParameters, assign_senders

from .contracts import BaselineArtifacts, RunStatus

_INFERENCE_REASON = "v0_1_inferential_disabled"
_AVAILABILITY_CONTRAST = "availability_only"


def _plain(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _string_identifier(value: object) -> str:
    plain = _plain(value)
    if isinstance(plain, str):
        return plain
    return canonical_json(
        {
            "type": f"{type(plain).__module__}.{type(plain).__qualname__}",
            "value": plain,
        }
    )


def _context_payload(
    value: Hashable, context_keys: tuple[str, ...]
) -> dict[str, object]:
    return dict(node_context_mapping(value, context_keys))


def _context_fields(
    value: Hashable, context_keys: tuple[str, ...]
) -> tuple[str, str]:
    return node_context_fields(value, context_keys)


def _row_context_fields(
    row: pd.Series, context_keys: tuple[str, ...]
) -> tuple[str, str]:
    return context_fields(row, context_keys)


def _contrast_fields(
    spec: ContrastSpec, context_keys: tuple[str, ...]
) -> tuple[str, str]:
    positive = [node for node, weight in spec.weights.items() if weight > 0]
    if len(positive) == 1:
        return _context_fields(positive[0], context_keys)
    payload = {
        "contrast": spec.name,
        "weights": [
            {
                "context": _context_payload(node, context_keys),
                "weight": weight,
            }
            for node, weight in spec.weights.items()
        ],
    }
    return stable_id("contrast_scope", payload), canonical_json(payload)


def _finite_or_none(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    numeric = float(cast(Any, value))
    return numeric if math.isfinite(numeric) else None


def _reason(*values: object) -> str | None:
    reasons: list[str] = []
    for value in values:
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        reasons.extend(part for part in str(value).split(";") if part)
    unique = tuple(dict.fromkeys(reasons))
    return ";".join(unique) if unique else None


def _typed_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    template = empty_table(name)
    if frame.empty:
        return template
    result = frame.loc[:, template.columns].copy()
    for column in template:
        dtype = template[column].dtype
        if pd.api.types.is_string_dtype(dtype):
            result[column] = result[column].astype("string")
        elif pd.api.types.is_bool_dtype(dtype):
            result[column] = result[column].astype(bool)
        elif pd.api.types.is_integer_dtype(dtype):
            result[column] = pd.to_numeric(result[column], errors="raise").astype(
                "int64"
            )
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(
                "float64"
            )
    return result


def _sender_assignment(
    artifacts: BaselineArtifacts,
    parameters: SenderEvidenceParameters | None,
) -> pd.DataFrame:
    source = artifacts.availability.sample_interactions.copy(deep=True)
    if source.empty:
        return artifacts.sender_assignment.table.copy(deep=True)
    context_keys = tuple(artifacts.config.context_keys)
    fields = [_row_context_fields(row, context_keys) for _, row in source.iterrows()]
    stable_context_ids = [context_id for context_id, _ in fields]
    source["context_id"] = stable_context_ids
    for column in (
        "sample_id",
        "subject_id",
        "sender",
        "receiver",
        "interaction_id",
    ):
        source[column] = source[column].map(_string_identifier)
    if parameters is not None:
        return assign_senders(source, parameters).table
    return artifacts.sender_assignment.table.copy(deep=True)


def _response_table(artifacts: BaselineArtifacts) -> pd.DataFrame:
    specs = {spec.name: spec for spec in artifacts.response.contrast_specs}
    context_keys = tuple(artifacts.config.context_keys)
    records: list[dict[str, object]] = []
    for row in artifacts.response.contrasts.itertuples(index=False):
        context_id, context_json = _contrast_fields(
            specs[str(row.contrast)], context_keys
        )
        status = "ok" if str(row.status) == "ok" else "not_estimable"
        records.append(
            {
                "context_id": context_id,
                "context_json": context_json,
                "receiver": str(row.receiver),
                "gene": str(row.gene),
                "contrast": str(row.contrast),
                "effect_size": _finite_or_none(row.effect),
                "standard_error": _finite_or_none(row.standard_error),
                "z_score": _finite_or_none(row.z_score),
                "precision": _finite_or_none(row.precision),
                "p_value": None,
                "n_subjects": int(cast(Any, row.n_subjects)),
                "status": status,
                "reason_code": _reason(row.reason_code, _INFERENCE_REASON),
            }
        )
    return _typed_table("responses", pd.DataFrame.from_records(records))


def _availability_score_rows(
    artifacts: BaselineArtifacts,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    source = artifacts.availability.sample_interactions.copy(deep=True)
    if source.empty:
        return _typed_table("sample_scores", pd.DataFrame())
    context_keys = tuple(artifacts.config.context_keys)
    fields = [_row_context_fields(row, context_keys) for _, row in source.iterrows()]
    source["context_id"] = [context_id for context_id, _ in fields]
    source["context_json"] = [context_json for _, context_json in fields]
    for column in (
        "sample_id",
        "subject_id",
        "sender",
        "receiver",
        "interaction_id",
    ):
        source[column] = source[column].map(_string_identifier)
    assignment_columns = [
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
        "assignment_weight",
        "reason_code",
    ]
    source = source.merge(
        assignment.loc[:, assignment_columns].rename(
            columns={"reason_code": "assignment_reason"}
        ),
        how="left",
        on=["context_id", "sender", "receiver", "interaction_id"],
        validate="many_to_one",
        sort=False,
    )
    functional_id = stable_id(
        "availability_scoring_function",
        {
            "config_digest": artifacts.config.digest,
            "resource": artifacts.resource_bundle.resource_id,
            "version": artifacts.resource_bundle.version,
        },
    )
    records: list[dict[str, object]] = []
    requested_modes = {
        CommunicationMode(mode).value for mode in artifacts.config.communication_modes
    }
    for row in source.itertuples(index=False):
        edge_id = stable_id(
            "communication_edge",
            {
                "interaction_id": str(row.interaction_id),
                "receiver": str(row.receiver),
                "sender": str(row.sender),
            },
        )
        design_row_id = stable_id(
            "design_row",
            {"context_id": row.context_id, "sample_id": str(row.sample_id)},
        )
        for mode, column in (
            ("state", "availability_state"),
            ("ecosystem", "availability_ecosystem"),
        ):
            if mode not in requested_modes:
                continue
            availability = _finite_or_none(getattr(row, column))
            reason = _reason(
                "integrated_strength_unavailable",
                getattr(row, "assignment_reason", None),
            )
            records.append(
                {
                    "subject_id": str(row.subject_id),
                    "sample_id": str(row.sample_id),
                    "context_id": str(row.context_id),
                    "context_json": str(row.context_json),
                    "design_row_id": design_row_id,
                    "edge_id": edge_id,
                    "scoring_functional_id": functional_id,
                    "repeat_id": "repeat-0",
                    "fold_id": "in_sample",
                    "mode": mode,
                    "availability": availability,
                    "downstream": None,
                    "sender_component": _finite_or_none(row.assignment_weight),
                    "prior_quality": None,
                    "abundance_component": None,
                    "comm_strength": None,
                    "eligible": False,
                    "status": "missing",
                    "reason_code": reason,
                }
            )
    return _typed_table("sample_scores", pd.DataFrame.from_records(records))


def _integrated_score_rows(artifacts: BaselineArtifacts) -> pd.DataFrame:
    context_keys = tuple(artifacts.config.context_keys)
    records: list[dict[str, object]] = []
    for row in artifacts.sample_scores.itertuples(index=False):
        context_id, context_json = _context_fields(row.context, context_keys)
        status = "ok" if str(row.status) == "ok" else "missing"
        edge_id = stable_id(
            "communication_edge",
            {
                "interaction_id": str(row.interaction_id),
                "receiver": str(row.receiver),
                "sender": str(row.sender),
            },
        )
        records.append(
            {
                "subject_id": str(row.subject_id),
                "sample_id": str(row.sample_id),
                "context_id": context_id,
                "context_json": context_json,
                "design_row_id": stable_id(
                    "design_row",
                    {"context_id": context_id, "sample_id": str(row.sample_id)},
                ),
                "edge_id": edge_id,
                "scoring_functional_id": str(row.scoring_function_id),
                "repeat_id": "repeat-0",
                "fold_id": str(row.fold_id),
                "mode": str(row.mode),
                "availability": _finite_or_none(row.availability),
                "downstream": _finite_or_none(row.downstream_activity),
                "sender_component": _finite_or_none(row.sender_component),
                "prior_quality": _finite_or_none(row.prior_quality),
                "abundance_component": _finite_or_none(row.abundance_component),
                "comm_strength": _finite_or_none(row.comm_strength),
                "eligible": status == "ok",
                "status": status,
                "reason_code": _reason(
                    row.reason_code,
                    row.functional_reason_code,
                    "abundance_component_not_scored"
                    if _finite_or_none(row.abundance_component) is None
                    else None,
                ),
            }
        )
    return _typed_table("sample_scores", pd.DataFrame.from_records(records))


def _interaction_table(
    artifacts: BaselineArtifacts,
    sample_scores: pd.DataFrame,
) -> pd.DataFrame:
    if sample_scores.empty:
        return _typed_table("interactions", pd.DataFrame())
    edge_lookup: dict[str, tuple[str, str, str]] = {}
    if not artifacts.sample_scores.empty:
        for row in artifacts.sample_scores.itertuples(index=False):
            edge_id = stable_id(
                "communication_edge",
                {
                    "interaction_id": str(row.interaction_id),
                    "receiver": str(row.receiver),
                    "sender": str(row.sender),
                },
            )
            edge_lookup[edge_id] = (
                str(row.sender),
                str(row.receiver),
                str(row.interaction_id),
            )
    else:
        for row in artifacts.availability.sample_interactions.itertuples(index=False):
            edge_id = stable_id(
                "communication_edge",
                {
                    "interaction_id": str(row.interaction_id),
                    "receiver": str(row.receiver),
                    "sender": str(row.sender),
                },
            )
            edge_lookup[edge_id] = (
                str(row.sender),
                str(row.receiver),
                str(row.interaction_id),
            )
    scored = sample_scores.copy()
    scored[["sender", "receiver", "interaction_id"]] = pd.DataFrame(
        [edge_lookup[edge_id] for edge_id in scored["edge_id"]],
        index=scored.index,
    )
    contrast_by_function = {
        run.functional.scoring_function_id: run.contrast
        for run in artifacts.score_runs
    }
    scored["contrast"] = scored["scoring_functional_id"].map(
        contrast_by_function
    ).fillna(_AVAILABILITY_CONTRAST)
    subject_keys = [
        "context_id",
        "context_json",
        "sender",
        "receiver",
        "interaction_id",
        "mode",
        "contrast",
        "subject_id",
    ]
    numeric = [
        "availability",
        "sender_component",
        "comm_strength",
        "prior_quality",
    ]
    per_subject = (
        scored.groupby(subject_keys, observed=True, sort=False, dropna=False)[numeric]
        .mean()
        .reset_index()
    )
    group_keys = subject_keys[:-1]
    summary = (
        per_subject.groupby(group_keys, observed=True, sort=False, dropna=False)
        .agg(
            availability=("availability", "mean"),
            assignment_weight=("sender_component", "mean"),
            comm_strength=("comm_strength", "mean"),
            prior_quality=("prior_quality", "mean"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    records: list[dict[str, object]] = []
    integrated = bool(artifacts.score_runs)
    for row in summary.itertuples(index=False):
        comm_strength = _finite_or_none(row.comm_strength) if integrated else None
        prior_quality = _finite_or_none(row.prior_quality) if integrated else None
        availability = _finite_or_none(row.availability)
        assignment_weight = _finite_or_none(row.assignment_weight)
        status = (
            "ok"
            if availability is not None
            and (not integrated or comm_strength is not None)
            else "missing"
        )
        records.append(
            {
                "context_id": str(row.context_id),
                "context_json": str(row.context_json),
                "sender": str(row.sender),
                "receiver": str(row.receiver),
                "interaction_id": str(row.interaction_id),
                "mode": str(row.mode),
                "contrast": str(row.contrast),
                "availability": availability,
                "assignment_weight": assignment_weight,
                "comm_strength": comm_strength,
                "comm_probability": None,
                "prior_quality": prior_quality,
                "n_subjects": int(cast(Any, row.n_subjects)),
                "status": status,
                "reason_code": _reason(
                    _INFERENCE_REASON,
                    None if integrated else "integrated_strength_unavailable",
                    None if status == "ok" else "missing_score_component",
                ),
            }
        )
    return _typed_table("interactions", pd.DataFrame.from_records(records))


def baseline_result_tables(
    artifacts: BaselineArtifacts,
    *,
    sender_parameters: SenderEvidenceParameters | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all required v0.1 tables without writing to disk."""

    if not isinstance(artifacts, BaselineArtifacts):
        raise TypeError("artifacts must be BaselineArtifacts")
    assignment = _sender_assignment(artifacts, sender_parameters)
    sample_scores = (
        _integrated_score_rows(artifacts)
        if artifacts.score_runs
        else _availability_score_rows(artifacts, assignment)
    )
    return {
        "interactions": _interaction_table(artifacts, sample_scores),
        "differential": empty_table("differential"),
        "responses": _response_table(artifacts),
        "sample_scores": sample_scores,
        "signatures": empty_table("signatures"),
    }


def _package_version() -> str:
    try:
        return version("CRYCHIC")
    except PackageNotFoundError:
        return "0.0.0"


def write_baseline_result(
    artifacts: BaselineArtifacts,
    destination: str | Path,
    *,
    input_digest: str | None = None,
    git_commit: str | None = None,
    git_dirty: bool = False,
    package_version: str | None = None,
    run_id: str | None = None,
    sender_parameters: SenderEvidenceParameters | None = None,
    persist_edge_evidence: bool = False,
) -> CrychicResult:
    """Persist a complete exploratory baseline with atomic publication.

    The potentially large edge-evidence ledger is opt-in. When omitted, the
    run manifest records that choice without materializing another ledger copy.
    """

    if not isinstance(persist_edge_evidence, bool):
        raise TypeError("persist_edge_evidence must be a bool")
    tables = baseline_result_tables(
        artifacts, sender_parameters=sender_parameters
    )
    bundle_key = (
        f"{artifacts.resource_bundle.resource_id}:"
        f"{artifacts.resource_bundle.version}"
    )
    resources = {bundle_key: artifacts.resource_bundle.manifest_digest}
    if artifacts.target_prior is not None:
        resources[
            f"{artifacts.target_prior.resource_id}:{artifacts.target_prior.version}"
        ] = artifacts.target_prior.manifest_digest
    provenance = RunProvenance(
        package_version=package_version or _package_version(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        config_digest=artifacts.config.digest,
        input_digest=input_digest,
        resource_digests=resources,
        result_schema_version=RESULT_SCHEMA_VERSION,
        seed_lineage=SeedLineage(artifacts.config.random_seed),
    )
    resolved_run_id = run_id or stable_id(
        "baseline_run",
        {
            "config_digest": artifacts.config.digest,
            "input_digest": input_digest,
            "method_version": artifacts.method_version,
            "resource_digests": resources,
            "workflow_digest": artifacts.run_parameters_digest,
        },
    )
    response_ok = (
        not artifacts.response.contrasts.empty
        and bool((artifacts.response.contrasts["status"] == "ok").any())
    )
    availability_ok = not artifacts.availability.sample_interactions.empty
    sender_ok = (
        not artifacts.sender_assignment.table.empty
        and bool((artifacts.sender_assignment.table["status"] == "ok").any())
    )
    attribution_ok = any(
        run.status is RunStatus.OK for run in artifacts.attribution_runs
    )
    attribution_failed = any(
        run.status is RunStatus.FAILED for run in artifacts.attribution_runs
    )
    attribution_stage: dict[str, object]
    if artifacts.target_prior is None:
        attribution_stage = {
            "name": "attribution",
            "status": "skipped",
            "reason_code": "target_prior_not_provided",
        }
    elif attribution_ok:
        attribution_stage = {
            "name": "attribution",
            "status": "complete",
            "reason_code": None,
        }
    elif attribution_failed:
        attribution_stage = {
            "name": "attribution",
            "status": "failed",
            "reason_code": "all_attribution_branches_failed",
        }
    else:
        attribution_stage = {
            "name": "attribution",
            "status": "not_estimable",
            "reason_code": "attribution_not_estimable",
        }
    warnings = list(artifacts.reason_codes)
    if not persist_edge_evidence:
        warnings.append("edge_evidence_not_persisted")
    if input_digest is None:
        warnings.append("input_digest_not_supplied")
    if not response_ok:
        warnings.append("no_estimable_response_contrast")
    if not availability_ok:
        warnings.append("no_supported_interactions")
    if not sender_ok:
        warnings.append("no_supported_sender_assignments")
    manifest = {
        "run_id": resolved_run_id,
        "mode": "exploratory",
        "workflow_parameters": artifacts.run_parameters,
        "stages": [
            {"name": "validate", "status": "complete", "reason_code": None},
            {"name": "pseudobulk", "status": "complete", "reason_code": None},
            {
                "name": "response",
                "status": "complete" if response_ok else "not_estimable",
                "reason_code": (
                    None if response_ok else "no_estimable_response_contrast"
                ),
            },
            {
                "name": "availability",
                "status": "complete" if availability_ok else "not_estimable",
                "reason_code": (
                    None if availability_ok else "no_supported_interactions"
                ),
            },
            {
                "name": "sender_assignment",
                "status": "complete" if sender_ok else "not_estimable",
                "reason_code": (
                    None if sender_ok else "no_supported_sender_assignments"
                ),
            },
            attribution_stage,
            {
                "name": "scoring",
                "status": "complete" if artifacts.score_runs else "skipped",
                "reason_code": (
                    None
                    if artifacts.score_runs
                    else "integrated_strength_unavailable"
                ),
            },
            {"name": "results", "status": "complete", "reason_code": None},
        ],
        "warnings": sorted(set(warnings)),
    }
    return write_result(
        destination,
        config=artifacts.config,
        provenance=provenance,
        run_manifest=manifest,
        tables=tables,
        edge_evidence=(artifacts.edge_evidence if persist_edge_evidence else None),
    )


__all__ = ["baseline_result_tables", "write_baseline_result"]
