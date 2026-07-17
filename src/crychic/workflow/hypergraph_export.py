"""Auditable post-fit export of cross-fit sender scores to a hypergraph."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import node_context_fields
from crychic.network import (
    CommunicationEdgeRecord,
    CommunicationHypergraph,
    HyperedgeStatus,
    TargetKind,
    build_communication_hypergraph,
)
from crychic.resources import Interaction, ResourceBundle

from .crossfit import CrossFitArtifacts

CROSSFIT_HYPERGRAPH_EXPORT_VERSION = "crossfit_sender_hypergraph_export_v1"
CROSSFIT_HYPERGRAPH_WEIGHT_SEMANTICS = (
    "descriptive_subject_equal_sender_resolved_strength_v1"
)

_EXPECTED_MODES = ("ecosystem", "state")
_SOURCE_SCORE_COLUMN = "sender_resolved_strength"
_SOURCE_KEY = (
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "sender",
)
_AGGREGATE_KEY = (
    "contrast_manifest_id",
    "contrast_name",
    "context_id",
    "receiver",
    "family_id",
    "interaction_id",
    "mode",
    "sender",
)
_STATUS_PRECEDENCE = {
    HyperedgeStatus.NOT_ESTIMABLE: 1,
    HyperedgeStatus.FILTERED: 2,
    HyperedgeStatus.FAILED: 3,
}


def _error(
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


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _error(
            f"{field} must be a canonical non-empty string",
            code="invalid_crossfit_hypergraph_source",
            field=field,
            remediation="Regenerate the producer-owned cross-fit artifacts",
        )
    return value


def _is_missing(value: object) -> bool:
    return value is None or value is pd.NA or value is pd.NaT or (
        isinstance(value, float) and math.isnan(value)
    )


def _optional_string(value: object, *, field: str) -> str | None:
    if _is_missing(value):
        return None
    return _required_string(value, field=field)


def _optional_finite_float(value: object, *, field: str) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise _error(
            f"{field} must be finite numeric data or missing",
            code="invalid_crossfit_hypergraph_source",
            field=field,
            remediation="Regenerate sender-resolved scores from intact inputs",
        )
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise _error(
            f"{field} must be finite numeric data or missing",
            code="invalid_crossfit_hypergraph_source",
            field=field,
            remediation="Regenerate sender-resolved scores from intact inputs",
        ) from error
    if not math.isfinite(numeric):
        raise _error(
            f"{field} must be finite numeric data or missing",
            code="invalid_crossfit_hypergraph_source",
            field=field,
            remediation="Regenerate sender-resolved scores from intact inputs",
        )
    return numeric


def _validate_resource_entity(
    interaction: Interaction,
    *,
    entity: str,
) -> tuple[str, tuple[str, ...]]:
    if entity == "ligand":
        name = interaction.ligand_name
        members = interaction.ligand_subunits
        is_complex = interaction.ligand_is_complex
    elif entity == "receptor":
        name = interaction.receptor_name
        members = interaction.receptor_subunits
        is_complex = interaction.receptor_is_complex
    else:  # pragma: no cover - private exhaustive call site
        raise AssertionError("unexpected interaction entity")
    canonical_name = _required_string(name, field=f"{entity}_name")
    canonical_members = tuple(
        _required_string(member, field=f"{entity}_subunits") for member in members
    )
    if tuple(sorted(canonical_members)) != canonical_members or len(
        set(canonical_members)
    ) != len(canonical_members):
        raise _error(
            f"Resource {entity} members are not canonical and unique",
            code="invalid_crossfit_hypergraph_resource",
            field=f"{entity}_subunits",
            remediation="Reload the interaction resource through a reviewed adapter",
        )
    represented_as_complex = not (
        len(canonical_members) == 1 and canonical_members[0] == canonical_name
    )
    if represented_as_complex != is_complex:
        raise _error(
            f"Resource {entity} complex flag disagrees with its expanded members",
            code="invalid_crossfit_hypergraph_resource",
            field=f"{entity}_is_complex",
            remediation=(
                "Preserve explicit complex names, flags, and expanded members in the "
                "ResourceBundle"
            ),
        )
    return canonical_name, canonical_members


def _validated_interactions(
    resource_bundle: ResourceBundle,
) -> dict[str, Interaction]:
    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    repeated = ResourceBundle(
        resource_id=resource_bundle.resource_id,
        version=resource_bundle.version,
        species=resource_bundle.species,
        gene_namespace=resource_bundle.gene_namespace,
        interactions=resource_bundle.interactions,
        mapping_report=resource_bundle.mapping_report,
        manifest_digest=resource_bundle.manifest_digest,
        source_files=resource_bundle.source_files,
        license=resource_bundle.license,
        citation=resource_bundle.citation,
    )
    if repeated != resource_bundle:
        raise _error(
            "ResourceBundle failed canonical reconstruction",
            code="crossfit_hypergraph_resource_integrity_violation",
            field="resource_bundle",
            remediation="Reload the immutable resource bundle",
        )
    result: dict[str, Interaction] = {}
    for interaction in repeated.interactions:
        if (
            interaction.version != repeated.version
            or interaction.species is not repeated.species
            or interaction.gene_namespace is not repeated.gene_namespace
        ):
            raise _error(
                "Interaction governance fields disagree with the ResourceBundle",
                code="invalid_crossfit_hypergraph_resource",
                field="interactions",
                remediation="Load one frozen resource release and namespace",
            )
        _validate_resource_entity(interaction, entity="ligand")
        _validate_resource_entity(interaction, entity="receptor")
        result[interaction.interaction_id] = interaction
    return result


def _source_status(
    raw_status: object,
    value: float | None,
    reason_code: str | None,
) -> HyperedgeStatus:
    status = _required_string(raw_status, field="status")
    if status in {"ok", "observed"}:
        resolved = HyperedgeStatus.OBSERVED
        valid = value is not None and reason_code is None
    elif status == "structural_zero":
        resolved = HyperedgeStatus.STRUCTURAL_ZERO
        valid = value == 0.0 and reason_code is not None
    elif status == "not_estimable":
        resolved = HyperedgeStatus.NOT_ESTIMABLE
        valid = value is None and reason_code is not None
    elif status == "failed":
        resolved = HyperedgeStatus.FAILED
        valid = value is None and reason_code is not None
    elif status == "filtered":
        resolved = HyperedgeStatus.FILTERED
        valid = value is None and reason_code is not None
    else:
        raise _error(
            f"Unsupported sender score status {status!r}",
            code="invalid_crossfit_hypergraph_status",
            field="status",
            remediation="Use released sender score availability states",
        )
    if not valid:
        raise _error(
            "Sender score value/reason does not match its status",
            code="invalid_crossfit_hypergraph_status",
            field="sender_resolved_strength",
            remediation="Regenerate the intact family-common scoring application",
        )
    return resolved


@dataclass(frozen=True, slots=True)
class _SourceRow:
    contrast_manifest_id: str
    contrast_name: str
    context_id: str
    context_json: str
    receiver: str
    family_id: str
    driver_id: str
    interaction_id: str
    mode: str
    sender: str
    sample_id: str
    subject_id: str
    value: float | None
    status: HyperedgeStatus
    reason_code: str | None
    fold_id: str
    functional_id: str
    application_id: str
    binding_id: str
    source_row_id: str

    @property
    def aggregate_key(self) -> tuple[str, ...]:
        return tuple(str(getattr(self, field)) for field in _AGGREGATE_KEY)


@dataclass(frozen=True, slots=True)
class _Collection:
    contrast_manifest_id: str
    contrast_name: str
    receiver: str
    collection_id: str
    functional_ids: tuple[str, ...]
    application_ids: tuple[str, ...]
    binding_ids: tuple[str, ...]
    fold_ids: tuple[str, ...]
    score_versions: tuple[str, ...]


def _context_mapping(fold: Any, functional: Any) -> dict[str, str]:
    matching = [
        (encoder, application)
        for encoder, application in zip(
            fold.design_encoders, fold.design_applications, strict=True
        )
        if encoder.contrast.name == functional.contrast_name
    ]
    if len(matching) != 1:
        raise _error(
            "Family-common functional does not map to exactly one design encoder",
            code="invalid_crossfit_hypergraph_context",
            field="contrast_name",
            remediation="Rerun cross-fit with contrast-unique frozen encoders",
        )
    encoder, design_application = matching[0]
    if encoder.contrast != functional.sender_functional.contrast:
        raise _error(
            "Sender and design functionals use different contrast definitions",
            code="invalid_crossfit_hypergraph_context",
            field="contrast",
            remediation="Rerun the intact subject cross-fit workflow",
        )
    contexts: dict[str, str] = {}
    for node, declared_context_id in functional.sender_functional.contrast_context_ids:
        context_id, context_json = node_context_fields(
            cast(Hashable, node), encoder.context_keys
        )
        if context_id != declared_context_id:
            raise _error(
                "Frozen sender context ID disagrees with canonical context encoding",
                code="invalid_crossfit_hypergraph_context",
                field="context_id",
                remediation="Regenerate contexts from the declared context keys",
            )
        contexts[context_id] = context_json
    if set(contexts) != set(functional.context_ids):
        raise _error(
            "Family-common functional context coverage is inconsistent",
            code="invalid_crossfit_hypergraph_context",
            field="context_ids",
            remediation="Rerun the intact family-common scoring workflow",
        )
    heldout_contexts = set(design_application.sample_context_ids)
    if not heldout_contexts.issubset(contexts):
        raise _error(
            "Held-out design contains context rows outside its functional",
            code="invalid_crossfit_hypergraph_context",
            field="sample_context_ids",
            remediation="Apply each contrast only to its declared contexts",
        )
    return contexts


def _expected_application_keys(
    fold: Any,
    functional: Any,
) -> set[tuple[str, ...]]:
    matching_design = [
        application
        for encoder, application in zip(
            fold.design_encoders, fold.design_applications, strict=True
        )
        if encoder.contrast.name == functional.contrast_name
    ]
    if len(matching_design) != 1:  # guarded by _context_mapping
        raise AssertionError("contrast design lookup changed")
    design_application = matching_design[0]
    samples = tuple(
        (sample_id, subject_id, context_id)
        for sample_id, subject_id, context_id in zip(
            design_application.sample_ids,
            design_application.sample_subject_ids,
            design_application.sample_context_ids,
            strict=True,
        )
        if context_id in set(functional.context_ids)
    )
    interactions = {
        item.interaction_id: (item.family_id, item.driver_id)
        for item in functional.interactions
    }
    if len(interactions) != len(functional.interactions):
        raise _error(
            "Functional interaction IDs are not unique",
            code="invalid_crossfit_hypergraph_coverage",
            field="interaction_id",
            remediation="Regenerate the frozen family interaction mapping",
        )
    candidates = {
        interaction_id: senders
        for receiver, interaction_id, senders in (
            functional.sender_functional.candidate_sender_manifest
        )
        if receiver == functional.receiver and interaction_id in interactions
    }
    interactions = {
        interaction_id: family_driver
        for interaction_id, family_driver in interactions.items()
        if interaction_id in candidates
    }
    return {
        (
            sample_id,
            subject_id,
            context_id,
            functional.receiver,
            family_id,
            driver_id,
            interaction_id,
            mode,
            sender,
        )
        for sample_id, subject_id, context_id in samples
        for interaction_id, (family_id, driver_id) in interactions.items()
        for sender in candidates[interaction_id]
        for mode in _EXPECTED_MODES
    }


def _validated_source_rows(
    artifacts: CrossFitArtifacts,
    resource_bundle: ResourceBundle,
    interactions: Mapping[str, Interaction],
) -> tuple[list[_SourceRow], dict[tuple[str, str, str], _Collection]]:
    rows: list[_SourceRow] = []
    collection_members: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    seen_sample_grain: set[tuple[str, ...]] = set()
    resource_ids = set(interactions)
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        training_bundle = fold.training.resource_bundle
        universe = fold.training.frozen_interaction_universe
        universe._require_intact()
        if training_bundle != resource_bundle or (
            universe.resource_id != resource_bundle.resource_id
            or universe.resource_version != resource_bundle.version
            or universe.resource_manifest_digest != resource_bundle.manifest_digest
            or not set(universe.interaction_ids).issubset(resource_ids)
        ):
            raise _error(
                "Cross-fit interaction universe does not match ResourceBundle",
                code="crossfit_hypergraph_resource_universe_mismatch",
                field="resource_bundle",
                remediation="Pass the exact frozen ResourceBundle used for cross-fit",
            )
        for functional, application, binding in zip(
            fold.family_common_functionals,
            fold.family_common_applications,
            fold.family_common_bindings,
            strict=True,
        ):
            functional._require_intact()
            application._require_intact()
            binding._require_intact()
            if functional.fold_id != fold.fold_id:
                raise _error(
                    "Family-common functional references another fold",
                    code="invalid_crossfit_hypergraph_lineage",
                    field="fold_id",
                    remediation="Rerun the producer-owned cross-fit workflow",
                )
            contexts = _context_mapping(fold, functional)
            if not set(functional.interaction_ids).issubset(resource_ids):
                raise _error(
                    "Functional interactions are outside the resource universe",
                    code="crossfit_hypergraph_resource_universe_mismatch",
                    field="interaction_id",
                    remediation="Use the exact ResourceBundle fitted by cross-fit",
                )
            table = application.sender_scores
            expected_keys = _expected_application_keys(fold, functional)
            actual_keys = [
                tuple(
                    _required_string(value, field=field)
                    for field, value in zip(_SOURCE_KEY, values, strict=True)
                )
                for values in table.loc[:, list(_SOURCE_KEY)].itertuples(
                    index=False, name=None
                )
            ]
            if len(actual_keys) != len(set(actual_keys)):
                raise _error(
                    "Sender scores contain duplicate sample-granularity rows",
                    code="duplicate_crossfit_hypergraph_sample_row",
                    field="sample_id",
                    remediation="Regenerate exact-coverage sender score tables",
                )
            if set(actual_keys) != expected_keys:
                raise _error(
                    "Sender scores do not exactly cover held-out samples "
                    "and candidates",
                    code="incomplete_crossfit_hypergraph_coverage",
                    field="sender_scores",
                    remediation=(
                        "Regenerate all observed, structural-zero, and not-estimable "
                        "sender rows"
                    ),
                )
            interaction_mapping = {
                item.interaction_id: (item.family_id, item.driver_id)
                for item in functional.interactions
            }
            collection_key = (
                functional.contrast_manifest_id,
                functional.contrast_name,
                functional.receiver,
            )
            collection_members.setdefault(collection_key, []).append(
                {
                    "application_id": application.application_id,
                    "binding_id": binding.binding_id,
                    "fold_id": fold.fold_id,
                    "functional_id": functional.family_common_functional_id,
                    "score_version": functional.score_version,
                }
            )
            for source in table.itertuples(index=False):
                source_values = {
                    field: _required_string(getattr(source, field), field=field)
                    for field in _SOURCE_KEY
                }
                interaction_id = source_values["interaction_id"]
                expected_family_driver = interaction_mapping.get(interaction_id)
                if expected_family_driver != (
                    source_values["family_id"],
                    source_values["driver_id"],
                ):
                    raise _error(
                        "Sender row disagrees with its frozen family interaction",
                        code="invalid_crossfit_hypergraph_lineage",
                        field="family_id",
                        remediation="Regenerate the family-common application",
                    )
                if source_values["context_id"] not in contexts:
                    raise _error(
                        "Sender row context is outside its contrast functional",
                        code="invalid_crossfit_hypergraph_context",
                        field="context_id",
                        remediation="Apply the functional only to declared contexts",
                    )
                if (
                    source.family_common_functional_id
                    != functional.family_common_functional_id
                    or source.sender_functional_id
                    != functional.sender_functional.sender_functional_id
                    or source.score_version != functional.score_version
                ):
                    raise _error(
                        "Sender row functional lineage is inconsistent",
                        code="invalid_crossfit_hypergraph_lineage",
                        field="family_common_functional_id",
                        remediation="Regenerate the intact scoring application",
                    )
                value = _optional_finite_float(
                    source.sender_resolved_strength,
                    field=_SOURCE_SCORE_COLUMN,
                )
                reason = _optional_string(source.reason_code, field="reason_code")
                status = _source_status(source.status, value, reason)
                sample_grain = (
                    functional.contrast_manifest_id,
                    *tuple(source_values[field] for field in _SOURCE_KEY),
                )
                if sample_grain in seen_sample_grain:
                    raise _error(
                        "A sender score sample is repeated across cross-fit folds",
                        code="duplicate_crossfit_hypergraph_sample_row",
                        field="sample_id",
                        remediation="Restore disjoint held-out fold coverage",
                    )
                seen_sample_grain.add(sample_grain)
                source_payload = {
                    "application_id": application.application_id,
                    "binding_id": binding.binding_id,
                    "fold_id": fold.fold_id,
                    "functional_id": functional.family_common_functional_id,
                    "reason_code": reason,
                    "row": source_values,
                    "score_version": functional.score_version,
                    "status": status.value,
                    "value": value,
                }
                rows.append(
                    _SourceRow(
                        contrast_manifest_id=functional.contrast_manifest_id,
                        contrast_name=functional.contrast_name,
                        context_id=source_values["context_id"],
                        context_json=contexts[source_values["context_id"]],
                        receiver=source_values["receiver"],
                        family_id=source_values["family_id"],
                        driver_id=source_values["driver_id"],
                        interaction_id=interaction_id,
                        mode=source_values["mode"],
                        sender=source_values["sender"],
                        sample_id=source_values["sample_id"],
                        subject_id=source_values["subject_id"],
                        value=value,
                        status=status,
                        reason_code=reason,
                        fold_id=fold.fold_id,
                        functional_id=functional.family_common_functional_id,
                        application_id=application.application_id,
                        binding_id=binding.binding_id,
                        source_row_id=stable_id(
                            "crossfit_hypergraph_sender_source_row",
                            source_payload,
                            schema_version="1",
                        ),
                    )
                )
    collections: dict[tuple[str, str, str], _Collection] = {}
    for key, raw_members in collection_members.items():
        members = sorted(raw_members, key=canonical_json)
        fold_ids = tuple(sorted({member["fold_id"] for member in members}))
        if len(fold_ids) != len(artifacts.folds):
            raise _error(
                "Receiver scoring collection does not cover every outer fold",
                code="incomplete_crossfit_hypergraph_functional_collection",
                field="fold_id",
                remediation="Export only complete receiver-by-contrast collections",
            )
        collection_id = stable_id(
            "crossfit_hypergraph_functional_collection",
            {
                "contrast_manifest_id": key[0],
                "contrast_name": key[1],
                "crossfit_id": artifacts.crossfit_id,
                "members": members,
                "receiver": key[2],
                "repeat_id": artifacts.spec.repeat_id,
            },
            schema_version="1",
        )
        collections[key] = _Collection(
            contrast_manifest_id=key[0],
            contrast_name=key[1],
            receiver=key[2],
            collection_id=collection_id,
            functional_ids=tuple(
                sorted({member["functional_id"] for member in members})
            ),
            application_ids=tuple(
                sorted({member["application_id"] for member in members})
            ),
            binding_ids=tuple(sorted({member["binding_id"] for member in members})),
            fold_ids=fold_ids,
            score_versions=tuple(
                sorted({member["score_version"] for member in members})
            ),
        )
    if not rows or not collections:
        raise _error(
            "Cross-fit artifacts contain no family-common sender scores",
            code="crossfit_hypergraph_sender_scores_unavailable",
            field="family_common_applications",
            remediation="Run cross-fit with penalty tuning and family-common scoring",
        )
    return rows, collections


def _unavailable_status(rows: list[_SourceRow]) -> HyperedgeStatus | None:
    unavailable = [
        row.status
        for row in rows
        if row.status
        not in {HyperedgeStatus.OBSERVED, HyperedgeStatus.STRUCTURAL_ZERO}
    ]
    if not unavailable:
        return None
    return max(unavailable, key=lambda status: _STATUS_PRECEDENCE[status])


def _aggregate_status_and_weight(
    rows: list[_SourceRow],
) -> tuple[HyperedgeStatus, float | None, str | None, tuple[dict[str, object], ...]]:
    by_subject: dict[str, list[_SourceRow]] = {}
    for row in rows:
        by_subject.setdefault(row.subject_id, []).append(row)
    subject_records: list[dict[str, object]] = []
    subject_values: list[float] = []
    unavailable_status: HyperedgeStatus | None = None
    for subject_id in sorted(by_subject):
        subject_rows = by_subject[subject_id]
        subject_unavailable = _unavailable_status(subject_rows)
        if subject_unavailable is None:
            values = [cast(float, row.value) for row in subject_rows]
            subject_value = math.fsum(values) / len(values)
            subject_values.append(subject_value)
            subject_status = (
                HyperedgeStatus.STRUCTURAL_ZERO
                if all(
                    row.status is HyperedgeStatus.STRUCTURAL_ZERO
                    for row in subject_rows
                )
                else HyperedgeStatus.OBSERVED
            )
        else:
            subject_value = None
            subject_status = subject_unavailable
            if (
                unavailable_status is None
                or _STATUS_PRECEDENCE[subject_unavailable]
                > _STATUS_PRECEDENCE[unavailable_status]
            ):
                unavailable_status = subject_unavailable
        subject_records.append(
            {
                "sample_count": len(subject_rows),
                "source_row_ids": sorted(row.source_row_id for row in subject_rows),
                "status": subject_status.value,
                "subject_id": subject_id,
                "technical_sample_mean": subject_value,
            }
        )
    if unavailable_status is not None:
        reasons = sorted(
            {
                row.reason_code
                for row in rows
                if row.status
                not in {HyperedgeStatus.OBSERVED, HyperedgeStatus.STRUCTURAL_ZERO}
                and row.reason_code is not None
            }
        )
        reason = (
            reasons[0]
            if len(reasons) == 1
            else "multiple_source_unavailable_reasons_v1"
        )
        return unavailable_status, None, reason, tuple(subject_records)
    weight = math.fsum(subject_values) / len(subject_values)
    if all(row.status is HyperedgeStatus.STRUCTURAL_ZERO for row in rows):
        reasons = sorted({cast(str, row.reason_code) for row in rows})
        reason = (
            reasons[0]
            if len(reasons) == 1
            else "all_source_rows_structural_zero_multiple_reasons_v1"
        )
        return HyperedgeStatus.STRUCTURAL_ZERO, 0.0, reason, tuple(subject_records)
    return HyperedgeStatus.OBSERVED, weight, None, tuple(subject_records)


def _edge_record(
    rows: list[_SourceRow],
    *,
    artifacts: CrossFitArtifacts,
    resource_bundle: ResourceBundle,
    interaction: Interaction,
    collection: _Collection,
) -> CommunicationEdgeRecord:
    rows = sorted(rows, key=lambda row: row.source_row_id)
    first = rows[0]
    if any(row.aggregate_key != first.aggregate_key for row in rows):
        raise AssertionError("aggregate grouping changed")
    if len({row.context_json for row in rows}) != 1:
        raise _error(
            "A canonical context ID maps to multiple context payloads",
            code="invalid_crossfit_hypergraph_context",
            field="context_json",
            remediation="Regenerate contexts from a single frozen schema",
        )
    driver_ids = tuple(sorted({row.driver_id for row in rows}))
    if len(driver_ids) != 1:
        raise _error(
            "An aggregated family interaction maps to multiple drivers",
            code="invalid_crossfit_hypergraph_lineage",
            field="driver_id",
            remediation="Keep driver-family mappings fold-stable before aggregation",
        )
    status, weight, reason, subject_records = _aggregate_status_and_weight(rows)
    ligand, ligand_members = _validate_resource_entity(interaction, entity="ligand")
    receptor, receptor_members = _validate_resource_entity(
        interaction, entity="receptor"
    )
    source_row_ids = tuple(row.source_row_id for row in rows)
    source_record_id = stable_id(
        "crossfit_communication_hypergraph_record",
        {
            "biological_key": {
                field: getattr(first, field) for field in _AGGREGATE_KEY
            },
            "collection_id": collection.collection_id,
            "source_row_ids": list(source_row_ids),
            "weight_semantics": CROSSFIT_HYPERGRAPH_WEIGHT_SEMANTICS,
        },
        schema_version="1",
    )
    source_fold_ids = tuple(sorted({row.fold_id for row in rows}))
    missing_fold_ids = tuple(
        fold_id for fold_id in collection.fold_ids if fold_id not in source_fold_ids
    )
    if missing_fold_ids:
        status = HyperedgeStatus.NOT_ESTIMABLE
        weight = None
        reason = "family_interaction_or_sender_not_present_in_all_folds_v1"
    unavailable_rows = [
        {
            "reason_code": row.reason_code,
            "source_row_id": row.source_row_id,
            "status": row.status.value,
        }
        for row in rows
        if row.status
        not in {HyperedgeStatus.OBSERVED, HyperedgeStatus.STRUCTURAL_ZERO}
    ]
    components = {
        "aggregation": {
            "sample_stage": "technical_samples_to_subject_context_mean_v1",
            "status_policy": "complete_observed_or_structural_zero_else_unavailable_v1",
            "subject_stage": "subject_equal_mean_v1",
        },
        "contrast": {
            "manifest_id": first.contrast_manifest_id,
            "name": first.contrast_name,
        },
        "driver_ids": list(driver_ids),
        "missing_source_fold_ids": list(missing_fold_ids),
        "source_component": _SOURCE_SCORE_COLUMN,
        "subject_count": len(subject_records),
        "target": {
            "entity_id": first.family_id,
            "kind": TargetKind.PROGRAM.value,
            "semantics": "fold_frozen_driver_family_not_gene_signature_v1",
        },
    }
    provenance = {
        "adapter_version": CROSSFIT_HYPERGRAPH_EXPORT_VERSION,
        "collection_id": collection.collection_id,
        "crossfit_id": artifacts.crossfit_id,
        "missing_source_fold_ids": list(missing_fold_ids),
        "resource": {
            "manifest_digest": resource_bundle.manifest_digest,
            "resource_id": resource_bundle.resource_id,
            "version": resource_bundle.version,
        },
        "source_application_ids": list(collection.application_ids),
        "source_binding_ids": list(collection.binding_ids),
        "source_fold_ids": list(collection.fold_ids),
        "source_functional_ids": list(collection.functional_ids),
        "source_row_ids": list(source_row_ids),
        "source_score_versions": list(collection.score_versions),
        "subject_aggregation": list(subject_records),
        "unavailable_source_rows": unavailable_rows,
    }
    return CommunicationEdgeRecord(
        source_record_id=source_record_id,
        context_id=first.context_id,
        context_json=first.context_json,
        sender=first.sender,
        ligand=ligand,
        ligand_members=ligand_members,
        receptor=receptor,
        receptor_members=receptor_members,
        receiver=first.receiver,
        target=first.family_id,
        target_kind=TargetKind.PROGRAM,
        interaction_id=first.interaction_id,
        mode=first.mode,
        weight=weight,
        weight_semantics=CROSSFIT_HYPERGRAPH_WEIGHT_SEMANTICS,
        uncertainty=None,
        uncertainty_kind=None,
        status=status,
        reason_code=reason,
        source_artifact_id=artifacts.crossfit_id,
        scoring_functional_id=collection.collection_id,
        fold_id=source_fold_ids[0] if len(source_fold_ids) == 1 else None,
        repeat_id=artifacts.spec.repeat_id,
        components_json=canonical_json(components),
        provenance_json=canonical_json(provenance),
    )


def build_crossfit_communication_hypergraph(
    artifacts: CrossFitArtifacts,
    resource_bundle: ResourceBundle,
) -> CommunicationHypergraph:
    """Aggregate held-out sender-resolved scores into a descriptive hypergraph.

    Technical samples are averaged within subject and context before subjects
    receive equal weight. Any unavailable source row makes its complete edge
    unavailable. This post-fit representation never releases p/q values,
    probabilities, or gene-signature claims.
    """

    if type(artifacts) is not CrossFitArtifacts:
        raise TypeError("artifacts must be producer-owned CrossFitArtifacts")
    artifacts._require_intact()
    interactions = _validated_interactions(resource_bundle)
    source_rows, collections = _validated_source_rows(
        artifacts, resource_bundle, interactions
    )
    grouped: dict[tuple[str, ...], list[_SourceRow]] = {}
    for row in source_rows:
        grouped.setdefault(row.aggregate_key, []).append(row)
    records: list[CommunicationEdgeRecord] = []
    for key in sorted(grouped):
        rows = grouped[key]
        first = rows[0]
        collection = collections[
            (
                first.contrast_manifest_id,
                first.contrast_name,
                first.receiver,
            )
        ]
        records.append(
            _edge_record(
                rows,
                artifacts=artifacts,
                resource_bundle=resource_bundle,
                interaction=interactions[first.interaction_id],
                collection=collection,
            )
        )
    return build_communication_hypergraph(records)


__all__ = [
    "CROSSFIT_HYPERGRAPH_EXPORT_VERSION",
    "CROSSFIT_HYPERGRAPH_WEIGHT_SEMANTICS",
    "build_crossfit_communication_hypergraph",
]
