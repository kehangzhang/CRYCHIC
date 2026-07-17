"""Atomic persistence for the dedicated directional integrated-LR sidecar."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

from crychic.core import canonical_digest, canonical_json, stable_id
from crychic.design import node_context_fields
from crychic.results.errors import (
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
)

from .crossfit_persistence import (
    CROSSFIT_COMPONENT_TABLE,
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE,
    CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE,
    CROSSFIT_RESULT_SCHEMA_VERSION,
    CrossFitResult,
)
from .directional_integrated_lr import (
    DIRECTIONAL_INTEGRATED_LR_ANALYSIS_TRACK,
    DIRECTIONAL_INTEGRATED_LR_COLUMNS,
    DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_COLUMNS,
    DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION,
    DIRECTIONAL_INTEGRATED_LR_SEMANTICS,
    DirectionalIntegratedLRCollection,
    _opportunity_status_counts,
    _status_counts,
    _table_digest,
)

DIRECTIONAL_INTEGRATED_LR_RESULT_SCHEMA_VERSION = "2.0.0"
DIRECTIONAL_INTEGRATED_LR_COLLECTION_SCHEMA_VERSION = "4.0.0"
DIRECTIONAL_INTEGRATED_LR_SCORE_TABLE = "scores"
DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_TABLE = "opportunities"

_ARTIFACT_KIND = "crychic.directional_integrated_lr_result"
_PRODUCER = "crychic.workflow.directional_integrated_lr_persistence.v2"
_COLLECTION_PRODUCER = "crychic.workflow.directional_integrated_lr.v4"
_STATUS_FILENAME = "_status.json"
_MANIFEST_FILENAME = "directional_integrated_lr_manifest.json"
_COMPLETE = "complete"
_INCOMPLETE = "incomplete"
_TABLE_SCHEMA_VERSION = "2.0.0"
_TABLE_FILENAMES = {
    DIRECTIONAL_INTEGRATED_LR_SCORE_TABLE: ("directional_integrated_lr_scores.parquet"),
    DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_TABLE: (
        "directional_integrated_lr_opportunities.parquet"
    ),
}
_TABLE_COLUMNS = {
    DIRECTIONAL_INTEGRATED_LR_SCORE_TABLE: DIRECTIONAL_INTEGRATED_LR_COLUMNS,
    DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_TABLE: (
        DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_COLUMNS
    ),
}
_SCORE_PRIMARY_KEY = (
    "fold_id",
    "receiver",
    "channel_role",
    "sample_id",
    "subject_id",
    "context_id",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
)
_OPPORTUNITY_PRIMARY_KEY = ("fold_id", "receiver")
_TABLE_PRIMARY_KEYS = {
    DIRECTIONAL_INTEGRATED_LR_SCORE_TABLE: _SCORE_PRIMARY_KEY,
    DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_TABLE: _OPPORTUNITY_PRIMARY_KEY,
}
_PARENT_TABLES = (
    CROSSFIT_COMPONENT_TABLE,
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE,
    CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE,
)
_MEMBER_COMPONENTS = frozenset(
    {
        "receptor_eligible",
        "ligand_contrast_gate",
        "receptor_gate",
        "availability",
        "ligand_availability",
        "subject_prevalence",
        "resource_evidence",
        "prior_quality",
        "member_evidence_score",
        "within_family_lr_weight",
        "within_family_entropy",
        "family_core_strength",
        "sender_unresolved_strength",
    }
)
_REPLAY_COMPONENTS = (
    "sender_unresolved_strength",
    "family_core_strength",
    "within_family_lr_weight",
)
_ABSENT_RECEIVER_REASON = "receiver_absent_in_outer_training"
_PAIR_NE_REASON = "directional_pair_not_estimable"
_FITTED_SCORE_PARENTS = (
    "directional_binding_id",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_binding_id",
    "family_common_member_scores_digest",
    "family_common_family_scores_digest",
    "edge_evidence_digest",
    "incremental_training_artifact_id",
    "incremental_application_id",
)
_OPPORTUNITY_PARENT_COLUMNS = (
    "forward_response_id",
    "reverse_response_id",
    "forward_incremental_training_artifact_id",
    "reverse_incremental_training_artifact_id",
    "forward_incremental_application_id",
    "reverse_incremental_application_id",
    "forward_family_common_functional_id",
    "reverse_family_common_functional_id",
    "forward_family_common_application_id",
    "reverse_family_common_application_id",
    "forward_family_common_binding_id",
    "reverse_family_common_binding_id",
)
_CLAIM_FLAGS = (
    "paired_score_comparison_allowed",
    "active_inhibition_allowed",
    "supports_active_inhibition_claim",
    "formal_inference_allowed",
)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain one JSON object")
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(f"{canonical_json(value)}\n", encoding="utf-8")


def _mark_incomplete(path: Path, error_type: str | None = None) -> None:
    marker: dict[str, object] = {
        "artifact_schema_version": (DIRECTIONAL_INTEGRATED_LR_RESULT_SCHEMA_VERSION),
        "artifact_kind": _ARTIFACT_KIND,
        "status": _INCOMPLETE,
    }
    if error_type is not None:
        marker["error_type"] = error_type
    try:
        _write_json(path / _STATUS_FILENAME, marker)
    except OSError:
        pass


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a JSON array")
    return list(value)


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    result = str(value)
    return result if result else None


def _optional_float(value: object, *, field_name: str) -> float | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    result = float(cast(Any, value))
    if math.isnan(result):
        return None
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite or missing")
    return result


def _exact_parent(
    value: CrossFitResult | str | Path,
) -> CrossFitResult:
    if isinstance(value, CrossFitResult):
        if type(value) is not CrossFitResult:
            raise TypeError("source_crossfit_result must be an exact CrossFitResult")
        supplied = value
        loaded = CrossFitResult.load(supplied.path)
        if loaded.manifest != supplied.manifest:
            raise ResultValidationError(
                "Supplied cross-fit parent differs from its current result directory",
                code="directional_integrated_lr_parent_object_mismatch",
                field="source_crossfit_result",
                remediation="Pass the exact, freshly loaded parent result",
            )
    elif isinstance(value, (str, Path)):
        loaded = CrossFitResult.load(value)
    else:
        raise TypeError("source_crossfit_result must be CrossFitResult or a path")
    if loaded.manifest.get("schema_version") != CROSSFIT_RESULT_SCHEMA_VERSION:
        raise ResultValidationError(
            "Directional integrated-LR persistence requires an exact v7 parent",
            code="directional_integrated_lr_parent_schema_mismatch",
            field="source_crossfit_result",
            remediation="Regenerate the parent with the current v7 writer",
        )
    return loaded


def _parent_source(parent: CrossFitResult) -> dict[str, Any]:
    manifest = parent.manifest
    source = _mapping(
        manifest.get("source_crossfit_manifest"),
        field_name="source_crossfit_manifest",
    )
    if canonical_digest(source) != manifest.get("source_crossfit_manifest_digest"):
        raise ValueError("parent source cross-fit manifest digest changed")
    return source


def _source_pair(source: Mapping[str, Any], pair_spec_id: str) -> dict[str, Any]:
    spec = _mapping(source.get("spec"), field_name="source.spec")
    pairs = _sequence(
        spec.get("directional_pairs"), field_name="source.spec.directional_pairs"
    )
    matches = [
        _mapping(pair, field_name="source directional pair")
        for pair in pairs
        if isinstance(pair, Mapping) and pair.get("pair_spec_id") == pair_spec_id
    ]
    if len(matches) != 1:
        raise ValueError("parent does not contain exactly one requested pair")
    return matches[0]


def _role_contract(
    pair: Mapping[str, Any], role: str
) -> tuple[str, str, dict[str, Any], str]:
    if role not in {"forward", "reverse"}:
        raise ValueError("channel role must be forward or reverse")
    channel = _required_text(
        pair.get(f"{role}_channel_name"), field_name=f"pair.{role}_channel_name"
    )
    contrast = _mapping(
        pair.get(f"{role}_contrast"), field_name=f"pair.{role}_contrast"
    )
    contrast_name = _required_text(
        contrast.get("name"), field_name=f"pair.{role}_contrast.name"
    )
    contrast_id = _required_text(
        pair.get(f"{role}_contrast_id"),
        field_name=f"pair.{role}_contrast_id",
    )
    expected_contrast_id = stable_id("contrast", contrast)
    if contrast_id != expected_contrast_id:
        raise ValueError("directional contrast identity changed")
    return channel, contrast_name, contrast, contrast_id


def _source_fold_map(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    folds = _sequence(source.get("fold_artifacts"), field_name="fold_artifacts")
    result: dict[str, dict[str, Any]] = {}
    for raw in folds:
        fold = _mapping(raw, field_name="fold artifact")
        fold_id = _required_text(fold.get("fold_id"), field_name="fold_id")
        if fold_id in result:
            raise ValueError("source manifest has duplicate folds")
        result[fold_id] = fold
    if not result:
        raise ValueError("source manifest has no folds")
    return result


def _universe_contract(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, str], ...]], tuple[str, ...]]:
    universe = _mapping(
        source.get("directional_lr_hypothesis_universe"),
        field_name="directional_lr_hypothesis_universe",
    )
    receiver_universe = _mapping(
        source.get("receiver_universe"), field_name="receiver_universe"
    )
    receiver_ids = tuple(
        _required_text(value, field_name="receiver_universe.receiver_ids")
        for value in _sequence(
            receiver_universe.get("receiver_ids"),
            field_name="receiver_universe.receiver_ids",
        )
    )
    if not receiver_ids or len(receiver_ids) != len(set(receiver_ids)):
        raise ValueError("receiver universe axis is empty or duplicated")
    if (
        universe.get("receiver_ids") != list(receiver_ids)
        or universe.get("receiver_axis_id") != receiver_universe.get("receiver_axis_id")
        or universe.get("receiver_universe_id") != receiver_universe.get("universe_id")
    ):
        raise ValueError("directional LR universe differs from receiver universe")
    molecular_by_membership = {
        _required_text(row.get("membership_id"), field_name="membership_id"): (
            _required_text(
                row.get("molecular_lr_equivalence_id"),
                field_name="molecular_lr_equivalence_id",
            )
        )
        for row in (
            _mapping(value, field_name="membership")
            for value in _sequence(
                universe.get("memberships"), field_name="universe.memberships"
            )
        )
    }
    if not molecular_by_membership or len(molecular_by_membership) != universe.get(
        "membership_count"
    ):
        raise ValueError("directional LR membership axis is inconsistent")
    hypotheses: dict[tuple[str, str], dict[str, Any]] = {}
    for value in _sequence(
        universe.get("hypotheses"), field_name="universe.hypotheses"
    ):
        row = _mapping(value, field_name="hypothesis")
        membership_id = _required_text(
            row.get("membership_id"), field_name="membership_id"
        )
        molecular_id = _required_text(
            row.get("molecular_lr_equivalence_id"),
            field_name="molecular_lr_equivalence_id",
        )
        if molecular_by_membership.get(membership_id) != molecular_id:
            raise ValueError("LR hypothesis differs from its molecular membership")
        key = (
            _required_text(row.get("interaction_id"), field_name="interaction_id"),
            _required_text(row.get("mode"), field_name="mode"),
        )
        if key in hypotheses:
            raise ValueError("directional LR hypothesis key is duplicated")
        hypotheses[key] = row
    if not hypotheses or len(hypotheses) != universe.get("hypothesis_count"):
        raise ValueError("directional LR hypothesis axis is inconsistent")
    by_receiver: dict[str, list[dict[str, str]]] = {
        receiver: [] for receiver in receiver_ids
    }
    seen_opportunity_ids: set[str] = set()
    for value in _sequence(
        universe.get("opportunities"), field_name="universe.opportunities"
    ):
        raw = _mapping(value, field_name="LR opportunity")
        receiver = _required_text(raw.get("receiver"), field_name="receiver")
        key = (
            _required_text(raw.get("interaction_id"), field_name="interaction_id"),
            _required_text(raw.get("mode"), field_name="mode"),
        )
        if receiver not in by_receiver or key not in hypotheses:
            raise ValueError("LR opportunity is outside the frozen axes")
        hypothesis = hypotheses[key]
        for field_name in (
            "driver_id",
            "family_id",
            "molecular_lr_equivalence_id",
            "membership_id",
            "hypothesis_id",
        ):
            if raw.get(field_name) != hypothesis.get(field_name):
                raise ValueError("LR opportunity differs from its hypothesis")
        opportunity_id = _required_text(
            raw.get("opportunity_id"), field_name="opportunity_id"
        )
        if opportunity_id in seen_opportunity_ids:
            raise ValueError("LR opportunity identity is duplicated")
        seen_opportunity_ids.add(opportunity_id)
        by_receiver[receiver].append(
            {
                "receiver": receiver,
                "interaction_id": key[0],
                "mode": key[1],
                "driver_id": _required_text(raw.get("driver_id"), field_name="driver"),
                "family_id": _required_text(raw.get("family_id"), field_name="family"),
                "molecular_lr_equivalence_id": _required_text(
                    raw.get("molecular_lr_equivalence_id"),
                    field_name="molecular_lr_equivalence_id",
                ),
                "receiver_family_lr_membership_id": _required_text(
                    raw.get("membership_id"), field_name="membership_id"
                ),
                "receiver_family_lr_hypothesis_id": _required_text(
                    raw.get("hypothesis_id"), field_name="hypothesis_id"
                ),
                "receiver_family_lr_opportunity_id": opportunity_id,
            }
        )
    expected_count = len(receiver_ids) * len(hypotheses)
    if (
        len(seen_opportunity_ids) != expected_count
        or universe.get("opportunity_count") != expected_count
        or any(
            {(row["interaction_id"], row["mode"]) for row in rows} != set(hypotheses)
            for rows in by_receiver.values()
        )
    ):
        raise ValueError("LR opportunity grid is not the exact receiver product")
    return (
        universe,
        {receiver: tuple(rows) for receiver, rows in by_receiver.items()},
        receiver_ids,
    )


def _design_contract(
    fold: Mapping[str, Any],
    pair: Mapping[str, Any],
    role: str,
) -> tuple[str, tuple[tuple[str, str, str], ...], str]:
    _channel, contrast_name, contrast, contrast_id = _role_contract(pair, role)
    entries = [
        _mapping(value, field_name="directional design application")
        for value in _sequence(
            fold.get("directional_design_applications"),
            field_name="directional_design_applications",
        )
    ]
    matches = [
        entry
        for entry in entries
        if entry.get("contrast_id") == contrast_id
        and entry.get("contrast_name") == contrast_name
    ]
    if len(matches) != 1:
        raise ValueError("fold lacks one exact directional design application")
    entry = matches[0]
    encoder = _mapping(entry.get("encoder"), field_name="design encoder")
    application = _mapping(entry.get("application"), field_name="design application")
    if (
        encoder.get("contrast") != contrast
        or application.get("application_scope") != "heldout"
        or application.get("encoder_id") != encoder.get("encoder_id")
        or application.get("context_regressor_id")
        != encoder.get("context_regressor_id")
        or application.get("nuisance_design_id") != encoder.get("nuisance_design_id")
    ):
        raise ValueError("directional design lineage differs from its contrast")
    expected_encoder_id = stable_id(
        "frozen_design_encoder",
        {
            "context_regressor_id": encoder.get("context_regressor_id"),
            "nuisance_design_id": encoder.get("nuisance_design_id"),
        },
        schema_version="3",
    )
    if encoder.get("encoder_id") != expected_encoder_id:
        raise ValueError("directional design encoder identity changed")
    context_keys = tuple(
        _required_text(value, field_name="context key")
        for value in _sequence(encoder.get("context_keys"), field_name="context_keys")
    )
    context_domain: set[str] = set()
    weight_nodes: set[str] = set()
    weight_total = 0.0
    for value in _sequence(contrast.get("weights"), field_name="contrast.weights"):
        weight = _mapping(value, field_name="contrast weight")
        if set(weight) != {"context", "weight"}:
            raise ValueError("directional contrast weight schema changed")
        node: Any = weight.get("context")
        node_token = canonical_json(node)
        if node_token in weight_nodes:
            raise ValueError("directional contrast contains duplicate context weights")
        weight_nodes.add(node_token)
        if isinstance(node, list):
            if not all(isinstance(item, list) and len(item) == 2 for item in node):
                raise ValueError("multi-factor contrast context is malformed")
            node = tuple((str(item[0]), item[1]) for item in node)
        context_domain.add(node_context_fields(node, context_keys)[0])
        numeric_weight = float(cast(Any, weight.get("weight")))
        if not math.isfinite(numeric_weight) or numeric_weight == 0.0:
            raise ValueError("directional contrast weight must be finite and nonzero")
        weight_total += numeric_weight
    if not context_domain or not math.isclose(weight_total, 0.0, abs_tol=1.0e-12):
        raise ValueError("directional contrast weights are empty or unbalanced")
    sample_ids = _sequence(application.get("sample_ids"), field_name="sample_ids")
    subject_ids = _sequence(
        application.get("sample_subject_ids"), field_name="sample_subject_ids"
    )
    context_ids = _sequence(
        application.get("sample_context_ids"), field_name="sample_context_ids"
    )
    if not (len(sample_ids) == len(subject_ids) == len(context_ids)):
        raise ValueError("directional design sample arrays differ in length")
    if (
        any(
            not isinstance(value, str) or not value or value != value.strip()
            for values in (sample_ids, subject_ids, context_ids)
            for value in values
        )
        or len(sample_ids) != len(set(sample_ids))
        or application.get("subject_ids") != sorted(set(subject_ids))
        or sample_ids
        != sorted(
            sample_ids,
            key=lambda value: canonical_json({"type": "builtins.str", "value": value}),
        )
    ):
        raise ValueError("directional design sample lineage is not canonical")
    application_identity = {
        "application_scope": application.get("application_scope"),
        "context_regressor_digest": application.get("context_regressor_digest"),
        "context_regressor_id": application.get("context_regressor_id"),
        "encoder_id": application.get("encoder_id"),
        "nuisance_design_id": application.get("nuisance_design_id"),
        "nuisance_matrix_digest": application.get("nuisance_matrix_digest"),
        "reason_code": application.get("reason_code"),
        "rows": [
            {
                "context_id": str(context_id),
                "sample_id": str(sample_id),
                "subject_id": str(subject_id),
            }
            for sample_id, subject_id, context_id in zip(
                sample_ids, subject_ids, context_ids, strict=True
            )
        ],
        "status": application.get("status"),
        "subject_ids": application.get("subject_ids"),
    }
    application_status = application.get("status")
    application_reason = application.get("reason_code")
    if application_status not in {"observed", "not_estimable"} or (
        (application_status == "observed") == (application_reason is not None)
    ):
        raise ValueError("directional design application status is invalid")
    if application.get("application_id") != stable_id(
        "frozen_design_application", application_identity, schema_version="3"
    ):
        raise ValueError("directional design application identity changed")
    rows = tuple(
        (str(sample_id), str(subject_id), str(context_id))
        for sample_id, subject_id, context_id in zip(
            sample_ids, subject_ids, context_ids, strict=True
        )
        if str(context_id) in context_domain
    )
    if (
        not rows
        or len(rows) != len({row[0] for row in rows})
        or {row[2] for row in rows} != context_domain
    ):
        raise ValueError("directional design does not cover its exact context domain")
    application_id = _required_text(
        application.get("application_id"), field_name="design_application_id"
    )
    return application_id, rows, str(stable_id("contrast_manifest", contrast))


def _one_record(
    records: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
) -> dict[str, Any]:
    if len(records) != 1:
        raise ValueError(f"{field_name} must contain exactly one record")
    return dict(records[0])


def _support_contract(
    parent: CrossFitResult,
    source: Mapping[str, Any],
    fold_map: Mapping[str, Mapping[str, Any]],
    receiver_ids: tuple[str, ...],
) -> dict[tuple[str, str], dict[str, Any]]:
    table = parent.read_receiver_training_support()
    support: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_row in table.to_dict(orient="records"):
        row = cast(dict[str, Any], raw_row)
        key = (str(row["fold_id"]), str(row["receiver"]))
        if key in support:
            raise ValueError("receiver support table has duplicate keys")
        support[key] = row
    expected_keys = {
        (fold_id, receiver) for fold_id in fold_map for receiver in receiver_ids
    }
    if set(support) != expected_keys:
        raise ValueError("receiver support does not cover fold-by-receiver axis")
    for fold_id, fold in fold_map.items():
        source_rows = _sequence(
            fold.get("receiver_training_support"),
            field_name="fold.receiver_training_support",
        )
        source_by_receiver = {
            str(_mapping(value, field_name="source support").get("receiver_id")): (
                _mapping(value, field_name="source support")
            )
            for value in source_rows
        }
        if set(source_by_receiver) != set(receiver_ids):
            raise ValueError("source support differs from receiver axis")
        for receiver in receiver_ids:
            persisted = support[(fold_id, receiver)]
            raw = source_by_receiver[receiver]
            if (
                raw.get("outer_fold_id") != fold_id
                or raw.get("support_record_id")
                != persisted["receiver_training_support_id"]
                or raw.get("status") != persisted["receiver_training_support_status"]
                or raw.get("reason_code")
                != _optional_text(persisted["receiver_training_support_reason_code"])
                or raw.get("receiver_universe_id") != persisted["receiver_universe_id"]
                or canonical_json(raw.get("training_cell_type_ids"))
                != persisted["training_cell_type_ids"]
            ):
                raise ValueError("receiver support table differs from source manifest")
    return support


def _component_source_rows(
    components: pd.DataFrame,
    *,
    fold_id: str,
    receiver: str,
    contrast_name: str,
    contrast_manifest_id: str,
    family_parent: Mapping[str, Any],
) -> tuple[dict[str, object], ...]:
    selected = components.loc[
        components["fold_id"].astype(str).eq(fold_id)
        & components["receiver"].astype(str).eq(receiver)
        & components["contrast"].astype(str).eq(contrast_name)
        & components["component_scope"].astype(str).eq("lr_member")
        & components["source_table"].astype(str).eq("member_scores")
    ].copy()
    if selected.empty or selected["sender"].notna().any():
        raise ValueError("supported channel lacks sender-unresolved member components")
    rows: list[dict[str, object]] = []
    for _source_row_id, group in selected.groupby(
        "source_row_id", sort=False, dropna=False
    ):
        if (
            len(group) != len(_MEMBER_COMPONENTS)
            or set(group["component"].astype(str)) != _MEMBER_COMPONENTS
            or group["component"].duplicated().any()
        ):
            raise ValueError("member component ledger is incomplete or duplicated")
        invariant_columns = (
            "crossfit_id",
            "spec_id",
            "fold_id",
            "contrast_id",
            "contrast",
            "sample_id",
            "subject_id",
            "context_id",
            "receiver",
            "family_id",
            "driver_id",
            "interaction_id",
            "mode",
            "row_status",
            "row_reason_code",
            "family_common_functional_id",
            "family_common_application_id",
            "family_common_binding_id",
        )
        if any(
            group[column].nunique(dropna=False) != 1 for column in invariant_columns
        ):
            raise ValueError("member component rows disagree on source lineage")
        base = group.iloc[0]
        if (
            str(base["contrast_id"]) != contrast_manifest_id
            or str(base["family_common_functional_id"])
            != family_parent.get("functional_id")
            or str(base["family_common_application_id"])
            != family_parent.get("application_id")
            or str(base["family_common_binding_id"]) != family_parent.get("binding_id")
        ):
            raise ValueError("component ledger differs from family-common lineage")
        values = {
            component: _optional_float(
                group.loc[
                    group["component"].astype(str).eq(component), "component_value"
                ].iloc[0],
                field_name=component,
            )
            for component in _REPLAY_COMPONENTS
        }
        raw_status = str(base["row_status"])
        source_status = "observed" if raw_status == "ok" else raw_status
        source_reason = _optional_text(base["row_reason_code"])
        if source_status not in {"observed", "structural_zero", "not_estimable"}:
            raise ValueError("component source status is unsupported")
        score = values["sender_unresolved_strength"]
        if (
            (
                source_status == "observed"
                and (score is None or source_reason is not None)
            )
            or (
                source_status == "structural_zero"
                and (score != 0.0 or source_reason is None)
            )
            or (
                source_status == "not_estimable"
                and (score is not None or source_reason is None)
            )
        ):
            raise ValueError("component source status/value semantics changed")
        rows.append(
            {
                "sample_id": str(base["sample_id"]),
                "subject_id": str(base["subject_id"]),
                "context_id": str(base["context_id"]),
                "receiver": receiver,
                "family_id": str(base["family_id"]),
                "driver_id": str(base["driver_id"]),
                "interaction_id": str(base["interaction_id"]),
                "mode": str(base["mode"]),
                "sender_unresolved_strength": score,
                "family_core_strength": values["family_core_strength"],
                "within_family_lr_weight": values["within_family_lr_weight"],
                "source_status": source_status,
                "source_reason_code": source_reason,
            }
        )
    return tuple(rows)


def _parent_source_sha(parent: CrossFitResult) -> dict[str, str]:
    records = _mapping(parent.manifest.get("tables"), field_name="parent.tables")
    return {
        name: _required_text(
            _mapping(records.get(name), field_name=f"parent.tables.{name}").get(
                "sha256"
            ),
            field_name=f"parent.tables.{name}.sha256",
        )
        for name in _PARENT_TABLES
    }


def _replay_tables(
    parent: CrossFitResult,
    pair: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[dict[str, object], ...]]:
    parent_manifest = parent.manifest
    source = _parent_source(parent)
    fold_map = _source_fold_map(source)
    universe, opportunities_by_receiver, receiver_ids = _universe_contract(source)
    support = _support_contract(parent, source, fold_map, receiver_ids)
    directional = parent.read_directional_channel_registry()
    components = parent.read_components()
    pair_spec_id = _required_text(pair.get("pair_spec_id"), field_name="pair_spec_id")
    directional = directional.loc[
        directional["pair_spec_id"].astype(str).eq(pair_spec_id)
    ].copy()
    score_rows: list[dict[str, object]] = []
    parent_rows: list[dict[str, object]] = []
    channel_parents: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fold_id, fold in sorted(fold_map.items()):
        family_records = [
            _mapping(value, field_name="family-common parent")
            for value in _sequence(
                fold.get("family_common_scoring_artifacts"),
                field_name="family_common_scoring_artifacts",
            )
        ]
        incremental_records = [
            _mapping(value, field_name="incremental parent")
            for value in _sequence(
                fold.get("receiver_incremental_artifacts"),
                field_name="receiver_incremental_artifacts",
            )
        ]
        for receiver in receiver_ids:
            support_row = support[(fold_id, receiver)]
            support_status = str(support_row["receiver_training_support_status"])
            support_reason = _optional_text(
                support_row["receiver_training_support_reason_code"]
            )
            for role in ("forward", "reverse"):
                channel, contrast_name, _contrast, contrast_id = _role_contract(
                    pair, role
                )
                design_id, design_rows, contrast_manifest_id = _design_contract(
                    fold, pair, role
                )
                registry_rows = directional.loc[
                    directional["fold_id"].astype(str).eq(fold_id)
                    & directional["receiver"].astype(str).eq(receiver)
                    & directional["channel_role"].astype(str).eq(role)
                ]
                fitted: dict[str, object]
                if support_status == "observed":
                    if len(registry_rows) != 1:
                        raise ValueError(
                            "supported receiver lacks one channel registry row"
                        )
                    registry = cast(dict[str, Any], registry_rows.iloc[0].to_dict())
                    if (
                        registry["channel"] != channel
                        or registry["contrast"] != contrast_name
                        or registry["contrast_id"] != contrast_id
                        or registry["combination_rule"] != pair.get("combination_rule")
                        or any(bool(registry[flag]) for flag in _CLAIM_FLAGS)
                    ):
                        raise ValueError(
                            "directional channel registry changed semantics"
                        )
                    binding_id = _required_text(
                        registry["binding_id"], field_name="binding_id"
                    )
                    pair_status = str(registry["status"])
                    pair_reason = _optional_text(registry["reason_code"])
                    family_parent = _one_record(
                        [
                            row
                            for row in family_records
                            if row.get("receiver") == receiver
                            and row.get("contrast_name") == contrast_name
                        ],
                        field_name="family-common parent",
                    )
                    incremental_parent = _one_record(
                        [
                            row
                            for row in incremental_records
                            if row.get("receiver") == receiver
                            and row.get("contrast_name") == contrast_name
                        ],
                        field_name="incremental parent",
                    )
                    if (
                        incremental_parent.get("training_artifact_id")
                        != registry["training_artifact_id"]
                        or incremental_parent.get("application_id")
                        != registry["application_id"]
                        or incremental_parent.get("response_artifact_id")
                        != registry["response_id"]
                    ):
                        raise ValueError(
                            "directional registry differs from incremental parent"
                        )
                    source_rows = _component_source_rows(
                        components,
                        fold_id=fold_id,
                        receiver=receiver,
                        contrast_name=contrast_name,
                        contrast_manifest_id=contrast_manifest_id,
                        family_parent=family_parent,
                    )
                    expected_grid = {
                        (
                            sample_id,
                            subject_id,
                            context_id,
                            opportunity["family_id"],
                            opportunity["driver_id"],
                            opportunity["interaction_id"],
                            opportunity["mode"],
                        )
                        for sample_id, subject_id, context_id in design_rows
                        for opportunity in opportunities_by_receiver[receiver]
                    }
                    actual_grid = {
                        tuple(
                            str(row[column])
                            for column in (
                                "sample_id",
                                "subject_id",
                                "context_id",
                                "family_id",
                                "driver_id",
                                "interaction_id",
                                "mode",
                            )
                        )
                        for row in source_rows
                    }
                    if actual_grid != expected_grid:
                        raise ValueError(
                            "components differ from exact design-by-LR grid"
                        )
                    static = {
                        (row["interaction_id"], row["mode"]): row
                        for row in opportunities_by_receiver[receiver]
                    }
                    fitted = {
                        "directional_binding_id": binding_id,
                        "directional_status": pair_status,
                        "directional_reason_code": pair_reason,
                        "family_common_functional_id": family_parent["functional_id"],
                        "family_common_application_id": family_parent["application_id"],
                        "family_common_binding_id": family_parent["binding_id"],
                        "family_common_member_scores_digest": family_parent[
                            "member_scores_digest"
                        ],
                        "family_common_family_scores_digest": family_parent[
                            "family_scores_digest"
                        ],
                        "edge_evidence_digest": family_parent["edge_evidence_digest"],
                        "incremental_training_artifact_id": registry[
                            "training_artifact_id"
                        ],
                        "incremental_application_id": registry["application_id"],
                        "response_id": registry["response_id"],
                    }
                    for source_row in source_rows:
                        opportunity = static[
                            (
                                str(source_row["interaction_id"]),
                                str(source_row["mode"]),
                            )
                        ]
                        source_status = str(source_row["source_status"])
                        source_reason = _optional_text(source_row["source_reason_code"])
                        if pair_status == "observed":
                            status = source_status
                            reason = source_reason
                            integrated_score = source_row["sender_unresolved_strength"]
                        else:
                            status = "not_estimable"
                            reason = pair_reason or _PAIR_NE_REASON
                            integrated_score = None
                        score_rows.append(
                            {
                                "crossfit_id": parent_manifest["crossfit_id"],
                                "crossfit_spec_id": parent_manifest["spec_id"],
                                "pair_spec_id": pair_spec_id,
                                "directional_lr_hypothesis_universe_id": universe[
                                    "universe_id"
                                ],
                                "fold_id": fold_id,
                                "receiver": receiver,
                                "receiver_training_support_id": support_row[
                                    "receiver_training_support_id"
                                ],
                                "receiver_training_support_status": support_status,
                                "receiver_training_support_reason_code": support_reason,
                                "directional_binding_id": binding_id,
                                "channel_role": role,
                                "channel": channel,
                                "contrast_manifest_id": contrast_manifest_id,
                                "contrast_name": contrast_name,
                                "design_application_id": design_id,
                                "sample_id": source_row["sample_id"],
                                "subject_id": source_row["subject_id"],
                                "context_id": source_row["context_id"],
                                "family_id": source_row["family_id"],
                                "driver_id": source_row["driver_id"],
                                "interaction_id": source_row["interaction_id"],
                                "molecular_lr_equivalence_id": opportunity[
                                    "molecular_lr_equivalence_id"
                                ],
                                "mode": source_row["mode"],
                                **{
                                    key: opportunity[key]
                                    for key in (
                                        "receiver_family_lr_membership_id",
                                        "receiver_family_lr_hypothesis_id",
                                        "receiver_family_lr_opportunity_id",
                                    )
                                },
                                "integrated_lr_score": integrated_score,
                                "family_core_strength": source_row[
                                    "family_core_strength"
                                ],
                                "within_family_lr_weight": source_row[
                                    "within_family_lr_weight"
                                ],
                                "source_status": source_status,
                                "source_reason_code": source_reason,
                                "status": status,
                                "reason_code": reason,
                                **{
                                    key: fitted[key]
                                    for key in _FITTED_SCORE_PARENTS
                                    if key != "directional_binding_id"
                                },
                                "score_version": (
                                    DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION
                                ),
                                "semantics": DIRECTIONAL_INTEGRATED_LR_SEMANTICS,
                                "analysis_track": (
                                    DIRECTIONAL_INTEGRATED_LR_ANALYSIS_TRACK
                                ),
                                "experimental": True,
                                "source_agnostic": False,
                                "cross_channel_comparable": False,
                                "paired_score_comparison_allowed": False,
                                "active_inhibition_allowed": False,
                                "supports_active_inhibition_claim": False,
                                "formal_inference_allowed": False,
                            }
                        )
                elif (
                    support_status == "not_estimable"
                    and support_reason == _ABSENT_RECEIVER_REASON
                ):
                    if not registry_rows.empty:
                        raise ValueError("training-absent receiver has channel parents")
                    fitted = {
                        "directional_binding_id": None,
                        "directional_status": None,
                        "directional_reason_code": None,
                        "family_common_functional_id": None,
                        "family_common_application_id": None,
                        "family_common_binding_id": None,
                        "family_common_member_scores_digest": None,
                        "family_common_family_scores_digest": None,
                        "edge_evidence_digest": None,
                        "incremental_training_artifact_id": None,
                        "incremental_application_id": None,
                        "response_id": None,
                    }
                    for sample_id, subject_id, context_id in design_rows:
                        for opportunity in opportunities_by_receiver[receiver]:
                            score_rows.append(
                                {
                                    "crossfit_id": parent_manifest["crossfit_id"],
                                    "crossfit_spec_id": parent_manifest["spec_id"],
                                    "pair_spec_id": pair_spec_id,
                                    "directional_lr_hypothesis_universe_id": universe[
                                        "universe_id"
                                    ],
                                    "fold_id": fold_id,
                                    "receiver": receiver,
                                    "receiver_training_support_id": support_row[
                                        "receiver_training_support_id"
                                    ],
                                    "receiver_training_support_status": support_status,
                                    "receiver_training_support_reason_code": (
                                        support_reason
                                    ),
                                    "directional_binding_id": None,
                                    "channel_role": role,
                                    "channel": channel,
                                    "contrast_manifest_id": contrast_manifest_id,
                                    "contrast_name": contrast_name,
                                    "design_application_id": design_id,
                                    "sample_id": sample_id,
                                    "subject_id": subject_id,
                                    "context_id": context_id,
                                    "family_id": opportunity["family_id"],
                                    "driver_id": opportunity["driver_id"],
                                    "interaction_id": opportunity["interaction_id"],
                                    "molecular_lr_equivalence_id": opportunity[
                                        "molecular_lr_equivalence_id"
                                    ],
                                    "mode": opportunity["mode"],
                                    **{
                                        key: opportunity[key]
                                        for key in (
                                            "receiver_family_lr_membership_id",
                                            "receiver_family_lr_hypothesis_id",
                                            "receiver_family_lr_opportunity_id",
                                        )
                                    },
                                    "integrated_lr_score": None,
                                    "family_core_strength": None,
                                    "within_family_lr_weight": None,
                                    "source_status": None,
                                    "source_reason_code": None,
                                    "status": "not_estimable",
                                    "reason_code": _ABSENT_RECEIVER_REASON,
                                    **{
                                        key: None
                                        for key in _FITTED_SCORE_PARENTS
                                        if key != "directional_binding_id"
                                    },
                                    "score_version": (
                                        DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION
                                    ),
                                    "semantics": DIRECTIONAL_INTEGRATED_LR_SEMANTICS,
                                    "analysis_track": (
                                        DIRECTIONAL_INTEGRATED_LR_ANALYSIS_TRACK
                                    ),
                                    "experimental": True,
                                    "source_agnostic": False,
                                    "cross_channel_comparable": False,
                                    "paired_score_comparison_allowed": False,
                                    "active_inhibition_allowed": False,
                                    "supports_active_inhibition_claim": False,
                                    "formal_inference_allowed": False,
                                }
                            )
                else:
                    raise ValueError("receiver training support status is unsupported")
                channel_parents[(fold_id, receiver, role)] = {
                    **fitted,
                    "design_application_id": design_id,
                    "contrast_name": contrast_name,
                    "contrast_manifest_id": contrast_manifest_id,
                    "channel": channel,
                }
                parent_rows.append(
                    {
                        "fold_id": fold_id,
                        "receiver": receiver,
                        "receiver_training_support_id": support_row[
                            "receiver_training_support_id"
                        ],
                        "receiver_training_support_status": support_status,
                        "receiver_training_support_reason_code": support_reason,
                        "directional_lr_hypothesis_universe_id": universe[
                            "universe_id"
                        ],
                        "receiver_family_lr_opportunity_axis_id": universe[
                            "opportunity_axis_id"
                        ],
                        "design_application_id": design_id,
                        "channel_role": role,
                        "channel": channel,
                        "contrast_name": contrast_name,
                        "contrast_manifest_id": contrast_manifest_id,
                        **{
                            key: fitted[key]
                            for key in (
                                "directional_binding_id",
                                "directional_status",
                                "directional_reason_code",
                                "family_common_functional_id",
                                "family_common_application_id",
                                "family_common_binding_id",
                                "family_common_member_scores_digest",
                                "family_common_family_scores_digest",
                                "edge_evidence_digest",
                                "incremental_training_artifact_id",
                                "incremental_application_id",
                            )
                        },
                    }
                )
    scores = pd.DataFrame(score_rows, columns=DIRECTIONAL_INTEGRATED_LR_COLUMNS)
    scores = scores.sort_values(
        list(_SCORE_PRIMARY_KEY), kind="stable", ignore_index=True
    )
    if scores.empty or scores.duplicated(list(_SCORE_PRIMARY_KEY)).any():
        raise ValueError("replayed directional score grid is empty or duplicated")
    opportunity_rows: list[dict[str, object]] = []
    counts = (
        scores.groupby(["fold_id", "receiver", "channel_role"], sort=False)
        .size()
        .to_dict()
    )
    for fold_id in sorted(fold_map):
        for receiver in receiver_ids:
            support_row = support[(fold_id, receiver)]
            support_status = str(support_row["receiver_training_support_status"])
            forward = channel_parents[(fold_id, receiver, "forward")]
            reverse = channel_parents[(fold_id, receiver, "reverse")]
            opportunity_binding_id: object
            if support_status == "observed":
                if (
                    forward["directional_binding_id"]
                    != reverse["directional_binding_id"]
                    or forward["directional_status"] != reverse["directional_status"]
                    or forward["directional_reason_code"]
                    != reverse["directional_reason_code"]
                ):
                    raise ValueError(
                        "directional channel roles disagree on pair status"
                    )
                status = forward["directional_status"]
                reason = forward["directional_reason_code"]
                opportunity_binding_id = forward["directional_binding_id"]
                parent_values = {
                    "forward_response_id": forward["response_id"],
                    "reverse_response_id": reverse["response_id"],
                    "forward_incremental_training_artifact_id": forward[
                        "incremental_training_artifact_id"
                    ],
                    "reverse_incremental_training_artifact_id": reverse[
                        "incremental_training_artifact_id"
                    ],
                    "forward_incremental_application_id": forward[
                        "incremental_application_id"
                    ],
                    "reverse_incremental_application_id": reverse[
                        "incremental_application_id"
                    ],
                    **{
                        f"{role}_family_common_{suffix}_id": channel_parents[
                            (fold_id, receiver, role)
                        ][f"family_common_{suffix}_id"]
                        for role in ("forward", "reverse")
                        for suffix in ("functional", "application", "binding")
                    },
                }
            else:
                status = "not_estimable"
                reason = _ABSENT_RECEIVER_REASON
                opportunity_binding_id = None
                parent_values = {column: None for column in _OPPORTUNITY_PARENT_COLUMNS}
            forward_count = int(counts.get((fold_id, receiver, "forward"), 0))
            reverse_count = int(counts.get((fold_id, receiver, "reverse"), 0))
            if forward_count <= 0 or forward_count != reverse_count:
                raise ValueError("directional opportunity lacks a symmetric score grid")
            opportunity_rows.append(
                {
                    "crossfit_id": parent_manifest["crossfit_id"],
                    "crossfit_spec_id": parent_manifest["spec_id"],
                    "pair_spec_id": pair_spec_id,
                    "directional_lr_hypothesis_universe_id": universe["universe_id"],
                    "receiver_family_lr_opportunity_axis_id": universe[
                        "opportunity_axis_id"
                    ],
                    "receiver_universe_id": parent_manifest["receiver_universe_id"],
                    "receiver_axis_id": parent_manifest["receiver_axis_id"],
                    "fold_id": fold_id,
                    "receiver": receiver,
                    "receiver_training_support_id": support_row[
                        "receiver_training_support_id"
                    ],
                    "receiver_training_support_status": support_status,
                    "receiver_training_support_reason_code": _optional_text(
                        support_row["receiver_training_support_reason_code"]
                    ),
                    "directional_binding_id": opportunity_binding_id,
                    **parent_values,
                    "forward_channel": pair["forward_channel_name"],
                    "reverse_channel": pair["reverse_channel_name"],
                    "status": status,
                    "reason_code": reason,
                    "lr_hypothesis_grid_status": "produced",
                    "lr_hypothesis_grid_reason_code": None,
                    "n_forward_score_rows": forward_count,
                    "n_reverse_score_rows": reverse_count,
                    "paired_score_comparison_allowed": False,
                    "active_inhibition_allowed": False,
                    "supports_active_inhibition_claim": False,
                    "formal_inference_allowed": False,
                }
            )
    opportunities = pd.DataFrame(
        opportunity_rows, columns=DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_COLUMNS
    ).sort_values(list(_OPPORTUNITY_PRIMARY_KEY), kind="stable", ignore_index=True)
    return (
        scores,
        opportunities,
        tuple(
            sorted(
                parent_rows,
                key=lambda row: (
                    str(row["fold_id"]),
                    str(row["receiver"]),
                    str(row["channel_role"]),
                ),
            )
        ),
    )


def _expected_collection_manifest(
    parent: CrossFitResult,
    pair: Mapping[str, Any],
    scores: pd.DataFrame,
    opportunities: pd.DataFrame,
    parent_bindings: tuple[dict[str, object], ...],
) -> dict[str, object]:
    source = _parent_source(parent)
    universe, _opportunities, _receivers = _universe_contract(source)
    parent_manifest = parent.manifest
    status_counts = _status_counts(scores)
    opportunity_status_counts = _opportunity_status_counts(opportunities)
    identity: dict[str, object] = {
        "active_inhibition_allowed": False,
        "channels": [pair["forward_channel_name"], pair["reverse_channel_name"]],
        "cross_channel_comparable": False,
        "crossfit_id": parent_manifest["crossfit_id"],
        "crossfit_spec_id": parent_manifest["spec_id"],
        "directional_lr_hypothesis_universe_id": universe["universe_id"],
        "molecular_lr_equivalence_universe_id": universe[
            "molecular_lr_equivalence_universe_id"
        ],
        "molecular_lr_axis_id": universe["molecular_lr_axis_id"],
        "experimental": True,
        "formal_inference_allowed": False,
        "opportunity_count": len(opportunities),
        "opportunity_registry_digest": _table_digest(opportunities),
        "opportunity_status_counts": [list(item) for item in opportunity_status_counts],
        "pair_spec_id": pair["pair_spec_id"],
        "parent_bindings": list(parent_bindings),
        "paired_score_comparison_allowed": False,
        "receiver_axis_id": parent_manifest["receiver_axis_id"],
        "receiver_family_lr_opportunity_axis_id": universe["opportunity_axis_id"],
        "receiver_universe_id": parent_manifest["receiver_universe_id"],
        "row_count": len(scores),
        "score_grid_complete_on_run_receiver_axis": True,
        "source_agnostic": False,
        "status_counts": [list(item) for item in status_counts],
        "supports_active_inhibition_claim": False,
        "table_digest": _table_digest(scores),
        "score_version": DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION,
        "semantics": DIRECTIONAL_INTEGRATED_LR_SEMANTICS,
    }
    collection_id = stable_id(
        "directional_integrated_lr_collection", identity, schema_version="4"
    )
    return {
        "collection_id": collection_id,
        "schema_version": DIRECTIONAL_INTEGRATED_LR_COLLECTION_SCHEMA_VERSION,
        "producer": _COLLECTION_PRODUCER,
        **identity,
        "grain": (
            "outer_fold_x_pair_x_channel_x_sample_x_receiver_x_family_x_lr_x_mode"
        ),
        "opportunity_registry_grain": "outer_fold_x_pair_x_run_receiver",
        "opportunity_registry_complete": True,
        "score_grid_scope": (
            "full_run_root_frozen_receiver_family_lr_hypothesis_universe"
        ),
        "status": "complete_descriptive",
        "comparability_scope": "within_channel_only",
        "combination_rule": "independent_contrast_views_not_additive_v1",
        "p_value": None,
        "q_value": None,
        "comm_probability": None,
    }


def _artifact_identity(manifest: Mapping[str, Any]) -> str:
    return str(
        stable_id(
            "directional_integrated_lr_result",
            {key: value for key, value in manifest.items() if key != "artifact_id"},
            schema_version="2",
        )
    )


def _load_table(root: Path, name: str, record: Mapping[str, Any]) -> pd.DataFrame:
    expected_record_keys = {
        "filename",
        "schema_version",
        "columns",
        "primary_key",
        "rows",
        "sha256",
        "logical_digest",
    }
    if (
        set(record) != expected_record_keys
        or record.get("filename") != _TABLE_FILENAMES[name]
        or record.get("schema_version") != _TABLE_SCHEMA_VERSION
        or record.get("columns") != list(_TABLE_COLUMNS[name])
        or record.get("primary_key") != list(_TABLE_PRIMARY_KEYS[name])
        or type(record.get("rows")) is not int
        or cast(int, record["rows"]) <= 0
    ):
        raise ValueError(f"{name} table manifest violates its exact schema")
    path = root / _TABLE_FILENAMES[name]
    if _sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{name} parquet SHA-256 changed")
    table = cast(
        pd.DataFrame,
        pd.read_parquet(path, engine="pyarrow"),  # type: ignore[call-overload]
    )
    if (
        tuple(table.columns) != _TABLE_COLUMNS[name]
        or len(table) != record["rows"]
        or table.empty
        or table.duplicated(list(_TABLE_PRIMARY_KEYS[name])).any()
    ):
        raise ValueError(f"{name} table columns, rows, or primary key changed")
    keys = [
        tuple(
            _required_text(value, field_name=column)
            for column, value in zip(_TABLE_PRIMARY_KEYS[name], row, strict=True)
        )
        for row in table.loc[:, list(_TABLE_PRIMARY_KEYS[name])].itertuples(
            index=False, name=None
        )
    ]
    if keys != sorted(keys):
        raise ValueError(f"{name} table is not in canonical stable order")
    if _table_digest(table) != record.get("logical_digest"):
        raise ValueError(f"{name} logical digest changed")
    return table.copy(deep=True)


def _validate_manifest_structure(
    manifest: Mapping[str, Any], parent: CrossFitResult
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected_keys = {
        "artifact_schema_version",
        "artifact_kind",
        "artifact_id",
        "status",
        "producer",
        "collection_id",
        "collection_manifest_schema_version",
        "collection_manifest",
        "source_crossfit_result_id",
        "source_crossfit_result_schema_version",
        "source_crossfit_id",
        "source_crossfit_spec_id",
        "source_crossfit_manifest_digest",
        "source_table_sha256",
        "pair_spec_id",
        "pair_spec",
        "directional_lr_hypothesis_universe_id",
        "molecular_lr_equivalence_universe_id",
        "molecular_lr_axis_id",
        "receiver_family_lr_opportunity_axis_id",
        "receiver_universe_id",
        "receiver_axis_id",
        "tables",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("artifact_schema_version")
        != DIRECTIONAL_INTEGRATED_LR_RESULT_SCHEMA_VERSION
        or manifest.get("artifact_kind") != _ARTIFACT_KIND
        or manifest.get("status") != _COMPLETE
        or manifest.get("producer") != _PRODUCER
        or manifest.get("collection_manifest_schema_version")
        != DIRECTIONAL_INTEGRATED_LR_COLLECTION_SCHEMA_VERSION
        or manifest.get("source_crossfit_result_schema_version")
        != CROSSFIT_RESULT_SCHEMA_VERSION
        or manifest.get("artifact_id") != _artifact_identity(manifest)
    ):
        raise ValueError("directional integrated-LR manifest violates its schema")
    parent_manifest = parent.manifest
    source = _parent_source(parent)
    pair_spec_id = _required_text(
        manifest.get("pair_spec_id"), field_name="pair_spec_id"
    )
    pair = _source_pair(source, pair_spec_id)
    universe, _grid, _receivers = _universe_contract(source)
    if (
        manifest.get("source_crossfit_result_id")
        != parent_manifest.get("crossfit_result_id")
        or manifest.get("source_crossfit_id") != parent_manifest.get("crossfit_id")
        or manifest.get("source_crossfit_spec_id") != parent_manifest.get("spec_id")
        or manifest.get("source_crossfit_manifest_digest")
        != parent_manifest.get("source_crossfit_manifest_digest")
        or manifest.get("source_table_sha256") != _parent_source_sha(parent)
        or manifest.get("pair_spec") != pair
        or manifest.get("directional_lr_hypothesis_universe_id")
        != universe.get("universe_id")
        or manifest.get("molecular_lr_equivalence_universe_id")
        != universe.get("molecular_lr_equivalence_universe_id")
        or manifest.get("molecular_lr_axis_id") != universe.get("molecular_lr_axis_id")
        or manifest.get("receiver_family_lr_opportunity_axis_id")
        != universe.get("opportunity_axis_id")
        or manifest.get("receiver_universe_id")
        != parent_manifest.get("receiver_universe_id")
        or manifest.get("receiver_axis_id") != parent_manifest.get("receiver_axis_id")
    ):
        raise ValueError("sidecar differs from its exact parent lineage")
    collection_manifest = _mapping(
        manifest.get("collection_manifest"), field_name="collection_manifest"
    )
    if manifest.get("collection_id") != collection_manifest.get("collection_id"):
        raise ValueError("sidecar collection identity changed")
    tables_raw = _mapping(manifest.get("tables"), field_name="tables")
    if set(tables_raw) != set(_TABLE_FILENAMES):
        raise ValueError("sidecar table registry changed")
    tables = {
        name: _mapping(tables_raw[name], field_name=f"tables.{name}")
        for name in _TABLE_FILENAMES
    }
    return pair, tables


def _validate_bundle(
    root: Path,
    parent: CrossFitResult,
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair, records = _validate_manifest_structure(manifest, parent)
    scores = _load_table(
        root,
        DIRECTIONAL_INTEGRATED_LR_SCORE_TABLE,
        records[DIRECTIONAL_INTEGRATED_LR_SCORE_TABLE],
    )
    opportunities = _load_table(
        root,
        DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_TABLE,
        records[DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_TABLE],
    )
    expected_scores, expected_opportunities, parent_bindings = _replay_tables(
        parent, pair
    )
    if _table_digest(scores) != _table_digest(expected_scores) or _table_digest(
        opportunities
    ) != _table_digest(expected_opportunities):
        raise ValueError("sidecar values differ from exact parent semantic replay")
    expected_collection = _expected_collection_manifest(
        parent,
        pair,
        expected_scores,
        expected_opportunities,
        parent_bindings,
    )
    if manifest.get("collection_manifest") != expected_collection:
        raise ValueError("embedded collection v3 manifest changed")
    if manifest.get("collection_id") != expected_collection["collection_id"]:
        raise ValueError("embedded collection identity changed")
    return scores, opportunities


@dataclass(frozen=True, slots=True, init=False)
class DirectionalIntegratedLRResult:
    """Validated read-only facade over one directional integrated-LR sidecar."""

    path: Path
    source_crossfit_result_path: Path
    _manifest: dict[str, object] = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "DirectionalIntegratedLRResult is producer-owned; use load() or the writer"
        )

    @classmethod
    def _from_validated(
        cls,
        path: Path,
        source_crossfit_result_path: Path,
        manifest: Mapping[str, Any],
    ) -> DirectionalIntegratedLRResult:
        self = object.__new__(cls)
        object.__setattr__(self, "path", path.resolve())
        object.__setattr__(
            self,
            "source_crossfit_result_path",
            source_crossfit_result_path.resolve(),
        )
        object.__setattr__(self, "_manifest", copy.deepcopy(dict(manifest)))
        return self

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        source_crossfit_result: CrossFitResult | str | Path,
    ) -> DirectionalIntegratedLRResult:
        """Load a sidecar only against its explicitly supplied exact v7 parent."""

        root = Path(path)
        parent = _exact_parent(source_crossfit_result)
        try:
            marker = _read_json(root / _STATUS_FILENAME)
        except Exception as error:
            raise ResultValidationError(
                "Directional integrated-LR status marker is missing or corrupted",
                code="invalid_directional_integrated_lr_status",
                field="path",
                remediation="Reject the sidecar and regenerate it",
            ) from error
        if marker.get("status") != _COMPLETE:
            raise IncompleteResultError(
                "Directional integrated-LR result is incomplete",
                code="incomplete_directional_integrated_lr_result",
                field="path",
                remediation="Inspect producer diagnostics and write to a new directory",
            )
        try:
            manifest = _read_json(root / _MANIFEST_FILENAME)
            if (
                set(marker)
                != {
                    "artifact_schema_version",
                    "artifact_kind",
                    "status",
                    "artifact_id",
                }
                or marker.get("artifact_schema_version")
                != DIRECTIONAL_INTEGRATED_LR_RESULT_SCHEMA_VERSION
                or marker.get("artifact_kind") != _ARTIFACT_KIND
                or marker.get("artifact_id") != manifest.get("artifact_id")
            ):
                raise ValueError("status marker differs from its manifest")
            _validate_bundle(root, parent, manifest)
        except (IncompleteResultError, ResultValidationError):
            raise
        except Exception as error:
            raise ResultValidationError(
                "Directional integrated-LR sidecar failed integrity validation",
                code="invalid_directional_integrated_lr_result",
                field="path",
                remediation=(
                    "Reject the sidecar and regenerate it from the exact parent"
                ),
            ) from error
        return cls._from_validated(root, parent.path, manifest)

    @property
    def manifest(self) -> dict[str, object]:
        """Return a defensive copy of the validated artifact manifest."""

        return copy.deepcopy(self._manifest)

    def _read_validated(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            parent = _exact_parent(self.source_crossfit_result_path)
            current = _read_json(self.path / _MANIFEST_FILENAME)
            if current != self._manifest:
                raise ResultValidationError(
                    "Directional integrated-LR manifest changed after load",
                    code="directional_integrated_lr_manifest_changed",
                    field="path",
                    remediation="Discard the facade and reject the mutated sidecar",
                )
            marker = _read_json(self.path / _STATUS_FILENAME)
            if marker != {
                "artifact_schema_version": (
                    DIRECTIONAL_INTEGRATED_LR_RESULT_SCHEMA_VERSION
                ),
                "artifact_kind": _ARTIFACT_KIND,
                "status": _COMPLETE,
                "artifact_id": current.get("artifact_id"),
            }:
                raise ValueError("sidecar status marker changed after load")
            return _validate_bundle(self.path, parent, current)
        except ResultValidationError:
            raise
        except Exception as error:
            raise ResultValidationError(
                "Directional integrated-LR sidecar changed after load",
                code="directional_integrated_lr_result_changed",
                field="path",
                remediation="Reject the mutated sidecar",
            ) from error

    def read_scores(self) -> pd.DataFrame:
        """Return exact independent channel rows after defensive revalidation."""

        scores, _opportunities = self._read_validated()
        return scores.copy(deep=True)

    def read_opportunities(self) -> pd.DataFrame:
        """Return the complete fold-by-receiver opportunity registry."""

        _scores, opportunities = self._read_validated()
        return opportunities.copy(deep=True)

    def query_scores(
        self,
        *,
        fold_id: str | None = None,
        receiver: str | None = None,
        channel_role: str | None = None,
        channel: str | None = None,
        contrast_name: str | None = None,
        sample_id: str | None = None,
        subject_id: str | None = None,
        context_id: str | None = None,
        family_id: str | None = None,
        driver_id: str | None = None,
        interaction_id: str | None = None,
        molecular_lr_equivalence_id: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Query one or both independent views without combining their values."""

        filters = {
            "fold_id": fold_id,
            "receiver": receiver,
            "channel_role": channel_role,
            "channel": channel,
            "contrast_name": contrast_name,
            "sample_id": sample_id,
            "subject_id": subject_id,
            "context_id": context_id,
            "family_id": family_id,
            "driver_id": driver_id,
            "interaction_id": interaction_id,
            "molecular_lr_equivalence_id": molecular_lr_equivalence_id,
            "mode": mode,
            "status": status,
        }
        _validate_filters(filters)
        if channel_role is not None and channel_role not in {"forward", "reverse"}:
            raise ValueError("channel_role must be forward, reverse, or None")
        if status is not None and status not in {
            "observed",
            "structural_zero",
            "not_estimable",
        }:
            raise ValueError("status is not a directional integrated-LR status")
        frame = self.read_scores()
        selected = pd.Series(True, index=frame.index, dtype=bool)
        for column, value in filters.items():
            if value is not None:
                selected &= frame[column].astype(str).eq(value)
        return (
            frame.loc[selected]
            .sort_values(list(_SCORE_PRIMARY_KEY), kind="stable")
            .reset_index(drop=True)
            .copy(deep=True)
        )

    def query_opportunities(
        self,
        *,
        fold_id: str | None = None,
        receiver: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Query persisted receiver opportunities with stable ordering."""

        filters = {"fold_id": fold_id, "receiver": receiver, "status": status}
        _validate_filters(filters)
        if status is not None and status not in {"observed", "not_estimable"}:
            raise ValueError("status must be observed, not_estimable, or None")
        frame = self.read_opportunities()
        selected = pd.Series(True, index=frame.index, dtype=bool)
        for column, value in filters.items():
            if value is not None:
                selected &= frame[column].astype(str).eq(value)
        return (
            frame.loc[selected]
            .sort_values(list(_OPPORTUNITY_PRIMARY_KEY), kind="stable")
            .reset_index(drop=True)
            .copy(deep=True)
        )


def _validate_filters(filters: Mapping[str, str | None]) -> None:
    for field_name, value in filters.items():
        if value is not None:
            _required_text(value, field_name=field_name)


def write_directional_integrated_lr_result(
    collection: DirectionalIntegratedLRCollection,
    destination: str | Path,
    *,
    source_crossfit_result: CrossFitResult,
) -> DirectionalIntegratedLRResult:
    """Atomically persist one producer-owned collection beside its exact v7 parent."""

    if type(collection) is not DirectionalIntegratedLRCollection:
        raise TypeError(
            "collection must be producer-owned DirectionalIntegratedLRCollection"
        )
    if type(source_crossfit_result) is not CrossFitResult:
        raise TypeError("source_crossfit_result must be an exact CrossFitResult")
    output = Path(destination)
    if output.exists():
        raise ResultWriteError(
            f"Directional integrated-LR destination {output.name!r} already exists",
            code="directional_integrated_lr_destination_exists",
            field="destination",
            remediation="Choose a new versioned result directory",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    _mark_incomplete(temporary)
    try:
        collection._require_intact()
        parent = _exact_parent(source_crossfit_result)
        parent_manifest = parent.manifest
        source = _parent_source(parent)
        producer_source = collection._source_artifacts.to_manifest()
        pair = collection._pair_spec.to_dict()
        if (
            producer_source != source
            or collection.crossfit_id != parent_manifest.get("crossfit_id")
            or collection.crossfit_spec_id != parent_manifest.get("spec_id")
            or pair != _source_pair(source, collection.pair_spec_id)
        ):
            raise ValueError("collection does not belong to the exact v7 parent")
        scores = collection.scores.sort_values(
            list(_SCORE_PRIMARY_KEY), kind="stable", ignore_index=True
        )
        opportunities = collection.opportunity_registry.sort_values(
            list(_OPPORTUNITY_PRIMARY_KEY), kind="stable", ignore_index=True
        )
        tables = {
            DIRECTIONAL_INTEGRATED_LR_SCORE_TABLE: scores,
            DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_TABLE: opportunities,
        }
        table_records: dict[str, dict[str, object]] = {}
        for name, table in tables.items():
            if tuple(table.columns) != _TABLE_COLUMNS[name]:
                raise ValueError(f"{name} columns changed before persistence")
            path = temporary / _TABLE_FILENAMES[name]
            table.to_parquet(
                path,
                index=False,
                engine="pyarrow",
                compression="zstd",
                row_group_size=131_072,
            )
            table_records[name] = {
                "filename": _TABLE_FILENAMES[name],
                "schema_version": _TABLE_SCHEMA_VERSION,
                "columns": list(_TABLE_COLUMNS[name]),
                "primary_key": list(_TABLE_PRIMARY_KEYS[name]),
                "rows": len(table),
                "sha256": _sha256_file(path),
                "logical_digest": _table_digest(table),
            }
        universe, _grid, _receivers = _universe_contract(source)
        collection_manifest = collection.to_manifest()
        manifest: dict[str, object] = {
            "artifact_schema_version": (
                DIRECTIONAL_INTEGRATED_LR_RESULT_SCHEMA_VERSION
            ),
            "artifact_kind": _ARTIFACT_KIND,
            "status": _COMPLETE,
            "producer": _PRODUCER,
            "collection_id": collection.collection_id,
            "collection_manifest_schema_version": (
                DIRECTIONAL_INTEGRATED_LR_COLLECTION_SCHEMA_VERSION
            ),
            "collection_manifest": collection_manifest,
            "source_crossfit_result_id": parent_manifest["crossfit_result_id"],
            "source_crossfit_result_schema_version": (CROSSFIT_RESULT_SCHEMA_VERSION),
            "source_crossfit_id": parent_manifest["crossfit_id"],
            "source_crossfit_spec_id": parent_manifest["spec_id"],
            "source_crossfit_manifest_digest": parent_manifest[
                "source_crossfit_manifest_digest"
            ],
            "source_table_sha256": _parent_source_sha(parent),
            "pair_spec_id": collection.pair_spec_id,
            "pair_spec": pair,
            "directional_lr_hypothesis_universe_id": universe["universe_id"],
            "molecular_lr_equivalence_universe_id": universe[
                "molecular_lr_equivalence_universe_id"
            ],
            "molecular_lr_axis_id": universe["molecular_lr_axis_id"],
            "receiver_family_lr_opportunity_axis_id": universe["opportunity_axis_id"],
            "receiver_universe_id": parent_manifest["receiver_universe_id"],
            "receiver_axis_id": parent_manifest["receiver_axis_id"],
            "tables": table_records,
        }
        manifest["artifact_id"] = _artifact_identity(cast(dict[str, Any], manifest))
        _write_json(temporary / _MANIFEST_FILENAME, manifest)
        _validate_bundle(temporary, parent, cast(dict[str, Any], manifest))
        _write_json(
            temporary / _STATUS_FILENAME,
            {
                "artifact_schema_version": (
                    DIRECTIONAL_INTEGRATED_LR_RESULT_SCHEMA_VERSION
                ),
                "artifact_kind": _ARTIFACT_KIND,
                "status": _COMPLETE,
                "artifact_id": manifest["artifact_id"],
            },
        )
        os.replace(temporary, output)
    except Exception as error:
        if temporary.exists():
            _mark_incomplete(temporary, type(error).__name__)
            if not output.exists():
                os.replace(temporary, output)
        raise ResultWriteError(
            "Directional integrated-LR write for "
            f"{output.name!r} failed and was marked incomplete",
            code="directional_integrated_lr_write_failed",
            field="destination",
            remediation="Inspect producer diagnostics and write to a new directory",
        ) from error
    return DirectionalIntegratedLRResult.load(
        output, source_crossfit_result=source_crossfit_result
    )


__all__ = [
    "DIRECTIONAL_INTEGRATED_LR_COLLECTION_SCHEMA_VERSION",
    "DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_TABLE",
    "DIRECTIONAL_INTEGRATED_LR_RESULT_SCHEMA_VERSION",
    "DIRECTIONAL_INTEGRATED_LR_SCORE_TABLE",
    "DirectionalIntegratedLRResult",
    "write_directional_integrated_lr_result",
]
