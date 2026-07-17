"""Post-fit directed heterogeneous communication hypergraphs.

This module is deliberately a pure consumer of fitted records.  It neither
imports fitting code nor exposes quantities that are eligible for inference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TypeVar, cast

import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id

HYPERGRAPH_SCHEMA_VERSION = "0.1.0"
HYPERGRAPH_ID_SCHEMA_VERSION = "1"


class HyperedgeStatus(StrEnum):
    """Availability state of a fitted communication record."""

    OBSERVED = "observed"
    STRUCTURAL_ZERO = "structural_zero"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"
    FILTERED = "filtered"


class TargetKind(StrEnum):
    """Biological type represented by the terminal target node."""

    GENE = "gene"
    TF = "tf"
    PROGRAM = "program"


class NodeType(StrEnum):
    """Explicit node types in the communication hypergraph."""

    CONTEXT = "context"
    SENDER = "sender"
    LIGAND = "ligand"
    LIGAND_COMPLEX = "ligand_complex"
    RECEPTOR = "receptor"
    RECEPTOR_COMPLEX = "receptor_complex"
    RECEIVER = "receiver"
    TARGET_GENE = "target_gene"
    TARGET_TF = "target_tf"
    TARGET_PROGRAM = "target_program"


EDGE_RECORD_COLUMNS = (
    "source_record_id",
    "context_id",
    "context_json",
    "sender",
    "ligand",
    "ligand_members",
    "receptor",
    "receptor_members",
    "receiver",
    "target",
    "target_kind",
    "interaction_id",
    "mode",
    "weight",
    "weight_semantics",
    "uncertainty",
    "uncertainty_kind",
    "status",
    "reason_code",
    "source_artifact_id",
    "scoring_functional_id",
    "fold_id",
    "repeat_id",
    "components_json",
    "provenance_json",
)

NODE_COLUMNS = (
    "node_id",
    "node_type",
    "entity_id",
    "label",
    "metadata_json",
)

HYPEREDGE_COLUMNS = (
    "hyperedge_id",
    "source_record_id",
    "context_node_id",
    "sender_node_id",
    "ligand_node_id",
    "receptor_node_id",
    "receiver_node_id",
    "target_node_id",
    "direction",
    "context_id",
    "context_json",
    "sender",
    "ligand",
    "receptor",
    "receiver",
    "target",
    "target_kind",
    "interaction_id",
    "mode",
    "weight",
    "weight_semantics",
    "uncertainty",
    "uncertainty_kind",
    "status",
    "reason_code",
    "source_artifact_id",
    "scoring_functional_id",
    "fold_id",
    "repeat_id",
    "components_json",
    "provenance_json",
)

COMPLEX_MEMBER_COLUMNS = (
    "membership_id",
    "complex_node_id",
    "member_node_id",
    "complex_role",
    "member_index",
)

DEGREE_COLUMNS = (
    "node_id",
    "node_type",
    "label",
    "in_degree",
    "out_degree",
    "conditioning_degree",
    "total_degree",
    "in_strength",
    "out_strength",
    "conditioning_strength",
    "total_strength",
    "edge_status_policy",
    "direction_policy",
    "weight_policy",
)

COMPONENT_COLUMNS = (
    "component_id",
    "node_id",
    "component_size",
    "edge_status_policy",
    "connectivity_policy",
)

_DIRECTION = "sender_ligand_to_receptor_receiver_target"
_UNAVAILABLE_STATUSES = {
    HyperedgeStatus.NOT_ESTIMABLE,
    HyperedgeStatus.FAILED,
    HyperedgeStatus.FILTERED,
}
_DEFAULT_TOPOLOGY_STATUSES = (
    HyperedgeStatus.OBSERVED,
    HyperedgeStatus.STRUCTURAL_ZERO,
)

EnumT = TypeVar("EnumT", bound=StrEnum)


def _contract_error(
    message: str,
    *,
    code: str,
    field: str | None = None,
    remediation: str | None = None,
) -> ContractError:
    return ContractError(
        message,
        code=code,
        field=field,
        remediation=remediation,
    )


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _contract_error(
            f"{field} must be a non-empty string",
            code="invalid_hypergraph_record",
            field=field,
            remediation="Provide an explicit fitted-record identifier or label",
        )
    return value.strip()


def _optional_string(value: object, field: str) -> str | None:
    if _is_missing(value):
        return None
    return _required_string(value, field)


def _optional_float(value: object, field: str) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise _contract_error(
            f"{field} must be numeric or missing",
            code="invalid_hypergraph_numeric",
            field=field,
        )
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError) as exc:
        raise _contract_error(
            f"{field} must be numeric or missing",
            code="invalid_hypergraph_numeric",
            field=field,
        ) from exc
    if not math.isfinite(result):
        raise _contract_error(
            f"{field} must be finite when present",
            code="invalid_hypergraph_numeric",
            field=field,
        )
    return result


def _forbidden_inference_name(name: str) -> bool:
    normalized = name.strip().lower()
    return (
        normalized in {"p", "q", "pvalue", "qvalue", "p_value", "q_value"}
        or "probability" in normalized
        or normalized in {"posterior", "posterior_odds"}
    )


def _check_json_keys(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _contract_error(
                    f"{path} must contain string keys",
                    code="invalid_hypergraph_json",
                    field=path,
                )
            if _forbidden_inference_name(key):
                raise _contract_error(
                    f"{path} cannot contain inferential field {key!r}",
                    code="forbidden_hypergraph_inference",
                    field=path,
                    remediation=(
                        "Keep p/q/probability/posterior quantities in inference results"
                    ),
                )
            _check_json_keys(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _check_json_keys(item, f"{path}[{index}]")


def _json_object(value: object, field: str) -> str:
    decoded: object
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _contract_error(
                f"{field} must be valid JSON",
                code="invalid_hypergraph_json",
                field=field,
            ) from exc
    else:
        decoded = value
    if not isinstance(decoded, Mapping):
        raise _contract_error(
            f"{field} must encode a JSON object",
            code="invalid_hypergraph_json",
            field=field,
        )
    _check_json_keys(decoded, field)
    try:
        return canonical_json(decoded)
    except (TypeError, ValueError) as exc:
        raise _contract_error(
            f"{field} contains a value that cannot be canonically serialized",
            code="invalid_hypergraph_json",
            field=field,
        ) from exc


def _canonical_table_json(value: object, field: str) -> str:
    normalized = _json_object(value, field)
    if not isinstance(value, str) or value != normalized:
        raise _contract_error(
            f"{field} must be stored as canonical JSON text",
            code="noncanonical_hypergraph_json",
            field=field,
        )
    return normalized


def _members(value: object, field: str) -> tuple[str, ...]:
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _contract_error(
                f"{field} must be a sequence or JSON array",
                code="invalid_complex_members",
                field=field,
            ) from exc
    if not isinstance(decoded, Sequence) or isinstance(
        decoded, (str, bytes, bytearray)
    ):
        raise _contract_error(
            f"{field} must be a non-empty sequence",
            code="invalid_complex_members",
            field=field,
        )
    normalized = tuple(sorted(_required_string(item, field) for item in decoded))
    if not normalized or len(set(normalized)) != len(normalized):
        raise _contract_error(
            f"{field} must contain unique members",
            code="invalid_complex_members",
            field=field,
        )
    return normalized


def _enum_value(value: object, enum_type: type[EnumT], field: str) -> EnumT:
    if not isinstance(value, str):
        raise _contract_error(
            f"{field} must be a string enum value",
            code="invalid_hypergraph_enum",
            field=field,
        )
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise _contract_error(
            f"{field} must be one of: {allowed}",
            code="invalid_hypergraph_enum",
            field=field,
        ) from exc


def _check_semantics(value: str, field: str) -> None:
    if _forbidden_inference_name(value) or any(
        token in value.lower() for token in ("p_value", "q_value")
    ):
        raise _contract_error(
            f"{field} cannot describe p/q/probability/posterior quantities",
            code="forbidden_hypergraph_inference",
            field=field,
        )


def _validate_availability(
    *,
    status: HyperedgeStatus,
    weight: float | None,
    uncertainty: float | None,
    uncertainty_kind: str | None,
    reason_code: str | None,
) -> None:
    if status is HyperedgeStatus.OBSERVED:
        if weight is None:
            raise _contract_error(
                "Observed hyperedges require a finite fitted weight",
                code="missing_hyperedge_weight",
                field="weight",
            )
        if reason_code is not None:
            raise _contract_error(
                "Observed hyperedges cannot carry a failure reason",
                code="invalid_hyperedge_reason",
                field="reason_code",
            )
    elif status is HyperedgeStatus.STRUCTURAL_ZERO:
        if weight is None or not math.isclose(weight, 0.0, abs_tol=0.0):
            raise _contract_error(
                "Structural-zero hyperedges require weight exactly zero",
                code="invalid_structural_zero",
                field="weight",
            )
        if uncertainty is not None:
            raise _contract_error(
                "Structural-zero hyperedges cannot carry fitted uncertainty",
                code="invalid_structural_zero",
                field="uncertainty",
            )
        if reason_code is None:
            raise _contract_error(
                "Structural-zero hyperedges require an explicit reason",
                code="missing_hyperedge_reason",
                field="reason_code",
            )
    else:
        if status not in _UNAVAILABLE_STATUSES:
            raise AssertionError("Unhandled hyperedge status")
        if weight is not None or uncertainty is not None:
            raise _contract_error(
                "Unavailable hyperedges must preserve missing weight and uncertainty",
                code="unavailable_hyperedge_has_value",
                field="weight",
                remediation="Use missing values; do not encode not-estimable as zero",
            )
        if reason_code is None:
            raise _contract_error(
                "Unavailable hyperedges require an explicit reason",
                code="missing_hyperedge_reason",
                field="reason_code",
            )
    if uncertainty is None and uncertainty_kind is not None:
        raise _contract_error(
            "uncertainty_kind requires an uncertainty value",
            code="invalid_hyperedge_uncertainty",
            field="uncertainty_kind",
        )
    if uncertainty is not None:
        if uncertainty < 0.0 or uncertainty_kind is None:
            raise _contract_error(
                "Uncertainty must be non-negative and have explicit semantics",
                code="invalid_hyperedge_uncertainty",
                field="uncertainty",
            )


@dataclass(frozen=True, slots=True)
class CommunicationEdgeRecord:
    """One explicit, already-fitted communication record."""

    source_record_id: str
    context_id: str
    context_json: str
    sender: str
    ligand: str
    ligand_members: tuple[str, ...]
    receptor: str
    receptor_members: tuple[str, ...]
    receiver: str
    target: str
    target_kind: TargetKind
    interaction_id: str
    mode: str
    weight: float | None
    weight_semantics: str
    uncertainty: float | None
    uncertainty_kind: str | None
    status: HyperedgeStatus
    reason_code: str | None
    source_artifact_id: str
    scoring_functional_id: str | None
    fold_id: str | None
    repeat_id: str | None
    components_json: str
    provenance_json: str

    def __post_init__(self) -> None:
        for field in (
            "source_record_id",
            "context_id",
            "sender",
            "ligand",
            "receptor",
            "receiver",
            "target",
            "interaction_id",
            "mode",
            "weight_semantics",
            "source_artifact_id",
        ):
            object.__setattr__(
                self, field, _required_string(getattr(self, field), field)
            )
        object.__setattr__(
            self, "context_json", _json_object(self.context_json, "context_json")
        )
        object.__setattr__(
            self,
            "components_json",
            _json_object(self.components_json, "components_json"),
        )
        object.__setattr__(
            self,
            "provenance_json",
            _json_object(self.provenance_json, "provenance_json"),
        )
        object.__setattr__(
            self, "ligand_members", _members(self.ligand_members, "ligand_members")
        )
        object.__setattr__(
            self,
            "receptor_members",
            _members(self.receptor_members, "receptor_members"),
        )
        object.__setattr__(
            self,
            "target_kind",
            _enum_value(self.target_kind, TargetKind, "target_kind"),
        )
        object.__setattr__(
            self, "status", _enum_value(self.status, HyperedgeStatus, "status")
        )
        object.__setattr__(self, "weight", _optional_float(self.weight, "weight"))
        object.__setattr__(
            self, "uncertainty", _optional_float(self.uncertainty, "uncertainty")
        )
        for field in (
            "uncertainty_kind",
            "reason_code",
            "scoring_functional_id",
            "fold_id",
            "repeat_id",
        ):
            object.__setattr__(
                self, field, _optional_string(getattr(self, field), field)
            )
        _check_semantics(self.weight_semantics, "weight_semantics")
        if self.uncertainty_kind is not None:
            _check_semantics(self.uncertainty_kind, "uncertainty_kind")
        _validate_availability(
            status=self.status,
            weight=self.weight,
            uncertainty=self.uncertainty,
            uncertainty_kind=self.uncertainty_kind,
            reason_code=self.reason_code,
        )


@dataclass(frozen=True, slots=True)
class HypergraphTables:
    """Primitive-valued tables suitable for Parquet persistence."""

    nodes: pd.DataFrame
    hyperedges: pd.DataFrame
    complex_members: pd.DataFrame
    schema_version: str = HYPERGRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", self.nodes.copy(deep=True))
        object.__setattr__(self, "hyperedges", self.hyperedges.copy(deep=True))
        object.__setattr__(
            self, "complex_members", self.complex_members.copy(deep=True)
        )


def _node_id(node_type: NodeType, entity_id: str, metadata_json: str) -> str:
    return stable_id(
        "hypergraph_node",
        {
            "entity_id": entity_id,
            "metadata_json": metadata_json,
            "node_type": node_type.value,
        },
        schema_version=HYPERGRAPH_ID_SCHEMA_VERSION,
    )


def _membership_id(
    complex_node_id: str,
    member_node_id: str,
    complex_role: str,
    member_index: int,
) -> str:
    return stable_id(
        "complex_membership",
        {
            "complex_node_id": complex_node_id,
            "complex_role": complex_role,
            "member_index": member_index,
            "member_node_id": member_node_id,
        },
        schema_version=HYPERGRAPH_ID_SCHEMA_VERSION,
    )


def _hyperedge_id(row: Mapping[str, object]) -> str:
    identity_fields = (
        "source_record_id",
        "context_node_id",
        "sender_node_id",
        "ligand_node_id",
        "receptor_node_id",
        "receiver_node_id",
        "target_node_id",
        "direction",
        "interaction_id",
        "mode",
        "weight_semantics",
        "source_artifact_id",
        "scoring_functional_id",
        "fold_id",
        "repeat_id",
    )
    return stable_id(
        "communication_hyperedge",
        {field: row[field] for field in identity_fields},
        schema_version=HYPERGRAPH_ID_SCHEMA_VERSION,
    )


def _target_node_type(kind: TargetKind) -> NodeType:
    return {
        TargetKind.GENE: NodeType.TARGET_GENE,
        TargetKind.TF: NodeType.TARGET_TF,
        TargetKind.PROGRAM: NodeType.TARGET_PROGRAM,
    }[kind]


def _records_from_frame(frame: pd.DataFrame) -> list[CommunicationEdgeRecord]:
    columns = set(map(str, frame.columns))
    forbidden = sorted(
        column for column in columns if _forbidden_inference_name(column)
    )
    if forbidden:
        raise _contract_error(
            "Hypergraph records cannot contain inferential columns: " + repr(forbidden),
            code="forbidden_hypergraph_inference",
            field="columns",
        )
    missing = set(EDGE_RECORD_COLUMNS).difference(columns)
    unknown = columns.difference(EDGE_RECORD_COLUMNS)
    if missing or unknown:
        raise _contract_error(
            "Hypergraph records must use the explicit fitted-record schema",
            code="invalid_hypergraph_record_schema",
            field="columns",
            remediation=f"missing={sorted(missing)!r}; unknown={sorted(unknown)!r}",
        )
    raw_rows = cast(list[dict[str, object]], frame.to_dict(orient="records"))
    return [
        CommunicationEdgeRecord(
            source_record_id=cast(str, row["source_record_id"]),
            context_id=cast(str, row["context_id"]),
            context_json=cast(str, row["context_json"]),
            sender=cast(str, row["sender"]),
            ligand=cast(str, row["ligand"]),
            ligand_members=cast(tuple[str, ...], row["ligand_members"]),
            receptor=cast(str, row["receptor"]),
            receptor_members=cast(tuple[str, ...], row["receptor_members"]),
            receiver=cast(str, row["receiver"]),
            target=cast(str, row["target"]),
            target_kind=cast(TargetKind, row["target_kind"]),
            interaction_id=cast(str, row["interaction_id"]),
            mode=cast(str, row["mode"]),
            weight=cast(float | None, row["weight"]),
            weight_semantics=cast(str, row["weight_semantics"]),
            uncertainty=cast(float | None, row["uncertainty"]),
            uncertainty_kind=cast(str | None, row["uncertainty_kind"]),
            status=cast(HyperedgeStatus, row["status"]),
            reason_code=cast(str | None, row["reason_code"]),
            source_artifact_id=cast(str, row["source_artifact_id"]),
            scoring_functional_id=cast(str | None, row["scoring_functional_id"]),
            fold_id=cast(str | None, row["fold_id"]),
            repeat_id=cast(str | None, row["repeat_id"]),
            components_json=cast(str, row["components_json"]),
            provenance_json=cast(str, row["provenance_json"]),
        )
        for row in raw_rows
    ]


def _records(
    records: pd.DataFrame | Sequence[CommunicationEdgeRecord],
) -> list[CommunicationEdgeRecord]:
    if isinstance(records, pd.DataFrame):
        return _records_from_frame(records)
    result = list(records)
    if not all(isinstance(record, CommunicationEdgeRecord) for record in result):
        raise TypeError("Every typed record must be a CommunicationEdgeRecord")
    return result


def _register_node(
    nodes: dict[tuple[str, str], dict[str, object]],
    *,
    node_type: NodeType,
    entity_id: str,
    label: str,
    metadata_json: str = "{}",
) -> str:
    node_id = _node_id(node_type, entity_id, metadata_json)
    key = (node_type.value, entity_id)
    candidate: dict[str, object] = {
        "node_id": node_id,
        "node_type": node_type.value,
        "entity_id": entity_id,
        "label": label,
        "metadata_json": metadata_json,
    }
    existing = nodes.get(key)
    if existing is not None and existing != candidate:
        raise _contract_error(
            f"Node identity {key!r} has inconsistent metadata",
            code="inconsistent_hypergraph_node",
            field="entity_id",
        )
    nodes[key] = candidate
    return node_id


def _register_entity_node(
    nodes: dict[tuple[str, str], dict[str, object]],
    memberships: dict[str, dict[str, object]],
    *,
    entity_id: str,
    members: tuple[str, ...],
    simple_type: NodeType,
    complex_type: NodeType,
    complex_role: str,
) -> str:
    if len(members) == 1 and members[0] == entity_id:
        return _register_node(
            nodes,
            node_type=simple_type,
            entity_id=entity_id,
            label=entity_id,
        )
    metadata_json = canonical_json({"members": list(members)})
    complex_node_id = _register_node(
        nodes,
        node_type=complex_type,
        entity_id=entity_id,
        label=entity_id,
        metadata_json=metadata_json,
    )
    for index, member in enumerate(members):
        member_node_id = _register_node(
            nodes,
            node_type=simple_type,
            entity_id=member,
            label=member,
        )
        membership_id = _membership_id(
            complex_node_id, member_node_id, complex_role, index
        )
        memberships[membership_id] = {
            "membership_id": membership_id,
            "complex_node_id": complex_node_id,
            "member_node_id": member_node_id,
            "complex_role": complex_role,
            "member_index": index,
        }
    return complex_node_id


def build_communication_hypergraph(
    records: pd.DataFrame | Sequence[CommunicationEdgeRecord],
) -> CommunicationHypergraph:
    """Build a deterministic post-fit hypergraph from explicit fitted records."""

    fitted_records = _records(records)
    source_ids = [record.source_record_id for record in fitted_records]
    if len(source_ids) != len(set(source_ids)):
        raise _contract_error(
            "source_record_id must be unique",
            code="duplicate_hypergraph_source_record",
            field="source_record_id",
        )

    nodes: dict[tuple[str, str], dict[str, object]] = {}
    memberships: dict[str, dict[str, object]] = {}
    hyperedges: list[dict[str, object]] = []
    for record in fitted_records:
        context_node_id = _register_node(
            nodes,
            node_type=NodeType.CONTEXT,
            entity_id=record.context_id,
            label=record.context_id,
            metadata_json=record.context_json,
        )
        sender_node_id = _register_node(
            nodes,
            node_type=NodeType.SENDER,
            entity_id=record.sender,
            label=record.sender,
        )
        ligand_node_id = _register_entity_node(
            nodes,
            memberships,
            entity_id=record.ligand,
            members=record.ligand_members,
            simple_type=NodeType.LIGAND,
            complex_type=NodeType.LIGAND_COMPLEX,
            complex_role="ligand_subunit",
        )
        receptor_node_id = _register_entity_node(
            nodes,
            memberships,
            entity_id=record.receptor,
            members=record.receptor_members,
            simple_type=NodeType.RECEPTOR,
            complex_type=NodeType.RECEPTOR_COMPLEX,
            complex_role="receptor_subunit",
        )
        receiver_node_id = _register_node(
            nodes,
            node_type=NodeType.RECEIVER,
            entity_id=record.receiver,
            label=record.receiver,
        )
        target_node_id = _register_node(
            nodes,
            node_type=_target_node_type(record.target_kind),
            entity_id=record.target,
            label=record.target,
        )
        edge: dict[str, object] = {
            "source_record_id": record.source_record_id,
            "context_node_id": context_node_id,
            "sender_node_id": sender_node_id,
            "ligand_node_id": ligand_node_id,
            "receptor_node_id": receptor_node_id,
            "receiver_node_id": receiver_node_id,
            "target_node_id": target_node_id,
            "direction": _DIRECTION,
            "context_id": record.context_id,
            "context_json": record.context_json,
            "sender": record.sender,
            "ligand": record.ligand,
            "receptor": record.receptor,
            "receiver": record.receiver,
            "target": record.target,
            "target_kind": record.target_kind.value,
            "interaction_id": record.interaction_id,
            "mode": record.mode,
            "weight": record.weight,
            "weight_semantics": record.weight_semantics,
            "uncertainty": record.uncertainty,
            "uncertainty_kind": record.uncertainty_kind,
            "status": record.status.value,
            "reason_code": record.reason_code,
            "source_artifact_id": record.source_artifact_id,
            "scoring_functional_id": record.scoring_functional_id,
            "fold_id": record.fold_id,
            "repeat_id": record.repeat_id,
            "components_json": record.components_json,
            "provenance_json": record.provenance_json,
        }
        edge["hyperedge_id"] = _hyperedge_id(edge)
        hyperedges.append(edge)

    node_frame = pd.DataFrame(nodes.values(), columns=NODE_COLUMNS).sort_values(
        "node_id", ignore_index=True
    )
    edge_frame = pd.DataFrame(hyperedges, columns=HYPEREDGE_COLUMNS).sort_values(
        "hyperedge_id", ignore_index=True
    )
    membership_frame = pd.DataFrame(
        memberships.values(), columns=COMPLEX_MEMBER_COLUMNS
    ).sort_values("membership_id", ignore_index=True)
    return CommunicationHypergraph.from_tables(
        HypergraphTables(
            nodes=node_frame,
            hyperedges=edge_frame,
            complex_members=membership_frame,
        )
    )


def _validate_columns(
    frame: pd.DataFrame, expected: tuple[str, ...], name: str
) -> None:
    actual = tuple(map(str, frame.columns))
    if actual != expected:
        raise _contract_error(
            f"{name} columns do not match the hypergraph schema",
            code="invalid_hypergraph_table_schema",
            field=name,
            remediation=f"expected={expected!r}",
        )


def _validate_tables(tables: HypergraphTables) -> None:
    if tables.schema_version != HYPERGRAPH_SCHEMA_VERSION:
        raise _contract_error(
            "Unsupported hypergraph schema version",
            code="unsupported_hypergraph_schema",
            field="schema_version",
        )
    _validate_columns(tables.nodes, NODE_COLUMNS, "nodes")
    _validate_columns(tables.hyperedges, HYPEREDGE_COLUMNS, "hyperedges")
    _validate_columns(tables.complex_members, COMPLEX_MEMBER_COLUMNS, "complex_members")
    for frame_name, frame in (
        ("nodes", tables.nodes),
        ("hyperedges", tables.hyperedges),
        ("complex_members", tables.complex_members),
    ):
        forbidden = [
            str(column)
            for column in frame.columns
            if _forbidden_inference_name(str(column))
        ]
        if forbidden:
            raise _contract_error(
                f"{frame_name} cannot contain inferential columns",
                code="forbidden_hypergraph_inference",
                field=frame_name,
            )

    if tables.nodes["node_id"].duplicated().any():
        raise _contract_error(
            "node_id must be unique",
            code="duplicate_hypergraph_node",
            field="node_id",
        )
    node_types: dict[str, str] = {}
    node_metadata: dict[str, str] = {}
    node_entities: dict[str, str] = {}
    for row in tables.nodes.itertuples(index=False):
        node_type = _enum_value(row.node_type, NodeType, "node_type")
        entity_id = _required_string(row.entity_id, "entity_id")
        label = _required_string(row.label, "label")
        if label != entity_id:
            raise _contract_error(
                "Node label and canonical entity identifier disagree",
                code="inconsistent_hypergraph_node",
                field="label",
            )
        metadata_json = _canonical_table_json(row.metadata_json, "metadata_json")
        node_id = _required_string(row.node_id, "node_id")
        if node_id != _node_id(node_type, entity_id, metadata_json):
            raise _contract_error(
                "node_id does not match canonical node identity",
                code="unstable_hypergraph_node_id",
                field="node_id",
            )
        node_types[node_id] = node_type.value
        node_metadata[node_id] = metadata_json
        node_entities[node_id] = entity_id

    if tables.hyperedges["hyperedge_id"].duplicated().any():
        raise _contract_error(
            "hyperedge_id must be unique",
            code="duplicate_hyperedge_id",
            field="hyperedge_id",
        )
    if tables.hyperedges["source_record_id"].duplicated().any():
        raise _contract_error(
            "source_record_id must be unique",
            code="duplicate_hypergraph_source_record",
            field="source_record_id",
        )
    role_types: dict[str, set[str]] = {
        "context_node_id": {NodeType.CONTEXT.value},
        "sender_node_id": {NodeType.SENDER.value},
        "ligand_node_id": {NodeType.LIGAND.value, NodeType.LIGAND_COMPLEX.value},
        "receptor_node_id": {
            NodeType.RECEPTOR.value,
            NodeType.RECEPTOR_COMPLEX.value,
        },
        "receiver_node_id": {NodeType.RECEIVER.value},
        "target_node_id": {
            NodeType.TARGET_GENE.value,
            NodeType.TARGET_TF.value,
            NodeType.TARGET_PROGRAM.value,
        },
    }
    for raw in cast(
        list[dict[str, object]], tables.hyperedges.to_dict(orient="records")
    ):
        for field in (
            "hyperedge_id",
            "source_record_id",
            "context_id",
            "sender",
            "ligand",
            "receptor",
            "receiver",
            "target",
            "interaction_id",
            "mode",
            "source_artifact_id",
        ):
            _required_string(raw[field], field)
        for field, allowed_types in role_types.items():
            node_id = _required_string(raw[field], field)
            if node_id not in node_types:
                raise _contract_error(
                    f"{field} references an unknown node",
                    code="dangling_hypergraph_node",
                    field=field,
                )
            if node_types[node_id] not in allowed_types:
                raise _contract_error(
                    f"{field} references the wrong node type",
                    code="invalid_hypergraph_direction",
                    field=field,
                )
        entity_roles = {
            "context_node_id": "context_id",
            "sender_node_id": "sender",
            "ligand_node_id": "ligand",
            "receptor_node_id": "receptor",
            "receiver_node_id": "receiver",
            "target_node_id": "target",
        }
        for node_field, entity_field in entity_roles.items():
            node_id = cast(str, raw[node_field])
            entity_id = _required_string(raw[entity_field], entity_field)
            if node_entities[node_id] != entity_id:
                raise _contract_error(
                    f"{entity_field} disagrees with its typed node",
                    code="inconsistent_hypergraph_node",
                    field=entity_field,
                )
        if raw["direction"] != _DIRECTION:
            raise _contract_error(
                "Hyperedge direction is not the canonical sender-to-target direction",
                code="invalid_hypergraph_direction",
                field="direction",
            )
        target_kind = _enum_value(raw["target_kind"], TargetKind, "target_kind")
        target_node_id = cast(str, raw["target_node_id"])
        if node_types[target_node_id] != _target_node_type(target_kind).value:
            raise _contract_error(
                "target_kind and target node type disagree",
                code="invalid_hypergraph_direction",
                field="target_kind",
            )
        context_json = _canonical_table_json(raw["context_json"], "context_json")
        if node_metadata[cast(str, raw["context_node_id"])] != context_json:
            raise _contract_error(
                "Context node metadata disagrees with the hyperedge context",
                code="inconsistent_hypergraph_context",
                field="context_json",
            )
        _canonical_table_json(raw["components_json"], "components_json")
        _canonical_table_json(raw["provenance_json"], "provenance_json")
        weight_semantics = _required_string(raw["weight_semantics"], "weight_semantics")
        _check_semantics(weight_semantics, "weight_semantics")
        uncertainty_kind = _optional_string(raw["uncertainty_kind"], "uncertainty_kind")
        if uncertainty_kind is not None:
            _check_semantics(uncertainty_kind, "uncertainty_kind")
        _validate_availability(
            status=_enum_value(raw["status"], HyperedgeStatus, "status"),
            weight=_optional_float(raw["weight"], "weight"),
            uncertainty=_optional_float(raw["uncertainty"], "uncertainty"),
            uncertainty_kind=uncertainty_kind,
            reason_code=_optional_string(raw["reason_code"], "reason_code"),
        )
        if raw["hyperedge_id"] != _hyperedge_id(raw):
            raise _contract_error(
                "hyperedge_id does not match canonical directed identity",
                code="unstable_hyperedge_id",
                field="hyperedge_id",
            )

    if tables.complex_members["membership_id"].duplicated().any():
        raise _contract_error(
            "membership_id must be unique",
            code="duplicate_complex_membership",
            field="membership_id",
        )
    member_sets: dict[str, list[tuple[int, str]]] = {}
    for row in tables.complex_members.itertuples(index=False):
        complex_node_id = _required_string(row.complex_node_id, "complex_node_id")
        member_node_id = _required_string(row.member_node_id, "member_node_id")
        role = _required_string(row.complex_role, "complex_role")
        if complex_node_id not in node_types or member_node_id not in node_types:
            raise _contract_error(
                "Complex membership references an unknown node",
                code="dangling_complex_membership",
                field="complex_node_id",
            )
        expected_pair = {
            "ligand_subunit": (
                NodeType.LIGAND_COMPLEX.value,
                NodeType.LIGAND.value,
            ),
            "receptor_subunit": (
                NodeType.RECEPTOR_COMPLEX.value,
                NodeType.RECEPTOR.value,
            ),
        }.get(role)
        if (
            expected_pair is None
            or (node_types[complex_node_id], node_types[member_node_id])
            != expected_pair
        ):
            raise _contract_error(
                "Complex membership has incompatible role or node types",
                code="invalid_complex_membership",
                field="complex_role",
            )
        if not isinstance(row.member_index, int) or row.member_index < 0:
            raise _contract_error(
                "member_index must be a non-negative integer",
                code="invalid_complex_membership",
                field="member_index",
            )
        expected_id = _membership_id(
            complex_node_id, member_node_id, role, row.member_index
        )
        if row.membership_id != expected_id:
            raise _contract_error(
                "membership_id does not match canonical membership identity",
                code="unstable_complex_membership_id",
                field="membership_id",
            )
        member_sets.setdefault(complex_node_id, []).append(
            (row.member_index, member_node_id)
        )
    complex_ids = tables.nodes.loc[
        tables.nodes["node_type"].isin(
            [NodeType.LIGAND_COMPLEX.value, NodeType.RECEPTOR_COMPLEX.value]
        ),
        "node_id",
    ]
    for complex_node_id in complex_ids:
        members = sorted(member_sets.get(complex_node_id, []))
        metadata = json.loads(node_metadata[complex_node_id])
        expected_members = cast(dict[str, list[str]], metadata).get("members")
        observed_members = [node_entities[member_id] for _, member_id in members]
        if expected_members != observed_members or [
            index for index, _ in members
        ] != list(range(len(members))):
            raise _contract_error(
                "Complex metadata and membership rows disagree",
                code="invalid_complex_membership",
                field="complex_members",
            )


@dataclass(frozen=True, slots=True)
class CommunicationHypergraph:
    """Immutable facade over a fitted communication hypergraph."""

    _nodes: pd.DataFrame
    _hyperedges: pd.DataFrame
    _complex_members: pd.DataFrame
    schema_version: str = HYPERGRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        tables = HypergraphTables(
            self._nodes,
            self._hyperedges,
            self._complex_members,
            self.schema_version,
        )
        _validate_tables(tables)
        object.__setattr__(self, "_nodes", tables.nodes)
        object.__setattr__(self, "_hyperedges", tables.hyperedges)
        object.__setattr__(self, "_complex_members", tables.complex_members)

    @classmethod
    def from_tables(cls, tables: HypergraphTables) -> CommunicationHypergraph:
        """Restore and validate a hypergraph from primitive persisted tables."""

        return cls(
            tables.nodes,
            tables.hyperedges,
            tables.complex_members,
            tables.schema_version,
        )

    @property
    def inference_eligible(self) -> bool:
        """The exploratory graph is never a source of formal inference."""

        return False

    @property
    def post_fit_only(self) -> bool:
        """The graph is structurally restricted to post-fit use."""

        return True

    @property
    def nodes(self) -> pd.DataFrame:
        """Return a defensive copy of typed nodes."""

        return self._nodes.copy(deep=True)

    @property
    def hyperedges(self) -> pd.DataFrame:
        """Return a defensive copy of directed hyperedges."""

        return self._hyperedges.copy(deep=True)

    @property
    def complex_members(self) -> pd.DataFrame:
        """Return a defensive copy of explicit complex membership rows."""

        return self._complex_members.copy(deep=True)

    def to_tables(self) -> HypergraphTables:
        """Return defensive, Parquet-ready primitive tables."""

        return HypergraphTables(
            self._nodes,
            self._hyperedges,
            self._complex_members,
            self.schema_version,
        )

    def query_edges(
        self,
        *,
        context_id: str | None = None,
        sender: str | None = None,
        receiver: str | None = None,
        target: str | None = None,
        status: HyperedgeStatus | str | None = None,
        mode: str | None = None,
    ) -> pd.DataFrame:
        """Filter directed records without changing missing-state semantics."""

        result = self._hyperedges
        filters: tuple[tuple[str, str | None], ...] = (
            ("context_id", context_id),
            ("sender", sender),
            ("receiver", receiver),
            ("target", target),
            ("mode", mode),
        )
        mask = pd.Series(True, index=result.index, dtype=bool)
        for column, value in filters:
            if value is not None:
                mask &= result[column].astype(str).eq(value)
        if status is not None:
            normalized_status = _enum_value(status, HyperedgeStatus, "status")
            mask &= result["status"].eq(normalized_status.value)
        return result.loc[mask].reset_index(drop=True).copy(deep=True)

    def query_nodes(
        self,
        *,
        node_type: NodeType | str | None = None,
        entity_id: str | None = None,
    ) -> pd.DataFrame:
        """Filter nodes by explicit biological type or entity identifier."""

        result = self._nodes
        mask = pd.Series(True, index=result.index, dtype=bool)
        if node_type is not None:
            normalized_type = _enum_value(node_type, NodeType, "node_type")
            mask &= result["node_type"].eq(normalized_type.value)
        if entity_id is not None:
            mask &= result["entity_id"].eq(entity_id)
        return result.loc[mask].reset_index(drop=True).copy(deep=True)

    def degree_table(
        self,
        *,
        statuses: Sequence[HyperedgeStatus | str] = _DEFAULT_TOPOLOGY_STATUSES,
    ) -> pd.DataFrame:
        """Compute directional incidence and absolute fitted-weight strength.

        Complex incidence is assigned to the complex node and is not propagated
        to subunits. Unavailable edges are excluded unless explicitly requested;
        if requested, they contribute incidence but zero strength.
        """

        normalized_statuses = _normalize_statuses(statuses)
        selected = self._hyperedges.loc[
            self._hyperedges["status"].isin(normalized_statuses)
        ]
        metrics: dict[str, dict[str, float]] = {
            cast(str, node_id): {
                "in_degree": 0.0,
                "out_degree": 0.0,
                "conditioning_degree": 0.0,
                "in_strength": 0.0,
                "out_strength": 0.0,
                "conditioning_strength": 0.0,
            }
            for node_id in self._nodes["node_id"]
        }
        for row in selected.itertuples(index=False):
            fitted_weight = _optional_float(row.weight, "weight")
            magnitude = 0.0 if fitted_weight is None else abs(fitted_weight)
            for field in ("sender_node_id", "ligand_node_id"):
                node_metrics = metrics[cast(str, getattr(row, field))]
                node_metrics["out_degree"] += 1.0
                node_metrics["out_strength"] += magnitude
            for field in (
                "receptor_node_id",
                "receiver_node_id",
                "target_node_id",
            ):
                node_metrics = metrics[cast(str, getattr(row, field))]
                node_metrics["in_degree"] += 1.0
                node_metrics["in_strength"] += magnitude
            context_metrics = metrics[cast(str, row.context_node_id)]
            context_metrics["conditioning_degree"] += 1.0
            context_metrics["conditioning_strength"] += magnitude

        status_policy = canonical_json(sorted(normalized_statuses))
        rows: list[dict[str, object]] = []
        for node in self._nodes.itertuples(index=False):
            values = metrics[cast(str, node.node_id)]
            rows.append(
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "label": node.label,
                    "in_degree": int(values["in_degree"]),
                    "out_degree": int(values["out_degree"]),
                    "conditioning_degree": int(values["conditioning_degree"]),
                    "total_degree": int(
                        values["in_degree"]
                        + values["out_degree"]
                        + values["conditioning_degree"]
                    ),
                    "in_strength": values["in_strength"],
                    "out_strength": values["out_strength"],
                    "conditioning_strength": values["conditioning_strength"],
                    "total_strength": values["in_strength"]
                    + values["out_strength"]
                    + values["conditioning_strength"],
                    "edge_status_policy": status_policy,
                    "direction_policy": (
                        "tail=sender+ligand;head=receptor+receiver+target;"
                        "context=conditioning;members=not_propagated"
                    ),
                    "weight_policy": "sum_absolute_fitted_weight;missing=zero_strength",
                }
            )
        return pd.DataFrame(rows, columns=DEGREE_COLUMNS).sort_values(
            "node_id", ignore_index=True
        )

    def hubs(
        self,
        *,
        top_k: int = 10,
        node_type: NodeType | str | None = None,
        direction: Literal["in", "out", "conditioning", "total"] = "total",
        weighted: bool = True,
        statuses: Sequence[HyperedgeStatus | str] = _DEFAULT_TOPOLOGY_STATUSES,
    ) -> pd.DataFrame:
        """Rank simple descriptive hubs using explicit degree policies."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if direction not in {"in", "out", "conditioning", "total"}:
            raise ValueError("direction must be in, out, conditioning, or total")
        result = self.degree_table(statuses=statuses)
        if node_type is not None:
            normalized_type = _enum_value(node_type, NodeType, "node_type")
            result = result.loc[result["node_type"].eq(normalized_type.value)]
        metric = f"{direction}_{'strength' if weighted else 'degree'}"
        ranked = result.sort_values(
            [metric, "node_id"], ascending=[False, True], ignore_index=True
        ).head(top_k)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        ranked.insert(1, "ranking_metric", metric)
        return ranked.copy(deep=True)

    def connected_components(
        self,
        *,
        statuses: Sequence[HyperedgeStatus | str] = _DEFAULT_TOPOLOGY_STATUSES,
    ) -> pd.DataFrame:
        """Return weak incidence components, including complex memberships."""

        normalized_statuses = _normalize_statuses(statuses)
        node_ids = [cast(str, value) for value in self._nodes["node_id"]]
        parent = {node_id: node_id for node_id in node_ids}

        def find(node_id: str) -> str:
            root = node_id
            while parent[root] != root:
                root = parent[root]
            while parent[node_id] != node_id:
                next_node = parent[node_id]
                parent[node_id] = root
                node_id = next_node
            return root

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            smaller, larger = sorted((left_root, right_root))
            parent[larger] = smaller

        selected = self._hyperedges.loc[
            self._hyperedges["status"].isin(normalized_statuses)
        ]
        incidence_fields = (
            "context_node_id",
            "sender_node_id",
            "ligand_node_id",
            "receptor_node_id",
            "receiver_node_id",
            "target_node_id",
        )
        for row in selected.itertuples(index=False):
            incident = [cast(str, getattr(row, field)) for field in incidence_fields]
            for node_id in incident[1:]:
                union(incident[0], node_id)
        for row in self._complex_members.itertuples(index=False):
            union(cast(str, row.complex_node_id), cast(str, row.member_node_id))

        groups: dict[str, list[str]] = {}
        for node_id in node_ids:
            groups.setdefault(find(node_id), []).append(node_id)
        status_policy = canonical_json(sorted(normalized_statuses))
        rows = []
        for members in groups.values():
            sorted_members = sorted(members)
            component_id = stable_id(
                "hypergraph_component",
                sorted_members,
                schema_version=HYPERGRAPH_ID_SCHEMA_VERSION,
            )
            for node_id in sorted_members:
                rows.append(
                    {
                        "component_id": component_id,
                        "node_id": node_id,
                        "component_size": len(sorted_members),
                        "edge_status_policy": status_policy,
                        "connectivity_policy": (
                            "weak_hyperedge_incidence+complex_membership"
                        ),
                    }
                )
        return pd.DataFrame(rows, columns=COMPONENT_COLUMNS).sort_values(
            ["component_id", "node_id"], ignore_index=True
        )


def _normalize_statuses(
    statuses: Sequence[HyperedgeStatus | str],
) -> tuple[str, ...]:
    if isinstance(statuses, (str, bytes, bytearray)):
        raise TypeError("statuses must be a sequence, not a scalar string")
    normalized = tuple(
        sorted(
            {
                _enum_value(value, HyperedgeStatus, "statuses").value
                for value in statuses
            }
        )
    )
    if not normalized:
        raise ValueError("statuses cannot be empty")
    return normalized


__all__ = [
    "COMPLEX_MEMBER_COLUMNS",
    "DEGREE_COLUMNS",
    "EDGE_RECORD_COLUMNS",
    "HYPEREDGE_COLUMNS",
    "HYPERGRAPH_SCHEMA_VERSION",
    "NODE_COLUMNS",
    "CommunicationEdgeRecord",
    "CommunicationHypergraph",
    "HyperedgeStatus",
    "HypergraphTables",
    "NodeType",
    "TargetKind",
    "build_communication_hypergraph",
]
