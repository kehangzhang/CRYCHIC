"""Runtime implementation of the normative schemas under ``schemas/``."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api import types as pd_types

from crychic.core import canonical_json, stable_id

from .errors import ResultValidationError

RESULT_SCHEMA_VERSION = "0.1.0"
EDGE_EVIDENCE_EXTENSION_VERSION = "1.0.0"
EDGE_EVIDENCE_EXTENSION_NAME = "edge_evidence"
TABLE_NAMES = (
    "interactions",
    "differential",
    "responses",
    "sample_scores",
    "signatures",
)


@dataclass(frozen=True, slots=True)
class ColumnContract:
    """Resolved persisted column contract."""

    dtype: str
    nullable: bool
    enum: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True, slots=True)
class TableContract:
    """Resolved table contract loaded from the normative JSON document."""

    name: str
    filename: str
    primary_key: tuple[str, ...]
    columns: Mapping[str, ColumnContract]
    inferential_null_columns: tuple[str, ...]
    reason_required_for_null_columns: tuple[str, ...]
    schema_filename: str


@dataclass(frozen=True, slots=True)
class ResultExtensionContract:
    """Resolved contract for an optional persisted result extension."""

    name: str
    extension_schema_version: str
    filename: str
    primary_key: tuple[str, ...]
    linked_tables: tuple[str, ...]
    columns: tuple[str, ...]
    schema_filename: str


def _schema_text(filename: str) -> str:
    installed = files("crychic").joinpath("schemas", filename)
    if installed.is_file():
        return installed.read_text(encoding="utf-8")
    source = Path(__file__).resolve().parents[3] / "schemas" / filename
    try:
        return source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ResultValidationError(
            f"Normative result schema {filename} is unavailable",
            code="missing_result_schema",
            field="schema",
            remediation="Install a complete CRYCHIC distribution",
        ) from exc


def load_schema_document(filename: str) -> dict[str, Any]:
    """Load one normative schema document without maintaining a duplicate."""

    try:
        value = json.loads(_schema_text(filename))
    except json.JSONDecodeError as exc:
        raise ResultValidationError(
            f"Normative result schema {filename} is invalid JSON",
            code="invalid_result_schema",
            field="schema",
            remediation="Restore the released schema document",
        ) from exc
    if not isinstance(value, dict):
        raise ResultValidationError(
            f"Normative result schema {filename} is not an object",
            code="invalid_result_schema",
            field="schema",
            remediation="Restore the released schema document",
        )
    return value


def _resolve_column(
    value: Mapping[str, Any], document: Mapping[str, Any]
) -> Mapping[str, Any]:
    reference = value.get("$ref")
    if reference is None:
        return value
    prefix = "#/$defs/"
    if not isinstance(reference, str) or not reference.startswith(prefix):
        raise ResultValidationError(
            "Table schema contains an unsupported column reference",
            code="invalid_result_schema",
            field="$ref",
            remediation="Use a local $defs column reference",
        )
    definitions = document.get("$defs")
    if not isinstance(definitions, Mapping):
        raise ResultValidationError(
            "Table schema is missing $defs",
            code="invalid_result_schema",
            field="$defs",
            remediation="Restore the released schema document",
        )
    resolved = definitions.get(reference.removeprefix(prefix))
    if not isinstance(resolved, Mapping):
        raise ResultValidationError(
            "Table schema references an unknown column definition",
            code="invalid_result_schema",
            field="$ref",
            remediation="Restore the released schema document",
        )
    return resolved


@cache
def table_contract(name: str) -> TableContract:
    """Resolve and cache a named table contract."""

    if name not in TABLE_NAMES:
        raise ResultValidationError(
            f"Unknown result table {name!r}",
            code="unknown_result_table",
            field="table",
            remediation=f"Use one of: {', '.join(TABLE_NAMES)}",
        )
    schema_filename = f"{name}.schema.json"
    document = load_schema_document(schema_filename)
    properties = document.get("properties")
    if not isinstance(properties, Mapping):
        raise ResultValidationError(
            f"Table schema {schema_filename} has no properties",
            code="invalid_result_schema",
            field="properties",
            remediation="Restore the released schema document",
        )

    def constant(property_name: str) -> Any:
        value = properties.get(property_name)
        if not isinstance(value, Mapping) or "const" not in value:
            raise ResultValidationError(
                f"Table schema {schema_filename} lacks {property_name!r}",
                code="invalid_result_schema",
                field=property_name,
                remediation="Restore the released schema document",
            )
        return value["const"]

    version = constant("result_schema_version")
    table_name = constant("table")
    filename = constant("filename")
    primary_key = constant("primary_key")
    inferential = constant("inferential_null_columns")
    reason_columns = constant("reason_required_for_null_columns")
    column_block = properties.get("columns")
    if (
        version != RESULT_SCHEMA_VERSION
        or table_name != name
        or not isinstance(filename, str)
        or not isinstance(primary_key, list)
        or not isinstance(inferential, list)
        or not isinstance(reason_columns, list)
        or not isinstance(column_block, Mapping)
    ):
        raise ResultValidationError(
            f"Table schema {schema_filename} has incompatible metadata",
            code="invalid_result_schema",
            field="result_schema_version",
            remediation="Use schemas from the installed result-schema version",
        )
    raw_columns = column_block.get("properties")
    if not isinstance(raw_columns, Mapping):
        raise ResultValidationError(
            f"Table schema {schema_filename} has no column definitions",
            code="invalid_result_schema",
            field="columns",
            remediation="Restore the released schema document",
        )
    columns: dict[str, ColumnContract] = {}
    for column_name, raw_contract in raw_columns.items():
        if not isinstance(column_name, str) or not isinstance(raw_contract, Mapping):
            raise ResultValidationError(
                "Table schema has an invalid column definition",
                code="invalid_result_schema",
                field="columns",
                remediation="Restore the released schema document",
            )
        resolved = _resolve_column(raw_contract, document)
        dtype = resolved.get("dtype")
        nullable = resolved.get("nullable")
        if dtype not in {"string", "float", "integer", "boolean"} or not isinstance(
            nullable, bool
        ):
            raise ResultValidationError(
                f"Column {column_name!r} has an invalid dtype/null contract",
                code="invalid_result_schema",
                field=column_name,
                remediation="Restore the released schema document",
            )
        enum = resolved.get("enum", ())
        if not isinstance(enum, (list, tuple)):
            raise ResultValidationError(
                f"Column {column_name!r} has an invalid enum",
                code="invalid_result_schema",
                field=column_name,
                remediation="Restore the released schema document",
            )
        columns[column_name] = ColumnContract(
            dtype=dtype,
            nullable=nullable,
            enum=tuple(enum),
            minimum=resolved.get("minimum"),
            maximum=resolved.get("maximum"),
        )
    return TableContract(
        name=name,
        filename=filename,
        primary_key=tuple(primary_key),
        columns=MappingProxyType(columns),
        inferential_null_columns=tuple(inferential),
        reason_required_for_null_columns=tuple(reason_columns),
        schema_filename=schema_filename,
    )


@cache
def edge_evidence_contract() -> ResultExtensionContract:
    """Resolve the versioned optional edge-evidence result extension."""

    schema_filename = "edge_evidence.schema.json"
    document = load_schema_document(schema_filename)
    properties = document.get("properties")
    if not isinstance(properties, Mapping):
        raise ResultValidationError(
            f"Extension schema {schema_filename} has no properties",
            code="invalid_result_schema",
            field="properties",
            remediation="Restore the released schema document",
        )

    def constant(property_name: str) -> Any:
        value = properties.get(property_name)
        if not isinstance(value, Mapping) or "const" not in value:
            raise ResultValidationError(
                f"Extension schema {schema_filename} lacks {property_name!r}",
                code="invalid_result_schema",
                field=property_name,
                remediation="Restore the released schema document",
            )
        return value["const"]

    version = constant("extension_schema_version")
    result_version = constant("result_schema_version")
    table_name = constant("table")
    filename = constant("filename")
    primary_key = constant("primary_key")
    linked_tables = constant("linked_tables")
    column_block = properties.get("columns")
    raw_columns = (
        column_block.get("properties")
        if isinstance(column_block, Mapping)
        else None
    )
    if (
        version != EDGE_EVIDENCE_EXTENSION_VERSION
        or result_version != RESULT_SCHEMA_VERSION
        or table_name != EDGE_EVIDENCE_EXTENSION_NAME
        or filename != "edge_evidence.parquet"
        or not isinstance(primary_key, list)
        or not isinstance(linked_tables, list)
        or linked_tables != ["sample_scores"]
        or not isinstance(raw_columns, Mapping)
    ):
        raise ResultValidationError(
            f"Extension schema {schema_filename} has incompatible metadata",
            code="invalid_result_schema",
            field="extension_schema_version",
            remediation="Use schemas from the installed result extension version",
        )
    if not all(isinstance(value, str) and value for value in primary_key):
        raise ResultValidationError(
            f"Extension schema {schema_filename} has an invalid primary key",
            code="invalid_result_schema",
            field="primary_key",
            remediation="Restore the released schema document",
        )
    if not all(isinstance(value, str) and value for value in raw_columns):
        raise ResultValidationError(
            f"Extension schema {schema_filename} has invalid columns",
            code="invalid_result_schema",
            field="columns",
            remediation="Restore the released schema document",
        )
    if not set(primary_key).issubset(raw_columns):
        raise ResultValidationError(
            f"Extension schema {schema_filename} primary key is incomplete",
            code="invalid_result_schema",
            field="primary_key",
            remediation="Restore the released schema document",
        )
    return ResultExtensionContract(
        name=table_name,
        extension_schema_version=version,
        filename=filename,
        primary_key=tuple(primary_key),
        linked_tables=tuple(linked_tables),
        columns=tuple(raw_columns),
        schema_filename=schema_filename,
    )


def validate_edge_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the optional ledger without retaining an additional table copy."""

    from crychic.workflow.contracts import EdgeEvidenceLedger

    contract = edge_evidence_contract()
    if not isinstance(frame, pd.DataFrame):
        raise ResultValidationError(
            "Edge evidence must be a pandas DataFrame",
            code="invalid_result_extension",
            field=contract.name,
            remediation="Supply an EdgeEvidenceLedger-compatible table",
        )
    try:
        validated_columns = tuple(EdgeEvidenceLedger(frame).table.columns)
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(
            "Edge evidence does not satisfy its released extension contract",
            code="invalid_result_extension",
            field=contract.name,
            remediation="Regenerate the ledger from validated workflow artifacts",
        ) from exc
    if validated_columns != contract.columns:
        raise ResultValidationError(
            "Edge evidence columns do not match its released schema order",
            code="invalid_result_extension",
            field=contract.name,
            remediation="Write columns in the released edge-evidence order",
        )
    return frame


_EDGE_EVIDENCE_PERSISTED_LINK_COLUMNS = (
    "subject_id",
    "sample_id",
    "context_id",
    "design_row_id",
    "edge_id",
    "scoring_functional_id",
    "repeat_id",
    "fold_id",
    "mode",
    "availability",
    "downstream",
    "sender_component",
    "prior_quality",
    "comm_strength",
    "eligible",
    "status",
)


def _persisted_edge_link_projection(frame: pd.DataFrame) -> pd.DataFrame:
    mode = frame["mode"].astype("string")
    status = frame["legacy_integrated_status"].astype("string")
    projected = pd.DataFrame(
        {
            "subject_id": frame["subject_id"].astype("string"),
            "sample_id": frame["sample_id"].astype("string"),
            "context_id": frame["context_id"].astype("string"),
            "design_row_id": [
                stable_id(
                    "design_row",
                    {"context_id": str(context_id), "sample_id": str(sample_id)},
                )
                for context_id, sample_id in frame[
                    ["context_id", "sample_id"]
                ].itertuples(index=False, name=None)
            ],
            "edge_id": [
                stable_id(
                    "communication_edge",
                    {
                        "interaction_id": str(interaction_id),
                        "receiver": str(receiver),
                        "sender": str(sender),
                    },
                )
                for sender, receiver, interaction_id in frame[
                    ["sender", "receiver", "interaction_id"]
                ].itertuples(index=False, name=None)
            ],
            "scoring_functional_id": frame["scoring_function_id"].astype("string"),
            "repeat_id": pd.Series("repeat-0", index=frame.index, dtype="string"),
            "fold_id": frame["fold_id"].astype("string"),
            "mode": mode,
            "availability": pd.to_numeric(
                frame["state_availability"].where(
                    mode.eq("state"), frame["ecosystem_availability"]
                ),
                errors="coerce",
            ).astype("float64"),
            "downstream": pd.to_numeric(
                frame["legacy_downstream_activity"], errors="coerce"
            ).astype("float64"),
            "sender_component": pd.to_numeric(
                frame["sender_weight"], errors="coerce"
            ).astype("float64"),
            "prior_quality": pd.to_numeric(
                frame["prior_quality"], errors="coerce"
            ).astype("float64"),
            "comm_strength": pd.to_numeric(
                frame["legacy_integrated_strength"], errors="coerce"
            ).astype("float64"),
            "eligible": status.eq("ok").astype(bool),
            "status": status.where(status.eq("ok"), "missing").astype("string"),
        },
        index=frame.index,
    )
    return projected.loc[:, list(_EDGE_EVIDENCE_PERSISTED_LINK_COLUMNS)]


def validate_edge_evidence_links(
    edge_evidence: pd.DataFrame,
    sample_scores: pd.DataFrame,
    *,
    chunk_size: int = 100_000,
) -> None:
    """Validate linked ledger rows against persisted sample-score keys and values.

    The comparison is chunked and retains only one 64-bit row fingerprint per
    record, avoiding a full many-column merge for multi-million-row ledgers.
    """

    linked_mask = edge_evidence["sample_score_status"].eq("linked").to_numpy()
    linked_count = int(linked_mask.sum())
    if linked_count == 0:
        return
    if linked_count != len(sample_scores):
        raise ResultValidationError(
            "Linked edge evidence does not cover the persisted sample-score table",
            code="invalid_result_extension_linkage",
            field="edge_evidence",
            remediation="Persist sample scores and edge evidence from one artifact",
        )
    actual = sample_scores.loc[
        :, list(_EDGE_EVIDENCE_PERSISTED_LINK_COLUMNS)
    ]
    actual_hashes = pd.util.hash_pandas_object(actual, index=False).to_numpy(
        dtype="uint64", copy=True
    )
    expected_hashes: NDArray[np.uint64] = np.empty(linked_count, dtype="uint64")
    output_offset = 0
    for start in range(0, len(edge_evidence), chunk_size):
        stop = min(start + chunk_size, len(edge_evidence))
        chunk_mask = linked_mask[start:stop]
        if not bool(chunk_mask.any()):
            continue
        chunk = edge_evidence.iloc[start:stop].loc[chunk_mask]
        projected = _persisted_edge_link_projection(chunk)
        hashes = pd.util.hash_pandas_object(projected, index=False).to_numpy(
            dtype="uint64", copy=False
        )
        expected_hashes[output_offset : output_offset + len(hashes)] = hashes
        output_offset += len(hashes)
    actual_hashes.sort()
    expected_hashes.sort()
    if not np.array_equal(actual_hashes, expected_hashes):
        raise ResultValidationError(
            "Edge evidence does not reproduce persisted sample-score keys and values",
            code="invalid_result_extension_linkage",
            field="edge_evidence",
            remediation="Persist sample scores and edge evidence from one artifact",
        )


def empty_table(name: str) -> pd.DataFrame:
    """Return an empty DataFrame with the exact persisted dtypes."""

    contract = table_contract(name)
    dtypes = {
        "string": "string",
        "float": "float64",
        "integer": "int64",
        "boolean": "bool",
    }
    return pd.DataFrame(
        {
            column: pd.Series(dtype=dtypes[column_contract.dtype])
            for column, column_contract in contract.columns.items()
        }
    )


def _validate_dtype(
    name: str,
    series: pd.Series,
    contract: ColumnContract,
    *,
    table: str,
) -> None:
    non_null = series.dropna()
    valid = False
    if contract.dtype == "string":
        valid = all(isinstance(value, str) for value in non_null.tolist())
    elif contract.dtype == "float":
        valid = non_null.empty or (
            pd_types.is_numeric_dtype(non_null.dtype)
            and not pd_types.is_bool_dtype(non_null.dtype)
        )
    elif contract.dtype == "integer":
        valid = pd_types.is_integer_dtype(series.dtype) and not pd_types.is_bool_dtype(
            series.dtype
        )
    elif contract.dtype == "boolean":
        valid = pd_types.is_bool_dtype(series.dtype)
    if not valid:
        raise ResultValidationError(
            f"Table {table!r} column {name!r} must have dtype {contract.dtype}",
            code="invalid_result_dtype",
            field=name,
            remediation="Write the table using the released persisted schema",
        )
    if contract.dtype == "string" and any(not value for value in non_null.tolist()):
        raise ResultValidationError(
            f"Table {table!r} column {name!r} contains an empty string",
            code="invalid_result_value",
            field=name,
            remediation="Use a stable identifier or an explicit scope sentinel",
        )
    if contract.dtype == "float" and any(
        not math.isfinite(float(value)) for value in non_null
    ):
        raise ResultValidationError(
            f"Table {table!r} column {name!r} contains a non-finite value",
            code="invalid_result_value",
            field=name,
            remediation="Represent unavailable estimates as null with a reason code",
        )
    if contract.enum:
        invalid = ~non_null.isin(contract.enum)
        if bool(invalid.any()):
            raise ResultValidationError(
                f"Table {table!r} column {name!r} contains an unsupported enum value",
                code="invalid_result_enum",
                field=name,
                remediation="Use a value defined by the released persisted schema",
            )
    if contract.minimum is not None and bool((non_null < contract.minimum).any()):
        raise ResultValidationError(
            f"Table {table!r} column {name!r} is below its minimum",
            code="invalid_result_range",
            field=name,
            remediation="Check units and score construction before persistence",
        )
    if contract.maximum is not None and bool((non_null > contract.maximum).any()):
        raise ResultValidationError(
            f"Table {table!r} column {name!r} exceeds its maximum",
            code="invalid_result_range",
            field=name,
            remediation="Check units and score construction before persistence",
        )


def validate_table(name: str, frame: pd.DataFrame) -> None:
    """Validate columns, types, keys, ranges, and v0.1 null semantics."""

    if not isinstance(frame, pd.DataFrame):
        raise ResultValidationError(
            f"Result table {name!r} must be a pandas DataFrame",
            code="invalid_result_table",
            field=name,
            remediation="Convert the producer artifact to its persisted table contract",
        )
    contract = table_contract(name)
    expected = tuple(contract.columns)
    missing = set(expected).difference(frame.columns)
    extra = set(frame.columns).difference(expected)
    if missing or extra:
        raise ResultValidationError(
            f"Result table {name!r} columns do not match its schema",
            code="invalid_result_columns",
            field=sorted(missing or extra)[0],
            remediation="Write exactly the required versioned table columns",
        )
    for column_name, column_contract in contract.columns.items():
        series = frame[column_name]
        if not column_contract.nullable and bool(series.isna().any()):
            raise ResultValidationError(
                f"Table {name!r} column {column_name!r} cannot contain null",
                code="invalid_result_null",
                field=column_name,
                remediation="Supply the required value or reject the result row",
            )
        _validate_dtype(column_name, series, column_contract, table=name)

    if bool(frame.duplicated(subset=list(contract.primary_key), keep=False).any()):
        raise ResultValidationError(
            f"Result table {name!r} has duplicate primary keys",
            code="duplicate_result_key",
            field=contract.primary_key[0],
            remediation="Emit one row per declared table grain",
        )
    for column_name in contract.inferential_null_columns:
        if bool(frame[column_name].notna().any()):
            raise ResultValidationError(
                f"v0.1 table {name!r} must keep {column_name!r} entirely null",
                code="v0_1_inferential_field_enabled",
                field=column_name,
                remediation="Set the field to null and record a reason_code",
            )
    reason_missing = frame["reason_code"].isna()
    for column_name in contract.reason_required_for_null_columns:
        invalid = frame[column_name].isna() & reason_missing
        if bool(invalid.any()):
            raise ResultValidationError(
                f"Table {name!r} null {column_name!r} requires reason_code",
                code="missing_result_reason",
                field="reason_code",
                remediation="Record why the value is missing or unavailable",
            )
    if "status" in frame and bool(((frame["status"] != "ok") & reason_missing).any()):
        raise ResultValidationError(
            f"Table {name!r} non-ok rows require reason_code",
            code="missing_result_reason",
            field="reason_code",
            remediation="Record the missing, filtered, failed, or estimability reason",
        )
    if "context_json" in frame:
        context_pairs: list[tuple[str, str]] = []
        for value in frame["context_json"]:
            try:
                decoded = json.loads(value)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ResultValidationError(
                    f"Table {name!r} has invalid canonical context JSON",
                    code="invalid_context_json",
                    field="context_json",
                    remediation="Serialize the normalized context mapping canonically",
                ) from exc
            if canonical_json(decoded) != value:
                raise ResultValidationError(
                    f"Table {name!r} context_json is not canonical",
                    code="invalid_context_json",
                    field="context_json",
                    remediation="Use sorted compact canonical JSON serialization",
                )
            if not isinstance(decoded, dict):
                raise ResultValidationError(
                    f"Table {name!r} context_json must encode an object",
                    code="invalid_context_json",
                    field="context_json",
                    remediation="Serialize the normalized context mapping",
                )
        for context_identifier, context_value in frame[
            ["context_id", "context_json"]
        ].itertuples(index=False, name=None):
            decoded = json.loads(context_value)
            namespace = (
                "contrast_scope"
                if set(decoded) == {"contrast", "weights"}
                else "context"
            )
            if context_identifier != stable_id(namespace, decoded):
                raise ResultValidationError(
                    f"Table {name!r} context ID does not match context_json",
                    code="invalid_context_id",
                    field="context_id",
                    remediation="Use the shared canonical context encoder",
                )
            context_pairs.append((context_identifier, context_value))
        pairs = pd.DataFrame(context_pairs, columns=["context_id", "context_json"])
        if (
            pairs.groupby("context_id")["context_json"].nunique().gt(1).any()
            or pairs.groupby("context_json")["context_id"].nunique().gt(1).any()
        ):
            raise ResultValidationError(
                f"Table {name!r} context ID and JSON are not one-to-one",
                code="invalid_context_id",
                field="context_id",
                remediation="Use one canonical ID for each context mapping",
            )
