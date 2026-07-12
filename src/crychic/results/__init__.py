"""Versioned persistence and immutable result queries."""

from ._schema import (
    EDGE_EVIDENCE_EXTENSION_NAME,
    EDGE_EVIDENCE_EXTENSION_VERSION,
    RESULT_SCHEMA_VERSION,
    TABLE_NAMES,
    edge_evidence_contract,
    empty_table,
    table_contract,
    validate_edge_evidence,
    validate_edge_evidence_links,
    validate_table,
)
from .errors import (
    IncompleteResultError,
    ResultError,
    ResultValidationError,
    ResultWriteError,
)
from .facade import CrychicResult, run_manifest_schema
from .persistence import write_result

__all__ = [
    "EDGE_EVIDENCE_EXTENSION_NAME",
    "EDGE_EVIDENCE_EXTENSION_VERSION",
    "RESULT_SCHEMA_VERSION",
    "TABLE_NAMES",
    "CrychicResult",
    "IncompleteResultError",
    "ResultError",
    "ResultValidationError",
    "ResultWriteError",
    "edge_evidence_contract",
    "empty_table",
    "run_manifest_schema",
    "table_contract",
    "validate_edge_evidence",
    "validate_edge_evidence_links",
    "validate_table",
    "write_result",
]
