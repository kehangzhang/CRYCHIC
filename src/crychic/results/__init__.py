"""Versioned persistence and immutable result queries."""

from crychic.scoring.contracts import SCORING_COLLECTION_EXTENSION_VERSION

from ._schema import (
    EDGE_EVIDENCE_EXTENSION_NAME,
    EDGE_EVIDENCE_EXTENSION_VERSION,
    RESULT_SCHEMA_VERSION,
    SCORING_COLLECTION_EXTENSION_NAME,
    TABLE_NAMES,
    edge_evidence_contract,
    empty_table,
    scoring_collections_contract,
    table_contract,
    validate_edge_evidence,
    validate_edge_evidence_links,
    validate_scoring_collection_links,
    validate_scoring_collections,
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
    "SCORING_COLLECTION_EXTENSION_NAME",
    "SCORING_COLLECTION_EXTENSION_VERSION",
    "TABLE_NAMES",
    "CrychicResult",
    "IncompleteResultError",
    "ResultError",
    "ResultValidationError",
    "ResultWriteError",
    "edge_evidence_contract",
    "empty_table",
    "run_manifest_schema",
    "scoring_collections_contract",
    "table_contract",
    "validate_edge_evidence",
    "validate_edge_evidence_links",
    "validate_scoring_collection_links",
    "validate_scoring_collections",
    "validate_table",
    "write_result",
]
