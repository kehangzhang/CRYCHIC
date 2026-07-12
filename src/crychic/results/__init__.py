"""Versioned persistence and immutable result queries."""

from ._schema import (
    RESULT_SCHEMA_VERSION,
    TABLE_NAMES,
    empty_table,
    table_contract,
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
    "RESULT_SCHEMA_VERSION",
    "TABLE_NAMES",
    "CrychicResult",
    "IncompleteResultError",
    "ResultError",
    "ResultValidationError",
    "ResultWriteError",
    "empty_table",
    "run_manifest_schema",
    "table_contract",
    "validate_table",
    "write_result",
]
