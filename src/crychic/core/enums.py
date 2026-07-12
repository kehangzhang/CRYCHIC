"""Shared enumerations with stable serialized values."""

from enum import StrEnum


class CommunicationMode(StrEnum):
    """Supported communication estimands."""

    STATE = "state"
    ECOSYSTEM = "ecosystem"


class ResultStatus(StrEnum):
    """Status values that keep null and failure semantics distinct."""

    OK = "ok"
    MISSING = "missing"
    NOT_ESTIMABLE = "not_estimable"
    FILTERED = "filtered"
    FAILED = "failed"
