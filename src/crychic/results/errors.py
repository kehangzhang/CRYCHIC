"""Result validation and persistence errors."""

from crychic.core import ContractError


class ResultError(ContractError):
    """Base class for result-directory contract failures."""


class ResultValidationError(ResultError):
    """Raised when a result or table violates its persisted schema."""


class IncompleteResultError(ResultError):
    """Raised when an interrupted result is loaded as successful."""


class ResultWriteError(ResultError):
    """Raised after an atomic result write fails and is marked incomplete."""
