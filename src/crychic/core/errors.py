"""Structured exceptions for public and cross-module contracts."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorDetails:
    """Machine-readable error metadata without data values."""

    code: str
    field: str | None = None
    remediation: str | None = None


class CrychicError(Exception):
    """Base class for actionable CRYCHIC errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        field: str | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = ErrorDetails(
            code=code,
            field=field,
            remediation=remediation,
        )

    def __str__(self) -> str:
        parts = [self.message, f"code={self.details.code}"]
        if self.details.field is not None:
            parts.append(f"field={self.details.field}")
        if self.details.remediation is not None:
            parts.append(f"remediation={self.details.remediation}")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, str | None]:
        """Return a log-safe representation of the error."""

        return {
            "message": self.message,
            "code": self.details.code,
            "field": self.details.field,
            "remediation": self.details.remediation,
        }


class ConfigurationError(CrychicError):
    """Raised when configuration fields are invalid or contradictory."""


class ContractError(CrychicError):
    """Raised when an in-memory or persisted contract is violated."""


class FeatureUnavailableError(CrychicError):
    """Raised when a planned public operation is not implemented yet."""
