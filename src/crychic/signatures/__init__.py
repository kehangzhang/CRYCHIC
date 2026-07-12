"""Observed, attributed, direction-consistent, and residual signatures."""

from .builder import build_signature_table
from .contracts import (
    CONTRIBUTION_COLUMNS,
    GENE_SIGNATURE_COLUMNS,
    DirectionAgreement,
    SignatureStatus,
    SignatureTable,
)

__all__ = [
    "CONTRIBUTION_COLUMNS",
    "GENE_SIGNATURE_COLUMNS",
    "DirectionAgreement",
    "SignatureStatus",
    "SignatureTable",
    "build_signature_table",
]
