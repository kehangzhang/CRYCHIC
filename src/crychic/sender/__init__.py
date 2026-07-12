"""Exploratory, evidence-based sender soft assignment."""

from .assignment import assign_senders
from .contracts import (
    ASSIGNMENT_GROUP_COLUMNS,
    SENDER_ASSIGNMENT_COLUMNS,
    SENDER_EVIDENCE_COMPONENTS,
    SenderAssignment,
    SenderAssignmentStatus,
    SenderCouplingStatus,
    SenderEvidenceParameters,
    sender_assignment_id,
)

__all__ = [
    "ASSIGNMENT_GROUP_COLUMNS",
    "SENDER_ASSIGNMENT_COLUMNS",
    "SENDER_EVIDENCE_COMPONENTS",
    "SenderAssignment",
    "SenderAssignmentStatus",
    "SenderCouplingStatus",
    "SenderEvidenceParameters",
    "assign_senders",
    "sender_assignment_id",
]
