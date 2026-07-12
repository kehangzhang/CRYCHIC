"""Subject-block fold and resampling contracts."""

from .contracts import (
    FoldEstimabilityResult,
    FoldManifest,
    FoldPlanningError,
    SubjectFoldPlan,
)
from .folds import DesignFoldChecker, FoldEstimabilityChecker, plan_subject_folds
from .validation import OOFCoverageAudit, validate_oof_subject_coverage

__all__ = [
    "DesignFoldChecker",
    "FoldEstimabilityChecker",
    "FoldEstimabilityResult",
    "FoldManifest",
    "FoldPlanningError",
    "OOFCoverageAudit",
    "SubjectFoldPlan",
    "plan_subject_folds",
    "validate_oof_subject_coverage",
]
