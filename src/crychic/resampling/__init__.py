"""Subject-block fold and resampling contracts."""

from .active_null import (
    ActiveNullPlan,
    ActiveNullPlanStatus,
    ActiveNullPriorLink,
    ActiveNullReasonCode,
    ActiveNullSpec,
    ActiveNullTierDiagnostic,
    materialize_active_null_target_prior,
    plan_active_null_target_prior,
    target_prior_content_id,
)
from .contracts import (
    FoldEstimabilityResult,
    FoldManifest,
    FoldPlanningError,
    SubjectFoldPlan,
)
from .exchangeability import (
    BootstrapSubjectDraw,
    ContextPermutationOperation,
    ContextPermutationPlan,
    ExchangeabilityDesign,
    ExchangeabilityMap,
    SubjectBootstrapPlan,
    apply_context_permutation,
    build_exchangeability_map,
    plan_context_permutations,
    plan_subject_bootstraps,
)
from .folds import DesignFoldChecker, FoldEstimabilityChecker, plan_subject_folds
from .validation import OOFCoverageAudit, validate_oof_subject_coverage

__all__ = [
    "ActiveNullPlan",
    "ActiveNullPlanStatus",
    "ActiveNullPriorLink",
    "ActiveNullReasonCode",
    "ActiveNullSpec",
    "ActiveNullTierDiagnostic",
    "BootstrapSubjectDraw",
    "ContextPermutationOperation",
    "ContextPermutationPlan",
    "DesignFoldChecker",
    "ExchangeabilityDesign",
    "ExchangeabilityMap",
    "FoldEstimabilityChecker",
    "FoldEstimabilityResult",
    "FoldManifest",
    "FoldPlanningError",
    "OOFCoverageAudit",
    "SubjectBootstrapPlan",
    "SubjectFoldPlan",
    "apply_context_permutation",
    "build_exchangeability_map",
    "materialize_active_null_target_prior",
    "plan_active_null_target_prior",
    "plan_context_permutations",
    "plan_subject_bootstraps",
    "plan_subject_folds",
    "target_prior_content_id",
    "validate_oof_subject_coverage",
]
