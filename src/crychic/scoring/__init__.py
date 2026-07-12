"""Frozen sample-level communication strength integration."""

from .contracts import (
    CORE_COMPONENTS,
    LEGACY_UNTRACKED_SCORE_VERSION,
    CommunicationScores,
    CommunicationScoreStatus,
    ScoringFunctional,
    ScoringFunctionalStatus,
    ScoringModelManifest,
    float64_array_digest,
    validate_common_functional,
)
from .integration import (
    mechanistic_strength,
    pair_softmin,
    score_communication,
    weighted_geometric_strength,
)

__all__ = [
    "CORE_COMPONENTS",
    "LEGACY_UNTRACKED_SCORE_VERSION",
    "CommunicationScoreStatus",
    "CommunicationScores",
    "ScoringFunctional",
    "ScoringFunctionalStatus",
    "ScoringModelManifest",
    "float64_array_digest",
    "mechanistic_strength",
    "pair_softmin",
    "score_communication",
    "validate_common_functional",
    "weighted_geometric_strength",
]
