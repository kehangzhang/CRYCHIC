"""Experimental context topology and balanced contrasts."""

from .audit import (
    DesignAudit,
    DesignAuditError,
    DesignStatus,
    SubjectDesign,
    SubjectDesignAudit,
    audit_sample_design,
    audit_subject_design,
    default_design_formula,
)
from .context_encoding import (
    ContextValues,
    canonical_context,
    context_fields,
    context_id,
    context_mapping,
    node_context_fields,
    node_context_mapping,
    plain_context_value,
)
from .context_graph import CanonicalContext, ContextEdge, ContextGraph, ContextNode
from .contrasts import (
    ContrastSpec,
    balanced_contrast,
    factorial_interaction_contrast,
    global_contrasts,
    global_one_vs_rest,
    local_contrasts,
    local_neighbor_contrast,
    marginal_factor_contrast,
)
from .frozen_encoder import (
    FrozenCovariateEncoding,
    FrozenDesignApplication,
    FrozenDesignEncoder,
    apply_frozen_design_encoder,
    fit_frozen_design_encoder,
)
from .repeated_measures import (
    FrozenRepeatedMeasuresDesign,
    RepeatedMeasuresDesignSpec,
    freeze_repeated_measures_design,
)

__all__ = [
    "CanonicalContext",
    "ContextEdge",
    "ContextGraph",
    "ContextNode",
    "ContextValues",
    "ContrastSpec",
    "DesignAudit",
    "DesignAuditError",
    "DesignStatus",
    "FrozenCovariateEncoding",
    "FrozenDesignApplication",
    "FrozenDesignEncoder",
    "FrozenRepeatedMeasuresDesign",
    "RepeatedMeasuresDesignSpec",
    "SubjectDesign",
    "SubjectDesignAudit",
    "apply_frozen_design_encoder",
    "audit_sample_design",
    "audit_subject_design",
    "balanced_contrast",
    "canonical_context",
    "context_fields",
    "context_id",
    "context_mapping",
    "default_design_formula",
    "factorial_interaction_contrast",
    "fit_frozen_design_encoder",
    "freeze_repeated_measures_design",
    "global_contrasts",
    "global_one_vs_rest",
    "local_contrasts",
    "local_neighbor_contrast",
    "marginal_factor_contrast",
    "node_context_fields",
    "node_context_mapping",
    "plain_context_value",
]
