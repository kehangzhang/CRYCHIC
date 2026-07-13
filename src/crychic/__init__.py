"""CRYCHIC public Python interface."""

from importlib.metadata import PackageNotFoundError, version

from .api import (
    AttributionSupportMethod,
    BaselineArtifacts,
    BaselineDryRunPlan,
    CommunicationMode,
    ContextGraph,
    ContrastSpec,
    CrossFitArtifacts,
    CrossFitSpec,
    Crychic,
    CrychicConfig,
    CrychicResult,
    ExpressionTransform,
    FoldTrainingSpec,
    InputSchema,
    input_schema_from_config,
    run_subject_crossfit,
    validate_anndata,
)
from .resources import (
    ResourceBundle,
    TargetPrior,
    load_cellchat_resource,
    load_cellphonedb_resource,
    load_nichenet_target_prior,
)

try:
    __version__ = version("CRYCHIC")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "AttributionSupportMethod",
    "BaselineArtifacts",
    "BaselineDryRunPlan",
    "CommunicationMode",
    "ContextGraph",
    "ContrastSpec",
    "CrossFitArtifacts",
    "CrossFitSpec",
    "Crychic",
    "CrychicConfig",
    "CrychicResult",
    "ExpressionTransform",
    "FoldTrainingSpec",
    "InputSchema",
    "ResourceBundle",
    "TargetPrior",
    "__version__",
    "input_schema_from_config",
    "load_cellchat_resource",
    "load_cellphonedb_resource",
    "load_nichenet_target_prior",
    "run_subject_crossfit",
    "validate_anndata",
]
