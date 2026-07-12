"""CRYCHIC public Python interface."""

from importlib.metadata import PackageNotFoundError, version

from .api import (
    AttributionSupportMethod,
    BaselineArtifacts,
    BaselineDryRunPlan,
    CommunicationMode,
    ContextGraph,
    ContrastSpec,
    Crychic,
    CrychicConfig,
    CrychicResult,
    ExpressionTransform,
    InputSchema,
    input_schema_from_config,
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
    "Crychic",
    "CrychicConfig",
    "CrychicResult",
    "ExpressionTransform",
    "InputSchema",
    "ResourceBundle",
    "TargetPrior",
    "__version__",
    "input_schema_from_config",
    "load_cellchat_resource",
    "load_cellphonedb_resource",
    "load_nichenet_target_prior",
    "validate_anndata",
]
