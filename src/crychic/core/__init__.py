"""Foundational contracts for CRYCHIC runtime modules."""

from .config import CrychicConfig
from .enums import CommunicationMode, ResultStatus
from .errors import (
    ConfigurationError,
    ContractError,
    CrychicError,
    FeatureUnavailableError,
)
from .ids import stable_id
from .protocols import SupportsCanonicalDict
from .provenance import RunProvenance
from .seed import SeedLineage
from .serialization import canonical_digest, canonical_json
from .version import (
    CONFIG_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    SchemaVersion,
)

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "CommunicationMode",
    "ConfigurationError",
    "ContractError",
    "CrychicConfig",
    "CrychicError",
    "FeatureUnavailableError",
    "ResultStatus",
    "RunProvenance",
    "SchemaVersion",
    "SeedLineage",
    "SupportsCanonicalDict",
    "canonical_digest",
    "canonical_json",
    "stable_id",
]
