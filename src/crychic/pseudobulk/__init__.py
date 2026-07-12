"""Sample-aware expression aggregation and eligibility contracts."""

from .aggregation import UnitKey, aggregate_pseudobulk
from .contracts import ExploratoryAggregate, MissingnessReason, PseudobulkDataset

__all__ = [
    "ExploratoryAggregate",
    "MissingnessReason",
    "PseudobulkDataset",
    "UnitKey",
    "aggregate_pseudobulk",
]
