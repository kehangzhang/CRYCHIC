"""Contracts for count and normalized-only sample aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
from scipy import sparse

from crychic.data import InputMode


class MissingnessReason(StrEnum):
    """State-expression eligibility for a sample/cell-type unit."""

    OBSERVED = "observed"
    LOW_CELL_COUNT = "low_cell_count"
    SAMPLING_ZERO = "sampling_zero"
    QC_FAILURE = "qc_failure"
    CONFIRMED_ABSENCE = "confirmed_absence"


_UNIT_COLUMNS = {
    "unit_id",
    "sample_id",
    "subject_id",
    "cell_type",
    "context",
    "matrix_row",
    "n_cells",
    "cell_proportion",
    "state_eligible",
    "abundance_eligible",
    "missingness_reason",
}


def _validate_contract(
    matrix: sparse.csr_matrix,
    detection: sparse.csr_matrix | None,
    unit_metadata: pd.DataFrame,
    feature_ids: tuple[str, ...],
    matrix_unit_ids: tuple[str, ...],
) -> None:
    missing_columns = _UNIT_COLUMNS.difference(unit_metadata.columns)
    if missing_columns:
        raise ValueError(f"unit_metadata is missing columns: {sorted(missing_columns)}")
    if matrix.shape != (len(matrix_unit_ids), len(feature_ids)):
        raise ValueError(
            "aggregate matrix shape must equal matrix units x features; "
            f"got {matrix.shape}, expected {(len(matrix_unit_ids), len(feature_ids))}"
        )
    if detection is not None and detection.shape != matrix.shape:
        raise ValueError("detection_fraction must match the aggregate matrix shape")
    if len(set(matrix_unit_ids)) != len(matrix_unit_ids):
        raise ValueError("matrix_unit_ids must be unique")
    if unit_metadata["unit_id"].duplicated().any():
        raise ValueError("unit_metadata unit_id values must be unique")
    known = set(unit_metadata["unit_id"])
    if not set(matrix_unit_ids).issubset(known):
        raise ValueError("every matrix_unit_id must have a unit_metadata row")


@dataclass(frozen=True, slots=True)
class PseudobulkDataset:
    """Raw-count sums for observed sample/cell-type units.

    ``unit_metadata`` contains the complete sample by cell-type grid. Units with
    no captured cells have a missing ``matrix_row`` and therefore no synthetic
    all-zero state-expression row.
    """

    counts: sparse.csr_matrix
    detection_fraction: sparse.csr_matrix
    unit_metadata: pd.DataFrame
    feature_ids: tuple[str, ...]
    matrix_unit_ids: tuple[str, ...]
    source_location: str

    def __post_init__(self) -> None:
        _validate_contract(
            self.counts,
            self.detection_fraction,
            self.unit_metadata,
            self.feature_ids,
            self.matrix_unit_ids,
        )

    @property
    def mode(self) -> InputMode:
        return InputMode.COUNTS

    @property
    def n_observed_units(self) -> int:
        return int(self.counts.shape[0])

    def state_counts(self, unit_id: str) -> sparse.csr_matrix | None:
        """Return one eligible state row, or ``None`` for missing/low-cell units."""

        rows = self.unit_metadata.loc[self.unit_metadata["unit_id"] == unit_id]
        if rows.empty:
            raise KeyError(unit_id)
        row = rows.iloc[0]
        if not bool(row["state_eligible"]) or pd.isna(row["matrix_row"]):
            return None
        return self.counts.getrow(int(row["matrix_row"]))


@dataclass(frozen=True, slots=True)
class ExploratoryAggregate:
    """Descriptive means for explicitly declared normalized-only input."""

    mean_expression: sparse.csr_matrix
    detection_fraction: sparse.csr_matrix | None
    unit_metadata: pd.DataFrame
    feature_ids: tuple[str, ...]
    matrix_unit_ids: tuple[str, ...]
    source_location: str
    expression_source: str
    expression_transform: str
    reason_codes: tuple[str, ...] = ("normalized_only",)

    def __post_init__(self) -> None:
        _validate_contract(
            self.mean_expression,
            self.detection_fraction,
            self.unit_metadata,
            self.feature_ids,
            self.matrix_unit_ids,
        )

    @property
    def mode(self) -> InputMode:
        return InputMode.NORMALIZED_ONLY

    @property
    def n_observed_units(self) -> int:
        return int(self.mean_expression.shape[0])

    def state_mean(self, unit_id: str) -> sparse.csr_matrix | None:
        """Return one eligible state row, or ``None`` for missing/low-cell units."""

        rows = self.unit_metadata.loc[self.unit_metadata["unit_id"] == unit_id]
        if rows.empty:
            raise KeyError(unit_id)
        row = rows.iloc[0]
        if not bool(row["state_eligible"]) or pd.isna(row["matrix_row"]):
            return None
        return self.mean_expression.getrow(int(row["matrix_row"]))
