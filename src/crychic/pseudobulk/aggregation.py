"""Sample-aware pseudobulk aggregation."""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

from crychic.data import InputMode, ValidatedInput

from .contracts import ExploratoryAggregate, MissingnessReason, PseudobulkDataset

UnitKey = tuple[Hashable, Hashable]


def _plain_value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _stable_value(value: Any) -> str:
    return json.dumps(
        _plain_value(value), sort_keys=True, default=str, ensure_ascii=True
    )


def _unit_id(sample: Any, context: tuple[tuple[str, Any], ...], cell_type: Any) -> str:
    payload = {
        "cell_type": _plain_value(cell_type),
        "context": [[key, _plain_value(value)] for key, value in context],
        "sample_id": _plain_value(sample),
    }
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _group_summaries(
    matrix: Any,
    group_codes: np.ndarray,
    *,
    n_groups: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Aggregate every observed group with two sparse matrix products."""

    n_cells = len(group_codes)
    group_sizes = np.bincount(group_codes, minlength=n_groups).astype(
        np.int64, copy=False
    )
    indicator = sparse.csr_matrix(
        (
            np.ones(n_cells, dtype=np.int64),
            (group_codes, np.arange(n_cells, dtype=np.int64)),
        ),
        shape=(n_groups, n_cells),
    )

    if sparse.issparse(matrix):
        values = sparse.csr_matrix(matrix)
        totals = sparse.csr_matrix(indicator @ values)
        binary = values.copy()
        binary.data = np.greater(binary.data, 0).astype(np.int8, copy=False)
        binary.eliminate_zeros()
        detected_counts = sparse.csr_matrix(indicator @ binary)
        cell_library = np.asarray(values.sum(axis=1)).ravel()
    else:
        values = np.asarray(matrix)
        totals = sparse.csr_matrix(indicator @ values)
        detected_counts = sparse.csr_matrix(indicator @ np.greater(values, 0))
        cell_library = np.asarray(values.sum(axis=1)).ravel()

    median_umi = (
        pd.DataFrame({"group_code": group_codes, "library_size": cell_library})
        .groupby("group_code", sort=True, observed=True)["library_size"]
        .median()
        .reindex(range(n_groups))
        .to_numpy(dtype=float)
    )
    return totals, detected_counts, median_umi, group_sizes


def _normalize_cell_types(
    observed: Sequence[Any], requested: Sequence[Hashable] | None
) -> tuple[Hashable, ...]:
    observed_unique = set(observed)
    if requested is None:
        selected = observed_unique
    else:
        selected = set(requested)
        if len(selected) != len(requested):
            raise ValueError("cell_types must not contain duplicates")
        missing = observed_unique.difference(selected)
        if missing:
            raise ValueError(
                "cell_types omits observed type(s): "
                + ", ".join(sorted(map(repr, missing)))
            )
    if not selected:
        raise ValueError("cell_types must contain at least one label")
    return tuple(sorted(selected, key=_stable_value))


def _normalize_missingness(
    missingness: Mapping[UnitKey, MissingnessReason | str] | None,
) -> dict[UnitKey, MissingnessReason]:
    normalized: dict[UnitKey, MissingnessReason] = {}
    for key, value in (missingness or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("missingness keys must be (sample_id, cell_type) tuples")
        try:
            reason = MissingnessReason(value)
        except ValueError as error:
            raise ValueError(
                f"unsupported missingness reason {value!r} for {key!r}"
            ) from error
        if reason not in {
            MissingnessReason.SAMPLING_ZERO,
            MissingnessReason.QC_FAILURE,
            MissingnessReason.CONFIRMED_ABSENCE,
        }:
            raise ValueError(
                f"explicit missingness for {key!r} must describe a zero-cell unit"
            )
        normalized[key] = reason
    return normalized


def aggregate_pseudobulk(
    validated: ValidatedInput,
    *,
    min_cells: int = 10,
    cell_types: Sequence[Hashable] | None = None,
    missingness: Mapping[UnitKey, MissingnessReason | str] | None = None,
) -> PseudobulkDataset | ExploratoryAggregate:
    """Aggregate expression by sample and cell type with explicit missingness.

    Count input is summed. Normalized-only input is averaged and remains a
    separately typed exploratory artifact. The output matrix contains only
    observed units; zero-cell units live only in ``unit_metadata``.
    """

    if not isinstance(min_cells, int) or isinstance(min_cells, bool) or min_cells < 1:
        raise ValueError("min_cells must be an integer >= 1")

    adata = validated.adata
    schema = validated.schema
    obs = adata.obs
    selected_cell_types = _normalize_cell_types(
        obs[schema.cell_type_key].tolist(), cell_types
    )
    explicit_missingness = _normalize_missingness(missingness)

    group_index: dict[UnitKey, int] = {}
    group_codes: npt.NDArray[np.int64] = np.empty(len(obs), dtype=np.int64)
    for position, (sample, cell_type) in enumerate(
        zip(obs[schema.sample_key], obs[schema.cell_type_key], strict=True)
    ):
        key = (sample, cell_type)
        code = group_index.setdefault(key, len(group_index))
        group_codes[position] = code

    totals, detected_counts, median_umi, group_sizes = _group_summaries(
        validated.matrix,
        group_codes,
        n_groups=len(group_index),
    )

    valid_grid_keys = {
        (row[schema.sample_key], cell_type)
        for _, row in validated.report.sample_metadata.iterrows()
        for cell_type in selected_cell_types
    }
    unknown_missingness = set(explicit_missingness).difference(valid_grid_keys)
    if unknown_missingness:
        raise ValueError(
            "missingness declares unknown sample/cell-type unit(s): "
            + ", ".join(sorted(map(repr, unknown_missingness)))
        )

    group_row_by_unit_id: dict[str, int] = {}
    metadata_rows: list[dict[str, Any]] = []
    for _, sample_row in validated.report.sample_metadata.iterrows():
        sample = sample_row[schema.sample_key]
        subject = sample_row[schema.subject_key]
        context = tuple((key, sample_row[key]) for key in schema.context_keys)
        sample_cell_count = int(
            sum(
                group_sizes[group_index[(sample, ct)]]
                for ct in selected_cell_types
                if (sample, ct) in group_index
            )
        )
        if sample_cell_count == 0:
            raise ValueError(f"sample {sample!r} has no cells in selected cell_types")

        for cell_type in selected_cell_types:
            key = (sample, cell_type)
            group_row = group_index.get(key)
            n_cells = 0 if group_row is None else int(group_sizes[group_row])
            identifier = _unit_id(sample, context, cell_type)
            if n_cells:
                if key in explicit_missingness:
                    raise ValueError(
                        f"missingness was declared for observed unit {key!r} with "
                        f"{n_cells} captured cell(s)"
                    )
                if group_row is None:  # pragma: no cover - protected by n_cells
                    raise RuntimeError("observed group is missing its sparse row")
                group_row_by_unit_id[identifier] = group_row
                reason = (
                    MissingnessReason.OBSERVED
                    if n_cells >= min_cells
                    else MissingnessReason.LOW_CELL_COUNT
                )
                state_eligible = n_cells >= min_cells
                abundance_eligible = True
            else:
                reason = explicit_missingness.get(key, MissingnessReason.SAMPLING_ZERO)
                state_eligible = False
                abundance_eligible = reason is not MissingnessReason.QC_FAILURE

            row: dict[str, Any] = {
                "unit_id": identifier,
                "sample_id": sample,
                "subject_id": subject,
                "cell_type": cell_type,
                "context": context,
                "matrix_row": pd.NA,
                "n_cells": n_cells,
                "cell_proportion": n_cells / sample_cell_count,
                "state_eligible": state_eligible,
                "abundance_eligible": abundance_eligible,
                "missingness_reason": reason.value,
                "library_size": pd.NA,
                "median_umi": pd.NA,
            }
            for context_key, context_value in context:
                # ``context`` already stores the canonical tuple. Its raw value
                # remains recoverable from that tuple when the user chose the
                # same field name.
                if context_key != "context":
                    row[context_key] = context_value
            metadata_rows.append(row)

    matrix_unit_ids = tuple(sorted(group_row_by_unit_id))
    matrix_row_by_id = {
        identifier: row for row, identifier in enumerate(matrix_unit_ids)
    }
    ordered_group_rows = np.fromiter(
        (group_row_by_unit_id[identifier] for identifier in matrix_unit_ids),
        dtype=np.int64,
        count=len(matrix_unit_ids),
    )
    ordered_sizes = group_sizes[ordered_group_rows]
    matrix = sparse.csr_matrix(totals[ordered_group_rows])
    if validated.mode is InputMode.COUNTS:
        matrix = matrix.astype(np.int64, copy=False)
    else:
        matrix = sparse.csr_matrix(
            sparse.diags(1.0 / ordered_sizes, format="csr") @ matrix
        )
    detection = sparse.csr_matrix(
        sparse.diags(1.0 / ordered_sizes, format="csr")
        @ detected_counts[ordered_group_rows]
    )
    group_library_sizes = np.asarray(totals.sum(axis=1)).ravel()

    unit_metadata = pd.DataFrame(metadata_rows).sort_values(
        "unit_id", kind="stable", ignore_index=True
    )
    unit_metadata["matrix_row"] = pd.array(
        unit_metadata["unit_id"].map(matrix_row_by_id).tolist(), dtype="Int64"
    )
    for row_index, metadata_row in unit_metadata.iterrows():
        identifier = metadata_row["unit_id"]
        group_row = group_row_by_unit_id.get(identifier)
        if group_row is None:
            continue
        if validated.mode is InputMode.COUNTS:
            unit_metadata.at[row_index, "library_size"] = int(
                group_library_sizes[group_row]
            )
            unit_metadata.at[row_index, "median_umi"] = float(median_umi[group_row])

    unit_metadata["n_cells"] = unit_metadata["n_cells"].astype(np.int64)
    unit_metadata["state_eligible"] = unit_metadata["state_eligible"].astype(bool)
    unit_metadata["abundance_eligible"] = unit_metadata["abundance_eligible"].astype(
        bool
    )

    if validated.mode is InputMode.COUNTS:
        return PseudobulkDataset(
            counts=matrix,
            detection_fraction=detection,
            unit_metadata=unit_metadata,
            feature_ids=validated.feature_ids,
            matrix_unit_ids=matrix_unit_ids,
            source_location=validated.report.expression_location,
        )

    normalized_detection = detection if validated.report.detection_eligible else None
    transform = validated.report.expression_transform
    if transform is None:  # pragma: no cover - protected by validation contract
        raise RuntimeError("normalized aggregate is missing its declared transform")
    return ExploratoryAggregate(
        mean_expression=matrix,
        detection_fraction=normalized_detection,
        unit_metadata=unit_metadata,
        feature_ids=validated.feature_ids,
        matrix_unit_ids=matrix_unit_ids,
        source_location=validated.report.expression_location,
        expression_source=validated.report.expression_source,
        expression_transform=transform,
        reason_codes=validated.report.reason_codes,
    )
