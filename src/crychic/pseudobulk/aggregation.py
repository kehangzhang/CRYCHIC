"""Sample-aware pseudobulk aggregation."""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
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


def _row_summary(
    matrix: Any, indices: list[int]
) -> tuple[np.ndarray, np.ndarray, float]:
    block = matrix[indices, :]
    if sparse.issparse(block):
        block = block.tocsr()
        total = np.asarray(block.sum(axis=0)).ravel()
        detected = np.asarray((block > 0).sum(axis=0)).ravel() / len(indices)
        per_cell = np.asarray(block.sum(axis=1)).ravel()
    else:
        array = np.asarray(block)
        total = np.asarray(array.sum(axis=0)).ravel()
        detected = np.asarray((array > 0).mean(axis=0)).ravel()
        per_cell = np.asarray(array.sum(axis=1)).ravel()
    return total, detected, float(np.median(per_cell))


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

    groups: dict[UnitKey, list[int]] = {}
    for position, (sample, cell_type) in enumerate(
        zip(obs[schema.sample_key], obs[schema.cell_type_key], strict=True)
    ):
        groups.setdefault((sample, cell_type), []).append(position)

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

    summaries: dict[str, tuple[np.ndarray, np.ndarray, float, int]] = {}
    metadata_rows: list[dict[str, Any]] = []
    for _, sample_row in validated.report.sample_metadata.iterrows():
        sample = sample_row[schema.sample_key]
        subject = sample_row[schema.subject_key]
        context = tuple((key, sample_row[key]) for key in schema.context_keys)
        sample_cell_count = int(
            sum(len(groups.get((sample, ct), ())) for ct in selected_cell_types)
        )
        if sample_cell_count == 0:
            raise ValueError(f"sample {sample!r} has no cells in selected cell_types")

        for cell_type in selected_cell_types:
            key = (sample, cell_type)
            indices = groups.get(key, [])
            n_cells = len(indices)
            identifier = _unit_id(sample, context, cell_type)
            if n_cells:
                if key in explicit_missingness:
                    raise ValueError(
                        f"missingness was declared for observed unit {key!r} with "
                        f"{n_cells} captured cell(s)"
                    )
                total, detected, median_total = _row_summary(validated.matrix, indices)
                summaries[identifier] = (total, detected, median_total, n_cells)
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

    matrix_unit_ids = tuple(sorted(summaries))
    matrix_row_by_id = {
        identifier: row for row, identifier in enumerate(matrix_unit_ids)
    }
    aggregate_rows: list[sparse.csr_matrix] = []
    detection_rows: list[sparse.csr_matrix] = []
    for identifier in matrix_unit_ids:
        total, detected, _, n_cells = summaries[identifier]
        values: np.ndarray
        if validated.mode is InputMode.COUNTS:
            values = total.astype(np.int64, copy=False)
        else:
            values = total / n_cells
        aggregate_rows.append(sparse.csr_matrix(values.reshape(1, -1)))
        detection_rows.append(sparse.csr_matrix(detected.reshape(1, -1)))

    matrix = sparse.vstack(aggregate_rows, format="csr")
    detection = sparse.vstack(detection_rows, format="csr")

    unit_metadata = pd.DataFrame(metadata_rows).sort_values(
        "unit_id", kind="stable", ignore_index=True
    )
    unit_metadata["matrix_row"] = pd.array(
        unit_metadata["unit_id"].map(matrix_row_by_id).tolist(), dtype="Int64"
    )
    for row_index, metadata_row in unit_metadata.iterrows():
        identifier = metadata_row["unit_id"]
        if identifier not in summaries:
            continue
        total, _, median_total, _ = summaries[identifier]
        if validated.mode is InputMode.COUNTS:
            unit_metadata.at[row_index, "library_size"] = int(total.sum())
            unit_metadata.at[row_index, "median_umi"] = median_total

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
