"""AnnData validation without mutation or eager whole-matrix conversion."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from .contracts import (
    ExpressionTransform,
    InputMode,
    InputReport,
    InputSchema,
    InputValidationError,
    ValidatedInput,
)

_NORMALIZED_TRANSFORMS = {"linear_normalized", "log1p_normalized"}


def _display(values: list[Any], *, limit: int = 5) -> str:
    shown = ", ".join(repr(value) for value in values[:limit])
    suffix = " ..." if len(values) > limit else ""
    return f"{shown}{suffix}"


def _stable_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)


def _is_canonical_metadata_scalar(value: object) -> bool:
    plain = value.item() if isinstance(value, np.generic) else value
    if isinstance(plain, (str, bool, int)):
        return True
    return isinstance(plain, float) and bool(np.isfinite(plain))


def _validate_declared_columns(obs: pd.DataFrame, schema: InputSchema) -> None:
    required = (
        schema.sample_key,
        schema.subject_key,
        schema.cell_type_key,
        *schema.context_keys,
        *schema.covariates,
    )
    missing = [key for key in required if key not in obs.columns]
    if missing:
        raise InputValidationError(
            "adata.obs is missing declared field(s) "
            f"{_display(missing)}; add them or update InputSchema"
        )

    for key in required:
        series = obs[key]
        missing_count = int(series.isna().sum())
        if missing_count:
            raise InputValidationError(
                f"adata.obs[{key!r}] contains {missing_count} missing value(s); "
                "fill or remove affected cells before validation"
            )
        if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
            empty = series.map(
                lambda value: isinstance(value, str) and not value.strip()
            )
            if bool(empty.any()):
                raise InputValidationError(
                    f"adata.obs[{key!r}] contains empty identifiers; "
                    "use explicit non-empty labels"
                )

    identifier_keys = (schema.sample_key, schema.subject_key, schema.cell_type_key)
    for key in identifier_keys:
        invalid = obs[key].map(lambda value: not isinstance(value, str))
        if bool(invalid.any()):
            examples = obs.loc[invalid, key].tolist()
            raise InputValidationError(
                f"adata.obs[{key!r}] identifiers must be strings; found "
                f"{_display(examples)}"
            )

    for key in (*schema.context_keys, *schema.covariates):
        invalid = pd.Series(
            [not _is_canonical_metadata_scalar(value) for value in obs[key]],
            index=obs.index,
            dtype=bool,
        )
        if bool(invalid.any()):
            examples = obs.loc[invalid, key].tolist()
            raise InputValidationError(
                f"adata.obs[{key!r}] context/covariate values must be canonical "
                "JSON scalars (string, boolean, integer, or finite float); found "
                f"{_display(examples)}"
            )


def _validate_sample_mapping(obs: pd.DataFrame, schema: InputSchema) -> pd.DataFrame:
    mapped_fields = (schema.subject_key, *schema.context_keys, *schema.covariates)
    grouped = obs.groupby(schema.sample_key, observed=True, sort=False, dropna=False)
    for key in mapped_fields:
        cardinality = grouped[key].nunique(dropna=False)
        invalid = cardinality[cardinality != 1]
        if not invalid.empty:
            samples = invalid.index.tolist()
            raise InputValidationError(
                f"each {schema.sample_key!r} must map to exactly one {key!r}; "
                f"conflicting sample(s): {_display(samples)}"
            )

    columns = [schema.sample_key, *mapped_fields]
    sample_metadata = (
        obs.loc[:, columns].drop_duplicates(subset=[schema.sample_key]).copy()
    )
    stable_sample_ids = pd.Series(
        [
            _stable_value(value)
            for value in sample_metadata[schema.sample_key].tolist()
        ],
        index=sample_metadata.index,
        dtype=object,
    )
    order = stable_sample_ids.sort_values(kind="stable").index
    return sample_metadata.loc[order].reset_index(drop=True)


def _matrix_kind(matrix: Any) -> str:
    if sparse.issparse(matrix):
        return str(matrix.format)
    return type(matrix).__name__


def _matrix_dtype(matrix: Any) -> str:
    dtype = getattr(matrix, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"


def _iter_matrix_blocks(
    matrix: Any, *, chunk_size: int = 4096
) -> Iterator[tuple[int, Any]]:
    if sparse.issparse(matrix) or isinstance(matrix, np.ndarray):
        yield 0, matrix
        return
    n_rows = int(matrix.shape[0])
    for start in range(0, n_rows, chunk_size):
        yield start, matrix[start : min(start + chunk_size, n_rows)]


def _numeric_values(block: Any, *, location: str) -> np.ndarray:
    if sparse.issparse(block):
        values = np.asarray(block.data)
    else:
        values = np.asarray(block)
    if not (
        np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)
    ):
        raise InputValidationError(
            f"expression matrix {location!r} has non-numeric dtype {values.dtype}"
        )
    if np.issubdtype(values.dtype, np.complexfloating):
        raise InputValidationError(
            f"expression matrix {location!r} has complex values; "
            "real expression is required"
        )
    return cast(np.ndarray, values)


def _first_coordinate(
    block: Any,
    invalid: np.ndarray,
    *,
    row_offset: int,
) -> tuple[int, int, Any]:
    if sparse.issparse(block):
        coo = block.tocoo(copy=False)
        coo_values = np.asarray(coo.data)
        invalid_index = int(np.flatnonzero(invalid)[0])
        # Conversion to COO preserves the pairing if storage order changes.
        value = coo_values[invalid_index]
        return (
            row_offset + int(coo.row[invalid_index]),
            int(coo.col[invalid_index]),
            value,
        )
    array = np.asarray(block)
    flat_index = int(np.flatnonzero(invalid)[0])
    row, column = np.unravel_index(flat_index, array.shape)
    return row_offset + int(row), int(column), array[row, column]


def _validate_matrix_values(
    matrix: Any, *, location: str, require_integer: bool
) -> None:
    if getattr(matrix, "ndim", 2) != 2:
        raise InputValidationError(
            f"expression matrix {location!r} must be two-dimensional"
        )

    for row_offset, block in _iter_matrix_blocks(matrix):
        values = _numeric_values(block, location=location)
        finite = np.isfinite(values)
        if not bool(np.all(finite)):
            row, column, value = _first_coordinate(
                block, ~finite, row_offset=row_offset
            )
            raise InputValidationError(
                f"expression matrix {location!r} contains non-finite value {value!r} "
                f"at cell {row}, gene {column}; replace or remove the affected value"
            )

        nonnegative = values >= 0
        if not bool(np.all(nonnegative)):
            row, column, value = _first_coordinate(
                block, ~nonnegative, row_offset=row_offset
            )
            raise InputValidationError(
                f"expression matrix {location!r} contains negative value {value!r} "
                f"at cell {row}, gene {column}; non-negative expression is required"
            )

        if require_integer:
            integer = np.equal(values, np.floor(values))
            if not bool(np.all(integer)):
                row, column, value = _first_coordinate(
                    block, ~integer, row_offset=row_offset
                )
                raise InputValidationError(
                    f"counts layer {location!r} contains non-integer value {value!r} "
                    f"at cell {row}, gene {column}; provide raw integer counts"
                )


def _select_expression(
    adata: AnnData, schema: InputSchema
) -> tuple[Any, InputMode, str, str, ExpressionTransform | None, bool]:
    if schema.counts_layer is not None and schema.counts_layer in adata.layers:
        location = f"layers[{schema.counts_layer!r}]"
        return (
            adata.layers[schema.counts_layer],
            InputMode.COUNTS,
            location,
            "raw_counts",
            None,
            True,
        )

    if not schema.expression_source or not schema.expression_transform:
        expected = (
            f"adata.layers[{schema.counts_layer!r}]"
            if schema.counts_layer is not None
            else "a counts layer"
        )
        raise InputValidationError(
            f"{expected} is unavailable; normalized-only input requires explicit "
            "expression_source and expression_transform"
        )
    if schema.expression_transform not in _NORMALIZED_TRANSFORMS:
        raise InputValidationError(
            f"unsupported expression_transform {schema.expression_transform!r}; "
            f"choose one of {sorted(_NORMALIZED_TRANSFORMS)}"
        )
    if schema.expression_layer is None:
        if adata.X is None:
            raise InputValidationError(
                "adata.X is empty; set expression_layer to an existing normalized layer"
            )
        matrix = adata.X
        location = "X"
    else:
        if schema.expression_layer not in adata.layers:
            raise InputValidationError(
                "normalized expression layer "
                f"{schema.expression_layer!r} is missing from adata.layers"
            )
        matrix = adata.layers[schema.expression_layer]
        location = f"layers[{schema.expression_layer!r}]"
    return (
        matrix,
        InputMode.NORMALIZED_ONLY,
        location,
        schema.expression_source,
        schema.expression_transform,
        schema.normalized_zero_is_nondetection,
    )


def _cell_type_support(obs: pd.DataFrame, schema: InputSchema) -> pd.DataFrame:
    group_keys = [*schema.context_keys, schema.cell_type_key]
    support = (
        obs.groupby(group_keys, observed=True, dropna=False, sort=False)
        .agg(
            n_cells=(schema.cell_type_key, "size"),
            n_samples=(schema.sample_key, "nunique"),
            n_subjects=(schema.subject_key, "nunique"),
        )
        .reset_index()
    )
    stable = support[group_keys].astype(str).agg("\x1f".join, axis=1)
    order = stable.sort_values(kind="stable").index
    return support.loc[order].reset_index(drop=True)


def validate_anndata(adata: AnnData, schema: InputSchema) -> ValidatedInput:
    """Validate AnnData against ``schema`` without changing caller-owned state.

    Raw counts are preferred. If the declared counts layer is absent, the caller
    must explicitly identify a normalized source and one of the supported
    transforms; that path is permanently marked ``normalized_only``.
    """

    if not isinstance(adata, AnnData):
        raise TypeError(
            f"adata must be an anndata.AnnData instance, got {type(adata)!r}"
        )
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise InputValidationError(
            f"AnnData must contain cells and genes, got shape {adata.shape}"
        )

    _validate_declared_columns(adata.obs, schema)
    sample_metadata = _validate_sample_mapping(adata.obs, schema)

    duplicate_mask = adata.var_names.duplicated(keep=False)
    duplicate_genes = tuple(
        sorted(set(map(str, adata.var_names[duplicate_mask])), key=_stable_value)
    )
    if duplicate_genes and not schema.allow_duplicate_genes:
        raise InputValidationError(
            "adata.var_names contains duplicate gene identifiers "
            f"({_display(list(duplicate_genes))}); map or aggregate duplicates "
            "explicitly, "
            "or opt in with allow_duplicate_genes=True for descriptive inspection"
        )

    matrix, mode, location, source, transform, detection_eligible = _select_expression(
        adata, schema
    )
    if tuple(matrix.shape) != tuple(adata.shape):
        raise InputValidationError(
            f"expression matrix {location!r} has shape {matrix.shape}, "
            f"expected {adata.shape}"
        )
    _validate_matrix_values(
        matrix,
        location=location,
        require_integer=mode is InputMode.COUNTS,
    )

    reason_codes = ("normalized_only",) if mode is InputMode.NORMALIZED_ONLY else ()
    warnings: list[str] = []
    if duplicate_genes:
        warnings.append("duplicate_genes_allowed")
    if mode is InputMode.NORMALIZED_ONLY and not detection_eligible:
        warnings.append("normalized_detection_unavailable")

    report = InputReport(
        mode=mode,
        reason_codes=reason_codes,
        expression_location=location,
        expression_source=source,
        expression_transform=transform,
        detection_eligible=detection_eligible,
        input_inference_eligible=mode is InputMode.COUNTS,
        n_obs=int(adata.n_obs),
        n_vars=int(adata.n_vars),
        n_samples=int(adata.obs[schema.sample_key].nunique()),
        n_subjects=int(adata.obs[schema.subject_key].nunique()),
        matrix_kind=_matrix_kind(matrix),
        matrix_dtype=_matrix_dtype(matrix),
        duplicate_genes=duplicate_genes,
        sample_metadata=sample_metadata,
        cell_type_support=_cell_type_support(adata.obs, schema),
        warnings=tuple(warnings),
    )
    return ValidatedInput(adata=adata, schema=schema, report=report, matrix=matrix)
