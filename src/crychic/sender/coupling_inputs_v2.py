"""Build auditable subject-contrast inputs for the M2 coupling prior."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from crychic.design import ContrastSpec, context_fields, node_context_fields

_EDGE_KEY = ("sender", "receiver", "interaction_id")
_ACTIVITY_COLUMNS = {
    "sample_id",
    "subject_id",
    "context_id",
    *_EDGE_KEY,
    "ligand_activity_raw",
}
_PROGRAM_COLUMNS = {
    "sample_id",
    "subject_id",
    "receiver",
    "interaction_id",
    "program_signed",
    "program_status",
}
_RESERVED_OUTPUT_COLUMNS = {
    "sample_id",
    "subject_id",
    "context_id",
    *_EDGE_KEY,
    "sender_effect",
    "receiver_effect",
}


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(sorted(values))
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ) or len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique canonical names")
    return result


def coupling_contrast_weights_v2(
    contrast: ContrastSpec,
    context_keys: Sequence[str],
) -> tuple[tuple[str, float], ...]:
    """Encode a declared contrast on the persisted context-ID axis."""

    if not isinstance(contrast, ContrastSpec):
        raise TypeError("contrast must be a ContrastSpec")
    if not contrast.estimable:
        raise ValueError("M2 requires an estimable declared contrast")
    keys = _names(context_keys, field_name="context_keys")
    if not keys:
        raise ValueError("M2 requires at least one context key")
    result = tuple(
        sorted(
            (
                node_context_fields(node, keys)[0],
                float(weight),
            )
            for node, weight in contrast.weights.items()
        )
    )
    if (
        len(result) < 2
        or len({context_id for context_id, _ in result}) != len(result)
        or any(not math.isfinite(weight) or weight == 0.0 for _, weight in result)
        or not math.isclose(sum(weight for _, weight in result), 0.0, abs_tol=1e-12)
    ):
        raise ValueError("M2 contrast contexts must be unique, finite, and balanced")
    return result


def _require_columns(
    table: pd.DataFrame,
    required: set[str],
    *,
    table_name: str,
) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{table_name} must be a pandas DataFrame")
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{table_name} is missing columns: {sorted(missing)}")
    return table.copy(deep=True)


def _numeric(table: pd.DataFrame, column: str, *, table_name: str) -> pd.Series:
    result = pd.to_numeric(table[column], errors="coerce")
    invalid = table[column].notna() & result.isna()
    if invalid.any():
        raise ValueError(f"{table_name}.{column} contains non-numeric values")
    return result.astype(float)


def _weighted_effects(
    table: pd.DataFrame,
    value_column: str,
    contrast_weights: tuple[tuple[str, float], ...],
) -> pd.DataFrame:
    index_columns = ["subject_id", *_EDGE_KEY]
    output_column = (
        "sender_effect" if value_column == "ligand_activity_raw" else "receiver_effect"
    )
    if table.empty:
        return pd.DataFrame(columns=[*index_columns, output_column])
    collapsed = (
        table.groupby([*index_columns, "context_id"], observed=True, sort=True)[
            value_column
        ]
        .mean()
        .unstack("context_id")
    )
    context_ids = [item[0] for item in contrast_weights]
    weights = pd.Series(
        [item[1] for item in contrast_weights],
        index=context_ids,
        dtype=float,
    )
    aligned = collapsed.reindex(columns=context_ids)
    complete = aligned.notna().all(axis=1)
    values = aligned.mul(weights, axis=1).sum(
        axis=1,
        min_count=len(context_ids),
    )
    values.loc[~complete] = np.nan
    return values.rename(output_column).reset_index()


def _normalized_sample_metadata(
    sample_metadata: pd.DataFrame,
    *,
    sample_key: str,
    subject_key: str,
    context_keys: tuple[str, ...],
    covariates: tuple[str, ...],
) -> pd.DataFrame:
    required = {sample_key, subject_key, *context_keys, *covariates}
    source = _require_columns(sample_metadata, required, table_name="sample_metadata")
    if source[sample_key].isna().any() or source[subject_key].isna().any():
        raise ValueError("sample metadata identifiers cannot be missing")
    if source[sample_key].duplicated().any():
        raise ValueError("sample metadata must contain one row per sample")
    source["sample_id"] = source[sample_key].astype(str)
    source["subject_id"] = source[subject_key].astype(str)
    for column in ("sample_id", "subject_id"):
        if (
            source[column].eq("").any()
            or source[column].str.strip().ne(source[column]).any()
        ):
            raise ValueError(f"sample metadata {column} values must be canonical")
    source["context_id"] = [
        context_fields(row, context_keys)[0]
        for _, row in source.loc[:, list(context_keys)].iterrows()
    ]
    return source.loc[:, ["sample_id", "subject_id", "context_id", *covariates]].copy()


def build_coupling_subject_effects_v2(
    sender_activity: pd.DataFrame,
    receiver_program: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    contrast: ContrastSpec,
    context_keys: Sequence[str],
    training_subject_ids: Sequence[str],
    sample_key: str = "sample_id",
    subject_key: str = "subject_id",
    numerical_covariates: Sequence[str] = (),
    categorical_covariates: Sequence[str] = (),
) -> pd.DataFrame:
    """Build one explicit paired contrast row per training subject and edge.

    Missing within-subject contexts remain NA. Independent-group subjects are
    therefore retained in the rectangular table but cannot masquerade as
    paired contrasts.
    """

    keys = _names(context_keys, field_name="context_keys")
    subjects = _names(training_subject_ids, field_name="training_subject_ids")
    if not subjects:
        raise ValueError("training_subject_ids cannot be empty")
    numerical = _names(numerical_covariates, field_name="numerical_covariates")
    categorical = _names(categorical_covariates, field_name="categorical_covariates")
    if set(numerical).intersection(categorical):
        raise ValueError("M2 covariate roles overlap")
    if _RESERVED_OUTPUT_COLUMNS.intersection((*numerical, *categorical)):
        raise ValueError("M2 covariates collide with subject-effect columns")
    weights = coupling_contrast_weights_v2(contrast, keys)
    context_ids = {item[0] for item in weights}

    activity = _require_columns(
        sender_activity,
        _ACTIVITY_COLUMNS,
        table_name="sender_activity",
    )
    program = _require_columns(
        receiver_program,
        _PROGRAM_COLUMNS,
        table_name="receiver_program",
    )
    for table_name, table, id_columns in (
        (
            "sender_activity",
            activity,
            ("sample_id", "subject_id", "context_id", *_EDGE_KEY),
        ),
        (
            "receiver_program",
            program,
            ("sample_id", "subject_id", "receiver", "interaction_id"),
        ),
    ):
        for column in id_columns:
            if table[column].isna().any():
                raise ValueError(f"{table_name}.{column} cannot be missing")
            table[column] = table[column].astype(str)
        if table.duplicated(list(id_columns)).any():
            raise ValueError(f"{table_name} keys must be unique")
    activity["ligand_activity_raw"] = _numeric(
        activity, "ligand_activity_raw", table_name="sender_activity"
    )
    program["program_signed"] = _numeric(
        program, "program_signed", table_name="receiver_program"
    )
    program.loc[
        ~program["program_status"].isin(["observed", "partial"]),
        "program_signed",
    ] = np.nan

    metadata = _normalized_sample_metadata(
        sample_metadata,
        sample_key=sample_key,
        subject_key=subject_key,
        context_keys=keys,
        covariates=(*numerical, *categorical),
    )
    if tuple(sorted(metadata["subject_id"].unique())) != subjects:
        raise ValueError("sample metadata must exactly cover training_subject_ids")
    sample_binding = activity.loc[
        :, ["sample_id", "subject_id", "context_id"]
    ].drop_duplicates()
    if sample_binding["sample_id"].duplicated().any():
        raise ValueError("sender activity maps one sample to multiple contexts")
    observed_binding = sample_binding.merge(
        metadata.loc[:, ["sample_id", "subject_id", "context_id"]],
        on="sample_id",
        how="left",
        suffixes=("_activity", "_metadata"),
        validate="one_to_one",
    )
    if (
        observed_binding["subject_id_metadata"].isna().any()
        or observed_binding["subject_id_activity"]
        .ne(observed_binding["subject_id_metadata"])
        .any()
        or observed_binding["context_id_activity"]
        .ne(observed_binding["context_id_metadata"])
        .any()
    ):
        raise ValueError("sender activity sample lineage differs from metadata")

    joined = activity.merge(
        program.loc[
            :,
            [
                "sample_id",
                "subject_id",
                "receiver",
                "interaction_id",
                "program_signed",
            ],
        ],
        on=["sample_id", "subject_id", "receiver", "interaction_id"],
        how="left",
        validate="many_to_one",
    )
    selected = joined.loc[joined["context_id"].isin(context_ids)].copy()
    sender_effects = _weighted_effects(selected, "ligand_activity_raw", weights)
    receiver_effects = _weighted_effects(selected, "program_signed", weights)

    edges = activity.loc[:, list(_EDGE_KEY)].drop_duplicates(ignore_index=True)
    subject_frame = pd.DataFrame({"subject_id": subjects})
    full = subject_frame.merge(edges, how="cross")
    full = full.merge(
        sender_effects,
        on=["subject_id", *_EDGE_KEY],
        how="left",
        validate="one_to_one",
    ).merge(
        receiver_effects,
        on=["subject_id", *_EDGE_KEY],
        how="left",
        validate="one_to_one",
    )

    selected_metadata = metadata.loc[metadata["context_id"].isin(context_ids)].copy()
    for column in numerical:
        selected_metadata[column] = _numeric(
            selected_metadata, column, table_name="sample_metadata"
        )
        context_means = selected_metadata.groupby(
            ["subject_id", "context_id"], observed=True, sort=True
        )[column].mean()
        numerical_subject_values = context_means.groupby("subject_id", sort=True).mean()
        full[column] = full["subject_id"].map(numerical_subject_values)
    for column in categorical:
        categorical_subject_values: dict[str, object] = {}
        for subject, group in selected_metadata.groupby(
            "subject_id", observed=True, sort=True
        ):
            values = group[column].dropna().astype(str).unique()
            categorical_subject_values[str(subject)] = (
                values[0] if len(values) == 1 else np.nan
            )
        full[column] = full["subject_id"].map(categorical_subject_values)

    columns = [
        "subject_id",
        *_EDGE_KEY,
        "sender_effect",
        "receiver_effect",
        *numerical,
        *categorical,
    ]
    return full.loc[:, columns].sort_values(
        [*_EDGE_KEY, "subject_id"],
        kind="stable",
        ignore_index=True,
    )


__all__ = [
    "build_coupling_subject_effects_v2",
    "coupling_contrast_weights_v2",
]
