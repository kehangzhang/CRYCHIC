"""Shared subject-level effect model for the D-common benchmark arm."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crychic.core import stable_id

from .multicondition import (
    EDGE_KEYS,
    METHOD_IDENTITY_KEYS,
    _normalize_subject_context,
    _prepared_score_table,
    _select_contrast,
)

_MODEL_VERSION = "subject_ols_batch_adjusted_v1"


def _valid_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(f"{field_name} must contain non-empty field names")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must be unique")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class DCommonSpec:
    """Frozen formula and support policy shared by every benchmark method."""

    reference: str
    target: str
    batch_keys: tuple[str, ...] = ()
    min_subjects_per_group: int = 3
    schema_version: str = "1.0.0"
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in ("reference", "target"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must be a non-empty context label")
        if self.reference == self.target:
            raise ValueError("reference and target contexts must differ")
        batch_keys = _valid_names(self.batch_keys, field_name="batch_keys")
        minimum = self.min_subjects_per_group
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
            raise ValueError("min_subjects_per_group must be an integer >= 2")
        if self.schema_version != "1.0.0":
            raise ValueError("DCommonSpec schema_version must be 1.0.0")
        payload = {
            "batch_keys": list(batch_keys),
            "min_subjects_per_group": minimum,
            "model_version": _MODEL_VERSION,
            "reference": self.reference,
            "schema_version": self.schema_version,
            "target": self.target,
        }
        object.__setattr__(self, "batch_keys", batch_keys)
        object.__setattr__(
            self,
            "spec_id",
            stable_id("d_common_spec", payload, schema_version=self.schema_version),
        )


@dataclass(frozen=True, slots=True)
class _FrozenDesign:
    table: pd.DataFrame
    design_matrix: np.ndarray
    model_columns: tuple[str, ...]
    effect_column: int
    design_kind: str
    batch_levels: tuple[tuple[str, tuple[str, ...]], ...]
    model_id: str
    mixed_subject_design: bool


def _sample_design(
    scores: pd.DataFrame,
    sample_design: pd.DataFrame,
    spec: DCommonSpec,
) -> pd.DataFrame:
    required = {"sample_id", "subject_id", "context", *spec.batch_keys}
    missing = required.difference(sample_design.columns)
    if missing:
        raise ValueError(f"sample_design is missing columns: {sorted(missing)}")
    design = sample_design.loc[:, sorted(required)].copy()
    if design.empty or design.isna().any().any():
        raise ValueError("sample_design fields must be complete")
    for column in required:
        design[column] = design[column].astype(str)
        if (
            design[column].str.strip().ne(design[column]).any()
            or design[column].eq("").any()
        ):
            raise ValueError("sample_design fields must be non-empty canonical strings")
    if design["sample_id"].duplicated().any():
        raise ValueError("sample_design must contain one row per sample_id")
    score_map = scores.loc[:, ["sample_id", "subject_id", "context"]].drop_duplicates()
    if score_map["sample_id"].duplicated().any():
        raise ValueError("scores map a sample_id to multiple subject/context values")
    mapped = score_map.merge(
        design,
        on="sample_id",
        how="left",
        suffixes=("_score", "_design"),
        validate="one_to_one",
    )
    if (
        mapped[list(spec.batch_keys)].isna().any().any()
        or mapped["subject_id_design"].isna().any()
    ):
        raise ValueError("sample_design does not cover every score sample")
    if (
        not mapped["subject_id_score"].eq(mapped["subject_id_design"]).all()
        or not mapped["context_score"].eq(mapped["context_design"]).all()
    ):
        raise ValueError("sample_design subject/context mapping disagrees with scores")
    return design.loc[design["sample_id"].isin(score_map["sample_id"])].reset_index(
        drop=True
    )


def _subject_context_design(design: pd.DataFrame, spec: DCommonSpec) -> pd.DataFrame:
    selected = design.loc[design["context"].isin((spec.reference, spec.target))]
    grouping = ["subject_id", "context"]
    if selected.empty:
        raise ValueError("sample_design contains neither requested context")
    for key in spec.batch_keys:
        counts = selected.groupby(grouping, observed=True, sort=False)[key].nunique()
        if counts.gt(1).any():
            raise ValueError(
                f"batch field {key!r} varies within a subject/context; "
                "D-common requires a subject-level batch value"
            )
    return selected.groupby(grouping, observed=True, sort=False).first().reset_index()


def _batch_encoding(
    table: pd.DataFrame, batch_keys: tuple[str, ...]
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    frozen_levels: list[tuple[str, tuple[str, ...]]] = []
    for key in batch_keys:
        levels = tuple(sorted(table[key].astype(str).unique()))
        frozen_levels.append((key, levels))
        for level in levels[1:]:
            columns.append(table[key].eq(level).to_numpy(dtype=float))
            names.append(f"batch:{key}={level}")
    if not columns:
        return np.empty((len(table), 0)), (), tuple(frozen_levels)
    return np.column_stack(columns), tuple(names), tuple(frozen_levels)


def _freeze_design(design: pd.DataFrame, spec: DCommonSpec) -> _FrozenDesign:
    subject_context = _subject_context_design(design, spec)
    context_counts = subject_context.groupby("subject_id", observed=True)[
        "context"
    ].nunique()
    paired_subject = context_counts.eq(2)
    single_context_subject = context_counts.eq(1)
    mixed = bool(paired_subject.any() and single_context_subject.any())
    if not paired_subject.any():
        design_kind = "unpaired_subject_ols"
        ordered = subject_context.sort_values(
            ["subject_id", "context"], kind="stable", ignore_index=True
        )
        encoded, batch_names, batch_levels = _batch_encoding(ordered, spec.batch_keys)
        target_indicator = ordered["context"].eq(spec.target).to_numpy(dtype=float)
        matrix = np.column_stack((np.ones(len(ordered)), target_indicator, encoded))
        model_columns = ("intercept", "target", *batch_names)
        effect_column = 1
    elif paired_subject.all():
        design_kind = "paired_difference_ols"
        reference = subject_context.loc[
            subject_context["context"].eq(spec.reference)
        ].set_index("subject_id")
        target_design = subject_context.loc[
            subject_context["context"].eq(spec.target)
        ].set_index("subject_id")
        subjects = tuple(sorted(set(reference.index).intersection(target_design.index)))
        ordered = pd.DataFrame({"subject_id": subjects})
        deltas: list[np.ndarray] = []
        names: list[str] = []
        level_rows: list[tuple[str, tuple[str, ...]]] = []
        for key in spec.batch_keys:
            frozen = tuple(sorted(subject_context[key].astype(str).unique()))
            level_rows.append((key, frozen))
            for level in frozen[1:]:
                delta = target_design.loc[list(subjects), key].eq(level).to_numpy(
                    dtype=float
                ) - reference.loc[list(subjects), key].eq(level).to_numpy(dtype=float)
                if np.any(delta != 0):
                    deltas.append(delta)
                    names.append(f"batch_delta:{key}={level}")
        encoded = np.column_stack(deltas) if deltas else np.empty((len(subjects), 0))
        matrix = np.column_stack((np.ones(len(subjects)), encoded))
        model_columns = ("target_minus_reference_intercept", *names)
        effect_column = 0
        batch_levels = tuple(level_rows)
    else:
        design_kind = "mixed_paired_unpaired_not_supported"
        ordered = subject_context.sort_values(
            ["subject_id", "context"], kind="stable", ignore_index=True
        )
        matrix = np.empty((len(ordered), 0))
        model_columns = ()
        effect_column = 0
        batch_levels = tuple(
            (key, tuple(sorted(ordered[key].astype(str).unique())))
            for key in spec.batch_keys
        )
    manifest_rows = design.loc[
        :, ["sample_id", "subject_id", "context", *spec.batch_keys]
    ].sort_values("sample_id", kind="stable")
    model_id = stable_id(
        "d_common_effect_model",
        {
            "batch_levels": batch_levels,
            "design_kind": design_kind,
            "model_columns": model_columns,
            "model_version": _MODEL_VERSION,
            "sample_design": manifest_rows.to_dict(orient="records"),
            "spec_id": spec.spec_id,
        },
    )
    return _FrozenDesign(
        table=ordered,
        design_matrix=matrix,
        model_columns=model_columns,
        effect_column=effect_column,
        design_kind=design_kind,
        batch_levels=batch_levels,
        model_id=model_id,
        mixed_subject_design=mixed,
    )


def _fit_ols(
    matrix: np.ndarray,
    response: np.ndarray,
    *,
    effect_column: int,
) -> tuple[float, float, str | None]:
    if len(response) <= matrix.shape[1]:
        return math.nan, math.nan, "insufficient_residual_degrees_of_freedom"
    rank = int(np.linalg.matrix_rank(matrix))
    reduced = np.delete(matrix, effect_column, axis=1)
    reduced_rank = 0 if reduced.shape[1] == 0 else int(np.linalg.matrix_rank(reduced))
    if rank < matrix.shape[1] or rank != reduced_rank + 1:
        return math.nan, math.nan, "target_not_estimable_after_batch_adjustment"
    coefficients, _, _, _ = np.linalg.lstsq(matrix, response, rcond=None)
    residual = response - matrix @ coefficients
    degrees = len(response) - rank
    variance = float(np.dot(residual, residual) / degrees)
    covariance = variance * np.linalg.inv(matrix.T @ matrix)
    standard_error = math.sqrt(
        max(0.0, float(covariance[effect_column, effect_column]))
    )
    return float(coefficients[effect_column]), standard_error, None


def _not_estimable_reason(group: pd.DataFrame, fallback: str) -> str:
    if group["status"].eq("resource_unavailable").all():
        return "resource_unavailable"
    if group["status"].eq("not_supported").all():
        return "method_or_resource_not_supported"
    if group["status"].eq("failed").any():
        return "method_run_failed"
    return fallback


def d_common_edge_effects(
    table: pd.DataFrame,
    sample_design: pd.DataFrame,
    *,
    spec: DCommonSpec,
    contrast: str | None = None,
    validated: bool = False,
) -> pd.DataFrame:
    """Fit one frozen subject-level batch-adjusted model to every method/edge.

    Method scores are first converted to the benchmark's direction-normalized
    comparison strength and averaged within subject/context. The formula,
    categorical levels, sample mapping, and model identifier are shared across
    all methods. Mixed paired/unpaired designs fail closed for the future
    repeated-measures backend.
    """

    if not isinstance(spec, DCommonSpec):
        raise TypeError("spec must be a DCommonSpec")
    scores = _normalize_subject_context(
        _select_contrast(_prepared_score_table(table, validated=validated), contrast)
    )
    design = _sample_design(scores, sample_design, spec)
    frozen = _freeze_design(design, spec)
    identity_keys = [*METHOD_IDENTITY_KEYS, "contrast", *EDGE_KEYS]
    rows: list[dict[str, object]] = []
    for keys, group in scores.groupby(identity_keys, observed=True, sort=False):
        identity = dict(zip(identity_keys, keys, strict=True))
        usable = group.loc[group["comparison_eligible"]]
        values = usable.groupby(["subject_id", "context"], observed=True, sort=False)[
            "comparison_strength"
        ].mean()
        if values.empty:
            n_reference = 0
            n_target = 0
        else:
            contexts = values.index.get_level_values("context")
            subjects = values.index.get_level_values("subject_id")
            n_reference = int(subjects[contexts == spec.reference].nunique())
            n_target = int(subjects[contexts == spec.target].nunique())
        n_pairs = 0
        effect = math.nan
        standard_error = math.nan
        reason: str | None = None
        if frozen.mixed_subject_design:
            reason = "mixed_paired_unpaired_requires_repeated_measures_backend"
        elif frozen.design_kind == "paired_difference_ols":
            pivot = values.unstack("context").reindex(
                index=frozen.table["subject_id"],
                columns=[spec.reference, spec.target],
            )
            complete = pivot.notna().all(axis=1).to_numpy()
            n_pairs = int(complete.sum())
            if n_pairs < spec.min_subjects_per_group:
                reason = "insufficient_paired_subjects"
            else:
                response = (
                    pivot.loc[complete, spec.target]
                    - pivot.loc[complete, spec.reference]
                ).to_numpy(dtype=float)
                effect, standard_error, reason = _fit_ols(
                    frozen.design_matrix[complete],
                    response,
                    effect_column=frozen.effect_column,
                )
        else:
            row_index = pd.MultiIndex.from_frame(
                frozen.table.loc[:, ["subject_id", "context"]]
            )
            aligned = values.reindex(row_index)
            complete = aligned.notna().to_numpy()
            context = frozen.table["context"].to_numpy()
            n_reference = int(np.sum(complete & (context == spec.reference)))
            n_target = int(np.sum(complete & (context == spec.target)))
            if min(n_reference, n_target) < spec.min_subjects_per_group:
                reason = "insufficient_subjects_per_context"
            else:
                effect, standard_error, reason = _fit_ols(
                    frozen.design_matrix[complete],
                    aligned.loc[complete].to_numpy(dtype=float),
                    effect_column=frozen.effect_column,
                )
        if reason is not None:
            reason = _not_estimable_reason(group, reason)
        rows.append(
            identity
            | {
                "reference": spec.reference,
                "target": spec.target,
                "effect": effect,
                "effect_semantics": "target_minus_reference_comparison_strength",
                "diagnostic_standard_error": standard_error,
                "n_reference_subjects": n_reference,
                "n_target_subjects": n_target,
                "n_paired_subjects": n_pairs,
                "design_kind": frozen.design_kind,
                "model_columns": frozen.model_columns,
                "d_common_spec_id": spec.spec_id,
                "d_common_effect_model_id": frozen.model_id,
                "model_version": _MODEL_VERSION,
                "status": "exploratory" if reason is None else "not_estimable",
                "reason_code": reason,
            }
        )
    return pd.DataFrame(rows)


__all__ = ["DCommonSpec", "d_common_edge_effects"]
