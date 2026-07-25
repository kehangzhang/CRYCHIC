"""Design-aware diagnostic effects for raw v7 sample-level edge scores."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from scipy.stats import t as t_distribution

from crychic.core import canonical_digest, stable_id
from crychic.scoring import SampleEdgeScoreV2

DESIGN_AWARE_DIFFERENTIAL_VERSION = "sample_level_design_aware_differential_v1"
DESIGN_AWARE_EFFECT_COLUMNS = (
    "event_id",
    "contrast_name",
    "effect",
    "standard_error",
    "statistic",
    "diagnostic_p_value",
    "diagnostic_ci_lower",
    "diagnostic_ci_upper",
    "p_value",
    "q_value",
    "ci_lower",
    "ci_upper",
    "direction",
    "n_input_samples",
    "n_subject_context_rows",
    "n_subjects",
    "n_clusters",
    "observed_fraction",
    "design_rank",
    "residual_df",
    "condition_number",
    "covariance_method",
    "status",
    "reason_code",
    "formal_inference_allowed",
    "formal_inference_status",
    "design_spec_id",
    "source_event_digest",
    "effect_id",
)
DESIGN_AWARE_OMNIBUS_COLUMNS = (
    "event_id",
    "statistic",
    "numerator_df",
    "denominator_df",
    "diagnostic_p_value",
    "p_value",
    "q_value",
    "n_subjects",
    "status",
    "reason_code",
    "formal_inference_allowed",
    "formal_inference_status",
    "design_spec_id",
    "source_event_digest",
    "omnibus_id",
)
_FORMAL_STATUS = "requires_full_pipeline_subject_resampling"
_USABLE_SCORE_STATUSES = {"observed", "low_evidence", "structural_impossible"}
_ALL_SCORE_STATUSES = {*_USABLE_SCORE_STATUSES, "not_estimable"}
_SAMPLE_EDGE_HEADS = {
    "sender_detection_raw",
    "parent_peak_raw",
    "parent_total_raw",
    "parent_mean_raw",
    "program_signed",
}


class DifferentialDesignKind(StrEnum):
    """Supported sample/subject allocation structures."""

    INDEPENDENT_TWO_GROUP = "independent_two_group"
    INDEPENDENT_MULTI_GROUP = "independent_multi_group"
    PAIRED = "paired"
    REPEATED = "repeated"
    MULTI_COHORT = "multi_cohort"
    CONTINUOUS = "continuous"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(
    values: Sequence[str], *, field_name: str, allow_empty: bool = True
) -> tuple[str, ...]:
    normalized = tuple(_name(value, field_name=field_name) for value in values)
    if (not allow_empty and not normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must be unique and have valid cardinality")
    return tuple(sorted(normalized))


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class DifferentialContrastSpec:
    """One pre-registered categorical condition contrast."""

    name: str
    weights: tuple[tuple[str, float], ...]
    contrast_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = _name(self.name, field_name="contrast_name")
        supplied = tuple(self.weights)
        if len(supplied) < 2:
            raise ValueError("differential contrast requires at least two levels")
        weights = tuple(
            sorted(
                (
                    _name(level, field_name="condition_level"),
                    _finite(weight, field_name="contrast_weight"),
                )
                for level, weight in supplied
            )
        )
        if len({level for level, _ in weights}) != len(weights):
            raise ValueError("differential contrast levels must be unique")
        if not any(weight > 0.0 for _, weight in weights) or not any(
            weight < 0.0 for _, weight in weights
        ):
            raise ValueError("differential contrast requires both signs")
        if not math.isclose(sum(weight for _, weight in weights), 0.0, abs_tol=1e-12):
            raise ValueError("differential contrast weights must sum to zero")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "contrast_id",
            stable_id(
                "differential_design_contrast",
                {"name": name, "weights": [list(item) for item in weights]},
                schema_version="1",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "name": self.name,
            "weights": [list(item) for item in self.weights],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class DifferentialDesignSpec:
    """Unified subject-level design and diagnostic covariance policy."""

    design_kind: DifferentialDesignKind | str
    condition_column: str
    condition_levels: tuple[str, ...] = ()
    subject_column: str = "subject_id"
    sample_column: str = "sample_id"
    batch_columns: tuple[str, ...] = ()
    continuous_covariates: tuple[str, ...] = ()
    categorical_covariates: tuple[str, ...] = ()
    cohort_column: str | None = None
    contrasts: tuple[DifferentialContrastSpec, ...] = ()
    precision_weight_column: str | None = "reliability_weight"
    minimum_subjects_per_level: int = 4
    minimum_clusters_for_cr2: int = 6
    maximum_condition_number: float = 1.0e10
    schema_version: str = "1.0.0"
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        kind = DifferentialDesignKind(self.design_kind)
        condition = _name(self.condition_column, field_name="condition_column")
        subject = _name(self.subject_column, field_name="subject_column")
        sample = _name(self.sample_column, field_name="sample_column")
        batches = _names(self.batch_columns, field_name="batch_columns")
        continuous = _names(
            self.continuous_covariates, field_name="continuous_covariates"
        )
        categorical = _names(
            self.categorical_covariates, field_name="categorical_covariates"
        )
        cohort = (
            None
            if self.cohort_column is None
            else _name(self.cohort_column, field_name="cohort_column")
        )
        levels = _names(
            self.condition_levels,
            field_name="condition_levels",
            allow_empty=kind is DifferentialDesignKind.CONTINUOUS,
        )
        if kind is DifferentialDesignKind.CONTINUOUS:
            if levels or self.contrasts:
                raise ValueError(
                    "continuous design cannot declare categorical contrasts"
                )
        else:
            minimum_levels = (
                3 if kind is DifferentialDesignKind.INDEPENDENT_MULTI_GROUP else 2
            )
            if len(levels) < minimum_levels:
                raise ValueError(
                    f"{kind.value} requires at least {minimum_levels} condition levels"
                )
            if (
                kind is DifferentialDesignKind.INDEPENDENT_TWO_GROUP
                and len(levels) != 2
            ):
                raise ValueError("independent_two_group requires exactly two levels")
            if not self.contrasts:
                raise ValueError("categorical designs require pre-registered contrasts")
        supplied_contrasts = tuple(self.contrasts)
        if any(
            not isinstance(item, DifferentialContrastSpec)
            for item in supplied_contrasts
        ):
            raise TypeError("contrasts must contain DifferentialContrastSpec values")
        contrasts = tuple(sorted(supplied_contrasts, key=lambda item: item.contrast_id))
        if len({item.name for item in contrasts}) != len(contrasts):
            raise ValueError("contrast names must be unique")
        if any(
            not set(level for level, _ in item.weights).issubset(levels)
            for item in contrasts
        ):
            raise ValueError("contrast weights contain undeclared condition levels")
        if kind is DifferentialDesignKind.MULTI_COHORT and cohort is None:
            raise ValueError("multi_cohort design requires cohort_column")
        if kind is not DifferentialDesignKind.MULTI_COHORT and cohort is not None:
            raise ValueError("cohort_column is only valid for multi_cohort design")
        roles = (condition, subject, sample, *batches, *continuous, *categorical)
        if cohort is not None:
            roles = (*roles, cohort)
        if len(roles) != len(set(roles)):
            raise ValueError("differential design field roles must be distinct")
        weight_column = self.precision_weight_column
        if weight_column is not None:
            weight_column = _name(weight_column, field_name="precision_weight_column")
            if weight_column in roles:
                raise ValueError("precision_weight_column must have a unique role")
        for field_name in (
            "minimum_subjects_per_level",
            "minimum_clusters_for_cr2",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{field_name} must be an integer >= 2")
        maximum_condition_number = _finite(
            self.maximum_condition_number, field_name="maximum_condition_number"
        )
        if maximum_condition_number <= 1.0:
            raise ValueError("maximum_condition_number must be > 1")
        if self.schema_version != "1.0.0":
            raise ValueError("DifferentialDesignSpec schema_version must be 1.0.0")
        payload = {
            "batch_columns": list(batches),
            "categorical_covariates": list(categorical),
            "cohort_column": cohort,
            "condition_column": condition,
            "condition_levels": list(levels),
            "continuous_covariates": list(continuous),
            "contrasts": [item.to_dict() for item in contrasts],
            "design_kind": kind.value,
            "maximum_condition_number": maximum_condition_number,
            "minimum_clusters_for_cr2": self.minimum_clusters_for_cr2,
            "minimum_subjects_per_level": self.minimum_subjects_per_level,
            "precision_weight_column": weight_column,
            "sample_column": sample,
            "schema_version": self.schema_version,
            "subject_column": subject,
            "version": DESIGN_AWARE_DIFFERENTIAL_VERSION,
        }
        object.__setattr__(self, "design_kind", kind)
        object.__setattr__(self, "condition_column", condition)
        object.__setattr__(self, "condition_levels", levels)
        object.__setattr__(self, "subject_column", subject)
        object.__setattr__(self, "sample_column", sample)
        object.__setattr__(self, "batch_columns", batches)
        object.__setattr__(self, "continuous_covariates", continuous)
        object.__setattr__(self, "categorical_covariates", categorical)
        object.__setattr__(self, "cohort_column", cohort)
        object.__setattr__(self, "contrasts", contrasts)
        object.__setattr__(self, "precision_weight_column", weight_column)
        object.__setattr__(self, "maximum_condition_number", maximum_condition_number)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "differential_design_spec",
                payload,
                schema_version=self.schema_version,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "design_kind": DifferentialDesignKind(self.design_kind).value,
            "condition_column": self.condition_column,
            "condition_levels": list(self.condition_levels),
            "subject_column": self.subject_column,
            "sample_column": self.sample_column,
            "batch_columns": list(self.batch_columns),
            "continuous_covariates": list(self.continuous_covariates),
            "categorical_covariates": list(self.categorical_covariates),
            "cohort_column": self.cohort_column,
            "contrasts": [item.to_dict() for item in self.contrasts],
            "precision_weight_column": self.precision_weight_column,
            "minimum_subjects_per_level": self.minimum_subjects_per_level,
            "minimum_clusters_for_cr2": self.minimum_clusters_for_cr2,
            "maximum_condition_number": self.maximum_condition_number,
            "schema_version": self.schema_version,
            "version": DESIGN_AWARE_DIFFERENTIAL_VERSION,
        }


@dataclass(frozen=True, slots=True)
class DesignAwareDifferentialResult:
    """Typed diagnostic effects and omnibus tests with formal fields withheld."""

    effects: pd.DataFrame
    omnibus: pd.DataFrame
    spec: DifferentialDesignSpec

    def __post_init__(self) -> None:
        if tuple(self.effects.columns) != DESIGN_AWARE_EFFECT_COLUMNS:
            raise ValueError("design-aware effect columns are invalid")
        if tuple(self.omnibus.columns) != DESIGN_AWARE_OMNIBUS_COLUMNS:
            raise ValueError("design-aware omnibus columns are invalid")
        if self.effects["effect_id"].duplicated().any():
            raise ValueError("design-aware effect IDs must be unique")
        if self.omnibus["omnibus_id"].duplicated().any():
            raise ValueError("design-aware omnibus IDs must be unique")
        for table in (self.effects, self.omnibus):
            if table["formal_inference_allowed"].any():
                raise ValueError(
                    "analytic design-aware results cannot claim formal inference"
                )
            if table["p_value"].notna().any() or table["q_value"].notna().any():
                raise ValueError("formal p/q fields require full-pipeline resampling")
        for column in ("ci_lower", "ci_upper"):
            if self.effects[column].notna().any():
                raise ValueError("formal confidence intervals require resampling")
        object.__setattr__(self, "effects", self.effects.copy(deep=True))
        object.__setattr__(self, "omnibus", self.omnibus.copy(deep=True))


@dataclass(frozen=True, slots=True)
class _WLSFit:
    beta: np.ndarray
    covariance: np.ndarray | None
    residual_df: float
    rank: int
    condition_number: float
    covariance_method: str


def _hc3_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    bread: np.ndarray,
) -> np.ndarray:
    weighted_design = np.sqrt(weights)[:, np.newaxis] * design
    weighted_residual = np.sqrt(weights) * residual
    leverage = np.einsum(
        "ij,jk,ik->i", weighted_design, bread, weighted_design, optimize=True
    )
    if np.any(leverage >= 1.0 - 1e-10):
        raise np.linalg.LinAlgError("HC3 leverage leaves no residual space")
    adjusted = weighted_residual / (1.0 - leverage)
    meat = weighted_design.T @ (adjusted[:, np.newaxis] ** 2 * weighted_design)
    covariance = bread @ meat @ bread
    return cast(np.ndarray, 0.5 * (covariance + covariance.T))


def _cr2_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    bread: np.ndarray,
) -> np.ndarray:
    weighted_design = np.sqrt(weights)[:, np.newaxis] * design
    weighted_residual = np.sqrt(weights) * residual
    meat = np.zeros_like(bread)
    for cluster in sorted(set(clusters.tolist())):
        selected = clusters == cluster
        cluster_design = weighted_design[selected]
        cluster_residual = weighted_residual[selected]
        leverage = cluster_design @ bread @ cluster_design.T
        adjustment_matrix = np.eye(int(selected.sum())) - 0.5 * (leverage + leverage.T)
        eigenvalues, eigenvectors = np.linalg.eigh(adjustment_matrix)
        if float(eigenvalues.min()) <= 1e-10:
            raise np.linalg.LinAlgError("CR2 cluster has no residual space")
        adjustment = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        score = cluster_design.T @ (adjustment @ cluster_residual)
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    return cast(np.ndarray, 0.5 * (covariance + covariance.T))


def _fit_wls(
    design: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    *,
    use_cr2: bool,
    minimum_clusters: int,
    maximum_condition_number: float,
) -> _WLSFit | None:
    if len(outcome) <= design.shape[1] or np.any(weights <= 0.0):
        return None
    weighted_design = np.sqrt(weights)[:, np.newaxis] * design
    singular = np.linalg.svd(weighted_design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(weighted_design))
    if rank != design.shape[1] or singular[-1] <= 0.0:
        return None
    condition_number = float(singular[0] / singular[-1])
    if condition_number > maximum_condition_number:
        return None
    bread = np.linalg.inv(weighted_design.T @ weighted_design)
    beta = bread @ (weighted_design.T @ (np.sqrt(weights) * outcome))
    residual = outcome - design @ beta
    residual_df = float(len(outcome) - rank)
    covariance: np.ndarray | None
    method: str
    try:
        if use_cr2:
            cluster_count = len(set(clusters.tolist()))
            residual_df = float(cluster_count - 1)
            if cluster_count < minimum_clusters:
                covariance = None
                method = "insufficient_clusters_for_cr2_diagnostic"
            else:
                covariance = _cr2_covariance(design, residual, weights, clusters, bread)
                method = "subject_cluster_cr2_diagnostic"
        else:
            covariance = _hc3_covariance(design, residual, weights, bread)
            method = "hc3_diagnostic"
    except np.linalg.LinAlgError:
        covariance = None
        method = "diagnostic_covariance_not_estimable"
    return _WLSFit(
        beta=beta,
        covariance=covariance,
        residual_df=residual_df,
        rank=rank,
        condition_number=condition_number,
        covariance_method=method,
    )


def _event_digest(table: pd.DataFrame, spec: DifferentialDesignSpec) -> str:
    columns = [
        "event_id",
        spec.sample_column,
        spec.subject_column,
        spec.condition_column,
        "score",
        "score_status",
        "out_of_fold",
        *spec.batch_columns,
        *spec.continuous_covariates,
        *spec.categorical_covariates,
    ]
    if spec.cohort_column is not None:
        columns.append(spec.cohort_column)
    if spec.precision_weight_column is not None:
        columns.append(spec.precision_weight_column)
    ordered = table.loc[:, columns].sort_values(
        [spec.subject_column, spec.sample_column], kind="stable"
    )
    records: list[list[object]] = []
    for row in ordered.itertuples(index=False, name=None):
        record: list[object] = []
        for value in row:
            if pd.isna(value):
                record.append(None)
            elif isinstance(value, float | np.floating):
                record.append({"float_hex": float(value).hex()})
            elif isinstance(value, np.generic):
                record.append(value.item())
            else:
                record.append(value)
        records.append(record)
    return cast(str, canonical_digest(records))


def _validate_input(
    score_table: pd.DataFrame, spec: DifferentialDesignSpec
) -> pd.DataFrame:
    if not isinstance(score_table, pd.DataFrame):
        raise TypeError("score_table must be a pandas DataFrame")
    required = {
        "event_id",
        spec.sample_column,
        spec.subject_column,
        spec.condition_column,
        "score",
        "score_status",
        "out_of_fold",
        *spec.batch_columns,
        *spec.continuous_covariates,
        *spec.categorical_covariates,
    }
    if spec.cohort_column is not None:
        required.add(spec.cohort_column)
    if spec.precision_weight_column is not None:
        required.add(spec.precision_weight_column)
    missing = required.difference(score_table.columns)
    if missing:
        raise ValueError(f"score_table is missing design columns: {sorted(missing)}")
    table = score_table.copy(deep=True)
    string_columns = {
        "event_id",
        spec.sample_column,
        spec.subject_column,
        "score_status",
        *spec.batch_columns,
        *spec.categorical_covariates,
    }
    if spec.design_kind is not DifferentialDesignKind.CONTINUOUS:
        string_columns.add(spec.condition_column)
    if spec.cohort_column is not None:
        string_columns.add(spec.cohort_column)
    for column in string_columns:
        if table[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values")
        table[column] = table[column].astype(str)
        if table[column].eq("").any():
            raise ValueError(f"{column} cannot contain empty values")
    if not set(table["score_status"]).issubset(_ALL_SCORE_STATUSES):
        raise ValueError("score_status contains unsupported values")
    if table.duplicated(["event_id", spec.sample_column]).any():
        raise ValueError("score_table requires one row per event and sample")
    if (
        not table["out_of_fold"]
        .map(lambda value: type(value) in {bool, np.bool_})
        .all()
    ):
        raise ValueError("out_of_fold must contain booleans")
    if not table["out_of_fold"].astype(bool).all():
        raise ValueError("design-aware primary effects require out-of-fold scores")
    numeric_score = pd.to_numeric(table["score"], errors="coerce")
    invalid_score = table["score"].notna() & numeric_score.isna()
    if invalid_score.any() or np.isinf(numeric_score.dropna()).any():
        raise ValueError("score must contain finite numeric values or NA")
    table["score"] = numeric_score.astype(float)
    numeric_columns = list(spec.continuous_covariates)
    if spec.design_kind is DifferentialDesignKind.CONTINUOUS:
        numeric_columns.append(spec.condition_column)
    if spec.precision_weight_column is not None:
        numeric_columns.append(spec.precision_weight_column)
    for column in numeric_columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        invalid = table[column].notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric.dropna()).any():
            raise ValueError(f"{column} must contain finite numeric values or NA")
        table[column] = numeric.astype(float)
    sample_roles = [
        spec.subject_column,
        spec.condition_column,
        *spec.batch_columns,
        *spec.continuous_covariates,
        *spec.categorical_covariates,
    ]
    if spec.cohort_column is not None:
        sample_roles.append(spec.cohort_column)
    inconsistent = table.groupby(spec.sample_column, observed=True)[
        sample_roles
    ].nunique(dropna=False)
    if (inconsistent > 1).any(axis=None):
        raise ValueError("sample metadata must be invariant across event rows")
    return table


def _subject_context_table(
    event: pd.DataFrame, spec: DifferentialDesignSpec
) -> tuple[pd.DataFrame, float, str | None]:
    numeric_score = pd.to_numeric(event["score"], errors="coerce")
    usable = (
        event["score_status"].isin(_USABLE_SCORE_STATUSES)
        & numeric_score.notna()
        & np.isfinite(numeric_score)
    )
    if spec.precision_weight_column is None:
        numeric_weight = pd.Series(1.0, index=event.index)
    else:
        numeric_weight = pd.to_numeric(
            event[spec.precision_weight_column], errors="coerce"
        )
        usable &= (
            numeric_weight.notna()
            & np.isfinite(numeric_weight)
            & (numeric_weight > 0.0)
        )
    for column in spec.continuous_covariates:
        numeric_covariate = pd.to_numeric(event[column], errors="coerce")
        usable &= numeric_covariate.notna() & np.isfinite(numeric_covariate)
    if spec.design_kind is DifferentialDesignKind.CONTINUOUS:
        numeric_condition = pd.to_numeric(event[spec.condition_column], errors="coerce")
        usable &= numeric_condition.notna() & np.isfinite(numeric_condition)
    observed_fraction = float(usable.mean())
    source = event.loc[usable].copy()
    source["score"] = numeric_score.loc[usable].astype(float)
    source["_precision_weight"] = numeric_weight.loc[usable].astype(float)
    if source.empty:
        return source, observed_fraction, "no_complete_subject_context_scores"
    group_columns = [spec.subject_column, spec.condition_column]
    categorical = [
        *spec.batch_columns,
        *spec.categorical_covariates,
        *(() if spec.cohort_column is None else (spec.cohort_column,)),
    ]
    for column in categorical:
        variation = source.groupby(group_columns, observed=True)[column].nunique()
        if (variation > 1).any():
            return (
                pd.DataFrame(),
                observed_fraction,
                f"{column}_varies_within_subject_context",
            )
    aggregation: dict[str, str] = {
        "score": "mean",
        "_precision_weight": "mean",
    }
    aggregation.update({column: "mean" for column in spec.continuous_covariates})
    aggregation.update({column: "first" for column in categorical})
    grouped = (
        source.groupby(group_columns, observed=True, sort=True)
        .agg(aggregation)
        .reset_index()
    )
    return cast(pd.DataFrame, grouped), observed_fraction, None


def _categorical_design(
    table: pd.DataFrame, spec: DifferentialDesignSpec
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    levels = spec.condition_levels
    condition = table[spec.condition_column].astype(str)
    parts: list[np.ndarray] = [
        condition.eq(level).to_numpy(dtype=float)[:, np.newaxis] for level in levels
    ]
    columns = [f"condition:{level}" for level in levels]
    for column in spec.continuous_covariates:
        values = table[column].to_numpy(dtype=float)
        centered = values - float(values.mean())
        scale = float(np.std(centered, ddof=0))
        if scale > 0.0:
            parts.append((centered / scale)[:, np.newaxis])
            columns.append(f"continuous:{column}")
    categorical = [
        *spec.batch_columns,
        *spec.categorical_covariates,
        *(() if spec.cohort_column is None else (spec.cohort_column,)),
    ]
    for column in categorical:
        values = table[column].astype(str)
        observed_levels = tuple(sorted(values.unique()))
        for level in observed_levels[1:]:
            parts.append(values.eq(level).to_numpy(dtype=float)[:, np.newaxis])
            columns.append(f"categorical:{column}={level}")
    return np.hstack(parts), tuple(columns), condition.to_numpy(dtype=str)


def _continuous_design(
    table: pd.DataFrame, spec: DifferentialDesignSpec
) -> tuple[np.ndarray, tuple[str, ...]]:
    condition = pd.to_numeric(table[spec.condition_column], errors="coerce")
    if condition.isna().any() or not np.isfinite(condition).all():
        return np.empty((len(table), 0)), ()
    centered_condition = condition.to_numpy(dtype=float) - float(condition.mean())
    parts = [np.ones((len(table), 1)), centered_condition[:, np.newaxis]]
    columns = ["intercept", f"continuous_condition:{spec.condition_column}"]
    for column in spec.continuous_covariates:
        values = table[column].to_numpy(dtype=float)
        centered = values - float(values.mean())
        scale = float(np.std(centered, ddof=0))
        if scale > 0.0:
            parts.append((centered / scale)[:, np.newaxis])
            columns.append(f"continuous:{column}")
    for column in (*spec.batch_columns, *spec.categorical_covariates):
        categorical_values = table[column].astype(str)
        for level in tuple(sorted(categorical_values.unique()))[1:]:
            parts.append(
                categorical_values.eq(level).to_numpy(dtype=float)[:, np.newaxis]
            )
            columns.append(f"categorical:{column}={level}")
    return np.hstack(parts), tuple(columns)


def _contrast_vector(
    contrast: DifferentialContrastSpec,
    columns: tuple[str, ...],
) -> np.ndarray:
    by_level = dict(contrast.weights)
    return cast(
        np.ndarray,
        np.asarray(
            [
                by_level.get(column.removeprefix("condition:"), 0.0)
                if column.startswith("condition:")
                else 0.0
                for column in columns
            ],
            dtype=float,
        ),
    )


def _effect_row(
    *,
    event_id: str,
    contrast_name: str,
    spec: DifferentialDesignSpec,
    digest: str,
    n_input_samples: int,
    n_subject_context_rows: int,
    n_subjects: int,
    n_clusters: int,
    observed_fraction: float,
    fit: _WLSFit | None,
    contrast_vector: np.ndarray | None,
    reason_code: str | None,
) -> dict[str, object]:
    if fit is None or contrast_vector is None:
        effect = standard_error = statistic = diagnostic_p = None
        diagnostic_lower = diagnostic_upper = None
        direction = None
        rank = 0
        residual_df = None
        condition_number = None
        covariance_method = "not_estimable"
        status = "not_estimable"
        reason = reason_code or "design_not_estimable"
    else:
        effect = float(contrast_vector @ fit.beta)
        rank = fit.rank
        residual_df = fit.residual_df
        condition_number = fit.condition_number
        covariance_method = fit.covariance_method
        status = "observed"
        reason = None
        direction = int(np.sign(effect))
        if fit.covariance is None:
            standard_error = statistic = diagnostic_p = None
            diagnostic_lower = diagnostic_upper = None
        else:
            variance = float(contrast_vector @ fit.covariance @ contrast_vector)
            standard_error = math.sqrt(max(0.0, variance))
            statistic = effect / standard_error if standard_error > 0.0 else None
            diagnostic_p = (
                float(2.0 * t_distribution.sf(abs(statistic), fit.residual_df))
                if statistic is not None and fit.residual_df > 0.0
                else None
            )
            critical = (
                float(t_distribution.ppf(0.975, fit.residual_df))
                if fit.residual_df > 0.0
                else math.nan
            )
            diagnostic_lower = (
                effect - critical * standard_error if math.isfinite(critical) else None
            )
            diagnostic_upper = (
                effect + critical * standard_error if math.isfinite(critical) else None
            )
    identity = {
        "event_id": event_id,
        "contrast_name": contrast_name,
        "effect": effect,
        "standard_error": standard_error,
        "statistic": statistic,
        "diagnostic_p_value": diagnostic_p,
        "diagnostic_ci_lower": diagnostic_lower,
        "diagnostic_ci_upper": diagnostic_upper,
        "p_value": None,
        "q_value": None,
        "ci_lower": None,
        "ci_upper": None,
        "direction": direction,
        "n_input_samples": n_input_samples,
        "n_subject_context_rows": n_subject_context_rows,
        "n_subjects": n_subjects,
        "n_clusters": n_clusters,
        "observed_fraction": observed_fraction,
        "design_rank": rank,
        "residual_df": residual_df,
        "condition_number": condition_number,
        "covariance_method": covariance_method,
        "status": status,
        "reason_code": reason,
        "formal_inference_allowed": False,
        "formal_inference_status": _FORMAL_STATUS,
        "design_spec_id": spec.spec_id,
        "source_event_digest": digest,
    }
    return {
        **identity,
        "effect_id": stable_id(
            "design_aware_differential_effect", identity, schema_version="1"
        ),
    }


def _omnibus_row(
    *,
    event_id: str,
    spec: DifferentialDesignSpec,
    digest: str,
    fit: _WLSFit | None,
    n_subjects: int,
    condition_columns: int,
) -> dict[str, object]:
    reason: str | None = None
    if fit is None or fit.covariance is None or condition_columns < 2:
        statistic = numerator_df = denominator_df = diagnostic_p = None
        status = "not_estimable"
        reason = "omnibus_design_or_covariance_not_estimable"
    else:
        contrast = np.zeros((condition_columns - 1, len(fit.beta)))
        for index in range(1, condition_columns):
            contrast[index - 1, index] = 1.0
            contrast[index - 1, 0] = -1.0
        difference = contrast @ fit.beta
        covariance = contrast @ fit.covariance @ contrast.T
        try:
            statistic = float(
                difference
                @ np.linalg.pinv(covariance)
                @ difference
                / (condition_columns - 1)
            )
            numerator_df = float(condition_columns - 1)
            denominator_df = fit.residual_df
            diagnostic_p = float(
                f_distribution.sf(statistic, numerator_df, denominator_df)
            )
            status = "observed"
        except (ValueError, np.linalg.LinAlgError):
            statistic = numerator_df = denominator_df = diagnostic_p = None
            status = "not_estimable"
            reason = "omnibus_covariance_singular"
    identity = {
        "event_id": event_id,
        "statistic": statistic,
        "numerator_df": numerator_df,
        "denominator_df": denominator_df,
        "diagnostic_p_value": diagnostic_p,
        "p_value": None,
        "q_value": None,
        "n_subjects": n_subjects,
        "status": status,
        "reason_code": reason,
        "formal_inference_allowed": False,
        "formal_inference_status": _FORMAL_STATUS,
        "design_spec_id": spec.spec_id,
        "source_event_digest": digest,
    }
    return {
        **identity,
        "omnibus_id": stable_id(
            "design_aware_differential_omnibus", identity, schema_version="1"
        ),
    }


def _categorical_event_rows(
    event_id: str,
    event: pd.DataFrame,
    spec: DifferentialDesignSpec,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    digest = _event_digest(event, spec)
    subject_table, observed_fraction, table_reason = _subject_context_table(event, spec)
    n_input = len(event)
    n_subjects = (
        int(subject_table[spec.subject_column].nunique())
        if not subject_table.empty
        else 0
    )
    reason: str | None = table_reason
    fit: _WLSFit | None = None
    columns: tuple[str, ...] = ()
    if subject_table.empty:
        reason = reason or "no_complete_subject_context_scores"
    elif set(subject_table[spec.condition_column].astype(str)) != set(
        spec.condition_levels
    ):
        reason = "condition_level_universe_mismatch"
    else:
        counts = subject_table.groupby(spec.condition_column, observed=True)[
            spec.subject_column
        ].nunique()
        if any(
            int(counts.get(level, 0)) < spec.minimum_subjects_per_level
            for level in spec.condition_levels
        ):
            reason = "insufficient_subjects_per_condition"
        if (
            spec.design_kind is DifferentialDesignKind.MULTI_COHORT
            and spec.cohort_column is not None
            and subject_table[spec.cohort_column].nunique() < 2
        ):
            reason = "insufficient_observed_cohorts"
        independent = spec.design_kind in {
            DifferentialDesignKind.INDEPENDENT_TWO_GROUP,
            DifferentialDesignKind.INDEPENDENT_MULTI_GROUP,
            DifferentialDesignKind.MULTI_COHORT,
        }
        subject_condition_counts = subject_table.groupby(
            spec.subject_column, observed=True
        )[spec.condition_column].nunique()
        if independent and (subject_condition_counts > 1).any():
            reason = "independent_design_reuses_subject_across_conditions"
        if spec.design_kind is DifferentialDesignKind.PAIRED:
            return _paired_event_rows(
                event_id,
                event,
                subject_table,
                observed_fraction,
                digest,
                spec,
            )
        if reason is None:
            design, columns, _ = _categorical_design(subject_table, spec)
            fit = _fit_wls(
                design,
                subject_table["score"].to_numpy(dtype=float),
                subject_table["_precision_weight"].to_numpy(dtype=float),
                subject_table[spec.subject_column].astype(str).to_numpy(),
                use_cr2=spec.design_kind is DifferentialDesignKind.REPEATED,
                minimum_clusters=spec.minimum_clusters_for_cr2,
                maximum_condition_number=spec.maximum_condition_number,
            )
            if fit is None:
                reason = "rank_deficient_or_ill_conditioned_design"
    effect_rows = [
        _effect_row(
            event_id=event_id,
            contrast_name=contrast.name,
            spec=spec,
            digest=digest,
            n_input_samples=n_input,
            n_subject_context_rows=len(subject_table),
            n_subjects=n_subjects,
            n_clusters=n_subjects,
            observed_fraction=observed_fraction,
            fit=fit,
            contrast_vector=(
                None if fit is None else _contrast_vector(contrast, columns)
            ),
            reason_code=reason,
        )
        for contrast in spec.contrasts
    ]
    omnibus = _omnibus_row(
        event_id=event_id,
        spec=spec,
        digest=digest,
        fit=fit,
        n_subjects=n_subjects,
        condition_columns=len(spec.condition_levels),
    )
    return effect_rows, omnibus


def _paired_event_rows(
    event_id: str,
    event: pd.DataFrame,
    subject_table: pd.DataFrame,
    observed_fraction: float,
    digest: str,
    spec: DifferentialDesignSpec,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    score_matrix = subject_table.pivot(
        index=spec.subject_column,
        columns=spec.condition_column,
        values="score",
    ).reindex(columns=spec.condition_levels)
    precision_matrix = subject_table.pivot(
        index=spec.subject_column,
        columns=spec.condition_column,
        values="_precision_weight",
    ).reindex(columns=spec.condition_levels)
    complete_index = score_matrix.dropna().index.intersection(
        precision_matrix.dropna().index
    )
    complete = score_matrix.loc[complete_index]
    complete_precision = precision_matrix.loc[complete_index]
    n_subjects = len(complete)
    effect_rows: list[dict[str, object]] = []
    for contrast in spec.contrasts:
        weights = np.asarray(
            [dict(contrast.weights).get(level, 0.0) for level in spec.condition_levels]
        )
        paired_effect = complete.to_numpy(dtype=float) @ weights
        paired_variance = (
            np.square(weights)[np.newaxis, :] / complete_precision.to_numpy(dtype=float)
        ).sum(axis=1)
        paired_precision = np.reciprocal(paired_variance)
        reason: str | None
        if n_subjects < spec.minimum_subjects_per_level:
            fit = None
            vector = None
            reason = "insufficient_complete_pairs"
        else:
            design = np.ones((n_subjects, 1))
            fit = _fit_wls(
                design,
                paired_effect,
                paired_precision,
                complete.index.astype(str).to_numpy(),
                use_cr2=False,
                minimum_clusters=spec.minimum_clusters_for_cr2,
                maximum_condition_number=spec.maximum_condition_number,
            )
            vector = np.ones(1) if fit is not None else None
            reason = None if fit is not None else "paired_mean_not_estimable"
        effect_rows.append(
            _effect_row(
                event_id=event_id,
                contrast_name=contrast.name,
                spec=spec,
                digest=digest,
                n_input_samples=len(event),
                n_subject_context_rows=len(subject_table),
                n_subjects=n_subjects,
                n_clusters=n_subjects,
                observed_fraction=observed_fraction,
                fit=fit,
                contrast_vector=vector,
                reason_code=reason,
            )
        )
    omnibus = _omnibus_row(
        event_id=event_id,
        spec=spec,
        digest=digest,
        fit=None,
        n_subjects=n_subjects,
        condition_columns=len(spec.condition_levels),
    )
    return effect_rows, omnibus


def _continuous_event_rows(
    event_id: str,
    event: pd.DataFrame,
    spec: DifferentialDesignSpec,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    digest = _event_digest(event, spec)
    subject_table, observed_fraction, table_reason = _subject_context_table(event, spec)
    n_subjects = (
        int(subject_table[spec.subject_column].nunique())
        if not subject_table.empty
        else 0
    )
    design, columns = (
        (np.empty((len(subject_table), 0)), ())
        if table_reason is not None
        else _continuous_design(subject_table, spec)
    )
    fit = (
        None
        if not columns
        else _fit_wls(
            design,
            subject_table["score"].to_numpy(dtype=float),
            subject_table["_precision_weight"].to_numpy(dtype=float),
            subject_table[spec.subject_column].astype(str).to_numpy(),
            use_cr2=(
                subject_table.groupby(spec.subject_column, observed=True).size().max()
                > 1
            ),
            minimum_clusters=spec.minimum_clusters_for_cr2,
            maximum_condition_number=spec.maximum_condition_number,
        )
    )
    vector = (
        None if fit is None else np.asarray([0.0, 1.0, *([0.0] * (len(columns) - 2))])
    )
    effect = _effect_row(
        event_id=event_id,
        contrast_name=f"slope:{spec.condition_column}",
        spec=spec,
        digest=digest,
        n_input_samples=len(event),
        n_subject_context_rows=len(subject_table),
        n_subjects=n_subjects,
        n_clusters=n_subjects,
        observed_fraction=observed_fraction,
        fit=fit,
        contrast_vector=vector,
        reason_code=(
            None
            if fit is not None
            else table_reason or "continuous_design_not_estimable"
        ),
    )
    omnibus = _omnibus_row(
        event_id=event_id,
        spec=spec,
        digest=digest,
        fit=None,
        n_subjects=n_subjects,
        condition_columns=0,
    )
    return [effect], omnibus


def fit_design_aware_differential(
    score_table: pd.DataFrame,
    spec: DifferentialDesignSpec,
) -> DesignAwareDifferentialResult:
    """Fit raw-score effects while withholding formal inference fields."""

    if not isinstance(spec, DifferentialDesignSpec):
        raise TypeError("spec must be DifferentialDesignSpec")
    table = _validate_input(score_table, spec)
    effect_rows: list[dict[str, object]] = []
    omnibus_rows: list[dict[str, object]] = []
    for event_id, event in table.groupby("event_id", observed=True, sort=True):
        if spec.design_kind is DifferentialDesignKind.CONTINUOUS:
            event_effects, event_omnibus = _continuous_event_rows(
                str(event_id), event, spec
            )
        else:
            event_effects, event_omnibus = _categorical_event_rows(
                str(event_id), event, spec
            )
        effect_rows.extend(event_effects)
        omnibus_rows.append(event_omnibus)
    effect_table = pd.DataFrame(effect_rows, columns=DESIGN_AWARE_EFFECT_COLUMNS)
    omnibus_table = pd.DataFrame(omnibus_rows, columns=DESIGN_AWARE_OMNIBUS_COLUMNS)
    return DesignAwareDifferentialResult(
        effects=effect_table,
        omnibus=omnibus_table,
        spec=spec,
    )


def build_sample_edge_differential_input(
    scores: SampleEdgeScoreV2,
    *,
    score_head: str,
    sample_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project one v2 raw head to the common design-aware event input grain."""

    if not isinstance(scores, SampleEdgeScoreV2):
        raise TypeError("scores must be SampleEdgeScoreV2")
    if score_head not in _SAMPLE_EDGE_HEADS:
        raise ValueError(f"score_head must be one of {sorted(_SAMPLE_EDGE_HEADS)}")
    table = scores.table.copy(deep=True)
    if score_head == "sender_detection_raw":
        identity_columns: tuple[str, ...] = (
            "sender",
            "receiver",
            "interaction_id",
        )
    else:
        identity_columns = ("receiver", "interaction_id")
        parent_key = [
            "sample_id",
            "context_id",
            "fold_id",
            *identity_columns,
        ]
        grouped_status = table.groupby(parent_key, observed=True)["status"]
        any_observed = grouped_status.transform(
            lambda values: values.eq("observed").any()
        )
        any_low_evidence = grouped_status.transform(
            lambda values: values.eq("low_evidence").any()
        )
        any_not_estimable = grouped_status.transform(
            lambda values: values.eq("not_estimable").any()
        )
        table["_parent_score_status"] = np.select(
            [any_observed, any_low_evidence | any_not_estimable],
            ["observed", "low_evidence"],
            default="structural_impossible",
        )
        if score_head == "program_signed":
            program_consistency = table.groupby(
                parent_key,
                observed=True,
                sort=False,
            ).agg(
                program_value_count=(
                    "program_signed",
                    lambda values: values.nunique(dropna=False),
                ),
                program_status_count=(
                    "program_status",
                    lambda values: values.nunique(dropna=False),
                ),
            )
            if (
                program_consistency["program_value_count"].gt(1).any()
                or program_consistency["program_status_count"].gt(1).any()
            ):
                raise ValueError(
                    "program_signed must be sender-invariant within each parent event"
                )
        table = table.drop_duplicates(
            ["sample_id", "context_id", "fold_id", *identity_columns]
        ).copy()
    identity_rows = tuple(
        tuple(str(value) for value in values)
        for values in table.loc[:, list(identity_columns)].itertuples(
            index=False,
            name=None,
        )
    )
    event_id_by_identity = {
        values: stable_id(
            "sample_edge_differential_event",
            {
                "score_head": score_head,
                **dict(zip(identity_columns, values, strict=True)),
            },
            schema_version="1",
        )
        for values in dict.fromkeys(identity_rows)
    }
    table["event_id"] = [event_id_by_identity[values] for values in identity_rows]
    table["score"] = table[score_head]
    if score_head == "sender_detection_raw":
        table["score_status"] = np.where(
            table["score"].notna(), table["status"], "not_estimable"
        )
    elif score_head == "program_signed":
        table["score_status"] = table["program_status"].map(
            {
                "observed": "observed",
                "partial": "low_evidence",
                "not_estimable": "not_estimable",
                "not_computed": "not_estimable",
            }
        )
        table["score_status"] = np.where(
            table["score"].notna(), table["score_status"], "not_estimable"
        )
    else:
        table["score_status"] = np.where(
            table["score"].notna(),
            table["_parent_score_status"],
            "not_estimable",
        )
    columns = [
        "event_id",
        "sample_id",
        "subject_id",
        "condition",
        "fold_id",
        "score",
        "score_status",
        "reliability_weight",
        "out_of_fold",
        *identity_columns,
    ]
    result = table.loc[:, columns]
    if sample_metadata is not None:
        if not isinstance(sample_metadata, pd.DataFrame):
            raise TypeError("sample_metadata must be a pandas DataFrame or None")
        if (
            "sample_id" not in sample_metadata
            or sample_metadata["sample_id"].duplicated().any()
        ):
            raise ValueError("sample_metadata requires unique sample_id values")
        if any(not isinstance(column, str) for column in sample_metadata.columns):
            raise ValueError("sample_metadata columns must be strings")
        extra: list[str] = [
            str(column) for column in sample_metadata if column != "sample_id"
        ]
        overlap = set(extra).intersection(str(column) for column in result.columns)
        if overlap:
            raise ValueError(
                f"sample_metadata duplicates score fields: {sorted(overlap)}"
            )
        result = result.merge(
            sample_metadata,
            on="sample_id",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    return cast(pd.DataFrame, result.reset_index(drop=True))


__all__ = [
    "DESIGN_AWARE_DIFFERENTIAL_VERSION",
    "DESIGN_AWARE_EFFECT_COLUMNS",
    "DESIGN_AWARE_OMNIBUS_COLUMNS",
    "DesignAwareDifferentialResult",
    "DifferentialContrastSpec",
    "DifferentialDesignKind",
    "DifferentialDesignSpec",
    "build_sample_edge_differential_input",
    "fit_design_aware_differential",
]
