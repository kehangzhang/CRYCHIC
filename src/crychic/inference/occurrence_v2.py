"""Design-aware two-part occurrence and conditional-intensity inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest, fisher_exact, norm

from crychic.core import canonical_digest, stable_id
from crychic.scoring import SampleEdgeScoreV2

from .design_aware import (
    DifferentialContrastSpec,
    DifferentialDesignKind,
    DifferentialDesignSpec,
    build_sample_edge_differential_input,
    fit_design_aware_differential,
)

TWO_PART_OCCURRENCE_VERSION = "fixed_threshold_design_aware_m4_v2"
TWO_PART_OCCURRENCE_EFFECT_COLUMNS = (
    "event_id",
    "contrast_name",
    "reference",
    "target",
    "reference_subjects",
    "target_subjects",
    "complete_pairs",
    "reference_occurrences",
    "target_occurrences",
    "reference_prevalence",
    "target_prevalence",
    "prevalence_difference",
    "odds_ratio",
    "log_odds_ratio",
    "standard_error",
    "statistic",
    "p_value",
    "q_value",
    "ci_lower",
    "ci_upper",
    "occurrence_method",
    "conditional_intensity_effect",
    "conditional_intensity_standard_error",
    "conditional_intensity_diagnostic_p_value",
    "conditional_intensity_status",
    "n_subject_context_rows",
    "observed_fraction",
    "activity_head",
    "activity_threshold_raw",
    "status",
    "reason_code",
    "formal_inference_allowed",
    "formal_inference_scope",
    "design_spec_id",
    "occurrence_spec_id",
    "source_event_digest",
    "effect_id",
)
_USABLE_STATUSES = {"observed", "low_evidence", "structural_impossible"}
_SCHEMA_VERSION = "2.0.0"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class TwoPartOccurrenceV2Spec:
    """Frozen activity threshold and design for the M4 occurrence channel."""

    design: DifferentialDesignSpec
    activity_head: str = "parent_mean_raw"
    activity_threshold_raw: float = 1.0
    probability_transition_scale: float = 0.25
    minimum_discordant_pairs: int = 0
    fdr_scope: str = "global_event_contrast_bh"
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.design, DifferentialDesignSpec):
            raise TypeError("design must be DifferentialDesignSpec")
        activity_head = _name(self.activity_head, field_name="activity_head")
        if activity_head not in {
            "sender_detection_raw",
            "parent_peak_raw",
            "parent_total_raw",
            "parent_mean_raw",
        }:
            raise ValueError("activity_head is not a raw v2 communication head")
        threshold = _finite(
            self.activity_threshold_raw, field_name="activity_threshold_raw"
        )
        transition = _finite(
            self.probability_transition_scale,
            field_name="probability_transition_scale",
        )
        if transition <= 0.0:
            raise ValueError("probability_transition_scale must be positive")
        minimum = self.minimum_discordant_pairs
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise ValueError("minimum_discordant_pairs must be an integer >= 0")
        if self.fdr_scope != "global_event_contrast_bh":
            raise ValueError("fdr_scope is unsupported")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "activity_head", activity_head)
        object.__setattr__(self, "activity_threshold_raw", threshold)
        object.__setattr__(self, "probability_transition_scale", transition)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "two_part_occurrence_v2_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "activity_head": self.activity_head,
            "activity_threshold_raw": self.activity_threshold_raw,
            "design_spec_id": self.design.spec_id,
            "fdr_scope": self.fdr_scope,
            "minimum_discordant_pairs": self.minimum_discordant_pairs,
            "probability_transition_scale": self.probability_transition_scale,
            "schema_version": self.schema_version,
            "version": TWO_PART_OCCURRENCE_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            **self._identity_payload(),
            "design": self.design.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TwoPartOccurrenceV2Result:
    """Subject-level states and occurrence/conditional-intensity effects."""

    subject_events: pd.DataFrame
    effects: pd.DataFrame
    spec: TwoPartOccurrenceV2Spec

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TwoPartOccurrenceV2Spec):
            raise TypeError("spec must be TwoPartOccurrenceV2Spec")
        required_subject = {
            "event_id",
            self.spec.design.sample_column,
            self.spec.design.subject_column,
            self.spec.design.condition_column,
            "activity_raw",
            "occurrence_state",
            "active_probability",
            "conditional_intensity",
            "measurement_status",
            "out_of_fold",
            "source_event_digest",
        }
        missing = required_subject.difference(self.subject_events.columns)
        if missing:
            raise ValueError(f"subject_events is missing columns: {sorted(missing)}")
        observed_subjects = self.subject_events["measurement_status"].eq("observed")
        states = self.subject_events.loc[observed_subjects, "occurrence_state"]
        if not states.map(lambda value: type(value) in {bool, np.bool_}).all():
            raise ValueError("observed occurrence_state values must be booleans")
        probabilities = pd.to_numeric(
            self.subject_events.loc[observed_subjects, "active_probability"],
            errors="coerce",
        )
        if probabilities.isna().any() or not probabilities.between(0.0, 1.0).all():
            raise ValueError("observed active_probability values must lie in [0, 1]")
        inactive = observed_subjects & ~self.subject_events["occurrence_state"].astype(
            bool
        )
        active = observed_subjects & self.subject_events["occurrence_state"].astype(
            bool
        )
        if (
            self.subject_events.loc[inactive, "conditional_intensity"].notna().any()
            or self.subject_events.loc[active, "conditional_intensity"].isna().any()
        ):
            raise ValueError(
                "conditional intensity must be present exactly when active"
            )
        if tuple(self.effects.columns) != TWO_PART_OCCURRENCE_EFFECT_COLUMNS:
            raise ValueError("two-part occurrence effect columns are invalid")
        if self.effects["effect_id"].duplicated().any():
            raise ValueError("two-part occurrence effect IDs must be unique")
        observed = self.effects["status"].eq("observed")
        if self.effects.loc[observed, ["p_value", "q_value"]].isna().any(axis=None):
            raise ValueError("observed occurrence effects require formal p/q values")
        if not self.effects.loc[observed, "formal_inference_allowed"].all():
            raise ValueError("observed occurrence tests must declare their scope")
        object.__setattr__(self, "subject_events", self.subject_events.copy(deep=True))
        object.__setattr__(self, "effects", self.effects.copy(deep=True))


@dataclass(frozen=True, slots=True)
class _LogisticFit:
    beta: np.ndarray
    covariance: np.ndarray
    method: str
    rank: int


def build_sample_edge_two_part_input(
    scores: SampleEdgeScoreV2,
    *,
    spec: TwoPartOccurrenceV2Spec,
    sample_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project the configured raw score head to the common M4 input grain."""

    if not isinstance(spec, TwoPartOccurrenceV2Spec):
        raise TypeError("spec must be TwoPartOccurrenceV2Spec")
    return cast(
        pd.DataFrame,
        build_sample_edge_differential_input(
            scores,
            score_head=spec.activity_head,
            sample_metadata=sample_metadata,
        ),
    )


def _input_columns(spec: TwoPartOccurrenceV2Spec) -> tuple[str, ...]:
    design = spec.design
    result = [
        "event_id",
        design.sample_column,
        design.subject_column,
        design.condition_column,
        "score",
        "score_status",
        "out_of_fold",
        *design.batch_columns,
        *design.continuous_covariates,
        *design.categorical_covariates,
    ]
    if design.cohort_column is not None:
        result.append(design.cohort_column)
    if design.precision_weight_column is not None:
        result.append(design.precision_weight_column)
    return tuple(result)


def _validate_input(
    activity_table: pd.DataFrame,
    spec: TwoPartOccurrenceV2Spec,
) -> pd.DataFrame:
    if not isinstance(activity_table, pd.DataFrame):
        raise TypeError("activity_table must be a pandas DataFrame")
    columns = _input_columns(spec)
    missing = set(columns).difference(activity_table.columns)
    if missing:
        raise ValueError(f"activity_table is missing columns: {sorted(missing)}")
    table = activity_table.loc[:, list(columns)].copy(deep=True)
    design = spec.design
    string_columns = {
        "event_id",
        design.sample_column,
        design.subject_column,
        "score_status",
        *design.batch_columns,
        *design.categorical_covariates,
    }
    if design.design_kind is not DifferentialDesignKind.CONTINUOUS:
        string_columns.add(design.condition_column)
    if design.cohort_column is not None:
        string_columns.add(design.cohort_column)
    for column in string_columns:
        if table[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values")
        table[column] = table[column].astype(str)
        if (
            table[column].eq("").any()
            or table[column].str.strip().ne(table[column]).any()
        ):
            raise ValueError(f"{column} values must be canonical")
    if table.duplicated(["event_id", design.sample_column]).any():
        raise ValueError("activity_table requires one row per event and sample")
    if not set(table["score_status"]).issubset({*_USABLE_STATUSES, "not_estimable"}):
        raise ValueError("activity_table has unsupported score_status values")
    if (
        not table["out_of_fold"]
        .map(lambda value: type(value) in {bool, np.bool_})
        .all()
        or not table["out_of_fold"].astype(bool).all()
    ):
        raise ValueError("M4 requires boolean out-of-fold rows")
    numeric_columns = ["score", *design.continuous_covariates]
    if design.design_kind is DifferentialDesignKind.CONTINUOUS:
        numeric_columns.append(design.condition_column)
    if design.precision_weight_column is not None:
        numeric_columns.append(design.precision_weight_column)
    for column in numeric_columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        invalid = table[column].notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric.dropna()).any():
            raise ValueError(f"{column} must contain finite numbers or NA")
        table[column] = numeric.astype(float)
    roles = [
        design.subject_column,
        design.condition_column,
        *design.batch_columns,
        *design.continuous_covariates,
        *design.categorical_covariates,
    ]
    if design.cohort_column is not None:
        roles.append(design.cohort_column)
    inconsistent = table.groupby(design.sample_column, observed=True)[roles].nunique(
        dropna=False
    )
    if (inconsistent > 1).any(axis=None):
        raise ValueError("sample metadata varies across event rows")
    return table


def _event_digest(event: pd.DataFrame, spec: TwoPartOccurrenceV2Spec) -> str:
    columns = _input_columns(spec)
    ordered = event.loc[:, list(columns)].sort_values(
        [spec.design.subject_column, spec.design.sample_column], kind="stable"
    )
    records: list[list[object]] = []
    for row in ordered.itertuples(index=False, name=None):
        record: list[object] = []
        for value in row:
            if pd.isna(value):
                record.append(None)
            elif isinstance(value, (float, np.floating)):
                record.append({"float_hex": float(value).hex()})
            elif isinstance(value, np.generic):
                record.append(value.item())
            else:
                record.append(value)
        records.append(record)
    return cast(str, canonical_digest(records))


def _subject_context_rows(
    event: pd.DataFrame,
    spec: TwoPartOccurrenceV2Spec,
    *,
    digest: str,
) -> tuple[pd.DataFrame, float, str | None]:
    design = spec.design
    score = pd.to_numeric(event["score"], errors="coerce")
    usable = (
        event["score_status"].isin(_USABLE_STATUSES)
        & score.notna()
        & np.isfinite(score)
    )
    for column in design.continuous_covariates:
        values = pd.to_numeric(event[column], errors="coerce")
        usable &= values.notna() & np.isfinite(values)
    if design.design_kind is DifferentialDesignKind.CONTINUOUS:
        exposure = pd.to_numeric(event[design.condition_column], errors="coerce")
        usable &= exposure.notna() & np.isfinite(exposure)
    observed_fraction = float(usable.mean())
    source = event.loc[usable].copy()
    source["activity_raw"] = score.loc[usable].astype(float)
    if source.empty:
        return source, observed_fraction, "no_complete_activity_measurements"
    group_columns = [design.subject_column, design.condition_column]
    categorical = [
        *design.batch_columns,
        *design.categorical_covariates,
        *(() if design.cohort_column is None else (design.cohort_column,)),
    ]
    for column in categorical:
        variation = source.groupby(group_columns, observed=True)[column].nunique()
        if (variation > 1).any():
            return (
                pd.DataFrame(),
                observed_fraction,
                f"{column}_varies_within_subject_context",
            )
    aggregation: dict[str, str] = {"activity_raw": "mean"}
    aggregation.update({column: "mean" for column in design.continuous_covariates})
    aggregation.update({column: "first" for column in categorical})
    if design.precision_weight_column is not None:
        aggregation[design.precision_weight_column] = "mean"
    grouped = (
        source.groupby(group_columns, observed=True, sort=True)
        .agg(aggregation)
        .reset_index()
    )
    grouped[design.sample_column] = [
        stable_id(
            "m4_subject_context",
            {
                "condition": (
                    float(condition)
                    if design.design_kind is DifferentialDesignKind.CONTINUOUS
                    else str(condition)
                ),
                "event_id": str(event["event_id"].iloc[0]),
                "subject_id": str(subject),
            },
            schema_version="1",
        )
        for subject, condition in grouped.loc[
            :, [design.subject_column, design.condition_column]
        ].itertuples(index=False, name=None)
    ]
    grouped["event_id"] = str(event["event_id"].iloc[0])
    grouped["occurrence_state"] = grouped["activity_raw"].gt(
        spec.activity_threshold_raw
    )
    grouped["active_probability"] = expit(
        (grouped["activity_raw"] - spec.activity_threshold_raw)
        / spec.probability_transition_scale
    )
    grouped["conditional_intensity"] = grouped["activity_raw"].where(
        grouped["occurrence_state"]
    )
    grouped["measurement_status"] = "observed"
    grouped["out_of_fold"] = True
    grouped["source_event_digest"] = digest
    return cast(pd.DataFrame, grouped), observed_fraction, None


def _pair_levels(
    contrast: DifferentialContrastSpec,
) -> tuple[str, str] | None:
    negative = [(level, weight) for level, weight in contrast.weights if weight < 0]
    positive = [(level, weight) for level, weight in contrast.weights if weight > 0]
    if (
        len(negative) != 1
        or len(positive) != 1
        or not math.isclose(negative[0][1], -1.0, abs_tol=1e-12)
        or not math.isclose(positive[0][1], 1.0, abs_tol=1e-12)
    ):
        return None
    return negative[0][0], positive[0][0]


def _wald_summary(
    log_odds_ratio: float,
    standard_error: float,
) -> tuple[float, float, float, float, float]:
    statistic = log_odds_ratio / standard_error
    p_value = float(2.0 * norm.sf(abs(statistic)))
    lower = log_odds_ratio - float(norm.ppf(0.975)) * standard_error
    upper = log_odds_ratio + float(norm.ppf(0.975)) * standard_error
    clipped = [float(np.clip(value, -700.0, 700.0)) for value in (lower, upper)]
    return (
        statistic,
        p_value,
        math.exp(clipped[0]),
        math.exp(clipped[1]),
        math.exp(float(np.clip(log_odds_ratio, -700.0, 700.0))),
    )


def _independent_exact(
    table: pd.DataFrame,
    *,
    design: DifferentialDesignSpec,
    reference: str,
    target: str,
) -> dict[str, object] | None:
    subject_conditions = table.groupby(design.subject_column, observed=True)[
        design.condition_column
    ].nunique()
    if (subject_conditions > 1).any():
        return None
    reference_values = table.loc[
        table[design.condition_column].astype(str).eq(reference), "occurrence_state"
    ].astype(bool)
    target_values = table.loc[
        table[design.condition_column].astype(str).eq(target), "occurrence_state"
    ].astype(bool)
    if (
        min(len(reference_values), len(target_values))
        < design.minimum_subjects_per_level
    ):
        return None
    reference_occurrences = int(reference_values.sum())
    target_occurrences = int(target_values.sum())
    contingency = np.asarray(
        [
            [target_occurrences, len(target_values) - target_occurrences],
            [reference_occurrences, len(reference_values) - reference_occurrences],
        ],
        dtype=int,
    )
    p_value = float(fisher_exact(contingency, alternative="two-sided").pvalue)
    cells = contingency.astype(float) + 0.5
    log_or = float(math.log(cells[0, 0] * cells[1, 1] / (cells[0, 1] * cells[1, 0])))
    standard_error = float(math.sqrt(np.reciprocal(cells).sum()))
    statistic, _, lower, upper, odds_ratio = _wald_summary(log_or, standard_error)
    return {
        "reference_subjects": len(reference_values),
        "target_subjects": len(target_values),
        "complete_pairs": 0,
        "reference_occurrences": reference_occurrences,
        "target_occurrences": target_occurrences,
        "reference_prevalence": float(reference_values.mean()),
        "target_prevalence": float(target_values.mean()),
        "prevalence_difference": float(target_values.mean() - reference_values.mean()),
        "odds_ratio": odds_ratio,
        "log_odds_ratio": log_or,
        "standard_error": standard_error,
        "statistic": statistic,
        "p_value": p_value,
        "ci_lower": lower,
        "ci_upper": upper,
        "occurrence_method": "fisher_exact_two_sided",
    }


def _paired_exact(
    table: pd.DataFrame,
    *,
    design: DifferentialDesignSpec,
    reference: str,
    target: str,
    minimum_discordant_pairs: int,
) -> dict[str, object] | None:
    matrix = table.pivot(
        index=design.subject_column,
        columns=design.condition_column,
        values="occurrence_state",
    ).reindex(columns=[reference, target])
    complete = matrix.dropna().astype(bool)
    if len(complete) < design.minimum_subjects_per_level:
        return None
    gained = int((~complete[reference] & complete[target]).sum())
    lost = int((complete[reference] & ~complete[target]).sum())
    discordant = gained + lost
    if discordant < minimum_discordant_pairs:
        return None
    p_value = (
        1.0 if discordant == 0 else float(binomtest(gained, discordant, p=0.5).pvalue)
    )
    log_or = math.log((gained + 0.5) / (lost + 0.5))
    standard_error = math.sqrt(1.0 / (gained + 0.5) + 1.0 / (lost + 0.5))
    statistic, _, lower, upper, odds_ratio = _wald_summary(log_or, standard_error)
    return {
        "reference_subjects": len(complete),
        "target_subjects": len(complete),
        "complete_pairs": len(complete),
        "reference_occurrences": int(complete[reference].sum()),
        "target_occurrences": int(complete[target].sum()),
        "reference_prevalence": float(complete[reference].mean()),
        "target_prevalence": float(complete[target].mean()),
        "prevalence_difference": float(
            complete[target].mean() - complete[reference].mean()
        ),
        "odds_ratio": odds_ratio,
        "log_odds_ratio": log_or,
        "standard_error": standard_error,
        "statistic": statistic,
        "p_value": p_value,
        "ci_lower": lower,
        "ci_upper": upper,
        "occurrence_method": "mcnemar_exact_binomial",
    }


def _logistic_design(
    table: pd.DataFrame,
    design: DifferentialDesignSpec,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if design.design_kind is DifferentialDesignKind.CONTINUOUS:
        exposure = table[design.condition_column].to_numpy(dtype=float)
        parts = [np.ones((len(table), 1)), exposure[:, np.newaxis]]
        columns = ["intercept", f"exposure:{design.condition_column}"]
    else:
        conditions = table[design.condition_column].astype(str)
        parts = [
            conditions.eq(level).to_numpy(dtype=float)[:, np.newaxis]
            for level in design.condition_levels
        ]
        columns = [f"condition:{level}" for level in design.condition_levels]
    for column in design.continuous_covariates:
        values = table[column].to_numpy(dtype=float)
        centered = values - float(values.mean())
        scale = float(np.std(centered, ddof=0))
        if scale > 0.0:
            parts.append((centered / scale)[:, np.newaxis])
            columns.append(f"continuous:{column}")
    categorical = [
        *design.batch_columns,
        *design.categorical_covariates,
        *(() if design.cohort_column is None else (design.cohort_column,)),
    ]
    for column in categorical:
        values = table[column].astype(str)
        for level in tuple(sorted(values.unique()))[1:]:
            parts.append(values.eq(level).to_numpy(dtype=float)[:, np.newaxis])
            columns.append(f"categorical:{column}={level}")
    return np.hstack(parts), tuple(columns)


def _fit_logistic(
    design: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    *,
    use_cluster: bool,
    maximum_condition_number: float,
    minimum_clusters: int,
) -> _LogisticFit | None:
    if len(outcome) <= design.shape[1] or len(np.unique(outcome)) < 2:
        return None
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        return None
    beta = np.zeros(design.shape[1], dtype=float)
    converged = False
    for _ in range(100):
        probability = expit(np.clip(design @ beta, -30.0, 30.0))
        variance = np.maximum(probability * (1.0 - probability), 1.0e-8)
        information = design.T @ (variance[:, np.newaxis] * design)
        singular = np.linalg.svd(information, compute_uv=False)
        if (
            singular[-1] <= 0.0
            or singular[0] / singular[-1] > maximum_condition_number**2
        ):
            return None
        step = np.linalg.solve(information, design.T @ (outcome - probability))
        beta = beta + step
        if float(np.max(np.abs(step))) < 1.0e-9:
            converged = True
            break
        if float(np.max(np.abs(beta))) > 30.0:
            return None
    if not converged:
        return None
    probability = expit(np.clip(design @ beta, -30.0, 30.0))
    variance = np.maximum(probability * (1.0 - probability), 1.0e-8)
    information = design.T @ (variance[:, np.newaxis] * design)
    bread = np.linalg.inv(information)
    score = design * (outcome - probability)[:, np.newaxis]
    if use_cluster:
        unique_clusters = tuple(sorted(set(clusters.tolist())))
        if len(unique_clusters) < minimum_clusters:
            return None
        cluster_scores = np.vstack(
            [score[clusters == cluster].sum(axis=0) for cluster in unique_clusters]
        )
        correction = (len(unique_clusters) / (len(unique_clusters) - 1.0)) * (
            (len(outcome) - 1.0) / (len(outcome) - rank)
        )
        meat = correction * (cluster_scores.T @ cluster_scores)
        method = "logistic_subject_cluster_sandwich"
    else:
        correction = len(outcome) / (len(outcome) - rank)
        meat = correction * (score.T @ score)
        method = "logistic_hc1_sandwich"
    covariance = bread @ meat @ bread
    covariance = 0.5 * (covariance + covariance.T)
    if np.any(~np.isfinite(covariance)):
        return None
    return _LogisticFit(
        beta=beta,
        covariance=covariance,
        method=method,
        rank=rank,
    )


def _logistic_contrast(
    table: pd.DataFrame,
    *,
    design_spec: DifferentialDesignSpec,
    contrast: DifferentialContrastSpec | None,
) -> dict[str, object] | None:
    design, columns = _logistic_design(table, design_spec)
    use_cluster = design_spec.design_kind in {
        DifferentialDesignKind.PAIRED,
        DifferentialDesignKind.REPEATED,
    }
    fit = _fit_logistic(
        design,
        table["occurrence_state"].to_numpy(dtype=float),
        table[design_spec.subject_column].astype(str).to_numpy(),
        use_cluster=use_cluster,
        maximum_condition_number=design_spec.maximum_condition_number,
        minimum_clusters=design_spec.minimum_clusters_for_cr2,
    )
    if fit is None:
        return None
    if design_spec.design_kind is DifferentialDesignKind.CONTINUOUS:
        vector = np.asarray(
            [1.0 if column.startswith("exposure:") else 0.0 for column in columns]
        )
        reference = target = None
        reference_rows = target_rows = pd.DataFrame()
    else:
        if contrast is None:
            return None
        levels = _pair_levels(contrast)
        if levels is None:
            return None
        reference, target = levels
        by_level = dict(contrast.weights)
        vector = np.asarray(
            [
                by_level.get(column.removeprefix("condition:"), 0.0)
                if column.startswith("condition:")
                else 0.0
                for column in columns
            ],
            dtype=float,
        )
        reference_rows = table.loc[
            table[design_spec.condition_column].astype(str).eq(reference)
        ]
        target_rows = table.loc[
            table[design_spec.condition_column].astype(str).eq(target)
        ]
        if (
            min(
                reference_rows[design_spec.subject_column].nunique(),
                target_rows[design_spec.subject_column].nunique(),
            )
            < design_spec.minimum_subjects_per_level
        ):
            return None
    log_or = float(vector @ fit.beta)
    variance = float(vector @ fit.covariance @ vector)
    if variance <= 0.0:
        return None
    standard_error = math.sqrt(variance)
    statistic, p_value, lower, upper, odds_ratio = _wald_summary(log_or, standard_error)
    return {
        "reference_subjects": (
            0
            if reference is None
            else int(reference_rows[design_spec.subject_column].nunique())
        ),
        "target_subjects": (
            0
            if target is None
            else int(target_rows[design_spec.subject_column].nunique())
        ),
        "complete_pairs": (
            int(
                table.groupby(design_spec.subject_column, observed=True)[
                    design_spec.condition_column
                ]
                .nunique()
                .ge(2)
                .sum()
            )
            if use_cluster
            else 0
        ),
        "reference_occurrences": (
            0 if reference is None else int(reference_rows["occurrence_state"].sum())
        ),
        "target_occurrences": (
            0 if target is None else int(target_rows["occurrence_state"].sum())
        ),
        "reference_prevalence": (
            None
            if reference is None
            else float(reference_rows["occurrence_state"].mean())
        ),
        "target_prevalence": (
            None if target is None else float(target_rows["occurrence_state"].mean())
        ),
        "prevalence_difference": (
            None
            if reference is None
            else float(
                target_rows["occurrence_state"].mean()
                - reference_rows["occurrence_state"].mean()
            )
        ),
        "odds_ratio": odds_ratio,
        "log_odds_ratio": log_or,
        "standard_error": standard_error,
        "statistic": statistic,
        "p_value": p_value,
        "ci_lower": lower,
        "ci_upper": upper,
        "occurrence_method": fit.method,
    }


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    observed = values.dropna().astype(float).sort_values(kind="stable")
    if observed.empty:
        return result
    count = len(observed)
    raw = observed.to_numpy() * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    result.loc[observed.index] = np.minimum(adjusted, 1.0)
    return result


def _identity_value(value: object) -> object:
    if (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))
    ):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _conditional_intensity_table(
    subject_events: pd.DataFrame,
    spec: TwoPartOccurrenceV2Spec,
) -> pd.DataFrame:
    design = spec.design
    table = subject_events.copy(deep=True)
    table["score"] = table["conditional_intensity"]
    table["score_status"] = np.where(
        table["conditional_intensity"].notna(), "observed", "not_estimable"
    )
    if design.precision_weight_column is not None:
        if design.precision_weight_column not in table:
            table[design.precision_weight_column] = 1.0
    return table


def fit_two_part_occurrence_v2(
    activity_table: pd.DataFrame,
    spec: TwoPartOccurrenceV2Spec,
) -> TwoPartOccurrenceV2Result:
    """Fit formal occurrence tests and a separate active-only intensity channel."""

    if not isinstance(spec, TwoPartOccurrenceV2Spec):
        raise TypeError("spec must be TwoPartOccurrenceV2Spec")
    table = _validate_input(activity_table, spec)
    subject_parts: list[pd.DataFrame] = []
    observed_fraction_by_event: dict[str, float] = {}
    reason_by_event: dict[str, str | None] = {}
    digest_by_event: dict[str, str] = {}
    for event_id, event in table.groupby("event_id", observed=True, sort=True):
        digest = _event_digest(event, spec)
        subject_rows, observed_fraction, reason = _subject_context_rows(
            event, spec, digest=digest
        )
        event_name = str(event_id)
        digest_by_event[event_name] = digest
        observed_fraction_by_event[event_name] = observed_fraction
        reason_by_event[event_name] = reason
        if not subject_rows.empty:
            subject_parts.append(subject_rows)
    subject_events = (
        pd.concat(subject_parts, ignore_index=True)
        if subject_parts
        else pd.DataFrame(
            columns=[
                "event_id",
                spec.design.sample_column,
                spec.design.subject_column,
                spec.design.condition_column,
                "activity_raw",
                "occurrence_state",
                "active_probability",
                "conditional_intensity",
                "measurement_status",
                "out_of_fold",
                "source_event_digest",
            ]
        )
    )
    conditional_by_key: dict[tuple[str, str], pd.Series] = {}
    if not subject_events.empty:
        conditional = fit_design_aware_differential(
            _conditional_intensity_table(subject_events, spec),
            spec.design,
        ).effects
        conditional_by_key = {
            (str(row["event_id"]), str(row["contrast_name"])): row
            for _, row in conditional.iterrows()
        }

    rows: list[dict[str, object]] = []
    contrasts: tuple[DifferentialContrastSpec | None, ...] = (
        (None,)
        if spec.design.design_kind is DifferentialDesignKind.CONTINUOUS
        else tuple(spec.design.contrasts)
    )
    for event_id in sorted(digest_by_event):
        event_subjects = subject_events.loc[subject_events["event_id"].eq(event_id)]
        for contrast in contrasts:
            contrast_name = (
                f"slope:{spec.design.condition_column}"
                if contrast is None
                else contrast.name
            )
            levels = None if contrast is None else _pair_levels(contrast)
            summary: dict[str, object] | None = None
            reason = reason_by_event[event_id]
            if reason is None and not event_subjects.empty:
                no_adjustment = not (
                    spec.design.batch_columns
                    or spec.design.continuous_covariates
                    or spec.design.categorical_covariates
                    or spec.design.cohort_column is not None
                )
                if (
                    levels is not None
                    and no_adjustment
                    and spec.design.design_kind
                    in {
                        DifferentialDesignKind.INDEPENDENT_TWO_GROUP,
                        DifferentialDesignKind.INDEPENDENT_MULTI_GROUP,
                    }
                ):
                    summary = _independent_exact(
                        event_subjects,
                        design=spec.design,
                        reference=levels[0],
                        target=levels[1],
                    )
                elif (
                    levels is not None
                    and no_adjustment
                    and spec.design.design_kind
                    in {
                        DifferentialDesignKind.PAIRED,
                        DifferentialDesignKind.REPEATED,
                    }
                ):
                    summary = _paired_exact(
                        event_subjects,
                        design=spec.design,
                        reference=levels[0],
                        target=levels[1],
                        minimum_discordant_pairs=spec.minimum_discordant_pairs,
                    )
                else:
                    summary = _logistic_contrast(
                        event_subjects,
                        design_spec=spec.design,
                        contrast=contrast,
                    )
            if summary is None:
                reason = reason or (
                    "contrast_requires_two_unit_weights"
                    if contrast is not None and levels is None
                    else "occurrence_design_not_estimable"
                )
                summary = {
                    "reference_subjects": 0,
                    "target_subjects": 0,
                    "complete_pairs": 0,
                    "reference_occurrences": 0,
                    "target_occurrences": 0,
                    "reference_prevalence": None,
                    "target_prevalence": None,
                    "prevalence_difference": None,
                    "odds_ratio": None,
                    "log_odds_ratio": None,
                    "standard_error": None,
                    "statistic": None,
                    "p_value": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "occurrence_method": "not_estimable",
                }
            conditional = conditional_by_key.get((event_id, contrast_name))
            status = "observed" if summary["p_value"] is not None else "not_estimable"
            identity = {
                "event_id": event_id,
                "contrast_name": contrast_name,
                "reference": None if levels is None else levels[0],
                "target": None if levels is None else levels[1],
                **summary,
                "q_value": None,
                "conditional_intensity_effect": (
                    None if conditional is None else conditional["effect"]
                ),
                "conditional_intensity_standard_error": (
                    None if conditional is None else conditional["standard_error"]
                ),
                "conditional_intensity_diagnostic_p_value": (
                    None if conditional is None else conditional["diagnostic_p_value"]
                ),
                "conditional_intensity_status": (
                    "not_estimable" if conditional is None else conditional["status"]
                ),
                "n_subject_context_rows": len(event_subjects),
                "observed_fraction": observed_fraction_by_event[event_id],
                "activity_head": spec.activity_head,
                "activity_threshold_raw": spec.activity_threshold_raw,
                "status": status,
                "reason_code": None if status == "observed" else reason,
                "formal_inference_allowed": status == "observed",
                "formal_inference_scope": (
                    "fixed_threshold_oof_occurrence_only"
                    if status == "observed"
                    else "not_estimable"
                ),
                "design_spec_id": spec.design.spec_id,
                "occurrence_spec_id": spec.spec_id,
                "source_event_digest": digest_by_event[event_id],
            }
            rows.append(identity)
    effects = pd.DataFrame(rows)
    effects["q_value"] = _benjamini_hochberg(effects["p_value"])
    effects["effect_id"] = [
        stable_id(
            "two_part_occurrence_v2_effect",
            {
                column: _identity_value(row[column])
                for column in map(str, effects.columns)
                if column != "effect_id"
            },
            schema_version=_SCHEMA_VERSION,
        )
        for _, row in effects.iterrows()
    ]
    effects = effects.loc[:, list(TWO_PART_OCCURRENCE_EFFECT_COLUMNS)].sort_values(
        ["event_id", "contrast_name"], kind="stable", ignore_index=True
    )
    return TwoPartOccurrenceV2Result(
        subject_events=subject_events.sort_values(
            ["event_id", spec.design.subject_column, spec.design.condition_column],
            kind="stable",
            ignore_index=True,
        ),
        effects=effects,
        spec=spec,
    )


__all__ = [
    "TWO_PART_OCCURRENCE_EFFECT_COLUMNS",
    "TWO_PART_OCCURRENCE_VERSION",
    "TwoPartOccurrenceV2Result",
    "TwoPartOccurrenceV2Spec",
    "build_sample_edge_two_part_input",
    "fit_two_part_occurrence_v2",
]
