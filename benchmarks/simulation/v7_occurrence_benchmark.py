"""Estimand-specific M4 occurrence benchmark on the integrated v7 DGP."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import fisher_exact, norm
from sklearn.metrics import average_precision_score, roc_auc_score

from crychic.core import stable_id
from crychic.inference import (
    TWO_PART_OCCURRENCE_EFFECT_COLUMNS,
    DifferentialContrastSpec,
    DifferentialDesignKind,
    DifferentialDesignSpec,
    TwoPartOccurrenceV2Spec,
    build_sample_edge_two_part_input,
)
from crychic.workflow import (
    CrossFitArtifacts,
    fit_crossfit_two_part_occurrence_v2,
)

SCHEMA_VERSION = "crychic-suggest-next2-v7-m4-occurrence-benchmark-v2"
M4_METHOD = "crychic_m4_design_aware"
RAW_METHOD = "raw_prevalence_difference"
FISHER_METHOD = "fisher_exact"
DCST_METHOD = "dcst_protocol_compatible_exact"
LOGISTIC_METHOD = "logistic_model"
BETA_BINOMIAL_METHOD = "beta_binomial"
M4_PROBABILITY_METHOD = "crychic_m4_working_probability"
RAW_PROBABILITY_METHOD = "raw_leave_one_out_prevalence"
M4_METHODS = (
    RAW_METHOD,
    FISHER_METHOD,
    DCST_METHOD,
    LOGISTIC_METHOD,
    BETA_BINOMIAL_METHOD,
    M4_METHOD,
)

M4_SUBJECT_EVENT_COLUMNS = (
    "schema_version",
    "dataset_id",
    "dgp_family",
    "design_kind",
    "event_id",
    "receiver",
    "interaction_id",
    "sample_id",
    "subject_id",
    "condition",
    "activity_raw",
    "occurrence_state",
    "active_probability",
    "conditional_intensity",
    "measurement_status",
    "out_of_fold",
    "truth_state_known",
    "truth_occurrence_state",
    "raw_leave_one_out_prevalence",
    "source_event_digest",
)
M4_EFFECT_COLUMNS = (
    "schema_version",
    "dataset_id",
    "dgp_family",
    "design_kind",
    "receiver",
    "interaction_id",
    *TWO_PART_OCCURRENCE_EFFECT_COLUMNS,
)
M4_ALIGNED_EFFECT_COLUMNS = (
    *M4_EFFECT_COLUMNS,
    "truth_known",
    "truth_label",
    "truth_occurrence_effect",
    "truth_direction",
)
M4_BASELINE_COLUMNS = (
    "schema_version",
    "dataset_id",
    "dgp_family",
    "design_kind",
    "method",
    "method_scope",
    "event_id",
    "contrast_name",
    "receiver",
    "interaction_id",
    "prevalence_effect",
    "log_odds_ratio",
    "ranking_score",
    "p_value",
    "q_value",
    "status",
    "reason_code",
    "formal_inference_allowed",
)
M4_METRIC_COLUMNS = (
    "schema_version",
    "dataset_id",
    "dgp_family",
    "design_kind",
    "method",
    "contrast_name",
    "metric",
    "value",
    "status",
    "reason_code",
    "n_observed",
    "n_positive",
    "n_negative",
)
M4_EFFECT_METRICS = (
    "estimable_fraction",
    "occurrence_auprc",
    "occurrence_auroc",
    "prevalence_effect_rmse",
    "prevalence_effect_bias",
    "direction_accuracy",
    "odds_ratio_bias",
    "log_odds_rmse",
    "type1_error_alpha_0_05",
    "fdr_at_q_0_10",
    "power_at_q_0_10",
)


def _name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _parent_truth(
    truth: pd.DataFrame,
    *,
    activity_head: str,
    event_universe: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "contrast_name",
        "receiver",
        "interaction_id",
        "truth_occurrence_effect",
        "truth_causal_parent",
        "truth_direction",
    }
    if not isinstance(truth, pd.DataFrame):
        raise TypeError("truth must be a pandas DataFrame")
    missing = required.difference(truth.columns)
    if missing:
        raise ValueError(f"M4 truth is missing columns: {sorted(missing)}")
    keys = ["contrast_name", "receiver", "interaction_id"]
    values = [
        "truth_occurrence_effect",
        "truth_causal_parent",
        "truth_direction",
    ]
    consistency = truth.groupby(keys, observed=True)[values].nunique(dropna=False)
    if consistency.gt(1).any(axis=None):
        raise ValueError("M4 parent truth must be sender-invariant")
    observed_truth = truth.drop_duplicates(keys).loc[:, [*keys, *values]].copy()
    observed_truth["event_id"] = [
        stable_id(
            "sample_edge_differential_event",
            {
                "score_head": activity_head,
                "receiver": str(receiver),
                "interaction_id": str(interaction_id),
            },
            schema_version="1",
        )
        for receiver, interaction_id in observed_truth.loc[
            :, ["receiver", "interaction_id"]
        ].itertuples(index=False, name=None)
    ]
    if observed_truth.duplicated(["contrast_name", "event_id"]).any():
        raise ValueError("M4 truth event keys must be unique")
    if tuple(event_universe.columns) != (
        "event_id",
        "receiver",
        "interaction_id",
    ):
        raise ValueError("M4 event universe does not match its frozen contract")
    if event_universe["event_id"].duplicated().any():
        raise ValueError("M4 event IDs map to multiple parent identities")
    contrasts = tuple(sorted(truth["contrast_name"].astype(str).unique()))
    complete = pd.concat(
        [event_universe.assign(contrast_name=contrast) for contrast in contrasts],
        ignore_index=True,
    )
    result = complete.merge(
        observed_truth.loc[
            :,
            [
                "contrast_name",
                "event_id",
                "truth_occurrence_effect",
                "truth_causal_parent",
                "truth_direction",
            ],
        ],
        on=["contrast_name", "event_id"],
        how="left",
        validate="one_to_one",
    )
    return result.sort_values(
        ["contrast_name", "event_id"], kind="stable", ignore_index=True
    )


def _event_universe(
    crossfit: CrossFitArtifacts,
    spec: TwoPartOccurrenceV2Spec,
) -> pd.DataFrame:
    parts = [
        build_sample_edge_two_part_input(fold.sample_edge_scores_v2, spec=spec).loc[
            :, ["event_id", "receiver", "interaction_id"]
        ]
        for fold in crossfit.folds
        if fold.sample_edge_scores_v2 is not None
    ]
    if len(parts) != len(crossfit.folds):
        raise ValueError("M4 event universe requires v2 scores in every fold")
    result = pd.concat(parts, ignore_index=True).drop_duplicates(ignore_index=True)
    if result.empty or result["event_id"].duplicated().any():
        raise ValueError("M4 event universe has ambiguous biological identities")
    return result.sort_values("event_id", kind="stable", ignore_index=True)


def _decorate_subject_events(
    subject_events: pd.DataFrame,
    raw_comparator_subject_events: pd.DataFrame,
    parent_truth: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    dataset_id: str,
    dgp_family: str,
    design_kind: str,
    condition_column: str,
) -> pd.DataFrame:
    identity = parent_truth.loc[
        :, ["event_id", "receiver", "interaction_id"]
    ].drop_duplicates()
    event_truth = (
        parent_truth.groupby("event_id", observed=True, sort=False)
        .agg(
            truth_event_known=(
                "truth_occurrence_effect",
                lambda values: values.notna().any(),
            ),
            truth_causal_parent=(
                "truth_causal_parent",
                lambda values: any(bool(value) for value in values.dropna()),
            ),
        )
        .reset_index()
    )
    metadata_required = {
        "subject_id",
        "condition",
        condition_column,
        "responder",
    }
    missing = metadata_required.difference(sample_metadata.columns)
    if missing:
        raise ValueError(f"M4 sample metadata is missing fields: {sorted(missing)}")
    metadata = sample_metadata.loc[:, sorted(metadata_required)].copy()
    metadata_keys = ["subject_id", condition_column]
    if metadata.duplicated(metadata_keys).any():
        raise ValueError("M4 metadata requires one row per subject and design context")
    result = (
        subject_events.merge(identity, on="event_id", validate="many_to_one")
        .merge(event_truth, on="event_id", validate="many_to_one")
        .merge(
            metadata,
            on=metadata_keys,
            how="left",
            validate="many_to_one",
        )
    )
    if result["responder"].isna().any():
        raise ValueError("M4 metadata does not cover the subject-event axis")
    causal_parent = result["truth_causal_parent"].astype(bool)
    if dgp_family == "occurrence_heterogeneity":
        truth_known = result["truth_event_known"].astype(bool)
        truth_state = (
            causal_parent
            & result["condition"].astype(str).ne("A")
            & result["responder"].astype(bool)
        )
    else:
        truth_known = result["truth_event_known"].astype(bool) & ~causal_parent
        truth_state = pd.Series(False, index=result.index)
    result["truth_state_known"] = truth_known.astype(bool)
    result["truth_occurrence_state"] = pd.array(
        truth_state.where(truth_known, pd.NA), dtype="boolean"
    )
    raw = raw_comparator_subject_events.copy(deep=True)
    raw_observed = raw["measurement_status"].eq("observed")
    raw_state = raw["occurrence_state"].astype(bool)
    raw_group = ["event_id", condition_column]
    raw_count = raw_observed.groupby([raw[column] for column in raw_group]).transform(
        "sum"
    )
    raw_total = (
        raw_state.where(raw_observed, False)
        .groupby([raw[column] for column in raw_group])
        .transform("sum")
    )
    raw["raw_leave_one_out_prevalence"] = (
        (raw_total - raw_state.astype(int)) / (raw_count - 1)
    ).where(raw_observed & raw_count.gt(1))
    raw_prediction = raw.loc[
        :, ["event_id", "sample_id", "raw_leave_one_out_prevalence"]
    ]
    if raw_prediction.duplicated(["event_id", "sample_id"]).any():
        raise ValueError("raw M4 comparator requires one subject-context row per event")
    result = result.merge(
        raw_prediction,
        on=["event_id", "sample_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    result.insert(0, "design_kind", design_kind)
    result.insert(0, "dgp_family", dgp_family)
    result.insert(0, "dataset_id", dataset_id)
    result.insert(0, "schema_version", SCHEMA_VERSION)
    return result.loc[:, list(M4_SUBJECT_EVENT_COLUMNS)].sort_values(
        ["event_id", "subject_id", "condition"],
        kind="stable",
        ignore_index=True,
    )


def _decorate_effects(
    effects: pd.DataFrame,
    parent_truth: pd.DataFrame,
    *,
    dataset_id: str,
    dgp_family: str,
    design_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = parent_truth.loc[
        :, ["event_id", "receiver", "interaction_id"]
    ].drop_duplicates()
    decorated = effects.merge(identity, on="event_id", validate="many_to_one")
    decorated.insert(0, "design_kind", design_kind)
    decorated.insert(0, "dgp_family", dgp_family)
    decorated.insert(0, "dataset_id", dataset_id)
    decorated.insert(0, "schema_version", SCHEMA_VERSION)
    decorated = decorated.loc[:, list(M4_EFFECT_COLUMNS)]
    aligned = decorated.merge(
        parent_truth.loc[
            :,
            [
                "contrast_name",
                "event_id",
                "truth_occurrence_effect",
                "truth_causal_parent",
                "truth_direction",
            ],
        ].rename(columns={"truth_causal_parent": "truth_label"}),
        on=["contrast_name", "event_id"],
        validate="one_to_one",
    )
    aligned["truth_known"] = aligned["truth_occurrence_effect"].notna()
    aligned["truth_label"] = pd.array(aligned["truth_label"], dtype="boolean")
    aligned = aligned.loc[:, list(M4_ALIGNED_EFFECT_COLUMNS)]
    order = ["contrast_name", "event_id"]
    return (
        decorated.sort_values(order, kind="stable", ignore_index=True),
        aligned.sort_values(order, kind="stable", ignore_index=True),
    )


def _bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = numeric.notna() & np.isfinite(numeric)
    if not finite.any():
        return result
    selected = numeric.loc[finite].clip(lower=0.0, upper=1.0)
    order = np.argsort(selected.to_numpy(dtype=float), kind="stable")
    ranked = selected.to_numpy(dtype=float)[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result.loc[selected.index] = restored
    return result


def _contrast_vector(
    contrast: DifferentialContrastSpec | None,
    columns: tuple[str, ...],
    design: DifferentialDesignSpec,
) -> np.ndarray | None:
    if design.design_kind is DifferentialDesignKind.CONTINUOUS:
        return np.asarray(
            [
                1.0 if name == f"exposure:{design.condition_column}" else 0.0
                for name in columns
            ]
        )
    if contrast is None:
        return None
    weights = dict(contrast.weights)
    return np.asarray(
        [
            weights.get(name.removeprefix("condition:"), 0.0)
            if name.startswith("condition:")
            else 0.0
            for name in columns
        ],
        dtype=float,
    )


def _logistic_design(
    table: pd.DataFrame,
    design: DifferentialDesignSpec,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if design.design_kind is DifferentialDesignKind.CONTINUOUS:
        exposure = pd.to_numeric(table[design.condition_column], errors="coerce")
        parts = [np.ones((len(table), 1)), exposure.to_numpy()[:, np.newaxis]]
        columns = ["intercept", f"exposure:{design.condition_column}"]
    else:
        condition = table[design.condition_column].astype(str)
        parts = [
            condition.eq(level).to_numpy(dtype=float)[:, np.newaxis]
            for level in design.condition_levels
        ]
        columns = [f"condition:{level}" for level in design.condition_levels]
    for column in design.continuous_covariates:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        centered = values - float(np.mean(values))
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


def _logistic_rows(
    subject_events: pd.DataFrame,
    effects: pd.DataFrame,
    design: DifferentialDesignSpec,
) -> dict[tuple[str, str], dict[str, object]]:
    try:
        import statsmodels.api as sm  # type: ignore[import-untyped]
    except ImportError:
        return {}
    contrasts: tuple[DifferentialContrastSpec | None, ...] = (
        (None,)
        if design.design_kind is DifferentialDesignKind.CONTINUOUS
        else tuple(design.contrasts)
    )
    result: dict[tuple[str, str], dict[str, object]] = {}
    for event_id, event in subject_events.groupby("event_id", observed=True, sort=True):
        event = event.loc[event["measurement_status"].eq("observed")].copy()
        if event.empty or event["occurrence_state"].nunique() < 2:
            continue
        matrix, columns = _logistic_design(event, design)
        outcome = event["occurrence_state"].astype(float).to_numpy()
        if (
            len(outcome) <= matrix.shape[1]
            or np.linalg.matrix_rank(matrix) < matrix.shape[1]
        ):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if design.design_kind in {
                    DifferentialDesignKind.PAIRED,
                    DifferentialDesignKind.REPEATED,
                }:
                    model = sm.GEE(
                        outcome,
                        matrix,
                        groups=event[design.subject_column].astype(str).to_numpy(),
                        family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Independence(),
                    ).fit(maxiter=100)
                    scope = "subject_clustered_logistic_gee"
                else:
                    model = sm.GLM(outcome, matrix, family=sm.families.Binomial()).fit(
                        cov_type="HC1", maxiter=100
                    )
                    scope = "logistic_glm_hc1"
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        covariance = np.asarray(model.cov_params(), dtype=float)
        parameters = np.asarray(model.params, dtype=float)
        for contrast in contrasts:
            contrast_name = (
                f"slope:{design.condition_column}"
                if contrast is None
                else contrast.name
            )
            vector = _contrast_vector(contrast, columns, design)
            if vector is None:
                continue
            estimate = float(vector @ parameters)
            variance = float(vector @ covariance @ vector)
            if (
                not math.isfinite(estimate)
                or not math.isfinite(variance)
                or variance <= 0.0
            ):
                continue
            standard_error = math.sqrt(variance)
            statistic = estimate / standard_error
            p_value = float(2.0 * norm.sf(abs(statistic)))
            prevalence = effects.loc[
                effects["event_id"].eq(event_id)
                & effects["contrast_name"].eq(contrast_name),
                "prevalence_difference",
            ]
            if len(prevalence) != 1:
                continue
            prevalence_value = pd.to_numeric(prevalence, errors="coerce").iloc[0]
            result[(str(event_id), contrast_name)] = {
                "prevalence_effect": (
                    None if pd.isna(prevalence_value) else float(prevalence_value)
                ),
                "log_odds_ratio": estimate,
                "ranking_score": -math.log10(max(p_value, 1.0e-300)),
                "p_value": p_value,
                "method_scope": scope,
            }
    return result


def _pair_levels(
    contrast_name: str,
    design: DifferentialDesignSpec,
) -> tuple[str, str] | None:
    selected = [item for item in design.contrasts if item.name == contrast_name]
    if len(selected) != 1:
        return None
    negative = [(level, weight) for level, weight in selected[0].weights if weight < 0]
    positive = [(level, weight) for level, weight in selected[0].weights if weight > 0]
    if (
        len(negative) != 1
        or len(positive) != 1
        or not math.isclose(negative[0][1], -1.0, abs_tol=1.0e-12)
        or not math.isclose(positive[0][1], 1.0, abs_tol=1.0e-12)
    ):
        return None
    return str(negative[0][0]), str(positive[0][0])


def _raw_prevalence_summary(
    event: pd.DataFrame,
    *,
    contrast_name: str,
    design: DifferentialDesignSpec,
) -> dict[str, float] | None:
    levels = _pair_levels(contrast_name, design)
    if levels is None:
        return None
    reference, target = levels
    observed = event.loc[event["measurement_status"].eq("observed")].copy()
    if design.design_kind in {
        DifferentialDesignKind.PAIRED,
        DifferentialDesignKind.REPEATED,
    }:
        matrix = observed.pivot(
            index=design.subject_column,
            columns=design.condition_column,
            values="occurrence_state",
        ).reindex(columns=[reference, target])
        complete = matrix.dropna().astype(bool)
        if len(complete) < design.minimum_subjects_per_level:
            return None
        reference_values = complete[reference]
        target_values = complete[target]
    else:
        reference_values = observed.loc[
            observed[design.condition_column].astype(str).eq(reference),
            "occurrence_state",
        ].astype(bool)
        target_values = observed.loc[
            observed[design.condition_column].astype(str).eq(target),
            "occurrence_state",
        ].astype(bool)
        if (
            min(len(reference_values), len(target_values))
            < design.minimum_subjects_per_level
        ):
            return None
    reference_n = len(reference_values)
    target_n = len(target_values)
    reference_k = int(reference_values.sum())
    target_k = int(target_values.sum())
    contingency = np.asarray(
        [
            [target_k, target_n - target_k],
            [reference_k, reference_n - reference_k],
        ],
        dtype=int,
    )
    corrected = contingency.astype(float) + 0.5
    log_odds = float(
        math.log(
            corrected[0, 0] * corrected[1, 1] / (corrected[0, 1] * corrected[1, 0])
        )
    )
    return {
        "prevalence_effect": float(target_values.mean() - reference_values.mean()),
        "log_odds_ratio": log_odds,
        "fisher_p_value": float(
            fisher_exact(contingency, alternative="two-sided").pvalue
        ),
    }


def _baseline_table(
    effects: pd.DataFrame,
    raw_comparator_effects: pd.DataFrame,
    raw_comparator_subject_events: pd.DataFrame,
    design: DifferentialDesignSpec,
) -> pd.DataFrame:
    logistic = _logistic_rows(
        raw_comparator_subject_events,
        raw_comparator_effects,
        design,
    )
    rows: list[dict[str, object]] = []
    for effect in effects.itertuples(index=False):
        observed = str(effect.status) == "observed"
        event_subjects = raw_comparator_subject_events.loc[
            raw_comparator_subject_events["event_id"]
            .astype(str)
            .eq(str(effect.event_id))
        ]
        raw = _raw_prevalence_summary(
            event_subjects,
            contrast_name=str(effect.contrast_name),
            design=design,
        )
        base = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": str(effect.dataset_id),
            "dgp_family": str(effect.dgp_family),
            "design_kind": str(effect.design_kind),
            "event_id": str(effect.event_id),
            "contrast_name": str(effect.contrast_name),
            "receiver": str(effect.receiver),
            "interaction_id": str(effect.interaction_id),
        }

        def append(
            method: str,
            scope: str,
            *,
            prevalence_effect: float | None = None,
            log_odds_ratio: float | None = None,
            ranking_score: float | None = None,
            p_value: float | None = None,
            status: str = "observed",
            reason_code: str | None = None,
            formal: bool = False,
            _base: dict[str, object] = base,
        ) -> None:
            rows.append(
                {
                    **_base,
                    "method": method,
                    "method_scope": scope,
                    "prevalence_effect": prevalence_effect,
                    "log_odds_ratio": log_odds_ratio,
                    "ranking_score": ranking_score,
                    "p_value": p_value,
                    "q_value": None,
                    "status": status,
                    "reason_code": reason_code,
                    "formal_inference_allowed": formal,
                }
            )

        if raw is not None:
            prevalence = float(raw["prevalence_effect"])
            log_odds = float(raw["log_odds_ratio"])
            append(
                RAW_METHOD,
                "descriptive_subject_prevalence",
                prevalence_effect=prevalence,
                log_odds_ratio=log_odds,
                ranking_score=abs(prevalence),
            )
        else:
            append(
                RAW_METHOD,
                "descriptive_subject_prevalence",
                status="not_estimable",
                reason_code="pairwise_prevalence_contrast_not_estimable",
            )

        if observed:
            prevalence = (
                None
                if pd.isna(effect.prevalence_difference)
                else float(effect.prevalence_difference)
            )
            log_odds = float(effect.log_odds_ratio)
            p_value = float(effect.p_value)
            append(
                M4_METHOD,
                str(effect.occurrence_method),
                prevalence_effect=prevalence,
                log_odds_ratio=log_odds,
                ranking_score=(
                    -math.log10(max(p_value, 1.0e-300))
                    if prevalence is None
                    else abs(prevalence)
                ),
                p_value=p_value,
                formal=bool(effect.formal_inference_allowed),
            )
        elif pd.notna(effect.prevalence_difference) and pd.notna(effect.log_odds_ratio):
            prevalence = float(effect.prevalence_difference)
            append(
                M4_METHOD,
                str(effect.occurrence_method),
                prevalence_effect=prevalence,
                log_odds_ratio=float(effect.log_odds_ratio),
                ranking_score=abs(prevalence),
                status="descriptive",
                reason_code=str(effect.reason_code),
            )
        else:
            reason = str(effect.reason_code)
            append(
                M4_METHOD,
                str(effect.occurrence_method),
                status="not_estimable",
                reason_code=reason,
            )

        fisher_eligible = raw is not None and design.design_kind in {
            DifferentialDesignKind.INDEPENDENT_TWO_GROUP,
            DifferentialDesignKind.INDEPENDENT_MULTI_GROUP,
            DifferentialDesignKind.MULTI_COHORT,
        }
        for method, scope in (
            (FISHER_METHOD, "two_sided_fisher_exact"),
            (DCST_METHOD, "dcst_protocol_compatible_binary_linkage_fisher_proxy"),
        ):
            if fisher_eligible:
                if raw is None:  # pragma: no cover - narrowed above
                    raise RuntimeError("Fisher baseline lost its prevalence summary")
                p_value = float(raw["fisher_p_value"])
                append(
                    method,
                    scope,
                    prevalence_effect=float(raw["prevalence_effect"]),
                    log_odds_ratio=float(raw["log_odds_ratio"]),
                    ranking_score=-math.log10(max(p_value, 1.0e-300)),
                    p_value=p_value,
                    formal=method == FISHER_METHOD,
                )
            else:
                append(
                    method,
                    scope,
                    status="not_estimable",
                    reason_code="independent_fisher_design_required",
                )

        logistic_row = logistic.get((str(effect.event_id), str(effect.contrast_name)))
        if logistic_row is None:
            append(
                LOGISTIC_METHOD,
                "design_matched_logistic",
                status="not_estimable",
                reason_code="logistic_fit_not_estimable",
            )
        else:
            append(
                LOGISTIC_METHOD,
                str(logistic_row["method_scope"]),
                prevalence_effect=(
                    None
                    if logistic_row["prevalence_effect"] is None
                    else float(logistic_row["prevalence_effect"])
                ),
                log_odds_ratio=float(logistic_row["log_odds_ratio"]),
                ranking_score=float(logistic_row["ranking_score"]),
                p_value=float(logistic_row["p_value"]),
            )
        append(
            BETA_BINOMIAL_METHOD,
            "subject_level_beta_binomial",
            status="not_estimable",
            reason_code="one_bernoulli_trial_per_subject_has_no_identifiable_overdispersion",
        )
    result = pd.DataFrame.from_records(rows, columns=M4_BASELINE_COLUMNS)
    for method, index in result.groupby("method", observed=True).groups.items():
        del method
        selected = pd.Index(index)
        result.loc[selected, "q_value"] = _bh(result.loc[selected, "p_value"])
    return result.sort_values(
        ["method", "contrast_name", "event_id"], kind="stable", ignore_index=True
    )


def _metric_row(
    first: pd.Series,
    *,
    metric: str,
    value: float | None,
    reason_code: str | None,
    n_observed: int,
    n_positive: int,
    n_negative: int,
) -> dict[str, object]:
    is_observed = value is not None and math.isfinite(value)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": str(first["dataset_id"]),
        "dgp_family": str(first["dgp_family"]),
        "design_kind": str(first["design_kind"]),
        "method": str(first["method"]),
        "contrast_name": str(first["contrast_name"]),
        "metric": metric,
        "value": value if is_observed else None,
        "status": "observed" if is_observed else "not_estimable",
        "reason_code": None if is_observed else reason_code or "metric_not_estimable",
        "n_observed": n_observed,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def _effect_metrics(
    baselines: pd.DataFrame,
    aligned: pd.DataFrame,
) -> list[dict[str, object]]:
    truth = aligned.loc[
        :,
        [
            "event_id",
            "contrast_name",
            "truth_known",
            "truth_occurrence_effect",
        ],
    ]
    merged = baselines.merge(
        truth,
        on=["event_id", "contrast_name"],
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for _, table in merged.groupby(
        ["method", "contrast_name"], observed=True, sort=True
    ):
        first = table.iloc[0]
        known = table["truth_known"].astype(bool)
        labels = pd.to_numeric(table["truth_occurrence_effect"], errors="coerce").ne(
            0.0
        )
        effect_available = table["status"].isin({"observed", "descriptive"})
        usable = known & effect_available
        n_observed = int(usable.sum())
        n_positive = int((known & labels).sum())
        n_negative = int((known & ~labels).sum())
        rows.append(
            _metric_row(
                first,
                metric="estimable_fraction",
                value=float(effect_available.mean()),
                reason_code=None,
                n_observed=n_observed,
                n_positive=n_positive,
                n_negative=n_negative,
            )
        )
        ranking = pd.to_numeric(table["ranking_score"], errors="coerce")
        rank_valid = usable & ranking.notna() & np.isfinite(ranking)
        rank_labels = labels.loc[rank_valid].to_numpy(dtype=bool)
        rank_values = ranking.loc[rank_valid].to_numpy(dtype=float)
        for metric, function in (
            ("occurrence_auprc", average_precision_score),
            ("occurrence_auroc", roc_auc_score),
        ):
            if len(rank_values) >= 2 and len(np.unique(rank_labels)) == 2:
                value = float(function(rank_labels, rank_values))
                reason = None
            else:
                value = None
                reason = "truth_is_single_class_or_ranking_unavailable"
            rows.append(
                _metric_row(
                    first,
                    metric=metric,
                    value=value,
                    reason_code=reason,
                    n_observed=int(rank_valid.sum()),
                    n_positive=n_positive,
                    n_negative=n_negative,
                )
            )
        predicted = pd.to_numeric(table["prevalence_effect"], errors="coerce")
        truth_effect = pd.to_numeric(table["truth_occurrence_effect"], errors="coerce")
        effect_valid = usable & predicted.notna() & np.isfinite(predicted)
        errors = (
            predicted.loc[effect_valid].to_numpy()
            - truth_effect.loc[effect_valid].to_numpy()
        )
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else None
        bias = float(np.mean(errors)) if len(errors) else None
        nonzero = effect_valid & truth_effect.ne(0.0)
        direction = (
            float(
                np.mean(
                    np.sign(predicted.loc[nonzero].to_numpy())
                    == np.sign(truth_effect.loc[nonzero].to_numpy())
                )
            )
            if nonzero.any()
            else None
        )
        for metric, value, reason in (
            ("prevalence_effect_rmse", rmse, "prevalence_effect_unavailable"),
            ("prevalence_effect_bias", bias, "prevalence_effect_unavailable"),
            ("direction_accuracy", direction, "no_nonzero_estimable_truth"),
            (
                "odds_ratio_bias",
                None,
                "finite_truth_odds_ratio_not_defined_by_v7_dgp",
            ),
            (
                "log_odds_rmse",
                None,
                "finite_truth_log_odds_not_defined_by_v7_dgp",
            ),
        ):
            rows.append(
                _metric_row(
                    first,
                    metric=metric,
                    value=value,
                    reason_code=reason,
                    n_observed=int(effect_valid.sum()),
                    n_positive=n_positive,
                    n_negative=n_negative,
                )
            )
        p_value = pd.to_numeric(table["p_value"], errors="coerce")
        q_value = pd.to_numeric(table["q_value"], errors="coerce")
        null = known & ~labels & p_value.notna() & np.isfinite(p_value)
        calls_valid = known & q_value.notna() & np.isfinite(q_value)
        calls = calls_valid & q_value.le(0.10)
        type1 = float(p_value.loc[null].lt(0.05).mean()) if null.any() else None
        fdr = (
            float((calls & ~labels).sum() / calls.sum())
            if calls.any()
            else (0.0 if calls_valid.any() else None)
        )
        positives = calls_valid & labels
        power = float(calls.loc[positives].mean()) if positives.any() else None
        for metric, value, reason in (
            ("type1_error_alpha_0_05", type1, "formal_p_values_unavailable"),
            ("fdr_at_q_0_10", fdr, "formal_q_values_unavailable"),
            ("power_at_q_0_10", power, "no_positive_truth_with_q_values"),
        ):
            rows.append(
                _metric_row(
                    first,
                    metric=metric,
                    value=value,
                    reason_code=reason,
                    n_observed=int(calls_valid.sum()),
                    n_positive=n_positive,
                    n_negative=n_negative,
                )
            )
    return rows


def _calibration_fit(
    truth: np.ndarray,
    probability: np.ndarray,
) -> tuple[float | None, float | None]:
    if len(truth) < 4 or len(np.unique(truth)) != 2:
        return None, None
    logit = np.log(np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)) - np.log1p(
        -np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    )
    design = np.column_stack([np.ones(len(logit)), logit])
    beta = np.zeros(2, dtype=float)
    for _ in range(100):
        fitted = expit(np.clip(design @ beta, -30.0, 30.0))
        variance = np.maximum(fitted * (1.0 - fitted), 1.0e-8)
        information = design.T @ (variance[:, np.newaxis] * design)
        if np.linalg.matrix_rank(information) < 2:
            return None, None
        step = np.linalg.solve(information, design.T @ (truth - fitted))
        beta += step
        if float(np.max(np.abs(step))) < 1.0e-8:
            break
        if float(np.max(np.abs(beta))) > 50.0:
            return None, None
    if not np.isfinite(beta).all():
        return None, None
    return float(beta[0]), float(beta[1])


def _calibration_metrics(subject_events: pd.DataFrame) -> list[dict[str, object]]:
    known = subject_events["truth_state_known"].astype(bool)
    truth = subject_events.loc[known, "truth_occurrence_state"].astype(bool).to_numpy()
    rows: list[dict[str, object]] = []
    for method, column in (
        (M4_PROBABILITY_METHOD, "active_probability"),
        (RAW_PROBABILITY_METHOD, "raw_leave_one_out_prevalence"),
    ):
        prediction = pd.to_numeric(subject_events.loc[known, column], errors="coerce")
        finite = prediction.notna() & np.isfinite(prediction)
        local_truth = truth[finite.to_numpy()]
        local_probability = prediction.loc[finite].to_numpy(dtype=float)
        n_positive = int(local_truth.sum())
        n_negative = int(len(local_truth) - n_positive)
        first = pd.Series(
            {
                "dataset_id": subject_events["dataset_id"].iloc[0],
                "dgp_family": subject_events["dgp_family"].iloc[0],
                "design_kind": subject_events["design_kind"].iloc[0],
                "method": method,
                "contrast_name": "__subject_state__",
            }
        )
        brier = (
            float(np.mean(np.square(local_probability - local_truth)))
            if len(local_truth)
            else None
        )
        intercept, slope = _calibration_fit(
            local_truth.astype(float), local_probability
        )
        for metric, value, reason in (
            ("subject_brier_score", brier, "subject_truth_or_probability_unavailable"),
            ("calibration_intercept", intercept, "calibration_model_not_estimable"),
            ("calibration_slope", slope, "calibration_model_not_estimable"),
        ):
            rows.append(
                _metric_row(
                    first,
                    metric=metric,
                    value=value,
                    reason_code=reason,
                    n_observed=len(local_truth),
                    n_positive=n_positive,
                    n_negative=n_negative,
                )
            )
    return rows


@dataclass(frozen=True, slots=True)
class V7M4OccurrenceBenchmarkResult:
    """M4 subject states, effects, comparator arms, and typed metrics."""

    subject_events: pd.DataFrame
    effects: pd.DataFrame
    aligned_effects: pd.DataFrame
    baselines: pd.DataFrame
    metrics: pd.DataFrame
    occurrence_manifest: dict[str, object]

    def __post_init__(self) -> None:
        contracts = (
            (self.subject_events, M4_SUBJECT_EVENT_COLUMNS, "subject_events"),
            (self.effects, M4_EFFECT_COLUMNS, "effects"),
            (self.aligned_effects, M4_ALIGNED_EFFECT_COLUMNS, "aligned_effects"),
            (self.baselines, M4_BASELINE_COLUMNS, "baselines"),
            (self.metrics, M4_METRIC_COLUMNS, "metrics"),
        )
        for table, columns, name in contracts:
            if not isinstance(table, pd.DataFrame) or tuple(table.columns) != columns:
                raise ValueError(f"M4 {name} does not match its frozen contract")
            object.__setattr__(self, name, table.copy(deep=True))
        if set(self.baselines["method"].astype(str)) != set(M4_METHODS):
            raise ValueError("M4 baseline table does not cover the full method axis")

    def to_manifest(self) -> dict[str, object]:
        observed = self.baselines["status"].eq("observed")
        descriptive = self.baselines["status"].eq("descriptive")
        return {
            "schema_version": SCHEMA_VERSION,
            "methods": list(M4_METHODS),
            "subject_event_rows": len(self.subject_events),
            "effect_rows": len(self.effects),
            "observed_method_rows": int(observed.sum()),
            "descriptive_method_rows": int(descriptive.sum()),
            "not_estimable_method_rows": int(
                self.baselines["status"].eq("not_estimable").sum()
            ),
            "occurrence": dict(self.occurrence_manifest),
            "dcst_scope": "protocol-compatible Fisher proxy, not the external package",
            "beta_binomial_scope": (
                "typed not-estimable because one Bernoulli trial per subject does "
                "not identify overdispersion"
            ),
        }


def run_v7_m4_occurrence_benchmark(
    crossfit: CrossFitArtifacts,
    *,
    dataset_id: str,
    design: DifferentialDesignSpec,
    sample_metadata: pd.DataFrame,
    truth: pd.DataFrame,
    dgp_family: str,
    design_kind: str,
    activity_head: str = "parent_mean_raw",
    activity_threshold_raw: float = 1.0,
    probability_transition_scale: float = 0.25,
    occurrence_state_source: str = "raw_activity_threshold",
    active_probability_threshold: float = 0.8,
    active_probability_mapping: str | None = None,
) -> V7M4OccurrenceBenchmarkResult:
    """Run the fixed M4 head and honest estimand-matched comparator panel."""

    dataset = _name(dataset_id, field="dataset_id")
    family = _name(dgp_family, field="dgp_family")
    design_name = _name(design_kind, field="design_kind")
    spec = TwoPartOccurrenceV2Spec(
        design=design,
        activity_head=activity_head,
        activity_threshold_raw=activity_threshold_raw,
        probability_transition_scale=probability_transition_scale,
        occurrence_state_source=occurrence_state_source,
        active_probability_threshold=active_probability_threshold,
        active_probability_mapping=active_probability_mapping,
    )
    fitted = fit_crossfit_two_part_occurrence_v2(
        crossfit,
        spec,
        sample_metadata=sample_metadata,
    )
    if spec.occurrence_state_source == "raw_activity_threshold":
        raw_comparator = fitted
    else:
        raw_comparator = fit_crossfit_two_part_occurrence_v2(
            crossfit,
            TwoPartOccurrenceV2Spec(
                design=design,
                activity_head=activity_head,
                activity_threshold_raw=activity_threshold_raw,
                probability_transition_scale=probability_transition_scale,
                occurrence_state_source="raw_activity_threshold",
            ),
            sample_metadata=sample_metadata,
        )
    parent_truth = _parent_truth(
        truth,
        activity_head=spec.activity_head,
        event_universe=_event_universe(crossfit, spec),
    )
    subject_events = _decorate_subject_events(
        fitted.occurrence.subject_events,
        raw_comparator.occurrence.subject_events,
        parent_truth,
        sample_metadata,
        dataset_id=dataset,
        dgp_family=family,
        design_kind=design_name,
        condition_column=design.condition_column,
    )
    effects, aligned = _decorate_effects(
        fitted.occurrence.effects,
        parent_truth,
        dataset_id=dataset,
        dgp_family=family,
        design_kind=design_name,
    )
    baselines = _baseline_table(
        effects,
        raw_comparator.occurrence.effects,
        raw_comparator.occurrence.subject_events,
        design,
    )
    metric_rows = [
        *_effect_metrics(baselines, aligned),
        *_calibration_metrics(subject_events),
    ]
    metrics = pd.DataFrame.from_records(
        metric_rows, columns=M4_METRIC_COLUMNS
    ).sort_values(
        ["method", "contrast_name", "metric"],
        kind="stable",
        ignore_index=True,
    )
    return V7M4OccurrenceBenchmarkResult(
        subject_events=subject_events,
        effects=effects,
        aligned_effects=aligned,
        baselines=baselines,
        metrics=metrics,
        occurrence_manifest={
            **fitted.to_manifest(),
            "raw_comparator_result_id": raw_comparator.result_id,
            "raw_comparator_state_source": "raw_activity_threshold",
        },
    )


__all__ = [
    "BETA_BINOMIAL_METHOD",
    "DCST_METHOD",
    "FISHER_METHOD",
    "LOGISTIC_METHOD",
    "M4_ALIGNED_EFFECT_COLUMNS",
    "M4_BASELINE_COLUMNS",
    "M4_EFFECT_COLUMNS",
    "M4_EFFECT_METRICS",
    "M4_METHOD",
    "M4_METHODS",
    "M4_METRIC_COLUMNS",
    "M4_SUBJECT_EVENT_COLUMNS",
    "RAW_METHOD",
    "SCHEMA_VERSION",
    "V7M4OccurrenceBenchmarkResult",
    "run_v7_m4_occurrence_benchmark",
]
