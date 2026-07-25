"""Preregistered E3 sender detection and attribution component swaps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.simulation.v7_integrated import (
    EFFECT_COLUMNS,
    SCORE_VIEW_COLUMNS,
    V7InferenceFitCache,
    build_v7_score_views,
    run_v7_inference_matrix,
)
from benchmarks.simulation.v7_metrics import (
    ALIGNED_EFFECT_COLUMNS,
    METRIC_COLUMNS,
    align_v7_effect_truth,
    evaluate_v7_integrated_matrix,
)
from crychic.inference import DifferentialDesignSpec
from crychic.sender import apply_sender_attribution_v2
from crychic.workflow import CrossFitArtifacts

SCHEMA_VERSION = "crychic-suggest-next2-v7-e3-sender-swap-v1"
E3_DETECTION_ARM = "nonconserving_sender_detection"
E3_LEGACY_ARM = "legacy_softmax_assignment"
E3_NULL_SENDER_ARM = "null_sender_attribution"
E3_M2_ARM = "null_sender_attribution_plus_m2"
E3_ARMS = (
    E3_DETECTION_ARM,
    E3_LEGACY_ARM,
    E3_NULL_SENDER_ARM,
    E3_M2_ARM,
)
E3_ARM_GENERATORS = {
    E3_DETECTION_ARM: "G3",
    E3_LEGACY_ARM: "G1",
    E3_NULL_SENDER_ARM: "G5",
    E3_M2_ARM: "G5",
}

E3_AUXILIARY_COLUMNS = (
    "schema_version",
    "dataset_id",
    "score_view",
    "contrast_scope",
    "fold_id",
    "sample_id",
    "subject_id",
    "condition",
    "context_id",
    "receiver",
    "interaction_id",
    "candidate_sender_count",
    "max_attribution",
    "null_sender_attribution",
    "attribution_entropy",
    "parent_score",
    "status",
    "reason_code",
)
E3_METRIC_COLUMNS = (
    "schema_version",
    "dataset_id",
    "dgp_family",
    "design_kind",
    "score_view",
    "contrast_name",
    "candidate_sender_count",
    "metric",
    "value",
    "status",
    "reason_code",
    "n_observed",
    "n_positive",
    "n_negative",
)
E3_METRICS = (
    "sender_auprc",
    "sender_auroc",
    "top1_sender_accuracy",
    "diagnostic_false_positive_rate_alpha_0_05",
    "mean_max_attribution",
    "mean_attribution_entropy",
    "mean_parent_score",
)

_EDGE_KEY = (
    "fold_id",
    "sample_id",
    "subject_id",
    "condition",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
)
_PARENT_KEY = (
    "fold_id",
    "sample_id",
    "subject_id",
    "condition",
    "context_id",
    "receiver",
    "interaction_id",
)
_PARENT_LOOKUP_KEY = tuple(column for column in _PARENT_KEY if column != "condition")


def _name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _reset_attribution_heads(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    result["coupling_prior"] = np.nan
    result["coupling_status"] = "not_computed"
    result["coupling_reason_code"] = "coupling_not_computed"
    result["coupling_functional_id"] = None
    result["active_probability"] = np.nan
    result["occurrence_status"] = "not_computed"
    result["occurrence_reason_code"] = "active_probability_not_computed"
    result["occurrence_functional_id"] = None
    result["sender_attribution"] = np.nan
    result["null_sender_attribution"] = np.nan
    result["attribution_entropy"] = np.nan
    result["attribution_status"] = "not_computed"
    result["attribution_reason_code"] = "sender_attribution_not_computed"
    result["attribution_functional_id"] = None
    return result


def _no_m2_tables(crossfit: CrossFitArtifacts) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in crossfit.folds:
        scores = fold.sample_edge_scores_v2
        functional = fold.sender_attribution_v2_functional
        transform = fold.absolute_activity_v2_transform
        if scores is None or functional is None or transform is None:
            raise ValueError("E3 requires M0 v2 and sender v2 in every fold")
        frames.append(
            apply_sender_attribution_v2(
                functional,
                _reset_attribution_heads(scores.table),
                activity_transform_id=transform.transform_manifest_id,
                coupling_functional=None,
            )
        )
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(list(_EDGE_KEY)).any():
        raise ValueError("E3 no-M2 application contains duplicate OOF child rows")
    return result


def _legacy_assignment_table(crossfit: CrossFitArtifacts) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in crossfit.folds:
        for application in fold.family_common_applications:
            table = application.sender_scores
            if "mode" in table:
                table = table.loc[table["mode"].astype(str).eq("state")]
            if table.empty:
                continue
            frames.append(
                table.loc[
                    :,
                    [
                        "sample_id",
                        "subject_id",
                        "context_id",
                        "sender",
                        "receiver",
                        "interaction_id",
                        "assignment_weight",
                    ],
                ].assign(
                    fold_id=str(fold.fold_id),
                    contrast_scope=str(application.functional.contrast_name),
                )
            )
    if not frames:
        raise ValueError("E3 legacy softmax arm has no held-out assignment rows")
    result = pd.concat(frames, ignore_index=True)
    key = [
        "contrast_scope",
        "fold_id",
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
    ]
    if result.duplicated(key).any():
        raise ValueError("E3 legacy softmax keys are not unique")
    return result


def _head_status(values: pd.Series, statuses: pd.Series) -> pd.Series:
    result = (
        statuses.astype(str)
        .map(
            {
                "observed": "observed",
                "partial": "low_evidence",
                "not_estimable": "not_estimable",
                "not_computed": "not_estimable",
            }
        )
        .fillna("not_estimable")
    )
    return result.where(pd.to_numeric(values, errors="coerce").notna(), "not_estimable")


def _replace_view(
    source: pd.DataFrame,
    *,
    arm: str,
    score: pd.Series,
    status: pd.Series,
    estimand: str,
    outcome_agnostic: bool,
    condition_gate_used: bool,
    source_version: str,
) -> pd.DataFrame:
    result = source.loc[:, list(SCORE_VIEW_COLUMNS)].copy(deep=True)
    result["schema_version"] = SCHEMA_VERSION
    result["generator_id"] = E3_ARM_GENERATORS[arm]
    result["score_view"] = arm
    result["estimand"] = estimand
    result["score"] = pd.to_numeric(score, errors="coerce").astype(float)
    result["score_status"] = status.astype(str)
    result["outcome_agnostic"] = outcome_agnostic
    result["condition_gate_used"] = condition_gate_used
    result["primary_view"] = arm == E3_DETECTION_ARM
    result["source_version"] = source_version
    return result


def _normalized_entropy(values: pd.Series, *, include_null: float | None) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if include_null is not None and math.isfinite(include_null):
        finite = np.concatenate((finite, np.asarray([include_null], dtype=float)))
    finite = finite[finite > 0.0]
    if not len(finite):
        return float("nan")
    finite = finite / float(finite.sum())
    if len(finite) <= 1:
        return 0.0
    return float(-np.sum(finite * np.log(finite)) / math.log(len(finite)))


def _auxiliary_table(
    table: pd.DataFrame,
    *,
    dataset_id: str,
    arm: str,
    contrast_scope: str,
    score_column: str,
    attribution: bool,
    parent_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source = table.copy(deep=True)
    if parent_lookup is not None:
        source = source.merge(
            parent_lookup,
            on=list(_PARENT_LOOKUP_KEY),
            how="left",
            validate="many_to_one",
            sort=False,
        )
    records: list[dict[str, object]] = []
    for key, group in source.groupby(list(_PARENT_KEY), observed=True, sort=False):
        first = group.iloc[0]
        score = pd.to_numeric(group[score_column], errors="coerce")
        null_value = None
        if (
            attribution
            and "null_sender_attribution" in group
            and pd.notna(first["null_sender_attribution"])
        ):
            null_value = float(first["null_sender_attribution"])
        elif arm == E3_LEGACY_ARM:
            null_value = 0.0
        max_attribution = (
            float(score.max()) if attribution and score.notna().any() else np.nan
        )
        entropy = (
            _normalized_entropy(score, include_null=null_value)
            if attribution and score.notna().any()
            else np.nan
        )
        parent_score = pd.to_numeric(
            pd.Series([first["parent_mean_raw"]]), errors="coerce"
        ).iloc[0]
        candidate_count = int(group["sender"].astype(str).nunique())
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "score_view": arm,
                "contrast_scope": contrast_scope,
                **dict(zip(_PARENT_KEY, key, strict=True)),
                "candidate_sender_count": candidate_count,
                "max_attribution": max_attribution,
                "null_sender_attribution": null_value,
                "attribution_entropy": entropy,
                "parent_score": parent_score,
                "status": "observed" if score.notna().any() else "not_estimable",
                "reason_code": (
                    None if score.notna().any() else "sender_arm_has_no_finite_scores"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=E3_AUXILIARY_COLUMNS)


def build_e3_sender_swap_score_views(
    crossfit: CrossFitArtifacts,
    *,
    dataset_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the four E3 sender arms on the same OOF sample axis."""

    if not isinstance(crossfit, CrossFitArtifacts):
        raise TypeError("crossfit must be CrossFitArtifacts")
    dataset = _name(dataset_id, field="dataset_id")
    all_views = build_v7_score_views(crossfit, dataset_id=dataset)
    detection = all_views.loc[
        all_views["generator_id"].eq("G3")
        & all_views["score_view"].eq("primary_sender_detection")
        & all_views["contrast_scope"].eq("__all__")
    ].reset_index(drop=True)
    with_m2 = all_views.loc[
        all_views["generator_id"].eq("G5")
        & all_views["score_view"].eq("sender_attribution_with_m2")
        & all_views["contrast_scope"].eq("__all__")
    ].reset_index(drop=True)
    legacy_view = all_views.loc[
        all_views["generator_id"].eq("G1")
        & all_views["score_view"].eq("primary_sender_resolved_state")
    ].reset_index(drop=True)
    if detection.empty or with_m2.empty or legacy_view.empty:
        raise ValueError("E3 requires connected G1, G3, and G5 score heads")

    no_m2 = _no_m2_tables(crossfit)
    no_m2_values = detection.loc[:, list(_EDGE_KEY)].merge(
        no_m2.loc[
            :,
            [
                *_EDGE_KEY,
                "sender_attribution",
                "null_sender_attribution",
                "attribution_entropy",
                "attribution_status",
                "parent_mean_raw",
            ],
        ],
        on=list(_EDGE_KEY),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    legacy_assignment = _legacy_assignment_table(crossfit)
    legacy_keys = [
        "contrast_scope",
        "fold_id",
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
    ]
    legacy_values = legacy_view.loc[:, legacy_keys].merge(
        legacy_assignment,
        on=legacy_keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    frames = [
        _replace_view(
            detection,
            arm=E3_DETECTION_ARM,
            score=detection["score"],
            status=detection["score_status"],
            estimand="sample_comparable_lr_intensity",
            outcome_agnostic=True,
            condition_gate_used=False,
            source_version="nonconserving_m0_v2_sender_detection",
        ),
        _replace_view(
            legacy_view,
            arm=E3_LEGACY_ARM,
            score=legacy_values["assignment_weight"],
            status=pd.Series(
                np.where(
                    pd.to_numeric(
                        legacy_values["assignment_weight"], errors="coerce"
                    ).notna(),
                    "observed",
                    "not_estimable",
                ),
                index=legacy_view.index,
            ),
            estimand="conditional_sender_attribution",
            outcome_agnostic=False,
            condition_gate_used=True,
            source_version="legacy_softmax_sender_assignment_v2",
        ),
        _replace_view(
            detection,
            arm=E3_NULL_SENDER_ARM,
            score=no_m2_values["sender_attribution"],
            status=_head_status(
                no_m2_values["sender_attribution"],
                no_m2_values["attribution_status"],
            ),
            estimand="conditional_sender_attribution",
            outcome_agnostic=True,
            condition_gate_used=False,
            source_version="null_sender_entmax15_without_m2_v1",
        ),
        _replace_view(
            with_m2,
            arm=E3_M2_ARM,
            score=with_m2["score"],
            status=with_m2["score_status"],
            estimand="conditional_sender_attribution",
            outcome_agnostic=True,
            condition_gate_used=False,
            source_version="null_sender_entmax15_plus_eb_coupling_v2",
        ),
    ]
    score_views = pd.concat(frames, ignore_index=True).loc[:, list(SCORE_VIEW_COLUMNS)]
    score_views = score_views.sort_values(
        ["score_view", "contrast_scope", "event_id", "fold_id", "sample_id"],
        kind="stable",
        ignore_index=True,
    )
    if set(score_views["score_view"].astype(str)) != set(E3_ARMS):
        raise RuntimeError("E3 did not produce every preregistered sender arm")
    if score_views.duplicated(
        ["generator_id", "score_view", "contrast_scope", "event_id", "sample_id"]
    ).any():
        raise ValueError("E3 score views contain duplicate sample-event rows")

    current = pd.concat(
        [fold.sample_edge_scores_v2.table for fold in crossfit.folds],
        ignore_index=True,
    )
    parent_lookup = current.loc[
        :, [*_PARENT_LOOKUP_KEY, "condition", "parent_mean_raw"]
    ].drop_duplicates(list(_PARENT_LOOKUP_KEY))
    auxiliary = pd.concat(
        [
            _auxiliary_table(
                current,
                dataset_id=dataset,
                arm=E3_DETECTION_ARM,
                contrast_scope="__all__",
                score_column="sender_detection_raw",
                attribution=False,
            ),
            *(
                _auxiliary_table(
                    group,
                    dataset_id=dataset,
                    arm=E3_LEGACY_ARM,
                    contrast_scope=str(scope),
                    score_column="assignment_weight",
                    attribution=True,
                    parent_lookup=parent_lookup,
                )
                for scope, group in legacy_assignment.groupby(
                    "contrast_scope", observed=True, sort=True
                )
            ),
            _auxiliary_table(
                no_m2,
                dataset_id=dataset,
                arm=E3_NULL_SENDER_ARM,
                contrast_scope="__all__",
                score_column="sender_attribution",
                attribution=True,
            ),
            _auxiliary_table(
                current,
                dataset_id=dataset,
                arm=E3_M2_ARM,
                contrast_scope="__all__",
                score_column="sender_attribution",
                attribution=True,
            ),
        ],
        ignore_index=True,
    ).loc[:, list(E3_AUXILIARY_COLUMNS)]
    auxiliary = auxiliary.sort_values(
        [
            "score_view",
            "contrast_scope",
            "fold_id",
            "sample_id",
            "receiver",
            "interaction_id",
        ],
        kind="stable",
        ignore_index=True,
    )
    return score_views, auxiliary


def _metric_row(
    *,
    dataset_id: str,
    dgp_family: str,
    design_kind: str,
    score_view: str,
    contrast_name: str,
    candidate_sender_count: int,
    metric: str,
    value: float | None,
    n_observed: int,
    n_positive: int,
    n_negative: int,
    reason_code: str | None = None,
) -> dict[str, object]:
    observed = value is not None and math.isfinite(value)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dgp_family": dgp_family,
        "design_kind": design_kind,
        "score_view": score_view,
        "contrast_name": contrast_name,
        "candidate_sender_count": candidate_sender_count,
        "metric": metric,
        "value": value if observed else None,
        "status": "observed" if observed else "not_estimable",
        "reason_code": None if observed else reason_code or "metric_not_estimable",
        "n_observed": n_observed,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def summarize_e3_sender_metrics(
    effects: pd.DataFrame,
    auxiliary: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    dgp_family: str,
    design_kind: str,
    candidate_sender_count: int,
) -> pd.DataFrame:
    """Summarize sender identity and attribution geometry for every E3 arm."""

    if tuple(effects.columns) != EFFECT_COLUMNS:
        raise ValueError("effects do not match the integrated v7 contract")
    if tuple(auxiliary.columns) != E3_AUXILIARY_COLUMNS:
        raise ValueError("auxiliary does not match the E3 contract")
    if (
        isinstance(candidate_sender_count, bool)
        or not isinstance(candidate_sender_count, int)
        or candidate_sender_count < 1
    ):
        raise ValueError("candidate_sender_count must be a positive integer")
    aligned = align_v7_effect_truth(effects, truth)
    records: list[dict[str, object]] = []
    for (score_view, contrast_name), group in aligned.groupby(
        ["score_view", "contrast_name"], observed=True, sort=True
    ):
        known = group.loc[group["status"].eq("observed") & group["truth_known"]].copy()
        ranking = pd.to_numeric(known["ranking_score"], errors="coerce").abs()
        usable = ranking.notna()
        known = known.loc[usable].copy()
        ranking = ranking.loc[usable]
        labels = known["truth_label"].astype(bool).to_numpy()
        scores = ranking.to_numpy(dtype=float)
        n_positive = int(labels.sum())
        n_negative = int((~labels).sum())
        auprc = auroc = None
        binary_reason = None
        if len(labels) and n_positive and n_negative:
            auprc = float(average_precision_score(labels, scores))
            auroc = float(roc_auc_score(labels, scores))
        else:
            binary_reason = "truth_is_single_class_or_unobserved"

        top1: list[float] = []
        for _, parent in known.groupby(
            ["receiver", "interaction_id"], observed=True, sort=False
        ):
            parent_scores = pd.to_numeric(
                parent["ranking_score"], errors="coerce"
            ).abs()
            if not parent["truth_label"].astype(bool).any() or parent_scores.empty:
                continue
            winners = parent.loc[parent_scores.eq(float(parent_scores.max()))]
            top1.append(float(winners["truth_label"].astype(bool).any()))
        negative_p = pd.to_numeric(
            known.loc[~known["truth_label"].astype(bool), "diagnostic_p_value"],
            errors="coerce",
        ).dropna()
        fpr = float(negative_p.lt(0.05).mean()) if len(negative_p) else None

        arm_auxiliary = auxiliary.loc[auxiliary["score_view"].eq(score_view)]
        exact_auxiliary = arm_auxiliary.loc[
            arm_auxiliary["contrast_scope"].eq(contrast_name)
        ]
        if not exact_auxiliary.empty:
            arm_auxiliary = exact_auxiliary
        else:
            arm_auxiliary = arm_auxiliary.loc[
                arm_auxiliary["contrast_scope"].eq("__all__")
            ]
        aux_count = len(arm_auxiliary)
        max_attribution = pd.to_numeric(
            arm_auxiliary["max_attribution"], errors="coerce"
        ).dropna()
        entropy = pd.to_numeric(
            arm_auxiliary["attribution_entropy"], errors="coerce"
        ).dropna()
        parent_score = pd.to_numeric(
            arm_auxiliary["parent_score"], errors="coerce"
        ).dropna()
        common = {
            "dataset_id": str(group["dataset_id"].iloc[0]),
            "dgp_family": dgp_family,
            "design_kind": design_kind,
            "score_view": str(score_view),
            "contrast_name": str(contrast_name),
            "candidate_sender_count": candidate_sender_count,
            "n_observed": len(known),
            "n_positive": n_positive,
            "n_negative": n_negative,
        }
        records.extend(
            [
                _metric_row(
                    **common,
                    metric="sender_auprc",
                    value=auprc,
                    reason_code=binary_reason,
                ),
                _metric_row(
                    **common,
                    metric="sender_auroc",
                    value=auroc,
                    reason_code=binary_reason,
                ),
                _metric_row(
                    **common,
                    metric="top1_sender_accuracy",
                    value=float(np.mean(top1)) if top1 else None,
                    reason_code="no_parent_with_observed_true_sender",
                ),
                _metric_row(
                    **common,
                    metric="diagnostic_false_positive_rate_alpha_0_05",
                    value=fpr,
                    reason_code="no_truth_negative_diagnostic_p_values",
                ),
                _metric_row(
                    **{**common, "n_observed": aux_count},
                    metric="mean_max_attribution",
                    value=(
                        float(max_attribution.mean()) if len(max_attribution) else None
                    ),
                    reason_code="arm_does_not_produce_attribution",
                ),
                _metric_row(
                    **{**common, "n_observed": aux_count},
                    metric="mean_attribution_entropy",
                    value=float(entropy.mean()) if len(entropy) else None,
                    reason_code="arm_does_not_produce_attribution",
                ),
                _metric_row(
                    **{**common, "n_observed": aux_count},
                    metric="mean_parent_score",
                    value=float(parent_score.mean()) if len(parent_score) else None,
                    reason_code="parent_score_not_estimable",
                ),
            ]
        )
    result = pd.DataFrame.from_records(records, columns=E3_METRIC_COLUMNS)
    return result.sort_values(
        ["score_view", "contrast_name", "metric"],
        kind="stable",
        ignore_index=True,
    )


@dataclass(frozen=True, slots=True)
class V7E3SenderSwapResult:
    """Complete descriptive E3 result on one generated dataset."""

    score_views: pd.DataFrame
    auxiliary: pd.DataFrame
    effects: pd.DataFrame
    aligned_effects: pd.DataFrame
    metrics: pd.DataFrame
    sender_metrics: pd.DataFrame

    def __post_init__(self) -> None:
        expected = (
            (self.score_views, SCORE_VIEW_COLUMNS, "score_views"),
            (self.auxiliary, E3_AUXILIARY_COLUMNS, "auxiliary"),
            (self.effects, EFFECT_COLUMNS, "effects"),
            (self.aligned_effects, ALIGNED_EFFECT_COLUMNS, "aligned_effects"),
            (self.metrics, METRIC_COLUMNS, "metrics"),
            (self.sender_metrics, E3_METRIC_COLUMNS, "sender_metrics"),
        )
        for table, columns, name in expected:
            if not isinstance(table, pd.DataFrame) or tuple(table.columns) != columns:
                raise ValueError(f"E3 {name} does not match its frozen contract")
            object.__setattr__(self, name, table.copy(deep=True))


def run_v7_e3_sender_swap(
    crossfit: CrossFitArtifacts,
    *,
    dataset_id: str,
    design: DifferentialDesignSpec,
    sample_metadata: pd.DataFrame,
    truth: pd.DataFrame,
    dgp_family: str,
    design_kind: str,
    candidate_sender_count: int,
    fit_cache: V7InferenceFitCache | None = None,
) -> V7E3SenderSwapResult:
    """Run every E3 sender arm through identical I1 and sender metrics."""

    score_views, auxiliary = build_e3_sender_swap_score_views(
        crossfit,
        dataset_id=dataset_id,
    )
    arms = tuple(
        sorted({(generator, "I1") for generator in E3_ARM_GENERATORS.values()})
    )
    effects = run_v7_inference_matrix(
        score_views,
        design=design,
        sample_metadata=sample_metadata,
        arms=arms,
        fit_cache=fit_cache,
    )
    family = _name(dgp_family, field="dgp_family")
    design_name = _name(design_kind, field="design_kind")
    aligned, metrics = evaluate_v7_integrated_matrix(
        score_views,
        effects,
        truth,
        dgp_family=family,
        design_kind=design_name,
    )
    sender_metrics = summarize_e3_sender_metrics(
        effects,
        auxiliary,
        truth,
        dgp_family=family,
        design_kind=design_name,
        candidate_sender_count=candidate_sender_count,
    )
    return V7E3SenderSwapResult(
        score_views=score_views,
        auxiliary=auxiliary,
        effects=effects,
        aligned_effects=aligned,
        metrics=metrics,
        sender_metrics=sender_metrics,
    )


__all__ = [
    "E3_ARMS",
    "E3_ARM_GENERATORS",
    "E3_AUXILIARY_COLUMNS",
    "E3_DETECTION_ARM",
    "E3_LEGACY_ARM",
    "E3_M2_ARM",
    "E3_METRICS",
    "E3_METRIC_COLUMNS",
    "E3_NULL_SENDER_ARM",
    "SCHEMA_VERSION",
    "V7E3SenderSwapResult",
    "build_e3_sender_swap_score_views",
    "run_v7_e3_sender_swap",
    "summarize_e3_sender_metrics",
]
