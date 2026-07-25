"""Preregistered E2 hard-gate component swaps for the v7 benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    evaluate_v7_integrated_matrix,
)
from crychic.inference import DifferentialDesignSpec
from crychic.workflow import (
    GATE_STAGE_LEDGER_COLUMNS,
    CrossFitArtifacts,
    build_v7_gate_stage_ledger,
)

SCHEMA_VERSION = "crychic-suggest-next2-v7-e2-component-swap-v1"
E2_GENERATOR_ID = "G3"
E2_BASE_ARM = "base_g3_i1"
E2_ANNOTATION_ARMS = {
    "receptor_eligibility_annotation": "receptor_eligible",
    "ligand_contrast_annotation": "ligand_contrast_supported",
    "family_selection_annotation": "family_selected",
    "downstream_support_annotation": "downstream_supported",
}
E2_HARD_GATE_ARMS = {
    "receptor_eligibility_gate": "receptor_eligible",
    "ligand_contrast_gate": "ligand_contrast_supported",
    "family_selection_gate": "family_selected",
    "downstream_support_gate": "downstream_supported",
}
E2_ARMS = (
    E2_BASE_ARM,
    *E2_ANNOTATION_ARMS,
    *E2_HARD_GATE_ARMS,
)

_JOIN_KEY = (
    "fold_id",
    "sample_id",
    "subject_id",
    "condition",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
)
_USABLE_SCORE_STATUSES = {"observed", "low_evidence", "structural_impossible"}


def _name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _arm_view(
    linked: pd.DataFrame,
    *,
    arm: str,
    contrast_name: str,
    stage: str | None,
    hard_gate: bool,
) -> pd.DataFrame:
    result = linked.loc[:, list(SCORE_VIEW_COLUMNS)].copy(deep=True)
    result["schema_version"] = SCHEMA_VERSION
    result["generator_id"] = E2_GENERATOR_ID
    result["score_view"] = arm
    result["contrast_scope"] = contrast_name
    result["condition_gate_used"] = hard_gate
    result["outcome_agnostic"] = not hard_gate
    result["primary_view"] = arm == E2_BASE_ARM
    result["source_version"] = (
        "m0_v2_gate_annotation_only_v1"
        if stage is not None and not hard_gate
        else "m0_v2_single_legacy_hard_gate_v1"
        if hard_gate
        else "m0_v2_component_swap_reference_v1"
    )
    if not hard_gate:
        return result
    if stage is None:  # pragma: no cover - internal invariant
        raise RuntimeError("hard-gate arm requires a stage")
    gate_status = linked[f"{stage}_status"].astype(str)
    base_status = result["score_status"].astype(str)
    base_usable = base_status.isin(_USABLE_SCORE_STATUSES) & result["score"].notna()
    failed = gate_status.eq("failed") & base_usable
    retained = gate_status.eq("passed")
    result.loc[failed, "score"] = 0.0
    result.loc[failed, "score_status"] = "low_evidence"
    unavailable = ~(retained | failed)
    result.loc[unavailable, "score"] = np.nan
    result.loc[unavailable, "score_status"] = "not_estimable"
    return result


def build_e2_component_swap_score_views(
    crossfit: CrossFitArtifacts,
    *,
    dataset_id: str,
    gate_stage_ledger: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build matched annotation-only and single-hard-gate G3 score views."""

    if not isinstance(crossfit, CrossFitArtifacts):
        raise TypeError("crossfit must be CrossFitArtifacts")
    dataset = _name(dataset_id, field="dataset_id")
    ledger = (
        build_v7_gate_stage_ledger(crossfit, dataset_id=dataset)
        if gate_stage_ledger is None
        else gate_stage_ledger.copy(deep=True)
    )
    if tuple(ledger.columns) != GATE_STAGE_LEDGER_COLUMNS:
        raise ValueError("gate_stage_ledger does not match its frozen contract")
    if set(ledger["dataset_id"].astype(str)) != {dataset}:
        raise ValueError("gate_stage_ledger dataset_id differs from the request")
    base = build_v7_score_views(crossfit, dataset_id=dataset)
    base = base.loc[
        base["generator_id"].eq("G3")
        & base["score_view"].eq("primary_sender_detection")
        & base["contrast_scope"].eq("__all__")
    ].copy()
    if base.empty or base.duplicated(list(_JOIN_KEY)).any():
        raise ValueError("E2 requires one unique G3 sender score per OOF child")

    frames: list[pd.DataFrame] = []
    for contrast_name, contrast_ledger in ledger.groupby(
        "contrast_name", observed=True, sort=True
    ):
        status_columns = [
            f"{stage}_status"
            for stage in {*E2_ANNOTATION_ARMS.values(), *E2_HARD_GATE_ARMS.values()}
        ]
        linked = base.merge(
            contrast_ledger.loc[:, [*_JOIN_KEY, *status_columns]],
            on=list(_JOIN_KEY),
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if len(linked) != len(base) or linked[status_columns].isna().any(axis=None):
            raise ValueError("E2 gate ledger does not cover the exact G3 child axis")
        contrast = str(contrast_name)
        frames.append(
            _arm_view(
                linked,
                arm=E2_BASE_ARM,
                contrast_name=contrast,
                stage=None,
                hard_gate=False,
            )
        )
        frames.extend(
            _arm_view(
                linked,
                arm=arm,
                contrast_name=contrast,
                stage=stage,
                hard_gate=False,
            )
            for arm, stage in E2_ANNOTATION_ARMS.items()
        )
        frames.extend(
            _arm_view(
                linked,
                arm=arm,
                contrast_name=contrast,
                stage=stage,
                hard_gate=True,
            )
            for arm, stage in E2_HARD_GATE_ARMS.items()
        )
    result = pd.concat(frames, ignore_index=True).loc[:, list(SCORE_VIEW_COLUMNS)]
    result = result.sort_values(
        ["score_view", "contrast_scope", "event_id", "fold_id", "sample_id"],
        kind="stable",
        ignore_index=True,
    )
    if set(result["score_view"].astype(str)) != set(E2_ARMS):
        raise RuntimeError("E2 did not produce every preregistered component arm")
    if result.duplicated(
        ["generator_id", "score_view", "contrast_scope", "event_id", "sample_id"]
    ).any():
        raise ValueError("E2 score views contain duplicate sample-event rows")
    return result, ledger


@dataclass(frozen=True, slots=True)
class V7E2ComponentSwapResult:
    """Complete descriptive E2 result on one generated dataset."""

    gate_stage_ledger: pd.DataFrame
    score_views: pd.DataFrame
    effects: pd.DataFrame
    aligned_effects: pd.DataFrame
    metrics: pd.DataFrame

    def __post_init__(self) -> None:
        expected = (
            (self.gate_stage_ledger, GATE_STAGE_LEDGER_COLUMNS, "gate_stage_ledger"),
            (self.score_views, SCORE_VIEW_COLUMNS, "score_views"),
            (self.effects, EFFECT_COLUMNS, "effects"),
            (self.aligned_effects, ALIGNED_EFFECT_COLUMNS, "aligned_effects"),
            (self.metrics, METRIC_COLUMNS, "metrics"),
        )
        for table, columns, name in expected:
            if not isinstance(table, pd.DataFrame) or tuple(table.columns) != columns:
                raise ValueError(f"E2 {name} does not match its frozen contract")
            object.__setattr__(self, name, table.copy(deep=True))


def run_v7_e2_component_swap(
    crossfit: CrossFitArtifacts,
    *,
    dataset_id: str,
    design: DifferentialDesignSpec,
    sample_metadata: pd.DataFrame,
    truth: pd.DataFrame,
    dgp_family: str,
    design_kind: str,
    fit_cache: V7InferenceFitCache | None = None,
) -> V7E2ComponentSwapResult:
    """Run all E2 arms through the same frozen I1 and truth evaluator."""

    score_views, ledger = build_e2_component_swap_score_views(
        crossfit,
        dataset_id=dataset_id,
    )
    effects = run_v7_inference_matrix(
        score_views,
        design=design,
        sample_metadata=sample_metadata,
        arms=((E2_GENERATOR_ID, "I1"),),
        fit_cache=fit_cache,
    )
    aligned, metrics = evaluate_v7_integrated_matrix(
        score_views,
        effects,
        truth,
        dgp_family=_name(dgp_family, field="dgp_family"),
        design_kind=_name(design_kind, field="design_kind"),
    )
    return V7E2ComponentSwapResult(
        gate_stage_ledger=ledger,
        score_views=score_views,
        effects=effects,
        aligned_effects=aligned,
        metrics=metrics,
    )


__all__ = [
    "E2_ANNOTATION_ARMS",
    "E2_ARMS",
    "E2_BASE_ARM",
    "E2_GENERATOR_ID",
    "E2_HARD_GATE_ARMS",
    "SCHEMA_VERSION",
    "V7E2ComponentSwapResult",
    "build_e2_component_swap_score_views",
    "run_v7_e2_component_swap",
]
