"""Integrated G0--G5 score views and diagnostic I0--I2 inference.

This module is benchmark-only.  It projects one already-fitted subject
cross-fit into the frozen suggest-next2 generator matrix without changing the
runtime estimator.  Analytic p-values remain diagnostic; formal fields are
always withheld until a complete-pipeline subject resampling campaign exists.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import digamma, polygamma
from scipy.stats import linregress
from scipy.stats import t as t_distribution

from crychic.core import stable_id
from crychic.inference import (
    DifferentialContrastSpec,
    DifferentialDesignKind,
    DifferentialDesignSpec,
    fit_design_aware_differential,
)
from crychic.workflow import CrossFitArtifacts
from crychic.workflow.crossfit import _V7PrimaryCrossFitArtifacts

SCHEMA_VERSION = "crychic-suggest-next2-v7-integrated-matrix-v1"
ALL_CONTRASTS = "__all__"
PARENT_SENDER = "__parent__"
_G0_LOG2_REFERENCE_SCALE = math.log1p(1_000_000.0) / math.log(2.0)
_USABLE_SCORE_STATUSES = {"observed", "low_evidence", "structural_impossible"}

SCORE_VIEW_COLUMNS = (
    "schema_version",
    "dataset_id",
    "generator_id",
    "score_view",
    "estimand",
    "resolution",
    "contrast_scope",
    "event_id",
    "sample_id",
    "subject_id",
    "condition",
    "context_id",
    "fold_id",
    "sender",
    "receiver",
    "interaction_id",
    "score",
    "score_status",
    "reliability_weight",
    "out_of_fold",
    "outcome_agnostic",
    "condition_gate_used",
    "primary_view",
    "source_version",
)

EFFECT_COLUMNS = (
    "schema_version",
    "dataset_id",
    "generator_id",
    "score_view",
    "estimand",
    "resolution",
    "inference_id",
    "contrast_scope",
    "event_id",
    "contrast_name",
    "sender",
    "receiver",
    "interaction_id",
    "effect",
    "standard_error",
    "statistic",
    "ranking_score",
    "diagnostic_p_value",
    "p_value",
    "q_value",
    "residual_df",
    "n_subjects",
    "observed_fraction",
    "covariance_method",
    "status",
    "reason_code",
    "formal_inference_allowed",
    "prior_df",
    "prior_variance",
)

_BIOLOGICAL_KEY = ("sender", "receiver", "interaction_id")
_V2_EDGE_KEY = (
    "sample_id",
    "subject_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
)
_V2_PARENT_KEY = (
    "sample_id",
    "subject_id",
    "condition",
    "context_id",
    "fold_id",
    "receiver",
    "interaction_id",
)


def _name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _event_id(sender: str, receiver: str, interaction_id: str) -> str:
    return stable_id(
        "v7_benchmark_biological_event",
        {
            "sender": sender,
            "receiver": receiver,
            "interaction_id": interaction_id,
        },
        schema_version="1",
    )


def _score_status(score: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(score, errors="coerce")
    return pd.Series(
        np.where(
            numeric.isna(),
            "not_estimable",
            np.where(numeric.eq(0.0), "low_evidence", "observed"),
        ),
        index=score.index,
        dtype=object,
    )


def _view_frame(
    source: pd.DataFrame,
    *,
    dataset_id: str,
    generator_id: str,
    score_view: str,
    estimand: str,
    resolution: str,
    score: pd.Series,
    status: pd.Series,
    contrast_scope: str = ALL_CONTRASTS,
    outcome_agnostic: bool,
    condition_gate_used: bool,
    primary_view: bool,
    source_version: str,
) -> pd.DataFrame:
    table = source.copy(deep=True)
    if "sender" not in table:
        table["sender"] = PARENT_SENDER
    table["event_id"] = [
        _event_id(str(sender), str(receiver), str(interaction_id))
        for sender, receiver, interaction_id in table.loc[
            :, list(_BIOLOGICAL_KEY)
        ].itertuples(index=False, name=None)
    ]
    table["schema_version"] = SCHEMA_VERSION
    table["dataset_id"] = dataset_id
    table["generator_id"] = generator_id
    table["score_view"] = score_view
    table["estimand"] = estimand
    table["resolution"] = resolution
    table["contrast_scope"] = contrast_scope
    table["score"] = pd.to_numeric(score, errors="coerce").astype(float)
    table["score_status"] = status.astype(str)
    table["outcome_agnostic"] = outcome_agnostic
    table["condition_gate_used"] = condition_gate_used
    table["primary_view"] = primary_view
    table["source_version"] = source_version
    if "reliability_weight" not in table:
        table["reliability_weight"] = 1.0
    if "out_of_fold" not in table:
        table["out_of_fold"] = True
    result = table.loc[:, list(SCORE_VIEW_COLUMNS)].copy()
    if (
        not result["score_status"]
        .isin({*_USABLE_SCORE_STATUSES, "not_estimable"})
        .all()
    ):
        raise ValueError(f"{generator_id}/{score_view} has invalid score statuses")
    if result.duplicated(["contrast_scope", "event_id", "sample_id"]).any():
        raise ValueError(f"{generator_id}/{score_view} has duplicate sample-event rows")
    return result


def _parent_source(
    v2: pd.DataFrame,
    value_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    grouped = v2.groupby(list(_V2_PARENT_KEY), observed=True, sort=False)
    consistency = grouped[value_column].nunique(dropna=False)
    if consistency.gt(1).any():
        raise ValueError(f"{value_column} must be sender-invariant per parent")
    source = v2.drop_duplicates(list(_V2_PARENT_KEY)).copy()
    source["sender"] = PARENT_SENDER
    score = pd.to_numeric(source[value_column], errors="coerce").astype(float)
    return source, score


def _parent_activity_status(v2: pd.DataFrame, source: pd.DataFrame) -> pd.Series:
    grouped = v2.groupby(list(_V2_PARENT_KEY), observed=True, sort=False)["status"]
    summary = grouped.agg(
        any_observed=lambda values: values.eq("observed").any(),
        any_low=lambda values: values.eq("low_evidence").any(),
        any_structural=lambda values: values.eq("structural_impossible").any(),
    ).reset_index()
    aligned = source.loc[:, list(_V2_PARENT_KEY)].merge(
        summary,
        on=list(_V2_PARENT_KEY),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    score_present = source["parent_mean_raw"].notna().to_numpy()
    return pd.Series(
        np.select(
            [
                ~score_present,
                aligned["any_observed"].to_numpy(dtype=bool),
                aligned["any_low"].to_numpy(dtype=bool),
                aligned["any_structural"].to_numpy(dtype=bool),
            ],
            ["not_estimable", "observed", "low_evidence", "structural_impossible"],
            default="not_estimable",
        ),
        index=source.index,
        dtype=object,
    )


def _head_status(
    source: pd.DataFrame,
    *,
    score: pd.Series,
    status_column: str,
) -> pd.Series:
    mapping = {
        "observed": "observed",
        "partial": "low_evidence",
        "not_estimable": "not_estimable",
        "not_computed": "not_estimable",
    }
    result = source[status_column].map(mapping).fillna("not_estimable")
    result = result.where(score.notna(), "not_estimable")
    return result.astype(str)


def _v1_views(
    fold: object,
    v2: pd.DataFrame,
    *,
    dataset_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    application = fold.application
    availability = application.availability.sample_interactions
    columns = [
        *_V2_EDGE_KEY,
        "ligand_absolute_evidence",
        "receptor_absolute_evidence",
    ]
    missing = set(columns).difference(availability.columns)
    if missing:
        raise ValueError(f"availability lacks frozen G0/G2 fields: {sorted(missing)}")
    if availability.duplicated(list(_V2_EDGE_KEY)).any():
        raise ValueError("availability G0/G2 candidate keys must be unique")
    base_columns = [
        *_V2_EDGE_KEY,
        "condition",
        "fold_id",
        "reliability_weight",
        "out_of_fold",
    ]
    source = v2.loc[:, base_columns].merge(
        availability.loc[:, columns],
        on=list(_V2_EDGE_KEY),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    ligand = pd.to_numeric(source["ligand_absolute_evidence"], errors="coerce")
    receptor = pd.to_numeric(source["receptor_absolute_evidence"], errors="coerce")
    g2_score = 0.5 * (ligand + receptor)
    g0_score = _G0_LOG2_REFERENCE_SCALE * g2_score
    return (
        _view_frame(
            source,
            dataset_id=dataset_id,
            generator_id="G0",
            score_view="primary_sender_detection",
            estimand="sample_comparable_lr_intensity",
            resolution="sender_lr_receiver_child",
            score=g0_score,
            status=_score_status(g0_score),
            outcome_agnostic=True,
            condition_gate_used=False,
            primary_view=True,
            source_version="simple_log2_cpm_geometric_lr_v1",
        ),
        _view_frame(
            source,
            dataset_id=dataset_id,
            generator_id="G2",
            score_view="primary_sender_detection",
            estimand="bounded_log_reference_lr_intensity",
            resolution="sender_lr_receiver_child",
            score=g2_score,
            status=_score_status(g2_score),
            outcome_agnostic=True,
            condition_gate_used=False,
            primary_view=True,
            source_version="log_reference_additive_m0_v1",
        ),
    )


def _v2_views(v2: pd.DataFrame, *, dataset_id: str) -> list[pd.DataFrame]:
    child_source = v2.copy(deep=True)
    child_score = pd.to_numeric(
        child_source["sender_detection_raw"], errors="coerce"
    ).astype(float)
    frames = [
        _view_frame(
            child_source,
            dataset_id=dataset_id,
            generator_id="G3",
            score_view="primary_sender_detection",
            estimand="sample_comparable_lr_intensity",
            resolution="sender_lr_receiver_child",
            score=child_score,
            status=child_source["status"],
            outcome_agnostic=True,
            condition_gate_used=False,
            primary_view=True,
            source_version=str(child_source["score_version"].iloc[0]),
        )
    ]
    parent_source, parent_score = _parent_source(v2, "parent_mean_raw")
    parent_status = _parent_activity_status(v2, parent_source)
    frames.append(
        _view_frame(
            parent_source,
            dataset_id=dataset_id,
            generator_id="G3",
            score_view="parent_mean",
            estimand="sample_comparable_parent_lr_intensity",
            resolution="lr_receiver_parent",
            score=parent_score,
            status=parent_status,
            outcome_agnostic=True,
            condition_gate_used=False,
            primary_view=False,
            source_version=str(parent_source["score_version"].iloc[0]),
        )
    )

    program_source, program_score = _parent_source(v2, "program_signed")
    program_status = _head_status(
        program_source,
        score=program_score,
        status_column="program_status",
    )
    for generator_id, view_name in (
        ("G4", "program_signed_component"),
        ("G5", "program_signed_annotation"),
    ):
        frames.append(
            _view_frame(
                program_source,
                dataset_id=dataset_id,
                generator_id=generator_id,
                score_view=view_name,
                estimand="signed_receiver_program",
                resolution="lr_receiver_parent",
                score=program_score,
                status=program_status,
                outcome_agnostic=True,
                condition_gate_used=False,
                primary_view=False,
                source_version="signed_program_v2",
            )
        )
    frames.append(
        _view_frame(
            parent_source,
            dataset_id=dataset_id,
            generator_id="G4",
            score_view="m0_parent_mean_component",
            estimand="sample_comparable_parent_lr_intensity",
            resolution="lr_receiver_parent",
            score=parent_score,
            status=parent_status,
            outcome_agnostic=True,
            condition_gate_used=False,
            primary_view=False,
            source_version=str(parent_source["score_version"].iloc[0]),
        )
    )
    frames.append(
        _view_frame(
            child_source,
            dataset_id=dataset_id,
            generator_id="G5",
            score_view="primary_sender_detection",
            estimand="sample_comparable_lr_intensity",
            resolution="sender_lr_receiver_child",
            score=child_score,
            status=child_source["status"],
            outcome_agnostic=True,
            condition_gate_used=False,
            primary_view=True,
            source_version=str(child_source["score_version"].iloc[0]),
        )
    )
    for score_column, status_column, view_name, estimand, source_version in (
        (
            "sender_attribution",
            "attribution_status",
            "sender_attribution_with_m2",
            "conditional_sender_attribution",
            "null_sender_attribution_v2_plus_eb_coupling_v2",
        ),
        (
            "coupling_prior",
            "coupling_status",
            "coupling_prior_annotation",
            "training_fold_sender_identity_prior",
            "eb_shrunken_coupling_v2",
        ),
    ):
        score = pd.to_numeric(child_source[score_column], errors="coerce").astype(float)
        frames.append(
            _view_frame(
                child_source,
                dataset_id=dataset_id,
                generator_id="G5",
                score_view=view_name,
                estimand=estimand,
                resolution="sender_lr_receiver_child",
                score=score,
                status=_head_status(
                    child_source,
                    score=score,
                    status_column=status_column,
                ),
                outcome_agnostic=True,
                condition_gate_used=False,
                primary_view=False,
                source_version=source_version,
            )
        )
    return frames


def _legacy_views(
    fold: object,
    v2: pd.DataFrame,
    *,
    dataset_id: str,
) -> list[pd.DataFrame]:
    metadata = v2.loc[
        :,
        [
            "sample_id",
            "subject_id",
            "condition",
            "context_id",
            "fold_id",
            "reliability_weight",
            "out_of_fold",
        ],
    ].drop_duplicates("sample_id")
    frames: list[pd.DataFrame] = []
    for application in fold.family_common_applications:
        sender_scores = application.sender_scores
        sender_scores = sender_scores.loc[sender_scores["mode"].eq("state")].copy()
        if sender_scores.empty:
            continue
        source = sender_scores.merge(
            metadata,
            on=["sample_id", "subject_id", "context_id"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        raw_status = source["status"].astype(str)
        status = raw_status.map(
            {
                "ok": "observed",
                "structural_zero": "low_evidence",
                "not_estimable": "not_estimable",
            }
        ).fillna("not_estimable")
        score = pd.to_numeric(source["sender_resolved_strength"], errors="coerce")
        status = status.where(score.notna(), "not_estimable")
        frames.append(
            _view_frame(
                source,
                dataset_id=dataset_id,
                generator_id="G1",
                score_view="primary_sender_resolved_state",
                estimand="legacy_condition_gated_family_common_strength",
                resolution="sender_lr_receiver_child",
                score=score,
                status=status,
                contrast_scope=str(application.functional.contrast_name),
                outcome_agnostic=False,
                condition_gate_used=True,
                primary_view=True,
                source_version=str(source["score_version"].iloc[0]),
            )
        )
    return frames


def build_v7_score_views(
    crossfit: CrossFitArtifacts,
    *,
    dataset_id: str,
) -> pd.DataFrame:
    """Build every frozen G0--G5 score view from one exact OOF cross-fit."""

    if not isinstance(crossfit, CrossFitArtifacts):
        raise TypeError("crossfit must be CrossFitArtifacts")
    dataset_id = _name(dataset_id, field="dataset_id")
    crossfit._require_intact()
    frames: list[pd.DataFrame] = []
    for fold in crossfit.folds:
        artifact = fold.sample_edge_scores_v2
        if artifact is None:
            raise ValueError("integrated v7 matrix requires M0 v2 in every fold")
        v2 = artifact.table
        g0, g2 = _v1_views(fold, v2, dataset_id=dataset_id)
        frames.extend((g0, g2))
        frames.extend(_v2_views(v2, dataset_id=dataset_id))
        frames.extend(_legacy_views(fold, v2, dataset_id=dataset_id))
    if not frames:
        raise ValueError("integrated v7 matrix produced no score views")
    result = pd.concat(frames, ignore_index=True).loc[:, list(SCORE_VIEW_COLUMNS)]
    result = result.sort_values(
        [
            "generator_id",
            "score_view",
            "contrast_scope",
            "event_id",
            "fold_id",
            "sample_id",
        ],
        kind="stable",
        ignore_index=True,
    )
    duplicates = result.duplicated(
        [
            "generator_id",
            "score_view",
            "contrast_scope",
            "event_id",
            "sample_id",
        ]
    )
    if duplicates.any():
        duplicate = result.loc[
            duplicates,
            ["generator_id", "score_view", "contrast_scope", "event_id", "sample_id"],
        ].iloc[0]
        raise ValueError(f"integrated score-view duplicate: {duplicate.to_dict()}")
    if not result["out_of_fold"].astype(bool).all():
        raise ValueError("integrated v7 score views must all be out of fold")
    return result


def build_v7_primary_score_views(
    crossfit: _V7PrimaryCrossFitArtifacts,
    *,
    dataset_id: str,
) -> pd.DataFrame:
    """Build G0/G2--G5 views from the lean PR10 execution profile.

    G1 is intentionally unavailable because its legacy family-common stages
    are omitted by the v7-primary profile.  This benchmark-only projection
    does not alter or relabel any v7 score.
    """

    if not isinstance(crossfit, _V7PrimaryCrossFitArtifacts):
        raise TypeError("crossfit must be v7-primary cross-fit artifacts")
    dataset_id = _name(dataset_id, field="dataset_id")
    crossfit._require_intact()
    frames: list[pd.DataFrame] = []
    for fold in crossfit.folds:
        artifact = fold.sample_edge_scores_v2
        if artifact is None:
            raise ValueError("v7-primary score views require M0 v2 in every fold")
        v2 = artifact.table
        g0, g2 = _v1_views(fold, v2, dataset_id=dataset_id)
        frames.extend((g0, g2))
        frames.extend(_v2_views(v2, dataset_id=dataset_id))
    if not frames:
        raise ValueError("v7-primary projection produced no score views")
    result = pd.concat(frames, ignore_index=True).loc[:, list(SCORE_VIEW_COLUMNS)]
    result = result.sort_values(
        [
            "generator_id",
            "score_view",
            "contrast_scope",
            "event_id",
            "fold_id",
            "sample_id",
        ],
        kind="stable",
        ignore_index=True,
    )
    duplicate_keys = [
        "generator_id",
        "score_view",
        "contrast_scope",
        "event_id",
        "sample_id",
    ]
    if result.duplicated(duplicate_keys).any():
        duplicate = result.loc[result.duplicated(duplicate_keys), duplicate_keys].iloc[
            0
        ]
        raise ValueError(f"v7-primary score-view duplicate: {duplicate.to_dict()}")
    if not result["out_of_fold"].astype(bool).all():
        raise ValueError("v7-primary score views must all be out of fold")
    if set(result["generator_id"].astype(str)) != {"G0", "G2", "G3", "G4", "G5"}:
        raise ValueError("v7-primary score views have an unexpected generator axis")
    return result


def build_legacy_g1_score_view_from_components(
    components: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    dataset_id: str,
    condition_column: str,
    contrast_name: str,
    expected_provenance: Mapping[str, object],
) -> pd.DataFrame:
    """Project a checksum-bound persisted family-common G1 sample score view."""

    dataset_id = _name(dataset_id, field="dataset_id")
    condition_column = _name(condition_column, field="condition_column")
    contrast_name = _name(contrast_name, field="contrast_name")
    required = {
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "fold_id",
        "contrast",
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
        "mode",
        "component",
        "component_value",
        "status",
        "score_version",
        "certification_status",
        "is_oof_certified",
        "formal_inference_status",
        "claim_scope",
        "source_table",
    }
    missing = required.difference(components.columns)
    if missing or components.empty:
        raise ValueError(
            f"legacy G1 components are empty or missing: {sorted(missing)}"
        )
    source = components.copy(deep=True)
    fixed_values: Mapping[str, object] = {
        "crossfit_id": expected_provenance.get("crossfit_id"),
        "spec_id": expected_provenance.get("spec_id"),
        "repeat_id": expected_provenance.get("repeat_id"),
        "contrast": contrast_name,
        "mode": "state",
        "component": "sender_resolved_strength",
        "score_version": expected_provenance.get("score_version"),
        "certification_status": expected_provenance.get("certification_status"),
        "is_oof_certified": expected_provenance.get("is_oof_certified"),
        "formal_inference_status": expected_provenance.get("formal_inference_status"),
        "claim_scope": expected_provenance.get("claim_scope"),
        "source_table": "sender_scores",
    }
    for column, expected in fixed_values.items():
        observed = set(source[column].drop_duplicates().tolist())
        if observed != {expected}:
            raise ValueError(
                f"legacy G1 {column} differs from its frozen provenance: "
                f"{sorted(map(str, observed))}"
            )
    expected_rows = expected_provenance.get("rows")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows < 1
        or len(source) != expected_rows
    ):
        raise ValueError(
            f"legacy G1 row count differs from its frozen provenance: {len(source)}"
        )
    identifiers = (
        "fold_id",
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
    )
    for column in identifiers:
        values = source[column].astype(str)
        if (
            source[column].isna().any()
            or values.eq("").any()
            or values.str.strip().ne(values).any()
        ):
            raise ValueError(f"legacy G1 {column} contains non-canonical values")
        source[column] = values
    duplicate_keys = ["sample_id", "sender", "receiver", "interaction_id"]
    if source.duplicated(duplicate_keys).any():
        raise ValueError("legacy G1 components duplicate sample-event rows")
    sample_folds = source.loc[:, ["sample_id", "fold_id"]].drop_duplicates()
    subject_folds = source.loc[:, ["subject_id", "fold_id"]].drop_duplicates()
    if (
        sample_folds["sample_id"].duplicated().any()
        or subject_folds["subject_id"].duplicated().any()
    ):
        raise ValueError(
            "legacy G1 samples or subjects occur in multiple held-out folds"
        )
    metadata_required = ("sample_id", "subject_id", condition_column)
    metadata_missing = set(metadata_required).difference(sample_metadata.columns)
    if metadata_missing or sample_metadata.empty:
        raise ValueError(
            f"legacy G1 sample metadata are empty or missing: "
            f"{sorted(metadata_missing)}"
        )
    design = sample_metadata.loc[:, list(metadata_required)].drop_duplicates()
    if design["sample_id"].duplicated().any():
        raise ValueError("legacy G1 metadata require unique sample IDs")
    if set(source["sample_id"]) != set(design["sample_id"].astype(str)):
        raise ValueError("legacy G1 component and metadata sample axes differ")
    design = design.rename(
        columns={
            "subject_id": "_metadata_subject_id",
            condition_column: "condition",
        }
    )
    source = source.merge(
        design,
        on="sample_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if not source["subject_id"].eq(source["_metadata_subject_id"].astype(str)).all():
        raise ValueError("legacy G1 component and metadata subject axes differ")
    source = source.drop(columns="_metadata_subject_id")
    status = source["status"].map(
        {
            "observed": "observed",
            "structural_zero": "low_evidence",
            "not_estimable": "not_estimable",
        }
    )
    if status.isna().any():
        raise ValueError("legacy G1 component status is unsupported")
    score = pd.to_numeric(source["component_value"], errors="coerce")
    usable = status.isin(_USABLE_SCORE_STATUSES)
    if (
        not np.isfinite(score.loc[usable]).all()
        or score.loc[status.eq("not_estimable")].notna().any()
    ):
        raise ValueError("legacy G1 component values disagree with their statuses")
    return _view_frame(
        source,
        dataset_id=dataset_id,
        generator_id="G1",
        score_view="primary_sender_resolved_state",
        estimand="legacy_condition_gated_family_common_strength",
        resolution="sender_lr_receiver_child",
        score=score,
        status=status,
        contrast_scope=contrast_name,
        outcome_agnostic=False,
        condition_gate_used=True,
        primary_view=True,
        source_version=str(expected_provenance["score_version"]),
    ).sort_values(
        ["event_id", "fold_id", "sample_id"],
        kind="stable",
        ignore_index=True,
    )


def _design_for_scope(
    design: DifferentialDesignSpec,
    contrast_scope: str,
) -> DifferentialDesignSpec:
    if contrast_scope == ALL_CONTRASTS or (
        design.design_kind is DifferentialDesignKind.CONTINUOUS
    ):
        return design
    matches = tuple(item for item in design.contrasts if item.name == contrast_scope)
    if len(matches) != 1:
        raise ValueError(f"unknown contrast scope {contrast_scope!r}")
    contrast = matches[0]
    levels = tuple(level for level, weight in contrast.weights if weight != 0.0)
    kind = design.design_kind
    if kind is DifferentialDesignKind.INDEPENDENT_MULTI_GROUP:
        kind = DifferentialDesignKind.INDEPENDENT_TWO_GROUP
    return DifferentialDesignSpec(
        design_kind=kind,
        condition_column=design.condition_column,
        condition_levels=levels,
        subject_column=design.subject_column,
        sample_column=design.sample_column,
        batch_columns=design.batch_columns,
        continuous_covariates=design.continuous_covariates,
        categorical_covariates=design.categorical_covariates,
        cohort_column=design.cohort_column,
        contrasts=(contrast,),
        precision_weight_column=design.precision_weight_column,
        minimum_subjects_per_level=design.minimum_subjects_per_level,
        minimum_clusters_for_cr2=design.minimum_clusters_for_cr2,
        maximum_condition_number=design.maximum_condition_number,
    )


def _merge_design_metadata(
    table: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    design: DifferentialDesignSpec,
) -> pd.DataFrame:
    if not isinstance(sample_metadata, pd.DataFrame):
        raise TypeError("sample_metadata must be a pandas DataFrame")
    if (
        "sample_id" not in sample_metadata
        or sample_metadata["sample_id"].duplicated().any()
    ):
        raise ValueError("sample_metadata requires unique sample_id values")
    required = {
        design.sample_column,
        design.subject_column,
        design.condition_column,
        *design.batch_columns,
        *design.continuous_covariates,
        *design.categorical_covariates,
    }
    if design.cohort_column is not None:
        required.add(design.cohort_column)
    if design.precision_weight_column is not None:
        required.add(design.precision_weight_column)
    missing_metadata = required.difference(table.columns).difference(
        sample_metadata.columns
    )
    if missing_metadata:
        raise ValueError(
            f"sample_metadata lacks design fields: {sorted(missing_metadata)}"
        )
    missing = sorted(required.difference(table.columns))
    result = table.copy(deep=True)
    if missing:
        result = result.merge(
            sample_metadata.loc[:, ["sample_id", *missing]],
            on="sample_id",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    if result.loc[:, sorted(required)].isna().any(axis=None):
        raise ValueError("merged score views contain missing design metadata")
    return result


def _event_metadata(table: pd.DataFrame) -> pd.DataFrame:
    columns = ["event_id", "sender", "receiver", "interaction_id"]
    metadata = table.loc[:, columns].drop_duplicates()
    if metadata["event_id"].duplicated().any():
        raise ValueError("event IDs do not identify one biological event")
    return metadata


def _effect_frame(
    source: pd.DataFrame,
    *,
    group: Mapping[str, object],
    inference_id: str,
    event_metadata: pd.DataFrame,
) -> pd.DataFrame:
    renamed = source.rename(columns={"statistic": "_source_statistic"}).copy()
    renamed = renamed.merge(
        event_metadata,
        on="event_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    statistic = pd.to_numeric(renamed["_source_statistic"], errors="coerce")
    renamed["statistic"] = statistic
    renamed["ranking_score"] = statistic.abs()
    renamed["schema_version"] = SCHEMA_VERSION
    renamed["dataset_id"] = str(group["dataset_id"])
    renamed["generator_id"] = str(group["generator_id"])
    renamed["score_view"] = str(group["score_view"])
    renamed["estimand"] = str(group["estimand"])
    renamed["resolution"] = str(group["resolution"])
    renamed["inference_id"] = inference_id
    renamed["contrast_scope"] = str(group["contrast_scope"])
    renamed["prior_df"] = np.nan
    renamed["prior_variance"] = np.nan
    for column in ("p_value", "q_value"):
        renamed[column] = np.nan
    renamed["formal_inference_allowed"] = False
    return renamed.loc[:, list(EFFECT_COLUMNS)]


def _subject_context_rows(
    event: pd.DataFrame,
    design: DifferentialDesignSpec,
) -> pd.DataFrame:
    numeric = pd.to_numeric(event["score"], errors="coerce")
    usable = event["score_status"].isin(_USABLE_SCORE_STATUSES) & numeric.notna()
    source = event.loc[usable].copy()
    source["score"] = numeric.loc[usable].astype(float)
    if source.empty:
        return source
    aggregation: dict[str, str] = {"score": "mean"}
    for column in design.continuous_covariates:
        aggregation[column] = "mean"
    for column in (
        *design.batch_columns,
        *design.categorical_covariates,
        *(() if design.cohort_column is None else (design.cohort_column,)),
    ):
        aggregation[column] = "first"
    return (
        source.groupby(
            [design.subject_column, design.condition_column],
            observed=True,
            sort=True,
        )
        .agg(aggregation)
        .reset_index()
    )


def _i0_row(
    *,
    event_id: str,
    contrast_name: str,
    effect: float | None,
    standard_error: float | None,
    residual_df: float | None,
    n_subjects: int,
    observed_fraction: float,
    reason_code: str | None,
    covariance_method: str,
) -> dict[str, object]:
    observed = (
        effect is not None
        and standard_error is not None
        and math.isfinite(effect)
        and math.isfinite(standard_error)
        and standard_error > 0.0
    )
    statistic = effect / standard_error if observed else None
    diagnostic_p = (
        float(2.0 * t_distribution.sf(abs(statistic), residual_df))
        if observed
        and residual_df is not None
        and math.isfinite(residual_df)
        and residual_df > 0.0
        else None
    )
    return {
        "event_id": event_id,
        "contrast_name": contrast_name,
        "effect": effect if observed else None,
        "standard_error": standard_error if observed else None,
        "statistic": statistic,
        "diagnostic_p_value": diagnostic_p,
        "residual_df": residual_df,
        "n_subjects": n_subjects,
        "observed_fraction": observed_fraction,
        "covariance_method": covariance_method if observed else "not_estimable",
        "status": "observed" if observed else "not_estimable",
        "reason_code": None if observed else reason_code or "raw_effect_not_estimable",
    }


def _welch_contrast(
    table: pd.DataFrame,
    design: DifferentialDesignSpec,
    contrast: DifferentialContrastSpec,
) -> tuple[float | None, float | None, float | None, int, str | None]:
    effect = 0.0
    variance = 0.0
    denominator = 0.0
    subjects: set[str] = set()
    for level, weight in contrast.weights:
        if weight == 0.0:
            continue
        group = table.loc[table[design.condition_column].astype(str).eq(level)]
        values = group["score"].to_numpy(dtype=float)
        subjects.update(group[design.subject_column].astype(str))
        if len(values) < design.minimum_subjects_per_level:
            return None, None, None, len(subjects), "insufficient_subjects_per_level"
        group_variance = float(np.var(values, ddof=1))
        component = weight * weight * group_variance / len(values)
        effect += weight * float(np.mean(values))
        variance += component
        if len(values) > 1:
            denominator += component * component / (len(values) - 1)
    if variance <= 0.0 or denominator <= 0.0:
        return None, None, None, len(subjects), "zero_or_undefined_welch_variance"
    return (
        effect,
        math.sqrt(variance),
        variance * variance / denominator,
        len(subjects),
        None,
    )


def _paired_contrast(
    table: pd.DataFrame,
    design: DifferentialDesignSpec,
    contrast: DifferentialContrastSpec,
) -> tuple[float | None, float | None, float | None, int, str | None]:
    levels = tuple(level for level, weight in contrast.weights if weight != 0.0)
    matrix = (
        table.pivot(
            index=design.subject_column,
            columns=design.condition_column,
            values="score",
        )
        .reindex(columns=levels)
        .dropna()
    )
    n_subjects = len(matrix)
    if n_subjects < design.minimum_subjects_per_level:
        return None, None, None, n_subjects, "insufficient_complete_pairs"
    weights = np.asarray([dict(contrast.weights)[level] for level in levels])
    differences = matrix.to_numpy(dtype=float) @ weights
    variance = float(np.var(differences, ddof=1))
    if variance <= 0.0:
        return None, None, None, n_subjects, "zero_paired_variance"
    return (
        float(np.mean(differences)),
        math.sqrt(variance / n_subjects),
        float(n_subjects - 1),
        n_subjects,
        None,
    )


def _fit_i0(table: pd.DataFrame, design: DifferentialDesignSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_id, event in table.groupby("event_id", observed=True, sort=True):
        subject_table = _subject_context_rows(event, design)
        observed_fraction = float(
            (
                event["score_status"].isin(_USABLE_SCORE_STATUSES)
                & pd.to_numeric(event["score"], errors="coerce").notna()
            ).mean()
        )
        if design.design_kind is DifferentialDesignKind.CONTINUOUS:
            n_subjects = int(subject_table[design.subject_column].nunique())
            if len(subject_table) < design.minimum_subjects_per_level:
                rows.append(
                    _i0_row(
                        event_id=str(event_id),
                        contrast_name=f"slope:{design.condition_column}",
                        effect=None,
                        standard_error=None,
                        residual_df=None,
                        n_subjects=n_subjects,
                        observed_fraction=observed_fraction,
                        reason_code="insufficient_subjects_for_raw_slope",
                        covariance_method="raw_ols_slope_diagnostic",
                    )
                )
                continue
            fitted = linregress(
                pd.to_numeric(subject_table[design.condition_column]),
                subject_table["score"],
            )
            rows.append(
                _i0_row(
                    event_id=str(event_id),
                    contrast_name=f"slope:{design.condition_column}",
                    effect=float(fitted.slope),
                    standard_error=float(fitted.stderr),
                    residual_df=float(len(subject_table) - 2),
                    n_subjects=n_subjects,
                    observed_fraction=observed_fraction,
                    reason_code=None,
                    covariance_method="raw_ols_slope_diagnostic",
                )
            )
            continue
        paired = design.design_kind in {
            DifferentialDesignKind.PAIRED,
            DifferentialDesignKind.REPEATED,
        }
        for contrast in design.contrasts:
            if paired:
                effect, se, df, n_subjects, reason = _paired_contrast(
                    subject_table, design, contrast
                )
                method = "raw_paired_difference_diagnostic"
            else:
                effect, se, df, n_subjects, reason = _welch_contrast(
                    subject_table, design, contrast
                )
                method = "raw_welch_contrast_diagnostic"
            rows.append(
                _i0_row(
                    event_id=str(event_id),
                    contrast_name=contrast.name,
                    effect=effect,
                    standard_error=se,
                    residual_df=df,
                    n_subjects=n_subjects,
                    observed_fraction=observed_fraction,
                    reason_code=reason,
                    covariance_method=method,
                )
            )
    return pd.DataFrame.from_records(rows)


def _moderated_prior(variances: np.ndarray, dfs: np.ndarray) -> tuple[float, float]:
    corrected = np.log(variances) - digamma(dfs / 2.0) + np.log(dfs / 2.0)
    extra = float(np.var(corrected, ddof=1) - np.mean(polygamma(1, dfs / 2.0)))
    if not math.isfinite(extra) or extra <= 1.0e-8:
        prior_df = 1_000_000.0
    else:

        def objective(value: float) -> float:
            return float(polygamma(1, value / 2.0) - extra)

        lower = 0.05
        upper = 1_000_000.0
        if objective(lower) * objective(upper) > 0.0:
            prior_df = 1_000_000.0
        else:
            prior_df = float(brentq(objective, lower, upper))
    prior_variance = float(np.exp(np.mean(corrected)))
    if not math.isfinite(prior_variance) or prior_variance <= 0.0:
        raise ValueError("empirical-Bayes prior variance is not finite and positive")
    return prior_df, prior_variance


def _moderate_i1(i1: pd.DataFrame) -> pd.DataFrame:
    result = i1.copy(deep=True)
    result["inference_id"] = "I2"
    result["covariance_method"] = (
        result["covariance_method"].astype(str)
        + "+empirical_bayes_moderated_effect_variance_v1"
    )
    for _, index in result.groupby(
        ["contrast_scope", "contrast_name"], observed=True, sort=False
    ).groups.items():
        local = result.loc[index]
        se = pd.to_numeric(local["standard_error"], errors="coerce")
        df = pd.to_numeric(local["residual_df"], errors="coerce")
        eligible = (
            local["status"].eq("observed")
            & se.gt(0.0)
            & np.isfinite(se)
            & df.gt(0.0)
            & np.isfinite(df)
        )
        eligible_index = local.index[eligible]
        if len(eligible_index) < 4:
            result.loc[index, "status"] = "not_estimable"
            result.loc[index, "reason_code"] = "fewer_than_four_moderation_events"
            result.loc[
                index,
                [
                    "standard_error",
                    "statistic",
                    "ranking_score",
                    "diagnostic_p_value",
                ],
            ] = np.nan
            continue
        variances = np.square(se.loc[eligible_index].to_numpy(dtype=float))
        dfs = df.loc[eligible_index].to_numpy(dtype=float)
        prior_df, prior_variance = _moderated_prior(variances, dfs)
        posterior = (prior_df * prior_variance + dfs * variances) / (prior_df + dfs)
        moderated_se = np.sqrt(posterior)
        effects = pd.to_numeric(
            result.loc[eligible_index, "effect"], errors="coerce"
        ).to_numpy(dtype=float)
        statistic = effects / moderated_se
        moderated_df = dfs + prior_df
        result.loc[eligible_index, "standard_error"] = moderated_se
        result.loc[eligible_index, "statistic"] = statistic
        result.loc[eligible_index, "ranking_score"] = np.abs(statistic)
        result.loc[eligible_index, "diagnostic_p_value"] = 2.0 * t_distribution.sf(
            np.abs(statistic), moderated_df
        )
        result.loc[eligible_index, "residual_df"] = moderated_df
        result.loc[eligible_index, "prior_df"] = prior_df
        result.loc[eligible_index, "prior_variance"] = prior_variance
        ineligible = local.index.difference(eligible_index)
        result.loc[ineligible, "status"] = "not_estimable"
        result.loc[ineligible, "reason_code"] = result.loc[
            ineligible, "reason_code"
        ].fillna("base_design_effect_not_moderatable")
        result.loc[
            ineligible,
            ["statistic", "ranking_score", "diagnostic_p_value"],
        ] = np.nan
    result["p_value"] = np.nan
    result["q_value"] = np.nan
    result["formal_inference_allowed"] = False
    return result.loc[:, list(EFFECT_COLUMNS)]


def _fit_one_view(
    table: pd.DataFrame,
    *,
    design: DifferentialDesignSpec,
    sample_metadata: pd.DataFrame,
    inference_id: str,
) -> pd.DataFrame:
    group = table.iloc[0].to_dict()
    scope = str(group["contrast_scope"])
    scoped_design = _design_for_scope(design, scope)
    merged = _merge_design_metadata(table, sample_metadata, scoped_design)
    if (
        scope != ALL_CONTRASTS
        and scoped_design.design_kind is not DifferentialDesignKind.CONTINUOUS
    ):
        merged = merged.loc[
            merged[scoped_design.condition_column]
            .astype(str)
            .isin(scoped_design.condition_levels)
        ].copy()
    event_metadata = _event_metadata(merged)
    score_input = merged.loc[
        :,
        [
            "event_id",
            "sample_id",
            "subject_id",
            "score",
            "score_status",
            "out_of_fold",
            *[
                column
                for column in (
                    scoped_design.condition_column,
                    *scoped_design.batch_columns,
                    *scoped_design.continuous_covariates,
                    *scoped_design.categorical_covariates,
                    *(
                        ()
                        if scoped_design.cohort_column is None
                        else (scoped_design.cohort_column,)
                    ),
                    *(
                        ()
                        if scoped_design.precision_weight_column is None
                        else (scoped_design.precision_weight_column,)
                    ),
                )
                if column not in {"sample_id", "subject_id"}
            ],
        ],
    ]
    score_input = score_input.loc[:, ~score_input.columns.duplicated()]
    if inference_id == "I0":
        raw = _fit_i0(score_input, scoped_design)
        return _effect_frame(
            raw,
            group=group,
            inference_id="I0",
            event_metadata=event_metadata,
        )
    fitted = fit_design_aware_differential(score_input, scoped_design)
    i1 = _effect_frame(
        fitted.effects,
        group=group,
        inference_id="I1",
        event_metadata=event_metadata,
    )
    return i1 if inference_id == "I1" else _moderate_i1(i1)


def _fit_cache_key(
    table: pd.DataFrame,
    *,
    design: DifferentialDesignSpec,
    base_inference_id: str,
) -> str:
    columns = [
        "contrast_scope",
        "event_id",
        "sample_id",
        "subject_id",
        "condition",
        "sender",
        "receiver",
        "interaction_id",
        "score",
        "score_status",
        "reliability_weight",
        "out_of_fold",
    ]
    ordered = table.loc[:, columns].sort_values(
        ["contrast_scope", "event_id", "sample_id"],
        kind="stable",
        ignore_index=True,
    )
    row_hashes = pd.util.hash_pandas_object(
        ordered,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(design.spec_id.encode("ascii"))
    digest.update(base_inference_id.encode("ascii"))
    digest.update(str(ordered.shape).encode("ascii"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _relabel_cached_effects(
    effects: pd.DataFrame,
    *,
    table: pd.DataFrame,
    inference_id: str,
) -> pd.DataFrame:
    group = table.iloc[0]
    result = effects.copy(deep=True)
    for column in (
        "dataset_id",
        "generator_id",
        "score_view",
        "estimand",
        "resolution",
        "contrast_scope",
    ):
        result[column] = str(group[column])
    result["inference_id"] = inference_id
    return result.loc[:, list(EFFECT_COLUMNS)]


@dataclass(slots=True)
class V7InferenceFitCache:
    """Dataset-local cache for exact design-aware score inputs."""

    _effects: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    requests: int = 0
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> pd.DataFrame | None:
        self.requests += 1
        cached = self._effects.get(key)
        if cached is None:
            return None
        self.hits += 1
        return cached

    def store(self, key: str, effects: pd.DataFrame) -> None:
        if key in self._effects:
            raise ValueError("v7 inference cache key is already populated")
        if tuple(effects.columns) != EFFECT_COLUMNS:
            raise ValueError("v7 inference cache effects violate the contract")
        self._effects[key] = effects.copy(deep=True)
        self.misses += 1

    def to_dict(self) -> dict[str, int | str]:
        return {
            "policy": "dataset_local_exact_score_axis_sha256_v1",
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self._effects),
        }


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric)]
    if len(finite) < 2:
        return pd.Series(np.nan, index=values.index, dtype=float)
    scale = float(np.std(finite, ddof=0))
    if not math.isfinite(scale) or scale <= 0.0:
        return pd.Series(0.0, index=values.index, dtype=float).where(
            numeric.notna(), np.nan
        )
    return (numeric - float(np.mean(finite))) / scale


def _g4_combined_effects(effects: pd.DataFrame) -> pd.DataFrame:
    intensity = effects.loc[
        effects["generator_id"].eq("G4")
        & effects["score_view"].eq("m0_parent_mean_component")
        & effects["inference_id"].eq("I1")
    ].copy()
    program = effects.loc[
        effects["generator_id"].eq("G4")
        & effects["score_view"].eq("program_signed_component")
        & effects["inference_id"].eq("I1")
    ].copy()
    if intensity.empty or program.empty:
        return pd.DataFrame(columns=EFFECT_COLUMNS)
    keys = [
        "dataset_id",
        "contrast_scope",
        "event_id",
        "contrast_name",
        "sender",
        "receiver",
        "interaction_id",
    ]
    merged = intensity.merge(
        program.loc[:, [*keys, "effect", "status"]],
        on=keys,
        suffixes=("_m0", "_program"),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    combined_parts: list[pd.DataFrame] = []
    for _, local in merged.groupby("contrast_name", observed=True, sort=False):
        local = local.copy()
        observed = local["status_m0"].eq("observed") & local["status_program"].eq(
            "observed"
        )
        local["effect"] = _zscore(local["effect_m0"]) + 0.25 * _zscore(
            local["effect_program"]
        )
        local["effect"] = local["effect"].where(observed)
        local["status"] = np.where(observed, "observed", "not_estimable")
        local["reason_code"] = np.where(
            observed,
            "ranking_only_no_component_covariance",
            "m0_or_program_component_not_estimable",
        )
        combined_parts.append(local)
    result = pd.concat(combined_parts, ignore_index=True)
    result["schema_version"] = SCHEMA_VERSION
    result["generator_id"] = "G4"
    result["score_view"] = "primary_fixed_effect_z_blend"
    result["estimand"] = "fixed_m0_plus_signed_program_ranking"
    result["resolution"] = "lr_receiver_parent"
    result["inference_id"] = "I1"
    result["standard_error"] = np.nan
    result["statistic"] = result["effect"]
    result["ranking_score"] = result["effect"].abs()
    result["diagnostic_p_value"] = np.nan
    result["p_value"] = np.nan
    result["q_value"] = np.nan
    result["residual_df"] = np.nan
    result["covariance_method"] = "post_effect_fixed_z_blend"
    result["formal_inference_allowed"] = False
    result["prior_df"] = np.nan
    result["prior_variance"] = np.nan
    return result.loc[:, list(EFFECT_COLUMNS)]


def run_v7_inference_matrix(
    score_views: pd.DataFrame,
    *,
    design: DifferentialDesignSpec,
    sample_metadata: pd.DataFrame,
    arms: Iterable[tuple[str, str]],
    fit_cache: V7InferenceFitCache | None = None,
) -> pd.DataFrame:
    """Run requested paired generator/inference arms on frozen score views."""

    if tuple(score_views.columns) != SCORE_VIEW_COLUMNS:
        raise ValueError("score_views does not match the integrated v7 contract")
    if not isinstance(design, DifferentialDesignSpec):
        raise TypeError("design must be DifferentialDesignSpec")
    requested = tuple(sorted(set(arms)))
    if not requested:
        raise ValueError("arms cannot be empty")
    if any(
        generator not in {f"G{index}" for index in range(6)}
        for generator, _ in requested
    ):
        raise ValueError("arms contain an unknown generator")
    if any(inference not in {"I0", "I1", "I2"} for _, inference in requested):
        raise ValueError("arms contain an unknown inference")
    frames: list[pd.DataFrame] = []
    cache = fit_cache if fit_cache is not None else V7InferenceFitCache()
    if not isinstance(cache, V7InferenceFitCache):
        raise TypeError("fit_cache must be V7InferenceFitCache or None")
    grouping = [
        "dataset_id",
        "generator_id",
        "score_view",
        "estimand",
        "resolution",
        "contrast_scope",
    ]
    for generator_id, inference_id in requested:
        selected = score_views.loc[score_views["generator_id"].eq(generator_id)]
        if selected.empty:
            raise ValueError(f"generator {generator_id} has no score views")
        for _, view in selected.groupby(grouping, observed=True, sort=True):
            base_inference_id = "I0" if inference_id == "I0" else "I1"
            cache_key = _fit_cache_key(
                view,
                design=design,
                base_inference_id=base_inference_id,
            )
            base = cache.get(cache_key)
            if base is None:
                base = _fit_one_view(
                    view,
                    design=design,
                    sample_metadata=sample_metadata,
                    inference_id=base_inference_id,
                )
                cache.store(cache_key, base)
            else:
                base = _relabel_cached_effects(
                    base,
                    table=view,
                    inference_id=base_inference_id,
                )
            frames.append(base if inference_id != "I2" else _moderate_i1(base))
    result = pd.concat(frames, ignore_index=True).loc[:, list(EFFECT_COLUMNS)]
    combined = _g4_combined_effects(result)
    if not combined.empty:
        result = pd.concat([result, combined], ignore_index=True)
    return result.sort_values(
        [
            "generator_id",
            "inference_id",
            "score_view",
            "contrast_scope",
            "contrast_name",
            "event_id",
        ],
        kind="stable",
        ignore_index=True,
    )


def g0_g2_equivalence_diagnostic(score_views: pd.DataFrame) -> dict[str, object]:
    """Quantify the frozen G0/G2 constant-scale equivalence without hiding it."""

    key = ["event_id", "sample_id"]
    g0 = score_views.loc[
        score_views["generator_id"].eq("G0")
        & score_views["score_view"].eq("primary_sender_detection"),
        [*key, "score"],
    ].rename(columns={"score": "g0"})
    g2 = score_views.loc[
        score_views["generator_id"].eq("G2")
        & score_views["score_view"].eq("primary_sender_detection"),
        [*key, "score"],
    ].rename(columns={"score": "g2"})
    merged = g0.merge(g2, on=key, how="inner", validate="one_to_one")
    finite = merged[["g0", "g2"]].notna().all(axis=1)
    residual = merged.loc[finite, "g0"] - (
        _G0_LOG2_REFERENCE_SCALE * merged.loc[finite, "g2"]
    )
    maximum = float(residual.abs().max()) if not residual.empty else math.nan
    return {
        "rows_g0": len(g0),
        "rows_g2": len(g2),
        "paired_finite_rows": int(finite.sum()),
        "expected_scale": _G0_LOG2_REFERENCE_SCALE,
        "maximum_absolute_scale_residual": maximum,
        "equivalent_within_1e_12": bool(math.isfinite(maximum) and maximum <= 1.0e-12),
    }


@dataclass(frozen=True, slots=True)
class V7IntegratedMatrixResult:
    """In-memory score and effect matrix for one generated dataset."""

    score_views: pd.DataFrame
    effects: pd.DataFrame
    equivalence_diagnostic: Mapping[str, object]


def run_v7_integrated_matrix(
    crossfit: CrossFitArtifacts,
    *,
    dataset_id: str,
    design: DifferentialDesignSpec,
    sample_metadata: pd.DataFrame,
    arms: Sequence[tuple[str, str]],
    fit_cache: V7InferenceFitCache | None = None,
) -> V7IntegratedMatrixResult:
    """Build score views once and run the requested frozen method arms."""

    score_views = build_v7_score_views(crossfit, dataset_id=dataset_id)
    effects = run_v7_inference_matrix(
        score_views,
        design=design,
        sample_metadata=sample_metadata,
        arms=arms,
        fit_cache=fit_cache,
    )
    return V7IntegratedMatrixResult(
        score_views=score_views,
        effects=effects,
        equivalence_diagnostic=g0_g2_equivalence_diagnostic(score_views),
    )


__all__ = [
    "ALL_CONTRASTS",
    "EFFECT_COLUMNS",
    "PARENT_SENDER",
    "SCHEMA_VERSION",
    "SCORE_VIEW_COLUMNS",
    "V7InferenceFitCache",
    "V7IntegratedMatrixResult",
    "build_legacy_g1_score_view_from_components",
    "build_v7_primary_score_views",
    "build_v7_score_views",
    "g0_g2_equivalence_diagnostic",
    "run_v7_inference_matrix",
    "run_v7_integrated_matrix",
]
