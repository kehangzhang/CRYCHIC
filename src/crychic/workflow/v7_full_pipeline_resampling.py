"""Full-refit bootstrap, condition permutation, and LOSO for the v7 estimator."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from typing import TypeAlias, cast

import numpy as np
import pandas as pd
from anndata import AnnData

from crychic.core import (
    ContractError,
    CrychicConfig,
    CrychicError,
    SeedLineage,
    canonical_digest,
    canonical_json,
    stable_id,
)
from crychic.data import validate_anndata
from crychic.inference import DifferentialDesignKind
from crychic.resampling import (
    BootstrapSubjectDraw,
    ContextPermutationOperation,
    ContextPermutationPlan,
    ExchangeabilityMap,
    SubjectBootstrapPlan,
    build_exchangeability_map,
    plan_context_permutations,
)
from crychic.resources import ResourceBundle, TargetPrior
from crychic.scoring import FrozenHypergraphPrior

from .crossfit import CrossFitArtifacts, CrossFitSpec, _run_subject_crossfit
from .full_pipeline_resampling import (
    _materialize_context_permutation,
    _materialize_subject_bootstrap,
    _MaterializedResample,
    _validated_materialized,
)
from .receiver_universe import FrozenReceiverUniverse
from .training import (
    _input_schema,
    _resource_bundle_content_id,
    _sanitized_raw_input_snapshot,
    _target_prior_content_id,
)
from .v7_estimator import (
    CrossFitV7EstimatorResult,
    V7EstimatorSpec,
    _frame_digest,
    fit_crossfit_v7_estimator,
)

V7_FULL_PIPELINE_RESAMPLING_VERSION = "v7_full_refit_bootstrap_permutation_loso_v2"
_SCHEMA_VERSION = "2.0.0"
_LOSO_POLICY = "remove_complete_subject_block_then_refit_v1"
_BOOTSTRAP_POLICY = "design_stratified_complete_subject_block_with_replacement_v1"
_RERUN_STAGES = (
    "fold_split",
    "availability_fitting",
    "absolute_activity_v2_fitting",
    "signed_program_v2_fitting",
    "sender_attribution_v2_fitting",
    "eb_shrunken_coupling_v2_fitting",
    "heldout_score_application",
    "design_aware_effect_fitting",
    "two_part_occurrence_v2_fitting",
    "hypergraph_shrinkage_v2_fitting",
)


class V7FullPipelineOperation(StrEnum):
    """Subject-level operations supported by the v7 full-refit executor."""

    SUBJECT_BOOTSTRAP = "subject_bootstrap"
    CONDITION_PERMUTATION = "condition_permutation"
    LEAVE_ONE_SUBJECT_OUT = "leave_one_subject_out"


class V7FullPipelineResampleStatus(StrEnum):
    """Availability of one complete v7 estimator rerun."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class V7LeaveOneSubjectOutPlan:
    """One deterministic complete-subject deletion plan."""

    resample_index: int
    omitted_subject_id: str
    source_subject_ids: tuple[str, ...]
    seed_lineage: SeedLineage
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.resample_index, bool)
            or not isinstance(self.resample_index, int)
            or self.resample_index < 0
        ):
            raise ValueError("resample_index must be a non-negative integer")
        omitted = _name(self.omitted_subject_id, field_name="omitted_subject_id")
        subjects = tuple(sorted(self.source_subject_ids))
        if (
            not subjects
            or len(subjects) != len(set(subjects))
            or any(
                _name(value, field_name="source_subject_id") != value
                for value in subjects
            )
            or omitted not in subjects
        ):
            raise ValueError("LOSO requires a complete unique source-subject universe")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        object.__setattr__(self, "omitted_subject_id", omitted)
        object.__setattr__(self, "source_subject_ids", subjects)
        object.__setattr__(
            self,
            "plan_id",
            stable_id(
                "v7_leave_one_subject_out_plan",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "loso_policy": _LOSO_POLICY,
            "omitted_subject_id": self.omitted_subject_id,
            "resample_index": self.resample_index,
            "seed_lineage": self.seed_lineage.to_dict(),
            "source_subject_ids": list(self.source_subject_ids),
        }

    def to_dict(self) -> dict[str, object]:
        return {"plan_id": self.plan_id, **self._identity_payload()}


V7ResamplingPlan: TypeAlias = (
    SubjectBootstrapPlan | ContextPermutationPlan | V7LeaveOneSubjectOutPlan
)
V7HypothesisAxes: TypeAlias = tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
_COMPACT_CONTINUOUS_COLUMNS = (
    "event_id",
    "contrast_name",
    "effect",
    "status",
    "reason_code",
    "effect_id",
)
_COMPACT_OCCURRENCE_COLUMNS = (
    "event_id",
    "contrast_name",
    "occurrence_effect",
    "occurrence_effect_scale",
    "status",
    "reason_code",
    "effect_id",
)
_COMPACT_HYPERGRAPH_COLUMNS = (
    "edge_id",
    "posterior_effect",
    "status",
    "reason_code",
    "shrinkage_record_id",
)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _positive_count(value: int, *, field_name: str, allow_zero: bool) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparison = ">= 0" if allow_zero else ">= 1"
        raise ValueError(f"{field_name} must be an integer {comparison}")
    return value


def _canonical_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(sorted(values))
    if any(_name(value, field_name=field_name) != value for value in result):
        raise ValueError(f"{field_name} contains a non-canonical name")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique names")
    return result


def _plain(value: object) -> Hashable:
    result = value.item() if isinstance(value, np.generic) else value
    if result is None or result is pd.NA or result is pd.NaT:
        raise ValueError("resampling strata cannot contain missing values")
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("resampling strata cannot contain non-finite values")
    if not isinstance(result, Hashable):
        raise TypeError("resampling strata must be hashable")
    canonical_json(result)
    return result


def _operation(plan: V7ResamplingPlan) -> V7FullPipelineOperation:
    if isinstance(plan, SubjectBootstrapPlan):
        return V7FullPipelineOperation.SUBJECT_BOOTSTRAP
    if isinstance(plan, ContextPermutationPlan):
        return V7FullPipelineOperation.CONDITION_PERMUTATION
    if isinstance(plan, V7LeaveOneSubjectOutPlan):
        return V7FullPipelineOperation.LEAVE_ONE_SUBJECT_OUT
    raise TypeError("unsupported v7 resampling plan")


def _plan_id(plan: V7ResamplingPlan) -> str:
    if isinstance(plan, SubjectBootstrapPlan):
        return cast(str, plan.bootstrap_id)
    if isinstance(plan, ContextPermutationPlan):
        return cast(str, plan.permutation_id)
    if isinstance(plan, V7LeaveOneSubjectOutPlan):
        return plan.plan_id
    raise TypeError("unsupported v7 resampling plan")


def _resample_index(plan: V7ResamplingPlan) -> int:
    return int(plan.resample_index)


def _plan_seed(plan: V7ResamplingPlan) -> SeedLineage:
    return plan.seed_lineage


def _plan_dict(plan: V7ResamplingPlan) -> dict[str, object]:
    if isinstance(plan, SubjectBootstrapPlan):
        return cast(dict[str, object], plan.to_dict())
    if isinstance(plan, ContextPermutationPlan):
        return cast(dict[str, object], plan.to_dict())
    if isinstance(plan, V7LeaveOneSubjectOutPlan):
        return plan.to_dict()
    raise TypeError("unsupported v7 resampling plan")


def _bootstrap_strata_columns(
    design: V7EstimatorSpec,
    *,
    strata_keys: tuple[str, ...],
) -> tuple[str, ...]:
    result = set(strata_keys)
    kind = DifferentialDesignKind(design.design.design_kind)
    if kind in {
        DifferentialDesignKind.INDEPENDENT_TWO_GROUP,
        DifferentialDesignKind.INDEPENDENT_MULTI_GROUP,
    }:
        result.add(design.design.condition_column)
    if design.design.cohort_column is not None:
        result.add(design.design.cohort_column)
    return tuple(sorted(result))


def _plan_design_stratified_bootstraps(
    sample_metadata: pd.DataFrame,
    exchangeability: ExchangeabilityMap,
    estimator_spec: V7EstimatorSpec,
    *,
    subject_column: str,
    n_bootstraps: int,
    strata_keys: tuple[str, ...],
    seed_lineage: SeedLineage,
) -> tuple[SubjectBootstrapPlan, ...]:
    if n_bootstraps == 0:
        return ()
    strata_columns = _bootstrap_strata_columns(
        estimator_spec,
        strata_keys=strata_keys,
    )
    required = {subject_column, *strata_columns}
    missing = required.difference(sample_metadata.columns)
    if missing:
        raise ValueError(f"bootstrap metadata is missing fields: {sorted(missing)}")
    subject_rows = sample_metadata.loc[:, list(required)].drop_duplicates()
    grouped = subject_rows.groupby(subject_column, observed=True, sort=False)
    subject_strata: dict[str, tuple[Hashable, ...]] = {}
    for subject, rows in grouped:
        values: list[Hashable] = []
        for column in strata_columns:
            unique = tuple({_plain(value) for value in rows[column]})
            if len(unique) != 1:
                raise ValueError(
                    f"bootstrap stratum {column!r} varies within subject {subject!r}"
                )
            values.append(unique[0])
        subject_strata[str(subject)] = tuple(values)
    if set(subject_strata) != set(exchangeability.subject_ids):
        raise ValueError("bootstrap subject universe differs from exchangeability map")
    by_stratum: defaultdict[tuple[Hashable, ...], list[str]] = defaultdict(list)
    for subject, stratum in subject_strata.items():
        by_stratum[stratum].append(subject)
    plans: list[SubjectBootstrapPlan] = []
    for resample_index in range(n_bootstraps):
        child_seed = seed_lineage.derive(
            "v7_design_stratified_subject_bootstrap",
            exchangeability.exchangeability_id,
            f"resample={resample_index}",
        )
        rng = child_seed.python_random()
        draws: list[BootstrapSubjectDraw] = []
        draw_index = 0
        for stratum in sorted(by_stratum, key=canonical_json):
            subjects = sorted(by_stratum[stratum])
            for _ in subjects:
                draws.append(
                    BootstrapSubjectDraw(
                        source_subject_id=rng.choice(subjects),
                        bootstrap_subject_id=(
                            f"v7_bootstrap_{resample_index:06d}_{draw_index:06d}"
                        ),
                        stratum=stratum,
                    )
                )
                draw_index += 1
        payload = {
            "exchangeability_id": exchangeability.exchangeability_id,
            "resample_index": resample_index,
            "draws": [
                {
                    "source_subject_id": draw.source_subject_id,
                    "bootstrap_subject_id": draw.bootstrap_subject_id,
                    "stratum": list(draw.stratum),
                }
                for draw in draws
            ],
            "seed_lineage": child_seed.to_dict(),
            "resampling_unit": "subject_block",
            "with_replacement": True,
        }
        plans.append(
            SubjectBootstrapPlan(
                bootstrap_id=stable_id(
                    "subject_bootstrap",
                    payload,
                    schema_version="1",
                ),
                exchangeability_id=exchangeability.exchangeability_id,
                resample_index=resample_index,
                draws=tuple(draws),
                seed_lineage=child_seed,
            )
        )
    return tuple(plans)


def _plan_loso(
    subject_ids: tuple[str, ...],
    *,
    seed_lineage: SeedLineage,
) -> tuple[V7LeaveOneSubjectOutPlan, ...]:
    return tuple(
        V7LeaveOneSubjectOutPlan(
            resample_index=index,
            omitted_subject_id=subject,
            source_subject_ids=subject_ids,
            seed_lineage=seed_lineage.derive(
                "v7_leave_one_subject_out",
                f"subject={subject}",
            ),
        )
        for index, subject in enumerate(subject_ids)
    )


def _sample_metadata_with_invariant_columns(
    adata: AnnData,
    config: CrychicConfig,
) -> pd.DataFrame:
    validated = validate_anndata(adata, _input_schema(config))
    metadata = validated.report.sample_metadata.copy(deep=True)
    source = adata.obs.copy(deep=False)
    grouped = source.groupby(config.sample_key, observed=True, sort=False)
    invariant_columns = [
        str(column)
        for column in source.columns
        if column != config.sample_key
        and grouped[column].nunique(dropna=False).le(1).all()
    ]
    extras = [
        column for column in invariant_columns if column not in metadata.columns
    ]
    if extras:
        sample_rows = source.loc[:, [config.sample_key, *extras]].drop_duplicates(
            config.sample_key,
            keep="first",
        )
        metadata = metadata.merge(
            sample_rows,
            on=config.sample_key,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    return cast(pd.DataFrame, metadata)


def _normalize_sample_metadata(
    metadata: pd.DataFrame,
    config: CrychicConfig,
) -> pd.DataFrame:
    metadata = metadata.copy(deep=True)
    rename: dict[str, str] = {}
    if config.sample_key != "sample_id":
        rename[config.sample_key] = "sample_id"
    if config.subject_key != "subject_id":
        rename[config.subject_key] = "subject_id"
    metadata = metadata.rename(columns=rename)
    if "condition" not in metadata.columns:
        if len(config.context_keys) == 1:
            metadata["condition"] = metadata[config.context_keys[0]].astype(str)
        else:
            metadata["condition"] = [
                canonical_json({key: row[key] for key in config.context_keys})
                for _, row in metadata.iterrows()
            ]
    return cast(pd.DataFrame, metadata)


def _normalized_sample_metadata(
    adata: AnnData,
    config: CrychicConfig,
) -> pd.DataFrame:
    return _normalize_sample_metadata(
        _sample_metadata_with_invariant_columns(adata, config),
        config,
    )


def _design_auxiliary_sample_columns(
    adata: AnnData,
    config: CrychicConfig,
    estimator_spec: V7EstimatorSpec,
) -> tuple[str, ...]:
    design = estimator_spec.design
    candidates = {
        design.condition_column,
        *design.batch_columns,
        *design.continuous_covariates,
        *design.categorical_covariates,
    }
    if design.cohort_column is not None:
        candidates.add(design.cohort_column)
    declared = {
        config.sample_key,
        config.subject_key,
        config.cell_type_key,
        *config.context_keys,
        *config.covariates,
    }
    auxiliary = tuple(sorted(candidates.difference(declared)))
    missing = set(auxiliary).difference(adata.obs.columns)
    if missing:
        raise ValueError(
            "v7 full-refit input is missing sample-level design fields: "
            f"{sorted(missing)}"
        )
    return auxiliary


def _materialize_loso(
    source: AnnData,
    config: CrychicConfig,
    plan: V7LeaveOneSubjectOutPlan,
    *,
    source_snapshot_id: str,
) -> _MaterializedResample:
    subjects = source.obs[config.subject_key].astype(str)
    selected = ~subjects.eq(plan.omitted_subject_id)
    if not selected.any() or selected.all():
        raise ValueError("LOSO plan must remove exactly one observed subject block")
    result = source[selected.to_numpy(), :].copy()
    input_id = stable_id(
        "materialized_v7_loso",
        {
            "loso_policy": _LOSO_POLICY,
            "plan_id": plan.plan_id,
            "source_snapshot_id": source_snapshot_id,
        },
        schema_version=_SCHEMA_VERSION,
    )
    return _validated_materialized(result, config, input_id=input_id)


def _stage_lineage(
    crossfit: CrossFitArtifacts,
    estimator: CrossFitV7EstimatorResult,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    stages: dict[str, tuple[str, ...]] = {
        "fold_split": (crossfit.fold_plan.plan_id,),
        "heldout_score_application": estimator.score_provenance_ids,
        "design_aware_effect_fitting": tuple(
            sorted(estimator.differential.effects["effect_id"].astype(str))
        ),
    }
    activity = tuple(
        sorted(
            fold.absolute_activity_v2_transform.transform_manifest_id
            for fold in crossfit.folds
            if fold.absolute_activity_v2_transform is not None
        )
    )
    if activity:
        stages["absolute_activity_v2_fitting"] = activity
    availability = tuple(
        sorted(fold.application.application_id for fold in crossfit.folds)
    )
    stages["availability_fitting"] = availability
    sender = tuple(
        sorted(
            fold.sender_attribution_v2_functional.functional_id
            for fold in crossfit.folds
            if fold.sender_attribution_v2_functional is not None
        )
    )
    if sender:
        stages["sender_attribution_v2_fitting"] = sender
    program = tuple(
        sorted(
            fold.signed_program_v2_functional.functional_id
            for fold in crossfit.folds
            if fold.signed_program_v2_functional is not None
        )
    )
    if program:
        stages["signed_program_v2_fitting"] = program
    coupling = tuple(
        sorted(
            fold.eb_shrunken_coupling_v2_functional.functional_id
            for fold in crossfit.folds
            if fold.eb_shrunken_coupling_v2_functional is not None
        )
    )
    if coupling:
        stages["eb_shrunken_coupling_v2_fitting"] = coupling
    if estimator.occurrence is not None:
        stages["two_part_occurrence_v2_fitting"] = (estimator.occurrence.result_id,)
    if estimator.hypergraph is not None:
        stages["hypergraph_shrinkage_v2_fitting"] = (estimator.hypergraph.fit.fit_id,)
    return tuple(sorted(stages.items()))


def _estimator_hypothesis_axes(
    estimator: CrossFitV7EstimatorResult,
) -> V7HypothesisAxes:
    axes: dict[str, tuple[tuple[str, str], ...]] = {
        "continuous_raw": tuple(
            sorted(
                (str(row.event_id), str(row.contrast_name))
                for row in estimator.differential.effects.itertuples(index=False)
            )
        )
    }
    if estimator.occurrence is not None:
        scale = _occurrence_effect_scale(estimator.spec)
        axes[f"occurrence_{scale}"] = tuple(
            sorted(
                (str(row.event_id), str(row.contrast_name))
                for row in estimator.occurrence.occurrence.effects.itertuples(
                    index=False
                )
            )
        )
    if estimator.hypergraph is not None:
        axes["hypergraph_posterior"] = tuple(
            sorted(
                (str(row.edge_id), estimator.hypergraph.contrast_name)
                for row in estimator.hypergraph.shrinkage.itertuples(index=False)
            )
        )
    for channel, keys in axes.items():
        if len(keys) != len(set(keys)):
            raise ValueError(f"v7 {channel} hypothesis axis contains duplicates")
    return tuple(sorted(axes.items()))


def _occurrence_effect_scale(spec: V7EstimatorSpec) -> str:
    return (
        "log_odds"
        if DifferentialDesignKind(spec.design.design_kind)
        is DifferentialDesignKind.CONTINUOUS
        else "prevalence_difference"
    )


def _compact_occurrence_effects(
    estimator: CrossFitV7EstimatorResult,
) -> pd.DataFrame:
    occurrence = estimator.occurrence
    if occurrence is None:
        return pd.DataFrame(columns=_COMPACT_OCCURRENCE_COLUMNS)
    scale = _occurrence_effect_scale(estimator.spec)
    source_column = (
        "log_odds_ratio" if scale == "log_odds" else "prevalence_difference"
    )
    table = occurrence.occurrence.effects.loc[
        :,
        [
            "event_id",
            "contrast_name",
            source_column,
            "status",
            "reason_code",
            "effect_id",
        ],
    ].copy()
    numeric = pd.to_numeric(table[source_column], errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    table["occurrence_effect"] = numeric.where(finite, np.nan)
    table["occurrence_effect_scale"] = scale
    table["status"] = np.where(finite, "observed", "not_estimable")
    table["reason_code"] = table["reason_code"].where(
        ~finite,
        None,
    )
    table.loc[
        ~finite & table["reason_code"].isna(),
        "reason_code",
    ] = "occurrence_resampling_effect_not_estimable"
    return table.loc[:, list(_COMPACT_OCCURRENCE_COLUMNS)]


def _require_axes_subset(
    child_axes: V7HypothesisAxes,
    point_axes: V7HypothesisAxes,
) -> None:
    point_by_channel = dict(point_axes)
    child_by_channel = dict(child_axes)
    if set(child_by_channel) != set(point_by_channel):
        raise ContractError(
            "V7 resample estimator channels differ from the point estimator",
            code="v7_resample_channel_axis_mismatch",
            field="channel",
            remediation="Rerun with the exact point estimator specification",
        )
    extras = {
        channel: sorted(set(child_by_channel[channel]).difference(point_keys))
        for channel, point_keys in point_by_channel.items()
        if set(child_by_channel[channel]).difference(point_keys)
    }
    if extras:
        raise ContractError(
            "V7 resample produced hypotheses outside the frozen point axis",
            code="v7_resample_hypothesis_axis_expansion",
            field="event_id,contrast_name",
            remediation=(
                "Freeze the root event universe before resampling and retain "
                "missing child hypotheses as not estimable"
            ),
        )


def _require_hypothesis_axis_subset(
    estimator: CrossFitV7EstimatorResult,
    point_axes: V7HypothesisAxes,
) -> None:
    _require_axes_subset(_estimator_hypothesis_axes(estimator), point_axes)


@dataclass(frozen=True, slots=True, kw_only=True)
class V7CompactEstimatorSnapshot:
    """Minimal per-resample effect tables retained after releasing the child."""

    crossfit_id: str
    source_estimator_id: str
    estimator_spec_id: str
    design_spec_id: str
    occurrence_result_id: str | None
    hypergraph_fit_id: str | None
    hypergraph_prior_id: str | None
    continuous_effects: pd.DataFrame
    occurrence_effects: pd.DataFrame
    hypergraph_effects: pd.DataFrame
    output_digest: str = field(init=False)
    snapshot_id: str = field(init=False)

    def __post_init__(self) -> None:
        crossfit_id = _name(self.crossfit_id, field_name="crossfit_id")
        source_id = _name(self.source_estimator_id, field_name="source_estimator_id")
        spec_id = _name(self.estimator_spec_id, field_name="estimator_spec_id")
        design_id = _name(self.design_spec_id, field_name="design_spec_id")
        occurrence_id = (
            None
            if self.occurrence_result_id is None
            else _name(self.occurrence_result_id, field_name="occurrence_result_id")
        )
        hypergraph_fit_id = (
            None
            if self.hypergraph_fit_id is None
            else _name(self.hypergraph_fit_id, field_name="hypergraph_fit_id")
        )
        hypergraph_prior_id = (
            None
            if self.hypergraph_prior_id is None
            else _name(self.hypergraph_prior_id, field_name="hypergraph_prior_id")
        )
        if (hypergraph_fit_id is None) != (hypergraph_prior_id is None):
            raise ValueError("compact M5 fit and prior IDs must coexist")
        expected_columns = {
            "continuous_effects": _COMPACT_CONTINUOUS_COLUMNS,
            "occurrence_effects": _COMPACT_OCCURRENCE_COLUMNS,
            "hypergraph_effects": _COMPACT_HYPERGRAPH_COLUMNS,
        }
        tables: dict[str, pd.DataFrame] = {}
        for field_name, columns in expected_columns.items():
            table = getattr(self, field_name)
            if not isinstance(table, pd.DataFrame) or tuple(table.columns) != columns:
                raise ValueError(f"{field_name} does not match the compact schema")
            key_columns = (
                ("edge_id",)
                if field_name == "hypergraph_effects"
                else ("event_id", "contrast_name")
            )
            if not table.empty and table.duplicated(list(key_columns)).any():
                raise ValueError(f"{field_name} contains duplicate effect keys")
            tables[field_name] = table.copy(deep=True)
        occurrence_table = tables["occurrence_effects"]
        if not occurrence_table.empty:
            scales = set(occurrence_table["occurrence_effect_scale"].astype(str))
            if scales not in ({"prevalence_difference"}, {"log_odds"}):
                raise ValueError(
                    "occurrence_effects must declare one supported effect scale"
                )
        if not tables["occurrence_effects"].empty and occurrence_id is None:
            raise ValueError("compact occurrence effects require result lineage")
        if not tables["hypergraph_effects"].empty and hypergraph_fit_id is None:
            raise ValueError("compact hypergraph effects require fit lineage")
        output_digest = str(
            canonical_digest(
                {
                    field_name: _frame_digest(table)
                    for field_name, table in tables.items()
                }
            )
        )
        payload = {
            "crossfit_id": crossfit_id,
            "design_spec_id": design_id,
            "estimator_spec_id": spec_id,
            "hypergraph_fit_id": hypergraph_fit_id,
            "hypergraph_prior_id": hypergraph_prior_id,
            "occurrence_result_id": occurrence_id,
            "output_digest": output_digest,
            "source_estimator_id": source_id,
            "version": V7_FULL_PIPELINE_RESAMPLING_VERSION,
        }
        object.__setattr__(self, "crossfit_id", crossfit_id)
        object.__setattr__(self, "source_estimator_id", source_id)
        object.__setattr__(self, "estimator_spec_id", spec_id)
        object.__setattr__(self, "design_spec_id", design_id)
        object.__setattr__(self, "occurrence_result_id", occurrence_id)
        object.__setattr__(self, "hypergraph_fit_id", hypergraph_fit_id)
        object.__setattr__(self, "hypergraph_prior_id", hypergraph_prior_id)
        for field_name, table in tables.items():
            object.__setattr__(self, field_name, table)
        object.__setattr__(self, "output_digest", output_digest)
        object.__setattr__(
            self,
            "snapshot_id",
            stable_id(
                "v7_compact_estimator_snapshot",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )

    @classmethod
    def from_estimator(
        cls,
        estimator: CrossFitV7EstimatorResult,
    ) -> V7CompactEstimatorSnapshot:
        estimator._require_intact()
        occurrence = _compact_occurrence_effects(estimator)
        hypergraph = (
            pd.DataFrame(columns=_COMPACT_HYPERGRAPH_COLUMNS)
            if estimator.hypergraph is None
            else estimator.hypergraph.shrinkage.loc[
                :,
                list(_COMPACT_HYPERGRAPH_COLUMNS),
            ]
        )
        return cls(
            crossfit_id=estimator.crossfit_id,
            source_estimator_id=estimator.estimator_id,
            estimator_spec_id=estimator.spec.spec_id,
            design_spec_id=estimator.spec.design.spec_id,
            occurrence_result_id=(
                None if estimator.occurrence is None else estimator.occurrence.result_id
            ),
            hypergraph_fit_id=(
                None
                if estimator.hypergraph is None
                else estimator.hypergraph.fit.fit_id
            ),
            hypergraph_prior_id=(
                None
                if estimator.hypergraph is None
                else estimator.hypergraph.fit.prior_id
            ),
            continuous_effects=estimator.differential.effects.loc[
                :,
                list(_COMPACT_CONTINUOUS_COLUMNS),
            ],
            occurrence_effects=occurrence,
            hypergraph_effects=hypergraph,
        )

    def _require_intact(self) -> None:
        repeated = V7CompactEstimatorSnapshot(
            crossfit_id=self.crossfit_id,
            source_estimator_id=self.source_estimator_id,
            estimator_spec_id=self.estimator_spec_id,
            design_spec_id=self.design_spec_id,
            occurrence_result_id=self.occurrence_result_id,
            hypergraph_fit_id=self.hypergraph_fit_id,
            hypergraph_prior_id=self.hypergraph_prior_id,
            continuous_effects=self.continuous_effects,
            occurrence_effects=self.occurrence_effects,
            hypergraph_effects=self.hypergraph_effects,
        )
        if (
            repeated.output_digest != self.output_digest
            or repeated.snapshot_id != self.snapshot_id
        ):
            raise ValueError("v7 compact estimator snapshot integrity failed")

    def to_manifest(self) -> dict[str, object]:
        self._require_intact()
        return {
            "snapshot_id": self.snapshot_id,
            "crossfit_id": self.crossfit_id,
            "source_estimator_id": self.source_estimator_id,
            "estimator_spec_id": self.estimator_spec_id,
            "design_spec_id": self.design_spec_id,
            "occurrence_result_id": self.occurrence_result_id,
            "hypergraph_fit_id": self.hypergraph_fit_id,
            "hypergraph_prior_id": self.hypergraph_prior_id,
            "output_digest": self.output_digest,
            "continuous_effect_rows": len(self.continuous_effects),
            "occurrence_effect_rows": len(self.occurrence_effects),
            "hypergraph_effect_rows": len(self.hypergraph_effects),
            "retention_policy": "minimal_effect_tables_only_v1",
        }


def _snapshot_hypothesis_axes(
    snapshot: V7CompactEstimatorSnapshot,
    *,
    hypergraph_contrast_name: str | None,
) -> V7HypothesisAxes:
    axes: dict[str, tuple[tuple[str, str], ...]] = {
        "continuous_raw": tuple(
            sorted(
                (str(row.event_id), str(row.contrast_name))
                for row in snapshot.continuous_effects.itertuples(index=False)
            )
        )
    }
    if snapshot.occurrence_result_id is not None:
        scales = tuple(
            sorted(snapshot.occurrence_effects["occurrence_effect_scale"].unique())
        )
        if len(scales) != 1:
            raise ValueError("compact occurrence snapshot must use one effect scale")
        axes[f"occurrence_{scales[0]}"] = tuple(
            sorted(
                (str(row.event_id), str(row.contrast_name))
                for row in snapshot.occurrence_effects.itertuples(index=False)
            )
        )
    if snapshot.hypergraph_fit_id is not None:
        contrast = _name(
            hypergraph_contrast_name,
            field_name="hypergraph_contrast_name",
        )
        axes["hypergraph_posterior"] = tuple(
            sorted(
                (str(row.edge_id), contrast)
                for row in snapshot.hypergraph_effects.itertuples(index=False)
            )
        )
    return tuple(sorted(axes.items()))


@dataclass(frozen=True, slots=True, kw_only=True)
class V7FullPipelineResampleRecord:
    """Small retained output from one complete v7 estimator rerun."""

    operation: V7FullPipelineOperation
    resample_index: int
    plan_id: str
    materialized_input_id: str
    plan_seed_lineage: SeedLineage
    crossfit_seed_lineage: SeedLineage
    status: V7FullPipelineResampleStatus
    crossfit_id: str | None
    estimator_id: str | None
    stage_lineage: tuple[tuple[str, tuple[str, ...]], ...]
    n_obs: int | None
    n_samples: int | None
    n_subjects: int | None
    omitted_subject_id: str | None
    failure_type: str | None
    failure_code: str | None
    _estimator: V7CompactEstimatorSnapshot | None = field(default=None, repr=False)
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        operation = V7FullPipelineOperation(self.operation)
        status = V7FullPipelineResampleStatus(self.status)
        if (
            isinstance(self.resample_index, bool)
            or not isinstance(self.resample_index, int)
            or self.resample_index < 0
        ):
            raise ValueError("resample_index must be a non-negative integer")
        plan_id = _name(self.plan_id, field_name="plan_id")
        input_id = _name(self.materialized_input_id, field_name="materialized_input_id")
        if not isinstance(self.plan_seed_lineage, SeedLineage) or not isinstance(
            self.crossfit_seed_lineage,
            SeedLineage,
        ):
            raise TypeError("resample lineages must be SeedLineage values")
        if self.crossfit_seed_lineage.path[: len(self.plan_seed_lineage.path)] != (
            self.plan_seed_lineage.path
        ):
            raise ValueError("cross-fit seed must descend from the plan seed")
        stages = tuple(sorted(self.stage_lineage))
        if len({name for name, _ in stages}) != len(stages):
            raise ValueError("stage_lineage stage names must be unique")
        for stage_name, identifiers in stages:
            _name(stage_name, field_name="stage_name")
            if not identifiers or any(
                _name(value, field_name="stage_identifier") != value
                for value in identifiers
            ):
                raise ValueError("stage_lineage identifiers must be non-empty")
        omitted = self.omitted_subject_id
        if operation is V7FullPipelineOperation.LEAVE_ONE_SUBJECT_OUT:
            omitted = _name(omitted, field_name="omitted_subject_id")
        elif omitted is not None:
            raise ValueError("only LOSO records may declare omitted_subject_id")
        if status is V7FullPipelineResampleStatus.SUCCEEDED:
            crossfit_id = _name(self.crossfit_id, field_name="crossfit_id")
            estimator_id = _name(self.estimator_id, field_name="estimator_id")
            if (
                not isinstance(self._estimator, V7CompactEstimatorSnapshot)
                or self._estimator.crossfit_id != crossfit_id
                or self._estimator.source_estimator_id != estimator_id
            ):
                raise ValueError(
                    "successful v7 resample requires its exact compact estimator"
                )
            counts = (
                self.n_obs,
                self.n_samples,
                self.n_subjects,
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in counts
            ):
                raise ValueError("successful v7 resample requires positive counts")
            if self.failure_type is not None or self.failure_code is not None:
                raise ValueError("successful v7 resample cannot carry failure metadata")
            failure_type = None
            failure_code = None
        else:
            if self.crossfit_id is not None or self.estimator_id is not None:
                raise ValueError("failed v7 resample cannot expose result identifiers")
            if self._estimator is not None or stages:
                raise ValueError("failed v7 resample cannot retain estimator stages")
            crossfit_id = None
            estimator_id = None
            failure_type = _name(self.failure_type, field_name="failure_type")
            failure_code = _name(self.failure_code, field_name="failure_code")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "materialized_input_id", input_id)
        object.__setattr__(self, "stage_lineage", stages)
        object.__setattr__(self, "omitted_subject_id", omitted)
        object.__setattr__(self, "crossfit_id", crossfit_id)
        object.__setattr__(self, "estimator_id", estimator_id)
        object.__setattr__(self, "failure_type", failure_type)
        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "v7_full_pipeline_resample_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "crossfit_id": self.crossfit_id,
            "crossfit_seed_lineage": self.crossfit_seed_lineage.to_dict(),
            "estimator_id": self.estimator_id,
            "failure_code": self.failure_code,
            "failure_type": self.failure_type,
            "materialized_input_id": self.materialized_input_id,
            "n_obs": self.n_obs,
            "n_samples": self.n_samples,
            "n_subjects": self.n_subjects,
            "omitted_subject_id": self.omitted_subject_id,
            "operation": self.operation.value,
            "plan_id": self.plan_id,
            "plan_seed_lineage": self.plan_seed_lineage.to_dict(),
            "resample_index": self.resample_index,
            "stage_lineage": [
                [stage, list(identifiers)] for stage, identifiers in self.stage_lineage
            ],
            "status": self.status.value,
            "version": V7_FULL_PIPELINE_RESAMPLING_VERSION,
        }

    @property
    def estimator(self) -> V7CompactEstimatorSnapshot | None:
        return self._estimator

    def _require_intact(self) -> None:
        if self._estimator is not None:
            self._estimator._require_intact()
        repeated = V7FullPipelineResampleRecord(
            operation=self.operation,
            resample_index=self.resample_index,
            plan_id=self.plan_id,
            materialized_input_id=self.materialized_input_id,
            plan_seed_lineage=self.plan_seed_lineage,
            crossfit_seed_lineage=self.crossfit_seed_lineage,
            status=self.status,
            crossfit_id=self.crossfit_id,
            estimator_id=self.estimator_id,
            stage_lineage=self.stage_lineage,
            n_obs=self.n_obs,
            n_samples=self.n_samples,
            n_subjects=self.n_subjects,
            omitted_subject_id=self.omitted_subject_id,
            failure_type=self.failure_type,
            failure_code=self.failure_code,
            _estimator=self._estimator,
        )
        if repeated.record_id != self.record_id:
            raise ValueError("v7 full-pipeline resample record integrity failed")

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "record_id": self.record_id,
            **self._identity_payload(),
            "estimator": (
                None if self._estimator is None else self._estimator.to_manifest()
            ),
        }


def _failure_metadata(error: Exception) -> tuple[str, str]:
    failure_type = f"{type(error).__module__}.{type(error).__qualname__}"
    if isinstance(error, CrychicError):
        return failure_type, error.details.code
    return failure_type, type(error).__qualname__


def _execute_plan(
    source: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    crossfit_spec: CrossFitSpec,
    estimator_spec: V7EstimatorSpec,
    exchangeability: ExchangeabilityMap,
    source_receiver_universe: FrozenReceiverUniverse,
    hypergraph_prior: FrozenHypergraphPrior | None,
    point_hypothesis_axes: V7HypothesisAxes,
    auxiliary_sample_columns: tuple[str, ...],
    plan: V7ResamplingPlan,
    *,
    source_snapshot_id: str,
) -> V7FullPipelineResampleRecord:
    operation = _operation(plan)
    plan_id = _plan_id(plan)
    crossfit_seed = _plan_seed(plan).derive("v7_full_pipeline_crossfit", plan_id)
    child_config = replace(config, random_seed=crossfit_seed.seed)
    if isinstance(plan, V7LeaveOneSubjectOutPlan):
        input_id = stable_id(
            "materialized_v7_loso",
            {
                "loso_policy": _LOSO_POLICY,
                "plan_id": plan.plan_id,
                "source_snapshot_id": source_snapshot_id,
            },
            schema_version=_SCHEMA_VERSION,
        )
    else:
        input_id = stable_id(
            "materialized_full_pipeline_resample",
            {
                "materialization_policy": (
                    "deterministic_complete_subject_or_sample_block_v1"
                ),
                "operation": (
                    "subject_bootstrap"
                    if isinstance(plan, SubjectBootstrapPlan)
                    else "context_permutation"
                ),
                "plan_id": plan_id,
                "source_snapshot_id": source_snapshot_id,
            },
            schema_version="1.0.0",
        )
    materialized = None
    try:
        if isinstance(plan, SubjectBootstrapPlan):
            materialized = _materialize_subject_bootstrap(
                source,
                child_config,
                plan,
                source_snapshot_id=source_snapshot_id,
            )
        elif isinstance(plan, ContextPermutationPlan):
            materialized = _materialize_context_permutation(
                source,
                child_config,
                exchangeability,
                plan,
                source_snapshot_id=source_snapshot_id,
            )
        else:
            materialized = _materialize_loso(
                source,
                child_config,
                plan,
                source_snapshot_id=source_snapshot_id,
            )
        child_snapshot = _sanitized_raw_input_snapshot(
            materialized.adata,
            child_config,
            auxiliary_sample_columns=auxiliary_sample_columns,
        )
        child = _run_subject_crossfit(
            child_snapshot,
            child_config,
            resource_bundle,
            target_prior,
            spec=crossfit_spec,
            _receiver_axis_source=source_receiver_universe,
        )
        estimator = fit_crossfit_v7_estimator(
            child,
            estimator_spec,
            sample_metadata=_normalized_sample_metadata(
                materialized.adata,
                child_config,
            ),
            hypergraph_prior=hypergraph_prior,
        )
        _require_hypothesis_axis_subset(estimator, point_hypothesis_axes)
        stage_lineage = _stage_lineage(child, estimator)
        snapshot = V7CompactEstimatorSnapshot.from_estimator(estimator)
        return V7FullPipelineResampleRecord(
            operation=operation,
            resample_index=_resample_index(plan),
            plan_id=plan_id,
            materialized_input_id=input_id,
            plan_seed_lineage=_plan_seed(plan),
            crossfit_seed_lineage=crossfit_seed,
            status=V7FullPipelineResampleStatus.SUCCEEDED,
            crossfit_id=child.crossfit_id,
            estimator_id=estimator.estimator_id,
            stage_lineage=stage_lineage,
            n_obs=materialized.n_obs,
            n_samples=materialized.n_samples,
            n_subjects=materialized.n_subjects,
            omitted_subject_id=(
                plan.omitted_subject_id
                if isinstance(plan, V7LeaveOneSubjectOutPlan)
                else None
            ),
            failure_type=None,
            failure_code=None,
            _estimator=snapshot,
        )
    except Exception as error:
        failure_type, failure_code = _failure_metadata(error)
        return V7FullPipelineResampleRecord(
            operation=operation,
            resample_index=_resample_index(plan),
            plan_id=plan_id,
            materialized_input_id=input_id,
            plan_seed_lineage=_plan_seed(plan),
            crossfit_seed_lineage=crossfit_seed,
            status=V7FullPipelineResampleStatus.FAILED,
            crossfit_id=None,
            estimator_id=None,
            stage_lineage=(),
            n_obs=None if materialized is None else materialized.n_obs,
            n_samples=None if materialized is None else materialized.n_samples,
            n_subjects=None if materialized is None else materialized.n_subjects,
            omitted_subject_id=(
                plan.omitted_subject_id
                if isinstance(plan, V7LeaveOneSubjectOutPlan)
                else None
            ),
            failure_type=failure_type,
            failure_code=failure_code,
            _estimator=None,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class V7FullPipelineResamplingResult:
    """Point estimator plus exact small outputs from every requested full refit."""

    point: CrossFitV7EstimatorResult
    plans: tuple[V7ResamplingPlan, ...]
    records: tuple[V7FullPipelineResampleRecord, ...]
    source_snapshot_id: str
    source_input_digest: str
    config_digest: str
    crossfit_spec_id: str
    estimator_spec_id: str
    resource_bundle_content_id: str
    target_prior_content_id: str
    exchangeability_id: str
    permutation_context_keys: tuple[str, ...]
    permutation_strata_keys: tuple[str, ...]
    permutation_immutable_covariates: tuple[str, ...]
    root_seed_lineage: SeedLineage
    requested_n_jobs: int
    effective_n_jobs: int
    hypothesis_axis_id: str = field(init=False)
    result_id: str = field(init=False)
    execution_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.point, CrossFitV7EstimatorResult):
            raise TypeError("point must be CrossFitV7EstimatorResult")
        plans = tuple(self.plans)
        records = tuple(self.records)
        if len(plans) != len(records) or not plans:
            raise ValueError("v7 resampling requires one record per non-empty plan set")
        plan_ids = tuple(_plan_id(plan) for plan in plans)
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("v7 resampling plan identifiers must be unique")
        if tuple(record.plan_id for record in records) != plan_ids:
            raise ValueError("v7 resampling records do not align with plan order")
        if tuple(record.operation for record in records) != tuple(
            _operation(plan) for plan in plans
        ):
            raise ValueError("v7 resampling record operations differ from plans")
        identifiers = {
            field_name: _name(getattr(self, field_name), field_name=field_name)
            for field_name in (
                "source_snapshot_id",
                "source_input_digest",
                "config_digest",
                "crossfit_spec_id",
                "estimator_spec_id",
                "resource_bundle_content_id",
                "target_prior_content_id",
                "exchangeability_id",
            )
        }
        permutation_context_keys = _canonical_names(
            self.permutation_context_keys,
            field_name="permutation_context_keys",
        )
        permutation_strata_keys = _canonical_names(
            self.permutation_strata_keys,
            field_name="permutation_strata_keys",
        )
        permutation_immutable_covariates = _canonical_names(
            self.permutation_immutable_covariates,
            field_name="permutation_immutable_covariates",
        )
        if not permutation_context_keys:
            raise ValueError("permutation_context_keys cannot be empty")
        if set(permutation_strata_keys).difference(permutation_immutable_covariates):
            raise ValueError("permutation strata must be declared immutable covariates")
        if self.point.spec.spec_id != identifiers["estimator_spec_id"]:
            raise ValueError("point estimator spec differs from resampling spec")
        point_axes = _estimator_hypothesis_axes(self.point)
        hypothesis_axis_id = stable_id(
            "v7_full_pipeline_hypothesis_axis",
            {
                "channels": [
                    {
                        "channel": channel,
                        "keys": [list(key) for key in keys],
                    }
                    for channel, keys in point_axes
                ],
                "estimator_spec_id": identifiers["estimator_spec_id"],
            },
            schema_version=_SCHEMA_VERSION,
        )
        for record in records:
            if record.estimator is not None:
                _require_axes_subset(
                    _snapshot_hypothesis_axes(
                        record.estimator,
                        hypergraph_contrast_name=(
                            self.point.spec.hypergraph_contrast_name
                        ),
                    ),
                    point_axes,
                )
        if not isinstance(self.root_seed_lineage, SeedLineage):
            raise TypeError("root_seed_lineage must be a SeedLineage")
        requested = _positive_count(
            self.requested_n_jobs,
            field_name="requested_n_jobs",
            allow_zero=False,
        )
        effective = _positive_count(
            self.effective_n_jobs,
            field_name="effective_n_jobs",
            allow_zero=False,
        )
        if effective > min(requested, len(plans)):
            raise ValueError("effective_n_jobs exceeds the requested or useful count")
        scientific_payload = {
            **identifiers,
            "hypothesis_axis_id": hypothesis_axis_id,
            "point_estimator_id": self.point.estimator_id,
            "permutation_context_keys": list(permutation_context_keys),
            "permutation_strata_keys": list(permutation_strata_keys),
            "permutation_immutable_covariates": list(permutation_immutable_covariates),
            "record_ids": [record.record_id for record in records],
            "root_seed_lineage": self.root_seed_lineage.to_dict(),
            "version": V7_FULL_PIPELINE_RESAMPLING_VERSION,
        }
        for field_name, value in identifiers.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "plans", plans)
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "permutation_context_keys",
            permutation_context_keys,
        )
        object.__setattr__(
            self,
            "permutation_strata_keys",
            permutation_strata_keys,
        )
        object.__setattr__(
            self,
            "permutation_immutable_covariates",
            permutation_immutable_covariates,
        )
        object.__setattr__(self, "hypothesis_axis_id", hypothesis_axis_id)
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "v7_full_pipeline_resampling_result",
                scientific_payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )
        object.__setattr__(
            self,
            "execution_id",
            stable_id(
                "v7_full_pipeline_resampling_execution",
                {
                    "effective_n_jobs": effective,
                    "requested_n_jobs": requested,
                    "result_id": self.result_id,
                },
                schema_version=_SCHEMA_VERSION,
            ),
        )

    @property
    def successful_records(self) -> tuple[V7FullPipelineResampleRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.status is V7FullPipelineResampleStatus.SUCCEEDED
        )

    def _require_intact(self) -> None:
        self.point._require_intact()
        for record in self.records:
            record._require_intact()
        repeated = V7FullPipelineResamplingResult(
            point=self.point,
            plans=self.plans,
            records=self.records,
            source_snapshot_id=self.source_snapshot_id,
            source_input_digest=self.source_input_digest,
            config_digest=self.config_digest,
            crossfit_spec_id=self.crossfit_spec_id,
            estimator_spec_id=self.estimator_spec_id,
            resource_bundle_content_id=self.resource_bundle_content_id,
            target_prior_content_id=self.target_prior_content_id,
            exchangeability_id=self.exchangeability_id,
            permutation_context_keys=self.permutation_context_keys,
            permutation_strata_keys=self.permutation_strata_keys,
            permutation_immutable_covariates=(self.permutation_immutable_covariates),
            root_seed_lineage=self.root_seed_lineage,
            requested_n_jobs=self.requested_n_jobs,
            effective_n_jobs=self.effective_n_jobs,
        )
        if (
            repeated.hypothesis_axis_id != self.hypothesis_axis_id
            or repeated.result_id != self.result_id
            or repeated.execution_id != self.execution_id
        ):
            raise ValueError("v7 full-pipeline resampling integrity failed")

    def continuous_effect_ledger(self) -> pd.DataFrame:
        """Return one typed event/contrast row for every successful rerun."""

        rows: list[dict[str, object]] = []
        for record in self.successful_records:
            estimator = record.estimator
            assert estimator is not None
            for row in cast(
                list[dict[str, object]],
                estimator.continuous_effects.to_dict(orient="records"),
            ):
                rows.append(
                    {
                        "operation": record.operation.value,
                        "resample_index": record.resample_index,
                        "plan_id": record.plan_id,
                        "record_id": record.record_id,
                        "estimator_id": estimator.source_estimator_id,
                        **row,
                    }
                )
        return pd.DataFrame(rows)

    def occurrence_effect_ledger(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for record in self.successful_records:
            estimator = record.estimator
            assert estimator is not None
            if estimator.occurrence_effects.empty:
                continue
            for row in cast(
                list[dict[str, object]],
                estimator.occurrence_effects.to_dict(orient="records"),
            ):
                rows.append(
                    {
                        "operation": record.operation.value,
                        "resample_index": record.resample_index,
                        "plan_id": record.plan_id,
                        "record_id": record.record_id,
                        "estimator_id": estimator.source_estimator_id,
                        **row,
                    }
                )
        return pd.DataFrame(rows)

    def hypergraph_effect_ledger(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for record in self.successful_records:
            estimator = record.estimator
            assert estimator is not None
            if estimator.hypergraph_effects.empty:
                continue
            contrast_name = self.point.spec.hypergraph_contrast_name
            assert contrast_name is not None
            for row in cast(
                list[dict[str, object]],
                estimator.hypergraph_effects.to_dict(orient="records"),
            ):
                rows.append(
                    {
                        "operation": record.operation.value,
                        "resample_index": record.resample_index,
                        "plan_id": record.plan_id,
                        "record_id": record.record_id,
                        "estimator_id": estimator.source_estimator_id,
                        "contrast_name": contrast_name,
                        **row,
                    }
                )
        return pd.DataFrame(rows)

    def to_manifest(self) -> dict[str, object]:
        self._require_intact()
        counts = {
            operation.value: sum(
                record.operation is operation for record in self.records
            )
            for operation in V7FullPipelineOperation
        }
        succeeded = {
            operation.value: sum(
                record.operation is operation
                and record.status is V7FullPipelineResampleStatus.SUCCEEDED
                for record in self.records
            )
            for operation in V7FullPipelineOperation
        }
        configured_stages = tuple(
            sorted(
                {
                    stage
                    for record in self.successful_records
                    for stage, _ in record.stage_lineage
                }
            )
        )
        return {
            "result_id": self.result_id,
            "execution_id": self.execution_id,
            "version": V7_FULL_PIPELINE_RESAMPLING_VERSION,
            "point_estimator": self.point.to_manifest(),
            "source_snapshot_id": self.source_snapshot_id,
            "source_input_digest": self.source_input_digest,
            "config_digest": self.config_digest,
            "crossfit_spec_id": self.crossfit_spec_id,
            "estimator_spec_id": self.estimator_spec_id,
            "hypothesis_axis_id": self.hypothesis_axis_id,
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "target_prior_content_id": self.target_prior_content_id,
            "exchangeability_id": self.exchangeability_id,
            "permutation_context_keys": list(self.permutation_context_keys),
            "permutation_strata_keys": list(self.permutation_strata_keys),
            "permutation_immutable_covariates": list(
                self.permutation_immutable_covariates
            ),
            "root_seed_lineage": self.root_seed_lineage.to_dict(),
            "requested_n_jobs": self.requested_n_jobs,
            "effective_n_jobs": self.effective_n_jobs,
            "plan_counts": counts,
            "successful_counts": succeeded,
            "bootstrap_policy": _BOOTSTRAP_POLICY,
            "loso_policy": _LOSO_POLICY,
            "full_pipeline_refit_per_resample": True,
            "rerun_stage_catalog": list(_RERUN_STAGES),
            "configured_rerun_stages_observed": list(configured_stages),
            "large_crossfit_children_retained": False,
            "resample_retention_policy": "minimal_effect_tables_only_v1",
            "formal_inference_status": (
                "requires_v7_resampling_finalizer_and_calibration"
            ),
            "plans": [_plan_dict(plan) for plan in self.plans],
            "records": [record.to_dict() for record in self.records],
        }


def run_v7_full_pipeline_resampling(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    crossfit_spec: CrossFitSpec,
    estimator_spec: V7EstimatorSpec,
    hypergraph_prior: FrozenHypergraphPrior | None = None,
    n_bootstraps: int = 0,
    n_permutations: int = 0,
    run_loso: bool = True,
    strata_keys: Sequence[str] | None = None,
    immutable_covariates: Sequence[str] | None = None,
    multi_context_permutation_operation: (
        ContextPermutationOperation | str | None
    ) = None,
    seed_lineage: SeedLineage | None = None,
    n_jobs: int = 1,
    progress_callback: (
        Callable[[V7FullPipelineResampleRecord, int, int], None] | None
    ) = None,
) -> V7FullPipelineResamplingResult:
    """Refit every configured v7 estimator stage in each subject-level resample."""

    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be CrychicConfig")
    if not isinstance(crossfit_spec, CrossFitSpec):
        raise TypeError("crossfit_spec must be CrossFitSpec")
    crossfit_spec._require_intact()
    if not isinstance(estimator_spec, V7EstimatorSpec):
        raise TypeError("estimator_spec must be V7EstimatorSpec")
    if (hypergraph_prior is None) != (estimator_spec.hypergraph_spec is None):
        raise ValueError("hypergraph_prior must match the configured M5 stage")
    bootstraps = _positive_count(
        n_bootstraps,
        field_name="n_bootstraps",
        allow_zero=True,
    )
    permutations = _positive_count(
        n_permutations,
        field_name="n_permutations",
        allow_zero=True,
    )
    if not isinstance(run_loso, bool):
        raise TypeError("run_loso must be boolean")
    requested_jobs = _positive_count(n_jobs, field_name="n_jobs", allow_zero=False)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    auxiliary_sample_columns = _design_auxiliary_sample_columns(
        adata,
        config,
        estimator_spec,
    )
    snapshot = _sanitized_raw_input_snapshot(
        adata,
        config,
        auxiliary_sample_columns=auxiliary_sample_columns,
    )
    snapshot._require_intact()
    source_metadata = _sample_metadata_with_invariant_columns(
        snapshot.adata,
        config,
    )
    normalized_metadata = _normalize_sample_metadata(source_metadata, config)
    resolved_strata = tuple(
        crossfit_spec.strata_keys if strata_keys is None else strata_keys
    )
    permutation_context_keys = tuple(
        sorted({*config.context_keys, estimator_spec.design.condition_column})
    )
    permutation_strata_keys = set(resolved_strata)
    if estimator_spec.design.cohort_column is not None:
        permutation_strata_keys.add(estimator_spec.design.cohort_column)
    resolved_permutation_strata = tuple(sorted(permutation_strata_keys))
    supplied_immutable = (
        resolved_strata if immutable_covariates is None else immutable_covariates
    )
    resolved_immutable = tuple(
        sorted({*supplied_immutable, *resolved_permutation_strata})
    )
    exchangeability = build_exchangeability_map(
        source_metadata,
        sample_key=config.sample_key,
        subject_key=config.subject_key,
        context_keys=permutation_context_keys,
        strata_keys=resolved_permutation_strata,
        immutable_covariates=resolved_immutable,
        multi_context_operation=multi_context_permutation_operation,
    )
    lineage = (
        SeedLineage(config.random_seed).derive(
            "v7_full_pipeline_resampling",
            exchangeability.exchangeability_id,
            crossfit_spec.spec_id,
            estimator_spec.spec_id,
        )
        if seed_lineage is None
        else seed_lineage
    )
    if not isinstance(lineage, SeedLineage):
        raise TypeError("seed_lineage must be SeedLineage or None")

    point_crossfit = _run_subject_crossfit(
        snapshot,
        config,
        resource_bundle,
        target_prior,
        spec=crossfit_spec,
    )
    point = fit_crossfit_v7_estimator(
        point_crossfit,
        estimator_spec,
        sample_metadata=normalized_metadata,
        hypergraph_prior=hypergraph_prior,
    )
    bootstrap_plans = _plan_design_stratified_bootstraps(
        normalized_metadata,
        exchangeability,
        estimator_spec,
        subject_column="subject_id",
        n_bootstraps=bootstraps,
        strata_keys=resolved_strata,
        seed_lineage=lineage,
    )
    permutation_plans = (
        plan_context_permutations(
            exchangeability,
            n_permutations=permutations,
            seed_lineage=lineage,
        )
        if permutations
        else ()
    )
    loso_plans = (
        _plan_loso(exchangeability.subject_ids, seed_lineage=lineage)
        if run_loso
        else ()
    )
    plans: tuple[V7ResamplingPlan, ...] = (
        *bootstrap_plans,
        *permutation_plans,
        *loso_plans,
    )
    if not plans:
        raise ValueError("request bootstrap, permutation, or LOSO full refits")
    effective_jobs = min(requested_jobs, len(plans))
    execute = partial(
        _execute_plan,
        snapshot.adata,
        config,
        resource_bundle,
        target_prior,
        crossfit_spec,
        estimator_spec,
        exchangeability,
        point_crossfit.receiver_universe,
        hypergraph_prior,
        _estimator_hypothesis_axes(point),
        auxiliary_sample_columns,
        source_snapshot_id=snapshot.snapshot_id,
    )
    total_plans = len(plans)
    if effective_jobs == 1:
        serial_records: list[V7FullPipelineResampleRecord] = []
        for completed, plan in enumerate(plans, start=1):
            record = execute(plan=plan)
            serial_records.append(record)
            if progress_callback is not None:
                progress_callback(record, completed, total_plans)
        records = tuple(serial_records)
    else:
        with ThreadPoolExecutor(
            max_workers=effective_jobs,
            thread_name_prefix="crychic-v7-full-refit",
        ) as executor:
            pending = {
                executor.submit(execute, plan=plan): index
                for index, plan in enumerate(plans)
            }
            ordered: list[V7FullPipelineResampleRecord | None] = [
                None
            ] * total_plans
            completed = 0
            for future in as_completed(pending):
                record = future.result()
                ordered[pending[future]] = record
                completed += 1
                if progress_callback is not None:
                    progress_callback(record, completed, total_plans)
            if any(record is None for record in ordered):  # pragma: no cover
                raise RuntimeError("v7 full-refit executor lost a planned record")
            records = tuple(cast(V7FullPipelineResampleRecord, row) for row in ordered)
    return V7FullPipelineResamplingResult(
        point=point,
        plans=plans,
        records=records,
        source_snapshot_id=snapshot.snapshot_id,
        source_input_digest=snapshot.identity.input_digest,
        config_digest=config.digest,
        crossfit_spec_id=crossfit_spec.spec_id,
        estimator_spec_id=estimator_spec.spec_id,
        resource_bundle_content_id=_resource_bundle_content_id(resource_bundle),
        target_prior_content_id=_target_prior_content_id(target_prior),
        exchangeability_id=exchangeability.exchangeability_id,
        permutation_context_keys=permutation_context_keys,
        permutation_strata_keys=resolved_permutation_strata,
        permutation_immutable_covariates=resolved_immutable,
        root_seed_lineage=lineage,
        requested_n_jobs=requested_jobs,
        effective_n_jobs=effective_jobs,
    )


__all__ = [
    "V7_FULL_PIPELINE_RESAMPLING_VERSION",
    "V7FullPipelineOperation",
    "V7FullPipelineResampleRecord",
    "V7FullPipelineResampleStatus",
    "V7FullPipelineResamplingResult",
    "V7LeaveOneSubjectOutPlan",
    "run_v7_full_pipeline_resampling",
]
