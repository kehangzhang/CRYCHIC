"""Strict workflow adapters from family-common OOF scores to inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd

from crychic.core import CommunicationMode, ContractError, CrychicError, stable_id
from crychic.inference import (
    FrozenHypothesisUniverse,
    FullPipelineEffectDistribution,
    FullPipelineEffectDistributionSpec,
    FullPipelineEffectResamplingKind,
    FullPipelineResampleEffectRecord,
    FullPipelineResampleEffectStatus,
    G3FrequencyCalibrationGate,
    HypothesisDeclaration,
    HypothesisPrefilterStatus,
    HypothesisRole,
    OOFContextEffectResult,
    OOFEffectSpec,
    fit_oof_context_effect,
    summarize_full_pipeline_effect_distribution,
)
from crychic.scoring import (
    FamilyCommonScoringApplication,
    FamilyCommonScoringFunctional,
)

from .crossfit import CrossFitArtifacts
from .full_pipeline_resampling import (
    FullPipelineResampleStatus,
    FullPipelineResamplingResult,
    aligned_full_pipeline_resamples,
)
from .receiver_universe import ReceiverTrainingSupportStatus

_SCHEMA_VERSION = "1.0.0"
_SCORE_COLUMN = "integrated_lr_score"
_SCORE_SEMANTICS = "family_common_integrated_lr_score_v1"
_FROZEN_EFFECT_ENDPOINT = "family_common_integrated_lr_context_effect_v1"
_FROZEN_OMNIBUS_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_FROZEN_ENDPOINTS = frozenset((_FROZEN_EFFECT_ENDPOINT, _FROZEN_OMNIBUS_ENDPOINT))
_FROZEN_TARGET_PRODUCER_MARKER = "crychic.workflow.frozen_family_effect_target.v1"
_OOF_COLUMNS = (
    "subject_id",
    "sample_id",
    "fold_id",
    "context_id",
    "score",
    "score_status",
    "scoring_function_id",
)


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FamilyEffectTarget:
    """Stable family-common hypothesis target at one fixed score grain."""

    contrast_name: str
    receiver: str
    family_id: str
    mode: CommunicationMode
    score_column: str = field(default=_SCORE_COLUMN, init=False)
    hypothesis_id: str = field(init=False)
    target_id: str = field(init=False)

    def __post_init__(self) -> None:
        contrast = _name(self.contrast_name, field_name="contrast_name")
        receiver = _name(self.receiver, field_name="receiver")
        family = _name(self.family_id, field_name="family_id")
        mode = CommunicationMode(self.mode)
        payload = {
            "contrast_name": contrast,
            "receiver": receiver,
            "family_id": family,
            "mode": mode.value,
            "score_column": _SCORE_COLUMN,
            "score_semantics": _SCORE_SEMANTICS,
        }
        hypothesis_id = stable_id(
            "family_effect_hypothesis",
            payload,
            schema_version=_SCHEMA_VERSION,
        )
        object.__setattr__(self, "contrast_name", contrast)
        object.__setattr__(self, "receiver", receiver)
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "hypothesis_id", hypothesis_id)
        object.__setattr__(
            self,
            "target_id",
            stable_id(
                "family_effect_target",
                {**payload, "hypothesis_id": hypothesis_id},
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "hypothesis_id": self.hypothesis_id,
            "contrast_name": self.contrast_name,
            "receiver": self.receiver,
            "family_id": self.family_id,
            "mode": self.mode.value,
            "score_column": self.score_column,
            "score_semantics": _SCORE_SEMANTICS,
        }


@dataclass(frozen=True, slots=True, init=False)
class FrozenFamilyEffectTarget:
    """Producer-owned family-effect target bound to one complete universe."""

    universe_id: str
    universe_name: str
    universe_multiplicity_denominator: int
    universe_multiplicity_family_sizes: tuple[tuple[str, int], ...]
    declaration_id: str
    hypothesis_key: str
    hypothesis_id: str
    endpoint: str
    contrast_name: str
    receiver: str
    family_id: str
    mode: CommunicationMode
    role: HypothesisRole
    multiplicity_family: str
    parent_key: str | None
    score_column: str
    target_id: str
    _universe: FrozenHypothesisUniverse
    _declaration: HypothesisDeclaration
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenFamilyEffectTarget is producer-owned; use "
            "build_frozen_family_effect_target()"
        )

    @classmethod
    def _from_declaration(
        cls,
        universe: FrozenHypothesisUniverse,
        declaration: HypothesisDeclaration,
    ) -> FrozenFamilyEffectTarget:
        universe._require_intact()
        declaration._require_intact()
        if universe.declaration_for(declaration.hypothesis_id) is not declaration:
            raise ContractError(
                "Hypothesis declaration is not the exact universe child",
                code="family_effect_declaration_universe_mismatch",
                field="declaration_id",
                remediation="Resolve the declaration from the frozen universe",
            )
        self = object.__new__(cls)
        values: dict[str, object] = {
            "universe_id": universe.universe_id,
            "universe_name": universe.universe_name,
            "universe_multiplicity_denominator": (universe.multiplicity_denominator),
            "universe_multiplicity_family_sizes": (universe.multiplicity_family_sizes),
            "declaration_id": declaration.declaration_id,
            "hypothesis_key": declaration.hypothesis_key,
            "hypothesis_id": declaration.hypothesis_id,
            "endpoint": declaration.endpoint,
            "contrast_name": declaration.contrast_name,
            "receiver": declaration.receiver,
            "family_id": declaration.family_id,
            "mode": declaration.mode,
            "role": declaration.role,
            "multiplicity_family": declaration.multiplicity_family,
            "parent_key": declaration.parent_key,
            "score_column": _SCORE_COLUMN,
            "_universe": universe,
            "_declaration": declaration,
            "_producer_marker": _FROZEN_TARGET_PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "target_id",
            stable_id(
                "frozen_family_effect_target",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    @property
    def q_value_release_allowed(self) -> bool:
        return False

    @property
    def complete_universe_coverage_claimed(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "universe_name": self.universe_name,
            "universe_multiplicity_denominator": (
                self.universe_multiplicity_denominator
            ),
            "universe_multiplicity_family_sizes": [
                [name, count] for name, count in self.universe_multiplicity_family_sizes
            ],
            "declaration_id": self.declaration_id,
            "hypothesis_key": self.hypothesis_key,
            "hypothesis_id": self.hypothesis_id,
            "endpoint": self.endpoint,
            "contrast_name": self.contrast_name,
            "receiver": self.receiver,
            "family_id": self.family_id,
            "mode": self.mode.value,
            "role": self.role.value,
            "multiplicity_family": self.multiplicity_family,
            "parent_key": self.parent_key,
            "score_column": self.score_column,
            "score_semantics": _SCORE_SEMANTICS,
            "complete_universe_coverage_claimed": False,
            "q_value_release_allowed": False,
        }

    def _require_intact(self) -> None:
        try:
            self._universe._require_intact()
            self._declaration._require_intact()
            resolved = self._universe.declaration_for(self.hypothesis_id)
            expected = (
                self._universe.universe_id,
                self._universe.universe_name,
                self._universe.multiplicity_denominator,
                self._universe.multiplicity_family_sizes,
                resolved.declaration_id,
                resolved.hypothesis_key,
                resolved.hypothesis_id,
                resolved.endpoint,
                resolved.contrast_name,
                resolved.receiver,
                resolved.family_id,
                resolved.mode,
                resolved.role,
                resolved.multiplicity_family,
                resolved.parent_key,
                _SCORE_COLUMN,
            )
            observed = (
                self.universe_id,
                self.universe_name,
                self.universe_multiplicity_denominator,
                self.universe_multiplicity_family_sizes,
                self.declaration_id,
                self.hypothesis_key,
                self.hypothesis_id,
                self.endpoint,
                self.contrast_name,
                self.receiver,
                self.family_id,
                self.mode,
                self.role,
                self.multiplicity_family,
                self.parent_key,
                self.score_column,
            )
            expected_id = stable_id(
                "frozen_family_effect_target",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _FROZEN_TARGET_PRODUCER_MARKER
                and resolved is self._declaration
                and observed == expected
                and self.endpoint in _FROZEN_ENDPOINTS
                and (
                    self.endpoint != _FROZEN_OMNIBUS_ENDPOINT
                    or self.role is HypothesisRole.PRIMARY
                )
                and resolved.prefilter_status is HypothesisPrefilterStatus.INCLUDED
                and expected_id == self.target_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen family-effect target failed integrity validation",
                code="frozen_family_effect_target_integrity_violation",
                field="target_id",
                remediation="Rebuild the target from the intact frozen universe",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen family-effect target failed integrity validation",
                code="frozen_family_effect_target_integrity_violation",
                field="target_id",
                remediation="Rebuild the target from the intact frozen universe",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "target_id": self.target_id,
            **self._identity_payload(),
            "declaration": self._declaration.to_dict(),
        }


FamilyEffectTargetLike: TypeAlias = FamilyEffectTarget | FrozenFamilyEffectTarget


def build_frozen_family_effect_target(
    universe: FrozenHypothesisUniverse,
    hypothesis_id: str,
) -> FrozenFamilyEffectTarget:
    """Bind one included family-common endpoint from the complete universe."""

    if not isinstance(universe, FrozenHypothesisUniverse):
        raise TypeError("universe must be a FrozenHypothesisUniverse")
    universe._require_intact()
    declaration = universe.declaration_for(hypothesis_id)
    if declaration.endpoint not in _FROZEN_ENDPOINTS:
        raise ContractError(
            "Hypothesis endpoint is not the frozen family-common effect endpoint",
            code="family_effect_hypothesis_endpoint_mismatch",
            field="endpoint",
            remediation=(
                "Use a supported family-common context-effect or primary "
                "omnibus endpoint"
            ),
        )
    if (
        declaration.endpoint == _FROZEN_OMNIBUS_ENDPOINT
        and declaration.role is not HypothesisRole.PRIMARY
    ):
        raise ContractError(
            "The driver-family context omnibus must be a primary hypothesis",
            code="family_omnibus_hypothesis_role_mismatch",
            field="role",
            remediation="Declare the omnibus as primary with no hierarchy parent",
        )
    if declaration.prefilter_status is not HypothesisPrefilterStatus.INCLUDED:
        raise ContractError(
            "Prefiltered hypothesis cannot produce an effect target",
            code="filtered_family_effect_hypothesis",
            field="prefilter_status",
            remediation="Retain the hypothesis as typed not-estimable coverage",
        )
    return FrozenFamilyEffectTarget._from_declaration(universe, declaration)


def _require_effect_target(target: FamilyEffectTargetLike) -> None:
    if isinstance(target, FrozenFamilyEffectTarget):
        target._require_intact()
    elif not isinstance(target, FamilyEffectTarget):
        raise TypeError("target must be a legacy or frozen family-effect target")


def _matched_chains(
    artifacts: CrossFitArtifacts,
    target: FamilyEffectTargetLike,
) -> tuple[
    tuple[str, FamilyCommonScoringFunctional, FamilyCommonScoringApplication], ...
]:
    _require_effect_target(target)
    artifacts._require_intact()
    chains = []
    for fold in artifacts.folds:
        matched = [
            (functional, application)
            for functional, application in zip(
                fold.family_common_functionals,
                fold.family_common_applications,
                strict=True,
            )
            if functional.contrast_name == target.contrast_name
            and functional.receiver == target.receiver
            and target.family_id in functional.family_ids
        ]
        if len(matched) != 1:
            raise ContractError(
                "Family effect target must resolve to one chain in every fold",
                code="family_effect_target_chain_mismatch",
                field="contrast_name,receiver,family_id",
                remediation="Use a target present in every cross-fit fold",
            )
        functional, application = matched[0]
        functional._require_intact()
        application._require_intact()
        chains.append((fold.fold_id, functional, application))
    if not chains:
        raise ContractError(
            "Cross-fit result contains no family-common scoring chains",
            code="family_effect_stage_not_connected",
            field="family_common_applications",
            remediation="Run the complete family-common scoring stage",
        )
    return tuple(chains)


def _parent_bound_effect_chains(
    artifacts: CrossFitArtifacts,
    target: FamilyEffectTargetLike,
    score_target: FrozenFamilyEffectTarget | None,
) -> tuple[
    tuple[
        str,
        FamilyCommonScoringFunctional,
        FamilyCommonScoringFunctional,
        FamilyCommonScoringApplication,
    ],
    ...,
]:
    """Bind child contrast weights to its primary parent's score application."""

    contrast_chains = _matched_chains(artifacts, target)
    if score_target is None:
        if (
            isinstance(target, FrozenFamilyEffectTarget)
            and target.role is HypothesisRole.SECONDARY
        ):
            raise ContractError(
                "Frozen secondary effects require their primary score target",
                code="family_effect_secondary_parent_score_target_missing",
                field="score_target",
                remediation=(
                    "Pass the exact primary omnibus target from the frozen universe"
                ),
            )
        return tuple(
            (fold_id, functional, functional, application)
            for fold_id, functional, application in contrast_chains
        )
    if not isinstance(score_target, FrozenFamilyEffectTarget):
        raise TypeError("score_target must be a frozen primary target or None")
    if not isinstance(target, FrozenFamilyEffectTarget):
        raise ContractError(
            "Parent-bound post-hoc scoring requires a frozen child target",
            code="family_effect_parent_binding_requires_frozen_target",
            field="target",
            remediation="Build both targets from the exact frozen universe",
        )
    target._require_intact()
    score_target._require_intact()
    if (
        target._universe is not score_target._universe
        or target.universe_id != score_target.universe_id
        or target.role is not HypothesisRole.SECONDARY
        or target.endpoint != _FROZEN_EFFECT_ENDPOINT
        or score_target.role is not HypothesisRole.PRIMARY
        or score_target.endpoint != _FROZEN_OMNIBUS_ENDPOINT
        or target.parent_key != score_target.hypothesis_key
        or (
            target.receiver,
            target.family_id,
            target.mode,
        )
        != (
            score_target.receiver,
            score_target.family_id,
            score_target.mode,
        )
    ):
        raise ContractError(
            "Post-hoc effect score target is not its exact primary omnibus parent",
            code="family_effect_parent_target_mismatch",
            field="universe_id,parent_key,endpoint,role,receiver,family_id,mode",
            remediation=(
                "Resolve the primary parent and secondary child from one intact "
                "frozen universe"
            ),
        )
    score_chains = _matched_chains(artifacts, score_target)
    values: list[
        tuple[
            str,
            FamilyCommonScoringFunctional,
            FamilyCommonScoringFunctional,
            FamilyCommonScoringApplication,
        ]
    ] = []
    for contrast_chain, score_chain in zip(
        contrast_chains,
        score_chains,
        strict=True,
    ):
        contrast_fold, contrast_functional, _ = contrast_chain
        score_fold, score_functional, score_application = score_chain
        if contrast_fold != score_fold:
            raise ContractError(
                "Parent and child family-effect chains differ by fold",
                code="family_effect_parent_child_fold_mismatch",
                field="fold_id",
                remediation="Use one intact cross-fit artifact for both targets",
            )
        child_contexts = tuple(sorted(contrast_functional.context_ids))
        parent_contexts = tuple(sorted(score_functional.context_ids))
        if child_contexts != parent_contexts:
            raise ContractError(
                "Parent and child family-effect context universes differ",
                code="family_effect_parent_child_context_universe_mismatch",
                field="context_ids",
                remediation=(
                    "Freeze post-hoc contrasts over the primary omnibus context "
                    "universe"
                ),
            )
        values.append(
            (
                contrast_fold,
                contrast_functional,
                score_functional,
                score_application,
            )
        )
    return tuple(values)


def build_family_effect_oof_spec(
    artifacts: CrossFitArtifacts,
    target: FamilyEffectTargetLike,
    *,
    minimum_clusters_for_diagnostic_se: int = 8,
) -> OOFEffectSpec:
    """Derive exact context-ID contrast weights from intact fold functionals."""

    if not isinstance(artifacts, CrossFitArtifacts):
        raise TypeError("artifacts must be CrossFitArtifacts")
    _require_effect_target(target)
    observed: set[tuple[tuple[str, float], ...]] = set()
    for _, functional, _ in _matched_chains(artifacts, target):
        contrast = functional.sender_functional.contrast
        context_by_node = dict(functional.sender_functional.contrast_context_ids)
        weights = tuple(
            sorted(
                (context_by_node[node], float(weight))
                for node, weight in contrast.weights.items()
            )
        )
        if set(context_by_node) != set(contrast.weights):
            raise ContractError(
                "Functional context manifest differs from its contrast",
                code="family_effect_context_manifest_mismatch",
                field="contrast_context_ids",
                remediation="Use intact common-sender functional provenance",
            )
        observed.add(weights)
    if len(observed) != 1:
        raise ContractError(
            "Family effect contrast weights differ across folds",
            code="family_effect_fold_contrast_mismatch",
            field="contrast_weights",
            remediation="Use one pre-registered contrast across all folds",
        )
    return OOFEffectSpec(
        hypothesis_id=target.hypothesis_id,
        contrast_name=target.contrast_name,
        contrast_weights=next(iter(observed)),
        minimum_clusters_for_diagnostic_se=minimum_clusters_for_diagnostic_se,
    )


def family_effect_oof_score_table(
    artifacts: CrossFitArtifacts,
    target: FamilyEffectTargetLike,
    effect_spec: OOFEffectSpec,
    *,
    score_target: FrozenFamilyEffectTarget | None = None,
) -> pd.DataFrame:
    """Extract one strict integrated family score per held-out sample.

    For a frozen secondary, ``score_target`` binds values to the exact primary
    omnibus application. The secondary functional contributes contrast weights
    and context identity only.
    """

    if not isinstance(effect_spec, OOFEffectSpec):
        raise TypeError("effect_spec must be OOFEffectSpec")
    _require_effect_target(target)
    if (
        effect_spec.hypothesis_id != target.hypothesis_id
        or effect_spec.contrast_name != target.contrast_name
    ):
        raise ContractError(
            "OOF effect specification does not belong to the family target",
            code="family_effect_spec_target_mismatch",
            field="hypothesis_id,contrast_name",
            remediation="Build the effect spec from the exact family target",
        )
    rows: list[dict[str, object]] = []
    for (
        fold_id,
        contrast_functional,
        score_functional,
        application,
    ) in _parent_bound_effect_chains(artifacts, target, score_target):
        if set(contrast_functional.context_ids) != set(effect_spec.context_ids):
            raise ContractError(
                "OOF effect contexts differ from the scoring functional",
                code="family_effect_context_universe_mismatch",
                field="context_ids",
                remediation="Use context IDs derived from the fold functionals",
            )
        selected = application.family_scores.loc[
            lambda table: (
                table["family_id"].eq(target.family_id)
                & table["mode"].eq(target.mode.value)
            )
        ].copy()
        if selected.empty or selected["sample_id"].duplicated().any():
            raise ContractError(
                "Family score target requires exactly one row per held-out sample",
                code="family_effect_sample_grain_mismatch",
                field="sample_id",
                remediation="Select one family, mode, receiver, and contrast",
            )
        for source in selected.itertuples(index=False):
            source_status = str(source.status)
            if source_status == "ok":
                value = float(cast(Any, source.integrated_lr_score))
                if not np.isfinite(value):
                    raise ContractError(
                        "Observed integrated family score must be finite",
                        code="family_effect_invalid_observed_score",
                        field=_SCORE_COLUMN,
                        remediation="Preserve the intact family scoring application",
                    )
                score_status = "observed"
            elif source_status == "structural_zero":
                value = float(cast(Any, source.integrated_lr_score))
                if value != 0.0:
                    raise ContractError(
                        "Structural-zero integrated family score must equal zero",
                        code="family_effect_invalid_structural_zero",
                        field=_SCORE_COLUMN,
                        remediation="Preserve structural-zero score semantics",
                    )
                score_status = "structural_zero"
            else:
                value = np.nan
                score_status = "not_estimable"
            rows.append(
                {
                    "subject_id": str(source.subject_id),
                    "sample_id": str(source.sample_id),
                    "fold_id": fold_id,
                    "context_id": str(source.context_id),
                    "score": value,
                    "score_status": score_status,
                    "scoring_function_id": (
                        score_functional.family_common_functional_id
                    ),
                }
            )
    result = pd.DataFrame(rows, columns=_OOF_COLUMNS).sort_values(
        ["fold_id", "subject_id", "context_id", "sample_id"],
        kind="stable",
        ignore_index=True,
    )
    if result["sample_id"].duplicated().any():
        raise ContractError(
            "Held-out sample appears in more than one cross-fit family score",
            code="family_effect_oof_sample_leakage",
            field="sample_id",
            remediation="Repair fold/application provenance",
        )
    return result


def fit_crossfit_family_effect(
    artifacts: CrossFitArtifacts,
    target: FamilyEffectTargetLike,
    effect_spec: OOFEffectSpec,
    *,
    score_target: FrozenFamilyEffectTarget | None = None,
) -> OOFContextEffectResult:
    """Fit the unpenalized OOF effect from strict family-common sample scores."""

    return fit_oof_context_effect(
        family_effect_oof_score_table(
            artifacts,
            target,
            effect_spec,
            score_target=score_target,
        ),
        effect_spec,
    )


def _failure_reason(error: Exception) -> str:
    if isinstance(error, CrychicError):
        reason_code: str = error.details.code
        return reason_code
    return f"family_effect_adapter_failed_{type(error).__qualname__}"


def full_pipeline_family_effect_records(
    resampling: FullPipelineResamplingResult,
    target: FamilyEffectTargetLike,
    effect_spec: OOFEffectSpec,
    *,
    score_target: FrozenFamilyEffectTarget | None = None,
) -> tuple[FullPipelineResampleEffectRecord, ...]:
    """Refit one effect for every retained full-pipeline cross-fit child."""

    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    _require_effect_target(target)
    if (
        effect_spec.hypothesis_id != target.hypothesis_id
        or effect_spec.contrast_name != target.contrast_name
    ):
        raise ContractError(
            "OOF effect specification does not belong to the family target",
            code="family_effect_spec_target_mismatch",
            field="hypothesis_id,contrast_name",
            remediation="Build the effect spec from the exact family target",
        )
    if not resampling.retain_children:
        raise ContractError(
            "Effect adaptation requires retained full-pipeline children",
            code="full_pipeline_effect_children_not_retained",
            field="retain_children",
            remediation="Rerun resampling with retain_children=True",
        )
    values: list[FullPipelineResampleEffectRecord] = []
    for _, workflow_record in aligned_full_pipeline_resamples(resampling):
        kind = FullPipelineEffectResamplingKind(workflow_record.operation.value)
        if workflow_record.status is FullPipelineResampleStatus.FAILED:
            values.append(
                FullPipelineResampleEffectRecord(
                    full_pipeline_record_id=workflow_record.record_id,
                    plan_id=workflow_record.plan_id,
                    crossfit_id=None,
                    resampling_kind=kind,
                    resample_index=workflow_record.resample_index,
                    effect_spec_id=effect_spec.spec_id,
                    hypothesis_id=effect_spec.hypothesis_id,
                    effect_result_id=None,
                    effect=None,
                    status=FullPipelineResampleEffectStatus.FAILED,
                    reason_code=workflow_record.failure_code,
                )
            )
            continue
        child = workflow_record.child
        if child is None:
            raise ContractError(
                "Successful resample is missing its retained cross-fit child",
                code="full_pipeline_effect_child_missing",
                field="record_id",
                remediation="Reject the result and rerun with child retention",
            )
        if child.crossfit_id != workflow_record.crossfit_id:
            raise ContractError(
                "Retained cross-fit child differs from its workflow record",
                code="full_pipeline_effect_child_provenance_mismatch",
                field="crossfit_id",
                remediation="Reject the corrupted resampling result",
            )
        if child.spec.spec_id != resampling.crossfit_spec_id:
            raise ContractError(
                "Retained child uses a different cross-fit specification",
                code="full_pipeline_effect_child_spec_mismatch",
                field="crossfit_spec_id",
                remediation="Use children produced by the frozen resampling spec",
            )
        receiver_support = tuple(
            support
            for fold in child.folds
            for support in fold.receiver_training_support
            if support.receiver_id == target.receiver
        )
        if len(receiver_support) != len(child.folds):
            values.append(
                FullPipelineResampleEffectRecord(
                    full_pipeline_record_id=workflow_record.record_id,
                    plan_id=workflow_record.plan_id,
                    crossfit_id=child.crossfit_id,
                    resampling_kind=kind,
                    resample_index=workflow_record.resample_index,
                    effect_spec_id=effect_spec.spec_id,
                    hypothesis_id=effect_spec.hypothesis_id,
                    effect_result_id=None,
                    effect=None,
                    status=FullPipelineResampleEffectStatus.FAILED,
                    reason_code="family_effect_receiver_support_incomplete",
                )
            )
            continue
        absent_support = tuple(
            support
            for support in receiver_support
            if support.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE
        )
        if absent_support:
            reasons = {support.reason_code for support in absent_support}
            values.append(
                FullPipelineResampleEffectRecord(
                    full_pipeline_record_id=workflow_record.record_id,
                    plan_id=workflow_record.plan_id,
                    crossfit_id=child.crossfit_id,
                    resampling_kind=kind,
                    resample_index=workflow_record.resample_index,
                    effect_spec_id=effect_spec.spec_id,
                    hypothesis_id=effect_spec.hypothesis_id,
                    effect_result_id=None,
                    effect=None,
                    status=(
                        FullPipelineResampleEffectStatus.NOT_ESTIMABLE
                        if reasons == {"receiver_absent_in_outer_training"}
                        else FullPipelineResampleEffectStatus.FAILED
                    ),
                    reason_code=(
                        "receiver_absent_in_outer_training"
                        if reasons == {"receiver_absent_in_outer_training"}
                        else "family_effect_receiver_support_invalid"
                    ),
                )
            )
            continue
        try:
            effect = fit_crossfit_family_effect(
                child,
                target,
                effect_spec,
                score_target=score_target,
            )
            observed = effect.effect_status == "observed"
            values.append(
                FullPipelineResampleEffectRecord(
                    full_pipeline_record_id=workflow_record.record_id,
                    plan_id=workflow_record.plan_id,
                    crossfit_id=child.crossfit_id,
                    resampling_kind=kind,
                    resample_index=workflow_record.resample_index,
                    effect_spec_id=effect_spec.spec_id,
                    hypothesis_id=effect_spec.hypothesis_id,
                    effect_result_id=effect.result_id,
                    effect=effect.effect if observed else None,
                    status=(
                        FullPipelineResampleEffectStatus.OBSERVED
                        if observed
                        else FullPipelineResampleEffectStatus.NOT_ESTIMABLE
                    ),
                    reason_code=None if observed else effect.reason_code,
                )
            )
        except Exception as error:
            values.append(
                FullPipelineResampleEffectRecord(
                    full_pipeline_record_id=workflow_record.record_id,
                    plan_id=workflow_record.plan_id,
                    crossfit_id=child.crossfit_id,
                    resampling_kind=kind,
                    resample_index=workflow_record.resample_index,
                    effect_spec_id=effect_spec.spec_id,
                    hypothesis_id=effect_spec.hypothesis_id,
                    effect_result_id=None,
                    effect=None,
                    status=FullPipelineResampleEffectStatus.FAILED,
                    reason_code=_failure_reason(error),
                )
            )
    return tuple(values)


def summarize_family_effect_full_pipeline(
    point_artifacts: CrossFitArtifacts,
    resampling: FullPipelineResamplingResult,
    target: FamilyEffectTargetLike,
    distribution_spec: FullPipelineEffectDistributionSpec,
    *,
    calibration_gate: G3FrequencyCalibrationGate | None = None,
    score_target: FrozenFamilyEffectTarget | None = None,
) -> FullPipelineEffectDistribution:
    """Fit point/resampled effects and summarize them behind the G3-F gate."""

    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    _require_effect_target(target)
    if not isinstance(distribution_spec, FullPipelineEffectDistributionSpec):
        raise TypeError("distribution_spec must be FullPipelineEffectDistributionSpec")
    if calibration_gate is not None:
        if not isinstance(calibration_gate, G3FrequencyCalibrationGate):
            raise TypeError("calibration_gate must be producer-owned G3 gate or None")
        calibration_gate._require_intact()
        if (
            isinstance(target, FrozenFamilyEffectTarget)
            and calibration_gate.hypothesis_universe_id is not None
            and calibration_gate.hypothesis_universe_id != target.universe_id
        ):
            raise ContractError(
                "G3-F gate is bound to a different frozen hypothesis universe",
                code="family_effect_gate_universe_mismatch",
                field="hypothesis_universe_id",
                remediation="Use calibration evidence for this exact universe",
            )
    if point_artifacts.spec.spec_id != resampling.crossfit_spec_id:
        raise ContractError(
            "Point and resampling runs use different cross-fit specifications",
            code="full_pipeline_effect_crossfit_spec_mismatch",
            field="crossfit_spec_id",
            remediation="Use one frozen CrossFitSpec for point and resampled runs",
        )
    if (
        isinstance(target, FamilyEffectTarget)
        and isinstance(calibration_gate, G3FrequencyCalibrationGate)
        and calibration_gate.formal_release_allowed
    ):
        raise ContractError(
            "A passed G3-F gate requires a frozen-universe effect target",
            code="legacy_family_effect_target_formal_release_forbidden",
            field="target",
            remediation=(
                "Build the target from an included declaration in a frozen "
                "hypothesis universe"
            ),
        )
    if (
        distribution_spec.effect_spec.hypothesis_id != target.hypothesis_id
        or distribution_spec.effect_spec.contrast_name != target.contrast_name
    ):
        raise ContractError(
            "Distribution specification does not belong to the family target",
            code="family_effect_distribution_target_mismatch",
            field="hypothesis_id,contrast_name",
            remediation="Build the distribution spec from this target's OOF spec",
        )
    point_effect = fit_crossfit_family_effect(
        point_artifacts,
        target,
        distribution_spec.effect_spec,
        score_target=score_target,
    )
    records = full_pipeline_family_effect_records(
        resampling,
        target,
        distribution_spec.effect_spec,
        score_target=score_target,
    )
    return summarize_full_pipeline_effect_distribution(
        point_effect,
        distribution_spec,
        records,
        calibration_gate=calibration_gate,
    )


__all__ = [
    "FamilyEffectTarget",
    "FrozenFamilyEffectTarget",
    "build_family_effect_oof_spec",
    "build_frozen_family_effect_target",
    "family_effect_oof_score_table",
    "fit_crossfit_family_effect",
    "full_pipeline_family_effect_records",
    "summarize_family_effect_full_pipeline",
]
