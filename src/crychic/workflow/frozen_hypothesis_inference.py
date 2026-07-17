"""Complete frozen-universe orchestration for empirical hierarchical inference."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from crychic.core import (
    CommunicationMode,
    ContractError,
    CrychicError,
    stable_id,
)
from crychic.inference.full_pipeline import (
    FullPipelineEffectDistribution,
    FullPipelineEffectDistributionSpec,
    G3FrequencyCalibrationGate,
)
from crychic.inference.hierarchical import (
    FrozenHierarchicalFDRCollection,
    FrozenHierarchicalFDRSpec,
    HypothesisPValueRecord,
    evaluate_hierarchical_fdr,
    freeze_hierarchical_fdr_spec,
)
from crychic.inference.hypotheses import (
    FrozenHypothesisCoverage,
    FrozenHypothesisUniverse,
    HypothesisCoverageRecord,
    HypothesisCoverageStatus,
    HypothesisDeclaration,
    HypothesisPrefilterStatus,
    HypothesisRole,
    validate_frozen_hypothesis_coverage,
)
from crychic.inference.omnibus import FullPipelineOmnibusDistribution
from crychic.resampling import ContextPermutationPlan, SubjectBootstrapPlan

from .crossfit import CrossFitArtifacts
from .full_pipeline_effects import (
    FrozenFamilyEffectTarget,
    build_family_effect_oof_spec,
    build_frozen_family_effect_target,
    summarize_family_effect_full_pipeline,
)
from .full_pipeline_omnibus import (
    build_family_omnibus_spec,
    summarize_family_omnibus_full_pipeline,
)
from .full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
)

_SCHEMA_VERSION = "1.0.0"
_PRIMARY_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_SECONDARY_ENDPOINT = "family_common_integrated_lr_context_effect_v1"
_MINIMUM_CONTEXT_PERMUTATIONS = 1_000
_COLLECTION_MARKER = "crychic.workflow.frozen_hypothesis_inference.v1"
_RECORD_MARKER = "crychic.workflow.frozen_hypothesis_inference_record.v1"
_PLAN_SET_MISMATCH = "frozen_hypothesis_context_permutation_plan_set_mismatch"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_name(value: object | None, *, field_name: str) -> str | None:
    return None if value is None else _name(value, field_name=field_name)


def _candidate_p(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("candidate p-value must be numeric, not boolean")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("candidate p-value must be finite and in [0, 1]") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("candidate p-value must be finite and in [0, 1]")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenHypothesisInferenceInput:
    """Point/resampling parents and a preregistered secondary effect spec."""

    hypothesis_id: str
    point_artifacts: CrossFitArtifacts = field(repr=False)
    resampling: FullPipelineResamplingResult = field(repr=False)
    secondary_distribution_spec: FullPipelineEffectDistributionSpec | None = None
    point_crossfit_id: str = field(init=False)
    resampling_result_id: str = field(init=False)
    input_id: str = field(init=False)

    def __post_init__(self) -> None:
        hypothesis_id = _name(self.hypothesis_id, field_name="hypothesis_id")
        if not isinstance(self.point_artifacts, CrossFitArtifacts):
            raise TypeError("point_artifacts must be CrossFitArtifacts")
        if not isinstance(self.resampling, FullPipelineResamplingResult):
            raise TypeError("resampling must be FullPipelineResamplingResult")
        self.point_artifacts._require_intact()
        self.resampling._require_intact()
        point_crossfit_id = _name(
            self.point_artifacts.crossfit_id,
            field_name="point_crossfit_id",
        )
        resampling_result_id = _name(
            self.resampling.result_id,
            field_name="resampling_result_id",
        )
        if self.secondary_distribution_spec is not None:
            if not isinstance(
                self.secondary_distribution_spec,
                FullPipelineEffectDistributionSpec,
            ):
                raise TypeError(
                    "secondary_distribution_spec must be a typed distribution "
                    "spec or None"
                )
            self.secondary_distribution_spec._require_intact()
        object.__setattr__(self, "hypothesis_id", hypothesis_id)
        object.__setattr__(self, "point_crossfit_id", point_crossfit_id)
        object.__setattr__(self, "resampling_result_id", resampling_result_id)
        object.__setattr__(
            self,
            "input_id",
            stable_id(
                "frozen_hypothesis_inference_input",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "point_crossfit_id": self.point_crossfit_id,
            "resampling_result_id": self.resampling_result_id,
            "secondary_distribution_spec_id": (
                None
                if self.secondary_distribution_spec is None
                else self.secondary_distribution_spec.spec_id
            ),
        }

    def _require_intact(self) -> None:
        try:
            self.point_artifacts._require_intact()
            self.resampling._require_intact()
            if self.secondary_distribution_spec is not None:
                self.secondary_distribution_spec._require_intact()
            expected_id = stable_id(
                "frozen_hypothesis_inference_input",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self.point_crossfit_id == self.point_artifacts.crossfit_id
                and self.resampling_result_id == self.resampling.result_id
                and self.input_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen hypothesis inference input failed integrity validation",
                code="frozen_hypothesis_inference_input_integrity_violation",
                field="input_id",
                remediation=(
                    "Recreate the input from intact point and resampling parents"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Frozen hypothesis inference input failed integrity validation",
                code="frozen_hypothesis_inference_input_integrity_violation",
                field="input_id",
                remediation=(
                    "Recreate the input from intact point and resampling parents"
                ),
            )


@dataclass(frozen=True, slots=True)
class _PermutationLineage:
    plan_ids: tuple[str, ...]
    full_pipeline_record_ids: tuple[str, ...]


def _context_permutation_lineage(
    resampling: FullPipelineResamplingResult,
) -> _PermutationLineage:
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    resampling._require_intact()
    if not resampling.retain_children:
        raise ContractError(
            "Frozen inference requires retained full-pipeline children",
            code="frozen_hypothesis_resampling_children_not_retained",
            field="retain_children",
            remediation="Rerun every context permutation with retain_children=True",
        )
    if len(resampling.plans) != len(resampling.records):
        raise ContractError(
            "Full-pipeline plans and records are not exactly aligned",
            code="frozen_hypothesis_resampling_plan_alignment_mismatch",
            field="plans,records",
            remediation="Use one intact full-pipeline resampling result",
        )
    exchangeability_id = resampling.exchangeability.exchangeability_id
    selected: list[tuple[int, str, str]] = []
    for plan, record in zip(resampling.plans, resampling.records, strict=True):
        if not isinstance(record, FullPipelineResampleRecord):
            raise TypeError("resampling records must be FullPipelineResampleRecord")
        if isinstance(plan, ContextPermutationPlan):
            operation = FullPipelineResamplingOperation.CONTEXT_PERMUTATION
            plan_id = plan.permutation_id
        elif isinstance(plan, SubjectBootstrapPlan):
            operation = FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP
            plan_id = plan.bootstrap_id
        else:
            raise TypeError("resampling contains an unsupported plan")
        if (
            record.operation is not operation
            or record.plan_id != plan_id
            or record.resample_index != plan.resample_index
            or plan.exchangeability_id != exchangeability_id
            or record.exchangeability_id != exchangeability_id
            or record.plan_seed_lineage != plan.seed_lineage
        ):
            raise ContractError(
                "Full-pipeline plan and execution lineage are misaligned",
                code="frozen_hypothesis_resampling_plan_alignment_mismatch",
                field=("plan_id,resample_index,exchangeability_id,seed_lineage"),
                remediation="Use the intact plan and execution record pairs",
            )
        if isinstance(plan, ContextPermutationPlan):
            selected.append((plan.resample_index, plan_id, record.record_id))
    if not selected:
        raise ContractError(
            "Frozen inference requires context-permutation plans",
            code="frozen_hypothesis_context_permutations_missing",
            field="operation",
            remediation="Run the preregistered context permutations",
        )
    ordered = tuple(sorted(selected))
    for field_name, values in (
        ("resample_index", tuple(item[0] for item in ordered)),
        ("plan_id", tuple(item[1] for item in ordered)),
        ("record_id", tuple(item[2] for item in ordered)),
    ):
        if len(values) != len(set(values)):
            raise ContractError(
                "Context-permutation lineage contains duplicate values",
                code="duplicate_frozen_hypothesis_permutation_lineage",
                field=field_name,
                remediation="Retain each preregistered permutation exactly once",
            )
    return _PermutationLineage(
        plan_ids=tuple(item[1] for item in ordered),
        full_pipeline_record_ids=tuple(item[2] for item in ordered),
    )


def _require_v1_universe(universe: FrozenHypothesisUniverse) -> None:
    if not isinstance(universe, FrozenHypothesisUniverse):
        raise TypeError("universe must be a FrozenHypothesisUniverse")
    universe._require_intact()
    for declaration in universe.declarations:
        expected_endpoint = (
            _PRIMARY_ENDPOINT
            if declaration.role is HypothesisRole.PRIMARY
            else _SECONDARY_ENDPOINT
        )
        if declaration.mode is not CommunicationMode.STATE:
            raise ContractError(
                "The frozen v1 hierarchy accepts state mode only",
                code="frozen_hypothesis_v1_mode_mismatch",
                field="mode",
                remediation="Exclude ecosystem hypotheses from the v1 hierarchy",
            )
        if declaration.endpoint != expected_endpoint:
            raise ContractError(
                "Frozen hypothesis role uses an unsupported v1 endpoint",
                code="frozen_hypothesis_v1_endpoint_mismatch",
                field="endpoint,role",
                remediation=(
                    "Use the primary family omnibus and secondary family-common "
                    "context-effect endpoints"
                ),
            )


def _validated_inputs(
    universe: FrozenHypothesisUniverse,
    inputs: Sequence[FrozenHypothesisInferenceInput],
) -> dict[str, FrozenHypothesisInferenceInput]:
    supplied = tuple(inputs)
    if any(not isinstance(item, FrozenHypothesisInferenceInput) for item in supplied):
        raise TypeError("inputs must contain FrozenHypothesisInferenceInput values")
    for item in supplied:
        item._require_intact()
    identifiers = tuple(item.hypothesis_id for item in supplied)
    if len(identifiers) != len(set(identifiers)):
        raise ContractError(
            "Frozen inference inputs contain duplicate hypotheses",
            code="duplicate_frozen_hypothesis_inference_input",
            field="hypothesis_id",
            remediation="Provide one point/resampling input per included hypothesis",
        )
    expected = {
        declaration.hypothesis_id
        for declaration in universe.declarations
        if declaration.prefilter_status is HypothesisPrefilterStatus.INCLUDED
    }
    observed = set(identifiers)
    if observed != expected:
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        raise ContractError(
            "Frozen inference inputs do not exactly cover included hypotheses",
            code=(
                "missing_frozen_hypothesis_inference_input"
                if missing and not extra
                else (
                    "extra_frozen_hypothesis_inference_input"
                    if extra and not missing
                    else "frozen_hypothesis_inference_input_universe_mismatch"
                )
            ),
            field="hypothesis_id",
            remediation=(
                "Provide exactly one input for every included declaration and none "
                "for prefiltered declarations"
            ),
        )
    point_ids = {item.point_crossfit_id for item in supplied}
    resampling_ids = {item.resampling_result_id for item in supplied}
    if len(point_ids) > 1 or len(resampling_ids) > 1:
        raise ContractError(
            "Included hypotheses do not share exact point and resampling parents",
            code="frozen_hypothesis_inference_parent_run_mismatch",
            field="point_crossfit_id,resampling_result_id",
            remediation=(
                "Use one complete point cross-fit and one full-pipeline resampling "
                "result for the entire frozen universe"
            ),
        )
    by_id = {item.hypothesis_id: item for item in supplied}
    for declaration in universe.declarations:
        if declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED:
            continue
        item = by_id[declaration.hypothesis_id]
        distribution_spec = item.secondary_distribution_spec
        if declaration.role is HypothesisRole.PRIMARY:
            if distribution_spec is not None:
                raise ContractError(
                    "Primary omnibus input cannot carry a secondary effect spec",
                    code="primary_hypothesis_secondary_spec_forbidden",
                    field="secondary_distribution_spec",
                    remediation="Let the primary omnibus adapter build its exact spec",
                )
            continue
        if distribution_spec is None:
            raise ContractError(
                "Secondary hypothesis requires a preregistered effect "
                "distribution spec",
                code="secondary_hypothesis_distribution_spec_missing",
                field="secondary_distribution_spec",
                remediation=(
                    "Freeze the contrast, direction, and minimum effect before "
                    "resampling"
                ),
            )
        effect_spec = distribution_spec.effect_spec
        if (
            effect_spec.hypothesis_id != declaration.hypothesis_id
            or effect_spec.contrast_name != declaration.contrast_name
        ):
            raise ContractError(
                "Secondary effect spec does not bind its exact declaration",
                code="secondary_hypothesis_distribution_spec_mismatch",
                field="hypothesis_id,contrast_name",
                remediation="Build the spec for this exact frozen secondary",
            )
    return by_id


@dataclass(frozen=True, slots=True)
class _CandidateResult:
    target: FrozenFamilyEffectTarget
    score_target: FrozenFamilyEffectTarget
    source_result_id: str
    status: HypothesisCoverageStatus
    reason_code: str | None
    candidate_p_value: float | None
    point_crossfit_id: str
    resampling_result_id: str
    context_permutation_plan_ids: tuple[str, ...]
    context_permutation_record_ids: tuple[str, ...]


def _require_secondary_spec_matches_artifacts(
    item: FrozenHypothesisInferenceInput,
    target: FrozenFamilyEffectTarget,
) -> FullPipelineEffectDistributionSpec:
    supplied = item.secondary_distribution_spec
    assert supplied is not None
    supplied.effect_spec._require_intact()
    repeated = FullPipelineEffectDistributionSpec(
        effect_spec=supplied.effect_spec,
        minimum_effect=supplied.minimum_effect,
        specificity_direction=supplied.specificity_direction,
    )
    if repeated.spec_id != supplied.spec_id:
        raise ContractError(
            "Secondary distribution spec failed integrity validation",
            code="secondary_hypothesis_distribution_spec_integrity_violation",
            field="spec_id",
            remediation="Recreate the immutable effect distribution spec",
        )
    expected_effect_spec = build_family_effect_oof_spec(
        item.point_artifacts,
        target,
    )
    if expected_effect_spec.spec_id != supplied.effect_spec.spec_id:
        raise ContractError(
            "Secondary effect spec differs from its exact OOF contrast functional",
            code="secondary_hypothesis_effect_spec_artifact_mismatch",
            field="effect_spec_id",
            remediation=(
                "Build the preregistered effect spec from these exact point "
                "artifacts and frozen target"
            ),
        )
    return supplied


def _primary_candidate(
    item: FrozenHypothesisInferenceInput,
    target: FrozenFamilyEffectTarget,
    lineage: _PermutationLineage,
) -> _CandidateResult:
    spec = build_family_omnibus_spec(item.point_artifacts, target)
    distribution = summarize_family_omnibus_full_pipeline(
        item.point_artifacts,
        item.resampling,
        target,
        spec,
    )
    if not isinstance(distribution, FullPipelineOmnibusDistribution):
        raise TypeError("primary adapter returned an unsupported distribution")
    distribution._require_intact()
    if (
        distribution.hypothesis_id != target.hypothesis_id
        or distribution.omnibus_spec_id != spec.spec_id
        or distribution.permutation_plan_ids != lineage.plan_ids
        or distribution.full_pipeline_record_ids != lineage.full_pipeline_record_ids
    ):
        raise ContractError(
            "Primary omnibus distribution lineage differs from its exact input",
            code="frozen_primary_omnibus_distribution_binding_mismatch",
            field=(
                "hypothesis_id,omnibus_spec_id,permutation_plan_ids,"
                "full_pipeline_record_ids"
            ),
            remediation="Recompute the omnibus from the exact retained children",
        )
    candidate = _candidate_p(distribution.diagnostic_empirical_p)
    complete = (
        distribution.diagnostic_status == "diagnostic_complete_minimum_1000"
        and distribution.n_permutation_total == len(lineage.plan_ids)
        and distribution.n_permutation_observed == len(lineage.plan_ids)
        and len(lineage.plan_ids) >= _MINIMUM_CONTEXT_PERMUTATIONS
        and candidate is not None
    )
    return _CandidateResult(
        target=target,
        score_target=target,
        source_result_id=distribution.distribution_id,
        status=(
            HypothesisCoverageStatus.OBSERVED
            if complete
            else HypothesisCoverageStatus.NOT_ESTIMABLE
        ),
        reason_code=None if complete else distribution.formal_reason_code,
        candidate_p_value=candidate,
        point_crossfit_id=item.point_artifacts.crossfit_id,
        resampling_result_id=item.resampling.result_id,
        context_permutation_plan_ids=lineage.plan_ids,
        context_permutation_record_ids=lineage.full_pipeline_record_ids,
    )


def _secondary_runtime_reason(
    distribution: FullPipelineEffectDistribution,
    *,
    n_plans: int,
) -> str | None:
    if distribution.point_effect is None:
        return "full_pipeline_point_effect_not_observed"
    if distribution.point_n_clusters < distribution.minimum_point_clusters:
        return "full_pipeline_point_clusters_below_minimum"
    if distribution.point_uncertainty_status != "observed_cr2_diagnostic":
        return "full_pipeline_point_cr2_uncertainty_not_estimable"
    if distribution.n_permutation_total != n_plans:
        return "full_pipeline_permutation_plan_count_mismatch"
    if distribution.n_permutation_observed != distribution.n_permutation_total:
        return "full_pipeline_permutation_distribution_incomplete"
    if distribution.n_permutation_observed < _MINIMUM_CONTEXT_PERMUTATIONS:
        return "full_pipeline_permutation_observed_below_1000"
    if distribution.diagnostic_empirical_permutation_p is None:
        return "full_pipeline_runtime_diagnostics_not_estimable"
    return None


def _secondary_candidate(
    item: FrozenHypothesisInferenceInput,
    target: FrozenFamilyEffectTarget,
    parent_target: FrozenFamilyEffectTarget,
    lineage: _PermutationLineage,
) -> _CandidateResult:
    distribution_spec = _require_secondary_spec_matches_artifacts(item, target)
    distribution = summarize_family_effect_full_pipeline(
        item.point_artifacts,
        item.resampling,
        target,
        distribution_spec,
        calibration_gate=None,
        score_target=parent_target,
    )
    if not isinstance(distribution, FullPipelineEffectDistribution):
        raise TypeError("secondary adapter returned an unsupported distribution")
    distribution._require_intact()
    if (
        distribution.hypothesis_id != target.hypothesis_id
        or distribution.effect_spec_id != distribution_spec.effect_spec.spec_id
        or distribution.distribution_spec_id != distribution_spec.spec_id
        or distribution.p_value is not None
        or distribution.q_value is not None
        or distribution.calibration_gate_id is not None
    ):
        raise ContractError(
            "Secondary effect distribution is not exactly bound or is pre-released",
            code="frozen_secondary_effect_distribution_binding_mismatch",
            field=(
                "hypothesis_id,effect_spec_id,distribution_spec_id,"
                "calibration_gate_id,p_value,q_value"
            ),
            remediation=(
                "Recompute the candidate distribution without a per-effect gate"
            ),
        )
    candidate = _candidate_p(distribution.diagnostic_empirical_permutation_p)
    reason = _secondary_runtime_reason(distribution, n_plans=len(lineage.plan_ids))
    if reason is None and candidate is None:
        reason = "full_pipeline_runtime_diagnostics_not_estimable"
    return _CandidateResult(
        target=target,
        score_target=parent_target,
        source_result_id=distribution.distribution_id,
        status=(
            HypothesisCoverageStatus.OBSERVED
            if reason is None
            else HypothesisCoverageStatus.NOT_ESTIMABLE
        ),
        reason_code=reason,
        candidate_p_value=candidate,
        point_crossfit_id=item.point_artifacts.crossfit_id,
        resampling_result_id=item.resampling.result_id,
        context_permutation_plan_ids=lineage.plan_ids,
        context_permutation_record_ids=lineage.full_pipeline_record_ids,
    )


def _failure_reason(error: Exception) -> str:
    if isinstance(error, CrychicError):
        reason: str = error.details.code
        return reason
    return f"frozen_hypothesis_inference_failed_{type(error).__qualname__}"


def _failure_source_id(
    declaration: HypothesisDeclaration,
    item: FrozenHypothesisInferenceInput,
    reason_code: str,
) -> str:
    identifier: str = stable_id(
        "failed_frozen_hypothesis_inference",
        {
                "declaration_id": declaration.declaration_id,
                "point_crossfit_id": item.point_artifacts.crossfit_id,
                "resampling_result_id": item.resampling.result_id,
                "reason_code": reason_code,
            },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


@dataclass(frozen=True, slots=True, init=False)
class FrozenHypothesisInferenceRecord:
    """Producer-owned candidate-p row bound to one exact declaration and target."""

    universe_id: str
    declaration_id: str
    hypothesis_id: str
    endpoint: str
    role: HypothesisRole
    parent_key: str | None
    target_id: str | None
    score_target_id: str | None
    point_crossfit_id: str | None
    resampling_result_id: str | None
    source_result_id: str
    p_value_record_id: str
    status: HypothesisCoverageStatus
    reason_code: str | None
    candidate_p_value: float | None
    context_permutation_plan_ids: tuple[str, ...]
    context_permutation_record_ids: tuple[str, ...]
    record_id: str
    _universe: FrozenHypothesisUniverse
    _declaration: HypothesisDeclaration
    _target: FrozenFamilyEffectTarget | None
    _score_target: FrozenFamilyEffectTarget | None
    _p_value_record: HypothesisPValueRecord
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenHypothesisInferenceRecord is producer-owned; use "
            "run_frozen_hypothesis_inference()"
        )

    @classmethod
    def _from_parts(
        cls,
        universe: FrozenHypothesisUniverse,
        declaration: HypothesisDeclaration,
        p_value_record: HypothesisPValueRecord,
        *,
        candidate: _CandidateResult | None,
    ) -> FrozenHypothesisInferenceRecord:
        universe._require_intact()
        declaration._require_intact()
        p_value_record._require_intact()
        if universe.declaration_for(declaration.hypothesis_id) is not declaration:
            raise ContractError(
                "Inference record declaration is not the exact universe child",
                code="frozen_hypothesis_inference_declaration_mismatch",
                field="declaration_id",
                remediation="Resolve declarations from the exact frozen universe",
            )
        filtered = declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED
        if filtered != (candidate is None):
            raise ContractError(
                "Candidate presence differs from the frozen prefilter decision",
                code="frozen_hypothesis_inference_prefilter_mismatch",
                field="prefilter_status",
                remediation="Do not fit filtered rows or omit included rows",
            )
        target = None if candidate is None else candidate.target
        score_target = None if candidate is None else candidate.score_target
        values: dict[str, object] = {
            "universe_id": universe.universe_id,
            "declaration_id": declaration.declaration_id,
            "hypothesis_id": declaration.hypothesis_id,
            "endpoint": declaration.endpoint,
            "role": declaration.role,
            "parent_key": declaration.parent_key,
            "target_id": None if target is None else target.target_id,
            "score_target_id": (
                None if score_target is None else score_target.target_id
            ),
            "point_crossfit_id": (
                None if candidate is None else candidate.point_crossfit_id
            ),
            "resampling_result_id": (
                None if candidate is None else candidate.resampling_result_id
            ),
            "source_result_id": p_value_record.source_result_id,
            "p_value_record_id": p_value_record.record_id,
            "status": p_value_record.status,
            "reason_code": p_value_record.reason_code,
            "candidate_p_value": (
                None if candidate is None else candidate.candidate_p_value
            ),
            "context_permutation_plan_ids": (
                () if candidate is None else candidate.context_permutation_plan_ids
            ),
            "context_permutation_record_ids": (
                () if candidate is None else candidate.context_permutation_record_ids
            ),
        }
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_universe", universe)
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_score_target", score_target)
        object.__setattr__(self, "_p_value_record", p_value_record)
        object.__setattr__(self, "_producer_marker", _RECORD_MARKER)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "frozen_hypothesis_inference_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    @property
    def effective_p_value(self) -> float:
        value: float = self._p_value_record.effective_p_value
        return value

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "declaration_id": self.declaration_id,
            "hypothesis_id": self.hypothesis_id,
            "endpoint": self.endpoint,
            "role": self.role.value,
            "parent_key": self.parent_key,
            "target_id": self.target_id,
            "score_target_id": self.score_target_id,
            "point_crossfit_id": self.point_crossfit_id,
            "resampling_result_id": self.resampling_result_id,
            "source_result_id": self.source_result_id,
            "p_value_record_id": self.p_value_record_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "candidate_p_value": self.candidate_p_value,
            "effective_p_value": self.effective_p_value,
            "context_permutation_plan_ids": list(self.context_permutation_plan_ids),
            "context_permutation_record_ids": list(self.context_permutation_record_ids),
        }

    def _require_intact(self) -> None:
        try:
            self._universe._require_intact()
            self._declaration._require_intact()
            self._p_value_record._require_intact()
            resolved = self._universe.declaration_for(self.hypothesis_id)
            candidate_p = _candidate_p(self.candidate_p_value)
            plan_ids = tuple(self.context_permutation_plan_ids)
            record_ids = tuple(self.context_permutation_record_ids)
            filtered = resolved.prefilter_status is HypothesisPrefilterStatus.FILTERED
            targets_valid = False
            if filtered:
                targets_valid = (
                    self._target is None
                    and self._score_target is None
                    and self.target_id is None
                    and self.score_target_id is None
                    and self.point_crossfit_id is None
                    and self.resampling_result_id is None
                    and not plan_ids
                    and not record_ids
                    and candidate_p is None
                    and self.status is HypothesisCoverageStatus.NOT_ESTIMABLE
                    and self.reason_code == resolved.filter_reason_code
                )
            elif self._target is not None and self._score_target is not None:
                self._target._require_intact()
                self._score_target._require_intact()
                primary_target = self._score_target
                targets_valid = (
                    self._target._universe is self._universe
                    and primary_target._universe is self._universe
                    and self._target.hypothesis_id == self.hypothesis_id
                    and self.target_id == self._target.target_id
                    and self.score_target_id == primary_target.target_id
                    and self.point_crossfit_id is not None
                    and self.resampling_result_id is not None
                    and len(plan_ids) == len(record_ids)
                    and len(plan_ids) == len(set(plan_ids))
                    and len(record_ids) == len(set(record_ids))
                    and (
                        self.status is not HypothesisCoverageStatus.OBSERVED
                        or (candidate_p is not None and bool(plan_ids))
                    )
                    and (
                        resolved.role is not HypothesisRole.PRIMARY
                        or primary_target is self._target
                    )
                    and (
                        resolved.role is not HypothesisRole.SECONDARY
                        or (
                            primary_target.role is HypothesisRole.PRIMARY
                            and primary_target.endpoint == _PRIMARY_ENDPOINT
                            and resolved.parent_key == primary_target.hypothesis_key
                            and (
                                resolved.receiver,
                                resolved.family_id,
                                resolved.mode,
                            )
                            == (
                                primary_target.receiver,
                                primary_target.family_id,
                                primary_target.mode,
                            )
                        )
                    )
                )
            expected_id = stable_id(
                "frozen_hypothesis_inference_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _RECORD_MARKER
                and resolved is self._declaration
                and self.universe_id == self._universe.universe_id
                and self.declaration_id == resolved.declaration_id
                and self.endpoint == resolved.endpoint
                and self.role is resolved.role
                and self.parent_key == resolved.parent_key
                and self.hypothesis_id == self._p_value_record.hypothesis_id
                and self.source_result_id == self._p_value_record.source_result_id
                and self.p_value_record_id == self._p_value_record.record_id
                and self.status is self._p_value_record.status
                and self.reason_code == self._p_value_record.reason_code
                and (
                    self.status is not HypothesisCoverageStatus.OBSERVED
                    or candidate_p == self._p_value_record.p_value
                )
                and (
                    self.status is HypothesisCoverageStatus.OBSERVED
                    or self._p_value_record.p_value is None
                )
                and targets_valid
                and self.record_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen hypothesis inference record failed integrity validation",
                code="frozen_hypothesis_inference_record_integrity_violation",
                field="record_id",
                remediation="Recompute the complete frozen inference collection",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen hypothesis inference record failed integrity validation",
                code="frozen_hypothesis_inference_record_integrity_violation",
                field="record_id",
                remediation="Recompute the complete frozen inference collection",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"record_id": self.record_id, **self._identity_payload()}


def _filtered_source_id(declaration: HypothesisDeclaration) -> str:
    identifier: str = stable_id(
        "filtered_frozen_hypothesis_inference",
        {
                "declaration_id": declaration.declaration_id,
                "filter_reason_code": declaration.filter_reason_code,
            },
        schema_version=_SCHEMA_VERSION,
    )
    return identifier


def _p_value_record(
    declaration: HypothesisDeclaration,
    candidate: _CandidateResult | None,
) -> HypothesisPValueRecord:
    if candidate is None:
        return HypothesisPValueRecord(
            hypothesis_id=declaration.hypothesis_id,
            source_result_id=_filtered_source_id(declaration),
            status=HypothesisCoverageStatus.NOT_ESTIMABLE,
            reason_code=declaration.filter_reason_code,
        )
    return HypothesisPValueRecord(
        hypothesis_id=declaration.hypothesis_id,
        source_result_id=candidate.source_result_id,
        status=candidate.status,
        p_value=(
            candidate.candidate_p_value
            if candidate.status is HypothesisCoverageStatus.OBSERVED
            else None
        ),
        reason_code=(
            None
            if candidate.status is HypothesisCoverageStatus.OBSERVED
            else candidate.reason_code
        ),
    )


def _synchronize_observed_plan_sets(
    candidates: dict[str, _CandidateResult],
) -> dict[str, _CandidateResult]:
    observed_plan_sets = {
        candidate.context_permutation_plan_ids
        for candidate in candidates.values()
        if candidate.status is HypothesisCoverageStatus.OBSERVED
    }
    if len(observed_plan_sets) <= 1:
        return candidates
    return {
        hypothesis_id: (
            replace(
                candidate,
                status=HypothesisCoverageStatus.FAILED,
                reason_code=_PLAN_SET_MISMATCH,
            )
            if candidate.status is HypothesisCoverageStatus.OBSERVED
            else candidate
        )
        for hypothesis_id, candidate in candidates.items()
    }


@dataclass(frozen=True, slots=True, init=False)
class FrozenHypothesisInferenceCollection:
    """Producer-owned complete collection of candidates, coverage, and q-values."""

    universe_id: str
    hierarchical_procedure_id: str
    input_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    p_value_record_ids: tuple[str, ...]
    common_context_permutation_plan_ids: tuple[str, ...]
    coverage_id: str
    hierarchical_collection_id: str
    calibration_gate_id: str | None
    records: tuple[FrozenHypothesisInferenceRecord, ...]
    coverage: FrozenHypothesisCoverage
    hierarchical_fdr: FrozenHierarchicalFDRCollection
    collection_id: str
    _universe: FrozenHypothesisUniverse
    _hierarchical_spec: FrozenHierarchicalFDRSpec
    _inputs: tuple[FrozenHypothesisInferenceInput, ...]
    _p_value_records: tuple[HypothesisPValueRecord, ...]
    _calibration_gate: G3FrequencyCalibrationGate | None
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenHypothesisInferenceCollection is producer-owned; use "
            "run_frozen_hypothesis_inference()"
        )

    @classmethod
    def _from_parts(
        cls,
        universe: FrozenHypothesisUniverse,
        hierarchical_spec: FrozenHierarchicalFDRSpec,
        inputs: Sequence[FrozenHypothesisInferenceInput],
        records: Sequence[FrozenHypothesisInferenceRecord],
        p_value_records: Sequence[HypothesisPValueRecord],
        coverage: FrozenHypothesisCoverage,
        hierarchical_fdr: FrozenHierarchicalFDRCollection,
        calibration_gate: G3FrequencyCalibrationGate | None,
    ) -> FrozenHypothesisInferenceCollection:
        ordered_inputs = tuple(sorted(inputs, key=lambda item: item.hypothesis_id))
        ordered_records = tuple(sorted(records, key=lambda item: item.hypothesis_id))
        ordered_p = tuple(sorted(p_value_records, key=lambda item: item.hypothesis_id))
        observed_sets = {
            record.context_permutation_plan_ids
            for record in ordered_records
            if record.status is HypothesisCoverageStatus.OBSERVED
        }
        if len(observed_sets) > 1:
            raise ContractError(
                "Observed hypotheses use different context-permutation plan sets",
                code=_PLAN_SET_MISMATCH,
                field="context_permutation_plan_ids",
                remediation="Rerun every hypothesis with one synchronized plan set",
            )
        common_plan_ids = next(iter(observed_sets), ())
        self = object.__new__(cls)
        values: dict[str, object] = {
            "universe_id": universe.universe_id,
            "hierarchical_procedure_id": hierarchical_spec.procedure_id,
            "input_ids": tuple(item.input_id for item in ordered_inputs),
            "record_ids": tuple(item.record_id for item in ordered_records),
            "p_value_record_ids": tuple(item.record_id for item in ordered_p),
            "common_context_permutation_plan_ids": common_plan_ids,
            "coverage_id": coverage.coverage_id,
            "hierarchical_collection_id": hierarchical_fdr.collection_id,
            "calibration_gate_id": hierarchical_fdr.calibration_gate_id,
            "records": ordered_records,
            "coverage": coverage,
            "hierarchical_fdr": hierarchical_fdr,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_universe", universe)
        object.__setattr__(self, "_hierarchical_spec", hierarchical_spec)
        object.__setattr__(self, "_inputs", ordered_inputs)
        object.__setattr__(self, "_p_value_records", ordered_p)
        object.__setattr__(self, "_calibration_gate", calibration_gate)
        object.__setattr__(self, "_producer_marker", _COLLECTION_MARKER)
        object.__setattr__(
            self,
            "collection_id",
            stable_id(
                "frozen_hypothesis_inference_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    @property
    def q_value_release_allowed(self) -> bool:
        value: bool = self.hierarchical_fdr.q_value_release_allowed
        return value

    @property
    def context_permutation_plan_set_synchronized(self) -> bool:
        return True

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "hierarchical_procedure_id": self.hierarchical_procedure_id,
            "input_ids": list(self.input_ids),
            "record_ids": list(self.record_ids),
            "p_value_record_ids": list(self.p_value_record_ids),
            "common_context_permutation_plan_ids": list(
                self.common_context_permutation_plan_ids
            ),
            "coverage_id": self.coverage_id,
            "hierarchical_collection_id": self.hierarchical_collection_id,
            "calibration_gate_id": self.calibration_gate_id,
        }

    def _require_intact(self) -> None:
        try:
            self._universe._require_intact()
            self._hierarchical_spec._require_intact()
            for item in self._inputs:
                item._require_intact()
            for inference_record in self.records:
                inference_record._require_intact()
            for p_value_record in self._p_value_records:
                p_value_record._require_intact()
            self.coverage._require_intact()
            self.hierarchical_fdr._require_intact()
            expected_coverage = validate_frozen_hypothesis_coverage(
                self._universe,
                tuple(
                    HypothesisCoverageRecord(
                        hypothesis_id=record.hypothesis_id,
                        status=record.status,
                        reason_code=record.reason_code,
                    )
                    for record in self._p_value_records
                ),
            )
            expected_hierarchical = evaluate_hierarchical_fdr(
                self._universe,
                self._p_value_records,
                spec=self._hierarchical_spec,
                calibration_gate=self._calibration_gate,
            )
            observed_sets = {
                record.context_permutation_plan_ids
                for record in self.records
                if record.status is HypothesisCoverageStatus.OBSERVED
            }
            common_plan_ids = next(iter(observed_sets), ())
            identifiers = tuple(record.hypothesis_id for record in self.records)
            p_identifiers = tuple(
                record.hypothesis_id for record in self._p_value_records
            )
            by_hypothesis = {
                record.hypothesis_id: record for record in self._p_value_records
            }
            input_by_hypothesis = {item.hypothesis_id: item for item in self._inputs}
            expected_input_ids = tuple(
                declaration.hypothesis_id
                for declaration in self._universe.declarations
                if declaration.prefilter_status is HypothesisPrefilterStatus.INCLUDED
            )
            point_ids = {item.point_crossfit_id for item in self._inputs}
            resampling_ids = {item.resampling_result_id for item in self._inputs}
            expected_id = stable_id(
                "frozen_hypothesis_inference_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            release_runtime_valid = (
                not self.hierarchical_fdr.q_value_release_allowed
                or all(
                    declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED
                    or by_hypothesis[declaration.hypothesis_id].status
                    is HypothesisCoverageStatus.OBSERVED
                    for declaration in self._universe.declarations
                )
            )
            if self.hierarchical_fdr.q_value_release_allowed and any(
                declaration.prefilter_status is HypothesisPrefilterStatus.INCLUDED
                for declaration in self._universe.declarations
            ):
                release_runtime_valid = release_runtime_valid and (
                    len(common_plan_ids) >= _MINIMUM_CONTEXT_PERMUTATIONS
                )
            valid = (
                self._producer_marker == _COLLECTION_MARKER
                and identifiers == self._universe.hypothesis_ids
                and p_identifiers == self._universe.hypothesis_ids
                and tuple(item.hypothesis_id for item in self._inputs)
                == expected_input_ids
                and len(observed_sets) <= 1
                and len(point_ids) <= 1
                and len(resampling_ids) <= 1
                and self.universe_id == self._universe.universe_id
                and self.hierarchical_procedure_id
                == self._hierarchical_spec.procedure_id
                and self.input_ids == tuple(item.input_id for item in self._inputs)
                and self.record_ids
                == tuple(record.record_id for record in self.records)
                and self.p_value_record_ids
                == tuple(record.record_id for record in self._p_value_records)
                and all(
                    record.p_value_record_id
                    == by_hypothesis[record.hypothesis_id].record_id
                    for record in self.records
                )
                and all(
                    record.point_crossfit_id
                    == input_by_hypothesis[record.hypothesis_id].point_crossfit_id
                    and record.resampling_result_id
                    == input_by_hypothesis[record.hypothesis_id].resampling_result_id
                    for record in self.records
                    if record.hypothesis_id in input_by_hypothesis
                )
                and self.common_context_permutation_plan_ids == common_plan_ids
                and self.coverage_id == expected_coverage.coverage_id
                and self.coverage.coverage_id == expected_coverage.coverage_id
                and self.hierarchical_collection_id
                == expected_hierarchical.collection_id
                and self.hierarchical_fdr.collection_id
                == expected_hierarchical.collection_id
                and self.calibration_gate_id
                == expected_hierarchical.calibration_gate_id
                and release_runtime_valid
                and self.collection_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen hypothesis inference collection failed integrity validation",
                code="frozen_hypothesis_inference_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute from the exact universe and intact parents",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen hypothesis inference collection failed integrity validation",
                code="frozen_hypothesis_inference_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute from the exact universe and intact parents",
            )

    def result_for(self, hypothesis_id: str) -> FrozenHypothesisInferenceRecord:
        self._require_intact()
        identifier = _name(hypothesis_id, field_name="hypothesis_id")
        matched = tuple(
            record for record in self.records if record.hypothesis_id == identifier
        )
        if len(matched) != 1:
            raise ContractError(
                "Hypothesis is absent from the frozen inference collection",
                code="frozen_hypothesis_inference_result_not_found",
                field="hypothesis_id",
                remediation="Use an exact hypothesis ID from the frozen universe",
            )
        return matched[0]

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "collection_id": self.collection_id,
            **self._identity_payload(),
            "q_value_release_allowed": self.q_value_release_allowed,
            "context_permutation_plan_set_synchronized": True,
            "records": [record.to_dict() for record in self.records],
            "coverage": self.coverage.to_dict(),
            "hierarchical_fdr": self.hierarchical_fdr.to_dict(),
        }


def run_frozen_hypothesis_inference(
    universe: FrozenHypothesisUniverse,
    inputs: Sequence[FrozenHypothesisInferenceInput],
    *,
    calibration_gate: G3FrequencyCalibrationGate | None = None,
) -> FrozenHypothesisInferenceCollection:
    """Fit every included v1 hypothesis and adjust the exact frozen collection."""

    _require_v1_universe(universe)
    by_id = _validated_inputs(universe, inputs)
    primary_by_key = {
        declaration.hypothesis_key: declaration
        for declaration in universe.declarations
        if declaration.role is HypothesisRole.PRIMARY
    }
    candidates: dict[str, _CandidateResult] = {}
    for declaration in universe.declarations:
        if declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED:
            continue
        item = by_id[declaration.hypothesis_id]
        target = build_frozen_family_effect_target(
            universe,
            declaration.hypothesis_id,
        )
        if declaration.role is HypothesisRole.PRIMARY:
            score_target = target
        else:
            assert declaration.parent_key is not None
            parent = primary_by_key[declaration.parent_key]
            score_target = build_frozen_family_effect_target(
                universe,
                parent.hypothesis_id,
            )
        lineage = _PermutationLineage((), ())
        try:
            lineage = _context_permutation_lineage(item.resampling)
            if declaration.role is HypothesisRole.PRIMARY:
                candidate = _primary_candidate(item, target, lineage)
            else:
                candidate = _secondary_candidate(
                    item,
                    target,
                    score_target,
                    lineage,
                )
        except Exception as error:
            reason = _failure_reason(error)
            candidate = _CandidateResult(
                target=target,
                score_target=score_target,
                source_result_id=_failure_source_id(declaration, item, reason),
                status=HypothesisCoverageStatus.FAILED,
                reason_code=reason,
                candidate_p_value=None,
                point_crossfit_id=item.point_artifacts.crossfit_id,
                resampling_result_id=item.resampling.result_id,
                context_permutation_plan_ids=lineage.plan_ids,
                context_permutation_record_ids=(lineage.full_pipeline_record_ids),
            )
        candidates[declaration.hypothesis_id] = candidate
    candidates = _synchronize_observed_plan_sets(candidates)
    p_value_records: list[HypothesisPValueRecord] = []
    inference_records: list[FrozenHypothesisInferenceRecord] = []
    for declaration in universe.declarations:
        candidate_for_record = candidates.get(declaration.hypothesis_id)
        p_value_record = _p_value_record(declaration, candidate_for_record)
        p_value_records.append(p_value_record)
        inference_records.append(
            FrozenHypothesisInferenceRecord._from_parts(
                universe,
                declaration,
                p_value_record,
                candidate=candidate_for_record,
            )
        )
    coverage = validate_frozen_hypothesis_coverage(
        universe,
        tuple(
            HypothesisCoverageRecord(
                hypothesis_id=record.hypothesis_id,
                status=record.status,
                reason_code=record.reason_code,
            )
            for record in p_value_records
        ),
    )
    hierarchical_spec = freeze_hierarchical_fdr_spec()
    hierarchical_fdr = evaluate_hierarchical_fdr(
        universe,
        p_value_records,
        spec=hierarchical_spec,
        calibration_gate=calibration_gate,
    )
    return FrozenHypothesisInferenceCollection._from_parts(
        universe,
        hierarchical_spec,
        tuple(by_id.values()),
        inference_records,
        p_value_records,
        coverage,
        hierarchical_fdr,
        calibration_gate,
    )


__all__ = [
    "FrozenHypothesisInferenceCollection",
    "FrozenHypothesisInferenceInput",
    "FrozenHypothesisInferenceRecord",
    "run_frozen_hypothesis_inference",
]
