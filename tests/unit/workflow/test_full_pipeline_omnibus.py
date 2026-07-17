from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import crychic.workflow.full_pipeline_omnibus as omnibus_module
from crychic.core import ContractError
from crychic.inference import (
    HypothesisDeclaration,
    HypothesisRole,
    freeze_hypothesis_universe,
)
from crychic.inference.omnibus import (
    FullPipelineOmnibusRecordStatus,
    OOFContextOmnibusSpec,
)
from crychic.resampling import ContextPermutationPlan, SubjectBootstrapPlan
from crychic.workflow.crossfit import CrossFitArtifacts
from crychic.workflow.full_pipeline_effects import (
    FrozenFamilyEffectTarget,
    build_frozen_family_effect_target,
)
from crychic.workflow.full_pipeline_omnibus import (
    build_family_omnibus_spec,
    fit_crossfit_family_omnibus,
    full_pipeline_family_omnibus_records,
    summarize_family_omnibus_full_pipeline,
)
from crychic.workflow.full_pipeline_resampling import (
    FullPipelineResampleRecord,
    FullPipelineResampleStatus,
    FullPipelineResamplingOperation,
    FullPipelineResamplingResult,
)

_OMNIBUS_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_EFFECT_ENDPOINT = "family_common_integrated_lr_context_effect_v1"


def _declaration(
    *,
    endpoint: str = _OMNIBUS_ENDPOINT,
    role: HypothesisRole = HypothesisRole.PRIMARY,
    parent_key: str | None = None,
) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint=endpoint,
        contrast_name="stim_vs_control",
        receiver="Receiver",
        family_id="family-1",
        mode="state",
        role=role,
        multiplicity_family=(
            "gene-state-primary"
            if role is HypothesisRole.PRIMARY
            else "gene-state-secondary"
        ),
        parent_key=parent_key,
    )


def _target() -> FrozenFamilyEffectTarget:
    declaration = _declaration()
    universe = freeze_hypothesis_universe(
        (declaration,),
        universe_name="family-omnibus-v1",
    )
    return build_frozen_family_effect_target(universe, declaration.hypothesis_id)


def _spec(
    target: FrozenFamilyEffectTarget,
    context_ids: tuple[str, ...] = ("context-a", "context-b"),
) -> OOFContextOmnibusSpec:
    return OOFContextOmnibusSpec(
        hypothesis_id=target.hypothesis_id,
        omnibus_name=f"{target.contrast_name}:{target.target_id}",
        context_ids=context_ids,
    )


def _artifacts(
    crossfit_id: str = "crossfit-point",
    spec_id: str = "crossfit-spec",
) -> CrossFitArtifacts:
    artifacts = object.__new__(CrossFitArtifacts)
    object.__setattr__(artifacts, "crossfit_id", crossfit_id)
    object.__setattr__(artifacts, "spec", SimpleNamespace(spec_id=spec_id))
    return artifacts


def _permutation_plan(index: int) -> ContextPermutationPlan:
    plan = object.__new__(ContextPermutationPlan)
    object.__setattr__(plan, "permutation_id", f"permutation-{index}")
    object.__setattr__(plan, "exchangeability_id", "exchangeability")
    object.__setattr__(plan, "resample_index", index)
    object.__setattr__(plan, "seed_lineage", f"seed-{index}")
    return plan


def _bootstrap_plan(index: int) -> SubjectBootstrapPlan:
    plan = object.__new__(SubjectBootstrapPlan)
    object.__setattr__(plan, "bootstrap_id", f"bootstrap-{index}")
    object.__setattr__(plan, "exchangeability_id", "exchangeability")
    object.__setattr__(plan, "resample_index", index)
    object.__setattr__(plan, "seed_lineage", f"seed-bootstrap-{index}")
    return plan


def _workflow_record(
    plan: ContextPermutationPlan | SubjectBootstrapPlan,
    *,
    child: CrossFitArtifacts | None,
    failed: bool = False,
) -> FullPipelineResampleRecord:
    record = object.__new__(FullPipelineResampleRecord)
    is_permutation = isinstance(plan, ContextPermutationPlan)
    plan_id = plan.permutation_id if is_permutation else plan.bootstrap_id
    object.__setattr__(
        record,
        "operation",
        (
            FullPipelineResamplingOperation.CONTEXT_PERMUTATION
            if is_permutation
            else FullPipelineResamplingOperation.SUBJECT_BOOTSTRAP
        ),
    )
    object.__setattr__(record, "resample_index", plan.resample_index)
    object.__setattr__(record, "plan_id", plan_id)
    object.__setattr__(record, "exchangeability_id", plan.exchangeability_id)
    object.__setattr__(record, "plan_seed_lineage", plan.seed_lineage)
    object.__setattr__(
        record,
        "status",
        FullPipelineResampleStatus.FAILED
        if failed
        else FullPipelineResampleStatus.SUCCEEDED,
    )
    object.__setattr__(
        record,
        "crossfit_id",
        None if child is None else child.crossfit_id,
    )
    object.__setattr__(
        record,
        "failure_code",
        "synthetic_pipeline_failure" if failed else None,
    )
    object.__setattr__(record, "_child", child)
    object.__setattr__(record, "record_id", f"workflow-{plan_id}")
    return record


def _resampling(
    pairs: tuple[
        tuple[
            ContextPermutationPlan | SubjectBootstrapPlan,
            FullPipelineResampleRecord,
        ],
        ...,
    ],
    *,
    retain_children: bool = True,
    crossfit_spec_id: str = "crossfit-spec",
) -> FullPipelineResamplingResult:
    result = object.__new__(FullPipelineResamplingResult)
    object.__setattr__(result, "plans", tuple(plan for plan, _ in pairs))
    object.__setattr__(result, "records", tuple(record for _, record in pairs))
    object.__setattr__(
        result,
        "exchangeability",
        SimpleNamespace(exchangeability_id="exchangeability"),
    )
    object.__setattr__(result, "retain_children", retain_children)
    object.__setattr__(result, "crossfit_spec_id", crossfit_spec_id)
    return result


def test_target_builder_accepts_primary_omnibus_and_rejects_secondary() -> None:
    primary = _declaration()
    secondary = _declaration(
        role=HypothesisRole.SECONDARY,
        parent_key=primary.hypothesis_key,
    )
    universe = freeze_hypothesis_universe(
        (primary, secondary),
        universe_name="primary-secondary-omnibus",
    )

    target = build_frozen_family_effect_target(universe, primary.hypothesis_id)
    assert target.endpoint == _OMNIBUS_ENDPOINT
    assert target.role is HypothesisRole.PRIMARY
    with pytest.raises(ContractError) as error:
        build_frozen_family_effect_target(universe, secondary.hypothesis_id)
    assert error.value.details.code == "family_omnibus_hypothesis_role_mismatch"

    effect_primary = _declaration(endpoint=_EFFECT_ENDPOINT)
    effect_secondary = _declaration(
        endpoint=_EFFECT_ENDPOINT,
        role=HypothesisRole.SECONDARY,
        parent_key=effect_primary.hypothesis_key,
    )
    effect_universe = freeze_hypothesis_universe(
        (effect_primary, effect_secondary),
        universe_name="legacy-effect-endpoint",
    )
    effect_target = build_frozen_family_effect_target(
        effect_universe,
        effect_secondary.hypothesis_id,
    )
    assert effect_target.role is HypothesisRole.SECONDARY


def test_build_spec_freezes_exact_contexts_and_target_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    artifacts = _artifacts()
    chains = (
        ("fold-1", SimpleNamespace(context_ids=("context-b", "context-a")), object()),
        ("fold-2", SimpleNamespace(context_ids=("context-a", "context-b")), object()),
    )
    monkeypatch.setattr(omnibus_module, "_matched_chains", lambda *args: chains)

    spec = build_family_omnibus_spec(artifacts, target)

    assert spec.context_ids == ("context-a", "context-b")
    assert spec.hypothesis_id == target.hypothesis_id
    assert target.target_id in spec.omnibus_name

    mismatched = (
        chains[0],
        ("fold-2", SimpleNamespace(context_ids=("context-a", "context-c")), object()),
    )
    monkeypatch.setattr(omnibus_module, "_matched_chains", lambda *args: mismatched)
    with pytest.raises(ContractError) as error:
        build_family_omnibus_spec(artifacts, target)
    assert error.value.details.code == "family_omnibus_fold_context_mismatch"


def test_point_fit_uses_exact_family_score_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    spec = _spec(target)
    artifacts = _artifacts()
    score_table = pd.DataFrame({"sentinel": [1]})
    sentinel = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        omnibus_module,
        "_context_ids",
        lambda *args: spec.context_ids,
    )

    def extract(*args: object) -> pd.DataFrame:
        captured["extract_args"] = args
        return score_table

    def fit(table: pd.DataFrame, supplied: OOFContextOmnibusSpec) -> object:
        captured["fit_args"] = (table, supplied)
        return sentinel

    monkeypatch.setattr(omnibus_module, "family_effect_oof_score_table", extract)
    monkeypatch.setattr(omnibus_module, "fit_oof_context_omnibus", fit)

    result = fit_crossfit_family_omnibus(artifacts, target, spec)

    assert result is sentinel
    extraction_spec = captured["extract_args"][2]
    assert extraction_spec.hypothesis_id == target.hypothesis_id
    assert extraction_spec.contrast_name == target.contrast_name
    assert set(extraction_spec.context_ids) == set(spec.context_ids)
    assert captured["fit_args"] == (score_table, spec)


def test_records_require_retained_children() -> None:
    target = _target()
    spec = _spec(target)
    plan = _permutation_plan(0)
    record = _workflow_record(plan, child=_artifacts("child-0"))
    result = _resampling(((plan, record),), retain_children=False)

    with pytest.raises(ContractError) as error:
        full_pipeline_family_omnibus_records(result, target, spec)
    assert error.value.details.code == "full_pipeline_omnibus_children_not_retained"


def test_permutation_records_preserve_lineage_and_typed_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    spec = _spec(target)
    plans = tuple(_permutation_plan(index) for index in range(4))
    children = tuple(_artifacts(f"child-{index}") for index in range(3))
    records = (
        _workflow_record(plans[0], child=children[0]),
        _workflow_record(plans[1], child=children[1]),
        _workflow_record(plans[2], child=children[2]),
        _workflow_record(plans[3], child=None, failed=True),
    )
    bootstrap = _bootstrap_plan(9)
    bootstrap_record = _workflow_record(bootstrap, child=_artifacts("bootstrap-child"))
    resampling = _resampling(
        (*tuple(zip(plans, records, strict=True)), (bootstrap, bootstrap_record))
    )

    def fit(
        child: CrossFitArtifacts,
        supplied_target: FrozenFamilyEffectTarget,
        supplied_spec: OOFContextOmnibusSpec,
    ) -> object:
        assert supplied_target is target and supplied_spec is spec
        if child.crossfit_id == "child-0":
            return SimpleNamespace(
                observed=True,
                result_id="omnibus-observed",
                wald_statistic=3.5,
                reason_code=None,
            )
        if child.crossfit_id == "child-1":
            return SimpleNamespace(
                observed=False,
                result_id="omnibus-ne",
                wald_statistic=None,
                reason_code="oof_omnibus_covariance_rank_deficient",
            )
        raise ContractError(
            "synthetic adapter failure",
            code="synthetic_omnibus_failure",
            field="fit",
        )

    monkeypatch.setattr(omnibus_module, "fit_crossfit_family_omnibus", fit)

    result = full_pipeline_family_omnibus_records(resampling, target, spec)

    assert len(result) == 4
    assert tuple(row.plan_id for row in result) == tuple(
        plan.permutation_id for plan in plans
    )
    assert tuple(row.full_pipeline_record_id for row in result) == tuple(
        record.record_id for record in records
    )
    assert tuple(row.status for row in result) == (
        FullPipelineOmnibusRecordStatus.OBSERVED,
        FullPipelineOmnibusRecordStatus.NOT_ESTIMABLE,
        FullPipelineOmnibusRecordStatus.FAILED,
        FullPipelineOmnibusRecordStatus.FAILED,
    )
    assert result[1].reason_code == "oof_omnibus_covariance_rank_deficient"
    assert result[2].reason_code == "synthetic_omnibus_failure"
    assert result[3].reason_code == "synthetic_pipeline_failure"


def test_plan_alignment_fails_closed_before_refitting() -> None:
    target = _target()
    spec = _spec(target)
    plan = _permutation_plan(0)
    record = _workflow_record(plan, child=_artifacts("child-0"))
    object.__setattr__(record, "plan_id", "different-plan")
    resampling = _resampling(((plan, record),))

    with pytest.raises(ContractError) as error:
        full_pipeline_family_omnibus_records(resampling, target, spec)
    assert error.value.details.code == "full_pipeline_omnibus_plan_alignment_mismatch"


def test_summary_checks_crossfit_lineage_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    spec = _spec(target)
    point = _artifacts()
    plan = _permutation_plan(0)
    record = _workflow_record(plan, child=_artifacts("child-0"))
    mismatch = _resampling(((plan, record),), crossfit_spec_id="different-spec")
    with pytest.raises(ContractError) as error:
        summarize_family_omnibus_full_pipeline(point, mismatch, target, spec)
    assert error.value.details.code == "full_pipeline_omnibus_crossfit_spec_mismatch"

    resampling = _resampling(((plan, record),))
    point_result = object()
    omnibus_records = (object(),)
    sentinel = object()
    monkeypatch.setattr(
        omnibus_module,
        "fit_crossfit_family_omnibus",
        lambda *args: point_result,
    )
    monkeypatch.setattr(
        omnibus_module,
        "full_pipeline_family_omnibus_records",
        lambda *args: omnibus_records,
    )
    monkeypatch.setattr(
        omnibus_module,
        "summarize_full_pipeline_omnibus",
        lambda *args: sentinel,
    )

    result = summarize_family_omnibus_full_pipeline(
        point,
        resampling,
        target,
        spec,
    )
    assert result is sentinel
