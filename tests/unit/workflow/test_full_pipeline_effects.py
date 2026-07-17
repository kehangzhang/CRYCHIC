from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tests.support.g3f import passed_g3f_gate

import crychic.workflow.full_pipeline_effects as effects_module
from crychic.core import ContractError
from crychic.design import balanced_contrast
from crychic.inference import (
    FullPipelineEffectDistributionSpec,
    G3FrequencyCalibrationGate,
    HypothesisDeclaration,
    HypothesisPrefilterPolicy,
    HypothesisPrefilterStatus,
    HypothesisRole,
    OOFEffectSpec,
    SpecificityDirection,
    freeze_hypothesis_universe,
)
from crychic.workflow import (
    CrossFitArtifacts,
    FamilyEffectTarget,
    FrozenFamilyEffectTarget,
    FullPipelineResamplingResult,
    build_family_effect_oof_spec,
    build_frozen_family_effect_target,
    family_effect_oof_score_table,
    summarize_family_effect_full_pipeline,
)

_ENDPOINT = "family_common_integrated_lr_context_effect_v1"


def _declaration(
    *,
    endpoint: str = _ENDPOINT,
    receiver: str = "Receiver",
    family_id: str = "family-1",
    filtered: bool = False,
) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint=endpoint,
        contrast_name="stim_vs_control",
        receiver=receiver,
        family_id=family_id,
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="gene-state-primary",
        prefilter_policy=(
            HypothesisPrefilterPolicy.EXTERNAL_RESOURCE
            if filtered
            else HypothesisPrefilterPolicy.NONE
        ),
        prefilter_status=(
            HypothesisPrefilterStatus.FILTERED
            if filtered
            else HypothesisPrefilterStatus.INCLUDED
        ),
        filter_reason_code="resource_scope_excluded" if filtered else None,
    )


def _frozen_target() -> FrozenFamilyEffectTarget:
    declaration = _declaration()
    other = HypothesisDeclaration(
        endpoint=_ENDPOINT,
        contrast_name=declaration.contrast_name,
        receiver=declaration.receiver,
        family_id=declaration.family_id,
        mode=declaration.mode,
        role=HypothesisRole.SECONDARY,
        multiplicity_family=declaration.multiplicity_family,
        parent_key=declaration.hypothesis_key,
    )
    universe = freeze_hypothesis_universe(
        (declaration, other),
        universe_name="family-common-gene-state-v1",
    )
    return build_frozen_family_effect_target(
        universe,
        declaration.hypothesis_id,
    )


def _effect_spec(
    target: FamilyEffectTarget | FrozenFamilyEffectTarget,
) -> OOFEffectSpec:
    return OOFEffectSpec(
        hypothesis_id=target.hypothesis_id,
        contrast_name=target.contrast_name,
        contrast_weights=(("stim", 1.0), ("control", -1.0)),
    )


def _distribution_spec(
    target: FamilyEffectTarget | FrozenFamilyEffectTarget,
) -> FullPipelineEffectDistributionSpec:
    return FullPipelineEffectDistributionSpec(
        effect_spec=_effect_spec(target),
        minimum_effect=0.0,
        specificity_direction=SpecificityDirection.GREATER,
    )


def _fake_workflow_parents() -> tuple[
    CrossFitArtifacts,
    FullPipelineResamplingResult,
]:
    point = object.__new__(CrossFitArtifacts)
    object.__setattr__(point, "spec", SimpleNamespace(spec_id="crossfit-spec"))
    resampling = object.__new__(FullPipelineResamplingResult)
    object.__setattr__(resampling, "crossfit_spec_id", "crossfit-spec")
    return point, resampling


def _passed_gate() -> G3FrequencyCalibrationGate:
    return passed_g3f_gate(_frozen_target()._universe)


def test_frozen_target_is_producer_owned_and_binds_complete_universe_lineage() -> None:
    declaration = _declaration()
    other = _declaration(receiver="Receiver-2", family_id="family-2")
    universe = freeze_hypothesis_universe(
        (declaration, other),
        universe_name="family-common-gene-state-v1",
    )
    reordered = freeze_hypothesis_universe(
        (other, declaration),
        universe_name="family-common-gene-state-v1",
    )
    first = build_frozen_family_effect_target(universe, declaration.hypothesis_id)
    repeated = build_frozen_family_effect_target(
        reordered,
        declaration.hypothesis_id,
    )

    assert first.target_id == repeated.target_id
    assert first.universe_id == universe.universe_id
    assert first.universe_multiplicity_denominator == 2
    assert first.universe_multiplicity_family_sizes == (("gene-state-primary", 2),)
    assert first.hypothesis_id == declaration.hypothesis_id
    assert first.endpoint == _ENDPOINT
    assert first.contrast_name == declaration.contrast_name
    assert first.receiver == declaration.receiver
    assert first.family_id == declaration.family_id
    assert first.mode == declaration.mode
    assert first.q_value_release_allowed is False
    assert first.complete_universe_coverage_claimed is False
    payload = first.to_dict()
    assert payload["declaration"] == declaration.to_dict()
    assert payload["complete_universe_coverage_claimed"] is False
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenFamilyEffectTarget()


def test_target_identity_changes_when_complete_universe_changes() -> None:
    declaration = _declaration()
    first_universe = freeze_hypothesis_universe(
        (declaration,),
        universe_name="family-common-gene-state-v1",
    )
    expanded_universe = freeze_hypothesis_universe(
        (
            declaration,
            _declaration(receiver="Receiver-2", family_id="family-2"),
        ),
        universe_name="family-common-gene-state-v1",
    )
    first = build_frozen_family_effect_target(
        first_universe,
        declaration.hypothesis_id,
    )
    expanded = build_frozen_family_effect_target(
        expanded_universe,
        declaration.hypothesis_id,
    )

    assert first.hypothesis_id == expanded.hypothesis_id
    assert first.universe_id != expanded.universe_id
    assert first.target_id != expanded.target_id


def test_builder_rejects_wrong_endpoint_filtered_and_unknown_hypotheses() -> None:
    wrong = _declaration(endpoint="different-endpoint")
    wrong_universe = freeze_hypothesis_universe(
        (wrong,),
        universe_name="wrong-endpoint",
    )
    with pytest.raises(ContractError) as endpoint_error:
        build_frozen_family_effect_target(wrong_universe, wrong.hypothesis_id)
    assert endpoint_error.value.details.code == (
        "family_effect_hypothesis_endpoint_mismatch"
    )

    filtered = _declaration(filtered=True)
    filtered_universe = freeze_hypothesis_universe(
        (filtered,),
        universe_name="filtered",
    )
    with pytest.raises(ContractError) as filtered_error:
        build_frozen_family_effect_target(
            filtered_universe,
            filtered.hypothesis_id,
        )
    assert filtered_error.value.details.code == "filtered_family_effect_hypothesis"

    with pytest.raises(ContractError) as unknown_error:
        build_frozen_family_effect_target(
            filtered_universe,
            "hypothesis_unknown",
        )
    assert unknown_error.value.details.code == "hypothesis_not_in_frozen_universe"


def test_frozen_target_revalidates_binding_and_parent_tampering() -> None:
    target = _frozen_target()
    object.__setattr__(target, "receiver", "forged")
    with pytest.raises(ContractError) as target_error:
        target.to_dict()
    assert target_error.value.details.code == (
        "frozen_family_effect_target_integrity_violation"
    )

    parent_target = _frozen_target()
    object.__setattr__(parent_target._declaration, "family_id", "forged")
    with pytest.raises(ContractError) as parent_error:
        parent_target.to_dict()
    assert parent_error.value.details.code == (
        "frozen_family_effect_target_integrity_violation"
    )


def test_built_oof_spec_uses_frozen_declaration_hypothesis_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _frozen_target()
    artifacts = object.__new__(CrossFitArtifacts)
    contrast = balanced_contrast(
        ("stim",),
        ("control",),
        name=target.contrast_name,
    )
    functional = SimpleNamespace(
        sender_functional=SimpleNamespace(
            contrast=contrast,
            contrast_context_ids=tuple(
                (node, f"context-{index}")
                for index, node in enumerate(contrast.weights)
            ),
        )
    )
    monkeypatch.setattr(
        effects_module,
        "_matched_chains",
        lambda artifacts, selected: (("fold-1", functional, object()),),
    )

    spec = build_family_effect_oof_spec(artifacts, target)

    assert spec.hypothesis_id == target.hypothesis_id
    assert spec.hypothesis_id == target._declaration.hypothesis_id


def test_effect_spec_mismatch_fails_before_score_extraction() -> None:
    target = _frozen_target()
    artifacts = object.__new__(CrossFitArtifacts)
    wrong = OOFEffectSpec(
        hypothesis_id="different-hypothesis",
        contrast_name=target.contrast_name,
        contrast_weights=(("stim", 1.0), ("control", -1.0)),
    )

    with pytest.raises(ContractError) as error:
        family_effect_oof_score_table(artifacts, target, wrong)
    assert error.value.details.code == "family_effect_spec_target_mismatch"


def test_legacy_target_without_gate_remains_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point, resampling = _fake_workflow_parents()
    target = FamilyEffectTarget(
        contrast_name="stim_vs_control",
        receiver="Receiver",
        family_id="family-1",
        mode="state",
    )
    distribution_spec = _distribution_spec(target)
    sentinel = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        effects_module,
        "fit_crossfit_family_effect",
        lambda *args, **kwargs: "point-effect",
    )
    monkeypatch.setattr(
        effects_module,
        "full_pipeline_family_effect_records",
        lambda *args, **kwargs: ("record",),
    )

    def summarize(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        effects_module,
        "summarize_full_pipeline_effect_distribution",
        summarize,
    )

    result = summarize_family_effect_full_pipeline(
        point,
        resampling,
        target,
        distribution_spec,
    )

    assert result is sentinel
    assert captured["kwargs"] == {"calibration_gate": None}


def test_passed_gate_refuses_legacy_target_but_accepts_frozen_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point, resampling = _fake_workflow_parents()
    legacy = FamilyEffectTarget(
        contrast_name="stim_vs_control",
        receiver="Receiver",
        family_id="family-1",
        mode="state",
    )
    gate = _passed_gate()
    with pytest.raises(ContractError) as legacy_error:
        summarize_family_effect_full_pipeline(
            point,
            resampling,
            legacy,
            _distribution_spec(legacy),
            calibration_gate=gate,
        )
    assert legacy_error.value.details.code == (
        "legacy_family_effect_target_formal_release_forbidden"
    )

    frozen = _frozen_target()
    sentinel = object()
    monkeypatch.setattr(
        effects_module,
        "fit_crossfit_family_effect",
        lambda *args, **kwargs: "point-effect",
    )
    monkeypatch.setattr(
        effects_module,
        "full_pipeline_family_effect_records",
        lambda *args, **kwargs: ("record",),
    )
    monkeypatch.setattr(
        effects_module,
        "summarize_full_pipeline_effect_distribution",
        lambda *args, **kwargs: sentinel,
    )

    result = summarize_family_effect_full_pipeline(
        point,
        resampling,
        frozen,
        _distribution_spec(frozen),
        calibration_gate=gate,
    )

    assert result is sentinel
