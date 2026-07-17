from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from tests.integration.test_graph_fused_crossfit_registry import _run_registry
from tests.integration.test_subject_crossfit import (
    _adata,
    _bundle,
    _config,
    _persistable_spec,
    _prior,
)

from crychic.core import ContractError
from crychic.design import global_one_vs_rest, node_context_fields
from crychic.workflow import (
    CrossFitArtifacts,
    GraphFusedFamilyEffectCollection,
    GraphFusedIntegratedLRCollection,
    GraphFusedIntegratedLRSpec,
    build_graph_fused_integrated_lr_scores,
    derive_graph_fused_family_effects,
    run_subject_crossfit,
)
from crychic.workflow import graph_fused_integrated as integrated_module


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def integrated_sources() -> tuple[
    CrossFitArtifacts,
    GraphFusedFamilyEffectCollection,
    GraphFusedIntegratedLRSpec,
    GraphFusedIntegratedLRCollection,
]:
    contexts = ("control", "stim")
    contrasts = (
        global_one_vs_rest(
            contexts,
            "control",
            name="control_global_one_vs_rest",
        ),
        global_one_vs_rest(
            contexts,
            "stim",
            name="stim_global_one_vs_rest",
        ),
    )
    base = _persistable_spec()
    crossfit_spec = replace(
        base,
        contrasts=contrasts,
        training_spec=replace(base.training_spec, sender_contrasts=contrasts),
    )
    adata = _adata(tuple(f"p{index}" for index in range(1, 9)))
    crossfit = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=crossfit_spec,
    )
    registry = _run_registry(adata, crossfit=crossfit)
    effects = derive_graph_fused_family_effects(crossfit, registry)
    spec = GraphFusedIntegratedLRSpec(
        context_contrasts={
            node_context_fields("control", _config().context_keys)[0]: (
                "control_global_one_vs_rest"
            ),
            node_context_fields("stim", _config().context_keys)[0]: (
                "stim_global_one_vs_rest"
            ),
        }
    )
    collection = build_graph_fused_integrated_lr_scores(
        crossfit,
        effects,
        spec=spec,
    )
    return crossfit, effects, spec, collection


def test_integrated_collection_binds_graph_and_family_common_parents(
    integrated_sources: tuple[
        CrossFitArtifacts,
        GraphFusedFamilyEffectCollection,
        GraphFusedIntegratedLRSpec,
        GraphFusedIntegratedLRCollection,
    ],
) -> None:
    crossfit, effects, spec, collection = integrated_sources
    family = collection.family_scores
    lr = collection.integrated_lr_scores

    assert collection.crossfit_id == crossfit.crossfit_id
    assert collection.graph_effect_collection_id == effects.collection_id
    assert not family.empty and not lr.empty
    assert not family.duplicated(
        [
            "fold_id",
            "contrast_id",
            "context_id",
            "receiver",
            "sample_id",
            "family_id",
            "mode",
        ]
    ).any()
    assert not lr.duplicated(
        [
            "fold_id",
            "contrast_id",
            "context_id",
            "receiver",
            "sample_id",
            "family_id",
            "mode",
            "driver_id",
            "interaction_id",
        ]
    ).any()
    assert set(family["context_id"]) == set(spec.contrast_by_context)
    assert set(family["graph_effect_collection_id"]) == {effects.collection_id}
    assert family[
        [
            "family_common_functional_id",
            "family_common_application_id",
            "family_common_family_scores_digest",
            "family_common_member_scores_digest",
            "edge_evidence_digest",
        ]
    ].notna().all().all()
    assert not family["cross_receiver_comparable"].any()
    assert not family["cross_context_comparable"].any()
    assert not family["cross_mode_comparable"].any()
    assert not family["formal_inference_allowed"].any()
    assert not lr["cross_receiver_comparable"].any()
    assert not lr["cross_context_comparable"].any()
    assert not lr["cross_mode_comparable"].any()
    assert not lr["formal_inference_allowed"].any()
    assert {
        "incremental_downstream_gain",
        "differential_effect",
        "family_core_strength",
        "sender_unresolved_strength",
    }.isdisjoint(family.columns) and {
        "incremental_downstream_gain",
        "differential_effect",
        "family_core_strength",
        "sender_unresolved_strength",
    }.isdisjoint(lr.columns)
    assert collection.to_manifest()["comparability_scope"] == (
        "within_same_spec_receiver_context_mode_only"
    )


def test_complete_member_allocations_conserve_integrated_family_score(
    integrated_sources: tuple[
        CrossFitArtifacts,
        GraphFusedFamilyEffectCollection,
        GraphFusedIntegratedLRSpec,
        GraphFusedIntegratedLRCollection,
    ],
) -> None:
    _, _, _, collection = integrated_sources
    family = collection.family_scores
    lr = collection.integrated_lr_scores
    keys = [
        "fold_id",
        "contrast_id",
        "context_id",
        "receiver",
        "sample_id",
        "family_id",
        "mode",
    ]
    family_by_key = family.set_index(keys)
    checked = 0
    for raw_key, group in lr.groupby(keys, observed=True, sort=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        family_row = family_by_key.loc[key]
        if group["integrated_lr_score"].notna().all():
            assert float(group["integrated_lr_score"].sum()) == pytest.approx(
                float(family_row["integrated_family_score"])
            )
            checked += 1
    assert checked > 0


def test_mixed_supported_and_ne_keeps_family_core_but_fails_member_allocation(
) -> None:
    spec = GraphFusedIntegratedLRSpec(
        context_contrasts={"control": "control_global_one_vs_rest"}
    )
    effect = pd.Series(
        {
            "status": "observed",
            "reason_code": None,
            "family_coefficient": 0.5,
            "bounded_conditional_gain": 0.6,
        }
    )
    source = pd.Series(
        {
            "family_availability": 0.8,
            "receptor_eligible": True,
            "ligand_contrast_gate_status": "supported",
            "ligand_contrast_supported_interaction_count": 1,
            "ligand_contrast_not_estimable_interaction_count": 1,
        }
    )
    members = pd.DataFrame(
        [
            {
                "receptor_eligible": True,
                "ligand_contrast_gate_status": "supported",
                "ligand_contrast_gate_reason_code": None,
                "within_family_lr_weight": None,
                "within_family_entropy": None,
                "lr_identifiability_status": "unresolved",
            },
            {
                "receptor_eligible": True,
                "ligand_contrast_gate_status": "not_estimable",
                "ligand_contrast_gate_reason_code": "gate_ne",
                "within_family_lr_weight": None,
                "within_family_entropy": None,
                "lr_identifiability_status": "unresolved",
            },
        ]
    )

    family = integrated_module._family_outcome(effect, source, members, spec)
    member_outcomes = [
        integrated_module._member_outcome(row, family)
        for _, row in members.iterrows()
    ]

    assert family["status"] == "observed"
    assert float(family["integrated_family_score"]) > 0.0
    assert {status for _, status, _ in member_outcomes} == {"not_estimable"}
    assert all(value is None for value, _, _ in member_outcomes)


def test_nonpositive_heldout_gain_is_observed_zero_not_fit_structural_zero() -> None:
    spec = GraphFusedIntegratedLRSpec(
        context_contrasts={"control": "control_global_one_vs_rest"}
    )
    effect = pd.Series(
        {
            "status": "observed",
            "reason_code": None,
            "family_coefficient": 0.5,
            "bounded_conditional_gain": 0.0,
        }
    )
    source = pd.Series(
        {
            "family_availability": 0.8,
            "receptor_eligible": True,
            "ligand_contrast_gate_status": "supported",
            "ligand_contrast_supported_interaction_count": 1,
            "ligand_contrast_not_estimable_interaction_count": 0,
        }
    )
    members = pd.DataFrame(
        [
            {
                "receptor_eligible": True,
                "ligand_contrast_gate_status": "supported",
                "within_family_lr_weight": 1.0,
                "within_family_entropy": 0.0,
                "lr_identifiability_status": "resolved",
            }
        ]
    )

    family = integrated_module._family_outcome(effect, source, members, spec)

    assert family["family_selected"] is True
    assert family["status"] == "observed"
    assert family["integrated_family_score"] == pytest.approx(0.0)


def test_sample_manifest_preserves_independent_context_subject_axes() -> None:
    table = pd.DataFrame(
        [
            {"sample_id": "s-control", "subject_id": "p1", "context_id": "control"},
            {"sample_id": "s-stim", "subject_id": "p2", "context_id": "stim"},
        ]
    )

    assert integrated_module._sample_manifest(table, field_name="sample") == {
        "s-control": ("p1", "control"),
        "s-stim": ("p2", "stim"),
    }

    inconsistent = pd.concat(
        [
            table,
            pd.DataFrame(
                [{"sample_id": "s-control", "subject_id": "p2", "context_id": "stim"}]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ContractError) as caught:
        integrated_module._sample_manifest(inconsistent, field_name="sample")
    assert caught.value.details.code == "graph_fused_integrated_source_mismatch"


def test_spec_rejects_postfit_selection_threshold() -> None:
    with pytest.raises(ValueError, match=r"fixed at 0\.0"):
        GraphFusedIntegratedLRSpec(
            context_contrasts={"control": "control_global_one_vs_rest"},
            family_selection_threshold=0.1,
        )


def test_integrated_builder_rejects_noncanonical_context_labels(
    integrated_sources: tuple[
        CrossFitArtifacts,
        GraphFusedFamilyEffectCollection,
        GraphFusedIntegratedLRSpec,
        GraphFusedIntegratedLRCollection,
    ],
) -> None:
    crossfit, effects, _, _ = integrated_sources
    spec = GraphFusedIntegratedLRSpec(
        context_contrasts={
            "control": "control_global_one_vs_rest",
            "stim": "stim_global_one_vs_rest",
        }
    )

    with pytest.raises(ContractError) as caught:
        build_graph_fused_integrated_lr_scores(crossfit, effects, spec=spec)

    assert caught.value.details.code == "graph_fused_integrated_source_mismatch"


def test_integrated_builder_rejects_contrast_mapped_to_wrong_focal_context(
    integrated_sources: tuple[
        CrossFitArtifacts,
        GraphFusedFamilyEffectCollection,
        GraphFusedIntegratedLRSpec,
        GraphFusedIntegratedLRCollection,
    ],
) -> None:
    crossfit, effects, valid_spec, _ = integrated_sources
    first, second = tuple(valid_spec.contrast_by_context.items())
    spec = GraphFusedIntegratedLRSpec(
        context_contrasts={
            first[0]: second[1],
            second[0]: first[1],
        }
    )

    with pytest.raises(ContractError) as caught:
        build_graph_fused_integrated_lr_scores(crossfit, effects, spec=spec)

    assert caught.value.details.code == "graph_fused_integrated_source_mismatch"


def test_integrated_collection_rejects_private_table_mutation(
    integrated_sources: tuple[
        CrossFitArtifacts,
        GraphFusedFamilyEffectCollection,
        GraphFusedIntegratedLRSpec,
        GraphFusedIntegratedLRCollection,
    ],
) -> None:
    _, _, _, collection = integrated_sources
    collection._family_scores.loc[0, "integrated_family_score"] = 0.123

    with pytest.raises(ContractError) as caught:
        _ = collection.family_scores
    assert caught.value.details.code == (
        "graph_fused_integrated_collection_integrity_violation"
    )
