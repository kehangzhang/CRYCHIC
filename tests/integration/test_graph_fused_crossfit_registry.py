from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from anndata import AnnData
from tests.integration.test_subject_crossfit import (
    _adata,
    _bundle,
    _config,
    _prior,
    _spec,
)

from crychic.attribution import GraphPenaltyTuningSpec
from crychic.core import ContractError, stable_id
from crychic.design import ContextGraph
from crychic.workflow import (
    CrossFitArtifacts,
    GraphFusedCrossFitRegistry,
    GraphFusedIntegratedLRSpec,
    GraphFusedWorkflowSpec,
    build_graph_fused_integrated_lr_scores,
    derive_graph_fused_family_effects,
    run_graph_fused_crossfit,
    run_subject_crossfit,
)


def _graph_spec() -> GraphFusedWorkflowSpec:
    return GraphFusedWorkflowSpec(
        tuning_spec=GraphPenaltyTuningSpec(
            lambda1_values=(0.0,),
            lambda2_values=(0.0,),
            lambda_f_values=(0.0,),
        )
    )


def _run_registry(
    adata: AnnData, *, crossfit: CrossFitArtifacts | None = None
) -> GraphFusedCrossFitRegistry:
    if crossfit is None:
        crossfit = run_subject_crossfit(
            adata,
            _config(),
            _bundle(),
            _prior(),
            spec=_spec(),
        )
    return run_graph_fused_crossfit(
        adata,
        crossfit,
        ContextGraph.chain(("control", "stim")),
        _graph_spec(),
    )


def test_graph_fused_crossfit_registry_is_complete_and_row_order_stable() -> None:
    adata = _adata(tuple(f"p{index}" for index in range(1, 9)))
    crossfit = run_subject_crossfit(adata, _config(), _bundle(), _prior(), spec=_spec())
    first = _run_registry(adata, crossfit=crossfit)

    shuffled = adata[adata.obs_names[::-1], :].copy()
    second = _run_registry(shuffled, crossfit=crossfit)

    assert len(first.records) == 4
    assert {record.status for record in first.records} == {"observed"}
    assert first.registry_id == second.registry_id
    assert first.table_digest == second.table_digest
    assert set(first.table["status"]) == {"observed"}
    for record in first.records:
        assert not set(record.training_subject_ids).intersection(
            record.heldout_subject_ids
        )
        assert record.workflow is not None
        assert record.application is not None
        assert set(record.workflow.problem.heldout_subject_ids) == set(
            record.heldout_subject_ids
        )


def test_graph_fused_crossfit_registry_fails_closed_on_raw_root_mismatch() -> None:
    adata = _adata(tuple(f"p{index}" for index in range(1, 9)))
    crossfit = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
    )
    poisoned = deepcopy(adata)
    poisoned.layers["counts"] = poisoned.layers["counts"].copy()
    poisoned.layers["counts"][0, 0] += 1

    with pytest.raises(ContractError) as caught:
        run_graph_fused_crossfit(
            poisoned,
            crossfit,
            ContextGraph.chain(("control", "stim")),
            _graph_spec(),
        )

    assert caught.value.details.code == "graph_fused_crossfit_root_input_mismatch"


def test_graph_fused_crossfit_registry_retains_typed_ne_for_small_outer_scope() -> None:
    adata = _adata()
    registry = _run_registry(adata)

    assert len(registry.records) == 4
    assert {record.status for record in registry.records} == {"not_estimable"}
    assert all(record.reason_code for record in registry.records)
    assert all(record.workflow is None for record in registry.records)
    assert all(record.application is None for record in registry.records)


def test_graph_fused_registry_covers_training_absent_run_receiver_without_parents(
) -> None:
    adata = _adata()
    heldout_only = adata.obs["subject_id"].eq("p1") & adata.obs["cell_type"].eq(
        "Sender"
    )
    adata.obs.loc[heldout_only, "cell_type"] = "Novel"
    crossfit = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
    )
    registry = _run_registry(adata, crossfit=crossfit)

    assert registry.receiver_universe_id == crossfit.receiver_universe.universe_id
    assert registry.receiver_axis_id == crossfit.receiver_universe.receiver_axis_id
    assert registry.receiver_ids == crossfit.receiver_universe.receiver_ids
    assert registry.receiver_family_universe.receiver_ids == registry.receiver_ids
    assert registry.receiver_family_universe.family_ids == registry.family_ids
    assert registry.receiver_family_universe.family_axis_id == registry.family_axis_id
    assert len(registry.receiver_family_universe.opportunity_ids) == (
        len(registry.receiver_ids) * len(registry.family_ids)
    )
    assert registry.outer_fold_ids == tuple(
        sorted(fold.fold_id for fold in crossfit.folds)
    )
    assert len(registry.records) == (
        len(crossfit.folds) * len(crossfit.receiver_universe.receiver_ids)
    )
    for fold in crossfit.folds:
        fold_records = tuple(
            record for record in registry.records if record.fold_id == fold.fold_id
        )
        assert tuple(record.receiver for record in fold_records) == (
            crossfit.receiver_universe.receiver_ids
        )
        support_by_receiver = {
            support.receiver_id: support
            for support in fold.receiver_training_support
        }
        for record in fold_records:
            support = support_by_receiver[record.receiver]
            assert record.receiver_training_support_id == support.support_record_id
            assert record.receiver_training_support_status == support.status.value
            assert record.receiver_training_support_reason_code == support.reason_code

    absent = tuple(
        record
        for record in registry.records
        if record.receiver == "Novel"
        and record.receiver_training_support_status == "not_estimable"
    )
    assert absent
    assert {record.status for record in absent} == {"not_estimable"}
    assert {record.reason_code for record in absent} == {
        "receiver_absent_in_outer_training"
    }
    assert all(record.inner_partition_id is None for record in absent)
    assert all(record.workflow is None for record in absent)
    assert all(record.application is None for record in absent)
    absent_rows = registry.table.loc[
        registry.table["record_id"].isin(record.record_id for record in absent)
    ]
    assert absent_rows[
        ["inner_partition_id", "workflow_id", "application_id"]
    ].isna().all(axis=None)
    assert registry.to_manifest()["receiver_ids"] == list(
        crossfit.receiver_universe.receiver_ids
    )

    effects = derive_graph_fused_family_effects(crossfit, registry)
    novel = effects.family_effects.loc[
        effects.family_effects["receiver"].eq("Novel")
        & effects.family_effects["receiver_training_support_status"].eq(
            "not_estimable"
        )
    ]
    assert not novel.empty
    assert set(novel["family_id"]) == set(registry.family_ids)
    assert set(novel["status"]) == {"not_estimable"}
    assert set(novel["reason_code"]) == {
        "receiver_absent_in_outer_training"
    }
    assert novel[
        [
            "receiver_family_training_artifact_id",
            "family_basis_id",
            "workflow_id",
            "application_id",
            "problem_id",
            "fit_id",
        ]
    ].isna().all(axis=None)
    assert novel["receiver_family_opportunity_id"].notna().all()

    with pytest.raises(ContractError) as caught:
        build_graph_fused_integrated_lr_scores(
            crossfit,
            effects,
            spec=GraphFusedIntegratedLRSpec(
                context_contrasts={
                    "control": "stim_vs_control",
                    "stim": "control_vs_stim",
                }
            ),
        )
    assert caught.value.details.code == (
        "graph_fused_integrated_receiver_training_support_unavailable"
    )


def test_graph_fused_crossfit_registry_identity_is_tamper_evident() -> None:
    registry = _run_registry(_adata(tuple(f"p{index}" for index in range(1, 9))))
    object.__setattr__(registry, "registry_id", "tampered")

    with pytest.raises(ContractError) as caught:
        registry.to_manifest()

    assert caught.value.details.code == (
        "graph_fused_crossfit_registry_integrity_violation"
    )


def test_graph_family_universe_keeps_fully_predeclared_absent_receiver() -> None:
    adata = _adata()
    crossfit = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=replace(
            _spec(),
            predeclared_receiver_ids=("Ghost", "Receiver", "Sender"),
        ),
    )
    registry = _run_registry(adata, crossfit=crossfit)
    effects = derive_graph_fused_family_effects(crossfit, registry)

    assert "Ghost" in registry.receiver_family_universe.receiver_ids
    ghost_opportunities = {
        family_id: opportunity_id
        for receiver, family_id, opportunity_id in (
            registry.receiver_family_universe.opportunity_ids
        )
        if receiver == "Ghost"
    }
    assert set(ghost_opportunities) == set(registry.family_ids)
    ghost = effects.family_effects.loc[
        effects.family_effects["receiver"].eq("Ghost")
    ]
    expected_rows = sum(
        len(record.heldout_subject_ids) * len(registry.graph.nodes)
        for record in registry.records
        if record.receiver == "Ghost"
    ) * len(registry.family_ids)
    assert len(ghost) == expected_rows
    assert set(ghost["receiver_family_opportunity_id"]) == set(
        ghost_opportunities.values()
    )
    assert set(ghost["reason_code"]) == {
        "receiver_absent_in_outer_training"
    }
    assert ghost[
        [
            "receiver_family_training_artifact_id",
            "family_basis_id",
            "workflow_id",
            "application_id",
        ]
    ].isna().all(axis=None)


def test_graph_fused_crossfit_record_rejects_an_intact_application_from_another_fit(
) -> None:
    registry = _run_registry(_adata(tuple(f"p{index}" for index in range(1, 9))))
    record = registry.records[0]
    donor = next(
        item
        for item in registry.records[1:]
        if item.application is not None
        and record.application is not None
        and item.application.workflow_id != record.application.workflow_id
    )
    assert donor.application is not None

    object.__setattr__(record, "application", donor.application)
    object.__setattr__(
        record,
        "record_id",
        stable_id(
            "graph_fused_crossfit_record",
            record._identity_payload(),
            schema_version="1",
        ),
    )

    with pytest.raises(ContractError) as caught:
        record.to_dict()

    assert caught.value.details.code == "graph_fused_crossfit_record_parent_mismatch"
