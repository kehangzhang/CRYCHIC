from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from tests.unit.workflow.test_full_pipeline_resampling import (
    _bundle,
    _config,
    _prior,
    _spec,
)

from crychic.core import ContractError, SeedLineage, stable_id
from crychic.inference import (
    DifferentialContrastSpec,
    DifferentialDesignSpec,
    TwoPartOccurrenceV2Spec,
)
from crychic.resampling import build_exchangeability_map
from crychic.scoring import (
    AbsoluteActivityV2Spec,
    UncertaintyAwareHypergraphShrinkageV2Spec,
    freeze_hypergraph_prior,
)
from crychic.workflow import (
    V7EstimatorSpec,
    V7FullPipelineExecutionBackend,
    V7FullPipelineInferenceSpec,
    V7FullPipelineOperation,
    V7FullPipelineResampleStatus,
    evaluate_v7_full_pipeline_calibration,
    finalize_v7_full_pipeline_inference,
    fit_crossfit_v7_estimator,
    run_v7_full_pipeline_resampling,
)
from crychic.workflow.crossfit import (
    _run_subject_crossfit,
    _run_v7_primary_crossfit,
)
from crychic.workflow.training import _sanitized_raw_input_snapshot
from crychic.workflow.v7_full_pipeline_inference import _finalize_channel
from crychic.workflow.v7_full_pipeline_resampling import (
    _plan_design_stratified_bootstraps,
    _require_hypothesis_axis_subset,
)


def _independent_design() -> DifferentialDesignSpec:
    return DifferentialDesignSpec(
        design_kind="independent_two_group",
        condition_column="condition",
        condition_levels=("control", "stim"),
        contrasts=(
            DifferentialContrastSpec(
                name="stim_vs_control",
                weights=(("control", -1.0), ("stim", 1.0)),
            ),
        ),
        precision_weight_column=None,
        minimum_subjects_per_level=3,
    )


def _paired_design() -> DifferentialDesignSpec:
    return DifferentialDesignSpec(
        design_kind="paired",
        condition_column="condition",
        condition_levels=("control", "stim"),
        contrasts=(
            DifferentialContrastSpec(
                name="stim_vs_control",
                weights=(("control", -1.0), ("stim", 1.0)),
            ),
        ),
        precision_weight_column=None,
        minimum_subjects_per_level=4,
    )


def _continuous_design() -> DifferentialDesignSpec:
    return DifferentialDesignSpec(
        design_kind="continuous",
        condition_column="dose",
        continuous_covariates=("receiver_fraction",),
        precision_weight_column=None,
        minimum_subjects_per_level=4,
    )


def _five_subject_adata() -> AnnData:
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    names: list[str] = []
    for subject_index in range(5):
        subject = f"s{subject_index + 1}"
        for condition in ("control", "stim"):
            condition_shift = 2 if condition == "stim" else 0
            for cell_type, values in (
                ("Sender", [12 + subject_index + condition_shift, 0, 1]),
                ("Receiver", [0, 11 + subject_index, 2 + condition_shift]),
            ):
                rows.append(values)
                metadata.append(
                    {
                        "sample_id": f"{subject}-{condition}",
                        "subject_id": subject,
                        "cell_type": cell_type,
                        "condition": condition,
                    }
                )
                names.append(f"{subject}-{condition}-{cell_type}")
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.int64))
    adata = AnnData(
        X=sparse.csr_matrix(counts.shape, dtype=float),
        obs=pd.DataFrame(metadata, index=names),
        var=pd.DataFrame(index=("L", "R", "T")),
    )
    adata.layers["counts"] = counts
    return adata


def _passing_calibration_metrics() -> pd.DataFrame:
    design_kinds = (
        "continuous",
        "independent_multi_group",
        "independent_two_group",
        "multi_cohort",
        "paired",
        "repeated",
    )
    return pd.DataFrame(
        [
            {
                "scenario_id": f"null-{design_kind}",
                "design_kind": design_kind,
                "n_replicates": 1_000,
                "empirical_type_i": 0.05,
                "empirical_fdr": 0.08,
                "ci_coverage": 0.95,
            }
            for design_kind in design_kinds
        ]
    )


def test_independent_bootstrap_is_stratified_by_condition() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{condition}-{index}",
                "subject_id": f"{condition}-{index}",
                "condition": condition,
            }
            for condition in ("control", "stim")
            for index in range(3)
        ]
    )
    exchangeability = build_exchangeability_map(
        metadata,
        sample_key="sample_id",
        subject_key="subject_id",
        context_keys=("condition",),
    )
    plans = _plan_design_stratified_bootstraps(
        metadata,
        exchangeability,
        V7EstimatorSpec(design=_independent_design()),
        subject_column="subject_id",
        n_bootstraps=8,
        strata_keys=(),
        seed_lineage=SeedLineage(17),
    )
    condition_by_subject = dict(
        zip(metadata["subject_id"], metadata["condition"], strict=True)
    )

    for plan in plans:
        draw_conditions = [
            condition_by_subject[draw.source_subject_id] for draw in plan.draws
        ]
        assert draw_conditions.count("control") == 3
        assert draw_conditions.count("stim") == 3


def test_continuous_m5_accepts_the_registered_slope_contrast() -> None:
    spec = V7EstimatorSpec(
        design=_continuous_design(),
        score_head="sender_detection_raw",
        hypergraph_spec=UncertaintyAwareHypergraphShrinkageV2Spec(
            minimum_observed_edges=3
        ),
        hypergraph_contrast_name="slope:dose",
    )

    assert spec.hypergraph_contrast_name == "slope:dose"


def test_v7_primary_execution_is_exactly_equal_to_the_full_score_path() -> None:
    adata = _five_subject_adata()
    config = _config()
    crossfit_spec = replace(
        _spec(),
        absolute_activity_v2_spec=AbsoluteActivityV2Spec(),
    )
    snapshot = _sanitized_raw_input_snapshot(adata, config)
    full = _run_subject_crossfit(
        snapshot,
        config,
        _bundle(),
        _prior(),
        spec=crossfit_spec,
    )
    primary = _run_v7_primary_crossfit(
        snapshot,
        config,
        _bundle(),
        _prior(),
        spec=crossfit_spec,
    )
    sample_metadata = adata.obs.loc[
        :, ["sample_id", "subject_id", "condition"]
    ].drop_duplicates("sample_id")
    estimator_spec = V7EstimatorSpec(design=_paired_design())
    full_estimator = fit_crossfit_v7_estimator(
        full,
        estimator_spec,
        sample_metadata=sample_metadata,
    )
    primary_estimator = fit_crossfit_v7_estimator(
        primary,
        estimator_spec,
        sample_metadata=sample_metadata,
    )

    assert full.crossfit_id != primary.crossfit_id
    assert full_estimator.output_digest == primary_estimator.output_digest
    assert full_estimator.score_provenance_ids == primary_estimator.score_provenance_ids
    pd.testing.assert_frame_equal(
        full_estimator.differential.effects,
        primary_estimator.differential.effects,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        full_estimator.differential.omnibus,
        primary_estimator.differential.omnibus,
        check_exact=True,
    )
    assert all(fold.application.sender_assignments for fold in full.folds)
    assert all(not fold.application.sender_assignments for fold in primary.folds)
    assert all(not fold.receiver_family_models for fold in primary.folds)
    assert all(not fold.receiver_incremental_models for fold in primary.folds)
    assert all(not fold.family_common_functionals for fold in primary.folds)


def test_v7_full_refits_are_parallel_deterministic_and_cover_all_operations() -> None:
    crossfit_spec = replace(
        _spec(),
        absolute_activity_v2_spec=AbsoluteActivityV2Spec(),
    )
    design = _paired_design()
    estimator_spec = V7EstimatorSpec(
        design=design,
        occurrence_spec=TwoPartOccurrenceV2Spec(
            design=design,
            activity_threshold_raw=1.0,
        ),
    )
    arguments = {
        "crossfit_spec": crossfit_spec,
        "estimator_spec": estimator_spec,
        "n_bootstraps": 1,
        "n_permutations": 1,
        "run_loso": True,
    }

    serial = run_v7_full_pipeline_resampling(
        _five_subject_adata(),
        _config(),
        _bundle(),
        _prior(),
        n_jobs=1,
        **arguments,
    )
    progress: list[tuple[str, int, int]] = []
    parallel = run_v7_full_pipeline_resampling(
        _five_subject_adata(),
        _config(),
        _bundle(),
        _prior(),
        n_jobs=3,
        progress_callback=lambda record, completed, total: progress.append(
            (record.record_id, completed, total)
        ),
        **arguments,
    )
    process_progress: list[tuple[str, int, int]] = []
    process = run_v7_full_pipeline_resampling(
        _five_subject_adata(),
        _config(),
        _bundle(),
        _prior(),
        n_jobs=3,
        execution_backend=V7FullPipelineExecutionBackend.PROCESS,
        progress_callback=lambda record, completed, total: process_progress.append(
            (record.record_id, completed, total)
        ),
        **arguments,
    )

    assert serial.result_id == parallel.result_id == process.result_id
    assert serial.execution_id != parallel.execution_id
    assert parallel.execution_id != process.execution_id
    assert tuple(record.record_id for record in serial.records) == tuple(
        record.record_id for record in parallel.records
    )
    assert tuple(record.record_id for record in serial.records) == tuple(
        record.record_id for record in process.records
    )
    assert len(progress) == len(parallel.records)
    assert sorted(completed for _, completed, _ in progress) == list(
        range(1, len(parallel.records) + 1)
    )
    assert {total for _, _, total in progress} == {len(parallel.records)}
    assert sorted(completed for _, completed, _ in process_progress) == list(
        range(1, len(process.records) + 1)
    )
    assert {total for _, _, total in process_progress} == {len(process.records)}
    assert len(serial.records) == 7
    assert {record.operation for record in serial.records} == {
        V7FullPipelineOperation.SUBJECT_BOOTSTRAP,
        V7FullPipelineOperation.CONDITION_PERMUTATION,
        V7FullPipelineOperation.LEAVE_ONE_SUBJECT_OUT,
    }
    assert all(
        record.status is V7FullPipelineResampleStatus.SUCCEEDED
        for record in serial.records
    )
    for record in serial.records:
        stages = {stage for stage, _ in record.stage_lineage}
        assert {
            "fold_split",
            "availability_fitting",
            "absolute_activity_v2_fitting",
            "heldout_score_application",
            "design_aware_effect_fitting",
            "two_part_occurrence_v2_fitting",
        }.issubset(stages)
        assert record.estimator is not None
    assert not serial.continuous_effect_ledger().empty
    assert not serial.occurrence_effect_ledger().empty
    assert serial.hypergraph_effect_ledger().empty
    inference = finalize_v7_full_pipeline_inference(serial)
    assert not inference.continuous_effects.empty
    assert not inference.occurrence_effects.empty
    assert inference.hypergraph_effects.empty
    assert inference.continuous_effects["diagnostic_empirical_p_value"].notna().all()
    assert not inference.continuous_effects["formal_inference_allowed"].any()
    assert inference.continuous_effects[["p_value", "q_value"]].isna().all(axis=None)
    assert set(inference.continuous_effects["formal_inference_status"]) == {
        "calibration_gate_missing"
    }
    manifest = serial.to_manifest()
    assert manifest["full_pipeline_refit_per_resample"] is True
    assert manifest["crossfit_execution_profile"] == "v7_primary_m0_m5_v1"
    assert "family_common" in manifest["omitted_legacy_diagnostic_stages"]
    assert manifest["large_crossfit_children_retained"] is False
    assert manifest["resample_retention_policy"] == "minimal_effect_tables_only_v1"
    assert manifest["hypothesis_axis_id"] == serial.hypothesis_axis_id
    assert manifest["execution_backend"] == "serial_v1"
    assert process.to_manifest()["execution_backend"] == (
        "bounded_initialized_spawn_process_pool_v1"
    )
    assert manifest["permutation_context_keys"] == ["condition"]
    assert (
        "hypergraph_shrinkage_v2_fitting"
        not in manifest["configured_rerun_stages_observed"]
    )
    tampered = serial.records[0].estimator
    assert tampered is not None
    tampered.continuous_effects.loc[0, "effect"] += 1.0
    with pytest.raises(ValueError, match="integrity"):
        serial.to_manifest()


def test_v7_finalizer_releases_only_complete_calibrated_distributions() -> None:
    gate = evaluate_v7_full_pipeline_calibration(_passing_calibration_metrics())
    assert gate.passed
    spec = V7FullPipelineInferenceSpec(
        minimum_bootstraps=2,
        minimum_permutations=2,
    )
    point = pd.DataFrame(
        [
            {
                "event_id": "edge-1",
                "contrast_name": "stim_vs_control",
                "point_effect": 2.0,
                "point_standard_error_diagnostic": 0.5,
                "point_status": "observed",
                "point_reason_code": None,
                "source_result_id": "point-effect-1",
                "channel": "continuous_raw",
            }
        ]
    )
    operations_and_values = (
        (V7FullPipelineOperation.SUBJECT_BOOTSTRAP, 1.0),
        (V7FullPipelineOperation.SUBJECT_BOOTSTRAP, 3.0),
        (V7FullPipelineOperation.CONDITION_PERMUTATION, -0.5),
        (V7FullPipelineOperation.CONDITION_PERMUTATION, 2.5),
        (V7FullPipelineOperation.LEAVE_ONE_SUBJECT_OUT, 1.8),
        (V7FullPipelineOperation.LEAVE_ONE_SUBJECT_OUT, 2.2),
    )
    records = tuple(
        SimpleNamespace(operation=operation) for operation, _ in operations_and_values
    )
    ledger = pd.DataFrame(
        [
            {
                "operation": operation.value,
                "record_id": f"record-{index}",
                "event_id": "edge-1",
                "contrast_name": "stim_vs_control",
                "resample_effect": value,
                "resample_status": "observed",
            }
            for index, (operation, value) in enumerate(operations_and_values)
        ]
    )
    resampling = SimpleNamespace(records=records, result_id="resampling-result-1")

    result = _finalize_channel(
        point,
        ledger,
        resampling=resampling,
        spec=spec,
        gate=gate,
    ).iloc[0]

    assert result["formal_inference_allowed"]
    assert result["formal_inference_status"] == "released"
    assert result["standard_error"] == pytest.approx(np.sqrt(2.0))
    assert result["p_value"] == pytest.approx(2.0 / 3.0)
    assert result["q_value"] == pytest.approx(2.0 / 3.0)
    assert result["loso_max_abs_delta"] == pytest.approx(0.2)
    assert result["loso_sign_agreement"] == 1.0

    failing_metrics = _passing_calibration_metrics()
    failing_metrics.loc[0, "empirical_type_i"] = 0.10
    failed_gate = evaluate_v7_full_pipeline_calibration(failing_metrics)
    assert not failed_gate.passed
    assert failed_gate.failure_reasons == ("empirical_type_i_outside_0_035_0_065",)


def test_v7_full_refit_reestimates_m5_and_accepts_structural_zero_se() -> None:
    edge_rows = [
        {
            "edge_id": stable_id(
                "sample_edge_differential_event",
                {
                    "score_head": "sender_detection_raw",
                    "sender": sender,
                    "receiver": receiver,
                    "interaction_id": "lr1",
                },
                schema_version="1",
            ),
            "sender": sender,
            "receiver": receiver,
            "interaction": "lr1",
        }
        for sender in ("Receiver", "Sender")
        for receiver in ("Receiver", "Sender")
    ]
    prior = freeze_hypergraph_prior(
        pd.DataFrame(edge_rows),
        view_columns=("sender", "receiver", "interaction"),
    )
    estimator_spec = V7EstimatorSpec(
        design=_paired_design(),
        score_head="sender_detection_raw",
        hypergraph_spec=UncertaintyAwareHypergraphShrinkageV2Spec(
            minimum_observed_edges=3
        ),
        hypergraph_contrast_name="stim_vs_control",
    )

    result = run_v7_full_pipeline_resampling(
        _five_subject_adata(),
        _config(),
        _bundle(),
        _prior(),
        crossfit_spec=replace(
            _spec(),
            absolute_activity_v2_spec=AbsoluteActivityV2Spec(),
        ),
        estimator_spec=estimator_spec,
        hypergraph_prior=prior,
        n_bootstraps=1,
        run_loso=False,
    )

    point = result.point.hypergraph
    child = result.records[0].estimator
    assert point is not None and child is not None
    assert point.fit.prior_id == child.hypergraph_prior_id == prior.prior_id
    assert point.fit.fit_id != child.hypergraph_fit_id
    assert "hypergraph_shrinkage_v2_fitting" in {
        stage for stage, _ in result.records[0].stage_lineage
    }
    ledger = result.hypergraph_effect_ledger()
    assert len(ledger) == 4
    assert ledger["status"].eq("observed").all()
    assert ledger["posterior_effect"].eq(0.0).any()


def test_v7_resample_cannot_expand_the_frozen_point_hypothesis_axis() -> None:
    point_axes = (("continuous_raw", (("event-1", "contrast-1"),)),)
    expanded = SimpleNamespace(
        differential=SimpleNamespace(
            effects=pd.DataFrame(
                {
                    "event_id": ["event-1", "event-extra"],
                    "contrast_name": ["contrast-1", "contrast-1"],
                }
            )
        ),
        occurrence=None,
        hypergraph=None,
    )

    with pytest.raises(ContractError) as error:
        _require_hypothesis_axis_subset(expanded, point_axes)

    assert error.value.details.code == "v7_resample_hypothesis_axis_expansion"
