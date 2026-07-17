from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import crychic.workflow.active_probability_pipeline as pipeline_module
from crychic.core import (
    CommunicationMode,
    ContractError,
    CrychicConfig,
    SeedLineage,
)
from crychic.design import balanced_contrast
from crychic.inference import (
    ActiveEdgeScoreStatus,
    ActiveProbabilitySpec,
    NullActiveEdgeScoreRecord,
    NullScoreDistribution,
    PointActiveEdgeScoreRecord,
    estimate_active_probabilities,
)
from crychic.resampling import ActiveNullSpec
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.scoring import (
    ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
    ActiveEdgeCandidate,
    freeze_active_edge_universe,
)
from crychic.workflow.active_edge_scores import CrossFitActiveEdgePointRecords
from crychic.workflow.active_null_rerun import (
    ActiveNullRerunResult,
    ActiveNullRerunStatus,
)
from crychic.workflow.active_probability_pipeline import (
    ActiveProbabilityPipelineLineage,
    ActiveProbabilityPipelineResult,
    run_active_probability_pipeline,
)
from crychic.workflow.crossfit import CrossFitArtifacts, CrossFitSpec
from crychic.workflow.training import FoldTrainingSpec

_SCORE_VERSION = "receiver_gain_percentile_mechanistic_conserved_sender_v4"


def _inputs() -> tuple[
    AnnData,
    CrychicConfig,
    ResourceBundle,
    TargetPrior,
    CrossFitSpec,
]:
    adata = AnnData(
        X=np.zeros((1, 3), dtype=float),
        obs=pd.DataFrame(
            {
                "sample_id": ["sample-1"],
                "subject_id": ["subject-1"],
                "cell_type": ["Sender"],
                "condition": ["control"],
            },
            index=["cell-1"],
        ),
        var=pd.DataFrame(index=["L", "R", "T"]),
    )
    adata.layers["counts"] = np.zeros((1, 3), dtype=np.int64)
    config = CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=901,
    )
    interaction = Interaction(
        interaction_id="lr1",
        source_interaction_id="lr1",
        ligand_name="L",
        receptor_name="R",
        ligand_subunits=("L",),
        receptor_subunits=("R",),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="test",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    bundle = ResourceBundle(
        resource_id="active-probability-pipeline-test",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(interaction,),
        mapping_report=MappingReport(1, 1, 2),
        manifest_digest="resource-manifest",
        source_files=("test.tsv",),
        license="CC0",
        citation="Synthetic test",
    )
    prior = TargetPrior(
        resource_id="active-probability-pipeline-prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=("T",),
        driver_ids=("L",),
        indptr=(0, 1),
        target_indices=(0,),
        weights=(1.0,),
        ranks=(1,),
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(1, 1, 2),
        manifest_digest="prior-manifest",
    )
    crossfit_spec = CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("stim",),
                ("control",),
                name="stim_vs_control",
            ),
        ),
        training_spec=FoldTrainingSpec(min_cells=1, max_interactions=1),
        allowed_n_splits=(2,),
    )
    return adata, config, bundle, prior, crossfit_spec


def _point_artifacts(crossfit_spec: CrossFitSpec) -> CrossFitArtifacts:
    artifacts = object.__new__(CrossFitArtifacts)
    object.__setattr__(artifacts, "crossfit_id", "point-crossfit")
    object.__setattr__(artifacts, "receiver_scoring_registry_id", "point-registry")
    object.__setattr__(artifacts, "spec", crossfit_spec)
    object.__setattr__(
        artifacts,
        "root_input_identity",
        SimpleNamespace(
            identity_id="input-identity",
            input_digest="input-digest",
            config_digest="config-digest",
        ),
    )
    object.__setattr__(
        artifacts,
        "fold_plan",
        SimpleNamespace(plan_id="fold-plan"),
    )
    return artifacts


def _point_components(
    artifacts: CrossFitArtifacts,
) -> tuple[Any, CrossFitActiveEdgePointRecords]:
    candidate = ActiveEdgeCandidate(
        contrast_id="contrast-manifest",
        context_id="condition=stim",
        sender="Sender",
        receiver="Receiver",
        interaction_id="lr1",
        driver_id="L",
        mode=CommunicationMode.STATE,
    )
    universe = freeze_active_edge_universe(
        (candidate,),
        universe_name="pipeline-test-universe",
        contrast_id=candidate.contrast_id,
        score_version=_SCORE_VERSION,
    )
    point = PointActiveEdgeScoreRecord(
        candidate_edge_id=candidate.candidate_edge_id,
        stratum_id=candidate.stratum_id(score_version=_SCORE_VERSION),
        score_version=_SCORE_VERSION,
        source_score_collection_id="point-score-source",
        n_subjects=4,
        score=0.8,
        status=ActiveEdgeScoreStatus.OBSERVED,
        reason_code=None,
    )
    collection = object.__new__(CrossFitActiveEdgePointRecords)
    values = {
        "source_crossfit_id": artifacts.crossfit_id,
        "source_oof_audit_id": "point-oof-audit",
        "source_registry_id": artifacts.receiver_scoring_registry_id,
        "active_edge_universe_id": universe.universe_id,
        "contrast_id": universe.contrast_id,
        "contrast_name": "stim_vs_control",
        "repeat_id": artifacts.spec.repeat_id,
        "score_version": _SCORE_VERSION,
        "score_spec_id": "score-spec",
        "source_score_collection_id": "point-score-source",
        "source_functional_ids": ("functional-1",),
        "source_application_ids": ("application-1",),
        "source_sender_score_digests": ("sender-digest-1",),
        "records": (point,),
        "collection_id": "point-collection",
        "_producer_marker": "test-producer",
    }
    for name, value in values.items():
        object.__setattr__(collection, name, value)
    return universe, collection


def _null_result(
    point_artifacts: CrossFitArtifacts,
    universe: Any,
    point_records: CrossFitActiveEdgePointRecords,
    *,
    null_status: ActiveEdgeScoreStatus = ActiveEdgeScoreStatus.OBSERVED,
    source_point_collection_id: str | None = None,
) -> ActiveNullRerunResult:
    plan_id = "null-plan-0"
    rerun_id = "null-rerun-record-0"
    if null_status is ActiveEdgeScoreStatus.OBSERVED:
        null_record = NullActiveEdgeScoreRecord(
            candidate_edge_id=point_records.records[0].candidate_edge_id,
            plan_id=plan_id,
            stratum_id=point_records.records[0].stratum_id,
            score_version=_SCORE_VERSION,
            null_rerun_record_id=rerun_id,
            source_score_collection_id="null-score-source",
            n_subjects=4,
            score=0.2,
            status=null_status,
            reason_code=None,
        )
        rerun_status = ActiveNullRerunStatus.SUCCEEDED
        rerun_reason = None
    else:
        null_record = NullActiveEdgeScoreRecord(
            candidate_edge_id=point_records.records[0].candidate_edge_id,
            plan_id=plan_id,
            stratum_id=point_records.records[0].stratum_id,
            score_version=_SCORE_VERSION,
            null_rerun_record_id=rerun_id,
            source_score_collection_id=None,
            n_subjects=None,
            score=None,
            status=null_status,
            reason_code="synthetic_null_failure",
        )
        rerun_status = ActiveNullRerunStatus.FAILED
        rerun_reason = "synthetic_null_failure"
    distribution = NullScoreDistribution(
        active_null_id="active-null-run",
        active_null_spec_id=ActiveNullSpec().active_null_spec_id,
        candidate_universe_id=universe.universe_id,
        candidate_universe_policy_id=ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
        score_spec_id=point_records.score_spec_id,
        point_records=point_records.records,
        plan_ids=(plan_id,),
        null_records=(null_record,),
    )
    result = object.__new__(ActiveNullRerunResult)
    values = {
        "source_point_crossfit_id": point_artifacts.crossfit_id,
        "source_point_collection_id": (
            point_records.collection_id
            if source_point_collection_id is None
            else source_point_collection_id
        ),
        "source_point_oof_audit_id": point_records.source_oof_audit_id,
        "source_point_registry_id": point_records.source_registry_id,
        "source_point_score_collection_id": (point_records.source_score_collection_id),
        "source_input_identity_id": "input-identity",
        "source_input_digest": "input-digest",
        "source_snapshot_id": "input-snapshot",
        "config_digest": "config-digest",
        "crossfit_spec_id": point_artifacts.spec.spec_id,
        "repeat_id": point_artifacts.spec.repeat_id,
        "fold_plan_id": "fold-plan",
        "crossfit_seed_lineage": SeedLineage(901).derive("point-crossfit"),
        "partition_seed_lineage": None,
        "root_seed_lineage": SeedLineage(901).derive("active-null"),
        "resource_bundle_content_id": "resource-content",
        "source_target_prior_content_id": "prior-content",
        "candidate_universe_id": universe.universe_id,
        "score_version": point_records.score_version,
        "score_spec_id": point_records.score_spec_id,
        "active_null_spec_id": ActiveNullSpec().active_null_spec_id,
        "active_null_id": distribution.active_null_id,
        "requests": (SimpleNamespace(plan_id=plan_id),),
        "records": (
            SimpleNamespace(
                null_rerun_record_id=rerun_id,
                status=rerun_status,
                reason_code=rerun_reason,
            ),
        ),
        "distribution": distribution,
        "retain_children": False,
        "requested_n_jobs": 1,
        "effective_n_jobs": 1,
        "result_id": "active-null-result",
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _patch_composition(
    monkeypatch: pytest.MonkeyPatch,
    *,
    null_status: ActiveEdgeScoreStatus = ActiveEdgeScoreStatus.OBSERVED,
    source_point_collection_id: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    _, _, _, _, crossfit_spec = _inputs()
    artifacts = _point_artifacts(crossfit_spec)
    universe, points = _point_components(artifacts)
    null = _null_result(
        artifacts,
        universe,
        points,
        null_status=null_status,
        source_point_collection_id=source_point_collection_id,
    )
    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", lambda self: None)
    monkeypatch.setattr(
        CrossFitActiveEdgePointRecords,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(ActiveNullRerunResult, "_require_intact", lambda self: None)
    monkeypatch.setattr(
        ActiveNullRerunResult,
        "to_manifest",
        lambda self: {"execution_backend": "serial_v1"},
    )
    calls: list[str] = []

    def run_point(*args: Any, **kwargs: Any) -> CrossFitArtifacts:
        calls.append("point")
        assert kwargs == {"spec": crossfit_spec}
        return artifacts

    def freeze(*args: Any, **kwargs: Any) -> Any:
        calls.append("freeze")
        assert args == (artifacts,)
        assert kwargs == {
            "contrast_id_or_name": "stim_vs_control",
            "universe_name": None,
        }
        return universe

    def adapt(*args: Any, **kwargs: Any) -> CrossFitActiveEdgePointRecords:
        calls.append("adapt")
        assert args == (artifacts, universe)
        assert not kwargs
        return points

    def rerun(*args: Any, **kwargs: Any) -> ActiveNullRerunResult:
        calls.append("null")
        assert args[-2:] == (artifacts, universe)
        assert kwargs["n_plans"] == 1
        assert kwargs["n_jobs"] == 1
        assert kwargs["retain_children"] is False
        assert isinstance(kwargs["active_null_spec"], ActiveNullSpec)
        return null

    def infer(*args: Any, **kwargs: Any) -> Any:
        calls.append("probability")
        return estimate_active_probabilities(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_subject_crossfit", run_point)
    monkeypatch.setattr(
        pipeline_module,
        "freeze_crossfit_active_edge_universe",
        freeze,
    )
    monkeypatch.setattr(
        pipeline_module,
        "adapt_crossfit_active_edge_point_records",
        adapt,
    )
    monkeypatch.setattr(pipeline_module, "run_active_null_reruns", rerun)
    monkeypatch.setattr(pipeline_module, "estimate_active_probabilities", infer)
    return calls, {
        "artifacts": artifacts,
        "universe": universe,
        "points": points,
        "null": null,
        "crossfit_spec": crossfit_spec,
    }


def test_pipeline_composes_exact_children_with_stable_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, children = _patch_composition(monkeypatch)
    adata, config, bundle, prior, _ = _inputs()

    result = run_active_probability_pipeline(
        adata,
        config,
        bundle,
        prior,
        crossfit_spec=children["crossfit_spec"],
        contrast_id_or_name="stim_vs_control",
        n_plans=1,
    )

    assert calls == ["point", "freeze", "adapt", "null", "probability"]
    assert result.point_artifacts is children["artifacts"]
    assert result.universe is children["universe"]
    assert result.point_records is children["points"]
    assert result.null_rerun is children["null"]
    assert result.lineage.point_crossfit_id == children["artifacts"].crossfit_id
    assert result.lineage.point_collection_id == children["points"].collection_id
    assert result.lineage.null_distribution_id == (
        children["null"].distribution.distribution_id
    )
    assert result.lineage.probability_collection_id == (
        result.probability_collection.collection_id
    )
    assert result.calibration_gate is None
    assert result.comm_probability_release_allowed is False
    assert result.probability_collection.records[0].comm_probability is None
    manifest = result.to_manifest()
    assert manifest["pipeline_id"] == result.pipeline_id
    assert manifest["comm_probability_release_allowed"] is False
    assert manifest["retain_null_children"] is False
    assert manifest["active_null_execution_backend"] == "serial_v1"
    assert manifest["active_null_requested_n_jobs"] == 1
    assert manifest["active_null_effective_n_jobs"] == 1
    assert manifest["n_candidates"] == 1
    assert manifest["n_null_plans"] == 1

    with pytest.raises(TypeError, match="producer-owned"):
        ActiveProbabilityPipelineResult()
    with pytest.raises(TypeError, match="producer-owned"):
        ActiveProbabilityPipelineLineage()


def test_typed_null_failure_reaches_probability_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, children = _patch_composition(
        monkeypatch,
        null_status=ActiveEdgeScoreStatus.FAILED,
    )
    adata, config, bundle, prior, _ = _inputs()

    result = run_active_probability_pipeline(
        adata,
        config,
        bundle,
        prior,
        crossfit_spec=children["crossfit_spec"],
        contrast_id_or_name="stim_vs_control",
        n_plans=1,
    )

    record = result.probability_collection.records[0]
    assert result.null_rerun.status.value == "failed"
    assert record.active_null_empirical_p_value is None
    assert record.candidate_comm_probability is None
    assert record.comm_probability is None
    assert record.probability_reason_code == "active_null_incomplete_score_matrix"


def test_pipeline_rejects_point_null_lineage_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, children = _patch_composition(
        monkeypatch,
        source_point_collection_id="foreign-point-collection",
    )
    adata, config, bundle, prior, _ = _inputs()

    with pytest.raises(ValueError, match="point and active-null artifacts"):
        run_active_probability_pipeline(
            adata,
            config,
            bundle,
            prior,
            crossfit_spec=children["crossfit_spec"],
            contrast_id_or_name="stim_vs_control",
            n_plans=1,
        )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "n_plans", (0, -1, True)
)
def test_invalid_plan_count_fails_before_point_crossfit(
    monkeypatch: pytest.MonkeyPatch,
    n_plans: Any,
) -> None:
    adata, config, bundle, prior, crossfit_spec = _inputs()
    called = False

    def point(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("point cross-fit must not run")

    monkeypatch.setattr(pipeline_module, "run_subject_crossfit", point)
    with pytest.raises(ValueError, match="n_plans"):
        run_active_probability_pipeline(
            adata,
            config,
            bundle,
            prior,
            crossfit_spec=crossfit_spec,
            contrast_id_or_name="stim_vs_control",
            n_plans=n_plans,
        )
    assert called is False


def test_pipeline_has_no_caller_probability_release_switch() -> None:
    parameters = inspect.signature(run_active_probability_pipeline).parameters
    assert "release" not in parameters
    assert "release_probability" not in parameters
    assert parameters["calibration_gate"].default is None
    assert parameters["n_jobs"].default == 1
    assert parameters["retain_children"].default is False


def test_lineage_tampering_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, children = _patch_composition(monkeypatch)
    adata, config, bundle, prior, _ = _inputs()
    result = run_active_probability_pipeline(
        adata,
        config,
        bundle,
        prior,
        crossfit_spec=children["crossfit_spec"],
        contrast_id_or_name="stim_vs_control",
        n_plans=1,
    )
    object.__setattr__(result.lineage, "point_collection_id", "tampered")

    with pytest.raises(ContractError) as error:
        result._require_intact()
    assert error.value.details.code == "active_probability_pipeline_integrity_violation"


def test_probability_spec_is_frozen_before_point_crossfit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adata, config, bundle, prior, crossfit_spec = _inputs()
    probability_spec = ActiveProbabilitySpec()
    object.__setattr__(probability_spec, "minimum_null_plans", 1)
    called = False

    def point(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("point cross-fit must not run")

    monkeypatch.setattr(pipeline_module, "run_subject_crossfit", point)
    with pytest.raises(ContractError) as error:
        run_active_probability_pipeline(
            adata,
            config,
            bundle,
            prior,
            crossfit_spec=crossfit_spec,
            contrast_id_or_name="stim_vs_control",
            n_plans=1,
            active_probability_spec=probability_spec,
        )
    assert error.value.details.code == "active_probability_spec_integrity_violation"
    assert called is False
