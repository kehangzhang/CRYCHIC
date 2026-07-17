from __future__ import annotations

import time
from dataclasses import replace
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import crychic.workflow.full_pipeline_resampling as resampling_module
from crychic.core import ContractError, CrychicConfig, stable_id
from crychic.design import balanced_contrast
from crychic.inference import OOFEffectSpec
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.workflow import (
    CrossFitArtifacts,
    CrossFitSpec,
    FamilyEffectTarget,
    FoldTrainingSpec,
    FrozenReceiverUniverse,
    FullPipelineResampleStatus,
    FullPipelineResamplingResult,
    FullPipelineResamplingStatus,
    full_pipeline_family_effect_records,
    run_full_pipeline_resampling,
)
from crychic.workflow.training import SanitizedRawInputSnapshot


def _adata() -> AnnData:
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    names: list[str] = []
    for subject_index, subject in enumerate(("s1", "s2", "s3", "s4")):
        for condition in ("control", "stim"):
            for cell_type, values in (
                ("Sender", [10 + subject_index, 0, 1]),
                ("Receiver", [0, 10 + subject_index, 2]),
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


def _config() -> CrychicConfig:
    return CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=31,
    )


def _bundle() -> ResourceBundle:
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
    return ResourceBundle(
        resource_id="resampling-test",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(interaction,),
        mapping_report=MappingReport(1, 1, 2),
        manifest_digest="a" * 64,
        source_files=("test.tsv",),
        license="CC0",
        citation="Synthetic test",
    )


def _prior() -> TargetPrior:
    return TargetPrior(
        resource_id="resampling-prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="interaction",
        target_ids=("T",),
        driver_ids=("lr1",),
        indptr=(0, 1),
        target_indices=(0,),
        weights=(1.0,),
        ranks=None,
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(1, 1, 2),
        manifest_digest="b" * 64,
    )


def _spec() -> CrossFitSpec:
    return CrossFitSpec(
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


def _fake_child(
    crossfit_id: str,
    *,
    config_digest: str = "synthetic-child-config",
    receiver_universe: FrozenReceiverUniverse,
) -> CrossFitArtifacts:
    child = object.__new__(CrossFitArtifacts)
    object.__setattr__(child, "crossfit_id", crossfit_id)
    object.__setattr__(child, "spec", _spec())
    object.__setattr__(
        child,
        "root_input_identity",
        SimpleNamespace(config_digest=config_digest),
    )
    object.__setattr__(
        child,
        "receiver_universe",
        receiver_universe,
    )
    object.__setattr__(
        child,
        "folds",
        (
            SimpleNamespace(
                training=SimpleNamespace(
                    resource_bundle_content_id=(
                        resampling_module._resource_bundle_content_id(_bundle())
                    ),
                    target_prior_content_id=(
                        resampling_module._target_prior_content_id(_prior())
                    ),
                )
            ),
        ),
    )
    return child


def _resampled_receiver_universe(
    adata: AnnData,
    config: CrychicConfig,
    source: FrozenReceiverUniverse,
) -> FrozenReceiverUniverse:
    snapshot = resampling_module._sanitized_raw_input_snapshot(adata, config)
    observed = tuple(
        sorted(snapshot.adata.obs[config.cell_type_key].astype(str).unique())
    )
    return resampling_module.freeze_receiver_universe(
        snapshot.identity,
        observed,
        predeclared_receiver_ids=source.receiver_ids,
    )


def test_every_materialized_resample_calls_full_runner_and_records_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _adata()
    original_obs = source.obs.copy(deep=True)
    calls: list[tuple[AnnData, CrychicConfig]] = []

    def runner(
        snapshot: SanitizedRawInputSnapshot,
        config: CrychicConfig,
        resource_bundle: ResourceBundle,
        target_prior: TargetPrior,
        *,
        spec: CrossFitSpec,
        _receiver_axis_source: FrozenReceiverUniverse | None = None,
    ) -> CrossFitArtifacts:
        assert (
            resource_bundle is not None
            and target_prior is not None
            and spec is not None
        )
        assert _receiver_axis_source is not None
        adata = snapshot.adata
        calls.append((adata.copy(), config))
        if len(calls) == 2:
            raise ContractError(
                "synthetic runner failure",
                code="synthetic_full_pipeline_failure",
                field="runner",
            )
        child_id = stable_id(
            "test_crossfit",
            {"obs_names": list(map(str, adata.obs_names))},
        )
        return _fake_child(
            child_id,
            config_digest=config.digest,
            receiver_universe=_resampled_receiver_universe(
                adata,
                config,
                _receiver_axis_source,
            ),
        )

    monkeypatch.setattr(resampling_module, "_run_subject_crossfit", runner)
    result = run_full_pipeline_resampling(
        source,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
        n_bootstraps=1,
        n_permutations=1,
    )

    assert len(calls) == 2
    assert [record.status for record in result.records] == [
        FullPipelineResampleStatus.SUCCEEDED,
        FullPipelineResampleStatus.FAILED,
    ]
    assert result.status is FullPipelineResamplingStatus.PARTIAL_FAILURE
    assert result.records[1].failure_code == "synthetic_full_pipeline_failure"
    assert result.children == ()
    assert all(record.child is None for record in result.records)
    assert calls[0][1].random_seed != calls[1][1].random_seed

    bootstrap = calls[0][0]
    assert bootstrap.obs_names.is_unique
    assert set(bootstrap.obs["subject_id"]).isdisjoint(set(original_obs["subject_id"]))
    assert bootstrap.obs.groupby("subject_id", observed=True).size().eq(4).all()
    assert (
        bootstrap.obs.groupby("subject_id", observed=True)["sample_id"]
        .nunique()
        .eq(2)
        .all()
    )
    permutation = calls[1][0]
    assert (
        permutation.obs.groupby("sample_id", observed=True)["condition"]
        .nunique()
        .eq(1)
        .all()
    )
    pd.testing.assert_frame_equal(source.obs, original_obs)

    manifest = result.to_manifest()
    assert manifest["full_pipeline_refit_per_resample"] is True
    assert manifest["score_table_relabeling_only"] is False
    assert manifest["retain_children"] is False
    assert {
        "nuisance_response_fitting",
        "feature_and_interaction_filtering",
        "hyperparameter_tuning",
        "receiver_attribution",
        "sender_assignment",
        "family_common_scoring",
    }.issubset(manifest["rerun_stages"])
    assert manifest["inferential_fields_available"] == []
    assert set(manifest["inferential_fields_unavailable"]) == {
        "p_value",
        "q_value",
        "communication_probability",
        "posterior_probability",
    }


def test_bounded_resampling_parallelism_preserves_scientific_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker_lock = Lock()
    inflight = 0
    maximum_inflight = 0

    def runner(
        snapshot: SanitizedRawInputSnapshot,
        config: CrychicConfig,
        resource_bundle: ResourceBundle,
        target_prior: TargetPrior,
        *,
        spec: CrossFitSpec,
        _receiver_axis_source: FrozenReceiverUniverse | None = None,
    ) -> CrossFitArtifacts:
        nonlocal inflight, maximum_inflight
        del resource_bundle, target_prior, spec
        assert _receiver_axis_source is not None
        with tracker_lock:
            inflight += 1
            maximum_inflight = max(maximum_inflight, inflight)
        try:
            time.sleep(0.03)
            child_id = stable_id(
                "parallel_resampling_test_child",
                {
                    "config_digest": config.digest,
                    "obs_names": list(map(str, snapshot.adata.obs_names)),
                },
            )
            return _fake_child(
                child_id,
                config_digest=config.digest,
                receiver_universe=_resampled_receiver_universe(
                    snapshot.adata,
                    config,
                    _receiver_axis_source,
                ),
            )
        finally:
            with tracker_lock:
                inflight -= 1

    monkeypatch.setattr(resampling_module, "_run_subject_crossfit", runner)
    kwargs = {
        "spec": _spec(),
        "n_bootstraps": 2,
        "n_permutations": 2,
    }
    serial = run_full_pipeline_resampling(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        n_jobs=1,
        **kwargs,
    )
    with tracker_lock:
        maximum_inflight = 0
    parallel = run_full_pipeline_resampling(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        n_jobs=10,
        **kwargs,
    )

    assert serial.result_id == parallel.result_id
    assert tuple(record.record_id for record in serial.records) == tuple(
        record.record_id for record in parallel.records
    )
    assert serial.requested_n_jobs == serial.effective_n_jobs == 1
    assert parallel.requested_n_jobs == 10
    assert parallel.effective_n_jobs == 4
    assert serial.execution_metadata_id != parallel.execution_metadata_id
    assert serial.to_manifest()["execution_backend"] == "serial_v1"
    assert parallel.to_manifest()["execution_backend"] == (
        "bounded_shared_snapshot_thread_pool_v1"
    )
    assert parallel.to_manifest()["execution_metadata_id"] == (
        parallel.execution_metadata_id
    )
    assert maximum_inflight >= 2

    scientific_result_id = parallel.result_id
    object.__setattr__(parallel, "requested_n_jobs", 2)
    with pytest.raises(ContractError) as error:
        parallel._require_intact()
    assert error.value.details.code == "full_pipeline_resampling_integrity_violation"
    assert parallel.result_id == scientific_result_id


@pytest.mark.parametrize("n_jobs", (0, -1, True))  # type: ignore[untyped-decorator]
def test_full_pipeline_resampling_rejects_invalid_n_jobs(n_jobs: int) -> None:
    with pytest.raises(ValueError, match="n_jobs must be an integer >= 1"):
        run_full_pipeline_resampling(
            _adata(),
            _config(),
            _bundle(),
            _prior(),
            spec=_spec(),
            n_bootstraps=1,
            n_jobs=n_jobs,
        )


def test_retain_children_is_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children: list[CrossFitArtifacts] = []

    def runner(
        snapshot: SanitizedRawInputSnapshot,
        config: CrychicConfig,
        resource_bundle: ResourceBundle,
        target_prior: TargetPrior,
        *,
        spec: CrossFitSpec,
        _receiver_axis_source: FrozenReceiverUniverse | None = None,
    ) -> CrossFitArtifacts:
        del resource_bundle, target_prior, spec
        assert _receiver_axis_source is not None
        adata = snapshot.adata
        child = _fake_child(
            "crossfit-retained",
            config_digest=config.digest,
            receiver_universe=_resampled_receiver_universe(
                adata,
                config,
                _receiver_axis_source,
            ),
        )
        children.append(child)
        return child

    monkeypatch.setattr(
        resampling_module,
        "_run_subject_crossfit",
        runner,
    )
    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", lambda self: None)

    result = run_full_pipeline_resampling(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
        n_bootstraps=1,
        retain_children=True,
    )

    assert result.children == tuple(children)
    assert result.records[0].child is children[0]
    assert result.to_manifest()["child_retention_policy"] == (
        "retain_successful_crossfit_children_v1"
    )
    with pytest.raises(ContractError) as lineage_error:
        replace(result, resource_bundle_content_id="forged-resource")
    assert lineage_error.value.details.code == (
        "full_pipeline_resampling_child_lineage_mismatch"
    )


def test_bootstrap_keeps_source_receiver_axis_when_rare_receiver_is_not_drawn() -> None:
    adata = _adata()
    rare_rows = adata.obs["subject_id"].eq("s3") & adata.obs["cell_type"].eq("Receiver")
    adata.obs.loc[rare_rows, "cell_type"] = "RareReceiver"

    result = run_full_pipeline_resampling(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
        n_bootstraps=1,
        retain_children=True,
    )

    assert result.status is FullPipelineResamplingStatus.SUCCEEDED
    record = result.records[0]
    child = record.child
    binding = record.receiver_universe_reuse_binding
    assert child is not None and binding is not None
    assert result.to_manifest()["source_receiver_ids"] == [
        "RareReceiver",
        "Receiver",
        "Sender",
    ]
    assert child.receiver_universe.receiver_ids == (
        "RareReceiver",
        "Receiver",
        "Sender",
    )
    assert "RareReceiver" not in child.receiver_universe.observed_cell_type_ids
    assert child.receiver_universe.universe_id != result.source_receiver_universe_id
    assert child.receiver_universe.receiver_axis_id == result.source_receiver_axis_id
    assert binding.child_receiver_universe_id == child.receiver_universe.universe_id
    assert binding.receiver_axis_id == result.source_receiver_axis_id
    record_manifest = record.to_dict()
    result_manifest = result.to_manifest()
    assert record_manifest["receiver_universe_reuse_binding"] == binding.to_dict()
    assert result_manifest["source_receiver_universe"] == (
        result._source_receiver_universe.to_dict()
    )
    assert all(
        next(
            support
            for support in fold.receiver_training_support
            if support.receiver_id == "RareReceiver"
        ).reason_code
        == "receiver_absent_in_outer_training"
        for fold in child.folds
    )
    assert all(
        model.receiver_family_artifact.receiver != "RareReceiver"
        for fold in child.folds
        for model in fold.receiver_family_models
    )
    rare_coverage = child.oof_receiver_coverage.loc[
        child.oof_receiver_coverage["receiver"].eq("RareReceiver")
    ]
    assert not rare_coverage.empty
    assert set(rare_coverage["receiver_training_support_reason_code"]) == {
        "receiver_absent_in_outer_training"
    }
    assert rare_coverage["response_artifact_id"].isna().all()
    assert rare_coverage["incremental_training_artifact_id"].isna().all()

    target = FamilyEffectTarget(
        contrast_name="stim_vs_control",
        receiver="RareReceiver",
        family_id="predeclared-rare-family",
        mode="state",
    )
    effect_spec = OOFEffectSpec(
        hypothesis_id=target.hypothesis_id,
        contrast_name=target.contrast_name,
        contrast_weights=(("stim", 1.0), ("control", -1.0)),
    )
    effect_records = full_pipeline_family_effect_records(
        result,
        target,
        effect_spec,
    )
    assert len(effect_records) == 1
    assert effect_records[0].status.value == "not_estimable"
    assert effect_records[0].reason_code == ("receiver_absent_in_outer_training")


def test_failed_workflow_record_maps_to_neutral_failed_effect_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_runner(*args, **kwargs):
        raise ContractError(
            "synthetic full-pipeline failure",
            code="synthetic_pipeline_failure",
            field="runner",
        )

    monkeypatch.setattr(resampling_module, "_run_subject_crossfit", failed_runner)
    result = run_full_pipeline_resampling(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
        n_bootstraps=1,
        retain_children=True,
    )
    target = FamilyEffectTarget(
        contrast_name="stim_vs_control",
        receiver="Receiver",
        family_id="family-placeholder",
        mode="state",
    )
    effect_spec = OOFEffectSpec(
        hypothesis_id=target.hypothesis_id,
        contrast_name=target.contrast_name,
        contrast_weights=(("stim", 1.0), ("control", -1.0)),
    )

    records = full_pipeline_family_effect_records(result, target, effect_spec)

    assert len(records) == 1
    assert records[0].status.value == "failed"
    assert records[0].reason_code == "synthetic_pipeline_failure"
    assert records[0].full_pipeline_record_id == result.records[0].record_id


def test_resampling_manifest_rejects_tampering_and_duplicate_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def runner(
        snapshot: SanitizedRawInputSnapshot,
        config: CrychicConfig,
        *args: object,
        _receiver_axis_source: FrozenReceiverUniverse | None = None,
        **kwargs: object,
    ) -> CrossFitArtifacts:
        del args, kwargs
        assert _receiver_axis_source is not None
        adata = snapshot.adata
        return _fake_child(
            "crossfit-integrity",
            config_digest=config.digest,
            receiver_universe=_resampled_receiver_universe(
                adata,
                config,
                _receiver_axis_source,
            ),
        )

    monkeypatch.setattr(
        resampling_module,
        "_run_subject_crossfit",
        runner,
    )
    result = run_full_pipeline_resampling(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
        n_bootstraps=1,
    )
    with pytest.raises(ContractError) as duplicate_error:
        FullPipelineResamplingResult(
            exchangeability=result.exchangeability,
            plans=(result.plans[0], result.plans[0]),
            records=(result.records[0], result.records[0]),
            source_input_identity_id=result.source_input_identity_id,
            source_input_digest=result.source_input_digest,
            source_snapshot_id=result.source_snapshot_id,
            config_digest=result.config_digest,
            crossfit_spec_id=result.crossfit_spec_id,
            resource_bundle_content_id=result.resource_bundle_content_id,
            target_prior_content_id=result.target_prior_content_id,
            root_seed_lineage=result.root_seed_lineage,
            retain_children=False,
            source_receiver_universe_id=result.source_receiver_universe_id,
            source_receiver_axis_id=result.source_receiver_axis_id,
            _source_receiver_universe=result._source_receiver_universe,
        )
    assert duplicate_error.value.details.code == (
        "duplicate_full_pipeline_resampling_plan"
    )

    tampered = replace(result)
    object.__setattr__(tampered.records[0], "plan_id", "forged-plan")
    with pytest.raises(ContractError) as integrity_error:
        tampered.to_manifest()
    assert integrity_error.value.details.code == (
        "full_pipeline_resampling_integrity_violation"
    )
