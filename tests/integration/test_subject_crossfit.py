from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import crychic.response.autonomous as autonomous_module
import crychic.scoring.receiver_family as receiver_scoring_module
import crychic.sender.common as common_sender_module
import crychic.workflow.crossfit as crossfit_module
import crychic.workflow.training as training_module
from crychic.attribution import PenaltyTuningSpec
from crychic.core import ContractError, CrychicConfig
from crychic.design import balanced_contrast
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    ResourceManifest,
    Species,
    TargetPrior,
)
from crychic.resources.autonomous_registry import (
    ReceiverAutonomousProgramRegistration,
)
from crychic.response import (
    build_receiver_autonomous_program_resource,
    load_receiver_autonomous_program_resource,
)
from crychic.scoring import (
    mark_family_common_scoring_application_not_estimable,
    mark_family_common_scoring_not_estimable,
    mark_receiver_program_application_not_estimable,
)
from crychic.sender import CommonSenderApplication, ContrastCommonSenderParameters
from crychic.workflow import (
    CrossFitArtifacts,
    CrossFitSpec,
    FoldTrainingSpec,
    run_subject_crossfit,
)


def _interaction(interaction_id: str, ligand: str, receptor: str) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=interaction_id,
        ligand_name=ligand,
        receptor_name=receptor,
        ligand_subunits=(ligand,),
        receptor_subunits=(receptor,),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="crossfit-fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _bundle() -> ResourceBundle:
    return ResourceBundle(
        resource_id="crossfit_lr",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(
            _interaction("i1", "L1", "R1"),
            _interaction("i2", "L2", "R2"),
        ),
        mapping_report=MappingReport(2, 2, 4),
        manifest_digest="c" * 64,
        source_files=("crossfit.tsv",),
        license="CC0",
        citation="Synthetic cross-fit fixture",
    )


def _prior() -> TargetPrior:
    return TargetPrior(
        resource_id="crossfit_prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="interaction",
        target_ids=("T1", "T2"),
        driver_ids=("i1", "i2"),
        indptr=(0, 1, 2),
        target_indices=(0, 1),
        weights=(1.0, 1.0),
        ranks=None,
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(2, 2, 4),
        manifest_digest="d" * 64,
    )


def _trusted_target_resource(root: Path, monkeypatch: pytest.MonkeyPatch):
    payload = b"feature_id\tgeneric_program\nT1\t1\nT2\t1\n"
    (root / "programs.tsv").write_bytes(payload)
    record = {
        "resource_id": "crossfit_trusted_autonomous_programs",
        "version": "1",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "source_url": "https://example.invalid/crossfit-autonomous-programs",
        "license": "CC0-1.0",
        "citation": "Synthetic static program used only by integration tests.",
        "retrieved_at": "2026-07-14",
        "adapter_version": "crychic-receiver-autonomous-feature-program-tsv-v1",
        "payloads": [
            {
                "path": "programs.tsv",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "role": "receiver_autonomous_feature_by_program_matrix_v1",
                "bytes": len(payload),
            }
        ],
        "transformation_log": ["Defined independently of expression data."],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(record), encoding="utf-8")
    manifest_digest = ResourceManifest.from_json(manifest_path).digest
    matrix_digest = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("T1", "T2"),
        program_ids=("generic_program",),
        resource_id="digest-only",
        version="1",
        manifest_digest="0" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    ).matrix_digest
    registration = ReceiverAutonomousProgramRegistration(
        registration_id="test.crossfit.trusted.autonomous.v1",
        manifest_digest=manifest_digest,
        resource_id="crossfit_trusted_autonomous_programs",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        expected_license="CC0-1.0",
        adapter_version="crychic-receiver-autonomous-feature-program-tsv-v1",
        review_scope="synthetic_benchmark_only",
        payload_path="programs.tsv",
        payload_role="receiver_autonomous_feature_by_program_matrix_v1",
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=len(payload),
        expected_matrix_digest=matrix_digest,
    )
    monkeypatch.setattr(
        autonomous_module,
        "require_receiver_autonomous_program_registration",
        lambda registration_id: registration,
    )
    return load_receiver_autonomous_program_resource(
        root,
        manifest_path=manifest_path,
        registration_id=registration.registration_id,
    )


def _adata(
    subjects: tuple[str, ...] = ("p1", "p2", "p3", "p4"),
) -> AnnData:
    genes = ("L1", "R1", "L2", "R2", "T1", "T2")
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    obs_names: list[str] = []
    for subject_index, subject in enumerate(subjects):
        high = 70 + 5 * subject_index
        low = 2 + subject_index
        for condition in ("control", "stim"):
            sample = f"sample-{subject}-{condition}"
            for cell_type, profile in (
                ("Sender", [high, 0, low, 0, 1, 1]),
                ("Receiver", [0, high, 0, low, 1, 1]),
            ):
                for cell_index in range(3):
                    rows.append(profile)
                    metadata.append(
                        {
                            "sample_id": sample,
                            "subject_id": subject,
                            "cell_type": cell_type,
                            "condition": condition,
                        }
                    )
                    obs_names.append(f"{subject}-{condition}-{cell_type}-{cell_index}")
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.int64))
    adata = AnnData(
        X=sparse.csr_matrix(counts.shape, dtype=np.float64),
        obs=pd.DataFrame(metadata, index=obs_names),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["counts"] = counts
    adata.uns["test_only_poison"] = {"must_be_removed": True}
    adata.obsm["test_only_embedding"] = np.ones((adata.n_obs, 2))
    return adata


def _independent_adata() -> AnnData:
    genes = ("L1", "R1", "L2", "R2", "T1", "T2")
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    obs_names: list[str] = []
    allocations = tuple(
        [(f"control-{index}", "control") for index in range(4)]
        + [(f"stim-{index}", "stim") for index in range(4)]
    )
    for subject_index, (subject, condition) in enumerate(allocations):
        high = 70 + 3 * subject_index
        low = 3 + subject_index
        target_one = low if condition == "control" else 25 + subject_index
        target_two = 2 + subject_index if condition == "control" else 14 + subject_index
        sample = f"sample-{subject}"
        for cell_type, profile in (
            ("Sender", [high, 0, low, 0, 1, 1]),
            ("Receiver", [0, high, 0, low, target_one, target_two]),
        ):
            for cell_index in range(3):
                rows.append(profile)
                metadata.append(
                    {
                        "sample_id": sample,
                        "subject_id": subject,
                        "cell_type": cell_type,
                        "condition": condition,
                    }
                )
                obs_names.append(f"{subject}-{cell_type}-{cell_index}")
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.int64))
    adata = AnnData(
        X=sparse.csr_matrix(counts.shape, dtype=np.float64),
        obs=pd.DataFrame(metadata, index=obs_names),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["counts"] = counts
    return adata


def _config() -> CrychicConfig:
    return CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=19,
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
        training_spec=FoldTrainingSpec(
            min_cells=1,
            max_interactions=1,
            sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
        ),
        allowed_n_splits=(2,),
    )


def _run(adata: AnnData):
    return run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
    )


def test_public_entry_accepts_no_caller_folds_or_fitted_artifacts() -> None:
    parameters = inspect.signature(run_subject_crossfit).parameters

    assert tuple(parameters) == (
        "adata",
        "config",
        "resource_bundle",
        "target_prior",
        "spec",
    )
    forbidden = {
        "fold_plan",
        "training_subject_ids",
        "test_subject_ids",
        "response_matrix",
        "availability_matrix",
        "training_artifacts",
        "model_manifest_id",
    }
    assert forbidden.isdisjoint(inspect.signature(CrossFitSpec).parameters)
    with pytest.raises(TypeError, match="producer-owned"):
        CrossFitArtifacts()


def test_subject_crossfit_runs_real_fit_apply_and_exact_oof_audit() -> None:
    result = _run(_adata())

    assert result.completed_stage_oof_verified
    assert result.is_oof_certified is False
    assert result.certification_status == ("verified_train_only_oof_partial_pipeline")
    assert result.coverage_audit.subject_ids == ("p1", "p2", "p3", "p4")
    assert result.coverage_audit.n_rows == 8
    assert result.coverage_audit.to_dict()["common_functional_validated"] is True
    assert len(result.folds) == 2
    assert not result.oof_sender_assignments.empty
    assert set(result.oof_sender_assignments["subject_id"]) == {
        "p1",
        "p2",
        "p3",
        "p4",
    }
    assert set(result.oof_sender_assignments["assignment_mode"]) == {
        "frozen_contrast_common_partial_not_oof"
    }
    assert set(result.oof_sender_assignments["functional_status"]) == {"out_of_fold"}
    assert set(result.oof_coverage["design_status"]) == {"observed"}
    assert result.oof_coverage["design_encoder_id"].notna().all()
    expected_receiver_rows = result.coverage_audit.n_rows * len(
        result.folds[0].training.cell_type_ids
    )
    assert len(result.oof_receiver_coverage) == expected_receiver_rows
    assert not result.oof_receiver_coverage.duplicated(
        ["fold_id", "sample_id", "contrast_id", "receiver"]
    ).any()
    assert set(result.oof_receiver_coverage["official_incremental_status"]) == {
        "not_estimable"
    }
    assert set(result.oof_receiver_coverage["reason_code"]) == {
        "receiver_autonomous_nuisance_not_frozen"
    }
    assert result.receiver_coverage_audit_id
    assert all(len(fold.design_encoders) == 1 for fold in result.folds)
    assert all(
        application.status == "observed"
        for fold in result.folds
        for application in fold.design_applications
    )
    assert all(fold.receiver_family_models for fold in result.folds)
    assert all(
        len(fold.receiver_responses)
        == len(fold.response_precisions)
        == len(fold.receiver_incremental_models)
        == len(fold.receiver_response_applications)
        == len(fold.receiver_incremental_applications)
        == len(fold.receiver_family_models)
        == len(fold.receiver_program_models)
        == len(fold.receiver_program_applications)
        for fold in result.folds
    )
    assert all(
        model.training_subject_ids == fold.training.training_subject_ids
        and not model.is_oof_certified
        for fold in result.folds
        for model in fold.receiver_family_models
    )
    assert all(
        not set(application.heldout_subject_ids).intersection(
            model.training_subject_ids
        )
        and not application.is_oof_certified
        for fold in result.folds
        for model, application in zip(
            fold.receiver_family_models,
            fold.receiver_family_applications,
            strict=True,
        )
    )
    assert all(
        program.receiver_family_artifact.training_artifact_id
        == family.receiver_family_artifact.training_artifact_id
        and application.training_artifact.training_artifact_id
        == program.training_artifact_id
        and program.family_ids
        == family.receiver_family_artifact.family_basis.family_ids
        for fold in result.folds
        for family, program, application in zip(
            fold.receiver_family_models,
            fold.receiver_program_models,
            fold.receiver_program_applications,
            strict=True,
        )
    )
    manifests = {fold.fold_id: fold for fold in result.fold_plan.folds}
    for row in result.oof_sender_assignments.itertuples(index=False):
        manifest = manifests[str(row.fold_id)]
        assert str(row.subject_id) in manifest.test_subject_ids
        assert str(row.subject_id) not in manifest.train_subject_ids
    for fold in result.folds:
        for response, precision, model, response_application, application in zip(
            fold.receiver_responses,
            fold.response_precisions,
            fold.receiver_incremental_models,
            fold.receiver_response_applications,
            fold.receiver_incremental_applications,
            strict=True,
        ):
            assert precision.response_artifact_id == response.artifact_id
            assert model.response_artifact_id == response.artifact_id
            assert model.precision_transform_id == precision.precision_transform_id
            assert response_application.training_response_id == response.artifact_id
            assert application.training_artifact_id == model.training_artifact_id
            assert (
                application.response_application_id
                == response_application.application_id
            )
    manifest = result.to_manifest()
    assert manifest["completed_stage_oof_verified"] is True
    assert manifest["complete_pipeline_oof_certified"] is False
    assert manifest["receiver_coverage_audit_id"] == (result.receiver_coverage_audit_id)
    assert manifest["receiver_coverage_status_counts"] == {
        "diagnostic_status": {
            str(status): int(count)
            for status, count in result.oof_receiver_coverage["diagnostic_status"]
            .value_counts()
            .items()
        },
        "official_incremental_status": {"not_estimable": expected_receiver_rows},
    }
    assert "common_scoring_functional" in manifest["remaining_stages"]
    assert "response_precision" not in manifest["remaining_stages"]
    assert "receiver_autonomous_nuisance" in manifest["remaining_stages"]
    assert "subject_blocked_inner_tuning" in manifest["remaining_stages"]
    assert (
        "source_agnostic_receiver_program_score" not in (manifest["remaining_stages"])
    )
    assert sum(manifest["receiver_program_training_status_counts"].values()) == sum(
        len(fold.receiver_program_models) for fold in result.folds
    )
    assert sum(manifest["receiver_program_application_status_counts"].values()) == sum(
        len(fold.receiver_program_applications) for fold in result.folds
    )
    assert all(
        len(item["receiver_program_artifacts"])
        == len(result.folds[0].receiver_program_models)
        for item in manifest["fold_artifacts"]
    )
    observed_programs = [
        artifact
        for item in manifest["fold_artifacts"]
        for artifact in item["receiver_program_artifacts"]
        if artifact["training_status"] == "observed"
    ]
    assert observed_programs
    assert all(
        artifact["reference_transform_id"]
        and artifact["reference_row_manifest_id"]
        and artifact["reference_subject_summary_digest"]
        and artifact["reference_summary_method"]
        == "technical_row_mean_then_context_equal_subject_mean_v1"
        and artifact["center_method"] == "reference_subject_equal_feature_median_v2"
        and artifact["scale_method"] == "reference_subject_equal_scaled_mad_floor_v2"
        for artifact in observed_programs
    )


def test_aggregate_context_lineage_requires_complete_training_coverage() -> None:
    config = _config()
    spec = _spec()
    prepared = crossfit_module._prepare_raw_fold(
        _adata(),
        config,
        min_cells=spec.training_spec.min_cells,
    )
    reference_contexts = {
        context for context, weight in spec.contrasts[0].weights.items() if weight < 0
    }
    _, sample_ids, subject_ids, context_ids = crossfit_module._response_expression(
        prepared,
        receiver="Receiver",
        contexts=reference_contexts,
    )

    assert crossfit_module._training_response_lineage_is_valid(
        prepared,
        contexts=reference_contexts,
        sample_ids=sample_ids,
        subject_ids=subject_ids,
        context_ids=context_ids,
    )
    assert not crossfit_module._training_response_lineage_is_valid(
        prepared,
        contexts=reference_contexts,
        sample_ids=sample_ids[1:],
        subject_ids=subject_ids[1:],
        context_ids=context_ids[1:],
    )
    assert crossfit_module._training_response_lineage_reason(
        prepared,
        contexts=reference_contexts,
        sample_ids=sample_ids[1:],
        subject_ids=subject_ids[1:],
        context_ids=context_ids[1:],
    ) == "training_reference_expression_incomplete_sample_coverage"
    poisoned_contexts = ("forged-context", *context_ids[1:])
    assert not crossfit_module._training_response_lineage_is_valid(
        prepared,
        contexts=reference_contexts,
        sample_ids=sample_ids,
        subject_ids=subject_ids,
        context_ids=poisoned_contexts,
    )
    duplicated_samples = (*sample_ids, sample_ids[0])
    assert not crossfit_module._training_response_lineage_is_valid(
        prepared,
        contexts=reference_contexts,
        sample_ids=duplicated_samples,
        subject_ids=(*subject_ids, subject_ids[0]),
        context_ids=(*context_ids, context_ids[0]),
    )


def test_training_reference_context_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = crossfit_module._response_expression

    def poisoned_context(*args: object, **kwargs: object):
        expression, sample_ids, subject_ids, context_ids = original(*args, **kwargs)
        requested = kwargs.get("contexts")
        if requested is not None and len(requested) == 1 and context_ids:
            context_ids = ("forged-reference-context", *context_ids[1:])
        return expression, sample_ids, subject_ids, context_ids

    monkeypatch.setattr(
        crossfit_module,
        "_response_expression",
        poisoned_context,
    )
    result = _run(_adata())
    receiver_family_models = [
        model
        for fold in result.folds
        for model in fold.receiver_family_models
        if model.receiver_family_artifact.receiver == "Receiver"
    ]
    receiver_program_models = [
        model
        for fold in result.folds
        for model in fold.receiver_program_models
        if model.receiver == "Receiver"
    ]

    assert receiver_family_models
    assert receiver_program_models
    assert {model.reason_code for model in receiver_family_models} == {
        "training_reference_expression_lineage_mismatch"
    }
    assert {model.status for model in receiver_program_models} == {"not_estimable"}
    assert {model.reason_code for model in receiver_program_models} == {
        "training_reference_expression_lineage_mismatch"
    }


def test_training_reference_incomplete_coverage_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = crossfit_module._response_expression

    def drop_reference_row(*args: object, **kwargs: object):
        expression, sample_ids, subject_ids, context_ids = original(*args, **kwargs)
        requested = kwargs.get("contexts")
        if requested is not None and len(requested) == 1 and sample_ids:
            return (
                expression[1:],
                sample_ids[1:],
                subject_ids[1:],
                context_ids[1:],
            )
        return expression, sample_ids, subject_ids, context_ids

    monkeypatch.setattr(
        crossfit_module,
        "_response_expression",
        drop_reference_row,
    )
    result = _run(_adata())
    receiver_family_models = [
        model
        for fold in result.folds
        for model in fold.receiver_family_models
        if model.receiver_family_artifact.receiver == "Receiver"
    ]
    receiver_program_models = [
        model
        for fold in result.folds
        for model in fold.receiver_program_models
        if model.receiver == "Receiver"
    ]

    assert receiver_family_models
    assert receiver_program_models
    assert {model.reason_code for model in receiver_family_models} == {
        "training_reference_expression_incomplete_sample_coverage"
    }
    assert {model.status for model in receiver_program_models} == {"not_estimable"}
    assert {model.reason_code for model in receiver_program_models} == {
        "training_reference_expression_incomplete_sample_coverage"
    }


def test_subject_crossfit_supports_independent_subject_groups() -> None:
    result = _run(_independent_adata())
    receiver_models = [
        model
        for fold in result.folds
        for model in fold.receiver_incremental_models
        if model.receiver == "Receiver"
    ]
    receiver_applications = [
        application
        for fold in result.folds
        for model, application in zip(
            fold.receiver_incremental_models,
            fold.receiver_incremental_applications,
            strict=True,
        )
        if model.receiver == "Receiver"
    ]

    assert receiver_models
    assert receiver_applications
    assert all(model.diagnostic_functional is not None for model in receiver_models)
    assert all(
        model.diagnostic_functional is not None
        and model.diagnostic_functional.loss_design
        == "independent_subject_pseudocontrasts_v1"
        for model in receiver_models
    )
    assert all(
        application.diagnostic_status == "observed"
        and application.diagnostic_reason_code is None
        and application.diagnostic_application is not None
        and application.diagnostic_application.status == "observed"
        and application.diagnostic_application.loss_aggregation.endswith(
            "frozen_independent_pseudocontrast_then_equal_subject_mean_v1"
        )
        for application in receiver_applications
    )
    receiver_coverage = result.oof_receiver_coverage.loc[
        result.oof_receiver_coverage["receiver"].eq("Receiver")
    ]
    expected_samples = {
        sample_id
        for application in receiver_applications
        for sample_id in application.heldout_sample_ids
    }
    assert not receiver_coverage.empty
    assert len(receiver_coverage) == sum(
        len(application.heldout_sample_ids) for application in receiver_applications
    )
    assert set(receiver_coverage["sample_id"]) == expected_samples
    assert set(receiver_coverage["diagnostic_status"]) == {"observed"}
    assert not receiver_coverage.duplicated(
        ["fold_id", "sample_id", "contrast_id", "receiver"]
    ).any()


def test_independent_group_inner_one_se_tuning_fails_closed() -> None:
    base = _spec()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("T1", "T2"),
        program_ids=("generic_program",),
        resource_id="crossfit-autonomous-programs",
        version="1",
        manifest_digest="3" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    tuning_spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0,),
        lambda2_fractions=(0.0,),
        inner_allowed_n_splits=(2,),
    )
    spec = CrossFitSpec(
        contrasts=base.contrasts,
        training_spec=base.training_spec,
        allowed_n_splits=base.allowed_n_splits,
        autonomous_program_resource=resource,
        penalty_tuning_spec=tuning_spec,
    )

    result = run_subject_crossfit(
        _independent_adata(), _config(), _bundle(), _prior(), spec=spec
    )

    receiver_models = [
        model
        for fold in result.folds
        for model in fold.receiver_incremental_models
        if model.receiver == "Receiver"
    ]
    assert receiver_models
    assert all(model.penalty_tuning_artifact is not None for model in receiver_models)
    assert {
        model.penalty_tuning_artifact.reason_code
        for model in receiver_models
        if model.penalty_tuning_artifact is not None
    } == {"independent_inner_tuning_one_se_not_supported"}
    assert all(model.diagnostic_functional is None for model in receiver_models)
    assert all(model.diagnostic_status == "not_estimable" for model in receiver_models)


def test_subject_crossfit_caller_declared_autonomous_resource_is_noncertifying() -> (
    None
):
    base = _spec()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("T1", "T2"),
        program_ids=("generic_program",),
        resource_id="crossfit-autonomous-programs",
        version="1",
        manifest_digest="a" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    spec = CrossFitSpec(
        contrasts=base.contrasts,
        training_spec=base.training_spec,
        allowed_n_splits=base.allowed_n_splits,
        autonomous_program_resource=resource,
    )

    result = run_subject_crossfit(_adata(), _config(), _bundle(), _prior(), spec=spec)

    assert set(result.oof_receiver_coverage["reason_code"]) == {
        "receiver_autonomous_nuisance_not_frozen"
    }
    models = [
        model for fold in result.folds for model in fold.receiver_incremental_models
    ]
    assert all(
        model.autonomous_program_resource_id == resource.artifact_id for model in models
    )
    assert all(
        (model.autonomous_projection_id is not None)
        == (model.diagnostic_functional is not None)
        for model in models
    )
    assert "receiver_autonomous_nuisance" in result.to_manifest()["remaining_stages"]


def test_crossfit_binds_typed_inner_tuning_children_without_fallback() -> None:
    base = _spec()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("T1", "T2"),
        program_ids=("generic_program",),
        resource_id="crossfit-autonomous-programs",
        version="1",
        manifest_digest="b" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    tuning_spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0,),
        lambda2_fractions=(0.0,),
        inner_allowed_n_splits=(2,),
    )
    spec = CrossFitSpec(
        contrasts=base.contrasts,
        training_spec=base.training_spec,
        allowed_n_splits=base.allowed_n_splits,
        autonomous_program_resource=resource,
        penalty_tuning_spec=tuning_spec,
    )

    result = run_subject_crossfit(_adata(), _config(), _bundle(), _prior(), spec=spec)

    models = [
        model for fold in result.folds for model in fold.receiver_incremental_models
    ]
    assert models
    assert all(model.penalty_tuning_spec_id == tuning_spec.spec_id for model in models)
    assert all(model.penalty_tuning_artifact is not None for model in models)
    assert all(model.diagnostic_functional is None for model in models)
    assert all(model.selected_penalty_candidate_id is None for model in models)
    assert {
        model.penalty_tuning_artifact.reason_code
        for model in models
        if model.penalty_tuning_artifact is not None
    } == {
        "no_estimable_subject_fold_plan",
        "no_training_eligible_receiver_family",
    }
    assert all(
        len(fold.family_common_functionals) == len(fold.receiver_incremental_models)
        and len(fold.family_common_applications)
        == len(fold.receiver_incremental_applications)
        for fold in result.folds
    )
    manifest = result.to_manifest()
    child = manifest["fold_artifacts"][0]["receiver_incremental_artifacts"][0]
    assert child["penalty_tuning_spec_id"] == tuning_spec.spec_id
    assert child["penalty_tuning_artifact_id"] is not None
    assert child["selected_penalty_candidate_id"] is None
    assert child["selected_resolved_penalty_id"] is None

    fold = result.folds[0]
    functional = fold.family_common_functionals[0]
    application = fold.family_common_applications[0]
    program_application = fold.receiver_program_applications[0]
    design_application = next(
        item
        for encoder, item in zip(
            fold.design_encoders,
            fold.design_applications,
            strict=True,
        )
        if encoder.contrast.name == functional.contrast_name
    )
    edge_evidence = crossfit_module._family_common_edge_evidence(
        functional,
        design_application,
        fold.application.availability.sample_interactions,
    )
    sender_application = crossfit_module._completed_common_sender_application(
        functional.sender_functional,
        design_application,
        fold.application.availability.sample_interactions,
        receiver=functional.receiver,
        interaction_ids=functional.interaction_ids,
    )
    unrelated_availability = fold.application.availability.sample_interactions.copy(
        deep=True
    )
    unrelated_mask = ~unrelated_availability["receiver"].astype(str).eq(
        functional.receiver
    )
    unrelated_availability.loc[unrelated_mask, "ligand_availability"] = 0.0
    unrelated_sender = crossfit_module._completed_common_sender_application(
        functional.sender_functional,
        design_application,
        unrelated_availability,
        receiver=functional.receiver,
        interaction_ids=functional.interaction_ids,
    )
    pd.testing.assert_frame_equal(sender_application.table, unrelated_sender.table)
    assert application.heldout_reason_code is not None

    omitted_subject = application.heldout_subject_ids[-1]
    subset_edges = edge_evidence.loc[
        ~edge_evidence["subject_id"].eq(omitted_subject)
    ].reset_index(drop=True)
    subset_samples = set(subset_edges["sample_id"].astype(str))
    subset_sender = CommonSenderApplication(
        sender_application.table.loc[
            sender_application.table["sample_id"].astype(str).isin(subset_samples)
        ].reset_index(drop=True),
        sender_application.functional,
    )
    subset_lineage = (
        subset_edges.loc[:, ["sample_id", "subject_id", "context_id"]]
        .drop_duplicates()
        .sort_values("sample_id", kind="stable")
    )
    assert functional.receiver_program_artifact is not None
    subset_program = mark_receiver_program_application_not_estimable(
        functional.receiver_program_artifact,
        sample_ids=tuple(subset_lineage["sample_id"].astype(str)),
        sample_subject_ids=tuple(subset_lineage["subject_id"].astype(str)),
        sample_context_ids=tuple(subset_lineage["context_id"].astype(str)),
        reason_code="test_subset_receiver_program_scope",
    )
    subset_application = mark_family_common_scoring_application_not_estimable(
        functional,
        subset_edges,
        subset_sender,
        heldout_reason_code=application.heldout_reason_code,
        receiver_program_application=subset_program,
    )
    with pytest.raises(
        ValueError, match="receiver-program application is incompatible"
    ):
        replace(
            fold,
            family_common_applications=(
                subset_application,
                *fold.family_common_applications[1:],
            ),
        )

    altered_edges = edge_evidence.copy(deep=True)
    altered_edges["prior_quality"] = 0.123
    altered_application = mark_family_common_scoring_application_not_estimable(
        functional,
        altered_edges,
        sender_application,
        heldout_reason_code=application.heldout_reason_code,
        receiver_program_application=program_application,
    )
    with pytest.raises(ContractError) as error:
        replace(
            fold,
            family_common_applications=(
                altered_application,
                *fold.family_common_applications[1:],
            ),
        )
    assert error.value.details.code == "family_common_crossfit_input_mismatch"

    altered_availability = fold.application.availability.sample_interactions.copy(
        deep=True
    )
    altered_availability["ligand_availability"] = 0.0
    altered_sender = crossfit_module._completed_common_sender_application(
        functional.sender_functional,
        design_application,
        altered_availability,
        receiver=functional.receiver,
        interaction_ids=functional.interaction_ids,
    )
    altered_sender_application = mark_family_common_scoring_application_not_estimable(
        functional,
        edge_evidence,
        altered_sender,
        heldout_reason_code=application.heldout_reason_code,
        receiver_program_application=program_application,
    )
    with pytest.raises(ContractError) as error:
        replace(
            fold,
            family_common_applications=(
                altered_sender_application,
                *fold.family_common_applications[1:],
            ),
        )
    assert error.value.details.code == "family_common_crossfit_input_mismatch"

    assert functional.incremental_reason_code is not None
    forged_functional = mark_family_common_scoring_not_estimable(
        functional.receiver_family,
        functional.sender_functional,
        fold_id=functional.fold_id,
        receiver_incremental_training_artifact_id=(
            functional.receiver_incremental_training_artifact_id
        ),
        tuning_manifest_id="forged-tuning-manifest",
        selected_penalty_id=functional.selected_penalty_id,
        autonomous_program_resource_id="forged-autonomous-resource",
        reason_code=functional.incremental_reason_code,
        receiver_program_artifact=functional.receiver_program_artifact,
    )
    forged_application = mark_family_common_scoring_application_not_estimable(
        forged_functional,
        edge_evidence,
        sender_application,
        heldout_reason_code=application.heldout_reason_code,
        receiver_program_application=program_application,
    )
    with pytest.raises(ValueError, match="authoritative training lineage"):
        replace(
            fold,
            family_common_functionals=(
                forged_functional,
                *fold.family_common_functionals[1:],
            ),
            family_common_applications=(
                forged_application,
                *fold.family_common_applications[1:],
            ),
        )


def test_trusted_tuned_receiver_is_officially_observed_out_of_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _spec()
    resource = _trusted_target_resource(tmp_path, monkeypatch)
    tuning_spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.1),
        lambda2_fractions=(0.0,),
        inner_allowed_n_splits=(2,),
    )
    spec = CrossFitSpec(
        contrasts=base.contrasts,
        training_spec=base.training_spec,
        allowed_n_splits=(2,),
        autonomous_program_resource=resource,
        penalty_tuning_spec=tuning_spec,
    )

    result = run_subject_crossfit(
        _adata(tuple(f"p{index}" for index in range(1, 9))),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )

    receiver_models = [
        model
        for fold in result.folds
        for model in fold.receiver_incremental_models
        if model.receiver == "Receiver"
    ]
    receiver_applications = [
        application
        for fold in result.folds
        for model, application in zip(
            fold.receiver_incremental_models,
            fold.receiver_incremental_applications,
            strict=True,
        )
        if model.receiver == "Receiver"
    ]
    assert receiver_models and len(receiver_models) == len(result.folds)
    assert all(len(model.training_subject_ids) == 4 for model in receiver_models)
    assert all(
        model.inner_fold_plan is not None
        and model.inner_fold_plan.effective_n_splits == 2
        for model in receiver_models
    )
    assert all(
        model.penalty_tuning_artifact is not None
        and model.penalty_tuning_artifact.status == "selected"
        and model.penalty_tuning_artifact.is_oof_certified
        for model in receiver_models
    )
    assert all(model.is_oof_certified for model in receiver_models)
    assert all(
        model.official_incremental_status == "observed" and model.reason_code is None
        for model in receiver_models
    )
    assert all(application.is_oof_certified for application in receiver_applications)
    assert all(
        application.official_incremental_status == "observed"
        and application.reason_code is None
        for application in receiver_applications
    )
    coverage = result.oof_receiver_coverage.loc[
        result.oof_receiver_coverage["receiver"].eq("Receiver")
    ]
    assert set(coverage["official_incremental_status"]) == {"observed"}
    assert coverage["reason_code"].isna().all()


def test_untrusted_tuned_diagnostic_cannot_cross_family_common_official_gate() -> None:
    base = _spec()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("T1", "T2"),
        program_ids=("generic_program",),
        resource_id="caller-built-autonomous-programs",
        version="1",
        manifest_digest="e" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    spec = CrossFitSpec(
        contrasts=base.contrasts,
        training_spec=base.training_spec,
        allowed_n_splits=(2,),
        autonomous_program_resource=resource,
        penalty_tuning_spec=PenaltyTuningSpec(
            lambda1_fractions=(1.0, 0.1),
            lambda2_fractions=(0.0,),
            inner_allowed_n_splits=(2,),
        ),
    )

    result = run_subject_crossfit(
        _adata(tuple(f"p{index}" for index in range(1, 9))),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )

    receiver_chains = [
        (model, application, functional, common_application)
        for fold in result.folds
        for model, application, functional, common_application in zip(
            fold.receiver_incremental_models,
            fold.receiver_incremental_applications,
            fold.family_common_functionals,
            fold.family_common_applications,
            strict=True,
        )
        if model.receiver == "Receiver"
    ]
    assert receiver_chains
    for model, application, functional, common_application in receiver_chains:
        assert model.diagnostic_functional is not None
        assert model.official_incremental_status == "not_estimable"
        assert model.reason_code == "receiver_autonomous_nuisance_not_frozen"
        assert application.diagnostic_application is not None
        assert application.official_incremental_status == "not_estimable"
        assert functional.incremental_functional is None
        assert functional.incremental_reason_code == (
            "receiver_autonomous_nuisance_not_frozen"
        )
        assert common_application.incremental_application_id is None
        assert common_application.heldout_reason_code == (
            "receiver_autonomous_nuisance_not_frozen"
        )


def test_receiver_coverage_audit_is_order_stable_and_rejects_context_poison() -> None:
    result = _run(_adata())
    reversed_rows = result.oof_receiver_coverage.iloc[::-1].reset_index(drop=True)

    assert (
        crossfit_module._receiver_coverage_identity(
            reversed_rows, fold_plan_id=result.fold_plan.plan_id
        )
        == result.receiver_coverage_audit_id
    )

    poisoned = result.oof_receiver_coverage.copy(deep=True)
    poisoned.loc[poisoned.index[0], "contrast_context"] = "poisoned-context"
    with pytest.raises(ValueError, match="parent lineage"):
        CrossFitArtifacts._from_workflow(
            spec=result.spec,
            fold_plan=result.fold_plan,
            folds=result.folds,
            oof_coverage=result.oof_coverage,
            oof_receiver_coverage=poisoned,
            oof_sender_assignments=result.oof_sender_assignments,
            coverage_audit=result.coverage_audit,
        )


@pytest.mark.parametrize(
    ("property_name", "column_name", "poison"),
    [
        ("oof_coverage", "functional_status", "poisoned"),
        ("oof_receiver_coverage", "official_incremental_status", "observed"),
        ("oof_sender_assignments", "functional_status", "poisoned"),
    ],
)
def test_crossfit_tables_are_defensive_copies(
    property_name: str,
    column_name: str,
    poison: str,
) -> None:
    result = _run(_adata())
    crossfit_id = result.crossfit_id
    manifest = result.to_manifest()
    exported = getattr(result, property_name)

    exported.loc[exported.index[0], column_name] = poison

    assert result.crossfit_id == crossfit_id
    assert result.to_manifest() == manifest
    assert getattr(result, property_name).loc[0, column_name] != poison


@pytest.mark.parametrize(
    ("private_name", "column_name", "poison"),
    [
        ("_oof_coverage", "functional_status", "poisoned"),
        ("_oof_receiver_coverage", "official_incremental_status", "observed"),
        ("_oof_sender_assignments", "functional_status", "poisoned"),
    ],
)
def test_manifest_rejects_forced_private_table_poison(
    private_name: str,
    column_name: str,
    poison: str,
) -> None:
    result = _run(_adata())
    private_table = object.__getattribute__(result, private_name)
    private_table.loc[private_table.index[0], column_name] = poison

    with pytest.raises(ContractError) as error:
        result.to_manifest()
    assert error.value.details.code == "crossfit_artifact_integrity_violation"


def test_crossfit_spec_rejects_forced_policy_mutation() -> None:
    spec = _spec()
    object.__setattr__(spec, "downstream_minimum_scale", 99.0)

    with pytest.raises(ContractError) as error:
        spec.to_dict()
    assert error.value.details.code == "crossfit_spec_integrity_violation"


def test_crossfit_spec_rejects_forced_nested_sender_policy_mutation() -> None:
    spec = _spec()
    object.__setattr__(spec.training_spec.sender_parameters, "min_subjects", 999)

    with pytest.raises(ContractError) as error:
        spec.to_dict()
    assert error.value.details.code == "crossfit_spec_integrity_violation"


@pytest.mark.parametrize(
    ("target", "field_name", "poison"),
    [
        ("plan", "repeat_id", "poisoned-repeat"),
        ("plan", "allowed_n_splits", (3, 2)),
        ("fold", "design_matrix_id", "poisoned-design"),
    ],
)
def test_crossfit_manifest_rejects_forced_fold_plan_mutation(
    target: str,
    field_name: str,
    poison: object,
) -> None:
    result = _run(_adata())
    artifact = result.fold_plan if target == "plan" else result.fold_plan.folds[0]
    object.__setattr__(artifact, field_name, poison)

    with pytest.raises(ContractError) as error:
        result.to_manifest()
    assert error.value.details.code == "crossfit_artifact_integrity_violation"


@pytest.mark.parametrize(
    "target",
    ["coverage_audit", "availability_table", "receiver_family_application"],
)
def test_crossfit_manifest_rejects_forced_nested_child_mutation(
    target: str,
) -> None:
    result = _run(_adata())
    if target == "coverage_audit":
        object.__setattr__(result.coverage_audit, "subject_ids", ("POISON",))
    elif target == "availability_table":
        table = result.folds[0].application.availability.sample_interactions
        table.loc[table.index[0], "ligand_availability"] = 0.123
    else:
        application = result.folds[0].receiver_family_applications[0]
        object.__setattr__(application, "active_family_ids", ("POISON",))

    with pytest.raises(ContractError) as error:
        result.to_manifest()
    assert error.value.details.code == "crossfit_artifact_integrity_violation"


def test_crossfit_fold_rejects_forced_receiver_program_child_replacement() -> None:
    result = _run(_adata())
    fold = result.folds[0]
    target = next(
        application
        for application in fold.receiver_program_applications
        if application.downstream_application is not None
    )
    assert target.downstream_application is not None
    forged_scores = target.downstream_application.receiver_program_score.copy()
    forged_scores[0, 0] += 0.25
    forged_scores.setflags(write=False)
    object.__setattr__(
        target.downstream_application,
        "receiver_program_score",
        forged_scores,
    )

    with pytest.raises(ContractError) as error:
        replace(
            fold,
            receiver_program_applications=tuple(
                target if item.application_id == target.application_id else item
                for item in fold.receiver_program_applications
            ),
        )
    assert error.value.details.code == (
        "receiver_program_application_integrity_violation"
    )


def test_orchestrator_passes_only_disjoint_sanitized_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = crossfit_module.fit_training_artifacts
    original_apply = crossfit_module.apply_training_artifacts
    calls: list[tuple[str, tuple[str, ...]]] = []

    def inspected_fit(adata: AnnData, *args: object, **kwargs: object):
        assert not adata.uns
        assert not adata.obsm
        subjects = tuple(sorted(adata.obs["subject_id"].astype(str).unique()))
        calls.append(("fit", subjects))
        return original_fit(adata, *args, **kwargs)

    def inspected_apply(artifacts, adata: AnnData):
        assert not adata.uns
        assert not adata.obsm
        subjects = tuple(sorted(adata.obs["subject_id"].astype(str).unique()))
        assert not set(subjects).intersection(artifacts.training_subject_ids)
        calls.append(("apply", subjects))

        def forbidden_fit(*args: object, **kwargs: object) -> None:
            raise AssertionError("held-out application called a fit function")

        with monkeypatch.context() as application_scope:
            application_scope.setattr(
                training_module,
                "_fit_interaction_universe",
                forbidden_fit,
            )
            application_scope.setattr(
                common_sender_module,
                "fit_contrast_common_sender_functional",
                forbidden_fit,
            )
            return original_apply(artifacts, adata)

    monkeypatch.setattr(crossfit_module, "fit_training_artifacts", inspected_fit)
    monkeypatch.setattr(crossfit_module, "apply_training_artifacts", inspected_apply)

    result = _run(_adata())

    assert [stage for stage, _ in calls] == ["fit", "apply", "fit", "apply"]
    for fold_result in result.folds:
        assert not set(fold_result.training.training_subject_ids).intersection(
            fold_result.application.heldout_subject_ids
        )


def test_test_subject_expression_poison_leaves_its_fold_training_id_unchanged() -> None:
    original = _adata()
    first = _run(original)
    target_fold = first.fold_plan.folds[0]
    poisoned_subject = target_fold.test_subject_ids[0]
    poisoned = original.copy()
    mask = poisoned.obs["subject_id"].astype(str).eq(poisoned_subject).to_numpy()
    counts = sparse.csr_matrix(poisoned.layers["counts"]).tolil(copy=True)
    counts[mask, 0] = 0
    counts[mask, 2] = 500
    poisoned.layers["counts"] = counts.tocsr()

    second = _run(poisoned)
    first_by_fold = {item.fold_id: item for item in first.folds}
    second_by_fold = {item.fold_id: item for item in second.folds}
    unchanged = first_by_fold[target_fold.fold_id]
    changed_application = second_by_fold[target_fold.fold_id]

    assert second.fold_plan.to_dict() == first.fold_plan.to_dict()
    assert changed_application.training.training_artifact_id == (
        unchanged.training.training_artifact_id
    )
    assert tuple(
        functional.sender_functional_id
        for functional in changed_application.training.sender_functionals
    ) == tuple(
        functional.sender_functional_id
        for functional in unchanged.training.sender_functionals
    )
    assert changed_application.application.heldout_input_digest != (
        unchanged.application.heldout_input_digest
    )
    assert changed_application.training.frozen_interaction_universe.to_dict() == (
        unchanged.training.frozen_interaction_universe.to_dict()
    )
    assert tuple(
        model.receiver_family_artifact.receptor_gate_manifest_id
        for model in changed_application.receiver_family_models
    ) == tuple(
        model.receiver_family_artifact.receptor_gate_manifest_id
        for model in unchanged.receiver_family_models
    )
    assert tuple(
        model.training_artifact_id
        for model in changed_application.receiver_family_models
    ) == tuple(model.training_artifact_id for model in unchanged.receiver_family_models)
    assert tuple(
        model.downstream_functional.downstream_functional_id
        for model in changed_application.receiver_family_models
        if model.downstream_functional is not None
    ) == tuple(
        model.downstream_functional.downstream_functional_id
        for model in unchanged.receiver_family_models
        if model.downstream_functional is not None
    )
    assert tuple(
        model.training_artifact_id
        for model in changed_application.receiver_program_models
    ) == tuple(
        model.training_artifact_id for model in unchanged.receiver_program_models
    )
    assert tuple(
        model.downstream_functional.reference_transform_id
        for model in changed_application.receiver_program_models
        if model.downstream_functional is not None
    ) == tuple(
        model.downstream_functional.reference_transform_id
        for model in unchanged.receiver_program_models
        if model.downstream_functional is not None
    )
    assert tuple(
        response.artifact_id for response in changed_application.receiver_responses
    ) == tuple(response.artifact_id for response in unchanged.receiver_responses)
    assert tuple(
        precision.precision_transform_id
        for precision in changed_application.response_precisions
    ) == tuple(
        precision.precision_transform_id for precision in unchanged.response_precisions
    )
    assert tuple(
        model.training_artifact_id
        for model in changed_application.receiver_incremental_models
    ) == tuple(
        model.training_artifact_id for model in unchanged.receiver_incremental_models
    )
    first_rows = first.oof_sender_assignments.loc[
        first.oof_sender_assignments["fold_id"].eq(target_fold.fold_id)
    ].reset_index(drop=True)
    second_rows = second.oof_sender_assignments.loc[
        second.oof_sender_assignments["fold_id"].eq(target_fold.fold_id)
    ].reset_index(drop=True)
    assert not second_rows.equals(first_rows)


def test_heldout_receiver_target_poison_changes_apply_not_training_models() -> None:
    original = _adata()
    first = _run(original)
    target_fold = first.fold_plan.folds[0]
    poisoned = original.copy()
    receiver_mask = (
        poisoned.obs["subject_id"].astype(str).isin(target_fold.test_subject_ids)
        & poisoned.obs["cell_type"].astype(str).eq("Receiver")
    ).to_numpy()
    counts = sparse.csr_matrix(poisoned.layers["counts"]).tolil(copy=True)
    counts[receiver_mask, 4] = 10_000
    poisoned.layers["counts"] = counts.tocsr()

    second = _run(poisoned)
    before = {fold.fold_id: fold for fold in first.folds}[target_fold.fold_id]
    after = {fold.fold_id: fold for fold in second.folds}[target_fold.fold_id]

    assert tuple(
        model.training_artifact_id for model in before.receiver_family_models
    ) == (tuple(model.training_artifact_id for model in after.receiver_family_models))
    assert tuple(
        response.artifact_id for response in before.receiver_responses
    ) == tuple(response.artifact_id for response in after.receiver_responses)
    assert tuple(
        precision.precision_transform_id for precision in before.response_precisions
    ) == tuple(
        precision.precision_transform_id for precision in after.response_precisions
    )
    assert tuple(
        model.training_artifact_id for model in before.receiver_incremental_models
    ) == tuple(
        model.training_artifact_id for model in after.receiver_incremental_models
    )
    assert tuple(
        application.application_id
        for application in before.receiver_response_applications
    ) != tuple(
        application.application_id
        for application in after.receiver_response_applications
    )
    assert tuple(
        application.application_id
        for application in before.receiver_incremental_applications
    ) != tuple(
        application.application_id
        for application in after.receiver_incremental_applications
    )
    before_scores = [
        application.downstream_application.receiver_program_score
        for application in before.receiver_family_applications
        if application.downstream_application is not None
    ]
    after_scores = [
        application.downstream_application.receiver_program_score
        for application in after.receiver_family_applications
        if application.downstream_application is not None
    ]
    assert before_scores and len(before_scores) == len(after_scores)
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(before_scores, after_scores, strict=True)
    )


def test_receiver_family_heldout_application_cannot_call_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_apply = crossfit_module.apply_receiver_family_scoring_artifact

    def inspected_apply(*args: object, **kwargs: object):
        def forbidden_fit(*inner_args: object, **inner_kwargs: object) -> None:
            raise AssertionError("receiver-family heldout application called fit")

        with monkeypatch.context() as application_scope:
            application_scope.setattr(
                receiver_scoring_module,
                "fit_downstream_functional",
                forbidden_fit,
            )
            application_scope.setattr(
                crossfit_module,
                "fit_receiver_family_training_artifacts",
                forbidden_fit,
            )
            return original_apply(*args, **kwargs)

    monkeypatch.setattr(
        crossfit_module,
        "apply_receiver_family_scoring_artifact",
        inspected_apply,
    )

    result = _run(_adata())

    assert all(fold.receiver_family_applications for fold in result.folds)


def test_receiver_incremental_heldout_application_cannot_call_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_apply = crossfit_module._apply_receiver_incremental_chains

    def inspected_apply(*args: object, **kwargs: object):
        def forbidden_fit(*inner_args: object, **inner_kwargs: object) -> None:
            raise AssertionError("receiver incremental held-out application called fit")

        with monkeypatch.context() as application_scope:
            for name in (
                "fit_fold_gene_response",
                "fit_response_precision",
                "fit_receiver_incremental_training_artifact",
            ):
                application_scope.setattr(crossfit_module, name, forbidden_fit)
            return original_apply(*args, **kwargs)

    monkeypatch.setattr(
        crossfit_module,
        "_apply_receiver_incremental_chains",
        inspected_apply,
    )

    result = _run(_adata())

    assert all(fold.receiver_incremental_applications for fold in result.folds)


def test_crossfit_batches_receiver_family_training_once_per_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = crossfit_module.fit_receiver_family_training_artifacts
    calls: list[tuple[str, tuple[str, ...]]] = []

    def counted_fit(*args: object, **kwargs: object):
        calls.append((str(kwargs["fold_id"]), tuple(kwargs["receivers"])))
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(
        crossfit_module,
        "fit_receiver_family_training_artifacts",
        counted_fit,
    )

    result = _run(_adata())

    assert len(calls) == len(result.fold_plan.folds)
    assert tuple(fold_id for fold_id, _ in calls) == tuple(
        fold.fold_id for fold in result.fold_plan.folds
    )
    assert all(receivers == ("Receiver", "Sender") for _, receivers in calls)


def test_missing_heldout_receiver_is_preserved_as_not_estimable() -> None:
    original = _adata()
    first = _run(original)
    target_fold = first.fold_plan.folds[0]
    poisoned_subject = target_fold.test_subject_ids[0]
    target_rows = (
        original.obs["subject_id"].astype(str).eq(poisoned_subject)
        & original.obs["cell_type"].astype(str).eq("Receiver")
    ).to_numpy()
    missing_receiver = original[~target_rows].copy()
    extra_rows: list[list[int]] = []
    extra_obs: list[dict[str, str]] = []
    for subject in (poisoned_subject,):
        for condition in ("control", "stim"):
            extra_rows.extend([[80, 0, 4, 0, 0, 0]] * 3)
            extra_obs.append(
                {
                    "sample_id": f"sample-{subject}-{condition}",
                    "subject_id": subject,
                    "cell_type": "Sender",
                    "condition": condition,
                }
            )
            extra_obs.extend([extra_obs[-1].copy(), extra_obs[-1].copy()])
    expanded_counts = sparse.vstack(
        [
            sparse.csr_matrix(missing_receiver.layers["counts"]),
            sparse.csr_matrix(extra_rows, dtype=np.int64),
        ],
        format="csr",
    )
    expanded_obs = pd.concat(
        [
            missing_receiver.obs,
            pd.DataFrame(
                extra_obs,
                index=[f"extra-sender-{index}" for index in range(len(extra_obs))],
            ),
        ]
    )
    missing_receiver = AnnData(
        X=sparse.csr_matrix(expanded_counts.shape, dtype=np.float64),
        obs=expanded_obs,
        var=missing_receiver.var.copy(),
    )
    missing_receiver.layers["counts"] = expanded_counts
    assert target_rows.any()
    second = run_subject_crossfit(
        missing_receiver,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
    )
    target = {fold.fold_id: fold for fold in second.folds}[target_fold.fold_id]
    pairs = tuple(
        zip(
            target.receiver_family_models,
            target.receiver_family_applications,
            strict=True,
        )
    )

    assert any(
        model.receiver_family_artifact.receiver == "Receiver"
        and application.application_status == "frozen_application_partial_not_estimable"
        and application.reason_code == "heldout_receiver_expression_lineage_mismatch"
        for model, application in pairs
    )
    assert any(
        model.receiver == "Receiver"
        and application.status == "not_estimable"
        and application.reason_code
        == "heldout_receiver_expression_incomplete_sample_coverage"
        for model, application in zip(
            target.receiver_program_models,
            target.receiver_program_applications,
            strict=True,
        )
    )
    assert {model.receiver_family_artifact.receiver for model, _ in pairs} == set(
        target.training.cell_type_ids
    )
    missing_rows = second.oof_receiver_coverage.loc[
        second.oof_receiver_coverage["fold_id"].eq(target_fold.fold_id)
        & second.oof_receiver_coverage["subject_id"].eq(poisoned_subject)
        & second.oof_receiver_coverage["receiver"].eq("Receiver")
    ]
    assert len(missing_rows) == 2
    assert set(missing_rows["diagnostic_status"]) == {"not_estimable"}
    assert set(missing_rows["diagnostic_reason_code"]) == {
        "incomplete_heldout_receiver_response"
    }
    assert set(missing_rows["official_incremental_status"]) == {"not_estimable"}


def test_normalized_only_input_cannot_claim_stage_oof_verification() -> None:
    normalized = _adata()
    normalized.X = sparse.csr_matrix(normalized.layers["counts"], dtype=np.float64)
    del normalized.layers["counts"]
    config = CrychicConfig(
        context_keys=("condition",),
        counts_layer=None,
        expression_source="synthetic_linear_normalized",
        expression_transform="linear_normalized",
        normalized_zero_is_nondetection=True,
        design="~ condition",
    )

    with pytest.raises(ValueError, match="requires raw counts"):
        run_subject_crossfit(
            normalized,
            config,
            _bundle(),
            _prior(),
            spec=_spec(),
        )


def test_heldout_covariate_poison_does_not_change_fold_encoder_identity() -> None:
    original = _adata()
    preliminary = _run(original)
    site_by_subject: dict[str, str] = {}
    for fold in preliminary.fold_plan.folds:
        for index, subject in enumerate(fold.test_subject_ids):
            site_by_subject[subject] = ("a", "b")[index]
    original.obs["site"] = original.obs["subject_id"].astype(str).map(site_by_subject)
    config = CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        covariates=("site",),
        design="~ site + condition",
        random_seed=19,
    )
    first = run_subject_crossfit(
        original,
        config,
        _bundle(),
        _prior(),
        spec=_spec(),
    )
    target_fold = first.fold_plan.folds[0]
    poisoned = original.copy()
    poisoned_subject = target_fold.test_subject_ids[0]
    poisoned.obs.loc[
        poisoned.obs["subject_id"].astype(str).eq(poisoned_subject), "site"
    ] = "test-only-level"

    second = run_subject_crossfit(
        poisoned,
        config,
        _bundle(),
        _prior(),
        spec=_spec(),
    )
    first_by_fold = {fold.fold_id: fold for fold in first.folds}
    second_by_fold = {fold.fold_id: fold for fold in second.folds}
    before = first_by_fold[target_fold.fold_id]
    after = second_by_fold[target_fold.fold_id]

    assert before.design_encoders[0].encoder_id == after.design_encoders[0].encoder_id
    assert before.design_applications[0].status == "observed"
    assert after.design_applications[0].status == "not_estimable"
    assert after.design_applications[0].reason_code == "unseen_heldout_level:site"
    selected = second.oof_coverage["fold_id"].eq(target_fold.fold_id)
    assert set(second.oof_coverage.loc[selected, "design_status"]) == {"not_estimable"}
