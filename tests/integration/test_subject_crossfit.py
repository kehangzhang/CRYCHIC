from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import crychic.response.autonomous as autonomous_module
import crychic.scoring.receiver_family as receiver_scoring_module
import crychic.sender.common as common_sender_module
import crychic.workflow.crossfit as crossfit_module
import crychic.workflow.repeated_crossfit as repeated_crossfit_module
import crychic.workflow.training as training_module
from crychic import Crychic, fit_crossfit_family_effect
from crychic.attribution import (
    DirectionalContrastPairSpec,
    GainCalibrationSpec,
    PenaltyTuningSpec,
    PenaltyValidationLossEstimand,
    freeze_receiver_family_opportunity_universe,
)
from crychic.core import (
    ContractError,
    CrychicConfig,
    SeedLineage,
    canonical_digest,
    canonical_json,
    stable_id,
)
from crychic.design import balanced_contrast
from crychic.inference import (
    FullPipelineEffectDistributionSpec,
    HypothesisDeclaration,
    HypothesisRole,
    SpecificityDirection,
    freeze_hypothesis_universe,
)
from crychic.resampling import SubjectFoldPlan
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
    AutonomousProgramReviewScope,
    ReceiverAutonomousProgramRegistration,
)
from crychic.response import (
    ReceiverAutonomousProgramResource,
    build_receiver_autonomous_program_resource,
    load_receiver_autonomous_program_resource,
)
from crychic.response.repeated_fold import RepeatedMeasuresFoldResponseArtifact
from crychic.results import ResultValidationError, ResultWriteError
from crychic.scoring import (
    SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    AbsoluteActivityV2Spec,
    FrozenLatentNuisanceSpec,
    PlannedScoringCollectionManifest,
    ReceiverScoringFunctionalManifest,
    ScoringCollectionDocument,
    ScoringCollectionManifest,
    mark_family_common_scoring_application_not_estimable,
    mark_family_common_scoring_not_estimable,
    mark_receiver_program_application_not_estimable,
)
from crychic.sender import CommonSenderApplication, ContrastCommonSenderParameters
from crychic.workflow import (
    CrossFitArtifacts,
    CrossFitResult,
    CrossFitSampleEdgeV2Result,
    CrossFitSpec,
    FamilyEffectTarget,
    FoldTrainingSpec,
    FullPipelineResamplingStatus,
    RepeatedCrossFitDiagnostics,
    RepeatedCrossFitSpec,
    adapt_crossfit_active_edge_point_records,
    build_family_effect_oof_spec,
    build_frozen_family_effect_target,
    freeze_crossfit_active_edge_universe,
    recommended_crossfit_spec,
    run_full_pipeline_resampling,
    run_repeated_subject_crossfit,
    run_subject_crossfit,
    summarize_family_effect_full_pipeline,
    write_crossfit_result,
    write_crossfit_sample_edge_v2_result,
)
from crychic.workflow.certification import (
    CROSSFIT_OOF_DESCRIPTIVE_SCOPE,
    CrossFitOOFCertificationAudit,
    CrossFitOOFRequirementRecord,
    audit_crossfit_oof_readiness,
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


def _ligand_prior_with_unmapped_driver() -> TargetPrior:
    return TargetPrior(
        resource_id="crossfit_ligand_prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=("T1", "T2"),
        driver_ids=("L1", "L2", "UNMAPPED"),
        indptr=(0, 1, 2, 3),
        target_indices=(0, 1, 0),
        weights=(1.0, 1.0, 1.0),
        ranks=None,
        direction=1,
        evidence="synthetic-unmapped-driver-regression",
        mapping_report=MappingReport(3, 3, 5),
        manifest_digest="e" * 64,
    )


def _trusted_target_resource(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    review_scope: AutonomousProgramReviewScope = "synthetic_benchmark_only",
) -> ReceiverAutonomousProgramResource:
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
        review_scope=review_scope,
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


def _independent_adata(n_subjects_per_context: int = 4) -> AnnData:
    genes = ("L1", "R1", "L2", "R2", "T1", "T2")
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    obs_names: list[str] = []
    allocations = tuple(
        [(f"control-{index}", "control") for index in range(n_subjects_per_context)]
        + [(f"stim-{index}", "stim") for index in range(n_subjects_per_context)]
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


def _mixed_adata(*, n_paired: int = 8, n_single_per_context: int = 4) -> AnnData:
    genes = ("L1", "R1", "L2", "R2", "T1", "T2")
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    obs_names: list[str] = []
    allocations = [
        *((f"paired-{index}", ("control", "stim")) for index in range(n_paired)),
        *(
            (f"control-only-{index}", ("control",))
            for index in range(n_single_per_context)
        ),
        *((f"stim-only-{index}", ("stim",)) for index in range(n_single_per_context)),
    ]
    for subject_index, (subject, conditions) in enumerate(allocations):
        for condition in conditions:
            sample = f"sample-{subject}-{condition}"
            condition_shift = 16 if condition == "stim" else 0
            target_one = 4 + subject_index % 7 + condition_shift
            target_two = 3 + (2 * subject_index) % 9 + condition_shift // 2
            high = 70 + 3 * (subject_index % 5)
            low = 2 + subject_index % 4
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
                    obs_names.append(f"{subject}-{condition}-{cell_type}-{cell_index}")
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.int64))
    adata = AnnData(
        X=sparse.csr_matrix(counts.shape, dtype=np.float64),
        obs=pd.DataFrame(metadata, index=obs_names),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["counts"] = counts
    return adata


def _repeated_multi_context_adata() -> AnnData:
    genes = ("L1", "R1", "L2", "R2", "T1", "T2")
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    obs_names: list[str] = []
    patterns = (
        ("a", "b", "c"),
        ("a", "b"),
        ("b", "c"),
        ("a", "c"),
    )
    allocations = [
        (f"pattern-{pattern_index}-{subject_index}", contexts)
        for pattern_index, contexts in enumerate(patterns)
        for subject_index in range(4)
    ]
    context_shift = {"a": 0, "b": 7, "c": 18}
    for subject_index, (subject, contexts) in enumerate(allocations):
        for condition in contexts:
            sample = f"sample-{subject}-{condition}"
            shift = context_shift[condition]
            target_one = 5 + subject_index % 6 + shift
            target_two = 3 + (2 * subject_index) % 7 + shift // 2
            high = 70 + 2 * (subject_index % 7)
            low = 2 + subject_index % 5
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
                    obs_names.append(f"{subject}-{condition}-{cell_type}-{cell_index}")
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


def _directional_spec() -> CrossFitSpec:
    base = _spec()
    forward = balanced_contrast(
        ("stim",),
        ("control",),
        name="stim_vs_control",
        family="treatment",
    )
    reverse = balanced_contrast(
        ("control",),
        ("stim",),
        name="control_vs_stim",
        family="treatment",
    )
    pair = DirectionalContrastPairSpec(
        forward_contrast=forward,
        reverse_contrast=reverse,
    )
    return CrossFitSpec(
        contrasts=(reverse, forward),
        directional_pairs=(pair,),
        training_spec=replace(base.training_spec, sender_contrasts=None),
        allowed_n_splits=base.allowed_n_splits,
    )


def test_directional_crossfit_spec_is_order_invariant_and_opt_in() -> None:
    base = _spec()
    explicit_default = replace(base, directional_pairs=())
    directional = _directional_spec()
    reversed_input = replace(
        directional,
        contrasts=tuple(reversed(directional.contrasts)),
        directional_pairs=tuple(reversed(directional.directional_pairs)),
    )

    assert explicit_default.spec_id == base.spec_id
    assert "directional_pairs" not in base.to_dict()
    assert directional.spec_id == reversed_input.spec_id
    assert directional.spec_id != base.spec_id
    assert directional.to_dict()["directional_pairs"] == [
        directional.directional_pairs[0].to_dict()
    ]

    forward = directional.directional_pairs[0].forward_contrast
    with pytest.raises(ValueError, match="registered in contrasts"):
        CrossFitSpec(
            contrasts=(forward,),
            directional_pairs=directional.directional_pairs,
            training_spec=replace(base.training_spec, sender_contrasts=None),
            allowed_n_splits=(2,),
        )
    with pytest.raises(ValueError, match="at most one directional pair"):
        opposite_pair = DirectionalContrastPairSpec(
            forward_contrast=directional.directional_pairs[0].reverse_contrast,
            reverse_contrast=directional.directional_pairs[0].forward_contrast,
        )
        CrossFitSpec(
            contrasts=directional.contrasts,
            directional_pairs=(
                directional.directional_pairs[0],
                opposite_pair,
            ),
            training_spec=replace(base.training_spec, sender_contrasts=None),
            allowed_n_splits=(2,),
        )


def test_directional_crossfit_emits_exact_typed_binding_registry(
    tmp_path: Path,
) -> None:
    result = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )

    pair = result.spec.directional_pairs[0]
    expected = {
        (fold.fold_id, pair.pair_spec_id, receiver)
        for fold in result.folds
        for receiver in fold.training.cell_type_ids
    }
    observed = {
        (fold.fold_id, binding.pair_spec_id, binding.receiver)
        for fold in result.folds
        for binding in fold.directional_response_bindings
    }
    assert observed == expected
    assert all(
        binding.forward_channel_name == "increased_activation_compatible"
        and binding.reverse_channel_name == "reduced_activation_compatible"
        and binding.supports_active_inhibition_claim is False
        and binding.paired_score_comparison_allowed is False
        and binding.formal_inference_allowed is False
        for fold in result.folds
        for binding in fold.directional_response_bindings
    )
    bindings = tuple(
        binding
        for fold in result.folds
        for binding in fold.directional_response_bindings
    )
    assert [binding.status for binding in bindings].count("observed") == 2
    assert [binding.status for binding in bindings].count("not_estimable") == 2
    assert all(
        (binding.response_pair is not None) == (binding.status == "observed")
        for binding in bindings
    )
    manifest = result.to_manifest()
    assert manifest["directional_pair_stage_connected"] is True
    assert manifest["directional_registry_complete"] is True
    assert manifest["directional_supported_binding_registry_complete"] is True
    assert manifest["n_directional_receiver_pair_absent_training"] == 0
    assert manifest["n_directional_receiver_pair_opportunities"] == len(expected)
    assert manifest["n_directional_response_bindings"] == len(expected)
    assert manifest["directional_binding_status_counts"] == {
        "not_estimable": 2,
        "observed": 2,
    }
    assert manifest["directional_diagnostic_complete"] is False
    assert all(
        len(item["directional_response_bindings"])
        == len(result.spec.directional_pairs) * len(fold.training.cell_type_ids)
        for item, fold in zip(manifest["fold_artifacts"], result.folds, strict=True)
    )

    persisted = write_crossfit_result(result, tmp_path / "directional-crossfit")
    loaded = CrossFitResult.load(persisted.path)
    registry = loaded.read_directional_channel_registry()
    opportunities = loaded.read_directional_opportunities()
    assert loaded.manifest["schema_version"] == "9.0.0"
    assert len(registry) == 2 * len(expected)
    assert len(opportunities) == 2 * len(expected)
    assert set(registry["channel_role"]) == {"forward", "reverse"}
    assert set(opportunities["receiver_training_support_status"]) == {"observed"}
    assert opportunities["channel_opportunity_id"].is_unique
    assert opportunities.groupby("pair_opportunity_id").size().eq(2).all()
    assert registry["status"].value_counts().to_dict() == {
        "observed": 4,
        "not_estimable": 4,
    }
    assert len(
        loaded.query_directional_channels(channel="increased_activation_compatible")
    ) == len(expected)
    assert loaded.read_semantic_integrated_lr_scores().empty
    assert loaded.read_semantic_differential_effects().empty
    semantic_statuses = {
        row[0]: (row[1], row[2])
        for row in cast(
            list[list[object]],
            loaded.semantic_score_manifest["view_statuses"],
        )
    }
    assert semantic_statuses["integrated_lr_score"] == (
        "not_produced",
        "directional_contrasts_require_dedicated_integrated_lr_collection",
    )
    assert (
        semantic_statuses["differential_effect"]
        == semantic_statuses["integrated_lr_score"]
    )


def test_directional_manifest_does_not_hide_absent_receiver_opportunities(
    tmp_path: Path,
) -> None:
    adata = _adata()
    heldout_only = adata.obs["subject_id"].eq("p1") & adata.obs["cell_type"].eq(
        "Sender"
    )
    adata.obs.loc[heldout_only, "cell_type"] = "Novel"

    result = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )
    manifest = result.to_manifest()

    supported = sum(
        record.status == "observed"
        for fold in result.folds
        for record in fold.receiver_training_support
    )
    absent = sum(
        record.status == "not_estimable"
        for fold in result.folds
        for record in fold.receiver_training_support
    )
    n_pairs = len(result.spec.directional_pairs)
    assert manifest["n_directional_response_bindings"] == n_pairs * supported
    assert manifest["n_directional_receiver_pair_absent_training"] == (n_pairs * absent)
    assert manifest["n_directional_receiver_pair_opportunities"] == (
        n_pairs * len(result.receiver_universe.receiver_ids) * len(result.folds)
    )
    assert manifest["directional_supported_binding_registry_complete"] is True
    assert manifest["directional_registry_complete"] is False
    assert manifest["directional_diagnostic_complete"] is False

    persisted = write_crossfit_result(
        result,
        tmp_path / "directional-absent-receiver-opportunities",
    )
    loaded = CrossFitResult.load(persisted.path)
    opportunities = loaded.read_directional_opportunities()
    expected_keys = {
        (pair.pair_spec_id, fold.fold_id, receiver, role)
        for pair in result.spec.directional_pairs
        for fold in result.folds
        for receiver in result.receiver_universe.receiver_ids
        for role in ("forward", "reverse")
    }
    observed_keys = set(
        opportunities[
            ["pair_spec_id", "fold_id", "receiver", "channel_role"]
        ].itertuples(index=False, name=None)
    )
    assert observed_keys == expected_keys
    assert (
        len(opportunities) == 2 * manifest["n_directional_receiver_pair_opportunities"]
    )
    assert opportunities["channel_opportunity_id"].is_unique
    assert opportunities.groupby("pair_opportunity_id").size().eq(2).all()

    absent_rows = loaded.query_directional_opportunities(
        receiver_training_support_status="not_estimable"
    )
    assert len(absent_rows) == 2 * n_pairs * absent
    assert set(absent_rows["channel_role"]) == {"forward", "reverse"}
    assert set(absent_rows["status"]) == {"not_estimable"}
    assert set(absent_rows["reason_code"]) == {"receiver_absent_in_outer_training"}
    assert set(absent_rows["receiver_training_support_reason_code"]) == {
        "receiver_absent_in_outer_training"
    }
    assert (
        absent_rows[
            [
                "binding_id",
                "response_id",
                "training_artifact_id",
                "application_id",
                "response_pair_id",
                "response_channel_id",
            ]
        ]
        .isna()
        .all(axis=None)
    )
    assert (
        absent_rows[
            [
                "pair_opportunity_id",
                "channel_opportunity_id",
                "receiver_training_support_id",
                "channel",
                "contrast_id",
                "contrast",
            ]
        ]
        .notna()
        .all(axis=None)
    )
    assert "structural_zero" not in set(opportunities["status"])


def test_directional_mixed_cr2_partial_features_are_typed_not_estimable() -> None:
    result = run_subject_crossfit(
        _mixed_adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )

    responses = [
        response
        for fold in result.folds
        for response in fold.receiver_responses
        if response.receiver == "Receiver"
    ]
    bindings = [
        binding
        for fold in result.folds
        for binding in fold.directional_response_bindings
        if binding.receiver == "Receiver"
    ]
    assert responses
    assert all(
        isinstance(response, RepeatedMeasuresFoldResponseArtifact)
        and response.status == "ok"
        and response.cr2_backend_eligible
        and np.isnan(response.effect).any()
        for response in responses
    )
    assert bindings
    assert all(
        binding.status == "not_estimable"
        and binding.response_pair is None
        and binding.reason_code
        == (
            "directional_components_not_estimable["
            "forward_response:partial_feature_response_not_supported_by_"
            "directional_pair;"
            "reverse_response:partial_feature_response_not_supported_by_"
            "directional_pair]"
        )
        for binding in bindings
    )


def test_root_family_universe_is_prefit_and_bound_for_ordinary_crossfit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    crossfit_identity_payloads: list[dict[str, object]] = []
    original_freeze = crossfit_module.freeze_receiver_family_opportunity_universe
    original_plan = crossfit_module.plan_subject_folds
    original_fit = crossfit_module._fit_training_artifacts_from_prepared
    original_stable_id = crossfit_module.stable_id

    def tracked_freeze(*args, **kwargs):
        events.append("freeze_root_family_universe")
        return original_freeze(*args, **kwargs)

    def tracked_plan(*args, **kwargs):
        events.append("plan_subject_folds")
        return original_plan(*args, **kwargs)

    def tracked_fit(*args, **kwargs):
        events.append("fit_training_fold")
        return original_fit(*args, **kwargs)

    def tracked_stable_id(kind, payload, *args, **kwargs):
        result = original_stable_id(kind, payload, *args, **kwargs)
        if kind == "subject_crossfit":
            crossfit_identity_payloads.append(payload)
        return result

    monkeypatch.setattr(
        crossfit_module,
        "freeze_receiver_family_opportunity_universe",
        tracked_freeze,
    )
    monkeypatch.setattr(crossfit_module, "plan_subject_folds", tracked_plan)
    monkeypatch.setattr(
        crossfit_module,
        "_fit_training_artifacts_from_prepared",
        tracked_fit,
    )
    monkeypatch.setattr(crossfit_module, "stable_id", tracked_stable_id)

    result = _run(_adata())
    universe = result.receiver_family_opportunity_universe
    assert events.count("freeze_root_family_universe") == 1
    assert events.index("freeze_root_family_universe") < events.index(
        "plan_subject_folds"
    )
    assert events.index("plan_subject_folds") < events.index("fit_training_fold")
    assert universe.receiver_ids == result.receiver_universe.receiver_ids
    assert universe.receiver_universe_id == result.receiver_universe.universe_id
    assert len(universe.opportunity_ids) == (
        len(universe.receiver_ids) * len(universe.family_ids)
    )

    manifest = result.to_manifest()
    assert manifest["receiver_family_opportunity_universe"] == universe.to_dict()
    assert manifest["receiver_family_opportunity_universe_id"] == universe.universe_id
    assert manifest["family_axis_id"] == universe.family_axis_id
    assert manifest["receiver_family_opportunity_axis_id"] == (
        universe.opportunity_axis_id
    )
    assert crossfit_identity_payloads
    assert all(
        payload["receiver_family_opportunity_universe_id"] == universe.universe_id
        and payload["family_axis_id"] == universe.family_axis_id
        and payload["receiver_family_opportunity_axis_id"]
        == universe.opportunity_axis_id
        for payload in crossfit_identity_payloads
    )
    for fold in result.folds:
        supported = {
            record.receiver_id
            for record in fold.receiver_training_support
            if record.status == "observed"
        }
        parents = {
            model.receiver_family_artifact.receiver: model.receiver_family_artifact
            for model in fold.receiver_family_models
        }
        assert set(parents) == supported
        for parent in parents.values():
            assert parent.prior_manifest_digest == universe.prior_manifest_digest
            assert parent.source_basis.feature_ids == universe.feature_ids
            assert parent.source_basis.driver_ids == universe.driver_ids
            assert parent.family_basis.feature_ids == universe.feature_ids
            assert parent.family_basis.family_definitions == universe.family_definitions
            assert parent.family_basis.family_ids == universe.family_ids
            assert (
                parent.family_basis.strict_cosine_threshold == universe.cosine_threshold
            )


def test_directional_lr_universe_is_prefit_and_bound_to_crossfit_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    crossfit_identity_payloads: list[dict[str, object]] = []
    original_family_freeze = crossfit_module.freeze_receiver_family_opportunity_universe
    original_freeze = crossfit_module.freeze_receiver_family_lr_hypothesis_universe
    original_plan = crossfit_module.plan_subject_folds
    original_fit = crossfit_module._fit_training_artifacts_from_prepared
    original_stable_id = crossfit_module.stable_id

    def tracked_family_freeze(*args, **kwargs):
        events.append("freeze_root_family_universe")
        return original_family_freeze(*args, **kwargs)

    def tracked_freeze(*args, **kwargs):
        events.append("freeze_directional_lr_universe")
        return original_freeze(*args, **kwargs)

    def tracked_plan(*args, **kwargs):
        events.append("plan_subject_folds")
        return original_plan(*args, **kwargs)

    def tracked_fit(*args, **kwargs):
        events.append("fit_training_fold")
        return original_fit(*args, **kwargs)

    def tracked_stable_id(kind, payload, *args, **kwargs):
        result = original_stable_id(kind, payload, *args, **kwargs)
        if kind == "subject_crossfit":
            crossfit_identity_payloads.append(payload)
        return result

    monkeypatch.setattr(
        crossfit_module,
        "freeze_receiver_family_opportunity_universe",
        tracked_family_freeze,
    )
    monkeypatch.setattr(
        crossfit_module,
        "freeze_receiver_family_lr_hypothesis_universe",
        tracked_freeze,
    )
    monkeypatch.setattr(crossfit_module, "plan_subject_folds", tracked_plan)
    monkeypatch.setattr(
        crossfit_module,
        "_fit_training_artifacts_from_prepared",
        tracked_fit,
    )
    monkeypatch.setattr(crossfit_module, "stable_id", tracked_stable_id)

    result = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )
    universe = result.directional_lr_hypothesis_universe
    assert universe is not None
    root_family_universe = result.receiver_family_opportunity_universe
    assert universe._receiver_family_universe is root_family_universe
    assert events.index("freeze_root_family_universe") < events.index(
        "freeze_directional_lr_universe"
    )
    assert events.index("freeze_directional_lr_universe") < events.index(
        "plan_subject_folds"
    )
    assert events.index("plan_subject_folds") < events.index("fit_training_fold")

    manifest = result.to_manifest()
    assert manifest["receiver_family_opportunity_universe"] == (
        root_family_universe.to_dict()
    )
    assert manifest["directional_lr_hypothesis_universe"] == universe.to_dict()
    assert manifest["directional_lr_hypothesis_universe"]["universe_id"] == (
        universe.universe_id
    )
    assert crossfit_identity_payloads
    assert all(
        payload["directional_lr_hypothesis_universe_id"] == universe.universe_id
        for payload in crossfit_identity_payloads
    )
    raw_folds = cast(list[dict[str, object]], manifest["fold_artifacts"])
    raw_by_fold = {str(record["fold_id"]): record for record in raw_folds}
    directional_contrast_ids = {
        contrast_id
        for pair in result.spec.directional_pairs
        for contrast_id in (pair.forward_contrast_id, pair.reverse_contrast_id)
    }
    for fold in result.folds:
        expected_designs = [
            {
                "contrast_id": crossfit_module._contrast_id(encoder.contrast),
                "contrast_name": encoder.contrast.name,
                "encoder": encoder.to_dict(),
                "application": application.to_dict(),
            }
            for encoder, application in zip(
                fold.design_encoders,
                fold.design_applications,
                strict=True,
            )
            if crossfit_module._contrast_id(encoder.contrast)
            in directional_contrast_ids
        ]
        assert raw_by_fold[fold.fold_id]["directional_design_applications"] == (
            expected_designs
        )


def test_nondirectional_crossfit_omits_directional_lr_universe_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crossfit_identity_payloads: list[dict[str, object]] = []
    original_stable_id = crossfit_module.stable_id

    def tracked_stable_id(kind, payload, *args, **kwargs):
        result = original_stable_id(kind, payload, *args, **kwargs)
        if kind == "subject_crossfit":
            crossfit_identity_payloads.append(payload)
        return result

    monkeypatch.setattr(crossfit_module, "stable_id", tracked_stable_id)
    result = _run(_adata())
    manifest = result.to_manifest()

    assert manifest["receiver_family_opportunity_universe"] == (
        result.receiver_family_opportunity_universe.to_dict()
    )
    assert result.directional_lr_hypothesis_universe is None
    assert "directional_lr_hypothesis_universe" not in manifest
    assert all(
        "directional_design_applications" not in fold
        for fold in cast(list[dict[str, object]], manifest["fold_artifacts"])
    )
    assert crossfit_identity_payloads
    assert all(
        "directional_lr_hypothesis_universe_id" not in payload
        for payload in crossfit_identity_payloads
    )
    assert all(
        payload["receiver_family_opportunity_universe_id"]
        == result.receiver_family_opportunity_universe.universe_id
        for payload in crossfit_identity_payloads
    )


def test_ordinary_crossfit_rejects_root_family_universe_tampering() -> None:
    result = _run(_adata())
    object.__setattr__(
        result.receiver_family_opportunity_universe,
        "family_axis_id",
        "poisoned-family-axis",
    )

    with pytest.raises(ContractError) as caught:
        result.to_manifest()
    assert caught.value.details.code == "crossfit_artifact_integrity_violation"


def test_directional_crossfit_from_workflow_requires_prefit_lr_universe() -> None:
    result = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )

    with pytest.raises(ValueError, match="missing its pre-fit LR hypothesis universe"):
        CrossFitArtifacts._from_workflow(
            spec=result.spec,
            root_input_identity=result.root_input_identity,
            receiver_universe=result.receiver_universe,
            receiver_family_opportunity_universe=(
                result.receiver_family_opportunity_universe
            ),
            fold_plan=result.fold_plan,
            folds=result.folds,
            oof_coverage=result.oof_coverage,
            oof_receiver_coverage=result.oof_receiver_coverage,
            oof_sender_assignments=result.oof_sender_assignments,
            coverage_audit=result.coverage_audit,
        )


@pytest.mark.parametrize(
    "tamper_target",
    ("receiver", "resource", "prior", "family", "mapping"),
)
def test_directional_crossfit_rejects_lr_universe_parent_tampering(
    tamper_target: str,
) -> None:
    result = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )
    universe = result.directional_lr_hypothesis_universe
    assert universe is not None

    if tamper_target == "receiver":
        object.__setattr__(universe, "receiver_ids", ("POISON",))
    elif tamper_target == "resource":
        object.__setattr__(
            universe._resource_bundle,
            "manifest_digest",
            "poisoned-resource-manifest",
        )
    elif tamper_target == "prior":
        object.__setattr__(
            universe._target_prior,
            "manifest_digest",
            "poisoned-prior-manifest",
        )
    elif tamper_target == "family":
        object.__setattr__(
            universe._receiver_family_universe,
            "family_axis_id",
            "poisoned-family-axis",
        )
    else:
        object.__setattr__(
            universe.memberships[0],
            "driver_id",
            "POISON",
        )

    with pytest.raises(ContractError) as caught:
        result.to_manifest()
    assert caught.value.details.code == "crossfit_artifact_integrity_violation"


def test_directional_fold_family_models_preserve_frozen_lr_axis_exactly() -> None:
    result = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )
    universe = result.directional_lr_hypothesis_universe
    assert universe is not None
    expected_mapping = tuple(
        sorted(
            (membership.interaction_id, membership.driver_id)
            for membership in universe.memberships
        )
    )

    for fold in result.folds:
        assert fold.training.resource_bundle_content_id == (
            universe.resource_bundle_content_id
        )
        assert fold.training.target_prior_content_id == universe.target_prior_content_id
        for model in fold.receiver_family_models:
            receiver_family = model.receiver_family_artifact
            assert receiver_family.driver_by_interaction == expected_mapping
            assert receiver_family.family_basis.family_ids == universe.family_ids
            assert (
                stable_id(
                    "root_feature_axis",
                    {"feature_ids": list(receiver_family.family_basis.feature_ids)},
                    schema_version="1",
                )
                == universe.feature_axis_id
            )


def _run(adata: AnnData):
    return run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
    )


def test_absolute_activity_v2_is_opt_in_and_emits_exact_heldout_rows(
    tmp_path: Path,
) -> None:
    legacy = _spec()
    v7_spec = replace(
        legacy,
        absolute_activity_v2_spec=AbsoluteActivityV2Spec(),
    )

    assert "absolute_activity_v2_spec" not in legacy.to_dict()
    assert v7_spec.spec_id != legacy.spec_id
    result = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=v7_spec,
    )

    scores = result.oof_sample_edge_scores_v2
    assert not scores.empty
    assert set(scores["subject_id"]) == {"p1", "p2", "p3", "p4"}
    assert scores["out_of_fold"].all()
    assert scores["sender_detection_raw"].notna().any()
    assert not scores["structural_impossibility"].any()
    for fold in result.folds:
        transform = fold.absolute_activity_v2_transform
        fold_scores = fold.sample_edge_scores_v2
        assert transform is not None
        assert fold_scores is not None
        assert transform.spec.spec_id == v7_spec.absolute_activity_v2_spec.spec_id
        assert not set(transform.training_subject_ids).intersection(
            fold_scores.provenance.application_subject_ids
        )
        availability = fold.application.availability.sample_interactions
        assert len(fold_scores.table) == len(availability)
        assert set(fold_scores.table["activity_functional_id"]) == {
            fold_scores.provenance.provenance_id
        }
    manifest = result.to_manifest()
    assert manifest["absolute_activity_v2_stage_connected"] is True
    assert manifest["absolute_activity_v2_score_rows"] == len(scores)
    assert manifest["absolute_activity_v2_formal_inference_eligible"] is False
    persisted = write_crossfit_sample_edge_v2_result(
        result, tmp_path / "sample-edge-v2"
    )
    loaded = CrossFitSampleEdgeV2Result.load(persisted.path)
    assert loaded.manifest["crossfit_id"] == result.crossfit_id
    assert sum(len(child.scores.table) for child in loaded.fold_artifacts) == len(
        scores
    )


def _persistable_spec() -> CrossFitSpec:
    base = _spec()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("T1", "T2"),
        program_ids=("generic_program",),
        resource_id="crossfit-persistence-autonomous-programs",
        version="1",
        manifest_digest="9" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    return replace(
        base,
        autonomous_program_resource=resource,
        penalty_tuning_spec=PenaltyTuningSpec(
            lambda1_fractions=(1.0,),
            lambda2_fractions=(0.0,),
            inner_allowed_n_splits=(2,),
        ),
    )


def _persistable_run() -> CrossFitArtifacts:
    return run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_persistable_spec(),
    )


def test_state_only_config_avoids_unrequested_ecosystem_score_rows() -> None:
    result = run_subject_crossfit(
        _adata(),
        replace(_config(), communication_modes=("state",)),
        _bundle(),
        _prior(),
        spec=_persistable_spec(),
    )

    applications = [
        application
        for fold in result.folds
        for application in fold.family_common_applications
    ]
    assert applications
    for application in applications:
        assert set(application.family_scores["mode"]) == {"state"}
        assert set(application.member_scores["mode"]) == {"state"}
        assert set(application.sender_scores["mode"]) == {"state"}


def _as_v3_scoring_registry(
    current: ScoringCollectionDocument,
) -> ScoringCollectionDocument:
    version = SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION
    collections: list[ScoringCollectionManifest] = []
    for collection in current.collections:
        children = tuple(
            ReceiverScoringFunctionalManifest(
                receiver=child.receiver,
                scoring_functional_id=child.scoring_functional_id,
                source_score_key_digest=child.source_score_key_digest,
                source_score_row_count=child.source_score_row_count,
                provenance_status=child.provenance_status,
                contract_version=version,
                receiver_family_model_id=child.receiver_family_model_id,
                receiver_incremental_model_id=child.receiver_incremental_model_id,
                filter_universe_id=child.filter_universe_id,
                score_version=child.score_version,
                functional_status=child.functional_status,
                registry_status=child.registry_status,
                emission_status=child.emission_status,
                reason_code=child.reason_code,
            )
            for child in collection.children
        )
        collections.append(
            ScoringCollectionManifest.planned_receiver_registry(
                contrast=collection.contrast,
                repeat_id=collection.repeat_id,
                fold_id=collection.fold_id,
                planned_receivers=collection.planned_receivers,
                children=children,
                contract_version=version,
            )
        )
    planned = tuple(
        PlannedScoringCollectionManifest.from_collection(
            collection,
            filter_universe_id=cast(
                str,
                collection.children[0].filter_universe_id,
            ),
        )
        for collection in collections
    )
    return ScoringCollectionDocument(
        collections=tuple(collections),
        planned_collections=planned,
        extension_schema_version=version,
    )


def test_crossfit_result_atomic_round_trip_and_corruption_rejection(
    tmp_path: Path,
) -> None:
    artifacts = _persistable_run()
    destination = tmp_path / "crossfit-result"

    result = write_crossfit_result(artifacts, destination)
    loaded = CrossFitResult.load(destination)
    components = loaded.read_components()
    differential = loaded.read_descriptive_differential()

    assert result.path == destination.resolve()
    assert result.manifest == loaded.manifest
    assert result.manifest["crossfit_id"] == artifacts.crossfit_id
    assert result.manifest["complete_pipeline_oof_certified"] is False
    assert result.manifest["formal_inference_status"] == (
        "not_available_descriptive_only"
    )
    assert len(result.manifest["applications"]) == sum(
        len(fold.family_common_applications) for fold in artifacts.folds
    )
    assert set(components["component_scope"]) == {
        "family",
        "lr_member",
        "sender_lr_member",
    }
    assert (
        components.loc[
            components["component_scope"].eq("family"),
            ["driver_id", "interaction_id", "sender"],
        ]
        .isna()
        .all()
        .all()
    )
    assert (
        components.loc[components["component_scope"].eq("lr_member"), "interaction_id"]
        .notna()
        .all()
    )
    assert not components["is_oof_certified"].any()
    assert not differential["is_oof_certified"].any()
    assert set(differential["formal_inference_status"]) == {
        "not_available_descriptive_only"
    }
    assert not {
        "p_value",
        "q_value",
        "comm_probability",
        "posterior",
        "confidence_interval",
        "standard_error",
    }.intersection(differential.columns)
    assert set(differential["status"]).issubset(
        {"observed", "not_estimable", "structural_zero"}
    )
    assert not tuple(tmp_path.glob(".crossfit-result.tmp-*"))

    with pytest.raises(ResultWriteError) as duplicate_error:
        write_crossfit_result(artifacts, destination)
    assert duplicate_error.value.details.code == ("crossfit_result_destination_exists")

    component_path = destination / "family_common_components.parquet"
    component_path.write_bytes(component_path.read_bytes() + b"corruption")
    with pytest.raises(ResultValidationError) as corrupted_error:
        CrossFitResult.load(destination)
    assert corrupted_error.value.details.code == ("crossfit_result_digest_mismatch")


@pytest.mark.parametrize("legacy_version", ("1.0.0", "2.0.0"))
def test_crossfit_result_legacy_compatibility_is_explicit_and_noncertifying(
    tmp_path: Path,
    legacy_version: str,
) -> None:
    destination = tmp_path / "legacy-crossfit-result"
    artifacts = _persistable_run()
    write_crossfit_result(artifacts, destination)
    manifest_path = destination / "crossfit_manifest.json"
    status_path = destination / "_status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source_crossfit_manifest"]
    if legacy_version == "1.0.0":
        source.pop("oof_certification_audit")
        source.pop("oof_certification_audit_id")
    else:
        registry = _as_v3_scoring_registry(artifacts.receiver_scoring_registry)
        source["receiver_scoring_registry_id"] = registry.registry_id
        source["receiver_scoring_registry"] = registry.to_dict()
        current_audit = CrossFitOOFCertificationAudit.from_dict(
            cast(dict[str, object], source["oof_certification_audit"])
        )
        requirements = tuple(
            (
                CrossFitOOFRequirementRecord(
                    requirement_code="authoritative_v3_receiver_registry",
                    status=requirement.status,
                    reason_code=requirement.reason_code,
                    repeat_id=requirement.repeat_id,
                    fold_id=requirement.fold_id,
                    contrast=requirement.contrast,
                    receiver=requirement.receiver,
                    evidence_ids=(cast(str, registry.registry_id),),
                )
                if requirement.requirement_code == "authoritative_v4_receiver_registry"
                else requirement
            )
            for requirement in current_audit.requirement_ledger
        )
        legacy_audit = CrossFitOOFCertificationAudit._from_requirements(
            source_crossfit_id=current_audit.source_crossfit_id,
            source_registry_id=registry.registry_id,
            requirements=requirements,
        )
        source["oof_certification_audit"] = legacy_audit.to_dict()
        source["oof_certification_audit_id"] = legacy_audit.audit_id
    manifest.pop("contrast_common_stage_connected")
    manifest.pop("contrast_common_collections")
    manifest.pop("semantic_score_collection")
    manifest.pop("receiver_universe_id")
    manifest.pop("receiver_axis_id")
    for field_name in (
        "receiver_family_opportunity_universe_id",
        "family_axis_id",
        "receiver_family_opportunity_axis_id",
        "receiver_family_opportunity_universe",
    ):
        manifest.pop(field_name)
        source.pop(field_name)
    for table_name in (
        "contrast_common_lr_scores",
        "contrast_common_sender_lr_scores",
        "directional_channel_registry",
        "receiver_training_support",
        "semantic_availability_scores",
        "semantic_receiver_program_scores",
        "semantic_integrated_lr_scores",
        "semantic_differential_effects",
    ):
        table_record = manifest["tables"].pop(table_name)
        (destination / table_record["filename"]).unlink()
    manifest["schema_version"] = legacy_version
    manifest["source_crossfit_manifest_digest"] = canonical_digest(source)
    for table in manifest["tables"].values():
        table["schema_version"] = legacy_version
    manifest["crossfit_result_id"] = stable_id(
        "crossfit_result",
        {key: value for key, value in manifest.items() if key != "crossfit_result_id"},
        schema_version="1",
    )
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    status = {
        "schema_version": legacy_version,
        "status": "complete",
        "crossfit_result_id": manifest["crossfit_result_id"],
    }
    status_path.write_text(
        f"{canonical_json(status)}\n",
        encoding="utf-8",
    )

    loaded = CrossFitResult.load(destination)

    assert loaded.manifest["schema_version"] == legacy_version
    assert loaded.manifest["complete_pipeline_oof_certified"] is False
    assert not loaded.read_components()["is_oof_certified"].any()


def test_crossfit_result_rejects_status_manifest_identity_mismatch(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "status-mismatch-crossfit-result"
    write_crossfit_result(_persistable_run(), destination)
    status_path = destination / "_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["crossfit_result_id"] = "crossfit-result-poisoned"
    status_path.write_text(f"{canonical_json(status)}\n", encoding="utf-8")

    with pytest.raises(ResultValidationError) as error:
        CrossFitResult.load(destination)
    assert error.value.details.code == "crossfit_result_status_manifest_mismatch"


def test_full_pipeline_bootstrap_smoke_reruns_real_crossfit() -> None:
    result = run_full_pipeline_resampling(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
        n_bootstraps=1,
    )

    assert result.status is FullPipelineResamplingStatus.SUCCEEDED
    assert len(result.records) == 1
    assert result.records[0].crossfit_id is not None
    assert result.records[0].child is None
    assert result.children == ()
    assert result.to_manifest()["full_pipeline_refit_per_resample"] is True


def test_family_effect_adapter_real_crossfit_and_retained_resample_smoke() -> None:
    point = _persistable_run()
    first_functional = next(
        functional
        for fold in point.folds
        for functional in fold.family_common_functionals
    )
    matching = [
        functional
        for fold in point.folds
        for functional in fold.family_common_functionals
        if functional.contrast_name == first_functional.contrast_name
        and functional.receiver == first_functional.receiver
    ]
    common_families = set(matching[0].family_ids).intersection(
        *(set(functional.family_ids) for functional in matching[1:])
    )
    target = FamilyEffectTarget(
        contrast_name=first_functional.contrast_name,
        receiver=first_functional.receiver,
        family_id=sorted(common_families)[0],
        mode="state",
    )
    effect_spec = build_family_effect_oof_spec(point, target)
    point_effect = fit_crossfit_family_effect(point, target, effect_spec)
    resampling = run_full_pipeline_resampling(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=point.spec,
        n_bootstraps=1,
        retain_children=True,
    )
    distribution_spec = FullPipelineEffectDistributionSpec(
        effect_spec=effect_spec,
        minimum_effect=0.0,
        specificity_direction=SpecificityDirection.GREATER,
    )

    distribution = summarize_family_effect_full_pipeline(
        point,
        resampling,
        target,
        distribution_spec,
    )

    assert point_effect.spec_id == effect_spec.spec_id
    assert len(distribution.resample_record_ids) == 1
    assert distribution.formal_inference_allowed is False
    assert distribution.standard_error is None


def test_real_crossfit_bootstrap_support_object_chain_is_complete() -> None:
    point = _persistable_run()
    first_functional = next(
        functional
        for fold in point.folds
        for functional in fold.family_common_functionals
    )
    matching = [
        functional
        for fold in point.folds
        for functional in fold.family_common_functionals
        if functional.contrast_name == first_functional.contrast_name
        and functional.receiver == first_functional.receiver
    ]
    common_families = set(matching[0].family_ids).intersection(
        *(set(functional.family_ids) for functional in matching[1:])
    )
    family_id = sorted(common_families)[0]
    primary = HypothesisDeclaration(
        endpoint="driver_family_receiver_context_omnibus_v1",
        contrast_name=first_functional.contrast_name,
        receiver=first_functional.receiver,
        family_id=family_id,
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="integration-primary",
    )
    secondary = HypothesisDeclaration(
        endpoint="family_common_integrated_lr_context_effect_v1",
        contrast_name=first_functional.contrast_name,
        receiver=first_functional.receiver,
        family_id=family_id,
        mode="state",
        role=HypothesisRole.SECONDARY,
        multiplicity_family="integration-secondary",
        parent_key=primary.hypothesis_key,
    )
    universe = freeze_hypothesis_universe(
        (primary, secondary),
        universe_name="real-chain-bootstrap-support-v1",
    )
    secondary_target = build_frozen_family_effect_target(
        universe,
        secondary.hypothesis_id,
    )
    distribution_spec = FullPipelineEffectDistributionSpec(
        effect_spec=build_family_effect_oof_spec(point, secondary_target),
        minimum_effect=0.0,
        specificity_direction=SpecificityDirection.GREATER,
    )
    resampling = run_full_pipeline_resampling(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=point.spec,
        n_bootstraps=1,
        retain_children=True,
    )

    model = Crychic(
        _config(),
        resource_bundle=_bundle(),
        target_prior=_prior(),
    )
    document = model.summarize_bootstrap_support(
        point,
        resampling,
        universe=universe,
        distribution_specs=(distribution_spec,),
    )

    assert resampling.status is FullPipelineResamplingStatus.SUCCEEDED
    assert len(resampling.children) == 1
    assert document.registry["resampling_lineage"]["resampling_result_id"] == (
        resampling.result_id
    )
    specificity_row = document.specificity_support.iloc[0]
    selection_row = document.selection_frequency.iloc[0]
    assert specificity_row["hypothesis_id"] == secondary.hypothesis_id
    assert specificity_row["status"] == "not_estimable"
    assert specificity_row["n_bootstrap_total"] == 1
    assert not specificity_row["specificity_support_release_allowed"]
    assert selection_row["hypothesis_id"] == primary.hypothesis_id
    assert selection_row["status"] == "not_estimable"
    assert selection_row["n_bootstrap_plans_total"] == 1
    assert not selection_row["selection_frequency_release_allowed"]


def test_crossfit_facade_persists_reloadable_descriptive_result(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "facade-crossfit-result"
    model = Crychic(
        _config(),
        resource_bundle=_bundle(),
        target_prior=_prior(),
    )

    result = model.fit_crossfit(
        _adata(),
        spec=_persistable_spec(),
        output_dir=destination,
    )

    assert isinstance(result, CrossFitResult)
    loaded = CrossFitResult.load(destination)
    assert loaded.manifest == result.manifest
    assert loaded.manifest["complete_pipeline_oof_certified"] is False
    assert loaded.manifest["formal_inference_status"] == (
        "not_available_descriptive_only"
    )

    components = loaded.read_components()
    differential = loaded.read_descriptive_differential()
    assert not components.empty
    assert not differential.empty
    assert not components["is_oof_certified"].any()
    assert not differential["is_oof_certified"].any()
    forbidden_inference_fields = {
        "p_value",
        "q_value",
        "comm_probability",
        "posterior",
        "confidence_interval",
        "standard_error",
    }
    assert forbidden_inference_fields.isdisjoint(components.columns)
    assert forbidden_inference_fields.isdisjoint(differential.columns)


def test_analyze_default_profile_runs_the_real_crossfit_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Crychic(
        _config(),
        resource_bundle=_bundle(),
        target_prior=_prior(),
    )
    base = _spec()
    spec = recommended_crossfit_spec(
        contrasts=base.contrasts,
        training_spec=FoldTrainingSpec(
            min_cells=1,
            sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
        ),
        root_seed=19,
    )

    def forbidden_legacy(*args: object, **kwargs: object) -> object:
        raise AssertionError("crossfit_descriptive_v1 must not call legacy fit")

    monkeypatch.setattr(Crychic, "fit", forbidden_legacy)
    observed = model.analyze(_adata(), spec=spec)

    assert isinstance(observed, CrossFitArtifacts)
    assert observed.spec.spec_id == spec.spec_id
    assert observed.fold_plan.folds


def test_public_entry_accepts_no_caller_folds_or_fitted_artifacts() -> None:
    parameters = inspect.signature(run_subject_crossfit).parameters

    assert tuple(parameters) == (
        "adata",
        "config",
        "resource_bundle",
        "target_prior",
        "spec",
        "n_jobs",
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


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_public_entry_rejects_invalid_outer_fold_jobs_before_snapshot(
    invalid: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_snapshot(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid n_jobs must fail before snapshot construction")

    monkeypatch.setattr(
        crossfit_module,
        "_sanitized_raw_input_snapshot",
        forbidden_snapshot,
    )

    with pytest.raises(ValueError, match="n_jobs must be an integer >= 1"):
        run_subject_crossfit(
            _adata(),
            _config(),
            _bundle(),
            _prior(),
            spec=_spec(),
            n_jobs=cast(int, invalid),
        )


def test_outer_fold_parallelism_preserves_identity_tables_and_plan_order() -> None:
    adata = _adata()
    serial = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
        n_jobs=1,
    )
    parallel = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
        n_jobs=2,
    )

    assert parallel.crossfit_id == serial.crossfit_id
    assert parallel.coverage_table_digest == serial.coverage_table_digest
    assert (
        parallel.receiver_coverage_table_digest == serial.receiver_coverage_table_digest
    )
    assert (
        parallel.sender_assignment_table_digest == serial.sender_assignment_table_digest
    )
    expected_fold_order = tuple(fold.fold_id for fold in serial.fold_plan.folds)
    assert tuple(fold.fold_id for fold in serial.folds) == expected_fold_order
    assert tuple(fold.fold_id for fold in parallel.folds) == expected_fold_order
    pd.testing.assert_frame_equal(parallel.oof_coverage, serial.oof_coverage)
    pd.testing.assert_frame_equal(
        parallel.oof_receiver_coverage,
        serial.oof_receiver_coverage,
    )
    pd.testing.assert_frame_equal(
        parallel.oof_sender_assignments,
        serial.oof_sender_assignments,
    )


def test_outer_fold_parallel_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fold(
        fold: object,
        *,
        context: object,
    ) -> object:
        del fold, context
        raise RuntimeError("injected outer fold failure")

    monkeypatch.setattr(crossfit_module, "_run_crossfit_fold", fail_fold)

    with pytest.raises(RuntimeError, match="injected outer fold failure"):
        run_subject_crossfit(
            _adata(),
            _config(),
            _bundle(),
            _prior(),
            spec=_spec(),
            n_jobs=2,
        )


def test_repeat_index_changes_repeat_identity_not_algorithm_policy() -> None:
    base = _spec()
    repeated = replace(base, repeat_index=1)

    assert repeated.spec_id == base.spec_id
    assert repeated.repeat_id != base.repeat_id
    assert base.to_dict()["repeat_index"] == 0
    assert repeated.to_dict()["repeat_index"] == 1
    with pytest.raises(ValueError, match="repeat_index"):
        replace(base, repeat_index=-1)


def test_default_outer_partition_policy_preserves_partition_identity() -> None:
    spec = _spec()
    result = _run(_adata())

    assert spec.spec_id == "subject_crossfit_spec_a7f64b4101e21229987413e0f5aa9bb0"
    assert spec.repeat_id == "subject_crossfit_repeat_18475ebb4574c64abf8a8fdd91bb90ce"
    assert "outer_fold_partition_seed" not in spec.to_dict()
    assert result.fold_plan.partition_seed_lineage is None
    assert "partition_seed_lineage" not in result.fold_plan.to_dict()
    assert result.fold_plan.plan_id == (
        "subject_fold_plan_bcea0ce54a9d69b0bdba3a22ffe3aa7c"
    )
    assert [fold.fold_id for fold in result.fold_plan.folds] == [
        "subject_fold_218f1768ece07284cde48fa7e33efa6a",
        "subject_fold_e713d5365a1199325444c995b43a997c",
    ]
    assert result.crossfit_id == ("subject_crossfit_115bffa3ab4a57bcfaebcf3bf4e13346")


def test_autonomous_program_use_scope_preserves_legacy_and_binds_biology() -> None:
    diagnostic = _spec()
    biological = replace(
        diagnostic,
        autonomous_program_use_scope="biological_analysis",
    )

    assert "autonomous_program_use_scope" not in diagnostic.to_dict()
    assert biological.to_dict()["autonomous_program_use_scope"] == (
        "biological_analysis"
    )
    assert biological.spec_id != diagnostic.spec_id
    repeated = replace(biological, repeat_index=1)
    assert repeated.autonomous_program_use_scope == "biological_analysis"
    assert repeated.spec_id == biological.spec_id
    assert repeated.repeat_id != biological.repeat_id
    with pytest.raises(ValueError, match="autonomous_program_use_scope"):
        replace(
            diagnostic,
            autonomous_program_use_scope="exploratory",  # type: ignore[arg-type]
        )


def test_biological_use_scope_rejects_nonbiological_registered_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = _trusted_target_resource(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="biological-reference"):
        replace(
            _spec(),
            autonomous_program_resource=synthetic,
            autonomous_program_use_scope="biological_analysis",
        )

    biological = _trusted_target_resource(
        tmp_path,
        monkeypatch,
        review_scope="biological_reference",
    )
    spec = replace(
        _spec(),
        autonomous_program_resource=biological,
        autonomous_program_use_scope="biological_analysis",
    )
    assert spec.autonomous_program_resource is biological
    assert biological.is_biological_reference_trusted


def test_biological_scope_rejects_unverified_resource_and_spoofed_subclass() -> None:
    unverified = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("T1", "T2"),
        program_ids=("generic_program",),
        resource_id="unverified-autonomous-programs",
        version="1",
        manifest_digest="f" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    with pytest.raises(ValueError, match="biological-reference"):
        replace(
            _spec(),
            autonomous_program_resource=unverified,
            autonomous_program_use_scope="biological_analysis",
        )

    class SpoofedResource(ReceiverAutonomousProgramResource):
        def _require_producer_owned(self) -> None:
            return None

        @property
        def is_biological_reference_trusted(self) -> bool:
            return True

    spoofed = object.__new__(SpoofedResource)
    with pytest.raises(TypeError, match="producer-owned"):
        replace(
            _spec(),
            autonomous_program_resource=spoofed,
            autonomous_program_use_scope="biological_analysis",
        )


@pytest.mark.parametrize("invalid", [True, -1, 2**63, 1.5])
def test_outer_fold_partition_seed_validation(invalid: object) -> None:
    with pytest.raises(ValueError, match="outer_fold_partition_seed"):
        replace(_spec(), outer_fold_partition_seed=invalid)


def test_outer_partition_seed_is_repeat_specific_and_deterministic() -> None:
    base = replace(_spec(), outer_fold_partition_seed=881902)
    repeated = replace(base, repeat_index=1)
    adata = _adata(("p1", "p2", "p3", "p4"))

    first = run_subject_crossfit(adata, _config(), _bundle(), _prior(), spec=base)
    rerun = run_subject_crossfit(adata, _config(), _bundle(), _prior(), spec=base)
    second_repeat = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=repeated,
    )

    assert first.fold_plan.to_dict() == rerun.fold_plan.to_dict()
    assert first.fold_plan.partition_seed_lineage is not None
    assert second_repeat.fold_plan.partition_seed_lineage is not None
    assert (
        first.fold_plan.partition_seed_lineage.to_dict()
        != second_repeat.fold_plan.partition_seed_lineage.to_dict()
    )


def test_explicit_outer_partition_seed_pairs_parameter_sensitivity() -> None:
    base = replace(_spec(), outer_fold_partition_seed=881902)
    changed_sender = replace(
        base.training_spec.sender_parameters,
        ligand_contrast_minimum_effect=0.02,
    )
    changed = replace(
        base,
        training_spec=replace(
            base.training_spec,
            sender_parameters=changed_sender,
        ),
    )
    adata = _adata(("p1", "p2", "p3", "p4"))
    first = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=base,
    )
    second = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=changed,
    )

    first_partitions = [
        (fold.train_subject_ids, fold.test_subject_ids)
        for fold in first.fold_plan.folds
    ]
    second_partitions = [
        (fold.train_subject_ids, fold.test_subject_ids)
        for fold in second.fold_plan.folds
    ]
    assert first_partitions == second_partitions
    assert first.spec.spec_id != second.spec.spec_id
    assert first.spec.repeat_id != second.spec.repeat_id
    assert first.fold_plan.plan_id != second.fold_plan.plan_id
    assert first.fold_plan.partition_seed_lineage is not None
    assert (
        first.fold_plan.partition_seed_lineage.to_dict()
        == second.fold_plan.partition_seed_lineage.to_dict()
    )

    def support_effects(result: CrossFitArtifacts) -> list[tuple[str, str, float]]:
        return [
            (support.receiver, support.interaction_id, cast(float, support.mean_effect))
            for fold in result.folds
            for functional in fold.training.sender_functionals
            for support in functional.contrast_supports
        ]

    assert support_effects(first) == support_effects(second)

    plan = first.fold_plan
    poisoned_plan = SubjectFoldPlan(
        repeat_id=plan.repeat_id,
        requested_n_splits=plan.requested_n_splits,
        effective_n_splits=plan.effective_n_splits,
        allowed_n_splits=plan.allowed_n_splits,
        subject_ids=plan.subject_ids,
        folds=plan.folds,
        seed_lineage=plan.seed_lineage,
        partition_seed_lineage=SeedLineage(123).derive("wrong-partition"),
        reduction_reason_code=plan.reduction_reason_code,
        rejected_candidate_reasons=plan.rejected_candidate_reasons,
    )
    with pytest.raises(ValueError, match="partition seed lineage"):
        CrossFitArtifacts._from_workflow(
            spec=first.spec,
            root_input_identity=first.root_input_identity,
            receiver_universe=first.receiver_universe,
            receiver_family_opportunity_universe=(
                first.receiver_family_opportunity_universe
            ),
            fold_plan=poisoned_plan,
            folds=first.folds,
            oof_coverage=first.oof_coverage,
            oof_receiver_coverage=first.oof_receiver_coverage,
            oof_sender_assignments=first.oof_sender_assignments,
            coverage_audit=first.coverage_audit,
        )


def test_repeated_crossfit_rejects_valid_children_from_different_root_inputs() -> None:
    base = _spec()
    config = _config()
    bundle = _bundle()
    prior = _prior()
    first_input = _adata()
    second_input = _adata()
    second_counts = second_input.layers["counts"].copy()
    second_counts.data[0] += 1
    second_input.layers["counts"] = second_counts
    first = run_subject_crossfit(
        first_input,
        config,
        bundle,
        prior,
        spec=base,
    )
    second = run_subject_crossfit(
        second_input,
        config,
        bundle,
        prior,
        spec=replace(base, repeat_index=1),
    )

    with pytest.raises(
        ValueError,
        match=r"does not derive from the root identity|run-level universe",
    ):
        CrossFitArtifacts._from_workflow(
            spec=first.spec,
            root_input_identity=second.root_input_identity,
            receiver_universe=second.receiver_universe,
            receiver_family_opportunity_universe=(
                second.receiver_family_opportunity_universe
            ),
            fold_plan=first.fold_plan,
            folds=first.folds,
            oof_coverage=first.oof_coverage,
            oof_receiver_coverage=first.oof_receiver_coverage,
            oof_sender_assignments=first.oof_sender_assignments,
            coverage_audit=first.coverage_audit,
        )
    with pytest.raises(ValueError, match="bound root input"):
        root_family_universe = freeze_receiver_family_opportunity_universe(
            prior,
            feature_ids=tuple(map(str, first_input.var_names)),
            receiver_ids=first.receiver_universe.receiver_ids,
            receiver_universe_id=first.receiver_universe.universe_id,
            receiver_axis_id=first.receiver_universe.receiver_axis_id,
            prior_content_id=training_module._target_prior_content_id(prior),
            root_input_identity_id=first.root_input_identity.identity_id,
            root_input_digest=first.root_input_identity.input_digest,
            cosine_threshold=base.family_cosine_threshold,
        )
        RepeatedCrossFitDiagnostics._from_workflow(
            spec=RepeatedCrossFitSpec(crossfit_spec=base, n_repeats=2),
            repeats=(first, second),
            root_input_identity=first.root_input_identity,
            resource_bundle_content_id=(
                training_module._resource_bundle_content_id(bundle)
            ),
            target_prior_content_id=training_module._target_prior_content_id(prior),
            receiver_family_opportunity_universe=root_family_universe,
        )


def test_root_input_identity_is_cell_row_order_invariant() -> None:
    adata = _adata()
    reversed_adata = adata[::-1].copy()

    original = training_module._sanitized_raw_input_identity(adata, _config())
    reordered = training_module._sanitized_raw_input_identity(
        reversed_adata,
        _config(),
    )

    assert original.identity_id == reordered.identity_id
    assert original.input_digest == reordered.input_digest
    assert original.subject_content_digests == reordered.subject_content_digests


def test_subject_crossfit_accepts_categorical_numeric_sample_ids() -> None:
    adata = _adata(("107", "1015", "1016", "1256"))
    for column in ("sample_id", "subject_id", "cell_type", "condition"):
        adata.obs[column] = pd.Categorical(
            adata.obs[column],
            categories=tuple(dict.fromkeys(adata.obs[column])),
        )

    result = _run(adata)

    assert result.completed_stage_oof_verified
    assert set(result.fold_plan.subject_ids) == {"107", "1015", "1016", "1256"}
    assert len(result.folds) == 2


def test_sanitized_snapshot_rejects_expression_or_metadata_mutation() -> None:
    snapshot = training_module._sanitized_raw_input_snapshot(_adata(), _config())
    counts = snapshot.adata.layers["counts"]

    with pytest.raises(ValueError, match="read-only"):
        counts.data[0] += 1

    replacement = counts.copy()
    snapshot.adata.layers["counts"] = replacement
    with pytest.raises(ContractError) as expression_error:
        snapshot._require_intact()
    assert expression_error.value.details.code == (
        "root_input_snapshot_integrity_violation"
    )

    bypass_snapshot = training_module._sanitized_raw_input_snapshot(
        _adata(),
        _config(),
    )
    bypass_counts = bypass_snapshot.adata.layers["counts"]
    bypass_counts.data.setflags(write=True)
    bypass_counts.data[0] += 1
    bypass_counts.data.setflags(write=False)
    with pytest.raises(ContractError) as bypass_error:
        bypass_snapshot._require_intact()
    assert bypass_error.value.details.code == (
        "root_input_snapshot_integrity_violation"
    )

    foreign_input = _adata()
    foreign_input.layers["counts"].data[0] += 1
    foreign_snapshot = training_module._sanitized_raw_input_snapshot(
        foreign_input,
        _config(),
    )
    object.__setattr__(foreign_snapshot, "identity", snapshot.identity)
    with pytest.raises(ContractError) as swapped_identity_error:
        foreign_snapshot._require_intact()
    assert swapped_identity_error.value.details.code == (
        "root_input_snapshot_integrity_violation"
    )

    metadata_snapshot = training_module._sanitized_raw_input_snapshot(
        _adata(),
        _config(),
    )
    metadata_snapshot.adata.obs.loc[
        metadata_snapshot.adata.obs.index[0], "condition"
    ] = "poisoned"
    with pytest.raises(ContractError) as metadata_error:
        metadata_snapshot._require_intact()
    assert metadata_error.value.details.code == (
        "root_input_snapshot_integrity_violation"
    )


def test_prepared_fold_derives_scope_digest_without_rehashing_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adata = _adata()
    identity = training_module._sanitized_raw_input_identity(adata, _config())
    selected_subjects = ("p1", "p2")
    scope = adata[adata.obs["subject_id"].astype(str).isin(selected_subjects)].copy()

    def fail_rehash(*args: object, **kwargs: object) -> None:
        raise AssertionError("fold fast path must not rehash cell rows")

    monkeypatch.setattr(training_module, "_subject_content_digests", fail_rehash)
    prepared = training_module._prepare_raw_fold(
        scope,
        _config(),
        min_cells=1,
        root_input_identity=identity,
    )

    assert prepared.input_digest == identity.scope_digest(selected_subjects)
    with pytest.raises(ContractError) as error:
        training_module._fit_training_artifacts_from_prepared(
            prepared,
            _config(),
            _bundle(),
            _prior(),
            spec=FoldTrainingSpec(
                min_cells=999,
                max_interactions=1,
                sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
            ),
        )
    assert error.value.details.code == "prepared_raw_fold_policy_mismatch"


def test_subject_crossfit_runs_real_fit_apply_and_exact_oof_audit() -> None:
    result = _run(_adata())

    registry = result.receiver_scoring_registry
    assert registry.registry_id == result.receiver_scoring_registry_id
    assert len(registry.collections) == len(result.folds) * len(result.spec.contrasts)
    folds_by_id = {fold.fold_id: fold for fold in result.folds}
    for collection in registry.collections:
        assert collection.planned_receivers == tuple(
            sorted(folds_by_id[collection.fold_id].training.cell_type_ids)
        )
        assert tuple(child.receiver for child in collection.children) == (
            collection.planned_receivers
        )
        assert collection.emitted_receivers == ()
        assert {child.registry_status for child in collection.children} == {
            "functional_not_produced"
        }
        assert {child.reason_code for child in collection.children} == {
            "family_common_functional_not_requested_without_penalty_tuning"
        }

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
        functional.training_input_digest == fold.training.training_input_digest
        and functional.frozen_interaction_ids
        == fold.training.frozen_interaction_universe.interaction_ids
        for fold in result.folds
        for functional in fold.training.sender_functionals
    )
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
    expected_family_edge_policy = (
        "max_sender_local_availability_with_frozen_train_only_ligand_gate_v2"
    )
    assert manifest["family_edge_evidence_policy"] == expected_family_edge_policy
    assert all(
        artifact["edge_evidence_policy_id"] == expected_family_edge_policy
        for fold_artifact in manifest["fold_artifacts"]
        for artifact in fold_artifact["family_common_scoring_artifacts"]
    )
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
    prepared = training_module._prepare_raw_fold(
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
    assert (
        crossfit_module._training_response_lineage_reason(
            prepared,
            contexts=reference_contexts,
            sample_ids=sample_ids[1:],
            subject_ids=subject_ids[1:],
            context_ids=context_ids[1:],
        )
        == "training_reference_expression_incomplete_sample_coverage"
    )
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


def test_subject_crossfit_mixed_design_uses_parented_cr2_response() -> None:
    result = _run(_mixed_adata())
    chains = [
        (response, precision, model, response_application, application)
        for fold in result.folds
        for response, precision, model, response_application, application in zip(
            fold.receiver_responses,
            fold.response_precisions,
            fold.receiver_incremental_models,
            fold.receiver_response_applications,
            fold.receiver_incremental_applications,
            strict=True,
        )
        if model.receiver == "Receiver"
    ]

    assert chains
    for response, precision, model, response_application, application in chains:
        assert isinstance(response, RepeatedMeasuresFoldResponseArtifact)
        assert response.status == "ok"
        assert response.cr2_backend_eligible
        assert not response.is_cr1_exploratory
        assert response.formal_inference_allowed is False
        assert precision.lineage_mode == (
            "repeated_measures_cr2_fold_response_parented_v1"
        )
        assert precision.method == "repeated_cr2_standardized_inverse_variance_v1"
        assert precision.response_artifact_id == response.artifact_id
        assert precision.estimable
        assert model.diagnostic_functional is not None
        assert model.diagnostic_functional.loss_design == (
            "mixed_subject_equal_full_prediction_loss_v1"
        )
        assert model.diagnostic_status == "observed"
        assert model.official_incremental_status == "not_estimable"
        assert model.reason_code == "receiver_autonomous_nuisance_not_frozen"
        assert response_application.status == "ok"
        assert application.diagnostic_status == "observed"
        assert application.official_incremental_status == "not_estimable"
        assert application.reason_code == "receiver_autonomous_nuisance_not_frozen"
        assert not application.is_oof_certified

    registry = result.receiver_scoring_registry
    assert registry.is_authoritative_registry
    assert registry.planned_collections

    first_response, first_precision, *_ = chains[0]
    object.__setattr__(first_precision, "response_artifact_id", "forged-response")
    with pytest.raises(ContractError) as precision_error:
        first_precision.require_response_compatible(first_response)
    assert precision_error.value.details.code == (
        "precision_transform_integrity_violation"
    )

    second_response = chains[1][0]
    object.__setattr__(second_response, "repeated_effect_artifact_id", "forged-effect")
    with pytest.raises(ContractError) as response_error:
        second_response.to_dict()
    assert response_error.value.details.code == (
        "repeated_fold_response_integrity_violation"
    )


def test_subject_crossfit_mixed_design_runs_subject_blocked_inner_tuning() -> None:
    spec = replace(
        _spec(),
        penalty_tuning_spec=PenaltyTuningSpec(
            lambda1_fractions=(1.0,),
            lambda2_fractions=(0.0,),
            inner_allowed_n_splits=(2,),
            min_inner_train_subjects_per_context=2,
            min_inner_validation_subjects_per_context=1,
        ),
    )
    result = run_subject_crossfit(
        _mixed_adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )
    models = [
        model
        for fold in result.folds
        for model in fold.receiver_incremental_models
        if model.receiver == "Receiver"
    ]

    assert models
    for model in models:
        plan = model.inner_fold_plan
        tuning = model.penalty_tuning_artifact
        assert plan is not None
        assert tuning is not None
        assert tuning.status == "selected"
        assert tuning.reason_code is None
        assert tuning.inner_fold_plan_id == plan.plan_id
        assert tuning.validation_loss_estimand is (
            PenaltyValidationLossEstimand.MIXED_SUBJECT_PREDICTION
        )
        assert tuning.is_oof_certified
        assert tuning.training_subject_ids == model.training_subject_ids
        fold_by_id = {fold.fold_id: fold for fold in plan.folds}
        for evaluation in tuning.evaluations:
            inner_fold = fold_by_id[evaluation.inner_fold_id]
            assert evaluation.inner_fold_manifest_id == inner_fold.fold_id
            assert evaluation.inner_training_subject_ids == (
                inner_fold.train_subject_ids
            )
            assert evaluation.validation_subject_ids == inner_fold.test_subject_ids
            assert set(evaluation.inner_training_subject_ids).isdisjoint(
                evaluation.validation_subject_ids
            )
            assert set(evaluation.inner_training_subject_ids).union(
                evaluation.validation_subject_ids
            ) == set(model.training_subject_ids)
            assert evaluation.validation_loss_estimand is (
                PenaltyValidationLossEstimand.MIXED_SUBJECT_PREDICTION
            )
            assert evaluation.training_functional_id is not None
            assert evaluation.heldout_application_id is not None
            assert np.all(np.isfinite(evaluation.subject_losses))
            assert np.all(evaluation.subject_losses >= 0.0)
        assert model.diagnostic_functional is not None
        assert model.diagnostic_functional.loss_design == (
            "mixed_subject_equal_full_prediction_loss_v1"
        )
        assert model.diagnostic_status == "observed"
        assert model.official_incremental_status == "not_estimable"
        assert model.reason_code == "receiver_autonomous_nuisance_not_frozen"


def test_subject_crossfit_mixed_design_fails_closed_below_cluster_minimum() -> None:
    result = _run(_mixed_adata(n_paired=4, n_single_per_context=2))
    chains = [
        (response, precision, model, application)
        for fold in result.folds
        for response, precision, model, application in zip(
            fold.receiver_responses,
            fold.response_precisions,
            fold.receiver_incremental_models,
            fold.receiver_incremental_applications,
            strict=True,
        )
        if model.receiver == "Receiver"
    ]

    assert chains
    assert all(
        isinstance(response, RepeatedMeasuresFoldResponseArtifact)
        and response.status == "not_estimable"
        and not response.is_cr1_exploratory
        and not response.cr2_backend_eligible
        and response.reason_code == "insufficient_subject_clusters"
        and not precision.estimable
        and precision.method == "repeated_cr2_standardized_inverse_variance_v1"
        and model.diagnostic_status == "not_estimable"
        and model.diagnostic_reason_code == "insufficient_subject_clusters"
        and model.official_incremental_status == "not_estimable"
        and application.diagnostic_status == "not_estimable"
        for response, precision, model, application in chains
    )


def test_subject_crossfit_multi_context_repetition_uses_cr2_response() -> None:
    base = _spec()
    contrast = balanced_contrast(
        ("c",),
        ("a", "b"),
        name="c_vs_a_b",
    )
    spec = replace(
        base,
        contrasts=(contrast,),
        training_spec=replace(
            base.training_spec,
            sender_contrasts=(contrast,),
        ),
    )
    result = run_subject_crossfit(
        _repeated_multi_context_adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )
    chains = [
        (response, model, application)
        for fold in result.folds
        for response, model, application in zip(
            fold.receiver_responses,
            fold.receiver_incremental_models,
            fold.receiver_incremental_applications,
            strict=True,
        )
        if model.receiver == "Receiver"
    ]

    assert chains
    assert all(
        isinstance(response, RepeatedMeasuresFoldResponseArtifact)
        and response.status == "ok"
        and response.cr2_backend_eligible
        and not response.is_cr1_exploratory
        and not response.formal_inference_allowed
        and response.repeated_design.n_repeated_subject_clusters > 0
        and len(response.repeated_design.contrast_context_ids) == 3
        and model.diagnostic_functional is not None
        and model.diagnostic_functional.loss_design
        == "mixed_subject_equal_full_prediction_loss_v1"
        and model.diagnostic_status == "observed"
        and model.official_incremental_status == "not_estimable"
        and model.reason_code == "receiver_autonomous_nuisance_not_frozen"
        and application.diagnostic_status == "observed"
        and application.official_incremental_status == "not_estimable"
        and application.reason_code == "receiver_autonomous_nuisance_not_frozen"
        for response, model, application in chains
    )


def _independent_tuning_crossfit_spec() -> CrossFitSpec:
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
    return CrossFitSpec(
        contrasts=base.contrasts,
        training_spec=base.training_spec,
        allowed_n_splits=base.allowed_n_splits,
        autonomous_program_resource=resource,
        penalty_tuning_spec=tuning_spec,
    )


def test_independent_group_inner_one_se_tuning_is_subject_blocked() -> None:
    result = run_subject_crossfit(
        _independent_adata(8),
        _config(),
        _bundle(),
        _prior(),
        spec=_independent_tuning_crossfit_spec(),
    )

    receiver_models = [
        model
        for fold in result.folds
        for model in fold.receiver_incremental_models
        if model.receiver == "Receiver"
    ]
    assert receiver_models
    assert all(model.penalty_tuning_artifact is not None for model in receiver_models)
    assert all(model.inner_fold_plan is not None for model in receiver_models)
    for model in receiver_models:
        tuning = model.penalty_tuning_artifact
        assert tuning is not None
        assert tuning.status == "selected"
        assert tuning.reason_code is None
        assert tuning.validation_loss_estimand is (
            PenaltyValidationLossEstimand.INDEPENDENT_SUBJECT_PREDICTION
        )
        assert tuning.is_oof_certified
        assert tuning.training_subject_ids == model.training_subject_ids
        for evaluation in tuning.evaluations:
            assert set(evaluation.inner_training_subject_ids).isdisjoint(
                evaluation.validation_subject_ids
            )
            assert set(evaluation.inner_training_subject_ids).union(
                evaluation.validation_subject_ids
            ) == set(model.training_subject_ids)
            assert evaluation.validation_loss_estimand is (
                PenaltyValidationLossEstimand.INDEPENDENT_SUBJECT_PREDICTION
            )
            assert np.all(np.isfinite(evaluation.subject_losses))
            assert np.all(evaluation.subject_losses >= 0)
        assert model.diagnostic_functional is not None
        assert model.diagnostic_functional.loss_design == (
            "independent_subject_pseudocontrasts_v1"
        )


def test_independent_group_inner_one_se_low_support_fails_closed() -> None:
    result = run_subject_crossfit(
        _independent_adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_independent_tuning_crossfit_spec(),
    )

    receiver_models = [
        model
        for fold in result.folds
        for model in fold.receiver_incremental_models
        if model.receiver == "Receiver"
    ]
    assert receiver_models
    for model in receiver_models:
        assert model.inner_fold_plan is None
        assert model.penalty_tuning_artifact is not None
        assert model.penalty_tuning_artifact.status == "not_estimable"
        assert model.penalty_tuning_artifact.reason_code == (
            "no_estimable_subject_fold_plan"
        )
        assert model.penalty_tuning_artifact.validation_loss_estimand is (
            PenaltyValidationLossEstimand.INDEPENDENT_SUBJECT_PREDICTION
        )
        assert model.diagnostic_functional is None
        assert model.diagnostic_status == "not_estimable"


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

    registry = result.receiver_scoring_registry
    registered_ids = {
        child.scoring_functional_id
        for collection in registry.collections
        for child in collection.children
    }
    expected_ids = {
        functional.family_common_functional_id
        for fold in result.folds
        for functional in fold.family_common_functionals
    }
    assert registered_ids == expected_ids
    assert all(
        child.registry_status == "functional_registered"
        and child.receiver_family_model_id
        and child.receiver_incremental_model_id
        and child.filter_universe_id
        and child.score_version
        for collection in registry.collections
        for child in collection.children
    )

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
        communication_modes=tuple(
            sorted(mode.value for mode in _config().communication_modes)
        ),
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


def test_fold_learned_latent_nuisance_runs_through_full_crossfit_chain(
    tmp_path: Path,
) -> None:
    base = _spec()
    latent_spec = FrozenLatentNuisanceSpec(
        max_components=1,
        min_control_features=2,
        min_training_subjects=2,
        minimum_explained_fraction=0.01,
    )
    spec = CrossFitSpec(
        contrasts=base.contrasts,
        training_spec=base.training_spec,
        allowed_n_splits=(2,),
        latent_nuisance_spec=latent_spec,
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
        (model, application, common_functional, common_application)
        for fold in result.folds
        for model, application, common_functional, common_application in zip(
            fold.receiver_incremental_models,
            fold.receiver_incremental_applications,
            fold.family_common_functionals,
            fold.family_common_applications,
            strict=True,
        )
        if model.receiver == "Receiver"
    ]
    assert receiver_chains and len(receiver_chains) == len(result.folds)
    for model, application, common_functional, common_application in receiver_chains:
        assert model.latent_nuisance_spec_id == latent_spec.spec_id
        assert model.latent_nuisance_artifact is not None
        assert model.latent_nuisance_artifact.status == "observed"
        assert model.latent_nuisance_artifact.training_subject_ids == (
            model.training_subject_ids
        )
        assert model.autonomous_program_resource_id is None
        assert model.autonomous_program_source_id == (
            model.diagnostic_functional.autonomous_basis_id
            if model.diagnostic_functional is not None
            else None
        )
        assert model.inner_fold_plan is not None
        assert model.penalty_tuning_artifact is not None
        assert model.penalty_tuning_artifact.status == "selected"
        assert model.is_oof_certified
        assert application.is_oof_certified
        assert application.official_incremental_status == "observed"
        assert common_functional.incremental_functional is not None
        assert common_functional.autonomous_program_resource_id == (
            model.autonomous_program_source_id
        )
        assert common_application.heldout_reason_code is None

    manifest = result.to_manifest()
    assert "receiver_autonomous_nuisance" not in manifest["remaining_stages"]
    receiver_records = [
        record
        for fold in manifest["fold_artifacts"]
        for record in fold["receiver_incremental_artifacts"]
        if record["receiver"] == "Receiver"
    ]
    assert receiver_records
    assert all(
        record["latent_nuisance_spec_id"] == latent_spec.spec_id
        and record["latent_nuisance_artifact_id"] is not None
        and record["latent_nuisance_artifact"]["status"] == "observed"
        and record["autonomous_program_source_id"] is not None
        for record in receiver_records
    )
    audit = audit_crossfit_oof_readiness(result)
    nuisance_requirements = [
        record
        for record in audit.requirement_ledger
        if record.requirement_code
        == "approved_receiver_autonomous_nuisance_policy_present"
    ]
    assert len(nuisance_requirements) == 1
    assert nuisance_requirements[0].satisfied
    assert latent_spec.spec_id in nuisance_requirements[0].evidence_ids
    persisted = write_crossfit_result(result, tmp_path / "latent-crossfit")
    loaded = CrossFitResult.load(persisted.path)
    loaded_receiver_records = [
        record
        for fold in loaded.manifest["source_crossfit_manifest"]["fold_artifacts"]
        for record in fold["receiver_incremental_artifacts"]
        if record["receiver"] == "Receiver"
    ]
    assert loaded_receiver_records == receiver_records
    manifest_path = persisted.path / "crossfit_manifest.json"
    status_path = persisted.path / "_status.json"
    forged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged_source = forged_manifest["source_crossfit_manifest"]
    forged_records = [
        record
        for fold in forged_source["fold_artifacts"]
        for record in fold["receiver_incremental_artifacts"]
        if record["receiver"] == "Receiver"
    ]
    forged_records[0]["latent_nuisance_artifact"]["control_feature_ids"][0] = "T1"
    forged_manifest["source_crossfit_manifest_digest"] = canonical_digest(forged_source)
    forged_manifest["crossfit_result_id"] = stable_id(
        "crossfit_result",
        {
            key: value
            for key, value in forged_manifest.items()
            if key != "crossfit_result_id"
        },
        schema_version="1",
    )
    manifest_path.write_text(
        f"{canonical_json(forged_manifest)}\n",
        encoding="utf-8",
    )
    forged_status = json.loads(status_path.read_text(encoding="utf-8"))
    forged_status["crossfit_result_id"] = forged_manifest["crossfit_result_id"]
    status_path.write_text(
        f"{canonical_json(forged_status)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ResultValidationError) as tamper_error:
        CrossFitResult.load(persisted.path)
    assert tamper_error.value.details.code == "invalid_crossfit_result_manifest"


def test_trusted_tuned_all_receiver_chains_pass_descriptive_oof_audit(
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
        gain_calibration_spec=GainCalibrationSpec(
            min_subjects=4,
            min_supported_families=1,
            min_subjects_per_family=4,
            min_positive_observations=4,
            min_distinct_positive_gains=4,
        ),
    )
    adata = _adata(tuple(f"p{index}" for index in range(1, 9)))
    counts = np.asarray(adata.layers["counts"].toarray(), dtype=np.int64)
    sender_mask = adata.obs["cell_type"].astype(str).eq("Sender").to_numpy()
    counts[sender_mask, 1] = counts[sender_mask, 0]
    counts[sender_mask, 3] = counts[sender_mask, 2]
    subject_number = (
        adata.obs["subject_id"].astype(str).str.removeprefix("p").astype(int).to_numpy()
    )
    stim_mask = adata.obs["condition"].astype(str).eq("stim").to_numpy()
    counts[stim_mask, 4] = 13 + 2 * subject_number[stim_mask]
    counts[sender_mask & stim_mask, 0] *= 3
    adata.layers["counts"] = sparse.csr_matrix(counts)

    result = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )
    audit = audit_crossfit_oof_readiness(result)

    assert audit.complete
    assert audit.is_oof_descriptive_certified
    assert audit.formal_inference_allowed is False
    assert audit.failed_requirements == ()
    active_universe = freeze_crossfit_active_edge_universe(
        result,
        contrast_id_or_name="stim_vs_control",
    )
    active_points = adapt_crossfit_active_edge_point_records(
        result,
        active_universe,
    )
    assert active_points.active_edge_universe_id == active_universe.universe_id
    assert tuple(record.candidate_edge_id for record in active_points.records) == (
        active_universe.candidate_edge_ids
    )
    assert len(active_points.records) == len(active_universe.candidates) > 0
    assert {record.score_version for record in active_points.records} == {
        active_universe.score_version
    }
    assert all(
        model.status == "observed" and application.status == "observed"
        for fold in result.folds
        for model, application in zip(
            fold.receiver_program_models,
            fold.receiver_program_applications,
            strict=True,
        )
    )
    assert all(
        model.is_oof_certified and application.is_oof_certified
        for fold in result.folds
        for model, application in zip(
            fold.receiver_incremental_models,
            fold.receiver_incremental_applications,
            strict=True,
        )
    )
    for fold in result.folds:
        assert len(fold.cross_receiver_common_functionals) == len(spec.contrasts)
        assert len(fold.cross_receiver_common_applications) == len(spec.contrasts)
        global_functional = fold.cross_receiver_common_functionals[0]
        global_application = fold.cross_receiver_common_applications[0]
        assert not global_functional.common_functional_across_receivers
        assert global_functional.receiver_balanced_descriptive_collection
        assert global_functional.receiver_ids == tuple(
            sorted(fold.training.cell_type_ids)
        )
        assert (
            tuple(
                receiver for receiver, _ in global_functional.receiver_gain_calibrations
            )
            == global_functional.receiver_ids
        )
        assert all(
            calibration is not None and calibration.is_estimable
            for _, calibration in global_functional.receiver_gain_calibrations
        )
        assert all(
            calibration is not None
            and calibration.n_positive_observations == 4
            and calibration.n_distinct_positive_gains == 4
            for _, calibration in global_functional.receiver_gain_calibrations
        )
        assert global_functional.all_receivers_gain_calibrated
        assert global_functional.cross_receiver_percentile_rank_eligible
        assert global_application.functional.global_common_functional_id == (
            global_functional.global_common_functional_id
        )
        lr_scores = global_application.global_lr_scores
        sender_scores = global_application.global_sender_lr_scores
        assert set(lr_scores["global_common_functional_id"]) == {
            global_functional.global_common_functional_id
        }
        assert set(sender_scores["global_common_functional_id"]) == {
            global_functional.global_common_functional_id
        }
        assert "within_family_lr_weight" not in lr_scores
        assert "assignment_weight" in sender_scores
        assert "normalized_entropy" in sender_scores
        assert "training_receiver_scale_factor" not in lr_scores
        assert "training_receiver_scale_factor" not in sender_scores
        assert "calibrated_family_gain_percentile" in lr_scores
        assert "gain_calibration_binding_id" in sender_scores
        observed_lr = lr_scores.loc[lr_scores["status"].eq("observed")]
        observed_sender = sender_scores.loc[sender_scores["status"].eq("observed")]
        assert not observed_lr.empty
        assert not observed_sender.empty
        assert observed_lr["global_lr_score"].between(0.0, 1.0).all()
        assert observed_sender["global_sender_lr_score"].between(0.0, 1.0).all()
        assert (
            observed_sender["global_sender_lr_score"]
            <= observed_sender["global_lr_score"] + 1e-12
        ).all()
        assert np.allclose(
            observed_sender["global_sender_lr_score"],
            observed_sender["global_lr_score"] * observed_sender["assignment_weight"],
        )
        sender_group_keys = [
            "sample_id",
            "subject_id",
            "context_id",
            "receiver",
            "interaction_id",
            "mode",
        ]
        for _, sender_group in observed_sender.groupby(
            sender_group_keys, observed=True, sort=False
        ):
            assert sender_group["assignment_weight"].sum() == pytest.approx(1.0)
            assert sender_group["global_sender_lr_score"].sum() == pytest.approx(
                sender_group["global_lr_score"].iloc[0]
            )

    persisted = write_crossfit_result(result, tmp_path / "certified-crossfit")
    assert persisted.manifest["complete_pipeline_oof_certified"] is True
    assert persisted.manifest["claim_scope"] == CROSSFIT_OOF_DESCRIPTIVE_SCOPE
    assert persisted.manifest["certification_status"] == (
        CROSSFIT_OOF_DESCRIPTIVE_SCOPE
    )
    assert persisted.manifest["formal_inference_status"] == (
        "not_available_descriptive_only"
    )
    components = persisted.read_components()
    differential = persisted.read_descriptive_differential()
    assert components["is_oof_certified"].all()
    assert differential["is_oof_certified"].all()
    assert set(components["claim_scope"]) == {CROSSFIT_OOF_DESCRIPTIVE_SCOPE}
    assert set(differential["claim_scope"]) == {CROSSFIT_OOF_DESCRIPTIVE_SCOPE}


def test_explicit_outer_seed_pairs_inner_tuning_parameter_sensitivity(
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
    first_spec = CrossFitSpec(
        contrasts=base.contrasts,
        outer_fold_partition_seed=99173,
        training_spec=base.training_spec,
        allowed_n_splits=(2,),
        autonomous_program_resource=resource,
        penalty_tuning_spec=tuning_spec,
    )
    changed_sender = replace(
        first_spec.training_spec.sender_parameters,
        ligand_contrast_minimum_effect=0.02,
    )
    second_spec = replace(
        first_spec,
        training_spec=replace(
            first_spec.training_spec,
            sender_parameters=changed_sender,
        ),
    )
    adata = _adata(tuple(f"p{index}" for index in range(1, 9)))

    first = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=first_spec,
    )
    second = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=second_spec,
    )

    def inner_plans(
        result: CrossFitArtifacts,
    ) -> dict[
        tuple[tuple[str, ...], tuple[str, ...], str, str],
        tuple[
            tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
            dict[str, object],
            str,
            str,
        ],
    ]:
        manifests = {fold.fold_id: fold for fold in result.fold_plan.folds}
        records = {}
        for fold in result.folds:
            outer = manifests[fold.fold_id]
            for model in fold.receiver_incremental_models:
                plan = model.inner_fold_plan
                tuning = model.penalty_tuning_artifact
                if plan is None or tuning is None:
                    continue
                assert plan.partition_seed_lineage is not None
                key = (
                    outer.train_subject_ids,
                    outer.test_subject_ids,
                    model.receiver,
                    model.contrast_name,
                )
                records[key] = (
                    tuple(
                        (inner.train_subject_ids, inner.test_subject_ids)
                        for inner in plan.folds
                    ),
                    plan.partition_seed_lineage.to_dict(),
                    plan.plan_id,
                    tuning.tuning_id,
                )
        return records

    first_plans = inner_plans(first)
    second_plans = inner_plans(second)
    assert first_plans
    assert set(first_plans) == set(second_plans)
    for key in first_plans:
        first_partitions, first_lineage, first_plan_id, first_tuning_id = first_plans[
            key
        ]
        second_partitions, second_lineage, second_plan_id, second_tuning_id = (
            second_plans[key]
        )
        assert first_partitions == second_partitions
        assert first_lineage == second_lineage
        assert first_plan_id != second_plan_id
        assert first_tuning_id != second_tuning_id


def test_repeated_crossfit_refits_complete_children_and_emits_no_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _spec()
    resource = _trusted_target_resource(tmp_path, monkeypatch)
    crossfit_spec = CrossFitSpec(
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
    repeat_spec = RepeatedCrossFitSpec(
        crossfit_spec=crossfit_spec,
        n_repeats=2,
    )
    caller_input = _adata(tuple(f"p{index}" for index in range(1, 9)))
    original_run = repeated_crossfit_module._run_subject_crossfit
    observed_snapshots: list[training_module.SanitizedRawInputSnapshot] = []

    def run_and_mutate_caller(
        snapshot: training_module.SanitizedRawInputSnapshot,
        *args: object,
        **kwargs: object,
    ):
        observed_snapshots.append(snapshot)
        if len(observed_snapshots) == 1:
            caller_input.layers["counts"].data[0] += 999
        return original_run(snapshot, *args, **kwargs)

    monkeypatch.setattr(
        repeated_crossfit_module,
        "_run_subject_crossfit",
        run_and_mutate_caller,
    )

    result = run_repeated_subject_crossfit(
        caller_input,
        _config(),
        _bundle(),
        _ligand_prior_with_unmapped_driver(),
        spec=repeat_spec,
    )

    assert isinstance(result, RepeatedCrossFitDiagnostics)
    assert len(observed_snapshots) == 2
    assert observed_snapshots[0] is observed_snapshots[1]
    assert observed_snapshots[0].adata is not caller_input
    assert result.formal_inference_status == (
        "not_computed_repeated_crossfit_diagnostic_only"
    )
    assert result.is_inference_eligible is False
    assert [item.spec.repeat_index for item in result.repeats] == [0, 1]
    assert len({item.spec.repeat_id for item in result.repeats}) == 2
    assert len({item.crossfit_id for item in result.repeats}) == 2
    assert len({item.receiver_universe.universe_id for item in result.repeats}) == 1
    assert (
        len({item.receiver_universe.receiver_axis_id for item in result.repeats}) == 1
    )
    registry = result.repeat_registry
    assert len(registry) == 2
    assert registry["partition_id"].nunique() == 2
    assert registry["receiver_universe_id"].nunique() == 1
    assert registry["receiver_axis_id"].nunique() == 1
    assert (
        registry["n_receiver_fold_opportunities"]
        == registry["n_receivers"] * registry["n_folds"]
    ).all()
    assert registry["n_receiver_fold_not_estimable"].ge(0).all()
    assert registry["oof_coverage_complete"].all()
    assert registry["family_common_stage_connected"].all()
    assert result.diagnostic_status == (
        "partially_observed_descriptive_repeat_stability"
    )

    events = result.family_fold_events
    values = result.subject_family_repeat_values
    stability = result.family_repeat_stability
    subject_points = result.subject_family_point_estimates
    family_points = result.family_point_estimates
    assert not events.empty and not values.empty and not stability.empty
    assert not subject_points.empty and not family_points.empty
    assert not subject_points.duplicated(
        ["subject_id", "contrast_name", "receiver", "family_id"]
    ).any()
    assert not family_points.duplicated(
        ["contrast_name", "receiver", "family_id"]
    ).any()
    assert set(subject_points["status"]) <= {"observed", "not_estimable"}
    assert set(family_points["status"]) <= {"observed", "not_estimable"}
    assert set(family_points["formal_inference_status"]) == {
        "not_computed_repeated_crossfit_diagnostic_only"
    }
    assert any("UNMAPPED" in driver_ids for driver_ids in stability["driver_ids"])
    assert not events.duplicated(
        [
            "repeat_index",
            "fold_id",
            "contrast_name",
            "receiver",
            "family_id",
        ]
    ).any()
    assert not values.duplicated(
        [
            "repeat_index",
            "subject_id",
            "contrast_name",
            "receiver",
            "family_id",
        ]
    ).any()
    assert set(stability["n_subject_repeat_opportunities"]) == {16}
    assert set(stability["n_distinct_partitions_overall"]) == {2}
    fit_estimable = stability.loc[stability["fit_estimable_fraction"].gt(0.0)]
    assert not fit_estimable.empty
    assert set(fit_estimable["n_distinct_complete_selection_partitions"]) == {2}
    assert fit_estimable["diagnostic_conditional_fit_selection_fraction"].notna().all()
    structural = stability.loc[stability["family_estimable_fraction"].eq(0.0)]
    assert not structural.empty
    assert set(structural["status"]) == {"not_estimable"}
    assert structural["structurally_determined_fraction"].eq(1.0).all()
    assert set(structural["effect_stability_status"]) == {"not_estimable"}
    unequal_target = fit_estimable.iloc[0]
    unequal = values.loc[
        values["contrast_name"].eq(unequal_target["contrast_name"])
        & values["receiver"].eq(unequal_target["receiver"])
        & values["family_id"].eq(unequal_target["family_id"])
    ].copy()
    for repeat_index, repeat_rows in unequal.groupby("repeat_index", sort=True):
        ordered_subjects = sorted(repeat_rows["subject_id"].astype(str).unique())
        selected_subjects = set(ordered_subjects[:5])
        selected_rows = unequal["repeat_index"].eq(repeat_index) & unequal[
            "subject_id"
        ].isin(selected_subjects)
        other_rows = unequal["repeat_index"].eq(repeat_index) & ~unequal[
            "subject_id"
        ].isin(selected_subjects)
        unequal.loc[selected_rows, "fold_id"] = f"unequal-a-{repeat_index}"
        unequal.loc[other_rows, "fold_id"] = f"unequal-b-{repeat_index}"
        unequal.loc[selected_rows, "family_selected"] = True
        unequal.loc[other_rows, "family_selected"] = False
    unequal["family_available"] = True
    unequal["family_estimable"] = True
    unequal["effect_status"] = "observed"
    unequal["bounded_incremental_gain"] = 0.1
    unequal_stability = repeated_crossfit_module._build_family_stability(
        unequal,
        repeats=result.repeats,
        partition_by_repeat={
            int(row.repeat_index): str(row.partition_id)
            for row in registry.itertuples(index=False)
        },
    ).iloc[0]
    assert unequal_stability[
        "repeat_subject_exposure_selection_fraction_median"
    ] == pytest.approx(5 / 8)
    assert unequal_stability["repeat_fit_selection_fraction_median"] == 0.5
    assert set(stability["formal_inference_status"]) == {
        "not_computed_repeated_crossfit_diagnostic_only"
    }
    assert {
        "p_value",
        "q_value",
        "comm_probability",
        "specificity_support",
        "confidence_interval",
    }.isdisjoint(stability.columns)
    incomplete = values.copy(deep=True)
    target = stability.iloc[0]
    target_family_rows = (
        incomplete["contrast_name"].eq(target["contrast_name"])
        & incomplete["receiver"].eq(target["receiver"])
        & incomplete["family_id"].eq(target["family_id"])
    )
    target_pairs = set(
        incomplete.loc[
            target_family_rows & incomplete["subject_id"].eq("p1"),
            ["repeat_index", "fold_id"],
        ].itertuples(index=False, name=None)
    )
    target_rows = target_family_rows & pd.Series(
        list(
            zip(
                incomplete["repeat_index"],
                incomplete["fold_id"],
                strict=True,
            )
        ),
        index=incomplete.index,
    ).isin(target_pairs)
    incomplete.loc[target_rows, "family_available"] = False
    incomplete.loc[target_rows, "family_estimable"] = False
    incomplete["family_selected"] = incomplete["family_selected"].astype(object)
    incomplete.loc[target_rows, "family_selected"] = None
    incomplete.loc[target_rows, "bounded_incremental_gain"] = None
    incomplete_stability = repeated_crossfit_module._build_family_stability(
        incomplete,
        repeats=result.repeats,
        partition_by_repeat={
            int(row.repeat_index): str(row.partition_id)
            for row in registry.itertuples(index=False)
        },
    )
    incomplete_row = incomplete_stability.loc[
        incomplete_stability["contrast_name"].eq(target["contrast_name"])
        & incomplete_stability["receiver"].eq(target["receiver"])
        & incomplete_stability["family_id"].eq(target["family_id"])
    ].iloc[0]
    assert incomplete_row["status"] == "not_estimable"
    assert incomplete_row["reason_code"] == (
        "insufficient_complete_estimable_family_coverage"
    )
    assert incomplete_row["n_repeats_with_observed_selection"] == 0
    assert pd.isna(incomplete_row["repeat_subject_exposure_selection_fraction_maximum"])
    assert pd.isna(incomplete_row["repeat_fit_selection_fraction_maximum"])
    assert incomplete_row["effect_stability_status"] == "not_estimable"
    assert incomplete_row["effect_stability_reason_code"] == (
        "insufficient_complete_repeat_effect_coverage"
    )
    assert incomplete_row["n_repeats_with_observed_effect"] == 0
    manifest = result.to_manifest()
    assert manifest["receiver_universe_id"] == (
        result.repeats[0].receiver_universe.universe_id
    )
    assert manifest["receiver_axis_id"] == (
        result.repeats[0].receiver_universe.receiver_axis_id
    )
    assert manifest["receiver_ids"] == list(
        result.repeats[0].receiver_universe.receiver_ids
    )
    assert manifest["inferential_fields_available"] == []
    assert manifest["is_inference_eligible"] is False
    assert manifest["diagnostics_schema_version"] == "3.0.0"
    assert manifest["repeat_point_estimates_formal_inference_allowed"] is False
    assert manifest["n_subject_family_point_estimate_rows"] == len(subject_points)
    assert manifest["n_family_point_estimate_rows"] == len(family_points)
    assert manifest["family_universe_policy"] == (
        "root_target_prior_feature_receiver_opportunity_universe_v1"
    )
    assert manifest["receiver_family_opportunity_universe_id"] == (
        result.receiver_family_opportunity_universe.universe_id
    )
    assert manifest["receiver_family_opportunity_universe"] == (
        result.receiver_family_opportunity_universe.to_dict()
    )
    assert manifest["selection_stability_status_counts"]["not_estimable"] > 0
    assert manifest["effect_stability_status_counts"] == {
        "not_estimable": len(stability)
    }
    with pytest.raises(TypeError, match="producer-owned"):
        RepeatedCrossFitDiagnostics()

    original_collect_views = repeated_crossfit_module._collect_fold_views

    def omit_registered_view(*args: object, **kwargs: object):
        views = original_collect_views(*args, **kwargs)
        views.pop(next(iter(views)))
        return views

    monkeypatch.setattr(
        repeated_crossfit_module,
        "_collect_fold_views",
        omit_registered_view,
    )
    with pytest.raises(ContractError) as registry_error:
        RepeatedCrossFitDiagnostics._from_workflow(
            spec=result.spec,
            repeats=result.repeats,
            root_input_identity=result.root_input_identity,
            resource_bundle_content_id=result.resource_bundle_content_id,
            target_prior_content_id=result.target_prior_content_id,
            receiver_family_opportunity_universe=(
                result.receiver_family_opportunity_universe
            ),
        )
    assert registry_error.value.details.code == (
        "repeated_crossfit_receiver_registry_mismatch"
    )
    monkeypatch.setattr(
        repeated_crossfit_module,
        "_collect_fold_views",
        original_collect_views,
    )

    private = object.__getattribute__(result, "_family_repeat_stability")
    private.loc[
        private.index[0],
        "diagnostic_conditional_fit_selection_fraction",
    ] = 0.123456789
    with pytest.raises(ContractError) as error:
        result.to_manifest()
    assert error.value.details.code == (
        "repeated_crossfit_diagnostics_integrity_violation"
    )


def test_repeated_crossfit_keeps_rare_receiver_opportunities_across_partitions() -> (
    None
):
    adata = _adata()
    rare_rows = adata.obs["subject_id"].eq("p1") & adata.obs["cell_type"].eq("Sender")
    adata.obs.loc[rare_rows, "cell_type"] = "Novel"
    repeat_spec = RepeatedCrossFitSpec(
        crossfit_spec=replace(
            _spec(),
            predeclared_receiver_ids=("Ghost", "Novel", "Receiver", "Sender"),
        ),
        n_repeats=2,
    )

    result = run_repeated_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=repeat_spec,
    )

    assert len({repeat.receiver_universe.universe_id for repeat in result.repeats}) == 1
    assert all(
        repeat.receiver_universe.receiver_ids
        == ("Ghost", "Novel", "Receiver", "Sender")
        for repeat in result.repeats
    )
    for repeat in result.repeats:
        novel_support = tuple(
            support
            for fold in repeat.folds
            for support in fold.receiver_training_support
            if support.receiver_id == "Novel"
        )
        assert len(novel_support) == len(repeat.folds)
        assert any(
            support.reason_code == "receiver_absent_in_outer_training"
            for support in novel_support
        )
    registry = result.repeat_registry
    assert registry["receiver_universe_id"].nunique() == 1
    assert registry["receiver_axis_id"].nunique() == 1
    assert registry["n_receiver_fold_not_estimable"].gt(0).all()
    assert (
        registry["n_receiver_fold_opportunities"]
        == registry["n_receivers"] * registry["n_folds"]
    ).all()
    root_families = result.receiver_family_opportunity_universe.family_ids
    ghost_events = result.family_fold_events.loc[
        result.family_fold_events["receiver"].eq("Ghost")
    ]
    ghost_subjects = result.subject_family_repeat_values.loc[
        result.subject_family_repeat_values["receiver"].eq("Ghost")
    ]
    ghost_stability = result.family_repeat_stability.loc[
        result.family_repeat_stability["receiver"].eq("Ghost")
    ]
    ghost_subject_points = result.subject_family_point_estimates.loc[
        result.subject_family_point_estimates["receiver"].eq("Ghost")
    ]
    ghost_family_points = result.family_point_estimates.loc[
        result.family_point_estimates["receiver"].eq("Ghost")
    ]
    n_folds = len(result.repeats[0].fold_plan.folds)
    n_subjects = len(result.repeats[0].fold_plan.subject_ids)
    n_contrasts = len(repeat_spec.crossfit_spec.contrasts)
    assert len(ghost_events) == 2 * n_folds * n_contrasts * len(root_families)
    assert len(ghost_subjects) == 2 * n_subjects * n_contrasts * len(root_families)
    assert len(ghost_stability) == n_contrasts * len(root_families)
    assert len(ghost_subject_points) == n_subjects * n_contrasts * len(root_families)
    assert len(ghost_family_points) == n_contrasts * len(root_families)
    assert set(ghost_subject_points["status"]) == {"not_estimable"}
    assert set(ghost_family_points["status"]) == {"not_estimable"}
    assert result.diagnostic_status == (
        "not_estimable_family_common_stage_not_connected"
    )
    assert set(ghost_events["family_id"]) == set(root_families)
    assert set(ghost_events["selection_status"]) == {"not_estimable"}
    assert set(ghost_events["reason_code"]) == {"receiver_absent_in_outer_training"}
    assert not ghost_events["family_available"].any()
    assert not ghost_subjects["family_estimable"].any()
    receiver_events = result.family_fold_events.loc[
        result.family_fold_events["receiver"].eq("Receiver")
    ]
    assert set(receiver_events["reason_code"]) == {
        "family_common_functional_not_requested_without_penalty_tuning"
    }


def test_unadjusted_tuned_diagnostic_reaches_noncertified_family_common() -> None:
    base = _spec()
    spec = CrossFitSpec(
        contrasts=base.contrasts,
        training_spec=base.training_spec,
        allowed_n_splits=(2,),
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
        assert application.reason_code == "receiver_autonomous_nuisance_not_frozen"
        assert functional.incremental_functional is model.diagnostic_functional
        assert functional.incremental_reason_code is None
        assert functional.autonomous_program_resource_id is None
        assert not functional.is_oof_certified
        assert common_application.incremental_application_id == (
            application.diagnostic_application.application_id
        )
        assert common_application.heldout_reason_code is None
        assert not common_application.is_oof_certified
        assert len(common_application.family_scores) > 0
    assert not result.is_oof_certified


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
            root_input_identity=result.root_input_identity,
            receiver_universe=result.receiver_universe,
            receiver_family_opportunity_universe=(
                result.receiver_family_opportunity_universe
            ),
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


def test_crossfit_spec_rejects_forced_autonomous_use_scope_mutation() -> None:
    spec = _spec()
    object.__setattr__(spec, "autonomous_program_use_scope", "biological_analysis")

    with pytest.raises(ContractError) as error:
        spec.to_dict()
    assert error.value.details.code == "crossfit_spec_integrity_violation"


def test_crossfit_spec_rejects_forced_nested_sender_policy_mutation() -> None:
    spec = _spec()
    object.__setattr__(spec.training_spec.sender_parameters, "min_subjects", 999)

    with pytest.raises(ContractError) as error:
        spec.to_dict()
    assert error.value.details.code == "crossfit_spec_integrity_violation"


def test_crossfit_manifest_rejects_forced_root_input_digest_mutation() -> None:
    result = _run(_adata())
    object.__setattr__(
        result.root_input_identity,
        "input_digest",
        "poisoned-root-input",
    )

    with pytest.raises(ContractError) as error:
        result.to_manifest()
    assert error.value.details.code == "crossfit_artifact_integrity_violation"


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


def test_orchestrator_passes_only_disjoint_preaggregated_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = crossfit_module._fit_training_artifacts_from_prepared
    original_apply = crossfit_module._apply_training_artifacts_from_prepared
    calls: list[tuple[str, tuple[str, ...]]] = []

    def inspected_fit(prepared, *args: object, **kwargs: object):
        assert not hasattr(prepared, "validated")
        subjects = tuple(
            sorted(prepared.sample_metadata["subject_id"].astype(str).unique())
        )
        aggregate_subjects = tuple(
            sorted(prepared.aggregate.unit_metadata["subject_id"].astype(str).unique())
        )
        assert aggregate_subjects == subjects
        calls.append(("fit", subjects))
        return original_fit(prepared, *args, **kwargs)

    def inspected_apply(artifacts, prepared):
        assert not hasattr(prepared, "validated")
        subjects = tuple(
            sorted(prepared.sample_metadata["subject_id"].astype(str).unique())
        )
        aggregate_subjects = tuple(
            sorted(prepared.aggregate.unit_metadata["subject_id"].astype(str).unique())
        )
        assert aggregate_subjects == subjects
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
            return original_apply(artifacts, prepared)

    monkeypatch.setattr(
        crossfit_module,
        "_fit_training_artifacts_from_prepared",
        inspected_fit,
    )
    monkeypatch.setattr(
        crossfit_module,
        "_apply_training_artifacts_from_prepared",
        inspected_apply,
    )

    result = _run(_adata())

    assert [stage for stage, _ in calls] == ["fit", "apply", "fit", "apply"]
    for fold_result in result.folds:
        assert not set(fold_result.training.training_subject_ids).intersection(
            fold_result.application.heldout_subject_ids
        )


def test_orchestrator_fits_training_availability_once_per_outer_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = training_module._fit_interaction_universe
    calls: list[tuple[str, ...]] = []

    def counted_fit(prepared, *args: object, **kwargs: object):
        calls.append(prepared.subject_ids)
        return original(prepared, *args, **kwargs)

    monkeypatch.setattr(training_module, "_fit_interaction_universe", counted_fit)

    result = _run(_adata())

    assert len(calls) == len(result.folds)
    assert sorted(calls) == sorted(
        fold.training.training_subject_ids for fold in result.folds
    )


def test_family_common_parent_availability_digest_is_fold_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = crossfit_module._table_digest
    calls: list[int] = []

    def counted_digest(table_name: str, table: pd.DataFrame) -> str:
        if table_name == "family_common_source_availability":
            calls.append(len(table))
        return original(table_name, table)

    monkeypatch.setattr(crossfit_module, "_table_digest", counted_digest)

    result = _persistable_run()

    assert all(len(fold.family_common_bindings) > 1 for fold in result.folds)
    assert len(calls) == len(result.folds)


def test_heldout_only_cell_type_is_excluded_from_frozen_training_universe() -> None:
    adata = _adata()
    heldout_only = adata.obs["subject_id"].eq("p1") & adata.obs["cell_type"].eq(
        "Sender"
    )
    adata.obs.loc[heldout_only, "cell_type"] = "Novel"

    result = _run(adata)

    assert result.receiver_universe.receiver_ids == (
        "Novel",
        "Receiver",
        "Sender",
    )
    assert all(
        tuple(record.receiver_id for record in fold.receiver_training_support)
        == result.receiver_universe.receiver_ids
        for fold in result.folds
    )
    excluded_folds = [
        fold
        for fold in result.folds
        if "Novel" in fold.application.excluded_cell_type_ids
    ]
    assert excluded_folds
    assert all("Novel" not in fold.training.cell_type_ids for fold in excluded_folds)
    assert all(
        next(
            record
            for record in fold.receiver_training_support
            if record.receiver_id == "Novel"
        ).reason_code
        == "receiver_absent_in_outer_training"
        for fold in excluded_folds
    )
    assert all(
        all(
            model.receiver_family_artifact.receiver != "Novel"
            for model in fold.receiver_family_models
        )
        for fold in excluded_folds
    )
    excluded_fold_ids = {fold.fold_id for fold in excluded_folds}
    novel_coverage = result.oof_receiver_coverage.loc[
        result.oof_receiver_coverage["fold_id"].isin(excluded_fold_ids)
        & result.oof_receiver_coverage["receiver"].eq("Novel")
    ]
    assert not novel_coverage.empty
    assert set(novel_coverage["receiver_training_support_status"]) == {"not_estimable"}
    assert set(novel_coverage["reason_code"]) == {"receiver_absent_in_outer_training"}
    assert (
        novel_coverage[
            [
                "response_artifact_id",
                "precision_transform_id",
                "incremental_training_artifact_id",
                "response_application_id",
                "incremental_application_id",
            ]
        ]
        .isna()
        .all(axis=None)
    )
    registry = result.receiver_scoring_registry
    assert all(
        collection.planned_receivers == result.receiver_universe.receiver_ids
        for collection in registry.collections
    )
    absent_children = [
        child
        for collection in registry.collections
        if collection.fold_id in excluded_fold_ids
        for child in collection.children
        if child.receiver == "Novel"
    ]
    assert absent_children
    assert all(
        child.receiver_training_support_status == "not_estimable"
        and child.reason_code == "receiver_absent_in_outer_training"
        and child.receiver_family_model_id is None
        and child.receiver_incremental_model_id is None
        for child in absent_children
    )
    absent_failures = [
        requirement
        for requirement in result.oof_certification_audit.failed_requirements
        if requirement.fold_id in excluded_fold_ids and requirement.receiver == "Novel"
    ]
    assert absent_failures
    assert {requirement.reason_code for requirement in absent_failures} == {
        "receiver_absent_in_outer_training"
    }
    assert any(
        "Novel" in fold["heldout_excluded_cell_type_ids"]
        for fold in result.to_manifest()["fold_artifacts"]
    )

    all_novel = _adata()
    all_novel.obs.loc[all_novel.obs["subject_id"].eq("p1"), "cell_type"] = "Novel"
    complete = _run(all_novel)
    p1_rows = complete.oof_coverage.loc[complete.oof_coverage["subject_id"].eq("p1")]
    assert not p1_rows.empty
    assert set(p1_rows["functional_status"]) == {"out_of_fold"}
    assert any(
        "Novel" in fold.application.excluded_cell_type_ids for fold in complete.folds
    )

    planned = _run(_adata())
    disjoint_subjects = planned.fold_plan.folds[0].test_subject_ids
    disjoint = _adata()
    disjoint.obs.loc[
        disjoint.obs["subject_id"].astype(str).isin(disjoint_subjects),
        "cell_type",
    ] = "Novel"
    disjoint_result = _run(disjoint)
    assert set(disjoint_result.oof_coverage["subject_id"]) == {
        "p1",
        "p2",
        "p3",
        "p4",
    }
    target_fold = next(
        fold
        for fold in disjoint_result.folds
        if set(fold.application.heldout_subject_ids) == set(disjoint_subjects)
    )
    assert target_fold.application.excluded_cell_type_ids == ("Novel",)
    assert target_fold.application.availability.sample_interactions.empty


def test_predeclared_absent_receiver_remains_typed_ne_without_models() -> None:
    spec = replace(
        _spec(),
        predeclared_receiver_ids=("Ghost", "Receiver", "Sender"),
    )
    result = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )

    assert result.receiver_universe.receiver_ids == (
        "Ghost",
        "Receiver",
        "Sender",
    )
    assert result.receiver_universe.observed_cell_type_ids == (
        "Receiver",
        "Sender",
    )
    root_family_universe = result.receiver_family_opportunity_universe
    ghost_opportunities = {
        family_id: opportunity_id
        for receiver, family_id, opportunity_id in root_family_universe.opportunity_ids
        if receiver == "Ghost"
    }
    assert set(ghost_opportunities) == set(root_family_universe.family_ids)
    for fold in result.folds:
        support = next(
            record
            for record in fold.receiver_training_support
            if record.receiver_id == "Ghost"
        )
        assert support.status == "not_estimable"
        assert support.reason_code == "receiver_absent_in_outer_training"
        assert all(
            model.receiver_family_artifact.receiver != "Ghost"
            for model in fold.receiver_family_models
        )
        assert all(
            binding.receiver != "Ghost"
            for binding in fold.directional_response_bindings
        )

    ghost_rows = result.oof_receiver_coverage.loc[
        result.oof_receiver_coverage["receiver"].eq("Ghost")
    ]
    assert len(ghost_rows) == len(
        result.oof_receiver_coverage.loc[
            result.oof_receiver_coverage["receiver"].eq("Receiver")
        ]
    )
    assert set(ghost_rows["official_incremental_status"]) == {"not_estimable"}
    assert set(ghost_rows["reason_code"]) == {"receiver_absent_in_outer_training"}


def test_predeclared_absent_receiver_does_not_perturb_observed_receiver_chain() -> None:
    base_spec = replace(_spec(), outer_fold_partition_seed=20260716)
    baseline = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=base_spec,
    )
    extended = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=replace(
            base_spec,
            predeclared_receiver_ids=("Ghost", "Receiver", "Sender"),
        ),
    )

    assert extended.root_input_identity == baseline.root_input_identity
    assert extended.fold_plan.partition_seed_lineage == (
        baseline.fold_plan.partition_seed_lineage
    )
    original_by_test_subjects = {
        fold.application.heldout_subject_ids: fold for fold in baseline.folds
    }
    augmented_by_test_subjects = {
        fold.application.heldout_subject_ids: fold for fold in extended.folds
    }
    assert augmented_by_test_subjects.keys() == original_by_test_subjects.keys()
    assert extended.receiver_universe.receiver_axis_id != (
        baseline.receiver_universe.receiver_axis_id
    )
    model_lineage = {
        "diagnostic_functional_id",
        "fold_id",
        "precision_transform_id",
        "receiver_family_training_artifact_id",
        "response_artifact_id",
        "training_artifact_id",
    }
    application_lineage = {
        "application_id",
        "diagnostic_application_id",
        "diagnostic_functional_id",
        "response_application_id",
        "training_artifact_id",
    }
    program_lineage = {
        "application_id",
        "downstream_application_id",
        "training_artifact_id",
    }
    for test_subjects, original in original_by_test_subjects.items():
        augmented = augmented_by_test_subjects[test_subjects]
        assert augmented.training.training_artifact_id == (
            original.training.training_artifact_id
        )
        assert augmented.application.application_id == (
            original.application.application_id
        )
        for before, after in zip(
            original.receiver_incremental_models,
            augmented.receiver_incremental_models,
            strict=True,
        ):
            assert {
                key: value
                for key, value in before.to_dict().items()
                if key not in model_lineage
            } == {
                key: value
                for key, value in after.to_dict().items()
                if key not in model_lineage
            }
        for before, after in zip(
            original.receiver_incremental_applications,
            augmented.receiver_incremental_applications,
            strict=True,
        ):
            assert {
                key: value
                for key, value in before.to_dict().items()
                if key not in application_lineage
            } == {
                key: value
                for key, value in after.to_dict().items()
                if key not in application_lineage
            }
        for before, after in zip(
            original.receiver_program_applications,
            augmented.receiver_program_applications,
            strict=True,
        ):
            assert {
                key: value
                for key, value in before.to_dict().items()
                if key not in program_lineage
            } == {
                key: value
                for key, value in after.to_dict().items()
                if key not in program_lineage
            }


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
