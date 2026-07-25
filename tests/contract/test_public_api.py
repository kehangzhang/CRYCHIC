import inspect
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import crychic
import crychic.api as public_api
import crychic.api.facade as facade_module
import crychic.inference as inference
import crychic.response as response
import crychic.results as results
import crychic.workflow as workflow
from crychic.core import FeatureUnavailableError
from crychic.core._validation import record_validation, validation_is_cached
from crychic.design import balanced_contrast


def _adata() -> AnnData:
    adata = AnnData(
        X=np.asarray([[1]], dtype=np.int64),
        obs=pd.DataFrame(
            {
                "sample_id": ["s1"],
                "subject_id": ["p1"],
                "condition": ["control"],
                "cell_type": ["A"],
            },
            index=["cell-1"],
        ),
        var=pd.DataFrame(index=["G"]),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def test_public_api_has_reviewed_v0_1_symbols() -> None:
    assert set(crychic.__all__) == {
        "ActiveNullSpec",
        "ActiveProbabilityPipelineLineage",
        "ActiveProbabilityPipelineResult",
        "ActiveProbabilityResult",
        "ActiveProbabilitySpec",
        "AnalysisProfile",
        "AttributionSupportMethod",
        "BaselineArtifacts",
        "BaselineDryRunPlan",
        "CalibrationCampaignKind",
        "CalibrationGeneratorManifest",
        "CalibrationGeneratorProfile",
        "CalibrationReplayAttestation",
        "CalibrationReplayRegistry",
        "CalibrationReplayRequest",
        "CommunicationHypergraph",
        "CommunicationMode",
        "ContextGraph",
        "ContrastSpec",
        "CrossFitArtifacts",
        "CrossFitLayeredSignatureExport",
        "CrossFitOOFCertificationAudit",
        "CrossFitOOFRequirementRecord",
        "CrossFitResult",
        "CrossFitSpec",
        "CrossFitV7EstimatorResult",
        "Crychic",
        "CrychicConfig",
        "CrychicResult",
        "DirectionalIntegratedLRCollection",
        "DirectionalIntegratedLRResult",
        "DirectionalTargetProgramEffect",
        "DirectionalTargetProgramScoreCollection",
        "DirectionalTargetProgramScoreSpec",
        "ExpressionTransform",
        "FoldTrainingSpec",
        "FullPipelineResamplingResult",
        "GraphFusedCrossFitRegistry",
        "GraphFusedFamilyEffectCollection",
        "GraphFusedIntegratedLRCollection",
        "GraphFusedIntegratedLRSpec",
        "GraphFusedWorkflowSpec",
        "FrozenDirectionalTargetProgramUniverse",
        "FrozenLatentNuisanceSpec",
        "G3CalibrationMetric",
        "G3CalibrationScenarioResult",
        "G3CalibrationScenarioStatus",
        "G3FrequencyCalibrationCampaign",
        "G3FrequencyCalibrationEvidence",
        "G3FrequencyCalibrationGate",
        "G3FrequencyCalibrationProtocol",
        "G3FrequencyCalibrationReplicate",
        "G3FrequencyDependenceStructure",
        "G3FrequencyGateStatus",
        "G3FrequencyPermutationDiagnostic",
        "G3FrequencyReplicateStatus",
        "G3FCalibrationResult",
        "G3PCalibrationCampaign",
        "G3PCalibrationEvidence",
        "G3PCalibrationGate",
        "G3PCalibrationProtocol",
        "G3PCalibrationResult",
        "G3PCalibrationScenarioResult",
        "G3PCalibrationScenarioStatus",
        "G3PCalibrationStratumResult",
        "G3PCandidateStatus",
        "G3PDependenceStructure",
        "G3PGateStatus",
        "G3PReplicatePredictions",
        "G3PReplicateStatus",
        "G3PSimulationTruth",
        "InputSchema",
        "OOFContextEffectResult",
        "OOFEffectFormalEligibility",
        "OOFEffectFormalEligibilityStatus",
        "OOFEffectSpec",
        "RepeatedMeasuresCR2FeatureEffect",
        "RepeatedMeasuresCR2ReceiverEffect",
        "RepeatedMeasuresCR2Status",
        "RepeatedCrossFitDiagnostics",
        "RepeatedCrossFitSpec",
        "ResourceBundle",
        "SemanticScoreCollection",
        "TargetPrior",
        "V7EstimatorSpec",
        "V7CompactEstimatorSnapshot",
        "V7FullPipelineCalibrationGate",
        "V7FullPipelineInferenceResult",
        "V7FullPipelineInferenceSpec",
        "V7FullPipelineResamplingResult",
        "__version__",
        "assess_oof_effect_formal_eligibility",
        "build_crossfit_communication_hypergraph",
        "build_crossfit_semantic_scores",
        "build_directional_integrated_lr_scores",
        "build_g3_frequency_calibration_gate",
        "build_g3p_calibration_gate",
        "build_g3p_replicate_from_pipeline",
        "build_graph_fused_integrated_lr_scores",
        "derive_graph_fused_family_effects",
        "audit_crossfit_oof_readiness",
        "export_crossfit_layered_signatures",
        "evaluate_v7_full_pipeline_calibration",
        "fit_crossfit_family_effect",
        "fit_crossfit_v7_estimator",
        "fit_oof_context_effect",
        "freeze_g3p_simulation_truth",
        "freeze_crossfit_directional_target_program_universe",
        "freeze_directional_target_program_universe",
        "fit_repeated_measures_cr2_receiver_effect",
        "input_schema_from_config",
        "finalize_v7_full_pipeline_inference",
        "recommended_crossfit_spec",
        "load_cellchat_resource",
        "load_cellphonedb_resource",
        "load_nichenet_target_prior",
        "run_full_pipeline_resampling",
        "run_graph_fused_crossfit",
        "run_active_probability_pipeline",
        "run_repeated_subject_crossfit",
        "run_subject_crossfit",
        "run_v7_full_pipeline_resampling",
        "score_crossfit_directional_target_programs",
        "summarize_g3_frequency_calibration_campaign",
        "summarize_attested_g3_frequency_calibration_campaign",
        "summarize_attested_g3p_calibration_campaign",
        "summarize_g3p_calibration_campaign",
        "validate_anndata",
        "write_crossfit_result",
        "write_directional_integrated_lr_result",
        "write_active_probability_result",
        "write_g3f_calibration_result",
        "write_g3p_calibration_result",
    }
    assert (
        crychic.AttributionSupportMethod.RELATIVE_COEFFICIENT_V1.value
        == "gated_prior_attribution_v1"
    )
    assert crychic.AnalysisProfile is public_api.AnalysisProfile
    assert crychic.AnalysisProfile.CROSSFIT_DESCRIPTIVE_V1.value == (
        "crossfit_descriptive_v1"
    )
    assert crychic.AnalysisProfile.LEGACY_V01.value == "legacy_v01"
    assert (
        crychic.recommended_crossfit_spec
        is workflow.recommended_crossfit_spec
        is public_api.recommended_crossfit_spec
    )
    assert (
        crychic.run_v7_full_pipeline_resampling
        is workflow.run_v7_full_pipeline_resampling
        is public_api.run_v7_full_pipeline_resampling
    )
    assert (
        crychic.finalize_v7_full_pipeline_inference
        is workflow.finalize_v7_full_pipeline_inference
        is public_api.finalize_v7_full_pipeline_inference
    )
    assert (
        crychic.AttributionSupportMethod.GATED_RESPONSE_NORM_V2.value
        == "gated_response_norm_attribution_support_v2"
    )
    assert (
        crychic.AttributionSupportMethod.EXPLAINED_SHARE_V3.value
        == "explained_share_attribution_support_v3"
    )
    assert crychic.FrozenLatentNuisanceSpec is public_api.FrozenLatentNuisanceSpec
    assert (
        crychic.DirectionalTargetProgramScoreCollection
        is workflow.DirectionalTargetProgramScoreCollection
        is public_api.DirectionalTargetProgramScoreCollection
    )
    assert (
        crychic.DirectionalIntegratedLRCollection
        is workflow.DirectionalIntegratedLRCollection
        is public_api.DirectionalIntegratedLRCollection
    )
    assert (
        crychic.DirectionalIntegratedLRResult
        is workflow.DirectionalIntegratedLRResult
        is public_api.DirectionalIntegratedLRResult
    )
    assert (
        crychic.write_directional_integrated_lr_result
        is workflow.write_directional_integrated_lr_result
        is public_api.write_directional_integrated_lr_result
    )
    assert (
        crychic.build_directional_integrated_lr_scores
        is workflow.build_directional_integrated_lr_scores
        is public_api.build_directional_integrated_lr_scores
    )
    assert (
        crychic.SemanticScoreCollection
        is workflow.SemanticScoreCollection
        is public_api.SemanticScoreCollection
    )
    assert (
        crychic.build_crossfit_semantic_scores
        is workflow.build_crossfit_semantic_scores
        is public_api.build_crossfit_semantic_scores
    )
    assert (
        crychic.score_crossfit_directional_target_programs
        is workflow.score_crossfit_directional_target_programs
        is public_api.score_crossfit_directional_target_programs
    )
    assert (
        crychic.GraphFusedCrossFitRegistry
        is workflow.GraphFusedCrossFitRegistry
        is public_api.GraphFusedCrossFitRegistry
    )
    assert (
        crychic.GraphFusedFamilyEffectCollection
        is workflow.GraphFusedFamilyEffectCollection
        is public_api.GraphFusedFamilyEffectCollection
    )
    assert (
        crychic.derive_graph_fused_family_effects
        is workflow.derive_graph_fused_family_effects
        is public_api.derive_graph_fused_family_effects
    )
    assert (
        crychic.GraphFusedWorkflowSpec
        is workflow.GraphFusedWorkflowSpec
        is public_api.GraphFusedWorkflowSpec
    )
    assert (
        crychic.run_graph_fused_crossfit
        is workflow.run_graph_fused_crossfit
        is public_api.run_graph_fused_crossfit
    )
    assert isinstance(crychic.__version__, str)
    assert not hasattr(crychic, "TopoCCC")


def test_postfit_export_symbols_and_facade_signatures_are_stable() -> None:
    assert (
        crychic.RepeatedMeasuresCR2ReceiverEffect
        is response.RepeatedMeasuresCR2ReceiverEffect
        is public_api.RepeatedMeasuresCR2ReceiverEffect
    )
    assert (
        crychic.fit_repeated_measures_cr2_receiver_effect
        is response.fit_repeated_measures_cr2_receiver_effect
        is public_api.fit_repeated_measures_cr2_receiver_effect
    )
    assert (
        crychic.OOFEffectFormalEligibility
        is inference.OOFEffectFormalEligibility
        is public_api.OOFEffectFormalEligibility
    )
    assert (
        crychic.assess_oof_effect_formal_eligibility
        is inference.assess_oof_effect_formal_eligibility
        is public_api.assess_oof_effect_formal_eligibility
    )
    assert (
        crychic.fit_crossfit_family_effect
        is workflow.fit_crossfit_family_effect
        is public_api.fit_crossfit_family_effect
    )
    assert (
        crychic.G3FrequencyCalibrationProtocol
        is inference.G3FrequencyCalibrationProtocol
        is public_api.G3FrequencyCalibrationProtocol
    )
    assert (
        crychic.G3FrequencyCalibrationReplicate
        is inference.G3FrequencyCalibrationReplicate
        is public_api.G3FrequencyCalibrationReplicate
    )
    assert (
        crychic.summarize_g3_frequency_calibration_campaign
        is inference.summarize_g3_frequency_calibration_campaign
        is public_api.summarize_g3_frequency_calibration_campaign
    )
    assert (
        crychic.summarize_attested_g3_frequency_calibration_campaign
        is inference.summarize_attested_g3_frequency_calibration_campaign
        is public_api.summarize_attested_g3_frequency_calibration_campaign
    )
    assert (
        crychic.G3PCalibrationCampaign
        is inference.G3PCalibrationCampaign
        is public_api.G3PCalibrationCampaign
    )
    assert (
        crychic.G3PCalibrationProtocol
        is inference.G3PCalibrationProtocol
        is public_api.G3PCalibrationProtocol
    )
    assert (
        crychic.G3PReplicatePredictions
        is inference.G3PReplicatePredictions
        is public_api.G3PReplicatePredictions
    )
    assert (
        crychic.G3PCalibrationEvidence
        is inference.G3PCalibrationEvidence
        is public_api.G3PCalibrationEvidence
    )
    assert (
        crychic.G3PCalibrationScenarioResult
        is inference.G3PCalibrationScenarioResult
        is public_api.G3PCalibrationScenarioResult
    )
    assert (
        crychic.build_g3p_calibration_gate
        is inference.build_g3p_calibration_gate
        is public_api.build_g3p_calibration_gate
    )
    assert (
        crychic.summarize_g3p_calibration_campaign
        is inference.summarize_g3p_calibration_campaign
        is public_api.summarize_g3p_calibration_campaign
    )
    assert (
        crychic.CalibrationGeneratorManifest
        is inference.CalibrationGeneratorManifest
        is public_api.CalibrationGeneratorManifest
    )
    assert (
        crychic.summarize_attested_g3p_calibration_campaign
        is inference.summarize_attested_g3p_calibration_campaign
        is public_api.summarize_attested_g3p_calibration_campaign
    )
    assert (
        crychic.ActiveProbabilityPipelineResult
        is workflow.ActiveProbabilityPipelineResult
        is public_api.ActiveProbabilityPipelineResult
    )
    assert (
        crychic.ActiveProbabilityResult
        is results.ActiveProbabilityResult
        is public_api.ActiveProbabilityResult
    )
    assert (
        crychic.run_active_probability_pipeline
        is workflow.run_active_probability_pipeline
        is public_api.run_active_probability_pipeline
    )
    assert (
        crychic.write_active_probability_result
        is results.write_active_probability_result
        is public_api.write_active_probability_result
    )
    assert (
        crychic.G3FCalibrationResult
        is results.G3FCalibrationResult
        is public_api.G3FCalibrationResult
    )
    assert (
        crychic.write_g3f_calibration_result
        is results.write_g3f_calibration_result
        is public_api.write_g3f_calibration_result
    )
    assert (
        crychic.G3PCalibrationResult
        is results.G3PCalibrationResult
        is public_api.G3PCalibrationResult
    )
    assert (
        crychic.write_g3p_calibration_result
        is results.write_g3p_calibration_result
        is public_api.write_g3p_calibration_result
    )
    assert {
        "ACTIVE_PROBABILITY_ARTIFACT_KIND",
        "ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE",
        "ACTIVE_PROBABILITY_NULL_SOURCE_TABLE",
        "ACTIVE_PROBABILITY_RESULT_SCHEMA_VERSION",
        "ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE",
        "ACTIVE_PROBABILITY_TABLE",
        "ActiveProbabilityResult",
        "G3FCalibrationResult",
        "G3F_CALIBRATION_HYPOTHESIS_TABLE",
        "G3F_CALIBRATION_PERMUTATION_TABLE",
        "G3F_CALIBRATION_REPLICATE_TABLE",
        "G3F_CALIBRATION_RESULT_SCHEMA_VERSION",
        "G3F_CALIBRATION_SCENARIO_TABLE",
        "G3PCalibrationResult",
        "G3P_CALIBRATION_CANDIDATE_TABLE",
        "G3P_CALIBRATION_REPLICATE_TABLE",
        "G3P_CALIBRATION_RESULT_SCHEMA_VERSION",
        "G3P_CALIBRATION_SCENARIO_TABLE",
        "write_active_probability_result",
        "write_g3f_calibration_result",
        "write_g3p_calibration_result",
    }.issubset(results.__all__)
    assert crychic.CommunicationHypergraph is workflow.CommunicationHypergraph
    assert (
        crychic.CrossFitLayeredSignatureExport
        is workflow.CrossFitLayeredSignatureExport
    )
    assert (
        crychic.build_crossfit_communication_hypergraph
        is workflow.build_crossfit_communication_hypergraph
        is public_api.build_crossfit_communication_hypergraph
    )
    assert (
        crychic.export_crossfit_layered_signatures
        is workflow.export_crossfit_layered_signatures
        is public_api.export_crossfit_layered_signatures
    )
    assert tuple(
        inspect.signature(crychic.Crychic.export_crossfit_signatures).parameters
    ) == ("self", "artifacts", "mode", "entropy_threshold")
    assert tuple(
        inspect.signature(crychic.Crychic.export_crossfit_hypergraph).parameters
    ) == ("self", "artifacts", "resource_bundle")
    assert tuple(
        inspect.signature(crychic.Crychic.summarize_bootstrap_support).parameters
    ) == (
        "self",
        "point_artifacts",
        "resampling",
        "universe",
        "distribution_specs",
    )
    assert tuple(
        inspect.signature(crychic.Crychic.fit_active_probability).parameters
    ) == (
        "self",
        "adata",
        "spec",
        "contrast_id_or_name",
        "n_plans",
        "n_jobs",
        "resource_bundle",
        "target_prior",
        "output_dir",
        "universe_name",
        "active_null_spec",
        "active_probability_spec",
        "calibration_gate",
        "calibration_result",
        "calibration_replay_registry",
        "retain_children",
    )
    assert tuple(inspect.signature(crychic.Crychic.analyze).parameters) == (
        "self",
        "adata",
        "profile",
        "spec",
        "resource_bundle",
        "target_prior",
        "output_dir",
    )
    assert tuple(inspect.signature(crychic.Crychic.fit_descriptive).parameters) == (
        "self",
        "adata",
        "spec",
        "n_jobs",
        "resource_bundle",
        "target_prior",
        "output_dir",
    )
    assert tuple(inspect.signature(crychic.Crychic.fit_crossfit).parameters) == (
        "self",
        "adata",
        "spec",
        "n_jobs",
        "resource_bundle",
        "target_prior",
        "output_dir",
    )
    assert (
        inspect.signature(crychic.Crychic.analyze).parameters["profile"].default
        is crychic.AnalysisProfile.CROSSFIT_DESCRIPTIVE_V1
    )
    assert (
        "release"
        not in inspect.signature(crychic.Crychic.fit_active_probability).parameters
    )
    assert tuple(inspect.signature(workflow.run_active_null_reruns).parameters) == (
        "adata",
        "config",
        "resource_bundle",
        "target_prior",
        "point_artifacts",
        "universe",
        "n_plans",
        "n_jobs",
        "active_null_spec",
        "retain_children",
    )
    assert tuple(
        inspect.signature(crychic.run_active_probability_pipeline).parameters
    ) == (
        "adata",
        "config",
        "resource_bundle",
        "target_prior",
        "crossfit_spec",
        "contrast_id_or_name",
        "n_plans",
        "n_jobs",
        "universe_name",
        "active_null_spec",
        "active_probability_spec",
        "calibration_gate",
        "retain_children",
    )

    model = crychic.Crychic(crychic.CrychicConfig(context_keys=["condition"]))
    invalid: Any = object()
    with pytest.raises(TypeError, match="CrossFitArtifacts"):
        model.export_crossfit_signatures(invalid)
    with pytest.raises(TypeError, match="CrossFitArtifacts"):
        model.export_crossfit_hypergraph(invalid)
    with pytest.raises(TypeError, match="CrossFitArtifacts"):
        model.summarize_bootstrap_support(
            invalid,
            invalid,
            universe=invalid,
            distribution_specs=(),
        )


def test_v0_1_facade_preserves_config_and_validates_input() -> None:
    config = crychic.CrychicConfig(context_keys=["condition"])
    model = crychic.Crychic(config)

    assert model.config is config
    assert isinstance(model.input_schema, crychic.InputSchema)
    assert model.validate(_adata()).report.n_obs == 1
    with pytest.raises(TypeError, match="AnnData"):
        model.validate(object())


def test_v0_1_fit_requires_a_versioned_resource() -> None:
    model = crychic.Crychic(crychic.CrychicConfig(context_keys=["condition"]))

    with pytest.raises(FeatureUnavailableError) as fit_error:
        model.fit(_adata())
    assert fit_error.value.details.code == "resource_bundle_missing"


def test_crossfit_facade_requires_typed_spec_and_versioned_resource() -> None:
    model = crychic.Crychic(crychic.CrychicConfig(context_keys=["condition"]))
    spec = crychic.CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("treated",),
                ("control",),
                name="treated_vs_control",
            ),
        ),
    )

    invalid: Any = object()
    with pytest.raises(TypeError, match="CrossFitSpec"):
        model.fit_crossfit(_adata(), spec=invalid)
    with pytest.raises(FeatureUnavailableError) as fit_error:
        model.fit_crossfit(_adata(), spec=spec)
    assert fit_error.value.details.code == "resource_bundle_missing"
    with pytest.raises(TypeError, match="CrossFitSpec"):
        model.fit_active_probability(
            _adata(),
            spec=invalid,
            contrast_id_or_name="treated_vs_control",
            n_plans=1,
        )
    with pytest.raises(FeatureUnavailableError) as probability_error:
        model.fit_active_probability(
            _adata(),
            spec=spec,
            contrast_id_or_name="treated_vs_control",
            n_plans=1,
        )
    assert probability_error.value.details.code == "resource_bundle_missing"


def test_crossfit_facade_reuses_validation_scope_for_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = crychic.Crychic(crychic.CrychicConfig(context_keys=["condition"]))
    object.__setattr__(model, "_resource_bundle", object())
    object.__setattr__(model, "_target_prior", object())
    spec = crychic.CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("treated",),
                ("control",),
                name="treated_vs_control",
            ),
        ),
    )
    artifact = object()
    persisted = object()

    def fake_run(*args: object, **kwargs: object) -> object:
        record_validation(artifact)
        return artifact

    def fake_write(observed: object, output_dir: object) -> object:
        assert observed is artifact
        assert validation_is_cached(observed)
        return persisted

    monkeypatch.setattr(facade_module, "run_subject_crossfit", fake_run)
    monkeypatch.setattr(facade_module, "write_crossfit_result", fake_write)

    result = model.fit_crossfit(_adata(), spec=spec, output_dir="result")

    assert result is persisted
    assert not validation_is_cached(artifact)


def test_repeated_and_resampling_facades_require_typed_specs() -> None:
    model = crychic.Crychic(crychic.CrychicConfig(context_keys=["condition"]))
    crossfit_spec = crychic.CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("treated",),
                ("control",),
                name="treated_vs_control",
            ),
        ),
    )
    repeated_spec = crychic.RepeatedCrossFitSpec(
        crossfit_spec=crossfit_spec,
        n_repeats=2,
    )
    invalid: Any = object()

    with pytest.raises(TypeError, match="RepeatedCrossFitSpec"):
        model.fit_repeated_crossfit(_adata(), spec=invalid)
    with pytest.raises(TypeError, match="CrossFitSpec"):
        model.resample_crossfit(_adata(), spec=invalid, n_bootstraps=1)
    with pytest.raises(FeatureUnavailableError) as repeated_error:
        model.fit_repeated_crossfit(_adata(), spec=repeated_spec)
    assert repeated_error.value.details.code == "resource_bundle_missing"
    with pytest.raises(FeatureUnavailableError) as resampling_error:
        model.resample_crossfit(_adata(), spec=crossfit_spec, n_bootstraps=1)
    assert resampling_error.value.details.code == "resource_bundle_missing"


def test_facade_requires_typed_configuration() -> None:
    invalid: Any = {}
    with pytest.raises(TypeError, match="CrychicConfig"):
        crychic.Crychic(invalid)
