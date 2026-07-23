from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import crychic
from crychic.attribution import AttributionSupportMethod
from crychic.core import CrychicConfig
from crychic.design import ContextGraph
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.response import ResponseMethod
from crychic.sender import SenderEvidenceParameters
from crychic.workflow import (
    BaselineArtifacts,
    BaselineMode,
    PlanStatus,
    RunStatus,
    dry_run_baseline,
    fit_baseline,
)


def _adata() -> AnnData:
    genes = ["L", "E", "R", "T", "AUTO", "H"]
    settings = (
        ("c1", "p1", "control", 1, 1),
        ("c2", "p2", "control", 2, 2),
        ("t1", "p3", "treated", 10, 15),
        ("t2", "p4", "treated", 12, 18),
    )
    rows: list[list[int]] = []
    observations: list[dict[str, str]] = []
    for sample, subject, condition, target, autonomous in settings:
        for cell_type in ("Sender", "Receiver"):
            for replicate in range(2):
                if cell_type == "Sender":
                    values = [20 + replicate, 10 + replicate, 0, 1, 1, 30]
                else:
                    values = [
                        0,
                        0,
                        20 + replicate,
                        target + replicate,
                        autonomous + replicate,
                        30,
                    ]
                rows.append(values)
                observations.append(
                    {
                        "sample_id": sample,
                        "subject_id": subject,
                        "condition": condition,
                        "cell_type": cell_type,
                    }
                )
    obs = pd.DataFrame(
        observations,
        index=[f"cell-{index}" for index in range(len(observations))],
    )
    adata = AnnData(
        X=sparse.csr_matrix(np.asarray(rows, dtype=np.int64)),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def _interaction(interaction_id: str, ligand: str) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=interaction_id,
        ligand_name=ligand,
        receptor_name="R",
        ligand_subunits=(ligand,),
        receptor_subunits=("R",),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="synthetic",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        evidence=("synthetic",),
    )


def _bundle() -> ResourceBundle:
    return ResourceBundle(
        resource_id="synthetic_lr",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(
            _interaction("i_signal", "L"),
            _interaction("i_exogenous", "E"),
        ),
        mapping_report=MappingReport(
            source_rows=2,
            loaded_rows=2,
            mapped_entities=3,
        ),
        manifest_digest="a" * 64,
        source_files=("synthetic.csv",),
        license="CC0",
        citation="Synthetic integration fixture",
    )


def _prior() -> TargetPrior:
    return TargetPrior(
        resource_id="synthetic_prior",
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
        evidence="synthetic target evidence",
        mapping_report=MappingReport(
            source_rows=1,
            loaded_rows=1,
            mapped_entities=2,
        ),
        manifest_digest="b" * 64,
    )


def _config() -> CrychicConfig:
    return CrychicConfig(context_keys=("condition",), random_seed=17)


def _positive_contrast(artifacts: BaselineArtifacts) -> str:
    response = artifacts.response
    return next(
        spec.name
        for spec in response.contrast_specs
        if spec.mode == "global_one_vs_rest" and spec.weights.get("treated") == 1.0
    )


def test_dry_run_reports_support_resources_and_supplied_context_graph() -> None:
    graph = ContextGraph.chain(("control", "treated"))
    plan = dry_run_baseline(
        _adata(),
        _config(),
        resource_bundle=_bundle(),
        target_prior=_prior(),
        context_graph=graph,
        min_cells=2,
    )

    assert plan.can_fit
    assert plan.context_graph is graph
    assert plan.support_table["response_support_ready"].all()
    assert set(plan.support_table["n_eligible_subjects"]) == {2}
    assert set(plan.resource_table["status"]) == {PlanStatus.READY.value}
    assert {stage.name for stage in plan.stages} == {
        "validate",
        "pseudobulk",
        "context_graph",
        "design_audit",
        "gene_response",
        "availability",
        "sender_assignment",
        "attribution",
        "scoring",
    }
    assert plan.design_audit.ready
    assert plan.design_audit.rank == plan.design_audit.n_columns
    assert plan.contrast_table["estimable"].all()
    assert set(plan.contrast_table["contrast"]) == {
        "global:'control'",
        "global:'treated'",
    }
    assert "equivalent_contrast_aliases_omitted" in plan.warnings

    blocked = dry_run_baseline(_adata(), _config(), min_cells=2)
    assert not blocked.can_fit
    lr_row = blocked.resource_table.set_index("resource_kind").loc["lr_bundle"]
    assert lr_row["status"] == PlanStatus.BLOCKED.value
    assert lr_row["reason_code"] == "resource_bundle_missing"


def test_dry_run_and_fit_block_a_rank_deficient_declared_design() -> None:
    adata = _adata()
    adata.obs["batch"] = adata.obs["condition"]
    config = CrychicConfig(
        context_keys=("condition",),
        covariates=("batch",),
        design="~ batch + condition",
        random_seed=17,
    )

    plan = dry_run_baseline(
        adata,
        config,
        resource_bundle=_bundle(),
        min_cells=2,
    )

    assert not plan.can_fit
    assert not plan.design_audit.ready
    assert "design_rank_deficient" in plan.warnings
    assert (
        next(stage for stage in plan.stages if stage.name == "design_audit").status
        is PlanStatus.BLOCKED
    )
    with pytest.raises(ValueError, match="design_rank_deficient"):
        fit_baseline(adata, config, _bundle(), min_cells=2)


def test_dry_run_and_fit_block_mixed_paired_unpaired_subjects() -> None:
    adata = _adata()
    adata.obs.loc[adata.obs["sample_id"] == "t1", "subject_id"] = "p1"

    plan = dry_run_baseline(
        adata,
        _config(),
        resource_bundle=_bundle(),
        min_cells=2,
    )

    assert not plan.can_fit
    assert set(plan.contrast_table["subject_design"]) == {
        "mixed_paired_unpaired_subjects"
    }
    assert not plan.contrast_table["subject_design_estimable"].any()
    assert "mixed_paired_unpaired_design_not_supported" in plan.warnings
    with pytest.raises(ValueError, match="registered_contrast_not_estimable"):
        fit_baseline(adata, _config(), _bundle(), min_cells=2)


def test_fit_baseline_preserves_units_common_functional_and_missing_evidence() -> None:
    artifacts = fit_baseline(
        _adata(),
        _config(),
        _bundle(),
        target_prior=_prior(),
        sender_parameters=SenderEvidenceParameters(min_subjects=2),
        min_cells=2,
        min_pooled_availability=0.0,
    )

    assert artifacts.mode is BaselineMode.EXPLORATORY_STRENGTH
    assert not artifacts.inference_eligible
    assert artifacts.validated_input.report.n_samples == 4
    assert artifacts.validated_input.report.n_subjects == 4
    assert artifacts.aggregate.n_observed_units == 8
    assert not artifacts.sender_assignment.inference_eligible
    assert artifacts.sender_assignment.causal_interpretation == (
        "evidence_based_non_causal"
    )
    assert set(artifacts.sample_scores["sample_id"]) == {"c1", "c2", "t1", "t2"}
    assert set(artifacts.sample_scores["subject_id"]) == {"p1", "p2", "p3", "p4"}
    assert set(artifacts.response.contrasts["method"]) == {
        ResponseMethod.EMM_INDEPENDENT.value
    }
    assert {spec.name for spec in artifacts.response.contrast_specs} == {
        "global:'control'",
        "global:'treated'",
    }

    contrast = _positive_contrast(artifacts)
    attribution = next(
        run
        for run in artifacts.attribution_runs
        if run.receiver == "Receiver" and run.contrast == contrast
    )
    assert attribution.status is RunStatus.OK
    assert dict(attribution.receptor_gates).keys() == {"L"}
    assert dict(attribution.receptor_gates)["L"] > 0
    assert attribution.attribution is not None
    assert attribution.precision_method == "winsorized_median_normalized_v2"
    assert attribution.precision_transform_id is not None
    positive_precision = attribution.attribution.precision_weights
    assert attribution.n_positive_precision_features == int(
        np.count_nonzero(positive_precision)
    )
    assert attribution.n_positive_precision_features >= 2
    assert np.median(positive_precision[positive_precision > 0]) == pytest.approx(1.0)
    autonomous_index = artifacts.response.feature_ids.index("AUTO")
    assert attribution.attribution.residual[autonomous_index] > 0

    receiver_scores = artifacts.sample_scores.loc[
        (artifacts.sample_scores["receiver"] == "Receiver")
        & (artifacts.sample_scores["contrast"] == contrast)
    ]
    assert set(receiver_scores["context"]) == {"control", "treated"}
    assert set(receiver_scores["mode"]) == {"state", "ecosystem"}
    assert receiver_scores["scoring_function_id"].nunique() == 1
    assert set(receiver_scores["score_version"]) == {"geometric_v1_tracked"}
    assert receiver_scores["model_manifest_id"].notna().all()
    assert receiver_scores["model_manifest_id"].nunique() == 1
    assert set(receiver_scores["functional_reason_code"]) == {
        "exploratory_not_cross_fitted"
    }
    assert all(
        run.functional.model_manifest.filter_universe_id
        == artifacts.availability.filter_universe_id
        for run in artifacts.score_runs
    )
    collections = artifacts.scoring_collections
    assert collections
    assert all(
        not collection.common_functional_across_receivers
        for collection in collections
    )
    assert {
        child.scoring_functional_id
        for collection in collections
        for child in collection.children
    } == {
        run.functional.scoring_function_id for run in artifacts.score_runs
    }
    assert sum(
        collection.source_score_row_count for collection in collections
    ) == len(artifacts.sample_scores)
    availability_parameters = artifacts.run_parameters["availability"]
    assert availability_parameters["filter_application"] == "training_selection_v1"
    assert availability_parameters["application_subject_ids"] == (
        "p1",
        "p2",
        "p3",
        "p4",
    )
    frozen_filter = availability_parameters["frozen_interaction_universe"]
    assert frozen_filter["filter_universe_id"] == (
        artifacts.availability.filter_universe_id
    )
    assert frozen_filter["interaction_ids"] == (
        artifacts.availability.frozen_interaction_universe.interaction_ids
    )
    assert frozen_filter["training_subject_ids"] == (
        artifacts.availability.frozen_interaction_universe.training_subject_ids
    )
    assert frozen_filter["selection_policy"] == "training_pooled_support_v1"
    sender_specificity = receiver_scores.loc[
        receiver_scores["mode"] == "state"
    ].groupby(["sample_id", "interaction_id"], observed=True)["sender_component"]
    np.testing.assert_allclose(sender_specificity.sum().to_numpy(), 1.0)

    signal = receiver_scores.loc[
        (receiver_scores["sample_id"] == "c1")
        & (receiver_scores["sender"] == "Sender")
        & (receiver_scores["interaction_id"] == "i_signal")
    ].set_index("mode")
    assert signal.loc["state", "availability"] > signal.loc["ecosystem", "availability"]
    assert (
        signal.loc["state", "comm_strength"] > signal.loc["ecosystem", "comm_strength"]
    )

    exogenous = receiver_scores.loc[receiver_scores["interaction_id"] == "i_exogenous"]
    assert set(exogenous["status"]) == {"missing_core_evidence"}
    assert set(exogenous["reason_code"]) == {
        "missing_core_evidence:downstream,prior_quality"
    }
    assert exogenous["comm_strength"].isna().all()

    downstream = artifacts.downstream_activity.loc[
        (artifacts.downstream_activity["receiver"] == "Receiver")
        & (artifacts.downstream_activity["contrast"] == contrast)
        & (artifacts.downstream_activity["interaction_id"] == "i_signal")
    ]
    assert downstream["target_weight_id"].nunique() == 1
    assert downstream["receiver_program_score"].notna().all()
    assert (
        downstream["downstream_activity"]
        <= downstream["receiver_program_score"] + 1e-12
    ).all()
    assert downstream["incremental_downstream"].isna().all()
    assert set(downstream["incremental_downstream_status"]) == {"not_estimable"}
    assert set(downstream["incremental_downstream_reason_code"]) == {
        "cross_fitted_receiver_null_not_implemented"
    }
    assert (
        downstream["prior_quality_source"]
        .str.contains("constant=1.0;resource=synthetic_prior")
        .all()
    )
    evidence = artifacts.edge_evidence
    evidence_key = [
        "contrast",
        "fold_id",
        "sample_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
        "mode",
    ]
    assert not evidence.empty
    assert not evidence.duplicated(evidence_key).any()
    assert len(evidence.loc[evidence["sample_score_status"] == "linked"]) == len(
        artifacts.sample_scores
    )
    signal_evidence = evidence.loc[
        (evidence["contrast"] == contrast)
        & (evidence["sample_id"] == "c1")
        & (evidence["sender"] == "Sender")
        & (evidence["receiver"] == "Receiver")
        & (evidence["interaction_id"] == "i_signal")
    ].set_index("mode")
    assert set(signal_evidence.index) == {"state", "ecosystem"}
    assert set(signal_evidence["state_availability_status"]) == {"observed"}
    assert set(signal_evidence["ecosystem_availability_status"]) == {"observed"}
    assert set(signal_evidence["receptor_gate_status"]) == {"observed"}
    assert signal_evidence["receptor_gate"].notna().all()
    assert set(signal_evidence["receiver_program_status"]) == {"observed"}
    assert signal_evidence["receiver_program_score"].notna().all()
    assert signal_evidence["incremental_downstream"].isna().all()
    assert set(signal_evidence["incremental_downstream_reason_code"]) == {
        "cross_fitted_receiver_null_not_implemented"
    }
    assert set(signal_evidence["attribution_support_method"]) == {
        AttributionSupportMethod.RELATIVE_COEFFICIENT_V1.value
    }
    assert set(signal_evidence["attribution_support_status"]) == {"observed"}
    assert set(signal_evidence["sender_status"]) == {"ok"}
    assert signal_evidence["sender_weight"].notna().all()
    assert set(signal_evidence["prior_quality_status"]) == {"observed"}
    assert set(signal_evidence["legacy_integrated_status"]) == {"ok"}
    assert set(signal_evidence["sample_score_status"]) == {"linked"}
    assert set(signal_evidence["score_version"]) == {"geometric_v1_tracked"}
    assert signal_evidence["model_manifest_id"].nunique() == 1
    assert signal_evidence["scoring_function_id"].nunique() == 1
    tampered = evidence.copy(deep=True)
    linked_strength = (
        tampered["sample_score_status"].eq("linked")
        & tampered["legacy_integrated_strength"].notna()
    )
    tampered_index = tampered.index[linked_strength][0]
    original_strength = float(tampered.at[tampered_index, "legacy_integrated_strength"])
    tampered.at[tampered_index, "legacy_integrated_strength"] = (
        0.0 if original_strength > 0 else 1.0
    )
    with pytest.raises(
        ValueError, match="does not reproduce sample score comm_strength"
    ):
        replace(artifacts, edge_evidence=tampered)
    assert not {
        "p",
        "p_value",
        "q",
        "q_value",
        "posterior_probability",
    }.intersection(artifacts.sample_scores.columns)


def test_fit_baseline_is_deterministic() -> None:
    arguments = {
        "target_prior": _prior(),
        "sender_parameters": SenderEvidenceParameters(min_subjects=2),
        "min_cells": 2,
        "min_pooled_availability": 0.0,
    }
    first = fit_baseline(_adata(), _config(), _bundle(), **arguments)
    second = fit_baseline(_adata(), _config(), _bundle(), **arguments)

    pd.testing.assert_frame_equal(first.sample_scores, second.sample_scores)
    pd.testing.assert_frame_equal(first.edge_evidence, second.edge_evidence)
    pd.testing.assert_frame_equal(
        first.availability.sample_interactions,
        second.availability.sample_interactions,
    )
    pd.testing.assert_frame_equal(
        first.sender_assignment.table,
        second.sender_assignment.table,
    )
    pd.testing.assert_frame_equal(first.response.contrasts, second.response.contrasts)
    assert [
        functional.scoring_function_id for functional in first.scoring_functionals
    ] == [functional.scoring_function_id for functional in second.scoring_functionals]
    for left, right in zip(
        first.attribution_runs, second.attribution_runs, strict=True
    ):
        assert left.status is right.status
        assert left.receptor_gates == right.receptor_gates
        if left.attribution is not None and right.attribution is not None:
            np.testing.assert_array_equal(
                left.attribution.coefficients, right.attribution.coefficients
            )


def test_downstream_support_v2_is_opt_in_and_manifest_ready() -> None:
    arguments = {
        "target_prior": _prior(),
        "sender_parameters": SenderEvidenceParameters(min_subjects=2),
        "min_cells": 2,
        "min_pooled_availability": 0.0,
    }
    default = fit_baseline(_adata(), _config(), _bundle(), **arguments)
    explicit_v1 = fit_baseline(
        _adata(),
        _config(),
        _bundle(),
        downstream_attribution_support_method=(
            AttributionSupportMethod.RELATIVE_COEFFICIENT_V1
        ),
        **arguments,
    )
    candidate_v2 = fit_baseline(
        _adata(),
        _config(),
        _bundle(),
        downstream_attribution_support_method=(
            AttributionSupportMethod.GATED_RESPONSE_NORM_V2
        ),
        **arguments,
    )

    pd.testing.assert_frame_equal(default.sample_scores, explicit_v1.sample_scores)
    assert default.run_parameters == explicit_v1.run_parameters
    assert set(default.downstream_activity["target_weight_method"]) == {
        AttributionSupportMethod.RELATIVE_COEFFICIENT_V1.value
    }
    assert set(candidate_v2.downstream_activity["target_weight_method"]) == {
        AttributionSupportMethod.GATED_RESPONSE_NORM_V2.value
    }
    support = candidate_v2.downstream_activity["attribution_support"].dropna()
    assert support.between(0.0, 1.0).all()
    provenance = candidate_v2.run_parameters["attribution"]["downstream_support"]
    assert provenance["method"] == (
        AttributionSupportMethod.GATED_RESPONSE_NORM_V2.value
    )
    assert provenance["gate_aware"]
    assert provenance["application"] == "multiply_sample_target_activity"
    assert "downstream_support" not in default.run_parameters["attribution"]


def test_unknown_downstream_support_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="not-a-support-method"):
        fit_baseline(
            _adata(),
            _config(),
            _bundle(),
            downstream_attribution_support_method="not-a-support-method",
            min_cells=2,
        )


def test_missing_optional_prior_returns_explicit_availability_baseline() -> None:
    artifacts = fit_baseline(
        _adata(),
        _config(),
        _bundle(),
        min_cells=2,
        min_pooled_availability=0.0,
    )

    assert artifacts.mode is BaselineMode.AVAILABILITY_BASELINE
    assert artifacts.sample_scores.empty
    assert not artifacts.score_runs
    assert "target_prior_not_provided" in artifacts.reason_codes
    assert artifacts.attribution_runs
    assert all(
        run.status in {RunStatus.UNAVAILABLE, RunStatus.NOT_ESTIMABLE}
        for run in artifacts.attribution_runs
    )
    evidence = artifacts.edge_evidence
    assert not evidence.empty
    assert evidence["state_availability"].notna().all()
    assert set(evidence["state_availability_status"]) == {"observed"}
    for column in (
        "receptor_gate",
        "receiver_program_score",
        "incremental_downstream",
        "attribution_support",
        "prior_quality",
        "legacy_integrated_strength",
        "score_version",
        "model_manifest_id",
        "scoring_function_id",
    ):
        assert evidence[column].isna().all()
    assert evidence["receptor_gate_reason_code"].notna().all()
    assert evidence["receiver_program_reason_code"].notna().all()
    assert evidence["legacy_integrated_reason_code"].notna().all()
    assert set(evidence["sample_score_status"]) == {"not_emitted"}


def test_low_sender_support_cannot_become_complete_core_evidence() -> None:
    artifacts = fit_baseline(
        _adata(),
        _config(),
        _bundle(),
        target_prior=_prior(),
        min_cells=2,
        min_pooled_availability=0.0,
    )

    assert set(artifacts.sender_assignment.table["status"]) == {"low_support"}
    assert artifacts.mode is BaselineMode.AVAILABILITY_BASELINE
    assert not artifacts.score_runs
    assert "score_branch_no_complete_core_evidence" in artifacts.reason_codes
    scored_evidence = artifacts.edge_evidence.loc[
        artifacts.edge_evidence["score_version"].notna()
        & (artifacts.edge_evidence["interaction_id"] == "i_signal")
    ]
    assert not scored_evidence.empty
    assert set(scored_evidence["sender_status"]) == {"low_support"}
    assert set(scored_evidence["sender_reason_code"]) == {
        "insufficient_subject_support"
    }
    assert scored_evidence["sender_weight"].notna().all()
    assert scored_evidence["legacy_integrated_strength"].isna().all()
    assert set(scored_evidence["legacy_integrated_reason_code"]) == {
        "missing_core_evidence:sender"
    }
    assert set(scored_evidence["sample_score_status"]) == {"not_emitted"}
    assert set(scored_evidence["sample_score_reason_code"]) == {
        "branch_has_no_complete_core_evidence"
    }


def test_multifactor_context_uses_one_id_across_availability_and_sender() -> None:
    adata = _adata()
    adata.obs["batch"] = adata.obs["subject_id"].map(
        {"p1": "b1", "p2": "b2", "p3": "b1", "p4": "b2"}
    )
    config = CrychicConfig(context_keys=("condition", "batch"), random_seed=17)

    artifacts = fit_baseline(
        adata,
        config,
        _bundle(),
        min_cells=2,
        min_pooled_availability=0.0,
    )

    assert set(artifacts.availability.sample_interactions["context_id"]) == set(
        artifacts.sender_assignment.table["context_id"]
    )


def test_dry_run_blocks_resource_with_no_mapped_interactions() -> None:
    base = _bundle()
    unmapped = ResourceBundle(
        resource_id=base.resource_id,
        version=base.version,
        species=base.species,
        gene_namespace=base.gene_namespace,
        interactions=(_interaction("unmapped", "NOT_A_GENE"),),
        mapping_report=MappingReport(source_rows=1, loaded_rows=1, mapped_entities=2),
        manifest_digest=base.manifest_digest,
        source_files=base.source_files,
        license=base.license,
        citation=base.citation,
    )

    plan = dry_run_baseline(_adata(), _config(), resource_bundle=unmapped, min_cells=2)

    row = plan.resource_table.set_index("resource_kind").loc["lr_bundle"]
    assert not plan.can_fit
    assert row["status"] == PlanStatus.BLOCKED.value
    assert row["reason_code"] == "no_interactions_map_to_input"


def test_duplicate_gene_opt_in_is_validation_only_for_workflow() -> None:
    adata = _adata()
    adata.var_names = ["L", "E", "R", "T", "AUTO", "L"]
    config = CrychicConfig(
        context_keys=("condition",),
        allow_duplicate_genes=True,
        random_seed=17,
    )
    model = crychic.Crychic(config, resource_bundle=_bundle())

    plan = model.dry_run(adata, min_cells=2)

    assert not plan.can_fit
    validate_stage = next(stage for stage in plan.stages if stage.name == "validate")
    assert validate_stage.status is PlanStatus.BLOCKED
    assert "duplicate_genes_fit_unsupported" in plan.warnings
    with pytest.raises(ValueError, match="validation-only"):
        model.fit(adata, min_cells=2, min_pooled_availability=0.0)


def test_public_facade_atomically_persists_queryable_v0_1_result(tmp_path) -> None:
    model = crychic.Crychic(_config(), resource_bundle=_bundle(), target_prior=_prior())
    result = model.fit(
        _adata(),
        output_dir=tmp_path / "result",
        input_digest="c" * 64,
        sender_parameters=SenderEvidenceParameters(min_subjects=2),
        min_cells=2,
        min_pooled_availability=0.0,
    )

    assert isinstance(result, crychic.CrychicResult)
    assert not result.has_edge_evidence
    assert result.has_scoring_collections
    assert "edge_evidence_not_persisted" in result.manifest["warnings"]
    assert result.manifest["mode"] == "exploratory"
    assert result.manifest["workflow_parameters"]["pseudobulk"] == {"min_cells": 2}
    assert len(result.manifest["workflow_digest"]) == 64
    profiling = result.manifest["profiling"]
    assert profiling["clock"] == "time.perf_counter"
    assert profiling["scope"] == "fit_baseline_excludes_result_persistence"
    stage_seconds = profiling["stage_seconds"]
    assert set(stage_seconds) == {
        "validate_design",
        "pseudobulk",
        "response",
        "availability",
        "sender_assignment",
        "attribution",
        "scoring_preparation",
        "score_integration",
        "score_materialization",
        "edge_evidence",
        "finalization",
        "artifact_validation",
        "fit_total",
    }
    assert all(value >= 0.0 for value in stage_seconds.values())
    assert stage_seconds["fit_total"] >= max(
        value for name, value in stage_seconds.items() if name != "fit_total"
    )
    availability_parameters = result.manifest["workflow_parameters"]["availability"]
    assert availability_parameters["filter_application"] == "training_selection_v1"
    assert availability_parameters["application_subject_ids"] == (
        "p1",
        "p2",
        "p3",
        "p4",
    )
    frozen_filter = availability_parameters["frozen_interaction_universe"]
    assert frozen_filter["training_subject_ids"] == ("p1", "p2", "p3", "p4")
    assert frozen_filter["selection_policy"] == "training_pooled_support_v1"
    assert str(frozen_filter["filter_universe_id"]).startswith(
        "availability_filter_universe_"
    )
    interactions = result.read_table("interactions")
    responses = result.read_table("responses")
    sample_scores = result.read_table("sample_scores")
    assert not interactions.empty
    assert not responses.empty
    assert not sample_scores.empty
    collections = result.read_scoring_collections()
    assert collections
    assert all(
        not collection.common_functional_across_receivers
        for collection in collections
    )
    assert sum(
        collection.source_score_row_count for collection in collections
    ) == len(sample_scores)
    assert interactions["comm_probability"].isna().all()
    assert responses["p_value"].isna().all()
    contrast = next(
        value for value in interactions["contrast"].unique() if "treated" in value
    )
    ranked = result.rank_interactions(
        context={"condition": "treated"},
        receiver="Receiver",
        contrast=str(contrast),
    )
    assert not ranked.empty
    assert ranked["comm_strength"].notna().any()


def test_public_facade_persists_opt_in_edge_evidence_extension(tmp_path) -> None:
    model = crychic.Crychic(_config(), resource_bundle=_bundle(), target_prior=_prior())
    result = model.fit(
        _adata(),
        output_dir=tmp_path / "result-with-edge-evidence",
        input_digest="d" * 64,
        sender_parameters=SenderEvidenceParameters(min_subjects=2),
        min_cells=2,
        min_pooled_availability=0.0,
        persist_edge_evidence=True,
    )

    assert isinstance(result, crychic.CrychicResult)
    assert result.has_edge_evidence
    assert result.has_scoring_collections
    assert "edge_evidence_not_persisted" not in result.manifest["warnings"]
    extension = result.manifest["extensions"]["edge_evidence"]
    assert extension["rows"] > 0
    assert extension["linked_tables"]["sample_scores"] == (
        result.manifest["tables"]["sample_scores"]["sha256"]
    )
    linked = result.read_edge_evidence(
        filters={"sample_score_status": "linked"},
        columns=["sample_id", "interaction_id", "sample_score_status"],
    )
    assert len(linked) == result.manifest["tables"]["sample_scores"]["rows"]
    assert set(linked["sample_score_status"]) == {"linked"}


def test_public_facade_persists_opt_in_v2_support_provenance(tmp_path) -> None:
    model = crychic.Crychic(_config(), resource_bundle=_bundle(), target_prior=_prior())
    result = model.fit(
        _adata(),
        output_dir=tmp_path / "candidate-v2",
        input_digest="e" * 64,
        sender_parameters=SenderEvidenceParameters(min_subjects=2),
        min_cells=2,
        min_pooled_availability=0.0,
        downstream_attribution_support_method=(
            AttributionSupportMethod.GATED_RESPONSE_NORM_V2
        ),
    )

    provenance = result.manifest["workflow_parameters"]["attribution"][
        "downstream_support"
    ]
    assert provenance["method"] == (
        AttributionSupportMethod.GATED_RESPONSE_NORM_V2.value
    )
    assert provenance["version"] == 2
    assert provenance["gate_aware"]
    assert provenance["experimental"]


def test_public_facade_persists_explicit_availability_only_components(
    tmp_path,
) -> None:
    model = crychic.Crychic(_config(), resource_bundle=_bundle())
    result = model.fit(
        _adata(),
        output_dir=tmp_path / "availability-result",
        input_digest="d" * 64,
        min_cells=2,
        min_pooled_availability=0.0,
    )

    assert isinstance(result, crychic.CrychicResult)
    interactions = result.read_table("interactions")
    sample_scores = result.read_table("sample_scores")
    assert set(interactions["contrast"]) == {"availability_only"}
    assert interactions["availability"].notna().all()
    assert interactions["comm_strength"].isna().all()
    assert sample_scores["availability"].notna().all()
    assert sample_scores["comm_strength"].isna().all()
    assert set(sample_scores["status"]) == {"missing"}


def test_single_context_manifest_marks_response_not_estimable(tmp_path) -> None:
    adata = _adata()[_adata().obs["condition"] == "control"].copy()
    model = crychic.Crychic(_config(), resource_bundle=_bundle())

    result = model.fit(
        adata,
        output_dir=tmp_path / "single-context",
        input_digest="f" * 64,
        min_cells=2,
        min_pooled_availability=0.0,
    )

    assert isinstance(result, crychic.CrychicResult)
    stages = {stage["name"]: stage for stage in result.manifest["stages"]}
    assert stages["response"] == {
        "name": "response",
        "status": "not_estimable",
        "reason_code": "no_estimable_response_contrast",
    }
    assert result.read_table("responses").empty


def test_empty_pooled_support_marks_availability_not_estimable(tmp_path) -> None:
    model = crychic.Crychic(_config(), resource_bundle=_bundle())

    result = model.fit(
        _adata(),
        output_dir=tmp_path / "empty-availability",
        input_digest="0" * 64,
        min_cells=2,
        min_pooled_availability=1.0,
    )

    assert isinstance(result, crychic.CrychicResult)
    stages = {stage["name"]: stage for stage in result.manifest["stages"]}
    assert stages["availability"] == {
        "name": "availability",
        "status": "not_estimable",
        "reason_code": "no_supported_interactions",
    }
    assert "no_supported_interactions" in result.manifest["warnings"]
    assert result.read_table("interactions").empty


def test_normalized_input_warnings_reach_persisted_manifest(tmp_path) -> None:
    adata = _adata()
    config = CrychicConfig(
        context_keys=("condition",),
        counts_layer=None,
        expression_source="synthetic normalized expression",
        expression_transform="linear_normalized",
        random_seed=17,
    )
    model = crychic.Crychic(config, resource_bundle=_bundle())

    result = model.fit(
        adata,
        output_dir=tmp_path / "normalized-result",
        input_digest="1" * 64,
        min_cells=2,
        min_pooled_availability=0.0,
    )

    assert isinstance(result, crychic.CrychicResult)
    assert "normalized_only" in result.manifest["warnings"]
    assert "normalized_detection_unavailable" in result.manifest["warnings"]


def test_run_identity_changes_with_effective_workflow_and_honors_modes(
    tmp_path,
) -> None:
    config = CrychicConfig(
        context_keys=("condition",),
        communication_modes=("state",),
        random_seed=17,
    )
    model = crychic.Crychic(config, resource_bundle=_bundle(), target_prior=_prior())
    common = {
        "input_digest": "e" * 64,
        "sender_parameters": SenderEvidenceParameters(min_subjects=2),
        "min_cells": 2,
        "min_pooled_availability": 0.0,
        "git_commit": "ccc6795cd",
        "git_dirty": True,
    }
    first = model.fit(
        _adata(), output_dir=tmp_path / "max-one", max_interactions=1, **common
    )
    second = model.fit(
        _adata(), output_dir=tmp_path / "max-two", max_interactions=2, **common
    )

    assert isinstance(first, crychic.CrychicResult)
    assert isinstance(second, crychic.CrychicResult)
    assert first.manifest["run_id"] != second.manifest["run_id"]
    assert first.manifest["workflow_digest"] != second.manifest["workflow_digest"]
    assert first.provenance["git_commit"] == "ccc6795cd"
    assert first.provenance["git_dirty"] is True
    assert set(first.read_table("sample_scores")["mode"]) == {"state"}
    assert set(first.read_table("interactions")["mode"]) == {"state"}
