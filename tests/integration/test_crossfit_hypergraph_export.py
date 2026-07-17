from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import crychic
from crychic.attribution import PenaltyTuningSpec
from crychic.core import ContractError, CrychicConfig
from crychic.design import balanced_contrast
from crychic.network import HyperedgeStatus
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.response import build_receiver_autonomous_program_resource
from crychic.sender import ContrastCommonSenderParameters
from crychic.workflow.crossfit import (
    CrossFitArtifacts,
    CrossFitSpec,
    run_subject_crossfit,
)
from crychic.workflow.hypergraph_export import (
    CROSSFIT_HYPERGRAPH_WEIGHT_SEMANTICS,
    build_crossfit_communication_hypergraph,
)
from crychic.workflow.training import FoldTrainingSpec


def _interaction(
    interaction_id: str,
    ligand: str,
    receptor: str,
    *,
    ligand_subunits: tuple[str, ...] | None = None,
) -> Interaction:
    members = ligand_subunits or (ligand,)
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=interaction_id,
        ligand_name=ligand,
        receptor_name=receptor,
        ligand_subunits=members,
        receptor_subunits=(receptor,),
        ligand_is_complex=len(members) > 1,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="crossfit-hypergraph-fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _bundle(*, complex_ligand: bool = False) -> ResourceBundle:
    first = (
        _interaction(
            "i1",
            "L_COMPLEX",
            "R1",
            ligand_subunits=("L1", "L2"),
        )
        if complex_ligand
        else _interaction("i1", "L1", "R1")
    )
    return ResourceBundle(
        resource_id="crossfit_hypergraph_lr",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(first, _interaction("i2", "L2", "R2")),
        mapping_report=MappingReport(2, 2, 5 if complex_ligand else 4),
        manifest_digest=("e" if complex_ligand else "c") * 64,
        source_files=("crossfit-hypergraph.tsv",),
        license="CC0",
        citation="Synthetic cross-fit hypergraph fixture",
    )


def _prior() -> TargetPrior:
    return TargetPrior(
        resource_id="crossfit_hypergraph_prior",
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


def _adata(*, split_fold_universe: bool = False) -> AnnData:
    genes = ("L1", "R1", "L2", "R2", "T1", "T2")
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    obs_names: list[str] = []
    for subject_index, subject in enumerate(("p1", "p2", "p3", "p4")):
        # The preregistered seed places p1/p3 and p2/p4 in opposite training
        # folds. Keep the split-universe fixture aligned to that frozen partition.
        first_interaction = subject in {"p1", "p3"}
        if split_fold_universe:
            ligand_one = receptor_one = 100 if first_interaction else 1
            ligand_two = receptor_two = 1 if first_interaction else 100
        else:
            ligand_one = receptor_one = 70 + 5 * subject_index
            ligand_two = receptor_two = 2 + subject_index
        for condition in ("control", "stim"):
            for replicate in ("r1", "r2"):
                sample = f"sample-{subject}-{condition}-{replicate}"
                for cell_type, profile in (
                    (
                        "Sender",
                        [ligand_one, 0, ligand_two, 0, 1, 1],
                    ),
                    (
                        "Receiver",
                        [0, receptor_one, 0, receptor_two, 1, 1],
                    ),
                ):
                    for cell_index in range(2):
                        rows.append(profile)
                        metadata.append(
                            {
                                "sample_id": sample,
                                "subject_id": subject,
                                "cell_type": cell_type,
                                "condition": condition,
                            }
                        )
                        obs_names.append(
                            f"{subject}-{condition}-{replicate}-{cell_type}-"
                            f"{cell_index}"
                        )
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


def _spec(*, sender_min_subjects: int = 2) -> CrossFitSpec:
    autonomous = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("T1", "T2"),
        program_ids=("generic_program",),
        resource_id="crossfit-hypergraph-autonomous-programs",
        version="1",
        manifest_digest="9" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
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
            sender_parameters=ContrastCommonSenderParameters(
                min_subjects=sender_min_subjects
            ),
        ),
        allowed_n_splits=(2,),
        autonomous_program_resource=autonomous,
        penalty_tuning_spec=PenaltyTuningSpec(
            lambda1_fractions=(1.0,),
            lambda2_fractions=(0.0,),
            inner_allowed_n_splits=(2,),
        ),
    )


def _run(
    *,
    bundle: ResourceBundle | None = None,
    split_fold_universe: bool = False,
    sender_min_subjects: int = 2,
) -> tuple[CrossFitArtifacts, ResourceBundle]:
    resolved_bundle = bundle or _bundle()
    return (
        run_subject_crossfit(
            _adata(split_fold_universe=split_fold_universe),
            _config(),
            resolved_bundle,
            _prior(),
            spec=_spec(sender_min_subjects=sender_min_subjects),
        ),
        resolved_bundle,
    )


@pytest.fixture(scope="module")
def standard_run() -> tuple[CrossFitArtifacts, ResourceBundle]:
    return _run()


def test_real_crossfit_export_is_program_level_collection_and_order_stable(
    standard_run: tuple[CrossFitArtifacts, ResourceBundle],
) -> None:
    artifacts, bundle = standard_run
    graph = build_crossfit_communication_hypergraph(artifacts, bundle)
    edges = graph.hyperedges

    assert not edges.empty
    assert set(edges["target_kind"]) == {"program"}
    assert set(edges["weight_semantics"]) == {
        CROSSFIT_HYPERGRAPH_WEIGHT_SEMANTICS
    }
    assert edges["fold_id"].isna().all()
    assert edges["uncertainty"].isna().all()
    assert edges["uncertainty_kind"].isna().all()
    raw_functional_ids = {
        functional.family_common_functional_id
        for fold in artifacts.folds
        for functional in fold.family_common_functionals
    }
    assert raw_functional_ids.isdisjoint(edges["scoring_functional_id"])
    for edge in edges.itertuples(index=False):
        provenance = json.loads(edge.provenance_json)
        components = json.loads(edge.components_json)
        assert set(provenance["source_fold_ids"]) == {
            fold.fold_id for fold in artifacts.folds
        }
        assert set(provenance["source_functional_ids"]).issubset(
            raw_functional_ids
        )
        assert {item["sample_count"] for item in provenance["subject_aggregation"]}
        assert all(
            item["sample_count"] == 2
            for item in provenance["subject_aggregation"]
        )
        assert components["source_component"] == "sender_resolved_strength"
        assert components["target"]["semantics"] == (
            "fold_frozen_driver_family_not_gene_signature_v1"
        )

    reordered = CrossFitArtifacts._from_workflow(
        spec=artifacts.spec,
        root_input_identity=artifacts.root_input_identity,
        receiver_universe=artifacts.receiver_universe,
        receiver_family_opportunity_universe=(
            artifacts.receiver_family_opportunity_universe
        ),
        fold_plan=artifacts.fold_plan,
        folds=tuple(reversed(artifacts.folds)),
        oof_coverage=artifacts.oof_coverage.sample(frac=1.0, random_state=7),
        oof_receiver_coverage=artifacts.oof_receiver_coverage.sample(
            frac=1.0, random_state=11
        ),
        oof_sender_assignments=artifacts.oof_sender_assignments.sample(
            frac=1.0, random_state=13
        ),
        coverage_audit=artifacts.coverage_audit,
    )
    repeated = build_crossfit_communication_hypergraph(reordered, bundle)
    pd.testing.assert_frame_equal(graph.hyperedges, repeated.hyperedges)
    pd.testing.assert_frame_equal(graph.nodes, repeated.nodes)
    pd.testing.assert_frame_equal(graph.complex_members, repeated.complex_members)


def test_root_and_facade_postfit_exports_match_direct_results(
    standard_run: tuple[CrossFitArtifacts, ResourceBundle],
) -> None:
    artifacts, bundle = standard_run
    model = crychic.Crychic(
        _config(),
        resource_bundle=bundle,
        target_prior=_prior(),
    )

    direct_signatures = crychic.export_crossfit_layered_signatures(artifacts)
    facade_signatures = model.export_crossfit_signatures(artifacts)
    direct_graph = crychic.build_crossfit_communication_hypergraph(
        artifacts, bundle
    )
    facade_graph = model.export_crossfit_hypergraph(artifacts)

    assert isinstance(facade_signatures, crychic.CrossFitLayeredSignatureExport)
    assert isinstance(facade_graph, crychic.CommunicationHypergraph)
    assert facade_signatures.export_id == direct_signatures.export_id
    pd.testing.assert_frame_equal(facade_graph.hyperedges, direct_graph.hyperedges)
    assert not facade_signatures.inference_eligible
    assert not facade_graph.inference_eligible


def test_crossfold_candidate_union_is_fail_closed() -> None:
    artifacts, bundle = _run(split_fold_universe=True)
    fold_universes = {
        fold.training.frozen_interaction_universe.interaction_ids
        for fold in artifacts.folds
    }
    assert fold_universes == {
        ("i1",),
        ("i2",),
    }

    edges = build_crossfit_communication_hypergraph(artifacts, bundle).hyperedges

    assert set(edges["interaction_id"]) == {"i1", "i2"}
    assert set(edges["status"]) == {HyperedgeStatus.NOT_ESTIMABLE.value}
    assert edges["weight"].isna().all()
    assert set(edges["reason_code"]) == {
        "family_interaction_or_sender_not_present_in_all_folds_v1"
    }
    for value in edges["provenance_json"]:
        provenance = json.loads(value)
        assert len(provenance["missing_source_fold_ids"]) == 1
        assert len(provenance["source_fold_ids"]) == len(artifacts.folds)


def test_source_not_estimable_rows_fail_closed_and_preserve_reason() -> None:
    artifacts, bundle = _run(sender_min_subjects=10)
    edges = build_crossfit_communication_hypergraph(artifacts, bundle).hyperedges
    unavailable = edges.loc[edges["status"].eq("not_estimable")]

    assert not unavailable.empty
    assert unavailable["weight"].isna().all()
    assert set(unavailable["reason_code"]) == {
        "insufficient_complete_subject_ligand_contrasts"
    }
    for value in unavailable["provenance_json"]:
        source_rows = json.loads(value)["unavailable_source_rows"]
        assert source_rows
        assert {row["reason_code"] for row in source_rows} == {
            "insufficient_complete_subject_ligand_contrasts"
        }


def test_complex_resource_is_expanded_into_typed_memberships() -> None:
    artifacts, bundle = _run(bundle=_bundle(complex_ligand=True))
    graph = build_crossfit_communication_hypergraph(artifacts, bundle)
    complex_nodes = graph.nodes.loc[graph.nodes["node_type"].eq("ligand_complex")]

    assert complex_nodes["entity_id"].tolist() == ["L_COMPLEX"]
    assert json.loads(complex_nodes["metadata_json"].iloc[0]) == {
        "members": ["L1", "L2"]
    }
    assert graph.complex_members["complex_role"].tolist() == [
        "ligand_subunit",
        "ligand_subunit",
    ]
    assert graph.complex_members["member_index"].tolist() == [0, 1]


def test_missing_or_different_resource_is_rejected(
    standard_run: tuple[CrossFitArtifacts, ResourceBundle],
) -> None:
    artifacts, bundle = standard_run
    missing = ResourceBundle(
        resource_id=bundle.resource_id,
        version=bundle.version,
        species=bundle.species,
        gene_namespace=bundle.gene_namespace,
        interactions=(bundle.interactions[0],),
        mapping_report=MappingReport(1, 1, 2),
        manifest_digest="a" * 64,
        source_files=bundle.source_files,
        license=bundle.license,
        citation=bundle.citation,
    )

    with pytest.raises(ContractError) as error:
        build_crossfit_communication_hypergraph(artifacts, missing)

    assert error.value.details.code == "crossfit_hypergraph_resource_universe_mismatch"


def test_subject_equal_aggregation_is_not_sample_weighted() -> None:
    import crychic.workflow.hypergraph_export as export_module

    def row(
        sample_id: str,
        subject_id: str,
        value: float | None,
        status: HyperedgeStatus,
        reason: str | None = None,
    ) -> export_module._SourceRow:
        return export_module._SourceRow(
            contrast_manifest_id="contrast-1",
            contrast_name="contrast",
            context_id="context-1",
            context_json='{"condition":"stim"}',
            receiver="Receiver",
            family_id="family-1",
            driver_id="i1",
            interaction_id="i1",
            mode="state",
            sender="Sender",
            sample_id=sample_id,
            subject_id=subject_id,
            value=value,
            status=status,
            reason_code=reason,
            fold_id="fold-1",
            functional_id="functional-1",
            application_id="application-1",
            binding_id="binding-1",
            source_row_id=f"row-{sample_id}",
        )

    rows = [
        row("a1", "a", 0.0, HyperedgeStatus.STRUCTURAL_ZERO, "zero"),
        row("a2", "a", 1.0, HyperedgeStatus.OBSERVED),
        row("b1", "b", 1.0, HyperedgeStatus.OBSERVED),
    ]

    status, weight, reason, subjects = export_module._aggregate_status_and_weight(rows)

    assert status is HyperedgeStatus.OBSERVED
    assert weight == pytest.approx(0.75, rel=0.0, abs=1e-12)
    assert reason is None
    assert [item["technical_sample_mean"] for item in subjects] == [0.5, 1.0]

    unavailable = [
        *rows,
        row("b2", "b", None, HyperedgeStatus.NOT_ESTIMABLE, "source_ne"),
    ]
    status, weight, reason, _ = export_module._aggregate_status_and_weight(
        unavailable
    )
    assert status is HyperedgeStatus.NOT_ESTIMABLE
    assert weight is None
    assert reason == "source_ne"
