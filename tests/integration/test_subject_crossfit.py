from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import crychic.sender.common as common_sender_module
import crychic.workflow.crossfit as crossfit_module
import crychic.workflow.training as training_module
from crychic.core import CrychicConfig
from crychic.design import balanced_contrast
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.sender import ContrastCommonSenderParameters
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


def _adata() -> AnnData:
    genes = ("L1", "R1", "L2", "R2", "T1", "T2")
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    obs_names: list[str] = []
    for subject_index, subject in enumerate(("p1", "p2", "p3", "p4")):
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
                    obs_names.append(
                        f"{subject}-{condition}-{cell_type}-{cell_index}"
                    )
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
    assert result.certification_status == (
        "verified_train_only_oof_partial_pipeline"
    )
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
    assert set(result.oof_sender_assignments["functional_status"]) == {
        "out_of_fold"
    }
    manifests = {fold.fold_id: fold for fold in result.fold_plan.folds}
    for row in result.oof_sender_assignments.itertuples(index=False):
        manifest = manifests[str(row.fold_id)]
        assert str(row.subject_id) in manifest.test_subject_ids
        assert str(row.subject_id) not in manifest.train_subject_ids
    manifest = result.to_manifest()
    assert manifest["completed_stage_oof_verified"] is True
    assert manifest["complete_pipeline_oof_certified"] is False
    assert "common_scoring_functional" in manifest["remaining_stages"]


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
    first_rows = first.oof_sender_assignments.loc[
        first.oof_sender_assignments["fold_id"].eq(target_fold.fold_id)
    ].reset_index(drop=True)
    second_rows = second.oof_sender_assignments.loc[
        second.oof_sender_assignments["fold_id"].eq(target_fold.fold_id)
    ].reset_index(drop=True)
    assert not second_rows.equals(first_rows)


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
