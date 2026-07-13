from __future__ import annotations

import inspect
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import crychic.sender.common as common_sender_module
import crychic.workflow.training as training_module
from crychic.availability import BatchAvailability, InteractionFilterApplication
from crychic.core import ContractError, CrychicConfig
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
    FoldTrainingSpec,
    TrainingArtifacts,
    apply_training_artifacts,
    fit_training_artifacts,
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
        source="raw-boundary-fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _bundle() -> ResourceBundle:
    return ResourceBundle(
        resource_id="raw_boundary_lr",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(
            _interaction("i1", "L1", "R1"),
            _interaction("i2", "L2", "R2"),
        ),
        mapping_report=MappingReport(2, 2, 4),
        manifest_digest="a" * 64,
        source_files=("raw-boundary.tsv",),
        license="CC0",
        citation="Synthetic raw-boundary fixture",
    )


def _prior() -> TargetPrior:
    return TargetPrior(
        resource_id="raw_boundary_prior",
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
        manifest_digest="b" * 64,
    )


def _adata(subject_ids: tuple[str, ...], *, first_high: bool) -> AnnData:
    genes = ("L1", "R1", "L2", "R2", "T1", "T2")
    high, low = (100, 1) if first_high else (1, 100)
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    obs_names: list[str] = []
    for subject in subject_ids:
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
    adata.uns["full_data_poison"] = {"must_not_cross_scope": [1, 2, 3]}
    adata.obsm["full_data_embedding"] = np.ones((adata.n_obs, 2))
    return adata


def _config() -> CrychicConfig:
    return CrychicConfig(context_keys=("condition",), counts_layer="counts")


def _fit() -> TrainingArtifacts:
    return _fit_adata(_adata(("train-1", "train-2"), first_high=True))


def _fit_adata(adata: AnnData) -> TrainingArtifacts:
    return fit_training_artifacts(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=FoldTrainingSpec(
            min_cells=1,
            max_interactions=1,
            sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
        ),
    )


def test_training_entry_accepts_only_raw_and_preregistered_inputs() -> None:
    parameters = inspect.signature(fit_training_artifacts).parameters

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
        "response_matrix",
        "family_basis",
        "functional",
        "model_manifest_id",
    }
    assert forbidden.isdisjoint(parameters)
    with pytest.raises(TypeError, match="producer-owned"):
        TrainingArtifacts()


def test_training_entry_rejects_forced_nested_sender_policy_mutation() -> None:
    spec = FoldTrainingSpec(
        min_cells=1,
        max_interactions=1,
        sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
    )
    object.__setattr__(spec.sender_parameters, "min_subjects", 999)

    with pytest.raises(ContractError) as error:
        fit_training_artifacts(
            _adata(("train-1", "train-2"), first_high=True),
            _config(),
            _bundle(),
            _prior(),
            spec=spec,
        )
    assert error.value.details.code == "fold_training_spec_integrity_violation"


def test_application_rejects_forced_training_availability_policy_mutation() -> None:
    artifacts = _fit()
    object.__setattr__(artifacts.spec.availability_parameters, "complex_power", 99.0)

    with pytest.raises(ContractError) as error:
        apply_training_artifacts(
            artifacts,
            _adata(("heldout-1", "heldout-2"), first_high=False),
        )
    assert error.value.details.code == "training_artifact_integrity_violation"


def test_application_rejects_forced_training_subject_scope_mutation() -> None:
    artifacts = _fit()
    object.__setattr__(artifacts, "training_subject_ids", ("train-1",))

    with pytest.raises(ContractError) as error:
        apply_training_artifacts(
            artifacts,
            _adata(("train-2", "heldout-1"), first_high=False),
        )
    assert error.value.details.code == "training_artifact_integrity_violation"


def test_application_rejects_forced_training_resource_content_mutation() -> None:
    artifacts = _fit()
    interaction = artifacts.resource_bundle.interactions[0]
    object.__setattr__(interaction, "ligand_subunits", ("POISON",))

    with pytest.raises(ContractError) as error:
        apply_training_artifacts(
            artifacts,
            _adata(("heldout-1", "heldout-2"), first_high=False),
        )
    assert error.value.details.code == "training_artifact_integrity_violation"


def test_training_artifact_rejects_forced_target_prior_content_mutation() -> None:
    artifacts = _fit()
    poisoned = tuple(value + 1.0 for value in artifacts.target_prior.weights)
    object.__setattr__(artifacts.target_prior, "weights", poisoned)

    with pytest.raises(ContractError) as error:
        artifacts._require_intact()
    assert error.value.details.code == "training_artifact_integrity_violation"


def test_application_rejects_forced_frozen_universe_mutation() -> None:
    artifacts = _fit()
    object.__setattr__(
        artifacts.frozen_interaction_universe,
        "interaction_ids",
        (),
    )

    with pytest.raises(ContractError) as error:
        apply_training_artifacts(
            artifacts,
            _adata(("heldout-1", "heldout-2"), first_high=False),
        )
    assert error.value.details.code == "training_artifact_integrity_violation"


def test_training_application_rejects_nested_availability_table_mutation() -> None:
    application = apply_training_artifacts(
        _fit(),
        _adata(("heldout-1", "heldout-2"), first_high=False),
    )
    application.availability.sample_interactions.loc[
        application.availability.sample_interactions.index[0],
        "ligand_availability",
    ] = 0.123

    with pytest.raises(ContractError) as error:
        application._require_intact()
    assert error.value.details.code == "training_application_integrity_violation"


def test_training_application_rejects_nested_sender_table_mutation() -> None:
    application = apply_training_artifacts(
        _fit(),
        _adata(("heldout-1", "heldout-2"), first_high=False),
    )
    assignment = application.sender_assignments[0]
    assignment.table.loc[assignment.table.index[0], "assignment_weight"] = 0.0

    with pytest.raises(ContractError) as error:
        application._require_intact()
    assert error.value.details.code == "training_application_integrity_violation"


def test_raw_training_scope_derives_identity_and_fits_a_real_universe() -> None:
    artifacts = _fit()

    assert artifacts.training_subject_ids == ("train-1", "train-2")
    assert artifacts.training_sample_ids == (
        "sample-train-1-control",
        "sample-train-1-stim",
        "sample-train-2-control",
        "sample-train-2-stim",
    )
    assert artifacts.cell_type_ids == ("Receiver", "Sender")
    assert artifacts.completed_stages == (
        "availability_filter",
        "common_sender_functional",
    )
    assert "common_sender_functional" not in artifacts.remaining_stages
    assert "common_scoring_functional" in artifacts.remaining_stages
    assert artifacts.certification_status == "training_only_partial_not_oof"
    assert artifacts.is_oof_certified is False
    assert artifacts.frozen_interaction_universe.interaction_ids == ("i1",)
    assert artifacts.frozen_interaction_universe.training_subject_ids == (
        "train-1",
        "train-2",
    )
    assert len(artifacts.sender_functionals) == 1
    sender_functional = artifacts.sender_functionals[0]
    assert sender_functional.training_subject_ids == (
        "train-1",
        "train-2",
    )
    assert len(sender_functional.context_ids) == 2
    assert sender_functional.contrast_manifest_id
    assert sender_functional.filter_universe_id == (
        artifacts.frozen_interaction_universe.filter_universe_id
    )


def test_training_stage_receives_only_sanitized_declared_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = training_module._fit_interaction_universe

    def inspect_scope(
        prepared: training_module._PreparedRawFold,
        resource_bundle: ResourceBundle,
        spec: FoldTrainingSpec,
    ) -> BatchAvailability:
        sanitized = prepared.validated.adata
        assert not sanitized.uns
        assert not sanitized.obsm
        assert tuple(sanitized.obs.columns) == (
            "sample_id",
            "subject_id",
            "cell_type",
            "condition",
        )
        return original(prepared, resource_bundle, spec)

    monkeypatch.setattr(
        training_module,
        "_fit_interaction_universe",
        inspect_scope,
    )

    artifacts = _fit()
    assert artifacts.completed_stages == (
        "availability_filter",
        "common_sender_functional",
    )


def _assert_declared_metadata_changes_digest(metadata_kind: str) -> None:
    original = _adata(("train-1", "train-2"), first_high=True)
    changed = original.copy()
    selected = changed.obs["subject_id"].eq("train-1")
    if metadata_kind == "context":
        changed.obs.loc[selected, "condition"] = "control"
    else:
        changed.obs.loc[selected, "cell_type"] = changed.obs.loc[
            selected, "cell_type"
        ].map({"Sender": "Receiver", "Receiver": "Sender"})

    first = _fit_adata(original)
    second = _fit_adata(changed)

    assert first.frozen_interaction_universe.interaction_ids == (
        second.frozen_interaction_universe.interaction_ids
    )
    assert first.training_input_digest != second.training_input_digest
    assert first.training_artifact_id != second.training_artifact_id


def test_training_context_changes_input_and_artifact_digest() -> None:
    _assert_declared_metadata_changes_digest("context")


def test_training_cell_type_changes_input_and_artifact_digest() -> None:
    _assert_declared_metadata_changes_digest("cell_type")


def test_ordered_var_names_are_part_of_training_input_digest() -> None:
    original = _adata(("train-1", "train-2"), first_high=True)
    changed = original.copy()
    feature_ids = changed.var_names.astype(str).tolist()
    feature_ids[-1] = "UNUSED_TARGET_RENAMED"
    changed.var_names = feature_ids

    first = _fit_adata(original)
    second = _fit_adata(changed)

    assert first.frozen_interaction_universe.interaction_ids == (
        second.frozen_interaction_universe.interaction_ids
    )
    assert first.training_input_digest != second.training_input_digest
    assert first.training_artifact_id != second.training_artifact_id


def test_training_does_not_mutate_caller_anndata() -> None:
    adata = _adata(("train-1", "train-2"), first_high=True)
    before = adata.copy()
    before_uns = deepcopy(adata.uns)

    _fit_adata(adata)

    pd.testing.assert_frame_equal(adata.obs, before.obs)
    pd.testing.assert_frame_equal(adata.var, before.var)
    assert tuple(adata.var_names) == tuple(before.var_names)
    assert (sparse.csr_matrix(adata.X) != sparse.csr_matrix(before.X)).nnz == 0
    assert (
        sparse.csr_matrix(adata.layers["counts"])
        != sparse.csr_matrix(before.layers["counts"])
    ).nnz == 0
    assert adata.uns == before_uns
    np.testing.assert_array_equal(
        adata.obsm["full_data_embedding"],
        before.obsm["full_data_embedding"],
    )


def test_application_uses_frozen_universe_when_training_fits_are_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _fit()

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("application called a training fit function")

    monkeypatch.setattr(training_module, "fit_training_artifacts", forbidden_fit)
    monkeypatch.setattr(training_module, "_fit_interaction_universe", forbidden_fit)
    monkeypatch.setattr(
        training_module,
        "fit_contrast_common_sender_functional",
        forbidden_fit,
    )
    monkeypatch.setattr(
        common_sender_module,
        "fit_contrast_common_sender_functional",
        forbidden_fit,
    )
    heldout = _adata(("test-1",), first_high=False)
    application = apply_training_artifacts(artifacts, heldout)

    assert application.is_oof_certified is False
    assert application.application_status == "frozen_application_partial_not_oof"
    assert application.heldout_subject_ids == ("test-1",)
    assert application.training_artifact_id == artifacts.training_artifact_id
    assert application.availability.filter_application is (
        InteractionFilterApplication.FROZEN_APPLICATION_V1
    )
    assert set(
        application.availability.sample_interactions["interaction_id"].astype(str)
    ) == {"i1"}
    assert application.availability.filter_universe_id == (
        artifacts.frozen_interaction_universe.filter_universe_id
    )
    assert len(application.sender_assignments) == 1
    sender_assignment = application.sender_assignments[0]
    assert sender_assignment.functional.sender_functional_id == (
        artifacts.sender_functionals[0].sender_functional_id
    )
    assert sender_assignment.table["assignment_weight"].notna().any()
    assert not sender_assignment.is_oof_certified


def test_heldout_expression_poison_cannot_change_training_artifacts() -> None:
    first = _fit()
    heldout_a = _adata(("test-1",), first_high=True)
    heldout_b = _adata(("test-1",), first_high=False)

    applied_a = apply_training_artifacts(first, heldout_a)
    applied_b = apply_training_artifacts(first, heldout_b)

    assert first.frozen_interaction_universe.interaction_ids == ("i1",)
    assert applied_a.availability.filter_universe_id == (
        applied_b.availability.filter_universe_id
    )
    assert applied_a.heldout_input_digest != applied_b.heldout_input_digest
    assert not applied_a.availability.sample_interactions.equals(
        applied_b.availability.sample_interactions
    )
    assert len(first.sender_functionals) == 1
    assert len(applied_a.sender_assignments) == 1
    assert len(applied_b.sender_assignments) == 1
    assert applied_a.sender_assignments[0].functional.sender_functional_id == (
        first.sender_functionals[0].sender_functional_id
    )
    assert applied_b.sender_assignments[0].functional.sender_functional_id == (
        first.sender_functionals[0].sender_functional_id
    )
    assert not applied_a.sender_assignments[0].table.equals(
        applied_b.sender_assignments[0].table
    )


def test_application_rejects_actual_subject_overlap() -> None:
    artifacts = _fit()

    with pytest.raises(ValueError, match="overlaps training subjects"):
        apply_training_artifacts(
            artifacts,
            _adata(("train-2",), first_high=False),
        )
