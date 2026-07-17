from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import crychic.response.autonomous as autonomous_module
import crychic.scoring.downstream as downstream_module
import crychic.workflow.receiver_incremental as incremental_module
from crychic.attribution import (
    GainCalibrationSpec,
    PenaltyTuningSpec,
    PenaltyValidationLossEstimand,
    PrecisionTransformResult,
    ReceiverFamilyTrainingArtifact,
    fit_receiver_family_training_artifact,
    fit_response_precision,
)
from crychic.attribution.precision import fit_repeated_response_precision
from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)
from crychic.core import ContractError, SeedLineage
from crychic.design import (
    FrozenDesignEncoder,
    SubjectDesign,
    apply_frozen_design_encoder,
    balanced_contrast,
    fit_frozen_design_encoder,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.resources import (
    GeneNamespace,
    MappingReport,
    ResourceManifest,
    Species,
    TargetPrior,
)
from crychic.resources.autonomous_registry import (
    ReceiverAutonomousProgramRegistration,
)
from crychic.response import (
    FoldGeneResponseArtifact,
    apply_fold_gene_response,
    build_receiver_autonomous_program_resource,
    fit_fold_gene_response,
    load_receiver_autonomous_program_resource,
)
from crychic.response.repeated_fold import (
    apply_repeated_measures_fold_response,
    fit_repeated_measures_cr2_fold_response,
    fit_repeated_measures_fold_response,
)
from crychic.scoring import (
    DownstreamRowManifest,
    FrozenLatentNuisanceSpec,
    apply_incremental_downstream_functional,
)
from crychic.workflow import (
    ReceiverIncrementalApplication,
    ReceiverIncrementalTrainingArtifact,
    apply_receiver_incremental_training_artifact,
    fit_receiver_incremental_training_artifact,
)


def _metadata(subjects: tuple[str, ...], *, prefix: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"{prefix}{subject}:{condition}",
                "subject_id": f"{prefix}{subject}",
                "condition": condition,
            }
            for subject in subjects
            for condition in ("ctrl", "stim")
        ]
    )


def _independent_metadata(
    n_subjects_per_context: int,
    *,
    prefix: str = "",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"{prefix}c{index}:ctrl",
                "subject_id": f"{prefix}c{index}",
                "condition": "ctrl",
            }
            for index in range(1, n_subjects_per_context + 1)
        ]
        + [
            {
                "sample_id": f"{prefix}s{index}:stim",
                "subject_id": f"{prefix}s{index}",
                "condition": "stim",
            }
            for index in range(1, n_subjects_per_context + 1)
        ]
    )


def _encoder(metadata: pd.DataFrame) -> FrozenDesignEncoder:
    return fit_frozen_design_encoder(
        metadata,
        contrast=balanced_contrast(("stim",), ("ctrl",), name="stim_vs_ctrl"),
        context_keys=("condition",),
        formula="~ condition",
    )


def _aggregate(
    metadata: pd.DataFrame,
    *,
    reverse_units: bool = False,
    offset: int = 0,
    missing_samples: frozenset[str] = frozenset(),
) -> PseudobulkDataset:
    counts: list[tuple[int, int]] = []
    units: list[dict[str, object]] = []
    matrix_ids: list[str] = []
    for position, (_, row) in enumerate(metadata.reset_index(drop=True).iterrows()):
        sample_id = str(row["sample_id"])
        condition = str(row["condition"])
        first = 12 + position + offset + (18 if condition == "stim" else 0)
        unit_id = f"unit:{sample_id}"
        missing = sample_id in missing_samples
        matrix_row: int | pd._libs.missing.NAType
        if missing:
            matrix_row = pd.NA
        else:
            matrix_row = len(counts)
            counts.append((first, 120 - first))
            matrix_ids.append(unit_id)
        units.append(
            {
                "unit_id": unit_id,
                "sample_id": sample_id,
                "subject_id": str(row["subject_id"]),
                "cell_type": "Receiver",
                "context": (("condition", condition),),
                "matrix_row": matrix_row,
                "n_cells": 0 if missing else 20,
                "cell_proportion": 1.0,
                "state_eligible": not missing,
                "abundance_eligible": True,
                "missingness_reason": "sampling_zero" if missing else "observed",
            }
        )
    unit_metadata = pd.DataFrame(units)
    if reverse_units:
        unit_metadata = unit_metadata.iloc[::-1].reset_index(drop=True)
    matrix = sparse.csr_matrix(np.asarray(counts, dtype=np.int64))
    return PseudobulkDataset(
        counts=matrix,
        detection_fraction=sparse.csr_matrix(matrix.toarray() > 0),
        unit_metadata=unit_metadata,
        feature_ids=("G1", "G2"),
        matrix_unit_ids=tuple(matrix_ids),
        source_location="synthetic",
    )


def _prior(columns: Mapping[str, Mapping[str, float]]) -> TargetPrior:
    drivers = tuple(sorted(columns))
    targets = tuple(sorted({key for values in columns.values() for key in values}))
    target_index = {target: index for index, target in enumerate(targets)}
    indices: list[int] = []
    weights: list[float] = []
    indptr = [0]
    for driver in drivers:
        for target, weight in sorted(columns[driver].items()):
            indices.append(target_index[target])
            weights.append(weight)
        indptr.append(len(weights))
    return TargetPrior(
        resource_id="tiny-prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=targets,
        driver_ids=drivers,
        indptr=tuple(indptr),
        target_indices=tuple(indices),
        weights=tuple(weights),
        ranks=None,
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(len(weights), len(weights), len(targets)),
        manifest_digest="tiny-prior-manifest",
    )


def _receiver_family(
    subjects: tuple[str, ...],
    *,
    fold_id: str = "fold-1",
    receptor_value: float = 0.9,
) -> ReceiverFamilyTrainingArtifact:
    rows = [
        {
            "sample_id": f"{subject}:ctrl",
            "subject_id": subject,
            "context_id": "ctrl",
            "receiver": "Receiver",
            "interaction_id": interaction,
            "receptor_availability": receptor_value,
        }
        for subject in subjects
        for interaction in ("i1", "i2")
    ]
    universe = FrozenInteractionUniverse(
        interaction_ids=("i1", "i2"),
        training_subject_ids=subjects,
        resource_id="tiny-lr",
        resource_version="1",
        resource_manifest_digest="tiny-lr-manifest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )
    availability = BatchAvailability(
        sample_interactions=pd.DataFrame(rows),
        mapping_summary=pd.DataFrame(),
        resource_id="tiny-lr",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=universe,
        filter_application=InteractionFilterApplication.TRAINING_SELECTION_V1,
        application_subject_ids=subjects,
    )
    return fit_receiver_family_training_artifact(
        availability,
        _prior({"D1": {"G1": 1.0}, "D2": {"G2": 1.0}}),
        receiver="Receiver",
        fold_id=fold_id,
        feature_ids=("G1", "G2"),
        driver_by_interaction={"i1": "D1", "i2": "D2"},
        receptor_gate_threshold=0.1,
        cosine_threshold=0.99,
    )


def _training_parents() -> tuple[
    FrozenDesignEncoder,
    FoldGeneResponseArtifact,
    PrecisionTransformResult,
    ReceiverFamilyTrainingArtifact,
]:
    metadata = _metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    response = fit_fold_gene_response(
        _aggregate(metadata),
        encoder,
        receiver="Receiver",
        fold_id="fold-1",
        training_input_digest="training-input",
    )
    precision = fit_response_precision(response, min_positive_features=2)
    family = _receiver_family(response.training_subject_ids)
    return encoder, response, precision, family


def _latent_training_parents(
    *,
    include_ineligible_target: bool = False,
) -> tuple[
    FrozenDesignEncoder,
    FoldGeneResponseArtifact,
    PrecisionTransformResult,
    ReceiverFamilyTrainingArtifact,
]:
    metadata = _metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    counts: list[tuple[int, int, int, int]] = []
    units: list[dict[str, object]] = []
    matrix_ids: list[str] = []
    for position, (_, row) in enumerate(metadata.reset_index(drop=True).iterrows()):
        sample_id = str(row["sample_id"])
        subject_id = str(row["subject_id"])
        subject_index = int(subject_id[1:])
        is_stim = str(row["condition"]) == "stim"
        latent_value = (subject_index - 2.5) * 4.0
        counts.append(
            (
                30 + 12 * int(is_stim) + subject_index,
                80 + int(latent_value) + 3 * int(is_stim),
                65 + int(2.0 * latent_value) + 2 * int(is_stim),
                90 - int(latent_value) + int(is_stim),
            )
        )
        unit_id = f"unit:{sample_id}"
        matrix_ids.append(unit_id)
        units.append(
            {
                "unit_id": unit_id,
                "sample_id": sample_id,
                "subject_id": subject_id,
                "cell_type": "Receiver",
                "context": (("condition", str(row["condition"])),),
                "matrix_row": position,
                "n_cells": 20,
                "cell_proportion": 1.0,
                "state_eligible": True,
                "abundance_eligible": True,
                "missingness_reason": "observed",
            }
        )
    matrix = sparse.csr_matrix(np.asarray(counts, dtype=np.int64))
    aggregate = PseudobulkDataset(
        counts=matrix,
        detection_fraction=sparse.csr_matrix(matrix.toarray() > 0),
        unit_metadata=pd.DataFrame(units),
        feature_ids=("G1", "G2", "G3", "G4"),
        matrix_unit_ids=tuple(matrix_ids),
        source_location="synthetic-latent-nuisance",
    )
    response = fit_fold_gene_response(
        aggregate,
        encoder,
        receiver="Receiver",
        fold_id="fold-latent",
        training_input_digest="latent-training-input",
    )
    precision = fit_response_precision(response, min_positive_features=2)
    subjects = response.training_subject_ids
    interaction_ids = ("i1", "i2") if include_ineligible_target else ("i1",)
    universe = FrozenInteractionUniverse(
        interaction_ids=interaction_ids,
        training_subject_ids=subjects,
        resource_id="tiny-lr",
        resource_version="1",
        resource_manifest_digest="tiny-lr-manifest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )
    availability = BatchAvailability(
        sample_interactions=pd.DataFrame(
            [
                {
                    "sample_id": f"{subject}:ctrl",
                    "subject_id": subject,
                    "context_id": "ctrl",
                    "receiver": "Receiver",
                    "interaction_id": interaction_id,
                    "receptor_availability": (0.9 if interaction_id == "i1" else 0.0),
                }
                for subject in subjects
                for interaction_id in interaction_ids
            ]
        ),
        mapping_summary=pd.DataFrame(),
        resource_id="tiny-lr",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=universe,
        filter_application=InteractionFilterApplication.TRAINING_SELECTION_V1,
        application_subject_ids=subjects,
    )
    family = fit_receiver_family_training_artifact(
        availability,
        _prior(
            {
                "D1": {"G1": 1.0},
                **({"D2": {"G2": 1.0}} if include_ineligible_target else {}),
            }
        ),
        receiver="Receiver",
        fold_id="fold-latent",
        feature_ids=("G1", "G2", "G3", "G4"),
        driver_by_interaction={
            "i1": "D1",
            **({"i2": "D2"} if include_ineligible_target else {}),
        },
        receptor_gate_threshold=0.1,
        cosine_threshold=0.99,
    )
    return encoder, response, precision, family


def _independent_training_parents() -> tuple[
    FrozenDesignEncoder,
    FoldGeneResponseArtifact,
    PrecisionTransformResult,
    ReceiverFamilyTrainingArtifact,
]:
    metadata = _independent_metadata(4)
    encoder = _encoder(metadata)
    response = fit_fold_gene_response(
        _aggregate(metadata),
        encoder,
        receiver="Receiver",
        fold_id="fold-independent",
        training_input_digest="training-input-independent",
    )
    precision = fit_response_precision(response, min_positive_features=2)
    family = _receiver_family(
        response.training_subject_ids,
        fold_id="fold-independent",
    )
    return encoder, response, precision, family


def _small_tuning_spec(**changes: object) -> PenaltyTuningSpec:
    arguments: dict[str, object] = {
        "lambda1_fractions": (1.0, 0.1),
        "lambda2_fractions": (0.0,),
        "inner_allowed_n_splits": (2,),
    }
    arguments.update(changes)
    return PenaltyTuningSpec(**arguments)  # type: ignore[arg-type]


def _mixed_incremental_inputs() -> Any:
    sample_subject_ids = (
        "p1",
        "p1",
        "p2",
        "p2",
        "c1",
        "c2",
        "s1",
        "s2",
    )
    sample_context_ids = (
        "ctrl",
        "stim",
        "ctrl",
        "stim",
        "ctrl",
        "ctrl",
        "stim",
        "stim",
    )
    sample_ids = tuple(
        f"{subject}:{context}:{index}"
        for index, (subject, context) in enumerate(
            zip(sample_subject_ids, sample_context_ids, strict=True)
        )
    )
    return incremental_module._IncrementalFitInputs(
        response_matrix=np.column_stack(
            (
                np.asarray([0.0, 2.0, 0.1, 2.1, -0.1, 0.0, 1.9, 2.0]),
                np.asarray([3.0, 3.0, 3.1, 3.1, 2.9, 3.0, 2.9, 3.0]),
            )
        ),
        sample_ids=sample_ids,
        sample_subject_ids=sample_subject_ids,
        sample_context_ids=sample_context_ids,
        nuisance_matrix=np.ones((8, 1)),
        context_regressor=np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0]),
        reference_mask=np.asarray([True, False, True, False, True, True, False, False]),
        receiver="Receiver",
        contrast_name="stim_vs_ctrl",
        context_regressor_id="stim_vs_ctrl_regressor_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=("G1", "G2"),
        family_ids=("family-1",),
        nuisance_column_ids=("intercept",),
        family_basis=np.asarray([[1.0], [0.0]]),
        latent_control_exclusion_basis=np.asarray([[1.0], [0.0]]),
        autonomous_program_resource=None,
        latent_nuisance_spec=None,
        precision_weights=np.ones(2),
        minimum_scale=0.25,
        null_loss_floor=1e-8,
    )


def _trusted_autonomous_resource(root: Path, monkeypatch: pytest.MonkeyPatch):
    payload = b"feature_id\tgeneric_program\nG1\t1\nG2\t1\n"
    payload_path = root / "programs.tsv"
    payload_path.write_bytes(payload)
    manifest_record = {
        "resource_id": "test_trusted_autonomous_programs",
        "version": "1",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "source_url": "https://example.invalid/test-autonomous-programs",
        "license": "CC0-1.0",
        "citation": "Synthetic static program used only by unit tests.",
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
    manifest_path.write_text(json.dumps(manifest_record), encoding="utf-8")
    digest = ResourceManifest.from_json(manifest_path).digest
    matrix_digest = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("G1", "G2"),
        program_ids=("generic_program",),
        resource_id="digest-only",
        version="1",
        manifest_digest="0" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    ).matrix_digest
    registration = ReceiverAutonomousProgramRegistration(
        registration_id="test.trusted.autonomous.v1",
        manifest_digest=digest,
        resource_id="test_trusted_autonomous_programs",
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


def test_public_fit_and_apply_signatures_accept_only_typed_parents() -> None:
    fit_parameters = inspect.signature(
        fit_receiver_incremental_training_artifact
    ).parameters
    apply_parameters = inspect.signature(
        apply_receiver_incremental_training_artifact
    ).parameters

    assert tuple(fit_parameters) == (
        "encoder",
        "response",
        "precision",
        "receiver_family",
        "autonomous_program_resource",
        "latent_nuisance_spec",
        "minimum_scale",
        "null_loss_floor",
        "lambda1",
        "lambda2",
        "penalty_tuning_spec",
        "gain_calibration_spec",
        "inner_partition_seed_lineage",
    )
    assert tuple(apply_parameters) == (
        "model",
        "response_application",
        "design_application",
    )
    forbidden = {
        "response_matrix",
        "sample_ids",
        "reference_mask",
        "family_basis",
        "context_regressor_id",
        "nuisance_design_id",
    }
    assert forbidden.isdisjoint(fit_parameters)
    assert forbidden.isdisjoint(apply_parameters)
    with pytest.raises(TypeError, match="producer-owned"):
        ReceiverIncrementalTrainingArtifact()
    with pytest.raises(TypeError, match="producer-owned"):
        ReceiverIncrementalApplication()


def test_typed_fit_joins_samples_and_keeps_official_status_honest() -> None:
    encoder, response, precision, family = _training_parents()

    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family, lambda2=0.01
    )

    assert model.diagnostic_status == "observed"
    assert model.diagnostic_functional is not None
    assert model.diagnostic_functional.training_sample_ids == response.sample_ids
    assert set(model.diagnostic_functional.reference_sample_ids) == {
        f"p{index}:ctrl" for index in range(1, 5)
    }
    assert model.family_ids == family.eligible_family_ids
    assert model.certification_status == (
        "formula_nuisance_incremental_diagnostic_only"
    )
    assert model.official_incremental_status == "not_estimable"
    assert model.reason_code == "receiver_autonomous_nuisance_not_frozen"
    assert model.is_oof_certified is False
    assert model.to_dict()["response_artifact_id"] == response.artifact_id


def test_tuned_formula_nuisance_keeps_diagnostic_separate_from_official() -> None:
    encoder, response, precision, family = _training_parents()
    tuning_spec = _small_tuning_spec(lambda1_fractions=(1.0,))

    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        penalty_tuning_spec=tuning_spec,
    )

    assert model.inner_fold_plan is not None
    assert model.penalty_tuning_artifact is not None
    assert model.penalty_tuning_artifact.status == "selected"
    assert model.penalty_tuning_artifact.is_oof_certified
    assert model.gain_calibration_spec is not None
    assert model.gain_calibration_artifact is not None
    assert model.gain_calibration_artifact.status == "not_estimable"
    assert model.gain_calibration_artifact.reason_code == (
        "insufficient_inner_oof_subjects"
    )
    expected_subjects = set(response.training_subject_ids)
    for evaluation in model.penalty_tuning_artifact.evaluations:
        assert set(evaluation.inner_training_subject_ids).isdisjoint(
            evaluation.validation_subject_ids
        )
        assert (
            set(evaluation.inner_training_subject_ids).union(
                evaluation.validation_subject_ids
            )
            == expected_subjects
        )
    assert model.diagnostic_status == "observed"
    assert model.diagnostic_functional is not None
    assert model.official_incremental_status == "not_estimable"
    assert model.reason_code == "receiver_autonomous_nuisance_not_frozen"
    assert not model.is_oof_certified

    heldout_metadata = _metadata(("q1", "q2"))
    heldout_design = apply_frozen_design_encoder(encoder, heldout_metadata)
    heldout_response = apply_fold_gene_response(
        _aggregate(heldout_metadata, offset=7), response, heldout_design
    )
    application = apply_receiver_incremental_training_artifact(
        model, heldout_response, heldout_design
    )

    assert application.diagnostic_status == "observed"
    assert application.diagnostic_application is not None
    assert application.official_incremental_status == "not_estimable"
    assert application.reason_code == "receiver_autonomous_nuisance_not_frozen"
    assert not application.is_oof_certified


def test_caller_declared_autonomous_resource_remains_noncertifying() -> None:
    encoder, response, precision, family = _training_parents()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("G1", "G2"),
        program_ids=("generic_program",),
        resource_id="test-autonomous-programs",
        version="1",
        manifest_digest="a" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )

    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family, resource
    )

    assert model.autonomous_program_resource_id == resource.artifact_id
    assert model.autonomous_projection_id is not None
    assert model.certification_status == "formula_nuisance_incremental_diagnostic_only"
    assert model.official_incremental_status == "not_estimable"
    assert model.reason_code == "receiver_autonomous_nuisance_not_frozen"
    assert model.is_oof_certified is False
    assert model.diagnostic_functional is not None
    assert model.diagnostic_functional.autonomous_program_ids == ("generic_program",)


def test_subject_blocked_inner_tuning_has_complete_verified_lineage() -> None:
    encoder, response, precision, family = _training_parents()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("G1", "G2"),
        program_ids=("generic_program",),
        resource_id="test-autonomous-programs",
        version="1",
        manifest_digest="c" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    tuning_spec = _small_tuning_spec()

    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        resource,
        penalty_tuning_spec=tuning_spec,
    )

    assert model.inner_fold_plan is not None
    assert model.penalty_tuning_artifact is not None
    tuning = model.penalty_tuning_artifact
    assert model.inner_fold_plan.partition_seed_lineage is None
    assert "partition_seed_lineage" not in model.inner_fold_plan.to_dict()
    assert model.inner_fold_plan.plan_id == (
        "subject_fold_plan_90f4410c6d923f1a03f522a4f1d38c5d"
    )
    assert tuple(fold.fold_id for fold in model.inner_fold_plan.folds) == (
        "subject_fold_bb831a289fa98184b20854f9928ffaf2",
        "subject_fold_d78fc0d232566059819396df119cd4e7",
    )
    assert tuning.tuning_id == (
        "penalty_tuning_artifact_ae50c0a458fc9c0f69a758dbde8c0103"
    )
    assert model.training_artifact_id == (
        "receiver_incremental_training_artifact_13748856b002f8528086df13c2afc42a"
    )
    assert tuning.status == "selected"
    assert tuning.is_oof_certified
    assert tuning.validation_loss_estimand is (
        PenaltyValidationLossEstimand.PAIRED_SUBJECT_CONTRAST
    )
    assert "outer_frozen_representation" in tuning.certification_status
    assert tuning.inner_fold_plan_id == model.inner_fold_plan.plan_id
    assert len(tuning.evaluations) == (
        len(tuning_spec.candidates) * model.inner_fold_plan.effective_n_splits
    )
    expected_subjects = set(response.training_subject_ids)
    for evaluation in tuning.evaluations:
        assert "outer_frozen_representation" in evaluation.verification_status
        assert evaluation.validation_loss_estimand is (
            PenaltyValidationLossEstimand.PAIRED_SUBJECT_CONTRAST
        )
        assert set(evaluation.inner_training_subject_ids).isdisjoint(
            evaluation.validation_subject_ids
        )
        assert (
            set(evaluation.inner_training_subject_ids).union(
                evaluation.validation_subject_ids
            )
            == expected_subjects
        )
        assert evaluation.status == "observed"
        assert evaluation.scale_resolution_id is not None
        assert evaluation.resolved_penalty_id is not None
        assert evaluation.resolved_lambda1 is not None
        assert evaluation.resolved_lambda1 >= 0
        assert evaluation.resolved_lambda2 is not None
        assert evaluation.resolved_lambda2 >= 0
        assert evaluation.training_functional_id is not None
        assert evaluation.heldout_application_id is not None
    assert model.diagnostic_functional is not None
    assert model.selected_penalty_candidate_id == tuning.selected_candidate_id
    assert (
        model.selected_penalty_candidate_id
        == model.diagnostic_functional.penalty_candidate_id
    )
    assert (
        model.selected_penalty_scale_resolution_id
        == model.diagnostic_functional.penalty_scale_resolution_id
    )
    assert (
        model.selected_resolved_penalty_id
        == model.diagnostic_functional.resolved_penalty_id
    )
    assert model.lambda1 == model.diagnostic_functional.lambda1
    assert model.lambda2 == model.diagnostic_functional.lambda2
    assert model.gain_calibration_spec is not None
    assert model.gain_calibration_spec_id == model.gain_calibration_spec.spec_id
    assert model.gain_calibration_artifact is not None
    calibration = model.gain_calibration_artifact
    assert calibration.status == "not_estimable"
    assert calibration.reason_code == "insufficient_inner_oof_subjects"
    assert calibration.tuning_id == tuning.tuning_id
    assert calibration.outer_incremental_functional_id == (
        model.diagnostic_functional.incremental_functional_id
    )
    assert calibration.training_subject_ids == response.training_subject_ids
    assert calibration.family_ids == model.family_ids
    assert not model.is_oof_certified
    assert model.reason_code == "receiver_autonomous_nuisance_not_frozen"
    model.to_dict()


def test_latent_nuisance_is_refit_inside_every_inner_training_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, response, precision, family = _latent_training_parents()
    latent_spec = FrozenLatentNuisanceSpec(
        max_components=1,
        min_control_features=2,
        min_training_subjects=2,
        minimum_explained_fraction=0.01,
    )
    tuning_spec = _small_tuning_spec()
    captured: list[Any] = []
    original = downstream_module.fit_fold_latent_nuisance

    def capture_latent_fit(*args: Any, **kwargs: Any) -> Any:
        artifact = original(*args, **kwargs)
        captured.append(artifact)
        return artifact

    monkeypatch.setattr(
        downstream_module,
        "fit_fold_latent_nuisance",
        capture_latent_fit,
    )
    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        None,
        latent_nuisance_spec=latent_spec,
        penalty_tuning_spec=tuning_spec,
    )

    assert model.inner_fold_plan is not None
    assert model.latent_nuisance_artifact is not None
    assert model.latent_nuisance_artifact.status == "observed"
    assert model.latent_nuisance_artifact.training_subject_ids == (
        response.training_subject_ids
    )
    assert model.diagnostic_functional is not None
    assert model.autonomous_program_source_id == (
        model.diagnostic_functional.autonomous_basis_id
    )
    assert model.is_oof_certified
    assert model.official_incremental_status == "observed"
    for inner_fold in model.inner_fold_plan.folds:
        inner_artifacts = [
            artifact for artifact in captured if artifact.fold_id == inner_fold.fold_id
        ]
        assert len(inner_artifacts) == len(tuning_spec.candidates)
        assert len({artifact.artifact_id for artifact in inner_artifacts}) == 1
        for artifact in inner_artifacts:
            assert artifact.status == "observed"
            assert artifact.training_subject_ids == inner_fold.train_subject_ids
            assert set(artifact.training_subject_ids).isdisjoint(
                inner_fold.test_subject_ids
            )
            assert {
                subject_id for subject_id in artifact.training_sample_subject_ids
            } == set(inner_fold.train_subject_ids)
    outer_artifacts = [
        artifact for artifact in captured if artifact.fold_id == response.fold_id
    ]
    assert len(outer_artifacts) == 1
    model.to_dict()


def test_latent_controls_exclude_targets_from_ineligible_families() -> None:
    encoder, response, precision, family = _latent_training_parents(
        include_ineligible_target=True
    )
    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        None,
        latent_nuisance_spec=FrozenLatentNuisanceSpec(
            max_components=1,
            min_control_features=2,
            min_training_subjects=2,
            minimum_explained_fraction=0.01,
        ),
    )

    assert len(model.family_ids) == 1
    assert model.latent_nuisance_artifact is not None
    assert model.latent_nuisance_artifact.status == "observed"
    assert model.latent_nuisance_artifact.family_count == 2
    assert model.latent_nuisance_artifact.control_feature_ids == ("G3", "G4")


def test_latent_not_estimable_artifact_is_retained_for_diagnostics() -> None:
    encoder, response, precision, family = _latent_training_parents()
    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        None,
        latent_nuisance_spec=FrozenLatentNuisanceSpec(
            max_components=1,
            min_control_features=5,
            min_training_subjects=2,
            minimum_explained_fraction=0.01,
        ),
    )

    assert model.diagnostic_functional is None
    assert model.latent_nuisance_artifact is not None
    assert model.latent_nuisance_artifact.status == "not_estimable"
    assert model.latent_nuisance_artifact.reason_code == (
        "latent_nuisance_insufficient_control_features"
    )
    assert model.diagnostic_reason_code == (
        "latent_nuisance_insufficient_control_features"
    )
    assert model.autonomous_program_source_id is None
    assert model.autonomous_program_verification_status is None
    payload = model.to_dict()
    assert payload["latent_nuisance_artifact"] is not None


def test_hybrid_latent_basis_binds_the_exact_static_prefix() -> None:
    encoder, response, precision, family = _latent_training_parents()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[0.0], [0.0], [0.0], [1.0]]),
        feature_ids=("G1", "G2", "G3", "G4"),
        program_ids=("static-program",),
        resource_id="hybrid-static-programs",
        version="1",
        manifest_digest="7" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        resource,
        latent_nuisance_spec=FrozenLatentNuisanceSpec(
            max_components=1,
            min_control_features=2,
            min_training_subjects=2,
            minimum_explained_fraction=0.01,
        ),
    )

    assert model.diagnostic_functional is not None
    assert model.latent_nuisance_artifact is not None
    artifact = model.latent_nuisance_artifact
    functional = model.diagnostic_functional
    assert artifact.status == "observed"
    assert artifact.static_program_ids == ("static-program",)
    assert functional.autonomous_basis_source == (
        "static_plus_fold_learned_control_feature_latent_v1"
    )
    assert functional.autonomous_program_ids == (
        "static-program",
        *artifact.program_ids,
    )
    np.testing.assert_allclose(
        functional.autonomous_basis[:, :1],
        artifact.static_coordinate_basis,
    )
    np.testing.assert_allclose(
        functional.autonomous_basis[:, 1:],
        artifact.coordinate_basis,
    )
    functional.to_dict()


def test_independent_inner_tuning_uses_subject_local_losses_and_applies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, response, precision, family = _independent_training_parents()
    resource = _trusted_autonomous_resource(tmp_path, monkeypatch)
    captured: dict[str, tuple[Any, Any]] = {}
    original_apply = incremental_module._apply_incremental_subset

    def capture_application(
        functional: Any,
        inputs: Any,
        *,
        subject_ids: tuple[str, ...],
    ) -> Any:
        application = original_apply(
            functional,
            inputs,
            subject_ids=subject_ids,
        )
        captured[application.application_id] = (functional, application)
        return application

    monkeypatch.setattr(
        incremental_module,
        "_apply_incremental_subset",
        capture_application,
    )
    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        resource,
        penalty_tuning_spec=_small_tuning_spec(),
    )

    assert model.inner_fold_plan is not None
    assert model.penalty_tuning_artifact is not None
    tuning = model.penalty_tuning_artifact
    assert tuning.status == "selected"
    assert tuning.is_oof_certified
    assert tuning.validation_loss_estimand is (
        PenaltyValidationLossEstimand.INDEPENDENT_SUBJECT_PREDICTION
    )
    assert model.diagnostic_functional is not None
    assert model.diagnostic_functional.loss_design == (
        "independent_subject_pseudocontrasts_v1"
    )
    assert model.is_oof_certified
    for evaluation in tuning.evaluations:
        assert evaluation.validation_loss_estimand is (
            PenaltyValidationLossEstimand.INDEPENDENT_SUBJECT_PREDICTION
        )
        functional, application = captured[evaluation.heldout_application_id]
        assert functional.training_subject_ids == (
            evaluation.inner_training_subject_ids
        )
        assert set(functional.training_subject_ids).isdisjoint(
            evaluation.validation_subject_ids
        )
        expected_losses = []
        for subject in evaluation.validation_subject_ids:
            selected = np.asarray(
                [
                    row_subject == subject
                    for row_subject in application.sample_subject_ids
                ],
                dtype=bool,
            )
            expected_losses.append(
                float(np.mean(application.sample_full_losses[selected]))
            )
        np.testing.assert_allclose(
            evaluation.subject_losses,
            np.asarray(expected_losses),
        )

    heldout_metadata = _independent_metadata(2, prefix="heldout-")
    heldout_design = apply_frozen_design_encoder(encoder, heldout_metadata)
    heldout_response = apply_fold_gene_response(
        _aggregate(heldout_metadata, offset=9),
        response,
        heldout_design,
    )
    application = apply_receiver_incremental_training_artifact(
        model,
        heldout_response,
        heldout_design,
    )

    assert application.diagnostic_status == "observed"
    assert application.official_incremental_status == "observed"
    assert application.reason_code is None
    assert application.is_oof_certified
    assert set(application.heldout_subject_ids).isdisjoint(model.training_subject_ids)
    application.to_dict()


def test_inner_tuning_allocation_planner_classifies_mixed_subjects() -> None:
    inputs = SimpleNamespace(
        sample_ids=("p1:ctrl", "p1:stim", "c2:ctrl", "s2:stim"),
        sample_subject_ids=("p1", "p1", "c2", "s2"),
        sample_context_ids=("ctrl", "stim", "ctrl", "stim"),
    )

    allocation = incremental_module._inner_tuning_subject_design(
        inputs  # type: ignore[arg-type]
    )

    assert allocation.design is SubjectDesign.MIXED
    assert not allocation.ready
    assert allocation.reason_code == "mixed_paired_unpaired_design_not_supported"


def test_mixed_inner_split_preserves_outer_full_prediction_estimand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _mixed_incremental_inputs()
    fold_lineage = SeedLineage(17).derive("crossfit", "mixed-repeat", "k=2", "fold=0")
    fold = incremental_module.FoldManifest(
        repeat_id="mixed-repeat",
        train_subject_ids=("p1", "p2"),
        test_subject_ids=("c1", "c2", "s1", "s2"),
        design_matrix_id="mixed-inner-design",
        contrast_ids=("mixed-inner-contrast",),
        context_support={"ctrl": 2, "stim": 2},
        test_context_support={"ctrl": 2, "stim": 2},
        estimable=True,
        reason_code=None,
        seed_lineage=fold_lineage,
        requested_n_splits=2,
        effective_n_splits=2,
        fold_index=0,
    )
    captured_loss_designs: list[str] = []

    def fake_plan(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(folds=(fold,), plan_id="mixed-plan")

    def fake_select(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            selected_candidate_id=None,
            inner_fold_ids=(fold.fold_id,),
        )

    original_fit = incremental_module._fit_incremental_subset

    def capture_fit(*args: object, **kwargs: object) -> Any:
        functional = original_fit(*args, **kwargs)
        captured_loss_designs.append(functional.loss_design)
        return functional

    monkeypatch.setattr(incremental_module, "plan_subject_folds", fake_plan)
    monkeypatch.setattr(incremental_module, "select_penalty_candidate", fake_select)
    monkeypatch.setattr(incremental_module, "_fit_incremental_subset", capture_fit)

    incremental_module._fit_subject_blocked_penalty_tuning(
        inputs,
        spec=PenaltyTuningSpec(
            lambda1_fractions=(0.0,),
            lambda2_fractions=(0.0,),
            inner_allowed_n_splits=(2,),
        ),
        tuning_scope_id="mixed-scope",
        outer_fold_id="outer-mixed",
        inner_partition_seed_lineage=None,
    )

    assert captured_loss_designs == ["mixed_subject_equal_full_prediction_loss_v1"]


def test_mixed_subject_blocked_tuning_runs_complete_candidate_fold_grid() -> None:
    spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.1),
        lambda2_fractions=(0.0,),
        inner_allowed_n_splits=(2,),
        min_inner_train_subjects_per_context=2,
        min_inner_validation_subjects_per_context=1,
    )

    outcome = incremental_module._fit_subject_blocked_penalty_tuning(
        _mixed_incremental_inputs(),
        spec=spec,
        tuning_scope_id="mixed-real-plan-scope",
        outer_fold_id="outer-mixed-real-plan",
        inner_partition_seed_lineage=None,
    )
    plan, tuning = outcome.plan, outcome.tuning

    assert plan is not None
    assert plan.effective_n_splits == 2
    assert tuning.status == "selected"
    assert tuning.validation_loss_estimand is (
        PenaltyValidationLossEstimand.MIXED_SUBJECT_PREDICTION
    )
    assert len(tuning.evaluations) == len(spec.candidates) * len(plan.folds)
    assert all(item.status == "observed" for item in tuning.evaluations)
    assert all(
        item.validation_loss_estimand
        is PenaltyValidationLossEstimand.MIXED_SUBJECT_PREDICTION
        for item in tuning.evaluations
    )
    assert len(outcome.selected_inner_functionals) == len(plan.folds)
    assert len(outcome.selected_inner_applications) == len(plan.folds)
    assert all(
        item.penalty_candidate_id == tuning.selected_candidate_id
        for item in outcome.selected_inner_functionals
    )
    tuning.to_dict()


def test_explicit_inner_partition_lineage_is_persisted_and_requires_tuning() -> None:
    encoder, response, precision, family = _training_parents()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("G1", "G2"),
        program_ids=("generic_program",),
        resource_id="test-autonomous-programs",
        version="1",
        manifest_digest="9" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    lineage = SeedLineage(808).derive("paired-inner-partition")

    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        resource,
        penalty_tuning_spec=_small_tuning_spec(),
        inner_partition_seed_lineage=lineage,
    )

    assert model.inner_fold_plan is not None
    assert model.inner_fold_plan.partition_seed_lineage == lineage
    assert model.inner_fold_plan.to_dict()["partition_seed_lineage"] == (
        lineage.to_dict()
    )

    with pytest.raises(ValueError, match="requires an explicit penalty_tuning_spec"):
        fit_receiver_incremental_training_artifact(
            encoder,
            response,
            precision,
            family,
            inner_partition_seed_lineage=lineage,
        )
    with pytest.raises(TypeError, match="inner_partition_seed_lineage"):
        fit_receiver_incremental_training_artifact(
            encoder,
            response,
            precision,
            family,
            penalty_tuning_spec=_small_tuning_spec(),
            inner_partition_seed_lineage=123,  # type: ignore[arg-type]
        )


def test_inner_tuning_plan_failure_is_typed_and_never_falls_back() -> None:
    encoder, response, precision, family = _training_parents()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("G1", "G2"),
        program_ids=("generic_program",),
        resource_id="test-autonomous-programs",
        version="1",
        manifest_digest="d" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )

    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        resource,
        penalty_tuning_spec=_small_tuning_spec(min_inner_train_subjects_per_context=3),
    )

    assert model.inner_fold_plan is None
    assert model.penalty_tuning_artifact is not None
    assert model.penalty_tuning_artifact.status == "not_estimable"
    assert model.penalty_tuning_artifact.reason_code == (
        "no_estimable_subject_fold_plan"
    )
    assert model.diagnostic_functional is None
    assert model.diagnostic_status == "not_estimable"
    assert model.diagnostic_reason_code == "no_estimable_subject_fold_plan"
    assert model.selected_penalty_candidate_id is None
    assert model.gain_calibration_spec is not None
    assert model.gain_calibration_artifact is None
    model.to_dict()


def test_gain_calibration_spec_requires_subject_blocked_tuning() -> None:
    encoder, response, precision, family = _training_parents()

    with pytest.raises(ValueError, match="requires penalty_tuning_spec"):
        fit_receiver_incremental_training_artifact(
            encoder,
            response,
            precision,
            family,
            gain_calibration_spec=GainCalibrationSpec(),
        )


@pytest.mark.parametrize("target", ["tuning", "plan", "selected_penalty"])
def test_tuned_training_rejects_forced_nested_lineage_mutation(target: str) -> None:
    encoder, response, precision, family = _training_parents()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("G1", "G2"),
        program_ids=("generic_program",),
        resource_id="test-autonomous-programs",
        version="1",
        manifest_digest="f" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        resource,
        penalty_tuning_spec=_small_tuning_spec(),
    )
    assert model.penalty_tuning_artifact is not None
    assert model.inner_fold_plan is not None

    if target == "tuning":
        object.__setattr__(model.penalty_tuning_artifact, "tuning_id", "poison")
    elif target == "plan":
        object.__setattr__(model.inner_fold_plan, "plan_id", "poison")
    else:
        object.__setattr__(model, "selected_resolved_penalty_id", "poison")

    with pytest.raises(ContractError) as error:
        model.to_dict()
    assert error.value.details.code == (
        "receiver_incremental_training_integrity_violation"
    )


def test_unknown_inner_value_error_is_not_downgraded_to_not_estimable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, response, precision, family = _training_parents()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("G1", "G2"),
        program_ids=("generic_program",),
        resource_id="test-autonomous-programs",
        version="1",
        manifest_digest="1" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )

    monkeypatch.setattr(
        incremental_module,
        "_fit_incremental_subset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("simulated programming defect")
        ),
    )

    with pytest.raises(ValueError, match="simulated programming defect"):
        fit_receiver_incremental_training_artifact(
            encoder,
            response,
            precision,
            family,
            resource,
            penalty_tuning_spec=_small_tuning_spec(),
        )


def test_known_inner_solver_failure_is_recorded_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, response, precision, family = _training_parents()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("G1", "G2"),
        program_ids=("generic_program",),
        resource_id="test-autonomous-programs",
        version="1",
        manifest_digest="2" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )

    monkeypatch.setattr(
        incremental_module,
        "_fit_incremental_subset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError(
                "incremental downstream family solver did not converge: max iterations"
            )
        ),
    )

    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        resource,
        penalty_tuning_spec=_small_tuning_spec(),
    )

    assert model.penalty_tuning_artifact is not None
    assert model.penalty_tuning_artifact.status == "failed"
    assert {item.status for item in model.penalty_tuning_artifact.evaluations} == {
        "failed"
    }
    assert model.diagnostic_status == "not_estimable"
    assert model.diagnostic_reason_code == "no_candidate_complete_inner_coverage"


def test_trusted_resource_and_verified_tuning_certify_outer_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, response, precision, family = _training_parents()
    resource = _trusted_autonomous_resource(tmp_path, monkeypatch)
    fixed = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family, resource
    )
    assert not fixed.is_oof_certified
    assert fixed.official_incremental_status == "not_estimable"
    assert fixed.reason_code == "subject_blocked_inner_tuning_not_connected"
    assert fixed.certification_status == (
        "trusted_autonomous_incremental_not_oof_certified_v1"
    )
    model = fit_receiver_incremental_training_artifact(
        encoder,
        response,
        precision,
        family,
        resource,
        penalty_tuning_spec=_small_tuning_spec(),
    )

    assert model.is_oof_certified
    assert model.official_incremental_status == "observed"
    assert model.reason_code is None
    assert "outer_frozen_representation" in model.certification_status

    heldout_metadata = _metadata(("q1", "q2"))
    heldout_design = apply_frozen_design_encoder(encoder, heldout_metadata)
    heldout_response = apply_fold_gene_response(
        _aggregate(heldout_metadata, offset=7), response, heldout_design
    )
    application = apply_receiver_incremental_training_artifact(
        model, heldout_response, heldout_design
    )

    assert application.diagnostic_status == "observed"
    assert application.official_incremental_status == "observed"
    assert application.reason_code is None
    assert application.is_oof_certified
    assert "outer_frozen_representation" in application.certification_status
    assert set(application.heldout_subject_ids).isdisjoint(model.training_subject_ids)
    application.to_dict()


def test_autonomous_resource_without_aligned_support_fails_closed() -> None:
    encoder, response, precision, family = _training_parents()
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("OTHER1", "OTHER2"),
        program_ids=("unsupported_program",),
        resource_id="unsupported-autonomous-programs",
        version="1",
        manifest_digest="b" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )

    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family, resource
    )

    assert model.diagnostic_status == "not_estimable"
    assert model.diagnostic_reason_code == "autonomous_program_support_not_estimable"
    assert model.diagnostic_functional is None
    assert model.autonomous_program_resource_id == resource.artifact_id
    assert model.autonomous_projection_id is None
    model.to_dict()


def test_training_wrapper_rejects_a_different_valid_functional() -> None:
    encoder, response, precision, family = _training_parents()
    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family, lambda2=0.01
    )
    other = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family, lambda2=0.02
    )
    assert other.diagnostic_functional is not None

    object.__setattr__(model, "diagnostic_functional", other.diagnostic_functional)

    with pytest.raises(ContractError) as error:
        model.to_dict()
    assert error.value.details.code == (
        "receiver_incremental_training_integrity_violation"
    )


def test_heldout_application_is_order_stable_and_does_not_fit() -> None:
    encoder, response, precision, family = _training_parents()
    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family
    )
    metadata = _metadata(("q1", "q2"))
    design = apply_frozen_design_encoder(
        encoder, metadata.sample(frac=1.0, random_state=11)
    )
    heldout_response = apply_fold_gene_response(
        _aggregate(metadata, reverse_units=True, offset=7), response, design
    )

    with patch.object(
        incremental_module,
        "fit_incremental_downstream_functional",
        side_effect=AssertionError("apply must not fit"),
    ):
        applied = apply_receiver_incremental_training_artifact(
            model, heldout_response, design
        )

    assert applied.diagnostic_application is not None
    assert applied.heldout_sample_ids == tuple(sorted(metadata["sample_id"]))
    assert applied.heldout_subject_ids == ("q1", "q2")
    assert applied.official_incremental_status == "not_estimable"
    assert applied.reason_code == "receiver_autonomous_nuisance_not_frozen"
    assert applied.is_oof_certified is False
    applied.to_dict()


def test_application_wrapper_rejects_valid_diagnostic_from_other_values() -> None:
    encoder, response, precision, family = _training_parents()
    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family
    )
    assert model.diagnostic_functional is not None
    metadata = _metadata(("q1", "q2"))
    design = apply_frozen_design_encoder(encoder, metadata)
    heldout_response = apply_fold_gene_response(
        _aggregate(metadata, offset=7), response, design
    )
    row_manifest = DownstreamRowManifest(
        sample_ids=heldout_response.sample_ids,
        subject_ids=heldout_response.sample_subject_ids,
        context_ids=heldout_response.sample_context_ids,
    )
    alternate_values = np.asarray(heldout_response.sample_values).copy()
    alternate_values[0, 0] += 100.0
    alternate = apply_incremental_downstream_functional(
        model.diagnostic_functional,
        alternate_values,
        row_manifest=row_manifest,
        design_sample_ids=design.sample_ids,
        nuisance_matrix=design.nuisance_matrix,
        context_regressor=design.context_regressor,
        context_regressor_id=design.context_regressor_id,
        nuisance_design_id=design.nuisance_design_id,
        feature_ids=heldout_response.feature_ids,
        nuisance_column_ids=model.diagnostic_functional.nuisance_column_ids,
    )
    assert alternate.status == "observed"

    with pytest.raises(ContractError) as error:
        ReceiverIncrementalApplication._from_application(
            model=model,
            response_application=heldout_response,
            design_application=design,
            diagnostic_application=alternate,
            diagnostic_reason_code=None,
        )
    assert error.value.details.code == "receiver_incremental_parent_mismatch"


def test_wrong_family_parent_and_training_subject_overlap_are_rejected() -> None:
    encoder, response, precision, _ = _training_parents()
    wrong_family = _receiver_family(
        response.training_subject_ids, fold_id="different-fold"
    )
    with pytest.raises(ContractError) as mismatch:
        fit_receiver_incremental_training_artifact(
            encoder, response, precision, wrong_family
        )
    assert mismatch.value.details.code == "receiver_incremental_parent_mismatch"

    family = _receiver_family(response.training_subject_ids)
    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family
    )
    overlapping_design = apply_frozen_design_encoder(
        encoder, _metadata(("p1",), prefix="heldout-")
    )
    object.__setattr__(overlapping_design, "sample_subject_ids", ("p1", "p1"))
    with pytest.raises(ContractError):
        overlapping_design.to_dict()

    heldout_metadata = _metadata(("p1",))
    with pytest.raises(ValueError, match="overlaps training subjects"):
        apply_frozen_design_encoder(encoder, heldout_metadata)
    assert model.training_subject_ids == response.training_subject_ids


def test_no_eligible_family_is_explicit_diagnostic_ne() -> None:
    encoder, response, precision, _ = _training_parents()
    no_family = _receiver_family(response.training_subject_ids, receptor_value=0.01)

    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, no_family
    )

    assert model.diagnostic_functional is None
    assert model.diagnostic_status == "not_estimable"
    assert model.diagnostic_reason_code == "no_training_eligible_receiver_family"
    assert model.official_incremental_status == "not_estimable"
    assert model.reason_code == "receiver_autonomous_nuisance_not_frozen"


def test_incomplete_training_response_coverage_fails_closed() -> None:
    metadata = _metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    response = fit_fold_gene_response(
        _aggregate(metadata, missing_samples=frozenset({"p1:ctrl"})),
        encoder,
        receiver="Receiver",
        fold_id="fold-1",
        training_input_digest="training-input-incomplete",
    )
    precision = fit_response_precision(response, min_positive_features=2)
    family = _receiver_family(response.training_subject_ids)

    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, family
    )

    assert response.missing_sample_ids == ("p1:ctrl",)
    assert model.diagnostic_functional is None
    assert model.diagnostic_status == "not_estimable"
    assert model.diagnostic_reason_code == (
        "training_receiver_response_incomplete_sample_coverage"
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("minimum_scale", -1.0, "minimum_scale must be finite and positive"),
        ("null_loss_floor", 0.0, "null_loss_floor must be finite and positive"),
        ("lambda1", -1.0, "lambda1 must be finite and non-negative"),
        ("lambda2", float("nan"), "lambda2 must be finite and non-negative"),
    ],
)
def test_hyperparameters_are_validated_before_not_estimable_short_circuit(
    field_name: str,
    invalid_value: float,
    message: str,
) -> None:
    metadata = _metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    response = fit_fold_gene_response(
        _aggregate(metadata, missing_samples=frozenset({"p1:ctrl"})),
        encoder,
        receiver="Receiver",
        fold_id="fold-1",
        training_input_digest="training-input-incomplete",
    )
    precision = fit_response_precision(response, min_positive_features=2)
    family = _receiver_family(response.training_subject_ids)

    with pytest.raises(ValueError, match=message):
        fit_receiver_incremental_training_artifact(
            encoder,
            response,
            precision,
            family,
            **{field_name: invalid_value},
        )


def test_not_estimable_artifact_revalidates_hyperparameters() -> None:
    encoder, response, precision, _ = _training_parents()
    no_family = _receiver_family(response.training_subject_ids, receptor_value=0.01)
    model = fit_receiver_incremental_training_artifact(
        encoder, response, precision, no_family
    )

    object.__setattr__(model, "lambda1", -1.0)
    with pytest.raises(ContractError, match="integrity"):
        model.to_dict()


def _mixed_repeated_metadata() -> pd.DataFrame:
    allocations = {
        "p1": ("ctrl", "stim"),
        "p2": ("ctrl", "stim"),
        "c1": ("ctrl",),
        "c2": ("ctrl",),
        "c3": ("ctrl",),
        "s1": ("stim",),
        "s2": ("stim",),
        "s3": ("stim",),
    }
    return pd.DataFrame(
        [
            {
                "sample_id": f"{subject}:{condition}",
                "subject_id": subject,
                "condition": condition,
            }
            for subject, conditions in allocations.items()
            for condition in conditions
        ]
    )


def test_cr2_repeated_parent_can_reach_descriptive_oof_without_releasing_pq(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _mixed_repeated_metadata()
    encoder = _encoder(metadata)
    aggregate = _aggregate(metadata)
    fit_arguments: dict[str, object] = {
        "receiver": "Receiver",
        "fold_id": "fold-cr2",
        "training_input_digest": "training-input-cr2",
        "min_subjects_per_context": 2,
        "min_subject_clusters": 6,
    }
    cr1 = fit_repeated_measures_fold_response(
        aggregate,
        encoder,
        metadata,
        **fit_arguments,  # type: ignore[arg-type]
    )
    cr2 = fit_repeated_measures_cr2_fold_response(
        aggregate,
        encoder,
        metadata,
        **fit_arguments,  # type: ignore[arg-type]
    )
    assert cr1.status == "exploratory"
    assert cr2.status == "ok"
    feature_scale = incremental_module._receiver_incremental_feature_scale(
        encoder, cr2, minimum_scale=0.25
    )
    cr1_precision = fit_repeated_response_precision(
        cr1,
        feature_scale=feature_scale,
        min_positive_features=2,
    )
    cr2_precision = fit_repeated_response_precision(
        cr2,
        feature_scale=feature_scale,
        min_positive_features=2,
    )
    insufficient_cr2_precision = fit_repeated_response_precision(
        cr2,
        feature_scale=feature_scale,
        min_positive_features=3,
    )
    family = _receiver_family(cr2.training_subject_ids, fold_id="fold-cr2")
    resource = _trusted_autonomous_resource(tmp_path, monkeypatch)
    tuning = _small_tuning_spec()

    cr1_model = fit_receiver_incremental_training_artifact(
        encoder,
        cr1,
        cr1_precision,
        family,
        resource,
        penalty_tuning_spec=tuning,
    )
    cr2_model = fit_receiver_incremental_training_artifact(
        encoder,
        cr2,
        cr2_precision,
        family,
        resource,
        penalty_tuning_spec=tuning,
    )
    insufficient_cr2_model = fit_receiver_incremental_training_artifact(
        encoder,
        cr2,
        insufficient_cr2_precision,
        family,
        resource,
        penalty_tuning_spec=tuning,
    )

    assert cr1_model.diagnostic_status == "observed"
    assert cr1_model.official_incremental_status == "not_estimable"
    assert cr1_model.reason_code == (
        "repeated_measures_cr1_diagnostic_only_no_formal_inference"
    )
    assert not cr1_model.is_oof_certified
    assert cr2_model.diagnostic_status == "observed"
    assert cr2_model.official_incremental_status == "observed"
    assert cr2_model.reason_code is None
    assert cr2_model.is_oof_certified
    assert cr2.formal_inference_allowed is False
    assert {"p", "p_value", "q", "q_value"}.isdisjoint(cr2_model.to_dict())
    assert not insufficient_cr2_precision.estimable
    assert insufficient_cr2_model.diagnostic_status == "not_estimable"
    assert insufficient_cr2_model.diagnostic_reason_code == (
        "insufficient_response_precision_support"
    )
    assert insufficient_cr2_model.official_incremental_status == "not_estimable"
    assert insufficient_cr2_model.reason_code == (
        "insufficient_response_precision_support"
    )
    assert insufficient_cr2_model.reason_code != (
        "repeated_measures_cr1_diagnostic_only_no_formal_inference"
    )

    heldout_metadata = _metadata(("q1", "q2"))
    heldout_design = apply_frozen_design_encoder(encoder, heldout_metadata)
    heldout_response = apply_repeated_measures_fold_response(
        _aggregate(heldout_metadata), cr2, heldout_design
    )
    application = apply_receiver_incremental_training_artifact(
        cr2_model, heldout_response, heldout_design
    )
    assert application.diagnostic_status == "observed"
    assert application.official_incremental_status == "observed"
    assert application.reason_code is None
    assert application.is_oof_certified
    assert {"p", "p_value", "q", "q_value"}.isdisjoint(application.to_dict())
