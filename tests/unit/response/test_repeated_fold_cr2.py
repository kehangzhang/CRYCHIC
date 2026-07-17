from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from crychic.attribution.precision import fit_repeated_response_precision
from crychic.core import ContractError
from crychic.design import (
    apply_frozen_design_encoder,
    balanced_contrast,
    fit_frozen_design_encoder,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.response import (
    RepeatedMeasuresCR2ReceiverEffect,
    RepeatedMeasuresReceiverEffect,
)
from crychic.response.repeated_fold import (
    apply_repeated_measures_fold_response,
    fit_repeated_measures_cr2_fold_response,
    fit_repeated_measures_fold_response,
)
from crychic.workflow.crossfit import _response_backend_manifest


def _mixed_metadata(*, include_sixth_subject: bool = True) -> pd.DataFrame:
    allocations = {
        "p1": ("control", "case"),
        "p2": ("control", "case"),
        "p3": ("control",),
        "p4": ("control",),
        "p5": ("case",),
    }
    if include_sixth_subject:
        allocations["p6"] = ("case",)
    return pd.DataFrame(
        [
            {
                "sample_id": f"{subject}-{context}",
                "subject_id": subject,
                "condition": context,
            }
            for subject, contexts in allocations.items()
            for context in contexts
        ]
    )


def _heldout_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"q{subject_index}-{context}",
                "subject_id": f"q{subject_index}",
                "condition": context,
            }
            for subject_index in (1, 2)
            for context in ("control", "case")
        ]
    )


def _aggregate(
    metadata: pd.DataFrame, *, degenerate_third_feature: bool = False
) -> PseudobulkDataset:
    subjects = {
        subject: index
        for index, subject in enumerate(
            sorted(metadata["subject_id"].astype(str).unique()), start=1
        )
    }
    counts: list[tuple[int, int, int]] = []
    units: list[dict[str, object]] = []
    matrix_unit_ids: list[str] = []
    for row in metadata.itertuples(index=False):
        sample_id = str(row.sample_id)
        subject_id = str(row.subject_id)
        context = str(row.condition)
        subject_index = subjects[subject_id]
        stimulated = context == "case"
        values = (
            100 + 17 * subject_index + (83 + 5 * (subject_index % 2)) * stimulated,
            210 + 11 * subject_index + (41 + 3 * (subject_index % 3)) * stimulated,
            (
                0
                if degenerate_third_feature
                else 730 + 13 * subject_index - (19 + subject_index % 2) * stimulated
            ),
        )
        unit_id = f"unit:{sample_id}"
        matrix_row = len(counts)
        counts.append(values)
        matrix_unit_ids.append(unit_id)
        units.append(
            {
                "unit_id": unit_id,
                "sample_id": sample_id,
                "subject_id": subject_id,
                "cell_type": "Receiver",
                "context": (("condition", context),),
                "matrix_row": matrix_row,
                "n_cells": 20,
                "cell_proportion": 1.0,
                "state_eligible": True,
                "abundance_eligible": True,
                "missingness_reason": "observed",
            }
        )
    matrix = sparse.csr_matrix(np.asarray(counts, dtype=np.int64))
    return PseudobulkDataset(
        counts=matrix,
        detection_fraction=sparse.csr_matrix(matrix.toarray() > 0),
        unit_metadata=pd.DataFrame(units),
        feature_ids=("G1", "G2", "G3"),
        matrix_unit_ids=tuple(matrix_unit_ids),
        source_location="synthetic-cr2-fold-test",
    )


def _encoder(metadata: pd.DataFrame):
    return fit_frozen_design_encoder(
        metadata,
        contrast=balanced_contrast(("case",), ("control",), name="case_vs_control"),
        context_keys=("condition",),
        formula="~ condition",
    )


def test_cr2_fold_response_preserves_cr1_and_binds_backend_precision() -> None:
    metadata = _mixed_metadata()
    encoder = _encoder(metadata)
    aggregate = _aggregate(metadata)

    cr1 = fit_repeated_measures_fold_response(
        aggregate,
        encoder,
        metadata,
        receiver="Receiver",
        fold_id="fold-1",
        training_input_digest="training-input",
        min_subjects_per_context=2,
        min_subject_clusters=6,
    )
    cr2 = fit_repeated_measures_cr2_fold_response(
        aggregate,
        encoder,
        metadata,
        receiver="Receiver",
        fold_id="fold-1",
        training_input_digest="training-input",
        min_subjects_per_context=2,
        min_subject_clusters=6,
    )

    assert isinstance(cr1.repeated_effect, RepeatedMeasuresReceiverEffect)
    assert cr1.status == "exploratory"
    assert cr1.is_cr1_exploratory
    assert not cr1.cr2_backend_eligible
    assert cr1.method == "formula_ols_subject_cluster_cr1_exploratory_v1"
    cr1.to_dict()

    assert isinstance(cr2.repeated_effect, RepeatedMeasuresCR2ReceiverEffect)
    assert cr2.status == "ok"
    assert cr2.observed_status == "ok"
    assert cr2.cr2_backend_eligible
    assert not cr2.is_cr1_exploratory
    assert not cr2.formal_inference_allowed
    assert "cr2" in cr2.method
    assert cr2.artifact_id != cr1.artifact_id
    assert np.isfinite(cr2.effect).all()
    assert np.isfinite(cr2.standard_error).all()
    assert np.all(cr2.raw_precision > 0.0)
    cr2.to_dict()

    cr1_manifest = _response_backend_manifest(cr1)
    cr2_manifest = _response_backend_manifest(cr2)
    assert cr1_manifest["artifact_kind"] == "repeated_cr1_fold_response_v1"
    assert cr2_manifest["artifact_kind"] == "repeated_cr2_fold_response_v1"
    cr1_diagnostics = cr1_manifest["feature_diagnostics"]
    cr2_diagnostics = cr2_manifest["feature_diagnostics"]
    assert isinstance(cr1_diagnostics, list)
    assert isinstance(cr2_diagnostics, list)
    assert {row["backend"] for row in cr1_diagnostics} == {cr1.method}
    assert {row["status"] for row in cr1_diagnostics} == {"exploratory"}
    assert not any(row["formal_backend_eligible"] for row in cr1_diagnostics)
    assert {row["backend"] for row in cr2_diagnostics} == {
        "subject_equal_wls_cluster_cr2_v1"
    }
    assert {row["status"] for row in cr2_diagnostics} == {"eligible"}
    assert all(row["formal_backend_eligible"] for row in cr2_diagnostics)
    assert [row["n_effective_clusters"] for row in cr1_diagnostics] == [
        effect.n_contrast_subject_clusters
        for effect in cr1.repeated_effect.feature_effects
    ]
    assert [row["n_effective_clusters"] for row in cr2_diagnostics] == [
        effect.n_effective_clusters for effect in cr2.repeated_effect.feature_effects
    ]

    cr1_precision = fit_repeated_response_precision(
        cr1,
        feature_scale=np.ones(3),
        min_positive_features=2,
    )
    cr2_precision = fit_repeated_response_precision(
        cr2,
        feature_scale=np.ones(3),
        min_positive_features=2,
    )
    assert cr1_precision.method == (
        "repeated_cr1_diagnostic_standardized_inverse_variance_v1"
    )
    assert cr1_precision.lineage_mode == ("repeated_measures_fold_response_parented_v1")
    assert cr2_precision.method == "repeated_cr2_standardized_inverse_variance_v1"
    assert cr2_precision.lineage_mode == (
        "repeated_measures_cr2_fold_response_parented_v1"
    )
    assert "cr1" not in cr2_precision.weight_mode
    assert cr2_precision.estimable
    cr2_precision.require_response_compatible(cr2)

    heldout_metadata = _heldout_metadata()
    heldout_design = apply_frozen_design_encoder(encoder, heldout_metadata)
    heldout = apply_repeated_measures_fold_response(
        _aggregate(heldout_metadata), cr2, heldout_design
    )
    assert heldout.status == "ok"
    assert heldout.reason_code is None
    heldout.require_compatible(cr2, heldout_design)


def test_cr1_fold_response_current_identity_golden_is_frozen() -> None:
    """Freeze the current legacy CR1 IDs without changing production payloads."""

    metadata = _mixed_metadata()
    encoder = _encoder(metadata)
    response = fit_repeated_measures_fold_response(
        _aggregate(metadata),
        encoder,
        metadata,
        receiver="Receiver",
        fold_id="fold-1",
        training_input_digest="training-input",
        min_subjects_per_context=2,
        min_subject_clusters=6,
    )
    precision = fit_repeated_response_precision(
        response,
        feature_scale=np.ones(3),
        min_positive_features=2,
    )
    heldout_metadata = _heldout_metadata()
    application = apply_repeated_measures_fold_response(
        _aggregate(heldout_metadata),
        response,
        apply_frozen_design_encoder(encoder, heldout_metadata),
    )

    assert response.artifact_id == (
        "repeated_measures_fold_response_3b3cb34b4ffa782b92537cf128fdd353"
    )
    assert application.application_id == (
        "repeated_measures_fold_response_application_845b011051b5272de2b5ed90a63c79c0"
    )
    assert precision.precision_transform_id == (
        "precision_transform_e36f499f426c6b7089ff8cb41b5a618c"
    )


def test_cr2_fold_response_fails_closed_without_cr1_fallback() -> None:
    metadata = _mixed_metadata(include_sixth_subject=False)
    encoder = _encoder(metadata)
    response = fit_repeated_measures_cr2_fold_response(
        _aggregate(metadata),
        encoder,
        metadata,
        receiver="Receiver",
        fold_id="fold-ne",
        training_input_digest="training-input-ne",
        min_subjects_per_context=2,
        min_subject_clusters=6,
    )

    assert isinstance(response.repeated_effect, RepeatedMeasuresCR2ReceiverEffect)
    assert response.status == "not_estimable"
    assert response.reason_code == "insufficient_subject_clusters"
    assert not response.cr2_backend_eligible
    assert not response.is_cr1_exploratory
    assert np.isnan(response.effect).all()
    assert np.isnan(response.raw_precision).all()
    precision = fit_repeated_response_precision(
        response,
        feature_scale=np.ones(3),
        min_positive_features=2,
    )
    assert not precision.estimable
    assert precision.reason_code == "insufficient_response_precision_support"
    assert "cr2" in precision.method

    heldout_metadata = _heldout_metadata()
    heldout_design = apply_frozen_design_encoder(encoder, heldout_metadata)
    heldout = apply_repeated_measures_fold_response(
        _aggregate(heldout_metadata), response, heldout_design
    )
    assert heldout.status == "not_estimable"
    assert heldout.reason_code == (
        "training_response_not_estimable:insufficient_subject_clusters"
    )


def test_cr2_fold_response_retains_partial_feature_eligibility() -> None:
    metadata = _mixed_metadata()
    response = fit_repeated_measures_cr2_fold_response(
        _aggregate(metadata, degenerate_third_feature=True),
        _encoder(metadata),
        metadata,
        receiver="Receiver",
        fold_id="fold-partial",
        training_input_digest="training-input-partial",
        min_subjects_per_context=2,
        min_subject_clusters=6,
    )

    effects = response.repeated_effect.feature_effects
    assert response.status == "ok"
    assert response.cr2_backend_eligible
    assert sum(effect.formal_backend_eligible for effect in effects) == 2
    assert effects[2].reason_code == "degenerate_cr2_contrast_variance"
    assert np.isfinite(response.raw_precision[:2]).all()
    assert np.isnan(response.raw_precision[2])

    estimable = fit_repeated_response_precision(
        response,
        feature_scale=np.ones(3),
        min_positive_features=2,
    )
    insufficient = fit_repeated_response_precision(
        response,
        feature_scale=np.ones(3),
        min_positive_features=3,
    )
    assert estimable.estimable
    assert estimable.n_positive_features == 2
    assert not insufficient.estimable
    assert insufficient.reason_code == "insufficient_response_precision_support"


def test_cr2_fold_response_rejects_backend_tampering() -> None:
    metadata = _mixed_metadata()
    response = fit_repeated_measures_cr2_fold_response(
        _aggregate(metadata),
        _encoder(metadata),
        metadata,
        receiver="Receiver",
        fold_id="fold-tamper",
        training_input_digest="training-input-tamper",
        min_subjects_per_context=2,
        min_subject_clusters=6,
    )
    object.__setattr__(
        response, "method", "formula_ols_subject_cluster_cr1_exploratory_v1"
    )

    with pytest.raises(ContractError) as error:
        response.to_dict()
    assert error.value.details.code == "repeated_fold_response_integrity_violation"
