from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from crychic.core import ContractError
from crychic.design import (
    FrozenDesignEncoder,
    apply_frozen_design_encoder,
    balanced_contrast,
    fit_frozen_design_encoder,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.response import (
    FoldGeneResponseApplication,
    FoldGeneResponseArtifact,
    apply_fold_gene_response,
    fit_fold_gene_response,
)


def _sample_metadata(
    subjects: tuple[str, ...],
    *,
    prefix: str = "",
    paired: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, subject in enumerate(subjects):
        contexts = (
            ("ctrl", "stim") if paired else (("ctrl",) if index % 2 == 0 else ("stim",))
        )
        for context in contexts:
            rows.append(
                {
                    "sample_id": f"{prefix}{subject}:{context}",
                    "subject_id": f"{prefix}{subject}",
                    "condition": context,
                    "batch": "b" if index % 2 else "a",
                }
            )
    return pd.DataFrame(rows)


def _encoder(
    metadata: pd.DataFrame, *, formula: str = "~ condition"
) -> FrozenDesignEncoder:
    return fit_frozen_design_encoder(
        metadata,
        contrast=balanced_contrast(("stim",), ("ctrl",), name="stim_vs_ctrl"),
        context_keys=("condition",),
        covariates=("batch",) if "batch" in formula else (),
        formula=formula,
    )


def _aggregate(
    metadata: pd.DataFrame,
    counts_by_sample: dict[str, tuple[int, int] | list[tuple[int, int]]],
    *,
    reverse_metadata: bool = False,
    missing_samples: frozenset[str] = frozenset(),
) -> PseudobulkDataset:
    matrix_rows: list[tuple[int, int]] = []
    matrix_ids: list[str] = []
    unit_rows: list[dict[str, object]] = []
    for _, sample in metadata.iterrows():
        sample_id = str(sample["sample_id"])
        declared = counts_by_sample[sample_id]
        technical = declared if isinstance(declared, list) else [declared]
        for technical_index, counts in enumerate(technical):
            unit_id = f"unit:{sample_id}:{technical_index}"
            is_missing = sample_id in missing_samples
            matrix_row: int | pd._libs.missing.NAType
            if is_missing:
                matrix_row = pd.NA
            else:
                matrix_row = len(matrix_rows)
                matrix_rows.append(counts)
                matrix_ids.append(unit_id)
            unit_rows.append(
                {
                    "unit_id": unit_id,
                    "sample_id": sample_id,
                    "subject_id": str(sample["subject_id"]),
                    "cell_type": "R",
                    "context": (("condition", sample["condition"]),),
                    "matrix_row": matrix_row,
                    "n_cells": 0 if is_missing else 20,
                    "cell_proportion": 1.0,
                    "state_eligible": not is_missing,
                    "abundance_eligible": True,
                    "missingness_reason": (
                        "sampling_zero" if is_missing else "observed"
                    ),
                }
            )
    units = pd.DataFrame(unit_rows)
    if reverse_metadata:
        units = units.iloc[::-1].reset_index(drop=True)
    count_matrix = sparse.csr_matrix(np.asarray(matrix_rows, dtype=np.int64))
    return PseudobulkDataset(
        counts=count_matrix,
        detection_fraction=sparse.csr_matrix(np.asarray(matrix_rows, dtype=float) > 0),
        unit_metadata=units,
        feature_ids=("G1", "G2"),
        matrix_unit_ids=tuple(matrix_ids),
        source_location="synthetic",
    )


def _counts(metadata: pd.DataFrame, *, offset: int = 0) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for index, row in metadata.reset_index(drop=True).iterrows():
        base = 10 + index + offset
        first = base + (10 if row["condition"] == "stim" else 0)
        result[str(row["sample_id"])] = (first, 100 - first)
    return result


def _log_cpm(counts: tuple[int, int]) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    return np.log1p(values / values.sum() * 1_000_000.0)


def test_paired_fold_response_matches_hand_subject_differences() -> None:
    metadata = _sample_metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    counts = _counts(metadata)

    artifact = fit_fold_gene_response(
        _aggregate(metadata, counts),
        encoder,
        receiver="R",
        fold_id="fold-1",
        training_input_digest="training-input",
    )

    differences = []
    for subject in ("p1", "p2", "p3", "p4"):
        differences.append(
            _log_cpm(counts[f"{subject}:stim"]) - _log_cpm(counts[f"{subject}:ctrl"])
        )
    difference_matrix = np.vstack(differences)
    np.testing.assert_allclose(artifact.effect, difference_matrix.mean(axis=0))
    np.testing.assert_allclose(
        artifact.standard_error,
        difference_matrix.std(axis=0, ddof=1) / np.sqrt(4),
    )
    np.testing.assert_allclose(artifact.raw_precision, 1.0 / artifact.standard_error**2)
    assert artifact.status == "ok"
    assert artifact.method == "frozen_formula_paired_subject_effects_v1"
    assert artifact.residual_df == 3
    assert artifact.to_dict()["residual_df"] == 3
    assert artifact.min_subjects_per_context == 2
    assert artifact.encoder_id == encoder.encoder_id
    assert artifact.training_subject_ids == encoder.training_subject_ids
    artifact.require_compatible(encoder)


def test_formula_response_uses_exact_sample_keyed_frozen_design() -> None:
    metadata = _sample_metadata(tuple(f"p{index}" for index in range(8)), paired=False)
    metadata["batch"] = ["a", "a", "b", "b", "a", "a", "b", "b"]
    encoder = _encoder(metadata, formula="~ batch + condition")
    counts = _counts(metadata)
    artifact = fit_fold_gene_response(
        _aggregate(metadata, counts),
        encoder,
        receiver="R",
        fold_id="fold-formula",
        training_input_digest="input-formula",
    )

    response_by_sample = {
        sample_id: artifact.sample_values[index]
        for index, sample_id in enumerate(artifact.sample_ids)
    }
    response = np.vstack(
        [response_by_sample[sample] for sample in encoder.training_sample_ids]
    )
    design = np.column_stack(
        (encoder.training_nuisance_matrix, encoder.training_context_regressor)
    )
    expected = np.linalg.pinv(design) @ response

    np.testing.assert_allclose(artifact.effect, expected[-1])
    assert artifact.status == "ok"
    assert artifact.method == "frozen_formula_independent_subjects_v1"
    assert artifact.residual_df == len(design) - np.linalg.matrix_rank(design)


def test_technical_receiver_units_are_averaged_before_model_fit() -> None:
    metadata = _sample_metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    counts: dict[str, tuple[int, int] | list[tuple[int, int]]] = _counts(metadata)
    counts["p1:ctrl"] = [(10, 90), (30, 70)]

    artifact = fit_fold_gene_response(
        _aggregate(metadata, counts),
        encoder,
        receiver="R",
        fold_id="fold-technical",
        training_input_digest="input-technical",
    )

    position = artifact.sample_ids.index("p1:ctrl")
    np.testing.assert_allclose(
        artifact.sample_values[position],
        (_log_cpm((10, 90)) + _log_cpm((30, 70))) / 2.0,
    )
    assert artifact.n_model_samples == len(metadata)
    assert len(artifact.sample_ids) == len(metadata)


def test_fit_identity_is_row_order_stable_and_ignores_fold_external_poison() -> None:
    training = _sample_metadata(("p1", "p2", "p3", "p4"))
    heldout = _sample_metadata(("q1",), prefix="heldout-")
    combined = pd.concat([training, heldout], ignore_index=True)
    encoder = _encoder(training)
    combined_counts = _counts(combined)
    first = fit_fold_gene_response(
        _aggregate(combined, combined_counts),
        encoder,
        receiver="R",
        fold_id="fold-stable",
        training_input_digest="input-stable",
    )
    poisoned = dict(combined_counts)
    poisoned["heldout-q1:ctrl"] = (99, 1)
    poisoned["heldout-q1:stim"] = (1, 99)
    second = fit_fold_gene_response(
        _aggregate(combined, poisoned, reverse_metadata=True),
        encoder,
        receiver="R",
        fold_id="fold-stable",
        training_input_digest="input-stable",
    )

    assert second.artifact_id == first.artifact_id
    np.testing.assert_array_equal(second.sample_values, first.sample_values)


def test_non_ascii_sample_ids_keep_the_frozen_canonical_order() -> None:
    metadata = _sample_metadata(("a", "e", "x", "\u00e9"))
    encoder = _encoder(metadata)

    artifact = fit_fold_gene_response(
        _aggregate(metadata, _counts(metadata), reverse_metadata=True),
        encoder,
        receiver="R",
        fold_id="fold-unicode",
        training_input_digest="input-unicode",
    )

    assert artifact.sample_ids == encoder.training_sample_ids
    assert artifact.to_dict()["artifact_id"] == artifact.artifact_id


def test_training_lineage_and_numeric_poison_change_artifact_identity() -> None:
    metadata = _sample_metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    counts = _counts(metadata)
    first = fit_fold_gene_response(
        _aggregate(metadata, counts),
        encoder,
        receiver="R",
        fold_id="fold-id",
        training_input_digest="input-a",
    )
    changed_counts = dict(counts)
    changed_counts["p1:ctrl"] = (40, 60)
    numeric_poison = fit_fold_gene_response(
        _aggregate(metadata, changed_counts),
        encoder,
        receiver="R",
        fold_id="fold-id",
        training_input_digest="input-a",
    )
    lineage_poison = fit_fold_gene_response(
        _aggregate(metadata, counts),
        encoder,
        receiver="R",
        fold_id="fold-id",
        training_input_digest="input-b",
    )

    assert numeric_poison.artifact_id != first.artifact_id
    assert lineage_poison.artifact_id != first.artifact_id


def test_minimum_subject_support_is_frozen_in_response_identity() -> None:
    metadata = _sample_metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    aggregate = _aggregate(metadata, _counts(metadata))
    support_two = fit_fold_gene_response(
        aggregate,
        encoder,
        receiver="R",
        fold_id="fold-support",
        training_input_digest="input-support",
        min_subjects_per_context=2,
    )
    support_three = fit_fold_gene_response(
        aggregate,
        encoder,
        receiver="R",
        fold_id="fold-support",
        training_input_digest="input-support",
        min_subjects_per_context=3,
    )
    unavailable = fit_fold_gene_response(
        aggregate,
        encoder,
        receiver="R",
        fold_id="fold-support",
        training_input_digest="input-support",
        min_subjects_per_context=5,
    )

    assert support_two.status == support_three.status == "ok"
    np.testing.assert_array_equal(support_two.effect, support_three.effect)
    assert support_two.response_input_digest != support_three.response_input_digest
    assert support_two.artifact_id != support_three.artifact_id
    assert support_three.to_dict()["min_subjects_per_context"] == 3
    assert unavailable.status == "not_estimable"
    assert unavailable.reason_code == "insufficient_complete_pair_support"
    assert unavailable.min_subjects_per_context == 5

    object.__setattr__(support_three, "min_subjects_per_context", 1)
    with pytest.raises(ContractError, match="integrity"):
        support_three.to_dict()


def test_training_artifact_is_producer_owned_immutable_and_self_validating() -> None:
    metadata = _sample_metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    artifact = fit_fold_gene_response(
        _aggregate(metadata, _counts(metadata)),
        encoder,
        receiver="R",
        fold_id="fold-contract",
        training_input_digest="input-contract",
    )

    with pytest.raises(TypeError, match="producer-owned"):
        FoldGeneResponseArtifact()
    for values in (
        artifact.sample_values,
        artifact.effect,
        artifact.standard_error,
        artifact.raw_precision,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)
    object.__setattr__(artifact, "effect_digest", "poison")
    with pytest.raises(ContractError, match="integrity"):
        artifact.to_dict()


def test_training_artifact_binds_residual_df_to_identity() -> None:
    metadata = _sample_metadata(("p1", "p2", "p3", "p4"))
    artifact = fit_fold_gene_response(
        _aggregate(metadata, _counts(metadata)),
        _encoder(metadata),
        receiver="R",
        fold_id="fold-df-contract",
        training_input_digest="input-df-contract",
    )

    object.__setattr__(artifact, "residual_df", 2)
    with pytest.raises(ContractError, match="integrity"):
        artifact.to_dict()


def test_conflicting_aggregate_sample_mapping_is_rejected() -> None:
    metadata = _sample_metadata(("p1", "p2", "p3", "p4"))
    encoder = _encoder(metadata)
    aggregate = _aggregate(metadata, _counts(metadata))
    poisoned_metadata = aggregate.unit_metadata.copy()
    poisoned_metadata.loc[
        poisoned_metadata["sample_id"].eq("p1:ctrl"), "subject_id"
    ] = "other-subject"
    poisoned = PseudobulkDataset(
        counts=aggregate.counts,
        detection_fraction=aggregate.detection_fraction,
        unit_metadata=poisoned_metadata,
        feature_ids=aggregate.feature_ids,
        matrix_unit_ids=aggregate.matrix_unit_ids,
        source_location=aggregate.source_location,
    )

    with pytest.raises(ValueError, match="frozen subject/context mapping"):
        fit_fold_gene_response(
            poisoned,
            encoder,
            receiver="R",
            fold_id="fold-conflict",
            training_input_digest="input-conflict",
        )


def test_heldout_application_is_parent_bound_and_does_not_refit_statistics() -> None:
    training = _sample_metadata(("p1", "p2", "p3", "p4"))
    heldout = _sample_metadata(("q1", "q2"), prefix="heldout-")
    encoder = _encoder(training)
    response = fit_fold_gene_response(
        _aggregate(training, _counts(training)),
        encoder,
        receiver="R",
        fold_id="fold-apply",
        training_input_digest="input-apply",
    )
    design_application = apply_frozen_design_encoder(encoder, heldout)
    first = apply_fold_gene_response(
        _aggregate(heldout, _counts(heldout)), response, design_application
    )
    poison_counts = _counts(heldout, offset=20)
    second = apply_fold_gene_response(
        _aggregate(heldout, poison_counts), response, design_application
    )

    assert first.status == second.status == "ok"
    assert (
        first.training_response_id
        == second.training_response_id
        == response.artifact_id
    )
    assert first.design_application_id == design_application.application_id
    assert first.application_id != second.application_id
    assert response.effect_digest == response.to_dict()["effect_digest"]
    first.require_compatible(response, design_application)
    with pytest.raises(TypeError, match="producer-owned"):
        FoldGeneResponseApplication()


def test_heldout_missing_receiver_sample_is_explicit_not_estimable() -> None:
    training = _sample_metadata(("p1", "p2", "p3", "p4"))
    heldout = _sample_metadata(("q1", "q2"), prefix="heldout-")
    encoder = _encoder(training)
    response = fit_fold_gene_response(
        _aggregate(training, _counts(training)),
        encoder,
        receiver="R",
        fold_id="fold-missing",
        training_input_digest="input-missing",
    )
    design_application = apply_frozen_design_encoder(encoder, heldout)
    missing_sample = "heldout-q1:ctrl"
    application = apply_fold_gene_response(
        _aggregate(
            heldout,
            _counts(heldout),
            missing_samples=frozenset({missing_sample}),
        ),
        response,
        design_application,
    )

    assert application.status == "not_estimable"
    assert application.reason_code == "incomplete_heldout_receiver_response"
    assert application.missing_sample_ids == (missing_sample,)
    position = application.sample_ids.index(missing_sample)
    assert np.isnan(application.sample_values[position]).all()
    application.require_compatible(response, design_application)


def test_heldout_application_rejects_wrong_design_parent_and_feature_order() -> None:
    training = _sample_metadata(("p1", "p2", "p3", "p4"))
    heldout = _sample_metadata(("q1", "q2"), prefix="heldout-")
    encoder = _encoder(training)
    response = fit_fold_gene_response(
        _aggregate(training, _counts(training)),
        encoder,
        receiver="R",
        fold_id="fold-parent",
        training_input_digest="input-parent",
    )
    other_encoder = fit_frozen_design_encoder(
        training,
        contrast=balanced_contrast(("ctrl",), ("stim",), name="ctrl_vs_stim"),
        context_keys=("condition",),
        formula="~ condition",
    )
    wrong_design = apply_frozen_design_encoder(other_encoder, heldout)
    with pytest.raises(ContractError, match="does not match"):
        apply_fold_gene_response(
            _aggregate(heldout, _counts(heldout)), response, wrong_design
        )

    aggregate = _aggregate(heldout, _counts(heldout))
    reversed_features = PseudobulkDataset(
        counts=aggregate.counts[:, ::-1].tocsr(),
        detection_fraction=aggregate.detection_fraction[:, ::-1].tocsr(),
        unit_metadata=aggregate.unit_metadata,
        feature_ids=aggregate.feature_ids[::-1],
        matrix_unit_ids=aggregate.matrix_unit_ids,
        source_location=aggregate.source_location,
    )
    with pytest.raises(ContractError, match="feature order"):
        apply_fold_gene_response(
            reversed_features,
            response,
            apply_frozen_design_encoder(encoder, heldout),
        )
