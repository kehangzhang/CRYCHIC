from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from crychic.attribution import (
    PrecisionTransformResult,
    fit_response_precision,
    winsorized_normalized_precision,
)
from crychic.core import ContractError
from crychic.design import balanced_contrast, fit_frozen_design_encoder
from crychic.pseudobulk import PseudobulkDataset
from crychic.response import FoldGeneResponseArtifact, fit_fold_gene_response


def _fit(raw: np.ndarray, **kwargs: object) -> PrecisionTransformResult:
    parameters: dict[str, object] = {
        "feature_ids": tuple(f"G{index}" for index in range(len(raw))),
        "receiver": "Receiver",
        "contrast_name": "stim_vs_ctrl",
        "fold_id": "fold-1",
    }
    parameters.update(kwargs)
    return winsorized_normalized_precision(raw, **parameters)  # type: ignore[arg-type]


def _response_artifact(
    *, training_input_digest: str = "precision-training-input"
) -> FoldGeneResponseArtifact:
    metadata_rows: list[dict[str, str]] = []
    unit_rows: list[dict[str, object]] = []
    count_rows: list[tuple[int, int]] = []
    matrix_unit_ids: list[str] = []
    for subject_index in range(4):
        subject = f"p{subject_index}"
        for condition in ("ctrl", "stim"):
            sample = f"{subject}:{condition}"
            metadata_rows.append(
                {
                    "sample_id": sample,
                    "subject_id": subject,
                    "condition": condition,
                }
            )
            unit_id = f"unit:{sample}"
            first = (
                10 + subject_index if condition == "ctrl" else 24 + 2 * subject_index
            )
            count_rows.append((first, 100 - first))
            matrix_unit_ids.append(unit_id)
            unit_rows.append(
                {
                    "unit_id": unit_id,
                    "sample_id": sample,
                    "subject_id": subject,
                    "cell_type": "Receiver",
                    "context": (("condition", condition),),
                    "matrix_row": len(count_rows) - 1,
                    "n_cells": 20,
                    "cell_proportion": 1.0,
                    "state_eligible": True,
                    "abundance_eligible": True,
                    "missingness_reason": "observed",
                }
            )
    metadata = pd.DataFrame(metadata_rows)
    encoder = fit_frozen_design_encoder(
        metadata,
        contrast=balanced_contrast(("stim",), ("ctrl",), name="stim_vs_ctrl"),
        context_keys=("condition",),
        formula="~ condition",
    )
    counts = sparse.csr_matrix(np.asarray(count_rows, dtype=np.int64))
    aggregate = PseudobulkDataset(
        counts=counts,
        detection_fraction=counts.astype(bool).astype(float),
        unit_metadata=pd.DataFrame(unit_rows),
        feature_ids=("G0", "G1"),
        matrix_unit_ids=tuple(matrix_unit_ids),
        source_location="synthetic",
    )
    return fit_fold_gene_response(
        aggregate,
        encoder,
        receiver="Receiver",
        fold_id="fold-1",
        training_input_digest=training_input_digest,
    )


def test_precision_is_winsorized_and_positive_median_normalized() -> None:
    result = _fit(
        np.asarray([1.0, 4.0, 100.0, np.nan, 0.0]),
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    np.testing.assert_allclose(result.values, [0.25, 1.0, 25.0, 0.0, 0.0])
    assert np.median(result.values[result.values > 0]) == pytest.approx(1.0)
    assert result.n_positive_features == 3
    assert result.estimable
    assert result.reason_code is None
    assert result.lineage_mode == "exploratory_unparented_v2"
    assert result.response_artifact_id is None
    assert result.training_subject_ids == ()


def test_precision_transform_records_insufficient_support() -> None:
    result = _fit(np.asarray([np.nan, 2.0, 0.0]), min_positive_features=2)

    np.testing.assert_allclose(result.values, [0.0, 1.0, 0.0])
    assert not result.estimable
    assert result.reason_code == "insufficient_response_precision_support"


def test_precision_transform_id_changes_with_fitted_bounds() -> None:
    first = _fit(np.asarray([1.0, 2.0, 100.0]))
    second = _fit(np.asarray([1.0, 2.0, 10.0]))

    assert first.precision_transform_id != second.precision_transform_id


def test_precision_quantile_types_are_canonicalized() -> None:
    raw = np.asarray([1.0, 2.0, 3.0])
    python = _fit(raw, lower_quantile=0.0, upper_quantile=1.0)
    numpy = _fit(
        raw,
        lower_quantile=np.int64(0),
        upper_quantile=np.float32(1.0),
    )

    assert numpy.precision_transform_id == python.precision_transform_id
    assert type(numpy.lower_quantile) is float
    assert type(numpy.upper_quantile) is float


def test_precision_quantiles_reject_boolean_values() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        _fit(
            np.asarray([1.0, 2.0]),
            lower_quantile=False,
            upper_quantile=True,
        )


def test_precision_preserves_exact_whitespace_bearing_identifiers() -> None:
    result = winsorized_normalized_precision(
        np.asarray([1.0, 2.0]),
        feature_ids=(" G1 ", "G2"),
        receiver=" Receiver ",
        contrast_name=" contrast ",
        fold_id=" fold ",
    )

    assert result.feature_ids == (" G1 ", "G2")
    assert result.receiver == " Receiver "


def test_precision_transform_id_binds_values_when_summary_bounds_match() -> None:
    first = _fit(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        lower_quantile=0.0,
        upper_quantile=1.0,
    )
    second = _fit(
        np.asarray([1.0, 2.5, 3.0, 3.5, 5.0]),
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert (first.lower_bound, first.upper_bound, first.normalization_median) == (
        second.lower_bound,
        second.upper_bound,
        second.normalization_median,
    )
    assert first.precision_transform_id != second.precision_transform_id
    assert first.raw_precision_digest != second.raw_precision_digest
    assert first.transformed_precision_digest != second.transformed_precision_digest


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("feature_ids", ("X0", "G1", "G2")),
        ("receiver", "OtherReceiver"),
        ("contrast_name", "other_contrast"),
        ("fold_id", "fold-2"),
    ],
)
def test_precision_transform_id_binds_scope(
    field_name: str, replacement: object
) -> None:
    raw = np.asarray([1.0, 2.0, 3.0])
    first = _fit(raw)
    second = _fit(raw, **{field_name: replacement})

    assert first.precision_transform_id != second.precision_transform_id


def test_precision_values_are_backed_by_immutable_storage() -> None:
    result = _fit(np.asarray([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        result.values.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        result.values[0] = 99.0


def test_precision_transform_detects_post_serialization_mutation() -> None:
    result = _fit(np.asarray([1.0, 2.0, 3.0]))
    object.__setattr__(result, "values", result.values.copy())
    result.values[0] = 99.0

    with pytest.raises(ContractError) as error:
        result.to_dict()

    assert error.value.details.code == "precision_transform_integrity_violation"


def test_precision_transform_is_producer_owned() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        PrecisionTransformResult()


def test_precision_transform_rejects_feature_misalignment() -> None:
    with pytest.raises(ValueError, match="align"):
        winsorized_normalized_precision(
            np.asarray([1.0, 2.0]),
            feature_ids=("G1",),
            receiver="Receiver",
            contrast_name="stim_vs_ctrl",
            fold_id="fold-1",
        )


def test_precision_transform_rejects_incompatible_consumption_scope() -> None:
    result = _fit(np.asarray([1.0, 2.0, 3.0]))

    with pytest.raises(ContractError) as error:
        result.require_compatible(
            feature_ids=("G1", "G0", "G2"),
            receiver="Receiver",
            contrast_name="stim_vs_ctrl",
            fold_id="fold-1",
        )

    assert error.value.details.code == "precision_transform_scope_mismatch"


@pytest.mark.parametrize("minimum", [True, 1.5, 0])
def test_precision_transform_rejects_invalid_minimum_support(
    minimum: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _fit(
            np.asarray([1.0, 2.0]),
            min_positive_features=minimum,
        )


def test_response_precision_binds_complete_training_parent_lineage() -> None:
    response = _response_artifact()

    result = fit_response_precision(response)

    result.require_response_compatible(response)
    assert result.lineage_mode == "fold_gene_response_parented_v1"
    assert result.response_artifact_id == response.artifact_id
    assert result.training_row_manifest_id == response.training_sample_manifest_digest
    assert result.training_subject_ids == response.training_subject_ids
    assert result.encoder_id == response.encoder_id
    assert result.feature_ids == response.feature_ids
    assert result.raw_precision_digest
    assert result.values.flags.writeable is False
    with pytest.raises(ValueError, match="WRITEABLE"):
        result.values.setflags(write=True)


def test_response_precision_same_scope_different_parent_has_distinct_identity() -> None:
    first_parent = _response_artifact(training_input_digest="first-input")
    second_parent = _response_artifact(training_input_digest="second-input")

    first = fit_response_precision(first_parent)
    second = fit_response_precision(second_parent)

    assert first.feature_ids == second.feature_ids
    assert first.receiver == second.receiver
    assert first.contrast_name == second.contrast_name
    assert first.fold_id == second.fold_id
    assert first.raw_precision_digest == second.raw_precision_digest
    assert first.response_artifact_id != second.response_artifact_id
    assert first.precision_transform_id != second.precision_transform_id


def test_response_precision_rejects_parent_mismatch() -> None:
    first_parent = _response_artifact(training_input_digest="first-input")
    second_parent = _response_artifact(training_input_digest="second-input")
    result = fit_response_precision(first_parent)

    with pytest.raises(ContractError) as error:
        result.require_response_compatible(second_parent)

    assert error.value.details.code == "precision_transform_parent_mismatch"


def test_exploratory_precision_rejects_response_parent_compatibility() -> None:
    response = _response_artifact()
    result = _fit(response.raw_precision)

    with pytest.raises(ContractError) as error:
        result.require_response_compatible(response)

    assert error.value.details.code == "precision_transform_parent_mismatch"


def test_response_precision_detects_lineage_mutation() -> None:
    response = _response_artifact()
    result = fit_response_precision(response)
    object.__setattr__(result, "response_artifact_id", "poisoned-parent")

    with pytest.raises(ContractError) as error:
        result.to_dict()

    assert error.value.details.code == "precision_transform_integrity_violation"


def test_response_precision_rejects_mutated_response_parent() -> None:
    response = _response_artifact()
    object.__setattr__(response, "artifact_id", "poisoned-response")

    with pytest.raises(ContractError) as error:
        fit_response_precision(response)

    assert error.value.details.code == "fold_gene_response_integrity_violation"
