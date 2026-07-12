from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from crychic.data import (
    InputMode,
    InputSchema,
    InputValidationError,
    validate_anndata,
)


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "subject_id": ["p1", "p1", "p2", "p2"],
            "cell_type": ["A", "B", "A", "B"],
            "condition": ["control", "control", "treated", "treated"],
            "batch": ["b1", "b1", "b2", "b2"],
        },
        index=["c1", "c2", "c3", "c4"],
    )


def _adata(matrix: object) -> ad.AnnData:
    adata = ad.AnnData(
        X=np.zeros((4, 3), dtype=float),
        obs=_obs(),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )
    adata.layers["counts"] = matrix
    return adata


def _schema(**kwargs: object) -> InputSchema:
    return InputSchema(
        context_keys=("condition",),
        covariates=("batch",),
        **kwargs,
    )


def test_dense_counts_validation_is_non_mutating() -> None:
    counts = np.array([[1, 0, 2], [0, 3, 0], [4, 1, 0], [0, 0, 5]], dtype=np.int64)
    adata = _adata(counts.copy())
    before_obs = adata.obs.copy(deep=True)
    before_var = adata.var.copy(deep=True)
    before_x = np.asarray(adata.X).copy()
    before_counts = np.asarray(adata.layers["counts"]).copy()

    validated = validate_anndata(adata, _schema())

    assert validated.mode is InputMode.COUNTS
    assert validated.report.reason_codes == ()
    assert validated.report.input_inference_eligible
    assert validated.report.detection_eligible
    assert validated.report.expression_location == "layers['counts']"
    assert validated.report.n_samples == 2
    assert validated.report.n_subjects == 2
    assert validated.feature_ids == ("G1", "G2", "G3")
    pd.testing.assert_frame_equal(adata.obs, before_obs)
    pd.testing.assert_frame_equal(adata.var, before_var)
    np.testing.assert_array_equal(np.asarray(adata.X), before_x)
    np.testing.assert_array_equal(np.asarray(adata.layers["counts"]), before_counts)


@pytest.mark.parametrize("matrix_type", [sparse.csr_matrix, sparse.csc_matrix])
def test_sparse_counts_are_validated_without_densifying(matrix_type: type) -> None:
    counts = matrix_type(
        np.array([[1, 0, 2], [0, 3, 0], [4, 1, 0], [0, 0, 5]], dtype=float)
    )
    validated = validate_anndata(_adata(counts), _schema())

    assert sparse.issparse(validated.matrix)
    assert validated.report.matrix_kind in {"csr", "csc"}


def test_backed_h5ad_is_validated_without_mutation(tmp_path: Path) -> None:
    counts = np.array([[1, 0, 2], [0, 3, 0], [4, 1, 0], [0, 0, 5]], dtype=np.int64)
    path = tmp_path / "input.h5ad"
    _adata(counts).write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    try:
        before_obs = backed.obs.copy(deep=True)
        validated = validate_anndata(backed, _schema())
        assert validated.report.n_obs == 4
        assert validated.report.expression_location == "layers['counts']"
        pd.testing.assert_frame_equal(backed.obs, before_obs)
    finally:
        backed.file.close()


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (-1.0, "negative value"),
        (1.5, "non-integer value"),
        (np.nan, "non-finite value"),
    ],
)
def test_invalid_counts_report_location_and_coordinate(
    bad_value: float, message: str
) -> None:
    counts = np.ones((4, 3), dtype=float)
    counts[2, 1] = bad_value

    with pytest.raises(InputValidationError, match=message) as error:
        validate_anndata(_adata(counts), _schema())

    assert "cell 2, gene 1" in str(error.value)
    assert "layers['counts']" in str(error.value)


def test_duplicate_gene_ids_are_rejected_by_default() -> None:
    adata = _adata(np.ones((4, 3), dtype=np.int64))
    adata.var_names = ["G1", "G1", "G2"]

    with pytest.raises(InputValidationError, match="duplicate gene identifiers"):
        validate_anndata(adata, _schema())


def test_duplicate_gene_ids_can_be_reported_for_descriptive_inspection() -> None:
    adata = _adata(np.ones((4, 3), dtype=np.int64))
    adata.var_names = ["G1", "G1", "G2"]

    validated = validate_anndata(adata, _schema(allow_duplicate_genes=True))

    assert validated.report.duplicate_genes == ("G1",)
    assert "duplicate_genes_allowed" in validated.report.warnings


@pytest.mark.parametrize("field", ["subject_id", "condition", "batch"])
def test_each_sample_must_map_to_one_subject_context_and_covariate(field: str) -> None:
    adata = _adata(np.ones((4, 3), dtype=np.int64))
    adata.obs.loc["c2", field] = "conflict"

    with pytest.raises(InputValidationError, match="map to exactly one") as error:
        validate_anndata(adata, _schema())

    assert field in str(error.value)
    assert "s1" in str(error.value)


def test_missing_and_empty_required_metadata_are_rejected() -> None:
    adata = _adata(np.ones((4, 3), dtype=np.int64))
    missing = adata.copy()
    del missing.obs["cell_type"]
    with pytest.raises(InputValidationError, match="missing declared field"):
        validate_anndata(missing, _schema())

    adata.obs.loc["c1", "sample_id"] = ""
    with pytest.raises(InputValidationError, match="empty identifiers"):
        validate_anndata(adata, _schema())


def test_sample_subject_and_cell_type_identifiers_must_be_strings() -> None:
    for field in ("sample_id", "subject_id", "cell_type"):
        adata = _adata(np.ones((4, 3), dtype=np.int64))
        adata.obs[field] = adata.obs[field].astype(object)
        adata.obs.iloc[0, adata.obs.columns.get_loc(field)] = 1

        with pytest.raises(InputValidationError, match="must be strings"):
            validate_anndata(adata, _schema())


@pytest.mark.parametrize("field", ["condition", "batch"])
def test_context_and_covariate_values_must_be_canonical_json_scalars(
    field: str,
) -> None:
    adata = _adata(np.ones((4, 3), dtype=np.int64))
    adata.obs[field] = pd.to_datetime(
        ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
    )

    with pytest.raises(InputValidationError, match="canonical JSON scalars"):
        validate_anndata(adata, _schema())


def test_numeric_context_and_covariate_values_are_supported() -> None:
    adata = _adata(np.ones((4, 3), dtype=np.int64))
    adata.obs["condition"] = [0, 0, 1, 1]
    adata.obs["batch"] = [1.5, 1.5, 2.5, 2.5]

    validated = validate_anndata(adata, _schema())

    assert set(validated.report.sample_metadata["condition"]) == {0, 1}


def test_normalized_only_requires_declared_source_and_transform() -> None:
    adata = ad.AnnData(
        X=np.array([[0.0, 1.0], [2.0, 3.0]]),
        obs=pd.DataFrame(
            {
                "sample_id": ["s1", "s1"],
                "subject_id": ["p1", "p1"],
                "cell_type": ["A", "A"],
                "condition": ["control", "control"],
            },
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    with pytest.raises(InputValidationError, match="requires explicit"):
        validate_anndata(adata, InputSchema(context_keys=("condition",)))

    schema = InputSchema(
        context_keys=("condition",),
        expression_source="published normalized matrix",
        expression_transform="log1p_normalized",
    )
    validated = validate_anndata(adata, schema)
    assert validated.mode is InputMode.NORMALIZED_ONLY
    assert validated.report.reason_codes == ("normalized_only",)
    assert not validated.report.input_inference_eligible
    assert not validated.report.detection_eligible
    assert "normalized_detection_unavailable" in validated.report.warnings


def test_normalized_detection_requires_explicit_zero_semantics() -> None:
    adata = ad.AnnData(
        X=sparse.csr_matrix([[0.0, 1.0], [2.0, 3.0]]),
        obs=pd.DataFrame(
            {
                "sample_id": ["s1", "s1"],
                "subject_id": ["p1", "p1"],
                "cell_type": ["A", "A"],
                "condition": ["control", "control"],
            },
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    schema = InputSchema(
        context_keys=("condition",),
        expression_source="library-size normalized counts",
        expression_transform="linear_normalized",
        normalized_zero_is_nondetection=True,
    )

    validated = validate_anndata(adata, schema)

    assert validated.report.detection_eligible


def test_sample_report_order_does_not_depend_on_cell_order() -> None:
    adata = _adata(np.ones((4, 3), dtype=np.int64))
    reordered = adata[[3, 1, 2, 0], :].copy()

    first = validate_anndata(adata, _schema()).report
    second = validate_anndata(reordered, _schema()).report

    pd.testing.assert_frame_equal(first.sample_metadata, second.sample_metadata)
    pd.testing.assert_frame_equal(first.cell_type_support, second.cell_type_support)
