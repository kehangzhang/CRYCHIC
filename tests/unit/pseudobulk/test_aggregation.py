from __future__ import annotations

from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from crychic.data import InputSchema, validate_anndata
from crychic.pseudobulk import (
    ExploratoryAggregate,
    MissingnessReason,
    PseudobulkDataset,
    aggregate_pseudobulk,
)
from crychic.pseudobulk import aggregation as aggregation_module

COUNTS = np.array(
    [
        [1, 0],
        [3, 2],
        [0, 4],
        [2, 2],
        [0, 0],
    ],
    dtype=np.int64,
)


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s1", "s2", "s2"],
            "subject_id": ["p1", "p1", "p1", "p2", "p2"],
            "cell_type": ["A", "A", "B", "A", "A"],
            "condition": ["control", "control", "control", "treated", "treated"],
        },
        index=["c1", "c2", "c3", "c4", "c5"],
    )


def _count_adata(matrix: object = COUNTS) -> ad.AnnData:
    adata = ad.AnnData(
        X=np.zeros(COUNTS.shape),
        obs=_obs(),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    adata.layers["counts"] = matrix
    return adata


def _validated_counts(matrix: object = COUNTS):
    return validate_anndata(
        _count_adata(matrix), InputSchema(context_keys=("condition",))
    )


def _unit(
    result: PseudobulkDataset | ExploratoryAggregate, sample: str, cell_type: str
):
    rows = result.unit_metadata.loc[
        (result.unit_metadata["sample_id"] == sample)
        & (result.unit_metadata["cell_type"] == cell_type)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def _matrix_row(matrix: sparse.csr_matrix, unit: pd.Series) -> np.ndarray:
    return matrix.getrow(int(unit["matrix_row"])).toarray().ravel()


def test_hand_calculated_count_aggregation_and_qc() -> None:
    result = aggregate_pseudobulk(_validated_counts(), min_cells=2)
    assert isinstance(result, PseudobulkDataset)
    assert result.counts.shape == (3, 2)

    s1_a = _unit(result, "s1", "A")
    np.testing.assert_array_equal(_matrix_row(result.counts, s1_a), [4, 2])
    np.testing.assert_allclose(_matrix_row(result.detection_fraction, s1_a), [1.0, 0.5])
    assert s1_a["n_cells"] == 2
    assert s1_a["cell_proportion"] == pytest.approx(2 / 3)
    assert s1_a["library_size"] == 6
    assert s1_a["median_umi"] == pytest.approx(3.0)
    assert s1_a["state_eligible"]
    assert s1_a["missingness_reason"] == MissingnessReason.OBSERVED.value

    s1_b = _unit(result, "s1", "B")
    np.testing.assert_array_equal(_matrix_row(result.counts, s1_b), [0, 4])
    assert not s1_b["state_eligible"]
    assert s1_b["abundance_eligible"]
    assert s1_b["missingness_reason"] == MissingnessReason.LOW_CELL_COUNT.value
    assert result.state_counts(s1_b["unit_id"]) is None

    s2_a = _unit(result, "s2", "A")
    np.testing.assert_array_equal(_matrix_row(result.counts, s2_a), [2, 2])
    assert s2_a["cell_proportion"] == pytest.approx(1.0)
    assert s2_a["median_umi"] == pytest.approx(2.0)


def test_zero_cell_unit_has_no_synthetic_state_row() -> None:
    result = aggregate_pseudobulk(_validated_counts(), min_cells=2)
    s2_b = _unit(result, "s2", "B")

    assert s2_b["n_cells"] == 0
    assert s2_b["cell_proportion"] == 0
    assert pd.isna(s2_b["matrix_row"])
    assert not s2_b["state_eligible"]
    assert s2_b["abundance_eligible"]
    assert s2_b["missingness_reason"] == MissingnessReason.SAMPLING_ZERO.value
    assert result.state_counts(s2_b["unit_id"]) is None
    assert result.counts.shape[0] == 3
    assert len(result.unit_metadata) == 4


def test_explicit_structural_absence_and_qc_failure_remain_distinct() -> None:
    confirmed = aggregate_pseudobulk(
        _validated_counts(),
        min_cells=2,
        missingness={("s2", "B"): MissingnessReason.CONFIRMED_ABSENCE},
    )
    confirmed_row = _unit(confirmed, "s2", "B")
    assert confirmed_row["missingness_reason"] == "confirmed_absence"
    assert confirmed_row["abundance_eligible"]
    assert pd.isna(confirmed_row["matrix_row"])

    failed = aggregate_pseudobulk(
        _validated_counts(),
        min_cells=2,
        missingness={("s2", "B"): MissingnessReason.QC_FAILURE},
    )
    failed_row = _unit(failed, "s2", "B")
    assert failed_row["missingness_reason"] == "qc_failure"
    assert not failed_row["abundance_eligible"]


@pytest.mark.parametrize(
    "matrix",
    [COUNTS.copy(), sparse.csr_matrix(COUNTS), sparse.csc_matrix(COUNTS)],
)
def test_dense_csr_and_csc_paths_match(matrix: object) -> None:
    reference = aggregate_pseudobulk(_validated_counts(COUNTS), min_cells=2)
    observed = aggregate_pseudobulk(_validated_counts(matrix), min_cells=2)

    np.testing.assert_array_equal(observed.counts.toarray(), reference.counts.toarray())
    np.testing.assert_allclose(
        observed.detection_fraction.toarray(), reference.detection_fraction.toarray()
    )
    pd.testing.assert_frame_equal(observed.unit_metadata, reference.unit_metadata)


@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
def test_group_indicator_summaries_match_scalar_reference(matrix_kind: str) -> None:
    rng = np.random.default_rng(20260713)
    dense = rng.integers(0, 6, size=(41, 23), dtype=np.int64)
    dense[rng.random(dense.shape) < 0.72] = 0
    codes = np.repeat(np.arange(7, dtype=np.int64), [3, 5, 4, 8, 7, 6, 8])
    order = rng.permutation(len(codes))
    dense = dense[order]
    codes = codes[order]
    if matrix_kind == "dense":
        matrix: object = dense
    else:
        encoded = sparse.csr_matrix(dense)
        # Explicit sparse zeroes must not count as detected expression.
        encoded.data[::11] = 0
        encoded.eliminate_zeros()
        encoded.data = encoded.data.astype(np.int64, copy=False)
        dense = encoded.toarray()
        matrix = encoded if matrix_kind == "csr" else encoded.tocsc()

    totals, detected, medians, sizes = aggregation_module._group_summaries(
        matrix, codes, n_groups=7
    )
    expected_totals = np.vstack(
        [dense[codes == code].sum(axis=0) for code in range(7)]
    )
    expected_detected = np.vstack(
        [np.greater(dense[codes == code], 0).sum(axis=0) for code in range(7)]
    )
    expected_medians = np.array(
        [np.median(dense[codes == code].sum(axis=1)) for code in range(7)]
    )

    np.testing.assert_array_equal(totals.toarray(), expected_totals)
    np.testing.assert_array_equal(detected.toarray(), expected_detected)
    np.testing.assert_array_equal(medians, expected_medians)
    np.testing.assert_array_equal(sizes, np.bincount(codes, minlength=7))


def test_sparse_aggregation_does_not_slice_input_rows_per_group() -> None:
    class NoRowSliceCsr(sparse.csr_matrix):
        def __getitem__(self, key: object) -> object:
            raise AssertionError(f"unexpected sparse row slice: {key!r}")

    validated = _validated_counts(sparse.csr_matrix(COUNTS))
    guarded = replace(validated, matrix=NoRowSliceCsr(validated.matrix))

    result = aggregate_pseudobulk(guarded, min_cells=2)

    assert isinstance(result, PseudobulkDataset)
    np.testing.assert_array_equal(
        _matrix_row(result.counts, _unit(result, "s1", "A")), [4, 2]
    )
    np.testing.assert_array_equal(
        _matrix_row(result.counts, _unit(result, "s1", "B")), [0, 4]
    )
    np.testing.assert_array_equal(
        _matrix_row(result.counts, _unit(result, "s2", "A")), [2, 2]
    )


def test_aggregation_is_invariant_to_cell_row_order() -> None:
    adata = _count_adata()
    reordered = adata[[4, 2, 0, 3, 1], :].copy()
    schema = InputSchema(context_keys=("condition",))

    first = aggregate_pseudobulk(validate_anndata(adata, schema), min_cells=2)
    second = aggregate_pseudobulk(validate_anndata(reordered, schema), min_cells=2)

    assert first.matrix_unit_ids == second.matrix_unit_ids
    np.testing.assert_array_equal(first.counts.toarray(), second.counts.toarray())
    np.testing.assert_allclose(
        first.detection_fraction.toarray(), second.detection_fraction.toarray()
    )
    pd.testing.assert_frame_equal(first.unit_metadata, second.unit_metadata)


def test_normalized_only_path_computes_mean_not_sum() -> None:
    normalized = COUNTS.astype(float)
    adata = ad.AnnData(
        X=normalized,
        obs=_obs(),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    schema = InputSchema(
        context_keys=("condition",),
        expression_source="published log-normalized matrix",
        expression_transform="log1p_normalized",
    )
    result = aggregate_pseudobulk(validate_anndata(adata, schema), min_cells=2)

    assert isinstance(result, ExploratoryAggregate)
    s1_a = _unit(result, "s1", "A")
    np.testing.assert_allclose(_matrix_row(result.mean_expression, s1_a), [2.0, 1.0])
    assert result.detection_fraction is None
    assert result.reason_codes == ("normalized_only",)
    assert pd.isna(s1_a["library_size"])
    assert pd.isna(s1_a["median_umi"])


def test_normalized_detection_is_computed_only_when_declared() -> None:
    adata = ad.AnnData(
        X=sparse.csr_matrix(COUNTS.astype(float)),
        obs=_obs(),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    schema = InputSchema(
        context_keys=("condition",),
        expression_source="linear normalized counts",
        expression_transform="linear_normalized",
        normalized_zero_is_nondetection=True,
    )
    result = aggregate_pseudobulk(validate_anndata(adata, schema), min_cells=2)

    assert isinstance(result, ExploratoryAggregate)
    assert result.detection_fraction is not None
    s1_a = _unit(result, "s1", "A")
    np.testing.assert_allclose(_matrix_row(result.detection_fraction, s1_a), [1.0, 0.5])


def test_min_cell_threshold_boundary_is_inclusive() -> None:
    at_boundary = aggregate_pseudobulk(_validated_counts(), min_cells=2)
    above_boundary = aggregate_pseudobulk(_validated_counts(), min_cells=3)

    assert _unit(at_boundary, "s1", "A")["state_eligible"]
    assert not _unit(above_boundary, "s1", "A")["state_eligible"]


def test_missingness_cannot_be_attached_to_observed_or_unknown_unit() -> None:
    with pytest.raises(ValueError, match="observed unit"):
        aggregate_pseudobulk(
            _validated_counts(), missingness={("s1", "A"): "confirmed_absence"}
        )
    with pytest.raises(ValueError, match="unknown sample/cell-type"):
        aggregate_pseudobulk(
            _validated_counts(), missingness={("not-a-sample", "B"): "sampling_zero"}
        )


def test_user_context_field_named_context_does_not_overwrite_canonical_tuple() -> None:
    adata = _count_adata()
    adata.obs = adata.obs.rename(columns={"condition": "context"})
    validated = validate_anndata(adata, InputSchema(context_keys=("context",)))

    result = aggregate_pseudobulk(validated, min_cells=2)

    observed = set(result.unit_metadata["context"])
    assert observed == {
        (("context", "control"),),
        (("context", "treated"),),
    }
