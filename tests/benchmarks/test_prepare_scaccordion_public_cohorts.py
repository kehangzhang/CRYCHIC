from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.comprehensive.prepare_scaccordion_public_cohorts import (
    _write_sample_h5ads,
    deterministic_stratified_indices,
    largest_remainder_allocation,
    select_aki_samples,
)
from scipy import sparse


def test_largest_remainder_is_exact_capacity_bounded_and_deterministic() -> None:
    counts = pd.Series({("a", "x"): 10, ("a", "y"): 20, ("b", "x"): 30})
    first = largest_remainder_allocation(
        counts, 41, minimum_per_nonempty_stratum=2
    )
    second = largest_remainder_allocation(
        counts, 41, minimum_per_nonempty_stratum=2
    )

    assert first.equals(second)
    assert first.sum() == 41
    assert (first >= 2).all()
    assert (first <= counts).all()


def test_aki_selection_keeps_disease_and_largest_controls() -> None:
    obs = pd.DataFrame(
        {
            "sample": ["d1"] * 2
            + ["d2"] * 3
            + ["c1"] * 2
            + ["c2"] * 4
            + ["c3"] * 3,
            "label": ["AKI"] * 5 + ["LivingDonor"] * 9,
        }
    )
    selected, metadata = select_aki_samples(
        obs,
        sample_column="sample",
        label_column="label",
        control_label="LivingDonor",
        control_samples=2,
    )

    assert set(selected) == {"d1", "d2", "c2", "c3"}
    assert len(metadata) == 4


def test_deterministic_stratified_indices_are_exact() -> None:
    obs = pd.DataFrame(
        {
            "sample": ["a"] * 10 + ["b"] * 10,
            "cell_type": ["x"] * 5 + ["y"] * 5 + ["x"] * 4 + ["y"] * 6,
        }
    )
    first, strata = deterministic_stratified_indices(
        obs,
        sample_column="sample",
        cell_type_column="cell_type",
        selected_samples=("a", "b"),
        target_cells=13,
        seed=7,
        minimum_per_stratum=1,
    )
    second, _ = deterministic_stratified_indices(
        obs,
        sample_column="sample",
        cell_type_column="cell_type",
        selected_samples=("a", "b"),
        target_cells=13,
        seed=7,
        minimum_per_stratum=1,
    )

    assert np.array_equal(first, second)
    assert len(first) == 13
    assert strata["selected_cells"].sum() == 13
    assert (strata["selected_cells"] >= 1).all()


def test_sample_writer_does_not_copy_full_anndata_payload(tmp_path) -> None:
    class MatrixOnlySource:
        X = sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3))
        obs = pd.DataFrame(
            {
                "sample": ["s1", "s1", "s2", "s2"],
                "label": ["a", "a", "b", "b"],
                "fine": ["x", "y", "x", "y"],
            },
            index=["c1", "c2", "c3", "c4"],
        )
        var = pd.DataFrame(index=["g1", "g2", "g3"])
        obs_names = obs.index
        var_names = var.index

        def __getitem__(self, key):
            raise AssertionError("full AnnData slicing copies unrelated payloads")

    metadata = _write_sample_h5ads(
        MatrixOnlySource(),
        positions=np.arange(4),
        sample_column="sample",
        label_column="label",
        annotation_columns={"cell_type_fine": "fine"},
        symbol_column=None,
        sample_dir=tmp_path / "samples",
    )

    assert metadata["cell_count"].tolist() == [2, 2]
    assert metadata["feature_count"].tolist() == [3, 3]
