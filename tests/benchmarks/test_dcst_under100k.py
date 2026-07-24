from __future__ import annotations

import pytest

from benchmarks.comprehensive import run_dcst_under100k as dcst


def test_subject_count_configuration_keeps_only_strictly_eligible_settings() -> None:
    dcst.configure_simulation(
        sweep="subject_count",
        subjects_per_condition=45,
        receiver_cells_in_condition_2=500,
        seed=123,
    )

    assert dcst.DATASET_ID == "dcst_subject_count_n45_b500_seed123"
    assert sum(dcst.CELL_COUNTS.values()) * dcst.N_SUBJECTS_PER_CONDITION == 90_000
    assert dcst.EXPRESSION_PROBABILITIES[("C2", "A")][0] == 0.25

    with pytest.raises(ValueError, match="strictly below"):
        dcst.configure_simulation(
            sweep="subject_count",
            subjects_per_condition=50,
            receiver_cells_in_condition_2=500,
            seed=123,
        )


def test_receiver_cell_configuration_matches_independent_paper_sweep() -> None:
    dcst.configure_simulation(
        sweep="receiver_cell_count",
        subjects_per_condition=20,
        receiver_cells_in_condition_2=50,
        seed=321,
    )

    assert dcst.DATASET_ID == "dcst_receiver_cell_count_n20_b50_seed321"
    assert dcst.CELL_COUNTS[("C2", "B")] == 50
    assert dcst.CELL_COUNTS[("C1", "B")] == 500
    assert dcst.EXPRESSION_PROBABILITIES[("C2", "A")][0] == 0.10
    assert sum(dcst.CELL_COUNTS.values()) * dcst.N_SUBJECTS_PER_CONDITION == 31_000


def test_receiver_sweep_rejects_nonpaper_subject_count() -> None:
    with pytest.raises(ValueError, match="fixes 20"):
        dcst.configure_simulation(
            sweep="receiver_cell_count",
            subjects_per_condition=15,
            receiver_cells_in_condition_2=500,
            seed=123,
        )
