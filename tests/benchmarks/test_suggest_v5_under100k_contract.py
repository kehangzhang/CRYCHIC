from __future__ import annotations

import copy

import pytest

from benchmarks.comprehensive.suggest_v5_under100k_contract import (
    load_contract,
    scope_rows,
    validate_contract,
)


def test_frozen_contract_has_exact_under100k_scope() -> None:
    payload = load_contract()
    rows = scope_rows(payload)

    assert len(rows) == 9
    assert all(row["observed_single_cells"] < 100_000 for row in rows)
    assert {row["benchmark_id"] for row in rows} >= {
        "tensor_cell2cell_planted_12context",
        "staccato_condition_batch_simulation",
        "dcst_subject_count_sweep",
        "dcst_receiver_cell_count_sweep",
        "scaccordion_pdac",
        "scaccordion_kidney_aki",
        "scaccordion_rcc",
    }


def test_dcst_scans_are_independent_and_strictly_below_limit() -> None:
    payload = load_contract()
    records = {item["benchmark_id"]: item for item in payload["benchmarks"]}

    subject = records["dcst_subject_count_sweep"]["design"]
    receiver = records["dcst_receiver_cell_count_sweep"]["design"]
    assert subject["subjects_per_condition"] == list(range(5, 50, 5))
    assert subject["excluded_subjects_per_condition"] == [50, 55]
    assert receiver["receiver_cells_in_condition_2"] == list(range(50, 501, 50))
    assert "receiver_cells_in_condition_2" not in subject
    assert "subjects_per_condition" not in receiver or isinstance(
        receiver["subjects_per_condition"], int
    )


def test_contract_rejects_equal_100k_setting() -> None:
    payload = copy.deepcopy(load_contract())
    for record in payload["benchmarks"]:
        if record["benchmark_id"] == "dcst_subject_count_sweep":
            record["observed_single_cells"] = 100_000
            break

    with pytest.raises(ValueError, match="ineligible benchmark"):
        validate_contract(payload)


def test_contract_rejects_missing_eligible_cohort() -> None:
    payload = copy.deepcopy(load_contract())
    payload["benchmarks"] = [
        item
        for item in payload["benchmarks"]
        if item["benchmark_id"] != "scaccordion_rcc"
    ]

    with pytest.raises(ValueError, match="benchmark scope drift"):
        validate_contract(payload)
