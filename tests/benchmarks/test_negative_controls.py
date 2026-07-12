from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from benchmarks.simulation.generate import simulate_ccc
from benchmarks.simulation.run_negative_controls import (
    SCENARIOS,
    run_negative_controls,
)


def test_negative_control_runner_is_truth_scoped_and_reproducible(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic"
    summary = run_negative_controls(
        output,
        seed=20260712,
        n_subjects=6,
        mean_cells_per_sample=180,
    )

    assert summary["scenarios"] == list(SCENARIOS)
    assert summary["real_data_truth_metrics_included"] is False
    assert summary["formal_inference"] == {
        "p_values": None,
        "q_values": None,
        "fdr": None,
        "reason_code": "v0_1_inferential_disabled",
    }
    truth = summary["edge_truth_metrics"]
    assert truth["scope"] == "synthetic_edge_level_truth_only"
    assert truth["n_positive"] == 1
    assert truth["n_negative"] == len(SCENARIOS) * 3 - 1
    assert 0.0 <= truth["auroc"] <= 1.0
    assert 0.0 <= truth["average_precision"] <= 1.0

    metrics = pd.read_csv(output / "scenario_metrics.csv").set_index("scenario")
    assert set(metrics.index) == set(SCENARIOS)
    assert abs(metrics.loc["abundance_only", "main_state_relative_change"]) < 0.20
    assert abs(metrics.loc["abundance_only", "main_ecosystem_relative_change"]) > 0.05
    assert (
        metrics.loc["receiver_autonomous", "autonomous_response_effect_mean"]
        > metrics.loc["receiver_autonomous", "prior_target_response_effect_mean"]
    )
    assert (
        metrics.loc["receiver_autonomous", "main_attribution_explained_fraction"] < 0.35
    )
    assert metrics.loc["receptor_knockout", "main_stim_receptor_availability"] == 0.0
    assert metrics.loc["receptor_knockout", "main_stim_state"] == 0.0

    edge_metrics = pd.read_csv(output / "edge_metrics.csv")
    forbidden = {"p_value", "q_value", "fdr", "comm_probability"}
    assert forbidden.isdisjoint(edge_metrics.columns)
    assert edge_metrics["truth_active"].sum() == 1

    checks = pd.read_csv(output / "checks.csv")
    assert checks["passed"].all()
    raw_summary = (output / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in raw_summary
    assert json.loads(raw_summary) == summary


def test_same_seed_produces_identical_cell_level_fixture() -> None:
    first = simulate_ccc(
        "global_null", seed=8128, n_subjects=3, mean_cells_per_sample=90
    )
    second = simulate_ccc(
        "global_null", seed=8128, n_subjects=3, mean_cells_per_sample=90
    )

    pd.testing.assert_frame_equal(first.adata.obs, second.adata.obs)
    assert (first.adata.layers["counts"] != second.adata.layers["counts"]).nnz == 0
