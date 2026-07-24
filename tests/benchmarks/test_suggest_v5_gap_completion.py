from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.comprehensive.complete_suggest_v5_posthoc import (
    _dtw,
    _ranking_metrics,
    _scale_aligned,
)
from benchmarks.comprehensive.run_native_patient_availability import _patient_distance
from benchmarks.comprehensive.run_scaccordion_gap_completion import (
    best_kbarycenter,
    canonical_spearman_distance,
)
from benchmarks.literature.compare_latest_single_sample import _compare


def test_tensor_extended_helpers_are_deterministic() -> None:
    truth = np.asarray([0.0, 1.0, 2.0, 1.0])
    predicted = truth * 7.0
    assert np.allclose(_scale_aligned(predicted, truth), truth)
    assert _dtw(truth, truth) == 0.0
    metrics = _ranking_metrics(
        np.asarray([1, 1, 0, 0]), np.asarray([4.0, 3.0, 2.0, 1.0]), top=2
    )
    assert metrics["auroc"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["precision_at_2"] == 1.0


def test_single_sample_comparison_requires_exact_values() -> None:
    old = pd.DataFrame({"key": ["a", "b"], "score": [0.0, 1.0], "status": ["ok", "ok"]})
    exact = _compare(
        old, old.copy(), keys=("key",), numeric=("score",), text=("status",)
    )
    assert exact["exactly_consistent"] is True
    altered = old.copy()
    altered.loc[1, "score"] = 1.0 + np.finfo(float).eps
    mismatch = _compare(
        old, altered, keys=("key",), numeric=("score",), text=("status",)
    )
    assert mismatch["exactly_consistent"] is False
    assert mismatch["numeric_mismatches"] == 1


def test_canonical_spearman_uses_event_ranks() -> None:
    pytest.importorskip("conorm")
    first = pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["B", "B", "A"],
            "ligand": ["L1", "L2", "L3"],
            "receptor": ["R1", "R2", "R3"],
            "lr_means": [1.0, 2.0, 3.0],
        }
    )
    second = first.copy()
    second["lr_means"] = [10.0, 20.0, 30.0]
    distance, events = canonical_spearman_distance(
        {"s1": first, "s2": second}, ["s1", "s2"]
    )
    assert events == 3
    assert np.allclose(distance, 0.0)


def test_kbarycenter_separates_two_simple_distributions() -> None:
    pytest.importorskip("ot")
    distributions = np.asarray(
        [[0.95, 0.90, 0.05, 0.10], [0.05, 0.10, 0.95, 0.90]], dtype=float
    )
    cost = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    distance = np.abs(distributions[0, :, None] - distributions[0, None, :])
    labels, _, _, _ = best_kbarycenter(
        distance,
        distributions,
        cost,
        k=2,
        starts=3,
        seed=11,
        max_iterations=20,
        regularization=0.01,
    )
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_native_patient_distance_uses_pairwise_complete_events(tmp_path: Path) -> None:
    paths = []
    values = {
        "s1": [
            ("A", "B", "L1", "R1", 1.0),
            ("A", "B", "L2", "R2", 2.0),
            ("A", "B", "L3", "R3", 3.0),
        ],
        "s2": [
            ("A", "B", "L1", "R1", 10.0),
            ("A", "B", "L2", "R2", 20.0),
            ("A", "B", "L3", "R3", 30.0),
        ],
    }
    for sample, rows in values.items():
        frame = pd.DataFrame(
            rows,
            columns=["sender", "receiver", "ligand", "receptor", "availability_state"],
        )
        frame["sample_id"] = sample
        path = tmp_path / f"{sample}.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
    distance, overlap, sample_ids, events = _patient_distance(paths)
    assert sample_ids == ["s1", "s2"]
    assert events == 3
    assert overlap[0, 1] == 3
    assert np.allclose(distance, 0.0)
