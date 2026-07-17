from __future__ import annotations

from itertools import product
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from benchmarks.openproblems.common import (
    load_truth,
    materialize_complete_pairs,
    official_metrics,
    random_events_predictions,
    raw_official_metrics,
)


def test_load_truth_preserves_released_row_and_category_order(tmp_path: Path) -> None:
    labels = pd.Categorical(
        ["Target", "Source", "Other"],
        categories=["Other", "Target", "Source"],
        ordered=True,
    )
    data = ad.AnnData(
        X=np.ones((3, 1), dtype=np.int64),
        obs=pd.DataFrame({"label": labels}, index=["c1", "c2", "c3"]),
        var=pd.DataFrame(index=["Gene"]),
    )
    expected = pd.DataFrame(
        {
            "source": ["Source", "Other", "Target"],
            "target": ["Target", "Source", "Other"],
            "response": [1, 0, 0],
        }
    )
    data.uns["ccc_target"] = expected
    path = tmp_path / "input.h5ad"
    data.write_h5ad(path)

    truth, observed_labels = load_truth(path)

    pd.testing.assert_frame_equal(truth, expected)
    assert observed_labels == ("Other", "Target", "Source")


def test_random_events_reproduces_upstream_rng_consumption() -> None:
    labels = ("B", "A", "C")
    observed = random_events_predictions(
        labels=labels,
        resource_rows=7,
        resource_id="resource",
        n_events=8,
        seed=11,
    )

    rng = np.random.default_rng(seed=11)
    rng.choice(7, 8)
    expected_sparse = pd.DataFrame(
        {
            "source": rng.choice(labels, 8),
            "target": rng.choice(labels, 8),
            "score": rng.uniform(0.0, 1.0, 8),
        }
    ).drop_duplicates(["source", "target"], keep="first")
    expected = materialize_complete_pairs(
        expected_sparse,
        method_id="random_events",
        method_name="Random Events",
        method_scope="openproblems_v1_reproduced_baseline",
        resource_id="resource",
        aggregation="first",
        labels=labels,
    )
    pd.testing.assert_frame_equal(observed, expected)


def test_official_metrics_report_raw_and_baseline_scaled_values() -> None:
    labels = tuple(f"C{index}" for index in range(5))
    pairs = [pair for pair in product(labels, labels) if pair[0] != pair[1]]
    truth = pd.DataFrame(pairs, columns=["source", "target"])
    truth["response"] = [int(index in {1, 5, 10, 15}) for index in range(20)]

    random_scores = truth.loc[:, ["source", "target"]].copy()
    random_scores["score"] = np.linspace(1.0, 0.0, len(random_scores))
    random_predictions = materialize_complete_pairs(
        random_scores,
        method_id="random",
        method_name="Random",
        method_scope="test",
        resource_id="resource",
        aggregation="first",
        labels=labels,
    )
    baseline_raw = raw_official_metrics(truth, random_predictions)
    baseline = official_metrics(
        truth,
        random_predictions,
        random_predictions=random_predictions,
    )
    assert baseline["openproblems_score"] == 0.0
    assert baseline["precision_recall_auc"] == 0.0
    assert baseline["odds_ratio"] == 0.0
    assert baseline["precision_recall_auc_raw"] == baseline_raw[
        "precision_recall_auc_raw"
    ]

    perfect_scores = truth.loc[:, ["source", "target"]].copy()
    perfect_scores["score"] = np.where(
        truth["response"].eq(1),
        2.0 - np.arange(len(truth)) / 100.0,
        -np.arange(len(truth), dtype=float),
    )
    perfect_predictions = materialize_complete_pairs(
        perfect_scores,
        method_id="perfect",
        method_name="Perfect",
        method_scope="test",
        resource_id="resource",
        aggregation="first",
        labels=labels,
    )
    perfect = official_metrics(
        truth,
        perfect_predictions,
        random_predictions=random_predictions,
    )
    assert perfect["precision_recall_auc_raw"] == 1.0
    assert perfect["odds_ratio_raw"] == 1.0
    assert perfect["precision_recall_auc"] == 1.0
    assert perfect["odds_ratio"] == 1.0
    assert perfect["openproblems_score"] == 1.0
