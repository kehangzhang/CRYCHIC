from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from benchmarks.openproblems.common import (
    OFFICIAL_V1_RANDOM_AUPRC,
    OFFICIAL_V1_RANDOM_ODDS_RATIO,
    materialize_complete_pairs,
    write_predictions,
)
from benchmarks.openproblems.evaluate_source_target import run


def test_evaluator_combines_local_raw_scaled_and_official_reference(
    tmp_path: Path,
) -> None:
    labels = tuple(f"C{i}" for i in range(5))
    truth = pd.DataFrame(
        [pair for pair in product(labels, labels) if pair[0] != pair[1]],
        columns=["source", "target"],
    )
    truth["response"] = [int(i in {1, 5, 10, 15}) for i in range(len(truth))]
    data = ad.AnnData(
        X=np.ones((5, 1), dtype=np.int64),
        obs=pd.DataFrame(
            {"label": pd.Categorical(labels, categories=labels)}, index=labels
        ),
        var=pd.DataFrame(index=["Gene"]),
    )
    data.uns["ccc_target"] = truth
    input_path = tmp_path / "input.h5ad"
    data.write_h5ad(input_path)

    scores = truth.loc[:, ["source", "target"]].copy()
    scores["score"] = np.where(
        truth["response"].eq(1),
        2.0 - np.arange(len(truth)) / 100.0,
        -np.arange(len(truth), dtype=float),
    )
    prediction = materialize_complete_pairs(
        scores,
        method_id="local_perfect",
        method_name="Local perfect fixture",
        method_scope="test",
        resource_id="test",
        aggregation="max",
        labels=labels,
    )
    prediction_dir = tmp_path / "predictions"
    write_predictions(prediction_dir, prediction)

    official_path = tmp_path / "official.json"
    official_path.write_text(
        json.dumps(
            [
                {
                    "dataset_id": "mouse_brain_atlas",
                    "method_id": "random_events",
                    "metric_values": {
                        "auprc": OFFICIAL_V1_RANDOM_AUPRC,
                        "odds_ratio": OFFICIAL_V1_RANDOM_ODDS_RATIO,
                    },
                    "scaled_scores": {"auprc": 0.0, "odds_ratio": 0.0},
                    "mean_score": 0.0,
                    "commit_sha": "official",
                    "code_version": "v1",
                },
                {
                    "dataset_id": "mouse_brain_atlas",
                    "method_id": "true_events",
                    "metric_values": {"auprc": 1.0, "odds_ratio": 1.0},
                    "scaled_scores": {"auprc": 1.0, "odds_ratio": 1.0},
                    "mean_score": 1.0,
                    "commit_sha": "official",
                    "code_version": "v1",
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation"
    manifest = run(
        input_path,
        (prediction_dir,),
        official_path,
        output,
        overwrite=False,
    )

    assert manifest["status"] == "complete"
    assert manifest["dataset"]["truth_rows"] == 20
    local = pd.read_csv(output / "local_metrics.tsv", sep="\t")
    assert local.loc[0, "precision_recall_auc_raw"] == 1.0
    assert local.loc[0, "odds_ratio_raw"] == 1.0
    assert local.loc[0, "openproblems_score"] == 1.0
    combined = pd.read_csv(output / "benchmark_metrics.tsv", sep="\t")
    assert set(combined["method_id"]) == {
        "local_perfect",
        "random_events",
        "true_events",
    }
