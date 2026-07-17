from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from benchmarks.literature.evaluate_cytokine_activity import (
    RANDOM_AUPRC,
    RANDOM_ODDS_RATIO,
    fisher_rank_curves,
    run,
)


def test_fisher_curve_detects_top_rank_enrichment() -> None:
    truth = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "target": ["T", "T"],
            "response": [1, 0],
        }
    )
    scores = pd.DataFrame(
        {
            "method_id": ["method"] * 200,
            "method_name": ["Method"] * 200,
            "ligand": ["L1"] * 100 + ["L2"] * 100,
            "target": ["T"] * 200,
            "source": [f"S{i}" for i in range(200)],
            "receptor": [f"R{i}" for i in range(200)],
            "score": np.arange(200, 0, -1, dtype=float),
        }
    )

    result = fisher_rank_curves(scores, truth)
    first = result.loc[result["rank_cutoff"].eq(100)].iloc[0]

    assert first["status"] == "observed"
    assert first["true_positive"] == 100
    assert first["false_positive"] == 0
    assert np.isinf(first["odds_ratio"])


def test_end_to_end_cytokine_evaluation_writes_both_protocols(
    tmp_path: Path,
) -> None:
    truth = pd.DataFrame(
        {
            "ligand": ["L1", "L1", "L2", "L2"],
            "target": ["A", "B", "A", "B"],
            "response": [1, 0, 0, 1],
        }
    )
    data = ad.AnnData(
        X=np.ones((2, 2), dtype=np.int64),
        obs=pd.DataFrame(
            {"label": pd.Categorical(["A", "B"])}, index=["a", "b"]
        ),
        var=pd.DataFrame(index=["L1", "L2"]),
    )
    data.uns["ccc_target"] = truth
    input_path = tmp_path / "input.h5ad"
    data.write_h5ad(input_path)

    crychic_path = tmp_path / "crychic.parquet"
    pd.DataFrame(
        {
            "source": ["A", "A", "B", "B"],
            "target": ["A", "B", "A", "B"],
            "ligand": ["L1", "L1", "L2", "L2"],
            "receptor": ["R"] * 4,
            "availability_state": [0.9, 0.1, 0.2, 0.8],
            "availability_ecosystem": [0.9, 0.1, 0.2, 0.8],
            "state_status": ["observed"] * 4,
            "ecosystem_status": ["observed"] * 4,
        }
    ).to_parquet(crychic_path, index=False)
    liana_path = tmp_path / "liana.parquet"
    pd.DataFrame(
        {
            "method_id": ["liana"] * 4,
            "method_name": ["LIANA"] * 4,
            "source": ["A", "A", "B", "B"],
            "target": ["A", "B", "A", "B"],
            "ligand": ["L1", "L1", "L2", "L2"],
            "receptor": ["R"] * 4,
            "score": [0.8, 0.2, 0.1, 0.7],
            "score_direction": ["higher"] * 4,
            "status": ["observed"] * 4,
        }
    ).to_parquet(liana_path, index=False)
    official_path = tmp_path / "official.json"
    official_path.write_text(
        json.dumps(
            [
                {
                    "dataset_id": "tnbc_data",
                    "method_id": "random_events",
                    "mean_score": 0.0,
                    "scaled_scores": {"auprc": 0.0, "odds_ratio": 0.0},
                    "metric_values": {
                        "auprc": RANDOM_AUPRC,
                        "odds_ratio": RANDOM_ODDS_RATIO,
                    },
                    "code_version": "v1",
                    "commit_sha": "commit",
                }
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "evaluation"
    manifest = run(
        input_path,
        crychic_path,
        liana_path,
        official_path,
        output,
        overwrite=False,
    )

    assert manifest["status"] == "complete"
    assert manifest["input"]["truth_rows"] == 4
    controlled = pd.read_csv(
        output / "controlled_unique_ligand_target_metrics.tsv", sep="\t"
    )
    assert set(controlled["method_id"]) == {
        "crychic_availability_state",
        "crychic_availability_ecosystem",
        "liana",
    }
    assert set(controlled["aggregation"]) == {"max", "sum"}
    assert (output / "paper_fisher_or_curves.tsv").is_file()
