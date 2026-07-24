from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.comprehensive.run_scaccordion_under100k import (
    METHODS,
    _sample_id,
    align_sample_universe,
    canonical_event_matrix,
    evaluate_distance,
    summarize_metrics,
)


def _table(scale: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "lr_means": [scale, scale * 2.0],
        }
    )


def test_paper_prefixed_graph_filename_maps_to_sample_id() -> None:
    from pathlib import Path

    assert _sample_id(Path("ct_Peng_PDAC_processed|N1.csv")) == "N1"


def test_canonical_event_matrix_keeps_missing_as_zero() -> None:
    tables = {"s1": _table(1.0), "s2": _table(2.0).iloc[:1].copy()}
    matrix = canonical_event_matrix(tables, ("s1", "s2"), normalize=False)

    assert matrix.shape == (2, 2)
    assert list(matrix.columns) == ["s1", "s2"]
    assert np.isfinite(matrix.to_numpy()).all()


def test_sample_universe_mismatch_is_rejected() -> None:
    metadata = pd.DataFrame(
        {"sample_id": ["s1", "s3"], "label": ["A", "B"]}
    ).set_index("sample_id", drop=False)
    with pytest.raises(ValueError, match="sample mismatch"):
        align_sample_universe({"s1": _table(1.0), "s2": _table(2.0)}, metadata)


def test_distance_evaluation_and_summary_are_label_safe() -> None:
    distance = np.asarray(
        [
            [0.0, 0.1, 2.0, 2.1],
            [0.1, 0.0, 2.1, 2.0],
            [2.0, 2.1, 0.0, 0.1],
            [2.1, 2.0, 0.1, 0.0],
        ]
    )
    rows = evaluate_distance(
        benchmark_id="scaccordion_pdac",
        method=METHODS[0],
        distance=distance,
        labels=pd.Series(["control", "control", "case", "case"]),
        cluster_counts=(2, 3),
        starts=4,
        seed=11,
    )
    summary = summarize_metrics(pd.DataFrame.from_records(rows))

    assert len(rows) == 2
    assert rows[0]["ari"] == pytest.approx(1.0)
    assert summary.iloc[0]["ari_at_true_k"] == pytest.approx(1.0)
    assert summary.iloc[0]["maximum_ari_over_k"] == pytest.approx(1.0)
