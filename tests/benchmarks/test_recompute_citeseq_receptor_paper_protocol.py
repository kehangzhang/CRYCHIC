from __future__ import annotations

import numpy as np
import pytest
from benchmarks.literature.recompute_citeseq_receptor_paper_protocol import (
    _pr_curve_yardstick_008 as pr_curve_yardstick_008,
)
from benchmarks.literature.recompute_citeseq_receptor_paper_protocol import (
    average_precision,
    paper_balanced_auprc,
    tie_aware_auroc,
    yardstick_008_pr_auc,
)


def test_yardstick_008_pr_auc_is_tie_aware_and_trapezoidal() -> None:
    labels = np.array([1, 0, 0, 1, 1])
    scores = np.array([0.9, 0.9, 0.8, 0.4, 0.3])

    recall, precision = pr_curve_yardstick_008(labels, scores)

    assert recall.tolist() == pytest.approx([0.0, 1 / 3, 1 / 3, 2 / 3, 1.0])
    assert precision.tolist() == pytest.approx([1.0, 0.5, 1 / 3, 0.5, 0.6])
    assert yardstick_008_pr_auc(labels, scores) == pytest.approx(0.5722222222)
    assert average_precision(labels, scores) == pytest.approx(0.5333333333)


def test_paper_balanced_auprc_uses_replacement_and_100_repeats() -> None:
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    summary, replicates = paper_balanced_auprc(
        labels,
        scores,
        dataset="real",
        method="method",
        universe_mode="resource_fixed",
    )

    assert summary["paper_balanced_auprc_mean"] == pytest.approx(1.0)
    assert summary["paper_repeats_valid"] == 100
    assert replicates["replacement"].all()
    assert set(replicates["n_negative_draws"]) == {2}
    assert (replicates["n_unique_negative_draws"] < 2).any()


def test_tie_aware_auroc_and_degenerate_metrics() -> None:
    assert tie_aware_auroc(
        np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.6])
    ) == pytest.approx(0.75)
    assert np.isnan(tie_aware_auroc(np.array([1, 1]), np.array([0.9, 0.8])))
    assert np.isnan(yardstick_008_pr_auc(np.array([0, 0]), np.array([0.9, 0.8])))


def test_old_trapezoidal_pr_auc_exposes_all_tied_artifact() -> None:
    labels = np.array([1, 1, 0, 0])
    tied_scores = np.ones(4)

    assert tie_aware_auroc(labels, tied_scores) == pytest.approx(0.5)
    assert yardstick_008_pr_auc(labels, tied_scores) == pytest.approx(0.75)
