from __future__ import annotations

import numpy as np

from benchmarks.literature.receiver_program_soft import (
    fit_receiver_program_reliability,
    receiver_program_weights,
)


def test_reliability_opens_for_concordant_and_closes_for_antagonistic_program() -> None:
    lr = np.r_[np.ones(300), -np.ones(300)]
    concordant = lr.copy()
    antagonistic = -lr
    opened = fit_receiver_program_reliability(lr, concordant, maximum_alpha=4.0)
    closed = fit_receiver_program_reliability(lr, antagonistic, maximum_alpha=4.0)
    assert opened.reliability == 1.0
    assert opened.effective_alpha == 1.5
    assert closed.reliability == 0.0
    assert closed.effective_alpha == 0.0


def test_missing_program_is_neutral_and_direction_orients_weight() -> None:
    fit = fit_receiver_program_reliability(
        np.r_[np.ones(300), -np.ones(300)],
        np.r_[np.ones(300), -np.ones(300)],
        maximum_alpha=1.0,
    )
    weight, evidence = receiver_program_weights(
        np.array([1.0, -1.0, 1.0]),
        np.array([2.0, 2.0, np.nan]),
        fit,
    )
    assert weight[0] > 1.0
    assert weight[1] < 1.0
    assert weight[2] == 1.0
    assert evidence[2] == 0.0


def test_insufficient_program_coverage_fails_closed() -> None:
    fit = fit_receiver_program_reliability(np.ones(20), np.ones(20), maximum_alpha=4.0)
    assert fit.n_concordance_edges == 20
    assert fit.effective_alpha == 0.0
    weights, _ = receiver_program_weights(np.ones(20), np.ones(20), fit)
    np.testing.assert_array_equal(weights, np.ones(20))
