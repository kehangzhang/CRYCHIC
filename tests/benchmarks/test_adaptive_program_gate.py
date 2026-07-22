from __future__ import annotations

import numpy as np

from benchmarks.literature.adaptive_program_gate import (
    adaptive_program_gate_weights,
    fit_adaptive_program_gate,
)


def _fit(program: np.ndarray, *, reduction: float = 0.5):
    return fit_adaptive_program_gate(
        np.ones(len(program)),
        program,
        np.ones(len(program), dtype=bool),
        relative_fdp_reduction=reduction,
        minimum_calls=20,
        minimum_sign_concordance=0.55,
        minimum_positive_tail_count=10,
        minimum_positive_tail_fraction=0.05,
    )


def test_flat_magnitudes_fall_back_to_sign_only() -> None:
    program = np.r_[np.full(70, 2.0), np.full(30, -2.0)]
    fit = _fit(program)
    assert fit.gate_mode == "sign_only"
    assert fit.threshold == 0.0
    weights = adaptive_program_gate_weights(np.ones(len(program)), program, fit)
    assert int(weights.sum()) == 70


def test_separable_tail_opens_stronger_gate() -> None:
    program = np.r_[np.full(40, 5.0), np.full(30, 1.0), np.full(30, -1.0)]
    fit = _fit(program)
    assert fit.gate_mode == "enriched_tail"
    assert fit.threshold == 5.0
    assert fit.retained_positive_calls == 40


def test_unreliable_direction_and_insufficient_calls_leave_scores_unchanged() -> None:
    antagonistic = _fit(np.r_[np.ones(40), -np.ones(60)])
    assert antagonistic.gate_mode == "no_gate"
    insufficient = fit_adaptive_program_gate(
        np.ones(10),
        np.ones(10),
        np.ones(10, dtype=bool),
        relative_fdp_reduction=0.5,
        minimum_calls=20,
    )
    assert insufficient.gate_mode == "no_gate"
    np.testing.assert_array_equal(
        adaptive_program_gate_weights(np.ones(10), np.ones(10), insufficient),
        np.ones(10),
    )


def test_missing_program_evidence_is_neutral() -> None:
    program = np.r_[np.full(40, 5.0), np.full(30, 1.0), np.full(30, -1.0)]
    fit = _fit(program)
    weights = adaptive_program_gate_weights(
        np.array([1.0, 1.0, -1.0]),
        np.array([np.nan, 6.0, 6.0]),
        fit,
    )
    np.testing.assert_array_equal(weights, np.array([1.0, 1.0, 0.0]))


def test_tail_must_repeat_across_validation_folds() -> None:
    program = np.r_[np.full(40, 5.0), np.full(30, 1.0), np.full(30, -1.0)]
    folds = np.r_[np.zeros(40, dtype=int), np.ones(60, dtype=int)]
    fit = fit_adaptive_program_gate(
        np.ones(len(program)),
        program,
        np.ones(len(program), dtype=bool),
        relative_fdp_reduction=0.5,
        minimum_calls=20,
        minimum_sign_concordance=0.55,
        minimum_positive_tail_count=10,
        minimum_positive_tail_fraction=0.05,
        validation_fold=folds,
        minimum_validation_tail_count=5,
        minimum_validation_concordance_gain=0.0,
    )
    assert fit.gate_mode == "sign_only"
