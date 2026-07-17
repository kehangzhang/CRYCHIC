from __future__ import annotations

from dataclasses import replace

import pytest

from crychic.inference.calibration_metrics import (
    BinaryCalibrationRecord,
    BinomialBoundSide,
    CalibrationNotEstimableError,
    compute_binary_calibration_metrics,
    compute_fixed_bin_ece,
    compute_logistic_recalibration,
    exact_binomial_one_sided_bound,
    replicate_cluster_bootstrap_ece_upper_bound,
)


def _golden_records() -> tuple[BinaryCalibrationRecord, ...]:
    probabilities = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)
    outcomes = (0, 1, 0, 0, 1, 0, 1, 1)
    return tuple(
        BinaryCalibrationRecord(
            replicate_id=f"replicate-{index // 2}",
            observation_id=f"edge-{index}",
            outcome=outcome,
            probability=probability,
        )
        for index, (probability, outcome) in enumerate(
            zip(probabilities, outcomes, strict=True)
        )
    )


def test_binary_calibration_metrics_match_hand_computable_golden_values() -> None:
    result = compute_binary_calibration_metrics(
        _golden_records(), baseline_probability=0.5, n_bins=4
    )

    assert result.n_observations == 8
    assert result.n_replicates == 4
    assert result.observed_prevalence == 0.5
    assert result.brier_score == pytest.approx(0.2, abs=1e-15)
    assert result.prevalence_only_brier_score == 0.25
    assert result.brier_relative_improvement == pytest.approx(0.2, abs=1e-15)
    assert result.fixed_bin_ece.ece == pytest.approx(0.25, abs=1e-15)
    assert result.logistic_recalibration.calibration_in_the_large == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.logistic_recalibration.intercept == pytest.approx(0.0, abs=1e-12)
    assert result.logistic_recalibration.slope == pytest.approx(
        0.8048888150605223, rel=1e-10, abs=1e-12
    )
    assert result.logistic_recalibration.gradient_infinity_norm <= 1e-10
    assert result.logistic_recalibration.curvature_minimum_eigenvalue > 0.0


def test_fixed_bin_ece_has_frozen_boundaries_and_explicit_empty_bins() -> None:
    records = (
        BinaryCalibrationRecord(
            replicate_id="r1", observation_id="zero", outcome=0, probability=0.0
        ),
        BinaryCalibrationRecord(
            replicate_id="r1", observation_id="quarter", outcome=0, probability=0.25
        ),
        BinaryCalibrationRecord(
            replicate_id="r2", observation_id="one", outcome=1, probability=1.0
        ),
    )

    result = compute_fixed_bin_ece(records, n_bins=4)

    first = result.replicates[0]
    second = result.replicates[1]
    assert [item.n_observations for item in first.bins] == [1, 1, 0, 0]
    assert [item.n_observations for item in second.bins] == [0, 0, 0, 1]
    assert first.bins[0].includes_upper is False
    assert first.bins[-1].includes_upper is True
    assert first.bins[2].mean_probability is None
    assert first.bins[2].observed_frequency is None
    assert first.bins[2].absolute_gap is None
    assert result.ece == pytest.approx(0.0625, abs=1e-15)
    assert result.aggregation_semantics == (
        "within_replicate_then_equal_replicate_mean_v1"
    )


def test_metrics_equal_weight_replicates_with_different_candidate_counts() -> None:
    records = (
        BinaryCalibrationRecord(
            replicate_id="small",
            observation_id="false-positive",
            outcome=0,
            probability=0.9,
        ),
        BinaryCalibrationRecord(
            replicate_id="small",
            observation_id="true-positive",
            outcome=1,
            probability=0.9,
        ),
        *tuple(
            BinaryCalibrationRecord(
                replicate_id="large",
                observation_id=f"null-{index}",
                outcome=1 if index == 0 else 0,
                probability=0.1,
            )
            for index in range(8)
        ),
        BinaryCalibrationRecord(
            replicate_id="large",
            observation_id="active",
            outcome=1,
            probability=0.9,
        ),
    )

    result = compute_binary_calibration_metrics(
        records, baseline_probability=0.5, n_bins=2
    )

    # Equal-replicate values are Brier=(0.41 + 0.89/9)/2 and
    # ECE=(0.4 + 0.03333...)/2. Candidate pooling gives different values.
    assert result.brier_score == pytest.approx(0.2544444444444444, abs=1e-15)
    assert result.fixed_bin_ece.ece == pytest.approx(0.21666666666666667, abs=1e-15)
    assert result.prevalence_only_brier_score == 0.25
    assert result.observed_prevalence == pytest.approx(13.0 / 36.0, abs=1e-15)
    assert result.brier_score != pytest.approx(1.71 / 11.0, abs=1e-15)
    assert result.fixed_bin_ece.ece != pytest.approx(0.08181818181818182)

    expanded = (
        *records[:2],
        *tuple(
            replace(item, observation_id=f"{item.observation_id}-copy-{copy}")
            for item in records[2:]
            for copy in range(3)
        ),
    )
    expanded_result = compute_binary_calibration_metrics(
        expanded, baseline_probability=0.5, n_bins=2
    )

    # Replicating every candidate in only the large dataset cannot alter any
    # equal-replicate point metric, including the weighted logistic fit.
    assert expanded_result.brier_score == pytest.approx(result.brier_score, abs=1e-15)
    assert expanded_result.fixed_bin_ece.ece == pytest.approx(
        result.fixed_bin_ece.ece, abs=1e-15
    )
    assert expanded_result.observed_prevalence == pytest.approx(
        result.observed_prevalence, abs=1e-15
    )
    assert expanded_result.logistic_recalibration.calibration_in_the_large == (
        pytest.approx(
            result.logistic_recalibration.calibration_in_the_large,
            rel=1e-10,
            abs=1e-12,
        )
    )
    assert expanded_result.logistic_recalibration.slope == pytest.approx(
        result.logistic_recalibration.slope, rel=1e-10, abs=1e-12
    )


def test_metrics_and_seeded_cluster_bootstrap_are_row_order_invariant() -> None:
    records = _golden_records()
    reversed_records = tuple(reversed(records))

    point = compute_binary_calibration_metrics(
        records, baseline_probability=0.5, n_bins=4
    )
    reordered_point = compute_binary_calibration_metrics(
        reversed_records, baseline_probability=0.5, n_bins=4
    )
    interval = replicate_cluster_bootstrap_ece_upper_bound(
        records,
        n_bins=4,
        confidence_level=0.95,
        n_resamples=1_000,
        seed=1729,
    )
    reordered_interval = replicate_cluster_bootstrap_ece_upper_bound(
        reversed_records,
        n_bins=4,
        confidence_level=0.95,
        n_resamples=1_000,
        seed=1729,
    )

    assert point == reordered_point
    assert interval == reordered_interval
    assert interval.ece == pytest.approx(0.25, abs=1e-15)
    assert interval.upper_bound == pytest.approx(0.35, abs=1e-15)
    assert interval.n_replicates == 4
    assert interval.seed == 1729


def test_bootstrap_resamples_whole_replicates_and_requires_valid_seed() -> None:
    records = _golden_records()

    first = replicate_cluster_bootstrap_ece_upper_bound(
        records, n_bins=4, n_resamples=250, seed=11
    )
    repeated = replicate_cluster_bootstrap_ece_upper_bound(
        records, n_bins=4, n_resamples=250, seed=11
    )
    changed = replicate_cluster_bootstrap_ece_upper_bound(
        records, n_bins=4, n_resamples=250, seed=12
    )

    assert first == repeated
    assert first.resampling_semantics.startswith("whole_replicate_cluster")
    assert changed.seed != first.seed
    with pytest.raises(ValueError, match="explicit non-negative integer"):
        replicate_cluster_bootstrap_ece_upper_bound(records, n_resamples=10, seed=-1)
    with pytest.raises(ValueError, match="explicit non-negative integer"):
        replicate_cluster_bootstrap_ece_upper_bound(records, n_resamples=10, seed=True)
    with pytest.raises(ValueError, match="confidence_level"):
        replicate_cluster_bootstrap_ece_upper_bound(
            records, n_resamples=10, seed=1, confidence_level=1.0
        )
    with pytest.raises(ValueError, match="n_resamples"):
        replicate_cluster_bootstrap_ece_upper_bound(records, n_resamples=0, seed=1)


def test_cluster_bootstrap_refuses_one_replicate() -> None:
    records = tuple(
        replace(item, replicate_id="only-replicate") for item in _golden_records()
    )

    with pytest.raises(CalibrationNotEstimableError) as error:
        replicate_cluster_bootstrap_ece_upper_bound(records, n_resamples=100, seed=7)
    assert error.value.reason_code == "calibration_cluster_bootstrap_too_few_replicates"


def test_exact_one_sided_clopper_pearson_bounds_match_golden_values() -> None:
    zero_upper = exact_binomial_one_sided_bound(0, 20, side=BinomialBoundSide.UPPER)
    all_lower = exact_binomial_one_sided_bound(20, 20, side=BinomialBoundSide.LOWER)
    middle_upper = exact_binomial_one_sided_bound(5, 20, side=BinomialBoundSide.UPPER)
    middle_lower = exact_binomial_one_sided_bound(5, 20, side=BinomialBoundSide.LOWER)

    assert zero_upper.bound == pytest.approx(0.13910834066826516, rel=1e-14)
    assert all_lower.bound == pytest.approx(0.8608916593317348, rel=1e-14)
    assert middle_upper.bound == pytest.approx(0.4555824040017489, rel=1e-14)
    assert middle_lower.bound == pytest.approx(0.10408083591013613, rel=1e-14)
    assert middle_upper.estimate == 0.25
    assert middle_upper.semantics == "exact_clopper_pearson_one_sided_v1"


@pytest.mark.parametrize(
    ("successes", "trials"),
    ((-1, 10), (11, 10), (True, 10), (1, 0)),
)
def test_binomial_bounds_reject_invalid_counts(successes: int, trials: int) -> None:
    with pytest.raises(ValueError):
        exact_binomial_one_sided_bound(successes, trials, side=BinomialBoundSide.UPPER)


@pytest.mark.parametrize("probability", (float("nan"), float("inf"), -0.01, 1.01))
def test_records_reject_nonfinite_or_illegal_probabilities(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        BinaryCalibrationRecord(
            replicate_id="r1",
            observation_id="edge-1",
            outcome=1,
            probability=probability,
        )


@pytest.mark.parametrize("outcome", (-1, 0.5, 2, "1"))
def test_records_reject_nonbinary_outcomes(outcome: object) -> None:
    with pytest.raises(ValueError, match="exactly binary"):
        BinaryCalibrationRecord(
            replicate_id="r1",
            observation_id="edge-1",
            outcome=outcome,  # type: ignore[arg-type]
            probability=0.5,
        )


def test_empty_duplicate_and_untyped_record_collections_fail_closed() -> None:
    record = _golden_records()[0]

    with pytest.raises(ValueError, match="at least one"):
        compute_fixed_bin_ece(())
    with pytest.raises(ValueError, match="unique"):
        compute_fixed_bin_ece((record, record))
    with pytest.raises(TypeError, match="BinaryCalibrationRecord"):
        compute_fixed_bin_ece((record, object()))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_bins"):
        compute_fixed_bin_ece((record,), n_bins=0)


def test_single_class_and_constant_predictions_do_not_emit_logistic_nan() -> None:
    single_class = tuple(replace(item, outcome=0) for item in _golden_records())
    constant_prediction = tuple(
        replace(item, probability=0.5) for item in _golden_records()
    )

    with pytest.raises(CalibrationNotEstimableError) as single_error:
        compute_logistic_recalibration(single_class)
    assert single_error.value.reason_code == "calibration_single_outcome_class"
    with pytest.raises(CalibrationNotEstimableError) as constant_error:
        compute_logistic_recalibration(constant_prediction)
    assert constant_error.value.reason_code == "calibration_constant_prediction"


def test_complete_or_quasi_complete_separation_is_not_estimable() -> None:
    records = tuple(
        BinaryCalibrationRecord(
            replicate_id=f"r-{index // 2}",
            observation_id=f"edge-{index}",
            outcome=outcome,
            probability=probability,
        )
        for index, (probability, outcome) in enumerate(
            zip((0.1, 0.2, 0.8, 0.9), (0, 0, 1, 1), strict=True)
        )
    )

    with pytest.raises(CalibrationNotEstimableError) as error:
        compute_logistic_recalibration(records)
    assert error.value.reason_code == "calibration_slope_separation"


def test_exact_zero_and_one_are_clipped_only_for_logistic_diagnostics() -> None:
    probabilities = (0.0, 0.0, 0.3, 0.4, 0.6, 0.7, 1.0, 1.0)
    outcomes = (0, 1, 0, 0, 1, 0, 1, 0)
    records = tuple(
        BinaryCalibrationRecord(
            replicate_id=f"r-{index // 2}",
            observation_id=f"edge-{index}",
            outcome=outcome,
            probability=probability,
        )
        for index, (probability, outcome) in enumerate(
            zip(probabilities, outcomes, strict=True)
        )
    )

    logistic = compute_logistic_recalibration(records, probability_clip=1e-5)
    ece = compute_fixed_bin_ece(records, n_bins=2)

    assert logistic.n_clipped_low == 2
    assert logistic.n_clipped_high == 2
    assert logistic.probability_clip == 1e-5
    assert ece.replicates[0].bins[0].mean_probability == 0.0
    assert ece.replicates[-1].bins[1].mean_probability == 1.0


def test_zero_baseline_brier_and_nonconvergence_are_explicitly_not_estimable() -> None:
    all_null = tuple(replace(item, outcome=0) for item in _golden_records())

    with pytest.raises(CalibrationNotEstimableError) as baseline_error:
        compute_binary_calibration_metrics(all_null, baseline_probability=0.0)
    assert baseline_error.value.reason_code == "calibration_baseline_brier_zero"
    with pytest.raises(CalibrationNotEstimableError) as convergence_error:
        compute_logistic_recalibration(_golden_records(), max_iterations=1)
    assert convergence_error.value.reason_code == "calibration_slope_nonconvergent"
