from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from tests.support.g3p import (
    TEST_SCORE_SPEC_ID,
    TEST_SCORE_VERSION,
    default_passed_g3p_gate,
)

from crychic.core import ContractError
from crychic.inference.active_probability import (
    ActiveEdgeScoreStatus,
    ActiveProbabilityCollection,
    ActiveProbabilitySpec,
    G3PCalibrationEvidence,
    G3PCalibrationGate,
    G3PCalibrationScenarioResult,
    G3PGateStatus,
    LocalFDRFitStatus,
    NullActiveEdgeScoreRecord,
    NullScoreDistribution,
    PointActiveEdgeScoreRecord,
    build_g3p_calibration_gate,
    estimate_active_probabilities,
    fit_beta_uniform_mixture,
)
from crychic.resampling import ActiveNullSpec
from crychic.scoring import ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID


def _point(
    edge: str,
    score: float | None,
    *,
    stratum: str = "stratum-1",
    status: ActiveEdgeScoreStatus = ActiveEdgeScoreStatus.OBSERVED,
    reason: str | None = None,
) -> PointActiveEdgeScoreRecord:
    return PointActiveEdgeScoreRecord(
        candidate_edge_id=edge,
        stratum_id=stratum,
        score_version=TEST_SCORE_VERSION,
        source_score_collection_id="point-collection",
        n_subjects=8,
        score=score,
        status=status,
        reason_code=reason,
    )


def _null(
    edge: str,
    plan: str,
    score: float | None,
    *,
    stratum: str = "stratum-1",
    status: ActiveEdgeScoreStatus = ActiveEdgeScoreStatus.OBSERVED,
    reason: str | None = None,
) -> NullActiveEdgeScoreRecord:
    available = status in {
        ActiveEdgeScoreStatus.OBSERVED,
        ActiveEdgeScoreStatus.STRUCTURAL_ZERO,
    }
    return NullActiveEdgeScoreRecord(
        candidate_edge_id=edge,
        plan_id=plan,
        stratum_id=stratum,
        score_version=TEST_SCORE_VERSION,
        null_rerun_record_id=f"rerun-{plan}",
        source_score_collection_id=f"null-collection-{plan}" if available else None,
        n_subjects=8 if available else None,
        score=score,
        status=status,
        reason_code=reason,
    )


def _distribution(
    points: tuple[PointActiveEdgeScoreRecord, ...],
    plans: tuple[str, ...],
    nulls: tuple[NullActiveEdgeScoreRecord, ...],
) -> NullScoreDistribution:
    return NullScoreDistribution(
        active_null_id="active-null-v1",
        active_null_spec_id=ActiveNullSpec().active_null_spec_id,
        candidate_universe_id="candidate-universe-v1",
        candidate_universe_policy_id=ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID,
        score_spec_id=TEST_SCORE_SPEC_ID,
        point_records=points,
        plan_ids=plans,
        null_records=nulls,
    )


def test_score_records_enforce_typed_value_reason_semantics() -> None:
    observed = _point("edge-a", 0.5)
    zero = _point(
        "edge-b",
        0.0,
        status=ActiveEdgeScoreStatus.STRUCTURAL_ZERO,
        reason="receptor_ineligible",
    )

    assert observed.n_subjects == 8
    assert observed.reason_code is None
    assert zero.score == 0.0
    assert zero.reason_code == "receptor_ineligible"
    absent = replace(
        _point(
            "edge-absent",
            None,
            status=ActiveEdgeScoreStatus.NOT_ESTIMABLE,
            reason="candidate_missing_from_all_point_applications",
        ),
        n_subjects=0,
    )
    assert absent.n_subjects == 0
    assert absent.status is ActiveEdgeScoreStatus.NOT_ESTIMABLE
    with pytest.raises(ValueError, match="n_subjects >= 1"):
        replace(observed, n_subjects=0)
    with pytest.raises(ValueError, match="reason_code"):
        _point("edge-c", 0.5, reason="not-allowed")
    with pytest.raises(ValueError, match="score=None"):
        _point(
            "edge-d",
            0.5,
            status=ActiveEdgeScoreStatus.NOT_ESTIMABLE,
            reason="missing",
        )
    with pytest.raises(ValueError, match="collection and subjects"):
        NullActiveEdgeScoreRecord(
            candidate_edge_id="edge-a",
            plan_id="plan-a",
            stratum_id="stratum-1",
            score_version="global-score-v3",
            null_rerun_record_id="rerun-a",
            score=0.5,
            status=ActiveEdgeScoreStatus.OBSERVED,
            reason_code=None,
        )


def test_distribution_requires_exact_canonical_candidate_plan_rectangle() -> None:
    points = (_point("edge-b", 0.7), _point("edge-a", 0.6))
    plans = ("plan-b", "plan-a")
    nulls = tuple(
        _null(edge, plan, 0.2)
        for edge in ("edge-b", "edge-a")
        for plan in ("plan-b", "plan-a")
    )

    result = _distribution(points, plans, nulls)
    reordered = _distribution(
        tuple(reversed(points)),
        tuple(reversed(plans)),
        tuple(reversed(nulls)),
    )

    assert result.complete_rectangle is True
    assert result.candidate_edge_ids == ("edge-a", "edge-b")
    assert result.plan_ids == ("plan-a", "plan-b")
    assert result.distribution_id == reordered.distribution_id
    with pytest.raises(ContractError) as missing:
        _distribution(points, plans, nulls[:-1])
    assert missing.value.details.code == "active_null_incomplete_score_matrix"
    mismatched = replace(nulls[0], stratum_id="other-stratum")
    with pytest.raises(ContractError) as source_error:
        _distribution(points, plans, (mismatched, *nulls[1:]))
    assert source_error.value.details.code == "active_null_source_mismatch"


def test_complete_small_matrix_keeps_plus_one_p_fallback_without_probability() -> None:
    plans = ("plan-0", "plan-1", "plan-2")
    points = (
        _point("edge-a", 0.75),
        _point(
            "edge-zero",
            0.0,
            status=ActiveEdgeScoreStatus.STRUCTURAL_ZERO,
            reason="receptor_ineligible",
        ),
    )
    values = {
        "edge-a": (0.25, 0.75, 0.90),
        "edge-zero": (0.0, 0.25, 0.50),
    }
    nulls = tuple(
        _null(edge, plan, values[edge][index])
        for edge in ("edge-a", "edge-zero")
        for index, plan in enumerate(plans)
    )
    distribution = _distribution(points, plans, nulls)

    result = estimate_active_probabilities(
        distribution,
        spec=ActiveProbabilitySpec(),
    )
    by_edge = {item.candidate_edge_id: item for item in result.records}

    assert by_edge["edge-a"].active_null_empirical_p_value == pytest.approx(0.75)
    assert by_edge["edge-zero"].active_null_empirical_p_value == 1.0
    assert all(item.candidate_comm_probability is None for item in result.records)
    assert all(item.comm_probability is None for item in result.records)
    assert result.diagnostics[0].reason_code == "active_null_insufficient_plans"
    assert result.comm_probability_release_allowed is False


def test_nonobserved_null_cell_blocks_entire_stratum_p_and_probability() -> None:
    points = (
        _point("edge-a", 0.8),
        _point(
            "edge-b",
            0.0,
            status=ActiveEdgeScoreStatus.STRUCTURAL_ZERO,
            reason="receptor_ineligible",
        ),
    )
    plans = ("plan-a", "plan-b")
    nulls = (
        _null("edge-a", "plan-a", 0.3),
        _null(
            "edge-a",
            "plan-b",
            None,
            status=ActiveEdgeScoreStatus.FAILED,
            reason="active_null_rerun_failed",
        ),
        _null("edge-b", "plan-a", 0.3),
        _null("edge-b", "plan-b", 0.2),
    )

    distribution = _distribution(points, plans, nulls)
    result = estimate_active_probabilities(
        distribution,
        spec=ActiveProbabilitySpec(),
    )

    assert all(item.active_null_empirical_p_value is None for item in result.records)
    assert all(item.comm_probability is None for item in result.records)
    assert result.diagnostics[0].complete_score_matrix is False
    assert result.diagnostics[0].reason_code == ("active_null_incomplete_score_matrix")
    structural = next(
        item for item in result.records if item.candidate_edge_id == "edge-b"
    )
    assert structural.comm_probability is None


def test_bum_multistart_is_deterministic_and_accepts_null_only_solution() -> None:
    uniform = (np.arange(700, dtype=float) + 0.5) / 700
    alternative = ((np.arange(300, dtype=float) + 0.5) / 300) ** (1.0 / 0.3)
    values = np.concatenate([uniform, alternative])

    fitted = fit_beta_uniform_mixture(
        values,
        stratum_id="mixture",
        n_null_plans=200,
    )
    repeated = fit_beta_uniform_mixture(
        tuple(reversed(values)),
        stratum_id="mixture",
        n_null_plans=200,
    )
    null_only = fit_beta_uniform_mixture(
        np.linspace(0.5, 1.0, 400),
        stratum_id="null-only",
        n_null_plans=200,
    )

    assert fitted.status is LocalFDRFitStatus.OBSERVED
    assert fitted.diagnostic_id == repeated.diagnostic_id
    assert fitted.pi0 == pytest.approx(0.7, abs=0.08)
    assert fitted.a == pytest.approx(0.3, abs=0.05)
    assert fitted.projected_gradient_norm is not None
    assert fitted.projected_gradient_norm <= 1e-6
    assert fitted.log_likelihood_spread is not None
    assert fitted.log_likelihood_spread <= 1e-8
    assert null_only.status is LocalFDRFitStatus.OBSERVED
    assert null_only.pi0 == 1.0
    assert null_only.a == 0.5


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def release_case() -> tuple[
    NullScoreDistribution,
    ActiveProbabilitySpec,
    ActiveProbabilityCollection,
    ActiveProbabilityCollection,
]:
    spec = ActiveProbabilitySpec()
    gate = default_passed_g3p_gate(spec)
    plans = tuple(f"plan-{index:03d}" for index in range(200))
    uniform = (np.arange(170, dtype=float) + 0.5) / 170
    alternative = ((np.arange(30, dtype=float) + 0.5) / 30) ** (1.0 / 0.3)
    target_p = np.concatenate([uniform, alternative])
    extreme_counts = np.clip(np.rint(target_p * 201 - 1), 0, 200).astype(int)
    point_scores = (200 - extreme_counts) / 200
    points = (
        *(
            _point(f"edge-{index:03d}", float(score))
            for index, score in enumerate(point_scores)
        ),
        _point(
            "edge-structural-zero",
            0.0,
            status=ActiveEdgeScoreStatus.STRUCTURAL_ZERO,
            reason="receptor_ineligible",
        ),
    )
    null_grid = (np.arange(200, dtype=float) + 0.5) / 200
    nulls = tuple(
        _null(point.candidate_edge_id, plan, float(null_grid[plan_index]))
        for point in points
        for plan_index, plan in enumerate(plans)
    )
    distribution = _distribution(points, plans, nulls)
    candidate = estimate_active_probabilities(distribution, spec=spec)
    released = estimate_active_probabilities(
        distribution,
        spec=spec,
        calibration_gate=gate,
    )
    return distribution, spec, candidate, released


def test_candidate_probability_never_populates_release_without_gate(
    release_case: tuple[
        NullScoreDistribution,
        ActiveProbabilitySpec,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    _, _, candidate, _ = release_case
    observed = tuple(
        item
        for item in candidate.records
        if item.point_status is ActiveEdgeScoreStatus.OBSERVED
    )

    assert candidate.diagnostics[0].status is LocalFDRFitStatus.OBSERVED
    assert all(item.active_null_empirical_p_value is not None for item in observed)
    assert all(item.candidate_comm_probability is not None for item in observed)
    assert all(item.comm_probability is None for item in candidate.records)
    assert {item.probability_reason_code for item in candidate.records} == {
        "g3p_gate_not_passed"
    }
    ranked = sorted(
        observed,
        key=lambda item: cast(float, item.active_null_empirical_p_value),
    )
    assert cast(float, ranked[0].candidate_comm_probability) >= cast(
        float, ranked[-1].candidate_comm_probability
    )


def test_exact_passed_gate_releases_candidate_values_and_zero_semantics(
    release_case: tuple[
        NullScoreDistribution,
        ActiveProbabilitySpec,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    _, _, candidate, released = release_case
    candidate_by_edge = {item.candidate_edge_id: item for item in candidate.records}

    assert released.calibration_gate_status is G3PGateStatus.PASSED
    assert released.comm_probability_release_allowed is True
    assert all(item.comm_probability is not None for item in released.records)
    for item in released.records:
        assert (
            item.comm_probability
            == candidate_by_edge[item.candidate_edge_id].candidate_comm_probability
        )
        assert item.probability_reason_code is None
    zero = next(
        item
        for item in released.records
        if item.candidate_edge_id == "edge-structural-zero"
    )
    assert zero.active_null_empirical_p_value == 1.0
    assert zero.candidate_comm_probability == 0.0
    assert zero.comm_probability == 0.0


def test_g3p_gate_is_producer_owned_and_enforces_grid_and_thresholds(
    release_case: tuple[
        NullScoreDistribution,
        ActiveProbabilitySpec,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    _, spec, _, released = release_case
    assert released.calibration_gate_status is G3PGateStatus.PASSED
    with pytest.raises(TypeError, match="producer-owned"):
        G3PCalibrationScenarioResult()
    with pytest.raises(TypeError, match="producer-owned"):
        G3PCalibrationEvidence()
    with pytest.raises(TypeError):
        G3PCalibrationGate()
    with pytest.raises(TypeError, match="producer-owned"):
        build_g3p_calibration_gate(object())  # type: ignore[arg-type]
    assert default_passed_g3p_gate(spec).comm_probability_release_allowed


def test_wrong_gate_binding_is_rejected_instead_of_silently_released(
    release_case: tuple[
        NullScoreDistribution,
        ActiveProbabilitySpec,
        ActiveProbabilityCollection,
        ActiveProbabilityCollection,
    ],
) -> None:
    distribution, spec, _, _ = release_case
    gate = default_passed_g3p_gate(spec)
    foreign = replace(distribution, active_null_spec_id="different-active-null-spec")

    with pytest.raises(ContractError) as error:
        estimate_active_probabilities(
            foreign,
            spec=spec,
            calibration_gate=gate,
        )
    assert error.value.details.code == "g3p_gate_runtime_binding_mismatch"


def test_probability_spec_refuses_runtime_threshold_or_estimator_substitution() -> None:
    with pytest.raises(ValueError, match="minimum_candidate_edges=200"):
        ActiveProbabilitySpec(minimum_candidate_edges=199)
    with pytest.raises(ValueError, match="BUM estimator"):
        ActiveProbabilitySpec(estimator="caller-selected-model")
