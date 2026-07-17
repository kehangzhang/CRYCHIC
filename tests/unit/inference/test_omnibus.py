from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.inference.omnibus import (
    FullPipelineOmnibusDistribution,
    FullPipelineOmnibusRecord,
    FullPipelineOmnibusRecordStatus,
    OOFContextOmnibusResult,
    OOFContextOmnibusSpec,
    build_full_pipeline_omnibus_record,
    canonical_helmert_basis,
    fit_oof_context_omnibus,
    summarize_full_pipeline_omnibus,
)


def _spec(
    contexts: tuple[str, ...] = ("A", "B"),
    *,
    minimum_clusters: int = 8,
) -> OOFContextOmnibusSpec:
    return OOFContextOmnibusSpec(
        hypothesis_id="receiver-R-family-F-context-omnibus",
        omnibus_name="all-contexts",
        context_ids=contexts,
        minimum_clusters_for_diagnostic_se=minimum_clusters,
    )


def _scores(
    contexts: tuple[str, ...] = ("A", "B"),
    *,
    singular: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_index in range(16):
        subject = f"s{subject_index:02d}"
        fold = f"fold-{subject_index % 2}"
        fold_shift = 2.0 if fold == "fold-1" else 0.0
        subject_shift = subject_index * 0.11
        for context_index, context in enumerate(contexts):
            if singular:
                noise = 0.0
            else:
                noise = ((subject_index * 7 + context_index * 3) % 11 - 5) * 0.08 + (
                    subject_index % 3
                ) * context_index * 0.03
            rows.append(
                {
                    "subject_id": subject,
                    "sample_id": f"{subject}-{context}",
                    "fold_id": fold,
                    "context_id": context,
                    "score": (subject_shift + fold_shift + context_index * 1.5 + noise),
                    "score_status": "observed",
                    "scoring_function_id": f"functional-{fold}",
                }
            )
    return pd.DataFrame(rows)


def _record(
    spec: OOFContextOmnibusSpec,
    index: int,
    statistic: float | None,
    *,
    status: FullPipelineOmnibusRecordStatus = (
        FullPipelineOmnibusRecordStatus.OBSERVED
    ),
) -> FullPipelineOmnibusRecord:
    available = status is not FullPipelineOmnibusRecordStatus.FAILED
    observed = status is FullPipelineOmnibusRecordStatus.OBSERVED
    return build_full_pipeline_omnibus_record(
        full_pipeline_record_id=f"pipeline-{index}",
        plan_id=f"permutation-plan-{index}",
        crossfit_id=f"crossfit-{index}" if available else None,
        resample_index=index,
        omnibus_spec_id=spec.spec_id,
        hypothesis_id=spec.hypothesis_id,
        omnibus_result_id=f"omnibus-result-{index}" if available else None,
        wald_statistic=statistic if observed else None,
        status=status,
        reason_code=None if observed else "permutation_omnibus_not_estimable",
    )


def test_spec_and_canonical_helmert_basis_are_order_invariant() -> None:
    first = _spec(("C", "A", "B"))
    second = _spec(("B", "C", "A"))
    basis = canonical_helmert_basis(("C", "A", "B"))
    expected = np.asarray(
        [
            [1 / math.sqrt(2), -1 / math.sqrt(2), 0.0],
            [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)],
        ]
    )

    assert first.context_ids == ("A", "B", "C")
    assert first.spec_id == second.spec_id
    assert first.support_effect_spec_id == second.support_effect_spec_id
    assert first.degrees_of_freedom == 2
    np.testing.assert_allclose(basis, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(basis.sum(axis=1), 0.0, atol=1e-14)
    assert not basis.flags.writeable

    with pytest.raises(ValueError, match="at least two"):
        _spec(("A",))
    with pytest.raises(ValueError, match="unique"):
        _spec(("A", "A"))


def test_two_context_wald_matches_hand_quadratic_form() -> None:
    result = fit_oof_context_omnibus(_scores(), _spec())

    assert result.status == "observed"
    assert result.wald_statistic is not None
    assert result.degrees_of_freedom == 1
    assert result.helmert_estimates.shape == (1,)
    assert result.helmert_covariance.shape == (1, 1)
    expected = float(result.helmert_estimates[0] ** 2 / result.helmert_covariance[0, 0])
    assert result.wald_statistic == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result.formal_inference_allowed is False
    assert not result.helmert_basis.flags.writeable
    assert not result.helmert_covariance.flags.writeable


def test_three_context_wald_matches_hand_matrix_solve() -> None:
    spec = _spec(("A", "B", "C"))
    result = fit_oof_context_omnibus(_scores(spec.context_ids), spec)

    assert result.status == "observed"
    assert result.wald_statistic is not None
    expected = float(
        result.helmert_estimates
        @ np.linalg.solve(result.helmert_covariance, result.helmert_estimates)
    )
    assert result.wald_statistic == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert np.linalg.matrix_rank(result.helmert_basis) == 2
    assert np.all(np.linalg.eigvalsh(result.helmert_covariance) > 0)


def test_omnibus_is_response_scale_equivariant() -> None:
    spec = _spec(("A", "B", "C"))
    table = _scores(spec.context_ids)
    scaled = table.copy()
    scaled["score"] *= 1.0e-8

    ordinary = fit_oof_context_omnibus(table, spec)
    tiny = fit_oof_context_omnibus(scaled, spec)

    assert ordinary.status == tiny.status == "observed"
    assert tiny.wald_statistic == pytest.approx(ordinary.wald_statistic)
    np.testing.assert_allclose(
        tiny.helmert_estimates,
        ordinary.helmert_estimates * 1.0e-8,
        rtol=1.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tiny.helmert_covariance,
        ordinary.helmert_covariance * 1.0e-16,
        rtol=1.0e-10,
        atol=0.0,
    )


def test_score_row_and_context_order_do_not_change_result_identity() -> None:
    first_spec = _spec(("C", "A", "B"))
    second_spec = _spec(("B", "C", "A"))
    table = _scores(("A", "B", "C"))
    shuffled = table.sample(frac=1.0, random_state=17).reset_index(drop=True)

    first = fit_oof_context_omnibus(table, first_spec)
    second = fit_oof_context_omnibus(shuffled, second_spec)

    assert first.result_id == second.result_id
    assert first.source_table_digest == second.source_table_digest
    np.testing.assert_array_equal(first.context_estimates, second.context_estimates)
    np.testing.assert_array_equal(first.helmert_covariance, second.helmert_covariance)


def test_singular_cr2_covariance_and_source_rank_fail_closed() -> None:
    contexts = ("A", "B", "C")
    singular = fit_oof_context_omnibus(
        _scores(contexts, singular=True),
        _spec(contexts),
    )

    assert singular.status == "not_estimable"
    assert singular.wald_statistic is None
    assert singular.reason_code == "oof_omnibus_covariance_rank_deficient"

    incomplete = _scores(contexts).loc[lambda frame: frame["context_id"].ne("C")]
    missing_context = fit_oof_context_omnibus(incomplete, _spec(contexts))
    assert missing_context.status == "not_estimable"
    assert missing_context.wald_statistic is None
    assert missing_context.reason_code == (
        "oof_omnibus_source_oof_effect_context_universe_mismatch"
    )


def test_omnibus_rejects_point_cluster_support_below_preregistered_minimum() -> None:
    table = _scores().loc[
        lambda frame: frame["subject_id"].isin(
            tuple(f"s{index:02d}" for index in range(6))
        )
    ]

    result = fit_oof_context_omnibus(table, _spec())

    assert result.n_clusters == 6
    assert result.status == "not_estimable"
    assert result.wald_statistic is None
    assert result.reason_code == "oof_omnibus_insufficient_cluster_support"

    five_subjects = _scores().loc[
        lambda frame: frame["subject_id"].isin(
            tuple(f"s{index:02d}" for index in range(5))
        )
    ]
    formal_floor = fit_oof_context_omnibus(
        five_subjects,
        _spec(minimum_clusters=4),
    )
    assert formal_floor.status == "not_estimable"
    assert formal_floor.reason_code == "oof_omnibus_insufficient_cluster_support"


def test_empirical_permutation_p_uses_plus_one_and_stays_diagnostic() -> None:
    spec = _spec()
    point = fit_oof_context_omnibus(_scores(), spec)
    assert point.wald_statistic is not None
    records = (
        _record(spec, 0, 0.0),
        _record(spec, 1, point.wald_statistic - 1.0),
        _record(spec, 2, point.wald_statistic + 1.0),
    )

    result = summarize_full_pipeline_omnibus(point, spec, records)
    reordered = summarize_full_pipeline_omnibus(point, spec, tuple(reversed(records)))

    assert result.distribution_id == reordered.distribution_id
    assert result.n_permutation_total == 3
    assert result.n_permutation_observed == 3
    assert result.permutation_plan_ids == tuple(
        f"permutation-plan-{index}" for index in range(3)
    )
    assert result.full_pipeline_record_ids == tuple(
        f"pipeline-{index}" for index in range(3)
    )
    assert result.diagnostic_empirical_p == pytest.approx(0.5)
    assert result.diagnostic_status == "diagnostic_permutation_count_below_1000"
    assert result.formal_reason_code == (
        "full_pipeline_omnibus_permutations_below_1000"
    )
    assert result.p_value is None and result.q_value is None
    assert result.formal_inference_allowed is False
    assert not result.permutation_statistics.flags.writeable


def test_complete_1000_permutations_do_not_bypass_the_g3_gate() -> None:
    spec = _spec()
    point = fit_oof_context_omnibus(_scores(), spec)
    assert point.wald_statistic is not None
    records = tuple(
        _record(spec, index, point.wald_statistic + (index % 5) - 2.0)
        for index in range(1_000)
    )

    result = summarize_full_pipeline_omnibus(point, spec, records)

    assert result.n_permutation_observed == 1_000
    assert result.diagnostic_status == "diagnostic_complete_minimum_1000"
    assert result.diagnostic_empirical_p is not None
    assert result.formal_reason_code == "g3_frequency_calibration_gate_not_applied"
    assert result.p_value is None and result.q_value is None
    assert result.formal_inference_allowed is False


def test_missing_permutation_keeps_distribution_incomplete() -> None:
    spec = _spec()
    point = fit_oof_context_omnibus(_scores(), spec)
    assert point.wald_statistic is not None
    records = tuple(
        _record(
            spec,
            index,
            point.wald_statistic,
            status=(
                FullPipelineOmnibusRecordStatus.NOT_ESTIMABLE
                if index == 999
                else FullPipelineOmnibusRecordStatus.OBSERVED
            ),
        )
        for index in range(1_000)
    )

    result = summarize_full_pipeline_omnibus(point, spec, records)

    assert result.n_permutation_total == 1_000
    assert result.n_permutation_observed == 999
    assert result.diagnostic_empirical_p == pytest.approx(1.0)
    assert result.diagnostic_status == (
        "diagnostic_incomplete_permutation_distribution"
    )
    assert result.formal_reason_code == (
        "full_pipeline_omnibus_permutation_distribution_incomplete"
    )


def test_duplicate_and_wrong_lineage_records_fail_closed() -> None:
    spec = _spec()
    point = fit_oof_context_omnibus(_scores(), spec)
    assert point.wald_statistic is not None
    record = _record(spec, 0, point.wald_statistic)
    with pytest.raises(ContractError) as duplicate_error:
        summarize_full_pipeline_omnibus(point, spec, (record, record))
    assert duplicate_error.value.details.code == (
        "duplicate_full_pipeline_omnibus_record"
    )

    wrong_spec = _spec(("A", "C"))
    wrong = _record(wrong_spec, 1, point.wald_statistic)
    with pytest.raises(ContractError) as lineage_error:
        summarize_full_pipeline_omnibus(point, spec, (record, wrong))
    assert lineage_error.value.details.code == (
        "full_pipeline_omnibus_record_spec_mismatch"
    )


def test_producer_owned_artifacts_and_tampering_are_rejected() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        OOFContextOmnibusResult()
    with pytest.raises(TypeError, match="producer-owned"):
        FullPipelineOmnibusRecord()
    with pytest.raises(TypeError, match="producer-owned"):
        FullPipelineOmnibusDistribution()

    tampered_spec = _spec()
    object.__setattr__(tampered_spec, "context_ids", ("B", "A"))
    with pytest.raises(ContractError) as spec_error:
        tampered_spec.to_dict()
    assert spec_error.value.details.code == "oof_omnibus_spec_integrity_violation"

    spec = _spec()
    point = fit_oof_context_omnibus(_scores(), spec)
    assert point.wald_statistic is not None
    record = _record(spec, 0, point.wald_statistic)
    distribution = summarize_full_pipeline_omnibus(point, spec, (record,))

    object.__setattr__(point, "wald_statistic", point.wald_statistic + 1.0)
    with pytest.raises(ContractError) as point_error:
        point.to_dict()
    assert point_error.value.details.code == "oof_omnibus_result_integrity_violation"

    intact_point = fit_oof_context_omnibus(_scores(), spec)
    assert intact_point.wald_statistic is not None
    object.__setattr__(record, "wald_statistic", record.wald_statistic + 1.0)
    with pytest.raises(ContractError) as record_error:
        record.to_dict()
    assert record_error.value.details.code == (
        "full_pipeline_omnibus_record_integrity_violation"
    )

    object.__setattr__(distribution, "diagnostic_status", "forged")
    with pytest.raises(ContractError) as distribution_error:
        distribution.to_dict()
    assert distribution_error.value.details.code == (
        "full_pipeline_omnibus_distribution_integrity_violation"
    )
