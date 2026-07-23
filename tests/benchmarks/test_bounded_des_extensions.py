from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.literature.evaluate_bounded_des_extensions import (
    CONTRACTS,
    EVENT_BUDGETS,
    METHOD_ID,
    _assert_expected_parity,
    _build_spatial_expected,
    _endpoint_availability,
    _permute_slice_labels,
    _rank_stability,
    _resample_slices,
    _summarize_resamples,
    _supporting_analysis_availability,
    _validate_typed_values,
)


def _sample_values(dataset: str = "ms") -> pd.DataFrame:
    contract = CONTRACTS[dataset]
    pairs = [
        ("A", "B", 5.0),
        ("A", "C", -4.0),
        ("A", "D", 3.0),
        ("B", "C", -2.0),
        ("B", "D", 1.0),
    ]
    rows: list[dict[str, object]] = []
    for condition, condition_shift in (
        (contract.reference_condition, 0.0),
        (contract.target_condition, 1.0),
    ):
        for sample_index in range(3):
            for sender, receiver, effect in pairs:
                rows.append(
                    {
                        "sample_id": f"{condition}_{sample_index}",
                        "subject_id": f"subject_{condition}_{sample_index}",
                        "condition": condition,
                        "sender": sender,
                        "receiver": receiver,
                        "endpoint_value": (
                            sample_index * 0.01 + condition_shift * effect
                        ),
                        "status": "observed",
                        "reason_code": None,
                    }
                )
            rows.append(
                {
                    "sample_id": f"{condition}_{sample_index}",
                    "subject_id": f"subject_{condition}_{sample_index}",
                    "condition": condition,
                    "sender": "C",
                    "receiver": "D",
                    "endpoint_value": np.nan,
                    "status": "not_estimable",
                    "reason_code": "source_missing",
                }
            )
    return pd.DataFrame.from_records(rows)


def test_v6_event_level_variants_are_explicitly_not_estimable() -> None:
    result = _endpoint_availability(CONTRACTS["ms"])
    legacy = result.loc[result["des_variant"].eq("legacy_bounded_pair_rank_des")]
    strict = result.loc[result["claim_scope"].eq("suggestion_v6_strict_endpoint")]

    assert len(legacy) == 1
    assert legacy.iloc[0]["status"] == "observed"
    assert set(strict["status"]) == {"not_estimable"}
    assert strict["reason_code"].notna().all()
    assert set(
        strict.loc[
            strict["des_variant"].eq("top_k_count_matched_des"), "event_budget"
        ].dropna()
    ) == set(EVENT_BUDGETS)
    assert set(result["dataset_role"]) == {"independent_validation"}
    assert set(result["independent_validation"]) == {True}
    assert not result["cross_dataset_pooling_allowed"].any()


def test_kuppe_is_never_labelled_independent_validation() -> None:
    result = _endpoint_availability(CONTRACTS["kuppe"])
    assert set(result["dataset_role"]) == {"development"}
    assert not result["independent_validation"].any()


def test_supporting_availability_does_not_invent_an_event_count_curve() -> None:
    result = _supporting_analysis_availability(CONTRACTS["ms"])
    curve = result.loc[result["analysis"].eq("predicted_event_count_curve")].iloc[0]
    assert curve["status"] == "not_estimable"
    assert curve["reason_code"] == "candidate_event_selection_ledger_not_exported"
    assert result.loc[
        result["analysis"].eq("cell_pair_coverage"), "status"
    ].item() == "observed"


@pytest.mark.parametrize(
    ("scores", "statuses", "message"),
    (
        ([1.0, 0.0], ["observed", "not_estimable"], "zero imputation"),
        ([np.nan, np.nan], ["observed", "not_estimable"], "observed rows"),
    ),
)
def test_typed_missingness_rejects_invalid_values(
    scores: list[float], statuses: list[str], message: str
) -> None:
    table = pd.DataFrame({"value": scores, "status": statuses})
    with pytest.raises(ValueError, match=message):
        _validate_typed_values(table, value_column="value", label="fixture")


def test_spatial_expected_rebuild_keeps_missing_pairs_out_of_ranking() -> None:
    rankings, expected = _build_spatial_expected(
        _sample_values(), contract=CONTRACTS["ms"]
    )

    missing = rankings.loc[
        rankings["sender"].eq("C") & rankings["receiver"].eq("D")
    ].iloc[0]
    assert missing["status"] == "not_estimable"
    assert pd.isna(missing["spatial_rank"])
    assert not missing["ranking_eligible"]
    assert set(expected["rankable_pairs"]) == {5}

    top = expected.loc[expected["top_fraction"].eq(0.4)]
    assert top["is_expected"].sum() == 2
    selected = top.loc[top["is_expected"], ["condition", "sender", "receiver"]]
    assert set(map(tuple, selected.to_numpy())) == {
        (CONTRACTS["ms"].target_condition, "A", "B"),
        (CONTRACTS["ms"].reference_condition, "A", "C"),
    }


def test_expected_parity_fails_closed_on_membership_change() -> None:
    _, expected = _build_spatial_expected(
        _sample_values(), contract=CONTRACTS["ms"]
    )
    _assert_expected_parity(expected, expected.copy())

    altered = expected.copy()
    altered.loc[0, "is_expected"] = not bool(altered.loc[0, "is_expected"])
    with pytest.raises(ValueError, match="membership differs"):
        _assert_expected_parity(expected, altered)


def test_resampling_and_permutation_preserve_condition_sizes() -> None:
    values = _sample_values()
    contract = CONTRACTS["ms"]
    rng = np.random.default_rng(7)

    bootstrap = _resample_slices(values, contract=contract, rng=rng)
    original_counts = (
        values.loc[:, ["sample_id", "condition"]]
        .drop_duplicates()["condition"]
        .value_counts()
        .to_dict()
    )
    bootstrap_counts = (
        bootstrap.loc[:, ["sample_id", "condition"]]
        .drop_duplicates()["condition"]
        .value_counts()
        .to_dict()
    )
    assert bootstrap_counts == original_counts

    permuted = _permute_slice_labels(values, rng=rng)
    permutation_counts = (
        permuted.loc[:, ["sample_id", "condition"]]
        .drop_duplicates()["condition"]
        .value_counts()
        .to_dict()
    )
    assert permutation_counts == original_counts


def test_rank_stability_is_one_for_identical_rankings() -> None:
    rankings, _ = _build_spatial_expected(
        _sample_values(), contract=CONTRACTS["ms"]
    )
    rho, common, status, reason = _rank_stability(rankings, rankings.copy())
    assert common == 5
    assert rho == pytest.approx(1.0)
    assert status == "observed"
    assert reason is None


def test_resample_summary_excludes_not_estimable_values() -> None:
    full = pd.DataFrame(
        {
            "method": [METHOD_ID],
            "condition": ["control"],
            "top_fraction": [0.1],
            "des": [0.4],
            "status": ["observed"],
        }
    )
    scores = pd.DataFrame(
        {
            "resample_kind": ["slice_bootstrap", "slice_bootstrap"],
            "condition": ["control", "control"],
            "top_fraction": [0.1, 0.1],
            "status": ["observed", "not_estimable"],
            "des": [0.6, np.nan],
        }
    )
    summary = _summarize_resamples(scores, full, contract=CONTRACTS["ms"])
    assert summary.loc[0, "resamples_observed"] == 1
    assert summary.loc[0, "resamples_not_estimable"] == 1
    assert summary.loc[0, "resampled_des_mean"] == pytest.approx(0.6)
    assert summary.loc[0, "resample_observed_fraction"] == pytest.approx(0.5)
