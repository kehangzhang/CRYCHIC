from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.metrics.coverage_risk import (
    COVERAGE_STAGES,
    RISK_METRICS,
    build_coverage_ledger,
    build_coverage_risk_curve,
    summarize_coverage_ledger,
)


def _records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold, passing in ((0.1, 3), (0.2, 2)):
        for index in range(4):
            failure_stage = None
            if index >= passing:
                failure_stage = "receptor_eligible" if index == 3 else "state_eligible"
            failed = False
            row: dict[str, object] = {
                "dataset": "synthetic",
                "method": "crychic_candidate",
                "gate_threshold": threshold,
                "comparison_id": f"edge-{index}",
            }
            for stage in COVERAGE_STAGES:
                if stage == failure_stage:
                    failed = True
                row[stage] = "not_estimable" if failed else "eligible"
                row[f"{stage}_reason_code"] = (
                    f"failed_at_{failure_stage}" if failed else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _risk_metrics(*, missing_loso: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in (0.1, 0.2):
        row: dict[str, object] = {
            "dataset": "synthetic",
            "method": "crychic_candidate",
            "gate_threshold": threshold,
        }
        values = {
            "synthetic_auprc": 0.8,
            "ligand_only_false_activation": 0.0,
            "loso_stability": 0.7,
            "supportive_biology_direction": 1.0,
        }
        for metric in RISK_METRICS:
            missing = missing_loso and metric == "loso_stability" and threshold == 0.2
            row[metric] = None if missing else values[metric]
            row[f"{metric}_status"] = "not_estimable" if missing else "observed"
            row[f"{metric}_reason_code"] = "insufficient_subjects" if missing else None
        rows.append(row)
    return pd.DataFrame(rows)


def test_hierarchical_ledger_and_fixed_denominator_coverage() -> None:
    ledger = build_coverage_ledger(_records())
    summary = summarize_coverage_ledger(ledger)
    final = summary.loc[summary["stage"].eq("lr_identifiable")].set_index(
        "gate_threshold"
    )

    assert len(ledger) == 2 * 4 * len(COVERAGE_STAGES)
    assert final.loc[0.1, "n_comparisons"] == 4
    assert final.loc[0.1, "comparison_coverage"] == pytest.approx(0.75)
    assert final.loc[0.2, "comparison_coverage"] == pytest.approx(0.5)
    failed = ledger.loc[
        ledger["gate_threshold"].eq(0.1)
        & ledger["comparison_id"].eq("edge-3")
        & ledger["stage"].eq("lr_identifiable")
    ].iloc[0]
    assert failed["first_failure_stage"] == "receptor_eligible"
    assert failed["first_failure_reason_code"] == "failed_at_receptor_eligible"


def test_coverage_risk_curve_never_selects_a_threshold() -> None:
    curve = build_coverage_risk_curve(
        build_coverage_ledger(_records()),
        _risk_metrics(),
        coverage_stage="lr_identifiable",
    )

    assert not curve.selection_allowed
    assert not curve.table["threshold_selected"].any()
    assert set(curve.table["curve_status"]) == {"observed"}
    assert curve.table.set_index("gate_threshold").loc[
        0.1, "comparison_coverage"
    ] == pytest.approx(0.75)


def test_not_estimable_risk_is_preserved_without_imputation() -> None:
    curve = build_coverage_risk_curve(
        build_coverage_ledger(_records()),
        _risk_metrics(missing_loso=True),
    )
    row = curve.table.set_index("gate_threshold").loc[0.2]

    assert row["curve_status"] == "not_estimable"
    assert pd.isna(row["loso_stability"])
    assert "loso_stability" in row["curve_reason_code"]


def test_noneligible_stage_requires_reason_and_cannot_reactivate() -> None:
    missing_reason = _records()
    selected = missing_reason["comparison_id"].eq("edge-3")
    missing_reason.loc[selected, "receptor_eligible_reason_code"] = None
    with pytest.raises(ValueError, match="requires a reason"):
        build_coverage_ledger(missing_reason)

    reactivated = _records()
    selected = reactivated["comparison_id"].eq("edge-3")
    reactivated.loc[selected, "target_basis_eligible"] = "eligible"
    reactivated.loc[selected, "target_basis_eligible_reason_code"] = None
    with pytest.raises(ValueError, match="eligible again"):
        build_coverage_ledger(reactivated)


def test_thresholds_require_same_universe_and_monotone_eligibility() -> None:
    different_universe = _records().loc[
        lambda table: (
            ~(table["gate_threshold"].eq(0.2) & table["comparison_id"].eq("edge-3"))
        )
    ]
    with pytest.raises(ValueError, match="fixed comparison universe"):
        build_coverage_ledger(different_universe)

    reactivated = _records()
    selected = reactivated["gate_threshold"].eq(0.2) & reactivated["comparison_id"].eq(
        "edge-3"
    )
    for stage in COVERAGE_STAGES:
        reactivated.loc[selected, stage] = "eligible"
        reactivated.loc[selected, f"{stage}_reason_code"] = None
    with pytest.raises(ValueError, match="higher gate threshold reactivates"):
        build_coverage_ledger(reactivated)


def test_coverage_and_risk_threshold_keys_must_match() -> None:
    risks = _risk_metrics().loc[lambda table: table["gate_threshold"].eq(0.1)]

    with pytest.raises(ValueError, match="thresholds must match"):
        build_coverage_risk_curve(build_coverage_ledger(_records()), risks)
