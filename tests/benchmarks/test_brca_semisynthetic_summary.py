from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.comprehensive.summarize_brca_semisynthetic import (
    PRIMARY_POLICY,
    SUMMARY_METRICS,
    _mean_ci,
    _method_summary,
    _paired_differences,
    _primary_rows,
)


def _record(
    dataset: str,
    method: str,
    auprc: float,
    *,
    view: str | None = None,
    engine: str | None = None,
) -> dict[str, object]:
    policy = PRIMARY_POLICY[method]
    record: dict[str, object] = {
        "dataset_id": dataset,
        "base_method_id": method,
        "view_label": policy["view_label"] if view is None else view,
        "differential_engine": (
            policy["differential_engine"] if engine is None else engine
        ),
        "formal_did_p_value": method != "scseqcommdiff",
    }
    record.update(dict.fromkeys(SUMMARY_METRICS, 1.0))
    record["omnibus_auprc"] = auprc
    return record


def test_mean_ci_handles_constant_and_missing_values() -> None:
    interval = _mean_ci([0.5, 0.5, np.nan, 0.5])

    assert interval == {"n": 3, "mean": 0.5, "ci_low": 0.5, "ci_high": 0.5}


def test_primary_summary_never_selects_sensitivity_view() -> None:
    records: list[dict[str, object]] = []
    values = {
        "crychic": (0.90, 0.92),
        "cellchat": (0.94, 0.96),
        "liana_rank_aggregate": (0.60, 0.70),
        "scseqcommdiff": (0.85, 0.87),
    }
    for dataset_index, dataset in enumerate(("r1", "r2")):
        for method, method_values in values.items():
            records.append(_record(dataset, method, method_values[dataset_index]))
    records.append(
        _record(
            "r1",
            "crychic",
            1.0,
            view="posthoc_sensitivity",
            engine="within_sample_rank_mean",
        )
    )
    primary = _primary_rows(pd.DataFrame.from_records(records))
    summary = _method_summary(primary)
    paired, gate = _paired_differences(
        primary,
        summary,
        noninferiority_margin=0.05,
    )

    assert len(primary) == 8
    assert "posthoc_sensitivity" not in set(primary["view_label"])
    assert summary.iloc[0]["base_method_id"] == "cellchat"
    assert summary.iloc[1]["base_method_id"] == "crychic"
    assert gate["strongest_baseline"] == "cellchat"
    assert gate["status"] == "PASS"
    comparison = paired.loc[paired["baseline_method_id"].eq("cellchat")].iloc[0]
    assert np.isclose(comparison["mean_auprc_difference"], -0.04)
