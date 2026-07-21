from __future__ import annotations

import pandas as pd

from benchmarks.literature.run_adaptive_multiview_real import (
    METHOD_FALLBACK,
    _rc4_coverage_fallback,
)


def test_rc4_fallback_preserves_structural_absence_and_uses_c_only() -> None:
    strict = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "method": ["strict", "strict"],
            "method_version": ["v", "v"],
            "resource": ["r", "r"],
            "ranking_semantics": ["s", "s"],
            "condition": ["T", "T"],
            "sender": ["A", "A"],
            "receiver": ["B", "C"],
            "ranked_strength": [2.0, float("nan")],
            "condition_specific_directed_lr": [2.0, float("nan")],
            "estimable_directed_lr": [3.0, float("nan")],
            "status": ["observed", "not_estimable"],
            "reason_code": [None, "structural_absence"],
        }
    )
    component = pd.DataFrame(
        {
            "method": ["C", "C"],
            "condition": ["T", "T"],
            "sender": ["A", "A"],
            "receiver": ["B", "C"],
            "ranked_strength": [1.0, 3.0],
            "estimable_directed_lr": [4.0, 5.0],
            "status": ["observed", "observed"],
        }
    )
    result = _rc4_coverage_fallback(
        strict,
        component,
        method=METHOD_FALLBACK,
        reason_prefix="adaptive_not_estimable",
        semantics="test",
    )
    assert result["method"].eq(METHOD_FALLBACK).all()
    fallback = result.loc[result["receiver"].eq("C")].iloc[0]
    assert fallback["status"] == "observed"
    assert (
        fallback["condition_specific_directed_lr"]
        != fallback["condition_specific_directed_lr"]
    )
    assert "used_crychic_wald" in fallback["reason_code"]
