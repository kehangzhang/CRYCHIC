from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.literature.run_bounded_program_blend_real import (
    METHOD_FALLBACK,
    _coverage_fallback,
)


def test_coverage_fallback_uses_rc3_only_for_missing_strict_pairs() -> None:
    strict = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "sender": ["x", "x", "x", "x"],
            "receiver": ["y", "z", "y", "z"],
            "ranked_strength": [2.0, np.nan, 1.0, np.nan],
            "condition_specific_directed_lr": [2.0, np.nan, 1.0, np.nan],
            "estimable_directed_lr": [10.0, np.nan, 10.0, np.nan],
            "status": ["observed", "not_estimable", "observed", "not_estimable"],
            "reason_code": [None, "missing", None, "missing"],
            "method": ["strict"] * 4,
            "method_version": ["test"] * 4,
            "ranking_semantics": ["test"] * 4,
        }
    )
    rc3 = strict.copy()
    rc3["method"] = "CRYCHIC_RC3_program_soft_fallback"
    rc3.loc[rc3["receiver"].eq("z"), "ranked_strength"] = 0.25
    rc3.loc[rc3["receiver"].eq("z"), "estimable_directed_lr"] = 20.0
    rc3.loc[rc3["receiver"].eq("z"), "status"] = "observed"
    result = _coverage_fallback(strict, rc3)
    assert result["method"].eq(METHOD_FALLBACK).all()
    assert result.loc[result["receiver"].eq("y"), "ranked_strength"].eq(1.0).all()
    assert result.loc[result["receiver"].eq("z"), "ranked_strength"].eq(0.25).all()
