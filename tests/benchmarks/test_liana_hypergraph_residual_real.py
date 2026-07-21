from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.literature.run_liana_hypergraph_residual_real import (
    METHOD_FALLBACK,
    _coverage_fallback,
    _select_calls,
    _validate_frozen_rc2,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repository_rc2_acceptance_and_config_are_checksum_bound() -> None:
    config = _validate_frozen_rc2(
        REPO_ROOT / "benchmarks/configs/liana_hypergraph_residual_rc2_v1.json",
        REPO_ROOT
        / "benchmarks/results/liana_hypergraph_residual_rc2_v1/acceptance.json",
    )
    assert config["candidate_status"] == "benchmark_only_unreleased"


def test_select_calls_keeps_pvalue_set_and_uses_supplied_direction() -> None:
    table = pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["B", "B", "A"],
            "interaction_id": ["i1", "i2", "i3"],
            "interaction_pvalue": [0.01, 0.2, 0.01],
        }
    )
    calls = _select_calls(
        table,
        np.array([-0.2, 2.0, 0.3]),
        target="case",
        reference="control",
    )
    assert calls["interaction_id"].tolist() == ["i3", "i1"]
    assert calls["condition"].tolist() == ["case", "control"]


def test_coverage_fallback_uses_wald_only_for_unavailable_pairs() -> None:
    strict = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "method": ["strict", "strict"],
            "method_version": ["v", "v"],
            "resource": ["r", "r"],
            "ranking_semantics": ["s", "s"],
            "condition": ["case", "case"],
            "sender": ["A", "A"],
            "receiver": ["A", "B"],
            "ranked_strength": [3.0, np.nan],
            "condition_specific_directed_lr": [3.0, np.nan],
            "estimable_directed_lr": [10.0, np.nan],
            "status": ["observed", "not_estimable"],
            "reason_code": [None, "missing"],
        }
    )
    component = pd.DataFrame(
        {
            "method": ["C", "C"],
            "condition": ["case", "case"],
            "sender": ["A", "A"],
            "receiver": ["A", "B"],
            "ranked_strength": [1.0, 2.0],
            "estimable_directed_lr": [20.0, 20.0],
            "status": ["observed", "observed"],
        }
    )
    result = _coverage_fallback(strict, component)
    assert result["method"].eq(METHOD_FALLBACK).all()
    assert result["status"].eq("observed").all()
    assert result.loc[result["receiver"].eq("A"), "ranked_strength"].iloc[0] == 1.0
    assert result.loc[result["receiver"].eq("B"), "ranked_strength"].iloc[0] == 1.0


def test_mutated_acceptance_is_rejected(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="acceptance checksum mismatch"):
        _validate_frozen_rc2(
            REPO_ROOT
            / "benchmarks/configs/liana_hypergraph_residual_rc2_v1.json",
            acceptance,
        )
