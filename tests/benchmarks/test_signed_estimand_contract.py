from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.adapters.common import sha256_file
from benchmarks.comprehensive.signed_estimand_contract import (
    SCHEMA_VERSION,
    deterministic_score_table,
    evaluate_contract,
    run_contract,
)


def test_fixture_has_exact_means_and_fold_centered_nondegenerate_noise() -> None:
    table = deterministic_score_table({"A": 3.0, "B": 2.0, "C": 1.0})

    assert len(table) == 48
    assert table["subject_id"].nunique() == 16
    assert table.groupby("fold_id")["subject_id"].nunique().to_dict() == {
        "fold-0": 8,
        "fold-1": 8,
    }
    observed = table.groupby("context_id", observed=True)["score"].mean()
    pd.testing.assert_series_equal(
        observed.sort_index(),
        pd.Series(
            {"A": 3.0, "B": 2.0, "C": 1.0},
            name="score",
            index=pd.Index(("A", "B", "C"), name="context_id"),
        ),
    )
    assert table.groupby("context_id", observed=True)["score"].std().gt(0).all()


def test_signed_estimand_contract_passes_every_case_and_invariant() -> None:
    cases, invariants = evaluate_contract()

    assert len(cases) == 15
    assert len(invariants) == 11
    assert cases["passed"].all()
    assert invariants["passed"].all()
    observed = cases.set_index("case_id")
    expected = {
        "two_group_A_minus_B": 1.0,
        "two_group_B_minus_A": -1.0,
        "three_group_A_minus_C": 2.0,
        "three_group_A_one_vs_rest": 1.5,
        "factorial_DID_positive": 2.0,
        "factorial_DID_negative": -2.0,
        "chain_edge_C3_minus_C2": 2.0,
    }
    for case_id, truth in expected.items():
        assert observed.loc[case_id, "observed_effect"] == pytest.approx(
            truth, abs=1.0e-10
        )
    assert np.isfinite(observed["observed_effect"]).all()
    assert set(cases["formal_p_value_status"]) == {
        "NE_requires_full_pipeline_resampling"
    }

    invariant = invariants.set_index("invariant_id")
    assert invariant.loc["missing_score_not_zero", "observed_status"] == (
        "not_estimable"
    )
    assert invariant.loc[
        "missing_context_cell_type_not_zero", "observed_status"
    ] == "not_estimable"


def test_contract_publication_is_atomic_checksum_bound_and_non_overwriting(
    tmp_path: Path,
) -> None:
    output = tmp_path / "signed-contract"
    manifest = run_contract(output, repo_root=Path(__file__).resolve().parents[2])

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest["gate_a"] == "PASS"
    published = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert published == manifest
    for record in manifest["outputs"].values():
        path = output / record["path"]
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "Gate A: **PASS**" in report

    try:
        run_contract(output, repo_root=Path(__file__).resolve().parents[2])
    except FileExistsError:
        pass
    else:  # pragma: no cover - protects the no-overwrite result contract
        raise AssertionError("run_contract overwrote an existing benchmark")
