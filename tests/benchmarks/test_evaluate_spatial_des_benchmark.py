from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.literature.evaluate_spatial_des_benchmark import (
    _parse_condition_map,
    _parse_dataset_map,
    _parse_expected_filters,
    _resolved_analysis_unit,
    _validate_run_binding,
    evaluate_rankings,
)


def test_evaluator_aligns_condition_aliases_and_unordered_mode() -> None:
    pairs = [("A", "A"), ("A", "B"), ("B", "B")]
    rankings = pd.DataFrame(
        {
            "dataset": ["local"] * 6,
            "method": ["m"] * 6,
            "method_version": ["1"] * 6,
            "resource": ["r"] * 6,
            "ranking_semantics": ["test"] * 6,
            "condition": ["case"] * 3 + ["ctrl"] * 3,
            "sender": [pair[0] for pair in pairs] * 2,
            "receiver": [pair[1] for pair in pairs] * 2,
            "ranked_strength": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
            "status": ["observed"] * 6,
        }
    )
    rows = []
    for condition in ("disease", "control"):
        for fraction in (0.1, 0.2, 0.3, 0.4):
            for sender, receiver in pairs:
                rows.append(
                    {
                        "dataset": "truth",
                        "scenario": "multi_sample",
                        "condition": condition,
                        "top_fraction": fraction,
                        "sender": sender,
                        "receiver": receiver,
                        "is_expected": sender == "A" and receiver == "A",
                    }
                )
    expected = pd.DataFrame.from_records(rows)

    scores, coverage = evaluate_rankings(
        rankings,
        expected,
        scenario="multi_sample",
        dataset_map={"local": "truth"},
        condition_map={"case": "disease", "ctrl": "control"},
    )

    assert len(scores) == 8
    assert scores["des"].eq(1.0).all()
    assert scores["cell_pair_direction"].eq("unordered_directions_collapsed").all()
    assert coverage["expected_pair_coverage_fraction"].eq(1.0).all()


def test_condition_map_parser_rejects_ambiguous_values() -> None:
    assert _parse_condition_map(["A=control", "B=case"]) == {
        "A": "control",
        "B": "case",
    }
    with pytest.raises(ValueError, match="SOURCE=EXPECTED"):
        _parse_condition_map(["bad"])
    with pytest.raises(ValueError, match="duplicate"):
        _parse_condition_map(["A=one", "A=two"])


def test_dataset_binding_is_explicit_and_fail_closed() -> None:
    rankings = pd.DataFrame(
        {
            "dataset": ["method_dataset"],
            "method": ["m"],
            "method_version": ["1"],
            "resource": ["r"],
            "ranking_semantics": ["test"],
            "condition": ["case"],
            "sender": ["A"],
            "receiver": ["A"],
            "ranked_strength": [1.0],
            "status": ["observed"],
        }
    )
    expected = pd.DataFrame(
        {
            "dataset": ["truth_dataset"] * 4,
            "scenario": ["multi_sample"] * 4,
            "condition": ["case"] * 4,
            "top_fraction": [0.1, 0.2, 0.3, 0.4],
            "sender": ["A"] * 4,
            "receiver": ["A"] * 4,
            "is_expected": [True] * 4,
        }
    )

    with pytest.raises(ValueError, match="explicit dataset map"):
        evaluate_rankings(rankings, expected, scenario="multi_sample")
    scores, _ = evaluate_rankings(
        rankings,
        expected,
        scenario="multi_sample",
        dataset_map={"method_dataset": "truth_dataset"},
    )
    assert scores["dataset"].eq("truth_dataset").all()
    assert _parse_dataset_map(["method=truth"]) == {"method": "truth"}
    with pytest.raises(ValueError, match="SOURCE=EXPECTED"):
        _parse_dataset_map(["bad"])


def test_condition_aware_analysis_unit_is_explicitly_condition_level() -> None:
    assert _resolved_analysis_unit("condition_aware", None) == "condition_level"
    assert _resolved_analysis_unit("multi_sample", "subject_id") == "subject_id"
    assert _resolved_analysis_unit("multi_sample", None) is None
    with pytest.raises(ValueError, match="unsupported analysis unit"):
        _resolved_analysis_unit("condition_aware", "cells")


def test_subject_run_cannot_be_evaluated_in_condition_level_scenario(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "analysis_unit": {
                    "replicate_key": "subject_id",
                    "subject_key": "subject_id",
                    "primary_panel": True,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _validate_run_binding(
        manifest, scenario="multi_sample", analysis_unit="subject_id"
    ) == {
        "replicate_key": "subject_id",
        "subject_key": "subject_id",
        "primary_panel": True,
    }
    with pytest.raises(ValueError, match="analysis mismatch"):
        _validate_run_binding(manifest, scenario="condition_aware", analysis_unit=None)
    with pytest.raises(ValueError, match="analysis mismatch"):
        _validate_run_binding(
            manifest, scenario="multi_sample", analysis_unit="sample_id"
        )


def test_evaluator_filters_one_frozen_spatial_variant() -> None:
    rankings = pd.DataFrame(
        {
            "dataset": ["local", "local"],
            "method": ["m", "m"],
            "method_version": ["1", "1"],
            "resource": ["r", "r"],
            "ranking_semantics": ["test", "test"],
            "condition": ["case", "case"],
            "sender": ["A", "A"],
            "receiver": ["A", "B"],
            "ranked_strength": [2.0, 1.0],
            "status": ["observed", "observed"],
        }
    )
    expected = pd.DataFrame(
        {
            "dataset": ["truth"] * 4,
            "scenario": ["multi_sample"] * 4,
            "condition": ["case"] * 4,
            "top_fraction": [0.1] * 4,
            "sender": ["A", "A", "A", "A"],
            "receiver": ["A", "B", "A", "B"],
            "variant": ["primary", "primary", "sensitivity", "sensitivity"],
            "is_expected": [True, False, False, True],
        }
    )

    scores, _ = evaluate_rankings(
        rankings,
        expected,
        scenario="multi_sample",
        dataset_map={"local": "truth"},
        expected_filters={"variant": "primary"},
    )

    assert scores.loc[0, "des"] == pytest.approx(1.0)
    assert _parse_expected_filters(["variant=primary"]) == {"variant": "primary"}
    with pytest.raises(ValueError, match="COLUMN=VALUE"):
        _parse_expected_filters(["bad"])
