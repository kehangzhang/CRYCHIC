from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.literature.evaluate_spatial_des_benchmark import (
    _parse_condition_map,
    _parse_dataset_map,
    _parse_expected_filters,
    _parse_ranking_filters,
    _resolved_analysis_unit,
    _sha256,
    _validate_expected_run_bindings,
    _validate_run_binding,
    evaluate_rankings,
)
from benchmarks.literature.evaluate_spatial_des_benchmark import (
    run as run_evaluation,
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
        exclude_self_pairs=False,
        ranking_statistic="raw_strength",
    )

    assert len(scores) == 8
    assert scores["des"].eq(1.0).all()
    assert scores["score_type"].eq("std").all()
    assert scores["weight_exponent"].eq(1.0).all()
    assert scores["fgsea_analogue"].eq(
        "fgsea_scoreType=std;gseaParam=1"
    ).all()
    assert scores["tie_policy"].eq("fgsea_native_stable_input_order").all()
    assert scores["cell_pair_direction"].eq("unordered_directions_collapsed").all()
    assert coverage["expected_pair_coverage_fraction"].eq(1.0).all()


def test_exclude_self_pairs_filters_rankings_and_full_truth_universe() -> None:
    pairs = [("A", "A"), ("A", "B"), ("A", "C")]
    rankings = pd.DataFrame(
        {
            "dataset": ["fixture"] * 3,
            "method": ["m"] * 3,
            "method_version": ["1"] * 3,
            "resource": ["r"] * 3,
            "ranking_semantics": ["test"] * 3,
            "condition": ["case"] * 3,
            "sender": [pair[0] for pair in pairs],
            "receiver": [pair[1] for pair in pairs],
            "ranked_strength": [3.0, 2.0, 1.0],
            "status": ["observed"] * 3,
        }
    )
    expected = pd.DataFrame.from_records(
        {
            "dataset": "fixture",
            "scenario": "multi_sample",
            "condition": "case",
            "top_fraction": fraction,
            "sender": sender,
            "receiver": receiver,
            "is_expected": receiver in {"A", "B"},
        }
        for fraction in (0.1, 0.2, 0.3, 0.4)
        for sender, receiver in pairs
    )

    scores, coverage = evaluate_rankings(
        rankings,
        expected,
        scenario="multi_sample",
        exclude_self_pairs=True,
    )

    assert scores["status"].eq("observed").all()
    assert scores["des"].eq(1.0).all()
    assert coverage["expected_pairs"].eq(1).all()
    assert coverage["expected_pair_coverage_fraction"].eq(1.0).all()
    assert coverage["rank_pairs_eligible"].eq(2).all()


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
        exclude_self_pairs=False,
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


def test_schema_bound_analysis_unit_manifest_must_be_complete(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "crychic-liana-condition-aware-s4-benchmark-v1",
                "status": "prepared",
                "analysis_unit": {
                    "replicate_key": "subject_id",
                    "subject_key": "subject_id",
                    "primary_panel": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not complete"):
        _validate_run_binding(
            manifest,
            scenario="multi_sample",
            analysis_unit="subject_id",
        )


def test_v7_real_manifest_binds_frozen_subject_inputs_and_ranking(
    tmp_path: Path,
) -> None:
    ranking = tmp_path / "condition_cell_pair_rankings.tsv"
    ranking.write_text("fixture\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "crychic-suggest-next2-v7-real-e1-run-v1",
                "status": "complete",
                "analysis_unit": {
                    "replicate_key": "subject_id",
                    "subject_key": "subject_id",
                    "primary_panel": True,
                },
                "inputs": {
                    "h5ad": {"sha256": "i" * 64},
                    "preparation_manifest": {"sha256": "m" * 64},
                    "lr_resource": {"sha256": "r" * 64},
                    "lr_resource_manifest": {"sha256": "s" * 64},
                },
                "outputs": {
                    "condition_cell_pair_rankings.tsv": {
                        "sha256": _sha256(ranking)
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    binding = _validate_run_binding(
        manifest,
        scenario="multi_sample",
        analysis_unit="subject_id",
        ranking_paths=[ranking],
    )
    assert binding["input_sha256"] == "i" * 64
    assert binding["input_manifest_sha256"] == "m" * 64
    assert binding["resource_sha256"] == "r" * 64
    assert binding["resource_manifest_sha256"] == "s" * 64
    assert binding["primary_panel"] is True
    assert _validate_expected_run_bindings(
        binding,
        input_sha256="i" * 64,
        resource_sha256="r" * 64,
        resource_manifest_sha256="s" * 64,
    ) == {
        "input_sha256": "i" * 64,
        "resource_sha256": "r" * 64,
        "resource_manifest_sha256": "s" * 64,
    }


@pytest.mark.parametrize(
    "schema_version",
    [
        "crychic-scseqcommdiff-paper-benchmark-v1",
        "crychic-scseqcommdiff-paper-benchmark-v2",
    ],
)
def test_scseqcommdiff_manifest_binds_subject_unit_and_ranking(
    tmp_path: Path, schema_version: str
) -> None:
    ranking = tmp_path / "condition_cell_pair_rankings.tsv"
    ranking.write_text("fixture\n", encoding="utf-8")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "status": "complete",
                "dataset_id": "fixture",
                "preflight": {
                    "input": {
                        "sample_unit_key": "subject_id",
                        "sha256": "i" * 64,
                        "manifest_sha256": "m" * 64,
                    },
                    "resource": {
                        "sha256": "r" * 64,
                        "manifest_sha256": "s" * 64,
                    },
                },
                "outputs": {
                    "condition_cell_pair_rankings.tsv": {
                        "sha256": _sha256(ranking)
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    binding = _validate_run_binding(
        manifest,
        scenario="multi_sample",
        analysis_unit="subject_id",
        ranking_paths=[ranking],
    )
    assert binding["replicate_key"] == "subject_id"
    assert binding["binding_source"] == "preflight.input.sample_unit_key"
    assert binding["input_sha256"] == "i" * 64
    assert _validate_expected_run_bindings(
        binding,
        input_sha256="i" * 64,
        resource_sha256="r" * 64,
        resource_manifest_sha256="s" * 64,
    )["input_sha256"] == "i" * 64
    with pytest.raises(ValueError, match="input_sha256 differs"):
        _validate_expected_run_bindings(
            binding,
            input_sha256="x" * 64,
            resource_sha256=None,
            resource_manifest_sha256=None,
        )

    ranking.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        _validate_run_binding(
            manifest,
            scenario="multi_sample",
            analysis_unit="subject_id",
            ranking_paths=[ranking],
        )


def test_sample_effect_manifest_is_labeled_sensitivity_only(tmp_path: Path) -> None:
    ranking = tmp_path / "condition_cell_pair_rankings.tsv"
    ranking.write_text("fixture\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "crychic-sample-effect-des-ranking-v1",
                "status": "complete",
                "specification": {"statistical_unit": "subject_id"},
                "input": {"sha256": "u" * 64},
                "outputs": {"rankings": {"sha256": _sha256(ranking)}},
            }
        ),
        encoding="utf-8",
    )

    binding = _validate_run_binding(
        manifest,
        scenario="multi_sample",
        analysis_unit="subject_id",
        ranking_paths=[ranking],
    )
    assert binding["primary_panel"] is False
    assert binding["panel_role"] == "continuous_common_sensitivity_only"
    assert binding["upstream_interactions_sha256"] == "u" * 64


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
        exclude_self_pairs=False,
    )

    assert scores.loc[0, "des"] == pytest.approx(1.0)
    assert _parse_expected_filters(["variant=primary"]) == {"variant": "primary"}
    with pytest.raises(ValueError, match="COLUMN=VALUE"):
        _parse_expected_filters(["bad"])


def test_bound_runner_filters_one_crychic_ranking_variant(tmp_path: Path) -> None:
    ranking_path = tmp_path / "condition_cell_pair_rankings.tsv"
    ranking = pd.DataFrame(
        {
            "dataset": ["fixture"] * 4,
            "method": ["CRYCHIC_v7_G3_I1"] * 4,
            "method_version": ["v7"] * 4,
            "resource": ["r"] * 4,
            "ranking_semantics": ["count_selected_directed_lr_events"] * 4,
            "condition": ["case"] * 4,
            "sender": ["A", "A", "A", "A"],
            "receiver": ["B", "C", "B", "C"],
            "ranked_strength": [2.0, 1.0, 200.0, 100.0],
            "status": ["observed"] * 4,
            "des_variant": [
                "diagnostic_one_se_native_count_des",
                "diagnostic_one_se_native_count_des",
                "top_k_count_des",
                "top_k_count_des",
            ],
        }
    )
    ranking.to_csv(ranking_path, sep="\t", index=False)
    expected_path = tmp_path / "expected.tsv"
    expected = pd.DataFrame.from_records(
        {
            "dataset": "fixture",
            "scenario": "multi_sample",
            "condition": "case",
            "top_fraction": fraction,
            "sender": sender,
            "receiver": receiver,
            "is_expected": receiver == "B",
        }
        for fraction in (0.1, 0.2, 0.3, 0.4)
        for sender, receiver in (("A", "B"), ("A", "C"))
    )
    expected.to_csv(expected_path, sep="\t", index=False)
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": "crychic-suggest-next2-v7-real-e1-run-v1",
                "status": "complete",
                "analysis_unit": {
                    "replicate_key": "subject_id",
                    "subject_key": "subject_id",
                    "primary_panel": True,
                },
                "outputs": {
                    ranking_path.name: {"sha256": _sha256(ranking_path)}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = run_evaluation(
        [ranking_path],
        expected_path,
        tmp_path / "evaluation",
        scenario="multi_sample",
        analysis_unit="subject_id",
        dataset_map={},
        condition_map={},
        expected_filters={},
        ranking_filters={
            "des_variant": "diagnostic_one_se_native_count_des"
        },
        score_type="pos",
        exclude_self_pairs=True,
        ranking_statistic="raw_cardinality",
        overwrite=False,
        run_manifest_path=run_manifest,
    )

    assert manifest["ranking_filters"] == {
        "des_variant": "diagnostic_one_se_native_count_des"
    }
    assert manifest["outputs"]["scores"]["rows"] == 4
    assert _parse_ranking_filters(["des_variant=native"]) == {
        "des_variant": "native"
    }
    with pytest.raises(ValueError, match="COLUMN=VALUE"):
        _parse_ranking_filters(["bad"])
