from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.literature.summarize_spatial_des_benchmark import (
    SUMMARY_FILENAME,
    run,
    summarize_evaluations,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evaluation(
    root: Path,
    *,
    method: str,
    values: list[float | None],
    expected_sha256: str = "a" * 64,
    scenario: str = "multi_sample",
    table_scenario: str | None = None,
    analysis_unit: str = "subject_id",
    variant: str | None = "primary",
    conditions: tuple[str, str] = ("control", "case"),
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    rank_coverage: float = 0.75,
    expected_coverage: list[float] | None = None,
    ranking_semantics: str | None = None,
    schema_version: str = "crychic-spatial-des-evaluation-v1",
    tie_policy: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    strata = [
        (condition, fraction) for condition in conditions for fraction in fractions
    ]
    if len(values) != len(strata):
        raise ValueError("fixture values must match strata")
    expected_coverage = expected_coverage or [1.0] * len(strata)
    shared = {
        "dataset": ["fixture"] * len(strata),
        "scenario": [table_scenario or scenario] * len(strata),
        "method": [method] * len(strata),
        "method_version": ["1.0"] * len(strata),
        "resource": ["ConnectomeDB2020"] * len(strata),
        "ranking_semantics": [
            ranking_semantics or f"{analysis_unit}_ranking"
        ]
        * len(strata),
        "condition": [condition for condition, _ in strata],
        "top_fraction": [fraction for _, fraction in strata],
    }
    statuses = [
        "observed" if value is not None else "not_estimable" for value in values
    ]
    scores = pd.DataFrame(
        {
            **shared,
            "status": statuses,
            "reason_code": ["" if value is not None else "missing" for value in values],
            "metric": ["spatial_des"] * len(strata),
            "des": values,
        }
    )
    coverage = pd.DataFrame(
        {
            **shared,
            "status": statuses,
            "reason_code": ["" if value is not None else "missing" for value in values],
            "expected_set_available": [True] * len(strata),
            "expected_pair_coverage_fraction": expected_coverage,
            "rank_rows_total": [20] * len(strata),
            "rank_pairs_eligible": [15] * len(strata),
            "rank_eligible_fraction": [rank_coverage] * len(strata),
        }
    )
    score_path = root / "spatial_des_scores.tsv"
    coverage_path = root / "spatial_des_coverage.tsv"
    scores.to_csv(score_path, sep="\t", index=False, lineterminator="\n")
    coverage.to_csv(coverage_path, sep="\t", index=False, lineterminator="\n")
    manifest = {
        "schema_version": schema_version,
        "status": "complete",
        "scenario": scenario,
        "analysis_unit": analysis_unit,
        "expected_filters": {} if variant is None else {"variant": variant},
        "cell_pair_direction": "unordered_directions_collapsed",
        "score": "unweighted_fgsea_positive_running_sum_analogue",
        "inputs": {
            "expected_sets": {
                "filename": "truth.tsv",
                "sha256": expected_sha256,
            }
        },
        "outputs": {
            "scores": {
                "filename": score_path.name,
                "rows": len(scores),
                "sha256": _sha256(score_path),
            },
            "coverage": {
                "filename": coverage_path.name,
                "rows": len(coverage),
                "sha256": _sha256(coverage_path),
            },
        },
    }
    if tie_policy is not None:
        manifest["tie_policy"] = tie_policy
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_summary_ranks_complete_methods_and_reports_coverage(tmp_path: Path) -> None:
    first = _write_evaluation(
        tmp_path / "first",
        method="high",
        values=[0.8] * 8,
        rank_coverage=0.75,
        expected_coverage=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0],
    )
    second = _write_evaluation(
        tmp_path / "second",
        method="low",
        values=[0.2] * 8,
        rank_coverage=0.5,
    )

    summary, _, panels = summarize_evaluations([first.parent, second])

    assert len(panels) == 1
    result = summary.set_index("method")
    assert result.loc["high", "des_strata_observed"] == 8
    assert result.loc["high", "des_median"] == pytest.approx(0.8)
    assert result.loc["high", "des_mean"] == pytest.approx(0.8)
    assert result.loc["high", "median_rank"] == 1
    assert result.loc["low", "median_rank"] == 2
    assert result.loc["high", "mean_rank"] == 1
    assert result.loc["low", "mean_rank"] == 2
    assert bool(result.loc["high", "rank_eligible"])
    assert result.loc["high", "rank_eligible_fraction_min"] == pytest.approx(0.75)
    assert result.loc["high", "rank_eligible_fraction_mean"] == pytest.approx(0.75)
    assert result.loc["high", "expected_set_coverage_min"] == pytest.approx(0.5)
    assert result.loc["high", "expected_set_coverage_mean"] == pytest.approx(0.8125)


def test_incomplete_method_is_not_ranked_but_remains_reported(tmp_path: Path) -> None:
    manifest = _write_evaluation(
        tmp_path / "incomplete",
        method="partial",
        values=[0.5] * 7 + [None],
    )

    summary, _, _ = summarize_evaluations([manifest])

    row = summary.iloc[0]
    assert row["des_strata_observed"] == 7
    assert not bool(row["rank_eligible"])
    assert pd.isna(row["median_rank"])
    assert pd.isna(row["mean_rank"])
    assert row["des_median"] == pytest.approx(0.5)


def test_expected_checksum_unit_and_variant_form_separate_panels(
    tmp_path: Path,
) -> None:
    subject = _write_evaluation(
        tmp_path / "subject",
        method="same",
        values=[0.8] * 8,
        expected_sha256="a" * 64,
        analysis_unit="subject_id",
        variant="primary",
    )
    sample = _write_evaluation(
        tmp_path / "sample",
        method="same",
        values=[0.4] * 8,
        expected_sha256="b" * 64,
        analysis_unit="sample_id",
        variant="primary",
    )
    sensitivity = _write_evaluation(
        tmp_path / "sensitivity",
        method="same",
        values=[0.2] * 8,
        expected_sha256="a" * 64,
        analysis_unit="subject_id",
        variant="juxta_only",
    )

    summary, _, panels = summarize_evaluations([subject, sample, sensitivity])

    assert len(panels) == 3
    assert summary["comparison_panel_id"].nunique() == 3
    assert summary["median_rank"].tolist() == [1, 1, 1]
    assert set(summary["analysis_unit"]) == {"subject_id", "sample_id"}
    assert set(summary["expected_variant"]) == {"primary", "juxta_only"}


def test_tie_policies_form_separate_comparison_panels(tmp_path: Path) -> None:
    native = _write_evaluation(
        tmp_path / "native",
        method="same",
        values=[0.8] * 8,
        schema_version="crychic-spatial-des-evaluation-v2",
        tie_policy="fgsea_native",
    )
    simultaneous = _write_evaluation(
        tmp_path / "simultaneous",
        method="same",
        values=[0.4] * 8,
        schema_version="crychic-spatial-des-evaluation-v2",
        tie_policy="simultaneous",
    )

    summary, _, panels = summarize_evaluations([native, simultaneous])

    assert len(panels) == 2
    assert summary["comparison_panel_id"].nunique() == 2
    assert {panel["tie_policy"] for panel in panels.values()} == {
        "fgsea_native",
        "simultaneous",
    }


def test_explicit_condition_level_unit_overrides_campaign_path_hint(
    tmp_path: Path,
) -> None:
    manifest = _write_evaluation(
        tmp_path / "subject_condition_aware" / "method",
        method="method",
        values=[0.5] * 8,
        scenario="condition_aware",
        analysis_unit="condition_level",
    )

    summary, _, _ = summarize_evaluations([manifest])

    assert summary.loc[0, "analysis_unit"] == "condition_level"


@pytest.mark.parametrize("aggregation_marker", ["subject_equal", "subject-equal"])
def test_subject_equal_aggregation_does_not_override_condition_level_unit(
    tmp_path: Path, aggregation_marker: str
) -> None:
    manifest = _write_evaluation(
        tmp_path / aggregation_marker,
        method="crychic",
        values=[0.5] * 8,
        scenario="condition_aware",
        analysis_unit="condition_level",
        ranking_semantics=f"sum_positive_{aggregation_marker}_directed_lr_effects",
    )

    summary, bundles, _ = summarize_evaluations([manifest])

    assert summary.loc[0, "analysis_unit"] == "condition_level"
    assert bundles[0].analysis_unit_source == "evaluation_manifest"


@pytest.mark.parametrize(
    "ranking_semantics",
    [
        "subject_id_ranking",
        "sample_id_ranking",
        "subject_equal_subject_id_ranking",
    ],
)
def test_explicit_condition_level_rejects_true_unit_semantics(
    tmp_path: Path, ranking_semantics: str
) -> None:
    manifest = _write_evaluation(
        tmp_path / ranking_semantics,
        method="method",
        values=[0.5] * 8,
        scenario="condition_aware",
        analysis_unit="condition_level",
        ranking_semantics=ranking_semantics,
    )

    with pytest.raises(ValueError, match=r"explicit analysis_unit.*conflicts"):
        summarize_evaluations([manifest])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("scenario", "scenario disagrees"),
        ("fraction", "fractions"),
        ("checksum", "SHA256 mismatch"),
    ],
)
def test_mismatched_contracts_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest = _write_evaluation(
        tmp_path / mutation,
        method="method",
        values=[0.1] * 8,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "scenario":
        payload["scenario"] = "condition_aware"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "fraction":
        score_path = manifest.parent / payload["outputs"]["scores"]["filename"]
        scores = pd.read_csv(score_path, sep="\t")
        scores.loc[scores.index[-1], "top_fraction"] = 0.5
        scores.to_csv(score_path, sep="\t", index=False, lineterminator="\n")
        payload["outputs"]["scores"]["sha256"] = _sha256(score_path)
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    else:
        score_path = manifest.parent / payload["outputs"]["scores"]["filename"]
        score_path.write_text(score_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        summarize_evaluations([manifest])


def test_cli_writer_binds_inputs_and_output_checksum(tmp_path: Path) -> None:
    evaluation = _write_evaluation(
        tmp_path / "evaluation",
        method="method",
        values=list(np.linspace(0.1, 0.8, 8)),
    )
    output = tmp_path / "summary"

    manifest = run([evaluation.parent], output)

    summary_path = output / SUMMARY_FILENAME
    assert manifest["status"] == "complete"
    assert manifest["output"]["sha256"] == _sha256(summary_path)
    assert manifest["output"]["rows"] == 1
    assert manifest["protocol"]["no_cross_panel_ranking"] is True
    assert len(manifest["comparison_panels"]) == 1
    assert len(manifest["inputs"]["evaluation_manifest_aggregate_sha256"]) == 64
    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run([evaluation], output)


@pytest.mark.parametrize("mutation", ["boolean", "status"])
def test_coverage_contract_mismatch_fails_closed(tmp_path: Path, mutation: str) -> None:
    manifest = _write_evaluation(
        tmp_path / mutation,
        method="method",
        values=[0.1] * 8,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    coverage_path = manifest.parent / payload["outputs"]["coverage"]["filename"]
    coverage = pd.read_csv(coverage_path, sep="\t")
    if mutation == "boolean":
        coverage["expected_set_available"] = coverage["expected_set_available"].astype(
            object
        )
        coverage.loc[coverage.index[0], "expected_set_available"] = "invalid"
    else:
        coverage.loc[coverage.index[0], "status"] = "not_estimable"
    coverage.to_csv(coverage_path, sep="\t", index=False, lineterminator="\n")
    payload["outputs"]["coverage"]["sha256"] = _sha256(coverage_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"booleans|statuses disagree"):
        summarize_evaluations([manifest])
