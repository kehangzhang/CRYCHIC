from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import sha256_file
from benchmarks.comprehensive.evaluate_g1 import (
    METRIC_IDS,
    METRIC_REGISTRY_PATH,
    average_precision,
    evaluate_g1_events,
    prevalence_adjusted_average_precision,
    run_evaluation,
    validate_event_table,
)


def test_output_metric_ids_are_registered_g1_metrics() -> None:
    registry = pd.read_csv(METRIC_REGISTRY_PATH, sep="\t")
    registered = set(registry["metric_id"])

    assert set(METRIC_IDS).issubset(registered)
    assert "native_ap" not in METRIC_IDS
    assert "coverage" not in METRIC_IDS


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["fixture"] * 5,
            "method_id": ["method"] * 5,
            "resource_mode": ["H-common"] * 5,
            "scenario_id": ["sparse"] * 5,
            "event_id": [f"event_{index}" for index in range(5)],
            "status": ["ok", "ok", "ok", "ok", "not_returned"],
            "truth_label": [1, 0, 1, 0, 1],
            "truth_effect": [2.0, -1.0, 0.5, -2.0, 100.0],
            "predicted_score": [0.9, 0.8, 0.7, 0.6, np.nan],
            "predicted_effect": [1.5, -0.5, 0.0, -1.5, np.nan],
        }
    )


def _by_metric(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.set_index("metric_id")
    assert result.index.is_unique
    return result


def test_target_prevalence_ap_uses_both_class_weights_and_tie_groups() -> None:
    labels = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])

    assert average_precision(labels, scores) == pytest.approx(5.0 / 6.0)
    # Positive item weight=0.1/2=0.05, negative item weight=0.9/2=0.45.
    # The two recall increments have adjusted precision 1 and 0.1/0.55.
    assert prevalence_adjusted_average_precision(
        labels, scores, target_prevalence=0.1
    ) == pytest.approx(0.5 + 0.5 * (0.1 / 0.55))

    tied = np.ones(4)
    assert prevalence_adjusted_average_precision(
        labels, tied, target_prevalence=0.1
    ) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="labels must be binary"):
        average_precision(np.asarray([0.5, 1.0]), np.asarray([0.0, 1.0]))


def test_evaluator_excludes_non_ok_rows_and_reports_coverage() -> None:
    metrics = _by_metric(
        evaluate_g1_events(_events(), target_prevalence=0.1, threshold=0.65)
    )

    assert tuple(metrics.index) == METRIC_IDS
    assert metrics.loc[
        "prevalence_adjusted_ap", "native_ap_diagnostic"
    ] == pytest.approx(5.0 / 6.0)
    assert metrics.loc["prevalence_adjusted_ap", "value"] == pytest.approx(
        0.5 + 0.5 * (0.1 / 0.55)
    )
    assert metrics.loc["differential_auroc", "value"] == pytest.approx(0.75)
    assert metrics.loc["differential_mcc", "value"] == pytest.approx(
        1.0 / math.sqrt(3.0)
    )
    assert metrics.loc["effect_rmse", "value"] == pytest.approx(0.5)
    assert metrics.loc["effect_spearman", "value"] == pytest.approx(1.0)
    assert metrics.loc["sign_accuracy", "value"] == pytest.approx(0.75)
    assert metrics.loc[
        "prevalence_adjusted_ap", "coverage_diagnostic"
    ] == pytest.approx(0.8)
    assert metrics.loc["prevalence_adjusted_ap", "n_status_ok"] == 4
    assert metrics.loc["prevalence_adjusted_ap", "n_used"] == 4
    assert metrics.loc["effect_rmse", "effect_coverage_diagnostic"] == pytest.approx(
        0.8
    )
    assert set(metrics["status"]) == {"ok"}


def test_single_class_and_invalid_effect_correlation_are_explicit_ne() -> None:
    events = _events().iloc[:3].copy()
    events["truth_label"] = 1
    events["predicted_effect"] = 0.0
    events["status"] = "ok"
    metrics = _by_metric(
        evaluate_g1_events(events, target_prevalence=0.2, threshold=0.5)
    )

    for metric_id in METRIC_IDS[:3]:
        assert metrics.loc[metric_id, "status"] == "NE"
        assert metrics.loc[metric_id, "reason_code"] == "single_class_truth"
        assert pd.isna(metrics.loc[metric_id, "value"])
    assert metrics.loc["effect_rmse", "status"] == "ok"
    assert metrics.loc["effect_spearman", "status"] == "NE"
    assert metrics.loc["effect_spearman", "reason_code"] == "constant_predicted_effect"
    assert metrics.loc["sign_accuracy", "value"] == pytest.approx(0.0)
    assert metrics.loc[
        "prevalence_adjusted_ap", "coverage_diagnostic"
    ] == pytest.approx(1.0)


def test_zero_truth_effects_make_sign_accuracy_ne_without_hiding_rmse() -> None:
    events = _events().iloc[:2].copy()
    events["status"] = "ok"
    events["truth_effect"] = 0.0
    events["predicted_effect"] = [0.0, 1.0]

    metrics = _by_metric(
        evaluate_g1_events(events, target_prevalence=0.5, threshold=0.5)
    )

    assert metrics.loc["effect_rmse", "value"] == pytest.approx(1 / math.sqrt(2))
    assert metrics.loc["effect_spearman", "status"] == "NE"
    assert metrics.loc["effect_spearman", "reason_code"] == "constant_truth_effect"
    assert metrics.loc["sign_accuracy", "status"] == "NE"
    assert metrics.loc["sign_accuracy", "reason_code"] == "no_nonzero_truth_effects"


def test_atomic_bundle_hashes_metrics_and_requires_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "events.tsv"
    with np.errstate(invalid="ignore"):
        _events().to_csv(source, sep="\t", index=False)
    output = tmp_path / "result"

    manifest = run_evaluation(
        input_path=source,
        output_dir=output,
        target_prevalence=0.1,
        threshold=0.65,
    )

    metrics_path = output / "metrics.tsv"
    manifest_path = output / "manifest.json"
    assert metrics_path.is_file()
    assert manifest_path.is_file()
    assert manifest["output"]["sha256"] == sha256_file(metrics_path)
    assert manifest["input"]["sha256"] == sha256_file(source)
    assert manifest["input"]["filename"] == source.name
    assert "path" not in manifest["input"]
    assert manifest["metrics"]["rows"] == len(METRIC_IDS)
    assert manifest["interpretation"]["non_ok_imputed_as_zero"] is False
    assert manifest["interpretation"]["calibration_metrics_computed"] is False
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted == manifest
    assert "NaN" not in manifest_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_evaluation(
            input_path=source,
            output_dir=output,
            target_prevalence=0.1,
            threshold=0.65,
        )
    replaced = run_evaluation(
        input_path=source,
        output_dir=output,
        target_prevalence=0.1,
        threshold=0.75,
        overwrite=True,
    )
    assert replaced["parameters"]["decision_threshold"] == pytest.approx(0.75)


def test_validation_rejects_duplicate_events_and_nonbinary_truth() -> None:
    duplicate = pd.concat([_events(), _events().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate group/event"):
        validate_event_table(duplicate)

    invalid = _events()
    invalid.loc[0, "truth_label"] = 2
    with pytest.raises(ValueError, match="truth_label must be binary"):
        validate_event_table(invalid)
