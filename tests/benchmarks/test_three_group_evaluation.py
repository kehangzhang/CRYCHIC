from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.comprehensive.evaluate_three_group import (
    RunRecord,
    _active_metrics,
    _method_summary,
    _tie_inclusive_top_k,
    _validate_run_binding,
)


def test_tie_inclusive_top_k_expands_boundary() -> None:
    selected = _tie_inclusive_top_k(pd.Series([0.9, 0.8, 0.8, 0.1]), 2)

    assert selected.tolist() == [True, True, True, False]


def test_active_metrics_score_detection_direction_and_ties() -> None:
    table = pd.DataFrame(
        {
            "truth_label": [1, 0, 1, 0],
            "truth_effect": [2.0, 0.0, -1.0, 0.0],
            "truth_direction": [1, 0, -1, 0],
            "effect": [0.9, 0.1, -0.8, 0.1],
            "status": ["observed"] * 4,
            "p_value": [np.nan] * 4,
        }
    )

    metrics = _active_metrics(table)

    assert metrics["metric_status"] == "observed"
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["prevalence_adjusted_ap"] == pytest.approx(1.0)
    assert metrics["top_k_recall"] == pytest.approx(1.0)
    assert metrics["direction_accuracy"] == pytest.approx(1.0)


def test_method_summary_ranks_only_complete_methods() -> None:
    rows = []
    for method, values in (("crychic", [0.8, 0.6]), ("cellchat", [0.5, np.nan])):
        for index, value in enumerate(values):
            rows.append(
                {
                    "method": method,
                    "dataset_id": f"d{index}",
                    "scenario": "active",
                    "metric_status": "observed" if np.isfinite(value) else "NE",
                    "contrast": f"c{index}",
                    "auprc": value,
                    "prevalence_adjusted_ap": value,
                    "auroc": value,
                    "top_k_recall": value,
                    "direction_accuracy": value,
                    "effect_spearman": value,
                    "coverage": 1.0,
                }
            )

    summary = _method_summary(pd.DataFrame(rows)).set_index("method")

    assert summary.loc["crychic", "primary_rank"] == 1
    assert not bool(summary.loc["cellchat", "rank_eligible"])
    assert pd.isna(summary.loc["cellchat", "primary_rank"])


def test_method_summary_uses_average_tie_ranks_and_frozen_expected_count() -> None:
    rows = []
    for method in ("crychic", "cellchat", "liana_rank_aggregate"):
        rows.append(
            {
                "method": method,
                "dataset_id": "d0",
                "scenario": "active",
                "metric_status": "observed",
                "contrast": "B_vs_A",
                "auprc": 0.2,
                "prevalence_adjusted_ap": 0.2,
                "auroc": 0.5,
                "top_k_recall": 0.0,
                "direction_accuracy": 0.5,
                "effect_spearman": 0.0,
                "coverage": 1.0,
            }
        )

    summary = _method_summary(
        pd.DataFrame(rows), expected_active_contrasts=2
    ).set_index("method")

    assert not summary["rank_eligible"].any()
    assert summary["primary_rank"].isna().all()


def test_active_metrics_is_stable_to_tsv_scale_float_perturbations() -> None:
    table = pd.DataFrame(
        {
            "truth_label": [1, 0, 0],
            "truth_effect": [1.0, 0.0, 0.0],
            "truth_direction": [1, 0, 0],
            "effect": [0.2, 0.1, 0.1 + 5e-17],
            "status": ["observed"] * 3,
            "p_value": [np.nan] * 3,
        }
    )

    baseline = _active_metrics(table)
    table.loc[2, "effect"] = 0.1 - 5e-17
    observed = _active_metrics(table)

    assert observed["prevalence_adjusted_ap"] == baseline["prevalence_adjusted_ap"]
    assert observed["auroc"] == baseline["auroc"]


def _external_record(method: str, manifest: dict[str, object]) -> RunRecord:
    return RunRecord(
        method=method,
        dataset_id="d1",
        contrast=None,
        directory=Path("run"),
        manifest=manifest,
        kind="external",
    )


def test_run_binding_rejects_wrong_cellchat_layer_and_version() -> None:
    manifest: dict[str, object] = {
        "method": {"id": "cellchat", "version": "2.1.2"},
        "input": {"sha256": "input"},
        "resource": {"mode": "H-common", "payload_sha256": "resource"},
        "code": {"commit": "commit", "dirty": False},
        "parameters": {
            "layer": None,
            "nboot": 100,
            "min_cells": 10,
            "sample_level_execution": True,
        },
    }
    record = _external_record("cellchat", manifest)

    _validate_run_binding(
        record,
        expected_input_sha256="input",
        expected_resource_sha256="resource",
        expected_code_commit="commit",
    )
    manifest["parameters"]["layer"] = "counts"  # type: ignore[index]
    with pytest.raises(ValueError, match="CellChat simulation protocol"):
        _validate_run_binding(
            record,
            expected_input_sha256="input",
            expected_resource_sha256="resource",
            expected_code_commit="commit",
        )
    manifest["parameters"]["layer"] = None  # type: ignore[index]
    manifest["method"]["version"] = "2.2.0"  # type: ignore[index]
    with pytest.raises(ValueError, match="method version"):
        _validate_run_binding(
            record,
            expected_input_sha256="input",
            expected_resource_sha256="resource",
            expected_code_commit="commit",
        )


def test_run_binding_requires_crychic_algorithm_commit_identity() -> None:
    manifest: dict[str, object] = {
        "method": {
            "id": "crychic",
            "version": "source-tree",
            "benchmark_identity": "generic_multigroup_baseline",
            "entrypoint": "benchmarks.adapters.crychic.run_hcommon",
            "algorithm_code": {"commit": "other", "dirty": False},
        },
        "input": {"sha256": "input"},
        "resource": {"mode": "H-common", "table_sha256": "resource"},
        "code": {"commit": "commit", "dirty": False},
        "parameters": {
            "benchmark_scope": "independent_subject_three_group_hcommon",
            "input_mode": "counts",
            "counts_layer": "counts",
            "workflow": {"subject_fixed_effects": False, "min_cells": 10},
        },
    }

    with pytest.raises(ValueError, match="generic baseline provenance mismatch"):
        _validate_run_binding(
            _external_record("crychic", manifest),
            expected_input_sha256="input",
            expected_resource_sha256="resource",
            expected_code_commit="commit",
        )
