from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.comprehensive.evaluate_three_group import (
    RunRecord,
    _active_metrics,
    _active_multigroup_metrics,
    _method_summary,
    _multigroup_method_summary,
    _multigroup_metrics,
    _target_crychic_view,
    _tie_inclusive_top_k,
    _validate_run_binding,
)


def test_context_invariant_crychic_view_serves_every_registered_contrast() -> None:
    table = pd.DataFrame(
        {
            "run_id": ["mechanistic", "other"],
            "score": [0.7, 0.2],
        }
    )
    manifest = {
        "source_result": {
            "score_views": [
                {
                    "run_id": "mechanistic",
                    "view_scope": "context_invariant_mechanistic_sample_score",
                    "primary_score": True,
                    "contrast_candidates": [],
                },
                {
                    "run_id": "other",
                    "view_scope": "single_scoring_functional",
                    "primary_score": False,
                    "contrast_candidates": [],
                },
            ]
        }
    }

    selected = _target_crychic_view(table, manifest, target="B")

    assert selected["run_id"].tolist() == ["mechanistic"]


def test_crychic_view_selection_rejects_ambiguous_invariant_primaries() -> None:
    table = pd.DataFrame({"run_id": ["first", "second"], "score": [0.7, 0.2]})
    manifest = {
        "source_result": {
            "score_views": [
                {
                    "run_id": run_id,
                    "view_scope": "context_invariant_mechanistic_sample_score",
                    "primary_score": True,
                    "contrast_candidates": [],
                }
                for run_id in ("first", "second")
            ]
        }
    }

    with pytest.raises(ValueError, match="context-invariant primary view"):
        _target_crychic_view(table, manifest, target="B")


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


def _decomposed_event_effects(
    *,
    dataset_id: str = "active-seed-1",
    scenario: str = "active",
    include_missing_null_event: bool = False,
) -> pd.DataFrame:
    truth = {
        "E1": {"A_vs_B": 1.0, "A_vs_C": 2.0, "B_vs_C": 0.0},
        "E2": {"A_vs_B": 0.0, "A_vs_C": -1.0, "B_vs_C": 1.0},
        "E3": {"A_vs_B": 0.0, "A_vs_C": 0.0, "B_vs_C": 0.0},
        "E4": {"A_vs_B": 0.0, "A_vs_C": 0.0, "B_vs_C": 0.0},
    }
    if include_missing_null_event:
        truth["E5"] = {"A_vs_B": 0.0, "A_vs_C": 0.0, "B_vs_C": 0.0}
    rows: list[dict[str, object]] = []
    for event, contrast_effects in truth.items():
        for contrast, truth_effect in contrast_effects.items():
            missing = include_missing_null_event and event == "E5"
            planted = 0.0 if scenario == "global_null" else truth_effect
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "scenario": scenario,
                    "seed": 1,
                    "method": "crychic",
                    "method_label": "CRYCHIC generic baseline",
                    "run_directory": "fixture-run",
                    "sender": "Sender",
                    "receiver": "Receiver",
                    "ligand": event,
                    "receptor": f"R{event}",
                    "contrast": contrast,
                    "truth_label": int(planted != 0.0),
                    "truth_effect": planted,
                    "truth_direction": int(np.sign(planted)),
                    "effect": np.nan if missing else planted,
                    "status": "not_estimable" if missing else "observed",
                }
            )
    return pd.DataFrame.from_records(rows)


def test_decomposed_metrics_separate_detection_localization_direction() -> None:
    metrics, confusion = _active_multigroup_metrics(_decomposed_event_effects())

    assert metrics["omnibus_status"] == "observed"
    assert metrics["omnibus_auprc"] == pytest.approx(1.0)
    assert metrics["omnibus_auroc"] == pytest.approx(1.0)
    assert metrics["omnibus_mcc_at_truth_k"] == pytest.approx(1.0)
    assert metrics["omnibus_precision_at_k"] == pytest.approx(1.0)
    assert metrics["omnibus_recall_at_k"] == pytest.approx(1.0)
    assert metrics["localization_macro_auprc"] == pytest.approx(1.0)
    assert metrics["localization_micro_auprc"] == pytest.approx(1.0)
    assert metrics["localization_hamming_loss"] == pytest.approx(0.0)
    assert metrics["localization_exact_set_accuracy"] == pytest.approx(1.0)
    assert metrics["positive_direction_ap"] == pytest.approx(1.0)
    assert metrics["negative_direction_ap"] == pytest.approx(1.0)
    assert metrics["direction_accuracy_all_active"] == pytest.approx(1.0)
    assert metrics["direction_accuracy_detected_active"] == pytest.approx(1.0)
    assert metrics["effect_rmse_native_scale"] == pytest.approx(0.0)
    assert metrics["effect_mae_native_scale"] == pytest.approx(0.0)
    assert metrics["effect_pearson"] == pytest.approx(1.0)
    assert metrics["effect_spearman"] == pytest.approx(1.0)
    assert metrics["top_effect_recovery"] == pytest.approx(1.0)
    assert metrics["ci_coverage_status"] == (
        "NE_no_comparable_confidence_intervals"
    )
    assert len(confusion) == 3
    assert all(record["false_positive"] == 0 for record in confusion)
    assert all(record["false_negative"] == 0 for record in confusion)


def test_decomposed_metrics_preserve_missing_events_and_report_coverage() -> None:
    metrics, _ = _active_multigroup_metrics(
        _decomposed_event_effects(include_missing_null_event=True)
    )

    assert metrics["event_coverage"] == pytest.approx(0.8)
    assert metrics["contrast_cell_coverage"] == pytest.approx(0.8)
    assert metrics["n_complete_events"] == 4
    assert metrics["n_observed_event_contrast_cells"] == 12
    assert metrics["omnibus_status"] == "observed"
    assert metrics["omnibus_auprc"] == pytest.approx(1.0)


def test_multigroup_tables_keep_null_diagnostics_and_rank_complete_methods() -> None:
    active = _decomposed_event_effects()
    null = _decomposed_event_effects(
        dataset_id="null-seed-1", scenario="global_null"
    )
    metrics, confusion = _multigroup_metrics(
        pd.concat((active, null), ignore_index=True)
    )
    summary = _multigroup_method_summary(metrics).set_index("method")

    assert len(metrics) == 2
    assert len(confusion) == 3
    null_row = metrics.loc[metrics["scenario"].eq("global_null")].iloc[0]
    assert null_row["effect_status"] == "observed_null_diagnostics"
    assert null_row["effect_all_zero_fraction"] == pytest.approx(1.0)
    assert null_row["omnibus_status"] == "NE"
    assert summary.loc["crychic", "primary_rank"] == 1
    assert bool(summary.loc["crychic", "rank_eligible"])
    assert pd.isna(summary.loc["cellchat", "primary_rank"])


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
