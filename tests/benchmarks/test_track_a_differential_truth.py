from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import sha256_file
from benchmarks.metrics.track_a_differential_truth import (
    NEGATIVE_CONTROL_DIAGNOSTIC,
    PRIMARY_ESTIMAND,
    SENSITIVITY_ESTIMAND,
    SINGLE_CLASS_REASON,
    merge_simulation_records,
    paired_edge_differences,
    prepare_estimands,
    run_track_a_differential_suite,
    select_crychic_run_id,
    simulation_truth_records,
    summarize_differential_truth,
)

EDGES = (
    ("Sender", "Receiver", "known", "CXCL10", "CXCR3"),
    ("Sender", "Receiver", "other-1", "CCL5", "CCR5"),
    ("Bystander", "Receiver", "other-2", "EGF", "EGFR"),
)
KNOWN_EDGE = {
    "sender": "Sender",
    "receiver": "Receiver",
    "interaction_id": "known",
    "ligand": "CXCL10",
    "receptor": "CXCR3",
}


def _external_long(*, omit_known_reference: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scores = {
        "ctrl": {"known": 0.8, "other-1": 0.5, "other-2": 0.2},
        "stim": {"known": 0.1, "other-1": 0.5, "other-2": 0.9},
    }
    for subject in ("S1", "S2", "S3", "S4"):
        for context in ("ctrl", "stim"):
            for sender, receiver, interaction, ligand, receptor in EDGES:
                omitted = (
                    omit_known_reference
                    and context == "ctrl"
                    and interaction == "known"
                )
                rows.append(
                    {
                        "run_id": "external-run",
                        "dataset_id": "synthetic_active",
                        "method_id": "cellphonedb",
                        "method_version": "test-v1",
                        "analysis_track": "lr_stlr",
                        "resource_mode": "H-common",
                        "resource_id": "fixture",
                        "resource_version": "1",
                        "universe_id": "fixture-universe",
                        "universe_member": True,
                        "universe_size": len(EDGES),
                        "sample_id": f"{subject}:{context}",
                        "subject_id": subject,
                        "context_json": json.dumps({"condition": context}),
                        "sender": sender,
                        "receiver": receiver,
                        "interaction_id": interaction,
                        "ligand": ligand,
                        "receptor": receptor,
                        "score": (
                            math.nan if omitted else scores[context][interaction]
                        ),
                        "score_name": "lower_is_better_fixture",
                        "score_direction": "lower",
                        "status": "not_returned" if omitted else "ok",
                    }
                )
    return pd.DataFrame(rows)


def _truth(*, positive: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "synthetic_active",
                "contrast": "stim_vs_ctrl",
                "universe_id": "fixture-universe",
                "sender": sender,
                "receiver": receiver,
                "interaction_id": interaction,
                "ligand": ligand,
                "receptor": receptor,
                "is_positive": int(positive and interaction == "known"),
                "truth_scope": "simulation",
                "truth_status": "estimable" if positive else "single_class",
                "reason_code": None if positive else SINGLE_CLASS_REASON,
            }
            for sender, receiver, interaction, ligand, receptor in EDGES
        ]
    )


def _edge_tables(
    *, omit_known_reference: bool = False, positive: bool = True
) -> dict[str, pd.DataFrame]:
    _, prepared = prepare_estimands(
        _external_long(omit_known_reference=omit_known_reference),
        context_key="condition",
        contrast="stim_vs_ctrl",
    )
    return {
        item.estimand: paired_edge_differences(
            item.subject_context,
            _truth(positive=positive),
            reference="ctrl",
            target="stim",
            min_pairs=4,
            estimand=item.estimand,
        )
        for item in prepared
    }


def test_lower_native_score_is_oriented_before_paired_difference() -> None:
    edges = _edge_tables()[PRIMARY_ESTIMAND].set_index("interaction_id")

    assert edges.loc["known", "effect"] == pytest.approx(0.7)
    assert edges.loc["known", "effect_rank"] == 1
    assert edges.loc["known", "positive_direction_fraction"] == 1
    assert edges.loc["other-2", "effect"] == pytest.approx(-0.7)


def test_not_returned_is_not_native_zero_but_enters_rank_sensitivity() -> None:
    edges = _edge_tables(omit_known_reference=True)
    native = edges[PRIMARY_ESTIMAND].set_index("interaction_id").loc["known"]
    rank = edges[SENSITIVITY_ESTIMAND].set_index("interaction_id").loc["known"]

    assert native["status"] == "not_estimable"
    assert native["n_pairs"] == 0
    assert math.isnan(float(native["effect"]))
    assert rank["status"] == "observed"
    assert rank["n_pairs"] == 4
    assert rank["effect"] == pytest.approx(1.0)


def test_active_truth_metrics_rank_known_edge_first() -> None:
    edges = _edge_tables()[PRIMARY_ESTIMAND]
    summary = summarize_differential_truth(edges, top_k=1, known_edge=KNOWN_EDGE)

    assert summary["status"] == "observed"
    assert summary["auroc"] == pytest.approx(1.0)
    assert summary["average_precision"] == pytest.approx(1.0)
    assert summary["top_k_precision"] == pytest.approx(1.0)
    assert summary["known_edge_effect_rank"] == pytest.approx(1.0)


def test_all_zero_truth_is_explicit_single_class_with_known_edge_diagnostic() -> None:
    edges = _edge_tables(positive=False)[PRIMARY_ESTIMAND]
    summary = summarize_differential_truth(edges, top_k=1, known_edge=KNOWN_EDGE)

    assert summary["status"] == "not_estimable"
    assert summary["reason_code"] == SINGLE_CLASS_REASON
    assert math.isnan(float(summary["auroc"]))
    assert math.isnan(float(summary["average_precision"]))
    assert summary["known_edge_status"] == "observed"
    assert summary["known_edge_effect"] == pytest.approx(0.7)


def test_crychic_view_selection_uses_manifest_provenance() -> None:
    manifest = {
        "source_result": {
            "score_views": [
                {"contrast_candidates": ["global:'stim'"], "run_id": "stim-run"},
                {"contrast_candidates": ["global:'ctrl'"], "run_id": "ctrl-run"},
            ]
        }
    }

    assert (
        select_crychic_run_id(manifest, contrast_candidate="global:'stim'")
        == "stim-run"
    )
    manifest["source_result"]["score_views"].append(  # type: ignore[index]
        {"contrast_candidates": ["global:'stim'"], "run_id": "ambiguous"}
    )
    with pytest.raises(ValueError, match="exactly one"):
        select_crychic_run_id(manifest, contrast_candidate="global:'stim'")


def _summary_row(*, scenario: str, positive: bool) -> dict[str, object]:
    edges = _edge_tables(positive=positive)[PRIMARY_ESTIMAND]
    metrics = summarize_differential_truth(edges, top_k=1, known_edge=KNOWN_EDGE)
    return {
        "dataset": f"synthetic_{scenario}",
        "scenario": scenario,
        "method": "cellphonedb",
        "method_version": "test-v1",
        "analysis_track": "lr_stlr",
        "resource": "fixture",
        "resource_version": "1",
        "resource_mode": "H-common",
        "score_semantics": "lower_is_better_fixture",
        "score_direction": "lower",
        "universe_id": "fixture-universe",
        "estimand": PRIMARY_ESTIMAND,
        "missing_policy": "paired_observed_native_scores_only_no_zero_imputation",
        "scale_comparability": "native_score_scale_not_cross_method_comparable",
        **metrics,
    }


def test_simulation_records_do_not_invent_negative_control_error_rates() -> None:
    records = simulation_truth_records(
        pd.DataFrame(
            [
                _summary_row(scenario="active", positive=True),
                _summary_row(scenario="global_null", positive=False),
            ]
        )
    )
    negative = records.loc[records["scenario"].eq("global_null")]
    classification = negative.loc[negative["metric"].eq("differential_auroc")].iloc[0]
    known = negative.loc[negative["metric"].eq("known_cxcl10_effect")].iloc[0]

    assert classification["status"] == "not_estimable"
    assert classification["reason_code"] == SINGLE_CLASS_REASON
    assert pd.isna(classification["estimate"])
    assert known["status"] == "observed"
    assert known["reason_code"] == NEGATIVE_CONTROL_DIAGNOSTIC
    assert not any(
        token in metric.lower()
        for metric in records["metric"].astype(str)
        for token in ("fdr", "type_i", "p_value", "q_value")
    )


def test_top_k_f1_is_zero_when_precision_and_recall_are_zero() -> None:
    edges = _edge_tables(positive=True)[PRIMARY_ESTIMAND].copy()
    positive = edges["is_positive"].eq(1)
    edges.loc[positive, "effect"] = -1.0
    edges.loc[~positive, "effect"] = 1.0
    edges["effect_rank"] = edges["effect"].rank(
        ascending=False, method="average"
    )

    metrics = summarize_differential_truth(edges, top_k=1, known_edge=KNOWN_EDGE)

    assert metrics["top_k_precision"] == 0.0
    assert metrics["top_k_recall"] == 0.0
    assert metrics["top_k_f1"] == 0.0


def test_merge_records_preserves_finalizer_contract() -> None:
    track_a = simulation_truth_records(
        pd.DataFrame([_summary_row(scenario="active", positive=True)])
    )
    track_b = pd.DataFrame(
        [
            {
                "truth_scope": "simulation",
                "method": "nichenet_prior_activity",
                "metric": "receiver_response_recovered",
                "estimate": 1.0,
                "status": "observed",
                "reason_code": None,
            }
        ]
    )
    merged = merge_simulation_records(track_a, track_b)

    assert set(merged["record_source"]) == {
        "track_a_differential_truth",
        "track_b_receiver_program_truth",
    }
    assert set(merged["truth_scope"]) == {"simulation"}


def test_suite_cli_writes_two_estimands_and_combined_records(tmp_path: Path) -> None:
    input_dir = tmp_path / "suite" / "active" / "cellphonedb"
    input_dir.mkdir(parents=True)
    table_path = input_dir / "interactions_long.parquet"
    _external_long().to_parquet(table_path, index=False)
    manifest = {
        "status": "complete",
        "method": {"id": "cellphonedb"},
        "output": {
            "table": table_path.name,
            "sha256": sha256_file(table_path),
            "rows": len(_external_long()),
        },
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    truth_path = tmp_path / "truth.tsv"
    _truth().to_csv(truth_path, sep="\t", index=False)
    track_b_path = tmp_path / "track_b.tsv"
    pd.DataFrame(
        [
            {
                "truth_scope": "simulation",
                "method": "nichenet_prior_activity",
                "metric": "receiver_response_recovered",
                "estimate": 1.0,
                "status": "observed",
                "reason_code": None,
            }
        ]
    ).to_csv(track_b_path, sep="\t", index=False)
    specification = {
        "schema_version": "crychic-track-a-differential-suite-v1",
        "suite_root": "suite",
        "track_a_truth": truth_path.name,
        "track_b_simulation_records": track_b_path.name,
        "context_key": "condition",
        "reference": "ctrl",
        "target": "stim",
        "contrast": "stim_vs_ctrl",
        "min_pairs": 4,
        "top_k": 1,
        "truth_universe_size": len(EDGES),
        "positive_scenario": "active",
        "known_edge": KNOWN_EDGE,
        "methods": ["cellphonedb"],
        "scenarios": [{"scenario": "active", "dataset": "synthetic_active"}],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specification), encoding="utf-8")
    output = tmp_path / "output"

    result = run_track_a_differential_suite(
        spec_path, output, repo_root=tmp_path, overwrite=False
    )

    assert result["schema_version"] == "crychic-track-a-differential-metrics-v1"
    summary = pd.read_csv(output / "track_a_differential_summary.tsv", sep="\t")
    combined = pd.read_csv(output / "simulation_records.tsv", sep="\t")
    assert set(summary["estimand"]) == {PRIMARY_ESTIMAND, SENSITIVITY_ESTIMAND}
    assert len(summary) == 2
    assert set(combined["record_source"]) == {
        "track_a_differential_truth",
        "track_b_receiver_program_truth",
    }
    assert (output / "track_a_differential_report.md").is_file()
    assert (output / "manifest.json").is_file()


def test_known_edge_matching_rejects_wrong_identity() -> None:
    edges = _edge_tables()[PRIMARY_ESTIMAND]
    wrong = dict(KNOWN_EDGE)
    wrong["interaction_id"] = "absent"

    with pytest.raises(ValueError, match="exactly one"):
        summarize_differential_truth(edges, top_k=1, known_edge=wrong)


def test_single_class_reason_is_not_numeric_zero() -> None:
    record = simulation_truth_records(
        pd.DataFrame([_summary_row(scenario="global_null", positive=False)])
    ).loc[lambda table: table["metric"].eq("differential_average_precision")]

    assert np.isnan(record["estimate"].iloc[0])
    assert record["status"].iloc[0] == "not_estimable"
