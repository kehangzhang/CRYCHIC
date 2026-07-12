from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import sha256_file
from benchmarks.metrics.track_b import (
    PROXY_SEMANTICS,
    TrackBPrepared,
    load_track_b_truth,
    prepare_track_b_long,
    run_track_b_suite,
    simulation_truth_table,
    summarize_track_b,
    track_b_edge_effects,
    track_b_stability,
)


def _track_b_long(*, independent: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    receivers = ("Receiver", "Bystander")
    ligands = ("CXCL10", "CCL5", "EGF")
    if independent:
        samples = [(f"R{i}:ctrl", f"R{i}", "ctrl") for i in range(1, 5)] + [
            (f"T{i}:stim", f"T{i}", "stim") for i in range(1, 5)
        ]
    else:
        samples = [
            (f"S{i}:{context}", f"S{i}", context)
            for i in range(1, 5)
            for context in ("ctrl", "stim")
        ]
    for sample_id, subject_id, context in samples:
        for receiver in receivers:
            for ligand in ligands:
                if receiver == "Receiver" and context == "ctrl":
                    score = {"CXCL10": 0.2, "CCL5": 0.5, "EGF": 0.8}[ligand]
                elif receiver == "Receiver":
                    score = {"CXCL10": 0.9, "CCL5": 0.5, "EGF": 0.1}[ligand]
                else:
                    score = {"CXCL10": 0.3, "CCL5": 0.6, "EGF": 0.9}[ligand]
                rows.append(
                    {
                        "run_id": "track-b-run",
                        "dataset_id": "synthetic_target_only",
                        "method_id": "nichenet_prior_activity",
                        "method_version": "proxy-v1",
                        "analysis_track": "ligand_target_program",
                        "resource_mode": "native",
                        "resource_id": "nichenet_prior",
                        "resource_version": "1",
                        "universe_id": "track-b-universe",
                        "universe_member": True,
                        "universe_size": 6,
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "context_json": json.dumps({"condition": context}),
                        "sender": "__source_agnostic__",
                        "receiver": receiver,
                        "interaction_id": f"{ligand}|program",
                        "interaction_direction": "ligand_to_target_program",
                        "ligand": ligand,
                        "target": "__weighted_target_program__",
                        "score": score,
                        "score_name": "ligand_target_program_proxy",
                        "score_direction": "higher",
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def _paired_analysis() -> tuple[TrackBPrepared, pd.DataFrame, pd.DataFrame]:
    prepared = prepare_track_b_long(_track_b_long(), context_key="condition")
    effects = track_b_edge_effects(
        prepared,
        reference="ctrl",
        target="stim",
        design="paired",
        min_support=4,
    )
    _, stability = track_b_stability(
        prepared,
        reference="ctrl",
        target="stim",
        design="paired",
        min_support=4,
        top_k=1,
    )
    return prepared, effects, stability


def test_prepare_ranks_ligands_within_sample_receiver() -> None:
    prepared = prepare_track_b_long(_track_b_long(), context_key="condition")
    selected = prepared.ranked.loc[
        prepared.ranked["sample_id"].eq("S1:stim")
        & prepared.ranked["receiver"].eq("Receiver")
    ].set_index("ligand")

    assert selected.loc["CXCL10", "comparison_rank"] == 1
    assert selected.loc["EGF", "comparison_rank"] == 3
    assert selected.loc["CXCL10", "comparison_strength"] == pytest.approx(1.0)
    assert prepared.identity["analysis_track"] == "ligand_target_program"


def test_paired_effect_reports_receiver_cxcl10_rank() -> None:
    _, effects, _ = _paired_analysis()
    receiver = effects.loc[effects["receiver"].eq("Receiver")].set_index("ligand")

    assert receiver.loc["CXCL10", "effect"] == pytest.approx(2 / 3)
    assert receiver.loc["CXCL10", "effect_rank"] == 1
    assert receiver.loc["CXCL10", "effect_percentile"] == pytest.approx(1.0)
    assert receiver.loc["CXCL10", "positive_direction_fraction"] == 1.0
    assert receiver.loc["CXCL10", "n_pairs"] == 4
    assert receiver.loc["EGF", "effect"] == pytest.approx(-2 / 3)


def test_paired_loso_stability_is_subject_level() -> None:
    prepared, _, _ = _paired_analysis()
    detail, summary = track_b_stability(
        prepared,
        reference="ctrl",
        target="stim",
        design="paired",
        min_support=4,
        top_k=1,
    )
    receiver = summary.loc[summary["receiver"].eq("Receiver")].iloc[0]

    assert receiver["stability_design"] == "paired_leave_one_subject_out"
    assert receiver["median_effect_spearman"] == pytest.approx(1.0)
    assert receiver["median_top_k_jaccard"] == pytest.approx(1.0)
    assert receiver["n_stability_folds_estimable"] == 4
    assert set(detail["held_subject"].dropna()) == {"S1", "S2", "S3", "S4"}


def test_unpaired_effect_and_split_half_use_disjoint_subjects() -> None:
    prepared = prepare_track_b_long(
        _track_b_long(independent=True), context_key="condition"
    )
    effects = track_b_edge_effects(
        prepared,
        reference="ctrl",
        target="stim",
        design="independent",
        min_support=4,
    )
    detail, summary = track_b_stability(
        prepared,
        reference="ctrl",
        target="stim",
        design="independent",
        min_support=4,
        top_k=1,
        n_repeats=8,
        random_seed=17,
    )
    cxcl10 = effects.loc[
        effects["receiver"].eq("Receiver") & effects["ligand"].eq("CXCL10")
    ].iloc[0]

    assert cxcl10["effect"] == pytest.approx(2 / 3)
    assert cxcl10["n_reference_subjects"] == 4
    assert cxcl10["n_target_subjects"] == 4
    assert len(detail.loc[detail["receiver"].eq("Receiver")]) == 8
    assert (
        summary.loc[summary["receiver"].eq("Receiver"), "status"].iloc[0] == "observed"
    )


def test_track_b_rejects_sender_claim() -> None:
    table = _track_b_long()
    table.loc[0, "sender"] = "Sender"

    with pytest.raises(ValueError, match="source-agnostic sender"):
        prepare_track_b_long(table, context_key="condition")


def _truth_tables(tmp_path: Path) -> tuple[Path, Path]:
    track_b = tmp_path / "track_b_truth.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "synthetic_target_only",
                "scenario": "target_only",
                "contrast": "stim_vs_ctrl",
                "expected_receiver_response": True,
                "truth_scope": "simulation_scenario_level",
                "metric_scope": "ligand_target_program",
                "lr_edge_truth_available": False,
                "reason_code": "track_b_is_not_lr_truth",
            }
        ]
    ).to_csv(track_b, sep="\t", index=False)
    integrated = tmp_path / "integrated_truth.tsv"
    pd.DataFrame(
        [
            {
                "dataset_id": "synthetic_target_only",
                "scenario": "target_only",
                "expected_integrated_edge": False,
            }
        ]
    ).to_csv(integrated, sep="\t", index=False)
    return track_b, integrated


def test_target_only_scope_diagnostic_and_report_truth(tmp_path: Path) -> None:
    _, effects, stability = _paired_analysis()
    track_b_path, integrated_path = _truth_tables(tmp_path)
    truth = load_track_b_truth(track_b_path, integrated_truth_path=integrated_path)
    summary = summarize_track_b(
        effects,
        stability,
        scenario="target_only",
        receiver_of_interest="Receiver",
        truth=truth,
    )
    row = summary.iloc[0]

    assert row["receiver_program_positive"]
    assert row["expected_receiver_response_recovered"]
    assert not row["expected_integrated_edge"]
    assert row["scope_diagnostic"] == (
        "program_positive_integrated_edge_negative_as_expected"
    )
    assert row["proxy_semantics"] == PROXY_SEMANTICS
    report = simulation_truth_table(summary)
    assert set(report["truth_scope"]) == {"simulation"}
    assert "receiver_response_recovered" in set(report["metric"])


def test_all_nonpositive_receiver_effects_have_no_positive_rank() -> None:
    _, effects, stability = _paired_analysis()
    receiver = effects["receiver"].eq("Receiver")
    effects.loc[receiver, "effect"] = -effects.loc[receiver, "effect"].abs() - 0.1
    effects.loc[receiver, "positive_effect_rank"] = np.nan
    summary = summarize_track_b(
        effects,
        stability,
        receiver_of_interest="Receiver",
    )
    row = summary.iloc[0]

    assert row["all_receiver_program_effects_nonpositive"]
    assert row["positive_rank_status"] == "not_estimable"
    assert row["positive_rank_reason_code"] == (
        "all_receiver_program_effects_nonpositive"
    )


def test_suite_cli_artifacts_require_resolved_transform(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    table_path = input_dir / "interactions_long.parquet"
    _track_b_long().to_parquet(table_path, index=False)
    table_sha = sha256_file(table_path)
    manifest = {
        "status": "complete",
        "method": {
            "id": "nichenet_prior_activity",
            "native_nichenet_claim": False,
        },
        "output": {
            "table": table_path.name,
            "sha256": table_sha,
            "rows": len(_track_b_long()),
        },
        "parameters": {
            "resolved_expression_transform": "input_continuous_expression_preserved"
        },
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    track_b_truth, integrated_truth = _truth_tables(tmp_path)
    spec = {
        "schema_version": "crychic-track-b-suite-v1",
        "track_b_truth": track_b_truth.name,
        "integrated_truth": integrated_truth.name,
        "top_k": 1,
        "n_repeats": 8,
        "random_seed": 9,
        "require_resolved_expression_transform": True,
        "inputs": [
            {
                "dataset": "synthetic_target_only",
                "scenario": "target_only",
                "receiver_of_interest": "Receiver",
                "path": str(table_path.relative_to(tmp_path)),
                "context_key": "condition",
                "reference": "ctrl",
                "target": "stim",
                "design": "paired",
                "min_support": 4,
            }
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "output"

    result = run_track_b_suite(spec_path, output, repo_root=tmp_path, overwrite=False)

    assert result["schema_version"] == "crychic-track-b-metrics-v1"
    assert (output / "track_b_summary.tsv").is_file()
    assert (output / "track_b_detail.tsv").is_file()
    assert (output / "track_b_effects.tsv").is_file()
    assert (output / "track_b_stability.tsv").is_file()
    assert (output / "track_b_simulation_truth.tsv").is_file()
    persisted = pd.read_csv(output / "track_b_summary.tsv", sep="\t")
    assert persisted.loc[0, "scope_diagnostic_consistent"]
