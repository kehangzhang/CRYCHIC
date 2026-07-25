from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmarks.adapters.common import sha256_file
from benchmarks.simulation.run_v7_campaign import (
    DATASET_TABLES,
    build_v7_campaign_plan,
    build_v7_diagnostic_truth,
    run_v7_campaign,
)
from benchmarks.simulation.run_v7_e5_derived_campaign import (
    OUTPUT_TABLES as E5_DERIVED_TABLES,
)
from benchmarks.simulation.run_v7_e5_derived_campaign import run_derived_campaign
from benchmarks.simulation.v7_dgp import generate_v7_dgp
from benchmarks.simulation.v7_protocol import (
    AMENDMENT_SCHEMA_VERSION,
    load_v7_benchmark_protocol,
)


def test_campaign_plan_unions_profiles_without_recomputing_datasets() -> None:
    protocol = load_v7_benchmark_protocol()
    plan = build_v7_campaign_plan(
        protocol,
        phase="smoke",
        profiles=("score_primary", "inference_crossover"),
        maximum_replicates=1,
        maximum_datasets=2,
    )

    assert plan["dataset_id"].nunique() == 2
    assert len(plan) == 2 * (6 + 3)
    assert plan["run_id"].is_unique
    grouped = plan.groupby("dataset_id", observed=True)
    assert grouped["seed"].nunique().eq(1).all()
    assert grouped["candidate_sender_count"].nunique().eq(1).all()
    for _, rows in grouped:
        unique_arms = set(
            rows.loc[:, ["generator_id", "inference_id"]].itertuples(
                index=False, name=None
            )
        )
        assert len(unique_arms) == 8
        assert ("G3", "I1") in unique_arms


def test_continuous_truth_maps_only_for_the_descriptive_crossfit_diagnostic() -> None:
    fixture = generate_v7_dgp(
        dataset_id="continuous-diagnostic",
        dgp_family="global_null",
        design_kind="continuous",
        seed=22,
        candidate_sender_count=2,
        cells_per_type=1,
        subjects_per_level=4,
    )
    mapped = build_v7_diagnostic_truth(
        fixture.truth,
        contrast_names=("B_vs_A",),
    )

    assert set(fixture.truth["contrast_name"]) == {"slope:dose"}
    assert set(mapped["contrast_name"]) == {"B_vs_A"}
    assert mapped["truth"].equals(fixture.truth["truth_causal_sender"])


def test_one_dataset_campaign_persists_checksums_logs_and_resumes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    first = run_v7_campaign(
        phase="smoke",
        profiles=("score_primary", "inference_crossover"),
        output_root=output,
        maximum_replicates=1,
        maximum_datasets=1,
        jobs=1,
        allow_dirty=True,
    )
    assert first["status"] == "completed"
    assert first["datasets"] == 1
    assert first["completed"] == 1
    runs = pd.read_csv(output / "runs.tsv", sep="\t")
    assert runs["status"].tolist() == ["completed"]
    result = Path(runs.loc[0, "result_directory"])
    manifest_path = result / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["protocol"]["schema_version"] == AMENDMENT_SCHEMA_VERSION
    assert manifest["formal_inference_allowed"] is False
    assert manifest["maximum_system_memory_fraction"] < 0.8
    cache = manifest["inference_fit_cache"]
    assert cache["requests"] == cache["hits"] + cache["misses"]
    assert cache["entries"] == cache["misses"]
    assert cache["hits"] > 0
    assert set(manifest["experiments"]) == {
        "E2_hard_gate_attrition",
        "E3_sender_detection_attribution",
        "E5_hypergraph",
        "E6_m4_occurrence",
    }
    assert manifest["experiments"]["E6_m4_occurrence"][
        "fixed_threshold_formal_inference_allowed"
    ]
    assert not manifest["experiments"]["E6_m4_occurrence"][
        "release_calibration_complete"
    ]
    assert set(manifest["outputs"]) == set(DATASET_TABLES)
    for record in manifest["outputs"].values():
        path = result / record["filename"]
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]
    events = [
        json.loads(line)
        for line in (result / "run.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    completed_stages = {
        event["stage"] for event in events if event["event"] == "completed"
    }
    assert {
        "generate_raw_dgp",
        "subject_crossfit",
        "integrated_score_inference_matrix",
        "truth_metrics",
        "m4_occurrence_prevalence",
        "e2_hard_gate_component_swaps",
        "e3_sender_detection_attribution_swaps",
        "e5_hypergraph_topology_swaps",
        "v7_diagnostics",
        "persist_outputs",
    } == completed_stages
    for aggregate in (
        "all_e2_metrics.parquet",
        "all_e3_metrics.parquet",
        "all_e3_sender_metrics.parquet",
        "all_e5_metrics.parquet",
        "all_e5_fit_diagnostics.parquet",
        "all_e5_topology_diagnostics.parquet",
        "all_m4_metrics.parquet",
        "all_m4_baselines.parquet",
        "e2_metric_summary.tsv",
        "e3_metric_summary.tsv",
        "e3_sender_metric_summary.tsv",
        "e5_metric_summary.tsv",
        "m4_metric_summary.tsv",
    ):
        assert (output / aggregate).is_file()

    derived_output = tmp_path / "e5-derived"
    derived = run_derived_campaign(output, derived_output, jobs=1)
    assert derived["status"] == "completed"
    derived_runs = pd.read_csv(derived_output / "runs.tsv", sep="\t")
    assert derived_runs["status"].tolist() == ["completed"]
    derived_result = Path(derived_runs.loc[0, "result_directory"])
    derived_manifest_path = derived_result / "manifest.json"
    derived_manifest = json.loads(derived_manifest_path.read_text(encoding="utf-8"))
    assert set(derived_manifest["outputs"]) == set(E5_DERIVED_TABLES)
    assert not derived_manifest["formal_inference_allowed"]
    derived_events = [
        json.loads(line)
        for line in (derived_result / "run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {
        event["stage"] for event in derived_events if event["event"] == "completed"
    } == {
        "read_authenticated_source",
        "e5_hypergraph_topology_swaps",
        "persist_outputs",
    }
    derived_before = derived_manifest_path.stat().st_mtime_ns
    derived_resumed = run_derived_campaign(output, derived_output, jobs=1)
    assert derived_resumed["status"] == "completed"
    assert derived_manifest_path.stat().st_mtime_ns == derived_before

    before = manifest_path.stat().st_mtime_ns
    resumed = run_v7_campaign(
        phase="smoke",
        profiles=("score_primary", "inference_crossover"),
        output_root=output,
        maximum_replicates=1,
        maximum_datasets=1,
        jobs=1,
        allow_dirty=True,
    )
    assert resumed["status"] == "completed"
    assert manifest_path.stat().st_mtime_ns == before
