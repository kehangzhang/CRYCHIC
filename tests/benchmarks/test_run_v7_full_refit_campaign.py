from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.adapters.common import sha256_file
from benchmarks.simulation.run_v7_full_refit_campaign import (
    OUTPUT_TABLES,
    build_v7_full_refit_plan,
    load_v7_full_refit_frozen_config,
    run_v7_full_refit_dataset,
)
from benchmarks.simulation.v7_protocol import (
    AMENDMENT_CONFIG,
    load_v7_benchmark_protocol,
)


def _continuous_plan() -> dict[str, object]:
    return {
        "phase": "smoke",
        "family_role": "null_calibration",
        "dgp_family": "global_null",
        "design_kind": "continuous",
        "replicate_index": 1,
        "seed": 123,
        "dataset_id": "pr10-debug-bootstrap",
        "candidate_sender_count": 2,
        "cells_per_type": 1,
        "subjects_per_level": 4,
    }


def test_full_refit_plan_selects_unique_frozen_datasets() -> None:
    plan = build_v7_full_refit_plan(
        load_v7_benchmark_protocol(AMENDMENT_CONFIG),
        phase="smoke",
        maximum_replicates=1,
        dgp_families=("global_null",),
        design_kinds=("continuous",),
    )

    assert len(plan) == 1
    assert plan["dataset_id"].is_unique
    assert plan.loc[0, "dgp_family"] == "global_null"
    assert plan.loc[0, "design_kind"] == "continuous"


def test_pr10_smoke_config_authenticates_the_m4_protocol() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "configs"
        / "suggest_next2_v7_pr10_smoke_v1.json"
    )
    frozen = load_v7_full_refit_frozen_config(path)
    manifest = frozen.to_manifest()

    assert manifest["config_sha256"] == sha256_file(path)
    assert manifest["base_protocol_sha256"] == sha256_file(AMENDMENT_CONFIG)
    assert frozen.config["experiment"]["design_kinds"] == [
        "independent_two_group",
        "independent_multi_group",
        "paired",
        "repeated",
        "multi_cohort",
        "continuous",
    ]


def test_one_pr10_dataset_persists_resumes_and_keeps_formal_fields_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pr10"
    first = run_v7_full_refit_dataset(
        protocol_path=AMENDMENT_CONFIG,
        dataset_plan=_continuous_plan(),
        output_root=output,
        n_bootstraps=1,
        n_permutations=1,
        run_loso=False,
        resample_jobs=2,
    )

    assert first.status == "completed"
    assert first.planned_resamples == 2
    assert first.successful_resamples == 2
    result = Path(first.result_directory)
    manifest_path = result / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not manifest["formal_inference_allowed"]
    assert manifest["formal_inference_reason"] == "calibration_gate_missing"
    assert manifest["resampling"]["permutation_context_keys"] == [
        "condition",
        "dose",
    ]
    assert set(manifest["resampling"]["configured_rerun_stages_observed"]) == {
        "fold_split",
        "availability_fitting",
        "absolute_activity_v2_fitting",
        "signed_program_v2_fitting",
        "sender_attribution_v2_fitting",
        "eb_shrunken_coupling_v2_fitting",
        "heldout_score_application",
        "design_aware_effect_fitting",
        "two_part_occurrence_v2_fitting",
        "hypergraph_shrinkage_v2_fitting",
    }
    assert set(manifest["outputs"]) == set(OUTPUT_TABLES)
    for record in (
        *manifest["outputs"].values(),
        *manifest["resampling_artifacts"].values(),
    ):
        path = result / record["filename"]
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]
    occurrence = pd.read_parquet(result / "full_refit_occurrence_effects.parquet")
    assert set(occurrence["channel"]) == {"occurrence_log_odds"}
    assert not occurrence["formal_inference_allowed"].any()
    assert occurrence[["p_value", "q_value"]].isna().all(axis=None)

    with gzip.open(result / "resampling_records.jsonl.gz", "rt") as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == 2
    assert {record["status"] for record in records} == {"succeeded"}
    events = [
        json.loads(line)
        for line in (result / "run.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["stage"] == "full_refit_resample" for event in events) == 2

    before = manifest_path.stat().st_mtime_ns
    resumed = run_v7_full_refit_dataset(
        protocol_path=AMENDMENT_CONFIG,
        dataset_plan=_continuous_plan(),
        output_root=output,
        n_bootstraps=1,
        n_permutations=1,
        run_loso=False,
        resample_jobs=2,
    )
    assert resumed.manifest_sha256 == first.manifest_sha256
    assert manifest_path.stat().st_mtime_ns == before
    with pytest.raises(FileExistsError, match="different protocol or resampling"):
        run_v7_full_refit_dataset(
            protocol_path=AMENDMENT_CONFIG,
            dataset_plan=_continuous_plan(),
            output_root=output,
            n_bootstraps=2,
            n_permutations=1,
            run_loso=False,
            resample_jobs=2,
        )
