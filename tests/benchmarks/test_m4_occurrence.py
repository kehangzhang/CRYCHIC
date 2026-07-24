from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmarks.comprehensive.evaluate_m4_occurrence import (
    M0_METHOD,
    M4_METHOD,
    RAW_METHOD,
    _calibration_metrics,
    _evaluate_dataset,
    _method_summary,
    _seed_metrics,
)
from benchmarks.comprehensive.generate_m4_occurrence_fixture import (
    FAMILY_COUNTS,
    SCHEMA_VERSION,
    _dataset_seed,
    generate,
)
from crychic.inference import OccurrenceContrastSpec


def test_m4_generator_is_reproducible_and_keeps_truth_separate(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = generate(first, seeds=(20280101,), n_subjects_per_group=8)
    second_manifest = generate(second, seeds=(20280101,), n_subjects_per_group=8)

    assert first_manifest["schema_version"] == SCHEMA_VERSION
    assert first_manifest["records"][0]["events"] == sum(FAMILY_COUNTS.values())
    assert (
        first_manifest["records"][0]["input_sha256"]
        == second_manifest["records"][0]["input_sha256"]
    )
    record = first_manifest["records"][0]
    assert all(family not in record["dataset_id"] for family in FAMILY_COUNTS)
    events = pd.read_csv(first / record["input"], sep="\t")
    assert "expected_occurrence_differential" not in events.columns
    assert events["occurrence"].isna().any()
    truth = pd.read_csv(first / first_manifest["truth"]["filename"], sep="\t")
    assert truth["expected_occurrence_differential"].sum() == 90
    persisted = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["leakage_controls"]["truth_is_separate_from_method_inputs"]


def test_m4_single_seed_separates_occurrence_from_continuous_magnitude(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    manifest = generate(fixture, seeds=(20280101,), n_subjects_per_group=16)
    record = manifest["records"][0]
    truth = pd.read_csv(fixture / manifest["truth"]["filename"], sep="\t").drop(
        columns=["dataset_id", "root_seed"]
    )
    scores, calibration, odds = _evaluate_dataset(
        str(record["dataset_id"]),
        int(record["root_seed"]),
        fixture / str(record["input"]),
        str(record["input_sha256"]),
        truth,
        OccurrenceContrastSpec(minimum_subjects_per_group=8),
    )
    seed_metrics = _seed_metrics(scores)
    calibration_metrics = _calibration_metrics(calibration, odds)
    summary = _method_summary(seed_metrics, calibration_metrics).set_index("method")

    assert (
        summary.loc[M4_METHOD, "average_precision"]
        > summary.loc[M0_METHOD, "average_precision"]
    )
    assert summary.loc[M4_METHOD, "auroc"] > summary.loc[M0_METHOD, "auroc"]
    assert summary.loc[M4_METHOD, "event_coverage"] >= 0.95
    assert summary.loc[M4_METHOD, "prevalence_brier"] < summary.loc[
        RAW_METHOD, "prevalence_brier"
    ]


def test_m4_dataset_seed_is_stable_and_seed_specific() -> None:
    assert _dataset_seed(11) == _dataset_seed(11)
    assert _dataset_seed(11) != _dataset_seed(12)
