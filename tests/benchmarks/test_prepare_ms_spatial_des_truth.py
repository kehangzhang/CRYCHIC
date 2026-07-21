from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.literature import prepare_ms_spatial_des_truth as module
from benchmarks.literature.prepare_ms_spatial_des_truth import (
    CHRONIC_ACTIVE,
    CONTROL,
    MSSpatialSample,
    build_ms_spatial_des_truth,
    compute_unordered_pearson_correlations,
    prepare_ms_spatial_des_truth,
)


def test_unordered_pearson_self_switch_and_missingness_contract() -> None:
    proportions = pd.DataFrame(
        {
            "B": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "constant": [2.0] * 6,
            "sparse": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
        }
    )

    without_self = compute_unordered_pearson_correlations(
        proportions, include_self=False
    )
    with_self = compute_unordered_pearson_correlations(proportions, include_self=True)

    assert len(without_self) == 6
    assert len(with_self) == 10
    assert (with_self["sender"] <= with_self["receiver"]).all()
    ab = without_self.set_index(["sender", "receiver"]).loc[("A", "B")]
    assert ab["pearson_r"] == pytest.approx(-1.0)
    constant = without_self.set_index(["sender", "receiver"]).loc[("A", "constant")]
    assert constant["status"] == "not_estimable"
    assert constant["reason_code"] == "constant_proportion"
    assert pd.isna(constant["pearson_r"])
    sparse = without_self.set_index(["sender", "receiver"]).loc[("A", "sparse")]
    assert sparse["reason_code"] == "insufficient_complete_spots"
    self_a = with_self.set_index(["sender", "receiver"]).loc[("A", "A")]
    assert self_a["pearson_r"] == pytest.approx(1.0)


def _synthetic_correlations() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = [
        ("c1", "control_1", CONTROL),
        ("c2", "control_2", CONTROL),
        ("c3", "control_3", CONTROL),
        ("d1", "disease_1", CHRONIC_ACTIVE),
        ("d2", "disease_2", CHRONIC_ACTIVE),
        ("d3", "disease_2", CHRONIC_ACTIVE),
        ("d4", "disease_3", CHRONIC_ACTIVE),
    ]
    design = pd.DataFrame(samples, columns=["sample_id", "subject_id", "condition"])
    values = {
        ("A", "A"): {
            "c1": 1.0,
            "c2": 1.0,
            "c3": 1.0,
            "d1": 1.0,
            "d2": 1.0,
            "d3": 1.0,
            "d4": 1.0,
        },
        ("A", "B"): {
            "c1": 0.75,
            "c2": 0.75,
            "c3": 0.75,
            "d1": 0.25,
            "d2": 0.25,
            "d3": 0.25,
            "d4": 0.25,
        },
        ("A", "C"): {
            "c1": 0.25,
            "c2": 0.25,
            "c3": 0.25,
            "d1": 0.75,
            "d2": 0.75,
            "d3": 0.75,
            "d4": 0.75,
        },
        ("B", "C"): {
            "c1": 0.2,
            "c2": 0.3,
            "c3": 0.4,
            "d1": 0.2,
            "d2": 0.3,
            "d3": 0.3,
            "d4": 0.4,
        },
    }
    rows = []
    for (sender, receiver), sample_values in values.items():
        for sample_id, value in sample_values.items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "sender": sender,
                    "receiver": receiver,
                    "pearson_r": value,
                    "status": "observed",
                    "reason_code": "",
                }
            )
    return pd.DataFrame.from_records(rows), design


def test_rankings_have_direction_subject_units_and_stable_ties() -> None:
    correlations, design = _synthetic_correlations()

    first = build_ms_spatial_des_truth(
        correlations,
        design,
        top_fractions=(0.5, 1.0),
        multi_sample_unit="subject_id",
    )
    second = build_ms_spatial_des_truth(
        correlations.sample(frac=1, random_state=17).reset_index(drop=True),
        design.sample(frac=1, random_state=23).reset_index(drop=True),
        top_fractions=(0.5, 1.0),
        multi_sample_unit="subject_id",
    )

    pd.testing.assert_frame_equal(first.pair_rankings, second.pair_rankings)
    pd.testing.assert_frame_equal(first.expected_sets, second.expected_sets)
    rankings = first.pair_rankings.set_index(["scenario", "sender", "receiver"])
    assert rankings.loc[("condition_aware", "A", "B"), "direction_condition"] == CONTROL
    assert (
        rankings.loc[("condition_aware", "A", "C"), "direction_condition"]
        == CHRONIC_ACTIVE
    )
    assert rankings.loc[("condition_aware", "A", "B"), "spatial_rank"] == 1
    assert rankings.loc[("condition_aware", "A", "C"), "spatial_rank"] == 2
    assert not bool(rankings.loc[("condition_aware", "A", "A"), "ranking_eligible"])
    assert rankings.loc[("multi_sample", "A", "B"), "n_chronic_active_units"] == 3
    assert (
        "two_sided_mann_whitney"
        in rankings.loc[("multi_sample", "A", "B"), "p_value_semantics"]
    )
    assert "q_value" not in first.pair_rankings.columns

    expected = first.expected_sets
    group_sizes = expected.groupby(
        ["scenario", "condition", "top_fraction"], observed=True
    ).size()
    assert set(group_sizes) == {4}
    top_control = expected.loc[
        expected["scenario"].eq("condition_aware")
        & expected["condition"].eq(CONTROL)
        & expected["top_fraction"].eq(0.5)
        & expected["is_expected"]
    ]
    selected_pairs = list(
        top_control[["sender", "receiver"]].itertuples(index=False, name=None)
    )
    assert selected_pairs == [("A", "B")]


def test_nonestimable_pairs_remain_in_fixed_expected_universe() -> None:
    correlations, design = _synthetic_correlations()
    missing = correlations["sample_id"].eq("d4") & correlations["sender"].eq("B")
    correlations.loc[missing, "pearson_r"] = np.nan
    correlations.loc[missing, "status"] = "not_estimable"
    correlations.loc[missing, "reason_code"] = "constant_proportion"

    result = build_ms_spatial_des_truth(
        correlations,
        design,
        top_fractions=(0.5,),
    )

    assert len(result.expected_sets.groupby(["scenario", "condition"])) == 4
    assert set(
        result.expected_sets.groupby(["scenario", "condition"], observed=True).size()
    ) == {4}
    assert (
        result.sample_correlations.loc[
            result.sample_correlations["status"].eq("not_estimable"), "pearson_r"
        ]
        .isna()
        .all()
    )


def test_top_count_rule_distinguishes_floor_and_ceil() -> None:
    correlations, design = _synthetic_correlations()
    floor = build_ms_spatial_des_truth(
        correlations,
        design,
        top_fractions=(0.5,),
        top_count_rule="floor",
    )
    ceil = build_ms_spatial_des_truth(
        correlations,
        design,
        top_fractions=(0.5,),
        top_count_rule="ceil",
    )

    assert set(floor.expected_sets["top_count"]) == {1}
    assert set(floor.expected_sets["top_count_rule"]) == {"floor"}
    assert set(ceil.expected_sets["top_count"]) == {1, 2}
    assert set(ceil.expected_sets["top_count_rule"]) == {"ceil"}


def _write_public_fixture(root: Path, sample: MSSpatialSample, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    proportions = rng.dirichlet(np.arange(1, 10), size=12)
    table = pd.DataFrame(proportions, columns=module.CELL_TYPES)
    table.insert(0, "niches", "WM")
    table.insert(0, "array_col", np.arange(len(table)))
    table.insert(0, "array_row", np.arange(len(table)))
    table.insert(0, "cellId", [f"spot_{index}" for index in range(len(table))])
    meta_path = root / f"{sample.file_stem}.meta.tsv"
    table.to_csv(meta_path, sep="\t", index=False, lineterminator="\n")
    md5 = hashlib.md5(meta_path.read_bytes(), usedforsecurity=False).hexdigest()
    dataset = {
        "name": f"ms-subcortical-lesions/visium-{sample.file_stem}",
        "sampleCount": len(table),
        "fileVersions": {
            "outMeta": {"size": meta_path.stat().st_size, "md5": md5[:10]}
        },
        "metaFields": [{"name": column} for column in table.columns],
    }
    (root / f"{sample.file_stem}.dataset.json").write_text(
        json.dumps(dataset), encoding="utf-8"
    )


def test_public_manifest_validation_dry_run_and_compact_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster = (
        MSSpatialSample("c1", "C1", "C1", CONTROL),
        MSSpatialSample("c2", "C2", "C2", CONTROL),
        MSSpatialSample("d1", "D1", "D1", CHRONIC_ACTIVE),
        MSSpatialSample("d2", "D2", "D2", CHRONIC_ACTIVE),
    )
    monkeypatch.setattr(module, "MS_SPATIAL_SAMPLES", roster)
    source = tmp_path / "source"
    source.mkdir()
    for index, sample in enumerate(roster):
        _write_public_fixture(source, sample, seed=100 + index)
    output = tmp_path / "output"

    dry_manifest = prepare_ms_spatial_des_truth(
        source,
        output,
        include_self=True,
        dry_run=True,
    )

    assert dry_manifest["status"] == "dry_run"
    assert dry_manifest["design"]["samples_per_condition"] == {
        CONTROL: 2,
        CHRONIC_ACTIVE: 2,
    }
    assert dry_manifest["protocol"]["include_self_pairs"] is True
    assert not output.exists()

    manifest = prepare_ms_spatial_des_truth(
        source,
        output,
        include_self=True,
    )

    assert manifest["status"] == "complete"
    assert (output / "manifest.json").is_file()
    parsed = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert parsed["protocol"]["q_values_generated"] is False
    expected = pd.read_csv(output / "ms_spatial_expected_sets.tsv", sep="\t")
    assert len(expected) == 2 * 2 * 4 * 45
    assert set(expected["is_expected"]) <= {True, False}
    for record in parsed["outputs"].values():
        assert len(record["sha256"]) == 64


def test_public_manifest_size_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster = (
        MSSpatialSample("c1", "C1", "C1", CONTROL),
        MSSpatialSample("c2", "C2", "C2", CONTROL),
        MSSpatialSample("d1", "D1", "D1", CHRONIC_ACTIVE),
        MSSpatialSample("d2", "D2", "D2", CHRONIC_ACTIVE),
    )
    monkeypatch.setattr(module, "MS_SPATIAL_SAMPLES", roster)
    source = tmp_path / "source"
    source.mkdir()
    for index, sample in enumerate(roster):
        _write_public_fixture(source, sample, seed=200 + index)
    manifest_path = source / "c1.dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fileVersions"]["outMeta"]["size"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"size .* != manifest"):
        prepare_ms_spatial_des_truth(
            source,
            tmp_path / "output",
            include_self=False,
            dry_run=True,
        )
