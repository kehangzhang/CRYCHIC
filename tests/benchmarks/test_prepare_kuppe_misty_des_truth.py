from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.literature import prepare_kuppe_misty_des_truth as module
from benchmarks.literature.prepare_kuppe_misty_des_truth import (
    CTRL,
    IZ,
    KuppeSlide,
    build_kuppe_misty_des_truth,
    prepare_kuppe_misty_des_truth,
)

RECOMPUTATION = (
    "protocol-level recomputation from public CELLxGENE inputs; "
    "not the authors' published MISTy importance table"
)


def _synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = [
        ("c1", "ctrl_1", CTRL),
        ("c2", "ctrl_2", CTRL),
        ("i1", "iz_repeat", IZ),
        ("i2", "iz_repeat", IZ),
        ("i3", "iz_2", IZ),
    ]
    design = pd.DataFrame(samples, columns=["sample_id", "subject_id", "condition"])
    cell_types = ("A", "B", "Cycling.cells")
    rows: list[dict[str, object]] = []
    for sample_index, (sample_id, _, condition) in enumerate(samples):
        for view_index, view in enumerate(("intra", "juxta_5", "para_15")):
            for predictor in cell_types:
                for target in cell_types:
                    value = (
                        sample_index * 10
                        + view_index * 3
                        + cell_types.index(predictor)
                        + cell_types.index(target) / 10
                    )
                    rows.append(
                        {
                            "dataset": module.DATASET_ID,
                            "sample_id": sample_id,
                            "condition": condition,
                            "view": view,
                            "Predictor": predictor,
                            "Target": target,
                            "Importance": value,
                            "recomputation_status": RECOMPUTATION,
                        }
                    )
    importance = pd.DataFrame.from_records(rows)

    # For c1 A/B, B-as-target has the largest raw values but fails target R2.
    overrides = {
        ("intra", "A", "B"): 101.0,
        ("intra", "B", "A"): 12.0,
        ("juxta_5", "A", "B"): 99.0,
        ("juxta_5", "B", "A"): 4.0,
        ("para_15", "A", "B"): 100.0,
        ("para_15", "B", "A"): 8.0,
    }
    for (view, predictor, target), value in overrides.items():
        mask = (
            importance["sample_id"].eq("c1")
            & importance["view"].eq(view)
            & importance["Predictor"].eq(predictor)
            & importance["Target"].eq(target)
        )
        importance.loc[mask, "Importance"] = value

    performance_rows: list[dict[str, object]] = []
    for sample_id, _, condition in samples:
        for target in cell_types:
            r2 = 9.0 if (sample_id, target) == ("c1", "B") else 20.0
            performance_rows.append(
                {
                    "dataset": module.DATASET_ID,
                    "sample_id": sample_id,
                    "condition": condition,
                    "target": target,
                    "measure": "multi.R2",
                    "value": r2,
                    "recomputation_status": RECOMPUTATION,
                }
            )
    performance = pd.DataFrame.from_records(performance_rows)
    return importance, performance, design


def test_r2_filter_bidirectional_max_alias_and_variants() -> None:
    importance, performance, design = _synthetic_tables()

    result = build_kuppe_misty_des_truth(
        importance,
        performance,
        design,
        include_self=True,
        top_fractions=(0.5, 1.0),
    )

    strengths = result.sample_pair_strengths.set_index(
        ["variant", "sample_id", "sender", "receiver"]
    )
    assert strengths.loc[("juxta_only", "c1", "A", "B"), "spatial_importance"] == 4
    assert strengths.loc[("para_only", "c1", "A", "B"), "spatial_importance"] == 8
    assert (
        strengths.loc[("spatial_neighbor_max", "c1", "A", "B"), "spatial_importance"]
        == 8
    )
    assert strengths.loc[("all_view_max", "c1", "A", "B"), "spatial_importance"] == 12
    assert (
        strengths.loc[
            ("spatial_neighbor_max", "c1", "A", "B"),
            "eligible_target_directions",
        ]
        == 1
    )
    assert "Cycling cells" in set(result.sample_pair_strengths["sender"]) | set(
        result.sample_pair_strengths["receiver"]
    )
    assert "Cycling.cells" not in set(result.sample_pair_strengths["sender"]) | set(
        result.sample_pair_strengths["receiver"]
    )
    assert len(result.sample_pair_strengths) == 4 * 5 * 6
    assert set(result.sample_pair_strengths["variant"]) == set(module.VARIANT_VIEWS)


def test_subject_unit_averages_repeats_and_expected_universe_is_stable() -> None:
    importance, performance, design = _synthetic_tables()

    subject = build_kuppe_misty_des_truth(
        importance,
        performance,
        design,
        include_self=False,
        top_fractions=(0.5,),
        multi_sample_unit="subject_id",
    )
    sample = build_kuppe_misty_des_truth(
        importance,
        performance,
        design,
        include_self=False,
        top_fractions=(0.5,),
        multi_sample_unit="sample_id",
    )

    subject_ranks = subject.pair_rankings.set_index(
        ["variant", "scenario", "sender", "receiver"]
    )
    sample_ranks = sample.pair_rankings.set_index(
        ["variant", "scenario", "sender", "receiver"]
    )
    key = ("spatial_neighbor_max", "multi_sample", "A", "B")
    assert subject_ranks.loc[key, "n_IZ_units"] == 2
    assert sample_ranks.loc[key, "n_IZ_units"] == 3
    assert "raw_two_sided_mann_whitney" in subject_ranks.loc[key, "p_value_semantics"]
    assert "q_value" not in subject.pair_rankings.columns

    expected_sizes = subject.expected_sets.groupby(
        ["variant", "scenario", "condition", "top_fraction"], observed=True
    ).size()
    assert set(expected_sizes) == {3}
    assert set(subject.expected_sets["variant"]) == set(module.VARIANT_VIEWS)
    shuffled = build_kuppe_misty_des_truth(
        importance.sample(frac=1, random_state=11).reset_index(drop=True),
        performance.sample(frac=1, random_state=12).reset_index(drop=True),
        design.sample(frac=1, random_state=13).reset_index(drop=True),
        include_self=False,
        top_fractions=(0.5,),
    )
    pd.testing.assert_frame_equal(subject.pair_rankings, shuffled.pair_rankings)
    pd.testing.assert_frame_equal(subject.expected_sets, shuffled.expected_sets)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_fixture(
    root: Path,
    roster: tuple[KuppeSlide, ...],
    raw_cell_types: tuple[str, ...],
) -> Path:
    samples = [(slide.sample_id, slide.subject_id, slide.condition) for slide in roster]
    design = pd.DataFrame(samples, columns=["sample_id", "subject_id", "condition"])
    importance_rows = []
    performance_rows = []
    for sample_index, sample in enumerate(design.itertuples(index=False)):
        for view_index, view in enumerate(("intra", "juxta_5", "para_15")):
            for predictor in raw_cell_types:
                for target in raw_cell_types:
                    importance_rows.append(
                        {
                            "dataset": module.DATASET_ID,
                            "sample_id": sample.sample_id,
                            "condition": sample.condition,
                            "view": view,
                            "Predictor": predictor,
                            "Target": target,
                            "Importance": sample_index + view_index + 0.5,
                            "recomputation_status": RECOMPUTATION,
                        }
                    )
        for target in raw_cell_types:
            performance_rows.append(
                {
                    "dataset": module.DATASET_ID,
                    "sample_id": sample.sample_id,
                    "condition": sample.condition,
                    "target": target,
                    "measure": "multi.R2",
                    "value": 20.0,
                    "recomputation_status": RECOMPUTATION,
                }
            )
    importance = pd.DataFrame.from_records(importance_rows)
    performance = pd.DataFrame.from_records(performance_rows)
    importance_path = root / "combined_importance.tsv.gz"
    performance_path = root / "combined_performance.tsv.gz"
    importance.to_csv(importance_path, sep="\t", index=False)
    performance.to_csv(performance_path, sep="\t", index=False)
    manifest = {
        "schema_version": module.RUN_SCHEMA,
        "dataset_id": module.DATASET_ID,
        "status": "complete",
        "recomputation_status": RECOMPUTATION,
        "software": {
            "mistyR": module.MISTYR_VERSION,
            "mistyR_tag_commit": module.MISTYR_TAG_COMMIT,
        },
        "samples": [
            {
                "sample_id": slide.sample_id,
                "condition": slide.condition,
                "status": "complete",
            }
            for slide in roster
        ],
        "outputs": {
            "importances": {
                "filename": importance_path.name,
                "sha256": _sha256(importance_path),
                "rows": len(importance),
            },
            "performance": {
                "filename": performance_path.name,
                "sha256": _sha256(performance_path),
                "rows": len(performance),
            },
        },
    }
    path = root / "run.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_checksum_recomputation_and_compact_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster = tuple(
        [KuppeSlide(f"c{index}", f"c{index}", CTRL) for index in range(4)]
        + [KuppeSlide(f"i{index}", f"i{index}", IZ) for index in range(9)]
    )
    raw_types = ("A", "Cycling.cells")
    monkeypatch.setattr(module, "FROZEN_SLIDES", roster)
    monkeypatch.setattr(module, "RAW_CELL_TYPES", raw_types)
    run_manifest = _write_run_fixture(tmp_path, roster, raw_types)

    output = tmp_path / "output"
    dry = prepare_kuppe_misty_des_truth(
        run_manifest, output, include_self=True, dry_run=True
    )
    assert dry["status"] == "dry_run"
    assert dry["source"]["author_importance_table"] is False
    assert not output.exists()

    complete = prepare_kuppe_misty_des_truth(run_manifest, output, include_self=True)
    assert complete["status"] == "complete"
    written = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert written["protocol"]["q_values_generated"] is False
    assert written["protocol"]["primary_variant"] == "spatial_neighbor_max"
    for record in written["outputs"].values():
        path = output / record["filename"]
        assert _sha256(path) == record["sha256"]


def test_manifest_checksum_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster = tuple(
        [KuppeSlide(f"c{index}", f"c{index}", CTRL) for index in range(4)]
        + [KuppeSlide(f"i{index}", f"i{index}", IZ) for index in range(9)]
    )
    raw_types = ("A", "Cycling.cells")
    monkeypatch.setattr(module, "FROZEN_SLIDES", roster)
    monkeypatch.setattr(module, "RAW_CELL_TYPES", raw_types)
    run_manifest = _write_run_fixture(tmp_path, roster, raw_types)
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    manifest["outputs"]["importances"]["sha256"] = "0" * 64
    run_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_kuppe_misty_des_truth(
            run_manifest,
            tmp_path / "output",
            include_self=False,
            dry_run=True,
        )
