from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.adapters.common import (
    INFERENTIAL_REASON,
    materialize_fixed_universe,
    sha256_file,
    validate_long_table,
)
from benchmarks.comprehensive.export_liana_components import (
    COMPONENT_SPECS,
    run,
)


def _source_bundle(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    source = tmp_path / "source"
    raw_dir = source / "raw"
    raw_dir.mkdir(parents=True)
    sample_metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "subject_id": ["p1", "p2"],
            "context_json": ['{"condition":"A"}', '{"condition":"B"}'],
        }
    )
    support = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "cell_type": ["A", "B", "A", "B"],
            "n_cells": [20, 20, 20, 2],
        }
    )
    resource = pd.DataFrame(
        {
            "interaction_id": ["i1", "i2"],
            "native_interaction_id": ["L1|R1", pd.NA],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "method_covered": [True, False],
        }
    )
    observed = pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "sender": ["A", "B"],
            "receiver": ["B", "A"],
            "interaction_id": ["i1", "i1"],
            "target": [pd.NA, pd.NA],
            "score": [0.2, 0.8],
            "specificity_score": [0.1, 0.2],
            "within_dataset_p_value": [0.9, 0.1],
            "within_dataset_p_value_semantics": ["within sample"] * 2,
        }
    )
    template = materialize_fixed_universe(
        observed,
        sample_metadata=sample_metadata,
        support=support,
        resource=resource,
        dataset_id="fixture_dataset",
        run_id="fixture_source_run",
        method_id="liana_rank_aggregate",
        method_version="1.7.3",
        analysis_track="lr_stlr",
        resource_mode="native",
        resource_id="fixture_resource",
        resource_version="v1",
        score_name="magnitude_rank",
        score_direction="lower",
        specificity_score_name="specificity_rank",
        min_cells=10,
    )
    template_path = source / "interactions_long.parquet"
    template.to_parquet(template_path, index=False)
    raw = pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "source": ["A", "B"],
            "target": ["B", "A"],
            "ligand_complex": ["L1", "L1"],
            "receptor_complex": ["R1", "R1"],
            "lr_means": [0.2, 0.8],
            "cellphone_pvals": [0.9, 0.1],
            "scaled_weight": [2.0, 4.0],
            "lr_logfc": [-1.0, 1.0],
            "spec_weight": [0.25, 0.75],
            "lrscore": [0.6, 0.9],
        }
    )
    raw.to_parquet(raw_dir / "sample_0000.parquet", index=False)
    (source / "manifest.json").write_text(
        json.dumps({"run_id": "fixture_source_run"}) + "\n", encoding="utf-8"
    )
    return raw_dir, template_path, template


def _ordered_status(table: pd.DataFrame) -> pd.DataFrame:
    keys = ["sample_id", "sender", "receiver", "interaction_id", "target"]
    result = table.loc[:, [*keys, "status", "reason_code"]].copy()
    result["target"] = result["target"].astype("string")
    return result.sort_values(keys, kind="stable", ignore_index=True)


def test_export_writes_six_valid_fixed_universe_component_bundles(
    tmp_path: Path,
) -> None:
    raw_dir, template_path, template = _source_bundle(tmp_path)
    output_dir = tmp_path / "components"

    manifest = run(raw_dir, template_path, output_dir, batch_size=3)

    assert manifest["status"] == "complete"
    assert set(manifest["components"]) == {spec.method_id for spec in COMPONENT_SPECS}
    assert manifest["statistical_scope"]["differential_p_value"] is None
    persisted = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["template"]["sha256"] == sha256_file(template_path)

    expected_status = _ordered_status(template)
    for spec in COMPONENT_SPECS:
        method_dir = output_dir / spec.method_id
        table_path = method_dir / "interactions_long.parquet"
        table = validate_long_table(pd.read_parquet(table_path))
        pd.testing.assert_frame_equal(_ordered_status(table), expected_status)
        assert len(table) == len(template)
        assert set(table["method_id"]) == {spec.method_id}
        assert set(table["score_name"]) == {spec.raw_column}
        assert set(table["score_direction"]) == {spec.direction}
        assert table.loc[~table["status"].eq("ok"), "score"].isna().all()
        assert table.loc[table["status"].eq("ok"), "score"].notna().all()
        assert table["differential_effect"].isna().all()
        assert table["differential_p_value"].isna().all()
        assert table["differential_q_value"].isna().all()
        assert table["specificity_score"].isna().all()
        observed = table.loc[table["status"].eq("ok")].set_index(["sender", "receiver"])
        if spec.raw_column == "cellphone_pvals":
            assert observed.loc[("A", "B"), "score"] == pytest.approx(0.9)
            assert observed.loc[("B", "A"), "rank"] == pytest.approx(1.0)
            pd.testing.assert_series_equal(
                table["within_dataset_p_value"], table["score"], check_names=False
            )
        else:
            assert table["within_dataset_p_value"].isna().all()
        if spec.direction == "higher":
            assert observed.loc[("B", "A"), "rank"] == pytest.approx(1.0)

        component_manifest = json.loads(
            (method_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert component_manifest["score_semantics"]["raw_column"] == spec.raw_column
        assert component_manifest["statistical_scope"]["differential_q_value"] is None
        assert (
            component_manifest["status_counts"] == manifest["template"]["status_counts"]
        )
        assert component_manifest["output"]["sha256"] == sha256_file(table_path)
        assert set(table.loc[table["status"].eq("ok"), "reason_code"].astype(str)) == {
            INFERENTIAL_REASON
        }

    assert not list(tmp_path.glob(".components.staging-*"))


def test_export_fails_closed_when_an_ok_template_row_has_no_raw_row(
    tmp_path: Path,
) -> None:
    raw_dir, template_path, _ = _source_bundle(tmp_path)
    (raw_dir / "sample_0000.parquet").unlink()
    output_dir = tmp_path / "components"

    with pytest.raises(ValueError, match=r"status='ok'.*absent"):
        run(raw_dir, template_path, output_dir, batch_size=2)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".components.staging-*"))


def test_failed_overwrite_keeps_the_last_complete_export(tmp_path: Path) -> None:
    raw_dir, template_path, _ = _source_bundle(tmp_path)
    output_dir = tmp_path / "components"
    run(raw_dir, template_path, output_dir, batch_size=4)
    original_manifest = (output_dir / "manifest.json").read_bytes()
    raw_path = raw_dir / "sample_0000.parquet"
    duplicate = pd.read_parquet(raw_path)
    pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True).to_parquet(
        raw_path, index=False
    )

    with pytest.raises(ValueError, match="duplicate edge keys"):
        run(raw_dir, template_path, output_dir, overwrite=True, batch_size=4)

    assert (output_dir / "manifest.json").read_bytes() == original_manifest
    assert not list(tmp_path.glob(".components.staging-*"))


def test_export_refuses_to_replace_an_existing_output_without_flag(
    tmp_path: Path,
) -> None:
    raw_dir, template_path, _ = _source_bundle(tmp_path)
    output_dir = tmp_path / "components"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="--overwrite"):
        run(raw_dir, template_path, output_dir)
