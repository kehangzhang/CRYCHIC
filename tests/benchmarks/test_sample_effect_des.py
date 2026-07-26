from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.literature.sample_effect_des import (
    SampleEffectDESSpec,
    _sha256,
    _validate_source_run_manifest,
    build_sample_effect_rankings,
)


def _scores() -> pd.DataFrame:
    rows = []
    samples = [
        ("a1", "p1", "A"),
        ("a2", "p2", "A"),
        ("a3", "p3", "A"),
        ("b1", "p4", "B"),
        ("b2", "p5", "B"),
        ("b3", "p6", "B"),
    ]
    for sample, subject, condition in samples:
        for sender, receiver, interaction, score in (
            ("X", "Y", "i1", 0.9 if condition == "B" else 0.2),
            ("Y", "X", "i2", 0.8 if condition == "B" else 0.1),
            ("Z", "Z", "i3", 0.1 if condition == "B" else 0.7),
        ):
            rows.append(
                {
                    "dataset_id": "toy",
                    "method_id": "method",
                    "method_version": "1",
                    "resource_id": "resource",
                    "sample_id": sample,
                    "subject_id": subject,
                    "context_json": json.dumps({"condition": condition}),
                    "sender": sender,
                    "receiver": receiver,
                    "interaction_id": interaction,
                    "score": score,
                    "score_direction": "higher",
                    "status": "ok",
                }
            )
    return pd.DataFrame.from_records(rows)


def test_directions_are_collapsed_and_effects_are_condition_specific() -> None:
    rankings, effects = build_sample_effect_rankings(
        _scores(),
        SampleEffectDESSpec(context_key="condition", reference="A", target="B"),
    )

    xy = rankings.loc[
        rankings["sender"].eq("X") & rankings["receiver"].eq("Y")
    ].set_index("condition")
    zz = rankings.loc[
        rankings["sender"].eq("Z") & rankings["receiver"].eq("Z")
    ].set_index("condition")
    assert xy.loc["B", "ranked_strength"] > 0
    assert xy.loc["A", "ranked_strength"] == pytest.approx(0)
    assert zz.loc["A", "ranked_strength"] > 0
    assert zz.loc["B", "ranked_strength"] == pytest.approx(0)
    assert len(effects) == 3
    assert not {"p", "p_value", "q", "q_value"}.intersection(effects.columns)


def test_not_returned_is_zero_but_structural_missing_is_not_imputed() -> None:
    scores = _scores()
    mask = scores["interaction_id"].eq("i1") & scores["sample_id"].eq("b1")
    scores.loc[mask, ["score", "status"]] = [np.nan, "not_returned"]
    missing = scores["interaction_id"].eq("i3") & scores["sample_id"].eq("a1")
    scores.loc[missing, ["score", "status"]] = [np.nan, "cell_type_missing"]

    rankings, effects = build_sample_effect_rankings(
        scores,
        SampleEffectDESSpec(context_key="condition", reference="A", target="B"),
    )

    assert not rankings.empty
    i3 = effects.loc[effects["interaction_id"].eq("i3")].iloc[0]
    assert i3["status"] == "not_estimable"
    assert pd.isna(i3["effect"])
    zz = rankings.loc[
        rankings["sender"].eq("Z") & rankings["receiver"].eq("Z")
    ]
    assert zz["status"].eq("not_estimable").all()
    assert zz["ranked_strength"].isna().all()


def test_duplicate_sample_edges_and_bad_context_fail_closed() -> None:
    scores = _scores()
    duplicated = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)
    spec = SampleEffectDESSpec(context_key="condition", reference="A", target="B")
    with pytest.raises(ValueError, match="duplicate sample-edge"):
        build_sample_effect_rankings(duplicated, spec)

    scores.loc[scores["sample_id"].eq("a1"), "context_json"] = "not-json"
    with pytest.raises(ValueError, match="invalid JSON"):
        build_sample_effect_rankings(scores, spec)


def test_source_run_manifest_binds_input_cohort_and_resource(tmp_path: Path) -> None:
    input_parquet = tmp_path / "interactions_long.parquet"
    input_parquet.write_bytes(b"fixture")
    manifest = tmp_path / "manifest.json"
    payload = {
        "schema_version": "crychic-external-adapter-manifest-v1",
        "status": "complete",
        "run_id": "run-fixture",
        "dataset_id": "toy",
        "method": {"id": "method"},
        "input": {"sha256": "i" * 64, "shape": [6, 3]},
        "resource": {
            "resource_id": "resource",
            "payload_sha256": "r" * 64,
            "manifest_sha256": "m" * 64,
        },
        "output": {
            "table": input_parquet.name,
            "sha256": _sha256(input_parquet),
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    binding = _validate_source_run_manifest(
        manifest,
        input_parquet,
        dataset_id="toy",
        method_id="method",
    )
    assert binding["input_h5ad_sha256"] == "i" * 64
    assert binding["resource_payload_sha256"] == "r" * 64

    payload["dataset_id"] = "wrong"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        _validate_source_run_manifest(
            manifest,
            input_parquet,
            dataset_id="toy",
            method_id="method",
        )
