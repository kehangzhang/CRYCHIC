from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from tests.support.sample_edge_v2 import sample_edge_scores

from crychic.scoring import SampleEdgeScoreV2

ROOT = Path(__file__).resolve().parents[3]


def test_sample_edge_v2_contract_separates_raw_parent_and_attribution() -> None:
    scores = sample_edge_scores()

    assert scores.table["sender_detection_raw"].tolist() == [2.0, 1.0]
    assert scores.table["parent_peak_raw"].unique().tolist() == [2.0]
    assert scores.table["parent_total_raw"].unique().tolist() == [3.0]
    assert scores.table["parent_mean_raw"].unique().tolist() == [1.5]
    assert scores.table["sender_attribution"].sum() == pytest.approx(0.9)
    assert scores.table["null_sender_attribution"].iloc[0] == pytest.approx(0.1)


def test_missing_sender_does_not_invalidate_other_candidate() -> None:
    scores = sample_edge_scores()
    table = scores.table.copy()
    missing = table["sender"].eq("sender-b")
    table.loc[missing, ["ligand_activity_raw", "sender_detection_raw"]] = np.nan
    table.loc[missing, "status"] = "not_estimable"
    table.loc[missing, "reason_code"] = "sender_measurement_missing"
    table.loc[missing, "coverage_status"] = "not_measured_or_not_estimable"
    table.loc[missing, "sender_attribution"] = np.nan
    table.loc[missing, "attribution_status"] = "not_estimable"
    table.loc[missing, "attribution_reason_code"] = "sender_measurement_missing"
    table["null_sender_attribution"] = 0.2
    table.loc[~missing, "sender_attribution"] = 0.8
    table["effective_candidate_count"] = 1
    table["parent_peak_raw"] = 2.0
    table["parent_total_raw"] = 2.0
    table["parent_mean_raw"] = 2.0

    observed = SampleEdgeScoreV2(table=table, provenance=scores.provenance)

    valid = observed.table.loc[~missing].iloc[0]
    assert valid["sender_detection_raw"] == 2.0
    assert valid["sender_attribution"] == 0.8
    assert observed.table["effective_candidate_count"].unique().tolist() == [1]


def test_parent_summary_mismatch_is_rejected() -> None:
    scores = sample_edge_scores()
    table = scores.table.copy()
    table["parent_total_raw"] = 4.0

    with pytest.raises(ValueError, match="parent summaries"):
        SampleEdgeScoreV2(table=table, provenance=scores.provenance)


def test_low_evidence_is_not_structural_impossibility() -> None:
    scores = sample_edge_scores()
    table = scores.table.copy()
    table.loc[0, "status"] = "low_evidence"
    table.loc[0, "reason_code"] = "low_evidence"

    observed = SampleEdgeScoreV2(table=table, provenance=scores.provenance)

    assert observed.table.loc[0, "sender_detection_raw"] == 2.0
    assert not bool(observed.table.loc[0, "structural_impossibility"])


def test_optional_head_requires_applied_functional_lineage() -> None:
    scores = sample_edge_scores()
    missing_lineage = scores.table.copy()
    missing_lineage["attribution_functional_id"] = None

    with pytest.raises(ValueError, match="attribution_functional_id"):
        SampleEdgeScoreV2(table=missing_lineage, provenance=scores.provenance)

    impossible_lineage = scores.table.copy()
    impossible_lineage["program_functional_id"] = "program-not-applied"
    with pytest.raises(ValueError, match="not-computed program_status"):
        SampleEdgeScoreV2(table=impossible_lineage, provenance=scores.provenance)


def test_sample_edge_v2_schema_is_valid() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "sample_edge_scores_v2.schema.json").read_text(
            encoding="utf-8"
        )
    )

    Draft202012Validator.check_schema(schema)
    required = schema["properties"]["columns"]["required"]
    assert tuple(required) == tuple(sample_edge_scores().table.columns)
