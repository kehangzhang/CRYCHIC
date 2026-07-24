from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.adapters.common import sha256_file
from benchmarks.comprehensive.evaluate_m5_rewiring import (
    SCHEMA_VERSION,
    _retention_table,
    _rewiring_diagnostics,
    evaluate,
)
from benchmarks.comprehensive.generate_m5_hprior_fixture import (
    VIEW_COLUMNS,
    generate,
)
from crychic.scoring import (
    freeze_hypergraph_prior,
    rewire_hypergraph_prior_degree_matched,
)


def test_m5_rewiring_diagnostics_report_exact_bounded_edge_fraction(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    manifest = generate(fixture, seeds=(20290101,), n_edges=600)
    topology = pd.read_csv(fixture / manifest["hypergraph_prior"]["filename"], sep="\t")
    full = freeze_hypergraph_prior(topology, view_columns=VIEW_COLUMNS)
    rewired = rewire_hypergraph_prior_degree_matched(
        full, fraction=0.25, seed=20260741
    )

    diagnostics = _rewiring_diagnostics(
        full, rewired, requested_fraction=0.25
    )

    assert diagnostics["changed_edges"] == 150
    assert diagnostics["actual_edge_fraction"] == 0.25
    assert diagnostics["degree_profiles_equal"] is True
    assert diagnostics["parent_prior_id"] == full.prior_id


def test_m5_gain_retention_is_relative_to_correct_and_permuted_endpoints() -> None:
    summary = pd.DataFrame.from_records(
        [
            {
                "method": "crychic_m5_full_hypergraph_prior",
                "effect_mse": 0.2,
                "average_precision": 0.8,
            },
            {
                "method": "degree_matched_partial_rewire_10pct",
                "effect_mse": 0.24,
                "average_precision": 0.76,
            },
            {
                "method": "degree_matched_partial_rewire_25pct",
                "effect_mse": 0.3,
                "average_precision": 0.7,
            },
            {
                "method": "degree_matched_partial_rewire_50pct",
                "effect_mse": 0.4,
                "average_precision": 0.6,
            },
            {
                "method": "degree_matched_permuted_hypergraph_prior",
                "effect_mse": 0.6,
                "average_precision": 0.4,
            },
        ]
    )

    retention = _retention_table(summary).set_index("method")

    assert retention.loc[
        "degree_matched_partial_rewire_10pct", "mse_topology_gain_retention"
    ] == pytest.approx(0.9)
    assert retention.loc[
        "degree_matched_partial_rewire_25pct", "ap_topology_gain_retention"
    ] == pytest.approx(0.75)


def test_m5_rewiring_evaluator_writes_descriptive_degree_matched_outputs(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    manifest = generate(
        fixture,
        seeds=(20290101, 20290102, 20290103),
        n_edges=600,
    )
    tracked = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "benchmarks/configs/suggestions_next_m5_rewiring_v1.json"
        ).read_text(encoding="utf-8")
    )
    tracked["development"] = {
        "seeds": manifest["seeds"],
        "fixture_manifest_sha256": sha256_file(fixture / "manifest.json"),
        "minimum_paired_seeds": 3,
        "bootstrap_replicates": 200,
        "bootstrap_seed": 11,
    }
    tracked["gates"]["minimum_gain_retention"] = {
        "degree_matched_partial_rewire_10pct": -10.0,
        "degree_matched_partial_rewire_25pct": -10.0,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(tracked, indent=2) + "\n", encoding="utf-8")
    output = tmp_path / "output"

    result = evaluate(fixture, config, output, role="development")

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["claim_boundary"]["formal_inference_allowed"] is False
    assert result["acceptance"]["checks"]["all_degree_profiles_equal"] is True
    metrics = pd.read_csv(output / "replicate_metrics.tsv", sep="\t")
    assert not metrics["formal_inference_allowed"].any()
    assert not {"p_value", "q_value", "probability"}.intersection(metrics.columns)
