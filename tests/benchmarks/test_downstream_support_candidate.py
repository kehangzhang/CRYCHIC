from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.adapters.common import sha256_file
from benchmarks.simulation.evaluate_downstream_support_candidate import (
    DEVELOPMENT_SCOPE,
    build_candidate_benchmark_config,
    evaluate_candidate_decision,
)

from crychic.attribution import AttributionSupportMethod


def _decision_summary(*, target_v2_effect: float = -0.01) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        "active": ((0.10, 2.0, 1.0), (0.08, 3.0, 0.875)),
        "ligand_only": ((0.05, 1.0, 1.0), (0.01, 4.0, 0.625)),
        "target_only": ((0.02, 8.0, 0.625), (target_v2_effect, 30.0, 0.25)),
        "receptor_knockout": ((-0.5, 45.0, 0.0), (-0.7, 45.0, 0.0)),
    }
    methods = (
        AttributionSupportMethod.RELATIVE_COEFFICIENT_V1.value,
        AttributionSupportMethod.GATED_RESPONSE_NORM_V2.value,
    )
    for scenario, pair in values.items():
        for method, (effect, rank, direction) in zip(methods, pair, strict=True):
            rows.append(
                {
                    "scenario": scenario,
                    "support_method": method,
                    "known_edge_effect": effect,
                    "known_edge_effect_rank": rank,
                    "known_edge_positive_direction_fraction": direction,
                }
            )
    return pd.DataFrame(rows)


def _decision_rule() -> dict[str, object]:
    return {
        "active_min_positive_direction_fraction": 0.75,
        "active_max_effect_rank": 5,
        "require_active_positive_effect": True,
        "require_strict_effect_reduction": ["ligand_only", "target_only"],
        "reduction_epsilon": 0.0,
    }


def test_candidate_recommends_only_when_active_and_both_negatives_pass() -> None:
    comparison, decision = evaluate_candidate_decision(
        _decision_summary(), _decision_rule()
    )

    assert decision["development_scope"] == DEVELOPMENT_SCOPE
    assert not decision["primary_benchmark_eligible"]
    assert decision["active_retained"]
    assert decision["negative_control_effect_reduced"] == {
        "ligand_only": True,
        "target_only": True,
    }
    assert decision["recommend_future_default_change"]
    assert len(comparison) == 4

    _, failed = evaluate_candidate_decision(
        _decision_summary(target_v2_effect=0.03), _decision_rule()
    )
    assert not failed["recommend_future_default_change"]
    assert failed["current_default_remains"] == (
        AttributionSupportMethod.RELATIVE_COEFFICIENT_V1.value
    )


def test_generated_candidate_config_never_relies_on_an_implicit_default(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "synthetic_active.h5ad"
    input_path.write_bytes(b"holdout")
    manifest = {
        "records": [
            {
                "scenario": "active",
                "path": input_path.name,
                "sha256": sha256_file(input_path),
                "scenario_seed": 17,
            }
        ]
    }
    spec = {
        "scenarios": ["active"],
        "support_methods": [
            AttributionSupportMethod.RELATIVE_COEFFICIENT_V1.value,
            AttributionSupportMethod.GATED_RESPONSE_NORM_V2.value,
        ],
        "workflow": {"min_cells": 10},
        "analysis": {"communication_mode": "state"},
        "resolved_database_root": str(tmp_path),
        "resolved_harmonized_manifest": str(tmp_path / "resource.json"),
        "resolved_target_prior_manifest": str(tmp_path / "prior.json"),
        "target_prior_release": "v2_2021",
    }

    config = build_candidate_benchmark_config(
        spec,
        manifest,
        input_root=tmp_path,
        output_root=tmp_path / "output",
    )
    datasets = config["datasets"]
    methods = {
        value["workflow"]["downstream_attribution_support_method"]
        for value in datasets.values()
    }

    assert methods == {method.value for method in AttributionSupportMethod}
    assert set(datasets) == {"holdout_active_v1", "holdout_active_v2"}
