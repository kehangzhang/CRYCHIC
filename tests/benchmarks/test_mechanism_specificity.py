from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]
from benchmarks.metrics.mechanism_specificity import (
    REQUIRED_SCENARIOS,
    MechanismSpecificitySpecification,
    evaluate_mechanism_specificity,
    specification_from_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _specification(
    *, expected_seed_count: int = 4
) -> MechanismSpecificitySpecification:
    return MechanismSpecificitySpecification(
        evaluation_phase="test_holdout",
        expected_seed_count=expected_seed_count,
        minimum_margin=0.02,
        confidence_level=0.95,
        require_ci_lower_above=0.0,
        active_max_rank=5,
        active_min_positive_direction_fraction=0.75,
        max_ligand_only_to_active_ratio=0.5,
        max_comparison_coverage_loss=0.05,
        integrated_positive_threshold=0.0,
        max_negative_false_activation_rate=0.0,
        negative_scenarios=tuple(
            scenario for scenario in REQUIRED_SCENARIOS if scenario != "active"
        ),
        receptor_knockout_must_be_zero_or_negative=True,
    )


def _records(*, n_seeds: int = 4) -> pd.DataFrame:
    integrated = {
        "active": 0.80,
        "global_null": 0.0,
        "abundance_only": 0.0,
        "ligand_only": -0.05,
        "target_only": 0.0,
        "receiver_autonomous": 0.0,
        "receptor_knockout": -0.10,
    }
    rows: list[dict[str, object]] = []
    for seed in range(n_seeds):
        for scenario in REQUIRED_SCENARIOS:
            paired_variation = (
                seed * 0.001 if scenario in {"active", "ligand_only"} else 0.0
            )
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "availability_effect": 0.8 if scenario == "active" else 0.0,
                    "receptor_gate": (0.0 if scenario == "receptor_knockout" else 1.0),
                    "receiver_program_effect": (
                        0.8
                        if scenario in {"active", "target_only", "receiver_autonomous"}
                        else 0.0
                    ),
                    "incremental_downstream_effect": (
                        0.7 if scenario in {"active", "target_only"} else 0.0
                    ),
                    "sender_effect": 0.6 if scenario == "active" else 0.0,
                    "integrated_lr_effect": integrated[scenario] + paired_variation,
                    "comparison_coverage": 0.90,
                    "reference_comparison_coverage": 0.92,
                    "known_edge_rank": 2.0,
                    "known_edge_positive_direction": scenario == "active",
                    "status": "observed",
                    "reason_code": None,
                }
            )
    return pd.DataFrame(rows)


def test_frozen_truth_and_config_cover_all_seven_scenarios() -> None:
    truth = yaml.safe_load(
        (REPO_ROOT / "benchmarks/truth/component_truth_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(
        (REPO_ROOT / "benchmarks/configs/mechanism_specificity_v2.json").read_text(
            encoding="utf-8"
        )
    )

    assert tuple(truth["scenarios"]) == REQUIRED_SCENARIOS
    assert tuple(config["scenarios"]) == REQUIRED_SCENARIOS
    assert config["evaluation_phases"]["development"]["required_paired_seeds"] == 50
    assert (
        config["evaluation_phases"]["independent_holdout"]["required_paired_seeds"]
        == 200
    )
    assert config["primary_endpoint"]["minimum_margin"] == 0.02
    assert config["primary_endpoint"]["require_ci_lower_above"] == 0.0
    assert not config["decision_policy"]["gate_alone_switches_default"]
    assert not config["decision_policy"]["candidate_default_change_allowed"]
    assert all(
        set(components)
        == {
            "availability",
            "receptor",
            "receiver_program",
            "incremental_downstream",
            "sender",
            "integrated",
        }
        for components in truth["scenarios"].values()
    )


def test_config_resolves_development_and_holdout_seed_contracts() -> None:
    config = json.loads(
        (REPO_ROOT / "benchmarks/configs/mechanism_specificity_v2.json").read_text(
            encoding="utf-8"
        )
    )

    development = specification_from_config(config, evaluation_phase="development")
    holdout = specification_from_config(config, evaluation_phase="independent_holdout")

    assert development.expected_seed_count == 50
    assert holdout.expected_seed_count == 200
    assert holdout.negative_scenarios == tuple(
        scenario for scenario in REQUIRED_SCENARIOS if scenario != "active"
    )


def test_complete_paired_seed_evidence_passes_all_gates() -> None:
    result = evaluate_mechanism_specificity(_records(), _specification())
    indexed = result.set_index("metric")

    primary = indexed.loc["paired_known_edge_active_minus_ligand_only"]
    assert primary["status"] == "observed"
    assert primary["estimate"] == pytest.approx(0.85)
    assert primary["ci_lower"] > 0.0
    assert primary["n_eligible_seeds"] == 4
    assert bool(primary["gate_passed"])
    assert indexed.loc["active_comparison_coverage_loss", "estimate"] == pytest.approx(
        0.02
    )
    assert indexed.loc["target_only_false_activation_rate", "estimate"] == 0.0
    overall = indexed.loc["g1_5_mechanism_specificity_gate"]
    assert overall["status"] == "observed"
    assert bool(overall["gate_passed"])
    assert overall["estimate"] == 1.0


def test_equal_counts_with_mismatched_seed_sets_are_not_estimable() -> None:
    records = _records()
    ligand = records["scenario"].eq("ligand_only")
    records.loc[ligand, "seed"] = records.loc[ligand, "seed"] + 1

    result = evaluate_mechanism_specificity(records, _specification()).set_index(
        "metric"
    )

    primary = result.loc["paired_known_edge_active_minus_ligand_only"]
    assert primary["status"] == "not_estimable"
    assert bool(pd.isna(primary["estimate"]))
    assert "insufficient_paired_seed_support" in str(primary["reason_code"])
    ligand_false_activation = result.loc["ligand_only_false_activation_rate"]
    assert ligand_false_activation["status"] == "not_estimable"
    assert ligand_false_activation["n_eligible_seeds"] == 3
    overall = result.loc["g1_5_mechanism_specificity_gate"]
    assert overall["status"] == "not_estimable"
    assert bool(pd.isna(overall["gate_passed"]))


def test_nonobserved_seed_support_is_not_imputed() -> None:
    records = _records()
    missing = records["scenario"].eq("active") & records["seed"].eq(0)
    records.loc[missing, ["status", "reason_code"]] = [
        "not_estimable",
        "component_evidence_missing",
    ]
    records.loc[
        missing,
        [
            "availability_effect",
            "receptor_gate",
            "receiver_program_effect",
            "incremental_downstream_effect",
            "sender_effect",
            "integrated_lr_effect",
            "comparison_coverage",
            "reference_comparison_coverage",
            "known_edge_rank",
        ],
    ] = float("nan")

    result = evaluate_mechanism_specificity(records, _specification()).set_index(
        "metric"
    )

    assert result.loc["active_max_rank", "status"] == "not_estimable"
    assert result.loc["g1_5_mechanism_specificity_gate", "status"] == "not_estimable"


def test_false_activation_and_coverage_fail_without_becoming_ne() -> None:
    records = _records()
    target = records["scenario"].eq("target_only") & records["seed"].eq(0)
    records.loc[target, "integrated_lr_effect"] = 0.10
    active = records["scenario"].eq("active")
    records.loc[active, "comparison_coverage"] = 0.80

    result = evaluate_mechanism_specificity(records, _specification()).set_index(
        "metric"
    )

    false_activation = result.loc["target_only_false_activation_rate"]
    assert false_activation["status"] == "observed"
    assert false_activation["estimate"] == pytest.approx(0.25)
    assert not bool(false_activation["gate_passed"])
    coverage = result.loc["active_comparison_coverage_loss"]
    assert coverage["estimate"] == pytest.approx(0.12)
    assert not bool(coverage["gate_passed"])
    overall = result.loc["g1_5_mechanism_specificity_gate"]
    assert overall["status"] == "observed"
    assert not bool(overall["gate_passed"])
    assert "target_only_false_activation_rate" in overall["reason_code"]


def test_duplicate_seed_scenario_records_are_rejected() -> None:
    records = _records()
    duplicate = pd.concat([records, records.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="one row per seed and scenario"):
        evaluate_mechanism_specificity(duplicate, _specification())
