from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]
from benchmarks.adapters.common import sha256_file
from benchmarks.evaluate_mechanism_specificity import run_evaluation
from benchmarks.metrics.mechanism_specificity import (
    COMPONENTS,
    MACRO_EDGE_ID,
    REQUIRED_SCENARIOS,
    ComponentTruthMatrix,
    MechanismSpecificitySpecification,
    component_truth_from_mapping,
    evaluate_mechanism_specificity,
    specification_from_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _truth_mapping() -> dict[str, object]:
    value = yaml.safe_load(
        (REPO_ROOT / "benchmarks/truth/component_truth_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    return cast(dict[str, object], value)


def _truth() -> ComponentTruthMatrix:
    return component_truth_from_mapping(_truth_mapping())


def _specification(
    *, expected_seed_count: int = 4
) -> MechanismSpecificitySpecification:
    return MechanismSpecificitySpecification(
        evaluation_phase="test_holdout",
        expected_seed_count=expected_seed_count,
        minimum_known_edges=2,
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
        component_positive_threshold=0.0,
        component_zero_maximum=0.0,
        minimum_component_conformance_rate=1.0,
    )


def _component_value(code: str) -> float:
    return {
        "positive": 0.8,
        "zero": 0.0,
        "allowed_positive": 0.4,
        "ecosystem_only": 0.0,
    }[code]


def _records(*, n_seeds: int = 4) -> pd.DataFrame:
    truth = _truth()
    rows: list[dict[str, object]] = []
    for edge_index, edge_id in enumerate(truth.known_edge_ids):
        for seed in range(n_seeds):
            for scenario in REQUIRED_SCENARIOS:
                components = {
                    component: _component_value(truth.code(scenario, component))
                    for component in COMPONENTS
                }
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "known_edge_id": edge_id,
                        "availability_effect": components["availability"],
                        "receptor_gate": components["receptor"],
                        "receiver_program_effect": components["receiver_program"],
                        "incremental_downstream_effect": components[
                            "incremental_downstream"
                        ],
                        "sender_effect": components["sender"],
                        "integrated_lr_effect": components["integrated"],
                        "comparison_coverage": 0.90,
                        "reference_comparison_coverage": 0.92,
                        "known_edge_rank": float(edge_index + 2),
                        "known_edge_positive_direction": scenario == "active",
                        "status": "observed",
                        "reason_code": None,
                    }
                )
    return pd.DataFrame(rows)


def _metric(
    result: pd.DataFrame,
    metric: str,
    *,
    known_edge_id: str,
    aggregation: str,
    scenario: str | None = None,
    component: str | None = None,
) -> pd.Series:
    selected = result.loc[
        result["metric"].eq(metric)
        & result["known_edge_id"].eq(known_edge_id)
        & result["aggregation"].eq(aggregation)
    ]
    if scenario is not None:
        selected = selected.loc[selected["scenario"].eq(scenario)]
    if component is not None:
        selected = selected.loc[selected["component"].eq(component)]
    assert len(selected) == 1
    return selected.iloc[0]


def test_frozen_truth_and_config_define_multi_positive_contract() -> None:
    truth_mapping = _truth_mapping()
    truth = _truth()
    config = json.loads(
        (REPO_ROOT / "benchmarks/configs/mechanism_specificity_v2.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(truth.known_edge_ids) == 3
    assert len(set(truth.known_edge_ids)) == 3
    truth_scenarios = cast(Mapping[str, object], truth_mapping["scenarios"])
    assert tuple(truth_scenarios) == REQUIRED_SCENARIOS
    assert tuple(config["scenarios"]) == REQUIRED_SCENARIOS
    assert "known_edge_id" in config["evidence_record_contract"]["required_columns"]
    assert config["known_edge_contract"]["minimum_known_edges"] == 2
    assert config["component_truth"]["truth_set_id"] == truth.truth_set_id
    assert config["component_truth"]["sha256"] == sha256_file(
        REPO_ROOT / "benchmarks/truth/component_truth_matrix.yaml"
    )
    assert config["evaluation_phases"]["development"]["required_paired_seeds"] == 50
    assert (
        config["evaluation_phases"]["independent_holdout"]["required_paired_seeds"]
        == 200
    )
    assert not config["decision_policy"]["gate_alone_switches_default"]
    assert not config["decision_policy"]["candidate_default_change_allowed"]


def test_config_resolves_seed_edge_and_component_gates() -> None:
    config = json.loads(
        (REPO_ROOT / "benchmarks/configs/mechanism_specificity_v2.json").read_text(
            encoding="utf-8"
        )
    )

    development = specification_from_config(config, evaluation_phase="development")
    holdout = specification_from_config(config, evaluation_phase="independent_holdout")

    assert development.expected_seed_count == 50
    assert holdout.expected_seed_count == 200
    assert holdout.minimum_known_edges == 2
    assert holdout.minimum_component_conformance_rate == 1.0


def test_complete_multi_edge_evidence_passes_edge_macro_and_component_gates() -> None:
    truth = _truth()
    result = evaluate_mechanism_specificity(_records(), _specification(), truth)
    first_edge = truth.known_edge_ids[0]

    edge_primary = _metric(
        result,
        "paired_known_edge_active_minus_ligand_only",
        known_edge_id=first_edge,
        aggregation="edge",
    )
    assert edge_primary["estimate"] == pytest.approx(0.8)
    assert edge_primary["ci_lower"] > 0
    assert bool(edge_primary["gate_passed"])

    macro_primary = _metric(
        result,
        "paired_known_edge_active_minus_ligand_only",
        known_edge_id=MACRO_EDGE_ID,
        aggregation="macro_equal_edge",
    )
    assert macro_primary["estimate"] == pytest.approx(0.8)
    assert macro_primary["n_eligible_edges"] == 3
    assert bool(macro_primary["gate_passed"])

    conformance = _metric(
        result,
        "component_truth_conformance_rate",
        known_edge_id=first_edge,
        aggregation="edge",
        scenario="target_only",
        component="receiver_program",
    )
    assert conformance["truth_code"] == "positive"
    assert conformance["estimate"] == 1.0
    assert bool(conformance["gate_passed"])

    overall = _metric(
        result,
        "g1_5_mechanism_specificity_gate",
        known_edge_id="__all_known_edges__",
        aggregation="overall",
    )
    assert overall["status"] == "observed"
    assert overall["estimate"] == 1.0
    assert bool(overall["gate_passed"])


def test_allowed_positive_and_ecosystem_only_have_explicit_diagnostics() -> None:
    truth = _truth()
    result = evaluate_mechanism_specificity(_records(), _specification(), truth)
    edge = truth.known_edge_ids[0]

    allowed = _metric(
        result,
        "component_allowed_positive_rate",
        known_edge_id=edge,
        aggregation="edge",
        scenario="ligand_only",
        component="receiver_program",
    )
    assert allowed["truth_code"] == "allowed_positive"
    assert allowed["estimate"] == 1.0
    assert pd.isna(allowed["gate_passed"])
    assert "not_integrated_evidence" in str(allowed["reason_code"])

    ecosystem = _metric(
        result,
        "component_false_activation_rate",
        known_edge_id=edge,
        aggregation="edge",
        scenario="abundance_only",
        component="availability",
    )
    assert ecosystem["truth_code"] == "ecosystem_only"
    assert ecosystem["estimate"] == 0.0
    assert "ecosystem_change_allowed" in str(ecosystem["reason_code"])


def test_one_edge_with_mismatched_seed_sets_is_ne_without_unpaired_fallback() -> None:
    truth = _truth()
    records = _records()
    edge = truth.known_edge_ids[0]
    ligand = records["scenario"].eq("ligand_only") & records["known_edge_id"].eq(edge)
    records.loc[ligand, "seed"] = records.loc[ligand, "seed"] + 1

    result = evaluate_mechanism_specificity(records, _specification(), truth)
    primary = _metric(
        result,
        "paired_known_edge_active_minus_ligand_only",
        known_edge_id=edge,
        aggregation="edge",
    )
    assert primary["status"] == "not_estimable"
    assert "insufficient_paired_seed_support" in str(primary["reason_code"])
    macro = _metric(
        result,
        "paired_known_edge_active_minus_ligand_only",
        known_edge_id=MACRO_EDGE_ID,
        aggregation="macro_equal_edge",
    )
    assert macro["status"] == "not_estimable"


def test_missing_registered_edge_is_ne_not_silently_removed() -> None:
    truth = _truth()
    missing_edge = truth.known_edge_ids[-1]
    records = _records().loc[lambda table: table["known_edge_id"].ne(missing_edge)]

    result = evaluate_mechanism_specificity(records, _specification(), truth)
    primary = _metric(
        result,
        "paired_known_edge_active_minus_ligand_only",
        known_edge_id=missing_edge,
        aggregation="edge",
    )
    assert primary["status"] == "not_estimable"
    overall = _metric(
        result,
        "g1_5_mechanism_specificity_gate",
        known_edge_id="__all_known_edges__",
        aggregation="overall",
    )
    assert overall["status"] == "not_estimable"


def test_component_false_activation_fails_observed_gate() -> None:
    truth = _truth()
    records = _records()
    edge = truth.known_edge_ids[1]
    affected = (
        records["known_edge_id"].eq(edge)
        & records["scenario"].eq("receiver_autonomous")
        & records["seed"].eq(0)
    )
    records.loc[affected, "incremental_downstream_effect"] = 0.2

    result = evaluate_mechanism_specificity(records, _specification(), truth)
    conformance = _metric(
        result,
        "component_truth_conformance_rate",
        known_edge_id=edge,
        aggregation="edge",
        scenario="receiver_autonomous",
        component="incremental_downstream",
    )
    assert conformance["status"] == "observed"
    assert conformance["estimate"] == pytest.approx(0.75)
    assert not bool(conformance["gate_passed"])
    false_activation = _metric(
        result,
        "component_false_activation_rate",
        known_edge_id=edge,
        aggregation="edge",
        scenario="receiver_autonomous",
        component="incremental_downstream",
    )
    assert false_activation["estimate"] == pytest.approx(0.25)
    overall = _metric(
        result,
        "g1_5_mechanism_specificity_gate",
        known_edge_id="__all_known_edges__",
        aggregation="overall",
    )
    assert overall["status"] == "observed"
    assert not bool(overall["gate_passed"])


def test_integrated_false_activation_and_coverage_fail_without_becoming_ne() -> None:
    truth = _truth()
    records = _records()
    edge = truth.known_edge_ids[0]
    target = (
        records["known_edge_id"].eq(edge)
        & records["scenario"].eq("target_only")
        & records["seed"].eq(0)
    )
    records.loc[target, "integrated_lr_effect"] = 0.1
    active = records["known_edge_id"].eq(edge) & records["scenario"].eq("active")
    records.loc[active, "comparison_coverage"] = 0.8

    result = evaluate_mechanism_specificity(records, _specification(), truth)
    false_activation = _metric(
        result,
        "target_only_false_activation_rate",
        known_edge_id=edge,
        aggregation="edge",
    )
    assert false_activation["estimate"] == pytest.approx(0.25)
    assert not bool(false_activation["gate_passed"])
    coverage = _metric(
        result,
        "active_comparison_coverage_loss",
        known_edge_id=edge,
        aggregation="edge",
    )
    assert coverage["estimate"] == pytest.approx(0.12)
    assert not bool(coverage["gate_passed"])


def test_unregistered_and_duplicate_edge_records_are_rejected() -> None:
    truth = _truth()
    records = _records()
    unknown = records.copy()
    unknown.loc[0, "known_edge_id"] = "post_hoc_edge"
    with pytest.raises(ValueError, match="unregistered edges"):
        evaluate_mechanism_specificity(unknown, _specification(), truth)

    duplicate = pd.concat([records, records.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per seed/scenario/edge"):
        evaluate_mechanism_specificity(duplicate, _specification(), truth)


def test_truth_parser_rejects_missing_component() -> None:
    raw = _truth_mapping()
    scenarios = cast(dict[str, dict[str, Any]], raw["scenarios"])
    del scenarios["active"]["sender"]

    with pytest.raises(ValueError, match="must define all components"):
        component_truth_from_mapping(raw)


def _test_config(path: Path) -> Path:
    config = json.loads(
        (REPO_ROOT / "benchmarks/configs/mechanism_specificity_v2.json").read_text(
            encoding="utf-8"
        )
    )
    config["evaluation_phases"]["development"]["required_paired_seeds"] = 4
    config["evaluation_phases"]["independent_holdout"]["required_paired_seeds"] = 4
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _assert_cli_atomically_writes_auditable_outputs(
    tmp_path: Path, *, suffix: str
) -> None:
    evidence = tmp_path / f"evidence{suffix}"
    records = _records()
    if suffix == ".csv":
        records.to_csv(evidence, index=False)
    else:
        records.to_parquet(evidence, index=False)
    config = _test_config(tmp_path / "config.json")
    truth = REPO_ROOT / "benchmarks/truth/component_truth_matrix.yaml"
    output = tmp_path / f"result-{suffix.removeprefix('.')}"

    payload = run_evaluation(
        evidence,
        config,
        truth,
        output,
        evaluation_phase="independent_holdout",
    )

    assert output.is_dir()
    assert (output / "metrics.csv").is_file()
    assert (output / "metrics.json").is_file()
    written = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert payload["evaluation_completed_for_supplied_input"] is True
    assert payload["gate_passed_for_supplied_input"] is True
    assert written["default_switch_allowed"] is False
    assert written["does_not_claim_unsupplied_seed_campaigns"] is True
    assert written["expected_seed_count"] == 4
    assert len(written["known_edge_ids"]) == 3
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))

    with pytest.raises(FileExistsError, match="already exists"):
        run_evaluation(
            evidence,
            config,
            truth,
            output,
            evaluation_phase="independent_holdout",
        )


def test_cli_atomically_writes_auditable_csv_and_json(tmp_path: Path) -> None:
    _assert_cli_atomically_writes_auditable_outputs(tmp_path, suffix=".csv")


def test_cli_accepts_parquet_evidence(tmp_path: Path) -> None:
    _assert_cli_atomically_writes_auditable_outputs(tmp_path, suffix=".parquet")


def test_cli_rejects_truth_content_outside_frozen_hash(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.csv"
    _records().to_csv(evidence, index=False)
    config = _test_config(tmp_path / "config.json")
    source_truth = REPO_ROOT / "benchmarks/truth/component_truth_matrix.yaml"
    altered_truth = tmp_path / "component_truth_matrix.yaml"
    altered_truth.write_text(
        source_truth.read_text(encoding="utf-8")
        + "\n# altered after preregistration\n",
        encoding="utf-8",
    )
    output = tmp_path / "result"

    with pytest.raises(ValueError, match="truth SHA256"):
        run_evaluation(
            evidence,
            config,
            altered_truth,
            output,
            evaluation_phase="development",
        )

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))
