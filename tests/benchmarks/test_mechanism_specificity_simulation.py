from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]
from benchmarks.metrics.mechanism_specificity import (
    REQUIRED_SCENARIOS,
    ComponentTruthMatrix,
    component_truth_from_mapping,
    evaluate_mechanism_specificity,
    specification_from_config,
)
from benchmarks.simulation.mechanism_specificity import (
    DEVELOPMENT_PHASE,
    EDGE_PROFILES,
    EVIDENCE_COLUMNS,
    GENERATOR_SCHEMA_VERSION,
    HOLDOUT_PHASE,
    PHASE_SEED_NAMESPACES,
    generate_mechanism_specificity_evidence,
    phase_seed_lineages,
)
from benchmarks.simulation.run_mechanism_specificity import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    publish_generated_campaign,
    run_campaign,
)

from crychic.scoring import (
    apply_incremental_downstream_functional,
    fit_incremental_downstream_functional,
    mechanistic_strength,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_CONFIG_PATH = REPO_ROOT / "benchmarks/configs/mechanism_specificity_v2.json"
LIVE_CONFIG_PATH = REPO_ROOT / "benchmarks/configs/mechanism_specificity_v3.json"
TRUTH_PATH = REPO_ROOT / "benchmarks/truth/component_truth_matrix.yaml"
SUMMARY_PATH = REPO_ROOT / "benchmarks/results/g1_5_v2_summary.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth() -> ComponentTruthMatrix:
    value = yaml.safe_load(TRUTH_PATH.read_text(encoding="utf-8"))
    return component_truth_from_mapping(cast(dict[str, object], value))


def _config() -> dict[str, object]:
    return cast(
        dict[str, object], json.loads(LIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    )


def _write_test_config(path: Path, *, phase: str, seed_count: int) -> Path:
    config = _config()
    phases = cast(dict[str, dict[str, object]], config["evaluation_phases"])
    phases[phase]["required_paired_seeds"] = seed_count
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_published_summary_is_locked_to_frozen_inputs_and_sources() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    inputs = cast(dict[str, str], summary["inputs"])

    assert inputs == {
        "config_sha256": _sha256(HISTORICAL_CONFIG_PATH),
        "truth_sha256": _sha256(TRUTH_PATH),
        # v2 is a historical locked campaign. The live generator now uses the
        # sample-keyed incremental API and must publish under a new campaign.
        "generator_sha256": (
            "8288453ba365723cbbcf1590c0ee62524c14d03c578e5bc6e4b2fd9cbb4d2a4c"
        ),
        "runner_sha256": (
            "871a89b7ea061f3ca7ac5592fde422a05d37ab0d8a5ce9fc35fca07eadf95d85"
        ),
    }
    assert inputs["generator_sha256"] != _sha256(
        REPO_ROOT / "benchmarks/simulation/mechanism_specificity.py"
    )
    assert inputs["runner_sha256"] != _sha256(
        REPO_ROOT / "benchmarks/simulation/run_mechanism_specificity.py"
    )
    assert summary["truth_set_id"] == _truth().truth_set_id
    assert summary["default_switch_allowed"] is False
    assert summary["real_data_accuracy_claim"] is False

    phases = {
        str(item["evaluation_phase"]): cast(dict[str, object], item)
        for item in cast(list[dict[str, object]], summary["phases"])
    }
    assert phases[DEVELOPMENT_PHASE]["evidence_rows"] == 50 * 3 * 7
    assert phases[HOLDOUT_PHASE]["evidence_rows"] == 200 * 3 * 7
    assert phases[HOLDOUT_PHASE]["may_tune_candidate"] is False
    assert phases[HOLDOUT_PHASE]["tuning_performed"] is False
    assert phases[HOLDOUT_PHASE]["holdout_tuning_forbidden"] is True
    assert all(bool(item["overall_gate_passed"]) for item in phases.values())


def test_live_generator_uses_a_distinct_v3_contract_and_default_output() -> None:
    config = _config()
    contract = cast(dict[str, object], config["generator_contract"])

    assert config["schema_version"] == "crychic-mechanism-specificity-v3"
    assert GENERATOR_SCHEMA_VERSION == (
        "crychic-g1.5-sample-keyed-structural-zero-development-generator-v3"
    )
    assert contract["schema_version"] == GENERATOR_SCHEMA_VERSION
    assert contract["phase_seed_namespaces"] == PHASE_SEED_NAMESPACES
    assert all(":v3:" in namespace for namespace in PHASE_SEED_NAMESPACES.values())
    assert DEFAULT_CONFIG == LIVE_CONFIG_PATH
    assert DEFAULT_OUTPUT_ROOT.name == "g1_5_sample_keyed_development_v3"
    assert (
        _sha256(HISTORICAL_CONFIG_PATH)
        == cast(
            dict[str, str],
            json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["inputs"],
        )["config_sha256"]
    )


def test_small_generator_is_deterministic_and_calls_real_scoring_apis() -> None:
    truth = _truth()
    call_count = len(truth.known_edge_ids) * len(REQUIRED_SCENARIOS)

    with (
        patch(
            "benchmarks.simulation.mechanism_specificity."
            "fit_incremental_downstream_functional",
            wraps=fit_incremental_downstream_functional,
        ) as fit,
        patch(
            "benchmarks.simulation.mechanism_specificity."
            "apply_incremental_downstream_functional",
            wraps=apply_incremental_downstream_functional,
        ) as apply,
        patch(
            "benchmarks.simulation.mechanism_specificity.mechanistic_strength",
            wraps=mechanistic_strength,
        ) as integrate,
    ):
        first = generate_mechanism_specificity_evidence(
            phase=DEVELOPMENT_PHASE,
            seed_count=1,
            truth=truth,
        )

    second = generate_mechanism_specificity_evidence(
        phase=DEVELOPMENT_PHASE,
        seed_count=1,
        truth=truth,
    )
    pd.testing.assert_frame_equal(first.evidence, second.evidence)
    assert first.functional_ids == second.functional_ids
    assert fit.call_count == call_count
    assert apply.call_count == call_count
    assert integrate.call_count == call_count
    assert set(first.application_statuses) == {"observed"}
    audit = first.audit_summary()
    assert audit["model_call_counts"] == {
        "fit_incremental_downstream_functional": call_count,
        "apply_incremental_downstream_functional": call_count,
        "mechanistic_strength": call_count,
    }
    assert audit["gain_denominator_status_counts"] == {
        "positive_receiver_null_loss_ratio_v1": 12,
        "zero_receiver_contrast_structural_zero_v1": 9,
    }
    assert audit["gain_denominator_status_counts_by_scenario"] == {
        scenario: {
            (
                "zero_receiver_contrast_structural_zero_v1"
                if scenario in {"global_null", "abundance_only", "ligand_only"}
                else "positive_receiver_null_loss_ratio_v1"
            ): len(truth.known_edge_ids)
        }
        for scenario in REQUIRED_SCENARIOS
    }


def test_generated_evidence_is_strict_model_derived_and_passes_small_gate() -> None:
    truth = _truth()
    generated = generate_mechanism_specificity_evidence(
        phase=DEVELOPMENT_PHASE,
        seed_count=4,
        truth=truth,
    )
    evidence = generated.evidence

    configured_columns = cast(
        list[str],
        cast(dict[str, object], _config()["evidence_record_contract"])[
            "required_columns"
        ],
    )
    assert tuple(evidence.columns) == EVIDENCE_COLUMNS == tuple(configured_columns)
    assert len(evidence) == 4 * 3 * 7
    assert not evidence.duplicated(["seed", "known_edge_id", "scenario"]).any()
    assert set(evidence["status"]) == {"observed"}
    assert evidence["reason_code"].isna().all()

    active = evidence.loc[evidence["scenario"].eq("active")]
    ligand = evidence.loc[evidence["scenario"].eq("ligand_only")]
    target = evidence.loc[evidence["scenario"].eq("target_only")]
    autonomous = evidence.loc[evidence["scenario"].eq("receiver_autonomous")]
    knockout = evidence.loc[evidence["scenario"].eq("receptor_knockout")]
    assert (active["incremental_downstream_effect"] > 0).all()
    assert (active["integrated_lr_effect"] > 0).all()
    assert (ligand["availability_effect"] > 0).all()
    assert (ligand["incremental_downstream_effect"] == 0).all()
    assert (ligand["integrated_lr_effect"] == 0).all()
    assert (target["receiver_program_effect"] > 0).all()
    assert (target["incremental_downstream_effect"] > 0).all()
    assert (target["integrated_lr_effect"] == 0).all()
    assert (autonomous["receiver_program_effect"] > 0).all()
    assert (autonomous["incremental_downstream_effect"] == 0).all()
    assert (knockout["receptor_gate"] == 0).all()
    assert (knockout["incremental_downstream_effect"] > 0).all()
    assert (knockout["integrated_lr_effect"] == 0).all()

    records = cast(list[dict[str, Any]], evidence.to_dict(orient="records"))
    for row in records:
        profile = EDGE_PROFILES[str(row["known_edge_id"])]
        _, _, expected = mechanistic_strength(
            availability=float(row["availability_effect"])
            * float(row["receptor_gate"]),
            incremental_downstream=float(row["incremental_downstream_effect"]),
            prior_quality=profile.prior_quality,
            sender_weight=float(row["sender_effect"]),
        )
        assert expected is not None
        assert float(row["integrated_lr_effect"]) == pytest.approx(expected, abs=1e-12)

    specification = replace(
        specification_from_config(_config(), evaluation_phase=DEVELOPMENT_PHASE),
        expected_seed_count=4,
    )
    metrics = evaluate_mechanism_specificity(evidence, specification, truth)
    overall = metrics.loc[
        metrics["metric"].eq("g1_5_mechanism_specificity_gate")
        & metrics["aggregation"].eq("overall")
    ].iloc[0]
    assert overall["status"] == "observed"
    assert bool(overall["gate_passed"])


def test_phase_namespaces_are_disjoint_and_scenarios_share_paired_components() -> None:
    development = phase_seed_lineages(DEVELOPMENT_PHASE, 20)
    holdout = phase_seed_lineages(HOLDOUT_PHASE, 20)
    assert {item.seed for item in development}.isdisjoint(
        {item.seed for item in holdout}
    )

    evidence = generate_mechanism_specificity_evidence(
        phase=DEVELOPMENT_PHASE,
        seed_count=3,
        truth=_truth(),
    ).evidence
    active = evidence.loc[evidence["scenario"].eq("active")].set_index(
        ["seed", "known_edge_id"]
    )
    ligand = evidence.loc[evidence["scenario"].eq("ligand_only")].set_index(
        ["seed", "known_edge_id"]
    )
    pd.testing.assert_series_equal(
        active["availability_effect"],
        ligand["availability_effect"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        active["receptor_gate"], ligand["receptor_gate"], check_names=False
    )
    pd.testing.assert_series_equal(
        active["sender_effect"], ligand["sender_effect"], check_names=False
    )


def test_atomic_campaign_bundle_contains_evidence_manifest_and_metrics(
    tmp_path: Path,
) -> None:
    seed_count = 4
    generated = generate_mechanism_specificity_evidence(
        phase=DEVELOPMENT_PHASE,
        seed_count=seed_count,
        truth=_truth(),
    )
    config = _write_test_config(
        tmp_path / "config.json",
        phase=DEVELOPMENT_PHASE,
        seed_count=seed_count,
    )
    output = tmp_path / "campaign"

    result = publish_generated_campaign(
        generated,
        config_path=config,
        truth_path=TRUTH_PATH,
        output_dir=output,
    )

    assert result["gate_passed_for_supplied_input"] is True
    assert (output / "evidence.parquet").is_file()
    assert (output / "generation_manifest.json").is_file()
    assert (output / "metrics/metrics.csv").is_file()
    assert (output / "metrics/metrics.json").is_file()
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["seed_count"] == seed_count
    assert manifest["evidence_rows"] == seed_count * 3 * 7
    assert manifest["phase_policy"]["tuning_performed"] is False
    assert manifest["claims"]["integrated_values_are_scenario_assigned"] is False
    assert manifest["audit"]["application_status_counts"] == {
        "observed": seed_count * 3 * 7
    }
    assert manifest["audit"]["gain_denominator_status_counts"] == {
        "positive_receiver_null_loss_ratio_v1": seed_count * 3 * 4,
        "zero_receiver_contrast_structural_zero_v1": seed_count * 3 * 3,
    }
    assert manifest["frozen_design"]["generator_schema_version"] == (
        GENERATOR_SCHEMA_VERSION
    )
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))

    with pytest.raises(FileExistsError, match="already exists"):
        publish_generated_campaign(
            generated,
            config_path=config,
            truth_path=TRUTH_PATH,
            output_dir=output,
        )


def test_independent_holdout_policy_cannot_enable_tuning(tmp_path: Path) -> None:
    generated = generate_mechanism_specificity_evidence(
        phase=HOLDOUT_PHASE,
        seed_count=2,
        truth=_truth(),
    )
    config = _config()
    phases = cast(dict[str, dict[str, object]], config["evaluation_phases"])
    phases[HOLDOUT_PHASE]["required_paired_seeds"] = 2
    phases[HOLDOUT_PHASE]["may_tune_candidate"] = True
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "holdout"

    with pytest.raises(ValueError, match="must forbid tuning"):
        publish_generated_campaign(
            generated,
            config_path=config_path,
            truth_path=TRUTH_PATH,
            output_dir=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))


def test_live_sample_keyed_runner_cannot_republish_historical_holdout(
    tmp_path: Path,
) -> None:
    generated = generate_mechanism_specificity_evidence(
        phase=HOLDOUT_PHASE,
        seed_count=2,
        truth=_truth(),
    )
    config = _config()
    phases = cast(dict[str, dict[str, object]], config["evaluation_phases"])
    phases[HOLDOUT_PHASE]["required_paired_seeds"] = 2
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="already been inspected"):
        publish_generated_campaign(
            generated,
            config_path=config_path,
            truth_path=TRUTH_PATH,
            output_dir=tmp_path / "holdout",
        )
    with pytest.raises(ValueError, match="already been inspected"):
        run_campaign(
            phase=HOLDOUT_PHASE,
            config_path=config_path,
            truth_path=TRUTH_PATH,
            output_dir=tmp_path / "holdout",
        )


def test_live_generator_rejects_the_historical_v2_config(tmp_path: Path) -> None:
    generated = generate_mechanism_specificity_evidence(
        phase=DEVELOPMENT_PHASE,
        seed_count=2,
        truth=_truth(),
    )

    with pytest.raises(ValueError, match="requires mechanism_specificity_v3"):
        publish_generated_campaign(
            generated,
            config_path=HISTORICAL_CONFIG_PATH,
            truth_path=TRUTH_PATH,
            output_dir=tmp_path / "historical-config",
        )


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("schema_version", "generator schema"),
        ("phase_seed_namespaces", "seed namespaces"),
    ],
)
def test_live_config_binds_generator_schema_and_seed_namespaces(
    tmp_path: Path,
    field_name: str,
    message: str,
) -> None:
    generated = generate_mechanism_specificity_evidence(
        phase=DEVELOPMENT_PHASE,
        seed_count=2,
        truth=_truth(),
    )
    config = _config()
    contract = cast(dict[str, object], config["generator_contract"])
    contract[field_name] = "poisoned"
    config_path = tmp_path / f"{field_name}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        publish_generated_campaign(
            generated,
            config_path=config_path,
            truth_path=TRUTH_PATH,
            output_dir=tmp_path / field_name,
        )
