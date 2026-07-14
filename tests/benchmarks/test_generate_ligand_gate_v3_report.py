from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")

from benchmarks.report.generate_ligand_gate_v3_report import generate_report

SCHEMA = "crychic-public-family-common-g1.5-campaign-v1"
CONTROLS = (
    "global_null",
    "abundance_only",
    "ligand_only",
    "target_only",
    "receiver_autonomous",
    "receptor_knockout",
)
VIEWS = (
    ("state", "member_unresolved"),
    ("state", "sender_resolved"),
    ("ecosystem", "member_unresolved"),
    ("ecosystem", "sender_resolved"),
)
CAMPAIGN_CHECKS = (
    "active_and_paired_ligand_only_executed",
    "all_generated_inputs_complete_paired",
    "all_observed_receiver_programs_use_subject_equal_reference_transform",
    "all_runs_used_public_crossfit_workflow",
    "eligible_for_campaign_metric_interpretation",
    "executed_full_seven_scenario_contract",
    "executed_multiple_seeds",
    "frozen_contract_contains_all_seven_g1_5_scenarios",
    "frozen_contract_contains_multiple_known_edges",
    "frozen_contract_contains_multiple_seeds",
    "not_estimable_components_never_count_as_pass",
)


def _control_metric(scenario: str, rate: float, n_seeds: int) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "n_seed_runs": n_seeds,
        "n_seed_mode_evaluations": n_seeds * 2,
        "any_positive_integrated_family_rate": rate,
        "any_positive_raw_integrated_family_rate": rate,
        "mean_positive_integrated_families": rate,
        "mean_positive_raw_integrated_families": rate,
    }


def _reason(scenario: str) -> str:
    if scenario == "ligand_only":
        return "family_not_selected"
    if scenario == "receptor_knockout":
        return "receptor_interaction_ineligible"
    return "ligand_contrast_not_supported"


def _active_pairs(seeds: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "seed_id": seed,
            "known_edge_id": "synthetic_edge_a",
            "mode": mode,
            "score_kind": score_kind,
            "active_minus_ligand_only_margin": 0.01,
            "active_recovered": True,
            "coverage_loss": 0.0,
            "active_mean": 0.01,
            "reference_mean": 0.0,
        }
        for seed in seeds
        for mode, score_kind in VIEWS
    ]


def _summary(
    *,
    seeds: tuple[str, ...],
    scenarios: tuple[str, ...],
    target_rate: float,
    scope: str,
    source_hash: str,
) -> dict[str, Any]:
    records = []
    for seed in seeds:
        for scenario in scenarios:
            record: dict[str, Any] = {
                "seed_id": seed,
                "scenario": scenario,
                "profile": "quick",
            }
            if scenario != "active":
                record["known_edge_scores"] = [
                    {"reason_counts": {_reason(scenario): 4}}
                ]
            records.append(record)
    controls = [
        _control_metric(
            scenario,
            target_rate if scenario == "target_only" else 0.0,
            len(seeds),
        )
        for scenario in scenarios
        if scenario != "active"
    ]
    campaign = len(scenarios) == 7
    checks = {name: campaign for name in CAMPAIGN_CHECKS}
    if not campaign:
        checks.update(
            {
                "active_and_paired_ligand_only_executed": True,
                "all_generated_inputs_complete_paired": True,
                (
                    "all_observed_receiver_programs_use_subject_equal_"
                    "reference_transform"
                ): True,
                "all_runs_used_public_crossfit_workflow": True,
                "frozen_contract_contains_all_seven_g1_5_scenarios": True,
                "frozen_contract_contains_multiple_known_edges": True,
                "frozen_contract_contains_multiple_seeds": True,
                "not_estimable_components_never_count_as_pass": True,
            }
        )
    return {
        "schema_version": SCHEMA,
        "campaign_id": "test_campaign",
        "scope": scope,
        "claims": {
            "development_only": True,
            "biological_validation": False,
            "complete_pipeline_oof_certification": False,
            "default_switch_allowed": False,
            "family_common_full_oof_certification": False,
            "method_superiority": False,
        },
        "checks": checks,
        "registry": {
            "all_preregistered_scenarios": ["active", *CONTROLS],
            "all_preregistered_seed_ids": [
                "public-g15-001",
                "public-g15-002",
                "public-g15-003",
            ],
            "executed_scenarios": list(scenarios),
            "executed_seed_ids": list(seeds),
            "profile": "quick",
            "sha256": "a" * 64,
        },
        "known_edges": [
            {
                "known_edge_id": "synthetic_edge_a",
                "harmonized_interaction_id": "harmonized_edge_a",
                "ligand": "LIGA",
                "receptor": "RECA",
            },
            {
                "known_edge_id": "synthetic_edge_b",
                "harmonized_interaction_id": "harmonized_edge_b",
                "ligand": "LIGB",
                "receptor": "RECB",
            },
        ],
        "records": records,
        "source_sha256": {
            "benchmarks/report/publication.mplstyle": source_hash,
        },
        "aggregate_metrics": {
            "control_family_false_positive_and_selection": controls,
            "active_vs_ligand_only": {"paired_records": _active_pairs(seeds)},
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    style = repo_root / "benchmarks/report/publication.mplstyle"
    source_hash = hashlib.sha256(style.read_bytes()).hexdigest()
    all_scenarios = ("active", *CONTROLS)
    pre = tmp_path / "pre.json"
    full7 = tmp_path / "full7.json"
    seed003 = tmp_path / "seed003.json"
    _write_json(
        pre,
        _summary(
            seeds=("public-g15-001", "public-g15-002"),
            scenarios=all_scenarios,
            target_rate=1.0,
            scope="development_full_seven_scenario_diagnostic",
            source_hash=source_hash,
        ),
    )
    _write_json(
        full7,
        _summary(
            seeds=("public-g15-001", "public-g15-002"),
            scenarios=all_scenarios,
            target_rate=0.0,
            scope="development_full_seven_scenario_diagnostic",
            source_hash=source_hash,
        ),
    )
    _write_json(
        seed003,
        _summary(
            seeds=("public-g15-003",),
            scenarios=("active", "ligand_only", "target_only", "receptor_knockout"),
            target_rate=0.0,
            scope="single_seed_public_workflow_debug_excluded_from_campaign",
            source_hash=source_hash,
        ),
    )
    return pre, seed003, full7


def _tree_sha256(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generate_ligand_gate_report_preserves_claim_boundary(tmp_path: Path) -> None:
    pre, seed003, full7 = _write_inputs(tmp_path)

    output = tmp_path / "report"
    reproduced = tmp_path / "report_reproduced"
    result = generate_report(
        pre_summary=pre,
        seed003_summary=seed003,
        full7_summary=full7,
        output_dir=output,
        generated_at="2026-07-14T00:00:00+00:00",
    )
    generate_report(
        pre_summary=pre,
        seed003_summary=seed003,
        full7_summary=full7,
        output_dir=reproduced,
        generated_at="2026-07-14T00:00:00+00:00",
    )

    assert result["artifact_count"] == 18
    assert _tree_sha256(output) == _tree_sha256(reproduced)
    manifest = json.loads((output / "report_manifest.json").read_text())
    assert manifest["claims"] == {
        "biological_validation": False,
        "complete_pipeline_oof_certification": False,
        "family_common_full_oof_certification": False,
        "default_switch_allowed": False,
        "method_superiority": False,
    }
    metrics = json.loads((output / "metrics_summary.json").read_text())
    assert metrics["claims"] == manifest["claims"]
    target_rows = metrics["target_only_pre_post"]
    assert {
        (row["snapshot"], row["estimand"]): row["any_positive_family_rate"]
        for row in target_rows
    } == {
        ("pre_gate", "paired_integrated"): 1.0,
        ("pre_gate", "raw_integrated"): 1.0,
        ("full7", "paired_integrated"): 0.0,
        ("full7", "raw_integrated"): 0.0,
        ("seed003", "paired_integrated"): 0.0,
        ("seed003", "raw_integrated"): 0.0,
    }
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "no real biological validation" in report
    assert "does not establish method superiority" in report
    assert "--generated-at 2026-07-14T00:00:00+00:00" in report
    for stem in (
        "figure01_target_only_pre_post",
        "figure02_active_recovery_margins",
        "figure03_control_specificity_gate_reasons",
    ):
        assert (output / f"source_data/{stem}.csv").is_file()
        for suffix in ("png", "pdf", "svg"):
            assert (output / f"figures/{stem}.{suffix}").stat().st_size > 0
    svg = (output / "figures/figure01_target_only_pre_post.svg").read_text()
    assert "2026-07-14T00:00:00+00:00" in svg
    pdf = (output / "figures/figure01_target_only_pre_post.pdf").read_bytes()
    assert b"D:20260714000000Z" in pdf


def test_rejects_seed003_without_explicit_debug_scope(tmp_path: Path) -> None:
    pre, seed003, full7 = _write_inputs(tmp_path)
    payload = json.loads(seed003.read_text())
    payload["scope"] = "development_full_seven_scenario_diagnostic"
    _write_json(seed003, payload)

    with pytest.raises(ValueError, match="excluded from campaign claims"):
        generate_report(
            pre_summary=pre,
            seed003_summary=seed003,
            full7_summary=full7,
            output_dir=tmp_path / "rejected",
            generated_at="2026-07-14T00:00:00+00:00",
        )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("case", "message"),
    (
        ("registry", "same registry SHA256"),
        ("known_edges", "same known-edge set"),
        ("campaign_check", "failed campaign contract checks"),
        ("campaign_seed", "exactly seeds 001 and 002"),
    ),
)
def test_rejects_cross_snapshot_campaign_contract_drift(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    pre, seed003, full7 = _write_inputs(tmp_path)
    target = pre if case in {"campaign_check", "campaign_seed"} else full7
    payload = json.loads(target.read_text())
    if case == "registry":
        payload["registry"]["sha256"] = "b" * 64
    elif case == "known_edges":
        payload["known_edges"][0]["known_edge_id"] = "changed_edge"
    elif case == "campaign_check":
        payload["checks"]["executed_multiple_seeds"] = False
    elif case == "campaign_seed":
        payload["records"][0]["seed_id"] = "public-g15-003"
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(case)
    _write_json(target, payload)

    with pytest.raises(ValueError, match=message):
        generate_report(
            pre_summary=pre,
            seed003_summary=seed003,
            full7_summary=full7,
            output_dir=tmp_path / "rejected",
            generated_at="2026-07-14T00:00:00+00:00",
        )
