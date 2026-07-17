from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import benchmarks.simulation.graph_fusion_g2 as simulation_module
import numpy as np
import pandas as pd
import pytest
from benchmarks.metrics.graph_fusion_g2 import (
    FUSED_METHOD,
    METHODS,
    NO_TOPOLOGY_METHOD,
    UNFUSED_METHOD,
    WRONG_TOPOLOGY_METHOD,
    G2GateSpecification,
    driver_family_context_macro_auprc,
    evaluate_graph_fusion_g2_gate,
)
from benchmarks.simulation.graph_fusion_g2 import (
    FROZEN_COLLINEARITY_NAMES,
    FROZEN_SIGNALS,
    FROZEN_SNR_NAMES,
    FROZEN_TOPOLOGIES,
    load_g2_manifest,
    run_g2_problem,
    simulate_g2_problem,
)
from benchmarks.simulation.run_graph_fusion_g2 import (
    OUTPUT_FILENAMES,
    render_g2_report,
    run_g2_campaign,
)

from crychic.attribution import solve_graph_fused_nonnegative_elastic_net

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "benchmarks/configs/graph_fusion_g2_v1.json"


def _paired_records(
    scenario_ids: tuple[str, ...],
    *,
    replicates: int,
    fused_difference: float = 0.05,
    no_topology_difference: float = 0.0,
    wrong_topology_difference: float = -0.01,
) -> pd.DataFrame:
    method_differences = {
        UNFUSED_METHOD: 0.0,
        FUSED_METHOD: fused_difference,
        NO_TOPOLOGY_METHOD: no_topology_difference,
        WRONG_TOPOLOGY_METHOD: wrong_topology_difference,
    }
    return pd.DataFrame.from_records(
        [
            {
                "scenario_id": scenario_id,
                "replicate": replicate,
                "scenario_seed": 10_000 * scenario_index + replicate,
                "method": method,
                "macro_auprc": 0.5 + difference,
                "status": "observed",
                "reason_code": None,
            }
            for scenario_index, scenario_id in enumerate(scenario_ids)
            for replicate in range(replicates)
            for method, difference in method_differences.items()
        ]
    )


def _small_gate(scenario_ids: tuple[str, ...]) -> G2GateSpecification:
    return G2GateSpecification(
        scenario_ids=scenario_ids,
        minimum_paired_replicates_per_scenario=3,
        bootstrap_replicates=200,
        bootstrap_seed=71,
    )


def test_manifest_freezes_complete_factorial_and_gate_policy() -> None:
    manifest = load_g2_manifest(CONFIG_PATH)

    assert len(manifest.scenarios) == 16
    assert {scenario.topology for scenario in manifest.scenarios} == set(
        FROZEN_TOPOLOGIES
    )
    assert {scenario.signal for scenario in manifest.scenarios} == set(FROZEN_SIGNALS)
    assert {scenario.snr_name for scenario in manifest.scenarios} == set(
        FROZEN_SNR_NAMES
    )
    assert {
        scenario.collinearity_name for scenario in manifest.scenarios
    } == set(FROZEN_COLLINEARITY_NAMES)
    assert len({scenario.scenario_id for scenario in manifest.scenarios}) == 16
    assert manifest.smoke_replicates == 1
    assert manifest.formal_replicates == 200
    assert manifest.gate_specification.minimum_paired_replicates_per_scenario == 200
    assert manifest.gate_specification.improvement_margin == pytest.approx(0.02)
    assert manifest.gate_specification.noninferiority_margin == pytest.approx(-0.02)


def test_manifest_rejects_post_registration_factor_changes(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["scenario_manifest"]["topologies"] = ["product", "chain"]
    poisoned = tmp_path / "poisoned.json"
    poisoned.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen order"):
        load_g2_manifest(poisoned)


def test_problem_is_deterministic_and_all_methods_share_seed_and_problem() -> None:
    manifest = load_g2_manifest(CONFIG_PATH)
    scenario = manifest.scenarios[0]
    first = simulate_g2_problem(manifest, scenario, 0)
    second = simulate_g2_problem(manifest, scenario, 0)

    assert first.problem_id == second.problem_id
    assert first.scenario_seed == second.scenario_seed
    assert first.graph_id != first.wrong_topology_graph_id
    for node in first.graph.nodes:
        np.testing.assert_array_equal(first.responses[node], second.responses[node])
        np.testing.assert_array_equal(
            first.matrices[node].toarray(), second.matrices[node].toarray()
        )

    assert (
        simulation_module.solve_graph_fused_nonnegative_elastic_net
        is solve_graph_fused_nonnegative_elastic_net
    )
    with patch.object(
        simulation_module,
        "solve_graph_fused_nonnegative_elastic_net",
        wraps=solve_graph_fused_nonnegative_elastic_net,
    ) as public_solver:
        results = run_g2_problem(manifest, first)
    assert public_solver.call_count == 4
    assert tuple(results["method"]) == METHODS
    assert results["scenario_seed"].nunique() == 1
    assert results["problem_id"].nunique() == 1
    assert set(results["status"]) == {"observed"}
    scores = results.set_index("method")["macro_auprc"]
    assert scores[NO_TOPOLOGY_METHOD] == scores[UNFUSED_METHOD]


def test_primary_macro_auprc_is_equal_context_and_not_global_prevalence_weighted() -> (
    None
):
    truth = np.asarray([[1, 0, 0], [1, 1, 0]], dtype=bool)
    perfect = np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    imperfect = np.asarray([[1.0, 3.0, 2.0], [3.0, 1.0, 2.0]])

    assert driver_family_context_macro_auprc(truth, perfect) == pytest.approx(1.0)
    assert driver_family_context_macro_auprc(truth, imperfect) < 1.0
    with pytest.raises(ValueError, match="positive and negative"):
        driver_family_context_macro_auprc(
            np.ones((2, 3), dtype=bool), perfect
        )


def test_smoke_is_explicitly_ne_even_when_observed_improvement_is_large() -> None:
    manifest = load_g2_manifest(CONFIG_PATH)
    results = _paired_records(
        tuple(scenario.scenario_id for scenario in manifest.scenarios),
        replicates=1,
        fused_difference=0.2,
    )

    report = evaluate_graph_fusion_g2_gate(
        results, manifest.gate_specification
    )

    assert report.status == "NE"
    assert report.gate_passed is None
    assert report.primary_interval is None
    assert report.control_intervals == ()
    assert report.reason_code == (
        "insufficient_complete_paired_replicates_per_scenario"
    )


def test_paired_bootstrap_gate_passes_and_fails_only_from_frozen_margins() -> None:
    scenarios = ("scenario-a", "scenario-b")
    spec = _small_gate(scenarios)

    passed = evaluate_graph_fusion_g2_gate(
        _paired_records(scenarios, replicates=3), spec
    )
    assert passed.status == "PASS"
    assert passed.gate_passed is True
    assert passed.primary_interval is not None
    assert passed.primary_interval.ci_lower == pytest.approx(0.05)
    assert all(
        interval.ci_lower >= spec.noninferiority_margin
        for interval in passed.control_intervals
    )

    failed = evaluate_graph_fusion_g2_gate(
        _paired_records(
            scenarios,
            replicates=3,
            wrong_topology_difference=-0.03,
        ),
        spec,
    )
    assert failed.status == "FAIL"
    assert failed.gate_passed is False
    wrong = next(
        interval
        for interval in failed.control_intervals
        if interval.comparison == "wrong_topology_minus_unfused"
    )
    assert wrong.ci_lower < spec.noninferiority_margin


def test_failed_or_missing_pair_cannot_fall_back_to_unpaired_gate() -> None:
    scenarios = ("scenario-a", "scenario-b")
    spec = _small_gate(scenarios)
    records = _paired_records(scenarios, replicates=3)
    mask = (
        records["scenario_id"].eq("scenario-b")
        & records["replicate"].eq(2)
        & records["method"].eq(FUSED_METHOD)
    )
    records.loc[mask, "status"] = "failed"
    records.loc[mask, "macro_auprc"] = np.nan
    records.loc[mask, "reason_code"] = "solver_failed"

    report = evaluate_graph_fusion_g2_gate(records, spec)

    assert report.status == "NE"
    assert report.complete_pairs_by_comparison[
        "fused_correct_minus_unfused"
    ]["scenario-b"] == 2

    mismatched = _paired_records(scenarios, replicates=3)
    mismatch = (
        mismatched["scenario_id"].eq("scenario-a")
        & mismatched["replicate"].eq(0)
        & mismatched["method"].eq(FUSED_METHOD)
    )
    mismatched.loc[mismatch, "scenario_seed"] += 1
    with pytest.raises(ValueError, match="share one scenario_seed"):
        evaluate_graph_fusion_g2_gate(mismatched, spec)


def test_atomic_smoke_report_publishes_ne_and_complete_provenance(
    tmp_path: Path,
) -> None:
    manifest = load_g2_manifest(CONFIG_PATH)
    scenario_ids = tuple(scenario.scenario_id for scenario in manifest.scenarios)
    synthetic_results = _paired_records(scenario_ids, replicates=1)
    output = tmp_path / "g2-smoke"

    with patch(
        "benchmarks.simulation.run_graph_fusion_g2.run_g2_simulation",
        return_value=synthetic_results,
    ):
        observed = run_g2_campaign(
            config_path=CONFIG_PATH,
            output_dir=output,
        )

    assert observed == output.resolve()
    assert {path.name for path in output.iterdir()} == set(OUTPUT_FILENAMES)
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    run_manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    report = (output / "report.md").read_text(encoding="utf-8")
    assert gate["status"] == "NE"
    assert gate["gate_passed"] is None
    assert run_manifest["gate_status"] == "NE"
    assert run_manifest["benchmark"]["config_sha256"] == manifest.source_sha256
    assert len(cast(dict[str, str], run_manifest["source_sha256"])) == 5
    assert "Gate status: **NE**" in report
    assert "cannot support a default switch" in report

    with pytest.raises(FileExistsError):
        run_g2_campaign(config_path=CONFIG_PATH, output_dir=output)


def test_report_keeps_secondary_metrics_outside_primary_gate() -> None:
    manifest = load_g2_manifest(CONFIG_PATH)
    scenario_ids = tuple(scenario.scenario_id for scenario in manifest.scenarios)
    results = _paired_records(scenario_ids, replicates=1)
    gate = evaluate_graph_fusion_g2_gate(results, manifest.gate_specification)

    report = render_g2_report(
        manifest,
        results,
        gate,
        profile="smoke",
        replicates_per_scenario=1,
    )

    assert "driver_family x context` macro-AUPRC" in report
    assert "Secondary AUROC, coefficient RMSE, and jump localization" in report
    assert "they never replace the primary gate" in report
