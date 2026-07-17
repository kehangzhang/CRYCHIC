from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from benchmarks.simulation.hierarchical_g3_smoke import (
    SCENARIOS,
    main,
    run_smoke,
)


def test_candidate_smoke_is_deterministic_and_never_formal() -> None:
    first = run_smoke(repetitions=3, jobs=1, seed=41)
    second = run_smoke(repetitions=3, jobs=1, seed=41)

    assert first == second
    assert first["scope"] == (
        "small_diagnostic_algorithm_smoke_candidate_decisions_only_"
        "not_g3_calibration"
    )
    assert first["checks"] == {
        "all_evaluations_candidate_only": True,
        "formal_q_values_never_present": True,
        "g3_gate_constructed": False,
    }
    assert first["claims"] == {
        "g3_passed": False,
        "formal_fdr_control_established": False,
        "formal_q_values_released": False,
        "method_superiority": False,
    }
    scenarios = cast(list[dict[str, Any]], first["scenarios"])
    assert {item["scenario"] for item in scenarios} == {
        item.name for item in SCENARIOS
    }
    for scenario in scenarios:
        assert scenario["candidate_release_status_counts"] == {
            "candidate_only": 3
        }
        assert scenario["formal_q_values_present"] is False
        for metric in (
            "primary_fdr",
            "primary_type_i_any_false_rejection_probability",
            "primary_null_rejection_rate",
            "selected_family_average_child_fdp",
        ):
            assert 0.0 <= float(scenario[metric]) <= 1.0


def test_cli_writes_compact_diagnostic_json(tmp_path: Path) -> None:
    output = tmp_path / "hierarchical_g3_smoke_diagnostic.json"

    main(
        (
            "--repetitions",
            "1",
            "--jobs",
            "1",
            "--seed",
            "7",
            "--output",
            str(output),
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["configuration"]["total_candidate_evaluations"] == len(SCENARIOS)
    assert payload["procedure"]["calibration_gate_supplied"] is False
    assert payload["procedure"]["decision_fields_used"] == "candidate_only"
    assert payload["claims"]["g3_passed"] is False
