"""Validate the frozen suggest_v5 benchmark scope below 100,000 cells."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "crychic-suggest-v5-under100k-contract-v1"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "suggest_v5_under100k_v1.json"
)


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the frozen scope contract."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_contract(payload)
    return payload


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return value


def validate_contract(payload: Mapping[str, Any]) -> None:
    """Reject scope drift and paper-design substitutions."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected scope-contract schema")
    eligibility = _require_mapping(payload.get("eligibility"), "eligibility")
    upper_bound = eligibility.get("strict_upper_bound")
    if upper_bound != 100_000:
        raise ValueError("strict upper bound must remain 100000")

    benchmarks = _require_sequence(payload.get("benchmarks"), "benchmarks")
    records: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(benchmarks):
        record = _require_mapping(raw, f"benchmarks[{index}]")
        benchmark_id = record.get("benchmark_id")
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise ValueError("every benchmark needs a non-empty benchmark_id")
        if benchmark_id in records:
            raise ValueError(f"duplicate benchmark_id: {benchmark_id}")
        records[benchmark_id] = record
        count = record.get("observed_single_cells")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"invalid observed cell count for {benchmark_id}")
        if record.get("eligible") is not True or count >= upper_bound:
            raise ValueError(f"ineligible benchmark was included: {benchmark_id}")
        _require_sequence(record.get("methods"), f"{benchmark_id}.methods")
        _require_sequence(record.get("metrics"), f"{benchmark_id}.metrics")
        _require_mapping(record.get("design"), f"{benchmark_id}.design")
        _require_mapping(record.get("source"), f"{benchmark_id}.source")

    required_ids = {
        "tensor_cell2cell_planted_12context",
        "staccato_condition_batch_simulation",
        "dcst_subject_count_sweep",
        "dcst_receiver_cell_count_sweep",
        "scaccordion_pdac",
        "scaccordion_kidney_aki",
        "scaccordion_rcc",
        "signed_estimand_contract",
        "score_generator_engine_crossover",
    }
    if set(records) != required_ids:
        missing = sorted(required_ids.difference(records))
        unexpected = sorted(set(records).difference(required_ids))
        raise ValueError(
            f"benchmark scope drift; missing={missing}, unexpected={unexpected}"
        )

    subject_design = _require_mapping(
        records["dcst_subject_count_sweep"]["design"], "DCST subject design"
    )
    if subject_design.get("subjects_per_condition") != list(range(5, 50, 5)):
        raise ValueError("DCST eligible subject sweep must be 5 through 45")
    if subject_design.get("excluded_subjects_per_condition") != [50, 55]:
        raise ValueError("DCST 100k and 110k settings must remain excluded")
    for subjects in subject_design["subjects_per_condition"]:
        if 2 * subjects * 2 * 500 >= upper_bound:
            raise ValueError("DCST subject sweep contains an oversized setting")

    receiver_design = _require_mapping(
        records["dcst_receiver_cell_count_sweep"]["design"],
        "DCST receiver design",
    )
    if receiver_design.get("receiver_cells_in_condition_2") != list(
        range(50, 501, 50)
    ):
        raise ValueError("DCST receiver-cell sweep must be 50 through 500")

    scaccordion_counts = {
        benchmark_id: records[benchmark_id]["observed_single_cells"]
        for benchmark_id in (
            "scaccordion_pdac",
            "scaccordion_kidney_aki",
            "scaccordion_rcc",
        )
    }
    if scaccordion_counts != {
        "scaccordion_pdac": 57_530,
        "scaccordion_kidney_aki": 76_020,
        "scaccordion_rcc": 50_236,
    }:
        raise ValueError("eligible scACCorDiON cohort counts drifted")

    excluded = _require_sequence(
        payload.get("excluded_scaccordion_cohorts"),
        "excluded_scaccordion_cohorts",
    )
    if len(excluded) != 4:
        raise ValueError("all four oversized scACCorDiON cohorts must be recorded")
    if any(
        _require_mapping(item, "excluded cohort").get("observed_single_cells", 0)
        < upper_bound
        for item in excluded
    ):
        raise ValueError("an eligible cohort was placed in the exclusion list")


def scope_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact rows used by reports and notebooks."""

    validate_contract(payload)
    rows = []
    for record in payload["benchmarks"]:
        rows.append(
            {
                "benchmark_id": record["benchmark_id"],
                "section": record["section"],
                "kind": record["kind"],
                "observed_single_cells": record["observed_single_cells"],
                "fidelity": record["fidelity"],
                "method_count": len(record["methods"]),
                "metric_count": len(record["metrics"]),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    payload = load_contract(args.config)
    print(json.dumps(scope_rows(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
