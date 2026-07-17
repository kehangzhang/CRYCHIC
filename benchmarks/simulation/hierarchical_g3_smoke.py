"""Diagnostic-only small simulation of hierarchical FDR candidate decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from crychic.inference.hierarchical import (
    HierarchicalFDRReleaseStatus,
    HypothesisPValueRecord,
    evaluate_hierarchical_fdr,
    freeze_hierarchical_fdr_spec,
)
from crychic.inference.hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisCoverageStatus,
    HypothesisDeclaration,
    HypothesisRole,
    freeze_hypothesis_universe,
)

SCHEMA_VERSION = "crychic-hierarchical-g3-smoke-diagnostic-v1"
DEFAULT_SEED = 20260715
DEFAULT_REPETITIONS = 200
N_PRIMARY = 8
N_CHILDREN = 3
N_ACTIVE_FAMILIES = 2
_SOURCE_PATHS = (
    "benchmarks/simulation/hierarchical_g3_smoke.py",
    "src/crychic/inference/hierarchical.py",
    "src/crychic/inference/hypotheses.py",
)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    signal: bool
    block_correlation: float
    primary_shift: float
    child_shift: float


SCENARIOS = (
    Scenario("global_null_independent", False, 0.0, 0.0, 0.0),
    Scenario("global_null_positive_block", False, 0.45, 0.0, 0.0),
    Scenario("mixed_sparse_independent", True, 0.0, 2.8, 3.0),
    Scenario("mixed_sparse_negative_block", True, -0.20, 2.8, 3.0),
)


@dataclass(frozen=True, slots=True)
class SimulationContract:
    universe: FrozenHypothesisUniverse
    primary_ids: tuple[str, ...]
    child_ids_by_primary: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ReplicateSummary:
    scenario: str
    primary_discoveries: int
    primary_false_discoveries: int
    primary_true_discoveries: int
    selected_primary_families: int
    selected_children: int
    selected_family_average_child_fdp: float
    selected_signal_family_child_power_sum: float
    selected_signal_family_count: int
    end_to_end_true_child_discoveries: int
    secondary_test_level: float
    release_status: str
    formal_q_values_present: bool


def _primary(index: int) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint="driver_family_receiver_context_omnibus_v1",
        contrast_name="all_contexts_v1",
        receiver=f"Receiver-{index}",
        family_id=f"family-{index}",
        mode="state",
        role=HypothesisRole.PRIMARY,
        multiplicity_family="g3-smoke-primary",
    )


def _child(parent: HypothesisDeclaration, index: int) -> HypothesisDeclaration:
    return HypothesisDeclaration(
        endpoint="family_common_integrated_lr_context_effect_v1",
        contrast_name=f"posthoc-{index}",
        receiver=parent.receiver,
        family_id=parent.family_id,
        mode=parent.mode,
        role=HypothesisRole.SECONDARY,
        multiplicity_family="g3-smoke-secondary",
        parent_key=parent.hypothesis_key,
    )


@lru_cache(maxsize=1)
def _contract() -> SimulationContract:
    primary = tuple(_primary(index) for index in range(N_PRIMARY))
    children = tuple(
        tuple(_child(parent, index) for index in range(N_CHILDREN))
        for parent in primary
    )
    universe = freeze_hypothesis_universe(
        (*primary, *(child for family in children for child in family)),
        universe_name="hierarchical-g3-smoke-diagnostic-v1",
    )
    return SimulationContract(
        universe=universe,
        primary_ids=tuple(item.hypothesis_id for item in primary),
        child_ids_by_primary=tuple(
            tuple(item.hypothesis_id for item in family) for family in children
        ),
    )


def _scenario(name: str) -> Scenario:
    matched = tuple(item for item in SCENARIOS if item.name == name)
    if len(matched) != 1:
        raise ValueError(f"unknown smoke scenario: {name!r}")
    return matched[0]


def _block_cholesky(correlation: float) -> np.ndarray:
    size = N_CHILDREN + 1
    covariance: np.ndarray = np.full(
        (size, size), correlation, dtype=np.float64
    )
    np.fill_diagonal(covariance, 1.0)
    result: np.ndarray = np.linalg.cholesky(covariance)
    return result


def _simulate_p_values(
    scenario: Scenario,
    *,
    replicate: int,
    seed: int,
) -> tuple[dict[str, float], frozenset[str], frozenset[str]]:
    contract = _contract()
    scenario_index = next(
        index for index, item in enumerate(SCENARIOS) if item.name == scenario.name
    )
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, scenario_index, replicate))
    )
    active_family_indexes = (
        frozenset(range(N_ACTIVE_FAMILIES)) if scenario.signal else frozenset()
    )
    true_primary = frozenset(
        contract.primary_ids[index] for index in active_family_indexes
    )
    true_children = frozenset(
        contract.child_ids_by_primary[index][0] for index in active_family_indexes
    )
    cholesky = (
        None
        if scenario.block_correlation == 0.0
        else _block_cholesky(scenario.block_correlation)
    )
    p_values: dict[str, float] = {}
    for family_index, primary_id in enumerate(contract.primary_ids):
        noise = rng.standard_normal(N_CHILDREN + 1)
        if cholesky is not None:
            noise = cholesky @ noise
        shifts: np.ndarray = np.zeros(N_CHILDREN + 1, dtype=np.float64)
        if family_index in active_family_indexes:
            shifts[0] = scenario.primary_shift
            shifts[1] = scenario.child_shift
        family_p = ndtr(-(noise + shifts))
        p_values[primary_id] = float(family_p[0])
        for child_index, child_id in enumerate(
            contract.child_ids_by_primary[family_index]
        ):
            p_values[child_id] = float(family_p[child_index + 1])
    return p_values, true_primary, true_children


def _p_value_records(
    scenario: Scenario,
    replicate: int,
    p_values: dict[str, float],
) -> tuple[HypothesisPValueRecord, ...]:
    return tuple(
        HypothesisPValueRecord(
            hypothesis_id=declaration.hypothesis_id,
            source_result_id=(
                f"diagnostic:{scenario.name}:replicate={replicate}:"
                f"{declaration.hypothesis_id}"
            ),
            status=HypothesisCoverageStatus.OBSERVED,
            p_value=p_values[declaration.hypothesis_id],
        )
        for declaration in _contract().universe.declarations
    )


def _run_replicate(task: tuple[str, int, int]) -> ReplicateSummary:
    scenario_name, replicate, seed = task
    scenario = _scenario(scenario_name)
    contract = _contract()
    p_values, true_primary, true_children = _simulate_p_values(
        scenario,
        replicate=replicate,
        seed=seed,
    )
    result = evaluate_hierarchical_fdr(
        contract.universe,
        _p_value_records(scenario, replicate, p_values),
        spec=freeze_hierarchical_fdr_spec(),
    )
    if result.release_status is not HierarchicalFDRReleaseStatus.CANDIDATE_ONLY:
        raise RuntimeError("diagnostic smoke unexpectedly crossed a release boundary")
    if result.q_value_release_allowed or any(
        row.q_value is not None or row.rejected_at_alpha is not None
        for row in result.records
    ):
        raise RuntimeError("diagnostic smoke unexpectedly produced formal decisions")

    rows = {row.hypothesis_id: row for row in result.records}
    selected_primary = {
        hypothesis_id
        for hypothesis_id in contract.primary_ids
        if rows[hypothesis_id].candidate_rejected_at_alpha
    }
    selected_children = {
        child_id
        for family in contract.child_ids_by_primary
        for child_id in family
        if rows[child_id].candidate_rejected_at_alpha
    }
    false_primary = selected_primary.difference(true_primary)
    selected_family_fdp_sum = 0.0
    selected_signal_power_sum = 0.0
    selected_signal_family_count = 0
    for family_index, primary_id in enumerate(contract.primary_ids):
        if primary_id not in selected_primary:
            continue
        family_children = set(contract.child_ids_by_primary[family_index])
        discoveries = selected_children.intersection(family_children)
        true_family_children = true_children.intersection(family_children)
        false_discoveries = discoveries.difference(true_family_children)
        selected_family_fdp_sum += len(false_discoveries) / max(len(discoveries), 1)
        if true_family_children:
            selected_signal_power_sum += len(
                discoveries.intersection(true_family_children)
            ) / len(true_family_children)
            selected_signal_family_count += 1
    selected_family_average_fdp = selected_family_fdp_sum / max(
        len(selected_primary), 1
    )
    return ReplicateSummary(
        scenario=scenario.name,
        primary_discoveries=len(selected_primary),
        primary_false_discoveries=len(false_primary),
        primary_true_discoveries=len(selected_primary.intersection(true_primary)),
        selected_primary_families=len(selected_primary),
        selected_children=len(selected_children),
        selected_family_average_child_fdp=selected_family_average_fdp,
        selected_signal_family_child_power_sum=selected_signal_power_sum,
        selected_signal_family_count=selected_signal_family_count,
        end_to_end_true_child_discoveries=len(
            selected_children.intersection(true_children)
        ),
        secondary_test_level=result.secondary_test_level,
        release_status=result.release_status.value,
        formal_q_values_present=False,
    )


def _source_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in _SOURCE_PATHS
    }


def _scenario_summary(
    scenario: Scenario,
    records: Sequence[ReplicateSummary],
) -> dict[str, object]:
    n_records = len(records)
    if not n_records:
        raise ValueError("scenario summary requires replicate records")
    primary_fdp = tuple(
        item.primary_false_discoveries / max(item.primary_discoveries, 1)
        for item in records
    )
    primary_truth_count = N_ACTIVE_FAMILIES if scenario.signal else 0
    true_child_count = N_ACTIVE_FAMILIES if scenario.signal else 0
    selected_signal_count = sum(
        item.selected_signal_family_count for item in records
    )
    selected_signal_power_sum = sum(
        item.selected_signal_family_child_power_sum for item in records
    )
    release_counts = Counter(item.release_status for item in records)
    return {
        "scenario": scenario.name,
        "truth": "mixed_sparse_signal" if scenario.signal else "global_null",
        "dependence": (
            "independent"
            if scenario.block_correlation == 0.0
            else "gaussian_copula_equicorrelated_within_family_block"
        ),
        "block_correlation": scenario.block_correlation,
        "repetitions": n_records,
        "primary_fdr": float(np.mean(primary_fdp)),
        "primary_type_i_any_false_rejection_probability": float(
            np.mean([item.primary_false_discoveries > 0 for item in records])
        ),
        "primary_null_rejection_rate": sum(
            item.primary_false_discoveries for item in records
        )
        / (n_records * (N_PRIMARY - primary_truth_count)),
        "primary_power": (
            None
            if not primary_truth_count
            else sum(item.primary_true_discoveries for item in records)
            / (n_records * primary_truth_count)
        ),
        "selected_family_average_child_fdp": float(
            np.mean([item.selected_family_average_child_fdp for item in records])
        ),
        "selected_family_average_child_power": (
            None
            if not selected_signal_count
            else selected_signal_power_sum / selected_signal_count
        ),
        "active_family_end_to_end_child_power": (
            None
            if not true_child_count
            else sum(item.end_to_end_true_child_discoveries for item in records)
            / (n_records * true_child_count)
        ),
        "mean_selected_primary_families": float(
            np.mean([item.selected_primary_families for item in records])
        ),
        "mean_selected_children": float(
            np.mean([item.selected_children for item in records])
        ),
        "mean_secondary_test_level": float(
            np.mean([item.secondary_test_level for item in records])
        ),
        "candidate_release_status_counts": dict(sorted(release_counts.items())),
        "formal_q_values_present": any(
            item.formal_q_values_present for item in records
        ),
    }


def run_smoke(
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    jobs: int = 1,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Run deterministic candidate-only simulations without constructing a G3 gate."""

    if isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be an integer >= 1")
    if isinstance(jobs, bool) or jobs < 1:
        raise ValueError("jobs must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    tasks = tuple(
        (scenario.name, replicate, seed)
        for scenario in SCENARIOS
        for replicate in range(repetitions)
    )
    if jobs == 1:
        records = tuple(_run_replicate(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            records = tuple(executor.map(_run_replicate, tasks, chunksize=8))
    scenario_summaries = tuple(
        _scenario_summary(
            scenario,
            tuple(item for item in records if item.scenario == scenario.name),
        )
        for scenario in SCENARIOS
    )
    spec = freeze_hierarchical_fdr_spec()
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "small_diagnostic_algorithm_smoke_candidate_decisions_only_"
            "not_g3_calibration"
        ),
        "configuration": {
            "seed": seed,
            "repetitions_per_scenario": repetitions,
            "jobs": jobs,
            "n_primary_families": N_PRIMARY,
            "n_children_per_family": N_CHILDREN,
            "n_active_families_in_mixed_scenarios": N_ACTIVE_FAMILIES,
            "total_candidate_evaluations": len(tasks),
        },
        "procedure": {
            **spec.to_dict(),
            "calibration_gate_supplied": False,
            "decision_fields_used": "candidate_only",
        },
        "metric_definitions": {
            "primary_fdr": "mean(V_primary/max(R_primary,1))",
            "primary_type_i_any_false_rejection_probability": (
                "probability of at least one false primary candidate rejection"
            ),
            "selected_family_average_child_fdp": (
                "mean over repetitions of sum selected-family FDP/max(R_primary,1)"
            ),
            "selected_family_average_child_power": (
                "mean child power among selected signal-family events only"
            ),
        },
        "scenarios": list(scenario_summaries),
        "provenance": {"source_sha256": _source_sha256()},
        "checks": {
            "all_evaluations_candidate_only": all(
                item.release_status
                == HierarchicalFDRReleaseStatus.CANDIDATE_ONLY.value
                for item in records
            ),
            "formal_q_values_never_present": not any(
                item.formal_q_values_present for item in records
            ),
            "g3_gate_constructed": False,
        },
        "claims": {
            "g3_passed": False,
            "formal_fdr_control_established": False,
            "formal_q_values_released": False,
            "method_superiority": False,
        },
    }


def _default_output() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/hierarchical_g3_smoke_diagnostic_v1.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=_default_output())
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_smoke(
        repetitions=args.repetitions,
        jobs=args.jobs,
        seed=args.seed,
    )
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
