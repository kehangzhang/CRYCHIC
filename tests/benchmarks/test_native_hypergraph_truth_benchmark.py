from __future__ import annotations

import copy

import pandas as pd
import yaml  # type: ignore[import-untyped]

from benchmarks.metrics.mechanism_specificity import component_truth_from_mapping
from benchmarks.simulation.mechanism_specificity import (
    generate_mechanism_specificity_evidence_from_lineages,
)
from benchmarks.simulation.native_hypergraph_truth_benchmark import (
    DEFAULT_CONFIG,
    _config,
    holdout_seed_lineages,
    replicate_metrics,
    score_candidate_universe,
    summarize_metrics,
)


def _evidence() -> pd.DataFrame:
    components = {
        "active": (0.8, 0.8, 0.8, 0.7),
        "global_null": (0.0, 0.0, 0.0, 0.0),
        "abundance_only": (0.0, 0.8, 0.8, 0.0),
        "ligand_only": (0.8, 0.8, 0.8, 0.0),
        "target_only": (0.0, 0.8, 0.0, 0.0),
        "receiver_autonomous": (0.0, 0.8, 0.0, 0.0),
        "receptor_knockout": (0.8, 0.0, 0.8, 0.0),
    }
    records = []
    for seed in (11, 12):
        for scenario, values in components.items():
            availability, receptor, sender, integrated = values
            records.append(
                {
                    "seed": seed,
                    "known_edge_id": "edge",
                    "scenario": scenario,
                    "availability_effect": availability,
                    "receptor_gate": receptor,
                    "sender_effect": sender,
                    "integrated_lr_effect": integrated,
                    "status": "observed",
                }
            )
    return pd.DataFrame.from_records(records)


def test_native_hyperedge_exact_ap_separates_ligand_only_near_miss() -> None:
    candidates = score_candidate_universe(_evidence())
    _, macro = replicate_metrics(candidates)
    summary = macro.groupby("method", observed=True).mean(numeric_only=True)

    assert candidates.groupby("method", observed=True).size().nunique() == 1
    assert summary.loc["crychic_native_hyperedge", "exact_hyperedge_ap"] == 1.0
    assert summary.loc["clique_expansion", "partial_canonical_lr_ap"] == 1.0
    assert (
        summary.loc["crychic_native_hyperedge", "partial_canonical_lr_ap"]
        < summary.loc["clique_expansion", "partial_canonical_lr_ap"]
    )


def test_frozen_gate_uses_paired_seed_differences_and_equal_coverage() -> None:
    config, _ = _config(DEFAULT_CONFIG)
    test_config = copy.deepcopy(config)
    test_config["holdout"]["seed_count"] = 2
    test_config["gate"]["minimum_complete_seeds"] = 2
    _, macro = replicate_metrics(score_candidate_universe(_evidence()))

    summary, paired, gate = summarize_metrics(macro, test_config)

    assert gate["status"] == "PASS"
    assert summary["coverage_mean"].eq(1.0).all()
    assert paired["exact_difference_ci_low"].gt(0.2).all()


def test_custom_holdout_lineages_preserve_complete_g15_evidence() -> None:
    config, truth_path = _config(DEFAULT_CONFIG)
    truth = component_truth_from_mapping(
        yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    )
    lineages = holdout_seed_lineages(config)[:2]

    generated = generate_mechanism_specificity_evidence_from_lineages(
        phase="native_hypergraph_truth_test",
        seed_lineages=lineages,
        truth=truth,
    )

    assert generated.seed_count == 2
    assert generated.row_count == 2 * 3 * 7
    assert set(generated.evidence["status"]) == {"observed"}
