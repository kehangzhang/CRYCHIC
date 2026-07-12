"""Run reproducible v0.1 synthetic positive and negative controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.metrics.exploratory import paired_descriptive_effects
from benchmarks.simulation.generate import (
    AUTONOMOUS_TARGETS,
    GENES,
    TARGETS,
    Scenario,
    simulate_ccc,
)
from crychic.attribution import attribute_target_prior
from crychic.availability import estimate_bundle_availability
from crychic.data import InputSchema, validate_anndata
from crychic.design import ContextGraph, balanced_contrast
from crychic.pseudobulk import aggregate_pseudobulk
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.response import estimate_gene_response

SCENARIOS: tuple[Scenario, ...] = (
    "active",
    "global_null",
    "abundance_only",
    "receiver_autonomous",
    "ligand_only",
    "target_only",
    "receptor_knockout",
)
MAIN_INTERACTION = "CXCL10_CXCR3"
EDGE_TARGETS: dict[str, tuple[str, ...]] = {
    MAIN_INTERACTION: TARGETS,
    "GAPDH_RPLP0": ("MALAT1",),
    "MALAT1_ACTB": ("RPLP0",),
}


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _scenario_seed(seed: int, scenario: Scenario) -> int:
    payload = f"crychic-v0.1-negative-control:{seed}:{scenario}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _interaction(interaction_id: str, ligand: str, receptor: str) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=interaction_id,
        ligand_name=ligand,
        receptor_name=receptor,
        ligand_subunits=(ligand,),
        receptor_subunits=(receptor,),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="crychic_synthetic_fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        pathway="synthetic",
    )


def _toy_bundle() -> ResourceBundle:
    interactions = (
        _interaction(MAIN_INTERACTION, "CXCL10", "CXCR3"),
        _interaction("GAPDH_RPLP0", "GAPDH", "RPLP0"),
        _interaction("MALAT1_ACTB", "MALAT1", "ACTB"),
    )
    return ResourceBundle(
        resource_id="crychic_synthetic_lr",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=interactions,
        mapping_report=MappingReport(
            source_rows=len(interactions),
            loaded_rows=len(interactions),
            mapped_entities=6,
        ),
        manifest_digest="synthetic-no-external-resource",
        source_files=("benchmarks/simulation/run_negative_controls.py",),
        license="CC0-1.0",
        citation="CRYCHIC synthetic benchmark fixture",
    )


def _toy_prior(driver: str, targets: tuple[str, ...]) -> TargetPrior:
    target_ids = tuple(sorted(targets))
    return TargetPrior(
        resource_id=f"synthetic_prior_{driver}",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=target_ids,
        driver_ids=(driver,),
        indptr=(0, len(target_ids)),
        target_indices=tuple(range(len(target_ids))),
        weights=tuple(1.0 for _ in target_ids),
        ranks=tuple(range(1, len(target_ids) + 1)),
        direction=1,
        evidence="synthetic_truth",
        mapping_report=MappingReport(
            source_rows=len(target_ids),
            loaded_rows=len(target_ids),
            mapped_entities=len(target_ids) + 1,
        ),
        manifest_digest="synthetic-no-external-resource",
    )


def _mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else math.nan


def _relative_change(reference: float, target: float) -> float:
    if not math.isfinite(reference) or reference == 0 or not math.isfinite(target):
        return math.nan
    return target / reference - 1.0


def _positive_explained_fraction(
    positive_response: np.ndarray, predicted: np.ndarray
) -> float:
    denominator = float(np.dot(positive_response, positive_response))
    if denominator == 0:
        return 0.0
    residual = positive_response - predicted
    value = 1.0 - float(np.dot(residual, residual)) / denominator
    return float(np.clip(value, 0.0, 1.0))


def _edge_rows(
    *,
    scenario: Scenario,
    availability: pd.DataFrame,
    response_effects: pd.Series,
) -> list[dict[str, Any]]:
    sender_receiver = availability.loc[
        (availability["sender"].astype(str) == "Sender")
        & (availability["receiver"].astype(str) == "Receiver")
    ].copy()
    state_effects = paired_descriptive_effects(
        sender_receiver,
        value="availability_state",
        edge_keys=("interaction_id",),
        context_key="condition",
        reference="ctrl",
        target="stim",
    ).set_index("interaction_id")
    ecosystem_effects = paired_descriptive_effects(
        sender_receiver,
        value="availability_ecosystem",
        edge_keys=("interaction_id",),
        context_key="condition",
        reference="ctrl",
        target="stim",
    ).set_index("interaction_id")

    records: list[dict[str, Any]] = []
    for interaction in _toy_bundle().interactions:
        interaction_id = interaction.interaction_id
        edge = sender_receiver.loc[
            sender_receiver["interaction_id"].astype(str) == interaction_id
        ]
        context_means = edge.groupby("condition", observed=True)[
            [
                "ligand_availability",
                "receptor_availability",
                "availability_state",
                "availability_ecosystem",
            ]
        ].mean()
        targets = EDGE_TARGETS[interaction_id]
        signed_response = response_effects.reindex(GENES).to_numpy(dtype=float)
        stim_receptor = float(context_means.loc["stim", "receptor_availability"])
        _, attribution = attribute_target_prior(
            _toy_prior(interaction.ligand_name, targets),
            GENES,
            {interaction.ligand_name: stim_receptor},
            signed_response,
            lambda2=1e-6,
        )
        explained = _positive_explained_fraction(
            attribution.positive_response, attribution.predicted
        )
        target_effect = _mean(response_effects.reindex(targets))
        downstream = 1.0 - math.exp(-max(target_effect, 0.0))
        state_effect = float(state_effects.loc[interaction_id, "effect"])
        ecosystem_effect = float(ecosystem_effects.loc[interaction_id, "effect"])
        availability_change = max(state_effect, 0.0)
        recovery_score = float(np.cbrt(availability_change * downstream * explained))
        records.append(
            {
                "scenario": scenario,
                "interaction_id": interaction_id,
                "truth_active": scenario == "active"
                and interaction_id == MAIN_INTERACTION,
                "state_effect": state_effect,
                "ecosystem_effect": ecosystem_effect,
                "target_response_effect": target_effect,
                "downstream_activity_proxy": downstream,
                "attribution_explained_fraction": explained,
                "attribution_coefficient": float(attribution.coefficients[0]),
                "signed_residual_ratio": float(
                    np.linalg.norm(attribution.residual)
                    / max(np.linalg.norm(attribution.signed_response), 1e-12)
                ),
                "recovery_score": recovery_score,
                "ctrl_ligand_availability": float(
                    context_means.loc["ctrl", "ligand_availability"]
                ),
                "stim_ligand_availability": float(
                    context_means.loc["stim", "ligand_availability"]
                ),
                "ctrl_receptor_availability": float(
                    context_means.loc["ctrl", "receptor_availability"]
                ),
                "stim_receptor_availability": stim_receptor,
                "ctrl_state": float(context_means.loc["ctrl", "availability_state"]),
                "stim_state": float(context_means.loc["stim", "availability_state"]),
                "ctrl_ecosystem": float(
                    context_means.loc["ctrl", "availability_ecosystem"]
                ),
                "stim_ecosystem": float(
                    context_means.loc["stim", "availability_ecosystem"]
                ),
            }
        )
    return records


def _run_scenario(
    scenario: Scenario,
    *,
    seed: int,
    n_subjects: int,
    mean_cells_per_sample: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario_seed = _scenario_seed(seed, scenario)
    simulated = simulate_ccc(
        scenario,
        n_subjects=n_subjects,
        mean_cells_per_sample=mean_cells_per_sample,
        seed=scenario_seed,
    )
    validated = validate_anndata(
        simulated.adata,
        InputSchema(
            context_keys=("condition",),
            species="human",
            gene_namespace="HGNC symbol",
        ),
    )
    aggregate = aggregate_pseudobulk(validated, min_cells=3)
    contrast = balanced_contrast(
        ("stim",), ("ctrl",), name="stim-v-ctrl", family="synthetic"
    )
    response = estimate_gene_response(
        aggregate,
        ContextGraph.chain(("ctrl", "stim")),
        receivers=("Receiver",),
        contrasts=(contrast,),
        min_samples_per_context=2,
        min_subjects_per_context=2,
    )
    response_rows = response.contrasts.loc[
        (response.contrasts["receiver"] == "Receiver")
        & (response.contrasts["contrast"] == "stim-v-ctrl")
    ]
    if set(response_rows["status"]) != {"ok"}:
        reasons = sorted(set(response_rows["reason_code"].dropna()))
        raise RuntimeError(f"scenario {scenario} response is not estimable: {reasons}")
    response_effects = response_rows.set_index("gene")["effect"]

    availability_result = estimate_bundle_availability(
        aggregate,
        _toy_bundle(),
        context_keys=("condition",),
        min_pooled_availability=0.0,
    )
    edge_rows = _edge_rows(
        scenario=scenario,
        availability=availability_result.sample_interactions,
        response_effects=response_effects,
    )
    main = next(row for row in edge_rows if row["interaction_id"] == MAIN_INTERACTION)

    metadata = aggregate.unit_metadata
    proportions = metadata.loc[
        metadata["cell_type"].isin(["Sender", "Receiver"])
    ].copy()
    proportion_means = proportions.groupby(["condition", "cell_type"], observed=True)[
        "cell_proportion"
    ].mean()
    scenario_metrics: dict[str, Any] = {
        "scenario": scenario,
        "scenario_seed": scenario_seed,
        "n_cells": simulated.adata.n_obs,
        "n_subjects": n_subjects,
        "expected_state_change": simulated.expected_state_change,
        "expected_ecosystem_change": simulated.expected_ecosystem_change,
        "expected_receiver_response": simulated.expected_receiver_response,
        "expected_integrated_edge": simulated.expected_integrated_edge,
        "truth_response_effect_mean": _mean(
            response_effects.reindex(simulated.response_genes)
        ),
        "prior_target_response_effect_mean": _mean(response_effects.reindex(TARGETS)),
        "autonomous_response_effect_mean": _mean(
            response_effects.reindex(AUTONOMOUS_TARGETS)
        ),
        "main_state_effect": main["state_effect"],
        "main_state_relative_change": _relative_change(
            main["ctrl_state"], main["stim_state"]
        ),
        "main_ecosystem_effect": main["ecosystem_effect"],
        "main_ecosystem_relative_change": _relative_change(
            main["ctrl_ecosystem"], main["stim_ecosystem"]
        ),
        "main_attribution_explained_fraction": main["attribution_explained_fraction"],
        "main_signed_residual_ratio": main["signed_residual_ratio"],
        "main_recovery_score": main["recovery_score"],
        "main_ctrl_state": main["ctrl_state"],
        "main_stim_state": main["stim_state"],
        "main_ctrl_receptor_availability": main["ctrl_receptor_availability"],
        "main_stim_receptor_availability": main["stim_receptor_availability"],
        "ctrl_sender_proportion": float(proportion_means.loc["ctrl", "Sender"]),
        "stim_sender_proportion": float(proportion_means.loc["stim", "Sender"]),
        "ctrl_receiver_proportion": float(proportion_means.loc["ctrl", "Receiver"]),
        "stim_receiver_proportion": float(proportion_means.loc["stim", "Receiver"]),
    }
    return scenario_metrics, edge_rows


def _binary_truth_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return {
            "auroc": math.nan,
            "average_precision": math.nan,
            "n_positive": positive,
            "n_negative": negative,
        }

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = float(ranks[labels].sum())
    auroc = (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)

    thresholds = np.unique(scores)[::-1]
    true_positive = 0
    selected = 0
    previous_recall = 0.0
    average_precision = 0.0
    for threshold in thresholds:
        tied = scores == threshold
        selected += int(tied.sum())
        true_positive += int(labels[tied].sum())
        recall = true_positive / positive
        precision = true_positive / selected
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return {
        "auroc": float(auroc),
        "average_precision": float(average_precision),
        "n_positive": positive,
        "n_negative": negative,
    }


def _check(
    scenario: str,
    name: str,
    observed: float,
    expectation: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "check": name,
        "observed": observed,
        "expectation": expectation,
        "passed": bool(passed),
    }


def _checks(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = metrics.set_index("scenario")
    active_score = float(rows.loc["active", "main_recovery_score"])
    checks = [
        _check(
            "active",
            "positive_control_response",
            float(rows.loc["active", "prior_target_response_effect_mean"]),
            "> 0.5 log1p(CPM)",
            float(rows.loc["active", "prior_target_response_effect_mean"]) > 0.5,
        ),
        _check(
            "active",
            "positive_control_state",
            float(rows.loc["active", "main_state_effect"]),
            "> 0.02",
            float(rows.loc["active", "main_state_effect"]) > 0.02,
        ),
        _check(
            "global_null",
            "integrated_score_below_positive_control",
            float(rows.loc["global_null", "main_recovery_score"]),
            "< 50% of active positive control",
            float(rows.loc["global_null", "main_recovery_score"]) < 0.5 * active_score,
        ),
        _check(
            "abundance_only",
            "state_stability",
            abs(float(rows.loc["abundance_only", "main_state_relative_change"])),
            "absolute relative change < 0.15",
            abs(float(rows.loc["abundance_only", "main_state_relative_change"])) < 0.15,
        ),
        _check(
            "abundance_only",
            "ecosystem_sensitivity",
            abs(float(rows.loc["abundance_only", "main_ecosystem_relative_change"])),
            "absolute relative change > 0.10",
            abs(float(rows.loc["abundance_only", "main_ecosystem_relative_change"]))
            > 0.10,
        ),
        _check(
            "receiver_autonomous",
            "autonomous_response_detected",
            float(rows.loc["receiver_autonomous", "autonomous_response_effect_mean"]),
            "> 0.5 log1p(CPM)",
            float(rows.loc["receiver_autonomous", "autonomous_response_effect_mean"])
            > 0.5,
        ),
        _check(
            "receiver_autonomous",
            "prior_attribution_remains_weak",
            float(
                rows.loc[
                    "receiver_autonomous",
                    "main_attribution_explained_fraction",
                ]
            ),
            "positive response explained fraction < 0.35",
            float(
                rows.loc[
                    "receiver_autonomous",
                    "main_attribution_explained_fraction",
                ]
            )
            < 0.35,
        ),
        _check(
            "receiver_autonomous",
            "availability_separated_from_response",
            abs(float(rows.loc["receiver_autonomous", "main_state_effect"])),
            "absolute state effect < 0.10",
            abs(float(rows.loc["receiver_autonomous", "main_state_effect"])) < 0.10,
        ),
        _check(
            "ligand_only",
            "availability_without_downstream_response",
            float(rows.loc["ligand_only", "main_state_effect"]),
            "state effect > 0.02 while target response < 0.5",
            float(rows.loc["ligand_only", "main_state_effect"]) > 0.02
            and float(rows.loc["ligand_only", "prior_target_response_effect_mean"])
            < 0.5,
        ),
        _check(
            "ligand_only",
            "recovery_below_positive_control",
            float(rows.loc["ligand_only", "main_recovery_score"]),
            "< 60% of active positive control",
            float(rows.loc["ligand_only", "main_recovery_score"]) < 0.6 * active_score,
        ),
        _check(
            "target_only",
            "downstream_without_availability",
            float(rows.loc["target_only", "prior_target_response_effect_mean"]),
            "response > 0.5 and absolute state effect < 0.10",
            float(rows.loc["target_only", "prior_target_response_effect_mean"]) > 0.5
            and abs(float(rows.loc["target_only", "main_state_effect"])) < 0.10,
        ),
        _check(
            "target_only",
            "recovery_below_positive_control",
            float(rows.loc["target_only", "main_recovery_score"]),
            "< 50% of active positive control",
            float(rows.loc["target_only", "main_recovery_score"]) < 0.5 * active_score,
        ),
        _check(
            "receptor_knockout",
            "receptor_gate_zero",
            float(rows.loc["receptor_knockout", "main_stim_receptor_availability"]),
            "stimulated receptor availability == 0",
            math.isclose(
                float(
                    rows.loc[
                        "receptor_knockout",
                        "main_stim_receptor_availability",
                    ]
                ),
                0.0,
                abs_tol=1e-12,
            ),
        ),
        _check(
            "receptor_knockout",
            "interaction_state_gate_zero",
            float(rows.loc["receptor_knockout", "main_stim_state"]),
            "stimulated interaction state == 0",
            math.isclose(
                float(rows.loc["receptor_knockout", "main_stim_state"]),
                0.0,
                abs_tol=1e-12,
            ),
        ),
    ]
    return pd.DataFrame(checks)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_negative_controls(
    output_dir: Path,
    *,
    seed: int = 20260712,
    n_subjects: int = 8,
    mean_cells_per_sample: int = 240,
) -> dict[str, Any]:
    """Run all v0.1 controls and write compact, truth-scoped summaries."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if n_subjects < 2:
        raise ValueError("n_subjects must be at least 2")
    if mean_cells_per_sample < 30:
        raise ValueError("mean_cells_per_sample must be at least 30")

    scenario_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        metrics, edges = _run_scenario(
            scenario,
            seed=seed,
            n_subjects=n_subjects,
            mean_cells_per_sample=mean_cells_per_sample,
        )
        scenario_records.append(metrics)
        edge_records.extend(edges)

    metrics_table = pd.DataFrame(scenario_records)
    edge_table = pd.DataFrame(edge_records)
    checks_table = _checks(metrics_table)
    truth_metrics = _binary_truth_metrics(
        edge_table["truth_active"].to_numpy(dtype=bool),
        edge_table["recovery_score"].to_numpy(dtype=float),
    )
    summary: dict[str, Any] = {
        "benchmark": "crychic_v0.1_synthetic_negative_controls",
        "seed": seed,
        "scenarios": list(SCENARIOS),
        "n_subjects": n_subjects,
        "mean_cells_per_sample": mean_cells_per_sample,
        "all_checks_passed": bool(checks_table["passed"].all()),
        "checks_passed": int(checks_table["passed"].sum()),
        "checks_total": len(checks_table),
        "edge_truth_metrics": {
            "scope": "synthetic_edge_level_truth_only",
            "interpretation": "single-positive smoke metric; not calibration",
            **truth_metrics,
        },
        "real_data_truth_metrics_included": False,
        "formal_inference": {
            "p_values": None,
            "q_values": None,
            "fdr": None,
            "reason_code": "v0_1_inferential_disabled",
        },
        "score_semantics": (
            "exploratory recovery score; not a communication probability"
        ),
        "versions": {
            "python": platform.python_version(),
            "CRYCHIC": _package_version("CRYCHIC"),
            "anndata": _package_version("anndata"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(output_dir / "scenario_metrics.csv", index=False)
    edge_table.to_csv(output_dir / "edge_metrics.csv", index=False)
    checks_table.to_csv(output_dir / "checks.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _json_value(summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--mean-cells-per-sample", type=int, default=240)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_negative_controls(
        args.output_dir,
        seed=args.seed,
        n_subjects=args.n_subjects,
        mean_cells_per_sample=args.mean_cells_per_sample,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
