"""Generate cell-level counts for CRYCHIC negative-control scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

Scenario = Literal[
    "active",
    "global_null",
    "abundance_only",
    "receiver_autonomous",
    "ligand_only",
    "target_only",
    "receptor_knockout",
]

GENES = (
    "CXCL10",
    "CXCR3",
    "CCL5",
    "CCR5",
    "VEGFA",
    "FLT1",
    "CXCL12",
    "CXCR4",
    "EGF",
    "EGFR",
    "ISG15",
    "IFIT1",
    "MX1",
    "OAS1",
    "STAT1",
    "B2M",
    "GAPDH",
    "RPLP0",
    "MALAT1",
    "ACTB",
    "FOS",
    "JUN",
    "DUSP1",
)
TARGETS = ("ISG15", "IFIT1", "MX1", "OAS1", "STAT1", "B2M")
AUTONOMOUS_TARGETS = ("FOS", "JUN", "DUSP1")
_GENE_INDEX = {gene: index for index, gene in enumerate(GENES)}
DECOY_INTERACTIONS = (
    ("CCL5", "CCR5"),
    ("VEGFA", "FLT1"),
    ("CXCL12", "CXCR4"),
    ("EGF", "EGFR"),
)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Synthetic AnnData together with edge and component truth."""

    adata: ad.AnnData
    scenario: Scenario
    seed: int
    active_interaction: str | None
    response_genes: tuple[str, ...]
    expected_state_change: bool
    expected_ecosystem_change: bool
    expected_receiver_response: bool
    expected_integrated_edge: bool


def _scenario_truth(scenario: Scenario) -> tuple[bool, bool, bool, bool]:
    return {
        "active": (True, True, True, True),
        "global_null": (False, False, False, False),
        "abundance_only": (False, True, False, False),
        "receiver_autonomous": (False, False, True, False),
        "ligand_only": (True, True, False, False),
        "target_only": (False, False, True, False),
        "receptor_knockout": (False, False, True, False),
    }[scenario]


def _mean_profile(cell_type: str, condition: str, scenario: Scenario) -> np.ndarray:
    rates: np.ndarray = np.full(len(GENES), 0.03, dtype=float)
    rates[_GENE_INDEX["ISG15"]] = 0.10
    rates[_GENE_INDEX["IFIT1"]] = 0.08
    rates[_GENE_INDEX["MX1"]] = 0.08
    rates[_GENE_INDEX["OAS1"]] = 0.08
    rates[_GENE_INDEX["STAT1"]] = 0.12
    rates[_GENE_INDEX["B2M"]] = 0.20
    rates[_GENE_INDEX["GAPDH"]] = 2.5
    rates[_GENE_INDEX["RPLP0"]] = 2.2
    rates[_GENE_INDEX["MALAT1"]] = 3.0
    rates[_GENE_INDEX["ACTB"]] = 2.0
    rates[_GENE_INDEX["FOS"]] = 0.08
    rates[_GENE_INDEX["JUN"]] = 0.08
    rates[_GENE_INDEX["DUSP1"]] = 0.08
    if cell_type == "Sender":
        rates[_GENE_INDEX["CXCL10"]] = 1.0
        rates[_GENE_INDEX["CXCR3"]] = 0.02
        for ligand, _ in DECOY_INTERACTIONS:
            rates[_GENE_INDEX[ligand]] = 0.7
    elif cell_type == "Receiver":
        rates[_GENE_INDEX["CXCL10"]] = 0.02
        rates[_GENE_INDEX["CXCR3"]] = 0.9
        for _, receptor in DECOY_INTERACTIONS:
            rates[_GENE_INDEX[receptor]] = 0.7
    else:
        rates[_GENE_INDEX["CXCL10"]] = 0.05
        rates[_GENE_INDEX["CXCR3"]] = 0.05

    if condition == "stim":
        if scenario in {"active", "ligand_only"} and cell_type == "Sender":
            rates[_GENE_INDEX["CXCL10"]] *= 8.0
        if (
            scenario
            in {
                "active",
                "target_only",
                "receptor_knockout",
            }
            and cell_type == "Receiver"
        ):
            for gene in TARGETS:
                rates[_GENE_INDEX[gene]] *= 5.0
        if scenario == "receiver_autonomous" and cell_type == "Receiver":
            for gene in AUTONOMOUS_TARGETS:
                rates[_GENE_INDEX[gene]] *= 8.0
        if scenario == "receptor_knockout" and cell_type == "Receiver":
            rates[_GENE_INDEX["CXCR3"]] = 0.0
    return rates


def simulate_ccc(
    scenario: Scenario,
    *,
    n_subjects: int = 8,
    mean_cells_per_sample: int = 240,
    seed: int = 20260712,
) -> SimulationResult:
    """Simulate a paired experiment without cell-level replication claims."""
    valid: tuple[Scenario, ...] = (
        "active",
        "global_null",
        "abundance_only",
        "receiver_autonomous",
        "ligand_only",
        "target_only",
        "receptor_knockout",
    )
    if scenario not in valid:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {valid}")
    if n_subjects < 2:
        raise ValueError("n_subjects must be at least 2")
    if mean_cells_per_sample < 30:
        raise ValueError("mean_cells_per_sample must be at least 30")

    rng = np.random.default_rng(seed)
    cell_types = ("Sender", "Receiver", "Bystander")
    baseline_proportions = np.array([0.30, 0.35, 0.35])
    counts_parts: list[np.ndarray] = []
    obs_parts: list[pd.DataFrame] = []

    for subject_index in range(n_subjects):
        subject = f"S{subject_index + 1:02d}"
        subject_scale = float(rng.lognormal(mean=0.0, sigma=0.15))
        for condition in ("ctrl", "stim"):
            proportions = baseline_proportions.copy()
            if scenario == "abundance_only" and condition == "stim":
                # Change the sender-receiver abundance product enough to make the
                # ecosystem proxy diagnostic while leaving within-type state fixed.
                proportions = np.array([0.70, 0.10, 0.20])
            concentration = proportions * 120.0
            realized = rng.dirichlet(concentration)
            total_cells = max(30, int(rng.poisson(mean_cells_per_sample)))
            sizes = rng.multinomial(total_cells, realized)
            sample_id = f"{subject}:{condition}"

            for cell_type, n_cells in zip(cell_types, sizes, strict=True):
                if n_cells == 0:
                    continue
                mean = _mean_profile(cell_type, condition, scenario) * subject_scale
                cell_depth = rng.lognormal(mean=0.0, sigma=0.35, size=n_cells)
                expected = cell_depth[:, None] * mean[None, :]
                dispersion = 0.10
                gamma_shape = 1.0 / dispersion
                gamma_scale = expected * dispersion
                latent = rng.gamma(shape=gamma_shape, scale=gamma_scale)
                part: np.ndarray = np.asarray(
                    rng.poisson(latent), dtype=np.int32
                )
                counts_parts.append(part)
                obs_parts.append(
                    pd.DataFrame(
                        {
                            "sample_id": sample_id,
                            "subject_id": subject,
                            "condition": condition,
                            "cell_type": cell_type,
                        },
                        index=[
                            f"{sample_id}:{cell_type}:{cell_index:04d}"
                            for cell_index in range(n_cells)
                        ],
                    )
                )

    counts = np.vstack(counts_parts)
    obs = pd.concat(obs_parts, axis=0)
    matrix = sparse.csr_matrix(counts)
    var = pd.DataFrame(index=pd.Index(GENES, name="gene_symbol"))
    adata = ad.AnnData(X=matrix.copy(), obs=obs, var=var)
    adata.layers["counts"] = matrix
    adata.uns["simulation_truth"] = {
        "scenario": scenario,
        "seed": seed,
        "active_interaction": "CXCL10_CXCR3",
        "target_genes": list(TARGETS),
        "expressed_decoy_interactions": [list(pair) for pair in DECOY_INTERACTIONS],
        "autonomous_response_genes": list(AUTONOMOUS_TARGETS),
        "cell_level_values_are_not_independent_replicates": True,
    }

    state, ecosystem, response, integrated = _scenario_truth(scenario)
    response_genes = (
        AUTONOMOUS_TARGETS if scenario == "receiver_autonomous" else TARGETS
    )
    return SimulationResult(
        adata=adata,
        scenario=scenario,
        seed=seed,
        active_interaction="CXCL10_CXCR3" if integrated else None,
        response_genes=response_genes,
        expected_state_change=state,
        expected_ecosystem_change=ecosystem,
        expected_receiver_response=response,
        expected_integrated_edge=integrated,
    )
