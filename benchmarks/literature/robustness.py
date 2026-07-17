"""Shared Dimitrov et al. robustness benchmark primitives.

The paper treats each method's unperturbed top-ranked interactions as an
operational reference, not as biological ground truth.  This module keeps that
distinction explicit and supplies deterministic perturbations that can be used
from both the CRYCHIC and LIANA runtime environments.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

EDGE_COLUMNS = ("source", "target", "ligand", "receptor")
PERTURBATION_KINDS = (
    "cell_subsampling",
    "label_reshuffling",
    "resource_selective",
    "resource_nonselective",
)


@dataclass(frozen=True, slots=True)
class PerturbationSpec:
    """One deterministic robustness perturbation."""

    kind: str
    proportion: float
    replicate: int
    seed: int

    def __post_init__(self) -> None:
        if self.kind not in PERTURBATION_KINDS:
            raise ValueError(f"unsupported perturbation kind: {self.kind!r}")
        if not np.isfinite(self.proportion) or not 0 < self.proportion <= 0.4:
            raise ValueError("perturbation proportion must lie in (0, 0.4]")
        if isinstance(self.replicate, bool) or self.replicate < 1:
            raise ValueError("replicate must be an integer >= 1")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32 integer")


def perturbation_seed(
    kind: str,
    proportion: float,
    replicate: int,
    *,
    base_seed: int = 20260717,
) -> int:
    """Return a stable uint32 seed for one perturbation."""

    payload = f"{base_seed}|{kind}|{proportion:.8f}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def build_design(
    *,
    proportions: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4),
    replicates: int = 5,
    base_seed: int = 20260717,
) -> tuple[PerturbationSpec, ...]:
    """Construct the paper's four 0--40% perturbation series."""

    if isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be an integer >= 1")
    if not proportions or len(set(proportions)) != len(proportions):
        raise ValueError("proportions must be non-empty and unique")
    specs = []
    for kind in PERTURBATION_KINDS:
        for proportion in sorted(proportions):
            for replicate in range(1, replicates + 1):
                specs.append(
                    PerturbationSpec(
                        kind=kind,
                        proportion=proportion,
                        replicate=replicate,
                        seed=perturbation_seed(
                            kind,
                            proportion,
                            replicate,
                            base_seed=base_seed,
                        ),
                    )
                )
    return tuple(specs)


def _labels(data: ad.AnnData, label_key: str) -> np.ndarray:
    if label_key not in data.obs:
        raise ValueError(f"input is missing label column {label_key!r}")
    labels = data.obs[label_key].astype(str).to_numpy()
    if pd.isna(labels).any() or np.any(labels == ""):
        raise ValueError("cell labels must be complete and non-empty")
    if len(np.unique(labels)) < 2:
        raise ValueError("robustness perturbations require at least two labels")
    return labels


def subsample_cells(
    data: ad.AnnData,
    *,
    label_key: str,
    proportion: float,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Remove the requested fraction independently within every cell label."""

    if not 0 < proportion < 1:
        raise ValueError("subsampling proportion must lie in (0, 1)")
    labels = _labels(data, label_key)
    rng = np.random.default_rng(seed)
    retained: list[np.ndarray] = []
    group_audit: dict[str, dict[str, int]] = {}
    for label in sorted(np.unique(labels)):
        positions = np.flatnonzero(labels == label)
        keep_n = max(1, round(len(positions) * (1.0 - proportion)))
        chosen = np.sort(rng.choice(positions, size=keep_n, replace=False))
        retained.append(chosen)
        group_audit[str(label)] = {
            "before": len(positions),
            "after": keep_n,
        }
    keep = np.sort(np.concatenate(retained))
    result = data[keep].copy()
    return result, {
        "requested_proportion": float(proportion),
        "actual_proportion": float(1.0 - result.n_obs / data.n_obs),
        "cells_before": int(data.n_obs),
        "cells_after": int(result.n_obs),
        "groups": group_audit,
    }


def reshuffle_labels(
    data: ad.AnnData,
    *,
    label_key: str,
    proportion: float,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Mislabel a global fraction from the other-label empirical distribution."""

    if not 0 < proportion < 1:
        raise ValueError("reshuffling proportion must lie in (0, 1)")
    labels = _labels(data, label_key)
    rng = np.random.default_rng(seed)
    selected_n = max(1, round(len(labels) * proportion))
    selected = np.sort(rng.choice(len(labels), size=selected_n, replace=False))
    shuffled = labels.copy()
    for position in selected:
        candidates = labels[labels != labels[position]]
        shuffled[position] = str(rng.choice(candidates))
    result = data.copy()
    result.obs[label_key] = pd.Categorical(shuffled)
    mismatch = shuffled != labels
    if not mismatch[selected].all() or mismatch.sum() != len(selected):
        raise RuntimeError("label reshuffling did not produce the intended mismatch")
    return result, {
        "requested_proportion": float(proportion),
        "actual_proportion": float(mismatch.mean()),
        "cells": int(data.n_obs),
        "mismatched_cells": int(mismatch.sum()),
        "replacement_distribution": "empirical labels excluding the original label",
    }


def replace_resource_interactions(
    resource: pd.DataFrame,
    *,
    genes: tuple[str, ...],
    proportion: float,
    seed: int,
    preserved_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace LR rows with unique random pairs drawn from input HVGs."""

    required = {"ligand", "receptor"}
    if required.difference(resource.columns) or resource.empty:
        raise ValueError("resource must contain ligand and receptor columns")
    if not 0 < proportion < 1:
        raise ValueError("resource replacement proportion must lie in (0, 1)")
    result = resource.loc[:, ["ligand", "receptor"]].astype(str).copy()
    if result.isna().any().any() or result.duplicated().any():
        raise ValueError("resource LR rows must be complete and unique")
    resource_genes = {
        part
        for value in result.loc[:, ["ligand", "receptor"]].to_numpy().ravel()
        for part in str(value).split("_")
        if part
    }
    gene_pool = tuple(
        gene
        for gene in dict.fromkeys(str(gene) for gene in genes if str(gene))
        if gene not in resource_genes
    )
    if len(gene_pool) < 2:
        raise ValueError("at least two candidate genes are required")
    pairs = list(result.itertuples(index=False, name=None))
    eligible = np.array(
        [index for index, pair in enumerate(pairs) if pair not in preserved_pairs],
        dtype=int,
    )
    replace_n = round(len(result) * proportion)
    if replace_n < 1 or replace_n > len(eligible):
        raise ValueError(
            "requested replacement count is incompatible with preserved resource rows"
        )
    rng = np.random.default_rng(seed)
    shuffled_genes = np.asarray(gene_pool, dtype=object)
    rng.shuffle(shuffled_genes)
    split = len(shuffled_genes) // 2
    ligand_pool = shuffled_genes[:split]
    receptor_pool = shuffled_genes[split:]
    if not len(ligand_pool) or not len(receptor_pool):
        raise ValueError("false LR source and target gene pools must be non-empty")
    replace_indices = np.sort(rng.choice(eligible, size=replace_n, replace=False))
    occupied = set(pairs)
    generated: list[tuple[str, str]] = []
    attempts = 0
    max_attempts = max(100_000, 100 * replace_n)
    while len(generated) < replace_n and attempts < max_attempts:
        pair = (str(rng.choice(ligand_pool)), str(rng.choice(receptor_pool)))
        attempts += 1
        if pair in occupied:
            continue
        occupied.add(pair)
        generated.append(pair)
    if len(generated) != replace_n:
        raise RuntimeError("could not generate enough unique false LR interactions")
    result.loc[replace_indices, ["ligand", "receptor"]] = generated
    if result.duplicated().any():
        raise RuntimeError("resource replacement introduced duplicate LR rows")
    original = resource.loc[:, ["ligand", "receptor"]].astype(str)
    changed = (result != original).any(axis=1)
    if int(changed.sum()) != replace_n:
        raise RuntimeError("resource replacement count differs from its design")
    return result.reset_index(drop=True), {
        "requested_proportion": float(proportion),
        "actual_proportion": float(replace_n / len(result)),
        "resource_rows": len(result),
        "replaced_rows": replace_n,
        "preserved_rows": len(result) - len(eligible),
        "candidate_genes": len(gene_pool),
        "candidate_ligand_genes": len(ligand_pool),
        "candidate_receptor_genes": len(receptor_pool),
        "generation_attempts": attempts,
    }


def select_top_edges(
    scores: pd.DataFrame,
    *,
    top_n: int = 250,
) -> pd.DataFrame:
    """Select exactly ``top_n`` edges per method with deterministic tie breaks."""

    required = {"method", "score", "score_direction", *EDGE_COLUMNS}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"score table is missing columns: {sorted(missing)}")
    if isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be an integer >= 1")
    records: list[pd.DataFrame] = []
    for method_name, method in scores.groupby("method", sort=True, observed=True):
        directions = method["score_direction"].astype(str).unique()
        if len(directions) != 1 or directions[0] not in {"higher", "lower"}:
            raise ValueError("each method must declare one valid score direction")
        selected = method.loc[:, [*EDGE_COLUMNS, "score"]].copy()
        selected["score"] = pd.to_numeric(selected["score"], errors="coerce")
        selected = selected.loc[np.isfinite(selected["score"])].copy()
        if directions[0] == "lower":
            selected["oriented_score"] = -selected["score"]
        else:
            selected["oriented_score"] = selected["score"]
        selected = selected.groupby(
            list(EDGE_COLUMNS), sort=True, observed=True, as_index=False
        ).agg(
            score=("score", "min" if directions[0] == "lower" else "max"),
            oriented_score=("oriented_score", "max"),
        )
        selected = selected.sort_values(
            ["oriented_score", *EDGE_COLUMNS],
            ascending=[False, True, True, True, True],
            kind="stable",
            ignore_index=True,
        ).head(top_n)
        selected.insert(0, "rank", np.arange(1, len(selected) + 1))
        selected.insert(0, "method", str(method_name))
        selected["score_direction"] = directions[0]
        records.append(selected)
    if not records:
        raise ValueError("score table produced no finite method rankings")
    return pd.concat(records, ignore_index=True)


def top_edge_overlap(
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
) -> dict[str, float | int]:
    """Compare one perturbed top list to its unperturbed operational reference."""

    for name, table in (("baseline", baseline), ("perturbed", perturbed)):
        missing = set(EDGE_COLUMNS).difference(table.columns)
        if missing:
            raise ValueError(f"{name} top edges are missing: {sorted(missing)}")
    baseline_set = set(
        baseline.loc[:, list(EDGE_COLUMNS)]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    perturbed_set = set(
        perturbed.loc[:, list(EDGE_COLUMNS)]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if not baseline_set or not perturbed_set:
        raise ValueError("top-edge overlap requires two non-empty rankings")
    intersection = len(baseline_set & perturbed_set)
    union = len(baseline_set | perturbed_set)
    return {
        "baseline_edges": len(baseline_set),
        "perturbed_edges": len(perturbed_set),
        "intersection_edges": intersection,
        "baseline_recovery": intersection / len(baseline_set),
        "paper_tpr": intersection / len(perturbed_set),
        "jaccard": intersection / union,
    }
