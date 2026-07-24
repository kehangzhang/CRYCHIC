#!/usr/bin/env python3
"""Run the paper-native DCST sweeps that remain below 100,000 cells.

The published ``simulation_A-B_C1-C2.Rmd`` has two independent one-dimensional
experiments: a subject-count sweep and a receiver-cell-count sweep. This module
does not silently turn them into a Cartesian grid.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact

SCHEMA = "crychic-dcst-under100k-v1"
DATASET_ID = "dcst_subject_count_n15_b500_seed123"
CONDITIONS = ("C1", "C2")
CELL_TYPES = ("A", "B")
GENES = ("L1", "L2", "R1", "R2")
INTERACTIONS = (("L1_R1", "L1", "R1"), ("L2_R2", "L2", "R2"))
N_SUBJECTS_PER_CONDITION = 15
EXPRESSION_THRESHOLD = 0.25
SEED = 123
SWEEP = "subject_count"

# Exact parameter slice from simulation_A-B_C1-C2.Rmd, lines 68-75.
EXPRESSION_PROBABILITIES: dict[tuple[str, str], tuple[float, ...]] = {
    ("C1", "A"): (0.30, 0.30, 0.05, 0.05),
    ("C1", "B"): (0.10, 0.00, 0.30, 0.30),
    ("C2", "A"): (0.25, 0.30, 0.05, 0.05),
    ("C2", "B"): (0.30, 0.00, 0.30, 0.30),
}
CELL_COUNTS: dict[tuple[str, str], int] = {
    (condition, cell_type): 500 for condition in CONDITIONS for cell_type in CELL_TYPES
}


def configure_simulation(
    *,
    sweep: str,
    subjects_per_condition: int,
    receiver_cells_in_condition_2: int,
    seed: int,
) -> None:
    """Configure one exact setting from the two published DCST sweeps."""

    global DATASET_ID
    global N_SUBJECTS_PER_CONDITION
    global EXPRESSION_PROBABILITIES
    global CELL_COUNTS
    global SEED
    global SWEEP
    if sweep not in {"subject_count", "receiver_cell_count"}:
        raise ValueError("sweep must be subject_count or receiver_cell_count")
    if subjects_per_condition < 1:
        raise ValueError("subjects_per_condition must be positive")
    if receiver_cells_in_condition_2 < 1:
        raise ValueError("receiver_cells_in_condition_2 must be positive")
    if sweep == "subject_count":
        if receiver_cells_in_condition_2 != 500:
            raise ValueError("subject_count sweep fixes all cell types at 500 cells")
        probabilities = {
            ("C1", "A"): (0.30, 0.30, 0.05, 0.05),
            ("C1", "B"): (0.10, 0.00, 0.30, 0.30),
            ("C2", "A"): (0.25, 0.30, 0.05, 0.05),
            ("C2", "B"): (0.30, 0.00, 0.30, 0.30),
        }
        counts = {
            (condition, cell_type): 500
            for condition in CONDITIONS
            for cell_type in CELL_TYPES
        }
    else:
        if subjects_per_condition != 20:
            raise ValueError("receiver_cell_count sweep fixes 20 pseudo-subjects/group")
        probabilities = {
            ("C1", "A"): (0.30, 0.30, 0.05, 0.05),
            ("C1", "B"): (0.10, 0.00, 0.30, 0.30),
            ("C2", "A"): (0.10, 0.30, 0.05, 0.05),
            ("C2", "B"): (0.30, 0.00, 0.30, 0.30),
        }
        counts = {
            ("C1", "A"): 500,
            ("C1", "B"): 500,
            ("C2", "A"): 500,
            ("C2", "B"): receiver_cells_in_condition_2,
        }
    total_cells = sum(counts.values()) * subjects_per_condition
    if total_cells >= 100_000:
        raise ValueError("DCST setting is not strictly below 100000 cells")
    DATASET_ID = (
        f"dcst_{sweep}_n{subjects_per_condition}_b{receiver_cells_in_condition_2}"
        f"_seed{seed}"
    )
    N_SUBJECTS_PER_CONDITION = subjects_per_condition
    EXPRESSION_PROBABILITIES = probabilities
    CELL_COUNTS = counts
    SEED = seed
    SWEEP = sweep


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_metadata(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _crychic_version() -> str:
    """Resolve an installed version while permitting direct source execution."""

    try:
        return importlib.metadata.version("CRYCHIC")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree-uninstalled"


def _tracked_asset(path: Path) -> dict[str, Any]:
    metadata = _git_metadata(path)
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    metadata.update(
        {
            "path": str(path),
            "files_including_git_metadata": len(files),
            "bytes_including_git_metadata": sum(item.stat().st_size for item in files),
        }
    )
    return metadata


def audit_assets(output: Path, vendor_root: Path, repo_root: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tensor = vendor_root / "CCC-Benchmark"
    staccato = vendor_root / "STACCato"
    dcst = vendor_root / "Differential_Cell_Signaling_Test"
    records = [
        {
            "asset": "tensor_cell2cell_original_simulation",
            "local_status": "partial",
            "vendor_path": str(tensor),
            "vendor_commit": _git_metadata(tensor)["commit"],
            "original_simulation_payload_present": False,
            "runnable_here": False,
            "reason": (
                "vendor contains PBMC/BALF timing code only; the published "
                "3-cell-type x 300-LR x 12-context tensor and Code Ocean analysis "
                "code are absent; Code Ocean capsule returned HTTP 403 during audit"
            ),
        },
        {
            "asset": "staccato_original_simulation",
            "local_status": "partial",
            "vendor_path": str(staccato),
            "vendor_commit": _git_metadata(staccato)["commit"],
            "original_simulation_payload_present": False,
            "runnable_here": False,
            "reason": (
                "repository contains ASD/SLE tutorial notebooks but no simulation "
                "payload; tutorial input is an external Dropbox asset; tensorregress, "
                "R.matlab, and rTensor are not installed"
            ),
        },
        {
            "asset": "dcst_sample_count_simulation",
            "local_status": "protocol_reconstructable",
            "vendor_path": str(dcst),
            "vendor_commit": _git_metadata(dcst)["commit"],
            "original_simulation_payload_present": True,
            "runnable_here": True,
            "reason": (
                "published binary-expression parameters and linkage/Fisher functions "
                "are local; the newest Rmd references missing helper files, so this "
                "track uses a documented protocol-compatible reimplementation"
            ),
        },
    ]
    table = pd.DataFrame.from_records(records)
    table.to_csv(output / "asset_audit.tsv", sep="\t", index=False)
    package_versions: dict[str, str | None] = {}
    for name in ("anndata", "numpy", "pandas", "scipy", "CRYCHIC"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    env_checks = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "crychic_source_repo": _tracked_asset(repo_root),
        "packages": package_versions,
        "r_dependency_audit": {
            "tensorregress": False,
            "R.matlab": False,
            "rTensor": False,
            "dominoSignal": False,
            "statmod": "1.5.2",
        },
    }
    _write_json(
        output / "asset_audit_manifest.json",
        {
            "schema_version": f"{SCHEMA}-asset-audit",
            "status": "complete",
            "records": len(records),
            "asset_audit_sha256": _sha256(output / "asset_audit.tsv"),
            "environment": env_checks,
        },
    )


def _exact_binary_pool(
    rng: np.random.Generator, probabilities: Sequence[float], n_cells: int
) -> np.ndarray:
    result = np.zeros((n_cells, len(GENES)), dtype=np.int32)
    for column, probability in enumerate(probabilities):
        n_active = int(round(probability * n_cells))
        if n_active:
            selected = rng.choice(n_cells, size=n_active, replace=False)
            result[selected, column] = 1
    return result


def _truth_table() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for interaction_id, ligand, receptor in INTERACTIONS:
        for sender in CELL_TYPES:
            for receiver in CELL_TYPES:
                direction = 0
                if interaction_id == "L1_R1" and sender == "A" and receiver == "B":
                    direction = -1
                elif interaction_id == "L1_R1" and sender == "B" and receiver == "B":
                    direction = 1
                records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "contrast": "C2_vs_C1",
                        "sender": sender,
                        "receiver": receiver,
                        "interaction_id": interaction_id,
                        "ligand": ligand,
                        "receptor": receptor,
                        "truth_label": int(direction != 0),
                        "truth_direction": direction,
                        "truth_definition": (
                            "published_threshold_crossing"
                            if direction
                            else "published_null_or_structural_absence"
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def prepare(output: Path, vendor_root: Path, repo_root: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"non-empty output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("input", "resource", "truth", "logs", "dcst", "crychic", "evaluation"):
        (output / name).mkdir()

    rng = np.random.default_rng(SEED)
    pools = {
        key: _exact_binary_pool(rng, probabilities, CELL_COUNTS[key])
        for key, probabilities in EXPRESSION_PROBABILITIES.items()
    }
    count_parts: list[np.ndarray] = []
    obs_parts: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        for sample_index in range(1, N_SUBJECTS_PER_CONDITION + 1):
            sample_id = f"{condition}_S{sample_index:02d}"
            for cell_type in CELL_TYPES:
                n_cells = CELL_COUNTS[(condition, cell_type)]
                source = pools[(condition, cell_type)]
                indices = rng.integers(0, len(source), size=n_cells)
                counts = source[indices].copy()
                count_parts.append(counts)
                obs_parts.append(
                    pd.DataFrame(
                        {
                            "sample_id": sample_id,
                            "subject_id": sample_id,
                            "family_id": sample_id,
                            "condition": condition,
                            "cell_type": cell_type,
                        },
                        index=[
                            f"{sample_id}:{cell_type}:{index:04d}"
                            for index in range(n_cells)
                        ],
                    )
                )
    counts = sparse.csr_matrix(np.vstack(count_parts), dtype=np.int32)
    library_size = np.asarray(counts.sum(axis=1)).ravel()
    scaling = np.zeros_like(library_size, dtype=float)
    positive = library_size > 0
    scaling[positive] = 1.0e4 / library_size[positive]
    normalized = sparse.csr_matrix(counts.astype(float).multiply(scaling[:, None]))
    normalized.data = np.log1p(normalized.data)
    adata = ad.AnnData(
        X=normalized,
        obs=pd.concat(obs_parts, axis=0),
        var=pd.DataFrame(index=pd.Index(GENES, name="gene_symbol")),
    )
    adata.layers["counts"] = counts
    adata.uns["simulation_contract"] = {
        "source": "Mitchell et al. 2026 DCST simulation_A-B_C1-C2.Rmd",
        "source_seed": SEED,
        "realization": 1,
        "published_sweep": SWEEP,
        "subjects_per_condition": N_SUBJECTS_PER_CONDITION,
        "cells_per_cell_type_per_subject": {
            f"{condition}:{cell_type}": CELL_COUNTS[(condition, cell_type)]
            for condition in CONDITIONS
            for cell_type in CELL_TYPES
        },
        "expression_threshold_strictly_greater_than": EXPRESSION_THRESHOLD,
        "x_semantics": "log1p_counts_per_10000",
        "counts_layer": "counts",
        "note": (
            "Independent Python RNG realization of the exact published parameter "
            "slice and exact-count pooled-cell bootstrap protocol; not byte-identical "
            "to the authors' R RNG stream."
        ),
    }
    input_path = output / "input" / f"{DATASET_ID}.h5ad"
    adata.write_h5ad(input_path, compression="gzip")

    resource = pd.DataFrame(
        {
            "harmonized_interaction_id": [item[0] for item in INTERACTIONS],
            "ligand": [item[1] for item in INTERACTIONS],
            "receptor": [item[2] for item in INTERACTIONS],
            "cellchat_source_interaction_id": [item[0] for item in INTERACTIONS],
            "cellchat_covered": [True, True],
            "liana_source_interaction_id": [item[0] for item in INTERACTIONS],
            "liana_covered": [True, True],
            "scseqcommdiff_source_interaction_id": [item[0] for item in INTERACTIONS],
            "scseqcommdiff_covered": [True, True],
        }
    )
    resource_path = output / "resource" / "harmonized_lr.tsv"
    resource.to_csv(resource_path, sep="\t", index=False, lineterminator="\n")
    _write_json(
        output / "resource" / "manifest.json",
        {
            "schema_version": "crychic-harmonized-lr-v1",
            "resource_id": "dcst_paper_two_lr_hcommon",
            "version": "2026-paper-parameter-slice",
            "species": "synthetic_human_namespace",
            "gene_namespace": "synthetic symbols",
            "license": "CC-BY-4.0 source protocol; derived local fixture",
            "citation": "Mitchell et al., Bioinformatics 2026, btag089",
            "payload": {
                "filename": resource_path.name,
                "rows": len(resource),
                "sha256": _sha256(resource_path),
            },
        },
    )
    prior = pd.DataFrame(
        {
            "driver": ["L1", "L2"],
            "target": ["R1", "R2"],
            "weight": [1.0, 1.0],
            "rank": [1, 1],
        }
    )
    prior_path = output / "resource" / "synthetic_target_prior.tsv"
    prior.to_csv(prior_path, sep="\t", index=False, lineterminator="\n")
    truth = _truth_table()
    truth_path = output / "truth" / "event_truth.tsv"
    truth.to_csv(truth_path, sep="\t", index=False, lineterminator="\n")

    vendor_simulation = (
        vendor_root
        / "Differential_Cell_Signaling_Test/scripts/simulation/simulation_A-B_C1-C2.Rmd"
    )
    embedded_functions = (
        vendor_root
        / "Differential_Cell_Signaling_Test/scripts/differential_signaling_pancvax"
        / "08_simulated_signaling_bootstrapping.Rmd"
    )
    input_manifest = {
        "schema_version": f"{SCHEMA}-input",
        "status": "complete",
        "dataset_id": DATASET_ID,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "cells": int(adata.n_obs),
        "samples": int(adata.obs["sample_id"].nunique()),
        "subjects": int(adata.obs["subject_id"].nunique()),
        "inferential_unit": "cell_bootstrap_pseudo_sample",
        "biological_subjects_represented": None,
        "unit_limitation": (
            "The paper simulation resamples pooled cells within each condition. "
            "The 30 unit IDs are algorithm-facing pseudo-subjects, not independent "
            "biological donors."
        ),
        "contexts": int(adata.obs["condition"].nunique()),
        "cell_types": int(adata.obs["cell_type"].nunique()),
        "subjects_by_context": {
            str(key): int(value)
            for key, value in adata.obs.groupby("condition", observed=True)[
                "subject_id"
            ]
            .nunique()
            .items()
        },
        "cells_by_context": {
            str(key): int(value)
            for key, value in adata.obs.groupby("condition", observed=True)
            .size()
            .items()
        },
        "parameters": {
            "seed": SEED,
            "published_sweep": SWEEP,
            "subjects_per_condition": N_SUBJECTS_PER_CONDITION,
            "cells_per_cell_type_per_subject": {
                f"{condition}:{cell_type}": CELL_COUNTS[(condition, cell_type)]
                for condition in CONDITIONS
                for cell_type in CELL_TYPES
            },
            "strict_expression_threshold": EXPRESSION_THRESHOLD,
            "expression_probabilities": {
                f"{condition}:{cell_type}": dict(zip(GENES, values, strict=True))
                for (condition, cell_type), values in EXPRESSION_PROBABILITIES.items()
            },
        },
        "source_protocol": {
            "vendor_commit": _git_metadata(
                vendor_root / "Differential_Cell_Signaling_Test"
            )["commit"],
            "parameter_file": str(vendor_simulation),
            "parameter_file_sha256": _sha256(vendor_simulation),
            "embedded_function_file": str(embedded_functions),
            "embedded_function_file_sha256": _sha256(embedded_functions),
            "parameter_lines": "68-75",
            "bootstrap_function_lines": "38-213 and 360-477",
        },
        "output": {
            "filename": input_path.name,
            "sha256": _sha256(input_path),
        },
        "resource_sha256": _sha256(resource_path),
        "target_prior_sha256": _sha256(prior_path),
        "truth_sha256": _sha256(truth_path),
        "crychic_code": _git_metadata(repo_root),
    }
    _write_json(output / "input" / "manifest.json", input_manifest)
    command_rows = [
        {
            "stage": "run_dcst",
            "command": shlex.join(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "run-dcst",
                    "--output",
                    str(output),
                ]
            ),
        },
        {
            "stage": "run_crychic",
            "command": shlex.join(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "run-crychic",
                    "--output",
                    str(output),
                    "--repo-root",
                    str(repo_root),
                ]
            ),
        },
        {
            "stage": "evaluate",
            "command": shlex.join(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "evaluate",
                    "--output",
                    str(output),
                    "--repo-root",
                    str(repo_root),
                ]
            ),
        },
    ]
    pd.DataFrame(command_rows).to_csv(output / "commands.tsv", sep="\t", index=False)


def _bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = numeric.notna()
    if not finite.any():
        return result
    selected = numeric.loc[finite]
    order = np.argsort(selected.to_numpy(), kind="stable")
    ordered = selected.to_numpy()[order]
    adjusted = np.minimum.accumulate(
        (ordered * len(ordered) / np.arange(1, len(ordered) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result.loc[finite] = restored
    return result


def run_dcst(output: Path) -> None:
    started = time.perf_counter()
    input_path = output / "input" / f"{DATASET_ID}.h5ad"
    source = ad.read_h5ad(input_path)
    try:
        counts = sparse.csr_matrix(source.layers["counts"])
        gene_index = {
            gene: index for index, gene in enumerate(source.var_names.astype(str))
        }
        sample_meta = source.obs.loc[:, ["sample_id", "condition"]].drop_duplicates()
        if sample_meta["sample_id"].duplicated().any():
            raise ValueError("sample_id maps to multiple conditions")
        activity: dict[tuple[str, str, str], bool] = {}
        for sample_id in sample_meta["sample_id"].astype(str):
            for cell_type in CELL_TYPES:
                mask = (
                    source.obs["sample_id"].astype(str).eq(sample_id)
                    & source.obs["cell_type"].astype(str).eq(cell_type)
                ).to_numpy()
                selected = counts[mask]
                prevalence = np.asarray((selected > 0).mean(axis=0)).ravel()
                for gene in GENES:
                    activity[(sample_id, cell_type, gene)] = bool(
                        prevalence[gene_index[gene]] > EXPRESSION_THRESHOLD
                    )
        truth = pd.read_csv(output / "truth" / "event_truth.tsv", sep="\t")
        presence_rows: list[dict[str, Any]] = []
        condition_map = sample_meta.set_index("sample_id")["condition"].astype(str)
        for sample_id in sample_meta["sample_id"].astype(str):
            for row in truth.itertuples(index=False):
                present = (
                    activity[(sample_id, str(row.sender), str(row.ligand))]
                    and activity[(sample_id, str(row.receiver), str(row.receptor))]
                )
                presence_rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": sample_id,
                        "condition": condition_map.loc[sample_id],
                        "sender": row.sender,
                        "receiver": row.receiver,
                        "interaction_id": row.interaction_id,
                        "ligand": row.ligand,
                        "receptor": row.receptor,
                        "linkage_present": int(present),
                    }
                )
        presence = pd.DataFrame.from_records(presence_rows)
        keys = ["sender", "receiver", "interaction_id", "ligand", "receptor"]
        results: list[dict[str, Any]] = []
        for edge, frame in presence.groupby(keys, observed=True, sort=True):
            c1 = frame.loc[frame["condition"].eq("C1"), "linkage_present"]
            c2 = frame.loc[frame["condition"].eq("C2"), "linkage_present"]
            contingency = np.asarray(
                [
                    [int(c2.sum()), int(len(c2) - c2.sum())],
                    [int(c1.sum()), int(len(c1) - c1.sum())],
                ]
            )
            p_value = float(fisher_exact(contingency, alternative="two-sided").pvalue)
            results.append(
                {
                    **dict(zip(keys, edge, strict=True)),
                    "effect": float(c2.mean() - c1.mean()),
                    "p_value": p_value,
                    "C1_present": int(c1.sum()),
                    "C1_total": len(c1),
                    "C2_present": int(c2.sum()),
                    "C2_total": len(c2),
                    "status": "observed",
                    "reason_code": "",
                    "effect_semantics": "C2_minus_C1_sample_linkage_prevalence",
                }
            )
        event = pd.DataFrame.from_records(results)
        event["q_value_bh"] = _bh(event["p_value"])
        presence.to_csv(
            output / "dcst" / "sample_linkage_presence.tsv.gz", sep="\t", index=False
        )
        event.to_csv(output / "dcst" / "event_results.tsv", sep="\t", index=False)
        _write_json(
            output / "dcst" / "manifest.json",
            {
                "schema_version": f"{SCHEMA}-method-run",
                "status": "complete",
                "dataset_id": DATASET_ID,
                "method": {
                    "id": "dcst_protocol_reimplementation",
                    "label": "DCST published simulation protocol",
                    "software_identity": "protocol-compatible reimplementation",
                    "formal_exact_package_claim": False,
                },
                "input_sha256": _sha256(input_path),
                "parameters": {
                    "linkage_call": "ligand_and_receptor_prevalence_strictly_gt_0.25",
                    "statistical_test": "two_sided_fisher_exact",
                    "multiple_testing": "BH across 8 frozen events",
                    "inferential_unit": "bootstrap sample",
                },
                "elapsed_seconds_internal": time.perf_counter() - started,
                "outputs": {
                    "sample_linkage_presence.tsv.gz": {
                        "rows": len(presence),
                        "sha256": _sha256(
                            output / "dcst" / "sample_linkage_presence.tsv.gz"
                        ),
                    },
                    "event_results.tsv": {
                        "rows": len(event),
                        "sha256": _sha256(output / "dcst" / "event_results.tsv"),
                    },
                },
            },
        )
    finally:
        source.file.close() if source.isbacked else None


def _crychic_objects(resource_sha: str, prior_sha: str) -> tuple[Any, Any]:
    from crychic.resources import (
        GeneNamespace,
        Interaction,
        MappingReport,
        ResourceBundle,
        Species,
        TargetPrior,
    )

    interactions = tuple(
        Interaction(
            interaction_id=interaction_id,
            source_interaction_id=interaction_id,
            ligand_name=ligand,
            receptor_name=receptor,
            ligand_subunits=(ligand,),
            receptor_subunits=(receptor,),
            ligand_is_complex=False,
            receptor_is_complex=False,
            direction="Ligand-Receptor",
            source="dcst_published_simulation",
            version="2026",
            species=Species.HUMAN,
            gene_namespace=GeneNamespace.HGNC_SYMBOL,
            evidence=("published_synthetic_parameter",),
        )
        for interaction_id, ligand, receptor in INTERACTIONS
    )
    bundle = ResourceBundle(
        resource_id="dcst_paper_two_lr_hcommon",
        version="2026-paper-parameter-slice",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=interactions,
        mapping_report=MappingReport(source_rows=2, loaded_rows=2, mapped_entities=4),
        manifest_digest=resource_sha,
        source_files=("harmonized_lr.tsv",),
        license="CC-BY-4.0 source protocol; derived local fixture",
        citation="Mitchell et al., Bioinformatics 2026, btag089",
    )
    prior = TargetPrior(
        resource_id="dcst_synthetic_target_prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=("R1", "R2"),
        driver_ids=("L1", "L2"),
        indptr=(0, 1, 2),
        target_indices=(0, 1),
        weights=(1.0, 1.0),
        ranks=(1, 1),
        direction=1,
        evidence="synthetic_identity_mapping_for_compatibility_only",
        mapping_report=MappingReport(source_rows=2, loaded_rows=2, mapped_entities=4),
        manifest_digest=prior_sha,
    )
    return bundle, prior


def run_crychic(output: Path, repo_root: Path) -> None:
    started = time.perf_counter()
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))
    from benchmarks.adapters.crychic.readback import convert_result_to_long
    from benchmarks.metrics.multicondition import (
        external_long_to_score_table,
        unpaired_edge_effects,
    )
    from crychic import Crychic, CrychicConfig
    from crychic.sender import SenderEvidenceParameters

    input_path = output / "input" / f"{DATASET_ID}.h5ad"
    resource_path = output / "resource" / "harmonized_lr.tsv"
    prior_path = output / "resource" / "synthetic_target_prior.tsv"
    resource_sha = _sha256(resource_path)
    prior_sha = _sha256(prior_path)
    bundle, prior = _crychic_objects(resource_sha, prior_sha)
    source = ad.read_h5ad(input_path)
    config = CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        sample_key="sample_id",
        subject_key="subject_id",
        cell_type_key="cell_type",
        design="~ condition",
        communication_modes=("state",),
        random_seed=SEED,
    )
    model = Crychic(config, resource_bundle=bundle, target_prior=prior)
    plan = model.dry_run(
        source,
        min_cells=30,
        min_samples_per_context=4,
        min_subjects_per_context=4,
    )
    if not plan.can_fit:
        raise RuntimeError(f"CRYCHIC dry run failed: {plan.warnings}")
    code = _git_metadata(repo_root)
    result = model.fit(
        source,
        output_dir=output / "crychic" / "result",
        input_digest=_sha256(input_path),
        git_commit=str(code["commit"]),
        git_dirty=bool(code["dirty"]),
        package_version=_crychic_version(),
        sender_parameters=SenderEvidenceParameters(min_subjects=4),
        min_cells=30,
        min_samples_per_context=4,
        min_subjects_per_context=4,
        min_pooled_availability=0.0,
        max_interactions=None,
        subject_fixed_effects=False,
        lambda1=0.0,
        lambda2=0.0,
        cosine_threshold=0.95,
        prior_quality=1.0,
    )
    table, views = convert_result_to_long(
        result,
        source,
        bundle,
        dataset_id=DATASET_ID,
        resource_mode="H-common",
        communication_mode="state",
    )
    long_path = output / "crychic" / "interactions_long.parquet"
    table.to_parquet(long_path, index=False)
    candidate = "global:'C2'"
    selected_ids = [
        str(view["run_id"])
        for view in views
        if candidate in list(view.get("contrast_candidates", []))
    ]
    if len(selected_ids) != 1:
        raise RuntimeError(f"expected one C2 score view, observed={selected_ids}")
    selected = table.loc[table["run_id"].astype(str).eq(selected_ids[0])].copy()
    mapped = external_long_to_score_table(
        selected,
        context_key="condition",
        contrast="C2_vs_C1",
        dataset=DATASET_ID,
    )
    effects = unpaired_edge_effects(
        mapped,
        reference="C1",
        target="C2",
        min_subjects=4,
        contrast="C2_vs_C1",
        validated=True,
    )
    columns = [
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "effect",
        "status",
        "reason_code",
    ]
    event = effects.loc[:, columns].copy()
    event["p_value"] = np.nan
    event["q_value_bh"] = np.nan
    event["effect_semantics"] = "C2_minus_C1_independent_subject_rank_strength"
    event.to_csv(output / "crychic" / "event_results.tsv", sep="\t", index=False)
    _write_json(
        output / "crychic" / "manifest.json",
        {
            "schema_version": f"{SCHEMA}-method-run",
            "status": "complete",
            "dataset_id": DATASET_ID,
            "method": {
                "id": "crychic_generic_synthetic_prior",
                "label": "CRYCHIC generic multigroup baseline",
                "version": _crychic_version(),
                "source_module": str(Path(sys.modules["crychic"].__file__).resolve()),
                "rc9_claim": False,
                "formal_p_or_q_emitted": False,
            },
            "code": code,
            "input_sha256": _sha256(input_path),
            "resource_sha256": resource_sha,
            "target_prior_sha256": prior_sha,
            "synthetic_prior_limitation": (
                "L1->R1 and L2->R2 identity mappings are compatibility priors, not "
                "a biological NicheNet prior; this track evaluates event recovery only."
            ),
            "parameters": {
                "fit": "legacy exploratory generic baseline",
                "counts_layer": "counts",
                "communication_mode": "state",
                "min_cells": 30,
                "min_samples_per_context": 4,
                "min_subjects_per_context": 4,
                "max_interactions": None,
                "lambda1": 0.0,
                "lambda2": 0.0,
            },
            "source_result": {
                "run_id": result.manifest["run_id"],
                "result_schema_version": result.manifest["result_schema_version"],
                "selected_score_view": selected_ids[0],
                "score_views": views,
            },
            "elapsed_seconds_internal": time.perf_counter() - started,
            "outputs": {
                "interactions_long.parquet": {
                    "rows": len(table),
                    "sha256": _sha256(long_path),
                },
                "event_results.tsv": {
                    "rows": len(event),
                    "sha256": _sha256(output / "crychic" / "event_results.tsv"),
                },
            },
        },
    )


def evaluate(output: Path, repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    from benchmarks.comprehensive.evaluate_g1 import (
        average_precision,
        prevalence_adjusted_average_precision,
        tie_aware_auroc,
    )

    truth = pd.read_csv(output / "truth" / "event_truth.tsv", sep="\t")
    keys = ["sender", "receiver", "interaction_id", "ligand", "receptor"]
    frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    method_files = {
        "crychic_generic_synthetic_prior": output / "crychic" / "event_results.tsv",
        "dcst_protocol_reimplementation": output / "dcst" / "event_results.tsv",
    }
    for method, path in method_files.items():
        result = pd.read_csv(path, sep="\t")
        merged = truth.merge(result, on=keys, how="left", validate="one_to_one")
        merged["method"] = method
        observed = (
            merged["status"].isin(("observed", "exploratory"))
            & merged["effect"].notna()
        )
        score = pd.to_numeric(merged.loc[observed, "effect"], errors="raise").abs()
        labels = merged.loc[observed, "truth_label"].to_numpy(dtype=int)
        positives = merged["truth_label"].eq(1)
        eligible = observed.mean() >= 0.80 and observed.loc[positives].all()
        auroc = tie_aware_auroc(labels, score) if eligible else math.nan
        auprc = average_precision(labels, score) if eligible else math.nan
        adjusted_ap = (
            prevalence_adjusted_average_precision(labels, score, target_prevalence=0.10)
            if eligible
            else math.nan
        )
        direction = (
            float(
                np.mean(
                    np.sign(merged.loc[positives, "effect"].to_numpy(dtype=float))
                    == merged.loc[positives, "truth_direction"].to_numpy(dtype=int)
                )
            )
            if eligible
            else math.nan
        )
        p_value = pd.to_numeric(merged["p_value"], errors="coerce")
        null = merged["truth_label"].eq(0)
        positive = merged["truth_label"].eq(1)
        discoveries = p_value.lt(0.05)
        native_significance_available = bool(p_value.notna().all())
        metrics.append(
            {
                "method": method,
                "coverage": float(observed.mean()),
                "n_events": len(merged),
                "n_truth_positive": int(positives.sum()),
                "metric_status": "observed" if eligible else "NE",
                "auroc": auroc,
                "auprc": auprc,
                "prevalence_adjusted_ap_10pct": adjusted_ap,
                "direction_accuracy": direction,
                "native_p_rows": int(p_value.notna().sum()),
                "sensitivity_at_005": (
                    float(discoveries.loc[positive].mean())
                    if native_significance_available
                    else math.nan
                ),
                "specificity_at_005": (
                    float((~discoveries.loc[null]).mean())
                    if native_significance_available
                    else math.nan
                ),
                "native_type1_error_005": (
                    float(discoveries.loc[null].mean())
                    if native_significance_available
                    else math.nan
                ),
                "native_empirical_fdr_005": (
                    float((discoveries & null).sum() / discoveries.sum())
                    if discoveries.any() and native_significance_available
                    else 0.0
                    if native_significance_available
                    else math.nan
                ),
            }
        )
        frames.append(merged)
    scored = pd.concat(frames, ignore_index=True)
    metric_table = pd.DataFrame.from_records(metrics)
    metric_table["primary_rank"] = metric_table["prevalence_adjusted_ap_10pct"].rank(
        ascending=False, method="average"
    )
    scored.to_csv(output / "evaluation" / "event_scored.tsv.gz", sep="\t", index=False)
    metric_table.to_csv(
        output / "evaluation" / "method_metrics.tsv", sep="\t", index=False
    )
    lines = [
        "# DCST known-truth parameter-slice benchmark",
        "",
        (
            "This is one full 30,000-cell cohort realization at the published "
            "n=15 pseudo-samples/group and 500 cells/cell type setting. It is "
            "not the authors' complete 25-replicate Monte Carlo sweep."
        ),
        (
            "The 30 inferential units are pooled-cell bootstrap pseudo-samples, "
            "not independent biological donors; this track cannot establish "
            "biological-subject calibration."
        ),
        "",
        "| Rank | Method | Adjusted AP | AUPRC | AUROC | Direction | Coverage |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_table.sort_values("primary_rank").itertuples(index=False):
        lines.append(
            f"| {row.primary_rank:g} | {row.method} | "
            f"{row.prevalence_adjusted_ap_10pct:.4f} | {row.auprc:.4f} | "
            f"{row.auroc:.4f} | {row.direction_accuracy:.4f} | {row.coverage:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "DCST uses the published strict 25% linkage threshold and two-sided "
                "Fisher test, implemented locally from the paper's source functions."
            ),
            (
                "CRYCHIC is the generic exploratory baseline with a synthetic identity "
                "target prior. It is not RC9 and emits no formal p/q values."
            ),
            "",
        ]
    )
    (output / "evaluation" / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_elapsed(value: str) -> float | None:
    parts = value.strip().split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    except ValueError:
        return None
    return None


def _time_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    elapsed = re.search(r"Elapsed \(wall clock\) time .*?\):\s*([0-9:.]+)", text)
    exit_code = re.search(r"Exit status:\s*(\d+)", text)
    return {
        "status": "complete" if exit_code and exit_code.group(1) == "0" else "failed",
        "wall_time_seconds": _parse_elapsed(elapsed.group(1)) if elapsed else None,
        "peak_rss_kib": int(rss.group(1)) if rss else None,
        "exit_status": int(exit_code.group(1)) if exit_code else None,
        "time_file": path.name,
        "time_file_sha256": _sha256(path),
    }


def finalize_under100k(output: Path, repo_root: Path) -> None:
    """Bind one configured paper-sweep realization without legacy track state."""

    required = (
        output / "input" / "manifest.json",
        output / "truth" / "event_truth.tsv",
        output / "dcst" / "manifest.json",
        output / "crychic" / "manifest.json",
        output / "evaluation" / "method_metrics.tsv",
        output / "evaluation" / "event_scored.tsv.gz",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"cannot finalize incomplete DCST realization: {missing}"
        )
    input_manifest = json.loads((output / "input" / "manifest.json").read_text())
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != output / "manifest.json":
            inventory.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "protocol_status": (
                "paper_native_independent_sweep_protocol_reimplementation"
            ),
            "dataset_id": DATASET_ID,
            "published_sweep": SWEEP,
            "dimensions": {
                "cells": int(input_manifest["cells"]),
                "pseudo_subjects": int(input_manifest["subjects"]),
                "contexts": int(input_manifest["contexts"]),
                "cell_types": int(input_manifest["cell_types"]),
                "genes": int(input_manifest["shape"][1]),
                "frozen_events": len(_truth_table()),
            },
            "parameters": input_manifest["parameters"],
            "source_repository": _git_metadata(repo_root),
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
            },
            "method_manifests": {
                "dcst": _sha256(output / "dcst" / "manifest.json"),
                "crychic": _sha256(output / "crychic" / "manifest.json"),
            },
            "evaluation": {
                "method_metrics_sha256": _sha256(
                    output / "evaluation" / "method_metrics.tsv"
                ),
                "event_scores_sha256": _sha256(
                    output / "evaluation" / "event_scored.tsv.gz"
                ),
            },
            "artifact_inventory": inventory,
        },
    )


def run_full(
    output: Path,
    vendor_root: Path,
    repo_root: Path,
    *,
    sweep: str,
    subjects_per_condition: int,
    receiver_cells_in_condition_2: int,
    seed: int,
) -> None:
    """Execute one complete configured setting with checksum-bound outputs."""

    configure_simulation(
        sweep=sweep,
        subjects_per_condition=subjects_per_condition,
        receiver_cells_in_condition_2=receiver_cells_in_condition_2,
        seed=seed,
    )
    prepare(output, vendor_root, repo_root)
    run_dcst(output)
    run_crychic(output, repo_root)
    evaluate(output, repo_root)
    finalize_under100k(output, repo_root)
    verify(output)


def verify(output: Path) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    inventory = manifest.get("artifact_inventory", [])
    if not isinstance(inventory, list):
        raise ValueError("artifact_inventory must be a list")
    for raw in inventory:
        if not isinstance(raw, Mapping):
            failures.append("non-object artifact inventory entry")
            continue
        relative = str(raw.get("path", ""))
        path = output / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif _sha256(path) != raw.get("sha256"):
            failures.append(f"checksum:{relative}")
    runner = manifest.get("runner")
    if not isinstance(runner, Mapping):
        failures.append("runner:missing")
    else:
        runner_path = Path(str(runner.get("path", "")))
        if not runner_path.is_file() or _sha256(runner_path) != runner.get("sha256"):
            failures.append("runner:checksum")
    if failures:
        raise RuntimeError("manifest verification failed: " + ",".join(failures))
    print(
        json.dumps(
            {
                "status": "verified",
                "manifest_sha256": _sha256(manifest_path),
                "artifacts_verified": len(inventory),
                "runner_verified": True,
            },
            sort_keys=True,
        )
    )


def finalize_root(output: Path) -> None:
    asset_manifest = output / "asset_audit_manifest.json"
    runner = Path(__file__).resolve()
    campaign_runner = runner.with_name("run_dcst_25rep.py")
    readme = output / "README.md"
    required = (
        asset_manifest,
        runner,
        output / "asset_audit.tsv",
        readme,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"simulation-track root inputs are missing: {missing}")
    track_manifests = sorted(
        child / "manifest.json"
        for child in output.iterdir()
        if child.is_dir() and (child / "manifest.json").is_file()
    )
    if not track_manifests:
        raise FileNotFoundError(
            f"no completed simulation-track manifests under: {output}"
        )
    tracks = []
    for track_manifest in track_manifests:
        track_payload = json.loads(track_manifest.read_text(encoding="utf-8"))
        tracks.append(
            {
                "id": str(track_payload.get("dataset_id", track_manifest.parent.name)),
                "status": str(track_payload["status"]),
                "path": track_manifest.parent.name,
                "manifest_sha256": _sha256(track_manifest),
            }
        )
    runners = [runner]
    if campaign_runner.is_file():
        runners.append(campaign_runner)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "crychic-expanded-simulation-tracks-v1",
            "status": (
                "complete"
                if all(track["status"] == "complete" for track in tracks)
                else "partial"
            ),
            "tracks_completed": sum(track["status"] == "complete" for track in tracks),
            "tracks": tracks,
            "asset_audit": {
                "table_sha256": _sha256(output / "asset_audit.tsv"),
                "manifest_sha256": _sha256(asset_manifest),
            },
            "runner": {
                "path": str(runner),
                "sha256": _sha256(runner),
            },
            "runners": [
                {"path": str(path), "sha256": _sha256(path)} for path in runners
            ],
            "readme_sha256": _sha256(readme),
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "audit",
        "prepare",
        "run-dcst",
        "run-crychic",
        "evaluate",
        "verify",
        "finalize-root",
    ):
        item = sub.add_parser(command)
        item.add_argument("--output", required=True, type=Path)
        if command in {"audit", "prepare"}:
            item.add_argument("--vendor-root", required=True, type=Path)
        if command in {"audit", "prepare", "run-crychic", "evaluate"}:
            item.add_argument("--repo-root", required=True, type=Path)
    full = sub.add_parser("run-full")
    full.add_argument("--output", required=True, type=Path)
    full.add_argument("--vendor-root", required=True, type=Path)
    full.add_argument("--repo-root", required=True, type=Path)
    full.add_argument(
        "--sweep",
        required=True,
        choices=("subject_count", "receiver_cell_count"),
    )
    full.add_argument("--subjects-per-condition", required=True, type=int)
    full.add_argument("--receiver-cells-in-condition-2", required=True, type=int)
    full.add_argument("--seed", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if args.command == "audit":
        audit_assets(output, args.vendor_root.resolve(), args.repo_root.resolve())
    elif args.command == "prepare":
        prepare(output, args.vendor_root.resolve(), args.repo_root.resolve())
    elif args.command == "run-dcst":
        run_dcst(output)
    elif args.command == "run-crychic":
        run_crychic(output, args.repo_root.resolve())
    elif args.command == "evaluate":
        evaluate(output, args.repo_root.resolve())
    elif args.command == "verify":
        verify(output)
    elif args.command == "finalize-root":
        finalize_root(output)
    elif args.command == "run-full":
        run_full(
            output,
            args.vendor_root.resolve(),
            args.repo_root.resolve(),
            sweep=args.sweep,
            subjects_per_condition=args.subjects_per_condition,
            receiver_cells_in_condition_2=args.receiver_cells_in_condition_2,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
