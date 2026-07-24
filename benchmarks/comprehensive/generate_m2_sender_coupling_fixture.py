"""Generate paired cell-level fixtures for M2 sender-specific coupling."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.generate_three_group_fixture import (
    _freeze_simulation_resource,
)
from benchmarks.simulation.generate import GENES, TARGETS

SCHEMA_VERSION = "crychic-m2-sender-coupling-fixture-v1"
CONDITIONS = ("ctrl", "stim")
KNOWN_LIGAND = "CXCL10"
KNOWN_RECEPTOR = "CXCR3"
KNOWN_RECEIVER = "Receiver"
CANDIDATE_ROLES: Mapping[str, str] = {
    "TrueSender": "coupled_active",
    "BatchDecoy": "batch_confounded",
    "CompositionDecoy": "composition_confounded",
    "IndependentDecoy": "independent_null",
}
CELL_TYPES = (*CANDIDATE_ROLES, KNOWN_RECEIVER)
_GENE_INDEX = {gene: index for index, gene in enumerate(GENES)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _dataset_seed(root_seed: int) -> int:
    payload = f"crychic:m2-sender-coupling:v1:{root_seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _standardized(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    scale = float(np.std(centered, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("synthetic latent variable has zero variance")
    return centered / scale


def _mean_profile(cell_type: str) -> np.ndarray:
    rates = np.full(len(GENES), 0.03, dtype=float)
    for gene, value in (
        ("ISG15", 0.40),
        ("IFIT1", 0.40),
        ("MX1", 0.40),
        ("OAS1", 0.40),
        ("STAT1", 0.50),
        ("B2M", 0.60),
        ("GAPDH", 2.50),
        ("RPLP0", 2.20),
        ("MALAT1", 3.00),
        ("ACTB", 2.00),
    ):
        rates[_GENE_INDEX[gene]] = value
    if cell_type in CANDIDATE_ROLES:
        rates[_GENE_INDEX[KNOWN_LIGAND]] = 3.00
        rates[_GENE_INDEX[KNOWN_RECEPTOR]] = 0.02
    elif cell_type == KNOWN_RECEIVER:
        rates[_GENE_INDEX[KNOWN_LIGAND]] = 0.02
        rates[_GENE_INDEX[KNOWN_RECEPTOR]] = 0.90
    else:  # pragma: no cover - guarded by the fixture cell-type manifest
        raise ValueError(f"unknown M2 cell type: {cell_type}")
    return rates


def _subject_design(n_subjects: int, rng: np.random.Generator) -> pd.DataFrame:
    true_signal = _standardized(rng.normal(size=n_subjects))
    composition = _standardized(rng.normal(size=n_subjects))
    independent = _standardized(rng.normal(size=n_subjects))
    batch = np.resize(np.asarray([-1.0, 1.0]), n_subjects)
    rng.shuffle(batch)
    batch = _standardized(batch)
    true_proportion = _standardized(rng.normal(size=n_subjects))
    noise = rng.normal(scale=0.06, size=(n_subjects, 5))
    return pd.DataFrame(
        {
            "subject_id": [f"S{index + 1:02d}" for index in range(n_subjects)],
            "true_signal": true_signal,
            "batch_contrast": batch,
            "composition_signal": composition,
            "independent_signal": independent,
            "true_proportion_signal": true_proportion,
            "true_sender_log_effect": 0.85 + 0.80 * true_signal + noise[:, 0],
            "batch_decoy_log_effect": 0.85 + 1.20 * batch + noise[:, 1],
            "composition_decoy_log_effect": (0.85 + 1.20 * composition + noise[:, 2]),
            "independent_decoy_log_effect": (0.85 + 0.80 * independent + noise[:, 3]),
            "receiver_program_log_effect": (
                0.85
                + 0.65 * true_signal
                + 1.00 * batch
                + 1.00 * composition
                + noise[:, 4]
            ),
        }
    )


def _simulate_dataset(
    *, root_seed: int, n_subjects: int, mean_cells_per_sample: int
) -> tuple[ad.AnnData, pd.DataFrame]:
    seed = _dataset_seed(root_seed)
    rng = np.random.default_rng(seed)
    design = _subject_design(n_subjects, rng)
    base_proportions = np.asarray([0.17, 0.17, 0.17, 0.17, 0.32])
    base_logits = np.log(base_proportions)
    counts_parts: list[np.ndarray] = []
    obs_parts: list[pd.DataFrame] = []

    for subject in design.itertuples(index=False):
        subject_scale = float(rng.lognormal(mean=0.0, sigma=0.12))
        batch_positive = float(subject.batch_contrast) > 0.0
        for condition in CONDITIONS:
            stimulated = condition == "stim"
            logits = base_logits.copy()
            if stimulated:
                logits[0] += 0.15 * float(subject.true_proportion_signal)
                logits[2] += 1.20 * float(subject.composition_signal)
            probabilities = np.exp(logits - float(np.max(logits)))
            probabilities /= probabilities.sum()
            realized = rng.dirichlet(probabilities * 220.0)
            total_cells = max(100, int(rng.poisson(mean_cells_per_sample)))
            minimum_per_type = 12
            reserved = minimum_per_type * len(CELL_TYPES)
            sizes = minimum_per_type + rng.multinomial(total_cells - reserved, realized)
            sample_id = f"{subject.subject_id}:{condition}"
            assay_batch = (
                "B"
                if (stimulated and batch_positive)
                or (not stimulated and not batch_positive)
                else "A"
            )

            sender_effects = {
                "TrueSender": float(subject.true_sender_log_effect),
                "BatchDecoy": float(subject.batch_decoy_log_effect),
                "CompositionDecoy": float(subject.composition_decoy_log_effect),
                "IndependentDecoy": float(subject.independent_decoy_log_effect),
            }
            for cell_type, n_cells in zip(CELL_TYPES, sizes, strict=True):
                if n_cells == 0:
                    continue
                mean = _mean_profile(cell_type) * subject_scale
                if stimulated and cell_type in CANDIDATE_ROLES:
                    mean[_GENE_INDEX[KNOWN_LIGAND]] *= np.exp(
                        np.clip(sender_effects[cell_type], -2.0, 3.5)
                    )
                if stimulated and cell_type == KNOWN_RECEIVER:
                    multiplier = np.exp(
                        np.clip(subject.receiver_program_log_effect, -2.0, 3.5)
                    )
                    for gene in TARGETS:
                        mean[_GENE_INDEX[gene]] *= multiplier
                cell_depth = rng.lognormal(mean=0.0, sigma=0.30, size=n_cells)
                expected = cell_depth[:, None] * mean[None, :]
                dispersion = 0.08
                latent = rng.gamma(
                    shape=1.0 / dispersion,
                    scale=expected * dispersion,
                )
                part = np.asarray(rng.poisson(latent), dtype=np.int32)
                counts_parts.append(part)
                obs_parts.append(
                    pd.DataFrame(
                        {
                            "sample_id": sample_id,
                            "subject_id": str(subject.subject_id),
                            "condition": condition,
                            "cell_type": cell_type,
                            "assay_batch": assay_batch,
                            "composition_design_index": (
                                float(subject.composition_signal) if stimulated else 0.0
                            ),
                        },
                        index=[
                            f"{sample_id}:{cell_type}:{index:04d}"
                            for index in range(n_cells)
                        ],
                    )
                )

    counts = sparse.csr_matrix(np.vstack(counts_parts))
    obs = pd.concat(obs_parts, axis=0)
    var = pd.DataFrame(index=pd.Index(GENES, name="gene_symbol"))
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    scale = np.divide(
        1.0e4,
        library,
        out=np.zeros_like(library),
        where=library > 0,
    )
    normalized = sparse.csr_matrix(counts.astype(np.float64).multiply(scale[:, None]))
    normalized.data = np.log1p(normalized.data)
    adata = ad.AnnData(X=normalized, obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.obs["family_id"] = adata.obs["subject_id"].astype(str)
    adata.uns["expression_contract"] = {
        "design": "paired_subject",
        "x_semantics": "log1p_counts_per_10000",
        "counts_layer": "counts",
    }
    adata.uns["simulation_truth"] = {
        "fixture": SCHEMA_VERSION,
        "root_seed": int(root_seed),
        "dataset_seed": int(seed),
        "true_sender": "TrueSender",
        "candidate_roles": dict(CANDIDATE_ROLES),
        "target_genes": list(TARGETS),
        "cell_level_values_are_not_independent_replicates": True,
    }
    return adata, design


def _dataset_spec(
    *, dataset_id: str, input_path: Path, input_sha256: str, seed: int
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "input": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "output_name": dataset_id,
        "input_mode": "counts",
        "lr_resource": "synthetic_hcommon",
        "target_prior": None,
        "benchmark_scope": "paired_subject_m2_sender_specificity_stress",
        "config": {
            "context_keys": ["condition"],
            "counts_layer": "counts",
            "sample_key": "sample_id",
            "subject_key": "subject_id",
            "cell_type_key": "cell_type",
            "species": "human",
            "gene_namespace": "hgnc_symbol",
            "design": "~ condition",
            "communication_modes": ["state"],
            "random_seed": int(seed),
        },
        "workflow": {
            "min_cells": 10,
            "min_samples_per_context": 4,
            "min_subjects_per_context": 4,
            "min_pooled_availability": 0.0,
            "max_interactions": None,
            "subject_fixed_effects": True,
            "lambda1": 0.0,
            "lambda2": 0.0,
            "cosine_threshold": 0.95,
            "prior_quality": 1.0,
        },
    }


def _generate_task(
    *,
    staged: str,
    output_dir: str,
    replicate: int,
    root_seed: int,
    n_subjects: int,
    mean_cells_per_sample: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    staged_path = Path(staged)
    final_output = Path(output_dir)
    dataset_id = f"m2_sender_r{replicate:03d}"
    adata, design = _simulate_dataset(
        root_seed=root_seed,
        n_subjects=n_subjects,
        mean_cells_per_sample=mean_cells_per_sample,
    )
    dataset_dir = staged_path / "inputs" / dataset_id
    dataset_dir.mkdir()
    input_path = dataset_dir / f"{dataset_id}.h5ad"
    adata.write_h5ad(input_path, compression="gzip")
    design_path = dataset_dir / "generation_latents.tsv"
    design.to_csv(design_path, sep="\t", index=False, lineterminator="\n")
    input_sha = sha256_file(input_path)
    _write_json(
        dataset_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "root_seed": int(root_seed),
            "dataset_seed": _dataset_seed(root_seed),
            "inferential_unit": "subject_id",
            "design": "paired_subject",
            "conditions": list(CONDITIONS),
            "output": {
                "filename": input_path.name,
                "sha256": input_sha,
                "shape": [int(adata.n_obs), int(adata.n_vars)],
            },
            "generation_latents": {
                "filename": design_path.name,
                "sha256": sha256_file(design_path),
                "benchmark_method_access_allowed": False,
            },
        },
    )
    record = {
        "dataset_id": dataset_id,
        "root_seed": int(root_seed),
        "dataset_seed": _dataset_seed(root_seed),
        "h5ad": str(input_path.relative_to(staged_path)),
        "sha256": input_sha,
        "cells": int(adata.n_obs),
    }
    spec = _dataset_spec(
        dataset_id=dataset_id,
        input_path=final_output / input_path.relative_to(staged_path),
        input_sha256=input_sha,
        seed=_dataset_seed(root_seed),
    )
    return record, spec


def _publish(staged: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    if not output.exists():
        os.replace(staged, output)
        return
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def generate(
    output_dir: Path,
    resource_path: Path,
    *,
    seeds: Sequence[int],
    n_subjects: int = 20,
    mean_cells_per_sample: int = 240,
    n_jobs: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write immutable M2 cell-level inputs, truth, and run configuration."""

    output_dir = output_dir.resolve()
    resource_path = resource_path.resolve()
    resolved_seeds = tuple(map(int, seeds))
    if not resolved_seeds or len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("seeds must be non-empty and unique")
    if n_subjects < 12:
        raise ValueError("M2 fixture requires at least 12 paired subjects")
    if mean_cells_per_sample < 100:
        raise ValueError("mean_cells_per_sample must be at least 100")
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    source_resource = pd.read_csv(resource_path, sep="\t")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        resource, frozen_resource_path, frozen_resource_manifest = (
            _freeze_simulation_resource(
                source_resource,
                resource_path,
                staged / "resource",
            )
        )
        known = resource.loc[
            resource["ligand"].astype(str).eq(KNOWN_LIGAND)
            & resource["receptor"].astype(str).eq(KNOWN_RECEPTOR)
        ]
        if len(known) != 1:
            raise ValueError("H-common resource must contain one CXCL10-CXCR3 row")
        interaction_id = str(known.iloc[0]["harmonized_interaction_id"])
        program = pd.DataFrame.from_records(
            [
                {
                    "program_id": "CXCL10_CXCR3_activation_targets_v1",
                    "interaction_id": interaction_id,
                    "ligand": KNOWN_LIGAND,
                    "receptor": KNOWN_RECEPTOR,
                    "receiver": KNOWN_RECEIVER,
                    "target_gene": gene,
                    "weight": 1.0 / len(TARGETS),
                    "expected_program_direction": 1,
                }
                for gene in TARGETS
            ]
        )
        program_path = staged / "program_definitions.tsv"
        program.to_csv(program_path, sep="\t", index=False, lineterminator="\n")
        truth = pd.DataFrame.from_records(
            [
                {
                    "sender": sender,
                    "sender_role": role,
                    "expected_coupled_sender": sender == "TrueSender",
                    "receiver": KNOWN_RECEIVER,
                    "interaction_id": interaction_id,
                }
                for sender, role in CANDIDATE_ROLES.items()
            ]
        )
        truth_path = staged / "sender_truth.tsv"
        truth.to_csv(truth_path, sep="\t", index=False, lineterminator="\n")

        (staged / "inputs").mkdir()
        records: list[dict[str, Any]] = []
        datasets: dict[str, Any] = {}
        workers = min(n_jobs, len(resolved_seeds))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(
                    _generate_task,
                    staged=str(staged),
                    output_dir=str(output_dir),
                    replicate=replicate,
                    root_seed=root_seed,
                    n_subjects=n_subjects,
                    mean_cells_per_sample=mean_cells_per_sample,
                )
                for replicate, root_seed in enumerate(resolved_seeds, start=1)
            ]
            for future in as_completed(futures):
                record, spec = future.result()
                records.append(record)
                datasets[str(record["dataset_id"])] = spec
        records.sort(key=lambda record: str(record["dataset_id"]))
        datasets = dict(sorted(datasets.items()))

        config = {
            "schema_version": "canonical-v0.1",
            "database_root": str(
                (Path(__file__).resolve().parents[2] / "../databases").resolve()
            ),
            "output_root": str((output_dir.parent / "runs/crychic").resolve()),
            "resources": {
                "synthetic_hcommon": {
                    "adapter": "harmonized_fixture",
                    "manifest": str(
                        (
                            output_dir / frozen_resource_manifest.relative_to(staged)
                        ).resolve()
                    ),
                    "species": "human",
                }
            },
            "datasets": datasets,
        }
        config_path = staged / "crychic_config.json"
        _write_json(config_path, config)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "seeds": list(resolved_seeds),
            "design": "paired_subject",
            "n_subjects": int(n_subjects),
            "mean_cells_per_sample": int(mean_cells_per_sample),
            "generation_workers": int(workers),
            "conditions": list(CONDITIONS),
            "candidate_roles": dict(CANDIDATE_ROLES),
            "resource": {
                "filename": str(frozen_resource_path.relative_to(staged)),
                "manifest": str(frozen_resource_manifest.relative_to(staged)),
                "sha256": sha256_file(frozen_resource_path),
            },
            "program_definition": {
                "filename": program_path.name,
                "sha256": sha256_file(program_path),
                "rows": int(len(program)),
            },
            "truth": {
                "filename": truth_path.name,
                "sha256": sha256_file(truth_path),
                "rows": int(len(truth)),
            },
            "crychic_config": {
                "filename": config_path.name,
                "sha256": sha256_file(config_path),
            },
            "records": records,
            "total_cells": int(sum(int(record["cells"]) for record in records)),
            "leakage_controls": {
                "dataset_ids_mask_sender_truth": True,
                "generation_latents_unavailable_to_methods": True,
                "composition_design_index_is_observed_method_input": True,
                "subject_is_inferential_unit": True,
            },
        }
        _write_json(staged / "manifest.json", manifest)
        _publish(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-subjects", type=int, default=20)
    parser.add_argument("--mean-cells-per-sample", type=int, default=240)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = generate(
        args.output_dir,
        args.resource,
        seeds=args.seeds,
        n_subjects=args.n_subjects,
        mean_cells_per_sample=args.mean_cells_per_sample,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
