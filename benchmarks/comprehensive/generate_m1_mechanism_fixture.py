"""Generate fresh-seed paired fixtures for M1 mechanism-family stress tests."""

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
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.generate_three_group_fixture import (
    _freeze_simulation_resource,
)
from benchmarks.simulation.export_multimethod_controls import SCENARIOS
from benchmarks.simulation.generate import TARGETS, simulate_ccc

SCHEMA_VERSION = "crychic-m1-mechanism-fixture-v1"
CONDITIONS = ("ctrl", "stim")
KNOWN_LIGAND = "CXCL10"
KNOWN_RECEPTOR = "CXCR3"
KNOWN_SENDER = "Sender"
KNOWN_RECEIVER = "Receiver"
EXPECTED_COMPONENTS: Mapping[str, tuple[bool, bool, bool]] = {
    # LR contrast, target-program contrast, direction-concordant integrated event.
    "active": (True, True, True),
    "global_null": (False, False, False),
    "abundance_only": (False, False, False),
    "receiver_autonomous": (False, False, False),
    "ligand_only": (True, False, False),
    "target_only": (False, True, False),
    "receptor_knockout": (True, True, False),
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _scenario_seed(root_seed: int, scenario: str) -> int:
    payload = f"crychic:m1-mechanism:v1:{root_seed}:{scenario}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _normalized_copy(adata: Any) -> Any:
    counts = sparse.csr_matrix(adata.layers["counts"], dtype=np.int32)
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    scale = np.divide(
        1.0e4,
        library,
        out=np.zeros_like(library),
        where=library > 0,
    )
    normalized = counts.astype(np.float64).multiply(scale[:, None])
    normalized = sparse.csr_matrix(normalized)
    normalized.data = np.log1p(normalized.data)
    result = adata.copy()
    result.X = normalized
    result.layers["counts"] = counts
    result.obs["family_id"] = result.obs["subject_id"].astype(str)
    result.uns["expression_contract"] = {
        "design": "paired_subject",
        "x_semantics": "log1p_counts_per_10000",
        "counts_layer": "counts",
    }
    return result


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
        "benchmark_scope": "paired_subject_m1_mechanism_family_stress",
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


def _generate_dataset_task(
    *,
    staged: str,
    output_dir: str,
    replicate: int,
    arm: int,
    root_seed: int,
    scenario: str,
    n_subjects: int,
    mean_cells_per_sample: int,
    known_interaction: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    staged_path = Path(staged)
    final_output = Path(output_dir)
    dataset_id = f"m1_mechanism_r{replicate:03d}_arm{arm:02d}"
    scenario_seed = _scenario_seed(root_seed, scenario)
    simulation = simulate_ccc(
        cast(Any, scenario),
        n_subjects=n_subjects,
        mean_cells_per_sample=mean_cells_per_sample,
        seed=scenario_seed,
    )
    adata = _normalized_copy(simulation.adata)
    dataset_dir = staged_path / "inputs" / dataset_id
    dataset_dir.mkdir()
    input_path = dataset_dir / f"{dataset_id}.h5ad"
    adata.write_h5ad(input_path, compression="gzip")
    input_sha = sha256_file(input_path)
    input_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "root_seed": root_seed,
        "scenario_seed": scenario_seed,
        "scenario": scenario,
        "inferential_unit": "subject_id",
        "design": "paired_subject",
        "conditions": list(CONDITIONS),
        "output": {
            "filename": input_path.name,
            "sha256": input_sha,
            "shape": [int(adata.n_obs), int(adata.n_vars)],
        },
    }
    manifest_path = dataset_dir / "manifest.json"
    _write_json(manifest_path, input_manifest)
    record = {
        "dataset_id": dataset_id,
        "root_seed": root_seed,
        "scenario_seed": scenario_seed,
        "scenario": scenario,
        "h5ad": str(input_path.relative_to(staged_path)),
        "manifest": str(manifest_path.relative_to(staged_path)),
        "sha256": input_sha,
        "cells": int(adata.n_obs),
    }
    lr_change, program_change, integrated = EXPECTED_COMPONENTS[scenario]
    truth = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "root_seed": root_seed,
        "scenario_seed": scenario_seed,
        "scenario": scenario,
        "sender": KNOWN_SENDER,
        "receiver": KNOWN_RECEIVER,
        "interaction_id": known_interaction,
        "ligand": KNOWN_LIGAND,
        "receptor": KNOWN_RECEPTOR,
        "expected_lr_change": lr_change,
        "expected_target_program_change": program_change,
        "expected_program_direction": 1,
        "expected_integrated_edge": integrated,
    }
    spec = _dataset_spec(
        dataset_id=dataset_id,
        input_path=(final_output / input_path.relative_to(staged_path)),
        input_sha256=input_sha,
        seed=scenario_seed,
    )
    return record, truth, spec


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
    n_subjects: int = 10,
    mean_cells_per_sample: int = 180,
    n_jobs: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write paired H5AD inputs, component truth, program map, and manifests."""

    output_dir = output_dir.resolve()
    resource_path = resource_path.resolve()
    if not seeds or len(set(map(int, seeds))) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if n_subjects < 4:
        raise ValueError("n_subjects must be at least four")
    if mean_cells_per_sample < 60:
        raise ValueError("mean_cells_per_sample must be at least 60")
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
        known_resource = resource.loc[
            resource["ligand"].astype(str).eq(KNOWN_LIGAND)
            & resource["receptor"].astype(str).eq(KNOWN_RECEPTOR)
        ]
        if len(known_resource) != 1:
            raise ValueError("H-common resource must contain one CXCL10-CXCR3 row")
        known_interaction = str(known_resource.iloc[0]["harmonized_interaction_id"])
        program = pd.DataFrame.from_records(
            [
                {
                    "program_id": "CXCL10_CXCR3_activation_targets_v1",
                    "interaction_id": known_interaction,
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

        inputs = staged / "inputs"
        inputs.mkdir()
        records: list[dict[str, Any]] = []
        truth_records: list[dict[str, Any]] = []
        datasets: dict[str, Any] = {}
        tasks = [
            (replicate, arm, int(root_seed), scenario)
            for replicate, root_seed in enumerate(map(int, seeds), start=1)
            for arm, scenario in enumerate(SCENARIOS, start=1)
        ]
        workers = min(int(n_jobs), len(tasks))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(
                    _generate_dataset_task,
                    staged=str(staged),
                    output_dir=str(output_dir),
                    replicate=replicate,
                    arm=arm,
                    root_seed=root_seed,
                    scenario=scenario,
                    n_subjects=n_subjects,
                    mean_cells_per_sample=mean_cells_per_sample,
                    known_interaction=known_interaction,
                )
                for replicate, arm, root_seed, scenario in tasks
            ]
            for future in as_completed(futures):
                record, truth_record, spec = future.result()
                records.append(record)
                truth_records.append(truth_record)
                datasets[str(record["dataset_id"])] = spec
        records.sort(key=lambda record: str(record["dataset_id"]))
        truth_records.sort(key=lambda record: str(record["dataset_id"]))
        datasets = dict(sorted(datasets.items()))

        truth = pd.DataFrame.from_records(truth_records)
        truth_path = staged / "mechanism_truth.tsv"
        truth.to_csv(truth_path, sep="\t", index=False, lineterminator="\n")
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
            "seeds": list(map(int, seeds)),
            "scenarios": list(SCENARIOS),
            "design": "paired_subject",
            "n_subjects": int(n_subjects),
            "mean_cells_per_sample": int(mean_cells_per_sample),
            "generation_workers": workers,
            "conditions": list(CONDITIONS),
            "resource": {
                "filename": str(frozen_resource_path.relative_to(staged)),
                "manifest": str(frozen_resource_manifest.relative_to(staged)),
                "sha256": sha256_file(frozen_resource_path),
                "rows": len(resource),
            },
            "program_definition": {
                "filename": program_path.name,
                "sha256": sha256_file(program_path),
                "rows": len(program),
                "value_transform": ("weighted_mean_clip_log1p_cpm_over_log1p_1000000"),
            },
            "records": records,
            "outputs": {
                "truth": {
                    "filename": truth_path.name,
                    "sha256": sha256_file(truth_path),
                    "rows": len(truth),
                },
                "crychic_config": {
                    "filename": config_path.name,
                    "sha256": sha256_file(config_path),
                },
            },
        }
        _write_json(staged / "manifest.json", manifest)
        _publish(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--seeds", required=True, type=_parse_seeds)
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument("--mean-cells-per-sample", type=int, default=180)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = generate(
        args.output_dir,
        args.resource,
        seeds=args.seeds,
        n_subjects=args.n_subjects,
        mean_cells_per_sample=args.mean_cells_per_sample,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "records": len(manifest["records"]),
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
