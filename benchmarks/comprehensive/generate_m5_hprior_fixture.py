"""Generate fixed-topology noisy-effect fixtures for M5 H_prior shrinkage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-m5-hprior-fixture-v1"
VIEW_COLUMNS = ("sender", "ligand", "receptor", "receiver", "pathway")
TOPOLOGY_SEED = 20260724


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _dataset_seed(root_seed: int) -> int:
    payload = f"crychic:m5-hprior:v1:{root_seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _topology(n_edges: int) -> pd.DataFrame:
    if n_edges < 200:
        raise ValueError("M5 topology requires at least 200 edges")
    rng = np.random.default_rng(TOPOLOGY_SEED)
    levels = {
        "sender": tuple(f"S{index:02d}" for index in range(8)),
        "ligand": tuple(f"L{index:02d}" for index in range(12)),
        "receptor": tuple(f"R{index:02d}" for index in range(12)),
        "receiver": tuple(f"C{index:02d}" for index in range(8)),
        "pathway": tuple(f"P{index:02d}" for index in range(6)),
    }
    combinations: set[tuple[str, ...]] = set()
    while len(combinations) < n_edges:
        combinations.add(tuple(str(rng.choice(levels[view])) for view in VIEW_COLUMNS))
    ordered = sorted(combinations)
    return pd.DataFrame.from_records(
        [
            {
                "edge_id": f"E{index + 1:05d}",
                **dict(zip(VIEW_COLUMNS, values, strict=True)),
            }
            for index, values in enumerate(ordered)
        ]
    )


def _simulate(
    topology: pd.DataFrame, *, root_seed: int, observation_noise_sd: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(_dataset_seed(root_seed))
    node_effects = {
        view: {
            node: float(rng.normal(scale=0.75))
            for node in sorted(topology[view].unique())
        }
        for view in VIEW_COLUMNS
    }
    components = np.column_stack(
        [
            topology[view].map(node_effects[view]).to_numpy(dtype=float)
            for view in VIEW_COLUMNS
        ]
    )
    true_effect = components.mean(axis=1) + rng.normal(scale=0.12, size=len(topology))
    true_effect = (true_effect - true_effect.mean()) / true_effect.std(ddof=1)
    estimate = true_effect + rng.normal(scale=observation_noise_sd, size=len(topology))
    threshold = float(np.quantile(np.abs(true_effect), 0.70))
    truth = pd.DataFrame(
        {
            "edge_id": topology["edge_id"],
            "true_effect": true_effect,
            "expected_active": np.abs(true_effect) >= threshold,
            "expected_direction": np.sign(true_effect).astype(int),
        }
    )
    observed = pd.DataFrame(
        {
            "edge_id": topology["edge_id"],
            "estimate": estimate,
            "observation_precision": 1.0 / observation_noise_sd**2,
        }
    )
    return observed, truth


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
    *,
    seeds: Sequence[int],
    n_edges: int = 1_500,
    observation_noise_sd: float = 1.15,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one fixed H_prior, seed-specific estimates, and separate truth."""

    output_dir = output_dir.resolve()
    resolved_seeds = tuple(map(int, seeds))
    if not resolved_seeds or len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not np.isfinite(observation_noise_sd) or observation_noise_sd <= 0.0:
        raise ValueError("observation_noise_sd must be finite and positive")
    topology = _topology(n_edges)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    try:
        topology_path = staged / "hypergraph_prior.tsv.gz"
        topology.to_csv(
            topology_path,
            sep="\t",
            index=False,
            compression=compression,
            lineterminator="\n",
        )
        inputs = staged / "inputs"
        inputs.mkdir()
        records: list[dict[str, Any]] = []
        truth_frames: list[pd.DataFrame] = []
        for replicate, root_seed in enumerate(resolved_seeds, start=1):
            dataset_id = f"m5_hprior_r{replicate:03d}"
            observed, truth = _simulate(
                topology,
                root_seed=root_seed,
                observation_noise_sd=observation_noise_sd,
            )
            dataset_dir = inputs / dataset_id
            dataset_dir.mkdir()
            input_path = dataset_dir / "edge_estimates.tsv.gz"
            observed.to_csv(
                input_path,
                sep="\t",
                index=False,
                compression=compression,
                lineterminator="\n",
            )
            truth.insert(0, "root_seed", root_seed)
            truth.insert(0, "dataset_id", dataset_id)
            truth_frames.append(truth)
            records.append(
                {
                    "dataset_id": dataset_id,
                    "root_seed": root_seed,
                    "dataset_seed": _dataset_seed(root_seed),
                    "input": str(input_path.relative_to(staged)),
                    "input_sha256": sha256_file(input_path),
                    "edges": int(len(observed)),
                }
            )
        truth_table = pd.concat(truth_frames, ignore_index=True)
        truth_path = staged / "edge_truth.tsv.gz"
        truth_table.to_csv(
            truth_path,
            sep="\t",
            index=False,
            compression=compression,
            lineterminator="\n",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "seeds": list(resolved_seeds),
            "n_edges": int(n_edges),
            "observation_noise_sd": float(observation_noise_sd),
            "topology_seed": TOPOLOGY_SEED,
            "view_columns": list(VIEW_COLUMNS),
            "hypergraph_prior": {
                "filename": topology_path.name,
                "sha256": sha256_file(topology_path),
                "outcome_blind": True,
            },
            "truth": {
                "filename": truth_path.name,
                "sha256": sha256_file(truth_path),
                "rows": int(len(truth_table)),
            },
            "records": records,
            "leakage_controls": {
                "one_prior_is_frozen_across_all_seeds": True,
                "prior_is_generated_before_seed_outcomes": True,
                "truth_is_separate_from_method_inputs": True,
                "edge_ids_mask_activity_truth": True,
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
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-edges", type=int, default=1_500)
    parser.add_argument("--observation-noise-sd", type=float, default=1.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = generate(
        args.output_dir,
        seeds=args.seeds,
        n_edges=args.n_edges,
        observation_noise_sd=args.observation_noise_sd,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
