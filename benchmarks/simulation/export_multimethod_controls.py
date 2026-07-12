"""Export normalized/count-layer synthetic controls for every benchmark method."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.simulation.generate import (
    DECOY_INTERACTIONS,
    Scenario,
    simulate_ccc,
)

SCENARIOS: tuple[Scenario, ...] = (
    "active",
    "global_null",
    "abundance_only",
    "receiver_autonomous",
    "ligand_only",
    "target_only",
    "receptor_knockout",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_seed(seed: int, scenario: Scenario) -> int:
    payload = f"crychic-multimethod-control-v1:{seed}:{scenario}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def export_controls(
    output_dir: Path,
    *,
    n_subjects: int,
    mean_cells_per_sample: int,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    """Write one checksum-pinned h5ad per preregistered mechanism scenario."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        scenario_seed = _scenario_seed(seed, scenario)
        simulation = simulate_ccc(
            scenario,
            n_subjects=n_subjects,
            mean_cells_per_sample=mean_cells_per_sample,
            seed=scenario_seed,
        )
        adata = simulation.adata
        counts = sparse.csr_matrix(adata.layers["counts"], dtype=np.int32)
        library_size = np.asarray(counts.sum(axis=1)).ravel()
        normalized = counts.astype(np.float32).copy()
        scale = np.divide(
            10_000.0,
            library_size,
            out=np.zeros_like(library_size, dtype=np.float32),
            where=library_size > 0,
        )
        normalized = (sparse.diags(scale) @ normalized).tocsr()
        normalized.data = np.log1p(normalized.data)
        adata.X = normalized
        adata.obs["scenario"] = scenario
        adata.uns["simulation_truth"].update(
            {
                "dataset_id": f"synthetic_{scenario}",
                "scenario_seed": scenario_seed,
                "primary_contrast": "stim_vs_ctrl",
                "truth_scope": "simulation",
                "x_semantics": "log1p library-normalized to 10000",
                "counts_layer_semantics": "raw non-negative integer counts",
            }
        )
        output = output_dir / f"synthetic_{scenario}.h5ad"
        adata.write_h5ad(output, compression="gzip", compression_opts=4)
        records.append(
            {
                "dataset_id": f"synthetic_{scenario}",
                "scenario": scenario,
                "scenario_seed": scenario_seed,
                "path": output.name,
                "sha256": _sha256(output),
                "shape": [adata.n_obs, adata.n_vars],
                "n_subjects": n_subjects,
                "n_samples": int(adata.obs["sample_id"].nunique()),
                "active_interaction": simulation.active_interaction,
                "expected_state_change": simulation.expected_state_change,
                "expected_ecosystem_change": simulation.expected_ecosystem_change,
                "expected_receiver_response": (
                    simulation.expected_receiver_response
                ),
                "expected_integrated_edge": simulation.expected_integrated_edge,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "crychic-multimethod-synthetic-v1",
        "seed": seed,
        "n_subjects": n_subjects,
        "mean_cells_per_sample": mean_cells_per_sample,
        "statistical_unit": "subject_id",
        "cell_level_values_are_not_independent_replicates": True,
        "expressed_harmonized_decoy_interactions": [
            list(pair) for pair in DECOY_INTERACTIONS
        ],
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(records).to_csv(
        output_dir / "scenario_truth.tsv", sep="\t", index=False
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--mean-cells-per-sample", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = export_controls(
        args.output_dir,
        n_subjects=args.n_subjects,
        mean_cells_per_sample=args.mean_cells_per_sample,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
