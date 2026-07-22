"""Generate checksum-bound three-group CCC fixtures with planted event effects."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-three-group-fixture-v2"
CONDITIONS = ("A", "B", "C")
CONTRASTS = (("B", "A"), ("C", "A"), ("C", "B"))
DEFAULT_GROUP_SUBJECTS = {"A": 8, "B": 10, "C": 12}
CELL_TYPES = ("Sender", "Receiver", "Bystander")
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
RECEIVER_TARGETS = ("ISG15", "IFIT1", "MX1", "OAS1", "STAT1", "B2M")
BYSTANDER_TARGETS = ("FOS", "JUN", "DUSP1")
PROGRAM_LEVELS = {
    ("Sender", "Receiver", "CXCL10", "CXCR3"): {"A": 0.0, "B": 1.0, "C": 2.0},
    ("Sender", "Bystander", "CCL5", "CCR5"): {"A": 0.0, "B": 2.0, "C": 0.0},
    ("Bystander", "Receiver", "VEGFA", "FLT1"): {"A": 0.0, "B": 0.0, "C": 2.0},
}
SIMULATION_LR_PAIRS = (
    ("CXCL10", "CXCR3"),
    ("CCL5", "CCR5"),
    ("VEGFA", "FLT1"),
    ("CXCL12", "CXCR4"),
    ("EGF", "EGFR"),
)
_GENE_INDEX = {gene: index for index, gene in enumerate(GENES)}


def _base_profile(cell_type: str) -> np.ndarray:
    rates = np.full(len(GENES), 0.03, dtype=float)
    for gene, value in {
        "GAPDH": 2.5,
        "RPLP0": 2.2,
        "MALAT1": 3.0,
        "ACTB": 2.0,
        "ISG15": 0.10,
        "IFIT1": 0.08,
        "MX1": 0.08,
        "OAS1": 0.08,
        "STAT1": 0.12,
        "B2M": 0.20,
        "FOS": 0.08,
        "JUN": 0.08,
        "DUSP1": 0.08,
    }.items():
        rates[_GENE_INDEX[gene]] = value

    if cell_type == "Sender":
        rates[_GENE_INDEX["CXCL10"]] = 0.9
        rates[_GENE_INDEX["CCL5"]] = 0.8
        rates[_GENE_INDEX["CXCL12"]] = 0.6
        rates[_GENE_INDEX["EGF"]] = 0.6
    elif cell_type == "Receiver":
        rates[_GENE_INDEX["CXCR3"]] = 0.9
        rates[_GENE_INDEX["FLT1"]] = 0.8
        rates[_GENE_INDEX["CXCR4"]] = 0.7
        rates[_GENE_INDEX["EGFR"]] = 0.7
    elif cell_type == "Bystander":
        rates[_GENE_INDEX["VEGFA"]] = 0.8
        rates[_GENE_INDEX["CCR5"]] = 0.8
    else:
        raise ValueError(f"unknown cell type: {cell_type}")
    return rates


def _profile(cell_type: str, condition: str, scenario: str) -> np.ndarray:
    rates = _base_profile(cell_type)
    if scenario == "global_null":
        return rates
    if scenario != "active":
        raise ValueError(f"unknown scenario: {scenario}")

    if condition == "B":
        if cell_type == "Sender":
            rates[_GENE_INDEX["CXCL10"]] *= 3.0
            rates[_GENE_INDEX["CCL5"]] *= 8.0
        if cell_type == "Receiver":
            for gene in RECEIVER_TARGETS:
                rates[_GENE_INDEX[gene]] *= 2.0
        if cell_type == "Bystander":
            for gene in BYSTANDER_TARGETS:
                rates[_GENE_INDEX[gene]] *= 5.0
    elif condition == "C":
        if cell_type == "Sender":
            rates[_GENE_INDEX["CXCL10"]] *= 8.0
        if cell_type == "Bystander":
            rates[_GENE_INDEX["VEGFA"]] *= 8.0
        if cell_type == "Receiver":
            for gene in RECEIVER_TARGETS:
                rates[_GENE_INDEX[gene]] *= 5.0
    return rates


def simulate_three_group(
    scenario: str,
    *,
    group_subjects: Mapping[str, int],
    mean_cells_per_sample: int,
    seed: int,
) -> ad.AnnData:
    """Simulate three independent groups with biological subjects as replicates."""
    if set(group_subjects) != set(CONDITIONS):
        raise ValueError("group_subjects must define exactly A, B, and C")
    if any(
        isinstance(value, bool) or not 8 <= value <= 40
        for value in group_subjects.values()
    ):
        raise ValueError("every condition must contain between 8 and 40 subjects")
    if len(set(group_subjects.values())) == 1:
        raise ValueError(
            "the registered three-group design requires unequal group sizes"
        )
    if mean_cells_per_sample < 60:
        raise ValueError("mean_cells_per_sample must be at least 60")
    rng = np.random.default_rng(seed)
    proportions = np.array([0.34, 0.36, 0.30])
    count_parts: list[np.ndarray] = []
    observation_parts: list[pd.DataFrame] = []

    for condition in CONDITIONS:
        for subject_index in range(int(group_subjects[condition])):
            subject_id = f"{condition}:S{subject_index + 1:02d}"
            sample_id = subject_id
            subject_scale = float(rng.lognormal(mean=0.0, sigma=0.12))
            realized = rng.dirichlet(proportions * 160.0)
            total_cells = max(60, int(rng.poisson(mean_cells_per_sample)))
            sizes = rng.multinomial(total_cells, realized)
            for cell_type, n_cells in zip(CELL_TYPES, sizes, strict=True):
                if n_cells == 0:
                    continue
                mean = _profile(cell_type, condition, scenario) * subject_scale * 10.0
                cell_depth = rng.lognormal(mean=0.0, sigma=0.30, size=n_cells)
                expected = cell_depth[:, None] * mean[None, :]
                dispersion = 0.12
                latent = rng.gamma(
                    shape=1.0 / dispersion,
                    scale=expected * dispersion,
                )
                counts = np.asarray(rng.poisson(latent), dtype=np.int32)
                count_parts.append(counts)
                observation_parts.append(
                    pd.DataFrame(
                        {
                            "sample_id": sample_id,
                            "subject_id": subject_id,
                            "family_id": subject_id,
                            "condition": condition,
                            "cell_type": cell_type,
                        },
                        index=[
                            f"{sample_id}:{cell_type}:{cell_index:04d}"
                            for cell_index in range(n_cells)
                        ],
                    )
                )

    counts = sparse.csr_matrix(np.vstack(count_parts), dtype=np.int32)
    library_size = np.asarray(counts.sum(axis=1)).ravel()
    if (library_size <= 0).any():
        raise RuntimeError("simulated cells must have positive library sizes")
    normalized = counts.astype(np.float64).multiply((1.0e4 / library_size)[:, None])
    normalized = sparse.csr_matrix(normalized)
    normalized.data = np.log1p(normalized.data)
    observations = pd.concat(observation_parts, axis=0)
    variables = pd.DataFrame(index=pd.Index(GENES, name="gene_symbol"))
    result = ad.AnnData(X=normalized, obs=observations, var=variables)
    result.layers["counts"] = counts
    result.uns["expression_contract"] = {
        "design": "independent_subject_groups",
        "x_semantics": "log1p_counts_per_10000",
        "counts_layer": "counts",
    }
    return result


def _truth_table(
    resource: pd.DataFrame,
    *,
    dataset_id: str,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target, reference in CONTRASTS:
        for sender in CELL_TYPES:
            for receiver in CELL_TYPES:
                for row in resource.itertuples(index=False):
                    key = (sender, receiver, str(row.ligand), str(row.receptor))
                    levels = PROGRAM_LEVELS.get(key, {}) if scenario == "active" else {}
                    effect = float(levels.get(target, 0.0) - levels.get(reference, 0.0))
                    records.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "dataset_id": dataset_id,
                            "scenario": scenario,
                            "seed": seed,
                            "contrast": f"{target}_vs_{reference}",
                            "target": target,
                            "reference": reference,
                            "sender": sender,
                            "receiver": receiver,
                            "interaction_id": str(row.harmonized_interaction_id),
                            "ligand": str(row.ligand),
                            "receptor": str(row.receptor),
                            "truth_effect": effect,
                            "truth_label": int(effect != 0.0),
                            "truth_direction": int(np.sign(effect)),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _freeze_simulation_resource(
    source: pd.DataFrame,
    source_path: Path,
    directory: Path,
) -> tuple[pd.DataFrame, Path, Path]:
    source_manifest_path = source_path.with_name("manifest.json")
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else {}
    )
    if not isinstance(source_manifest, dict):
        raise ValueError("the source resource manifest must contain an object")
    required = {
        "harmonized_interaction_id",
        "ligand",
        "receptor",
        "scseqcommdiff_source_interaction_id",
        "scseqcommdiff_covered",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            "the four-method source resource is incomplete: "
            f"missing={sorted(missing)}"
        )
    pair_order = {pair: index for index, pair in enumerate(SIMULATION_LR_PAIRS)}
    pairs = list(
        zip(source["ligand"].astype(str), source["receptor"].astype(str), strict=True)
    )
    selected = source.loc[[pair in pair_order for pair in pairs]].copy()
    observed = set(
        zip(
            selected["ligand"].astype(str),
            selected["receptor"].astype(str),
            strict=True,
        )
    )
    if (
        observed != set(SIMULATION_LR_PAIRS)
        or len(selected) != len(SIMULATION_LR_PAIRS)
    ):
        raise ValueError(
            "the source resource does not contain the exact simulation LR axis"
        )
    covered = selected["scseqcommdiff_covered"].astype(str).str.lower().eq("true")
    if not covered.all():
        raise ValueError("the simulation LR axis is not covered by scSeqCommDiff")
    selected["_order"] = [
        pair_order[(str(ligand), str(receptor))]
        for ligand, receptor in zip(
            selected["ligand"], selected["receptor"], strict=True
        )
    ]
    selected = selected.sort_values("_order", kind="stable").drop(columns="_order")
    directory.mkdir()
    table_path = directory / "harmonized_lr.tsv"
    selected.to_csv(table_path, sep="\t", index=False, lineterminator="\n")
    manifest_path = directory / "manifest.json"
    manifest = {
        "schema_version": "crychic-harmonized-lr-v1",
        "resource_id": "crychic_three_group_four_method_hcommon",
        "version": "2026-07-23",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "license": str(
            source_manifest.get(
                "license", "intersection-only; retain upstream method licenses"
            )
        ),
        "citation": source_manifest.get(
            "citation",
            [
                "CellChat, CellPhoneDB, LIANA, and ConnectomeDB source resources; "
                "see the checksum-bound source manifest."
            ],
        ),
        "construction": {
            "rule": "exact five-pair simulation subset of the four-method common axis",
            "source_rows": len(source),
            "retained_interactions": len(selected),
        },
        "source": {
            "filename": source_path.name,
            "sha256": sha256_file(source_path),
            "manifest_sha256": (
                sha256_file(source_manifest_path)
                if source_manifest_path.is_file()
                else None
            ),
        },
        "payload": {
            "filename": table_path.name,
            "rows": len(selected),
            "bytes": table_path.stat().st_size,
            "sha256": sha256_file(table_path),
        },
    }
    _write_json(manifest_path, manifest)
    return selected, table_path, manifest_path


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


def _crychic_dataset_spec(
    *,
    dataset_id: str,
    input_path: Path,
    input_sha256: str,
    seed: int,
    benchmark_scope: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "input": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "output_name": dataset_id,
        "input_mode": "counts",
        "lr_resource": "synthetic_hcommon",
        "target_prior": "nichenet_human",
        "benchmark_scope": benchmark_scope,
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
            "subject_fixed_effects": False,
            "lambda1": 0.0,
            "lambda2": 0.0,
            "cosine_threshold": 0.95,
            "prior_quality": 1.0,
        },
    }


def generate(
    output_dir: Path,
    resource_path: Path,
    *,
    seeds: Sequence[int],
    group_subjects: Mapping[str, int] = DEFAULT_GROUP_SUBJECTS,
    mean_cells_per_sample: int = 180,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate full and pairwise H5AD inputs, truth, manifests, and CRYCHIC config."""
    output_dir = output_dir.resolve()
    resource_path = resource_path.resolve()
    source_resource = pd.read_csv(resource_path, sep="\t")
    required = {"harmonized_interaction_id", "ligand", "receptor"}
    missing = required.difference(source_resource.columns)
    if missing or source_resource.empty:
        raise ValueError(f"resource is invalid: missing={sorted(missing)}")
    if source_resource.duplicated(["ligand", "receptor"]).any():
        raise ValueError("resource ligand-receptor pairs must be unique")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be non-empty and unique")

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
        inputs = staged / "inputs"
        inputs.mkdir()
        truth_parts: list[pd.DataFrame] = []
        records: list[dict[str, Any]] = []
        crychic_datasets: dict[str, Any] = {}
        for replicate_index, seed in enumerate(seeds, start=1):
            for arm_index, scenario in enumerate(("active", "global_null"), start=1):
                # The method-facing identifier deliberately does not expose the arm.
                dataset_id = f"three_group_r{replicate_index:02d}_arm{arm_index:02d}"
                dataset_dir = inputs / dataset_id
                dataset_dir.mkdir()
                adata = simulate_three_group(
                    scenario,
                    group_subjects=group_subjects,
                    mean_cells_per_sample=mean_cells_per_sample,
                    seed=int(seed),
                )
                full_path = dataset_dir / f"{dataset_id}.h5ad"
                adata.write_h5ad(full_path, compression="gzip")
                full_sha = sha256_file(full_path)
                full_manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": dataset_id,
                    "scenario": scenario,
                    "seed": int(seed),
                    "inferential_unit": "subject_id",
                    "design": "independent_subject_groups",
                    "group_subjects": {
                        key: int(value) for key, value in group_subjects.items()
                    },
                    "expression": {
                        "X": "log1p_counts_per_10000",
                        "counts_layer": "counts",
                    },
                    "output": {
                        "filename": full_path.name,
                        "sha256": full_sha,
                        "shape": [int(adata.n_obs), int(adata.n_vars)],
                    },
                }
                _write_json(dataset_dir / "manifest.json", full_manifest)

                pair_records: list[dict[str, Any]] = []
                for target, reference in CONTRASTS:
                    contrast = f"{target}_vs_{reference}"
                    pair = adata[
                        adata.obs["condition"].astype(str).isin((target, reference))
                    ].copy()
                    pair_path = dataset_dir / f"{dataset_id}.{contrast}.h5ad"
                    pair.write_h5ad(pair_path, compression="gzip")
                    pair_sha = sha256_file(pair_path)
                    pair_manifest = {
                        "schema_version": SCHEMA_VERSION,
                        "dataset_id": dataset_id,
                        "contrast": contrast,
                        "target": target,
                        "reference": reference,
                        "inferential_unit": "subject_id",
                        "design": "independent_subject_groups",
                        "output": {
                            "filename": pair_path.name,
                            "sha256": pair_sha,
                            "shape": [int(pair.n_obs), int(pair.n_vars)],
                        },
                    }
                    pair_manifest_path = dataset_dir / f"{contrast}.manifest.json"
                    _write_json(pair_manifest_path, pair_manifest)
                    pair_records.append(
                        {
                            "contrast": contrast,
                            "target": target,
                            "reference": reference,
                            "h5ad": str(pair_path.relative_to(staged)),
                            "manifest": str(pair_manifest_path.relative_to(staged)),
                        }
                    )
                records.append(
                    {
                        "dataset_id": dataset_id,
                        "scenario": scenario,
                        "seed": int(seed),
                        "h5ad": str(full_path.relative_to(staged)),
                        "manifest": str(
                            (dataset_dir / "manifest.json").relative_to(staged)
                        ),
                        "pairs": pair_records,
                    }
                )
                truth_parts.append(
                    _truth_table(
                        resource,
                        dataset_id=dataset_id,
                        scenario=scenario,
                        seed=int(seed),
                    )
                )
                crychic_datasets[dataset_id] = _crychic_dataset_spec(
                    dataset_id=dataset_id,
                    input_path=(output_dir / full_path.relative_to(staged)).resolve(),
                    input_sha256=full_sha,
                    seed=int(seed),
                    benchmark_scope="independent_subject_three_group_hcommon",
                )
        truth = pd.concat(truth_parts, ignore_index=True)
        truth_path = staged / "event_truth.tsv"
        truth.to_csv(truth_path, sep="\t", index=False, lineterminator="\n")
        crychic_config = {
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
                            output_dir
                            / frozen_resource_manifest.relative_to(staged)
                        ).resolve()
                    ),
                    "species": "human",
                },
                "nichenet_human": {
                    "manifest": str(
                        (
                            Path(__file__).resolve().parents[2]
                            / "resources/nichenet_human_v2_2021.json"
                        ).resolve()
                    ),
                    "release": "v2_2021",
                },
            },
            "datasets": crychic_datasets,
        }
        config_path = staged / "crychic_config.json"
        _write_json(config_path, crychic_config)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "seeds": list(map(int, seeds)),
            "design": "independent_subject_groups",
            "group_subjects": {
                key: int(value) for key, value in group_subjects.items()
            },
            "mean_cells_per_sample": mean_cells_per_sample,
            "conditions": list(CONDITIONS),
            "contrasts": [
                f"{target}_vs_{reference}" for target, reference in CONTRASTS
            ],
            "resource": {
                "filename": str(frozen_resource_path.relative_to(staged)),
                "manifest": str(frozen_resource_manifest.relative_to(staged)),
                "sha256": sha256_file(frozen_resource_path),
                "rows": len(resource),
                "source_sha256": sha256_file(resource_path),
            },
            "records": records,
            "outputs": {
                "truth": {
                    "filename": truth_path.name,
                    "rows": len(truth),
                    "sha256": sha256_file(truth_path),
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
        seeds = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _parse_group_subjects(value: str) -> dict[str, int]:
    try:
        sizes = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "group sizes must be three comma-separated integers"
        ) from error
    if len(sizes) != len(CONDITIONS):
        raise argparse.ArgumentTypeError("group sizes must define A, B, and C")
    return dict(zip(CONDITIONS, sizes, strict=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--seeds", type=_parse_seeds, default=(20260723,))
    parser.add_argument(
        "--group-subjects",
        type=_parse_group_subjects,
        default=DEFAULT_GROUP_SUBJECTS,
        metavar="A,B,C",
    )
    parser.add_argument("--mean-cells-per-sample", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = generate(
        args.output_dir,
        args.resource,
        seeds=args.seeds,
        group_subjects=args.group_subjects,
        mean_cells_per_sample=args.mean_cells_per_sample,
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
