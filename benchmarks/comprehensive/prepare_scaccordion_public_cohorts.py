"""Prepare public AKI/RCC cells for the under-100k scACCorDiON benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-scaccordion-public-cohort-preparation-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def largest_remainder_allocation(
    counts: pd.Series, target: int, *, minimum_per_nonempty_stratum: int = 0
) -> pd.Series:
    """Allocate an exact integer target under per-stratum capacities."""

    values = counts.astype(int).sort_index()
    if values.empty or (values <= 0).any():
        raise ValueError("allocation counts must be positive")
    if target < 1 or target > int(values.sum()):
        raise ValueError("allocation target exceeds available observations")
    minimum = np.minimum(values.to_numpy(), minimum_per_nonempty_stratum)
    if int(minimum.sum()) > target:
        raise ValueError("stratum minima exceed allocation target")
    capacities = values.to_numpy() - minimum
    remaining = target - int(minimum.sum())
    if remaining == 0:
        return pd.Series(minimum, index=values.index, dtype=int)
    weights = capacities.astype(float)
    if weights.sum() == 0:
        raise ValueError("no residual capacity for allocation")
    raw = remaining * weights / weights.sum()
    extra = np.floor(raw).astype(int)
    extra = np.minimum(extra, capacities)
    shortfall = remaining - int(extra.sum())
    fractions = raw - np.floor(raw)
    order = sorted(
        range(len(values)),
        key=lambda idx: (-fractions[idx], str(values.index[idx])),
    )
    while shortfall > 0:
        progressed = False
        for idx in order:
            if extra[idx] < capacities[idx]:
                extra[idx] += 1
                shortfall -= 1
                progressed = True
                if shortfall == 0:
                    break
        if not progressed:
            raise RuntimeError("allocation could not satisfy the exact target")
    result = pd.Series(minimum + extra, index=values.index, dtype=int)
    if int(result.sum()) != target or (result > values).any():
        raise RuntimeError("invalid largest-remainder allocation")
    return result


def select_aki_samples(
    obs: pd.DataFrame,
    *,
    sample_column: str,
    label_column: str,
    control_label: str,
    control_samples: int,
) -> tuple[list[str], pd.DataFrame]:
    """Keep all disease specimens and the largest public control specimens."""

    sample_meta = (
        obs.groupby(sample_column, observed=True)
        .agg(
            label=(label_column, "first"),
            source_cells=(label_column, "size"),
            label_count=(label_column, "nunique"),
        )
        .reset_index()
    )
    if (sample_meta["label_count"] != 1).any():
        raise ValueError("a sample maps to more than one phenotype label")
    is_control = sample_meta["label"].astype(str).eq(control_label)
    controls = sample_meta.loc[is_control].copy()
    disease = sample_meta.loc[~is_control].copy()
    if len(controls) < control_samples or disease.empty:
        raise ValueError("AKI source lacks the requested disease/control design")
    controls = controls.sort_values(
        ["source_cells", sample_column], ascending=[False, True]
    ).head(control_samples)
    selected = pd.concat([disease, controls], ignore_index=True).sort_values(
        sample_column
    )
    return selected[sample_column].astype(str).tolist(), selected


def deterministic_stratified_indices(
    obs: pd.DataFrame,
    *,
    sample_column: str,
    cell_type_column: str,
    selected_samples: Sequence[str],
    target_cells: int,
    seed: int,
    minimum_per_stratum: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Select exact cells while preserving sample-by-cell-type strata."""

    selected_mask = obs[sample_column].astype(str).isin(selected_samples)
    source = obs.loc[selected_mask, [sample_column, cell_type_column]].copy()
    source["_position"] = np.flatnonzero(selected_mask.to_numpy())
    source[sample_column] = source[sample_column].astype(str)
    source[cell_type_column] = source[cell_type_column].astype(str)
    grouped = source.groupby(
        [sample_column, cell_type_column], observed=True, sort=True
    )
    counts = grouped.size()
    allocation = largest_remainder_allocation(
        counts,
        target_cells,
        minimum_per_nonempty_stratum=minimum_per_stratum,
    )
    selected_positions: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for key, frame in grouped:
        available = frame["_position"].to_numpy(dtype=int)
        requested = int(allocation.loc[key])
        digest = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).digest()
        local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(local_seed)
        chosen = np.sort(rng.choice(available, requested, replace=False))
        selected_positions.append(chosen)
        records.append(
            {
                "sample_id": key[0],
                "cell_type": key[1],
                "source_cells": len(available),
                "selected_cells": requested,
                "selection_seed": local_seed,
            }
        )
    positions = np.sort(np.concatenate(selected_positions))
    if len(positions) != target_cells or len(np.unique(positions)) != target_cells:
        raise RuntimeError("stratified selection did not produce an exact unique set")
    return positions, pd.DataFrame.from_records(records)


def _gene_names(adata: Any, symbol_column: str | None) -> pd.Index:
    if symbol_column is None:
        names = pd.Index(adata.var_names.astype(str))
    else:
        if symbol_column not in adata.var:
            raise ValueError(f"missing gene-symbol column: {symbol_column}")
        names = pd.Index(adata.var[symbol_column].astype(str))
    if names.isna().any() or (names == "").any():
        raise ValueError("gene symbols contain missing or empty values")
    return names


def _write_sample_h5ads(
    adata: Any,
    *,
    positions: np.ndarray,
    sample_column: str,
    label_column: str,
    annotation_columns: Mapping[str, str],
    symbol_column: str | None,
    sample_dir: Path,
) -> pd.DataFrame:
    import anndata as ad

    sample_dir.mkdir(parents=True, exist_ok=False)
    names = _gene_names(adata, symbol_column)
    selected_obs = adata.obs.iloc[positions].copy()
    records: list[dict[str, Any]] = []
    for sample_id in sorted(selected_obs[sample_column].astype(str).unique()):
        local_mask = selected_obs[sample_column].astype(str).eq(sample_id).to_numpy()
        local_positions = positions[local_mask]
        local_obs = adata.obs.iloc[local_positions]
        frame = pd.DataFrame(index=adata.obs_names[local_positions].copy())
        frame["sample_id"] = sample_id
        labels = local_obs[label_column].astype(str).unique()
        if len(labels) != 1:
            raise ValueError(f"sample {sample_id} has multiple labels")
        frame["label"] = labels[0]
        for output_column, source_column in annotation_columns.items():
            frame[output_column] = local_obs[source_column].astype(str).to_numpy()
        # Slice only the inferential matrix.  A full AnnData copy also deep-copies
        # unrelated ``uns`` payloads, which can exceed the expression matrix.
        prepared = ad.AnnData(
            X=adata.X[local_positions, :].copy(),
            obs=frame,
            var=adata.var.copy(),
        )
        prepared.var_names = names.copy()
        prepared.var_names_make_unique()
        prepared.var.index.name = None
        prepared.raw = None
        output = sample_dir / f"{sample_id}.h5ad"
        prepared.write_h5ad(output, compression="gzip")
        records.append(
            {
                "sample_id": sample_id,
                "label": labels[0],
                "cell_count": int(prepared.n_obs),
                "feature_count": int(prepared.n_vars),
                "file": output.name,
                "sha256": sha256_file(output),
            }
        )
    return pd.DataFrame.from_records(records)


def prepare_aki(
    args: argparse.Namespace, adata: Any
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    required = {
        args.sample_column,
        args.label_column,
        args.major_cell_type_column,
    }
    missing = required.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"AKI source lacks columns: {sorted(missing)}")
    samples, source_metadata = select_aki_samples(
        adata.obs,
        sample_column=args.sample_column,
        label_column=args.label_column,
        control_label=args.control_label,
        control_samples=args.control_samples,
    )
    if len(samples) != args.expected_samples:
        raise ValueError(
            f"AKI selection yielded {len(samples)} rather than "
            f"{args.expected_samples} samples"
        )
    positions, strata = deterministic_stratified_indices(
        adata.obs,
        sample_column=args.sample_column,
        cell_type_column=args.major_cell_type_column,
        selected_samples=samples,
        target_cells=args.target_cells,
        seed=args.seed,
        minimum_per_stratum=args.minimum_per_stratum,
    )
    metadata = _write_sample_h5ads(
        adata,
        positions=positions,
        sample_column=args.sample_column,
        label_column=args.label_column,
        annotation_columns={"cell_type_major": args.major_cell_type_column},
        symbol_column=args.gene_symbol_column,
        sample_dir=args.output / "samples",
    )
    metadata = metadata.merge(
        source_metadata.loc[:, [args.sample_column, "source_cells"]].rename(
            columns={args.sample_column: "sample_id"}
        ),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    return (
        metadata,
        strata,
        "paper_design_matched_public_reconstruction_not_authors_subset",
    )


def prepare_rcc(
    args: argparse.Namespace, adata: Any
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    required = {
        args.sample_column,
        args.label_column,
        args.coarse_cell_type_column,
        args.fine_cell_type_column,
    }
    missing = required.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"RCC source lacks columns: {sorted(missing)}")
    if adata.n_obs != args.target_cells:
        raise ValueError(
            f"RCC source has {adata.n_obs}, expected exact {args.target_cells} cells"
        )
    positions = np.arange(adata.n_obs, dtype=int)
    metadata = _write_sample_h5ads(
        adata,
        positions=positions,
        sample_column=args.sample_column,
        label_column=args.label_column,
        annotation_columns={
            "cell_type_coarse": args.coarse_cell_type_column,
            "cell_type_fine": args.fine_cell_type_column,
        },
        symbol_column=args.gene_symbol_column,
        sample_dir=args.output / "samples",
    )
    if len(metadata) != args.expected_samples:
        raise ValueError(
            f"RCC source yielded {len(metadata)}, expected "
            f"{args.expected_samples} samples"
        )
    strata = (
        adata.obs.assign(_sample=adata.obs[args.sample_column].astype(str))
        .groupby(["_sample", args.fine_cell_type_column], observed=True)
        .size()
        .rename("selected_cells")
        .reset_index()
        .rename(
            columns={"_sample": "sample_id", args.fine_cell_type_column: "cell_type"}
        )
    )
    strata["source_cells"] = strata["selected_cells"]
    strata["selection_seed"] = pd.NA
    return metadata, strata, "exact_public_raw_cohort_reconstruction"


def run(args: argparse.Namespace) -> int:
    import anndata as ad

    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    # Older AnnData releases cannot reliably combine backed CSR matrices with
    # arbitrary row indices.  Loading once avoids repeated full-matrix reads
    # and remains well below the registered 80% memory ceiling for both cohorts.
    adata = ad.read_h5ad(args.input)
    if args.cohort == "aki":
        metadata, strata, protocol_status = prepare_aki(args, adata)
    else:
        metadata, strata, protocol_status = prepare_rcc(args, adata)
    metadata.to_csv(args.output / "sample_metadata.tsv", sep="\t", index=False)
    strata.to_csv(args.output / "selection_strata.tsv", sep="\t", index=False)
    total_cells = int(metadata["cell_count"].sum())
    if total_cells != args.target_cells:
        raise RuntimeError(
            f"prepared cohort has {total_cells}, expected {args.target_cells} cells"
        )
    artifacts = {
        name: {"sha256": sha256_file(args.output / name)}
        for name in ("sample_metadata.tsv", "selection_strata.tsv")
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "cohort": args.cohort,
        "protocol_status": protocol_status,
        "selection_policy": (
            "all 27 disease specimens plus nine largest living-donor specimens; "
            "exact proportional sample-by-major-cell-type selection"
            if args.cohort == "aki"
            else "all cells and all samples from exact GSE242299 public H5AD"
        ),
        "dimensions": {
            "cells": total_cells,
            "samples": len(metadata),
            "labels": metadata["label"].value_counts().sort_index().to_dict(),
            "features": int(metadata["feature_count"].iloc[0]),
        },
        "parameters": {
            key: json_safe(value)
            for key, value in vars(args).items()
            if key not in {"input", "output", "repo_root"}
        },
        "input": {
            "path": str(args.input.resolve()),
            "size_bytes": args.input.stat().st_size,
            "sha256": sha256_file(args.input),
        },
        "execution": {
            "wall_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_repository": git_metadata(args.repo_root),
        "artifacts": artifacts,
    }
    _write_json(args.output / "manifest.json", manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("aki", "rcc"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--gene-symbol-column", default=None)
    parser.add_argument("--major-cell-type-column", default="subclass.l1")
    parser.add_argument("--coarse-cell-type-column", default=None)
    parser.add_argument("--fine-cell-type-column", default=None)
    parser.add_argument("--control-label", default="LivingDonor")
    parser.add_argument("--control-samples", type=int, default=9)
    parser.add_argument("--target-cells", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--minimum-per-stratum", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
