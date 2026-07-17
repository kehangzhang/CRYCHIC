"""Run CRYCHIC static availability through the Dimitrov perturbations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import time
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from benchmarks.literature.robustness import (
    PerturbationSpec,
    reshuffle_labels,
    select_top_edges,
    subsample_cells,
)
from benchmarks.openproblems.common import sha256_file, write_json
from crychic.availability import AvailabilityParameters, estimate_bundle_availability
from crychic.data import InputSchema, validate_anndata
from crychic.pseudobulk import aggregate_pseudobulk
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _subunits(value: object) -> tuple[str, ...]:
    return tuple(part for part in str(value).split("_") if part)


def _resource_bundle(table: pd.DataFrame, *, resource_id: str) -> ResourceBundle:
    required = {"ligand", "receptor"}
    if required.difference(table.columns) or table.empty:
        raise ValueError("robustness resource lacks ligand/receptor rows")
    canonical = (
        table.loc[:, ["ligand", "receptor"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    interactions: list[Interaction] = []
    genes: set[str] = set()
    for index, row in enumerate(canonical.itertuples(index=False)):
        ligand = _subunits(row.ligand)
        receptor = _subunits(row.receptor)
        genes.update((*ligand, *receptor))
        interactions.append(
            Interaction(
                interaction_id=f"{resource_id}::{index:05d}",
                source_interaction_id=f"{resource_id}::{index:05d}",
                ligand_name=str(row.ligand),
                receptor_name=str(row.receptor),
                ligand_subunits=ligand,
                receptor_subunits=receptor,
                ligand_is_complex=len(ligand) > 1,
                receptor_is_complex=len(receptor) > 1,
                direction="Ligand-Receptor",
                source=resource_id,
                version="dimitrov-robustness-v1",
                species=Species.HUMAN,
                gene_namespace=GeneNamespace.HGNC_SYMBOL,
                evidence=("LIANA consensus",),
            )
        )
    payload = canonical.to_csv(sep="\t", index=False, lineterminator="\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    return ResourceBundle(
        resource_id=resource_id,
        version="dimitrov-robustness-v1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=tuple(interactions),
        mapping_report=MappingReport(
            source_rows=len(canonical),
            loaded_rows=len(canonical),
            mapped_entities=len(genes),
            notes=("Dimitrov PBMC3k robustness resource",),
        ),
        manifest_digest=digest,
        source_files=("generated_resource.parquet",),
        license="Benchmark use subject to LIANA and upstream resource licenses",
        citation="Dimitrov D et al. Nature Communications 2022;13:3224.",
    )


def _run_method(
    data: ad.AnnData,
    resource: pd.DataFrame,
    *,
    top_n: int,
) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter()
    working = data.copy()
    for column in ("sample_id", "subject_id", "context", "cell_type"):
        if column in working.obs and isinstance(
            working.obs[column].dtype, pd.CategoricalDtype
        ):
            working.obs[column] = working.obs[column].astype("object")
    validated = validate_anndata(
        working,
        InputSchema(
            context_keys=("context",),
            counts_layer="counts",
            sample_key="sample_id",
            subject_key="subject_id",
            cell_type_key="cell_type",
            species=Species.HUMAN.value,
            gene_namespace=GeneNamespace.HGNC_SYMBOL.value,
        ),
    )
    aggregate = aggregate_pseudobulk(validated, min_cells=5)
    bundle = _resource_bundle(resource, resource_id="liana_consensus_robustness")
    availability = estimate_bundle_availability(
        aggregate,
        bundle,
        context_keys=("context",),
        parameters=AvailabilityParameters(),
        min_pooled_availability=0.0,
        max_interactions=None,
    )
    raw = availability.sample_interactions
    selected = raw.loc[raw["state_status"].astype(str).eq("observed")]
    scores = pd.DataFrame(
        {
            "method": "CRYCHIC availability-state",
            "source": selected["sender"].astype(str),
            "target": selected["receiver"].astype(str),
            "ligand": selected["ligand"].astype(str),
            "receptor": selected["receptor"].astype(str),
            "score": pd.to_numeric(selected["availability_state"], errors="coerce"),
            "score_direction": "higher",
        }
    )
    return select_top_edges(scores, top_n=top_n), time.perf_counter() - started


def _tag(
    table: pd.DataFrame,
    *,
    kind: str,
    proportion: float,
    replicate: int,
    seed: int,
) -> pd.DataFrame:
    result = table.copy()
    result.insert(0, "seed", int(seed))
    result.insert(0, "replicate", int(replicate))
    result.insert(0, "proportion", float(proportion))
    result.insert(0, "perturbation", kind)
    return result


def run(
    input_h5ad: Path,
    baseline_resource: Path,
    output_dir: Path,
    *,
    liana_run_dir: Path | None,
    top_n: int,
    base_seed: int,
    baseline_only: bool,
    overwrite: bool,
) -> dict[str, object]:
    """Run the static CRYCHIC component on the shared perturbation design."""

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"CRYCHIC robustness output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = ad.read_h5ad(input_h5ad)
    if None in data.layers:
        del data.layers[None]
    resource = pd.read_parquet(baseline_resource).loc[:, ["ligand", "receptor"]]
    resource = resource.drop_duplicates(ignore_index=True)
    baseline_top, baseline_elapsed = _run_method(data, resource, top_n=top_n)
    baseline = _tag(
        baseline_top,
        kind="baseline",
        proportion=0.0,
        replicate=0,
        seed=base_seed,
    )
    baseline_path = output_dir / "baseline_top_predictions.parquet"
    baseline.to_parquet(baseline_path, index=False)
    top_tables = [baseline]
    audit_rows: list[dict[str, Any]] = [
        {
            "perturbation": "baseline",
            "proportion": 0.0,
            "replicate": 0,
            "seed": base_seed,
            "actual_proportion": 0.0,
            "cells_before": data.n_obs,
            "cells_after": data.n_obs,
            "resource_rows": len(resource),
            "elapsed_seconds": baseline_elapsed,
        }
    ]
    if not baseline_only:
        if liana_run_dir is None:
            raise ValueError("full CRYCHIC robustness run requires liana_run_dir")
        design = pd.read_csv(liana_run_dir / "design.tsv", sep="\t")
        design = design.loc[design["perturbation"].ne("baseline")]
        for row in design.itertuples(index=False):
            spec = PerturbationSpec(
                kind=str(row.perturbation),
                proportion=float(row.proportion),
                replicate=int(row.replicate),
                seed=int(row.seed),
            )
            variant_data = data
            variant_resource = resource
            if spec.kind == "cell_subsampling":
                variant_data, audit = subsample_cells(
                    data,
                    label_key="cell_type",
                    proportion=spec.proportion,
                    seed=spec.seed,
                )
            elif spec.kind == "label_reshuffling":
                variant_data, audit = reshuffle_labels(
                    data,
                    label_key="cell_type",
                    proportion=spec.proportion,
                    seed=spec.seed,
                )
            else:
                resource_file = str(row.resource_file)
                if not resource_file or resource_file == "nan":
                    raise ValueError(
                        "resource perturbation is missing its resource file"
                    )
                variant_resource = pd.read_parquet(liana_run_dir / resource_file)
                audit = {
                    "actual_proportion": float(row.actual_proportion),
                    "resource_rows": int(row.resource_rows),
                    "replaced_rows": int(row.resource_replaced_rows),
                }
            top, elapsed = _run_method(
                variant_data,
                variant_resource,
                top_n=top_n,
            )
            top_tables.append(
                _tag(
                    top,
                    kind=spec.kind,
                    proportion=spec.proportion,
                    replicate=spec.replicate,
                    seed=spec.seed,
                )
            )
            audit_rows.append(
                {
                    "perturbation": spec.kind,
                    "proportion": spec.proportion,
                    "replicate": spec.replicate,
                    "seed": spec.seed,
                    "actual_proportion": audit["actual_proportion"],
                    "cells_before": audit.get("cells_before", audit.get("cells")),
                    "cells_after": audit.get("cells_after", audit.get("cells")),
                    "resource_rows": audit.get("resource_rows"),
                    "resource_replaced_rows": audit.get("replaced_rows"),
                    "elapsed_seconds": elapsed,
                }
            )
            pd.concat(top_tables, ignore_index=True).to_parquet(
                output_dir / "top_predictions.partial.parquet", index=False
            )
    tops = pd.concat(top_tables, ignore_index=True)
    tops_path = output_dir / "top_predictions.parquet"
    audit_path = output_dir / "design.tsv"
    tops.to_parquet(tops_path, index=False)
    pd.DataFrame.from_records(audit_rows).to_csv(
        audit_path, sep="\t", index=False, na_rep=""
    )
    (output_dir / "top_predictions.partial.parquet").unlink(missing_ok=True)
    manifest: dict[str, object] = {
        "schema_version": "crychic-dimitrov-robustness-crychic-v1",
        "status": "complete",
        "scope": "static single-sample availability; not differential comm_strength",
        "input": {
            "filename": input_h5ad.name,
            "sha256": sha256_file(input_h5ad),
            "shape": [int(data.n_obs), int(data.n_vars)],
        },
        "resource": {
            "filename": baseline_resource.name,
            "sha256": sha256_file(baseline_resource),
            "rows": len(resource),
        },
        "protocol": {
            "baseline_only": baseline_only,
            "top_n": top_n,
            "tie_policy": "exact top_n with deterministic edge-key tie break",
            "base_seed": base_seed,
            "shared_liana_design": None
            if liana_run_dir is None
            else str(liana_run_dir.resolve()),
        },
        "versions": {
            name: _version(name)
            for name in ("CRYCHIC", "anndata", "numpy", "pandas", "scipy")
        },
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            baseline_path.name: sha256_file(baseline_path),
            tops_path.name: {"sha256": sha256_file(tops_path), "rows": len(tops)},
            audit_path.name: sha256_file(audit_path),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("baseline_resource", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--liana-run-dir", type=Path)
    parser.add_argument("--top-n", type=int, default=250)
    parser.add_argument("--base-seed", type=int, default=20260717)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.input_h5ad,
        args.baseline_resource,
        args.output_dir,
        liana_run_dir=args.liana_run_dir,
        top_n=args.top_n,
        base_seed=args.base_seed,
        baseline_only=args.baseline_only,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
