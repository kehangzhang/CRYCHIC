"""Run current LIANA methods through the Dimitrov PBMC3k perturbations."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anndata as ad
import liana as li
import numpy as np
import pandas as pd

from benchmarks.literature.robustness import (
    PERTURBATION_KINDS,
    PerturbationSpec,
    build_design,
    replace_resource_interactions,
    reshuffle_labels,
    select_top_edges,
    subsample_cells,
)
from benchmarks.literature.run_cytokine_liana import _score_table
from benchmarks.openproblems.common import sha256_file, write_json

PAPER_METHOD_IDS = frozenset(
    {
        "cellchat_composite",
        "cellphonedb_composite",
        "connectome_specificity",
        "logfc_specificity",
        "natmi_specificity",
        "singlecellsignalr_lrscore",
        "liana_specificity_consensus",
    }
)


@contextmanager
def _exclusive_run_lock(output_dir: Path) -> Iterator[None]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.run.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another LIANA robustness process is using {output_dir}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _run_methods(
    data: ad.AnnData,
    resource: pd.DataFrame,
    *,
    label_key: str,
    n_perms: int,
    n_jobs: int,
    seed: int,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    started = time.perf_counter()
    np.random.seed(seed)
    rank = li.mt.rank_aggregate(
        data.copy(),
        groupby=label_key,
        resource=resource,
        expr_prop=0.1,
        min_cells=5,
        return_all_lrs=True,
        use_raw=False,
        n_perms=n_perms,
        seed=seed,
        n_jobs=n_jobs,
        inplace=False,
        verbose=False,
    )
    rank_elapsed = time.perf_counter() - started
    chat_started = time.perf_counter()
    np.random.seed(seed)
    chat = li.mt.cellchat(
        data.copy(),
        groupby=label_key,
        resource=resource,
        expr_prop=0.1,
        min_cells=5,
        return_all_lrs=True,
        use_raw=False,
        n_perms=n_perms,
        seed=seed,
        n_jobs=n_jobs,
        inplace=False,
        verbose=False,
    )
    chat_elapsed = time.perf_counter() - chat_started
    if not isinstance(rank, pd.DataFrame) or rank.empty:
        raise RuntimeError("LIANA rank aggregate returned no interactions")
    if not isinstance(chat, pd.DataFrame) or chat.empty:
        raise RuntimeError("LIANA CellChat returned no interactions")
    scores = _score_table(rank, chat)
    scores = scores.loc[scores["method_id"].isin(PAPER_METHOD_IDS)].rename(
        columns={"method_name": "method"}
    )
    top = select_top_edges(scores, top_n=top_n)
    return top, {
        "rank_aggregate_seconds": rank_elapsed,
        "cellchat_seconds": chat_elapsed,
    }


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


def _preserved_pairs(
    baseline: pd.DataFrame,
    external_baseline: Path | None,
) -> frozenset[tuple[str, str]]:
    tables = [baseline]
    if external_baseline is not None:
        external = pd.read_parquet(external_baseline)
        missing = {"ligand", "receptor"}.difference(external.columns)
        if missing:
            raise ValueError(
                f"external baseline top table is missing: {sorted(missing)}"
            )
        tables.append(external)
    combined = pd.concat(tables, ignore_index=True)
    return frozenset(
        combined.loc[:, ["ligand", "receptor"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )


def _audit_row(
    spec: PerturbationSpec,
    audit: dict[str, Any],
    elapsed: dict[str, float],
    *,
    resource_file: str | None,
) -> dict[str, Any]:
    return {
        "perturbation": spec.kind,
        "proportion": spec.proportion,
        "replicate": spec.replicate,
        "seed": spec.seed,
        "actual_proportion": audit["actual_proportion"],
        "cells_before": audit.get("cells_before", audit.get("cells")),
        "cells_after": audit.get("cells_after", audit.get("cells")),
        "mismatched_cells": audit.get("mismatched_cells"),
        "resource_rows": audit.get("resource_rows"),
        "resource_replaced_rows": audit.get("replaced_rows"),
        "resource_preserved_rows": audit.get("preserved_rows"),
        "resource_file": resource_file,
        **elapsed,
    }


def _resume_state(
    output_dir: Path,
    design: tuple[PerturbationSpec, ...],
    *,
    top_n: int,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], set[tuple[str, float, int, int]]]:
    top_path = output_dir / "top_predictions.partial.parquet"
    audit_path = output_dir / "design.partial.tsv"
    if not top_path.exists() or not audit_path.exists():
        raise FileNotFoundError(
            "resume requires top_predictions.partial.parquet and design.partial.tsv"
        )
    top = pd.read_parquet(top_path)
    audit = pd.read_csv(audit_path, sep="\t")
    key_columns = ["perturbation", "proportion", "replicate", "seed"]
    missing_top = set(key_columns).difference(top.columns)
    missing_audit = set(key_columns).difference(audit.columns)
    if missing_top or missing_audit:
        raise ValueError(
            "partial robustness state lacks design keys: "
            f"top={sorted(missing_top)}, audit={sorted(missing_audit)}"
        )
    if audit.duplicated(key_columns).any():
        raise ValueError("partial robustness audit has duplicate design rows")
    expected = {
        (spec.kind, float(spec.proportion), int(spec.replicate), int(spec.seed))
        for spec in design
    }
    completed: set[tuple[str, float, int, int]] = set()
    for row in audit.loc[audit["perturbation"].ne("baseline")].itertuples(index=False):
        key = (
            str(row.perturbation),
            float(row.proportion),
            int(row.replicate),
            int(row.seed),
        )
        if key not in expected:
            raise ValueError(
                f"partial robustness state has unexpected design row: {key}"
            )
        completed.add(key)
    baseline = top.loc[top["perturbation"].eq("baseline")]
    if baseline.empty or len(audit.loc[audit["perturbation"].eq("baseline")]) != 1:
        raise ValueError("partial robustness state must contain one baseline design")
    methods = frozenset(baseline["method"].astype(str))
    if not methods:
        raise ValueError("partial robustness baseline contains no methods")
    baseline_counts = baseline.groupby("method", observed=True).size()
    if not baseline_counts.eq(top_n).all():
        raise ValueError(
            "partial robustness baseline does not contain the requested "
            f"top {top_n} for every method"
        )
    top_keys = set()
    for key, group in top.loc[top["perturbation"].ne("baseline")].groupby(
        key_columns, sort=False, observed=True
    ):
        normalized = (str(key[0]), float(key[1]), int(key[2]), int(key[3]))
        if frozenset(group["method"].astype(str)) != methods:
            raise ValueError(
                f"partial robustness prediction methods are incomplete for {normalized}"
            )
        if not group.groupby("method", observed=True).size().eq(top_n).all():
            raise ValueError(
                f"partial robustness prediction rows are incomplete for {normalized}"
            )
        edge_keys = ["method", "source", "target", "ligand", "receptor"]
        if group.duplicated(edge_keys).any():
            raise ValueError(
                "partial robustness predictions contain duplicate edges for "
                f"{normalized}"
            )
        top_keys.add(normalized)
    if top_keys != completed:
        raise ValueError(
            "partial robustness predictions and audit disagree: "
            f"predictions_only={sorted(top_keys - completed)}, "
            f"audit_only={sorted(completed - top_keys)}"
        )
    return [top], audit.to_dict("records"), completed


def _run_unlocked(
    input_h5ad: Path,
    output_dir: Path,
    *,
    crychic_baseline_top: Path | None,
    label_key: str,
    top_n: int,
    n_perms: int,
    n_jobs: int,
    replicates: int,
    base_seed: int,
    overwrite: bool,
    resume: bool = False,
) -> dict[str, object]:
    """Run one baseline and all four paper perturbation series."""

    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    if output_dir.exists() and any(output_dir.iterdir()) and not (overwrite or resume):
        raise FileExistsError(f"LIANA robustness output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resource_dir = output_dir / "resource_perturbations"
    resource_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    data = ad.read_h5ad(input_h5ad)
    if label_key not in data.obs or "highly_variable" not in data.var:
        raise ValueError("prepared PBMC3k input lacks labels or HVG annotations")
    data.obs[label_key] = data.obs[label_key].astype("category")
    resource = (
        li.rs.select_resource("consensus")
        .loc[:, ["ligand", "receptor"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(["ligand", "receptor"], kind="stable", ignore_index=True)
    )
    baseline_path = output_dir / "baseline_top_predictions.parquet"
    baseline_resource_path = output_dir / "baseline_resource.parquet"
    design = build_design(replicates=replicates, base_seed=base_seed)
    if resume:
        if not baseline_resource_path.exists() or not baseline_path.exists():
            raise FileNotFoundError(
                "resume requires baseline_resource.parquet and "
                "baseline_top_predictions.parquet"
            )
        previous_resource = pd.read_parquet(baseline_resource_path)
        if not previous_resource.equals(resource):
            raise ValueError("current LIANA resource differs from partial run resource")
        top_tables, audit_rows, completed = _resume_state(
            output_dir, design, top_n=top_n
        )
        baseline = pd.read_parquet(baseline_path)
        partial_baseline = top_tables[0].loc[
            top_tables[0]["perturbation"].eq("baseline")
        ]
        comparison_columns = [
            "method",
            "source",
            "target",
            "ligand",
            "receptor",
            "rank",
        ]
        if (
            not baseline.loc[:, comparison_columns]
            .reset_index(drop=True)
            .equals(partial_baseline.loc[:, comparison_columns].reset_index(drop=True))
        ):
            raise ValueError("partial robustness baseline differs from baseline file")
    else:
        resource.to_parquet(baseline_resource_path, index=False)
        baseline, baseline_elapsed = _run_methods(
            data,
            resource,
            label_key=label_key,
            n_perms=n_perms,
            n_jobs=n_jobs,
            seed=base_seed,
            top_n=top_n,
        )
        baseline = _tag(
            baseline,
            kind="baseline",
            proportion=0.0,
            replicate=0,
            seed=base_seed,
        )
        baseline.to_parquet(baseline_path, index=False)
        top_tables = [baseline]
        audit_rows = [
            {
                "perturbation": "baseline",
                "proportion": 0.0,
                "replicate": 0,
                "seed": base_seed,
                "actual_proportion": 0.0,
                "cells_before": data.n_obs,
                "cells_after": data.n_obs,
                "mismatched_cells": 0,
                "resource_rows": len(resource),
                "resource_replaced_rows": 0,
                "resource_preserved_rows": 0,
                "resource_file": baseline_resource_path.name,
                **baseline_elapsed,
            }
        ]
        completed = set()
    preserved = _preserved_pairs(baseline, crychic_baseline_top)
    hvg_genes = tuple(
        map(str, data.var_names[data.var["highly_variable"].astype(bool)])
    )
    audit_rows[0]["resource_preserved_rows"] = len(preserved)
    for spec in design:
        key = (spec.kind, spec.proportion, spec.replicate, spec.seed)
        if key in completed:
            continue
        variant_data = data
        variant_resource = resource
        resource_file: str | None = None
        if spec.kind == "cell_subsampling":
            variant_data, audit = subsample_cells(
                data,
                label_key=label_key,
                proportion=spec.proportion,
                seed=spec.seed,
            )
        elif spec.kind == "label_reshuffling":
            variant_data, audit = reshuffle_labels(
                data,
                label_key=label_key,
                proportion=spec.proportion,
                seed=spec.seed,
            )
        else:
            selective = spec.kind == "resource_selective"
            variant_resource, audit = replace_resource_interactions(
                resource,
                genes=hvg_genes,
                proportion=spec.proportion,
                seed=spec.seed,
                preserved_pairs=preserved if selective else frozenset(),
            )
            filename = (
                f"{spec.kind}_p{round(100 * spec.proportion):02d}"
                f"_r{spec.replicate}.parquet"
            )
            path = resource_dir / filename
            variant_resource.to_parquet(path, index=False)
            resource_file = str(path.relative_to(output_dir))
        top, elapsed = _run_methods(
            variant_data,
            variant_resource,
            label_key=label_key,
            n_perms=n_perms,
            n_jobs=n_jobs,
            seed=spec.seed,
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
        audit_rows.append(_audit_row(spec, audit, elapsed, resource_file=resource_file))
        pd.concat(top_tables, ignore_index=True).to_parquet(
            output_dir / "top_predictions.partial.parquet", index=False
        )
        pd.DataFrame.from_records(audit_rows).to_csv(
            output_dir / "design.partial.tsv", sep="\t", index=False, na_rep=""
        )
    tops = pd.concat(top_tables, ignore_index=True)
    audit_table = pd.DataFrame.from_records(audit_rows)
    tops_path = output_dir / "top_predictions.parquet"
    design_path = output_dir / "design.tsv"
    tops.to_parquet(tops_path, index=False)
    audit_table.to_csv(design_path, sep="\t", index=False, na_rep="")
    partial_top = output_dir / "top_predictions.partial.parquet"
    partial_design = output_dir / "design.partial.tsv"
    partial_top.unlink(missing_ok=True)
    partial_design.unlink(missing_ok=True)
    manifest: dict[str, object] = {
        "schema_version": "crychic-dimitrov-robustness-liana-v1",
        "status": "complete",
        "input": {
            "filename": input_h5ad.name,
            "sha256": sha256_file(input_h5ad),
            "shape": [int(data.n_obs), int(data.n_vars)],
            "label_key": label_key,
        },
        "resource": {
            "id": "liana_consensus",
            "rows": len(resource),
            "baseline_sha256": sha256_file(baseline_resource_path),
            "selectively_preserved_lr_pairs": len(preserved),
            "external_crychic_baseline": None
            if crychic_baseline_top is None
            else {
                "filename": crychic_baseline_top.name,
                "sha256": sha256_file(crychic_baseline_top),
            },
        },
        "protocol": {
            "paper": "Dimitrov et al. Nature Communications 2022;13:3224",
            "perturbations": list(PERTURBATION_KINDS),
            "proportions": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            "replicates": replicates,
            "top_n": top_n,
            "tie_policy": "exact top_n with deterministic edge-key tie break",
            "base_seed": base_seed,
            "n_perms": n_perms,
            "resumed_from_completed_perturbations": len(completed),
        },
        "methods": sorted(tops["method"].unique()),
        "versions": {
            name: _version(name)
            for name in ("liana", "anndata", "scanpy", "numpy", "pandas")
        },
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            tops_path.name: {"sha256": sha256_file(tops_path), "rows": len(tops)},
            design_path.name: {
                "sha256": sha256_file(design_path),
                "rows": len(audit_table),
            },
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def run(
    input_h5ad: Path,
    output_dir: Path,
    *,
    crychic_baseline_top: Path | None,
    label_key: str,
    top_n: int,
    n_perms: int,
    n_jobs: int,
    replicates: int,
    base_seed: int,
    overwrite: bool,
    resume: bool = False,
) -> dict[str, object]:
    """Run one process at a time for a shared robustness output directory."""

    with _exclusive_run_lock(output_dir):
        return _run_unlocked(
            input_h5ad,
            output_dir,
            crychic_baseline_top=crychic_baseline_top,
            label_key=label_key,
            top_n=top_n,
            n_perms=n_perms,
            n_jobs=n_jobs,
            replicates=replicates,
            base_seed=base_seed,
            overwrite=overwrite,
            resume=resume,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--crychic-baseline-top", type=Path)
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--top-n", type=int, default=250)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260717)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        args.input_h5ad,
        args.output_dir,
        crychic_baseline_top=args.crychic_baseline_top,
        label_key=args.label_key,
        top_n=args.top_n,
        n_perms=args.n_perms,
        n_jobs=args.n_jobs,
        replicates=args.replicates,
        base_seed=args.base_seed,
        overwrite=args.overwrite,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
