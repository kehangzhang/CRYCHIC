"""Generate BRCA-shaped paired 2x2 fixtures with planted CCC effects."""

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

SCHEMA_VERSION = "crychic-brca-semisynthetic-fixture-v2"
CONDITIONS = ("PreE", "OnE", "PreNE", "OnNE")
PAIRWISE_CONTRASTS = {
    "E": {"target": "OnE", "reference": "PreE"},
    "NE": {"target": "OnNE", "reference": "PreNE"},
}
EVENT_CLASSES = (
    "positive_did",
    "negative_did",
    "time_main_only",
    "expansion_main_only",
)
EVENT_KEYS = ("sender", "receiver", "interaction_id", "ligand", "receptor")
_CONDITIONS_BY_CLASS = {
    "positive_did": ("OnE",),
    "negative_did": ("OnNE",),
    "time_main_only": ("OnE", "OnNE"),
    "expansion_main_only": ("PreE", "OnE"),
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _portable_string_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.Index(
        result.index.astype(str).to_numpy(dtype=object),
        name=result.index.name,
    )
    for column in result.columns:
        if isinstance(result[column].dtype, pd.StringDtype | pd.CategoricalDtype):
            result[column] = result[column].astype(str).to_numpy(dtype=object)
    return result


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


def _counts_matrix(adata: ad.AnnData) -> sparse.csr_matrix:
    if "counts" not in adata.layers:
        raise ValueError("BRCA source must retain layers['counts']")
    counts = sparse.csr_matrix(adata.layers["counts"], dtype=np.int32, copy=True)
    if (
        np.any(counts.data < 0)
        or np.any(~np.isfinite(counts.data))
        or not np.array_equal(counts.data, np.round(counts.data))
    ):
        raise ValueError("counts must be finite non-negative integers")
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    return counts


def _validate_source(source: ad.AnnData) -> tuple[str, ...]:
    required = {
        "sample_id",
        "subject_id",
        "timepoint",
        "expansion",
        "condition",
        "cell_type",
    }
    missing = required.difference(source.obs.columns)
    if missing:
        raise ValueError(f"BRCA metadata is missing: {sorted(missing)}")
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("BRCA cell and gene identifiers must be unique")
    metadata = source.obs.loc[:, sorted(required)]
    if metadata.isna().any().any():
        raise ValueError("BRCA design metadata must be complete")
    metadata = metadata.astype(str)
    if set(metadata["condition"]) != set(CONDITIONS):
        raise ValueError("condition must contain PreE, OnE, PreNE, and OnNE")
    expected_condition = metadata["timepoint"] + metadata["expansion"]
    if not expected_condition.eq(metadata["condition"]).all():
        raise ValueError("condition does not equal timepoint + expansion")
    design = metadata[
        ["sample_id", "subject_id", "timepoint", "expansion", "condition"]
    ].drop_duplicates()
    if design.duplicated("sample_id", keep=False).any():
        raise ValueError("one sample maps to multiple design cells")
    if design[["subject_id", "expansion"]].drop_duplicates().duplicated(
        "subject_id", keep=False
    ).any():
        raise ValueError("one subject maps to multiple expansion groups")
    support = design.groupby(
        ["subject_id", "timepoint"], observed=True, sort=False
    ).size()
    if len(support) != 2 * design["subject_id"].nunique() or not support.eq(1).all():
        raise ValueError("every subject must have exactly one Pre and one On sample")
    _counts_matrix(source)
    cell_types = tuple(sorted(metadata["cell_type"].unique()))
    if len(cell_types) < 2:
        raise ValueError("semi-synthetic CCC requires at least two cell types")
    return cell_types


def _eligible_subjects(
    obs: pd.DataFrame,
    cell_types: Sequence[str],
    *,
    minimum_cells: int,
) -> tuple[str, ...]:
    counts = (
        obs.assign(
            subject_id=obs["subject_id"].astype(str),
            timepoint=obs["timepoint"].astype(str),
            cell_type=obs["cell_type"].astype(str),
        )
        .groupby(
            ["subject_id", "timepoint", "cell_type"],
            observed=True,
            sort=False,
        )
        .size()
        .unstack(["timepoint", "cell_type"], fill_value=0)
    )
    expected = pd.MultiIndex.from_product(
        [("Pre", "On"), tuple(cell_types)], names=["timepoint", "cell_type"]
    )
    counts = counts.reindex(columns=expected, fill_value=0)
    selected = tuple(sorted(counts.index[(counts >= minimum_cells).all(axis=1)]))
    design = obs.loc[
        obs["subject_id"].astype(str).isin(selected),
        ["subject_id", "expansion"],
    ].drop_duplicates()
    group_counts = design["expansion"].astype(str).value_counts()
    if any(int(group_counts.get(group, 0)) < 4 for group in ("E", "NE")):
        raise ValueError("eligible BRCA subset requires at least four subjects per arm")
    return selected


def _downsample_indices(
    obs: pd.DataFrame,
    subjects: Sequence[str],
    *,
    cap_per_sample_cell_type: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eligible = obs["subject_id"].astype(str).isin(subjects).to_numpy()
    positions = np.flatnonzero(eligible)
    selected: list[np.ndarray] = []
    grouped = obs.iloc[positions].groupby(
        ["sample_id", "cell_type"], observed=True, sort=True
    )
    for _, group in grouped:
        group_positions = np.asarray(
            [obs.index.get_loc(label) for label in group.index], dtype=np.int64
        )
        if len(group_positions) > cap_per_sample_cell_type:
            group_positions = np.sort(
                rng.choice(
                    group_positions,
                    size=cap_per_sample_cell_type,
                    replace=False,
                )
            )
        selected.append(group_positions)
    if not selected:
        raise ValueError("cell downsampling selected no observations")
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def _cell_type_expression(
    counts: sparse.csr_matrix,
    obs: pd.DataFrame,
    gene_indices: Sequence[int],
    cell_types: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    selected = counts[:, np.asarray(gene_indices, dtype=np.int64)]
    means: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
    labels = obs["cell_type"].astype(str).to_numpy()
    for cell_type in cell_types:
        mask = labels == cell_type
        if not mask.any():
            raise ValueError(f"cell type has no eligible cells: {cell_type}")
        group = selected[mask]
        means.append(np.asarray(group.mean(axis=0)).ravel())
        fractions.append(np.asarray(group.getnnz(axis=0), dtype=float) / group.shape[0])
    return np.vstack(means), np.vstack(fractions)


def _select_resource(
    resource: pd.DataFrame,
    source: ad.AnnData,
    eligible_cells: np.ndarray,
    cell_types: Sequence[str],
    eligible_target_ligands: set[str],
    *,
    n_interactions: int,
) -> pd.DataFrame:
    required = {"harmonized_interaction_id", "ligand", "receptor"}
    missing = required.difference(resource.columns)
    if missing or resource.empty:
        raise ValueError(f"H-common resource is invalid: missing={sorted(missing)}")
    genes = set(map(str, source.var_names))
    candidate = resource.loc[
        resource["ligand"].astype(str).isin(genes & eligible_target_ligands)
        & resource["receptor"].astype(str).isin(genes)
    ].copy()
    candidate = candidate.drop_duplicates(["ligand", "receptor"], keep="first")
    lr_genes = tuple(
        sorted(
            set(candidate["ligand"].astype(str))
            | set(candidate["receptor"].astype(str))
        )
    )
    gene_index = {str(gene): index for index, gene in enumerate(source.var_names)}
    indices = [gene_index[gene] for gene in lr_genes]
    local_index = {gene: index for index, gene in enumerate(lr_genes)}
    counts = _counts_matrix(source)[eligible_cells]
    obs = source.obs.iloc[eligible_cells]
    means, fractions = _cell_type_expression(
        counts, obs, indices, cell_types
    )
    records: list[dict[str, Any]] = []
    for row in candidate.to_dict(orient="records"):
        ligand = str(row["ligand"])
        receptor = str(row["receptor"])
        ligand_values = means[:, local_index[ligand]]
        receptor_values = means[:, local_index[receptor]]
        ligand_fraction = fractions[:, local_index[ligand]]
        receptor_fraction = fractions[:, local_index[receptor]]
        sender_index = int(np.argmax(ligand_values))
        receiver_index = int(np.argmax(receptor_values))
        quality = float(
            min(ligand_fraction[sender_index], receptor_fraction[receiver_index])
            * np.sqrt(
                max(ligand_values[sender_index], 0.0)
                * max(receptor_values[receiver_index], 0.0)
            )
        )
        records.append(
            {
                **row,
                "selected_sender": str(cell_types[sender_index]),
                "selected_receiver": str(cell_types[receiver_index]),
                "selection_quality": quality,
            }
        )
    ranked = pd.DataFrame.from_records(records).sort_values(
        ["selection_quality", "harmonized_interaction_id"],
        ascending=[False, True],
        kind="stable",
    )
    selected_rows: list[pd.Series] = []
    used_genes: set[str] = set()
    for _, row in ranked.iterrows():
        ligand = str(row["ligand"])
        receptor = str(row["receptor"])
        if ligand in used_genes or receptor in used_genes or ligand == receptor:
            continue
        if float(row["selection_quality"]) <= 0.0:
            continue
        selected_rows.append(row)
        used_genes.update((ligand, receptor))
        if len(selected_rows) == n_interactions:
            break
    if len(selected_rows) != n_interactions:
        raise ValueError(
            f"only {len(selected_rows)} disjoint observable LR pairs were available"
        )
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _target_map(
    prior: pd.DataFrame,
    ligands: Sequence[str],
    source_genes: set[str],
    *,
    targets_per_ligand: int,
) -> dict[str, tuple[str, ...]]:
    required = {"ligand", "target", "weight"}
    missing = required.difference(prior.columns)
    if missing:
        raise ValueError(f"target prior is incomplete: missing={sorted(missing)}")
    selected = prior.loc[
        prior["ligand"].astype(str).isin(ligands)
        & prior["target"].astype(str).isin(source_genes)
        & pd.to_numeric(prior["weight"], errors="coerce").gt(0.0)
    ].copy()
    selected["ligand"] = selected["ligand"].astype(str)
    selected["target"] = selected["target"].astype(str)
    sort_columns = [
        "ligand",
        *(("rank",) if "rank" in selected else ()),
        "target",
    ]
    selected = selected.sort_values(sort_columns, kind="stable")
    result = {
        ligand: tuple(
            selected.loc[selected["ligand"].eq(ligand), "target"]
            .drop_duplicates()
            .head(targets_per_ligand)
        )
        for ligand in ligands
    }
    missing_ligands = [ligand for ligand, targets in result.items() if not targets]
    if missing_ligands:
        raise ValueError(f"selected ligands lack target overlap: {missing_ligands}")
    return result


def _event_plan(
    resource: pd.DataFrame,
    *,
    seed: int,
    events_per_class: int,
) -> pd.DataFrame:
    required = events_per_class * len(EVENT_CLASSES)
    if required > len(resource):
        raise ValueError("resource is too small for the requested planted event grid")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(resource))
    classes = np.full(len(resource), "no_effect", dtype=object)
    offset = 0
    for event_class in EVENT_CLASSES:
        positions = order[offset : offset + events_per_class]
        classes[positions] = event_class
        offset += events_per_class
    plan = resource.loc[
        :,
        [
            "harmonized_interaction_id",
            "ligand",
            "receptor",
            "selected_sender",
            "selected_receiver",
            "selection_quality",
        ],
    ].copy()
    plan["event_class"] = classes
    plan["truth_label"] = plan["event_class"].isin(
        ("positive_did", "negative_did")
    ).astype(int)
    plan["truth_direction"] = plan["event_class"].map(
        {"positive_did": 1, "negative_did": -1}
    ).fillna(0).astype(int)
    return plan


def _injected_counts(
    counts: sparse.csr_matrix,
    obs: pd.DataFrame,
    genes: Sequence[str],
    plan: pd.DataFrame,
    targets: Mapping[str, Sequence[str]],
    *,
    ligand_add: int,
    receptor_add: int,
    target_add: int,
) -> sparse.csr_matrix:
    gene_index = {gene: index for index, gene in enumerate(genes)}
    condition = obs["condition"].astype(str).to_numpy()
    cell_type = obs["cell_type"].astype(str).to_numpy()
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []

    def add(rows: np.ndarray, selected_genes: Sequence[str], value: int) -> None:
        columns = np.asarray([gene_index[gene] for gene in selected_genes], dtype=int)
        if not len(rows) or not len(columns):
            return
        row_parts.append(np.repeat(rows, len(columns)))
        column_parts.append(np.tile(columns, len(rows)))
        value_parts.append(np.full(len(rows) * len(columns), value, dtype=np.int32))

    for row in plan.itertuples(index=False):
        event_class = str(row.event_class)
        if event_class == "no_effect":
            continue
        conditions = _CONDITIONS_BY_CLASS[event_class]
        sender_rows = np.flatnonzero(
            np.isin(condition, conditions) & (cell_type == str(row.selected_sender))
        )
        receiver_rows = np.flatnonzero(
            np.isin(condition, conditions) & (cell_type == str(row.selected_receiver))
        )
        ligand = str(row.ligand)
        receptor = str(row.receptor)
        target_genes = tuple(
            gene
            for gene in targets[ligand]
            if gene not in {ligand, receptor}
        )
        add(sender_rows, (ligand,), ligand_add)
        add(receiver_rows, (receptor,), receptor_add)
        add(receiver_rows, target_genes, target_add)

    if not row_parts:
        return counts.copy()
    additions = sparse.coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=counts.shape,
        dtype=np.int32,
    ).tocsr()
    result = (counts + additions).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _normalized(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    library_size = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(library_size <= 0):
        raise ValueError("every retained cell must have positive library size")
    result = counts.astype(np.float64).multiply((1.0e4 / library_size)[:, None])
    result = sparse.csr_matrix(result)
    result.data = np.log1p(result.data)
    return result


def _truth_table(
    plan: pd.DataFrame,
    resource: pd.DataFrame,
    cell_types: Sequence[str],
    *,
    dataset_id: str,
    seed: int,
) -> pd.DataFrame:
    plan_by_interaction = plan.set_index("harmonized_interaction_id")
    records: list[dict[str, Any]] = []
    for sender in cell_types:
        for receiver in cell_types:
            for row in resource.itertuples(index=False):
                interaction_id = str(row.harmonized_interaction_id)
                planted = plan_by_interaction.loc[interaction_id]
                exact = (
                    sender == str(planted["selected_sender"])
                    and receiver == str(planted["selected_receiver"])
                )
                event_class = str(planted["event_class"]) if exact else "no_effect"
                direction = (
                    int(planted["truth_direction"])
                    if exact
                    else 0
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "dataset_id": dataset_id,
                        "seed": int(seed),
                        "sender": sender,
                        "receiver": receiver,
                        "interaction_id": interaction_id,
                        "ligand": str(row.ligand),
                        "receptor": str(row.receptor),
                        "event_class": event_class,
                        "truth_label": int(direction != 0),
                        "truth_direction": direction,
                        "planted_main_effect_control": int(
                            event_class in {"time_main_only", "expansion_main_only"}
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _dataset_spec(
    dataset_id: str,
    input_path: Path,
    input_sha256: str,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "input": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "output_name": dataset_id,
        "input_mode": "counts",
        "lr_resource": "brca_semisynthetic_hcommon",
        "target_prior": "nichenet_human",
        "benchmark_scope": "paired_brca_shaped_2x2_semisynthetic_sample_unit",
        "config": {
            "context_keys": ["timepoint", "expansion"],
            "counts_layer": "counts",
            "sample_key": "sample_id",
            "subject_key": "analysis_subject_id",
            "cell_type_key": "cell_type",
            "species": "human",
            "gene_namespace": "hgnc_symbol",
            "design": "~ timepoint * expansion",
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


def _write_pairwise_inputs(
    fixture: ad.AnnData,
    dataset_dir: Path,
    staged_root: Path,
    *,
    dataset_id: str,
) -> dict[str, dict[str, Any]]:
    """Write the two native pairwise inputs required by scSeqCommDiff."""

    pairwise_dir = dataset_dir / "pairwise"
    pairwise_dir.mkdir()
    records: dict[str, dict[str, Any]] = {}
    for expansion, contrast in PAIRWISE_CONTRASTS.items():
        mask = fixture.obs["expansion"].astype(str).eq(expansion).to_numpy()
        subset = fixture[mask].copy()
        observed = set(subset.obs["condition"].astype(str))
        expected = {contrast["target"], contrast["reference"]}
        if observed != expected:
            raise RuntimeError(
                f"{expansion} pairwise input has conditions {sorted(observed)}"
            )
        subset.uns["semisynthetic_contract"] = {
            **dict(subset.uns["semisynthetic_contract"]),
            "native_pairwise_expansion": expansion,
            "native_pairwise_target": contrast["target"],
            "native_pairwise_reference": contrast["reference"],
            "native_pairwise_methods_pairing_support": False,
        }
        filename = f"{dataset_id}.{expansion}_On_vs_Pre.h5ad"
        output_path = pairwise_dir / filename
        subset.write_h5ad(output_path, compression="gzip")
        design = subset.obs[
            ["sample_id", "subject_id", "condition", "cell_type"]
        ].astype(str).drop_duplicates()
        sample_design = design[
            ["sample_id", "subject_id", "condition"]
        ].drop_duplicates()
        if sample_design.duplicated("sample_id", keep=False).any():
            raise RuntimeError("one pairwise sample maps to multiple design rows")
        units = sample_design.groupby("condition", observed=True).size().to_dict()
        record = {
            "schema_version": "crychic-brca-semisynthetic-pairwise-input-v1",
            "dataset_id": f"{dataset_id}__{expansion}_On_vs_Pre",
            "parent_dataset_id": dataset_id,
            "contrast": {
                "expansion": expansion,
                "target": contrast["target"],
                "reference": contrast["reference"],
                "sample_unit_key": "sample_id",
                "pairing": "subject_id",
                "native_pairwise_methods_pairing_support": False,
            },
            "dimensions": {
                "n_cells": int(subset.n_obs),
                "n_genes": int(subset.n_vars),
                "n_samples": int(sample_design["sample_id"].nunique()),
                "n_subjects": int(sample_design["subject_id"].nunique()),
                "n_cell_types": int(design["cell_type"].nunique()),
                "units_by_condition": {
                    str(key): int(value) for key, value in units.items()
                },
            },
            "output": {
                "filename": filename,
                "path": str(output_path.relative_to(staged_root)),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            },
        }
        manifest_path = (
            pairwise_dir
            / f"{dataset_id}.{expansion}_On_vs_Pre.manifest.json"
        )
        _write_json(manifest_path, record)
        record["manifest"] = {
            "path": str(manifest_path.relative_to(staged_root)),
            "sha256": sha256_file(manifest_path),
        }
        records[expansion] = record
    return records


def generate(
    input_h5ad: Path,
    resource_path: Path,
    target_prior_path: Path,
    output_dir: Path,
    *,
    seeds: Sequence[int],
    minimum_source_cells: int = 20,
    cap_per_sample_cell_type: int = 40,
    n_interactions: int = 30,
    events_per_class: int = 4,
    targets_per_ligand: int = 12,
    background_genes: int = 200,
    ligand_add: int = 6,
    receptor_add: int = 6,
    target_add: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate checksum-bound planted 2x2 fixtures from a real BRCA background."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if minimum_source_cells < 10 or cap_per_sample_cell_type < minimum_source_cells:
        raise ValueError("cell cap must be at least the source-cell minimum >= 10")
    if n_interactions < events_per_class * len(EVENT_CLASSES):
        raise ValueError("interaction axis cannot hold the planted event grid")
    positive_settings = (
        targets_per_ligand,
        background_genes,
        ligand_add,
        receptor_add,
        target_add,
    )
    if min(positive_settings) < 1:
        raise ValueError("gene and injection settings must be positive")

    input_h5ad = input_h5ad.expanduser().resolve()
    resource_path = resource_path.expanduser().resolve()
    target_prior_path = target_prior_path.expanduser().resolve()
    for path in (input_h5ad, resource_path, target_prior_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    source: ad.AnnData | None = None
    published = False
    try:
        source = ad.read_h5ad(input_h5ad)
        cell_types = _validate_source(source)
        eligible_subjects = _eligible_subjects(
            source.obs,
            cell_types,
            minimum_cells=minimum_source_cells,
        )
        eligible_mask = source.obs["subject_id"].astype(str).isin(eligible_subjects)
        eligible_cells = np.flatnonzero(eligible_mask.to_numpy())
        source_genes = set(map(str, source.var_names))
        prior = pd.read_parquet(target_prior_path)
        prior_overlap = prior.loc[
            prior["target"].astype(str).isin(source_genes)
            & pd.to_numeric(prior["weight"], errors="coerce").gt(0.0)
        ]
        target_support = prior_overlap.groupby(
            prior_overlap["ligand"].astype(str), observed=True
        )["target"].nunique()
        eligible_target_ligands = set(
            target_support.index[target_support >= targets_per_ligand]
        )
        source_resource = pd.read_csv(resource_path, sep="\t")
        selected = _select_resource(
            source_resource,
            source,
            eligible_cells,
            cell_types,
            eligible_target_ligands,
            n_interactions=n_interactions,
        )
        target_map = _target_map(
            prior,
            tuple(selected["ligand"].astype(str)),
            source_genes,
            targets_per_ligand=targets_per_ligand,
        )
        required_genes = set(selected["ligand"].astype(str)) | set(
            selected["receptor"].astype(str)
        )
        required_genes.update(
            gene for targets in target_map.values() for gene in targets
        )
        eligible_counts = _counts_matrix(source)[eligible_cells]
        gene_means = np.asarray(eligible_counts.mean(axis=0)).ravel()
        ranked_background = [
            str(source.var_names[index])
            for index in np.argsort(-gene_means, kind="stable")
            if str(source.var_names[index]) not in required_genes
        ][:background_genes]
        selected_genes_set = required_genes | set(ranked_background)
        selected_genes = tuple(
            str(gene) for gene in source.var_names if str(gene) in selected_genes_set
        )
        gene_positions = source.var_names.get_indexer(selected_genes)
        if np.any(gene_positions < 0):
            raise RuntimeError("selected gene axis is not contained in the source")

        resource_dir = staged / "resource"
        inputs_dir = staged / "inputs"
        resource_dir.mkdir()
        inputs_dir.mkdir()
        frozen_resource = selected.drop(
            columns=["selected_sender", "selected_receiver", "selection_quality"]
        )
        resource_output = resource_dir / "harmonized_lr.tsv"
        frozen_resource.to_csv(
            resource_output, sep="\t", index=False, lineterminator="\n"
        )
        source_manifest_path = resource_path.with_name("manifest.json")
        source_manifest: dict[str, Any] = {}
        if source_manifest_path.is_file():
            raw_manifest: object = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(raw_manifest, dict):
                raise ValueError("source resource manifest must be an object")
            source_manifest = raw_manifest
        resource_manifest = {
            "schema_version": "crychic-harmonized-lr-v1",
            "resource_id": "crychic_brca_semisynthetic_hcommon",
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
                    "CellChat, CellPhoneDB, LIANA, scSeqCommDiff, and "
                    "ConnectomeDB source resources; see the checksum-bound "
                    "source manifest."
                ],
            ),
            "construction": {
                "rule": "disjoint high-observability BRCA subset with planted truth",
                "source_rows": len(source_resource),
                "retained_interactions": len(frozen_resource),
            },
            "source": {
                "filename": resource_path.name,
                "sha256": sha256_file(resource_path),
                "manifest_sha256": (
                    sha256_file(source_manifest_path)
                    if source_manifest_path.is_file()
                    else None
                ),
            },
            "payload": {
                "filename": resource_output.name,
                "rows": len(frozen_resource),
                "bytes": resource_output.stat().st_size,
                "sha256": sha256_file(resource_output),
            },
        }
        resource_manifest_path = resource_dir / "manifest.json"
        _write_json(resource_manifest_path, resource_manifest)

        truth_parts: list[pd.DataFrame] = []
        plan_parts: list[pd.DataFrame] = []
        records: list[dict[str, Any]] = []
        datasets: dict[str, Any] = {}
        source_counts = _counts_matrix(source)
        for replicate, seed in enumerate(seeds, start=1):
            dataset_id = f"brca_semisynthetic_r{replicate:02d}"
            cell_indices = _downsample_indices(
                source.obs,
                eligible_subjects,
                cap_per_sample_cell_type=cap_per_sample_cell_type,
                seed=int(seed),
            )
            obs = source.obs.iloc[cell_indices].copy()
            obs["analysis_subject_id"] = obs["sample_id"].astype(str)
            counts = source_counts[cell_indices][:, gene_positions].tocsr()
            plan = _event_plan(
                selected,
                seed=int(seed),
                events_per_class=events_per_class,
            )
            injected = _injected_counts(
                counts,
                obs,
                selected_genes,
                plan,
                target_map,
                ligand_add=ligand_add,
                receptor_add=receptor_add,
                target_add=target_add,
            )
            fixture = ad.AnnData(
                X=_normalized(injected),
                obs=_portable_string_frame(obs),
                var=_portable_string_frame(source.var.iloc[gene_positions]),
            )
            fixture.layers["counts"] = injected
            fixture.uns["semisynthetic_contract"] = {
                "schema_version": SCHEMA_VERSION,
                "source_dataset": "brca_anti_pd1",
                "true_pairing_key": "subject_id",
                "fit_unit_key": "analysis_subject_id",
                "primary_estimand": "(On-Pre in E) - (On-Pre in NE)",
                "structural_absence_is_missing": True,
            }
            dataset_dir = inputs_dir / dataset_id
            dataset_dir.mkdir()
            h5ad_path = dataset_dir / f"{dataset_id}.h5ad"
            fixture.write_h5ad(h5ad_path, compression="gzip")
            h5ad_sha = sha256_file(h5ad_path)
            pairwise_inputs = _write_pairwise_inputs(
                fixture,
                dataset_dir,
                staged,
                dataset_id=dataset_id,
            )
            plan_output = plan.rename(
                columns={"harmonized_interaction_id": "interaction_id"}
            ).copy()
            plan_output.insert(0, "dataset_id", dataset_id)
            plan_output.insert(1, "seed", int(seed))
            plan_parts.append(plan_output)
            truth_parts.append(
                _truth_table(
                    plan,
                    frozen_resource,
                    cell_types,
                    dataset_id=dataset_id,
                    seed=int(seed),
                )
            )
            design = obs[
                ["sample_id", "subject_id", "timepoint", "expansion", "condition"]
            ].drop_duplicates()
            record = {
                "dataset_id": dataset_id,
                "seed": int(seed),
                "h5ad": str(h5ad_path.relative_to(staged)),
                "sha256": h5ad_sha,
                "shape": [int(fixture.n_obs), int(fixture.n_vars)],
                "n_samples": int(design["sample_id"].nunique()),
                "n_subjects": int(design["subject_id"].nunique()),
                "subjects_by_expansion": {
                    str(key): int(value)
                    for key, value in design[
                        ["subject_id", "expansion"]
                    ].drop_duplicates()["expansion"].value_counts().items()
                },
                "pairwise_inputs": pairwise_inputs,
            }
            _write_json(dataset_dir / "manifest.json", record)
            records.append(record)
            datasets[dataset_id] = _dataset_spec(
                dataset_id,
                output_dir / h5ad_path.relative_to(staged),
                h5ad_sha,
                seed=int(seed),
            )

        truth = pd.concat(truth_parts, ignore_index=True)
        plans = pd.concat(plan_parts, ignore_index=True)
        truth_path = staged / "event_truth.tsv"
        plan_path = staged / "injection_plan.tsv"
        truth.to_csv(truth_path, sep="\t", index=False, lineterminator="\n")
        plans.to_csv(plan_path, sep="\t", index=False, lineterminator="\n")
        config = {
            "schema_version": "canonical-v0.1",
            "database_root": str(
                (Path(__file__).resolve().parents[2] / "../databases").resolve()
            ),
            "output_root": str((output_dir.parent / "runs/crychic").resolve()),
            "resources": {
                "brca_semisynthetic_hcommon": {
                    "adapter": "harmonized_fixture",
                    "manifest": str(
                        (
                            output_dir
                            / resource_manifest_path.relative_to(staged)
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
            "datasets": datasets,
        }
        config_path = staged / "crychic_config.json"
        _write_json(config_path, config)
        subject_design = source.obs.loc[
            source.obs["subject_id"].astype(str).isin(eligible_subjects),
            ["subject_id", "expansion"],
        ].drop_duplicates()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "source": {
                "filename": input_h5ad.name,
                "sha256": sha256_file(input_h5ad),
            },
            "design": {
                "primary_estimand": "(On-Pre in E) - (On-Pre in NE)",
                "true_pairing_key": "subject_id",
                "fit_unit_key": "analysis_subject_id",
                "eligible_subjects": list(eligible_subjects),
                "subjects_by_expansion": {
                    str(key): int(value)
                    for key, value in subject_design["expansion"].value_counts().items()
                },
                "cell_types": list(cell_types),
                "minimum_source_cells": minimum_source_cells,
                "cap_per_sample_cell_type": cap_per_sample_cell_type,
            },
            "injection": {
                "events_per_class": events_per_class,
                "classes": list(EVENT_CLASSES),
                "ligand_add": ligand_add,
                "receptor_add": receptor_add,
                "target_add": target_add,
                "targets_per_ligand": targets_per_ligand,
                "background_genes": background_genes,
            },
            "seeds": list(map(int, seeds)),
            "records": records,
            "resource": {
                "filename": str(resource_output.relative_to(staged)),
                "manifest": str(resource_manifest_path.relative_to(staged)),
                "rows": len(frozen_resource),
                "sha256": sha256_file(resource_output),
            },
            "outputs": {
                "truth": {
                    "filename": truth_path.name,
                    "rows": len(truth),
                    "sha256": sha256_file(truth_path),
                },
                "injection_plan": {
                    "filename": plan_path.name,
                    "rows": len(plans),
                    "sha256": sha256_file(plan_path),
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
        if source is not None and source.isbacked:
            source.file.close()
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
    parser.add_argument("--input-h5ad", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--target-prior", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=_parse_seeds, default=(20260723,))
    parser.add_argument("--minimum-source-cells", type=int, default=20)
    parser.add_argument("--cap-per-sample-cell-type", type=int, default=40)
    parser.add_argument("--n-interactions", type=int, default=30)
    parser.add_argument("--events-per-class", type=int, default=4)
    parser.add_argument("--targets-per-ligand", type=int, default=12)
    parser.add_argument("--background-genes", type=int, default=200)
    parser.add_argument("--ligand-add", type=int, default=6)
    parser.add_argument("--receptor-add", type=int, default=6)
    parser.add_argument("--target-add", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = generate(
        args.input_h5ad,
        args.resource,
        args.target_prior,
        args.output_dir,
        seeds=args.seeds,
        minimum_source_cells=args.minimum_source_cells,
        cap_per_sample_cell_type=args.cap_per_sample_cell_type,
        n_interactions=args.n_interactions,
        events_per_class=args.events_per_class,
        targets_per_ligand=args.targets_per_ligand,
        background_genes=args.background_genes,
        ligand_add=args.ligand_add,
        receptor_add=args.receptor_add,
        target_add=args.target_add,
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
