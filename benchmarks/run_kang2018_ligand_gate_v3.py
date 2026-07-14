"""Run the compact Kang 2018 common-sender ligand-gate v3 diagnostic.

This is an exploratory real-data development benchmark. It deliberately keeps
the receiver-autonomous nuisance stage unresolved and therefore cannot certify
family-common scores or support a method-superiority claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import resource as process_resource
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

import anndata as ad
import numpy as np
import pandas as pd

from crychic import load_nichenet_target_prior
from crychic.attribution import PenaltyTuningSpec
from crychic.core import CrychicConfig, canonical_digest
from crychic.design import balanced_contrast
from crychic.resources import (
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
    load_cellchat_resource,
)
from crychic.sender import (
    ContrastCommonSenderParameters,
    interaction_ligand_contrast_gate,
)
from crychic.workflow import (
    AutonomousProgramUseScope,
    CrossFitArtifacts,
    CrossFitFoldArtifacts,
    CrossFitSpec,
    FoldTrainingSpec,
    run_subject_crossfit,
)

SCHEMA_VERSION = "crychic-kang2018-ligand-gate-v3-benchmark-v1"
CONFIG_SCHEMA_VERSION = "crychic-kang2018-ligand-gate-v3-config-v1"
EXPECTED_CONFIG_SHA256 = (
    "31f0001e8c5db4b2b68c17bc0953a310c684f9f596be500db72bb669fbbb48c5"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "benchmarks/configs/kang2018_ligand_gate_v3.json"
DEFAULT_WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DEFAULT_OUTPUT = Path("benchmark_work/kang2018_ligand_gate_v3.json")

SOURCE_PATHS = (
    "benchmarks/configs/kang2018_ligand_gate_v3.json",
    "benchmarks/run_kang2018_ligand_gate_v3.py",
    "src/crychic/attribution/tuning.py",
    "src/crychic/resources/cellchat.py",
    "src/crychic/resources/nichenet.py",
    "src/crychic/scoring/family_common.py",
    "src/crychic/sender/common.py",
    "src/crychic/sender/contracts.py",
    "src/crychic/workflow/crossfit.py",
    "src/crychic/workflow/training.py",
)

CLAIMS = {
    "literature_informed": True,
    "exploratory": True,
    "certifying": False,
    "biological_validation": False,
    "method_superiority": False,
    "default_switch_allowed": False,
}
SUBJECT_DIGEST_DOMAIN = "crychic-kang2018-subject-set-manifest-v1"
SUBJECT_DIGEST_SEMANTICS = (
    "deterministic_nonplaintext_identifier_for_alignment_not_anonymization"
)


class TargetLinkRecord(TypedDict):
    """One checksum-pinned NicheNet link selected by source rank."""

    target: str
    rank: int
    weight: float


def sha256_file(path: Path) -> str:
    """Hash one file without retaining its content."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_set_manifest(subject_ids: Iterable[object]) -> dict[str, object]:
    """Return a stable set identity without serializing plaintext subject IDs."""

    subjects = tuple(sorted({str(value).strip() for value in subject_ids}))
    if not subjects or any(not subject for subject in subjects):
        raise ValueError("subject set must contain non-empty identifiers")
    digest = hashlib.sha256()
    digest.update(SUBJECT_DIGEST_DOMAIN.encode("ascii"))
    for subject in subjects:
        digest.update(b"\0")
        digest.update(subject.encode("utf-8"))
    return {
        "n_subjects": len(subjects),
        "subject_set_digest": digest.hexdigest(),
        "digest_domain": SUBJECT_DIGEST_DOMAIN,
        "privacy_semantics": SUBJECT_DIGEST_SEMANTICS,
    }


def _portable_path(path: Path, *roots: Path) -> str:
    resolved = path.resolve()
    for root in roots:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.name


def verify_input_artifact(
    input_path: Path,
    *,
    dataset_config: Mapping[str, object],
    expected_gene_count: int,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Verify the complete H5AD and conversion lineage before analysis."""

    expected = _mapping(dataset_config["expected_h5ad"], field="expected_h5ad")
    expected_bytes = int(cast(int, expected["bytes"]))
    observed_bytes = input_path.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError("Kang H5AD byte size does not match the frozen input artifact")
    expected_sha256 = str(expected["sha256"])
    observed_sha256 = sha256_file(input_path)
    if observed_sha256 != expected_sha256:
        raise ValueError("Kang H5AD checksum does not match the frozen input artifact")

    expected_conversion = _mapping(
        expected["crychic_conversion"], field="expected_h5ad.crychic_conversion"
    )
    adata = ad.read_h5ad(input_path, backed="r")
    try:
        observed_conversion = _mapping(
            adata.uns.get("crychic_conversion"),
            field="adata.uns.crychic_conversion",
        )
        mismatches = {
            key: {
                "expected": str(expected_value),
                "observed": str(observed_conversion.get(key)),
            }
            for key, expected_value in expected_conversion.items()
            if str(observed_conversion.get(key)) != str(expected_value)
        }
        if mismatches:
            raise ValueError(
                "Kang H5AD crychic_conversion lineage does not match the frozen "
                f"input artifact: {sorted(mismatches)}"
            )
        if adata.n_vars != expected_gene_count:
            raise ValueError(
                "Kang H5AD does not contain the frozen full-transcriptome feature "
                f"count: {adata.n_vars} != {expected_gene_count}"
            )
        available_genes = tuple(map(str, adata.var_names))
        source_shape = [int(adata.n_obs), int(adata.n_vars)]
    finally:
        adata.file.close()
    return (
        {
            "expected_bytes": expected_bytes,
            "observed_bytes": observed_bytes,
            "expected_sha256": expected_sha256,
            "observed_sha256": observed_sha256,
            "checksum_verified": True,
            "conversion_lineage_verified": True,
            "verified_conversion": dict(expected_conversion),
            "source_shape": source_shape,
        },
        available_genes,
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return cast(Sequence[object], value)


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    values = tuple(str(item).strip() for item in _sequence(value, field=field))
    if (
        not values
        or any(not item for item in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{field} must contain unique non-empty strings")
    return values


def _float_value(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _int_value(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def load_benchmark_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Load and validate the frozen diagnostic configuration."""

    config_path = Path(path).resolve()
    observed_config_sha256 = sha256_file(config_path)
    if observed_config_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError(
            "Kang ligand-gate config checksum mismatch: the complete frozen "
            "benchmark policy changed"
        )
    config = dict(
        _mapping(json.loads(config_path.read_text(encoding="utf-8")), field="config")
    )
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported Kang ligand-gate benchmark config")

    dataset = _mapping(config.get("dataset"), field="dataset")
    if str(dataset.get("dataset_id")) != "GSE96583_Kang2018_batch2":
        raise ValueError("Kang benchmark dataset_id changed")
    expected_h5ad = _mapping(dataset.get("expected_h5ad"), field="expected_h5ad")
    if (
        expected_h5ad.get("sha256")
        != "5ee14c9df0a58e385828f9339dc93eff52fae7614218e82ed9c9a022804b30ac"
        or expected_h5ad.get("bytes") != 72865662
    ):
        raise ValueError("Kang H5AD identity changed from the frozen artifact")
    if _strings(dataset.get("cell_types"), field="dataset.cell_types") != (
        "CD14+ Monocytes",
        "CD8 T cells",
    ):
        raise ValueError("Kang diagnostic must freeze the two registered cell types")
    if dataset.get("expected_subject_count") != 8:
        raise ValueError("Kang diagnostic requires all eight paired donors")
    if (
        dataset.get("require_all_available_subjects") is not True
        or dataset.get("require_complete_pairing") is not True
    ):
        raise ValueError(
            "Kang diagnostic requires every available, completely paired donor"
        )
    conditions = _mapping(dataset.get("conditions"), field="dataset.conditions")
    if conditions != {"positive": "stim", "negative": "ctrl"}:
        raise ValueError("Kang diagnostic contrast must remain stim minus ctrl")

    definitions = _sequence(
        config.get("diagnostic_interactions"), field="diagnostic_interactions"
    )
    expected = (
        "CXCL10_CXCR3",
        "IFNB1_IFNAR1_IFNAR2",
        "CCL5_CCR5",
    )
    observed = tuple(
        str(_mapping(value, field="diagnostic_interactions[]").get("diagnostic_id"))
        for value in definitions
    )
    if observed != expected:
        raise ValueError("the three diagnostic interactions or their order changed")

    gene_subset = _mapping(config.get("gene_subset"), field="gene_subset")
    if (
        gene_subset.get("ann_data_feature_scope") != "full_input_transcriptome"
        or gene_subset.get("expected_input_gene_count") != 32938
        or gene_subset.get("expected_diagnostic_driver_count") != 3
        or gene_subset.get("expected_diagnostic_link_count") != 60
    ):
        raise ValueError("Kang normalization or diagnostic-prior scope changed")
    if gene_subset.get("nichenet_top_h5ad_observable_targets_per_ligand") != 20:
        raise ValueError(
            "Kang diagnostic requires 20 H5AD-observable NicheNet targets per ligand"
        )
    gene_sets = _mapping(
        gene_subset.get("pre_specified_gene_sets"),
        field="gene_subset.pre_specified_gene_sets",
    )
    if tuple(gene_sets) != (
        "interferon_stimulated",
        "antigen_presentation",
        "housekeeping",
    ):
        raise ValueError("pre-specified gene-set categories changed")
    for name, genes in gene_sets.items():
        _strings(genes, field=f"pre_specified_gene_sets.{name}")
    if gene_subset.get("include_lr_subunits") is not True:
        raise ValueError("diagnostic LR subunits must remain in the gene subset")

    crossfit = _mapping(config.get("crossfit"), field="crossfit")
    if tuple(
        _sequence(
            crossfit.get("outer_allowed_n_splits"),
            field="crossfit.outer_allowed_n_splits",
        )
    ) != (2,):
        raise ValueError("Kang diagnostic requires exactly two outer folds")
    frozen_counts = {
        "minimum_train_subjects_per_context": 4,
        "minimum_test_subjects_per_context": 4,
        "sender_minimum_subjects": 4,
    }
    if any(crossfit.get(key) != value for key, value in frozen_counts.items()):
        raise ValueError("outer train/test and sender support must remain four donors")
    partition_seed = crossfit.get("outer_fold_partition_seed")
    if (
        not isinstance(partition_seed, int)
        or isinstance(partition_seed, bool)
        or partition_seed == crossfit.get("random_seed")
    ):
        raise ValueError("outer fold partition seed must be an independent integer")
    if crossfit.get("sensitivity_comparison_scope") != "contrast_support_only":
        raise ValueError("sensitivity comparison scope must remain support-only")
    defaults = _mapping(config.get("defaults"), field="defaults")
    for key in (
        "ligand_contrast_minimum_effect",
        "receptor_gate_threshold",
    ):
        value = _float_value(defaults.get(key), field=f"defaults.{key}")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"defaults.{key} must lie in [0, 1]")
    if defaults["receptor_gate_threshold"] == 0:
        raise ValueError("receptor gate threshold must be positive")
    if config.get("claims") != CLAIMS:
        raise ValueError("exploratory benchmark claim boundary changed")
    return config


def select_diagnostic_interactions(
    bundle: ResourceBundle,
    definitions: Sequence[Mapping[str, object]],
) -> tuple[ResourceBundle, list[dict[str, object]]]:
    """Select the exact three source rows and return a frozen small bundle."""

    selected = []
    manifest: list[dict[str, object]] = []
    for definition in definitions:
        diagnostic_id = str(definition["diagnostic_id"])
        source_id = str(definition["source_interaction_id"])
        matches = [
            interaction
            for interaction in bundle.interactions
            if interaction.source_interaction_id == source_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{diagnostic_id} must map to exactly one CellChat interaction"
            )
        interaction = matches[0]
        expected_ligand = _strings(
            definition["ligand_subunits"], field="ligand_subunits"
        )
        expected_receptor = _strings(
            definition["receptor_subunits"], field="receptor_subunits"
        )
        if (
            interaction.ligand_name != str(definition["ligand"])
            or interaction.receptor_name != str(definition["receptor"])
            or interaction.ligand_subunits != expected_ligand
            or interaction.receptor_subunits != expected_receptor
        ):
            raise ValueError(f"CellChat molecular identity changed for {diagnostic_id}")
        selected.append(interaction)
        manifest.append(
            {
                **dict(definition),
                "harmonized_interaction_id": interaction.interaction_id,
                "cellchat_resource_id": bundle.resource_id,
                "cellchat_version": bundle.version,
            }
        )

    if len({item.interaction_id for item in selected}) != len(definitions):
        raise ValueError("diagnostic interaction selection is not one-to-one")
    genes = {
        gene
        for interaction in selected
        for gene in (*interaction.ligand_subunits, *interaction.receptor_subunits)
    }
    diagnostic_bundle = replace(
        bundle,
        interactions=tuple(selected),
        mapping_report=MappingReport(
            source_rows=bundle.mapping_report.source_rows,
            loaded_rows=len(selected),
            mapped_entities=len(genes),
            notes=(
                "pre_specified_kang2018_three_interaction_diagnostic_subset",
                "source_rows_checksum_verified_before_selection",
            ),
        ),
    )
    return diagnostic_bundle, manifest


def _top_observable_targets(
    target_prior: TargetPrior,
    *,
    ligand: str,
    top_n: int,
    available_genes: set[str],
) -> tuple[list[TargetLinkRecord], list[TargetLinkRecord]]:
    if ligand not in target_prior.driver_ids:
        raise ValueError(f"NicheNet target prior has no driver {ligand!r}")
    if target_prior.ranks is None:
        raise ValueError("Kang diagnostic requires source-ranked NicheNet links")
    index = target_prior.driver_ids.index(ligand)
    start, stop = target_prior.indptr[index : index + 2]
    records: list[TargetLinkRecord] = []
    for offset in range(start, stop):
        records.append(
            {
                "target": target_prior.target_ids[target_prior.target_indices[offset]],
                "rank": target_prior.ranks[offset],
                "weight": float(target_prior.weights[offset]),
            }
        )
    records.sort(key=lambda row: (row["rank"], row["target"]))
    selected = [row for row in records if row["target"] in available_genes][:top_n]
    if len(selected) != top_n or len({row["target"] for row in selected}) != top_n:
        raise ValueError(
            f"NicheNet {ligand} does not provide {top_n} unique H5AD-observable targets"
        )
    maximum_selected_rank = max(row["rank"] for row in selected)
    skipped = [
        row
        for row in records
        if row["rank"] <= maximum_selected_rank and row["target"] not in available_genes
    ]
    return selected, skipped


def build_diagnostic_target_prior(
    target_prior: TargetPrior,
    *,
    definitions: Sequence[Mapping[str, object]],
    gene_subset_config: Mapping[str, object],
    available_genes: Sequence[str],
) -> tuple[TargetPrior, dict[str, object]]:
    """Freeze the three-ligand target prior and diagnostic readout manifest."""

    top_n = int(
        cast(
            int,
            gene_subset_config["nichenet_top_h5ad_observable_targets_per_ligand"],
        )
    )
    available = set(map(str, available_genes))
    targets_by_ligand: dict[str, list[TargetLinkRecord]] = {}
    skipped_by_ligand: dict[str, list[TargetLinkRecord]] = {}
    requested: set[str] = set()
    for definition in definitions:
        ligand = str(definition["ligand"])
        targets, skipped = _top_observable_targets(
            target_prior,
            ligand=ligand,
            top_n=top_n,
            available_genes=available,
        )
        targets_by_ligand[ligand] = targets
        skipped_by_ligand[ligand] = skipped
        requested.update(str(row["target"]) for row in targets)
        requested.update(
            _strings(definition["ligand_subunits"], field="ligand_subunits")
        )
        requested.update(
            _strings(definition["receptor_subunits"], field="receptor_subunits")
        )

    gene_sets = _mapping(
        gene_subset_config["pre_specified_gene_sets"],
        field="pre_specified_gene_sets",
    )
    normalized_gene_sets: dict[str, list[str]] = {}
    for name, raw_genes in gene_sets.items():
        genes = sorted(_strings(raw_genes, field=f"pre_specified_gene_sets.{name}"))
        normalized_gene_sets[str(name)] = genes
        requested.update(genes)

    retained = tuple(sorted(requested.intersection(available)))
    missing = sorted(requested.difference(available))
    required_lr = {
        gene
        for definition in definitions
        for key in ("ligand_subunits", "receptor_subunits")
        for gene in _strings(definition[key], field=key)
    }
    absent_lr = sorted(required_lr.difference(retained))
    if absent_lr:
        raise ValueError(f"Kang input is missing diagnostic LR subunits: {absent_lr}")

    driver_ids = tuple(sorted(targets_by_ligand))
    target_ids = tuple(
        sorted(
            {
                record["target"]
                for records in targets_by_ligand.values()
                for record in records
            }
        )
    )
    absent_targets = sorted(set(target_ids).difference(available))
    if absent_targets:
        raise RuntimeError(
            "diagnostic TargetPrior contains targets absent from the frozen H5AD: "
            f"{absent_targets}"
        )
    target_index = {target: index for index, target in enumerate(target_ids)}
    target_indices: list[int] = []
    weights: list[float] = []
    ranks: list[int] = []
    indptr = [0]
    link_manifest: list[dict[str, object]] = []
    for ligand in driver_ids:
        for record in targets_by_ligand[ligand]:
            target_indices.append(target_index[record["target"]])
            weights.append(record["weight"])
            ranks.append(record["rank"])
            link_manifest.append({"ligand": ligand, **record})
        indptr.append(len(target_indices))
    expected_drivers = int(
        cast(int, gene_subset_config["expected_diagnostic_driver_count"])
    )
    expected_links = int(
        cast(int, gene_subset_config["expected_diagnostic_link_count"])
    )
    if len(driver_ids) != expected_drivers or len(weights) != expected_links:
        raise ValueError("diagnostic TargetPrior driver/link coverage changed")
    selected_link_digest = canonical_digest(link_manifest)
    if selected_link_digest != str(
        gene_subset_config["expected_diagnostic_target_link_digest"]
    ):
        raise ValueError("diagnostic TargetPrior link identity changed")
    diagnostic_prior = TargetPrior(
        resource_id=f"{target_prior.resource_id}_kang2018_diagnostic",
        version=f"{target_prior.version}+three_ligand_top{top_n}_h5ad_observable",
        species=target_prior.species,
        gene_namespace=target_prior.gene_namespace,
        driver_kind=target_prior.driver_kind,
        target_ids=target_ids,
        driver_ids=driver_ids,
        indptr=tuple(indptr),
        target_indices=tuple(target_indices),
        weights=tuple(weights),
        ranks=tuple(ranks),
        direction=target_prior.direction,
        evidence=(
            f"{target_prior.evidence}; checksum-pinned first {top_n} H5AD-observable "
            "links per ligand by source rank for the Kang diagnostic subset"
        ),
        mapping_report=MappingReport(
            source_rows=target_prior.mapping_report.source_rows,
            loaded_rows=len(weights),
            mapped_entities=len(driver_ids) + len(target_ids),
            notes=(
                "pre_specified_kang2018_three_ligand_h5ad_observable_target_prior",
                f"selected_link_digest={selected_link_digest}",
            ),
        ),
        manifest_digest=target_prior.manifest_digest,
    )
    manifest = {
        "selection_policy": (
            "per_ligand_first_20_h5ad_observable_nichenet_links_by_source_rank"
        ),
        "model_target_prior_scope": gene_subset_config["model_target_prior_scope"],
        "diagnostic_target_prior_resource_id": diagnostic_prior.resource_id,
        "diagnostic_target_prior_version": diagnostic_prior.version,
        "diagnostic_driver_ids": list(driver_ids),
        "diagnostic_target_count": len(target_ids),
        "diagnostic_link_count": len(weights),
        "selected_target_link_digest": selected_link_digest,
        "selected_target_links": link_manifest,
        "nichenet_top_h5ad_observable_targets_per_ligand": top_n,
        "nichenet_targets": targets_by_ligand,
        "unavailable_higher_ranked_nichenet_targets": skipped_by_ligand,
        "all_diagnostic_prior_targets_observed_in_h5ad": True,
        "pre_specified_gene_sets": normalized_gene_sets,
        "requested_gene_count": len(requested),
        "available_requested_gene_count": len(retained),
        "available_requested_genes": list(retained),
        "missing_requested_genes": missing,
    }
    return diagnostic_prior, manifest


def _prepare_input(
    input_path: Path,
    *,
    dataset_config: Mapping[str, object],
    gene_scope_config: Mapping[str, object],
    minimum_cells: int,
) -> tuple[ad.AnnData, dict[str, object]]:
    adata = ad.read_h5ad(input_path)
    required_obs = {"sample_id", "subject_id", "condition", "cell_type"}
    missing_obs = required_obs.difference(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"Kang AnnData lacks metadata columns: {sorted(missing_obs)}")
    if not adata.var_names.is_unique:
        raise ValueError("Kang AnnData gene symbols must be unique")
    expected_gene_count = int(cast(int, gene_scope_config["expected_input_gene_count"]))
    if adata.n_vars != expected_gene_count:
        raise ValueError("Kang analysis must retain the frozen full-transcriptome axis")

    cell_types = _strings(dataset_config["cell_types"], field="dataset.cell_types")
    expected_subject_count = int(cast(int, dataset_config["expected_subject_count"]))
    conditions = _mapping(dataset_config["conditions"], field="dataset.conditions")
    expected_contexts = {str(conditions["positive"]), str(conditions["negative"])}
    obs = adata.obs
    source_subjects = set(obs["subject_id"].astype(str).unique())
    cell_mask = obs["cell_type"].astype(str).isin(cell_types).to_numpy()
    subset = adata[cell_mask, :].copy()
    if subset.n_vars != adata.n_vars:
        raise RuntimeError("cell filtering unexpectedly changed the gene axis")
    subjects = set(subset.obs["subject_id"].astype(str).unique())
    contexts = set(subset.obs["condition"].astype(str).unique())
    if (
        len(subjects) != expected_subject_count
        or subjects != source_subjects
        or contexts != expected_contexts
    ):
        raise ValueError(
            "Kang subset does not contain the exact eight-donor paired contrast"
        )

    counts = (
        subset.obs.assign(
            subject_id=subset.obs["subject_id"].astype(str),
            condition=subset.obs["condition"].astype(str),
            cell_type=subset.obs["cell_type"].astype(str),
        )
        .groupby(["subject_id", "condition", "cell_type"], observed=True)
        .size()
    )
    expected_keys = {
        (subject, context, cell_type)
        for subject in subjects
        for context in expected_contexts
        for cell_type in cell_types
    }
    if set(counts.index) != expected_keys:
        raise ValueError("Kang subset lacks a donor-condition-cell-type block")
    if int(counts.min()) < minimum_cells:
        raise ValueError(
            "Kang subset violates the frozen minimum cell count: "
            f"observed={int(counts.min())}, required={minimum_cells}"
        )

    count_distribution = (
        counts.rename("n_cells")
        .reset_index()
        .groupby(["condition", "cell_type"], observed=True)
        .agg(
            n_subjects=("n_cells", "count"),
            total_cells=("n_cells", "sum"),
            minimum_cells=("n_cells", "min"),
            median_cells=("n_cells", "median"),
            maximum_cells=("n_cells", "max"),
        )
        .reset_index()
    )
    audit = {
        "source_shape": [int(adata.n_obs), int(adata.n_vars)],
        "analysis_shape": [int(subset.n_obs), int(subset.n_vars)],
        "subject_manifest": subject_set_manifest(subjects),
        "cell_types": list(cell_types),
        "conditions": sorted(expected_contexts),
        "all_subjects_paired": True,
        "normalization_scope": {
            "ann_data_feature_scope": gene_scope_config["ann_data_feature_scope"],
            "library_denominator": gene_scope_config["library_denominator"],
            "input_gene_count": int(adata.n_vars),
            "analysis_gene_count": int(subset.n_vars),
            "all_input_genes_retained_before_fold_normalization": True,
        },
        "minimum_cells_per_subject_condition_cell_type": int(counts.min()),
        "maximum_cells_per_subject_condition_cell_type": int(counts.max()),
        "cell_count_distribution": [
            {
                "condition": str(row.condition),
                "cell_type": str(row.cell_type),
                "n_subjects": int(row.n_subjects),
                "total_cells": int(row.total_cells),
                "minimum_cells": int(row.minimum_cells),
                "median_cells": float(row.median_cells),
                "maximum_cells": int(row.maximum_cells),
            }
            for row in count_distribution.itertuples(index=False)
        ],
    }
    return subset, audit


def build_crossfit_spec(
    config: Mapping[str, object],
    *,
    minimum_effect: float,
    receptor_gate_threshold: float,
) -> tuple[CrychicConfig, CrossFitSpec]:
    """Build the fixed two-fold public workflow policy."""

    dataset = _mapping(config["dataset"], field="dataset")
    conditions = _mapping(dataset["conditions"], field="dataset.conditions")
    policy = _mapping(config["crossfit"], field="crossfit")
    sender_parameters = ContrastCommonSenderParameters(
        min_subjects=_int_value(
            policy["sender_minimum_subjects"], field="sender_minimum_subjects"
        ),
        prevalence_threshold=_float_value(
            policy["sender_prevalence_threshold"],
            field="sender_prevalence_threshold",
        ),
        softmax_temperature=_float_value(
            policy["sender_softmax_temperature"],
            field="sender_softmax_temperature",
        ),
        ligand_contrast_confidence_level=_float_value(
            policy["ligand_contrast_confidence_level"],
            field="ligand_contrast_confidence_level",
        ),
        ligand_contrast_minimum_effect=minimum_effect,
    )
    tuning = PenaltyTuningSpec(
        lambda1_fractions=tuple(
            _float_value(value, field="penalty_lambda1_fractions[]")
            for value in _sequence(
                policy["penalty_lambda1_fractions"],
                field="penalty_lambda1_fractions",
            )
        ),
        lambda2_fractions=tuple(
            _float_value(value, field="penalty_lambda2_fractions[]")
            for value in _sequence(
                policy["penalty_lambda2_fractions"],
                field="penalty_lambda2_fractions",
            )
        ),
        inner_allowed_n_splits=tuple(
            _int_value(value, field="inner_allowed_n_splits[]")
            for value in _sequence(
                policy["inner_allowed_n_splits"],
                field="inner_allowed_n_splits",
            )
        ),
        min_inner_train_subjects_per_context=_int_value(
            policy["minimum_inner_train_subjects_per_context"],
            field="minimum_inner_train_subjects_per_context",
        ),
        min_inner_validation_subjects_per_context=_int_value(
            policy["minimum_inner_validation_subjects_per_context"],
            field="minimum_inner_validation_subjects_per_context",
        ),
        root_seed=_int_value(policy["random_seed"], field="random_seed"),
    )
    contrast = balanced_contrast(
        (str(conditions["positive"]),),
        (str(conditions["negative"]),),
        name="stim_vs_ctrl",
    )
    spec = CrossFitSpec(
        contrasts=(contrast,),
        outer_fold_partition_seed=_int_value(
            policy["outer_fold_partition_seed"],
            field="outer_fold_partition_seed",
        ),
        training_spec=FoldTrainingSpec(
            min_cells=_int_value(
                policy["minimum_cells_per_sample_cell_type"],
                field="minimum_cells_per_sample_cell_type",
            ),
            min_pooled_availability=_float_value(
                policy["minimum_pooled_availability"],
                field="minimum_pooled_availability",
            ),
            max_interactions=None,
            sender_parameters=sender_parameters,
        ),
        allowed_n_splits=tuple(
            _int_value(value, field="outer_allowed_n_splits[]")
            for value in _sequence(
                policy["outer_allowed_n_splits"],
                field="outer_allowed_n_splits",
            )
        ),
        min_train_subjects_per_context=_int_value(
            policy["minimum_train_subjects_per_context"],
            field="minimum_train_subjects_per_context",
        ),
        min_test_subjects_per_context=_int_value(
            policy["minimum_test_subjects_per_context"],
            field="minimum_test_subjects_per_context",
        ),
        receptor_gate_threshold=receptor_gate_threshold,
        family_cosine_threshold=_float_value(
            policy["family_cosine_threshold"], field="family_cosine_threshold"
        ),
        downstream_minimum_scale=_float_value(
            policy["downstream_minimum_scale"], field="downstream_minimum_scale"
        ),
        autonomous_program_resource=None,
        autonomous_program_use_scope=cast(
            AutonomousProgramUseScope,
            policy["autonomous_program_use_scope"],
        ),
        penalty_tuning_spec=tuning,
    )
    crychic_config = CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=_int_value(policy["random_seed"], field="random_seed"),
    )
    return crychic_config, spec


def _missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _count_values(values: Sequence[object]) -> dict[str, int]:
    counts: Counter[str] = Counter(
        "none" if _missing(value) else str(value) for value in values
    )
    return dict(sorted(counts.items()))


def _finite_float(value: object) -> float | None:
    if _missing(value):
        return None
    numeric = float(cast(float, value))
    return numeric if math.isfinite(numeric) else None


def paired_effect_summary(
    table: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str,
    positive_context_id: str,
    negative_context_id: str,
) -> list[dict[str, object]]:
    """Summarize finite subject-paired effects without serializing row values."""

    required = {
        *group_columns,
        "subject_id",
        "context_id",
        "status",
        "reason_code",
        value_column,
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"score table lacks compact-summary columns: {sorted(missing)}"
        )
    if table.empty:
        return []
    working = table.copy()
    working["_numeric_value"] = pd.to_numeric(working[value_column], errors="coerce")
    records: list[dict[str, object]] = []
    grouping: str | list[str]
    grouping = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for raw_key, group in working.groupby(grouping, observed=True, sort=True):
        keys: tuple[object, ...]
        if len(group_columns) == 1:
            keys = (raw_key,)
        elif isinstance(raw_key, tuple):
            keys = raw_key
        else:  # pragma: no cover - pandas returns tuples for multi-column groups
            raise RuntimeError("multi-column group key is not a tuple")
        context_values = (
            group.groupby(["subject_id", "context_id"], observed=True, sort=True)[
                "_numeric_value"
            ]
            .mean()
            .unstack("context_id")
        )
        if (
            positive_context_id in context_values
            and negative_context_id in context_values
        ):
            effects = (
                context_values[positive_context_id]
                - context_values[negative_context_id]
            ).dropna()
        else:
            effects = pd.Series(dtype=float)
        values = effects.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        stats: dict[str, object]
        if finite.size:
            stats = {
                "mean_paired_effect": float(np.mean(finite)),
                "median_paired_effect": float(np.median(finite)),
                "sample_standard_deviation": (
                    None if finite.size < 2 else float(np.std(finite, ddof=1))
                ),
                "minimum_paired_effect": float(np.min(finite)),
                "maximum_paired_effect": float(np.max(finite)),
            }
        else:
            stats = {
                "mean_paired_effect": None,
                "median_paired_effect": None,
                "sample_standard_deviation": None,
                "minimum_paired_effect": None,
                "maximum_paired_effect": None,
            }
        records.append(
            {
                **dict(zip(group_columns, map(str, keys), strict=True)),
                "value_column": value_column,
                "n_rows": len(group),
                "n_finite_rows": int(group["_numeric_value"].notna().sum()),
                "n_complete_subjects": int(finite.size),
                "status_counts": _count_values(group["status"].tolist()),
                "reason_counts": _count_values(group["reason_code"].tolist()),
                **stats,
            }
        )
    return records


def _score_tables(
    artifacts: CrossFitArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_parts: list[pd.DataFrame] = []
    member_parts: list[pd.DataFrame] = []
    sender_parts: list[pd.DataFrame] = []
    for fold in artifacts.folds:
        for application in fold.family_common_applications:
            for table, destination in (
                (application.family_scores, family_parts),
                (application.member_scores, member_parts),
                (application.sender_scores, sender_parts),
            ):
                frame = table.copy()
                frame.insert(0, "fold_id", fold.fold_id)
                destination.append(frame)
    return tuple(
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for parts in (family_parts, member_parts, sender_parts)
    )  # type: ignore[return-value]


def compact_paired_summaries(
    artifacts: CrossFitArtifacts,
) -> dict[str, object]:
    """Build fold-local and pooled OOF family/member/sender summaries."""

    context_weights = {
        tuple(functional.contrast_weights)
        for fold in artifacts.folds
        for functional in fold.training.sender_functionals
    }
    if len(context_weights) != 1:
        raise ValueError("cross-fit folds do not share one frozen contrast mapping")
    weights = next(iter(context_weights))
    positive = [context for context, weight in weights if weight > 0]
    negative = [context for context, weight in weights if weight < 0]
    if len(positive) != 1 or len(negative) != 1:
        raise ValueError(
            "compact summary requires one positive and one negative context"
        )
    positive_context_id, negative_context_id = positive[0], negative[0]
    family, member, sender = _score_tables(artifacts)

    def summarize(
        table: pd.DataFrame,
        groups: Sequence[str],
        value: str,
    ) -> list[dict[str, object]]:
        return paired_effect_summary(
            table,
            group_columns=groups,
            value_column=value,
            positive_context_id=positive_context_id,
            negative_context_id=negative_context_id,
        )

    return {
        "estimand": "subject_level_stim_minus_ctrl",
        "within_run_descriptive_only": True,
        "comparable_across_threshold_sensitivity_runs": False,
        "positive_context_id": positive_context_id,
        "negative_context_id": negative_context_id,
        "family": {
            "fold_level": summarize(
                family,
                ("fold_id", "receiver", "family_id", "mode"),
                "integrated_lr_score",
            )
        },
        "member": {
            "fold_level": summarize(
                member,
                ("fold_id", "receiver", "interaction_id", "mode"),
                "sender_unresolved_strength",
            ),
            "oof_pooled": summarize(
                member,
                ("receiver", "interaction_id", "mode"),
                "sender_unresolved_strength",
            ),
        },
        "sender": {
            "fold_level": summarize(
                sender,
                ("fold_id", "receiver", "interaction_id", "mode", "sender"),
                "sender_resolved_strength",
            ),
            "oof_pooled": summarize(
                sender,
                ("receiver", "interaction_id", "mode", "sender"),
                "sender_resolved_strength",
            ),
        },
    }


def _fold_partition_record(fold: CrossFitFoldArtifacts) -> dict[str, object]:
    training = subject_set_manifest(fold.training.training_subject_ids)
    heldout = subject_set_manifest(fold.application.heldout_subject_ids)
    identity = {
        "training_subject_set_digest": training["subject_set_digest"],
        "heldout_subject_set_digest": heldout["subject_set_digest"],
        "n_training_subjects": training["n_subjects"],
        "n_heldout_subjects": heldout["n_subjects"],
    }
    return {
        "fold_partition_id": canonical_digest(identity),
        "training_subject_manifest": training,
        "heldout_subject_manifest": heldout,
    }


def subject_partition_manifest(artifacts: CrossFitArtifacts) -> dict[str, object]:
    """Describe the outer partition independently of threshold-sensitive IDs."""

    partitions = []
    for fold in artifacts.folds:
        record = _fold_partition_record(fold)
        partitions.append({"fold_id": fold.fold_id, **record})
    partitions.sort(key=lambda row: str(row["fold_partition_id"]))
    identity_records = [
        {key: value for key, value in record.items() if key != "fold_id"}
        for record in partitions
    ]
    lineage = artifacts.fold_plan.partition_seed_lineage
    if lineage is None or artifacts.spec.outer_fold_partition_seed is None:
        raise ValueError("Kang sensitivity runs require an explicit partition seed")
    return {
        "subject_partition_manifest_id": canonical_digest(identity_records),
        "outer_fold_partition_seed": artifacts.spec.outer_fold_partition_seed,
        "partition_seed_lineage": lineage.to_dict(),
        "partitions": partitions,
        "sensitivity_alignment_scope": "contrast_support_only",
        "downstream_inner_tuned_scores_comparable_across_sensitivity_runs": False,
    }


def _expected_audit_keys(
    artifacts: CrossFitArtifacts,
    interaction_ids: Sequence[str],
) -> set[tuple[str, str, str]]:
    return {
        (fold.fold_id, receiver, interaction_id)
        for fold in artifacts.folds
        for receiver in fold.training.cell_type_ids
        for interaction_id in interaction_ids
    }


def compact_support_records(
    artifacts: CrossFitArtifacts,
    *,
    interaction_manifest: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Extract only the audit fields of v3 training-fold contrast supports."""

    diagnostics = {
        str(item["harmonized_interaction_id"]): str(item["diagnostic_id"])
        for item in interaction_manifest
    }
    records: list[dict[str, object]] = []
    observed_keys: list[tuple[str, str, str]] = []
    for fold in artifacts.folds:
        partition = _fold_partition_record(fold)
        for functional in fold.training.sender_functionals:
            for support in functional.contrast_supports:
                if support.interaction_id not in diagnostics:
                    continue
                observed_keys.append(
                    (fold.fold_id, support.receiver, support.interaction_id)
                )
                gate = interaction_ligand_contrast_gate(
                    functional,
                    support.receiver,
                    support.interaction_id,
                )
                records.append(
                    {
                        "fold_id": fold.fold_id,
                        **partition,
                        "receiver": support.receiver,
                        "diagnostic_id": diagnostics[support.interaction_id],
                        "interaction_id": support.interaction_id,
                        "sender_functional_id": functional.sender_functional_id,
                        "support_id": support.support_id,
                        "n_complete": support.n_complete,
                        "minimum_complete_subjects": (
                            support.minimum_complete_subjects
                        ),
                        "mean_effect": _finite_float(support.mean_effect),
                        "raw_one_sided_p_value": _finite_float(
                            support.raw_one_sided_p_value
                        ),
                        "holm_adjusted_p_value": float(support.holm_adjusted_p_value),
                        "holm_rank": support.holm_rank,
                        "multiplicity_family_size": (support.multiplicity_family_size),
                        "multiplicity_family_id": (support.multiplicity_family_id),
                        "status": support.status.value,
                        "reason_code": support.reason_code,
                        "ligand_contrast_gate": gate.gate,
                        "ligand_contrast_gate_id": gate.gate_id,
                        "ligand_contrast_gate_status": gate.status.value,
                        "ligand_contrast_gate_reason_code": gate.reason_code,
                    }
                )
    expected_keys = _expected_audit_keys(artifacts, tuple(diagnostics))
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != (
        expected_keys
    ):
        raise ValueError(
            "contrast support output lacks exact fold-receiver-interaction key coverage"
        )
    return sorted(
        records,
        key=lambda row: (
            row["fold_id"],
            row["receiver"],
            row["diagnostic_id"],
        ),
    )


def compact_receptor_gate_records(
    artifacts: CrossFitArtifacts,
    *,
    interaction_manifest: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Extract exact fold-receiver-interaction receptor gate provenance."""

    diagnostics = {
        str(item["harmonized_interaction_id"]): str(item["diagnostic_id"])
        for item in interaction_manifest
    }
    records: list[dict[str, object]] = []
    observed_keys: list[tuple[str, str, str]] = []
    for fold in artifacts.folds:
        partition = _fold_partition_record(fold)
        for model in fold.receiver_family_models:
            artifact = model.receiver_family_artifact
            driver_by_interaction = dict(artifact.driver_by_interaction)
            gate_by_driver = dict(artifact.receptor_gates)
            source_basis = artifact.source_basis
            eligible_by_driver = dict(
                zip(
                    source_basis.driver_ids,
                    source_basis.receptor_eligible.tolist(),
                    strict=True,
                )
            )
            for interaction_id, diagnostic_id in diagnostics.items():
                driver = driver_by_interaction.get(interaction_id)
                if driver is None or driver not in gate_by_driver:
                    raise ValueError(
                        "receiver gate artifact lacks a diagnostic interaction driver"
                    )
                key = (fold.fold_id, artifact.receiver, interaction_id)
                observed_keys.append(key)
                records.append(
                    {
                        "fold_id": fold.fold_id,
                        **partition,
                        "receiver": artifact.receiver,
                        "diagnostic_id": diagnostic_id,
                        "interaction_id": interaction_id,
                        "driver_id": driver,
                        "receptor_gate": float(gate_by_driver[driver]),
                        "receptor_gate_threshold": (
                            source_basis.receptor_gate_threshold
                        ),
                        "receptor_eligible": bool(eligible_by_driver[driver]),
                        "receptor_gate_policy": source_basis.gate_policy.to_dict(),
                        "receptor_gate_manifest_id": (
                            artifact.receptor_gate_manifest_id
                        ),
                        "receptor_evidence_digest": (artifact.receptor_evidence_digest),
                    }
                )
    expected_keys = _expected_audit_keys(artifacts, tuple(diagnostics))
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != (
        expected_keys
    ):
        raise ValueError(
            "receptor gate output lacks exact fold-receiver-interaction key coverage"
        )
    return sorted(
        records,
        key=lambda row: (
            row["fold_id"],
            row["receiver"],
            row["diagnostic_id"],
        ),
    )


def _fold_manifests(artifacts: CrossFitArtifacts) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for fold in artifacts.folds:
        partition = _fold_partition_record(fold)
        records.append(
            {
                "fold_id": fold.fold_id,
                **partition,
                "sender_functionals": [
                    {
                        "sender_functional_id": functional.sender_functional_id,
                        "filter_universe_id": functional.filter_universe_id,
                        "frozen_interaction_ids": list(
                            functional.frozen_interaction_ids
                        ),
                        "candidate_sender_manifest": [
                            [receiver, interaction, list(senders)]
                            for receiver, interaction, senders in (
                                functional.candidate_sender_manifest
                            )
                        ],
                        "training_availability_digest": (
                            functional.training_availability_digest
                        ),
                        "training_input_digest": functional.training_input_digest,
                    }
                    for functional in fold.training.sender_functionals
                ],
            }
        )
    return records


def compact_incremental_tuning_records(
    artifacts: CrossFitArtifacts,
) -> list[dict[str, object]]:
    """Return deidentified inner-tuning decisions for each receiver-fold model."""

    records: list[dict[str, object]] = []
    for fold in artifacts.folds:
        for model in fold.receiver_incremental_models:
            tuning = model.penalty_tuning_artifact
            candidates: list[dict[str, object]] = []
            if tuning is not None:
                summary_by_id = {
                    summary.candidate_id: summary for summary in tuning.summaries
                }
                comparison_by_id = {
                    comparison.candidate_id: comparison
                    for comparison in tuning.candidate_comparisons
                }
                for candidate in tuning.spec.candidates:
                    summary = summary_by_id[candidate.candidate_id]
                    comparison = comparison_by_id.get(candidate.candidate_id)
                    candidates.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "lambda1_fraction": candidate.lambda1_fraction,
                            "lambda2_fraction": candidate.lambda2_fraction,
                            "status": summary.status,
                            "reason_code": summary.reason_code,
                            "mean_subject_equal_loss": summary.mean_loss,
                            "subject_equal_standard_error": (summary.standard_error),
                            "mean_paired_loss_difference_to_best": (
                                None
                                if comparison is None
                                else comparison.mean_loss_difference
                            ),
                            "relative_mean_loss_difference_to_best": (
                                None
                                if comparison is None
                                or tuning.best_mean_loss is None
                                or tuning.best_mean_loss == 0
                                else comparison.mean_loss_difference
                                / tuning.best_mean_loss
                            ),
                            "paired_difference_standard_error": (
                                None
                                if comparison is None
                                else comparison.standard_error
                            ),
                            "within_paired_one_se": (
                                None if comparison is None else comparison.within_one_se
                            ),
                            "is_best": (
                                candidate.candidate_id == tuning.best_candidate_id
                            ),
                            "is_selected": (
                                candidate.candidate_id == tuning.selected_candidate_id
                            ),
                        }
                    )
            selected = None if tuning is None else tuning.selected_candidate
            functional = model.diagnostic_functional
            coefficients = (
                np.empty(0, dtype=np.float64)
                if functional is None
                else np.asarray(functional.family_coefficients, dtype=np.float64)
            )
            records.append(
                {
                    "fold_id": fold.fold_id,
                    "receiver": model.receiver,
                    "diagnostic_status": model.diagnostic_status,
                    "diagnostic_reason_code": model.diagnostic_reason_code,
                    "official_status": model.official_incremental_status,
                    "official_reason_code": model.reason_code,
                    "tuning_status": None if tuning is None else tuning.status,
                    "tuning_reason_code": (
                        None if tuning is None else tuning.reason_code
                    ),
                    "tuning_oof_certified": bool(
                        tuning is not None and tuning.is_oof_certified
                    ),
                    "n_inner_folds": (
                        0 if tuning is None else len(tuning.inner_fold_ids)
                    ),
                    "n_training_subjects": len(model.training_subject_ids),
                    "selected_candidate_id": (
                        None if tuning is None else tuning.selected_candidate_id
                    ),
                    "selected_lambda1_fraction": (
                        None if selected is None else selected.lambda1_fraction
                    ),
                    "selected_lambda2_fraction": (
                        None if selected is None else selected.lambda2_fraction
                    ),
                    "resolved_lambda1": (None if functional is None else model.lambda1),
                    "resolved_lambda2": (None if functional is None else model.lambda2),
                    "n_families": len(model.family_ids),
                    "n_strictly_positive_family_coefficients": int(
                        np.count_nonzero(coefficients > 0)
                    ),
                    "n_effectively_positive_family_coefficients": int(
                        np.count_nonzero(coefficients > 1e-12)
                    ),
                    "effective_coefficient_tolerance": 1e-12,
                    "maximum_family_coefficient": (
                        None if not len(coefficients) else float(np.max(coefficients))
                    ),
                    "candidate_diagnostics": candidates,
                }
            )
    return sorted(records, key=lambda row: (row["fold_id"], row["receiver"]))


def _compact_crossfit_audit(artifacts: CrossFitArtifacts) -> dict[str, object]:
    manifest = artifacts.to_manifest()
    functionals = [
        functional
        for fold in artifacts.folds
        for functional in fold.family_common_functionals
    ]
    applications = [
        application
        for fold in artifacts.folds
        for application in fold.family_common_applications
    ]
    models = [
        model for fold in artifacts.folds for model in fold.receiver_incremental_models
    ]
    return {
        "crossfit_id": artifacts.crossfit_id,
        "crossfit_spec_id": artifacts.spec.spec_id,
        "n_folds": len(artifacts.folds),
        "certification_status": artifacts.certification_status,
        "completed_stage_oof_verified": artifacts.completed_stage_oof_verified,
        "complete_pipeline_oof_certified": artifacts.is_oof_certified,
        "remaining_stages": manifest["remaining_stages"],
        "receiver_autonomous_program_resource": None,
        "receiver_autonomous_nuisance_intentionally_unresolved": True,
        "incremental_official_status_counts": _count_values(
            [model.official_incremental_status for model in models]
        ),
        "incremental_diagnostic_status_counts": _count_values(
            [model.diagnostic_status for model in models]
        ),
        "incremental_diagnostic_reason_counts": _count_values(
            [model.diagnostic_reason_code for model in models]
        ),
        "incremental_reason_counts": _count_values(
            [model.reason_code for model in models]
        ),
        "incremental_tuning_records": compact_incremental_tuning_records(artifacts),
        "family_common_functional_status_counts": _count_values(
            [
                "observed"
                if functional.incremental_functional is not None
                else "not_estimable"
                for functional in functionals
            ]
        ),
        "family_common_application_status_counts": _count_values(
            [
                "observed"
                if application.heldout_reason_code is None
                else "not_estimable"
                for application in applications
            ]
        ),
        "family_common_application_reason_counts": _count_values(
            [application.heldout_reason_code for application in applications]
        ),
    }


def _source_provenance(repository_root: Path) -> dict[str, object]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "git_commit": completed.stdout.strip(),
        "git_worktree_dirty": bool(status),
        "git_status_porcelain": status,
        "source_sha256": {
            relative: sha256_file(repository_root / relative)
            for relative in SOURCE_PATHS
        },
    }


def _resource_files(database_root: Path) -> dict[str, list[dict[str, object]]]:
    paths = {
        "cellchat": (
            database_root / "cellchat/CHECKSUMS.sha256",
            database_root / "cellchat/CELLCHATDB_VERSION.txt",
            database_root / "cellchat/human_cofactor.csv",
            database_root / "cellchat/human_complex.csv",
            database_root / "cellchat/human_interaction.csv",
        ),
        "nichenet": (
            database_root / "nichenet/v2_2021/manifest.json",
            database_root / "nichenet/v2_2021/ligand_target_top250.parquet",
        ),
    }
    return {
        resource_name: [
            {
                "path": str(path.relative_to(database_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in resource_paths
        ]
        for resource_name, resource_paths in paths.items()
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("CRYCHIC", "anndata", "numpy", "pandas", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def run_benchmark(
    *,
    workspace_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    minimum_effect: float | None = None,
    receptor_gate_threshold: float | None = None,
) -> dict[str, object]:
    """Execute the checksum-pinned two-cell-type real-data diagnostic."""

    total_started = time.perf_counter()
    workspace_root = workspace_root.resolve()
    config_path = config_path.resolve()
    config = load_benchmark_config(config_path)
    defaults = _mapping(config["defaults"], field="defaults")
    resolved_minimum_effect = _float_value(
        defaults["ligand_contrast_minimum_effect"]
        if minimum_effect is None
        else minimum_effect,
        field="ligand_contrast_minimum_effect",
    )
    resolved_receptor_threshold = _float_value(
        defaults["receptor_gate_threshold"]
        if receptor_gate_threshold is None
        else receptor_gate_threshold,
        field="receptor_gate_threshold",
    )
    if not math.isfinite(resolved_minimum_effect) or not (
        0 <= resolved_minimum_effect <= 1
    ):
        raise ValueError("minimum_effect must lie in [0, 1]")
    if not math.isfinite(resolved_receptor_threshold) or not (
        0 < resolved_receptor_threshold <= 1
    ):
        raise ValueError("receptor_gate_threshold must lie in (0, 1]")

    database_root = workspace_root / "databases"
    full_bundle = load_cellchat_resource(database_root, Species.HUMAN)
    full_target_prior = load_nichenet_target_prior(database_root)
    definitions = [
        _mapping(item, field="diagnostic_interactions[]")
        for item in _sequence(
            config["diagnostic_interactions"], field="diagnostic_interactions"
        )
    ]
    bundle, interaction_manifest = select_diagnostic_interactions(
        full_bundle, definitions
    )
    selected_interaction_id_digest = canonical_digest(
        [interaction.interaction_id for interaction in bundle.interactions]
    )

    dataset_config = _mapping(config["dataset"], field="dataset")
    relative_input = Path(str(dataset_config["relative_path"]))
    input_path = workspace_root / relative_input
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    gene_scope_config = _mapping(config["gene_subset"], field="gene_subset")
    input_verification, available_genes = verify_input_artifact(
        input_path,
        dataset_config=dataset_config,
        expected_gene_count=int(
            cast(int, gene_scope_config["expected_input_gene_count"])
        ),
    )
    diagnostic_target_prior, gene_manifest = build_diagnostic_target_prior(
        full_target_prior,
        definitions=definitions,
        gene_subset_config=gene_scope_config,
        available_genes=available_genes,
    )
    policy = _mapping(config["crossfit"], field="crossfit")
    adata, input_audit = _prepare_input(
        input_path,
        dataset_config=dataset_config,
        gene_scope_config=gene_scope_config,
        minimum_cells=int(cast(int, policy["minimum_cells_per_sample_cell_type"])),
    )
    crychic_config, spec = build_crossfit_spec(
        config,
        minimum_effect=resolved_minimum_effect,
        receptor_gate_threshold=resolved_receptor_threshold,
    )
    crossfit_started = time.perf_counter()
    artifacts = run_subject_crossfit(
        adata,
        crychic_config,
        bundle,
        diagnostic_target_prior,
        spec=spec,
    )
    crossfit_elapsed = time.perf_counter() - crossfit_started
    support_records = compact_support_records(
        artifacts,
        interaction_manifest=interaction_manifest,
    )
    receptor_gate_records = compact_receptor_gate_records(
        artifacts,
        interaction_manifest=interaction_manifest,
    )
    paired_summaries = compact_paired_summaries(artifacts)
    partition_manifest = subject_partition_manifest(artifacts)
    crossfit_audit = _compact_crossfit_audit(artifacts)
    if "receiver_autonomous_nuisance" not in cast(
        Sequence[str], crossfit_audit["remaining_stages"]
    ):
        raise ValueError("real-data diagnostic unexpectedly closed nuisance stage")

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "kang2018_real_data_small_ligand_gate_v3_development_diagnostic",
        "claims": dict(CLAIMS),
        "interpretation": (
            "literature-informed exploratory/noncertifying diagnostic with no "
            "biological-validation or method-superiority claim; ligand contrast "
            "supports are training-fold diagnostics and family-common score rows do "
            "not close the complete-pipeline receiver-autonomous nuisance adjustment"
        ),
        "sensitivity_comparison_policy": {
            "scope": policy["sensitivity_comparison_scope"],
            "aligned_by": "subject_partition_manifest_id",
            "contrast_support_comparison_allowed": True,
            "family_member_sender_score_comparison_allowed": False,
            "reason": (
                "downstream family/member/sender scores include threshold-sensitive "
                "bases and separately inner-tuned fits"
            ),
        },
        "configuration": {
            "path": _portable_path(config_path, REPOSITORY_ROOT, workspace_root),
            "sha256": EXPECTED_CONFIG_SHA256,
            "complete_config_checksum_verified": True,
            "ligand_contrast_minimum_effect": resolved_minimum_effect,
            "receptor_gate_threshold": resolved_receptor_threshold,
            "crychic_config": crychic_config.to_dict(),
            "crossfit_spec": spec.to_dict(),
        },
        "input": {
            "path": _portable_path(input_path, workspace_root),
            **input_verification,
            **input_audit,
        },
        "resources": {
            "cellchat": {
                "resource_id": full_bundle.resource_id,
                "version": full_bundle.version,
                "manifest_digest": full_bundle.manifest_digest,
                "n_source_interactions": len(full_bundle.interactions),
                "n_selected_interactions": len(bundle.interactions),
                "selected_interaction_id_digest": (selected_interaction_id_digest),
            },
            "nichenet": {
                "resource_id": full_target_prior.resource_id,
                "version": full_target_prior.version,
                "manifest_digest": full_target_prior.manifest_digest,
                "n_source_drivers": len(full_target_prior.driver_ids),
                "n_source_targets": len(full_target_prior.target_ids),
                "diagnostic_target_prior": {
                    "resource_id": diagnostic_target_prior.resource_id,
                    "version": diagnostic_target_prior.version,
                    "n_drivers": len(diagnostic_target_prior.driver_ids),
                    "n_targets": len(diagnostic_target_prior.target_ids),
                    "n_links": diagnostic_target_prior.nnz,
                    "selected_target_link_digest": gene_manifest[
                        "selected_target_link_digest"
                    ],
                },
            },
            "verified_files": _resource_files(database_root),
        },
        "frozen_manifest": {
            "diagnostic_interactions": interaction_manifest,
            "selected_interaction_id_digest": selected_interaction_id_digest,
            "diagnostic_target_prior_and_readout": gene_manifest,
            "normalization_scope": input_audit["normalization_scope"],
            "subject_partition": partition_manifest,
            "folds": _fold_manifests(artifacts),
        },
        "fold_receiver_interaction_supports": support_records,
        "fold_receiver_interaction_receptor_gates": receptor_gate_records,
        "paired_summaries": paired_summaries,
        "crossfit_audit": crossfit_audit,
        "runtime": {
            "crossfit_elapsed_seconds": crossfit_elapsed,
            "total_elapsed_seconds": time.perf_counter() - total_started,
            "process_peak_rss_kib": int(
                process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "software_versions": _package_versions(),
        "source_provenance": _source_provenance(REPOSITORY_ROOT),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
        help="Workspace containing databases/ and benchmark_work/.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--minimum-effect",
        type=float,
        default=None,
        help="Override the ligand availability-scale null effect.",
    )
    parser.add_argument(
        "--receptor-gate-threshold",
        type=float,
        default=None,
        help="Override the training-fold receptor eligibility threshold.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = workspace_root / output
    result = run_benchmark(
        workspace_root=workspace_root,
        config_path=args.config,
        minimum_effect=args.minimum_effect,
        receptor_gate_threshold=args.receptor_gate_threshold,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
