"""Run checksum-bound paired v7 validation on Kang or BRCA intervention data."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from benchmarks.adapters.common import (
    git_metadata,
    json_safe,
    prepare_output,
    python_environment,
    sha256_file,
    write_json,
)
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.simulation.v7_integrated import (
    V7InferenceFitCache,
    build_v7_primary_score_views,
    g0_g2_equivalence_diagnostic,
    run_v7_inference_matrix,
)
from crychic import CrychicConfig
from crychic.design import balanced_contrast
from crychic.inference import DifferentialContrastSpec, DifferentialDesignSpec
from crychic.resources import ResourceBundle, TargetPrior, load_nichenet_target_prior
from crychic.scoring import AbsoluteActivityV2Spec, SignedProgramV2Spec
from crychic.sender import (
    ContrastCommonSenderParameters,
    EBShrunkenCouplingV2Spec,
    SenderAttributionV2Spec,
)
from crychic.workflow import CrossFitSpec, FoldTrainingSpec
from crychic.workflow.crossfit import (
    _run_v7_primary_crossfit,
    _V7PrimaryCrossFitArtifacts,
)
from crychic.workflow.training import _sanitized_raw_input_snapshot

SCHEMA_VERSION = "crychic-suggest-next2-v7-intervention-run-v1"
CONFIG_SCHEMA_VERSION = "crychic-suggest-next2-v7-intervention-v1"
FROZEN_ESTIMATOR_COMMIT = "5dc22aaa87896eafbeb687aea8d8e96e3654af01"
FROZEN_SCORE_PROJECTION_COMMIT = "35db184d62653ff8c960e2b135fa8e3060e1d394"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "benchmarks/configs/suggest_next2_v7_intervention_v1.json"
)
INFERENCE_SCORE_VIEWS = {
    "G0": ("primary_sender_detection",),
    "G2": ("primary_sender_detection",),
    "G3": ("primary_sender_detection",),
    "G4": ("m0_parent_mean_component", "program_signed_component"),
    "G5": ("primary_sender_detection",),
}
DATASET_SLUGS = ("kang", "brca_e", "brca_ne")


@dataclass(frozen=True, slots=True)
class InterventionDatasetContract:
    slug: str
    dataset_id: str
    role: str
    condition_column: str
    reference: str
    target: str
    contrast_name: str
    seed: int
    outer_fold_partition_seed: int
    allowed_outer_folds: tuple[int, ...]
    input_schema: str
    input_sha256: str
    input_manifest_sha256: str
    cells: int
    genes: int
    samples: int
    subjects: int
    cell_types: tuple[str, ...]
    truth_schema: str | None = None
    truth_sha256: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML object required: {path}")
    return value


def _input_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "columns": list(map(str, table.columns)),
        "sha256": sha256_file(path),
    }


def _append_log(path: Path, event: str, **fields: object) -> None:
    record = {"event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + "\n"
        )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_intervention_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, InterventionDatasetContract]]:
    """Load and validate the preregistered paired-intervention protocol."""

    config = _read_json(path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "preregistered_before_v7_intervention_refit"
        or config.get("estimator_commit") != FROZEN_ESTIMATOR_COMMIT
        or config.get("score_projection_commit") != FROZEN_SCORE_PROJECTION_COMMIT
    ):
        raise ValueError("v7 intervention config schema or frozen source changed")
    datasets = config.get("datasets")
    resource = config.get("resource")
    prior = config.get("target_prior")
    estimator = config.get("estimator")
    endpoints = config.get("endpoints")
    release = config.get("release")
    if not all(
        isinstance(value, Mapping)
        for value in (datasets, resource, prior, estimator, endpoints, release)
    ):
        raise ValueError("v7 intervention config sections must be mappings")
    assert isinstance(datasets, Mapping)
    assert isinstance(resource, Mapping)
    assert isinstance(prior, Mapping)
    assert isinstance(estimator, Mapping)
    assert isinstance(endpoints, Mapping)
    assert isinstance(release, Mapping)
    if set(datasets) != set(DATASET_SLUGS):
        raise ValueError("v7 intervention dataset axis changed")
    if (
        resource.get("schema_version") != "crychic-harmonized-lr-v1"
        or resource.get("resource_id")
        != "crychic_misc_four_method_harmonized_simple_lr"
        or resource.get("interactions") != 455
        or not _valid_sha256(resource.get("table_sha256"))
        or not _valid_sha256(resource.get("manifest_sha256"))
        or prior.get("release") != "v2_2021"
        or not _valid_sha256(prior.get("manifest_sha256"))
    ):
        raise ValueError("v7 intervention resource contract changed")
    expected_estimator: dict[str, object] = {
        "execution_profile": "v7_primary_m0_m5_v1",
        "bottleneck_penalty": 0.0,
        "minimum_sender_calibration_subjects": 4,
        "minimum_signed_program_samples": 4,
        "minimum_signed_program_subjects": 3,
        "minimum_coupling_subjects": 4,
        "minimum_coupling_edges_for_eb": 2,
        "attribution_coupling_weight": 0.25,
        "minimum_cells_per_sample_cell_type": 10,
        "minimum_subjects_per_level": 4,
        "minimum_clusters_for_cr2": 6,
        "minimum_common_sender_subjects": 4,
        "minimum_pooled_availability": 0.0,
        "minimum_train_subjects_per_context": 4,
        "minimum_test_subjects_per_context": 2,
        "inference_id": "I1",
        "generators": ["G0", "G2", "G3", "G4", "G5"],
        "inference_score_views": {
            key: list(value) for key, value in INFERENCE_SCORE_VIEWS.items()
        },
    }
    if dict(estimator) != expected_estimator:
        raise ValueError("v7 intervention estimator contract changed")
    expected_release = {
        "formal_inference_allowed": False,
        "diagnostic_p_values_are_not_formal": True,
        "real_data_edge_auroc_forbidden": True,
        "causal_sender_claim_forbidden": True,
        "dataset_specific_parameter_selection_forbidden": True,
        "kang_exogenous_ifnb_sender_recovery_required": False,
        "brca_e_and_ne_must_be_fit_separately": True,
        "brca_cross_subtype_effect_is_descriptive_only": True,
    }
    if dict(release) != expected_release:
        raise ValueError("v7 intervention release boundary changed")
    expected_endpoints = {
        "kang": [
            "paired_gene_effect_direction",
            "donor_direction_consistency",
            "cell_type_composition_stability",
            "g4_signed_program_direction",
            "m0_program_concordance",
            "endogenous_ifnb_guardrail",
        ],
        "brca": [
            "paired_on_minus_pre_effect",
            "g4_signed_program_direction",
            "m0_program_concordance",
            "descriptive_expander_minus_nonexpander_effect",
        ],
    }
    if dict(endpoints) != expected_endpoints:
        raise ValueError("v7 intervention endpoint contract changed")

    contracts: dict[str, InterventionDatasetContract] = {}
    for slug in DATASET_SLUGS:
        record = datasets.get(slug)
        if not isinstance(record, Mapping):
            raise ValueError(f"v7 intervention config lacks dataset {slug!r}")
        truth_schema = record.get("truth_schema")
        truth_sha256 = record.get("truth_sha256")
        if slug == "kang":
            if truth_schema != "kang2018_ifnb_supportive_v1" or not _valid_sha256(
                truth_sha256
            ):
                raise ValueError("Kang truth contract changed")
        elif truth_schema is not None or truth_sha256 is not None:
            raise ValueError("BRCA real data cannot declare edge-level truth")
        hashes = (record.get("input_sha256"), record.get("input_manifest_sha256"))
        if any(not _valid_sha256(value) for value in hashes):
            raise ValueError(f"v7 intervention input hashes are invalid for {slug}")
        contracts[slug] = InterventionDatasetContract(
            slug=slug,
            dataset_id=str(record["dataset_id"]),
            role=str(record["role"]),
            condition_column=str(record["condition_column"]),
            reference=str(record["reference"]),
            target=str(record["target"]),
            contrast_name=str(record["contrast_name"]),
            seed=int(record["seed"]),
            outer_fold_partition_seed=int(record["outer_fold_partition_seed"]),
            allowed_outer_folds=tuple(map(int, record["allowed_outer_folds"])),
            input_schema=str(record["input_schema"]),
            input_sha256=str(record["input_sha256"]),
            input_manifest_sha256=str(record["input_manifest_sha256"]),
            cells=int(record["cells"]),
            genes=int(record["genes"]),
            samples=int(record["samples"]),
            subjects=int(record["subjects"]),
            cell_types=tuple(map(str, record["cell_types"])),
            truth_schema=(None if truth_schema is None else str(truth_schema)),
            truth_sha256=(None if truth_sha256 is None else str(truth_sha256)),
        )
    expected_contracts = {
        "kang": (
            "GSE96583_Kang2018_batch2",
            "legacy_development_supportive_silver_intervention",
            "ctrl",
            "stim",
            24673,
            32938,
            16,
            8,
        ),
        "brca_e": (
            "brca_anti_pd1__E_On_vs_Pre",
            "secondary_reused_real_intervention_descriptive",
            "PreE",
            "OnE",
            18465,
            24414,
            18,
            9,
        ),
        "brca_ne": (
            "brca_anti_pd1__NE_On_vs_Pre",
            "secondary_reused_real_intervention_descriptive",
            "PreNE",
            "OnNE",
            30243,
            24414,
            40,
            20,
        ),
    }
    observed_contracts = {
        slug: (
            item.dataset_id,
            item.role,
            item.reference,
            item.target,
            item.cells,
            item.genes,
            item.samples,
            item.subjects,
        )
        for slug, item in contracts.items()
    }
    if observed_contracts != expected_contracts:
        raise ValueError("v7 intervention dataset roles or dimensions changed")
    expected_folds = {"kang": (4, 2), "brca_e": (3, 2), "brca_ne": (3, 2)}
    if {
        slug: contract.allowed_outer_folds for slug, contract in contracts.items()
    } != expected_folds:
        raise ValueError("v7 intervention outer-fold policy changed")
    return config, contracts


def _validate_input_manifest(
    contract: InterventionDatasetContract,
    input_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if sha256_file(input_path) != contract.input_sha256:
        raise ValueError("v7 intervention H5AD checksum mismatch")
    if sha256_file(manifest_path) != contract.input_manifest_sha256:
        raise ValueError("v7 intervention input-manifest checksum mismatch")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != contract.input_schema:
        raise ValueError("v7 intervention input-manifest schema mismatch")
    if contract.slug == "kang":
        output = manifest.get("output")
        design = manifest.get("design")
        if (
            manifest.get("status") != "complete"
            or manifest.get("dataset_id") != contract.dataset_id
            or not isinstance(output, Mapping)
            or output.get("sha256") != contract.input_sha256
            or output.get("shape") != [contract.cells, contract.genes]
            or not isinstance(design, Mapping)
            or design.get("samples") != contract.samples
            or design.get("subjects") != contract.subjects
            or design.get("paired_subjects") != contract.subjects
            or design.get("sample_key") != "sample_id"
            or design.get("subject_key") != "subject_id"
            or design.get("condition_key") != contract.condition_column
            or design.get("cell_type_key") != "cell_type"
        ):
            raise ValueError("Kang preparation manifest differs from its contract")
    else:
        output = manifest.get("output")
        dimensions = manifest.get("dimensions")
        contrast = manifest.get("contrast")
        if (
            manifest.get("dataset_id") != contract.dataset_id
            or not isinstance(output, Mapping)
            or output.get("sha256") != contract.input_sha256
            or not isinstance(dimensions, Mapping)
            or dimensions.get("n_cells") != contract.cells
            or dimensions.get("n_genes") != contract.genes
            or dimensions.get("n_samples") != contract.samples
            or dimensions.get("n_subjects") != contract.subjects
            or not isinstance(contrast, Mapping)
            or contrast.get("pairing") != "subject_id"
            or contrast.get("reference") != contract.reference
            or contrast.get("target") != contract.target
        ):
            raise ValueError("BRCA preparation manifest differs from its contract")
    return manifest


def validate_paired_input(
    data: ad.AnnData,
    contract: InterventionDatasetContract,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate exact sample/subject pairing before any estimator is fitted."""

    if data.shape != (contract.cells, contract.genes):
        raise ValueError("v7 intervention H5AD dimensions changed")
    required = {"sample_id", "subject_id", contract.condition_column, "cell_type"}
    missing = required.difference(data.obs.columns)
    if missing:
        raise ValueError(f"v7 intervention metadata are missing: {sorted(missing)}")
    if "counts" not in data.layers:
        raise ValueError("v7 intervention input requires raw counts layer")
    if not data.var_names.is_unique:
        raise ValueError("v7 intervention gene identifiers must be unique")
    observed_cell_types = tuple(sorted(map(str, data.obs["cell_type"].unique())))
    if observed_cell_types != tuple(sorted(contract.cell_types)):
        raise ValueError("v7 intervention cell-type universe changed")
    metadata = data.obs.loc[
        :, ["sample_id", "subject_id", contract.condition_column]
    ].copy()
    for column in metadata.columns:
        metadata[column] = metadata[column].astype(str)
    sample_map = metadata.drop_duplicates()
    if sample_map["sample_id"].duplicated().any():
        raise ValueError("one sample maps to multiple subjects or conditions")
    if len(sample_map) != contract.samples:
        raise ValueError("v7 intervention sample count changed")
    observed_conditions = set(sample_map[contract.condition_column])
    expected_conditions = {contract.reference, contract.target}
    if observed_conditions != expected_conditions:
        raise ValueError("v7 intervention condition axis changed")
    subject_conditions = sample_map.groupby("subject_id", observed=True)[
        contract.condition_column
    ].agg(lambda values: tuple(sorted(set(map(str, values)))))
    if (
        len(subject_conditions) != contract.subjects
        or any(set(values) != expected_conditions for values in subject_conditions)
        or sample_map.groupby("subject_id", observed=True).size().ne(2).any()
    ):
        raise ValueError("v7 intervention requires one complete pair per subject")
    return (
        sample_map.sort_values("sample_id", kind="stable", ignore_index=True),
        {
            "cells": int(data.n_obs),
            "genes": int(data.n_vars),
            "samples": len(sample_map),
            "subjects": len(subject_conditions),
            "paired_subjects": len(subject_conditions),
            "conditions": sorted(observed_conditions),
            "cell_types": list(observed_cell_types),
            "counts_dtype": str(data.layers["counts"].dtype),
        },
    )


def build_intervention_crossfit_spec(
    contract: InterventionDatasetContract,
    estimator: Mapping[str, object],
) -> tuple[CrychicConfig, CrossFitSpec]:
    """Build the frozen paired v7-primary cross-fit policy."""

    contrast = balanced_contrast(
        (contract.target,),
        (contract.reference,),
        name=contract.contrast_name,
    )
    config = CrychicConfig(
        context_keys=(contract.condition_column,),
        counts_layer="counts",
        design=f"~ {contract.condition_column}",
        random_seed=contract.seed,
    )
    spec = CrossFitSpec(
        contrasts=(contrast,),
        predeclared_receiver_ids=contract.cell_types,
        outer_fold_partition_seed=contract.outer_fold_partition_seed,
        training_spec=FoldTrainingSpec(
            min_cells=int(estimator["minimum_cells_per_sample_cell_type"]),
            min_pooled_availability=float(estimator["minimum_pooled_availability"]),
            max_interactions=None,
            sender_parameters=ContrastCommonSenderParameters(
                min_subjects=int(estimator["minimum_common_sender_subjects"]),
                contrast_unit="paired_subject",
            ),
        ),
        allowed_n_splits=contract.allowed_outer_folds,
        min_train_subjects_per_context=int(
            estimator["minimum_train_subjects_per_context"]
        ),
        min_test_subjects_per_context=int(
            estimator["minimum_test_subjects_per_context"]
        ),
        absolute_activity_v2_spec=AbsoluteActivityV2Spec(
            bottleneck_penalty=float(estimator["bottleneck_penalty"]),
        ),
        sender_attribution_v2_spec=SenderAttributionV2Spec(
            minimum_calibration_subjects=int(
                estimator["minimum_sender_calibration_subjects"]
            ),
        ),
        signed_program_v2_spec=SignedProgramV2Spec(
            minimum_training_samples=int(estimator["minimum_signed_program_samples"]),
            minimum_training_subjects=int(estimator["minimum_signed_program_subjects"]),
        ),
        eb_shrunken_coupling_v2_spec=EBShrunkenCouplingV2Spec(
            contrast_name=contract.contrast_name,
            minimum_subjects=int(estimator["minimum_coupling_subjects"]),
            minimum_observed_edges_for_eb=int(
                estimator["minimum_coupling_edges_for_eb"]
            ),
            attribution_coupling_weight=float(estimator["attribution_coupling_weight"]),
        ),
    )
    return config, spec


def build_intervention_design(
    contract: InterventionDatasetContract,
    estimator: Mapping[str, object],
) -> DifferentialDesignSpec:
    """Return paired target-minus-reference I1 inference."""

    return DifferentialDesignSpec(
        design_kind="paired",
        condition_column=contract.condition_column,
        condition_levels=(contract.reference, contract.target),
        contrasts=(
            DifferentialContrastSpec(
                name=contract.contrast_name,
                weights=((contract.reference, -1.0), (contract.target, 1.0)),
            ),
        ),
        precision_weight_column=None,
        minimum_subjects_per_level=int(estimator["minimum_subjects_per_level"]),
        minimum_clusters_for_cr2=int(estimator["minimum_clusters_for_cr2"]),
    )


def _inference_score_views(
    score_views: pd.DataFrame,
    estimator: Mapping[str, object],
) -> pd.DataFrame:
    expected = {key: list(value) for key, value in INFERENCE_SCORE_VIEWS.items()}
    if estimator.get("inference_score_views") != expected:
        raise ValueError("v7 intervention inference score-view contract changed")
    selected = score_views.loc[
        [
            str(view) in INFERENCE_SCORE_VIEWS.get(str(generator), ())
            for generator, view in score_views.loc[
                :, ["generator_id", "score_view"]
            ].itertuples(index=False, name=None)
        ]
    ].copy()
    observed = (
        selected.loc[:, ["generator_id", "score_view"]]
        .drop_duplicates()
        .groupby("generator_id", observed=True, sort=True)["score_view"]
        .agg(lambda values: tuple(sorted(map(str, values))))
        .to_dict()
    )
    expected_observed = {
        generator: tuple(sorted(views))
        for generator, views in INFERENCE_SCORE_VIEWS.items()
    }
    if observed != expected_observed:
        raise ValueError(f"v7 intervention score-view matrix is incomplete: {observed}")
    return selected.reset_index(drop=True)


def _crossfit_manifest(crossfit: _V7PrimaryCrossFitArtifacts) -> dict[str, object]:
    crossfit._require_intact()
    return {
        "crossfit_id": crossfit.crossfit_id,
        "execution_profile": crossfit.execution_profile,
        "spec_id": crossfit.spec.spec_id,
        "fold_plan_id": crossfit.fold_plan.plan_id,
        "effective_n_splits": crossfit.fold_plan.effective_n_splits,
        "fold_ids": [fold.fold_id for fold in crossfit.folds],
        "oof_sample_edge_score_table_digest": (
            crossfit.oof_sample_edge_score_table_digest
        ),
        "formal_inference_allowed": False,
    }


def _score_geometry(score_views: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    keys = ["generator_id", "score_view", "resolution"]
    for key, group in score_views.groupby(keys, observed=True, sort=True):
        values = pd.to_numeric(group["score"], errors="coerce")
        finite = values[np.isfinite(values)]
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "rows": len(group),
                "finite_rows": len(finite),
                "na_fraction": float(values.isna().mean()),
                "zero_fraction": float(values.eq(0.0).mean()),
                "unique_finite_values": int(finite.nunique()),
                "tie_fraction": (
                    math.nan
                    if finite.empty
                    else float(1.0 - finite.nunique() / len(finite))
                ),
                "median": math.nan if finite.empty else float(finite.median()),
                "iqr": (
                    math.nan
                    if finite.empty
                    else float(finite.quantile(0.75) - finite.quantile(0.25))
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _effect_summary(effects: pd.DataFrame) -> pd.DataFrame:
    return (
        effects.assign(observed=effects["status"].astype(str).eq("observed"))
        .groupby(
            ["generator_id", "score_view", "resolution", "inference_id"],
            observed=True,
            sort=True,
        )
        .agg(
            rows=("event_id", "size"),
            observed_rows=("observed", "sum"),
            median_abs_effect=("effect", lambda values: values.abs().median()),
            median_ranking_score=("ranking_score", "median"),
        )
        .reset_index()
    )


def summarize_mechanism_effects(
    effects: pd.DataFrame,
    contract: InterventionDatasetContract,
) -> pd.DataFrame:
    """Summarize signed-program direction without treating it as edge truth."""

    required = {
        "generator_id",
        "score_view",
        "inference_id",
        "contrast_name",
        "event_id",
        "receiver",
        "effect",
        "status",
        "p_value",
        "q_value",
        "formal_inference_allowed",
    }
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"mechanism effects are missing columns: {sorted(missing)}")
    if effects["formal_inference_allowed"].astype(bool).any():
        raise ValueError("real intervention mechanism summary cannot use formal tests")
    if effects[["p_value", "q_value"]].notna().any(axis=None):
        raise ValueError("real intervention mechanism summary requires withheld p/q")
    program = effects.loc[
        effects["generator_id"].eq("G4")
        & effects["score_view"].eq("program_signed_component")
        & effects["inference_id"].eq("I1")
        & effects["contrast_name"].eq(contract.contrast_name)
    ].copy()
    intensity = effects.loc[
        effects["generator_id"].eq("G4")
        & effects["score_view"].eq("m0_parent_mean_component")
        & effects["inference_id"].eq("I1")
        & effects["contrast_name"].eq(contract.contrast_name)
    ].copy()
    if program.empty or intensity.empty:
        raise ValueError("v7 intervention lacks G4 intensity or program effects")
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all_receivers", program)]
    scopes.extend(
        (str(receiver), group)
        for receiver, group in program.groupby("receiver", observed=True, sort=True)
    )
    for receiver, group in scopes:
        observed = group.loc[group["status"].astype(str).eq("observed")].copy()
        values = pd.to_numeric(observed["effect"], errors="coerce")
        values = values[np.isfinite(values)]
        rows.append(
            {
                "dataset": contract.slug,
                "contrast_name": contract.contrast_name,
                "receiver": receiver,
                "metric": "g4_signed_program_positive_fraction",
                "value": math.nan if values.empty else float(values.gt(0.0).mean()),
                "numerator": int(values.gt(0.0).sum()),
                "denominator": len(values),
                "status": "observed" if not values.empty else "not_estimable",
                "claim_scope": "supportive_receiver_mechanism_direction_only",
            }
        )
    merge_keys = ["event_id", "contrast_name", "receiver", "interaction_id"]
    merged = intensity.loc[:, [*merge_keys, "effect", "status"]].merge(
        program.loc[:, [*merge_keys, "effect", "status"]],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_intensity", "_program"),
        sort=False,
    )
    observed = merged.loc[
        merged["status_intensity"].eq("observed")
        & merged["status_program"].eq("observed")
    ].copy()
    left = pd.to_numeric(observed["effect_intensity"], errors="coerce")
    right = pd.to_numeric(observed["effect_program"], errors="coerce")
    finite = np.isfinite(left) & np.isfinite(right)
    left = left.loc[finite]
    right = right.loc[finite]
    sign_concordance = (
        math.nan if left.empty else float(np.sign(left).eq(np.sign(right)).mean())
    )
    rank_concordance = math.nan
    if len(left) >= 3 and left.nunique() > 1 and right.nunique() > 1:
        rank_concordance = float(spearmanr(left, right).statistic)
    for metric, value in (
        ("m0_program_sign_concordance", sign_concordance),
        ("m0_program_effect_spearman", rank_concordance),
    ):
        rows.append(
            {
                "dataset": contract.slug,
                "contrast_name": contract.contrast_name,
                "receiver": "all_receivers",
                "metric": metric,
                "value": value,
                "numerator": None,
                "denominator": len(left),
                "status": "observed" if math.isfinite(value) else "not_estimable",
                "claim_scope": "supportive_receiver_mechanism_direction_only",
            }
        )
    return pd.DataFrame.from_records(rows)


def _selected_gene_pseudobulk(
    data: ad.AnnData,
    *,
    genes: Sequence[str],
    condition_column: str,
) -> pd.DataFrame:
    requested = tuple(dict.fromkeys(map(str, genes)))
    observed_genes = tuple(gene for gene in requested if gene in data.var_names)
    if not observed_genes:
        return pd.DataFrame(
            columns=(
                "sample_id",
                "subject_id",
                condition_column,
                "cell_type",
                "gene",
                "log1p_cpm",
            )
        )
    counts = data.layers["counts"]
    library_by_cell = np.asarray(counts.sum(axis=1)).reshape(-1).astype(float)
    positions = [int(data.var_names.get_loc(gene)) for gene in observed_genes]
    selected = counts[:, positions]
    selected_array = (
        selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
    ).astype(float, copy=False)
    obs = data.obs.loc[
        :, ["sample_id", "subject_id", condition_column, "cell_type"]
    ].copy()
    labels = obs["sample_id"].astype(str) + "\x1f" + obs["cell_type"].astype(str)
    codes, unique_labels = pd.factorize(labels, sort=True)
    libraries = np.bincount(
        codes, weights=library_by_cell, minlength=len(unique_labels)
    )
    gene_sums = np.column_stack(
        [
            np.bincount(
                codes, weights=selected_array[:, index], minlength=len(unique_labels)
            )
            for index in range(len(observed_genes))
        ]
    )
    group_meta = obs.assign(_group=labels).drop_duplicates("_group")
    group_meta = group_meta.set_index("_group").reindex(unique_labels).reset_index()
    records: list[dict[str, object]] = []
    for group_index, row in group_meta.iterrows():
        library = float(libraries[group_index])
        for gene_index, gene in enumerate(observed_genes):
            value = (
                math.nan
                if library <= 0.0
                else float(
                    np.log1p(1.0e6 * gene_sums[group_index, gene_index] / library)
                )
            )
            records.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "subject_id": str(row["subject_id"]),
                    condition_column: str(row[condition_column]),
                    "cell_type": str(row["cell_type"]),
                    "gene": gene,
                    "log1p_cpm": value,
                }
            )
    return pd.DataFrame.from_records(records)


def paired_gene_effects(
    data: ad.AnnData,
    contract: InterventionDatasetContract,
    truth: Mapping[str, object],
) -> pd.DataFrame:
    """Calculate preregistered raw-count paired gene-response diagnostics."""

    raw_observations = truth.get("expected_observations")
    if not isinstance(raw_observations, Sequence):
        raise ValueError("Kang truth lacks expected_observations")
    programs: list[Mapping[str, object]] = []
    all_genes: list[str] = []
    for item in raw_observations:
        if isinstance(item, Mapping) and item.get("level") == "receiver_gene_response":
            genes = item.get("genes")
            if not isinstance(genes, Sequence) or isinstance(genes, str):
                raise ValueError("Kang receiver-gene truth requires gene lists")
            programs.append(item)
            all_genes.extend(map(str, genes))
    pseudobulk = _selected_gene_pseudobulk(
        data,
        genes=all_genes,
        condition_column=contract.condition_column,
    )
    records: list[dict[str, object]] = []
    for program in programs:
        expectation_id = str(program["id"])
        genes = tuple(map(str, cast(Sequence[object], program["genes"])))
        declared_receivers = program.get("receiver_cell_types")
        receivers = (
            contract.cell_types
            if declared_receivers is None
            else tuple(map(str, cast(Sequence[object], declared_receivers)))
        )
        for receiver in receivers:
            for gene in genes:
                local = pseudobulk.loc[
                    pseudobulk["cell_type"].eq(receiver) & pseudobulk["gene"].eq(gene)
                ]
                if local.empty:
                    differences = np.asarray([], dtype=float)
                    reason = (
                        "gene_not_in_input"
                        if gene not in data.var_names
                        else "receiver_gene_not_observed"
                    )
                else:
                    matrix = local.pivot(
                        index="subject_id",
                        columns=contract.condition_column,
                        values="log1p_cpm",
                    ).reindex(columns=[contract.reference, contract.target])
                    complete = matrix.dropna()
                    differences = (
                        complete[contract.target] - complete[contract.reference]
                    ).to_numpy(dtype=float)
                    reason = None
                observed = len(differences) >= 4
                records.append(
                    {
                        "truth_set_id": str(truth["truth_set_id"]),
                        "expectation_id": expectation_id,
                        "gene": gene,
                        "receiver": receiver,
                        "contrast_name": contract.contrast_name,
                        "n_complete_pairs": len(differences),
                        "mean_effect": (
                            math.nan if not observed else float(np.mean(differences))
                        ),
                        "median_effect": (
                            math.nan if not observed else float(np.median(differences))
                        ),
                        "positive_pair_fraction": (
                            math.nan
                            if not observed
                            else float(np.mean(differences > 0.0))
                        ),
                        "status": "observed" if observed else "not_estimable",
                        "reason_code": None
                        if observed
                        else reason or "fewer_than_four_pairs",
                        "claim_scope": "supportive_receiver_gene_response_only",
                    }
                )
    return pd.DataFrame.from_records(records)


def paired_composition_stability(
    data: ad.AnnData,
    contract: InterventionDatasetContract,
) -> pd.DataFrame:
    """Measure within-donor cell-type composition stability."""

    grouped = (
        data.obs.assign(
            sample_id=data.obs["sample_id"].astype(str),
            subject_id=data.obs["subject_id"].astype(str),
            _condition=data.obs[contract.condition_column].astype(str),
            cell_type=data.obs["cell_type"].astype(str),
        )
        .groupby(
            ["sample_id", "subject_id", "_condition", "cell_type"],
            observed=True,
            sort=True,
        )
        .size()
        .rename("cells")
        .reset_index()
    )
    grouped["fraction"] = grouped["cells"] / grouped.groupby(
        "sample_id", observed=True
    )["cells"].transform("sum")
    rows: list[dict[str, object]] = []
    for subject, local in grouped.groupby("subject_id", observed=True, sort=True):
        matrix = (
            local.pivot_table(
                index="cell_type",
                columns="_condition",
                values="fraction",
                fill_value=0.0,
                observed=True,
            )
            .reindex(index=contract.cell_types, fill_value=0.0)
            .reindex(columns=[contract.reference, contract.target], fill_value=0.0)
        )
        reference = matrix[contract.reference].to_numpy(dtype=float)
        target = matrix[contract.target].to_numpy(dtype=float)
        correlation = math.nan
        if np.unique(reference).size > 1 and np.unique(target).size > 1:
            correlation = float(spearmanr(reference, target).statistic)
        rows.append(
            {
                "subject_id": str(subject),
                "contrast_name": contract.contrast_name,
                "cell_types": len(contract.cell_types),
                "spearman": correlation,
                "mean_absolute_fraction_change": float(
                    np.mean(np.abs(target - reference))
                ),
                "maximum_absolute_fraction_change": float(
                    np.max(np.abs(target - reference))
                ),
                "status": "observed" if math.isfinite(correlation) else "not_estimable",
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_kang_truth(
    truth: Mapping[str, object],
    gene_effects: pd.DataFrame,
    composition: pd.DataFrame,
    contract: InterventionDatasetContract,
) -> pd.DataFrame:
    """Apply the preregistered supportive-silver acceptance thresholds."""

    observations = truth.get("expected_observations")
    if not isinstance(observations, Sequence):
        raise ValueError("Kang truth lacks expected observations")
    records: list[dict[str, object]] = []
    for item in observations:
        if not isinstance(item, Mapping):
            raise ValueError("Kang truth observations must be mappings")
        expectation_id = str(item["id"])
        level = str(item["level"])
        acceptance = item.get("acceptance")
        acceptance_map = acceptance if isinstance(acceptance, Mapping) else {}
        if level == "data_contract":
            records.append(
                {
                    "expectation_id": expectation_id,
                    "metric": "complete_paired_subjects",
                    "value": contract.subjects,
                    "threshold": contract.subjects,
                    "passed": True,
                    "status": "observed",
                    "claim_scope": "data_contract",
                }
            )
        elif level == "composition":
            finite = composition.loc[composition["status"].eq("observed")]
            median_correlation = float(finite["spearman"].median())
            median_change = float(finite["mean_absolute_fraction_change"].median())
            correlation_threshold = float(acceptance_map["median_subject_spearman_min"])
            change_threshold = float(
                acceptance_map["median_absolute_fraction_change_max"]
            )
            for metric, value, threshold, passed in (
                (
                    "median_subject_spearman",
                    median_correlation,
                    correlation_threshold,
                    median_correlation >= correlation_threshold,
                ),
                (
                    "median_absolute_fraction_change",
                    median_change,
                    change_threshold,
                    median_change <= change_threshold,
                ),
            ):
                records.append(
                    {
                        "expectation_id": expectation_id,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "passed": bool(passed),
                        "status": "observed"
                        if math.isfinite(value)
                        else "not_estimable",
                        "claim_scope": "supportive_composition_stability_only",
                    }
                )
        elif level == "receiver_gene_response":
            local = gene_effects.loc[
                gene_effects["expectation_id"].eq(expectation_id)
                & gene_effects["status"].eq("observed")
            ]
            positive_fraction = (
                math.nan
                if local.empty
                else float(pd.to_numeric(local["mean_effect"]).gt(0.0).mean())
            )
            consistency = (
                math.nan
                if local.empty
                else float(pd.to_numeric(local["positive_pair_fraction"]).median())
            )
            positive_threshold = float(
                acceptance_map["eligible_gene_celltype_positive_fraction_min"]
            )
            metrics = [
                (
                    "eligible_gene_celltype_positive_fraction",
                    positive_fraction,
                    positive_threshold,
                    positive_fraction >= positive_threshold,
                )
            ]
            if "donor_direction_consistency_median_min" in acceptance_map:
                consistency_threshold = float(
                    acceptance_map["donor_direction_consistency_median_min"]
                )
                metrics.append(
                    (
                        "donor_direction_consistency_median",
                        consistency,
                        consistency_threshold,
                        consistency >= consistency_threshold,
                    )
                )
            for metric, value, threshold, passed in metrics:
                records.append(
                    {
                        "expectation_id": expectation_id,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "passed": bool(passed) if math.isfinite(value) else False,
                        "status": "observed"
                        if math.isfinite(value)
                        else "not_estimable",
                        "claim_scope": "supportive_receiver_gene_response_only",
                    }
                )
        elif level == "interpretation_guardrail":
            records.append(
                {
                    "expectation_id": expectation_id,
                    "metric": "endogenous_ifnb_sender_recovery_required",
                    "value": 0.0,
                    "threshold": 0.0,
                    "passed": True,
                    "status": "guardrail_applied",
                    "claim_scope": "no_causal_sender_claim_from_exogenous_stimulus",
                }
            )
    return pd.DataFrame.from_records(records)


def _resource_compatibility(
    data: ad.AnnData,
    bundle: ResourceBundle,
    target_prior: TargetPrior,
) -> dict[str, object]:
    genes = set(map(str, data.var_names))
    measurable = sum(
        set(interaction.ligand_subunits).issubset(genes)
        and set(interaction.receptor_subunits).issubset(genes)
        for interaction in bundle.interactions
    )
    return {
        "resource_interactions": len(bundle.interactions),
        "fully_observable_lr_interactions": measurable,
        "fully_observable_lr_fraction": measurable / len(bundle.interactions),
        "observable_target_prior_drivers": len(set(target_prior.driver_ids) & genes),
        "target_prior_drivers": len(target_prior.driver_ids),
        "observable_target_prior_targets": len(set(target_prior.target_ids) & genes),
        "target_prior_targets": len(target_prior.target_ids),
    }


def _load_target_prior(
    database_root: Path,
    nichenet_manifest: Path,
    prior_contract: Mapping[str, object],
) -> TargetPrior:
    release = str(prior_contract["release"])
    native_manifest = database_root / "nichenet" / release / "manifest.json"
    if nichenet_manifest.resolve() != native_manifest.resolve():
        raise ValueError("v7 intervention requires the native NicheNet manifest path")
    if sha256_file(native_manifest) != prior_contract["manifest_sha256"]:
        raise ValueError("v7 intervention NicheNet manifest checksum mismatch")
    return load_nichenet_target_prior(database_root, release=release)


def run(
    *,
    dataset: str,
    input_h5ad: Path,
    input_manifest: Path,
    output_dir: Path,
    resource_path: Path,
    resource_manifest: Path,
    database_root: Path,
    nichenet_manifest: Path,
    truth_path: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    threads: int = 1,
    fold_jobs: int = 1,
    overwrite: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Fit one frozen paired intervention cohort and persist bounded evidence."""

    if dataset not in DATASET_SLUGS:
        raise ValueError(f"dataset must be one of {DATASET_SLUGS}")
    for name, value in (("threads", threads), ("fold_jobs", fold_jobs)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    protocol, contracts = load_intervention_config(config_path)
    contract = contracts[dataset]
    estimator = cast(Mapping[str, object], protocol["estimator"])
    resource_contract = cast(Mapping[str, object], protocol["resource"])
    prior_contract = cast(Mapping[str, object], protocol["target_prior"])
    if dataset == "kang" and truth_path is None:
        raise ValueError("Kang intervention validation requires its frozen truth file")
    if dataset != "kang" and truth_path is not None:
        raise ValueError("BRCA real intervention runs cannot declare edge-level truth")
    code = git_metadata(REPOSITORY_ROOT)
    if code["dirty"] and not allow_dirty:
        raise RuntimeError("v7 intervention benchmark refuses a dirty worktree")
    _validate_input_manifest(contract, input_h5ad, input_manifest)
    if sha256_file(resource_path) != resource_contract["table_sha256"]:
        raise ValueError("v7 intervention H-common table checksum mismatch")
    if sha256_file(resource_manifest) != resource_contract["manifest_sha256"]:
        raise ValueError("v7 intervention H-common manifest checksum mismatch")
    if sha256_file(nichenet_manifest) != prior_contract["manifest_sha256"]:
        raise ValueError("v7 intervention NicheNet manifest checksum mismatch")
    truth: dict[str, Any] | None = None
    if truth_path is not None:
        if sha256_file(truth_path) != contract.truth_sha256:
            raise ValueError("Kang supportive truth checksum mismatch")
        truth = _read_yaml(truth_path)
        if (
            truth.get("truth_set_id") != contract.truth_schema
            or truth.get("status") != "supportive_silver_standard"
            or cast(Mapping[str, object], truth.get("evaluation_policy", {})).get(
                "do_not_compute_edge_auroc"
            )
            is not True
        ):
            raise ValueError("Kang supportive truth release boundary changed")

    output = prepare_output(output_dir, overwrite=overwrite)
    log_path = output / "run.jsonl"
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "dataset": dataset,
        "dataset_id": contract.dataset_id,
        "dataset_role": contract.role,
        "analysis_unit": {
            "sample_key": "sample_id",
            "subject_key": "subject_id",
            "pairing": "complete_within_subject",
            "contrast": "target_minus_reference",
        },
        "formal_inference_allowed": False,
        "inputs": {
            "h5ad": _input_record(input_h5ad),
            "preparation_manifest": _input_record(input_manifest),
            "protocol": _input_record(config_path),
            "resource": _input_record(resource_path),
            "resource_manifest": _input_record(resource_manifest),
            "nichenet_manifest": _input_record(nichenet_manifest),
            "truth": None if truth_path is None else _input_record(truth_path),
        },
        "parameters": {"threads": threads, "fold_jobs": fold_jobs},
        "code": code,
        "outputs": None,
        "failure": None,
    }
    write_json(output / "manifest.json", manifest)
    _append_log(
        log_path,
        "run_started",
        elapsed_seconds=0.0,
        dataset=dataset,
        threads=threads,
        fold_jobs=fold_jobs,
    )
    data: ad.AnnData | None = None
    try:
        bundle = harmonized_resource_bundle(resource_path, resource_manifest)
        if (
            bundle.resource_id != resource_contract["resource_id"]
            or len(bundle.interactions) != resource_contract["interactions"]
        ):
            raise ValueError("v7 intervention in-memory H-common resource changed")
        target_prior = _load_target_prior(
            database_root, nichenet_manifest, prior_contract
        )
        _append_log(
            log_path,
            "resources_loaded",
            elapsed_seconds=time.perf_counter() - started,
            interactions=len(bundle.interactions),
            target_prior_drivers=len(target_prior.driver_ids),
        )
        data = ad.read_h5ad(input_h5ad)
        sample_metadata, input_audit = validate_paired_input(data, contract)
        compatibility = _resource_compatibility(data, bundle, target_prior)
        _append_log(
            log_path,
            "dataset_validated",
            elapsed_seconds=time.perf_counter() - started,
            cells=data.n_obs,
            genes=data.n_vars,
            samples=len(sample_metadata),
            subjects=contract.subjects,
        )
        crychic_config, crossfit_spec = build_intervention_crossfit_spec(
            contract, estimator
        )
        design = build_intervention_design(contract, estimator)
        snapshot = _sanitized_raw_input_snapshot(data, crychic_config)
        with threadpool_limits(limits=threads):
            crossfit = _run_v7_primary_crossfit(
                snapshot,
                crychic_config,
                bundle,
                target_prior,
                spec=crossfit_spec,
                n_jobs=fold_jobs,
            )
        _append_log(
            log_path,
            "crossfit_completed",
            elapsed_seconds=time.perf_counter() - started,
            effective_n_splits=crossfit.fold_plan.effective_n_splits,
        )
        score_views = build_v7_primary_score_views(
            crossfit, dataset_id=contract.dataset_id
        )
        inference_scores = _inference_score_views(score_views, estimator)
        fit_cache = V7InferenceFitCache()
        effects = run_v7_inference_matrix(
            inference_scores,
            design=design,
            sample_metadata=sample_metadata,
            arms=tuple((generator, "I1") for generator in INFERENCE_SCORE_VIEWS),
            fit_cache=fit_cache,
        )
        if effects["formal_inference_allowed"].astype(bool).any() or effects[
            ["p_value", "q_value"]
        ].notna().any(axis=None):
            raise RuntimeError(
                "real intervention I1 accidentally released formal tests"
            )
        _append_log(
            log_path,
            "paired_inference_completed",
            elapsed_seconds=time.perf_counter() - started,
            score_rows=len(score_views),
            inference_score_rows=len(inference_scores),
            effect_rows=len(effects),
        )
        mechanism = summarize_mechanism_effects(effects, contract)
        tables: dict[str, pd.DataFrame] = {
            "score_views.parquet": score_views,
            "effects.parquet": effects,
            "score_geometry.tsv": _score_geometry(score_views),
            "effect_summary.tsv": _effect_summary(effects),
            "mechanism_summary.tsv": mechanism,
        }
        truth_summary: pd.DataFrame | None = None
        if dataset == "kang":
            assert truth is not None
            gene_effect_table = paired_gene_effects(data, contract, truth)
            composition = paired_composition_stability(data, contract)
            truth_summary = summarize_kang_truth(
                truth, gene_effect_table, composition, contract
            )
            tables.update(
                {
                    "paired_gene_effects.tsv": gene_effect_table,
                    "paired_composition_stability.tsv": composition,
                    "supportive_truth_summary.tsv": truth_summary,
                }
            )
            _append_log(
                log_path,
                "supportive_truth_evaluated",
                elapsed_seconds=time.perf_counter() - started,
                truth_rows=len(truth_summary),
                passed=int(truth_summary["passed"].astype(bool).sum()),
            )
        output_records: dict[str, object] = {}
        for filename, table in tables.items():
            path = output / filename
            if path.suffix == ".parquet":
                table.to_parquet(path, index=False, compression="zstd")
            else:
                table.to_csv(path, sep="\t", index=False)
            output_records[filename] = _output_record(path, table)
        _append_log(
            log_path,
            "run_completed",
            elapsed_seconds=time.perf_counter() - started,
            output_tables=len(tables),
        )
        output_records[log_path.name] = {
            "filename": log_path.name,
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
        }
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - started,
                "input_audit": input_audit,
                "resource": {
                    "resource_id": bundle.resource_id,
                    "version": bundle.version,
                    "manifest_digest": bundle.manifest_digest,
                    "interactions": len(bundle.interactions),
                    "target_prior_resource_id": target_prior.resource_id,
                    "target_prior_manifest_digest": target_prior.manifest_digest,
                },
                "resource_compatibility": compatibility,
                "crossfit": _crossfit_manifest(crossfit),
                "crossfit_spec": crossfit_spec.to_dict(),
                "differential_design": design.to_dict(),
                "inference_cache": fit_cache.to_dict(),
                "inference_score_views": {
                    key: list(value) for key, value in INFERENCE_SCORE_VIEWS.items()
                },
                "g0_g2_equivalence": g0_g2_equivalence_diagnostic(score_views),
                "truth": (
                    None
                    if truth is None
                    else {
                        "truth_set_id": truth["truth_set_id"],
                        "status": truth["status"],
                        "edge_auroc_computed": False,
                        "acceptance_rows": len(cast(pd.DataFrame, truth_summary)),
                    }
                ),
                "release": dict(cast(Mapping[str, object], protocol["release"])),
                "environment": python_environment(
                    environment_name="crychic_project_python",
                    packages=(
                        "CRYCHIC",
                        "anndata",
                        "numpy",
                        "pandas",
                        "pyarrow",
                        "scipy",
                        "threadpoolctl",
                    ),
                    threads=threads,
                ),
                "outputs": output_records,
            }
        )
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        _append_log(
            log_path,
            "run_failed",
            elapsed_seconds=time.perf_counter() - started,
            error_type=f"{type(error).__module__}.{type(error).__qualname__}",
            message=str(error),
        )
        manifest.update(
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "failure": {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": str(error),
                },
            }
        )
        write_json(output / "manifest.json", manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_SLUGS, required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--nichenet-manifest", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fold-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = run(
        dataset=arguments.dataset,
        input_h5ad=arguments.input_h5ad.resolve(),
        input_manifest=arguments.input_manifest.resolve(),
        output_dir=arguments.output_dir.resolve(),
        resource_path=arguments.resource.resolve(),
        resource_manifest=arguments.resource_manifest.resolve(),
        database_root=arguments.database_root.resolve(),
        nichenet_manifest=arguments.nichenet_manifest.resolve(),
        truth_path=(None if arguments.truth is None else arguments.truth.resolve()),
        config_path=arguments.config.resolve(),
        threads=arguments.threads,
        fold_jobs=arguments.fold_jobs,
        overwrite=arguments.overwrite,
        allow_dirty=arguments.allow_dirty,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
