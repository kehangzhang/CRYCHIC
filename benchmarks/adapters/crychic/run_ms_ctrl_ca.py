"""Run the formal CRYCHIC cross-fit for Lerma-Martin MS Ctrl versus CA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import anndata as ad
import pandas as pd

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    python_environment,
    sha256_file,
    write_json,
)
from benchmarks.adapters.crychic.des_postprocess import (
    adjusted_subject_directed_lr_effects,
    condition_ranking_diagnostics,
    heldout_sample_coverage_audit,
    score_reason_waterfall,
    sender_specific_direct_scores,
    stable_breadth_unordered_cell_pair_rankings,
    subject_equal_directed_lr_effects,
    unordered_cell_pair_des_rankings,
)
from benchmarks.adapters.crychic.run_kuppe_ctrl_iz import (
    load_connectomedb2020_bundle,
)
from benchmarks.adapters.crychic.score_layers import (
    SCORE_LAYER_VALUE_COLUMNS,
    build_multigroup_score_layers,
    summarize_score_layer,
)
from crychic import Crychic, CrychicConfig
from crychic.attribution import GainCalibrationSpec, PenaltyTuningSpec
from crychic.data import InputSchema, validate_anndata
from crychic.design import balanced_contrast
from crychic.resources import (
    GeneNamespace,
    ResourceBundle,
    Species,
    TargetPrior,
    load_nichenet_target_prior,
)
from crychic.scoring import FrozenLatentNuisanceSpec
from crychic.sender import ContrastCommonSenderParameters
from crychic.workflow import CrossFitSpec, FoldTrainingSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ID = "UCSC_Lerma_Martin_MS_snRNA_CA_vs_Ctrl"
DES_DATASET_ID = "LermaMartin_MS_CA_vs_Ctrl"
PREPARATION_SCHEMA = "crychic-prepared-subset-v1"
DOWNSAMPLE_PREPARATION_SCHEMA = "crychic-ms-ctrl-ca-downsample-v1"
FIGURE3_PREPARATION_SCHEMA = "crychic-figure3-ms-paper-matched-subset-v1"
RUN_SCHEMA = "crychic-ms-ctrl-ca-crossfit-run-v3"
CONTRAST = "CA_vs_Ctrl"
REFERENCE = "Ctrl"
TARGET = "CA"
OUTER_FOLDS = 2
EXPECTED_SHAPE = (75_004, 32_115)
EXPECTED_OUTPUT_SHA256 = (
    "612fe9c4cdaf88694a47e13ba4458c828f206cd945eba9e9c72f46e9bd0196c7"
)
EXPECTED_SOURCE_SHA256 = (
    "3019c05759439ceacff29db2b80dae79beb1b38687f574581803d1da51a2d3ba"
)
EXPECTED_SAMPLES = {REFERENCE: 6, TARGET: 6}
EXPECTED_SUBJECTS = {REFERENCE: 6, TARGET: 5}
EXPECTED_SAMPLES_PER_SUBJECT = {
    REFERENCE: (1, 1, 1, 1, 1, 1),
    TARGET: (1, 1, 1, 1, 2),
}
FIGURE3_EXPECTED_SHAPE = (69_168, 32_115)
FIGURE3_EXPECTED_OUTPUT_SHA256 = (
    "433717d9fd98e57e15a444a338a6e1002f7ca3224022321d28a386c0a8498e1c"
)
FIGURE3_EXPECTED_SAMPLES = {REFERENCE: 5, TARGET: 6}
FIGURE3_EXPECTED_SUBJECTS = {REFERENCE: 5, TARGET: 5}
FIGURE3_EXPECTED_SAMPLES_PER_SUBJECT = {
    REFERENCE: (1, 1, 1, 1, 1),
    TARGET: (1, 1, 1, 1, 2),
}
SCORE_FILENAME = "sender_lr_scores.parquet"
SCORE_LAYER_FILENAME = "sender_lr_score_layers.parquet"
DIRECTED_EFFECT_FILENAME = "directed_lr_effects.parquet"
MECHANISTIC_DIRECTED_EFFECT_FILENAME = "mechanistic_directed_lr_effects.parquet"
SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME = "sender_specific_directed_lr_effects.parquet"
UNORDERED_RANKING_FILENAME = "condition_cell_pair_rankings.tsv"
MECHANISTIC_UNORDERED_RANKING_FILENAME = "mechanistic_condition_cell_pair_rankings.tsv"
PAIR_OPPORTUNITY_FILENAME = "pair_opportunity.tsv"
REASON_WATERFALL_FILENAME = "downstream_reason_waterfall.tsv"
EDGE_COMPONENT_DELTA_FILENAME = "edge_component_delta.parquet"

SCORE_COLUMNS = (
    "crossfit_result_id",
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "sample_id",
    "subject_id",
    "lesion_type",
    "batch",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    "ligand",
    "receptor",
    "family_id",
    "driver_id",
    "mode",
    "global_sender_lr_score",
    "status",
    "reason_code",
    "score_name",
    "score_direction",
    "score_semantics",
    "formal_inference_allowed",
)
SCORE_LAYER_COLUMNS = (*SCORE_COLUMNS, *SCORE_LAYER_VALUE_COLUMNS)

DIRECTED_EDGE_COLUMNS = (
    "sender",
    "receiver",
    "interaction_id",
    "ligand",
    "receptor",
    "family_id",
    "driver_id",
)


class MSCrossFitResult(Protocol):
    """Narrow current-result interface consumed by the MS runner."""

    path: Path

    @property
    def manifest(self) -> Mapping[str, object]: ...

    def query_contrast_common_sender_lr_scores(
        self,
        *,
        contrast: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame: ...

    def query_contrast_common_lr_scores(
        self,
        *,
        contrast: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame: ...

    def read_state_semantic_availability(self) -> pd.DataFrame: ...


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return cast(dict[str, Any], value)


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _required_sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return text


def _cohort_expectations(
    schema_version: str,
) -> tuple[
    tuple[int, int], str, dict[str, int], dict[str, int], dict[str, tuple[int, ...]]
]:
    """Return the frozen cohort contract selected by an input manifest."""
    if schema_version == FIGURE3_PREPARATION_SCHEMA:
        return (
            FIGURE3_EXPECTED_SHAPE,
            FIGURE3_EXPECTED_OUTPUT_SHA256,
            FIGURE3_EXPECTED_SAMPLES,
            FIGURE3_EXPECTED_SUBJECTS,
            FIGURE3_EXPECTED_SAMPLES_PER_SUBJECT,
        )
    return (
        EXPECTED_SHAPE,
        EXPECTED_OUTPUT_SHA256,
        EXPECTED_SAMPLES,
        EXPECTED_SUBJECTS,
        EXPECTED_SAMPLES_PER_SUBJECT,
    )


def validate_subset_manifest(
    input_h5ad: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify the checksum-bound frozen CA/Ctrl primary subset."""

    input_path = Path(input_h5ad).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    manifest = _read_json_object(manifest_file, label="MS subset manifest")
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("input_manifest.lineage is missing")
    schema_version = manifest.get("schema_version")
    # Older canonical subset manifests kept the preparation schema only in
    # lineage.  Treat that exact shape as the full frozen subset for backward
    # compatibility; downsample manifests must continue to declare their
    # distinct top-level schema explicitly.
    if schema_version is None and lineage.get("schema_version") == PREPARATION_SCHEMA:
        schema_version = PREPARATION_SCHEMA
    if schema_version not in {
        PREPARATION_SCHEMA,
        DOWNSAMPLE_PREPARATION_SCHEMA,
        FIGURE3_PREPARATION_SCHEMA,
    }:
        raise ValueError("input_manifest.schema_version is not a supported MS schema")
    expected_lineage = {
        "schema_version": PREPARATION_SCHEMA,
        "dataset_id": DATASET_ID,
        "selection_field": "lesion_type",
        "selection_values": [TARGET, REFERENCE],
        "primary_contrast": CONTRAST,
        "design": "independent subjects; ~ batch + lesion_type",
        "preserve_observation_order": True,
        "preserve_variable_axis": True,
    }
    for field, expected in expected_lineage.items():
        if lineage.get(field) != expected:
            raise ValueError(
                f"input_manifest.lineage.{field} differs from the frozen MS subset"
            )
    source_sha256 = _required_sha256(
        lineage.get("source_sha256"), field="input_manifest.lineage.source_sha256"
    )
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError("input_manifest.lineage.source_sha256 is not canonical")
    if lineage.get("source_filename") != "UCSC_Lerma_Martin_MS_snRNA.h5ad":
        raise ValueError("input_manifest.lineage.source_filename is not canonical")
    if manifest.get("output") != input_path.name:
        raise ValueError("input_manifest.output does not match input_h5ad")
    expected_shape, cohort_output_sha, expected_samples, expected_subjects, _ = (
        _cohort_expectations(schema_version)
    )
    expected_output_sha = _required_sha256(
        manifest.get("output_sha256"), field="input_manifest.output_sha256"
    )
    if schema_version in {PREPARATION_SCHEMA, FIGURE3_PREPARATION_SCHEMA} and (
        expected_output_sha != cohort_output_sha
    ):
        raise ValueError("input_manifest.output_sha256 is not the canonical MS digest")
    if sha256_file(input_path) != expected_output_sha:
        raise ValueError("input_manifest.output_sha256 does not match input_h5ad")
    if schema_version in {
        PREPARATION_SCHEMA,
        FIGURE3_PREPARATION_SCHEMA,
    } and manifest.get("shape") != list(expected_shape):
        raise ValueError("input_manifest.shape differs from the frozen MS subset")
    if schema_version == DOWNSAMPLE_PREPARATION_SCHEMA:
        declared_shape = manifest.get("shape")
        if not isinstance(declared_shape, list) or len(declared_shape) != 2:
            raise ValueError("downsample MS manifest must declare shape")
        probe = ad.read_h5ad(input_path, backed="r")
        try:
            if list(probe.shape) != declared_shape:
                raise ValueError("downsample MS shape does not match input_h5ad")
        finally:
            if probe.isbacked:
                probe.file.close()
    if manifest.get("samples_by_context") != expected_samples:
        raise ValueError("input_manifest.samples_by_context is invalid")
    if manifest.get("subjects_by_context") != expected_subjects:
        raise ValueError("input_manifest.subjects_by_context is invalid")
    return manifest


def build_crossfit_configuration(
    *, seed: int, min_cells: int
) -> tuple[CrychicConfig, CrossFitSpec]:
    """Freeze the two-fold CA-versus-Ctrl descriptive cross-fit policy."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(min_cells, bool) or not isinstance(min_cells, int) or min_cells < 1:
        raise ValueError("min_cells must be a positive integer")
    contrast = balanced_contrast((TARGET,), (REFERENCE,), name=CONTRAST)
    tuning = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.3, 0.1),
        lambda2_fractions=(0.1, 0.0),
        inner_allowed_n_splits=(2,),
        min_inner_train_subjects_per_context=1,
        min_inner_validation_subjects_per_context=1,
        root_seed=seed,
    )
    spec = CrossFitSpec(
        contrasts=(contrast,),
        outer_fold_partition_seed=seed,
        training_spec=FoldTrainingSpec(
            min_cells=min_cells,
            min_pooled_availability=0.0,
            max_interactions=None,
            sender_parameters=ContrastCommonSenderParameters(
                min_subjects=2,
                contrast_unit="independent_subject",
            ),
        ),
        allowed_n_splits=(OUTER_FOLDS,),
        min_train_subjects_per_context=2,
        min_test_subjects_per_context=1,
        latent_nuisance_spec=FrozenLatentNuisanceSpec(),
        penalty_tuning_spec=tuning,
        gain_calibration_spec=GainCalibrationSpec(
            min_inner_folds=2,
            min_subjects=6,
            min_supported_families=3,
            min_subjects_per_family=2,
            min_positive_observations=6,
            min_distinct_positive_gains=2,
        ),
    )
    config = CrychicConfig(
        context_keys=("lesion_type",),
        counts_layer="counts",
        sample_key="sample_id",
        subject_key="subject_id",
        cell_type_key="cell_type",
        covariates=("batch",),
        categorical_covariates=("batch",),
        species=Species.HUMAN.value,
        gene_namespace=GeneNamespace.HGNC_SYMBOL.value,
        design="~ batch + lesion_type",
        communication_modes=("state",),
        random_seed=seed,
    )
    return config, spec


def _validate_input_contract(
    data: ad.AnnData,
    config: CrychicConfig,
    *,
    expected_lineage: Mapping[str, object],
    expected_samples: Mapping[str, int] = EXPECTED_SAMPLES,
    expected_subjects: Mapping[str, int] = EXPECTED_SUBJECTS,
    expected_samples_per_subject: Mapping[
        str, tuple[int, ...]
    ] = EXPECTED_SAMPLES_PER_SUBJECT,
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, object]]]:
    validated = validate_anndata(
        data,
        InputSchema(
            context_keys=tuple(config.context_keys),
            counts_layer=config.counts_layer,
            sample_key=config.sample_key,
            subject_key=config.subject_key,
            cell_type_key=config.cell_type_key,
            covariates=tuple(config.covariates),
            species=config.species,
            gene_namespace=config.gene_namespace,
        ),
    )
    sample_metadata = validated.report.sample_metadata.loc[
        :, ["sample_id", "subject_id", "lesion_type", "batch"]
    ].copy()
    observed_conditions = set(sample_metadata["lesion_type"].astype(str))
    if observed_conditions != {REFERENCE, TARGET}:
        raise ValueError(
            "MS lesion_type must contain exactly Ctrl and CA; "
            f"observed={sorted(observed_conditions)}"
        )
    subject_sets = {
        condition: set(
            sample_metadata.loc[
                sample_metadata["lesion_type"].astype(str).eq(condition),
                "subject_id",
            ].astype(str)
        )
        for condition in (REFERENCE, TARGET)
    }
    overlap = subject_sets[REFERENCE].intersection(subject_sets[TARGET])
    if overlap:
        raise ValueError(
            "MS subject_id crosses Ctrl and CA, invalidating independent "
            f"subject blocks: {sorted(overlap)}"
        )
    subject_support = {
        condition: len(subjects) for condition, subjects in subject_sets.items()
    }
    if subject_support != dict(expected_subjects):
        raise ValueError(
            f"MS subject support differs from the frozen cohort: {subject_support}"
        )
    sample_support = {
        condition: int(
            sample_metadata.loc[
                sample_metadata["lesion_type"].astype(str).eq(condition), "sample_id"
            ].nunique()
        )
        for condition in (REFERENCE, TARGET)
    }
    if sample_support != dict(expected_samples):
        raise ValueError(
            f"MS sample support differs from the frozen cohort: {sample_support}"
        )
    samples_per_subject = (
        sample_metadata.groupby(
            ["lesion_type", "subject_id"], observed=True, sort=True
        )["sample_id"]
        .nunique()
        .reset_index(name="n_samples")
    )
    observed_distribution = {
        condition: tuple(
            sorted(
                samples_per_subject.loc[
                    samples_per_subject["lesion_type"].astype(str).eq(condition),
                    "n_samples",
                ].astype(int)
            )
        )
        for condition in (REFERENCE, TARGET)
    }
    if observed_distribution != dict(expected_samples_per_subject):
        raise ValueError(
            "MS repeated-sample distribution differs from the frozen cohort: "
            f"{observed_distribution}"
        )
    repeated = (
        samples_per_subject.loc[samples_per_subject["n_samples"].gt(1)]
        .sort_values(["lesion_type", "subject_id"], kind="stable")
        .to_dict(orient="records")
    )
    subset = data.uns.get("crychic_analysis_subset")
    if not isinstance(subset, Mapping):
        raise ValueError("input AnnData lacks crychic_analysis_subset lineage")
    lineage_fields = (
        "schema_version",
        "dataset_id",
        "source_filename",
        "source_sha256",
        "selection_field",
        "preserve_observation_order",
        "preserve_variable_axis",
        "design",
        "primary_contrast",
    )
    for field in lineage_fields:
        if subset.get(field) != expected_lineage.get(field):
            raise ValueError(
                f"input AnnData lineage {field!r} disagrees with subset manifest"
            )
    if list(subset.get("selection_values", ())) != list(
        cast(Sequence[object], expected_lineage.get("selection_values", ()))
    ):
        raise ValueError(
            "input AnnData lineage 'selection_values' disagrees with subset manifest"
        )
    return sample_metadata, subject_support, repeated


def _validate_resource_compatibility(
    data: ad.AnnData,
    bundle: ResourceBundle,
    target_prior: TargetPrior,
) -> dict[str, int]:
    if (
        target_prior.species is not bundle.species
        or target_prior.gene_namespace is not bundle.gene_namespace
    ):
        raise ValueError("ConnectomeDB2020 and target prior are incompatible")
    genes = set(map(str, data.var_names))
    input_compatible = [
        interaction
        for interaction in bundle.interactions
        if {
            *interaction.ligand_subunits,
            *interaction.receptor_subunits,
        }.issubset(genes)
    ]
    if not input_compatible:
        raise ValueError("ConnectomeDB2020 has no input-compatible LR pair")
    drivers = (
        {interaction.ligand_name for interaction in input_compatible}
        if target_prior.driver_kind == "ligand"
        else {interaction.interaction_id for interaction in input_compatible}
    )
    prior_overlap = drivers.intersection(target_prior.driver_ids)
    if not prior_overlap:
        raise ValueError("input-compatible interactions lack target-prior coverage")
    return {
        "input_compatible_interactions": len(input_compatible),
        "target_prior_covered_drivers": len(prior_overlap),
    }


def compact_sender_lr_scores(
    result: MSCrossFitResult,
    data: ad.AnnData,
    bundle: ResourceBundle,
    *,
    source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Export current v9 held-out MS sender-LR rows at sample grain."""

    if source is None:
        source = result.query_contrast_common_sender_lr_scores(
            contrast=CONTRAST,
            mode="state",
            status=None,
        )
    if not isinstance(source, pd.DataFrame) or source.empty:
        raise ValueError("v9 result has no CA_vs_Ctrl state sender-LR rows")
    required = {
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "fold_id",
        "contrast_id",
        "contrast",
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
        "global_sender_lr_score",
        "status",
        "reason_code",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"v9 sender-LR result is missing columns: {sorted(missing)}")
    manifest = result.manifest
    if manifest.get("schema_version") != "9.0.0":
        raise ValueError("MS formal runner requires CrossFitResult schema 9.0.0")
    if manifest.get("status") not in {None, "complete"}:
        raise ValueError("CrossFitResult manifest is not complete")
    result_id = _required_text(
        manifest.get("crossfit_result_id"), field="crossfit_result_id"
    )
    sample_metadata = (
        data.obs.loc[:, ["sample_id", "subject_id", "lesion_type", "batch"]]
        .astype(str)
        .drop_duplicates()
    )
    if sample_metadata["sample_id"].duplicated().any():
        raise ValueError("input sample_id maps to multiple subject/context rows")
    bound = source.merge(
        sample_metadata.rename(columns={"subject_id": "prepared_subject_id"}),
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if bound["prepared_subject_id"].isna().any():
        raise ValueError("v9 result contains samples absent from input")
    if (
        not bound["subject_id"]
        .astype(str)
        .eq(bound["prepared_subject_id"].astype(str))
        .all()
    ):
        raise ValueError("v9 sender-LR subject lineage disagrees with input")
    if set(bound["contrast"].astype(str)) != {CONTRAST} or set(
        bound["mode"].astype(str)
    ) != {"state"}:
        raise ValueError("v9 sender-LR contrast or mode is inconsistent")
    resource = pd.DataFrame.from_records(
        [
            {
                "interaction_id": interaction.interaction_id,
                "ligand": interaction.ligand_name,
                "receptor": interaction.receptor_name,
            }
            for interaction in bundle.interactions
        ]
    )
    bound = bound.merge(
        resource, on="interaction_id", how="left", validate="many_to_one"
    )
    if bound[["ligand", "receptor"]].isna().any().any():
        raise ValueError("v9 sender-LR result contains interactions outside resource")
    statuses = bound["status"].astype(str)
    if not set(statuses).issubset({"observed", "structural_zero", "not_estimable"}):
        raise ValueError("v9 sender-LR result contains unsupported statuses")
    score = pd.to_numeric(bound["global_sender_lr_score"], errors="coerce")
    observed = statuses.eq("observed")
    structural = statuses.eq("structural_zero")
    not_estimable = statuses.eq("not_estimable")
    if (
        score.loc[observed].isna().any()
        or not score.loc[observed].between(0.0, 1.0).all()
        or not score.loc[structural].eq(0.0).all()
        or score.loc[not_estimable].notna().any()
    ):
        raise ValueError("v9 sender-LR status and score semantics are inconsistent")
    key = ["sample_id", "sender", "receiver", "interaction_id"]
    if bound.duplicated(key).any():
        raise ValueError("v9 sender-LR rows are not unique at sample-edge grain")

    output = pd.DataFrame(index=bound.index)
    for column in (
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "fold_id",
        "contrast_id",
        "contrast",
        "sample_id",
        "subject_id",
        "lesion_type",
        "batch",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "family_id",
        "driver_id",
        "mode",
        "status",
        "reason_code",
    ):
        output[column] = bound[column]
    output.insert(0, "crossfit_result_id", result_id)
    output["global_sender_lr_score"] = score
    output["score_name"] = "global_sender_lr_score"
    output["score_direction"] = "higher_is_stronger"
    output["score_semantics"] = (
        "heldout_receiver_balanced_descriptive_sender_lr_strength"
    )
    output["formal_inference_allowed"] = False
    return output.loc[:, list(SCORE_COLUMNS)].sort_values(
        ["sample_id", "sender", "receiver", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )


def multigroup_score_layers(
    result: MSCrossFitResult,
    compact_scores: pd.DataFrame,
    *,
    sender_components: pd.DataFrame | None = None,
    lr_components: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the preregistered mechanism/annotation/strict MS score layers."""

    if sender_components is None:
        sender_components = result.query_contrast_common_sender_lr_scores(
            contrast=CONTRAST,
            mode="state",
            status=None,
        )
    if lr_components is None:
        lr_components = result.query_contrast_common_lr_scores(
            contrast=CONTRAST,
            mode="state",
            status=None,
        )
    downstream_path = result.path / "descriptive_differential.parquet"
    if not downstream_path.is_file():
        raise FileNotFoundError(
            f"persisted downstream differential table is missing: {downstream_path}"
        )
    downstream = pd.read_parquet(downstream_path)
    layers = build_multigroup_score_layers(
        compact_scores,
        sender_components,
        lr_components,
        downstream,
        policy="annotate",
    )
    if tuple(layers.columns) != SCORE_LAYER_COLUMNS:
        raise RuntimeError("MS score-layer columns do not match the released contract")
    return layers


def _heldout_fold_audit(
    scores: pd.DataFrame, expected_sample_metadata: pd.DataFrame
) -> dict[str, object]:
    coverage = heldout_sample_coverage_audit(scores, expected_sample_metadata)
    subject_folds = scores.loc[:, ["subject_id", "fold_id"]].drop_duplicates()
    if subject_folds["subject_id"].duplicated().any():
        raise ValueError("one subject appears in multiple held-out outer folds")
    folds = sorted(subject_folds["fold_id"].astype(str).unique())
    if len(folds) != OUTER_FOLDS:
        raise ValueError(
            f"expected {OUTER_FOLDS} held-out folds, observed {len(folds)}"
        )
    repeated_sample_folds = (
        scores.loc[:, ["subject_id", "sample_id", "fold_id"]]
        .drop_duplicates()
        .groupby("subject_id", observed=True)["fold_id"]
        .nunique()
    )
    if repeated_sample_folds.gt(1).any():
        raise ValueError("technical samples from one subject cross held-out folds")
    return {
        **coverage,
        "n_folds": len(folds),
        "subject_heldout_once": True,
        "technical_samples_subject_blocked": True,
        "heldout_subjects_by_fold": {
            fold: sorted(
                subject_folds.loc[
                    subject_folds["fold_id"].astype(str).eq(fold), "subject_id"
                ].astype(str)
            )
            for fold in folds
        },
    }


def _method_version() -> str:
    try:
        package_version = importlib.metadata.version("CRYCHIC")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source-tree"
    commit = git_metadata(REPO_ROOT).get("commit")
    return (
        package_version
        if not isinstance(commit, str) or not commit
        else f"{package_version}@{commit[:12]}"
    )


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": sha256_file(path),
        "columns": list(table.columns),
    }


def run_ms_ctrl_ca(
    input_h5ad: str | Path,
    output_dir: str | Path,
    *,
    input_manifest: str | Path,
    connectomedb_resource: str | Path,
    connectomedb_manifest: str | Path,
    database_root: str | Path,
    nichenet_release: str = "v2_2021",
    nichenet_manifest: str | Path | None = None,
    seed: int = 20260717,
    threads: int = 8,
    fold_jobs: int = 1,
    min_cells: int = 10,
    overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the checksum-bound Lerma-Martin MS descriptive cross-fit."""

    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if isinstance(fold_jobs, bool) or not isinstance(fold_jobs, int) or fold_jobs < 1:
        raise ValueError("fold_jobs must be a positive integer")
    input_path = Path(input_h5ad).expanduser().resolve()
    preparation_path = Path(input_manifest).expanduser().resolve()
    preparation = validate_subset_manifest(input_path, preparation_path)
    cohort_schema = str(
        preparation.get("schema_version") or preparation["lineage"]["schema_version"]
    )
    _, _, expected_samples, expected_subjects, expected_samples_per_subject = (
        _cohort_expectations(cohort_schema)
    )
    bundle, resource_provenance = load_connectomedb2020_bundle(
        connectomedb_resource, connectomedb_manifest
    )
    database = Path(database_root).expanduser().resolve()
    prior_manifest_path = (
        None
        if nichenet_manifest is None
        else Path(nichenet_manifest).expanduser().resolve()
    )
    target_prior = load_nichenet_target_prior(
        database,
        release=nichenet_release,
        manifest_path=prior_manifest_path,
    )
    config, spec = build_crossfit_configuration(seed=seed, min_cells=min_cells)
    output = prepare_output(output_dir, overwrite=overwrite)
    started = time.perf_counter()
    method_version = _method_version()
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "status": "running",
        "dataset_id": DATASET_ID,
        "contrast": {
            "name": CONTRAST,
            "reference": REFERENCE,
            "target": TARGET,
            "effect_direction": "CA_minus_Ctrl",
        },
        "input": {
            "filename": input_path.name,
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
            "subset_manifest": {
                "filename": preparation_path.name,
                "sha256": sha256_file(preparation_path),
                "source_sha256": preparation["lineage"]["source_sha256"],
                "schema_version": cohort_schema,
            },
        },
        "resource": resource_provenance,
        "target_prior": {
            "resource_id": target_prior.resource_id,
            "version": target_prior.version,
            "manifest_digest": target_prior.manifest_digest,
            "driver_kind": target_prior.driver_kind,
            "drivers": len(target_prior.driver_ids),
            "targets": len(target_prior.target_ids),
            "release": nichenet_release,
            "manifest_filename": (
                None if prior_manifest_path is None else prior_manifest_path.name
            ),
        },
        "parameters": {
            "seed": seed,
            "threads": threads,
            "blas_threads_per_fold": threads,
            "fold_jobs": fold_jobs,
            "effective_fold_jobs": min(fold_jobs, OUTER_FOLDS),
            "outer_folds": OUTER_FOLDS,
            "min_cells": min_cells,
            "max_interactions": None,
            "communication_mode": "state",
            "des_postprocess": {
                "statistical_unit": "subject_id",
                "technical_sample_policy": "mean_within_subject_and_lesion_type",
                "min_subjects_per_condition": 3,
                "cell_pair_mode": "unordered_directions_collapsed",
                "direct_response": "sender_specific_ligand_x_receptor_availability",
                "covariate_adjustment": "categorical_fixed_effects:batch",
                "uncertainty": (
                    "contrast_specific_hc2_heteroskedasticity_robust_standard_error"
                ),
                "primary_aggregation": "one_standard_error_stable_lr_breadth",
                "release_status": "exploratory_v3",
            },
            "config": config.to_dict(),
            "crossfit_spec": spec.to_dict(),
        },
        "method": {"id": "crychic", "version": method_version},
        "environment": python_environment(
            environment_name="crychic_project_uv",
            packages=(
                "CRYCHIC",
                "anndata",
                "numpy",
                "pandas",
                "pyarrow",
                "threadpoolctl",
            ),
            threads=threads,
        ),
        "code": git_metadata(repo_root),
        "score_semantics": {
            "downstream_confirmed_sample_score": (
                "heldout_receiver_balanced_descriptive_sender_lr_strength"
            ),
            "primary_sample_score": "sender_specific_availability_state",
            "primary_score_policy": "direct_differential_head_exploratory_v3",
            "mechanistic_score": (
                "heldout_availability_x_prior_quality_x_frozen_sender_assignment"
            ),
            "downstream_support": "signed_receiver_family_loss_ratio_annotation",
            "directed_effect": (
                "CA_minus_Ctrl_subject_level_batch_adjusted_hc2_direct_sender_lr"
            ),
            "primary_ranking": (
                "one_standard_error_stable_sender_specific_differential_lr_breadth"
            ),
            "legacy_mechanistic_ranking_retained": True,
            "technical_sample_policy": "mean_within_subject_and_lesion_type",
            "cells_are_independent_replicates": False,
            "formal_inference_allowed": False,
            "p_value": "not_emitted",
            "q_value": "not_emitted",
        },
        "outputs": None,
        "failure": None,
    }
    write_json(output / "manifest.json", manifest)
    data: ad.AnnData | None = None
    try:
        data = ad.read_h5ad(input_path)
        sample_metadata, subject_support, repeated_subjects = _validate_input_contract(
            data,
            config,
            expected_lineage=cast(Mapping[str, object], preparation["lineage"]),
            expected_samples=expected_samples,
            expected_subjects=expected_subjects,
            expected_samples_per_subject=expected_samples_per_subject,
        )
        compatibility = _validate_resource_compatibility(data, bundle, target_prior)
        model = Crychic(config, resource_bundle=bundle, target_prior=target_prior)
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

        with threadpool_limits(limits=threads):
            result = cast(
                MSCrossFitResult,
                model.fit_descriptive(
                    data,
                    spec=spec,
                    n_jobs=fold_jobs,
                    output_dir=output / "crossfit_result",
                ),
            )
        sender_components = result.query_contrast_common_sender_lr_scores(
            contrast=CONTRAST,
            mode="state",
            status=None,
        )
        lr_components = result.query_contrast_common_lr_scores(
            contrast=CONTRAST,
            mode="state",
            status=None,
        )
        scores = compact_sender_lr_scores(
            result,
            data,
            bundle,
            source=sender_components,
        )
        score_layers = multigroup_score_layers(
            result,
            scores,
            sender_components=sender_components,
            lr_components=lr_components,
        )
        del sender_components, lr_components
        fold_audit = _heldout_fold_audit(scores, sample_metadata)
        strict_directed_effects = subject_equal_directed_lr_effects(
            scores,
            reference=REFERENCE,
            target=TARGET,
            condition_column="lesion_type",
            edge_columns=DIRECTED_EDGE_COLUMNS,
            min_subjects_per_condition=3,
        )
        mechanistic_scores = score_layers.copy(deep=False)
        mechanistic_scores["status"] = score_layers["selected_score_status"]
        mechanistic_scores["reason_code"] = score_layers["selected_score_reason_code"]
        mechanistic_directed_effects = subject_equal_directed_lr_effects(
            mechanistic_scores,
            reference=REFERENCE,
            target=TARGET,
            condition_column="lesion_type",
            edge_columns=DIRECTED_EDGE_COLUMNS,
            score_column="selected_score",
            min_subjects_per_condition=3,
        )
        del mechanistic_scores
        mechanistic_rankings = unordered_cell_pair_des_rankings(
            mechanistic_directed_effects,
            dataset=DES_DATASET_ID,
            method="crychic",
            method_version=method_version,
            resource=bundle.resource_id,
        )
        semantic_availability = result.read_state_semantic_availability()
        direct_scores = sender_specific_direct_scores(
            score_layers,
            semantic_availability,
            condition_column="lesion_type",
            covariate_columns=("batch",),
            edge_columns=DIRECTED_EDGE_COLUMNS,
            response_transform="identity",
        )
        del semantic_availability
        reason_waterfall = score_reason_waterfall(score_layers, direct_scores)
        direct_effects = adjusted_subject_directed_lr_effects(
            direct_scores,
            reference=REFERENCE,
            target=TARGET,
            condition_column="lesion_type",
            categorical_covariates=("batch",),
            edge_columns=DIRECTED_EDGE_COLUMNS,
            min_subjects_per_condition=3,
        )
        del direct_scores
        rankings, pair_opportunity = stable_breadth_unordered_cell_pair_rankings(
            direct_effects,
            dataset=DES_DATASET_ID,
            method="crychic",
            method_version=method_version,
            resource=bundle.resource_id,
        )
        ranking_diagnostics = condition_ranking_diagnostics(rankings)
        if ranking_diagnostics["any_degenerate_condition"]:
            raise RuntimeError("primary condition-level ranking is degenerate")
        direct_component = direct_effects.loc[
            :,
            [
                *DIRECTED_EDGE_COLUMNS,
                "mean_strength_reference",
                "mean_strength_target",
                "effect_target_minus_reference",
                "effect_standard_error_hc2",
                "one_standard_error_stable",
                "status",
                "reason_code",
            ],
        ].rename(
            columns={
                column: f"direct_{column}"
                for column in (
                    "mean_strength_reference",
                    "mean_strength_target",
                    "effect_target_minus_reference",
                    "effect_standard_error_hc2",
                    "one_standard_error_stable",
                    "status",
                    "reason_code",
                )
            }
        )
        mechanistic_component = mechanistic_directed_effects.loc[
            :,
            [
                *DIRECTED_EDGE_COLUMNS,
                "mean_strength_reference",
                "mean_strength_target",
                "effect_target_minus_reference",
                "status",
                "reason_code",
            ],
        ].rename(
            columns={
                column: f"mechanistic_{column}"
                for column in (
                    "mean_strength_reference",
                    "mean_strength_target",
                    "effect_target_minus_reference",
                    "status",
                    "reason_code",
                )
            }
        )
        edge_component_delta = direct_component.merge(
            mechanistic_component,
            on=list(DIRECTED_EDGE_COLUMNS),
            how="outer",
            validate="one_to_one",
            sort=False,
        )
        score_path = output / SCORE_FILENAME
        score_layer_path = output / SCORE_LAYER_FILENAME
        strict_directed_path = output / DIRECTED_EFFECT_FILENAME
        mechanistic_directed_path = output / MECHANISTIC_DIRECTED_EFFECT_FILENAME
        direct_effect_path = output / SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME
        ranking_path = output / UNORDERED_RANKING_FILENAME
        mechanistic_ranking_path = output / MECHANISTIC_UNORDERED_RANKING_FILENAME
        pair_opportunity_path = output / PAIR_OPPORTUNITY_FILENAME
        reason_waterfall_path = output / REASON_WATERFALL_FILENAME
        edge_component_path = output / EDGE_COMPONENT_DELTA_FILENAME
        scores.to_parquet(score_path, index=False, compression="zstd")
        score_layers.to_parquet(score_layer_path, index=False, compression="zstd")
        strict_directed_effects.to_parquet(
            strict_directed_path, index=False, compression="zstd"
        )
        mechanistic_directed_effects.to_parquet(
            mechanistic_directed_path, index=False, compression="zstd"
        )
        direct_effects.to_parquet(direct_effect_path, index=False, compression="zstd")
        rankings.to_csv(ranking_path, sep="\t", index=False, lineterminator="\n")
        mechanistic_rankings.to_csv(
            mechanistic_ranking_path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        pair_opportunity.to_csv(
            pair_opportunity_path, sep="\t", index=False, lineterminator="\n"
        )
        reason_waterfall.to_csv(
            reason_waterfall_path, sep="\t", index=False, lineterminator="\n"
        )
        edge_component_delta.to_parquet(
            edge_component_path, index=False, compression="zstd"
        )
        crossfit_manifest_path = result.path / "crossfit_manifest.json"
        if not crossfit_manifest_path.is_file():
            raise FileNotFoundError(
                f"persisted cross-fit manifest is missing: {crossfit_manifest_path}"
            )
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - started,
                "input": {
                    **manifest["input"],
                    "shape": [int(data.n_obs), int(data.n_vars)],
                    "samples": len(sample_metadata),
                    "subjects": int(sample_metadata["subject_id"].nunique()),
                    "subject_support": subject_support,
                    "repeated_subjects": repeated_subjects,
                },
                "resource": {**resource_provenance, **compatibility},
                "crossfit_result": {
                    "directory": result.path.name,
                    "schema_version": result.manifest.get("schema_version"),
                    "crossfit_result_id": result.manifest.get("crossfit_result_id"),
                    "manifest_sha256": sha256_file(crossfit_manifest_path),
                    "heldout_fold_audit": fold_audit,
                },
                "score_layer_diagnostics": {
                    "mechanistic_sender_lr_score": summarize_score_layer(
                        score_layers,
                        score_column="mechanistic_sender_lr_score",
                        status_column="mechanistic_status",
                    ),
                    "downstream_confirmed_sender_lr_score": summarize_score_layer(
                        score_layers,
                        score_column="downstream_confirmed_sender_lr_score",
                        status_column="downstream_confirmed_status",
                    ),
                    "selected_score": summarize_score_layer(
                        score_layers,
                        score_column="selected_score",
                        status_column="selected_score_status",
                    ),
                    "primary_condition_rankings": ranking_diagnostics,
                    "sender_specific_direct_effect_signs": {
                        "positive": int(
                            direct_effects.loc[
                                direct_effects["status"].eq("observed"),
                                "effect_target_minus_reference",
                            ]
                            .gt(0.0)
                            .sum()
                        ),
                        "negative": int(
                            direct_effects.loc[
                                direct_effects["status"].eq("observed"),
                                "effect_target_minus_reference",
                            ]
                            .lt(0.0)
                            .sum()
                        ),
                        "zero": int(
                            direct_effects.loc[
                                direct_effects["status"].eq("observed"),
                                "effect_target_minus_reference",
                            ]
                            .eq(0.0)
                            .sum()
                        ),
                    },
                },
                "outputs": {
                    SCORE_FILENAME: _output_record(score_path, scores),
                    SCORE_LAYER_FILENAME: _output_record(
                        score_layer_path, score_layers
                    ),
                    DIRECTED_EFFECT_FILENAME: _output_record(
                        strict_directed_path, strict_directed_effects
                    ),
                    MECHANISTIC_DIRECTED_EFFECT_FILENAME: _output_record(
                        mechanistic_directed_path, mechanistic_directed_effects
                    ),
                    SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME: _output_record(
                        direct_effect_path, direct_effects
                    ),
                    UNORDERED_RANKING_FILENAME: _output_record(ranking_path, rankings),
                    MECHANISTIC_UNORDERED_RANKING_FILENAME: _output_record(
                        mechanistic_ranking_path, mechanistic_rankings
                    ),
                    PAIR_OPPORTUNITY_FILENAME: _output_record(
                        pair_opportunity_path, pair_opportunity
                    ),
                    REASON_WATERFALL_FILENAME: _output_record(
                        reason_waterfall_path, reason_waterfall
                    ),
                    EDGE_COMPONENT_DELTA_FILENAME: _output_record(
                        edge_component_path, edge_component_delta
                    ),
                },
            }
        )
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if data is not None and data.isbacked:
            data.file.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--connectomedb-resource", type=Path, required=True)
    parser.add_argument("--connectomedb-manifest", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--nichenet-release", default="v2_2021")
    parser.add_argument("--nichenet-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="BLAS threads available to each concurrently executing outer fold",
    )
    parser.add_argument("--fold-jobs", type=int, default=1)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = run_ms_ctrl_ca(
        args.input_h5ad,
        args.output_dir,
        input_manifest=args.input_manifest,
        connectomedb_resource=args.connectomedb_resource,
        connectomedb_manifest=args.connectomedb_manifest,
        database_root=args.database_root,
        nichenet_release=args.nichenet_release,
        nichenet_manifest=args.nichenet_manifest,
        seed=args.seed,
        threads=args.threads,
        fold_jobs=args.fold_jobs,
        min_cells=args.min_cells,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(Path(args.output_dir).resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DIRECTED_EDGE_COLUMNS",
    "SCORE_COLUMNS",
    "SCORE_LAYER_COLUMNS",
    "SCORE_LAYER_FILENAME",
    "build_crossfit_configuration",
    "compact_sender_lr_scores",
    "multigroup_score_layers",
    "run_ms_ctrl_ca",
    "validate_subset_manifest",
]
