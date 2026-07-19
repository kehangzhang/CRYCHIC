"""Run the formal descriptive CRYCHIC cross-fit for Kuppe CTRL versus IZ."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    python_environment,
    sha256_file,
    write_json,
)
from benchmarks.adapters.crychic.des_postprocess import (
    heldout_sample_coverage_audit,
    subject_equal_directed_lr_effects,
    unordered_cell_pair_des_rankings,
)
from benchmarks.literature.connectomedb2020 import (
    PAPER_METHODS,
    validate_resource_table,
)
from benchmarks.literature.connectomedb2020 import (
    REQUIRED_COLUMNS as CONNECTOMEDB_COLUMNS,
)
from crychic import Crychic, CrychicConfig
from crychic.attribution import GainCalibrationSpec, PenaltyTuningSpec
from crychic.data import InputSchema, validate_anndata
from crychic.design import balanced_contrast
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
    build_interaction_id,
    load_nichenet_target_prior,
)
from crychic.scoring import FrozenLatentNuisanceSpec
from crychic.sender import ContrastCommonSenderParameters
from crychic.workflow import CrossFitSpec, FoldTrainingSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ID = "Kuppe_MI_CTRL_vs_IZ"
PREPARATION_SCHEMA = "crychic-kuppe-ctrl-iz-preparation-v1"
DOWNSAMPLE_PREPARATION_SCHEMA = "crychic-kuppe-ctrl-iz-downsample-v1"
RESOURCE_SCHEMA = "crychic-connectomedb2020-resource-v1"
RUN_SCHEMA = "crychic-kuppe-ctrl-iz-crossfit-run-v1"
CONTRAST = "IZ_vs_CTRL"
REFERENCE = "CTRL"
TARGET = "IZ"
OUTER_FOLDS = 3
SCORE_FILENAME = "sender_lr_scores.parquet"
DIFFERENCE_FILENAME = "sender_lr_differences.parquet"
RANKING_FILENAME = "cell_pair_direction_ranking.parquet"
DIRECTED_EFFECT_FILENAME = "directed_lr_effects.parquet"
UNORDERED_RANKING_FILENAME = "condition_cell_pair_rankings.tsv"
CONNECTOMEDB_ADAPTER_COLUMNS = (
    "harmonized_interaction_id",
    *CONNECTOMEDB_COLUMNS,
    *(
        column
        for method in PAPER_METHODS
        for column in (f"{method}_source_interaction_id", f"{method}_covered")
    ),
)

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
    "condition",
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

DIFFERENCE_COLUMNS = (
    "sender",
    "receiver",
    "interaction_id",
    "ligand",
    "receptor",
    "mean_strength_CTRL",
    "mean_strength_IZ",
    "n_subjects_CTRL",
    "n_subjects_IZ",
    "strength_difference_IZ_minus_CTRL",
    "status",
    "reason_code",
    "effect_semantics",
    "formal_inference_allowed",
)

RANKING_COLUMNS = (
    "sender",
    "receiver",
    "direction",
    "reference_condition",
    "target_condition",
    "n_comparable_lr",
    "differential_lr_count",
    "summed_positive_strength_difference",
    "mean_positive_strength_difference",
    "minimum_reference_subjects",
    "minimum_target_subjects",
    "rank_by_differential_lr_count",
    "rank_by_summed_positive_strength_difference",
    "status",
    "reason_code",
    "effect_semantics",
    "formal_inference_allowed",
)

DIRECTED_EDGE_COLUMNS = (
    "sender",
    "receiver",
    "interaction_id",
    "ligand",
    "receptor",
    "family_id",
    "driver_id",
)


class KuppeCrossFitResult(Protocol):
    """Narrow current-result interface consumed by the runner."""

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


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(value: Mapping[str, object], *, digest_field: str) -> str:
    payload = {key: item for key, item in value.items() if key != digest_field}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


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


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def validate_preparation_manifest(
    input_h5ad: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify that the H5AD is the checksum-bound counts-ready Kuppe subset."""

    input_path = Path(input_h5ad).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    manifest = _read_json_object(manifest_file, label="Kuppe preparation manifest")
    schema_version = manifest.get("schema_version")
    if schema_version not in {PREPARATION_SCHEMA, DOWNSAMPLE_PREPARATION_SCHEMA}:
        raise ValueError(
            "input_manifest.schema_version is not the Kuppe CTRL/IZ schema"
        )
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("input_manifest.dataset_id is not Kuppe_MI_CTRL_vs_IZ")
    if manifest.get("mode") != "counts_ready_h5ad":
        raise ValueError(
            "input_manifest.mode must be counts_ready_h5ad; rerun "
            "prepare_kuppe_ctrl_iz.py without --audit-only"
        )
    expected_payload_digest = _required_text(
        manifest.get("manifest_payload_sha256"),
        field="input_manifest.manifest_payload_sha256",
    )
    observed_payload_digest = _payload_sha256(
        manifest, digest_field="manifest_payload_sha256"
    )
    if observed_payload_digest != expected_payload_digest:
        raise ValueError("input_manifest.manifest_payload_sha256 is invalid")
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("input_manifest.output is missing")
    if output.get("filename") != input_path.name:
        raise ValueError("input_manifest.output.filename does not match input_h5ad")
    if output.get("size_bytes") != input_path.stat().st_size:
        raise ValueError("input_manifest.output.size_bytes does not match input_h5ad")
    observed_sha256 = sha256_file(input_path)
    if output.get("sha256") != observed_sha256:
        raise ValueError("input_manifest.output.sha256 does not match input_h5ad")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or selection.get("values") != [
        REFERENCE,
        TARGET,
    ]:
        raise ValueError("input_manifest.selection must freeze CTRL and IZ in order")
    matrices = manifest.get("matrices")
    if not isinstance(matrices, Mapping) or matrices.get("counts_layer") != "counts":
        raise ValueError("input_manifest.matrices must declare layers['counts']")
    if schema_version == DOWNSAMPLE_PREPARATION_SCHEMA:
        downsample = manifest.get("downsample")
        if not isinstance(downsample, Mapping) or float(
            downsample.get("fraction", 0.0)
        ) <= 0:
            raise ValueError("downsample manifest must declare a positive fraction")
        if output.get("shape") != list(manifest.get("cohort", {}).get("shape", ())):
            raise ValueError("downsample output shape disagrees with cohort metadata")
    return manifest


def _license_text(value: object) -> str:
    if isinstance(value, Mapping):
        entries = [
            f"{key}={item}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(item).strip()
        ]
        if entries:
            return "; ".join(entries)
    text = str(value).strip()
    if not text:
        raise ValueError("connectomedb_manifest.license is missing")
    return text


def load_connectomedb2020_bundle(
    table_path: str | Path,
    manifest_path: str | Path,
) -> tuple[ResourceBundle, dict[str, object]]:
    """Build a simple-LR bundle from a checksum-pinned ConnectomeDB2020 TSV."""

    table_file = Path(table_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not table_file.is_file():
        raise FileNotFoundError(table_file)
    manifest = _read_json_object(
        manifest_file, label="ConnectomeDB2020 resource manifest"
    )
    if manifest.get("schema_version") != RESOURCE_SCHEMA:
        raise ValueError("connectomedb_manifest.schema_version is unsupported")
    expected_manifest_digest = _required_text(
        manifest.get("manifest_payload_sha256"),
        field="connectomedb_manifest.manifest_payload_sha256",
    )
    if (
        _payload_sha256(manifest, digest_field="manifest_payload_sha256")
        != expected_manifest_digest
    ):
        raise ValueError("connectomedb_manifest.manifest_payload_sha256 is invalid")
    if manifest.get("species") != "human" or manifest.get("gene_namespace") not in {
        "HGNC symbol",
        "hgnc_symbol",
    }:
        raise ValueError("ConnectomeDB2020 must declare human species and HGNC symbols")
    if manifest.get("directionality") != "ligand_to_receptor":
        raise ValueError("ConnectomeDB2020 directionality must be ligand_to_receptor")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("connectomedb_manifest.payload is missing")
    if payload.get("filename") != table_file.name:
        raise ValueError("connectomedb_manifest.payload.filename does not match TSV")
    if payload.get("bytes") != table_file.stat().st_size:
        raise ValueError("connectomedb_manifest.payload.bytes does not match TSV")
    table_sha256 = sha256_file(table_file)
    if payload.get("sha256") != table_sha256:
        raise ValueError("connectomedb_manifest.payload.sha256 does not match TSV")
    expected_rows = _positive_integer(
        manifest.get("rows"), field="connectomedb_manifest.rows"
    )
    raw = pd.read_csv(table_file, sep="\t", dtype=str, keep_default_na=False)
    if tuple(raw.columns) != CONNECTOMEDB_ADAPTER_COLUMNS:
        raise ValueError(
            "ConnectomeDB2020 TSV columns differ from its frozen benchmark schema"
        )
    table = validate_resource_table(
        raw.loc[:, list(CONNECTOMEDB_COLUMNS)], expected_rows=expected_rows
    ).merge(
        raw.loc[
            :,
            [
                "ligand",
                "receptor",
                "harmonized_interaction_id",
                "crychic_source_interaction_id",
                "crychic_covered",
            ],
        ],
        on=["ligand", "receptor"],
        how="left",
        validate="one_to_one",
    )
    source_ids = table["crychic_source_interaction_id"].astype(str)
    harmonized_ids = table["harmonized_interaction_id"].astype(str)
    if (
        source_ids.eq("").any()
        or source_ids.duplicated().any()
        or not source_ids.equals(harmonized_ids)
        or not table["crychic_covered"].astype(str).eq("True").all()
    ):
        raise ValueError(
            "ConnectomeDB2020 CRYCHIC adapter IDs or coverage flags are invalid"
        )

    resource_id = _required_text(
        manifest.get("resource_id"), field="connectomedb_manifest.resource_id"
    )
    version = _required_text(
        manifest.get("version"), field="connectomedb_manifest.version"
    )
    citation = _required_text(
        manifest.get("citation"), field="connectomedb_manifest.citation"
    )
    interactions: list[Interaction] = []
    mapped_genes: set[str] = set()
    for row in table.itertuples(index=False):
        ligand = str(row.ligand)
        receptor = str(row.receptor)
        source_id = str(row.crychic_source_interaction_id)
        interaction_id = build_interaction_id(
            resource_id=resource_id,
            version=version,
            species=Species.HUMAN,
            source_interaction_id=source_id,
            ligand_subunits=(ligand,),
            receptor_subunits=(receptor,),
            direction="Ligand-Receptor",
        )
        evidence = tuple(
            value
            for value in (
                f"source={row.source}" if str(row.source) else "",
                f"pmid_support={row.pmid_support}" if str(row.pmid_support) else "",
            )
            if value
        )
        interactions.append(
            Interaction(
                interaction_id=interaction_id,
                source_interaction_id=source_id,
                ligand_name=ligand,
                receptor_name=receptor,
                ligand_subunits=(ligand,),
                receptor_subunits=(receptor,),
                ligand_is_complex=False,
                receptor_is_complex=False,
                direction="Ligand-Receptor",
                source=resource_id,
                version=version,
                species=Species.HUMAN,
                gene_namespace=GeneNamespace.HGNC_SYMBOL,
                evidence=evidence,
            )
        )
        mapped_genes.update((ligand, receptor))
    bundle = ResourceBundle(
        resource_id=resource_id,
        version=version,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=tuple(interactions),
        mapping_report=MappingReport(
            source_rows=len(table),
            loaded_rows=len(interactions),
            mapped_entities=len(mapped_genes),
            notes=(
                "directed monomeric ligand-receptor adaptation",
                "model columns limited to ligand and receptor",
            ),
        ),
        manifest_digest=expected_manifest_digest,
        source_files=(table_file.name, manifest_file.name),
        license=_license_text(manifest.get("license")),
        citation=citation,
    )
    provenance: dict[str, object] = {
        "schema_version": RESOURCE_SCHEMA,
        "resource_id": bundle.resource_id,
        "version": bundle.version,
        "interactions": len(bundle.interactions),
        "table": {
            "filename": table_file.name,
            "bytes": table_file.stat().st_size,
            "sha256": table_sha256,
        },
        "manifest": {
            "filename": manifest_file.name,
            "bytes": manifest_file.stat().st_size,
            "sha256": sha256_file(manifest_file),
            "payload_sha256": expected_manifest_digest,
        },
        "simple_lr_adaptation": True,
        "source_interaction_id_column": "crychic_source_interaction_id",
        "model_columns": ["ligand", "receptor"],
        "observation_label_dependency": False,
    }
    return bundle, provenance


def build_crossfit_configuration(
    *, seed: int, min_cells: int
) -> tuple[CrychicConfig, CrossFitSpec]:
    """Freeze the three-fold CTRL-versus-IZ descriptive cross-fit policy."""

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
        context_keys=("condition",),
        counts_layer="counts",
        sample_key="sample_id",
        subject_key="subject_id",
        cell_type_key="cell_type",
        species=Species.HUMAN.value,
        gene_namespace=GeneNamespace.HGNC_SYMBOL.value,
        design="~ condition",
        communication_modes=("state",),
        random_seed=seed,
    )
    return config, spec


def _validate_input_contract(
    data: ad.AnnData,
    config: CrychicConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    validated = validate_anndata(
        data,
        InputSchema(
            context_keys=tuple(config.context_keys),
            counts_layer=config.counts_layer,
            sample_key=config.sample_key,
            subject_key=config.subject_key,
            cell_type_key=config.cell_type_key,
            species=config.species,
            gene_namespace=config.gene_namespace,
        ),
    )
    sample_metadata = validated.report.sample_metadata.loc[
        :, ["sample_id", "subject_id", "condition"]
    ].copy()
    observed_conditions = set(sample_metadata["condition"].astype(str))
    if observed_conditions != {REFERENCE, TARGET}:
        raise ValueError(
            "prepared input condition must contain exactly CTRL and IZ; "
            f"observed={sorted(observed_conditions)}"
        )
    subject_sets = {
        condition: set(
            sample_metadata.loc[
                sample_metadata["condition"].astype(str).eq(condition), "subject_id"
            ].astype(str)
        )
        for condition in (REFERENCE, TARGET)
    }
    overlap = subject_sets[REFERENCE].intersection(subject_sets[TARGET])
    if overlap:
        raise ValueError(
            "prepared input subject_id crosses CTRL and IZ; subject-blocked "
            f"independent design is invalid: {sorted(overlap)}"
        )
    support = {condition: len(subjects) for condition, subjects in subject_sets.items()}
    if min(support.values()) < OUTER_FOLDS:
        raise ValueError(
            "three-fold subject-blocked cross-fit requires at least three "
            f"subjects per condition; observed={support}"
        )
    return sample_metadata, support


def _validate_resource_compatibility(
    data: ad.AnnData,
    bundle: ResourceBundle,
    target_prior: TargetPrior,
) -> dict[str, int]:
    if (
        target_prior.species is not bundle.species
        or target_prior.gene_namespace is not bundle.gene_namespace
    ):
        raise ValueError(
            "ConnectomeDB2020 and target_prior species/gene namespace differ"
        )
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
        raise ValueError(
            "ConnectomeDB2020 has no simple ligand-receptor pair represented in "
            "input var_names"
        )
    if target_prior.driver_kind == "ligand":
        drivers = {interaction.ligand_name for interaction in input_compatible}
    else:
        drivers = {interaction.interaction_id for interaction in input_compatible}
    prior_overlap = drivers.intersection(target_prior.driver_ids)
    if not prior_overlap:
        raise ValueError(
            "input-compatible ConnectomeDB2020 interactions have no target-prior "
            "driver coverage"
        )
    return {
        "input_compatible_interactions": len(input_compatible),
        "target_prior_covered_drivers": len(prior_overlap),
    }


def compact_sender_lr_scores(
    result: KuppeCrossFitResult,
    data: ad.AnnData,
    bundle: ResourceBundle,
) -> pd.DataFrame:
    """Export current v9 held-out sender-LR rows at their native sample grain."""

    source = result.query_contrast_common_sender_lr_scores(
        contrast=CONTRAST,
        mode="state",
        status=None,
    )
    if not isinstance(source, pd.DataFrame) or source.empty:
        raise ValueError(
            "v9 cross-fit result has no contrast-common state sender-LR rows for "
            "IZ_vs_CTRL"
        )
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
        raise ValueError("Kuppe formal runner requires CrossFitResult schema 9.0.0")
    result_id = _required_text(
        manifest.get("crossfit_result_id"), field="crossfit_result_id"
    )
    sample_metadata = (
        data.obs.loc[:, ["sample_id", "subject_id", "condition"]]
        .astype(str)
        .drop_duplicates()
    )
    if sample_metadata["sample_id"].duplicated().any():
        raise ValueError("input sample_id maps to multiple subject/condition rows")
    bound = source.merge(
        sample_metadata.rename(columns={"subject_id": "prepared_subject_id"}),
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if bound["prepared_subject_id"].isna().any():
        raise ValueError("v9 sender-LR result contains samples absent from input")
    if (
        not bound["subject_id"]
        .astype(str)
        .eq(bound["prepared_subject_id"].astype(str))
        .all()
    ):
        raise ValueError("v9 sender-LR subject lineage disagrees with input")
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
        resource,
        on="interaction_id",
        how="left",
        validate="many_to_one",
    )
    if bound[["ligand", "receptor"]].isna().any().any():
        raise ValueError("v9 sender-LR result contains interactions outside resource")
    statuses = set(bound["status"].astype(str))
    allowed = {"observed", "structural_zero", "not_estimable"}
    if not statuses.issubset(allowed):
        raise ValueError(f"v9 sender-LR result has unsupported statuses: {statuses}")
    score = pd.to_numeric(bound["global_sender_lr_score"], errors="coerce")
    observed = bound["status"].astype(str).eq("observed")
    structural = bound["status"].astype(str).eq("structural_zero")
    not_estimable = bound["status"].astype(str).eq("not_estimable")
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
    result_table = pd.DataFrame(index=bound.index)
    for column in (
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "fold_id",
        "contrast_id",
        "contrast",
        "sample_id",
        "subject_id",
        "condition",
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
        result_table[column] = bound[column]
    result_table.insert(0, "crossfit_result_id", result_id)
    result_table["global_sender_lr_score"] = score
    result_table["score_name"] = "global_sender_lr_score"
    result_table["score_direction"] = "higher_is_stronger"
    result_table["score_semantics"] = (
        "heldout_receiver_balanced_descriptive_sender_lr_strength"
    )
    result_table["formal_inference_allowed"] = False
    result_table = result_table.loc[:, list(SCORE_COLUMNS)].sort_values(
        ["sample_id", "sender", "receiver", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )
    return result_table


def subject_equal_sender_lr_differences(
    scores: pd.DataFrame,
    *,
    min_subjects_per_condition: int = 3,
) -> pd.DataFrame:
    """Compute descriptive IZ-minus-CTRL strengths after subject-equal averaging."""

    if min_subjects_per_condition < 1:
        raise ValueError("min_subjects_per_condition must be positive")
    missing = set(SCORE_COLUMNS).difference(scores.columns)
    if missing:
        raise ValueError(f"compact sender-LR scores are missing: {sorted(missing)}")
    edge = ["sender", "receiver", "interaction_id", "ligand", "receptor"]
    base = scores.loc[:, edge].drop_duplicates(ignore_index=True)
    if base.duplicated(["sender", "receiver", "interaction_id"]).any():
        raise ValueError("resource mapping is not unique for sender-receiver-LR rows")
    usable = scores.loc[
        scores["status"].astype(str).isin(("observed", "structural_zero"))
    ].copy()
    usable["value"] = pd.to_numeric(usable["global_sender_lr_score"], errors="coerce")
    if usable["value"].isna().any():
        raise ValueError("eligible sender-LR rows must have finite strengths")
    subject = (
        usable.groupby(
            ["condition", "subject_id", *edge],
            observed=True,
            sort=False,
        )["value"]
        .mean()
        .reset_index()
    )
    summary = (
        subject.groupby(["condition", *edge], observed=True, sort=False)
        .agg(
            mean_strength=("value", "mean"),
            n_subjects=("value", "size"),
        )
        .reset_index()
    )
    reference = summary.loc[summary["condition"].astype(str).eq(REFERENCE)].drop(
        columns="condition"
    )
    target = summary.loc[summary["condition"].astype(str).eq(TARGET)].drop(
        columns="condition"
    )
    reference = reference.rename(
        columns={
            "mean_strength": "mean_strength_CTRL",
            "n_subjects": "n_subjects_CTRL",
        }
    )
    target = target.rename(
        columns={
            "mean_strength": "mean_strength_IZ",
            "n_subjects": "n_subjects_IZ",
        }
    )
    result = base.merge(reference, on=edge, how="left", validate="one_to_one").merge(
        target, on=edge, how="left", validate="one_to_one"
    )
    for column in ("n_subjects_CTRL", "n_subjects_IZ"):
        result[column] = result[column].fillna(0).astype(np.int64)
    estimable = (
        result["mean_strength_CTRL"].notna()
        & result["mean_strength_IZ"].notna()
        & result["n_subjects_CTRL"].ge(min_subjects_per_condition)
        & result["n_subjects_IZ"].ge(min_subjects_per_condition)
    )
    result["strength_difference_IZ_minus_CTRL"] = np.nan
    result.loc[estimable, "strength_difference_IZ_minus_CTRL"] = (
        result.loc[estimable, "mean_strength_IZ"]
        - result.loc[estimable, "mean_strength_CTRL"]
    )
    result["status"] = np.where(estimable, "observed", "not_estimable")
    result["reason_code"] = np.where(
        estimable, None, "insufficient_subject_score_coverage"
    )
    result["effect_semantics"] = (
        "IZ_minus_CTRL_subject_equal_mean_global_sender_lr_score_descriptive"
    )
    result["formal_inference_allowed"] = False
    return result.loc[:, list(DIFFERENCE_COLUMNS)].sort_values(
        ["sender", "receiver", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )


def cell_pair_direction_ranking(differences: pd.DataFrame) -> pd.DataFrame:
    """Rank cell pairs separately for positive IZ and positive CTRL differences."""

    missing = set(DIFFERENCE_COLUMNS).difference(differences.columns)
    if missing:
        raise ValueError(f"sender-LR differences are missing: {sorted(missing)}")
    pair_base = differences.loc[:, ["sender", "receiver"]].drop_duplicates()
    observed = differences.loc[differences["status"].eq("observed")].copy()
    observed["signed"] = pd.to_numeric(
        observed["strength_difference_IZ_minus_CTRL"], errors="coerce"
    )
    if observed["signed"].isna().any():
        raise ValueError("observed sender-LR differences must be finite")
    frames: list[pd.DataFrame] = []
    directions = (
        ("IZ_over_CTRL", REFERENCE, TARGET, 1.0),
        ("CTRL_over_IZ", TARGET, REFERENCE, -1.0),
    )
    for direction, reference_condition, target_condition, sign in directions:
        working = observed.copy()
        working["positive"] = (sign * working["signed"]).clip(lower=0.0)
        working["is_positive"] = working["positive"].gt(0.0)
        summary = (
            working.groupby(["sender", "receiver"], observed=True, sort=False)
            .agg(
                n_comparable_lr=("interaction_id", "size"),
                differential_lr_count=("is_positive", "sum"),
                summed_positive_strength_difference=("positive", "sum"),
                minimum_reference_subjects=(
                    "n_subjects_CTRL"
                    if reference_condition == REFERENCE
                    else "n_subjects_IZ",
                    "min",
                ),
                minimum_target_subjects=(
                    "n_subjects_IZ"
                    if target_condition == TARGET
                    else "n_subjects_CTRL",
                    "min",
                ),
            )
            .reset_index()
        )
        frame = pair_base.merge(
            summary, on=["sender", "receiver"], how="left", validate="one_to_one"
        )
        for column in (
            "n_comparable_lr",
            "differential_lr_count",
            "minimum_reference_subjects",
            "minimum_target_subjects",
        ):
            frame[column] = frame[column].fillna(0).astype(np.int64)
        frame["summed_positive_strength_difference"] = frame[
            "summed_positive_strength_difference"
        ].fillna(0.0)
        frame["mean_positive_strength_difference"] = np.where(
            frame["differential_lr_count"].gt(0),
            frame["summed_positive_strength_difference"]
            / frame["differential_lr_count"].clip(lower=1),
            0.0,
        )
        frame["direction"] = direction
        frame["reference_condition"] = reference_condition
        frame["target_condition"] = target_condition
        estimable = frame["n_comparable_lr"].gt(0)
        frame["rank_by_differential_lr_count"] = np.nan
        frame["rank_by_summed_positive_strength_difference"] = np.nan
        frame.loc[estimable, "rank_by_differential_lr_count"] = frame.loc[
            estimable, "differential_lr_count"
        ].rank(method="min", ascending=False)
        frame.loc[estimable, "rank_by_summed_positive_strength_difference"] = frame.loc[
            estimable, "summed_positive_strength_difference"
        ].rank(method="min", ascending=False)
        frame["status"] = np.where(estimable, "observed", "not_estimable")
        frame["reason_code"] = np.where(
            estimable, None, "no_subject_supported_lr_difference"
        )
        frame["effect_semantics"] = (
            "positive_target_minus_reference_subject_equal_mean_"
            "global_sender_lr_score_descriptive"
        )
        frame["formal_inference_allowed"] = False
        frames.append(frame.loc[:, list(RANKING_COLUMNS)])
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        [
            "direction",
            "rank_by_differential_lr_count",
            "rank_by_summed_positive_strength_difference",
            "sender",
            "receiver",
        ],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )


def _method_version() -> str:
    try:
        return importlib.metadata.version("CRYCHIC")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": sha256_file(path),
        "columns": list(table.columns),
    }


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
            f"expected {OUTER_FOLDS} held-out outer folds; observed {len(folds)}"
        )
    return {
        **coverage,
        "n_folds": len(folds),
        "subject_heldout_once": True,
        "heldout_subjects_by_fold": {
            fold: sorted(
                subject_folds.loc[
                    subject_folds["fold_id"].astype(str).eq(fold), "subject_id"
                ].astype(str)
            )
            for fold in folds
        },
    }


def run_kuppe_ctrl_iz(
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
    """Execute the checksum-bound Kuppe CTRL/IZ descriptive cross-fit."""

    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if isinstance(fold_jobs, bool) or not isinstance(fold_jobs, int) or fold_jobs < 1:
        raise ValueError("fold_jobs must be a positive integer")
    input_path = Path(input_h5ad).expanduser().resolve()
    preparation_path = Path(input_manifest).expanduser().resolve()
    preparation = validate_preparation_manifest(input_path, preparation_path)
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
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "status": "running",
        "dataset_id": DATASET_ID,
        "contrast": {
            "name": CONTRAST,
            "reference": REFERENCE,
            "target": TARGET,
            "effect_direction": "IZ_minus_CTRL",
        },
        "input": {
            "filename": input_path.name,
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
            "preparation_manifest": {
                "filename": preparation_path.name,
                "sha256": sha256_file(preparation_path),
                "payload_sha256": preparation["manifest_payload_sha256"],
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
            "config": config.to_dict(),
            "crossfit_spec": spec.to_dict(),
        },
        "method": {"id": "crychic", "version": _method_version()},
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
            "sample_score": (
                "heldout_receiver_balanced_descriptive_sender_lr_strength"
            ),
            "sample_score_direction": "higher_is_stronger",
            "difference": "IZ_minus_CTRL_subject_equal_mean",
            "cell_pair_directions": ["IZ_over_CTRL", "CTRL_over_IZ"],
            "primary_spatial_des_ranking": (
                "sum_positive_subject_equal_directed_lr_effects_after_unordered_"
                "cell_pair_collapse"
            ),
            "cells_are_independent_replicates": False,
            "sample_replicates_within_subject_are_averaged": True,
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
        sample_metadata, subject_support = _validate_input_contract(data, config)
        compatibility = _validate_resource_compatibility(data, bundle, target_prior)
        model = Crychic(config, resource_bundle=bundle, target_prior=target_prior)
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

        with threadpool_limits(limits=threads):
            result = cast(
                KuppeCrossFitResult,
                model.fit_descriptive(
                    data,
                    spec=spec,
                    n_jobs=fold_jobs,
                    output_dir=output / "crossfit_result",
                ),
            )
        scores = compact_sender_lr_scores(result, data, bundle)
        fold_audit = _heldout_fold_audit(scores, sample_metadata)
        differences = subject_equal_sender_lr_differences(scores)
        legacy_ranking = cell_pair_direction_ranking(differences)
        directed_effects = subject_equal_directed_lr_effects(
            scores,
            reference=REFERENCE,
            target=TARGET,
            condition_column="condition",
            edge_columns=DIRECTED_EDGE_COLUMNS,
            min_subjects_per_condition=3,
        )
        des_rankings = unordered_cell_pair_des_rankings(
            directed_effects,
            dataset=DATASET_ID,
            method="crychic",
            method_version=_method_version(),
            resource=bundle.resource_id,
        )
        score_path = output / SCORE_FILENAME
        difference_path = output / DIFFERENCE_FILENAME
        ranking_path = output / RANKING_FILENAME
        directed_effect_path = output / DIRECTED_EFFECT_FILENAME
        unordered_ranking_path = output / UNORDERED_RANKING_FILENAME
        scores.to_parquet(score_path, index=False, compression="zstd")
        differences.to_parquet(difference_path, index=False, compression="zstd")
        legacy_ranking.to_parquet(ranking_path, index=False, compression="zstd")
        directed_effects.to_parquet(
            directed_effect_path, index=False, compression="zstd"
        )
        des_rankings.to_csv(
            unordered_ranking_path,
            sep="\t",
            index=False,
            lineterminator="\n",
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
                },
                "resource": {**resource_provenance, **compatibility},
                "crossfit_result": {
                    "directory": result.path.name,
                    "schema_version": result.manifest.get("schema_version"),
                    "crossfit_result_id": result.manifest.get("crossfit_result_id"),
                    "manifest_sha256": sha256_file(crossfit_manifest_path),
                    "heldout_fold_audit": fold_audit,
                },
                "outputs": {
                    SCORE_FILENAME: _output_record(score_path, scores),
                    DIFFERENCE_FILENAME: _output_record(difference_path, differences),
                    RANKING_FILENAME: _output_record(ranking_path, legacy_ranking),
                    DIRECTED_EFFECT_FILENAME: _output_record(
                        directed_effect_path, directed_effects
                    ),
                    UNORDERED_RANKING_FILENAME: _output_record(
                        unordered_ranking_path, des_rankings
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
    manifest = run_kuppe_ctrl_iz(
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
    "DIFFERENCE_COLUMNS",
    "RANKING_COLUMNS",
    "SCORE_COLUMNS",
    "build_crossfit_configuration",
    "cell_pair_direction_ranking",
    "compact_sender_lr_scores",
    "load_connectomedb2020_bundle",
    "run_kuppe_ctrl_iz",
    "subject_equal_sender_lr_differences",
    "validate_preparation_manifest",
]
