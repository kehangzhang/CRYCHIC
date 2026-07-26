"""Run the frozen v7-primary estimator on Kuppe or locked MS real data."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmarks.adapters.common import (
    git_metadata,
    json_safe,
    prepare_output,
    python_environment,
    sha256_file,
    write_json,
)
from benchmarks.adapters.crychic import run_kuppe_ctrl_iz as kuppe_adapter
from benchmarks.adapters.crychic import run_ms_ctrl_ca as ms_adapter
from benchmarks.literature.evaluate_spatial_des_benchmark import evaluate_rankings
from benchmarks.literature.event_level_des import (
    EVENT_BUDGETS,
    pair_rankings_from_events,
    select_top_k_events_by_scope,
)
from benchmarks.literature.run_event_level_des_real import CONTRACTS as DES_CONTRACTS
from benchmarks.simulation.v7_integrated import (
    V7InferenceFitCache,
    build_v7_primary_score_views,
    g0_g2_equivalence_diagnostic,
    run_v7_inference_matrix,
)
from crychic.inference import DifferentialContrastSpec, DifferentialDesignSpec
from crychic.resources import ResourceBundle, TargetPrior
from crychic.scoring import AbsoluteActivityV2Spec, SignedProgramV2Spec
from crychic.sender import (
    EBShrunkenCouplingV2Spec,
    SenderAttributionV2Spec,
)
from crychic.workflow import CrossFitSpec
from crychic.workflow.crossfit import (
    _run_v7_primary_crossfit,
    _V7PrimaryCrossFitArtifacts,
)
from crychic.workflow.training import _sanitized_raw_input_snapshot

SCHEMA_VERSION = "crychic-suggest-next2-v7-real-e1-run-v1"
CONFIG_SCHEMA_VERSION = "crychic-suggest-next2-v7-real-e1-v1"
FROZEN_ESTIMATOR_COMMIT = "5dc22aaa87896eafbeb687aea8d8e96e3654af01"
FROZEN_SCORE_PROJECTION_COMMIT = "35db184d62653ff8c960e2b135fa8e3060e1d394"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "benchmarks/configs/suggest_next2_v7_real_e1_v1.json"
PRIMARY_SCORE_VIEW = "primary_sender_detection"
SENDER_RESOLVED_GENERATORS = ("G0", "G2", "G3", "G5")


@dataclass(frozen=True, slots=True)
class RealDatasetContract:
    slug: str
    dataset_id: str
    role: str
    condition_column: str
    reference: str
    target: str
    contrast_name: str
    batch_columns: tuple[str, ...]
    seed: int
    outer_fold_partition_seed: int


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
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
    record = {
        "elapsed_seconds": fields.pop("elapsed_seconds", None),
        "event": event,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + "\n"
        )


def load_real_e1_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, RealDatasetContract]]:
    """Load the checksum-recorded real E1 protocol and typed dataset contracts."""

    config = _read_json(path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "preregistered_before_v7_real_refit"
        or config.get("estimator_commit") != FROZEN_ESTIMATOR_COMMIT
        or config.get("score_projection_commit") != FROZEN_SCORE_PROJECTION_COMMIT
    ):
        raise ValueError("v7 real E1 config schema or status is unsupported")
    estimator = config.get("estimator")
    spatial = config.get("spatial_des")
    release = config.get("release")
    datasets = config.get("datasets")
    if not all(
        isinstance(value, Mapping) for value in (estimator, spatial, release, datasets)
    ):
        raise ValueError("v7 real E1 config sections must be mappings")
    assert isinstance(estimator, Mapping)
    assert isinstance(spatial, Mapping)
    assert isinstance(release, Mapping)
    assert isinstance(datasets, Mapping)
    if (
        estimator.get("execution_profile") != "v7_primary_m0_m5_v1"
        or estimator.get("generators") != ["G0", "G2", "G3", "G4", "G5"]
        or estimator.get("inference_id") != "I1"
        or estimator.get("minimum_cells_per_sample_cell_type") != 10
        or spatial.get("sender_resolved_generators") != list(SENDER_RESOLVED_GENERATORS)
        or spatial.get("endpoints")
        != [
            "continuous_weighted_des",
            "diagnostic_one_se_native_count_des",
            "top_k_count_des",
        ]
        or spatial.get("event_budgets") != list(EVENT_BUDGETS)
        or spatial.get("top_k_scope") != "global_across_both_effect_directions"
        or spatial.get("score_type") != "pos"
        or spatial.get("weight_exponent") != 1.0
        or spatial.get("tie_policy") != "fgsea_native"
        or spatial.get("exclude_self_pairs") is not True
        or release.get("formal_inference_allowed") is not False
        or release.get("diagnostic_p_values_are_not_formal") is not True
        or release.get("real_data_edge_auroc_forbidden") is not True
        or release.get("spatial_colocalization_is_an_indirect_silver_endpoint")
        is not True
        or release.get("g4_sender_resolved_des_forbidden") is not True
        or release.get("ms_parameter_selection_forbidden") is not True
    ):
        raise ValueError("v7 real E1 estimator, endpoint, or release boundary changed")
    contracts: dict[str, RealDatasetContract] = {}
    for slug in ("kuppe", "ms"):
        record = datasets.get(slug)
        if not isinstance(record, Mapping):
            raise ValueError(f"v7 real E1 config lacks dataset {slug!r}")
        contracts[slug] = RealDatasetContract(
            slug=slug,
            dataset_id=str(record["dataset_id"]),
            role=str(record["role"]),
            condition_column=str(record["condition_column"]),
            reference=str(record["reference"]),
            target=str(record["target"]),
            contrast_name=str(record["contrast_name"]),
            batch_columns=tuple(map(str, record["batch_columns"])),
            seed=int(record["seed"]),
            outer_fold_partition_seed=int(record["outer_fold_partition_seed"]),
        )
    expected_contracts = {
        "kuppe": (
            "Kuppe_MI_CTRL_vs_IZ",
            "development_non_independent",
            "condition",
            "CTRL",
            "IZ",
            "IZ_vs_CTRL",
            (),
            20260717,
            20260717,
        ),
        "ms": (
            "LermaMartin_MS_CA_vs_Ctrl",
            "reused_locked_external_not_fresh_independent",
            "lesion_type",
            "Ctrl",
            "CA",
            "CA_vs_Ctrl",
            ("batch",),
            20260717,
            20260718,
        ),
    }
    observed_contracts = {
        slug: (
            item.dataset_id,
            item.role,
            item.condition_column,
            item.reference,
            item.target,
            item.contrast_name,
            item.batch_columns,
            item.seed,
            item.outer_fold_partition_seed,
        )
        for slug, item in contracts.items()
    }
    if observed_contracts != expected_contracts:
        raise ValueError("v7 real E1 dataset roles or design contracts changed")
    return config, contracts


def _load_resources(
    resource_path: Path,
    resource_manifest: Path,
    database_root: Path,
    *,
    nichenet_release: str,
    nichenet_manifest: Path | None,
) -> tuple[ResourceBundle, TargetPrior, dict[str, object]]:
    bundle, resource_provenance = kuppe_adapter.load_connectomedb2020_bundle(
        resource_path,
        resource_manifest,
    )
    target_prior = kuppe_adapter.load_nichenet_target_prior(
        database_root,
        release=nichenet_release,
        manifest_path=nichenet_manifest,
    )
    return (
        bundle,
        target_prior,
        {
            "lr_resource": resource_provenance,
            "target_prior": {
                "resource_id": target_prior.resource_id,
                "version": target_prior.version,
                "manifest_digest": target_prior.manifest_digest,
                "driver_kind": target_prior.driver_kind,
                "drivers": len(target_prior.driver_ids),
                "targets": len(target_prior.target_ids),
                "release": nichenet_release,
                "manifest": (
                    None
                    if nichenet_manifest is None
                    else _input_record(nichenet_manifest)
                ),
            },
        },
    )


def _load_dataset(
    contract: RealDatasetContract,
    input_path: Path,
    input_manifest: Path,
    *,
    min_cells: int,
) -> tuple[ad.AnnData, Any, CrossFitSpec, pd.DataFrame, dict[str, object]]:
    if contract.slug == "kuppe":
        preparation = kuppe_adapter.validate_preparation_manifest(
            input_path,
            input_manifest,
        )
        config, base_spec = kuppe_adapter.build_crossfit_configuration(
            seed=contract.seed,
            min_cells=min_cells,
        )
        data = ad.read_h5ad(input_path)
        metadata, support = kuppe_adapter._validate_input_contract(data, config)
        audit: dict[str, object] = {
            "preparation_schema": preparation["schema_version"],
            "subject_support": support,
        }
    else:
        preparation = ms_adapter.validate_subset_manifest(input_path, input_manifest)
        config, base_spec = ms_adapter.build_crossfit_configuration(
            seed=contract.seed,
            min_cells=min_cells,
            outer_fold_partition_seed=contract.outer_fold_partition_seed,
        )
        data = ad.read_h5ad(input_path)
        cohort_schema = str(
            preparation.get("schema_version")
            or cast(Mapping[str, object], preparation["lineage"])["schema_version"]
        )
        _, _, expected_samples, expected_subjects, expected_samples_per_subject = (
            ms_adapter._cohort_expectations(cohort_schema)
        )
        metadata, support, repeated = ms_adapter._validate_input_contract(
            data,
            config,
            expected_lineage=cast(Mapping[str, object], preparation["lineage"]),
            expected_samples=expected_samples,
            expected_subjects=expected_subjects,
            expected_samples_per_subject=expected_samples_per_subject,
        )
        audit = {
            "preparation_schema": cohort_schema,
            "subject_support": support,
            "repeated_subjects": repeated,
        }
    if base_spec.outer_fold_partition_seed != contract.outer_fold_partition_seed:
        raise ValueError("real E1 outer-fold partition seed differs from its contract")
    observed = set(metadata[contract.condition_column].astype(str))
    if observed != {contract.reference, contract.target}:
        raise ValueError(
            f"real E1 condition axis changed: expected "
            f"{[contract.reference, contract.target]}, observed={sorted(observed)}"
        )
    return data, config, base_spec, metadata, audit


def build_v7_real_crossfit_spec(
    base: CrossFitSpec,
    contract: RealDatasetContract,
    estimator_config: Mapping[str, object],
) -> CrossFitSpec:
    """Attach the frozen M0--M2 heads to an existing dataset fold contract."""

    return replace(
        base,
        absolute_activity_v2_spec=AbsoluteActivityV2Spec(
            bottleneck_penalty=float(estimator_config["bottleneck_penalty"]),
        ),
        sender_attribution_v2_spec=SenderAttributionV2Spec(
            minimum_calibration_subjects=int(
                estimator_config["minimum_sender_calibration_subjects"]
            ),
        ),
        signed_program_v2_spec=SignedProgramV2Spec(
            minimum_training_samples=int(
                estimator_config["minimum_signed_program_samples"]
            ),
            minimum_training_subjects=int(
                estimator_config["minimum_signed_program_subjects"]
            ),
        ),
        eb_shrunken_coupling_v2_spec=EBShrunkenCouplingV2Spec(
            contrast_name=contract.contrast_name,
            minimum_subjects=int(estimator_config["minimum_coupling_subjects"]),
            minimum_observed_edges_for_eb=int(
                estimator_config["minimum_coupling_edges_for_eb"]
            ),
            attribution_coupling_weight=float(
                estimator_config["attribution_coupling_weight"]
            ),
        ),
    )


def build_v7_real_design(
    contract: RealDatasetContract,
    estimator_config: Mapping[str, object],
) -> DifferentialDesignSpec:
    """Return the exact subject-level I1 design for one real cohort."""

    return DifferentialDesignSpec(
        design_kind="independent_two_group",
        condition_column=contract.condition_column,
        condition_levels=(contract.reference, contract.target),
        batch_columns=contract.batch_columns,
        contrasts=(
            DifferentialContrastSpec(
                name=contract.contrast_name,
                weights=((contract.reference, -1.0), (contract.target, 1.0)),
            ),
        ),
        precision_weight_column=None,
        minimum_subjects_per_level=int(estimator_config["minimum_subjects_per_level"]),
        minimum_clusters_for_cr2=int(estimator_config["minimum_clusters_for_cr2"]),
    )


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


def _resource_lookup(bundle: ResourceBundle) -> pd.DataFrame:
    table = pd.DataFrame.from_records(
        [
            {
                "interaction_id": item.interaction_id,
                "ligand": item.ligand_name,
                "receptor": item.receptor_name,
            }
            for item in bundle.interactions
        ]
    )
    if table.empty or table["interaction_id"].duplicated().any():
        raise ValueError(
            "real E1 resource interaction IDs must be non-empty and unique"
        )
    return table


def build_sender_resolved_event_ledger(
    effects: pd.DataFrame,
    bundle: ResourceBundle,
    contract: RealDatasetContract,
) -> pd.DataFrame:
    """Normalize v7 I1 sender effects for DES without releasing formal tests."""

    required = {
        "generator_id",
        "score_view",
        "inference_id",
        "contrast_name",
        "sender",
        "receiver",
        "interaction_id",
        "effect",
        "standard_error",
        "statistic",
        "p_value",
        "q_value",
        "formal_inference_allowed",
        "status",
    }
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"real E1 effects are missing columns: {sorted(missing)}")
    selected = effects.loc[
        effects["generator_id"].isin(SENDER_RESOLVED_GENERATORS)
        & effects["score_view"].eq(PRIMARY_SCORE_VIEW)
        & effects["inference_id"].eq("I1")
        & effects["contrast_name"].eq(contract.contrast_name)
    ].copy()
    if selected.empty:
        raise ValueError("real E1 effects contain no sender-resolved primary rows")
    observed_generators = set(selected["generator_id"].astype(str))
    if observed_generators != set(SENDER_RESOLVED_GENERATORS):
        raise ValueError(
            f"real E1 sender-resolved generators changed: {sorted(observed_generators)}"
        )
    if selected["formal_inference_allowed"].astype(bool).any():
        raise ValueError("real E1 sender ledger cannot consume formal inference fields")
    if selected[["p_value", "q_value"]].notna().any(axis=None):
        raise ValueError("real E1 sender ledger requires withheld formal p/q values")
    keys = ["generator_id", "sender", "receiver", "interaction_id"]
    if selected.duplicated(keys).any():
        raise ValueError("real E1 sender-resolved effect keys are duplicated")
    selected = selected.merge(
        _resource_lookup(bundle),
        on="interaction_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if selected[["ligand", "receptor"]].isna().any(axis=None):
        raise ValueError("real E1 effects do not map exactly to the frozen LR resource")
    effect = pd.to_numeric(selected["effect"], errors="coerce")
    standard_error = pd.to_numeric(selected["standard_error"], errors="coerce")
    statistic = pd.to_numeric(selected["statistic"], errors="coerce")
    observed = selected["status"].astype(str).eq("observed")
    finite_observed = (
        np.isfinite(effect) & np.isfinite(standard_error) & np.isfinite(statistic)
    )
    if not finite_observed.loc[observed].all():
        raise ValueError(
            "observed real E1 effects require finite effect, SE, and statistic"
        )
    if not standard_error.loc[observed].gt(0.0).all():
        raise ValueError("observed real E1 effects require a positive standard error")
    selected["reference_condition"] = contract.reference
    selected["target_condition"] = contract.target
    selected["effect_target_minus_reference"] = effect
    selected["effect_signal_to_noise"] = statistic
    selected["condition"] = np.select(
        (effect.gt(0.0), effect.lt(0.0)),
        (contract.target, contract.reference),
        default="tied",
    )
    selected["abs_effect"] = effect.abs()
    selected["event_evidence"] = statistic.abs()
    selected["one_standard_error_stable"] = (
        observed
        & standard_error.gt(0.0)
        & np.isfinite(standard_error)
        & effect.abs().gt(standard_error)
    )
    selected["native_selected"] = selected["one_standard_error_stable"] & effect.ne(0.0)
    sender = selected["sender"].astype(str).to_numpy()
    receiver = selected["receiver"].astype(str).to_numpy()
    selected["pair_sender"] = np.minimum(sender, receiver)
    selected["pair_receiver"] = np.maximum(sender, receiver)
    selected["formal_inference_allowed"] = False
    return selected.sort_values(
        ["generator_id", "condition", "sender", "receiver", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )


def _pair_axes(
    cell_types: Sequence[str],
    contract: RealDatasetContract,
) -> pd.DataFrame:
    names = tuple(sorted(set(map(str, cell_types))))
    if len(names) < 2 or any(not name or name != name.strip() for name in names):
        raise ValueError("real E1 DES requires at least two canonical cell types")
    records = [
        {
            "condition": condition,
            "sender": names[left],
            "receiver": names[right],
            "ranked_strength": 0.0,
            "status": "observed",
        }
        for condition in (contract.reference, contract.target)
        for left in range(len(names))
        for right in range(left, len(names))
    ]
    return pd.DataFrame.from_records(records)


def _load_expected_sets(
    truth_manifest_path: Path,
    *,
    dataset_slug: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    contract = DES_CONTRACTS[dataset_slug]
    manifest = _read_json(truth_manifest_path)
    if (
        manifest.get("schema_version") != contract["truth_schema"]
        or manifest.get("status") != "complete"
        or manifest.get("dataset_id") != contract["truth_dataset"]
    ):
        raise ValueError("real E1 spatial truth manifest does not match its contract")
    outputs = manifest.get("outputs")
    record = outputs.get("expected_sets") if isinstance(outputs, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError("real E1 spatial truth lacks expected_sets")
    filename = record.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("real E1 spatial expected-set filename is invalid")
    path = truth_manifest_path.parent / filename
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError("real E1 spatial expected-set checksum mismatch")
    return (
        pd.read_csv(path, sep="\t"),
        manifest,
        {
            "manifest": _input_record(truth_manifest_path),
            "expected_sets": _input_record(path),
        },
    )


def evaluate_v7_real_spatial_des(
    ledger: pd.DataFrame,
    *,
    pair_axes: pd.DataFrame,
    expected: pd.DataFrame,
    contract: RealDatasetContract,
    resource_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate continuous, diagnostic native, and fixed-K sender DES tracks."""

    des_contract = DES_CONTRACTS[contract.slug]
    score_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    ranking_parts: list[pd.DataFrame] = []
    for generator in SENDER_RESOLVED_GENERATORS:
        local = ledger.loc[ledger["generator_id"].eq(generator)].copy()
        observed = local["status"].astype(str).eq("observed")
        endpoint_specs: list[tuple[str, int | None, pd.Series, str | None]] = [
            (
                "continuous_weighted_des",
                None,
                observed & local["effect_target_minus_reference"].ne(0.0),
                "abs_effect",
            ),
            (
                "diagnostic_one_se_native_count_des",
                None,
                local["native_selected"].astype(bool),
                None,
            ),
        ]
        eligible = observed & local["effect_target_minus_reference"].ne(0.0)
        for budget in EVENT_BUDGETS:
            selected = select_top_k_events_by_scope(
                local,
                budget=budget,
                evidence_column="event_evidence",
                eligible=eligible,
                scope="global_across_both_effect_directions",
            )
            endpoint_specs.append(("top_k_count_des", budget, selected, None))
        for endpoint, budget, selected, weight_column in endpoint_specs:
            metadata = {
                "dataset": contract.dataset_id,
                "method": f"CRYCHIC_v7_{generator}_I1",
                "method_version": "suggest-next2-v7-primary",
                "resource": resource_id,
                "ranking_semantics": (
                    "sum_absolute_subject_level_effect"
                    if weight_column is not None
                    else "count_selected_directed_lr_events"
                ),
            }
            rankings = pair_rankings_from_events(
                local,
                pair_axes,
                selected=selected,
                weight_column=weight_column,
                metadata=metadata,
            )
            scores, coverage = evaluate_rankings(
                rankings,
                expected,
                scenario="multi_sample",
                dataset_map={contract.dataset_id: str(des_contract["truth_dataset"])},
                condition_map=dict(des_contract["condition_map"]),
                expected_filters=dict(des_contract["expected_filters"]),
                score_type="pos",
                weight_exponent=1.0,
                tie_policy="fgsea_native",
                exclude_self_pairs=True,
                ranking_statistic="raw_cardinality",
            )
            for table in (rankings, scores, coverage):
                table.insert(0, "generator_id", generator)
                table.insert(1, "des_variant", endpoint)
                table.insert(2, "event_budget", budget)
            ranking_parts.append(rankings)
            score_parts.append(scores)
            coverage_parts.append(coverage)
    return (
        pd.concat(ranking_parts, ignore_index=True, sort=False),
        pd.concat(score_parts, ignore_index=True, sort=False),
        pd.concat(coverage_parts, ignore_index=True, sort=False),
    )


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


def run(
    *,
    dataset: str,
    input_h5ad: Path,
    input_manifest: Path,
    output_dir: Path,
    resource_path: Path,
    resource_manifest: Path,
    database_root: Path,
    truth_manifest: Path,
    config_path: Path = DEFAULT_CONFIG,
    nichenet_release: str = "v2_2021",
    nichenet_manifest: Path | None = None,
    threads: int = 1,
    fold_jobs: int = 1,
    min_cells: int = 10,
    overwrite: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Fit one checksum-bound real cohort and persist v7 E1 evidence."""

    if dataset not in {"kuppe", "ms"}:
        raise ValueError("dataset must be 'kuppe' or 'ms'")
    for name, value in (
        ("threads", threads),
        ("fold_jobs", fold_jobs),
        ("min_cells", min_cells),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    protocol, contracts = load_real_e1_config(config_path)
    contract = contracts[dataset]
    estimator_config = cast(Mapping[str, object], protocol["estimator"])
    frozen_min_cells = int(estimator_config["minimum_cells_per_sample_cell_type"])
    if min_cells != frozen_min_cells:
        raise ValueError(
            f"real E1 requires frozen min_cells={frozen_min_cells}, "
            f"observed={min_cells}"
        )
    code = git_metadata(REPOSITORY_ROOT)
    if code["dirty"] and not allow_dirty:
        raise RuntimeError("v7 real E1 benchmark refuses a dirty worktree")
    output = prepare_output(output_dir, overwrite=overwrite)
    started = time.perf_counter()
    log_path = output / "run.jsonl"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "dataset": contract.slug,
        "dataset_id": contract.dataset_id,
        "dataset_role": contract.role,
        "formal_inference_allowed": False,
        "analysis_unit": {
            "replicate_key": "subject_id",
            "subject_key": "subject_id",
            "primary_panel": True,
        },
        "inputs": {
            "h5ad": _input_record(input_h5ad),
            "preparation_manifest": _input_record(input_manifest),
            "protocol": _input_record(config_path),
            "spatial_truth": _input_record(truth_manifest),
        },
        "parameters": {
            "threads": threads,
            "fold_jobs": fold_jobs,
            "min_cells": min_cells,
            "nichenet_release": nichenet_release,
        },
        "code": code,
        "outputs": None,
        "failure": None,
    }
    write_json(output / "manifest.json", manifest)
    _append_log(
        log_path,
        "run_started",
        elapsed_seconds=0.0,
        dataset=contract.slug,
        dataset_role=contract.role,
        fold_jobs=fold_jobs,
        threads=threads,
    )
    data: ad.AnnData | None = None
    try:
        bundle, target_prior, resource_provenance = _load_resources(
            resource_path,
            resource_manifest,
            database_root,
            nichenet_release=nichenet_release,
            nichenet_manifest=nichenet_manifest,
        )
        _append_log(
            log_path,
            "resources_loaded",
            elapsed_seconds=time.perf_counter() - started,
            interactions=len(bundle.interactions),
            target_prior_drivers=len(target_prior.driver_ids),
        )
        data, crychic_config, base_spec, sample_metadata, input_audit = _load_dataset(
            contract,
            input_h5ad,
            input_manifest,
            min_cells=min_cells,
        )
        _append_log(
            log_path,
            "dataset_validated",
            elapsed_seconds=time.perf_counter() - started,
            cells=int(data.n_obs),
            genes=int(data.n_vars),
            samples=int(sample_metadata[crychic_config.sample_key].nunique()),
            subjects=int(sample_metadata[crychic_config.subject_key].nunique()),
        )
        if contract.slug == "kuppe":
            compatibility = kuppe_adapter._validate_resource_compatibility(
                data,
                bundle,
                target_prior,
            )
        else:
            compatibility = ms_adapter._validate_resource_compatibility(
                data,
                bundle,
                target_prior,
            )
        crossfit_spec = build_v7_real_crossfit_spec(
            base_spec,
            contract,
            estimator_config,
        )
        design = build_v7_real_design(contract, estimator_config)
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
            crossfit,
            dataset_id=contract.dataset_id,
        )
        fit_cache = V7InferenceFitCache()
        effects = run_v7_inference_matrix(
            score_views,
            design=design,
            sample_metadata=sample_metadata,
            arms=tuple(
                (generator, "I1") for generator in ("G0", "G2", "G3", "G4", "G5")
            ),
            fit_cache=fit_cache,
        )
        _append_log(
            log_path,
            "inference_completed",
            elapsed_seconds=time.perf_counter() - started,
            effect_rows=len(effects),
            score_rows=len(score_views),
        )
        ledger = build_sender_resolved_event_ledger(effects, bundle, contract)
        expected, truth, truth_provenance = _load_expected_sets(
            truth_manifest,
            dataset_slug=contract.slug,
        )
        rankings, des_scores, des_coverage = evaluate_v7_real_spatial_des(
            ledger,
            pair_axes=_pair_axes(
                data.obs[crychic_config.cell_type_key].astype(str).unique(),
                contract,
            ),
            expected=expected,
            contract=contract,
            resource_id=bundle.resource_id,
        )
        _append_log(
            log_path,
            "spatial_des_completed",
            elapsed_seconds=time.perf_counter() - started,
            score_rows=len(des_scores),
        )
        score_geometry = _score_geometry(score_views)
        effect_summary = _effect_summary(effects)
        des_summary = (
            des_scores.loc[des_scores["status"].astype(str).eq("observed")]
            .groupby(
                ["generator_id", "des_variant", "event_budget"],
                observed=True,
                sort=True,
                dropna=False,
            )["des"]
            .agg(["count", "median", "mean", "min", "max"])
            .reset_index()
        )
        tables = {
            "score_views.parquet": score_views,
            "effects.parquet": effects,
            "sender_resolved_event_ledger.parquet": ledger,
            "condition_cell_pair_rankings.tsv": rankings,
            "spatial_des_scores.tsv": des_scores,
            "spatial_des_coverage.tsv": des_coverage,
            "spatial_des_summary.tsv": des_summary,
            "score_geometry.tsv": score_geometry,
            "effect_summary.tsv": effect_summary,
        }
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
                "resource": resource_provenance,
                "resource_compatibility": compatibility,
                "crossfit": _crossfit_manifest(crossfit),
                "crossfit_spec": crossfit_spec.to_dict(),
                "differential_design": design.to_dict(),
                "inference_cache": fit_cache.to_dict(),
                "g0_g2_equivalence": g0_g2_equivalence_diagnostic(score_views),
                "spatial_truth": {
                    "schema_version": truth["schema_version"],
                    "dataset_id": truth["dataset_id"],
                    **truth_provenance,
                },
                "environment": python_environment(
                    environment_name="crychic_project_python",
                    packages=(
                        "CRYCHIC",
                        "anndata",
                        "numpy",
                        "pandas",
                        "pyarrow",
                        "scipy",
                        "statsmodels",
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
    finally:
        if data is not None and data.isbacked:
            data.file.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kuppe", "ms"), required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--nichenet-release", default="v2_2021")
    parser.add_argument("--nichenet-manifest", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fold-jobs", type=int, default=1)
    parser.add_argument("--min-cells", type=int, default=10)
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
        truth_manifest=arguments.truth_manifest.resolve(),
        config_path=arguments.config.resolve(),
        nichenet_release=arguments.nichenet_release,
        nichenet_manifest=(
            None
            if arguments.nichenet_manifest is None
            else arguments.nichenet_manifest.resolve()
        ),
        threads=arguments.threads,
        fold_jobs=arguments.fold_jobs,
        min_cells=arguments.min_cells,
        overwrite=arguments.overwrite,
        allow_dirty=arguments.allow_dirty,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
