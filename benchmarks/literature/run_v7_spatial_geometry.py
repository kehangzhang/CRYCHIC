"""Run frozen sparse Visium geometry diagnostics for Kuppe or MS."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import psutil  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits

from benchmarks.adapters.common import (
    git_metadata,
    json_safe,
    prepare_output,
    python_environment,
    sha256_file,
    write_json,
)
from benchmarks.literature.prepare_kuppe_misty_des_truth import (
    FROZEN_SLIDES,
    ONTOLOGY_ALIASES,
    RAW_CELL_TYPES,
)
from benchmarks.literature.prepare_ms_spatial_des_truth import (
    CELL_TYPES as MS_CELL_TYPES,
)
from benchmarks.literature.prepare_ms_spatial_des_truth import (
    MS_SPATIAL_SAMPLES,
)
from benchmarks.literature.spatial_geometry import (
    DistanceBand,
    SectionGeometryResult,
    cell_type_permutations,
    compute_section_geometry,
    condition_geometry_effects,
    distance_bands,
    distance_decay_table,
    expected_geometry_sets,
    section_association_table,
)

SCHEMA_VERSION = "crychic-suggest-next2-v7-spatial-geometry-run-v1"
CONFIG_SCHEMA_VERSION = "crychic-suggest-next2-v7-spatial-geometry-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "benchmarks/configs/suggest_next2_v7_spatial_geometry_v1.json"
)
PROFILES = ("smoke", "formal")
DATASETS = ("kuppe", "ms")
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True, slots=True)
class SpatialDatasetContract:
    slug: str
    dataset_id: str
    reference: str
    target: str
    input_schema: str
    truth_manifest_sha256: str
    samples: int
    cell_types: int
    primary_analysis_unit: str
    sensitivity_analysis_unit: str
    input_manifest_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SpatialSectionSpec:
    dataset: str
    sample_id: str
    subject_id: str
    condition: str
    table_path: Path
    table_sha256: str
    expected_rows: int
    raw_cell_types: tuple[str, ...]
    output_cell_types: tuple[str, ...]


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


def _append_log(path: Path, event: str, **fields: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                json_safe({"event": event, **fields}),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_spatial_geometry_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, SpatialDatasetContract], tuple[DistanceBand, ...]]:
    config = _read_json(path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "preregistered_before_geometry_metric_inspection"
    ):
        raise ValueError("spatial geometry config schema or status changed")
    datasets = config.get("datasets")
    geometry = config.get("geometry")
    nulls = config.get("nulls")
    comparison = config.get("condition_comparison")
    alignment = config.get("algorithm_alignment")
    execution = config.get("execution")
    release = config.get("release")
    if not all(
        isinstance(value, Mapping)
        for value in (
            datasets,
            geometry,
            nulls,
            comparison,
            alignment,
            execution,
            release,
        )
    ):
        raise ValueError("spatial geometry config sections must be mappings")
    assert isinstance(datasets, Mapping)
    assert isinstance(geometry, Mapping)
    assert isinstance(nulls, Mapping)
    assert isinstance(comparison, Mapping)
    assert isinstance(alignment, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(release, Mapping)
    if set(datasets) != set(DATASETS):
        raise ValueError("spatial geometry dataset axis changed")
    if (
        geometry.get("coordinate_system") != "visium_hex_array_lattice_v1"
        or geometry.get("x_formula") != "array_col / 2"
        or geometry.get("y_formula") != "sqrt(3) * array_row / 2"
        or geometry.get("distance_unit") != "nearest_neighbor_spot_spacing"
        or geometry.get("association")
        != "symmetric_row_normalized_bivariate_cross_moran_v1"
        or geometry.get("standardization")
        != "within_section_rank_normal_then_population_sd"
        or geometry.get("component_policy")
        != "B1_contact_connected_components_no_cross_component_pairs"
        or geometry.get("include_self_cell_pairs") is not False
        or geometry.get("spot_pair_policy")
        != "all_sparse_pairs_within_band_and_contact_component"
    ):
        raise ValueError("spatial geometry metric contract changed")
    raw_bands = geometry.get("distance_bands")
    if not isinstance(raw_bands, Sequence):
        raise ValueError("spatial geometry distance bands must be a sequence")
    bands = distance_bands(cast(Sequence[Mapping[str, object]], raw_bands))
    observed_bands = tuple(
        (band.name, band.lower_exclusive, band.upper_inclusive) for band in bands
    )
    expected_bands = (
        ("contact", 0.0, 1.01),
        ("short", 1.01, 2.01),
        ("local", 2.01, 4.01),
        ("diffuse", 4.01, 8.01),
    )
    if observed_bands != expected_bands:
        raise ValueError("spatial geometry distance-band contract changed")
    expected_nulls = {
        "smoke_coordinate_permutations": 49,
        "formal_coordinate_permutations": 999,
        "coordinate_rule": (
            "within_slide_abundance_row_permutation_shared_across_bands"
        ),
        "smoke_cell_label_permutations": 49,
        "formal_cell_label_permutations": 999,
        "cell_label_rule": (
            "one_dataset_global_cell_type_bijection_shared_across_all_sections_"
            "and_conditions_per_replicate"
        ),
        "empirical_p_rule": "plus_one_two_sided_absolute_association",
        "coordinate_max_t_family": ("all_unordered_cell_pairs_within_slide_and_band"),
        "seed": 20260729,
    }
    if dict(nulls) != expected_nulls:
        raise ValueError("spatial geometry null contract changed")
    expected_comparison = {
        "effect": "target_mean_minus_reference_mean",
        "rank": "absolute_effect_descending_then_canonical_pair",
        "minimum_units_per_condition": 2,
        "top_fractions": [0.1, 0.2, 0.3, 0.4],
        "primary_unit": "subject_id",
        "sensitivity_unit": "sample_id",
        "formal_inference_allowed": False,
    }
    if dict(comparison) != expected_comparison:
        raise ValueError("spatial geometry comparison contract changed")
    expected_alignment = {
        "analysis_unit": "subject_id",
        "generators": ["G0", "G2", "G3", "G5"],
        "mechanisms": ["all", "contact", "secreted"],
        "evaluated_bands": ["contact", "short", "local", "diffuse"],
        "mechanism_band_hypotheses": {
            "all": ["contact", "short", "local", "diffuse"],
            "contact": ["contact"],
            "secreted": ["local", "diffuse"],
        },
        "datasets": {
            "kuppe": {
                "algorithm_dataset_id": "Kuppe_MI_CTRL_vs_IZ",
                "geometry_dataset_id": "Kuppe_MI_spatial_CTRL_vs_IZ",
                "condition_map_algorithm_to_geometry": {
                    "CTRL": "CTRL",
                    "IZ": "IZ",
                },
            },
            "ms": {
                "algorithm_dataset_id": "LermaMartin_MS_CA_vs_Ctrl",
                "geometry_dataset_id": (
                    "lerma_martin_ms_ctrl_vs_chronic_active"
                ),
                "condition_map_algorithm_to_geometry": {
                    "Ctrl": "control",
                    "CA": "chronic_active",
                },
            },
        },
        "resource": {
            "resource_id": "ConnectomeDB2020_Hou_2020_human",
            "payload_sha256": (
                "e781363288a26c15e03246500111bfecb818eef997f5ebe1b936aaa465151c3a"
            ),
            "manifest_sha256": (
                "3dd10324ae0fc3b903ee09fece3fbeb1933db94c92fe1fbf4eb644d407d6b7af"
            ),
        },
        "event_endpoints": [
            "continuous_weighted_des",
            "diagnostic_one_se_native_count_des",
            "top_k_count_des",
        ],
        "event_budgets": [100, 250, 500, 1000],
        "top_k_scope": "global_across_both_effect_directions",
        "continuous_weight": "abs_effect",
        "top_k_evidence": "event_evidence",
        "truth_variants": [
            "rank_top",
            "coordinate_max_t_supported_rank_top",
            "cell_label_max_t_supported_rank_top",
            "coordinate_and_cell_label_max_t_supported_rank_top",
        ],
        "diagnostic_alpha": 0.05,
        "direction_collapse": (
            "sum_both_directed_lr_events_to_unordered_pair"
        ),
        "exclude_self_pairs": True,
        "score_type": "pos",
        "weight_exponent": 1.0,
        "tie_policy": "fgsea_native",
        "ranking_statistic": "raw_cardinality",
        "formal_inference_allowed": False,
    }
    if dict(alignment) != expected_alignment:
        raise ValueError("spatial geometry algorithm-alignment contract changed")
    if dict(execution) != {
        "maximum_memory_fraction": 0.8,
        "default_sample_jobs": 13,
        "blas_threads_per_worker": 1,
    }:
        raise ValueError("spatial geometry execution contract changed")
    expected_release = {
        "geometry_is_indirect_silver_evidence": True,
        "coordinate_and_label_p_values_are_endpoint_diagnostics_only": True,
        "ccc_formal_inference_allowed": False,
        "causal_sender_claim_allowed": False,
        "ordinary_spot_pearson_is_not_spatial_geometry": True,
        "cross_platform_replication_status": (
            "not_estimable_both_cohorts_are_visium_and_cell_ontologies_do_not_align"
        ),
    }
    if dict(release) != expected_release:
        raise ValueError("spatial geometry release boundary changed")
    contracts: dict[str, SpatialDatasetContract] = {}
    for slug in DATASETS:
        record = datasets.get(slug)
        if not isinstance(record, Mapping):
            raise ValueError(f"spatial geometry config lacks {slug}")
        hashes = [record.get("truth_manifest_sha256")]
        if slug == "kuppe":
            hashes.append(record.get("input_manifest_sha256"))
        if any(not _valid_sha256(value) for value in hashes):
            raise ValueError(f"spatial geometry hashes are invalid for {slug}")
        contracts[slug] = SpatialDatasetContract(
            slug=slug,
            dataset_id=str(record["dataset_id"]),
            reference=str(record["reference"]),
            target=str(record["target"]),
            input_schema=str(record["input_schema"]),
            truth_manifest_sha256=str(record["truth_manifest_sha256"]),
            samples=int(record["samples"]),
            cell_types=int(record["cell_types"]),
            primary_analysis_unit=str(record["primary_analysis_unit"]),
            sensitivity_analysis_unit=str(record["sensitivity_analysis_unit"]),
            input_manifest_sha256=(
                None
                if record.get("input_manifest_sha256") is None
                else str(record["input_manifest_sha256"])
            ),
        )
    expected_contracts = {
        "kuppe": (
            "Kuppe_MI_spatial_CTRL_vs_IZ",
            "CTRL",
            "IZ",
            13,
            11,
            "subject_id",
            "sample_id",
        ),
        "ms": (
            "lerma_martin_ms_ctrl_vs_chronic_active",
            "control",
            "chronic_active",
            11,
            9,
            "subject_id",
            "sample_id",
        ),
    }
    observed_contracts = {
        slug: (
            contract.dataset_id,
            contract.reference,
            contract.target,
            contract.samples,
            contract.cell_types,
            contract.primary_analysis_unit,
            contract.sensitivity_analysis_unit,
        )
        for slug, contract in contracts.items()
    }
    if observed_contracts != expected_contracts:
        raise ValueError("spatial geometry dataset contract changed")
    return config, contracts, bands


def _validate_truth_manifest(
    path: Path,
    contract: SpatialDatasetContract,
) -> dict[str, Any]:
    if sha256_file(path) != contract.truth_manifest_sha256:
        raise ValueError("spatial geometry truth-manifest checksum mismatch")
    manifest = _read_json(path)
    if (
        manifest.get("schema_version")
        not in {"crychic-kuppe-misty-des-truth-v1", "crychic-ms-spatial-des-truth-v1"}
        or manifest.get("status") != "complete"
        or manifest.get("dataset_id") != contract.dataset_id
    ):
        raise ValueError("spatial geometry legacy truth manifest changed")
    return manifest


def _kuppe_sections(
    input_root: Path,
    input_manifest_path: Path,
    truth_manifest_path: Path,
    contract: SpatialDatasetContract,
) -> tuple[list[SpatialSectionSpec], dict[str, object]]:
    if contract.input_manifest_sha256 is None or (
        sha256_file(input_manifest_path) != contract.input_manifest_sha256
    ):
        raise ValueError("Kuppe spatial input-manifest checksum mismatch")
    manifest = _read_json(input_manifest_path)
    truth = _validate_truth_manifest(truth_manifest_path, contract)
    slides = manifest.get("slides")
    if (
        manifest.get("schema_version") != contract.input_schema
        or manifest.get("status") != "complete"
        or manifest.get("dataset_id") != contract.dataset_id
        or not isinstance(slides, list)
        or len(slides) != contract.samples
    ):
        raise ValueError("Kuppe spatial input manifest changed")
    by_sample = {slide.sample_id: slide for slide in FROZEN_SLIDES}
    if set(by_sample) != {
        str(record.get("sample_id")) for record in slides if isinstance(record, Mapping)
    }:
        raise ValueError("Kuppe spatial slide roster changed")
    output_cell_types = tuple(
        ONTOLOGY_ALIASES.get(name, name) for name in RAW_CELL_TYPES
    )
    sections: list[SpatialSectionSpec] = []
    records: list[dict[str, object]] = []
    for raw in slides:
        if not isinstance(raw, Mapping):
            raise ValueError("Kuppe spatial slide record must be a mapping")
        sample_id = str(raw["sample_id"])
        frozen = by_sample[sample_id]
        input_record = raw.get("input")
        spots = raw.get("spots")
        if not isinstance(input_record, Mapping) or not isinstance(spots, Mapping):
            raise ValueError("Kuppe spatial slide lacks input or spot provenance")
        filename = str(input_record["filename"])
        if Path(filename).name != filename:
            raise ValueError("Kuppe spatial input filename must be a basename")
        table_path = input_root / filename
        expected_sha = str(input_record["sha256"])
        if sha256_file(table_path) != expected_sha:
            raise ValueError(f"Kuppe spatial table checksum mismatch: {sample_id}")
        if raw.get("condition") != frozen.condition:
            raise ValueError("Kuppe spatial section condition changed")
        sections.append(
            SpatialSectionSpec(
                dataset=contract.slug,
                sample_id=sample_id,
                subject_id=frozen.subject_id,
                condition=frozen.condition,
                table_path=table_path,
                table_sha256=expected_sha,
                expected_rows=int(spots["retained"]),
                raw_cell_types=tuple(RAW_CELL_TYPES),
                output_cell_types=output_cell_types,
            )
        )
        records.append(_input_record(table_path))
    return sections, {
        "input_manifest": _input_record(input_manifest_path),
        "legacy_truth_manifest": _input_record(truth_manifest_path),
        "legacy_truth_schema": truth["schema_version"],
        "section_tables": records,
    }


def _ms_sections(
    input_root: Path,
    truth_manifest_path: Path,
    contract: SpatialDatasetContract,
) -> tuple[list[SpatialSectionSpec], dict[str, object]]:
    truth = _validate_truth_manifest(truth_manifest_path, contract)
    source = truth.get("source")
    files = source.get("files") if isinstance(source, Mapping) else None
    if not isinstance(files, list) or len(files) != contract.samples:
        raise ValueError("MS spatial truth lacks its frozen section provenance")
    by_sample = {
        str(record.get("sample_id")): record
        for record in files
        if isinstance(record, Mapping)
    }
    if set(by_sample) != {sample.sample_id for sample in MS_SPATIAL_SAMPLES}:
        raise ValueError("MS spatial section roster changed")
    sections: list[SpatialSectionSpec] = []
    records: list[dict[str, object]] = []
    for sample in MS_SPATIAL_SAMPLES:
        source_record = by_sample[sample.sample_id]
        table_path = input_root / f"{sample.file_stem}.meta.tsv"
        dataset_path = input_root / f"{sample.file_stem}.dataset.json"
        expected_table_sha = str(source_record["meta_sha256"])
        expected_dataset_sha = str(source_record["dataset_sha256"])
        if (
            sha256_file(table_path) != expected_table_sha
            or sha256_file(dataset_path) != expected_dataset_sha
            or source_record.get("subject_id") != sample.subject_id
            or source_record.get("condition") != sample.condition
        ):
            raise ValueError(
                f"MS spatial section provenance changed: {sample.sample_id}"
            )
        sections.append(
            SpatialSectionSpec(
                dataset=contract.slug,
                sample_id=sample.sample_id,
                subject_id=sample.subject_id,
                condition=sample.condition,
                table_path=table_path,
                table_sha256=expected_table_sha,
                expected_rows=int(source_record["meta_rows"]),
                raw_cell_types=tuple(MS_CELL_TYPES),
                output_cell_types=tuple(MS_CELL_TYPES),
            )
        )
        records.extend((_input_record(table_path), _input_record(dataset_path)))
    return sections, {
        "legacy_truth_manifest": _input_record(truth_manifest_path),
        "legacy_truth_schema": truth["schema_version"],
        "section_files": records,
    }


def load_sections(
    *,
    dataset: str,
    input_root: Path,
    input_manifest_path: Path | None,
    truth_manifest_path: Path,
    contract: SpatialDatasetContract,
) -> tuple[list[SpatialSectionSpec], dict[str, object]]:
    if dataset == "kuppe":
        if input_manifest_path is None:
            raise ValueError("Kuppe spatial geometry requires its input manifest")
        return _kuppe_sections(
            input_root,
            input_manifest_path,
            truth_manifest_path,
            contract,
        )
    if input_manifest_path is not None:
        raise ValueError("MS spatial geometry uses its truth manifest as provenance")
    return _ms_sections(input_root, truth_manifest_path, contract)


def _section_worker(payload: Mapping[str, object]) -> SectionGeometryResult:
    spec = cast(SpatialSectionSpec, payload["spec"])
    bands = cast(tuple[DistanceBand, ...], payload["bands"])
    table = pd.read_csv(
        spec.table_path, sep="\t", compression="infer", low_memory=False
    )
    required = {"array_row", "array_col", *spec.raw_cell_types}
    missing = required.difference(table.columns)
    if missing or len(table) != spec.expected_rows:
        raise ValueError(
            f"spatial section {spec.sample_id} columns or row count changed: "
            f"missing={sorted(missing)}, rows={len(table)}"
        )
    abundance = table.loc[:, list(spec.raw_cell_types)].apply(
        pd.to_numeric, errors="coerce"
    )
    values = abundance.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError(f"spatial section {spec.sample_id} abundance contract changed")
    abundance.columns = list(spec.output_cell_types)
    with threadpool_limits(limits=1):
        return compute_section_geometry(
            dataset=spec.dataset,
            sample_id=spec.sample_id,
            subject_id=spec.subject_id,
            condition=spec.condition,
            coordinates=table.loc[:, ["array_row", "array_col"]],
            abundance=abundance,
            bands=bands,
            coordinate_permutations=int(payload["coordinate_permutations"]),
            seed=int(payload["seed"]),
        )


def _system_memory_fraction() -> float:
    memory = psutil.virtual_memory()
    return float(1.0 - memory.available / memory.total)


def _geometry_summary(
    effects: pd.DataFrame,
    decay: pd.DataFrame,
) -> pd.DataFrame:
    effect_summary = (
        effects.assign(
            coordinate_reject=effects["coordinate_empirical_p_value"].le(0.05),
            coordinate_max_t_reject=effects["coordinate_max_t_p_value"].le(0.05),
            label_reject=effects["cell_label_empirical_p_value"].le(0.05),
        )
        .groupby(["analysis_unit", "band"], observed=True, sort=True)
        .agg(
            pair_rows=("sender", "size"),
            observed_pairs=("status", lambda values: values.eq("observed").sum()),
            median_absolute_effect=("absolute_effect", "median"),
            median_coordinate_null_z=("coordinate_null_z", "median"),
            coordinate_rejection_fraction=("coordinate_reject", "mean"),
            coordinate_max_t_rejection_fraction=("coordinate_max_t_reject", "mean"),
            cell_label_rejection_fraction=("label_reject", "mean"),
        )
        .reset_index()
    )
    decay_observed = decay.loc[decay["status"].eq("observed")]
    decay_record = pd.DataFrame.from_records(
        [
            {
                "analysis_unit": "sample_id",
                "band": "distance_decay",
                "pair_rows": len(decay),
                "observed_pairs": len(decay_observed),
                "median_absolute_effect": float(
                    decay_observed[
                        "absolute_association_slope_per_log_distance"
                    ].median()
                ),
                "median_coordinate_null_z": math.nan,
                "coordinate_rejection_fraction": math.nan,
                "coordinate_max_t_rejection_fraction": math.nan,
                "cell_label_rejection_fraction": math.nan,
            }
        ]
    )
    return pd.concat([effect_summary, decay_record], ignore_index=True)


def run(
    *,
    dataset: str,
    profile: str,
    input_root: Path,
    truth_manifest_path: Path,
    output_dir: Path,
    input_manifest_path: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    sample_jobs: int | None = None,
    overwrite: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    if dataset not in DATASETS or profile not in PROFILES:
        raise ValueError("unsupported spatial geometry dataset or profile")
    protocol, contracts, bands = load_spatial_geometry_config(config_path)
    contract = contracts[dataset]
    nulls = cast(Mapping[str, object], protocol["nulls"])
    comparison = cast(Mapping[str, object], protocol["condition_comparison"])
    execution = cast(Mapping[str, object], protocol["execution"])
    coordinate_replicates = int(nulls[f"{profile}_coordinate_permutations"])
    label_replicates = int(nulls[f"{profile}_cell_label_permutations"])
    jobs = int(execution["default_sample_jobs"]) if sample_jobs is None else sample_jobs
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("sample_jobs must be a positive integer")
    code = git_metadata(REPOSITORY_ROOT)
    if code["dirty"] and not allow_dirty:
        raise RuntimeError("spatial geometry benchmark refuses a dirty worktree")
    sections, input_provenance = load_sections(
        dataset=dataset,
        input_root=input_root,
        input_manifest_path=input_manifest_path,
        truth_manifest_path=truth_manifest_path,
        contract=contract,
    )
    jobs = min(jobs, len(sections))
    maximum_memory_fraction = float(execution["maximum_memory_fraction"])
    if _system_memory_fraction() >= maximum_memory_fraction:
        raise RuntimeError("system memory is already above the spatial geometry limit")
    output = prepare_output(output_dir, overwrite=overwrite)
    log_path = output / "run.jsonl"
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "dataset": dataset,
        "dataset_id": contract.dataset_id,
        "profile": profile,
        "formal_inference_allowed": False,
        "inputs": {
            "protocol": _input_record(config_path),
            **input_provenance,
        },
        "parameters": {
            "coordinate_permutations": coordinate_replicates,
            "cell_label_permutations": label_replicates,
            "sample_jobs": jobs,
            "maximum_memory_fraction": maximum_memory_fraction,
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
        dataset=dataset,
        profile=profile,
        sections=len(sections),
        sample_jobs=jobs,
    )
    try:
        payloads = [
            {
                "spec": section,
                "bands": bands,
                "coordinate_permutations": coordinate_replicates,
                "seed": int(nulls["seed"]),
            }
            for section in sections
        ]
        results: list[SectionGeometryResult] = []
        completed = 0
        durations: list[float] = []
        if jobs == 1:
            for payload in payloads:
                section_started = time.perf_counter()
                result = _section_worker(payload)
                results.append(result)
                completed += 1
                durations.append(time.perf_counter() - section_started)
                _append_log(
                    log_path,
                    "section_completed",
                    elapsed_seconds=time.perf_counter() - started,
                    sample_id=result.sample_id,
                    completed=completed,
                    total=len(sections),
                    eta_seconds=float(np.mean(durations) * (len(sections) - completed)),
                    system_memory_fraction=_system_memory_fraction(),
                )
        else:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs, mp_context=context
            ) as executor:
                submitted = {
                    executor.submit(_section_worker, payload): cast(
                        SpatialSectionSpec, payload["spec"]
                    ).sample_id
                    for payload in payloads
                }
                for future in concurrent.futures.as_completed(submitted):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    elapsed = time.perf_counter() - started
                    _append_log(
                        log_path,
                        "section_completed",
                        elapsed_seconds=elapsed,
                        sample_id=result.sample_id,
                        completed=completed,
                        total=len(sections),
                        eta_seconds=(
                            elapsed * (len(sections) - completed) / completed
                        ),
                        system_memory_fraction=_system_memory_fraction(),
                    )
        results.sort(key=lambda item: item.sample_id)
        if {result.sample_id for result in results} != {
            section.sample_id for section in sections
        }:
            raise RuntimeError("spatial geometry section result roster changed")
        associations = pd.concat(
            [section_association_table(result) for result in results],
            ignore_index=True,
        )
        band_audit = pd.concat(
            [result.band_audit for result in results], ignore_index=True
        )
        decay = distance_decay_table(associations, band_audit)
        label_axis = cell_type_permutations(
            results[0].cell_types,
            replicates=label_replicates,
            seed=int(nulls["seed"]),
            dataset=dataset,
        )
        effects = pd.concat(
            [
                condition_geometry_effects(
                    results,
                    reference=contract.reference,
                    target=contract.target,
                    analysis_unit=unit,
                    label_permutations=label_axis,
                    minimum_units_per_condition=int(
                        comparison["minimum_units_per_condition"]
                    ),
                )
                for unit in (
                    contract.primary_analysis_unit,
                    contract.sensitivity_analysis_unit,
                )
            ],
            ignore_index=True,
        )
        expected = expected_geometry_sets(
            effects,
            top_fractions=cast(Sequence[float], comparison["top_fractions"]),
        )
        summary = _geometry_summary(effects, decay)
        tables = {
            "section_band_audit.tsv": band_audit,
            "sample_band_associations.parquet": associations,
            "sample_distance_decay.parquet": decay,
            "condition_geometry_effects.tsv": effects,
            "geometry_expected_sets.tsv": expected,
            "geometry_summary.tsv": summary,
        }
        output_records: dict[str, object] = {}
        for filename, table in tables.items():
            path = output / filename
            if path.suffix == ".parquet":
                table.to_parquet(path, index=False, compression="zstd")
            else:
                table.to_csv(path, sep="\t", index=False)
            output_records[filename] = {
                "filename": filename,
                "bytes": path.stat().st_size,
                "rows": len(table),
                "sha256": sha256_file(path),
            }
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
                "analysis_units": {
                    "primary": contract.primary_analysis_unit,
                    "sensitivity": contract.sensitivity_analysis_unit,
                    "primary_units": int(
                        effects.loc[
                            effects["analysis_unit"].eq(contract.primary_analysis_unit),
                            ["n_reference_units", "n_target_units"],
                        ]
                        .drop_duplicates()
                        .sum(axis=1)
                        .iloc[0]
                    ),
                },
                "geometry": dict(cast(Mapping[str, object], protocol["geometry"])),
                "release": dict(cast(Mapping[str, object], protocol["release"])),
                "environment": python_environment(
                    environment_name="crychic_project_python",
                    packages=(
                        "numpy",
                        "pandas",
                        "pyarrow",
                        "scipy",
                        "psutil",
                        "threadpoolctl",
                    ),
                    threads=1,
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
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-jobs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for name, value in _THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    arguments = _parser().parse_args(argv)
    manifest = run(
        dataset=arguments.dataset,
        profile=arguments.profile,
        input_root=arguments.input_root.resolve(),
        input_manifest_path=(
            None
            if arguments.input_manifest is None
            else arguments.input_manifest.resolve()
        ),
        truth_manifest_path=arguments.truth_manifest.resolve(),
        output_dir=arguments.output_dir.resolve(),
        config_path=arguments.config.resolve(),
        sample_jobs=arguments.sample_jobs,
        overwrite=arguments.overwrite,
        allow_dirty=arguments.allow_dirty,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
