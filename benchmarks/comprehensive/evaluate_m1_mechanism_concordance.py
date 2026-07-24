"""Evaluate M0 activity and an isolated signed M1 mechanism-concordance head."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.comprehensive.generate_m1_mechanism_fixture import (
    SCHEMA_VERSION as FIXTURE_SCHEMA,
)
from crychic.availability import AvailabilityParameters
from crychic.core import CrychicConfig
from crychic.pseudobulk import PseudobulkDataset
from crychic.scoring import (
    SIGNED_PROGRAM_CONCORDANCE_FORMULA,
    SIGNED_PROGRAM_CONCORDANCE_VERSION,
    build_absolute_activity_heads,
    signed_geometric_program_concordance,
)
from crychic.workflow import fit_baseline

SCHEMA_VERSION = "crychic-m1-mechanism-concordance-evaluation-v1"
CONFIG_SCHEMA_VERSION = "crychic-suggestions-next-m1-config-v1"
M0_METHOD = "crychic_m0_absolute_activity_effect"
M1_METHOD = "crychic_m1_signed_program_concordance"
M1_SCORE_VERSION = SIGNED_PROGRAM_CONCORDANCE_VERSION
PROGRAM_TRANSFORM = "weighted_mean_clip_log1p_cpm_over_log1p_1000000"
CONCORDANCE_FORMULA = SIGNED_PROGRAM_CONCORDANCE_FORMULA


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _validated_config(
    path: Path, *, role: str, fixture_manifest: Path
) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("M1 configuration schema is unsupported")
    expected_candidate = {
        "name": M1_METHOD,
        "score_version": M1_SCORE_VERSION,
        "absolute_expression_reference": 1_000_000.0,
        "program_transform": PROGRAM_TRANSFORM,
        "concordance_formula": CONCORDANCE_FORMULA,
        "tuned_parameters": 0,
        "m0_activity_is_retained_separately": True,
        "mechanism_support_is_contrast_diagnostic": True,
        "formal_inference_allowed": False,
    }
    candidate = config.get("candidate")
    if not isinstance(candidate, Mapping) or any(
        candidate.get(key) != value for key, value in expected_candidate.items()
    ):
        raise ValueError("M1 candidate identity or frozen formula changed")
    boundary = config.get("claim_boundary")
    if not isinstance(boundary, Mapping) or not all(
        boundary.get(field) is True
        for field in (
            "experimental_head_only",
            "m0_parent_activity_is_not_replaced",
            "active_inhibition_is_not_claimed",
            "probability_is_not_emitted",
            "p_and_q_values_are_not_emitted",
            "mechanism_family_fixture_is_not_general_sota_evidence",
        )
    ):
        raise ValueError("M1 claim boundary changed")
    if role not in {"development", "validation"}:
        raise ValueError("role must be development or validation")
    section = config.get(role)
    if not isinstance(section, Mapping):
        raise ValueError(f"M1 configuration lacks {role} section")
    expected_sha = section.get("fixture_manifest_sha256")
    if not isinstance(expected_sha, str) or expected_sha.startswith("PENDING"):
        raise ValueError(f"{role} fixture checksum is not frozen")
    if sha256_file(fixture_manifest) != expected_sha:
        raise ValueError(f"{role} fixture manifest checksum differs")
    return config


def _program_definition(fixture: Path, manifest: Mapping[str, Any]) -> pd.DataFrame:
    record = manifest.get("program_definition")
    if not isinstance(record, Mapping):
        raise ValueError("fixture does not bind a program definition")
    path = fixture / str(record["filename"])
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError("program definition checksum differs")
    table = pd.read_csv(path, sep="\t")
    required = {
        "program_id",
        "interaction_id",
        "receiver",
        "target_gene",
        "weight",
        "expected_program_direction",
    }
    missing = required.difference(table.columns)
    if missing or table.empty:
        raise ValueError(f"program definition is incomplete: {sorted(missing)}")
    weights = pd.to_numeric(table["weight"], errors="coerce")
    if weights.isna().any() or (weights <= 0).any() or not np.isfinite(weights).all():
        raise ValueError("program weights must be finite and positive")
    table["weight"] = weights.astype(float)
    for _, group in table.groupby("program_id", observed=True, sort=False):
        if not math.isclose(float(group["weight"].sum()), 1.0, abs_tol=1e-12):
            raise ValueError("each frozen program must sum to one")
        if group["target_gene"].astype(str).duplicated().any():
            raise ValueError("target genes must be unique within a program")
    directions = pd.to_numeric(table["expected_program_direction"], errors="coerce")
    if directions.isna().any() or not set(directions.astype(int)).issubset({-1, 1}):
        raise ValueError("program directions must be -1 or 1")
    table["expected_program_direction"] = directions.astype(int)
    return table


def _program_sample_table(
    aggregate: PseudobulkDataset,
    definition: pd.DataFrame,
    *,
    expression_reference: float,
) -> pd.DataFrame:
    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("M1 program evidence requires count pseudobulk")
    if not math.isfinite(expression_reference) or expression_reference <= 0:
        raise ValueError("expression_reference must be finite and positive")
    programs = definition["program_id"].astype(str).unique()
    if len(programs) != 1:
        raise ValueError("M1 v1 fixture requires exactly one frozen program")
    program = definition.loc[definition["program_id"].astype(str).eq(programs[0])]
    receiver_values = program["receiver"].astype(str).unique()
    interaction_values = program["interaction_id"].astype(str).unique()
    direction_values = program["expected_program_direction"].astype(int).unique()
    if not (
        len(receiver_values) == len(interaction_values) == len(direction_values) == 1
    ):
        raise ValueError("one program must map to one receiver, interaction, and sign")
    feature_index = {gene: index for index, gene in enumerate(aggregate.feature_ids)}
    target_genes = tuple(program["target_gene"].astype(str))
    missing = set(target_genes).difference(feature_index)
    if missing:
        raise ValueError(
            f"program targets are absent from pseudobulk: {sorted(missing)}"
        )
    columns = np.asarray([feature_index[gene] for gene in target_genes], dtype=int)
    weights = program["weight"].to_numpy(dtype=float)
    metadata = (
        aggregate.unit_metadata.set_index("unit_id", drop=False)
        .loc[list(aggregate.matrix_unit_ids)]
        .reset_index(drop=True)
    )
    if "condition" not in metadata.columns:
        raise ValueError("aggregate metadata lacks condition")
    receiver = receiver_values[0]
    selected_rows = metadata["cell_type"].astype(str).eq(receiver) & metadata[
        "state_eligible"
    ].astype(bool)
    row_indices = np.flatnonzero(selected_rows.to_numpy())
    if not len(row_indices):
        raise ValueError("frozen program receiver has no eligible pseudobulk rows")
    counts = aggregate.counts[row_indices]
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    target_counts = counts[:, columns].toarray().astype(float)
    cpm = np.divide(
        target_counts * 1_000_000.0,
        library[:, None],
        out=np.zeros_like(target_counts),
        where=library[:, None] > 0,
    )
    evidence = (
        np.clip(np.log1p(cpm) / math.log1p(expression_reference), 0.0, 1.0) @ weights
    )
    selected = metadata.iloc[row_indices]
    result = selected.loc[:, ["sample_id", "subject_id", "condition"]].copy()
    result["receiver"] = receiver
    result["interaction_id"] = interaction_values[0]
    result["program_id"] = programs[0]
    result["receiver_program_raw"] = evidence
    result["expected_program_direction"] = int(direction_values[0])
    if result.duplicated(["sample_id", "receiver", "interaction_id"]).any():
        raise ValueError("program sample keys must be unique")
    return result.reset_index(drop=True)


def _fit_dataset(
    dataset_id: str,
    dataset_spec: Mapping[str, Any],
    resource_path: str,
    resource_manifest_path: str,
    definition_records: list[dict[str, Any]],
    expression_reference: float,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    input_path = Path(str(dataset_spec["input"])).resolve()
    if sha256_file(input_path) != str(dataset_spec["input_sha256"]):
        raise ValueError(f"M1 input checksum mismatch: {dataset_id}")
    adata = ad.read_h5ad(input_path)
    try:
        bundle = harmonized_resource_bundle(resource_path, resource_manifest_path)
        workflow = cast(Mapping[str, Any], dataset_spec["workflow"])
        with threadpool_limits(limits=1):
            artifacts = fit_baseline(
                adata,
                CrychicConfig.from_dict(
                    cast(Mapping[str, Any], dataset_spec["config"])
                ),
                bundle,
                target_prior=None,
                availability_parameters=AvailabilityParameters(
                    absolute_expression_reference=expression_reference
                ),
                **dict(workflow),
            )
        heads = build_absolute_activity_heads(
            artifacts.availability, artifacts.sender_assignment
        )
        definition = pd.DataFrame.from_records(definition_records)
        program = _program_sample_table(
            artifacts.aggregate,
            definition,
            expression_reference=expression_reference,
        )
        identity = definition.iloc[0]
        selected = heads.loc[
            heads["sender"].astype(str).eq("Sender")
            & heads["receiver"].astype(str).eq(str(identity["receiver"]))
            & heads["interaction_id"].astype(str).eq(str(identity["interaction_id"]))
        ].copy()
        if selected.empty or selected.duplicated("sample_id").any():
            raise ValueError(f"known M1 event is not one row per sample: {dataset_id}")
        sample_context = (
            adata.obs.loc[:, ["sample_id", "subject_id", "condition"]]
            .astype(str)
            .drop_duplicates(ignore_index=True)
        )
        if sample_context.duplicated("sample_id").any():
            raise ValueError("sample_id must map to one subject and condition")
        selected = selected.merge(
            sample_context,
            on=["sample_id", "subject_id"],
            how="left",
            validate="one_to_one",
        )
        output = selected.loc[
            :,
            [
                "sample_id",
                "subject_id",
                "condition",
                "sender",
                "receiver",
                "interaction_id",
                "sender_detection",
                "parent_activity_raw",
            ],
        ].merge(
            program,
            on=["sample_id", "subject_id", "condition", "receiver", "interaction_id"],
            how="left",
            validate="one_to_one",
        )
        if output[["sender_detection", "receiver_program_raw"]].isna().any().any():
            raise ValueError(f"M1 evidence is incomplete: {dataset_id}")
        timing = {
            "dataset_id": dataset_id,
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "subjects": int(adata.obs["subject_id"].astype(str).nunique()),
            "elapsed_seconds": time.perf_counter() - started,
            **{
                f"stage_{name}_seconds": float(value)
                for name, value in artifacts.stage_seconds.items()
            },
        }
        return dataset_id, output, timing
    finally:
        if adata.isbacked:
            adata.file.close()


def _paired_effect(table: pd.DataFrame, value: str) -> tuple[float, float, int]:
    collapsed = table.groupby(["subject_id", "condition"], observed=True, sort=False)[
        value
    ].mean()
    wide = collapsed.unstack("condition")
    if not {"ctrl", "stim"}.issubset(wide.columns):
        raise ValueError("paired effect requires ctrl and stim")
    paired = wide.loc[:, ["ctrl", "stim"]].dropna()
    if len(paired) < 4:
        raise ValueError("paired effect requires at least four complete subjects")
    difference = paired["stim"] - paired["ctrl"]
    return float(difference.mean()), float((difference > 0).mean()), len(difference)


def _effect_table(
    heads: Mapping[str, pd.DataFrame], truth: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in truth.itertuples(index=False):
        table = heads[str(record.dataset_id)]
        lr_effect, lr_positive, pairs = _paired_effect(table, "sender_detection")
        program_effect, program_positive, program_pairs = _paired_effect(
            table, "receiver_program_raw"
        )
        if program_pairs != pairs:
            raise ValueError("LR and program paired subject coverage differs")
        direction = int(record.expected_program_direction)
        m1_effect = signed_geometric_program_concordance(
            lr_effect,
            program_effect,
            expected_program_direction=direction,
        )
        rows.append(
            {
                "dataset_id": str(record.dataset_id),
                "root_seed": int(record.root_seed),
                "scenario_seed": int(record.scenario_seed),
                "scenario": str(record.scenario),
                "expected_lr_change": bool(record.expected_lr_change),
                "expected_target_program_change": bool(
                    record.expected_target_program_change
                ),
                "expected_integrated_edge": bool(record.expected_integrated_edge),
                "expected_program_direction": direction,
                "paired_subjects": pairs,
                "m0_lr_effect": lr_effect,
                "m0_positive_subject_fraction": lr_positive,
                "receiver_program_effect": program_effect,
                "program_positive_subject_fraction": program_positive,
                "m1_concordance_effect": m1_effect,
                "m0_ranking_score": abs(lr_effect),
                "m1_ranking_score": abs(m1_effect),
                "m1_same_direction": bool(m1_effect != 0.0),
                "formal_inference_allowed": False,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["root_seed", "scenario"], kind="stable", ignore_index=True
    )


def _seed_metrics(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, family in effects.groupby("root_seed", observed=True, sort=True):
        labels = family["expected_integrated_edge"].astype(int).to_numpy()
        if labels.sum() != 1 or len(labels) != 7:
            raise ValueError("each seed must contain one active and six null families")
        for method, score_column in (
            (M0_METHOD, "m0_ranking_score"),
            (M1_METHOD, "m1_ranking_score"),
        ):
            scores = family[score_column].to_numpy(dtype=float)
            active_score = float(scores[labels == 1][0])
            negative_scores = scores[labels == 0]
            rank = float(
                pd.Series(scores)
                .rank(method="average", ascending=False)
                .loc[np.flatnonzero(labels == 1)[0]]
            )
            rows.append(
                {
                    "root_seed": int(seed),
                    "method": method,
                    "average_precision": float(average_precision_score(labels, scores)),
                    "auroc": float(roc_auc_score(labels, scores)),
                    "active_rank": rank,
                    "active_score": active_score,
                    "maximum_negative_score": float(negative_scores.max()),
                    "active_minus_maximum_negative": float(
                        active_score - negative_scores.max()
                    ),
                    "active_retained": bool(active_score > 0.0),
                    "event_coverage": 1.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def _method_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_metrics.groupby("method", observed=True, sort=True)
        .agg(
            seeds=("root_seed", "nunique"),
            average_precision=("average_precision", "mean"),
            auroc=("auroc", "mean"),
            median_active_rank=("active_rank", "median"),
            mean_active_score=("active_score", "mean"),
            mean_maximum_negative_score=("maximum_negative_score", "mean"),
            mean_active_margin=("active_minus_maximum_negative", "mean"),
            active_retention_fraction=("active_retained", "mean"),
            minimum_event_coverage=("event_coverage", "min"),
        )
        .reset_index()
        .sort_values(["average_precision", "auroc"], ascending=False)
        .reset_index(drop=True)
    )


def _paired_bootstrap(
    seed_metrics: pd.DataFrame,
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    pivot = seed_metrics.pivot(index="root_seed", columns="method", values=metric)
    pair = pivot.loc[:, [M1_METHOD, M0_METHOD]].dropna()
    delta = pair[M1_METHOD].to_numpy(dtype=float) - pair[M0_METHOD].to_numpy(
        dtype=float
    )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(delta, size=(replicates, len(delta)), replace=True).mean(
        axis=1
    )
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {
        "metric": metric,
        "candidate": M1_METHOD,
        "reference": M0_METHOD,
        "paired_seeds": len(delta),
        "candidate_mean": float(pair[M1_METHOD].mean()),
        "reference_mean": float(pair[M0_METHOD].mean()),
        "candidate_minus_reference": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def _scenario_summary(effects: pd.DataFrame) -> pd.DataFrame:
    return (
        effects.groupby("scenario", observed=True, sort=True)
        .agg(
            seeds=("root_seed", "nunique"),
            m0_mean_abs_effect=("m0_ranking_score", "mean"),
            program_mean_abs_effect=(
                "receiver_program_effect",
                lambda x: np.mean(np.abs(x)),
            ),
            m1_mean_abs_effect=("m1_ranking_score", "mean"),
            m1_nonzero_fraction=("m1_same_direction", "mean"),
            mean_paired_subjects=("paired_subjects", "mean"),
        )
        .reset_index()
    )


def _gate(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    scenario: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    gates = cast(Mapping[str, Any], config["gates"])
    section = cast(Mapping[str, Any], config[role])
    candidate = summary.set_index("method").loc[M1_METHOD]
    comparisons = paired.set_index("metric")
    families = scenario.set_index("scenario")
    active = float(families.loc["active", "m1_mean_abs_effect"])
    maximum_partial = float(
        families.loc[
            ["ligand_only", "target_only", "receptor_knockout"],
            "m1_mean_abs_effect",
        ].max()
    )
    checks = {
        "minimum_paired_seeds": int(comparisons["paired_seeds"].min())
        >= int(section.get("minimum_paired_seeds", len(section["seeds"]))),
        "average_precision_superiority": float(
            comparisons.loc["average_precision", "ci_low"]
        )
        > float(gates["average_precision_delta_ci_lower_minimum_exclusive"]),
        "auroc_superiority": float(comparisons.loc["auroc", "ci_low"])
        > float(gates["auroc_delta_ci_lower_minimum_exclusive"]),
        "active_retention": float(candidate["active_retention_fraction"])
        >= float(gates["minimum_active_retention_fraction"]),
        "active_rank": float(candidate["median_active_rank"])
        <= float(gates["maximum_median_active_rank"]),
        "partial_mechanism_suppression": maximum_partial
        <= active * float(gates["maximum_partial_to_active_ratio"]),
        "coverage": float(candidate["minimum_event_coverage"])
        >= float(gates["minimum_event_coverage"]),
    }
    return {
        "role": role,
        "checks": {key: bool(value) for key, value in checks.items()},
        "active_m1_mean_abs_effect": active,
        "maximum_partial_m1_mean_abs_effect": maximum_partial,
        "maximum_partial_to_active_ratio_observed": (
            maximum_partial / active if active > 0 else None
        ),
        "status": (
            "DEVELOPMENT_PASS"
            if role == "development" and all(checks.values())
            else "DEVELOPMENT_FAIL"
            if role == "development"
            else "ACCEPT"
            if all(checks.values())
            else "REJECT"
        ),
    }


def _report(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    scenario: pd.DataFrame,
    gate: Mapping[str, Any],
) -> str:
    lines = [
        "# CRYCHIC M1 signed mechanism-concordance benchmark",
        "",
        f"Gate status: **{gate['status']}**",
        "",
        (
            "M1 is a contrast-level mechanism diagnostic layered beside M0. "
            "It does not replace parent activity and emits no probability or "
            "p/q values."
        ),
        "",
        (
            "| Method | AP | AUROC | Median active rank | Active retention | "
            "Active margin |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.average_precision:.4f} | {row.auroc:.4f} | "
            f"{row.median_active_rank:.2f} | {row.active_retention_fraction:.3f} | "
            f"{row.mean_active_margin:.5f} |"
        )
    lines.extend(("", "## Mechanism families", ""))
    lines.append(
        "| Scenario | M0 abs effect | Program abs effect | M1 abs effect | M1 nonzero |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for row in scenario.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.m0_mean_abs_effect:.5f} | "
            f"{row.program_mean_abs_effect:.5f} | {row.m1_mean_abs_effect:.5f} | "
            f"{row.m1_nonzero_fraction:.3f} |"
        )
    lines.extend(("", "## Paired seed comparisons", ""))
    for row in paired.itertuples(index=False):
        lines.append(
            f"- {row.metric}: M1-M0 {row.candidate_minus_reference:.4f}, "
            f"95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}], "
            f"wins/ties/losses {row.wins}/{row.ties}/{row.losses}."
        )
    lines.extend(("", "## Gate checks", ""))
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in cast(Mapping[str, Any], gate["checks"]).items()
    )
    return "\n".join(lines) + "\n"


def evaluate(
    *,
    fixture: Path,
    config_path: Path,
    role: str,
    output_dir: Path,
    n_jobs: int,
) -> dict[str, Any]:
    """Fit M0, derive frozen target-program evidence, and score M1 feedback."""

    started = time.perf_counter()
    fixture = fixture.resolve()
    manifest_path = fixture / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != FIXTURE_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise ValueError("M1 fixture is incomplete or schema-mismatched")
    config_path = config_path.resolve()
    config = _validated_config(config_path, role=role, fixture_manifest=manifest_path)
    section = cast(Mapping[str, Any], config[role])
    if tuple(map(int, manifest["seeds"])) != tuple(map(int, section["seeds"])):
        raise ValueError(f"{role} fixture seeds differ from preregistration")
    truth_path = fixture / str(
        cast(Mapping[str, Any], manifest["outputs"])["truth"]["filename"]
    )
    if (
        sha256_file(truth_path)
        != cast(Mapping[str, Any], manifest["outputs"])["truth"]["sha256"]
    ):
        raise ValueError("mechanism truth checksum differs")
    truth = pd.read_csv(truth_path, sep="\t")
    definition = _program_definition(fixture, manifest)
    crychic_config = _read_json(fixture / "crychic_config.json")
    datasets = crychic_config.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("fixture CRYCHIC dataset registry is empty")
    resource_path = fixture / str(
        cast(Mapping[str, Any], manifest["resource"])["filename"]
    )
    resource_manifest = fixture / str(
        cast(Mapping[str, Any], manifest["resource"])["manifest"]
    )
    candidate = cast(Mapping[str, Any], config["candidate"])
    workers = max(1, min(int(n_jobs), len(datasets)))
    heads: dict[str, pd.DataFrame] = {}
    timings: list[dict[str, Any]] = []
    records = definition.to_dict(orient="records")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(
                _fit_dataset,
                str(dataset_id),
                cast(Mapping[str, Any], spec),
                str(resource_path),
                str(resource_manifest),
                records,
                float(candidate["absolute_expression_reference"]),
            ): str(dataset_id)
            for dataset_id, spec in datasets.items()
        }
        for future in as_completed(futures):
            dataset_id, table, timing = future.result()
            heads[dataset_id] = table
            timings.append(timing)
    if set(heads) != set(map(str, datasets)):
        raise RuntimeError("M1 parallel fit did not return every dataset")
    effects = _effect_table(heads, truth)
    seed_metrics = _seed_metrics(effects)
    summary = _method_summary(seed_metrics)
    scenario = _scenario_summary(effects)
    bootstrap_replicates = int(section.get("bootstrap_replicates", 20_000))
    bootstrap_seed = int(section.get("bootstrap_seed", 20260724))
    paired = pd.DataFrame.from_records(
        [
            _paired_bootstrap(
                seed_metrics,
                metric="average_precision",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
            _paired_bootstrap(
                seed_metrics,
                metric="auroc",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 1,
            ),
        ]
    )
    gate = _gate(summary, paired, scenario, config=config, role=role)

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    published = False
    try:
        tables = {
            "mechanism_effects.tsv": effects,
            "seed_metrics.tsv": seed_metrics,
            "method_summary.tsv": summary,
            "scenario_summary.tsv": scenario,
            "paired_comparisons.tsv": paired,
            "timing.tsv": pd.DataFrame.from_records(timings).sort_values(
                "dataset_id", kind="stable", ignore_index=True
            ),
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False, lineterminator="\n")
        (staged / "REPORT.md").write_text(
            _report(summary, paired, scenario, gate), encoding="utf-8"
        )
        output_records: dict[str, Any] = {
            filename: {
                "rows": len(table),
                "bytes": (staged / filename).stat().st_size,
                "sha256": sha256_file(staged / filename),
            }
            for filename, table in tables.items()
        }
        output_records["REPORT.md"] = {
            "bytes": (staged / "REPORT.md").stat().st_size,
            "sha256": sha256_file(staged / "REPORT.md"),
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "role": role,
            "gate": gate,
            "candidate": dict(candidate),
            "claim_boundary": dict(cast(Mapping[str, Any], config["claim_boundary"])),
            "fixture": {
                "directory": str(fixture),
                "manifest_sha256": sha256_file(manifest_path),
            },
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "parallel_workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "outputs": output_records,
        }
        _write_json(staged / "manifest.json", result)
        os.replace(staged, output_dir)
        published = True
        return result
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("development", "validation"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    result = evaluate(
        fixture=args.fixture,
        config_path=args.config,
        role=args.role,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(json_safe(result), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
