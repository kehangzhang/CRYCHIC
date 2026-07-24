"""Evaluate the preregistered M0 absolute sender-detection head."""

from __future__ import annotations

import argparse
import json
import math
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
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.adapters.crychic.score_layers import (
    sender_response_detection_evidence_score,
)
from benchmarks.comprehensive.evaluate_component_crossover import (
    _effect_arm,
    _read_component_long,
    _source_truth,
)
from benchmarks.comprehensive.evaluate_three_group import _multigroup_metrics
from crychic.availability import AvailabilityParameters
from crychic.core import CrychicConfig
from crychic.scoring import (
    ABSOLUTE_ACTIVITY_SCORE_VERSION,
    build_absolute_activity_heads,
)
from crychic.workflow import fit_baseline

SCHEMA_VERSION = "crychic-m0-absolute-activity-evaluation-v1"
CONFIG_SCHEMA_VERSION = "crychic-suggestions-next-m0-config-v1"
M0_LAYER = "crychic_m0_absolute_sender_detection"
RC12_LAYER = "crychic_rc12_sender_response_detection"
CANONICAL_LAYER = "crychic_canonical_mechanistic"
STRICT_LAYER = "crychic_legacy_strict_geometric"
ENGINE = "native_raw_mean"
M0_ARM = f"{M0_LAYER}__{ENGINE}"
RC12_ARM = f"{RC12_LAYER}__{ENGINE}"
CANONICAL_ARM = f"{CANONICAL_LAYER}__{ENGINE}"
STRICT_ARM = f"{STRICT_LAYER}__{ENGINE}"

_HEAD_KEYS = (
    "sample_id",
    "subject_id",
    "sender",
    "receiver",
    "interaction_id",
)


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
    path: Path,
    *,
    role: str,
    fixture_manifest: Path,
    source_manifest: Path,
) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("M0 configuration schema is unsupported")
    candidate = config.get("candidate")
    expected_candidate = {
        "name": M0_LAYER,
        "score_version": ABSOLUTE_ACTIVITY_SCORE_VERSION,
        "absolute_expression_reference": 1_000_000.0,
        "ligand_weight": 0.5,
        "receptor_weight": 0.5,
        "sender_detection_conserved": False,
        "conditional_attribution_conserved": True,
        "null_sender_enabled": True,
        "formal_inference_allowed": False,
    }
    if not isinstance(candidate, Mapping) or any(
        candidate.get(key) != value for key, value in expected_candidate.items()
    ):
        raise ValueError("M0 candidate identity or frozen parameters changed")
    boundary = config.get("claim_boundary")
    if not isinstance(boundary, Mapping) or not all(
        boundary.get(field) is True
        for field in (
            "experimental_head_only",
            "communication_strength_is_not_replaced",
            "active_probability_is_not_emitted",
            "p_and_q_values_are_not_emitted",
            "expression_driven_fixture_is_not_general_sota_evidence",
        )
    ):
        raise ValueError("M0 claim boundary changed")
    if role not in {"development", "validation"}:
        raise ValueError("role must be development or validation")
    section = config.get(role)
    if not isinstance(section, Mapping):
        raise ValueError(f"M0 configuration lacks {role} section")
    expected_fixture = section.get("fixture_manifest_sha256")
    expected_source = section.get("source_evaluation_manifest_sha256")
    if not isinstance(expected_fixture, str) or expected_fixture.startswith("PENDING"):
        raise ValueError(f"{role} fixture checksum is not frozen")
    if not isinstance(expected_source, str) or expected_source.startswith("PENDING"):
        raise ValueError(f"{role} source-evaluation checksum is not frozen")
    if sha256_file(fixture_manifest) != expected_fixture:
        raise ValueError(f"{role} fixture manifest checksum differs")
    if sha256_file(source_manifest) != expected_source:
        raise ValueError(f"{role} source-evaluation manifest checksum differs")
    return config


def _fit_m0_dataset(
    dataset_id: str,
    dataset_spec: Mapping[str, Any],
    resource_path: str,
    resource_manifest_path: str,
    expression_reference: float,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    input_path = Path(str(dataset_spec["input"])).resolve()
    if sha256_file(input_path) != str(dataset_spec["input_sha256"]):
        raise ValueError(f"M0 input checksum mismatch: {dataset_id}")
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
        if heads.empty or set(heads["score_version"].astype(str)) != {
            ABSOLUTE_ACTIVITY_SCORE_VERSION
        }:
            raise RuntimeError(f"M0 head is empty or version-mismatched: {dataset_id}")
        if heads["formal_inference_allowed"].any():
            raise RuntimeError("M0 head cannot enable formal inference")
        output = heads.loc[
            :,
            [
                *_HEAD_KEYS,
                "sender_detection",
                "parent_activity_raw",
                "mechanism_support",
                "sender_attribution",
                "sender_attribution_with_null",
                "null_sender_attribution",
                "candidate_sender_count",
            ],
        ].copy()
        timing = {
            "dataset_id": dataset_id,
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "subjects": int(adata.obs["subject_id"].astype(str).nunique()),
            "head_rows": len(output),
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


def _fit_all_m0_heads(
    fixture: Path,
    *,
    config: Mapping[str, Any],
    role: str,
    n_jobs: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    fixture_manifest = _read_json(fixture / "manifest.json")
    crychic_config = _read_json(fixture / "crychic_config.json")
    section = cast(Mapping[str, Any], config[role])
    expected_seeds = tuple(map(int, cast(Sequence[object], section["seeds"])))
    observed_seeds = tuple(map(int, cast(Sequence[object], fixture_manifest["seeds"])))
    if observed_seeds != expected_seeds:
        raise ValueError(f"{role} fixture seeds differ from preregistration")
    datasets = crychic_config.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("fixture CRYCHIC dataset registry is empty")
    resource_path = fixture / "resource" / "harmonized_lr.tsv"
    resource_manifest = fixture / "resource" / "manifest.json"
    resource_record = cast(Mapping[str, Any], fixture_manifest["resource"])
    if sha256_file(resource_path) != resource_record["sha256"]:
        raise ValueError("fixture H-common resource checksum differs")
    workers = max(1, min(int(n_jobs), len(datasets)))
    heads: dict[str, pd.DataFrame] = {}
    timings: list[dict[str, Any]] = []
    candidate = cast(Mapping[str, Any], config["candidate"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fit_m0_dataset,
                str(dataset_id),
                cast(Mapping[str, Any], spec),
                str(resource_path),
                str(resource_manifest),
                float(candidate["absolute_expression_reference"]),
            ): str(dataset_id)
            for dataset_id, spec in datasets.items()
        }
        for future in as_completed(futures):
            dataset_id, table, timing = future.result()
            heads[dataset_id] = table
            timings.append(timing)
    if set(heads) != set(map(str, datasets)):
        raise RuntimeError("M0 parallel fit did not return every registered dataset")
    return heads, pd.DataFrame.from_records(timings).sort_values(
        "dataset_id", kind="stable", ignore_index=True
    )


def _layer_geometry(
    score: pd.Series,
    *,
    dataset_id: str,
    contrast: str,
    method: str,
) -> dict[str, Any]:
    numeric = pd.to_numeric(score, errors="coerce").astype(float)
    finite = numeric.dropna()
    if finite.empty:
        return {
            "dataset_id": dataset_id,
            "contrast": contrast,
            "method": method,
            "rows": len(numeric),
            "finite_rows": 0,
            "zero_fraction": math.nan,
            "tie_fraction": math.nan,
            "unique_scores": 0,
            "interquartile_range": math.nan,
        }
    _, counts = np.unique(finite.to_numpy(dtype=float), return_counts=True)
    tied = int(counts[counts > 1].sum())
    return {
        "dataset_id": dataset_id,
        "contrast": contrast,
        "method": method,
        "rows": len(numeric),
        "finite_rows": len(finite),
        "zero_fraction": float(finite.eq(0.0).mean()),
        "tie_fraction": tied / len(finite),
        "unique_scores": len(counts),
        "interquartile_range": float(finite.quantile(0.75) - finite.quantile(0.25)),
    }


def _method_summary(metrics: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    active = (
        metrics.loc[metrics["scenario"].astype(str).eq("active")]
        .groupby("method", observed=True, sort=True)
        .agg(
            active_seeds=("seed", "nunique"),
            omnibus_auprc=("omnibus_auprc", "mean"),
            omnibus_auroc=("omnibus_auroc", "mean"),
            localization_macro_auprc=("localization_macro_auprc", "mean"),
            direction_accuracy=("direction_accuracy_all_active", "mean"),
            effect_spearman=("effect_spearman", "mean"),
            minimum_event_coverage=("event_coverage", "min"),
        )
        .reset_index()
    )
    null = (
        metrics.loc[metrics["scenario"].astype(str).eq("global_null")]
        .groupby("method", observed=True, sort=True)
        .agg(
            null_seeds=("seed", "nunique"),
            global_null_effect_sd=("effect_standard_deviation", "mean"),
            global_null_effect_range=("effect_dynamic_range", "mean"),
        )
        .reset_index()
    )
    geometry_summary = (
        geometry.groupby("method", observed=True, sort=True)
        .agg(
            mean_zero_fraction=("zero_fraction", "mean"),
            mean_tie_fraction=("tie_fraction", "mean"),
            mean_unique_scores=("unique_scores", "mean"),
            mean_interquartile_range=("interquartile_range", "mean"),
        )
        .reset_index()
    )
    return (
        active.merge(null, on="method", validate="one_to_one")
        .merge(geometry_summary, on="method", validate="one_to_one")
        .sort_values(["omnibus_auprc", "omnibus_auroc"], ascending=False)
        .reset_index(drop=True)
    )


def _paired_comparison(
    metrics: pd.DataFrame,
    *,
    metric: str,
    candidate: str,
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    active = metrics.loc[metrics["scenario"].astype(str).eq("active")]
    pivot = active.pivot(index="seed", columns="method", values=metric)
    missing = {candidate, reference}.difference(pivot.columns)
    if missing:
        raise ValueError(f"paired comparison arms are absent: {sorted(missing)}")
    pair = pivot.loc[:, [candidate, reference]].dropna()
    delta = pair[candidate].to_numpy(dtype=float) - pair[reference].to_numpy(
        dtype=float
    )
    if len(delta):
        rng = np.random.default_rng(seed)
        sampled = rng.choice(delta, size=(replicates, len(delta)), replace=True).mean(
            axis=1
        )
        low, high = np.quantile(sampled, (0.025, 0.975))
    else:
        low = high = math.nan
    return {
        "scenario": "active",
        "metric": metric,
        "candidate": candidate,
        "reference": reference,
        "paired_seeds": len(delta),
        "candidate_mean": float(pair[candidate].mean()),
        "reference_mean": float(pair[reference].mean()),
        "candidate_minus_reference": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def _gate(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    gates = cast(Mapping[str, Any], config["gates"])
    section = cast(Mapping[str, Any], config[role])
    by_method = summary.set_index("method")
    candidate = by_method.loc[M0_ARM]
    canonical = by_method.loc[CANONICAL_ARM]
    strict = by_method.loc[STRICT_ARM]
    by_metric = paired.set_index("metric")
    checks = {
        "minimum_paired_seeds": int(by_metric["paired_seeds"].min())
        >= int(section.get("minimum_paired_seeds", len(section["seeds"]))),
        "auprc_noninferiority": float(by_metric.loc["omnibus_auprc", "ci_low"])
        >= float(gates["auprc_delta_ci_lower_minimum"]),
        "auroc_superiority": float(by_metric.loc["omnibus_auroc", "ci_low"])
        > float(gates["auroc_delta_ci_lower_minimum_exclusive"]),
        "direction_retained": float(candidate["direction_accuracy"])
        >= float(canonical["direction_accuracy"])
        - float(gates["direction_accuracy_maximum_degradation"]),
        "coverage": float(candidate["minimum_event_coverage"])
        >= float(gates["minimum_event_coverage"]),
        "zero_fraction_reduced": float(candidate["mean_zero_fraction"])
        < float(strict["mean_zero_fraction"]),
        "tie_fraction_reduced": float(candidate["mean_tie_fraction"])
        < float(strict["mean_tie_fraction"]),
    }
    return {
        "role": role,
        "checks": {key: bool(value) for key, value in checks.items()},
        "sender_invariant_tests_required": bool(
            gates["require_sender_invariant_tests"]
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
    summary: pd.DataFrame, paired: pd.DataFrame, gate: Mapping[str, Any]
) -> str:
    lines = [
        "# CRYCHIC M0 absolute activity benchmark",
        "",
        f"Gate status: **{gate['status']}**",
        "",
        (
            "M0 is an experimental unsigned detection head. It does not replace "
            "communication strength and emits no calibrated probability or p/q values."
        ),
        "",
        (
            "| Method | AUPRC | AUROC | Localization AP | Direction | "
            "Null SD | Zero | Tie | Coverage |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.omnibus_auprc:.4f} | "
            f"{row.omnibus_auroc:.4f} | {row.localization_macro_auprc:.4f} | "
            f"{row.direction_accuracy:.4f} | {row.global_null_effect_sd:.5f} | "
            f"{row.mean_zero_fraction:.4f} | {row.mean_tie_fraction:.4f} | "
            f"{row.minimum_event_coverage:.3f} |"
        )
    lines.extend(("", "## Paired seed comparisons", ""))
    for row in paired.itertuples(index=False):
        lines.append(
            f"- {row.metric}: M0-reference delta {row.candidate_minus_reference:.4f}, "
            f"95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}], "
            f"wins/ties/losses {row.wins}/{row.ties}/{row.losses}."
        )
    lines.extend(("", "## Gate checks", ""))
    checks = cast(Mapping[str, Any], gate["checks"])
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()
    )
    return "\n".join(lines) + "\n"


def evaluate(
    *,
    fixture: Path,
    source_evaluation: Path,
    legacy_runs_dir: Path,
    config_path: Path,
    role: str,
    output_dir: Path,
    n_jobs: int,
) -> dict[str, Any]:
    """Run the checksum-bound M0 candidate and compare frozen score heads."""

    started = time.perf_counter()
    fixture = fixture.resolve()
    source_evaluation = source_evaluation.resolve()
    legacy_runs_dir = legacy_runs_dir.resolve()
    config_path = config_path.resolve()
    fixture_manifest_path = fixture / "manifest.json"
    source_manifest_path = source_evaluation / "manifest.json"
    config = _validated_config(
        config_path,
        role=role,
        fixture_manifest=fixture_manifest_path,
        source_manifest=source_manifest_path,
    )
    truth, source_manifest = _source_truth(source_evaluation)
    m0_heads, timing = _fit_all_m0_heads(
        fixture, config=config, role=role, n_jobs=n_jobs
    )
    component_cache: dict[str, pd.DataFrame] = {}
    effects: list[pd.DataFrame] = []
    geometry: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for (dataset_id, contrast), selected_truth in truth.groupby(
        ["dataset_id", "contrast"], observed=True, sort=True
    ):
        run_names = tuple(selected_truth["run_directory"].astype(str).unique())
        if len(run_names) != 1:
            raise ValueError("one dataset must bind exactly one legacy CRYCHIC run")
        run_name = run_names[0]
        if run_name not in component_cache:
            component, run_provenance = _read_component_long(legacy_runs_dir / run_name)
            component_cache[run_name] = component
            provenance.append(run_provenance)
        target = str(selected_truth["target"].iloc[0])
        reference = str(selected_truth["reference"].iloc[0])
        view = (
            component_cache[run_name]
            .loc[
                component_cache[run_name]["contrast_view"]
                .astype(str)
                .eq(f"global:'{target}'")
            ]
            .copy()
        )
        heads = m0_heads[str(dataset_id)]
        view = view.merge(
            heads.loc[:, [*_HEAD_KEYS, "sender_detection"]],
            on=list(_HEAD_KEYS),
            how="left",
            validate="many_to_one",
            sort=False,
        )
        if view["sender_detection"].isna().any():
            raise ValueError(f"M0 head does not cover legacy event rows: {dataset_id}")
        mechanism = view["availability"].astype(float) * view["prior_quality"].astype(
            float
        )
        layers = {
            STRICT_LAYER: view["comm_strength"].astype(float),
            CANONICAL_LAYER: mechanism * view["sender_component"].astype(float),
            RC12_LAYER: sender_response_detection_evidence_score(
                view["sender_component"], view["downstream"]
            ),
            M0_LAYER: view["sender_detection"].astype(float),
        }
        truth_table = selected_truth.drop(columns="run_directory")
        for layer, score in layers.items():
            long_table = view.copy()
            long_table["score"] = score
            long_table["score_name"] = layer
            long_table["status"] = "ok"
            long_table["reason_code"] = "m0_absolute_activity_observed"
            effects.append(
                _effect_arm(
                    long_table,
                    truth_table,
                    layer=layer,
                    engine=ENGINE,
                    target=target,
                    reference=reference,
                    contrast=str(contrast),
                    dataset_id=str(dataset_id),
                )
            )
            geometry.append(
                _layer_geometry(
                    score,
                    dataset_id=str(dataset_id),
                    contrast=str(contrast),
                    method=f"{layer}__{ENGINE}",
                )
            )
    effect_table = pd.concat(effects, ignore_index=True)
    metrics, confusion = _multigroup_metrics(effect_table)
    geometry_table = pd.DataFrame.from_records(geometry)
    summary = _method_summary(metrics, geometry_table)
    section = cast(Mapping[str, Any], config[role])
    replicates = int(section.get("bootstrap_replicates", 20_000))
    bootstrap_seed = int(section.get("bootstrap_seed", 20260724))
    paired = pd.DataFrame.from_records(
        [
            _paired_comparison(
                metrics,
                metric="omnibus_auprc",
                candidate=M0_ARM,
                reference=RC12_ARM,
                replicates=replicates,
                seed=bootstrap_seed,
            ),
            _paired_comparison(
                metrics,
                metric="omnibus_auroc",
                candidate=M0_ARM,
                reference=RC12_ARM,
                replicates=replicates,
                seed=bootstrap_seed + 1,
            ),
            _paired_comparison(
                metrics,
                metric="direction_accuracy_all_active",
                candidate=M0_ARM,
                reference=CANONICAL_ARM,
                replicates=replicates,
                seed=bootstrap_seed + 2,
            ),
        ]
    )
    gate = _gate(summary, paired, config=config, role=role)

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
            "multigroup_metrics.tsv": metrics,
            "contrast_confusion.tsv": confusion,
            "method_summary.tsv": summary,
            "paired_comparisons.tsv": paired,
            "score_geometry.tsv": geometry_table,
            "timing.tsv": timing,
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False, lineterminator="\n")
        report = _report(summary, paired, gate)
        (staged / "REPORT.md").write_text(report, encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "role": role,
            "gate": gate,
            "candidate": dict(cast(Mapping[str, Any], config["candidate"])),
            "claim_boundary": dict(cast(Mapping[str, Any], config["claim_boundary"])),
            "fixture": {
                "directory": str(fixture),
                "manifest_sha256": sha256_file(fixture_manifest_path),
            },
            "source_evaluation": {
                "directory": str(source_evaluation),
                "manifest_sha256": sha256_file(source_manifest_path),
                "schema_version": source_manifest["schema_version"],
            },
            "legacy_runs_directory": str(legacy_runs_dir),
            "legacy_run_provenance": sorted(
                provenance, key=lambda item: item["run_directory"]
            ),
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "parallel_workers": int(n_jobs),
            "elapsed_seconds": time.perf_counter() - started,
            "outputs": {
                filename: {
                    "rows": len(table),
                    "bytes": (staged / filename).stat().st_size,
                    "sha256": sha256_file(staged / filename),
                }
                for filename, table in tables.items()
            }
            | {
                "REPORT.md": {
                    "bytes": (staged / "REPORT.md").stat().st_size,
                    "sha256": sha256_file(staged / "REPORT.md"),
                }
            },
        }
        _write_json(staged / "manifest.json", manifest)
        os.replace(staged, output_dir)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--source-evaluation", required=True, type=Path)
    parser.add_argument("--legacy-runs-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", choices=("development", "validation"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    manifest = evaluate(
        fixture=args.fixture,
        source_evaluation=args.source_evaluation,
        legacy_runs_dir=args.legacy_runs_dir,
        config_path=args.config,
        role=args.role,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(json_safe(manifest), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
