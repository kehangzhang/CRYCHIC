"""Evaluate frozen H_prior shrinkage and topology-matched controls."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.generate_m5_hprior_fixture import (
    SCHEMA_VERSION as FIXTURE_SCHEMA,
)
from crychic.scoring import (
    HYPERGRAPH_SHRINKAGE_VERSION,
    HypergraphShrinkageSpec,
    fit_hypergraph_prior_shrinkage,
    freeze_hypergraph_prior,
    permute_hypergraph_prior_degree_matched,
    select_hypergraph_prior_views,
)

SCHEMA_VERSION = "crychic-m5-hprior-evaluation-v1"
CONFIG_SCHEMA_VERSION = "crychic-suggestions-next-m5-config-v1"
RAW_METHOD = "no_hypergraph_prior_raw_effect"
FULL_METHOD = "crychic_m5_full_hypergraph_prior"
PERMUTED_METHOD = "degree_matched_permuted_hypergraph_prior"
LIGAND_METHOD = "ligand_only_hypergraph_prior"
RECEPTOR_METHOD = "receptor_only_hypergraph_prior"
PATHWAY_METHOD = "pathway_only_hypergraph_prior"
PARTIAL_METHODS = (LIGAND_METHOD, RECEPTOR_METHOD, PATHWAY_METHOD)


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
        raise ValueError("M5 configuration schema is unsupported")
    expected_candidate = {
        "name": FULL_METHOD,
        "score_version": HYPERGRAPH_SHRINKAGE_VERSION,
        "node_ridge_penalty": 1.0,
        "edge_residual_penalty": 1.0,
        "model": "beta_equals_incidence_theta_plus_edge_residual",
        "view_columns": ["sender", "ligand", "receptor", "receiver", "pathway"],
        "permutation_policy": "independent_view_stub_permutation_exact_degree_match",
        "permutation_seed": 8675309,
        "tuned_parameters": 0,
        "formal_inference_allowed": False,
    }
    candidate = config.get("candidate")
    if not isinstance(candidate, Mapping) or any(
        candidate.get(key) != value for key, value in expected_candidate.items()
    ):
        raise ValueError("M5 candidate identity or fixed policy changed")
    boundary = config.get("claim_boundary")
    if not isinstance(boundary, Mapping) or not all(
        boundary.get(field) is True
        for field in (
            "hprior_is_frozen_before_outcomes",
            "result_hypergraph_is_not_used_as_prior",
            "degree_matched_permutation_is_mandatory",
            "no_prior_and_partial_view_controls_are_mandatory",
            "m5_emits_no_p_q_or_probability",
            "effect_summary_fixture_is_not_general_sota_evidence",
        )
    ):
        raise ValueError("M5 claim boundary changed")
    if role not in {"development", "validation"}:
        raise ValueError("role must be development or validation")
    section = config.get(role)
    if not isinstance(section, Mapping):
        raise ValueError(f"M5 configuration lacks {role} section")
    expected_sha = section.get("fixture_manifest_sha256")
    if not isinstance(expected_sha, str) or expected_sha.startswith("PENDING"):
        raise ValueError(f"{role} fixture checksum is not frozen")
    if sha256_file(fixture_manifest) != expected_sha:
        raise ValueError(f"{role} fixture manifest checksum differs")
    return config


def _method_metrics(
    *,
    dataset_id: str,
    root_seed: int,
    method: str,
    score: np.ndarray,
    truth: pd.DataFrame,
) -> dict[str, object]:
    true_effect = truth["true_effect"].to_numpy(dtype=float)
    labels = truth["expected_active"].astype(int).to_numpy()
    expected_direction = truth["expected_direction"].to_numpy(dtype=int)
    return {
        "dataset_id": dataset_id,
        "root_seed": root_seed,
        "method": method,
        "effect_mse": float(np.mean((score - true_effect) ** 2)),
        "effect_spearman": float(spearmanr(score, true_effect).statistic),
        "average_precision": float(average_precision_score(labels, np.abs(score))),
        "auroc": float(roc_auc_score(labels, np.abs(score))),
        "direction_accuracy": float(np.mean(np.sign(score) == expected_direction)),
        "formal_inference_allowed": False,
    }


def _evaluate_dataset(
    *,
    dataset_id: str,
    root_seed: int,
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    priors: Mapping[str, Any],
    spec: HypergraphShrinkageSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = truth.sort_values("edge_id", kind="stable", ignore_index=True)
    estimates = estimates.sort_values("edge_id", kind="stable", ignore_index=True)
    rows = [
        _method_metrics(
            dataset_id=dataset_id,
            root_seed=root_seed,
            method=RAW_METHOD,
            score=estimates["estimate"].to_numpy(dtype=float),
            truth=truth,
        )
    ]
    score_tables = [
        pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "root_seed": root_seed,
                "edge_id": estimates["edge_id"],
                "method": RAW_METHOD,
                "effect_estimate": estimates["estimate"],
                "formal_inference_allowed": False,
            }
        )
    ]
    fits: list[dict[str, object]] = []
    for method, prior in priors.items():
        result, fit = fit_hypergraph_prior_shrinkage(
            estimates,
            prior=prior,
            spec=spec,
            precision_column="observation_precision",
        )
        score = result["shrunk_estimate"].to_numpy(dtype=float)
        rows.append(
            _method_metrics(
                dataset_id=dataset_id,
                root_seed=root_seed,
                method=method,
                score=score,
                truth=truth,
            )
        )
        score_tables.append(
            pd.DataFrame(
                {
                    "dataset_id": dataset_id,
                    "root_seed": root_seed,
                    "edge_id": result["edge_id"],
                    "method": method,
                    "effect_estimate": score,
                    "formal_inference_allowed": False,
                }
            )
        )
        fits.append(
            {
                "dataset_id": dataset_id,
                "root_seed": root_seed,
                "method": method,
                "topology_kind": prior.topology_kind,
                "views": "|".join(prior.view_names),
                **fit.to_dict(),
            }
        )
    metrics = pd.DataFrame.from_records(rows)
    scores = pd.concat(score_tables, ignore_index=True)
    fit_table = pd.DataFrame.from_records(fits)
    return metrics, scores, fit_table


def _method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby("method", observed=True, sort=True)
        .agg(
            seeds=("root_seed", "nunique"),
            effect_mse=("effect_mse", "mean"),
            effect_spearman=("effect_spearman", "mean"),
            average_precision=("average_precision", "mean"),
            auroc=("auroc", "mean"),
            direction_accuracy=("direction_accuracy", "mean"),
        )
        .reset_index()
    )
    summary["mse_rank"] = summary["effect_mse"].rank(method="min", ascending=True)
    summary["average_precision_rank"] = summary["average_precision"].rank(
        method="min", ascending=False
    )
    summary["auroc_rank"] = summary["auroc"].rank(method="min", ascending=False)
    return summary.sort_values(
        ["mse_rank", "average_precision_rank", "method"],
        kind="stable",
        ignore_index=True,
    )


def _paired_bootstrap(
    metrics: pd.DataFrame,
    *,
    baseline: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    candidate = metrics.loc[metrics["method"].eq(FULL_METHOD), ["root_seed", metric]]
    reference = metrics.loc[metrics["method"].eq(baseline), ["root_seed", metric]]
    paired = candidate.merge(
        reference,
        on="root_seed",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    ).dropna()
    differences = (
        paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
    ).to_numpy(dtype=float)
    if len(differences) < 2:
        raise ValueError("M5 paired bootstrap requires at least two seeds")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(replicates, len(differences)))
    bootstrap = differences[draws].mean(axis=1)
    return {
        "candidate": FULL_METHOD,
        "baseline": baseline,
        "metric": metric,
        "paired_seeds": int(len(differences)),
        "mean_delta": float(differences.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "win_fraction": float(
            (differences < 0.0).mean()
            if metric == "effect_mse"
            else (differences > 0.0).mean()
        ),
    }


def _gate(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    role: str,
) -> dict[str, object]:
    gates = cast(Mapping[str, Any], config["gates"])
    section = cast(Mapping[str, Any], config[role])
    indexed = summary.set_index("method")
    full = indexed.loc[FULL_METHOD]

    def row(baseline: str, metric: str) -> pd.Series:
        selected = comparisons.loc[
            comparisons["baseline"].eq(baseline) & comparisons["metric"].eq(metric)
        ]
        if len(selected) != 1:
            raise ValueError(f"missing M5 comparison: {baseline}, {metric}")
        return selected.iloc[0]

    raw_mse = row(RAW_METHOD, "effect_mse")
    permuted_mse = row(PERMUTED_METHOD, "effect_mse")
    raw_ap = row(RAW_METHOD, "average_precision")
    permuted_ap = row(PERMUTED_METHOD, "average_precision")
    partial_mse = [row(method, "effect_mse") for method in PARTIAL_METHODS]
    direction = row(RAW_METHOD, "direction_accuracy")
    checks = {
        "minimum_paired_seeds": int(raw_mse["paired_seeds"])
        >= int(section["minimum_paired_seeds"]),
        "mse_gain_vs_raw": float(raw_mse["ci_high"])
        < float(gates["mse_delta_vs_raw_ci_upper_exclusive"]),
        "mse_topology_gain": float(permuted_mse["ci_high"])
        < float(gates["mse_delta_vs_permuted_ci_upper_exclusive"]),
        "ap_gain_vs_raw": float(raw_ap["ci_low"])
        > float(gates["ap_delta_vs_raw_ci_lower_exclusive"]),
        "ap_topology_gain": float(permuted_ap["ci_low"])
        > float(gates["ap_delta_vs_permuted_ci_lower_exclusive"]),
        "full_better_than_each_partial": all(
            float(value["ci_high"])
            < float(gates["mse_delta_vs_partial_ci_upper_exclusive"])
            for value in partial_mse
        ),
        "direction_noninferiority": float(direction["ci_low"])
        >= float(gates["direction_delta_vs_raw_ci_lower_minimum"]),
        "rank_one_mse": float(full["mse_rank"]) == 1.0,
        "rank_one_ap": float(full["average_precision_rank"]) == 1.0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "status": ("DEVELOPMENT_PASS" if role == "development" else "ACCEPT")
        if all(checks.values())
        else "REJECT",
        "checks": checks,
        "claim_boundary": "synthetic additive H_prior effect-summary estimand only",
    }


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


def evaluate(
    fixture_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    role: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the locked M5 H_prior benchmark and topology controls."""

    started = time.perf_counter()
    fixture_dir = fixture_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    fixture_manifest_path = fixture_dir / "manifest.json"
    fixture = _read_json(fixture_manifest_path)
    if fixture.get("schema_version") != FIXTURE_SCHEMA:
        raise ValueError("M5 fixture schema is unsupported")
    config = _validated_config(
        config_path, role=role, fixture_manifest=fixture_manifest_path
    )
    section = cast(Mapping[str, Any], config[role])
    if list(map(int, fixture.get("seeds", []))) != list(
        map(int, cast(Sequence[int], section["seeds"]))
    ):
        raise ValueError("M5 fixture seeds differ from frozen role seeds")
    topology_record = cast(Mapping[str, Any], fixture["hypergraph_prior"])
    topology_path = fixture_dir / str(topology_record["filename"])
    if sha256_file(topology_path) != topology_record["sha256"]:
        raise ValueError("M5 H_prior checksum differs")
    topology = pd.read_csv(topology_path, sep="\t")
    candidate = cast(Mapping[str, Any], config["candidate"])
    full_prior = freeze_hypergraph_prior(
        topology, view_columns=tuple(map(str, candidate["view_columns"]))
    )
    permuted = permute_hypergraph_prior_degree_matched(
        full_prior, seed=int(candidate["permutation_seed"])
    )
    if full_prior.degree_profile() != permuted.degree_profile():
        raise RuntimeError("M5 topology control failed exact degree matching")
    priors = {
        FULL_METHOD: full_prior,
        PERMUTED_METHOD: permuted,
        LIGAND_METHOD: select_hypergraph_prior_views(
            full_prior, view_names=("ligand",)
        ),
        RECEPTOR_METHOD: select_hypergraph_prior_views(
            full_prior, view_names=("receptor",)
        ),
        PATHWAY_METHOD: select_hypergraph_prior_views(
            full_prior, view_names=("pathway",)
        ),
    }
    spec = HypergraphShrinkageSpec(
        node_ridge_penalty=float(candidate["node_ridge_penalty"]),
        edge_residual_penalty=float(candidate["edge_residual_penalty"]),
    )
    truth_record = cast(Mapping[str, Any], fixture["truth"])
    truth_path = fixture_dir / str(truth_record["filename"])
    if sha256_file(truth_path) != truth_record["sha256"]:
        raise ValueError("M5 truth checksum differs")
    truth = pd.read_csv(truth_path, sep="\t")
    metric_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    for record in cast(Sequence[Mapping[str, Any]], fixture["records"]):
        dataset_id = str(record["dataset_id"])
        input_path = fixture_dir / str(record["input"])
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"M5 input checksum mismatch: {dataset_id}")
        estimates = pd.read_csv(input_path, sep="\t")
        dataset_truth = truth.loc[truth["dataset_id"].eq(dataset_id)].drop(
            columns=["dataset_id", "root_seed"]
        )
        metrics, scores, fits = _evaluate_dataset(
            dataset_id=dataset_id,
            root_seed=int(record["root_seed"]),
            estimates=estimates,
            truth=dataset_truth,
            priors=priors,
            spec=spec,
        )
        metric_frames.append(metrics)
        score_frames.append(scores)
        fit_frames.append(fits)
    metrics = pd.concat(metric_frames, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)
    fits = pd.concat(fit_frames, ignore_index=True)
    summary = _method_summary(metrics)
    comparisons = pd.DataFrame.from_records(
        [
            _paired_bootstrap(
                metrics,
                baseline=baseline,
                metric=metric,
                replicates=int(section["bootstrap_replicates"]),
                seed=int(section["bootstrap_seed"]) + index,
            )
            for index, (baseline, metric) in enumerate(
                (
                    (RAW_METHOD, "effect_mse"),
                    (PERMUTED_METHOD, "effect_mse"),
                    (RAW_METHOD, "average_precision"),
                    (PERMUTED_METHOD, "average_precision"),
                    (LIGAND_METHOD, "effect_mse"),
                    (RECEPTOR_METHOD, "effect_mse"),
                    (PATHWAY_METHOD, "effect_mse"),
                    (RAW_METHOD, "direction_accuracy"),
                )
            )
        ]
    )
    acceptance = _gate(summary, comparisons, config=config, role=role)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        tables = {
            "edge_scores.tsv": scores,
            "replicate_metrics.tsv": metrics,
            "fit_diagnostics.tsv": fits,
            "method_summary.tsv": summary,
            "paired_comparisons.tsv": comparisons,
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False, lineterminator="\n")
        _write_json(staged / "acceptance.json", acceptance)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "role": role,
            "acceptance": acceptance,
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "fixture": {
                "path": str(fixture_dir),
                "manifest_sha256": sha256_file(fixture_manifest_path),
            },
            "configuration": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "candidate": dict(candidate),
            "full_prior_id": full_prior.prior_id,
            "permuted_prior_id": permuted.prior_id,
            "degree_profiles_equal": True,
            "datasets": len(metric_frames),
            "edges_per_dataset": int(fixture["n_edges"]),
            "elapsed_seconds": time.perf_counter() - started,
            "tables": {
                filename: {
                    "sha256": sha256_file(staged / filename),
                    "rows": int(len(table)),
                }
                for filename, table in tables.items()
            },
            "acceptance_sha256": sha256_file(staged / "acceptance.json"),
            "claim_boundary": {
                "formal_inference_allowed": False,
                "general_sota_claim_allowed": False,
                "validated_estimand": "synthetic additive node-effect shrinkage",
            },
        }
        _write_json(staged / "manifest.json", manifest)
        _publish(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "validation"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = evaluate(
        args.fixture_dir,
        args.config,
        args.output_dir,
        role=args.role,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
