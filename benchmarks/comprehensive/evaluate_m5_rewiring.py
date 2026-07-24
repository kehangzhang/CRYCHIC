"""Evaluate M5 robustness to bounded degree-matched H-prior rewiring."""

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

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.evaluate_m5_hprior import (
    FULL_METHOD,
    PERMUTED_METHOD,
    RAW_METHOD,
    _evaluate_dataset,
    _method_summary,
)
from benchmarks.comprehensive.generate_m5_hprior_fixture import (
    SCHEMA_VERSION as FIXTURE_SCHEMA,
)
from crychic.scoring import (
    HYPERGRAPH_SHRINKAGE_VERSION,
    FrozenHypergraphPrior,
    HypergraphShrinkageSpec,
    freeze_hypergraph_prior,
    permute_hypergraph_prior_degree_matched,
    rewire_hypergraph_prior_degree_matched,
)

SCHEMA_VERSION = "crychic-m5-rewiring-evaluation-v1"
CONFIG_SCHEMA_VERSION = "crychic-suggestions-next-m5-rewiring-config-v1"
REWIRE_FRACTIONS = (0.10, 0.25, 0.50)
GATED_FRACTIONS = (0.10, 0.25)
VIEW_COLUMNS = ("sender", "ligand", "receptor", "receiver", "pathway")
REWIRE_SEED_BASE = 20260740
FULL_PERMUTATION_SEED = 8675309


def rewired_method_name(fraction: float) -> str:
    """Return the canonical method key for one preregistered fraction."""

    percentage = int(round(float(fraction) * 100.0))
    return f"degree_matched_partial_rewire_{percentage:02d}pct"


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
        raise ValueError("M5 rewiring configuration schema is unsupported")
    expected_candidate = {
        "name": "crychic_m5_rewiring_robustness",
        "base_method": FULL_METHOD,
        "score_version": HYPERGRAPH_SHRINKAGE_VERSION,
        "node_ridge_penalty": 1.0,
        "edge_residual_penalty": 1.0,
        "view_columns": list(VIEW_COLUMNS),
        "rewire_fractions": list(REWIRE_FRACTIONS),
        "gated_rewire_fractions": list(GATED_FRACTIONS),
        "rewire_seed_base": REWIRE_SEED_BASE,
        "full_permutation_seed": FULL_PERMUTATION_SEED,
        "rewire_policy": (
            "shared_edge_subset_independent_view_derangement_exact_degree_match"
        ),
        "tuned_parameters": 0,
        "formal_inference_allowed": False,
    }
    candidate = config.get("candidate")
    if not isinstance(candidate, Mapping) or any(
        candidate.get(key) != value for key, value in expected_candidate.items()
    ):
        raise ValueError("M5 rewiring candidate identity or fixed policy changed")
    boundary = config.get("claim_boundary")
    if not isinstance(boundary, Mapping) or not all(
        boundary.get(field) is True
        for field in (
            "base_m5_formula_is_unchanged",
            "rewiring_is_outcome_blind",
            "every_view_degree_profile_is_preserved",
            "fifty_percent_rewiring_is_diagnostic_only",
            "no_p_q_or_probability_is_emitted",
            "synthetic_robustness_is_not_real_data_evidence",
        )
    ):
        raise ValueError("M5 rewiring claim boundary changed")
    if role not in {"development", "validation"}:
        raise ValueError("role must be development or validation")
    section = config.get(role)
    if not isinstance(section, Mapping):
        raise ValueError(f"M5 rewiring configuration lacks {role} section")
    expected_sha = section.get("fixture_manifest_sha256")
    if not isinstance(expected_sha, str) or expected_sha.startswith("PENDING"):
        raise ValueError(f"{role} fixture checksum is not frozen")
    if sha256_file(fixture_manifest) != expected_sha:
        raise ValueError(f"{role} fixture manifest checksum differs")
    return config


def _rewiring_diagnostics(
    full: FrozenHypergraphPrior,
    rewired: FrozenHypergraphPrior,
    *,
    requested_fraction: float,
) -> dict[str, object]:
    if tuple(edge.edge_id for edge in full.edges) != tuple(
        edge.edge_id for edge in rewired.edges
    ):
        raise ValueError("rewired prior edge universe differs")
    edge_changed = np.asarray(
        [
            source.memberships != target.memberships
            for source, target in zip(full.edges, rewired.edges, strict=True)
        ],
        dtype=bool,
    )
    membership_changed = sum(
        source_value != target_value
        for source, target in zip(full.edges, rewired.edges, strict=True)
        for source_value, target_value in zip(
            source.memberships, target.memberships, strict=True
        )
    )
    return {
        "method": rewired_method_name(requested_fraction),
        "requested_edge_fraction": float(requested_fraction),
        "changed_edges": int(edge_changed.sum()),
        "total_edges": len(full.edges),
        "actual_edge_fraction": float(edge_changed.mean()),
        "changed_memberships": int(membership_changed),
        "total_memberships": len(full.edges) * len(full.view_names),
        "actual_membership_fraction": float(
            membership_changed / (len(full.edges) * len(full.view_names))
        ),
        "degree_profiles_equal": full.degree_profile() == rewired.degree_profile(),
        "prior_id": rewired.prior_id,
        "parent_prior_id": rewired.parent_prior_id,
        "topology_kind": rewired.topology_kind,
    }


def _paired_bootstrap(
    metrics: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    left = metrics.loc[metrics["method"].eq(candidate), ["root_seed", metric]]
    right = metrics.loc[metrics["method"].eq(baseline), ["root_seed", metric]]
    paired = left.merge(
        right,
        on="root_seed",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    ).dropna()
    differences = (
        paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
    ).to_numpy(dtype=float)
    if len(differences) < 2:
        raise ValueError("M5 rewiring bootstrap requires at least two paired seeds")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(replicates, len(differences)))
    bootstrap = differences[draws].mean(axis=1)
    lower_is_better = metric == "effect_mse"
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "paired_seeds": len(differences),
        "mean_delta": float(differences.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "win_fraction": float(
            (differences < 0.0).mean()
            if lower_is_better
            else (differences > 0.0).mean()
        ),
    }


def _retention_table(summary: pd.DataFrame) -> pd.DataFrame:
    indexed = summary.set_index("method")
    full = indexed.loc[FULL_METHOD]
    permuted = indexed.loc[PERMUTED_METHOD]
    full_mse_gain = float(permuted["effect_mse"] - full["effect_mse"])
    full_ap_gain = float(full["average_precision"] - permuted["average_precision"])
    if full_mse_gain <= 0.0 or full_ap_gain <= 0.0:
        raise ValueError("correct H-prior lacks positive topology gain")
    rows = []
    for fraction in REWIRE_FRACTIONS:
        method = rewired_method_name(fraction)
        current = indexed.loc[method]
        rows.append(
            {
                "method": method,
                "requested_edge_fraction": fraction,
                "mse_topology_gain_retention": float(
                    (permuted["effect_mse"] - current["effect_mse"])
                    / full_mse_gain
                ),
                "ap_topology_gain_retention": float(
                    (current["average_precision"] - permuted["average_precision"])
                    / full_ap_gain
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _gate(
    *,
    diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
    retention: pd.DataFrame,
    config: Mapping[str, Any],
    role: str,
) -> dict[str, object]:
    gates = cast(Mapping[str, Any], config["gates"])
    section = cast(Mapping[str, Any], config[role])
    comparison_index = comparisons.set_index(["candidate", "baseline", "metric"])
    retention_index = retention.set_index("method")

    def comparison(candidate: str, baseline: str, metric: str) -> pd.Series:
        return cast(pd.Series, comparison_index.loc[(candidate, baseline, metric)])

    checks: dict[str, bool] = {
        "minimum_paired_seeds": bool(
            comparisons["paired_seeds"].min()
            >= int(section["minimum_paired_seeds"])
        ),
        "all_degree_profiles_equal": bool(
            diagnostics["degree_profiles_equal"].all()
        ),
        "rewire_fraction_accuracy": bool(
            (
                diagnostics["actual_edge_fraction"]
                - diagnostics["requested_edge_fraction"]
            )
            .abs()
            .max()
            <= float(gates["maximum_edge_fraction_absolute_error"])
        ),
    }
    minimum_retention = cast(Mapping[str, Any], gates["minimum_gain_retention"])
    for fraction in GATED_FRACTIONS:
        method = rewired_method_name(fraction)
        label = f"{int(round(fraction * 100)):02d}pct"
        mse_raw = comparison(method, RAW_METHOD, "effect_mse")
        mse_permuted = comparison(method, PERMUTED_METHOD, "effect_mse")
        ap_raw = comparison(method, RAW_METHOD, "average_precision")
        ap_permuted = comparison(method, PERMUTED_METHOD, "average_precision")
        threshold = float(minimum_retention[method])
        checks[f"{label}_mse_gain_vs_raw"] = float(mse_raw["ci_high"]) < 0.0
        checks[f"{label}_mse_gain_vs_permuted"] = (
            float(mse_permuted["ci_high"]) < 0.0
        )
        checks[f"{label}_ap_gain_vs_raw"] = float(ap_raw["ci_low"]) > 0.0
        checks[f"{label}_ap_gain_vs_permuted"] = (
            float(ap_permuted["ci_low"]) > 0.0
        )
        checks[f"{label}_mse_gain_retained"] = (
            float(retention_index.loc[method, "mse_topology_gain_retention"])
            >= threshold
        )
        checks[f"{label}_ap_gain_retained"] = (
            float(retention_index.loc[method, "ap_topology_gain_retention"])
            >= threshold
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "status": (
            "DEVELOPMENT_PASS" if role == "development" else "ACCEPT"
        )
        if all(checks.values())
        else "REJECT",
        "checks": checks,
        "claim_boundary": "synthetic bounded H-prior rewiring robustness only",
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
    """Run the preregistered M5 bounded-rewiring robustness benchmark."""

    started = time.perf_counter()
    fixture_dir = fixture_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    fixture_manifest_path = fixture_dir / "manifest.json"
    fixture = _read_json(fixture_manifest_path)
    if fixture.get("schema_version") != FIXTURE_SCHEMA:
        raise ValueError("M5 rewiring fixture schema is unsupported")
    config = _validated_config(
        config_path, role=role, fixture_manifest=fixture_manifest_path
    )
    section = cast(Mapping[str, Any], config[role])
    if list(map(int, fixture.get("seeds", []))) != list(
        map(int, cast(Sequence[int], section["seeds"]))
    ):
        raise ValueError("M5 rewiring fixture seeds differ from frozen role seeds")

    topology_record = cast(Mapping[str, Any], fixture["hypergraph_prior"])
    topology_path = fixture_dir / str(topology_record["filename"])
    if sha256_file(topology_path) != topology_record["sha256"]:
        raise ValueError("M5 rewiring H-prior checksum differs")
    topology = pd.read_csv(topology_path, sep="\t")
    full = freeze_hypergraph_prior(topology, view_columns=VIEW_COLUMNS)
    permuted = permute_hypergraph_prior_degree_matched(
        full, seed=FULL_PERMUTATION_SEED
    )
    rewired_priors = {
        rewired_method_name(fraction): rewire_hypergraph_prior_degree_matched(
            full,
            fraction=fraction,
            seed=REWIRE_SEED_BASE + index,
        )
        for index, fraction in enumerate(REWIRE_FRACTIONS)
    }
    diagnostics = pd.DataFrame.from_records(
        [
            _rewiring_diagnostics(
                full,
                rewired_priors[rewired_method_name(fraction)],
                requested_fraction=fraction,
            )
            for fraction in REWIRE_FRACTIONS
        ]
    )
    priors = {FULL_METHOD: full, PERMUTED_METHOD: permuted, **rewired_priors}
    spec = HypergraphShrinkageSpec(
        node_ridge_penalty=1.0,
        edge_residual_penalty=1.0,
    )

    truth_record = cast(Mapping[str, Any], fixture["truth"])
    truth_path = fixture_dir / str(truth_record["filename"])
    if sha256_file(truth_path) != truth_record["sha256"]:
        raise ValueError("M5 rewiring truth checksum differs")
    truth = pd.read_csv(truth_path, sep="\t")
    metric_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    for record in cast(Sequence[Mapping[str, Any]], fixture["records"]):
        dataset_id = str(record["dataset_id"])
        input_path = fixture_dir / str(record["input"])
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"M5 rewiring input checksum mismatch: {dataset_id}")
        estimates = pd.read_csv(input_path, sep="\t")
        dataset_truth = truth.loc[truth["dataset_id"].eq(dataset_id)].drop(
            columns=["dataset_id", "root_seed"]
        )
        metrics, _, fits = _evaluate_dataset(
            dataset_id=dataset_id,
            root_seed=int(record["root_seed"]),
            estimates=estimates,
            truth=dataset_truth,
            priors=priors,
            spec=spec,
        )
        metric_frames.append(metrics)
        fit_frames.append(fits)
    metrics = pd.concat(metric_frames, ignore_index=True)
    fits = pd.concat(fit_frames, ignore_index=True)
    summary = _method_summary(metrics)

    comparison_specs: list[tuple[str, str, str]] = [
        (FULL_METHOD, RAW_METHOD, "effect_mse"),
        (FULL_METHOD, PERMUTED_METHOD, "effect_mse"),
        (FULL_METHOD, RAW_METHOD, "average_precision"),
        (FULL_METHOD, PERMUTED_METHOD, "average_precision"),
    ]
    for fraction in REWIRE_FRACTIONS:
        method = rewired_method_name(fraction)
        comparison_specs.extend(
            (method, baseline, metric)
            for baseline in (RAW_METHOD, PERMUTED_METHOD, FULL_METHOD)
            for metric in ("effect_mse", "average_precision")
        )
    comparisons = pd.DataFrame.from_records(
        [
            _paired_bootstrap(
                metrics,
                candidate=candidate,
                baseline=baseline,
                metric=metric,
                replicates=int(section["bootstrap_replicates"]),
                seed=int(section["bootstrap_seed"]) + index,
            )
            for index, (candidate, baseline, metric) in enumerate(comparison_specs)
        ]
    )
    retention = _retention_table(summary)
    acceptance = _gate(
        diagnostics=diagnostics,
        comparisons=comparisons,
        retention=retention,
        config=config,
        role=role,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        tables = {
            "replicate_metrics.tsv": metrics,
            "fit_diagnostics.tsv": fits,
            "method_summary.tsv": summary,
            "paired_comparisons.tsv": comparisons,
            "rewiring_diagnostics.tsv": diagnostics,
            "gain_retention.tsv": retention,
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
            "base_prior_id": full.prior_id,
            "full_permutation_prior_id": permuted.prior_id,
            "rewired_prior_ids": {
                method: prior.prior_id for method, prior in rewired_priors.items()
            },
            "datasets": len(metric_frames),
            "edges_per_dataset": int(fixture["n_edges"]),
            "elapsed_seconds": time.perf_counter() - started,
            "tables": {
                filename: {
                    "sha256": sha256_file(staged / filename),
                    "rows": len(table),
                }
                for filename, table in tables.items()
            },
            "acceptance_sha256": sha256_file(staged / "acceptance.json"),
            "claim_boundary": {
                "formal_inference_allowed": False,
                "general_sota_claim_allowed": False,
                "validated_estimand": "synthetic bounded H-prior rewiring robustness",
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
