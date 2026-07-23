"""Run a preregistered native-hyperedge mechanism-near-miss benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from scipy.stats import t
from sklearn.metrics import average_precision_score

from benchmarks.adapters.common import (
    canonical_digest,
    git_metadata,
    json_safe,
    sha256_file,
)
from benchmarks.metrics.mechanism_specificity import (
    REQUIRED_SCENARIOS,
    component_truth_from_mapping,
)
from benchmarks.simulation.mechanism_specificity import (
    frozen_design_manifest,
    generate_mechanism_specificity_evidence_from_lineages,
)
from crychic.core import SeedLineage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "benchmarks/configs/native_hypergraph_truth_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark_work/native_hypergraph_truth_v1"
SCHEMA_VERSION = "crychic-native-hypergraph-truth-result-v1"
CONFIG_SCHEMA_VERSION = "crychic-native-hypergraph-truth-benchmark-v1"
PHASE = "native_hypergraph_truth_independent_holdout_v1"
METHODS = (
    "crychic_native_hyperedge",
    "pairwise_union",
    "clique_expansion",
    "simple_complex_min",
)
BASELINES = METHODS[1:]
FORMULAS = {
    "crychic_native_hyperedge": "integrated_lr_effect",
    "pairwise_union": "max(availability_effect,receptor_gate,sender_effect)",
    "clique_expansion": (
        "cuberoot(availability_effect*receptor_gate*sender_effect)"
    ),
    "simple_complex_min": (
        "min(availability_effect,receptor_gate,sender_effect)"
    ),
}
KEY_COLUMNS = ("seed", "known_edge_id", "scenario")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


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


def _json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return cast(dict[str, Any], value)


def _yaml_object(path: Path) -> dict[str, object]:
    value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be an object: {path}")
    return cast(dict[str, object], value)


def _config(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = path.expanduser().resolve()
    raw = _json_object(resolved)
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported native hypergraph truth config")
    truth_raw = raw.get("truth")
    holdout = raw.get("holdout")
    methods = raw.get("methods")
    endpoints = raw.get("truth_endpoints")
    gate = raw.get("gate")
    if not all(
        isinstance(item, dict)
        for item in (truth_raw, holdout, methods, endpoints, gate)
    ):
        raise ValueError("native hypergraph config sections must be objects")
    truth_raw = cast(dict[str, Any], truth_raw)
    truth_path = (REPO_ROOT / str(truth_raw["path"])).resolve()
    if (
        not truth_path.is_file()
        or sha256_file(truth_path) != str(truth_raw["sha256"])
    ):
        raise ValueError("component truth checksum does not match config")
    methods = cast(dict[str, Any], methods)
    if tuple(methods) != METHODS or {
        method: str(methods[method].get("formula")) for method in methods
    } != FORMULAS:
        raise ValueError("method formulas differ from the frozen implementation")
    universe = raw.get("candidate_universe")
    if not isinstance(universe, dict) or tuple(universe.get("scenarios", ())) != tuple(
        REQUIRED_SCENARIOS
    ):
        raise ValueError("candidate scenario universe differs from the generator")
    exact = cast(dict[str, Any], cast(dict[str, Any], endpoints)["exact_hyperedge"])
    partial = cast(
        dict[str, Any], cast(dict[str, Any], endpoints)["partial_canonical_lr"]
    )
    if tuple(exact.get("positive_scenarios", ())) != ("active",) or tuple(
        partial.get("positive_scenarios", ())
    ) != ("active", "ligand_only"):
        raise ValueError("truth endpoint labels differ from the frozen evaluator")
    return raw, truth_path


def holdout_seed_lineages(config: Mapping[str, Any]) -> tuple[SeedLineage, ...]:
    """Derive the call-order-independent holdout seed set from the config."""

    holdout = cast(Mapping[str, Any], config["holdout"])
    count = int(holdout["seed_count"])
    if count < 2 or bool(holdout.get("tuning_performed")):
        raise ValueError("holdout requires at least two seeds and no tuning")
    root = SeedLineage(int(holdout["root_seed"])).derive(str(holdout["namespace"]))
    result = tuple(
        root.derive("campaign-seed", f"index={index:03d}")
        for index in range(count)
    )
    if len({item.seed for item in result}) != count:
        raise RuntimeError("native hypergraph holdout seed collision")
    return result


def score_candidate_universe(evidence: pd.DataFrame) -> pd.DataFrame:
    """Score the identical candidate universe with frozen representation heads."""

    required = {
        *KEY_COLUMNS,
        "availability_effect",
        "receptor_gate",
        "sender_effect",
        "integrated_lr_effect",
        "status",
    }
    missing = required.difference(evidence.columns)
    if evidence.empty or missing:
        raise ValueError(f"mechanism evidence is invalid: missing={sorted(missing)}")
    if evidence.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("mechanism candidate identities must be unique")
    if not evidence["status"].astype(str).eq("observed").all():
        raise ValueError("hypergraph truth benchmark forbids missing candidates")
    numeric_columns = (
        "availability_effect",
        "receptor_gate",
        "sender_effect",
        "integrated_lr_effect",
    )
    numeric = evidence.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (
        numeric.to_numpy(dtype=float) < 0.0
    ).any():
        raise ValueError("hypergraph component scores must be finite and non-negative")
    availability = numeric["availability_effect"].to_numpy(dtype=float)
    receptor = numeric["receptor_gate"].to_numpy(dtype=float)
    sender = numeric["sender_effect"].to_numpy(dtype=float)
    scores = {
        "crychic_native_hyperedge": numeric["integrated_lr_effect"].to_numpy(
            dtype=float
        ),
        "pairwise_union": np.maximum.reduce([availability, receptor, sender]),
        "clique_expansion": np.cbrt(availability * receptor * sender),
        "simple_complex_min": np.minimum.reduce([availability, receptor, sender]),
    }
    base = evidence.loc[:, list(KEY_COLUMNS)].copy()
    base["exact_truth"] = base["scenario"].astype(str).eq("active").astype(int)
    base["partial_truth"] = base["scenario"].astype(str).isin(
        {"active", "ligand_only"}
    ).astype(int)
    parts = []
    for method in METHODS:
        part = base.copy()
        part.insert(3, "method", method)
        part["score"] = scores[method]
        part["coverage"] = 1.0
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["seed", "known_edge_id", "method", "scenario"],
        kind="stable",
        ignore_index=True,
    )


def _average_precision(labels: pd.Series, scores: pd.Series) -> float:
    truth = labels.to_numpy(dtype=int)
    values = scores.to_numpy(dtype=float)
    if set(truth) != {0, 1} or not np.isfinite(values).all():
        raise ValueError("AP requires finite scores and both truth classes")
    return float(average_precision_score(truth, values))


def replicate_metrics(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return edge-level and equal-edge seed-level AP records."""

    records: list[dict[str, Any]] = []
    for (seed, edge, method), group in candidates.groupby(
        ["seed", "known_edge_id", "method"], observed=True, sort=True
    ):
        if len(group) != len(REQUIRED_SCENARIOS) or set(group["scenario"]) != set(
            REQUIRED_SCENARIOS
        ):
            raise ValueError("one seed-edge-method candidate universe is incomplete")
        records.append(
            {
                "seed": int(seed),
                "known_edge_id": str(edge),
                "method": str(method),
                "candidate_count": len(group),
                "coverage": float(group["coverage"].mean()),
                "exact_hyperedge_ap": _average_precision(
                    group["exact_truth"], group["score"]
                ),
                "partial_canonical_lr_ap": _average_precision(
                    group["partial_truth"], group["score"]
                ),
            }
        )
    edge = pd.DataFrame.from_records(records)
    macro = (
        edge.groupby(["seed", "method"], observed=True, sort=True)
        .agg(
            known_edge_count=("known_edge_id", "nunique"),
            candidate_count=("candidate_count", "sum"),
            coverage=("coverage", "mean"),
            exact_hyperedge_ap=("exact_hyperedge_ap", "mean"),
            partial_canonical_lr_ap=("partial_canonical_lr_ap", "mean"),
        )
        .reset_index()
    )
    return edge, macro


def _mean_interval(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("paired t intervals require at least two finite values")
    mean = float(array.mean())
    radius = float(t.ppf(0.975, len(array) - 1)) * float(
        array.std(ddof=1) / math.sqrt(len(array))
    )
    return {
        "n": len(array),
        "mean": mean,
        "ci_low": mean - radius,
        "ci_high": mean + radius,
    }


def summarize_metrics(
    macro: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize method AP and apply the frozen paired-superiority gate."""

    expected_seeds = int(cast(Mapping[str, Any], config["holdout"])["seed_count"])
    if macro["seed"].nunique() != expected_seeds:
        raise ValueError("macro metrics do not contain every holdout seed")
    records = []
    for method, group in macro.groupby("method", observed=True, sort=True):
        exact = _mean_interval(group["exact_hyperedge_ap"].to_numpy(dtype=float))
        partial = _mean_interval(
            group["partial_canonical_lr_ap"].to_numpy(dtype=float)
        )
        records.append(
            {
                "method": str(method),
                "n_seeds": exact["n"],
                "candidate_count_per_seed": int(group["candidate_count"].iloc[0]),
                "coverage_mean": float(group["coverage"].mean()),
                **{f"exact_{key}": value for key, value in exact.items()},
                **{f"partial_{key}": value for key, value in partial.items()},
            }
        )
    summary = pd.DataFrame.from_records(records).sort_values(
        "exact_mean", ascending=False, kind="stable", ignore_index=True
    )
    crychic = macro.loc[
        macro["method"].eq("crychic_native_hyperedge"),
        ["seed", "exact_hyperedge_ap", "partial_canonical_lr_ap"],
    ].rename(
        columns={
            "exact_hyperedge_ap": "crychic_exact",
            "partial_canonical_lr_ap": "crychic_partial",
        }
    )
    differences = []
    for baseline in BASELINES:
        comparator = macro.loc[
            macro["method"].eq(baseline),
            ["seed", "exact_hyperedge_ap", "partial_canonical_lr_ap"],
        ].rename(
            columns={
                "exact_hyperedge_ap": "baseline_exact",
                "partial_canonical_lr_ap": "baseline_partial",
            }
        )
        paired = crychic.merge(comparator, on="seed", validate="one_to_one")
        exact = _mean_interval(
            (paired["crychic_exact"] - paired["baseline_exact"]).to_numpy(float)
        )
        partial = _mean_interval(
            (paired["crychic_partial"] - paired["baseline_partial"]).to_numpy(
                float
            )
        )
        differences.append(
            {
                "baseline": baseline,
                **{f"exact_difference_{key}": value for key, value in exact.items()},
                **{
                    f"partial_difference_{key}": value
                    for key, value in partial.items()
                },
            }
        )
    paired = pd.DataFrame.from_records(differences)
    gate_config = cast(Mapping[str, Any], config["gate"])
    crychic_exact = float(
        summary.loc[
            summary["method"].eq("crychic_native_hyperedge"), "exact_mean"
        ].iloc[0]
    )
    counts_identical = summary["candidate_count_per_seed"].nunique() == 1
    coverage_identical = summary["coverage_mean"].nunique() == 1 and summary[
        "coverage_mean"
    ].eq(1.0).all()
    checks = {
        "complete_seed_count": expected_seeds
        >= int(gate_config["minimum_complete_seeds"]),
        "minimum_crychic_exact_ap": crychic_exact
        >= float(gate_config["minimum_crychic_exact_ap"]),
        "exact_ap_superior_to_every_baseline": bool(
            paired["exact_difference_ci_low"].gt(
                float(gate_config["minimum_exact_ap_paired_difference_ci_low"])
            ).all()
        ),
        "identical_candidate_count": bool(counts_identical),
        "identical_coverage": bool(coverage_identical),
        "partial_ap_reported": bool(summary["partial_mean"].notna().all()),
    }
    gate = {
        "name": "native_hypergraph_exact_truth_superiority",
        "status": "PASS" if all(checks.values()) else "REJECT",
        "checks": checks,
        "thresholds": dict(gate_config),
        "crychic_exact_ap": crychic_exact,
    }
    return summary, paired, gate


def run_benchmark(
    config_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate, score, and atomically publish the frozen holdout campaign."""

    config, truth_path = _config(config_path)
    truth = component_truth_from_mapping(_yaml_object(truth_path))
    lineages = holdout_seed_lineages(config)
    generated = generate_mechanism_specificity_evidence_from_lineages(
        phase=PHASE,
        seed_lineages=lineages,
        truth=truth,
    )
    candidates = score_candidate_universe(generated.evidence)
    edge_metrics, macro_metrics = replicate_metrics(candidates)
    summary, paired, gate = summarize_metrics(macro_metrics, config)

    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        paths = {
            "evidence.parquet": generated.evidence,
            "candidate_scores.parquet": candidates,
            "edge_replicate_metrics.tsv": edge_metrics,
            "macro_replicate_metrics.tsv": macro_metrics,
            "method_summary.tsv": summary,
            "paired_differences.tsv": paired,
        }
        written: list[Path] = []
        for name, table in paths.items():
            path = staged / name
            if path.suffix == ".parquet":
                table.to_parquet(path, index=False)
            else:
                table.to_csv(path, sep="\t", index=False)
            written.append(path)
        gate_path = staged / "gate.json"
        _write_json(gate_path, gate)
        written.append(gate_path)
        report = [
            "# Native hypergraph truth benchmark",
            "",
            f"Gate: **{gate['status']}**",
            "",
            "| Method | Exact AP | 95% CI | Partial canonical-LR AP | Coverage |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in summary.itertuples(index=False):
            report.append(
                f"| {row.method} | {row.exact_mean:.4f} | "
                f"[{row.exact_ci_low:.4f}, {row.exact_ci_high:.4f}] | "
                f"{row.partial_mean:.4f} | {row.coverage_mean:.4f} |"
            )
        report.extend(
            [
                "",
                "Exact truth requires the complete sender-LR-receiver-program event. "
                "Partial truth accepts active and ligand-only canonical LR events and "
                "is reported as a specificity tradeoff, not a promotion gate.",
                "",
                "All methods score the same seven candidates for every seed and known "
                "edge. Missing candidates fail the campaign; no score is zero-imputed.",
            ]
        )
        report_path = staged / "REPORT.md"
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        written.append(report_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "evaluation_phase": PHASE,
            "benchmark_scope": "synthetic_mechanism_near_miss_hyperedge_truth",
            "claim_limit": "not_real_data_and_not_canonical_pairwise_accuracy",
            "holdout": {
                "root_seed": config["holdout"]["root_seed"],
                "namespace": config["holdout"]["namespace"],
                "seed_count": len(lineages),
                "seed_digest": canonical_digest(
                    {"derived_seeds": [item.seed for item in lineages]},
                    prefix="native-hypergraph-seeds",
                ),
                "tuning_performed": False,
            },
            "algorithm_path": {
                "receiver_program": "fit/apply_incremental_downstream_functional",
                "native_hyperedge": "mechanistic_strength.sender_resolved",
                "integrated_values_are_scenario_assigned": False,
            },
            "candidate_policy": config["candidate_universe"],
            "truth_endpoints": config["truth_endpoints"],
            "method_formulas": FORMULAS,
            "aggregation": config["aggregation"],
            "gate": gate,
            "generation_audit": generated.audit_summary(),
            "frozen_design": frozen_design_manifest(),
            "inputs": {
                "config": {
                    "filename": Path(config_path).name,
                    "sha256": sha256_file(config_path),
                },
                "truth": {
                    "filename": truth_path.name,
                    "sha256": sha256_file(truth_path),
                },
            },
            "outputs": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in written
            },
            "code": git_metadata(REPO_ROOT),
        }
        _write_json(staged / "manifest.json", manifest)
        _publish(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_benchmark(
        args.config,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": manifest["status"], "gate": manifest["gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
