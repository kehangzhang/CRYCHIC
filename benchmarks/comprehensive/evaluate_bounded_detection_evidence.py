"""Evaluate the frozen soft-guard detection head on three-group holdouts."""

from __future__ import annotations

import argparse
import json
import math
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
from benchmarks.adapters.crychic.score_layers import (
    BOUNDED_DETECTION_DOWNSTREAM_WEIGHT,
    BOUNDED_DETECTION_MECHANISM_FLOOR,
    SENDER_RESPONSE_DETECTION_DOWNSTREAM_WEIGHT,
    bounded_detection_evidence_score,
    sender_response_detection_evidence_score,
)
from benchmarks.comprehensive.evaluate_component_crossover import (
    _effect_arm,
    _read_component_long,
    _source_truth,
)
from benchmarks.comprehensive.evaluate_three_group import _multigroup_metrics

SCHEMA_VERSION = "crychic-bounded-detection-evidence-evaluation-v1"
CONFIG_SCHEMA_VERSION = "crychic-bounded-detection-evidence-config-v1"
BASE_LAYER = "canonical_mechanistic"
CANDIDATE_LAYER = "bounded_detection_evidence_f075_w005"
UPPER_DIAGNOSTIC_LAYER = "sender_downstream_w010"
SUPPORTED_CANDIDATE_LAYERS = frozenset((CANDIDATE_LAYER, UPPER_DIAGNOSTIC_LAYER))
ENGINE = "native_raw_mean"
BASE_ARM = f"{BASE_LAYER}__{ENGINE}"
CANDIDATE_ARM = f"{CANDIDATE_LAYER}__{ENGINE}"


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


def _load_config(path: Path, *, role: str, source_manifest: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("bounded detection configuration schema is unsupported")
    candidate = config.get("candidate")
    boundary = config.get("claim_boundary")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("name") not in SUPPORTED_CANDIDATE_LAYERS
    ):
        raise ValueError("bounded detection candidate identity was altered")
    if candidate.get("formal_inference_allowed") is not False:
        raise ValueError("bounded detection candidate cannot emit formal inference")
    if not isinstance(boundary, Mapping) or not all(
        boundary.get(field) is True
        for field in (
            "detection_evidence_is_unsigned",
            "canonical_signed_effect_is_retained",
            "communication_strength_is_not_replaced",
            "p_and_q_values_are_not_emitted",
        )
    ):
        raise ValueError("bounded detection claim boundary was altered")
    if role not in {"development", "validation"}:
        raise ValueError("role must be development or validation")
    section = cast(
        Mapping[str, Any],
        config["selection" if role == "development" else "validation"],
    )
    expected_sha = section[f"{role}_source_manifest_sha256"]
    if sha256_file(source_manifest) != expected_sha:
        raise ValueError(f"{role} source evaluation manifest checksum differs")
    return config


def _score_layers(
    table: pd.DataFrame,
    *,
    mechanism_floor: float = BOUNDED_DETECTION_MECHANISM_FLOOR,
    downstream_weight: float = BOUNDED_DETECTION_DOWNSTREAM_WEIGHT,
    sender_response_downstream_weight: float = (
        SENDER_RESPONSE_DETECTION_DOWNSTREAM_WEIGHT
    ),
) -> dict[str, pd.Series]:
    required = {
        "availability",
        "prior_quality",
        "sender_component",
        "downstream",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"component ledger is missing: {sorted(missing)}")
    numeric = table.loc[:, sorted(required)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("bounded detection validation requires complete components")
    if ((numeric < 0.0) | (numeric > 1.0)).any().any():
        raise ValueError("bounded detection components must lie in [0, 1]")
    mechanism = table["availability"].astype(float) * table["prior_quality"].astype(
        float
    )
    sender = table["sender_component"].astype(float)
    downstream = table["downstream"].astype(float)
    return {
        BASE_LAYER: mechanism * sender,
        CANDIDATE_LAYER: bounded_detection_evidence_score(
            mechanism,
            sender,
            downstream,
            mechanism_floor=mechanism_floor,
            downstream_weight=downstream_weight,
        ),
        UPPER_DIAGNOSTIC_LAYER: sender_response_detection_evidence_score(
            sender,
            downstream,
            downstream_weight=sender_response_downstream_weight,
        ),
    }


def _method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    active = (
        metrics.loc[metrics["scenario"].astype(str).eq("active")]
        .groupby("method", observed=True, sort=True)
        .agg(
            active_seeds=("seed", "nunique"),
            omnibus_auprc=("omnibus_auprc", "mean"),
            omnibus_auroc=("omnibus_auroc", "mean"),
            localization_macro_auprc=("localization_macro_auprc", "mean"),
            positive_direction_ap=("positive_direction_ap", "mean"),
            negative_direction_ap=("negative_direction_ap", "mean"),
            direction_accuracy=("direction_accuracy_all_active", "mean"),
            minimum_event_coverage=("event_coverage", "min"),
            effect_zero_fraction=("effect_all_zero_fraction", "mean"),
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
            global_null_zero_fraction=("effect_all_zero_fraction", "mean"),
        )
        .reset_index()
    )
    return active.merge(null, on="method", validate="one_to_one").sort_values(
        ["omnibus_auprc", "omnibus_auroc"],
        ascending=False,
        kind="stable",
        ignore_index=True,
    )


def _paired_metric(
    metrics: pd.DataFrame,
    *,
    scenario: str,
    metric: str,
    direction: str,
    replicates: int,
    seed: int,
    baseline_arm: str = BASE_ARM,
    candidate_arm: str = CANDIDATE_ARM,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = metrics.loc[metrics["scenario"].astype(str).eq(scenario)]
    pivot = selected.pivot(index="seed", columns="method", values=metric)
    required = {baseline_arm, candidate_arm}
    missing = required.difference(pivot.columns)
    if missing:
        raise ValueError(f"paired validation arms are absent: {sorted(missing)}")
    pair = pivot.loc[:, [baseline_arm, candidate_arm]].dropna()
    raw = pair[candidate_arm].to_numpy(dtype=float) - pair[baseline_arm].to_numpy(
        dtype=float
    )
    if direction == "higher":
        oriented = raw
    elif direction == "lower":
        oriented = -raw
    else:
        raise ValueError("metric direction must be higher or lower")
    if len(oriented):
        rng = np.random.default_rng(seed)
        sampled = rng.choice(
            oriented, size=(replicates, len(oriented)), replace=True
        ).mean(axis=1)
        low, high = np.quantile(sampled, (0.025, 0.975))
    else:
        low = high = math.nan
    detail = pd.DataFrame(
        {
            "scenario": scenario,
            "metric": metric,
            "direction": direction,
            "seed": pair.index.astype(int),
            "baseline": pair[baseline_arm].to_numpy(dtype=float),
            "candidate": pair[candidate_arm].to_numpy(dtype=float),
            "raw_candidate_minus_baseline": raw,
            "oriented_improvement": oriented,
        }
    )
    return (
        {
            "scenario": scenario,
            "metric": metric,
            "direction": direction,
            "paired_seeds": len(oriented),
            "baseline_mean": float(pair[baseline_arm].mean()),
            "candidate_mean": float(pair[candidate_arm].mean()),
            "raw_candidate_minus_baseline": float(raw.mean()),
            "oriented_mean_improvement": float(oriented.mean()),
            "oriented_ci_low": float(low),
            "oriented_ci_high": float(high),
            "wins": int((oriented > 0.0).sum()),
            "ties": int((oriented == 0.0).sum()),
            "losses": int((oriented < 0.0).sum()),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
        },
        detail,
    )


def _validation(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    role: str,
    candidate_arm: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    policy = cast(Mapping[str, Any], config["validation"])
    replicates = int(policy["bootstrap_replicates"])
    bootstrap_seed = int(policy["bootstrap_seed"])
    specifications = (
        ("active", "omnibus_auprc", "higher"),
        ("active", "omnibus_auroc", "higher"),
        ("global_null", "effect_standard_deviation", "lower"),
    )
    records: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for index, (scenario, metric, direction) in enumerate(specifications):
        record, detail = _paired_metric(
            metrics,
            scenario=scenario,
            metric=metric,
            direction=direction,
            replicates=replicates,
            seed=bootstrap_seed + index,
            candidate_arm=candidate_arm,
        )
        records.append(record)
        details.append(detail)
    validation = pd.DataFrame.from_records(records)
    paired = pd.concat(details, ignore_index=True)
    candidate = summary.loc[summary["method"].astype(str).eq(candidate_arm)]
    if len(candidate) != 1:
        raise ValueError("candidate summary is absent or duplicated")
    enough_seeds = (
        validation["paired_seeds"].ge(int(policy["minimum_paired_seeds"])).all()
    )
    ci_pass = (
        validation["oriented_ci_low"]
        .gt(float(policy["required_oriented_ci_lower"]))
        .all()
    )
    coverage_pass = float(candidate["minimum_event_coverage"].iloc[0]) >= float(
        policy["required_event_coverage"]
    )
    gate = {
        "role": role,
        "primary_detection_head": candidate_arm,
        "signed_effect_head_retained": BASE_ARM,
        "enough_paired_seeds": bool(enough_seeds),
        "all_oriented_95pct_ci_lower_gt_zero": bool(ci_pass),
        "coverage_pass": bool(coverage_pass),
        "independent_validation": role == "validation",
        "status": (
            "ACCEPT"
            if role == "validation" and enough_seeds and ci_pass and coverage_pass
            else "DEVELOPMENT_ONLY"
            if role == "development"
            else "REJECT"
        ),
    }
    return validation, paired, gate


def _report(
    summary: pd.DataFrame, validation: pd.DataFrame, gate: Mapping[str, Any]
) -> str:
    lines = [
        "# Bounded detection evidence benchmark",
        "",
        f"Gate status: **{gate['status']}**",
        "",
        (
            "Detection evidence and signed effects are separate heads. The candidate "
            "does not replace communication strength and emits no formal p/q values."
        ),
        "",
        "| Method | AUPRC | AUROC | Localization AP | Direction | Null SD | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.method),
                    f"{row.omnibus_auprc:.4f}",
                    f"{row.omnibus_auroc:.4f}",
                    f"{row.localization_macro_auprc:.4f}",
                    f"{row.direction_accuracy:.4f}",
                    f"{row.global_null_effect_sd:.5f}",
                    f"{row.minimum_event_coverage:.3f}",
                )
            )
            + " |"
        )
    lines.extend(("", "## Paired validation", ""))
    for row in validation.itertuples(index=False):
        lines.append(
            f"- {row.metric}: oriented mean {row.oriented_mean_improvement:.4f}, "
            f"95% CI [{row.oriented_ci_low:.4f}, {row.oriented_ci_high:.4f}], "
            f"wins/ties/losses {row.wins}/{row.ties}/{row.losses}."
        )
    return "\n".join(lines) + "\n"


def evaluate(
    *,
    source_evaluation: Path,
    runs_dir: Path,
    config_path: Path,
    role: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate one checksum-bound development or validation campaign."""

    started = time.perf_counter()
    source_evaluation = source_evaluation.resolve()
    runs_dir = runs_dir.resolve()
    config_path = config_path.resolve()
    source_manifest_path = source_evaluation / "manifest.json"
    config = _load_config(config_path, role=role, source_manifest=source_manifest_path)
    candidate = cast(Mapping[str, Any], config["candidate"])
    truth, source_manifest = _source_truth(source_evaluation)
    cache: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    effects: list[pd.DataFrame] = []
    for (dataset_id, contrast), selected_truth in truth.groupby(
        ["dataset_id", "contrast"], observed=True, sort=True
    ):
        run_names = tuple(selected_truth["run_directory"].astype(str).unique())
        if len(run_names) != 1:
            raise ValueError("one dataset must bind one CRYCHIC run directory")
        run_name = run_names[0]
        if run_name not in cache:
            cache[run_name] = _read_component_long(runs_dir / run_name)
            provenance.append(cache[run_name][1])
        component_long = cache[run_name][0]
        target = str(selected_truth["target"].iloc[0])
        reference = str(selected_truth["reference"].iloc[0])
        view = component_long.loc[
            component_long["contrast_view"].astype(str).eq(f"global:'{target}'")
        ].copy()
        if view.empty:
            raise ValueError(f"score view is absent: {run_name}/{target}")
        layers = _score_layers(
            view,
            mechanism_floor=float(
                candidate.get("mechanism_floor", BOUNDED_DETECTION_MECHANISM_FLOOR)
            ),
            downstream_weight=float(
                candidate.get(
                    "bounded_downstream_weight", BOUNDED_DETECTION_DOWNSTREAM_WEIGHT
                )
                if candidate.get("name") == UPPER_DIAGNOSTIC_LAYER
                else candidate["downstream_weight"]
            ),
            sender_response_downstream_weight=float(
                candidate["downstream_weight"]
                if candidate.get("name") == UPPER_DIAGNOSTIC_LAYER
                else SENDER_RESPONSE_DETECTION_DOWNSTREAM_WEIGHT
            ),
        )
        truth_table = selected_truth.drop(columns="run_directory")
        for layer, score in layers.items():
            long_table = view.copy()
            long_table["score"] = score
            long_table["score_name"] = layer
            long_table["status"] = "ok"
            long_table["reason_code"] = "bounded_detection_evidence_observed"
            effect = _effect_arm(
                long_table,
                truth_table,
                layer=layer,
                engine=ENGINE,
                target=target,
                reference=reference,
                contrast=str(contrast),
                dataset_id=str(dataset_id),
            )
            effect["run_directory"] = run_name
            effects.append(effect)
    effect_table = pd.concat(effects, ignore_index=True)
    metrics, confusion = _multigroup_metrics(effect_table)
    summary = _method_summary(metrics)
    candidate_arm = f"{candidate['name']}__{ENGINE}"
    validation, paired, gate = _validation(
        metrics,
        summary,
        config=config,
        role=role,
        candidate_arm=candidate_arm,
    )

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
            "validation_summary.tsv": validation,
            "paired_seed_metrics.tsv": paired,
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False, lineterminator="\n")
        report = _report(summary, validation, gate)
        (staged / "REPORT.md").write_text(report, encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "role": role,
            "gate": gate,
            "candidate": dict(candidate),
            "claim_boundary": dict(cast(Mapping[str, Any], config["claim_boundary"])),
            "source_evaluation": {
                "directory": str(source_evaluation),
                "manifest_sha256": sha256_file(source_manifest_path),
                "schema_version": source_manifest["schema_version"],
            },
            "source_runs": sorted(provenance, key=lambda item: item["run_directory"]),
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "elapsed_seconds": time.perf_counter() - started,
            "outputs": {
                filename: {
                    "bytes": (staged / filename).stat().st_size,
                    "rows": len(table),
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
    parser.add_argument("--source-evaluation", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", choices=("development", "validation"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = evaluate(
        source_evaluation=args.source_evaluation,
        runs_dir=args.runs_dir,
        config_path=args.config,
        role=args.role,
        output_dir=args.output_dir,
    )
    print(json.dumps(json_safe(manifest), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
