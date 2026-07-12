"""Evaluate opt-in downstream attribution support on unseen synthetic holdouts.

This is post-benchmark method development. Its outputs are intentionally kept
outside the frozen primary benchmark and are not finalizer inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json
from benchmarks.adapters.crychic.run_hcommon import run_hcommon_from_benchmark
from benchmarks.metrics.track_a_differential_truth import (
    PRIMARY_ESTIMAND,
    paired_edge_differences,
    prepare_estimands,
    select_crychic_run_id,
    summarize_differential_truth,
)
from benchmarks.simulation.export_multimethod_controls import export_controls
from crychic.attribution import AttributionSupportMethod
from crychic.core import stable_id

CANDIDATE_SCHEMA = "crychic-downstream-support-candidate-v1"
CANDIDATE_OUTPUT_SCHEMA = "crychic-downstream-support-candidate-output-v1"
DEVELOPMENT_SCOPE = "post_benchmark_candidate_not_primary_benchmark"
SUPPORTED_SCENARIOS = frozenset(
    {"active", "ligand_only", "target_only", "receptor_knockout"}
)


def _resolve(path: str, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return cast(dict[str, Any], value)


def _constant(table: pd.DataFrame, column: str) -> str:
    values = table[column].astype("string").dropna().astype(str).drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"candidate field {column!r} must be constant")
    return str(values.iloc[0])


def _validated_spec(path: Path, repo_root: Path) -> dict[str, Any]:
    spec = _json_object(path)
    if spec.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError("unsupported downstream-support candidate specification")
    if spec.get("development_scope") != DEVELOPMENT_SCOPE:
        raise ValueError("candidate specification must remain post-benchmark")
    scenarios_raw = spec.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("candidate specification requires scenarios")
    scenarios = tuple(str(value) for value in scenarios_raw)
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("candidate scenarios must be unique")
    if not {"active", "ligand_only", "target_only"}.issubset(scenarios):
        raise ValueError("candidate requires active, ligand_only and target_only")
    if not set(scenarios).issubset(SUPPORTED_SCENARIOS):
        raise ValueError("candidate contains an unsupported holdout scenario")
    methods_raw = spec.get("support_methods")
    if not isinstance(methods_raw, list) or len(methods_raw) != 2:
        raise ValueError("candidate requires explicit v1 and v2 support methods")
    methods = tuple(AttributionSupportMethod(str(value)) for value in methods_raw)
    if set(methods) != set(AttributionSupportMethod):
        raise ValueError("candidate support_methods must contain exactly v1 and v2")
    seed = spec.get("holdout_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("holdout_seed must be a non-negative integer")
    if seed == 20260712:
        raise ValueError("holdout seed must differ from the frozen benchmark seed")
    for field in (
        "database_root",
        "harmonized_resource",
        "harmonized_manifest",
        "target_prior_manifest",
    ):
        resolved = _resolve(str(spec[field]), repo_root)
        if not resolved.exists():
            raise FileNotFoundError(f"candidate resource is missing: {resolved}")
        spec[f"resolved_{field}"] = str(resolved)
    return spec


def build_candidate_benchmark_config(
    spec: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    *,
    input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build the exact adapter config with explicit v1/v2 workflow choices."""

    raw_records = input_manifest.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("holdout input manifest lacks records")
    by_scenario = {
        str(record["scenario"]): cast(Mapping[str, Any], record)
        for record in raw_records
        if isinstance(record, Mapping)
    }
    workflow_base = dict(cast(Mapping[str, Any], spec["workflow"]))
    datasets: dict[str, Any] = {}
    for scenario in cast(list[str], spec["scenarios"]):
        record = by_scenario.get(scenario)
        if record is None:
            raise ValueError(f"holdout input manifest lacks {scenario}")
        input_path = (input_root / str(record["path"])).resolve()
        observed_sha = sha256_file(input_path)
        if observed_sha != str(record["sha256"]):
            raise ValueError(f"holdout input checksum mismatch for {scenario}")
        for raw_method in cast(list[str], spec["support_methods"]):
            method = AttributionSupportMethod(raw_method)
            label = (
                "v1"
                if method is AttributionSupportMethod.RELATIVE_COEFFICIENT_V1
                else "v2"
            )
            dataset_key = f"holdout_{scenario}_{label}"
            datasets[dataset_key] = {
                "dataset_id": f"holdout_{scenario}",
                "input": str(input_path),
                "input_sha256": observed_sha,
                "output_name": dataset_key,
                "input_mode": "counts",
                "lr_resource": "synthetic_hcommon",
                "target_prior": "nichenet_human",
                "benchmark_scope": DEVELOPMENT_SCOPE,
                "config": {
                    "context_keys": ["condition"],
                    "counts_layer": "counts",
                    "sample_key": "sample_id",
                    "subject_key": "subject_id",
                    "cell_type_key": "cell_type",
                    "species": "human",
                    "gene_namespace": "hgnc_symbol",
                    "design": "~ condition",
                    "communication_modes": [
                        str(cast(Mapping[str, Any], spec["analysis"])[
                            "communication_mode"
                        ])
                    ],
                    "random_seed": int(record["scenario_seed"]),
                },
                "workflow": {
                    **workflow_base,
                    "downstream_attribution_support_method": method.value,
                },
            }
    return {
        "schema_version": "canonical-v0.1",
        "database_root": str(spec["resolved_database_root"]),
        "output_root": str(output_root.resolve()),
        "resources": {
            "synthetic_hcommon": {
                "adapter": "harmonized_fixture",
                "manifest": str(spec["resolved_harmonized_manifest"]),
                "species": "human",
            },
            "nichenet_human": {
                "manifest": str(spec["resolved_target_prior_manifest"]),
                "release": str(spec["target_prior_release"]),
            },
        },
        "datasets": datasets,
    }


def _truth_table(
    score_table: pd.DataFrame,
    *,
    scenario: str,
    contrast: str,
    known_edge: Mapping[str, str],
) -> pd.DataFrame:
    edge_keys = ("sender", "receiver", "interaction_id", "ligand", "receptor")
    truth = score_table.loc[:, list(edge_keys)].drop_duplicates(ignore_index=True)
    known = pd.Series(True, index=truth.index)
    for key in edge_keys:
        known &= truth[key].astype(str).eq(str(known_edge[key]))
    if int(known.sum()) != 1:
        raise ValueError("candidate known edge must match exactly one frozen edge")
    truth["is_positive"] = (known & (scenario == "active")).astype(int)
    truth["truth_scope"] = "simulation"
    truth["dataset"] = _constant(score_table, "dataset")
    truth["contrast"] = contrast
    truth["universe_id"] = _constant(score_table, "universe_id")
    return truth


def _context(value: object, key: str) -> str:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict) or key not in parsed:
        raise ValueError(f"context_json lacks {key!r}")
    return str(parsed[key])


def _known_downstream_effect(
    run_dir: Path,
    *,
    scoring_functional_id: str,
    known_edge: Mapping[str, str],
    context_key: str,
    reference: str,
    target: str,
) -> dict[str, object]:
    table = pd.read_parquet(run_dir / "result" / "sample_scores.parquet")
    edge_id = stable_id(
        "communication_edge",
        {
            "interaction_id": known_edge["interaction_id"],
            "receiver": known_edge["receiver"],
            "sender": known_edge["sender"],
        },
    )
    selected = table.loc[
        table["scoring_functional_id"].astype(str).eq(scoring_functional_id)
        & table["edge_id"].astype(str).eq(edge_id)
        & table["mode"].astype(str).eq("state")
        & table["status"].astype(str).eq("ok")
    ].copy()
    selected["context"] = [
        _context(value, context_key) for value in selected["context_json"]
    ]
    values = (
        selected.groupby(["subject_id", "context"], observed=True)["downstream"]
        .mean()
        .unstack("context")
    )
    if reference not in values or target not in values:
        return {
            "known_downstream_reference_mean": math.nan,
            "known_downstream_target_mean": math.nan,
            "known_downstream_paired_effect": math.nan,
            "known_downstream_positive_direction_fraction": math.nan,
            "known_downstream_n_pairs": 0,
        }
    paired = values[[reference, target]].dropna()
    difference = paired[target] - paired[reference]
    return {
        "known_downstream_reference_mean": (
            float(paired[reference].mean()) if len(paired) else math.nan
        ),
        "known_downstream_target_mean": (
            float(paired[target].mean()) if len(paired) else math.nan
        ),
        "known_downstream_paired_effect": (
            float(difference.mean()) if len(difference) else math.nan
        ),
        "known_downstream_positive_direction_fraction": (
            float((difference > 0).mean()) if len(difference) else math.nan
        ),
        "known_downstream_n_pairs": len(difference),
    }


def evaluate_candidate_decision(
    summary: pd.DataFrame, decision_rule: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the pre-specified active-retention and negative-reduction rule."""

    required = {
        "scenario",
        "support_method",
        "known_edge_effect",
        "known_edge_effect_rank",
        "known_edge_positive_direction_fraction",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"candidate summary is missing columns: {sorted(missing)}")
    v1_method = AttributionSupportMethod.RELATIVE_COEFFICIENT_V1.value
    v2_method = AttributionSupportMethod.GATED_RESPONSE_NORM_V2.value
    rows: list[dict[str, object]] = []
    for scenario, group in summary.groupby("scenario", observed=True, sort=False):
        v1 = group.loc[group["support_method"].eq(v1_method)]
        v2 = group.loc[group["support_method"].eq(v2_method)]
        if len(v1) != 1 or len(v2) != 1:
            raise ValueError(
                f"candidate scenario {scenario} requires one v1 and one v2"
            )
        left = v1.iloc[0]
        right = v2.iloc[0]
        v1_effect = float(left["known_edge_effect"])
        v2_effect = float(right["known_edge_effect"])
        rows.append(
            {
                "scenario": str(scenario),
                "v1_known_edge_effect": v1_effect,
                "v2_known_edge_effect": v2_effect,
                "v2_minus_v1_known_edge_effect": v2_effect - v1_effect,
                "v1_known_edge_rank": float(left["known_edge_effect_rank"]),
                "v2_known_edge_rank": float(right["known_edge_effect_rank"]),
                "v1_positive_direction_fraction": float(
                    left["known_edge_positive_direction_fraction"]
                ),
                "v2_positive_direction_fraction": float(
                    right["known_edge_positive_direction_fraction"]
                ),
                "v2_strictly_reduces_effect": (
                    v2_effect
                    < v1_effect - float(decision_rule.get("reduction_epsilon", 0.0))
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    active = comparison.loc[comparison["scenario"].eq("active")]
    if len(active) != 1:
        raise ValueError("candidate decision requires one active comparison")
    active_row = active.iloc[0]
    active_retained = (
        (
            not bool(decision_rule.get("require_active_positive_effect", True))
            or float(active_row["v2_known_edge_effect"]) > 0
        )
        and float(active_row["v2_positive_direction_fraction"])
        >= float(decision_rule["active_min_positive_direction_fraction"])
        and float(active_row["v2_known_edge_rank"])
        <= float(decision_rule["active_max_effect_rank"])
    )
    reduction_scenarios = tuple(
        map(str, decision_rule["require_strict_effect_reduction"])
    )
    reduction_checks: dict[str, bool] = {}
    for scenario in reduction_scenarios:
        matched = comparison.loc[comparison["scenario"].eq(scenario)]
        if len(matched) != 1:
            raise ValueError(f"candidate decision lacks {scenario}")
        reduction_checks[scenario] = bool(
            matched["v2_strictly_reduces_effect"].iloc[0]
        )
    recommend = active_retained and all(reduction_checks.values())
    decision: dict[str, object] = {
        "development_scope": DEVELOPMENT_SCOPE,
        "primary_benchmark_eligible": False,
        "active_retained": active_retained,
        "negative_control_effect_reduced": reduction_checks,
        "recommend_future_default_change": recommend,
        "current_default_remains": (
            AttributionSupportMethod.RELATIVE_COEFFICIENT_V1.value
        ),
        "candidate_method": AttributionSupportMethod.GATED_RESPONSE_NORM_V2.value,
        "decision_semantics": (
            "recommend_only_if_active_direction_and_top_k_are_retained_and_all_"
            "predeclared_negative_control_effects_strictly_decrease"
        ),
    }
    return comparison, decision


def _markdown_table(table: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if value is None or value is pd.NA:
            return ""
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            return "" if math.isnan(numeric) else f"{numeric:.6g}"
        return str(value).replace("|", "\\|")

    headers = list(map(str, table.columns))
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _candidate_report(
    summary: pd.DataFrame, comparison: pd.DataFrame, decision: Mapping[str, object]
) -> str:
    summary_columns = [
        "scenario",
        "support_version",
        "known_edge_effect",
        "known_edge_effect_rank",
        "known_edge_positive_direction_fraction",
        "known_downstream_paired_effect",
        "auroc",
        "average_precision",
    ]
    lines = [
        "# Downstream attribution-support v2 holdout audit",
        "",
        "This is post-benchmark method development on newly seeded synthetic ",
        "holdouts. It is not part of the frozen primary benchmark, is not a ",
        "finalizer input, and does not revise any published benchmark metric.",
        "",
        "## CRYCHIC-only holdout results",
        "",
        _markdown_table(summary.loc[:, summary_columns]),
        "",
        "## Paired v1/v2 comparison",
        "",
        _markdown_table(comparison),
        "",
        "## Pre-specified decision",
        "",
        f"- Active retained: `{decision['active_retained']}`",
        f"- Negative controls reduced: `{decision['negative_control_effect_reduced']}`",
        "- Recommend a future default change: "
        f"`{decision['recommend_future_default_change']}`",
        "- Current default remains: "
        f"`{decision['current_default_remains']}`",
        "",
        "Passing this gate qualifies v2 for further default-change evaluation; ",
        "it is not an immediate default switch. In particular, ligand-only can ",
        "remain a positive top-ranked edge after attenuation, and receptor-knockout ",
        "magnitude must be monitored in additional holdouts.",
        "",
        "The candidate changes only downstream attribution support. It emits no ",
        "p-values, q-values, FDR, or formal type-I-error claims.",
        "",
    ]
    return "\n".join(lines)


def run_candidate_holdout(
    specification_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate unseen holdouts, run CRYCHIC v1/v2, and persist an audit."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    spec_path = Path(specification_path).resolve()
    spec = _validated_spec(spec_path, root)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"candidate output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    input_root = output / "holdout_inputs"
    input_manifest = export_controls(
        input_root,
        n_subjects=int(spec["n_subjects"]),
        mean_cells_per_sample=int(spec["mean_cells_per_sample"]),
        seed=int(spec["holdout_seed"]),
        overwrite=False,
    )
    benchmark_config = build_candidate_benchmark_config(
        spec,
        input_manifest,
        input_root=input_root,
        output_root=output / "runs",
    )
    benchmark_path = output / "candidate_benchmark_config.json"
    write_json(benchmark_path, benchmark_config)

    analysis = cast(Mapping[str, Any], spec["analysis"])
    known_edge = {
        str(key): str(value)
        for key, value in cast(Mapping[str, Any], spec["known_edge"]).items()
    }
    summary_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    for scenario in cast(list[str], spec["scenarios"]):
        for raw_method in cast(list[str], spec["support_methods"]):
            method = AttributionSupportMethod(raw_method)
            version = (
                1
                if method is AttributionSupportMethod.RELATIVE_COEFFICIENT_V1
                else 2
            )
            dataset_key = f"holdout_{scenario}_v{version}"
            run_dir = output / "runs" / scenario / f"v{version}"
            adapter_manifest = run_hcommon_from_benchmark(
                benchmark_path,
                dataset_key,
                run_dir,
                harmonized_resource=str(spec["resolved_harmonized_resource"]),
                harmonized_manifest=str(spec["resolved_harmonized_manifest"]),
                database_root=str(spec["resolved_database_root"]),
                communication_mode=str(analysis["communication_mode"]),
                blas_threads=int(spec["threads"]),
                overwrite=False,
                repo_root=root,
            )
            configured_method = cast(Mapping[str, Any], adapter_manifest["parameters"])[
                "workflow"
            ]["downstream_attribution_support_method"]
            if str(configured_method) != method.value:
                raise RuntimeError("adapter manifest lost the support-method opt-in")
            result_manifest_path = run_dir / "result" / "run_manifest.json"
            result_manifest = _json_object(result_manifest_path)
            attribution_parameters = cast(
                Mapping[str, Any], result_manifest["workflow_parameters"]
            )["attribution"]
            if version == 2:
                provenance = cast(Mapping[str, Any], attribution_parameters)[
                    "downstream_support"
                ]
                if cast(Mapping[str, Any], provenance).get("method") != method.value:
                    raise RuntimeError("result manifest lost v2 support provenance")

            table = pd.read_parquet(run_dir / "interactions_long.parquet")
            selected_run_id = select_crychic_run_id(
                adapter_manifest,
                contrast_candidate=str(
                    analysis["crychic_primary_contrast_candidate"]
                ),
            )
            table = table.loc[table["run_id"].astype(str).eq(selected_run_id)].copy()
            score_table, estimands = prepare_estimands(
                table,
                context_key=str(analysis["context_key"]),
                contrast=str(analysis["contrast"]),
            )
            primary = next(
                item for item in estimands if item.estimand == PRIMARY_ESTIMAND
            )
            truth = _truth_table(
                score_table,
                scenario=scenario,
                contrast=str(analysis["contrast"]),
                known_edge=known_edge,
            )
            edges = paired_edge_differences(
                primary.subject_context,
                truth,
                reference=str(analysis["reference"]),
                target=str(analysis["target"]),
                min_pairs=int(analysis["min_pairs"]),
                estimand=PRIMARY_ESTIMAND,
            )
            metrics = summarize_differential_truth(
                edges,
                top_k=int(analysis["top_k"]),
                known_edge=known_edge,
            )
            view = next(
                value
                for value in cast(
                    list[Mapping[str, Any]],
                    cast(Mapping[str, Any], adapter_manifest["source_result"])[
                        "score_views"
                    ],
                )
                if str(value["run_id"]) == selected_run_id
            )
            downstream = _known_downstream_effect(
                run_dir,
                scoring_functional_id=str(view["scoring_functional_id"]),
                known_edge=known_edge,
                context_key=str(analysis["context_key"]),
                reference=str(analysis["reference"]),
                target=str(analysis["target"]),
            )
            summary_rows.append(
                {
                    "development_scope": DEVELOPMENT_SCOPE,
                    "primary_benchmark_eligible": False,
                    "scenario": scenario,
                    "dataset": _constant(score_table, "dataset"),
                    "support_method": method.value,
                    "support_version": version,
                    "support_formula": method.to_dict()["formula"],
                    "holdout_seed": int(spec["holdout_seed"]),
                    "scenario_seed": int(
                        next(
                            record["scenario_seed"]
                            for record in cast(
                                list[dict[str, Any]], input_manifest["records"]
                            )
                            if record["scenario"] == scenario
                        )
                    ),
                    "selected_run_id": selected_run_id,
                    **metrics,
                    **downstream,
                }
            )
            run_records.append(
                {
                    "scenario": scenario,
                    "support_method": method.value,
                    "adapter_manifest": str(run_dir / "manifest.json"),
                    "adapter_manifest_sha256": sha256_file(run_dir / "manifest.json"),
                    "long_table_sha256": sha256_file(
                        run_dir / "interactions_long.parquet"
                    ),
                    "result_manifest": str(result_manifest_path),
                    "result_manifest_sha256": sha256_file(result_manifest_path),
                }
            )

    summary = pd.DataFrame(summary_rows)
    comparison, decision = evaluate_candidate_decision(
        summary, cast(Mapping[str, Any], spec["decision_rule"])
    )
    summary_path = output / "candidate_summary.tsv"
    comparison_path = output / "candidate_comparison.tsv"
    decision_path = output / "candidate_decision.json"
    report_path = output / "candidate_report.md"
    summary.to_csv(summary_path, sep="\t", index=False)
    comparison.to_csv(comparison_path, sep="\t", index=False)
    write_json(decision_path, cast(dict[str, Any], decision))
    report_path.write_text(
        _candidate_report(summary, comparison, decision), encoding="utf-8"
    )
    outputs = {
        "summary": summary_path,
        "comparison": comparison_path,
        "decision": decision_path,
        "report": report_path,
        "benchmark_config": benchmark_path,
    }
    manifest: dict[str, Any] = {
        "schema_version": CANDIDATE_OUTPUT_SCHEMA,
        "development_scope": DEVELOPMENT_SCOPE,
        "primary_benchmark_eligible": False,
        "finalizer_integration": False,
        "specification": str(spec_path),
        "specification_sha256": sha256_file(spec_path),
        "holdout_input_manifest": {
            "path": str(input_root / "manifest.json"),
            "sha256": sha256_file(input_root / "manifest.json"),
        },
        "support_methods": [
            AttributionSupportMethod(value).to_dict()
            for value in cast(list[str], spec["support_methods"])
        ],
        "runs": run_records,
        "decision": decision,
        "outputs": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": (
                    len(summary)
                    if name == "summary"
                    else len(comparison)
                    if name == "comparison"
                    else None
                ),
            }
            for name, path in outputs.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run_candidate_holdout(
        args.specification,
        args.output_dir,
        overwrite=args.overwrite,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
