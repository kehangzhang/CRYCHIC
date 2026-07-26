"""Evaluate condition-specific unordered cell-pair rankings with spatial DES."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.metrics.spatial_des import (
    ScoreType,
    SpatialDESSpec,
    TiePolicy,
    evaluate_spatial_des,
)

SCHEMA_VERSION = "crychic-spatial-des-evaluation-v2"
METHOD_COLUMNS = ("method", "method_version", "resource", "ranking_semantics")
ANALYSIS_UNITS = frozenset({"condition_level", "sample_id", "subject_id"})
COMPARISON_TRACKS = frozenset(
    {
        "legacy_unspecified",
        "native_cardinality",
        "continuous_strength",
        "fixed_k_100",
        "fixed_k_250",
        "fixed_k_500",
        "fixed_k_1000",
    }
)
SCSEQCOMMDIFF_MANIFEST_SCHEMAS = frozenset(
    {
        "crychic-scseqcommdiff-paper-benchmark-v1",
        "crychic-scseqcommdiff-paper-benchmark-v2",
    }
)
V7_REAL_E1_MANIFEST_SCHEMA = "crychic-suggest-next2-v7-real-e1-run-v1"
RankingStatistic = str
RANKING_STATISTICS = frozenset(
    {"raw_cardinality", "raw_strength", "average_rank"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_rankings(
    rankings: pd.DataFrame,
    expected_sets: pd.DataFrame,
    *,
    scenario: str,
    dataset_map: dict[str, str] | None = None,
    condition_map: dict[str, str] | None = None,
    expected_filters: dict[str, str] | None = None,
    score_type: ScoreType = "std",
    weight_exponent: float = 1.0,
    tie_policy: TiePolicy = "fgsea_native",
    exclude_self_pairs: bool = True,
    ranking_statistic: RankingStatistic = "raw_cardinality",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align one or more method rankings and evaluate a frozen spatial scenario."""
    if not scenario or scenario != scenario.strip():
        raise ValueError("scenario must be a canonical non-empty string")
    if ranking_statistic not in RANKING_STATISTICS:
        raise ValueError(f"unsupported ranking_statistic: {ranking_statistic!r}")
    required_rank = {
        "dataset",
        *METHOD_COLUMNS,
        "condition",
        "sender",
        "receiver",
        "ranked_strength",
        "status",
    }
    missing_rank = required_rank.difference(rankings.columns)
    if missing_rank or rankings.empty:
        raise ValueError(f"rankings are empty or missing: {sorted(missing_rank)}")
    required_expected = {
        "dataset",
        "scenario",
        "condition",
        "top_fraction",
        "sender",
        "receiver",
        "is_expected",
    }
    missing_expected = required_expected.difference(expected_sets.columns)
    if missing_expected or expected_sets.empty:
        raise ValueError(
            f"expected sets are empty or missing: {sorted(missing_expected)}"
        )
    expected = expected_sets.loc[expected_sets["scenario"].eq(scenario)].copy()
    for column, value in (expected_filters or {}).items():
        if column not in expected.columns:
            raise ValueError(f"expected filter column is missing: {column}")
        expected = expected.loc[expected[column].astype(str).eq(value)].copy()
    if expected.empty:
        raise ValueError(
            f"expected sets do not contain scenario {scenario!r} with filters "
            f"{expected_filters or {}}"
        )
    datasets = expected["dataset"].astype(str).unique()
    if len(datasets) != 1:
        raise ValueError("expected scenario must contain exactly one dataset")
    expected_dataset = str(datasets[0])
    ranked = rankings.copy(deep=True)
    observed_datasets = set(ranked["dataset"].astype(str))
    mapping = dataset_map or {}
    unused_sources = set(mapping).difference(observed_datasets)
    if unused_sources:
        raise ValueError(
            f"dataset map contains unused sources: {sorted(unused_sources)}"
        )
    invalid_targets = set(mapping.values()).difference({expected_dataset})
    if invalid_targets:
        raise ValueError(
            "dataset map targets disagree with the frozen truth dataset: "
            f"{sorted(invalid_targets)} != {expected_dataset!r}"
        )
    ranked["dataset"] = ranked["dataset"].astype(str).replace(mapping)
    aligned_datasets = set(ranked["dataset"].astype(str))
    if aligned_datasets != {expected_dataset}:
        raise ValueError(
            "ranking/expected datasets disagree; provide an explicit dataset map: "
            f"{sorted(observed_datasets)} != {[expected_dataset]}"
        )
    ranked["scenario"] = scenario
    if condition_map:
        ranked["condition"] = ranked["condition"].replace(condition_map)
    if not isinstance(exclude_self_pairs, bool):
        raise ValueError("exclude_self_pairs must be boolean")
    if exclude_self_pairs:
        ranked = ranked.loc[
            ranked["sender"].astype(str).ne(ranked["receiver"].astype(str))
        ].copy()
        expected = expected.loc[
            expected["sender"].astype(str).ne(expected["receiver"].astype(str))
        ].copy()
    if ranking_statistic == "average_rank":
        rank_groups = ["dataset", "scenario", *METHOD_COLUMNS, "condition"]
        ranked["ranked_strength"] = ranked.groupby(
            rank_groups, sort=False, observed=True
        )["ranked_strength"].transform(
            lambda values: pd.to_numeric(values, errors="coerce").rank(
                method="average", ascending=True
            )
        )
    observed_conditions = set(ranked["condition"].astype(str))
    expected_conditions = set(expected["condition"].astype(str))
    if observed_conditions != expected_conditions:
        raise ValueError(
            "ranking/expected conditions disagree: "
            f"{sorted(observed_conditions)} != {sorted(expected_conditions)}"
        )
    tables = evaluate_spatial_des(
        ranked,
        expected,
        SpatialDESSpec(
            method_columns=METHOD_COLUMNS,
            stratum_columns=("dataset", "scenario"),
            expected_member_column="is_expected",
            score_type=score_type,
            weight_exponent=weight_exponent,
            tie_policy=tie_policy,
            cell_pair_mode="unordered",
        ),
    )
    return tables.scores, tables.coverage


def _parse_condition_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("condition maps must use SOURCE=EXPECTED")
        source, expected = value.split("=", 1)
        if (
            not source
            or not expected
            or source != source.strip()
            or expected != expected.strip()
        ):
            raise ValueError("condition maps require canonical non-empty labels")
        if source in result:
            raise ValueError(f"duplicate condition map source: {source}")
        result[source] = expected
    return result


def _parse_dataset_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("dataset maps must use SOURCE=EXPECTED")
        source, expected = value.split("=", 1)
        if (
            not source
            or not expected
            or source != source.strip()
            or expected != expected.strip()
        ):
            raise ValueError("dataset maps require canonical non-empty labels")
        if source in result:
            raise ValueError(f"duplicate dataset map source: {source}")
        result[source] = expected
    return result


def _parse_filters(values: list[str], *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} filters must use COLUMN=VALUE")
        column, expected = value.split("=", 1)
        if (
            not column
            or not expected
            or column != column.strip()
            or expected != expected.strip()
        ):
            raise ValueError(
                f"{label} filters require canonical non-empty labels"
            )
        if column in result:
            raise ValueError(f"duplicate {label} filter column: {column}")
        result[column] = expected
    return result


def _parse_expected_filters(values: list[str]) -> dict[str, str]:
    return _parse_filters(values, label="expected")


def _parse_ranking_filters(values: list[str]) -> dict[str, str]:
    return _parse_filters(values, label="ranking")


def _filter_rankings(
    rankings: pd.DataFrame, filters: Mapping[str, str]
) -> pd.DataFrame:
    filtered = rankings
    for column, value in filters.items():
        if column not in filtered.columns:
            raise ValueError(f"ranking filter column is missing: {column}")
        filtered = filtered.loc[filtered[column].astype(str).eq(value)]
    if filtered.empty:
        raise ValueError("ranking filters removed every row")
    return filtered.copy()


def _resolved_comparison_track(
    value: str | None, *, ranking_filters: Mapping[str, str]
) -> str:
    if value is None:
        if ranking_filters:
            raise ValueError(
                "ranking filters require an explicit comparison track"
            )
        return "legacy_unspecified"
    if value not in COMPARISON_TRACKS:
        raise ValueError(f"unsupported comparison track: {value!r}")
    return value


def _resolved_analysis_unit(scenario: str, value: str | None) -> str | None:
    resolved = (
        "condition_level" if value is None and scenario == "condition_aware" else value
    )
    if resolved is not None and resolved not in ANALYSIS_UNITS:
        raise ValueError(f"unsupported analysis unit: {resolved!r}")
    return resolved


def _validate_run_binding(
    run_manifest_path: Path,
    *,
    scenario: str,
    analysis_unit: str | None,
    ranking_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    payload: object = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run manifest must be a JSON object")
    schema = payload.get("schema_version")
    if schema is not None and payload.get("status") != "complete":
        raise ValueError("schema-bound run manifest is not complete")
    analysis = payload.get("analysis_unit")
    binding_source = "analysis_unit.replicate_key"
    provenance: dict[str, object] = {}
    expected_ranking_sha256: object = None
    if isinstance(analysis, Mapping):
        replicate_key = analysis.get("replicate_key")
        subject_key = analysis.get("subject_key")
        primary_panel = analysis.get("primary_panel")
        outputs = payload.get("outputs")
        if isinstance(outputs, Mapping):
            record = outputs.get("condition_cell_pair_rankings.tsv")
            if isinstance(record, Mapping):
                expected_ranking_sha256 = record.get("sha256")
        inputs = payload.get("inputs")
        resource = payload.get("resource")
        if isinstance(inputs, Mapping):
            h5ad = inputs.get("h5ad")
            if schema == V7_REAL_E1_MANIFEST_SCHEMA:
                input_manifest = inputs.get("preparation_manifest")
                resource_input = inputs.get("lr_resource")
                resource_manifest = inputs.get("lr_resource_manifest")
            else:
                input_manifest = inputs.get("manifest")
                resource_input = None
                resource_manifest = inputs.get("resource_manifest")
            if isinstance(h5ad, Mapping):
                provenance["input_sha256"] = h5ad.get("sha256")
            if isinstance(input_manifest, Mapping):
                provenance["input_manifest_sha256"] = input_manifest.get("sha256")
            if isinstance(resource_input, Mapping):
                provenance["resource_sha256"] = resource_input.get("sha256")
            if isinstance(resource_manifest, Mapping):
                provenance["resource_manifest_sha256"] = resource_manifest.get(
                    "sha256"
                )
        if "resource_sha256" not in provenance and isinstance(resource, Mapping):
            provenance["resource_sha256"] = resource.get("sha256")
    elif schema in SCSEQCOMMDIFF_MANIFEST_SCHEMAS:
        if payload.get("status") != "complete":
            raise ValueError("scSeqCommDiff run manifest is not complete")
        preflight = payload.get("preflight")
        if not isinstance(preflight, Mapping):
            raise ValueError("scSeqCommDiff run manifest lacks preflight binding")
        input_record = preflight.get("input")
        resource_record = preflight.get("resource")
        if not isinstance(input_record, Mapping) or not isinstance(
            resource_record, Mapping
        ):
            raise ValueError("scSeqCommDiff run manifest lacks input/resource binding")
        replicate_key = input_record.get("sample_unit_key")
        subject_key = "subject_id" if replicate_key == "subject_id" else None
        primary_panel = None
        binding_source = "preflight.input.sample_unit_key"
        provenance = {
            "dataset_id": payload.get("dataset_id"),
            "input_sha256": input_record.get("sha256"),
            "input_manifest_sha256": input_record.get("manifest_sha256"),
            "resource_sha256": resource_record.get("sha256"),
            "resource_manifest_sha256": resource_record.get("manifest_sha256"),
        }
        outputs = payload.get("outputs")
        if isinstance(outputs, Mapping):
            record = outputs.get("condition_cell_pair_rankings.tsv")
            if isinstance(record, Mapping):
                expected_ranking_sha256 = record.get("sha256")
    elif schema == "crychic-sample-effect-des-ranking-v1":
        if payload.get("status") != "complete":
            raise ValueError("sample-effect ranking manifest is not complete")
        specification = payload.get("specification")
        if not isinstance(specification, Mapping):
            raise ValueError("sample-effect ranking manifest lacks its specification")
        replicate_key = specification.get("statistical_unit")
        subject_key = "subject_id" if replicate_key == "subject_id" else None
        primary_panel = False
        binding_source = "specification.statistical_unit"
        input_record = payload.get("input")
        if isinstance(input_record, Mapping):
            provenance["upstream_interactions_sha256"] = input_record.get("sha256")
        source_run = payload.get("source_run")
        if isinstance(source_run, Mapping):
            provenance.update(
                {
                    "source_run_manifest_sha256": source_run.get("sha256"),
                    "input_sha256": source_run.get("input_h5ad_sha256"),
                    "resource_sha256": source_run.get("resource_payload_sha256"),
                    "resource_manifest_sha256": source_run.get(
                        "resource_manifest_sha256"
                    ),
                }
            )
        provenance["panel_role"] = "continuous_common_sensitivity_only"
        outputs = payload.get("outputs")
        if isinstance(outputs, Mapping):
            record = outputs.get("rankings")
            if isinstance(record, Mapping):
                expected_ranking_sha256 = record.get("sha256")
    else:
        raise ValueError("run manifest lacks a supported analysis-unit binding")
    if replicate_key not in {"sample_id", "subject_id"}:
        raise ValueError("run manifest replicate_key is unsupported")
    resolved = _resolved_analysis_unit(scenario, analysis_unit)
    if scenario != "multi_sample" or resolved != replicate_key:
        raise ValueError(
            "run/evaluation analysis mismatch: "
            f"replicate_key={replicate_key!r} requires "
            f"scenario='multi_sample' and analysis_unit={replicate_key!r}"
        )
    if ranking_paths:
        observed_hashes = {_sha256(path) for path in ranking_paths}
        if (
            not isinstance(expected_ranking_sha256, str)
            or observed_hashes != {expected_ranking_sha256}
        ):
            raise ValueError("run manifest does not bind the supplied ranking payload")
    result: dict[str, Any] = {
        "replicate_key": replicate_key,
        "subject_key": subject_key,
        "primary_panel": primary_panel,
    }
    if schema is not None:
        result.update(
            {
                "manifest_schema_version": schema,
                "binding_source": binding_source,
                **provenance,
            }
        )
    return result


def _validate_expected_run_bindings(
    run_binding: Mapping[str, object] | None,
    *,
    input_sha256: str | None,
    resource_sha256: str | None,
    resource_manifest_sha256: str | None,
) -> dict[str, str | None]:
    expected = {
        "input_sha256": input_sha256,
        "resource_sha256": resource_sha256,
        "resource_manifest_sha256": resource_manifest_sha256,
    }
    for field, value in expected.items():
        if value is None:
            continue
        if run_binding is None or run_binding.get(field) != value:
            raise ValueError(f"run manifest {field} differs from the frozen panel")
    return expected


def run(
    ranking_paths: list[Path],
    expected_path: Path,
    output_dir: Path,
    *,
    scenario: str,
    analysis_unit: str | None,
    dataset_map: dict[str, str],
    condition_map: dict[str, str],
    expected_filters: dict[str, str],
    ranking_filters: dict[str, str] | None = None,
    comparison_track: str | None = None,
    score_type: ScoreType = "std",
    weight_exponent: float = 1.0,
    tie_policy: TiePolicy = "fgsea_native",
    exclude_self_pairs: bool = True,
    ranking_statistic: RankingStatistic = "raw_cardinality",
    overwrite: bool,
    run_manifest_path: Path | None = None,
    expected_input_sha256: str | None = None,
    expected_resource_sha256: str | None = None,
    expected_resource_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if not ranking_paths or any(not path.is_file() for path in ranking_paths):
        raise FileNotFoundError("one or more ranking paths are missing")
    if not expected_path.is_file():
        raise FileNotFoundError(expected_path)
    if run_manifest_path is not None and not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    selected_ranking_filters = ranking_filters or {}
    resolved_comparison_track = _resolved_comparison_track(
        comparison_track,
        ranking_filters=selected_ranking_filters,
    )
    resolved_analysis_unit = _resolved_analysis_unit(scenario, analysis_unit)
    run_binding = (
        _validate_run_binding(
            run_manifest_path,
            scenario=scenario,
            analysis_unit=analysis_unit,
            ranking_paths=ranking_paths,
        )
        if run_manifest_path is not None
        else None
    )
    expected_bindings = _validate_expected_run_bindings(
        run_binding,
        input_sha256=expected_input_sha256,
        resource_sha256=expected_resource_sha256,
        resource_manifest_sha256=expected_resource_manifest_sha256,
    )
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings = pd.concat(
        [pd.read_csv(path, sep="\t") for path in ranking_paths], ignore_index=True
    )
    rankings = _filter_rankings(rankings, selected_ranking_filters)
    expected = pd.read_csv(expected_path, sep="\t")
    scores, coverage = evaluate_rankings(
        rankings,
        expected,
        scenario=scenario,
        dataset_map=dataset_map,
        condition_map=condition_map,
        expected_filters=expected_filters,
        score_type=score_type,
        weight_exponent=weight_exponent,
        tie_policy=tie_policy,
        exclude_self_pairs=exclude_self_pairs,
        ranking_statistic=ranking_statistic,
    )
    score_path = output_dir / "spatial_des_scores.tsv"
    coverage_path = output_dir / "spatial_des_coverage.tsv"
    scores.to_csv(score_path, sep="\t", index=False)
    coverage.to_csv(coverage_path, sep="\t", index=False)
    observed = scores.loc[scores["status"].eq("observed")]
    summary = (
        observed.groupby([*METHOD_COLUMNS], observed=True, sort=True)["des"]
        .agg(["count", "median", "mean"])
        .reset_index()
    )
    summary_path = output_dir / "spatial_des_method_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    weighting = "unweighted" if weight_exponent == 0.0 else "weighted"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scenario": scenario,
        "analysis_unit": resolved_analysis_unit,
        "run_binding": run_binding,
        "expected_run_bindings": expected_bindings,
        "dataset_map": dataset_map,
        "condition_map": condition_map,
        "expected_filters": expected_filters,
        "ranking_filters": selected_ranking_filters,
        "comparison_track": resolved_comparison_track,
        "cell_pair_direction": "unordered_directions_collapsed",
        "score": (
            f"{weighting}_fgsea_{score_type}_running_sum_gseaParam="
            f"{weight_exponent:g};ranking_statistic={ranking_statistic}"
        ),
        "score_type": score_type,
        "weight_exponent": weight_exponent,
        "tie_policy": tie_policy,
        "self_pair_policy": (
            "excluded_from_rank_and_truth" if exclude_self_pairs else "included"
        ),
        "ranking_statistic": ranking_statistic,
        "inputs": {
            "rankings": [
                {
                    "path": str(path.resolve()),
                    "filename": path.name,
                    "sha256": _sha256(path),
                }
                for path in ranking_paths
            ],
            "expected_sets": {
                "path": str(expected_path.resolve()),
                "filename": expected_path.name,
                "sha256": _sha256(expected_path),
            },
            "run_manifest": (
                {
                    "path": str(run_manifest_path.resolve()),
                    "filename": run_manifest_path.name,
                    "sha256": _sha256(run_manifest_path),
                }
                if run_manifest_path is not None
                else None
            ),
        },
        "outputs": {
            "scores": {
                "filename": score_path.name,
                "rows": len(scores),
                "sha256": _sha256(score_path),
            },
            "coverage": {
                "filename": coverage_path.name,
                "rows": len(coverage),
                "sha256": _sha256(coverage_path),
            },
            "method_summary": {
                "filename": summary_path.name,
                "rows": len(summary),
                "sha256": _sha256(summary_path),
            },
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", action="append", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--analysis-unit", choices=sorted(ANALYSIS_UNITS))
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-resource-sha256")
    parser.add_argument("--expected-resource-manifest-sha256")
    parser.add_argument("--dataset-map", action="append", default=[])
    parser.add_argument("--condition-map", action="append", default=[])
    parser.add_argument("--expected-filter", action="append", default=[])
    parser.add_argument("--ranking-filter", action="append", default=[])
    parser.add_argument("--comparison-track", choices=sorted(COMPARISON_TRACKS))
    parser.add_argument(
        "--score-type",
        choices=("std", "pos", "abs"),
        default="std",
        help="fgsea scoreType; paper-compatible default is std",
    )
    parser.add_argument(
        "--weight-exponent",
        type=float,
        default=1.0,
        help="fgsea gseaParam; paper-compatible default is 1",
    )
    parser.add_argument(
        "--tie-policy",
        choices=("fgsea_native", "simultaneous"),
        default="fgsea_native",
        help="paper-compatible fgsea order or tie-safe simultaneous blocks",
    )
    parser.add_argument(
        "--include-self-pairs",
        action="store_true",
        help="retain diagonal cell pairs as a sensitivity analysis",
    )
    parser.add_argument(
        "--ranking-statistic",
        choices=sorted(RANKING_STATISTICS),
        default="raw_cardinality",
        help="statistic passed to fgsea; average_rank is a sensitivity arm",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.ranking,
        args.expected,
        args.output_dir,
        scenario=args.scenario,
        analysis_unit=args.analysis_unit,
        dataset_map=_parse_dataset_map(args.dataset_map),
        condition_map=_parse_condition_map(args.condition_map),
        expected_filters=_parse_expected_filters(args.expected_filter),
        ranking_filters=_parse_ranking_filters(args.ranking_filter),
        comparison_track=args.comparison_track,
        score_type=args.score_type,
        weight_exponent=args.weight_exponent,
        tie_policy=args.tie_policy,
        exclude_self_pairs=not args.include_self_pairs,
        ranking_statistic=args.ranking_statistic,
        overwrite=args.overwrite,
        run_manifest_path=args.run_manifest,
        expected_input_sha256=args.expected_input_sha256,
        expected_resource_sha256=args.expected_resource_sha256,
        expected_resource_manifest_sha256=args.expected_resource_manifest_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
