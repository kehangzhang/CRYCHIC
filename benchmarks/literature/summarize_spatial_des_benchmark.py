"""Summarize checksum-compatible spatial DES evaluations without panel mixing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVALUATION_SCHEMA = "crychic-spatial-des-evaluation-v1"
SUMMARY_SCHEMA = "crychic-spatial-des-benchmark-summary-v1"
METHOD_COLUMNS = ("method", "method_version", "resource", "ranking_semantics")
TOP_FRACTIONS = (0.1, 0.2, 0.3, 0.4)
EXPECTED_DES_STRATA = 8
SUMMARY_FILENAME = "spatial_des_benchmark_summary.tsv"
MANIFEST_FILENAME = "manifest.json"
ANALYSIS_UNITS = frozenset({"subject_id", "sample_id", "condition_level"})


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationBundle:
    """Validated evaluation tables and their comparison-panel contract."""

    manifest_path: Path
    manifest_sha256: str
    dataset: str
    scenario: str
    expected_sha256: str
    expected_filters: dict[str, str]
    analysis_unit: str
    analysis_unit_source: str
    conditions: tuple[str, ...]
    fractions: tuple[float, ...]
    cell_pair_direction: str
    score_semantics: str
    scores_sha256: str
    coverage_sha256: str
    scores: pd.DataFrame
    coverage: pd.DataFrame


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _contract_id(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"spatial_des_panel_{digest[:24]}"


def _canonical_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _canonical_sha256(value: object, *, field: str) -> str:
    checksum = _canonical_string(value, field=field)
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return checksum


def _canonical_filters(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("evaluation expected_filters must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        column = _canonical_string(key, field="expected filter column")
        result[column] = _canonical_string(item, field=f"expected filter {column}")
    return dict(sorted(result.items()))


def _resolve_manifest(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    manifest = candidate / MANIFEST_FILENAME if candidate.is_dir() else candidate
    if not manifest.is_file():
        raise FileNotFoundError(f"evaluation manifest does not exist: {manifest}")
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid evaluation manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"evaluation manifest must contain an object: {path}")
    return value


def _validated_output(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    output_name: str,
) -> tuple[Path, str, int]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(output_name), dict):
        raise ValueError(f"evaluation manifest lacks output {output_name!r}")
    record = outputs[output_name]
    filename = _canonical_string(
        record.get("filename"), field=f"{output_name} filename"
    )
    if Path(filename).name != filename:
        raise ValueError(f"{output_name} filename must be a basename")
    checksum = _canonical_sha256(record.get("sha256"), field=f"{output_name} SHA256")
    rows = record.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError(f"{output_name} rows must be a positive integer")
    path = manifest_path.parent / filename
    if not path.is_file():
        raise FileNotFoundError(f"evaluation output does not exist: {path}")
    observed = _sha256_file(path)
    if observed != checksum:
        raise ValueError(
            f"evaluation {output_name} SHA256 mismatch: {observed} != {checksum}"
        )
    return path, checksum, rows


def _text_unit_candidates(value: str) -> set[str]:
    normalized = value.lower().replace("multi_sample", "").replace("multi-sample", "")
    normalized = re.sub(
        r"(?<![a-z0-9])subject[_-]equal(?![a-z0-9])", "", normalized
    )
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
    result: set[str] = set()
    if "subject" in tokens or "subjects" in tokens or "donor" in tokens:
        result.add("subject_id")
    if "sample" in tokens or "samples" in tokens or "section" in tokens:
        result.add("sample_id")
    return result


def _analysis_unit(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    ranking_semantics: Sequence[str],
    *,
    scenario: str,
) -> tuple[str, str]:
    explicit_value = manifest.get("analysis_unit")
    if explicit_value is None:
        comparison = manifest.get("comparison_contract")
        if isinstance(comparison, dict):
            explicit_value = comparison.get("analysis_unit")
    explicit: str | None = None
    if explicit_value is not None:
        explicit = _canonical_string(explicit_value, field="analysis_unit")
        if explicit not in ANALYSIS_UNITS:
            raise ValueError(f"unsupported analysis_unit: {explicit!r}")

    semantic_units: set[str] = set()
    for value in ranking_semantics:
        semantic_units.update(_text_unit_candidates(value))
    campaign_units = _text_unit_candidates(manifest_path.parent.parent.name)
    if explicit is not None:
        conflicting = semantic_units.difference({explicit})
        if conflicting:
            raise ValueError(
                f"explicit analysis_unit {explicit!r} conflicts with "
                f"{sorted(conflicting)}"
            )
        return explicit, "evaluation_manifest"
    if len(semantic_units) > 1:
        raise ValueError(
            f"evaluation analysis unit is ambiguous: {sorted(semantic_units)}"
        )
    if len(semantic_units) == 1:
        unit = next(iter(semantic_units))
        conflicting = campaign_units.difference({unit})
        if conflicting:
            raise ValueError(
                f"ranking semantics analysis unit {unit!r} conflicts with campaign "
                f"{sorted(conflicting)}"
            )
        return unit, "ranking_semantics"
    if len(campaign_units) > 1:
        raise ValueError(
            f"evaluation campaign analysis unit is ambiguous: {sorted(campaign_units)}"
        )
    if len(campaign_units) == 1:
        return next(iter(campaign_units)), "evaluation_campaign_path"
    if scenario == "multi_sample":
        raise ValueError(
            "multi_sample evaluation must declare or unambiguously encode "
            "subject_id versus sample_id"
        )
    return "condition_level", "scenario_default"


def _canonical_fraction(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("top_fractions must use 0.1, 0.2, 0.3, or 0.4")
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (str, int, float)):
        raise ValueError("top_fractions must use 0.1, 0.2, 0.3, or 0.4")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("top_fractions must use 0.1, 0.2, 0.3, or 0.4") from error
    for supported in TOP_FRACTIONS:
        if math.isclose(numeric, supported, rel_tol=0.0, abs_tol=1e-12):
            return supported
    raise ValueError("top_fractions must use 0.1, 0.2, 0.3, or 0.4")


def _validate_table_identity(
    table: pd.DataFrame,
    *,
    label: str,
    scenario: str,
) -> tuple[str, tuple[str, ...], tuple[float, ...]]:
    identity = ["dataset", "scenario", *METHOD_COLUMNS, "condition", "top_fraction"]
    missing = set(identity).difference(table.columns)
    if missing or table.empty:
        raise ValueError(f"{label} table is empty or missing: {sorted(missing)}")
    for column in identity[:-1]:
        if table[column].isna().any():
            raise ValueError(f"{label} identity {column} must not be missing")
        values = table[column].astype(str)
        if values.eq("").any() or values.str.strip().ne(values).any():
            raise ValueError(f"{label} identity {column} must be canonical")
        table[column] = values
    if set(table["scenario"]) != {scenario}:
        raise ValueError(f"{label} scenario disagrees with evaluation manifest")
    datasets = tuple(sorted(set(table["dataset"])))
    if len(datasets) != 1:
        raise ValueError(f"{label} must contain exactly one dataset")
    table["top_fraction"] = table["top_fraction"].map(_canonical_fraction)
    fractions = tuple(sorted(set(table["top_fraction"])))
    if fractions != TOP_FRACTIONS:
        raise ValueError(f"{label} must contain fractions {TOP_FRACTIONS}")
    conditions = tuple(sorted(set(table["condition"])))
    if len(conditions) != 2:
        raise ValueError(f"{label} must contain exactly two conditions")
    if table.duplicated(identity).any():
        raise ValueError(f"{label} contains duplicate method-condition-fraction rows")
    method_identity = list(METHOD_COLUMNS)
    counts = table.groupby(method_identity, sort=False, observed=True).size()
    if not counts.eq(EXPECTED_DES_STRATA).all():
        raise ValueError(f"every {label} method must contain exactly 8 strata")
    expected_strata = {
        (condition, fraction) for condition in conditions for fraction in fractions
    }
    for _, group in table.groupby(method_identity, sort=False, observed=True):
        observed = set(
            group.loc[:, ["condition", "top_fraction"]].itertuples(
                index=False, name=None
            )
        )
        if observed != expected_strata:
            raise ValueError(f"every {label} method must use identical 8 strata")
    return datasets[0], conditions, fractions


def _validate_scores(table: pd.DataFrame) -> None:
    required = {"status", "reason_code", "des", "metric"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"scores table is missing: {sorted(missing)}")
    if not table["metric"].eq("spatial_des").all():
        raise ValueError("scores metric must be spatial_des")
    if table["status"].isna().any():
        raise ValueError("scores status must not be missing")
    numeric = pd.to_numeric(table["des"], errors="coerce")
    supplied = table["des"].notna()
    if (supplied & numeric.isna()).any() or np.isinf(numeric.fillna(0.0)).any():
        raise ValueError("DES values must be finite numeric values or missing")
    observed = table["status"].eq("observed")
    if numeric.loc[observed].isna().any():
        raise ValueError("observed DES rows require finite values")
    if numeric.loc[~observed].notna().any():
        raise ValueError("non-observed DES rows require missing values")
    table["des"] = numeric.astype(float)


def _validate_coverage(table: pd.DataFrame) -> None:
    required = {
        "status",
        "expected_set_available",
        "expected_pair_coverage_fraction",
        "rank_eligible_fraction",
        "rank_rows_total",
        "rank_pairs_eligible",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"coverage table is missing: {sorted(missing)}")
    available = table["expected_set_available"]
    if available.isna().any():
        raise ValueError("expected_set_available must contain non-missing booleans")
    if pd.api.types.is_bool_dtype(available):
        table["expected_set_available"] = available.astype(bool)
    else:
        normalized = available.map(
            lambda value: {"True": True, "False": False}.get(str(value))
        )
        if normalized.isna().any():
            raise ValueError("expected_set_available must contain non-missing booleans")
        table["expected_set_available"] = normalized.astype(bool)
    rank_coverage = pd.to_numeric(table["rank_eligible_fraction"], errors="coerce")
    invalid_rank_coverage = (
        rank_coverage.isna().any()
        or np.isinf(rank_coverage).any()
        or not rank_coverage.between(0, 1).all()
    )
    if invalid_rank_coverage:
        raise ValueError("rank_eligible_fraction must be finite in [0, 1]")
    expected = pd.to_numeric(table["expected_pair_coverage_fraction"], errors="coerce")
    supplied = table["expected_pair_coverage_fraction"].notna()
    if (supplied & expected.isna()).any() or np.isinf(expected.fillna(0.0)).any():
        raise ValueError("expected_pair_coverage_fraction must be numeric or missing")
    if not expected.dropna().between(0, 1).all():
        raise ValueError("expected_pair_coverage_fraction must lie in [0, 1]")
    table["rank_eligible_fraction"] = rank_coverage.astype(float)
    table["expected_pair_coverage_fraction"] = expected.astype(float)
    for column in ("rank_rows_total", "rank_pairs_eligible"):
        numeric = pd.to_numeric(table[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any() or (numeric % 1 != 0).any():
            raise ValueError(f"coverage {column} must contain non-negative integers")
        table[column] = numeric.astype(np.int64)
    if (table["rank_pairs_eligible"] > table["rank_rows_total"]).any():
        raise ValueError("rank_pairs_eligible cannot exceed rank_rows_total")


def _load_evaluation(path: str | Path) -> EvaluationBundle:
    manifest_path = _resolve_manifest(path)
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema_version") != EVALUATION_SCHEMA:
        raise ValueError(f"unsupported evaluation schema: {manifest_path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"evaluation is not complete: {manifest_path}")
    scenario = _canonical_string(manifest.get("scenario"), field="scenario")
    expected_filters = _canonical_filters(manifest.get("expected_filters"))
    direction = _canonical_string(
        manifest.get("cell_pair_direction"), field="cell_pair_direction"
    )
    score_semantics = _canonical_string(manifest.get("score"), field="score")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(
        inputs.get("expected_sets"), dict
    ):
        raise ValueError("evaluation manifest lacks inputs.expected_sets")
    expected_sha256 = _canonical_sha256(
        inputs["expected_sets"].get("sha256"), field="expected set SHA256"
    )

    score_path, score_sha256, score_rows = _validated_output(
        manifest, manifest_path, output_name="scores"
    )
    coverage_path, coverage_sha256, coverage_rows = _validated_output(
        manifest, manifest_path, output_name="coverage"
    )
    scores = pd.read_csv(score_path, sep="\t", low_memory=False)
    coverage = pd.read_csv(coverage_path, sep="\t", low_memory=False)
    if len(scores) != score_rows or len(coverage) != coverage_rows:
        raise ValueError("evaluation output row count disagrees with manifest")
    dataset, conditions, fractions = _validate_table_identity(
        scores, label="scores", scenario=scenario
    )
    coverage_dataset, coverage_conditions, coverage_fractions = (
        _validate_table_identity(coverage, label="coverage", scenario=scenario)
    )
    if (coverage_dataset, coverage_conditions, coverage_fractions) != (
        dataset,
        conditions,
        fractions,
    ):
        raise ValueError(
            "scores and coverage dataset/condition/fraction strata disagree"
        )
    _validate_scores(scores)
    _validate_coverage(coverage)
    key = ["dataset", "scenario", *METHOD_COLUMNS, "condition", "top_fraction"]
    if set(scores.loc[:, key].itertuples(index=False, name=None)) != set(
        coverage.loc[:, key].itertuples(index=False, name=None)
    ):
        raise ValueError("scores and coverage method strata disagree")
    score_status = scores.set_index(key)["status"].sort_index()
    coverage_status = coverage.set_index(key)["status"].sort_index()
    if not score_status.equals(coverage_status):
        raise ValueError("scores and coverage statuses disagree")
    analysis_unit, unit_source = _analysis_unit(
        manifest,
        manifest_path,
        tuple(sorted(set(scores["ranking_semantics"]))),
        scenario=scenario,
    )
    return EvaluationBundle(
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        dataset=dataset,
        scenario=scenario,
        expected_sha256=expected_sha256,
        expected_filters=expected_filters,
        analysis_unit=analysis_unit,
        analysis_unit_source=unit_source,
        conditions=conditions,
        fractions=fractions,
        cell_pair_direction=direction,
        score_semantics=score_semantics,
        scores_sha256=score_sha256,
        coverage_sha256=coverage_sha256,
        scores=scores,
        coverage=coverage,
    )


def _panel_contract(bundle: EvaluationBundle) -> dict[str, Any]:
    return {
        "dataset": bundle.dataset,
        "scenario": bundle.scenario,
        "expected_sets_sha256": bundle.expected_sha256,
        "expected_filters": bundle.expected_filters,
        "expected_variant": bundle.expected_filters.get("variant", "not_applicable"),
        "analysis_unit": bundle.analysis_unit,
        "conditions": list(bundle.conditions),
        "top_fractions": list(bundle.fractions),
        "cell_pair_direction": bundle.cell_pair_direction,
        "score_semantics": bundle.score_semantics,
    }


def summarize_evaluations(
    evaluation_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, list[EvaluationBundle], dict[str, dict[str, Any]]]:
    """Validate evaluations and summarize methods within checksum-frozen panels."""

    if not evaluation_paths:
        raise ValueError("at least one evaluation is required")
    manifests = [_resolve_manifest(path) for path in evaluation_paths]
    if len(set(manifests)) != len(manifests):
        raise ValueError("evaluation manifests must not be duplicated")
    bundles = [_load_evaluation(path) for path in manifests]

    panels: dict[str, dict[str, Any]] = {}
    score_tables: list[pd.DataFrame] = []
    coverage_tables: list[pd.DataFrame] = []
    for evaluation_index, bundle in enumerate(bundles):
        contract = _panel_contract(bundle)
        panel_id = _contract_id(contract)
        previous = panels.setdefault(panel_id, contract)
        if previous != contract:
            raise RuntimeError("comparison panel digest collision")
        score_tables.append(
            bundle.scores.assign(
                comparison_panel_id=panel_id,
                _evaluation_index=evaluation_index,
            )
        )
        coverage_tables.append(
            bundle.coverage.assign(
                comparison_panel_id=panel_id,
                _evaluation_index=evaluation_index,
            )
        )

    scores = pd.concat(score_tables, ignore_index=True)
    coverage = pd.concat(coverage_tables, ignore_index=True)
    identity = [
        "comparison_panel_id",
        "dataset",
        "scenario",
        *METHOD_COLUMNS,
        "condition",
        "top_fraction",
    ]
    if scores.duplicated(identity).any() or coverage.duplicated(identity).any():
        raise ValueError(
            "the same method and stratum occurs in multiple evaluations within one "
            "comparison panel"
        )
    if set(scores.loc[:, identity].itertuples(index=False, name=None)) != set(
        coverage.loc[:, identity].itertuples(index=False, name=None)
    ):
        raise ValueError("combined scores and coverage identities disagree")
    joined = scores.merge(
        coverage.loc[
            :,
            [
                *identity,
                "expected_set_available",
                "expected_pair_coverage_fraction",
                "rank_eligible_fraction",
                "rank_rows_total",
                "rank_pairs_eligible",
            ],
        ],
        on=identity,
        how="left",
        validate="one_to_one",
    )
    group_columns = ["comparison_panel_id", "dataset", "scenario", *METHOD_COLUMNS]
    records: list[dict[str, object]] = []
    for key, group in joined.groupby(group_columns, sort=True, observed=True):
        identity_record = dict(zip(group_columns, key, strict=True))
        panel = panels[str(identity_record["comparison_panel_id"])]
        observed = group["status"].eq("observed") & group["des"].notna()
        des_values = group.loc[observed, "des"].astype(float)
        expected_coverage = (
            group["expected_pair_coverage_fraction"].dropna().astype(float)
        )
        rank_coverage = group["rank_eligible_fraction"].astype(float)
        all_expected_available = bool(
            group["expected_set_available"].fillna(False).astype(bool).all()
        )
        rank_eligible = (
            len(group) == EXPECTED_DES_STRATA
            and int(observed.sum()) == EXPECTED_DES_STRATA
            and all_expected_available
        )
        status_counts = {
            str(label): int(count)
            for label, count in group["status"].value_counts(sort=False).items()
        }
        records.append(
            {
                **identity_record,
                "analysis_unit": panel["analysis_unit"],
                "expected_sets_sha256": panel["expected_sets_sha256"],
                "expected_variant": panel["expected_variant"],
                "expected_filters_json": _canonical_json(panel["expected_filters"]),
                "conditions_json": json.dumps(
                    panel["conditions"], separators=(",", ":")
                ),
                "top_fractions_json": json.dumps(
                    panel["top_fractions"], separators=(",", ":")
                ),
                "des_strata_expected": EXPECTED_DES_STRATA,
                "des_strata_observed": int(observed.sum()),
                "des_median": (
                    float(des_values.median()) if not des_values.empty else math.nan
                ),
                "des_mean": (
                    float(des_values.mean()) if not des_values.empty else math.nan
                ),
                "rank_eligible": bool(rank_eligible),
                "median_rank": pd.NA,
                "median_rank_tie_size": pd.NA,
                "mean_rank": pd.NA,
                "mean_rank_tie_size": pd.NA,
                "rank_eligible_fraction_min": float(rank_coverage.min()),
                "rank_eligible_fraction_mean": float(rank_coverage.mean()),
                "expected_set_coverage_strata_observed": len(expected_coverage),
                "expected_set_coverage_min": (
                    float(expected_coverage.min())
                    if not expected_coverage.empty
                    else math.nan
                ),
                "expected_set_coverage_mean": (
                    float(expected_coverage.mean())
                    if not expected_coverage.empty
                    else math.nan
                ),
                "expected_set_available_all": all_expected_available,
                "rank_rows_total_min": int(group["rank_rows_total"].min()),
                "rank_rows_total_max": int(group["rank_rows_total"].max()),
                "rank_pairs_eligible_min": int(group["rank_pairs_eligible"].min()),
                "rank_pairs_eligible_max": int(group["rank_pairs_eligible"].max()),
                "des_status_counts_json": json.dumps(
                    status_counts, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    summary = pd.DataFrame.from_records(records)
    for _panel_id, indexes in summary.groupby(
        "comparison_panel_id", sort=True, observed=True
    ).groups.items():
        panel_indexes = pd.Index(indexes)
        eligible_indexes = panel_indexes[
            summary.loc[panel_indexes, "rank_eligible"].to_numpy(dtype=bool)
        ]
        if eligible_indexes.empty:
            continue
        medians = summary.loc[eligible_indexes, "des_median"].astype(float)
        ranks = medians.rank(method="min", ascending=False).astype(np.int64)
        summary.loc[eligible_indexes, "median_rank"] = ranks
        tie_sizes = medians.map(medians.value_counts()).astype(np.int64)
        summary.loc[eligible_indexes, "median_rank_tie_size"] = tie_sizes
        means = summary.loc[eligible_indexes, "des_mean"].astype(float)
        mean_ranks = means.rank(method="min", ascending=False).astype(np.int64)
        summary.loc[eligible_indexes, "mean_rank"] = mean_ranks
        mean_tie_sizes = means.map(means.value_counts()).astype(np.int64)
        summary.loc[eligible_indexes, "mean_rank_tie_size"] = mean_tie_sizes
    summary["median_rank"] = summary["median_rank"].astype("Int64")
    summary["median_rank_tie_size"] = summary["median_rank_tie_size"].astype("Int64")
    summary["mean_rank"] = summary["mean_rank"].astype("Int64")
    summary["mean_rank_tie_size"] = summary["mean_rank_tie_size"].astype("Int64")
    summary = summary.sort_values(
        [
            "comparison_panel_id",
            "rank_eligible",
            "median_rank",
            "des_median",
            "method",
            "method_version",
            "resource",
            "ranking_semantics",
        ],
        ascending=[True, False, True, False, True, True, True, True],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )
    return summary, bundles, panels


def _write_tsv_atomic(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        table.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(
    evaluation_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the unified summary and a checksum/protocol manifest."""

    destination = Path(output_dir).expanduser().resolve()
    summary_path = destination / SUMMARY_FILENAME
    manifest_path = destination / MANIFEST_FILENAME
    conflicts = [path.name for path in (summary_path, manifest_path) if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {conflicts}")
    summary, bundles, panels = summarize_evaluations(evaluation_paths)
    destination.mkdir(parents=True, exist_ok=True)
    _write_tsv_atomic(summary, summary_path)
    input_hashes = sorted(bundle.manifest_sha256 for bundle in bundles)
    aggregate = hashlib.sha256("\n".join(input_hashes).encode("ascii")).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "protocol": {
            "expected_des_strata_per_method": EXPECTED_DES_STRATA,
            "required_top_fractions": list(TOP_FRACTIONS),
            "required_conditions_per_panel": 2,
            "rank_eligibility": (
                "all 8 condition-by-fraction DES rows are observed and finite, and "
                "all expected sets are declared available"
            ),
            "ranking": (
                "within comparison_panel_id by DES median descending; tied medians "
                "share minimum rank; incomplete methods are not ranked"
            ),
            "secondary_ranking": (
                "within comparison_panel_id by DES mean descending; tied means "
                "share minimum rank; incomplete methods are not ranked"
            ),
            "panel_isolation": (
                "dataset, scenario, expected-set SHA256, expected filters/variant, "
                "analysis unit, condition/fraction strata, cell-pair direction, and "
                "score semantics"
            ),
            "coverage_reporting": (
                "rank_eligible_fraction and expected_pair_coverage_fraction are "
                "reported separately as minimum and arithmetic mean across 8 strata"
            ),
            "no_cross_panel_ranking": True,
        },
        "comparison_panels": [
            {"comparison_panel_id": panel_id, **contract}
            for panel_id, contract in sorted(panels.items())
        ],
        "inputs": {
            "evaluation_manifests": [
                {
                    "path_hint": (
                        f"{bundle.manifest_path.parent.parent.name}/"
                        f"{bundle.manifest_path.parent.name}/"
                        f"{bundle.manifest_path.name}"
                    ),
                    "sha256": bundle.manifest_sha256,
                    "scores_sha256": bundle.scores_sha256,
                    "coverage_sha256": bundle.coverage_sha256,
                    "dataset": bundle.dataset,
                    "scenario": bundle.scenario,
                    "expected_sets_sha256": bundle.expected_sha256,
                    "expected_filters": bundle.expected_filters,
                    "analysis_unit": bundle.analysis_unit,
                    "analysis_unit_source": bundle.analysis_unit_source,
                }
                for bundle in bundles
            ],
            "evaluation_manifest_aggregate_sha256": aggregate,
        },
        "output": {
            "filename": summary_path.name,
            "rows": len(summary),
            "bytes": summary_path.stat().st_size,
            "sha256": _sha256_file(summary_path),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
        },
    }
    _write_json_atomic(payload, manifest_path)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.evaluation, args.output_dir, overwrite=args.overwrite)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "ANALYSIS_UNITS",
    "EVALUATION_SCHEMA",
    "MANIFEST_FILENAME",
    "METHOD_COLUMNS",
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA",
    "TOP_FRACTIONS",
    "EvaluationBundle",
    "run",
    "summarize_evaluations",
]
