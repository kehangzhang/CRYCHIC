"""Reconstruct spatial DES expected sets for the Lerma-Martin MS cohort.

The public UCSC Cell Browser ``meta.tsv`` files contain one row per Visium
spot and nine cell-type proportions.  This module computes within-section
Pearson co-localization for canonical unordered cell-type pairs, then builds
the two expected-set rankings described by Cesaro et al.:

* condition-aware: rank the absolute difference between condition means;
* multi-sample: rank two-sided Mann-Whitney U p-values, then absolute effects.

No multiple-testing correction is performed or reported. Figure 3 defaults to
the declared tissue/sample unit and floor-sized expected sets. Subject-collapsed
units and ceil-sized sets remain explicit sensitivity options.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DATASET_ID = "lerma_martin_ms_ctrl_vs_chronic_active"
CONTROL = "control"
CHRONIC_ACTIVE = "chronic_active"
CONDITIONS = (CONTROL, CHRONIC_ACTIVE)
CELL_TYPES = ("AS", "BC", "EC", "MG", "NEU", "OL", "OPC", "SC", "TC")
TOP_FRACTIONS = (0.1, 0.2, 0.3, 0.4)
TopCountRule = Literal["floor", "ceil"]
SCHEMA_VERSION = "crychic-ms-spatial-des-truth-v1"


@dataclass(frozen=True, slots=True)
class MSSpatialSample:
    """Frozen public-section roster and biological-unit mapping."""

    file_stem: str
    sample_id: str
    subject_id: str
    condition: str


MS_SPATIAL_SAMPLES = (
    MSSpatialSample("co37", "CO37", "CO37 P5B3", CONTROL),
    MSSpatialSample("co40", "CO40", "PDCO40 A1B2", CONTROL),
    MSSpatialSample("co41", "CO41", "CO41 A1C4", CONTROL),
    MSSpatialSample("co74", "CO74", "CO74 A1A2", CONTROL),
    MSSpatialSample("co85", "CO85", "CO85 A3C2", CONTROL),
    MSSpatialSample("ms197D", "MS197D", "MS197 P2D3", CHRONIC_ACTIVE),
    MSSpatialSample("ms229", "MS229", "MS229 P2C2", CHRONIC_ACTIVE),
    MSSpatialSample("ms377I", "MS377I", "MS377 A2D4", CHRONIC_ACTIVE),
    MSSpatialSample("ms377N", "MS377N", "MS377 A2D2", CHRONIC_ACTIVE),
    MSSpatialSample("ms377T", "MS377T", "MS377 A2D4", CHRONIC_ACTIVE),
    MSSpatialSample("ms411", "MS411", "MS411 A2A2", CHRONIC_ACTIVE),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MSSpatialTruthTables:
    """Compact reconstructed spatial evidence and expected memberships."""

    sample_correlations: pd.DataFrame
    pair_rankings: pd.DataFrame
    expected_sets: pd.DataFrame

    def __post_init__(self) -> None:
        for field in ("sample_correlations", "pair_rankings", "expected_sets"):
            value = getattr(self, field)
            if not isinstance(value, pd.DataFrame) or value.empty:
                raise ValueError(f"{field} must be a non-empty DataFrame")
            object.__setattr__(self, field, value.copy(deep=True))


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _canonical_fractions(values: Sequence[float]) -> tuple[float, ...]:
    fractions: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("top fractions must be finite numbers in (0, 1]")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0 < numeric <= 1:
            raise ValueError("top fractions must be finite numbers in (0, 1]")
        fractions.append(numeric)
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("top fractions must be non-empty and unique")
    return tuple(sorted(fractions))


def compute_unordered_pearson_correlations(
    proportions: pd.DataFrame,
    *,
    include_self: bool,
    min_complete_spots: int = 3,
) -> pd.DataFrame:
    """Compute pairwise-complete Pearson correlations for one spatial sample.

    Column names define the cell types.  Pair endpoints are lexicalized into
    ``sender <= receiver`` solely to provide the shared DES column names; they
    are not communication directions.
    """

    if not isinstance(proportions, pd.DataFrame) or proportions.empty:
        raise ValueError("proportions must be a non-empty DataFrame")
    if not isinstance(include_self, bool):
        raise ValueError("include_self must be boolean")
    if (
        isinstance(min_complete_spots, bool)
        or not isinstance(min_complete_spots, int)
        or min_complete_spots < 3
    ):
        raise ValueError("min_complete_spots must be an integer >= 3")
    names = [str(value) for value in proportions.columns]
    if any(not value or value != value.strip() for value in names):
        raise ValueError("cell-type columns must be canonical non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("cell-type columns must be unique")
    numeric = proportions.copy(deep=True)
    numeric.columns = names
    for column in names:
        supplied = numeric[column].notna()
        converted = pd.to_numeric(numeric[column], errors="coerce")
        if (supplied & converted.isna()).any():
            raise ValueError(f"cell-type proportion {column!r} must be numeric")
        numeric[column] = converted.astype(float)

    ordered_names = sorted(names)
    pair_iterator = (
        combinations_with_replacement(ordered_names, 2)
        if include_self
        else combinations(ordered_names, 2)
    )
    records: list[dict[str, object]] = []
    for sender, receiver in pair_iterator:
        left = numeric[sender].to_numpy(dtype=float, copy=False)
        right = numeric[receiver].to_numpy(dtype=float, copy=False)
        complete = np.isfinite(left) & np.isfinite(right)
        n_complete = int(complete.sum())
        correlation = math.nan
        if n_complete < min_complete_spots:
            status = "not_estimable"
            reason = "insufficient_complete_spots"
        else:
            left_complete = left[complete]
            right_complete = right[complete]
            if np.ptp(left_complete) == 0 or np.ptp(right_complete) == 0:
                status = "not_estimable"
                reason = "constant_proportion"
            else:
                correlation = (
                    1.0
                    if sender == receiver
                    else float(np.corrcoef(left_complete, right_complete)[0, 1])
                )
                if not math.isfinite(correlation):
                    correlation = math.nan
                    status = "not_estimable"
                    reason = "nonfinite_pearson_result"
                else:
                    correlation = min(1.0, max(-1.0, correlation))
                    status = "observed"
                    reason = ""
        records.append(
            {
                "sender": sender,
                "receiver": receiver,
                "pearson_r": correlation,
                "n_complete_spots": n_complete,
                "status": status,
                "reason_code": reason,
            }
        )
    return pd.DataFrame.from_records(records)


def _validate_design(sample_design: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "subject_id", "condition"}
    missing = required.difference(sample_design.columns)
    if missing:
        raise ValueError(f"sample_design is missing columns: {sorted(missing)}")
    design = sample_design.loc[:, sorted(required)].copy(deep=True)
    if design.empty or design.isna().any().any():
        raise ValueError("sample_design identifiers must be complete")
    for column in required:
        design[column] = design[column].astype(str)
        if (
            design[column].eq("").any()
            or design[column].str.strip().ne(design[column]).any()
        ):
            raise ValueError(f"sample_design {column} must be canonical")
    if design["sample_id"].duplicated().any():
        raise ValueError("sample_design sample_id values must be unique")
    invalid = set(design["condition"]).difference(CONDITIONS)
    if invalid or set(design["condition"]) != set(CONDITIONS):
        raise ValueError(
            f"sample_design must contain exactly conditions {list(CONDITIONS)}"
        )
    subject_conditions = design.groupby("subject_id", sort=False)["condition"].nunique()
    if (subject_conditions > 1).any():
        raise ValueError("a subject_id cannot occur in both conditions")
    return design.sort_values("sample_id", kind="stable", ignore_index=True)


def _validate_sample_correlations(
    sample_correlations: pd.DataFrame, sample_design: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "sample_id",
        "sender",
        "receiver",
        "pearson_r",
        "status",
        "reason_code",
    }
    missing = required.difference(sample_correlations.columns)
    if missing:
        raise ValueError(f"sample_correlations is missing columns: {sorted(missing)}")
    table = sample_correlations.copy(deep=True)
    samples = set(table["sample_id"].astype(str))
    expected_samples = set(sample_design["sample_id"])
    if samples != expected_samples:
        raise ValueError(
            "sample_correlations samples disagree with sample_design: "
            f"missing={sorted(expected_samples - samples)}, "
            f"unexpected={sorted(samples - expected_samples)}"
        )
    for column in ("sample_id", "sender", "receiver", "status", "reason_code"):
        if table[column].isna().any():
            raise ValueError(f"sample_correlations {column} must not be missing")
        table[column] = table[column].astype(str)
    if table.duplicated(["sample_id", "sender", "receiver"]).any():
        raise ValueError("sample_correlations contains duplicate sample-pair rows")
    if (table["sender"] > table["receiver"]).any():
        raise ValueError("sample correlations must use canonical unordered pairs")
    pair_sets = {
        frozenset(
            group.loc[:, ["sender", "receiver"]].itertuples(index=False, name=None)
        )
        for _, group in table.groupby("sample_id", sort=False, observed=True)
    }
    if len(pair_sets) != 1:
        raise ValueError("all samples must contain the same cell-type pair universe")
    numeric = pd.to_numeric(table["pearson_r"], errors="coerce")
    observed = table["status"].eq("observed")
    if (
        numeric.loc[observed].isna().any()
        or (numeric.loc[observed].abs() > 1 + 1e-12).any()
    ):
        raise ValueError("observed Pearson correlations must be finite in [-1, 1]")
    if numeric.loc[~observed].notna().any():
        raise ValueError("non-observed Pearson correlation rows must be missing")
    table["pearson_r"] = numeric
    return table


def _condition_summary(table: pd.DataFrame, *, unit_column: str) -> pd.DataFrame:
    unit = (
        table.loc[table["status"].eq("observed")]
        .groupby(
            [unit_column, "condition", "sender", "receiver"],
            sort=True,
            observed=True,
            as_index=False,
        )["pearson_r"]
        .mean()
    )
    grouped = (
        unit.groupby(["condition", "sender", "receiver"], sort=True, observed=True)[
            "pearson_r"
        ]
        .agg(["mean", "count"])
        .reset_index()
    )
    means = grouped.pivot(
        index=["sender", "receiver"], columns="condition", values="mean"
    )
    counts = grouped.pivot(
        index=["sender", "receiver"], columns="condition", values="count"
    )
    result = means.reindex(columns=CONDITIONS).rename(
        columns={
            CONTROL: "mean_control",
            CHRONIC_ACTIVE: "mean_chronic_active",
        }
    )
    count_table = (
        counts.reindex(columns=CONDITIONS)
        .fillna(0)
        .astype(int)
        .rename(
            columns={
                CONTROL: "n_control_units",
                CHRONIC_ACTIVE: "n_chronic_active_units",
            }
        )
    )
    return cast(pd.DataFrame, result.join(count_table).reset_index())


def _rank_condition_aware(merged: pd.DataFrame) -> pd.DataFrame:
    result = _condition_summary(merged, unit_column="sample_id")
    result["scenario"] = "condition_aware"
    result["u_statistic"] = math.nan
    result["p_value"] = math.nan
    result["p_value_semantics"] = "not_applicable_condition_mean_ranking"
    estimable = result[["mean_control", "mean_chronic_active"]].notna().all(axis=1)
    result["effect_chronic_active_minus_control"] = (
        result["mean_chronic_active"] - result["mean_control"]
    )
    result["abs_effect"] = result["effect_chronic_active_minus_control"].abs()
    result["status"] = np.where(estimable, "observed", "not_estimable")
    result["reason_code"] = np.where(
        estimable, "", "condition_has_no_observed_sample_correlations"
    )
    return result


def _rank_multi_sample(
    merged: pd.DataFrame,
    *,
    multi_sample_unit: Literal["subject_id", "sample_id"],
) -> pd.DataFrame:
    unit = (
        merged.loc[merged["status"].eq("observed")]
        .groupby(
            [multi_sample_unit, "condition", "sender", "receiver"],
            sort=True,
            observed=True,
            as_index=False,
        )["pearson_r"]
        .mean()
    )
    records: list[dict[str, object]] = []
    all_pairs = merged.loc[:, ["sender", "receiver"]].drop_duplicates()
    for sender, receiver in all_pairs.sort_values(
        ["sender", "receiver"], kind="stable"
    ).itertuples(index=False, name=None):
        pair = unit.loc[unit["sender"].eq(sender) & unit["receiver"].eq(receiver)]
        control = pair.loc[pair["condition"].eq(CONTROL), "pearson_r"].to_numpy(
            dtype=float
        )
        chronic = pair.loc[pair["condition"].eq(CHRONIC_ACTIVE), "pearson_r"].to_numpy(
            dtype=float
        )
        mean_control = float(np.mean(control)) if control.size else math.nan
        mean_chronic = float(np.mean(chronic)) if chronic.size else math.nan
        effect = mean_chronic - mean_control
        if control.size < 2 or chronic.size < 2:
            statistic = p_value = math.nan
            status = "not_estimable"
            reason = "fewer_than_two_observed_units_in_a_condition"
        else:
            test = mannwhitneyu(
                chronic,
                control,
                alternative="two-sided",
                method="auto",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
            status = "observed"
            reason = ""
        records.append(
            {
                "sender": sender,
                "receiver": receiver,
                "mean_control": mean_control,
                "mean_chronic_active": mean_chronic,
                "n_control_units": int(control.size),
                "n_chronic_active_units": int(chronic.size),
                "scenario": "multi_sample",
                "u_statistic": statistic,
                "p_value": p_value,
                "p_value_semantics": (
                    "raw_two_sided_mann_whitney_u_scipy_method_auto;"
                    "no_multiple_testing_correction"
                ),
                "effect_chronic_active_minus_control": effect,
                "abs_effect": abs(effect),
                "status": status,
                "reason_code": reason,
            }
        )
    return pd.DataFrame.from_records(records)


def _assign_stable_ranks(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    effect = result["effect_chronic_active_minus_control"]
    result["direction_condition"] = np.select(
        [effect > 0, effect < 0],
        [CHRONIC_ACTIVE, CONTROL],
        default="tied",
    )
    eligible = result["status"].eq("observed") & effect.ne(0)
    if result["scenario"].nunique() != 1:
        raise ValueError("each ranking table must contain exactly one scenario")
    scenario = str(result["scenario"].iloc[0])
    if scenario == "condition_aware":
        sort_columns = ["abs_effect", "sender", "receiver"]
        ascending = [False, True, True]
    elif scenario == "multi_sample":
        eligible &= result["p_value"].notna()
        sort_columns = ["p_value", "abs_effect", "sender", "receiver"]
        ascending = [True, False, True, True]
    else:
        raise ValueError(f"unsupported ranking scenario: {scenario}")
    ranked = result.loc[eligible].sort_values(
        sort_columns, ascending=ascending, kind="stable"
    )
    rank_map = pd.Series(
        np.arange(1, len(ranked) + 1, dtype=np.int64), index=ranked.index
    )
    result["spatial_rank"] = rank_map.reindex(result.index).astype("Int64")
    result["ranking_eligible"] = eligible
    result["dataset"] = DATASET_ID
    columns = [
        "dataset",
        "scenario",
        "sender",
        "receiver",
        "mean_control",
        "mean_chronic_active",
        "effect_chronic_active_minus_control",
        "abs_effect",
        "u_statistic",
        "p_value",
        "p_value_semantics",
        "n_control_units",
        "n_chronic_active_units",
        "direction_condition",
        "spatial_rank",
        "ranking_eligible",
        "status",
        "reason_code",
    ]
    return result.loc[:, columns].sort_values(
        ["scenario", "spatial_rank", "sender", "receiver"],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )


def _expected_memberships(
    rankings: pd.DataFrame,
    fractions: tuple[float, ...],
    *,
    top_count_rule: TopCountRule,
) -> pd.DataFrame:
    if top_count_rule not in {"floor", "ceil"}:
        raise ValueError("top_count_rule must be 'floor' or 'ceil'")
    records: list[dict[str, object]] = []
    for scenario, scenario_table in rankings.groupby(
        "scenario", sort=True, observed=True
    ):
        rankable = scenario_table.loc[scenario_table["ranking_eligible"]]
        n_rankable = len(rankable)
        for condition in CONDITIONS:
            for fraction in fractions:
                top_count = (
                    (math.floor if top_count_rule == "floor" else math.ceil)(
                        fraction * n_rankable
                    )
                    if n_rankable
                    else 0
                )
                for row in scenario_table.itertuples(index=False):
                    rank = row.spatial_rank
                    selected = (
                        not pd.isna(rank)
                        and int(cast(float, rank)) <= top_count
                        and row.direction_condition == condition
                    )
                    records.append(
                        {
                            "dataset": DATASET_ID,
                            "scenario": scenario,
                            "condition": condition,
                            "top_fraction": fraction,
                            "sender": row.sender,
                            "receiver": row.receiver,
                            "is_expected": bool(selected),
                            "spatial_rank": rank,
                            "rankable_pairs": n_rankable,
                            "top_count": top_count,
                            "top_count_rule": top_count_rule,
                            "cell_pair_direction": "unordered_canonical",
                        }
                    )
    result = pd.DataFrame.from_records(records)
    result["spatial_rank"] = result["spatial_rank"].astype("Int64")
    return result.sort_values(
        ["scenario", "condition", "top_fraction", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )


def build_ms_spatial_des_truth(
    sample_correlations: pd.DataFrame,
    sample_design: pd.DataFrame,
    *,
    top_fractions: Sequence[float] = TOP_FRACTIONS,
    multi_sample_unit: Literal["subject_id", "sample_id"] = "sample_id",
    top_count_rule: TopCountRule = "floor",
) -> MSSpatialTruthTables:
    """Build condition-aware and multi-sample spatial expected sets."""

    if multi_sample_unit not in {"subject_id", "sample_id"}:
        raise ValueError("multi_sample_unit must be 'subject_id' or 'sample_id'")
    fractions = _canonical_fractions(top_fractions)
    design = _validate_design(sample_design)
    correlations = _validate_sample_correlations(sample_correlations, design)
    merged = correlations.merge(
        design, on="sample_id", how="left", validate="many_to_one"
    )
    condition_aware = _assign_stable_ranks(_rank_condition_aware(merged))
    multi_sample = _assign_stable_ranks(
        _rank_multi_sample(merged, multi_sample_unit=multi_sample_unit)
    )
    rankings = pd.concat(
        [condition_aware, multi_sample], ignore_index=True
    ).sort_values(
        ["scenario", "spatial_rank", "sender", "receiver"],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )
    expected = _expected_memberships(
        rankings, fractions, top_count_rule=top_count_rule
    )
    correlations = merged.assign(dataset=DATASET_ID).loc[
        :,
        [
            "dataset",
            "sample_id",
            "subject_id",
            "condition",
            "sender",
            "receiver",
            "pearson_r",
            *(["n_complete_spots"] if "n_complete_spots" in merged.columns else []),
            "status",
            "reason_code",
        ],
    ]
    correlations = correlations.sort_values(
        ["condition", "sample_id", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )
    return MSSpatialTruthTables(
        sample_correlations=correlations,
        pair_rankings=rankings,
        expected_sets=expected,
    )


def _validate_public_sample(
    input_root: Path, sample: MSSpatialSample
) -> tuple[pd.DataFrame, dict[str, object]]:
    meta_path = input_root / f"{sample.file_stem}.meta.tsv"
    dataset_path = input_root / f"{sample.file_stem}.dataset.json"
    missing = [
        str(path.name) for path in (meta_path, dataset_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"MS spatial sample {sample.sample_id} is missing files: {missing}"
        )
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid dataset manifest: {dataset_path.name}") from error
    expected_name = f"ms-subcortical-lesions/visium-{sample.file_stem}"
    if dataset.get("name") != expected_name:
        raise ValueError(
            f"{dataset_path.name} dataset name {dataset.get('name')!r} != "
            f"{expected_name!r}"
        )
    out_meta = dataset.get("fileVersions", {}).get("outMeta")
    if not isinstance(out_meta, dict):
        raise ValueError(f"{dataset_path.name} lacks fileVersions.outMeta")
    expected_size = out_meta.get("size")
    if expected_size != meta_path.stat().st_size:
        raise ValueError(
            f"{meta_path.name} size {meta_path.stat().st_size} != manifest "
            f"{expected_size}"
        )
    md5_prefix = str(out_meta.get("md5", ""))
    observed_md5 = _md5_file(meta_path)
    if not md5_prefix or not observed_md5.startswith(md5_prefix):
        raise ValueError(
            f"{meta_path.name} MD5 {observed_md5} does not match manifest prefix "
            f"{md5_prefix!r}"
        )
    table = pd.read_csv(meta_path, sep="\t", low_memory=False)
    if dataset.get("sampleCount") != len(table):
        raise ValueError(
            f"{meta_path.name} rows {len(table)} != manifest sampleCount "
            f"{dataset.get('sampleCount')}"
        )
    manifest_columns = [
        field.get("name")
        for field in dataset.get("metaFields", [])
        if isinstance(field, dict)
    ]
    if manifest_columns != list(table.columns):
        raise ValueError(f"{meta_path.name} columns disagree with dataset manifest")
    missing_columns = {"cellId", *CELL_TYPES}.difference(table.columns)
    if missing_columns:
        raise ValueError(
            f"{meta_path.name} is missing required columns: {sorted(missing_columns)}"
        )
    if table["cellId"].isna().any() or table["cellId"].duplicated().any():
        raise ValueError(
            f"{meta_path.name} spot identifiers must be complete and unique"
        )
    proportions = table.loc[:, CELL_TYPES].apply(pd.to_numeric, errors="coerce")
    if (
        proportions.isna().any().any()
        or not np.isfinite(proportions.to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"{meta_path.name} proportions must be finite and complete")
    if (proportions < 0).any().any() or (proportions > 1).any().any():
        raise ValueError(f"{meta_path.name} proportions must lie in [0, 1]")
    row_sums = proportions.sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{meta_path.name} cell-type proportions must sum to one")
    provenance = {
        "sample_id": sample.sample_id,
        "subject_id": sample.subject_id,
        "condition": sample.condition,
        "meta_file": meta_path.name,
        "meta_bytes": meta_path.stat().st_size,
        "meta_rows": len(table),
        "meta_sha256": _sha256_file(meta_path),
        "meta_md5": observed_md5,
        "cell_browser_md5_prefix": md5_prefix,
        "dataset_file": dataset_path.name,
        "dataset_sha256": _sha256_file(dataset_path),
        "cell_browser_dataset_name": expected_name,
        "cell_browser_sample_count": int(dataset["sampleCount"]),
    }
    return proportions, provenance


def _load_public_inputs(
    input_root: Path,
    *,
    include_self: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    expected_stems = {sample.file_stem for sample in MS_SPATIAL_SAMPLES}
    observed_meta = {
        path.name.removesuffix(".meta.tsv") for path in input_root.glob("*.meta.tsv")
    }
    observed_dataset = {
        path.name.removesuffix(".dataset.json")
        for path in input_root.glob("*.dataset.json")
    }
    if observed_meta != expected_stems or observed_dataset != expected_stems:
        raise ValueError(
            "input root file roster disagrees with the frozen 11-section cohort: "
            f"meta_missing={sorted(expected_stems - observed_meta)}, "
            f"meta_unexpected={sorted(observed_meta - expected_stems)}, "
            f"dataset_missing={sorted(expected_stems - observed_dataset)}, "
            f"dataset_unexpected={sorted(observed_dataset - expected_stems)}"
        )
    correlation_tables: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    design_records: list[dict[str, str]] = []
    for sample in MS_SPATIAL_SAMPLES:
        proportions, record = _validate_public_sample(input_root, sample)
        correlations = compute_unordered_pearson_correlations(
            proportions, include_self=include_self
        ).assign(sample_id=sample.sample_id)
        correlation_tables.append(correlations)
        provenance.append(record)
        design_records.append(
            {
                "sample_id": sample.sample_id,
                "subject_id": sample.subject_id,
                "condition": sample.condition,
            }
        )
    return (
        pd.concat(correlation_tables, ignore_index=True),
        pd.DataFrame.from_records(design_records),
        provenance,
    )


def _diagnostics(tables: MSSpatialTruthTables) -> dict[str, Any]:
    rankings = tables.pair_rankings
    scenario_records: dict[str, object] = {}
    for scenario, group in rankings.groupby("scenario", sort=True, observed=True):
        eligible = group.loc[group["ranking_eligible"]]
        primary = "abs_effect" if scenario == "condition_aware" else "p_value"
        tied_rows = int(eligible[primary].duplicated(keep=False).sum())
        scenario_records[str(scenario)] = {
            "pair_rows": len(group),
            "rankable_pairs": len(eligible),
            "not_estimable_pairs": int(group["status"].ne("observed").sum()),
            "zero_direction_pairs": int(group["direction_condition"].eq("tied").sum()),
            "rows_in_primary_rank_ties": tied_rows,
        }
    return {
        "sample_correlation_rows": len(tables.sample_correlations),
        "non_estimable_sample_correlations": int(
            tables.sample_correlations["status"].ne("observed").sum()
        ),
        "pair_rankings": scenario_records,
        "expected_members_by_scenario_condition_fraction": [
            {
                "scenario": str(key[0]),
                "condition": str(key[1]),
                "top_fraction": float(cast(float, key[2])),
                "members": int(group["is_expected"].sum()),
                "universe": len(group),
            }
            for key, group in tables.expected_sets.groupby(
                ["scenario", "condition", "top_fraction"],
                sort=True,
                observed=True,
            )
        ],
    }


def _write_tsv_atomic(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        table.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_ms_spatial_des_truth(
    input_root: str | Path,
    output_root: str | Path,
    *,
    include_self: bool,
    multi_sample_unit: Literal["subject_id", "sample_id"] = "sample_id",
    top_fractions: Sequence[float] = TOP_FRACTIONS,
    top_count_rule: TopCountRule = "floor",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate public files, reconstruct expected sets, and write compact outputs."""

    source = Path(input_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"MS spatial input root does not exist: {source}")
    fractions = _canonical_fractions(top_fractions)
    correlations, design, source_records = _load_public_inputs(
        source, include_self=include_self
    )
    tables = build_ms_spatial_des_truth(
        correlations,
        design,
        top_fractions=fractions,
        multi_sample_unit=multi_sample_unit,
        top_count_rule=top_count_rule,
    )
    filenames = {
        "sample_correlations": "ms_spatial_sample_correlations.tsv",
        "pair_rankings": "ms_spatial_pair_rankings.tsv",
        "expected_sets": "ms_spatial_expected_sets.tsv",
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run" if dry_run else "complete",
        "dataset_id": DATASET_ID,
        "source": {
            "study": "Lerma-Martin et al., Nature Neuroscience (2024)",
            "study_doi": "10.1038/s41593-024-01796-z",
            "benchmark_doi": "10.1093/nargab/lqaf084",
            "cell_browser_collection": "ms-subcortical-lesions",
            "files": source_records,
        },
        "design": {
            "conditions": list(CONDITIONS),
            "samples_per_condition": {
                str(key): int(value)
                for key, value in design["condition"].value_counts(sort=False).items()
            },
            "subjects_per_condition": {
                str(key): int(value)
                for key, value in design.groupby("condition", sort=False)["subject_id"]
                .nunique()
                .items()
            },
            "sample_and_subject_are_distinct": True,
            "multi_sample_unit": multi_sample_unit,
            "repeated_sample_aggregation": (
                "arithmetic mean of observed section Pearson correlations within "
                "subject and condition"
                if multi_sample_unit == "subject_id"
                else "none; Figure 3 tissue/sample unit"
            ),
        },
        "protocol": {
            "cell_types": list(CELL_TYPES),
            "cell_pair_direction": "unordered_canonical_sender_le_receiver",
            "include_self_pairs": include_self,
            "pearson_missing_policy": (
                "pairwise finite spots; >=3; constant proportions are not_estimable; "
                "never zero-imputed"
            ),
            "condition_aware": (
                "arithmetic sample mean per condition; rank absolute chronic_active "
                "minus control difference"
            ),
            "multi_sample": (
                "two-sided scipy.stats.mannwhitneyu(method='auto') on declared "
                "analysis units; rank raw p_value ascending then abs effect descending"
            ),
            "p_value_adjustment": "none",
            "q_values_generated": False,
            "top_fractions": list(fractions),
            "top_count_rule": (
                f"{top_count_rule}(top_fraction * rankable_pairs)"
            ),
            "direction_rule": (
                "positive chronic_active-minus-control effect => chronic_active; "
                "negative => control; exact zero => tied and never expected"
            ),
            "tie_policy": (
                "stable sort; condition-aware abs_effect then canonical pair; "
                "multi-sample raw p_value, abs_effect, then canonical pair"
            ),
            "expected_table_encoding": (
                "full fixed unordered-pair universe with boolean is_expected for "
                "every scenario-condition-fraction"
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scipy": _version("scipy"),
        },
        "diagnostics": _diagnostics(tables),
        "outputs": {key: {"filename": value} for key, value in filenames.items()},
    }
    if dry_run:
        return payload
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    output_paths = {key: destination / value for key, value in filenames.items()}
    conflicts = [
        path.name for path in (*output_paths.values(), manifest_path) if path.exists()
    ]
    if conflicts:
        raise FileExistsError(f"refusing to overwrite existing outputs: {conflicts}")
    for key, table in (
        ("sample_correlations", tables.sample_correlations),
        ("pair_rankings", tables.pair_rankings),
        ("expected_sets", tables.expected_sets),
    ):
        path = output_paths[key]
        _write_tsv_atomic(table, path)
        payload["outputs"][key].update(
            {
                "rows": len(table),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json_atomic(payload, manifest_path)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    self_group = parser.add_mutually_exclusive_group(required=True)
    self_group.add_argument("--include-self", action="store_true")
    self_group.add_argument("--exclude-self", action="store_true")
    parser.add_argument(
        "--multi-sample-unit",
        choices=("subject_id", "sample_id"),
        default="sample_id",
    )
    parser.add_argument(
        "--top-count-rule", choices=("floor", "ceil"), default="floor"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = prepare_ms_spatial_des_truth(
        args.input_root,
        args.output_root,
        include_self=bool(args.include_self),
        multi_sample_unit=args.multi_sample_unit,
        top_count_rule=args.top_count_rule,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "CELL_TYPES",
    "CHRONIC_ACTIVE",
    "CONDITIONS",
    "CONTROL",
    "DATASET_ID",
    "MS_SPATIAL_SAMPLES",
    "TOP_FRACTIONS",
    "MSSpatialSample",
    "MSSpatialTruthTables",
    "build_ms_spatial_des_truth",
    "compute_unordered_pearson_correlations",
    "prepare_ms_spatial_des_truth",
]
