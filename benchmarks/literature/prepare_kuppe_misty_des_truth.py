"""Build Kuppe spatial DES expected sets from recomputed MISTy outputs.

This postprocessor consumes the checksum-bound combined artifacts written by
``run_kuppe_spatial_misty.R``.  The values are a protocol-level recomputation
from public CELLxGENE files; they are not the authors' unpublished importance
table.  MISTy target models are filtered at ``multi.R2 >= 10`` before any
directional or view aggregation, matching Kuppe's ``summarize_interactions``
script.  Figure 3 defaults to sample/library units, excludes self-pairs through
the caller's explicit switch, and uses floor-sized expected sets. Subject units
and ceil-sized sets remain explicit sensitivity options.
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
from itertools import combinations, combinations_with_replacement, product
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DATASET_ID = "Kuppe_MI_spatial_CTRL_vs_IZ"
RUN_SCHEMA = "crychic-kuppe-spatial-misty-run-v1"
OUTPUT_SCHEMA = "crychic-kuppe-misty-des-truth-v1"
CTRL = "CTRL"
IZ = "IZ"
CONDITIONS = (CTRL, IZ)
MISTYR_VERSION = "1.3.5"
MISTYR_TAG_COMMIT = "19248ea7e02803063d1e1112a8af6c3f06c59e03"
RECOMPUTATION_MARKER = "not the authors' published MISTy importance table"
MULTI_R2_THRESHOLD = 10.0
TOP_FRACTIONS = (0.1, 0.2, 0.3, 0.4)
TopCountRule = Literal["floor", "ceil"]
ONTOLOGY_ALIASES = {"Cycling.cells": "Cycling cells"}
RAW_CELL_TYPES = (
    "Adipocyte",
    "Cardiomyocyte",
    "Endothelial",
    "Fibroblast",
    "Lymphoid",
    "Mast",
    "Myeloid",
    "Neuronal",
    "Pericyte",
    "Cycling.cells",
    "vSMCs",
)
VARIANT_VIEWS: dict[str, tuple[str, ...]] = {
    "spatial_neighbor_max": ("juxta_5", "para_15"),
    "juxta_only": ("juxta_5",),
    "para_only": ("para_15",),
    "all_view_max": ("intra", "juxta_5", "para_15"),
}
PRIMARY_VARIANT = "spatial_neighbor_max"


@dataclass(frozen=True, slots=True)
class KuppeSlide:
    """Frozen slide, biological-subject, and condition assignment."""

    sample_id: str
    subject_id: str
    condition: str


FROZEN_SLIDES = (
    KuppeSlide("control_P1", "P1", CTRL),
    KuppeSlide("control_P17", "P17", CTRL),
    KuppeSlide("control_P7", "P7", CTRL),
    KuppeSlide("control_P8", "P8", CTRL),
    KuppeSlide("GT_IZ_P13", "P13", IZ),
    KuppeSlide("GT_IZ_P15", "P15", IZ),
    KuppeSlide("GT_IZ_P9", "P9", IZ),
    KuppeSlide("GT_IZ_P9_rep2", "P9", IZ),
    KuppeSlide("IZ_BZ_P2", "P2", IZ),
    KuppeSlide("IZ_P10", "P10", IZ),
    KuppeSlide("IZ_P15", "P15", IZ),
    KuppeSlide("IZ_P16", "P16", IZ),
    KuppeSlide("IZ_P3", "P3", IZ),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class KuppeMistyDESTables:
    """Compact sample strengths, rankings, and expected memberships."""

    sample_pair_strengths: pd.DataFrame
    pair_rankings: pd.DataFrame
    expected_sets: pd.DataFrame

    def __post_init__(self) -> None:
        for field in ("sample_pair_strengths", "pair_rankings", "expected_sets"):
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


def _sample_design(slides: Sequence[KuppeSlide] | None = None) -> pd.DataFrame:
    if slides is None:
        slides = FROZEN_SLIDES
    return pd.DataFrame.from_records(
        [
            {
                "sample_id": slide.sample_id,
                "subject_id": slide.subject_id,
                "condition": slide.condition,
            }
            for slide in slides
        ]
    )


def _validate_design(sample_design: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "subject_id", "condition"}
    missing = required.difference(sample_design.columns)
    if missing:
        raise ValueError(f"sample_design is missing columns: {sorted(missing)}")
    result = sample_design.loc[:, ["sample_id", "subject_id", "condition"]].copy()
    if result.empty or result.isna().any().any():
        raise ValueError("sample_design identifiers must be complete")
    for column in required:
        result[column] = result[column].astype(str)
        if (
            result[column].eq("").any()
            or result[column].str.strip().ne(result[column]).any()
        ):
            raise ValueError(f"sample_design {column} must be canonical")
    if result["sample_id"].duplicated().any():
        raise ValueError("sample_design sample_id values must be unique")
    if set(result["condition"]) != set(CONDITIONS):
        raise ValueError(f"sample_design must contain conditions {list(CONDITIONS)}")
    cross_condition = result.groupby("subject_id", sort=False)["condition"].nunique()
    if (cross_condition > 1).any():
        raise ValueError("a subject_id cannot occur in both conditions")
    return result.sort_values("sample_id", kind="stable", ignore_index=True)


def _canonical_identifiers(
    table: pd.DataFrame, columns: Sequence[str], label: str
) -> None:
    for column in columns:
        if table[column].isna().any():
            raise ValueError(f"{label} {column} must not be missing")
        values = table[column].astype(str)
        if values.eq("").any() or values.str.strip().ne(values).any():
            raise ValueError(f"{label} {column} must contain canonical strings")


def _validate_analysis_tables(
    importances: pd.DataFrame,
    performance: pd.DataFrame,
    sample_design: pd.DataFrame,
    *,
    expected_raw_cell_types: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    design = _validate_design(sample_design)
    importance_required = {
        "dataset",
        "sample_id",
        "condition",
        "view",
        "Predictor",
        "Target",
        "Importance",
        "recomputation_status",
    }
    performance_required = {
        "dataset",
        "sample_id",
        "condition",
        "target",
        "measure",
        "value",
        "recomputation_status",
    }
    importance_missing = importance_required.difference(importances.columns)
    performance_missing = performance_required.difference(performance.columns)
    if importance_missing:
        raise ValueError(
            "combined importance table is missing columns: "
            f"{sorted(importance_missing)}"
        )
    if performance_missing:
        raise ValueError(
            "combined performance table is missing columns: "
            f"{sorted(performance_missing)}"
        )
    if importances.empty or performance.empty:
        raise ValueError("combined MISTy tables must not be empty")

    importance = importances.loc[:, sorted(importance_required)].copy(deep=True)
    perf = performance.loc[:, sorted(performance_required)].copy(deep=True)
    _canonical_identifiers(
        importance,
        (
            "dataset",
            "sample_id",
            "condition",
            "view",
            "Predictor",
            "Target",
            "recomputation_status",
        ),
        "importance",
    )
    _canonical_identifiers(
        perf,
        (
            "dataset",
            "sample_id",
            "condition",
            "target",
            "measure",
            "recomputation_status",
        ),
        "performance",
    )
    for table, label in ((importance, "importance"), (perf, "performance")):
        markers = set(table["recomputation_status"].astype(str))
        if len(markers) != 1 or RECOMPUTATION_MARKER not in next(iter(markers)):
            raise ValueError(
                f"combined {label} table does not declare public-data recomputation"
            )
    for table in (importance, perf):
        if set(table["dataset"].astype(str)) != {DATASET_ID}:
            raise ValueError(f"combined MISTy table dataset must be {DATASET_ID!r}")
        observed_samples = set(table["sample_id"].astype(str))
        expected_samples = set(design["sample_id"])
        if observed_samples != expected_samples:
            raise ValueError(
                "combined MISTy table sample roster mismatch: "
                f"missing={sorted(expected_samples - observed_samples)}, "
                f"unexpected={sorted(observed_samples - expected_samples)}"
            )
        condition_map = design.set_index("sample_id")["condition"]
        expected_conditions = table["sample_id"].astype(str).map(condition_map)
        if not table["condition"].astype(str).eq(expected_conditions).all():
            raise ValueError("combined MISTy sample/condition assignments disagree")

    valid_views = {view for views in VARIANT_VIEWS.values() for view in views}
    observed_views = set(importance["view"].astype(str))
    if observed_views != valid_views:
        raise ValueError(
            f"importance views {sorted(observed_views)} != {sorted(valid_views)}"
        )
    importance_key = ["sample_id", "view", "Predictor", "Target"]
    if importance.duplicated(importance_key).any():
        raise ValueError("combined importance table contains duplicate identity rows")
    supplied_importance = importance["Importance"].notna()
    numeric_importance = pd.to_numeric(importance["Importance"], errors="coerce")
    if (supplied_importance & numeric_importance.isna()).any() or np.isinf(
        numeric_importance.fillna(0.0)
    ).any():
        raise ValueError("Importance must contain finite numeric values or missing")
    importance["Importance"] = numeric_importance.astype(float)

    performance_key = ["sample_id", "target", "measure"]
    if perf.duplicated(performance_key).any():
        raise ValueError("combined performance table contains duplicate identity rows")
    multi_r2 = perf.loc[perf["measure"].eq("multi.R2")].copy()
    if multi_r2.empty:
        raise ValueError("combined performance table contains no multi.R2 rows")
    supplied_r2 = multi_r2["value"].notna()
    numeric_r2 = pd.to_numeric(multi_r2["value"], errors="coerce")
    if not supplied_r2.all() or numeric_r2.isna().any() or np.isinf(numeric_r2).any():
        raise ValueError("multi.R2 values must be finite numeric values")
    perf.loc[multi_r2.index, "value"] = numeric_r2.astype(float)

    predictors = set(importance["Predictor"].astype(str))
    targets = set(importance["Target"].astype(str))
    performance_targets = set(multi_r2["target"].astype(str))
    if predictors != targets or targets != performance_targets:
        raise ValueError(
            "importance Predictor/Target and performance target panels disagree"
        )
    raw_cell_types = tuple(sorted(targets))
    if expected_raw_cell_types is not None and set(raw_cell_types) != set(
        expected_raw_cell_types
    ):
        raise ValueError(
            "MISTy feature panel disagrees with the frozen Kuppe panel: "
            f"observed={list(raw_cell_types)}"
        )

    expected_importance_keys = set(
        product(
            design["sample_id"],
            sorted(valid_views),
            raw_cell_types,
            raw_cell_types,
        )
    )
    observed_importance_keys = set(
        importance.loc[:, importance_key].itertuples(index=False, name=None)
    )
    if observed_importance_keys != expected_importance_keys:
        raise ValueError(
            "combined importance table is not a complete sample/view panel"
        )
    expected_r2_keys = set(product(design["sample_id"], raw_cell_types))
    observed_r2_keys = set(
        multi_r2.loc[:, ["sample_id", "target"]].itertuples(index=False, name=None)
    )
    if observed_r2_keys != expected_r2_keys:
        raise ValueError("multi.R2 table is not a complete sample-target panel")
    return importance, perf, raw_cell_types


def _aligned_cell_types(raw_cell_types: Sequence[str]) -> tuple[str, ...]:
    aligned = tuple(ONTOLOGY_ALIASES.get(value, value) for value in raw_cell_types)
    if len(set(aligned)) != len(aligned):
        raise ValueError("ontology aliases collapse distinct MISTy cell types")
    return tuple(sorted(aligned))


def _pair_universe(
    cell_types: Sequence[str], *, include_self: bool
) -> tuple[tuple[str, str], ...]:
    if not isinstance(include_self, bool):
        raise ValueError("include_self must be boolean")
    iterator = (
        combinations_with_replacement(cell_types, 2)
        if include_self
        else combinations(cell_types, 2)
    )
    return tuple(iterator)


def _sample_pair_strengths(
    importances: pd.DataFrame,
    performance: pd.DataFrame,
    sample_design: pd.DataFrame,
    raw_cell_types: Sequence[str],
    *,
    include_self: bool,
    multi_r2_threshold: float,
) -> pd.DataFrame:
    r2 = performance.loc[
        performance["measure"].eq("multi.R2"), ["sample_id", "target", "value"]
    ].rename(columns={"target": "Target", "value": "multi_r2"})
    merged = importances.merge(
        r2, on=["sample_id", "Target"], how="left", validate="many_to_one"
    )
    if merged["multi_r2"].isna().any():
        raise RuntimeError("validated importance rows lost their target multi.R2")
    merged["target_passes_multi_r2"] = merged["multi_r2"].ge(multi_r2_threshold)
    merged["Predictor"] = merged["Predictor"].map(
        lambda value: ONTOLOGY_ALIASES.get(str(value), str(value))
    )
    merged["Target"] = merged["Target"].map(
        lambda value: ONTOLOGY_ALIASES.get(str(value), str(value))
    )
    merged["sender"] = np.minimum(merged["Predictor"], merged["Target"])
    merged["receiver"] = np.maximum(merged["Predictor"], merged["Target"])
    aligned_types = _aligned_cell_types(raw_cell_types)
    pairs = _pair_universe(aligned_types, include_self=include_self)
    design = _validate_design(sample_design)

    records: list[dict[str, object]] = []
    for variant, views in VARIANT_VIEWS.items():
        view_rows = merged.loc[merged["view"].isin(views)]
        for sample in design.itertuples(index=False):
            sample_rows = view_rows.loc[view_rows["sample_id"].eq(sample.sample_id)]
            for sender, receiver in pairs:
                pair_rows = sample_rows.loc[
                    sample_rows["sender"].eq(sender)
                    & sample_rows["receiver"].eq(receiver)
                ]
                eligible = pair_rows.loc[pair_rows["target_passes_multi_r2"]]
                finite = eligible.loc[np.isfinite(eligible["Importance"])]
                eligible_target_directions = int(eligible["Target"].nunique())
                if finite.empty:
                    importance = math.nan
                    status = "not_estimable"
                    reason = (
                        "no_target_passed_multi_r2_threshold"
                        if eligible.empty
                        else "no_finite_importance_after_target_multi_r2_filter"
                    )
                    candidate_views = "[]"
                else:
                    importance = float(finite["Importance"].max())
                    status = "observed"
                    reason = ""
                    candidate_views = json.dumps(
                        sorted(set(finite["view"].astype(str))), separators=(",", ":")
                    )
                records.append(
                    {
                        "dataset": DATASET_ID,
                        "variant": variant,
                        "sample_id": sample.sample_id,
                        "subject_id": sample.subject_id,
                        "condition": sample.condition,
                        "sender": sender,
                        "receiver": receiver,
                        "spatial_importance": importance,
                        "finite_candidate_rows": len(finite),
                        "eligible_target_directions": eligible_target_directions,
                        "candidate_views_json": candidate_views,
                        "status": status,
                        "reason_code": reason,
                    }
                )
    return pd.DataFrame.from_records(records).sort_values(
        ["variant", "condition", "sample_id", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )


def _rank_variant_scenario(
    strengths: pd.DataFrame,
    *,
    variant: str,
    scenario: Literal["condition_aware", "multi_sample"],
    multi_sample_unit: Literal["subject_id", "sample_id"],
) -> pd.DataFrame:
    variant_rows = strengths.loc[strengths["variant"].eq(variant)]
    pairs = variant_rows.loc[:, ["sender", "receiver"]].drop_duplicates()
    unit_column = "sample_id" if scenario == "condition_aware" else multi_sample_unit
    observed = variant_rows.loc[variant_rows["status"].eq("observed")]
    units = cast(
        pd.DataFrame,
        observed.groupby(
            [unit_column, "condition", "sender", "receiver"],
            sort=True,
            observed=True,
            as_index=False,
        )["spatial_importance"]
        .mean(),
    )
    units = units.sort_values(
        ["condition", unit_column, "sender", "receiver"], kind="stable"
    )
    records: list[dict[str, object]] = []
    for sender, receiver in pairs.sort_values(
        ["sender", "receiver"], kind="stable"
    ).itertuples(index=False, name=None):
        pair = units.loc[units["sender"].eq(sender) & units["receiver"].eq(receiver)]
        ctrl_values = pair.loc[
            pair["condition"].eq(CTRL), "spatial_importance"
        ].to_numpy(dtype=float)
        iz_values = pair.loc[pair["condition"].eq(IZ), "spatial_importance"].to_numpy(
            dtype=float
        )
        ctrl_mean = float(np.mean(ctrl_values)) if ctrl_values.size else math.nan
        iz_mean = float(np.mean(iz_values)) if iz_values.size else math.nan
        effect = iz_mean - ctrl_mean
        if scenario == "condition_aware":
            statistic = p_value = math.nan
            p_semantics = "not_applicable_condition_mean_ranking"
            if ctrl_values.size and iz_values.size:
                status = "observed"
                reason = ""
            else:
                status = "not_estimable"
                reason = "condition_has_no_observed_sample_importances"
        else:
            p_semantics = (
                "raw_two_sided_mann_whitney_u_scipy_method_auto;"
                "no_multiple_testing_correction"
            )
            if ctrl_values.size >= 2 and iz_values.size >= 2:
                test = mannwhitneyu(
                    iz_values,
                    ctrl_values,
                    alternative="two-sided",
                    method="auto",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
                status = "observed"
                reason = ""
            else:
                statistic = p_value = math.nan
                status = "not_estimable"
                reason = "fewer_than_two_observed_units_in_a_condition"
        records.append(
            {
                "dataset": DATASET_ID,
                "variant": variant,
                "scenario": scenario,
                "sender": sender,
                "receiver": receiver,
                "mean_CTRL": ctrl_mean,
                "mean_IZ": iz_mean,
                "effect_IZ_minus_CTRL": effect,
                "abs_effect": abs(effect),
                "u_statistic": statistic,
                "p_value": p_value,
                "p_value_semantics": p_semantics,
                "n_CTRL_units": int(ctrl_values.size),
                "n_IZ_units": int(iz_values.size),
                "status": status,
                "reason_code": reason,
            }
        )
    result = pd.DataFrame.from_records(records)
    effect_values = result["effect_IZ_minus_CTRL"]
    result["direction_condition"] = np.select(
        [effect_values > 0, effect_values < 0], [IZ, CTRL], default="tied"
    )
    eligible = result["status"].eq("observed") & effect_values.ne(0)
    if scenario == "condition_aware":
        sort_columns = ["abs_effect", "sender", "receiver"]
        ascending = [False, True, True]
    else:
        eligible &= result["p_value"].notna()
        sort_columns = ["p_value", "abs_effect", "sender", "receiver"]
        ascending = [True, False, True, True]
    ranked = result.loc[eligible].sort_values(
        sort_columns, ascending=ascending, kind="stable"
    )
    rank_map = pd.Series(
        np.arange(1, len(ranked) + 1, dtype=np.int64), index=ranked.index
    )
    result["spatial_rank"] = rank_map.reindex(result.index).astype("Int64")
    result["ranking_eligible"] = eligible
    columns = [
        "dataset",
        "variant",
        "scenario",
        "sender",
        "receiver",
        "mean_CTRL",
        "mean_IZ",
        "effect_IZ_minus_CTRL",
        "abs_effect",
        "u_statistic",
        "p_value",
        "p_value_semantics",
        "n_CTRL_units",
        "n_IZ_units",
        "direction_condition",
        "spatial_rank",
        "ranking_eligible",
        "status",
        "reason_code",
    ]
    return result.loc[:, columns].sort_values(
        ["spatial_rank", "sender", "receiver"],
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
    for (variant, scenario), table in rankings.groupby(
        ["variant", "scenario"], sort=True, observed=True
    ):
        rankable_pairs = int(table["ranking_eligible"].sum())
        for condition in CONDITIONS:
            for fraction in fractions:
                top_count = (
                    (math.floor if top_count_rule == "floor" else math.ceil)(
                        fraction * rankable_pairs
                    )
                    if rankable_pairs
                    else 0
                )
                for row in table.itertuples(index=False):
                    selected = (
                        not pd.isna(row.spatial_rank)
                        and int(cast(float, row.spatial_rank)) <= top_count
                        and row.direction_condition == condition
                    )
                    records.append(
                        {
                            "dataset": DATASET_ID,
                            "variant": variant,
                            "scenario": scenario,
                            "condition": condition,
                            "top_fraction": fraction,
                            "sender": row.sender,
                            "receiver": row.receiver,
                            "is_expected": bool(selected),
                            "spatial_rank": row.spatial_rank,
                            "rankable_pairs": rankable_pairs,
                            "top_count": top_count,
                            "top_count_rule": top_count_rule,
                            "cell_pair_direction": "unordered_canonical",
                        }
                    )
    result = pd.DataFrame.from_records(records)
    result["spatial_rank"] = result["spatial_rank"].astype("Int64")
    return result.sort_values(
        [
            "variant",
            "scenario",
            "condition",
            "top_fraction",
            "sender",
            "receiver",
        ],
        kind="stable",
        ignore_index=True,
    )


def build_kuppe_misty_des_truth(
    importances: pd.DataFrame,
    performance: pd.DataFrame,
    sample_design: pd.DataFrame,
    *,
    include_self: bool,
    multi_sample_unit: Literal["subject_id", "sample_id"] = "sample_id",
    multi_r2_threshold: float = MULTI_R2_THRESHOLD,
    top_fractions: Sequence[float] = TOP_FRACTIONS,
    top_count_rule: TopCountRule = "floor",
    expected_raw_cell_types: Sequence[str] | None = None,
) -> KuppeMistyDESTables:
    """Build all Kuppe MISTy spatial variants and DES expected memberships."""

    if multi_sample_unit not in {"subject_id", "sample_id"}:
        raise ValueError("multi_sample_unit must be 'subject_id' or 'sample_id'")
    if isinstance(multi_r2_threshold, bool):
        raise ValueError("multi_r2_threshold must be a finite number")
    threshold = float(multi_r2_threshold)
    if not math.isfinite(threshold):
        raise ValueError("multi_r2_threshold must be a finite number")
    fractions = _canonical_fractions(top_fractions)
    design = _validate_design(sample_design)
    importance, perf, raw_cell_types = _validate_analysis_tables(
        importances,
        performance,
        design,
        expected_raw_cell_types=expected_raw_cell_types,
    )
    strengths = _sample_pair_strengths(
        importance,
        perf,
        design,
        raw_cell_types,
        include_self=include_self,
        multi_r2_threshold=threshold,
    )
    ranking_tables = [
        _rank_variant_scenario(
            strengths,
            variant=variant,
            scenario=scenario,
            multi_sample_unit=multi_sample_unit,
        )
        for variant in VARIANT_VIEWS
        for scenario in ("condition_aware", "multi_sample")
    ]
    rankings = pd.concat(ranking_tables, ignore_index=True).sort_values(
        ["variant", "scenario", "spatial_rank", "sender", "receiver"],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )
    expected = _expected_memberships(
        rankings, fractions, top_count_rule=top_count_rule
    )
    return KuppeMistyDESTables(
        sample_pair_strengths=strengths,
        pair_rankings=rankings,
        expected_sets=expected,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must contain an object: {path}")
    return value


def _validated_artifact(
    run_root: Path, record: object, *, label: str
) -> tuple[Path, int]:
    if not isinstance(record, dict):
        raise ValueError(f"run manifest output {label} must be an object")
    filename = record.get("filename")
    checksum = record.get("sha256")
    rows = record.get("rows")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError(f"run manifest output {label} filename must be a basename")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(value not in "0123456789abcdef" for value in checksum)
    ):
        raise ValueError(f"run manifest output {label} has invalid SHA256")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError(f"run manifest output {label} has invalid row count")
    path = run_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"run manifest output {label} does not exist: {path}")
    observed = _sha256_file(path)
    if observed != checksum:
        raise ValueError(
            f"run manifest output {label} SHA256 mismatch: {observed} != {checksum}"
        )
    return path, rows


def _load_validated_run(
    run_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = _read_json(run_manifest_path)
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError("unsupported Kuppe MISTy run manifest schema")
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("Kuppe MISTy run manifest dataset_id mismatch")
    if manifest.get("status") != "complete":
        raise ValueError("Kuppe MISTy run must be complete for all 13 slides")
    recomputation_status = manifest.get("recomputation_status")
    if (
        not isinstance(recomputation_status, str)
        or RECOMPUTATION_MARKER not in recomputation_status
    ):
        raise ValueError("run manifest does not declare public-data recomputation")
    software = manifest.get("software")
    if not isinstance(software, dict):
        raise ValueError("run manifest software record is missing")
    if software.get("mistyR") != MISTYR_VERSION:
        raise ValueError(f"run manifest must use mistyR {MISTYR_VERSION}")
    if software.get("mistyR_tag_commit") != MISTYR_TAG_COMMIT:
        raise ValueError("run manifest mistyR tag commit mismatch")

    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("run manifest samples must be a list")
    sample_records: list[dict[str, str]] = []
    for record in samples:
        if not isinstance(record, dict):
            raise ValueError("run manifest sample records must be objects")
        if record.get("status") != "complete":
            raise ValueError("every Kuppe MISTy sample must have complete status")
        sample_records.append(
            {
                "sample_id": str(record.get("sample_id")),
                "condition": str(record.get("condition")),
            }
        )
    observed = pd.DataFrame.from_records(sample_records)
    if observed.empty or observed["sample_id"].duplicated().any():
        raise ValueError("run manifest sample roster is empty or duplicated")
    frozen_design = _sample_design()
    expected_condition = frozen_design.set_index("sample_id")["condition"]
    if set(observed["sample_id"]) != set(frozen_design["sample_id"]):
        raise ValueError("run manifest must contain the frozen 13-slide roster")
    observed_conditions = observed["sample_id"].map(expected_condition)
    if not observed["condition"].eq(observed_conditions).all():
        raise ValueError("run manifest sample conditions disagree with frozen roster")
    if observed["condition"].value_counts().to_dict() != {IZ: 9, CTRL: 4}:
        raise ValueError("run manifest must contain exactly 4 CTRL and 9 IZ slides")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("run manifest combined outputs are missing")
    run_root = run_manifest_path.parent
    importance_path, expected_importance_rows = _validated_artifact(
        run_root, outputs.get("importances"), label="importances"
    )
    performance_path, expected_performance_rows = _validated_artifact(
        run_root, outputs.get("performance"), label="performance"
    )
    importances = pd.read_csv(importance_path, sep="\t", low_memory=False)
    performance = pd.read_csv(performance_path, sep="\t", low_memory=False)
    if len(importances) != expected_importance_rows:
        raise ValueError("combined importance row count disagrees with run manifest")
    if len(performance) != expected_performance_rows:
        raise ValueError("combined performance row count disagrees with run manifest")
    for table, label in ((importances, "importance"), (performance, "performance")):
        if "recomputation_status" not in table.columns:
            raise ValueError(f"combined {label} table lacks recomputation_status")
        if set(table["recomputation_status"].dropna().astype(str)) != {
            recomputation_status
        }:
            raise ValueError(
                f"combined {label} recomputation marker disagrees with manifest"
            )
    _validate_analysis_tables(
        importances,
        performance,
        frozen_design,
        expected_raw_cell_types=RAW_CELL_TYPES,
    )
    return importances, performance, frozen_design, manifest


def _diagnostics(tables: KuppeMistyDESTables) -> dict[str, Any]:
    strengths = tables.sample_pair_strengths
    rankings = tables.pair_rankings
    variant_strengths = {
        str(variant): {
            "sample_pair_rows": len(group),
            "observed_rows": int(group["status"].eq("observed").sum()),
            "not_estimable_rows": int(group["status"].ne("observed").sum()),
        }
        for variant, group in strengths.groupby("variant", sort=True, observed=True)
    }
    ranking_records: dict[str, object] = {}
    for (variant, scenario), group in rankings.groupby(
        ["variant", "scenario"], sort=True, observed=True
    ):
        primary = "abs_effect" if scenario == "condition_aware" else "p_value"
        eligible = group.loc[group["ranking_eligible"]]
        ranking_records[f"{variant}:{scenario}"] = {
            "pair_rows": len(group),
            "rankable_pairs": len(eligible),
            "not_estimable_pairs": int(group["status"].ne("observed").sum()),
            "zero_direction_pairs": int(group["direction_condition"].eq("tied").sum()),
            "rows_in_primary_rank_ties": int(
                eligible[primary].duplicated(keep=False).sum()
            ),
        }
    return {
        "variant_sample_strengths": variant_strengths,
        "pair_rankings": ranking_records,
        "expected_members_by_variant_scenario_condition_fraction": [
            {
                "variant": str(key[0]),
                "scenario": str(key[1]),
                "condition": str(key[2]),
                "top_fraction": float(cast(float, key[3])),
                "members": int(group["is_expected"].sum()),
                "universe": len(group),
            }
            for key, group in tables.expected_sets.groupby(
                ["variant", "scenario", "condition", "top_fraction"],
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


def prepare_kuppe_misty_des_truth(
    run_manifest: str | Path,
    output_root: str | Path,
    *,
    include_self: bool,
    multi_sample_unit: Literal["subject_id", "sample_id"] = "sample_id",
    top_count_rule: TopCountRule = "floor",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate a full MISTy run and write compact DES truth artifacts."""

    run_manifest_path = Path(run_manifest).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if not run_manifest_path.is_file():
        raise FileNotFoundError(
            f"Kuppe MISTy run manifest not found: {run_manifest_path}"
        )
    importances, performance, design, source_manifest = _load_validated_run(
        run_manifest_path
    )
    tables = build_kuppe_misty_des_truth(
        importances,
        performance,
        design,
        include_self=include_self,
        multi_sample_unit=multi_sample_unit,
        top_count_rule=top_count_rule,
        expected_raw_cell_types=RAW_CELL_TYPES,
    )
    filenames = {
        "sample_pair_strengths": "kuppe_misty_sample_pair_strengths.tsv",
        "pair_rankings": "kuppe_misty_pair_rankings.tsv",
        "expected_sets": "kuppe_misty_expected_sets.tsv",
    }
    payload: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "dry_run" if dry_run else "complete",
        "dataset_id": DATASET_ID,
        "source": {
            "run_manifest_filename": run_manifest_path.name,
            "run_manifest_sha256": _sha256_file(run_manifest_path),
            "run_schema_version": source_manifest["schema_version"],
            "run_recomputation_status": source_manifest["recomputation_status"],
            "author_importance_table": False,
            "scope": (
                "protocol-level MISTy 1.3.5 recomputation from public CELLxGENE "
                "inputs; not the authors' published MISTy importance table"
            ),
            "combined_outputs": source_manifest["outputs"],
        },
        "design": {
            "conditions": list(CONDITIONS),
            "slides_per_condition": {CTRL: 4, IZ: 9},
            "subjects_per_condition": {CTRL: 4, IZ: 7},
            "multi_sample_unit": multi_sample_unit,
            "repeated_sections": {
                "P9": ["GT_IZ_P9", "GT_IZ_P9_rep2"],
                "P15": ["GT_IZ_P15", "IZ_P15"],
            },
            "repeated_section_aggregation": (
                "arithmetic mean of observed slide importance within subject, "
                "condition, variant, and unordered pair"
                if multi_sample_unit == "subject_id"
                else "none; Figure 3 sample/library unit"
            ),
        },
        "protocol": {
            "primary_variant": PRIMARY_VARIANT,
            "variants": {key: list(value) for key, value in VARIANT_VIEWS.items()},
            "variant_aggregation": (
                "maximum finite Importance across selected views and both MISTy "
                "Predictor-to-Target directions after target filtering"
            ),
            "multi_r2_filter": {
                "measure": "multi.R2",
                "threshold": MULTI_R2_THRESHOLD,
                "comparison": ">=",
                "grain": "sample_id x Target",
                "order": "before direction and view aggregation",
                "source_compatibility": (
                    "Kuppe summarize_interactions.R best_performers R2 >= 10"
                ),
            },
            "ontology_aliases": ONTOLOGY_ALIASES,
            "ontology_alias_scope": "cross-modality label alignment only",
            "cell_pair_direction": "unordered_canonical_sender_le_receiver",
            "include_self_pairs": include_self,
            "missing_policy": (
                "no finite post-filter importance is not_estimable; never zero-imputed"
            ),
            "condition_aware": (
                "arithmetic slide mean per condition; rank absolute IZ-minus-CTRL "
                "difference"
            ),
            "multi_sample": (
                "two-sided scipy.stats.mannwhitneyu(method='auto') on declared "
                "analysis units; rank raw p_value ascending then abs effect descending"
            ),
            "p_value_adjustment": "none",
            "q_values_generated": False,
            "top_fractions": list(TOP_FRACTIONS),
            "top_count_rule": (
                f"{top_count_rule}(top_fraction * rankable_pairs)"
            ),
            "direction_rule": (
                "positive IZ-minus-CTRL effect => IZ; negative => CTRL; exact zero "
                "=> tied and never expected"
            ),
            "tie_policy": (
                "stable sort; condition-aware abs_effect then canonical pair; "
                "multi-sample raw p_value, abs_effect, then canonical pair"
            ),
            "expected_table_encoding": (
                "full fixed unordered-pair universe with boolean is_expected for "
                "every variant-scenario-condition-fraction"
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scipy": _version("scipy"),
            "source_mistyR": MISTYR_VERSION,
            "source_mistyR_tag_commit": MISTYR_TAG_COMMIT,
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
        ("sample_pair_strengths", tables.sample_pair_strengths),
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
    parser.add_argument("run_manifest", type=Path)
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
    payload = prepare_kuppe_misty_des_truth(
        args.run_manifest,
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
    "CONDITIONS",
    "CTRL",
    "DATASET_ID",
    "FROZEN_SLIDES",
    "IZ",
    "MULTI_R2_THRESHOLD",
    "ONTOLOGY_ALIASES",
    "PRIMARY_VARIANT",
    "RAW_CELL_TYPES",
    "TOP_FRACTIONS",
    "VARIANT_VIEWS",
    "KuppeMistyDESTables",
    "KuppeSlide",
    "build_kuppe_misty_des_truth",
    "prepare_kuppe_misty_des_truth",
]
